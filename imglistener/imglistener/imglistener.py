#!/usr/bin/env python3
import math
import random
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import sensor_msgs_py.point_cloud2 as pcl2
import std_msgs.msg
from cv_bridge import CvBridge

from geometry_msgs.msg import Quaternion, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry , Path
from sensor_msgs.msg import Image, Imu, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from tf_transformations import quaternion_from_euler, euler_from_quaternion, quaternion_matrix

# Import our custom FastSLAM classes
from .particlefilter import PF, Landmark 

# --- Constants ---
MIN_KEYPOINTS = 5
MIN_INLIERS = 8

# Camera Intrinsics (Kinect)
CX = 318.525
CY = 241.181
F = 526.61


class FastSlamNode(Node):
    """
    Main ROS 2 Node for the FastSLAM project.
    Ingests RGB-D, Odometry, and IMU data to estimate robot pose and build a feature map.
    """
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()

        # Frame IDs
        self.odom_frame = 'orb_odom'
        self.base_frame = 'base_link'
        self.camera_frame = 'kinect_depth'  

        # --- Subscribers ---
        self.subscription = self.create_subscription(Image, '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/serf01/nav_rgbd_1/depth/image_raw', self.depth_subscription, 10)
        self.wheel_sub = self.create_subscription(Odometry, '/serf01/odometry/wheel', self.wheel_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/serf01/odometry/imu', self.imu_callback, 10)

        # --- Publishers ---
        self.current_pc_pub = self.create_publisher(PointCloud2, '/global_cloud', 10)
        self.odom_pub = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)
        self.path_pub = self.create_publisher(Path, '/orb_path', 10)
        self.pf_particles_marker_pub = self.create_publisher(MarkerArray, '/pf_particles_marker', 10)
        self.cov_pub = self.create_publisher(MarkerArray, '/landmark_covariances', 10)

        # --- TF Setup ---
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.kinect_to_base_matrix = None
        self.base_to_kinect_matrix = None
        self.static_tf_ready = False

        # --- State Variables ---
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame
        self.depth_image = None
        self.current_time = None

        # Odometry accumulators (integrated between camera frames)
        self.prev_wheel_x = None
        self.prev_wheel_y = None
        self.prev_imu_yaw = None
        self.odom_forward_accum = 0.0
        self.odom_lateral_accum = 0.0
        self.imu_dtheta_accum = 0.0
        
        # --- Particle Filter Initialization ---
        self.MAX_MISSED_FRAMES = 3
        self.N_PARTICLES = 100
        self.pf = PF(num_particles=self.N_PARTICLES, x=0.0, y=0.0, yaw=0.0)
        
        self.get_logger().info("FastSLAM Node Initialized with {} particles.".format(self.N_PARTICLES))

    # ==========================================
    # Sensor Callbacks & Data Accumulation
    # ==========================================
    def try_get_camera_tf(self):
        """Fetches the static transform between base_link and the camera once."""
        if self.static_tf_ready:
            return True
        try:
            if not self.tf_buffer.can_transform(self.base_frame, self.camera_frame, rclpy.time.Time(), timeout=Duration(seconds=0.2)):
                return False
            t = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time(), timeout=Duration(seconds=0.2))
            quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            self.kinect_to_base_matrix = quaternion_matrix(quat)
            self.kinect_to_base_matrix[0, 3] = t.transform.translation.x
            self.kinect_to_base_matrix[1, 3] = t.transform.translation.y
            self.kinect_to_base_matrix[2, 3] = t.transform.translation.z
            self.base_to_kinect_matrix = np.linalg.inv(self.kinect_to_base_matrix)
            self.static_tf_ready = True
            return True
        except Exception:
            return False

    def depth_subscription(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def imu_callback(self, msg: Imu):
        """Accumulates yaw changes from the IMU between camera frames."""
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        _, _, yaw = euler_from_quaternion(quat)
        yaw = wrap_angle(yaw)
        
        if self.prev_imu_yaw is not None:
            dyaw = wrap_angle(yaw - self.prev_imu_yaw)
            if abs(dyaw) < 0.5: # Reject massive jumps
                self.imu_dtheta_accum += dyaw
        self.prev_imu_yaw = yaw
    
    def wheel_callback(self, msg: Odometry):
        """Accumulates local forward/lateral translation from wheel encoders."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _, _, odom_yaw = euler_from_quaternion(quat)
        
        if self.prev_wheel_x is not None:
            dx = x - self.prev_wheel_x
            dy = y - self.prev_wheel_y
            
            # Rotate global dx/dy back into the robot's local frame
            c, s = math.cos(odom_yaw), math.sin(odom_yaw)
            self.odom_forward_accum += c * dx + s * dy
            self.odom_lateral_accum += -s * dx + c * dy
            
        self.prev_wheel_x = x
        self.prev_wheel_y = y

    def _compute_feature_covariance(self, cam_x, cam_z, global_yaw):
        """Calculates measurement uncertainty based on depth distance."""
        d = math.sqrt(cam_x**2 + cam_z**2)
        if d < 0.1:
            return np.diag([1e-5, 1e-5])
        
        # 1. Errors in METERS
        a = 0.001477
        b = 0.002294 
        sigma_d = a + b * (d - 0.4)**2
        

        sigma_a = (d * math.sin(math.radians(0.1))) / 3.0

        # 2. Base Cartesian Covariance 
        S_squared = np.diag([sigma_a**2, sigma_d**2])

        # 3. Calculate alpha 
        alpha = math.atan2(cam_x, cam_z)

        # 4. Total rotation = Robot Heading + Camera Ray Angle
        total_angle = global_yaw + alpha
        c, s = math.cos(total_angle), math.sin(total_angle)
        
        R = np.array([
            [c, -s],
            [s,  c]
        ])

        # 5. Rotate the matrix to match the global frame
        return R @ S_squared @ R.T


    # ==========================================
    # Main FastSLAM Pipeline
    # ==========================================
    def listener_callback(self, msg):
        self.current_time = msg.header.stamp
        if not self.try_get_camera_tf() or self.depth_image is None: 
            return

        # 1. Fetch Odometry Control Input (u_t)
        u_forward = self.odom_forward_accum
        u_lateral = self.odom_lateral_accum
        u_dtheta = self.imu_dtheta_accum
        
        # Reset accumulators for the next frame
        self.odom_forward_accum, self.odom_lateral_accum, self.imu_dtheta_accum = 0.0, 0.0, 0.0

        # 2. Extract Measurements (z_t)
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        kp, des, points = feature_detector(frame, self.depth_image)

        if not kp or len(kp) < MIN_KEYPOINTS: 
            return

        # 3. Process Each Particle Hypothesis Individually
        for particle in self.pf.particles:
            P_in, Q_in = [], []
            innovations, S_mats = [], []
            matched_current_indices = set()
            matched_map_indices = set()

            # --- A. Data Association ---
            # Extract basic lists for the matcher
            map_pts = [lm.pos_3d for lm in particle.map]
            map_des = [lm.descriptor for lm in particle.map]
            
            vis_indices = get_visible_landmarks(map_pts, particle.x, particle.y, particle.yaw, self.base_to_kinect_matrix)
            
            if len(vis_indices) > 0 and des is not None and len(map_des) > 0:
                vis_des = np.array([map_des[idx] for idx in vis_indices])
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(vis_des, des)

                cos_yaw, sin_yaw = math.cos(-particle.yaw), math.sin(-particle.yaw)
                
                # --- B. Landmark Kalman Filter Updates ---
                for m in matches:
                    if m.distance >= 50: continue # Reject poor descriptor matches
                    
                    map_idx = vis_indices[m.queryIdx]
                    pt_w = particle.map[map_idx].pos_3d

                    # Transform map point to camera frame
                    dx, dy = pt_w[0] - particle.x, pt_w[1] - particle.y
                    base_x = dx * cos_yaw - dy * sin_yaw
                    base_y = dx * sin_yaw + dy * cos_yaw
                    pt_cam = self.base_to_kinect_matrix @ np.array([base_x, base_y, pt_w[2], 1.0])
                    
                    if pt_cam[2] <= 0.1: continue

                    # Save corresponding 3D points for Kabsch (Visual Odometry)
                    P_in.append([pt_cam[0], pt_cam[1], pt_cam[2]])
                    Q_in.append(points[m.trainIdx])
                    
                    matched_current_indices.add(m.trainIdx)
                    matched_map_indices.add(map_idx)

                    # Measurement Update via Linear Kalman Filter
                    z_meas = np.array([points[m.trainIdx][0], points[m.trainIdx][2]]) # Measured X, Z
                    R = self._compute_feature_covariance(pt_cam[0], pt_cam[2], particle.yaw)
                    Q_kf = np.diag([0.001, 0.001]) 
                    
                    v, S = particle.update_landmark(map_idx, z_meas, R, Q_kf)
                    if v is not None and S is not None:
                        innovations.append(v)
                        S_mats.append(S)

            # Update metrics for map pruning
            for map_idx in vis_indices:
                lm = particle.map[map_idx]
                lm.times_expected += 1
                if map_idx in matched_map_indices:
                    lm.times_seen += 1
                    lm.missed_frames = 0
                else:
                    lm.missed_frames += 1

            # --- C. Predict Particle Motion ---
            d_fwd, d_lat, d_yaw = u_forward, u_lateral, u_dtheta
            inlier_count = 0
            
            # Attempt Visual Odometry if we have enough matches
            if len(P_in) >= MIN_KEYPOINTS:
                R_2d, t_2d, _, _, inlier_count, _ = ransac_kabsch(np.array(P_in), np.array(Q_in))
                if R_2d is not None and inlier_count >= MIN_INLIERS:
                    d_fwd, d_lat = t_2d[1], -t_2d[0]
                    d_yaw = math.atan2(R_2d[1, 0], R_2d[0, 0])
            
            # Dynamic motion noise based on movement magnitude
            sigmas = {
                'forward': 0.05 * abs(d_fwd) + 0.01,
                'lateral': 0.05 * abs(d_lat) + 0.01,
                'yaw': 0.08 * abs(d_yaw) + 0.01
            }
            
            particle.predict_motion(d_fwd, d_lat, d_yaw, sigmas)

            # --- D. Update Particle Weight ---
            log_likelihood = 0.0
            for v, S in zip(innovations, S_mats):
                try:
                    S_inv = np.linalg.inv(S)
                    det_S = max(np.linalg.det(S), 1e-12)
                    mahalanobis = v.T @ S_inv @ v
                    log_prob = -0.5 * (2.0 * math.log(2 * math.pi) + math.log(det_S) + mahalanobis)
                    log_likelihood += log_prob
                except np.linalg.LinAlgError:
                    pass
            
            particle.weight *= (math.exp(log_likelihood) + 1e-300)

            # --- E. Map Maintenance ---
            c_y, s_y = math.cos(particle.yaw), math.sin(particle.yaw)
            for pt_idx, pt in enumerate(points):
                # Add unassociated features as new landmarks
                if pt_idx not in matched_current_indices and len(particle.map) < 500:
                    pt_base = self.kinect_to_base_matrix @ np.array([pt[0], pt[1], pt[2], 1.0])
                    x_g = pt_base[0] * c_y - pt_base[1] * s_y + particle.x
                    y_g = pt_base[0] * s_y + pt_base[1] * c_y + particle.y
                    
                    new_cov = self._compute_feature_covariance(pt[0], pt[2], particle.yaw)
                    new_landmark = Landmark(x_g, y_g, pt_base[2], des[pt_idx], new_cov)
                    particle.map.append(new_landmark)
            
            particle.prune_map(self.MAX_MISSED_FRAMES)

        # 4. Finalize Frame
        self.pf.normalize_weights()
        best_particle = self.pf.get_best_particle()

        # 5. Output Data to ROS/Rviz
        publish_odometry(self, self.current_time, best_particle.x, best_particle.y, best_particle.yaw)
        if len(best_particle.map) > 0:
            best_map_pts = np.array([lm.pos_3d for lm in best_particle.map])
            
            # Publish the 3D dots
            pointcloud(best_map_pts, self.current_pc_pub, self.current_time, self.odom_frame)
            
            # NEW: Publish the covariance ellipses around the dots!
            publish_landmark_covariances(self, self.current_time, best_particle.map)
            
        publish_pf_particles_markers(self, self.current_time)


# ==========================================
# Helper Functions
# ==========================================
def wrap_angle(angle): 
    return math.atan2(math.sin(angle), math.cos(angle))

def get_visible_landmarks(map_points_3d, global_x, global_y, global_yaw, base_to_kinect_matrix):
    """Filters the map to return only landmarks currently inside the camera's FOV."""
    vis_indices = []
    cos_yaw, sin_yaw = math.cos(-global_yaw), math.sin(-global_yaw)
    
    for i, pt in enumerate(map_points_3d):
        dx, dy = pt[0] - global_x, pt[1] - global_y
        if (dx*dx + dy*dy) > 16.0: # Cull points farther than 4 meters away
            continue
            
        pt_cam = base_to_kinect_matrix @ np.array([dx * cos_yaw - dy * sin_yaw, dx * sin_yaw + dy * cos_yaw, pt[2], 1.0])
        if pt_cam[2] <= 0.1 or pt_cam[2] > 6.0: 
            continue
            
        u, v = (pt_cam[0] * F / pt_cam[2]) + CX, (pt_cam[1] * F / pt_cam[2]) + CY
        if 0 <= u < 640 and 0 <= v < 480: 
            vis_indices.append(i)
            
    return vis_indices

def feature_detector(frame, depth_image):
    """Extracts ORB features and projects them into 3D using the depth map."""
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(frame, None)
    if not kp: 
        return [], None, np.array([])
        
    points, filtered_kp, filtered_des = [], [], []
    for i, keypoint in enumerate(kp):
        x, y = round(keypoint.pt[0]), round(keypoint.pt[1])
        if not (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]): 
            continue
            
        distance = depth_image[y, x]
        if distance == 0 or np.isnan(distance) or distance < 50 or distance > 5000: 
            continue
            
        filtered_kp.append(keypoint)
        filtered_des.append(des[i])
        
        depth = distance / 1000.0 # Convert mm to meters
        points.append([depth * (x - CX) / F, depth * (y - CY) / F, depth])
        
    return filtered_kp, np.array(filtered_des), np.array(points)

def pointcloud(points, pc_pub, stamp, frame_id="orb_odom"):
    header = std_msgs.msg.Header(stamp=stamp, frame_id=frame_id)
    pc_pub.publish(pcl2.create_cloud_xyz32(header, points.tolist() if isinstance(points, np.ndarray) else points))

def publish_odometry(node, current_time, global_x, global_y, global_yaw):
    odom_msg = Odometry()
    odom_msg.header.stamp = current_time
    odom_msg.header.frame_id = node.odom_frame
    odom_msg.child_frame_id = node.base_frame
    odom_msg.pose.pose.position.x = float(global_x)
    odom_msg.pose.pose.position.y = float(global_y)
    
    q = quaternion_from_euler(0, 0, global_yaw)
    odom_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    node.odom_pub.publish(odom_msg)

    # Broadcast TF
    t_msg = TransformStamped()
    t_msg.header.stamp, t_msg.header.frame_id, t_msg.child_frame_id = current_time, node.odom_frame, node.base_frame
    t_msg.transform.translation.x, t_msg.transform.translation.y = float(global_x), float(global_y)
    t_msg.transform.rotation = odom_msg.pose.pose.orientation
    node.tf_broadcaster.sendTransform(t_msg)

    # Append to Path for visualization
    if len(node.path_msg.poses) == 0 or math.sqrt((global_x - node.path_msg.poses[-1].pose.position.x)**2 + (global_y - node.path_msg.poses[-1].pose.position.y)**2) > 0.1:
        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = current_time, node.odom_frame
        pose.pose.position.x, pose.pose.position.y = float(global_x), float(global_y)
        pose.pose.orientation = odom_msg.pose.pose.orientation
        node.path_msg.poses.append(pose)

    node.path_msg.header.stamp = current_time
    node.path_pub.publish(node.path_msg)

def kabsch_2d(P, Q):
    P_2d, Q_2d = P[:, [0, 2]], Q[:, [0, 2]]
    P_mean, Q_mean = np.mean(P_2d, axis=0), np.mean(Q_2d, axis=0)
    P_c, Q_c = P_2d - P_mean, Q_2d - Q_mean
    num = np.sum(Q_c[:, 0] * P_c[:, 1] - Q_c[:, 1] * P_c[:, 0])
    den = np.sum(Q_c[:, 0] * P_c[:, 0] + Q_c[:, 1] * P_c[:, 1])
    theta = math.atan2(num, den)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R, P_mean - (R @ Q_mean)

def calculate_errors(P, Q, R, t): 
    return np.linalg.norm(P[:, [0, 2]] - ((R @ Q[:, [0, 2]].T).T + t), axis=1)

def ransac_kabsch(P, Q, iterations=200, threshold=0.05):
    """Finds the best 2D rigid transformation between two point clouds robustly."""
    n, best_count, best_inliers = len(P), 0, None
    if n < MIN_KEYPOINTS: 
        return None, None, P, Q, 0, 0
    if n < 8:
        R_hyp, t_hyp = kabsch_2d(P, Q)
        inlier_count = np.sum(calculate_errors(P, Q, R_hyp, t_hyp) < threshold)
        if inlier_count >= 3: 
            return R_hyp, t_hyp, P, Q, int(inlier_count), n - int(inlier_count)
        return None, None, P, Q, 0, 0

    for _ in range(iterations):
        idx = random.sample(range(n), 3)
        inliers = calculate_errors(P, Q, *kabsch_2d(P[idx], Q[idx])) < threshold
        inlier_count = np.sum(inliers)
        if inlier_count > best_count and inlier_count >= 3:
            best_count, best_inliers = inlier_count, inliers

    if best_inliers is None or np.sum(best_inliers) < 3: 
        return None, None, P, Q, 0, 0
    R, t = kabsch_2d(P[best_inliers], Q[best_inliers])
    return R, t, P[best_inliers], Q[best_inliers], int(np.sum(best_inliers)), n - int(np.sum(best_inliers))

def publish_pf_particles_markers(node, current_time):
    """Visualizes the swarm of particle hypotheses in RViz."""
    marker_array = MarkerArray()
    marker_array.markers.append(Marker(action=Marker.DELETEALL))
    
    for idx, particle in enumerate(node.pf.particles):
        weight = particle.weight
        m = Marker(ns="pf_particles", id=idx, type=Marker.ARROW, action=Marker.ADD)
        m.header.frame_id, m.header.stamp = node.odom_frame, current_time
        m.pose.position.x, m.pose.position.y = float(particle.x), float(particle.y)
        q = quaternion_from_euler(0, 0, float(particle.yaw))
        m.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        
        # Scale and color based on weight for visual clarity
        m.scale.x = 0.2 + (0.5 * weight)
        m.scale.y = m.scale.z = 0.05    
        m.color.r, m.color.g, m.color.b, m.color.a = min(1.0, weight * node.pf.N * 1.5), 0.5, 0.1, 0.8
        marker_array.markers.append(m)
    
    node.pf_particles_marker_pub.publish(marker_array)


def publish_landmark_covariances(node, current_time, best_map):
    """
    Visualizes the 2D covariance matrices of the landmarks as ellipses in RViz.
    Uses Eigenvalues for scale and Eigenvectors for rotation.
    """
    marker_array = MarkerArray()
    
    # 1. Clear old markers from the previous frame
    delete_marker = Marker(action=Marker.DELETEALL)
    marker_array.markers.append(delete_marker)
    
    # NEW LOOP: Iterate over your Landmark objects directly
    for idx, lm in enumerate(best_map):
        try:
            pt = lm.pos_3d
            cov = lm.covariance  # <-- NOTE: If your Landmark class named this 'cov' instead of 'covariance', change this!
            
            # 2. Extract Eigenvalues and Eigenvectors
            vals, vecs = np.linalg.eigh(cov)
            
            # Sort in descending order to find the primary axis of variance
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            
            # 3. Get rotation angle from the primary eigenvector
            angle = math.atan2(vecs[1, 0], vecs[0, 0])
            
            # 4. Calculate 3-Sigma confidence interval bounds (approx 99.7% confidence)
            scale_x = 6.0 * math.sqrt(max(vals[0], 1e-9))
            scale_y = 6.0 * math.sqrt(max(vals[1], 1e-9))
            
            # 5. Build the Ellipse Marker
            m = Marker(ns="landmark_covariances", id=idx, type=Marker.CYLINDER, action=Marker.ADD)
            m.header.frame_id = node.odom_frame
            m.header.stamp = current_time
            
            # Position at the landmark's 3D coordinates
            m.pose.position.x = float(pt[0])
            m.pose.position.y = float(pt[1])
            m.pose.position.z = float(pt[2])
            
            # Rotate to align with the eigenvectors
            q = quaternion_from_euler(0, 0, angle)
            m.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            
            # Scale the ellipse
            m.scale.x = scale_x
            m.scale.y = scale_y
            m.scale.z = 0.01 # Flatten the cylinder into a 2D disc
            
            # Color it translucent blue
            m.color.r = 0.0
            m.color.g = 0.5
            m.color.b = 1.0
            m.color.a = 0.3 
            
            marker_array.markers.append(m)
            
        except np.linalg.LinAlgError:
            continue
            
    # Publish the array!
    node.cov_pub.publish(marker_array)  



def main():
    rclpy.init()
    node = FastSlamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__': 
    main()