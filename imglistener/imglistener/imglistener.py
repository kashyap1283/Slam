#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pcl2
import std_msgs.msg
import numpy as np
import random
import csv
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Quaternion, TransformStamped, PoseStamped
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from tf_transformations import quaternion_from_euler, euler_from_quaternion, quaternion_matrix

from imglistener.ekf import ExtKalman   

MIN_KEYPOINTS = 5
MIN_INLIERS   = 12
cx = 318.525
cy = 241.181
f  = 526.61

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()
        
        # --- Camera Subscriptions ---
        self.subscription = self.create_subscription(
            Image, '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/serf01/nav_rgbd_1/depth/image_raw', self.depth_subscription, 10)

        # --- IMU & Wheel Odom Subscriptions ---
        self.wheel_sub = self.create_subscription(
            Odometry, '/serf01/odometry/wheel', self.wheel_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/serf01/odometry/imu', self.imu_callback, 10)

        # --- Publishers ---
        self.pc_pub         = self.create_publisher(PointCloud2, '/orb_pointcloud', 10)
        self.current_pc_pub = self.create_publisher(PointCloud2, '/global_cloud', 10)
        self.odom_pub       = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)
        self.path_pub       = self.create_publisher(Path, '/orb_path', 10)

        # --- TF & Path Setup ---
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.kinect_to_base_matrix = None        

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'orb_odom'
        
        self.depth_image = None
        self.prev_kp     = None
        self.prev_frame  = None
        
        self.global_x    = 0.0
        self.global_y    = 0.0
        self.global_yaw  = 0.0

        self.initialized      = False
        self.init_frame_count = 0
        self.INIT_FRAMES      = 15
        self.theta_last       = 0.0

        self.map_points_3d      = []
        self.map_descriptors    = []
        self.map_times_seen     = []
        self.map_times_expected = []
        
        # --- Odometry Accumulators ---
        self.prev_wheel_x   = None
        self.prev_wheel_y   = None
        self.prev_wheel_yaw = None
        self.prev_imu_yaw   = None
        
        self.odom_dx_accum     = 0.0
        self.odom_dy_accum     = 0.0
        self.imu_dtheta_accum  = 0.0
        
        # Control input vector [dx, dy, dtheta] used by the Process Model
        self.current_u = np.array([0.0, 0.0, 0.0])
        
        # --- EKF Initialization ---
        self.ekf = ExtKalman(
            x=np.array([0.0, 0.0, 0.0]),
            state_func=self.ekf_state_func,
            meas_func=self.ekf_meas_func,
            JF=np.eye(3), 
            JH=np.eye(3), 
            R=np.diag([0.01, 0.01, 0.01]), 
            Q=np.diag([5, 5, 1])
        )

    # ---------------------------------------------------------
    # IMU & Wheel Callbacks for dead-reckoning fallback
    # ---------------------------------------------------------
    def imu_callback(self, msg: Imu):
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        _, _, yaw = euler_from_quaternion(quat)
        
        if self.prev_imu_yaw is not None:
            dyaw = yaw - self.prev_imu_yaw
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
            self.imu_dtheta_accum += dyaw
            
        self.prev_imu_yaw = yaw

    def wheel_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _, _, yaw = euler_from_quaternion(quat)
        
        if self.prev_wheel_x is not None:
            # Global differences
            dx_global = x - self.prev_wheel_x
            dy_global = y - self.prev_wheel_y
            
            # Rotate into robot's local frame
            cos_y = math.cos(self.prev_wheel_yaw)
            sin_y = math.sin(self.prev_wheel_yaw)
            dx_local =  dx_global * cos_y + dy_global * sin_y
            dy_local = -dx_global * sin_y + dy_global * cos_y
            
            self.odom_dx_accum += dx_local
            self.odom_dy_accum += dy_local
            
        self.prev_wheel_x = x
        self.prev_wheel_y = y
        self.prev_wheel_yaw = yaw

    # ---------------------------------------------------------
    # EKF Functions
    # ---------------------------------------------------------
    def ekf_state_func(self, x):
        dx, dy, dtheta = self.current_u
        theta = x[2]
        
        new_x = x[0] + dx * math.cos(theta) - dy * math.sin(theta)
        new_y = x[1] + dx * math.sin(theta) + dy * math.cos(theta)
        new_theta = x[2] + dtheta
        
        # Keep yaw wrapped
        new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))
        
        return np.array([new_x, new_y, new_theta])
    
    def ekf_meas_func(self, x):
        return x.copy()

    def depth_subscription(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def listener_callback(self, msg):
        self.current_time = msg.header.stamp
        
        if self.kinect_to_base_matrix is None:
            try:
                t = self.tf_buffer.lookup_transform('base_link', 'kinect_depth', rclpy.time.Time())
                quat = [t.transform.rotation.x, t.transform.rotation.y, 
                        t.transform.rotation.z, t.transform.rotation.w]
                self.kinect_to_base_matrix = quaternion_matrix(quat)
                self.kinect_to_base_matrix[0, 3] = t.transform.translation.x
                self.kinect_to_base_matrix[1, 3] = t.transform.translation.y
                self.kinect_to_base_matrix[2, 3] = t.transform.translation.z
                self.base_to_kinect_matrix = np.linalg.inv(self.kinect_to_base_matrix)
            except Exception as e:
                self.get_logger().info(f"Waiting for static TF: {e}")
                return
            
        if self.depth_image is None:
            return

        # ── 1. GRAB CONTROL INPUTS (WHEELS & IMU) ─────────────────────────
        u_dx     = self.odom_dx_accum
        u_dy     = self.odom_dy_accum
        u_dtheta = self.imu_dtheta_accum
        
        self.odom_dx_accum    = 0.0
        self.odom_dy_accum    = 0.0
        self.imu_dtheta_accum = 0.0
        
        self.current_u = np.array([u_dx, u_dy, u_dtheta])

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        kp, des, points = feature_detector(frame, self.depth_image)

        if not kp or len(kp) < MIN_KEYPOINTS:
            return

        # ── PHASE 1: silent map build ──────────────────────────────────────────
        if not self.initialized:
            self.add_new_landmarks(points, des, set())
            self.init_frame_count += 1
            if self.init_frame_count >= self.INIT_FRAMES:
                self.initialized = True
            self.prev_kp    = kp
            self.prev_frame = frame
            return

        # ── PHASE 2: normal tracking ───────────────────────────────────────────
        if self.prev_kp is None:
            self.prev_kp    = kp
            self.prev_frame = frame
            return

        try:
            vis_indices = get_visible_landmarks(
                self.map_points_3d, self.map_times_expected,
                self.global_x, self.global_y, self.global_yaw, self.base_to_kinect_matrix)

            P, Q = [], []
            matched_current_indices = set()

            if len(vis_indices) > 0 and des is not None:
                vis_des_array = np.array([self.map_descriptors[i] for i in vis_indices])
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                map_matches = bf.match(vis_des_array, des)

                cos_yaw = math.cos(-self.global_yaw)
                sin_yaw = math.sin(-self.global_yaw)

                for m in map_matches:
                    if m.distance < 60:
                        map_idx = vis_indices[m.queryIdx]
                        self.map_times_seen[map_idx] += 1

                        pt = self.map_points_3d[map_idx]
                        dx = pt[0] - self.global_x
                        dy = pt[1] - self.global_y
                        dz = pt[2]

                        base_x =  dx * cos_yaw - dy * sin_yaw
                        base_y =  dx * sin_yaw + dy * cos_yaw
                        base_z =  dz

                        pt_base_loop = np.array([base_x, base_y, base_z, 1.0])
                        pt_cam_loop = self.base_to_kinect_matrix @ pt_base_loop
                        cam_x, cam_y, cam_z = pt_cam_loop[0], pt_cam_loop[1], pt_cam_loop[2]
                        
                        if cam_z > 0.1:
                            observed_depth  = points[m.trainIdx][2]
                            depth_tolerance = 0.4 / (1.0 + 5.0 * abs(self.theta_last))
                            depth_ratio     = abs(cam_z - observed_depth) / max(cam_z, observed_depth)
                            if depth_ratio > depth_tolerance:
                                continue

                        # FIX: Build P and Q in the native CAMERA frame to preserve geometry
                        P.append([cam_x, cam_y, cam_z])
                        Q.append(points[m.trainIdx])
                        
                        matched_current_indices.add(m.trainIdx)

            P = np.array(P)
            Q = np.array(Q)

            n_matches     = len(P)
            ransac_thresh = 0.05 if n_matches >= 15 else 0.10

            R_2d, t_2d, P_in, Q_in, inlier_count, outlier_count = \
                ransac_kabsch(P, Q, threshold=ransac_thresh)

            # ── 2. PREPARE EKF MATHEMATICS ─────────────────────────────────────
            
            theta_current = self.ekf.x[2]
            cos_t = math.cos(theta_current)
            sin_t = math.sin(theta_current)

            JF = np.array([
                [1.0, 0.0, -u_dx * sin_t - u_dy * cos_t],
                [0.0, 1.0,  u_dx * cos_t - u_dy * sin_t],
                [0.0, 0.0,  1.0]
            ])
            self.ekf.setJF(JF)
            self.ekf.setJH(np.eye(3))
            self.ekf.setQ(np.diag([0.01, 0.01, 0.005]))

            # ── 3. APPLY VO AS A MEASUREMENT ───────────────────────────────────
            if R_2d is None or inlier_count < MIN_INLIERS:
                self.get_logger().warn(f"VO weak. Coasting on Wheel/IMU.")
                
                z_pose = self.ekf_state_func(self.ekf.x)
                self.ekf.setR(np.diag([1e6, 1e6, 1e6]))
                
                self.theta_last = u_dtheta
                self.add_new_landmarks(points, des, matched_current_indices)
            else:
                # ==========================================================
                # THE LEVER ARM FIX
                # We build the 4x4 matrix of the Camera's motion, then 
                # multiply it by the TF tree to extract the Base's true motion.
                # ==========================================================
                T_cam = np.eye(4)
                T_cam[0, 0] = R_2d[0, 0]
                T_cam[0, 2] = R_2d[0, 1]
                T_cam[2, 0] = R_2d[1, 0]
                T_cam[2, 2] = R_2d[1, 1]
                T_cam[0, 3] = t_2d[0] # X trans
                T_cam[2, 3] = t_2d[1] # Z trans

                # T_base = (Base->Cam) * T_cam * (Cam->Base)
                T_base = self.kinect_to_base_matrix @ T_cam @ self.base_to_kinect_matrix
                
                vo_dx = T_base[0, 3]
                vo_dy = T_base[1, 3]
                vo_dtheta = math.atan2(T_base[1, 0], T_base[0, 0])
                # ==========================================================
                
                z_yaw = theta_current + vo_dtheta
                z_x   = self.ekf.x[0] + vo_dx * cos_t - vo_dy * sin_t
                z_y   = self.ekf.x[1] + vo_dx * sin_t + vo_dy * cos_t
                z_pose = np.array([z_x, z_y, z_yaw])
                
                self.ekf.setR(np.diag([0.2, 0.2, 0.05])) 
                
                self.theta_last = vo_dtheta

            # ── 4. UPDATE EKF & EXTRACT STATE ──────────────────────────────────
            self.ekf.update(z_pose)
            
            self.global_x   = self.ekf.x[0]
            self.global_y   = self.ekf.x[1]
            self.global_yaw = self.ekf.x[2]
            
            self.global_yaw = math.atan2(math.sin(self.global_yaw), math.cos(self.global_yaw))

            # ── 5. PUBLISH & CLEANUP ───────────────────────────────────────────
            if len(self.map_points_3d) > 0:
                pointcloud(np.array(self.map_points_3d),
                           self.current_pc_pub, self.current_time, "orb_odom")

            publish_odometry(self, self.current_time,
                             self.global_x, self.global_y, self.global_yaw, inlier_count)

            if inlier_count >= MIN_INLIERS:
                self.add_new_landmarks(points, des, matched_current_indices)
                self.prune_landmarks()

            self.get_logger().info(
                f"Smoothed Pose: X:{self.global_x:.2f} Y:{self.global_y:.2f} "
                f"Yaw:{math.degrees(self.global_yaw):.1f}°")

        finally:
            self.prev_kp    = kp
            self.prev_frame = frame

    def add_new_landmarks(self, current_points, current_descriptors, matched_indices):
        cos_yaw   = math.cos(self.global_yaw)
        sin_yaw   = math.sin(self.global_yaw)
        new_count = 0

        for i in range(len(current_points)):
            if i in matched_indices:
                continue

            x_cam, y_cam, z_cam = current_points[i]
            pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
            pt_base = self.kinect_to_base_matrix @ pt_cam
            x_base, y_base, z_base = pt_base[0], pt_base[1], pt_base[2]
            
            x_global = x_base * cos_yaw - y_base * sin_yaw + self.global_x
            y_global = x_base * sin_yaw + y_base * cos_yaw + self.global_y
            z_global = z_base

            self.map_points_3d.append([x_global, y_global, z_global])
            self.map_descriptors.append(current_descriptors[i])
            self.map_times_seen.append(1)
            self.map_times_expected.append(1)
            new_count += 1

        return new_count, self.map_points_3d


    def prune_landmarks(self):
        if not self.map_points_3d:
            return 0

        good_points, good_descriptors, good_seen, good_expected = [], [], [], []
        removed_count = 0

        cos_yaw = math.cos(-self.global_yaw)
        sin_yaw = math.sin(-self.global_yaw)

        for i in range(len(self.map_points_3d)):
            seen     = self.map_times_seen[i]
            expected = self.map_times_expected[i]
            ratio    = seen / float(expected) if expected > 0 else 0.0

            pt    = self.map_points_3d[i]
            dx    = pt[0] - self.global_x
            dy    = pt[1] - self.global_y
            cam_z = dx * cos_yaw - dy * sin_yaw

            if cam_z < 0:
                removed_count += 1
                continue

            if expected > 4 and ratio < 0.35:
                removed_count += 1
                continue

            good_points.append(self.map_points_3d[i])
            good_descriptors.append(self.map_descriptors[i])
            good_seen.append(seen)
            good_expected.append(expected)

        self.map_points_3d      = good_points
        self.map_descriptors    = good_descriptors
        self.map_times_seen     = good_seen
        self.map_times_expected = good_expected
        return removed_count

# ---------------------------------------------------------------------------

def get_visible_landmarks(map_points_3d, map_times_expected,
                          global_x, global_y, global_yaw,base_to_kinect_matrix):
    vis_indices = []
    cos_yaw = math.cos(-global_yaw)
    sin_yaw = math.sin(-global_yaw)

    MAX_LANDMARK_DISTANCE = 4.0

    for i, pt in enumerate(map_points_3d):
        dx = pt[0] - global_x
        dy = pt[1] - global_y

        if (dx * dx + dy * dy) > (MAX_LANDMARK_DISTANCE * MAX_LANDMARK_DISTANCE):
            continue

        dz = pt[2]

        base_x =  dx * cos_yaw - dy * sin_yaw
        base_y =  dx * sin_yaw + dy * cos_yaw
        base_z =  dz

        pt_base = np.array([base_x, base_y, base_z, 1.0])
        pt_cam = base_to_kinect_matrix @ pt_base
        cam_x, cam_y, cam_z = pt_cam[0], pt_cam[1], pt_cam[2]

        if cam_z <= 0.1 or cam_z > 4.0:
            continue

        u = (cam_x * f / cam_z) + cx
        v = (cam_y * f / cam_z) + cy

        if 0 <= u < 640 and 0 <= v < 480:
            vis_indices.append(i)
            map_times_expected[i] += 1

    return vis_indices


def feature_detector(frame, depth_image):
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(frame, None)

    if not kp:
        return [], None, np.array([])

    points, Filtered_kp, Filtered_des = [], [], []
    img = frame.copy()

    for i, keypoint in enumerate(kp):
        x, y = round(keypoint.pt[0]), round(keypoint.pt[1])

        if not (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]):
            continue

        distance = depth_image[y, x]
        if distance == 0 or np.isnan(distance) or distance < 50 or distance > 5000:
            continue

        Filtered_kp.append(keypoint)
        Filtered_des.append(des[i])

        depth = distance / 1000.0
        X = depth * (x - cx) / f
        Y = depth * (y - cy) / f
        points.append([X, Y, depth])

        cv2.putText(img, f"{depth:.1f} m", [x, y],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    img = cv2.drawKeypoints(img, Filtered_kp, None, color=(0, 255, 0), flags=0)
    cv2.imshow("FeatureDetector", img)
    cv2.waitKey(1)

    return Filtered_kp, np.array(Filtered_des), np.array(points)


def pointcloud(points, pc_pub, stamp, frame_id="orb_odom"):
    header = std_msgs.msg.Header()
    header.stamp, header.frame_id = stamp, frame_id
    points_list = points.tolist() if isinstance(points, np.ndarray) else points
    cloud_msg   = pcl2.create_cloud_xyz32(header, points_list)
    pc_pub.publish(cloud_msg)


def publish_odometry(node, current_time, global_x, global_y, global_yaw,inlier_count):
    odom_msg = Odometry()
    odom_msg.header.stamp    = current_time
    odom_msg.header.frame_id = 'orb_odom'
    odom_msg.child_frame_id  = 'base_link'

    odom_msg.pose.pose.position.x = float(global_x)
    odom_msg.pose.pose.position.y = float(global_y)
    odom_msg.pose.pose.position.z = 0.0

    q = quaternion_from_euler(0, 0, global_yaw)
    odom_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    cov_x = 0.5
    cov_y = 0.05
    cov_yaw = 0.02

    if inlier_count < 25:
        penalty = 10.0
        cov_x *= penalty
        cov_y *= penalty
        cov_yaw *= penalty

    odom_msg.pose.covariance = [
        cov_x, 0.0,   0.0, 0.0, 0.0, 0.0,
        0.0,   cov_y, 0.0, 0.0, 0.0, 0.0,
        0.0,   0.0,   0.1, 0.0, 0.0, 0.0,
        0.0,   0.0,   0.0, 0.1, 0.0, 0.0,
        0.0,   0.0,   0.0, 0.0, 0.1, 0.0,
        0.0,   0.0,   0.0, 0.0, 0.0, cov_yaw
    ]

    node.odom_pub.publish(odom_msg)

    t_msg = TransformStamped()
    t_msg.header.stamp            = current_time
    t_msg.header.frame_id         = 'orb_odom'
    t_msg.child_frame_id          = 'base_link'
    t_msg.transform.translation.x = float(global_x)
    t_msg.transform.translation.y = float(global_y)
    t_msg.transform.translation.z = 0.0
    t_msg.transform.rotation      = odom_msg.pose.pose.orientation
    node.tf_broadcaster.sendTransform(t_msg)

    if len(node.path_msg.poses) == 0:
        pose = PoseStamped()
        pose.header.stamp     = current_time
        pose.header.frame_id  = 'orb_odom'
        pose.pose.position.x  = float(global_x)
        pose.pose.position.y  = float(global_y)
        pose.pose.position.z  = 0.0
        pose.pose.orientation = odom_msg.pose.pose.orientation
        node.path_msg.poses.append(pose)
    else:
        last_pose = node.path_msg.poses[-1]
        dist = math.sqrt(
            (global_x - last_pose.pose.position.x) ** 2 +
            (global_y - last_pose.pose.position.y) ** 2)
        if dist > 0.1:
            pose = PoseStamped()
            pose.header.stamp     = current_time
            pose.header.frame_id  = 'orb_odom'
            pose.pose.position.x  = float(global_x)
            pose.pose.position.y  = float(global_y)
            pose.pose.position.z  = 0.0
            pose.pose.orientation = odom_msg.pose.pose.orientation
            node.path_msg.poses.append(pose)

    node.path_msg.header.stamp = current_time
    node.path_pub.publish(node.path_msg)


def kabsch_2d(P, Q):
    # Restored to original indices to keep the Camera geometry intact.
    P_2d, Q_2d = P[:, [0, 2]], Q[:, [0, 2]]
    P_mean, Q_mean = np.mean(P_2d, axis=0), np.mean(Q_2d, axis=0)
    P_c,    Q_c    = P_2d - P_mean,         Q_2d - Q_mean

    num   = np.sum(Q_c[:, 0] * P_c[:, 1] - Q_c[:, 1] * P_c[:, 0])
    den   = np.sum(Q_c[:, 0] * P_c[:, 0] + Q_c[:, 1] * P_c[:, 1])
    theta = math.atan2(num, den)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    t = P_mean - (R @ Q_mean)
    return R, t


def calculate_errors(P, Q, R, t):
    P_2d, Q_2d    = P[:, [0, 2]], Q[:, [0, 2]]
    Q_transformed = (R @ Q_2d.T).T + t
    return np.linalg.norm(P_2d - Q_transformed, axis=1)


def ransac_kabsch(P, Q, iterations=200, threshold=0.05):
    best_inliers, best_count = None, 0
    n = len(P)

    if n < MIN_KEYPOINTS:
        return None, None, P, Q, 0, 0

    for _ in range(iterations):
        idx      = random.sample(range(n), 3)
        P_s, Q_s = P[idx], Q[idx]

        R_hyp, t_hyp = kabsch_2d(P_s, Q_s)
        errors        = calculate_errors(P, Q, R_hyp, t_hyp)
        inliers       = errors < threshold
        inlier_count  = np.sum(inliers)

        if inlier_count > best_count and inlier_count >= 3:
            best_count   = inlier_count
            best_inliers = inliers

    if best_inliers is None or np.sum(best_inliers) < MIN_KEYPOINTS:
        return None, None, P, Q, 0, 0

    P_in, Q_in = P[best_inliers], Q[best_inliers]
    R, t       = kabsch_2d(P_in, Q_in)

    return R, t, P_in, Q_in, int(np.sum(best_inliers)), n - int(np.sum(best_inliers))


def main():
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()