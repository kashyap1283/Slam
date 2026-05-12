#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pcl2
import std_msgs.msg
import numpy as np
import random
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf_transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

MIN_KEYPOINTS = 5
cx = 318.525
cy = 241.181 
f = 526.61 

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')  

        self.bridge = CvBridge()               
        self.subscription = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback, 10)   
        self.depth_sub= self.create_subscription(Image,
            '/serf01/nav_rgbd_1/depth/image_raw', self.depth_subscription, 10)  
        
        self.pc_pub = self.create_publisher(PointCloud2, '/orb_pointcloud', 10)
        self.current_pc_pub = self.create_publisher(PointCloud2, '/global_cloud', 10)
        self.odom_pub = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)
        self.tf_broadcaster = TransformBroadcaster(self)    
        self.path_pub = self.create_publisher(Path, '/orb_path', 10)
        
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'orb_odom'

        self.depth_image = None
        self.prev_kp = None
        self.prev_frame = None
        
        self.global_x = 0.0
        self.global_y = 0.0
        self.global_yaw = 0.0

        # --- SLAM Map Data Structures ---
        self.map_points_3d = []        
        self.map_descriptors = []      
        self.map_times_seen = []       
        self.map_times_expected = []   

    def depth_subscription(self,msg):
       self.depth_image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough") 

    def listener_callback(self, msg):
        self.current_time = msg.header.stamp

        if self.depth_image is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        kp, des, points = feature_detector(frame, self.depth_image)

        # Early exit if not enough keypoints
        if not kp or len(kp) < MIN_KEYPOINTS:
            return

        # 1. First Frame Initialization
        if self.prev_kp is None:
            self.prev_kp = kp
            self.prev_frame = frame
            
            self.add_new_landmarks(points, des, set()) 
            self.get_logger().info(f"Initialized map with {len(self.map_points_3d)} landmarks.")
            return

        try:
            # 2. Frustum Culling
            vis_indices = get_visible_landmarks(
                self.map_points_3d, self.map_times_expected, 
                self.global_x, self.global_y, self.global_yaw
            )

            P, Q = [], []
            matched_current_indices = set()

            # 3. Map-to-Frame Matching
            if len(vis_indices) > 0 and des is not None:
                vis_des_array = np.array([self.map_descriptors[i] for i in vis_indices])
                
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                map_matches = bf.match(vis_des_array, des)
                
                # --- CHANGE 1: Simplify matching loop to use Absolute coordinates ---
                for m in map_matches:
                    if m.distance < 60:
                        map_idx = vis_indices[m.queryIdx]
                        self.map_times_seen[map_idx] += 1 
                        
                        # Get absolute Global point [X, Y, Z]
                        pt_global = self.map_points_3d[map_idx]
                        
                        # Format as [X, 0, Y] so Kabsch's [:, [0,2]] slice grabs global X and Y
                        P.append([pt_global[0], 0.0, pt_global[1]])

                        # Get local camera point and convert to Robot Base frame
                        cam_x, cam_y, cam_z = points[m.trainIdx]
                        local_x = cam_z   # Camera Z is Robot Forward (X)
                        local_y = -cam_x  # Camera -X is Robot Left (Y)
                        
                        # Format as [X, 0, Y] so Kabsch's [:, [0,2]] slice grabs local X and Y
                        Q.append([local_x, 0.0, local_y])
                        
                        matched_current_indices.add(m.trainIdx)

            P = np.array(P)
            Q = np.array(Q)

            # 4. RANSAC and Absolute Pose Update
            R_2d, t_2d, P_in, Q_in, inlier_count, outlier_count = ransac_kabsch(P, Q)

            if R_2d is None:
                self.get_logger().warn("Lost tracking: Not enough valid inliers.")
                return

            # --- CHANGE 2: Replace "+=" with Absolute Assignment ---
            # t_2d and R_2d now represent your EXACT global position and rotation.
            self.global_yaw = math.atan2(R_2d[1, 0], R_2d[0, 0])
            self.global_x = t_2d[0]
            self.global_y = t_2d[1]
            # -------------------------------------------------------

            # Pointcloud generation
            if len(self.map_points_3d) > 0:
                pointcloud(np.array(self.map_points_3d), self.current_pc_pub, self.current_time, "orb_odom")

            publish_odometry(self, self.current_time, self.global_x, self.global_y, self.global_yaw)
            
            # 5. Map Maintenance
            added, map_points = self.add_new_landmarks(points, des, matched_current_indices)       
            removed = self.prune_landmarks()

            self.get_logger().info(f"Pose: X:{self.global_x:.2f} Y:{self.global_y:.2f} Yaw:{math.degrees(self.global_yaw):.1f}°")
            self.get_logger().info(f"Visible Matches: {len(P)} Inliers: {inlier_count} Outliers: {outlier_count}")

        finally:
            self.prev_kp = kp
            self.prev_frame = frame

    def add_new_landmarks(self, current_points, current_descriptors, matched_indices):
        cos_yaw = math.cos(self.global_yaw)
        sin_yaw = math.sin(self.global_yaw)
        new_count = 0

        for i in range(len(current_points)):
            if i in matched_indices:
                continue 

            x_cam, y_cam, z_cam = current_points[i]
            x_base, y_base, z_base = z_cam, -x_cam, -y_cam

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
        if not self.map_points_3d: return 0

        good_points, good_descriptors, good_seen, good_expected = [], [], [], []
        removed_count = 0

        for i in range(len(self.map_points_3d)):
            seen = self.map_times_seen[i]
            expected = self.map_times_expected[i]
            ratio = seen / float(expected) if expected > 0 else 0.0
            
            if expected > 3 and ratio < 0.25:
                removed_count += 1
                continue 
                
            good_points.append(self.map_points_3d[i])
            good_descriptors.append(self.map_descriptors[i])
            good_seen.append(seen)
            good_expected.append(expected)

        self.map_points_3d = good_points
        self.map_descriptors = good_descriptors
        self.map_times_seen = good_seen
        self.map_times_expected = good_expected

        return removed_count

#--------------------------------------------------------------------------------------------------

def get_visible_landmarks(map_points_3d, map_times_expected, global_x, global_y, global_yaw):
    vis_indices = []
    cos_yaw = math.cos(-global_yaw)
    sin_yaw = math.sin(-global_yaw)

    for i, pt in enumerate(map_points_3d):
        dx = pt[0] - global_x
        dy = pt[1] - global_y
        dz = pt[2]

        base_x = dx * cos_yaw - dy * sin_yaw
        base_y = dx * sin_yaw + dy * cos_yaw
        base_z = dz

        cam_x = -base_y
        cam_y = -base_z
        cam_z = base_x

        if cam_z <= 0.1 or cam_z > 5.0:  
            continue

        u = (cam_x * f / cam_z) + cx
        v = (cam_y * f / cam_z) + cy

        if 0 <= u < 640 and 0 <= v < 480:
            vis_indices.append(i)
            map_times_expected[i] += 1  

    return vis_indices

def feature_detector(frame, depth_image):
    orb = cv2.ORB_create()                        
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
        
        cv2.putText(img, f"{depth:.1f} m", [x,y], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1) 
        
    img = cv2.drawKeypoints(img, Filtered_kp, None, color=(0,255,0), flags=0)
    cv2.imshow("FeatureDetector", img)
    cv2.waitKey(1)
    
    return Filtered_kp, np.array(Filtered_des), np.array(points)

def pointcloud(points, pc_pub, stamp, frame_id="orb_odom"):
    header = std_msgs.msg.Header()
    header.stamp, header.frame_id = stamp, frame_id
    points_list = points.tolist() if isinstance(points, np.ndarray) else points
    cloud_msg = pcl2.create_cloud_xyz32(header, points_list)
    pc_pub.publish(cloud_msg)

def publish_odometry(node, current_time, global_x, global_y, global_yaw):
    odom_msg = Odometry()
    odom_msg.header.stamp = current_time
    odom_msg.header.frame_id = 'orb_odom'
    odom_msg.child_frame_id = 'base_link'

    odom_msg.pose.pose.position.x = float(global_x)
    odom_msg.pose.pose.position.y = float(global_y)
    odom_msg.pose.pose.position.z = 0.0

    q = quaternion_from_euler(0, 0, global_yaw)
    odom_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    odom_msg.pose.covariance = [
        0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.05, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.05, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.05
    ]

    node.odom_pub.publish(odom_msg)

    t_msg = TransformStamped()
    t_msg.header.stamp = current_time
    t_msg.header.frame_id = 'orb_odom'
    t_msg.child_frame_id = 'base_link'
    t_msg.transform.translation.x = float(global_x)
    t_msg.transform.translation.y = float(global_y)
    t_msg.transform.translation.z = 0.0
    t_msg.transform.rotation = odom_msg.pose.pose.orientation
    node.tf_broadcaster.sendTransform(t_msg)

    if len(node.path_msg.poses) == 0:
        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = current_time, 'orb_odom'
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(global_x), float(global_y), 0.0
        pose.pose.orientation = odom_msg.pose.pose.orientation
        node.path_msg.poses.append(pose)
    else:
        last_pose = node.path_msg.poses[-1]
        dist = math.sqrt((global_x - last_pose.pose.position.x)**2 + (global_y - last_pose.pose.position.y)**2)
        if dist > 0.1:  
            pose = PoseStamped()
            pose.header.stamp, pose.header.frame_id = current_time, 'orb_odom'
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(global_x), float(global_y), 0.0
            pose.pose.orientation = odom_msg.pose.pose.orientation
            node.path_msg.poses.append(pose)
    
    node.path_msg.header.stamp = current_time
    node.path_pub.publish(node.path_msg)

def kabsch_2d(P, Q):
    P_2d, Q_2d = P[:, [0, 2]], Q[:, [0, 2]] 
    P_mean, Q_mean = np.mean(P_2d, axis=0), np.mean(Q_2d, axis=0)
    P_c, Q_c = P_2d - P_mean, Q_2d - Q_mean

    # Updated to np.sum for math safety and speed
    num = np.sum(Q_c[:, 0] * P_c[:, 1] - Q_c[:, 1] * P_c[:, 0])
    den = np.sum(Q_c[:, 0] * P_c[:, 0] + Q_c[:, 1] * P_c[:, 1])

    theta = math.atan2(num, den)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = P_mean - (R @ Q_mean)

    return R, t

def calculate_errors(P, Q, R, t):
    P_2d, Q_2d = P[:, [0, 2]], Q[:, [0, 2]] 
    Q_transformed = (R @ Q_2d.T).T + t
    return np.linalg.norm(P_2d - Q_transformed, axis=1)

def ransac_kabsch(P, Q, iterations=200, threshold=0.05):
    best_inliers, best_count = None, 0
    n = len(P)
    
    if n < MIN_KEYPOINTS: 
        return None, None, P, Q, 0, 0

    for _ in range(iterations):
        idx = random.sample(range(n), 3)
        P_s, Q_s = P[idx], Q[idx]

        R_hyp, t_hyp = kabsch_2d(P_s, Q_s)
        
        errors = calculate_errors(P, Q, R_hyp, t_hyp)
        inliers = errors < threshold
        inlier_count = np.sum(inliers)

        # Added safety check to require at least 3 inliers before accepting subset
        if inlier_count > best_count and inlier_count >= 3:
            best_count = inlier_count
            best_inliers = inliers

    if best_inliers is None or np.sum(best_inliers) < MIN_KEYPOINTS:
        return None, None, P, Q, 0, 0

    P_in, Q_in = P[best_inliers], Q[best_inliers]
    R, t = kabsch_2d(P_in, Q_in)
    
    inlier_count = np.sum(best_inliers)
    outlier_count = n - inlier_count

    return R, t, P_in, Q_in, inlier_count, outlier_count

def main():     
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()