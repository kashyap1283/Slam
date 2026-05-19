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

        self.subscription = self.create_subscription(
            Image, '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/serf01/nav_rgbd_1/depth/image_raw', self.depth_subscription, 10)
        self.wheel_sub = self.create_subscription(
            Odometry, '/serf01/odometry/wheel', self.wheel_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/serf01/odometry/imu', self.imu_callback, 10)

        self.pc_pub         = self.create_publisher(PointCloud2, '/orb_pointcloud', 10)
        self.current_pc_pub = self.create_publisher(PointCloud2, '/global_cloud', 10)
        self.odom_pub       = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)
        self.path_pub       = self.create_publisher(Path, '/orb_path', 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer      = Buffer()
        self.tf_listener    = TransformListener(self.tf_buffer, self)
        self.kinect_to_base_matrix = None

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'orb_odom'

        self.depth_image = None
        self.prev_kp     = None
        self.prev_frame  = None

        self.global_x   = 0.0
        self.global_y   = 0.0
        self.global_yaw = 0.0

        self.initialized      = False
        self.init_frame_count = 0
        self.INIT_FRAMES      = 15
        self.theta_last       = 0.0

        self.map_points_3d      = []  # global metres [x, y, z]
        self.map_descriptors    = []
        self.map_times_seen     = []
        self.map_times_expected = []

        self.prev_wheel_x   = None
        self.prev_wheel_y   = None
        self.prev_wheel_yaw = None
        self.prev_imu_yaw   = None

        self.odom_dx_accum    = 0.0
        self.odom_dy_accum    = 0.0
        self.imu_dtheta_accum = 0.0
        self.current_u        = np.array([0.0, 0.0, 0.0])

        self.ekf = ExtKalman(
            x=np.array([0.0, 0.0, 0.0]),
            state_func=self.ekf_state_func,
            meas_func=self.ekf_meas_func,
            JF=np.eye(3),
            JH=np.eye(3),
            R=np.diag([0.01, 0.01, 0.01]),
            Q=np.diag([5, 5, 1])
        )

    # ── IMU & Wheel ───────────────────────────────────────────────────

    def imu_callback(self, msg: Imu):
        quat = [msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w]
        _, _, yaw = euler_from_quaternion(quat)
        if self.prev_imu_yaw is not None:
            dyaw = math.atan2(math.sin(yaw - self.prev_imu_yaw),
                              math.cos(yaw - self.prev_imu_yaw))
            self.imu_dtheta_accum += dyaw
        self.prev_imu_yaw = yaw

    def wheel_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _, _, yaw = euler_from_quaternion(quat)
        if self.prev_wheel_x is not None:
            dx_g = x - self.prev_wheel_x
            dy_g = y - self.prev_wheel_y
            cos_y = math.cos(self.prev_wheel_yaw)
            sin_y = math.sin(self.prev_wheel_yaw)
            self.odom_dx_accum +=  dx_g * cos_y + dy_g * sin_y
            self.odom_dy_accum += -dx_g * sin_y + dy_g * cos_y
        self.prev_wheel_x, self.prev_wheel_y, self.prev_wheel_yaw = x, y, yaw

    # ── EKF ──────────────────────────────────────────────────────────

    def ekf_state_func(self, x):
        dx, dy, dtheta = self.current_u
        theta = x[2]
        return np.array([
            x[0] + dx * math.cos(theta) - dy * math.sin(theta),
            x[1] + dx * math.sin(theta) + dy * math.cos(theta),
            math.atan2(math.sin(x[2] + dtheta), math.cos(x[2] + dtheta))
        ])

    def ekf_meas_func(self, x):
        return x.copy()

    def depth_subscription(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    # ── Main callback ─────────────────────────────────────────────────

    def listener_callback(self, msg):
        self.current_time = msg.header.stamp

        if self.kinect_to_base_matrix is None:
            try:
                t = self.tf_buffer.lookup_transform(
                    'base_link', 'kinect_depth', rclpy.time.Time())
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

        u_dx     = self.odom_dx_accum
        u_dy     = self.odom_dy_accum
        u_dtheta = self.imu_dtheta_accum
        self.odom_dx_accum = self.odom_dy_accum = self.imu_dtheta_accum = 0.0
        self.current_u = np.array([u_dx, u_dy, u_dtheta])

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # FIX 1: feature_detector now returns base-frame XY points (metres)
        # No kinect_to_base applied inside add_new_landmarks anymore
        kp, des, pts_base = feature_detector(
            frame, self.depth_image, self.kinect_to_base_matrix)

        if not kp or len(kp) < MIN_KEYPOINTS:
            return

        # ── Phase 1: silent map build ─────────────────────────────────
        if not self.initialized:
            self.add_new_landmarks(pts_base, des, set())
            self.init_frame_count += 1
            if self.init_frame_count >= self.INIT_FRAMES:
                self.initialized = True
            self.prev_kp    = kp
            self.prev_frame = frame
            return

        if self.prev_kp is None:
            self.prev_kp    = kp
            self.prev_frame = frame
            return

        try:
            vis_indices = get_visible_landmarks(
                self.map_points_3d, self.map_times_expected,
                self.global_x, self.global_y, self.global_yaw,
                self.base_to_kinect_matrix)

            # FIX 2: P and Q built in robot-local base-frame XY (metres)
            P, Q = [], []
            matched_current_indices = set()

            cos_yaw = math.cos(-self.global_yaw)
            sin_yaw = math.sin(-self.global_yaw)

            if len(vis_indices) > 0 and des is not None:
                vis_des_array = np.array([self.map_descriptors[i] for i in vis_indices])
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                map_matches = bf.match(vis_des_array, des)

                for m in map_matches:
                    if m.distance < 60:
                        map_idx = vis_indices[m.queryIdx]
                        self.map_times_seen[map_idx] += 1

                        pt = self.map_points_3d[map_idx]
                        dx = pt[0] - self.global_x
                        dy = pt[1] - self.global_y

                        # Map point in robot-local base XY
                        local_x =  dx * cos_yaw - dy * sin_yaw
                        local_y =  dx * sin_yaw + dy * cos_yaw

                        # FIX 2: P = map point local XY, Q = observed base-frame XY
                        P.append([local_x, local_y])
                        Q.append(pts_base[m.trainIdx][:2])  # base XY in metres
                        matched_current_indices.add(m.trainIdx)

            P = np.array(P)
            Q = np.array(Q)

            n_matches     = len(P)
            ransac_thresh = 0.05 if n_matches >= 15 else 0.10

            # FIX 3: ransac_kabsch now works on Nx2 XY — no XZ slicing
            R_2d, t_2d, P_in, Q_in, inlier_count, outlier_count = \
                ransac_kabsch(P, Q, threshold=ransac_thresh)

            # ── EKF ──────────────────────────────────────────────────
            theta_current = self.ekf.x[2]
            cos_t = math.cos(theta_current)
            sin_t = math.sin(theta_current)

            JF = np.array([
                [cos_t, -sin_t, -u_dx * sin_t - u_dy * cos_t],
                [sin_t,  cos_t,  u_dx * cos_t - u_dy * sin_t],
                [0.0,    0.0,    1.0]
            ])
            self.ekf.setJF(JF)
            self.ekf.setJH(np.eye(3))
            self.ekf.setQ(np.diag([0.01, 0.01, 0.005]))

            if R_2d is None or inlier_count < MIN_INLIERS:
                self.get_logger().warn("VO weak. Coasting on Wheel/IMU.")
                z_pose = self.ekf_state_func(self.ekf.x)
                self.ekf.setR(np.diag([1e6, 1e6, 1e6]))
                self.theta_last = u_dtheta
                self.add_new_landmarks(pts_base, des, matched_current_indices)
            else:
                # FIX 4: correct yaw and translation from 2D XY kabsch
                vo_dtheta = math.atan2(R_2d[1, 0], R_2d[0, 0])
                vo_dx     = t_2d[0]   # base X = robot forward
                vo_dy     = t_2d[1]   # base Y = robot lateral

                z_yaw = theta_current + vo_dtheta
                z_x   = self.ekf.x[0] + vo_dx * cos_t - vo_dy * sin_t
                z_y   = self.ekf.x[1] + vo_dx * sin_t + vo_dy * cos_t
                z_pose = np.array([z_x, z_y, z_yaw])

                self.ekf.setR(np.diag([0.2, 0.2, 0.05]))
                self.theta_last = vo_dtheta

            self.ekf.update(z_pose)

            self.global_x   = self.ekf.x[0]
            self.global_y   = self.ekf.x[1]
            self.global_yaw = math.atan2(
                math.sin(self.ekf.x[2]), math.cos(self.ekf.x[2]))

            # Publish pointcloud
            if len(self.map_points_3d) > 0:
                pointcloud(np.array(self.map_points_3d),
                           self.current_pc_pub, self.current_time, "orb_odom")

            publish_odometry(self, self.current_time,
                             self.global_x, self.global_y,
                             self.global_yaw, inlier_count)

            if inlier_count >= MIN_INLIERS:
                self.add_new_landmarks(pts_base, des, matched_current_indices)
                self.prune_landmarks()

            self.get_logger().info(
                f"Pose X:{self.global_x:.2f} Y:{self.global_y:.2f} "
                f"Yaw:{math.degrees(self.global_yaw):.1f}° | "
                f"Inliers:{inlier_count} Map:{len(self.map_points_3d)}")

        finally:
            self.prev_kp    = kp
            self.prev_frame = frame

    # ── Landmark management ──────────────────────────────────────────

    def add_new_landmarks(self, pts_base, descriptors, matched_indices):
        """
        pts_base: Nx3 array of base-frame XY points in metres.
        FIX 5: No kinect_to_base transform here — already done in feature_detector.
        """
        cos_yaw = math.cos(self.global_yaw)
        sin_yaw = math.sin(self.global_yaw)

        for i, pt in enumerate(pts_base):
            if i in matched_indices:
                continue
            # Rotate base-frame point into global frame
            gx = pt[0] * cos_yaw - pt[1] * sin_yaw + self.global_x
            gy = pt[0] * sin_yaw + pt[1] * cos_yaw + self.global_y
            gz = pt[2]

            self.map_points_3d.append([gx, gy, gz])
            self.map_descriptors.append(descriptors[i])
            self.map_times_seen.append(1)
            self.map_times_expected.append(1)

    def prune_landmarks(self):
        if not self.map_points_3d:
            return

        good_pts, good_des, good_seen, good_exp = [], [], [], []
        cos_yaw = math.cos(-self.global_yaw)
        sin_yaw = math.sin(-self.global_yaw)

        for i in range(len(self.map_points_3d)):
            seen     = self.map_times_seen[i]
            expected = self.map_times_expected[i]
            ratio    = seen / float(expected) if expected > 0 else 0.0

            pt  = self.map_points_3d[i]
            dx  = pt[0] - self.global_x
            dy  = pt[1] - self.global_y
            # cam_z approximation: forward component in robot frame
            cam_z = dx * cos_yaw - dy * sin_yaw

            if cam_z < 0:
                continue
            if expected > 4 and ratio < 0.35:
                continue

            good_pts.append(self.map_points_3d[i])
            good_des.append(self.map_descriptors[i])
            good_seen.append(seen)
            good_exp.append(expected)

        self.map_points_3d      = good_pts
        self.map_descriptors    = good_des
        self.map_times_seen     = good_seen
        self.map_times_expected = good_exp


# ── Standalone functions ───────────────────────────────────────────────────

def get_visible_landmarks(map_points_3d, map_times_expected,
                          global_x, global_y, global_yaw,
                          base_to_kinect_matrix):
    vis_indices = []
    cos_yaw = math.cos(-global_yaw)
    sin_yaw = math.sin(-global_yaw)
    MAX_DIST = 4.0

    for i, pt in enumerate(map_points_3d):
        dx = pt[0] - global_x
        dy = pt[1] - global_y

        if dx*dx + dy*dy > MAX_DIST * MAX_DIST:
            continue

        # Local base frame
        base_x =  dx * cos_yaw - dy * sin_yaw
        base_y =  dx * sin_yaw + dy * cos_yaw
        base_z =  pt[2]

        pt_cam = base_to_kinect_matrix @ np.array([base_x, base_y, base_z, 1.0])
        cam_z  = pt_cam[2]

        if cam_z <= 0.1 or cam_z > 4.0:
            continue

        u = (pt_cam[0] * f / cam_z) + cx
        v = (pt_cam[1] * f / cam_z) + cy

        if 0 <= u < 640 and 0 <= v < 480:
            vis_indices.append(i)
            map_times_expected[i] += 1

    return vis_indices


def feature_detector(frame, depth_image, kinect_to_base_matrix):
    """
    FIX 1: Returns base-frame XY points in metres directly.
    Caller (add_new_landmarks) must NOT apply kinect_to_base again.
    """
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(frame, None)

    if not kp:
        return [], None, np.array([])

    pts_base, filt_kp, filt_des = [], [], []
    img = frame.copy()

    for i, keypoint in enumerate(kp):
        px, py = round(keypoint.pt[0]), round(keypoint.pt[1])

        if not (0 <= px < depth_image.shape[1] and 0 <= py < depth_image.shape[0]):
            continue

        raw_depth = depth_image[py, px]
        if raw_depth == 0 or np.isnan(raw_depth) or raw_depth < 50 or raw_depth > 5000:
            continue

        filt_kp.append(keypoint)
        filt_des.append(des[i])

        depth_m = float(raw_depth) / 1000.0
        X_c = depth_m * (px - cx) / f
        Y_c = depth_m * (py - cy) / f
        Z_c = depth_m

        # Transform to base frame (metres)
        pt_b = kinect_to_base_matrix @ np.array([X_c, Y_c, Z_c, 1.0])
        pts_base.append([pt_b[0], pt_b[1], pt_b[2]])

        cv2.putText(img, f"{depth_m:.1f}m", (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    img2 = cv2.drawKeypoints(img, filt_kp, None, color=(0, 255, 0), flags=0)
    cv2.imshow("FeatureDetector", img2)
    cv2.waitKey(1)

    return (filt_kp,
            np.array(filt_des) if filt_des else None,
            np.array(pts_base) if pts_base else np.array([]))


def kabsch_2d(P, Q):
    """
    FIX 3: Operates on Nx2 XY directly — no XZ slicing.
    P, Q already base-frame XY in metres.
    """
    P_mean, Q_mean = np.mean(P, axis=0), np.mean(Q, axis=0)
    Pc, Qc = P - P_mean, Q - Q_mean

    num   = np.sum(Qc[:, 0] * Pc[:, 1] - Qc[:, 1] * Pc[:, 0])
    den   = np.sum(Qc[:, 0] * Pc[:, 0] + Qc[:, 1] * Pc[:, 1])
    theta = math.atan2(num, den)

    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    t = P_mean - R @ Q_mean
    return R, t


def calculate_errors(P, Q, R, t):
    Q_transformed = (R @ Q.T).T + t
    return np.linalg.norm(P - Q_transformed, axis=1)


def ransac_kabsch(P, Q, iterations=200, threshold=0.05):
    """P, Q: Nx2 base XY metres. threshold: 0.05 m."""
    best_inliers, best_count = None, 0
    n = len(P)

    if n < MIN_KEYPOINTS:
        return None, None, P, Q, 0, 0

    for _ in range(iterations):
        idx      = random.sample(range(n), 3)
        R_hyp, t_hyp = kabsch_2d(P[idx], Q[idx])
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


def pointcloud(points, pc_pub, stamp, frame_id="orb_odom"):
    header = std_msgs.msg.Header()
    header.stamp, header.frame_id = stamp, frame_id
    pc_pub.publish(pcl2.create_cloud_xyz32(
        header, points.tolist() if isinstance(points, np.ndarray) else points))


def publish_odometry(node, current_time, global_x, global_y, global_yaw, inlier_count):
    q = quaternion_from_euler(0, 0, global_yaw)
    orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    odom_msg = Odometry()
    odom_msg.header.stamp    = current_time
    odom_msg.header.frame_id = 'orb_odom'
    odom_msg.child_frame_id  = 'base_link'
    odom_msg.pose.pose.position.x  = float(global_x)
    odom_msg.pose.pose.position.y  = float(global_y)
    odom_msg.pose.pose.position.z  = 0.0
    odom_msg.pose.pose.orientation = orientation

    cov_x = 0.5 if inlier_count < 25 else 0.05
    cov_y = 0.5 if inlier_count < 25 else 0.05
    cov_yaw = 0.2 if inlier_count < 25 else 0.02

    odom_msg.pose.covariance = [
        cov_x, 0.0,   0.0, 0.0, 0.0, 0.0,
        0.0,   cov_y, 0.0, 0.0, 0.0, 0.0,
        0.0,   0.0,   0.1, 0.0, 0.0, 0.0,
        0.0,   0.0,   0.0, 0.1, 0.0, 0.0,
        0.0,   0.0,   0.0, 0.0, 0.1, 0.0,
        0.0,   0.0,   0.0, 0.0, 0.0, cov_yaw
    ]
    node.odom_pub.publish(odom_msg)

    tf = TransformStamped()
    tf.header.stamp            = current_time
    tf.header.frame_id         = 'orb_odom'
    tf.child_frame_id          = 'base_link'
    tf.transform.translation.x = float(global_x)
    tf.transform.translation.y = float(global_y)
    tf.transform.translation.z = 0.0
    tf.transform.rotation      = orientation
    node.tf_broadcaster.sendTransform(tf)

    if not node.path_msg.poses:
        add = True
    else:
        last = node.path_msg.poses[-1].pose.position
        add  = math.hypot(global_x - last.x, global_y - last.y) > 0.1

    if add:
        ps = PoseStamped()
        ps.header.stamp     = current_time
        ps.header.frame_id  = 'orb_odom'
        ps.pose.position.x  = float(global_x)
        ps.pose.position.y  = float(global_y)
        ps.pose.position.z  = 0.0
        ps.pose.orientation = orientation
        node.path_msg.poses.append(ps)

    node.path_msg.header.stamp = current_time
    node.path_pub.publish(node.path_msg)


def main():
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()