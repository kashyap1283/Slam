#!/usr/bin/env python3
import math
import random

import cv2
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pcl2
import std_msgs.msg
from cv_bridge import CvBridge
from geometry_msgs.msg import Quaternion, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, PointCloud2
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from tf_transformations import (
    quaternion_from_euler,
    euler_from_quaternion,
    quaternion_matrix,
)
from visualization_msgs.msg import Marker, MarkerArray

from .ekf import ExtKalman


MIN_KEYPOINTS = 5
MIN_INLIERS = 8

cx = 318.525
cy = 241.181
f = 526.61


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()

        # ---------------- Frame names ----------------
        self.odom_frame = 'orb_odom'
        self.base_frame = 'base_link'
        self.camera_frame = 'kinect_depth'   # change this if your real TF frame is different

        # ---------------- Subscriptions ----------------
        self.subscription = self.create_subscription(
            Image, '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/serf01/nav_rgbd_1/depth/image_raw', self.depth_subscription, 10
        )
        self.wheel_sub = self.create_subscription(
            Odometry, '/serf01/odometry/wheel', self.wheel_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/serf01/odometry/imu', self.imu_callback, 10
        )

        # ---------------- Publishers ----------------
        self.pc_pub = self.create_publisher(PointCloud2, '/orb_pointcloud', 10)
        self.current_pc_pub = self.create_publisher(PointCloud2, '/global_cloud', 10)
        self.odom_pub = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)
        self.path_pub = self.create_publisher(Path, '/orb_path', 10)
        self.cov_pub = self.create_publisher(MarkerArray, '/landmark_covariances', 10)

        # ---------------- TF ----------------
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.kinect_to_base_matrix = None
        self.base_to_kinect_matrix = None
        self.static_tf_ready = False

        # ---------------- Path ----------------
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame

        # ---------------- Image state ----------------
        self.depth_image = None
        self.prev_kp = None
        self.prev_frame = None
        self.current_time = None

        # ---------------- Robot state ----------------
        self.global_x = 0.0
        self.global_y = 0.0
        self.global_yaw = 0.0

        self.initialized = False
        self.init_frame_count = 0
        self.INIT_FRAMES = 15

        # ---------------- Map ----------------
        self.map_points_3d = []
        self.map_descriptors = []
        self.map_times_seen = []
        self.map_times_expected = []
        self.map_missed_frames = []
        self.map_covariances = []
        self.MAX_MISSED_FRAMES = 3

        # ---------------- Odom accumulators ----------------
        self.prev_wheel_x = None
        self.prev_wheel_y = None
        self.prev_imu_yaw = None

        self.odom_forward_accum = 0.0
        self.odom_lateral_accum = 0.0
        self.imu_dtheta_accum = 0.0

        self.current_u = np.array([0.0, 0.0, 0.0])

        # ---------------- Noise params ----------------
        self.MAX_X_ERROR_RATE = 0.03
        self.MAX_Y_ERROR_RATE = 0.03
        self.MAX_YAW_ERROR_RATE = 0.05

        # ---------------- EKF ----------------
        self.ekf = ExtKalman(
            x=np.array([0.0, 0.0, 0.0]),
            state_func=self.ekf_state_func,
            meas_func=self.ekf_meas_func,
            JF=np.eye(3),
            JH=np.eye(3),
            R=np.diag([0.05, 0.05, 0.02]),
            Q=np.diag([0.002, 0.002, 0.001]),
        )

        self.LOW_INLIER_THRESHOLD = 20
        self.HIGH_INLIER_THRESHOLD = 35

    # ---------------------------------------------------------
    # TF helpers
    # ---------------------------------------------------------
    def try_get_camera_tf(self):
        if self.static_tf_ready:
            return True

        try:
            if not self.tf_buffer.can_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            ):
                self.get_logger().warn(
                    f"Waiting for TF: {self.base_frame} -> {self.camera_frame}"
                )
                return False

            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )

            quat = [
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            ]

            self.kinect_to_base_matrix = quaternion_matrix(quat)
            self.kinect_to_base_matrix[0, 3] = t.transform.translation.x
            self.kinect_to_base_matrix[1, 3] = t.transform.translation.y
            self.kinect_to_base_matrix[2, 3] = t.transform.translation.z
            self.base_to_kinect_matrix = np.linalg.inv(self.kinect_to_base_matrix)

            self.static_tf_ready = True
            self.get_logger().info(
                f"Got static TF: {self.base_frame} <- {self.camera_frame}"
            )
            return True

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return False

    # ---------------------------------------------------------
    # IMU & Wheel Callbacks
    # ---------------------------------------------------------
    def imu_callback(self, msg: Imu):
        quat = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]
        _, _, yaw = euler_from_quaternion(quat)
        yaw = wrap_angle(yaw)

        if self.prev_imu_yaw is not None:
            dyaw = wrap_angle(yaw - self.prev_imu_yaw)
            if abs(dyaw) < 0.5:
                self.imu_dtheta_accum += dyaw

        self.prev_imu_yaw = yaw

    def wheel_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.prev_wheel_x is not None:
            dx_global = x - self.prev_wheel_x
            dy_global = y - self.prev_wheel_y

            c = math.cos(self.global_yaw)
            s = math.sin(self.global_yaw)

            d_forward = c * dx_global + s * dy_global
            d_lateral = -s * dx_global + c * dy_global

            self.odom_forward_accum += d_forward
            self.odom_lateral_accum += d_lateral

        self.prev_wheel_x = x
        self.prev_wheel_y = y

    # ---------------------------------------------------------
    # EKF Functions
    # ---------------------------------------------------------
    def ekf_state_func(self, x):
        d_forward, d_lateral, dtheta = self.current_u
        theta = x[2]

        c = math.cos(theta)
        s = math.sin(theta)

        dx_global = d_forward * c - d_lateral * s
        dy_global = d_forward * s + d_lateral * c

        new_x = x[0] + dx_global
        new_y = x[1] + dy_global
        new_theta = wrap_angle(theta + dtheta)
        return np.array([new_x, new_y, new_theta])

    def ekf_meas_func(self, x):
        return x.copy()

    def _update_landmark_ekf(self, map_idx, cam_x, cam_y, cam_z):
        pt_cam_new = np.array([cam_x, cam_y, cam_z, 1.0])
        pt_base_new = self.kinect_to_base_matrix @ pt_cam_new
        x_b_new, y_b_new, z_b_new = pt_base_new[0], pt_base_new[1], pt_base_new[2]

        x_global_new = (
            x_b_new * math.cos(self.global_yaw)
            - y_b_new * math.sin(self.global_yaw)
            + self.global_x
        )
        y_global_new = (
            x_b_new * math.sin(self.global_yaw)
            + y_b_new * math.cos(self.global_yaw)
            + self.global_y
        )
        z_meas = np.array([x_global_new, y_global_new])

        R_new = self._compute_feature_covariance(cam_x, cam_z, self.global_yaw)

        pt_old = self.map_points_3d[map_idx]
        x_old_2d = np.array([pt_old[0], pt_old[1]])
        P_old = self.map_covariances[map_idx]

        try:
            S_inv = np.linalg.inv(P_old + R_new)
            K = P_old @ S_inv

            x_updated = x_old_2d + K @ (z_meas - x_old_2d)
            P_updated = (np.eye(2) - K) @ P_old

            self.map_points_3d[map_idx][0] = float(x_updated[0])
            self.map_points_3d[map_idx][1] = float(x_updated[1])
            self.map_points_3d[map_idx][2] = float((pt_old[2] + z_b_new) / 2.0)
            self.map_covariances[map_idx] = P_updated
        except np.linalg.LinAlgError:
            pass

    # ---------------------------------------------------------
    # Dynamic Covariance & Sensor Modeling
    # ---------------------------------------------------------
    def _compute_dynamic_q(self, d):
        if d < 0.001:
            return np.diag([1e-5, 1e-5, 1e-6])

        max_err_x = self.MAX_X_ERROR_RATE * d
        max_err_y = self.MAX_Y_ERROR_RATE * d
        max_err_yaw = self.MAX_YAW_ERROR_RATE * d

        sigma_x = max_err_x / 3.0
        sigma_y = max_err_y / 3.0
        sigma_yaw = max_err_yaw / 3.0

        return np.diag([sigma_x**2, sigma_y**2, sigma_yaw**2])

    def _compute_feature_covariance(self, cam_x, cam_z, global_yaw):
        d = math.sqrt(cam_x**2 + cam_z**2)
        if d < 0.001:
            return np.diag([1e-5, 1e-5])

        theta_cam = math.atan2(cam_x, cam_z)

        sigma_d = (d * 0.01) / 3.0
        sigma_a = (d * math.sin(math.radians(3.0))) / 3.0

        S = np.array([
            [sigma_a, 0.0],
            [0.0, sigma_d]
        ])

        global_theta = wrap_angle(global_yaw + theta_cam)
        R_rot = np.array([
            [math.cos(global_theta), -math.sin(global_theta)],
            [math.sin(global_theta), math.cos(global_theta)],
        ])

        S_squared = S @ S
        cov_matrix = R_rot @ S_squared @ R_rot.T
        return cov_matrix

    def _update_filter_noise(self, state, driven_distance):
        dynamic_Q = self._compute_dynamic_q(driven_distance)

        if state == 'low':
            self.ekf.setR(np.diag([50.0, 50.0, 10.0]))
            self.ekf.setQ(dynamic_Q * 2.0)
        elif state == 'medium':
            self.ekf.setR(np.diag([0.1, 0.1, 0.05]))
            self.ekf.setQ(dynamic_Q)
        elif state == 'high':
            self.ekf.setR(np.diag([0.02, 0.02, 0.01]))
            self.ekf.setQ(dynamic_Q)

    def _commit_predict(self, driven_distance):
        d_forward, d_lateral, dtheta = self.current_u
        theta = float(self.ekf.x[2])

        c = math.cos(theta)
        s = math.sin(theta)

        JF = np.array([
            [c, -s, -d_forward * s - d_lateral * c],
            [s,  c,  d_forward * c - d_lateral * s],
            [0.0, 0.0, 1.0],
        ])

        self.ekf.setJF(JF)
        self.ekf.setJH(np.eye(3))

        dynamic_Q = self._compute_dynamic_q(driven_distance)
        self.ekf.setQ(dynamic_Q)

        x_pred, P_pred = self.ekf.predictState()
        self.ekf.x = x_pred.flatten()
        self.ekf.P = P_pred
        self.ekf.x[2] = wrap_angle(self.ekf.x[2])

    def _publish_current_state(self, inlier_count, vis_indices=None):
        if vis_indices is None:
            vis_indices = []

        self.global_x = float(self.ekf.x[0])
        self.global_y = float(self.ekf.x[1])
        self.global_yaw = wrap_angle(float(self.ekf.x[2]))

        if len(self.map_points_3d) > 0:
            pointcloud(
                np.array(self.map_points_3d),
                self.current_pc_pub,
                self.current_time,
                self.odom_frame,
            )

        publish_odometry(
            self,
            self.current_time,
            self.global_x,
            self.global_y,
            self.global_yaw,
            inlier_count,
        )

        self._publish_covariances(vis_indices)

    def _publish_covariances(self, vis_indices):
        if not self.map_points_3d or not self.map_covariances:
            return

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i in vis_indices:
            if i >= len(self.map_points_3d):
                continue

            pt = self.map_points_3d[i]
            cov = self.map_covariances[i]
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])

            m = Marker()
            m.header.frame_id = self.odom_frame
            m.header.stamp = self.current_time
            m.ns = "covariances"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            m.pose.position.x = float(pt[0])
            m.pose.position.y = float(pt[1])
            m.pose.position.z = float(pt[2])

            q = quaternion_from_euler(0, 0, angle)
            m.pose.orientation.x = float(q[0])
            m.pose.orientation.y = float(q[1])
            m.pose.orientation.z = float(q[2])
            m.pose.orientation.w = float(q[3])

            m.scale.x = 6.0 * math.sqrt(abs(eigenvalues[0]))
            m.scale.y = 6.0 * math.sqrt(abs(eigenvalues[1]))
            m.scale.z = 0.05

            m.color.a = 0.4
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 0.0

            marker_array.markers.append(m)

        self.cov_pub.publish(marker_array)

    # ---------------------------------------------------------
    # Sensor callbacks
    # ---------------------------------------------------------
    def depth_subscription(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def listener_callback(self, msg):
        self.current_time = msg.header.stamp

        if not self.try_get_camera_tf():
            return

        if self.depth_image is None:
            return

        u_forward = self.odom_forward_accum
        u_lateral = self.odom_lateral_accum
        u_dtheta = self.imu_dtheta_accum

        self.odom_forward_accum = 0.0
        self.odom_lateral_accum = 0.0
        self.imu_dtheta_accum = 0.0

        self.current_u = np.array([u_forward, u_lateral, u_dtheta])
        driven_distance = math.sqrt(u_forward**2 + u_lateral**2)

        rotation_magnitude = abs(u_dtheta)
        is_high_rotation = rotation_magnitude > 0.3

        strafing_laterally = abs(u_lateral) > 0.01
        moving_backward = u_forward < -0.01
        is_high_slip = strafing_laterally or moving_backward

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        kp, des, points = feature_detector(frame, self.depth_image)

        self._commit_predict(driven_distance)

        if not kp or len(kp) < MIN_KEYPOINTS:
            self._update_filter_noise('low', driven_distance)
            self._publish_current_state(0)
            self.prune_landmarks()
            self.prev_kp = kp
            self.prev_frame = frame
            return

        if not self.initialized:
            self.global_x = float(self.ekf.x[0])
            self.global_y = float(self.ekf.x[1])
            self.global_yaw = wrap_angle(float(self.ekf.x[2]))

            self.add_new_landmarks(points, des, set())
            self.init_frame_count += 1

            self._publish_current_state(0)
            self.prune_landmarks()

            if self.init_frame_count >= self.INIT_FRAMES:
                self.initialized = True

            self.prev_kp = kp
            self.prev_frame = frame
            return

        if self.prev_kp is None:
            self._publish_current_state(0)
            self.prune_landmarks()
            self.prev_kp = kp
            self.prev_frame = frame
            return

        inlier_count = 0
        raw_match_count = 0
        vis_indices = []

        try:
            self.global_x = float(self.ekf.x[0])
            self.global_y = float(self.ekf.x[1])
            self.global_yaw = wrap_angle(float(self.ekf.x[2]))

            vis_indices = get_visible_landmarks(
                self.map_points_3d,
                self.global_x,
                self.global_y,
                self.global_yaw,
                self.base_to_kinect_matrix,
            )

            P = []
            Q_mat = []
            matched_current_indices = set()
            matched_map_indices = set()

            if len(vis_indices) > 0 and des is not None and len(self.map_descriptors) > 0:
                vis_des_array = np.array([self.map_descriptors[i] for i in vis_indices])

                if len(vis_des_array) > 0:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    map_matches = bf.match(vis_des_array, des)
                    raw_match_count = len(map_matches)

                    cos_yaw = math.cos(-self.global_yaw)
                    sin_yaw = math.sin(-self.global_yaw)

                    for m in map_matches:
                        if m.distance >= 60:
                            continue

                        map_idx = vis_indices[m.queryIdx]
                        pt = self.map_points_3d[map_idx]

                        dx = pt[0] - self.global_x
                        dy = pt[1] - self.global_y
                        dz = pt[2]

                        base_x = dx * cos_yaw - dy * sin_yaw
                        base_y = dx * sin_yaw + dy * cos_yaw
                        base_z = dz

                        pt_base = np.array([base_x, base_y, base_z, 1.0])
                        pt_cam = self.base_to_kinect_matrix @ pt_base
                        cam_x, cam_y, cam_z = pt_cam[0], pt_cam[1], pt_cam[2]

                        if cam_z <= 0.1:
                            continue

                        observed_depth = points[m.trainIdx][2]

                        if is_high_rotation:
                            depth_tolerance = 3.0
                        elif is_high_slip:
                            depth_tolerance = 2.0
                        else:
                            depth_tolerance = 1.5

                        depth_ratio = abs(cam_z - observed_depth) / max(cam_z, observed_depth)
                        if depth_ratio > depth_tolerance:
                            continue

                        P.append([cam_x, cam_y, cam_z])
                        Q_mat.append(points[m.trainIdx])

                        matched_current_indices.add(m.trainIdx)
                        matched_map_indices.add(map_idx)

                        x_cam_new, y_cam_new, z_cam_new = points[m.trainIdx]
                        self._update_landmark_ekf(map_idx, x_cam_new, y_cam_new, z_cam_new)

            for map_idx in vis_indices:
                self.map_times_expected[map_idx] += 1
                if map_idx in matched_map_indices:
                    self.map_times_seen[map_idx] += 1
                    self.map_missed_frames[map_idx] = 0
                else:
                    self.map_missed_frames[map_idx] += 1

            P = np.array(P)
            Q_mat = np.array(Q_mat)

            R_2d = None
            t_2d = None

            if len(P) >= MIN_KEYPOINTS:
                n_matches = len(P)

                if is_high_rotation:
                    ransac_thresh = 0.40 if n_matches < 10 else (0.30 if n_matches < 15 else 0.15)
                elif is_high_slip:
                    ransac_thresh = 0.30 if n_matches < 10 else (0.20 if n_matches < 15 else 0.12)
                else:
                    ransac_thresh = 0.25 if n_matches < 10 else (0.15 if n_matches < 15 else 0.08)

                R_2d, t_2d, P_in, Q_in, inlier_count, outlier_count = ransac_kabsch(
                    P, Q_mat, threshold=ransac_thresh
                )

            if R_2d is None or inlier_count < MIN_INLIERS:
                self._update_filter_noise('low', driven_distance)

                if is_high_rotation and inlier_count < 3:
                    theta = float(self.ekf.x[2])
                    c = math.cos(theta)
                    s = math.sin(theta)

                    dx_global = u_forward * c - u_lateral * s
                    dy_global = u_forward * s + u_lateral * c

                    z_x = self.ekf.x[0] + dx_global
                    z_y = self.ekf.x[1] + dy_global
                    z_yaw = wrap_angle(self.ekf.x[2] + u_dtheta)

                    z_pose = np.array([z_x, z_y, z_yaw])
                    self.ekf.update(z_pose)
                    self.ekf.x[2] = wrap_angle(self.ekf.x[2])
            else:
                if inlier_count < self.LOW_INLIER_THRESHOLD:
                    self._update_filter_noise('low', driven_distance)
                elif inlier_count < self.HIGH_INLIER_THRESHOLD:
                    self._update_filter_noise('medium', driven_distance)
                else:
                    self._update_filter_noise('high', driven_distance)

                vo_dtheta = wrap_angle(math.atan2(R_2d[1, 0], R_2d[0, 0]))

                alpha_xy = 0.10 if inlier_count < 25 else 0.25
                alpha_yaw = 0.55 if inlier_count < 25 else 0.80

                if is_high_rotation:
                    alpha_xy *= 0.3
                elif is_high_slip:
                    alpha_xy *= 0.5

                theta_current = wrap_angle(float(self.ekf.x[2]))
                cos_t = math.cos(theta_current)
                sin_t = math.sin(theta_current)

                vo_forward = t_2d[1]
                vo_lateral = -t_2d[0]

                dx_vo = vo_forward * cos_t - vo_lateral * sin_t
                dy_vo = vo_forward * sin_t + vo_lateral * cos_t

                z_x = self.ekf.x[0] + alpha_xy * dx_vo
                z_y = self.ekf.x[1] + alpha_xy * dy_vo
                z_yaw = wrap_angle(theta_current + alpha_yaw * vo_dtheta)

                z_pose = np.array([z_x, z_y, z_yaw])
                self.ekf.update(z_pose)
                self.ekf.x[2] = wrap_angle(self.ekf.x[2])

            self.global_x = float(self.ekf.x[0])
            self.global_y = float(self.ekf.x[1])
            self.global_yaw = wrap_angle(float(self.ekf.x[2]))

            if inlier_count >= MIN_INLIERS:
                self.add_new_landmarks(points, des, matched_current_indices)
            elif len(points) >= MIN_KEYPOINTS:
                self.get_logger().info("Unmapped territory detected (0 inliers). Seeding new map features.")
                self.add_new_landmarks(points, des, set())

            self._publish_current_state(inlier_count, vis_indices)
            self.prune_landmarks()

            self.get_logger().info(
                f"Pose: X:{self.global_x:.2f} Y:{self.global_y:.2f} "
                f"Yaw:{math.degrees(self.global_yaw):.1f}° "
                f"Inliers:{inlier_count} MapPts:{len(self.map_points_3d)} "
                f"vis:{len(vis_indices)} raw:{raw_match_count} kept:{len(P)} "
                f"ufwd:{u_forward:.3f} ulat:{u_lateral:.3f} "
                f"d:{driven_distance:.3f}m"
            )

        finally:
            self.prev_kp = kp
            self.prev_frame = frame

    # ---------------------------------------------------------
    # Map Management
    # ---------------------------------------------------------
    def add_new_landmarks(self, current_points, current_descriptors, matched_indices):
        if current_descriptors is None or len(current_points) == 0:
            return 0, self.map_points_3d

        cos_yaw = math.cos(self.global_yaw)
        sin_yaw = math.sin(self.global_yaw)
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

            point_covariance = self._compute_feature_covariance(x_cam, z_cam, self.global_yaw)
            self.map_covariances.append(point_covariance)

            self.map_points_3d.append([x_global, y_global, z_global])
            self.map_descriptors.append(current_descriptors[i])
            self.map_times_seen.append(1)
            self.map_times_expected.append(1)
            self.map_missed_frames.append(0)
            new_count += 1

        return new_count, self.map_points_3d

    def prune_landmarks(self):
        if not self.map_points_3d:
            return 0

        good_points = []
        good_descriptors = []
        good_seen = []
        good_expected = []
        good_missed = []
        good_covariances = []
        removed_count = 0

        for i in range(len(self.map_points_3d)):
            times_seen = self.map_times_seen[i]
            missed = self.map_missed_frames[i]
            expected = self.map_times_expected[i]

            if missed > self.MAX_MISSED_FRAMES:
                removed_count += 1
                continue

            if expected > 3 and times_seen < 3:
                removed_count += 1
                continue

            good_points.append(self.map_points_3d[i])
            good_descriptors.append(self.map_descriptors[i])
            good_seen.append(self.map_times_seen[i])
            good_expected.append(self.map_times_expected[i])
            good_missed.append(self.map_missed_frames[i])
            good_covariances.append(self.map_covariances[i])

        self.map_points_3d = good_points
        self.map_descriptors = good_descriptors
        self.map_times_seen = good_seen
        self.map_times_expected = good_expected
        self.map_missed_frames = good_missed
        self.map_covariances = good_covariances

        return removed_count


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------
def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def get_visible_landmarks(map_points_3d, global_x, global_y, global_yaw, base_to_kinect_matrix):
    vis_indices = []
    cos_yaw = math.cos(-global_yaw)
    sin_yaw = math.sin(-global_yaw)

    MAX_LANDMARK_DISTANCE = 6.0

    for i, pt in enumerate(map_points_3d):
        dx = pt[0] - global_x
        dy = pt[1] - global_y

        if (dx * dx + dy * dy) > (MAX_LANDMARK_DISTANCE * MAX_LANDMARK_DISTANCE):
            continue

        dz = pt[2]

        base_x = dx * cos_yaw - dy * sin_yaw
        base_y = dx * sin_yaw + dy * cos_yaw
        base_z = dz

        pt_base = np.array([base_x, base_y, base_z, 1.0])
        pt_cam = base_to_kinect_matrix @ pt_base
        cam_x, cam_y, cam_z = pt_cam[0], pt_cam[1], pt_cam[2]

        if cam_z <= 0.1 or cam_z > 6.0:
            continue

        u = (cam_x * f / cam_z) + cx
        v = (cam_y * f / cam_z) + cy

        if 0 <= u < 640 and 0 <= v < 480:
            vis_indices.append(i)

    return vis_indices


def feature_detector(frame, depth_image):
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(frame, None)

    if not kp:
        return [], None, np.array([])

    points = []
    filtered_kp = []
    filtered_des = []

    img = frame.copy()

    for i, keypoint in enumerate(kp):
        x, y = round(keypoint.pt[0]), round(keypoint.pt[1])

        if not (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]):
            continue

        distance = depth_image[y, x]
        if distance == 0 or np.isnan(distance) or distance < 50 or distance > 5000:
            continue

        filtered_kp.append(keypoint)
        filtered_des.append(des[i])

        depth = distance / 1000.0
        X = depth * (x - cx) / f
        Y = depth * (y - cy) / f
        points.append([X, Y, depth])

        cv2.putText(
            img,
            f"{depth:.1f} m",
            [x, y],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

    img = cv2.drawKeypoints(img, filtered_kp, None, color=(0, 255, 0), flags=0)
    cv2.imshow("FeatureDetector", img)
    cv2.waitKey(1)

    return filtered_kp, np.array(filtered_des), np.array(points)


def pointcloud(points, pc_pub, stamp, frame_id="orb_odom"):
    header = std_msgs.msg.Header()
    header.stamp = stamp
    header.frame_id = frame_id

    points_list = points.tolist() if isinstance(points, np.ndarray) else points
    cloud_msg = pcl2.create_cloud_xyz32(header, points_list)
    pc_pub.publish(cloud_msg)


def publish_odometry(node, current_time, global_x, global_y, global_yaw, inlier_count):
    odom_msg = Odometry()
    odom_msg.header.stamp = current_time
    odom_msg.header.frame_id = node.odom_frame
    odom_msg.child_frame_id = node.base_frame

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
        cov_x, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, cov_y, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.1, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, cov_yaw,
    ]

    node.odom_pub.publish(odom_msg)

    t_msg = TransformStamped()
    t_msg.header.stamp = current_time
    t_msg.header.frame_id = node.odom_frame
    t_msg.child_frame_id = node.base_frame
    t_msg.transform.translation.x = float(global_x)
    t_msg.transform.translation.y = float(global_y)
    t_msg.transform.translation.z = 0.0
    t_msg.transform.rotation = odom_msg.pose.pose.orientation
    node.tf_broadcaster.sendTransform(t_msg)

    if len(node.path_msg.poses) == 0:
        pose = PoseStamped()
        pose.header.stamp = current_time
        pose.header.frame_id = node.odom_frame
        pose.pose.position.x = float(global_x)
        pose.pose.position.y = float(global_y)
        pose.pose.position.z = 0.0
        pose.pose.orientation = odom_msg.pose.pose.orientation
        node.path_msg.poses.append(pose)
    else:
        last_pose = node.path_msg.poses[-1]
        dist = math.sqrt(
            (global_x - last_pose.pose.position.x) ** 2
            + (global_y - last_pose.pose.position.y) ** 2
        )
        if dist > 0.1:
            pose = PoseStamped()
            pose.header.stamp = current_time
            pose.header.frame_id = node.odom_frame
            pose.pose.position.x = float(global_x)
            pose.pose.position.y = float(global_y)
            pose.pose.position.z = 0.0
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

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    t = P_mean - (R @ Q_mean)
    return R, t


def calculate_errors(P, Q, R, t):
    P_2d, Q_2d = P[:, [0, 2]], Q[:, [0, 2]]
    Q_transformed = (R @ Q_2d.T).T + t
    return np.linalg.norm(P_2d - Q_transformed, axis=1)


def ransac_kabsch(P, Q, iterations=200, threshold=0.05):
    best_inliers = None
    best_count = 0
    n = len(P)

    if n < MIN_KEYPOINTS:
        return None, None, P, Q, 0, 0

    if n < 8:
        R_hyp, t_hyp = kabsch_2d(P, Q)
        errors = calculate_errors(P, Q, R_hyp, t_hyp)
        inlier_count = np.sum(errors < threshold)
        if inlier_count >= 3:
            return R_hyp, t_hyp, P, Q, int(inlier_count), n - int(inlier_count)
        inlier_count = np.sum(errors < threshold * 2)
        if inlier_count >= 3:
            return R_hyp, t_hyp, P, Q, int(inlier_count), n - int(inlier_count)
        return None, None, P, Q, 0, 0

    for _ in range(iterations):
        idx = random.sample(range(n), 3)
        P_s, Q_s = P[idx], Q[idx]

        R_hyp, t_hyp = kabsch_2d(P_s, Q_s)
        errors = calculate_errors(P, Q, R_hyp, t_hyp)
        inliers = errors < threshold
        inlier_count = np.sum(inliers)

        if inlier_count > best_count and inlier_count >= 3:
            best_count = inlier_count
            best_inliers = inliers

    if best_inliers is None or np.sum(best_inliers) < 3:
        return None, None, P, Q, 0, 0

    P_in, Q_in = P[best_inliers], Q[best_inliers]
    R, t = kabsch_2d(P_in, Q_in)

    return R, t, P_in, Q_in, int(np.sum(best_inliers)), n - int(np.sum(best_inliers))


def main():
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()