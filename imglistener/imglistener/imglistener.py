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

from geometry_msgs.msg import Quaternion, TransformStamped, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, PointCloud2, Imu
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from tf_transformations import quaternion_from_euler, quaternion_matrix
from std_msgs.msg import ColorRGBA

from .particlefilter import PF
import cProfile


class FastSlamNode(Node):
    def __init__(self):
        super().__init__('fastslam_node')
        self.bridge = CvBridge()

        # Parameter List
        self.declare_parameters(
            namespace='',
            parameters=[
                ('min_keypoints', 5),
                ('min_inliers', 6),
                ('visual_inlier_threshold', 20),
                ('camera_cx', 318.525),
                ('camera_cy', 241.181),
                ('camera_f', 526.61),
                ('odom_frame', 'orb_odom'),
                ('base_frame', 'base_link'),
                ('camera_frame', 'kinect_depth'),
                ('max_missed_frames', 3),
                ('num_particles', 400),
                ('max_map_landmarks', 1000),
                ('min_depth_mm', 50),
                ('max_depth_mm', 5000),
                ('resample_threshold', 0.50),
            ]
        )

        self.min_keypoints         = self.get_parameter('min_keypoints').value
        self.min_inliers           = self.get_parameter('min_inliers').value
        self.visual_inlier_threshold = self.get_parameter('visual_inlier_threshold').value
        self.cx                    = self.get_parameter('camera_cx').value
        self.cy                    = self.get_parameter('camera_cy').value
        self.focal_length          = self.get_parameter('camera_f').value
        self.odom_frame            = self.get_parameter('odom_frame').value
        self.base_frame            = self.get_parameter('base_frame').value
        self.camera_frame          = self.get_parameter('camera_frame').value
        self.max_missed_frames     = self.get_parameter('max_missed_frames').value
        self.num_particles         = self.get_parameter('num_particles').value
        self.max_map_landmarks     = self.get_parameter('max_map_landmarks').value
        self.min_depth_mm          = self.get_parameter('min_depth_mm').value
        self.max_depth_mm          = self.get_parameter('max_depth_mm').value
        self.resample_threshold    = self.get_parameter('resample_threshold').value

        # subscriptions
        self.subscription    = self.create_subscription(Image,    '/serf01/nav_rgbd_1/rgb/image_raw',   self.listener_callback, 10)
        self.depth_sub       = self.create_subscription(Image,    '/serf01/nav_rgbd_1/depth/image_raw', self.depth_subscription, 10)
        self.wheel_odom_sub  = self.create_subscription(Odometry, '/serf01/odometry/wheel',             self.wheel_callback, 10)
        self.imu_sub         = self.create_subscription(Imu,      '/serf01/odometry/imu',               self.imu_callback, 10)

        # publishers
        self.current_pc_pub        = self.create_publisher(PointCloud2,  '/global_cloud',                      10)
        self.odom_pub              = self.create_publisher(Odometry,     '/serf01/odometry/project_slam',       10)
        self.path_pub              = self.create_publisher(Path,         '/orb_path',                           10)
        self.pf_particles_marker_pub = self.create_publisher(MarkerArray, '/pf_particles_marker',              10)

        # TF setup - we need the static kinect->base transform before we can do anything
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer      = Buffer()
        self.tf_listener    = TransformListener(self.tf_buffer, self, spin_thread=False)

        # these get filled in once by try_get_camera_tf()
        self.kinect_to_base_R   = None
        self.kinect_to_base_t   = None
        self.base_to_kinect_R   = None
        self.base_to_kinect_R_T = None
        self.base_to_kinect_t   = None
        self.static_tf_ready    = False

        # general state
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame
        self.depth_image   = None
        self.current_time  = None
        self.prev_time_sec = None
        self.frame_idx     = 0

        # latest velocity readings from wheel and IMU - updated asynchronously
        self.latest_v_x = 0.0
        self.latest_v_y = 0.0
        self.latest_w   = 0.0

        self.prev_des         = None
        self.prev_points_base = None

        # flag for which sensor is currently being used (just for debugging)
        self.use_camera = True

        # main SLAM objects
        self.pf  = PF(num_particles=self.num_particles, x=0.0, y=0.0, yaw=0.0)
        self.orb = cv2.ORB_create(nfeatures=2000, fastThreshold=15)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def wheel_callback(self, msg):
        # just cache the latest velocity - used in listener_callback to compute motion delta
        self.latest_v_x = msg.twist.twist.linear.x
        self.latest_v_y = msg.twist.twist.linear.y

    def imu_callback(self, msg):
        self.latest_w = msg.angular_velocity.z

    def try_get_camera_tf(self):
        # only need to do this once - the kinect is rigidly mounted
        if self.static_tf_ready:
            return True
        try:
            if not self.tf_buffer.can_transform(self.base_frame, self.camera_frame,
                                                 rclpy.time.Time(), timeout=Duration(seconds=0.2)):
                return False

            t = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame,
                                                 rclpy.time.Time(), timeout=Duration(seconds=0.2))
            quat = [t.transform.rotation.x, t.transform.rotation.y,
                    t.transform.rotation.z, t.transform.rotation.w]

            kinect_to_base_matrix = quaternion_matrix(quat)
            kinect_to_base_matrix[0, 3] = t.transform.translation.x
            kinect_to_base_matrix[1, 3] = t.transform.translation.y
            kinect_to_base_matrix[2, 3] = t.transform.translation.z

            base_to_kinect_matrix = np.linalg.inv(kinect_to_base_matrix)

            self.kinect_to_base_R   = kinect_to_base_matrix[:3, :3]
            self.kinect_to_base_t   = kinect_to_base_matrix[:3, 3]
            self.base_to_kinect_R   = base_to_kinect_matrix[:3, :3]
            self.base_to_kinect_R_T = self.base_to_kinect_R.T
            self.base_to_kinect_t   = base_to_kinect_matrix[:3, 3]

            self.static_tf_ready = True
            return True
        except Exception:
            return False

    def depth_subscription(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def feature_detector(self, frame, depth_image):
        kp, des = self.orb.detectAndCompute(frame, None)
        if not kp:
            return [], None, np.array([])

        points, filtered_kp, filtered_des = [], [], []

        for i, keypoint in enumerate(kp):
            x, y = round(keypoint.pt[0]), round(keypoint.pt[1])

            # skip if keypoint falls outside the depth image (can happen near edges)
            if not (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]):
                continue

            distance = depth_image[y, x]
            if distance == 0 or np.isnan(distance) or distance < self.min_depth_mm or distance > self.max_depth_mm:
                continue

            filtered_kp.append(keypoint)
            filtered_des.append(des[i])

            # back-project pixel to 3D using pinhole model: X = Z*(u-cx)/f
            depth = distance / 1000.0
            points.append([
                depth * (x - self.cx) / self.focal_length,
                depth * (y - self.cy) / self.focal_length,
                depth
            ])

        # debug window - can comment this out if it's too slow
        img2 = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
        cv2.imshow("ORB Feature Detector", img2)
        cv2.waitKey(1)

        return filtered_kp, np.array(filtered_des), np.array(points)

    def _compute_feature_covariance(self, cam_x, cam_z, global_yaw):
        d = math.sqrt(cam_x**2 + cam_z**2)
        if d < 0.1:
            return 1e-5, 0.0, 0.0, 1e-5

        sigma_d = 0.001477 + 0.002294 * (d - 0.4)**2   # range noise
        sigma_a = d * 0.0005817764                       # angular noise scales with distance

        sa, sd = sigma_a**2, sigma_d**2
        alpha = math.atan2(cam_x, cam_z)
        total_angle = global_yaw + alpha

        c = math.cos(total_angle)
        s = math.sin(total_angle)
        c2, s2, cs = c*c, s*s, c*s

        # rotate the diagonal (sigma_d, sigma_a) covariance into the global frame
        cov00 = sa * c2 + sd * s2
        cov01 = (sa - sd) * cs
        cov11 = sa * s2 + sd * c2

        return cov00, cov01, cov01, cov11

    def listener_callback(self, msg):
        # this is the main per-frame SLAM loop, triggered by each RGB image
        self.current_time = msg.header.stamp
        current_time_sec  = self.current_time.sec + (self.current_time.nanosec * 1e-9)

        # wait until we have both the camera TF and a depth image
        if not self.try_get_camera_tf() or self.depth_image is None:
            return

        # first frame - just save features and return, nothing to match against yet
        if self.prev_time_sec is None:
            self.prev_time_sec = current_time_sec
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            kp, des, points_cam = self.feature_detector(frame, self.depth_image)
            self.prev_des = des
            if len(points_cam) > 0:
                self.prev_points_base = points_cam @ self.kinect_to_base_R.T + self.kinect_to_base_t
            return

        dt = current_time_sec - self.prev_time_sec
        self.prev_time_sec = current_time_sec

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        kp, des, points_cam = self.feature_detector(frame, self.depth_image)

        if not kp or len(kp) < self.min_keypoints:
            return

        # transform current keypoints from camera frame to base_link frame
        if len(points_cam) > 0:
            points_base = points_cam @ self.kinect_to_base_R.T + self.kinect_to_base_t
        else:
            points_base = np.empty((0, 3), dtype=np.float64)

        self.frame_idx += 1
        matched_P_prev, matched_P_curr = None, None

        # match current frame against previous frame
        if self.prev_des is not None and des is not None:
            matches = self.bf.match(self.prev_des, des)
            matches = [m for m in matches if m.distance < 75]
            matches.sort(key=lambda m: m.distance)

            if len(matches) >= self.min_keypoints:
                matched_P_prev = np.asarray([self.prev_points_base[m.queryIdx] for m in matches], dtype=np.float64)
                matched_P_curr = np.asarray([points_base[m.trainIdx]           for m in matches], dtype=np.float64)

        # estimate motion from visual tracking
        vis_d_fwd, vis_d_lat, vis_d_yaw, inlier_count = self._estimate_frame_motion(
            matched_P_prev, matched_P_curr, particle=None)

        # wheel + IMU fallback (simple v*dt integration)
        odom_d_fwd = self.latest_v_x * dt
        odom_d_lat = self.latest_v_y * dt
        odom_d_yaw = self.latest_w   * dt

        # pick the motion source - camera if we have enough inliers, else wheel+IMU
        if inlier_count >= self.visual_inlier_threshold:
            d_fwd, d_lat, d_yaw = vis_d_fwd, vis_d_lat, vis_d_yaw
            perform_visual_update = True
            self.use_camera = True
        else:
            d_fwd, d_lat, d_yaw = odom_d_fwd, odom_d_lat, odom_d_yaw
            perform_visual_update = False
            self.use_camera = False

        # skip near-zero motion to avoid spurious updates
        moved_enough = (abs(d_fwd) > 0.01 or abs(d_lat) > 0.01 or abs(d_yaw) > 0.01)

        # noise sigmas - camera motion is noisier in direction, wheel is noisier in yaw
        if moved_enough:
            if perform_visual_update:
                sigmas = {
                    'forward': 0.01 * abs(d_fwd) + 0.008,
                    'lateral': 0.01 * abs(d_lat) + 0.008,
                    'yaw':     0.01 * abs(d_yaw) + 0.008,
                }
            else:
                sigmas = {
                    'forward': 0.005 * abs(d_fwd) + 0.006,
                    'lateral': 0.005 * abs(d_lat) + 0.006,
                    'yaw':     0.005 * abs(d_yaw) + 0.003,
                }
        else:
            sigmas = {'forward': 0.0, 'lateral': 0.0, 'yaw': 0.0}

        # main particle loop - predict + update every particle
        for particle in self.pf.particles:
            particle.predict_motion(d_fwd, d_lat, d_yaw, sigmas)

            if perform_visual_update and moved_enough:
                matched_curr, matched_map, innovations, s_matrices = self._process_particle_data_association(particle, des, points_base, points_cam)
                self._evaluate_particle_weights(particle, innovations, s_matrices)
                self._extend_particle_landmark_map(particle, points_cam, des, matched_curr, points_base)

        self.prev_des         = des
        self.prev_points_base = points_base

        # normalize, check if we need to resample, then publish
        if perform_visual_update and moved_enough:
            self.pf.normalize_weights()
            best_particle = self.pf.get_best_particle()

            if self.pf.effective_sample_size() < self.num_particles * self.resample_threshold:
                self.pf.systematic_resampling()
                self.get_logger().info("Resampling")
        else:
            best_particle = self.pf.get_best_particle()

        publish_odometry(self, self.current_time, best_particle.x, best_particle.y, best_particle.yaw)
        publish_pf_particles_markers(self, self.current_time)

        if best_particle.num_landmarks > 0:
            # slice dynamically shrinks after prune_map runs
            best_map_pts = best_particle.map_pts[:best_particle.num_landmarks]
            pointcloud(best_map_pts, self.current_pc_pub, self.current_time, self.odom_frame)

    def _estimate_frame_motion(self, matched_P_prev, matched_P_curr, particle=None):
        d_fwd, d_lat, d_yaw = 0.0, 0.0, 0.0
        inlier_count = 0

        if matched_P_prev is None or len(matched_P_prev) < self.min_keypoints:
            return d_fwd, d_lat, d_yaw, inlier_count

        if particle is None:
            # global frame RANSAC for sensor selection
            R_2d, t_2d, _, _, inlier_count_out, _ = ransac_kabsch(
                matched_P_prev, matched_P_curr,
                iterations=100, strict_thresh=0.02, relaxed_thresh=0.05,
                min_kpts=self.min_keypoints, min_inls=self.min_inliers
            )
        else:
            # particle-local RANSAC - tighter threshold since we already have a pose hypothesis
            P_prev_local = self._to_particle_local(matched_P_prev, particle.x, particle.y, particle.yaw)
            P_curr_local = self._to_particle_local(matched_P_curr, particle.x, particle.y, particle.yaw)
            R_2d, t_2d, _, _, inlier_count_out, _ = ransac_kabsch(
                P_prev_local, P_curr_local,
                iterations=100, strict_thresh=0.01, relaxed_thresh=0.05,
                min_kpts=self.min_keypoints, min_inls=self.min_inliers
            )

        if R_2d is not None and inlier_count_out >= self.min_inliers:
            d_fwd, d_lat = t_2d[0], t_2d[1]
            d_yaw = math.atan2(R_2d[1, 0], R_2d[0, 0])
            inlier_count = inlier_count_out

        print("Inliers:", inlier_count)
        return d_fwd, d_lat, d_yaw, inlier_count

    def _to_particle_local(self, pts, x, y, yaw):
        # rotate world-frame points into the particle's local coordinate frame
        c, s = math.cos(-yaw), math.sin(-yaw)
        out  = np.empty_like(pts, dtype=np.float64)
        dx   = pts[:, 0] - x
        dy   = pts[:, 1] - y
        out[:, 0] = dx * c - dy * s
        out[:, 1] = dx * s + dy * c
        out[:, 2] = pts[:, 2]
        return out

    def _process_particle_data_association(self, particle, des, points_base, points_cam):
        innovations, s_matrices = [], []
        matched_current_indices = set()
        matched_map_indices     = set()

        map_pts_np, map_des_np = particle.get_map_arrays()

        # only check landmarks that are actually visible from this particle's pose
        vis_indices = get_visible_landmarks(
            map_pts_np, particle.x, particle.y, particle.yaw,
            self.base_to_kinect_R_T, self.base_to_kinect_t,
            self.cx, self.cy, self.focal_length
        )

        if len(vis_indices) > 0 and des is not None and len(map_des_np) > 0:
            vis_des = map_des_np[vis_indices]
            matches = self.bf.match(vis_des, des)

            # stricter threshold for map matching than for frame-to-frame
            valid_matches = [m for m in matches if m.distance < 60]
            valid_matches.sort(key=lambda m: m.distance)

            if len(valid_matches) > 0:
                map_indices  = [vis_indices[m.queryIdx] for m in valid_matches]
                train_indices = [m.trainIdx for m in valid_matches]
                match_pairs  = list(zip(map_indices, train_indices))

                cos_yaw, sin_yaw = math.cos(-particle.yaw), math.sin(-particle.yaw)

                # cache rotation matrix rows to avoid repeated indexing in the loop
                R00, R01, R02 = self.base_to_kinect_R[0, 0], self.base_to_kinect_R[0, 1], self.base_to_kinect_R[0, 2]
                R20, R21, R22 = self.base_to_kinect_R[2, 0], self.base_to_kinect_R[2, 1], self.base_to_kinect_R[2, 2]
                tx, tz = self.base_to_kinect_t[0], self.base_to_kinect_t[2]

                for map_idx, train_idx in match_pairs:
                    pt_w_x = map_pts_np[map_idx, 0]
                    pt_w_y = map_pts_np[map_idx, 1]
                    pt_w_z = map_pts_np[map_idx, 2]

                    # transform landmark from world -> base -> camera frame
                    dx, dy = pt_w_x - particle.x, pt_w_y - particle.y
                    base_x = dx * cos_yaw - dy * sin_yaw
                    base_y = dx * sin_yaw + dy * cos_yaw

                    cam_x = R00 * base_x + R01 * base_y + R02 * pt_w_z + tx
                    cam_z = R20 * base_x + R21 * base_y + R22 * pt_w_z + tz

                    if cam_z <= 0.1:
                        continue

                    matched_current_indices.add(train_idx)
                    matched_map_indices.add(map_idx)

                    z0, z1 = points_cam[train_idx][0], points_cam[train_idx][2]
                    r00, r01, r10, r11 = self._compute_feature_covariance(cam_x, cam_z, particle.yaw)

                    # EKF update for this landmark
                    z_meas = np.array([z0,   z1])
                    z_pred = np.array([cam_x, cam_z])
                    R_meas = np.array([[r00, r01], [r10, r11]])

                    # measurement Jacobian H: d(cam_x, cam_z) / d(landmark_x, landmark_y)
                    H00 =  R00 * cos_yaw + R01 * sin_yaw
                    H01 = -R00 * sin_yaw + R01 * cos_yaw
                    H10 =  R20 * cos_yaw + R21 * sin_yaw
                    H11 = -R20 * sin_yaw + R21 * cos_yaw
                    JH  = np.array([[H00, H01], [H10, H11]])

                    v, S = particle.update_landmark(map_idx, z_meas, z_pred, JH, R_meas)

                    if v is not None:
                        innovations.append(v)
                        s_matrices.append(S)

        # update seen/missed counters for all visible landmarks
        for map_idx in vis_indices:
            particle.map_expected[map_idx] += 1
            if map_idx in matched_map_indices:
                particle.map_seen[map_idx]   += 1
                particle.map_missed[map_idx]  = 0
            else:
                particle.map_missed[map_idx] += 1

        return matched_current_indices, matched_map_indices, innovations, s_matrices

    def _evaluate_particle_weights(self, particle, innovations, s_matrices):
        # log-likelihood weight update - using log to avoid underflow with many landmarks
        log_likelihood = 0.0
        log_2pi = 1.8378770664093453   # log(2*pi)

        for v, S in zip(innovations, s_matrices):
            s00, s01, s10, s11 = S
            det_S = max(s00 * s11 - s01 * s10, 1e-12)
            v0, v1 = v
            # Mahalanobis distance: v^T * S^-1 * v (2x2 closed form)
            mahalanobis = (v0 * (s11 * v0 - s01 * v1) + v1 * (-s10 * v0 + s00 * v1)) / det_S
            log_prob = -0.5 * (log_2pi + math.log(det_S) + mahalanobis)
            log_likelihood += log_prob

        particle.weight *= (math.exp(log_likelihood) + 1e-300)

    def _extend_particle_landmark_map(self, particle, points_cam, des, matched_current_indices, points_base):
        c_y, s_y = math.cos(particle.yaw), math.sin(particle.yaw)

        # only add new landmarks from unmatched observations
        unmatched_indices = [idx for idx, _ in enumerate(points_cam)
                             if idx not in matched_current_indices]

        # cap how many we add per frame to avoid blowing up the map too fast
        if len(unmatched_indices) > 5:
            unmatched_indices = random.sample(unmatched_indices, 5)

        for pt_idx in unmatched_indices:
            if particle.num_landmarks < self.max_map_landmarks:
                pt_cam_z0, pt_cam_z1 = points_cam[pt_idx][0], points_cam[pt_idx][2]
                pt_base = points_base[pt_idx]

                # rotate base-frame point into world frame
                x_g = pt_base[0] * c_y - pt_base[1] * s_y + particle.x
                y_g = pt_base[0] * s_y + pt_base[1] * c_y + particle.y

                c00, c01, c10, c11 = self._compute_feature_covariance(pt_cam_z0, pt_cam_z1, particle.yaw)
                particle.add_landmark(x_g, y_g, pt_base[2], des[pt_idx], c00, c01, c10, c11)

        # prune stale landmarks every 15 frames instead of every frame - saves time
        if self.frame_idx % 15 == 0:
            particle.prune_map(self.max_missed_frames)


# ---------------------------------------------------------------------------
# standalone geometry helpers (not methods so they can be tested separately)
# ---------------------------------------------------------------------------

def ransac_kabsch(P, Q, iterations, strict_thresh, relaxed_thresh, min_kpts, min_inls):
    """RANSAC over Kabsch 2D alignment. Returns (R, t, inlier_P, inlier_Q, n_inliers, n_outliers)."""
    best_inliers = None
    best_count   = 0
    n = len(P)

    if n < min_kpts:
        return None, None, P, Q, 0, 0

    P2 = P[:, :2]
    Q2 = Q[:, :2]

    for _ in range(iterations):
        idx = random.sample(range(n), 3)
        R_hyp, t_hyp = kabsch_2d(P2[idx], Q2[idx])
        errors = calculate_errors(P, Q, R_hyp, t_hyp)

        inliers      = errors < strict_thresh
        inlier_count = np.sum(inliers)

        if inlier_count > best_count and inlier_count >= 4:
            # re-fit on current inliers and check with relaxed threshold
            R_refined, t_refined = kabsch_2d(P2[inliers], Q2[inliers])
            refined_errors   = calculate_errors(P, Q, R_refined, t_refined)
            refined_inliers  = refined_errors < relaxed_thresh
            refined_count    = np.sum(refined_inliers)

            if refined_count > best_count:
                best_count   = refined_count
                best_inliers = refined_inliers

            # early exit if basically all points agree
            if best_count / float(n) > 0.90:
                break

        elif inlier_count > best_count:
            best_count   = inlier_count
            best_inliers = inliers

    if best_inliers is None or np.sum(best_inliers) < min_inls:
        return None, None, P, Q, 0, 0

    R, t          = kabsch_2d(P2[best_inliers], Q2[best_inliers])
    inlier_count  = np.sum(best_inliers)
    outlier_count = n - inlier_count
    return R, t, P[best_inliers], Q[best_inliers], int(inlier_count), int(outlier_count)


def kabsch_2d(P2, Q2):
    # find the optimal 2D rotation+translation between two point sets
    # using the closed-form solution (SVD-free for 2D)
    P_mean = np.mean(P2, axis=0)
    Q_mean = np.mean(Q2, axis=0)
    P_c = P2 - P_mean
    Q_c = Q2 - Q_mean

    num = np.dot(Q_c[:, 0], P_c[:, 1]) - np.dot(Q_c[:, 1], P_c[:, 0])
    den = np.dot(Q_c[:, 0], P_c[:, 0]) + np.dot(Q_c[:, 1], P_c[:, 1])

    theta = math.atan2(num, den)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R, P_mean - (R @ Q_mean)


def calculate_errors(P, Q, R, t):
    diff = P[:, :2] - (Q[:, :2] @ R.T + t)
    return np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def get_visible_landmarks(pts, global_x, global_y, global_yaw,
                           base_to_kinect_R_T, base_to_kinect_t, cx, cy, focal):
    """Return indices of landmarks currently in the camera FOV."""
    if pts.shape[0] == 0:
        return []

    dx = pts[:, 0] - global_x
    dy = pts[:, 1] - global_y

    # quick spatial cull: only consider landmarks within 4m (saves doing full projection)
    dist_sq      = dx*dx + dy*dy
    spatial_mask = dist_sq <= 16.0
    if not np.any(spatial_mask):
        return []

    candidate_indices = np.where(spatial_mask)[0]
    dx_m, dy_m, dz_m = dx[spatial_mask], dy[spatial_mask], pts[spatial_mask, 2]

    # rotate into base frame
    cos_yaw, sin_yaw = math.cos(-global_yaw), math.sin(-global_yaw)
    base_x = dx_m * cos_yaw - dy_m * sin_yaw
    base_y = dx_m * sin_yaw + dy_m * cos_yaw

    pts_base3        = np.empty((len(dx_m), 3), dtype=np.float64)
    pts_base3[:, 0]  = base_x
    pts_base3[:, 1]  = base_y
    pts_base3[:, 2]  = dz_m

    # project into camera frame
    pts_cam        = pts_base3 @ base_to_kinect_R_T + base_to_kinect_t
    cam_x, cam_y, cam_z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # FOV and depth check (640x480 image)
    vis_mask  = (cam_z > 0.1) & (cam_z <= 6.0)
    u = (cam_x * focal / cam_z) + cx
    v = (cam_y * focal / cam_z) + cy
    vis_mask &= (u >= 0) & (u < 640) & (v >= 0) & (v < 480)

    return candidate_indices[vis_mask].tolist()


def pointcloud(points, pc_pub, stamp, frame_id="orb_odom"):
    header = std_msgs.msg.Header(stamp=stamp, frame_id=frame_id)
    pc_pub.publish(pcl2.create_cloud_xyz32(
        header, points.tolist() if isinstance(points, np.ndarray) else points))


def publish_odometry(node, current_time, global_x, global_y, global_yaw):
    node.last_pub_pose = (global_x, global_y, global_yaw)

    odom_msg = Odometry()
    odom_msg.header.stamp        = current_time
    odom_msg.header.frame_id     = node.odom_frame
    odom_msg.child_frame_id      = node.base_frame
    odom_msg.pose.pose.position.x = float(global_x)
    odom_msg.pose.pose.position.y = float(global_y)

    q = quaternion_from_euler(0, 0, global_yaw)
    odom_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    node.odom_pub.publish(odom_msg)

    # also broadcast TF so RViz can show the robot
    t_msg = TransformStamped()
    t_msg.header.stamp       = current_time
    t_msg.header.frame_id    = node.odom_frame
    t_msg.child_frame_id     = node.base_frame
    t_msg.transform.translation.x = float(global_x)
    t_msg.transform.translation.y = float(global_y)
    t_msg.transform.translation.z = 0.0
    t_msg.transform.rotation       = odom_msg.pose.pose.orientation
    node.tf_broadcaster.sendTransform(t_msg)

    # only append to path if we moved at least 1cm (avoids thousands of duplicate poses)
    if (len(node.path_msg.poses) == 0 or
            math.sqrt((global_x - node.path_msg.poses[-1].pose.position.x)**2 +
                      (global_y - node.path_msg.poses[-1].pose.position.y)**2) > 0.01):
        pose = PoseStamped()
        pose.header.stamp    = current_time
        pose.header.frame_id = node.odom_frame
        pose.pose.position.x = float(global_x)
        pose.pose.position.y = float(global_y)
        pose.pose.orientation = odom_msg.pose.pose.orientation
        node.path_msg.poses.append(pose)

    node.path_msg.header.stamp = current_time
    node.path_pub.publish(node.path_msg)


def publish_pf_particles_markers(node, current_time):
    # draw top-50 particles as arrows in RViz (LINE_LIST: each pair of points = one arrow)
    marker = Marker()
    marker.header.frame_id = node.odom_frame
    marker.header.stamp    = current_time
    marker.ns              = "pf_particles_vector"
    marker.id              = 0
    marker.type            = Marker.LINE_LIST
    marker.action          = Marker.ADD
    marker.scale.x         = 0.03
    base_color = ColorRGBA(r=1.0, g=0.5, b=0.1, a=0.8)

    top_particles = sorted(node.pf.particles, key=lambda p: p.weight, reverse=True)[:50]

    for particle in top_particles:
        p_start = Point(x=float(particle.x), y=float(particle.y), z=0.0)
        p_end   = Point(
            x=float(particle.x + 0.25 * math.cos(particle.yaw)),
            y=float(particle.y + 0.25 * math.sin(particle.yaw)),
            z=0.0
        )
        marker.points.append(p_start)
        marker.points.append(p_end)
        marker.colors.append(base_color)
        marker.colors.append(base_color)

    marker_array = MarkerArray()
    marker_array.markers.append(marker)
    node.pf_particles_marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = FastSlamNode()
    # profiling is always on - dump goes to fastslam.prof_RANSAC2000
    cProfile.runctx('rclpy.spin(node)', globals(), locals(), 'fastslam.prof_RANSAC2000')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()