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

        self.create_subscription(
            Image,
            '/serf01/nav_rgbd_1/rgb/image_raw',
            self.listener_callback,
            10
        )

        self.create_subscription(
            Image,
            '/serf01/nav_rgbd_1/depth/image_raw',
            self.depth_subscription,
            10
        )

        self.pc_pub = self.create_publisher(PointCloud2, '/orb_pointcloud', 10)
        self.current_pc_pub = self.create_publisher(PointCloud2, '/global_cloud', 10)
        self.odom_pub = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.path_pub = self.create_publisher(Path, '/orb_path', 10)

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'orb_odom'

        self.depth_image = None

        self.global_x = 0.0
        self.global_y = 0.0
        self.global_yaw = 0.0

        self.map_points_3d = []
        self.map_descriptors = []
        self.map_times_seen = []
        self.map_times_expected = []

        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def depth_subscription(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def listener_callback(self, msg):

        if self.depth_image is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        kp, des, points = feature_detector(self, frame, self.depth_image)

        if kp is None or len(kp) < MIN_KEYPOINTS:   
            return

        if len(self.map_points_3d) == 0:
            self.add_new_landmarks(points, des, set())
            return

        vis_indices = get_visible_landmarks(
            self.map_points_3d,
            self.map_times_expected,
            self.global_x,
            self.global_y,
            self.global_yaw
        )

        if len(vis_indices) == 0 or des is None:
            return

        vis_des = np.array([self.map_descriptors[i] for i in vis_indices])

        map_matches = self.bf.match(vis_des, des)

        P_global = []
        Q_local = []
        matched_current = set()

        cos_yaw = math.cos(-self.global_yaw)
        sin_yaw = math.sin(-self.global_yaw)

        for m in map_matches:

            if m.distance > 65:
                continue

            map_idx = vis_indices[m.queryIdx]
            lm = self.map_points_3d[map_idx]
            obs = points[m.trainIdx]

            x_cam, y_cam, z_cam = obs

            x_robot = z_cam
            y_robot = -x_cam

            P_global.append([lm[0], lm[1]])
            Q_local.append([x_robot, y_robot])

            matched_current.add(m.trainIdx)
            self.map_times_seen[map_idx] += 1

        P_global = np.array(P_global)
        Q_local = np.array(Q_local)

        R, t, theta = ransac_kabsch(
            P_global,
            Q_local
        )

        if R is None:
            return

        
        new_x = float(t[0])
        new_y = float(t[1])
        new_yaw = math.atan2(math.sin(theta), math.cos(theta))

        dx = new_x - self.global_x
        dy = new_y - self.global_y
        dyaw = math.atan2(math.sin(new_yaw - self.global_yaw), math.cos(new_yaw - self.global_yaw))

        if abs(dyaw) > math.radians(10):
            return

        if math.sqrt(dx*dx + dy*dy) > 0.25:
            return

        self.global_x = new_x
        self.global_y = new_y
        self.global_yaw = new_yaw

        publish_odometry(self, msg.header.stamp,
                         self.global_x,
                         self.global_y,
                         self.global_yaw)

        if len(self.map_points_3d) > 0:
            pointcloud(
                np.array(self.map_points_3d),
                self.current_pc_pub,
                msg.header.stamp,
                "orb_odom"
            )

        self.add_new_landmarks(points, des, matched_current)
        self.prune_landmarks()

    def add_new_landmarks(self, pts, des, matched):

        cos_yaw = math.cos(self.global_yaw)
        sin_yaw = math.sin(self.global_yaw)

        for i in range(len(pts)):

            if i in matched:
                continue

            x, y, z = pts[i]

            xb = z
            yb = -x

            gx = xb * cos_yaw - yb * sin_yaw + self.global_x
            gy = xb * sin_yaw + yb * cos_yaw + self.global_y

            self.map_points_3d.append([gx, gy, z])
            self.map_descriptors.append(des[i])
            self.map_times_seen.append(1)
            self.map_times_expected.append(1)

    def prune_landmarks(self):

        keep_p, keep_d, keep_s, keep_e = [], [], [], []

        for i in range(len(self.map_points_3d)):

            if self.map_times_expected[i] > 3:

                ratio = self.map_times_seen[i] / self.map_times_expected[i]

                if ratio < 0.25:
                    continue

            keep_p.append(self.map_points_3d[i])
            keep_d.append(self.map_descriptors[i])
            keep_s.append(self.map_times_seen[i])
            keep_e.append(self.map_times_expected[i])

        self.map_points_3d = keep_p
        self.map_descriptors = keep_d
        self.map_times_seen = keep_s
        self.map_times_expected = keep_e


def feature_detector(node, frame, depth):

    kp, des = node.orb.detectAndCompute(frame, None)

    if kp is None or des is None :
        return None, None, None

    pts = []
    kps = []
    desc = []

    img = frame.copy()

    for i, p in enumerate(kp):

        x, y = int(p.pt[0]), int(p.pt[1])

        if x < 0 or y < 0 or x >= depth.shape[1] or y >= depth.shape[0]:
            continue

        d = depth[y, x]

        if d == 0 or np.isnan(d) or d > 5000:
            continue

        d = d / 1000.0

        X = d * (x - cx) / f
        Y = d * (y - cy) / f

        pts.append([X, Y, d])

        kps.append(p)
        desc.append(des[i])

        cv2.putText(
            img,
            f"{d:.2f}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )

    img = cv2.drawKeypoints(
        img,
        kps,
        None,
        color=(0,255,0),
        flags=0
    )

    cv2.imshow("FeatureDetector", img)
    cv2.waitKey(1)

    return kps, np.array(desc), np.array(pts)


def get_visible_landmarks(map_points, expected, x, y, yaw):
    vis = []
    cyaw = math.cos(-yaw)
    syaw = math.sin(-yaw)
    half_fov = math.atan2(320.0, f)   # ≈ 31° half-angle

    for i, p in enumerate(map_points):
        dx = p[0] - x
        dy = p[1] - y
        lx =  dx * cyaw - dy * syaw
        ly =  dx * syaw + dy * cyaw

        if lx <= 0.1 or lx > 5.0:     # depth gate
            continue
        if abs(math.atan2(ly, lx)) > half_fov:  # FOV gate
            continue

        vis.append(i)
        expected[i] += 1
    return vis


def ransac_kabsch(P, Q, iters=400, th=0.08):

    if len(P) < MIN_KEYPOINTS:
        return None, None, None

    best = 0
    best_in = None

    for _ in range(iters):

        idx = random.sample(range(len(P)), 3)

        R, t, tht = kabsch(P[idx], Q[idx])
        if R is None or t is None:
            continue

        Qp = (R @ Q.T).T + t

        err = np.linalg.norm(P - Qp, axis=1)

        inl = err < th

        if np.sum(inl) > best:
            best = np.sum(inl)
            best_in = inl

    if best_in is None:
        return None, None, None

    P_in = P[best_in]
    Q_in = Q[best_in]

    R, t, theta = kabsch(P_in, Q_in)

    return R, t, theta

def kabsch(P, Q):
    Pc = P - np.mean(P, axis=0)
    Qc = Q - np.mean(Q, axis=0)

    # Reject degenerate geometry
    if np.sum(Qc[:,0]**2 + Qc[:,1]**2) < 1e-3:
        return None, None, None

    num = np.sum(Qc[:,0]*Pc[:,1] - Qc[:,1]*Pc[:,0])
    den = np.sum(Qc[:,0]*Pc[:,0] + Qc[:,1]*Pc[:,1])

    theta = math.atan2(num, den)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    t = np.mean(P, axis=0) - R @ np.mean(Q, axis=0)
    return R, t, theta


def pointcloud(points, pub, stamp, frame):

    header = std_msgs.msg.Header()
    header.stamp = stamp
    header.frame_id = frame

    pub.publish(pcl2.create_cloud_xyz32(header, points.tolist()))


def publish_odometry(node, stamp, x, y, yaw):

    msg = Odometry()

    msg.header.stamp = stamp
    msg.header.frame_id = "orb_odom"
    msg.child_frame_id = "base_link"

    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = 0.0

    q = quaternion_from_euler(0, 0, yaw)

    msg.pose.pose.orientation = Quaternion(
        x=float(q[0]),
        y=float(q[1]),
        z=float(q[2]),
        w=float(q[3])
    )

    node.odom_pub.publish(msg)

    t = TransformStamped()

    t.header.stamp = stamp
    t.header.frame_id = "orb_odom"
    t.child_frame_id = "base_link"

    t.transform.translation.x = float(x)
    t.transform.translation.y = float(y)
    t.transform.translation.z = 0.0

    t.transform.rotation = msg.pose.pose.orientation

    node.tf_broadcaster.sendTransform(t)

    pose = PoseStamped()

    pose.header.stamp = stamp
    pose.header.frame_id = "orb_odom"

    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = 0.0

    pose.pose.orientation = msg.pose.pose.orientation

    node.path_msg.poses.append(pose)

    node.path_msg.header.stamp = stamp

    node.path_pub.publish(node.path_msg)
    if node.path_msg.poses:
        last = node.path_msg.poses[-1]
        dist = math.sqrt((x - last.pose.position.x)**2 + (y - last.pose.position.y)**2)
        if dist < 0.05:
        # still publish odom and tf, just skip the path append
            ...
            return


def main():
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()