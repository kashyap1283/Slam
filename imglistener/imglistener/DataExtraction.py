#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import csv
from datetime import datetime
import math


class OdometryExporter(Node):
    def __init__(self):
        super().__init__('odometry_exporter')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"ekf_evaluation_export_{timestamp}.csv"

        self.slam_data  = {'timestamp': None, 'x': None, 'y': None, 'theta': None}
        self.wheel_data = {'timestamp': None, 'x': None, 'y': None, 'theta': None}
        self.imu_data   = {'timestamp': None, 'theta': None}

        # IMU yaw offset — captured on first message so relative yaw starts at 0
        self.imu_yaw_offset = None

        # Wheel odometry origin — captured on first message so relative XY starts at (0,0)
        self._wheel_start_raw_x = None
        self._wheel_start_raw_y = None
        self._wheel_start_theta = None
        self._wheel_prev_x      = None
        self._wheel_prev_y      = None
        self.driven_dist        = 0.0

        self.init_csv()

        self.slam_sub  = self.create_subscription(
            Odometry, '/serf01/odometry/project_slam', self.slam_callback,  10)
        self.wheel_sub = self.create_subscription(
            Odometry, '/serf01/odometry/wheel',        self.wheel_callback, 10)
        self.imu_sub   = self.create_subscription(
            Imu,      '/serf01/odometry/imu',          self.imu_callback,   10)

        self.get_logger().info(
            f"OdometryExporter started — logging to {self.csv_filename}")

    # ------------------------------------------------------------------ #

    def init_csv(self):
        with open(self.csv_filename, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp',
                # EKF (SLAM) pose
                'slam_pose_x',
                'slam_pose_y',
                'slam_rotation_theta',
                # Wheel odometry pose (offset from start, no rotation — matches EKF convention)
                'wheel_pose_x',
                'wheel_pose_y',
                'wheel_rotation_theta',
                # IMU yaw (relative, starts at 0)
                'imu_rotation_theta',
                # Driven distance (x-axis for error-vs-distance plots)
                'driven_dist_m',
                # XY errors (SLAM vs wheel)
                'xy_euclidean_error_m',
                'x_error_m',
                'y_error_m',
                # Yaw error (SLAM vs IMU)
                'yaw_error_deg',
            ])

    # ------------------------------------------------------------------ #

    def quaternion_to_theta(self, quat):
        """Extract yaw in degrees [-180, 180] from a quaternion."""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        theta_rad = math.atan2(2.0 * (w * z + x * y),
                               1.0 - 2.0 * (y * y + z * z))
        return math.degrees(theta_rad)

    @staticmethod
    def wrap_180(deg):
        """Wrap an angle in degrees to [-180, 180]."""
        return (deg + 180.0) % 360.0 - 180.0

    # ------------------------------------------------------------------ #

    def slam_callback(self, msg: Odometry):
        """
        EKF output: /serf01/odometry/project_slam
        Reads pose.pose.position.x/y directly — EKF publishes in orb_odom frame
        starting at (0, 0, 0°).
        """
        self.slam_data['timestamp'] = (msg.header.stamp.sec
                                       + msg.header.stamp.nanosec / 1e9)
        self.slam_data['x']     = msg.pose.pose.position.x
        self.slam_data['y']     = msg.pose.pose.position.y
        self.slam_data['theta'] = self.quaternion_to_theta(msg.pose.pose.orientation)
        self.write_row(self.slam_data['timestamp'])

    def wheel_callback(self, msg: Odometry):
        """
        Wheel odometry ground truth: /serf01/odometry/wheel
        Published in world/odom frame. We subtract the initial position so the
        wheel reference starts at (0, 0) matching the EKF starting pose.

        NO rotation applied — the EKF process model (ekf_state_func) uses raw
        global-frame deltas directly (FIX 1/2), so the wheel ground truth must
        use the same convention to produce valid error metrics.

        Sign convention (ROS REP-105):
            Forward  → raw_x increases → standard_x > 0
            Backward → raw_x decreases → standard_x < 0
            Left     → raw_y increases → standard_y > 0
            Right    → raw_y decreases → standard_y < 0
        """
        raw_x     = msg.pose.pose.position.x
        raw_y     = msg.pose.pose.position.y
        raw_theta = self.quaternion_to_theta(msg.pose.pose.orientation)

        # Capture origin on first message
        if self._wheel_start_raw_x is None:
            self._wheel_start_raw_x = raw_x
            self._wheel_start_raw_y = raw_y
            self._wheel_start_theta = raw_theta
            self._wheel_prev_x      = raw_x
            self._wheel_prev_y      = raw_y
            self.get_logger().info(
                f"Wheel origin captured: x={raw_x:.4f} y={raw_y:.4f} "
                f"theta={raw_theta:.2f}°")

        # Relative pose — offset from start, no rotation
        standard_x     = raw_x - self._wheel_start_raw_x
        standard_y     = raw_y - self._wheel_start_raw_y
        standard_theta = self.wrap_180(raw_theta - self._wheel_start_theta)

        # Accumulate driven distance using consecutive raw positions
        dx = raw_x - self._wheel_prev_x
        dy = raw_y - self._wheel_prev_y
        self.driven_dist   += math.sqrt(dx**2 + dy**2)
        self._wheel_prev_x  = raw_x
        self._wheel_prev_y  = raw_y

        self.wheel_data['timestamp'] = (msg.header.stamp.sec
                                        + msg.header.stamp.nanosec / 1e9)
        self.wheel_data['x']     = standard_x
        self.wheel_data['y']     = standard_y
        self.wheel_data['theta'] = standard_theta
        self.write_row(self.wheel_data['timestamp'])

    def imu_callback(self, msg: Imu):
        """
        IMU yaw: /serf01/odometry/imu
        Subtracts the first reading so relative yaw starts at 0°, matching
        the EKF starting yaw of 0°.
        """
        raw_theta = self.quaternion_to_theta(msg.orientation)

        if self.imu_yaw_offset is None:
            self.imu_yaw_offset = raw_theta
            self.get_logger().info(
                f"IMU yaw offset captured: {self.imu_yaw_offset:.2f}° "
                f"(relative yaw will start at 0°)")

        relative_theta = self.wrap_180(raw_theta - self.imu_yaw_offset)

        self.imu_data['timestamp'] = (msg.header.stamp.sec
                                      + msg.header.stamp.nanosec / 1e9)
        self.imu_data['theta']     = relative_theta
        self.write_row(self.imu_data['timestamp'])

    # ------------------------------------------------------------------ #

    def write_row(self, source_ts):
        """
        Write one CSV row. source_ts is the timestamp of the callback that
        triggered this write — each row is stamped with its own sensor time,
        not the last SLAM time.
        """
        slam_x  = self.slam_data['x']
        slam_y  = self.slam_data['y']
        slam_th = self.slam_data['theta']
        wheel_x = self.wheel_data['x']
        wheel_y = self.wheel_data['y']
        wheel_th = self.wheel_data['theta']
        imu_th  = self.imu_data['theta']

        # XY Euclidean error (SLAM vs wheel)
        if slam_x is not None and wheel_x is not None:
            x_err  = slam_x - wheel_x
            y_err  = slam_y - wheel_y
            xy_err = math.sqrt(x_err**2 + y_err**2)
        else:
            x_err = y_err = xy_err = ''

        # Yaw error (SLAM vs IMU)
        if slam_th is not None and imu_th is not None:
            yaw_err = self.wrap_180(slam_th - imu_th)
        else:
            yaw_err = ''

        # Live terminal print (only when all three sources have data)
        if slam_x is not None and wheel_x is not None and imu_th is not None:
            yaw_str = f"{yaw_err:.2f}°" if yaw_err != '' else 'N/A'
            self.get_logger().info(
                f"d={self.driven_dist:.3f}m | "
                f"SLAM=({slam_x:.3f},{slam_y:.3f},{slam_th:.1f}°) | "
                f"Wheel=({wheel_x:.3f},{wheel_y:.3f}) | "
                f"XY_err={xy_err:.4f}m (dx={x_err:+.4f} dy={y_err:+.4f}) | "
                f"Yaw_err={yaw_str}"
            )

        def fmt(v, d=6):
            return f"{v:.{d}f}" if isinstance(v, float) else (v if v is not None else '')

        with open(self.csv_filename, 'a', newline='') as f:
            csv.writer(f).writerow([
                fmt(source_ts, 4),
                fmt(slam_x)       if slam_x  is not None else '',
                fmt(slam_y)       if slam_y  is not None else '',
                fmt(slam_th,  4)  if slam_th is not None else '',
                fmt(wheel_x)      if wheel_x is not None else '',
                fmt(wheel_y)      if wheel_y is not None else '',
                fmt(wheel_th, 4)  if wheel_th is not None else '',
                fmt(imu_th,   4)  if imu_th  is not None else '',
                fmt(self.driven_dist, 4),
                fmt(xy_err)       if xy_err  != '' else '',
                fmt(x_err)        if x_err   != '' else '',
                fmt(y_err)        if y_err   != '' else '',
                fmt(yaw_err,  4)  if yaw_err != '' else '',
            ])


# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    exporter = OdometryExporter()
    rclpy.spin(exporter)
    exporter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()