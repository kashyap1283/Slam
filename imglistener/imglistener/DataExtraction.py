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
        self.csv_filename = f"odometry_export_{timestamp}.csv"
        
        self.slam_data  = {'timestamp': None, 'x': None, 'y': None, 'theta': None}
        self.wheel_data = {'timestamp': None, 'x': None, 'y': None, 'theta': None}
        self.imu_data   = {'timestamp': None, 'theta': None}

        self.imu_yaw_offset = None  # captured on first IMU message

        # Driven distance tracking
        self._wheel_prev_x = None
        self._wheel_prev_y = None
        self.driven_dist   = 0.0

        self.init_csv()
        
        self.slam_sub  = self.create_subscription(Odometry, '/serf01/odometry/project_slam', self.slam_callback,  10)
        self.wheel_sub = self.create_subscription(Odometry, '/serf01/odometry/wheel',         self.wheel_callback, 10)
        self.imu_sub   = self.create_subscription(Imu,      '/serf01/odometry/imu',           self.imu_callback,   10)
        
        self.get_logger().info(f"Odometry Exporter started. Logging to {self.csv_filename}")

    # ------------------------------------------------------------------ #

    def init_csv(self):
        with open(self.csv_filename, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp',
                # raw poses
                'slam_pose_x', 'slam_pose_y',
                'wheel_pose_x', 'wheel_pose_y',
                'slam_rotation_theta', 'imu_rotation_theta',
                # driven distance (x-axis for sigma(d) plot)
                'driven_dist_m',
                # XY errors  ← NEW
                'xy_euclidean_error_m',   # euclidean distance SLAM vs wheel
                'x_error_m',              # signed: slam_x - wheel_x
                'y_error_m',              # signed: slam_y - wheel_y
                # Angle error  ← NEW
                'yaw_error_deg',          # signed: slam_theta - imu_theta
            ])

    def quaternion_to_theta(self, quat):
        """Yaw from quaternion → degrees in [-180, 180]."""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        theta_rad = math.atan2(2.0 * (w * z + x * y),
                               1.0 - 2.0 * (y * y + z * z))
        return math.degrees(theta_rad)

    @staticmethod
    def wrap_180(deg):
        return (deg + 180.0) % 360.0 - 180.0

    # ------------------------------------------------------------------ #

    def slam_callback(self, msg: Odometry):
        self.slam_data['timestamp'] = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.slam_data['x']     = msg.pose.pose.position.x
        self.slam_data['y']     = msg.pose.pose.position.y
        self.slam_data['theta'] = self.quaternion_to_theta(msg.pose.pose.orientation)
        self.write_row()

    def wheel_callback(self, msg: Odometry):
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        raw_theta = self.quaternion_to_theta(msg.pose.pose.orientation)

        # 1. Capture the exact state of the robot the moment it wakes up
        if not hasattr(self, '_wheel_start_raw_x'):
            self._wheel_start_raw_x = raw_x
            self._wheel_start_raw_y = raw_y
            self._wheel_start_theta = math.radians(raw_theta) # We MUST capture initial yaw
            self._wheel_prev_x = 0.0
            self._wheel_prev_y = 0.0

        # 2. Calculate the raw difference from the start position
        dx_raw = raw_x - self._wheel_start_raw_x
        dy_raw = raw_y - self._wheel_start_raw_y

        # 3. Rotate the coordinate system so the robot's starting direction is +X
        # We rotate by the negative initial yaw to "cancel out" whatever direction it woke up facing
        phi = -self._wheel_start_theta
        standard_x = dx_raw * math.cos(phi) - dy_raw * math.sin(phi)
        standard_y = dx_raw * math.sin(phi) + dy_raw * math.cos(phi)
        
        # Standardize the rotation so the robot starts at 0.0 degrees
        standard_theta = self.wrap_180(raw_theta - math.degrees(self._wheel_start_theta))

        # 4. Accumulate driven distance using the clean, standard coordinates
        if self._wheel_prev_x is not None:
            self.driven_dist += math.sqrt((standard_x - self._wheel_prev_x)**2 + (standard_y - self._wheel_prev_y)**2)
            
        self._wheel_prev_x = standard_x
        self._wheel_prev_y = standard_y

        # 5. Save the robust, standardized data
        self.wheel_data['timestamp'] = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.wheel_data['x']     = standard_x
        self.wheel_data['y']     = standard_y
        self.wheel_data['theta'] = standard_theta
        self.write_row()

    def imu_callback(self, msg: Imu):
        """
        Subtract the initial IMU heading so that IMU yaw starts at 90°,
        matching the SLAM starting yaw (robot faces +Y = 90° in world frame).
        Raw IMU starts at ~172° (absolute compass); SLAM starts at 90°.
        offset = raw_first_reading - 90.0  →  172.33 - 90.0 = 82.33°
        """
        raw_theta = self.quaternion_to_theta(msg.orientation)

        if self.imu_yaw_offset is None:
            self.imu_yaw_offset = raw_theta - 0.0
            self.get_logger().info(f"IMU yaw offset captured: {self.imu_yaw_offset:.2f}°")

        relative_theta = self.wrap_180(raw_theta - self.imu_yaw_offset)

        self.imu_data['timestamp'] = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.imu_data['theta']     = relative_theta
        self.write_row()

    # ------------------------------------------------------------------ #

    def write_row(self):
        ts = (self.slam_data['timestamp']
              or self.wheel_data['timestamp']
              or self.imu_data['timestamp']
              or '')

        slam_x  = self.slam_data['x']
        slam_y  = self.slam_data['y']
        slam_th = self.slam_data['theta']
        wheel_x = self.wheel_data['x']
        wheel_y = self.wheel_data['y']
        imu_th  = self.imu_data['theta']

        # ── XY euclidean error (SLAM vs wheel) ───────────────────────────
        if slam_x is not None and wheel_x is not None:
            x_err  = slam_x - wheel_x
            y_err  = slam_y - wheel_y
            xy_err = math.sqrt(x_err ** 2 + y_err ** 2)
        else:
            x_err = y_err = xy_err = ''

        # ── Angle error (SLAM vs IMU) ─────────────────────────────────────
        if slam_th is not None and imu_th is not None:
            yaw_err = self.wrap_180(slam_th - imu_th)
        else:
            yaw_err = ''

        # ── terminal live print ───────────────────────────────────────────
        if slam_x is not None and wheel_x is not None:
            yaw_str = f"{yaw_err:.2f}°" if yaw_err != '' else 'N/A'
            self.get_logger().info(
                f"d={self.driven_dist:.3f}m | "
                f"XY_err={xy_err:.4f}m (dx={x_err:+.4f} dy={y_err:+.4f}) | "
                f"Yaw_err={yaw_str}"
            )

        # ── write CSV row ─────────────────────────────────────────────────
        def fmt(v, d=6):
            return f"{v:.{d}f}" if isinstance(v, float) else v

        with open(self.csv_filename, 'a', newline='') as f:
            csv.writer(f).writerow([
                fmt(ts, 4),
                fmt(slam_x)      if slam_x  is not None else '',
                fmt(slam_y)      if slam_y  is not None else '',
                fmt(wheel_x)     if wheel_x is not None else '',
                fmt(wheel_y)     if wheel_y is not None else '',
                fmt(slam_th, 4)  if slam_th is not None else '',
                fmt(imu_th,  4)  if imu_th  is not None else '',
                fmt(self.driven_dist, 4),
                fmt(xy_err)      if xy_err  != '' else '',
                fmt(x_err)       if x_err   != '' else '',
                fmt(y_err)       if y_err   != '' else '',
                fmt(yaw_err, 4)  if yaw_err != '' else '',
            ])


def main(args=None):
    rclpy.init(args=args)
    exporter = OdometryExporter()
    rclpy.spin(exporter)
    exporter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()