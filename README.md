# FastSLAM — RGB-D + Wheel/IMU Fusion (ROS 2)

FastSLAM  implementation for the `serf01` omnidirectional robot using a Kinect-style
RGB-D camera. Built as part of a robotics project at university.

The idea is to run a particle filter where each particle carries its own EKF landmark map.
Motion comes from either visual odometry (ORB + RANSAC/Kabsch) or wheel+IMU as fallback,
depending on how many RANSAC inliers we get that frame.

---

## How it works

Each RGB frame triggers the main loop:

1. Detect ORB keypoints, filter out ones with invalid depth, back-project to 3D
2. Match current frame against previous frame using BFMatcher (Hamming < 75)
3. Run RANSAC + Kabsch to estimate the 2D rigid transform (motion estimate)
4. If inlier count >= `visual_inlier_threshold`, use camera motion — otherwise fall back
   to wheel velocity + IMU angular rate integrated over dt
5. Predict all particles forward with the chosen motion + gaussian noise
6. For each particle: find visible landmarks, match descriptors against current frame,
   run EKF update per landmark, accumulate log-likelihood as particle weight
7. Normalize weights, resample if N_eff < threshold, publish best particle as odometry

The FastSLAM posterior factorization:

```
p(x_{1:t}, m | z_{1:t}, u_{1:t}) = p(x_{1:t} | z_{1:t}, u_{1:t}) * prod_k p(m_k | x_{1:t}, z_{1:t})
```

Trajectory is sampled by the particle filter, each landmark gets an independent 2D EKF.

---

## Repo layout

```
.
├── README.md
├── imglistener/
│   ├── package.xml
│   ├── setup.py
│   ├── setup.cfg
│   └── imglistener/
│       ├── __init__.py
│       ├── imglistener.py       # FastSlamNode - main loop, RANSAC, sensor fusion
│       ├── particlefilter.py    # Particle class (pre-allocated arrays) + PF manager
│       ├── ekf.py               # Generic EKF (used per landmark inside each particle)
│       └── DataExtraction.py    # Evaluation logger - writes CSV for offline analysis
├── build/
├── install/
├── log/
└── fastslam.prof_RANSAC2000     # cProfile output from last run
```

---

## Building

Tested on ROS 2 Humble. Should work on Iron/Jazzy.

**Dependencies:**
- ROS 2 Humble / Iron / Jazzy
- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- ROS packages: `cv_bridge`, `tf2_ros`, `tf_transformations`, `sensor_msgs_py`,
  `nav_msgs`, `geometry_msgs`, `visualization_msgs`

```bash
source /opt/ros/<distro>/setup.bash
colcon build --packages-select imglistener
source install/setup.bash
```

---

## Running

Three terminals:

```bash
# Terminal 1 - SLAM node (also dumps cProfile to fastslam.prof_RANSAC2000)
ros2 run imglistener imglistener

# Terminal 2 - evaluation logger (optional, writes ekf_evaluation_export_<timestamp>.csv)
ros2 run imglistener data_extractor

# Terminal 3 - replay a bag
ros2 bag play path/to/rosbag --clock
```

---

## Parameters

Declared on `fastslam_node`, can be overridden at launch.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `min_keypoints` | 5 | Skip frame if fewer valid keypoints |
| `min_inliers` | 6 | Minimum RANSAC inliers to accept the motion estimate |
| `visual_inlier_threshold` | 20 | Below this, fall back to wheel+IMU |
| `camera_cx`, `camera_cy` | 318.525, 241.181 | Camera principal point (pixels) |
| `camera_f` | 526.61 | Focal length (pixels) |
| `odom_frame` | `orb_odom` | Frame ID for published odometry |
| `base_frame` | `base_link` | Robot base frame |
| `camera_frame` | `kinect_depth` | Depth camera optical frame |
| `max_missed_frames` | 3 | Consecutive misses before a landmark is pruned |
| `num_particles` | 100 | Number of particles in the filter |
| `max_map_landmarks` | 1000 | Per-particle landmark cap |
| `min_depth_mm`, `max_depth_mm` | 50, 5000 | Valid depth range for back-projection |
| `resample_threshold` | 0.50 | Resample when N_eff drops below this fraction of N |

---

## Topics

**Published:**

| Topic | Type | Description |
|-------|------|-------------|
| `/serf01/odometry/project_slam` | `nav_msgs/Odometry` | Best particle pose |
| `/orb_path` | `nav_msgs/Path` | Accumulated trajectory |
| `/global_cloud` | `sensor_msgs/PointCloud2` | Best particle landmark map |
| `/pf_particles_marker` | `visualization_msgs/MarkerArray` | Top-50 particle arrows in RViz |
| TF: `orb_odom -> base_link` | — | Live transform |

**Subscribed:**

| Topic | Type | Description |
|-------|------|-------------|
| `/serf01/nav_rgbd_1/rgb/image_raw` | `sensor_msgs/Image` | RGB frames |
| `/serf01/nav_rgbd_1/depth/image_raw` | `sensor_msgs/Image` | Depth frames |
| `/serf01/odometry/wheel` | `nav_msgs/Odometry` | Wheel velocity (linear.x, linear.y) |
| `/serf01/odometry/imu` | `sensor_msgs/Imu` | IMU angular rate (angular_velocity.z) |

---

## Evaluation logger

`DataExtraction.py` subscribes to the SLAM output, wheel odometry, and IMU simultaneously
and writes a CSV for offline error analysis. File is named
`ekf_evaluation_export_<YYYYMMDD_HHMMSS>.csv`.

Both wheel odom and IMU yaw are offset-corrected on the first message so they start at
(0, 0, 0 deg) — same starting condition as the SLAM estimate.

| Column | Description |
|--------|-------------|
| `timestamp` | ROS time in seconds |
| `slam_pose_x/y`, `slam_rotation_theta` | SLAM estimated pose |
| `wheel_pose_x/y`, `wheel_rotation_theta` | Wheel odom (relative from start) |
| `imu_rotation_theta` | IMU yaw (relative, starts at 0 deg) |
| `driven_dist_m` | Accumulated driven distance — useful as x-axis for error plots |
| `xy_euclidean_error_m`, `x_error_m`, `y_error_m` | SLAM vs wheel position error |
| `yaw_error_deg` | SLAM vs IMU yaw error |

---

## Known issues / TODOs
- Resampling too much on straights
- Loop closure is not implemented — the filter will drift on long trajectories.

## Acknowledgements

We used AI tools to assist in writing parts of this codebase—specifically helper functions and example snippets—as well as drafting the documentation. While AI helped us write these components, the project team made all core design and architectural decisions. We thoroughly verified, tested, and debugged all AI-assisted code to ensure its accuracy, maintaining full ownership of the final implementation.