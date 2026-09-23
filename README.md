FastSLAM 1.0 — RGB-D + Wheel/IMU Fusion (ROS 2)
FastSLAM 1.0 implementation for the serf01 omnidirectional robot using a Kinect-style RGB-D camera. Built as part of a robotics project at university.

The idea is to run a particle filter where each particle carries its own EKF landmark map. Motion comes from either visual odometry (ORB + RANSAC/Kabsch) or wheel+IMU as fallback, depending on how many RANSAC inliers we get that frame.

How it works
Each RGB frame triggers the main loop:

Detect ORB keypoints, filter out ones with invalid depth, back-project to 3D
Match current frame against previous frame using BFMatcher (Hamming < 75)
Run RANSAC + Kabsch to estimate the 2D rigid transform (motion estimate)
If inlier count >= visual_inlier_threshold, use camera motion — otherwise fall back to wheel velocity + IMU angular rate integrated over dt
Predict all particles forward with the chosen motion + gaussian noise
For each particle: find visible landmarks, match descriptors against current frame, run EKF update per landmark, accumulate log-likelihood as particle weight
Normalize weights, resample if N_eff < threshold, publish best particle as odometry
The FastSLAM posterior factorization:

p(x_{1:t}, m | z_{1:t}, u_{1:t}) = p(x_{1:t} | z_{1:t}, u_{1:t}) * prod_k p(m_k | x_{1:t}, z_{1:t})
Trajectory is sampled by the particle filter, each landmark gets an independent 2D EKF.
