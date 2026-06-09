import numpy as np
import math
import random

class Landmark:
    """Represents a single feature in a particle's map."""
    def __init__(self, x, y, z, descriptor, initial_covariance):
        self.pos_3d = np.array([x, y, z])
        self.descriptor = descriptor
        self.covariance = initial_covariance
        
        # Tracking metrics to decide if we keep or delete this landmark
        self.times_seen = 1
        self.times_expected = 1
        self.missed_frames = 0

class Particle:
    """Represents a single robot hypothesis and its unique map."""
    def __init__(self, initial_weights,x=0.0, y=0.0, yaw=0.0):
        self.weight = initial_weights
        
        # Robot Pose State
        self.x = x
        self.y = y
        self.yaw = yaw
        
        # Map State (List of Landmark objects)
        self.map = [] 

    def predict_motion(self, d_forward, d_lateral, d_yaw, sigmas):
        """Moves the particle based on odometry, adding unique Gaussian noise."""
        noisy_fwd = d_forward + random.gauss(0, sigmas['forward'])
        noisy_lat = d_lateral + random.gauss(0, sigmas['lateral'])
        noisy_yaw = d_yaw + random.gauss(0, sigmas['yaw'])

        # Apply motion in the global frame
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += noisy_fwd * c - noisy_lat * s
        self.y += noisy_fwd * s + noisy_lat * c
        self.yaw = self._wrap_angle(self.yaw + noisy_yaw)

    def update_landmark(self, landmark_idx, v_inn, R, Q_kf, particle_yaw, kinect_to_base_matrix=None):
        """
        Updates an existing landmark's mean position and covariance using the EKF.
        Takes the pre-calculated innovation (v_inn) from the camera optical frame, 
        applies the Kalman Gain, and strictly rotates it into the global map frame.
        """
        landmark = self.map[landmark_idx]
        
        # 1. Prediction step for the landmark covariance
        P_pred = landmark.covariance + Q_kf
        
        # 2. Measurement Covariance
        S = P_pred + R
        
        try:
            # 3. Kalman Gain
            K = P_pred @ np.linalg.inv(S)
            
            # Calculate the spatial correction strictly in the Camera Optical frame
            correction_cam = K @ v_inn                    
            
            # 4. Map camera optical corrections straight to local robot base coordinates
            # correction_cam[1] is camera depth (+Z) -> maps to robot local forward (+X)
            # correction_cam[0] is camera lateral (+X) -> maps to robot local right (-Y). Negate for ROS Left (+Y)
            robot_local_x = correction_cam[1]
            robot_local_y = -correction_cam[0]

            # 5. Rotate the local robot base displacements to match global map coordinates
            c = math.cos(particle_yaw)
            s = math.sin(particle_yaw)

            delta_world_x = robot_local_x * c - robot_local_y * s
            delta_world_y = robot_local_x * s + robot_local_y * c

            # Apply corrections cleanly to the global 2D plane
            landmark.pos_3d[0] += delta_world_x
            landmark.pos_3d[1] += delta_world_y

            # 6. Update EKF Landmark uncertainty using the stable Joseph Form
            IKH = np.eye(2) - K
            landmark.covariance = IKH @ P_pred @ IKH.T + K @ R @ K.T

            return v_inn, S

        except np.linalg.LinAlgError:
            return None, None
            
    def prune_map(self, max_missed=3):
        """Removes dead/ghost landmarks to keep the map clean and fast."""
        clean_map = []
        for lm in self.map:
            # Rule 1: Don't keep if missed too many times in a row
            if lm.missed_frames > max_missed:
                continue
            # Rule 2: Don't keep if we expected to see it a lot, but rarely actually did
            if lm.times_expected > 3 and lm.times_seen < 3:
                continue
            clean_map.append(lm)
            
        self.map = clean_map

    @staticmethod
    def _wrap_angle(angle):
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle


class PF:
    """Wrapper class to manage the collection of particles."""
    def __init__(self, num_particles,  x=0.0, y=0.0, yaw=0.0):
        self.N = num_particles
        initial_weights = 1.0 / self.N
        self.particles = [Particle(initial_weights, x, y, yaw) for _ in range(self.N)]
        
    def normalize_weights(self):
        """Ensures all particle weights sum to 1.0."""
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0.0:
            for p in self.particles:
                p.weight /= total_weight
        else:
            # If all weights crashed to 0, reset evenly
            for p in self.particles:
                p.weight = 1.0 / self.N

    def get_best_particle(self):
        """Returns the particle object with the highest weight."""
        return max(self.particles, key=lambda p: p.weight)