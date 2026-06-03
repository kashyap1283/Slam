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

    def update_landmark(self, landmark_idx, z_meas, R, Q_kf):
        """
        Updates a specific landmark using the Linear Kalman Filter.
        Returns the innovation (v) and innovation covariance (S) for weight updating.
        """
        landmark = self.map[landmark_idx]
        
        # 1. Prediction step for the landmark covariance
        P_pred = landmark.covariance + Q_kf
        
        # 2. Measurement Covariance
        S = P_pred + R
        
        try:
            # 3. Kalman Gain
            K = P_pred @ np.linalg.inv(S)
            
            x_old_2d = landmark.pos_3d[0:2]
            v = z_meas - x_old_2d # Innovation
            
            # 4. Update Map State
            x_up = x_old_2d + K @ v
            landmark.pos_3d[0:2] = x_up
            landmark.covariance = (np.eye(2) - K) @ P_pred
            
            return v, S
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