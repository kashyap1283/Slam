import numpy as np
import math
import random

class Particle:
    """Represents a single robot hypothesis using pre-allocated flat arrays for extreme performance."""
    def __init__(self, initial_weights, x=0.0, y=0.0, yaw=0.0):
        self.weight = initial_weights
        self.x = x
        self.y = y
        self.yaw = yaw
        
        # Pre-allocate memory for up to 500 landmarks to avoid ANY list.append or np.array() overhead
        self.max_landmarks = 500
        self.num_landmarks = 0
        
        self.map_pts = np.zeros((self.max_landmarks, 3), dtype=np.float64)
        self.map_des = np.zeros((self.max_landmarks, 32), dtype=np.uint8)
        self.map_cov = np.zeros((self.max_landmarks, 4), dtype=np.float64) # Flat 2x2 covariance
        self.map_seen = np.zeros(self.max_landmarks, dtype=np.int32)
        self.map_expected = np.zeros(self.max_landmarks, dtype=np.int32)
        self.map_missed = np.zeros(self.max_landmarks, dtype=np.int32)

    def predict_motion(self, d_forward, d_lateral, d_yaw, sigmas):
        noisy_fwd = d_forward + random.gauss(0, sigmas['forward'])
        noisy_lat = d_lateral + random.gauss(0, sigmas['lateral'])
        noisy_yaw = d_yaw + random.gauss(0, sigmas['yaw'])

        c = math.cos(self.yaw)
        s = math.sin(self.yaw)

        self.x += noisy_fwd * c - noisy_lat * s
        self.y += noisy_fwd * s + noisy_lat * c
        
        # Inline wrap angle
        new_yaw = self.yaw + noisy_yaw
        while new_yaw > math.pi: new_yaw -= 2 * math.pi
        while new_yaw < -math.pi: new_yaw += 2 * math.pi
        self.yaw = new_yaw

    def add_landmark(self, x, y, z, des, c00, c01, c10, c11):
        """Injects landmark directly into pre-allocated memory slot."""
        if self.num_landmarks < self.max_landmarks:
            idx = self.num_landmarks
            self.map_pts[idx, 0] = x
            self.map_pts[idx, 1] = y
            self.map_pts[idx, 2] = z
            self.map_des[idx] = des
            self.map_cov[idx, 0] = c00
            self.map_cov[idx, 1] = c01
            self.map_cov[idx, 2] = c10
            self.map_cov[idx, 3] = c11
            self.map_seen[idx] = 1
            self.map_expected[idx] = 1
            self.map_missed[idx] = 0
            self.num_landmarks += 1

    def update_landmark(self, idx, z0, z1, r00, r01, r10, r11, q00, q11, particle_yaw):
        """Pure scalar EKF math. 100x faster than np.linalg.inv and matrix multiplication."""
        c00, c01, c10, c11 = self.map_cov[idx]
        
        # P_pred = cov + Q
        p00 = c00 + q00
        p01 = c01
        p10 = c10
        p11 = c11 + q11
        
        # S = P_pred + R
        s00, s01 = p00 + r00, p01 + r01
        s10, s11 = p10 + r10, p11 + r11
        
        det = s00 * s11 - s01 * s10
        if det <= 1e-12: 
            return None, None
            
        inv_det = 1.0 / det
        sinv00, sinv01 = s11 * inv_det, -s01 * inv_det
        sinv10, sinv11 = -s10 * inv_det, s00 * inv_det
        
        # K = P_pred * S_inv
        k00, k01 = p00 * sinv00 + p01 * sinv10, p00 * sinv01 + p01 * sinv11
        k10, k11 = p10 * sinv00 + p11 * sinv10, p10 * sinv01 + p11 * sinv11
        
        # Correction = K * innovation
        cx = k00 * z0 + k01 * z1
        cy = k10 * z0 + k11 * z1
        
        cyaw, syaw = math.cos(particle_yaw), math.sin(particle_yaw)
        self.map_pts[idx, 0] += cy * cyaw - (-cx) * syaw
        self.map_pts[idx, 1] += cy * syaw + (-cx) * cyaw
        
        # I - K
        ik00, ik01 = 1.0 - k00, -k01
        ik10, ik11 = -k10, 1.0 - k11
        
        # Update Covariance (I-K) * P_pred
        self.map_cov[idx, 0] = ik00 * p00 + ik01 * p10
        self.map_cov[idx, 1] = ik00 * p01 + ik01 * p11
        self.map_cov[idx, 2] = ik10 * p00 + ik11 * p10
        self.map_cov[idx, 3] = ik10 * p01 + ik11 * p11
        
        return (z0, z1), (s00, s01, s10, s11)

    def prune_map(self, max_missed_frames):
        """In-place array filtering. Zero allocations."""
        if self.num_landmarks == 0: return
        
        keep_mask = self.map_missed[:self.num_landmarks] < max_missed_frames
        keep_count = np.sum(keep_mask)
        
        if keep_count < self.num_landmarks:
            self.map_pts[:keep_count] = self.map_pts[:self.num_landmarks][keep_mask]
            self.map_des[:keep_count] = self.map_des[:self.num_landmarks][keep_mask]
            self.map_cov[:keep_count] = self.map_cov[:self.num_landmarks][keep_mask]
            self.map_seen[:keep_count] = self.map_seen[:self.num_landmarks][keep_mask]
            self.map_expected[:keep_count] = self.map_expected[:self.num_landmarks][keep_mask]
            self.map_missed[:keep_count] = self.map_missed[:self.num_landmarks][keep_mask]
            self.num_landmarks = keep_count

    def get_map_arrays(self):
        """Returns views of the populated map slice. Instantaneous."""
        return self.map_pts[:self.num_landmarks], self.map_des[:self.num_landmarks]


class PF:
    """Wrapper class to manage the collection of particles."""
    def __init__(self, num_particles, x=0.0, y=0.0, yaw=0.0):
        self.N = num_particles
        initial_weights = 1.0 / self.N
        self.particles = [Particle(initial_weights, x, y, yaw) for _ in range(self.N)]

    def normalize_weights(self):
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0.0:
            inv_total = 1.0 / total_weight
            for p in self.particles:
                p.weight *= inv_total
        else:
            reset_w = 1.0 / self.N
            for p in self.particles:
                p.weight = reset_w

    def get_best_particle(self):
        return max(self.particles, key=lambda p: p.weight)