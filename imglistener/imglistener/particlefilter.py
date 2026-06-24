import numpy as np
import math
import random
import copy
from .ekf import ExtKalman

class Particle:
    """Represents a single robot hypothesis using pre-allocated flat arrays for extreme performance."""
    
    __slots__ = [
        'weight', 'x', 'y', 'yaw', 
        'max_landmarks', 'num_landmarks', 
        'map_pts', 'map_des', 'map_cov', 
        'map_seen', 'map_expected', 'map_missed'
    ]

    def __init__(self, initial_weights, x=0.0, y=0.0, yaw=0.0):
        self.weight = initial_weights
        self.x = x
        self.y = y
        self.yaw = yaw
        
        self.max_landmarks = 1000 
        self.num_landmarks = 0
        
        self.map_pts = np.zeros((self.max_landmarks, 3), dtype=np.float64)
        self.map_des = np.zeros((self.max_landmarks, 32), dtype=np.uint8)
        self.map_cov = np.zeros((self.max_landmarks, 4), dtype=np.float64) 
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
        
        new_yaw = self.yaw + noisy_yaw
        while new_yaw > math.pi: new_yaw -= 2 * math.pi
        while new_yaw < -math.pi: new_yaw += 2 * math.pi
        self.yaw = new_yaw

    def add_landmark(self, x, y, z, des, c00, c01, c10, c11):
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

    def update_landmark(self, idx, z_meas, z_pred, JH, R_meas):
        """
        EKF update using the external ExtKalman class.
        Overrides the manual update with standard EKF formulation.
        """
        # 1. Extract 2D State (X, Y) and Covariance from flat arrays
        x_state = self.map_pts[idx, 0:2].copy()
        P_cov = np.array([
            [self.map_cov[idx, 0], self.map_cov[idx, 1]],
            [self.map_cov[idx, 2], self.map_cov[idx, 3]]
        ])

        # 2. Setup EKF Functions
        # FastSLAM landmarks are static, so the state transition F(x) = x
        def state_func(x): return x
        
        # The measurement prediction is pre-calculated in the particle loop
        # for spatial efficiency, so we just return it.
        def meas_func(x): return z_pred

        JF = np.eye(2)               # State Jacobian is Identity
        Q_proc = np.eye(2) * 0.001   # Process noise

        # 3. Initialize and run EKF
        ekf = ExtKalman(x_state, state_func, meas_func, JF, JH, R_meas, Q_proc)
        ekf.P = P_cov  # Override initialized covariance with the mapped landmark covariance

        ekf.commitPrediction()
        x_new, P_new = ekf.update(z_meas)

        # 4. Write updated state and covariance back to flat arrays
        self.map_pts[idx, 0] = x_new[0]
        self.map_pts[idx, 1] = x_new[1]
        
        self.map_cov[idx, 0] = P_new[0, 0]
        self.map_cov[idx, 1] = P_new[0, 1]
        self.map_cov[idx, 2] = P_new[1, 0]
        self.map_cov[idx, 3] = P_new[1, 1]

        # 5. Return Innovation and S-matrix for particle weight evaluation
        # S = H * P_pred * H^T + R
        P_pred = P_cov + Q_proc
        S = JH @ P_pred @ JH.T + R_meas
        v = z_meas - z_pred

        return (v[0], v[1]), (S[0, 0], S[0, 1], S[1, 0], S[1, 1])

    def prune_map(self, max_missed_frames):
        """Removes bad landmarks and explicitly shrinks the arrays to update RViz."""
        if self.num_landmarks == 0: 
            return
        
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
        return self.map_pts[:self.num_landmarks], self.map_des[:self.num_landmarks]


class PF:
    """Wrapper class to manage the collection of particles."""
    
    __slots__ = ['N', 'particles']
    
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

    def effective_sample_size(self):
        weight_sq_sum = sum(p.weight * p.weight for p in self.particles)
        if weight_sq_sum <= 1e-300:
            return 0.0
        return 1.0 / weight_sq_sum

    def get_best_particle(self):
        return max(self.particles, key=lambda p: p.weight)

    def systematic_resampling(self):
        N = self.N
        weights = np.array([p.weight for p in self.particles], dtype=np.float64)
        weights /= weights.sum()

        C = np.cumsum(weights)
        u = np.random.uniform(0, 1.0 / N)
        k = np.arange(1, N + 1)
        u_k = u + (k - 1) / N
        indices = np.searchsorted(C, u_k, side='left')

        import copy
        self.particles = [copy.deepcopy(self.particles[i]) for i in indices]

        uniform_w = 1.0 / N
        for p in self.particles:
            p.weight = uniform_w

