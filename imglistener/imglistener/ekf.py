#!/usr/bin/env python3
import numpy as np


class ExtKalman:
    def __init__(self, x, state_func, meas_func, JF, JH, R, Q):
        self.x = x.astype(float)
        self.state_func = state_func
        self.meas_func = meas_func
        self.JF = JF.astype(float)
        self.JH = JH.astype(float)
        self.R = R.astype(float)
        self.Q = Q.astype(float)

        # State covariance must not be initialized to process noise.
        self.P = np.eye(len(x), dtype=float) * 1.0

    def setJF(self, JF):
        self.JF = JF.astype(float)

    def setJH(self, JH):
        self.JH = JH.astype(float)

    def setR(self, R):
        self.R = R.astype(float)

    def setQ(self, Q):
        self.Q = Q.astype(float)

    def predictState(self):
        """
        Propagate state and covariance forward.
        Does NOT mutate self.x or self.P.
        """
        x_pred = self.state_func(self.x)
        P_pred = self.JF @ self.P @ self.JF.T + self.Q
        return x_pred, P_pred

    def predictMeasurement(self, x_pred):
        return self.meas_func(x_pred)

    def commitPrediction(self):
        """
        Predict and store the result in self.x, self.P.
        Call this once per cycle before update().
        """
        x_pred, P_pred = self.predictState()
        self.x = x_pred.flatten()
        self.P = P_pred
        return self.x, self.P

    def update(self, z):
        """
        Update step only.
        Assumes self.x and self.P already contain the predicted prior.
        """
        x_pred = self.x.copy()
        P_pred = self.P.copy()

        z_pred = self.predictMeasurement(x_pred)

        PHT = P_pred @ self.JH.T
        S = self.JH @ PHT + self.R
        K = PHT @ np.linalg.inv(S)

        innov = z - z_pred
        self.x = (x_pred + K @ innov).flatten()

        # Joseph form for numerical stability
        n = len(self.x)
        I_KH = np.eye(n) - K @ self.JH
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        return self.x, self.P