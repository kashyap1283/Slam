from random import *
from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

class ExtKalman:
	def __init__(self, x, state_func, meas_func, JF, JH, R, Q):
		self.x = x
		self.state_func = state_func
		self.meas_func = meas_func
		self.JF = JF
		self.JH = JH
		self.R = R
		self.Q = Q
		self.P = Q # initialize

	# set Jacobi matrix of the state transition
	def setJF(self, JF):
		self.JF = JF
		
	# set Jacobi Matrix of the measurement function
	def setJH(self, JH):
		self.JH = JH

	# set measurement noise -- eg. for EKF
	def setR(self, R):
		self.R = R

	# set model noise -- eg. for EKF
	def setQ(self, Q):
		self.Q = Q

	def predictState(self):
		pstate = self.state_func(self.x)
		pP = np.matmul(self.JF, np.matmul(self.P, self.JF.transpose()))+self.Q
		return pstate, pP

	# return measurement prediction (\hat z_{t|t-1})
	def predictMeasurement(self):
		pmeas = self.meas_func(self.x)
		return pmeas
	
	# return matrix K
	def computeKalmanGain(self):
		x_tt1, P_tt1 = self.predictState()
		
		PHT = np.matmul(P_tt1, self.JH.transpose())         # PH^\top
		HPHT = np.matmul(self.JH, PHT)                      # HPH^\top
		HPHTpRi = np.linalg.inv(HPHT + self.R)             # (HPH^\top + R)^{-1}
		K = np.matmul(PHT, HPHTpRi)
		return K

	# Update self.x and self.P, return tuple (x_{t|t}, P_{t_t})
	def update(self, z):
		print("State:", self.x)
		x_tt1, P_tt1 = self.predictState()
		print("Predicted state:", x_tt1)
		z_tt1 = self.predictMeasurement()
		print("Predicted measurement:", z_tt1)
		print("Actual measurement:", z)
		K = self.computeKalmanGain()
		self.x = x_tt1 + np.matmul(K, (z-z_tt1))
		self.x = self.x.flatten()
		self.P = P_tt1 - np.matmul(K, np.matmul(self.JH, P_tt1))
		return self.x, self.P
	
