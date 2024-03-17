############################
#                          #
#    Author: Amin Mirzai   #
############################                         
import numpy as np
import matplotlib.pyplot as plt
"""
This code sets up a Kalman Filter with some initial parameters and runs a simple prediction and update loop. 
You can adjust the matrices F, B, H, Q, R, and P to fit the dynamics of your system, 
and the initial state x0 to match your starting conditions. 
The predict method advances the state based on the system dynamics and control input, 
while the update method adjusts the state based on the new measurement z.
"""

import numpy as np

class KalmanFilter(object):
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        self.n = F.shape[1]
        self.F = F
        self.H = H
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0

    def predict(self, u=None):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x

# Define the initial parameters
F = np.array([[1, 1], [0, 1]])    # State transition matrix
H = np.array([[1, 0],              # Observation matrix for radar
              [1, 0]])             # Observation matrix for camera
B = np.array([[0], [1]])           # Control input matrix
Q = np.array([[1, 0], [0, 1]])    # Process noise covariance
R = np.array([[1, 0],              # Measurement noise covariance for radar
              [0, 1]])             # Measurement noise covariance for camera
P = np.array([[1, 0], [0, 1]])    # Initial state covariance
x0 = np.array([[0], [1]])         # Initial state

kf = KalmanFilter(F=F, B=B, H=H, Q=Q, R=R, P=P, x0=x0)

# Example usage
u = np.array([[1]])               # Control input
z_radar = np.array([[np.random.rand()]]) # Radar measurement
z_camera = np.array([[np.random.rand()]]) # Camera measurement
z = np.vstack((z_radar, z_camera)) # Combined measurement vector


pred_list = []
update_list = []
for _ in range(10):
    pred = kf.predict(u=u)
    update = kf.update(z=z)
    # filling up the radar measurement
    pred_list.append(pred[0][0])
    update_list.append(update[0][0])

    #print(f"Predicted state: {pred}")
    #print(f"Updated state: {update}")

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(pred_list)
plt.title("Prediction_radar")

plt.subplot(2,1,2)
plt.plot(update_list)
plt.title("Update_radar")
plt.show()


