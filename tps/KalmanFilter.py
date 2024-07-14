import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas, base_x = 0, base_y = 0):
        self.u = [[u_x], [u_y]]
        self.xk = np.array([[base_x], [base_y], [0], [0]])
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0], [0, (dt**4)/4, 0, (dt**3)/2], [(dt**3)/2, 0, dt**2, 0], [0, (dt**3)/2, 0, dt**2]])
        self.Q = self.Q * std_acc**2
        self.R = np.array([[x_sdt_meas, 0], [0, y_sdt_meas]])
        self.P = np.array([[int(j == i) for j in range(4)] for i in range(4)])
    def predict(self):
        self.xk = self.A @ self.xk + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.xk
    def update(self, zk):
        Sk = self.H @ self.P @ self.H.T + self.R
        Kk = self.P @ self.H.T @ np.linalg.inv(Sk)
        self.xk = self.xk + Kk @ (zk - self.H @ self.xk)
        self.P = (np.identity(len(Kk)) - Kk @ self.H) @ self.P
        return self.xk
