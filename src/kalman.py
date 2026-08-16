import numpy as np


class CornerKalmanFilter:
    """
    Kalman filter for tracking the four marker corners.

    State:
        [x1, y1, x2, y2, x3, y3, x4, y4,
         vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4]
    """

    def __init__(self, q=0.3, r=0.6):
        self.state_size = 16
        self.measurement_size = 8

        self.x = np.zeros((16, 1), dtype=np.float64)

        self.P = np.eye(16, dtype=np.float64)

        self.q = q
        self.r = r

        self.initialized = False

    def get_A(self, dt=1.0):
        """
        Constant-velocity system model.
        """

        A = np.eye(16, dtype=np.float64)

        for i in range(8):
            A[i, i + 8] = dt

        return A

    def get_H(self):
        """
        Measurement model.

        We measure only x/y corner positions,
        not velocities.
        """

        H = np.zeros((8, 16), dtype=np.float64)

        H[:, :8] = np.eye(8)

        return H

    def get_Q(self):
        """
        Process noise covariance.
        """

        position_noise = np.zeros((8, 8))

        velocity_noise = np.eye(8) * (self.q ** 2)

        return np.block(
            [
                [position_noise, position_noise],
                [position_noise, velocity_noise],
            ]
        )

    def get_R(self):
        """
        Measurement noise covariance.
        """

        return np.eye(8) * (self.r ** 2)

    def initialize(self, corners):
        """
        Initialize state from four detected corners.
        """

        positions = corners.reshape(8, 1).astype(np.float64)

        self.x[:8] = positions
        self.x[8:] = 0

        self.initialized = True

    def predict(self, dt=1.0):
        """
        Prediction step.
        """

        A = self.get_A(dt)
        Q = self.get_Q()

        self.x = A @ self.x

        self.P = (
            A @ self.P @ A.T
            + Q
        )

        return self.get_corners()

    def correct(self, corners):
        """
        Correction step using detected corner positions.
        """

        H = self.get_H()
        R = self.get_R()

        z = corners.reshape(8, 1).astype(np.float64)

        innovation = z - H @ self.x

        innovation_covariance = (
            H @ self.P @ H.T
            + R
        )

        kalman_gain = (
            self.P
            @ H.T
            @ np.linalg.inv(innovation_covariance)
        )

        self.x = (
            self.x
            + kalman_gain @ innovation
        )

        identity = np.eye(16)

        self.P = (
            identity
            - kalman_gain @ H
        ) @ self.P

        return self.get_corners()

    def get_corners(self):
        """
        Return filtered corner positions
        in OpenCV-compatible format.
        """

        return self.x[:8].reshape(4, 1, 2).astype(
            np.float32
        )