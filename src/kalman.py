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

        self.x = np.zeros(
            (self.state_size, 1),
            dtype=np.float64,
        )

        self.P = np.eye(
            self.state_size,
            dtype=np.float64,
        )

        self.q = q
        self.r = r

        self.initialized = False

    def get_A(self, dt=1.0):
        """
        Constant-velocity system model.
        """

        A = np.eye(
            self.state_size,
            dtype=np.float64,
        )

        for i in range(8):
            A[i, i + 8] = dt

        return A

    def get_H(self):
        """
        Measurement model.

        We measure only x/y corner positions,
        not velocities.
        """

        H = np.zeros(
            (
                self.measurement_size,
                self.state_size,
            ),
            dtype=np.float64,
        )

        H[:, :8] = np.eye(
            self.measurement_size,
            dtype=np.float64,
        )

        return H

    def get_Q(self):
        """
        Process noise covariance.
        """

        position_noise = np.zeros(
            (8, 8),
            dtype=np.float64,
        )

        velocity_noise = (
            np.eye(8, dtype=np.float64)
            * (self.q ** 2)
        )

        return np.block(
            [
                [
                    position_noise,
                    position_noise,
                ],
                [
                    position_noise,
                    velocity_noise,
                ],
            ]
        )

    def get_R(self):
        """
        Measurement noise covariance.
        """

        return (
            np.eye(
                self.measurement_size,
                dtype=np.float64,
            )
            * (self.r ** 2)
        )

    def initialize(self, corners):
        """
        Initialize state from four detected corners.
        """

        if corners is None:
            raise ValueError(
                "Corners cannot be None."
            )

        positions = np.asarray(
            corners,
            dtype=np.float64,
        ).reshape(8, 1)

        self.x[:8] = positions

        # Initial velocities are zero.
        self.x[8:] = 0.0

        # Reset covariance when initializing.
        self.P = np.eye(
            self.state_size,
            dtype=np.float64,
        )

        self.initialized = True

    def predict(self, dt=1.0):
        """
        Prediction step.

        Returns predicted corner positions.
        """

        if not self.initialized:
            return None

        A = self.get_A(dt)
        Q = self.get_Q()

        # State prediction:
        # x_k = A * x_(k-1)
        self.x = A @ self.x

        # Covariance prediction:
        # P_k = A * P_(k-1) * A^T + Q
        self.P = (
            A
            @ self.P
            @ A.T
            + Q
        )

        return self.get_corners()

    def correct(self, corners):
        """
        Correction step using detected corner positions.
        """

        if corners is None:
            return self.get_corners()

        if not self.initialized:
            self.initialize(corners)
            return self.get_corners()

        H = self.get_H()
        R = self.get_R()

        z = np.asarray(
            corners,
            dtype=np.float64,
        ).reshape(8, 1)

        # Innovation / residual:
        # y = z - Hx
        innovation = (
            z
            - H @ self.x
        )

        # Innovation covariance:
        # S = HPH^T + R
        innovation_covariance = (
            H
            @ self.P
            @ H.T
            + R
        )

        # Kalman gain:
        # K = PH^T S^-1
        kalman_gain = (
            self.P
            @ H.T
            @ np.linalg.inv(
                innovation_covariance
            )
        )

        # Correct state:
        # x = x + Ky
        self.x = (
            self.x
            + kalman_gain @ innovation
        )

        identity = np.eye(
            self.state_size,
            dtype=np.float64,
        )

        # Correct covariance:
        # P = (I - KH)P
        self.P = (
            identity
            - kalman_gain @ H
        ) @ self.P

        return self.get_corners()

    def get_corners(self):
        """
        Return filtered corner positions
        in OpenCV-compatible format.

        Shape:
            (4, 1, 2)
        """

        if not self.initialized:
            return None

        return (
            self.x[:8]
            .reshape(4, 1, 2)
            .astype(np.float32)
        )

    def reset(self):
        """
        Reset the Kalman filter to its initial state.
        """

        self.x = np.zeros(
            (self.state_size, 1),
            dtype=np.float64,
        )

        self.P = np.eye(
            self.state_size,
            dtype=np.float64,
        )

        self.initialized = False