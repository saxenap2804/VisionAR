import cv2
import numpy as np


def create_camera_matrix(frame_width, frame_height):
    """Create an approximate camera intrinsic matrix."""

    focal_length = frame_width

    return np.array(
        [
            [focal_length, 0, frame_width / 2],
            [0, focal_length, frame_height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def estimate_pose(marker_width, marker_height, image_corners, camera_matrix):
    """Estimate marker pose from its four projected corners."""

    object_points = np.array(
        [
            [0, 0, 0],
            [marker_width, 0, 0],
            [marker_width, marker_height, 0],
            [0, marker_height, 0],
        ],
        dtype=np.float32,
    )

    image_points = image_corners.reshape(4, 2).astype(np.float32)

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
    )

    if not success:
        return None, None

    return rotation_vector, translation_vector