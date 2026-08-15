import math
import numpy as np


def projection_matrix(camera_matrix, homography):
    """
    Compute a 3D projection matrix from the
    camera calibration matrix and homography.
    """

    homography = homography * -1

    rotation_translation = np.dot(
        np.linalg.inv(camera_matrix),
        homography,
    )

    column_1 = rotation_translation[:, 0]
    column_2 = rotation_translation[:, 1]
    column_3 = rotation_translation[:, 2]

    scale = math.sqrt(
        np.linalg.norm(column_1, 2)
        * np.linalg.norm(column_2, 2)
    )

    rotation_1 = column_1 / scale
    rotation_2 = column_2 / scale
    translation = column_3 / scale

    c = rotation_1 + rotation_2

    p = np.cross(
        rotation_1,
        rotation_2,
    )

    d = np.cross(
        c,
        p,
    )

    rotation_1 = (
        c / np.linalg.norm(c, 2)
        + d / np.linalg.norm(d, 2)
    ) / math.sqrt(2)

    rotation_2 = (
        c / np.linalg.norm(c, 2)
        - d / np.linalg.norm(d, 2)
    ) / math.sqrt(2)

    rotation_3 = np.cross(
        rotation_1,
        rotation_2,
    )

    projection = np.stack(
        (
            rotation_1,
            rotation_2,
            rotation_3,
            translation,
        )
    ).T

    return np.dot(
        camera_matrix,
        projection,
    )