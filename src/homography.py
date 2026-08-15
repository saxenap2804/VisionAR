import cv2
import numpy as np


def estimate_homography(
    reference_keypoints,
    frame_keypoints,
    matches,
    min_matches=12
):
    """
    Estimate homography between the reference marker and webcam frame.

    Returns:
        homography: 3x3 transformation matrix or None
        mask: RANSAC inlier mask or None
    """

    if len(matches) < min_matches:
        return None, None

    src_points = np.float32(
        [
            reference_keypoints[m.queryIdx].pt
            for m in matches
        ]
    ).reshape(-1, 1, 2)

    dst_points = np.float32(
        [
            frame_keypoints[m.trainIdx].pt
            for m in matches
        ]
    ).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        5.0
    )

    return homography, mask