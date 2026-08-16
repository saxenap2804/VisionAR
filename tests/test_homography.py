import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from homography import estimate_homography


def create_keypoints(points):
    """Convert coordinate pairs into OpenCV KeyPoint objects."""

    return [
        cv2.KeyPoint(float(x), float(y), 1)
        for x, y in points
    ]


def create_matches(count):
    """Create simple one-to-one OpenCV matches."""

    return [
        cv2.DMatch(i, i, 0.0)
        for i in range(count)
    ]


def test_identity_homography():
    points = [
        (0, 0),
        (100, 0),
        (100, 100),
        (0, 100),
    ]

    keypoints_1 = create_keypoints(points)
    keypoints_2 = create_keypoints(points)

    matches = create_matches(4)

    homography, mask = estimate_homography(
        keypoints_1,
        keypoints_2,
        matches,
        min_matches=4,
    )

    assert homography is not None

    normalized = (
        homography / homography[2, 2]
    )

    np.testing.assert_allclose(
        normalized,
        np.eye(3),
        atol=1e-5,
    )


def test_translation_homography():
    source = [
        (0, 0),
        (100, 0),
        (100, 100),
        (0, 100),
    ]

    destination = [
        (20, 30),
        (120, 30),
        (120, 130),
        (20, 130),
    ]

    source_keypoints = create_keypoints(source)
    destination_keypoints = create_keypoints(destination)

    matches = create_matches(4)

    homography, mask = estimate_homography(
        source_keypoints,
        destination_keypoints,
        matches,
        min_matches=4,
    )

    assert homography is not None

    point = np.float32(
        [[[50, 50]]]
    )

    transformed = cv2.perspectiveTransform(
        point,
        homography,
    )

    expected = np.float32(
        [[[70, 80]]]
    )

    np.testing.assert_allclose(
        transformed,
        expected,
        atol=1e-3,
    )


def test_insufficient_matches():
    points = [
        (0, 0),
        (100, 0),
        (100, 100),
    ]

    keypoints_1 = create_keypoints(points)
    keypoints_2 = create_keypoints(points)

    matches = create_matches(3)

    homography, mask = estimate_homography(
        keypoints_1,
        keypoints_2,
        matches,
        min_matches=4,
    )

    assert homography is None