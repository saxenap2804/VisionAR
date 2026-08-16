import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kalman import CornerKalmanFilter


def sample_corners():
    return np.array(
        [
            [[100.0, 100.0]],
            [[200.0, 100.0]],
            [[200.0, 200.0]],
            [[100.0, 200.0]],
        ],
        dtype=np.float32,
    )


def test_initialization():
    kalman = CornerKalmanFilter()

    corners = sample_corners()

    kalman.initialize(corners)

    assert kalman.initialized is True

    output = kalman.get_corners()

    assert output.shape == (4, 1, 2)

    np.testing.assert_allclose(
        output,
        corners,
        atol=1e-5,
    )


def test_predict_returns_valid_corners():
    kalman = CornerKalmanFilter()

    corners = sample_corners()

    kalman.initialize(corners)

    predicted = kalman.predict()

    assert predicted is not None
    assert predicted.shape == (4, 1, 2)


def test_correct_moves_estimate_toward_measurement():
    kalman = CornerKalmanFilter()

    original = sample_corners()

    kalman.initialize(original)

    kalman.predict()

    measurement = original.copy()
    measurement[:, :, 0] += 10.0

    before = kalman.get_corners().copy()

    corrected = kalman.correct(measurement)

    assert corrected.shape == (4, 1, 2)

    before_error = np.mean(
        np.abs(
            measurement[:, :, 0]
            - before[:, :, 0]
        )
    )

    corrected_error = np.mean(
        np.abs(
            measurement[:, :, 0]
            - corrected[:, :, 0]
        )
    )

    assert corrected_error < before_error


def test_reset():
    kalman = CornerKalmanFilter()

    kalman.initialize(
        sample_corners()
    )

    kalman.reset()

    assert kalman.initialized is False
    assert kalman.get_corners() is None