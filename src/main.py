import cv2
import numpy as np

from detector import FeatureDetector
from matcher import FeatureMatcher
from homography import estimate_homography
from pose_estimator import create_camera_matrix
from projection import projection_matrix
from obj_loader import OBJModel
from renderer import render_obj
from kalman import CornerKalmanFilter


MIN_MATCHES = 12


def homography_from_corners(marker_corners, filtered_corners):
    """
    Compute a homography using the original marker corners
    and the Kalman-filtered detected corners.
    """

    src = marker_corners.reshape(4, 2).astype(np.float32)
    dst = filtered_corners.reshape(4, 2).astype(np.float32)

    homography, _ = cv2.findHomography(
        src,
        dst,
        0,
    )

    return homography


def main():
    detector = FeatureDetector()
    matcher = FeatureMatcher()
    kalman_filter = CornerKalmanFilter()

    marker = cv2.imread("assets/markers/marker.png")

    if marker is None:
        raise FileNotFoundError(
            "Could not load assets/markers/marker.png"
        )

    obj = OBJModel(
        "assets/models/model.obj"
    )

    marker_keypoints, marker_descriptors = detector.detect(
        marker
    )

    marker_height, marker_width = marker.shape[:2]

    marker_corners = np.float32(
        [
            [0, 0],
            [marker_width - 1, 0],
            [marker_width - 1, marker_height - 1],
            [0, marker_height - 1],
        ]
    ).reshape(-1, 1, 2)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to access the webcam."
        )

    success, first_frame = cap.read()

    if not success:
        raise RuntimeError(
            "Unable to read webcam frame."
        )

    frame_height, frame_width = first_frame.shape[:2]

    camera_matrix = create_camera_matrix(
        frame_width,
        frame_height,
    )

    print(
        f"Marker keypoints: {len(marker_keypoints)}"
    )

    print(
        f"OBJ vertices: {len(obj.vertices)}"
    )

    print(
        f"OBJ faces: {len(obj.faces)}"
    )

    print(
        "VisionAR Kalman tracking started."
    )

    print(
        "Press Q to quit."
    )

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_keypoints, frame_descriptors = detector.detect(
            frame
        )

        matches = matcher.match(
            marker_descriptors,
            frame_descriptors,
        )

        best_matches = matches[:50]

        raw_homography, mask = estimate_homography(
            marker_keypoints,
            frame_keypoints,
            best_matches,
            MIN_MATCHES,
        )

        status = "Marker not detected"

        filtered_corners = None

        # ------------------------------
        # Marker detected
        # ------------------------------
        if raw_homography is not None:

            detected_corners = cv2.perspectiveTransform(
                marker_corners,
                raw_homography,
            )

            # Initialize filter on first successful detection
            if not kalman_filter.initialized:
                kalman_filter.initialize(
                    detected_corners
                )

            # Predict next location
            kalman_filter.predict()

            # Correct prediction using current detection
            filtered_corners = kalman_filter.correct(
                detected_corners
            )

            status = "Kalman tracking active"

        # ------------------------------
        # Marker temporarily lost
        # ------------------------------
        elif kalman_filter.initialized:

            filtered_corners = kalman_filter.predict()

            status = "Kalman prediction"

        # ------------------------------
        # Render using filtered corners
        # ------------------------------
        if filtered_corners is not None:

            filtered_homography = homography_from_corners(
                marker_corners,
                filtered_corners,
            )

            if filtered_homography is not None:

                projection = projection_matrix(
                    camera_matrix,
                    filtered_homography,
                )

                frame = render_obj(
                    frame,
                    obj,
                    projection,
                    marker_width,
                    marker_height,
                    scale=3.0,
                )

                # Draw filtered marker boundary
                frame = cv2.polylines(
                    frame,
                    [
                        np.int32(
                            filtered_corners
                        )
                    ],
                    True,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # ------------------------------
        # UI
        # ------------------------------
        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (
                (0, 255, 0)
                if status == "Kalman tracking active"
                else (0, 255, 255)
            ),
            2,
        )

        cv2.putText(
            frame,
            f"Matches: {len(matches)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "VisionAR - Kalman Stabilized AR",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()