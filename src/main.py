import argparse
import time

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
MAX_BEST_MATCHES = 50


def parse_args():
    """
    Parse VisionAR command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "VisionAR - real-time marker-based augmented reality "
            "using Python and OpenCV"
        )
    )

    parser.add_argument(
        "--marker",
        default="assets/markers/marker.png",
        help="Path to the reference marker image",
    )

    parser.add_argument(
        "--model",
        default="assets/models/model.obj",
        help="Path to the OBJ model",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Scale factor for the 3D model",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index",
    )

    parser.add_argument(
        "--rectangle",
        action="store_true",
        help="Draw the tracked marker boundary",
    )

    parser.add_argument(
        "--matches",
        action="store_true",
        help="Display ORB feature matches between marker and webcam frame",
    )

    return parser.parse_args()


def homography_from_corners(marker_corners, filtered_corners):
    """
    Compute a homography using the original marker corners
    and Kalman-filtered detected corners.
    """

    src = marker_corners.reshape(
        4,
        2,
    ).astype(np.float32)

    dst = filtered_corners.reshape(
        4,
        2,
    ).astype(np.float32)

    homography, _ = cv2.findHomography(
        src,
        dst,
        0,
    )

    return homography


def main():
    # --------------------------------------------------
    # Parse command-line arguments
    # --------------------------------------------------

    args = parse_args()

    # --------------------------------------------------
    # Initialize VisionAR components
    # --------------------------------------------------

    detector = FeatureDetector()
    matcher = FeatureMatcher()
    kalman_filter = CornerKalmanFilter()

    # --------------------------------------------------
    # Load reference marker
    # --------------------------------------------------

    marker = cv2.imread(
        args.marker
    )

    if marker is None:
        raise FileNotFoundError(
            f"Could not load marker: {args.marker}"
        )

    # --------------------------------------------------
    # Load OBJ model
    # --------------------------------------------------

    obj = OBJModel(
        args.model
    )

    # --------------------------------------------------
    # Detect reference marker features
    # --------------------------------------------------

    marker_keypoints, marker_descriptors = detector.detect(
        marker
    )

    if marker_descriptors is None:
        raise RuntimeError(
            "No ORB descriptors were detected in the marker."
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

    # --------------------------------------------------
    # Start webcam
    # --------------------------------------------------

    cap = cv2.VideoCapture(
        args.camera
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to access camera index {args.camera}."
        )

    success, first_frame = cap.read()

    if not success:
        cap.release()

        raise RuntimeError(
            "Unable to read webcam frame."
        )

    frame_height, frame_width = first_frame.shape[:2]

    # --------------------------------------------------
    # Build approximate camera matrix
    # --------------------------------------------------

    camera_matrix = create_camera_matrix(
        frame_width,
        frame_height,
    )

    # --------------------------------------------------
    # FPS tracking
    # --------------------------------------------------

    previous_time = time.perf_counter()
    fps = 0.0

    # --------------------------------------------------
    # Startup information
    # --------------------------------------------------

    print(
        f"Marker: {args.marker}"
    )

    print(
        f"Model: {args.model}"
    )

    print(
        f"Scale: {args.scale}"
    )

    print(
        f"Camera index: {args.camera}"
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
        f"Rectangle overlay: {args.rectangle}"
    )

    print(
        f"Match visualization: {args.matches}"
    )

    print(
        "VisionAR Kalman tracking started."
    )

    print(
        "Press Q to quit."
    )

    # --------------------------------------------------
    # Main AR loop
    # --------------------------------------------------

    while True:
        success, frame = cap.read()

        if not success:
            print(
                "Unable to read webcam frame."
            )
            break

        # ----------------------------------------------
        # Calculate FPS
        # ----------------------------------------------

        current_time = time.perf_counter()

        delta_time = (
            current_time
            - previous_time
        )

        if delta_time > 0:
            current_fps = (
                1.0 / delta_time
            )

            if fps == 0.0:
                fps = current_fps

            else:
                fps = (
                    0.9 * fps
                    + 0.1 * current_fps
                )

        previous_time = current_time

        # ----------------------------------------------
        # Detect ORB features in webcam frame
        # ----------------------------------------------

        frame_keypoints, frame_descriptors = detector.detect(
            frame
        )

        # ----------------------------------------------
        # Match marker against webcam frame
        # ----------------------------------------------

        matches = matcher.match(
            marker_descriptors,
            frame_descriptors,
        )

        best_matches = matches[
            :MAX_BEST_MATCHES
        ]

        # ----------------------------------------------
        # Optional feature-match visualization
        # ----------------------------------------------

        if args.matches:

            match_view = cv2.drawMatches(
                marker,
                marker_keypoints,
                frame,
                frame_keypoints,
                best_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            cv2.imshow(
                "VisionAR - Feature Matches",
                match_view,
            )

        # ----------------------------------------------
        # Estimate homography
        # ----------------------------------------------

        raw_homography, mask = estimate_homography(
            marker_keypoints,
            frame_keypoints,
            best_matches,
            MIN_MATCHES,
        )

        # ----------------------------------------------
        # Calculate RANSAC confidence
        # ----------------------------------------------

        if (
            mask is not None
            and len(mask) > 0
        ):

            inliers = int(
                mask.ravel().sum()
            )

            inlier_ratio = (
                inliers
                / len(mask)
            )

        else:

            inliers = 0
            inlier_ratio = 0.0

        confidence_percent = int(
            inlier_ratio * 100
        )

        status = (
            "Marker not detected"
        )

        filtered_corners = None

        # ==============================================
        # Marker successfully detected
        # ==============================================

        if raw_homography is not None:

            detected_corners = cv2.perspectiveTransform(
                marker_corners,
                raw_homography,
            )

            # Initialize Kalman filter
            if not kalman_filter.initialized:

                kalman_filter.initialize(
                    detected_corners
                )

            # Kalman prediction
            kalman_filter.predict()

            # Kalman correction
            filtered_corners = kalman_filter.correct(
                detected_corners
            )

            status = (
                "Kalman tracking active"
            )

        # ==============================================
        # Marker temporarily lost
        # ==============================================

        elif kalman_filter.initialized:

            filtered_corners = (
                kalman_filter.predict()
            )

            status = (
                "Kalman prediction"
            )

        # ==============================================
        # Render AR model using filtered corners
        # ==============================================

        if filtered_corners is not None:

            filtered_homography = homography_from_corners(
                marker_corners,
                filtered_corners,
            )

            if filtered_homography is not None:

                # --------------------------------------
                # Generate projection matrix
                # --------------------------------------

                projection = projection_matrix(
                    camera_matrix,
                    filtered_homography,
                )

                # --------------------------------------
                # Render OBJ model
                # --------------------------------------

                frame = render_obj(
                    frame,
                    obj,
                    projection,
                    marker_width,
                    marker_height,
                    scale=args.scale,
                )

                # --------------------------------------
                # Marker boundary color
                # --------------------------------------

                if status == "Kalman tracking active":

                    boundary_color = (
                        0,
                        255,
                        0,
                    )

                else:

                    boundary_color = (
                        0,
                        255,
                        255,
                    )

                # --------------------------------------
                # Optional marker rectangle
                # --------------------------------------

                if args.rectangle:

                    frame = cv2.polylines(
                        frame,
                        [
                            np.int32(
                                filtered_corners
                            )
                        ],
                        True,
                        boundary_color,
                        2,
                        cv2.LINE_AA,
                    )

        # ==============================================
        # Determine status color
        # ==============================================

        if status == "Kalman tracking active":

            status_color = (
                0,
                255,
                0,
            )

        elif status == "Kalman prediction":

            status_color = (
                0,
                255,
                255,
            )

        else:

            status_color = (
                0,
                0,
                255,
            )

        # ==============================================
        # Information overlay
        # ==============================================

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2,
        )

        cv2.putText(
            frame,
            (
                f"Matches: {len(matches)} "
                f"| Inliers: {inliers}"
            ),
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence_percent}%",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # ==============================================
        # Display AR output
        # ==============================================

        cv2.imshow(
            "VisionAR - Kalman Stabilized AR",
            frame,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()