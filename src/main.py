import cv2
import numpy as np

from detector import FeatureDetector
from matcher import FeatureMatcher
from homography import estimate_homography


MIN_MATCHES = 12


def main():
    detector = FeatureDetector()
    matcher = FeatureMatcher()

    marker = cv2.imread("assets/markers/marker.png")

    if marker is None:
        raise FileNotFoundError(
            "Could not load assets/markers/marker.png"
        )

    marker_keypoints, marker_descriptors = detector.detect(marker)

    print(f"Marker keypoints: {len(marker_keypoints)}")

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
        raise RuntimeError("Unable to access the webcam.")

    print("VisionAR homography detection started.")
    print("Press Q to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_keypoints, frame_descriptors = detector.detect(frame)

        matches = matcher.match(
            marker_descriptors,
            frame_descriptors
        )

        best_matches = matches[:50]

        homography, mask = estimate_homography(
            marker_keypoints,
            frame_keypoints,
            best_matches,
            MIN_MATCHES
        )

        status = "Marker not detected"

        if homography is not None:
            projected_corners = cv2.perspectiveTransform(
                marker_corners,
                homography
            )

            frame = cv2.polylines(
                frame,
                [np.int32(projected_corners)],
                True,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

            status = "Marker detected"

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if homography is not None else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Matches: {len(matches)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "VisionAR - Homography Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()