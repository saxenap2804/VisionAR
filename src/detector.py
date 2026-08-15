import cv2


class FeatureDetector:
    """Detect ORB keypoints and descriptors in an image."""

    def __init__(self, max_features=1500):
        self.orb = cv2.ORB_create(nfeatures=max_features)

    def detect(self, image):
        if image is None:
            raise ValueError("Input image cannot be None.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.orb.detectAndCompute(
            gray,
            None
        )

        return keypoints, descriptors