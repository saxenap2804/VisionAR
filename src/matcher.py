import cv2


class FeatureMatcher:
    """Match ORB descriptors using Hamming distance."""

    def __init__(self):
        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )

    def match(self, reference_descriptors, frame_descriptors):
        if reference_descriptors is None or frame_descriptors is None:
            return []

        matches = self.matcher.match(
            reference_descriptors,
            frame_descriptors
        )

        return sorted(matches, key=lambda match: match.distance)