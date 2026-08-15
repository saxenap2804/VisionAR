import cv2
import numpy as np


def draw_cube(
    frame,
    rotation_vector,
    translation_vector,
    camera_matrix,
    marker_width,
    marker_height,
):
    """Project and draw a 3D cube on top of the marker."""

    size = min(marker_width, marker_height) * 0.5

    x_offset = (marker_width - size) / 2
    y_offset = (marker_height - size) / 2

    cube_points = np.float32(
        [
            [x_offset, y_offset, 0],
            [x_offset + size, y_offset, 0],
            [x_offset + size, y_offset + size, 0],
            [x_offset, y_offset + size, 0],

            [x_offset, y_offset, -size],
            [x_offset + size, y_offset, -size],
            [x_offset + size, y_offset + size, -size],
            [x_offset, y_offset + size, -size],
        ]
    )

    distortion = np.zeros((4, 1), dtype=np.float64)

    projected_points, _ = cv2.projectPoints(
        cube_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )

    points = np.int32(projected_points.reshape(-1, 2))

    # Bottom face
    cv2.polylines(frame, [points[:4]], True, (0, 255, 0), 3)

    # Top face
    cv2.polylines(frame, [points[4:]], True, (255, 0, 0), 3)

    # Vertical edges
    for i in range(4):
        cv2.line(
            frame,
            tuple(points[i]),
            tuple(points[i + 4]),
            (0, 255, 255),
            3,
        )

    return frame