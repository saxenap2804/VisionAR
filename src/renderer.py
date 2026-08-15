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
def render_obj(
    frame,
    obj,
    projection,
    marker_width,
    marker_height,
    scale=3.0,
):
    """Render an OBJ model on top of the marker."""

    vertices = np.array(
        obj.vertices,
        dtype=np.float32,
    )

    for face in obj.faces:
        points = np.array(
            [
                vertices[index - 1]
                for index in face
            ],
            dtype=np.float32,
        )

        points *= scale

        points[:, 0] += marker_width / 2
        points[:, 1] += marker_height / 2

        projected = cv2.perspectiveTransform(
            points.reshape(-1, 1, 3),
            projection,
        )

        image_points = np.int32(
            projected.reshape(-1, 2)
        )

        cv2.fillConvexPoly(
            frame,
            image_points,
            (137, 27, 211),
        )

    return frame