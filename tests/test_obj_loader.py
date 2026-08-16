import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from obj_loader import OBJModel


def test_load_model():
    model_path = ROOT / "assets" / "models" / "model.obj"

    model = OBJModel(str(model_path))

    assert model is not None
    assert len(model.vertices) > 0
    assert len(model.faces) > 0


def test_model_vertex_count():
    model_path = ROOT / "assets" / "models" / "model.obj"

    model = OBJModel(str(model_path))

    # Our current pyramid has 5 vertices.
    assert len(model.vertices) == 5


def test_model_face_count():
    model_path = ROOT / "assets" / "models" / "model.obj"

    model = OBJModel(str(model_path))

    # Our current pyramid has 5 faces.
    assert len(model.faces) == 5


def test_vertices_have_three_coordinates():
    model_path = ROOT / "assets" / "models" / "model.obj"

    model = OBJModel(str(model_path))

    for vertex in model.vertices:
        assert len(vertex) == 3


def test_face_indices_are_valid():
    model_path = ROOT / "assets" / "models" / "model.obj"

    model = OBJModel(str(model_path))

    vertex_count = len(model.vertices)

    for face in model.faces:
        for index in face:
            assert 1 <= index <= vertex_count