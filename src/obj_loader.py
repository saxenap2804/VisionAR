class OBJModel:
    """Simple Wavefront OBJ loader for vertices and faces."""

    def __init__(self, filename):
        self.vertices = []
        self.faces = []

        self._load(filename)

    def _load(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("#") or not line.strip():
                    continue

                values = line.strip().split()

                if not values:
                    continue

                if values[0] == "v":
                    vertex = list(map(float, values[1:4]))
                    self.vertices.append(vertex)

                elif values[0] == "f":
                    face = []

                    for value in values[1:]:
                        vertex_index = value.split("/")[0]
                        face.append(int(vertex_index))

                    self.faces.append(face)