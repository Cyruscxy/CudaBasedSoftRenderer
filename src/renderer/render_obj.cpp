#include "render_obj.h"

namespace renderer {

RenderObject::RenderObject(std::string const& meshFilename, std::string const& texFilename,
                           Vector3 loc, Vector3 euler, Vector3 s, float opacity) {
    mesh = std::make_shared<SimplePolygonMesh>(meshFilename, "obj");
    trans = std::make_shared<Transform>(loc, euler, s);
    texture_file = texFilename;
    alpha = opacity;
}

}