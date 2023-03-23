#pragma once

#include "mat4.h"
#include "simple_polygon_mesh.h"
#include "transform.h"

#include <memory>

namespace renderer {

using geometry::SimplePolygonMesh;
using geometry::Transform;
using geometry::Vector3;

struct RenderObject {

    std::shared_ptr<Transform> trans;
    std::shared_ptr<SimplePolygonMesh> mesh;
    std::string texture_file;
    float alpha;

    RenderObject() = default;
    RenderObject(std::string const& meshFilename, std::string const& texFilename, Vector3 loc, Vector3 euler, Vector3 s, float opacity);

};

}
