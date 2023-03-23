#pragma once

#include "mat4.h"
#include "quat.h"
#include <memory>

namespace geometry {

using Mat4 = geometry::Mat4;
using Vector3 = geometry::Vector3;

class Transform {
public:
    Transform() = default;
    Transform(Vector3 loc, Vector3 euler, Vector3 s) : location(loc), rotation(Quat::euler(euler)), scale(s) {}

    Mat4 local_to_world();
    Mat4 local_to_parent();
    Mat4 world_to_local();
    Mat4 parent_to_local();

    std::weak_ptr<Transform> parent;
    Vector3 location;
    Quat rotation;
    Vector3 scale;
};


}
