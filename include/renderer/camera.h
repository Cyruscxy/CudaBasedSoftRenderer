#pragma once

#include "mat4.h"
#include "transform.h"

namespace renderer {

using geometry::Vector3;
using geometry::Transform;
using geometry::Mat4;
struct Camera {
    float vertical_fov = 60.0f;
    float aspect_ratio = 1.77778f;
    float near_plane = 0.1f;
    std::shared_ptr<Transform> trans;

    Camera() = default;
    Camera(Vector3 loc, Vector3 euler, Vector3 s) { trans = std::make_shared<Transform>(loc, euler, s); }
    Mat4 projection() const { return Mat4::perspective(vertical_fov, aspect_ratio, near_plane); };
};

}
