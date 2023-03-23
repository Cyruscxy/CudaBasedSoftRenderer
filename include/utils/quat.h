#pragma once

#include "vector4.h"
#include "vector3.h"
#include "mat4.h"

struct Quat {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    float w = 0.f;

    using Vector3 = geometry::Vector3;
    using Vector4 = geometry::Vector4;
    using Mat4 = geometry::Mat4;

    static Quat fromComplex(Vector3 complex, float real) {
        return Quat{ complex.x, complex.y, complex.z, real };
    }

    static Quat euler(Vector3 angles) {
        if (angles == Vector3 {0.0f, 0.0f, 180.0f} || angles == Vector3 {180.0f, 0.0f, 0.0f})
            return Quat{0.0f, 0.0f, -1.0f, 0.0f};

        constexpr double rad = 3.14159265358979323846264338327950288f / 180.f;
        float c1 = std::cos(angles[2] * 0.5f * rad);
        float c2 = std::cos(angles[1] * 0.5f * rad);
        float c3 = std::cos(angles[0] * 0.5f * rad);
        float s1 = std::sin(angles[2] * 0.5f * rad);
        float s2 = std::sin(angles[1] * 0.5f * rad);
        float s3 = std::sin(angles[0] * 0.5f * rad);
        float x = c1 * c2 * s3 - s1 * s2 * c3;
        float y = c1 * s2 * c3 + s1 * c2 * s3;
        float z = s1 * c2 * c3 - c1 * s2 * s3;
        float w = c1 * c2 * c3 + s1 * s2 * s3;
        return Quat{x, y, z, w};
    }

    Mat4 to_mat() const {
        return Mat4{
                Vector4 {1 - 2 * y * y - 2 * z * z, 2 * x * y + 2 * z * w, 2 * x * z - 2 * y * w, 0.0f},
                Vector4 {2 * x * y - 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z + 2 * x * w, 0.0f},
                Vector4 {2 * x * z + 2 * y * w, 2 * y * z - 2 * x * w, 1 - 2 * x * x - 2 * y * y, 0.0f},
                Vector4 {0.0f, 0.0f, 0.0f, 1.0f}
        };
    }

    Quat conjugate() const {
        return Quat{ -x, -y, -z, w };
    }

    Quat unit() const {
        float n = std::sqrt(x * x + y * y + z * z + w * w);
        return Quat{ x / n, y / n, z / n, w / n };
    }

    Quat inverse() const {
        return conjugate().unit();
    };

};
