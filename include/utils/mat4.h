#pragma once

#include "vector4.h"

namespace geometry {

struct Mat4 {
    // column major
    Vector4 cols[4];

    static Mat4 identity();
    static Mat4 zero();

    // Matrix operation
    static Mat4 transpose(const Mat4& m);
    static Mat4 inverse(const Mat4& m);
    static Mat4 translate(Vector3 t);
    static Mat4 euler(Vector3 angles);
    static Mat4 angle_axis(float t, Vector3 axis);
    static Mat4 scale(Vector3);
    static Mat4 perspective(float fov, float ar, float n);

    // Access-by-index
    Vector4& operator[](int index) { return cols[index]; }
    Vector4 operator[](int index) const { return cols[index]; }

    double det() const;
    // operator
    Mat4& operator/=(double s);
    Mat4 operator*(const Mat4& m) const;
    Vector4 operator*(const Vector4& v) const;
};

}

#include "mat4.ipp"
