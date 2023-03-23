#pragma once

#include <cmath>
#include <sstream>
#include "vector3.h"

namespace geometry {

struct Vector4 {
    float x, y, z, w;

    static Vector4 zero() { return Vector4 { 0., 0., 0., 0. }; }
    static Vector4 constant(float c) { return Vector4 { c, c, c, c }; }
    static Vector4 fromVec3( Vector3 v, float c ) { return Vector4 { v.x, v.y, v.z, c }; }

    // Access-by-index
    float& operator[](int index) { return (&x)[index]; }
    float operator[](int index) const { return (&x)[index]; }

    // Mutate
    Vector4 operator+(const Vector4& v) const;
    Vector4 operator-(const Vector4& v) const;
    Vector4 operator*(float s) const;
    Vector4 operator/(float s) const;
    Vector4& operator+=(const Vector4& v);
    Vector4& operator-=(const Vector4& v);
    Vector4& operator*=(const float &s);
    Vector4& operator/=(const float &s);
    bool operator==(const Vector4& v) const;
    bool operator!=(const Vector4& v) const;
    const Vector4 operator-() const;
    Vector3 xzy();
};

std::ostream& operator<<(std::ostream& output, const Vector4& v);
std::istream& operator>>(std::istream& input, const Vector4& v);

}

#include "vector4.ipp"