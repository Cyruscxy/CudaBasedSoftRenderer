#pragma once

#include <limits>
#include <cmath>
#include <sstream>

namespace geometry {

struct Vector3 {
    float x, y, z;

    static Vector3 zero() { return Vector3 { 0.0, 0.0, 0.0 }; }
    static Vector3 constant(float c) { return Vector3 { c, c, c }; }
    static Vector3 infinity() {
        const float inf = std::numeric_limits<float>::infinity();
        return Vector3 { inf, inf, inf };
    }
    static Vector3 undefined() {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        return Vector3 { nan, nan, nan };
    }

    // Access-by-index
    float& operator[](int index) { return (&x)[index]; }
    float operator[](int index) const { return (&x)[index]; }

    // Overload operators
    Vector3 operator+(const Vector3& v) const;
    Vector3 operator-(const Vector3& v) const;
    Vector3 operator*(float s) const;
    Vector3 operator/(float s) const;
    Vector3& operator+=(const Vector3& v);
    Vector3& operator-=(const Vector3& v);
    Vector3& operator*=(const float& s);
    Vector3& operator/=(const float& s);
    bool operator==(const Vector3& v) const;
    bool operator!=(const Vector3& v) const;
    const Vector3 operator-() const;
    Vector3& normalize();

    float norm() const;
    float norm2() const;
};

template<typename T>
Vector3 operator*(const T s, const Vector3& v);

std::ostream& operator<<(std::ostream& output, const Vector3& v);
std::istream& operator>>(std::istream& input, const Vector3& v);



}

#include "vector3.ipp"