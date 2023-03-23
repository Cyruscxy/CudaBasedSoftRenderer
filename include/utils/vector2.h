#pragma once

#include <cmath>
#include <limits>

namespace geometry {

struct Vector2 {
    float x, y;

    static Vector2 zero() { return Vector2 { 0., 0. }; }
    static Vector2 constant(float c) { return Vector2 { c, c }; }
    static Vector2 infinity() {
        const float inf = std::numeric_limits<float>::infinity();
        return Vector2{inf, inf};
    }
    static Vector2 undefined() {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        return Vector2{nan, nan};
    }

    // Access-by-index
    float& operator[](int index) { return (&x)[index]; }
    float operator[](int index) const { return (&x)[index]; }

    Vector2 operator+(const Vector2& v) const;
    Vector2 operator-(const Vector2& v) const;
    Vector2 operator*(const Vector2& v) const;
    Vector2 operator/(const Vector2& v) const;
    Vector2 operator*(float s) const;
    Vector2 operator/(float s) const;
    Vector2& operator+=(const Vector2& v);
    Vector2& operator-=(const Vector2& v);
    Vector2& operator*=(const Vector2& v);
    Vector2& operator/=(const Vector2& v);
    Vector2& operator*=(const float s);
    Vector2& operator/=(const float s);
    bool operator==(const Vector2& v) const;
    bool operator!=(const Vector2& v) const;
    Vector2 operator-() const;

    Vector2 normalize() const;
};

}

#include "vector2.ipp"
