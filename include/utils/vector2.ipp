
namespace geometry{

inline Vector2 Vector2::operator+(const Vector2 &v) const {
    return Vector2 { v.x + x, y + v.y };
}

inline Vector2 Vector2::operator-(const Vector2 &v) const {
    return Vector2 { x - v.x, y - v.y };
}

inline Vector2 Vector2::operator*(const Vector2 &v) const {
    return Vector2 { x * v.x - y * v.y, x * v.y + y * v.x };
}

inline Vector2 Vector2::operator/(const Vector2 &v) const {
    const double denom = v.x * v.x + v.y * v.y;
    return Vector2 { x * v.x + y * v.y, y * v.x - x * v.y } / denom;
}

inline Vector2 Vector2::operator*(float s) const {
    return Vector2{ x * s, y * s };
}

inline Vector2 Vector2::operator/(float s) const {
    const float r = 1. / s;
    return Vector2{ x * r, y * r };
}

inline Vector2& Vector2::operator-=(const Vector2 &v) {
    x -= v.x;
    y -= v.y;
    return *this;
}

inline Vector2& Vector2::operator+=(const Vector2 &v) {
    x += v.x;
    y += v.y;
    return *this;
}

inline Vector2& Vector2::operator*=(const Vector2 &v) {
    Vector2 tmp = *this * v;
    *this = tmp;
    return *this;
}

inline Vector2& Vector2::operator/=(const Vector2 &v) {
    Vector2 tmp = *this / v;
    *this = tmp;
    return *this;
}

inline Vector2& Vector2::operator*=(const float s) {
    x *= s;
    y *= s;
    return *this;
}

inline Vector2& Vector2::operator/=(const float s) {
    x /= s;
    y /= s;
    return *this;
}

inline bool Vector2::operator==(const Vector2 &v) const {
    return x == v.x && y == v.y;
}

inline bool Vector2::operator!=(const Vector2 &v) const {
    return x != v.x || y != v.y;
}

inline Vector2 Vector2::operator-() const {
    return Vector2{-x, -y};
}

inline Vector2 Vector2::normalize() const {
    float r = 1. / std::sqrt(x * x + y * y);
    return *this * r;
}

}