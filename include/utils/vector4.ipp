
namespace geometry {

inline Vector4 Vector4::operator+(const Vector4& v) const {
    return Vector4 { x + v.x, y + v.y, z + v.z, w + v.w };
}

inline Vector4 Vector4::operator-(const Vector4& v) const {
    return Vector4 { x - v.x, y - v.y, z - v.z, w - v.w };
}


inline Vector4 Vector4::operator*(const float s) const {
    return Vector4 { x * s, y * s, z * s, w * s };
}

inline Vector4 Vector4::operator/(const float s) const {
    return Vector4 { x / s, y / s, z / s, w / s };
}

inline Vector4& Vector4::operator+=(const Vector4& v) {
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
};

inline Vector4& Vector4::operator-=(const Vector4& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
};

inline Vector4& Vector4::operator*=(const float& s) {
    x *= s;
    y *= s;
    z *= s;
    w *= s;
    return *this;
};

inline Vector4& Vector4::operator/=(const float& s) {
    x /= s;
    y /= s;
    z /= s;
    w /= s;
    return *this;
};

inline bool Vector4::operator==(const Vector4 &v) const {
    return x == v.x && y == v.y && z == v.z && w == v.w;
}

inline bool Vector4::operator!=(const Vector4 &v) const {
    return !operator==(v);
}

inline const Vector4 Vector4::operator-() const {
    return Vector4 { -x, -y, -z, -w };
}

inline Vector3 Vector4::xzy() {
    return Vector3{ x, y, z };
}

inline std::ostream& operator<<(std::ostream& output, const Vector4& v) {
    output << "<" << v.x << "," << v.y << "," << v.z << "," << v.w << ">";
    return output;
}

inline std::istream& operator>>(std::istream& input, Vector4& v) {
    float x, y, z, w;
    input >> x >> y >> z >> w;
    v = Vector4 {x, y, z, w};
    return input;
}

}