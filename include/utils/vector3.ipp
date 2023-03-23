
namespace geometry{

inline Vector3 Vector3::operator+(const Vector3 &v) const {
    return Vector3 { x + v.x, y + v.y, z + v.z };
}

inline Vector3 Vector3::operator-(const Vector3 &v) const {
    return Vector3 { x - v.x, y - v.y, z - v.z };
}

inline Vector3 Vector3::operator*(float s) const {
    return Vector3 { s * x, s * y, s * z };
}

inline Vector3 Vector3::operator/(float s) const {
    const float r = 1. / s;
    return Vector3 { x * r, y * r, z * r };
}

inline const Vector3 Vector3::operator-() const {
    return Vector3 { -x, -y, -z };
}

template<typename T>
inline Vector3 operator*(const T s, const Vector3& v) {
    return Vector3 { s * v.x, s * v.y, s * v.z };
}

inline Vector3& Vector3::operator+=(const Vector3 &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

inline Vector3& Vector3::operator-=(const Vector3 &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

inline Vector3& Vector3::operator*=(const float &s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
}

inline Vector3 &Vector3::operator/=(const float &s) {
    x /= s;
    y /= s;
    z /= s;
    return *this;
}

inline bool Vector3::operator==(const Vector3 &v) const {
    return x == v.x && y == v.y && z == v.z;
}

inline bool Vector3::operator!=(const Vector3 &v) const {
    return x != v.x || y != v.y || z != v.z;
}

inline float Vector3::norm() const {
    return ::std::sqrt( x * x + y * y + z * z );
}

inline float Vector3::norm2() const {
    return x * x + y * y + z * z;
}

inline Vector3& Vector3::normalize() {
    *this /= norm2();
    return *this;
}

inline std::ostream& operator<<( std::ostream& output, const Vector3& v ) {
    output << "<" << v.x << ", " << v.y << ", " << v.z << ">";
    return output;
}

inline std::istream& operator>>( std::istream& input, Vector3& v ) {
    float x, y, z;
    input >> x >> y >> z;
    v = Vector3 { x, y, z };
    return input;
}

}