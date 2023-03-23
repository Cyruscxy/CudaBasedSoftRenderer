
namespace geometry {

inline Mat4 Mat4::identity() {
    return Mat4 {
        Vector4 { 1., 0., 0., 0. },
        Vector4 { 0., 1., 0., 0. },
        Vector4 { 0., 0., 1., 0. },
        Vector4 { 0., 0., 0., 1. },
    };
}

inline Mat4 Mat4::zero() {
    return Mat4 {
            Vector4 { 0., 0., 0., 0. },
            Vector4 { 0., 0., 0., 0. },
            Vector4 { 0., 0., 0., 0. },
            Vector4 { 0., 0., 0., 0. },
    };
}

inline Mat4 Mat4::transpose(const Mat4 &m) {
    Mat4 r = zero();
    for ( int i = 0; i < 4; ++i ) {
        for ( int j = 0; j < 4; ++j ) {
            r[i][j] = m[j][i];
        }
    }
    return r;
}

inline Mat4 Mat4::inverse(const Mat4 &m) {
    Mat4 r = zero();
    r[0][0] = m[1][2] * m[2][3] * m[3][1] - m[1][3] * m[2][2] * m[3][1] +
              m[1][3] * m[2][1] * m[3][2] - m[1][1] * m[2][3] * m[3][2] -
              m[1][2] * m[2][1] * m[3][3] + m[1][1] * m[2][2] * m[3][3];
    r[0][1] = m[0][3] * m[2][2] * m[3][1] - m[0][2] * m[2][3] * m[3][1] -
              m[0][3] * m[2][1] * m[3][2] + m[0][1] * m[2][3] * m[3][2] +
              m[0][2] * m[2][1] * m[3][3] - m[0][1] * m[2][2] * m[3][3];
    r[0][2] = m[0][2] * m[1][3] * m[3][1] - m[0][3] * m[1][2] * m[3][1] +
              m[0][3] * m[1][1] * m[3][2] - m[0][1] * m[1][3] * m[3][2] -
              m[0][2] * m[1][1] * m[3][3] + m[0][1] * m[1][2] * m[3][3];
    r[0][3] = m[0][3] * m[1][2] * m[2][1] - m[0][2] * m[1][3] * m[2][1] -
              m[0][3] * m[1][1] * m[2][2] + m[0][1] * m[1][3] * m[2][2] +
              m[0][2] * m[1][1] * m[2][3] - m[0][1] * m[1][2] * m[2][3];
    r[1][0] = m[1][3] * m[2][2] * m[3][0] - m[1][2] * m[2][3] * m[3][0] -
              m[1][3] * m[2][0] * m[3][2] + m[1][0] * m[2][3] * m[3][2] +
              m[1][2] * m[2][0] * m[3][3] - m[1][0] * m[2][2] * m[3][3];
    r[1][1] = m[0][2] * m[2][3] * m[3][0] - m[0][3] * m[2][2] * m[3][0] +
              m[0][3] * m[2][0] * m[3][2] - m[0][0] * m[2][3] * m[3][2] -
              m[0][2] * m[2][0] * m[3][3] + m[0][0] * m[2][2] * m[3][3];
    r[1][2] = m[0][3] * m[1][2] * m[3][0] - m[0][2] * m[1][3] * m[3][0] -
              m[0][3] * m[1][0] * m[3][2] + m[0][0] * m[1][3] * m[3][2] +
              m[0][2] * m[1][0] * m[3][3] - m[0][0] * m[1][2] * m[3][3];
    r[1][3] = m[0][2] * m[1][3] * m[2][0] - m[0][3] * m[1][2] * m[2][0] +
              m[0][3] * m[1][0] * m[2][2] - m[0][0] * m[1][3] * m[2][2] -
              m[0][2] * m[1][0] * m[2][3] + m[0][0] * m[1][2] * m[2][3];
    r[2][0] = m[1][1] * m[2][3] * m[3][0] - m[1][3] * m[2][1] * m[3][0] +
              m[1][3] * m[2][0] * m[3][1] - m[1][0] * m[2][3] * m[3][1] -
              m[1][1] * m[2][0] * m[3][3] + m[1][0] * m[2][1] * m[3][3];
    r[2][1] = m[0][3] * m[2][1] * m[3][0] - m[0][1] * m[2][3] * m[3][0] -
              m[0][3] * m[2][0] * m[3][1] + m[0][0] * m[2][3] * m[3][1] +
              m[0][1] * m[2][0] * m[3][3] - m[0][0] * m[2][1] * m[3][3];
    r[2][2] = m[0][1] * m[1][3] * m[3][0] - m[0][3] * m[1][1] * m[3][0] +
              m[0][3] * m[1][0] * m[3][1] - m[0][0] * m[1][3] * m[3][1] -
              m[0][1] * m[1][0] * m[3][3] + m[0][0] * m[1][1] * m[3][3];
    r[2][3] = m[0][3] * m[1][1] * m[2][0] - m[0][1] * m[1][3] * m[2][0] -
              m[0][3] * m[1][0] * m[2][1] + m[0][0] * m[1][3] * m[2][1] +
              m[0][1] * m[1][0] * m[2][3] - m[0][0] * m[1][1] * m[2][3];
    r[3][0] = m[1][2] * m[2][1] * m[3][0] - m[1][1] * m[2][2] * m[3][0] -
              m[1][2] * m[2][0] * m[3][1] + m[1][0] * m[2][2] * m[3][1] +
              m[1][1] * m[2][0] * m[3][2] - m[1][0] * m[2][1] * m[3][2];
    r[3][1] = m[0][1] * m[2][2] * m[3][0] - m[0][2] * m[2][1] * m[3][0] +
              m[0][2] * m[2][0] * m[3][1] - m[0][0] * m[2][2] * m[3][1] -
              m[0][1] * m[2][0] * m[3][2] + m[0][0] * m[2][1] * m[3][2];
    r[3][2] = m[0][2] * m[1][1] * m[3][0] - m[0][1] * m[1][2] * m[3][0] -
              m[0][2] * m[1][0] * m[3][1] + m[0][0] * m[1][2] * m[3][1] +
              m[0][1] * m[1][0] * m[3][2] - m[0][0] * m[1][1] * m[3][2];
    r[3][3] = m[0][1] * m[1][2] * m[2][0] - m[0][2] * m[1][1] * m[2][0] +
              m[0][2] * m[1][0] * m[2][1] - m[0][0] * m[1][2] * m[2][1] -
              m[0][1] * m[1][0] * m[2][2] + m[0][0] * m[1][1] * m[2][2];
    r /= m.det();
    return r;
}

inline Mat4 Mat4::translate(Vector3 t) {
    Mat4 r = identity();
    r[3] = Vector4::fromVec3(t, 1.f);
    return r;
}

inline Mat4 Mat4::angle_axis(float t, Vector3 axis) {
    Mat4 ret = zero();
    constexpr float PI_F = 3.14159265358979323846264338327950288f;
    float c = std::cos(t * (PI_F / 180.f));
    float s = std::sin(t * (PI_F / 180.f));
    axis.normalize();
    Vector3 temp = axis * (1.f - c);
    ret[0][0] = c + temp[0] * axis[0];
    ret[0][1] = temp[0] * axis[1] + s * axis[2];
    ret[0][2] = temp[0] * axis[2] - s * axis[1];
    ret[1][0] = temp[1] * axis[0] - s * axis[2];
    ret[1][1] = c + temp[1] * axis[1];
    ret[1][2] = temp[1] * axis[2] + s * axis[0];
    ret[2][0] = temp[2] * axis[0] + s * axis[1];
    ret[2][1] = temp[2] * axis[1] - s * axis[0];
    ret[2][2] = c + temp[2] * axis[2];
    return ret;
}

inline Mat4 Mat4::euler(Vector3 angles) {
    return angle_axis(angles.z, Vector3{0.f, 0.f, 1.f}) *
            angle_axis(angles.y, Vector3{0.f, 1.f, 0.f}) *
            angle_axis(angles.x, Vector3{1.f, 0.f, 0.f});
}

inline Mat4 Mat4::scale(Vector3 s) {
    return Mat4 {
        Vector4 { s.x, 0.f, 0.f, 0.f },
        Vector4 { 0.f, s.y, 0.f, 0.f },
        Vector4 { 0.f, 0.f, s.z, 0.f },
        Vector4 { 0.f, 0.f, 0.f, 1.f },
    };
}

inline Mat4 Mat4::perspective(float fov, float ar, float n) {
    constexpr float rad = 3.14159265358979323846264338327950288f / 180.0f;
    float f = 1.0f / std::tan(fov * rad / 2.0f);
    return Mat4 {
        Vector4 { f / ar, 0.f,  0.f,      0.f },
        Vector4 { 0.f,    f,    0.f,      0.f },
        Vector4 { 0.f,    0.f, -1.f,     -1.f },
        Vector4 { 0.f,    0.f, -2.f * n,  0.f },
    };
}

inline double Mat4::det() const {
    return cols[0][3] * cols[1][2] * cols[2][1] * cols[3][0] -
           cols[0][2] * cols[1][3] * cols[2][1] * cols[3][0] -
           cols[0][3] * cols[1][1] * cols[2][2] * cols[3][0] +
           cols[0][1] * cols[1][3] * cols[2][2] * cols[3][0] +
           cols[0][2] * cols[1][1] * cols[2][3] * cols[3][0] -
           cols[0][1] * cols[1][2] * cols[2][3] * cols[3][0] -
           cols[0][3] * cols[1][2] * cols[2][0] * cols[3][1] +
           cols[0][2] * cols[1][3] * cols[2][0] * cols[3][1] +
           cols[0][3] * cols[1][0] * cols[2][2] * cols[3][1] -
           cols[0][0] * cols[1][3] * cols[2][2] * cols[3][1] -
           cols[0][2] * cols[1][0] * cols[2][3] * cols[3][1] +
           cols[0][0] * cols[1][2] * cols[2][3] * cols[3][1] +
           cols[0][3] * cols[1][1] * cols[2][0] * cols[3][2] -
           cols[0][1] * cols[1][3] * cols[2][0] * cols[3][2] -
           cols[0][3] * cols[1][0] * cols[2][1] * cols[3][2] +
           cols[0][0] * cols[1][3] * cols[2][1] * cols[3][2] +
           cols[0][1] * cols[1][0] * cols[2][3] * cols[3][2] -
           cols[0][0] * cols[1][1] * cols[2][3] * cols[3][2] -
           cols[0][2] * cols[1][1] * cols[2][0] * cols[3][3] +
           cols[0][1] * cols[1][2] * cols[2][0] * cols[3][3] +
           cols[0][2] * cols[1][0] * cols[2][1] * cols[3][3] -
           cols[0][0] * cols[1][2] * cols[2][1] * cols[3][3] -
           cols[0][1] * cols[1][0] * cols[2][2] * cols[3][3] +
           cols[0][0] * cols[1][1] * cols[2][2] * cols[3][3];
}

inline Mat4& Mat4::operator/=(double s) {
    for (auto & col : cols) {
        col /= s;
    }
    return *this;
}

inline Mat4 Mat4::operator*(const Mat4 &m) const {
    Mat4 ret = zero();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            ret[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                ret[i][j] += m[i][k] * cols[k][j];
            }
        }
    }
    return ret;
}

inline Vector4 Mat4::operator*(const Vector4 &v) const {
    return cols[0] * v[0] + cols[1] * v[1] + cols[2] * v[2] + cols[3] * v[3];
}

}