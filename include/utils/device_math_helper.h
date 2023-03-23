#pragma once

#include "mat4.h"

__device__ __inline__ float4
operator*(float s, float4 v) {
    return make_float4(s * v.x, s * v.y, s * v.z, s* v.w);
}

__device__ __inline__ float4
operator*(float4 v, float s) {
    return s * v;
}

__device__ __inline__ float3
operator*(float s, float3 v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ __inline__ float3
operator*(float3 v, float s) {
    return s * v;
}

__device__ __inline__ float2
operator*(float s, float2 v) {
    return make_float2(s * v.x, s * v.y);
}

__device__ __inline__ float2
operator*(float2 v, float s) {
    return s * v;
}

__device__ __inline__ float4
operator+(float4 rhs, float4 lhs) {
    return make_float4(rhs.x + lhs.x, rhs.y + lhs.y, rhs.z + lhs.z, rhs.w + lhs.w);
}

__device__ __inline__ float4
operator-(float4 rhs, float4 lhs) {
    return make_float4(rhs.x - lhs.x, rhs.y - lhs.y, rhs.z - lhs.z, rhs.w - lhs.w);
}

__device__ __inline__ float3
operator+(float3 rhs, float3 lhs) {
    return make_float3(rhs.x + lhs.x, rhs.y + lhs.y, rhs.z + lhs.z);
}

__device__ __inline__ float3
operator-(float3 rhs, float3 lhs) {
    return make_float3(rhs.x - lhs.x, rhs.y - lhs.y, rhs.z - lhs.z);
}


__device__ __inline__ float2
operator+(float2 rhs, float2 lhs) {
    return make_float2(rhs.x + lhs.x, rhs.y + lhs.y);
}

__device__ __inline__ float2
operator-(float2 rhs, float2 lhs) {
    return make_float2(rhs.x - lhs.x, rhs.y - lhs.y);
}

__device__ __inline__ float4
operator-(float4 v) {
    return make_float4(-v.x, -v.y, -v.z, -v.w);
}

__device__ __inline__ float3
operator-(float3 v) {
    return make_float3(-v.x, -v.y, -v.z);
}

__device__ __inline__ float2
operator-(float2 v) {
    return make_float2(-v.x, -v.y);
}

__device__ __inline__ float
dot(float4 rhs, float4 lhs) {
    return rhs.x * lhs.x + rhs.y * lhs.y + rhs.z * lhs.z + rhs.w * lhs.w;
}

__device__ __inline__ float
dot(float3 rhs, float3 lhs) {
    return rhs.x * lhs.x + rhs.y * lhs.y + rhs.z * lhs.z;
}

__device__ __inline__ float
dot(float2 rhs, float2 lhs) {
    return rhs.x * lhs.x + rhs.y * lhs.y;
}

__device__ __inline__ float4
operator*(Mat4 m, float4 v) {
    return make_float4(
            v.x * m.cols[0].x + v.y * m.cols[1].x + v.z * m.cols[2].x + v.w * m.cols[3].x,
            v.x * m.cols[0].y + v.y * m.cols[1].y + v.z * m.cols[2].y + v.w * m.cols[3].y,
            v.x * m.cols[0].z + v.y * m.cols[1].z + v.z * m.cols[2].z + v.w * m.cols[3].z,
            v.x * m.cols[0].w + v.y * m.cols[1].w + v.z * m.cols[2].w + v.w * m.cols[3].w);
}

__device__ __inline__ float4
fromVec3(float3 xyz, float w) {
    return make_float4(xyz.x, xyz.y, xyz.z, w);
}

__device__ __inline__ float3
xyz(float4 v) {
    return make_float3( v.x, v.y, v.z );
}


