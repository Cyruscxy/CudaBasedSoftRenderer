#pragma once

#include "vector4.h"
#include "vector3.h"
#include "vector2.h"
#include "vector_types.h"

using namespace geometry;

struct TriangleFaceIndex {
    uint3 Vertices;
    uint3 TexCoords;
    uint3 Normals;
};

struct FragmentAttribute {
    Vector2 TexCoord;
    Vector3 Normal;
    Vector2 DerivativesU;
    Vector2 DerivativesV;
    float   Depth;
};

struct ShadedFragment {
    Vector4 Color;
    float   Depth;
};

template<typename T>
struct PerPixelLinkedListNode {
    T   data;
    int last_offset;
};

struct AOITNode {
    float3 Color;
    float Depth;
    float Trans;
};

