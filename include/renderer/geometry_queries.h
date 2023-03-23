#pragma once

__device__ __inline__ bool
vertexInBox( float4 vertex ) {
    return  -vertex.w < vertex.x && vertex.x < vertex.w &&
            -vertex.w < vertex.y && vertex.y < vertex.w &&
            -vertex.w < vertex.z && vertex.z < vertex.w ;
}

__device__ __inline__ uint32_t
lineIntersectsWithTileTest( float left, float right, float bottom, float top, float x0, float y0, float x1, float y1 ) {
    float2 d = make_float2( x1 - x0, y1 - y0 );
    uint32_t intersected = 0;

    float t;
    float intersected_coord;
    // test upper bound
    t = ( top - y0) / d.y;
    intersected_coord = x0 + t * d.x;
    intersected |= 0.0f <= t && t <= 1.0f && left <= intersected_coord && intersected_coord <= right;

    // test lower bound
    t = ( bottom - y0 ) / d.y;
    intersected_coord = x0 + t * d.x;
    intersected |= 0.0f <= t && t <= 1.0f && left <= intersected_coord && intersected_coord <= right;

    // test left bound
    t = ( left - x0 ) / d.x;
    intersected_coord = y0 + t * d.y;
    intersected |= 0.0f <= t && t <= 1.0f && bottom <= intersected_coord && intersected_coord <= top;

    // test right bound
    t = ( right - x0 ) / d.x;
    intersected_coord = y0 + t * d.y;
    intersected |= 0.0f <= t && t <= 1.0f && bottom <= intersected_coord && intersected_coord <= top;

    return intersected;
}

__device__ __inline__ bool
pointInTriangleTest( float4 v0, float4 v1, float4 v2, float x, float y ) {
    float2 vec0 = make_float2(v1.x - v0.x, v1.y - v0.y);
    float2 vec1 = make_float2(v2.x - v1.x, v2.y - v1.y);
    float2 vec2 = make_float2(v0.x - v2.x, v0.y - v2.y);

    return  (( y - v0.y ) * vec0.x - ( x - v0.x ) * vec0.y) >= 0.0f &&
            (( y - v1.y ) * vec1.x - ( x - v1.x ) * vec1.y) >= 0.0f &&
            (( y - v2.y ) * vec2.x - ( x - v2.x ) * vec2.y) >= 0.0f;
}

__device__ __inline__ uint32_t
pointInTile( float left, float right, float bottom, float top, float x, float y ) {
    return x >= left && x <= right && y >= bottom && y <= top;
}