#pragma once

#include "device_math_helper.h"
#include "geometry_queries.h"
#include "pipeline_parameters.h"
#include "device_structs.h"
#include "device_structs_manage.h"

namespace renderer {

// should be set by host
__constant__ Mat4 device_local_to_clip ;
__constant__ Mat4 device_normal_to_world;
__constant__ float3 clip_to_fb_scale;
__constant__ float3 clip_to_fb_offset;
__constant__ Parameters PipelineParams;

template<typename T>
__device__ __inline__ void
exclusiveScan ( uint32_t tid, T * data ) {
    // exclusive scan
    // up-sweep
    for ( uint32_t e = 1; e < LOG2_BLK_SIZE; e += 1 ) {
        uint32_t step = 1 << (e - 1);
        if ( tid <  (BLK_SIZE >> e) ) {
            data[((tid + 1) << e) - 1] += data[((tid + 1) << e) - 1 - step];
        }
        __syncthreads();
    }

    if ( tid == 0 ) data[BLK_SIZE - 1] = 0;
    __syncthreads();
    // down-sweep
    for ( uint32_t e = LOG2_BLK_SIZE; e > 0; e -= 1 ) {
        uint32_t step = 1 << (e - 1);
        if ( tid < (BLK_SIZE >> e) ) {
            uint32_t tmp = data[((tid + 1) << e) - 1];
            data[((tid + 1) << e) - 1] += data[((tid + 1) << e) - 1 - step];
            data[((tid + 1) << e) - 1 - step] = tmp;
        }
        __syncthreads();
    }
}

template <uint32_t AOIT_NODE_CNT>
__global__ void setDeviceBuffer(
        float*          depth_buffer,
        int32_t *       start_offset_all,
        CuMemPatch *    mem_patch_pipeline,
        CuMemPatch *    mem_patch_sf,
        unsigned char * pre_allocated_mem_pipeline,
        unsigned char * pre_allocated_mem_sf,
        AOITNode *      aoit_nodes,
        uint32_t        width,
        uint32_t        height
) {
    uint32_t pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if ( pixelX >= width || pixelY >= height ) return;
    depth_buffer[pixelX + width * pixelY] = 1.0f;
    start_offset_all[pixelX + width * pixelY] = -1;

    if ( pixelX == 0 && pixelY == 0 ) {
        mem_patch_pipeline->mem = pre_allocated_mem_pipeline;
        mem_patch_pipeline->used = 0;
        mem_patch_pipeline->max_length = Rasterizer::MEM_PATCH_SIZE_PIPELINE;
        mem_patch_pipeline->locked = 0;
    }

    if ( pixelX == 0 && pixelY == 0 ) {
        mem_patch_sf->mem = pre_allocated_mem_sf;
        mem_patch_sf->used = 0;
        mem_patch_sf->max_length = Rasterizer::MEM_PATCH_SIZE_SHADED_FRAGMENT;
        mem_patch_sf->locked = 0;
    }

    auto local_aoit_nodes = aoit_nodes + (pixelX + pixelY * width) * AOIT_NODE_CNT;
    #pragma unroll AOIT_NODE_CNT
    for ( uint32_t i = 0; i < AOIT_NODE_CNT; ++i ) {
        local_aoit_nodes[i].Color = make_float3(0.0f, 0.0f, 0.0f);
        local_aoit_nodes[i].Depth = 1.0f;
        local_aoit_nodes[i].Trans = 1.0f;
    }
}

__global__ void resetDynamicMem(
        FragmentBuffer *    fragment_buffer,
        IntersectionList *  intersection_list,
        CuMemPatch*         mem_patch_pipeline
        ) {
    uint32_t bid_x = blockIdx.x;
    uint32_t bid_y = blockIdx.y;

    auto local_fragment_buffer = fragment_buffer + bid_x + bid_y * gridDim.x;
    for ( uint32_t i = 0; i < local_fragment_buffer->n_nodes; ++i ) {
        local_fragment_buffer->buffer[i] = nullptr;
    }
    local_fragment_buffer->n_nodes = 0;
    local_fragment_buffer->cnt = 0;

    auto local_intersection_list = intersection_list + bid_y * gridDim.x + bid_x;
    for ( uint32_t i = 0; i < local_intersection_list->n_nodes; ++i ) {
        local_intersection_list->buffer[i] = nullptr;
    }
    local_intersection_list->n_nodes = 0;
    local_intersection_list->cnt = 0;

    while ( atomicCAS(&mem_patch_pipeline->locked, 0, 1) != 0 ) { }
    mem_patch_pipeline->used = 0;
    atomicExch(&mem_patch_pipeline->locked, 0);
}

__global__ void vertexShading (
        float3 *    vertices,
        float3 *    normals,
        float4 *    shaded_vertices,
        float3 *    shaded_normal,
        uint32_t    n_vertices,
        uint32_t    n_normals
) {
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tid_local = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t tid_global = bid * blockDim.x * blockDim.y + tid_local;

    // transform vertices to clip space
    if ( tid_global < n_vertices ) {
    auto vertices_local = vertices + tid_global;
    float4 result = device_local_to_clip * fromVec3(*vertices_local, 1.0f);
    *(shaded_vertices + tid_global) = result;
    }

    // transform normal to world space
    if ( tid_global < n_normals ) {
    auto normal_local =  normals + tid_global;
    float4 result = device_normal_to_world * fromVec3(*normal_local, 0.0f);
    *(shaded_normal + tid_global) = xyz(result);
    }
}

__global__ void viewFrustumCulling(
    float4 *                shaded_vertices,
    TriangleFaceIndex *     faces,
    uint32_t *              visible_table,
    const uint32_t          n_faces
) {
    uint32_t bid            = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tid_local      = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t tid_global     = bid * blockDim.x * blockDim.y + tid_local;
    if ( tid_global >= n_faces ) return;

    auto v0 = (float4 *)(&shaded_vertices[faces[tid_global].Vertices.x]);
    auto v1 = (float4 *)(&shaded_vertices[faces[tid_global].Vertices.y]);
    auto v2 = (float4 *)(&shaded_vertices[faces[tid_global].Vertices.z]);

    uint32_t in_box = vertexInBox(*v0) || vertexInBox(*v1) || vertexInBox(*v2);

    visible_table[tid_global] = in_box;
}

__global__ void clipToScreen(
    float4 *            shaded_vertices,
    const uint32_t      n_vertices
) {

    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tid_local = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t tid_global = bid * blockDim.x * blockDim.y + tid_local;
    if ( tid_global >= n_vertices ) return;

    auto vertex = *(shaded_vertices + tid_global);
    float inv_w = 1.0f / vertex.w;
    vertex.x = vertex.x * clip_to_fb_scale.x * inv_w + clip_to_fb_offset.x;
    vertex.y = vertex.y * clip_to_fb_scale.y * inv_w + clip_to_fb_offset.y;
    vertex.z = vertex.z * clip_to_fb_scale.z * inv_w + clip_to_fb_offset.z;
    vertex.w = inv_w;

    *(float4 *)(shaded_vertices + tid_global) = vertex;
}

__global__ void backFaceCulling(
    float4 *                shaded_vertices,
    TriangleFaceIndex *     faces,
    uint32_t *              visible_table,
    const uint32_t          n_faces
) {
    uint32_t bid            = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tid_local      = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t tid_global     = bid * blockDim.x * blockDim.y + tid_local;

    if ( tid_global >= n_faces ) return;

    auto v0 = (float4 *)(&shaded_vertices[faces[tid_global].Vertices.x]);
    auto v1 = (float4 *)(&shaded_vertices[faces[tid_global].Vertices.y]);
    auto v2 = (float4 *)(&shaded_vertices[faces[tid_global].Vertices.z]);

    float2 vec0 = make_float2(v1->x - v0->x, v1->y - v0->y);
    float2 vec1 = make_float2(v2->x - v1->x, v2->y - v1->y);
    uint32_t face_forward = (vec0.x * vec1.y - vec0.y * vec1.x) >= 0;

    visible_table[tid_global] &= face_forward;

}

__global__ void backFaceReordering(
    float4 *                shaded_vertices,
    TriangleFaceIndex *     faces,
    float3 *                normals,
    const uint32_t          n_faces
) {
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tid_local = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t tid_global = bid * blockDim.x * blockDim.y + tid_local;
    if (tid_global >= n_faces) return;

    auto v0 = faces[tid_global].Vertices.x;
    auto v1 = faces[tid_global].Vertices.y;
    auto v2 = faces[tid_global].Vertices.z;

    float2 vec0 = make_float2(shaded_vertices[v1].x - shaded_vertices[v0].x,
                              shaded_vertices[v1].y - shaded_vertices[v0].y);
    float2 vec1 = make_float2(shaded_vertices[v2].x - shaded_vertices[v1].x,
                              shaded_vertices[v2].y - shaded_vertices[v1].y);
    uint32_t face_backward = (vec0.x * vec1.y - vec0.y * vec1.x) < 0;
    face_backward *= 0xffffffff;
    // reorder vertices so it can pass the pointInTriangleTest
    uint32_t temp = v1;
    v1 = (v1 & ~face_backward) | (v2 & face_backward);
    v2 = (v2 & ~face_backward) | (temp & face_backward);

    faces[tid_global].Vertices.y = v1;
    faces[tid_global].Vertices.z = v2;

    faces[tid_global].TexCoords.y =
            (faces[tid_global].TexCoords.y & ~face_backward) | (faces[tid_global].TexCoords.z & face_backward);
    faces[tid_global].TexCoords.z =
            (faces[tid_global].TexCoords.z & ~face_backward) | (faces[tid_global].TexCoords.y & face_backward);
    faces[tid_global].Normals.y =
            (faces[tid_global].Normals.y & ~face_backward) | (faces[tid_global].Normals.z & face_backward);
    faces[tid_global].Normals.z =
            (faces[tid_global].Normals.z & ~face_backward) | (faces[tid_global].Normals.y & face_backward);

    /*normals[faces[tid_global].Normals.x] =
            face_backward == 0 ? normals[faces[tid_global].Normals.x] : -normals[faces[tid_global].Normals.x];
    normals[faces[tid_global].Normals.y] =
            face_backward == 0 ? normals[faces[tid_global].Normals.y] : -normals[faces[tid_global].Normals.y];
    normals[faces[tid_global].Normals.z] =
            face_backward == 0 ? normals[faces[tid_global].Normals.z] : -normals[faces[tid_global].Normals.z];*/
}


template<typename T>
__device__ __inline__ void
scanUpSweep(uint32_t tid, T * data, uint32_t size, uint32_t log2Size) {
    for ( uint32_t e = 1; e < log2Size; e += 1 ) {
        uint32_t step = 1 << (e - 1);
        if ( tid <  (size >> e) ) {
            data[((tid + 1) << e) - 1] += data[((tid + 1) << e) - 1 - step];
        }
        __syncthreads();
    }
}


__global__ void tiling(
    float4 *                shaded_vertices,
    TriangleFaceIndex *     faces,
    IntersectionList *      intersection_list,
    CuMemPatch *            mem_patch,
    const uint32_t *        visible_table,
    const uint32_t          n_faces
) {

    uint32_t bid        = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tid_local  = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ uint32_t intersected_buffer[BLK_SIZE];
    __shared__ uint32_t index_buffer[BLK_SIZE];
    __shared__ float2   shaded_vertices_buffer[BLK_SIZE * 3];

    float left       = blockIdx.x * TILE_SIZE;
    float right      = left + TILE_SIZE;
    float bottom     = blockIdx.y * TILE_SIZE;
    float top        = bottom + TILE_SIZE;

    uint32_t loops              = (n_faces + BLK_SIZE - 1) / BLK_SIZE;

    auto local_intersection_list = intersection_list + bid;
    uint32_t faces_count = 0;
    for ( uint32_t i = 0; i < loops; ++i ) {
        uint32_t face_index = tid_local + i * BLK_SIZE;
        if ( face_index < n_faces  ) {
            shaded_vertices_buffer[tid_local * 3 + 0].x = shaded_vertices[faces[face_index].Vertices.x].x;
            shaded_vertices_buffer[tid_local * 3 + 0].y = shaded_vertices[faces[face_index].Vertices.x].y;
            shaded_vertices_buffer[tid_local * 3 + 1].x = shaded_vertices[faces[face_index].Vertices.y].x;
            shaded_vertices_buffer[tid_local * 3 + 1].y = shaded_vertices[faces[face_index].Vertices.y].y;
            shaded_vertices_buffer[tid_local * 3 + 2].x = shaded_vertices[faces[face_index].Vertices.z].x;
            shaded_vertices_buffer[tid_local * 3 + 2].y = shaded_vertices[faces[face_index].Vertices.z].y;

            uint32_t intersected =  pointInTile(left, right, bottom, top, shaded_vertices_buffer[tid_local * 3 + 0].x,
                                                shaded_vertices_buffer[tid_local * 3 + 0].y) ||
                                    pointInTile(left, right, bottom, top, shaded_vertices_buffer[tid_local * 3 + 1].x,
                                                shaded_vertices_buffer[tid_local * 3 + 1].y) ||
                                    pointInTile(left, right, bottom, top, shaded_vertices_buffer[tid_local * 3 + 2].x,
                                                shaded_vertices_buffer[tid_local * 3 + 2].y) ||
                                    lineIntersectsWithTileTest(left, right, bottom, top,
                                                               shaded_vertices_buffer[tid_local * 3 + 0].x,
                                                               shaded_vertices_buffer[tid_local * 3 + 0].y,
                                                               shaded_vertices_buffer[tid_local * 3 + 1].x,
                                                               shaded_vertices_buffer[tid_local * 3 + 1].y) ||
                                    lineIntersectsWithTileTest(left, right, bottom, top,
                                                               shaded_vertices_buffer[tid_local * 3 + 1].x,
                                                               shaded_vertices_buffer[tid_local * 3 + 1].y,
                                                               shaded_vertices_buffer[tid_local * 3 + 2].x,
                                                               shaded_vertices_buffer[tid_local * 3 + 2].y) ||
                                    lineIntersectsWithTileTest(left, right, bottom, top,
                                                               shaded_vertices_buffer[tid_local * 3 + 2].x,
                                                               shaded_vertices_buffer[tid_local * 3 + 2].y,
                                                               shaded_vertices_buffer[tid_local * 3 + 0].x,
                                                               shaded_vertices_buffer[tid_local * 3 + 0].y) ;

            intersected_buffer[tid_local] = intersected & visible_table[face_index];
        }
        else {
            intersected_buffer[tid_local] = 0;
        }
        index_buffer[tid_local] = intersected_buffer[tid_local];
        __syncthreads();

        exclusiveScan(tid_local, index_buffer);
        uint32_t cnt = index_buffer[BLK_SIZE - 1] + intersected_buffer[BLK_SIZE - 1];
        if ( tid_local == 0 ) {
            while ( atomicCAS(&local_intersection_list->locked, 0, 1) != 0 ) { }
            if ( cnt > local_intersection_list->n_nodes * IntersectionListConstants::NODE_SIZE - faces_count ) {
                assert( local_intersection_list->n_nodes < IntersectionListConstants::MAX_NODE_NUM - 1 );
                local_intersection_list->buffer[local_intersection_list->n_nodes] =
                        (uint32_t *) acquire(mem_patch, sizeof(uint32_t) * IntersectionListConstants::NODE_SIZE);
                local_intersection_list->n_nodes += 1;
            }
        }
        __syncthreads();

        if ( intersected_buffer[tid_local] ) {
            *at(local_intersection_list, faces_count + index_buffer[tid_local]) = face_index;
        }
        __syncthreads();

        if ( tid_local == 0 ) {
            atomicExch(&local_intersection_list->locked, 0);
        }
        faces_count += cnt;
        __syncthreads();
    }
    __syncthreads();

    if ( tid_local == 0 ) local_intersection_list->cnt = faces_count;
}

// Another Version with higher Computational Intensity, but is slower than the above version.
// Thus, not used.
/*__global__ void tiling_f(
        float4 *                shaded_vertices,
        TriangleFaceIndex *     faces,
        IntersectionList *      intersection_list,
        CuMemPatch *            mem_patch,
        const uint32_t *        visible_table,
        const uint32_t          width,
        const uint32_t          height,
        const uint32_t          n_faces
        ) {
    uint32_t bid        = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tid_local  = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t tid_global = bid * blockDim.x * blockDim.y + tid_local;

    __shared__ float2 face_vertices[BLK_SIZE * 3];
    __shared__ uint16_t index[BLK_SIZE];
    __shared__ uint16_t intersection[BLK_SIZE];
    __shared__ uint16_t visible[BLK_SIZE];
    if ( tid_global < n_faces ) {
        face_vertices[3 * tid_local + 0] = make_float2(shaded_vertices[faces[tid_global].Vertices.x].x,
                                                       shaded_vertices[faces[tid_global].Vertices.x].y);
        face_vertices[3 * tid_local + 1] = make_float2(shaded_vertices[faces[tid_global].Vertices.y].x,
                                                       shaded_vertices[faces[tid_global].Vertices.y].y);
        face_vertices[3 * tid_local + 2] = make_float2(shaded_vertices[faces[tid_global].Vertices.z].x,
                                                       shaded_vertices[faces[tid_global].Vertices.z].y);
        visible[tid_local] = visible_table[tid_global];
    }
    index[tid_local] = 0;
    intersection[tid_local] = 0;

    uint32_t tile_dim_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t tile_dim_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t n_tiles = tile_dim_y * tile_dim_x;

    uint32_t offset = bid % n_tiles;
    #pragma unroll 2
    for ( uint32_t i = 0; i < n_tiles ; ++i, offset = (offset + 1) % n_tiles ) {
        uint32_t offset_unroll = (offset + (n_tiles >> 1)) % n_tiles;

        uint32_t tile_y = offset / tile_dim_x;
        uint32_t tile_x = offset % tile_dim_x;
        float bottom = tile_y * TILE_SIZE;
        float top = bottom + TILE_SIZE;
        float left = tile_x * TILE_SIZE;
        float right = left + TILE_SIZE;

        auto local_intersection_list = intersection_list + offset;

        if ( tid_global < n_faces ) {
            intersection[tid_local] =   pointInTile(left, right, bottom, top, face_vertices[tid_local * 3 + 0].x,
                                                    face_vertices[tid_local * 3 + 0].y) ||
                                        pointInTile(left, right, bottom, top, face_vertices[tid_local * 3 + 1].x,
                                                    face_vertices[tid_local * 3 + 1].y) ||
                                        pointInTile(left, right, bottom, top, face_vertices[tid_local * 3 + 2].x,
                                                    face_vertices[tid_local * 3 + 2].y) ||
                                        lineIntersectsWithTileTest(left, right, bottom, top,
                                                                   face_vertices[tid_local * 3 + 0].x,
                                                                   face_vertices[tid_local * 3 + 0].y,
                                                                   face_vertices[tid_local * 3 + 1].x,
                                                                   face_vertices[tid_local * 3 + 1].y) ||
                                        lineIntersectsWithTileTest(left, right, bottom, top,
                                                                   face_vertices[tid_local * 3 + 1].x,
                                                                   face_vertices[tid_local * 3 + 1].y,
                                                                   face_vertices[tid_local * 3 + 2].x,
                                                                   face_vertices[tid_local * 3 + 2].y) ||
                                        lineIntersectsWithTileTest(left, right, bottom, top,
                                                                   face_vertices[tid_local * 3 + 2].x,
                                                                   face_vertices[tid_local * 3 + 2].y,
                                                                   face_vertices[tid_local * 3 + 0].x,
                                                                   face_vertices[tid_local * 3 + 0].y) ;
            intersection[tid_local] &= visible[tid_local];
        } else {
            intersection[tid_local] = 0;
        }

        index[tid_local] = intersection[tid_local];
        __syncthreads();

        // compute local index
        exclusiveScan(tid_local, index);

        uint32_t cnt = index[BLK_SIZE - 1] + intersection[BLK_SIZE - 1];

        if ( tid_local == 0 ) {
            // acquire the lock
            while ( atomicCAS(&local_intersection_list->locked, 0, 1) != 0 ) { }
            if ( cnt > local_intersection_list->n_nodes * IntersectionListConstants::NODE_SIZE - local_intersection_list->cnt ) {
                assert( local_intersection_list->n_nodes < IntersectionListConstants::MAX_NODE_NUM - 1 );
                local_intersection_list->buffer[local_intersection_list->n_nodes] =
                        (uint32_t *) acquire(mem_patch, sizeof(uint32_t) * IntersectionListConstants::NODE_SIZE);
                local_intersection_list->n_nodes += 1;
            }
        }
        __syncthreads();

        if ( intersection[tid_local] ) {
            uint32_t current_index = local_intersection_list->cnt + index[tid_local];
            *at(local_intersection_list, current_index) = tid_global;
        }
        __syncthreads();

        if ( tid_local == 0 ) {
            local_intersection_list->cnt += cnt;
            // release the lock
            atomicExch(&local_intersection_list->locked, 0);
        }

    }
}*/

__device__ __inline__ float3
barycentricCoord(float4 v0, float4 v1, float4 v2, float pixel_x, float pixel_y ) {
    float3 b_coord;
    b_coord.x = (-( pixel_x - v1.x ) * ( v2.y - v1.y ) + ( v2.x - v1.x ) * ( pixel_y - v1.y )) / ( -( v0.x - v1.x ) * ( v2.y - v1.y ) + ( v2.x - v1.x ) * ( v0.y - v1.y ) );
    b_coord.y = (-( pixel_x - v2.x ) * ( v0.y - v2.y ) + ( v0.x - v2.x ) * ( pixel_y - v2.y )) / ( -( v1.x - v2.x ) * ( v0.y - v2.y ) + ( v0.x - v2.x ) * ( v1.y - v2.y ) );
    b_coord.z = 1.0f - b_coord.x - b_coord.y;
    return b_coord;
}

// TODO: Per-pixel link list Version
template <uint32_t flag>
__global__ void rasterizeWithEarlyZ(
    float4 *                shaded_vertices,
    float3 *                shaded_normals,
    float2 *                device_tex_coords,
    float *                 depth_buffer,
    IntersectionList *      intersection_list,
    TriangleFaceIndex *     faces,
    FragmentBuffer *        fragment_buffer,
    int32_t *               start_offset_obj,
    CuMemPatch *            mem_patch,
    uint32_t                width,
    uint32_t                height,
    uint32_t                n_faces
) {
    uint32_t bid = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t tid_local = threadIdx.y * blockDim.x + threadIdx.x;
    if ( pixelX >= width || pixelY >= height ) return;

    float pixel_center_x = 0.5f + pixelX;
    float pixel_center_y = 0.5f + pixelY;

    __shared__ float                local_depth_buffer[BLK_SIZE];
    __shared__ uint32_t             test_passed[BLK_SIZE];
    __shared__ uint32_t             index[BLK_SIZE];
    __shared__ int32_t              local_start_offset[BLK_SIZE];
    __shared__ FragmentAttribute    local_fragment_buffer[BLK_SIZE];
    local_depth_buffer[tid_local] = depth_buffer[pixelY * width + pixelX];
    local_start_offset[tid_local] = -1;
    test_passed[tid_local] = 0;
    index[tid_local] = 0;

    auto local_intersection_list = intersection_list + bid;
    uint32_t n_intersected = local_intersection_list->cnt;
    uint32_t fragment_cnt = fragment_buffer[bid].cnt;
    for ( uint32_t i = 0; i < n_intersected; ++i ) {
        uint32_t face_index = *at(local_intersection_list, i);
        TriangleFaceIndex * current_face = faces + face_index;

        float4 v0 = shaded_vertices[current_face->Vertices.x];
        float4 v1 = shaded_vertices[current_face->Vertices.y];
        float4 v2 = shaded_vertices[current_face->Vertices.z];

        float4 vec0 = v1 - v0;
        float4 vec1 = v2 - v1;
        float4 vec2 = v0 - v2;

        if ( pointInTriangleTest(v0, v1, v2, pixel_center_x, pixel_center_y) ) {

            float3 b_coord = barycentricCoord(v0, v1, v2, pixel_center_x, pixel_center_y);
            float depth = b_coord.x * v0.z + b_coord.y * v1.z + b_coord.z * v2.z;
            // Early Z
            if ( local_depth_buffer[tid_local] < depth ) {
                if constexpr ( (flag & Pipeline_DepthWriteDisableBit) != 0 ) {
                    test_passed[tid_local] = 0;
                }
            }
            else { // depth test passed

                if constexpr ( (flag & Pipeline_DepthWriteDisableBit) == 0 ) {
                    local_depth_buffer[tid_local] = depth;
                }

                test_passed[tid_local] = 1;

                local_fragment_buffer[tid_local].Depth = depth;

                /*
                 * | ddx1   ddy1 | | dx |   | da1 |          | dx |            |  ddy2    -ddy1 | | da1 |
                 * |             | |    | = |     |    =>    |    | = 1/det(D) |                | |     |
                 * | ddx2   ddy2 | | dy |   | da2 |		     | dy |			   | -ddx2	   ddx1 | | da2 |
                 *
                 *	dx = (da1 * dy2 - da2 * dy1) / det(D)
                 *	dy = (da2 * dx1 - da1 * dx2) / det(D)
                 */
                float2 dd1;
                float2 dd2;
                dd1.x = v1.x - v0.x;
                dd2.x = v2.x - v1.x;
                dd1.y = v1.y - v0.y;
                dd2.y = v2.y - v1.y;
                float inv_det = 1.0f / (dd1.x * dd2.y - dd1.y * dd2.x);
                float a11 =  dd2.y * inv_det;
                float a12 = -dd1.y * inv_det;
                float a21 = -dd2.x * inv_det;
                float a22 =  dd1.x * inv_det;
                // compute derivative of TexCoord.x
                float da1 = device_tex_coords[current_face->TexCoords.y].x - device_tex_coords[current_face->TexCoords.x].x;
                float da2 = device_tex_coords[current_face->TexCoords.z].x - device_tex_coords[current_face->TexCoords.y].x;
                local_fragment_buffer[tid_local].DerivativesU.x = a11 * da1 + a12 * da2 ;    // dudx
                local_fragment_buffer[tid_local].DerivativesV.y = a21 * da1 + a22 * da2 ;    // dudy
                // compute derivative of TexCoord.y
                da1 = device_tex_coords[current_face->TexCoords.y].y - device_tex_coords[current_face->TexCoords.x].y;
                da2 = device_tex_coords[current_face->TexCoords.z].y - device_tex_coords[current_face->TexCoords.y].y;
                local_fragment_buffer[tid_local].DerivativesV.x = a11 * da1 + a12 * da2;    // dvdx
                local_fragment_buffer[tid_local].DerivativesV.y = a21 * da1 + a22 * da2;    // dvdy


                // perspective correct interpolation for texture coordinates & normals
                float correct_w = 1.0f / (b_coord.x * v0.w + b_coord.y * v1.w + b_coord.z * v2.w);

                float2 texture_coord;
                texture_coord.x = (b_coord.x * device_tex_coords[current_face->TexCoords.x].x * v0.w +
                                   b_coord.y * device_tex_coords[current_face->TexCoords.y].x * v1.w +
                                   b_coord.z * device_tex_coords[current_face->TexCoords.z].x * v2.w);
                texture_coord.y = (b_coord.x * device_tex_coords[current_face->TexCoords.x].y * v0.w +
                                   b_coord.y * device_tex_coords[current_face->TexCoords.y].y * v1.w +
                                   b_coord.z * device_tex_coords[current_face->TexCoords.z].y * v2.w);
                local_fragment_buffer[tid_local].TexCoord.x = texture_coord.x * correct_w;
                local_fragment_buffer[tid_local].TexCoord.y = texture_coord.y * correct_w;

                float3 normal;
                normal.x = (b_coord.x * shaded_normals[current_face->Normals.x].x * v0.w +
                            b_coord.y * shaded_normals[current_face->Normals.y].x * v1.w +
                            b_coord.z * shaded_normals[current_face->Normals.z].x * v2.w);
                normal.y = (b_coord.x * shaded_normals[current_face->Normals.x].y * v0.w +
                            b_coord.y * shaded_normals[current_face->Normals.y].y * v1.w +
                            b_coord.z * shaded_normals[current_face->Normals.z].y * v2.w);
                normal.z = (b_coord.x * shaded_normals[current_face->Normals.x].z * v0.w +
                            b_coord.y * shaded_normals[current_face->Normals.y].z * v1.w +
                            b_coord.z * shaded_normals[current_face->Normals.z].z * v2.w);
                *(float3 *)(&local_fragment_buffer[tid_local].Normal) = normal * correct_w;
            }
        }
        else {
            if constexpr ( (flag & Pipeline_DepthWriteDisableBit) != 0 ) {
                test_passed[tid_local] = 0;
            }
        }
        __syncthreads();

        if constexpr ( (flag & Pipeline_DepthWriteDisableBit) != 0 ) {
            index[tid_local] = test_passed[tid_local];
            __syncthreads();
            // compute the index of generated fragments in fragment buffer
            exclusiveScan(tid_local, index);
            uint32_t cnt = test_passed[BLK_SIZE - 1] + index[BLK_SIZE - 1];

            if ( tid_local == 0 ) {
                if ( cnt > fragment_buffer[bid].n_nodes * FragmentBufferConstants::NODE_SIZE - fragment_cnt) {
                    assert(fragment_buffer[bid].n_nodes < FragmentBufferConstants::MAX_NODE_NUM);
                    fragment_buffer[bid].buffer[fragment_buffer[bid].n_nodes] =
                            (PerPixelLinkedListNode<FragmentAttribute> *)acquire(mem_patch, sizeof(PerPixelLinkedListNode<FragmentAttribute>) * FragmentBufferConstants::NODE_SIZE);
                    fragment_buffer[bid].n_nodes += 1;
                }
            }
            __syncthreads();

            if ( test_passed[tid_local] ) {
                int32_t current_index = fragment_cnt + index[tid_local];
                auto frag = at(fragment_buffer + bid, current_index);
                frag->data = local_fragment_buffer[tid_local];
                frag->last_offset = local_start_offset[tid_local];
                local_start_offset[tid_local] = current_index;
            }
            fragment_cnt += cnt;
        }
    }

    // if not in Transparent mode, with early z, should only have BLK_SIZE fragments per block
    if constexpr ( (flag & Pipeline_DepthWriteDisableBit) == 0 ) {
        index[tid_local] = test_passed[tid_local];
        // compute the index of generated fragments in fragment buffer
        __syncthreads();
        exclusiveScan(tid_local, index);
        uint32_t cnt = test_passed[BLK_SIZE - 1] + index[BLK_SIZE - 1];

        if ( tid_local == 0 ) {
            if ( cnt > fragment_buffer[bid].n_nodes * FragmentBufferConstants::NODE_SIZE - fragment_buffer[bid].cnt ) {
                assert(fragment_buffer[bid].n_nodes < FragmentBufferConstants::MAX_NODE_NUM);
                fragment_buffer[bid].buffer[fragment_buffer[bid].n_nodes] = (PerPixelLinkedListNode<FragmentAttribute> *)
                        acquire(mem_patch, sizeof(PerPixelLinkedListNode<FragmentAttribute>) * FragmentBufferConstants::NODE_SIZE);
                fragment_buffer[bid].n_nodes += 1;
            }
        }
        __syncthreads();

        if ( test_passed[tid_local] ) {
            int32_t current_index = fragment_cnt + index[tid_local];
            auto frag = at(fragment_buffer + bid, current_index);
            frag->data = local_fragment_buffer[tid_local];
            frag->last_offset = local_start_offset[tid_local];
            local_start_offset[tid_local] = current_index;
        }
        fragment_cnt += cnt;
    }

    start_offset_obj[pixelY * width + pixelX] = local_start_offset[tid_local];
    if constexpr ( (flag & Pipeline_DepthWriteDisableBit) == 0 ) {
        depth_buffer[pixelY * width + pixelX] = local_depth_buffer[tid_local];
    }
    if ( tid_local == 0 ) fragment_buffer[bid].cnt = fragment_cnt;
}

template < uint32_t flag >
__global__ void fragmentShading(
    FragmentBuffer *            fragment_buffer,
    ShadedFragmentBuffer *      shaded_fragment_buffer,
    CuMemPatch *                mem_patch,
    cudaTextureObject_t         texture,
    int32_t *                   start_offset_obj,
    int32_t *                   start_offset_all,
    float                       alpha,
    uint32_t                    width,
    uint32_t                    height
) {
    uint32_t bid = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t tid_local = threadIdx.x + threadIdx.y * blockDim.x;
    uint32_t pixelX = blockIdx.x * TILE_SIZE + threadIdx.x;
    uint32_t pixelY = blockIdx.y * TILE_SIZE + threadIdx.y;
    if ( pixelX >= width || pixelY >= height ) return;

    auto local_shaded_fragment_buffer = shaded_fragment_buffer + bid;
    auto local_fragment_buffer = fragment_buffer + bid;

    __shared__ int32_t local_start_offset_sf[BLK_SIZE];
    __shared__ int32_t local_start_offset_obj[BLK_SIZE];
    __shared__ ShadedFragment local_shaded_fragment[BLK_SIZE];
    __shared__ uint16_t shaded[BLK_SIZE];
    __shared__ uint16_t index[BLK_SIZE];
    local_start_offset_sf[tid_local] = start_offset_all[pixelX + pixelY * width];
    local_start_offset_obj[tid_local] = start_offset_obj[pixelX + pixelY * width];

    uint32_t frag_cnt = local_fragment_buffer->cnt;
    if ( tid_local == 0 ) {
        while ( frag_cnt + local_shaded_fragment_buffer->cnt >
                local_shaded_fragment_buffer->n_nodes * ShadedFragmentBufferConstants::NODE_SIZE ) {
            assert(local_shaded_fragment_buffer->n_nodes < ShadedFragmentBufferConstants::MAX_NODE_NUM - 1);
            local_shaded_fragment_buffer->buffer[local_shaded_fragment_buffer->n_nodes] = (PerPixelLinkedListNode<ShadedFragment> *)
                    acquire(mem_patch, sizeof(PerPixelLinkedListNode<ShadedFragment>) * ShadedFragmentBufferConstants::NODE_SIZE);
            local_shaded_fragment_buffer->n_nodes += 1;
        }
    }
    __syncthreads();

    uint32_t cnt = 0;
    do {
        if ( local_start_offset_obj[tid_local] != -1 ) {
            auto frag_node = at(local_fragment_buffer, local_start_offset_obj[tid_local]);
            auto fragment = &frag_node->data;
            local_start_offset_obj[tid_local] = frag_node->last_offset;

            // compute mipmap level
            float2 ddx, ddy;
            ddx.x = fragment->DerivativesU.x * PipelineParams.width;
            ddx.y = fragment->DerivativesV.x * PipelineParams.width;
            ddy.x = fragment->DerivativesU.y * PipelineParams.height;
            ddy.y = fragment->DerivativesV.y * PipelineParams.height;
            float rho = max( dot(ddx, ddx), dot(ddy, ddy) );
            float lod = 0.5f * log2(rho + 1e-10f);

            // compute color
            float3 normal = *(float3 *)(&fragment->Normal);
            float normal_norm = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
            if ( normal_norm == 0 ) return;
            float denom = rsqrt(normal_norm);
            normal = normal * denom;

            // compute light
            float sun_factor = max(dot(PipelineParams.sun_direction, normal), 0.0f);
            float3 light = sun_factor * PipelineParams.sun_energy;
            float sky_factor = 0.5f * dot(PipelineParams.sky_direction, normal) + 0.5f;
            light = light + (PipelineParams.sky_energy - PipelineParams.ground_energy) * sky_factor + PipelineParams.ground_energy;

            // fetch color
            float4 color = tex2DLod<float4>(texture, fragment->TexCoord.x, fragment->TexCoord.y, 0);
            color.w = alpha;

            if constexpr ((flag & PipelineMask_RenderingMode) == Pipeline_Rendering_Opaque) {
                color.x *= light.x;
                color.y *= light.y;
                color.z *= light.z;
                color.w = 0.0f;
            }
            else {
                color.x *= light.x * color.w;
                color.y *= light.y * color.w;
                color.z *= light.z * color.w;
                color.w = 1 - color.w;
            }

            local_shaded_fragment[tid_local].Color = *(Vector4 *)(&color);
            local_shaded_fragment[tid_local].Depth = fragment->Depth;

            shaded[tid_local] = 1;
        }
        else {
            shaded[tid_local] = 0;
        }

        index[tid_local] = shaded[tid_local];
        __syncthreads();

        // compute indices
        exclusiveScan(tid_local, index);

        cnt = index[BLK_SIZE - 1] + shaded[BLK_SIZE - 1];

        // write to shaded fragment buffer
        if ( shaded[tid_local] == 1 ) {
            uint32_t current_index = local_shaded_fragment_buffer->cnt + index[tid_local];
            auto sf_node = at(local_shaded_fragment_buffer, current_index);
            sf_node->data = local_shaded_fragment[tid_local];
            sf_node->last_offset = local_start_offset_sf[tid_local];
            local_start_offset_sf[tid_local] = current_index;
        }
        __syncthreads();

        if ( tid_local == 0 ) {
            local_shaded_fragment_buffer->cnt += cnt;
        }

    } while ( cnt > 0 );

    start_offset_all[pixelX + pixelY * width] = local_start_offset_sf[tid_local];

}

template<uint32_t AOIT_NODE_CNT>
__device__ __inline__ void
insertFragment(
        float3 *        color,
        float *         depth,
        float *         trans,
        ShadedFragment* fragment
        ) {
    // find insertion index
    uint32_t index = 0;
    float prev_trans = 1.0f;
    #pragma unroll AOIT_NODE_CNT
    for ( uint32_t i = 0; i < AOIT_NODE_CNT; ++i ) {
        if ( fragment->Depth > depth[i] ) {
            index++;
            prev_trans = trans[i];
        }
    }

    // make room for new fragment
    #pragma unroll AOIT_NODE_CNT
    for ( uint32_t i = AOIT_NODE_CNT; i >= 1; --i ) {
        if ( i > index ) {
            color[i] = color[i - 1];
            depth[i] = depth[i - 1];
            trans[i] = trans[i - 1] * fragment->Color.w;
        }
    }

    // insert new fragment
    color[index].x = fragment->Color.x;
    color[index].y = fragment->Color.y;
    color[index].z = fragment->Color.z;
    depth[index] = fragment->Depth;
    trans[index] = prev_trans * fragment->Color.w;

    // blend last node
    float factor = trans[AOIT_NODE_CNT - 1] / trans[AOIT_NODE_CNT - 2];
    color[AOIT_NODE_CNT - 1] = color[AOIT_NODE_CNT - 1] + color[AOIT_NODE_CNT - 1] * factor;
    trans[AOIT_NODE_CNT - 1] = trans[AOIT_NODE_CNT];
}

// Adaptive Transparency Method
// Reference: https://www.intel.com/content/dam/develop/external/us/en/documents/37944-adaptive-transparency-hpg11.pdf
// Intel's implementation: https://github.com/GameTechDev/AOIT-Update/blob/master/OIT_DX11/AOIT%20Technique/AOIT.hlsl
template<uint32_t AOIT_NODE_CNT>
__global__ void computeVisNode(
        ShadedFragmentBuffer *  shaded_fragment_buffer,
        AOITNode *              node_array,
        float *                 depth_buffer,
        const int32_t *         start_offset_all,
        uint32_t                width,
        uint32_t                height
        ) {
    uint32_t bid = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t tid_local = threadIdx.x + threadIdx.y * blockDim.x;
    uint32_t pixelX = blockIdx.x * TILE_SIZE + threadIdx.x;
    uint32_t pixelY = blockIdx.y * TILE_SIZE + threadIdx.y;
    if ( pixelX >= width || pixelY >= height ) return;

    __shared__ float3 color[BLK_SIZE * (AOIT_NODE_CNT + 1)];
    __shared__ float depth[BLK_SIZE * (AOIT_NODE_CNT + 1)];
    __shared__ float trans[BLK_SIZE * (AOIT_NODE_CNT + 1)];
    __shared__ float local_depth_buffer[BLK_SIZE];
    local_depth_buffer[tid_local] = depth_buffer[pixelX + pixelY * width];

    int32_t start_offset = start_offset_all[pixelY * width + pixelX];
    auto local_node_array = node_array + (pixelY * width + pixelX) * AOIT_NODE_CNT;
    auto local_shaded_fragment_buffer = shaded_fragment_buffer + bid;

    #pragma unroll AOIT_NODE_CNT
    for ( uint32_t offset = 0; offset < AOIT_NODE_CNT; ++offset ) {
        depth[tid_local * (AOIT_NODE_CNT + 1) + offset] = local_node_array[offset].Depth;
        color[tid_local * (AOIT_NODE_CNT + 1) + offset].x = local_node_array[offset].Color.x;
        color[tid_local * (AOIT_NODE_CNT + 1) + offset].y = local_node_array[offset].Color.y;
        color[tid_local * (AOIT_NODE_CNT + 1) + offset].z = local_node_array[offset].Color.z;
        trans[tid_local * (AOIT_NODE_CNT + 1) + offset] = local_node_array[offset].Trans;
    }
    depth[tid_local * (AOIT_NODE_CNT + 1) + AOIT_NODE_CNT] = 1.0f;
    color[tid_local * (AOIT_NODE_CNT + 1) + AOIT_NODE_CNT] = make_float3(0.0f, 0.0f, 0.0f);
    trans[tid_local * (AOIT_NODE_CNT + 1) + AOIT_NODE_CNT] = 1.0f;

    while ( start_offset != -1 ) {
        auto shaded_frag = at(local_shaded_fragment_buffer, start_offset);
        start_offset = shaded_frag->last_offset;

        // Late Z test
        if ( shaded_frag->data.Depth > local_depth_buffer[tid_local] ) continue;

        insertFragment<AOIT_NODE_CNT>(
                color + tid_local * (AOIT_NODE_CNT + 1),
                depth + tid_local * (AOIT_NODE_CNT + 1),
                trans + tid_local * (AOIT_NODE_CNT + 1),
                &shaded_frag->data);
    }

    #pragma unroll AOIT_NODE_CNT
    for ( uint32_t offset = 0; offset < AOIT_NODE_CNT; ++offset ) {
        local_node_array[offset].Depth = depth[tid_local * (AOIT_NODE_CNT + 1) + offset];
        local_node_array[offset].Color.x = color[tid_local * (AOIT_NODE_CNT + 1) + offset].x;
        local_node_array[offset].Color.y = color[tid_local * (AOIT_NODE_CNT + 1) + offset].y;
        local_node_array[offset].Color.z = color[tid_local * (AOIT_NODE_CNT + 1) + offset].z;
        local_node_array[offset].Trans = trans[tid_local * (AOIT_NODE_CNT + 1) + offset];
    }
}

template<uint32_t AOIT_NODE_CNT, uint32_t flag>
__global__ void blending(
        AOITNode *      node_array,
        float4 *        sample_frame_buffer,
        uint32_t        width,
        uint32_t        height
        ) {
    uint32_t tid_local = threadIdx.x + threadIdx.y * blockDim.x;
    uint32_t pixelX = blockIdx.x * TILE_SIZE + threadIdx.x;
    uint32_t pixelY = blockIdx.y * TILE_SIZE + threadIdx.y;
    if ( pixelX >= width || pixelY >= height ) return;

    auto local_node_array = node_array + (pixelX + pixelY * width) * AOIT_NODE_CNT;

    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float trans = 1.0f;

    if constexpr ((flag & PipelineMask_RenderingMode) == Pipeline_Rendering_Transparent) {
        #pragma unroll AOIT_NODE_CNT
        for ( uint32_t i = 0; i < AOIT_NODE_CNT; ++i ) {
            color = color + trans * local_node_array[i].Color;
            trans = local_node_array[i].Trans;
        }
    }
    else if constexpr ((flag & PipelineMask_RenderingMode) == Pipeline_Rendering_Opaque) {
        color = local_node_array[0].Color;
        trans = local_node_array[0].Trans;
    }

    sample_frame_buffer[pixelX + pixelY * width] = fromVec3(color, trans);
}

__global__ void resolve(
        float4 *    sample_frame_buffer,
        float4 *    resolved_frame_buffer,
        uint32_t    width,
        uint32_t    height,
        uint32_t    ss_factor
        ) {
    uint32_t pixelX = blockIdx.x * TILE_SIZE + threadIdx.x;
    uint32_t pixelY = blockIdx.y * TILE_SIZE + threadIdx.y;
    if ( pixelX >= width || pixelY >= height ) return;

    float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for ( uint32_t y = 0; y < ss_factor; ++y ) {
        for ( uint32_t x = 0; x < ss_factor; ++ x ) {
            color = color + sample_frame_buffer[pixelX * ss_factor + x + (pixelY * ss_factor) * width * ss_factor];
        }
    }
    // white background
    color = color + color.w * make_float4(220.0f, 220.0f, 220.0f, 255.0f);
    float inv_ss = 1.0f / (ss_factor * ss_factor);
    color = color * inv_ss;
    color.w = 255.0f;

    resolved_frame_buffer[pixelX + pixelY * width] = color;
}

};
