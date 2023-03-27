#pragma once
#include <iostream>
#include <vector>
#include "vector3.h"
#include "mat4.h"
#include "framebuffer.h"
#include "camera.h"
#include "render_obj.h"
#include "vertex_attribute.h"
#include "texture.h"
#include "cuda_check.h"
#include "pipeline_parameters.h"
#include <cassert>
#include "device_structs.h"

namespace renderer {

class Rasterizer {
public:
    Rasterizer() = delete;
    Rasterizer(Rasterizer& other) = delete;
    Rasterizer(Rasterizer&& other) = delete;
    Rasterizer(const Rasterizer& other) = delete;
    Rasterizer(const std::vector<std::string>& meshes, const std::vector<std::vector<geometry::Vector3>>& trans,
               const std::vector<std::string>& texture_files, const std::vector<float>& opacity, Camera view);
    ~Rasterizer();

    void run();

    // pipeline rendering mode modification
    void setTransparent() { pipeline_flag |= 0x0001; }
    void setOpaque() { pipeline_flag &= (~0x0001); }
    void disableDepthBufferWrite() { pipeline_flag |= Pipeline_DepthWriteDisableBit; }
    void enableDepthBufferWrite() { pipeline_flag &= (~Pipeline_DepthWriteDisableBit); }
    void setMSAAScale(uint32_t s) {
        assert(s <= 4);
        supersample_factor = s;
        ss_height = height * supersample_factor;
        ss_width = width * supersample_factor;
    }

    static constexpr uint32_t AOIT_NODE_CNT = 8;
    static constexpr uint32_t MEM_PATCH_SIZE_PIPELINE = 1 << 28;
    static constexpr uint32_t MEM_PATCH_SIZE_SHADED_FRAGMENT = 1 << 29;
private:
    void setup();
    void load(uint32_t index);
    void setTransform(uint32_t obj_index);

    void clear();
    void clearDeviceObj(uint32_t obj_index);
    void resetPreAllocatedMem();

    // graphic pipeline
    void shadeVertices(uint32_t obj_index);
    void cullFaces(uint32_t obj_index);
    void clip2Screen(uint32_t obj_index);
    void backFaceProcessing(uint32_t obj_index);
    void tile(uint32_t obj_index);
    void rasterize(uint32_t obj_index);
    void shadeFragments(uint32_t obj_index);
    void computeAOITNodes();
    void alphaBlending();
    void writeBack();

private:

    // parameters for objects
    uint32_t n_objs;
    std::vector<uint32_t> n_faces;
    std::vector<uint32_t> n_vertices;
    std::vector<uint32_t> n_tex_coords;
    std::vector<uint32_t> n_normals;

    // device ptr for each obj
    std::vector<float3 *>                   device_vertices;
    std::vector<float2 *>                   device_tex_coords;
    std::vector<float3 *>                   device_normals;
    std::vector<float4 *>                   shaded_vertices;
    std::vector<float3 *>                   shaded_normal;
    std::vector<uint32_t *>                 visible_table;
    std::vector<TriangleFaceIndex *>        faces;
    std::vector<std::unique_ptr<Texture>>   textures;

    // device ptr for the whole pipline
    float *                     depth_buffer;
    float4 *                    device_frame_buffer;
    float4 *                    device_resolved_buffer;
    int32_t *                   start_offset_obj;
    int32_t *                   start_offset_all;
    FragmentBuffer *            fragment_buffer;
    ShadedFragmentBuffer *      shaded_fragment_buffer;
    IntersectionList *          intersection_list;
    AOITNode *                  aoit_fragments;
    CuMemPatch *                mem_patch_pipeline;
    CuMemPatch *                mem_patch_shaded_frag;
    unsigned char *             pre_allocated_mem_pipeline;
    unsigned char *             pre_allocated_mem_shaded_frag;

    // pipeline obj
    Camera                      camera;
    std::vector<RenderObject>   objs;
    FrameBuffer                 frameBuffer;

    uint32_t    pipeline_flag;
    uint32_t    width;
    uint32_t    height;
    uint32_t    ss_width;
    uint32_t    ss_height;
    uint32_t    supersample_factor;

};

}



