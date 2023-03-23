#pragma once

enum PipelineFlags : unsigned int{
    Pipeline_DepthWriteDisableBit   = 0x8000,

    Pipeline_Rendering_Opaque       = 0x0000,
    Pipeline_Rendering_Transparent  = 0x0001,

    PipelineMask_RenderingMode      = 0x000f,
};

constexpr unsigned int TILE_SIZE = 16;
constexpr unsigned int BLK_SIZE = TILE_SIZE * TILE_SIZE;
constexpr unsigned int LOG2_BLK_SIZE = 8;
const dim3 block_dim(TILE_SIZE, TILE_SIZE);

struct Parameters {
    unsigned int width;
    unsigned int height;
    float3 sun_energy;
    float3 sun_direction;
    float3 sky_energy;
    float3 sky_direction;
    float3 ground_energy;
};


