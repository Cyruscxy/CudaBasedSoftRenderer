#include <iostream>
#include <vector>
#include "texture.h"
#include "FreeImage.h"
#include "device_launch_parameters.h"

__device__ __inline__ uchar4
to_uchar4(float4 vec) {
    return make_uchar4(
            (unsigned char)vec.x,
            (unsigned char)vec.y,
            (unsigned char)vec.w,
            (unsigned char)vec.z);
}

__device__ __inline__ float4
operator+(const float4& lhs, const float4& rhs) {
    return make_float4(
            lhs.x + rhs.x,
            lhs.y + rhs.y,
            lhs.z + rhs.z,
            lhs.w + rhs.w);
}

__device__ __inline__ float4
operator/(const float4& lhs, const float& rhs) {
    return make_float4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}

__global__ void textureDownsampling(
        cudaSurfaceObject_t mipmap_output,
        cudaTextureObject_t mipmap_input,
        uint32_t width,
        uint32_t height
) {
    uint32_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    if ( pixelX >= width || pixelY >= height ) return;

    float inv_w = 1.0f / float(width);
    float inv_h = 1.0f / float(height);

    float4 rgba =
            tex2D<float4>(mipmap_input, inv_w * (pixelX + 0), inv_h * (pixelY + 0)) +
            tex2D<float4>(mipmap_input, inv_w * (pixelX + 1), inv_h * (pixelY + 0)) +
            tex2D<float4>(mipmap_input, inv_w * (pixelX + 1), inv_h * (pixelY + 1)) +
            tex2D<float4>(mipmap_input, inv_w * (pixelX + 0), inv_h * (pixelY + 1));
    rgba = rgba / 4.0f;

    surf2Dwrite(rgba, mipmap_output, pixelX * sizeof(float4), pixelY);
}

Texture::Texture(const std::string &image_path): base_image(image_path) {

    cudaError error;
    m_width = base_image.m_width;
    m_height = base_image.m_height;

    cudaResourceDesc resource_desc = {};
    auto* base_data = base_image.image.data();

    auto num_levels = static_cast<uint32_t>(std::log2(std::max(base_image.m_width, base_image.m_height)));

    cudaExtent extent = make_cudaExtent(m_width, m_height, 0);
    m_channel_format = cudaCreateChannelDesc<float4>();
    error = cudaMallocMipmappedArray(&m_d_mipmap_array, &m_channel_format, extent, num_levels);
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to allocate mipmap array with code " << error << std::endl;
    }

    // Copy base image to device
    cudaArray_t base;
    error = cudaGetMipmappedArrayLevel(&base, m_d_mipmap_array, 0);
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to get mipmap array with code " << error << std::endl;
    }

    cudaMemcpy3DParms copy_params = {0};
    copy_params.srcPtr          = make_cudaPitchedPtr(base_data, m_width * sizeof(float4), m_width, m_height);
    copy_params.dstArray        = base;
    copy_params.extent.width    = m_width;
    copy_params.extent.height   = m_height;
    copy_params.extent.depth    = 1;
    copy_params.kind            = cudaMemcpyHostToDevice;

    error = cudaMemcpy3D(&copy_params);
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to copy base mipmap to mipmap array with code " << error << std::endl;
    }

    for ( uint32_t level = 1; level < num_levels; ++level ) {
        cudaArray_t level_from;
        cudaArray_t level_to;

        // Get current level of mipmap
        error = cudaGetMipmappedArrayLevel(&level_to, m_d_mipmap_array, level);
        if ( error != cudaSuccess ) {
            std::cerr << "Error: Failed to get current level mipmap with code " << error << std::endl;
        }

        // Get last level of mipmap
        error = cudaGetMipmappedArrayLevel(&level_from, m_d_mipmap_array, level -1);
        if ( error != cudaSuccess ) {
            std::cerr << "Error: Failed to get last level mipmap with code " << error << std::endl;
        }

        cudaExtent level_to_size;
        error = cudaArrayGetInfo(nullptr, &level_to_size, nullptr, level_to);
        if ( error != cudaSuccess ) {
            std::cerr << "Error: Failed to get last level info with code " << error << std::endl;
        }

        uint32_t width = m_width >> level;
        uint32_t height = m_height >> level;

        cudaTextureObject_t texture_input;
        cudaResourceDesc texture_resource;
        memset(&texture_resource, 0, sizeof(cudaResourceDesc));

        texture_resource.resType            = cudaResourceTypeArray;
        texture_resource.res.array.array    = level_from;

        cudaTextureDesc texture_desc;
        memset(&texture_desc, 0, sizeof(cudaTextureDesc));

        texture_desc.normalizedCoords   = 1;
        texture_desc.filterMode         = cudaFilterModeLinear;
        texture_desc.addressMode[0]     = cudaAddressModeClamp;
        texture_desc.addressMode[1]     = cudaAddressModeClamp;
        texture_desc.addressMode[2]     = cudaAddressModeClamp;
        texture_desc.readMode           = cudaReadModeElementType;

        error = cudaCreateTextureObject(&texture_input, &texture_resource, &texture_desc, nullptr);
        if ( error != cudaSuccess ) {
            std::cerr << "Error: Failed to generate temporary texture object for mipmap level " << level - 1 << " with code " << error << std::endl;
        }

        cudaSurfaceObject_t surface_output;
        cudaResourceDesc surf_resource;
        memset(&surf_resource, 0, sizeof(cudaResourceDesc));

        surf_resource.resType            = cudaResourceTypeArray;
        surf_resource.res.array.array    = level_to;
        error = cudaCreateSurfaceObject(&surface_output, &surf_resource);
        if ( error != cudaSuccess ) {
            std::cerr << "Error: Failed to generate temporary surface object for mipmap level " << level << " with code " << error << std::endl;
        }

        dim3 block_dim(16, 16);
        dim3 grid_dim((width + block_dim.x - 1) / 16, (height + block_dim.y - 1) / 16);
        textureDownsampling<<<grid_dim, block_dim>>>(surface_output, texture_input, width, height);

        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if ( error != cudaSuccess ) {
            std::cerr << "Error: Failed to launch kernel to generate mipmap for level " << level << " with code " << error << std::endl;
        }

        error = cudaDestroyTextureObject(texture_input);
        if ( error != cudaSuccess ) {
            std::cerr << "Error: Failed to delete temporary texture object for mipmap level " << level - 1 << " with code " << error << std::endl;
        }
        error = cudaDestroySurfaceObject(surface_output);
        if ( error != cudaSuccess ) {
            std::cerr << "Error: Failed to delete temporary surface object for mipmap level " << level << " with code " << error << std::endl;
        }

    }

    resource_desc.resType               = cudaResourceTypeMipmappedArray;
    resource_desc.res.mipmap.mipmap     = m_d_mipmap_array;

    memset(&m_tex_description, 0, sizeof(m_tex_description));
    m_tex_description.normalizedCoords      = 1;
    m_tex_description.filterMode            = cudaFilterModeLinear;
    m_tex_description.addressMode[0]        = cudaAddressModeClamp;
    m_tex_description.addressMode[1]        = cudaAddressModeClamp;
    m_tex_description.addressMode[2]        = cudaAddressModeClamp;
    m_tex_description.maxMipmapLevelClamp   = num_levels - 1;
    m_tex_description.readMode              = cudaReadModeElementType;

    m_tex_obj = 0;

    error = cudaCreateTextureObject(&m_tex_obj, &resource_desc, &m_tex_description, nullptr);
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to copy image to create texture object with code " << error << std::endl;
    }
}

Texture::~Texture() {
    cudaError error;

    if ( m_d_mipmap_array ) {
        error = cudaFreeMipmappedArray(m_d_mipmap_array);
        if ( error != cudaSuccess ) {
            std::cerr << "Error: Failed to delete cudaMipmapedArray with code " << error << std::endl;
        }
    }
    if ( m_tex_obj ) {
        error = cudaDestroyTextureObject(m_tex_obj);
        if ( error != cudaSuccess ) {
            std::cerr << "Error: Failed to delete texture object with code " << error << std::endl;
        }
    }
}
