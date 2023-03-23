#pragma once

#include <string>
#include "picture.h"

#include <cuda.h>
#include <cuda_runtime.h>

class Texture {
public:
    Texture(const std::string &image_path);
    ~Texture();

    uint32_t getWidth() const { return m_width; }
    uint32_t getDepth() const { return m_width; }
    Picture& getBase() { return base_image; }

    cudaTextureObject_t getTexObj(){ return m_tex_obj; }

private:
    Picture base_image;
    uint32_t m_width;
    uint32_t m_height;

    cudaChannelFormatDesc   m_channel_format;

    cudaTextureDesc         m_tex_description;
    cudaTextureObject_t     m_tex_obj;

    cudaMipmappedArray_t    m_d_mipmap_array;

};
