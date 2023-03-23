#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>

class Picture {
public:
    Picture(): m_width(0), m_height(0) {};
    Picture(const std::string &filename);
    ~Picture() = default;

    std::vector<float4> image;
    uint32_t m_width;
    uint32_t m_height;
};
