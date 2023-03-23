#pragma once

#include <vector>
#include <iostream>
#include "vector4.h"
#include "vector3.h"
#include "FreeImage.h"

namespace renderer {

using geometry::Vector4;

struct FrameBuffer {
    FrameBuffer(uint32_t w, uint32_t h) : width(w), height(h) {
        image.assign(w * h, Vector4::zero());
    }

    void saveAsPNG() {
        FIBITMAP *tmp = FreeImage_Allocate(width, height, 32);
        if (!tmp) std::cerr << "Failed to allocate memory for result!" << std::endl;
        uint32_t pitch = FreeImage_GetPitch(tmp);

        auto to_pixel = [](float val) {
            return static_cast<unsigned char>(std::max(std::min(val, 255.0f), 0.0f));
        };

        BYTE *ptr = FreeImage_GetBits(tmp);
        for (uint32_t j = 0; j < height; ++j) {
            BYTE *pixel = (BYTE *) ptr;
            for (uint32_t i = 0; i < width; ++i) {
                pixel[0] = to_pixel(image[i + width * j].z);
                pixel[1] = to_pixel(image[i + width * j].y);
                pixel[2] = to_pixel(image[i + width * j].x);
                //pixel[3] = to_pixel(image[i + width * j].w);
                pixel[3] = 255;
                pixel += 4;
            }
            ptr += pitch;
        }

        if ( !FreeImage_Save(FIF_PNG, tmp, "../output/out.png") ) std::cout << "Failed to save image!" << std::endl ;
        FreeImage_Unload(tmp);
    };

    uint32_t width;
    uint32_t height;
    std::vector<Vector4> image;
};

}
