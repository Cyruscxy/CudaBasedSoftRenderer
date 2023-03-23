#include "picture.h"
#include "FreeImage.h"
#include <iostream>
#include <algorithm>

Picture::Picture(const std::string &filename) {

    FIBITMAP * image_bgr = FreeImage_Load(FIF_JPEG, filename.data());
    if ( image_bgr == nullptr ) {
        std::cerr << "Error Picture::load(): " << filename << " not found.\n";
    }
    uint32_t width = FreeImage_GetWidth(image_bgr);
    uint32_t height = FreeImage_GetHeight(image_bgr);
    uint32_t pitch = FreeImage_GetPitch(image_bgr);
    m_width = width;
    m_height = height;

    image.resize(width * height);

    // load base image
    BYTE * ptr = FreeImage_GetBits(image_bgr);
    for ( uint32_t j = 0; j < height; ++j ) {
        BYTE * pixel = (BYTE *) ptr;
        for ( uint32_t i = 0; i < width; ++i ) {
            image[i + j * width].x = pixel[2];
            image[i + j * width].y = pixel[1];
            image[i + j * width].z = pixel[0];
            image[i + j * width].w = 255.0f;
            pixel += 3;
        }
        ptr += pitch;
    }

    FreeImage_Unload(image_bgr);
}

