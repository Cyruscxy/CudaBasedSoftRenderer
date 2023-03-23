#include "cuda_check.h"
#include <iostream>

void CUDA_CHECK(cudaError cuda_status) {
    if ( cuda_status != cudaSuccess ) {
        std::cerr << "CUDA ERROR with error code: " << cuda_status << std::endl;
    }
}