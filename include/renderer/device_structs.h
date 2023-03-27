#pragma once

#include <cuda_runtime.h>
#include <assert.h>

// struct used in kernel function
struct CuMemPatch {
    unsigned char * mem;
    uint32_t max_length;    // in bytes
    uint32_t used;          // in bytes
    uint32_t locked;
};
//===========================================
namespace IntersectionListConstants{
    constexpr uint32_t MAX_NODE_NUM = 128;
    constexpr uint32_t NODE_SIZE = 1024;
    constexpr uint32_t LOG2_NODE_SIZE = 10;
};

// list recording which triangles intersected with a tile
struct IntersectionList {
    uint32_t * buffer[IntersectionListConstants::MAX_NODE_NUM];
    uint32_t n_nodes;
    uint32_t cnt;
    uint32_t locked;
};
//===========================================
namespace FragmentBufferConstants {
    constexpr uint32_t NODE_SIZE = 1024;
    constexpr uint32_t LOG2_NODE_SIZE = 10;
    constexpr uint32_t MAX_NODE_NUM = 16;
}

struct FragmentBuffer {
    PerPixelLinkedListNode<FragmentAttribute> * buffer[FragmentBufferConstants::MAX_NODE_NUM];
    uint32_t n_nodes;
    uint32_t cnt;
};
//===========================================
namespace ShadedFragmentBufferConstants {
    constexpr uint32_t NODE_SIZE = 1024;
    constexpr uint32_t LOG2_NODE_SIZE = 10;
    constexpr uint32_t MAX_NODE_NUM = 16;
}

struct ShadedFragmentBuffer {
    PerPixelLinkedListNode<ShadedFragment> * buffer[ShadedFragmentBufferConstants::MAX_NODE_NUM];
    uint32_t cnt;
    uint32_t n_nodes;
};
//===========================================

struct VisibleFaces {
    uint32_t*   faces;
    uint32_t    cnt;
    uint32_t    lock;
};

//===========================================



