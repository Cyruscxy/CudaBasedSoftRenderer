#pragma once

#include "device_structs.h"

namespace renderer {

__device__ __inline__ unsigned char*
acquire(CuMemPatch * mem_patch, uint32_t size_in_byte) {
    while ( atomicCAS(&(mem_patch->locked), 0, 1) != 0 ) { }
    assert(size_in_byte < mem_patch->max_length - mem_patch->used);
    unsigned char * patch = mem_patch->mem + mem_patch->used;
    mem_patch->used += size_in_byte;

    atomicExch(&(mem_patch->locked), 0);

    return patch;
}

__device__ __inline__ uint32_t *
at(IntersectionList* head, uint32_t index) {
    assert(index < head->n_nodes * IntersectionListConstants::NODE_SIZE);
    uint32_t node_index = index >> IntersectionListConstants::LOG2_NODE_SIZE;
    uint32_t offset = index & (IntersectionListConstants::NODE_SIZE - 1);

    return head->buffer[node_index] + offset;
}

__device__ __inline__ PerPixelLinkedListNode<FragmentAttribute> *
at(FragmentBuffer* head, uint32_t index) {
    assert(index < head->n_nodes * FragmentBufferConstants::NODE_SIZE);

    uint32_t node_index = index >> FragmentBufferConstants::LOG2_NODE_SIZE;
    uint32_t offset = index & (FragmentBufferConstants::NODE_SIZE - 1);

    return head->buffer[node_index] + offset;
}

__device__ __inline__ PerPixelLinkedListNode<ShadedFragment> *
at(ShadedFragmentBuffer* head, uint32_t index) {
    assert(index < head->n_nodes * ShadedFragmentBufferConstants::NODE_SIZE);

    uint32_t node_index = index >> ShadedFragmentBufferConstants::LOG2_NODE_SIZE;
    uint32_t offset = index & (ShadedFragmentBufferConstants::NODE_SIZE - 1);

    return head->buffer[node_index] + offset;
}

};

