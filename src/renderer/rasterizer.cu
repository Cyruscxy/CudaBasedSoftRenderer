#include "rasterizer.h"
#include "pipeline_funtions.h"

namespace renderer {

Rasterizer::Rasterizer(const std::vector<std::string>& meshes, const std::vector<std::vector<Vector3>>& trans,
                       const std::vector<std::string>& texture_files, const std::vector<float>& opacity, Camera camera_)
        : camera(std::move(camera_))
        , frameBuffer(1280, 720)

        , depth_buffer(nullptr)
        , device_frame_buffer(nullptr)
        , device_resolved_buffer(nullptr)
        , start_offset_obj(nullptr)
        , start_offset_all(nullptr)
        , mem_patch_pipeline(nullptr)
        , mem_patch_shaded_frag(nullptr)
        , pre_allocated_mem_pipeline(nullptr)
        , pre_allocated_mem_shaded_frag(nullptr)

        , fragment_buffer(nullptr)
        , shaded_fragment_buffer(nullptr)
        , intersection_list(nullptr)
        , aoit_fragments(nullptr)

        , width(1280)
        , height(720)
        , supersample_factor(1)
        , pipeline_flag(0) {

    n_objs = meshes.size();
    if ( meshes.size() > trans.size() ) throw std::runtime_error("Insufficient space information for given meshes!\n");
    for ( uint32_t i = 0; i < n_objs; ++i ) {
        objs.emplace_back(meshes[i], texture_files[i], trans[i][0], trans[i][1], trans[i][2], opacity[i]);
        n_faces.push_back(objs[i].mesh->nFaces());
        n_vertices.push_back(objs[i].mesh->nVertices());
        n_tex_coords.push_back(objs[i].mesh->nTexCoords());
        n_normals.push_back(objs[i].mesh->nNormals());
    }

    ss_height = supersample_factor * height;
    ss_width = supersample_factor * width;

    for ( uint32_t i = 0; i < n_objs; ++i ) {
        device_vertices.push_back(nullptr);
        device_tex_coords.push_back(nullptr);
        device_normals.push_back(nullptr);
        shaded_vertices.push_back(nullptr);
        shaded_normal.push_back(nullptr);
        visible_table.push_back(nullptr);
        faces.push_back(nullptr);
    }

}

Rasterizer::~Rasterizer() {
    clear();
}

void Rasterizer::run() {
    setup();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    cudaEventRecord(start, 0);
    for ( uint32_t index = 0; index < n_objs; ++ index) {
        load(index);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "Load time: " << elapsed_time << " ms" << std::endl;

    uint32_t grid_dim_x = (ss_width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t grid_dim_y = (ss_height + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid_dim(grid_dim_x, grid_dim_y);

    cudaEventRecord(start, 0);
    for ( uint32_t obj_index = 0; obj_index < n_objs; ++obj_index ) {
        setTransform(obj_index);

        // shade vertices
        shadeVertices(obj_index);

        // view frustum culling
        cullFaces(obj_index);

        // clip to screen transform
        clip2Screen(obj_index);

        // cull back face or reorder the orders of back face vertices
        backFaceProcessing(obj_index);

        // get faces intersected with each tile
        tile(obj_index);

        // rasterize
        rasterize(obj_index);

        // shade fragments
        shadeFragments(obj_index);

        // reset buffer
        resetPreAllocatedMem();

    }

    computeAOITNodes();

    alphaBlending();

    writeBack();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;

    CUDA_CHECK( cudaMemcpy(frameBuffer.image.data(), device_resolved_buffer, sizeof(float4) * frameBuffer.width * frameBuffer.height, cudaMemcpyDeviceToHost) );

    frameBuffer.saveAsPNG();

    for ( uint32_t index = 0; index < n_objs; ++ index) {
        clearDeviceObj(index);
    }
}


void Rasterizer::clear() {
    for ( uint32_t i = 0; i < n_objs; ++i) {
        clearDeviceObj(i);
    }

    CUDA_CHECK( cudaFree(fragment_buffer) );
    fragment_buffer = nullptr;
    CUDA_CHECK( cudaFree(shaded_fragment_buffer) );
    shaded_fragment_buffer = nullptr;
    CUDA_CHECK( cudaFree(mem_patch_pipeline) );
    mem_patch_pipeline = nullptr;
    CUDA_CHECK( cudaFree(mem_patch_shaded_frag) );
    mem_patch_shaded_frag = nullptr;
    CUDA_CHECK( cudaFree(pre_allocated_mem_pipeline) );
    pre_allocated_mem_pipeline = nullptr;
    CUDA_CHECK( cudaFree(pre_allocated_mem_shaded_frag) );
    pre_allocated_mem_shaded_frag = nullptr;

    CUDA_CHECK( cudaFree(depth_buffer) );
    depth_buffer = nullptr;
    CUDA_CHECK( cudaFree(device_frame_buffer) );
    device_frame_buffer = nullptr;
    CUDA_CHECK( cudaFree(device_resolved_buffer) );
    device_resolved_buffer = nullptr;
    CUDA_CHECK( cudaFree(intersection_list) );
    intersection_list = nullptr;

    CUDA_CHECK( cudaFree(aoit_fragments) );
    aoit_fragments = nullptr;
    CUDA_CHECK( cudaFree(start_offset_all) );
    start_offset_all = nullptr;
    CUDA_CHECK( cudaFree(start_offset_obj) );
    start_offset_obj = nullptr;
}

void Rasterizer::setup() {

    clear();

    uint32_t n_tiles = ((ss_width + TILE_SIZE - 1) / TILE_SIZE) * ((ss_height + TILE_SIZE - 1) / TILE_SIZE);

    CUDA_CHECK( cudaMalloc(&depth_buffer, sizeof(float) * ss_width * ss_height) );
    CUDA_CHECK( cudaMalloc(&device_frame_buffer, sizeof(float4) * ss_width * ss_height) );
    CUDA_CHECK( cudaMalloc(&device_resolved_buffer, sizeof(float4) * width * height) );
    CUDA_CHECK( cudaMalloc(&aoit_fragments, sizeof(AOITNode) * (AOIT_NODE_CNT) * ss_width * ss_height) );
    CUDA_CHECK( cudaMalloc(&fragment_buffer, sizeof(FragmentBuffer) * n_tiles) );
    CUDA_CHECK( cudaMalloc(&intersection_list, sizeof(IntersectionList) * n_tiles) );
    CUDA_CHECK( cudaMalloc(&start_offset_obj, sizeof(int32_t) * ss_width * ss_height) );
    CUDA_CHECK( cudaMalloc(&start_offset_all, sizeof(int32_t) * ss_width * ss_height) );
    CUDA_CHECK( cudaMalloc(&shaded_fragment_buffer, sizeof(ShadedFragmentBuffer) * n_tiles) );
    CUDA_CHECK( cudaMalloc(&mem_patch_pipeline, sizeof(CuMemPatch)) );
    CUDA_CHECK( cudaMalloc(&mem_patch_shaded_frag, sizeof(CuMemPatch)) );
    CUDA_CHECK( cudaMalloc(&pre_allocated_mem_pipeline, sizeof(unsigned char) * MEM_PATCH_SIZE_PIPELINE) );
    CUDA_CHECK( cudaMalloc(&pre_allocated_mem_shaded_frag, sizeof(unsigned char) * MEM_PATCH_SIZE_SHADED_FRAGMENT) );
    CUDA_CHECK( cudaMemset(pre_allocated_mem_pipeline, 0, sizeof(unsigned char) * MEM_PATCH_SIZE_PIPELINE) );
    CUDA_CHECK( cudaMemset(pre_allocated_mem_shaded_frag, 0, sizeof(unsigned char) * MEM_PATCH_SIZE_SHADED_FRAGMENT) );
    CUDA_CHECK( cudaMemset(device_frame_buffer, 0, sizeof(float4) * ss_width * ss_height) ) ;
    CUDA_CHECK( cudaMemset(fragment_buffer, 0, sizeof(FragmentBuffer) * n_tiles) ) ;
    CUDA_CHECK( cudaMemset(intersection_list, 0, sizeof(IntersectionList) * n_tiles) ) ;
    CUDA_CHECK( cudaMemset(shaded_fragment_buffer, 0, sizeof(ShadedFragmentBuffer) * n_tiles) ) ;

    // set global parameters
    Parameters pipeline_params {};
    pipeline_params.width = ss_width;
    pipeline_params.height = ss_height;

    pipeline_params.ground_energy = make_float3(0.01f, 0.01f, 0.01f);

    pipeline_params.sky_energy = make_float3(0.5f, 0.5f, 0.5f);
    pipeline_params.sky_direction = make_float3(0.0f, 0.0f, 1.0f);

    //pipeline_params.sun_energy = make_float3(1.0f, 1.0f, 1.0f);
    pipeline_params.sun_energy = make_float3(0.5f, 0.5f, 0.5f);
    Vector3 sun_direction = (camera.trans->local_to_world() * Vector4 {1.0f, 1.0f, 1.0f, 0.0f}).xzy();
    sun_direction.normalize();
    pipeline_params.sun_direction = make_float3(sun_direction.x, sun_direction.y, sun_direction.z);

    CUDA_CHECK( cudaMemcpyToSymbol(PipelineParams, &pipeline_params, sizeof(Parameters)) );

    float3 clip_to_fb_scale_ = make_float3((float)ss_width/ 2.0f, (float)ss_height / 2.0f, 0.5f);
    float3 clip_to_fb_offset_ = make_float3(0.5f * ss_width , 0.5f * ss_height, 0.5f);
    CUDA_CHECK( cudaMemcpyToSymbol(clip_to_fb_offset, &clip_to_fb_offset_, sizeof(float3)) );
    CUDA_CHECK( cudaMemcpyToSymbol(clip_to_fb_scale, &clip_to_fb_scale_, sizeof(float3)) );

    dim3 grid_dim((ss_width + TILE_SIZE - 1) / TILE_SIZE, (ss_height + TILE_SIZE - 1) / TILE_SIZE );
    setDeviceBuffer<AOIT_NODE_CNT><<<grid_dim, block_dim>>>(depth_buffer, start_offset_all, mem_patch_pipeline, mem_patch_shaded_frag, pre_allocated_mem_pipeline,
                                                            pre_allocated_mem_shaded_frag, aoit_fragments, ss_width, ss_height);
    cudaDeviceSynchronize();
    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to initial depth buffer with code " << error << std::endl;
    }
}

void Rasterizer::load(uint32_t index) {
    auto& obj = objs[index];
    auto& polygon_indices = obj.mesh->polygons;
    auto& vertex_coords = obj.mesh->vertexCoordinates;
    auto& vertex_tex_coords = obj.mesh->vertexTexCoords;
    auto& vertex_normals = obj.mesh->vertexNormals;

    CUDA_CHECK( cudaMalloc(&device_vertices[index], sizeof(float3) * n_vertices[index]) );
    CUDA_CHECK( cudaMalloc(&device_tex_coords[index], sizeof(float2) * n_tex_coords[index]) );
    CUDA_CHECK( cudaMalloc(&device_normals[index], sizeof(float3) * n_normals[index]) );
    CUDA_CHECK( cudaMalloc(&shaded_vertices[index], sizeof(float4) * n_vertices[index]) );
    CUDA_CHECK( cudaMalloc(&shaded_normal[index], sizeof(float3) * n_normals[index]) );
    CUDA_CHECK( cudaMalloc(&visible_table[index], sizeof(uint32_t) * n_faces[index]) );
    CUDA_CHECK( cudaMalloc(&faces[index], sizeof(TriangleFaceIndex) * n_faces[index]) );

    CUDA_CHECK( cudaMemset(visible_table[index], 0, sizeof(uint32_t) * n_faces[index]) );

    // copy vertex attributes data & indices
    CUDA_CHECK( cudaMemcpy(device_vertices[index], vertex_coords.data(), sizeof(float3) * n_vertices[index], cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(device_tex_coords[index], vertex_tex_coords.data(), sizeof(float2) * n_tex_coords[index], cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(device_normals[index], vertex_normals.data(), sizeof(float3) * n_normals[index], cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(faces[index], polygon_indices.data(), sizeof(TriangleFaceIndex) * n_faces[index], cudaMemcpyHostToDevice) );

    // set texture
    textures.push_back(std::make_unique<Texture>(obj.texture_file));

}


void Rasterizer::setTransform(uint32_t obj_index) {
    // set local-to-clip & normal-to-world transform matrix
    Mat4 world_to_clip = camera.projection() * camera.trans->world_to_local();
    Mat4 local_to_clip = world_to_clip * objs[obj_index].trans->local_to_world();
    CUDA_CHECK( cudaMemcpyToSymbol(device_local_to_clip, &local_to_clip, sizeof(Mat4)) );

    auto normal_to_world = [](Mat4 const& l2w ) {
        return Mat4::inverse( Mat4::transpose(Mat4 {
                Vector4 { l2w[0][0], l2w[0][1], l2w[0][2], 0.0f },
                Vector4 { l2w[1][0], l2w[1][1], l2w[1][2], 0.0f },
                Vector4 { l2w[2][0], l2w[2][1], l2w[2][2], 0.0f },
                Vector4 {      0.0f,      0.0f,      0.0f, 1.0f }
        }));
    };
    Mat4 n2w = normal_to_world(objs[obj_index].trans->local_to_world());
    CUDA_CHECK( cudaMemcpyToSymbol(device_normal_to_world, &n2w, sizeof(Mat4)) );
}

void Rasterizer::clearDeviceObj(uint32_t index) {
    CUDA_CHECK( cudaFree(device_vertices[index]) );
    device_vertices[index] = nullptr;
    CUDA_CHECK( cudaFree(device_tex_coords[index]) );
    device_tex_coords[index] = nullptr;
    CUDA_CHECK( cudaFree(device_normals[index]) );
    device_normals[index] = nullptr;
    CUDA_CHECK( cudaFree(shaded_vertices[index]) );
    shaded_vertices[index] = nullptr;
    CUDA_CHECK( cudaFree(shaded_normal[index]) );
    shaded_normal[index] = nullptr;
    CUDA_CHECK( cudaFree(visible_table[index]) );
    visible_table[index] = nullptr;
    CUDA_CHECK( cudaFree(faces[index]) );
    faces[index] = nullptr;
}

void Rasterizer::resetPreAllocatedMem() {
    uint32_t grid_dim_x = (ss_width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t grid_dim_y = (ss_height + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid_dim(grid_dim_x, grid_dim_y);

    resetDynamicMem<<<grid_dim, 1>>>(fragment_buffer, intersection_list, mem_patch_pipeline);

    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to launch kernel deleteDynamicMem with code " << error << std::endl;
    }
}

void Rasterizer::shadeVertices(uint32_t obj_index) {
    uint32_t n_blk = (std::max(n_normals[obj_index], n_vertices[obj_index]) + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE);
    vertexShading<<<n_blk, block_dim>>>(device_vertices[obj_index], device_normals[obj_index], shaded_vertices[obj_index],
                                        shaded_normal[obj_index], n_vertices[obj_index], n_normals[obj_index] );

    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to launch kernel vertexShading with code " << error << std::endl;
    }
}

void Rasterizer::cullFaces(uint32_t obj_index) {
    uint32_t n_blk = (n_faces[obj_index] + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE);
    viewFrustumCulling<<<n_blk, block_dim>>>(shaded_vertices[obj_index], faces[obj_index], visible_table[obj_index], n_faces[obj_index]);
    cudaDeviceSynchronize();

    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to launch kernel faceCulling with code " << error << std::endl;
    }
}

void Rasterizer::clip2Screen(uint32_t obj_index) {
    uint32_t n_blk = (n_vertices[obj_index] + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE);
    clipToScreen<<<n_blk, block_dim>>>(shaded_vertices[obj_index], n_vertices[obj_index]);
    cudaDeviceSynchronize();

    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to launch kernel clipToScreen with code " << error << std::endl;
    }
}


void Rasterizer::backFaceProcessing(uint32_t obj_index) {
    uint32_t n_blk = (n_faces[obj_index] + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE);
    if( (pipeline_flag & PipelineMask_RenderingMode) == Pipeline_Rendering_Opaque ) {
        backFaceCulling<<<n_blk, block_dim>>>(shaded_vertices[obj_index], faces[obj_index], visible_table[obj_index], n_faces[obj_index]);
    }
    else if ( (pipeline_flag & PipelineMask_RenderingMode) == Pipeline_Rendering_Transparent ) {
        backFaceReordering<<<n_blk, block_dim>>>(shaded_vertices[obj_index], faces[obj_index], device_normals[obj_index], n_faces[obj_index]);
    }
    else {
        throw std::runtime_error( "Unknown rendering mode flag." );
    }

    cudaDeviceSynchronize();
    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to process back faces with code " << error << std::endl;
    }
}

void Rasterizer::tile(uint32_t obj_index) {
    uint32_t grid_dim_x = (ss_width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t grid_dim_y = (ss_height + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid_dim(grid_dim_x, grid_dim_y);
    tiling<<<grid_dim, block_dim>>>(shaded_vertices[obj_index], faces[obj_index], intersection_list, mem_patch_pipeline,
                                    visible_table[obj_index], n_faces[obj_index]);
    /*uint32_t n_grid = (n_faces[obj_index] + BLK_SIZE - 1) / BLK_SIZE;
    tiling_f<<<n_grid, block_dim>>>(shaded_vertices[obj_index], faces[obj_index], intersection_list, mem_patch,
                                    visible_table[obj_index], width, height, n_faces[obj_index]);*/
    cudaDeviceSynchronize();

    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to launch kernel tiling with code " << error << std::endl;
    }
}


void Rasterizer::rasterize(uint32_t obj_index) {
    uint32_t grid_dim_x = (ss_width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t grid_dim_y = (ss_height + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid_dim(grid_dim_x, grid_dim_y);

    if ( (pipeline_flag & PipelineFlags::Pipeline_DepthWriteDisableBit) | (pipeline_flag & PipelineFlags::Pipeline_Rendering_Transparent) ) {
        rasterizeWithEarlyZ<PipelineFlags::Pipeline_DepthWriteDisableBit><<<grid_dim, block_dim>>>(
                shaded_vertices[obj_index], shaded_normal[obj_index], device_tex_coords[obj_index], depth_buffer, intersection_list,
                faces[obj_index], fragment_buffer, start_offset_obj, mem_patch_pipeline, ss_width, ss_height, n_faces[obj_index]);
    }
    else {
        rasterizeWithEarlyZ<0><<<grid_dim, block_dim>>>(
                shaded_vertices[obj_index], shaded_normal[obj_index], device_tex_coords[obj_index], depth_buffer,intersection_list,
                faces[obj_index], fragment_buffer, start_offset_obj, mem_patch_pipeline, ss_width, ss_height, n_faces[obj_index]);
    }
    cudaDeviceSynchronize();

    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to launch kernel rasterizeWithEarlyZ with code " << error << std::endl;
    }
}

void Rasterizer::shadeFragments(uint32_t obj_index) {
    uint32_t grid_dim_x = (ss_width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t grid_dim_y = (ss_height + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid_dim(grid_dim_x, grid_dim_y);
    if ( (pipeline_flag & PipelineMask_RenderingMode) == Pipeline_Rendering_Opaque ) {
        fragmentShading<Pipeline_Rendering_Opaque><<<grid_dim, block_dim>>>(fragment_buffer, shaded_fragment_buffer, mem_patch_shaded_frag,
                                                                            textures[obj_index]->getTexObj(), start_offset_obj, start_offset_all,
                                                                            objs[obj_index].alpha, ss_width, ss_height);
    }
    else if ( (pipeline_flag & PipelineMask_RenderingMode) == Pipeline_Rendering_Transparent ) {
        fragmentShading<Pipeline_Rendering_Transparent><<<grid_dim, block_dim>>>(fragment_buffer, shaded_fragment_buffer, mem_patch_shaded_frag,
                                                                                 textures[obj_index]->getTexObj(), start_offset_obj, start_offset_all,
                                                                                 objs[obj_index].alpha, ss_width, ss_height);
    }
    else {
        throw std::runtime_error("Wrong pipeline flags in rendering mode\n");
    }
    cudaDeviceSynchronize();
    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to launch kernel fragmentShading with code " << error << std::endl;
    }


}

void Rasterizer::computeAOITNodes() {
    uint32_t grid_dim_x = (ss_width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t grid_dim_y = (ss_height + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid_dim(grid_dim_x, grid_dim_y);
    computeVisNode<AOIT_NODE_CNT><<<grid_dim, block_dim>>>(shaded_fragment_buffer, aoit_fragments, depth_buffer, start_offset_all,
                                                           ss_width, ss_height);

    cudaDeviceSynchronize();
    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to launch kernel computeVisNode with code " << error << std::endl;
    }
}

void Rasterizer::alphaBlending() {
    uint32_t grid_dim_x = (ss_width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t grid_dim_y = (ss_height + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid_dim(grid_dim_x, grid_dim_y);

    if ((pipeline_flag & PipelineMask_RenderingMode) == Pipeline_Rendering_Transparent) {
        blending<AOIT_NODE_CNT, Pipeline_Rendering_Transparent>
                <<<grid_dim, block_dim>>>(aoit_fragments, device_frame_buffer, ss_width, ss_height);
    }
    else if ((pipeline_flag & PipelineMask_RenderingMode) == Pipeline_Rendering_Opaque) {
        blending<AOIT_NODE_CNT, Pipeline_Rendering_Opaque>
        <<<grid_dim, block_dim>>>(aoit_fragments, device_frame_buffer, ss_width, ss_height);
    }
    else {
        throw std::runtime_error("Wrong flag of rendering mode!\n");
    }
    cudaDeviceSynchronize();
    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to launch kernel blending with code " << error << std::endl;
    }
}

void Rasterizer::writeBack() {
    uint32_t grid_dim_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t grid_dim_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid_dim(grid_dim_x, grid_dim_y);
    resolve<<<grid_dim, block_dim>>>(device_frame_buffer, device_resolved_buffer, width, height, supersample_factor);
    cudaDeviceSynchronize();

    cudaError error = cudaGetLastError();
    if ( error != cudaSuccess ) {
        std::cerr << "Error: Failed to launch kernel writeFrame with code " << error << std::endl;
    }
}


}