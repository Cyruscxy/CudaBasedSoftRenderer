cmake_minimum_required(VERSION 3.20)

project(cudaRenderer LANGUAGES CXX CUDA )
find_package(CUDAToolkit 11.0 REQUIRED)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CUDA_STANDARD 17)
include_directories(include/geometry/)
include_directories(include/utils/)
include_directories(include/renderer/)
include_directories(FreeImage/include/)

file(GLOB_RECURSE CXX_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CUDA_SRCS ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(cudaRenderer ${CXX_SRCS} ${CUDA_SRCS} main.cpp)

target_compile_features(cudaRenderer PRIVATE cxx_std_17)
if ( ${CMAKE_SYSTEM_NAME} MATCHES "Windows" )
    target_link_libraries(cudaRenderer ${PROJECT_SOURCE_DIR}/FreeImage/lib/FreeImage.lib)
else()
    target_link_libraries(cudaRenderer ${PROJECT_SOURCE_DIR}/FreeImage/lib/libfreeimage.a)
endif()

target_link_libraries(cudaRenderer CUDA::cudart)

set_target_properties(cudaRenderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

