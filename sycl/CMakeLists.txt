cmake_minimum_required(VERSION 3.22.1)
project(CUDA_TEST LANGUAGES CUDA CXX)

#add_compile_definitions(CPU)

#set(CMAKE_C_COMPILER "nvc++")
#set(CMAKE_CXX_COMPILER "nvc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -m64 -cuda -gpu=cc70")

include_directories(${CMAKE_SOURCE_DIR}/../dep_cuda/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/../dep_cuda/ dep/lib)
add_subdirectory(src)
