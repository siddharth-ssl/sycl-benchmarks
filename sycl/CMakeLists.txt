cmake_minimum_required(VERSION 3.22.1)
project(SYCL_TEST LANGUAGES CXX)

#add_compile_definitions(CPU)

#set(CMAKE_C_COMPILER "nvc++")
#set(CMAKE_CXX_COMPILER "nvc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -std=c++20 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -fno-sycl-id-queries-fit-in-int")

include_directories(${CMAKE_SOURCE_DIR}/../dep_sycl/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/../dep_sycl/ dep/lib)
add_subdirectory(src)
