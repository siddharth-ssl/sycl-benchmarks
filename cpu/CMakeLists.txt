cmake_minimum_required(VERSION 3.22.1)
project(CUDA_TEST LANGUAGES CXX)

#add_compile_definitions(CPU)
#set(CMAKE_C_COMPILER "nvc++")
#set(CMAKE_CXX_COMPILER "nvc++")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -DSSLABS_AVX")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(${CMAKE_SOURCE_DIR}/../dep/include)
include_directories(${CMAKE_SOURCE_DIR}/../lbm/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/../dep/ dep/lib)
add_subdirectory(${CMAKE_SOURCE_DIR}/../lbm/ lbm/lib)
add_subdirectory(src)
