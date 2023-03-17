cmake_minimum_required(VERSION 3.22.1)
project(CUDA_TEST)

#add_compile_definitions(CPU)

#set(CMAKE_C_COMPILER "nvc++")
#set(CMAKE_CXX_COMPILER "nvc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -fast -Minfo=accel -gpu=managed,cc70")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(${CMAKE_SOURCE_DIR}/../dep/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/../dep/ dep/lib)
add_subdirectory(src)
