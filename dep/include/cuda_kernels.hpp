#include <grid.hpp>
#include <iostream>

/// @brief Global CUDA kernels
/// @tparam T 
/// @tparam D 
/// @return 

__device__ inline std::size_t 
bindx(std::size_t bx, std::size_t by, std::size_t bz, 
    std::size_t nbx, std::size_t nby, std::size_t nbz);
__device__ inline std::size_t
tindx(std::size_t x, std::size_t y, std::size_t z, 
    std::size_t npx, std::size_t npy, std::size_t npz);

template<typename T>
__global__ void 
fill_device(const T &t_val, std::size_t npx, std::size_t npy, std::size_t npz, T* device_data);

template<typename T, std::size_t D>
void 
fill(const T &t_val, block<T,D>& cgrid, T* device_data);

template<typename T>
__global__ void 
operations_device(std::size_t npx, std::size_t npy, std::size_t npz, T* device_data_1, T* device_data_2);

template<typename T, std::size_t D>
void 
operations(block<T,D>& cgrid, T* device_data_1, T* device_data_2);
