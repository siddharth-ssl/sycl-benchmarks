#include <assert.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <block.hpp>
//#include <omp.h>
#include <grid.hpp>
#include <cuda_kernels.hpp>

int main(int argc, char** argv)
{
    std::size_t itr = 100;
    /// @brief Declearing the number of threads and blocks along with the padding points
    std::size_t block_dim[3]  = {518,518,518};
    std::size_t grid_dim[3]   = {1,1,1};
    std::size_t pad_dim[3]    = {1,1,1};
    //std::cout << block_dim[1] << " " << std::endl;

    /// @brief Declear the grids 
    grid<block<double,3>,double,3> cuda_grid_1(block_dim, grid_dim, pad_dim);
    grid<block<double,3>,double,3> cuda_grid_2(block_dim, grid_dim, pad_dim);
    //grid<double,3>* ptr_grid = new grid<double,3>(block_dim, grid_dim, pad_dim);
    
    /// @brief Allocating the memory sizes for the grids
    cuda_grid_1.allocate();
    cuda_grid_2.allocate();
    
    /// @brief debuggig prints to check the corrrect number of blocks and threads in the grids
    auto arr = cuda_grid_1.get_block_size_padded();
    std::cout << "number of threads(padded) in the x-dir" << " " << arr[0] << std::endl;
    auto ptr = cuda_grid_2.get_grid_size();
    std::cout << "number of blocks in the x-dir" << " " << ptr[0] << std::endl;
    
    //std::cout << blockDim.x << std::endl;

    /// @brief Filling the teo grids with some floating point values
    cuda_grid_1.fill(0.2);
    cuda_grid_2.fill(-0.5);

    cuda_grid_1.communic_nn_blocks();
    cuda_grid_2.communic_nn_blocks();
    
    cuda_grid_1.copy_from_host_to_device();
    cuda_grid_2.copy_from_host_to_device();

    std::cout << "Filling done" << " " << cuda_grid_1.at(10)(0,2,4) << std::endl;
    //std::cout << "Device data" << " " << cuda_grid_1.at(10).data_device()[10] << std::endl;


    
    /// ----------------------------- ///
    /// @brief CUDA IMPLEMENTATION
    /// ----------------------------- ///
    //double* device_data_1;
    //double* device_data_2;
    
    auto start = std::chrono::high_resolution_clock::now();

    for (auto it=1; it<=itr; it++)
    {
    //#pragma omp parallel 
    //{
        for (auto bpair : cuda_grid_1.get_blocks())
        {
            const auto& b_index = bpair.first;
            auto t_block_1      = bpair.second;
            auto t_block_2      = cuda_grid_2.at(b_index);

            operations(t_block_1, t_block_1.data_device(), t_block_2.data_device());
        }
    //}
        std::cout << "Iteration" << " " <<  it << std::endl;
        cuda_grid_1.copy_from_device_to_host();
        cuda_grid_2.copy_from_device_to_host();
        cuda_grid_1.communic_nn_blocks();
        cuda_grid_2.communic_nn_blocks();
        cuda_grid_1.copy_from_host_to_device();
        cuda_grid_2.copy_from_host_to_device();
        std::cout << "The cuda resuls = " <<  cuda_grid_1.at(0)(1,2,4) << std::endl;
    }

    auto stop      = std::chrono::high_resolution_clock::now();
    auto duration  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "The cuda resuls = " <<  cuda_grid_1.at(0)(1,2,4) << " in a time of " << duration.count()/(1e6*itr) << " sec" << std::endl;

    for(auto bpair : cuda_grid_1.get_blocks())
    {
        bpair.second.dealloc();
        // std::cout << bpair.first << std::endl;
    }
    cuda_grid_1.get_blocks().clear();
    for(auto bpair : cuda_grid_2.get_blocks())
    {
        bpair.second.dealloc();
        // std::cout << bpair.first << std::endl;
    }
    cuda_grid_2.get_blocks().clear();

    return 0;
}
