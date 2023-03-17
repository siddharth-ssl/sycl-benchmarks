#include <assert.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <block.hpp>
#include <omp.h>
#include <grid.hpp>

template<typename T>
void open_acc_operations(const std::size_t nx, 
                    const std::size_t ny, 
                    const std::size_t* zmin, 
                    const std::size_t* zmax,  
                    T* data1, 
                    T* data2, 
                    std::size_t size)
{
    #pragma acc data copyin(data1[0:size], data2[0:size])
    #pragma acc parallel loop gang device_type(nvidia) gang vector_length(32) num_workers(4) ///default(present) 
    for (auto z=zmin[2]; z<=zmax[2]; z++) 
    {
        #pragma acc loop vector 
        for (auto y=zmin[1]; y<=zmax[1]; y++) 
        {
            #pragma acc loop worker
            for (auto x=zmin[0]; x<=zmax[0]; x++)
    {
        const auto i = x + nx * (y + ny * z);
        //const auto ip = x + 1 + nx * (y + ny * z);
        data1[i] = exp(-(data1[i] + data1[i] * data2[i] - data2[i]*data2[i])/(data1[i] * data1[i]));
        //data1[i] = data1[i]*data1[i];
        //std::cout << data2[i] << std::endl;
    }
    }
    }

    //#pragma acc data copy out(data2[0:size]);
}


int main(int argc, char** argv)
{
    std::size_t itr = 100;
    /// @brief Declearing the number of threads and blocks along with the padding points
    std::size_t block_dim[3]  = {128,128,128};
    std::size_t grid_dim[3]   = {4,4,4};
    std::size_t pad_dim[3]    = {1,1,1};
    block<double,3> test_block1(block_dim,pad_dim);
    block<double,3> test_block2(block_dim,pad_dim);

    test_block1.allocate();
    test_block2.allocate();
    test_block1.fill_padded(0.2);
    test_block2.fill_padded(-0.5);
    std::cout << test_block1(10,2,4) << " " << test_block2.data()[10] << std::endl;

    /// @brief Declear the grids 
    grid<block<double,3>, double,3> cuda_grid_1(block_dim, grid_dim, pad_dim);
    grid<block<double,3>, double,3> cuda_grid_2(block_dim, grid_dim, pad_dim);
    //grid<double,3>* ptr_grid = new grid<double,3>(block_dim, grid_dim, pad_dim);

    std::cout << "initialize" << std::endl;
    
    /// @brief Allocating the memory sizes for the grids
    cuda_grid_1.allocate();
    cuda_grid_2.allocate();
    std::cout << "allocation done" << std::endl;
    
    /// @brief debuggig prints to check the corrrect number of blocks and threads in the grids
    auto arr = cuda_grid_1.get_block_size_padded();
    std::cout << "number of threads(padded) in the x-dir" << " " << arr[0] << std::endl;
    auto ptr = cuda_grid_2.get_grid_size();
    std::cout << "number of blocks in the x-dir" << " " << ptr[0] << std::endl;
    
    //std::cout << blockDim.x << std::endl;

    /// @brief Filling the two grids with some floating point values
    cuda_grid_1.fill(0.2);
    cuda_grid_2.fill(-0.5);

    cuda_grid_1.communic_nn_blocks();
    cuda_grid_2.communic_nn_blocks();

    std::cout << "filling done" << " " << cuda_grid_1.at(0).data()[10] << " "  << std::endl;


    /// ----------------------------- ///
    /// @brief OPEN ACC
    /// ----------------------------- ///


    auto start_acc = std::chrono::high_resolution_clock::now();

    for (auto it=1; it<=itr; it++)
    {
    //#pragma omp parallel 
    //{
    for (auto bpair : cuda_grid_1.get_blocks())
    {
        const auto& b_index = bpair.first;
        auto t_block_1      = bpair.second;
        auto t_block_2      = cuda_grid_2.at(b_index);

        /// @brief Getting the data array padded size from the grids
        std::size_t data_size = t_block_1.size_of().first;

        const auto& nx = t_block_1.get_block_size_padded()[0];
        const auto& ny = t_block_1.get_block_size_padded()[1];
        const auto& zmin = t_block_1.get_zone_min();
        const auto& zmax = t_block_1.get_zone_max();
        open_acc_operations(nx, ny, zmin, zmax, t_block_1.data(), t_block_2.data(), data_size);
    }
    cuda_grid_1.communic_nn_blocks();        
    cuda_grid_2.communic_nn_blocks();
    //}
    std::cout << "Iteration" << " " <<  it << std::endl;
    }
    auto stop_acc     = std::chrono::high_resolution_clock::now();
    auto duration_acc = std::chrono::duration_cast<std::chrono::microseconds>(stop_acc - start_acc);

    std::cout << "The open acc resuls = " <<  cuda_grid_1.at(0,0,0)(10,20,30) << " in a time of " << duration_acc.count()/(1e6*itr) << " sec" << std::endl;

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
