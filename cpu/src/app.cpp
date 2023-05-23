#include <assert.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <block.hpp>
#include <omp.h>
#include <grid.hpp>
#include <lb_model.hpp>


template<class M, typename T, std::size_t D>
void 
operations_cpu (grid<matrix_block<M,T,D>,T,D>& cgrid1, grid<matrix_block<M,T,D>,T,D>& cgrid2)
{
    for (auto bpair : cgrid1.get_blocks())
    {
        const auto& t_bidx = bpair.first;
        auto t_block1      = bpair.second;
        auto t_block2      = cgrid2.at(t_bidx);

        const auto nx      = t_block1.get_block_size()[0];
        const auto ny      = t_block1.get_block_size()[1];
        const auto nz      = t_block1.get_block_size()[2];

	const auto zmin   = t_block1.get_zone_min();
	const auto zmax   = t_block1.get_zone_max();
    
        //#pragma omp parallel num_threads(8)
        //{
        #pragma omp for schedule(dynamic,1)

        for(std::size_t k=zmin[2]; k<=zmax[2]; k++)
        {
            for(std::size_t j=zmin[1]; j<=zmax[1]; j++)
            {
                for(std::size_t i=zmin[0]; i<=zmax[0]; i++)
                {
		    for(std::size_t g=0; g<t_block1.lb_model().num_grps; g++)
		    {
			for(std::size_t m=0; m<t_block1.lb_model().num_mems; m++)
			{
                           auto t_a = t_block1(m,g,i,j,k);
                           auto t_b = t_block2(m,g,i,j,k);

                           t_block1(m,g,i,j,k) = exp(-(t_a + t_a*t_b -t_b*t_b)/(t_a*t_a));
			}
		    }
                }
            }
        }
        //}
    }
    cgrid1.communic_nn_blocks();
    cgrid2.communic_nn_blocks();
}

int main(int argc, char** argv)
{
    std::size_t itr = 100;
    /// @brief Declearing the number of threads and blocks along with the padding points
    std::size_t block_dim[3]  = {32,32,32};
    std::size_t grid_dim[3]   = {1,1,1};
    std::size_t pad_dim[3]    = {1,1,1};
    //std::cout << block_dim[1] << " " << std::endl;

    /// @brief Declear the grids 
    grid<matrix_block<D3Q27SC<double>,double,3>,double,3> cuda_grid_1(block_dim, grid_dim, pad_dim);
    grid<matrix_block<D3Q27SC<double>,double,3>,double,3> cuda_grid_2(block_dim, grid_dim, pad_dim);
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

    std::cout << "filling done " << cuda_grid_2.at(0)(3,2,1,2,3) << " "  << cuda_grid_1.at(0).get_nn_blocks()[0] << std::endl;
    std::cout << " pxnn block " << cuda_grid_1.at(0).get_nn_blocks()[0] << std::endl;
    std::cout << " mxnn block " << cuda_grid_1.at(0).get_nn_blocks()[1] << std::endl;
    std::cout << " pynn block " << cuda_grid_1.at(0).get_nn_blocks()[2] << std::endl;
    std::cout << " mynn block " << cuda_grid_1.at(0).get_nn_blocks()[3] << std::endl;
    std::cout << " pznn block " << cuda_grid_1.at(0).get_nn_blocks()[4] << std::endl;
    std::cout << " mznn block " << cuda_grid_1.at(0).get_nn_blocks()[5] << std::endl;

    lb_model<D3Q27SC<double>,double> m_lb_model;
    m_lb_model.collide(cuda_grid_1, 0.01);


    /// ----------------------------- ///
    /// @brief CPU IMPLEMENTATION
    /// ----------------------------- ///

    auto start_cputime = std::chrono::high_resolution_clock::now();

    for (auto it=1; it<=itr; it++)
    {
        operations_cpu (cuda_grid_1, cuda_grid_2);
	    std::cout << "Iteration " << it << std::endl;
    }
    auto stop_cputime = std::chrono::high_resolution_clock::now();
    auto duration_cputime = std::chrono::duration_cast<std::chrono::microseconds>(stop_cputime - start_cputime);

    std::cout << "The resuls = " <<  cuda_grid_1.at(0,0,0)(3,2,1,2,3) << " in a time of " << duration_cputime.count()/(1e6*itr) << "sec" << std::endl;

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
