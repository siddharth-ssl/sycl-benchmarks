#include <assert.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <grid.hpp>
#include <lb_model.hpp>
#include <simulate.hpp>
#include <sycl/sycl.hpp>
#include <dpc_common.hpp>



int main(int argc, char** argv)
{
  sycl::property_list properties{sycl::property::queue::in_order()};
  sycl::queue         q(sycl::default_selector_v, dpc_common::exception_handler, properties);
    
    std::size_t itr = 100;
    /// @brief Declearing the number of threads and blocks along with the padding points
    std::size_t block_dim[3]  = {128,128,128};
    std::size_t grid_dim[3]   = {1,1,1};
    std::size_t pad_dim[3]    = {1,1,1};
    double m_dt = 2.0*M_PI/(block_dim[0]*grid_dim[0]);
    //std::cout << block_dim[1] << " " << std::endl;

    simulate<D3Q27SC<double>,double,3> solver(block_dim, grid_dim, pad_dim, q);
    solver.initialize_kida();
    std::cout << "KIDA KE " << solver.ke()/(32*32*32*64) << std::endl;
    
    /// ----------------------------- ///
    /// @brief CPU IMPLEMENTATION
    /// ----------------------------- ///

    auto start_cputime = std::chrono::high_resolution_clock::now();
    for(std::size_t it=1; it<=itr; it++)
    {
      solver.time_step(m_dt, it, q);
      std::cout << "KIDA KE " << solver.ke()/(32*32*32*64) << std::endl;
    }

    auto stop_cputime = std::chrono::high_resolution_clock::now();
    auto duration_cputime = std::chrono::duration_cast<std::chrono::microseconds>(stop_cputime - start_cputime);

    std::cout << "Total time elapsed " << duration_cputime.count()/(1e6*itr) << std::endl;
    return 0;
}
