#pragma once

#include <iostream>
#include "matrix_block.hpp"
#include "grid.hpp"
#include "lb_param_host.hpp"
#include "collide_sycl.hpp"
#include <sycl/sycl.hpp>

template<class M, typename T, std::size_t D>
class simulate
{
private:
    /* data */
    using lb_stencil = M;
    using block_type = matrix_block<lb_stencil, T, D>;
    using grid_type  = grid<block_type,T,D>;
    
    T m_U0;
    T m_tau0;
    T m_nu;
    T m_ke;
    lb_model<M,T>* m_lb_model;
    lb_param_host<lb_model<M,T>,T>* d_lb_param; 
    collide_sycl<lb_param_host<lb_model<M,T>,T>,T,D>* sycl_collide;
    grid_type* g;
    T* device_data;
    std::size_t m_size;

public:
    simulate(std::size_t* t_block_dim,
             std::size_t* t_grid_dim,
             std::size_t* t_pad_dim,
             sycl::queue& q) 
    {
        m_lb_model = new lb_model<M,T>;
        g          = new grid_type(t_block_dim, t_grid_dim, t_pad_dim);
        
        g->allocate();
        m_size = M::num_replicas*M::num_vars*(t_block_dim[0]+2 * t_pad_dim[0])*(t_block_dim[1]+2 * t_pad_dim[1])*(t_block_dim[2]+2 * t_pad_dim[2])*sizeof(T);
        device_data = (T*)sycl::malloc_device(m_size, q);

        lb_param_host<lb_model<M,T>,T> h_lb_param(q);
        d_lb_param = (lb_param_host<lb_model<M,T>,T>*)sycl::malloc_device(sizeof(lb_param_host<lb_model<M,T>,T>), q);
        q.memcpy(d_lb_param, &h_lb_param, sizeof(lb_param_host<lb_model<M,T>,T>)).wait();

        std::cout << "Running on " << std::endl;
        std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        m_U0   = 0.1 * std::sqrt(5.0 * M::T0/3.0);
        m_nu   = (m_U0 * 1.0)/1000.0;
        m_tau0 = m_nu/M::T0;
        m_ke   = 0.0; 
        
    }

    void
    copy_device_to_host(sycl::queue& q);

    void
    copy_host_to_device(sycl::queue& q);

    T  
    time_step(const T& t_dt, const std::size_t& it, sycl::queue& q);

    void 
    initialize_kida();

    T 
    ke();
    
    ~simulate()
    {
        
    }
};
