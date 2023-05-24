#pragma once

#include <iostream>
#include "matrix_block.hpp"
#include "grid.hpp"
#include "lb_model.hpp"

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
    grid_type* g;

public:
    simulate(std::size_t* t_block_dim,
             std::size_t* t_grid_dim,
             std::size_t* t_pad_dim) 
    {
        m_lb_model = new lb_model<M,T>;
        g          = new grid_type(t_block_dim, t_grid_dim, t_pad_dim);
        
        g->allocate();
        m_U0   = 0.1 * std::sqrt(5.0 * M::T0/3.0);
        m_nu   = (m_U0 * 1.0)/1000.0;
        m_tau0 = m_nu/M::T0;
        m_ke   = 0.0; 
    }

    void 
    time_step(const T& t_dt);

    void 
    initialize_kida();

    T 
    ke();
    
    ~simulate()
    {
        
    }
};
