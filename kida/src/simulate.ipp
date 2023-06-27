#include "simulate.hpp"

template<class M, typename T, std::size_t D>
void 
simulate<M,T,D>::initialize_kida()
{
    T t_ke(0.0);
    for (auto& bpair : (*g).get_blocks()) 
    {
        const auto bidx = bpair.first;
        auto& b = bpair.second;
        const auto btks = b.get_btks();
        //if (!b.empty()) 
        //{
            T xreal[3];
            T feq[M::num_vars];
            T temp_f[M::num_vars];
            const auto zmin = b.get_zone_min();
            const auto zmax = b.get_zone_max();
        
            for (std::size_t r = 0; r < M::num_replicas; r++)
            {
                for (std::size_t z = zmin[2]; z <= zmax[2]; z++)
                for (std::size_t y = zmin[1]; y <= zmax[1]; y++)
                for (std::size_t x = zmin[0]; x <= zmax[0]; x++) 
                {
                    
                    xreal[0] = static_cast<T>(x) + static_cast<T>(r*0.5) + static_cast<T>(btks[0]*b.get_block_size()[0]);
                    xreal[0] = 2.0*M_PI*(xreal[0] - 0.5)/(static_cast<T>((*g).get_grid_size()[0]*b.get_block_size()[0]));

                    xreal[1] = static_cast<T>(y) + static_cast<T>(r*0.5) + static_cast<T>(btks[1]*b.get_block_size()[1]);
                    xreal[1] = 2.0*M_PI*(xreal[1] - 0.5)/(static_cast<T>((*g).get_grid_size()[1]*b.get_block_size()[1]));
                    
                    xreal[2] = static_cast<T>(z) + static_cast<T>(r*0.5) + static_cast<T>(btks[2]*b.get_block_size()[2]);
                    xreal[2] = 2.0*M_PI*(xreal[2] - 0.5)/(static_cast<T>((*g).get_grid_size()[2]*b.get_block_size()[2]));
            
                    
                    T m[5];
                    m[0] = 1.; 
                    m[1] = m_U0*sin(xreal[0])*(cos(3*xreal[1])*cos(xreal[2]) - cos(xreal[1])*cos(3*xreal[2]));
                    m[2] = m_U0*sin(xreal[1])*(cos(3*xreal[2])*cos(xreal[0]) - cos(xreal[2])*cos(3*xreal[0]));
                    m[3] = m_U0*sin(xreal[2])*(cos(3*xreal[0])*cos(xreal[1]) - cos(xreal[0])*cos(3*xreal[1]));
                    m[4] = M::T0;
            

                    (*m_lb_model).moments_to_feq(m, feq);
                    (*m_lb_model).copy_fs_to(b, feq, x, y, z);
                    t_ke += (m[1]*m[1] + m[2]*m[2] + m[3]*m[3]);
                    
                    /*
                    for (std::size_t _g = 0; _g < m_num_grps; _g++) 
                    {
                        for (std::size_t _m = 0; _m < m_num_mems; _m++) 
                        {
                            //std::cout << b.var_index(_m,_g,x[0],x[1],x[2],r) << std::endl;
                            b(_m,_g,x[0],x[1],x[2],r) = static_cast<T>(b.var_index(_m,_g,x[0],x[1],x[2],r));
                        }
                    }
                    */
                }
            }
            //printf("CPU data = %lf \n", b.data()[10]);
            //std::cout << "CPU DATA INDEX " << static_cast<T>(b.var_index(2,3,2,4,6,0)) << std::endl;
        //}
        //else
        //{
        //    std::cout << " block does not exits" << std::endl;
        //}
    }
    g->communic_nn_blocks();
    m_ke =  0.5*t_ke/(m_U0*m_U0);
    //std::cout << "kida KE " << m_ke << std::endl;
}

template<class M, typename T, std::size_t D>
T 
simulate<M,T,D>::ke()
{
    T t_ke(0.0);

    for (auto& bpair : (*g).get_blocks()) 
    {
        const auto bidx = bpair.first;
        auto& b = bpair.second;

            T feq[M::num_vars];
            const auto zmin = b.get_zone_min();
            const auto zmax = b.get_zone_max();
        
            for (std::size_t r = 0; r < M::num_replicas; r++)
            {
                for (std::size_t z = zmin[2]; z <= zmax[2]; z++)
                for (std::size_t y = zmin[1]; y <= zmax[1]; y++)
                for (std::size_t x = zmin[0]; x <= zmax[0]; x++) 
                {   
                    T m[5];
                    m_lb_model->copy_fs_from(b, feq, x, y, z);
                    m_lb_model->fs_to_moments(feq, m);
                    t_ke += (m[1]*m[1] + m[2]*m[2] + m[3]*m[3]);
                }
            }
    }
    m_ke =  0.5*t_ke/(m_U0*m_U0);

    return m_ke;
}

template<class M, typename T, std::size_t D>
void 
simulate<M,T,D>::time_step(const T& t_dt, const std::size_t& it)
{
    T t_tauN = m_tau0/t_dt;
    T t_beta = 1.0 / (1.0 + 2.0 * t_tauN);
    g->communic_nn_blocks();
    m_lb_model->collide((*g), t_beta, it);
    g->communic_nn_blocks();
    m_lb_model->advect((*g));
    //g->communic_nn_blocks();
    //sslabs::copy_margins_to_sc_nn_pads((*g), t_comm);
}