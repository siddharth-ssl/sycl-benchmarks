#pragma once
#include <iostream>
#include <sycl/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "lb_param_host.hpp"
//#include <taral/domain.hpp>
//#include <taral/lbm/lbm.h>
//#include <sslabs/api3D/api3D.h>

template<class M, typename T, std::size_t D>
class collide_sycl
{
public:
    using block_type = matrix_block<D3Q27SC<T>, T, D>;
    using grid_type  = grid<block_type,T,D>;

    collide_sycl() 
    {

    }

    void 
    fs_to_moments(T* f, T* m, M* lb_param);
    
    void 
    moments_to_feq(T* f, T* m, M* lb_param);
    
    std::size_t 
    idx(const std::size_t t_m,
        const std::size_t t_g,
        const std::size_t t_x,
        const std::size_t t_y,
        const std::size_t t_z,
        const std::size_t t_rpl,
        const std::size_t num_padded_x, 
        const std::size_t num_padded_y,
        const std::size_t num_padded_z, 
        const std::size_t num_mems,
        const std::size_t num_replicas)
        {
          return t_m +
            num_mems *
             (t_x + num_padded_x *
                      (t_y + num_padded_y *
                               (t_z + num_padded_z *
                                    (t_rpl +
                                         num_replicas * (t_g)))));
        }

    std::size_t 
    idx(const std::size_t t_dv,
        const std::size_t t_x,
        const std::size_t t_y,
        const std::size_t t_z,
        const std::size_t t_rpl,
        const std::size_t num_padded_x, 
        const std::size_t num_padded_y,
        const std::size_t num_padded_z, 
        const std::size_t num_mems,
        const std::size_t num_replicas)
        {
          return t_x + num_padded_x *
                      (t_y + num_padded_y *
                               (t_z + num_padded_z *
                                    (t_rpl +
                                         num_replicas * (t_dv))));
        }

    int
    indx(int i, int j, int N1, int N2) { return i + N1*j; }

    void 
    copy_fs_from(T* f,
                 T* fs,
                 std::size_t x, 
                 std::size_t y,
                 std::size_t z,
                 std::size_t r, 
                 const std::size_t npx,
                 const std::size_t npy, 
                 const std::size_t npz, 
                 M* lb_param);
    
    void 
    copy_fs_from(T* f,
                 T* fs,
                 std::size_t x, 
                 std::size_t y,
                 std::size_t z,
                 std::size_t r, 
                 const std::size_t npx,
                 const std::size_t npy, 
                 const std::size_t npz, 
                 M* lb_param,
							   const std::size_t it);

    void 
    copy_fs_to(T* f,
               T* fs,
               std::size_t x, 
               std::size_t y,
               std::size_t z,
               std::size_t r, 
               const std::size_t npx,
               const std::size_t npy, 
               const std::size_t npz, 
               M* lb_param);

    void
    copy_fs_to(T* f,
               T* fs,
               std::size_t x, 
               std::size_t y,
               std::size_t z,
               std::size_t r, 
               const std::size_t npx,
               const std::size_t npy, 
               const std::size_t npz, 
               M* lb_param,
							 const std::size_t it);

    void 
    collide_cuda_device (const T beta, 
                         const std::size_t npx,
                         const std::size_t npy,
                         const std::size_t npz,
                         const std::size_t xmin,
                         const std::size_t xmax,
                         const std::size_t ymin,
                         const std::size_t ymax,
                         const std::size_t zmin,
                         const std::size_t zmax,
                         M* lb_param, 
                         T* data_device,
                         sycl::nd_item<3> item_ctl1
                         );

    void 
    cuda_test ();

    void
    collide_cuda(const std::size_t& it, block_type& b, const T beta, M* lb_model, T* d_data, sycl::queue& q);

    void
    collide(const std::size_t& it, grid_type& g, const T beta, M* lb_model, T* d_data, sycl::queue& q); 

};
