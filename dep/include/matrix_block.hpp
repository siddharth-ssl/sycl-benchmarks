#pragma once
#include <iostream>
#include "D3Q27SC.hpp"
#include <vector>

template<class M, typename T, std::size_t D>
class matrix_block
{
protected:
    /* cuda block data */
    std::size_t m_block_size[D];
    std::size_t m_pad_size[D];
    std::size_t m_block_size_padded[D];
    std::size_t m_min_index[D];
    std::size_t m_min_index_padded[D];
    std::size_t m_max_index[D];
    std::size_t m_max_index_padded[D];
    std::size_t m_num_mems;
    std::size_t m_num_grps;
    std::size_t m_num_vars;
    
    // std::size_t m_size;
    // std::size_t m_size_padded; 
    std::size_t m_bidx;
    std::vector<std::size_t> m_btks    = std::vector<std::size_t>(D);
    std::vector<std::size_t> m_nn_list = std::vector<std::size_t>(6);
    T* m_data;
    M  m_lb_stencil;

public:
    matrix_block ();
    
    matrix_block (std::size_t* t_block_size, 
           std::size_t* t_pad_size);
    
    std::vector<std::size_t> 
    get_block_size() const;

    std::vector<std::size_t> 
    get_block_size_padded() const;

    std::size_t* 
    get_zone_min();

    std::size_t* 
    get_zone_max();

    const std::size_t 
    tindx  (std::size_t m, std::size_t g, std::size_t x, std::size_t y, std::size_t z) const;

    const std::size_t 
    tindx  (std::size_t dv, std::size_t x, std::size_t y, std::size_t z) const;

    // __device__ inline std::size_t 
    // bidx (std::size_t bx, std::size_t by, std::size_t bz);

    // void allocate();

    void 
    allocate();

    void 
    fill_boundary_px(const matrix_block<M,T,D> &t_block);

    void 
    fill_boundary_py(const matrix_block<M,T,D> &t_block);

    void 
    fill_boundary_pz(const matrix_block<M,T,D> &t_block);

    void 
    fill_boundary_mx(const matrix_block<M,T,D> &t_block);

    void 
    fill_boundary_my(const matrix_block<M,T,D> &t_block);

    void 
    fill_boundary_mz(const matrix_block<M,T,D> &t_block);

    void 
    fill(const T &t_val);

    void 
    fill_padded(const T &T_val);

    T* 
    data();

    const T& 
    operator[] (std::size_t idx) const;
    
    T& 
    operator[] (std::size_t idx);

    const T& 
    operator() (std::size_t m, std::size_t g, std::size_t i, std::size_t j, std::size_t k) const;
    
    T& 
    operator() (std::size_t m, std::size_t g, std::size_t i, std::size_t j, std::size_t k);

    const T& 
    operator() (std::size_t dv, std::size_t i, std::size_t j, std::size_t k) const;
    
    T& 
    operator() (std::size_t dv, std::size_t i, std::size_t j, std::size_t k);

    std::pair<std::size_t, std::size_t> 
    size_of();

    void 
    dealloc();

    void 
    set_bidx(const std::size_t t_bidx);

    void 
    set_btks(const std::size_t t_bx,
	     const std::size_t t_by,
	     const std::size_t t_bz);

    std::vector<std::size_t>
    get_btks();

    void 
    set_nn_blocks(const std::vector<std::size_t>& t_nn_list);

    std::size_t 
    get_bidx();

    std::vector<std::size_t> 
    get_nn_blocks();

    M 
    lb_model() { return m_lb_stencil ;}
    
    ~matrix_block ();
};
