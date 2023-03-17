#pragma once
#include <iostream>
#include <vector>

template<typename T, std::size_t D>
class block
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
    
    // std::size_t m_size;
    // std::size_t m_size_padded; 
    std::size_t m_bidx;
    std::vector<std::size_t> m_nn_list = std::vector<std::size_t>(6);
    T* m_data;
    T* m_device_data;

public:
    block ();
    
    block (std::size_t* t_block_size, 
           std::size_t* t_pad_size);
    
    std::vector<std::size_t> 
    get_block_size() const;

    std::vector<std::size_t> 
    get_block_size_padded() const;

    std::size_t* 
    get_zone_min();

    std::size_t* 
    get_zone_max();

    std::size_t 
    tindx  (std::size_t x, std::size_t y, std::size_t z);

    // __device__ inline std::size_t 
    // bidx (std::size_t bx, std::size_t by, std::size_t bz);

    // void allocate();

    void 
    allocate();

    void 
    copy_from_device_to_host();

    void 
    copy_from_host_to_device();

    void 
    fill_boundary_px(const block<T,D> &t_block);

    void 
    fill_boundary_py(const block<T,D> &t_block);

    void 
    fill_boundary_pz(const block<T,D> &t_block);

    void 
    fill_boundary_mx(const block<T,D> &t_block);

    void 
    fill_boundary_my(const block<T,D> &t_block);

    void 
    fill_boundary_mz(const block<T,D> &t_block);

    void 
    fill(const T &t_val);

    void 
    fill_padded(const T &T_val);

    T* 
    data();

    T* 
    data_device();

    const T& 
    operator[] (std::size_t idx) const;
    
    T& 
    operator[] (std::size_t idx);

    const T& 
    operator() (std::size_t i, std::size_t j, std::size_t k) const;
    
    T& 
    operator() (std::size_t i, std::size_t j, std::size_t k);

    std::pair<std::size_t, std::size_t> 
    size_of();

    void 
    dealloc();

    void 
    set_bidx(const std::size_t t_bidx);

    void 
    set_nn_blocks(const std::vector<std::size_t>& t_nn_list);

    std::size_t 
    get_bidx();

    std::vector<std::size_t> 
    get_nn_blocks();
    
    ~block ();
};
