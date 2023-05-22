#pragma once
#include <block.hpp>
#include <matrix_block.hpp>
#include <nn_blocks.hpp>
#include <iostream>
#include <array>
#include <vector>
#include <map>
#include <utility>

template<class block, typename T, std::size_t D>
class grid : public nn_blocks<D>
{
private:
    std::size_t m_npx;
    std::size_t m_npy;
    std::size_t m_npz;
    std::size_t m_nbx;
    std::size_t m_nby;
    std::size_t m_nbz;
    
    block m_block;
    std::vector<std::pair<std::size_t,block>> m_block_list;
    //std::map<std::size_t,block> m_block_list;

    T* m_data_host;
    // T* m_data_device;

public:
    grid ();
    
    grid (std::size_t* t_block_size, std::size_t* t_grid_size, std::size_t* t_pad_size);
    
    std::size_t 
    bindx (std::size_t bx, std::size_t by, std::size_t bz);
    
    void 
    allocate();
    
    std::vector<std::size_t> 
    get_block_size() const;

    std::vector<std::size_t> 
    get_block_size_padded() const;
    
    std::vector<std::size_t>  
    get_grid_size() const;
    
    T* 
    data_host();

    void 
    fill(const T &t_val);

    void 
    fill_padded(const T &t_val);
    
    block  
    at(std::size_t x, std::size_t y, std::size_t z);

    block 
    at(std::size_t t_idx);

    std::pair<std::size_t, std::size_t>  
    size_of();

    std::vector<std::pair<std::size_t,block>> 
    get_blocks();
    
    void 
    communic_nn_blocks();

    ~grid ();
    
    
    /*
    void get_dim()
    {
        std::cout << block<T,D>::m_block_size[0] << " " << m_nby << " " << m_nbz << std::endl;
    }

    void 
    assign_device(std::size_t idx, T t_val)
    {
        m_data_device[idx] = t_val;
    }
    */

};
