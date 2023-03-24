#pragma once
#include <iostream>
#include <vector>

template<std::size_t D>
class nn_blocks
{
protected:
    std::vector<std::size_t> m_max_bidx = std::vector<std::size_t>(D);
    // std::size_t m_max_y;
    // std::size_t m_max_z;
    std::size_t m_bidx;
    std::vector<std::size_t> m_nn_list  = std::vector<std::size_t>(6);

public:
    nn_blocks();

    nn_blocks(std::size_t* t_num_blocks);

    void  
    set_bidx(std::size_t t_x, std::size_t t_y, std::size_t t_z);
    
    std::size_t 
    get_bidx(std::size_t t_x, std::size_t t_y, std::size_t t_z);

    std::vector<std::size_t> 
    get_nn_blocks(const std::size_t& t_x, const std::size_t& t_y, const std::size_t& t_z);

    ~nn_blocks();
};