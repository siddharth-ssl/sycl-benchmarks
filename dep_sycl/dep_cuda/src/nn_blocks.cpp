#include "nn_blocks.ipp"

template
nn_blocks<3>::nn_blocks();

template
nn_blocks<3>::nn_blocks(std::size_t* t_num_blocks);

template
nn_blocks<3>::~nn_blocks();

template
void 
nn_blocks<3>::set_bidx(std::size_t t_x, std::size_t t_y, std::size_t t_z);

template 
std::size_t 
nn_blocks<3>::get_bidx(std::size_t t_x, std::size_t t_y, std::size_t t_z);

template 
std::vector<std::size_t> 
nn_blocks<3>::get_nn_blocks(const std::size_t& t_x, const std::size_t& t_y, const std::size_t& t_z);