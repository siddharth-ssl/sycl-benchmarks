#include "grid.ipp"


template
grid<block<double,3>,double,3>::grid();

template
grid<block<double,3>,double,3>::grid(std::size_t* t_block_size, std::size_t* t_grid_size, std::size_t* t_pad_size);

template
grid<block<double,3>,double,3>::~grid();

template
std::size_t 
grid<block<double,3>,double,3>::bindx (std::size_t bx, std::size_t by, std::size_t bz);


template
void 
grid<block<double,3>,double,3>::allocate();

template 
void 
grid<block<double,3>,double,3>::copy_from_host_to_device();

template 
void 
grid<block<double,3>,double,3>::copy_from_device_to_host();

template
std::vector<std::size_t> 
grid<block<double,3>,double,3>::get_block_size() const;

template
std::vector<std::size_t> 
grid<block<double,3>,double,3>::get_block_size_padded() const;

template
std::vector<std::size_t>
grid<block<double,3>,double,3>::get_grid_size() const;

template
double* 
grid<block<double,3>,double,3>::data_host();


template
void 
grid<block<double,3>,double,3>::fill(const double &t_val);

template 
void 
grid<block<double,3>,double,3>::fill_padded(const double &t_val);

template
block<double,3> 
grid<block<double,3>,double,3>::at(std::size_t x, std::size_t y, std::size_t z);

template 
block<double,3> 
grid<block<double,3>,double,3>::at(std::size_t t_idx);

template
std::pair<std::size_t, std::size_t> 
grid<block<double,3>,double,3>::size_of();

template 
std::vector<std::pair<std::size_t, block<double,3>>> 
grid<block<double,3>,double,3>::get_blocks();

template 
void 
grid<block<double,3>,double,3>::communic_nn_blocks();