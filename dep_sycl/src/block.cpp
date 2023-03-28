#include "block.ipp"

//template
//class block<double,2>;

template 
class block<double,3>;

/*
template
block<double,3>::block(std::size_t* t_block_size, 
                  std::size_t* t_pad_size);

template
block<double,3>::~block();

template 
std::vector<std::size_t> 
block<double,3>::get_block_size() const;

template 
std::vector<std::size_t> 
block<double,3>::get_block_size_padded() const;

template 
std::size_t*
block<double,3>::get_zone_min(); 

template 
std::size_t* 
block<double,3>::get_zone_max(); 

template
std::size_t 
block<double,3>::tindx(std::size_t x, std::size_t y, std::size_t z);

template
void 
block<double,3>::allocate();

template 
void 
block<double,3>::copy_from_device_to_host();

template 
void 
block<double,3>::copy_from_host_to_device();

template
void 
block<double,3>::fill_boundary_px(const block<double,3> &t_block);

template
void 
block<double,3>::fill_boundary_py(const block<double,3> &t_block);

template
void 
block<double,3>::fill_boundary_pz(const block<double,3> &t_block);

template
void 
block<double,3>::fill_boundary_mx(const block<double,3> &t_block);

template
void 
block<double,3>::fill_boundary_my(const block<double,3> &t_block);

template
void 
block<double,3>::fill_boundary_mz(const block<double,3> &t_block);

template
void 
block<double,3>::fill(const double &t_val);

template 
void 
block<double,3>::fill_padded(const double &t_val);

template 
double *
block<double,3>::data();

template
double *
block<double,3>::data_device();

template 
const double& 
block<double,3>::operator[] (std::size_t idx) const;

template 
double& 
block<double,3>::operator[] (std::size_t idx);

template 
const double& 
block<double,3>::operator() (std::size_t i, std::size_t j, std::size_t k) const;

template 
double& 
block<double,3>::operator() (std::size_t i, std::size_t j, std::size_t k);

template
std::pair<std::size_t, std::size_t> 
block<double,3>::size_of();

template 
void 
block<double,3>::dealloc();

template 
void 
block<double,3>::set_bidx(const std::size_t t_bidx);

template 
std::size_t 
block<double,3>::get_bidx();

template 
void 
block<double,3>::set_nn_blocks(const std::vector<std::size_t>& t_nn_list);

template 
std::vector<std::size_t> 
block<double,3>::get_nn_blocks();
*/
/*
int main(int argc, char** argv)
{
    block<double1,3> cuda_block(4,4,4,1,1,1);
}
*/

