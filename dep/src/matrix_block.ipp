#include <block.hpp>

template<typename T, std::size_t D>
block<T,D>::block() {}

template<typename T, std::size_t D>
block<T,D>::block(std::size_t* t_block_size, 
                  std::size_t* t_pad_size) : m_data(nullptr)
{
    for (auto i = 0; i<D; i++)
    {
        m_block_size[i]        = t_block_size[i];
        m_pad_size[i]          = t_pad_size[i];
        m_block_size_padded[i] = m_block_size[i] + static_cast<std::size_t>(2)*m_pad_size[i];

        m_min_index_padded[i]  = static_cast<std::size_t>(0);
        m_min_index[i]         = m_min_index_padded[i] + m_pad_size[i];
        m_max_index_padded[i]  = m_block_size_padded[i] - static_cast<std::size_t>(1);
        m_max_index[i]         = m_max_index_padded[i] - m_pad_size[i];
    }
}

template<typename T, std::size_t D>
std::size_t* 
block<T,D>::get_zone_min()
{
    return m_min_index;
}

template<typename T, std::size_t D>
std::size_t* 
block<T,D>::get_zone_max()
{
    return m_max_index;
}

template<typename T, std::size_t D>
std::vector<std::size_t>
block<T,D>::get_block_size() const 
{
    std::vector<std::size_t> t_size(D);
    for(auto idx = 0; idx<D; idx++)
    {
        t_size[idx] = m_block_size[idx];
    }
    return t_size;
}

template<typename T, std::size_t D>
std::vector<std::size_t>
block<T,D>::get_block_size_padded() const 
{
    std::vector<std::size_t> t_size(D);
    for(auto idx = 0; idx<D; idx++)
    {
        t_size[idx] = m_block_size_padded[idx];
    }
    return t_size;
}

template<typename T, std::size_t D>
std::size_t 
block<T,D>::tindx(std::size_t x, std::size_t y, std::size_t z)
{
    return x + m_block_size_padded[0]*(y + m_block_size_padded[1]*z);
}

template<typename T, std::size_t D>
void 
block<T,D>::set_bidx(const std::size_t t_bidx)
{
    m_bidx = t_bidx;
}

template<typename T, std::size_t D> 
void 
block<T,D>::set_nn_blocks(const std::vector<std::size_t>& t_nn_list)
{
    m_nn_list = t_nn_list;
    return;
}

template<typename T, std::size_t D>
std::size_t 
block<T,D>::get_bidx()
{
    return m_bidx;
}

template<typename T, std::size_t D> 
std::vector<std::size_t> 
block<T,D>::get_nn_blocks()
{
    return m_nn_list;
}

template<typename T, std::size_t D>
void 
block<T,D>::allocate()
{
    std::size_t alloc_bytes = m_block_size_padded[0] * m_block_size_padded[1] * m_block_size_padded[2] * sizeof(T);
    m_data = m_data + m_bidx * alloc_bytes;
    m_data = (T*)malloc(alloc_bytes);
    //m_data = new T[m_block_size_padded[0] * m_block_size_padded[1] * m_block_size_padded[2]];
}

template<typename T, std::size_t D>
void 
block<T,D>::fill_boundary_px(const block<T,D> &t_block)
{
    std::size_t t_x_receive = m_max_index_padded[0];
    std::size_t t_x_send    = m_min_index[0];
    //std::vector<T> t_slice(m_block_size_padded[1]*m_block_size_padded[2]);
    
    for(std::size_t z=m_min_index_padded[2]; z<=m_max_index_padded[2]; z++)
    {
        for(std::size_t y=m_min_index_padded[1]; y<=m_max_index_padded[1]; y++)
        {
            std::size_t idx_receive = t_x_receive + m_block_size_padded[0] * (y + m_block_size_padded[1] * z);
            m_data[idx_receive] = t_block(t_x_send,y,z);
        }
    }
}

template<typename T, std::size_t D>
void 
block<T,D>::fill_boundary_py(const block<T,D> &t_block)
{
    std::size_t t_y_receive = m_max_index_padded[1];
    std::size_t t_y_send    = m_min_index[1];
    //std::vector<T> t_slice(m_block_size_padded[0]*m_block_size_padded[2]);
    
    for(std::size_t z=m_min_index_padded[2]; z<=m_max_index_padded[2]; z++)
    {
        for(std::size_t x=m_min_index_padded[0]; x<=m_max_index_padded[0]; x++)
        {
            std::size_t idx_receive = x + m_block_size_padded[0] * (t_y_receive + m_block_size_padded[1] * z);
            m_data[idx_receive] = t_block(x,t_y_send,z);
        }
    }
}

template<typename T, std::size_t D>
void 
block<T,D>::fill_boundary_pz(const block<T,D> &t_block)
{
    std::size_t t_z_receive = m_max_index_padded[2];
    std::size_t t_z_send    = m_min_index[2];
    //std::vector<T> t_slice(m_block_size_padded[0]*m_block_size_padded[1]);
    
    for(std::size_t y=m_min_index_padded[1]; y<=m_max_index_padded[1]; y++)
    {
        for(std::size_t x=m_min_index_padded[0]; x<=m_max_index_padded[0]; x++)
        {
            std::size_t idx_receive = x + m_block_size_padded[0] * (y + m_block_size_padded[1] * t_z_receive);
            m_data[idx_receive] = t_block(x,y,t_z_send);
        }
    }
}

template<typename T, std::size_t D>
void 
block<T,D>::fill_boundary_mx(const block<T,D> &t_block)
{
    std::size_t t_x_receive = m_min_index_padded[0];
    std::size_t t_x_send    = m_max_index[0];
    //std::vector<T> t_slice(m_block_size_padded[1]*m_block_size_padded[2]);
    
    for(std::size_t z=m_min_index_padded[2]; z<=m_max_index_padded[2]; z++)
    {
        for(std::size_t y=m_min_index_padded[1]; y<=m_max_index_padded[1]; y++)
        {
            std::size_t idx_receive = t_x_receive + m_block_size_padded[0] * (y + m_block_size_padded[1] * z);
            m_data[idx_receive] = t_block(t_x_send,y,z);
        }
    }
}

template<typename T, std::size_t D>
void 
block<T,D>::fill_boundary_my(const block<T,D> &t_block)
{
    std::size_t t_y_receive = m_min_index_padded[1];
    std::size_t t_y_send    = m_max_index[1];
    //std::vector<T> t_slice(m_block_size_padded[0]*m_block_size_padded[2]);
    
    for(std::size_t z=m_min_index_padded[2]; z<=m_max_index_padded[2]; z++)
    {
        for(std::size_t x=m_min_index_padded[0]; x<=m_max_index_padded[0]; x++)
        {
            std::size_t idx_receive = x + m_block_size_padded[0] * (t_y_receive + m_block_size_padded[1] * z);
            m_data[idx_receive] = t_block(x,t_y_send,z);
        }
    }
}

template<typename T, std::size_t D>
void 
block<T,D>::fill_boundary_mz(const block<T,D> &t_block)
{
    std::size_t t_z_receive = m_min_index_padded[2];
    std::size_t t_z_send    = m_max_index[2];
    //std::vector<T> t_slice(m_block_size_padded[0]*m_block_size_padded[1]);
    
    for(std::size_t y=m_min_index_padded[1]; y<=m_max_index_padded[1]; y++)
    {
        for(std::size_t x=m_min_index_padded[0]; x<=m_max_index_padded[0]; x++)
        {
            std::size_t idx_receive = x + m_block_size_padded[0] * (y + m_block_size_padded[1] * t_z_receive);
            m_data[idx_receive] = t_block(x,y,t_z_send);
            //std::cout << t_block(x,y,t_z_send) << std::endl;
        }
    }
}

template<typename T, std::size_t D>
void 
block<T,D>::fill(const T &t_val)
{
    for(std::size_t z=m_min_index[2];z<=m_max_index[2]; z++)
    {
        for(std::size_t y=m_min_index[1]; y<=m_max_index[1]; y++)
        {
            for(std::size_t x=m_min_index[0]; x<=m_max_index[0]; x++)
            {
                m_data[tindx(x,y,z)] = t_val;
            }
        }
    }
}

template<typename T, std::size_t D>
void 
block<T,D>::fill_padded(const T &t_val)
{
    for(std::size_t z=m_min_index_padded[2];z<=m_max_index_padded[2]; z++)
    {
        for(std::size_t y=m_min_index_padded[1]; y<=m_max_index_padded[1]; y++)
        {
            for(std::size_t x=m_min_index_padded[0]; x<=m_max_index_padded[0]; x++)
            {
                m_data[tindx(x,y,z)] = t_val;
            }
        }
    }
}

template<typename T, std::size_t D>
T* 
block<T,D>::data()
{
    return m_data;
}

template<typename T, std::size_t D>
const T&
block<T,D>::operator[] (std::size_t idx) const 
{
    return m_data[idx];
}

template<typename T, std::size_t D>
T&
block<T,D>::operator[] (std::size_t idx)
{
    return m_data[idx];
}

template<typename T, std::size_t D>
const T&
block<T,D>::operator() (std::size_t i, std::size_t j, std::size_t k) const 
{
    const auto nx  = m_block_size_padded[0];
    const auto ny  = m_block_size_padded[1];
    const auto nz  = m_block_size_padded[2];
    
    const auto idx = i + nx * (j + ny * k);
    return m_data[idx];
}

template<typename T, std::size_t D>
T&
block<T,D>::operator() (std::size_t i, std::size_t j, std::size_t k)
{
    const auto nx  = m_block_size_padded[0];
    const auto ny  = m_block_size_padded[1];
    const auto nz  = m_block_size_padded[2];
    
    const auto idx = i + nx * (j + ny * k);
    return m_data[idx];
}

template<typename T, std::size_t D>
std::pair<std::size_t, std::size_t> 
block<T,D>::size_of()
{
    std::size_t npx = get_block_size_padded()[0];
    std::size_t npy = get_block_size_padded()[1];
    std::size_t npz = get_block_size_padded()[2];
    
    std::size_t size        = npx*npy*npz;
    std::size_t alloc_bytes = npx*npy*npz*sizeof(T);

    return std::make_pair(size, alloc_bytes);
}

template<typename T, std::size_t D>
void 
block<T,D>::dealloc()
{
    //m_data = nullptr;
    free(m_data);
    //delete m_data;
}
template<typename T, std::size_t D>
block<T,D>::~block()
{
    //free(m_data);
}


/*
int main(int argc, char** argv)
{
    block<double1,3> cuda_block(4,4,4,1,1,1);
}
*/

