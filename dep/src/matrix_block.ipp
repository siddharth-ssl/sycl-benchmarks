#include <matrix_block.hpp>

template<class M, typename T, std::size_t D>
matrix_block<M,T,D>::matrix_block() {}

template<class M, typename T, std::size_t D>
matrix_block<M,T,D>::matrix_block(std::size_t* t_block_size, 
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

    m_num_mems             = M::num_mems;
    m_num_grps             = M::num_grps;
    m_num_vars             = M::num_vars;
}

template<class M, typename T, std::size_t D>
std::size_t* 
matrix_block<M,T,D>::get_zone_min()
{
    return m_min_index;
}

template<class M, typename T, std::size_t D>
std::size_t* 
matrix_block<M,T,D>::get_zone_max()
{
    return m_max_index;
}

template<class M, typename T, std::size_t D>
std::vector<std::size_t>
matrix_block<M,T,D>::get_block_size() const 
{
    std::vector<std::size_t> t_size(D);
    for(auto idx = 0; idx<D; idx++)
    {
        t_size[idx] = m_block_size[idx];
    }
    return t_size;
}

template<class M, typename T, std::size_t D>
std::vector<std::size_t>
matrix_block<M,T,D>::get_block_size_padded() const 
{
    std::vector<std::size_t> t_size(D);
    for(auto idx = 0; idx<D; idx++)
    {
        t_size[idx] = m_block_size_padded[idx];
    }
    return t_size;
}

template<class M, typename T, std::size_t D>
const std::size_t 
matrix_block<M,T,D>::tindx(std::size_t m, std::size_t g, std::size_t x, std::size_t y, std::size_t z) const 
{
    return m + m_num_mems * (x + m_block_size_padded[0]*(y + m_block_size_padded[1]*(z + m_block_size_padded[2]*g)));
}

template<class M, typename T, std::size_t D>
const std::size_t 
matrix_block<M,T,D>::tindx(std::size_t dv, std::size_t x, std::size_t y, std::size_t z) const 
{
    //return dv + (m_num_vars) * (x + m_block_size_padded[0]*(y + m_block_size_padded[1]*z));
    return x + m_block_size_padded[0]*(y + m_block_size_padded[1]*(z + m_block_size_padded[2]*dv));
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::set_bidx(const std::size_t t_bidx)
{
    m_bidx = t_bidx;
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::set_btks(const std::size_t t_bx,
		              const std::size_t t_by,
			      const std::size_t t_bz)
{
    m_btks[0] = t_bx;
    m_btks[1] = t_by;
    m_btks[2] = t_bz;
}

template<class M, typename T, std::size_t D>
std::vector<std::size_t> 
matrix_block<M,T,D>::get_btks()
{
    return m_btks;
}	

template<class M, typename T, std::size_t D> 
void 
matrix_block<M,T,D>::set_nn_blocks(const std::vector<std::size_t>& t_nn_list)
{
    m_nn_list = t_nn_list;
    return;
}

template<class M, typename T, std::size_t D>
std::size_t 
matrix_block<M,T,D>::get_bidx()
{
    return m_bidx;
}

template<class M, typename T, std::size_t D> 
std::vector<std::size_t> 
matrix_block<M,T,D>::get_nn_blocks()
{
    return m_nn_list;
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::allocate()
{
    std::size_t alloc_bytes = m_num_vars * m_block_size_padded[0] * m_block_size_padded[1] * m_block_size_padded[2] * sizeof(T);
    //std::cout << m_num_vars << std::endl; 
    m_data = m_data + m_bidx * alloc_bytes;
    m_data = (T*)malloc(alloc_bytes);
    //m_data = new T[m_block_size_padded[0] * m_block_size_padded[1] * m_block_size_padded[2]];
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::fill_boundary_px(const matrix_block<M,T,D> &t_block)
{
    std::size_t t_x_receive = m_max_index_padded[0];
    std::size_t t_x_send    = m_min_index[0];
    //std::vector<T> t_slice(m_block_size_padded[1]*m_block_size_padded[2]);
    
    for(std::size_t z=m_min_index_padded[2]; z<=m_max_index_padded[2]; z++)
    {
        for(std::size_t y=m_min_index_padded[1]; y<=m_max_index_padded[1]; y++)
        {
            for(std::size_t g=0; g<m_num_grps; g++)
            {
                for(std::size_t m=0; m<m_num_mems; m++)
                {
                    std::size_t idx_receive = tindx(m,g,t_x_receive,y,z) ;
                    m_data[idx_receive] = t_block(m,g,t_x_send,y,z);
                }
            }
        }
    }
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::fill_boundary_py(const matrix_block<M,T,D> &t_block)
{
    std::size_t t_y_receive = m_max_index_padded[1];
    std::size_t t_y_send    = m_min_index[1];
    //std::vector<T> t_slice(m_block_size_padded[0]*m_block_size_padded[2]);
    
    for(std::size_t z=m_min_index_padded[2]; z<=m_max_index_padded[2]; z++)
    {
        for(std::size_t x=m_min_index_padded[0]; x<=m_max_index_padded[0]; x++)
        {
            for(std::size_t g=0; g<m_num_grps; g++)
            {
                for(std::size_t m=0; m<m_num_mems; m++)
                {
                    std::size_t idx_receive = tindx(m,g,x,t_y_receive,z) ;
                    m_data[idx_receive] = t_block(m,g,x,t_y_send,z);
                }
            }
        }
    }
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::fill_boundary_pz(const matrix_block<M,T,D> &t_block)
{
    std::size_t t_z_receive = m_max_index_padded[2];
    std::size_t t_z_send    = m_min_index[2];
    //std::vector<T> t_slice(m_block_size_padded[0]*m_block_size_padded[1]);
    
    for(std::size_t y=m_min_index_padded[1]; y<=m_max_index_padded[1]; y++)
    {
        for(std::size_t x=m_min_index_padded[0]; x<=m_max_index_padded[0]; x++)
        {
            for(std::size_t g=0; g<m_num_grps; g++)
            {
                for(std::size_t m=0; m<m_num_mems; m++)
                {
                    std::size_t idx_receive = tindx(m,g,x,y,t_z_receive) ;
                    m_data[idx_receive] = t_block(m,g,x,y,t_z_send);
                }
            }
        }
    }
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::fill_boundary_mx(const matrix_block<M,T,D> &t_block)
{
    std::size_t t_x_receive = m_min_index_padded[0];
    std::size_t t_x_send    = m_max_index[0];
    //std::vector<T> t_slice(m_block_size_padded[1]*m_block_size_padded[2]);
    
    for(std::size_t z=m_min_index_padded[2]; z<=m_max_index_padded[2]; z++)
    {
        for(std::size_t y=m_min_index_padded[1]; y<=m_max_index_padded[1]; y++)
        {
            for(std::size_t g=0; g<m_num_grps; g++)
            {
                for(std::size_t m=0; m<m_num_mems; m++)
                {
                    std::size_t idx_receive = tindx(m,g,t_x_receive,y,z) ;
                    m_data[idx_receive] = t_block(m,g,t_x_send,y,z);
                }
            }
        }
    }
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::fill_boundary_my(const matrix_block<M,T,D> &t_block)
{
    std::size_t t_y_receive = m_min_index_padded[1];
    std::size_t t_y_send    = m_max_index[1];
    //std::vector<T> t_slice(m_block_size_padded[0]*m_block_size_padded[2]);
    
    for(std::size_t z=m_min_index_padded[2]; z<=m_max_index_padded[2]; z++)
    {
        for(std::size_t x=m_min_index_padded[0]; x<=m_max_index_padded[0]; x++)
        {
            for(std::size_t g=0; g<m_num_grps; g++)
            {
                for(std::size_t m=0; m<m_num_mems; m++)
                {
                    std::size_t idx_receive = tindx(m,g,x,t_y_receive,z) ;
                    m_data[idx_receive] = t_block(m,g,x,t_y_send,z);
                }
            }
        }
    }
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::fill_boundary_mz(const matrix_block<M,T,D> &t_block)
{
    std::size_t t_z_receive = m_min_index_padded[2];
    std::size_t t_z_send    = m_max_index[2];
    //std::vector<T> t_slice(m_block_size_padded[0]*m_block_size_padded[1]);
    
    for(std::size_t y=m_min_index_padded[1]; y<=m_max_index_padded[1]; y++)
    {
        for(std::size_t x=m_min_index_padded[0]; x<=m_max_index_padded[0]; x++)
        {
            for(std::size_t g=0; g<m_num_grps; g++)
            {
                for(std::size_t m=0; m<m_num_mems; m++)
                {
                    std::size_t idx_receive = tindx(m,g,x,y,t_z_receive) ;
                    m_data[idx_receive] = t_block(m,g,x,y,t_z_send);
                    //std::cout << t_block(x,y,t_z_send) << std::endl;
                }
            }
        }
    }
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::fill(const T &t_val)
{
    for(std::size_t z=m_min_index[2];z<=m_max_index[2]; z++)
    {
        for(std::size_t y=m_min_index[1]; y<=m_max_index[1]; y++)
        {
            for(std::size_t x=m_min_index[0]; x<=m_max_index[0]; x++)
            {
                for(std::size_t g=0; g<m_num_grps; g++)
                {
                    for(std::size_t m=0; m<m_num_mems; m++)
                    {
                        //std::cout << m_data[tindx(m,g,x,y,z)] << std::endl;
                        m_data[tindx(m,g,x,y,z)] = t_val;
                    }
                }
            }
        }
    }
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::fill_padded(const T &t_val)
{
    for(std::size_t z=m_min_index_padded[2];z<=m_max_index_padded[2]; z++)
    {
        for(std::size_t y=m_min_index_padded[1]; y<=m_max_index_padded[1]; y++)
        {
            for(std::size_t x=m_min_index_padded[0]; x<=m_max_index_padded[0]; x++)
            {
                for(std::size_t g=0; g<m_num_grps; g++)
                {
                    for(std::size_t m=0; m<m_num_mems; m++)
                    {
                        m_data[tindx(m,g,x,y,z)] = t_val;
                    }
                }
            }
        }
    }
}

template<class M, typename T, std::size_t D>
T* 
matrix_block<M,T,D>::data()
{
    return m_data;
}

template<class M, typename T, std::size_t D>
const T&
matrix_block<M,T,D>::operator[] (std::size_t idx) const 
{
    return m_data[idx];
}

template<class M, typename T, std::size_t D>
T&
matrix_block<M,T,D>::operator[] (std::size_t idx)
{
    return m_data[idx];
}

template<class M, typename T, std::size_t D>
const T&
matrix_block<M,T,D>::operator() (std::size_t m, std::size_t g, std::size_t i, std::size_t j, std::size_t k) const 
{
    return m_data[tindx(m,g,i,j,k)];
}

template<class M, typename T, std::size_t D>
T&
matrix_block<M,T,D>::operator() (std::size_t m, std::size_t g, std::size_t i, std::size_t j, std::size_t k)
{
    return m_data[tindx(m,g,i,j,k)];
}

template<class M, typename T, std::size_t D>
const T&
matrix_block<M,T,D>::operator() (std::size_t dv, std::size_t i, std::size_t j, std::size_t k) const 
{
    return m_data[tindx(dv,i,j,k)];
}

template<class M, typename T, std::size_t D>
T&
matrix_block<M,T,D>::operator() (std::size_t dv, std::size_t i, std::size_t j, std::size_t k)
{
    return m_data[tindx(dv,i,j,k)];
}

template<class M, typename T, std::size_t D>
std::pair<std::size_t, std::size_t> 
matrix_block<M,T,D>::size_of()
{
    std::size_t npx = get_block_size_padded()[0];
    std::size_t npy = get_block_size_padded()[1];
    std::size_t npz = get_block_size_padded()[2];
    
    std::size_t size        = m_num_vars*npx*npy*npz;
    std::size_t alloc_bytes = m_num_vars*npx*npy*npz*sizeof(T);

    return std::make_pair(size, alloc_bytes);
}

template<class M, typename T, std::size_t D>
void 
matrix_block<M,T,D>::dealloc()
{
    //m_data = nullptr;
    //free(m_data);
    //delete m_data;
}
template<class M, typename T, std::size_t D>
matrix_block<M,T,D>::~matrix_block()
{
    //free(m_data);
}


/*
int main(int argc, char** argv)
{
    block<double1,3> cuda_block(4,4,4,1,1,1);
}
*/

