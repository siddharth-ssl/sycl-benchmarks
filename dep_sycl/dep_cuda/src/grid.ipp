#include <grid.hpp>

template<class block, typename T, std::size_t D>
grid<block,T,D>::grid(){}

template<class block, typename T, std::size_t D>
grid<block,T,D>::grid(std::size_t* t_block_size, std::size_t* t_grid_size, std::size_t* t_pad_size) : nn_blocks<D>::nn_blocks(t_grid_size) 
{
    m_npx = t_block_size[0];
    m_npy = t_block_size[1];
    m_npz = t_block_size[2];
    
    m_nbx = t_grid_size[0];
    m_nby = t_grid_size[1];
    m_nbz = t_grid_size[2];

    m_block = block(t_block_size, t_pad_size);
    
}

template<class block, typename T, std::size_t D>
std::size_t 
grid<block,T,D>::bindx (std::size_t bx, std::size_t by, std::size_t bz)
{
    return bx + m_nbx*(by + m_nby*bz);
}

template<class block, typename T, std::size_t D>
void 
grid<block,T,D>::allocate()
{
    for(auto bz=0; bz<m_nbz; bz++)
    {
        for(auto by=0; by<m_nby; by++)
        {
            for(auto bx=0; bx<m_nbx; bx++)
            {
                const auto& t_bindx = bindx(bx, by, bz);
                //block t_b = block(t_block_size, t_pad_size);
                m_block.set_bidx(t_bindx);

                const auto& t_bx = bx;
                const auto& t_by = by;
                const auto& t_bz = bz;

                const auto t_nn_list = nn_blocks<D>::get_nn_blocks(t_bx,t_by,t_bz);
                //std::cout << t_bindx << " " <<  t_nn_list[1] << std::endl;
                m_block.set_nn_blocks(t_nn_list);
                m_block.allocate();
                //m_block_list[t_bindx] = std::make_pair(t_bindx, m_block);
                m_block_list.push_back(std::make_pair(t_bindx, m_block  ));
                //m_block_list.insert(std::pair<std::size_t, block>(t_bindx, m_block));
            }
        }
    }
    
}

template<class block, typename T, std::size_t D> 
std::vector<std::size_t> 
grid<block,T,D>::get_block_size() const 
{
    return m_block_list[0].second.get_block_size();
}

template<class block, typename T, std::size_t D> 
std::vector<std::size_t> 
grid<block,T,D>::get_block_size_padded() const 
{
    return m_block_list[0].second.get_block_size_padded();
}

template<class block, typename T, std::size_t D>
std::vector<std::size_t>  
grid<block,T,D>::get_grid_size() const 
{
    std::vector<std::size_t> t_grid_size(D);

    t_grid_size[0] = m_nbx;
    t_grid_size[1] = m_nby;
    t_grid_size[2] = m_nbz;

    return t_grid_size;
}

template<class block, typename T, std::size_t D>
void 
grid<block,T,D>::fill(const T &t_val)
{
    for(auto bpair : m_block_list)
    {
        bpair.second.fill(t_val);
    }
}

template<class block, typename T, std::size_t D>
void 
grid<block,T,D>::fill_padded(const T &t_val)
{
    for(auto bpair : m_block_list)
    {
        bpair.second.fill_padded(t_val);
    }
}

template<class block, typename T, std::size_t D>
void 
grid<block,T,D>::copy_from_host_to_device()
{
    for(auto bpair : m_block_list)
    {
        bpair.second.copy_from_host_to_device();
    }
}

template<class block, typename T, std::size_t D>
void 
grid<block,T,D>::copy_from_device_to_host()
{
    for(auto bpair : m_block_list)
    {
        bpair.second.copy_from_device_to_host();
    }
}

template<class block, typename T, std::size_t D>
block  
grid<block,T,D>::at(std::size_t x, std::size_t y, std::size_t z)
{
    return m_block_list.at(bindx(x,y,z)).second;
}

template<class block, typename T, std::size_t D>
block  
grid<block,T,D>::at(std::size_t t_idx)
{
    return m_block_list[t_idx].second;
}


template<class block, typename T, std::size_t D>
T* 
grid<block,T,D>::data_host()
{
    return m_data_host;
}

template<class block, typename T, std::size_t D>
std::pair<std::size_t, std::size_t> 
grid<block,T,D>::size_of()
{
    std::size_t npx = get_block_size_padded()[0];
    std::size_t npy = get_block_size_padded()[1];
    std::size_t npz = get_block_size_padded()[2];
    
    std::size_t size        = npx*npy*npz*m_nbx*m_nby*m_nbz;
    std::size_t alloc_bytes = npx*npy*npz*m_nbx*m_nby*m_nbz*sizeof(T);

    return std::make_pair(size, alloc_bytes);
}


template<class block, typename T, std::size_t D>
std::vector<std::pair<std::size_t, block>> 
grid<block,T,D>::get_blocks()
{
    return m_block_list;
}

template<class block, typename T, std::size_t D>
void 
grid<block,T,D>::communic_nn_blocks()
{
    for(auto bpair : m_block_list)
    {
        auto t_b = bpair.second;
        auto t_nn_list = t_b.get_nn_blocks();

        t_b.fill_boundary_px(at(t_nn_list[0]));
        t_b.fill_boundary_mx(at(t_nn_list[1]));
        t_b.fill_boundary_py(at(t_nn_list[2]));
        t_b.fill_boundary_my(at(t_nn_list[3]));
        t_b.fill_boundary_pz(at(t_nn_list[4]));
        t_b.fill_boundary_mz(at(t_nn_list[5]));
        /*
        for(auto nn_bpair : m_block_list)
        {
            const auto t_nnb = nn_bpair.second;
            const auto& t_nn_bidx = nn_bpair.second.get_bidx();
            //std::cout << t_nn_list[0] << " " << t_nn_bidx << std::endl;
            // px-wall
            if(t_nn_list[0] == t_nn_bidx) { t_b.fill_boundary_px(t_nnb); }//std::cout << t_nn_list[0] << " " << t_nn_bidx << " " << t_b(120,30,0) << " " << t_nnb(120,30,128) << std::endl;}
            // mx-wall
            if(t_nn_list[1] == t_nn_bidx) { t_b.fill_boundary_mx(t_nnb); }
            // py-wall
            if(t_nn_list[2] == t_nn_bidx) { t_b.fill_boundary_py(t_nnb); }
            // my-wall
            if(t_nn_list[3] == t_nn_bidx) { t_b.fill_boundary_my(t_nnb); }
            // pz-wall
            if(t_nn_list[4] == t_nn_bidx) { t_b.fill_boundary_pz(t_nnb); }
            // mz-wall
            if(t_nn_list[5] == t_nn_bidx) { t_b.fill_boundary_mz(t_nnb); }
        }

        //std::cout << "Neighbours are communicated for block " << t_b.get_bidx() << std::endl;
        */
    }
}

template<class block, typename T, std::size_t D>
grid<block,T,D>::~grid()
{
    //delete m_data_host;
    //m_block_list.clear();
}


