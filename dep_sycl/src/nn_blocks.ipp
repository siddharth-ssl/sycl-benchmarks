#include <nn_blocks.hpp>

template<std::size_t D>
nn_blocks<D>::nn_blocks(){}

template<std::size_t D>
nn_blocks<D>::nn_blocks(std::size_t* t_num_blocks)
{
    for(auto i=0; i<D; i++)
    {
        m_max_bidx[i] = t_num_blocks[i];
    }
}

template<std::size_t D>
nn_blocks<D>::~nn_blocks(){}

template<std::size_t D>
void 
nn_blocks<D>::set_bidx(std::size_t t_x, std::size_t t_y, std::size_t t_z)
{
    m_bidx = t_x + m_max_bidx[0] * (t_y + m_max_bidx[1] * t_z);
    return;
}

template<std::size_t D> 
std::size_t 
nn_blocks<D>::get_bidx(std::size_t t_x, std::size_t t_y, std::size_t t_z)
{
    auto t_bidx = t_x + m_max_bidx[0] * (t_y + m_max_bidx[1] * t_z);
    return t_bidx;
}

template<std::size_t D> 
std::vector<std::size_t> 
nn_blocks<D>::get_nn_blocks(const std::size_t& t_x, const std::size_t& t_y, const std::size_t& t_z)
{
    //std::size_t px, mx, py, my, pz, mz;

    auto px = t_x + static_cast<std::size_t>(1);
    auto mx = t_x - static_cast<std::size_t>(1);
    auto py = t_y + static_cast<std::size_t>(1);
    auto my = t_y - static_cast<std::size_t>(1);
    auto pz = t_z + static_cast<std::size_t>(1);
    auto mz = t_z - static_cast<std::size_t>(1); 

    //std::cout << px << " " << mx << " " << py  << " " << my << " " << pz << " " << mz << std::endl;

    if(t_x == 0) { mx = m_max_bidx[0] - static_cast<std::size_t>(1); }
    if(t_y == 0) { my = m_max_bidx[1] - static_cast<std::size_t>(1); }
    if(t_z == 0) { mz = m_max_bidx[2] - static_cast<std::size_t>(1); }

    if(t_x == m_max_bidx[0] - static_cast<std::size_t>(1)) { px = static_cast<std::size_t>(0); }
    if(t_y == m_max_bidx[1] - static_cast<std::size_t>(1)) { py = static_cast<std::size_t>(0); }
    if(t_z == m_max_bidx[2] - static_cast<std::size_t>(1)) { pz = static_cast<std::size_t>(0); }

    auto nn_px = get_bidx(px, t_y, t_z);
    auto nn_mx = get_bidx(mx, t_y, t_z);
    auto nn_py = get_bidx(t_x, py, t_z);
    auto nn_my = get_bidx(t_x, my, t_z);
    auto nn_pz = get_bidx(t_x, t_y, pz);
    auto nn_mz = get_bidx(t_x, t_y, mz);

    //std::cout << nn_px << " " << nn_mx << " " << nn_py << " " << nn_my << " " << nn_pz << " " << nn_mz << std::endl;

    m_nn_list[0]=(nn_px);
    m_nn_list[1]=(nn_mx);
    m_nn_list[2]=(nn_py);
    m_nn_list[3]=(nn_my);
    m_nn_list[4]=(nn_pz);
    m_nn_list[5]=(nn_mz);

    return m_nn_list;
}