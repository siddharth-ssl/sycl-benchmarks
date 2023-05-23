#include <iostrem>

template<class M, typename T, std::size_t D>
class simulate
{
private:
    /* data */
    using lb_stencil = M;
    using block_type = matrix_block<lb_stencil, T, D>;
    using grid_type  = grid<block_type,T,D>;
public:
    simulate(/* args */) 
    { 

    }

    void 
    timestep();

    void 
    initialize_kida();

    void 
    ke();
    
    ~simulate()
    {
        
    }
};
