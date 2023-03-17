#include <cuda_kernels.hpp>

/// @brief Global CUDA kernels
/// @tparam T 
/// @tparam D 
/// @return 

template<typename T>
__global__ void 
fill_device(const T &t_val, std::size_t npx, std::size_t npy, std::size_t npz, T* device_data)
{
    std::size_t b = bindx(blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
    std::size_t bsize = blockDim.x * blockDim.y * blockDim.z;
    T* t_grid = &device_data[b * bsize];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    if(z>=0 && z<npz)
    {
        if(y>=0 && y<npy)
        {
            if(x>=0 && x<npx)
            {
                t_grid[tindx(x,y,z,npx,npy,npz)] = t_val;
                //cgrid(tindx(x,y,z,npx,npy,npz)) = t_val;
                //cgrid.assign_device(tindx(x,y,z,npx,npy,npz), t_val);
            }
        }
    }
    

}

template<typename T, std::size_t D>
void 
fill(const T &t_val, block<T,D>& cgrid, T* device_data)
{
    const auto npx = cgrid.get_block_size_padded()[0];
    const auto npy = cgrid.get_block_size_padded()[1];
    const auto npz = cgrid.get_block_size_padded()[2];
    // const auto nbx = cgrid.get_grid_size()[0];
    // const auto nby = cgrid.get_grid_size()[1];
    // const auto nbz = cgrid.get_grid_size()[2];
    
    dim3 num_blocks(npx, npy, npz);
    dim3 num_threads(npx, npy, npz);

    //T* data = cgrid.data_device();

    fill_device<<<num_blocks, num_threads>>>(t_val, npx, npy, npz, device_data);
    //fill_device<<<4, 16>>>(t_val, npx, npy, npz, device_data);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        //fprintf(stderr, "initialise kernel error: %s\n", cudaGetErrorString(err));
        std::cout << "initialize kernel error: " << " " << cudaGetErrorString(err) << std::endl;
    cudaDeviceSynchronize();
    return;

}

template<typename T>
__global__ void 
operations_device(std::size_t npx, std::size_t npy, std::size_t npz, T* device_data_1, T* device_data_2)
{
    std::size_t b = bindx(blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
    std::size_t bsize = blockDim.x * blockDim.y * blockDim.z;
    T* t_grid_1 = &device_data_1[b * bsize];
    T* t_grid_2 = &device_data_2[b * bsize];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    if(z>=0 && z<npz)
    {
        if(y>=0 && y<npy)
        {
            if(x>=0 && x<npx)
            {
                auto t_a = t_grid_1[tindx(x,y,z,npx,npy,npz)];
                auto t_b = t_grid_2[tindx(x,y,z,npx,npy,npz)];
                t_grid_1[tindx(x,y,z,npx,npy,npz)] = exp(-(t_a + t_a*t_b - t_b*t_b)/(t_a*t_a)); //t_grid_1[tindx(x,y,z,npx,npy,npz)] + t_grid_2[tindx(x,y,z,npx,npy,npz)]/;
                //cgrid(tindx(x,y,z,npx,npy,npz)) = t_val;
                //cgrid.assign_device(tindx(x,y,z,npx,npy,npz), t_val);
            }
        }
    }
    

}

template<typename T, std::size_t D>
void 
operations(block<T,D>& cgrid, T* device_data_1, T* device_data_2)
{
    const auto npx = cgrid.get_block_size_padded()[0];
    const auto npy = cgrid.get_block_size_padded()[1];
    const auto npz = cgrid.get_block_size_padded()[2];
    //const auto nbx = cgrid.get_grid_size()[0];
    //const auto nby = cgrid.get_grid_size()[1];
    //const auto nbz = cgrid.get_grid_size()[2];
    
    dim3 num_blocks(npx, npy, npz);
    dim3 num_threads(npx, npy, npz);

    //T* data = cgrid.data_device();

    //operations_device<<<num_blocks, num_threads>>>(npx, npy, npz, device_data_1, device_data_2);
    operations_device<<<128>>>(npx, npy, npz, device_data_1, device_data_2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        //fprintf(stderr, "initialise kernel error: %s\n", cudaGetErrorString(err));
        std::cout << "initialize kernel error: " << " " << cudaGetErrorString(err) << std::endl;
    cudaDeviceSynchronize();
    return;

}

/*
__global__ void kernel(grid<double,3>** obj, std::size_t* nbx)
{
    if(blockIdx.x * blockDim.x + threadIdx.x == 0) {
        *nbx = (*obj)->get_grid_size()[0]; 
    }
}


__global__ void cudaAllocateGPUObj(grid<double,3>** obj, std::size_t* block_dim, std::size_t* grid_dim, std::size_t* pad_dim)
{
    if(blockIdx.x * blockDim.x + threadIdx.x == 0) {
        *obj = new grid<double,3>(block_dim, grid_dim, pad_dim);
        //(*obj)->allocate_host();
        //(*obj)->allocate_device();
    }
}
*/

// template<typename T, std::size_t D>
// __global__ void 
// grid<T,D>::fill_device()
// {
//     return;
// }
