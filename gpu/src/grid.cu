#pragma once
#include "../include/block.cu"
#include <iostream>
#include <array>

template<typename T, std::size_t D>
class grid : public block<T,D>
{
private:
    std::size_t m_nbx;
    std::size_t m_nby;
    std::size_t m_nbz;

    T* m_data_host;
    T* m_data_device;

public:
    __device__ __host__
    grid ();
    
    __device__ __host__
    grid (std::size_t* t_block_size, std::size_t* t_grid_size, std::size_t* t_pad_size);
    
    // __device__ inline std::size_t 
    std::size_t 
    bindx (std::size_t bx, std::size_t by, std::size_t bz);
    
    void 
    allocate_host();
    
    void 
    allocate_device();
    
    std::size_t* 
    get_block_size();
    
    __device__ __host__
    std::size_t* 
    get_grid_size();
    
    T* 
    data_host();
    
    __device__ __host__ 
    const T* 
    data_device() const;

    __device__ __host__ 
    T* 
    data_device();

    __device__ __host__
    const T& 
    operator() (std::size_t idx) const;
    
    __device__ __host__ 
    T& 
    operator() (std::size_t idx);
    
    void 
    fill_host(const T t_val);
    
    T 
    at(std::size_t idx);

    void 
    copy_data_device_to_host();

    void 
    copy_data_host_to_device();
    
    __device__ __host__
    ~grid ();

    void get_dim()
    {
        std::cout << block<T,D>::m_block_size[0] << " " << m_nby << " " << m_nbz << std::endl;
    }

    __device__ __host__ 
    void 
    assign_device(std::size_t idx, T t_val)
    {
        m_data_device[idx] = t_val;
    }

};

template<typename T, std::size_t D>
__device__ __host__
grid<T,D>::grid(){}

template<typename T, std::size_t D>
__device__ __host__
grid<T,D>::grid(std::size_t* t_block_size, std::size_t* t_grid_size, std::size_t* t_pad_size) 
            : block<T,D>::block(t_block_size, t_pad_size)
{
    m_nbx = t_grid_size[0];
    m_nby = t_grid_size[1];
    m_nbz = t_grid_size[2];
}

template<typename T, std::size_t D>
// __device__ inline std::size_t 
std::size_t 
grid<T,D>::bindx (std::size_t bx, std::size_t by, std::size_t bz)
{
    return bx + m_nbx*(by + m_nby*bz);
}

template<typename T, std::size_t D>
void 
grid<T,D>::allocate_host ()
{
    //grid<T,D>::fill();
    std::size_t npx = block<T,D>::m_block_size_padded[0];
    std::size_t npy = block<T,D>::m_block_size_padded[1];
    std::size_t npz = block<T,D>::m_block_size_padded[2];
    
    std::size_t alloc_bytes = npx*npy*npz*m_nbx*m_nby*m_nbz*sizeof(T);
    m_data_host = (T*)std::malloc(alloc_bytes);
    //cudaMalloc((void**)&m_data, alloc_bytes);
}

template<typename T, std::size_t D>
void 
grid<T,D>::allocate_device ()
{
    //grid<T,D>::fill();
    std::size_t npx = block<T,D>::m_block_size_padded[0];
    std::size_t npy = block<T,D>::m_block_size_padded[1];
    std::size_t npz = block<T,D>::m_block_size_padded[2];
    
    std::size_t alloc_bytes = npx*npy*npz*m_nbx*m_nby*m_nbz*sizeof(T);
    //m_data = (T*)std::malloc(alloc_bytes);
    cudaMalloc((void**)&m_data_device, alloc_bytes);
}

template<typename T, std::size_t D>
std::size_t*  
grid<T,D>::get_block_size()
{
    static std::size_t t_block_size[D];
    
    t_block_size[0] = block<T,D>::m_block_size_padded[0];
    t_block_size[1] = block<T,D>::m_block_size_padded[1];
    t_block_size[2] = block<T,D>::m_block_size_padded[2];

    return t_block_size;
}

template<typename T, std::size_t D>
__device__ __host__
std::size_t* 
grid<T,D>::get_grid_size()
{
    static std::size_t t_grid_size[D];

    t_grid_size[0] = m_nbx;
    t_grid_size[1] = m_nby;
    t_grid_size[2] = m_nbz;

    return t_grid_size;
}



template<typename T, std::size_t D>
__device__ __host__
const T&
grid<T,D>::operator() (std::size_t idx) const 
{
    return m_data_device[idx];
}

template<typename T, std::size_t D>
__device__ __host__
T&
grid<T,D>::operator() (std::size_t idx)
{
    return m_data_device[idx];
}


template<typename T, std::size_t D>
void 
grid<T,D>::fill_host(T t_val)
{
    std::size_t npx = block<T,D>::m_block_size_padded[0];
    std::size_t npy = block<T,D>::m_block_size_padded[1];
    std::size_t npz = block<T,D>::m_block_size_padded[2];
    
    std::size_t size = npx*npy*npz*m_nbx*m_nby*m_nbz;

    for(std::size_t i=0; i<size; i++)
    {
        m_data_host[i] = t_val;
    }
}

template<typename T, std::size_t D>
T 
grid<T,D>::at(std::size_t idx)
{
    return m_data_host[idx];
}

template<typename T, std::size_t D>
void 
grid<T,D>::copy_data_device_to_host()
{
    //grid<T,D>::fill();
    std::size_t npx = block<T,D>::m_block_size_padded[0];
    std::size_t npy = block<T,D>::m_block_size_padded[1];
    std::size_t npz = block<T,D>::m_block_size_padded[2];
    
    std::size_t alloc_bytes = npx*npy*npz*m_nbx*m_nby*m_nbz*sizeof(T);
    cudaMemcpy(m_data_host, m_data_device, alloc_bytes, cudaMemcpyDeviceToHost);
}

template<typename T, std::size_t D>
void 
grid<T,D>::copy_data_host_to_device()
{
    //grid<T,D>::fill();
    std::size_t npx = block<T,D>::m_block_size_padded[0];
    std::size_t npy = block<T,D>::m_block_size_padded[1];
    std::size_t npz = block<T,D>::m_block_size_padded[2];
    
    std::size_t alloc_bytes = npx*npy*npz*m_nbx*m_nby*m_nbz*sizeof(T);
    cudaMemcpy(m_data_device, m_data_host, alloc_bytes, cudaMemcpyHostToDevice);
}

template<typename T, std::size_t D>
T* 
grid<T,D>::data_host()
{
    return m_data_host;
}

template<typename T, std::size_t D>
__device__ __host__
T* 
grid<T,D>::data_device()
{
    return m_data_device;
}

template<typename T, std::size_t D>
__device__ __host__
const T* 
grid<T,D>::data_device() const 
{
    return m_data_device;
}


/// @brief Global CUDA kernels
/// @tparam T 
/// @tparam D 
/// @return 

__device__ inline std::size_t 
bindx(std::size_t bx, std::size_t by, std::size_t bz, 
    std::size_t nbx, std::size_t nby, std::size_t nbz)
{
    return  bx + nbx * (by + nby * bz);
}

__device__ inline std::size_t
tindx(std::size_t x, std::size_t y, std::size_t z, 
    std::size_t npx, std::size_t npy, std::size_t npz)
{
    return  x + npx * (y + npy * z);
}

template<typename T,std::size_t D>
__global__ void 
fill_device(const T t_val, std::size_t npx, std::size_t npy, std::size_t npz, grid<T,D>& cgrid)
{
    std::size_t b = bindx(blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
    std::size_t bsize = blockDim.x * blockDim.y * blockDim.z;
    T* t_grid = &cgrid.data_device()[b * bsize];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    if(z>=0 && z<npz)
    {
        if(y>=0 && y<npy)
        {
            if(x>=0 && x<npx)
            {
                //t_grid[tindx(x,y,z,npx,npy,npz)] = t_val;
                //cgrid(tindx(x,y,z,npx,npy,npz)) = t_val;
                cgrid.assign_device(tindx(x,y,z,npx,npy,npz), t_val);
            }
        }
    }
    

}

template<typename T, std::size_t D>
void 
fill(const T t_val, grid<T,D>** cgrid)
{
    //const auto npx = (*cgrid)->get_block_size()[0];
    //const auto npy = (*cgrid)->get_block_size()[1];
    //const auto npz = (*cgrid)->get_block_size()[2];
    const auto nbx = (*cgrid)->get_grid_size()[0];
    const auto nby = (*cgrid)->get_grid_size()[1];
    const auto nbz = (*cgrid)->get_grid_size()[2];
    
    dim3 num_blocks(nbx, nby, nbz);
    //dim3 num_threads(npx, npy, npz);

    //T* data = cgrid.data_device();

    //fill_device<<<num_blocks, num_threads>>>(t_val, npx, npy, npz, cgrid);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        //fprintf(stderr, "initialise kernel error: %s\n", cudaGetErrorString(err));
        std::cout << "initialize kernel error: " << " " << cudaGetErrorString(err) << std::endl;
    cudaDeviceSynchronize();
    return;

}


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


// template<typename T, std::size_t D>
// __global__ void 
// grid<T,D>::fill_device()
// {
//     return;
// }


template<typename T, std::size_t D>
__device__ __host__
grid<T,D>::~grid(){}

int main(int argc, char** argv)
{
    std::size_t block_dim[3]  = {4,4,4};
    std::size_t grid_dim[3]   = {16,16,16};
    std::size_t pad_dim[3]    = {1,1,1};
    ///std::cout << block_dim[1] << " " << std::endl;
    grid<double,3> cuda_grid(block_dim, grid_dim, pad_dim);
    //grid<double,3>* ptr_grid = new grid<double,3>(block_dim, grid_dim, pad_dim);
    cuda_grid.allocate_host();
    cuda_grid.allocate_device();
    
    std::size_t* arr = cuda_grid.get_block_size();
    std::cout << arr[0] << std::endl;
    std::size_t* ptr = cuda_grid.get_grid_size();
    std::cout << ptr[0] << std::endl;

    //std::cout << blockDim.x << std::endl;

    // cuda_grid.fill();
    // std::cout << cuda_grid.bindx(1,3,5) << std::endl;
    cuda_grid.get_dim();


    grid<double,3>** ptr_grid;
    cudaMalloc(&ptr_grid, sizeof(grid<double,3>*));
    cudaAllocateGPUObj<<<1,1>>>(ptr_grid, block_dim, grid_dim, pad_dim);
    std::size_t nx, *nbx;
    cudaMalloc((void**)&nbx, sizeof(std::size_t));
    kernel<<<1,1>>>(ptr_grid, nbx);
    cudaMemcpy(&nx, nbx, sizeof(std::size_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //cuda_grid.data()[10] = 1.0;
    
    //double* temp = cuda_grid.data();

    cuda_grid.fill_host(1.0);
    cuda_grid.copy_data_host_to_device();
    //fill(0.2, ptr_grid);
    cuda_grid.copy_data_device_to_host();

    //fill(1.0, cuda_grid);
    //cuda_grid(10) = 0.2;

    std::cout << cuda_grid.at(10) << " " << nx << std::endl;

    // double sample_val;
    //cudaMemcpy(&sample_val, &cuda_grid.data()[10], sizeof(double), cudaMemcpyDeviceToHost);
    //std::cout << sample_val << std::endl; 


    //std::cout << cuda_grid.get_data() << std::endl;
    //cuda_grid<double,3> arr2(new double(10)); 
    //arr2 = &cuda_grid.get_data()[1];

    //cuda_grid.data()[0] = static_cast<double>(1);


}
