#include <cuda_kernels.hpp>

/// @brief Global CUDA kernels
/// @tparam T 
/// @tparam D 
/// @return 

template<typename T>
void 
operations_device(std::size_t npx, 
                std::size_t npy, 
                std::size_t xmin, 
                std::size_t xmax,
                std::size_t ymin,
                std::size_t ymax, 
                std::size_t zmin,
                std::size_t zmax,  
                T* device_data_1, 
                T* device_data_2,
                sycl::nd_item<3> item_ct)
{
    //std::size_t b = bindx(blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y);
    //std::size_t bsize = blockDim.x * blockDim.y * blockDim.z;
    //T* t_grid_1 = &device_data_1[b * bsize];
    //T* t_grid_2 = &device_data_2[b * bsize];

    int x = item_ct.get_local_id(0) + item_ct.get_group(0) * item_ct.get_group_range(0);
    int y = item_ct.get_local_id(1) + item_ct.get_group(1) * item_ct.get_group_range(1);
    int z = item_ct.get_local_id(2) + item_ct.get_group(2) * item_ct.get_group_range(2);

    //std::cout << x << " " << y << " " << z << std::endl;
    //printf("x = %d, y = %d, z = %d, size = %d \n",x,y,z,int(b*bsize));
    //printf("Hello World From GPU!\n");

    //h << "in kernel" << " " << device_data_1[10] << sycl::endl;

    if(z>=zmin && z<=zmax)
    {
        if(y>=ymin && y<=ymax)
        {
            if(x>=xmin && x<=xmax)
            {
                auto t_a = device_data_1[tindx(x,y,z,npx,npy)];
                //auto t_b = device_data_2[tindx(x,y,z,npx,npy)];
                //device_data_1[tindx(x,y,z,npx,npy)] = exp(-(t_a + t_a*t_b - t_b*t_b)/(t_a*t_a)); //t_grid_1[tindx(x,y,z,npx,npy,npz)] + t_grid_2[tindx(x,y,z,npx,npy,npz)]/;
                device_data_2[tindx(x,y,z,npx,npy)] = exp(-(t_a + t_a*t_a)/(t_a*t_a));
                //h << "in kernel" << " " << device_data_1[10] << sycl::endl;
                //cgrid(tindx(x,y,z,npx,npy,npz)) = t_val;
                //cgrid.assign_device(tindx(x,y,z,npx,npy,npz), t_val);
            }
        }
    }
    //__syncthreads();

}

template<typename T, std::size_t D>
void 
operations(block<T,D>& t_block_1, 
        T* device_data_1, 
        T* device_data_2,
        sycl::queue& q)
{
    std::size_t npx  = t_block_1.get_block_size_padded()[0];
    std::size_t npy  = t_block_1.get_block_size_padded()[1];
    std::size_t npz  = t_block_1.get_block_size_padded()[2];
    std::size_t xmin = t_block_1.get_zone_min()[0];
    std::size_t xmax = t_block_1.get_zone_max()[0];
    std::size_t ymin = t_block_1.get_zone_min()[1];
    std::size_t ymax = t_block_1.get_zone_max()[1];
    std::size_t zmin = t_block_1.get_zone_min()[2];
    std::size_t zmax = t_block_1.get_zone_max()[2];

    std::size_t cuda_tx = static_cast<std::size_t>(10);
    std::size_t cuda_ty = static_cast<std::size_t>(10);
    std::size_t cuda_tz = static_cast<std::size_t>(10);
    std::size_t cuda_bx = npx/cuda_tx;
    std::size_t cuda_by = npy/cuda_ty;
    std::size_t cuda_bz = npz/cuda_tz;
    //std::cout << xmin << " " << xmax << " " << ymin << " " << ymax << " " << zmin << " " << zmax << std::endl;
    
    sycl::range<3> num_threads(cuda_tx, cuda_ty, cuda_tz);
    sycl::range<3> num_blocks(cuda_bx, cuda_by, cuda_bz);

    //dim3 num_blocks((130 + num_threads.x -1) / num_threads.x, (130 + num_threads.y -1) / num_threads.y, (130 + num_threads.z -1) / num_threads.z);
    //T* data = cgrid.data_device();

    
    //dpct::get_default_queue().parallel_for(
    //    sycl::nd_range<3>(num_blocks * num_threads, num_threads),
    //    [=](sycl::nd_item<3> item_ct) 
    //    {
    //        operations_device(npx, npy, xmin, xmax, ymin, ymax, zmin, zmax, device_data_1, device_data_2, item_ct);
    //    });

    //dpct::get_current_device().queues_wait_and_throw();

    auto operations_sycl = [&](sycl::handler& h){
        sycl::stream out(1024, 256, h);
        h.parallel_for(
            sycl::nd_range<3>(num_blocks * num_threads, num_threads),
            [=](sycl::nd_item<3> item_ct) 
            {
                //operations_device(npx, npy, xmin, xmax, ymin, ymax, zmin, zmax, device_data_1, device_data_2, item_ct);
                //out << "in parallel for " << " " << device_data_2[10] << sycl::endl;

                int x = item_ct.get_local_id(0) + item_ct.get_group(0) * item_ct.get_local_range(0);
                int y = item_ct.get_local_id(1) + item_ct.get_group(1) * item_ct.get_local_range(1);
                int z = item_ct.get_local_id(2) + item_ct.get_group(2) * item_ct.get_local_range(2);

                //out << item_ct.get_global_id()[2] << " " << ymin << " " << ymax << sycl::endl;
                //printf("x = %d, y = %d, z = %d, size = %d \n",x,y,z,int(b*bsize));
                //printf("Hello World From GPU!\n");

                //out << "in kernel" << " " << device_data_1[10] << sycl::endl;

                if(z>=zmin && z<=zmax)
                {
                    if(y>=ymin && y<=ymax)
                    {
                        if(x>=xmin && x<=xmax)
                        {
                            //out << x << " " << y << " " << z << sycl::endl;
                            auto t_a = device_data_1[tindx(x,y,z,npx,npy)];
                            auto t_b = device_data_2[tindx(x,y,z,npx,npy)];
                            //out << t_a << " " << t_b << sycl::endl;

                            device_data_1[tindx(x,y,z,npx,npy)] = exp(-(t_a + t_a*t_b - t_b*t_b)/(t_a*t_a)); //t_grid_1[tindx(x,y,z,npx,npy,npz)] + t_grid_2[tindx(x,y,z,npx,npy,npz)]/;
                            //device_data_2[tindx(x,y,z,npx,npy)] = exp(-(t_a + t_a*t_a)/(t_a*t_a));
                            //out << "in kernel" << " " << device_data_1[20] << sycl::endl;
                            //cgrid(tindx(x,y,z,npx,npy,npz)) = t_val;
                            //cgrid.assign_device(tindx(x,y,z,npx,npy,npz), t_val);
                        }
                    }
                }
            }
        );
    };
    try
    {
        q.submit(operations_sycl);
        q.wait();
    }
    catch (sycl::exception const& ex) 
    {
        std::cerr << "sycl kernel dpcpp error: " << ex.what() << std::endl;
    }

    return;

}
