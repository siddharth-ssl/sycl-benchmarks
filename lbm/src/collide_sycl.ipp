#include "collide_sycl.hpp"               

template<class M, typename T, std::size_t D>
void 
collide_sycl<M,T,D>::fs_to_moments(T* f, 
              T* m, 
              M* lb_param)
{
    //m[0] = 0.;
    //m[1] = 0.;
    //m[2] = 0.;
    //m[3] = 0.;
    //m[4] = 0.;
    for (std::size_t v = 0; v < (*lb_param).m_num_vars; v++) 
    {
        m[0] += f[v] * (*lb_param).m_wm[v];
        m[1] += f[v] * (*lb_param).m_cx[v];
        m[2] += f[v] * (*lb_param).m_cy[v];
        m[3] += f[v] * (*lb_param).m_cz[v];
        m[4] += f[v] * (*lb_param).m_c2[v];
    }
    //__syncthreads();
    T invs0 = static_cast<T>(1.) / m[0];
    m[1] *= invs0;
    m[2] *= invs0;
    m[3] *= invs0;
    m[4] = (m[4] - m[0] * (m[1] * m[1] + m[2] * m[2] + m[3] * m[3])) / (static_cast<T>(3.) * m[0]);
    return;
}

template<class M, typename T, std::size_t D>
void 
collide_sycl<M,T,D>::moments_to_feq(T* feq, 
               T* m, 
               M* lb_param)
{
  T f[100];
  const T rho   = m[0];
  const T ux    = m[1];
  const T uy    = m[2];
  const T uz    = m[3];
  const T theta = m[4];

  auto T0 = (*lb_param).m_T0;
  
  T rho1(0.),mxy(0.),mxz(0.),myz(0.);
  T mxx(0.),myy(0.),mzz(0.);
  T Vx(0.),Vy(0.),Vz(0.);
  T qx(0.),qy(0.),qz(0.),qm,q2m(0.);
  T oneBytheta = static_cast<T>(1.0)/theta ;
  T bx = (ux*oneBytheta);
  T by = (uy*oneBytheta);
  T bz = (uz*oneBytheta);    
  T del_theta = (theta-T0)/T0;
  T ga = static_cast<T>(0.5)*del_theta*oneBytheta;

  //compute_discrete_gaussian(bx, by, bz, ga, f, lb_param);
  
  for (std::size_t v = 0; v < (*lb_param).m_num_vars; v++) 
  {
    f[v] = (*lb_param).m_w[v] * std::exp (bx*(*lb_param).m_cx[v] 
                                     + by*(*lb_param).m_cy[v] 
                                     + bz*(*lb_param).m_cz[v] 
                                     + ga*(*lb_param).m_c2[v]);
  }
  
  const int N1(4), N2(4);
  T matA[N1*N2], matB[N2], matX[N1]; 
  
  for (std::size_t v = 0; v < (*lb_param).m_num_vars; v++) {
    rho1 += f[v] *  (*lb_param).m_wm[v];
    Vx   += f[v] *  (*lb_param).m_cx[v];
    Vy   += f[v] *  (*lb_param).m_cy[v];
    Vz   += f[v] *  (*lb_param).m_cz[v];
    mxx  += f[v] * (*lb_param).m_cx2[v];
    myy  += f[v] * (*lb_param).m_cy2[v];
    mzz  += f[v] * (*lb_param).m_cz2[v];
    mxy  += f[v] * (*lb_param).m_cxy[v];
    mxz  += f[v] * (*lb_param).m_czx[v];
    myz  += f[v] * (*lb_param).m_cyz[v];
    qx   += f[v] * (*lb_param).m_c2x[v];
    qy   += f[v] * (*lb_param).m_c2y[v];
    qz   += f[v] * (*lb_param).m_c2z[v];
    q2m  += f[v] *  (*lb_param).m_c4[v];
  }
  qm  = mxx+myy+mzz;
  T E = (ux*ux+uy*uy+uz*uz) + static_cast<T>(3.0)*theta;
  
  matA[indx(0,0,N1,N2)] = mxx-ux*Vx;
  matA[indx(0,1,N1,N2)] = mxy-ux*Vy;
  matA[indx(0,2,N1,N2)] = mxz-ux*Vz;
  matA[indx(0,3,N1,N2)] = qx -ux*qm;
   
  matA[indx(1,0,N1,N2)] = mxy-uy*Vx;
  matA[indx(1,1,N1,N2)] = myy-uy*Vy;
  matA[indx(1,2,N1,N2)] = myz-uy*Vz;
  matA[indx(1,3,N1,N2)] = qy -uy*qm;
   
  matA[indx(2,0,N1,N2)] = mxz-uz*Vx;
  matA[indx(2,1,N1,N2)] = myz-uz*Vy;
  matA[indx(2,2,N1,N2)] = mzz-uz*Vz;
  matA[indx(2,3,N1,N2)] = qz -uz*qm;
   
  matA[indx(3,0,N1,N2)] = qx -E*Vx;
  matA[indx(3,1,N1,N2)] = qy -E*Vy;
  matA[indx(3,2,N1,N2)] = qz -E*Vz;
  matA[indx(3,3,N1,N2)] = q2m-E*qm;
  
  matB[0] = rho1*ux-Vx;
  matB[1] = rho1*uy-Vy;
  matB[2] = rho1*uz-Vz;
  matB[3] = rho1*E -qm;

  matX[0] = 0.0;
  matX[1] = 0.0;
  matX[2] = 0.0;
  matX[3] = 0.0;

  //gauss_elimination(N1, N2, matA, matB, matX);
  T lTemp(0), multiplier(0);
  for (int j = 0; j < N2 - 1; j++) 
  {
    for (int i = j + 1; i < N1; i++) 
    {
      //if (A[i][j] != (T)0) {
        multiplier = matA[indx(i,j,N1,N2)] / matA[indx(j,j,N1,N2)];
        for (int k = j; k < N2; k++) {
          matA[indx(i,k,N1,N2)] = matA[indx(i,k,N1,N2)] - matA[indx(j,k,N1,N2)] * multiplier;
        }
        matB[i] = matB[i] - matB[j] * multiplier;
      //}
    }
  }
  for (int i = N1 - 1; i > -1; i--) 
  {
    for (int j = i + 1; j < N2; j++) 
    {
      lTemp = lTemp + matA[indx(i,j,N1,N2)] * matX[j];
    }
    matX[i] = (matB[i] - lTemp) / matA[indx(i,i,N1,N2)];
    lTemp = 0;
  }

  rho1 = rho / ( rho1 + matX[0]*Vx 
                      + matX[1]*Vy 
                      + matX[2]*Vz 
                      + matX[3]*qm );

  for (std::size_t v = 0; v < (*lb_param).m_num_vars; v++) 
  {
    feq[v] = f[v]*rho1 * (*lb_param).m_wm[v] * (static_cast<T>(1.0) + matX[0]*(*lb_param).m_cx[v] 
                             + matX[1]*(*lb_param).m_cy[v] 
                             + matX[2]*(*lb_param).m_cz[v] 
                             + matX[3]*(*lb_param).m_c2[v] );
  }
}

template<class M, typename T, std::size_t D>
void
collide_sycl<M,T,D>::copy_fs_from(T* f,
                              T* fs,
                              std::size_t x, 
                              std::size_t y,
                              std::size_t z,
                              std::size_t r, 
                              const std::size_t npx,
                              const std::size_t npy, 
                              const std::size_t npz, 
                              M* lb_param) 
{
  const auto num_grps     = (*lb_param).m_num_grps;
  const auto num_mems     = (*lb_param).m_num_mems;
  const auto num_replicas = (*lb_param).m_num_replicas;

  //printf("num_replicas %d \n", num_replicas);

  for (std::size_t g = 0; g < num_grps; g++) 
  {
    for (std::size_t m = 0; m < num_mems; m++) 
    {
      size_t t_idx = idx(m,g,x,y,z,r,npx,npy,npz,num_mems,num_replicas);
      fs[m + num_mems * g] = f[t_idx];
    }
  }

  // for (auto dv = 0; dv < (*lb_param).m_num_vars; dv++)
  // {
  //   auto t_idx = idx(dv,x,y,z,r,npx,npy,npz,num_mems,num_replicas);
  //   fs[dv] = f[t_idx];
  // }
  return;
}

template<class M, typename T, std::size_t D>
void
collide_sycl<M,T,D>::copy_fs_from(T* f,
                              T* fs,
                              std::size_t x, 
                              std::size_t y,
                              std::size_t z,
                              std::size_t r, 
                              const std::size_t npx,
                              const std::size_t npy, 
                              const std::size_t npz, 
                              M* lb_param,
							                const std::size_t it)
{
  const auto num_grps     = (*lb_param).m_num_grps;
  const auto num_mems     = (*lb_param).m_num_mems;
  const auto num_replicas = (*lb_param).m_num_replicas;

  fs[0] = f[idx(0, 0, x, y, z, r, npx, npy, npz, num_mems, num_replicas)];
  for (std::size_t g = 0; g < num_grps; g++) 
  {
    for (std::size_t m = 1; m < num_mems - 1; m+=2) 
    {
      fs[m + num_mems * g]     = f[idx(it%2 ? m : m + 1, g, x, y, z, r, npx, npy, npz, num_mems, num_replicas)];
	    fs[m + 1 + num_mems * g] = f[idx(it%2 ? m + 1 : m, g, x + (*lb_param).m_cx[m], y + (*lb_param).m_cy[m], z + (*lb_param).m_cz[m], r, npx, npy, npz, num_mems, num_replicas)];
    }
  }
//   for (auto dv = 0; dv < base_type::num_vars; dv++)
//   {
//     fs[dv] = f.at(dv,x,y,z,r);
//   }
  return;
}


template<class M, typename T, std::size_t D>
void
collide_sycl<M,T,D>::copy_fs_to(T* f,
                              T* fs,
                              std::size_t x, 
                              std::size_t y,
                              std::size_t z,
                              std::size_t r, 
                              const std::size_t npx,
                              const std::size_t npy, 
                              const std::size_t npz, 
                              M* lb_param) 
{
  const auto num_grps     = (*lb_param).m_num_grps;
  const auto num_mems     = (*lb_param).m_num_mems;
  const auto num_replicas = (*lb_param).m_num_replicas;

  for (std::size_t g = 0; g < num_grps; g++) 
  {
    for (std::size_t m = 0; m < num_mems; m++) 
    {
      size_t t_idx = idx(m,g,x,y,z,r,npx,npy,npz,num_mems,num_replicas);
      f[t_idx] = fs[m + num_mems * g];

      //printf("PRINT COLLIDE FROM GPU GLOBAL %f and %f \n", fs[1], f[1]);
    }
  }

  // for (auto dv = 0; dv < (*lb_param).m_num_vars; dv++)
  // {
  //   auto t_idx = idx(dv,x,y,z,r,npx,npy,npz,num_mems,num_replicas);
  //   //std::cout << t_idx << std::endl;
  //   f[t_idx] = fs[dv];
  // }
  return;
}

template<class M, typename T, std::size_t D>
void
collide_sycl<M,T,D>::copy_fs_to(T* f,
                              T* fs,
                              std::size_t x, 
                              std::size_t y,
                              std::size_t z,
                              std::size_t r, 
                              const std::size_t npx,
                              const std::size_t npy, 
                              const std::size_t npz, 
                              M* lb_param,
							                const std::size_t it)
{
  const auto num_grps     = (*lb_param).m_num_grps;
  const auto num_mems     = (*lb_param).m_num_mems;
  const auto num_replicas = (*lb_param).m_num_replicas;

  f[idx(0, 0, x, y, z, r, npx, npy, npz, num_mems, num_replicas)] = fs[0];
  for (std::size_t g = 0; g < num_grps; g++) 
  {
    for (std::size_t m = 1; m < num_mems - 1; m+=2) 
    {
      f[idx(it%2 ? m + 1 : m, g, x + (*lb_param).m_cx[m], y + (*lb_param).m_cy[m], z + (*lb_param).m_cz[m], r, npx, npy, npz, num_mems, num_replicas)] = fs[m + num_mems * g];
	    f[idx(it%2 ? m : m + 1, g, x, y, z, r, npx, npy, npz, num_mems, num_replicas)] = fs[m + 1 + num_mems * g];
    }
  }
//   for (auto dv = 0; dv < base_type::num_vars; dv++)
//   {
//     fs[dv] = f.at(dv,x,y,z,r);
//   }
  return;
}

template<class M , typename T, std::size_t D>
void 
collide_sycl<M,T,D>::collide_cuda_device (const T beta, 
                     const std::size_t npx,
                     const std::size_t npy,
                     const std::size_t npz,
                     const std::size_t xmin,
                     const std::size_t xmax,
                     const std::size_t ymin,
                     const std::size_t ymax,
                     const std::size_t zmin,
                     const std::size_t zmax,
                     M* lb_param, 
                     T* data_device,
                     sycl::nd_item<3> item_ctl1
                     )
{
  /*

  const std::size_t num_vars = 28; //lb_param.m_num_vars;

  int x = item_ctl1.get_local_id(0) + item_ctl1.get_group(0) * item_ctl1.get_local_range(0);
  int y = item_ctl1.get_local_id(1) + item_ctl1.get_group(1) * item_ctl1.get_local_range(1);
  int z = item_ctl1.get_local_id(2) + item_ctl1.get_group(2) * item_ctl1.get_local_range(2);

  //for(std::size_t r = 0; r < (*lb_param).m_num_replicas; r++)
  //{
  std::size_t r = 0;

  if(z>=zmin && z<=zmax)
  {
    if(y>=ymin && y<=ymax)
    {
      if(x>=xmin && x<=xmax)
      {
        T temp_f[28];
        T feq[28];
        T mom[5];
        T m[5];
        T f[28];

        for(int i=0; i<28; i++) 
        {
          temp_f[i] = 0.0;
          feq[i]    = 0.0;
          f[i]      = 0.0;
        }
        
        for(int i=0; i<5; i++)
        {
          mom[i] = 0.0;
          m[i]   = 0.0;
        }
        
        copy_fs_from (data_device, temp_f, x, y, z, r, npx, npy, npz, lb_param);
        fs_to_moments  (temp_f, mom, lb_param);
        moments_to_feq (feq, mom, lb_param);
        for(auto v=0; v < (*lb_param).m_num_vars; v++)
        {
          //temp_f[v] += static_cast<T>(2.) * beta * (feq[v] - temp_f[v]);
        }
        //copy_fs_to   (data_device, temp_f, x, y, z, r, npx, npy, npz, lb_param);
      }
    }
  //}
  }

  //printf("PRINT COLLIDE FROM GPU GLOBAL %f and %f \n", temp_f[1], data_device[1]);
  */
}

template<class M, typename T, std::size_t D>
void
collide_sycl<M,T,D>::collide_cuda(const std::size_t& it, block_type& b, const T beta, M* lb_model, T* d_data, sycl::queue& q)
{
  const std::size_t xmin = b.get_zone_min()[0];
  const std::size_t xmax = b.get_zone_max()[0];
  const std::size_t ymin = b.get_zone_min()[1];
  const std::size_t ymax = b.get_zone_max()[1];
  const std::size_t zmin = b.get_zone_min()[2];
  const std::size_t zmax = b.get_zone_max()[2]; 
  const auto npx  = b.get_block_size_padded()[0];
  const auto npy  = b.get_block_size_padded()[1];
  const auto npz  = b.get_block_size_padded()[2]; 
  const auto num_replicas = (*lb_model).m_num_replicas;
  const auto num_vars     = (*lb_model).m_num_vars; 
  const std::size_t ntx_cuda = 2;
  const std::size_t nty_cuda = 2;
  const std::size_t ntz_cuda = 2; 
  const std::size_t nbx_cuda = (npx)/ntx_cuda;
  const std::size_t nby_cuda = (npy)/nty_cuda;
  const std::size_t nbz_cuda = (npz)/ntz_cuda; 
  
  sycl::range<3> gblocks  (nbx_cuda, nby_cuda, nbz_cuda);
  sycl::range<3> gthreads (ntx_cuda, nty_cuda, ntz_cuda); 
  
  //std::cout << "nbx" << nbx_cuda << std::endl;
  //dpct::get_default_queue().parallel_for( 
  //  sycl::nd_range<3>(gblocks * gthreads, gthreads),
  //    [=](sycl::nd_item<3> item_ct1) 
  //    {
  //      collide_cuda_device (beta, npx,npy,npz,xmin,xmax,ymin,ymax,zmin,zmax,lb_model,d_data,item_ct1); 
  //    });
  
  //dpct::get_current_device().queues_wait_and_throw(); 

  auto collide_cuda_sycl = [&](sycl::handler& h)
  {
    sycl::stream out(1024, 256, h);
    h.parallel_for(
      sycl::nd_range<3>(gblocks * gthreads, gthreads),
      [=, this](sycl::nd_item<3> item_ct)
      {
        
        T temp_f[100];
        T feq[100];
        T mom[5];
        //collide_cuda_device (beta, npx,npy,npz,xmin,xmax,ymin,ymax,zmin,zmax,lb_model,d_data,item_ct);
        int x = item_ct.get_local_id(0) + item_ct.get_group(0) * item_ct.get_local_range(0);
        int y = item_ct.get_local_id(1) + item_ct.get_group(1) * item_ct.get_local_range(1);
        int z = item_ct.get_local_id(2) + item_ct.get_group(2) * item_ct.get_local_range(2);

        for(std::size_t r = 0; r < (*lb_model).m_num_replicas; r++)
        {
        //std::size_t r = 0;

        if(z>=zmin && z<=zmax)
        {
          if(y>=ymin && y<=ymax)
          {
            if(x>=xmin && x<=xmax)
            {
              //T temp_f[28];
              //T feq[28];
              //T mom[5];
              //T m[5];
              //T f[28];

              for(int i=0; i<100; i++) 
              {
                temp_f[i] = 0.0;
                feq[i]    = 0.0;
                //f[i]      = 0.0;
              }

              for(int i=0; i<5; i++)
              {
                mom[i] = 0.0;
                //m[i]   = 0.0;
              }

              //out << "idx = " << idx(1,x,y,z,r,npx,npy,npz,(*lb_model).m_num_mems, (*lb_model).m_num_replicas) << sycl::endl;

              copy_fs_from   (d_data, temp_f, x, y, z, r, npx, npy, npz, lb_model);
              // item_ct.barrier(sycl::access::fence_space::local_space);
              
              
              fs_to_moments  (temp_f, mom, lb_model);
              // item_ct.barrier(sycl::access::fence_space::local_space);
              moments_to_feq (feq, mom, lb_model);
              // item_ct.barrier(sycl::access::fence_space::local_space);
              
              for(int v = 0; v < (*lb_model).m_num_vars; v++)
              {
                temp_f[v] += static_cast<T>(2.) * beta * (feq[v] - temp_f[v]);
                // item_ct.barrier(sycl::access::fence_space::local_space);
              }

              copy_fs_to     (d_data, temp_f, x, y, z, r, npx, npy, npz, lb_model);
              // item_ct.barrier(sycl::access::fence_space::local_space);
              // out << "In the SYCL kernel: All execution finished" << sycl::endl;
            }
          }
        }
        }
      }
    );
  };

  try
  {
    q.submit(collide_cuda_sycl);
    q.wait();
  }
  catch (sycl::exception const& ex) 
  {
    std::cerr << "sycl kernel dpcpp error: " << ex.what() << std::endl;
  }
  
  return;
}

template<class M, typename T, std::size_t D>
void
collide_sycl<M,T,D>::collide(const std::size_t& it, grid_type& g, const T beta, M* lb_model, T* d_data, sycl::queue& q)
{
  for (auto& bpair : g.get_blocks()) 
  {
		auto& b = bpair.second;
    collide_cuda(it, b, beta, lb_model,  d_data, q);
  }
  return;
}
