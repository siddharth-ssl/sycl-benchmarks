#include "lb_model.hpp"

template<class M, typename T>
void
lb_model<M, T>::copy_fs_from(const matrix_block<M, T, 3>& f,
                              T* fs,
                              const std::size_t x,
                              const std::size_t y,
                              const std::size_t z) const 
{
  for (std::size_t g = 0; g < base_type::num_grps; g++) {
    for (std::size_t m = 0; m < base_type::num_mems; m++) {
      fs[m + base_type::num_mems * g] = f(m, g, x, y, z);
    }
  }
  
  // for (std::size_t dv = 0; dv < base_type::num_vars; dv++) {
  //   fs[dv] = f(dv, x, y, z);
  // }
  return;
}


template<class M, typename T>
void
lb_model<M, T>::copy_fs_from(const matrix_block<M, T, 3>& f,
                              T* fs,
                              const std::size_t x,
                              const std::size_t y,
                              const std::size_t z,
                              const std::size_t& it) const 
{
  //std::cout << it << std::endl;
  fs[0] = f(0,x,y,z);
  for (std::size_t dv = 1; dv < base_type::num_vars-1; dv+=2) {
      /*
      if(it%2 == 0)
      {
        std::cout << dv << std::endl;
      }
      else
      {
        std::cout << dv + 1 << std::endl; 
      }
      */
      //std::cout << (it%2 ? dv : dv + 1) << std::endl;
      fs[dv]     = f(it%2 ? dv : dv + 1, x, y, z);
      fs[dv + 1] = f(it%2 ? dv + 1 : dv, x + M::cx(dv), y + M::cy(dv), z + M::cz(dv));
    }
  return;
}


template<class M, typename T>
void
lb_model<M, T>::copy_fs_to(matrix_block<M, T, 3>& f,
                            const T* fs,
                            const std::size_t x,
                            const std::size_t y,
                            const std::size_t z) const
{
  for (std::size_t g = 0; g < base_type::num_grps; g++) {
    for (std::size_t m = 0; m < base_type::num_mems; m++) {
      f(m, g, x, y, z) = fs[m + base_type::num_mems * g];
    }
  }
  
  // for (std::size_t dv = 0; dv < base_type::num_vars; dv++) {
  //   f(dv, x, y, z) = fs[dv];
  // }
  return;
}

template<class M, typename T>
void
lb_model<M, T>::copy_fs_to(matrix_block<M, T, 3>& f,
                              const T* fs,
                              const std::size_t x,
                              const std::size_t y,
                              const std::size_t z,
                              const std::size_t& it) const 
{
  f(0,x,y,z) = fs[0];
  for (std::size_t dv = 1; dv < base_type::num_vars-1; dv+=2) {
      f(it%2 ? dv + 1 : dv, x + M::cx(dv), y + M::cy(dv), z + M::cz(dv)) = fs[dv];  
      f(it%2 ? dv : dv + 1, x, y, z)                                     = fs[dv + 1];
    }
  return;
}

template<class M, typename T>
void
lb_model<M, T>::fs_to_moments(const T* f, T* m) const
{
  m[0] = 0.;
  m[1] = 0.;
  m[2] = 0.;
  m[3] = 0.;
  m[4] = 0.;
  for (std::size_t v = 0; v < base_type::num_vars; v++) {
    m[0] += f[v] * base_type::wm(v);
    m[1] += f[v] * base_type::cx(v);
    m[2] += f[v] * base_type::cy(v);
    m[3] += f[v] * base_type::cz(v);
    m[4] += f[v] * base_type::c2(v);
  }
  const T invs0 = static_cast<T>(1.) / m[0];
  m[1] *= invs0;
  m[2] *= invs0;
  m[3] *= invs0;
  m[4] =
    (m[4] - m[0] * (m[1] * m[1] + m[2] * m[2] + m[3] * m[3])) / (static_cast<T>(3.) * m[0]);
  return;
}

template <typename T, int N1, int N2>
void 
gauss_elimination(T (&A)[N1][N2], 
                  T (&b)[N1], 
                  T (&x)[N1]) 
{
  //Disable this loop to disable partial pivoting
  /*T tempA, rTemp;
  for (int k = 0; k < N1; k++) {
    if (std::fabs(A[k][k]) == (T)0) {
      for (int i = 0; i < N1; i++) {
        if (std::fabs(A[k][i]) != (T)0) {
          if (std::fabs(A[i][k]) != (T)0) {
            for (int l = 0; l < N2; l++) {
              tempA = A[i][l];
              A[i][l] = A[k][l];
              A[k][l] = tempA;
            }
            rTemp = b[i];
            b[i] = b[k];
            b[k] = rTemp;
            break;
          }
        }
      }
    }
  }*/
  T lTemp(0), multiplier(0);
  for (int j = 0; j < N2 - 1; j++) {
    for (int i = j + 1; i < N1; i++) {
      //if (A[i][j] != (T)0) {
        multiplier = A[i][j] / A[j][j];
        for (int k = j; k < N2; k++) {
          A[i][k] = A[i][k] - A[j][k] * multiplier;
        }
        b[i] = b[i] - b[j] * multiplier;
      //}
    }
  }
  for (int i = N1 - 1; i > -1; i--) {
    for (int j = i + 1; j < N2; j++) {
      lTemp = lTemp + A[i][j] * x[j];
    }
    x[i] = (b[i] - lTemp) / A[i][i];
    lTemp = 0;
  }
}

template<class M, typename T>
void
lb_model<M, T>::compute_discrete_gaussian (const T bx, 
                           const T by,
                           const T bz,
                           const T ga,
                           T* f) const
{
  for (std::size_t v = 0; v < base_type::num_vars; v++) {
    f[v] = base_type::w(v) * std::exp (bx*base_type::cx(v) 
                                     + by*base_type::cy(v) 
                                     + bz*base_type::cz(v) 
                                     + ga*base_type::c2(v));
  }
  return;
}

template<class M, typename T>
void
lb_model<M, T>::moments_to_feq(const T* m, T* f) const
{
  const T rho   = m[0];
  const T ux    = m[1];
  const T uy    = m[2];
  const T uz    = m[3];
  const T theta = m[4];

  T rho1(0.),mxy(0.),mxz(0.),myz(0.);
  T mxx(0.),myy(0.),mzz(0.);
  T Vx(0.),Vy(0.),Vz(0.);
  T qx(0.),qy(0.),qz(0.),qm,q2m(0.);
  T oneBytheta = static_cast<T>(1.0)/theta ;
  T bx = (ux*oneBytheta);
  T by = (uy*oneBytheta);
  T bz = (uz*oneBytheta);    
  T del_theta = (theta-base_type::T0)/base_type::T0;
  T ga = static_cast<T>(0.5)*del_theta*oneBytheta;

  compute_discrete_gaussian(bx, by, bz, ga, f);
  T matA[4][4], matB[4], matX[4];    
  for (std::size_t v = 0; v < base_type::num_vars; v++) {
    rho1 += f[v] * base_type::wm(v);
    Vx   += f[v] * base_type::cx(v);
    Vy   += f[v] * base_type::cy(v);
    Vz   += f[v] * base_type::cz(v);
    mxx  += f[v] * cx2(v);
    myy  += f[v] * cy2(v);
    mzz  += f[v] * cz2(v);
    mxy  += f[v] * cxy(v);
    mxz  += f[v] * cxz(v);
    myz  += f[v] * cyz(v);
    qx   += f[v] * c2x(v);
    qy   += f[v] * c2y(v);
    qz   += f[v] * c2z(v);
    q2m  += f[v] * c4(v);
  }
  qm  = mxx+myy+mzz;
  T E = (ux*ux+uy*uy+uz*uz) + static_cast<T>(3.0)*theta;
  
  matA[0][0] = mxx-ux*Vx;
  matA[0][1] = mxy-ux*Vy;
  matA[0][2] = mxz-ux*Vz;
  matA[0][3] = qx -ux*qm;
   
  matA[1][0] = mxy-uy*Vx;
  matA[1][1] = myy-uy*Vy;
  matA[1][2] = myz-uy*Vz;
  matA[1][3] = qy -uy*qm;
   
  matA[2][0] = mxz-uz*Vx;
  matA[2][1] = myz-uz*Vy;
  matA[2][2] = mzz-uz*Vz;
  matA[2][3] = qz -uz*qm;
   
  matA[3][0] = qx -E*Vx;
  matA[3][1] = qy -E*Vy;
  matA[3][2] = qz -E*Vz;
  matA[3][3] = q2m-E*qm;
  
  matB[0] = rho1*ux-Vx;
  matB[1] = rho1*uy-Vy;
  matB[2] = rho1*uz-Vz;
  matB[3] = rho1*E -qm;
  
  gauss_elimination(matA, matB, matX);
  rho1 = rho / ( rho1 + matX[0]*Vx 
                      + matX[1]*Vy 
                      + matX[2]*Vz 
                      + matX[3]*qm );

  for (std::size_t v = 0; v < base_type::num_vars; v++) {
    f[v] = f[v]*rho1 * base_type::wm(v) * (static_cast<T>(1.0) + matX[0]*base_type::cx(v) 
                             + matX[1]*base_type::cy(v) 
                             + matX[2]*base_type::cz(v) 
                             + matX[3]*base_type::c2(v) );
  }
}


template<class M, typename T>
void
lb_model<M, T>::collide(T* f, const T beta) const
{
	T feq[base_type::num_vars];
	T mom[5];
	fs_to_moments(f, mom);
	moments_to_feq(mom, feq);
  //std::cout << mom[0] << std::endl; 
  //std::cout << base_type::num_vars << std::endl; 
  
	for (std::size_t v = 0; v < base_type::num_vars; v++) {
		f[v] += static_cast<T>(2.) * beta * (feq[v] - f[v]);
	}
  
	return;
}

template<class M, typename T>
void
lb_model<M, T>::collide(matrix_block<M, T, 3>& b, const T beta, const std::size_t& it) const
{
	T tmp_f[base_type::num_vars];
  //T tmp_f2[base_type::num_vars];
	const auto zmin = b.get_zone_min();
    const auto zmax = b.get_zone_max();
    
	for (std::size_t z = zmin[2]; z <= zmax[2]; z++)
      for (std::size_t y = zmin[1]; y <= zmax[1]; y++)
        for (std::size_t x = zmin[0]; x <= zmax[0]; x++) {
			copy_fs_from(b, tmp_f, x, y, z);
      //copy_fs_from(b, tmp_f2, x, y, z);
		  collide(tmp_f, beta);
			copy_fs_to(b, tmp_f, x, y, z);
		}
	return;
}

template<class M, typename T>
void
lb_model<M, T>::collide(grid_type& g, const T beta, const std::size_t& it) const
{
	for (auto& bpair : g.get_blocks()) {
		auto& b = bpair.second;
		collide(b, beta, it);
	}
	return;
}




// T tmp_cx[28] = { 0., +1., 0.,  0., -1., +1., -1., +1., 0.,  -1., 0., 0., -1., +1., -1., +1., +1., -1., +1., -1.,  0., 0.,  -1., +1., -1., +1., 0.,  0. };
// T tmp_cy[28] = { 0.,  0.,  +1., 0.,  -1., -1., +1., +1., 0., 0., -1., 0., -1., -1., +1., +1., +1., +1., 0., 0., +1., -1., -1., -1., 0.,  0.,  -1., +1. };
// T tmp_cz[28] = { 0.,  0.,  0.,  +1., -1., -1., -1., -1., 0.,  0., 0., -1., +1., +1., +1., +1., 0.,  0.,  +1., +1., +1., +1., 0.,  0.,  -1., -1., -1., -1. };


template <class M, typename T>
void lb_model<M, T>::advect(matrix_block<M, T, 3>& f) const
{
    const auto zmin = f.get_zone_min();
    const auto zmax = f.get_zone_max();

     for (std::size_t z = zmin[2]; z <= zmax[2]; z++)
       for (std::size_t y = zmin[1]; y <= zmax[1]; y++)
         for (std::size_t x = zmin[0]; x <= zmax[0]; x++) {
    
    //    f( 1, x, y, z) = f( 1, x - 1, y    , z    );
    //    f( 2, x, y, z) = f( 2, x    , y - 1, z    );
    //    f( 3, x, y, z) = f( 3, x    , y    , z - 1);
    //    f(12, x, y, z) = f(12, x + 1, y + 1, z - 1);
    //    f(13, x, y, z) = f(13, x - 1, y + 1, z - 1);
    //    f(14, x, y, z) = f(14, x + 1, y - 1, z - 1);
    //    f(15, x, y, z) = f(15, x - 1, y - 1, z - 1);
    //    f(16, x, y, z) = f(16, x - 1, y - 1, z    );
    //    f(17, x, y, z) = f(17, x + 1, y - 1, z    );
    //    f(18, x, y, z) = f(18, x - 1, y    , z - 1);
    //    f(19, x, y, z) = f(19, x + 1, y    , z - 1);
    //    f(20, x, y, z) = f(20, x    , y - 1, z - 1);
    //    f(21, x, y, z) = f(21, x    , y + 1, z - 1);
    //  }		 
		 
		 
           f(0, 2, x, y, z) = f(0, 2, x, y, z); // f(m,g,x,y,z)
           f(1, 2, x, y, z) = f(1, 2, x + 1, y, z);
           f(2, 2, x, y, z) = f(2, 2, x, y + 1, z);
           f(3, 2, x, y, z) = f(3, 2, x, y, z + 1);
         }
     for (std::size_t z = zmin[2]; z <= zmax[2]; z++)
       for (std::size_t y = zmin[1]; y <= zmax[1]; y++)
         for (std::size_t x = zmin[0]; x <= zmax[0]; x++) {
           f(0, 1, x, y, z) = f(0, 1, x + 1, y + 1, z + 1);
           f(1, 1, x, y, z) = f(1, 1, x - 1, y + 1, z + 1);
           f(2, 1, x, y, z) = f(2, 1, x + 1, y - 1, z + 1);
           f(3, 1, x, y, z) = f(3, 1, x - 1, y - 1, z + 1);
         }
     for (std::size_t z = zmin[2]; z <= zmax[2]; z++)
       for (std::size_t y = zmin[1]; y <= zmax[1]; y++)
         for (std::size_t x = zmin[0]; x <= zmax[0]; x++) {
           f(2, 5, x, y, z) = f(2, 5, x + 1, y + 1, z);
           f(3, 5, x, y, z) = f(3, 5, x - 1, y + 1, z);
         }
     for (std::size_t z = zmin[2]; z <= zmax[2]; z++)
       for (std::size_t y = zmin[1]; y <= zmax[1]; y++)
         for (std::size_t x = zmin[0]; x <= zmax[0]; x++) {
           f(0, 6, x, y, z) = f(0, 6, x + 1, y, z + 1);
           f(1, 6, x, y, z) = f(1, 6, x - 1, y, z + 1);
           f(2, 6, x, y, z) = f(2, 6, x, y + 1, z + 1);
           f(3, 6, x, y, z) = f(3, 6, x, y - 1, z + 1);
         }

     for (std::size_t z = zmax[2]; z >= zmin[2]; z--)
       for (std::size_t y = zmax[1]; y >= zmin[1]; y--)
         for (std::size_t x = zmax[0]; x >= zmin[0]; x--) {
           f(0, 0, x, y, z) = f(0, 0, x, y, z);
           f(1, 0, x, y, z) = f(1, 0, x - 1, y, z);
           f(2, 0, x, y, z) = f(2, 0, x, y - 1, z);
           f(3, 0, x, y, z) = f(3, 0, x, y, z - 1);
         }
     for (std::size_t z = zmax[2]; z >= zmin[2]; z--)
       for (std::size_t y = zmax[1]; y >= zmin[1]; y--)
         for (std::size_t x = zmax[0]; x >= zmin[0]; x--) {
           f(0, 3, x, y, z) = f(0, 3, x + 1, y + 1, z - 1);
           f(1, 3, x, y, z) = f(1, 3, x - 1, y + 1, z - 1);
           f(2, 3, x, y, z) = f(2, 3, x + 1, y - 1, z - 1);
           f(3, 3, x, y, z) = f(3, 3, x - 1, y - 1, z - 1);
         }
     for (std::size_t z = zmax[2]; z >= zmin[2]; z--)
       for (std::size_t y = zmax[1]; y >= zmin[1]; y--)
         for (std::size_t x = zmax[0]; x >= zmin[0]; x--) {
           f(0, 4, x, y, z) = f(0, 4, x - 1, y - 1, z);
           f(1, 4, x, y, z) = f(1, 4, x + 1, y - 1, z);
           f(2, 4, x, y, z) = f(2, 4, x - 1, y, z - 1);
           f(3, 4, x, y, z) = f(3, 4, x + 1, y, z - 1);
         }
     for (std::size_t z = zmax[2]; z >= zmin[2]; z--)
       for (std::size_t y = zmax[1]; y >= zmin[1]; y--)
         for (std::size_t x = zmax[0]; x >= zmin[0]; x--) {
           f(0, 5, x, y, z) = f(0, 5, x, y - 1, z - 1);
           f(1, 5, x, y, z) = f(1, 5, x, y + 1, z - 1);
          }

    return;
}



template <class M, typename T>
void lb_model<M, T>::advect(grid_type& g) const
{
	for (auto& bpair : g.get_blocks()) {
		const auto bidx = bpair.first;
		auto& b = bpair.second;
		advect(b);
	}
	return;
}
