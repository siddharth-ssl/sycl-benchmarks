#pragma once

#include <cmath>
#include "D3Q27SC.hpp"
#include <iostream>
#include <grid.hpp>

template<class M, typename T>
class lb_model : public D3Q27SC<T>
{
public:
  using base_type = M; //D3Q27SC<T>;
  using grid_type = grid<matrix_block<base_type,T,3>, T, 3>;

  lb_model() : D3Q27SC<T>()
  {
    m_c.resize(base_type::num_vars);
    for (std::size_t v = 0; v < base_type::num_vars; v++) {
      m_c[v] = std::array<T,3>({ base_type::cx(v), base_type::cy(v), base_type::cz(v) });
      m_cx2[v] = base_type::cx(v) * base_type::cx(v);
      m_cy2[v] = base_type::cy(v) * base_type::cy(v);
      m_cz2[v] = base_type::cz(v) * base_type::cz(v);
      m_cxy[v] = base_type::cx(v) * base_type::cy(v);
      m_cyz[v] = base_type::cy(v) * base_type::cz(v);
      m_cxz[v] = base_type::cx(v) * base_type::cz(v);
      m_c2x[v] = base_type::c2(v) * base_type::cx(v);
      m_c2y[v] = base_type::c2(v) * base_type::cy(v);
      m_c2z[v] = base_type::c2(v) * base_type::cz(v);
      m_c4[v]  = base_type::c2(v) * base_type::c2(v);
    }
  }


  base_type lb_stencil() { return stencil; }
  const std::array<T,3>& c(const std::size_t t_v) const { return m_c[t_v]; }
  std::size_t oppc(const std::size_t t_v) const { return m_oppc[t_v]; }
  const T maxc() const { return base_type::cmax; }
  T* cx2() { return m_cx2; }
  T* cy2() { return m_cy2; }
  T* cz2() { return m_cz2; }
  T* cxy() { return m_cxy; }
  T* cyz() { return m_cyz; }
  T* czx() { return m_cxz; }
  T* c2x() { return m_c2x; }
  T* c2y() { return m_c2y; }
  T* c2z() { return m_c2z; }
  T* c4()  { return  m_c4; }
  const T cx2(const std::size_t t_v) const { return m_cx2[t_v]; }
  const T cy2(const std::size_t t_v) const { return m_cy2[t_v]; }
  const T cz2(const std::size_t t_v) const { return m_cz2[t_v]; }
  const T cxy(const std::size_t t_v) const { return m_cxy[t_v]; }
  const T cyz(const std::size_t t_v) const { return m_cyz[t_v]; }
  const T cxz(const std::size_t t_v) const { return m_cxz[t_v]; }
  const T c2x(const std::size_t t_v) const { return m_c2x[t_v]; }
  const T c2y(const std::size_t t_v) const { return m_c2y[t_v]; }
  const T c2z(const std::size_t t_v) const { return m_c2z[t_v]; }
  const T c4(const std::size_t t_v) const { return m_c4[t_v]; }

  const T
  cx2(const std::size_t t_m, const std::size_t t_g) const
  { return m_cx2[t_m + base_type::num_mems * t_g]; }

  const T
  cy2(const std::size_t t_m, const std::size_t t_g) const
  { return m_cy2[t_m + base_type::num_mems * t_g]; }

  const T
  cz2(const std::size_t t_m, const std::size_t t_g) const
  { return m_cz2[t_m + base_type::num_mems * t_g]; }

  const T
  cxy(const std::size_t t_m, const std::size_t t_g) const
  { return m_cxy[t_m + base_type::num_mems * t_g]; }

  const T
  cyz(const std::size_t t_m, const std::size_t t_g) const
  { return m_cyz[t_m + base_type::num_mems * t_g]; }

  const T
  cxz(const std::size_t t_m, const std::size_t t_g) const
  { return m_cxz[t_m + base_type::num_mems * t_g]; }

  const T
  c2x(const std::size_t t_m, const std::size_t t_g) const
  { return m_c2x[t_m + base_type::num_mems * t_g]; }

  const T
  c2y(const std::size_t t_m, const std::size_t t_g) const
  { return m_c2y[t_m + base_type::num_mems * t_g]; }

  const T
  c2z(const std::size_t t_m, const std::size_t t_g) const
  { return m_c2z[t_m + base_type::num_mems * t_g]; }

  const T
  c4(const std::size_t t_m, const std::size_t t_g) const
  { return m_c4[t_m + base_type::num_mems * t_g]; }

  void copy_fs_from(const matrix_block<M, T, 3>& f,
                    T* fs,
                    const std::size_t x,
                    const std::size_t y,
                    const std::size_t z) const;

  void copy_fs_to(matrix_block<M, T, 3>& f,
                  const T* fs,
                  const std::size_t x,
                  const std::size_t y,
                  const std::size_t z) const;
  
  void fs_to_moments(const T* f, T* m) const;
  
  void
  compute_discrete_gaussian (const T bx,
                           const T by,
                           const T bz,
                           const T ga,
                           T* f) const;

  void
  moments_to_feq(const T* m, T* f) const;

/*
  void
  moments_to_fneq_grad13(const T* sij, const T* qneq, T* f) const 
*/

void 
collide(T* f, const T beta) const;

void
collide(matrix_block<base_type, T, 3>& b, const T beta) const;

void
collide(grid_type& g, const T beta) const;

void
advect(matrix_block<base_type, T, 3>& b) const;

void
advect(grid_type& g) const;

public:
  std::size_t m_oppc[base_type::num_vars];
  std::vector<std::array<T,3>> m_c;
  T m_cx2[base_type::num_vars];
  T m_cy2[base_type::num_vars];
  T m_cz2[base_type::num_vars];
  T m_cxy[base_type::num_vars];
  T m_cyz[base_type::num_vars];
  T m_cxz[base_type::num_vars];
  T m_c2[base_type::num_vars];
  T m_c2x[base_type::num_vars];
  T m_c2y[base_type::num_vars];
  T m_c2z[base_type::num_vars];
  T m_c4[base_type::num_vars];
  std::vector<std::array<std::size_t, 2*base_type::cmax>> m_indir;

  base_type stencil;

};

template <typename T, int N1, int N2>
void 
gauss_elimination(T (&A)[N1][N2], 
                  T (&b)[N1], 
                  T (&x)[N1]);