#pragma once
#include <array>
#include <iostream>
#include <vector>

template<typename T>
class D3Q27SC
{
public:
    static const std::size_t Q = 27;
    static const std::size_t num_replicas = 1;
    static constexpr std::size_t num_grps = 7;//(sizeof(T) == 4 ? 4 : 7);
    static constexpr std::size_t num_mems = 4;//(sizeof(T) == 4 ? 8 : 4);
    static const std::size_t num_vars = num_mems * num_grps;
    static const std::size_t cmax = 1;
    static constexpr T T0 = 1. / 3.;

    D3Q27SC()
    {
        const T W0 = 8. / 27.;
        const T WS = 2. / 27.;
        const T WF = 1. / 54.;
        const T WB = 1. / 216.;
        
        T tmp_w[28] = { W0, WS, WS, WS, WB, WB, WB, WB, 0., WS,
                         WS, WS, WB, WB, WB, WB, WF, WF, WF, WF,
                          WF, WF, WF, WF, WF, WF, WF, WF };
        T tmp_cx[28] = { 0., +1., 0.,  0., -1., +1., -1., +1., 0.,  -1., 0., 0., -1., +1., -1., +1., +1., -1., +1., -1.,  0., 0.,  -1., +1., -1., +1., 0.,  0. };
        T tmp_cy[28] = { 0.,  0.,  +1., 0.,  -1., -1., +1., +1., 0., 0., -1., 0., -1., -1., +1., +1., +1., +1., 0., 0., +1., -1., -1., -1., 0.,  0.,  -1., +1. };
        T tmp_cz[28] = { 0.,  0.,  0.,  +1., -1., -1., -1., -1., 0.,  0., 0., -1., +1., +1., +1., +1., 0.,  0.,  +1., +1., +1., +1., 0.,  0.,  -1., -1., -1., -1. };
        
        T tmp_wm[28] = { 1., 1., 1., 1., 1., 1., 1., 1., 0., 1.,
                           1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                           1., 1., 1., 1., 1., 1., 1., 1. };

        // T tmp_w[28] = { W0, WS, WS, WS, WS, WS, WS, WF, WF, WF, WF, WF, WF, WF, WF, WF, WF, WF, WF, WB, WB, WB, WB, WB, WB, WB, WB, 0. };
 
        // T tmp_cx[28] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1,0}; // x
		    // T tmp_cy[28] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,0}; // y
		    // T tmp_cz[28] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1,0}; // z
        
        // T tmp_wm[28] = { 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 1., 0. };
        
        for (std::size_t v = 0; v < 28; v++) 
        {
            m_w[v] = tmp_w[v];
            m_cx[v] = tmp_cx[v];
            m_wm[v] = tmp_wm[v];
            m_cy[v] = tmp_cy[v];
            m_cz[v] = tmp_cz[v];
        }
    
        // for (std::size_t v = 27; v < num_vars; v++) 
        // {
        //     m_w[v] = 0;
        //     m_cx[v] = 0;
        //     m_cy[v] = 0;
        //     m_cz[v] = 0;
        // }

        for (std::size_t v = 0; v < num_vars; v++) 
        {
            // if ((0 <= v and v <= 3) or (9 <= v and v <= 11) or (16 <= v and v <= 28)) 
            // {
            //     m_sc_dvs.push_back(v);
            //     m_wm[v] = 1.;
            //     m_c2[v] =(m_cx[v]*m_cx[v] + m_cy[v]*m_cy[v] + m_cz[v]*m_cz[v]);
            // }
            // if ((4 <= v and v <= 7) or (12 <= v and v <= 15)) 
            // {
            //     m_bcc_dvs.push_back(v);
            //     m_wm[v] = 1.;
            //     m_c2[v] =(m_cx[v]*m_cx[v] + m_cy[v]*m_cy[v] + m_cz[v]*m_cz[v]);
            // }
            // if ((8 <= v and v <= 8) or (28 <= v and v < num_vars)) 
            // {
            //     m_dummy_dvs.push_back(v);
            //     m_wm[v] = 0.;
            //     m_c2[v] = 0.;
            // }

            m_c2[v] =(m_cx[v]*m_cx[v] + m_cy[v]*m_cy[v] + m_cz[v]*m_cz[v]);
        }
        FINE_REFILL_NODE0_FROM = std::array<std::size_t,2>({  1, 100});
        FINE_REFILL_NODE1_FROM = std::array<std::size_t,2>({  3, 100});
        FINE_REFILL_NODE2_FROM = std::array<std::size_t,2>({  5, 100});
        INTERPOLATE0           = std::array<std::size_t,2>({  7, 100});
        INTERPOLATE1           = std::array<std::size_t,2>({  9, 100});
        INTERPOLATE2           = std::array<std::size_t,2>({ 11, 100});
        COARSE_REFILL_NODE2_TO = std::array<std::size_t,2>({ 13, 100});
        COARSE_REFILL_NODE3_TO = std::array<std::size_t,2>({ 15, 100});
    }

  T w(const std::size_t t_v) const { return m_w[t_v]; }

  T w(const std::size_t t_m, const std::size_t t_g) const
  {
    return m_w[t_m + num_mems * t_g];
  }

  T wm(const std::size_t t_v) const { return m_wm[t_v]; }

  T cx(const std::size_t t_v) const { return m_cx[t_v]; }

  T cx(const std::size_t t_m, const std::size_t t_g) const
  {
    return m_cx[t_m + num_mems * t_g];
  }

  T cy(const std::size_t t_v) const { return m_cy[t_v]; }

  T cy(const std::size_t t_m, const std::size_t t_g) const
  {
    return m_cy[t_m + num_mems * t_g];
  }

  T cz(const std::size_t t_v) const { return m_cz[t_v]; }

  T cz(const std::size_t t_m, const std::size_t t_g) const
  {
    return m_cz[t_m + num_mems * t_g];
  }
  T c2(const std::size_t t_v) const { return m_c2[t_v]; }

  T c2(const std::size_t t_m, const std::size_t t_g) const
  {
    return m_c2[t_m + num_mems * t_g];
  }

  T* w() { return m_w; }
  T* wm() { return m_wm; }
  T* cx() { return m_cx; }
  T* cy() { return m_cy; }
  T* cz() { return m_cz; }
  T* c2() { return m_c2; } 

  const std::vector<std::size_t>&
  sc_dvs() const { return m_sc_dvs; }

  const std::vector<std::size_t>&
  bcc_dvs() const { return m_bcc_dvs; }

  const std::vector<std::size_t>&
  dummy_dvs() const { return m_dummy_dvs; }

  std::array<std::size_t,2> FINE_REFILL_NODE0_FROM;
  std::array<std::size_t,2> FINE_REFILL_NODE1_FROM;
  std::array<std::size_t,2> FINE_REFILL_NODE2_FROM;
  std::array<std::size_t,2> INTERPOLATE0;
  std::array<std::size_t,2> INTERPOLATE1;
  std::array<std::size_t,2> INTERPOLATE2;
  std::array<std::size_t,2> COARSE_REFILL_NODE2_TO;
  std::array<std::size_t,2> COARSE_REFILL_NODE3_TO;

private:
  T m_w[num_vars];
  T m_wm[num_vars];
  T m_cx[num_vars];
  T m_cy[num_vars];
  T m_cz[num_vars];
  T m_c2[num_vars];
  std::vector<std::size_t> m_sc_dvs;
  std::vector<std::size_t> m_bcc_dvs;
  std::vector<std::size_t> m_dummy_dvs;   


};