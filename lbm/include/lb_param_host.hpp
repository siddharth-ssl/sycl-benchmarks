#pragma once
#include <iostream>
#include <sycl/sycl.hpp>
//#include <cuda.h>
//#include <params.hpp>
#include <lb_model.hpp>

template<class M, typename T>
class lb_param_host
{
public:
    T m_T0;
    std::size_t m_num_mems;
    std::size_t m_num_grps;
    std::size_t m_num_vars;
    std::size_t m_num_replicas;
    std::size_t m_size;

    T* m_w;
    T* m_wm;
    T* m_cx;
    T* m_cy;
    T* m_cz;
    T* m_c2;
    T* m_cx2;
    T* m_cy2;
    T* m_cz2;
    T* m_cxy;
    T* m_cyz;
    T* m_czx;
    T* m_c2x;
    T* m_c2y;
    T* m_c2z;
    T* m_c4;

    M* m_lb_model;


//public:
    lb_param_host(sycl::queue& q);
    //~lb_param_host();

    std::size_t num_vars();
    T* w();
    T* wm();
    T* cx();
    T* cy();
    T* cz();
    T* c2();
    T* cxy();
    T* cyz();
    T* czx();
    T* cx2();
    T* cy2();
    T* cz2();
    T* c2x();
    T* c2y();
    T* c2z();
    T* c4();
};
