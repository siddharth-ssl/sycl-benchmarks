#include<lb_param_host.hpp>

template<class M, typename T>
lb_param_host<M,T>::lb_param_host(sycl::queue& q)
{
    m_lb_model = new M;
    m_T0           = (*m_lb_model).T0;
    m_num_mems     = (*m_lb_model).num_mems;
    m_num_grps     = (*m_lb_model).num_grps;
    m_num_vars     = (*m_lb_model).num_vars;
    m_num_replicas = (*m_lb_model).num_replicas;
    
    
    m_w   = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_wm  = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_cx  = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_cy  = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_cz  = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_c2  = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);

    m_cx2 = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_cy2 = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_cz2 = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_cxy = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_cyz = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_czx = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_c2x = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_c2y = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_c2z = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);
    m_c4  = (T*)sycl::malloc_device(sizeof(T) * (*m_lb_model).num_vars, q);

    q.memcpy(m_w,   (*m_lb_model).lb_stencil().w(),   sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_wm,  (*m_lb_model).lb_stencil().wm(),  sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_cx,  (*m_lb_model).lb_stencil().cx(),  sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_cy,  (*m_lb_model).lb_stencil().cy(),  sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_cz,  (*m_lb_model).lb_stencil().cz(),  sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_c2,  (*m_lb_model).lb_stencil().c2(),  sizeof(T) * (*m_lb_model).num_vars).wait();

    q.memcpy(m_cx2, (*m_lb_model).cx2(), sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_cy2, (*m_lb_model).cy2(), sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_cz2, (*m_lb_model).cz2(), sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_cxy, (*m_lb_model).cxy(), sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_cyz, (*m_lb_model).cyz(), sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_czx, (*m_lb_model).czx(), sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_c2x, (*m_lb_model).c2x(), sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_c2y, (*m_lb_model).c2y(), sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_c2z, (*m_lb_model).c2z(), sizeof(T) * (*m_lb_model).num_vars).wait();
    q.memcpy(m_c4,  (*m_lb_model).c4(),  sizeof(T) * (*m_lb_model).num_vars).wait();

}

template<class M, typename T>
std::size_t 
lb_param_host<M,T>::num_vars()
{
    return m_num_vars;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::w()
{
    return m_w;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::wm()
{
    return m_wm;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::cx()
{
    return m_cx;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::cy()
{
    return m_cy;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::cz()
{
    return m_cz;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::c2()
{
    return m_c2;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::cx2()
{
    return m_cx2;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::cy2()
{
    return m_cy2;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::cz2()
{
    return m_cz2;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::cxy()
{
    return m_cxy;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::cyz()
{
    return m_cyz;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::czx()
{
    return m_czx;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::c2x()
{
    return m_c2x;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::c2y()
{
    return m_c2y;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::c2z()
{
    return m_c2z;
}

template<class M, typename T>
T* 
lb_param_host<M,T>::c4()
{
    return m_c4;
}
