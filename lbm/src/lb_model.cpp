#include "lb_model.ipp"

template class lb_model<D3Q27SC<double>,double>;

//template void gauss_elimination(double (&A)[int][int], 
//                                double (&b)[int], 
//                                double (&x)[int]); 