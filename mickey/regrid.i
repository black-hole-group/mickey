%module regrid
%{
  #define SWIG_FILE_WITH_INIT
  #include "regrid.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (int DIM1, int DIM2, double* INPLACE_ARRAY2) {(int sizex, int sizey, double *arr)};

void regrid(int nxnew, double *xnew, int nynew, double *ynew, int n1, double *r, int n2, double *th, int t1, int t2, double *rho, int t3, int t4, double *p, int t5, int t6, double *v1, int t7, int t8, double *v2, int t9, int t10, double *rhonew, int t11, int t12, double *pnew, int t13, int t14, double *vx, int t15, int t16, double *vy);

%include "regrid.h"