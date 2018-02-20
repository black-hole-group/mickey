/* Inspiration for writing this interface correctly came from these links:
• http://www.scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html#id9
• https://docs.scipy.org/doc/numpy-1.13.0/reference/swig.interface-file.html#a-common-example 
*/

%module regrid
%{
  #define SWIG_FILE_WITH_INIT
  #include "regrid.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

/* Typemaps for the arrays
   =========================
Here we define 3 typemaps for 12 arrays. Some of them have the same shape 
but are designed only for input, other for output. 
*/
%apply (int DIM1, double* IN_ARRAY1) {(int n1, double * in_arr1), (int n2, double * in_arr2), (int n3, double * in_arr3), (int n4, double * in_arr4)};
%apply (int DIM1, int DIM2, double* IN_ARRAY2) {(int l5, int c5, double * in_arr5), (int l6, int c6, double * in_arr6), (int l7, int c7, double * in_arr7), (int l8, int c8, double * in_arr8)};
%apply (int DIM1, int DIM2, double* INPLACE_ARRAY2) {(int lo1, int co1, double * out_arr1), (int lo2, int co2, double * out_arr2), (int lo3, int co3, double * out_arr3), (int lo4, int co4, double * out_arr4)};

%rename (regrid) regrid_func;

/* As opposed to the simple SWIG example, we don’t include the header, since
there is nothing there that we wish to expose to Python */
//%include "regrid.h"

/*  Wrapper for regrid that massages the types */
%inline %{	
    /*  takes as input 12 (!!) numpy arrays */
    void regrid_func(int n1, double *in_arr1, int n2, double *in_arr2, int n3, double *in_arr3, int n4, double *in_arr4, int l5, int c5, double *in_arr5, int l6, int c6, double *in_arr6, int l7, int c7, double *in_arr7, int l8, int c8, double *in_arr8, int lo1, int co1, double *out_arr1, int lo2, int co2, double *out_arr2, int lo3, int co3, double *out_arr3, int lo4, int co4, double *out_arr4) {
        /*  calls the original funcion, providing only the size of the first */
        regrid(n1, in_arr1, n2, in_arr2, n3, in_arr3, n4, in_arr4, in_arr5, in_arr6,  in_arr7,  in_arr8,  out_arr1,  out_arr2,  out_arr3,  out_arr4);
    }
%}
