/*
Performs change of coordinate basis and regridding of arrays from a
polar to cartesian basis.

Variables t* are temporary ones which are not required in the code, only
for the purposes of using numpy.
*/
void modifyArray(int nxnew, int nynew, double *xnew, int t1, int t2, double *ynew, int nx, int ny, double *rho, int t3, int t4, double *p, int t5, int t6, double *v1, int t7, int t8, double *v2) {
	// goes through new array
	for (int i=0; i<nxnew; i++) {
		for (int j=0; j<nynew; j++) {
  			// Need to use 1D index for accessing array elements 
			int n=i*nynew+j;

			

			arr[n] = i*j;
		}
	}
}
