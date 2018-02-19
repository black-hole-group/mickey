/*
Performs change of coordinate basis and regridding of arrays from a
polar to cartesian basis.

Variables t* are temporary ones which are not required in the code, only
for the purposes of using numpy.
*/
void regrid(int nxnew, double *xnew, int nynew, double *ynew, int nx, int ny, double *rho, int t3, int t4, double *p, int t5, int t6, double *v1, int t7, int t8, double *v2, int t9, int t10, double *rhonew, int t11, int t12, double *pnew, int t13, int t14, double *vx, int t15, int t16, double *vy) {
	// goes through new array
	for (int i=0; i<nxnew; i++) {
		for (int j=0; j<nynew; j++) {
  			// Need to use 1D index for accessing array elements 
			int n=i*nynew+j;

			// generates new polar coordinates arrays
			rnew=sqrt(xnew*xnew + ynew*ynew); // new r
			thnew=atan2(ynew, xnew);	// new theta

			// locates position in old coordinate arrays

		}
	}
}
