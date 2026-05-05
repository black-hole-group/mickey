#include "fastregrid.h"

#pragma acc routine seq
int search(double xref, int length, double *x) {
    /*
    Returns the index of the element in sorted array x nearest to xref.
    Uses binary search: O(log N) vs the previous O(N) linear scan.
    Assumes x is monotonically increasing.
    */
    int lo = 0, hi = length - 1;
    if (xref <= x[0]) return 0;
    if (xref >= x[hi]) return hi;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (x[mid] <= xref) lo = mid; else hi = mid;
    }
    return (xref - x[lo] <= x[hi] - xref) ? lo : hi;
}







void regrid(int nxnew, double *xnew, int nynew, double *ynew, int n1, double *r, int n2, double *th, double *rho, double *p, double *v1, double *v2, double *v3, double *rhonew, double *pnew, double *vx, double *vy, double *vz) {
	/*
	Performs change of coordinate basis and regridding of arrays from a
	polar to cartesian basis.

	Variables t* are temporary ones which are not required in the code, only
	for the purposes of using numpy.
	*/

	int i,j;
	int nnew, iref, jref, nref;
	double rnew,thnew;

	// goes through new array
    #pragma acc data copyout(rhonew[0:nxnew*nynew], pnew[0:nxnew*nynew], vx[0:nxnew*nynew], vy[0:nxnew*nynew], vz[0:nxnew*nynew]) copyin(xnew[0:nxnew],ynew[0:nynew],r[0:n1],th[0:n2],rho[0:n1*n2],p[0:n1*n2],v1[0:n1*n2],v2[0:n1*n2],v3[0:n1*n2])  
    {
	#pragma acc parallel loop collapse(2)
	for (i=0; i<nxnew; i++) {
		for (j=0; j<nynew; j++) {
  			// Need to use 1D index for accessing array elements 
			nnew=i*nynew+j;

			// generates new polar coordinates arrays
			rnew=sqrt(xnew[i]*xnew[i] + ynew[j]*ynew[j]); // new r
			thnew=atan2(ynew[j], xnew[i]);	// new theta

			// locates position in old coordinate arrays
			iref=search(rnew,n1,r);
			jref=search(thnew,n2,th);
			nref=iref*n2+jref;	// 1d index

			// assigns arrays in new coord. basis
			rhonew[nnew]=rho[nref];
			pnew[nnew]=p[nref];

			// cartesian components of velocity vector
			// use cos(thnew)=x/r, sin(thnew)=y/r — avoids trig lookup and
			// matches the Python reference (which uses thnew, not th[jref])
			if (rnew > 0.0) {
				double inv_rnew = 1.0 / rnew;
				double cth = xnew[i] * inv_rnew;
				double sth = ynew[j] * inv_rnew;
				vx[nnew]=v1[nref]*cth - v2[nref]*sth;
				vy[nnew]=v1[nref]*sth + v2[nref]*cth;
			} else {
				vx[nnew]=v1[nref];
				vy[nnew]=0.0;
			}
			vz[nnew]=v3[nref]; // vphi
		}	
	} // end acc kernels
	} // end acc data region
}

