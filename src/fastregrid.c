#include "fastregrid.h"

int search(double xref, int length, double *x) {
    /* 
    Returns the index corresponding to the element in array x with
    value nearest xref. Obtains the difference between the array
    elements and xref, and finds the minimum.
    
    Inspired on https://codereview.stackexchange.com/a/5146/161148
    */
    int i, minindex;
    double *diff;
    double minimum;

    // defines array diff=|x-xref|
    diff = (double *)malloc(sizeof(double)*length);
    for (i = 0; i < length; i++){
    	diff[i] = fabs(x[i]-xref);
    }

    // starting values for search
    minimum = diff[0];
    minindex=0;

    // minimum search
    for (i = 1; i < length; ++i) {
        if (minimum > diff[i]) {
            minimum = diff[i];
            minindex=i;
        }
    }

    free(diff);

    return minindex;
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
	#pragma omp parallel for private(j,nnew,rnew,thnew,iref,jref,nref) collapse(2)
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
			vx[nnew]=v1[nref]*cos(th[jref])-v2[nref]*sin(th[jref]);
			vy[nnew]=v1[nref]*sin(th[jref])+v2[nref]*cos(th[jref]);			
			vz[nnew]=v3[nref]; // vphi
		}
	}
}

