int min(const double *arr, size_t length) {
    // returns the index corresponding to the minimum value of array
    // https://codereview.stackexchange.com/a/5146/161148
    size_t i;
    double minimum = arr[0];
    int minindex=0;

    for (i = 1; i < length; ++i) {
        if (minimum > arr[i]) {
            minimum = arr[i];
            minindex=i;
        }
    }
    return minindex;
}




int search(double xref, int nx, double *x) {
	/* 
	Finds index in array x corresponding to the element with value nearest
	xref. 
	*/
	
	double min=x[0];

	// array of differences
	for (int i = 0; i<nx; ++i) {
	    if (abs(x[i]-xref) < dist) {
	        return 0;
	    }
	}
}





void regrid(int nxnew, double *xnew, int nynew, double *ynew, int nx, int ny, double *rho, int t3, int t4, double *p, int t5, int t6, double *v1, int t7, int t8, double *v2, int t9, int t10, double *rhonew, int t11, int t12, double *pnew, int t13, int t14, double *vx, int t15, int t16, double *vy) {
	/*
	Performs change of coordinate basis and regridding of arrays from a
	polar to cartesian basis.

	Variables t* are temporary ones which are not required in the code, only
	for the purposes of using numpy.
	*/

	// goes through new array
	for (int i=0; i<nxnew; i++) {
		for (int j=0; j<nynew; j++) {
  			// Need to use 1D index for accessing array elements 
			int n=i*nynew+j;

			// generates new polar coordinates arrays
			rnew=sqrt(xnew[i]*xnew[i] + ynew[j]*ynew[j]); // new r
			thnew=atan2(ynew[j], xnew[i]);	// new theta

			// locates position in old coordinate arrays


		}
	}
}


					iref=nmmn.lsd.search(rnew, r)
					jref=nmmn.lsd.search(thnew, th)