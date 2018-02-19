int search(double xref, size_t length, double *x) {
    /* 
    Returns the index corresponding to the element in array x with
    value nearest xref.
    
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

    return minindex;
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