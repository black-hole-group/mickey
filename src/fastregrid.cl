
int search(float xref, int length, global float *x) {
    /* 
    Returns the index corresponding to the element in array x with
    value nearest xref. Obtains the difference between the array
    elements and xref, and finds the minimum.
    
    Inspired on https://codereview.stackexchange.com/a/5146/161148
    */
    int i, minindex;
    float diff;
    float minimum;

    // starting values for search
    minimum = fabs(x[0]-xref); 
    minindex=0;

    // minimum search
    for (i = 1; i < length; ++i) {
    	diff=fabs(x[i]-xref);
        if (minimum > diff) {
            minimum = diff;
            minindex=i;
        }
    }

    return minindex;
}


__kernel void regrid(const int nxnew, __global const float *xnew, const int nynew, __global const float *ynew, const int n1, __global const float *r, const int n2, __global const float *th, __global const float *rho, __global const float *p, __global const float *v1, __global const float *v2, __global const float *v3, __global float *rhonew, __global float *pnew, __global float *vx, __global float *vy, __global float *vz) {
	int i = get_global_id(0);
    int j = get_global_id(1);
    int nnew=j*nxnew+i; //i*nynew+j;

    int iref, jref, nref;
	float rnew,thnew;

	// goes through new array
    if ((i < nxnew) && (j <nynew)) {
		// generates new polar coordinates arrays
		rnew=sqrt(xnew[i]*xnew[i] + ynew[j]*ynew[j]); // new r
		thnew=atan2(ynew[j], xnew[i]);	// new theta

		// locates position in old coordinate arrays
		iref=search(rnew,n1,r);
		jref=search(thnew,n2,th);
		nref=jref*n1+iref; //iref*n2+jref;	

		// assigns arrays in new coord. basis
		rhonew[nnew]=rho[nref];
		pnew[nnew]=p[nref];

		// cartesian components of velocity vector 
		vx[nnew]=v1[nref]*cos(th[jref])-v2[nref]*sin(th[jref]);
		vy[nnew]=v1[nref]*sin(th[jref])+v2[nref]*cos(th[jref]);			
		vz[nnew]=v3[nref]; // vphi
	}
}