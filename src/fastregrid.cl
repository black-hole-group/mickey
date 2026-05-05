
int search(float xref, int length, global float *x) {
    /*
    Returns the index of the element in sorted array x nearest to xref.
    Uses binary search: O(log N) vs the previous O(N) linear scan.
    Handles arrays sorted in either direction (e.g. mickey passes
    th = pi/2 - x2 which is monotonically decreasing).
    */
    int lo = 0, hi = length - 1;
    int ascending = (x[hi] >= x[0]);
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        int go_right = ascending ? (x[mid] <= xref) : (x[mid] >= xref);
        if (go_right) lo = mid; else hi = mid;
    }
    return (fabs(xref - x[lo]) <= fabs(xref - x[hi])) ? lo : hi;
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
		// use cos(thnew)=x/r, sin(thnew)=y/r — avoids trig lookup and
		// matches the Python reference (which uses thnew, not th[jref])
		if (rnew > 0.0f) {
			float inv_rnew = native_recip(rnew);
			float cth = xnew[i] * inv_rnew;
			float sth = ynew[j] * inv_rnew;
			vx[nnew]=v1[nref]*cth - v2[nref]*sth;
			vy[nnew]=v1[nref]*sth + v2[nref]*cth;
		} else {
			vx[nnew]=v1[nref];
			vy[nnew]=0.0f;
		}
		vz[nnew]=v3[nref]; // vphi
	}
}
