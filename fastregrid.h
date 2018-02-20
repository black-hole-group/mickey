#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

int search(double xref, int length, double *x);

void regrid(int nxnew, double *xnew, int nynew, double *ynew, int n1, double *r, int n2, double *th, double *rho, double *p, double *v1, double *v2, double *rhonew, double *pnew, double *vx, double *vy);
