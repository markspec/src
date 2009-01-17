/* Phase space evolution equation for time of 1D harmonic oscillator. */
/*
   Equation is v.dt/dx - c.x.dt/dv = 1 with c > 0 
   Analytical solution is t = 1/sqrt(c).arctan[sqrt(c).x/v] 
   for initial condition t=0 at x=0 and any nonzero v
*/
/*
  Copyright (C) 2009 University of Texas at Austin

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <math.h>
#include <float.h>

#include <rsf.h>


int main(int argc, char* argv[])
{
    int i,k,np;

    int nx,ix;                     /* number of grid points in x, counter */
    int nv,iv;                     /* number of grid points in v, counter */

    float dx,ox;                   /* increment in x, starting position */
    float dv,ov;                   /* increment in v, starting position */

    float fx,fv;                   /* factors in the system matrix */
    float x;                       /* spatial position */

    float **a,*b,c; 

    sf_axis arow,acol;
    sf_file out;

    sf_init(argc,argv);

    out = sf_output("out");

    if (!sf_getfloat("c",&c)) sf_error("Need c=");
    if (c <= 0.0) sf_error("Need strictly positive c");
    /* potential positive constant */

    if (!sf_getint("nx",&nx)) sf_error("Need nx=");
    /* number of x values. */
    if (!sf_getfloat("dx",&dx)) sf_error("Need dx=");
    /* x sampling. */
    if (!sf_getfloat("ox",&ox)) ox = -0.5*(nx-1)*dx;
    /* x origin. */

    if (!sf_getint("nv",&nv)) sf_error("Need nv=");
    /* number of v values. */
    if (!sf_getfloat("dv",&dv)) sf_error("Need dv=");
    /* v sampling. */
    if (!sf_getfloat("ov",&ov)) ov = -0.5*(nv-1)*dv;
    /* v origin. */

    /* output file parameters */ 
    arow = sf_maxa(np,0,1);
    sf_oaxa(out,arow,1);

    acol = sf_maxa(np,0,1);
    sf_oaxa(out,acol,2);

    sf_putstring(out,"label1","row");
    sf_putstring(out,"label2","column");

    /* memory allocations */
    b = sf_floatalloc(np);
    a = sf_floatalloc2(np,np);

    /* ---------------------------- */
    /* LINEAR SYSTEM MATRIX A.T = b */
    /* ---------------------------- */

    /* Linear system depends on stencils and indexing of t(ix,iv) vector */
    /* t^T = [t(1,1), ..., t(nx,1), ...., t(1,nv), ..., t(nx,nv)] */

    np = nv*nx;

    /* Right hand side vector */
    for (i = 0; i < np; i++) {
	b[i] = 1.0;
    }

    /* Initialize matrix A to zero */
    for (i = 0; i < np; i++) { /* matrix rows */
	for (k = 0; k < np; k++) { /* matrix column */
	    a[i][k] = 0.0;
	}
    }

    fv = 0.5*c/dv;
    fx = 0.5*ov/dx;

    /* First row */
    a[0][1]  =  fx;
    a[0][nx] = -fv*ox;

    /* Upper rows */
    for (i = 1; i < nx; i++) {
	a[i][i-1]  = -fx;
	a[i][i+1]  =  fx;
	a[i][i+nx] = -fv*(ox+ix*dx);
    }

    /* Medium rows */
    for (iv = 1; iv < (nv-1); iv++) {
	fx = 0.5*(ov+iv*dv)/dx;
	for (ix = 0; ix < nx; ix++){
	    i = iv*nx + ix;
	    x = ox + ix*dx;
	    a[i][i-nx] =  fv*x;
	    a[i][i-1]  = -fx;
	    a[i][i+1]  =  fx;
	    a[i][i+nx] = -fv*x;
	}
    }

    /* Lower rows */
    fx = 0.5*(ov+(nv-1)*dv)/dx;
    for (ix = 0; ix < nx; ix++){
	i = (nv-1)*nx + ix;
	a[i][i-nx] =  fv*(ox+ix*dx);
	a[i][i-1]  = -fx;
	a[i][i+1]  =  fx;
    }

    /* Last row */
    a[np-1][np-1-nx] =  fv*(ox+(nx-1)*dx);
    a[np-1][np-2]    = -fx;


    /* Boundary conditions */

    /* output matrix */
    sf_floatwrite(a[0],np*np,out);

    exit(0);
}
