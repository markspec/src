/* 1-D finite-difference wave extrapolation */
/*
  Copyright (C) 2008 University of Texas at Austin
  
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
#include <rsf.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
int main(int argc, char* argv[]) 
{
    int nx, nt, nk, ix, it, ik, sl;
    float dt, dx, x, x0,  dk, k0, k, tmp, tmpdt, pi=SF_PI;
    float *sig,  *new,  *old;
    sf_complex  *uk, *cur; 
    sf_complex  tmpex;
    kiss_fft_cfg cfg;
    float  *v, *vx; 
    sf_file inp, out, vel, grad;
    bool opt;    /* optimal padding */
    int npad;  /* padding */
    #ifdef _OPENMP
    int nth;
    #endif

    sf_init(argc,argv);
    inp = sf_input("in");
    out = sf_output("out");
    vel = sf_input("vel");   /* velocity */
    grad = sf_input("grad"); /* velocity gradient */

    if (SF_FLOAT != sf_gettype(inp)) sf_error("Need float input");
    if (SF_FLOAT != sf_gettype(vel)) sf_error("Need float input");
    if (!sf_histint(vel,"n1",&nx)) sf_error("No n1= in input");
    if (!sf_histfloat(vel,"d1",&dx)) sf_error("No d1= in input");
    if (!sf_histfloat(vel,"o1",&x0)) sf_error("No o1= in input");
    if (!sf_histint(inp,"n1",&nt)) sf_error("No n2= in input");
    if (!sf_histfloat(inp,"d1",&dt)) sf_error("No d2= in input");
    if (!sf_getbool("opt",&opt)) opt=true;
    /* if y, determine optimal size for efficiency */
    if (!sf_getint("pad",&npad)) npad=1;
    if (!sf_getint("sl",&sl)) sl=nx/2;

    sf_putint(out,"n1",nx);
    sf_putfloat(out,"d1",dx);
    sf_putfloat(out,"o1",x0);
    sf_putint(out,"n2",nt);
    sf_putfloat(out,"d2",dt);
    sf_putfloat(out,"o2",0.0); 
    nk = opt? kiss_fft_next_fast_size(nx*npad): nx*npad;
    if (nk != nx) sf_warning("padded to %d",nk);
    
    uk = sf_complexalloc(nk);

    sig    =  sf_floatalloc(nt);
    old    =  sf_floatalloc(nx);
    cur    =  sf_complexalloc(nx);
    new    =  sf_floatalloc(nx);

    v = sf_floatalloc(nx);
    vx = sf_floatalloc(nx);

    sf_floatread(v,nx,vel);
    sf_floatread(vx,nx,grad);
    sf_floatread(sig,nt,inp);		

    dk = 1./(nk*dx);
    k0 = -0.5/dx;
    sf_warning("k0: %f ",k0);
    cfg = kiss_fft_alloc(nk,0,NULL,NULL); 

    for (ix=0; ix < nx; ix++) {
     new[ix] =  0.0;
     cur[ix] =  sf_cmplx(0.0,0.0);
              }

    #ifdef _OPENMP
    #pragma omp parallel
   {nth = omp_get_num_threads();
    sf_warning("using %d threads",nth);}
    #endif

   /* dt=0.0;  DEBUG */
    /* propagation in time */
    for (it=0; it < nt; it++) {

             //     new[sl] += sig[it];
        
        for (ix=0; ix < nx; ix++) {
#ifdef SF_HAS_COMPLEX_H
	     old[ix] = creal(cur[ix]);
#else
             old[ix] =  sf_crealf(cur[ix]);
#endif
	     cur[ix] =sf_cmplx(new[ix],0.0);
              }

	kiss_fft_stride(cfg,(kiss_fft_cpx *)cur,(kiss_fft_cpx *)uk,1);/*compute  u(k) */
    #ifdef _OPENMP
    #pragma omp parallel for private(ik,ix,x,k,tmp,tmpex,tmpdt) 
    #endif

	  for (ix=0; ix < nx; ix++) {
              new[ix] = 0.0;
                  x = x0 + ix * dx;
#ifdef SF_HAS_COMPLEX_H
              for (ik=0; ik < nk/2+1; ik++) {
                  k = (k0 + ik * dk)*2.0*pi;
                  tmpdt = v[ix]*fabs(k)*dt;
                  tmp = x*k +0.5*v[ix]*(vx[ix]*k)*dt*dt;
                  tmpex = sf_cmplx(cosf(tmp),sinf(tmp));
                  if (ik == 0 || ik == nk/2) {new[ix] += creal(uk[ik]*tmpex)*cosf(tmpdt)*2.0;}
                  else { new[ix] += creal(uk[ik]*tmpex)*cosf(tmpdt)*4.0;}
                  }

#else
              for (ik=0; ik < nk/2+1; ik++) {
                  k = (k0 + ik * dk)*2.0*pi;
                  tmpdt = v[ix]*fabs(k)*dt;
                  tmp = x*k +0.5*v[ix]*(vx[ix]*k)*dt*dt;
                  tmpex = sf_cmplx(cosf(tmp),sinf(tmp));
                if (ik == 0 || ik == nk/2){new[ix] += sf_crealr(sf_crmul(sf_cmul(uk[ik],tmpex),cosf(tmpdt)*2.0));}
                else {new[ix] += sf_crealf(sf_crmul(sf_cmul(uk[ik],tmpex),cosf(tmpdt)*4.0));}
                  }
#endif
               new[ix] /= nk;
               new[ix] -= old[ix];
              }
           new[sl] = sig[it];
/*
#ifdef SF_HAS_COMPLEX_H
                  new[sl] += sf_cmplx(sig[it],0.0);
#else
	  new[sl] = sf_cadd(new[0],sf_cmplx(sig[it],0.0)); 
#endif
*/
/*
#ifdef SF_HAS_COMPLEX_H
             for(ix=0;ix<nx;ix++) wav[ix]=creal(new[ix]);
#else
             for(ix=0;ix<nx;ix++) wav[ix]=sf_crealf(new[ix]);
#endif
*/
      //  for (ix=0; ix < nx; ix++) new[ix] = sf_cmplx(wav[ix],0.0);
         sf_floatwrite(new,nx,out);
         }

   free(v);     
   free(vx);     
   free(new);     
   free(cur);     
   free(old);     
   free(uk);     
   free(sig);
sf_fileclose(vel);
sf_fileclose(grad);
sf_fileclose(inp);
sf_fileclose(out);
 
   exit(0); 
}           
           
