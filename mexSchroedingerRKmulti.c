#include "mex.h"
#include "randmtzig.h"
#include<string.h>
#include<math.h>

/*

Update Schroedinger equation for n steps.

Compile within Octave by:
mkoctfile --mex mexSchroedingerRKmulti.c
*/

void computeLaplacian(double *LapReal, double *LapImag, const double *PsiReal,
		      const double *PsiImag, const int n1, const int n2) {
  const double c1=0.0, c2=-1.0/30.0, c3=-1.0/60.0, c4=4.0/15.0, c5=13.0/15.0, c6=-21.0/5.0;
  const double Coeff[5][5] = { { c1,  c2, c3, c2, c1},
			       { c2,  c4, c5, c4, c2},
			       { c3,  c5, c6, c5, c3},
			       { c2,  c4, c5, c4, c2},
			       { c1,  c2, c3, c2, c1}};
  
  double realsum, imagsum;
  for (int k=0; k<n2; ++k) {
    for (int j=0; j<n1; ++j) {
      realsum = 0.0;
      imagsum = 0.0;
      for (int kk = -2; kk <= 2; ++kk) {
	for (int jj = -2; jj <= 2; ++jj) {
	  int kconv = k+kk;
	  int jconv = j+jj;
	  if ((kconv >= 0 && kconv<n2) && (jconv>=0 && jconv<n1)) {
	    int iadr = kconv*n1 + jconv;
	    realsum += PsiReal[iadr]*Coeff[kk+2][jj+2];
	    imagsum += PsiImag[iadr]*Coeff[kk+2][jj+2];
	  }
	}
      }
      LapReal[k*n1+j] = realsum;
      LapImag[k*n1+j] = imagsum;
    }
  }
}

void partialStep(double *KnReal, double *KnImag,
		 double *ConvReal, double *ConvImag,
		 double *PsiReal, double *PsiImag,
		 const double *Ener, int n1, int n2,
		 double invhbar, double impulsefactor) {
  computeLaplacian(ConvReal, ConvImag, PsiReal, PsiImag, n1,n2);
  for (int k=0; k<n2; ++k) {
    for (int j=0; j<n1; ++j) {
      int    iadr = k*n1 + j;
      double psi_real =  PsiReal[iadr];
      double psi_imag =  PsiImag[iadr];
      double energy = Ener[iadr];
      double ar = impulsefactor * ConvReal[iadr] + psi_real * energy;
      double ai = impulsefactor * ConvImag[iadr] + psi_imag * energy;
      KnReal[iadr] =  ai*invhbar;
      KnImag[iadr] = -ar*invhbar; 
    }
  }
}
  

  void  addKnMakePsibar(double *PsiOutRe, double *PsiOutIm, double *KnRe, double *KnIm,
			double *PsiReal, double *PsiImag, double dt, int n1, int n2) {
    for (int k=0; k<n2; ++k) {
      for (int j=0; j<n1; ++j) {
	int iadr = k*n1 + j;
	PsiOutRe[iadr] = PsiReal[iadr] + dt*KnRe[iadr];
	PsiOutIm[iadr] = PsiImag[iadr] + dt*KnIm[iadr];
      }
    }
  }
  


void  mexFunction (int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[])
{
  const mwSize *nr;
  mwSize n1, n2;
  int ngrid;
  int ngridsquare;
  nr = mxGetDimensions(prhs[0]);
  n1 = *nr;
  n2 = *(nr+1);
  
  double *Enery = mxGetPr(prhs[1]);

  int    counter = *(mxGetPr(prhs[2]));
  double deltaT = *(mxGetPr(prhs[3]));
  double hbar = *(mxGetPr(prhs[4]));
  double mass = *(mxGetPr(prhs[5]));
  double impulsefactor =   -0.5*hbar*hbar/mass;
  double invhbar =  1.0/hbar;
  double stepscale = invhbar*deltaT;
  const bool makeComplex = true;
  
  // Generate output array for Psi
  plhs[0] =  mxCreateNumericArray(mxGetNumberOfDimensions (prhs[0]),  mxGetDimensions (prhs[0]), mxGetClassID (prhs[0]), makeComplex); // Output.
  
  mxArray *ConvPointer = mxCreateNumericArray(mxGetNumberOfDimensions (prhs[0]), mxGetDimensions (prhs[0]), mxGetClassID (prhs[0]), makeComplex);
  mxArray *Psipointer =  mxCreateNumericArray(mxGetNumberOfDimensions (prhs[0]), mxGetDimensions (prhs[0]), mxGetClassID (prhs[0]), makeComplex); 
  double *PsibarRe =  mxGetPr(Psipointer);
  double *PsibarIm =  mxGetPi(Psipointer);

  // now we need 8 double * pointers to real and imaginary part of K1,K2,K3,K4: It says create- who cleans up?
  
  mxArray *K1ptr = mxCreateNumericArray(mxGetNumberOfDimensions (prhs[0]), mxGetDimensions (prhs[0]), mxGetClassID (prhs[0]), makeComplex);
  double *K1Re = mxGetPr(K1ptr);
  double *K1Im = mxGetPi(K1ptr);
  
  mxArray *K2ptr = mxCreateNumericArray(mxGetNumberOfDimensions (prhs[0]), mxGetDimensions (prhs[0]), mxGetClassID (prhs[0]), makeComplex);
  double *K2Re = mxGetPr(K2ptr);
  double *K2Im = mxGetPi(K2ptr);
  
  mxArray *K3ptr = mxCreateNumericArray(mxGetNumberOfDimensions (prhs[0]), mxGetDimensions (prhs[0]), mxGetClassID (prhs[0]), makeComplex);
  double *K3Re = mxGetPr(K3ptr);
  double *K3Im = mxGetPi(K3ptr);
  
  mxArray *K4ptr = mxCreateNumericArray(mxGetNumberOfDimensions (prhs[0]), mxGetDimensions (prhs[0]), mxGetClassID (prhs[0]), makeComplex);
  double *K4Re = mxGetPr(K4ptr);
  double *K4Im = mxGetPi(K4ptr);
    
  double *PsiReal, *PsiImag, *ConvReal, *ConvImag, *PsiOutReal, *PsiOutImag;
  
  PsiReal = mxGetPr(prhs[0]);   // input Psi real part
  PsiImag = mxGetPi(prhs[0]);   // input Psi imag part
  mxArray *reservePointer;
  if (PsiImag == NULL) {
    mexPrintf("mexSchroedingerRKmult: First input arg is not complex:\n mxGetPi(prhs[0]) delivered a null pointer.\n No worries - An extra matrix was allocated. \n");
    reservePointer = mxCreateDoubleMatrix(n1, n2, false);
    PsiImag = mxGetPr(reservePointer);
  }
  
  PsiOutReal = mxGetPr(plhs[0]);  // output Psi real part in rhs of Octave call
  PsiOutImag = mxGetPi(plhs[0]);  // output Psi imaginary part.
  ConvReal = mxGetPr(ConvPointer);  // used as temporary storage for the Laplacian
  ConvImag = mxGetPi(ConvPointer);  // used as temporary storage 

  while (counter>0) {
    partialStep(K1Re, K1Im, ConvReal, ConvImag, PsiReal, PsiImag, Enery,  n1, n2, invhbar, impulsefactor);
    addKnMakePsibar(PsibarRe, PsibarIm, K1Re, K1Im, PsiReal, PsiImag, deltaT/2, n1,n2); //PsibarRE,PsibarIm
    
    partialStep(K2Re, K2Im, ConvReal, ConvImag, PsibarRe, PsibarIm, Enery,  n1, n2, invhbar, impulsefactor);
    addKnMakePsibar(PsibarRe, PsibarIm, K2Re, K2Im, PsiReal, PsiImag, deltaT/2, n1,n2); // output psiRe and psiIm
    
    partialStep(K3Re, K3Im, ConvReal, ConvImag, PsibarRe, PsibarIm, Enery, n1, n2, invhbar, impulsefactor);
    addKnMakePsibar(PsibarRe, PsibarIm, K3Re, K3Im, PsiReal, PsiImag, deltaT, n1,n2); // output psiRe and psiIm

    partialStep(K4Re, K4Im, ConvReal, ConvImag, PsibarRe, PsibarIm, Enery, n1, n2, invhbar, impulsefactor);

    // now put it together: Add (K1+2*K2+2*K3+K4): abusing K1 as the sum.
    
    for (int k=0; k<n2; ++k) {
      for (int j=0; j<n1; ++j) {
	int iadr = k*n1 + j;
	double sumkjRe = K1Re[iadr] + 2*(K2Re[iadr]+K3Re[iadr]) + K4Re[iadr];
	double sumkjIm = K1Im[iadr] + 2*(K2Im[iadr]+K3Im[iadr]) + K4Im[iadr];
	K1Re[iadr] = sumkjRe;
	K1Im[iadr] = sumkjIm;
      }
    }
    
    addKnMakePsibar(PsiOutReal, PsiOutImag, K1Re, K1Im, PsiReal, PsiImag, deltaT/6.0, n1,n2);
    counter -= 1;
    PsiReal = PsiOutReal;   //  becomes input  in next round.
    PsiImag = PsiOutImag;   //  for next round.
  }
}


/* 
  HOw to extend to Runge-Kutta.

I need 4 additional fields  K1 K2 K3 K4 to store the
intermediate results.  
What is now the direct output is then first K1 then K2 
etc. 
this part goes into a subroutine computeUpdate.

    for (int k=0; k<n2; ++k) {
      for (int j=0; j<n1; ++j) {
	int    iadr = k*n1 + j;
	double psi_real =  PsiReal[iadr];
	double psi_imag =  PsiImag[iadr];
	double energy = Ener[iadr];
	double ar = impulsefactor * ConvReal[iadr] + psi_real * energy;
	double ai = impulsefactor * ConvImag[iadr] + psi_imag * energy;
	PsiOutReal[iadr] = psi_real + ai*stepscale;
	PsiOutImag[iadr] = psi_imag - ar*stepscale; 
      }
    }
k1=f(y∗(t0),t0)k2=f(y∗(t0)+k1h2,t0+h2)k3=f(y∗(t0)+k2h2,t0+h2)k4=f(y∗(t0)+k3h,t0+h)

K1 is the right hand side, i.e. based on the return from last round but without updating the solution
K2 is time step result for dt/2; and starting value for K3
K3 is again time step dt/2 starting value for K4 
K4 is time step dt from K3
Combination: (K1+2K2+2K3+K4)/

The Ki are actually the "slopes" of the function f(x,t). Basically the right side of the
equation dpsi/dt =  1/ih [-h^2/2m Laplace Psi(x,t) + V(x) Psi(x,t))] = F(x,t)
So in the first case, K1 is simply based on this. K1 = F(x,t);
Then we define Y =  Psi(x,t)+K1 * dt/2, and then compute the right side again to get K2
to get K3, case we add Psi(x,t)+K2*dt/2 , and compute the right side again. For the computation of
K4 we go forward by K3 * dt, full step. Only then do we compute the actual update.

So:  define the computation of the right side as an operator Y o Psi(x,t)
K1 = Y o Psi(x,t);   Next compute  Psi1(x,t) = Psi(x,t) + K1 * dt/2;
K2 = Y o Psi1(x,t);  Next compute  Psi2(x,t) = Psi(x,t) + K2 * dt/2;
K3 = Y o Psi2(x,t);  Next compute  Psi3(x,t) = Psi(x,t) + K3 * dt; 
K4 = Y o Psi3(x,t);  Done.
Now update;  Psi(x,t+dt) = Psi(x,t) + (K1 + 2 K2 + 2 K3 + K4) * dt/6;
*/
