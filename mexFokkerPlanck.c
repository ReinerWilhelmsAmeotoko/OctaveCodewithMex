#include "mex.h"
#include<string.h>
#include<math.h>

/*
[Psi, phi] = mexFokkerPlanck(Psi,phi,Energy,nrepeat, dt, beta, zeta);
                    
Using 4th order Runge-Kutta, this solves the PDE system:
zeta V = -Grad(Psi(x,t));
phi_dot + V . Grad(phi(x,t)) = - Div (V)
Psi = E(x,t) - (phi_0 - phi(x,t))/beta

In which Psi is the free energy density, phi is the information density
E(x,t) is the energy potential. The intermediate velocity V is computed
as gradient of Psi. In the implementation Div(V) is replaced by Laplacian(Psi).
zeta is a friction constant, and beta the inverse temperature beta= 1/(k_B T)
Reference: Lee Jinwoo and Hajime Tanaka, Local non-equilibrium thermodynamics,
Nature Scientific Reports 5 (7832) 2015.

Compile within Octave: First clear mexFokkerPlanck if necessary, then: 
mkoctfile --mex mexFokkerPlanck.c

Author: Reiner Wilhelms-Tricarico,  November 2020.
*/

void Nabla(double *dx, double *dy, double *Field, int n1, int n2) {
  const double Coeff[7] = {-1.0/60.0, 3.0/20.0, -3.0/4.0,
			   0.0, 3.0/4.0, -3.0/20.0, 1.0/60.0};
  const int nstop = 3;
  double realsum;
  for (int k=0; k<n2; ++k) {
    for (int j=0; j<n1; ++j) {
      realsum = 0.0;
      for (int kk=-nstop; kk<=nstop; ++kk) {
	int kconv = k+kk;
	if (kconv >= 0 && kconv <n2) {
	  int iadr = kconv*n1 + j;
	  realsum += Field[iadr]*Coeff[kk+nstop];
	}
      }
      dx[k*n1+j] = realsum;	
      realsum = 0.0;
      for (int jj=-nstop; jj<=nstop; ++jj) {
	int jconv = j+jj;
	if (jconv >= 0 && jconv <n1) {
	  int iadr = k*n1 + jconv;
	  realsum += Field[iadr]*Coeff[jj+nstop];
	}
      }
      dy[k*n1+j] = realsum;	
    }
  }
}

void  Laplacian(double *Lap, double *Field, int n1, int n2) {
  const double c1=0.0, c2=-1.0/30.0, c3=-1.0/60.0,
    c4=4.0/15.0, c5=13.0/15.0, c6=-21.0/5.0;
  
  const double Coeff[5][5] = { { c1,  c2, c3, c2, c1},
			       { c2,  c4, c5, c4, c2},
			       { c3,  c5, c6, c5, c3},
			       { c2,  c4, c5, c4, c2},
			       { c1,  c2, c3, c2, c1}};
  double realsum;
  for (int k=0; k<n2; ++k) {
    for (int j=0; j<n1; ++j) {
      realsum = 0.0;
      for (int kk = -2; kk <= 2; ++kk) {
	for (int jj = -2; jj <= 2; ++jj) {
	  int kconv = k+kk;
	  int jconv = j+jj;
	  if ((kconv >= 0 && kconv<n2) && (jconv>=0 && jconv<n1)) {
	    int iadr = kconv*n1 + jconv;
	    realsum += Field[iadr]*Coeff[kk+2][jj+2];
	  }
	}
      }
      Lap[k*n1+j] = realsum;
    }
  }
}

// compute K1,K2,K3,K4 for Runge-Kutta 4th order.
void computeKn(double *Kn, double *Psi, double *phi,
	       double beta, double zeta, int n1, int n2,
	       double *LapPsi, double *DxPsi, double *DyPsi,
	       double *Dx_phi, double *Dy_phi){
  Nabla(DxPsi,DyPsi, Psi, n1, n2);
  Laplacian(LapPsi, Psi, n1,n2);
  Nabla(Dx_phi,Dy_phi, phi, n1,n2);
  for (int k=0; k<n2; ++k) {
    for (int j=0; j<n1; ++j) {
      int iadr = k*n1 + j;
      double vdotnabla = DxPsi[iadr]*Dx_phi[iadr]+DyPsi[iadr]*Dy_phi[iadr];
      Kn[iadr] = (LapPsi[iadr] + vdotnabla)/zeta;
    }
  }
}

void computeIntermediates(double *Psibar, double *phibar,
			  double *Psi, double *phi,
			  double *Energy, double beta,
			  int n1, int n2, double *Kn, double dt) {
  for (int k=0; k<n2; ++k) {
    for (int j=0; j<n1; ++j) {
      int iadr = k*n1 + j;
      phibar[iadr] = phi[iadr] + Kn[iadr]*dt;
      Psibar[iadr] = Energy[iadr] + phibar[iadr]/beta;
    }
  }
}

void
mexFunction (int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[])
{
  const mwSize *nr;
  mwSize n1, n2;
  
  if (nrhs < 3) {
    mexPrintf("mexFokkerPlanck requires at least 3 input arguments: (Psi,phi,Energy). Full list with defaults:\n");
    mexPrintf("mexFokkerPlanck(Psi,phi,Energy,  counter=1, deltaT=0.001, beta=0.5, zeta=0.5)\n");
    return;
  }
  
  nr = mxGetDimensions(prhs[0]);
  n1 = *nr;
  n2 = *(nr+1);
  const bool makeComplex = false;
  
  int    counter=1;
  double deltaT=0.001;
  double beta=0.5;
  double zeta=0.5;

  if (nrhs >= 4)  counter =  *(mxGetPr(prhs[3]));
  if (nrhs >= 5)  deltaT  =  *(mxGetPr(prhs[4]));
  if (nrhs >= 6)  beta    =  *(mxGetPr(prhs[5]));
  if (nrhs >= 7)  zeta    =  *(mxGetPr(prhs[6]));
  
  if (counter < 0 || counter > 10000) {
    mexPrintf("The counter is absurdly big or negative: %i . Refused and return\n",counter);
    return;
  }
    
  double *Psi = mxGetPr(prhs[0]);
  double *phi = mxGetPr(prhs[1]);
  double *Energy = mxGetPr(prhs[2]);

  plhs[0] =  mxCreateNumericArray(mxGetNumberOfDimensions (prhs[0]),
				  mxGetDimensions (prhs[0]),
				  mxGetClassID (prhs[0]), makeComplex);
  double *PsiOut = mxGetPr(plhs[0]);
  plhs[1] =  mxCreateNumericArray(mxGetNumberOfDimensions (prhs[0]),
				  mxGetDimensions (prhs[0]),
				  mxGetClassID (prhs[0]),
				  makeComplex);
  double *phiOut = mxGetPr(plhs[1]); 
   
  // now we need space for Laplacian and nabla output:
  mxArray *LapPointer = mxCreateDoubleMatrix(n1, n2, false);
  double *Lap = mxGetPr(LapPointer);
  mxArray *DxPointer = mxCreateDoubleMatrix(n1, n2, false);
  double *DxPsi = mxGetPr(DxPointer);
  mxArray *DyPointer = mxCreateDoubleMatrix(n1, n2, false);
  double *DyPsi = mxGetPr(DyPointer);
  mxArray *DxPhiPointer = mxCreateDoubleMatrix(n1, n2, false);
  double *Dx_phi = mxGetPr(DxPhiPointer);
  mxArray *DyPhiPointer = mxCreateDoubleMatrix(n1, n2, false);
  double *Dy_phi = mxGetPr(DyPhiPointer);
  // temporary matrices (Lap, DxPsi, DyPsi, Dx_phi, Dy_phi)
  // Now I also need all the K1,K2,K3,K4
  mxArray *K1Pointer = mxCreateDoubleMatrix(n1, n2, false);
  double *K1 = mxGetPr(K1Pointer);
  mxArray *K2Pointer = mxCreateDoubleMatrix(n1, n2, false);
  double *K2 = mxGetPr(K2Pointer);
  mxArray *K3Pointer = mxCreateDoubleMatrix(n1, n2, false);
  double *K3 = mxGetPr(K3Pointer);
  mxArray *K4Pointer = mxCreateDoubleMatrix(n1, n2, false);
  double *K4 = mxGetPr(K4Pointer);
  // But I also need intermediaray PsiBar and phiBar:
  mxArray *PsibarPtr = mxCreateDoubleMatrix(n1, n2, false);
  double *Psibar = mxGetPr(PsibarPtr);
  mxArray *phibarPtr = mxCreateDoubleMatrix(n1, n2, false);
  double *phibar = mxGetPr(phibarPtr); 

  // In this while loop, note that initially Psi and phi point at
  // the 1st and 2nd input argument. At the end of the loop,
  // these are set to point at the two output arrays, so that the
  // result from one round are used as starting values for the next round. 
  while (counter > 0) {
    computeKn(K1, Psi, phi, beta, zeta, n1, n2,
	      Lap, DxPsi, DyPsi, Dx_phi, Dy_phi);       //  12 arguments. Output K1   
    computeIntermediates(Psibar,phibar, Psi, phi,
			 Energy, beta, n1, n2,  K1, deltaT/2.0);  //  10 arguments. Output Psibar and phibar.
    
    computeKn(K2, Psibar, phibar, beta, zeta, n1, n2,
	      Lap, DxPsi, DyPsi, Dx_phi, Dy_phi);
    computeIntermediates(Psibar,phibar, Psi, phi,
			 Energy, beta, n1, n2,  K2, deltaT/2.0);
    
    computeKn(K3, Psibar, phibar, beta, zeta, n1, n2,
	      Lap, DxPsi, DyPsi, Dx_phi, Dy_phi);
    computeIntermediates(Psibar,phibar, Psi, phi,
			 Energy, beta, n1, n2,  K3, deltaT);
    
    computeKn(K4, Psibar, phibar, beta, zeta, n1, n2,
	      Lap, DxPsi, DyPsi, Dx_phi, Dy_phi);
    // combine K1, .. K4 and store the result in K1:
    for (int k=0; k<n2; ++k) {
      for (int j=0; j<n1; ++j) {
	int iadr = k*n1 + j;
	double sumkj = K1[iadr] + 2*(K2[iadr]+K3[iadr]) + K4[iadr];
	K1[iadr] = sumkj;
      }
    }
    // generating output to the 1st and 2nd output arguments.
    computeIntermediates(PsiOut, phiOut, Psi, phi,
			 Energy, beta, n1, n2,  K1, deltaT/6.0);  
    Psi = PsiOut;  // use current output as input in the
    phi = phiOut;  // next round of the while loop.
    counter -= 1;
  }
}

