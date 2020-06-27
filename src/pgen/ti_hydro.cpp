/*==============================================================================
 * FILE: ti.cpp
 *
 *  PURPOSE: Problem generator for thermal instability.  Setup to use the
 *   Koyama & Inutsuka cooling function, so must use cgs units for calculations
 *
 *  TI test w/ Isotropic Conduction - hydro
 *   The coefficient of conductivity kappa should be defined in cgs unit
 *
 *  TI test w/ Anisotropic Conduction - mhd
 *   The coefficient of Spitzer Coulombic conductivity should be given.
 *
 *  Various test cases are possible
 *  (iprob=1) : TI test w/ isotropic conduction in 1D - sinusoidal fluctuations
 *  (iprob=2) : TI test w/ isotropic conduction in 2D
 *                       with sinusoidal fluctuations along the diagonal
 *  (iprob=3) : TI test w/ conduction in 2D with random pressure purtubation
 *              in case of anisotropic conduction : mhd - Diagonal field line
 *  (iprob=4) : TI test w/ conduction in 2D with random pressure purtubation
 *              in case of anisotropic conduction : mhd - Tangled field line
 *============================================================================*/


// C/C++ headers
#include <cmath>
#include <float.h>     // DBL_EPSILON for ran2()
#include <fstream>
#include <iostream>   // endl
#include <iomanip>
#include <math.h>      // pow()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../globals.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/utils.hpp" //ran2()

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// These constants are in cgs */
static const Real mbar = (1.27)*(1.6733e-24); // mean molecular weight
static const Real kb = 1.380658e-16; // Boltzmann constant
static const Real pi = 3.14159265358979323846;

/*static Real Growth_Rates_No_Conduction[12] = {0.0,
    4.5504559555398776e-15, 7.640163514875047e-15, 9.833475812271293e-15,
    1.1429025127328127e-14, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0};

static Real Growth_Rates_K10to6[12] = {0.0,
    4.550324370035095e-15, 7.63928935918264e-15, 9.830917368376283e-15,
    1.142364123910077e-14, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0};*/

static Real gam, gam1;
bool saturation_on;
static Real parker_conductivity(const AthenaArray<Real> &prim, int k, int j, int i);
void ParkerConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
//void PrandtlViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
  //                       const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void cooling(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

Real MyTimeStep(MeshBlock *pmb);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
    if (pin->GetOrAddBoolean("problem","cooling", false)){
        std::cout << "cooling enabled" << std::endl;
        EnrollUserExplicitSourceFunction(cooling);
    }
    else
        std::cout << "cooling disabled" << std::endl;
    
    if (pin->GetOrAddBoolean("problem", "user_dt", false)){
        std::cout << "user timestep enabled" << std::endl;
        EnrollUserTimeStepFunction(MyTimeStep);
    }
    else
        std::cout << "user timestep disabled" << std::endl;
    
    if (pin->GetOrAddBoolean("problem", "conduction_function", false)){
        std::cout << "Conduction function" << std::endl;
        EnrollConductionCoefficient(ParkerConduction);
    }
    else
        std::cout << "Constant Conduction" << std::endl;

//    EnrollViscosityCoefficient(PrandtlViscosity);
        
    saturation_on = pin->GetOrAddBoolean("problem", "saturation_on", true);
    return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
    AllocateUserOutputVariables(7);
    SetUserOutputVariableName(0, "rho");
    SetUserOutputVariableName(1, "n");
    SetUserOutputVariableName(2, "press");
    SetUserOutputVariableName(3, "Temp");
    SetUserOutputVariableName(4, "vel1");
    SetUserOutputVariableName(5, "vel2");
    SetUserOutputVariableName(6, "vel3");
    return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    int i=0,j=0,k=0,n=0,m=0,l=0;
    int iprob,
    nkx=16,nky=8,nkz=8;
    int nx1 = (ie-is)+1 + 2*(NGHOST);
    int nx2 = (je-js)+1 + 2*(NGHOST);
    int nx3 = (ke-ks)+1 + 2*(NGHOST);
    Real P_k, P0, n0, T0, kappa,chi, x1L, x2L, x3L, x12L, num,
    drho, dp,
    krho, krho_i, krho_ij, angle, phi, phi2, phi3, ksi, tmp,
    rho0, rho1, n_anal, r, r_ij, r_proj,
    beta, b0, Btot=0.0,Bsum=0.0;
    //***az,***ax,***ay,amp,kx[16],ky[8],kz[8],xa,ya,za;
    int64_t iseed = -1, rseed;
    // Read problem parameters
    P_k = pin->GetReal("problem","P_k"); // P/k in cgs [K/cm-3]
    n0    = pin->GetReal("problem","n0"); // number density in cgs
    x1L    = pin->GetReal("mesh","x1max");
    x2L   = pin->GetReal("mesh","x2max");
    x3L   = pin->GetReal("mesh","x3max");
    iprob = pin->GetInteger("problem","iprob"); //problem switch
    gam = peos->GetGamma();
    gam1 = gam - 1.0;

#if MAGNETIC_FIELDS_ENABLED
    //    beta  = par_getd("problem","beta"); // beta = Pgas/Pmag = 8piPgas/B^2
    b0    = pin->GetReal("problem","b0");
    //    b0 = sqrt(2.0*P_k*kb/beta); // B0 = B/sqrt(4pi) - beta=1E6
#endif

    // Set up Initial Temperature and density
    T0 = P_k / n0;
    rho0 = n0 * mbar;
    P0 = P_k * kb;
    //std::cout << rho0 << std::endl;
    drho  = pin->GetReal("problem","drho");
    n_anal = pin->GetReal("problem","n_anal"); // analytic growth rate - Field 1965
    num   = pin->GetReal("problem","num"); // wavenumber n
    kappa = pin->GetReal("problem","kappa_iso"); // conductivity
    
    /*if(kappa == 0 || kappa == 0.0) {
        n_anal = Growth_Rates_No_Conduction[(int)(num)];
    }
    else if(kappa == 1.0e6) {
        n_anal = Growth_Rates_K10to6[(int)(num)];
    }*/
    
    
    /* For (iprob=1) -- TI w/ isotropic conduction - growth rate test in 1D
     *   2 with sinusoidal fluctuations in density, pressure and velocity
     *    drho (initial fluctuation amplitude in density) should be given.
     *    krho = 2pi/x1L * n */
    if(iprob==1) {
        krho_i = 2.0 * pi * num / (float)(ie-is+1);
        //std::cout << ie << ' ' << is << std::endl;
        //std::cout << krho_i << std::endl;
        krho = 2.0 * pi * num / x1L;
        //std::cout << x1L << std::endl;
        //std::cout << krho;
        rho1 = rho0 * drho; // rho1 is amplitude of density fluctuation
        for (k=ks; k<=ke; ++k) {
        for (j=js; j<=je; ++j) {
        for (i=is; i<=ie; ++i) {
            phydro->u(IDN,k,j,i) = rho0 + rho1*cos(krho_i*(float)(i-is)); // U.d conserved density
            //std::cout << phydro->u(IDN,k,j,i) << std::endl;
            phydro->u(IM1,k,j,i) = (-1.0) * phydro->u(IDN,k,j,i) * (n_anal / krho) * (rho1 / rho0) * sin(krho_i*(i-is)); // U.M1 conserved momentum density in 1-direction
            phydro->u(IM2,k,j,i) = 0.0;
            phydro->u(IM3,k,j,i) = 0.0;
            phydro->u(IEN,k,j,i) = P0/gam1 - n_anal*n_anal/ (krho * krho * gam1) * rho1*cos(krho_i*(i-is)) + 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i); // U.E conserved total energy density
        } // i for loop
        } // j for loop
        } // k for loop
    } // if iprob==1

    /* For (iprob=2) -- TI w/ isotropic conduction - growth rate test in 2D
     *    with sinusoidal fluctuations in density, pressure and velocity
     *    drho (initial fluctuation amplitude in density) should be given.
     *    krho = 2pi/x12L * n  - x12L (diagonal length of the 2D box)      */
    if(iprob==2) {
        angle = atan(x2L/x1L);
        x12L = pow(x1L*x1L + x2L*x2L, 0.5); // length of diagonal in cgs
        r_ij = pow(pow((float)(ie-is),2.0)+pow((float)(je-js),2.0),0.5); // length of diagonal in code length
        krho = 2.0 * pi * num / x12L;
        krho_ij = 2.0 * pi * num / r_ij;
        rho1 = rho0 * drho;
        for (k=ks; k<=ke; ++k) {
        for (j=js; j<=je; ++j) {
        for (i=is; i<=ie; ++i) {
            r = pow(pow((float)(i-is),2.0)+pow((float)(j-js),2.0),0.5);
            if ((i-is) == 0) {
                phi = 0;
            }
            else{
                phi = atan((float)(j-js)/(float)(i-is));
            }
            r_proj = r * sin(phi + angle) / sin(2.0 * angle);
            phydro->u(IDN,k,j,i) = rho0 + rho1 * cos(krho_ij * r_proj);
            phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i) * n_anal * rho1 * sin(krho_ij * r_proj) / rho0 / krho * cos(angle) * (-1.0);
            phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i) * n_anal * rho1 * sin(krho_ij * r_proj) / rho0 / krho * sin(angle) * (-1.0);
            phydro->u(IM3,k,j,i) = 0.0;
            phydro->u(IEN,k,j,i) = (n0*kb*T0)/gam1 - n_anal*n_anal/ (krho * krho * gam1) * rho1 * cos(krho_ij * r_proj) + 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        } // i for loop
        } // j for loop
        } // k for loop
    } // if iprob == 2

    /* For (iprob=3) -- TI w/ (an)isotropic conduction - growth rate test in 2D
     *    with RANDOM fluctuations in pressure
     *    dp (initial fluctuation amplitude in pressure) should be given.
     *    iprob=3 - if MHD - diagonal streight field line /
     if HYDRO - isotropic conduction test with random perturbation in pressure  */
    if(iprob==3) {
        dp  = pin->GetReal("problem","dp"); // amplitude of pressure perturbations
        angle = atan(x2L/x1L);
				
				#ifdef MPI_PARALLEL
   				 rseed += Globals::my_rank;
				#endif
        for (k=ks; k<=ke; ++k) {
        for (j=js; j<=je; ++j) {
        for (i=is; i<=ie; ++i) {
                    phydro->u(IDN,k,j,i) = rho0;
                    phydro->u(IM1,k,j,i) = 0.0;
                    phydro->u(IM2,k,j,i) = 0.0;
                    phydro->u(IM3,k,j,i) = 0.0;
                    phydro->u(IEN,k,j,i) = P0/gam1 + P0/gam1 * (ran2(&rseed) - 0.5) * dp
                    + 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i))
                           + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i); // E = P0/(gam-1) + P0*(random # between -0.5 and +0.5)*(amplitude of pressure perturbation)/(gam-1) + (1/2)*(momentum^2)/rho0
            
                     #if MAGNETIC_FIELDS_ENABLED
                     pfield->b.x1f(k,j,i) = b0 * cos(angle);
                     if (i == ie) pfield->b.x1f(k,j,i+1) = b0 * cos(angle);
                     
                     pfield->b.x2f(k,j,i) = b0 * sin(angle);
                     if (j==je) pfield->b.x2f(k,j+1,i) = b0 * sin(angle);
                     
                     pfield->b.x3f(k,j,i) = 0.0;
                     if (k==ke) pfield->b.x3f(k+1,j,i) = 0.0;
            
                     phydro->u(IEN,k,j,i) += 0.5*(SQR(b0 * cos(angle))
                     + SQR(b0 * sin(angle)) + SQR(0.0));
                     #endif /* MHD */
                    
        } // i for loop
        } // j for loop
        } // k for loop
    } // if iprob==3


    if(iprob==4) {
//        std::cout << "(nx1,nx2) = " << nx1 << ',' <<  nx2 << std::endl;
//        std::cout << "(is,js,ks) = " << is << js << ks << std::endl;
        rseed = 0;
        Real A  = pin->GetReal("problem","A"); // amplitude of pressure perturbations
        int imax  = pin->GetReal("problem","imax"); // imax for maximum wavenumber
        Real n = pin->GetReal("problem","n"); // spectral index
        Real L = std::min(x1L,x2L);
        Real kmax = imax*2.0*pi/L;
/*        std::cout << "A = " << A << std::endl;
        std::cout << "imax = " << imax << std::endl;
        std::cout << "n = " << n << std::endl;
        std::cout << "L = " << L/3.08e18 << "pc" << std::endl;*/
        Real total = 0.0;
        Real mean2 = 0.0;
       /* Real **delta = new Real*[nx1];
        for (int i = 0; i < nx1; ++i) {
        	delta[i] = new Real[nx2];
        }*/
        AthenaArray<Real> delta;
        delta.NewAthenaArray(nx3,nx2,nx1);
        for (int ki=0; ki<=imax; ++ki) {
        for (int kj=0; kj<=imax; ++kj) {
                   Real k_i = ki*2.0*pi/L;
                   Real k_j = kj*2.0*pi/L;
                   Real phi_x = ran2(&rseed)*2.0*pi;
                   Real phi_y = ran2(&rseed)*2.0*pi;
                   for (k=ks; k<=ke; ++k) {
                   for (j=js; j<=je; ++j) {
                   for (i=is; i<=ie; ++i) {
                    //  std::cout << "(ki,kj) = " << ki << ',' << kj << std::endl;
                      if ((ki==0) && (kj==0)) {
                          delta(k,j,i) = 0; } 
                      Real x = pcoord->x1v(i);
                      Real y = pcoord->x2v(j);
                      Real z = pcoord->x3v(k);
                      if ((pow(k_i,2.0)+pow(k_j,2.0)) != 0) {
                      	delta(k,j,i) += pow((pow(k_i,2.0)+pow(k_j,2.0)),(n/2.0))*sin(x*k_i+phi_x)*sin(y*k_j+phi_y);
                      }
                      else {
                     	  delta(k,j,i) = 0.0;
                      }
                     // std::cout << "delta(" << k << ',' << j << ',' << i << ") = " << delta(k,j,i) << std::endl;
					         }}}	
        }}
        for (k=ks; k<=ke; ++k) {
        for (j=js; j<=je; ++j) {
        for (i=is; i<=ie; ++i) {
        	total += pow(delta(k,j,i),2.0);
        }}}
        #ifdef MPI_PARALLEL
        double global_total;
        MPI_Allreduce(&total, &global_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        mean2 = global_total/((ie-is+1)*(je-js+1));
        #endif
        #ifndef MPI_PARALLEL
        mean2 = total/((ie-is+1)*(je-js+1));
        #endif
        for (k=ks; k<=ke; ++k) {
        for (j=js; j<=je; ++j) {
        for (i=is; i<=ie; ++i) {
//                    std::cout << delta[i][j] << std::endl;
                    phydro->u(IDN,k,j,i) = rho0+rho0*delta(k,j,i)*pow((A*A/mean2),0.5);
                    phydro->u(IM1,k,j,i) = 0.0;
                    phydro->u(IM2,k,j,i) = 0.0;
                    phydro->u(IM3,k,j,i) = 0.0;
                    phydro->u(IEN,k,j,i) = P0/gam1
                    + 0.5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i))
                           + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i); // E = P0/(gam-1) + P0*(random # between -0.5 and +0.5)*(amplitude of pressure perturbation)/(gam-1) + (1/2)*(momentum^2)/rho0
            
                     #if MAGNETIC_FIELDS_ENABLED
                     pfield->b.x1f(k,j,i) = b0 * cos(angle);
                     if (i == ie) pfield->b.x1f(k,j,i+1) = b0 * cos(angle);
                     
                     pfield->b.x2f(k,j,i) = b0 * sin(angle);
                     if (j==je) pfield->b.x2f(k,j+1,i) = b0 * sin(angle);
                     
                     pfield->b.x3f(k,j,i) = 0.0;
                     if (k==ke) pfield->b.x3f(k+1,j,i) = 0.0;
            
                     phydro->u(IEN,k,j,i) += 0.5*(SQR(b0 * cos(angle))
                     + SQR(b0 * sin(angle)) + SQR(0.0));
                     #endif /* MHD */
                    
        } // i for loop
        } // j for loop
        } // k for loop
        delta.DeleteAthenaArray();
    } // if iprob==3

return;
} // ProblemGenerator

//========================================================================================
//! utility functions
//========================================================================================
        
//----------------------------------------------------------------------------------------
// Function to calculate the timestep required to resolve cooling (and later conduction)
//          tcool = 5/2 P/Edot_cool
//          tcond = (dx)^2 / kappa (?)
// Inputs:
//   pmb: pointer to MeshBlock
/*
Real cooling_timestep(MeshBlock *pmb)
{
    Real min_dt=1.0e10;
    
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
            for (int i=pmb->is; i<=pmb->ie; ++i) {
                Real dt;
                Real edot = fabs(edot_cool(k,j,i));
                Real press = pmb->phydro->w(IPR,k,j,i);
                dt = cfl_cool * 1.5*press/edot;
                dt = std::max( dt , dt_cutoff );
                min_dt = std::min(min_dt, dt);
            }
        }
    }
    edot_cool.DeleteAthenaArray();
    return min_dt;
}*/

//----------------------------------------------------------------------------------------
//! \fn void cooling()
//  \brief  Analytic fit to cooling in the diffuse ISM given by eq. (4) in
//   Koyama & Inutsuka, ApJ 564, L97 (2002)

void cooling(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
    Real HeatRate = 2.0e-26;
    Real Tmin = 10;
    Real n,coolrate=0.0;
    Real T,coolratepp,MaxdT,dT;
    Real Teq, logn, lognT;
    MeshBlock *pblock = pmb;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
    for (int i=pmb->is; i<=pmb->ie; ++i) {
    
    const Real &dens = prim(IDN,k,j,i);
    const Real &Press = prim(IPR,k,j,i);
        //std::cout << i << ' ' << Press << std::endl;
    /* Compute number density and Temperature */
    n = dens/mbar;
    logn = log10(n);
    T = fmax((Press/ (kb * n)),Tmin);
        //std::cout << T << std::endl;
    pblock->user_out_var(1,k,j,i) = T;
    /* Compute the minimun Temperature*/
    Teq = Tmin;
    
    /* KI cooling rate per particle */
    coolratepp = HeatRate * (n*(1.0e7*exp(-1.184e5/(T+1000.)) + 0.014*sqrt(T)*exp(-92.0/T)) - 1.0);
    
    /* Expected dT by KI cooling rate */
    dT = coolratepp*dt*(pmb->peos->GetGamma() - 1.0)/kb;
    
    if ((T-dT) <= 185.0){
        lognT = 3.9247499 - 1.8479378*logn + 1.5335032*logn*logn
        -0.47665872*pow(logn,3) + 0.076789136*pow(logn,4)-0.0049052587*pow(logn,5);
        Teq = pow(10.0,lognT) / n;
    }
    
    /* Compute maximum change in T allowed to keep T positive, and limit cooling
     * rate to this value */
    MaxdT = kb*(T-Teq)/(dt*(pmb->peos->GetGamma() - 1.0));
    coolrate = fmin(coolratepp,MaxdT);
        //std::cout << coolrate << std::endl;
        
    cons(IEN,k,j,i) -= dt*n*coolratepp;
    pblock->user_out_var(0,k,j,i) = coolratepp;
    }
    }
    }
    return;
}

//----------------------------------------------------------------------------------------
//! \fn MyTimeStep()
//  \brief  Custom timestep
        
Real MyTimeStep(MeshBlock *pmb)
{
    Real min_dt=FLT_MAX;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
            for (int i=pmb->is; i<=pmb->ie; ++i) {
                Real dt;
                dt = 1.0e+9; // calculate your own time step here
                min_dt = std::min(min_dt, dt);
            }
        }
    }
    return min_dt;
}

//----------------------------------------------------------------------------------------
//! \fn parker_conductivity()
//  \brief  Calculate the Parker conduction coefficient
        
static Real parker_conductivity(const AthenaArray<Real> &prim, int k,
      int j, int i) {
    Real T;
    T = prim(IPR,k,j,i)/prim(IDN,k,j,i);
    Real kappa = 2.5e3*std::sqrt(T);
    return kappa;
}

void ParkerConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                       const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) {
    for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
                Real kappa = parker_conductivity(prim, k, j, i);
                
                Real kappa_eff;
                if (saturation_on) {
                    Real dTdx = (prim(IPR,k,j,i+1)/prim(IDN,k,j,i+1) - prim(IPR,k,j,i-1)/
                                 prim(IDN,k,j,i-1))/(pmb->pcoord->dx1v(i-1) + pmb->pcoord->dx1v(i));
                    Real q_sat = 1.5*std::pow(prim(IPR,k,j,i), 1.5)/std::sqrt(prim(IDN,k,j,i));
                    kappa_eff = 1/( std::abs(dTdx/q_sat) + 1/kappa );
                } else {
                    kappa_eff = kappa;
                }
                
                phdif->kappa(ISO,k,j,i) = kappa_eff; //  dimensions [mass]/([length]*[time])
            }
        }
    }
    return;
}
/*
void PrandtlViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                       const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) {
    for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
                Real gam = pmb->peos->GetGamma();
                Real gam1 = gam - 1;
                phdif->nu(ISO,k,j,i) = phdif->Pr*gam1*phdif->kappa(ISO,k,j,i)/(gam);
            }
        }
    }
    return;
}*/

//========================================================================================
//! user output variables
//========================================================================================

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
    for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
    for(int i=is; i<=ie; i++) {
                user_out_var(0,k,j,i) = phydro->u(IDN,k,j,i); // density
                user_out_var(1,k,j,i) = phydro->u(IDN,k,j,i)/mbar; // number density
                user_out_var(2,k,j,i) = phydro->w(IPR,k,j,i); // pressure
                user_out_var(3,k,j,i) = phydro->w(IPR,k,j,i)/(kb * (phydro->u(IDN,k,j,i)/mbar)); // temperature
                user_out_var(4,k,j,i) = phydro->w(IVX,k,j,i); // x velocity
                user_out_var(5,k,j,i) = phydro->w(IVY,k,j,i); // y velocity
                user_out_var(6,k,j,i) = phydro->w(IVZ,k,j,i); // z velocity
    }
    }
    }
}
