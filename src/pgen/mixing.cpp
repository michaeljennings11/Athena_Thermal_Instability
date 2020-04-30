//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file kh.cpp
//  \brief Problem generator for KH instability.
//
// Sets up several different problems:
//   - iprob=1: slip surface with tanh profile

// C/C++ headers
#include <algorithm>  // min, max
#include <cmath>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/utils.hpp"

// These constants are in cgs */
static const Real kb = 1.380658e-16; // Boltzmann constant
static const Real mbar = (1.27)*(1.6733e-24); // mean molecular weight
static const Real pi = 3.14159265358979323846;

Real Thot, Tcold, nhot, ncold, vhot, shrwdth, amp, rhoh, rhoc, P0, offset, visc_factor;

void TempViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);

// BCs on L-x1 (left edge) of grid
void InnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void cooling(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

//----------------------------------------------------------------------------------------
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.

void Mesh::InitUserMeshData(ParameterInput *pin) {
  
  Thot = pin->GetReal("problem","Thot");
  Tcold = pin->GetReal("problem","Tcold");
  nhot = pin->GetReal("problem","nhot");
  ncold = pin->GetReal("problem","ncold");
  vhot = pin->GetReal("problem","vhot");
  shrwdth = pin->GetReal("problem","shrwdth");
  amp = pin->GetReal("problem","amp");
  offset = pin->GetOrAddReal("problem","offset",0.0);
  //Pr = pin->GetOrAddReal("problem","Pr",0.0); //Prandlt number
  
    
  rhoh = nhot*mbar;
  rhoc = ncold*mbar;
  P0 = nhot*kb*Thot;
    
  if (pin->GetOrAddBoolean("problem","cooling", false)){
      std::cout << "cooling enabled" << std::endl;
      EnrollUserExplicitSourceFunction(cooling);
  }
  else
      std::cout << "cooling disabled" << std::endl;
    
  if (pin->GetOrAddBoolean("problem","tempvisc_flag", false)){
        visc_factor = pin->GetOrAddReal("problem","visc_factor",1.0);
        EnrollViscosityCoefficient(TempViscosity);
  }
    
  EnrollUserBoundaryFunction(INNER_X1, InnerX1);

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
    AllocateUserOutputVariables(7);
    SetUserOutputVariableName(0, "n");
    SetUserOutputVariableName(1, "press");
    SetUserOutputVariableName(2, "Temp");
    SetUserOutputVariableName(3, "vel1");
    SetUserOutputVariableName(4, "vel2");
    SetUserOutputVariableName(5, "vel3");
    SetUserOutputVariableName(6, "rho");
    return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the Kelvin-Helmholz test

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int64_t iseed = -1 - gid;
  Real gm1 = peos->GetGamma() - 1.0;
  int iprob = pin->GetInteger("problem","iprob");

  //--- iprob=1.  Uniform stream with density ratio "drat" located in region -1/4<y<1/4
  // moving at (-vflow) seperated by two slip-surfaces from background medium with d=1
  // moving at (+vflow), random perturbations.  This is the classic, unresolved K-H test.

  if (iprob == 1) {
    

    Real dv = amp*vhot;
      
    Real x1size = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
    Real x2size = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
    Real x3size = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
      
    Real a = x1size*shrwdth/2.0;
      
    for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
    for (int i=is; i<=ie; i++) {
      phydro->u(IDN,k,j,i) = rhoh + (rhoc - rhoh)/2.0 * (1.0 + tanh((pcoord->x1v(i)-offset*x1size)/a));
      //if ((pcoord->x1v(i)) > 0)
          //phydro->u(IDN,k,j,i) = rhoc;
      phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*(dv * exp(-1.0 * ((pcoord->x1v(i)-offset*x1size)/a) * ((pcoord->x1v(i)-offset*x1size)/a)) * (sin(2.0*pi*(pcoord->x2v(j))/(x2size/1.0))+sin(2.0*pi*(pcoord->x2v(j))/(x2size/2.0))+sin(2.0*pi*(pcoord->x2v(j))/(x2size/3.0))+sin(2.0*pi*(pcoord->x2v(j))/(x2size/4.0))+sin(2.0*pi*(pcoord->x2v(j))/(x2size/5.0))+sin(2.0*pi*(pcoord->x2v(j))/(x2size/6.0))));
      phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*vhot/2.0 * (1.0 - tanh((pcoord->x1v(i)-offset*x1size)/a));
      phydro->u(IM3,k,j,i) = 0.0;
      
      phydro->u(IEN,k,j,i) = P0/gm1 + 0.5*(SQR(phydro->u(IM1,k,j,i)) +
      SQR(phydro->u(IM2,k,j,i)))/phydro->u(IDN,k,j,i);
    }}}

    // initialize uniform interface B
    if (MAGNETIC_FIELDS_ENABLED) {
      Real b0 = pin->GetReal("problem","b0");
      for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie+1; i++) {
        pfield->b.x1f(k,j,i) = b0;
      }}}
      for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x2f(k,j,i) = 0.0;
      }}}
      for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x3f(k,j,i) = 0.0;
      }}}
      if (NON_BAROTROPIC_EOS) {
        for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IEN,k,j,i) += 0.5*b0*b0;
        }}}
      }
    }
  }

  
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void InnerX1()
//  \brief Sets boundary condition on left X boundary

void InnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
        Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // set primitive variables in inlet ghost zones
  for (int k=ks; k<=ke; ++k) {
  for (int j=js; j<=je; ++j) {
    for (int i=1; i<=ngh; ++i) {
      
        prim(IDN,k,j,is-i) = rhoh;
        prim(IVX,k,j,is-i) = 0;
        prim(IVY,k,j,is-i) = vhot;
        prim(IVZ,k,j,is-i) = 0;
        prim(IPR,k,j,is-i) = P0;
      }
    }
  }}

//----------------------------------------------------------------------------------------
//! \fn void cooling()
//----------------------------------------------------------------------------------------

void cooling(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
    Real Tmin = 1.0e4;
    Real n;
    Real T,coolratepp,MaxdT,dT;
    Real Teq, logn, lognT;
    MeshBlock *pblock = pmb;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
    for (int i=pmb->is; i<=pmb->ie; ++i) {
        
    const Real &dens = prim(IDN,k,j,i);
    const Real &Press = prim(IPR,k,j,i);
    Real Lambda;
    
    /* Compute number density and Temperature */
    n = dens/mbar;

    T = fmax((Press/ (kb * n)),Tmin);
    
    /* Lambda */
    if (T <= 1.0e4)
        Lambda = 0;
    else if (T > 1.0e4 && T < 1.0e5)
        Lambda = 3.162e-30*pow(T,1.6);
    else if (T >= 1.0e5 && T < 2.884e5)
        Lambda = 3.162e-21*pow(T,-0.2);
    else if (T >= 2.884e5 && T < 4.732e5)
        Lambda = 6.31e-6*pow(T,-3.0);
    else if (T >= 4.732e5 && T < 2.113e6)
        Lambda = 1.047e-21*pow(T,-0.22);
    else if (T >= 2.113e6 && T < 3.981e6)
        Lambda = 3.981e-4*pow(T,-3.0);
    else if (T >= 3.981e6 && T < 1.995e7)
        Lambda = 4.169e-26*pow(T,0.33);
        
    coolratepp = n*Lambda;
        
    cons(IEN,k,j,i) -= dt*n*coolratepp;
    }
    }
    }
    return;
}


void TempViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) {
    
    Real T, n, Ln_c;
    
    for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
                const Real &dens = prim(IDN,k,j,i);
                const Real &Press = prim(IPR,k,j,i);
                n = dens/mbar;
                T = Press / (kb*n);
                
                if (T < 4.0e5)
                    phdif->nu(ISO,k,j,i) = 0.0;
                else if (T >= 4.0e5){
                    // Coulomb Logarithm for T > 4e5
                    Ln_c = 37.8 - log((T/1.0e8)*pow((n/1.0e-3),-0.5));
                    
                    phdif->nu(ISO,k,j,i) = visc_factor*5500.0*pow((T/1.0e8),(5.0/2.0))/(Ln_c/40.0);
                }
            }
        }
    }
    return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
    for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
    for(int i=is; i<=ie; i++) {
                user_out_var(0,k,j,i) = phydro->u(IDN,k,j,i)/mbar; // number density
                user_out_var(1,k,j,i) = phydro->w(IPR,k,j,i); // pressure
                user_out_var(2,k,j,i) = phydro->w(IPR,k,j,i)/(kb * (phydro->u(IDN,k,j,i)/mbar)); // temperature
                user_out_var(3,k,j,i) = phydro->w(IVX,k,j,i); // x velocity
                user_out_var(4,k,j,i) = phydro->w(IVY,k,j,i); // y velocity
                user_out_var(5,k,j,i) = phydro->w(IVZ,k,j,i); // z velocity
                user_out_var(6,k,j,i) = phydro->u(IDN,k,j,i); // density
    }
    }
    }
}

