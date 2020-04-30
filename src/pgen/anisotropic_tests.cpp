//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file anisotropic_tests.cpp
//  \brief Problem generator for anisotropic ring test.
//
//========================================================================================

// C/C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

// problem parameters which are useful to make global to this file
static Real brad, amp;

void MagField(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
    brad = pin->GetReal("problem","brad");
    amp = pin->GetReal("problem","amp");
    
    if (pin->GetOrAddBoolean("problem","const_mag", true)){
        std::cout << "constant magnetic field enabled" << std::endl;
        EnrollUserExplicitSourceFunction(MagField);
    }
    return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
    AllocateUserOutputVariables(3);
    SetUserOutputVariableName(0, "rho");
    SetUserOutputVariableName(1, "press");
    SetUserOutputVariableName(2, "Temp");
    return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief ring test problem generator for 2D.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    Real gm1 = peos->GetGamma() - 1.0;
    Real iso_cs =peos->GetIsoSoundSpeed();
    
    AthenaArray<Real> ax,ay,az;
    int nx1 = (ie-is)+1 + 2*(NGHOST);
    int nx2 = (je-js)+1 + 2*(NGHOST);
    int nx3 = (ke-ks)+1 + 2*(NGHOST);
    ax.NewAthenaArray(nx3,nx2,nx1);
    ay.NewAthenaArray(nx3,nx2,nx1);
    az.NewAthenaArray(nx3,nx2,nx1);
    
    // Read initial conditions, diffusion coefficients (if needed)
    Real T_back = pin->GetOrAddReal("problem","T_back",10.0);
    Real T_wedge = pin->GetOrAddReal("problem","T_wedge",12.0);
    Real radl = pin->GetOrAddReal("problem","radl",0.5);
    Real radu = pin->GetOrAddReal("problem","radu",0.7);
    Real thetal = pin->GetOrAddReal("problem","thetal",(5.0/12.0)*PI);
    Real thetau = pin->GetOrAddReal("problem","thetau",(7.0/12.0)*PI);
    int iprob = pin->GetInteger("problem","iprob");
    Real P_back = T_back;
    Real P_wedge = T_wedge;
    
    Real ang_2,cos_a2,sin_a2,lambda, rad, rad_sqr, theta;
    
    // Use vector potential to initialize field loop
    // the origin of the initial loop
    Real x0 = pin->GetOrAddReal("problem","x0",0.0);
    Real y0 = pin->GetOrAddReal("problem","y0",0.0);
    Real z0 = pin->GetOrAddReal("problem","z0",0.0);
    
    for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je+1; j++) {
            for (int i=is; i<=ie+1; i++) {
                
                // (iprob=1): field loop in x1-x2 plane (cylinder in 3D) */
                if (iprob==1) {
                    ax(k,j,i) = 0.0;
                    ay(k,j,i) = 0.0;
                    if ((SQR(pcoord->x1f(i)-x0) + SQR(pcoord->x2f(j)-y0)) < brad*brad) {
                        az(k,j,i) = amp*(brad - std::sqrt(SQR(pcoord->x1f(i)-x0) +
                                                          SQR(pcoord->x2f(j)-y0)));
                    } else {
                        az(k,j,i) = 0.0;
                    }
                }
            }
        }
    }
    // Initialize density.
    
    Real x1size = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
    Real x2size = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
    Real x3size = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;
    Real diag = std::sqrt(x1size*x1size + x2size*x2size + x3size*x3size);
    for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
            for (int i=is; i<=ie; i++) {
                // calculate angle between 0 and 2pi
                theta = atan2((pcoord->x2v(j)),(pcoord->x1v(i)));
                if(theta < 0) theta += 2*PI;
                
                phydro->u(IDN,k,j,i) = 1.0;
                phydro->u(IM1,k,j,i) = 0.0;//phydro->u(IDN,k,j,i)*vflow*x1size/diag;
                phydro->u(IM2,k,j,i) = 0.0;//phydro->u(IDN,k,j,i)*vflow*x2size/diag;
                phydro->u(IM3,k,j,i) = 0.0;//phydro->u(IDN,k,j,i)*vflow*x3size/diag;
                phydro->u(IEN,k,j,i) = P_back/gm1;
                if (((SQR(pcoord->x1v(i)) + SQR(pcoord->x2v(j)) + SQR(pcoord->x3v(k))) > radl*radl) && ((SQR(pcoord->x1v(i)) + SQR(pcoord->x2v(j)) + SQR(pcoord->x3v(k))) < radu*radu) && (theta > thetal) && (theta < thetau))  {
                    phydro->u(IDN,k,j,i) = 1.0;
                    phydro->u(IM1,k,j,i) = 0.0;//phydro->u(IDN,k,j,i)*vflow*x1size/diag;
                    phydro->u(IM2,k,j,i) = 0.0;//phydro->u(IDN,k,j,i)*vflow*x2size/diag;
                    phydro->u(IM3,k,j,i) = 0.0;//phydro->u(IDN,k,j,i)*vflow*x3size/diag;
                    phydro->u(IEN,k,j,i) = P_wedge/gm1;
                }
            }}}
    
    // initialize interface B
    for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
            for (int i=is; i<=ie+1; i++) {
                pfield->b.x1f(k,j,i) = (az(k,j+1,i) - az(k,j,i))/pcoord->dx2f(j) -
                (ay(k+1,j,i) - ay(k,j,i))/pcoord->dx3f(k);
            }}}
    for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je+1; j++) {
            for (int i=is; i<=ie; i++) {
                pfield->b.x2f(k,j,i) = (ax(k+1,j,i) - ax(k,j,i))/pcoord->dx3f(k) -
                (az(k,j,i+1) - az(k,j,i))/pcoord->dx1f(i);
            }}}
    for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je; j++) {
            for (int i=is; i<=ie; i++) {
                pfield->b.x3f(k,j,i) = (ay(k,j,i+1) - ay(k,j,i))/pcoord->dx1f(i) -
                (ax(k,j+1,i) - ax(k,j,i))/pcoord->dx2f(j);
            }}}
    
    ax.DeleteAthenaArray();
    ay.DeleteAthenaArray();
    az.DeleteAthenaArray();
    
    return;
}

//========================================================================================
//! \fn void MagField(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
//  \brief constant magnetic field source function
//========================================================================================
void MagField(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons){
    AthenaArray<Real> ax,ay,az;
    int nx1 = (pmb->ie-pmb->is)+1 + 2*(NGHOST);
    int nx2 = (pmb->je-pmb->js)+1 + 2*(NGHOST);
    int nx3 = (pmb->ke-pmb->ks)+1 + 2*(NGHOST);
    ax.NewAthenaArray(nx3,nx2,nx1);
    ay.NewAthenaArray(nx3,nx2,nx1);
    az.NewAthenaArray(nx3,nx2,nx1);
    for (int k=pmb->ks; k<=pmb->ke+1; k++) {
        for (int j=pmb->js; j<=pmb->je+1; j++) {
            for (int i=pmb->is; i<=pmb->ie+1; i++) {
                ax(k,j,i) = 0.0;
                ay(k,j,i) = 0.0;
                if ((SQR(pmb->pcoord->x1f(i)) + SQR(pmb->pcoord->x2f(j))) < brad*brad) {
                    az(k,j,i) = amp*(brad - std::sqrt(SQR(pmb->pcoord->x1f(i)) +
                                                      SQR(pmb->pcoord->x2f(j))));
                } else {
                    az(k,j,i) = 0.0;
                }
            } //i
        } //j
    } //k
    for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
            for (int i=pmb->is; i<=pmb->ie+1; i++) {
                pmb->pfield->b.x1f(k,j,i) = (az(k,j+1,i) - az(k,j,i))/pmb->pcoord->dx2f(j) -
                (ay(k+1,j,i) - ay(k,j,i))/pmb->pcoord->dx3f(k);
            }}}
    for (int k=pmb->ks; k<=pmb->ke; k++) {
        for (int j=pmb->js; j<=pmb->je+1; j++) {
            for (int i=pmb->is; i<=pmb->ie; i++) {
                pmb->pfield->b.x2f(k,j,i) = (ax(k+1,j,i) - ax(k,j,i))/pmb->pcoord->dx3f(k) -
                (az(k,j,i+1) - az(k,j,i))/pmb->pcoord->dx1f(i);
            }}}
    for (int k=pmb->ks; k<=pmb->ke+1; k++) {
        for (int j=pmb->js; j<=pmb->je; j++) {
            for (int i=pmb->is; i<=pmb->ie; i++) {
                pmb->pfield->b.x3f(k,j,i) = (ay(k,j,i+1) - ay(k,j,i))/pmb->pcoord->dx1f(i) -
                (ax(k,j+1,i) - ax(k,j,i))/pmb->pcoord->dx2f(j);
            }}}
    ax.DeleteAthenaArray();
    ay.DeleteAthenaArray();
    az.DeleteAthenaArray();
}


//========================================================================================
//! user output variables
//========================================================================================

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
    Real gm1 = peos->GetGamma() - 1.0;
    for(int k=ks; k<=ke; k++) {
        for(int j=js; j<=je; j++) {
            for(int i=is; i<=ie; i++) {
                user_out_var(0,k,j,i) = phydro->u(IDN,k,j,i); // density
                user_out_var(1,k,j,i) = phydro->u(IEN,k,j,i)*gm1; // pressure
                user_out_var(2,k,j,i) = phydro->u(IEN,k,j,i)*gm1/phydro->u(IDN,k,j,i); // temperature
            }
        }
    }
}

