//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartesian.cpp
//  \brief implements functions for Cartesian (x-y-z) coordinates in a derived class of
//  the Coordinates abstract base class.

// Athena++ headers
#include "coordinates.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
// Cartesian coordinates constructor

Cartesian::Cartesian(MeshBlock *pmb, ParameterInput *pin, bool flag)
  : Coordinates(pmb, pin, flag) {
  pmy_block = pmb;
  coarse_flag=flag;
  int il, iu, jl, ju, kl, ku, ng;
  if (coarse_flag==true) {
    il = pmb->cis; jl = pmb->cjs; kl = pmb->cks;
    iu = pmb->cie; ju = pmb->cje; ku = pmb->cke;
    ng=pmb->cnghost;
  } else {
    il = pmb->is; jl = pmb->js; kl = pmb->ks;
    iu = pmb->ie; ju = pmb->je; ku = pmb->ke;
    ng=NGHOST;
  }
  Mesh *pm=pmy_block->pmy_mesh;
  RegionSize& block_size = pmy_block->block_size;

  // allocate arrays for volume-centered coordinates and positions of cells
  int ncells1 = (iu-il+1) + 2*ng;
  int ncells2 = 1, ncells3 = 1;
  if (block_size.nx2 > 1) ncells2 = (ju-jl+1) + 2*ng;
  if (block_size.nx3 > 1) ncells3 = (ku-kl+1) + 2*ng;
  dx1v.NewAthenaArray(ncells1);
  dx2v.NewAthenaArray(ncells2);
  dx3v.NewAthenaArray(ncells3);
  x1v.NewAthenaArray(ncells1);
  x2v.NewAthenaArray(ncells2);
  x3v.NewAthenaArray(ncells3);
  // allocate arrays for volume- and face-centered geometry coefficients of cells
  h2f.NewAthenaArray(ncells1);
  dh2fd1.NewAthenaArray(ncells1);
  h31f.NewAthenaArray(ncells1);
  dh31fd1.NewAthenaArray(ncells1);
  h32f.NewAthenaArray(ncells2);
  dh32fd2.NewAthenaArray(ncells2);
  h2v.NewAthenaArray(ncells1);
  dh2vd1.NewAthenaArray(ncells1);
  h31v.NewAthenaArray(ncells1);
  dh31vd1.NewAthenaArray(ncells1);
  h32v.NewAthenaArray(ncells2);
  dh32vd2.NewAthenaArray(ncells2);

  // allocate arrays for area weighted positions for AMR/SMR MHD
  if ((pm->multilevel==true) && MAGNETIC_FIELDS_ENABLED) {
    x1s2.NewAthenaArray(ncells1);
    x1s3.NewAthenaArray(ncells1);
    x2s1.NewAthenaArray(ncells2);
    x2s3.NewAthenaArray(ncells2);
    x3s1.NewAthenaArray(ncells3);
    x3s2.NewAthenaArray(ncells3);
  }

  // initialize volume-averaged coordinates and spacing
  // x1-direction: x1v = dx/2
  for (int i=il-ng; i<=iu+ng; ++i) {
    x1v(i) = 0.5*(x1f(i+1) + x1f(i));
  }
  for (int i=il-ng; i<=iu+ng-1; ++i) {
    if (pmb->block_size.x1rat != 1.0) {
      dx1v(i) = x1v(i+1) - x1v(i);
    } else {
      // dx1v = dx1f constant for uniform mesh; may disagree with x1v(i+1) - x1v(i)
      dx1v(i) = dx1f(i);
    }
  }

  // x2-direction: x2v = dy/2
  if (pmb->block_size.nx2 == 1) {
    x2v(jl) = 0.5*(x2f(jl+1) + x2f(jl));
    dx2v(jl) = dx2f(jl);
  } else {
    for (int j=jl-ng; j<=ju+ng; ++j) {
      x2v(j) = 0.5*(x2f(j+1) + x2f(j));
    }
    for (int j=jl-ng; j<=ju+ng-1; ++j) {
      if (pmb->block_size.x2rat != 1.0) {
        dx2v(j) = x2v(j+1) - x2v(j);
      } else {
        // dx2v = dx2f constant for uniform mesh; may disagree with x2v(j+1) - x2v(j)
        dx2v(j) = dx2f(j);
      }
    }
  }

  // x3-direction: x3v = dz/2
  if (pmb->block_size.nx3 == 1) {
    x3v(kl) = 0.5*(x3f(kl+1) + x3f(kl));
    dx3v(kl) = dx3f(kl);
  } else {
    for (int k=kl-ng; k<=ku+ng; ++k) {
      x3v(k) = 0.5*(x3f(k+1) + x3f(k));
    }
    for (int k=kl-ng; k<=ku+ng-1; ++k) {
      if (pmb->block_size.x3rat != 1.0) {
        dx3v(k) = x3v(k+1) - x3v(k);
      } else {
        // dxkv = dx3f constant for uniform mesh; may disagree with x3v(k+1) - x3v(k)
        dx3v(k) = dx3f(k);
      }
    }
  }
  // initialize geometry coefficients
  // x1-direction
  for (int i=il-ng; i<=iu+ng; ++i) {
    h2v(i) = 1.0;
    h2f(i) = 1.0;
    h31v(i) = 1.0;
    h31f(i) = 1.0;
    dh2vd1(i) = 0.0;
    dh2fd1(i) = 0.0;
    dh31vd1(i) = 0.0;
    dh31fd1(i) = 0.0;
  }

  // x2-direction
  if (pmb->block_size.nx2 == 1) {
    h32v(jl) = 1.0;
    h32f(jl) = 1.0;
    dh32vd2(jl) = 0.0;
    dh32fd2(jl) = 0.0;
  } else {
    for (int j=jl-ng; j<=ju+ng; ++j) {
      h32v(j) = 1.0;
      h32f(j) = 1.0;
      dh32vd2(j) = 0.0;
      dh32fd2(j) = 0.0;
    }
  }

  // initialize area-averaged coordinates used with MHD AMR
  if ((pmb->pmy_mesh->multilevel==true) && MAGNETIC_FIELDS_ENABLED) {
    for (int i=il-ng; i<=iu+ng; ++i) {
      x1s2(i) = x1s3(i) = x1v(i);
    }
    if (pmb->block_size.nx2 == 1) {
      x2s1(jl) = x2s3(jl) = x2v(jl);
    } else {
      for (int j=jl-ng; j<=ju+ng; ++j) {
        x2s1(j) = x2s3(j) = x2v(j);
      }
    }
    if (pmb->block_size.nx3 == 1) {
      x3s1(kl) = x3s2(kl) = x3v(kl);
    } else {
      for (int k=kl-ng; k<=ku+ng; ++k) {
        x3s1(k) = x3s2(k) = x3v(k);
      }
    }
  }
}

// destructor

Cartesian::~Cartesian() {
  dx1v.DeleteAthenaArray();
  dx2v.DeleteAthenaArray();
  dx3v.DeleteAthenaArray();
  x1v.DeleteAthenaArray();
  x2v.DeleteAthenaArray();
  x3v.DeleteAthenaArray();
  if ((pmy_block->pmy_mesh->multilevel==true) && MAGNETIC_FIELDS_ENABLED) {
    x1s2.DeleteAthenaArray();
    x1s3.DeleteAthenaArray();
    x2s1.DeleteAthenaArray();
    x2s3.DeleteAthenaArray();
    x3s1.DeleteAthenaArray();
    x3s2.DeleteAthenaArray();
  }
}
