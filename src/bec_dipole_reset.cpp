// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "bec_dipole_reset.h"

#include "error.h"
#include "input.h"
#include "pair.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"


// #include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BecDipoleReset::BecDipoleReset(LAMMPS *lmp) : Command(lmp) {}

/* ---------------------------------------------------------------------- */

void BecDipoleReset::command(int narg, char **arg)
{
  if (narg > 0) error->all(FLERR,"Illegal BecDipoleReset command");
  if (atom->bec_flag){
    double **bec_dipole = atom->bec_dipole;
    int nlocal = atom->nlocal;
    if (comm->me == 0) {
      printf("\n");
      printf("--------------------------------\n");
      printf("Reset Dipole from BEC to zero\n");
      printf("Dipole below treat positions right now as the standard\n");
      printf("--------------------------------\n");
      printf("\n");
    }
    for (int i = 0; i < nlocal; i++) {
        bec_dipole[i][0] = 0.0;
        bec_dipole[i][1] = 0.0;
        bec_dipole[i][2] = 0.0;
    }
  }
}

