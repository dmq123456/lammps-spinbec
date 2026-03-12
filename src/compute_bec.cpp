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

#include "compute_bec.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

enum { MASSCENTER, GEOMCENTER };

/* ---------------------------------------------------------------------- */

ComputeBEC::ComputeBEC(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{
  if ((narg < 3) || (narg > 3)) error->all(FLERR, "Illegal compute dipole command");

  scalar_flag = 1;
  vector_flag = 1;
  array_flag = 1;
  size_vector = 4;
  extscalar = 0;
  extvector = 0;

  vector = new double[size_vector];
  vector[0] = vector[1] = vector[2] = vector[3] = 0.0;

}

/* ---------------------------------------------------------------------- */

ComputeBEC::~ComputeBEC()
{
  delete[] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeBEC::compute_vector()
{
  invoked_vector = update->ntimestep;

  const auto x = atom->x;
  const auto mask = atom->mask;
  const auto image = atom->image;
  const auto nlocal = atom->nlocal;

  double dipole[3] = {0.0, 0.0, 0.0};
  double dipole_mpi[3] = {0.0, 0.0, 0.0};
  double bec_all[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double bec_all_mpi[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int nlocal_mpi = 0;

  inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
  // compute dipole moment

  if (atom->bec_flag) {
    const auto bec_dipole = atom->bec_dipole;
    const auto bec = atom->bec;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
          double unwrap[3];
          domain->unmap(x[i], image[i], unwrap);
          dipole[0] += bec_dipole[i][0];
          dipole[1] += bec_dipole[i][1];
          dipole[2] += bec_dipole[i][2];
          
          bec_all[0] += bec[i][0];
          bec_all[1] += bec[i][1];
          bec_all[2] += bec[i][2];
          bec_all[3] += bec[i][3];
          bec_all[4] += bec[i][4];
          bec_all[5] += bec[i][5];
          bec_all[6] += bec[i][6];
          bec_all[7] += bec[i][7];
          bec_all[8] += bec[i][8];
      }
    }
  }

  MPI_Allreduce(dipole, dipole_mpi, 3, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(bec_all, bec_all_mpi, 9, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&nlocal, &nlocal_mpi, 1, MPI_INT, MPI_SUM, world);
  vector[0] = dipole_mpi[0];
  vector[1] = dipole_mpi[1];
  vector[2] = dipole_mpi[2];
  vector[3] = 0.0;
  for (int i = 0; i < 9; i++) {
    vector[3] += bec_all_mpi[i] * bec_all_mpi[i];
  }
  vector[3] = sqrt(vector[3]) / nlocal_mpi / 9;
}

/* ---------------------------------------------------------------------- */

double ComputeBEC::compute_scalar()
{
  if (invoked_vector != update->ntimestep) compute_vector();

  invoked_scalar = update->ntimestep;
  scalar = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
  return scalar;
}
