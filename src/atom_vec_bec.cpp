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

#include "atom_vec_bec.h"

#include "atom.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecBEC::AtomVecBEC(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::ATOMIC;
  mass_type = PER_TYPE;

  atom->bec_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = (char *) "bec bec_dipole";
  fields_copy = (char *) "bec_dipole";
  fields_comm = (char *) "";
  fields_comm_vel = (char *) "";
  // bec is written by the pair style for owned atoms only, so it must not be
  // reverse-communicated: ghost slots hold stale values (e.g. of atoms that
  // migrated away) which would be summed into their owners under MPI.
  fields_reverse = (char *) "";
  fields_border = (char *) "";
  fields_border_vel = (char *) "";
  fields_exchange = (char *) "bec_dipole";
  fields_restart = (char *) "bec_dipole";
  fields_create = (char *) "";
  fields_data_atom = (char *) "id type x";
  fields_data_vel = (char *) "id v";

  setup_fields();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecBEC::grow_pointers()
{
  bec = atom->bec;
  bec_dipole = atom->bec_dipole;
}

/* ----------------------------------------------------------------------
   clear extra forces starting at atom N
   nbytes = # of bytes to clear for a per-atom vector
   include f b/c this is invoked from within SPIN pair styles
------------------------------------------------------------------------- */

void AtomVecBEC::force_clear(int n, size_t nbytes)
{
  memset(&bec[n][0],0,9*nbytes);
  memset(&bec_dipole[n][0],0,3*nbytes);
}

/* ----------------------------------------------------------------------
   initialize per-atom BEC quantities of a newly created atom
   (grow() does not zero memory, so they would otherwise start as garbage)
------------------------------------------------------------------------- */

void AtomVecBEC::create_atom_post(int ilocal)
{
  for (int k = 0; k < 9; k++) bec[ilocal][k] = 0.0;
  for (int k = 0; k < 3; k++) bec_dipole[ilocal][k] = 0.0;
}

/* ----------------------------------------------------------------------
   same for an atom read from the Atoms section of a data file
------------------------------------------------------------------------- */

void AtomVecBEC::data_atom_post(int ilocal)
{
  for (int k = 0; k < 9; k++) bec[ilocal][k] = 0.0;
  for (int k = 0; k < 3; k++) bec_dipole[ilocal][k] = 0.0;
}
