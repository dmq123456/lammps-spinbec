/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(spin/sib_npt,FixNVESpinSIBNPT);
// clang-format on
#else

#ifndef LMP_FIX_NVE_SPIN_SIB_NPT_H
#define LMP_FIX_NVE_SPIN_SIB_NPT_H

#include "fix_npt.h"

namespace LAMMPS_NS {

class FixNVESpinSIBNPT : public FixNPT {
  friend class PairSpin;

 public:
  FixNVESpinSIBNPT(class LAMMPS *, int, char **);
  virtual ~FixNVESpinSIBNPT();
  virtual void init();
  void reset_dt();

  void ComputeInteractionsSpin(int);    // compute and advance single spin functions
  void as_vec(const double [3], const double [3], const double, double [3]);
  void bs_vec(const double [3], const double [3], const double, const double, double [3], const double);
  void rand_cut(const double, double [3]);
  void vec2mat(const double [3], double[3][3]);
  void invert_mat(const double[3][3], double[3][3]);

 protected:
  void nve_x();
  double hbar;
  double alpha, C_spin;  // transverse mag. damping
  double dts;           // magnetic timestep
  double temp;          // spin bath temperature
  double D;      // bath intensity var.
  double gil_factor;    // gilbert's prefactor
  class RanMars *random;
  int seed;

  int nlocal_max;    // max value of nlocal (for size of lists)

  int pair_spin_flag;          // magnetic pair flags
  int precession_spin_flag;    // magnetic precession flags
  int setforce_spin_flag;

  // pointers to magnetic pair styles

  int npairs, npairspin;    // # of pairs, and # of spin pairs
  class Pair *pair;
  class PairSpin **spin_pairs;    // vector of spin pairs

  // pointers to fix setforce/spin styles

  int nsetspin;
  class FixSetForceSpin *locksetforcespin;    // to be done

  // pointers to fix precession/spin styles

  int nprecspin;
  class FixPrecessionSpin **lockprecessionspin;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal fix NVE/spin command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Pair spin requires atom attribute spin

An atom/spin style with this attribute is needed.

*/
