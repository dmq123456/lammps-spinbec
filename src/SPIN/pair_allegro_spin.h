/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(spin/allegro_spin, PairAllegro_spin)

#else

#ifndef LMP_PAIR_ALLEGRO_SPIN_H
#define LMP_PAIR_ALLEGRO_SPIN_H

#include "pair_spin.h"

#include <torch/torch.h>
#include <vector>

namespace LAMMPS_NS {

class PairAllegro_spin : public PairSpin {
 public:
  PairAllegro_spin(class LAMMPS *);
  virtual ~PairAllegro_spin();
  void *extract(const char *, int &);
  virtual void compute(int, int);
  void compute_single_pair(int, double *);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();
  void allocate();
  // void *extract(const char *, int &);?

  double cutoff;
  torch::jit::Module model;
  torch::Device device = torch::kCPU;
  std::vector<int> type_mapper;

  int batch_size = -1;

  bool use_constant_bec;
  // Network spin field of the last compute(), owned atoms complete (ghost contributions
  // added by this pair's own reverse communication). compute_single_pair() returns it, so
  // fix nve/spin never sees what other fixes (langevin/spin, precession/spin, ...) write
  // into atom->fm, and nothing accumulates within a timestep.
  double **fm_net = nullptr;
  int nmax_fm_net = 0;

 public:
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  std::vector<double> bec_constant_per_type;

  // void write_restart(FILE *);
  // void read_restart(FILE *);
  // void write_restart_settings(FILE *);
  // void read_restart_settings(FILE *);

 protected:
  int debug_mode = 0;
};

}    // namespace LAMMPS_NS

#endif
#endif

    /* ERROR/WARNING messages:

E: Incorrect args in pair_spin command

Self-explanatory.

E: Spin simulations require metal unit style

Self-explanatory.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair spin requires atom attribute spin

The atom style defined does not have these attributes.

*/
