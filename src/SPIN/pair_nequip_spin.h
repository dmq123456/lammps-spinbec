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

PairStyle(spin/nequip_spin, PairNEQUIP_spin)

#else

#ifndef LMP_PAIR_NEQUIP_SPIN_H
#define LMP_PAIR_NEQUIP_SPIN_H

#include "pair_spin.h"

#include <torch/torch.h>

namespace LAMMPS_NS
{

  class PairNEQUIP_spin : public PairSpin
  {
  public:
    PairNEQUIP_spin(class LAMMPS *);
    virtual ~PairNEQUIP_spin();
    void *extract(const char *, int &);
    void compute(int, int);
    void compute_single_pair(int, double *);
    void settings(int, char **);
    void coeff(int, char **);
    double init_one(int, int);
    void init_style();
    void allocate();

    double cutoff;
    torch::jit::Module model;
    torch::Device device = torch::kCPU;

  protected:
    int *type_mapper;
    int debug_mode = 0;
  };

}

#endif
#endif
