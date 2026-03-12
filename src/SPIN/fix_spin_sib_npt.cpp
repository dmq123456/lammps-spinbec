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

/* ------------------------------------------------------------------------
   Contributing authors: Julien Tranchida (SNL)
                         Aidan Thompson (SNL)

   Please cite the related publication:
   Tranchida, J., Plimpton, S. J., Thibaudeau, P., & Thompson, A. P. (2018).
   Massively parallel symplectic algorithm for coupled magnetic spin dynamics
   and molecular dynamics. Journal of Computational Physics.
------------------------------------------------------------------------- */

#include "fix_spin_sib_npt.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_langevin_spin.h"
#include "fix_precession_spin.h"
#include "fix_setforce_spin.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "random_mars.h"
// #include "pair_hybrid.h"
#include "pair_spin.h"
#include "update.h"
#include "timer.h"
#include "integrate.h"
#include "pair.h"

#include <cstring>
#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_nve_spin[] =
  "fix NVE/spin_sib command:\n\n"
  "@article{tranchida2018massively,\n"
  "title={Massively parallel symplectic algorithm for coupled magnetic spin "
  "dynamics and molecular dynamics},\n"
  "author={Tranchida, J and Plimpton, SJ and Thibaudeau, P and Thompson, AP},\n"
  "journal={Journal of Computational Physics},\n"
  "volume={372},\n"
  "pages={406-425},\n"
  "year={2018},\n"
  "publisher={Elsevier}\n"
  "doi={10.1016/j.jcp.2018.06.042}\n"
  "}\n\n";

enum{NONE};

/* ---------------------------------------------------------------------- */

FixNVESpinSIBNPT::FixNVESpinSIBNPT(LAMMPS *lmp, int narg, char **arg) :
  FixNPT(lmp, narg, arg),
  pair(nullptr), spin_pairs(nullptr),
  locksetforcespin(nullptr), lockprecessionspin(nullptr)
{
  if (lmp->citeme) lmp->citeme->add(cite_fix_nve_spin);

  int iarg = 3;
  alpha = -1.0;
  seed  = -1;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"temp") == 0) {
      temp = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    }
    if (strcmp(arg[iarg],"alpha") == 0) {
      alpha = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    }
    if (strcmp(arg[iarg],"seed") == 0) {
      seed = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
    }
    iarg++;
  }

  if (alpha < 0.0) {
    error->all(FLERR,"Illegal langevin/spin command");
  }

  if (temp < 0.0) {
    error->all(FLERR,"Illegal langevin/spin command");
  }

  if (seed < 0) {
    error->all(FLERR,"Illegal langevin/spin command");
  }

  // example: fix 1 all spin/sib ${myTemp} ${Tdamp_spin} 5252

  // initialize Marsaglia RNG with processor-unique seed

  // random = new RanPark(lmp,seed + comm->me);
  random = new RanMars(lmp,seed + comm->me);

  time_integrate = 1;
  nlocal_max = 0;
  npairs = 0;
  npairspin = 0;

  // test nprec
  nprecspin = nsetspin = 0;

  // checking if map array or hash is defined

  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR,"Fix NVE/spin_sib requires an atom map, see atom_modify");

  // check if the atom/spin style is defined

  if (!atom->sp_flag)
    error->all(FLERR,"Fix NVE/spin_sib requires atom/spin style");

  // initialize the magnetic interaction flags

  pair_spin_flag = 0;
  precession_spin_flag = 0;
  setforce_spin_flag = 0;
}

/* ---------------------------------------------------------------------- */

FixNVESpinSIBNPT::~FixNVESpinSIBNPT()
{
  delete [] spin_pairs;
  delete [] lockprecessionspin;
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIBNPT::init()
{

  // set timesteps
  npairs = npairspin = 0;

  // set ptrs on Pair/Spin styles

  // loop 1: obtain # of Pairs, and # of Pair/Spin styles

  npairs = 1;
  npairspin = 1;

  // init length of vector of ptrs to Pair/Spin styles

  if (npairspin > 0) {
    spin_pairs = new PairSpin*[npairspin];
  }

  // loop 2: fill vector with ptrs to Pair/Spin styles

  int count1 = 0;
  if (npairspin == 1) {
    count1 = 1;
    spin_pairs[0] = (PairSpin *) force->pair_match("^spin",0,0);
  } else if (npairspin > 1) {
    for (int i = 0; i<npairs; i++) {
      if (force->pair_match("^spin",0,i)) {
        spin_pairs[count1] = (PairSpin *) force->pair_match("^spin",0,i);
        count1++;
      }
    }
  }

  if (count1 != npairspin)
    error->all(FLERR,"Incorrect number of spin pair styles");

  // set pair/spin and long/spin flags

  if (npairspin >= 1) pair_spin_flag = 1;

  // set ptrs for fix precession/spin styles

  // loop 1: obtain # of fix precession/spin styles

  int iforce;
  nprecspin = 0;
  for (iforce = 0; iforce < modify->nfix; iforce++) {
    if (utils::strmatch(modify->fix[iforce]->style,"^precession/spin")) {
      nprecspin++;
    }
  }

  // init length of vector of ptrs to precession/spin styles

  if (nprecspin > 0) {
    lockprecessionspin = new FixPrecessionSpin*[nprecspin];
  }

  // loop 2: fill vector with ptrs to precession/spin styles

  int count2 = 0;
  if (nprecspin > 0) {
    for (iforce = 0; iforce < modify->nfix; iforce++) {
      if (utils::strmatch(modify->fix[iforce]->style,"^precession/spin")) {
        precession_spin_flag = 1;
        lockprecessionspin[count2] = (FixPrecessionSpin *) modify->fix[iforce];
        count2++;
      }
    }
  }

  if (count2 != nprecspin)
    error->all(FLERR,"Incorrect number of precession/spin fixes");

  // ptrs FixSetForceSpin classes

  for (iforce = 0; iforce < modify->nfix; iforce++) {
    if (utils::strmatch(modify->fix[iforce]->style,"^setforce/spin")) {
      setforce_spin_flag = 1;
      locksetforcespin = (FixSetForceSpin *) modify->fix[iforce];
    }
  }

  nlocal_max = atom->nlocal;

  // LangevinSpin init
  gil_factor = 1.0/(1.0+(alpha)*(alpha));

  hbar = force->hplanck/LAMMPS_NS::MathConst::MY_2PI;  // eV/(rad.THz)
  double kb = force->boltz;             // eV/K
  

  D = (alpha*gil_factor*kb*temp); // eV
  C_spin = sqrt(2.0*D);
  dts = 2 * update->dt * gil_factor / hbar ;//

  FixNPT::init();
}

/* ---------------------------------------------------------------------- */

void FixNVESpinSIBNPT::nve_x()
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double dtspin;
  double Imat[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},{0.0, 0.0, 1.0}};
  double Acar[3],Bcar[3],Rcar[nlocal][3],Scar[nlocal][3],S_h_car[nlocal][3],car_tmp[3],Fcar[3];
  double Amat[3][3],Bmat[3][3],Inv_mat[3][3],Pmat[3][3],Qmat[3][3];
  // Scar: store original spin
  // S_h_car: store half step spin
  // Rcar: random vector

  // collect spin forces with potential, external magnetic field first
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      ComputeInteractionsSpin(i);
    }
  }
  
  double **fm = atom->fm;
  double **sp = atom->sp;

  // update s for all atoms with SIB

  // step 1: get S_h_car
  // TODO: Parallelize
  for (int i = 0; i < nlocal; i++) {
    if ((sp[i][3]>0) and (mask[i] & groupbit)){
      for (int j = 0; j < 3; j++) {
        Fcar[j] = fm[i][j]*hbar;
        Scar[i][j] = sp[i][j]; // Scar store original spin
        Rcar[i][j] = random->gaussian();
      }
      dtspin = dts / sp[i][3];
      // rand_cut(dtspin, Rcar[i]);
      as_vec(Scar[i],Fcar,alpha,Acar);
      vec2mat(Acar,Amat);
      bs_vec(Scar[i],Rcar[i],alpha,C_spin,Bcar,sp[i][3]);
      vec2mat(Bcar,Bmat);

      for (int j1 = 0; j1 < 3; j1++) {
        for (int j2 = 0; j2 < 3; j2++) {
          Pmat[j1][j2] = Imat[j1][j2] - 0.5 * dtspin * Amat[j1][j2] - 0.5 * sqrt(dtspin) * Bmat[j1][j2];
          Qmat[j1][j2] = Imat[j1][j2] + 0.5 * dtspin * Amat[j1][j2] + 0.5 * sqrt(dtspin) * Bmat[j1][j2];
        }
      }
      invert_mat(Pmat, Inv_mat);
      MathExtra::matvec(Qmat,Scar[i],car_tmp);
      MathExtra::matvec(Inv_mat,car_tmp,S_h_car[i]);
      
      // update spin with half step
      for (int j = 0; j < 3; j++) {
        sp[i][j] = (Scar[i][j] + S_h_car[i][j]) * 0.5;
      }
      MathExtra::norm3(sp[i]);
    } 
  }

  // force update with half step spin.
  force->pair->compute(0,0);
  // collect spin forces with potential, external magnetic field first
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      ComputeInteractionsSpin(i);
    }
  }

  // step 2: get final spin
  for (int i = 0; i < nlocal; i++) {
    if ((sp[i][3]>0) and (mask[i] & groupbit)){
      for (int j = 0; j < 3; j++) {
        Fcar[j] = fm[i][j]*hbar; // update fm from half step spin
        S_h_car[i][j] = sp[i][j]; // half step spin as input
      }
      dtspin = dts / sp[i][3];
      as_vec(S_h_car[i],Fcar,alpha,Acar);
      vec2mat(Acar,Amat);
      bs_vec(S_h_car[i],Rcar[i],alpha,C_spin,Bcar,sp[i][3]);
      vec2mat(Bcar,Bmat);

      for (int j1 = 0; j1 < 3; j1++) {
        for (int j2 = 0; j2 < 3; j2++) {
          Pmat[j1][j2] = Imat[j1][j2] - 0.5 * dtspin * Amat[j1][j2] - 0.5 * sqrt(dtspin) * Bmat[j1][j2];
          Qmat[j1][j2] = Imat[j1][j2] + 0.5 * dtspin * Amat[j1][j2] + 0.5 * sqrt(dtspin) * Bmat[j1][j2];
        }
      }
      invert_mat(Pmat, Inv_mat);
      MathExtra::matvec(Qmat,Scar[i],car_tmp);
      MathExtra::matvec(Inv_mat,car_tmp,S_h_car[i]);
      
      // update spin with half step
      for (int j = 0; j < 3; j++) {
        sp[i][j] = S_h_car[i][j];
      }
      MathExtra::norm3(sp[i]);
    }
  }
  FixNPT::nve_x();
}

/* ----------------------------------------------------------------------
   compute the magnetic torque for the spin ii
---------------------------------------------------------------------- */

void FixNVESpinSIBNPT::ComputeInteractionsSpin(int i)
{
  double spi[3], fmi[3];

  double **sp = atom->sp;
  double **fm = atom->fm;

  // force computation for spin i

  spi[0] = sp[i][0];
  spi[1] = sp[i][1];
  spi[2] = sp[i][2];

  fmi[0] = fmi[1] = fmi[2] = 0.0;

  // update magnetic pair interactions

  if (pair_spin_flag) {
    for (int k = 0; k < npairspin; k++) {
      spin_pairs[k]->compute_single_pair(i,fmi);
      // fmi[0] += fm[i][0];
      // fmi[1] += fm[i][1];
      // fmi[2] += fm[i][2];
    }
  }

  // update magnetic precession interactions

  if (precession_spin_flag) {
    for (int k = 0; k < nprecspin; k++) {
      lockprecessionspin[k]->compute_single_precession(i,spi,fmi);
    }
  }

  // Skip langevin damping
  // update langevin damping and random force

  // if (maglangevin_flag) {               // mag. langevin
  //   for (int k = 0; k < nlangspin; k++) {
  //     locklangevinspin[k]->compute_single_langevin(i,spi,fmi);
  //   }
  // }

  // update setforce of magnetic interactions

  if (setforce_spin_flag) {
    locksetforcespin->single_setforce_spin(i,fmi);
  }

  // replace the magnetic force fm[i] by its new value fmi

  fm[i][0] = fmi[0];
  fm[i][1] = fmi[1];
  fm[i][2] = fmi[2];
}

void FixNVESpinSIBNPT::reset_dt()
{
  FixNPT::reset_dt();
  double hbar = force->hplanck/LAMMPS_NS::MathConst::MY_2PI;  // eV/(rad.THz)
  dts = 2 * update->dt * gil_factor / hbar;
}

void FixNVESpinSIBNPT::as_vec(const double Scar[3], const double Fcar[3], const double alpha, double Acar[3])
{
  double Tcar[3];

  MathExtra::cross3(Scar, Fcar, Tcar);
  for (int i = 0; i < 3; i++){
    Acar[i] = -Fcar[i] - alpha * Tcar[i]; // +?
  }
}

void FixNVESpinSIBNPT::bs_vec(const double Scar[3], const double Rcar[3], const double alpha, const double C_spin, double Bcar[3], const double spin_norm)
{
  double Tcar[3];

  MathExtra::cross3(Scar, Rcar, Tcar);
  for (int i = 0; i < 3; i++){
    Bcar[i] = C_spin * (-Rcar[i] - alpha * Tcar[i]); // sqrt(spin_norm);
  }
}

void FixNVESpinSIBNPT::rand_cut(const double delta_t, double R[3])
{
  double A_h;
  A_h = sqrt(2.0 * fabs(log(delta_t)));
  for (int i = 0; i < 3; i++){
    if(R[i]>A_h){
      R[i]=A_h;
    }else if (R[i]<-A_h)
    {
      R[i]=-A_h;
    }
  }
}

void FixNVESpinSIBNPT::vec2mat(const double vec[3], double mat[3][3])
{
  mat[0][0]=0.0;
  mat[0][1]=vec[2];
  mat[0][2]=-vec[1];
  mat[1][0]=-vec[2];
  mat[1][1]=0.0;
  mat[1][2]=vec[0];
  mat[2][0]=vec[1];
  mat[2][1]=-vec[0];
  mat[2][2]=0.0;
}

void FixNVESpinSIBNPT::invert_mat(const double mat[3][3], double mat_inv[3][3])
{
  // From lammps/src/RIGID/fix_rattle.cpp::void FixRattle::solve3x3exactly
  double determ, determinv;

  // calculate the determinant of the matrix

  determ = mat[0][0]*mat[1][1]*mat[2][2] + mat[0][1]*mat[1][2]*mat[2][0] +
    mat[0][2]*mat[1][0]*mat[2][1] - mat[0][0]*mat[1][2]*mat[2][1] -
    mat[0][1]*mat[1][0]*mat[2][2] - mat[0][2]*mat[1][1]*mat[2][0];

  // check if matrix is actually invertible

  if (determ == 0.0) error->one(FLERR,"Rattle determinant = 0.0");

  // calculate the inverse 3x3 matrix: A^(-1) = (ai_jk)

  determinv = 1.0/determ;
  mat_inv[0][0] =  determinv * (mat[1][1]*mat[2][2] - mat[1][2]*mat[2][1]);
  mat_inv[0][1] = -determinv * (mat[0][1]*mat[2][2] - mat[0][2]*mat[2][1]);
  mat_inv[0][2] =  determinv * (mat[0][1]*mat[1][2] - mat[0][2]*mat[1][1]);
  mat_inv[1][0] = -determinv * (mat[1][0]*mat[2][2] - mat[1][2]*mat[2][0]);
  mat_inv[1][1] =  determinv * (mat[0][0]*mat[2][2] - mat[0][2]*mat[2][0]);
  mat_inv[1][2] = -determinv * (mat[0][0]*mat[1][2] - mat[0][2]*mat[1][0]);
  mat_inv[2][0] =  determinv * (mat[1][0]*mat[2][1] - mat[1][1]*mat[2][0]);
  mat_inv[2][1] = -determinv * (mat[0][0]*mat[2][1] - mat[0][1]*mat[2][0]);
  mat_inv[2][2] =  determinv * (mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]);
}
