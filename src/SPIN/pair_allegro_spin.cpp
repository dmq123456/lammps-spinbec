/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"
#include <pair_allegro_spin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <ATen/core/Dict.h>

// TODO: Only if MPI is available
#include <mpi.h>

// We have to do a backward compatability hack for <1.10
// https://discuss.pytorch.org/t/how-to-check-libtorch-version/77709/4
// Basically, the check in torch::jit::freeze
// (see https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp#L479)
// is wrong, and we have ro "reimplement" the function
// to get around that...
// it's broken in 1.8 and 1.9
// BUT the internal logic in the function is wrong in 1.10
// So we only use torch::jit::freeze in >=1.11
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
#define DO_TORCH_FREEZE_HACK
// For the hack, need more headers:
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#endif

using namespace LAMMPS_NS;

PairAllegro_spin::PairAllegro_spin(LAMMPS *lmp) : PairSpin(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;
  comm_reverse = 3;   // fm_net: ghost contributions are summed into their owners
  centroidstressflag =
      CENTROID_AVAIL; // Allow the use of the centroid/stress/atom. - Added by Hongyu Yu

  use_constant_bec = false;

  if (const char *env_p = std::getenv("ALLEGRO_DEBUG"))
  {
    std::cout << "PairAllegro_spin is in DEBUG mode, since ALLEGRO_DEBUG is in env\n";
    debug_mode = 1;
  }

  if (torch::cuda::is_available())
  {
    int deviceidx = -1;
    if (comm->nprocs > 1)
    {
      MPI_Comm shmcomm;
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
      int shmrank;
      MPI_Comm_rank(shmcomm, &shmrank);
      deviceidx = shmrank;
    }
    if (deviceidx >= 0)
    {
      int devicecount = torch::cuda::device_count();
      if (deviceidx >= devicecount)
      {
        if (debug_mode)
        {
          // To allow testing multi-rank calls, we need to support multiple ranks with one GPU
          std::cerr << "WARNING (Allegro_spin): my rank (" << deviceidx
                    << ") is bigger than the number of visible devices (" << devicecount
                    << "), wrapping around to use device " << deviceidx % devicecount
                    << " again!!!";
          deviceidx = deviceidx % devicecount;
        }
        else
        {
          // Otherwise, more ranks than GPUs is an error
          std::cerr << "ERROR (Allegro_spin): my rank (" << deviceidx
                    << ") is bigger than the number of visible devices (" << devicecount << ")!!!";
          error->all(
              FLERR,
              "pair_allegro_spin: mismatch between number of ranks and number of available GPUs");
        }
      }
    }
    device = c10::Device(torch::kCUDA, deviceidx);
  }
  else
  {
    device = torch::kCPU;
  }
  if (comm->me == 0)
    std::cout << "Allegro_spin is using device " << device << "\n";
}

PairAllegro_spin::~PairAllegro_spin()
{
  if (allocated)
  {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(emag);
  }
  memory->destroy(fm_net);
}

void PairAllegro_spin::init_style()
{
  PairSpin::init_style();

  if (atom->tag_enable == 0)
    error->all(FLERR, "Pair style PairAllegro_spin requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style PairAllegro_spin requires newton pair on");

  neighbor->requests[neighbor->nrequest - 1]->ghost = 1;
}

double PairAllegro_spin::init_one(int i, int j)
{
  return cutoff;
}

void PairAllegro_spin::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

void PairAllegro_spin::settings(int narg, char ** /*arg*/)
{
  // "allegro" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command, too many arguments");
}

void PairAllegro_spin::coeff(int narg, char **arg)
{
  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Requires at least: * * <model>.pth <type1> <type2> ... <typen>
  // Optionally followed by: bec_constant <val1> ... <val_n>
  if (narg < (3 + ntypes))
    error->all(FLERR,
               "Incorrect args for pair coefficients, should be * * <model>.pth <type1> <type2> "
               "... <typen> [bec_constant <val1> ... <val_n>]");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  std::vector<std::string> elements(ntypes);
  for (int i = 0; i < ntypes; i++)
  {
    elements[i] = arg[i + 1];
  }

  if (comm->me == 0)
    std::cout << "Allegro_spin: Loading model from " << arg[2] << "\n";

  std::unordered_map<std::string, std::string> metadata = {{"config", ""},
                                                           {"nequip_version", ""},
                                                           {"r_max", ""},
                                                           {"n_species", ""},
                                                           {"type_names", ""},
                                                           {"_jit_bailout_depth", ""},
                                                           {"_jit_fusion_strategy", ""},
                                                           {"allow_tf32", ""}};
  model = torch::jit::load(std::string(arg[2]), device, metadata);
  model.eval();

  // Check if model is a NequIP model
  if (metadata["nequip_version"].empty())
  {
    error->all(FLERR,
               "The indicated TorchScript file does not appear to be a deployed NequIP model; did "
               "you forget to run `nequip-deploy`?");
  }

  // If the model is not already frozen, we should freeze it:
  // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
  if (model.hasattr("training"))
  {
    if (comm->me == 0)
      std::cout << "Allegro_spin: Freezing TorchScript model...\n";
#ifdef DO_TORCH_FREEZE_HACK
    // Do the hack
    // Copied from the implementation of torch::jit::freeze,
    // except without the broken check
    // See https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
    bool optimize_numerics = true; // the default
    // the {} is preserved_attrs
    auto out_mod = freeze_module(model, {});
    // See 1.11 bugfix in https://github.com/pytorch/pytorch/pull/71436
    auto graph = out_mod.get_method("forward").graph();
    OptimizeFrozenGraph(graph, optimize_numerics);
    model = out_mod;
#else
    // Do it normally
    model = torch::jit::freeze(model);
#endif
  }

#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
  // Set JIT bailout to avoid long recompilations for many steps
  size_t jit_bailout_depth;
  if (metadata["_jit_bailout_depth"].empty())
  {
    // This is the default used in the Python code
    jit_bailout_depth = 2;
  }
  else
  {
    jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
  }
  torch::jit::getBailoutDepth() = jit_bailout_depth;
#else
  // In PyTorch >=1.11, this is now set_fusion_strategy
  torch::jit::FusionStrategy strategy;
  if (metadata["_jit_fusion_strategy"].empty())
  {
    // This is the default used in the Python code
    strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  }
  else
  {
    std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
    std::string fusion_type, fusion_depth;
    while (std::getline(strat_stream, fusion_type, ','))
    {
      std::getline(strat_stream, fusion_depth, ';');
      strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC
                                                  : torch::jit::FusionBehavior::DYNAMIC,
                          std::stoi(fusion_depth)});
    }
  }
  torch::jit::setFusionStrategy(strategy);
#endif

  // Set whether to allow TF32:
  bool allow_tf32;
  if (metadata["allow_tf32"].empty())
  {
    // Better safe than sorry
    allow_tf32 = false;
  }
  else
  {
    // It gets saved as an int 0/1
    allow_tf32 = std::stoi(metadata["allow_tf32"]);
  }
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);

  // Optional reproducibility switch: only when ALLEGRO_DETERMINISTIC is set in env.
  // Default (unset) keeps the original fast, non-deterministic behavior.
  // When enabled on GPU, also set env CUBLAS_WORKSPACE_CONFIG=:4096:8 (else torch errors).
  if (std::getenv("ALLEGRO_DETERMINISTIC"))
  {
    if (comm->me == 0)
      std::cout << "Allegro_spin: ALLEGRO_DETERMINISTIC set -> enabling deterministic algorithms\n";
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
    at::globalContext().setDeterministicAlgorithms(true);
#else
    at::globalContext().setDeterministicAlgorithms(true, /*warn_only=*/true);
#endif
    at::globalContext().setDeterministicCuDNN(true);
  }

  // std::cout << "Allegro_spin: Information from model: " << metadata.size() << " key-value pairs\n";
  // for( const auto& n : metadata ) {
  //   std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  // }

  cutoff = std::stod(metadata["r_max"]);

  type_mapper.resize(ntypes);
  std::stringstream ss;
  int n_species = std::stod(metadata["n_species"]);
  ss << metadata["type_names"];
  if (comm->me == 0)
  {
    std::cout << "Type mapping:"
              << "\n";
    std::cout << "Allegro_spin type | Allegro_spin name | LAMMPS type | LAMMPS name"
              << "\n";
  }
  for (int i = 0; i < n_species; i++)
  {
    std::string ele;
    ss >> ele;
    for (int itype = 1; itype <= ntypes; itype++)
    {
      if (ele.compare(arg[itype + 3 - 1]) == 0)
      {
        type_mapper[itype - 1] = i;
        if (comm->me == 0)
          std::cout << i << " | " << ele << " | " << itype << " | " << arg[itype + 3 - 1] << "\n";
      }
    }
  }

  // Parse optional bec_constant keyword
  int idx = 3 + ntypes;
  if (idx < narg) {
    if (strcmp(arg[idx], "bec_constant") != 0)
      error->all(FLERR, "Unexpected extra argument after type names in pair_coeff; "
                        "expected \"bec_constant\"");
    idx++;
    if (idx + ntypes != narg)
      error->all(FLERR, "Incorrect number of bec_constant values; expected one per type");
    use_constant_bec = true;
    bec_constant_per_type.resize(ntypes);
    for (int i = 0; i < ntypes; i++)
      bec_constant_per_type[i] = std::stod(arg[idx++]);
  }

  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      if ((type_mapper[i - 1] >= 0) && (type_mapper[j - 1] >= 0))
        setflag[i][j] = 1;

  char *batchstr = std::getenv("BATCHSIZE");
  if (batchstr != NULL)
  {
    batch_size = std::atoi(batchstr);
  }
}

void *PairAllegro_spin::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "cut") == 0)
    return (void *)&cutoff;
  return nullptr;
}

// Force and energy computation
void PairAllegro_spin::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // Get info from lammps:

  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
  // Add Atom spin
  double **fm = atom->fm; // force on spin
  double **sp = atom->sp; // spin vector
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // Number of local/real atoms
  int nlocal = atom->nlocal;
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  // Should be on.
  int newton_pair = force->newton_pair;

  // Number of local/real atoms
  int inum = list->inum;
  assert(inum == nlocal); // This should be true, if my understanding is correct
  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost;
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

  // Total number of bonds (sum of number of neighbors)
  int nedges = 0;

  // Number of bonds per atom
  std::vector<int> neigh_per_atom(nlocal, 0);

  // checking size of emag

  if (nlocal_max < nlocal)
  { // grow emag lists if necessary
    nlocal_max = nlocal;
    memory->grow(emag, nlocal_max, "pair/spin:emag");
  }

#pragma omp parallel for reduction(+                                    : nedges)
  for (int ii = 0; ii < nlocal; ii++)
  {
    int i = ilist[ii];
    emag[i] = 0.0;

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++)
    {
      int j = jlist[jj];
      j &= NEIGHMASK;

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq <= cutoff * cutoff)
      {
        neigh_per_atom[ii]++;
        nedges++;
      }
    }
  }

  // Cumulative sum of neighbors, for knowing where to fill in the edges tensor
  std::vector<int> cumsum_neigh_per_atom(nlocal);

  for (int ii = 1; ii < nlocal; ii++)
  {
    cumsum_neigh_per_atom[ii] = cumsum_neigh_per_atom[ii - 1] + neigh_per_atom[ii - 1];
  }

  torch::Tensor pos_tensor = torch::zeros({ntotal, 3});
  torch::Tensor spin_tensor = torch::zeros({ntotal, 3});
  torch::Tensor edges_tensor =
      torch::zeros({2, nedges}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor ij2type_tensor =
      torch::zeros({ntotal}, torch::TensorOptions().dtype(torch::kInt64));

  auto pos = pos_tensor.accessor<float, 2>();
  auto spin = spin_tensor.accessor<float, 2>();
  auto edges = edges_tensor.accessor<long, 2>();
  auto ij2type = ij2type_tensor.accessor<long, 1>();

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  if (debug_mode)
    printf("Allegro_spin edges: i j rij\n");
#pragma omp parallel for
  for (int i = 0; i < ntotal; i++)
  {
    int itype = type[i];

    ij2type[i] = type_mapper[itype - 1];

    pos[i][0] = x[i][0];
    pos[i][1] = x[i][1];
    pos[i][2] = x[i][2];

    if (sp[i][3] > 0)
    {
      spin[i][0] = sp[i][0];
      spin[i][1] = sp[i][1];
      spin[i][2] = sp[i][2];
    }
    else
    {
      spin[i][0] = 0;
      spin[i][1] = 0;
      spin[i][2] = 0;
    }
  }

#pragma omp parallel for
  for (int ii = 0; ii < nlocal; ii++)
  {
    int i = ilist[ii];
    int itag = tag[i];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];

    int edge_counter = cumsum_neigh_per_atom[ii];
    for (int jj = 0; jj < jnum; jj++)
    {
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq > cutoff * cutoff)
      {
        continue;
      }

      // TODO: double check order
      edges[0][edge_counter] = i;
      edges[1][edge_counter] = j;

      edge_counter++;

      if (debug_mode) printf("%d %d %.10g\n", itag - 1, jtag - 1, sqrt(rsq));
    }
  }
  if (debug_mode)
    printf("end Allegro_spin edges\n");

  // ---------------------------------------------------------------------------
  // Model evaluation. BATCHSIZE (env) = max number of local centre atoms per
  // forward call. Allegro is strictly local: the energy is a sum over centre
  // atoms, so forces / spin forces / virials of disjoint centre subsets add up
  // exactly, while peak GPU memory scales with the edges of one chunk only.
  // Unset or >= nlocal: a single call, identical to the original code.
  // ---------------------------------------------------------------------------
  torch::Tensor pos_d = pos_tensor.to(device);
  torch::Tensor spin_d = spin_tensor.to(device);
  torch::Tensor types_d = ij2type_tensor.to(device);
  torch::Tensor edges_d = edges_tensor.to(device);

  int chunk = (batch_size > 0 && batch_size < nlocal) ? batch_size : nlocal;
  c10::impl::GenericDict output(c10::StringType::get(), c10::AnyType::get());

  if (chunk >= nlocal)
  {
    c10::Dict<std::string, torch::Tensor> input;
    input.insert("pos", pos_d);
    input.insert("spin", spin_d);
    input.insert("edge_index", edges_d);
    input.insert("atom_types", types_d);
    std::vector<torch::IValue> input_vector(1, input);
    output = model.forward(input_vector).toGenericDict();
    // zero_mean_born inside the model subtracts the mean over local+ghost nodes, which differs
    // between MPI ranks; ghost nodes have no edges, so their output is exactly -mean -> remove it.
    // (Single rank: a constant shift, cancelled by the global mean removal below anyway.)
    if (atom->bec_flag && !use_constant_bec && nghost > 0 && output.contains("born_effective_charges"))
    {
      torch::Tensor b = output.at("born_effective_charges").toTensor();
      torch::Tensor b2 = b.reshape({ntotal, 9});
      output.insert_or_assign("born_effective_charges", (b2 - b2.narrow(0, nlocal, 1)).reshape(b.sizes()));
    }
  }
  else
  {
    // edge offsets per centre (edges are stored grouped by ii, see cumsum above)
    std::vector<long> edge_start(nlocal + 1, 0);
    for (int ii = 0; ii < nlocal; ii++) edge_start[ii + 1] = edge_start[ii] + neigh_per_atom[ii];

    torch::Tensor ilist_t = torch::from_blob(ilist, {nlocal}, torch::TensorOptions().dtype(torch::kInt32))
                                .to(torch::kInt64).to(device);
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor forces_acc = torch::zeros({ntotal, 3}, fopt);
    torch::Tensor spinf_acc;
    torch::Tensor ae_acc = torch::zeros({ntotal, 1}, fopt);
    torch::Tensor virial_acc, atom_virial_acc, bec_acc;
    bool have_sf = false, have_bec = false;

    for (int c0 = 0; c0 < nlocal; c0 += chunk)
    {
      int c1 = std::min(nlocal, c0 + chunk);
      long e0 = edge_start[c0], e1 = edge_start[c1];
      c10::Dict<std::string, torch::Tensor> input;
      input.insert("pos", pos_d);
      input.insert("spin", spin_d);
      input.insert("edge_index", edges_d.narrow(1, e0, e1 - e0).contiguous());
      input.insert("atom_types", types_d);
      std::vector<torch::IValue> input_vector(1, input);
      auto out = model.forward(input_vector).toGenericDict();

      torch::Tensor centres = ilist_t.narrow(0, c0, c1 - c0);

      forces_acc.add_(out.at("forces").toTensor().reshape({ntotal, 3}));
      if (out.contains("spin_forces"))
      {
        torch::Tensor sf = out.at("spin_forces").toTensor().reshape({ntotal, 3});
        if (!have_sf) { spinf_acc = torch::zeros({ntotal, 3}, fopt); have_sf = true; }
        spinf_acc.add_(sf);
      }
      // per-atom energy is exact only for this chunk's centres (others carry just the species shift)
      torch::Tensor ae = out.at("atomic_energy").toTensor().reshape({ntotal, 1});
      ae_acc.index_copy_(0, centres, ae.index_select(0, centres));
      if (vflag)
      {
        torch::Tensor v = out.at("virial").toTensor();
        if (!virial_acc.defined()) virial_acc = torch::zeros_like(v);
        virial_acc.add_(v);
      }
      if (vflag_atom)
      {
        torch::Tensor av = out.at("atom_virial").toTensor();
        if (!atom_virial_acc.defined()) atom_virial_acc = torch::zeros_like(av);
        atom_virial_acc.add_(av);
      }
      if (atom->bec_flag && !use_constant_bec && out.contains("born_effective_charges"))
      {
        // The model subtracts the mean over all graph nodes (zero_mean_born). Nodes that are not
        // centres of this chunk have no edges, so their output is exactly -mean: add it back.
        torch::Tensor b = out.at("born_effective_charges").toTensor().reshape({ntotal, 9});
        int ref = (nghost > 0) ? nlocal : ilist[(c1 < nlocal) ? c1 : 0];
        torch::Tensor bc = b.index_select(0, centres) - b.narrow(0, ref, 1);
        if (!have_bec) { bec_acc = torch::zeros({ntotal, 9}, fopt); have_bec = true; }
        bec_acc.index_copy_(0, centres, bc);
      }
    }
    output.insert("forces", forces_acc);
    if (have_sf) output.insert("spin_forces", spinf_acc);
    output.insert("atomic_energy", ae_acc);
    if (virial_acc.defined()) output.insert("virial", virial_acc);
    if (atom_virial_acc.defined()) output.insert("atom_virial", atom_virial_acc);
    if (have_bec) output.insert("born_effective_charges", bec_acc);
  }

  // if (debug_mode)
  //   printf("forward model\n");

  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<float, 2>();

  torch::Tensor spin_forces_tensor;
  bool have_spin_forces = true;
  try
  {
    spin_forces_tensor = output.at("spin_forces").toTensor().cpu();
  }
  catch (const std::out_of_range &)
  {
    have_spin_forces = false;
    static int warned_missing_spin_forces = 0;
    if (!warned_missing_spin_forces && comm->me == 0)
      error->warning(
          FLERR,
          "Pair style spin/allegro_spin model has no spin_forces output; spin torques will be set to zero");
    warned_missing_spin_forces = 1;
  }

  // torch::Tensor total_energy_tensor = output.at("total_energy").toTensor().cpu(); WRONG WITH MPI

  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor().cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();
  float atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<float>()[0];
  if (vflag)
  {
    torch::Tensor v_tensor = output.at("virial").toTensor().cpu();
    auto v = v_tensor.accessor<float, 3>();
    // Convert from 3x3 symmetric tensor format, which NequIP outputs, to the flattened form LAMMPS expects
    // First [0] index on v is batch
    virial[0] += v[0][0][0];
    virial[1] += v[0][1][1];
    virial[2] += v[0][2][2];
    virial[3] += v[0][0][1];
    virial[4] += v[0][0][2];
    virial[5] += v[0][1][2];
  }
  // std::cout << "atomic energy sum: " << atomic_energy_sum << std::endl;
  // std::cout << "Total energy: " << total_energy_tensor << "\n";
  // std::cout << "atomic energy shape: " << atomic_energy_tensor.sizes()[0] << "," << atomic_energy_tensor.sizes()[1] << std::endl;
  // std::cout << "atomic energies: " << atomic_energy_tensor << std::endl;

  // Write forces and per-atom energies (0-based tags here)

  eng_vdwl = 0.0;
  // const double hbar = hbar;
  if (atom->nmax > nmax_fm_net)
  {
    nmax_fm_net = atom->nmax;
    memory->grow(fm_net, nmax_fm_net, 3, "pair/allegro_spin:fm_net");
  }
  const float *spin_forces_ptr = nullptr;
  if (have_spin_forces) spin_forces_ptr = spin_forces_tensor.data_ptr<float>();
#pragma omp parallel for reduction(+                                    : eng_vdwl)
  for (int i = 0; i < ntotal; i++)
  {
    f[i][0] = forces[i][0];
    f[i][1] = forces[i][1];
    f[i][2] = forces[i][2];

    if (have_spin_forces && sp[i][3] > 0)
    {
      fm[i][0] = spin_forces_ptr[3 * i + 0] / hbar;
      fm[i][1] = spin_forces_ptr[3 * i + 1] / hbar;
      fm[i][2] = spin_forces_ptr[3 * i + 2] / hbar;
    }
    else
    {
      fm[i][0] = 0;
      fm[i][1] = 0;
      fm[i][2] = 0;
    }
    fm_net[i][0] = fm[i][0];
    fm_net[i][1] = fm[i][1];
    fm_net[i][2] = fm[i][2];

    if (i < nlocal)
    {
      if (eflag_atom)
        eatom[i] = atomic_energies[i][0];
      eng_vdwl += atomic_energies[i][0];
      emag[i] = atomic_energies[i][0];
    }
  }
  // add the ghost-atom contributions of the network field into the owners
  comm->reverse_comm_pair(this);

  if (vflag_atom)
  {
    torch::Tensor atomic_virial_tensor = output.at("atom_virial").toTensor().cpu();
    auto atomic_virial = atomic_virial_tensor.accessor<float, 3>();
    for (int i = 0; i < nlocal; i++)
    {
      cvatom[i][0] += 1.0 * atomic_virial[i][0][0]; // xx
      cvatom[i][1] += 1.0 * atomic_virial[i][1][1]; // yy
      cvatom[i][2] += 1.0 * atomic_virial[i][2][2]; // zz
      cvatom[i][3] += 1.0 * atomic_virial[i][0][1]; // xy
      cvatom[i][4] += 1.0 * atomic_virial[i][0][2]; // xz
      cvatom[i][5] += 1.0 * atomic_virial[i][1][2]; // yz
      cvatom[i][6] += 1.0 * atomic_virial[i][1][0]; // yx
      cvatom[i][7] += 1.0 * atomic_virial[i][2][0]; // zx
      cvatom[i][8] += 1.0 * atomic_virial[i][2][1]; // zy
    }
  }
  if (atom->bec_flag)
  {
    double **bec = atom->bec;

    double bec_all[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    if (use_constant_bec) {
      static int warned_constant_bec = 0;
      if (!warned_constant_bec && comm->me == 0)
        error->warning(FLERR, "Pair style spin/allegro_spin: using constant BEC from pair_coeff "
                              "bec_constant (model does not supply born_effective_charges)");
      warned_constant_bec = 1;

      int *type = atom->type;
      for (int i = 0; i < nlocal; i++) {
        double v = bec_constant_per_type[type[i] - 1];
        bec[i][0] = v;  bec[i][1] = 0;  bec[i][2] = 0;
        bec[i][3] = 0;  bec[i][4] = v;  bec[i][5] = 0;
        bec[i][6] = 0;  bec[i][7] = 0;  bec[i][8] = v;

        bec_all[0] += v;
        bec_all[4] += v;
        bec_all[8] += v;
      }
    } else {
      torch::Tensor bec_tensor = output.at("born_effective_charges").toTensor().cpu();
      auto becs_nn = bec_tensor.accessor<float, 2>();

      for (int i = 0; i < nlocal; i++)
      {
        bec[i][0] = becs_nn[i][0];
        bec[i][1] = becs_nn[i][1];
        bec[i][2] = becs_nn[i][2];
        bec[i][3] = becs_nn[i][3];
        bec[i][4] = becs_nn[i][4];
        bec[i][5] = becs_nn[i][5];
        bec[i][6] = becs_nn[i][6];
        bec[i][7] = becs_nn[i][7];
        bec[i][8] = becs_nn[i][8];

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

    double bec_all_mpi[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int nlocal_mpi = 0;

    MPI_Allreduce(bec_all, bec_all_mpi, 9, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&nlocal, &nlocal_mpi, 1, MPI_INT, MPI_SUM, world);
    double bec_ave_zero[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int ii = 0; ii < 9; ii++) {
      bec_ave_zero[ii] = bec_all_mpi[ii] / nlocal_mpi;
    }

    for (int i = 0; i < nlocal; i++)
    {
      for (int j = 0; j < 9; j++) {
        bec[i][j] -= bec_ave_zero[j];
      }
    }
  }
  if (debug_mode)
  {
    std::cout << "Allegro model output:" << std::endl;
    std::cout << "forces: " << forces_tensor << std::endl;
    std::cout << "atomic_energy: " << atomic_energy_tensor << std::endl;
    if (have_spin_forces)
      std::cout << "spin forces: " << spin_forces_tensor << std::endl;
    if (vflag)
      std::cout << "virial: " << output.at("virial").toTensor().cpu() << std::endl;
    if (atom->bec_flag)
      std::cout << "born_effective_charges: " << output.at("born_effective_charges").toTensor().cpu() << std::endl;
  }
}

void PairAllegro_spin::compute_single_pair(int ii, double fmi[3]){
  // Called by fix nve/spin (and spin/sib) before every single-spin update. The network field
  // is frozen over the timestep: return the one stored by the last compute().
  fmi[0] += fm_net[ii][0];
  fmi[1] += fm_net[ii][1];
  fmi[2] += fm_net[ii][2];
}

int PairAllegro_spin::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  for (int i = first; i < first + n; i++)
  {
    buf[m++] = fm_net[i][0];
    buf[m++] = fm_net[i][1];
    buf[m++] = fm_net[i][2];
  }
  return m;
}

void PairAllegro_spin::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++)
  {
    int j = list[i];
    fm_net[j][0] += buf[m++];
    fm_net[j][1] += buf[m++];
    fm_net[j][2] += buf[m++];
  }
}
