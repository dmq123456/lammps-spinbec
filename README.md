# LAMMPS with SpinGNN extensions

Modified source files of [LAMMPS](https://www.lammps.org) **stable_29Sep2021_update2**
(29 Sep 2021, Update 2) for running spin–lattice dynamics with SpinGNN potentials
(NequIP/Allegro-based, with spin and Born effective charges).

Added styles:

| Type | Name |
|---|---|
| pair | `spin/allegro_spin`, `spin/nequip_spin`, `allegro`, `nequip` |
| atom | `bec` (use with `atom_style hybrid spin bec`) |
| compute | `bec_dipole` |
| command | `bec_dipole_reset` |
| fix | `spin/sib`, `spin/sib_npt` |

Several existing files (e.g. `SPIN/fix_nve_spin`, `SPIN/compute_spin`, `fix_efield`,
`dump_custom`, `compute_property_atom`) were also modified.

This repository holds only the files that differ from or are added to the
official release. `install.sh` downloads the official release, copies these
files on top of it and builds LAMMPS.

<details>
<summary>Files in <code>src/</code> (18 added, 22 modified)</summary>

Added:

- `src/SPIN/fix_spin_sib.cpp`
- `src/SPIN/fix_spin_sib.h`
- `src/SPIN/fix_spin_sib_npt.cpp`
- `src/SPIN/fix_spin_sib_npt.h`
- `src/SPIN/pair_allegro_spin.cpp`
- `src/SPIN/pair_allegro_spin.h`
- `src/SPIN/pair_nequip_spin.cpp`
- `src/SPIN/pair_nequip_spin.h`
- `src/atom_vec_bec.cpp`
- `src/atom_vec_bec.h`
- `src/bec_dipole_reset.cpp`
- `src/bec_dipole_reset.h`
- `src/compute_bec.cpp`
- `src/compute_bec.h`
- `src/pair_allegro.cpp`
- `src/pair_allegro.h`
- `src/pair_nequip.cpp`
- `src/pair_nequip.h`

Modified:

- `src/BROWNIAN/fix_brownian.cpp`
- `src/MAKE/Makefile.mpi`
- `src/SPIN/atom_vec_spin.cpp`
- `src/SPIN/compute_spin.cpp`
- `src/SPIN/fix_nve_spin.cpp`
- `src/SPIN/fix_nve_spin.h`
- `src/SPIN/pair_spin.h`
- `src/atom.cpp`
- `src/atom.h`
- `src/compute_property_atom.cpp`
- `src/compute_property_atom.h`
- `src/domain.h`
- `src/dump_custom.cpp`
- `src/dump_custom.h`
- `src/fix_efield.cpp`
- `src/fix_efield.h`
- `src/fix_nh.cpp`
- `src/fix_nve.cpp`
- `src/library.cpp`
- `src/min_fire.cpp`
- `src/min_quickmin.cpp`
- `src/pair_hybrid.h`

</details>

## Requirements

- Linux x86_64
- A Python environment with **PyTorch 1.11** (the SpinGNN environment), activated
- CUDA toolkit with `nvcc` (CUDA 11.x recommended, matching PyTorch's CUDA 11.3)
- MPI C++ compiler `mpicxx` (the SpinGNN environment provides one)
- `cmake` ≥ 3.13, `gcc`, and `curl` or `wget`

cuDNN is **not** needed: `local/cudnn_stub/` contains a placeholder that only
satisfies PyTorch's CMake check (see `local/cudnn_stub/include/cudnn.h`).

## Install

```bash
git clone https://github.com/dmq123456/lammps-spinbec.git
cd lammps-spinbec
source /path/to/spingnn-env/bin/activate      # or: conda activate spingnn
bash install.sh
```

The executable is `lammps-stable_29Sep2021_update2/build/lmp_mpi`.

Options (environment variables):

```bash
JOBS=8 bash install.sh                        # number of parallel compile jobs
CUDA_HOME=/usr/local/cuda-11.8 bash install.sh # choose a CUDA toolkit
BUILD_TYPE=Debug bash install.sh              # debug build
LAMMPS_TARBALL=/path/to/lammps-stable_29Sep2021_update2.tar.gz bash install.sh  # use this release file
```

Offline installation: download
<https://github.com/lammps/lammps/archive/refs/tags/stable_29Sep2021_update2.tar.gz>
elsewhere, save it as `lammps-stable_29Sep2021_update2.tar.gz` **next to** (or inside)
this folder, and run `bash install.sh`; the file is found automatically.
`LAMMPS_TARBALL` is only needed if the file is somewhere else.

Do **not** replace the official `src/` folder by this one: some unmodified
files are not stored here, so the files must be copied *on top of* the
official `src/` (which is what `install.sh` does).

## Run

Activate the same Python environment first (LAMMPS loads the PyTorch libraries
from it), then e.g.

```bash
mpirun -np 4 /path/to/lmp_mpi -in in.lmp
```

## License

LAMMPS is distributed under the GNU General Public License v2 (GPL-2.0);
these modifications are distributed under the same license.

## Chunked model evaluation (2026-10-10, `src/SPIN/pair_allegro_spin.cpp`)

`pair_style spin/allegro_spin` now honours the environment variable `BATCHSIZE`:
the local atoms are split into groups of at most `BATCHSIZE` centre atoms and the
model is called once per group. Allegro is strictly local (energy = sum over centre
atoms), so energies, forces, spin forces and virials of the groups add up exactly,
while peak GPU memory depends only on the group size (~2-3 GB + ~10 MB per centre
atom for the NiI2 production model) instead of the system size.

```bash
BATCHSIZE=300 lmp_mpi -in in.lmp                       # one GPU
mpirun -np 2 -genv BATCHSIZE 300 lmp_mpi -in in.lmp     # one rank per GPU
```

Unset (or >= number of local atoms): a single call, as before. Validated against the
original code: forces <=1e-6 eV/A, spin forces ~2e-6 relative, BEC <=2e-6 e.
BATCHSIZE 150-300 costs <5% speed. Also fixed: with several MPI ranks the original code
left a rank-dependent constant offset in the Born effective charges (the model's
zero_mean_born averages over local+ghost nodes per rank); now removed in both paths.

## Spin field returned to fix nve/spin (2026-10-12)

`compute_single_pair()` used to return `atom->fm[i]`, which fix nve/spin overwrites with
field + langevin/spin noise + damping after every quarter-step update, so those terms
accumulated within a timestep and the spin thermostat drove the spins to ~2.2x the set
temperature. `compute()` now stores the network spin field in its own array (`fm_net`,
owned + ghost atoms) and sums the ghost contributions into their owners with the pair's
reverse communication; `compute_single_pair()` returns that array. The field therefore
contains neither thermostat terms nor what other fixes add to `atom->fm` in `post_force`
(e.g. `fix precession/spin`, which an earlier snapshot of `atom->fm` counted twice).
Requires `newton on`. Checked against the snapshot version: identical NVE and
langevin/spin runs; with `precession/spin` the Zeeman field now enters once; 1 vs 2 MPI
ranks agree.

Known issue (not fixed): the network spin field is frozen over a timestep, so the
sequential nve/spin scheme is not symplectic and NVE spin-lattice dynamics heats the
spins (energy drift ~0.014 meV/atom/ps at 1 fs, ~0.0024 at 0.5 fs, NiI2). Use
`fix langevin/spin T 0.01 seed` (not as fix #0, and before fix nve/spin) for spectra.
