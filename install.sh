#!/usr/bin/env bash
# Build LAMMPS (stable_29Sep2021_update2) with the SpinGNN extensions
# (pair styles spin/allegro_spin, spin/nequip_spin, allegro, nequip; atom style bec; ...).
#
#   usage:  bash install.sh
#
# Before running: activate a Python environment that provides PyTorch 1.11
# (e.g. the SpinGNN environment), and make sure nvcc, mpicxx and cmake are in PATH.
#
# Optional environment variables:
#   JOBS=8                     parallel make jobs (default: number of CPU cores)
#   BUILD_TYPE=Release         CMake build type (Release or Debug)
#   CUDA_HOME=/usr/local/cuda  CUDA toolkit to use (default: the nvcc found in PATH)
#   LAMMPS_TARBALL=/path/to/lammps-stable_29Sep2021_update2.tar.gz   official release to use
#
# The official release is looked for first as lammps-stable_29Sep2021_update2.tar.gz
# (or stable_29Sep2021_update2.tar.gz) in this folder and in its parent folder;
# it is downloaded only if none is found.
set -euo pipefail

LAMMPS_TAG=stable_29Sep2021_update2
LAMMPS_URL=https://github.com/lammps/lammps/archive/refs/tags/${LAMMPS_TAG}.tar.gz

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS=${JOBS:-$(nproc)}
BUILD_TYPE=${BUILD_TYPE:-Release}
STUB="$REPO/local/cudnn_stub"

say()  { echo -e "\n>>> $*"; }
fail() { echo -e "\nERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. prerequisites
say "Checking prerequisites"
command -v python >/dev/null || fail "python not found. Activate the SpinGNN environment first."
TORCH_VER=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null) \
    || fail "PyTorch cannot be imported. Activate the SpinGNN environment first."
TORCH_CMAKE=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
echo "  PyTorch  $TORCH_VER  ($TORCH_CMAKE)"
[[ $TORCH_VER == 1.11.* ]] || echo "  WARNING: SpinGNN was developed with PyTorch 1.11; found $TORCH_VER"

if [ -n "${CUDA_HOME:-}" ]; then
    NVCC="$CUDA_HOME/bin/nvcc"
else
    NVCC=$(command -v nvcc || true)
fi
[ -x "$NVCC" ] || fail "nvcc not found. Install the CUDA toolkit, or set CUDA_HOME=/path/to/cuda."
CUDA_ROOT=$(cd "$(dirname "$NVCC")/.." && pwd)
echo "  CUDA     $("$NVCC" --version | grep -o 'release [0-9.]*')  ($CUDA_ROOT)"

for tool in mpicxx cmake gcc tar; do
    command -v $tool >/dev/null || fail "$tool not found."
done
echo "  MPI      $(command -v mpicxx)"
CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
printf '%s\n%s\n' 3.13 "$CMAKE_VER" | sort -V -C || fail "cmake >= 3.13 required, found $CMAKE_VER"
echo "  cmake    $CMAKE_VER"
echo "  gcc      $(gcc -dumpversion)"

# ---------------------------------------------------------------- 2. LAMMPS source
if [ -f "$REPO/cmake/CMakeLists.txt" ]; then
    # already a full LAMMPS tree (e.g. the developer's own checkout): build in place
    LAMMPS_DIR="$REPO"
    say "Using the LAMMPS tree in $LAMMPS_DIR"
else
    LAMMPS_DIR="$REPO/lammps-$LAMMPS_TAG"
    if [ ! -f "$LAMMPS_DIR/cmake/CMakeLists.txt" ]; then
        # look for an already downloaded release next to this folder or inside it
        TARBALL=${LAMMPS_TARBALL:-}
        if [ -z "$TARBALL" ]; then
            for f in "$REPO/lammps-$LAMMPS_TAG.tar.gz" "$REPO/$LAMMPS_TAG.tar.gz" \
                     "$REPO/../lammps-$LAMMPS_TAG.tar.gz" "$REPO/../$LAMMPS_TAG.tar.gz"; do
                if [ -f "$f" ]; then TARBALL="$f"; break; fi
            done
        fi
        if [ -n "${LAMMPS_TARBALL:-}" ] && [ ! -f "$LAMMPS_TARBALL" ]; then
            fail "LAMMPS_TARBALL=$LAMMPS_TARBALL does not exist."
        fi
        TARBALL=${TARBALL:-$REPO/$LAMMPS_TAG.tar.gz}
        if [ ! -f "$TARBALL" ]; then
            say "Downloading official LAMMPS $LAMMPS_TAG (about 120 MB)"
            DL_HINT="Check your network/proxy, or download $LAMMPS_URL, put it next to this folder as lammps-$LAMMPS_TAG.tar.gz, and run install.sh again (or set LAMMPS_TARBALL=/path/to/file)"
            if command -v curl >/dev/null; then
                curl -fL --connect-timeout 30 --retry 3 -o "$TARBALL.part" "$LAMMPS_URL" \
                    || { rm -f "$TARBALL.part"; fail "download failed. $DL_HINT"; }
            elif command -v wget >/dev/null; then
                wget --timeout=30 --tries=3 -O "$TARBALL.part" "$LAMMPS_URL" \
                    || { rm -f "$TARBALL.part"; fail "download failed. $DL_HINT"; }
            else
                fail "neither curl nor wget found. $DL_HINT"
            fi
            mv "$TARBALL.part" "$TARBALL"
        fi
        say "Unpacking $TARBALL"
        tar -xzf "$TARBALL" -C "$REPO"
    fi
    grep -q '"29 Sep 2021"' "$LAMMPS_DIR/src/version.h" \
        || fail "$LAMMPS_DIR is not LAMMPS 29 Sep 2021. Remove it and run again."

    # overlay (not replace): files not in this repository stay as in the official release
    say "Copying the modified src/ into $LAMMPS_DIR/src"
    cp -a "$REPO/src/." "$LAMMPS_DIR/src/"
fi

# ---------------------------------------------------------------- 3. CMakeLists.txt
say "Patching cmake/CMakeLists.txt for libtorch"
CMAKE_FILE="$LAMMPS_DIR/cmake/CMakeLists.txt"
sed -i 's/set(CMAKE_CXX_STANDARD 1[0-9])/set(CMAKE_CXX_STANDARD 14)/' "$CMAKE_FILE"
if ! grep -q 'find_package(Torch REQUIRED)' "$CMAKE_FILE"; then
    cat >> "$CMAKE_FILE" <<'EOF'

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
EOF
fi

# ---------------------------------------------------------------- 4. cuDNN placeholder
say "Building the cuDNN placeholder (see local/cudnn_stub/include/cudnn.h)"
mkdir -p "$STUB/lib64"
gcc -shared -fPIC -Wl,-soname,libcudnn.so.8 -o "$STUB/lib64/libcudnn.so" "$STUB/cudnn_stub.c"

# ---------------------------------------------------------------- 5. configure and build
# PyTorch's CMake config also requires an MKL include directory, but nothing is
# compiled against or linked to MKL; the environment's include/ satisfies the check
# (and holds the real MKL headers if mkl-include is installed).
PY_INCLUDE="$(python -c 'import sys; print(sys.prefix)')/include"

BUILD_DIR="$LAMMPS_DIR/build"
say "Configuring in $BUILD_DIR ($BUILD_TYPE)"
cmake -S "$LAMMPS_DIR/cmake" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
    -DCUDNN_INCLUDE_DIR="$STUB/include" \
    -DCUDNN_LIBRARY="$STUB/lib64" \
    -DMKL_INCLUDE_DIR="$PY_INCLUDE" \
    -DPKG_SPIN=yes \
    -DPKG_MC=yes \
    -DLAMMPS_MACHINE=mpi

say "Compiling with $JOBS jobs (this takes a while)"
cmake --build "$BUILD_DIR" -j "$JOBS"

# ---------------------------------------------------------------- 6. self-check
LMP="$BUILD_DIR/lmp_mpi"
say "Self-check"
[ -x "$LMP" ] || fail "$LMP was not built."
HELP=$("$LMP" -h)
for style in spin/allegro_spin spin/nequip_spin bec_dipole; do
    grep -qw -- "$style" <<< "$HELP" || fail "style '$style' is missing from $LMP"
done
echo "  styles spin/allegro_spin, spin/nequip_spin, bec_dipole: OK"

cat <<EOF

Done. LAMMPS executable:
    $LMP
It needs the same Python environment (PyTorch libraries) at run time:
activate the environment before running it, or add it to PATH.
EOF
