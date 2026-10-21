#!/usr/bin/env bash
# Build LAMMPS (stable_29Sep2021_update2) with the SpinGNN extensions.
#
# This is install.sh with CUDA handling made robust for clusters where
#   * no `module load cuda` is done, and
#   * the system CUDA toolkit is older than the CUDA version PyTorch was built with
#     (e.g. system CUDA 11.0, PyTorch cu113).
#
# How the CUDA version is chosen
# ------------------------------
# PyTorch is linked against the CUDA runtime it was built with (for PyTorch 1.11
# that is usually CUDA 11.3).  Linking against an older system toolkit (11.0)
# fails with undefined symbols such as cudaUserObjectCreate and with mixed
# libcublas/libcublasLt internal symbols.
#
# Therefore:
#   * nvcc + headers may come from any installed toolkit (they are only needed
#     for CMake's CUDA language check; LAMMPS itself compiles no .cu files);
#   * if the toolkit is older than PyTorch's CUDA version, the CUDA runtime
#     libraries are taken from the activated conda environment (cudatoolkit),
#     which already matches PyTorch.
#
# A small "CUDA link root" is created for that case:
#   <CUDA_LINK_ROOT>/bin/nvcc          -> real nvcc (wrapper)
#   <CUDA_LINK_ROOT>/include           -> toolkit headers
#   <CUDA_LINK_ROOT>/lib64              -> conda env's CUDA 11.x libraries
#   <CUDA_LINK_ROOT>/lib64/stubs/...    -> toolkit libcuda.so stub
# and passed to CMake as CUDA_TOOLKIT_ROOT_DIR.  Nothing is installed system-wide
# and no `module load cuda` is required.
#
#   usage:  bash install-1.sh
#
# Before running: activate a Python environment that provides PyTorch 1.11
# (e.g. the SpinGNN environment).  nvcc is auto-detected; mpicxx, cmake and
# gcc must be available.
#
# Optional environment variables:
#   JOBS=8                     parallel make jobs (default: number of CPU cores)
#   BUILD_TYPE=Release         CMake build type (Release or Debug)
#   CUDA_HOME=/path/to/cuda    prefer this toolkit for nvcc/headers
#   CUDA_LINK_ROOT=/path       where to build the CUDA link root
#                              (default: ~/.cache/lammps/cuda-link-<torch-cuda>)
#   FORCE_ENV_CUDA=1           always use the conda env's CUDA runtime libraries
#   CHECK_ONLY=1               run prerequisites/CUDA setup, then exit (no build)
#   LAMMPS_TARBALL=/path/to/lammps-stable_29Sep2021_update2.tar.gz
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
ENV_PREFIX=$(python -c 'import sys; print(sys.prefix)')
TORCH_CUDA=$(python -c 'import torch; print(torch.version.cuda or "")')
echo "  PyTorch  $TORCH_VER  (CUDA ${TORCH_CUDA:-none})  ($TORCH_CMAKE)"
echo "  env      $ENV_PREFIX"
[[ $TORCH_VER == 1.11.* ]] || echo "  WARNING: SpinGNN was developed with PyTorch 1.11; found $TORCH_VER"

# Do not prepend $ENV_PREFIX/lib to LD_LIBRARY_PATH globally: on old-glibc
# clusters this can make system tools (cmake, gcc) load conda libraries that
# require a newer glibc.  A targeted override is used only for the lmp_mpi
# self-check at the end.
case ":${LD_LIBRARY_PATH:-}:" in
    *"/opt/software/cuda-11.0"*)
        echo "  WARNING: LD_LIBRARY_PATH contains /opt/software/cuda-11.0; unload that CUDA module before running lmp_mpi."
        ;;
esac

# ---- locate nvcc without requiring `module load cuda` ----
find_nvcc() {
    local cand
    if [ -n "${CUDA_HOME:-}" ]; then
        cand="$CUDA_HOME/bin/nvcc"
        if [ -x "$cand" ]; then echo "$cand"; return 0; fi
        echo "WARNING: CUDA_HOME=$CUDA_HOME has no bin/nvcc; searching other locations" >&2
    fi
    cand=$(command -v nvcc 2>/dev/null || true)
    if [ -n "$cand" ] && [ -x "$cand" ]; then echo "$cand"; return 0; fi
    for cand in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc /opt/software/cuda*/bin/nvcc; do
        if [ -x "$cand" ]; then echo "$cand"; return 0; fi
    done
    return 1
}

NVCC=$(find_nvcc) || fail "nvcc not found. Install a CUDA toolkit, set CUDA_HOME=/path/to/cuda, or make nvcc available in PATH."
NVCC_REAL=$(readlink -f "$NVCC" 2>/dev/null || echo "$NVCC")
TOOLKIT_ROOT_REAL=$(cd "$(dirname "$NVCC_REAL")/.." && pwd)
TOOLKIT_CUDA_VER=$("$NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)
[ -n "$TOOLKIT_CUDA_VER" ] || fail "Could not determine the CUDA version of $NVCC."
echo "  nvcc     $TOOLKIT_CUDA_VER  ($NVCC)"

for tool in mpicxx cmake gcc tar; do
    command -v $tool >/dev/null || fail "$tool not found."
done
echo "  MPI      $(command -v mpicxx)"
CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
printf '%s\n%s\n' 3.13 "$CMAKE_VER" | sort -V -C || fail "cmake >= 3.13 required, found $CMAKE_VER"
echo "  cmake    $CMAKE_VER"

# ---- host compiler check ----
# nvcc < 12 rejects gcc > 9.  Prefer CUDAHOSTCXX if the user set it, otherwise gcc.
GCC_BIN=${CUDAHOSTCXX:-$(command -v gcc || true)}
[ -n "$GCC_BIN" ] || fail "gcc not found."
GCC_VER=$("$GCC_BIN" -dumpversion 2>/dev/null || true)
GCC_MAJOR=${GCC_VER%%.*}
[ -n "$GCC_MAJOR" ] || fail "Could not determine the gcc version."
TOOLKIT_MAJOR=${TOOLKIT_CUDA_VER%%.*}

if [ "$GCC_MAJOR" -lt 5 ]; then
    fail "gcc $GCC_VER is too old for PyTorch 1.11 / C++14. Load a newer gcc (e.g. module load gcc/9.4.0)."
fi

if [ "$TOOLKIT_MAJOR" -lt 12 ] && [ "$GCC_MAJOR" -gt 9 ]; then
    GCC9=$(command -v gcc-9 2>/dev/null || true)
    GXX9=$(command -v g++-9 2>/dev/null || true)
    if [ -n "$GCC9" ] && [ -n "$GXX9" ]; then
        export CC="${CC:-$GCC9}"
        export CXX="${CXX:-$GXX9}"
        export CUDAHOSTCXX="${CUDAHOSTCXX:-$GCC9}"
        GCC_VER=$("$GCC9" -dumpversion)
        echo "  gcc      $GCC_VER  (using gcc-9 for CUDA $TOOLKIT_CUDA_VER)"
    else
        fail "CUDA $TOOLKIT_CUDA_VER nvcc does not support gcc $GCC_VER (> 9). Load gcc/9.4.0 (module load gcc/9.4.0) or make gcc-9/g++-9 available."
    fi
else
    echo "  gcc      $GCC_VER"
fi

# ---- decide which CUDA runtime libraries to link against ----
# version_ge A B  ->  true when A >= B
version_ge() {
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | tail -1)" = "$1" ]
}

USE_ENV_CUDA=0
if [ -z "$TORCH_CUDA" ]; then
    echo "  WARNING: PyTorch reports no CUDA build; using the toolkit's libraries."
elif [ "${FORCE_ENV_CUDA:-0}" = 1 ]; then
    echo "  FORCE_ENV_CUDA=1: using CUDA $TORCH_CUDA runtime from the conda env"
    USE_ENV_CUDA=1
elif version_ge "$TOOLKIT_CUDA_VER" "$TORCH_CUDA"; then
    echo "  CUDA libs: system toolkit $TOOLKIT_CUDA_VER (matches or is newer than PyTorch's $TORCH_CUDA)"
else
    echo "  CUDA libs: toolkit $TOOLKIT_CUDA_VER is older than PyTorch's $TORCH_CUDA"
    echo "             -> using the CUDA $TORCH_CUDA runtime from $ENV_PREFIX/lib"
    USE_ENV_CUDA=1
fi

if [ "$USE_ENV_CUDA" = 1 ]; then
    TORCH_CUDA_MAJOR=${TORCH_CUDA%%.*}
    CUDART_SONAME="libcudart.so.${TORCH_CUDA_MAJOR}.0"
    CUBLAS_SONAME="libcublas.so.${TORCH_CUDA_MAJOR}"
    CUBLASLT_SONAME="libcublasLt.so.${TORCH_CUDA_MAJOR}"

    for lib in "$CUDART_SONAME" "$CUBLAS_SONAME" "$CUBLASLT_SONAME"; do
        if [ ! -e "$ENV_PREFIX/lib/$lib" ]; then
            fail "PyTorch needs CUDA $TORCH_CUDA, but $ENV_PREFIX/lib/$lib is missing. Install cudatoolkit=$TORCH_CUDA into the environment, or use a CUDA $TORCH_CUDA toolkit."
        fi
    done

    CUDA_LINK_ROOT=${CUDA_LINK_ROOT:-$HOME/.cache/lammps/cuda-link-$TORCH_CUDA}
    case "$CUDA_LINK_ROOT" in
        ""|"/"|"$HOME"|"$HOME/") fail "Refusing unsafe CUDA_LINK_ROOT='$CUDA_LINK_ROOT'" ;;
    esac
    say "Creating CUDA link root $CUDA_LINK_ROOT"
    rm -rf "$CUDA_LINK_ROOT"
    mkdir -p "$CUDA_LINK_ROOT/bin" "$CUDA_LINK_ROOT/lib64/stubs"

    # nvcc wrapper: the real nvcc keeps finding its own headers/backends by its real path
    cat > "$CUDA_LINK_ROOT/bin/nvcc" <<EOF
#!/bin/sh
exec "$NVCC_REAL" "\$@"
EOF
    chmod 755 "$CUDA_LINK_ROOT/bin/nvcc"

    # headers from the system toolkit (11.0 is fine for compiling C++ with torch headers)
    CUDA_INC="$TOOLKIT_ROOT_REAL/include"
    if [ ! -d "$CUDA_INC" ] && [ -d "$TOOLKIT_ROOT_REAL/targets/x86_64-linux/include" ]; then
        CUDA_INC="$TOOLKIT_ROOT_REAL/targets/x86_64-linux/include"
    fi
    [ -d "$CUDA_INC" ] || fail "CUDA include directory not found under $TOOLKIT_ROOT_REAL"
    ln -s "$CUDA_INC" "$CUDA_LINK_ROOT/include"

    # libcuda.so is only needed at link time; use the toolkit stub or the driver lib
    CUDA_STUB=""
    for cand in "$TOOLKIT_ROOT_REAL/lib64/stubs/libcuda.so" "$TOOLKIT_ROOT_REAL/lib/stubs/libcuda.so" \
                /usr/lib64/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so; do
        if [ -e "$cand" ]; then CUDA_STUB="$cand"; break; fi
    done
    if [ -n "$CUDA_STUB" ]; then
        ln -s "$CUDA_STUB" "$CUDA_LINK_ROOT/lib64/stubs/libcuda.so"
    else
        echo "  WARNING: libcuda.so stub not found; CMake may complain about CUDA_CUDA_LIB" >&2
    fi

    # CUDA runtime libraries from the conda environment (these match PyTorch)
    for pat in 'libcudart.so*' 'libcublas.so*' 'libcublasLt.so*' \
               'libcufft.so*' 'libcurand.so*' 'libcusparse.so*' \
               'libcusolver.so*' 'libnvrtc.so*' 'libnvToolsExt.so*' \
               'libcudadevrt.a'; do
        for f in "$ENV_PREFIX"/lib/$pat; do
            if [ -e "$f" ]; then
                ln -sf "$f" "$CUDA_LINK_ROOT/lib64/"
            fi
        done
    done

    [ -e "$CUDA_LINK_ROOT/lib64/$CUDART_SONAME" ] || fail "staged CUDA root is missing $CUDART_SONAME"
    [ -e "$CUDA_LINK_ROOT/lib64/$CUBLAS_SONAME" ] || fail "staged CUDA root is missing $CUBLAS_SONAME"
    [ -e "$CUDA_LINK_ROOT/lib64/$CUBLASLT_SONAME" ] || fail "staged CUDA root is missing $CUBLASLT_SONAME"

    echo "  nvcc     $NVCC_REAL (CUDA $TOOLKIT_CUDA_VER)"
    echo "  CUDA libs $ENV_PREFIX/lib (CUDA $TORCH_CUDA)"
    echo "  link root $CUDA_LINK_ROOT"
    EFFECTIVE_CUDA_ROOT="$CUDA_LINK_ROOT"
else
    EFFECTIVE_CUDA_ROOT="$TOOLKIT_ROOT_REAL"
    echo "  CUDA libs $TOOLKIT_ROOT_REAL (CUDA $TOOLKIT_CUDA_VER)"
fi

if [ "${CHECK_ONLY:-0}" = 1 ]; then
    say "CHECK_ONLY=1: prerequisites and CUDA setup are OK; not building"
    exit 0
fi

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
            DL_HINT="Check your network/proxy, or download $LAMMPS_URL, put it next to this folder as lammps-$LAMMPS_TAG.tar.gz, and run install-1.sh again (or set LAMMPS_TARBALL=/path/to/file)"
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
"${CC:-gcc}" -shared -fPIC -Wl,-soname,libcudnn.so.8 -o "$STUB/lib64/libcudnn.so" "$STUB/cudnn_stub.c"

# ---------------------------------------------------------------- 5. configure and build
# PyTorch's CMake config also requires an MKL include directory, but nothing is
# compiled against or linked to MKL; the environment's include/ satisfies the check
# (and holds the real MKL headers if mkl-include is installed).
PY_INCLUDE="$ENV_PREFIX/include"

BUILD_DIR="$LAMMPS_DIR/build"
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
    say "Removing stale CMake cache (it may contain the old CUDA toolkit paths)"
    rm -f "$BUILD_DIR/CMakeCache.txt"
fi

say "Configuring in $BUILD_DIR ($BUILD_TYPE)"
cmake -S "$LAMMPS_DIR/cmake" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
    -DCUDA_TOOLKIT_ROOT_DIR="$EFFECTIVE_CUDA_ROOT" \
    -DCMAKE_CUDA_COMPILER="$EFFECTIVE_CUDA_ROOT/bin/nvcc" \
    -DCUDA_NVCC_EXECUTABLE="$EFFECTIVE_CUDA_ROOT/bin/nvcc" \
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
RUN_LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ! HELP=$(LD_LIBRARY_PATH="$RUN_LD_LIBRARY_PATH" "$LMP" -h 2>&1); then
    echo "$HELP" >&2
    fail "$LMP could not start. Make sure the spingnn environment is active and LD_LIBRARY_PATH does not point at CUDA 11.0."
fi
for style in spin/allegro_spin spin/nequip_spin bec_dipole; do
    grep -qw -- "$style" <<< "$HELP" || fail "style '$style' is missing from $LMP"
done
echo "  styles spin/allegro_spin spin/nequip_spin bec_dipole: OK"

if command -v ldd >/dev/null 2>&1; then
    echo "  CUDA runtime libraries resolved by lmp_mpi:"
    LD_LIBRARY_PATH="$RUN_LD_LIBRARY_PATH" ldd "$LMP" | grep -E 'libcudart|libcublas|cublasLt' || true
    if LD_LIBRARY_PATH="$RUN_LD_LIBRARY_PATH" ldd "$LMP" 2>/dev/null | grep -q '/opt/software/cuda-11.0'; then
        echo "  WARNING: lmp_mpi is resolving CUDA 11.0 libraries; check LD_LIBRARY_PATH."
    fi
fi

cat <<EOF

Done. LAMMPS executable:
    $LMP

At run time:
  * activate the same environment:  source $ENV_PREFIX/bin/activate
  * keep its CUDA runtime first:    export LD_LIBRARY_PATH=$ENV_PREFIX/lib:\${LD_LIBRARY_PATH:-}
  * do NOT module load cuda/11.0 (its runtime does not match PyTorch's CUDA $TORCH_CUDA).
EOF
