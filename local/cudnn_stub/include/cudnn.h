/*
 * Placeholder for cuDNN -- NOT the real library.
 *
 * The CMake config shipped with PyTorch insists on finding cuDNN (cudnn.h and
 * libcudnn.so, version >= 7) because PyTorch itself was built with it. LAMMPS
 * and the SpinGNN pair styles never call cuDNN, and the conda PyTorch build
 * carries its own copy, so these placeholders only let the configure step pass.
 * install.sh compiles cudnn_stub.c into lib64/libcudnn.so.
 */
#ifndef CUDNN_STUB_H
#define CUDNN_STUB_H
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 2
#define CUDNN_PATCHLEVEL 0
#endif
