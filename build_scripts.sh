#!/bin/sh

set -x
set -e

module load rocm/5.3.0
module load PrgEnv-gnu/8.3.3
module load craype-accel-amd-gfx90a
ml cmake

export CC=cc
export CXX=CC

export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_TARGET=gfx908
export OMPI_CC=hipcc

# Setup MGARD installation dir


adios_dir=/lustre/orion/csc143/scratch/gongq/frontier/SoftwareDev/ADIOS2/install-frontier/lib64/cmake/adios2/
mgr_dir=/ccs/home/gongq/frontier/MGARD/install-hip-frontier/lib64/cmake/mgard
zstd_dir=/ccs/home/gongq/frontier/MGARD/install-hip-frontier/lib64/cmake/zstd
install_dir=/ccs/home/gongq/frontier/MGARD/install-hip-frontier/
protobuf_dir=/ccs/home/gongq/frontier/MGARD/install-hip-frontier/cmake/protobuf

rm -f build/CMakeCache.txt
mkdir -p build
cmake -S .  -B ./build \
        -Dmgard_ROOT=${mgr_dir} \
        -DCMAKE_PREFIX_PATH="${zstd_dir};${install_dir};${protobuf_dir}" \
        -DADIOS2_DIR=${adios_dir} \
        -DCMAKE_C_COMPILER=hipcc \
        -DCMAKE_CXX_COMPILER=hipcc


cmake --build ./build
