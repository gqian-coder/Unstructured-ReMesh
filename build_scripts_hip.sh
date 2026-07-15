set -x
set -e

source ~/frontier_model_to_load.sh

unset LD_PRELOAD
export LIBRARY_PATH=/opt/xpmem/lib64:${LIBRARY_PATH:-}

mgard_install_dir=/ccs/proj/cfd164/mgard
adios_install_dir=/lustre/orion/cfd164/proj-shared/gongq/Software/ADIOS2/install-adios-cray

export CC=cc
export CXX=CC

export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_TARGET=gfx908
export OMPI_CC=hipcc

#rm build_cfd164/CMakeCache.txt

cmake -S .  -B ./build_cfd164 \
            -DCMAKE_PREFIX_PATH="${mgard_install_dir};${adios_install_dir};${ROCM_PATH}" \
            -DMPI_C_COMPILER=cc \
            -DMPI_CXX_COMPILER=CC \
            -DMPI_C_INCLUDE_PATH="${CRAY_MPICH_DIR}/include" \
            -DMPI_CXX_INCLUDE_PATH="${CRAY_MPICH_DIR}/include" \
            -DMPI_C_LIB_NAMES=mpi_cray \
            -DMPI_CXX_LIB_NAMES=mpi_cray \
            -DMPI_mpi_cray_LIBRARY="${CRAY_MPICH_DIR}/lib/libmpi_cray.so" \
            -DCMAKE_BUILD_TYPE=Release

cmake --build ./build_cfd164 -- -j8
