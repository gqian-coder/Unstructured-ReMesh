set -x
set -e

mgard_install_dir=/lustre/orion/cfd164/proj-shared/gongq/Software/MGARD/install-hip-frontier/
adios_install_dir=/lustre/orion/proj-shared/cfd164/gongq/Software/ADIOS2/install-adios/

export CC=cc
export CXX=CC

export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_TARGET=gfx908
export OMPI_CC=hipcc

rm build/CMakeCache.txt

cmake -S .  -B ./build\
            -DCMAKE_PREFIX_PATH="${mgard_install_dir};${adios_install_dir}"\ 
            -DCMAKE_C_COMPILER=hipcc\
            -DCMAKE_CXX_COMPILER=hipcc

cmake --build ./build
