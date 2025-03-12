set -x
set -e

mgard_install_dir=/lustre/orion/cfd164/proj-shared/gongq/Software/MGARD/install-hip-frontier/
adios_install_dir=/lustre/orion/proj-shared/cfd164/gongq/Software/ADIOS2/install-adios/

rm -rf build
mkdir build
cmake -S .  -B ./build \
            -Dmgard_ROOT=${install_dir}\
            -DCMAKE_CUDA_ARCHITECTURES=75\
            -DCMAKE_PREFIX_PATH="${mgard_install_dir};${adios_install_dir}"

cmake --build ./build
