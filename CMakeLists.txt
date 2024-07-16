cmake_minimum_required(VERSION 3.19)

project(mgard_standard)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED PATHS /home/qian/Software/ADIOS2/install-adios/) 
find_package(mgard REQUIRED)

find_package(CUDAToolkit REQUIRED)
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")

add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")

add_executable(mgard_ge mgard_ge.cpp)
target_link_libraries(mgard_ge adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgard_ge_cpu mgard_ge_cpu.cpp)
target_link_libraries(mgard_ge_cpu adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgard_adios_ge mgard_adios_ge.cpp)
target_link_libraries(mgard_adios_ge adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgard_adios_decompress mgard_adios_decompress.cpp)
target_link_libraries(mgard_adios_decompress adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(MeshGrid MeshGrid.cpp nonUniformMap.cpp)
target_link_libraries(MeshGrid adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgard_adios_remesh mgard_adios_remesh.cpp nonUniformMap.cpp)
target_link_libraries(mgard_adios_remesh adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgard_adios_decompress_remesh mgard_adios_decompress_remesh.cpp nonUniformMap.cpp)
target_link_libraries(mgard_adios_decompress_remesh adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(calc_err calc_err.cpp)
target_link_libraries(calc_err adios2::adios2 MPI::MPI_C mgard::mgard)
