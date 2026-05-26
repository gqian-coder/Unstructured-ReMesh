cmake_minimum_required(VERSION 3.19)

project(mgard_standard)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED) 
find_package(mgard REQUIRED)

add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")

add_executable(mgard_ge mgard_ge.cpp)
target_link_libraries(mgard_ge adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgard_ge_cpu mgard_ge_cpu.cpp)
target_link_libraries(mgard_ge_cpu adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgard_adios_ge mgard_adios_ge.cpp)
target_link_libraries(mgard_adios_ge adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgardPlug_adios_ge mgardPlug_adios_ge.cpp SystemTools.cpp)
target_link_libraries(mgardPlug_adios_ge adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgardPlug_adios_decompress mgardPlug_adios_decompress.cpp SystemTools.cpp)
target_link_libraries(mgardPlug_adios_decompress adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgard_adios_decompress mgard_adios_decompress.cpp)
target_link_libraries(mgard_adios_decompress adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(MeshGrid MeshGrid.cpp nonUniformMap.cpp SystemTools.cpp)
target_link_libraries(MeshGrid adios2::adios2 MPI::MPI_C)

# GPU-accelerated MeshGrid - portable across HIP, CUDA, and CPU
# Detect GPU backend from MGARD configuration
option(ENABLE_GPU_MESHGRID "Enable GPU acceleration for MeshGrid" ON)

if(ENABLE_GPU_MESHGRID)
    # Check for HIP
    find_package(hip QUIET)
    # Check for CUDA
    find_package(CUDAToolkit QUIET)
    
    if(hip_FOUND)
        message(STATUS "Building MeshGridGPU with HIP backend")
        set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "HIP GPU architectures")
        enable_language(HIP)
        
        # Build the GPU source as HIP, but keep other sources as CXX
        set_source_files_properties(MeshGridGPU.cpp PROPERTIES LANGUAGE HIP)
        
        # Build non-GPU sources separately to avoid HIP flags
        add_library(MeshGridGPU_support OBJECT nonUniformMap.cpp SystemTools.cpp)
        target_link_libraries(MeshGridGPU_support PRIVATE adios2::adios2 MPI::MPI_C)
        
        add_executable(MeshGridGPU MeshGridGPU.cpp $<TARGET_OBJECTS:MeshGridGPU_support>)
        # For Cray systems, explicitly add MPI include path for HIP compiler
        if(DEFINED ENV{CRAY_MPICH_DIR})
            target_include_directories(MeshGridGPU PRIVATE $ENV{CRAY_MPICH_DIR}/include)
        endif()
        target_include_directories(MeshGridGPU PRIVATE ${MPI_C_INCLUDE_DIRS})
        target_link_libraries(MeshGridGPU adios2::adios2 MPI::MPI_C hip::device)
        target_compile_definitions(MeshGridGPU PRIVATE __HIP_PLATFORM_AMD__)
        # Cray perftools-base injects __cray_dset_detect into CXX objects; the
        # HIP linker (ROCm clang++) doesn't auto-add the PAT stub the way the
        # Cray CC wrapper does.  The symbol is intentionally left undefined at
        # link time (PAT provides it at runtime via LD_PRELOAD when profiling).
        target_link_options(MeshGridGPU PRIVATE -Wl,--unresolved-symbols=ignore-in-object-files)
    elseif(CUDAToolkit_FOUND)
        message(STATUS "Building MeshGridGPU with CUDA backend")
        enable_language(CUDA)
        set_source_files_properties(MeshGridGPU.cpp PROPERTIES LANGUAGE CUDA)
        add_executable(MeshGridGPU MeshGridGPU.cpp nonUniformMap.cpp SystemTools.cpp)
        target_link_libraries(MeshGridGPU adios2::adios2 MPI::MPI_C CUDA::cudart)
    else()
        message(STATUS "Building MeshGridGPU with CPU-only backend (no GPU found)")
        add_executable(MeshGridGPU MeshGridGPU.cpp nonUniformMap.cpp SystemTools.cpp)
        target_link_libraries(MeshGridGPU adios2::adios2 MPI::MPI_C)
    endif()
else()
    message(STATUS "GPU MeshGrid disabled, building CPU-only version")
    add_executable(MeshGridGPU MeshGridGPU.cpp nonUniformMap.cpp SystemTools.cpp)
    target_link_libraries(MeshGridGPU adios2::adios2 MPI::MPI_C)
endif()

add_executable(mgard_adios_remesh mgard_adios_remesh.cpp nonUniformMap.cpp)
target_link_libraries(mgard_adios_remesh adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(mgard_adios_decompress_remesh mgard_adios_decompress_remesh.cpp nonUniformMap.cpp)
target_link_libraries(mgard_adios_decompress_remesh adios2::adios2 MPI::MPI_C mgard::mgard)

add_executable(calc_err calc_err.cpp)
target_link_libraries(calc_err adios2::adios2 MPI::MPI_C mgard::mgard)

# Single-block MGARD-API compression ratio test (no ADIOS2 operator layer).
add_executable(mgard_cr_block mgard_cr_block.cpp)
target_link_libraries(mgard_cr_block adios2::adios2 MPI::MPI_C mgard::mgard)

# Cell-centroid (cell-average + nodal-residual) compression wrapper.
# Uses CompressMGARDCentroidOperator ADIOS2 plugin loaded at runtime via
# ADIOS2_PLUGIN_PATH; no compile-time link against the plugin needed.
add_executable(mgardCentroid_adios_ge mgardCentroid_adios_ge.cpp)
target_link_libraries(mgardCentroid_adios_ge adios2::adios2 MPI::MPI_C mgard::mgard)
