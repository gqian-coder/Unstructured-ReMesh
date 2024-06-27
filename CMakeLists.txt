cmake_minimum_required(VERSION 3.9)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(mgard_standard)

find_package(MPI REQUIRED)
find_package(zstd REQUIRED /ccs/home/gongq/frontier/MGARD/install-serial/lib/cmake/zstd/)
find_package(ADIOS2 REQUIRED PATHS /lustre/orion/csc143/scratch/gongq/frontier/SoftwareDev/ADIOS2/install-frontier/)
find_package(mgard PATHS /ccs/home/gongq/frontier/MGARD/install-serial/) 

find_package(SZ3 REQUIRED PATHS /lustre/orion/csc143/scratch/gongq/frontier/SoftwareDev/SZ3/install/usr/local/lib64/cmake/SZ3/) 
set(SZ3_INCLUDES "/lustre/orion/csc143/scratch/gongq/frontier/SoftwareDev/SZ3/install/usr/local/include")
set(ZSTD_INCLUDES "/ccs/home/gongq/frontier/MGARD/install-serial/include/")
#find_library(SZ3_LIB SZ3 REQUIRED HINTS "/lustre/orion/csc143/scratch/gongq/frontier/SoftwareDev/SZ3/install/usr/local/lib64/")

find_package(zfp REQUIRED PATHS /lustre/orion/csc143/scratch/gongq/frontier/SoftwareDev/zfp/install/usr/local/)
set(ZFP_INCLUDES "/lustre/orion/csc143/scratch/gongq/frontier/SoftwareDev/zfp/install/usr/local/include")
find_library(ZFP_LIB zfp HINTS "/lustre/orion/csc143/scratch/gongq/frontier/SoftwareDev/zfp/install/usr/local/lib64/")


add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")

add_executable(compress_mgr
               compress_mgr.cpp)

add_executable(resampleCompress
               nonUniformMap.cpp
               resampleCompress.cpp)

add_executable(resampleCompress_mgrTime
               nonUniformMap.cpp
               resampleCompress_mgrTime.cpp)

add_executable(resampleCompress_xy
               nonUniformMap.cpp
               resampleCompress_xy.cpp)

add_executable(resampleCompress_m
               nonUniformMap.cpp
               resampleCompress_m.cpp)

add_executable(resampleCompress_zfp
               resampleCompress_zfp.cpp
               nonUniformMap.cpp)

add_executable(resampleCompress_zfp_xy
               resampleCompress_zfp_xy.cpp
               nonUniformMap.cpp)

add_executable(resampleCompress_sz
               resampleCompress_sz.cpp
               nonUniformMap.cpp)

add_executable(resampleCompress_sz_xy
               resampleCompress_sz_xy.cpp
               nonUniformMap.cpp)

target_link_libraries(compress_mgr mgard::mgard adios2::adios2)
target_link_libraries(resampleCompress_m adios2::adios2 mgard::mgard)
target_link_libraries(resampleCompress_mgrTime adios2::adios2 mgard::mgard)
target_link_libraries(resampleCompress adios2::adios2 mgard::mgard)
target_link_libraries(resampleCompress_xy adios2::adios2 mgard::mgard)

target_include_directories(resampleCompress_zfp PRIVATE ${ZFP_INCLUDES})
target_link_libraries(resampleCompress_zfp adios2::adios2 ${ZFP_LIB})
target_include_directories(resampleCompress_zfp_xy PRIVATE ${ZFP_INCLUDES})
target_link_libraries(resampleCompress_zfp_xy adios2::adios2 ${ZFP_LIB})

target_include_directories(resampleCompress_sz PRIVATE ${SZ3_INCLUDES} ${ZSTD_INCLUDES})
target_link_libraries(resampleCompress_sz adios2::adios2 SZ3::SZ3)

target_include_directories(resampleCompress_sz_xy PRIVATE ${SZ3_INCLUDES} ${ZSTD_INCLUDES})
target_link_libraries(resampleCompress_sz_xy adios2::adios2 SZ3::SZ3)


#target_link_libraries(resampleAMRCompress adios2::adios2 mgard::mgard)
