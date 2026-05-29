/*
 * MeshGridGPU.hpp
 * GPU kernel implementations for mesh-to-grid mapping computation
 * Portable across NVIDIA (CUDA), AMD (HIP), and CPU backends
 *
 *  Created on: Feb 4, 2026
 */

#ifndef MESHGRID_GPU_HPP
#define MESHGRID_GPU_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

// =============================================================================
// GPU Backend Detection and Portability Layer
// =============================================================================

// Try to detect GPU backends
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
    #include <hip/hip_runtime.h>
    #define GPU_ENABLED 1
    #define GPU_BACKEND_HIP 1
    // HIP compatibility macros
    #define gpuMalloc hipMalloc
    #define gpuFree hipFree
    #define gpuMemcpy hipMemcpy
    #define gpuMemset hipMemset
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuDeviceSynchronize hipDeviceSynchronize
    #define gpuGetLastError hipGetLastError
    #define gpuPeekAtLastError hipPeekAtLastError
    #define gpuGetErrorString hipGetErrorString
    #define gpuSuccess hipSuccess
    #define gpuGetDeviceCount hipGetDeviceCount
    #define gpuSetDevice hipSetDevice
    #define gpuStream_t hipStream_t
    #define gpuStreamCreate hipStreamCreate
    #define gpuStreamDestroy hipStreamDestroy
    #define gpuStreamSynchronize hipStreamSynchronize
    #define gpuMemcpyAsync hipMemcpyAsync
    #define gpuMemsetAsync hipMemsetAsync
    #define gpuError_t hipError_t
    #define GPU_KERNEL __global__
    #define GPU_DEVICE __device__
    #define GPU_HOST __host__
    #define GPU_HOST_DEVICE __host__ __device__
#elif defined(__CUDACC__) || defined(__NVCC__)
    #include <cuda_runtime.h>
    #define GPU_ENABLED 1
    #define GPU_BACKEND_CUDA 1
    // CUDA compatibility macros
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemcpy cudaMemcpy
    #define gpuMemset cudaMemset
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuGetLastError cudaGetLastError
    #define gpuPeekAtLastError cudaPeekAtLastError
    #define gpuGetErrorString cudaGetErrorString
    #define gpuSuccess cudaSuccess
    #define gpuGetDeviceCount cudaGetDeviceCount
    #define gpuSetDevice cudaSetDevice
    #define gpuStream_t cudaStream_t
    #define gpuStreamCreate cudaStreamCreate
    #define gpuStreamDestroy cudaStreamDestroy
    #define gpuStreamSynchronize cudaStreamSynchronize
    #define gpuMemcpyAsync cudaMemcpyAsync
    #define gpuMemsetAsync cudaMemsetAsync
    #define gpuError_t cudaError_t
    #define GPU_KERNEL __global__
    #define GPU_DEVICE __device__
    #define GPU_HOST __host__
    #define GPU_HOST_DEVICE __host__ __device__
#else
    #define GPU_ENABLED 0
    #define GPU_KERNEL
    #define GPU_DEVICE
    #define GPU_HOST
    #define GPU_HOST_DEVICE
#endif

namespace meshgrid
{
namespace gpu
{

// GPU error checking macro
#if GPU_ENABLED
#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        gpuError_t err = call;                                                 \
        if (err != gpuSuccess) {                                               \
            std::cerr << "GPU Error: " << gpuGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        }                                                                      \
    } while (0)
#else
#define GPU_CHECK(call) (void)0
#endif

/**
 * @brief Check if GPU device is available
 */
inline bool isGPUAvailable()
{
#if GPU_ENABLED
    int deviceCount = 0;
    gpuError_t err = gpuGetDeviceCount(&deviceCount);
    return (err == gpuSuccess && deviceCount > 0);
#else
    return false;
#endif
}

/**
 * @brief Get the number of available GPU devices
 */
inline int getGPUDeviceCount()
{
#if GPU_ENABLED
    int deviceCount = 0;
    gpuError_t err = gpuGetDeviceCount(&deviceCount);
    if (err != gpuSuccess) {
        return 0;
    }
    return deviceCount;
#else
    return 0;
#endif
}

// =============================================================================
// GPU Kernels
// =============================================================================

#if GPU_ENABLED

/**
 * @brief GPU Kernel: Map mesh nodes to uniform grid indices
 * Each thread handles one mesh node
 */
template <int MAX_DIMS = 4>
GPU_KERNEL void kernel_closest_Node2UniformGrid(
    size_t* __restrict__ nodeMapGrid,      // Output: grid index for each node
    const double* __restrict__ nodeCoordX, // Input: X coordinates
    const double* __restrict__ nodeCoordY, // Input: Y coordinates  
    const double* __restrict__ nodeCoordZ, // Input: Z coordinates (or nullptr for 2D)
    const size_t* __restrict__ resampleRate, // Grid dimensions
    const double* __restrict__ minvGrid,   // Minimum grid coordinates
    const double* __restrict__ spaceGrid,  // Grid spacing
    size_t nNodePt,                        // Number of mesh nodes
    int n_dims)                            // Number of dimensions
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nNodePt) return;

    size_t gridIndex = 0;
    size_t dimMultiplier = 1;
    
    // Process each dimension
    // Dimension 0 (typically Z)
    if (n_dims >= 1) {
        double posVal = nodeCoordX[idx] - minvGrid[0];
        size_t dimIdx = (size_t)floor(posVal / spaceGrid[0]);
        if (dimIdx >= resampleRate[0]) dimIdx = resampleRate[0] - 1;
        gridIndex += dimIdx * dimMultiplier;
        dimMultiplier *= resampleRate[0];
    }
    
    // Dimension 1 (typically X)
    if (n_dims >= 2) {
        double posVal = nodeCoordY[idx] - minvGrid[1];
        size_t dimIdx = (size_t)floor(posVal / spaceGrid[1]);
        if (dimIdx >= resampleRate[1]) dimIdx = resampleRate[1] - 1;
        gridIndex += dimIdx * dimMultiplier;
        dimMultiplier *= resampleRate[1];
    }
    
    // Dimension 2 (typically Y)
    if (n_dims >= 3 && nodeCoordZ != nullptr) {
        double posVal = nodeCoordZ[idx] - minvGrid[2];
        size_t dimIdx = (size_t)floor(posVal / spaceGrid[2]);
        if (dimIdx >= resampleRate[2]) dimIdx = resampleRate[2] - 1;
        gridIndex += dimIdx * dimMultiplier;
    }
    
    nodeMapGrid[idx] = gridIndex;
}

/**
 * @brief GPU Kernel: Count cluster sizes (atomic increment)
 */
GPU_KERNEL void kernel_count_clusters(
    const size_t* __restrict__ nodeMapGrid, // Input: grid index for each node
    size_t* __restrict__ nCluster,          // Output: count per grid point (use atomicAdd)
    size_t nNodePt)                         // Number of mesh nodes
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nNodePt) return;
    
    size_t gridIdx = nodeMapGrid[idx];
    atomicAdd((unsigned long long*)&nCluster[gridIdx], 1ULL);
}

/**
 * @brief GPU Kernel: Mark grid points that have at least one node mapped
 */
GPU_KERNEL void kernel_mark_occupied_grid(
    const size_t* __restrict__ nodeMapGrid, // Input: grid index for each node
    uint8_t* __restrict__ gridOccupied,     // Output: 1 if occupied, 0 otherwise
    size_t nNodePt)                         // Number of mesh nodes
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nNodePt) return;
    
    gridOccupied[nodeMapGrid[idx]] = 1;
}

/**
 * @brief GPU Kernel: Compute sparse map for occupied grid points
 * This is a sequential operation - run with 1 thread
 */
GPU_KERNEL void kernel_compute_sparse_map(
    const uint8_t* __restrict__ gridOccupied, // Input: occupied flags
    size_t* __restrict__ gridSparseMap,       // Output: dense -> sparse index map
    size_t* __restrict__ sparseCount,         // Output: total sparse count
    size_t nGridPt)                           // Number of grid points
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    size_t sparseIdx = 0;
    for (size_t i = 0; i < nGridPt; i++) {
        if (gridOccupied[i]) {
            gridSparseMap[i] = sparseIdx;
            sparseIdx++;
        }
    }
    *sparseCount = sparseIdx;
}

/**
 * @brief GPU Kernel: Remap node indices to sparse grid
 */
GPU_KERNEL void kernel_remap_to_sparse(
    const size_t* __restrict__ nodeMapGrid,    // Input: original grid mapping
    const size_t* __restrict__ gridSparseMap,  // Input: dense -> sparse map
    size_t* __restrict__ nodeMapGridSparse,    // Output: sparse grid mapping
    size_t nNodePt)                            // Number of mesh nodes
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nNodePt) return;
    
    size_t gridIdx = nodeMapGrid[idx];
    nodeMapGridSparse[idx] = gridSparseMap[gridIdx];
}

/**
 * @brief GPU Kernel: Create sparse cluster array
 */
GPU_KERNEL void kernel_create_sparse_clusters(
    const size_t* __restrict__ nCluster,       // Input: full cluster counts
    const uint8_t* __restrict__ gridOccupied,  // Input: occupied flags
    const size_t* __restrict__ gridSparseMap,  // Input: dense -> sparse map
    size_t* __restrict__ nClusterSparse,       // Output: sparse cluster counts
    size_t nGridPt)                            // Number of grid points
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nGridPt) return;
    
    if (gridOccupied[idx]) {
        nClusterSparse[gridSparseMap[idx]] = nCluster[idx];
    }
}

#endif // GPU_ENABLED

// =============================================================================
// Host Wrapper Functions
// =============================================================================

/**
 * @brief GPU-accelerated closest node to uniform grid mapping
 * Only uses GPU for very large datasets where the parallel speedup exceeds transfer overhead
 * For typical mesh sizes (<10M nodes), CPU is faster
 */
inline void closest_Node2UniformGrid_GPU(
    std::vector<size_t>& nodeMapGrid,
    const std::vector<std::vector<double>>& nodeCoord,
    const std::vector<size_t>& resampleRate,
    const std::vector<double>& minvGrid,
    const std::vector<double>& spaceGrid,
    bool useGPU = true)
{
    size_t nNodePt = nodeCoord[0].size();
    int n_dims = resampleRate.size();
    
    // GPU only beneficial for very large datasets (>10M nodes)
    // For smaller sizes, memory transfer overhead dominates
    constexpr size_t GPU_THRESHOLD = 10000000;
    
#if GPU_ENABLED
    if (useGPU && isGPUAvailable() && nNodePt > GPU_THRESHOLD) {
        
        // Allocate device memory
        size_t* d_nodeMapGrid = nullptr;
        double* d_nodeCoordX = nullptr;
        double* d_nodeCoordY = nullptr;
        double* d_nodeCoordZ = nullptr;
        size_t* d_resampleRate = nullptr;
        double* d_minvGrid = nullptr;
        double* d_spaceGrid = nullptr;
        
        GPU_CHECK(gpuMalloc(&d_nodeMapGrid, nNodePt * sizeof(size_t)));
        GPU_CHECK(gpuMalloc(&d_nodeCoordX, nNodePt * sizeof(double)));
        GPU_CHECK(gpuMalloc(&d_resampleRate, n_dims * sizeof(size_t)));
        GPU_CHECK(gpuMalloc(&d_minvGrid, n_dims * sizeof(double)));
        GPU_CHECK(gpuMalloc(&d_spaceGrid, n_dims * sizeof(double)));
        
        if (n_dims >= 2) {
            GPU_CHECK(gpuMalloc(&d_nodeCoordY, nNodePt * sizeof(double)));
        }
        if (n_dims >= 3) {
            GPU_CHECK(gpuMalloc(&d_nodeCoordZ, nNodePt * sizeof(double)));
        }
        
        // Copy data to device
        GPU_CHECK(gpuMemset(d_nodeMapGrid, 0, nNodePt * sizeof(size_t)));
        GPU_CHECK(gpuMemcpy(d_nodeCoordX, nodeCoord[0].data(), nNodePt * sizeof(double), gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(d_resampleRate, resampleRate.data(), n_dims * sizeof(size_t), gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(d_minvGrid, minvGrid.data(), n_dims * sizeof(double), gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(d_spaceGrid, spaceGrid.data(), n_dims * sizeof(double), gpuMemcpyHostToDevice));
        
        if (n_dims >= 2) {
            GPU_CHECK(gpuMemcpy(d_nodeCoordY, nodeCoord[1].data(), nNodePt * sizeof(double), gpuMemcpyHostToDevice));
        }
        if (n_dims >= 3) {
            GPU_CHECK(gpuMemcpy(d_nodeCoordZ, nodeCoord[2].data(), nNodePt * sizeof(double), gpuMemcpyHostToDevice));
        }
        
        // Launch kernel
        int blockSize = 256;
        int numBlocks = (nNodePt + blockSize - 1) / blockSize;
        
        kernel_closest_Node2UniformGrid<<<numBlocks, blockSize>>>(
            d_nodeMapGrid, d_nodeCoordX, d_nodeCoordY, d_nodeCoordZ,
            d_resampleRate, d_minvGrid, d_spaceGrid, nNodePt, n_dims);
        
        GPU_CHECK(gpuDeviceSynchronize());
        
        // Copy result back
        GPU_CHECK(gpuMemcpy(nodeMapGrid.data(), d_nodeMapGrid, nNodePt * sizeof(size_t), gpuMemcpyDeviceToHost));
        
        // Free device memory
        GPU_CHECK(gpuFree(d_nodeMapGrid));
        GPU_CHECK(gpuFree(d_nodeCoordX));
        GPU_CHECK(gpuFree(d_resampleRate));
        GPU_CHECK(gpuFree(d_minvGrid));
        GPU_CHECK(gpuFree(d_spaceGrid));
        if (d_nodeCoordY) GPU_CHECK(gpuFree(d_nodeCoordY));
        if (d_nodeCoordZ) GPU_CHECK(gpuFree(d_nodeCoordZ));
        
        return;
    }
#endif
    
    // CPU fallback (also used for datasets < 10M nodes)
    std::vector<size_t> dims(n_dims, 1);
    for (int d = 1; d < n_dims; d++) {
        dims[d] = dims[d-1] * resampleRate[d-1];
    }
    
    for (size_t i = 0; i < nNodePt; i++) {
        nodeMapGrid[i] = 0;
        for (int d = 0; d < n_dims; d++) {
            double posVal = nodeCoord[d][i] - minvGrid[d];
            size_t dimIdx = (size_t)std::floor(posVal / spaceGrid[d]);
            if (dimIdx >= resampleRate[d]) dimIdx = resampleRate[d] - 1;
            nodeMapGrid[i] += dimIdx * dims[d];
        }
    }
}

/**
 * @brief Grid sparsity check - CPU only
 * The sparse map computation is inherently sequential (prefix sum pattern)
 * GPU version would require parallel scan which adds complexity without benefit
 * for typical grid sizes
 */
inline void check_GridSparsity_GPU(
    const std::vector<size_t>& nodeMapGrid,
    size_t nGridPt,
    size_t& nGridSparse,
    std::vector<size_t>& GridSparseMap,
    std::vector<size_t>& nCluster,
    bool /*useGPU*/ = true)  // GPU not used - always CPU for this operation
{
    constexpr double threshGrid = 0.85;
    size_t nNodePt = nodeMapGrid.size();
    
    // CPU implementation - GPU not beneficial here due to sequential sparse map computation
    nGridSparse = 0;
    std::vector<bool> gridPointVal(nGridPt, false);
    
    for (size_t i = 0; i < nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        nCluster[k]++;
        gridPointVal[k] = true;
    }
    
    for (size_t i = 0; i < nGridPt; i++) {
        if (gridPointVal[i]) nGridSparse++;
    }
    
    double nonZerosRate = (double)nGridSparse / (double)nGridPt;
    if (nonZerosRate < threshGrid) {
        size_t sparseId = 0;
        for (size_t i = 0; i < nGridPt; i++) {
            if (gridPointVal[i]) {
                GridSparseMap[i] = sparseId;
                sparseId++;
            }
        }
    } else {
        nGridSparse = nGridPt;
    }
}

/**
 * @brief Print GPU info
 */
inline void printGPUInfo()
{
#if GPU_ENABLED
    int deviceCount = getGPUDeviceCount();
    if (deviceCount > 0) {
        std::cout << "GPU Backend: ";
#if GPU_BACKEND_HIP
        std::cout << "HIP (AMD)";
#elif GPU_BACKEND_CUDA
        std::cout << "CUDA (NVIDIA)";
#endif
        std::cout << ", Devices: " << deviceCount << std::endl;
        std::cout << "Note: GPU only used for node mapping with >10M nodes" << std::endl;
    } else {
        std::cout << "No GPU devices found, using CPU" << std::endl;
    }
#else
    std::cout << "CPU-only build (no GPU support compiled)" << std::endl;
#endif
}

} // namespace gpu
} // namespace meshgrid

#endif // MESHGRID_GPU_HPP
