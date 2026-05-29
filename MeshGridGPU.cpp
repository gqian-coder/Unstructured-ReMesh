/*
 * MeshGridGPU.cpp
 * GPU-accelerated mesh-to-grid mapping computation
 * Portable across NVIDIA (CUDA), AMD (HIP), and CPU backends
 *
 *  Created on: Feb 4, 2026
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "adios2.h"
#include "nonUniformMap.hpp"
#include "MeshGridGPU.hpp"
#include <time.h>
#include <zstd.h>

int rank, np_size;
MPI_Comm comm;

// Command-line option for GPU usage
bool g_useGPU = true;

void ReadInfo(std::string &inputFileName, size_t *ndim, size_t *nblocks)
{
    adios2::ADIOS ad(comm);
    adios2::IO io = ad.DeclareIO("InputOnce");
    io.SetParameter("SelectSteps", "0");
    auto e = io.Open(inputFileName, adios2::Mode::ReadRandomAccess);
    auto var = io.InquireVariable<int32_t>("/hpMusic_base/physdim");
    if (!var)
    {
        throw std::invalid_argument("The input file " + inputFileName +
                                    " does not have a variable int32_t /hpMusic_base/physdim");
    }
    int32_t idim;
    e.Get(var, &idim, adios2::Mode::Sync);
    *ndim = (size_t)idim;

    auto varX =
        io.InquireVariable<double>("/hpMusic_base/hpMusic_Zone/GridCoordinates/CoordinateX");
    if (!varX)
    {
        throw std::invalid_argument("The input file " + inputFileName +
                                    " does not have a variable double "
                                    "/hpMusic_base/hpMusic_Zone/GridCoordinates/CoordinateX");
    }
    auto bi = e.BlocksInfo(varX, 0);
    *nblocks = bi.size();
    e.Close();
}

void printUsage(const char* progName)
{
    std::cerr << "Usage: " << progName << " [options] input output percentile\n"
              << "\nRequired arguments:\n"
              << "  input      : sol or sol_aver BP file with mesh in it\n"
              << "  output     : mesh to grid mapping file\n"
              << "  percentile : of node spacing for grid selection (0.0-1.0)\n"
              << "\nOptions:\n"
              << "  --gpu      : Use GPU acceleration (default)\n"
              << "  --cpu      : Use CPU only\n"
              << "  --help     : Show this help message\n"
              << std::endl;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np_size);

    // Parse command line arguments
    std::vector<std::string> positionalArgs;
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "--gpu")
        {
            g_useGPU = true;
        }
        else if (arg == "--cpu")
        {
            g_useGPU = false;
        }
        else if (arg == "--help" || arg == "-h")
        {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return EXIT_SUCCESS;
        }
        else
        {
            positionalArgs.push_back(arg);
        }
    }

    if (positionalArgs.size() < 3)
    {
        if (!rank) printUsage(argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    std::string solFile = positionalArgs[0];
    std::string mappingFile = positionalArgs[1];
    double perc = std::stof(positionalArgs[2]);

    if (!rank)
    {
        std::cout << "=== MeshGrid GPU ===\n";
        meshgrid::gpu::printGPUInfo();
        std::cout << "GPU acceleration: " << (g_useGPU ? "enabled" : "disabled") << "\n";
        std::cout << "Read in : " << solFile << "\n";
        std::cout << "Write   : " << mappingFile << "\n";
        std::cout << "Percentile of grid spacing: " << perc << "\n";
    }

    size_t maxBlocks;
    size_t n_dims;
    ReadInfo(solFile, &n_dims, &maxBlocks);
    if (!rank)
    {
        std::cout << "ndim    : " << n_dims << "\n";
        std::cout << "nblocks : " << maxBlocks << "\n";
    }

    std::vector<size_t> resampleRate(n_dims, 1);

    adios2::ADIOS ad(comm);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::Engine reader = reader_io.Open(solFile, adios2::Mode::Read);
    adios2::IO writer_io = ad.DeclareIO("Output");
    adios2::Engine writer = writer_io.Open(mappingFile, adios2::Mode::Write);

    std::vector<adios2::Variable<double>> var_coord(n_dims);
    adios2::Variable<int64_t> var_connc;
    std::vector<double> spaceGrid(n_dims);
    std::vector<double> minvGrid(n_dims);
    std::vector<int64_t> nodeConnc(0);

    // output variables
    adios2::Variable<size_t> var_map, var_cluster;
    adios2::Variable<size_t> var_gridDim;
    adios2::Variable<uint8_t> var_sparse;
    var_map = writer_io.DefineVariable<size_t>("__mesh_grid_mapping__/MeshGridMap", {}, {},
                                               {adios2::UnknownDim});
    var_cluster = writer_io.DefineVariable<size_t>("__mesh_grid_mapping__/MeshGridCluster", {}, {},
                                                   {adios2::UnknownDim});
    var_gridDim = writer_io.DefineVariable<size_t>("__mesh_grid_mapping__/GridDim", {}, {},
                                                   {adios2::UnknownDim});
    var_sparse = writer_io.DefineVariable<uint8_t>("__mesh_grid_mapping__/GridSparsity", {}, {},
                                                   {adios2::UnknownDim});

    size_t nNodePt, nSGridPt;
    
    // Timing variables
    double total_mapping_time = 0.0;
    double total_sparsity_time = 0.0;

    while (true)
    {
        // Begin step
        adios2::StepStatus read_status = reader.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK)
        {
            break;
        }
        writer.BeginStep();

        // read node coordinates
        std::vector<std::string> coordVarName{"CoordinateZ", "CoordinateX", "CoordinateY"};
        for (size_t i = 0; i < n_dims; i++)
        {
            var_coord[i] = reader_io.InquireVariable<double>(
                "/hpMusic_base/hpMusic_Zone/GridCoordinates/" + coordVarName[i]);
        }
        var_connc = reader_io.InquireVariable<int64_t>(
            "/hpMusic_base/hpMusic_Zone/Elem/ElementConnectivity");

        auto bi = reader.BlocksInfo(var_coord[0], 0);
        uint8_t sparsity;
        size_t nBlocks = std::min(bi.size(), maxBlocks);
        size_t local_nBlocks = (size_t)std::ceil((double)nBlocks / (double)np_size);
        size_t blockId = rank * local_nBlocks;
        size_t last_Block = std::min(nBlocks, blockId + local_nBlocks);
        std::cout << "Rank " << rank << ": processing blocks " << blockId << " to " << last_Block - 1 << " (total " << nBlocks << " blocks)\n";

        while (blockId < last_Block)
        {
            std::cout << "Block " << blockId << ": ";
            size_t nGridPt = 1;
            std::vector<std::vector<double>> nodeCoord;
            for (size_t i = 0; i < n_dims; i++)
            {
                var_coord[i].SetBlockSelection(blockId);
                std::vector<double> var_in;
                reader.Get<double>(var_coord[i], var_in, adios2::Mode::Sync);
                reader.PerformGets();
                nodeCoord.push_back(var_in);
            }
            var_connc.SetBlockSelection(blockId);
            reader.Get<int64_t>(var_connc, nodeConnc, adios2::Mode::Sync);
            reader.PerformGets();
            nNodePt = nodeCoord[0].size();

            // Calculate the mesh to grid mapping using CPU function
            // (sel_Gridspace involves heap operations, not well suited for GPU)
            sel_Gridspace(nodeConnc, nodeCoord, n_dims, perc, spaceGrid, resampleRate);
            
            std::cout << "grid size: ";
            for (size_t i = 0; i < n_dims; i++)
            {
                nGridPt = nGridPt * resampleRate[i];
                std::cout << resampleRate[i];
                if (i < n_dims - 1)
                    std::cout << " x ";
            }
            std::cout << " = " << nGridPt << " points\n";

            for (size_t i = 0; i < n_dims; i++)
            {
                minvGrid[i] = *std::min_element(nodeCoord[i].begin(), nodeCoord[i].end());
            }
            
            std::vector<size_t> nodeMapGrid(nNodePt, 0);
            std::vector<size_t> GridSparseMap(nGridPt, nGridPt);
            std::vector<size_t> nCluster(nGridPt, 0);

            // GPU-accelerated closest node to grid mapping
            auto t_start = std::chrono::high_resolution_clock::now();
            meshgrid::gpu::closest_Node2UniformGrid_GPU(
                nodeMapGrid, nodeCoord, resampleRate, minvGrid, spaceGrid, g_useGPU);
            auto t_end = std::chrono::high_resolution_clock::now();
            double mapping_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            total_mapping_time += mapping_time;
            
            // GPU-accelerated sparsity check
            t_start = std::chrono::high_resolution_clock::now();
            meshgrid::gpu::check_GridSparsity_GPU(
                nodeMapGrid, nGridPt, nSGridPt, GridSparseMap, nCluster, g_useGPU);
            t_end = std::chrono::high_resolution_clock::now();
            double sparsity_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            total_sparsity_time += sparsity_time;
            
            std::cout << "  Mapping time: " << mapping_time << " ms, Sparsity check: " << sparsity_time << " ms\n";
            std::cout << "  Mesh nodes: " << nNodePt << ", Grid points: " << nGridPt 
                      << ", Sparse: " << nSGridPt << " (" 
                      << (100.0 * nSGridPt / nGridPt) << "%)\n";

            var_map.SetSelection(adios2::Box<adios2::Dims>({}, {nNodePt}));
            if (nSGridPt == nGridPt)
            {
                std::cout << "  Using high-dim compression (dense grid)\n";
                var_cluster.SetSelection(adios2::Box<adios2::Dims>({}, {nGridPt}));
                writer.Put<size_t>(var_map, nodeMapGrid.data(), adios2::Mode::Sync);
                writer.Put<size_t>(var_cluster, nCluster.data(), adios2::Mode::Sync);
                sparsity = 0;
            }
            else
            {
                std::cout << "  Using sparse grid compression\n";
                std::vector<size_t> nodeMapGridSparse(nNodePt, 0);
                std::vector<size_t> nClusterSparse(nSGridPt, 0);
                var_cluster.SetSelection(adios2::Box<adios2::Dims>({}, {nSGridPt}));
                
                // Remap to sparse indices
                for (size_t i = 0; i < nNodePt; i++)
                {
                    nodeMapGridSparse[i] = GridSparseMap[nodeMapGrid[i]];
                }
                for (size_t i = 0; i < nGridPt; i++)
                {
                    if (nCluster[i] > 0)
                    {
                        nClusterSparse[GridSparseMap[i]] = nCluster[i];
                    }
                }
                writer.Put<size_t>(var_map, nodeMapGridSparse.data(), adios2::Mode::Sync);
                writer.Put<size_t>(var_cluster, nClusterSparse.data(), adios2::Mode::Sync);
                nodeMapGridSparse.clear();
                nClusterSparse.clear();
                sparsity = 1;
            }

            var_gridDim.SetSelection(adios2::Box<adios2::Dims>({}, {n_dims}));
            writer.Put<size_t>(var_gridDim, resampleRate.data(), adios2::Mode::Sync);
            var_sparse.SetSelection(adios2::Box<adios2::Dims>({}, {1}));
            writer.Put<uint8_t>(var_sparse, &sparsity, adios2::Mode::Sync);
            writer.PerformPuts();

            // clear the memory for the next block of data
            nodeMapGrid.clear();
            GridSparseMap.clear();
            nCluster.clear();
            for (size_t i = 0; i < n_dims; i++)
                nodeCoord[i].clear();
            nodeCoord.clear();
            blockId++;
        }
        writer.EndStep();
        reader.EndStep();
        break;
    }
    reader.Close();
    writer.Close();

    // Print timing summary
    if (!rank)
    {
        std::cout << "\n=== Timing Summary ===\n";
        std::cout << "Total mapping time: " << total_mapping_time << " ms\n";
        std::cout << "Total sparsity check time: " << total_sparsity_time << " ms\n";
        std::cout << "Total compute time: " << (total_mapping_time + total_sparsity_time) << " ms\n";
    }

    MPI_Finalize();
    return 0;
}
