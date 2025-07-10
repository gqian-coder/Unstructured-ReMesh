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
#include <time.h>
#include <zstd.h>

int rank, np_size;
MPI_Comm comm;

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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np_size);

    // Parse command line arguments
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << "  input  output  percentile\n"
                  << "  input     :  sol or sol_aver BP file with mesh in it\n"
                  << "  output    :  mesh to grid mapping file \n"
                  << "  percentile: of node spacing for grid selection" << std::endl;
        return EXIT_FAILURE;
    }
    int cnt_argv = 1;
    std::string solFile(argv[cnt_argv++]);
    std::string mappingFile(argv[cnt_argv++]);
    double perc = std::stof(argv[cnt_argv++]);

    if (!rank)
    {
        std::cout << "Read in : " << solFile << "\n";
        std::cout << "Write   : " << mappingFile << "\n";
        std::cout << "percentile of grid spacing used for resample rate calculation: " << perc
                  << "\n";
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
    adios2::Variable<char> var_sparse;
    var_map = writer_io.DefineVariable<size_t>("__mesh_grid_mapping__/MeshGridMap", {}, {},
                                               {adios2::UnknownDim});
    var_cluster = writer_io.DefineVariable<size_t>("__mesh_grid_mapping__/MeshGridCluster", {}, {},
                                                   {adios2::UnknownDim});
    var_gridDim = writer_io.DefineVariable<size_t>("__mesh_grid_mapping__/GridDim", {}, {},
                                                   {adios2::UnknownDim});
    var_sparse = writer_io.DefineVariable<char>("__mesh_grid_mapping__/GridSparsity", {}, {},
                                                {adios2::UnknownDim});

    size_t nNodePt, nSGridPt;
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
        char sparsity;
        size_t nBlocks = std::min(bi.size(), maxBlocks);
        size_t local_nBlocks = (size_t)std::ceil((double)nBlocks / (double)np_size);
        size_t blockId = rank * local_nBlocks;
        size_t last_Block = std::min(nBlocks, blockId + local_nBlocks);
        std::cout << "data has " << nBlocks << " blocks\n";
        while (blockId < last_Block)
        {
            std::cout << "blockID = " << blockId << "\n";
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

            // calculate the mesh to grid mapping
            sel_Gridspace(nodeConnc, nodeCoord, n_dims, perc, spaceGrid, resampleRate);
            std::cout << "resampled grid size: ";
            for (size_t i = 0; i < n_dims; i++)
            {
                nGridPt = nGridPt * resampleRate[i];
                std::cout << resampleRate[i];
                if (i < n_dims - 1)
                    std::cout << " x ";
            }
            std::cout << ", at spacing: ";
            for (size_t i = 0; i < n_dims; i++)
                std::cout << spaceGrid[i] << " , ";
            std::cout << "\n";
            std::cout << "number of structured mesh nodes: " << nNodePt << "\n";
            std::cout << "Resample rate: " << (double)nGridPt / nNodePt << "\n";
            for (size_t i = 0; i < n_dims; i++)
            {
                minvGrid[i] = *std::min_element(nodeCoord[i].begin(), nodeCoord[i].end());
            }
            std::vector<size_t> nodeMapGrid(nNodePt, 0);
            std::vector<size_t> GridSparseMap(nGridPt, nGridPt);
            std::vector<size_t> nCluster(nGridPt, 0);

            closest_Node2UniformGrid(nodeMapGrid, nodeCoord, resampleRate, minvGrid, spaceGrid);
            check_GridSparsity(nodeMapGrid, nGridPt, nSGridPt, GridSparseMap, nCluster);

            var_map.SetSelection(adios2::Box<adios2::Dims>({}, {nNodePt}));
            if (nSGridPt == nGridPt)
            {
                std::cout << "high-dim compression\n";
                var_cluster.SetSelection(adios2::Box<adios2::Dims>({}, {nGridPt}));
                writer.Put<size_t>(var_map, nodeMapGrid.data(), adios2::Mode::Sync);
                writer.Put<size_t>(var_cluster, nCluster.data(), adios2::Mode::Sync);
                sparsity = 0;
            }
            else
            {
                std::vector<size_t> nodeMapGridSparse(nNodePt, 0);
                std::vector<size_t> nClusterSparse(nSGridPt, 0);
                var_cluster.SetSelection(adios2::Box<adios2::Dims>({}, {nSGridPt}));
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
            writer.Put<char>(var_sparse, &sparsity, adios2::Mode::Sync);
            writer.PerformPuts();

            std::cout << "resampled grid size = " << (float)nSGridPt / (float)nNodePt
                      << "X of the original mesh nodes\n";
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

    return 0;
}
