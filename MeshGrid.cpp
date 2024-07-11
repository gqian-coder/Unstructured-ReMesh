#include <cmath>
#include <fstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <dirent.h>

#include "adios2.h"
#include <zstd.h>
#include "nonUniformMap.hpp"
#include <time.h>

int main(int argc, char **argv) {
    // Parse command line arguments
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " dataPath"  << " filedName" ;
        std::cerr << " n-Dims " ;
        std::cerr << " percentile of node spacing for grid selection" << std::endl;
        return EXIT_FAILURE;
    }
    int cnt_argv = 1;
    std::string dpath(argv[cnt_argv++]);
    std::string fname(argv[cnt_argv++]);
    std::cout << "Read in : " << dpath + fname << "\n";
    
    // dimensions used for compression
    size_t n_dims = (size_t)std::stoi(argv[cnt_argv++]);
    std::vector<size_t>resampleRate(n_dims, 1);
    
    double perc = std::stof(argv[cnt_argv++]);
    std::cout << "percentile of grid spacing used for resample rate calculation: " << perc << "\n";

    adios2::ADIOS ad; 
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::Engine reader = reader_io.Open(dpath + fname, adios2::Mode::Read);
    adios2::IO writer_io = ad.DeclareIO("Output");
    adios2::Engine writer = writer_io.Open("Mesh2GridMap.bp", adios2::Mode::Write);

    std::vector<adios2::Variable<double>> var_coord(n_dims);
    adios2::Variable<int64_t> var_connc;
    std::vector<double> spaceGrid(n_dims);
    std::vector<double> minvGrid(n_dims);
    std::vector<int64_t> nodeConnc(0);
    std::vector<std::vector<double>> nodeCoord;

    // output variables
    adios2::Variable<size_t> var_map, var_cluster;
    adios2::Variable<double> var_space;
    adios2::Variable<char> var_sparse;
    var_map     = writer_io.DefineVariable<size_t>("MeshGridMap", {}, {}, {adios2::UnknownDim});
    var_cluster = writer_io.DefineVariable<size_t>("MeshGridCluster", {}, {}, {adios2::UnknownDim});
    var_space   = writer_io.DefineVariable<double>("GridCoordSpace", {}, {}, {adios2::UnknownDim});
    var_sparse  = writer_io.DefineVariable<char>("GridSparsity", {}, {}, {adios2::UnknownDim});

    size_t nNodePt, nSGridPt;
    while (true) {
        // Begin step
        adios2::StepStatus read_status = reader.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK) {
            break;
        }
        writer.BeginStep();

        // read node coordinates
        std::vector<std::string> coordVarName{"CoordinateZ", "CoordinateX", "CoordinateY"};
        for (size_t i=0; i<n_dims; i++) {
            var_coord[i] = reader_io.InquireVariable<double>("/hpMusic_base/hpMusic_Zone/GridCoordinates/"+coordVarName[i]);
        }
        var_connc = reader_io.InquireVariable<int64_t>("/hpMusic_base/hpMusic_Zone/Elem/ElementConnectivity");

        auto bi = reader.BlocksInfo(var_coord[0], 0);  
        size_t nBlocks = bi.size();
        std::cout << "data has " << nBlocks << " blocks\n";
        char sparsity;
        for (auto &info : bi) {
            std::cout << "blockID = " << info.BlockID << "\n";
            size_t nGridPt = 1;
            for (size_t i=0; i<n_dims; i++) {
                var_coord[i].SetBlockSelection(info.BlockID);
                std::vector<double> var_in;
                reader.Get<double>(var_coord[i], var_in, adios2::Mode::Sync);
                reader.PerformGets();
                nodeCoord.push_back(var_in);
            }
            var_connc.SetBlockSelection(info.BlockID);
            reader.Get<int64_t>(var_connc, nodeConnc, adios2::Mode::Sync);
            reader.PerformGets();
            nNodePt = nodeCoord[0].size();
            std::cout << "nNodePt = " << nNodePt << "\n";

            // calculate the mesh to grid mapping
            sel_Gridspace(nodeConnc, nodeCoord, n_dims, perc, spaceGrid, resampleRate);
            std::cout << "resample rate: ";
            for (size_t i=0; i<n_dims; i++) {
                nGridPt = nGridPt * resampleRate[i];
                std::cout << resampleRate[i];
                if (i<n_dims-1) std::cout << " x ";
            }
            std::cout << ", at spacing: ";
            for (size_t i=0; i<n_dims; i++) std::cout << spaceGrid[i] << " , ";
            std::cout << "\n";
            std::cout << "number of structured mesh nodes: " << nNodePt << "\n";
            std::cout << "Resample rate: " << (double)nGridPt / nNodePt << "\n";
            for (size_t i=0; i<n_dims; i++) {
                minvGrid[i] = *std::min_element(nodeCoord[i].begin(), nodeCoord[i].end());
            }
            std::vector <size_t> nodeMapGrid(nNodePt, 0);
            std::vector<size_t> GridSparseMap(nGridPt, nGridPt);
            std::vector<size_t> nCluster(nGridPt, 0);

            closest_Node2UniformGrid(nodeMapGrid, nodeCoord, resampleRate, minvGrid, spaceGrid);
            check_GridSparsity(nodeMapGrid, nGridPt, nSGridPt, GridSparseMap, nCluster);

            var_map.SetSelection(adios2::Box<adios2::Dims>({}, {nNodePt}));
            if (nSGridPt==nGridPt) {
                var_cluster.SetSelection(adios2::Box<adios2::Dims>({}, {nGridPt}));
                writer.Put<size_t>(var_map, nodeMapGrid.data(), adios2::Mode::Sync);
                writer.Put<size_t>(var_cluster, nCluster.data(), adios2::Mode::Sync);
                sparsity = 0; 
            } else {
                std::vector <size_t> nodeMapGridSparse(nNodePt, 0);
                std::vector<size_t> nClusterSparse(nSGridPt, 0);
                var_cluster.SetSelection(adios2::Box<adios2::Dims>({}, {nSGridPt}));
                for (size_t i=0; i<nNodePt; i++) {
                    nodeMapGridSparse[i] = GridSparseMap[nodeMapGrid[i]];
                }
                for (size_t i=0; i<nGridPt; i++) {
                    if (nCluster[i]>0) {
                        nClusterSparse[GridSparseMap[i]] = nCluster[i];
                    }
                }
                writer.Put<size_t>(var_map    , nodeMapGridSparse.data(), adios2::Mode::Sync);
                writer.Put<size_t>(var_cluster, nClusterSparse.data(), adios2::Mode::Sync);
                nodeMapGridSparse.clear();
                nClusterSparse.clear();
                sparsity = 1;
            }

            var_space.SetSelection(adios2::Box<adios2::Dims>({}, {n_dims}));
            writer.Put<double>(var_space, spaceGrid.data(), adios2::Mode::Sync);
            var_sparse.SetSelection(adios2::Box<adios2::Dims>({}, {1}));
            writer.Put<char>(var_sparse, &sparsity, adios2::Mode::Sync);
            writer.PerformPuts(); 
            
            std::cout << "resampled grid size = " << nSGridPt/nNodePt << "X of the original mesh nodes\n";
            // clear the memory for the next block of data
            nodeMapGrid.clear();
            GridSparseMap.clear();
            nCluster.clear();
        }
        writer.EndStep();
    	reader.EndStep();
    }
    reader.Close();
    writer.Close();

    return 0;
}







