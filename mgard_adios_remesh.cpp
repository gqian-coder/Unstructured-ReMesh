#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <dirent.h>

#include "adios2.h"
#include "mgard/compress_x.hpp"
#include "nonUniformMap.hpp"
#include <zstd.h>
#include <time.h>
#include <chrono> 

string to_string_ld(long double number) {
    long double temp = number;
    long double integerPart = floor(temp);
    int fractionCounter = 0;

    while (temp - integerPart > 0) {
        temp = temp * 10;
        integerPart = floor(temp);
        fractionCounter++;
    }

    fractionCounter = std::max(1, fractionCounter);

    std::stringstream stream;
    stream.precision(fractionCounter);
    stream << std::fixed << number;
    return stream.str();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    int cnt_argv = 1;
    std::string dpath(argv[cnt_argv++]);
    std::string fname(argv[cnt_argv++]);
    std::string map_name(argv[cnt_argv++]);
    int n_vars = std::stoi(argv[cnt_argv++]);
    std::vector<std::string> var_name(n_vars);
    for (int i=0; i< n_vars; i++) {
        var_name[i] = argv[cnt_argv++];
    }
    double tol     = std::stof(argv[cnt_argv++]);
    double ratio_t = std::stof(argv[cnt_argv++]);

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io   = ad.DeclareIO("Input");
    adios2::IO reader_io_m = ad.DeclareIO("InputMap");
    adios2::IO writer_io   = ad.DeclareIO("Output");

    if (rank==0) {
        std::cout << "write: " << "./" + fname + ".compressed" << "\n";
        std::cout << "readin: " << dpath + fname << "\n";
    }
    adios2::Engine reader      = reader_io.Open(dpath + fname, adios2::Mode::Read);
    adios2::Engine reader_mesh = reader_io_m.Open(map_name, adios2::Mode::Read);
    adios2::Engine writer = writer_io.Open(fname + ".remshCompressed", adios2::Mode::Write);

    adios2::Variable<size_t> var_map, var_cluster, var_gridDim;
    adios2::Variable<double> var_ad2;
    adios2::Variable<char> var_sparse;
    
    // interpolated data on grid nodes
    std::vector<adios2::Variable<double>> var_gd(n_vars);
    // residuals on mesh nodes
    std::vector<adios2::Variable<double>> var_rs(n_vars);
    for (int i=0; i<n_vars; i++) {
    	var_gd[i] = writer_io.DefineVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/" + var_name[i] + "_gridData", {}, {}, {adios2::UnknownDim});
        var_rs[i] = writer_io.DefineVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/" + var_name[i] + "_meshResi", {}, {}, {adios2::UnknownDim});
    }

    adios2::Operator op = ad.DefineOperator("mgard", "mgard");
    size_t ts = 0;
    double minv, maxv;
    double abs_tol, tol_data, tol_resi;
    double time_s = 0.0;
    while (true) {
        // Begin step
        adios2::StepStatus read_status = reader_mesh.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady) {
            // std::cout << "Stream not ready yet. Waiting...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK) {
            break;
        }
        reader.BeginStep(adios2::StepMode::Read, 10.0f);
    	writer.BeginStep();

        size_t step = reader.CurrentStep();
        if (rank==0) std::cout << "Process step " << step << ": " << std::endl;
        // read mesh mapping info
        var_map     = reader_io_m.InquireVariable<size_t>("MeshGridMap");    
        var_cluster = reader_io_m.InquireVariable<size_t>("MeshGridCluster");
        var_gridDim = reader_io_m.InquireVariable<size_t>("GridDim");  
        var_sparse  = reader_io_m.InquireVariable<char>("GridSparsity");

        auto bi = reader_mesh.BlocksInfo(var_map, ts);
        size_t nBlocks = bi.size();
        std::cout << "number of blocks in step " << ts << ": " << nBlocks << "\n";

        // set error bounds based on global value range -- across all blocks
        for (int i=0; i<n_vars; i++) {
            var_ad2 = reader_io.InquireVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/"+var_name[i]);
            minv = var_ad2.Min();
            maxv = var_ad2.Max();
            abs_tol = tol * (maxv-minv);
            if (rank==0) std::cout << var_name[i].c_str() << ": min/max = "<< minv << "/" << maxv << ", tol = "<< abs_tol << std::endl;
            tol_data = abs_tol * ratio_t;
            tol_resi = abs_tol * (1-ratio_t);
            var_gd[i].AddOperation(op, {{"tolerance", to_string_ld(tol_data)}, {"mode", "ABS"}});
            var_rs[i].AddOperation(op, {{"tolerance", to_string_ld(tol_resi)}, {"mode", "ABS"}});
        }
        //auto start = std::chrono::high_resolution_clock::now();
        for (auto &info : bi) {
            std::cout << "blockID = " << info.BlockID << "\n";
            std::vector<size_t> nodeMapGrid, nCluster, resampleRate;
            char sparsity;
            var_map.SetBlockSelection(info.BlockID);
            var_cluster.SetBlockSelection(info.BlockID);
            var_gridDim.SetBlockSelection(info.BlockID);
            var_sparse.SetBlockSelection(info.BlockID);
            reader_mesh.Get<size_t>(var_map    , nodeMapGrid , adios2::Mode::Sync);
            reader_mesh.Get<size_t>(var_cluster, nCluster    , adios2::Mode::Sync);
            reader_mesh.Get<size_t>(var_gridDim, resampleRate, adios2::Mode::Sync);
            reader_mesh.Get<char>(var_sparse , &sparsity   , adios2::Mode::Sync);
            reader_mesh.PerformGets(); 
        
            size_t nNodePt = nodeMapGrid.size();
            size_t nGridPt = nCluster.size();
            std::vector<double> GridPointVal(nGridPt);
            std::cout << "number of mesh nodes: " << nNodePt << ", number of grid nodes: " << nGridPt << "\n";

            for (int i=0; i<n_vars; i++) {
                std::cout << "compress " << var_name[i] << "\n";
                std::vector<double> var_in;
                std::fill(GridPointVal.begin(), GridPointVal.end(), 0);
                var_ad2 = reader_io.InquireVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/"+var_name[i]);
                var_ad2.SetBlockSelection(info.BlockID);
 
                reader.Get(var_ad2, var_in, adios2::Mode::Sync);
                reader.PerformGets();

                auto start = std::chrono::high_resolution_clock::now();
                // calculate the residuals and grid interpolation based on the mappings 
                // store the residual back to var_in
                calc_GridValResi(nodeMapGrid, nCluster, var_in, GridPointVal);
                // compress the residual and grid data separately
                var_rs[i].SetSelection(adios2::Box<adios2::Dims>({}, {nNodePt}));
                if (sparsity==0) {
                    std::cout << "resampleRate: " << resampleRate[0] << ", " << resampleRate[1] << ", " << resampleRate[2] << "\n";
                    var_gd[i].SetSelection(adios2::Box<adios2::Dims>({}, {resampleRate}));
                } else {
                    var_gd[i].SetSelection(adios2::Box<adios2::Dims>({}, {nGridPt}));
                }
                writer.Put<double>(var_rs[i], var_in.data()      , adios2::Mode::Sync);
                writer.Put<double>(var_gd[i], GridPointVal.data(), adios2::Mode::Sync);
                writer.PerformPuts();

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                time_s += (double)duration.count() / 1e6;

                var_in.clear();
            }
            nodeMapGrid.clear();
            nCluster.clear();
            resampleRate.clear();

            if (info.BlockID==5) break;
        }

        std::cout << "end\n"; 
        reader.EndStep();
        reader_mesh.EndStep();
    	writer.EndStep();
        ts ++;
    }
    reader.Close();
    reader_mesh.Close();
    writer.Close();

    MPI_Finalize();
    std::cout << "total time spent: " << time_s << " sec\n";
    return 0;
}
