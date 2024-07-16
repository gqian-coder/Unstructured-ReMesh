#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <dirent.h>

#include "adios2.h"
#include "mgard/compress_x.hpp"
#include <zstd.h>
#include <time.h>

#include "nonUniformMap.hpp"


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
    size_t nBlocks = (size_t)std::stoi(argv[cnt_argv++]); 
    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO reader_io_m = ad.DeclareIO("InputMap");
    adios2::IO writer_io = ad.DeclareIO("Output");

    if (rank==0) {
        std::cout << "write: " << "./" + fname + ".decompressed" << "\n";
        std::cout << "readin: " << dpath + fname << "\n";
    }
    adios2::Engine reader = reader_io.Open(dpath + fname, adios2::Mode::Read);
    adios2::Engine reader_mesh = reader_io_m.Open(dpath + map_name, adios2::Mode::Read);
    adios2::Engine writer = writer_io.Open(fname + ".remeshDecompressed", adios2::Mode::Write);

    adios2::Variable<size_t> var_map;

    size_t ts = 0;
    std::vector<adios2::Variable<double>> var_out(n_vars);
    for (int i=0; i<n_vars; i++) {
    	var_out[i] = writer_io.DefineVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/" + var_name[i], {}, {}, {adios2::UnknownDim});
    }

    while (true) {
        // Begin step
        adios2::StepStatus read_status = reader.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady) {
            // std::cout << "Stream not ready yet. Waiting...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK) {
            break;
        }
        reader_mesh.BeginStep(adios2::StepMode::Read, 10.0f);
    	writer.BeginStep();
        size_t step = reader.CurrentStep();
        if (rank==0) std::cout << "Process step " << step << ": " << std::endl;
        
        var_map = reader_io_m.InquireVariable<size_t>("MeshGridMap");
        auto bi = reader_mesh.BlocksInfo(var_map, ts);
        //size_t nBlocks = bi.size();

        for (auto &info : bi) {
            std::cout << "blockID = " << info.BlockID << "\n";
            var_map.SetBlockSelection(info.BlockID);
            std::vector<size_t> nodeMapGrid;
            reader_mesh.Get<size_t>(var_map, nodeMapGrid , adios2::Mode::Sync);
            reader_mesh.PerformGets();
            for (int i=0; i<n_vars; i++) {
                adios2::Variable<double> var_gd, var_rs;
                var_gd = reader_io.InquireVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/"+var_name[i] + "_gridData");
                var_rs = reader_io.InquireVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/"+var_name[i] + "_meshResi");
                std::cout << var_name[i].c_str() << " has " << nBlocks << " blocks\n";
                var_gd.SetBlockSelection(info.BlockID);
                var_rs.SetBlockSelection(info.BlockID);
                std::vector<double> GridPointVal, combinedVal; 
                reader.Get(var_gd, GridPointVal, adios2::Mode::Sync);
                reader.Get(var_rs, combinedVal , adios2::Mode::Sync);
                reader.PerformGets();

                size_t nNodePt = combinedVal.size();
		        std::cout << "total nodes:  " << combinedVal.size() << "\n";
                recompose_remesh(nodeMapGrid, GridPointVal, combinedVal); 
                 
                var_out[i].SetSelection(adios2::Box<adios2::Dims>({}, {combinedVal.size()}));
                writer.Put<double>(var_out[i], combinedVal.data(), adios2::Mode::Sync);
                writer.PerformPuts();
                std::cout << "Read block: " << info.BlockID << " size (byte) = " << nNodePt << std::endl;
                GridPointVal.clear();
                combinedVal.clear();
            }
            nodeMapGrid.clear();
            if (info.BlockID==nBlocks) break;
        }
        std::cout << "end\n"; 
        reader.EndStep();
    	writer.EndStep();
        reader_mesh.EndStep();
        ts ++;
    }
    reader.Close();
    reader_mesh.Close();
    writer.Close();

    MPI_Finalize();
    return 0;
}
