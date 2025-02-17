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


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    int cnt_argv = 1;
    std::string dpath(argv[cnt_argv++]);
    std::string fname(argv[cnt_argv++]);
    int n_vars = std::stoi(argv[cnt_argv++]);
    std::vector<std::string> var_name(n_vars);
    for (int i=0; i< n_vars; i++) {
        var_name[i] = argv[cnt_argv++];
    }
    
    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO writer_io = ad.DeclareIO("Output");

    if (rank==0) {
        std::cout << "write: " << "./" + fname + ".decompressed" << "\n";
        std::cout << "readin: " << dpath + fname << "\n";
    }
    adios2::Engine reader = reader_io.Open(dpath + fname, adios2::Mode::Read);
    adios2::Engine writer = writer_io.Open(fname + ".decompressed", adios2::Mode::Write);

    size_t ts = 0;
    adios2::Variable<int32_t> var_ad2_v2;
    std::vector<int32_t> var_in_v2;
    //size_t compressed_size;
  
    std::vector<adios2::Variable<double>> var_out(n_vars);
    for (int i=0; i<n_vars; i++) {
    	var_out[i] = writer_io.DefineVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/" + var_name[i], {}, {}, {adios2::UnknownDim});
    }

    adios2::Params params;
    size_t maxBlocks = std::stoi(argv[cnt_argv++]);;
    if (maxBlocks % np_size) {
        std::cout << "Error::The total number of blocks must be a multiple of the number of ranks\n";
        return -1;
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
    	writer.BeginStep();
        size_t step = reader.CurrentStep();
        if (rank==0) std::cout << "Process step " << step << ": " << std::endl;
        
        for (int i=0; i<n_vars; i++) {
            adios2::Variable<double> var_ad2;
            var_ad2 = reader_io.InquireVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/"+var_name[i]);
            auto bi = reader.BlocksInfo(var_ad2, ts);
            maxBlocks = (maxBlocks > bi.size()) ? bi.size() : maxBlocks;
            size_t nBlocks = (size_t) ((double)maxBlocks / (double)np_size);
            size_t blockId = rank*nBlocks;
            //std::cout << var_name[i].c_str() << " has " << maxBlocks << " blocks\n";
            //size_t b = 0;//rank;
             while (blockId < (rank+1)*nBlocks) { 
                var_ad2.SetBlockSelection(blockId);
                std::vector<double> var_in; 
                reader.Get(var_ad2, var_in, adios2::Mode::Sync);
                reader.PerformGets();
    	    	//std::cout << "total nodes:  " << var_in.size() << "\n";
	    	    // std::cout << "rank " << rank << ": " << var_in[0] << ", "<< var_in[10] << ", " << var_in[100] << ", " << var_in[1000] << "\n";
                var_out[i].SetSelection(adios2::Box<adios2::Dims>({}, {var_in.size()}));
                writer.Put<double>(var_out[i], var_in.data(), adios2::Mode::Sync);
                writer.PerformPuts();
                //std::cout << "Read block: " << blockId << " size (byte) = " << var_in.size() << std::endl;
                blockId ++;
            }
        }
        std::cout << "end\n"; 
        reader.EndStep();
    	writer.EndStep();
    }
    reader.Close();
    writer.Close();

    MPI_Finalize();
    return 0;
}
