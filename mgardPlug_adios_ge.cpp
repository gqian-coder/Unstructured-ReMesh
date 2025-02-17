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
    int n_vars = std::stoi(argv[cnt_argv++]);
    std::vector<std::string> var_name(n_vars);
    for (int i=0; i< n_vars; i++) {
        var_name[i] = argv[cnt_argv++];
    }
    double tol    = std::stof(argv[cnt_argv++]);
    int maxBlocks = std::stoi(argv[cnt_argv++]);
    //if (maxBlocks % np_size) {
    //    std::cout << "Error::The total number of blocks must be a multiple of the number of ranks\n";
    //    return -1; 
    //}
    
    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO writer_io = ad.DeclareIO("Output");

    if (rank==0) {
        std::cout << "write: " << "./" + fname + ".compressed" << "\n";
        std::cout << "readin: " << dpath + fname << "\n";
    }
    adios2::Engine reader = reader_io.Open(dpath + fname, adios2::Mode::Read);
    adios2::Engine writer = writer_io.Open(fname + ".compressed", adios2::Mode::Write);

    double time_s = 0.0;
  
    std::vector<adios2::Variable<double>> var_out(n_vars);
    for (int i=0; i<n_vars; i++) {
    	var_out[i] = writer_io.DefineVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/" + var_name[i], {}, {}, {adios2::UnknownDim});
    }

    int nBlocks       = (int) ((double)maxBlocks / (double)np_size);
    int blockId       = rank*nBlocks;
    int rankBlock_num = (rank<np_size-1) ? blockId+nBlocks : blockId+nBlocks + (maxBlocks % np_size);

    adios2::Params params;
    params["PluginName"] = "mgardReMesh";
    params["PluginLibrary"] = "CompressMGARDMeshToGridOperator";
    params["meshfile"]  = "../../dataset/Mesh2GridMap.bp";
    params["ebratio"]   = "0.7";
    params["mode"]      = "ABS";
    // Here we need to let MGARDPlug Operator know which blocks to be loaded for each rank
    // This list must remain the same across variables and steps
    std::ostringstream blockList;
    for (int i=blockId; i<rankBlock_num; i++) {
        if (i>0) blockList << " ";
        blockList << i;  
    }
    std::cout << "rank " << rank << " will readin block " << blockList.str() << "\n"; 
    params["blocks"] = blockList.str(); 
    // adios2::Operator op = ad.DefineOperator("mgard", "mgard");
    
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
            std::cout << var_name[i].c_str() << " has " << nBlocks << " blocks\n";
            double minv = var_ad2.Min();
            double maxv = var_ad2.Max();
            //size_t b = 0;//rank;
            double abs_tol = tol * (maxv-minv);
            params["tolerance"] = to_string_ld(abs_tol);
            if (rank==0) std::cout << var_name[i].c_str() << ": min/max = "<< minv << "/" << maxv << ", tol = "<< abs_tol << std::endl;
	        // var_out[i].AddOperation(op, {{"tolerance", to_string_ld(abs_tol)}, {"mode", "ABS"}});
            var_out[i].AddOperation("plugin", params);
            blockId = rank*nBlocks;
            while (blockId < rankBlock_num) { 
                var_ad2.SetBlockSelection(blockId);
                //std::cout << "rank " << rank << ", blockID = " << blockId << ", nBlocks = " << nBlocks << ", maxB = " << maxBlocks << "\n";
                std::vector<double> var_in; 
                reader.Get(var_ad2, var_in, adios2::Mode::Sync);
                reader.PerformGets();
		        // std::cout << "rank = " << rank << ": " << var_in[0] << ", "<< var_in[10] << ", " << var_in[100] << ", " << var_in[1000] << "\n"; 
                
                auto start = std::chrono::high_resolution_clock::now();
                var_out[i].SetSelection(adios2::Box<adios2::Dims>({}, {var_in.size()}));
                writer.Put<double>(var_out[i], var_in.data(), adios2::Mode::Sync);
                writer.PerformPuts();
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                time_s += (double)duration.count() / 1e6;

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

    std::cout << "total time spent: " << time_s << " sec\n";
    return 0;
}
