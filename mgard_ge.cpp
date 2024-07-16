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

template <typename T>
void error_calc(T *var_in, T *var_out, size_t data_size, T tolerance)
{
    T diff, norm_data = 0.0, abs_err=0.0, rmse=0.0;
    T minv = std::numeric_limits<T>::infinity();
    T maxv = -std::numeric_limits<T>::infinity();
    for (size_t i=0; i<data_size; i++) {
        minv       = std::min(minv, var_in[i]);
        maxv       = std::max(maxv, var_in[i]);
        norm_data += var_in[i] * var_in[i];
        diff    = std::abs(var_in[i] - var_out[i]);
        abs_err = (abs_err < diff) ? diff : abs_err;
        rmse   += diff*diff;
    }
    rmse      = std::sqrt(rmse / data_size);
    norm_data = std::sqrt(norm_data / data_size);

    std::cout << "Error print out: \n";
    std::cout << "requested error: ";
    std::cout << "l2 abs err = " << tolerance << "\n";

    std::cout << "L-inf: " << abs_err << ", relative by data_norm: " << abs_err / norm_data << ", by value range: "<< abs_err / (maxv-minv) << "\n";
    std::cout << "L2: " << rmse << ", relative by data_norm: " << rmse / norm_data << ", by value range: "<< rmse / (maxv-minv) << "\n";
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
    double s = std::stof(argv[cnt_argv++]);
    double tol = std::stof(argv[cnt_argv++]);
 
    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO writer_io = ad.DeclareIO("Output");

    if (rank==0) {
        std::cout << "write: " << "./" + fname + ".mgard" << "\n";
        std::cout << "readin: " << dpath + fname << "\n";
    }
    adios2::Engine reader = reader_io.Open(dpath + fname, adios2::Mode::Read);
    adios2::Engine writer = writer_io.Open(fname + ".mgard", adios2::Mode::Write);

    size_t compressed_size_step = 0;
    std::vector<size_t> compressed_size(n_vars, 0);
    std::vector<size_t> data_size(n_vars, 0);
    size_t ts = 0;
    adios2::Variable<int32_t> var_ad2_v2;
    std::vector<int32_t> var_in_v2;
    //size_t compressed_size;
  
    std::vector<adios2::Variable<double>> var_out(n_vars);
    for (int i=0; i<n_vars; i++) {
    	var_out[i] = writer_io.DefineVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/" + var_name[i], {}, {}, {adios2::UnknownDim});
    }
    double time_s = 0.0, compress_ts = 0.0;
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
            size_t nBlocks = bi.size();
            std::cout << var_name[i].c_str() << " has " << nBlocks << " blocks\n";
            double minv = var_ad2.Min();
            double maxv = var_ad2.Max();
            //size_t b = 0;//rank;
            double abs_tol = tol * (maxv-minv);
            if (rank==0) std::cout << var_name[i].c_str() << ": min/max = "<< minv << "/" << maxv << ", tol = "<< abs_tol << std::endl;
            for (auto &info : bi) {
                var_ad2.SetBlockSelection(info.BlockID);
                std::cout << "blockID = " << info.BlockID << "\n";
                std::vector<double> var_in; 
                reader.Get(var_ad2, var_in, adios2::Mode::Sync);
                reader.PerformGets();
                data_size[i] += var_in.size();
    		    std::cout << "total nodes:  " << var_in.size() << "\n";
	    	    std::cout << var_in[0] << ", "<< var_in[10] << "\n";
                auto start = std::chrono::high_resolution_clock::now();
		        mgard_x::Config config;
  		        config.lossless = mgard_x::lossless_type::Huffman_Zstd;
		        config.dev_type = mgard_x::device_type::CUDA;
		        config.dev_id   = 2;
		        std::vector<mgard_x::SIZE> shape{var_in.size()};
		        void *compressed_array_cpu = NULL;
		        mgard_x::compress(1, mgard_x::data_type::Double, shape, abs_tol, s,
                    mgard_x::error_bound_type::ABS, var_in.data(),
                    compressed_array_cpu, compressed_size_step, config, false);
		        void *decompressed_array_cpu = NULL;
		        mgard_x::decompress(compressed_array_cpu, compressed_size_step, 
                     decompressed_array_cpu, config, false);
		        compressed_size[i] += compressed_size_step;
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                compress_ts += (double)duration.count() / 1e6;
                var_out[i].SetSelection(adios2::Box<adios2::Dims>({}, {var_in.size()}));
                writer.Put<double>(var_out[i], (double *)decompressed_array_cpu, adios2::Mode::Sync);
                writer.PerformPuts();

                end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                time_s += (double)duration.count() / 1e6;
                std::cout << "Read block: " << info.BlockID << " size (byte) = " << var_in.size() << std::endl;
		        error_calc(var_in.data(), (double *)decompressed_array_cpu, var_in.size(), abs_tol);
		        free(compressed_array_cpu);
		        free(decompressed_array_cpu);
		        if (info.BlockID==5) break;
            }
        }
        std::cout << "end\n"; 
        reader.EndStep();
	writer.EndStep();
    }
    reader.Close();
    writer.Close();
    std::vector<size_t> gb_compressed(n_vars), lSize(n_vars);
    for (int i=0; i<n_vars; i++) {
        MPI_Allreduce(&compressed_size[i], &gb_compressed[i], 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&data_size[i], &lSize[i], 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
             printf("%s: compression ratio = %.4f (%ld / %ld)\n", var_name[i].c_str(), ((double)lSize[i]*8) / gb_compressed[i], lSize[i]*8, gb_compressed[i]);
             std::cout << "total time spent: " << time_s << " sec, compression time = " << compress_ts << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
