#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <dirent.h>

#include "adios2.h"
#include "mgard/compress.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    int cnt_argv = 1;
    std::string dpath(argv[cnt_argv++]);
    std::string fname(argv[cnt_argv++]);
    double s = std::stof(argv[cnt_argv++]);
    double tol = std::stof(argv[cnt_argv++]);
    int n_vars = std::stoi(argv[cnt_argv++]);
    std::vector<std::string> var_name(n_vars);
    for (int i=0; i< n_vars; i++) {
        var_name[i] = argv[cnt_argv++];
    }
    
    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO writer_io = ad.DeclareIO("Output");

    if (rank==0) {
        std::cout << "write: " << "./" + fname + ".mgard" << "\n";
        std::cout << "readin: " << dpath + fname << "\n";
    }
    adios2::Engine reader = reader_io.Open(dpath + fname, adios2::Mode::Read);
    adios2::Engine writer = writer_io.Open("./" + fname + ".mgard", adios2::Mode::Write);

    std::vector<size_t> compressed_size(n_vars, 0);
    std::vector<size_t> data_size(n_vars, 0);
    size_t ts = 0;
    adios2::Variable<int32_t> scalar_ad2;
    int32_t scalar_in;
    adios2::Variable<double> scalar_ad2_v2;
    double scalar_in_v2;
    adios2::Variable<int32_t> var_ad2_v2;
    std::vector<int32_t> var_in_v2;
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
        size_t step = reader.CurrentStep();
        if (rank==0) std::cout << "Process step " << step << ": " << std::endl;
        scalar_ad2_v2 = reader_io.InquireVariable<double>("/hpMusic_base/time");
        reader.Get(scalar_ad2_v2, scalar_in_v2, adios2::Mode::Sync);
        reader.PerformGets();
        std::cout << "step " << scalar_in_v2 << "\n";
        var_ad2_v2 = reader_io.InquireVariable<int32_t>("/hpMusic_base/hpMusic_Zone/n_total_node");
        reader.Get(var_ad2_v2, var_in_v2, adios2::Mode::Sync);
        reader.PerformGets();
        
        for (int i=0; i<n_vars; i++) {
            adios2::Variable<double> var_ad2;
            var_ad2 = reader_io.InquireVariable<double>(var_name[i]);
            auto bi = reader.BlocksInfo(var_ad2, ts);
            size_t nBlocks = bi.size();
            std::cout << var_name[i].c_str() << " has " << nBlocks << " blocks\n";
            std::vector<std::size_t> shape = var_ad2.Shape();
            adios2::Variable<double> var_out = writer_io.DefineVariable<double>(var_name[i], {}, {}, {adios2::UnknownDim});
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
                for (int k=0; k<19; k++) {
                    std::cout << var_in[k] << ", ";
                }
                std::cout << var_in[20] << "\n";
                const mgard::TensorMeshHierarchy<1, double> hierarchy({var_in.size()});
                const mgard::CompressedDataset<1, double> compressed = mgard::compress(hierarchy, var_in.data(), s, abs_tol);
                const mgard::DecompressedDataset<1, double> decompressed = mgard::decompress(compressed);
                var_out.SetSelection(adios2::Box<adios2::Dims>({}, {var_in.size()}));
                writer.Put<double>(var_out, decompressed.data(), adios2::Mode::Sync);
                //writer.Put<double>(var_out, var_in.data(), adios2::Mode::Sync);
                writer.PerformPuts();
                compressed_size[i] += compressed.size();
                std::cout << "Read block: " << info.BlockID << " size (byte) = " << var_in.size() << std::endl;
                break;
            }
        }
        std::cout << "end\n"; 
        reader.EndStep();
    }
    reader.Close();
    writer.Close();
    std::vector<size_t> gb_compressed(n_vars), lSize(n_vars);
    for (int i=0; i<n_vars; i++) {
        MPI_Allreduce(&compressed_size[i], &gb_compressed[i], 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&data_size[i], &lSize[i], 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
             printf("%s: compression ratio = %.4f (%ld / %ld)\n", var_name[i].c_str(), ((double)lSize[i]*8) / gb_compressed[i], lSize[i]*8, gb_compressed[i]);
        }
    }

    MPI_Finalize();
    return 0;
}
