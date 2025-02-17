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
void error_calc(T *var_in, T *var_out, size_t data_size, T base)
{
    T diff, abs_err=0.0, rmse=0.0;
    for (size_t i=0; i<data_size; i++) {
        //norm_data += var_in[i] * var_in[i];
        diff    = std::abs(var_in[i] - var_out[i]);
        abs_err = (abs_err < diff) ? diff : abs_err;
        rmse   += diff*diff;
    }
    rmse      = std::sqrt(rmse / data_size);

    //std::cout << "L-inf: " << abs_err << ", relative by data_norm: " << abs_err / norm_data << ", by value range: "<< abs_err / (maxv-minv) << "\n";
    std::cout << "L2: " << rmse << ", relative by base value: " << rmse / base << "\n"; 
}

template <typename T>
void qoi_calc(T *density, T *pressure, T *vx, T *vy, T *vz, size_t num, 
              T &PT, T &Temperature, T &mu, T &VTOT, T &Loss)
{
    T gamma = 1.4;
    T R     = 287.1;
    T mi    = gamma / (gamma-1);
    T mu_r  = 1.716e-5;
    T Tr    = 273.15;
    T S     = 110.4; 
    
    for (size_t i=0; i<num; i++) {
        VTOT[i]     = (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        pressure[i] = (pressure[i] - 0.5 * VTOT[i] * density[i]) * (gamma-1);
        VTOT[i]     = std::sqrt(VTOT[i]);

        Temperature[i] = pressure[i] / (density[i] * R);
        T C    = std::sqrt(gamma * pressure[i] / density[i]);
        T Mach = VTOT[i] / C;
        
        T temp   = std::pow((1 + (gamma-1)/2.0 * Mach * Mach), mi); 
        PT[i]    = pressure[i] * temp;
        mu[i]    = mu_r*((Temperature / Tr)**1.5)*(Tr+S) / (Temperature + S)
    } 
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    int cnt_argv = 1;
    std::string dpath(argv[cnt_argv++]);
    std::string fname1(argv[cnt_argv++]);
    std::string fname2(argv[cnt_argv++]);
    int n_vars = std::stoi(argv[cnt_argv++]);
    std::vector<std::string> var_name(n_vars);
    for (int i=0; i< n_vars; i++) {
        var_name[i] = argv[cnt_argv++];
    }
    size_t maxBlocks = std::stoi(argv[cnt_argv++]);

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io_1 = ad.DeclareIO("Input1");
    adios2::IO reader_io_2 = ad.DeclareIO("Input2");

    if (rank==0) {
        std::cout << "readin: " << dpath + fname1 << "\n";
        std::cout << "readin: " << dpath + fname2 << "\n";
    }
    adios2::Engine reader_1 = reader_io_1.Open(dpath + fname1, adios2::Mode::Read);
    adios2::Engine reader_2 = reader_io_2.Open(dpath + fname2, adios2::Mode::Read);

    size_t ts = 0;
    while (true) {
        // Begin step
        adios2::StepStatus read_status = reader_1.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady) {
            // std::cout << "Stream not ready yet. Waiting...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK) {
            break;
        }
        read_status = reader_2.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady) {
            // std::cout << "Stream not ready yet. Waiting...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK) {
            break;
        }

        size_t step = reader_1.CurrentStep();
        if (rank==0) std::cout << "Process step " << step << ": " << std::endl;
        for (int i=0; i<n_vars; i++) {
            adios2::Variable<double> var_ad1, var_ad2;
            var_ad1 = reader_io_1.InquireVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/"+var_name[i]);
            var_ad2 = reader_io_2.InquireVariable<double>("/hpMusic_base/hpMusic_Zone/FlowSolution/"+var_name[i]);
            auto bi = reader_1.BlocksInfo(var_ad1, ts);
            size_t nBlocks = bi.size();
            std::cout << var_name[i].c_str() << " has " << nBlocks << " blocks\n";
            double minv = var_ad1.Min();
            double maxv = var_ad1.Max();
            for (auto &info : bi) {
                var_ad1.SetBlockSelection(info.BlockID);
                var_ad2.SetBlockSelection(info.BlockID);
                std::cout << "blockID = " << info.BlockID << "\n";
                std::vector<double> var_in_1, var_in_2;
                reader_1.Get(var_ad1, var_in_1, adios2::Mode::Sync);
                reader_1.PerformGets();
                reader_2.Get(var_ad2, var_in_2, adios2::Mode::Sync);
                reader_2.PerformGets();
                
                var_in_1.clear();
                var_in_2.clear();
                if (info.BlockID==maxBlocks) break;
            }
        }
        ts ++;
        reader_1.EndStep();
        reader_2.EndStep();
    }
    reader_1.Close();
    reader_2.Close();

    MPI_Finalize();

    return 0;
}
