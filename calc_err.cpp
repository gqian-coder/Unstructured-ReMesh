#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>
#include <dirent.h>
#include <iomanip>

#include "adios2.h"
#include "mgard/compress_x.hpp"
#include <zstd.h>
#include <time.h>
#include <chrono>

template <typename T>
void error_calc(const std::string& var_name, T *var_in, T *var_out, size_t data_size, T minv, T maxv,
                T& total_abs_err, T& total_rmse, size_t& total_count)
{
    T diff, norm_data = 0.0, abs_err = 0.0, rmse = 0.0;
    for (size_t i = 0; i < data_size; i++) {
        norm_data += var_in[i] * var_in[i];
        diff = std::abs(var_in[i] - var_out[i]);
        abs_err = (abs_err < diff) ? diff : abs_err;
        rmse += diff * diff;
    }
    
    // Update running totals for aggregate statistics
    total_abs_err = (total_abs_err < abs_err) ? abs_err : total_abs_err;
    total_rmse += rmse;
    total_count += data_size;
    
    rmse = std::sqrt(rmse / data_size);
    norm_data = std::sqrt(norm_data / data_size);
    
    T value_range = maxv - minv;
    T rel_abs_err = (value_range > 0) ? abs_err / value_range : 0;
    T rel_rmse = (value_range > 0) ? rmse / value_range : 0;
    
    std::cout << "    L-inf: " << std::scientific << std::setprecision(6) << abs_err 
              << " (rel: " << rel_abs_err << ")\n";
    std::cout << "    RMSE:  " << rmse << " (rel: " << rel_rmse << ")\n";
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    if (argc != 3)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " original_file.bp compressed_file.bp\n";
            std::cerr << "  original_file.bp   - Original (uncompressed) BP file\n";
            std::cerr << "  compressed_file.bp - Compressed/decompressed BP file to compare\n";
            std::cerr << "\nAutomatically computes errors for all variables matching '*FlowSolution*'\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string fname1(argv[1]);
    std::string fname2(argv[2]);

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io_1 = ad.DeclareIO("Input1");
    adios2::IO reader_io_2 = ad.DeclareIO("Input2");

    if (rank == 0) {
        std::cout << "Original file:   " << fname1 << "\n";
        std::cout << "Compressed file: " << fname2 << "\n";
    }
    
    adios2::Engine reader_1 = reader_io_1.Open(fname1, adios2::Mode::Read);
    adios2::Engine reader_2 = reader_io_2.Open(fname2, adios2::Mode::Read);

    // Begin first step to discover variables
    adios2::StepStatus read_status = reader_1.BeginStep(adios2::StepMode::Read, 10.0f);
    if (read_status != adios2::StepStatus::OK) {
        if (rank == 0)
            std::cerr << "Failed to read first step from original file.\n";
        reader_1.Close();
        reader_2.Close();
        MPI_Finalize();
        return 1;
    }
    read_status = reader_2.BeginStep(adios2::StepMode::Read, 10.0f);
    if (read_status != adios2::StepStatus::OK) {
        if (rank == 0)
            std::cerr << "Failed to read first step from compressed file.\n";
        reader_1.EndStep();
        reader_1.Close();
        reader_2.Close();
        MPI_Finalize();
        return 1;
    }

    // Auto-detect all variables matching *FlowSolution* pattern
    std::map<std::string, adios2::Params> available_vars = reader_io_1.AvailableVariables();
    std::vector<std::string> var_name;
    std::vector<std::string> var_type;
    for (const auto &var_pair : available_vars) {
        const std::string &name = var_pair.first;
        if (name.find("FlowSolution") != std::string::npos) {
            auto it = var_pair.second.find("Type");
            if (it != var_pair.second.end() && 
                (it->second == "double" || it->second == "float")) {
                var_name.push_back(name);
                var_type.push_back(it->second);
            }
        }
    }

    int n_vars = var_name.size();
    if (rank == 0) {
        std::cout << "\nFound " << n_vars << " FlowSolution variables:\n";
        for (int i = 0; i < n_vars; i++) {
            std::cout << "  - " << var_name[i] << " (" << var_type[i] << ")\n";
        }
        std::cout << "\n";
    }

    if (n_vars == 0) {
        if (rank == 0) {
            std::cerr << "No FlowSolution variables found in the input file.\n";
        }
        reader_1.EndStep();
        reader_2.EndStep();
        reader_1.Close();
        reader_2.Close();
        MPI_Finalize();
        return 1;
    }

    // Track aggregate errors per variable
    std::vector<double> var_total_abs_err(n_vars, 0.0);
    std::vector<double> var_total_rmse(n_vars, 0.0);
    std::vector<size_t> var_total_count(n_vars, 0);
    std::vector<double> var_value_range(n_vars, 0.0);

    size_t ts = 0;
    bool first_step = true;
    while (true) {
        // Begin step (skip for first iteration since we already did it)
        if (!first_step) {
            read_status = reader_1.BeginStep(adios2::StepMode::Read, 10.0f);
            if (read_status == adios2::StepStatus::NotReady) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            } else if (read_status != adios2::StepStatus::OK) {
                break;
            }
            read_status = reader_2.BeginStep(adios2::StepMode::Read, 10.0f);
            if (read_status == adios2::StepStatus::NotReady) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            } else if (read_status != adios2::StepStatus::OK) {
                break;
            }
        }
        first_step = false;

        size_t step = reader_1.CurrentStep();
        if (rank == 0)
            std::cout << "=== Step " << step << " ===\n";

        for (int i = 0; i < n_vars; i++) {
            if (var_type[i] == "double") {
                adios2::Variable<double> var_ad1 = reader_io_1.InquireVariable<double>(var_name[i]);
                adios2::Variable<double> var_ad2 = reader_io_2.InquireVariable<double>(var_name[i]);
                
                if (!var_ad1 || !var_ad2) {
                    if (rank == 0)
                        std::cerr << "  Warning: " << var_name[i] << " not found in both files\n";
                    continue;
                }
                
                auto bi = reader_1.BlocksInfo(var_ad1, ts);
                size_t nBlocks = bi.size();
                double minv = var_ad1.Min();
                double maxv = var_ad1.Max();
                var_value_range[i] = maxv - minv;
                
                if (rank == 0)
                    std::cout << var_name[i] << " (" << nBlocks << " blocks, range: [" 
                              << minv << ", " << maxv << "])\n";

                size_t blockId = rank;
                while (blockId < nBlocks) {
                    var_ad1.SetBlockSelection(blockId);
                    var_ad2.SetBlockSelection(blockId);
                    
                    std::vector<double> var_in_1, var_in_2;
                    reader_1.Get(var_ad1, var_in_1, adios2::Mode::Sync);
                    reader_1.PerformGets();
                    reader_2.Get(var_ad2, var_in_2, adios2::Mode::Sync);
                    reader_2.PerformGets();

                    if (rank == 0 && nBlocks <= 4) {
                        std::cout << "  Block " << blockId << " (" << var_in_1.size() << " points):\n";
                        error_calc(var_name[i], var_in_1.data(), var_in_2.data(), var_in_1.size(), 
                                   minv, maxv, var_total_abs_err[i], var_total_rmse[i], var_total_count[i]);
                    } else {
                        // Silently accumulate for many blocks
                        double abs_err = 0.0, rmse = 0.0;
                        for (size_t j = 0; j < var_in_1.size(); j++) {
                            double diff = std::abs(var_in_1[j] - var_in_2[j]);
                            abs_err = (abs_err < diff) ? diff : abs_err;
                            rmse += diff * diff;
                        }
                        var_total_abs_err[i] = (var_total_abs_err[i] < abs_err) ? abs_err : var_total_abs_err[i];
                        var_total_rmse[i] += rmse;
                        var_total_count[i] += var_in_1.size();
                    }
                    
                    blockId += np_size;
                }
            } else {  // float
                adios2::Variable<float> var_ad1 = reader_io_1.InquireVariable<float>(var_name[i]);
                adios2::Variable<float> var_ad2 = reader_io_2.InquireVariable<float>(var_name[i]);
                
                if (!var_ad1 || !var_ad2) {
                    if (rank == 0)
                        std::cerr << "  Warning: " << var_name[i] << " not found in both files\n";
                    continue;
                }
                
                auto bi = reader_1.BlocksInfo(var_ad1, ts);
                size_t nBlocks = bi.size();
                float minv = var_ad1.Min();
                float maxv = var_ad1.Max();
                var_value_range[i] = maxv - minv;
                
                if (rank == 0)
                    std::cout << var_name[i] << " (" << nBlocks << " blocks, range: [" 
                              << minv << ", " << maxv << "])\n";

                size_t blockId = rank;
                while (blockId < nBlocks) {
                    var_ad1.SetBlockSelection(blockId);
                    var_ad2.SetBlockSelection(blockId);
                    
                    std::vector<float> var_in_1, var_in_2;
                    reader_1.Get(var_ad1, var_in_1, adios2::Mode::Sync);
                    reader_1.PerformGets();
                    reader_2.Get(var_ad2, var_in_2, adios2::Mode::Sync);
                    reader_2.PerformGets();

                    // Accumulate errors
                    float abs_err = 0.0f, rmse = 0.0f;
                    for (size_t j = 0; j < var_in_1.size(); j++) {
                        float diff = std::abs(var_in_1[j] - var_in_2[j]);
                        abs_err = (abs_err < diff) ? diff : abs_err;
                        rmse += diff * diff;
                    }
                    var_total_abs_err[i] = (var_total_abs_err[i] < abs_err) ? abs_err : var_total_abs_err[i];
                    var_total_rmse[i] += rmse;
                    var_total_count[i] += var_in_1.size();
                    
                    blockId += np_size;
                }
            }
        }
        
        ts++;
        reader_1.EndStep();
        reader_2.EndStep();
    }
    
    reader_1.Close();
    reader_2.Close();

    // Print summary
    if (rank == 0) {
        std::cout << "\n========================================\n";
        std::cout << "ERROR SUMMARY (across " << ts << " steps)\n";
        std::cout << "========================================\n";
        std::cout << std::left << std::setw(50) << "Variable" 
                  << std::right << std::setw(15) << "L-inf (abs)" 
                  << std::setw(15) << "L-inf (rel)"
                  << std::setw(15) << "RMSE (rel)" << "\n";
        std::cout << std::string(95, '-') << "\n";
        
        for (int i = 0; i < n_vars; i++) {
            if (var_total_count[i] > 0) {
                double rmse = std::sqrt(var_total_rmse[i] / var_total_count[i]);
                double rel_abs = (var_value_range[i] > 0) ? var_total_abs_err[i] / var_value_range[i] : 0;
                double rel_rmse = (var_value_range[i] > 0) ? rmse / var_value_range[i] : 0;
                
                // Extract just the variable name (last part of path)
                std::string short_name = var_name[i];
                size_t last_slash = short_name.rfind('/');
                if (last_slash != std::string::npos)
                    short_name = short_name.substr(last_slash + 1);
                
                std::cout << std::left << std::setw(50) << var_name[i]
                          << std::right << std::scientific << std::setprecision(4)
                          << std::setw(15) << var_total_abs_err[i]
                          << std::setw(15) << rel_abs
                          << std::setw(15) << rel_rmse << "\n";
            }
        }
        std::cout << "========================================\n";
    }

    MPI_Finalize();
    return 0;
}
