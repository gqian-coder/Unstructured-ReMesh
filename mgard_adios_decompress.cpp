#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
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

    if (argc != 2)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " compressed_input.bp\n";
            std::cerr << "  compressed_input.bp - Compressed BP file to decompress\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_file(argv[1]);
    // Generate output filename by replacing .bp with .decompressed.bp or appending .decompressed
    std::string output_file;
    size_t bp_pos = input_file.rfind(".bp");
    if (bp_pos != std::string::npos && bp_pos == input_file.length() - 3)
        output_file = input_file.substr(0, bp_pos) + ".decompressed.bp";
    else
        output_file = input_file + ".decompressed";

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO writer_io = ad.DeclareIO("Output");

    if (rank == 0)
    {
        std::cout << "Input file:  " << input_file << "\n";
        std::cout << "Output file: " << output_file << "\n";
    }
    adios2::Engine reader = reader_io.Open(input_file, adios2::Mode::Read);

    // Need to begin a step first to discover variables in BP files
    adios2::StepStatus read_status = reader.BeginStep(adios2::StepMode::Read, 10.0f);
    if (read_status != adios2::StepStatus::OK)
    {
        if (rank == 0)
            std::cerr << "Failed to read first step from input file.\n";
        reader.Close();
        MPI_Finalize();
        return 1;
    }

    // Auto-detect all double variables in the file
    std::map<std::string, adios2::Params> available_vars = reader_io.AvailableVariables();
    std::vector<std::string> var_name;
    for (const auto &var_pair : available_vars)
    {
        const std::string &name = var_pair.first;
        // Check if the variable is of type double
        auto it = var_pair.second.find("Type");
        if (it != var_pair.second.end() && it->second == "double")
        {
            var_name.push_back(name);
        }
    }

    int n_vars = var_name.size();
    if (rank == 0)
    {
        std::cout << "Found " << n_vars << " variables to decompress:\n";
        for (const auto &name : var_name)
        {
            std::cout << "  - " << name << "\n";
        }
    }

    if (n_vars == 0)
    {
        if (rank == 0)
        {
            std::cerr << "No variables found in the input file.\n";
        }
        reader.EndStep();
        reader.Close();
        MPI_Finalize();
        return 1;
    }

    adios2::Engine writer = writer_io.Open(output_file, adios2::Mode::Write);

    size_t total_size_bytes = 0;  // Track total size of decompressed data

    // Define output variables
    std::vector<adios2::Variable<double>> var_out(n_vars);
    for (int i = 0; i < n_vars; i++)
    {
        var_out[i] = writer_io.DefineVariable<double>(var_name[i], {}, {}, {adios2::UnknownDim});
    }

    int ts = 0;
    bool first_step = true;  // We already called BeginStep for variable discovery
    while (true)
    {
        // Begin step (skip for first iteration since we already did it)
        if (!first_step)
        {
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
        }
        first_step = false;

        writer.BeginStep();
        size_t step = reader.CurrentStep();
        if (rank == 0)
            std::cout << "Process step " << step << ": " << std::endl;

        for (int i = 0; i < n_vars; i++)
        {
            adios2::Variable<double> var_ad2;
            var_ad2 = reader_io.InquireVariable<double>(var_name[i]);
            auto bi = reader.BlocksInfo(var_ad2, ts);
            size_t nBlocks = bi.size();
            if (rank == 0)
                std::cout << var_name[i].c_str() << " has " << nBlocks << " blocks\n";

            size_t blockId = rank;
            while (blockId < nBlocks)
            {
                var_ad2.SetBlockSelection(blockId);
                if (rank == 0)
                    std::cout << "rank " << rank << ", blockID = " << blockId << "\n";
                std::vector<double> var_in;
                reader.Get(var_ad2, var_in, adios2::Mode::Sync);
                reader.PerformGets();

                // Only print total nodes for the first variable
                if (rank == 0 && i == 0)
                    std::cout << "total nodes:  " << var_in.size() << "\n";

                // Accumulate total size
                total_size_bytes += var_in.size() * sizeof(double);

                var_out[i].SetSelection(adios2::Box<adios2::Dims>({}, {var_in.size()}));
                writer.Put<double>(var_out[i], var_in.data(), adios2::Mode::Sync);
                writer.PerformPuts();

                blockId += np_size;
            }
        }
        if (rank == 0)
            std::cout << "end step " << ts << "\n";
        reader.EndStep();
        writer.EndStep();
        ts++;
    }
    reader.Close();
    writer.Close();

    MPI_Finalize();

    if (rank == 0)
    {
        double total_size_gb = total_size_bytes / (1024.0 * 1024.0 * 1024.0);
        double total_size_mb = total_size_bytes / (1024.0 * 1024.0);
        if (total_size_gb >= 1.0)
            std::cout << "Total data decompressed: " << total_size_gb << " GB\n";
        else
            std::cout << "Total data decompressed: " << total_size_mb << " MB\n";
    }
    return 0;
}
