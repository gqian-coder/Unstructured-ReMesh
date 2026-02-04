#include <chrono>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

#include "adios2.h"
#include "mgard/compress_x.hpp"
#include "mpi.h"
#include <time.h>
#include <zstd.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    if (argc < 2 || argc > 3)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " compressed_input.bp [decompressed_output.bp]\n";
            std::cerr << "  compressed_input.bp    - Compressed BP file to decompress\n";
            std::cerr << "  decompressed_output.bp - (Optional) Output filename\n";
            std::cerr << "\nAutomatically decompresses all variables matching '*FlowSolution*'\n";
            std::cerr << "Uses ADIOS2 plugin operator for decompression.\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_file(argv[1]);
    std::string output_file;
    
    if (argc == 3)
    {
        output_file = argv[2];
    }
    else
    {
        // Generate output filename by replacing .bp with _decompressed.bp
        size_t bp_pos = input_file.rfind(".bp");
        if (bp_pos != std::string::npos && bp_pos == input_file.length() - 3)
            output_file = input_file.substr(0, bp_pos) + "_decompressed.bp";
        else
            output_file = input_file + "_decompressed";
    }

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO writer_io = ad.DeclareIO("Output");

    if (rank == 0)
    {
        std::cout << "Input file:  " << input_file << "\n";
        std::cout << "Output file: " << output_file << "\n";
    }

    adios2::Engine reader = reader_io.Open(input_file, adios2::Mode::Read);

    // Begin first step to discover variables
    adios2::StepStatus read_status = reader.BeginStep(adios2::StepMode::Read, 10.0f);
    if (read_status != adios2::StepStatus::OK)
    {
        if (rank == 0)
            std::cerr << "Failed to read first step from input file.\n";
        reader.Close();
        MPI_Finalize();
        return 1;
    }

    // Auto-detect all variables matching *FlowSolution* pattern
    std::map<std::string, adios2::Params> available_vars = reader_io.AvailableVariables();
    std::vector<std::string> var_name;
    std::vector<std::string> var_type;
    for (const auto &var_pair : available_vars)
    {
        const std::string &name = var_pair.first;
        if (name.find("FlowSolution") != std::string::npos)
        {
            auto it = var_pair.second.find("Type");
            if (it != var_pair.second.end() && 
                (it->second == "double" || it->second == "float"))
            {
                var_name.push_back(name);
                var_type.push_back(it->second);
            }
        }
    }

    int n_vars = var_name.size();
    if (rank == 0)
    {
        std::cout << "Found " << n_vars << " FlowSolution variables to decompress:\n";
        for (int i = 0; i < n_vars; i++)
        {
            std::cout << "  - " << var_name[i] << " (" << var_type[i] << ")\n";
        }
    }

    if (n_vars == 0)
    {
        if (rank == 0)
        {
            std::cerr << "No FlowSolution variables found in the input file.\n";
        }
        reader.EndStep();
        reader.Close();
        MPI_Finalize();
        return 1;
    }

    adios2::Engine writer = writer_io.Open(output_file, adios2::Mode::Write);

    size_t total_size_bytes = 0;

    // Define output variables (matching input types)
    std::vector<adios2::Variable<double>> var_out_double(n_vars);
    std::vector<adios2::Variable<float>> var_out_float(n_vars);
    for (int i = 0; i < n_vars; i++)
    {
        if (var_type[i] == "double")
            var_out_double[i] = writer_io.DefineVariable<double>(var_name[i], {}, {}, {adios2::UnknownDim});
        else
            var_out_float[i] = writer_io.DefineVariable<float>(var_name[i], {}, {}, {adios2::UnknownDim});
    }

    size_t ts = 0;
    bool first_step = true;
    while (true)
    {
        // Begin step (skip for first iteration since we already did it)
        if (!first_step)
        {
            read_status = reader.BeginStep(adios2::StepMode::Read, 10.0f);
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
            if (var_type[i] == "double")
            {
                adios2::Variable<double> var_ad2 = reader_io.InquireVariable<double>(var_name[i]);
                auto bi = reader.BlocksInfo(var_ad2, ts);
                size_t nBlocks = bi.size();
                if (rank == 0)
                    std::cout << "  " << var_name[i] << ": " << nBlocks << " blocks\n";

                size_t blockId = rank;
                while (blockId < nBlocks)
                {
                    var_ad2.SetBlockSelection(blockId);
                    std::vector<double> var_in;
                    reader.Get(var_ad2, var_in, adios2::Mode::Sync);
                    reader.PerformGets();

                    total_size_bytes += var_in.size() * sizeof(double);

                    var_out_double[i].SetSelection(adios2::Box<adios2::Dims>({}, {var_in.size()}));
                    writer.Put<double>(var_out_double[i], var_in.data(), adios2::Mode::Sync);
                    writer.PerformPuts();

                    blockId += np_size;
                }
            }
            else  // float
            {
                adios2::Variable<float> var_ad2 = reader_io.InquireVariable<float>(var_name[i]);
                auto bi = reader.BlocksInfo(var_ad2, ts);
                size_t nBlocks = bi.size();
                if (rank == 0)
                    std::cout << "  " << var_name[i] << ": " << nBlocks << " blocks\n";

                size_t blockId = rank;
                while (blockId < nBlocks)
                {
                    var_ad2.SetBlockSelection(blockId);
                    std::vector<float> var_in;
                    reader.Get(var_ad2, var_in, adios2::Mode::Sync);
                    reader.PerformGets();

                    total_size_bytes += var_in.size() * sizeof(float);

                    var_out_float[i].SetSelection(adios2::Box<adios2::Dims>({}, {var_in.size()}));
                    writer.Put<float>(var_out_float[i], var_in.data(), adios2::Mode::Sync);
                    writer.PerformPuts();

                    blockId += np_size;
                }
            }
        }

        if (rank == 0)
            std::cout << "Completed step " << ts << "\n";
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
        std::cout << "\nDecompression complete.\n";
        std::cout << "  Steps processed: " << ts << "\n";
        std::cout << "  Variables: " << n_vars << "\n";
        if (total_size_gb >= 1.0)
            std::cout << "  Total data: " << total_size_gb << " GB\n";
        else
            std::cout << "  Total data: " << total_size_mb << " MB\n";
        std::cout << "  Output: " << output_file << "\n";
    }
    return 0;
}
