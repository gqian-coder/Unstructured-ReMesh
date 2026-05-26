#include <chrono>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

#include "adios2.h"
#include "mgard/compress_x.hpp"
#include "mpi.h"
#include <time.h>
#include <zstd.h>

std::string to_string_ld(long double number)
{
    long double temp = number;
    long double integerPart = floor(temp);
    int fractionCounter = 0;

    while (temp - integerPart > 0)
    {
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

int main(int argc, char **argv)
{
    // MGARD device selection: set MGARD_X_DEVICE_TYPE=SERIAL/HIP/CUDA externally.
    // If the env var is unset, MGARD uses AUTO (picks HIP on Frontier).
    MPI_Init(&argc, &argv);
    int rank, np_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    if (argc < 4 || argc > 5)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " input.bp output.bp error_bound [n_blocks]\n";
            std::cerr << "  input.bp    - Input BP file\n";
            std::cerr << "  output.bp   - Output BP file with compressed variables\n";
            std::cerr << "  error_bound - Relative error bound for compression\n";
            std::cerr << "  n_blocks    - (Optional) Maximum number of blocks to process\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_file(argv[1]);
    std::string output_file(argv[2]);
    double tol = std::stod(argv[3]);
    size_t maxBlocks = (argc == 5) ? (size_t)std::stoi(argv[4]) : SIZE_MAX;

    // Honour MAX_STEPS env var (default: all steps).
    const char *maxStepsEnv = std::getenv("MAX_STEPS");
    size_t maxSteps = maxStepsEnv ? (size_t)std::stoul(maxStepsEnv) : SIZE_MAX;

    // NO_PASSTHROUGH=1 : skip writing non-FlowSolution variables so the output
    // contains only compressed FlowSolution data for accurate CR measurement.
    bool no_passthrough = (std::getenv("NO_PASSTHROUGH") != nullptr);

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO writer_io = ad.DeclareIO("Output");

    if (rank == 0)
    {
        std::cout << "Input file:  " << input_file << "\n";
        std::cout << "Output file: " << output_file << "\n";
        std::cout << "Error bound: " << tol << "\n";
        if (maxBlocks != SIZE_MAX)
            std::cout << "Max blocks:  " << maxBlocks << "\n";
        else
            std::cout << "Max blocks:  all\n";
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

    // Auto-detect variables containing "FlowSolution" in their name
    // Optional: set MGARD_VAR_FILTER env var to restrict to variables containing that substring
    const char *var_filter_env = std::getenv("MGARD_VAR_FILTER");
    const std::string var_filter = var_filter_env ? std::string(var_filter_env) : "";
    std::map<std::string, adios2::Params> available_vars = reader_io.AvailableVariables();
    std::vector<std::string> var_name;         // variables to compress
    std::vector<std::string> passthru_name;    // all other double variables (written uncompressed)
    for (const auto &var_pair : available_vars)
    {
        const std::string &name = var_pair.first;
        auto typeIt = var_pair.second.find("Type");
        if (typeIt == var_pair.second.end() || typeIt->second != "double")
            continue; // only handle doubles for passthrough
        if (name.find("/FlowSolution/") != std::string::npos)
        {
            if (!var_filter.empty() && name.find(var_filter) == std::string::npos)
            {
                // FlowSolution variable excluded by filter: pass through uncompressed
                passthru_name.push_back(name);
                continue;
            }
            var_name.push_back(name);
        }
        else
        {
            passthru_name.push_back(name);
        }
    }

    int n_vars = var_name.size();
    if (rank == 0)
    {
        std::cout << "Found " << n_vars << " FlowSolution variables:\n";
        for (const auto &name : var_name)
        {
            std::cout << "  - " << name << "\n";
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

    size_t total_size_bytes = 0;  // Per-rank accumulator of uncompressed input bytes; reduced before reporting

    // Define compressed output variables
    std::vector<adios2::Variable<double>> var_out(n_vars);
    for (int i = 0; i < n_vars; i++)
    {
        var_out[i] = writer_io.DefineVariable<double>(
            var_name[i], {}, {}, {adios2::UnknownDim});
    }

    // Define passthrough output variables (no compression operator)
    int n_passthru = (int)passthru_name.size();
    std::vector<adios2::Variable<double>> passthru_out(n_passthru);
    for (int i = 0; i < n_passthru; i++)
        passthru_out[i] = writer_io.DefineVariable<double>(passthru_name[i], {}, {}, {adios2::UnknownDim});

    adios2::Operator op = ad.DefineOperator("mgard", "mgard");

    // Add operation to output variables once (before the timestep loop)
    // We need to compute tolerance based on min/max from the first step
    for (int i = 0; i < n_vars; i++)
    {
        adios2::Variable<double> var_ad2 = reader_io.InquireVariable<double>(var_name[i]);
        double minv = var_ad2.Min();
        double maxv = var_ad2.Max();
        double abs_tol = tol * (maxv - minv);
        if (rank == 0)
            std::cout << var_name[i].c_str() << ": min/max = " << minv << "/" << maxv
                      << ", abs_tol = " << abs_tol << std::endl;
        var_out[i].AddOperation(op,
                                {{"tolerance", to_string_ld(abs_tol)}, {"mode", "ABS"}});
    }

    int ts = 0;
    bool first_step = true;  // We already called BeginStep for variable discovery
    while (true)
    {
        // Begin step (skip for first iteration since we already did it)
        if (!first_step)
        {
            // Use 0.0f timeout: for static BP files all steps are already on disk,
            // so BeginStep must return immediately.  A non-zero timeout causes the
            // call to block for that many seconds when there are no more steps,
            // making the program appear to hang after compression finishes.
            adios2::StepStatus read_status = reader.BeginStep(adios2::StepMode::Read, 0.0f);
            if (read_status != adios2::StepStatus::OK)
            {
                // NotReady or EndOfStream — either way we are done.
                break;
            }
        }
        first_step = false;

        // Stop once we have processed maxSteps steps
        if ((size_t)ts >= maxSteps)
        {
            reader.EndStep();
            break;
        }

        writer.BeginStep();
        size_t step = reader.CurrentStep();
        if (rank == 0)
            std::cout << "Process step " << step << ": " << std::endl;
        for (int i = 0; i < n_vars; i++)
        {
            adios2::Variable<double> var_ad2;
            var_ad2 = reader_io.InquireVariable<double>(var_name[i]);
            auto bi = reader.BlocksInfo(var_ad2, ts);
            size_t nBlocks = std::min(bi.size(), maxBlocks);
            if (rank == 0)
                std::cout << var_name[i].c_str() << " has " << bi.size() << " blocks, processing " << nBlocks << " blocks\n";
            // Contiguous block distribution to preserve block ordering in output
            size_t blocksPerRank = (nBlocks + np_size - 1) / np_size;
            size_t startBlock = rank * blocksPerRank;
            size_t endBlock = std::min(startBlock + blocksPerRank, nBlocks);
            for (size_t blockId = startBlock; blockId < endBlock; blockId++)
            {
                std::cout << "Compress block " << blockId << "\n";
                var_ad2.SetBlockSelection(blockId);
                std::vector<double> var_in;
                reader.Get(var_ad2, var_in, adios2::Mode::Sync);
                reader.PerformGets();
                
                // Accumulate total size
                total_size_bytes += var_in.size() * sizeof(double);

                var_out[i].SetSelection(adios2::Box<adios2::Dims>({}, {var_in.size()}));
                writer.Put<double>(var_out[i], var_in.data(), adios2::Mode::Sync);
                writer.PerformPuts();
            }
        }
        // Pass through all non-FlowSolution double variables block by block.
        // Skipped when NO_PASSTHROUGH=1 so output contains only compressed
        // FlowSolution data for accurate compression-ratio measurement.
        for (int i = 0; !no_passthrough && i < n_passthru; i++)
        {
            adios2::Variable<double> vr = reader_io.InquireVariable<double>(passthru_name[i]);
            if (!vr) continue;
            auto bi = reader.BlocksInfo(vr, ts);
            size_t nBlocks = std::min(bi.size(), maxBlocks);
            size_t blocksPerRank = (nBlocks + np_size - 1) / np_size;
            size_t startBlock = rank * blocksPerRank;
            size_t endBlock = std::min(startBlock + blocksPerRank, nBlocks);
            for (size_t blockId = startBlock; blockId < endBlock; blockId++)
            {
                vr.SetBlockSelection(blockId);
                std::vector<double> buf;
                reader.Get(vr, buf, adios2::Mode::Sync);
                reader.PerformGets();
                passthru_out[i].SetSelection(adios2::Box<adios2::Dims>({}, {buf.size()}));
                writer.Put<double>(passthru_out[i], buf.data(), adios2::Mode::Sync);
                writer.PerformPuts();
            }
        }

        if (rank == 0)
            std::cout << "end step " << ts << "\n";
        writer.EndStep();

        ts++;
        reader.EndStep();
    }
    reader.Close();
    writer.Close();

    // Each rank only accumulated the bytes for the blocks it processed.
    // Sum across all ranks so the reported total is independent of np_size.
    unsigned long long local_bytes = static_cast<unsigned long long>(total_size_bytes);
    unsigned long long global_bytes = 0;
    MPI_Reduce(&local_bytes, &global_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0)
    {
        double total_size_gb = global_bytes / (1024.0 * 1024.0 * 1024.0);
        double total_size_mb = global_bytes / (1024.0 * 1024.0);
        if (total_size_gb >= 1.0)
            std::cout << "Total input data passed to compressor:  " << total_size_gb << " GB\n";
        else
            std::cout << "Total input data passed to compressor:  " << total_size_mb << " MB\n";
    }

    MPI_Finalize();
    return 0;
}
