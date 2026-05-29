/*
 * mgardCentroid_adios_ge.cpp
 *
 * Wrapper that compresses /volume/FlowSolution/* variables of the rotor-37
 * BP files (e.g. /lustre/orion/cfd164/proj-shared/3d_case/p1/sol/sol_2896000_aver.bp)
 * using the CompressMGARDCentroidOperator ADIOS2 plugin (cell-average +
 * nodal-residual decomposition + MGARD).
 *
 * Usage:
 *   mgardCentroid_adios_ge <input.bp> <output.bp> <rel_tolerance> [ebratio]
 *
 *   input.bp       - source BP file (contains /volume/Elem/ElementConnectivity,
 *                    /volume/GridCoordinates/*, /volume/FlowSolution/*).
 *   output.bp      - compressed output (also re-stores connectivity so the
 *                    decompressor can recover bar_u from it).
 *   rel_tolerance  - relative tolerance ε; absolute tolerance per variable is
 *                    ε * (max - min) computed from var.Min() / var.Max().
 *   ebratio        - fraction of the budget allocated to the residual (default 0.5).
 *
 * The plugin's `meshfile` parameter points back to <input.bp>, which already
 * holds the uncompressed connectivity at /volume/Elem/ElementConnectivity.
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "adios2.h"
#include "mpi.h"

namespace
{
constexpr const char *ZONE = "/hpMusic_base/hpMusic_Zone";
constexpr const char *FLOW_PREFIX = "/hpMusic_base/hpMusic_Zone/FlowSolution/";
constexpr const char *CONN_VAR = "/hpMusic_base/hpMusic_Zone/Elem/ElementConnectivity";
constexpr const char *ETYPE_ATTR = "/hpMusic_base/hpMusic_Zone/Elem/ElementType";

std::string to_string_ld(long double number)
{
    long double temp = number;
    long double integerPart = std::floor(temp);
    int frac = 0;
    while (temp - integerPart > 0)
    {
        temp = temp * 10;
        integerPart = std::floor(temp);
        frac++;
    }
    frac = std::max(1, frac);
    std::stringstream s;
    s.precision(frac);
    s << std::fixed << number;
    return s.str();
}

size_t NodesPerCellFromElementType(const std::string &etype)
{
    if (etype == "HEXA_8")  return 8;
    if (etype == "TETRA_4") return 4;
    if (etype == "PENTA_6") return 6; // CGNS prism
    if (etype == "PYRA_5")  return 5;
    if (etype == "QUAD_4")  return 4;
    if (etype == "TRI_3")   return 3;
    return 0;
}
} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, np_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    if (argc < 4 || argc > 6)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0]
                      << " <input.bp> <output.bp> <rel_tolerance> [ebratio [n_blocks]]\n";
        }
        MPI_Finalize();
        return 1;
    }
    std::string inputFile(argv[1]);
    std::string outputFile(argv[2]);
    double relTol = std::stod(argv[3]);
    double ebratio = (argc >= 5) ? std::stod(argv[4]) : 0.5;
    // n_blocks from command line (overrides MAX_BLOCKS env var when provided).
    size_t maxBlocksArg = (argc == 6) ? static_cast<size_t>(std::stoull(argv[5]))
                                      : std::numeric_limits<size_t>::max();

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO writer_io = ad.DeclareIO("Output");

    if (rank == 0)
    {
        std::cout << "Input file:    " << inputFile << "\n"
                  << "Output file:   " << outputFile << "\n"
                  << "Rel tolerance: " << relTol << "\n"
                  << "ebratio:       " << ebratio << " (residual fraction)\n";
    }

    adios2::Engine reader = reader_io.Open(inputFile, adios2::Mode::Read);
    auto step_status = reader.BeginStep(adios2::StepMode::Read, 10.0f);
    if (step_status != adios2::StepStatus::OK)
    {
        if (rank == 0)
            std::cerr << "Failed to BeginStep on " << inputFile << "\n";
        reader.Close();
        MPI_Finalize();
        return 1;
    }

    // ----- Discover FlowSolution variables and element type ----------------
    std::map<std::string, adios2::Params> available = reader_io.AvailableVariables();
    std::vector<std::string> varNames;
    for (auto &p : available)
    {
        const std::string &name = p.first;
        if (name.rfind(FLOW_PREFIX, 0) == 0)
        {
            auto it = p.second.find("Type");
            if (it != p.second.end() && it->second == "double")
                varNames.push_back(name);
        }
    }
    // FLOW_VAR_LIST=P_aver,Rho_aver : keep only the listed short names.
    if (const char *vlist = std::getenv("FLOW_VAR_LIST"); vlist && *vlist)
    {
        std::istringstream ss(vlist);
        std::string tok;
        std::vector<std::string> allowed;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) allowed.push_back(std::string(FLOW_PREFIX) + tok);
        std::vector<std::string> filtered;
        for (auto &n : varNames)
            for (auto &a : allowed)
                if (n == a) { filtered.push_back(n); break; }
        varNames = std::move(filtered);
    }

    if (varNames.empty())
    {
        if (rank == 0)
            std::cerr << "No double FlowSolution variables found under " << FLOW_PREFIX << "\n";
        reader.Close();
        MPI_Finalize();
        return 1;
    }

    // Element type from attribute.
    std::string elemType = "HEXA_8";
    {
        auto attrs = reader_io.AvailableAttributes();
        auto it = attrs.find(ETYPE_ATTR);
        if (it != attrs.end())
        {
            auto a = reader_io.InquireAttribute<std::string>(ETYPE_ATTR);
            if (a) elemType = a.Data().front();
        }
    }
    size_t nodesPerCell = NodesPerCellFromElementType(elemType);
    if (!nodesPerCell)
    {
        if (rank == 0)
            std::cerr << "Unsupported element type '" << elemType << "'\n";
        reader.Close();
        MPI_Finalize();
        return 1;
    }
    if (rank == 0)
    {
        std::cout << "Zone:          " << ZONE << "\n"
                  << "ElementType:   " << elemType
                  << "  (nodes_per_cell = " << nodesPerCell << ")\n"
                  << "Variables (" << varNames.size() << "):\n";
        for (auto &v : varNames) std::cout << "  - " << v << "\n";
    }

    // ----- Open writer -----------------------------------------------------
    adios2::Engine writer = writer_io.Open(outputFile, adios2::Mode::Write);

    // Re-emit connectivity (passthrough) so the decompressor can read it from
    // the output file as well.
    auto vConnIn = reader_io.InquireVariable<int64_t>(CONN_VAR);
    if (!vConnIn)
    {
        if (rank == 0) std::cerr << "Connectivity variable not found: " << CONN_VAR << "\n";
        reader.Close(); writer.Close(); MPI_Finalize(); return 1;
    }
    auto vConnOut = writer_io.DefineVariable<int64_t>(CONN_VAR, {}, {}, {adios2::UnknownDim});
    writer_io.DefineAttribute<std::string>(ETYPE_ATTR, elemType);

    // Compressed FlowSolution outputs.
    std::vector<adios2::Variable<double>> varOut(varNames.size());
    for (size_t i = 0; i < varNames.size(); ++i)
        varOut[i] = writer_io.DefineVariable<double>(varNames[i], {}, {}, {adios2::UnknownDim});

    // ----- Discover total number of blocks (use first variable) ------------
    auto v0 = reader_io.InquireVariable<double>(varNames[0]);
    size_t nBlocks = reader.BlocksInfo(v0, 0).size();

    // Contiguous block distribution to preserve ordering in output.
    size_t blocksPerRank = (nBlocks + np_size - 1) / np_size;
    size_t startBlock = static_cast<size_t>(rank) * blocksPerRank;
    size_t endBlock = std::min(startBlock + blocksPerRank, nBlocks);
    std::cout << "rank " << rank << ": blocks [" << startBlock << ", " << endBlock << ")\n";

    // ----- Operator parameters template ------------------------------------
    adios2::Params params;
    params["PluginName"]            = "mgardCentroid";
    params["verbose"]               = "0";
    // Set CENTROID_GPU=1 at runtime to use the GPU (HIP) operator.
    const bool useGPU = (std::getenv("CENTROID_GPU") != nullptr &&
                         std::string(std::getenv("CENTROID_GPU")) == "1");
    params["PluginLibrary"]         = useGPU ? "CompressMGARDCentroidOperator_GPU"
                                             : "CompressMGARDCentroidOperator";
    params["meshfile"]              = inputFile;       // connectivity source
    params["connectivity_variable"] = CONN_VAR;
    params["nodes_per_cell"]        = std::to_string(nodesPerCell);
    params["ebratio"]               = to_string_ld(ebratio);
    params["mode"]                  = "ABS";
    // Allow override via env var (MGARD_RESIDUAL_METHOD={mgard|huffman}) for
    // experimentation; default to huffman (typically denser on quantized
    // small-magnitude residuals).
    {
        const char *rm = std::getenv("MGARD_RESIDUAL_METHOD");
        params["residual_method"] = (rm && *rm) ? std::string(rm) : std::string("huffman");
    }

    using Clock = std::chrono::steady_clock;
    auto tStartAll = Clock::now();
    size_t totalBytesIn = 0;

    // Optional smoke-test limits (env-var overrides).
    auto envSize = [](const char *k, size_t def) {
        const char *v = std::getenv(k);
        return (v && *v) ? static_cast<size_t>(std::stoull(v)) : def;
    };
    size_t maxSteps  = envSize("MAX_STEPS",  std::numeric_limits<size_t>::max());
    size_t maxVars   = envSize("MAX_VARS",   std::numeric_limits<size_t>::max());
    // Command-line n_blocks takes priority over MAX_BLOCKS env var.
    size_t maxBlocks = (maxBlocksArg != std::numeric_limits<size_t>::max())
                       ? maxBlocksArg
                       : envSize("MAX_BLOCKS", std::numeric_limits<size_t>::max());
    // NO_PASSTHROUGH=1 : skip writing connectivity so output contains only
    // compressed FlowSolution data, enabling accurate CR measurement.
    bool no_passthrough = (std::getenv("NO_PASSTHROUGH") != nullptr);
    if (rank == 0 && no_passthrough)
        std::cout << "NO_PASSTHROUGH=1: connectivity will NOT be written.\n";
    if (rank == 0 && (maxSteps != std::numeric_limits<size_t>::max() ||
                      maxVars  != std::numeric_limits<size_t>::max() ||
                      maxBlocks != std::numeric_limits<size_t>::max()))
    {
        std::cout << "Limits: MAX_STEPS=" << maxSteps
                  << " MAX_VARS=" << maxVars
                  << " MAX_BLOCKS=" << maxBlocks << "\n";
    }

    int ts = 0;
    bool firstStep = true;
    while (true)
    {
        if (!firstStep)
        {
            // 0.0f timeout: static BP files have all steps on disk already;
            // a non-zero timeout blocks for that many seconds when there are
            // no more steps, making the program appear to hang after the last step.
            auto st = reader.BeginStep(adios2::StepMode::Read, 0.0f);
            if (st != adios2::StepStatus::OK)
                break;  // EndOfStream or NotReady -- either way we are done.
        }
        firstStep = false;

        writer.BeginStep();
        if (rank == 0)
            std::cout << "Step " << reader.CurrentStep() << ":\n";

        // Passthrough connectivity once (step 0 only -- it's time-invariant).
        // Skipped when NO_PASSTHROUGH=1 so output contains only compressed
        // FlowSolution data for accurate compression-ratio measurement.
        if (ts == 0 && !no_passthrough)
        {
            for (size_t bid = startBlock; bid < endBlock; ++bid)
            {
                vConnIn.SetBlockSelection(bid);
                std::vector<int64_t> conn;
                reader.Get(vConnIn, conn, adios2::Mode::Sync);
                vConnOut.SetSelection({{}, {conn.size()}});
                writer.Put(vConnOut, conn.data(), adios2::Mode::Sync);
            }
        }

        // Compress FlowSolution variables block by block.
        size_t varCount = std::min(varNames.size(), maxVars);
        for (size_t i = 0; i < varCount; ++i)
        {
            auto vIn = reader_io.InquireVariable<double>(varNames[i]);
            double minv = vIn.Min(), maxv = vIn.Max();
            double absTol = relTol * (maxv - minv);
            params["tolerance"] = to_string_ld(absTol);
            if (rank == 0)
                std::cout << "  " << varNames[i] << ": min/max=" << minv << "/" << maxv
                          << ", abs_tol=" << absTol << "\n";

            size_t bEnd = std::min(endBlock, startBlock + maxBlocks);
            for (size_t bid = startBlock; bid < bEnd; ++bid)
            {
                std::cout << "Compress block " << bid << "\n";
                vIn.SetBlockSelection(bid);
                std::vector<double> buf;
                reader.Get(vIn, buf, adios2::Mode::Sync);
                totalBytesIn += buf.size() * sizeof(double);

                params["blockid"] = std::to_string(bid);
                varOut[i].RemoveOperations();
                varOut[i].AddOperation("plugin", params);
                varOut[i].SetSelection({{}, {buf.size()}});
                writer.Put<double>(varOut[i], buf.data(), adios2::Mode::Sync);
            }
        }

        writer.EndStep();
        reader.EndStep();
        ts++;
        if (static_cast<size_t>(ts) >= maxSteps) break;
    }

    reader.Close();
    writer.Close();

    unsigned long long localBytes = static_cast<unsigned long long>(totalBytesIn);
    unsigned long long globalBytes = 0;
    MPI_Reduce(&localBytes, &globalBytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double secs = std::chrono::duration<double>(Clock::now() - tStartAll).count();
        double gb = globalBytes / (1024.0 * 1024.0 * 1024.0);
        std::cout << "Total uncompressed input: " << gb << " GB, wall time " << secs << " s\n";
    }

    MPI_Finalize();
    return 0;
}
