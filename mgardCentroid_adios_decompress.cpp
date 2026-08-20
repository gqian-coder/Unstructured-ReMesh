/*
 * mgardCentroid_adios_decompress.cpp
 *
 * Decompression test driver for the CompressMGARDCentroidOperator ADIOS2
 * plugin. Reads a BP file previously written by `mgardCentroid_adios_ge`
 * and writes a BP file with the reconstructed FlowSolution variables.
 *
 * ADIOS2 dispatches InverseOperate() automatically from the operator chain
 * recorded in the variable metadata, so we only need:
 *   - ADIOS2_PLUGIN_PATH pointing at libCompressMGARDCentroidOperator.so
 *     (or libCompressMGARDCentroidOperator_GPU.so for the GPU build).
 *   - The centroid operator's runtime backend env vars:
 *       CENTROID_DEVICE = cpu | gpu
 *       MGARD_X_DEVICE_TYPE = SERIAL | HIP    (optional, back-compat)
 *
 * Usage:
 *   mgardCentroid_adios_decompress <compressed.bp> [decompressed.bp]
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "adios2.h"
#ifdef CMG_WRAPPER_HAVE_HIP
#include <hip/hip_runtime.h>
#endif
#include "mpi.h"

namespace
{
/* Zone prefix is configurable: this tool was hardcoded to the hpMusic zone and
 * therefore could not read /volume/ datasets at all. Override the zone with
 * CENTROID_ZONE (e.g. "/volume"), or give the full paths with
 * CENTROID_FLOW_PREFIX / CENTROID_CONN_VAR. */
const std::string FLOW_PREFIX = [] {
    if (const char *e = std::getenv("CENTROID_FLOW_PREFIX")) return std::string(e);
    if (const char *z = std::getenv("CENTROID_ZONE")) return std::string(z) + "/FlowSolution/";
    return std::string("/volume/FlowSolution/");
}();
const std::string CONN_VAR = [] {
    if (const char *e = std::getenv("CENTROID_CONN_VAR")) return std::string(e);
    if (const char *z = std::getenv("CENTROID_ZONE")) return std::string(z) + "/Elem/ElementConnectivity";
    return std::string("/volume/Elem/ElementConnectivity");
}();
} // namespace


/* ---------------- multi-GPU block distribution ------------------------
 * Mirrors mgardCentroid_adios_ge.cpp so compress and decompress agree.
 * Each rank binds to GPU (rank % ngpus); the plugin's InverseOperate runs
 * inline in this thread during Get(), so it uses that current device.
 *
 * CENTROID_BLOCK_DIST=contiguous (default) | roundrobin
 * NOTE: whichever the file was WRITTEN with, decompression is correct either
 * way -- the operator reads the blockId from the bitstream and loads the
 * matching mesh block. The distribution only decides which rank does the work
 * and, as on the write side, the order blocks land in the output file.
 */
enum class BlockDist { Contiguous, RoundRobin };

BlockDist ParseBlockDist()
{
    if (const char *e = std::getenv("CENTROID_BLOCK_DIST"))
    {
        std::string s(e);
        for (auto &c : s) c = (char)std::tolower((unsigned char)c);
        if (s == "roundrobin" || s == "rr" || s == "round_robin") return BlockDist::RoundRobin;
    }
    return BlockDist::Contiguous;
}

bool OwnsBlock(BlockDist d, size_t bid, int rank, int np, size_t nBlocks)
{
    if (d == BlockDist::RoundRobin) return (bid % (size_t)np) == (size_t)rank;
    const size_t per   = (nBlocks + (size_t)np - 1) / (size_t)np;
    const size_t start = (size_t)rank * per;
    return bid >= start && bid < std::min(start + per, nBlocks);
}

int BindRankToGpu(int rank, int &ngpusOut)
{
    ngpusOut = 0;
#ifdef CMG_WRAPPER_HAVE_HIP
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n <= 0) return -1;
    ngpusOut = n;
    const int dev = rank % n;
    if (hipSetDevice(dev) != hipSuccess) return -1;
    return dev;
#else
    (void)rank;
    return -1;
#endif
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, np_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_size);

    const BlockDist blockDist = ParseBlockDist();
    int nGpus = 0;
    const int myGpu = BindRankToGpu(rank, nGpus);
    std::cout << "rank " << rank << "/" << np_size
              << (myGpu >= 0 ? " -> gpu " + std::to_string(myGpu) + "/" + std::to_string(nGpus)
                             : std::string(" -> cpu"))
              << "  dist=" << (blockDist == BlockDist::RoundRobin ? "roundrobin" : "contiguous")
              << "\n";

    if (argc < 2 || argc > 3)
    {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0]
                      << " <compressed.bp> [decompressed.bp]\n";
        MPI_Finalize();
        return 1;
    }
    std::string inputFile(argv[1]);
    std::string outputFile = (argc == 3)
        ? std::string(argv[2])
        : (inputFile.size() > 3 && inputFile.substr(inputFile.size() - 3) == ".bp"
               ? inputFile.substr(0, inputFile.size() - 3) + "_decompressed.bp"
               : inputFile + "_decompressed.bp");

    if (rank == 0)
    {
        std::cout << "Input  (compressed):   " << inputFile << "\n"
                  << "Output (decompressed): " << outputFile << "\n";
        if (const char *p = std::getenv("ADIOS2_PLUGIN_PATH"))
            std::cout << "ADIOS2_PLUGIN_PATH:    " << p << "\n";
        if (const char *d = std::getenv("CENTROID_DEVICE"))
            std::cout << "CENTROID_DEVICE:       " << d << "\n";
        if (const char *m = std::getenv("MGARD_X_DEVICE_TYPE"))
            std::cout << "MGARD_X_DEVICE_TYPE:   " << m << "\n";
    }

    adios2::ADIOS ad(MPI_COMM_WORLD);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO writer_io = ad.DeclareIO("Output");

    adios2::Engine reader = reader_io.Open(inputFile, adios2::Mode::Read);
    adios2::Engine writer = writer_io.Open(outputFile, adios2::Mode::Write);

    // ----- Discover variables on first step --------------------------------
    auto status = reader.BeginStep(adios2::StepMode::Read, 10.0f);
    if (status != adios2::StepStatus::OK)
    {
        if (rank == 0)
            std::cerr << "Failed to BeginStep on " << inputFile << "\n";
        reader.Close();
        writer.Close();
        MPI_Finalize();
        return 1;
    }

    auto avail = reader_io.AvailableVariables();
    std::vector<std::string> flowDouble;
    std::vector<std::string> flowFloat;
    for (auto &p : avail)
    {
        const std::string &name = p.first;
        if (name.rfind(FLOW_PREFIX, 0) != 0) continue;
        auto it = p.second.find("Type");
        if (it == p.second.end()) continue;
        if (it->second == "double") flowDouble.push_back(name);
        else if (it->second == "float") flowFloat.push_back(name);
    }
    bool hasConn = avail.count(CONN_VAR) > 0;
    if (rank == 0)
    {
        std::cout << "FlowSolution vars: " << flowDouble.size() << " double, "
                  << flowFloat.size() << " float; connectivity in file: "
                  << (hasConn ? "yes" : "no") << "\n";
    }

    // Self-contained decompression needs no help from this tool: the operator
    // embeds the connectivity/coordinate variable names in the bitstream and
    // defaults the mesh file to whatever file it is decoding. Setting the env
    // vars here used to do that job, but it tied the default to CONN_VAR
    // matching a variable in this file -- so naming a variable that was not
    // present silently disabled the default instead of overriding it.
    //
    // The two env vars are only for pointing at an EXTERNAL mesh, and they are
    // meaningful only as a pair, so reject a half-configured environment here
    // rather than letting it surface one block into the read.
    {
        const char *envMesh = std::getenv("CENTROID_MESHFILE");
        const char *envConn = std::getenv("CENTROID_CONN_VAR");
        const bool haveMesh = envMesh && *envMesh;
        const bool haveConn = envConn && *envConn;
        if (haveMesh != haveConn)
        {
            if (rank == 0)
                std::cerr << "ERROR: " << (haveMesh ? "CENTROID_MESHFILE" : "CENTROID_CONN_VAR")
                          << " is set but " << (haveMesh ? "CENTROID_CONN_VAR" : "CENTROID_MESHFILE")
                          << " is not. Set both to read an external mesh, or neither to read the "
                             "mesh from the compressed file.\n";
            reader.Close(); writer.Close(); MPI_Finalize(); return 1;
        }
        if (rank == 0)
        {
            if (haveMesh)
                std::cout << "External mesh: " << envMesh << " (" << envConn << ")\n";
            else
                std::cout << "Self-contained: connectivity/coordinates read from the compressed "
                             "file, named by the bitstream header.\n";
        }
    }
    if (flowDouble.empty() && flowFloat.empty())
    {
        if (rank == 0)
            std::cerr << "No FlowSolution variables found under " << FLOW_PREFIX << "\n";
        reader.Close(); writer.Close(); MPI_Finalize(); return 1;
    }

    // ----- Define output variables ----------------------------------------
    std::map<std::string, adios2::Variable<double>> outD;
    std::map<std::string, adios2::Variable<float>>  outF;
    for (auto &n : flowDouble)
        outD[n] = writer_io.DefineVariable<double>(n, {}, {}, {adios2::UnknownDim});
    for (auto &n : flowFloat)
        outF[n] = writer_io.DefineVariable<float>(n, {}, {}, {adios2::UnknownDim});
    adios2::Variable<int64_t> outConn;
    if (hasConn)
        outConn = writer_io.DefineVariable<int64_t>(CONN_VAR, {}, {}, {adios2::UnknownDim});

    using Clock = std::chrono::steady_clock;
    size_t totalBytesOut = 0;
    size_t totalBlocks   = 0;
    auto tStart = Clock::now();

    int ts = 0;
    bool firstStep = true;
    while (true)
    {
        if (!firstStep)
        {
            auto st = reader.BeginStep(adios2::StepMode::Read, 0.0f);
            if (st != adios2::StepStatus::OK) break;
        }
        firstStep = false;

        writer.BeginStep();
        if (rank == 0)
            std::cout << "Step " << reader.CurrentStep() << ":\n";

        // Passthrough connectivity (step 0 only — time-invariant).
        if (ts == 0 && hasConn)
        {
            auto vConn = reader_io.InquireVariable<int64_t>(CONN_VAR);
            size_t nB  = reader.BlocksInfo(vConn, 0).size();
            for (size_t bid = 0; bid < nB; ++bid)
            {
                if (!OwnsBlock(blockDist, bid, rank, np_size, nB)) continue;
                vConn.SetBlockSelection(bid);
                std::vector<int64_t> conn;
                reader.Get(vConn, conn, adios2::Mode::Sync);
                outConn.SetSelection({{}, {conn.size()}});
                writer.Put(outConn, conn.data(), adios2::Mode::Sync);
            }
        }

        // Decompress each FlowSolution variable, block by block.
        for (auto &n : flowDouble)
        {
            auto vIn = reader_io.InquireVariable<double>(n);
            size_t nB  = reader.BlocksInfo(vIn, ts).size();
            if (rank == 0)
                std::cout << "  " << n << ": " << nB << " blocks\n";
            for (size_t bid = 0; bid < nB; ++bid)
            {
                if (!OwnsBlock(blockDist, bid, rank, np_size, nB)) continue;
                vIn.SetBlockSelection(bid);
                std::vector<double> buf;
                reader.Get(vIn, buf, adios2::Mode::Sync); // triggers InverseOperate
                totalBytesOut += buf.size() * sizeof(double);
                totalBlocks   += 1;
                outD[n].SetSelection({{}, {buf.size()}});
                writer.Put(outD[n], buf.data(), adios2::Mode::Sync);
            }
        }
        for (auto &n : flowFloat)
        {
            auto vIn = reader_io.InquireVariable<float>(n);
            size_t nB  = reader.BlocksInfo(vIn, ts).size();
            if (rank == 0)
                std::cout << "  " << n << ": " << nB << " blocks\n";
            for (size_t bid = 0; bid < nB; ++bid)
            {
                if (!OwnsBlock(blockDist, bid, rank, np_size, nB)) continue;
                vIn.SetBlockSelection(bid);
                std::vector<float> buf;
                reader.Get(vIn, buf, adios2::Mode::Sync); // triggers InverseOperate
                totalBytesOut += buf.size() * sizeof(float);
                totalBlocks   += 1;
                outF[n].SetSelection({{}, {buf.size()}});
                writer.Put(outF[n], buf.data(), adios2::Mode::Sync);
            }
        }

        writer.EndStep();
        reader.EndStep();
        ts++;
    }

    reader.Close();
    writer.Close();

    auto tEnd = Clock::now();
    double secs = std::chrono::duration<double>(tEnd - tStart).count();
    if (rank == 0)
    {
        std::cout << "\n[decompress] steps=" << ts
                  << " blocks=" << totalBlocks
                  << " bytes_out=" << totalBytesOut
                  << " (" << (totalBytesOut / 1048576.0) << " MiB)"
                  << " time=" << secs << "s\n";
    }
    MPI_Finalize();
    return 0;
}
