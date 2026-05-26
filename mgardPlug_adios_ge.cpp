#include <chrono>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include "SystemTools.hpp"

#include "adios2.h"
#include "mgard/compress_x.hpp"
#include "mpi.h"
#include <chrono>
#include <time.h>
#include <zstd.h>

const char *FLOWPREFIX = "/hpMusic_base/hpMusic_Zone/FlowSolution/";
int rank, np_size;
MPI_Comm comm;

string to_string_ld(long double number)
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

void GetVarsAndBlocks(std::string &inputFileName, const char *prefix,
                      std::vector<std::string> &varNames, size_t *nBlocks)
{
    adios2::ADIOS ad(comm);
    adios2::IO io = ad.DeclareIO("InputOnce");
    io.SetParameter("SelectSteps", "0");
    auto e = io.Open(inputFileName, adios2::Mode::ReadRandomAccess);
    auto varlist = io.AvailableVariables(true);
    for (auto &v : varlist)
    {
        if (StringStartsWith(v.first, prefix))
        {
            varNames.push_back(v.first);
        }
    }
    *nBlocks = 0;
    if (varNames.size())
    {
        auto var = io.InquireVariable(varNames[0]);
        auto bi = e.BlocksInfo(var, 0);
        *nBlocks = bi.size();
    }
    e.Close();
}

/* Timers */
typedef std::chrono::duration<double> Seconds;
typedef std::chrono::time_point<std::chrono::steady_clock,
                                std::chrono::duration<double, std::chrono::steady_clock::period>>
    TimePoint;

struct Timers
{
    double total;
    double read;
    double compress;
    double write;
};

template <class T>
adios2::Variable<T> InquireVariable(adios2::IO &io, adios2::Engine &engine, const char *varname)
{
    auto var = io.InquireVariable<T>(varname);
    if (!var)
        std::cout << "ERROR:    rank " << rank << "  " << varname << " not found in "
                  << engine.Name() << std::endl;
    return var;
}

void CopyMesh(adios2::Engine &reader, adios2::IO &reader_io, adios2::Engine &writer,
              adios2::IO &writer_io, int blockId, int rankBlock_num, Timers *timers)
{
    if (rank == 0)
        std::cout << "    Copy mesh in first step " << std::endl;

    auto vxIn = InquireVariable<double>(reader_io, reader,
                                        "/hpMusic_base/hpMusic_Zone/GridCoordinates/CoordinateX");
    auto vyIn = InquireVariable<double>(reader_io, reader,
                                        "/hpMusic_base/hpMusic_Zone/GridCoordinates/CoordinateY");
    auto vzIn = InquireVariable<double>(reader_io, reader,
                                        "/hpMusic_base/hpMusic_Zone/GridCoordinates/CoordinateZ");
    auto vconnIn = InquireVariable<int64_t>(reader_io, reader,
                                            "/hpMusic_base/hpMusic_Zone/Elem/ElementConnectivity");
    auto vrangeIn =
        InquireVariable<int64_t>(reader_io, reader, "/hpMusic_base/hpMusic_Zone/Elem/ElementRange");
    auto vnelemIn =
        InquireVariable<int32_t>(reader_io, reader, "/hpMusic_base/hpMusic_Zone/n_total_elem");
    auto vnnodeIn =
        InquireVariable<int32_t>(reader_io, reader, "/hpMusic_base/hpMusic_Zone/n_total_node");
    auto vphysdimIn = InquireVariable<int32_t>(reader_io, reader, "/hpMusic_base/physdim");
    auto vcelldimIn = InquireVariable<int32_t>(reader_io, reader, "/hpMusic_base/celldim");

    auto vxOut = writer_io.DefineVariable<double>(
        "/hpMusic_base/hpMusic_Zone/GridCoordinates/CoordinateX", {}, {}, {adios2::UnknownDim});
    auto vyOut = writer_io.DefineVariable<double>(
        "/hpMusic_base/hpMusic_Zone/GridCoordinates/CoordinateY", {}, {}, {adios2::UnknownDim});
    auto vzOut = writer_io.DefineVariable<double>(
        "/hpMusic_base/hpMusic_Zone/GridCoordinates/CoordinateZ", {}, {}, {adios2::UnknownDim});
    auto vconnOut = writer_io.DefineVariable<int64_t>(
        "/hpMusic_base/hpMusic_Zone/Elem/ElementConnectivity", {}, {}, {adios2::UnknownDim});
    auto vrangeOut = writer_io.DefineVariable<int64_t>(
        "/hpMusic_base/hpMusic_Zone/Elem/ElementRange", {}, {}, {adios2::UnknownDim});
    auto vnelemOut = writer_io.DefineVariable<int32_t>("/hpMusic_base/hpMusic_Zone/n_total_elem",
                                                       {adios2::LocalValueDim});
    auto vnnodeOut = writer_io.DefineVariable<int32_t>("/hpMusic_base/hpMusic_Zone/n_total_node",
                                                       {adios2::LocalValueDim});
    auto vphysdimOut = writer_io.DefineVariable<int32_t>("/hpMusic_base/physdim");
    auto vcelldimOut = writer_io.DefineVariable<int32_t>("/hpMusic_base/celldim");

    /* Compressing the coordinates needs to be evaluated*/
    /*
    double rel_tol = 0.001;
    {
        double minv = vxIn.Min();
        double maxv = vxIn.Max();
        double abs_tol = rel_tol * (maxv - minv);
        std::string tolstr = to_string_ld(abs_tol);
        vxOut.AddOperation("mgard", {{"tolerance", tolstr}, {"mode", "ABS"}});
        if (rank == 0)
            std::cout << "    Coordinate X: min/max = " << minv << "/" << maxv
                      << ", tol = " << abs_tol << std::endl;
    }

    {
        double minv = vyIn.Min();
        double maxv = vyIn.Max();
        double abs_tol = rel_tol * (maxv - minv);
        std::string tolstr = to_string_ld(abs_tol);
        vyOut.AddOperation("mgard", {{"tolerance", tolstr}, {"mode", "ABS"}});
        if (rank == 0)
            std::cout << "    Coordinate Y: min/max = " << minv << "/" << maxv
                      << ", tol = " << abs_tol << std::endl;
    }

    {
        double minv = vzIn.Min();
        double maxv = vzIn.Max();
        double abs_tol = rel_tol * (maxv - minv);
        std::string tolstr = to_string_ld(abs_tol);
        vzOut.AddOperation("mgard", {{"tolerance", tolstr}, {"mode", "ABS"}});
        if (rank == 0)
            std::cout << "    Coordinate Z: min/max = " << minv << "/" << maxv
                      << ", tol = " << abs_tol << std::endl;
    }
    */

    for (int bid = blockId; bid < rankBlock_num; bid++)
    {
        std::vector<double> x, y, z;
        std::vector<int64_t> conn, range;
        int32_t nelem, nnode;
        int32_t physdim, celldim;

        TimePoint start_read = std::chrono::steady_clock::now();
        vxIn.SetBlockSelection(bid);
        vyIn.SetBlockSelection(bid);
        vzIn.SetBlockSelection(bid);
        vconnIn.SetBlockSelection(bid);
        vrangeIn.SetBlockSelection(bid);
        vnelemIn.SetBlockSelection(bid);
        vnnodeIn.SetBlockSelection(bid);
        reader.Get(vxIn, x, adios2::Mode::Deferred);
        reader.Get(vyIn, y, adios2::Mode::Deferred);
        reader.Get(vzIn, z, adios2::Mode::Deferred);
        reader.Get(vconnIn, conn, adios2::Mode::Deferred);
        reader.Get(vrangeIn, range, adios2::Mode::Deferred);
        reader.Get(vnelemIn, &nelem, adios2::Mode::Deferred);
        reader.Get(vnnodeIn, &nnode, adios2::Mode::Deferred);
        reader.Get(vphysdimIn, &physdim, adios2::Mode::Deferred);
        reader.Get(vcelldimIn, &celldim, adios2::Mode::Deferred);
        reader.PerformGets();
        TimePoint end_read = std::chrono::steady_clock::now();
        timers->read += Seconds(end_read - start_read).count();

        TimePoint start_compress = std::chrono::steady_clock::now();
        vxOut.SetSelection(adios2::Box<adios2::Dims>({}, {x.size()}));
        vyOut.SetSelection(adios2::Box<adios2::Dims>({}, {y.size()}));
        vzOut.SetSelection(adios2::Box<adios2::Dims>({}, {z.size()}));
        vconnOut.SetSelection(adios2::Box<adios2::Dims>({}, {conn.size()}));
        vrangeOut.SetSelection(adios2::Box<adios2::Dims>({}, {range.size()}));

        writer.Put(vxOut, x.data(), adios2::Mode::Sync);
        writer.Put(vyOut, y.data(), adios2::Mode::Sync);
        writer.Put(vzOut, z.data(), adios2::Mode::Sync);
        writer.Put(vconnOut, conn.data(), adios2::Mode::Sync);
        writer.Put(vrangeOut, range.data(), adios2::Mode::Sync);
        writer.Put(vnelemOut, nelem, adios2::Mode::Sync);
        writer.Put(vnnodeOut, nnode, adios2::Mode::Sync);
        if (!rank)
        {
            writer.Put(vphysdimOut, physdim, adios2::Mode::Sync);
            writer.Put(vcelldimOut, celldim, adios2::Mode::Sync);
        }
        TimePoint end_compress = std::chrono::steady_clock::now();
        timers->compress += Seconds(end_compress - start_compress).count();
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np_size);

    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << "  input  mapfile  output  tolerance\n"
                  << "  input     :  sol or sol_aver BP file\n"
                  << "  mapping   :  mesh to grid mapping file (MeshGrid output)\n"
                  << "  output    :  compressed data file \n"
                  << "  tolerance :  floating point accuracy value, relative error in data"
                  << std::endl;
        return EXIT_FAILURE;
    }

    int cnt_argv = 1;
    std::string inputPath(argv[cnt_argv++]);
    std::string mapPath(argv[cnt_argv++]);
    std::string outputPath(argv[cnt_argv++]);
    double tol = std::stof(argv[cnt_argv++]);

    adios2::ADIOS ad(comm);
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::IO writer_io = ad.DeclareIO("Output");

    if (rank == 0)
    {
        std::cout << "read : " << inputPath << "\n";
        std::cout << "       " << mapPath << "\n";
        std::cout << "write: " << outputPath << "\n";
    }

    std::cout << "rank " << rank << ": passed start\n";

    size_t maxBlocks;
    std::vector<std::string> var_name;
    GetVarsAndBlocks(inputPath, FLOWPREFIX, var_name, &maxBlocks);
    int n_vars = var_name.size();

    if (rank == 0)
    {
        std::cout << "Working on " << n_vars << " variables and " << maxBlocks << " blocks:\n";
        for (int i = 0; i < n_vars; i++)
        {
            std::cout << "    " << var_name[i] << "\n";
        }
    }

    if (!n_vars)
        return 0;

    adios2::Engine reader = reader_io.Open(inputPath, adios2::Mode::Read);
    adios2::Engine writer = writer_io.Open(outputPath, adios2::Mode::Write);

    std::vector<adios2::Variable<double>> var_out(n_vars);
    for (int i = 0; i < n_vars; i++)
    {
        var_out[i] = writer_io.DefineVariable<double>(var_name[i], {}, {}, {adios2::UnknownDim});
    }

    int nBlocks = (int)((double)maxBlocks / (double)np_size);
    int blockId = rank * nBlocks;
    int rankBlock_num =
        (rank < np_size - 1) ? blockId + nBlocks : blockId + nBlocks + (maxBlocks % np_size);

    adios2::Params params;
    params["PluginName"] = "mgardReMesh";
    params["PluginLibrary"] = "CompressMGARDMeshToGridOperator";
    params["meshfile"] = mapPath;
    params["ebratio"] = "0.7";
    params["mode"] = "ABS";
    params["residual_method"]="huffman";

    std::cout << "rank " << rank << " will read in blocks " << blockId << ".." << rankBlock_num - 1
              << "\n";

    Timers timers = {0.0, 0.0, 0.0, 0.0};
    TimePoint start_total = std::chrono::steady_clock::now();

    /****
       First, copy the Mesh to Grid Mapping into the output file as part of the first output step
    *****/
    adios2::IO map_io = ad.DeclareIO("InputMap");
    adios2::Engine mapreader = map_io.Open(mapPath, adios2::Mode::ReadRandomAccess);

    adios2::Variable<uint64_t> vinGridDim =
        map_io.InquireVariable<uint64_t>("__mesh_grid_mapping__/GridDim");
    adios2::Variable<uint8_t> vinGridSparsity =
        map_io.InquireVariable<uint8_t>("__mesh_grid_mapping__/GridSparsity");
    adios2::Variable<uint64_t> vinMeshGridCluster =
        map_io.InquireVariable<uint64_t>("__mesh_grid_mapping__/MeshGridCluster");
    adios2::Variable<uint64_t> vinMeshGridMap =
        map_io.InquireVariable<uint64_t>("__mesh_grid_mapping__/MeshGridMap");

    adios2::Variable<uint64_t> voutGridDim = writer_io.DefineVariable<uint64_t>(
        "__mesh_grid_mapping__/GridDim", {}, {}, {adios2::UnknownDim});
    adios2::Variable<uint8_t> voutGridSparsity = writer_io.DefineVariable<uint8_t>(
        "__mesh_grid_mapping__/GridSparsity", {}, {}, {adios2::UnknownDim});
    adios2::Variable<uint64_t> voutMeshGridCluster = writer_io.DefineVariable<uint64_t>(
        "__mesh_grid_mapping__/MeshGridCluster", {}, {}, {adios2::UnknownDim});
    adios2::Variable<uint64_t> voutMeshGridMap = writer_io.DefineVariable<uint64_t>(
        "__mesh_grid_mapping__/MeshGridMap", {}, {}, {adios2::UnknownDim});

    writer.BeginStep();
   /*  
    for (int bid = blockId; bid < rankBlock_num; bid++)
    {
        std::vector<uint64_t> gridDim;
        std::vector<uint8_t> gridSparsity;
        std::vector<uint64_t> meshGridCluster;
        std::vector<uint64_t> meshGridMap;
        vinGridDim.SetBlockSelection(bid);
        vinGridSparsity.SetBlockSelection(bid);
        vinMeshGridCluster.SetBlockSelection(bid);
        vinMeshGridMap.SetBlockSelection(bid);
        mapreader.Get(vinMeshGridCluster, meshGridCluster);
        mapreader.Get(vinGridDim, gridDim);
        mapreader.Get(vinGridSparsity, gridSparsity);
        mapreader.Get(vinMeshGridMap, meshGridMap);
        mapreader.PerformGets();
        voutGridDim.SetSelection({{}, {vinGridDim.Count()}});
        voutGridSparsity.SetSelection({{}, {vinGridSparsity.Count()}});
        voutMeshGridCluster.SetSelection({{}, {vinMeshGridCluster.Count()}});
        voutMeshGridMap.SetSelection({{}, {vinMeshGridMap.Count()}});
        writer.Put(voutGridDim, gridDim.data(), adios2::Mode::Sync);
        writer.Put(voutGridSparsity, gridSparsity.data(), adios2::Mode::Sync);
        writer.Put(voutMeshGridCluster, meshGridCluster.data(), adios2::Mode::Sync);
        writer.Put(voutMeshGridMap, meshGridMap.data(), adios2::Mode::Sync);
    }
   */ 
    int ts = 0;
    while (true)
    {
        adios2::StepStatus read_status = reader.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady)
        {
            // std::cout << "Stream not ready yet. Waiting...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK)
        {
            break;
        }

        size_t step = reader.CurrentStep();
        if (rank == 0)
            std::cout << "Process step " << step << ": " << std::endl;

        if (ts > 0)
        {
            writer.BeginStep();
        }
        else
        {
            /* In first step only, copy the mesh variables*/
        //    CopyMesh(reader, reader_io, writer, writer_io, blockId, rankBlock_num, &timers);
        }

        for (int i = 0; i < n_vars; i++)
        {
            adios2::Variable<double> var_ad2;
            var_ad2 = reader_io.InquireVariable<double>(var_name[i]);
            double minv = var_ad2.Min();
            double maxv = var_ad2.Max();
            double abs_tol = tol * (maxv - minv);
            params["tolerance"] = to_string_ld(abs_tol);
            if (rank == 0)
                std::cout << "    " << var_name[i].c_str() << ": min/max = " << minv << "/" << maxv
                          << ", tol = " << abs_tol << std::endl;

            for (int bid = blockId; bid < rankBlock_num; bid++)
            {
                TimePoint start_read = std::chrono::steady_clock::now();
                var_ad2.SetBlockSelection(bid);
                std::vector<double> var_in;
                reader.Get(var_ad2, var_in, adios2::Mode::Sync);
                reader.PerformGets();
                TimePoint end_read = std::chrono::steady_clock::now();
                timers.read += Seconds(end_read - start_read).count();

                TimePoint start_compress = std::chrono::steady_clock::now();
                params["blockid"] = std::to_string(bid);
                var_out[i].RemoveOperations();
                var_out[i].AddOperation("plugin", params);
                var_out[i].SetSelection(adios2::Box<adios2::Dims>({}, {var_in.size()}));

                writer.Put<double>(var_out[i], var_in.data(), adios2::Mode::Sync);
                TimePoint end_compress = std::chrono::steady_clock::now();
                timers.compress += Seconds(end_compress - start_compress).count();
            }
        }
        TimePoint start_write = std::chrono::steady_clock::now();
        writer.EndStep();
        TimePoint end_write = std::chrono::steady_clock::now();
        timers.write += Seconds(end_write - start_write).count();
        reader.EndStep();
        ts++;
    }
    reader.Close();
    writer.Close();
    TimePoint end_total = std::chrono::steady_clock::now();
    timers.total = Seconds(end_total - start_total).count();

    /*
        Timings
    */
    std::vector<Timers> tv(np_size);
    MPI_Gather(&timers, sizeof(Timers), MPI_CHAR, tv.data(), sizeof(Timers), MPI_CHAR, 0, comm);

    if (!rank)
    {
        Timers maxs = {0.0, 0.0, 0.0, 0.0};
        std::cout << "Rank :      read       compress    write      total      (seconds)\n";
        std::cout << "-----------------------------------------------------------------------\n";
        for (int r = 0; r < np_size; ++r)
        {
            std::cout << std::setw(5) << r << ":   " << std::fixed << std::setw(8)
                      << std::setprecision(3) << tv[r].read << "   " << std::setw(8)
                      << std::setprecision(3) << tv[r].compress << "   " << std::setw(8)
                      << std::setprecision(3) << tv[r].write << "   " << std::setw(8)
                      << std::setprecision(3) << tv[r].total << "\n";
            maxs.read = (maxs.read > tv[r].read ? maxs.read : tv[r].read);
            maxs.compress = (maxs.compress > tv[r].compress ? maxs.compress : tv[r].compress);
            maxs.write = (maxs.write > tv[r].write ? maxs.write : tv[r].write);
            maxs.total = (maxs.total > tv[r].total ? maxs.total : tv[r].total);
        }
        std::cout << "  max:   " << std::fixed << std::setw(8) << std::setprecision(3) << maxs.read
                  << "   " << std::setw(8) << std::setprecision(3) << maxs.compress << "   "
                  << std::setw(8) << std::setprecision(3) << maxs.write << "   " << std::setw(8)
                  << std::setprecision(3) << maxs.total << "\n";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
