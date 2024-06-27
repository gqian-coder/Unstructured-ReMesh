#include <cmath>
#include <fstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <dirent.h>

#include "adios2.h"
#include "mgard/compress.hpp"
#include <zstd.h>
#include "nonUniformMap.hpp"
#include <time.h>

const int blendR=0;

template <std::size_t N, typename Real>
void dequantizeDecode(const mgard::CompressedDataset<N, Real> &compressed, Real *dequantized) {
  const std::size_t ndof = compressed.hierarchy.ndof();
  mgard::MemoryBuffer<unsigned char> quantized =
      mgard::quantization_buffer(compressed.header, ndof);
  mgard::decompress(compressed.header, const_cast<void *>(compressed.data()),
             compressed.size(), quantized.data.get(), quantized.size);
  mgard::dequantize(compressed, quantized.data.get(), dequantized);

}

template <std::size_t N, typename Real>
mgard::CompressedDataset<N, Real>
quantizeEncode(const mgard::TensorMeshHierarchy<N, Real> &hierarchy, Real *const u,
         const Real tolerance) {
  const std::size_t ndof = hierarchy.ndof();
  mgard::pb::Header header;
  mgard::populate_defaults(header);
  hierarchy.populate(header);
  {
    mgard::pb::ErrorControl &e = *header.mutable_error_control();
    e.set_mode(mgard::pb::ErrorControl::ABSOLUTE);
    e.set_norm(mgard::pb::ErrorControl::S_NORM);
    e.set_s(0);
    e.set_tolerance(tolerance);
  }
  mgard::MemoryBuffer<unsigned char> quantized = mgard::quantization_buffer(header, ndof);
  mgard::quantize(hierarchy, header, Real(0), tolerance, u, quantized.data.get());
  mgard::MemoryBuffer<unsigned char> buffer =
      mgard::compress(header, quantized.data.get(), quantized.size);
  return mgard::CompressedDataset<N, Real>(hierarchy, header, 0, tolerance,
                                    buffer.data.release(), buffer.size);
}


int main(int argc, char **argv) {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Parse command line arguments
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0] << " dataPath"  << " filedName" ;
        std::cerr << " rel tolerance " << " split ratio (data : residual)";
        //std::cerr << " grid compression strategy (0: n-dim | 1: linear sparse)";
        std::cerr << " residual compression strategy (1: lossless | 2: quantization + lossless) | 3: lossy";
        std::cout << " grid mesh interpolation strategy (closest | interpolation)";
        std::cout << " radius used for grid mesh interpolation";
        std::cerr << " n-Dims " ;//<< " Smaple rate per dim ";
        std::cerr << " n-Vars " << " var names " ;
        std::cerr << " percentile of node spacing for grid selection" << std::endl;
        return EXIT_FAILURE;
    }

    int cnt_argv = 1;
    std::string dpath(argv[cnt_argv++]);
    std::string fname(argv[cnt_argv++]);
    std::cout << "Read in : " << dpath + fname << "\n";
    double s = 0; 
    std::string eb_str(argv[cnt_argv]);
    double tolerance = std::stof(argv[cnt_argv++]);
    double ratio_t  = std::stof(argv[cnt_argv++]);
    int option_grid;// = std::stoi(argv[cnt_argv++]); 
    int option_resi = std::stoi(argv[cnt_argv++]);
    std::cout << "relative tolerance: " << tolerance << ", split ratio: " << ratio_t  <<  ", residual compression option: " << option_resi << "\n";
    double radius = 0;
    std::string interpolation_option = argv[cnt_argv++];
    if (strcmp(interpolation_option.c_str(), "interpolation")==0) {
        radius = std::stof(argv[cnt_argv++]);
        std::cout << "Linear Shepard interpolation w/ radius = " << radius << " (scatter-->grid) & bilinear interpolation (grid --> scatter)\n";
    } else { 
        std::cout << "Closest interpolation (scatter --> grid) & mean-based interpolation (grid --> scatter) \n";
    }
    // dimensions used for compression
    int n_dims = std::stoi(argv[cnt_argv++]);
    std::vector<size_t>resampleRate(n_dims, 1);
    size_t nGridPt = 1;
    /*
    std::cout << "resample rate: ";
    for (int i=0; i<n_dims; i++) {
        resampleRate[i] = (size_t)std::stoi(argv[cnt_argv++]);
        nGridPt = nGridPt * resampleRate[i];
        std::cout << resampleRate[i]; 
        if (i<n_dims-1) std::cout << " x ";
    }
    std::cout << ", GridSize = " << nGridPt << "\n";
    */
    int n_vars = std::stoi(argv[cnt_argv++]);
    std::vector<std::string> var_name(n_vars);
    for (int i=0; i< n_vars; i++) {
        var_name[i] = argv[cnt_argv++];
        std::cout << "compression variable: " <<var_name[i]  << "\n";
    }
    double perc = std::stof(argv[cnt_argv++]);
    std::cout << "percentile of grid spacing used for resample rate calculation: " << perc << "\n";

    adios2::ADIOS ad(MPI_COMM_WORLD); 
    adios2::IO reader_io = ad.DeclareIO("Input");
    adios2::Engine reader = reader_io.Open(dpath + fname, adios2::Mode::Read);
    adios2::IO writer_io = ad.DeclareIO("Output");
    adios2::Engine writer = writer_io.Open("./vki/eb_" + eb_str + "/" + fname, adios2::Mode::Write);
    //adios2::Engine writer = writer_io.Open(fname, adios2::Mode::Write);

    std::vector<size_t> compressedBytes_1D(n_vars, 0);
    std::vector<size_t> resi_cSize(n_vars, 0), grid_cSize(n_vars, 0), data_size(n_vars, 0);
    std::vector<double> l2_comb(n_vars, 0), linf_comb(n_vars, 0);
    std::vector<double> l2_err(n_vars, 0), linf_err(n_vars, 0);
    std::vector<size_t> total_nbld(n_vars, 0);

    adios2::Variable<double> var_ad2;
    std::vector<double> spaceGrid(n_dims);
    std::vector<double> minvGrid(n_dims);

    size_t nNodePt;

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

        // read node coordinates
        std::vector<std::vector<double>> nodeCoord;
        std::vector<std::string> coordVarName{"CoordinateZ", "CoordinateY", "CoordinateX"};
        for (int i=0; i<n_dims; i++) {
            var_ad2 = reader_io.InquireVariable<double>(coordVarName[i]);
            nNodePt = var_ad2.Shape()[0];
            var_ad2.SetSelection(adios2::Box<adios2::Dims>({0}, {nNodePt}));
            std::vector<double> var_in;
            reader.Get<double>(var_ad2, var_in, adios2::Mode::Sync);
            reader.PerformGets();
            nodeCoord.push_back(var_in);
            std::cout << "nNodePt = " << nodeCoord[i].size() << "\n";
        }

        // read connectivity
        std::vector<double> nodeConnc(0);
        var_ad2 = reader_io.InquireVariable<double>("Conn");
        if (var_ad2) {
            var_ad2.SetSelection(adios2::Box<adios2::Dims>({0}, {var_ad2.Shape()[0]})); 
            reader.Get<double>(var_ad2, nodeConnc, adios2::Mode::Sync);
            reader.PerformGets();
        } else {
            adios2::Variable<int64_t> var_ad2_int = reader_io.InquireVariable<int64_t>("Conn");
            if (var_ad2_int) {
                std::vector<int64_t> nodeConnc_int;
                reader.Get<int64_t>(var_ad2_int, nodeConnc_int, adios2::Mode::Sync);
                reader.PerformGets();
                nodeConnc.resize(nodeConnc_int.size());
                std::transform(nodeConnc_int.begin(), nodeConnc_int.end(), nodeConnc.begin(),
                   [](int64_t value) { return static_cast<double>(value); });
                nodeConnc_int.clear();
            }       
        }
        //std::vector<size_t> GridDim(n_dims);
        sel_Gridspace(nodeConnc, nodeCoord, n_dims, perc, spaceGrid, resampleRate);
        std::cout << "resample rate: ";
        for (int i=0; i<n_dims; i++) {
            nGridPt = nGridPt * resampleRate[i];
            std::cout << resampleRate[i];
            if (i<n_dims-1) std::cout << " x ";
        }
        std::cout << ", at spacing: ";
        for (int i=0; i<n_dims; i++) std::cout << spaceGrid[i] << " , ";
        std::cout << "\n";

        std::cout << "number of structured mesh nodes: " << nNodePt << "\n";
        std::cout << "Resample rate: " << (double)nGridPt / nNodePt << "\n";
        // find the closest grid vertices for each node
        // id of the closest grid point
        
        for (int i=0; i<n_dims; i++) {
            minvGrid[i] = *std::min_element(nodeCoord[i].begin(), nodeCoord[i].end());
        }
        double max_space = *std::max_element(spaceGrid.begin(), spaceGrid.end());
        radius = radius * max_space;
        std::vector <size_t> nodeMapGrid(nNodePt, 0);
        closest_Node2UniformGrid(nodeMapGrid, nodeCoord, resampleRate, minvGrid, spaceGrid);

        for (int i=0; i<n_vars; i++) {
            var_ad2 = reader_io.InquireVariable<double>(var_name[i]);
            std::vector<std::size_t> shape = var_ad2.Shape();
            std::vector<double> var_in;
            reader.Get(var_ad2, var_in, adios2::Mode::Sync);
            reader.PerformGets();
            data_size[i] = var_in.size();
            double mag_v = 0, data_cnt=0;
            for (size_t k=0; k<data_size[i]; k++) {
                if (std::isnan(var_in[k])) { 
                    std::cout << "Nan @ var_in[" << k << "] = " << var_in[k] << "\n"; 
                    var_in[k] = 1e-5;
                }
                else {
                    mag_v += var_in[k]*var_in[k];
                    data_cnt += 1;
                }
            }
            mag_v = std::sqrt(mag_v / data_size[i]);
            double abs_tol = tolerance * mag_v;
            std::cout << var_name[i].c_str() << ": magnitude = "<< mag_v << " (" << var_ad2.Min() << " / " << var_ad2.Max() << "), tol = "<< abs_tol << std::endl;
            // 1D compression
            const mgard::TensorMeshHierarchy<1, double> hierarchy({var_in.size()});
            const mgard::CompressedDataset<1, double> compressed = mgard::compress(hierarchy, var_in.data(), s, abs_tol);
            const mgard::DecompressedDataset<1, double> decompressed = mgard::decompress(compressed);
            compressedBytes_1D[i] = compressed.size();
            std::cout << "compressed 1D data ok\n";

            // n-D compression
            std::vector<double> residual(nNodePt, 0.0);
            std::vector<double> GridPointVal(nGridPt, 0.0);
            std::vector<size_t> GridSparseMap(nGridPt, nGridPt);
//            std::vector<double> GridBlendVal(nGridPt, 0.0);
            // 1st pass: for grid points w/ associated nodes --> mean;
            //           for nodes, compute the residual --> mean - nodeVal
            if (radius > 0) {
                Shepard_3DScatterInterp(nodeMapGrid, nodeCoord, resampleRate, minvGrid, spaceGrid, var_in, GridPointVal, residual, radius);
                option_grid = 0;
            } else {
                if (strcmp(interpolation_option.c_str(), "MLE")==0) {
                    std::cout << "interpolate based on MLE\n";
                    uni_GridValResi_MLE(nodeMapGrid, nodeCoord, resampleRate, var_in, GridPointVal, residual);
                    option_grid = 0;
                } else {
                    std::cout << "interpolate based on mean\n"; 
                    option_grid = closest_GridValResi(nodeMapGrid, var_in, GridPointVal, residual);
                }
            }
//            uni_GridValResi_MLE(nodeMapGrid, nodeCoord, resampleRate, var_in, GridPointVal, residual);
            // 2nd pass: for grid points wo/ associated nodes --> nearest in x-axis
 //           total_nbld[i] = GridValBlend(GridPointVal, GridBlendVal, resampleRate, blendR); 
//            memcpy(GridBlendVal.data(), GridPointVal.data(), sizeof(double)*nGridPt); 
 
            double tol_data = abs_tol * ratio_t;
            double tol_resi = abs_tol * (1.0-ratio_t);
            // compress residual
            std::vector<double> residualRCT(nNodePt);
            std::vector<double> GridMeshRCT;
            if (option_grid == 0) {
                GridMeshRCT.resize(nGridPt);
                if (n_dims==2) {
                    std::cout << "Resample rate of 2D grids: " << (double)nGridPt / (double) nNodePt << "\n";
                    const mgard::TensorMeshHierarchy<2, double> rhierarchy({resampleRate[1], resampleRate[0]});
                    const mgard::CompressedDataset<2, double> rcompressed =
                        mgard::compress(rhierarchy, GridPointVal.data(), s, tol_data);
                    std::cout << "s = " << s << ", tol = " << tol_data << "\n";
                    std::cout << "compressed sampled data ok\n" ;
                    const mgard::DecompressedDataset<2, double> rdecompressed = mgard::decompress(rcompressed);
                    memcpy(GridMeshRCT.data(), rdecompressed.data(), sizeof(double)*nGridPt);
                    grid_cSize[i] = rcompressed.size();
                } else if (n_dims==3) {
                    std::cout << "Resample rate of 3D grids: " << (double)nGridPt / (double) nNodePt << " at " << resampleRate[2] << "x" << resampleRate[1] << "x" << resampleRate[0] << "\n";
                    const mgard::TensorMeshHierarchy<3, double> rhierarchy({resampleRate[2], resampleRate[1], resampleRate[0]});
                    std::cout << "s = " << s << ", tol = " << tol_data << "\n";
                    const mgard::CompressedDataset<3, double> rcompressed =
                        mgard::compress(rhierarchy, GridPointVal.data(), s, tol_data);
                    std::cout << "compressed sampled data ok ..." << rcompressed.size() << "\n" ;
                    const mgard::DecompressedDataset<3, double> rdecompressed = mgard::decompress(rcompressed);
                    memcpy(GridMeshRCT.data(), rdecompressed.data(), sizeof(double)*nGridPt);
                    grid_cSize[i] = rcompressed.size();
                }
            } else {
                std::vector<double> GridSparseVal = convertGrid2Vec(GridPointVal, GridSparseMap);
                size_t nSGridPt = GridSparseVal.size();
                GridMeshRCT.resize(nSGridPt);
                std::cout << "Resample rate of 1D sparse grids: " << (double)nSGridPt / (double) nNodePt << "\n";
                const mgard::TensorMeshHierarchy<1, double> rhierarchy({nSGridPt});
                const mgard::CompressedDataset<1, double> rcompressed =
                        mgard::compress(rhierarchy, GridSparseVal.data(), s, tol_data);
                const mgard::DecompressedDataset<1, double> rdecompressed = mgard::decompress(rcompressed);
                memcpy(GridMeshRCT.data(), rdecompressed.data(), sizeof(double)*nSGridPt);
                grid_cSize[i] = rcompressed.size();
                // compress the map -- not necessary, as the grid-map will remain to be the same across timesteps 
                //const size_t cBuffSize = ZSTD_compressBound(nGridPt*sizeof(size_t));
                //unsigned char *const zstd_rmap = new unsigned char[cBuffSize];
                //grid_cSize[i] += ZSTD_compress(zstd_rmap, cBuffSize, (void *)GridSparseMap.data(), nGridPt*sizeof(size_t), 1);
                std::cout << "compressed sampled data ok\n" ;
                std::cout << "sparse grid CR = " << (double) nSGridPt * sizeof(double) / (double) rcompressed.size() << "\n";
            }

            switch (option_resi) {
                case 1:
                {
                    const size_t cBuffSize = ZSTD_compressBound(nNodePt*sizeof(double));
                    unsigned char *const zstd_resi = new unsigned char[cBuffSize];
                    resi_cSize[i] = ZSTD_compress(zstd_resi, cBuffSize, (void *)residual.data(), nNodePt*sizeof(double), 1);
                    ZSTD_decompress(residualRCT.data(), nNodePt * sizeof(double), zstd_resi, resi_cSize[i]);
                    break;
                }
                case 2:
                {
                    const mgard::CompressedDataset<1, double> encoded_resi =
                    quantizeEncode(hierarchy, residual.data(), tol_resi);
                    resi_cSize[i] = encoded_resi.size();
                    dequantizeDecode(encoded_resi, residualRCT.data());
                    break;
                }
                case 3:
                {
                    const mgard::CompressedDataset<1, double> compressed_resi =
                    mgard::compress(hierarchy, residual.data(), s, tol_resi);
                    const mgard::DecompressedDataset<1, double> decompressed_resi = mgard::decompress(compressed_resi);
                    resi_cSize[i] = compressed_resi.size();
                    memcpy(residualRCT.data(), decompressed_resi.data(), sizeof(double)*nNodePt);
                    break;
                }
            }
            std::cout << "Compressed residual OK\n";
            // verify errors
            // comb data
            double interpolateVal, diff;
            std::vector<double> combValue(nNodePt);
            std::vector<int> dims(n_dims, 1);
            for (int d=1; d<n_dims; d++) {
                dims[d] = dims[d-1] * (int) nodeCoord[d-1].size();
            }
            size_t nCellVertices = std::pow(2, n_dims);
            size_t Dh = resampleRate[2] - 1;
            size_t Dc = resampleRate[1] - 1;
            size_t Dr = resampleRate[0] - 1;
            size_t r1, c1, h1, r2, c2, h2;
            for (size_t id=0; id<nNodePt; id++) {
                if (radius==0) {
                    if (option_grid==0) {
                        //std::cout << id << ", " << nNodePt << "\n";
                        interpolateVal = GridMeshRCT.at(nodeMapGrid[id]);
                    } else {
                        if (GridSparseMap[nodeMapGrid[id]] < nGridPt)
                            interpolateVal = GridMeshRCT.at(GridSparseMap[nodeMapGrid[id]]);
                        else
                            interpolateVal = 0;
                    }
                } else {
                    size_t k = nodeMapGrid[id];
                    std::vector <double> fieldVals;
                    h1 = (size_t) ((double)k / (double)dims[2]);
                    c1 = (size_t) ((double)(k-h1*dims[2]) / (double)dims[1]);
                    r1 = (size_t) (k - c1*dims[1] - h1*dims[2]);
                    h2 = (h1+1 > Dh) ? Dh : h1+1;
                    c2 = (c1+1 > Dc) ? Dc : c1+1;
                    r2 = (r1+1 > Dr) ? Dr : r1+1;
                    std::vector<std::vector<size_t>> index;
                    index.push_back({r1, r2, r1, r2, r1, r2, r1, r2});
                    index.push_back({c1, c1, c2, c2, c1, c1, c2, c2});
                    index.push_back({h1, h1, h1, h1, h2, h2, h2, h2});
                    std::vector<std::vector<double>> gCoord;
                    for (int d=0; d<n_dims; d++) {
                        std::vector<double>Coord(index[0].size());
                        for (size_t iv=0; iv<index[0].size(); iv++) {
                            Coord[iv] = nodeCoord[d][index[d][iv]]; 
                        }
                        gCoord.push_back(Coord);
                    }
                    double fVal = GridMeshRCT[k];
                    for (size_t iv=0; iv<nCellVertices; iv++) {
                        size_t kv = index[0][iv] + index[1][iv]*dims[1] +
                                    index[2][iv]*dims[2];
                        if (GridPointVal[kv]!=0) {
                            fVal = GridMeshRCT[kv];
                        }
                        fieldVals.push_back(fVal);
                    }
                    std::vector<double> nCoord {nodeCoord[0][id], nodeCoord[1][id], nodeCoord[2][id]};
                    interpolateVal = interpolateGridtoNode(fieldVals, index, nCoord, gCoord);
                }   
                combValue.at(id) = interpolateVal + residualRCT.at(id);
                diff = std::abs(combValue.at(id) - var_in[id]); 
                //if (diff > abs_tol*10)
                //    std::cout << id << ", " <<  diff << "\n";
                l2_comb[i] += diff*diff;
                linf_comb[i] = (diff > linf_comb[i]) ? diff : linf_comb[i];
            }
            l2_comb[i]   = std::sqrt(l2_comb[i] / nNodePt);
            l2_comb[i]   = l2_comb[i] / mag_v;
            linf_comb[i] = linf_comb[i] / mag_v;
            // 1D data
            for (size_t id=0; id<nNodePt; id++) {
                double diff = std::abs(decompressed.data()[id] - var_in.at(id));
                l2_err[i] += diff*diff;
                linf_err[i] = (diff > linf_err[i]) ? diff : linf_err[i];
            }
            l2_err[i]   = std::sqrt(l2_err[i] / nNodePt);
            l2_err[i]   = l2_err[i] / mag_v;
            linf_err[i] = linf_err[i] / mag_v;
            residual.clear();
            var_in.clear();
            std::cout << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << n_dims << "\n";
            adios2::Variable<double>var_out = writer_io.DefineVariable<double>(var_name[i],
                    shape, {0}, shape);
            var_out.SetSelection(adios2::Box<adios2::Dims>({0}, shape));
            writer.Put<double>(var_out, combValue.data(), adios2::Mode::Sync);
            writer.PerformPuts();
        }
        //std::cout << "end\n";
        reader.EndStep();
    }
    writer.Close();
    reader.Close();

    for (int i=0; i<n_vars; i++) {
        std::cout << "variable: " << var_name[i] << "\n";
        std::cout << "resampled grid compression ratio " << (double)nGridPt * sizeof(double)/ (double)grid_cSize[i] << ", compared to original nodes CR = " << (double)nNodePt * sizeof(double)/ (double)grid_cSize[i]<< "\n"; 
        std::cout << "residual compression ratio " << (double)data_size[i]*sizeof(double) / ((double)resi_cSize[i]) << "\n";
        std::cout << "combined compression ratio for 2D resampled data: " << (double)data_size[i] * sizeof(double) / (double)(resi_cSize[i] + grid_cSize[i]) << "\n";
        std::cout << "interpolated data's l2_err: " << l2_comb[i] << ", linf_err: " << linf_comb[i] << "\n";
        std::cout << "compression ratio for 1D data is: "
            << (double)data_size[i] * sizeof(double)/ (double)compressedBytes_1D[i]
            << std::endl;
        std::cout << "original 1D's l2_err: " << l2_err[i] << ", linf_err: " << linf_err[i] << "\n";
    }

    MPI_Finalize();
    
    return 0;
}
