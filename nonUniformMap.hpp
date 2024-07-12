#ifndef NONUNIFORMMAP
#define NONUNIFORMMAP

#include <cstddef>
#include <cstdint>
#include <vector>

void calc_GridValResi(std::vector<size_t> nodeMapGrid,
                      std::vector<size_t> nCluster,
                      std::vector<double> &var_in, /* store the residual value back*/
                      std::vector<double> &GridPointVal);

void sel_Gridspace(std::vector<int64_t> connc,
                    std::vector<std::vector<double>> nodeCoord,
                    int n_dims,
                    double perc, /* input <0-1.0> for <0%, 100%>th percentile */
                    std::vector<double> &spaceGrid,
                    std::vector<size_t> &GridDim);

void init_Gridspace(std::vector<std::vector<double>> nodeCoord, 
                    int n_dims, 
                    std::vector<size_t> resampleRate, 
                    std::vector<double> &spaceGrid, 
                    std::vector<double> &minvNode);

void GridCellDensity(std::vector<std::vector<double>> nodeCoord,
                    int n_dims,
                    std::vector<double> minvNode,
                    std::vector<double> spaceGrid,
                    std::vector<size_t> GridDim,
                    std::vector<size_t> &histGridCell);


void check_GridSparsity(std::vector<size_t> nodeMapGrid, size_t nGridPt, size_t &nGridSparse, 
                       std::vector<size_t> &GridSparseMap, std::vector<size_t> &nCluster);

void closest_GridValResi_m(std::vector<size_t> nodeMapGrid,
                        std::vector<double> var_in,
                        std::vector<size_t> nCluster,
                        std::vector<double> &GridPointVal,
                        std::vector<double> &residual);

int closest_GridValResi(std::vector<size_t> nodeMapGrid, 
                        std::vector<double> var_in, 
                        std::vector<double> &GridPointVal, 
                        std::vector<double> &residual);

void merge_Gridspace_2d(std::vector<size_t> &nodeMapGrid,
                     std::vector<size_t> resampleRate,
                     std::vector<size_t> &resampleRate_m);

void merge_Gridspace_3d(std::vector<size_t> &nodeMapGrid,
                     std::vector<size_t> resampleRate,
                     std::vector<size_t> &resampleRate_m);

void closest_Node2nonUniformGrid(std::vector<size_t> &nodeMapGrid,
                       std::vector<std::vector<double>> nodeCoord,
                       std::vector<std::vector<size_t>> MergedGridCoord,
                       std::vector<double> minvNode,
                       std::vector<double> spaceGrid);

void closest_Node2UniformGrid(std::vector<size_t> &nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<size_t> resampleRate,
                            std::vector<double> minvNode,
                            std::vector<double> spaceGrid);

std::vector<double> convertGrid2Vec(std::vector<double> GridPointVal, 
                                    std::vector<size_t> &GridSparseMap);


std::vector<double> closest_GridValResi_sparse(std::vector<size_t> nodeMapGrid,
                        std::vector<size_t> GridSparseMap,
                        std::vector<double> var_in,
                        std::vector<size_t> nCluster,
                        std::vector<double> &residual,
                        size_t nSGridPt);

void cellInterp_GridValResi(std::vector<size_t> nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<std::vector<size_t>> MergedGridCoord,
                            std::vector<double> spaceGrid,
                            std::vector<double> minvNode,
                            std::vector<double> var_in,
                            std::vector<double> &GridPointVal,
                            std::vector<double> &residual);

double interpolateGridtoNode(std::vector <double> fieldVals,
                             std::vector<std::vector<size_t>> index,
                             std::vector<double> nodeCoord,
                             std::vector<std::vector<double>> GridCoord);
                             
std::vector<std::vector<size_t>>
GridCellMerge3D_Mdir(std::vector<std::vector<size_t>> GridCoord,
                   std::vector<size_t> histGridCell,
                   std::vector<size_t> GridDim,
                   std::vector<double> thresh);

std::vector<std::vector<size_t>>
GridCellAMR3D(std::vector<std::vector<size_t>> GridCoord,
                   std::vector<size_t> histGridCell,
                   std::vector<size_t> GridDim,
                   std::vector<double> Mthresh  /*merge threshold*/,
                   std::vector<double> Sthresh /*split threshold*/);

std::vector<std::vector<size_t>>
GridCellAMR3D_LgC(std::vector<std::vector<size_t>> GridCoord,
                   std::vector<size_t> histGridCell,
                   std::vector<size_t> GridDim,
                   std::vector<double> Mthresh  /*merge threshold*/,
                   std::vector<double> Sthresh /*split threshold*/,
                   std::vector<double> Pthresh /*the P% largest cells for thresholding*/);

void closest_GridValResi_MLE(std::vector<size_t> nodeMapGrid,
    int n_dims,
    std::vector<std::vector<double>> nodeCoord,
    std::vector<size_t>resampleRate,
    std::vector<double> var_in,
    std::vector<double> &GridPointVal,
    std::vector<double> &residual);

void nonUni_GridVal_MLE(std::vector<size_t> nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<std::vector<size_t>> MergedGridCoord,
                            std::vector<double> spaceGrid,
                            std::vector<double> minvNode,
                            std::vector<double> var_in,
                            std::vector<double> &GridPointVal);

void nonUni_GridValResi_MLE(std::vector<size_t> nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<std::vector<size_t>> MergedGridCoord,
                            std::vector<double> spaceGrid,
                            std::vector<double> minvNode,
                            std::vector<double> var_in,
                            std::vector<double> &GridPointVal,
                            std::vector<double> &residual);

void nonUni_closest_GridVal_MLE(std::vector<size_t> nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<std::vector<size_t>> MergedGridCoord,
                            std::vector<double> spaceGrid,
                            std::vector<double> minvNode,
                            std::vector<double> var_in,
                            std::vector<double> &GridPointVal);

void nonUni_GridValResi_Ghost(std::vector<size_t> nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<std::vector<size_t>> MergedGridCoord,
                            std::vector<double> spaceGrid,
                            std::vector<double> minvNode,
                            std::vector<double> var_in,
                            std::vector<double> &GridPointVal,
                            std::vector<double> &residual);

void uni_GridValResi_MLE(std::vector<size_t> nodeMapGrid,
                        std::vector<std::vector<double>> nodeCoord,
                        std::vector<size_t>resampleRate,
                        std::vector<double> var_in,
                        std::vector<double> &GridPointVal,
                        std::vector<double> &residual);

void Shepard_3DScatterInterp(std::vector<size_t> nodeMapGrid,
                        std::vector<std::vector<double>> nodeCoord,
                        std::vector<size_t>resampleRate,
                        std::vector<double> minvGrid,
                        std::vector<double> spaceGrid,
                        std::vector<double> var_in,
                        std::vector<double> &GridPointVal,
                        std::vector<double> &residual,
                        double radius);

//#include "nonUniformMap.cpp"
#endif
