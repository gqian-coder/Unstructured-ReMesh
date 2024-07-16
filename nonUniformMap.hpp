#ifndef NONUNIFORMMAP
#define NONUNIFORMMAP

#include <cstddef>
#include <cstdint>
#include <vector>

void recompose_remesh(std::vector<size_t> nodeMapGrid,
                      std::vector<double> GridPointVal,
                      std::vector<double> &combinedVal);

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

void check_GridSparsity(std::vector<size_t> nodeMapGrid, size_t nGridPt, size_t &nGridSparse, 
                       std::vector<size_t> &GridSparseMap, std::vector<size_t> &nCluster);


void closest_Node2UniformGrid(std::vector<size_t> &nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<size_t> resampleRate,
                            std::vector<double> minvNode,
                            std::vector<double> spaceGrid);

//#include "nonUniformMap.cpp"
#endif
