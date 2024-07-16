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
#include <algorithm>    // std::min_element, std::max_element
#include <numeric>      // std::accumulate

#include "nonUniformMap.hpp"
#define DELTA 1e-6
#define resampleThresh 1e-2
#define threshGrid 0.85

void check_GridSparsity(std::vector<size_t> nodeMapGrid, size_t nGridPt, size_t &nGridSparse, 
                       std::vector<size_t> &GridSparseMap, std::vector<size_t> &nCluster)
{
    size_t nNodePt = nodeMapGrid.size();
    nGridSparse = 0;
    std::vector<bool> GridPointVal(nGridPt, false);

    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        nCluster[k] ++;
        GridPointVal[k] = true;
    }
    for (size_t i=0; i<nGridPt; i++) {
        if (GridPointVal[i]!=0) nGridSparse ++;
    }

    double nonZerosRate =  (double)nGridSparse / (double)nGridPt;
    //std::cout << "number of non-zero grid points: " << nGridSparse << " (" << nonZerosRate * 100.0 << "% of Grid pts) \n";
    if (nonZerosRate < threshGrid) {
	    size_t sparseId = 0;
        for (size_t i=0; i<nGridPt; i ++) {
            if (GridPointVal[i]) {
                GridSparseMap[i] = sparseId;
                sparseId ++;
            }
	    }
    } else {
        nGridSparse = nGridPt; 
    }
}

// cell: true|false --> left top corner of the cell | closest grid point
void closest_Node2UniformGrid(std::vector<size_t> &nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<size_t> resampleRate,
                            std::vector<double> minvGrid,
                            std::vector<double> spaceGrid)
{
    size_t nNodePt = nodeCoord[0].size();
    int n_dims = resampleRate.size();
    double posVal;
    std::vector<double> index(n_dims);
    std::vector<size_t> dims(n_dims, 1);
    for (int d=1; d<n_dims; d++) {
        dims[d] = dims[d-1] * resampleRate[d-1];
    }
    for (size_t i=0; i<nNodePt; i++) {
        for (int d=0; d<n_dims; d++) {
            posVal = nodeCoord[d][i] - minvGrid[d];
            index[d] = std::floor(posVal / spaceGrid[d]);
            if (index[d]>=resampleRate[d]) index[d] = resampleRate[d]-1;
            nodeMapGrid[i] += index[d] * dims[d];
        }
    }
}


void calc_GridValResi(std::vector<size_t> nodeMapGrid,
                      std::vector<size_t> nCluster, 
                      std::vector<double> &var_in, /* store the residual value back*/
                      std::vector<double> &GridPointVal)
{
    size_t nNodePt = var_in.size();
    size_t nGridPt = GridPointVal.size();
    for (size_t i=0; i<nNodePt; i++) {
        GridPointVal[nodeMapGrid[i]] += var_in[i];
    }
    for (size_t i=0; i<nGridPt; i++) {
        GridPointVal[i] = GridPointVal[i] / (double)nCluster[i];
    }
    for (size_t i=0; i<nNodePt; i++) {
        var_in[i] -= GridPointVal[nodeMapGrid[i]];
    }
}


void recompose_remesh(std::vector<size_t> nodeMapGrid,
                      std::vector<double> GridPointVal,
                      std::vector<double> &combinedVal)
{
    size_t nNodePt = combinedVal.size();
    for (size_t i=0; i<nNodePt; i++) {
        combinedVal[i] += GridPointVal[nodeMapGrid[i]];
    }
}


// Select "optimal" grid setup based on node spacing
void sel_Gridspace(std::vector<int64_t> connc, 
                    std::vector<std::vector<double>> nodeCoord, 
                    int n_dims, 
                    double perc, /* input <0-1.0> for <0%, 100%>th percentile */ 
                    std::vector<double> &spaceGrid, 
                    std::vector<size_t> &GridDim)
{
    if ((perc>1.0) || (perc<0.0)) {
        std::cout << "Error: percentile must be in between 0 and 1\n";
        return;
    }
    size_t nConnc = connc.size();
    std::cout << "number of connectivities: " << nConnc << "\n";
    double minv, maxv;
    for (int d=0; d<n_dims; d++) {
        minv = *std::min_element(nodeCoord[d].begin(), nodeCoord[d].end());
        maxv = *std::max_element(nodeCoord[d].begin(), nodeCoord[d].end());
        size_t cnt_nzr = 0;
        std::vector<double> spaceVertices;
        if (nConnc>0) {
            spaceVertices.resize(nConnc-1, 1e9); 
            for (size_t i=0; i<nConnc-1; i++) {
                int64_t prev = connc[i];
                int64_t curr = connc[i+1];
                double dist = std::abs(nodeCoord[d][curr] - nodeCoord[d][prev]);
                if (dist>0) {
                    spaceVertices[i] = dist; 
                    cnt_nzr ++;
                }
            }
        }
        else {
            spaceVertices.resize(nodeCoord[d].size()-1, 1e9);  
            for (size_t i=0; i<nodeCoord[d].size()-1; i++) {
                double dist = std::abs(nodeCoord[d][i+1] - nodeCoord[d][i]);
                if (dist>0) {
                    spaceVertices[i] = dist;
                    cnt_nzr ++;
                }
            }
        }
        std::make_heap(spaceVertices.begin(), spaceVertices.end(), std::greater<>{});
        //std::cout << "dim " << d << ": minimum spacing = " << spaceVertices.front() << ", range = " << minv << " -- " << maxv <<  "\n";
        size_t pth = (size_t)(perc * (double)cnt_nzr); 
        size_t accu_pth = pth;
        GridDim[d] = nodeCoord[d].size(); 
        while ((GridDim[d] > (int)(((double)nodeCoord[d].size()) * resampleThresh)) && (accu_pth<cnt_nzr)) {
            for (size_t p=0; p<pth; p++) {
                pop_heap(spaceVertices.begin(), spaceVertices.end(), std::greater<>{});
                spaceVertices.pop_back();
            }
            spaceGrid[d] = spaceVertices.front();
            GridDim[d]   = (int)((maxv - minv) / spaceGrid[d]);
            accu_pth += pth;
        }
        std::cout << "min, max at dim " << d << " = " << minv << ", " << maxv << ", at a spacing of " << spaceGrid[d] << ", w/ " << cnt_nzr << " non-zero points\n";
    }
}


