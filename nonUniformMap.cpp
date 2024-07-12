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


// merge the 2D grids to avoid discontinities/zeros
void merge_Gridspace_2d(std::vector<size_t> &nodeMapGrid,
                     std::vector<size_t> resampleRate,
                     std::vector<size_t> &resampleRate_m)
{
    // check which grid points have mapping value
    size_t gridSz = resampleRate[0]*resampleRate[1];
    size_t nnodes = nodeMapGrid.size();
    std::vector<size_t> index_map(gridSz, 0);
    for (size_t i=0; i<nnodes; i++) {
        index_map[nodeMapGrid[i]] = 1;
    }

    std::vector<std::vector<int8_t>> mark_dim(2);
    size_t rzR = 0, rzC = 0;
    // find the merged grid dimension
    mark_dim[0].resize(resampleRate[0], 0);
    for (size_t r=0; r<resampleRate[0]; r++) {
        bool sdata  = false;
        for (size_t c=0; c<resampleRate[1]; c++) {
            sdata = sdata | index_map[c*resampleRate[0]+r];
        }
        mark_dim[0][r] = (sdata ? 1 : 0);
    }
    rzR = std::accumulate(mark_dim[0].begin(), mark_dim[0].end(), 0);
    std::cout << "rzR = " << rzR << "\n";

    mark_dim[1].resize(resampleRate[1], 0);
    for (size_t c=0; c<resampleRate[1]; c++) {
        bool sdata = false;
        size_t k   = c*resampleRate[0];
        for (size_t r=0; r<resampleRate[0]; r++) {
            sdata = sdata | index_map[k+r];
        }
        mark_dim[1][c] = (sdata ? 1 : 0);
    }
    rzC = std::accumulate(mark_dim[1].begin(), mark_dim[1].end(), 0);
    std::cout << "dimension of merged grid: " << rzR << "x" << rzC << "\n"; 

    if ((rzC==resampleRate[1]) || (rzR==resampleRate[0])) {
        resampleRate_m[1] = resampleRate[1];
        resampleRate_m[0] = resampleRate[0];
        return;
    }
    std::vector<size_t> delta_r(resampleRate[0]);
    delta_r[0] = mark_dim[0][0];
    for (size_t r=1; r<resampleRate[0]; r++) {
        delta_r[r] = delta_r[r-1] + mark_dim[0][r];
    }
    size_t c_offs_m = 0;
    for (size_t c=0; c<resampleRate[1]; c++) {
        if (mark_dim[1][c]) {
            size_t c_offs   = c*resampleRate[0];
            for (size_t r=0; r<resampleRate[0]; r++) {
                size_t k = c_offs+r;
                if (mark_dim[0][r]) {
                    index_map[k] = c_offs_m + delta_r[r]-1;
                }
            }
            c_offs_m += rzR;
        }
    }
    
    // store the index back to nodeMapGrid
    for (size_t k=0; k<nnodes; k++) {
        nodeMapGrid[k] = index_map[nodeMapGrid[k]];
    }
    resampleRate_m[1] = rzC;
    resampleRate_m[0] = rzR;
    std::cout << "dimension of merged grid: " << rzC << "x" << rzR << "\n";

}

// convert 3D data to sitched 2D data
// stitch the 2D slices along the slowest changing axis 
// resampleRate[size-1] stores the slowset changing axis
void merge_Gridspace_3d(std::vector<size_t> &nodeMapGrid, 
                     std::vector<size_t> resampleRate,
                     std::vector<size_t> &resampleRate_m)
{
    // check which grid points have mapping value
    size_t gridSz = resampleRate[0]*resampleRate[1]*resampleRate[2];
    size_t nnodes = nodeMapGrid.size();
    std::vector<size_t> index_map(gridSz, 0); 
    for (size_t i=0; i<nnodes; i++) {
        index_map[nodeMapGrid[i]] = 1;
    }

    size_t n_dims = resampleRate.size();
    std::vector<std::vector<int8_t>> mark_dim(n_dims-1);
    size_t dim2 = resampleRate[0]*resampleRate[1];
    size_t rzR = 0, rzC = 0;
    // find the merged grid dimension
    mark_dim[0].resize(resampleRate[0]*resampleRate[2]);
    for (size_t h=0; h<resampleRate[2]; h++) {
        size_t offset = h * resampleRate[0];
        size_t h_offs = h*dim2;
        for (size_t r=0; r<resampleRate[0]; r++) {
            bool sdata  = false;
            for (size_t c=0; c<resampleRate[1]; c++) {
                sdata = sdata | index_map[h_offs+c*resampleRate[0]+r];
            }
            mark_dim[0][offset+r] = (sdata ? 1 : 0);
        }
        size_t temp = std::accumulate(mark_dim[0].begin()+offset, mark_dim[0].begin()+offset+resampleRate[0], 0);
        rzR = (rzR < temp) ? temp : rzR; 
    } 
    mark_dim[1].resize(resampleRate[1]*resampleRate[2]);
    for (size_t h=0; h<resampleRate[2]; h++) {
        size_t offset = h * resampleRate[1];
        size_t h_offs = h*dim2;
        for (size_t c=0; c<resampleRate[1]; c++) {
            bool sdata  = false;
            size_t k    = h_offs + c*resampleRate[0];
            for (size_t r=0; r<resampleRate[0]; r++) {
                sdata = sdata | index_map[k+r];
            }
            mark_dim[1][offset+c] = (sdata ? 1 : 0);
        }
        size_t temp = std::accumulate(mark_dim[1].begin()+offset, mark_dim[1].begin()+offset+resampleRate[1], 0);
        rzC = (rzC < temp) ? temp : rzC;
        //std::cout << "h = " << h << ", rzC = " << rzC << ", tempC = " << temp << "\n";
    }
    std::cout << "rzR = " << rzR << ", rzC = " << rzC << "\n";
    if (rzC==resampleRate[1]) {
        resampleRate_m[1] = resampleRate[1]*resampleRate[2];
        resampleRate_m[0] = resampleRate[0];
        return;
    }

    size_t rzC_m    = 0;
    size_t c_offs_m = 0;
    for (size_t h=0; h<resampleRate[2]; h++) {
        size_t h_offs    = h * dim2;
        size_t m_offs_c  = h * resampleRate[1];
        size_t m_offs_r  = h * resampleRate[0];
        size_t delta_c   = 0;

        std::vector<size_t> delta_r(resampleRate[0]);
        delta_r[0] = mark_dim[0][m_offs_r];
        for (size_t r=1; r<resampleRate[0]; r++) {
            delta_r[r] = delta_r[r-1] + mark_dim[0][m_offs_r+r];
        }

        for (size_t c=0; c<resampleRate[1]; c++) {
            if (mark_dim[1][m_offs_c+c]) {
                size_t c_offs   = c*resampleRate[0] + h_offs;
                for (size_t r=0; r<resampleRate[0]; r++) {
                    size_t k = c_offs+r;
                    if (mark_dim[0][m_offs_r+r]) {
                        index_map[k] = c_offs_m + delta_r[r]-1;  
                    }    
                }
                delta_c ++;
                c_offs_m += rzR;
            }
        }
        rzC_m += delta_c;
    }
    // store the index back to nodeMapGrid
    for (size_t k=0; k<nnodes; k++) {
        nodeMapGrid[k] = index_map[nodeMapGrid[k]]; 
    }
    resampleRate_m[1] = rzC_m;
    resampleRate_m[0] = rzR;
    std::cout << "dimension of merged grid: " << rzC_m << "x" << rzR << "\n";
}


int closest_GridValResi(std::vector<size_t> nodeMapGrid, 
                        std::vector<double> var_in, 
                        std::vector<double> &GridPointVal, 
                        std::vector<double> &residual)
{
    size_t nNodePt = residual.size();
    size_t nGridPt = GridPointVal.size();

    std::vector<size_t> nCluster(nGridPt, 0);
    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        nCluster[k] ++;
        GridPointVal[k] += var_in[i];
    }
    size_t nGridSparse = 0;
    for (size_t j=0; j<nGridPt; j++) {
        if (nCluster[j]>0) {
            GridPointVal[j] = GridPointVal[j] / (double)nCluster[j];
            if (std::abs(GridPointVal[j]) < DELTA) {
                GridPointVal[j] = 0;
            }
        }
        if (GridPointVal[j]!=0) {
            nGridSparse ++;
        }
    }
    double nonZerosRate =  (double)nGridSparse / (double)nGridPt;
    std::cout << "number of non-zero grid points: " << nGridSparse << " (" << nonZerosRate * 100.0 << "% of Grid pts) \n";
    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        residual[i] = var_in[i] - GridPointVal[k];
    }
    return ( (nonZerosRate < threshGrid) ? 1 : 0);
}


void closest_GridValResi_m(std::vector<size_t> nodeMapGrid,
                        std::vector<double> var_in,
                        std::vector<size_t> nCluster,
                        std::vector<double> &GridPointVal,
                        std::vector<double> &residual)
{
    size_t nNodePt = residual.size();
    size_t nGridPt = GridPointVal.size();

    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        GridPointVal[k] += var_in[i];
    }
    for (size_t j=0; j<nGridPt; j++) {
        if (nCluster[j]>0) {
            GridPointVal[j] = GridPointVal[j] / (double)nCluster[j];
            //if (std::abs(GridPointVal[j]) < DELTA) {
            //    GridPointVal[j] = 0;
            //}
        }
    }
    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        residual[i] = var_in[i] - GridPointVal[k];
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


void Shepard_3DScatterInterp(std::vector<size_t> nodeMapGrid, 
                        std::vector<std::vector<double>> nodeCoord, 
                        std::vector<size_t>resampleRate,
                        std::vector<double> minvGrid,
                        std::vector<double> spaceGrid,
                            std::vector<double> var_in, 
                        std::vector<double> &GridPointVal,
                        std::vector<double> &residual,
                        double radius)
{
    size_t nNodePt = nodeMapGrid.size();
    size_t nGridPt = GridPointVal.size();

    int n_dims = resampleRate.size();
    std::vector<double> maxvGrid(n_dims);
    std::vector<std::vector<double>> GridCoord;
    std::vector<size_t> dims(3, 1); 
    for (int i=0; i<n_dims; i++) {
        std::vector<double> Coord(resampleRate[i], minvGrid[i]);
        for (size_t k=0; k<resampleRate[i]; k++) {
            Coord[k] += spaceGrid[i] * (double)k;
        }
        GridCoord.push_back(Coord);
        if (i>0) {
            dims[i] = dims[i-1] * resampleRate[i-1];
        }
        maxvGrid[i] = minvGrid[i] + spaceGrid[i]*resampleRate[i];
    }

    std::vector<double> denominator(nGridPt, 0.0);
    double dist, dx, dy, dz, w, temp;
    size_t dim3, dim2, k;
    std::vector<int> rangeLeft(3), rangeRight(3);
    for (size_t i=0; i<nNodePt; i++) {
        for (int d=0; d<n_dims; d++) {
            temp  = nodeCoord[d][i] - radius;
            rangeLeft[d]  = (temp > minvGrid[d]) ? (int)std::round((temp - minvGrid[d]) / spaceGrid[d]) : 0;
            temp = nodeCoord[d][i] + radius;
            rangeRight[d] = (temp <= maxvGrid[d]) ? (int)std::round((temp - minvGrid[d]) / spaceGrid[d]) : resampleRate[d];
//            std::cout << rangeLeft[d] << " --> " << rangeRight[d] << ",";
        }
        //std::cout << "\n";
        for (int z=rangeLeft[2]; z<rangeRight[2]; z++) {
            dz = nodeCoord[2][i] - GridCoord[2][z];
            dz = dz * dz;
            dim3 = (size_t)z * dims[2];
            for (int y=rangeLeft[1]; y<rangeRight[1]; y++) {
                dy = nodeCoord[1][i] - GridCoord[1][y];
                dy = dy * dy;
                dim2 = (size_t)y * dims[1];
                for (int x=rangeLeft[0]; x<rangeRight[0]; x++) {
                    dx = nodeCoord[0][i] - GridCoord[0][x];
                    dx = dx * dx;
                    dist = (size_t)std::sqrt(dx + dy + dz);
                    if (dist < radius) {
                        w = (radius - dist) / (radius*dist);
                        w = w*w; 
                        k = x + dim2 + dim3;
                        GridPointVal[k] += var_in[i] * w;
                        denominator[k] += w;
                    } 
                }
            }
        }
        if (i % 5000 == 0) std::cout << i << " / " << nNodePt << "\n";
    }
    std::cout << "finish interpolation (scatter --> grid)\n";
    for (size_t j=0; j<nGridPt; j++) {
        if (denominator[j]>0) {
            GridPointVal[j] = GridPointVal[j] / (double)denominator[j];
            if (std::abs(GridPointVal[j]) < DELTA) GridPointVal[j] = 0;
        }
    }

    // bi-linear interpolation
    size_t h1, c1, r1, h2, c2, r2;
    size_t Dr = resampleRate[0] - 1;
    size_t Dc = resampleRate[1] - 1;
    size_t Dh = resampleRate[2] - 1;
    for (size_t i=0; i<nNodePt; i++) {
        k = nodeMapGrid[i];
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
                Coord[iv] = GridCoord[d][index[d][iv]];
            }
            gCoord.push_back(Coord);
        }        
 
        double fVal = GridPointVal[k];
        for (size_t iv=0; iv<index[0].size(); iv++) {
            size_t kv = index[0][iv] + index[1][iv]*dims[1] +
                        index[2][iv]*dims[2];
            if (GridPointVal[kv]!=0) {
                fVal = GridPointVal[kv];
            }
            fieldVals.push_back(fVal);
//            std::cout << fieldVals[iv] << ", ";
        }
//        std::cout << var_in[i] << "\n";
        std::vector<double> nCoord {nodeCoord[0][i], nodeCoord[1][i], nodeCoord[2][i]};
        double pp = interpolateGridtoNode(fieldVals, index, nCoord, gCoord);
        residual[i] = var_in[i] - pp;
    }
}


void init_Gridspace(std::vector<std::vector<double>> nodeCoord,
                    int n_dims,
                    std::vector<size_t> resampleRate,
                    std::vector<double> &spaceGrid,
                    std::vector<double> &minvGrid)
{
    double minv, maxv;
    for (int d=0; d<n_dims; d++) {
        minv = *std::min_element(nodeCoord[d].begin(), nodeCoord[d].end());
        maxv = *std::max_element(nodeCoord[d].begin(), nodeCoord[d].end());
        spaceGrid[d]= (maxv - minv) / (double)(resampleRate[d]-1);
        minvGrid[d] = minv;
        std::cout << "dim " << d << ": spacing = " << spaceGrid.at(d) << ", range = " << minv << " -- " << maxv <<  "\n";
    }
}


// histGridCell: GridDim[0]xGridDim[1]xGridDim[2]
void GridCellDensity(std::vector<std::vector<double>> nodeCoord,
                    int n_dims,
                    std::vector<double> minvGrid,
                    std::vector<double> spaceGrid, 
                    std::vector<size_t> GridDim,
                    std::vector<size_t> &histGridCell)
{
    std::vector<size_t> dims(n_dims, 1);
    for (int i=1; i<n_dims; i++) {
        dims[i] = dims[i-1] * (GridDim[i-1]-1);
//        std::cout << "dim: " << dims[i] << ", minvGrid: " << minvGrid[i] <<"\n";
    }
    double posVal;
    std::vector<size_t> index(n_dims);
    size_t nNodePt = nodeCoord[0].size();
    size_t nGridCell = histGridCell.size()-1;
    for (size_t i=0; i<nNodePt; i++) {
        size_t idx = 0;
        for (int d=0; d<n_dims; d++) {
            posVal = nodeCoord[d][i]-minvGrid[d];
            index[d] = (size_t)std::floor(posVal / spaceGrid[d]);
            if (index[d]>=GridDim[d]) index[d]=GridDim[d]-1;
            idx += index[d] * dims[d];
        }
        if (idx > nGridCell) idx = nGridCell; 
        histGridCell[idx] ++;
    }
}

void uni_GridValResi_MLE(std::vector<size_t> nodeMapGrid,
        std::vector<std::vector<double>> nodeCoord,
        std::vector<size_t>resampleRate,
        std::vector<double> var_in,
        std::vector<double> &GridPointVal,
        std::vector<double> &residual)
{
    int n_dims = resampleRate.size();
    std::vector<double> space(n_dims);
    std::vector<double> minv(n_dims);
    std::vector<double> maxv(n_dims);
    std::vector<size_t> dims(n_dims, 1);
    std::vector<std::vector<double>> resampleCoord;

    double radius = 0;
    for (int d=0; d<n_dims; d++) {
        if (d>0) {
            dims[d] = dims[d-1] * resampleRate[d-1];
        }
        minv[d] = *std::min_element(nodeCoord[d].begin(), nodeCoord[d].end());
        maxv[d] = *std::max_element(nodeCoord[d].begin(), nodeCoord[d].end());
        space[d]= (maxv[d] - minv[d]) / (double)resampleRate[d];
        radius += space[d]*space[d];
//        std::cout << "minv[" << d << "] = " << minv[d] << ", maxv[" << d << "] = " << maxv[d] <<", space["<< d << "] = " << space[d] << ", dims[" << d << "] = " << dims[d] << "\n";
        std::vector<double> rcoord(resampleRate[d], minv[d]);
        for (size_t i=0; i<resampleRate[d]; i++) {
            rcoord.at(i) += (double)i * space[d];
        }
        resampleCoord.push_back(rcoord);
    }
    radius = std::sqrt(radius);
    size_t nNodePt = residual.size();
    std::cout << "radius = " << radius << "\n";
    size_t nGridPt = GridPointVal.size();
    std::vector<double> dist_sum(nGridPt, 0.0);
    std::vector<double> w_val_sum(nGridPt, 0.0);
    std::vector<size_t> pos(n_dims);
    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        pos[2] = (size_t) ((double)k / (double)dims[2]);
        pos[1] = (size_t) ((double)(k - pos[2]*dims[2]) / (double)dims[1]);
        pos[0] = (size_t) (k - pos[1]*dims[1] - pos[2]*dims[2]);
        double w=0.0;
        for (int d=0; d<n_dims; d++) {
            double dist = resampleCoord[d][pos[d]] - nodeCoord[d][i];
            w += dist * dist;
        }
        w = std::sqrt(w);
        if (w < DELTA) w = DELTA;
        w = (w >= radius) ? 0 : (radius-w)/(radius*w);
        w = w*w;
        dist_sum.at(k) += w;
        w_val_sum.at(k) += w * var_in[i];
    }
    for (size_t i=0; i<nGridPt; i++) {
        if (dist_sum.at(i) > 0) {
            GridPointVal.at(i) = w_val_sum.at(i) / dist_sum.at(i);
//            std::cout << i << ": " << GridPointVal.at(i) << "\n";
        }
    }
    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        residual[i] = var_in[i] - GridPointVal[k];
//        std::cout << i << ": " << var_in[i] << " --> " << GridPointVal[k] << " --> " << residual[i] << "\n";
    }
}

std::vector<std::vector<size_t>>
GridCellAMR3D_LgC(std::vector<std::vector<size_t>> GridCoord,
                   std::vector<size_t> histGridCell,
                   std::vector<size_t> GridDim,
                   std::vector<double> Mthresh  /*merge threshold*/,
                   std::vector<double> Sthresh /*split threshold*/,
                   std::vector<double> Pthresh /*the K largest cells for thresholding*/)
{
    std::vector<std::vector<size_t>> MergedGridCoord;
    int n_dims = 3;
    size_t n_cells, k, mk, zdim, ydim; // continuous in x
    std::vector<size_t> pos(n_dims);
    std::vector<size_t> CellDim {GridDim[0]-1, GridDim[1]-1, GridDim[2]-1};
    size_t dim1 = CellDim[0];
    size_t dim2 = CellDim[0] * CellDim[1];
    std::vector<std::vector<size_t>> splitFlg;
    std::vector<size_t> nDel(n_dims, 0);
    std::vector<size_t> nAdd(n_dims, 0);
    std::vector<size_t> Nthresh {(size_t)(((double)CellDim[2]*CellDim[1])*Pthresh[0]), 
                                (size_t)(((double)CellDim[2]*CellDim[0])*Pthresh[1]), 
                                (size_t)(((double)CellDim[1]*CellDim[0])*Pthresh[2])};

    for (int d=0; d<n_dims; d++) {
        std::vector<size_t> Flg(GridDim[d], 0);
        splitFlg.push_back(Flg);
        std::cout << "Nthresh[" << d << "] = " << Nthresh[d] << "\n";
    }
    // pass 1: Y-Z plane
    std::vector<size_t> Mdensity(CellDim[2]*CellDim[1], 0);
    for (size_t r=0; r<CellDim[0]; r++) {
        for (size_t h=0; h<CellDim[2]; h++) {
            zdim = h * dim2;
            for (size_t c=0; c<CellDim[1]; c++) {
                mk = h*CellDim[1] + c;
                k  = zdim + c*dim1 + r;
                if (histGridCell[k]) {
                    Mdensity[mk] += histGridCell[k];
                }
            }
        }
        std::vector<size_t>density(Mdensity);
        std::make_heap(density.begin(), density.end(), std::less<>{});
//        std::cout << "density.pop " << density.front() << "\n";
        double checksum = 0.0;
        n_cells = 0;
        for (k=0; k<Nthresh[0]; k++) {
            pop_heap(density.begin(), density.end(), std::less<>{});
//            std::cout << "density.pop " << density.back() << "\n";
            if (density.back()==0) {
                n_cells = k;
                break;
            }   
            checksum += density.back();
            density.pop_back();
        } 
//        std::cout << r << ": mk = " << mk <<", checksum = " << checksum  << ", add = " << nAdd[0] << ", del = " << nDel[0]<< "\n";
        n_cells = (n_cells==0) ? Nthresh[0] : n_cells;
        checksum = checksum / (double) n_cells; 
//        std::cout << checksum << ", " << n_cells << "\n";
        if (checksum<=Mthresh[0]) {  // merge
            GridCoord[0][r] = 0;
            if (r>0) nDel[0] ++;
        } else if (checksum>=Sthresh[0]) {
            splitFlg[0][r] = (size_t) std::ceil(checksum/Sthresh[0]) -1;
            nAdd[0] += splitFlg[0][r];
        } else {
            std::fill(Mdensity.begin(), Mdensity.end(), 0.0);    
        }
    }

    std::cout << "dim 0: original " << GridDim[0] << ", del " << nDel[0] << ", add " << nAdd[0] << " coordinates\n";
    std::vector<size_t> MCoord(GridDim[0]-nDel[0]+nAdd[0], 0);
    k = 1;
    for (size_t i=1; i<GridDim[0]; i++) {
        if (GridCoord[0][i]) {
            MCoord[k] = GridCoord[0][i];
//              std::cout << MCoord[k] << ", ";
            k ++;
        }
        if (splitFlg[0][i]) {
            double delta_s = 1.0 / splitFlg[0][i];
            for (size_t s=0; s<splitFlg[0][i]; s++) {
                MCoord[k] = MCoord[k-1] + delta_s;
                k ++;
            }
        }
    }
//        std::cout << "}\n";
    MergedGridCoord.push_back(MCoord);

    // pass 2: X-Z plane
    Mdensity.resize(CellDim[2]*CellDim[0]);
    std::fill(Mdensity.begin(), Mdensity.end(), 0.0);
    for (size_t c=0; c<CellDim[1]; c++) {
        ydim = c*dim1;
        for (size_t h=0; h<CellDim[2]; h++) {
            zdim = h * dim2;
            size_t *px = &histGridCell.data()[ydim + zdim];
            mk = h*CellDim[0];
            for (size_t r=0; r<CellDim[0]; r++) {
                if (*px) {
                    Mdensity[mk] += *px;
                }
                px ++;
                mk ++;
            }
        }
        std::vector<size_t>density(Mdensity); 
        std::make_heap(density.begin(), density.end(), std::less<>{});
        double checksum = 0.0;
        n_cells = 0;
        for (k=0; k<Nthresh[1]; k++) {
            pop_heap(density.begin(), density.end(), std::less<>{});
            if (density.back()==0) {
                n_cells = k;
                break;
            }
            checksum += density.back();
            density.pop_back();
        }
        n_cells = (n_cells==0) ? Nthresh[1] : n_cells;
        checksum = checksum / (double) n_cells;
//        std::cout << checksum << "\n";
        if (checksum<=Mthresh[1]) { // merge
            GridCoord[1][c] = 0;
            if (c>0) nDel[1] ++;
        } else if (checksum>=Sthresh[1]) {
            splitFlg[1][c] = (size_t) std::ceil(checksum/Sthresh[1]) -1;
            nAdd[1] += splitFlg[0][c];
        } else {
            std::fill(Mdensity.begin(), Mdensity.end(), 0.0);
        }
    }

    std::cout << "dim 1: original " << GridDim[1] << ", del " << nDel[1] << ", add "<< nAdd[1] << " coordinates\n";
    MCoord.resize(GridDim[1]-nDel[1]+nAdd[1], 0);
    k = 1;
    for (size_t i=1; i<GridDim[1]; i++) {
        if (GridCoord[1][i]) {
            MCoord[k] = GridCoord[1][i];
//              std::cout << MCoord[k] << ", ";
            k ++;
        }
        if (splitFlg[1][i]) {
            double delta_s = 1.0 / splitFlg[1][i];
            for (size_t s=0; s<splitFlg[1][i]; s++) {
                MCoord[k] = MCoord[k-1] + delta_s;
                k ++;
            }
        }
    }
//        std::cout << "}\n";
    MergedGridCoord.push_back(MCoord);
//    std::cout << "finsih check Y-axis for coordinates merging...\n";

    // pass 3: X-Y plane
    Mdensity.resize(CellDim[0]*CellDim[1]);
    std::fill(Mdensity.begin(), Mdensity.end(), 0.0);
    for (size_t h=0; h<CellDim[2]; h++) {
        zdim = h * dim2;
        for (size_t c=0; c<CellDim[1]; c++) {
            ydim = c * dim1;
            mk = c * CellDim[0];
            size_t *px = &histGridCell.data()[ydim + zdim];
            for (size_t r=0; r<CellDim[0]; r++) {
                if (*px) {
                    Mdensity[mk] += *px;
                }
                px ++;
                mk ++;
            }
        }
        std::vector<size_t>density(Mdensity);
        std::make_heap(density.begin(), density.end(), std::less<>{});
        double checksum = 0.0;
        size_t n_cells = 0;
        for (k=0; k<Nthresh[2]; k++) {
            pop_heap(density.begin(), density.end(), std::less<>{});
            if (density.back()==0) {
                n_cells = k;
                break;
            }
            checksum += density.back();
            density.pop_back();
        }
        n_cells = (n_cells==0) ? Nthresh[2] : n_cells;
        checksum = checksum / (double) n_cells;
        if (checksum<=Mthresh[2]) {  // merge
            GridCoord[2][h] = 0;
            if(h>0) nDel[2] ++;
        } else if (checksum>=Sthresh[2]) {
            splitFlg[2][h] = (size_t) std::ceil(checksum/Sthresh[2]) -1;
            nAdd[2] += splitFlg[0][h];
        } else {
            std::fill(Mdensity.begin(), Mdensity.end(), 0.0); 
        }
    }
    std::cout << "dim 2: original " << GridDim[2] << ", del " << nDel[2] << ", add " << nAdd[2] <<" coordinates\n";
    MCoord.resize(GridDim[2]-nDel[2]+nAdd[2], 0);
    k = 1;

    for (size_t i=1; i<GridDim[2]; i++) {
        if (GridCoord[2][i]) {
            MCoord[k] = GridCoord[2][i];
//              std::cout << MCoord[k] << ", ";
            k ++;
        }
        if (splitFlg[2][i]) {
            double delta_s = 1.0 / splitFlg[2][i];
            for (size_t s=0; s<splitFlg[2][i]; s++) {
                MCoord[k] = MCoord[k-1] + delta_s;
                k ++;
            }
        }
    }
//        std::cout << "}\n";
    MergedGridCoord.push_back(MCoord);
//    std::cout << "finsih check Z-axis for coordinates merging...\n";
    return MergedGridCoord;
}


std::vector<std::vector<size_t>>
GridCellAMR3D(std::vector<std::vector<size_t>> GridCoord,
                   std::vector<size_t> histGridCell,
                   std::vector<size_t> GridDim,
                   std::vector<double> Mthresh  /*merge threshold*/,
                   std::vector<double> Sthresh /*split threshold*/)
{
    std::vector<std::vector<size_t>> MergedGridCoord;
    int n_dims = 3;
    size_t k, mk, zdim, ydim; // continuous in x
    std::vector<size_t> pos(n_dims);
    std::vector<size_t> CellDim {GridDim[0]-1, GridDim[1]-1, GridDim[2]-1};
    size_t dim1 = CellDim[0];
    size_t dim2 = CellDim[0] * CellDim[1];
    std::vector<std::vector<size_t>> splitFlg;
    std::vector<size_t> nDel(n_dims, 0);
    std::vector<size_t> nAdd(n_dims, 0);
    size_t checksum = 0, cnt_cells = 0;
    
    for (int d=0; d<n_dims; d++) {
        std::vector<size_t> Flg(GridDim[d], 0);
        splitFlg.push_back(Flg);
    }
    // pass 1: Y-Z plane
    std::vector<bool> mask_p(CellDim[2]*CellDim[1], false);
    for (size_t r=0; r<CellDim[0]; r++) {
        for (size_t h=0; h<CellDim[2]; h++) {
            zdim = h * dim2;
            for (size_t c=0; c<CellDim[1]; c++) {
                mk = h*CellDim[1] + c;
                k  = zdim + c*dim1 + r;
                if (histGridCell[k]) {
                    checksum += histGridCell[k];
                    if (!mask_p[mk]) {
                        cnt_cells ++;
                        mask_p[mk] = true;
                    }
                }
            }
        }
        double density = (double)checksum / (double)cnt_cells;
        if (density<=Mthresh[0]) {  // merge
            GridCoord[0][r] = 0;
            if (r>0) nDel[0] ++;
        } else if (density>=Sthresh[0]) {
            splitFlg[0][r] = (size_t) std::ceil(density/Sthresh[0]) -1;
            nAdd[0] += splitFlg[0][r];
        } else {
            checksum = 0;
            cnt_cells = 0;
            std::fill(mask_p.begin(), mask_p.end(), false);
        }
    }

    std::cout << "dim 0: original " << GridDim[0] << ", del " << nDel[0] << ", add " << nAdd[0] << " coordinates\n";
    std::vector<size_t> MCoord(GridDim[0]-nDel[0]+nAdd[0], 0);
    k = 1;
    for (size_t i=1; i<GridDim[0]; i++) {
        if (GridCoord[0][i]) {
            MCoord[k] = GridCoord[0][i];
//              std::cout << MCoord[k] << ", ";
            k ++;
        }
        if (splitFlg[0][i]) {
            double delta_s = 1.0 / splitFlg[0][i];
            for (size_t s=0; s<splitFlg[0][i]; s++) {
                MCoord[k] = MCoord[k-1] + delta_s;
                k ++;
            }
        }
    }
//        std::cout << "}\n";
    MergedGridCoord.push_back(MCoord);


    // pass 2: X-Z plane
    checksum = 0;
    cnt_cells = 0;
    mask_p.resize(CellDim[2]*CellDim[0]);
    std::fill(mask_p.begin(), mask_p.end(), false);
    for (size_t c=0; c<CellDim[1]; c++) {
        ydim = c*dim1;
        for (size_t h=0; h<CellDim[2]; h++) {
            zdim = h * dim2;
            size_t *px = &histGridCell.data()[ydim + zdim];
            mk = h*CellDim[0];
            for (size_t r=0; r<CellDim[0]; r++) {
                if (*px) {
                    checksum += *px;
                    if (!mask_p[mk]) {
                        cnt_cells ++;
                        mask_p[mk] = true;
                    }
                }
                px ++;
                mk ++;
            }
        }
        double density = (double)checksum / (double)cnt_cells;
//        std::cout << density << "\n";
        if (density<=Mthresh[1]) { // merge
            GridCoord[1][c] = 0;
            if (c>0) nDel[1] ++;
        } else if (density>=Sthresh[1]) {
            splitFlg[1][c] = (size_t) std::ceil(density/Sthresh[1]) -1;
            nAdd[1] += splitFlg[0][c];
        } else {
            checksum = 0;
            cnt_cells = 0;
            std::fill(mask_p.begin(), mask_p.end(), false);
        }
    }

    std::cout << "dim 1: original " << GridDim[1] << ", del " << nDel[1] << ", add "<< nAdd[1] << " coordinates\n";
    MCoord.resize(GridDim[1]-nDel[1]+nAdd[1], 0);
    k = 1;
    for (size_t i=1; i<GridDim[1]; i++) {
        if (GridCoord[1][i]) {
            MCoord[k] = GridCoord[1][i];
//              std::cout << MCoord[k] << ", ";
            k ++;
        }
        if (splitFlg[1][i]) {
            double delta_s = 1.0 / splitFlg[1][i];
            for (size_t s=0; s<splitFlg[1][i]; s++) {
                MCoord[k] = MCoord[k-1] + delta_s;
                k ++;
            }
        }
    }
//        std::cout << "}\n";
    MergedGridCoord.push_back(MCoord);
//    std::cout << "finsih check Y-axis for coordinates merging...\n";

    // pass 3: X-Y plane
    checksum = 0;
    cnt_cells = 0;
    mask_p.resize(CellDim[1] * CellDim[0]);
    std::fill(mask_p.begin(), mask_p.end(), false);
    for (size_t h=0; h<CellDim[2]; h++) {
        zdim = h * dim2;
        for (size_t c=0; c<CellDim[1]; c++) {
            ydim = c * dim1;
            mk = c * CellDim[0];
            size_t *px = &histGridCell.data()[ydim + zdim];
            for (size_t r=0; r<CellDim[0]; r++) {
                if (*px) {
                    checksum += *px;
                    if (!mask_p[mk]) {
                        cnt_cells ++;
                        mask_p[mk] = true;
                    }
                }
                px ++;
                mk ++;
            }
        }
        double density = (double)checksum / (double)cnt_cells;
        if (density<=Mthresh[2]) {  // merge
            GridCoord[2][h] = 0;
            if(h>0) nDel[2] ++;
        } else if (density>=Sthresh[2]) {
            splitFlg[2][h] = (size_t) std::ceil(density/Sthresh[2]) -1;
            nAdd[2] += splitFlg[0][h];
        } else {
            checksum = 0;
            cnt_cells = 0;
            std::fill(mask_p.begin(), mask_p.end(), false);
        }
    }

    std::cout << "dim 2: original " << GridDim[2] << ", del " << nDel[2] << ", add " << nAdd[2] <<" coordinates\n";
    MCoord.resize(GridDim[2]-nDel[2]+nAdd[2], 0);
    k = 1;
    for (size_t i=1; i<GridDim[2]; i++) {
        if (GridCoord[2][i]) {
            MCoord[k] = GridCoord[2][i];
//              std::cout << MCoord[k] << ", ";
            k ++;
        }
        if (splitFlg[2][i]) {
            double delta_s = 1.0 / splitFlg[2][i];
            for (size_t s=0; s<splitFlg[2][i]; s++) {
                MCoord[k] = MCoord[k-1] + delta_s;
                k ++;
            }
        }
    }
//        std::cout << "}\n";
    MergedGridCoord.push_back(MCoord);
//    std::cout << "finsih check Z-axis for coordinates merging...\n";
    return MergedGridCoord;
}



std::vector<std::vector<size_t>> 
GridCellMerge3D_Mdir(std::vector<std::vector<size_t>> GridCoord,
                   std::vector<size_t> histGridCell,
                   std::vector<size_t> GridDim,
                   std::vector<double> thresh) /* number of particles allowed in cells*/
{
    std::vector<std::vector<size_t>> MergedGridCoord;
    int n_dims = 3;
    size_t k, mk, zdim, ydim; // continuous in x
    std::vector<size_t> pos(n_dims);
    std::vector<size_t> CellDim {GridDim[0]-1, GridDim[1]-1, GridDim[2]-1}; 
    size_t dim1 = CellDim[0];
    size_t dim2 = CellDim[0] * CellDim[1];
    std::vector<size_t> nDel(n_dims, 0);
    size_t checksum = 0, cnt_cells = 0;
    // pass 1: Y-Z plane
    std::vector<bool> mask_p(CellDim[2]*CellDim[1], false);
    for (size_t r=0; r<CellDim[0]; r++) {
        for (size_t h=0; h<CellDim[2]; h++) {
            zdim = h * dim2;
            for (size_t c=0; c<CellDim[1]; c++) {
                mk = h*CellDim[1] + c;
                k  = zdim + c*dim1 + r;
                if (histGridCell[k]) {
                    checksum += histGridCell[k];
                    if (!mask_p[mk]) { 
                        cnt_cells ++;
                        mask_p[mk] = true; 
                    }
                }
            }
        }
        double density = (double)checksum / (double)cnt_cells;
        if (density<=thresh[0]) {  // merge
            GridCoord[0][r] = 0;
            if (r>0) nDel[0] ++;
        } else {
            checksum = 0;
            cnt_cells = 0;
            std::fill(mask_p.begin(), mask_p.end(), false);
        }
    }

    std::cout << "dim 0: original " << GridDim[0] << ", del " << nDel[0] << " coordinates\n";
    std::vector<size_t> MCoord(GridDim[0]-nDel[0], 0);
    k = 1;
    for (size_t i=1; i<GridDim[0]; i++) {
        if (GridCoord[0][i]) {
            MCoord[k] = GridCoord[0][i];
//              std::cout << MCoord[k] << ", ";
            k ++;
        }
    }
//        std::cout << "}\n";
    MergedGridCoord.push_back(MCoord);
//    std::cout << "finsih check X-axis for coordinates merging...\n";

    // pass 2: X-Z plane
    checksum = 0;
    cnt_cells = 0;
    mask_p.resize(CellDim[2]*(MergedGridCoord[0].size()));
    std::fill(mask_p.begin(), mask_p.end(), false);
    for (size_t c=0; c<CellDim[1]; c++) {
        ydim = c*dim1;
        for (size_t h=0; h<CellDim[2]; h++) {
            zdim = h * dim2;
            size_t *px = &histGridCell.data()[ydim + zdim];
            mk = h*MergedGridCoord[0].size();
            size_t r=0;
            while (r<CellDim[0]) {
                if (*px) {
                    checksum += *px; 
                    if (GridCoord[0][r+1]) {
                        if (!mask_p[mk]) { 
                            cnt_cells ++;
                            mask_p[mk] = true;
                        }
                        mk ++;
                    }
                }
                r ++;
                px ++;
            }
        }
        double density = (double)checksum / (double)cnt_cells;
//        std::cout << density << "\n";
        if (density<=thresh[1]) { // merge
            GridCoord[1][c] = 0;
            if (c>0) nDel[1] ++;
        } else {
            checksum = 0;
            cnt_cells = 0;
            std::fill(mask_p.begin(), mask_p.end(), false);
        }
    }

    std::cout << "dim 1: original " << GridDim[1] << ", del " << nDel[1] << " coordinates\n";
    MCoord.resize(GridDim[1]-nDel[1], 0);
    k = 1;
    for (size_t i=1; i<GridDim[1]; i++) {
        if (GridCoord[1][i]) {
            MCoord[k] = GridCoord[1][i];
//              std::cout << MCoord[k] << ", ";
            k ++;
        }
    }
//        std::cout << "}\n";
    MergedGridCoord.push_back(MCoord);
//    std::cout << "finsih check Y-axis for coordinates merging...\n";

    // pass 3: X-Y plane
    checksum = 0;
    cnt_cells = 0;
    mask_p.resize((CellDim[1]-nDel[1]) * (CellDim[0]-nDel[0]));
    std::fill(mask_p.begin(), mask_p.end(), false);
    for (size_t h=0; h<CellDim[2]; h++) {
        zdim = h * dim2;
        for (size_t c=0; c<CellDim[1]; c++) {
            ydim = c * dim1;
            mk = c*MergedGridCoord[0].size();
            size_t *px = &histGridCell.data()[ydim + zdim];
            size_t r = 0;
            while (r<CellDim[0]) {
                if (*px) {
                    checksum += *px; 
                    if (GridCoord[0][r+1] && GridCoord[1][c+1]) {
                        if (!mask_p[mk]) {
                            cnt_cells ++;
                            mask_p[mk] = true;
                        } 
                        mk ++;
                    }
                }
                px ++;
                r ++;
            }
        }
        double density = (double)checksum / (double)cnt_cells;
        if (density<=thresh[2]) {  // merge
            GridCoord[2][h] = 0;
            if(h>0) nDel[2] ++;
        } else {
            checksum = 0;
            cnt_cells = 0;
            std::fill(mask_p.begin(), mask_p.end(), false);    
        }
    }

    std::cout << "dim 2: original " << GridDim[2] << ", del " << nDel[2] << " coordinates\n";
    MCoord.resize(GridDim[2]-nDel[2], 0);
    k = 1;
    for (size_t i=1; i<GridDim[2]; i++) {
        if (GridCoord[2][i]) {
            MCoord[k] = GridCoord[2][i];
//              std::cout << MCoord[k] << ", ";
            k ++;
        }
    }
//        std::cout << "}\n";
    MergedGridCoord.push_back(MCoord);
//    std::cout << "finsih check Z-axis for coordinates merging...\n";
    return MergedGridCoord;
}


// cell: true|false --> left top corner of the cell | closest grid point
void closest_Node2nonUniformGrid(std::vector<size_t> &nodeMapGrid, 
                       std::vector<std::vector<double>> nodeCoord, 
                       std::vector<std::vector<size_t>> MergedGridCoord, 
                       std::vector<double> minvGrid,
                       std::vector<double> spaceGrid)
{
    size_t nNodePt = nodeCoord[0].size();
    int n_dims = MergedGridCoord.size();
    double posVal;
    std::vector<double> index(n_dims);
    std::vector<size_t> dims(n_dims, 1);
    std::cout << "non-uniform grid sample rate: " << MergedGridCoord[0].size();
    for (int d=1; d<n_dims; d++) {
        dims[d] = dims[d-1] * MergedGridCoord[d-1].size();
        std::cout << " x "<< MergedGridCoord[d].size(); 
    }
    std::cout << "\n";
    
    for (size_t i=0; i<nNodePt; i++) {
        for (int d=0; d<n_dims; d++) {
            posVal = nodeCoord[d][i] - minvGrid[d];
            index[d] = std::floor(posVal / spaceGrid[d]); 
            size_t p = 1, loc;
            double dprev, dnext;
            while (p<MergedGridCoord[d].size()) {
                if (index[d] <= (double)MergedGridCoord[d][p]) {
                    dprev = index[d] - (double)MergedGridCoord[d][p-1];
                    dnext = (double)MergedGridCoord[d][p] - index[d]; 
                    loc   = (dprev<dnext) ? p-1 : p;
                    nodeMapGrid[i] += loc * dims[d];
                    break;
                }
                p++;
            } 
        }
    }   
}


std::vector<double> convertGrid2Vec(std::vector<double> GridPointVal,   
                                    std::vector<size_t> &GridSparseMap)
{
    size_t nGridPt = GridPointVal.size();
    size_t nGridSparse = 0;
    for (size_t i=0; i<nGridPt; i ++) {
        if (GridPointVal[i]!=0) nGridSparse ++;
    }
    //std::cout << "number of non-zero grid points: " << nGridSparse << " (" << (double)nGridSparse / (double)nGridPt * 100.0 << "% of Grid pts) \n";
    std::vector<double> GridSparseVal(nGridSparse, 0.0);
    size_t sparseId = 0;
    for (size_t i=0; i<nGridPt; i ++) {
        if (GridPointVal[i]!=0) {
            GridSparseVal[sparseId] = GridPointVal[i];
            GridSparseMap[i] = sparseId;
            sparseId ++;
        } 
    }
    return GridSparseVal;
}

std::vector<double> closest_GridValResi_sparse(std::vector<size_t> nodeMapGrid,
                        std::vector<size_t> GridSparseMap,
                        std::vector<double> var_in,
                        std::vector<size_t> nCluster,
                        std::vector<double> &residual,
                        size_t nGridSparse)
{
    size_t nNodePt = residual.size();
    //size_t nGridPt = nodeMapGrid.size();
    std::cout << "nNodePt = " << nNodePt << "\n"; 
    std::vector<double> GridSparseVal(nGridSparse, 0.0);

    for (size_t i=0; i<nNodePt; i++) {
        size_t ig = nodeMapGrid[i];
        size_t k  = GridSparseMap[ig];
        GridSparseVal[k] += var_in[i] / (double)nCluster[ig];
    }
    /* 
    for (size_t i=0; i<nGridPt; i++) {
        if (nCluster[i]>0) {
            size_t k = GridSparseMap[i];
            GridSparseVal[k] = GridSparseVal[k] / (double)nCluster[i];
        }
    }
    */
    //std::vector<bool> GridSparseVal_flg(nGridSparse, false);
    for (size_t i=0; i<nNodePt; i++) {
        size_t k = GridSparseMap[nodeMapGrid[i]];
        //if (GridSparseVal_flg[k]==false) {
        //    GridSparseVal[k] = GridSparseVal[k] / (double)nCluster[nodeMapGrid[i]];
        //    GridSparseVal_flg[k] = true;
        //}
        residual[i] = var_in[i] - GridSparseVal[k];
    }
    return GridSparseVal;
}

std::vector<double> convertGrid2Vec_m(std::vector<double> GridPointVal,
                                    std::vector<size_t> GridSparseMap)
{
    size_t nGridSparse = GridSparseMap.size();
    //std::cout << "number of non-zero grid points: " << nGridSparse << " (" << (double)nGridSparse / (double)nGridPt * 100.0 << "% of Grid pts) \n";
    std::vector<double> GridSparseVal(nGridSparse, 0.0);
    for (size_t i=0; i<nGridSparse; i ++) {
        GridSparseVal[i] = GridPointVal[GridSparseMap[i]];
    }
    return GridSparseVal;
}

void nonUni_GridValResi_MLE(std::vector<size_t> nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<std::vector<size_t>> MergedGridCoord,
                            std::vector<double> spaceGrid,
                            std::vector<double> minvGrid,
                            std::vector<double> var_in,
                            std::vector<double> &GridPointVal,
                            std::vector<double> &residual)
{
    size_t nNodePt = residual.size();
    nonUni_closest_GridVal_MLE(nodeMapGrid, nodeCoord, MergedGridCoord, spaceGrid, minvGrid, var_in, GridPointVal);
    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        residual[i] = var_in[i] - GridPointVal[k];
    }
}

void nonUni_GridValResi_Ghost(std::vector<size_t> nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<std::vector<size_t>> MergedGridCoord,
                            std::vector<double> spaceGrid,
                            std::vector<double> minvGrid,
                            std::vector<double> var_in,
                            std::vector<double> &GridPointVal,
                            std::vector<double> &residual)
{
    size_t nNodePt = residual.size();
    size_t nGridPt = GridPointVal.size();
    int n_dims     = (int)MergedGridCoord.size();
    size_t Dr = MergedGridCoord[0].size() - 1;
    size_t Dc = MergedGridCoord[1].size() - 1;
    size_t Dh = MergedGridCoord[2].size() - 1;
    std::vector<int> dims(n_dims, 1);
    for (int d=1; d<n_dims; d++) {
        dims[d] = dims[d-1] * (int) MergedGridCoord[d-1].size();
    }
    size_t nCellVertices = (size_t)std::pow(2, n_dims);
    size_t r1, c1, h1, r2, c2, h2;

    std::vector<size_t> nCluster(nGridPt, 0);
    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        nCluster[k] ++;
        GridPointVal[k] += var_in[i];
    }
    for (size_t j=0; j<nGridPt; j++) {
        if (nCluster[j]>0) {
            GridPointVal[j] = GridPointVal[j] / (double)nCluster[j];
        }
//        if (j<100) std::cout << "nCluster[" << j << "] = " <<  nCluster[j] << "\n";
    }
//    nonUni_closest_GridVal_MLE(nodeMapGrid, nodeCoord, MergedGridCoord, spaceGrid, minvGrid, var_in, GridPointVal);

    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        std::vector <double> fieldVals;
        h1 = (size_t) ((double)k / (double)dims[2]);
        c1 = (size_t) ((double)(k-h1*dims[2]) / (double)dims[1]);
        r1 = (size_t) (k - c1*dims[1] - h1*dims[2]);
        h2 = (h1+1 > Dh) ? Dh : h1+1;
        c2 = (c1+1 > Dc) ? Dc : c1+1;
        r2 = (r1+1 > Dr) ? Dr : r1+1;
        std::vector<std::vector<size_t>> index;
        if (n_dims==3) {
            index.push_back({r1, r2, r1, r2, r1, r2, r1, r2});
            index.push_back({c1, c1, c2, c2, c1, c1, c2, c2});
            index.push_back({h1, h1, h1, h1, h2, h2, h2, h2});
        } else {
            index.push_back({r1, r2, r1, r2});
            index.push_back({c1, c1, c2, c2});
        }
//        std::cout << i << ": \n";
        double fVal = GridPointVal[k];
        for (size_t iv=0; iv<nCellVertices; iv++) {
            size_t kv = index[0][iv] + index[1][iv]*dims[1] +
                        index[2][iv]*dims[2];
            if (GridPointVal[kv]!=0) {
                fVal = GridPointVal[kv]; 
            }
            fieldVals.push_back(fVal);
//            std::cout << fieldVals[iv] << ", ";
        }
        std::vector<std::vector<double>> gCoord;
        for (int d=0; d<n_dims; d++) {
            std::vector<double>Coord(index[0].size());
            for (size_t iv=0; iv<index[0].size(); iv++) {
                Coord[iv] = MergedGridCoord[d][index[d][iv]] * spaceGrid[d] + minvGrid[d];
            }
            gCoord.push_back(Coord);
        }
//        std::cout << var_in[i] << "\n";
        std::vector<double> nCoord {nodeCoord[0][i], nodeCoord[1][i], nodeCoord[2][i]};
        double pp = interpolateGridtoNode(fieldVals, index, nCoord, gCoord);
        residual[i] = var_in[i] - pp;
    }
}

void nonUni_closest_GridVal_MLE(std::vector<size_t> nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<std::vector<size_t>> MergedGridCoord,
                            std::vector<double> spaceGrid,
                            std::vector<double> minvGrid,
                            std::vector<double> var_in,
                            std::vector<double> &GridPointVal)
{
    size_t nNodePt = nodeMapGrid.size();
    size_t nGridPt = GridPointVal.size();
    int n_dims     = (int)MergedGridCoord.size();
    double spaceGrid_min = *std::min_element(spaceGrid.begin(), spaceGrid.end());
    std::vector<int> dims(n_dims, 1);
    double radius = spaceGrid[0]*spaceGrid[0];
    for (int d=1; d<n_dims; d++) {
        dims[d] = dims[d-1] * (int) MergedGridCoord[d-1].size();
        radius += spaceGrid[d]*spaceGrid[d];
    }
    radius = std::sqrt(radius);
    std::vector<size_t> pos(n_dims);
    std::vector<double> wDist(nGridPt, 0.0);
    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        pos[2] = (size_t) ((double)k / (double)dims[2]);
        pos[1] = (size_t) ((double)(k - pos[2]*dims[2]) / (double)dims[1]);
        pos[0] = (size_t) (k - pos[1]*dims[1] - pos[2]*dims[2]);
        double w=0.0;
        for (int d=0; d<n_dims; d++) {
            double dist = ((MergedGridCoord[d][pos[d]]*spaceGrid[d] + minvGrid[d]) - nodeCoord[d][i])/spaceGrid_min;
            w += dist * dist;
        }
        w = std::sqrt(w);
        if (w<DELTA) w = DELTA;
        w = (radius-w)/(radius*w);
        w = w*w;
        size_t kv = pos[0] + pos[1]*dims[1] + pos[2]*dims[2];
//        w = 1.0 / std::sqrt(w);
        wDist[kv] += w;
        GridPointVal[kv] += w*var_in[i];
    }
    for (size_t j=0; j<nGridPt; j++) {
        if (wDist[j]>0) {
//            std::cout << j << ": " << GridPointVal[j] << ", " << wDist[j] <<  ", " << GridPointVal[j] / wDist[j] << "\n";
            GridPointVal[j] = GridPointVal[j] / wDist[j];
        }
    }
}


void nonUni_GridVal_MLE(std::vector<size_t> nodeMapGrid,
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<std::vector<size_t>> MergedGridCoord,
                            std::vector<double> spaceGrid,
                            std::vector<double> minvGrid,
                            std::vector<double> var_in,
                            std::vector<double> &GridPointVal)
{
    size_t nNodePt = nodeMapGrid.size();
    size_t nGridPt = GridPointVal.size();
    int n_dims     = (int)MergedGridCoord.size();
    size_t Dr = MergedGridCoord[0].size() - 1;
    size_t Dc = MergedGridCoord[1].size() - 1;
    size_t Dh = MergedGridCoord[2].size() - 1;
    double spaceGrid_min = *std::min_element(spaceGrid.begin(), spaceGrid.end());
    double radius = spaceGrid[0]*spaceGrid[0];
    std::vector<int> dims(n_dims, 1);
    for (int d=1; d<n_dims; d++) {
        dims[d] = dims[d-1] * (int) MergedGridCoord[d-1].size();
        radius += spaceGrid[d] * spaceGrid[d];
    }
    radius = std::sqrt(radius);
    size_t nCellVertices = (size_t)std::pow(2, n_dims);
    size_t r1, c1, h1, r2, c2, h2;
    std::vector<double> wDist(nGridPt, 0.0);
    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        h1 = (size_t) ((double)k / (double)dims[2]);
        c1 = (size_t) ((double)(k - h1*dims[2]) / (double)dims[1]);
        r1 = (size_t) (k - c1*dims[1] - h1*dims[2]);
        h2 = (h1+1 > Dh) ? Dh : h1+1;
        c2 = (c1+1 > Dc) ? Dc : c1+1;
        r2 = (r1+1 > Dr) ? Dr : r1+1;
        std::vector<std::vector<size_t>> index;
        if (n_dims==3) {
            index.push_back({r1, r2, r1, r2, r1, r2, r1, r2});
            index.push_back({c1, c1, c2, c2, c1, c1, c2, c2});
            index.push_back({h1, h1, h1, h1, h2, h2, h2, h2});
         }else {
            index.push_back({r1, r2, r1, r2});
            index.push_back({c1, c1, c2, c2});
        }
        for (size_t iv=0; iv<nCellVertices; iv++) {
            double w = 0.0;
            for (int d=0; d<n_dims; d++) {
                double dist = ((MergedGridCoord[d][index[d][iv]]*spaceGrid[d] + minvGrid[d]) - nodeCoord[d][i])/spaceGrid_min;
//                std::cout << i << ": " << dist << ", " << MergedGridCoord[d][index[d][iv]]*spaceGrid[d] + minvGrid[d] << ", " << nodeCoord[d][i] << "\n";
                w += dist * dist;
            }
            w = std::sqrt(w);
            if (w<DELTA) w = DELTA;
            w = (w >= radius) ? 0 : (radius-w)/(radius*w);
            w = w*w;
            size_t kv = index[0][iv] + index[1][iv]*dims[1] +
                        index[2][iv]*dims[2];
//            w = 1.0 / std::sqrt(w);
            wDist[kv] += w;
            GridPointVal[kv] += w*var_in[i];
        }
    }
    for (size_t j=0; j<nGridPt; j++) {
        if (wDist[j]>0) {
//            std::cout << j << ": " << GridPointVal[j] << ", " << wDist[j] <<  ", " << GridPointVal[j] / wDist[j] << "\n";
            GridPointVal[j] = GridPointVal[j] / wDist[j];
        }
    }
}

// each cell corner receives contributions from all nodes inside the cell
// which means, each grid point receives contributions from nodes in 2^d cells 
// only work for 3D
void cellInterp_GridValResi(std::vector<size_t> nodeMapGrid, 
                            std::vector<std::vector<double>> nodeCoord,
                            std::vector<std::vector<size_t>> MergedGridCoord,
                            std::vector<double> spaceGrid,
                            std::vector<double> minvGrid,
                            std::vector<double> var_in, 
                            std::vector<double> &GridPointVal, 
                            std::vector<double> &residual)
{
    size_t nNodePt = residual.size();
    int n_dims     = (int)MergedGridCoord.size();
    size_t Dr = MergedGridCoord[0].size() - 1;
    size_t Dc = MergedGridCoord[1].size() - 1;
    size_t Dh = MergedGridCoord[2].size() - 1;
    std::vector<int> dims(n_dims, 1);
    for (int d=1; d<n_dims; d++) {
        dims[d] = dims[d-1] * (int) MergedGridCoord[d-1].size();
    }
    size_t nCellVertices = (size_t)std::pow(2, n_dims);
    size_t r1, c1, h1, r2, c2, h2;
    nonUni_GridVal_MLE(nodeMapGrid, nodeCoord, MergedGridCoord, spaceGrid, minvGrid, var_in, GridPointVal);
    std::cout << "finish interpolation\n";

    for (size_t i=0; i<nNodePt; i++) {
        size_t k = nodeMapGrid[i];
        std::vector <double> fieldVals;
        h1 = (size_t) ((double)k / (double)dims[2]);
        c1 = (size_t) ((double)(k-h1*dims[2]) / (double)dims[1]);
        r1 = (size_t) (k - c1*dims[1] - h1*dims[2]);
        h2 = (h1+1 > Dh) ? Dh : h1+1;
        c2 = (c1+1 > Dc) ? Dc : c1+1;
        r2 = (r1+1 > Dr) ? Dr : r1+1;
        std::vector<std::vector<size_t>> index;
        if (n_dims==3) {
            index.push_back({r1, r2, r1, r2, r1, r2, r1, r2});
            index.push_back({c1, c1, c2, c2, c1, c1, c2, c2});
            index.push_back({h1, h1, h1, h1, h2, h2, h2, h2});
        } else {
            index.push_back({r1, r2, r1, r2});
            index.push_back({c1, c1, c2, c2});    
        }
//        std::cout << i << ": \n";
        for (size_t iv=0; iv<nCellVertices; iv++) {
            size_t kv = index[0][iv] + index[1][iv]*dims[1] + 
                        index[2][iv]*dims[2];
            fieldVals.push_back(GridPointVal[kv]);   
//            std::cout << fieldVals[iv] << ", ";
        }
//        std::cout << var_in[i] << "\n";
        std::vector<std::vector<double>> gCoord;
        for (int d=0; d<n_dims; d++) {
            std::vector<double>Coord(index[0].size());
            for (size_t iv=0; iv<index[0].size(); iv++) {
                Coord[iv] = MergedGridCoord[d][index[d][iv]] * spaceGrid[d] + minvGrid[d];
            }
            gCoord.push_back(Coord);
        }
        std::vector<double> nCoord {nodeCoord[0][i], nodeCoord[1][i], nodeCoord[2][i]};
        double pp = interpolateGridtoNode(fieldVals, index, nCoord, gCoord); 
        residual[i] = var_in[i] - pp; 
//        std::cout << var_in[i] << ", " << fieldVals[0] << ", " << fieldVals[1] << ", " << pp << ", " << residual[i] << "\n";
    }
}


double interpolateGridtoNode(std::vector <double> fieldVals, 
                             std::vector<std::vector<size_t>> index, 
                             std::vector<double> nodeCoord,
                             std::vector<std::vector<double>> GridCoord)
{
    double p1, p2, deltaX, deltaY, deltaZ, xl, xr, yl, yr, zl, zr, r1, r2, pp;
    size_t n_dims = (index[0].size()==8) ? 3 : 2;
    xl = GridCoord[0][0]; 
    xr = GridCoord[0][3]; 
    yl = GridCoord[1][0]; 
    yr = GridCoord[1][3]; 
    deltaX = xr - xl;
    deltaY = yr - yl;
    if (deltaX==0) {
        r1 = fieldVals[0]; 
        r2 = fieldVals[2];
    } else { 
        r1 = (fieldVals[0] * (xr-nodeCoord[0]) + 
            fieldVals[1] * (nodeCoord[0]-xl)) / deltaX;
        r2 = (fieldVals[2] * (xr-nodeCoord[0]) + 
            fieldVals[3] * (nodeCoord[0]-xl)) / deltaX; 
    } 
    if (deltaY==0) {
        p1 = (r1 + r2)/2;
    } else {
        p1 = (r1 * (yr-nodeCoord[1]) + r2 * (nodeCoord[1] - yl)) / deltaY; 
    }

    if (n_dims==3) {
        zl = GridCoord[2][0]; 
        zr = GridCoord[2][7]; 
        deltaZ = zr - zl;
        if (deltaZ == 0) {
            pp = p1;
        } else { 
            if (deltaX==0) {
                r1 = fieldVals[4]; 
                r2 = fieldVals[6];
            } else {
                r1 = (fieldVals[4] * (xr - nodeCoord[0])
                    + fieldVals[5] * (nodeCoord[0] - xl)) / deltaX;
                r2 = (fieldVals[6] * (xr - nodeCoord[0])
                    + fieldVals[7] * (nodeCoord[0] - xl)) / deltaX;
            } 
            if (deltaY==0) {
                p2 = (r1 + r2)/2.0;
            } else {
                p2 = (r1*(yr-nodeCoord[1]) + r2*(nodeCoord[1]-yl)) / deltaY; 
            }
            pp = (p1 * (zr-nodeCoord[2]) + p2 * (nodeCoord[2]-zl)) / deltaZ; 
        }
//        std::cout << r1 << ", " << r2 << ", " << p2 << ", " << pp <<"\n";
    } else { // 2D
        pp = p1;
    }
    return pp;
}

