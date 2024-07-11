Error-controlled lossy compression for unstructured data  

1. mgard\_adios\_ge.cpp: read data through ADIOS, compressing through MGARD-GPU, then writing the results out. The compression was conducted during the ADIOS write, as an operator
./mgard\_adios\_ge ../../dataset/ sol\_4114800\_aver.bp 0.0001 3 P\_aver Rho\_aver U\_aver

2. mgard\_adios\_decompress.cpp: read the compressed data through ADIOS, decompressing the data through MGARD-GPU, as an operator

3. MeshGrid: generating the mesh-to-grid mapping, saving it as a separate file for latter usage
