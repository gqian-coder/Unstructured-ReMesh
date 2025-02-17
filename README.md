Error-controlled lossy compression for unstructured data  

1. mgard\_adios\_ge.cpp: read data through ADIOS, compressing through MGARD-GPU, then writing the results out. The compression was conducted during the ADIOS write, as an operator
   
./mgard\_adios\_ge ../../dataset/ sol\_4114800\_aver.bp 3 P\_aver Rho\_aver U\_aver 0.001

2. mgard\_adios\_decompress.cpp: read the compressed data through ADIOS, decompressing the data through MGARD-GPU, as an operator
   
./mgard\_adios\_decompress ./ sol\_4114800\_aver.bp.compressed 3 P\_aver Rho\_aver U\_aver

3. MeshGrid: generating the mesh-to-grid mapping, saving it as a separate file for latter usage
   
./MeshGrid ../../dataset/ sol\_4114800\_aver.bp 3 0.2

4. mgard\_adios\_remesh.cpp: compress the unstructured data via the interpolation-based approach, using the pre-calculated mesh-grid mapping
   
./mgard\_adios\_remesh ../../dataset/ sol\_4114800\_aver.bp Mesh2GridMap.bp 3 P\_aver Rho\_aver U\_aver 0.001 0.3

5. mgard\_ge.cpp: compress data using mgard's high-level API
   
./mgard\_ge ../../dataset/ sol\_4114800\_aver.bp 3 P\_aver Rho\_aver U\_aver 0 0.001

6. mgard\_ge\_cpu.cpp: compress data using mgard's low-level API
   
./mgard\_ge\_cpu ../../dataset/ sol\_4114800\_aver.bp 3 P\_aver Rho\_aver U\_aver 0 0.001 

7. mgard\_adios\_decompress\_remesh: decompress the remsh/interpolation compressed data, using pre-calculated mesh-grid mapping
   
./mgard\_adios\_decompress\_remesh ./ sol\_4114800\_aver.bp.remshCompressed ../../dataset/Mesh2GridMap.bp 3 P\_aver Rho\_aver U\_aver 5  

8. calc\_err: taking two bp files and variable lists, computing the L2 error of the two
   
./calc\_err ./ sol\_4114800\_aver.bp sol\_4114800\_aver.bp.remshCompressed.remeshDecompressed 3 P\_aver Rho\_aver U\_aver

9. mgardPlug\_adios\_ge: compressing data using interpolation-based approach, using the pre-calculated mesh-grid mapping; conducted through adios plug operator
    
mpirun -np 4 ./mgardPlug\_adios\_ge ../../dataset/ sol_4114800_aver.bp 2 U_aver P_aver 1e-3 16

10. mgardPlug\_adios\_decompress: decompress the remsh/interpolation compressed data, using pre-calculated mesh-grid mapping; conducted through adios plug operator
    
mpirun -np 4 ./mgardPlug_adios_decompress ./ sol_4114800_aver.bp.compressed 2 P_aver U_aver 16
