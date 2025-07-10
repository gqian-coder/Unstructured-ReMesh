#!/bin/bash
ldmunge /home/adios/shared/Software/CompressMGARDMeshToGridOperator/CompressMGARDMeshToGridOperator/build:/opt/adios2/debug/lib:/opt/mgard/lib
export LD_LIBRARY_PATH

BINDIR=/home/adios/shared/Software/CompressMGARDMeshToGridOperator/Unstructured-ReMesh/build

# generate mesh
$BINDIR/MeshGrid sol/sol_4169000_aver.bp map.bp 0.15

# mgardPlug
$BINDIR/mgardPlug_adios_ge sol/sol_4169000_aver.bp sol/sol_compressed.bp 1e-4

# regular mgard
$BINDIR/mgard_adios_ge /lustre/orion/proj-shared/cfd164/norbert_vki_case_frontier/p1/sol/ sol_4169000_aver.bp 1 P_aver 1e-4 96 0
