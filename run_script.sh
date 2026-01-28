#!/bin/bash
ldmunge /home/adios/shared/Software/CompressMGARDMeshToGridOperator/CompressMGARDMeshToGridOperator/build:/opt/adios2/debug/lib:/opt/mgard/lib
export LD_LIBRARY_PATH

BINDIR=/home/adios/shared/Software/CompressMGARDMeshToGridOperator/Unstructured-ReMesh/build

# generate mesh
$BINDIR/MeshGrid sol/sol_4169000_aver.bp map.bp 0.15

# mgardPlug
$BINDIR/mgardPlug_adios_ge sol/sol_4169000_aver.bp sol/sol_compressed.bp 1e-4

# regular mgard - new interface: auto-detects FlowSolution variables
# Usage: mgard_adios_ge input.bp output.bp error_bound [n_blocks]
$BINDIR/mgard_adios_ge /lustre/orion/proj-shared/cfd164/norbert_vki_case_frontier/p1/sol/sol_4169000_aver.bp sol_compressed.bp 1e-4

# with optional n_blocks parameter (e.g., process only first 10 blocks)
# $BINDIR/mgard_adios_ge /lustre/orion/proj-shared/cfd164/norbert_vki_case_frontier/p1/sol/sol_4169000_aver.bp sol_compressed.bp 1e-4 10

# decompress - new interface: auto-detects all variables and blocks
# Usage: mgard_adios_decompress compressed_input.bp
$BINDIR/mgard_adios_decompress sol_compressed.bp
