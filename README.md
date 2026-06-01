Error-controlled lossy compression for unstructured data.

## Default MGARD via ADIOS operator

### 1) Compress (`mgard_adios_ge`)

`mgard_adios_ge` is the default ADIOS+MGARD path in this repo. It reads BP,
adds ADIOS `mgard` operator on FlowSolution variables, and writes compressed BP.

```bash
# Usage:
#   mgard_adios_ge <input.bp> <output.bp> <error_bound> [n_blocks]

./mgard_adios_ge input.bp output_compressed.bp 0.001

# Optional n_blocks: process first N blocks only
./mgard_adios_ge input.bp output_compressed.bp 0.001 10
```

Environment variables:

- `MGARD_X_DEVICE_TYPE=SERIAL|HIP|CUDA` controls MGARD device backend.
- `MGARD_VAR_FILTER=<substring>` compress only FlowSolution variables whose
	full name contains this substring; other FlowSolution variables are passed
	through uncompressed.
- `NO_PASSTHROUGH=1` disables passthrough of non-FlowSolution double variables
	so output mainly contains compressed FlowSolution data.
- `MAX_STEPS=<N>` limits number of timesteps processed.

### 2) Decompress (`mgard_adios_decompress`)

`mgard_adios_decompress` reads compressed BP and writes decompressed BP by
reading all `*FlowSolution*` variables (double/float).

```bash
# Usage:
#   mgard_adios_decompress <compressed_input.bp> [decompressed_output.bp]

./mgard_adios_decompress output_compressed.bp
# Output: output_compressed_decompressed.bp

./mgard_adios_decompress output_compressed.bp my_decompressed.bp
```

This tool auto-detects FlowSolution variables and processes all timesteps/blocks.

## Centroid operator workflow

### 1) Compress with centroid split (`mgardCentroid_adios_ge`)

`mgardCentroid_adios_ge` compresses FlowSolution variables using the
`CompressMGARDCentroidOperator` ADIOS2 plugin.

```bash
# Usage:
#   mgardCentroid_adios_ge <input.bp> <output.bp> <rel_tolerance> [ebratio [n_blocks]]

export ADIOS2_PLUGIN_PATH=/lustre/orion/cfd164/proj-shared/gongq/CompressMGARDCentroidOperator/build_hip
export CENTROID_DEVICE=gpu
export MGARD_X_DEVICE_TYPE=HIP

./mgardCentroid_adios_ge \
	../../sol_p1/sol_4114800_aver.bp \
	sol_4114800_aver.mgrC.bp \
	1e-3 0.8
```

Useful runtime options:

- `FLOW_VAR_LIST=P_aver,Rho_aver` to compress only selected FlowSolution vars.
- `NO_PASSTHROUGH=1` to skip writing connectivity/coordinate passthrough data.
- `MAX_STEPS`, `MAX_VARS`, `MAX_BLOCKS` for quick smoke tests.

### 2) Explicitly decompress (`mgardCentroid_adios_decompress`)

You can materialize a decompressed BP file before error analysis:

```bash
# Usage:
#   mgardCentroid_adios_decompress <compressed.bp> [decompressed.bp]

export ADIOS2_PLUGIN_PATH=/lustre/orion/cfd164/proj-shared/gongq/CompressMGARDCentroidOperator/build_hip
export CENTROID_DEVICE=gpu
export MGARD_X_DEVICE_TYPE=HIP

./mgardCentroid_adios_decompress \
	sol_4114800_aver.mgrC.bp \
	sol_4114800_aver.mgrC.decompressed.bp
```

## Error measurement (`calc_err`)

`calc_err` compares two BP files for all `FlowSolution` variables found in the
second file and present in the first file.

```bash
# Usage:
#   calc_err <original.bp> <compressed_or_decompressed.bp>

./calc_err sol_original.bp sol_4114800_aver.mgrC.bp
```

`calc_err` reports errors in two ways:

- Per-block printout:
	- `linf`: absolute max error on that block.
	- `rmse`: RMSE on that block.
	- `rel_linf` and `rel_rmse`: normalized by that block's local range
		(`block_max - block_min`).
- Final summary table:
	- `L-inf (rel)` and `RMSE (rel)` normalized by the variable global range
		(`global_max - global_min`) for the processed data.

Optional env vars for `calc_err`:

- `EB=<value>`: add per-block `PASS/FAIL` check on `rel_rmse <= EB`.
- `MAX_BLOCKS=<N>`: compare only first N blocks.
- `BLOCK_OFFSET=<N>`: align compressed block 0 with original block N.

## When mesh/connectivity are missing in compressed output

If compression was run with `NO_PASSTHROUGH=1`, the compressed BP may not
contain connectivity variables. Decompression still works if the plugin can
load connectivity from an external mesh BP.

Set these env vars before running `calc_err` (or `mgardCentroid_adios_decompress`):

```bash
export CENTROID_MESHFILE=/lustre/orion/cfd164/proj-shared/gongq/sol_p1/sol_4114800_aver.bp
export CENTROID_CONN_VAR=/hpMusic_base/hpMusic_Zone/Elem/ElementConnectivity
```

Then run error evaluation directly on the compressed file:

```bash
./calc_err \
	/lustre/orion/cfd164/proj-shared/gongq/sol_p1/sol_4114800_aver.bp \
	/lustre/orion/cfd164/proj-shared/gongq/Unstructured-ReMesh/build_cfd164/sol_4114800_aver.mgrC.allEB1e-5.bp
```

## Other tools in this repo

1. `MeshGrid`: generate mesh-to-grid mapping for remesh workflow.

```bash
./MeshGrid ../../dataset/ sol_4114800_aver.bp 3 0.2
```

2. `mgard_adios_remesh`: interpolation/remesh-based compression using
	 precomputed mesh-grid mapping.

```bash
./mgard_adios_remesh \
	../../dataset/ sol_4114800_aver.bp Mesh2GridMap.bp \
	3 P_aver Rho_aver U_aver 0.001 0.3
```

3. `mgard_adios_decompress_remesh`: decompress remesh-compressed data.

```bash
./mgard_adios_decompress_remesh \
	./ sol_4114800_aver.bp.remshCompressed ../../dataset/Mesh2GridMap.bp \
	3 P_aver Rho_aver U_aver 5
```

4. `mgard_ge`: MGARD high-level API test.

```bash
./mgard_ge ../../dataset/ sol_4114800_aver.bp 3 P_aver Rho_aver U_aver 0 0.001
```

5. `mgard_ge_cpu`: MGARD low-level API test.

```bash
./mgard_ge_cpu ../../dataset/ sol_4114800_aver.bp 3 P_aver Rho_aver U_aver 0 0.001
```

6. `mgardPlug_adios_ge`: remesh plugin-based compression (MPI).

```bash
mpirun -np 4 ./mgardPlug_adios_ge ../../dataset/ sol_4114800_aver.bp 2 U_aver P_aver 1e-3 16
```

7. `mgardPlug_adios_decompress`: remesh plugin-based decompression (MPI).

```bash
mpirun -np 4 ./mgardPlug_adios_decompress ./ sol_4114800_aver.bp.compressed 2 P_aver U_aver 16
```
