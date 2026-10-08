# engine/backend

Host-only code (core, runtime, models, loader, server and the model .cpp files) names no vendor API.

Rules
- Include only `engine/backend/*.h` for device or vendor concepts. No `<cuda_*.h>`, `<cublas*>`, `cuda*` calls, `dim3`, or `<<<>>>` in host-only files.
- Types: use the aliases from `types.h`: `backend::stream_t`, `backend::half_t`, `backend::bf16_t`, `backend::graph_t`, `backend::graph_exec_t`. Never spell `half`, `__half`, `__nv_bfloat16`, `cudaStream_t`.
- Calls: use the inline forwards in `runtime.h` (device, stream, memcpy/memset, pinned memory, graph capture and replay). Copy direction is `backend::Copy`, pinned flavor is `backend::Pin`.
- Scalar conversions on the host (weight conversion in loaders): `numeric.h` (`to_float`, `from_float_half`, `from_float_bf16`), round to nearest even.
- Kernel launches stay in modules/spec/comm (`src/modules`, `src/spec`, `src/kernels`), never in host-only files.

Backend selection
- `ENGINE_BACKEND_CUDA` is the default. Aliases map to the native types (no marshalling); every function is a one-line inline forward, so generated code equals calling `cuda*` directly. No virtual calls.
- Capability tiers per module entry point: `tiers.h` and `tier_table.inc`.

Adding a backend function
1. Add the inline forward to `runtime.h` (or the conversion to `numeric.h`, the type alias to `types.h`) inside the `ENGINE_BACKEND_CUDA` block, using `ENGINE_CHECK` for the aborting form and a `_try` variant when the caller reports failure.
2. Name it after the operation, not the vendor call (`memcpy_async`, not `cudaMemcpyAsync`).
3. A new backend provides the same names in its own block; call sites do not change.
