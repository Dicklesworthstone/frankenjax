# Architecture Boundary Decision

- schema: `frankenjax.architecture-boundary-report.v1`
- bead: `frankenjax-cstq.12`
- status: `pass`
- crates: `15`
- normal edges: `36`

| Boundary | Decision | Owners | User outcome |
|---|---|---|---|
| `user_api_facade` | `KeepCurrentBoundary` | `fj-api, fj-trace` | Users get one explicit Rust transform facade now; no extra compatibility wrapper or package rename is introduced in this bead. |
| `transform_stack` | `DeferExtraction` | `fj-core, fj-dispatch, fj-ad, fj-trace, fj-api` | Users keep current transform behavior while parity work closes advanced control-flow composition gaps. |
| `lowering_execution` | `DeferExtraction` | `fj-interpreters, fj-dispatch, fj-backend-cpu, fj-runtime` | Users keep the documented CPU-only interpreter/backend semantics without an invented XLA-like lowering layer. |
| `cpu_backend` | `KeepCurrentBoundary` | `fj-backend-cpu, fj-runtime` | Users get a concrete always-available CPU backend without waiting for GPU/TPU abstractions. |
| `ffi_boundary` | `KeepCurrentBoundary` | `fj-ffi` | Users get one auditable native interop boundary and the rest of the workspace stays unsafe-free. |
| `conformance_harness` | `KeepCurrentBoundary` | `fj-conformance` | Users and agents get one evidence plane for replaying claims without production crates depending on the harness. |

## Issues

No architecture boundary issues found.
