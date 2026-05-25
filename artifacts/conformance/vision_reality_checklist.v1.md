# Vision vs Reality Checklist — 2026-05-24

Agent-verified audit of README/FEATURE_PARITY claims against actual code.

## Vision Checklist

| Goal | Status | Evidence |
|------|--------|----------|
| **157 canonical primitives** | WORKING | fj-core Primitive enum: 157 variants, 152 eval+AD, 5 pmap fail-closed |
| **Reverse-mode AD (VJP)** | WORKING | fj-ad: 120+ primitives with real VJP rules, 292 tests |
| **Forward-mode AD (JVP)** | WORKING | fj-ad: jvp() + jvp_rule() for all differentiable primitives |
| **jit transform** | WORKING | fj-api: 15+ tests, transparent pass-through |
| **grad transform** | WORKING | fj-api: 20+ tests, tape-based reverse-mode |
| **vmap transform** | WORKING | fj-dispatch: BatchTrace with 61 batching rules, not just loops |
| **Transform composition** | WORKING | jit(grad), vmap(grad), grad(grad) all tested |
| **value_and_grad** | WORKING | fj-api: 6+ tests, shared forward pass |
| **jacobian/hessian** | WORKING | fj-api: tests for single/multi input/output |
| **checkpoint (remat)** | WORKING | fj-api: rematerialization for memory efficiency |
| **custom_vjp/custom_jvp** | WORKING | fj-api: 7+ tests, registry-based |
| **Linear algebra (Chol/QR/SVD/Eigh)** | WORKING | fj-lax/linalg.rs: 55 tests, complex support |
| **FFT (Fft/Ifft/Rfft/Irfft)** | WORKING | fj-lax/fft.rs: 29 tests, radix-2 fast path |
| **Control flow (cond/scan/while/switch)** | WORKING | fj-lax + fj-dispatch: AD through control flow |
| **ThreeFry RNG** | WORKING | fj-lax/threefry.rs: 34 tests, JAX-matched |
| **E-graph optimizer** | WORKING | fj-egraph: 87 rules, 48 tests |
| **861 oracle fixtures** | WORKING | fj-conformance: fixtures captured from JAX 0.9.2 |
| **Strict/Hardened modes** | WORKING | fj-cache: mode split with tests |
| **RaptorQ durability** | WORKING | fj-conformance: sidecar/scrub/proof pipeline |
| **make_jaxpr tracing** | WORKING | fj-trace: 69 tests, nested contexts |
| **11 DTypes** | WORKING | fj-core: BF16-Complex128, promotion rules |
| **pmap collectives** | EXCLUDED_SCOPE | Explicit V1 exclusion: requires multi-device backend infrastructure (GPU/TPU mesh). Returns typed UnsupportedFeature error. |
| **Python bindings** | PARTIAL | fj-py: alpha PyO3, smoke tests only |

## Summary

- **WORKING**: 22/24 claimed features fully implemented with tests
- **PARTIAL**: 1 (Python bindings - alpha, needs expansion)
- **EXCLUDED_SCOPE**: 1 (pmap collectives - V1 exclusion, requires multi-device backend)
- **NOT_STARTED**: 0
- **REGRESSED**: 0

## Evidence Sources

- Agent audit of fj-api/src/lib.rs: 126 tests, all transforms working
- Agent audit of fj-ad/src/lib.rs: 292 tests, real tape-based AD
- Agent audit of fj-dispatch/src/lib.rs: real batching rules, composition works
- Agent audit of fj-lax/src/lib.rs: 152/157 primitives implemented
- Prior session: 7,407 workspace tests passing via rch

## Gaps Requiring Beads

1. **fj-py expansion** - beyond smoke programs to user-defined tracing (tracked in FEATURE_PARITY)
2. **pmap/multi-device** - requires backend infrastructure (out of V1 scope)

No new beads required — existing tracking in FEATURE_PARITY.md is accurate.
