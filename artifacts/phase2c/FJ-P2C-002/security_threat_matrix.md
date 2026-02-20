# FJ-P2C-002 Security + Compatibility Threat Matrix (bd-3dl.13.3)

## Scope

Packet boundary: API transform front-door (jit/grad/vmap dispatch).

Primary subsystems:
- `fj-dispatch` (dispatch orchestration, transform execution)
- `fj-core` (Transform enum, composition proof, TraceTransformLedger)
- `fj-cache` (cache key generation, strict/hardened policy)
- `fj-ad` (reverse-mode AD, VJP rules)
- `fj-ledger` (evidence recording)

## Threat Matrix

| Threat class | Attack vector | Strict mitigation | Hardened mitigation | Fail-closed boundary | Residual risk | Evidence |
|---|---|---|---|---|---|---|
| Argument injection via oversized tensor | Tensor with extremely large leading dimension (e.g. 2^31 elements) causing OOM during vmap slice-and-stack iteration | Leading dimension extracted from tensor shape metadata; iteration count bounded by allocated tensor size. Rust's allocator will fail with OOM before stack corruption. No `unsafe` code | Same as strict. Hardened mode does not relax memory bounds | OOM is a process-level termination, not undefined behavior; `#![forbid(unsafe_code)]` prevents buffer overflows | Medium: no explicit vmap iteration cap enforced; relying on system OOM killer | `crates/fj-dispatch/src/lib.rs:262-352` |
| Transform composition bomb | Pathological stacking (e.g. 1000 nested Jit transforms or alternating Grad/Vmap) attempting stack overflow or CPU exhaustion | `verify_transform_composition` enforces at most one Grad and one Vmap per stack; evidence cardinality check bounds stack depth to evidence count; recursive dispatch via `execute_with_transforms` terminates when stack is empty | Same as strict; composition proof is not relaxed in hardened mode | Composition proof failure is fail-closed (TransformCompositionError) before any execution begins | Low: Rust stack guard prevents overflow; composition proof limits practical stack depth to ~3 transforms | `crates/fj-core/src/lib.rs:947-997` |
| Static argnums out-of-bounds | static_argnums indices referencing non-existent argument positions, potentially causing panic or incorrect arg partitioning | Not yet exposed in FrankenJAX; all args are passed to Jaxpr directly. When implemented, bounds-checking will validate indices against args.len() before partitioning | Same as strict with audit logging of clamped indices | Out-of-bounds index will be checked before any tracing begins | Negligible for V1: feature not exposed. Future: standard Rust bounds-check prevents memory safety issues | `artifacts/phase2c/FJ-P2C-002/legacy_anchor_map.v1.json` |
| API surface fingerprinting via error messages | Probing error message text to distinguish FrankenJAX from JAX, enabling targeted attacks against implementation-specific weaknesses | Error messages are FrankenJAX-native (no JAX error text mimicry); boundary-tagged with subsystem prefix. Deterministic messages do not leak internal state beyond the error type | Same as strict; hardened mode does not add implementation details to errors | Error messages reveal subsystem boundary (cache/interpreter/transform) but not internal variable state or memory layout | Medium: error differentiation is inherent in any reimplementation; message text is not security-sensitive for most use cases | `crates/fj-dispatch/src/lib.rs:45-107` |
| Cache key collision | Adversary crafts two semantically different programs that produce identical cache keys, causing incorrect cached result reuse | SHA-256 hash of canonical payload (mode + backend + jaxpr fingerprint + transforms + compile_options + hook); collision probability 2^-128 | Same as strict; unknown features added to hash increase entropy | Cache key collision is cryptographically infeasible with SHA-256 | Negligible: SHA-256 collision resistance is well-established | `crates/fj-cache/src/lib.rs` |
| Cache poisoning via unknown features | Injecting crafted unknown_incompatible_features strings to influence cache key or bypass validation | Strict mode: fail-closed rejection before any execution; unknown features list must be empty | Hardened mode: unknown features included in hash; different features = different cache key; no execution bypass possible | Strict: CacheKeyError::UnknownIncompatibleFeatures. Hardened: features hashed but not interpreted | Low: unknown features affect key identity only, not execution semantics | `crates/fj-dispatch/src/lib.rs:539-578` |
| Finite-difference epsilon manipulation | For nested transforms using finite-diff fallback, adversarial function f where f(x+eps) and f(x-eps) produce wildly different results (e.g. discontinuous function), yielding meaningless gradients | Epsilon is hardcoded (1e-6), not user-configurable; finite-diff is a known approximation with documented limitations | Same as strict; no additional guard | Finite-diff fallback always returns a value (no NaN propagation guard currently); incorrect gradient is a semantic issue, not a safety issue | Medium: finite-diff is inherently fragile for non-smooth functions; users should prefer symbolic AD (innermost grad) when possible | `crates/fj-dispatch/src/lib.rs:231-260` |
| Grad reverse-AD tape memory exhaustion | Large Jaxpr with many equations causing the AD tape (forward_with_tape) to consume excessive memory during reverse-mode differentiation | Tape size is O(E) where E = equation count; bounded by Jaxpr size which is bounded by original program construction | Same as strict; no tape size relaxation | OOM is process-level; no undefined behavior | Medium: no explicit tape size cap; large programs may legitimately require large tapes | `crates/fj-ad/src/lib.rs` |
| Evidence ledger injection | Crafted DispatchRequest fields that produce misleading evidence ledger entries (e.g. fake eqn_count signal) | Signals are computed internally from the actual Jaxpr and transform stack, not from user-supplied metadata; decision_id is the cache key (internally computed) | Same as strict | Ledger entries are append-only; signals are derived from actual computation state | Negligible: signal values are computed server-side from verified program state | `crates/fj-dispatch/src/lib.rs:154-179` |
| Vmap output arity oracle | Querying vmap with functions that return different output counts per iteration to probe implementation behavior | VmapInconsistentOutputArity detected at iteration i>0 if output count differs from iteration 0; early termination on first mismatch | Same as strict | Mismatch detection is fail-closed with expected/actual arity in error | Low: reveals only that FrankenJAX validates output consistency (not sensitive) | `crates/fj-dispatch/src/lib.rs:331-337` |

## Compatibility Envelope

| JAX API | FrankenJAX status | Strict mode | Hardened mode | Evidence |
|---|---|---|---|---|
| `jit(f)` basic wrapping | SUPPORTED | guaranteed | guaranteed | `crates/fj-dispatch/src/lib.rs:198` |
| `jit(f, static_argnums=...)` | NOT YET EXPOSED | out-of-scope for V1 | out-of-scope for V1 | `artifacts/phase2c/FJ-P2C-002/legacy_anchor_map.v1.json` |
| `jit(f, donate_argnums=...)` | NOT in V1 scope | out-of-scope | out-of-scope | N/A |
| `grad(f)` scalar-to-scalar | SUPPORTED (argnums=0 only) | guaranteed | guaranteed | `crates/fj-dispatch/src/lib.rs:204-229` |
| `grad(f, argnums=(0,1,...))` | NOT YET (argnums=0 only) | guaranteed for argnums=0 | guaranteed for argnums=0 | `crates/fj-ad/src/lib.rs:271` |
| `grad(f, has_aux=True)` | NOT in V1 scope | out-of-scope | out-of-scope | N/A |
| `vmap(f)` leading-axis | SUPPORTED | guaranteed | guaranteed | `crates/fj-dispatch/src/lib.rs:262-352` |
| `vmap(f, in_axes=(...))` arbitrary axes | NOT in V1 (axis 0 only) | axis 0 guaranteed | axis 0 guaranteed | `artifacts/phase2c/FJ-P2C-002/legacy_anchor_map.v1.json` |
| `value_and_grad(f)` | COMPOSABLE (jit + grad) | composable via separate calls | composable via separate calls | `artifacts/phase2c/FJ-P2C-002/legacy_anchor_map.v1.json` |
| `jacfwd(f)` | NOT in V1 scope | out-of-scope | out-of-scope | N/A |
| `jacrev(f)` | COMPOSABLE (vmap + grad) | composable via transform stacking | composable via transform stacking | `artifacts/phase2c/FJ-P2C-002/legacy_anchor_map.v1.json` |
| `custom_vjp` / `custom_jvp` | NOT in V1 scope | out-of-scope | out-of-scope | N/A |
| `make_jaxpr(f)` | SUPPORTED via `build_program` + `build_closed_jaxpr` | guaranteed | guaranteed | `crates/fj-trace/src/lib.rs` |
| Pytree args (dict, list, namedtuple) | NOT in V1 (flat args only) | out-of-scope | out-of-scope | N/A |
| Transform stacking: `jit(grad(f))` | SUPPORTED | guaranteed | guaranteed | `crates/fj-dispatch/src/lib.rs:496-536` |
| Transform stacking: `vmap(grad(f))` | SUPPORTED (finite-diff) | guaranteed | guaranteed | `crates/fj-dispatch/src/lib.rs:496-536` |
| Transform stacking: `jit(vmap(f))` | SUPPORTED | guaranteed | guaranteed | `crates/fj-dispatch/src/lib.rs:188-201` |
| Nested grad: `grad(grad(f))` | LIMITED (finite-diff fallback) | available with reduced accuracy | available with reduced accuracy | `crates/fj-dispatch/src/lib.rs:220-224` |

## Explicit Fail-Closed Rules

1. Transform composition proof failure terminates the request before cache key generation or execution.
2. Unknown incompatible features in strict mode cause CacheKeyError before execution; no fallback.
3. Non-scalar gradient input/output terminates the grad transform with deterministic error message.
4. Vmap leading-dimension mismatches terminate before completing iteration 0 if detected, or at iteration i if output arity changes.
5. Empty argument lists for Grad or Vmap terminate before any computation.
6. Empty vmap output (zero-length leading dimension) terminates before iteration begins.
7. All error types are boundary-tagged; no raw panics escape the dispatch boundary due to `#![forbid(unsafe_code)]` and comprehensive Result<T, E> propagation.
8. Cache key generation is fail-closed for malformed inputs; SHA-256 hash ensures collision resistance.
9. Evidence ledger signals are computed from verified internal state, not from user-supplied metadata.
