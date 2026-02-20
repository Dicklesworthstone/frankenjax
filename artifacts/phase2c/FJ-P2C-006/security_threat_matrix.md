# FJ-P2C-006 Security + Compatibility Threat Matrix (bd-3dl.17.3)

## Scope

Packet boundary: Backend bridge and platform routing subsystem — backend discovery, device assignment, backend-specific lowering, memory management, platform routing, multi-device coordination.

Primary subsystems:
- `fj-runtime` (RuntimeAdmissionModel, asupersync/ftui bridges)
- `fj-dispatch` (DispatchRequest.backend string, dispatch → eval_jaxpr routing)
- `fj-cache` (CacheBackend trait, InMemoryCache, FileCache, LruCache)
- `fj-interpreters` (eval_jaxpr: CPU-only interpreter)
- `fj-core` (Value, TensorValue, Literal — host-resident data model)

## Threat Matrix

| Threat class | Attack vector | Strict mitigation | Hardened mitigation | Fail-closed boundary | Residual risk | Evidence |
|---|---|---|---|---|---|---|
| Device spoofing | Claim GPU/TPU capability when only CPU is present. Attacker sets backend="gpu" hoping to trigger GPU-specific code paths or bypass CPU-only validation | V1: backend string is opaque metadata. eval_jaxpr interprets identically regardless of backend value. No GPU-specific code paths exist. Backend string flows into cache key only | Same as strict; hardened mode would log unknown backend and fall back to CPU | Backend string does not alter execution semantics in V1 (no device dispatch) | Negligible for V1: no device-specific code. Medium for future GPU support: must validate backend against available devices before dispatch | `crates/fj-dispatch/src/lib.rs:206-271`, anchor P2C006-A03 |
| Buffer overflow via wrong memory layout | Supply tensor data with mismatched shape/dtype, causing out-of-bounds read in eval_jaxpr | TensorValue::new() validates element count matches shape (product of dims). DType and Shape are checked at construction. Vec<Literal> is bounds-checked by Rust runtime. `#![forbid(unsafe_code)]` prevents unchecked access | Same as strict; hardened adds no additional buffer protections (Rust memory safety is unconditional) | TensorValue construction rejects shape/data mismatches. Rust bounds checking prevents OOB access | Negligible: Rust's memory safety model and `forbid(unsafe_code)` eliminate buffer overflow class. No raw pointer arithmetic exists | `crates/fj-core/src/lib.rs` (TensorValue::new), anchor P2C006-A10 |
| Cross-device data leakage | Device A's buffer contents leak to device B during cross-device transfer, exposing sensitive computation results | V1: single-device only. All Values are host-resident Vec<Literal>. No cross-device transfer exists. Clone semantics produce independent copies | Same as strict; no cross-device path exists | Single-device architecture eliminates cross-device leakage vector | Negligible for V1. Future multi-device: must zero buffers after transfer or use device-isolated memory pools | anchor P2C006-A23, P2C006-A13, P2C006-A14 |
| Resource exhaustion via device memory | Allocate maximally-sized tensors to exhaust device memory, causing OOM panic or denial of service | V1: host-memory only. Rust's global allocator handles OOM (abort by default, no panic unwinding). Vec allocation is bounded by available system RAM. No GPU memory pool to exhaust | Hardened mode: returns DispatchError for allocation failures instead of abort. Partial allocations released via Rust Drop semantics | Rust allocator OOM → abort (not exploitable panic). No device memory pool to fragment | Low: host OOM causes process abort (not exploitable). Future GPU backend should implement allocation budgets per-dispatch with configurable limits | anchor P2C006-A25, P2C006-A18 |
| Backend priority manipulation | Manipulate discovery to force execution on a slower/less-secure backend (e.g., forcing CPU when GPU available for timing attacks) | V1: no backend discovery. Backend is caller-specified string. No priority ordering. Caller has full control over backend selection by design | Hardened: fallback semantics are explicit and logged. No silent backend substitution | Caller explicitly chooses backend; no implicit priority can be manipulated | Negligible for V1: single backend. Low for future: backend discovery order must be deterministic and auditable | anchor P2C006-A02, P2C006-A04 |
| Platform routing bypass | Circumvent platform compatibility checks to run computation on unsupported hardware, producing incorrect results | V1: eval_jaxpr is platform-agnostic Rust code. No hardware-specific code generation. Computation correctness depends on Rust f64/i64 arithmetic, not hardware features | Same as strict; no hardware-specific code to bypass | Universal interpreter eliminates platform-specific correctness bugs | Negligible for V1. Medium for future compiled backends: must validate hardware capability before lowering to platform-specific IR | anchor P2C006-A07, P2C006-A21 |
| Backend plugin injection | Register malicious backend plugin that intercepts compilations to exfiltrate program structure or inject incorrect results | V1: no plugin system. No register_backend() equivalent. Backend string is inert metadata | Same as strict; no plugin mechanism exists | No extensibility surface for backend registration in V1 | Negligible for V1. High for future plugin system: must validate plugin signatures and sandbox plugin execution | anchor P2C006-A06 |
| Multi-device sharding oracle | Use pmap/sharding to observe timing differences between devices, creating a timing oracle for computation structure | V1: no multi-device support. No pmap, no sharding, no collective operations. Single-device sequential execution only | Same as strict; no multi-device code paths | Single-device architecture eliminates multi-device timing oracles | Negligible for V1. Low for future: sharding timing should be independent of data values | anchor P2C006-A15, P2C006-A24 |

## Compatibility Envelope

| JAX backend behavior | FrankenJAX status | Strict mode | Hardened mode | Evidence |
|---|---|---|---|---|
| `backends()` → dict of available XLA backends | DIVERGENT: backend is a caller-supplied string, not discovered | Backend string accepted unconditionally | Same as strict | anchor P2C006-A01 |
| `_discover_backends()` → probe PJRT plugins | NOT IMPLEMENTED: no runtime backend probing | Out-of-scope for V1 | Same as strict | anchor P2C006-A02 |
| `get_backend(platform)` → resolve to Client | NOT IMPLEMENTED: no backend resolution | Backend string is opaque metadata | Hardened: unknown backend → CPU fallback | anchor P2C006-A03 |
| `default_backend()` → highest-priority available | NOT IMPLEMENTED: implicit CPU default | CPU is the only backend | Same as strict | anchor P2C006-A04 |
| `local_devices()` → Device list per backend | NOT IMPLEMENTED: no Device abstraction | Out-of-scope for V1 | Same as strict | anchor P2C006-A05 |
| `register_backend(name, factory)` → plugin registration | NOT IN SCOPE: no plugin system | N/A | N/A | anchor P2C006-A06 |
| `backend_specific_translations` → per-platform lowering rules | NOT APPLICABLE: single eval_primitive() path | All primitives use Rust native math | Same as strict | anchor P2C006-A07 |
| `lower_fun()` → Jaxpr to XLA HLO conversion | NOT APPLICABLE: interpreter mode, no HLO | eval_jaxpr interprets directly | Same as strict | anchor P2C006-A08 |
| `xla_call_p` → JIT compilation primitive | DIVERGENT: Transform::Jit is no-op pass-through | Jit skipped in execute_with_transforms | Same as strict | anchor P2C006-A09 |
| `Buffer` → device-resident array | DIVERGENT: Value is always host-resident | Vec<Literal> in Rust heap | Same as strict | anchor P2C006-A10 |
| `Executable` → compiled XLA program | NOT APPLICABLE: no compilation step | eval_jaxpr interprets Jaxpr directly | Same as strict | anchor P2C006-A11 |
| `Client` → platform-specific backend client | NOT IMPLEMENTED: no Client abstraction | Future: BackendClient trait | Same as strict | anchor P2C006-A12 |
| `device_put(value, device)` → host to device | SUPPORTED: trivial identity (Value already host-resident) | Value is host-resident by definition | Same as strict | anchor P2C006-A13 |
| `device_get(buffer)` → device to host | SUPPORTED: trivial identity (no device memory) | DispatchResponse.outputs already host-resident | Same as strict | anchor P2C006-A14 |
| `process_index()` → multi-process host ID | NOT IMPLEMENTED: single-process only | Implicit process_index=0 | Same as strict | anchor P2C006-A15 |
| `transfer_to_device()` → cross-device buffer copy | NOT APPLICABLE: single-device architecture | N/A for V1 | N/A for V1 | anchor P2C006-A23 |
| `memory_stats()` → backend memory utilization | NOT IMPLEMENTED: no device memory tracking | Future: CacheStats as proxy | Same as strict | anchor P2C006-A25 |
| `backend_xla_version()` → XLA version string | NOT APPLICABLE: no XLA dependency | Future: fjx_version() crate version | Same as strict | anchor P2C006-A20 |

## Explicit Fail-Closed Rules

1. Backend string in DispatchRequest does not alter execution semantics — eval_jaxpr is backend-agnostic.
2. TensorValue::new() rejects shape/data mismatches before any computation occurs.
3. `#![forbid(unsafe_code)]` on all crates prevents memory safety violations in the backend bridge.
4. No cross-device transfer code paths exist in V1 — Value is always host-resident.
5. No backend plugin registration exists in V1 — no extensibility attack surface.
6. RuntimeAdmissionModel enforces mode-based gating (strict vs hardened) at dispatch entry.
7. Cache key includes backend string — different backends produce different cache keys (cache isolation).
8. Single-device, single-process execution eliminates multi-device and multi-host attack surface.
9. Rust allocator OOM triggers abort (not exploitable panic) — no unwind-based attacks.
10. All dispatch paths return Result<DispatchResponse, DispatchError> — no panicking code paths in the happy path.
