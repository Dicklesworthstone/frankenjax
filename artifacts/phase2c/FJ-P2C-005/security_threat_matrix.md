# FJ-P2C-005 Security + Compatibility Threat Matrix (bd-3dl.16.3)

## Scope

Packet boundary: Compilation cache/keying subsystem — cache key generation, strict/hardened feature gating, canonical payload construction, streaming hasher, key namespace isolation.

Primary subsystems:
- `fj-cache` (cache key generation, canonical payload, strict/hardened policy, streaming SHA-256 hasher)
- `fj-core` (Jaxpr fingerprint, canonical_fingerprint via OnceLock)
- `fj-dispatch` (DispatchRequest → CacheKeyInputRef construction, cache key in evidence ledger)

## Threat Matrix

| Threat class | Attack vector | Strict mitigation | Hardened mitigation | Fail-closed boundary | Residual risk | Evidence |
|---|---|---|---|---|---|---|
| Cache key collision | Craft two semantically-different programs that produce the same SHA-256 cache key, causing wrong cached result to be served | SHA-256 over typed, length-framed key material includes payload version, mode, backend, jaxpr fingerprint, transforms, compile_options, custom_hook, and unknown_features. User-controlled strings cannot shift field boundaries. Collision probability < 2^-128 | Same as strict; hardened adds unknown_features to hash, with the list length and each value length-framed | CacheKeyError if strict mode sees unknown incompatible features | Negligible: SHA-256 preimage resistance is well-established; delimiter aliasing is regression-tested | `crates/fj-cache/src/lib.rs::hash_canonical_payload_ref`, `crates/fj-cache/src/lib.rs::key_sensitivity_compile_option_delimiter_collision_regression` |
| Cache poisoning | Write malicious compiled artifact for a valid cache key, causing arbitrary code execution on cache read | FrankenJAX V1 does not persist cached artifacts (compute-only). No file cache or GCS backend. Cache key exists only for evidence ledger decision_id and determinism verification | Same as strict; no persistence means no poisoning vector | No persistence = no attack surface for cache poisoning | Negligible for V1: no cache storage. Future file cache must validate artifact integrity via SHA-256 on read | Legacy anchor P2C005-A04, P2C005-A07 |
| Cache timing side-channel | Cache hit/miss timing reveals computation identity (what programs are being compiled) | V1 does not cache compiled artifacts, so no timing difference between hit/miss. All dispatches execute fresh computation | Same as strict | No timing oracle exists in V1 | Negligible for V1. Future cache backends must consider constant-time lookup or timing jitter | Legacy anchor P2C005-A15 |
| Stale artifact serving | Cached result from old FrankenJAX version served to new version with different semantics | V1 uses the `fjx-v2-` namespace prefix plus payload-version key material for coarse version separation. No persistence means no stale artifacts | Same as strict | No persistence = no stale artifact risk in V1 | Low for V1 (no persistence). Medium for future: crate/version hash should be added to key material for cross-version invalidation | Legacy anchor P2C005-A18, P2C005-A16 |
| Cache exhaustion DoS | Flood cache with unique keys to fill storage, causing disk exhaustion or GCS cost explosion | V1 has no persistent cache. Cache keys are ephemeral per-dispatch. No storage to exhaust | Same as strict | No storage = no exhaustion vector | Negligible for V1. Future file cache should implement LRU eviction with configurable size cap | Legacy anchor P2C005-A07, P2C005-A17 |
| Canonical payload delimiter injection | Inject `\|`, `,`, `;`, `=`, or sentinel words into field values to shift field boundaries and create collisions | The cache-key payload is typed and length-framed: option presence, list cardinality, map cardinality, and every user-controlled string length are hashed before bytes | Same as strict; hardened unknown-feature lists are length-framed rather than comma-joined | `hash_canonical_payload_ref` is the single encoder used by owned and borrowed builders | Low: only SHA-256 collision risk remains; delimiter boundary confusion is covered by unit and conformance regressions | `crates/fj-cache/src/lib.rs::key_sensitivity_compile_option_delimiter_collision_regression`, `crates/fj-conformance/tests/cache_keying_oracle.rs::adversarial_compile_options_are_length_framed` |
| Unknown feature enumeration | Probe strict mode with different feature strings to enumerate which features are known vs unknown | Strict mode rejects all unknown features uniformly with CacheKeyError::UnknownIncompatibleFeatures. Error message includes the rejected feature list | Same behavior: rejection is uniform regardless of feature content | Uniform rejection prevents feature enumeration via error differentiation | Negligible: error message reveals which features were submitted (by design) but not which features are known | `crates/fj-cache/src/lib.rs:53-57` |
| Jaxpr fingerprint manipulation | Craft a Jaxpr that produces a specific canonical_fingerprint to match an existing cache key | canonical_fingerprint() uses deterministic formatting of inputs, outputs, consts, equations. SHA-256-like FNV hash would need second preimage. OnceLock ensures computation happens once | Same as strict | Fingerprint is FNV-1a (not SHA-256), but combined with SHA-256 in cache key provides adequate collision resistance | Low: FNV-1a in canonical_fingerprint is weaker than SHA-256, but it feeds into SHA-256 as part of the cache key. Standalone fingerprint collision is possible but cache key collision requires matching all other fields too | `crates/fj-core/src/lib.rs` (canonical_fingerprint) |

## Compatibility Envelope

| JAX cache behavior | FrankenJAX status | Strict mode | Hardened mode | Evidence |
|---|---|---|---|---|
| `cache_key.get()` → SHA-256 over HLO module + flags | SUPPORTED: SHA-256 over canonical payload (mode\|backend\|transforms\|compile\|hook\|unknown\|jaxpr) | guaranteed | guaranteed | `crates/fj-cache/src/lib.rs`, anchor P2C005-A01, P2C005-A02 |
| `CacheKey` hex digest format | DIVERGENT: FrankenJAX uses `fjx-v2-{hex}` prefix; JAX uses raw hex | guaranteed (different format, same semantics) | Same as strict | anchor P2C005-A03 |
| `_hash_xla_flags` compiler configuration | DIVERGENT: compile_options BTreeMap replaces XLA flags | Semantically equivalent: configuration affects key | Same as strict | anchor P2C005-A10 |
| `_hash_devices` device assignment | SUPPORTED via 'backend' field (currently 'cpu' only) | Single-backend only in V1 | Same as strict | anchor P2C005-A11 |
| `_hash_platform` OS/arch fingerprint | NOT IMPLEMENTED: no platform fingerprint in cache key | Out-of-scope for V1 | Same as strict | anchor P2C005-A16 |
| `_version_hash` JAX version in key | NOT IMPLEMENTED: namespace prefix provides coarse separation | Future work: add version string to canonical payload | Same as strict | anchor P2C005-A18 |
| `FileCache` local filesystem backend | NOT IMPLEMENTED: compute-only cache key | Future work: file-backed cache | Same as strict | anchor P2C005-A07 |
| `GcsCache` Google Cloud Storage backend | NOT IN SCOPE: JAX-specific infrastructure | N/A | N/A | anchor P2C005-A08 |
| `CacheInterface` abstract backend | NOT IMPLEMENTED: no trait defined yet | Future work: trait CacheBackend | Same as strict | anchor P2C005-A14 |
| `_compile_and_write_cache` cache write-through | NOT APPLICABLE: no compilation or persistence | N/A for V1 | N/A for V1 | anchor P2C005-A04 |
| `_read_from_cache` cache read | NOT APPLICABLE: no persistence | N/A for V1 | N/A for V1 | anchor P2C005-A05 |
| `linear_util.cache` WeakKeyDictionary memoization | DIVERGENT: SHA-256 content-addressed keys instead of identity-based | Equivalent caching semantics | Same as strict | anchor P2C005-A12 |
| `get_executable` two-tier cache (memory + persistent) | NOT IMPLEMENTED: single-tier ephemeral | Future work: two-tier cache | Same as strict | anchor P2C005-A13 |
| `_cache_hit_logging` telemetry | SUPPORTED via evidence ledger signals | Evidence signals capture dispatch telemetry | Same as strict | anchor P2C005-A15 |
| Cache eviction policy | NOT IMPLEMENTED: JAX also has no built-in eviction | N/A for V1 | N/A for V1 | anchor P2C005-A17 |
| Cross-version cache invalidation | NOT IMPLEMENTED: namespace prefix only | Future work: version hash in key | Same as strict | anchor P2C005-A18 |

## Explicit Fail-Closed Rules

1. Strict mode rejects unknown incompatible features with `CacheKeyError::UnknownIncompatibleFeatures` before any SHA-256 computation.
2. Hardened mode includes unknown features in hash (auditable progress) but does not reject.
3. Cache key generation is deterministic: same inputs always produce same key.
4. Canonical payload fields are separated by `|` to prevent cross-field confusion.
5. Transform stack ordering is preserved in canonical payload (comma-separated, in stack order).
6. Compile options are BTreeMap-ordered (deterministic key iteration).
7. Custom hook defaults to "none" when absent, ensuring consistent payload.
8. Jaxpr fingerprint is computed once via OnceLock (cached, deterministic).
9. Cache key namespace 'fjx-' prevents accidental collision with other key schemes.
10. `#![forbid(unsafe_code)]` on all crates prevents memory safety violations in cache key construction.
