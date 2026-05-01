# Cache Legacy Parity Ledger

- schema: `frankenjax.cache-legacy-parity-ledger.v1`
- key namespace: `fjx`

| Row | Surface | Status | Legacy Anchor | Rust Behavior |
|-----|---------|--------|---------------|---------------|
| `cache-key-hash-inputs` | `CacheKey` | `ModeledWithScopeDifference` | `P2C005-A01,P2C005-A02` | FrankenJAX hashes canonical Jaxpr structure and all Rust-side execution configuration with SHA-256. |
| `compile-options-sorted` | `CompilerOptions` | `Modeled` | `P2C005-A10` | Rust compile options are sorted by BTreeMap key order before hashing. |
| `backend-identity` | `BackendIdentity` | `ModeledWithScopeDifference` | `P2C005-A11,P2C005-A16` | V1 Rust cache keys include backend identity. Current runtime scope is CPU, while non-CPU strings still separate keys. |
| `transform-stack-order` | `TransformStack` | `Modeled` | `P2C005-A12,P2C005-A13` | Rust cache keys preserve exact transform order; grad,vmap and vmap,grad hash differently. |
| `namespace-version-material` | `VersioningInputs` | `ModeledWithScopeDifference` | `P2C005-A03,P2C005-A18` | Rust uses the fjx namespace as explicit coarse version material for all cache keys. |
| `unknown-metadata-policy` | `UnknownMetadata` | `Modeled` | `P2C005-A01,P2C005-A10` | Rust makes unknown incompatible feature material explicit at the API boundary. |
| `custom-cache-hook-material` | `CacheHooks` | `ModeledWithScopeDifference` | `P2C005-A12,P2C005-A15` | Rust exposes custom_hook as explicit key material instead of implicit function identity. |
| `compilation-cache-metadata` | `CompilationCacheMetadata` | `ModeledWithScopeDifference` | `P2C005-A04,P2C005-A06,P2C005-A13` | Rust V1 has cache-manager primitives and deterministic keys, but dispatch still interprets directly instead of compiling XLA executables. |
| `corrupt-read-bypass` | `CorruptReads` | `Modeled` | `P2C005-A05,P2C005-A07` | Rust file cache embeds payload digests and surfaces corrupt reads as CacheLookup::Corrupted. |
| `stale-write-blocking` | `StaleWrites` | `Modeled` | `P2C005-A04,P2C005-A07` | Rust file cache writes serialized artifacts through a temporary file and rename; failed writes do not create readable hits. |
| `hostile-key-material` | `HostileKeyMaterial` | `Modeled` | `P2C005-A02` | Rust hashes delimiter-bearing strings as bytes in their field positions; lifecycle tests prove hostile material does not alias clean keys. |
| `gcs-cache-exclusion` | `CompilationCacheMetadata` | `ExplicitExclusion` | `P2C005-A08` | FrankenJAX V1 intentionally excludes cloud cache backends; local file and in-memory cache primitives cover current scope. |
