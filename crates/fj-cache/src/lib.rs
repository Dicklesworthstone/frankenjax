#![forbid(unsafe_code)]

pub mod backend;
pub mod eviction;
pub mod persistence;
pub mod stability;

use fj_core::{CompatibilityMode, Jaxpr, Transform};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKeyInput {
    pub mode: CompatibilityMode,
    pub backend: String,
    pub jaxpr: Jaxpr,
    pub transform_stack: Vec<Transform>,
    pub compile_options: BTreeMap<String, String>,
    pub custom_hook: Option<String>,
    pub unknown_incompatible_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    pub namespace: &'static str,
    pub digest_hex: String,
}

impl CacheKey {
    #[must_use]
    pub fn as_string(&self) -> String {
        format!("{}-{}", self.namespace, self.digest_hex)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheKeyError {
    UnknownIncompatibleFeatures { features: Vec<String> },
}

impl std::fmt::Display for CacheKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownIncompatibleFeatures { features } => {
                write!(
                    f,
                    "strict mode rejected unknown incompatible features: {}",
                    features.join(",")
                )
            }
        }
    }
}

impl std::error::Error for CacheKeyError {}

pub fn build_cache_key(input: &CacheKeyInput) -> Result<CacheKey, CacheKeyError> {
    if input.mode == CompatibilityMode::Strict && !input.unknown_incompatible_features.is_empty() {
        return Err(CacheKeyError::UnknownIncompatibleFeatures {
            features: input.unknown_incompatible_features.clone(),
        });
    }

    let payload = canonical_payload(input);
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    let digest = hasher.finalize();

    Ok(CacheKey {
        namespace: "fjx",
        digest_hex: bytes_to_hex(&digest),
    })
}

#[must_use]
pub fn compatibility_matrix_row(input: &CacheKeyInput) -> String {
    format!(
        "mode={:?},backend={},unknown_features={}",
        input.mode,
        input.backend,
        input.unknown_incompatible_features.join(";")
    )
}

pub struct CacheKeyInputRef<'a> {
    pub mode: CompatibilityMode,
    pub backend: &'a str,
    pub jaxpr: &'a Jaxpr,
    pub transform_stack: &'a [Transform],
    pub compile_options: &'a BTreeMap<String, String>,
    pub custom_hook: Option<&'a str>,
    pub unknown_incompatible_features: &'a [String],
}

pub fn build_cache_key_ref(input: &CacheKeyInputRef<'_>) -> Result<CacheKey, CacheKeyError> {
    if input.mode == CompatibilityMode::Strict && !input.unknown_incompatible_features.is_empty() {
        return Err(CacheKeyError::UnknownIncompatibleFeatures {
            features: input.unknown_incompatible_features.to_vec(),
        });
    }

    // Stream canonical payload directly into the hasher to avoid allocating
    // an intermediate String. Each field is written with a separator prefix
    // to match the original canonical_payload_ref layout for compatibility.
    let mut hasher = Sha256::new();
    hash_canonical_payload_ref(&mut hasher, input);
    let digest = hasher.finalize();

    Ok(CacheKey {
        namespace: "fjx",
        digest_hex: bytes_to_hex(&digest),
    })
}

/// Write the canonical payload directly into a Digest, avoiding intermediate
/// String allocation. Layout matches `canonical_payload_ref` exactly.
fn hash_canonical_payload_ref(hasher: &mut Sha256, input: &CacheKeyInputRef<'_>) {
    use std::fmt::Write;

    // mode=<mode>|backend=<backend>|transforms=<t1,t2,...>|compile=<k1=v1;k2=v2>|hook=<hook>|unknown=<u1,u2>|jaxpr=<fp>
    let mut buf = String::new();
    let _ = write!(
        &mut buf,
        "mode={:?}|backend={}|transforms=",
        input.mode, input.backend
    );
    hasher.update(buf.as_bytes());

    for (i, t) in input.transform_stack.iter().enumerate() {
        if i > 0 {
            hasher.update(b",");
        }
        hasher.update(t.as_str().as_bytes());
    }

    hasher.update(b"|compile=");
    for (i, (key, value)) in input.compile_options.iter().enumerate() {
        if i > 0 {
            hasher.update(b";");
        }
        hasher.update(key.as_bytes());
        hasher.update(b"=");
        hasher.update(value.as_bytes());
    }

    hasher.update(b"|hook=");
    hasher.update(input.custom_hook.unwrap_or("none").as_bytes());

    hasher.update(b"|unknown=");
    for (i, feature) in input.unknown_incompatible_features.iter().enumerate() {
        if i > 0 {
            hasher.update(b",");
        }
        hasher.update(feature.as_bytes());
    }

    hasher.update(b"|jaxpr=");
    hasher.update(input.jaxpr.canonical_fingerprint().as_bytes());
}

fn canonical_payload(input: &CacheKeyInput) -> String {
    let transforms = input
        .transform_stack
        .iter()
        .map(|transform| transform.as_str())
        .collect::<Vec<_>>()
        .join(",");

    let compile_options = input
        .compile_options
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(";");

    let unknown = input.unknown_incompatible_features.join(",");

    format!(
        "mode={:?}|backend={}|transforms={}|compile={}|hook={}|unknown={}|jaxpr={}",
        input.mode,
        input.backend,
        transforms,
        compile_options,
        input.custom_hook.as_deref().unwrap_or("none"),
        unknown,
        input.jaxpr.canonical_fingerprint(),
    )
}

// ── Cache Manager ───────────────────────────────────────────────────

/// Result of a cache lookup operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheLookup {
    /// Cache hit — artifact data found and integrity verified.
    Hit { data: Vec<u8> },
    /// Cache miss — no entry for this key.
    Miss,
    /// Cache corruption — entry existed but failed integrity check (evicted).
    Corrupted { key: String },
}

/// Unified cache manager combining key generation, backend storage, and
/// integrity verification. Provides the single entry point for cache
/// operations in the dispatch pipeline.
pub struct CacheManager {
    backend: Box<dyn backend::CacheBackend>,
}

impl CacheManager {
    /// Create a cache manager wrapping any `CacheBackend` implementation.
    pub fn new(backend: Box<dyn backend::CacheBackend>) -> Self {
        Self { backend }
    }

    /// Create an in-memory cache manager (ephemeral, no persistence).
    #[must_use]
    pub fn in_memory() -> Self {
        Self::new(Box::new(backend::InMemoryCache::new()))
    }

    /// Create a file-backed cache manager at the given directory.
    #[must_use]
    pub fn file_backed(cache_dir: std::path::PathBuf) -> Self {
        Self::new(Box::new(backend::FileCache::new(cache_dir)))
    }

    /// Create a cache manager with LRU eviction wrapping an in-memory backend.
    #[must_use]
    pub fn in_memory_with_eviction(config: eviction::LruConfig) -> Self {
        Self::new(Box::new(eviction::LruCache::new(
            backend::InMemoryCache::new(),
            config,
        )))
    }

    /// Look up cached data by key. Verifies integrity on file-backed caches.
    pub fn get(&self, key: &CacheKey) -> CacheLookup {
        match self.backend.get(key) {
            Some(artifact) => {
                // Verify integrity.
                let actual_hex = sha256_hex(&artifact.data);
                if actual_hex == artifact.integrity_sha256_hex {
                    CacheLookup::Hit {
                        data: artifact.data,
                    }
                } else {
                    CacheLookup::Corrupted {
                        key: key.as_string(),
                    }
                }
            }
            None => CacheLookup::Miss,
        }
    }

    /// Store data under the given key with automatic integrity tagging.
    pub fn put(&mut self, key: &CacheKey, data: Vec<u8>) {
        let integrity_sha256_hex = sha256_hex(&data);
        self.backend.put(
            key,
            backend::CachedArtifact {
                data,
                integrity_sha256_hex,
            },
        );
    }

    /// Remove a single cached entry.
    pub fn evict(&mut self, key: &CacheKey) -> bool {
        self.backend.evict(key)
    }

    /// Return current cache statistics.
    pub fn stats(&self) -> backend::CacheStats {
        self.backend.stats()
    }

    /// Remove all cached entries.
    pub fn clear(&mut self) {
        self.backend.clear();
    }
}

pub(crate) fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX_LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX_LUT[(byte >> 4) as usize] as char);
        out.push(HEX_LUT[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Compute SHA-256 hex digest of arbitrary bytes (used by submodules).
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    bytes_to_hex(&sha256_bytes(data))
}

/// Compute raw SHA-256 digest bytes.
pub(crate) fn sha256_bytes(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

#[cfg(test)]
mod tests {
    use super::{CacheKeyError, CacheKeyInput, build_cache_key};
    use fj_core::{CompatibilityMode, Jaxpr, Transform};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    fn empty_jaxpr() -> Jaxpr {
        Jaxpr::new(vec![], vec![], vec![], vec![])
    }

    #[test]
    fn strict_mode_rejects_unknown_features() {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["mystery_field".to_owned()],
        };

        let err = build_cache_key(&input).expect_err("should reject unknown features");
        assert_eq!(
            err,
            CacheKeyError::UnknownIncompatibleFeatures {
                features: vec!["mystery_field".to_owned()],
            }
        );
    }

    #[test]
    fn hardened_mode_accepts_unknown_features() {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Hardened,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["mystery_field".to_owned()],
        };

        let key = build_cache_key(&input).expect("hardened mode should hash unknown features");
        assert!(key.as_string().starts_with("fjx-"));
    }

    #[test]
    fn cache_key_is_stable_for_identical_inputs() {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: Some("hook-a".to_owned()),
            unknown_incompatible_features: vec![],
        };

        let key_a = build_cache_key(&input).expect("key generation should succeed");
        let key_b = build_cache_key(&input).expect("key generation should succeed");
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn test_cache_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("cpu", "strict")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_cache_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    fn arb_transform() -> impl Strategy<Value = Transform> {
        prop_oneof![
            Just(Transform::Jit),
            Just(Transform::Grad),
            Just(Transform::Vmap),
        ]
    }

    #[test]
    fn streaming_hash_matches_owned_hash() {
        use super::{CacheKeyInputRef, build_cache_key_ref};
        use fj_core::ProgramSpec;

        // Test with a real program and transforms
        let jaxpr = fj_core::build_program(ProgramSpec::SquarePlusLinear);
        let transforms = vec![Transform::Jit, Transform::Grad];
        let compile_options = BTreeMap::new();
        let unknown: Vec<String> = vec![];

        let owned_input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: jaxpr.clone(),
            transform_stack: transforms.clone(),
            compile_options: compile_options.clone(),
            custom_hook: Some("test-hook".to_owned()),
            unknown_incompatible_features: unknown.clone(),
        };

        let ref_input = CacheKeyInputRef {
            mode: CompatibilityMode::Strict,
            backend: "cpu",
            jaxpr: &jaxpr,
            transform_stack: &transforms,
            compile_options: &compile_options,
            custom_hook: Some("test-hook"),
            unknown_incompatible_features: &unknown,
        };

        let owned_key = build_cache_key(&owned_input).expect("owned key should succeed");
        let ref_key = build_cache_key_ref(&ref_input).expect("ref key should succeed");
        assert_eq!(
            owned_key, ref_key,
            "streaming (ref) and owned cache key must produce identical hashes"
        );
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(
            fj_test_utils::property_test_case_count()
        ))]
        #[test]
        fn prop_cache_key_stability(
            backend in "[a-z]{3,6}",
            transforms in proptest::collection::vec(arb_transform(), 0..3),
        ) {
            let _seed = fj_test_utils::capture_proptest_seed();
            let input = CacheKeyInput {
                mode: CompatibilityMode::Hardened,
                backend,
                jaxpr: empty_jaxpr(),
                transform_stack: transforms,
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            };
            let key_a = build_cache_key(&input).expect("key gen");
            let key_b = build_cache_key(&input).expect("key gen");
            prop_assert_eq!(key_a, key_b);
        }
    }

    // ── CacheManager integration tests ─────────────────────────────

    #[test]
    fn cache_manager_in_memory_hit_miss_cycle() {
        use super::{CacheLookup, CacheManager};

        let mut mgr = CacheManager::in_memory();
        let key = build_cache_key(&CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .unwrap();

        // Miss on first lookup.
        assert_eq!(mgr.get(&key), CacheLookup::Miss);

        // Store and hit.
        mgr.put(&key, b"cached_result".to_vec());
        assert_eq!(
            mgr.get(&key),
            CacheLookup::Hit {
                data: b"cached_result".to_vec()
            }
        );

        // Evict and miss again.
        assert!(mgr.evict(&key));
        assert_eq!(mgr.get(&key), CacheLookup::Miss);
    }

    #[test]
    fn cache_manager_file_backed_survives_reopen() {
        use super::{CacheLookup, CacheManager};

        let dir = std::env::temp_dir().join("fj-cache-mgr-persist-test");
        let _ = std::fs::create_dir_all(&dir);

        let key = build_cache_key(&CacheKeyInput {
            mode: CompatibilityMode::Hardened,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![Transform::Jit],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .unwrap();

        // Write with first manager instance.
        {
            let mut mgr = CacheManager::file_backed(dir.clone());
            mgr.put(&key, b"persistent_data".to_vec());
        }

        // Read with second manager instance (simulates process restart).
        {
            let mgr = CacheManager::file_backed(dir.clone());
            assert_eq!(
                mgr.get(&key),
                CacheLookup::Hit {
                    data: b"persistent_data".to_vec()
                }
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_manager_eviction_enforces_budget() {
        use super::{CacheLookup, CacheManager};
        use super::eviction::LruConfig;

        let config = LruConfig {
            max_entries: 2,
            max_bytes: 0,
        };
        let mut mgr = CacheManager::in_memory_with_eviction(config);

        let make_key = |s: &str| super::CacheKey {
            namespace: "fjx",
            digest_hex: s.to_owned(),
        };

        mgr.put(&make_key("aaa"), b"first".to_vec());
        mgr.put(&make_key("bbb"), b"second".to_vec());
        mgr.put(&make_key("ccc"), b"third".to_vec());

        // "aaa" should have been evicted (LRU, max 2 entries).
        assert_eq!(mgr.get(&make_key("aaa")), CacheLookup::Miss);
        assert_eq!(mgr.stats().entry_count, 2);
    }

    #[test]
    fn cache_manager_stats_reflect_contents() {
        use super::CacheManager;

        let mut mgr = CacheManager::in_memory();
        assert_eq!(mgr.stats().entry_count, 0);
        assert_eq!(mgr.stats().total_bytes, 0);

        let key = super::CacheKey {
            namespace: "fjx",
            digest_hex: "test123".to_owned(),
        };
        mgr.put(&key, vec![0u8; 100]);
        assert_eq!(mgr.stats().entry_count, 1);
        assert_eq!(mgr.stats().total_bytes, 100);

        mgr.clear();
        assert_eq!(mgr.stats().entry_count, 0);
    }
}
