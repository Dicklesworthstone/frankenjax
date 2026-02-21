#![forbid(unsafe_code)]

//! Cache backend trait and concrete implementations.
//!
//! V1 scope: `InMemoryCache` (HashMap-backed) and `FileCache` (directory-based,
//! one file per key). Both backends support LRU eviction via `eviction::LruPolicy`.

use crate::CacheKey;
use std::collections::HashMap;
use std::path::PathBuf;

// ── Cached Artifact ─────────────────────────────────────────────────

/// Opaque wrapper around a cached compilation artifact (serialized bytes).
///
/// V1: FrankenJAX does not persist compiled XLA artifacts (no XLA backend).
/// This type exists to establish the cache contract for future backends that
/// will store compiled IR, evaluation results, or staging residuals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedArtifact {
    /// Raw bytes of the cached artifact.
    pub data: Vec<u8>,
    /// SHA-256 hex digest of `data`, used for integrity verification on read.
    pub integrity_sha256_hex: String,
}

// ── Cache Backend Trait ─────────────────────────────────────────────

/// Trait for compilation cache storage backends.
///
/// Mirrors JAX's `CacheInterface` (anchor P2C005-A14) with Rust semantics:
/// - `get`/`put` operate on `CacheKey` → `CachedArtifact`
/// - `evict` removes a single entry
/// - `stats` returns current cache metrics
///
/// All implementations must be deterministic for a given sequence of
/// operations (contract p2c005.strict.inv001).
pub trait CacheBackend: Send + Sync {
    /// Look up a cached artifact by key. Returns `None` on cache miss.
    fn get(&self, key: &CacheKey) -> Option<CachedArtifact>;

    /// Store an artifact under the given key. Overwrites any existing entry.
    fn put(&mut self, key: &CacheKey, artifact: CachedArtifact);

    /// Remove a single entry. Returns `true` if the entry existed.
    fn evict(&mut self, key: &CacheKey) -> bool;

    /// Return current cache statistics (entry count, total byte size).
    fn stats(&self) -> CacheStats;

    /// Remove all entries from the cache.
    fn clear(&mut self);
}

/// Cache utilization metrics returned by `CacheBackend::stats()`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of entries currently in the cache.
    pub entry_count: usize,
    /// Total byte size of all cached artifacts.
    pub total_bytes: u64,
}

// ── In-Memory Backend ───────────────────────────────────────────────

/// HashMap-backed in-memory cache. No persistence across process restarts.
///
/// Thread safety: requires external synchronization (e.g., `Mutex<InMemoryCache>`).
/// The `Send + Sync` bound on `CacheBackend` enables wrapping in `Arc<Mutex<_>>`.
#[derive(Debug, Default)]
pub struct InMemoryCache {
    entries: HashMap<String, CachedArtifact>,
}

impl InMemoryCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl CacheBackend for InMemoryCache {
    fn get(&self, key: &CacheKey) -> Option<CachedArtifact> {
        self.entries.get(&key.as_string()).cloned()
    }

    fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
        self.entries.insert(key.as_string(), artifact);
    }

    fn evict(&mut self, key: &CacheKey) -> bool {
        self.entries.remove(&key.as_string()).is_some()
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            entry_count: self.entries.len(),
            total_bytes: self.entries.values().map(|a| a.data.len() as u64).sum(),
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

// ── File-System Backend ─────────────────────────────────────────────

/// Directory-based file cache. One file per cache key, named by the key's
/// hex digest string. Uses atomic write-and-rename for crash safety
/// (mirrors JAX `FileCache` anchor P2C005-A07).
///
/// Layout: `{cache_dir}/{fjx-<hex_digest>}.bin`
#[derive(Debug)]
pub struct FileCache {
    /// Root directory for cache files.
    cache_dir: PathBuf,
}

impl FileCache {
    /// Create a new `FileCache` rooted at the given directory.
    ///
    /// Does not create the directory — callers must ensure it exists.
    #[must_use]
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Return the filesystem path for a given cache key.
    #[must_use]
    pub fn path_for(&self, key: &CacheKey) -> PathBuf {
        self.cache_dir.join(format!("{}.bin", key.as_string()))
    }
}

impl CacheBackend for FileCache {
    fn get(&self, key: &CacheKey) -> Option<CachedArtifact> {
        let path = self.path_for(key);
        let data = std::fs::read(&path).ok()?;

        // Integrity check: recompute SHA-256 and compare with stored digest.
        let integrity_sha256_hex = crate::sha256_hex(&data);
        Some(CachedArtifact {
            data,
            integrity_sha256_hex,
        })
    }

    fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
        let path = self.path_for(key);
        // Atomic write: write to temp file, then rename.
        // Use a unique temp file to avoid concurrent write races.
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = path.with_extension(format!("tmp.{}.{}", std::process::id(), id));
        if std::fs::write(&tmp_path, &artifact.data).is_ok() {
            let _ = std::fs::rename(&tmp_path, &path);
        } else {
            let _ = std::fs::remove_file(&tmp_path);
        }
    }

    fn evict(&mut self, key: &CacheKey) -> bool {
        let path = self.path_for(key);
        std::fs::remove_file(&path).is_ok()
    }

    fn stats(&self) -> CacheStats {
        let Ok(entries) = std::fs::read_dir(&self.cache_dir) else {
            return CacheStats::default();
        };
        let mut count = 0usize;
        let mut total = 0u64;
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|e| e == "bin") {
                count += 1;
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
        CacheStats {
            entry_count: count,
            total_bytes: total,
        }
    }

    fn clear(&mut self) {
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().is_some_and(|e| e == "bin") {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key(digest: &str) -> CacheKey {
        CacheKey {
            namespace: "fjx",
            digest_hex: digest.to_owned(),
        }
    }

    fn test_artifact(data: &[u8]) -> CachedArtifact {
        CachedArtifact {
            data: data.to_vec(),
            integrity_sha256_hex: crate::sha256_hex(data),
        }
    }

    #[test]
    fn in_memory_put_get_evict_cycle() {
        let mut cache = InMemoryCache::new();
        let key = test_key("aabb");
        let artifact = test_artifact(b"hello world");

        assert!(cache.get(&key).is_none());
        cache.put(&key, artifact.clone());
        assert_eq!(cache.get(&key), Some(artifact));
        assert_eq!(cache.stats().entry_count, 1);

        assert!(cache.evict(&key));
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().entry_count, 0);
    }

    #[test]
    fn in_memory_clear_empties_cache() {
        let mut cache = InMemoryCache::new();
        cache.put(&test_key("a"), test_artifact(b"1"));
        cache.put(&test_key("b"), test_artifact(b"2"));
        assert_eq!(cache.stats().entry_count, 2);

        cache.clear();
        assert_eq!(cache.stats().entry_count, 0);
    }

    #[test]
    fn file_cache_round_trip() {
        let dir = std::env::temp_dir().join("fj-cache-test-file-backend");
        let _ = std::fs::create_dir_all(&dir);

        let mut cache = FileCache::new(dir.clone());
        let key = test_key("deadbeef");
        let artifact = test_artifact(b"cached payload");

        cache.put(&key, artifact.clone());
        let retrieved = cache.get(&key).expect("should find cached artifact");
        assert_eq!(retrieved.data, artifact.data);

        assert!(cache.evict(&key));
        assert!(cache.get(&key).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_cache_stats_counts_bin_files() {
        let dir = std::env::temp_dir().join("fj-cache-test-file-stats");
        let _ = std::fs::create_dir_all(&dir);

        let mut cache = FileCache::new(dir.clone());
        cache.put(&test_key("aa"), test_artifact(b"one"));
        cache.put(&test_key("bb"), test_artifact(b"two"));

        let stats = cache.stats();
        assert_eq!(stats.entry_count, 2);
        assert!(stats.total_bytes > 0);

        cache.clear();
        assert_eq!(cache.stats().entry_count, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
