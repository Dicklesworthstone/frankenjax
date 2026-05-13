#![forbid(unsafe_code)]

//! Cache backend trait and concrete implementations.
//!
//! V1 scope: `InMemoryCache` (HashMap-backed) and `FileCache` (directory-based,
//! one file per key). Both backends support LRU eviction via `eviction::LruPolicy`.

use crate::CacheKey;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// Count of `put` operations that failed at the filesystem layer.
    ///
    /// Incremented when either the temp-file write or the rename step of the
    /// atomic-write pattern fails. The trait signature `fn put(&mut self, …)`
    /// returns `()`, so this counter is the only way for callers to observe
    /// silent persistence failures (disk full, permission denied, parent dir
    /// removed mid-flight, cross-device link). Read via
    /// [`FileCache::put_failure_count`].
    put_failures: AtomicU64,
    /// Count of `evict` operations whose underlying `remove_file` returned an
    /// error other than `NotFound`.
    ///
    /// The trait's `evict` contract is "Returns `true` if the entry existed",
    /// so `NotFound` is normal (file did not exist → return `false`) and is
    /// NOT counted. Any other IO error (permission denied, parent dir
    /// removed, etc.) means the file may have existed but could not be
    /// removed, and is recorded here so callers can detect silent eviction
    /// failure. Read via [`FileCache::evict_failure_count`].
    evict_failures: AtomicU64,
    /// Count of `clear` operations or per-entry removals that failed at the
    /// filesystem layer.
    ///
    /// Incremented when `read_dir` over the cache directory fails (entire
    /// clear() is silently skipped) or when any individual per-entry
    /// `remove_file` fails during the loop. Each per-entry failure bumps the
    /// counter by 1. Read via [`FileCache::clear_failure_count`].
    clear_failures: AtomicU64,
}

impl FileCache {
    /// Create a new `FileCache` rooted at the given directory.
    ///
    /// Does not create the directory — callers must ensure it exists.
    #[must_use]
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            put_failures: AtomicU64::new(0),
            evict_failures: AtomicU64::new(0),
            clear_failures: AtomicU64::new(0),
        }
    }

    /// Return the filesystem path for a given cache key.
    #[must_use]
    pub fn path_for(&self, key: &CacheKey) -> PathBuf {
        self.cache_dir.join(format!("{}.bin", key.as_string()))
    }

    /// Return the cumulative count of `put` operations that failed at the
    /// filesystem layer (temp-file write failure or atomic-rename failure).
    ///
    /// Callers monitor this to detect silent persistence problems that the
    /// `CacheBackend::put` trait method cannot surface via its `()` return
    /// type.
    #[must_use]
    pub fn put_failure_count(&self) -> u64 {
        self.put_failures.load(Ordering::Relaxed)
    }

    /// Return the cumulative count of `evict` operations whose underlying
    /// `remove_file` returned an error other than `NotFound`.
    ///
    /// `NotFound` is not counted because it matches the trait's "did not
    /// exist" semantics. Any other IO error means the entry may have
    /// existed and could not be removed.
    #[must_use]
    pub fn evict_failure_count(&self) -> u64 {
        self.evict_failures.load(Ordering::Relaxed)
    }

    /// Return the cumulative count of `clear` per-entry remove failures plus
    /// whole-directory `read_dir` failures.
    #[must_use]
    pub fn clear_failure_count(&self) -> u64 {
        self.clear_failures.load(Ordering::Relaxed)
    }
}

impl CacheBackend for FileCache {
    fn get(&self, key: &CacheKey) -> Option<CachedArtifact> {
        let path = self.path_for(key);
        let bytes = std::fs::read(&path).ok()?;

        match crate::persistence::deserialize(&bytes) {
            Ok(artifact) => Some(artifact),
            Err(_) => Some(CachedArtifact {
                data: bytes,
                integrity_sha256_hex: "corrupt-cache-artifact".to_owned(),
            }),
        }
    }

    fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
        let path = self.path_for(key);
        let bytes = crate::persistence::serialize(&artifact);
        // Atomic write: write to temp file, then rename.
        // Use a unique temp file to avoid concurrent write races.
        use std::sync::atomic::AtomicUsize;
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = path.with_extension(format!("tmp.{}.{}", std::process::id(), id));
        match std::fs::write(&tmp_path, bytes) {
            Ok(()) => {
                if std::fs::rename(&tmp_path, &path).is_err() {
                    // Atomic rename failed after a successful temp write.
                    // The cache silently dropped the artifact; surface the
                    // failure via put_failures so monitors can detect it.
                    self.put_failures.fetch_add(1, Ordering::Relaxed);
                    let _ = std::fs::remove_file(&tmp_path);
                }
            }
            Err(_) => {
                // Temp write failed — nothing to rename. Best-effort cleanup
                // in case a partial file was left behind, then record the
                // failure.
                let _ = std::fs::remove_file(&tmp_path);
                self.put_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn evict(&mut self, key: &CacheKey) -> bool {
        let path = self.path_for(key);
        match std::fs::remove_file(&path) {
            Ok(()) => true,
            Err(err) => {
                if err.kind() != std::io::ErrorKind::NotFound {
                    // The trait contract treats NotFound as "did not exist",
                    // which is a normal `false` return. Any other error
                    // means the file may have existed and we could not
                    // remove it — record the silent failure.
                    self.evict_failures.fetch_add(1, Ordering::Relaxed);
                }
                false
            }
        }
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
        match std::fs::read_dir(&self.cache_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().is_some_and(|e| e == "bin")
                        && std::fs::remove_file(&path).is_err()
                    {
                        // Per-entry remove failed (e.g., permission, file
                        // disappeared between read_dir and remove). Record
                        // so callers can detect partial clears.
                        self.clear_failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Err(_) => {
                // Cannot enumerate the cache dir; the whole clear was a
                // no-op rather than an error per current trait semantics.
                // Record so callers can detect this.
                self.clear_failures.fetch_add(1, Ordering::Relaxed);
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

    #[test]
    fn file_cache_corrupt_wire_returns_mismatched_artifact() {
        let dir =
            std::env::temp_dir().join(format!("fj-cache-test-corrupt-wire-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        let mut cache = FileCache::new(dir.clone());
        let key = test_key("corrupt");
        cache.put(&key, test_artifact(b"clean payload"));

        let path = cache.path_for(&key);
        let mut bytes = std::fs::read(&path).expect("cache file should exist");
        bytes[8] ^= 0xff;
        std::fs::write(&path, bytes).expect("cache file should be writable");

        let artifact = cache.get(&key).expect("corrupt artifact should be visible");
        assert_ne!(
            crate::sha256_hex(&artifact.data),
            artifact.integrity_sha256_hex,
            "corrupt wire payload must not be reported as integrity-clean"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_cache_put_failure_counter_increments_when_dir_missing() {
        // Point the cache at a path whose parent directory does not exist.
        // std::fs::write will fail with NotFound, exercising the
        // temp-write-failure branch of put().
        let dir = std::env::temp_dir().join(format!(
            "fj-cache-test-missing-parent-{}/never_created",
            std::process::id()
        ));

        let mut cache = FileCache::new(dir.clone());
        assert_eq!(cache.put_failure_count(), 0);

        let key = test_key("never_lands");
        cache.put(&key, test_artifact(b"this write will fail"));

        assert_eq!(
            cache.put_failure_count(),
            1,
            "put against a non-existent parent dir should bump put_failures"
        );

        // Underlying directory was never created, so stats() should report
        // an empty cache (read_dir falls back to default).
        let stats = cache.stats();
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.total_bytes, 0);

        // A second failing put bumps the counter again.
        cache.put(&key, test_artifact(b"still failing"));
        assert_eq!(cache.put_failure_count(), 2);
    }

    #[test]
    fn file_cache_put_failure_counter_stays_zero_on_success() {
        let dir = std::env::temp_dir().join(format!(
            "fj-cache-test-put-success-{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);

        let mut cache = FileCache::new(dir.clone());
        let key = test_key("ok");
        cache.put(&key, test_artifact(b"persists fine"));

        assert_eq!(cache.put_failure_count(), 0);
        assert_eq!(cache.stats().entry_count, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_cache_evict_missing_does_not_bump_failure_counter() {
        let dir =
            std::env::temp_dir().join(format!("fj-cache-test-evict-missing-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        let mut cache = FileCache::new(dir.clone());
        let key = test_key("absent");

        // Evicting a key that was never stored returns false and must NOT
        // bump evict_failures — NotFound is part of the trait's normal
        // "did not exist → false" contract.
        let removed = cache.evict(&key);
        assert!(!removed);
        assert_eq!(cache.evict_failure_count(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_cache_evict_success_does_not_bump_failure_counter() {
        let dir =
            std::env::temp_dir().join(format!("fj-cache-test-evict-success-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        let mut cache = FileCache::new(dir.clone());
        let key = test_key("present");
        cache.put(&key, test_artifact(b"data"));
        assert_eq!(cache.stats().entry_count, 1);

        let removed = cache.evict(&key);
        assert!(removed);
        assert_eq!(cache.evict_failure_count(), 0);
        assert_eq!(cache.stats().entry_count, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_cache_clear_bumps_counter_when_dir_missing() {
        // Point clear() at a directory that does not exist: read_dir fails,
        // bumping clear_failures by 1.
        let dir = std::env::temp_dir().join(format!(
            "fj-cache-test-clear-missing-{}/never_created",
            std::process::id()
        ));

        let mut cache = FileCache::new(dir.clone());
        assert_eq!(cache.clear_failure_count(), 0);

        cache.clear();
        assert_eq!(
            cache.clear_failure_count(),
            1,
            "clear() against a missing dir should record a single failure"
        );

        cache.clear();
        assert_eq!(cache.clear_failure_count(), 2);
    }

    #[test]
    fn file_cache_clear_success_does_not_bump_counter() {
        let dir = std::env::temp_dir().join(format!(
            "fj-cache-test-clear-success-{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);

        let mut cache = FileCache::new(dir.clone());
        cache.put(&test_key("a"), test_artifact(b"a"));
        cache.put(&test_key("b"), test_artifact(b"b"));
        assert_eq!(cache.stats().entry_count, 2);

        cache.clear();
        assert_eq!(cache.clear_failure_count(), 0);
        assert_eq!(cache.stats().entry_count, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
