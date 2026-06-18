#![forbid(unsafe_code)]

//! LRU eviction policy for cache backends.
//!
//! Provides a size-budgeted least-recently-used eviction tracker that wraps
//! any `CacheBackend`. JAX has no built-in eviction (anchor P2C005-A17);
//! FrankenJAX adds LRU as a configurable defense against cache exhaustion DoS
//! (threat matrix: "Cache exhaustion DoS").

use crate::CacheKey;
use crate::backend::{CacheBackend, CacheStats, CachedArtifact};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

/// Configuration for LRU eviction.
#[derive(Debug, Clone)]
pub struct LruConfig {
    /// Maximum number of entries before eviction triggers.
    pub max_entries: usize,
    /// Maximum total byte size before eviction triggers. 0 = unlimited.
    pub max_bytes: u64,
}

impl Default for LruConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_bytes: 256 * 1024 * 1024, // 256 MiB
        }
    }
}

/// LRU-evicting wrapper around any `CacheBackend`.
///
/// Tracks access order in a `Mutex<VecDeque>` of cache keys, enabling true
/// LRU behavior: both `get()` and `put()` update recency. On `put`, if the
/// cache exceeds `max_entries` or `max_bytes`, the least-recently-used
/// entry is evicted from the underlying backend.
pub struct LruCache<B: CacheBackend> {
    inner: B,
    config: LruConfig,
    /// Access-ordered queue: front = least recently used, back = most recent.
    /// Wrapped in Mutex so `get(&self)` can update recency.
    pub(crate) order: Arc<Mutex<VecDeque<CacheKey>>>,
}

impl<B: CacheBackend> std::fmt::Debug for LruCache<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let order = self.order.lock().unwrap_or_else(|e| e.into_inner());
        f.debug_struct("LruCache")
            .field("config", &self.config)
            .field("order", &*order)
            .finish_non_exhaustive()
    }
}

impl<B: CacheBackend> LruCache<B> {
    pub fn new(inner: B, config: LruConfig) -> Self {
        Self {
            inner,
            config,
            order: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn pop_oldest_cache_key(&self) -> Option<CacheKey> {
        let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
        order.pop_front()
    }

    fn restore_oldest_cache_key_if_absent(&self, key: CacheKey) {
        let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
        if !order.iter().any(|cached_key| cached_key == &key) {
            order.push_front(key);
        }
    }

    fn evict_inner_with_failure_status(&mut self, key: &CacheKey) -> (bool, bool) {
        let evict_failures_before = self.inner.evict_failure_count();
        let evicted = self.inner.evict(key);
        let evict_failed = self.inner.evict_failure_count() != evict_failures_before;
        (evicted, evict_failed)
    }

    /// Move a key to the most-recently-used position.
    /// Safe to call from `&self` thanks to interior mutability.
    fn touch(&self, key: &CacheKey) {
        let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = order.iter().position(|cached_key| cached_key == key) {
            order.remove(pos);
        }
        order.push_back(key.clone());
    }

    /// Evict least-recently-used entries until within budget.
    fn enforce_budget(&mut self) {
        loop {
            let evict_key = {
                let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
                if order.len() > self.config.max_entries {
                    order.pop_front()
                } else {
                    None
                }
            };
            let Some(evict_key) = evict_key else {
                break;
            };

            let (_, evict_failed) = self.evict_inner_with_failure_status(&evict_key);
            if evict_failed {
                self.restore_oldest_cache_key_if_absent(evict_key);
                break;
            }
        }

        // Evict by byte budget (if configured).
        if self.config.max_bytes > 0 {
            while self.inner.stats().total_bytes > self.config.max_bytes {
                let Some(evict_key) = self.pop_oldest_cache_key() else {
                    break;
                };
                let (_, evict_failed) = self.evict_inner_with_failure_status(&evict_key);
                if evict_failed {
                    self.restore_oldest_cache_key_if_absent(evict_key);
                    break;
                }
            }
        }
    }
}

impl<B: CacheBackend> CacheBackend for LruCache<B> {
    fn get(&self, key: &CacheKey) -> Option<CachedArtifact> {
        let result = self.inner.get(key);
        if result.is_some() {
            // Update recency on cache hit for true LRU behavior.
            self.touch(key);
        }
        result
    }

    fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
        let put_failures_before = self.inner.put_failure_count();
        self.inner.put(key, artifact);
        if self.inner.put_failure_count() != put_failures_before {
            return;
        }

        self.touch(key);
        self.enforce_budget();
    }

    fn evict(&mut self, key: &CacheKey) -> bool {
        let (evicted, evict_failed) = self.evict_inner_with_failure_status(key);

        if evicted || !evict_failed {
            let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
            order.retain(|cached_key| cached_key != key);
        }

        evicted
    }

    fn stats(&self) -> CacheStats {
        self.inner.stats()
    }

    fn clear(&mut self) {
        let clear_failures_before = self.inner.clear_failure_count();
        self.inner.clear();
        if self.inner.clear_failure_count() == clear_failures_before {
            let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
            order.clear();
        }
    }

    fn put_failure_count(&self) -> u64 {
        self.inner.put_failure_count()
    }

    fn evict_failure_count(&self) -> u64 {
        self.inner.evict_failure_count()
    }

    fn clear_failure_count(&self) -> u64 {
        self.inner.clear_failure_count()
    }
}

/// Configuration for TTL + LRU eviction.
#[derive(Debug, Clone)]
pub struct TtlLruConfig {
    /// Base LRU configuration (entry count and byte budget).
    pub lru: LruConfig,
    /// Time-to-live in seconds for each cache entry. 0 = no TTL.
    pub ttl_secs: u64,
}

impl Default for TtlLruConfig {
    fn default() -> Self {
        Self {
            lru: LruConfig::default(),
            ttl_secs: 3600, // 1 hour default TTL
        }
    }
}

/// TTL + LRU evicting wrapper around any `CacheBackend`.
///
/// Entries are evicted when they exceed the configured time-to-live (TTL),
/// in addition to standard LRU eviction by count and byte budget.
/// Uses `std::time::Instant` for monotonic time measurement.
#[derive(Debug)]
pub struct TtlLruCache<B: CacheBackend> {
    inner: LruCache<B>,
    ttl_secs: u64,
    /// Track insertion timestamps per cache key.
    insert_times: HashMap<CacheKey, std::time::Instant>,
}

impl<B: CacheBackend> TtlLruCache<B> {
    pub fn new(inner: B, config: TtlLruConfig) -> Self {
        Self {
            inner: LruCache::new(inner, config.lru),
            ttl_secs: config.ttl_secs,
            insert_times: HashMap::new(),
        }
    }

    /// Remove expired entries from the cache.
    pub fn sweep_expired(&mut self) {
        if self.ttl_secs == 0 {
            return;
        }
        let now = std::time::Instant::now();
        let ttl = std::time::Duration::from_secs(self.ttl_secs);
        let expired_keys: Vec<CacheKey> = self
            .insert_times
            .iter()
            .filter(|(_, inserted)| now.duration_since(**inserted) > ttl)
            .map(|(key, _)| key.clone())
            .collect();

        for key in &expired_keys {
            let evict_failures_before = self.inner.evict_failure_count();
            let evicted = self.inner.evict(key);
            let evict_failed = self.inner.evict_failure_count() != evict_failures_before;

            if evicted || !evict_failed {
                self.insert_times.remove(key);
            }
        }

        self.retain_active_insert_times();
    }

    fn retain_active_insert_times(&mut self) {
        let active_keys: HashSet<CacheKey> = {
            let order = self.inner.order.lock().unwrap_or_else(|e| e.into_inner());
            order.iter().cloned().collect()
        };
        self.insert_times.retain(|key, _| active_keys.contains(key));
    }

    /// Return the number of entries that would be expired by a sweep.
    pub fn expired_count(&self) -> usize {
        if self.ttl_secs == 0 {
            return 0;
        }
        let now = std::time::Instant::now();
        let ttl = std::time::Duration::from_secs(self.ttl_secs);
        self.insert_times
            .values()
            .filter(|inserted| now.duration_since(**inserted) > ttl)
            .count()
    }
}

impl<B: CacheBackend> CacheBackend for TtlLruCache<B> {
    fn get(&self, key: &CacheKey) -> Option<CachedArtifact> {
        // Check if the entry has expired before returning it
        if self.ttl_secs > 0
            && let Some(&inserted) = self.insert_times.get(key)
        {
            let ttl = std::time::Duration::from_secs(self.ttl_secs);
            if std::time::Instant::now().duration_since(inserted) > ttl {
                return None; // Expired
            }
        }
        self.inner.get(key)
    }

    fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
        let put_failures_before = self.inner.put_failure_count();
        self.inner.put(key, artifact);
        if self.inner.put_failure_count() != put_failures_before {
            return;
        }

        if self.ttl_secs > 0 {
            self.insert_times
                .insert(key.clone(), std::time::Instant::now());
            self.retain_active_insert_times();
            // Opportunistically sweep expired entries.
            self.sweep_expired();
        }
    }

    fn evict(&mut self, key: &CacheKey) -> bool {
        let evict_failures_before = self.inner.evict_failure_count();
        let evicted = self.inner.evict(key);
        let evict_failed = self.inner.evict_failure_count() != evict_failures_before;

        if evicted || !evict_failed {
            self.insert_times.remove(key);
        }

        evicted
    }

    fn stats(&self) -> CacheStats {
        self.inner.stats()
    }

    fn clear(&mut self) {
        let clear_failures_before = self.inner.clear_failure_count();
        self.inner.clear();
        if self.inner.clear_failure_count() == clear_failures_before {
            self.insert_times.clear();
        }
    }

    fn put_failure_count(&self) -> u64 {
        self.inner.put_failure_count()
    }

    fn evict_failure_count(&self) -> u64 {
        self.inner.evict_failure_count()
    }

    fn clear_failure_count(&self) -> u64 {
        self.inner.clear_failure_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InMemoryCache;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn test_key(digest: &str) -> CacheKey {
        CacheKey {
            namespace: "fjx",
            digest_hex: digest.to_owned(),
        }
    }

    fn namespaced_key(namespace: &'static str, digest: &str) -> CacheKey {
        CacheKey {
            namespace,
            digest_hex: digest.to_owned(),
        }
    }

    fn test_artifact(data: &[u8]) -> CachedArtifact {
        CachedArtifact {
            data: data.to_vec(),
            integrity_sha256_hex: crate::sha256_hex(data),
        }
    }

    #[derive(Debug, Default)]
    struct ReentrantAuditBackend {
        entries: HashMap<String, CachedArtifact>,
        order: Option<Arc<Mutex<VecDeque<CacheKey>>>>,
        evict_observed_unlocked: Arc<AtomicBool>,
        stats_observed_unlocked: Arc<AtomicBool>,
    }

    impl ReentrantAuditBackend {
        fn with_order(&mut self, order: Arc<Mutex<VecDeque<CacheKey>>>) {
            self.order = Some(order);
        }

        fn assert_order_unlocked(&self, method: &str) {
            if let Some(order) = &self.order {
                assert!(
                    order.try_lock().is_ok(),
                    "LRU order lock held during backend {method}"
                );
            }
        }
    }

    impl CacheBackend for ReentrantAuditBackend {
        fn get(&self, key: &CacheKey) -> Option<CachedArtifact> {
            self.entries.get(&key.as_string()).cloned()
        }

        fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
            self.entries.insert(key.as_string(), artifact);
        }

        fn evict(&mut self, key: &CacheKey) -> bool {
            self.assert_order_unlocked("evict");
            self.evict_observed_unlocked.store(true, Ordering::SeqCst);
            self.entries.remove(&key.as_string()).is_some()
        }

        fn stats(&self) -> CacheStats {
            self.assert_order_unlocked("stats");
            self.stats_observed_unlocked.store(true, Ordering::SeqCst);
            CacheStats {
                entry_count: self.entries.len(),
                total_bytes: self.entries.values().map(|a| a.data.len() as u64).sum(),
            }
        }

        fn clear(&mut self) {
            self.entries.clear();
        }
    }

    #[derive(Debug, Default)]
    struct FailingPutBackend {
        entries: HashMap<String, CachedArtifact>,
        fail_next_put: bool,
        put_failures: u64,
    }

    impl CacheBackend for FailingPutBackend {
        fn get(&self, key: &CacheKey) -> Option<CachedArtifact> {
            self.entries.get(&key.as_string()).cloned()
        }

        fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
            if self.fail_next_put {
                self.fail_next_put = false;
                self.put_failures += 1;
                return;
            }
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

        fn put_failure_count(&self) -> u64 {
            self.put_failures
        }
    }

    #[derive(Debug, Default)]
    struct FailingEvictBackend {
        entries: HashMap<String, CachedArtifact>,
        fail_next_evict: bool,
        evict_failures: u64,
    }

    impl CacheBackend for FailingEvictBackend {
        fn get(&self, key: &CacheKey) -> Option<CachedArtifact> {
            self.entries.get(&key.as_string()).cloned()
        }

        fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
            self.entries.insert(key.as_string(), artifact);
        }

        fn evict(&mut self, key: &CacheKey) -> bool {
            if self.fail_next_evict {
                self.fail_next_evict = false;
                self.evict_failures += 1;
                return false;
            }

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

        fn evict_failure_count(&self) -> u64 {
            self.evict_failures
        }
    }

    #[derive(Debug, Default)]
    struct FailingClearBackend {
        entries: HashMap<String, CachedArtifact>,
        fail_next_clear: bool,
        clear_failures: u64,
    }

    impl CacheBackend for FailingClearBackend {
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
            if self.fail_next_clear {
                self.fail_next_clear = false;
                self.clear_failures += 1;
                return;
            }

            self.entries.clear();
        }

        fn clear_failure_count(&self) -> u64 {
            self.clear_failures
        }
    }

    #[test]
    fn lru_evicts_oldest_when_over_max_entries() {
        let config = LruConfig {
            max_entries: 2,
            max_bytes: 0, // unlimited bytes
        };
        let mut cache = LruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"first"));
        cache.put(&test_key("b"), test_artifact(b"second"));
        assert_eq!(cache.stats().entry_count, 2);

        // Adding a third should evict "a" (oldest).
        cache.put(&test_key("c"), test_artifact(b"third"));
        assert_eq!(cache.stats().entry_count, 2);
        assert!(
            cache.get(&test_key("a")).is_none(),
            "oldest should be evicted"
        );
        assert!(cache.get(&test_key("b")).is_some());
        assert!(cache.get(&test_key("c")).is_some());
    }

    #[test]
    fn lru_evicts_non_default_namespace_keys() {
        let config = LruConfig {
            max_entries: 1,
            max_bytes: 0,
        };
        let mut cache = LruCache::new(InMemoryCache::new(), config);
        let first = namespaced_key("custom", "a");
        let second = namespaced_key("custom", "b");

        cache.put(&first, test_artifact(b"first"));
        cache.put(&second, test_artifact(b"second"));

        assert_eq!(cache.stats().entry_count, 1);
        assert!(
            cache.get(&first).is_none(),
            "LRU eviction must preserve the original key namespace"
        );
        assert!(cache.get(&second).is_some());
    }

    #[test]
    fn lru_evicts_by_byte_budget() {
        let config = LruConfig {
            max_entries: 100,
            max_bytes: 10, // 10 bytes max
        };
        let mut cache = LruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"12345")); // 5 bytes
        cache.put(&test_key("b"), test_artifact(b"67890")); // 5 bytes, total 10
        assert_eq!(cache.stats().entry_count, 2);

        // Adding more should trigger eviction to stay under 10 bytes.
        cache.put(&test_key("c"), test_artifact(b"XXXXX")); // 5 more bytes
        assert!(cache.stats().total_bytes <= 10);
    }

    #[test]
    fn lru_budget_failed_evict_restores_oldest_order_key() {
        let config = LruConfig {
            max_entries: 1,
            max_bytes: 0,
        };
        let mut cache = LruCache::new(FailingEvictBackend::default(), config);
        let oldest = test_key("oldest");
        let newest = test_key("newest");

        cache.put(&oldest, test_artifact(b"oldest"));
        cache.inner.fail_next_evict = true;
        cache.put(&newest, test_artifact(b"newest"));

        assert_eq!(cache.evict_failure_count(), 1);
        assert_eq!(
            cache.stats().entry_count,
            2,
            "backend keeps both entries because budget eviction failed"
        );
        {
            let order = cache.order.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(
                order.len(),
                2,
                "failed budget eviction must not drop LRU tracking"
            );
            assert_eq!(order.front(), Some(&oldest));
            assert_eq!(order.back(), Some(&newest));
        }
        assert!(cache.get(&oldest).is_some());
        assert!(cache.get(&newest).is_some());
    }

    #[test]
    fn lru_byte_budget_failed_evict_restores_oldest_order_key() {
        let config = LruConfig {
            max_entries: 100,
            max_bytes: 5,
        };
        let mut cache = LruCache::new(FailingEvictBackend::default(), config);
        let oldest = test_key("oldest");
        let newest = test_key("newest");

        cache.put(&oldest, test_artifact(b"12345"));
        cache.inner.fail_next_evict = true;
        cache.put(&newest, test_artifact(b"67890"));

        assert_eq!(cache.evict_failure_count(), 1);
        assert_eq!(
            cache.stats().total_bytes,
            10,
            "backend keeps both entries because byte-budget eviction failed"
        );
        {
            let order = cache.order.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(
                order.len(),
                2,
                "failed byte-budget eviction must not drop LRU tracking"
            );
            assert_eq!(order.front(), Some(&oldest));
            assert_eq!(order.back(), Some(&newest));
        }
        assert!(cache.get(&oldest).is_some());
        assert!(cache.get(&newest).is_some());
    }

    #[test]
    fn lru_releases_order_lock_before_backend_budget_calls() {
        let config = LruConfig {
            max_entries: 1,
            max_bytes: 5,
        };
        let mut cache = LruCache::new(ReentrantAuditBackend::default(), config);
        cache.inner.with_order(Arc::clone(&cache.order));

        let evict_observed = Arc::clone(&cache.inner.evict_observed_unlocked);
        let stats_observed = Arc::clone(&cache.inner.stats_observed_unlocked);

        cache.put(&test_key("a"), test_artifact(b"12345"));
        cache.put(&test_key("b"), test_artifact(b"67890"));

        assert!(
            evict_observed.load(Ordering::SeqCst),
            "entry-count eviction should call backend evict"
        );
        assert!(
            stats_observed.load(Ordering::SeqCst),
            "byte-budget enforcement should call backend stats"
        );
        assert_eq!(cache.stats().entry_count, 1);
    }

    #[test]
    fn lru_get_updates_recency() {
        // Verify true LRU: reading "a" makes it more recent than "b",
        // so inserting "c" should evict "b" (not "a").
        let config = LruConfig {
            max_entries: 2,
            max_bytes: 0,
        };
        let mut cache = LruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"first"));
        cache.put(&test_key("b"), test_artifact(b"second"));

        // Touch "a" via get — makes it most-recently-used.
        assert!(cache.get(&test_key("a")).is_some());

        // Insert "c" — should evict "b" (now the least recently used).
        cache.put(&test_key("c"), test_artifact(b"third"));
        assert_eq!(cache.stats().entry_count, 2);
        assert!(
            cache.get(&test_key("a")).is_some(),
            "recently-read 'a' should survive"
        );
        assert!(
            cache.get(&test_key("b")).is_none(),
            "untouched 'b' should be evicted"
        );
        assert!(cache.get(&test_key("c")).is_some());
    }

    #[test]
    fn lru_clear_resets_everything() {
        let config = LruConfig::default();
        let mut cache = LruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"data"));
        cache.clear();
        assert_eq!(cache.stats().entry_count, 0);
    }

    // ── TtlLruCache tests ───────────────────────────────────────────

    #[test]
    fn ttl_lru_basic_put_get() {
        let config = TtlLruConfig {
            lru: LruConfig {
                max_entries: 100,
                max_bytes: 0,
            },
            ttl_secs: 3600, // 1 hour — won't expire during test
        };
        let mut cache = TtlLruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"hello"));
        assert!(cache.get(&test_key("a")).is_some());
        assert_eq!(cache.stats().entry_count, 1);
    }

    #[test]
    fn ttl_lru_failed_put_does_not_record_timestamp() {
        let config = TtlLruConfig {
            lru: LruConfig::default(),
            ttl_secs: 3600,
        };
        let mut cache = TtlLruCache::new(FailingPutBackend::default(), config);
        let key = test_key("failed");

        cache.inner.inner.fail_next_put = true;
        cache.put(&key, test_artifact(b"not stored"));

        assert_eq!(cache.put_failure_count(), 1);
        assert!(
            !cache.insert_times.contains_key(&key),
            "failed put must not create TTL metadata for a missing artifact"
        );
        assert_eq!(cache.expired_count(), 0);
        assert_eq!(cache.stats().entry_count, 0);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn ttl_lru_evict_removes_entry_and_timestamp() {
        let config = TtlLruConfig {
            lru: LruConfig {
                max_entries: 100,
                max_bytes: 0,
            },
            ttl_secs: 3600,
        };
        let mut cache = TtlLruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"data"));
        assert!(cache.evict(&test_key("a")));
        assert!(cache.get(&test_key("a")).is_none());
        assert_eq!(cache.stats().entry_count, 0);
    }

    #[test]
    fn ttl_lru_failed_evict_keeps_expiry_metadata_to_block_stale_hit() {
        let config = TtlLruConfig {
            lru: LruConfig::default(),
            ttl_secs: 1,
        };
        let mut cache = TtlLruCache::new(FailingEvictBackend::default(), config);
        let key = test_key("a");

        cache.put(&key, test_artifact(b"stale"));
        cache.insert_times.insert(
            key.clone(),
            std::time::Instant::now() - std::time::Duration::from_secs(2),
        );
        cache.inner.inner.fail_next_evict = true;

        assert!(!cache.evict(&key));
        assert_eq!(cache.evict_failure_count(), 1);
        assert_eq!(
            cache.expired_count(),
            1,
            "failed eviction must leave TTL metadata in place"
        );
        assert!(
            cache.get(&key).is_none(),
            "failed eviction must not make expired stale bytes readable again"
        );
        assert_eq!(
            cache.stats().entry_count,
            1,
            "backend entry remains present because eviction failed"
        );
    }

    #[test]
    fn ttl_lru_failed_sweep_keeps_expiry_metadata_to_block_stale_hit() {
        let config = TtlLruConfig {
            lru: LruConfig::default(),
            ttl_secs: 1,
        };
        let mut cache = TtlLruCache::new(FailingEvictBackend::default(), config);
        let key = test_key("a");

        cache.put(&key, test_artifact(b"stale"));
        cache.insert_times.insert(
            key.clone(),
            std::time::Instant::now() - std::time::Duration::from_secs(2),
        );
        cache.inner.inner.fail_next_evict = true;

        cache.sweep_expired();

        assert_eq!(cache.evict_failure_count(), 1);
        assert_eq!(
            cache.expired_count(),
            1,
            "failed sweep eviction must leave TTL metadata in place"
        );
        assert!(
            cache.get(&key).is_none(),
            "failed sweep must not make expired stale bytes readable again"
        );
        assert_eq!(
            cache.stats().entry_count,
            1,
            "backend entry remains present because eviction failed"
        );
    }

    #[test]
    fn ttl_lru_clear_removes_all() {
        let config = TtlLruConfig::default();
        let mut cache = TtlLruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"one"));
        cache.put(&test_key("b"), test_artifact(b"two"));
        assert_eq!(cache.stats().entry_count, 2);

        cache.clear();
        assert_eq!(cache.stats().entry_count, 0);
        assert_eq!(cache.expired_count(), 0);
    }

    #[test]
    fn ttl_lru_failed_clear_keeps_expiry_metadata_to_block_stale_hit() {
        let config = TtlLruConfig {
            lru: LruConfig::default(),
            ttl_secs: 1,
        };
        let mut cache = TtlLruCache::new(FailingClearBackend::default(), config);
        let key = test_key("a");

        cache.put(&key, test_artifact(b"stale"));
        cache.insert_times.insert(
            key.clone(),
            std::time::Instant::now() - std::time::Duration::from_secs(2),
        );
        cache.inner.inner.fail_next_clear = true;

        cache.clear();

        assert_eq!(cache.clear_failure_count(), 1);
        assert_eq!(
            cache.expired_count(),
            1,
            "failed clear must leave TTL metadata in place"
        );
        {
            let order = cache.inner.order.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(
                order.len(),
                1,
                "failed clear must leave LRU order metadata in place"
            );
            assert_eq!(order.front(), Some(&key));
        }
        assert!(
            cache.get(&key).is_none(),
            "failed clear must not make expired stale bytes readable again"
        );
        assert_eq!(
            cache.stats().entry_count,
            1,
            "backend entry remains present because clear failed"
        );
    }

    #[test]
    fn ttl_lru_respects_lru_entry_limit() {
        let config = TtlLruConfig {
            lru: LruConfig {
                max_entries: 2,
                max_bytes: 0,
            },
            ttl_secs: 3600,
        };
        let mut cache = TtlLruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"first"));
        cache.put(&test_key("b"), test_artifact(b"second"));
        cache.put(&test_key("c"), test_artifact(b"third"));

        // LRU eviction should have removed "a"
        assert!(cache.get(&test_key("a")).is_none());
        assert!(cache.get(&test_key("b")).is_some());
        assert!(cache.get(&test_key("c")).is_some());
        assert_eq!(cache.stats().entry_count, 2);
    }

    #[test]
    fn ttl_lru_prunes_timestamps_for_lru_evictions() {
        let config = TtlLruConfig {
            lru: LruConfig {
                max_entries: 1,
                max_bytes: 0,
            },
            ttl_secs: 3600,
        };
        let mut cache = TtlLruCache::new(InMemoryCache::new(), config);
        let first = namespaced_key("custom", "a");
        let second = namespaced_key("custom", "b");

        cache.put(&first, test_artifact(b"first"));
        cache.put(&second, test_artifact(b"second"));

        assert_eq!(cache.stats().entry_count, 1);
        assert_eq!(
            cache.insert_times.len(),
            1,
            "TTL metadata should not retain LRU-evicted keys"
        );
        assert!(
            !cache.insert_times.contains_key(&first),
            "evicted key timestamp should be pruned with its original namespace"
        );
        assert!(cache.insert_times.contains_key(&second));
    }

    #[test]
    fn ttl_lru_zero_ttl_disables_expiry() {
        let config = TtlLruConfig {
            lru: LruConfig::default(),
            ttl_secs: 0,
        };
        let mut cache = TtlLruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"data"));
        assert_eq!(cache.expired_count(), 0);
        cache.sweep_expired(); // Should be a no-op
        assert!(cache.get(&test_key("a")).is_some());
    }

    #[test]
    fn ttl_lru_expired_count_zero_for_fresh_entries() {
        let config = TtlLruConfig {
            lru: LruConfig::default(),
            ttl_secs: 3600,
        };
        let mut cache = TtlLruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"data"));
        cache.put(&test_key("b"), test_artifact(b"more"));
        // Just inserted, so nothing expired yet
        assert_eq!(cache.expired_count(), 0);
    }

    #[test]
    fn ttl_lru_default_config_values() {
        let config = TtlLruConfig::default();
        assert_eq!(config.ttl_secs, 3600);
        assert_eq!(config.lru.max_entries, 1024);
        assert_eq!(config.lru.max_bytes, 256 * 1024 * 1024);
    }

    // ====================== THREAD SAFETY TESTS ======================

    #[test]
    fn lru_concurrent_get_does_not_deadlock() {
        use std::sync::Arc;
        use std::thread;

        let config = LruConfig {
            max_entries: 100,
            max_bytes: 0,
        };
        let mut cache = LruCache::new(InMemoryCache::new(), config);

        for i in 0..10 {
            cache.put(&test_key(&format!("key{}", i)), test_artifact(b"data"));
        }

        let cache = Arc::new(cache);
        let mut handles = vec![];

        for t in 0..4 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    for i in 0..10 {
                        let _ = cache.get(&test_key(&format!("key{}", (i + t) % 10)));
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().expect("thread should not panic");
        }
    }

    #[test]
    fn lru_order_mutex_not_poisoned_on_panic_recovery() {
        use std::panic;

        let config = LruConfig {
            max_entries: 10,
            max_bytes: 0,
        };
        let cache = LruCache::new(InMemoryCache::new(), config);

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _guard = cache.order.lock().unwrap();
            panic!("intentional panic while holding lock");
        }));

        assert!(result.is_err(), "panic should be caught");

        let guard_result = cache.order.lock();
        assert!(
            guard_result.is_err(),
            "mutex should be poisoned after panic"
        );

        let guard = guard_result.unwrap_or_else(|e| e.into_inner());
        drop(guard);
    }

    #[test]
    fn lru_get_touch_is_atomic() {
        use std::sync::Arc;
        use std::thread;

        let config = LruConfig {
            max_entries: 3,
            max_bytes: 0,
        };
        let mut cache = LruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"A"));
        cache.put(&test_key("b"), test_artifact(b"B"));
        cache.put(&test_key("c"), test_artifact(b"C"));

        let cache = Arc::new(cache);

        let cache_a = Arc::clone(&cache);
        let handle_a = thread::spawn(move || {
            for _ in 0..1000 {
                let _ = cache_a.get(&test_key("a"));
            }
        });

        let cache_b = Arc::clone(&cache);
        let handle_b = thread::spawn(move || {
            for _ in 0..1000 {
                let _ = cache_b.get(&test_key("b"));
            }
        });

        handle_a.join().expect("thread a should not panic");
        handle_b.join().expect("thread b should not panic");

        let order = cache.order.lock().unwrap();
        assert_eq!(order.len(), 3, "all keys should remain in order queue");
    }

    #[test]
    fn lru_cache_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<LruCache<InMemoryCache>>();
    }

    #[test]
    fn lru_cache_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<LruCache<InMemoryCache>>();
    }

    #[test]
    fn lru_cache_delegates_failure_counts_to_inner_file_cache() {
        use crate::backend::FileCache;

        // Point the cache under a regular file. FileCache::new creates normal
        // missing directories, but this path cannot become a directory.
        let parent_file = std::env::temp_dir().join(format!(
            "fj-cache-lru-delegation-parent-{}",
            std::process::id()
        ));
        std::fs::write(&parent_file, b"not a directory").expect("parent marker should write");
        let dir = parent_file.join("never_created");

        let inner = FileCache::new(dir);
        let mut cache = LruCache::new(inner, LruConfig::default());

        assert_eq!(cache.put_failure_count(), 0);
        cache.put(
            &CacheKey {
                namespace: "fjx",
                digest_hex: "deadbeef".to_owned(),
            },
            CachedArtifact {
                data: vec![1, 2, 3],
                integrity_sha256_hex: crate::sha256_hex(&[1, 2, 3]),
            },
        );
        assert_eq!(
            cache.put_failure_count(),
            1,
            "LruCache::put_failure_count must forward to inner FileCache"
        );

        let _ = std::fs::remove_file(&parent_file);
    }

    #[test]
    fn lru_failed_put_does_not_evict_live_entry() {
        let config = LruConfig {
            max_entries: 1,
            max_bytes: 0,
        };
        let mut cache = LruCache::new(FailingPutBackend::default(), config);
        let live = test_key("live");
        let failed = test_key("failed");

        cache.put(&live, test_artifact(b"live artifact"));
        cache.inner.fail_next_put = true;
        cache.put(&failed, test_artifact(b"not stored"));

        assert_eq!(cache.put_failure_count(), 1);
        {
            let order = cache.order.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(order.len(), 1);
            assert_eq!(order.front(), Some(&live));
        }
        assert!(
            cache.get(&live).is_some(),
            "failed put must not evict the existing live entry"
        );
        assert!(
            cache.get(&failed).is_none(),
            "failed put must not create a readable entry"
        );
        assert_eq!(cache.stats().entry_count, 1);
    }

    #[test]
    fn in_memory_cache_default_failure_counts_are_zero() {
        let cache = InMemoryCache::default();
        assert_eq!(cache.put_failure_count(), 0);
        assert_eq!(cache.evict_failure_count(), 0);
        assert_eq!(cache.clear_failure_count(), 0);
    }
}
