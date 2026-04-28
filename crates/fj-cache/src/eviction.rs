#![forbid(unsafe_code)]

//! LRU eviction policy for cache backends.
//!
//! Provides a size-budgeted least-recently-used eviction tracker that wraps
//! any `CacheBackend`. JAX has no built-in eviction (anchor P2C005-A17);
//! FrankenJAX adds LRU as a configurable defense against cache exhaustion DoS
//! (threat matrix: "Cache exhaustion DoS").

use crate::CacheKey;
use crate::backend::{CacheBackend, CacheStats, CachedArtifact};
use std::collections::VecDeque;
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
/// Tracks access order in a `Mutex<VecDeque>` of key strings, enabling true
/// LRU behavior: both `get()` and `put()` update recency. On `put`, if the
/// cache exceeds `max_entries` or `max_bytes`, the least-recently-used
/// entry is evicted from the underlying backend.
pub struct LruCache<B: CacheBackend> {
    inner: B,
    config: LruConfig,
    /// Access-ordered queue: front = least recently used, back = most recent.
    /// Wrapped in Mutex so `get(&self)` can update recency.
    pub(crate) order: Arc<Mutex<VecDeque<String>>>,
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

    fn order_key_to_cache_key(order_key: &str) -> Option<CacheKey> {
        let (_namespace, digest_hex) = order_key.split_once('-')?;
        Some(CacheKey {
            namespace: "fjx",
            digest_hex: digest_hex.to_owned(),
        })
    }

    fn pop_oldest_cache_key(&self) -> Option<CacheKey> {
        loop {
            let oldest_key_str = {
                let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
                order.pop_front()
            }?;
            if let Some(evict_key) = Self::order_key_to_cache_key(&oldest_key_str) {
                return Some(evict_key);
            }
        }
    }

    /// Move a key to the most-recently-used position.
    /// Safe to call from `&self` thanks to interior mutability.
    fn touch(&self, key_str: &str) {
        let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = order.iter().position(|k| k == key_str) {
            order.remove(pos);
        }
        order.push_back(key_str.to_owned());
    }

    /// Evict least-recently-used entries until within budget.
    fn enforce_budget(&mut self) {
        let entry_count_evictions = {
            let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
            let mut evict_keys = Vec::new();

            while order.len() > self.config.max_entries {
                if let Some(oldest_key_str) = order.pop_front() {
                    if let Some(evict_key) = Self::order_key_to_cache_key(&oldest_key_str) {
                        evict_keys.push(evict_key);
                    }
                } else {
                    break;
                }
            }

            evict_keys
        };

        for evict_key in entry_count_evictions {
            self.inner.evict(&evict_key);
        }

        // Evict by byte budget (if configured).
        if self.config.max_bytes > 0 {
            while self.inner.stats().total_bytes > self.config.max_bytes {
                let Some(evict_key) = self.pop_oldest_cache_key() else {
                    break;
                };
                self.inner.evict(&evict_key);
            }
        }
    }
}

impl<B: CacheBackend> CacheBackend for LruCache<B> {
    fn get(&self, key: &CacheKey) -> Option<CachedArtifact> {
        let result = self.inner.get(key);
        if result.is_some() {
            // Update recency on cache hit for true LRU behavior.
            self.touch(&key.as_string());
        }
        result
    }

    fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
        let key_str = key.as_string();
        self.touch(&key_str);
        self.inner.put(key, artifact);
        self.enforce_budget();
    }

    fn evict(&mut self, key: &CacheKey) -> bool {
        let key_str = key.as_string();
        {
            let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
            order.retain(|k| k != &key_str);
        }
        self.inner.evict(key)
    }

    fn stats(&self) -> CacheStats {
        self.inner.stats()
    }

    fn clear(&mut self) {
        {
            let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
            order.clear();
        }
        self.inner.clear();
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
    /// Track insertion timestamps per key string.
    insert_times: std::collections::HashMap<String, std::time::Instant>,
}

impl<B: CacheBackend> TtlLruCache<B> {
    pub fn new(inner: B, config: TtlLruConfig) -> Self {
        Self {
            inner: LruCache::new(inner, config.lru),
            ttl_secs: config.ttl_secs,
            insert_times: std::collections::HashMap::new(),
        }
    }

    /// Remove expired entries from the cache.
    pub fn sweep_expired(&mut self) {
        if self.ttl_secs == 0 {
            return;
        }
        let now = std::time::Instant::now();
        let ttl = std::time::Duration::from_secs(self.ttl_secs);
        let expired_keys: Vec<String> = self
            .insert_times
            .iter()
            .filter(|(_, inserted)| now.duration_since(**inserted) > ttl)
            .map(|(k, _)| k.clone())
            .collect();

        for key_str in &expired_keys {
            if let Some((ns, hex)) = key_str.split_once('-') {
                let evict_key = CacheKey {
                    namespace: match ns {
                        "fjx" => "fjx",
                        _ => "fjx",
                    },
                    digest_hex: hex.to_owned(),
                };
                self.inner.evict(&evict_key);
            }
            self.insert_times.remove(key_str);
        }

        // Prevent memory leak by cleaning up keys silently evicted by the underlying LRU
        if self.insert_times.len() > self.inner.config.max_entries.saturating_mul(2).max(1024) {
            let active_keys: std::collections::HashSet<String> = {
                let order = self.inner.order.lock().unwrap_or_else(|e| e.into_inner());
                order.iter().cloned().collect()
            };
            self.insert_times
                .retain(|k, _| active_keys.contains(k.as_str()));
        }
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
        if self.ttl_secs > 0 {
            let key_str = key.as_string();
            if let Some(&inserted) = self.insert_times.get(&key_str) {
                let ttl = std::time::Duration::from_secs(self.ttl_secs);
                if std::time::Instant::now().duration_since(inserted) > ttl {
                    return None; // Expired
                }
            }
        }
        self.inner.get(key)
    }

    fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
        let key_str = key.as_string();
        if self.ttl_secs > 0 {
            self.insert_times.insert(key_str, std::time::Instant::now());
        }
        self.inner.put(key, artifact);
        // Opportunistically sweep expired entries
        self.sweep_expired();
    }

    fn evict(&mut self, key: &CacheKey) -> bool {
        let key_str = key.as_string();
        self.insert_times.remove(&key_str);
        self.inner.evict(key)
    }

    fn stats(&self) -> CacheStats {
        self.inner.stats()
    }

    fn clear(&mut self) {
        self.insert_times.clear();
        self.inner.clear();
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

    fn test_artifact(data: &[u8]) -> CachedArtifact {
        CachedArtifact {
            data: data.to_vec(),
            integrity_sha256_hex: crate::sha256_hex(data),
        }
    }

    #[derive(Debug, Default)]
    struct ReentrantAuditBackend {
        entries: HashMap<String, CachedArtifact>,
        order: Option<Arc<Mutex<VecDeque<String>>>>,
        evict_observed_unlocked: Arc<AtomicBool>,
        stats_observed_unlocked: Arc<AtomicBool>,
    }

    impl ReentrantAuditBackend {
        fn with_order(&mut self, order: Arc<Mutex<VecDeque<String>>>) {
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
}
