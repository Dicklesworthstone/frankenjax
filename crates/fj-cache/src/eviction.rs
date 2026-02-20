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
/// Tracks access order in a `VecDeque` of key strings. On `put`, if the
/// cache exceeds `max_entries` or `max_bytes`, the least-recently-used
/// entry is evicted from the underlying backend.
#[derive(Debug)]
pub struct LruCache<B: CacheBackend> {
    inner: B,
    config: LruConfig,
    /// Access-ordered queue: front = least recently used, back = most recent.
    order: VecDeque<String>,
}

impl<B: CacheBackend> LruCache<B> {
    pub fn new(inner: B, config: LruConfig) -> Self {
        Self {
            inner,
            config,
            order: VecDeque::new(),
        }
    }

    /// Move a key to the most-recently-used position.
    fn touch(&mut self, key_str: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key_str) {
            self.order.remove(pos);
        }
        self.order.push_back(key_str.to_owned());
    }

    /// Evict least-recently-used entries until within budget.
    fn enforce_budget(&mut self) {
        // Evict by entry count.
        while self.order.len() > self.config.max_entries {
            if let Some(oldest_key_str) = self.order.pop_front() {
                // Construct a temporary CacheKey for eviction.
                if let Some((ns, hex)) = oldest_key_str.split_once('-') {
                    let evict_key = CacheKey {
                        namespace: match ns {
                            "fjx" => "fjx",
                            _ => "fjx", // V1: only fjx namespace
                        },
                        digest_hex: hex.to_owned(),
                    };
                    self.inner.evict(&evict_key);
                }
            }
        }

        // Evict by byte budget (if configured).
        if self.config.max_bytes > 0 {
            while self.inner.stats().total_bytes > self.config.max_bytes {
                if let Some(oldest_key_str) = self.order.pop_front() {
                    if let Some((_ns, hex)) = oldest_key_str.split_once('-') {
                        let evict_key = CacheKey {
                            namespace: "fjx",
                            digest_hex: hex.to_owned(),
                        };
                        self.inner.evict(&evict_key);
                    }
                } else {
                    break; // No more keys to evict
                }
            }
        }
    }
}

impl<B: CacheBackend> CacheBackend for LruCache<B> {
    fn get(&self, key: &CacheKey) -> Option<CachedArtifact> {
        self.inner.get(key)
        // NOTE: Ideally we'd call self.touch() here, but get() takes &self.
        // V1 approach: eviction is based on insertion order only.
        // Future: use interior mutability (RefCell or Mutex) for true LRU.
    }

    fn put(&mut self, key: &CacheKey, artifact: CachedArtifact) {
        let key_str = key.as_string();
        self.touch(&key_str);
        self.inner.put(key, artifact);
        self.enforce_budget();
    }

    fn evict(&mut self, key: &CacheKey) -> bool {
        let key_str = key.as_string();
        self.order.retain(|k| k != &key_str);
        self.inner.evict(key)
    }

    fn stats(&self) -> CacheStats {
        self.inner.stats()
    }

    fn clear(&mut self) {
        self.order.clear();
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InMemoryCache;

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
    fn lru_clear_resets_everything() {
        let config = LruConfig::default();
        let mut cache = LruCache::new(InMemoryCache::new(), config);

        cache.put(&test_key("a"), test_artifact(b"data"));
        cache.clear();
        assert_eq!(cache.stats().entry_count, 0);
    }
}
