#![forbid(unsafe_code)]

//! Cache key stability test harness.
//!
//! Detects accidental cache key drift by comparing generated keys against
//! golden reference values. If the canonical payload layout or hash function
//! changes, these tests fail immediately — preventing silent cache invalidation
//! across FrankenJAX versions (threat matrix: "Stale artifact serving").

use crate::{CacheKeyInput, build_cache_key};
use fj_core::{CompatibilityMode, Jaxpr, Transform};
use std::collections::BTreeMap;

/// A golden cache key reference for stability testing.
#[derive(Debug, Clone)]
pub struct GoldenKeyRef {
    /// Human-readable description of this test vector.
    pub description: &'static str,
    /// Expected hex digest (without 'fjx-' prefix).
    pub expected_digest_hex: &'static str,
    /// The input that should produce this key.
    pub input: CacheKeyInput,
}

/// Generate the standard set of golden key references.
///
/// These represent canonical test vectors that must remain stable across
/// releases. If any golden key changes, it indicates a cache key format
/// change that requires a version bump in the namespace prefix.
#[must_use]
pub fn golden_key_refs() -> Vec<GoldenKeyRef> {
    vec![
        GoldenKeyRef {
            description: "empty program, strict mode, no transforms",
            expected_digest_hex: "e26dbe3826d170f0c5b279392919ebb7fc28d2f6165a836f59c827a65ff8540d",
            input: CacheKeyInput {
                mode: CompatibilityMode::Strict,
                backend: "cpu".to_owned(),
                jaxpr: Jaxpr::new(vec![], vec![], vec![], vec![]),
                transform_stack: vec![],
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            },
        },
        GoldenKeyRef {
            description: "empty program, hardened mode, jit transform",
            expected_digest_hex: "5be109dd4dbf409f4bdf42d75bb797de2e474a2388610c0d591c45fa6d37fc4e",
            input: CacheKeyInput {
                mode: CompatibilityMode::Hardened,
                backend: "cpu".to_owned(),
                jaxpr: Jaxpr::new(vec![], vec![], vec![], vec![]),
                transform_stack: vec![Transform::Jit],
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            },
        },
        GoldenKeyRef {
            description: "empty program, strict mode, custom hook",
            expected_digest_hex: "bcf7cdbaac5a737150e3955476f3531acc94ea92c6c547f90df580610b291ca6",
            input: CacheKeyInput {
                mode: CompatibilityMode::Strict,
                backend: "cpu".to_owned(),
                jaxpr: Jaxpr::new(vec![], vec![], vec![], vec![]),
                transform_stack: vec![],
                compile_options: BTreeMap::new(),
                custom_hook: Some("my-hook".to_owned()),
                unknown_incompatible_features: vec![],
            },
        },
    ]
}

/// Compute current keys for all golden references and return them as
/// `(description, current_digest_hex)` pairs.
///
/// Use this to capture initial golden values or detect drift.
#[must_use]
pub fn capture_golden_keys() -> Vec<(String, String)> {
    golden_key_refs()
        .into_iter()
        .map(|g| {
            let key = build_cache_key(&g.input).expect("golden ref input must be valid");
            (g.description.to_owned(), key.digest_hex)
        })
        .collect()
}

/// Verify that current key generation matches a set of previously captured
/// golden digests. Returns a list of mismatches (empty = all stable).
#[must_use]
pub fn verify_golden_keys(golden: &[(String, String)]) -> Vec<String> {
    let current = capture_golden_keys();
    let mut mismatches = Vec::new();

    if golden.len() != current.len() {
        mismatches.push(format!(
            "GOLDEN_REF_COUNT_MISMATCH: snapshot_entries={}, current_refs={}",
            golden.len(),
            current.len()
        ));
    }

    for (index, ((desc, expected), (current_desc, actual))) in
        golden.iter().zip(current.iter()).enumerate()
    {
        if desc != current_desc {
            mismatches.push(format!(
                "GOLDEN_REF_DESCRIPTION_MISMATCH[{index}]: snapshot={desc}, current={current_desc}"
            ));
        }
        if expected != actual {
            mismatches.push(format!(
                "DRIFT[{index}]: {current_desc}: expected={expected}, actual={actual}"
            ));
        }
    }

    for (index, (desc, actual)) in current.iter().enumerate().skip(golden.len()) {
        mismatches.push(format!(
            "MISSING_GOLDEN_REF[{index}]: {desc}: actual={actual}"
        ));
    }

    for (index, (desc, expected)) in golden.iter().enumerate().skip(current.len()) {
        mismatches.push(format!(
            "EXTRA_GOLDEN_REF[{index}]: {desc}: expected={expected}"
        ));
    }

    mismatches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_keys_are_internally_consistent() {
        // Capture, then immediately verify — should always pass.
        let golden = capture_golden_keys();
        let mismatches = verify_golden_keys(&golden);
        assert!(
            mismatches.is_empty(),
            "golden key drift detected: {mismatches:?}"
        );
    }

    #[test]
    fn golden_refs_produce_valid_cache_keys() {
        for g in golden_key_refs() {
            let key = build_cache_key(&g.input).expect("golden ref should produce valid key");
            assert!(
                key.as_string().starts_with("fjx-"),
                "key should have fjx- prefix: {}",
                g.description
            );
            assert_eq!(
                key.digest_hex.len(),
                64,
                "SHA-256 hex digest should be 64 chars: {}",
                g.description
            );
        }
    }

    #[test]
    fn golden_refs_match_hardcoded_digests() {
        // Regression gate: if the cache key format changes, these will fail,
        // signaling that a namespace version bump is needed.
        for g in golden_key_refs() {
            assert!(
                !g.expected_digest_hex.is_empty(),
                "golden ref '{}' has empty expected_digest_hex — must be filled",
                g.description
            );
            let key = build_cache_key(&g.input).expect("golden ref should produce valid key");
            assert_eq!(
                key.digest_hex, g.expected_digest_hex,
                "cache key drift detected for '{}': expected={}, actual={}",
                g.description, g.expected_digest_hex, key.digest_hex
            );
        }
    }

    #[test]
    fn verify_golden_keys_rejects_truncated_snapshots() {
        let mut golden = capture_golden_keys();
        assert!(
            !golden.is_empty(),
            "fixture should contain at least one golden ref"
        );
        let removed = golden.remove(golden.len() - 1);
        let mismatches = verify_golden_keys(&golden);

        assert!(
            mismatches
                .iter()
                .any(|message| message.contains("GOLDEN_REF_COUNT_MISMATCH")),
            "truncated snapshot should report a count mismatch: {mismatches:?}"
        );
        assert!(
            mismatches.iter().any(|message| {
                message.contains("MISSING_GOLDEN_REF") && message.contains(&removed.0)
            }),
            "truncated snapshot should name the missing ref: {mismatches:?}"
        );
    }

    #[test]
    fn verify_golden_keys_rejects_renamed_snapshots() {
        let mut golden = capture_golden_keys();
        let original = golden[0].0.clone();
        golden[0].0 = "renamed cache-key fixture".to_owned();

        let mismatches = verify_golden_keys(&golden);

        assert!(
            mismatches.iter().any(|message| {
                message.contains("GOLDEN_REF_DESCRIPTION_MISMATCH[0]")
                    && message.contains(&original)
            }),
            "renamed snapshot should report the description mismatch: {mismatches:?}"
        );
    }

    #[test]
    fn verify_golden_keys_rejects_extra_snapshots() {
        let mut golden = capture_golden_keys();
        golden.push(("retired cache-key fixture".to_owned(), "0".repeat(64)));

        let mismatches = verify_golden_keys(&golden);

        assert!(
            mismatches
                .iter()
                .any(|message| message.contains("EXTRA_GOLDEN_REF")),
            "extra snapshot should be rejected: {mismatches:?}"
        );
    }

    #[test]
    fn distinct_golden_refs_produce_distinct_keys() {
        let keys = capture_golden_keys();
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(
                    keys[i].1, keys[j].1,
                    "golden refs should produce distinct keys: '{}' vs '{}'",
                    keys[i].0, keys[j].0
                );
            }
        }
    }
}
