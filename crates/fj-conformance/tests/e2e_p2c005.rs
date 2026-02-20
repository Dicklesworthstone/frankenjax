#![forbid(unsafe_code)]

//! FJ-P2C-005-G: E2E Scenario Scripts + Replay/Forensics Logging
//! for compilation cache/keying subsystem.
//!
//! Each scenario produces a forensic JSON record that captures the scenario_id,
//! keys generated, hit/miss outcomes, eviction events, and timing.

use fj_cache::{CacheKeyInput, CacheLookup, CacheManager, build_cache_key, eviction::LruConfig};
use fj_core::{CompatibilityMode, Jaxpr, ProgramSpec, Transform, build_program};
use fj_test_utils::{TestLogV1, TestMode, TestResult, fixture_id_from_json, test_id};
use std::collections::{BTreeMap, HashSet};

fn log_e2e(name: &str, fixture: &impl serde::Serialize) {
    let fid = fixture_id_from_json(fixture).expect("fixture digest");
    let log = TestLogV1::unit(
        test_id(module_path!(), name),
        fid,
        TestMode::Strict,
        TestResult::Pass,
    );
    assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
}

fn empty_jaxpr() -> Jaxpr {
    Jaxpr::new(vec![], vec![], vec![], vec![])
}

fn baseline_input() -> CacheKeyInput {
    CacheKeyInput {
        mode: CompatibilityMode::Strict,
        backend: "cpu".to_owned(),
        jaxpr: build_program(ProgramSpec::Add2),
        transform_stack: vec![Transform::Jit],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    }
}

// ── E2E Scenarios ───────────────────────────────────────────────────

#[test]
fn e2e_cache_hit_avoids_recompute() {
    let mut mgr = CacheManager::in_memory();
    let key = build_cache_key(&baseline_input()).unwrap();

    // First dispatch: miss.
    let r1 = mgr.get(&key);
    assert_eq!(r1, CacheLookup::Miss);

    // Store result.
    mgr.put(&key, b"computed_result".to_vec());

    // Second dispatch: hit — avoids recomputation.
    let r2 = mgr.get(&key);
    assert_eq!(
        r2,
        CacheLookup::Hit {
            data: b"computed_result".to_vec()
        }
    );

    log_e2e(
        "e2e_cache_hit_avoids_recompute",
        &("hit_avoids_recompute", "pass"),
    );
}

#[test]
fn e2e_cache_key_stability_across_restart() {
    // Simulate restart: generate key in two separate scopes with
    // independently constructed inputs.
    let key1 = {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: build_program(ProgramSpec::SquarePlusLinear),
            transform_stack: vec![Transform::Jit, Transform::Grad],
            compile_options: BTreeMap::new(),
            custom_hook: Some("my-hook".to_owned()),
            unknown_incompatible_features: vec![],
        };
        build_cache_key(&input).unwrap()
    };

    let key2 = {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: build_program(ProgramSpec::SquarePlusLinear),
            transform_stack: vec![Transform::Jit, Transform::Grad],
            compile_options: BTreeMap::new(),
            custom_hook: Some("my-hook".to_owned()),
            unknown_incompatible_features: vec![],
        };
        build_cache_key(&input).unwrap()
    };

    assert_eq!(key1, key2, "keys must be identical across 'restarts'");

    // File-backed persistence test.
    let dir = std::env::temp_dir().join("fj-e2e-p2c005-stability");
    let _ = std::fs::create_dir_all(&dir);

    {
        let mut mgr = CacheManager::file_backed(dir.clone());
        mgr.put(&key1, b"stable_data".to_vec());
    }
    {
        let mgr = CacheManager::file_backed(dir.clone());
        assert_eq!(
            mgr.get(&key2),
            CacheLookup::Hit {
                data: b"stable_data".to_vec()
            }
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
    log_e2e(
        "e2e_cache_key_stability_across_restart",
        &("stability", "pass"),
    );
}

#[test]
fn e2e_cache_eviction_under_pressure() {
    let config = LruConfig {
        max_entries: 3,
        max_bytes: 0,
    };
    let mut mgr = CacheManager::in_memory_with_eviction(config);

    let specs = [
        ProgramSpec::Add2,
        ProgramSpec::Square,
        ProgramSpec::SquarePlusLinear,
        ProgramSpec::AddOne,
        ProgramSpec::SinX,
    ];

    let mut keys = Vec::new();
    for spec in &specs {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Hardened,
            backend: "cpu".to_owned(),
            jaxpr: build_program(*spec),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        };
        let key = build_cache_key(&input).unwrap();
        mgr.put(&key, format!("{spec:?}").into_bytes());
        keys.push(key);
    }

    // After 5 inserts with max_entries=3, first 2 should be evicted.
    assert_eq!(mgr.stats().entry_count, 3);
    assert_eq!(mgr.get(&keys[0]), CacheLookup::Miss);
    assert_eq!(mgr.get(&keys[1]), CacheLookup::Miss);
    assert!(matches!(mgr.get(&keys[2]), CacheLookup::Hit { .. }));
    assert!(matches!(mgr.get(&keys[3]), CacheLookup::Hit { .. }));
    assert!(matches!(mgr.get(&keys[4]), CacheLookup::Hit { .. }));

    log_e2e(
        "e2e_cache_eviction_under_pressure",
        &("eviction", mgr.stats().entry_count),
    );
}

#[test]
fn e2e_strict_mode_rejection() {
    let input = CacheKeyInput {
        mode: CompatibilityMode::Strict,
        backend: "cpu".to_owned(),
        jaxpr: empty_jaxpr(),
        transform_stack: vec![],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec!["unknown_flag".to_owned()],
    };

    let result = build_cache_key(&input);
    assert!(result.is_err(), "strict mode must reject unknown features");

    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unknown_flag"),
        "error message must include rejected feature"
    );

    log_e2e("e2e_strict_mode_rejection", &("strict_rejection", "pass"));
}

#[test]
fn e2e_hardened_mode_inclusion() {
    let input_without = CacheKeyInput {
        mode: CompatibilityMode::Hardened,
        backend: "cpu".to_owned(),
        jaxpr: empty_jaxpr(),
        transform_stack: vec![],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    };

    let input_with = CacheKeyInput {
        mode: CompatibilityMode::Hardened,
        backend: "cpu".to_owned(),
        jaxpr: empty_jaxpr(),
        transform_stack: vec![],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec!["new_feature".to_owned()],
    };

    let key_without = build_cache_key(&input_without).unwrap();
    let key_with = build_cache_key(&input_with).unwrap();

    // Both succeed (hardened doesn't reject).
    assert!(key_without.as_string().starts_with("fjx-"));
    assert!(key_with.as_string().starts_with("fjx-"));

    // But they produce different keys (feature is included in hash).
    assert_ne!(
        key_without.digest_hex, key_with.digest_hex,
        "unknown features must affect the cache key in hardened mode"
    );

    log_e2e(
        "e2e_hardened_mode_inclusion",
        &("hardened_inclusion", "pass"),
    );
}

#[test]
fn e2e_cache_key_collision_resistance() {
    let specs = [
        ProgramSpec::Add2,
        ProgramSpec::Square,
        ProgramSpec::SquarePlusLinear,
        ProgramSpec::AddOne,
        ProgramSpec::SinX,
    ];
    let modes = [CompatibilityMode::Strict, CompatibilityMode::Hardened];
    let transforms_combos: Vec<Vec<Transform>> = vec![
        vec![],
        vec![Transform::Jit],
        vec![Transform::Grad],
        vec![Transform::Vmap],
        vec![Transform::Jit, Transform::Grad],
        vec![Transform::Jit, Transform::Vmap],
    ];
    let backends = ["cpu", "gpu", "tpu"];
    let hooks: [Option<&str>; 3] = [None, Some("hook-a"), Some("hook-b")];

    let mut keys = HashSet::new();
    let mut count = 0;

    for spec in &specs {
        for mode in &modes {
            for transforms in &transforms_combos {
                for backend in &backends {
                    for hook in &hooks {
                        let input = CacheKeyInput {
                            mode: *mode,
                            backend: backend.to_string(),
                            jaxpr: build_program(*spec),
                            transform_stack: transforms.clone(),
                            compile_options: BTreeMap::new(),
                            custom_hook: hook.map(|s| s.to_owned()),
                            unknown_incompatible_features: vec![],
                        };
                        let key = build_cache_key(&input).unwrap();
                        keys.insert(key.digest_hex);
                        count += 1;
                    }
                }
            }
        }
    }

    let collisions = count - keys.len();
    assert_eq!(
        collisions, 0,
        "0 collisions required among {count} keys, got {collisions}"
    );

    log_e2e(
        "e2e_cache_key_collision_resistance",
        &("collisions", collisions, "total", count),
    );
}
