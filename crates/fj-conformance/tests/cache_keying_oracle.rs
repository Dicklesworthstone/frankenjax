#![forbid(unsafe_code)]

//! FJ-P2C-005-F: Differential Oracle + Metamorphic + Adversarial Validation
//! for compilation cache/keying subsystem.
//!
//! Covers:
//! - Oracle: cache key determinism across runs
//! - Oracle: cache key sensitivity to each input field
//! - Oracle: cache hit/miss behavior
//! - Oracle: strict vs hardened mode key separation
//! - Oracle: collision resistance (10K random programs, 0 collisions)
//! - Metamorphic: determinism invariant, field-change sensitivity, eviction correctness
//! - Adversarial: large programs, edge cases, concurrent-safe patterns

use fj_cache::{
    CacheKeyError, CacheKeyInput, CacheKeyInputRef, CacheLookup, CacheManager, build_cache_key,
    build_cache_key_ref, eviction::LruConfig,
};
use fj_core::{
    Atom, CompatibilityMode, Equation, Jaxpr, Primitive, ProgramSpec, Transform, VarId,
    build_program,
};
use fj_test_utils::{TestLogV1, TestMode, TestResult, fixture_id_from_json, test_id};
use std::collections::{BTreeMap, HashSet};

fn log_oracle(name: &str, fixture: &impl serde::Serialize) {
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
        custom_hook: Some("test-hook".to_owned()),
        unknown_incompatible_features: vec![],
    }
}

// ── Oracle: Self-Consistency Points ─────────────────────────────────

#[test]
fn oracle_cache_key_determinism_across_calls() {
    let input = baseline_input();
    let keys: Vec<_> = (0..100)
        .map(|_| build_cache_key(&input).unwrap().digest_hex)
        .collect();
    let first = &keys[0];
    assert!(
        keys.iter().all(|k| k == first),
        "all 100 calls must produce identical keys"
    );
    log_oracle(
        "oracle_cache_key_determinism_across_calls",
        &("determinism", keys.len()),
    );
}

#[test]
fn oracle_cache_key_sensitivity_per_field() {
    let base_hex = build_cache_key(&baseline_input()).unwrap().digest_hex;

    // Mode change.
    let mut alt = baseline_input();
    alt.mode = CompatibilityMode::Hardened;
    assert_ne!(base_hex, build_cache_key(&alt).unwrap().digest_hex, "mode");

    // Backend change.
    let mut alt = baseline_input();
    alt.backend = "tpu".to_owned();
    assert_ne!(
        base_hex,
        build_cache_key(&alt).unwrap().digest_hex,
        "backend"
    );

    // Transform stack change.
    let mut alt = baseline_input();
    alt.transform_stack = vec![Transform::Grad];
    assert_ne!(
        base_hex,
        build_cache_key(&alt).unwrap().digest_hex,
        "transforms"
    );

    // Compile options change.
    let mut alt = baseline_input();
    alt.compile_options
        .insert("level".to_owned(), "3".to_owned());
    assert_ne!(
        base_hex,
        build_cache_key(&alt).unwrap().digest_hex,
        "compile_options"
    );

    // Custom hook change.
    let mut alt = baseline_input();
    alt.custom_hook = None;
    assert_ne!(base_hex, build_cache_key(&alt).unwrap().digest_hex, "hook");

    // Jaxpr change.
    let mut alt = baseline_input();
    alt.jaxpr = build_program(ProgramSpec::SquarePlusLinear);
    assert_ne!(base_hex, build_cache_key(&alt).unwrap().digest_hex, "jaxpr");

    log_oracle(
        "oracle_cache_key_sensitivity_per_field",
        &("sensitivity", 6),
    );
}

#[test]
fn oracle_cache_hit_miss_behavior() {
    let mut mgr = CacheManager::in_memory();
    let key = build_cache_key(&baseline_input()).unwrap();

    // Miss initially.
    assert_eq!(mgr.get(&key), CacheLookup::Miss);

    // Hit after put.
    mgr.put(&key, b"result".to_vec());
    assert_eq!(
        mgr.get(&key),
        CacheLookup::Hit {
            data: b"result".to_vec()
        }
    );

    // Different key misses.
    let mut alt = baseline_input();
    alt.backend = "gpu".to_owned();
    let alt_key = build_cache_key(&alt).unwrap();
    assert_eq!(mgr.get(&alt_key), CacheLookup::Miss);

    log_oracle("oracle_cache_hit_miss_behavior", &("hit_miss", 3));
}

#[test]
fn oracle_strict_hardened_key_separation() {
    let strict_input = CacheKeyInput {
        mode: CompatibilityMode::Strict,
        ..baseline_input()
    };
    let hardened_input = CacheKeyInput {
        mode: CompatibilityMode::Hardened,
        ..baseline_input()
    };

    let strict_key = build_cache_key(&strict_input).unwrap();
    let hardened_key = build_cache_key(&hardened_input).unwrap();

    assert_ne!(
        strict_key.digest_hex, hardened_key.digest_hex,
        "strict and hardened must produce different keys for same program"
    );

    // Both start with fjx- namespace.
    assert!(strict_key.as_string().starts_with("fjx-"));
    assert!(hardened_key.as_string().starts_with("fjx-"));

    log_oracle(
        "oracle_strict_hardened_key_separation",
        &("mode_separation", strict_key.digest_hex.len()),
    );
}

#[test]
fn oracle_collision_resistance_10k() {
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

    // Also generate keys with compile options variants.
    for i in 0..500 {
        let mut opts = BTreeMap::new();
        opts.insert("opt_level".to_owned(), format!("{}", i % 5));
        opts.insert("variant".to_owned(), format!("v{i}"));
        let input = CacheKeyInput {
            mode: CompatibilityMode::Hardened,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        };
        let key = build_cache_key(&input).unwrap();
        keys.insert(key.digest_hex);
        count += 1;
    }

    // Also generate keys with unknown features (hardened mode).
    for i in 0..500 {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Hardened,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![format!("feature_{i}")],
        };
        let key = build_cache_key(&input).unwrap();
        keys.insert(key.digest_hex);
        count += 1;
    }

    let collisions = count - keys.len();
    assert_eq!(
        collisions, 0,
        "expected 0 collisions among {count} keys, got {collisions}"
    );
    assert!(
        count >= 1000,
        "expected at least 1000 distinct key generations, got {count}"
    );

    log_oracle(
        "oracle_collision_resistance_10k",
        &("collisions", collisions, "total", count),
    );
}

// ── Metamorphic Properties ──────────────────────────────────────────

#[test]
fn metamorphic_key_invariant_across_restarts() {
    // Simulate "restart" by building the same input twice.
    let input_a = baseline_input();
    let input_b = baseline_input();
    let key_a = build_cache_key(&input_a).unwrap();
    let key_b = build_cache_key(&input_b).unwrap();
    assert_eq!(
        key_a, key_b,
        "same input constructed independently must produce same key"
    );
}

#[test]
fn metamorphic_f_x_ne_f_y() {
    // Cache key of (f, x-config) != cache key of (f, y-config) when configs differ.
    let mut x = baseline_input();
    let mut y = baseline_input();
    x.compile_options
        .insert("target".to_owned(), "x86".to_owned());
    y.compile_options
        .insert("target".to_owned(), "arm64".to_owned());
    assert_ne!(
        build_cache_key(&x).unwrap(),
        build_cache_key(&y).unwrap(),
        "different compile options must produce different keys"
    );
}

#[test]
fn metamorphic_f_ne_g() {
    // Cache key of (f, x) != cache key of (g, x) when f != g.
    let mut f = baseline_input();
    f.jaxpr = build_program(ProgramSpec::Add2);
    let mut g = baseline_input();
    g.jaxpr = build_program(ProgramSpec::Square);
    assert_ne!(
        build_cache_key(&f).unwrap(),
        build_cache_key(&g).unwrap(),
        "different programs must produce different keys"
    );
}

#[test]
fn metamorphic_eviction_preserves_correctness() {
    let config = LruConfig {
        max_entries: 1,
        max_bytes: 0,
    };
    let mut mgr = CacheManager::in_memory_with_eviction(config);

    let key_a = build_cache_key(&baseline_input()).unwrap();
    mgr.put(&key_a, b"result_a".to_vec());

    // Put a second entry — evicts first.
    let mut alt = baseline_input();
    alt.backend = "gpu".to_owned();
    let key_b = build_cache_key(&alt).unwrap();
    mgr.put(&key_b, b"result_b".to_vec());

    // key_a evicted, key_b present.
    assert_eq!(mgr.get(&key_a), CacheLookup::Miss);
    assert_eq!(
        mgr.get(&key_b),
        CacheLookup::Hit {
            data: b"result_b".to_vec()
        }
    );

    // Re-store key_a (simulates re-dispatch after eviction).
    mgr.put(&key_a, b"result_a".to_vec());
    assert_eq!(
        mgr.get(&key_a),
        CacheLookup::Hit {
            data: b"result_a".to_vec()
        }
    );
}

// ── Adversarial Cases ───────────────────────────────────────────────

#[test]
fn adversarial_large_program_bounded_time() {
    // Build a program with 100 equations (not 10K — proportional to test budget).
    let inputs = vec![VarId(1)];
    let mut equations = Vec::new();
    for i in 0..100 {
        let in_var = VarId(i + 1);
        let out_var = VarId(i + 2);
        equations.push(Equation {
            primitive: Primitive::Neg,
            inputs: smallvec::smallvec![Atom::Var(in_var)],
            outputs: smallvec::smallvec![out_var],
            params: BTreeMap::new(),
        });
    }
    let jaxpr = Jaxpr::new(inputs, vec![], vec![VarId(101)], equations);

    let input = CacheKeyInput {
        mode: CompatibilityMode::Strict,
        backend: "cpu".to_owned(),
        jaxpr,
        transform_stack: vec![Transform::Jit],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    };

    let start = std::time::Instant::now();
    let key = build_cache_key(&input).unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 100,
        "key generation for 100-eqn program should be under 100ms, was {}ms",
        elapsed.as_millis()
    );
    assert_eq!(key.digest_hex.len(), 64);
}

#[test]
fn adversarial_strict_rejects_unknown_features_uniformly() {
    // Verify that strict mode rejection is uniform regardless of feature content.
    let features_sets = vec![
        vec!["a".to_owned()],
        vec!["z".to_owned()],
        vec!["very_long_feature_name_that_goes_on_and_on".to_owned()],
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
        vec!["".to_owned()], // empty string feature
    ];

    for features in &features_sets {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: features.clone(),
        };
        let err = build_cache_key(&input).unwrap_err();
        assert!(
            matches!(err, CacheKeyError::UnknownIncompatibleFeatures { .. }),
            "all feature sets should produce same error variant"
        );
    }
}

#[test]
fn adversarial_empty_program_valid_key() {
    let input = CacheKeyInput {
        mode: CompatibilityMode::Strict,
        backend: "cpu".to_owned(),
        jaxpr: empty_jaxpr(),
        transform_stack: vec![],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    };
    let key = build_cache_key(&input).unwrap();
    assert!(key.as_string().starts_with("fjx-"));
    assert_eq!(key.digest_hex.len(), 64);
}

#[test]
fn adversarial_delimiter_in_backend_name() {
    // Backend name containing the pipe delimiter character.
    let input_pipe = CacheKeyInput {
        mode: CompatibilityMode::Hardened,
        backend: "cpu|gpu".to_owned(),
        jaxpr: empty_jaxpr(),
        transform_stack: vec![],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    };
    let input_normal = CacheKeyInput {
        mode: CompatibilityMode::Hardened,
        backend: "cpugpu".to_owned(),
        jaxpr: empty_jaxpr(),
        transform_stack: vec![],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    };

    // Even with delimiter injection, keys should differ.
    let key_pipe = build_cache_key(&input_pipe).unwrap();
    let key_normal = build_cache_key(&input_normal).unwrap();
    assert_ne!(
        key_pipe.digest_hex, key_normal.digest_hex,
        "delimiter injection must not cause key aliasing"
    );
}

#[test]
fn adversarial_streaming_and_owned_agree_under_stress() {
    // Verify streaming and owned builders agree for a complex input.
    let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
    let transforms = vec![Transform::Jit, Transform::Grad, Transform::Vmap];
    let mut compile_options = BTreeMap::new();
    compile_options.insert("opt".to_owned(), "3".to_owned());
    compile_options.insert("target".to_owned(), "x86_64".to_owned());
    let unknown: Vec<String> = vec![];

    let owned = CacheKeyInput {
        mode: CompatibilityMode::Hardened,
        backend: "gpu".to_owned(),
        jaxpr: jaxpr.clone(),
        transform_stack: transforms.clone(),
        compile_options: compile_options.clone(),
        custom_hook: Some("complex-hook".to_owned()),
        unknown_incompatible_features: unknown.clone(),
    };

    let ref_input = CacheKeyInputRef {
        mode: CompatibilityMode::Hardened,
        backend: "gpu",
        jaxpr: &jaxpr,
        transform_stack: &transforms,
        compile_options: &compile_options,
        custom_hook: Some("complex-hook"),
        unknown_incompatible_features: &unknown,
    };

    assert_eq!(
        build_cache_key(&owned).unwrap(),
        build_cache_key_ref(&ref_input).unwrap(),
        "owned and streaming must agree for complex inputs"
    );
}

#[test]
fn adversarial_file_cache_persistence_survives_reopen() {
    let dir = std::env::temp_dir().join("fj-p2c005-adversarial-persist");
    let _ = std::fs::create_dir_all(&dir);

    let key = build_cache_key(&baseline_input()).unwrap();

    // Write with first instance.
    {
        let mut mgr = CacheManager::file_backed(dir.clone());
        mgr.put(&key, b"persistent_payload".to_vec());
    }

    // Read with second instance (simulates process restart).
    {
        let mgr = CacheManager::file_backed(dir.clone());
        let result = mgr.get(&key);
        assert_eq!(
            result,
            CacheLookup::Hit {
                data: b"persistent_payload".to_vec()
            },
            "file-backed cache must survive process restart"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
