use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use fj_cache::{
    CacheKeyInput, CacheKeyInputRef, CacheManager, build_cache_key, build_cache_key_ref,
};
use fj_core::{CompatibilityMode, Jaxpr, ProgramSpec, Transform, build_program};
use std::collections::BTreeMap;

fn empty_jaxpr() -> Jaxpr {
    Jaxpr::new(vec![], vec![], vec![], vec![])
}

fn bench_build_cache_key_empty(c: &mut Criterion) {
    let input = CacheKeyInput {
        mode: CompatibilityMode::Strict,
        backend: "cpu".to_owned(),
        jaxpr: empty_jaxpr(),
        transform_stack: vec![],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    };

    c.bench_function("cache_key/owned/empty_program", |b| {
        b.iter(|| build_cache_key(black_box(&input)).unwrap())
    });
}

fn bench_build_cache_key_10eqn(c: &mut Criterion) {
    let input = CacheKeyInput {
        mode: CompatibilityMode::Strict,
        backend: "cpu".to_owned(),
        jaxpr: build_program(ProgramSpec::SquarePlusLinear),
        transform_stack: vec![Transform::Jit, Transform::Grad],
        compile_options: BTreeMap::new(),
        custom_hook: Some("hook".to_owned()),
        unknown_incompatible_features: vec![],
    };

    c.bench_function("cache_key/owned/10eqn_program", |b| {
        b.iter(|| build_cache_key(black_box(&input)).unwrap())
    });
}

fn bench_build_cache_key_ref_10eqn(c: &mut Criterion) {
    let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
    let transforms = vec![Transform::Jit, Transform::Grad];
    let compile_options = BTreeMap::new();
    let unknown: Vec<String> = vec![];

    let input = CacheKeyInputRef {
        mode: CompatibilityMode::Strict,
        backend: "cpu",
        jaxpr: &jaxpr,
        transform_stack: &transforms,
        compile_options: &compile_options,
        custom_hook: Some("hook"),
        unknown_incompatible_features: &unknown,
    };

    c.bench_function("cache_key/streaming/10eqn_program", |b| {
        b.iter(|| build_cache_key_ref(black_box(&input)).unwrap())
    });
}

fn bench_cache_hit_lookup(c: &mut Criterion) {
    let mut mgr = CacheManager::in_memory();
    let input = CacheKeyInput {
        mode: CompatibilityMode::Strict,
        backend: "cpu".to_owned(),
        jaxpr: build_program(ProgramSpec::Add2),
        transform_stack: vec![Transform::Jit],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    };
    let key = build_cache_key(&input).unwrap();
    mgr.put(&key, vec![0u8; 64]);

    c.bench_function("cache_lookup/hit/in_memory", |b| {
        b.iter(|| mgr.get(black_box(&key)))
    });
}

fn bench_cache_miss_lookup(c: &mut Criterion) {
    let mgr = CacheManager::in_memory();
    let key = fj_cache::CacheKey {
        namespace: "fjx",
        digest_hex: "nonexistent_key_hash".to_owned(),
    };

    c.bench_function("cache_lookup/miss/in_memory", |b| {
        b.iter(|| mgr.get(black_box(&key)))
    });
}

fn bench_compatibility_matrix_row(c: &mut Criterion) {
    let input = CacheKeyInput {
        mode: CompatibilityMode::Hardened,
        backend: "cpu".to_owned(),
        jaxpr: empty_jaxpr(),
        transform_stack: vec![],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec!["feat_a".to_owned()],
    };

    c.bench_function("compatibility_matrix_row", |b| {
        b.iter(|| fj_cache::compatibility_matrix_row(black_box(&input)))
    });
}

criterion_group!(
    benches,
    bench_build_cache_key_empty,
    bench_build_cache_key_10eqn,
    bench_build_cache_key_ref_10eqn,
    bench_cache_hit_lookup,
    bench_cache_miss_lookup,
    bench_compatibility_matrix_row,
);
criterion_main!(benches);
