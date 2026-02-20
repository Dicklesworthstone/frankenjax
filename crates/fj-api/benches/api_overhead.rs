use criterion::{Criterion, criterion_group, criterion_main};
use fj_api::{compose, grad, jit, value_and_grad, vmap};
use fj_core::{ProgramSpec, Transform, Value, build_program};

// ---------------------------------------------------------------------------
// 1. API Entry Point Overhead (individual transforms)
// ---------------------------------------------------------------------------

fn bench_api_jit_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("api_overhead");

    group.bench_function("jit/scalar_add", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Add2);
            jit(jaxpr)
                .call(vec![Value::scalar_i64(3), Value::scalar_i64(4)])
                .expect("jit should succeed");
        });
    });

    group.bench_function("grad/scalar_square", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Square);
            grad(jaxpr)
                .call(vec![Value::scalar_f64(3.0)])
                .expect("grad should succeed");
        });
    });

    group.bench_function("vmap/vector_add_one", |b| {
        let vec_arg = Value::vector_i64(&[1, 2, 3, 4, 5]).expect("vector");
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::AddOne);
            vmap(jaxpr)
                .call(vec![vec_arg.clone()])
                .expect("vmap should succeed");
        });
    });

    group.bench_function("value_and_grad/scalar_square", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Square);
            value_and_grad(jaxpr)
                .call(vec![Value::scalar_f64(3.0)])
                .expect("value_and_grad should succeed");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. API Wrapper vs Raw Dispatch (isolate wrapper overhead)
// ---------------------------------------------------------------------------

fn bench_api_vs_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("api_vs_dispatch");

    // API path
    group.bench_function("api_jit_add", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Add2);
            jit(jaxpr)
                .call(vec![Value::scalar_i64(2), Value::scalar_i64(5)])
                .expect("jit");
        });
    });

    // Direct dispatch path (bypassing fj-api wrappers)
    group.bench_function("dispatch_jit_add", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Add2);
            let mut ledger = fj_core::TraceTransformLedger::new(jaxpr);
            ledger.push_transform(Transform::Jit, "bench-jit-0".to_owned());
            fj_dispatch::dispatch(fj_dispatch::DispatchRequest {
                mode: fj_core::CompatibilityMode::Strict,
                ledger,
                args: vec![Value::scalar_i64(2), Value::scalar_i64(5)],
                backend: "cpu".to_owned(),
                compile_options: std::collections::BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            })
            .expect("dispatch");
        });
    });

    // API grad path
    group.bench_function("api_grad_square", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Square);
            grad(jaxpr)
                .call(vec![Value::scalar_f64(5.0)])
                .expect("grad");
        });
    });

    // Direct dispatch grad path
    group.bench_function("dispatch_grad_square", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Square);
            let mut ledger = fj_core::TraceTransformLedger::new(jaxpr);
            ledger.push_transform(Transform::Grad, "bench-grad-0".to_owned());
            fj_dispatch::dispatch(fj_dispatch::DispatchRequest {
                mode: fj_core::CompatibilityMode::Strict,
                ledger,
                args: vec![Value::scalar_f64(5.0)],
                backend: "cpu".to_owned(),
                compile_options: std::collections::BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            })
            .expect("dispatch");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Transform Composition Overhead via API
// ---------------------------------------------------------------------------

fn bench_api_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("api_composition");

    group.bench_function("jit_grad/builder", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Square);
            jit(jaxpr)
                .compose_grad()
                .call(vec![Value::scalar_f64(3.0)])
                .expect("jit(grad)");
        });
    });

    group.bench_function("jit_grad/compose_helper", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Square);
            compose(jaxpr, vec![Transform::Jit, Transform::Grad])
                .call(vec![Value::scalar_f64(3.0)])
                .expect("compose");
        });
    });

    group.bench_function("jit_vmap/builder", |b| {
        let vec_arg = Value::vector_i64(&[1, 2, 3]).expect("vector");
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::AddOne);
            jit(jaxpr)
                .compose_vmap()
                .call(vec![vec_arg.clone()])
                .expect("jit(vmap)");
        });
    });

    group.bench_function("vmap_grad/builder", |b| {
        let vec_arg = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Square);
            vmap(jaxpr)
                .compose_grad()
                .call(vec![vec_arg.clone()])
                .expect("vmap(grad)");
        });
    });

    group.bench_function("jit_vmap_grad/compose_helper", |b| {
        let vec_arg = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Square);
            compose(
                jaxpr,
                vec![Transform::Jit, Transform::Vmap, Transform::Grad],
            )
            .call(vec![vec_arg.clone()])
            .expect("compose 3-deep");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Mode Configuration Overhead
// ---------------------------------------------------------------------------

fn bench_api_mode_config(c: &mut Criterion) {
    let mut group = c.benchmark_group("api_mode_config");

    group.bench_function("strict_jit", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Add2);
            jit(jaxpr)
                .call(vec![Value::scalar_i64(1), Value::scalar_i64(2)])
                .expect("strict jit");
        });
    });

    group.bench_function("hardened_jit", |b| {
        b.iter(|| {
            let jaxpr = build_program(ProgramSpec::Add2);
            jit(jaxpr)
                .with_mode(fj_core::CompatibilityMode::Hardened)
                .call(vec![Value::scalar_i64(1), Value::scalar_i64(2)])
                .expect("hardened jit");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    api_benches,
    bench_api_jit_scalar,
    bench_api_vs_dispatch,
    bench_api_composition,
    bench_api_mode_config,
);
criterion_main!(api_benches);
