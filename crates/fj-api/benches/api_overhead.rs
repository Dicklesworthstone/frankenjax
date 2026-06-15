use criterion::{Criterion, criterion_group, criterion_main};
use fj_api::{compose, grad, jit, value_and_grad, vmap};
use fj_core::{
    Atom, Equation, Jaxpr, Literal, Primitive, ProgramSpec, Transform, Value, VarId, build_program,
};

fn build_deep_value_and_grad_jaxpr(node_count: usize) -> Jaxpr {
    let mut equations = Vec::with_capacity(node_count.saturating_mul(2));
    let mut current = VarId(1);
    let mut next_var_id = 2u32;

    for _ in 0..node_count {
        let squared = VarId(next_var_id);
        next_var_id += 1;
        equations.push(Equation {
            primitive: Primitive::Mul,
            inputs: vec![Atom::Var(current), Atom::Var(current)].into(),
            outputs: vec![squared].into(),
            params: std::collections::BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        });

        let shifted = VarId(next_var_id);
        next_var_id += 1;
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: vec![Atom::Var(squared), Atom::Lit(Literal::from_f64(1.0))].into(),
            outputs: vec![shifted].into(),
            params: std::collections::BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        });

        current = shifted;
    }

    Jaxpr::new(vec![VarId(1)], vec![], vec![current], equations)
}

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

    let repeated_jit = jit(build_program(ProgramSpec::Add2));
    group.bench_function("jit/scalar_add_repeated_call", |b| {
        b.iter(|| {
            repeated_jit
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
// 3. value_and_grad Runtime Efficiency (shared forward vs separate calls)
// ---------------------------------------------------------------------------

fn bench_value_and_grad_shared_vs_separate(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_and_grad_runtime");
    let input = Value::scalar_f64(1.5);

    let baseline_jaxpr = build_program(ProgramSpec::SquarePlusLinear);
    let shared = value_and_grad(baseline_jaxpr.clone());
    let separate_value = jit(baseline_jaxpr.clone());
    let separate_grad = grad(baseline_jaxpr);

    group.bench_function("shared/square_plus_linear", |b| {
        b.iter(|| {
            shared
                .call(vec![input.clone()])
                .expect("shared value_and_grad should succeed");
        });
    });

    group.bench_function("separate/square_plus_linear", |b| {
        b.iter(|| {
            separate_value
                .call(vec![input.clone()])
                .expect("separate value call should succeed");
            separate_grad
                .call(vec![input.clone()])
                .expect("separate grad call should succeed");
        });
    });

    let deep_jaxpr = build_deep_value_and_grad_jaxpr(100);
    let deep_shared = value_and_grad(deep_jaxpr.clone());
    let deep_separate_value = jit(deep_jaxpr.clone());
    let deep_separate_grad = grad(deep_jaxpr);

    group.bench_function("shared/deep_100_nodes", |b| {
        b.iter(|| {
            deep_shared
                .call(vec![input.clone()])
                .expect("shared deep value_and_grad should succeed");
        });
    });

    group.bench_function("separate/deep_100_nodes", |b| {
        b.iter(|| {
            deep_separate_value
                .call(vec![input.clone()])
                .expect("separate deep value call should succeed");
            deep_separate_grad
                .call(vec![input.clone()])
                .expect("separate deep grad call should succeed");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Transform Composition Overhead via API
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
// 5. Mode Configuration Overhead
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
// 6. Prepared dispatch metadata: same-binary A/B
//
// Isolates the lever for frankenjax-1yn9y. Both arms run the identical deep
// value_and_grad dispatch through `dispatch_ref`; the only difference is whether
// the args-independent composition proof + cache key are precomputed once
// (`prepared = Some`) or rederived on every call (`prepared = None`). A single
// binary measures both so the ratio is trustworthy (no cross-invocation drift).
// ---------------------------------------------------------------------------

fn bench_prepared_dispatch_metadata(c: &mut Criterion) {
    use fj_dispatch::{DispatchRequestRef, dispatch_ref, prepare_dispatch_meta};

    let mut group = c.benchmark_group("prepared_dispatch_metadata");
    let input = Value::scalar_f64(1.5);
    let jaxpr = build_deep_value_and_grad_jaxpr(100);
    let transforms = [Transform::Grad];
    let evidence: Vec<String> = vec!["fj-api-grad-0".to_owned()];
    let mut compile_options = std::collections::BTreeMap::new();
    compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
    let backend = "cpu";
    let mode = fj_core::CompatibilityMode::Strict;
    let features: Vec<String> = Vec::new();

    let meta = prepare_dispatch_meta(
        mode,
        &jaxpr,
        &transforms,
        &evidence,
        backend,
        &compile_options,
        None,
        &features,
    )
    .expect("prepare dispatch meta");

    group.bench_function("recompute/deep_100_nodes", |b| {
        b.iter(|| {
            dispatch_ref(DispatchRequestRef {
                mode,
                root_jaxpr: &jaxpr,
                transform_stack: &transforms,
                transform_evidence: &evidence,
                args: vec![input.clone()],
                backend,
                compile_options: compile_options.clone(),
                custom_hook: None,
                unknown_incompatible_features: &features,
                prepared: None,
            })
            .expect("recompute dispatch");
        });
    });

    group.bench_function("prepared/deep_100_nodes", |b| {
        b.iter(|| {
            dispatch_ref(DispatchRequestRef {
                mode,
                root_jaxpr: &jaxpr,
                transform_stack: &transforms,
                transform_evidence: &evidence,
                args: vec![input.clone()],
                backend,
                compile_options: compile_options.clone(),
                custom_hook: None,
                unknown_incompatible_features: &features,
                prepared: Some(&meta),
            })
            .expect("prepared dispatch");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 7. Repeated JIT compiled evaluator cache: same-binary A/B
//
// Isolates the lever for frankenjax-so4wo. The dispatch arm is the old repeated
// JIT route after args-independent metadata is already prepared. The API arm is
// a warmed `JitWrapped` call that reuses the compiled dense scalar evaluator.
// Both run Add2 with the same transform evidence and must return 7.
// ---------------------------------------------------------------------------

fn bench_jit_compiled_eval_cache(c: &mut Criterion) {
    use fj_dispatch::{DispatchRequestRef, dispatch_ref, prepare_dispatch_meta};

    let mut group = c.benchmark_group("jit_compiled_eval_cache");
    let jaxpr = build_program(ProgramSpec::Add2);
    let transforms = [Transform::Jit];
    let evidence: Vec<String> = vec!["fj-api-jit-0".to_owned()];
    let compile_options = std::collections::BTreeMap::new();
    let backend = "cpu";
    let mode = fj_core::CompatibilityMode::Strict;
    let features: Vec<String> = Vec::new();
    let meta = prepare_dispatch_meta(
        mode,
        &jaxpr,
        &transforms,
        &evidence,
        backend,
        &compile_options,
        None,
        &features,
    )
    .expect("prepare dispatch meta");

    let wrapped = jit(jaxpr.clone());
    wrapped
        .call(vec![Value::scalar_i64(3), Value::scalar_i64(4)])
        .expect("warm compiled jit");

    group.bench_function("dispatch_prepared/scalar_add", |b| {
        b.iter(|| {
            let response = dispatch_ref(DispatchRequestRef {
                mode,
                root_jaxpr: &jaxpr,
                transform_stack: &transforms,
                transform_evidence: &evidence,
                args: vec![Value::scalar_i64(3), Value::scalar_i64(4)],
                backend,
                compile_options: compile_options.clone(),
                custom_hook: None,
                unknown_incompatible_features: &features,
                prepared: Some(&meta),
            })
            .expect("prepared dispatch");
            assert_eq!(response.outputs, vec![Value::scalar_i64(7)]);
        });
    });

    group.bench_function("api_compiled/scalar_add", |b| {
        b.iter(|| {
            let outputs = wrapped
                .call(vec![Value::scalar_i64(3), Value::scalar_i64(4)])
                .expect("compiled jit");
            assert_eq!(outputs, vec![Value::scalar_i64(7)]);
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
    bench_value_and_grad_shared_vs_separate,
    bench_api_composition,
    bench_api_mode_config,
    bench_prepared_dispatch_metadata,
    bench_jit_compiled_eval_cache,
);
criterion_main!(api_benches);
