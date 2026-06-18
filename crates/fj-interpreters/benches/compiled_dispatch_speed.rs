//! Compiled-dispatch (CompiledJaxpr) vs eager `eval_jaxpr` on interpreter-bound workloads.
//!
//! Scalar `Add` chains are DISPATCH-bound — the per-equation kernel is trivial, so the
//! cost is the interpreter tax (slot env setup, per-equation `eval_primitive` dispatch,
//! and `BTreeMap<String,String>` param handling). This isolates exactly what the dense
//! compiled plan targets, so it quantifies the existing `compile_jaxpr_for_repeated_eval`
//! win over per-call `eval_jaxpr`, and BASELINES the tensor-param-prescan lever
//! (frankenjax-6dfew): re-run this bench before/after that change to measure it.
//!
//! Bit-exactness of compiled-vs-eager is guarded by the unit test
//! `compiled_jaxpr_eval_matches_eager_eval_jaxpr`; this file only measures speed.
use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, Value, VarId};
use fj_interpreters::{compile_jaxpr_for_repeated_eval, eval_jaxpr};
use std::collections::BTreeMap;
use std::hint::black_box;

/// `x -> x+1 -> x+2 -> ... ` : an n-equation Add chain. The added literal is `lit`, so
/// passing an I64 lit + scalar arg gives a pure-scalar chain, and an F64 lit + f64-vector
/// arg gives a small-TENSOR elementwise-broadcast chain (dense binary — the op NOT yet
/// pre-scanned in DenseEvalPlan, so it profiles the remaining dispatch gap for 6dfew).
fn build_chain_jaxpr(n: usize, lit: Literal) -> Jaxpr {
    let mut equations = Vec::with_capacity(n);
    for i in 0..n {
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![Atom::Var(VarId((i + 1) as u32)), Atom::Lit(lit)],
            outputs: smallvec::smallvec![VarId((i + 2) as u32)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
    }
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId((n + 1) as u32)],
        equations,
    )
}

fn bench_one(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
             tag: &str, jaxpr: &Jaxpr, args: &[Value]) {
    group.bench_function(format!("eager/{tag}"), |b| {
        b.iter(|| eval_jaxpr(black_box(jaxpr), black_box(args)).unwrap())
    });
    // Skip the compiled arm rather than panic if a workload is outside the dense subset.
    if let Some(compiled) = compile_jaxpr_for_repeated_eval(jaxpr) {
        group.bench_function(format!("compiled/{tag}"), |b| {
            b.iter(|| compiled.eval(black_box(args)).unwrap())
        });
    }
}

fn bench_compiled_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_dispatch");
    // Scalar Add chains: dispatch-bound, trivial kernel — pure interpreter tax.
    let scalar_args = [Value::scalar_i64(0)];
    for &n in &[8usize, 32, 128] {
        let jaxpr = build_chain_jaxpr(n, Literal::I64(1));
        bench_one(&mut group, &format!("scalar/n={n}"), &jaxpr, &scalar_args);
    }
    // Small-tensor f64 elementwise-broadcast chains: dense binary is NOT pre-scanned in
    // DenseEvalPlan, so this profiles the remaining per-call dispatch gap (frankenjax-6dfew).
    let tensor_args = [Value::vector_f64(&[1.0_f64; 64]).expect("vector_f64")];
    for &n in &[8usize, 32] {
        let jaxpr = build_chain_jaxpr(n, Literal::from_f64(1.0));
        bench_one(&mut group, &format!("tensor64/n={n}"), &jaxpr, &tensor_args);
    }
    group.finish();
}

criterion_group!(benches, bench_compiled_dispatch);
criterion_main!(benches);
