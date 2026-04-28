use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{
    Atom, Equation, Jaxpr, Literal, Primitive, ProgramSpec, Value, VarId, build_program,
};
use fj_interpreters::eval_jaxpr;
use fj_interpreters::partial_eval::{dce_jaxpr, partial_eval_jaxpr};
use fj_interpreters::staging::{execute_staged, stage_jaxpr};
use std::collections::BTreeMap;
use std::hint::black_box;

// ── Helpers ──────────────────────────────────────────────────────────

fn build_chain_jaxpr(n: usize) -> Jaxpr {
    let mut equations = Vec::with_capacity(n);
    for i in 0..n {
        let input_var = VarId((i + 1) as u32);
        let output_var = VarId((i + 2) as u32);
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![Atom::Var(input_var), Atom::Lit(Literal::I64(1))],
            outputs: smallvec::smallvec![output_var],
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

/// { a, b -> c = neg(a); d = mul(c, b) -> d }
fn make_neg_mul_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec::smallvec![Atom::Var(VarId(1))],
                outputs: smallvec::smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                outputs: smallvec::smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn make_scan_body_add_emit_carry_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3), VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec::smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![Atom::Var(VarId(3)), Atom::Lit(Literal::I64(0))],
                outputs: smallvec::smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn make_scan_sub_jaxpr_eval_jaxpr(reverse: bool) -> Jaxpr {
    let params = if reverse {
        BTreeMap::from([("reverse".to_owned(), "true".to_owned())])
    } else {
        BTreeMap::new()
    };
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3), VarId(4)],
        vec![Equation {
            primitive: Primitive::Scan,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3), VarId(4)],
            params,
            effects: vec![],
            sub_jaxprs: vec![make_scan_body_add_emit_carry_jaxpr()],
        }],
    )
}

// ── 1. Partial Eval Benchmarks ──────────────────────────────────────

fn bench_partial_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_eval");

    // PE of 10-equation chain, all-known (constant folding)
    group.bench_function("all_known/10eq", |b| {
        let jaxpr = build_chain_jaxpr(10);
        b.iter(|| {
            partial_eval_jaxpr(&jaxpr, &[false]).expect("PE should succeed");
        });
    });

    // PE of 10-equation chain, all-unknown
    group.bench_function("all_unknown/10eq", |b| {
        let jaxpr = build_chain_jaxpr(10);
        b.iter(|| {
            partial_eval_jaxpr(&jaxpr, &[true]).expect("PE should succeed");
        });
    });

    // PE of 10-equation chain, 50% known (mixed)
    group.bench_function("mixed/neg_mul", |b| {
        let jaxpr = make_neg_mul_jaxpr();
        b.iter(|| {
            partial_eval_jaxpr(&jaxpr, &[false, true]).expect("PE should succeed");
        });
    });

    // PE scaling: 100 equations
    group.bench_function("all_known/100eq", |b| {
        let jaxpr = build_chain_jaxpr(100);
        b.iter(|| {
            partial_eval_jaxpr(&jaxpr, &[false]).expect("PE should succeed");
        });
    });

    // PE scaling: 1000 equations
    group.bench_function("all_known/1000eq", |b| {
        let jaxpr = build_chain_jaxpr(1000);
        b.iter(|| {
            partial_eval_jaxpr(&jaxpr, &[false]).expect("PE should succeed");
        });
    });

    // Program spec PE
    group.bench_function("program_spec/square_plus_linear", |b| {
        let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
        b.iter(|| {
            partial_eval_jaxpr(&jaxpr, &[false]).expect("PE should succeed");
        });
    });

    group.finish();
}

// ── 2. Dead Code Elimination Benchmarks ─────────────────────────────

fn bench_dce(c: &mut Criterion) {
    let mut group = c.benchmark_group("dce");

    group.bench_function("all_used/10eq", |b| {
        let jaxpr = build_chain_jaxpr(10);
        b.iter(|| {
            dce_jaxpr(&jaxpr, &[true]);
        });
    });

    group.bench_function("all_used/100eq", |b| {
        let jaxpr = build_chain_jaxpr(100);
        b.iter(|| {
            dce_jaxpr(&jaxpr, &[true]);
        });
    });

    group.bench_function("all_used/1000eq", |b| {
        let jaxpr = build_chain_jaxpr(1000);
        b.iter(|| {
            dce_jaxpr(&jaxpr, &[true]);
        });
    });

    group.finish();
}

// ── 3. Staging Pipeline Benchmarks ──────────────────────────────────

fn bench_staging(c: &mut Criterion) {
    let mut group = c.benchmark_group("staging");

    // Full staging pipeline: PE + eval known + execute unknown
    group.bench_function("full_pipeline/neg_mul_mixed", |b| {
        let jaxpr = make_neg_mul_jaxpr();
        b.iter(|| {
            let staged = stage_jaxpr(&jaxpr, &[false, true], &[Value::scalar_i64(5)])
                .expect("stage should succeed");
            execute_staged(&staged, &[Value::scalar_i64(3)]).expect("execute should succeed")
        });
    });

    // All-known staging
    group.bench_function("full_pipeline/all_known_add2", |b| {
        let jaxpr = build_program(ProgramSpec::Add2);
        b.iter(|| {
            let staged = stage_jaxpr(
                &jaxpr,
                &[false, false],
                &[Value::scalar_i64(2), Value::scalar_i64(3)],
            )
            .expect("stage should succeed");
            execute_staged(&staged, &[]).expect("execute should succeed")
        });
    });

    // Staging 100-equation chain
    group.bench_function("full_pipeline/chain_100eq", |b| {
        let jaxpr = build_chain_jaxpr(100);
        b.iter(|| {
            let staged = stage_jaxpr(&jaxpr, &[false], &[Value::scalar_i64(0)])
                .expect("stage should succeed");
            execute_staged(&staged, &[]).expect("execute should succeed")
        });
    });

    group.finish();
}

// ── 4. Interpreter Eval Benchmarks ──────────────────────────────────

fn bench_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval");

    group.bench_function("scan_sub_jaxpr_add_emit_128", |b| {
        let jaxpr = make_scan_sub_jaxpr_eval_jaxpr(false);
        let xs_values: Vec<i64> = (0..128).collect();
        let inputs = vec![
            Value::scalar_i64(0),
            Value::vector_i64(&xs_values).expect("xs vector should build"),
        ];
        b.iter(|| {
            black_box(
                eval_jaxpr(black_box(&jaxpr), black_box(&inputs))
                    .expect("scan eval should succeed"),
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_partial_eval,
    bench_dce,
    bench_staging,
    bench_eval
);
criterion_main!(benches);
