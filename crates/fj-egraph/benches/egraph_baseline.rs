use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{Atom, Equation, Jaxpr, Primitive, VarId};
use fj_egraph::{OptimizationConfig, jaxpr_to_egraph, optimize_jaxpr, optimize_jaxpr_with_config};
use smallvec::smallvec;
use std::collections::BTreeMap;

fn build_redundant_add_zero() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(1)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Lit(fj_core::Literal::from_f64(0.0))],
            outputs: smallvec![VarId(1)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    )
}

fn build_redundant_mul_one() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(1)],
        vec![Equation {
            primitive: Primitive::Mul,
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Lit(fj_core::Literal::from_f64(1.0))],
            outputs: smallvec![VarId(1)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    )
}

fn build_double_neg() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(2)],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
        ],
    )
}

fn build_exp_log_chain() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(2)],
        vec![
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
        ],
    )
}

fn build_complex_polynomial() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(6)],
        vec![
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(2)), Atom::Lit(fj_core::Literal::from_f64(0.0))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(fj_core::Literal::from_f64(1.0))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(4))],
                outputs: smallvec![VarId(5)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(5)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(6)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
        ],
    )
}

fn bench_optimize_add_zero(c: &mut Criterion) {
    let jaxpr = build_redundant_add_zero();
    c.bench_function("egraph/optimize_add_zero", |b| {
        b.iter(|| optimize_jaxpr(&jaxpr))
    });
}

fn bench_optimize_mul_one(c: &mut Criterion) {
    let jaxpr = build_redundant_mul_one();
    c.bench_function("egraph/optimize_mul_one", |b| {
        b.iter(|| optimize_jaxpr(&jaxpr))
    });
}

fn bench_optimize_double_neg(c: &mut Criterion) {
    let jaxpr = build_double_neg();
    c.bench_function("egraph/optimize_double_neg", |b| {
        b.iter(|| optimize_jaxpr(&jaxpr))
    });
}

fn bench_optimize_exp_log(c: &mut Criterion) {
    let jaxpr = build_exp_log_chain();
    c.bench_function("egraph/optimize_exp_log", |b| {
        b.iter(|| optimize_jaxpr(&jaxpr))
    });
}

fn bench_optimize_complex_poly(c: &mut Criterion) {
    let jaxpr = build_complex_polynomial();
    c.bench_function("egraph/optimize_complex_poly", |b| {
        b.iter(|| optimize_jaxpr(&jaxpr))
    });
}

fn bench_optimize_safe_config(c: &mut Criterion) {
    let jaxpr = build_complex_polynomial();
    let config = OptimizationConfig::safe();
    c.bench_function("egraph/optimize_safe_config", |b| {
        b.iter(|| optimize_jaxpr_with_config(&jaxpr, &config))
    });
}

fn bench_optimize_aggressive_config(c: &mut Criterion) {
    let jaxpr = build_complex_polynomial();
    let config = OptimizationConfig::aggressive();
    c.bench_function("egraph/optimize_aggressive_config", |b| {
        b.iter(|| optimize_jaxpr_with_config(&jaxpr, &config))
    });
}

fn bench_jaxpr_to_egraph(c: &mut Criterion) {
    let jaxpr = build_complex_polynomial();
    c.bench_function("egraph/jaxpr_to_egraph", |b| {
        b.iter(|| jaxpr_to_egraph(&jaxpr))
    });
}

criterion_group!(
    benches,
    bench_optimize_add_zero,
    bench_optimize_mul_one,
    bench_optimize_double_neg,
    bench_optimize_exp_log,
    bench_optimize_complex_poly,
    bench_optimize_safe_config,
    bench_optimize_aggressive_config,
    bench_jaxpr_to_egraph,
);
criterion_main!(benches);
