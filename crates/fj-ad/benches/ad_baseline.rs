use criterion::{Criterion, criterion_group, criterion_main};
use fj_ad::{grad_jaxpr, jvp, value_and_grad_jaxpr};
use fj_core::{Atom, Equation, Jaxpr, Primitive, Value, VarId};
use smallvec::smallvec;
use std::collections::BTreeMap;

fn build_square_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(1)],
        vec![Equation {
            primitive: Primitive::Mul,
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(0))],
            outputs: smallvec![VarId(1)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    )
}

fn build_poly_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(4)],
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
                inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
        ],
    )
}

fn build_trig_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(3)],
        vec![
            Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Cos,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
        ],
    )
}

fn build_exp_log_jaxpr() -> Jaxpr {
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

fn bench_grad_square(c: &mut Criterion) {
    let jaxpr = build_square_jaxpr();
    let args = vec![Value::scalar_f64(3.0)];
    c.bench_function("ad/grad_square", |b| {
        b.iter(|| grad_jaxpr(&jaxpr, &args))
    });
}

fn bench_grad_polynomial(c: &mut Criterion) {
    let jaxpr = build_poly_jaxpr();
    let args = vec![Value::scalar_f64(2.0)];
    c.bench_function("ad/grad_poly_x3+x2+x", |b| {
        b.iter(|| grad_jaxpr(&jaxpr, &args))
    });
}

fn bench_grad_trig(c: &mut Criterion) {
    let jaxpr = build_trig_jaxpr();
    let args = vec![Value::scalar_f64(1.0)];
    c.bench_function("ad/grad_sin_cos_mul", |b| {
        b.iter(|| grad_jaxpr(&jaxpr, &args))
    });
}

fn bench_grad_exp_log(c: &mut Criterion) {
    let jaxpr = build_exp_log_jaxpr();
    let args = vec![Value::scalar_f64(1.0)];
    c.bench_function("ad/grad_exp_log", |b| {
        b.iter(|| grad_jaxpr(&jaxpr, &args))
    });
}

fn bench_value_and_grad_poly(c: &mut Criterion) {
    let jaxpr = build_poly_jaxpr();
    let args = vec![Value::scalar_f64(2.0)];
    c.bench_function("ad/value_and_grad_poly", |b| {
        b.iter(|| value_and_grad_jaxpr(&jaxpr, &args))
    });
}

fn bench_jvp_square(c: &mut Criterion) {
    let jaxpr = build_square_jaxpr();
    let primals = vec![Value::scalar_f64(3.0)];
    let tangents = vec![Value::scalar_f64(1.0)];
    c.bench_function("ad/jvp_square", |b| {
        b.iter(|| jvp(&jaxpr, &primals, &tangents))
    });
}

fn bench_jvp_polynomial(c: &mut Criterion) {
    let jaxpr = build_poly_jaxpr();
    let primals = vec![Value::scalar_f64(2.0)];
    let tangents = vec![Value::scalar_f64(1.0)];
    c.bench_function("ad/jvp_poly_x3+x2+x", |b| {
        b.iter(|| jvp(&jaxpr, &primals, &tangents))
    });
}

fn bench_jvp_trig(c: &mut Criterion) {
    let jaxpr = build_trig_jaxpr();
    let primals = vec![Value::scalar_f64(1.0)];
    let tangents = vec![Value::scalar_f64(1.0)];
    c.bench_function("ad/jvp_sin_cos_mul", |b| {
        b.iter(|| jvp(&jaxpr, &primals, &tangents))
    });
}

criterion_group!(
    benches,
    bench_grad_square,
    bench_grad_polynomial,
    bench_grad_trig,
    bench_grad_exp_log,
    bench_value_and_grad_poly,
    bench_jvp_square,
    bench_jvp_polynomial,
    bench_jvp_trig,
);
criterion_main!(benches);
