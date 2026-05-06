use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId};
use smallvec::smallvec;
use std::collections::BTreeMap;

fn build_simple_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0), VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    )
}

fn build_large_jaxpr() -> Jaxpr {
    let mut equations = Vec::new();
    let mut next_var = 2_u32;

    for i in 0..50 {
        equations.push(Equation {
            primitive: match i % 4 {
                0 => Primitive::Add,
                1 => Primitive::Mul,
                2 => Primitive::Sub,
                _ => Primitive::Div,
            },
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
            outputs: smallvec![VarId(next_var)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        });
        next_var += 1;
    }

    Jaxpr::new(
        vec![VarId(0), VarId(1)],
        vec![],
        vec![VarId(next_var - 1)],
        equations,
    )
}

fn bench_jaxpr_clone_simple(c: &mut Criterion) {
    let jaxpr = build_simple_jaxpr();
    c.bench_function("core/jaxpr_clone_simple", |b| {
        b.iter(|| jaxpr.clone())
    });
}

fn bench_jaxpr_clone_large(c: &mut Criterion) {
    let jaxpr = build_large_jaxpr();
    c.bench_function("core/jaxpr_clone_large", |b| {
        b.iter(|| jaxpr.clone())
    });
}

fn bench_jaxpr_fingerprint_simple(c: &mut Criterion) {
    let jaxpr = build_simple_jaxpr();
    c.bench_function("core/jaxpr_fingerprint_simple", |b| {
        b.iter(|| jaxpr.canonical_fingerprint())
    });
}

fn bench_jaxpr_fingerprint_large(c: &mut Criterion) {
    let jaxpr = build_large_jaxpr();
    c.bench_function("core/jaxpr_fingerprint_large", |b| {
        b.iter(|| jaxpr.canonical_fingerprint())
    });
}

fn bench_jaxpr_validate_simple(c: &mut Criterion) {
    let jaxpr = build_simple_jaxpr();
    c.bench_function("core/jaxpr_validate_simple", |b| {
        b.iter(|| jaxpr.validate_well_formed())
    });
}

fn bench_jaxpr_validate_large(c: &mut Criterion) {
    let jaxpr = build_large_jaxpr();
    c.bench_function("core/jaxpr_validate_large", |b| {
        b.iter(|| jaxpr.validate_well_formed())
    });
}

fn bench_tensor_value_new(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..1000).map(|i| Literal::from_f64(i as f64)).collect();
    let shape = Shape { dims: vec![10, 100] };
    c.bench_function("core/tensor_value_new_1k", |b| {
        b.iter(|| TensorValue::new(DType::F64, shape.clone(), elements.clone()))
    });
}

fn bench_value_scalar_f64(c: &mut Criterion) {
    c.bench_function("core/value_scalar_f64", |b| {
        b.iter(|| Value::scalar_f64(3.14159))
    });
}

fn bench_value_vector_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    c.bench_function("core/value_vector_f64_100", |b| {
        b.iter(|| Value::vector_f64(&data))
    });
}

fn bench_shape_element_count(c: &mut Criterion) {
    let shape = Shape { dims: vec![10, 20, 30, 40] };
    c.bench_function("core/shape_element_count", |b| {
        b.iter(|| shape.element_count())
    });
}

criterion_group!(
    benches,
    bench_jaxpr_clone_simple,
    bench_jaxpr_clone_large,
    bench_jaxpr_fingerprint_simple,
    bench_jaxpr_fingerprint_large,
    bench_jaxpr_validate_simple,
    bench_jaxpr_validate_large,
    bench_tensor_value_new,
    bench_value_scalar_f64,
    bench_value_vector_f64,
    bench_shape_element_count,
);
criterion_main!(benches);
