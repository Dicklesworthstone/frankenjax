use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{DType, Shape};
use fj_trace::{ShapedArray, make_jaxpr};

fn scalar_f64() -> ShapedArray {
    ShapedArray {
        dtype: DType::F64,
        shape: Shape { dims: vec![] },
    }
}

fn vector_f64(len: u32) -> ShapedArray {
    ShapedArray {
        dtype: DType::F64,
        shape: Shape { dims: vec![len] },
    }
}

fn bench_make_jaxpr_identity(c: &mut Criterion) {
    let in_avals = vec![scalar_f64()];
    c.bench_function("trace/make_jaxpr_identity", |b| {
        b.iter(|| {
            make_jaxpr(|inputs| vec![inputs[0].clone()], in_avals.clone())
        })
    });
}

fn bench_make_jaxpr_add(c: &mut Criterion) {
    let in_avals = vec![scalar_f64(), scalar_f64()];
    c.bench_function("trace/make_jaxpr_add", |b| {
        b.iter(|| {
            make_jaxpr(|inputs| vec![&inputs[0] + &inputs[1]], in_avals.clone())
        })
    });
}

fn bench_make_jaxpr_mul(c: &mut Criterion) {
    let in_avals = vec![scalar_f64(), scalar_f64()];
    c.bench_function("trace/make_jaxpr_mul", |b| {
        b.iter(|| {
            make_jaxpr(|inputs| vec![&inputs[0] * &inputs[1]], in_avals.clone())
        })
    });
}

fn bench_make_jaxpr_polynomial(c: &mut Criterion) {
    let in_avals = vec![scalar_f64()];
    c.bench_function("trace/make_jaxpr_poly_x2+x", |b| {
        b.iter(|| {
            make_jaxpr(
                |inputs| {
                    let x = &inputs[0];
                    let x2 = x * x;
                    vec![&x2 + x]
                },
                in_avals.clone(),
            )
        })
    });
}

fn bench_make_jaxpr_chain(c: &mut Criterion) {
    let in_avals = vec![scalar_f64(), scalar_f64()];
    c.bench_function("trace/make_jaxpr_chain_5ops", |b| {
        b.iter(|| {
            make_jaxpr(
                |inputs| {
                    let a = &inputs[0];
                    let b = &inputs[1];
                    let c = a + b;
                    let d = &c * a;
                    let e = &d + b;
                    let f = &e * &c;
                    vec![&f + &d]
                },
                in_avals.clone(),
            )
        })
    });
}

fn bench_make_jaxpr_vector(c: &mut Criterion) {
    let in_avals = vec![vector_f64(100), vector_f64(100)];
    c.bench_function("trace/make_jaxpr_vector_add", |b| {
        b.iter(|| {
            make_jaxpr(|inputs| vec![&inputs[0] + &inputs[1]], in_avals.clone())
        })
    });
}

criterion_group!(
    benches,
    bench_make_jaxpr_identity,
    bench_make_jaxpr_add,
    bench_make_jaxpr_mul,
    bench_make_jaxpr_polynomial,
    bench_make_jaxpr_chain,
    bench_make_jaxpr_vector,
);
criterion_main!(benches);
