//! Gauntlet bench for the dense elementwise-binary path (the foundational op:
//! residual adds, scaling, every elementwise expression). Dense same-shape add/mul
//! over the typed slice (broadcast_fold_contiguous_inner (1,1) case) vs the boxed
//! per-Literal path. Tests whether the library's bread-and-butter op ties JAX, and
//! whether the near-parity residual (store/alloc) is universal for bandwidth-bound
//! elementwise.
//!
//! Arm A: dense inputs -> dense binary path. Arm B: boxed inputs -> per-Literal.
//! JAX head-to-head: benchmarks/jax_comparison/elementwise_gauntlet.py.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use fj_core::{DType, Literal, LiteralBuffer, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

const N: usize = 1_048_576;

fn vshape(len: usize) -> Shape {
    Shape::vector(u32::try_from(len).unwrap())
}
fn f64_dense(v: &[f64]) -> Value {
    Value::Tensor(TensorValue::new_f64_values(vshape(v.len()), v.to_vec()).unwrap())
}
fn f64_boxed(v: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::F64,
            vshape(v.len()),
            LiteralBuffer::new(v.iter().map(|&x| Literal::from_f64(x)).collect()),
        )
        .unwrap(),
    )
}
fn f32_dense(v: &[f32]) -> Value {
    Value::Tensor(TensorValue::new_f32_values(vshape(v.len()), v.to_vec()).unwrap())
}
fn f32_boxed(v: &[f32]) -> Value {
    Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::F32,
            vshape(v.len()),
            LiteralBuffer::new(v.iter().map(|&x| Literal::from_f32(x)).collect()),
        )
        .unwrap(),
    )
}

fn bench_op(label: &str, prim: Primitive, dense: [Value; 2], boxed: [Value; 2], c: &mut Criterion) {
    let params = BTreeMap::new();
    let d = eval_primitive(prim, &dense, &params).unwrap();
    let b = eval_primitive(prim, &boxed, &params).unwrap();
    if let (Value::Tensor(dt), Value::Tensor(bt)) = (&d, &b) {
        for i in [0usize, N / 2, N - 1] {
            assert_eq!(dt.elements[i], bt.elements[i], "dense != boxed elementwise");
        }
    }
    let mut group = c.benchmark_group(label);
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("dense", |bn| {
        bn.iter(|| black_box(eval_primitive(prim, black_box(&dense), black_box(&params)).unwrap()));
    });
    group.bench_function("boxed", |bn| {
        bn.iter(|| black_box(eval_primitive(prim, black_box(&boxed), black_box(&params)).unwrap()));
    });
    group.finish();
}

// Same-binary A/B (trustworthy — one invocation) for the f32 add fix: native f32
// `a+b` vs the prior f64-widen `(f64::from(a)+f64::from(b)) as f32`. Bit-identical;
// decides whether native (8-wide) actually beats widen (4-wide) here.
fn bench_f32_add_impl_ab(c: &mut Criterion) {
    let a: Vec<f32> = (0..N).map(|i| (i as f32) * 1e-6 - 0.5).collect();
    let b: Vec<f32> = (0..N).map(|i| (i as f32) * 2e-6 + 0.25).collect();
    // sanity: native == widen bit-for-bit
    for i in [0usize, N / 2, N - 1] {
        let n = a[i] + b[i];
        let w = (f64::from(a[i]) + f64::from(b[i])) as f32;
        assert_eq!(n.to_bits(), w.to_bits(), "native f32 add != widen");
    }
    let mut group = c.benchmark_group("f32_add_impl_ab_1m");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("native_f32", |bn| {
        bn.iter(|| {
            let v: Vec<f32> = black_box(&a).iter().zip(black_box(&b)).map(|(&x, &y)| x + y).collect();
            black_box(v)
        });
    });
    group.bench_function("widen_f64", |bn| {
        bn.iter(|| {
            let v: Vec<f32> = black_box(&a)
                .iter()
                .zip(black_box(&b))
                .map(|(&x, &y)| (f64::from(x) + f64::from(y)) as f32)
                .collect();
            black_box(v)
        });
    });
    group.finish();
}

fn bench_elementwise(c: &mut Criterion) {
    let a64: Vec<f64> = (0..N).map(|i| (i as f64) * 1e-6 - 0.5).collect();
    let b64: Vec<f64> = (0..N).map(|i| (i as f64) * 2e-6 + 0.25).collect();
    let a32: Vec<f32> = (0..N).map(|i| (i as f32) * 1e-6 - 0.5).collect();
    let b32: Vec<f32> = (0..N).map(|i| (i as f32) * 2e-6 + 0.25).collect();
    bench_op("add_f64_1m", Primitive::Add, [f64_dense(&a64), f64_dense(&b64)], [f64_boxed(&a64), f64_boxed(&b64)], c);
    bench_op("add_f32_1m", Primitive::Add, [f32_dense(&a32), f32_dense(&b32)], [f32_boxed(&a32), f32_boxed(&b32)], c);
    bench_op("mul_f64_1m", Primitive::Mul, [f64_dense(&a64), f64_dense(&b64)], [f64_boxed(&a64), f64_boxed(&b64)], c);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_f32_add_impl_ab, bench_elementwise
}
criterion_main!(benches);
