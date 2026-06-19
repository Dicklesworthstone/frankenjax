//! Gauntlet bench for the dense complex constructor (bead frankenjax-7eqrs):
//! lax.complex(re, im) -> Complex128 (the FFT/signal real+imag combine). The dense
//! path zips the two f64 slices into packed (re,im) storage via new_complex_values;
//! the boxed path materializes a Vec<Literal> per element.
//!
//! Arm A: dense inputs -> dense complex constructor (committed).
//! Arm B: boxed inputs -> per-Literal path (pre-7eqrs).
//! JAX head-to-head: benchmarks/jax_comparison/complex_ctor_gauntlet.py.

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

fn bench_complex_ctor(c: &mut Criterion) {
    let re: Vec<f64> = (0..N).map(|i| (i as f64) * 1e-6 - 0.5).collect();
    let im: Vec<f64> = (0..N).map(|i| (i as f64) * 2e-6 + 0.25).collect();
    let dense = [f64_dense(&re), f64_dense(&im)];
    let boxed = [f64_boxed(&re), f64_boxed(&im)];
    let params = BTreeMap::new();

    // Sanity: dense path == boxed path.
    let d = eval_primitive(Primitive::Complex, &dense, &params).unwrap();
    let b = eval_primitive(Primitive::Complex, &boxed, &params).unwrap();
    if let (Value::Tensor(dt), Value::Tensor(bt)) = (&d, &b) {
        for i in [0usize, N / 2, N - 1] {
            assert_eq!(dt.elements[i], bt.elements[i], "dense != boxed complex ctor");
        }
    }

    let mut group = c.benchmark_group("complex_ctor_re_im_to_c128_1m");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("dense", |bn| {
        bn.iter(|| black_box(eval_primitive(Primitive::Complex, black_box(&dense), black_box(&params)).unwrap()));
    });
    group.bench_function("boxed", |bn| {
        bn.iter(|| black_box(eval_primitive(Primitive::Complex, black_box(&boxed), black_box(&params)).unwrap()));
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_complex_ctor
}
criterion_main!(benches);
