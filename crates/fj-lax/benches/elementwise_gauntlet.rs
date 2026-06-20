//! Gauntlet bench for the dense elementwise-binary path (the foundational op:
//! residual adds, scaling, every elementwise expression). Dense same-shape add/mul
//! over the typed slice (broadcast_fold_contiguous_inner (1,1) case) vs the boxed
//! per-Literal path. Tests whether the library's bread-and-butter op ties JAX, and
//! whether the near-parity residual (store/alloc) is universal for bandwidth-bound
//! elementwise.
//!
//! Arm A: dense inputs -> dense binary path. Arm B: boxed inputs -> per-Literal.
//! JAX head-to-head: benchmarks/jax_comparison/elementwise_gauntlet.py.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fj_core::{DType, Literal, LiteralBuffer, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

const N: usize = 1_048_576;
const DRAM_N: usize = 16_777_216;

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

fn bench_op(
    label: &str,
    len: usize,
    prim: Primitive,
    dense: [Value; 2],
    boxed: [Value; 2],
    c: &mut Criterion,
) {
    let params = BTreeMap::new();
    let d = eval_primitive(prim, &dense, &params).unwrap();
    let b = eval_primitive(prim, &boxed, &params).unwrap();
    if let (Value::Tensor(dt), Value::Tensor(bt)) = (&d, &b) {
        for i in [0usize, len / 2, len - 1] {
            assert_eq!(
                dt.elements.get(i),
                bt.elements.get(i),
                "dense != boxed elementwise"
            );
        }
    }
    let mut group = c.benchmark_group(label);
    group.throughput(Throughput::Elements(len as u64));
    group.bench_function("dense", |bn| {
        bn.iter(|| black_box(eval_primitive(prim, black_box(&dense), black_box(&params)).unwrap()));
    });
    group.bench_function("boxed", |bn| {
        bn.iter(|| black_box(eval_primitive(prim, black_box(&boxed), black_box(&params)).unwrap()));
    });
    group.finish();
}

fn bench_op3(
    label: &str,
    len: usize,
    prim: Primitive,
    dense: [Value; 3],
    boxed: [Value; 3],
    c: &mut Criterion,
) {
    let params = BTreeMap::new();
    let d = eval_primitive(prim, &dense, &params).unwrap();
    let b = eval_primitive(prim, &boxed, &params).unwrap();
    if let (Value::Tensor(dt), Value::Tensor(bt)) = (&d, &b) {
        for i in [0usize, len / 2, len - 1] {
            assert_eq!(
                dt.elements.get(i),
                bt.elements.get(i),
                "dense != boxed elementwise ternary"
            );
        }
    }
    let mut group = c.benchmark_group(label);
    group.throughput(Throughput::Elements(len as u64));
    group.bench_function("dense", |bn| {
        bn.iter(|| black_box(eval_primitive(prim, black_box(&dense), black_box(&params)).unwrap()));
    });
    group.bench_function("boxed", |bn| {
        bn.iter(|| black_box(eval_primitive(prim, black_box(&boxed), black_box(&params)).unwrap()));
    });
    group.finish();
}

fn bench_dense_dram_op(label: &str, prim: Primitive, inputs: [Value; 2], c: &mut Criterion) {
    let params = BTreeMap::new();
    let result = eval_primitive(prim, &inputs, &params).unwrap();
    if let Value::Tensor(t) = &result {
        assert_eq!(t.elements.len(), DRAM_N);
        assert_eq!(t.shape.dims, vec![DRAM_N as u32]);
    } else {
        panic!("elementwise DRAM bench must produce tensor");
    }
    let mut group = c.benchmark_group(label);
    group.throughput(Throughput::Elements(DRAM_N as u64));
    group.bench_function("dense", |bn| {
        bn.iter(|| {
            black_box(eval_primitive(prim, black_box(&inputs), black_box(&params)).unwrap())
        });
    });
    group.finish();
}

fn all_core_64k_threads(elems: usize) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    (elems / (1 << 16)).clamp(1, cores)
}

fn cap8_1m_threads(elems: usize) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    (elems / (1 << 20)).clamp(1, cores.min(8))
}

fn threaded_f64_add(a: &[f64], b: &[f64], threads: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![0.0f64; n];
    let threads = threads.max(1);
    let chunk = n.div_ceil(threads);
    std::thread::scope(|scope| {
        for ((blk, a_blk), b_blk) in out
            .chunks_mut(chunk)
            .zip(a.chunks(chunk))
            .zip(b.chunks(chunk))
        {
            scope.spawn(move || {
                for ((o, x), y) in blk.iter_mut().zip(a_blk).zip(b_blk) {
                    *o = *x + *y;
                }
            });
        }
    });
    out
}

fn bench_thread_policy_ab(c: &mut Criterion) {
    let a: Vec<f64> = (0..DRAM_N).map(|i| (i as f64) * 1e-9 - 0.5).collect();
    let b: Vec<f64> = (0..DRAM_N).map(|i| (i as f64) * 2e-9 + 0.25).collect();
    let old_threads = all_core_64k_threads(DRAM_N);
    let cap_threads = cap8_1m_threads(DRAM_N);
    let old = threaded_f64_add(&a, &b, old_threads);
    let cap = threaded_f64_add(&a, &b, cap_threads);
    for i in [0usize, DRAM_N / 2, DRAM_N - 1] {
        match (old.get(i), cap.get(i)) {
            (Some(old_value), Some(cap_value)) => assert_eq!(
                old_value.to_bits(),
                cap_value.to_bits(),
                "thread policy changed value bits"
            ),
            _ => panic!("thread policy sanity index out of range"),
        }
    }

    let mut group = c.benchmark_group("thread_policy_f64_add_16m");
    group.throughput(Throughput::Elements(DRAM_N as u64));
    group.bench_function(format!("all_core_64k_threads_{old_threads}"), |bn| {
        bn.iter(|| black_box(threaded_f64_add(black_box(&a), black_box(&b), old_threads)));
    });
    group.bench_function(format!("cap8_1m_threads_{cap_threads}"), |bn| {
        bn.iter(|| black_box(threaded_f64_add(black_box(&a), black_box(&b), cap_threads)));
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
        match (a.get(i), b.get(i)) {
            (Some(&x), Some(&y)) => {
                let n = x + y;
                let w = (f64::from(x) + f64::from(y)) as f32;
                assert_eq!(n.to_bits(), w.to_bits(), "native f32 add != widen");
            }
            _ => panic!("f32 add sanity index out of range"),
        }
    }
    let mut group = c.benchmark_group("f32_add_impl_ab_1m");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("native_f32", |bn| {
        bn.iter(|| {
            let v: Vec<f32> = black_box(&a)
                .iter()
                .zip(black_box(&b))
                .map(|(&x, &y)| x + y)
                .collect();
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
    let c64: Vec<f64> = (0..N).map(|i| (i as f64) * 3e-6 - 0.125).collect();
    let a32: Vec<f32> = (0..N).map(|i| (i as f32) * 1e-6 - 0.5).collect();
    let b32: Vec<f32> = (0..N).map(|i| (i as f32) * 2e-6 + 0.25).collect();
    let c32: Vec<f32> = (0..N).map(|i| (i as f32) * 3e-6 - 0.125).collect();
    bench_op(
        "add_f64_1m",
        N,
        Primitive::Add,
        [f64_dense(&a64), f64_dense(&b64)],
        [f64_boxed(&a64), f64_boxed(&b64)],
        c,
    );
    bench_op(
        "add_f32_1m",
        N,
        Primitive::Add,
        [f32_dense(&a32), f32_dense(&b32)],
        [f32_boxed(&a32), f32_boxed(&b32)],
        c,
    );
    bench_op(
        "mul_f64_1m",
        N,
        Primitive::Mul,
        [f64_dense(&a64), f64_dense(&b64)],
        [f64_boxed(&a64), f64_boxed(&b64)],
        c,
    );
    bench_op3(
        "fma_f64_1m",
        N,
        Primitive::Fma,
        [f64_dense(&a64), f64_dense(&b64), f64_dense(&c64)],
        [f64_boxed(&a64), f64_boxed(&b64), f64_boxed(&c64)],
        c,
    );
    bench_op3(
        "fma_f32_1m",
        N,
        Primitive::Fma,
        [f32_dense(&a32), f32_dense(&b32), f32_dense(&c32)],
        [f32_boxed(&a32), f32_boxed(&b32), f32_boxed(&c32)],
        c,
    );

    let dram_a64: Vec<f64> = (0..DRAM_N).map(|i| (i as f64) * 1e-9 - 0.5).collect();
    let dram_b64: Vec<f64> = (0..DRAM_N).map(|i| (i as f64) * 2e-9 + 0.25).collect();
    let dram_a32: Vec<f32> = (0..DRAM_N).map(|i| (i as f32) * 1e-9 - 0.5).collect();
    let dram_b32: Vec<f32> = (0..DRAM_N).map(|i| (i as f32) * 2e-9 + 0.25).collect();
    bench_dense_dram_op(
        "add_f64_16m",
        Primitive::Add,
        [f64_dense(&dram_a64), f64_dense(&dram_b64)],
        c,
    );
    bench_dense_dram_op(
        "add_f32_16m",
        Primitive::Add,
        [f32_dense(&dram_a32), f32_dense(&dram_b32)],
        c,
    );
    bench_dense_dram_op(
        "mul_f64_16m",
        Primitive::Mul,
        [f64_dense(&dram_a64), f64_dense(&dram_b64)],
        c,
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_f32_add_impl_ab, bench_thread_policy_ab, bench_elementwise
}
criterion_main!(benches);
