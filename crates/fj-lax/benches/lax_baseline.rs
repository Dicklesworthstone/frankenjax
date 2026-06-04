use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::{eval_primitive, eval_primitive_multi};
use std::collections::BTreeMap;

const LARGE_ELEMENTWISE_LEN: usize = 65_536;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn complex_vector(len: usize) -> Value {
    let elements: Vec<Literal> = (0..len)
        .map(|i| {
            let x = i as f64;
            Literal::from_complex128((x * 0.125).sin(), (x * 0.25).cos())
        })
        .collect();
    Value::Tensor(TensorValue {
        dtype: DType::Complex128,
        shape: Shape {
            dims: vec![len as u32],
        },
        elements: elements.into(),
    })
}

fn complex_matrix(rows: usize, cols: usize) -> Value {
    let elements: Vec<Literal> = (0..rows * cols)
        .map(|i| {
            let x = i as f64;
            Literal::from_complex128((x * 0.125).sin(), (x * 0.25).cos())
        })
        .collect();
    Value::Tensor(TensorValue {
        dtype: DType::Complex128,
        shape: Shape {
            dims: vec![rows as u32, cols as u32],
        },
        elements: elements.into(),
    })
}

fn real_vector(len: usize) -> Value {
    let elements: Vec<Literal> = (0..len)
        .map(|i| {
            let x = i as f64;
            Literal::from_f64((x * 0.125).sin() + (x * 0.03125).cos())
        })
        .collect();
    Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape {
            dims: vec![len as u32],
        },
        elements: elements.into(),
    })
}

fn real_matrix(rows: usize, cols: usize) -> Value {
    let elements: Vec<Literal> = (0..rows * cols)
        .map(|i| {
            let x = i as f64;
            Literal::from_f64((x * 0.125).sin() + (x * 0.03125).cos())
        })
        .collect();
    Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape {
            dims: vec![rows as u32, cols as u32],
        },
        elements: elements.into(),
    })
}

fn bench_add_scalar(c: &mut Criterion) {
    let inputs = [Value::scalar_i64(42), Value::scalar_i64(17)];
    let p = no_params();
    c.bench_function("eval/add_scalar_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &inputs, &p))
    });
}

fn bench_add_1k_vector(c: &mut Criterion) {
    let data: Vec<i64> = (0..1000).collect();
    let lhs = Value::vector_i64(&data).unwrap();
    let rhs = Value::vector_i64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/add_1k_i64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_mul_1k_vector(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let lhs = Value::vector_f64(&data).unwrap();
    let rhs = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/mul_1k_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_add_1k_f64_vector(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let lhs = Value::vector_f64(&data).unwrap();
    let rhs = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/add_1k_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_div_1k_f64_vector(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| (i as f64 + 1.0) * 0.001).collect();
    let lhs = Value::vector_f64(&data).unwrap();
    let rhs = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/div_1k_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Div, &[lhs.clone(), rhs.clone()], &p))
    });
}

// ── Large-array elementwise: quantifies the mcqr.30 data-model gap ──
// `Vec<Literal>` stores each F64 as a 24-byte enum (the Complex128Bits variant
// fixes the size), so a 64k F64 add moves ~3x the bytes of native f64 and the
// per-element match blocks autovectorization. These two benches measure the
// current path against a contiguous-f64 reference (same clone + add work) to
// size the achievable win from dense storage. The reference is bench-local
// (not shipped lib code).
fn bench_add_64k_f64_vec(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| i as f64 * 0.001)
        .collect();
    let lhs = Value::vector_f64(&data).unwrap();
    let rhs = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/add_64k_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_add_64k_f64_dense_reference(c: &mut Criterion) {
    let a: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| i as f64 * 0.001)
        .collect();
    let b: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| i as f64 * 0.001)
        .collect();
    c.bench_function("eval/add_64k_f64_dense_ref", |bencher| {
        // Mirror the eval path's input clone, then a contiguous (autovectorized)
        // f64 add — the achievable cost under a dense storage model.
        bencher.iter(|| {
            let a2 = a.clone();
            let b2 = b.clone();
            a2.iter()
                .zip(b2.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<f64>>()
        })
    });
}

// Large-array i64 elementwise add: the dense-i64-storage analog of the
// add_64k_f64 benches. `vector_i64` now builds dense i64 storage, so the
// same-shape add folds two contiguous i64 slices; the literal-ref input forces
// the Vec<Literal> path. Both run in one process for a same-worker ratio.
fn bench_add_64k_i64_vec(c: &mut Criterion) {
    let data: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64).collect();
    let lhs = Value::vector_i64(&data).unwrap();
    let rhs = Value::vector_i64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/add_64k_i64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_add_64k_i64_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN as i64)
        .map(Literal::I64)
        .collect();
    let make = || {
        Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
                elements.clone(),
            )
            .unwrap(),
        )
    };
    let lhs = make();
    let rhs = make();
    let p = no_params();
    c.bench_function("eval/add_64k_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Dense i64 same-shape Mul + i64 scalar broadcast Mul: the generalized i64
// elementwise fast paths (pass69) beyond the pass67 Add. Each paired with a
// Vec<Literal> reference run in the same process for a same-worker ratio.
fn i64_literal_vec_64k() -> Value {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN as i64)
        .map(Literal::I64)
        .collect();
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    )
}

fn bench_mul_64k_i64_vec(c: &mut Criterion) {
    let data: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64).collect();
    let lhs = Value::vector_i64(&data).unwrap();
    let rhs = Value::vector_i64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/mul_64k_i64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_mul_64k_i64_literal_reference(c: &mut Criterion) {
    let lhs = i64_literal_vec_64k();
    let rhs = i64_literal_vec_64k();
    let p = no_params();
    c.bench_function("eval/mul_64k_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_scalar_mul_64k_i64_vec(c: &mut Criterion) {
    let data: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64).collect();
    let tensor = Value::vector_i64(&data).unwrap();
    let scalar = Value::scalar_i64(3);
    let p = no_params();
    c.bench_function("eval/scalar_mul_64k_i64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[tensor.clone(), scalar.clone()], &p))
    });
}

fn bench_scalar_mul_64k_i64_literal_reference(c: &mut Criterion) {
    let tensor = i64_literal_vec_64k();
    let scalar = Value::scalar_i64(3);
    let p = no_params();
    c.bench_function("eval/scalar_mul_64k_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[tensor.clone(), scalar.clone()], &p))
    });
}

// Dense i64 comparisons (pass70): same-shape Lt + scalar-broadcast Lt, each
// paired with a Vec<Literal> reference run in the same process for a same-worker
// ratio. The scalar reference path runs the heavy generic compare_literals.
fn bench_lt_64k_i64_vec(c: &mut Criterion) {
    let data: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64).collect();
    let lhs = Value::vector_i64(&data).unwrap();
    let rhs = Value::vector_i64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/lt_64k_i64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_lt_64k_i64_literal_reference(c: &mut Criterion) {
    let lhs = i64_literal_vec_64k();
    let rhs = i64_literal_vec_64k();
    let p = no_params();
    c.bench_function("eval/lt_64k_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_lt_scalar_64k_i64_vec(c: &mut Criterion) {
    let data: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64).collect();
    let tensor = Value::vector_i64(&data).unwrap();
    let scalar = Value::scalar_i64(32_768);
    let p = no_params();
    c.bench_function("eval/lt_scalar_64k_i64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[tensor.clone(), scalar.clone()], &p))
    });
}

fn bench_lt_scalar_64k_i64_literal_reference(c: &mut Criterion) {
    let tensor = i64_literal_vec_64k();
    let scalar = Value::scalar_i64(32_768);
    let p = no_params();
    c.bench_function("eval/lt_scalar_64k_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[tensor.clone(), scalar.clone()], &p))
    });
}

fn bench_scalar_mul_1k_f64_vector(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let scalar = Value::scalar_f64(3.5);
    let tensor = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/scalar_mul_1k_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[scalar.clone(), tensor.clone()], &p))
    });
}

fn bench_tensor_sub_scalar_1k_f64_vector(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let tensor = Value::vector_f64(&data).unwrap();
    let scalar = Value::scalar_f64(0.25);
    let p = no_params();
    c.bench_function("eval/tensor_sub_scalar_1k_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sub, &[tensor.clone(), scalar.clone()], &p))
    });
}

fn bench_eq_1k_f64_vector(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let lhs = Value::vector_f64(&data).unwrap();
    let rhs = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/eq_1k_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Eq, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_lt_scalar_1k_f64_vector(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let tensor = Value::vector_f64(&data).unwrap();
    let scalar = Value::scalar_f64(0.5);
    let p = no_params();
    c.bench_function("eval/lt_scalar_1k_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[tensor.clone(), scalar.clone()], &p))
    });
}

fn bench_add_broadcast_bias_1k_f64(c: &mut Criterion) {
    // Bias-add pattern: [256, 4] + [4] -> [256, 4] (1024 output elements).
    let lhs_data: Vec<f64> = (0..1024).map(|i| i as f64 * 0.001).collect();
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![256, 4] },
            lhs_data.into_iter().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    );
    let rhs = Value::vector_f64(&[0.25, -0.5, 1.5, -2.0]).unwrap();
    let p = no_params();
    c.bench_function("eval/add_broadcast_bias_1k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Large multi-dim broadcast (bias pattern [256,256] (+) [256] -> [256,256], 64k
// outputs): the dense BroadcastOdometer fast paths (pass71) vs Vec<Literal>
// references run in the same process for a same-worker ratio. f64 literal hits
// the old materialize+decode f64 path; i64 literal hits the fully generic
// binary_literal_op loop.
const BCAST_N: u32 = 256;

fn bench_add_broadcast_256_f64_vec(c: &mut Criterion) {
    let lhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![BCAST_N, BCAST_N],
            },
            (0..(BCAST_N * BCAST_N) as i64)
                .map(|i| i as f64 * 0.001)
                .collect(),
        )
        .unwrap(),
    );
    let rhs = Value::vector_f64(&(0..BCAST_N).map(|i| i as f64 * 0.5).collect::<Vec<_>>()).unwrap();
    let p = no_params();
    c.bench_function("eval/add_broadcast_256_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_add_broadcast_256_f64_literal_reference(c: &mut Criterion) {
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![BCAST_N, BCAST_N],
            },
            (0..(BCAST_N * BCAST_N))
                .map(|i| Literal::from_f64(i as f64 * 0.001))
                .collect(),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![BCAST_N],
            },
            (0..BCAST_N)
                .map(|i| Literal::from_f64(i as f64 * 0.5))
                .collect(),
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/add_broadcast_256_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_add_broadcast_256_i64_vec(c: &mut Criterion) {
    let lhs = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![BCAST_N, BCAST_N],
            },
            (0..(BCAST_N * BCAST_N) as i64).collect(),
        )
        .unwrap(),
    );
    let rhs = Value::vector_i64(&(0..BCAST_N as i64).collect::<Vec<_>>()).unwrap();
    let p = no_params();
    c.bench_function("eval/add_broadcast_256_i64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_add_broadcast_256_i64_literal_reference(c: &mut Criterion) {
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![BCAST_N, BCAST_N],
            },
            (0..(BCAST_N * BCAST_N) as i64).map(Literal::I64).collect(),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![BCAST_N],
            },
            (0..BCAST_N as i64).map(Literal::I64).collect(),
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/add_broadcast_256_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Large multi-dim broadcast COMPARISON (Lt, bias [256,256] vs [256] -> Bool
// [256,256]): the dense BroadcastOdometer compare fast paths (pass73) vs
// Vec<Literal> references run in the same process. f64/i64 literal refs hit the
// generic per-element compare_literals broadcast loop.
fn bench_lt_broadcast_256_i64_vec(c: &mut Criterion) {
    let lhs = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![BCAST_N, BCAST_N],
            },
            (0..(BCAST_N * BCAST_N) as i64).collect(),
        )
        .unwrap(),
    );
    let rhs = Value::vector_i64(&(0..BCAST_N as i64).map(|i| i * 257).collect::<Vec<_>>()).unwrap();
    let p = no_params();
    c.bench_function("eval/lt_broadcast_256_i64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_lt_broadcast_256_i64_literal_reference(c: &mut Criterion) {
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![BCAST_N, BCAST_N],
            },
            (0..(BCAST_N * BCAST_N) as i64).map(Literal::I64).collect(),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![BCAST_N],
            },
            (0..BCAST_N as i64).map(|i| Literal::I64(i * 257)).collect(),
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/lt_broadcast_256_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_lt_broadcast_256_f64_vec(c: &mut Criterion) {
    let lhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![BCAST_N, BCAST_N],
            },
            (0..(BCAST_N * BCAST_N) as i64)
                .map(|i| i as f64 * 0.001)
                .collect(),
        )
        .unwrap(),
    );
    let rhs = Value::vector_f64(&(0..BCAST_N).map(|i| i as f64 * 0.5).collect::<Vec<_>>()).unwrap();
    let p = no_params();
    c.bench_function("eval/lt_broadcast_256_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_lt_broadcast_256_f64_literal_reference(c: &mut Criterion) {
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![BCAST_N, BCAST_N],
            },
            (0..(BCAST_N * BCAST_N))
                .map(|i| Literal::from_f64(i as f64 * 0.001))
                .collect(),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![BCAST_N],
            },
            (0..BCAST_N)
                .map(|i| Literal::from_f64(i as f64 * 0.5))
                .collect(),
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/lt_broadcast_256_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_nextafter_1k(c: &mut Criterion) {
    let lhs = real_vector(1000);
    let rhs_data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.05).cos()).collect();
    let rhs = Value::vector_f64(&rhs_data).unwrap();
    let inputs = [lhs, rhs];
    let p = no_params();
    c.bench_function("eval/nextafter_1k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Nextafter, &inputs, &p))
    });
}

fn bench_dot_100(c: &mut Criterion) {
    let data: Vec<i64> = (0..100).collect();
    let lhs = Value::vector_i64(&data).unwrap();
    let rhs = Value::vector_i64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/dot_100_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Dot, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_dot_256_matrix_f64(c: &mut Criterion) {
    let lhs = real_matrix(256, 256);
    let rhs = real_matrix(256, 256);
    let inputs = [lhs, rhs];
    let p = no_params();
    c.bench_function("eval/dot_256x256x256_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Dot, &inputs, &p))
    });
}

fn bench_concat_axis1_3x_f64(c: &mut Criterion) {
    // Concatenate three [256, 128] matrices along axis 1 -> [256, 384].
    let mk = |base: f64| {
        Value::Tensor(TensorValue {
            dtype: DType::F64,
            shape: Shape {
                dims: vec![256, 128],
            },
            elements: (0..256 * 128)
                .map(|i| Literal::from_f64(base + i as f64))
                .collect(),
        })
    };
    let a = mk(0.0);
    let b = mk(100_000.0);
    let d = mk(200_000.0);
    let mut p = no_params();
    p.insert("dimension".to_owned(), "1".to_owned());
    c.bench_function("eval/concat_axis1_3x256x128_f64", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::Concatenate,
                &[a.clone(), b.clone(), d.clone()],
                &p,
            )
        })
    });
}

fn bench_concat_axis0_3x_f64(c: &mut Criterion) {
    // Stack three [128, 256] matrices along axis 0 -> [384, 256] (common case).
    let mk = |base: f64| {
        Value::Tensor(TensorValue {
            dtype: DType::F64,
            shape: Shape {
                dims: vec![128, 256],
            },
            elements: (0..128 * 256)
                .map(|i| Literal::from_f64(base + i as f64))
                .collect(),
        })
    };
    let a = mk(0.0);
    let b = mk(100_000.0);
    let d = mk(200_000.0);
    let mut p = no_params();
    p.insert("dimension".to_owned(), "0".to_owned());
    c.bench_function("eval/concat_axis0_3x128x256_f64", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::Concatenate,
                &[a.clone(), b.clone(), d.clone()],
                &p,
            )
        })
    });
}

fn bench_transpose_256x256_f64(c: &mut Criterion) {
    let m = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape {
            dims: vec![256, 256],
        },
        elements: (0..256 * 256)
            .map(|i| Literal::from_f64(i as f64 * 0.001))
            .collect(),
    });
    let mut p = no_params();
    p.insert("permutation".to_owned(), "1,0".to_owned());
    c.bench_function("eval/transpose_256x256_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Transpose, std::slice::from_ref(&m), &p))
    });
}

fn bench_eig_48(c: &mut Criterion) {
    // Real eigendecomposition (Eig) via QR iteration: ~100 iterations, each
    // doing two O(n^3) internal matrix multiplies (matrix_mul / _complex).
    let n = 48usize;
    let data: Vec<f64> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            if i == j {
                (n as f64) + (i as f64)
            } else {
                (((i * 7 + j * 13) % 5) as f64) - 2.0
            }
        })
        .collect();
    let m = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape {
            dims: vec![n as u32, n as u32],
        },
        elements: data.into_iter().map(Literal::from_f64).collect(),
    });
    let p = no_params();
    c.bench_function("linalg/eig_48x48_f64", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::Eig, std::slice::from_ref(&m), &p))
    });
}

fn bench_cholesky_128_f64(c: &mut Criterion) {
    // Real SPD matrix A = B^T B + n*I (diagonally dominant, positive definite).
    let n = 128usize;
    let base: Vec<f64> = (0..n * n)
        .map(|idx| (((idx % 7) as f64) - 3.0) * 0.5)
        .collect();
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += base[k * n + i] * base[k * n + j];
            }
            a[i * n + j] = s + if i == j { n as f64 } else { 0.0 };
        }
    }
    let m = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape {
            dims: vec![n as u32, n as u32],
        },
        elements: a.into_iter().map(Literal::from_f64).collect(),
    });
    let p = no_params();
    c.bench_function("linalg/cholesky_128x128_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cholesky, std::slice::from_ref(&m), &p))
    });
}

fn bench_qr_128_f64(c: &mut Criterion) {
    let n = 128usize;
    let m = real_matrix(n, n);
    let p = no_params();
    c.bench_function("linalg/qr_128x128_f64", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&m), &p))
    });
}

fn bench_lu_128_f64(c: &mut Criterion) {
    let n = 128usize;
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let v = if i == j {
                (n as f64) + (i as f64) * 0.25 + 7.0
            } else {
                (((i * 19 + j * 23) % 13) as f64) * 0.125 - 0.75
            };
            data.push(v);
        }
    }
    let m = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape {
            dims: vec![n as u32, n as u32],
        },
        elements: data.into_iter().map(Literal::from_f64).collect(),
    });
    let p = no_params();
    c.bench_function("linalg/lu_128x128_f64", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::Lu, std::slice::from_ref(&m), &p))
    });
}

fn bench_svd_48_f64(c: &mut Criterion) {
    let n = 48usize;
    let m = real_matrix(n, n);
    let p = no_params();
    c.bench_function("linalg/svd_48x48_f64", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&m), &p))
    });
}

fn bench_matmul_2d_256(c: &mut Criterion) {
    // Public conformance-tested GEMM kernel: 256x256 @ 256x256.
    let (m, k, n) = (256usize, 256usize, 256usize);
    let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 1e-4).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 2e-4).collect();
    c.bench_function("linalg/matmul_2d_256x256x256_f64", |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::matmul_2d(&a, m, k, &b, n))
    });
}

fn bench_matmul_2d_512(c: &mut Criterion) {
    // Public conformance-tested GEMM kernel: 512x512 @ 512x512.
    let (m, k, n) = (512usize, 512usize, 512usize);
    let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 1e-4).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 2e-4).collect();
    c.bench_function("linalg/matmul_2d_512x512x512_f64", |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::matmul_2d(&a, m, k, &b, n))
    });
}

// 2D conv (1x32x32x8 input, 3x3x8x16 kernel, SAME): dense f64 conv path (pass87)
// vs the generic per-multiply Literal path. Same process for a same-worker ratio.
fn conv2d_inputs(dense: bool) -> (Value, Value) {
    let (n, h, w, cin) = (1usize, 32usize, 32usize, 8usize);
    let (kh, kw, cout) = (3usize, 3usize, 16usize);
    let ld = vec![n as u32, h as u32, w as u32, cin as u32];
    let rd = vec![kh as u32, kw as u32, cin as u32, cout as u32];
    let lhs: Vec<f64> = (0..n * h * w * cin)
        .map(|i| ((i as f64) * 0.013).sin())
        .collect();
    let rhs: Vec<f64> = (0..kh * kw * cin * cout)
        .map(|i| ((i as f64) * 0.017).cos())
        .collect();
    if dense {
        (
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: ld }, lhs).unwrap()),
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: rd }, rhs).unwrap()),
        )
    } else {
        (
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: ld },
                    lhs.into_iter().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            ),
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: rd },
                    rhs.into_iter().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            ),
        )
    }
}

fn conv2d_bench_params() -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("padding".to_owned(), "same".to_owned());
    p.insert("strides".to_owned(), "1".to_owned());
    p
}

fn bench_conv2d_32x32x8_3x3x16_f64_vec(c: &mut Criterion) {
    let (lhs, rhs) = conv2d_inputs(true);
    let p = conv2d_bench_params();
    c.bench_function("eval/conv2d_32x32x8_3x3x16_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Conv, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_conv2d_32x32x8_3x3x16_f64_literal_reference(c: &mut Criterion) {
    let (lhs, rhs) = conv2d_inputs(false);
    let p = conv2d_bench_params();
    c.bench_function("eval/conv2d_32x32x8_3x3x16_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Conv, &[lhs.clone(), rhs.clone()], &p))
    });
}

// 1D conv (1x1024x16 input, 5x16x32 kernel, SAME): dense f64 conv_1d path
// (pass88) vs the generic per-multiply Literal path. Same process.
fn conv1d_inputs(dense: bool) -> (Value, Value) {
    let (n, w, cin) = (1usize, 1024usize, 16usize);
    let (kw, cout) = (5usize, 32usize);
    let ld = vec![n as u32, w as u32, cin as u32];
    let rd = vec![kw as u32, cin as u32, cout as u32];
    let lhs: Vec<f64> = (0..n * w * cin)
        .map(|i| ((i as f64) * 0.011).sin())
        .collect();
    let rhs: Vec<f64> = (0..kw * cin * cout)
        .map(|i| ((i as f64) * 0.019).cos())
        .collect();
    if dense {
        (
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: ld }, lhs).unwrap()),
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: rd }, rhs).unwrap()),
        )
    } else {
        (
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: ld },
                    lhs.into_iter().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            ),
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: rd },
                    rhs.into_iter().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            ),
        )
    }
}

fn conv1d_bench_params() -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("padding".to_owned(), "same".to_owned());
    p.insert("strides".to_owned(), "1".to_owned());
    p
}

fn bench_conv1d_1024x16_5x32_f64_vec(c: &mut Criterion) {
    let (lhs, rhs) = conv1d_inputs(true);
    let p = conv1d_bench_params();
    c.bench_function("eval/conv1d_1024x16_5x32_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Conv, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_conv1d_1024x16_5x32_f64_literal_reference(c: &mut Criterion) {
    let (lhs, rhs) = conv1d_inputs(false);
    let p = conv1d_bench_params();
    c.bench_function("eval/conv1d_1024x16_5x32_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Conv, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_solve_24x24_24rhs(c: &mut Criterion) {
    // Multi-RHS linear solve: A (24x24, diagonally dominant => non-singular)
    // and B (24x24). Exercises solve_multi_rhs, which factorizes A once.
    let n = 24usize;
    let m = 24usize;
    let mut a_elems = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let v = if i == j {
                (n as f64) + (i as f64) * 0.5 + 3.0
            } else {
                (((i * 31 + j * 17) % 11) as f64) * 0.25 - 1.0
            };
            a_elems.push(Literal::from_f64(v));
        }
    }
    let a = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape {
            dims: vec![n as u32, n as u32],
        },
        elements: a_elems.into(),
    });
    let b_data: Vec<f64> = (0..n * m)
        .map(|i| ((i * 13 + 1) % 19) as f64 * 0.5 - 4.0)
        .collect();
    let b = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape {
            dims: vec![n as u32, m as u32],
        },
        elements: b_data.into_iter().map(Literal::from_f64).collect(),
    });
    let p = no_params();
    c.bench_function("eval/solve_24x24_24rhs_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Solve, &[a.clone(), b.clone()], &p))
    });
}

// Large-array i64 FULL reduction: unlike the axis reductions (which already pay
// the odometer's index work), the full reduce is a flat fold, so dense i64
// storage removes the dominant per-element Literal::I64 match + 24-byte stride.
fn bench_reduce_sum_64k_i64(c: &mut Criterion) {
    let data: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64).collect();
    let input = Value::vector_i64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/reduce_sum_64k_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_64k_i64_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN as i64)
        .map(Literal::I64)
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/reduce_sum_64k_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// 64k bool ReduceAnd full reduction: dense bool storage fold (pass80) vs the
// Vec<Literal> per-element match, run in the same process for a same-worker ratio.
fn bench_reduce_and_64k_bool_vec(c: &mut Criterion) {
    let data: Vec<bool> = (0..LARGE_ELEMENTWISE_LEN).map(|i| i != 40_000).collect();
    let input = Value::vector_bool(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/reduce_and_64k_bool_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceAnd, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_and_64k_bool_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::Bool(i != 40_000))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/reduce_and_64k_bool_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceAnd, std::slice::from_ref(&input), &p))
    });
}

// 256x256 bool ReduceAnd along axis 1: dense bool storage + out-index odometer
// (pass81) vs the Vec<Literal> per-element flat_to_multi decode + match. Same
// process for a same-worker ratio.
fn bench_reduce_and_256_axis1_bool_vec(c: &mut Criterion) {
    let n = 256usize;
    let data: Vec<bool> = (0..n * n).map(|i| i % 97 != 0).collect();
    let input = Value::Tensor(
        TensorValue::new_bool_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axes".to_owned(), "1".to_owned());
    c.bench_function("eval/reduce_and_256_axis1_bool_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceAnd, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_and_256_axis1_bool_literal_reference(c: &mut Criterion) {
    let n = 256usize;
    let elements: Vec<Literal> = (0..n * n).map(|i| Literal::Bool(i % 97 != 0)).collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            elements,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axes".to_owned(), "1".to_owned());
    c.bench_function("eval/reduce_and_256_axis1_bool_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceAnd, std::slice::from_ref(&input), &p))
    });
}

// 64k f64 Cumsum: dense scan fast path (pass82) vs the Vec<Literal> materialize +
// per-element Literal scan. Same process for a same-worker ratio.
fn bench_cumsum_64k_f64_vec(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| i as f64 * 0.001)
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("axis".to_owned(), "0".to_owned());
    c.bench_function("eval/cumsum_64k_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cumsum, std::slice::from_ref(&input), &p))
    });
}

fn bench_cumsum_64k_f64_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::from_f64(i as f64 * 0.001))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axis".to_owned(), "0".to_owned());
    c.bench_function("eval/cumsum_64k_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cumsum, std::slice::from_ref(&input), &p))
    });
}

// 64k i64 ascending sort: exercises the LSD radix path (pass74) vs the prior
// comparison sort. Pseudo-random keys (negatives, wide range, duplicates) so the
// sort does real work; ascending is the default direction.
fn bench_sort_64k_i64(c: &mut Criterion) {
    let data: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64)
        .map(|i| (i.wrapping_mul(2_654_435_761)).rem_euclid(1_000_003) - 500_000)
        .collect();
    let input = Value::vector_i64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("dimension".to_owned(), "0".to_owned());
    p.insert("descending".to_owned(), "false".to_owned());
    c.bench_function("eval/sort_64k_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &p))
    });
}

// 64k f64 ascending sort: exercises the LSD radix path (pass75, total_cmp-bit
// keys) vs the prior comparison sort. Pseudo-random keys with a few negatives.
fn bench_sort_64k_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| ((i as f64) * 1.000_173).sin() * 1e6 - (i as f64))
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("dimension".to_owned(), "0".to_owned());
    p.insert("descending".to_owned(), "false".to_owned());
    c.bench_function("eval/sort_64k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &p))
    });
}

// ConvertElementType over a 64k dense f64 tensor: dense fast path (pass103,
// reads as_f64_slice) vs the generic per-element Literal-materialize + convert.
fn bench_convert_64k_f64_to_i64(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| ((i as f64) * 1.000_173).sin() * 1e6 - (i as f64))
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "i64".to_owned());
    c.bench_function("eval/convert_64k_f64_to_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ConvertElementType, std::slice::from_ref(&input), &p))
    });
}

// BroadcastInDim: replicate a length-256 f64 vector to [256, 256] (64k output).
// Dense fast path (pass104, new_f64_values) vs generic Vec<Literal> build.
fn bench_broadcast_256_to_256x256_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..256).map(|i| (i as f64) * 0.5 - 3.0).collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("shape".to_owned(), "256,256".to_owned());
    p.insert("broadcast_dimensions".to_owned(), "1".to_owned());
    c.bench_function("eval/broadcast_256_to_256x256_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::BroadcastInDim, std::slice::from_ref(&input), &p))
    });
}

// Pad a 256x256 dense f64 tensor with 1 element of edge padding on each side
// (-> 258x258). Dense fast path (pass105, typed fill + placement into dense
// storage) vs the generic Literal fill + per-element placement.
fn bench_pad_256x256_to_258x258_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..256 * 256).map(|i| (i as f64) * 0.001 - 5.0).collect();
    let operand = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![256, 256],
            },
            data,
        )
        .unwrap(),
    );
    let pad_value = Value::scalar_f64(0.0);
    let mut p = BTreeMap::new();
    p.insert("padding_low".to_owned(), "1,1".to_owned());
    p.insert("padding_high".to_owned(), "1,1".to_owned());
    p.insert("padding_interior".to_owned(), "0,0".to_owned());
    let inputs = [operand, pad_value];
    c.bench_function("eval/pad_256x256_to_258x258_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Pad, &inputs, &p))
    });
}

// Descending f64 sort over a 64k axis: complement-key radix path (pass102) vs
// the generic O(n log n) descending comparison sort.
fn bench_sort_64k_f64_descending(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| ((i as f64) * 1.000_173).sin() * 1e6 - (i as f64))
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("dimension".to_owned(), "0".to_owned());
    p.insert("descending".to_owned(), "true".to_owned());
    c.bench_function("eval/sort_64k_f64_descending", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &p))
    });
}

// Dense f64 Argmax over a 64k axis: dense fast path (pass101) vs the generic
// per-element sort_key/compare_sort_keys scan.
fn bench_argmax_64k_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| ((i as f64) * 1.000_173).sin() * 1e6 - (i as f64))
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/argmax_64k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Argmax, std::slice::from_ref(&input), &p))
    });
}

// F32 sort: previously the generic O(n log n) comparison path (F32 is
// Literal-backed); now the LSD radix path (pass98). Same data shape as the f64
// bench.
fn bench_sort_64k_f32(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::F32Bits((((i as f32) * 1.000_173).sin() * 1e3 - (i as f32)).to_bits()))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape {
                dims: vec![LARGE_ELEMENTWISE_LEN as u32],
            },
            elements,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("dimension".to_owned(), "0".to_owned());
    p.insert("descending".to_owned(), "false".to_owned());
    c.bench_function("eval/sort_64k_f32", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &p))
    });
}

// U32 sort: previously the generic O(n log n) comparison path (U32 is
// Literal-backed); now the LSD radix path (pass99).
fn bench_sort_64k_u32(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::U32((i as u32).wrapping_mul(2_654_435_761) ^ (i as u32)))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::U32,
            Shape {
                dims: vec![LARGE_ELEMENTWISE_LEN as u32],
            },
            elements,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("dimension".to_owned(), "0".to_owned());
    p.insert("descending".to_owned(), "false".to_owned());
    c.bench_function("eval/sort_64k_u32", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &p))
    });
}

// Calibration: an unchanged op (full i64 reduce) used to confirm the radix A/B
// pair ran on the same worker (its ratio should be ~1.0 across the two runs).
fn bench_sort_calib_reduce_64k_i64(c: &mut Criterion) {
    let data: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64).collect();
    let input = Value::vector_i64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/sort_calib_reduce_64k_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// 64k-axis TopK (k=128): dense radix path (pass76) vs Vec<Literal> generic
// comparison-sort path, run in the same process for a same-worker ratio.
fn topk_params() -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("k".to_owned(), "128".to_owned());
    p
}

fn bench_topk_64k_k128_f64_vec(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| ((i as f64) * 1.000_173).sin() * 1e6 - (i as f64))
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = topk_params();
    c.bench_function("eval/topk_64k_k128_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::TopK, std::slice::from_ref(&input), &p))
    });
}

fn bench_topk_64k_k128_f64_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::from_f64(((i as f64) * 1.000_173).sin() * 1e6 - (i as f64)))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    );
    let p = topk_params();
    c.bench_function("eval/topk_64k_k128_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::TopK, std::slice::from_ref(&input), &p))
    });
}

// F32 TopK (k=128): previously the generic O(n log n) comparison path (F32 is
// Literal-backed); now the complement-key LSD radix path (pass100).
fn bench_topk_64k_k128_f32(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::F32Bits((((i as f32) * 1.000_173).sin() * 1e3 - (i as f32)).to_bits()))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    );
    let p = topk_params();
    c.bench_function("eval/topk_64k_k128_f32", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::TopK, std::slice::from_ref(&input), &p))
    });
}

fn bench_topk_64k_k128_i64_vec(c: &mut Criterion) {
    let data: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64)
        .map(|i| (i.wrapping_mul(2_654_435_761)).rem_euclid(1_000_003) - 500_000)
        .collect();
    let input = Value::vector_i64(&data).unwrap();
    let p = topk_params();
    c.bench_function("eval/topk_64k_k128_i64_vec", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::TopK, std::slice::from_ref(&input), &p))
    });
}

fn bench_topk_64k_k128_i64_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN as i64)
        .map(|i| Literal::I64((i.wrapping_mul(2_654_435_761)).rem_euclid(1_000_003) - 500_000))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    );
    let p = topk_params();
    c.bench_function("eval/topk_64k_k128_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::TopK, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_1k(c: &mut Criterion) {
    let data: Vec<i64> = (0..1000).collect();
    let input = Value::vector_i64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/reduce_sum_1k_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// Large-array F64 full reduction: quantifies the mcqr.30 data-model gap on the
// reduction path. A dense F64 tensor materializes a 24-byte `Vec<Literal>` and
// the per-element `as_f64()` match blocks the fold, moving ~3x the bytes of a
// contiguous f64 fold. Mirrors `add_64k` for the reduce side.
fn bench_reduce_sum_64k_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| i as f64 * 0.001)
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/reduce_sum_64k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// Reference: the same 64k F64 reduction over a NON-dense `Vec<Literal>`-backed
// tensor, which falls through to the generic per-element `as_f64()` loop (the
// pre-dense path). Run in the same process as `bench_reduce_sum_64k_f64` so the
// dense-vs-Literal ratio is measured on one worker, isolating the data-model
// win from cross-worker CPU variance.
fn bench_reduce_sum_64k_f64_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::from_f64(i as f64 * 0.001))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/reduce_sum_64k_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// Generalized dense fast path now also covers ReduceMax (and Prod/Min). These
// two measure the dense-vs-Literal ratio for the max case on one worker — the
// branch in `jax_max_f64` is present in both paths, so the win is the same
// data-model (8 vs 24 bytes/element, no enum match) the sum benches show.
fn bench_reduce_max_64k_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| i as f64 * 0.001)
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/reduce_max_64k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceMax, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_max_64k_f64_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::from_f64(i as f64 * 0.001))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/reduce_max_64k_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceMax, std::slice::from_ref(&input), &p))
    });
}

// Axis (partial) F64 reduction over a 256x256 matrix: the dense fast path uses
// an incremental out-index odometer over the contiguous f64 slice instead of
// decoding each element's full multi-index (flat_to_multi_into +
// multi_to_out_flat) and matching as_f64() on a 24-byte Literal. axis=1 reduces
// the contiguous inner axis (slowly-varying out_idx, good locality); axis=0
// reduces the strided outer axis (fast-cycling out_idx). Each is paired with a
// Vec<Literal>-backed reference run in the same process for same-worker ratios.
const REDUCE_AXIS_N: u32 = 256;

fn axis_reduce_dense_input() -> Value {
    let n = REDUCE_AXIS_N as usize;
    let data: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.001).collect();
    Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![REDUCE_AXIS_N, REDUCE_AXIS_N],
            },
            data,
        )
        .unwrap(),
    )
}

fn axis_reduce_literal_input() -> Value {
    let n = REDUCE_AXIS_N as usize;
    let elements: Vec<Literal> = (0..n * n)
        .map(|i| Literal::from_f64(i as f64 * 0.001))
        .collect();
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![REDUCE_AXIS_N, REDUCE_AXIS_N],
            },
            elements,
        )
        .unwrap(),
    )
}

fn axis_params(axis: &str) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("axes".to_owned(), axis.to_owned());
    p
}

fn bench_reduce_sum_256_axis1_f64(c: &mut Criterion) {
    let input = axis_reduce_dense_input();
    let p = axis_params("1");
    c.bench_function("eval/reduce_sum_256_axis1_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_256_axis1_f64_literal_reference(c: &mut Criterion) {
    let input = axis_reduce_literal_input();
    let p = axis_params("1");
    c.bench_function("eval/reduce_sum_256_axis1_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_256_axis0_f64(c: &mut Criterion) {
    let input = axis_reduce_dense_input();
    let p = axis_params("0");
    c.bench_function("eval/reduce_sum_256_axis0_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_256_axis0_f64_literal_reference(c: &mut Criterion) {
    let input = axis_reduce_literal_input();
    let p = axis_params("0");
    c.bench_function("eval/reduce_sum_256_axis0_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// Integral axis reduction (i64, no dense storage exists for i64) — measures the
// per-element multi-index decode removal alone (the odometer), since the
// data-model 24-byte match is unchanged for Vec<Literal> i64 storage.
fn axis_reduce_i64_dense_input() -> Value {
    let n = REDUCE_AXIS_N as usize;
    let data: Vec<i64> = (0..(n * n) as i64).collect();
    Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![REDUCE_AXIS_N, REDUCE_AXIS_N],
            },
            data,
        )
        .unwrap(),
    )
}

fn axis_reduce_i64_literal_input() -> Value {
    let n = REDUCE_AXIS_N as usize;
    let elements: Vec<Literal> = (0..n * n).map(|i| Literal::I64(i as i64)).collect();
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![REDUCE_AXIS_N, REDUCE_AXIS_N],
            },
            elements,
        )
        .unwrap(),
    )
}

fn bench_reduce_sum_256_axis1_i64(c: &mut Criterion) {
    let input = axis_reduce_i64_dense_input();
    let p = axis_params("1");
    c.bench_function("eval/reduce_sum_256_axis1_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_256_axis1_i64_literal_reference(c: &mut Criterion) {
    let input = axis_reduce_i64_literal_input();
    let p = axis_params("1");
    c.bench_function("eval/reduce_sum_256_axis1_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_256_axis0_i64(c: &mut Criterion) {
    let input = axis_reduce_i64_dense_input();
    let p = axis_params("0");
    c.bench_function("eval/reduce_sum_256_axis0_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_256_axis0_i64_literal_reference(c: &mut Criterion) {
    let input = axis_reduce_i64_literal_input();
    let p = axis_params("0");
    c.bench_function("eval/reduce_sum_256_axis0_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_window_64x64(c: &mut Criterion) {
    let input = real_matrix(64, 64);
    let mut params = BTreeMap::new();
    params.insert("reduce_op".to_owned(), "sum".to_owned());
    params.insert("window_dimensions".to_owned(), "3,3".to_owned());
    params.insert("window_strides".to_owned(), "1,1".to_owned());
    params.insert("padding".to_owned(), "SAME".to_owned());
    c.bench_function("eval/reduce_window_64x64_3x3_same_f64", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&input),
                &params,
            )
        })
    });
}

// 256x256 f64 maxpool (3x3 SAME): dense rank-2 f64 max path (pass85) vs the
// generic Literal reduce_window. Same process for a same-worker ratio.
fn maxpool_params() -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("reduce_op".to_owned(), "max".to_owned());
    p.insert("window_dimensions".to_owned(), "3,3".to_owned());
    p.insert("window_strides".to_owned(), "1,1".to_owned());
    p.insert("padding".to_owned(), "SAME".to_owned());
    p
}

fn bench_maxpool_256x256_f64_vec(c: &mut Criterion) {
    let n = 256usize;
    let data: Vec<f64> = (0..n * n)
        .map(|i| ((i as f64) * 0.123).sin() * 100.0)
        .collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            data,
        )
        .unwrap(),
    );
    let p = maxpool_params();
    c.bench_function("eval/maxpool_256x256_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&input), &p))
    });
}

fn bench_maxpool_256x256_f64_literal_reference(c: &mut Criterion) {
    let n = 256usize;
    let elements: Vec<Literal> = (0..n * n)
        .map(|i| Literal::from_f64(((i as f64) * 0.123).sin() * 100.0))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            elements,
        )
        .unwrap(),
    );
    let p = maxpool_params();
    c.bench_function("eval/maxpool_256x256_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&input), &p))
    });
}

// 256x256 f64 sumpool (2x2 stride 2 VALID, the general non-3x3-SAME sum path):
// dense rank-2 f64 sum path (pass86) vs the Literal loop. Same process.
fn sumpool_params() -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("reduce_op".to_owned(), "sum".to_owned());
    p.insert("window_dimensions".to_owned(), "2,2".to_owned());
    p.insert("window_strides".to_owned(), "2,2".to_owned());
    p.insert("padding".to_owned(), "VALID".to_owned());
    p
}

fn bench_sumpool_256x256_f64_vec(c: &mut Criterion) {
    let n = 256usize;
    let data: Vec<f64> = (0..n * n)
        .map(|i| ((i as f64) * 0.123).sin() * 100.0)
        .collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            data,
        )
        .unwrap(),
    );
    let p = sumpool_params();
    c.bench_function("eval/sumpool_256x256_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&input), &p))
    });
}

fn bench_sumpool_256x256_f64_literal_reference(c: &mut Criterion) {
    let n = 256usize;
    let elements: Vec<Literal> = (0..n * n)
        .map(|i| Literal::from_f64(((i as f64) * 0.123).sin() * 100.0))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            elements,
        )
        .unwrap(),
    );
    let p = sumpool_params();
    c.bench_function("eval/sumpool_256x256_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&input), &p))
    });
}

fn bench_sin_1k(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/sin_1k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sin, std::slice::from_ref(&input), &p))
    });
}

fn bench_sin_64k(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| i as f64 * 0.001)
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/sin_64k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sin, std::slice::from_ref(&input), &p))
    });
}

fn bench_exp_1k(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/exp_1k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Exp, std::slice::from_ref(&input), &p))
    });
}

fn bench_exp_64k(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| i as f64 * 0.001)
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/exp_64k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Exp, std::slice::from_ref(&input), &p))
    });
}

fn bench_square_1k(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/square_1k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Square, std::slice::from_ref(&input), &p))
    });
}

fn bench_integer_pow_1k(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("exponent".to_owned(), "3".to_owned());
    c.bench_function("eval/integer_pow_1k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::IntegerPow, std::slice::from_ref(&input), &p))
    });
}

fn bench_clamp_1k(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.002 - 0.5).collect();
    let input = Value::vector_f64(&data).unwrap();
    let inputs = [input, Value::scalar_f64(0.0), Value::scalar_f64(1.0)];
    let p = no_params();
    c.bench_function("eval/clamp_1k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Clamp, &inputs, &p))
    });
}

fn bench_select_1k(c: &mut Criterion) {
    let cond_elements: Vec<Literal> = (0..1000).map(|i| Literal::Bool(i % 3 == 0)).collect();
    let cond = Value::Tensor(TensorValue {
        dtype: DType::Bool,
        shape: Shape { dims: vec![1000] },
        elements: cond_elements.into(),
    });
    let true_data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let false_data: Vec<f64> = (0..1000).map(|i| -(i as f64) * 0.001).collect();
    let inputs = [
        cond,
        Value::vector_f64(&true_data).unwrap(),
        Value::vector_f64(&false_data).unwrap(),
    ];
    let p = no_params();
    c.bench_function("eval/select_1k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Select, &inputs, &p))
    });
}

// 64k i64 select: dense branch fast path (pass79) vs the generic per-element
// select_literal_as_dtype path. Bool cond is shared (no dense Bool storage);
// only the branch storage differs (dense i64 vs Vec<Literal>). Same process.
fn bench_select_64k_i64_vec(c: &mut Criterion) {
    let cond_elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::Bool(i % 3 == 0))
        .collect();
    let cond = Value::Tensor(TensorValue {
        dtype: DType::Bool,
        shape: Shape {
            dims: vec![LARGE_ELEMENTWISE_LEN as u32],
        },
        elements: cond_elements.into(),
    });
    let t: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64).collect();
    let f: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64).map(|i| -i).collect();
    let inputs = [
        cond,
        Value::vector_i64(&t).unwrap(),
        Value::vector_i64(&f).unwrap(),
    ];
    let p = no_params();
    c.bench_function("eval/select_64k_i64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Select, &inputs, &p))
    });
}

fn bench_select_64k_i64_literal_reference(c: &mut Criterion) {
    let cond_elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::Bool(i % 3 == 0))
        .collect();
    let cond = Value::Tensor(TensorValue {
        dtype: DType::Bool,
        shape: Shape {
            dims: vec![LARGE_ELEMENTWISE_LEN as u32],
        },
        elements: cond_elements.into(),
    });
    let lit = |sign: i64| {
        Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
                (0..LARGE_ELEMENTWISE_LEN as i64)
                    .map(|i| Literal::I64(sign * i))
                    .collect(),
            )
            .unwrap(),
        )
    };
    let inputs = [cond, lit(1), lit(-1)];
    let p = no_params();
    c.bench_function("eval/select_64k_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Select, &inputs, &p))
    });
}

fn bench_complex_mul_1k(c: &mut Criterion) {
    let lhs = complex_vector(1000);
    let rhs = complex_vector(1000);
    let inputs = [lhs, rhs];
    let p = no_params();
    c.bench_function("eval/complex_mul_1k_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &inputs, &p))
    });
}

fn bench_complex_ctor_1k(c: &mut Criterion) {
    let real = real_vector(1000);
    let imag = real_vector(1000);
    let inputs = [real, imag];
    let p = no_params();
    c.bench_function("eval/complex_ctor_1k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Complex, &inputs, &p))
    });
}

fn bench_complex_conj_1k(c: &mut Criterion) {
    let input = complex_vector(1000);
    let p = no_params();
    c.bench_function("eval/conj_1k_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Conj, std::slice::from_ref(&input), &p))
    });
}

fn bench_complex_neg_1k(c: &mut Criterion) {
    let input = complex_vector(1000);
    let p = no_params();
    c.bench_function("eval/neg_1k_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Neg, std::slice::from_ref(&input), &p))
    });
}

fn bench_complex_expm1_1k(c: &mut Criterion) {
    let input = complex_vector(1000);
    let p = no_params();
    c.bench_function("eval/expm1_1k_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Expm1, std::slice::from_ref(&input), &p))
    });
}

fn bench_complex_abs_1k(c: &mut Criterion) {
    let input = complex_vector(1000);
    let p = no_params();
    c.bench_function("eval/abs_1k_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Abs, std::slice::from_ref(&input), &p))
    });
}

fn bench_complex_real_1k(c: &mut Criterion) {
    let input = complex_vector(1000);
    let p = no_params();
    c.bench_function("eval/real_1k_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Real, std::slice::from_ref(&input), &p))
    });
}

fn bench_complex_imag_1k(c: &mut Criterion) {
    let input = complex_vector(1000);
    let p = no_params();
    c.bench_function("eval/imag_1k_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Imag, std::slice::from_ref(&input), &p))
    });
}

fn bench_complex_is_finite_1k(c: &mut Criterion) {
    let input = complex_vector(1000);
    let p = no_params();
    c.bench_function("eval/is_finite_1k_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::IsFinite, std::slice::from_ref(&input), &p))
    });
}

fn bench_fft_256(c: &mut Criterion) {
    let input = complex_vector(256);
    let p = no_params();
    c.bench_function("eval/fft_256_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Fft, std::slice::from_ref(&input), &p))
    });
}

fn bench_ifft_256(c: &mut Criterion) {
    let input = complex_vector(256);
    let p = no_params();
    c.bench_function("eval/ifft_256_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Ifft, std::slice::from_ref(&input), &p))
    });
}

// Non-power-of-two length: exercises the Bluestein O(n log n) path (the old
// fallback was the O(n²) direct DFT). 1000 is non-power-of-two and composite.
fn bench_fft_1000(c: &mut Criterion) {
    let input = complex_vector(1000);
    let p = no_params();
    c.bench_function("eval/fft_1000_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Fft, std::slice::from_ref(&input), &p))
    });
}

// 1009 is prime — the worst case for any radix decomposition; Bluestein still
// runs in O(n log n) via a length-2048 convolution.
fn bench_fft_1009_prime(c: &mut Criterion) {
    let input = complex_vector(1009);
    let p = no_params();
    c.bench_function("eval/fft_1009_prime_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Fft, std::slice::from_ref(&input), &p))
    });
}

// Batched non-power-of-two FFT along the last axis: 128 rows of length 1000.
// Exercises the shared Bluestein plan (chirp table + kernel FFT built once and
// reused across all 128 rows) vs the per-row rebuild.
fn bench_fft_batch_128x1000(c: &mut Criterion) {
    let input = complex_matrix(128, 1000);
    let p = no_params();
    c.bench_function("eval/fft_batch_128x1000_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Fft, std::slice::from_ref(&input), &p))
    });
}

fn bench_rfft_256(c: &mut Criterion) {
    let input = real_vector(256);
    let p = no_params();
    c.bench_function("eval/rfft_256_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Rfft, std::slice::from_ref(&input), &p))
    });
}

// Batched non-power-of-two real FFT: 64 rows of length 1000. Exercises the
// shared Bluestein plan (built once, reused across all 64 rows) in eval_rfft.
fn bench_rfft_batch_64x1000(c: &mut Criterion) {
    let input = real_matrix(64, 1000);
    let p = no_params();
    c.bench_function("eval/rfft_batch_64x1000_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Rfft, std::slice::from_ref(&input), &p))
    });
}

fn bench_irfft_256(c: &mut Criterion) {
    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "256".to_owned());
    let full = eval_primitive(Primitive::Rfft, &[real_vector(256)], &params).unwrap();
    c.bench_function("eval/irfft_256_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Irfft, std::slice::from_ref(&full), &params))
    });
}

fn bench_reshape(c: &mut Criterion) {
    let data: Vec<i64> = (0..1000).collect();
    let input = Value::vector_i64(&data).unwrap();
    let mut params = BTreeMap::new();
    params.insert("new_shape".into(), "10,100".into());
    c.bench_function("eval/reshape_1k_to_10x100", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Reshape, std::slice::from_ref(&input), &params))
    });
}

fn bench_gather_128_rows_16_cols(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..(128 * 16)).map(Literal::I64).collect();
    let operand = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 16],
            },
            elements,
        )
        .unwrap(),
    );
    let indices_data: Vec<i64> = (0..128).rev().collect();
    let indices = Value::vector_i64(&indices_data).unwrap();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".into(), "1,16".into());
    c.bench_function("eval/gather_128_rows_16_cols", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::Gather,
                &[operand.clone(), indices.clone()],
                &params,
            )
        })
    });
}

// Large contiguous gather (256x256 f64 operand, gather 256 rows reversed):
// dense fast path (pass83) vs the Vec<Literal> copy. Same process for a
// same-worker ratio.
fn bench_gather_256x256_f64_vec(c: &mut Criterion) {
    let n = 256usize;
    let data: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.001).collect();
    let operand = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            data,
        )
        .unwrap(),
    );
    let indices = Value::vector_i64(&(0..n as i64).rev().collect::<Vec<_>>()).unwrap();
    let mut p = BTreeMap::new();
    p.insert("slice_sizes".into(), "1,256".into());
    c.bench_function("eval/gather_256x256_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Gather, &[operand.clone(), indices.clone()], &p))
    });
}

fn bench_gather_256x256_f64_literal_reference(c: &mut Criterion) {
    let n = 256usize;
    let elements: Vec<Literal> = (0..n * n)
        .map(|i| Literal::from_f64(i as f64 * 0.001))
        .collect();
    let operand = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            elements,
        )
        .unwrap(),
    );
    let indices = Value::vector_i64(&(0..n as i64).rev().collect::<Vec<_>>()).unwrap();
    let mut p = BTreeMap::new();
    p.insert("slice_sizes".into(), "1,256".into());
    c.bench_function("eval/gather_256x256_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Gather, &[operand.clone(), indices.clone()], &p))
    });
}

fn bench_scatter_128_rows_16_cols(c: &mut Criterion) {
    let operand = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 16],
            },
            vec![Literal::I64(0); 128 * 16],
        )
        .unwrap(),
    );
    let indices_data: Vec<i64> = (0..128).rev().collect();
    let indices = Value::vector_i64(&indices_data).unwrap();
    let updates = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 16],
            },
            (0..(128 * 16)).map(Literal::I64).collect(),
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/scatter_128_rows_16_cols", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::Scatter,
                &[operand.clone(), indices.clone(), updates.clone()],
                &p,
            )
        })
    });
}

// Large scatter (256x256 f64 operand, overwrite 256 rows reversed): dense fast
// path (pass84) vs the Vec<Literal> materialize + copy. Same process.
fn bench_scatter_256x256_f64_vec(c: &mut Criterion) {
    let n = 256usize;
    let mk = |dense: bool| -> (Value, Value) {
        let op: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.001).collect();
        let up: Vec<f64> = (0..n * n).map(|i| -(i as f64) * 0.002).collect();
        if dense {
            (
                Value::Tensor(
                    TensorValue::new_f64_values(
                        Shape {
                            dims: vec![n as u32, n as u32],
                        },
                        op,
                    )
                    .unwrap(),
                ),
                Value::Tensor(
                    TensorValue::new_f64_values(
                        Shape {
                            dims: vec![n as u32, n as u32],
                        },
                        up,
                    )
                    .unwrap(),
                ),
            )
        } else {
            (
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        Shape {
                            dims: vec![n as u32, n as u32],
                        },
                        op.into_iter().map(Literal::from_f64).collect(),
                    )
                    .unwrap(),
                ),
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        Shape {
                            dims: vec![n as u32, n as u32],
                        },
                        up.into_iter().map(Literal::from_f64).collect(),
                    )
                    .unwrap(),
                ),
            )
        }
    };
    let indices = Value::vector_i64(&(0..n as i64).rev().collect::<Vec<_>>()).unwrap();
    let mut p = BTreeMap::new();
    p.insert("index_mode".into(), "clip".into());
    let (op_d, up_d) = mk(true);
    c.bench_function("eval/scatter_256x256_f64_vec", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::Scatter,
                &[op_d.clone(), indices.clone(), up_d.clone()],
                &p,
            )
        })
    });
}

fn bench_scatter_256x256_f64_literal_reference(c: &mut Criterion) {
    let n = 256usize;
    let op: Vec<Literal> = (0..n * n)
        .map(|i| Literal::from_f64(i as f64 * 0.001))
        .collect();
    let up: Vec<Literal> = (0..n * n)
        .map(|i| Literal::from_f64(-(i as f64) * 0.002))
        .collect();
    let operand = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            op,
        )
        .unwrap(),
    );
    let updates = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            up,
        )
        .unwrap(),
    );
    let indices = Value::vector_i64(&(0..n as i64).rev().collect::<Vec<_>>()).unwrap();
    let mut p = BTreeMap::new();
    p.insert("index_mode".into(), "clip".into());
    c.bench_function("eval/scatter_256x256_f64_literal_ref", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::Scatter,
                &[operand.clone(), indices.clone(), updates.clone()],
                &p,
            )
        })
    });
}

fn bench_slice_64_rows_16_cols(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..(128 * 16)).map(Literal::I64).collect();
    let operand = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 16],
            },
            elements,
        )
        .unwrap(),
    );
    let mut params = BTreeMap::new();
    params.insert("start_indices".into(), "32,0".into());
    params.insert("limit_indices".into(), "96,16".into());
    c.bench_function("eval/slice_64_rows_16_cols", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Slice, std::slice::from_ref(&operand), &params))
    });
}

fn bench_dynamic_slice_64_rows_16_cols(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..(128 * 16)).map(Literal::I64).collect();
    let operand = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 16],
            },
            elements,
        )
        .unwrap(),
    );
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".into(), "64,16".into());
    let start0 = Value::scalar_i64(32);
    let start1 = Value::scalar_i64(0);
    c.bench_function("eval/dynamic_slice_64_rows_16_cols", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::DynamicSlice,
                &[operand.clone(), start0.clone(), start1.clone()],
                &params,
            )
        })
    });
}

fn bench_dynamic_update_slice_64_rows_16_cols(c: &mut Criterion) {
    let operand = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 16],
            },
            vec![Literal::I64(0); 128 * 16],
        )
        .unwrap(),
    );
    let update = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape { dims: vec![64, 16] },
            (0..(64 * 16)).map(Literal::I64).collect(),
        )
        .unwrap(),
    );
    let start0 = Value::scalar_i64(32);
    let start1 = Value::scalar_i64(0);
    let p = no_params();
    c.bench_function("eval/dynamic_update_slice_64_rows_16_cols", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::DynamicUpdateSlice,
                &[
                    operand.clone(),
                    update.clone(),
                    start0.clone(),
                    start1.clone(),
                ],
                &p,
            )
        })
    });
}

fn bench_eq_1k(c: &mut Criterion) {
    let data: Vec<i64> = (0..1000).collect();
    let lhs = Value::vector_i64(&data).unwrap();
    let rhs = Value::vector_i64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/eq_1k_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Eq, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_bitwise_and_1k(c: &mut Criterion) {
    let lhs_data: Vec<i64> = (0..1000).map(|i| i * 3 + 1).collect();
    let rhs_data: Vec<i64> = (0..1000).map(|i| i * 5 + 7).collect();
    let lhs = Value::vector_i64(&lhs_data).unwrap();
    let rhs = Value::vector_i64(&rhs_data).unwrap();
    let p = no_params();
    c.bench_function("eval/bitwise_and_1k_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::BitwiseAnd, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_dispatch_overhead(c: &mut Criterion) {
    let inputs = [Value::scalar_i64(1), Value::scalar_i64(1)];
    let p = no_params();
    c.bench_function("eval/dispatch_overhead_add_scalar", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &inputs, &p))
    });
}

criterion_group!(
    benches,
    bench_dispatch_overhead,
    bench_add_scalar,
    bench_add_1k_vector,
    bench_mul_1k_vector,
    bench_add_1k_f64_vector,
    bench_div_1k_f64_vector,
    bench_add_64k_f64_vec,
    bench_add_64k_f64_dense_reference,
    bench_add_64k_i64_vec,
    bench_add_64k_i64_literal_reference,
    bench_mul_64k_i64_vec,
    bench_mul_64k_i64_literal_reference,
    bench_scalar_mul_64k_i64_vec,
    bench_scalar_mul_64k_i64_literal_reference,
    bench_lt_64k_i64_vec,
    bench_lt_64k_i64_literal_reference,
    bench_lt_scalar_64k_i64_vec,
    bench_lt_scalar_64k_i64_literal_reference,
    bench_scalar_mul_1k_f64_vector,
    bench_tensor_sub_scalar_1k_f64_vector,
    bench_eq_1k_f64_vector,
    bench_lt_scalar_1k_f64_vector,
    bench_add_broadcast_bias_1k_f64,
    bench_add_broadcast_256_f64_vec,
    bench_add_broadcast_256_f64_literal_reference,
    bench_add_broadcast_256_i64_vec,
    bench_add_broadcast_256_i64_literal_reference,
    bench_lt_broadcast_256_i64_vec,
    bench_lt_broadcast_256_i64_literal_reference,
    bench_lt_broadcast_256_f64_vec,
    bench_lt_broadcast_256_f64_literal_reference,
    bench_nextafter_1k,
    bench_dot_100,
    bench_dot_256_matrix_f64,
    bench_eig_48,
    bench_cholesky_128_f64,
    bench_qr_128_f64,
    bench_lu_128_f64,
    bench_svd_48_f64,
    bench_matmul_2d_256,
    bench_matmul_2d_512,
    bench_conv2d_32x32x8_3x3x16_f64_vec,
    bench_conv2d_32x32x8_3x3x16_f64_literal_reference,
    bench_conv1d_1024x16_5x32_f64_vec,
    bench_conv1d_1024x16_5x32_f64_literal_reference,
    bench_solve_24x24_24rhs,
    bench_concat_axis1_3x_f64,
    bench_concat_axis0_3x_f64,
    bench_transpose_256x256_f64,
    bench_reduce_sum_1k,
    bench_reduce_sum_64k_f64,
    bench_reduce_sum_64k_f64_literal_reference,
    bench_reduce_max_64k_f64,
    bench_reduce_max_64k_f64_literal_reference,
    bench_reduce_sum_256_axis1_f64,
    bench_reduce_sum_256_axis1_f64_literal_reference,
    bench_reduce_sum_256_axis0_f64,
    bench_reduce_sum_256_axis0_f64_literal_reference,
    bench_reduce_sum_256_axis1_i64,
    bench_reduce_sum_256_axis1_i64_literal_reference,
    bench_reduce_sum_256_axis0_i64,
    bench_reduce_sum_256_axis0_i64_literal_reference,
    bench_reduce_sum_64k_i64,
    bench_reduce_sum_64k_i64_literal_reference,
    bench_reduce_and_64k_bool_vec,
    bench_reduce_and_64k_bool_literal_reference,
    bench_reduce_and_256_axis1_bool_vec,
    bench_reduce_and_256_axis1_bool_literal_reference,
    bench_cumsum_64k_f64_vec,
    bench_cumsum_64k_f64_literal_reference,
    bench_sort_64k_i64,
    bench_sort_64k_f64,
    bench_argmax_64k_f64,
    bench_sort_64k_f64_descending,
    bench_convert_64k_f64_to_i64,
    bench_broadcast_256_to_256x256_f64,
    bench_pad_256x256_to_258x258_f64,
    bench_sort_64k_f32,
    bench_sort_64k_u32,
    bench_sort_calib_reduce_64k_i64,
    bench_topk_64k_k128_f64_vec,
    bench_topk_64k_k128_f32,
    bench_topk_64k_k128_f64_literal_reference,
    bench_topk_64k_k128_i64_vec,
    bench_topk_64k_k128_i64_literal_reference,
    bench_reduce_window_64x64,
    bench_maxpool_256x256_f64_vec,
    bench_maxpool_256x256_f64_literal_reference,
    bench_sumpool_256x256_f64_vec,
    bench_sumpool_256x256_f64_literal_reference,
    bench_sin_1k,
    bench_sin_64k,
    bench_exp_1k,
    bench_exp_64k,
    bench_square_1k,
    bench_integer_pow_1k,
    bench_clamp_1k,
    bench_select_1k,
    bench_select_64k_i64_vec,
    bench_select_64k_i64_literal_reference,
    bench_complex_mul_1k,
    bench_complex_ctor_1k,
    bench_complex_conj_1k,
    bench_complex_neg_1k,
    bench_complex_expm1_1k,
    bench_complex_abs_1k,
    bench_complex_real_1k,
    bench_complex_imag_1k,
    bench_complex_is_finite_1k,
    bench_fft_256,
    bench_ifft_256,
    bench_fft_1000,
    bench_fft_1009_prime,
    bench_fft_batch_128x1000,
    bench_rfft_256,
    bench_rfft_batch_64x1000,
    bench_irfft_256,
    bench_reshape,
    bench_gather_128_rows_16_cols,
    bench_gather_256x256_f64_vec,
    bench_gather_256x256_f64_literal_reference,
    bench_scatter_128_rows_16_cols,
    bench_scatter_256x256_f64_vec,
    bench_scatter_256x256_f64_literal_reference,
    bench_slice_64_rows_16_cols,
    bench_dynamic_slice_64_rows_16_cols,
    bench_dynamic_update_slice_64_rows_16_cols,
    bench_eq_1k,
    bench_bitwise_and_1k,
);
criterion_main!(benches);
