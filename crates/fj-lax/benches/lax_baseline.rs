use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::threefry::{random_key, random_normal, random_uniform};
use fj_lax::{eval_primitive, eval_primitive_multi};
use std::collections::BTreeMap;
use std::hint::black_box;

const LARGE_ELEMENTWISE_LEN: usize = 65_536;
const LARGE_RANDOM_LEN: usize = 1_048_576;

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

// Dense `(re, im)` f64-backed complex128 vector (the `as_complex_slice` fast
// path) — the steady-state representation upstream complex ops now produce.
fn complex_vector_dense(len: usize) -> Value {
    let values: Vec<(f64, f64)> = (0..len)
        .map(|i| {
            let x = i as f64;
            ((x * 0.125).sin(), (x * 0.25).cos())
        })
        .collect();
    Value::Tensor(
        TensorValue::new_complex_values(
            DType::Complex128,
            Shape {
                dims: vec![len as u32],
            },
            values,
        )
        .unwrap(),
    )
}

// Same logical data as `complex_matrix`, but backed by dense `(re, im)` f64
// storage (the `as_complex_slice` fast path) instead of per-element `Literal`s.
// This is the steady-state representation a complex pipeline produces upstream.
fn complex_matrix_dense(rows: usize, cols: usize) -> Value {
    let values: Vec<(f64, f64)> = (0..rows * cols)
        .map(|i| {
            let x = i as f64;
            ((x * 0.125).sin(), (x * 0.25).cos())
        })
        .collect();
    Value::Tensor(
        TensorValue::new_complex_values(
            DType::Complex128,
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            values,
        )
        .unwrap(),
    )
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

fn bench_random_uniform_1m(c: &mut Criterion) {
    let key = random_key(0x1234_5678_9ABC_DEF0);
    c.bench_function("random/uniform_1m_f64", |bencher| {
        bencher.iter(|| black_box(random_uniform(key, LARGE_RANDOM_LEN, -1.0, 1.0)));
    });
}

fn bench_random_normal_1m(c: &mut Criterion) {
    let key = random_key(0x0F0E_0D0C_0B0A_0908);
    c.bench_function("random/normal_1m_f64", |bencher| {
        bencher.iter(|| black_box(random_normal(key, LARGE_RANDOM_LEN)));
    });
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

// Broadcast Pow: [1024,1] ** [1,1024] -> [1024,1024] (~1M outputs). Each output
// is an independent powf on broadcast-gathered operands — compute-bound, threads.
fn bench_pow_broadcast_1024x1024_f64(c: &mut Criterion) {
    let a: Vec<f64> = (0..1024).map(|i| 1.0 + (i % 97) as f64 * 0.01).collect();
    let b: Vec<f64> = (0..1024).map(|i| 0.5 + (i % 13) as f64 * 0.1).collect();
    let lhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![1024, 1],
            },
            a,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![1, 1024],
            },
            b,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/pow_broadcast_1024x1024_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Pow, &[lhs.clone(), rhs.clone()], &p))
    });
}

// 1M-element same-shape Pow (a^b via f64::powf): per-element powf (~tens of ns)
// dominates memory traffic, so the elementwise op is compute-bound and threads.
fn bench_pow_1m_f64_vec(c: &mut Criterion) {
    let a: Vec<f64> = (0..1 << 20).map(|i| 1.0 + (i % 97) as f64 * 0.01).collect();
    let b: Vec<f64> = (0..1 << 20).map(|i| 0.5 + (i % 13) as f64 * 0.1).collect();
    let lhs = Value::vector_f64(&a).unwrap();
    let rhs = Value::vector_f64(&b).unwrap();
    let p = no_params();
    c.bench_function("eval/pow_1m_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Pow, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_atan2_1m_f64_vec(c: &mut Criterion) {
    let a: Vec<f64> = (0..1 << 20).map(|i| (i % 211) as f64 - 105.0).collect();
    let b: Vec<f64> = (0..1 << 20).map(|i| (i % 307) as f64 - 153.0).collect();
    let lhs = Value::vector_f64(&a).unwrap();
    let rhs = Value::vector_f64(&b).unwrap();
    let p = no_params();
    c.bench_function("eval/atan2_1m_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Atan2, &[lhs.clone(), rhs.clone()], &p))
    });
}

// 1M erf (Gaussian CDF) — a compute-bound unary transcendental that was on the
// serial path; routed through eval_unary_elementwise_parallel.
fn bench_erf_1m_f64_vec(c: &mut Criterion) {
    let a: Vec<f64> = (0..1 << 20)
        .map(|i| ((i % 4001) as f64 - 2000.0) * 0.001)
        .collect();
    let input = Value::vector_f64(&a).unwrap();
    let p = no_params();
    c.bench_function("eval/erf_1m_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Erf, std::slice::from_ref(&input), &p))
    });
}

// 256k polygamma(2, x) over a dense f64 tensor: polygamma_approx is a heavy
// series/asymptotic evaluation per element — compute-bound, threads.
fn bench_polygamma_n2_256k_f64(c: &mut Criterion) {
    let n = Value::scalar_i64(2);
    let x: Vec<f64> = (0..1 << 18)
        .map(|i| 0.5 + (i % 4096) as f64 * 0.01)
        .collect();
    let xt = Value::vector_f64(&x).unwrap();
    let inputs = [n, xt];
    let p = no_params();
    c.bench_function("eval/polygamma_n2_256k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Polygamma, &inputs, &p))
    });
}

// 256k same-shape Igamma(a, x): regularized lower incomplete gamma — a series /
// continued-fraction evaluation per element. Compute-bound, threads.
fn bench_igamma_256k_f64(c: &mut Criterion) {
    let a: Vec<f64> = (0..1 << 18).map(|i| 1.0 + (i % 97) as f64 * 0.05).collect();
    let x: Vec<f64> = (0..1 << 18)
        .map(|i| 0.5 + (i % 211) as f64 * 0.02)
        .collect();
    let inputs = [
        Value::vector_f64(&a).unwrap(),
        Value::vector_f64(&x).unwrap(),
    ];
    let p = no_params();
    c.bench_function("eval/igamma_256k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Igamma, &inputs, &p))
    });
}

fn bench_cbrt_1m_f64_vec(c: &mut Criterion) {
    let a: Vec<f64> = (0..1 << 20).map(|i| (i as f64) * 0.0007 - 300.0).collect();
    let input = Value::vector_f64(&a).unwrap();
    let p = no_params();
    c.bench_function("eval/cbrt_1m_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cbrt, std::slice::from_ref(&input), &p))
    });
}

// 1M tensor ** scalar (x ** 2.5): the ubiquitous scalar-power broadcast,
// compute-bound on powf — exercises the threaded scalar-broadcast expensive path.
fn bench_pow_scalar_1m_f64_vec(c: &mut Criterion) {
    let a: Vec<f64> = (0..1 << 20).map(|i| 1.0 + (i % 97) as f64 * 0.01).collect();
    let lhs = Value::vector_f64(&a).unwrap();
    let rhs = Value::scalar_f64(2.5);
    let p = no_params();
    c.bench_function("eval/pow_scalar_1m_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Pow, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_pow_scalar_1m_f64_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..1 << 20)
        .map(|i| Literal::from_f64(1.0 + (i % 97) as f64 * 0.01))
        .collect();
    let lhs = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape {
            dims: vec![1 << 20],
        },
        elements: elements.into(),
    });
    let rhs = Value::scalar_f64(2.5);
    let p = no_params();
    c.bench_function("eval/pow_scalar_1m_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Pow, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_atan2_scalar_1m_f64_vec(c: &mut Criterion) {
    let a: Vec<f64> = (0..1 << 20).map(|i| (i % 211) as f64 - 105.0).collect();
    let lhs = Value::vector_f64(&a).unwrap();
    let rhs = Value::scalar_f64(3.25);
    let p = no_params();
    c.bench_function("eval/atan2_scalar_1m_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Atan2, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_atan2_scalar_1m_f64_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..1 << 20)
        .map(|i| Literal::from_f64((i % 211) as f64 - 105.0))
        .collect();
    let lhs = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape {
            dims: vec![1 << 20],
        },
        elements: elements.into(),
    });
    let rhs = Value::scalar_f64(3.25);
    let p = no_params();
    c.bench_function("eval/atan2_scalar_1m_f64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Atan2, &[lhs.clone(), rhs.clone()], &p))
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
fn bench_add_broadcast_row_1024x1024_f64(c: &mut Criterion) {
    // Row broadcast [1024,1024] + [1024] (bias-add / normalization shape): the
    // dense broadcast path. Inner (last) dim is contiguous for both operands.
    let n = 1024usize;
    let mat: Vec<f64> = (0..n * n).map(|i| i as f64 * 1e-4).collect();
    let row: Vec<f64> = (0..n).map(|i| i as f64 * 2e-4).collect();
    let lhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            mat,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32],
            },
            row,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/add_broadcast_row_1024x1024_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_add_broadcast_col_1024x1024_f64(c: &mut Criterion) {
    // Column broadcast [1024,1024] + [1024,1]: inner dim contiguous for lhs,
    // broadcast (stride 0) for rhs.
    let n = 1024usize;
    let mat: Vec<f64> = (0..n * n).map(|i| i as f64 * 1e-4).collect();
    let col: Vec<f64> = (0..n).map(|i| i as f64 * 2e-4).collect();
    let lhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            mat,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, 1],
            },
            col,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/add_broadcast_col_1024x1024_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_add_broadcast_row_1024x1024_i64(c: &mut Criterion) {
    // Row broadcast [1024,1024] + [1024], i64 (integer bias-add / index math).
    let n = 1024usize;
    let mat: Vec<i64> = (0..n * n).map(|i| i as i64).collect();
    let row: Vec<i64> = (0..n).map(|i| i as i64 * 3).collect();
    let lhs = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            mat,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![n as u32],
            },
            row,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/add_broadcast_row_1024x1024_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_add_broadcast_col_1024x1024_i64(c: &mut Criterion) {
    // Column broadcast [1024,1024] + [1024,1], i64.
    let n = 1024usize;
    let mat: Vec<i64> = (0..n * n).map(|i| i as i64).collect();
    let col: Vec<i64> = (0..n).map(|i| i as i64 * 3).collect();
    let lhs = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            mat,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![n as u32, 1],
            },
            col,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/add_broadcast_col_1024x1024_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_bitand_same_shape_1024x1024_i64(c: &mut Criterion) {
    // Same-shape [1024,1024] & [1024,1024], i64 (integer bitmask combine).
    let n = 1024usize;
    let a: Vec<i64> = (0..(n * n) as i64).map(|i| i * 2654435761).collect();
    let b: Vec<i64> = (0..(n * n) as i64).map(|i| i ^ 0x5555_5555).collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new_i64_values(Shape { dims: dims.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new_i64_values(Shape { dims }, b).unwrap());
    let p = no_params();
    c.bench_function("eval/bitand_same_shape_1024x1024_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::BitwiseAnd, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_select_1024x1024_f64(c: &mut Criterion) {
    // jnp.where(mask, a, b): dense-bool cond + dense-f64 branches [1024,1024].
    // The cond is dense Bool storage (as a comparison now produces), exercising
    // the as_bool_slice + as_f64_slice select fast path.
    let n = 1024usize;
    let mask: Vec<bool> = (0..n * n).map(|i| (i * 7 + 1) % 3 == 0).collect();
    let a: Vec<f64> = (0..n * n).map(|i| i as f64 * 1e-4).collect();
    let b: Vec<f64> = (0..n * n).map(|i| -(i as f64) * 2e-4).collect();
    let dims = vec![n as u32, n as u32];
    let cond =
        Value::Tensor(TensorValue::new_bool_values(Shape { dims: dims.clone() }, mask).unwrap());
    let on_true =
        Value::Tensor(TensorValue::new_f64_values(Shape { dims: dims.clone() }, a).unwrap());
    let on_false = Value::Tensor(TensorValue::new_f64_values(Shape { dims }, b).unwrap());
    let p = no_params();
    c.bench_function("eval/select_1024x1024_f64", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::Select,
                &[cond.clone(), on_true.clone(), on_false.clone()],
                &p,
            )
        })
    });
}

fn bench_lt_same_shape_1024x1024_f64(c: &mut Criterion) {
    // Same-shape [1024,1024] < [1024,1024], f64 → Bool mask (the most common
    // comparison shape). Exercises the dense-Bool output write path.
    let n = 1024usize;
    let a: Vec<f64> = (0..n * n).map(|i| i as f64 * 1e-4).collect();
    let b: Vec<f64> = (0..n * n).map(|i| ((i * 7 + 3) % 1000) as f64).collect();
    let lhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            a,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            b,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/lt_same_shape_1024x1024_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_lt_broadcast_row_1024x1024_f64(c: &mut Criterion) {
    // Row broadcast [1024,1024] < [1024], f64 → Bool mask (thresholding).
    let n = 1024usize;
    let mat: Vec<f64> = (0..n * n).map(|i| i as f64 * 1e-4).collect();
    let row: Vec<f64> = (0..n).map(|i| i as f64 * 100.0).collect();
    let lhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            mat,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32],
            },
            row,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/lt_broadcast_row_1024x1024_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_lt_broadcast_row_1024x1024_i64(c: &mut Criterion) {
    // Row broadcast [1024,1024] < [1024], i64 → Bool mask.
    let n = 1024usize;
    let mat: Vec<i64> = (0..(n * n) as i64).collect();
    let row: Vec<i64> = (0..n as i64).map(|i| i * 1000).collect();
    let lhs = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            mat,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![n as u32],
            },
            row,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/lt_broadcast_row_1024x1024_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Lt, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_mul_broadcast_row_512x512_c128(c: &mut Criterion) {
    // Row broadcast [512,512] * [512], Complex128 (per-channel modulation).
    let n = 512usize;
    let mat: Vec<(f64, f64)> = (0..n * n)
        .map(|i| (i as f64 * 1e-4, i as f64 * 2e-4))
        .collect();
    let row: Vec<(f64, f64)> = (0..n)
        .map(|i| (i as f64 * 3e-4, -(i as f64) * 1e-4))
        .collect();
    let lhs = Value::Tensor(
        TensorValue::new_complex_values(
            DType::Complex128,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            mat,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_complex_values(
            DType::Complex128,
            Shape {
                dims: vec![n as u32],
            },
            row,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/mul_broadcast_row_512x512_c128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_mul_broadcast_col_512x512_c128(c: &mut Criterion) {
    // Column broadcast [512,512] * [512,1], Complex128.
    let n = 512usize;
    let mat: Vec<(f64, f64)> = (0..n * n)
        .map(|i| (i as f64 * 1e-4, i as f64 * 2e-4))
        .collect();
    let col: Vec<(f64, f64)> = (0..n)
        .map(|i| (i as f64 * 3e-4, -(i as f64) * 1e-4))
        .collect();
    let lhs = Value::Tensor(
        TensorValue::new_complex_values(
            DType::Complex128,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            mat,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_complex_values(
            DType::Complex128,
            Shape {
                dims: vec![n as u32, 1],
            },
            col,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/mul_broadcast_col_512x512_c128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[lhs.clone(), rhs.clone()], &p))
    });
}

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

fn bench_eigh_48(c: &mut Criterion) {
    // Real symmetric eigendecomposition (Eigh) via Jacobi. Diagonally dominant
    // symmetric 48×48 so the iterative path (not the 3×3 analytic shortcut)
    // runs and is well-conditioned.
    let n = 48usize;
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = if i == j {
                (n as f64) + (i as f64)
            } else {
                (((i * 7 + j * 13) % 5) as f64) - 2.0
            };
            data[i * n + j] = v;
            data[j * n + i] = v;
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
    c.bench_function("linalg/eigh_48x48_f64", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&m), &p))
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

fn bench_cholesky_512_f64(c: &mut Criterion) {
    // Real SPD matrix A = B^T B + n*I at n=512 (memory-bound regime).
    let n = 512usize;
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
    c.bench_function("linalg/cholesky_512x512_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cholesky, std::slice::from_ref(&m), &p))
    });
}

fn bench_cholesky_1024_f64(c: &mut Criterion) {
    // n=1024: trailing Schur updates are large enough that the blocked kernel's
    // GEMM auto-threads, the regime where blocked decisively beats scalar.
    let n = 1024usize;
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
    c.bench_function("linalg/cholesky_1024x1024_f64", |bencher| {
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

fn bench_qr_1024_f64(c: &mut Criterion) {
    let n = 1024usize;
    let m = real_matrix(n, n);
    let p = no_params();
    c.bench_function("linalg/qr_1024x1024_f64", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&m), &p))
    });
}

// Pseudoinverse of a tall real matrix: the m>=n branch forms the n×n Gram matrix
// AᵀA and pseudo-inverts it via a symmetric Jacobi eigensolve, which dominates at
// these sizes. real_f64_data() gives a well-conditioned deterministic matrix.
fn real_f64_data(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|i| {
            let x = i as f64;
            (x * 0.125).sin() + (x * 0.03125).cos()
        })
        .collect()
}

fn bench_pinv_256x128_f64(c: &mut Criterion) {
    let (m, n) = (256usize, 128usize);
    let a = real_f64_data(m, n);
    c.bench_function("linalg/pinv_256x128_f64", |bencher| {
        bencher.iter(|| fj_lax::linalg::pinv(std::hint::black_box(&a), m, n, 1e-15))
    });
}

fn bench_pinv_192x192_f64(c: &mut Criterion) {
    let n = 192usize;
    let a = real_f64_data(n, n);
    c.bench_function("linalg/pinv_192x192_f64", |bencher| {
        bencher.iter(|| fj_lax::linalg::pinv(std::hint::black_box(&a), n, n, 1e-15))
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

fn bench_lu_1024_f64(c: &mut Criterion) {
    // Large square LU: exercises the cache-blocked right-looking path (min(m,n)
    // >= LU_BLOCK_THRESHOLD) whose trailing update runs on the GEMM microkernel.
    let n = 1024usize;
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
    c.bench_function("linalg/lu_1024x1024_f64", |bencher| {
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

/// Same numeric data as `bench_svd_48_f64`, but typed Complex128 (imag 0) so it
/// forces the complex SVD kernel. Lets the real-path speedup be measured against
/// the complex path inside one binary on one worker (no cross-invocation drift).
fn bench_svd_48_complex_path(c: &mut Criterion) {
    let n = 48usize;
    let elements: Vec<Literal> = (0..n * n)
        .map(|i| {
            let x = i as f64;
            Literal::from_complex128((x * 0.125).sin() + (x * 0.03125).cos(), 0.0)
        })
        .collect();
    let m = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            elements,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("linalg/svd_48x48_complex_path", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&m), &p))
    });
}

/// full_matrices=true real SVD (real cyclic-Jacobi path, U extended to m×m).
fn bench_svd_48_full_f64(c: &mut Criterion) {
    let n = 48usize;
    let m = real_matrix(n, n);
    let mut p = no_params();
    p.insert("full_matrices".to_owned(), "true".to_owned());
    c.bench_function("linalg/svd_48x48_full_f64", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&m), &p))
    });
}

/// full_matrices=true on identical data typed Complex128 (imag 0) → the complex
/// max-pivot kernel. Same-binary baseline for the full real path's speedup.
fn bench_svd_48_full_complex_path(c: &mut Criterion) {
    let n = 48usize;
    let elements: Vec<Literal> = (0..n * n)
        .map(|i| {
            let x = i as f64;
            Literal::from_complex128((x * 0.125).sin() + (x * 0.03125).cos(), 0.0)
        })
        .collect();
    let m = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            elements,
        )
        .unwrap(),
    );
    let mut p = no_params();
    p.insert("full_matrices".to_owned(), "true".to_owned());
    c.bench_function("linalg/svd_48x48_full_complex_path", |bencher| {
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

fn bench_matmul_2d_1024(c: &mut Criterion) {
    // Large GEMM (B = 8MB, deeply spills L3): the regime where B-streaming
    // bandwidth, not the microkernel, binds. Used to measure panel-packing wins.
    let (m, k, n) = (1024usize, 1024usize, 1024usize);
    let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 1e-4).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 2e-4).collect();
    c.bench_function("linalg/matmul_2d_1024x1024x1024_f64", |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::matmul_2d(&a, m, k, &b, n))
    });
}

fn bench_matmul_2d_2048(c: &mut Criterion) {
    // B = 32MB — past per-CCD L3, so matmul_2d's full-B-per-row-block streaming
    // goes RAM/fabric-bound. Premise check for communication-avoiding cache-blocking.
    let (m, k, n) = (2048usize, 2048usize, 2048usize);
    let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 1e-5).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 2e-5).collect();
    c.bench_function("linalg/matmul_2d_2048x2048x2048_f64", |bencher| {
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

// CNN-layer conv2d: batch 4, 64x64x32 input, 3x3x64 kernel, SAME. Large enough
// that the dense im2col + GEMM trailing matmul auto-threads — the regime where
// im2col decisively beats the direct scalar-accumulate loop.
fn bench_conv2d_64x64x32_3x3x64_f64(c: &mut Criterion) {
    let (batch, h, w, c_in) = (4usize, 64usize, 64usize, 32usize);
    let (kh, kw, c_out) = (3usize, 3usize, 64usize);
    let lhs_data: Vec<f64> = (0..batch * h * w * c_in)
        .map(|i| ((i as f64) * 0.0011).sin())
        .collect();
    let rhs_data: Vec<f64> = (0..kh * kw * c_in * c_out)
        .map(|i| ((i as f64) * 0.0019).cos())
        .collect();
    let lhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![batch as u32, h as u32, w as u32, c_in as u32],
            },
            lhs_data,
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![kh as u32, kw as u32, c_in as u32, c_out as u32],
            },
            rhs_data,
        )
        .unwrap(),
    );
    let mut p = no_params();
    p.insert("padding".to_owned(), "same".to_owned());
    p.insert("strides".to_owned(), "1".to_owned());
    c.bench_function("eval/conv2d_64x64x32_3x3x64_f64", |bencher| {
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

// Trailing-axis f64 reduction [4096,128] over axis 1: 524288 elems, 4096 output
// rows. Triggers the threaded dense-f64 trailing-axis path — moderate regime
// where flat all-core fan-out was spawn-overhead-dominated.
fn bench_reduce_sum_4096x128_axis1_f64(c: &mut Criterion) {
    let (rows, cols) = (4096usize, 128usize);
    let data: Vec<f64> = (0..rows * cols).map(|i| (i as f64) * 1e-4).collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axes".to_string(), "1".to_string());
    c.bench_function("eval/reduce_sum_4096x128_axis1_f64", |bencher| {
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

// 4M f64 Cumsum along a single long axis (outer_count == 1): only chunked
// parallel-prefix can speed this up (line-threading does nothing with one line).
fn bench_cumsum_4m_f64_1d(c: &mut Criterion) {
    let data: Vec<f64> = (0..1 << 22).map(|i| (i as f64) * 0.001).collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("axis".to_owned(), "0".to_owned());
    c.bench_function("eval/cumsum_4m_f64_1d", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cumsum, std::slice::from_ref(&input), &p))
    });
}

// 4096x1024 f64 Cumsum along the last axis (4096 independent lines): exercises
// line-parallel scan over the outer dimension.
fn bench_cumsum_4096x1024_f64_axis1(c: &mut Criterion) {
    let data: Vec<f64> = (0..4096 * 1024).map(|i| (i as f64) * 0.001).collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![4096, 1024],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axis".to_owned(), "1".to_owned());
    c.bench_function("eval/cumsum_4096x1024_f64_axis1", |bencher| {
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
        bencher.iter(|| {
            eval_primitive(
                Primitive::ConvertElementType,
                std::slice::from_ref(&input),
                &p,
            )
        })
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

// OneHot: 2048 i64 indices -> num_classes=512 f64 (~1M output). Fill+scatter
// dense fast path (pass107) vs the generic per-element decode + Vec<Literal>.
fn bench_one_hot_2048x512_f64(c: &mut Criterion) {
    let data: Vec<i64> = (0..2048).map(|i| (i as i64 * 7) % 512).collect();
    let input = Value::vector_i64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("num_classes".to_owned(), "512".to_owned());
    p.insert("dtype".to_owned(), "f64".to_owned());
    c.bench_function("eval/one_hot_2048x512_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::OneHot, std::slice::from_ref(&input), &p))
    });
}

// Rev (reverse both axes of a 256x256 f64 tensor): dense odometer-gather fast
// path (pass106) vs the generic per-element vec![0;rank] + decode + Vec<Literal>.
fn bench_rev_256x256_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..256 * 256).map(|i| (i as f64) * 0.001 - 5.0).collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![256, 256],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axes".to_owned(), "0,1".to_owned());
    c.bench_function("eval/rev_256x256_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Rev, std::slice::from_ref(&input), &p))
    });
}

// Dense i64 BroadcastedIota over a large 2-D shape: dense fast path (pass108,
// new_i64_values) vs the generic per-element Literal build (+ per-element Result
// from literal_from_index_for_dtype). Output is built fresh every call (no input
// to amortize), so the Vec<Literal> output build dominates the generic path.
fn bench_broadcasted_iota_512x512_i64(c: &mut Criterion) {
    let mut p = BTreeMap::new();
    p.insert("shape".to_owned(), "512,512".to_owned());
    p.insert("dimension".to_owned(), "1".to_owned());
    p.insert("dtype".to_owned(), "i64".to_owned());
    c.bench_function("eval/broadcasted_iota_512x512_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::BroadcastedIota, &[], &p))
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

// Last-axis partial reduction: 4096x1024 f64 reduce axis 1 -> [4096]. Each of the
// 4096 output rows reduces a contiguous input block independently.
fn bench_reduce_sum_4096x1024_axis1_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..4096 * 1024).map(|i| (i as f64) * 0.001).collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![4096, 1024],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axes".to_owned(), "1".to_owned());
    c.bench_function("eval/reduce_sum_4096x1024_axis1_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// Leading-axis partial reduction: 4096x1024 f64 reduce axis 0 -> [1024]. Column
// reduction — vectorizable contiguous out[j] op= in[k*1024+j] block accumulation.
fn bench_reduce_sum_4096x1024_axis0_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..4096 * 1024).map(|i| (i as f64) * 0.001).collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![4096, 1024],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axes".to_owned(), "0".to_owned());
    c.bench_function("eval/reduce_sum_4096x1024_axis0_f64", |bencher| {
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

// 1M elements: above PARALLEL_MIN_ELEMS (262144), so the compute-bound exp fans
// out across cores. Measures the serial→threaded migration for exp/ln/sin/cos/tan.
fn bench_exp_1m(c: &mut Criterion) {
    let data: Vec<f64> = (0..4 * LARGE_RANDOM_LEN)
        .map(|i| (i as f64) * 1e-6 - 8.0)
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/exp_4m_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Exp, std::slice::from_ref(&input), &p))
    });
}

// 512k elements: just above PARALLEL_MIN_ELEMS — the moderate regime where a flat
// all-core fan-out (64 sequential OS-thread spawns) is spawn-overhead-dominated,
// and work-scaled thread counts win.
fn bench_exp_512k(c: &mut Criterion) {
    let data: Vec<f64> = (0..524_288usize).map(|i| (i as f64) * 1e-5 - 4.0).collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/exp_512k_f64", |bencher| {
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

fn bench_complex_mul_1m_literal(c: &mut Criterion) {
    let lhs = complex_vector(1 << 20);
    let rhs = complex_vector(1 << 20);
    let inputs = [lhs, rhs];
    let p = no_params();
    c.bench_function("eval/complex_mul_1m_complex128_literal", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &inputs, &p))
    });
}

fn bench_complex_mul_1m_dense(c: &mut Criterion) {
    let lhs = complex_vector_dense(1 << 20);
    let rhs = complex_vector_dense(1 << 20);
    let inputs = [lhs, rhs];
    let p = no_params();
    c.bench_function("eval/complex_mul_1m_complex128_dense", |bencher| {
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

// 256k dense-BF16 Exp2 (threaded): exercises the dense half-float unary fast path
// (u16 = 2B/elem, no per-`Literal` materialization, compute-bound transcendental so it
// threads). Paired with the boxed-`Literal` reference below for a same-invocation A/B
// (cross-invocation rch timings drift 20–60%).
fn bench_bf16_exp2_256k_dense(c: &mut Criterion) {
    let n = 1usize << 18;
    let bits: Vec<u16> = (0..n)
        .map(
            |i| match Literal::from_bf16_f64((i as f64 * 0.0007).sin().abs() * 3.0 + 0.5) {
                Literal::BF16Bits(b) => b,
                _ => 0,
            },
        )
        .collect();
    let input = Value::Tensor(
        TensorValue::new_half_float_values(
            DType::BF16,
            Shape {
                dims: vec![n as u32],
            },
            bits,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/exp2_256k_bf16_dense", |bencher| {
        bencher
            .iter(|| eval_primitive(Primitive::Exp2, std::slice::from_ref(black_box(&input)), &p))
    });
}

// Same workload via the boxed per-`Literal` path: `TensorValue::new` makes
// `as_half_float_slice()` return None, bypassing the dense fast path (the pre-change
// behavior — serial per-`Literal` map).
fn bench_bf16_exp2_256k_boxed(c: &mut Criterion) {
    let n = 1usize << 18;
    let lits: Vec<Literal> = (0..n)
        .map(|i| Literal::from_bf16_f64((i as f64 * 0.0007).sin().abs() * 3.0 + 0.5))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::BF16,
            Shape {
                dims: vec![n as u32],
            },
            lits,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/exp2_256k_bf16_boxed", |bencher| {
        bencher
            .iter(|| eval_primitive(Primitive::Exp2, std::slice::from_ref(black_box(&input)), &p))
    });
}

// 2M dense-BF16 same-shape Mul: exercises the dense half-float binary fast path (u16 =
// 2B/elem, no per-`Literal` materialization/dispatch). Cheap binop = memory-bound, not
// threaded — the win is pure per-`Literal`-dispatch elimination. Paired with the boxed
// reference for a same-invocation A/B.
fn bf16_bits_vec(n: usize, f: impl Fn(usize) -> f64) -> Vec<u16> {
    (0..n)
        .map(|i| match Literal::from_bf16_f64(f(i)) {
            Literal::BF16Bits(b) => b,
            _ => 0,
        })
        .collect()
}

fn bench_bf16_mul_2m_dense(c: &mut Criterion) {
    let n = 1usize << 21;
    let shape = Shape {
        dims: vec![n as u32],
    };
    let lhs = Value::Tensor(
        TensorValue::new_half_float_values(
            DType::BF16,
            shape.clone(),
            bf16_bits_vec(n, |i| (i as f64 * 0.013).sin() * 2.0),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_half_float_values(
            DType::BF16,
            shape,
            bf16_bits_vec(n, |i| (i as f64 * 0.0071).cos() + 1.5),
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/mul_2m_bf16_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_bf16_mul_2m_boxed(c: &mut Criterion) {
    let n = 1usize << 21;
    let shape = Shape {
        dims: vec![n as u32],
    };
    let boxed = |f: &dyn Fn(usize) -> f64| {
        let lits: Vec<Literal> = (0..n).map(|i| Literal::from_bf16_f64(f(i))).collect();
        Value::Tensor(TensorValue::new(DType::BF16, shape.clone(), lits).unwrap())
    };
    let lhs = boxed(&|i| (i as f64 * 0.013).sin() * 2.0);
    let rhs = boxed(&|i| (i as f64 * 0.0071).cos() + 1.5);
    let p = no_params();
    c.bench_function("eval/mul_2m_bf16_boxed", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[lhs.clone(), rhs.clone()], &p))
    });
}

// BF16 scalar*tensor (2M): dense half-float scalar-broadcast vs boxed (same-invocation
// A/B). Scaling a bf16 tensor by a scalar (attention 1/sqrt(d), norm scale, LR) is common.
fn bench_bf16_scalarmul_2m_dense(c: &mut Criterion) {
    let n = 1usize << 21;
    let t = Value::Tensor(
        TensorValue::new_half_float_values(
            DType::BF16,
            Shape {
                dims: vec![n as u32],
            },
            bf16_bits_vec(n, |i| (i as f64 * 0.013).sin() * 2.0),
        )
        .unwrap(),
    );
    let s = Value::Scalar(Literal::from_bf16_f64(1.75));
    let p = no_params();
    c.bench_function("eval/scalarmul_2m_bf16_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[t.clone(), s.clone()], &p))
    });
}

fn bench_bf16_scalarmul_2m_boxed(c: &mut Criterion) {
    let n = 1usize << 21;
    let lits: Vec<Literal> = (0..n)
        .map(|i| Literal::from_bf16_f64((i as f64 * 0.013).sin() * 2.0))
        .collect();
    let t = Value::Tensor(
        TensorValue::new(
            DType::BF16,
            Shape {
                dims: vec![n as u32],
            },
            lits,
        )
        .unwrap(),
    );
    let s = Value::Scalar(Literal::from_bf16_f64(1.75));
    let p = no_params();
    c.bench_function("eval/scalarmul_2m_bf16_boxed", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Mul, &[t.clone(), s.clone()], &p))
    });
}

// BF16 bias-add broadcast [4096,512] + [512] (2.1M out): dense half-float broadcast fast
// path vs the boxed per-`Literal` broadcast (same-invocation A/B). bias-add after a matmul
// is the ubiquitous NN pattern.
fn bench_bf16_biasadd_dense(c: &mut Criterion) {
    let (rows, cols) = (4096usize, 512usize);
    let shape_m = Shape {
        dims: vec![rows as u32, cols as u32],
    };
    let shape_b = Shape {
        dims: vec![cols as u32],
    };
    let mat = Value::Tensor(
        TensorValue::new_half_float_values(
            DType::BF16,
            shape_m,
            bf16_bits_vec(rows * cols, |i| (i as f64 * 0.0013).sin() * 2.0),
        )
        .unwrap(),
    );
    let bias = Value::Tensor(
        TensorValue::new_half_float_values(
            DType::BF16,
            shape_b,
            bf16_bits_vec(cols, |j| (j as f64 * 0.07).cos() + 1.5),
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/biasadd_bf16_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[mat.clone(), bias.clone()], &p))
    });
}

fn bench_bf16_biasadd_boxed(c: &mut Criterion) {
    let (rows, cols) = (4096usize, 512usize);
    let mat = {
        let lits: Vec<Literal> = (0..rows * cols)
            .map(|i| Literal::from_bf16_f64((i as f64 * 0.0013).sin() * 2.0))
            .collect();
        Value::Tensor(
            TensorValue::new(
                DType::BF16,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                lits,
            )
            .unwrap(),
        )
    };
    let bias = {
        let lits: Vec<Literal> = (0..cols)
            .map(|j| Literal::from_bf16_f64((j as f64 * 0.07).cos() + 1.5))
            .collect();
        Value::Tensor(
            TensorValue::new(
                DType::BF16,
                Shape {
                    dims: vec![cols as u32],
                },
                lits,
            )
            .unwrap(),
        )
    };
    let p = no_params();
    c.bench_function("eval/biasadd_bf16_boxed", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Add, &[mat.clone(), bias.clone()], &p))
    });
}

// Depthwise conv2d [1,56,56,128] 3x3 VALID (MobileNet-style). A/B: the dense channel-
// vectorized fast path (eval_primitive) vs the pre-change per-output-channel scalar loop
// that read the window channel-strided once per output channel (replicated locally).
fn depthwise_conv2d_general_ref(
    x: &[f64],
    k: &[f64],
    h: usize,
    w: usize,
    c: usize,
    kh: usize,
    kw: usize,
) -> Vec<f64> {
    let (oh, ow) = (h - kh + 1, w - kw + 1);
    let wc = w * c;
    let mut out = vec![0.0f64; oh * ow * c];
    let mut oi = 0usize;
    for o_h in 0..oh {
        for o_w in 0..ow {
            for co in 0..c {
                let mut acc = 0.0f64;
                for a in 0..kh {
                    for b in 0..kw {
                        acc += x[(o_h + a) * wc + (o_w + b) * c + co] * k[(a * kw + b) * c + co];
                    }
                }
                out[oi] = acc;
                oi += 1;
            }
        }
    }
    out
}

fn depthwise_conv_inputs() -> (Vec<f64>, Vec<f64>, usize, usize, usize, usize, usize) {
    let (h, w, c, kh, kw) = (56usize, 56usize, 128usize, 3usize, 3usize);
    let x: Vec<f64> = (0..h * w * c).map(|i| (i as f64 * 0.0011).sin()).collect();
    let k: Vec<f64> = (0..kh * kw * c)
        .map(|i| (i as f64 * 0.0023).cos())
        .collect();
    (x, k, h, w, c, kh, kw)
}

// Depthwise conv1d [1,1024,256] k=5 VALID (audio/sequence depthwise). A/B: channel-
// vectorized fast path vs the pre-change per-output-channel channel-strided scalar loop.
fn depthwise_conv1d_general_ref(x: &[f64], k: &[f64], w: usize, c: usize, kw: usize) -> Vec<f64> {
    let ow = w - kw + 1;
    let mut out = vec![0.0f64; ow * c];
    let mut oi = 0usize;
    for o_w in 0..ow {
        for co in 0..c {
            let mut acc = 0.0f64;
            for a in 0..kw {
                acc += x[(o_w + a) * c + co] * k[a * c + co];
            }
            out[oi] = acc;
            oi += 1;
        }
    }
    out
}

fn depthwise_conv1d_inputs() -> (Vec<f64>, Vec<f64>, usize, usize, usize) {
    let (w, c, kw) = (1024usize, 256usize, 5usize);
    let x: Vec<f64> = (0..w * c).map(|i| (i as f64 * 0.0011).sin()).collect();
    let k: Vec<f64> = (0..kw * c).map(|i| (i as f64 * 0.0023).cos()).collect();
    (x, k, w, c, kw)
}

fn bench_depthwise_conv1d_general(c: &mut Criterion) {
    let (x, k, w, ch, kw) = depthwise_conv1d_inputs();
    c.bench_function("conv/depthwise1d_1024x256_k5_general", |b| {
        b.iter(|| depthwise_conv1d_general_ref(black_box(&x), black_box(&k), w, ch, kw))
    });
}

fn bench_depthwise_conv1d_fast(c: &mut Criterion) {
    let (x, k, w, ch, kw) = depthwise_conv1d_inputs();
    let x_val = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![1, w as u32, ch as u32],
            },
            x,
        )
        .unwrap(),
    );
    let k_val = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![kw as u32, 1, ch as u32],
            },
            k,
        )
        .unwrap(),
    );
    let mut p = no_params();
    p.insert("padding".to_owned(), "valid".to_owned());
    p.insert("strides".to_owned(), "1".to_owned());
    p.insert("feature_group_count".to_owned(), ch.to_string());
    c.bench_function("conv/depthwise1d_1024x256_k5_fast", |b| {
        b.iter(|| eval_primitive(Primitive::Conv, &[x_val.clone(), k_val.clone()], &p))
    });
}

fn bench_depthwise_conv_general(c: &mut Criterion) {
    let (x, k, h, w, ch, kh, kw) = depthwise_conv_inputs();
    c.bench_function("conv/depthwise_56x56x128_3x3_general", |b| {
        b.iter(|| depthwise_conv2d_general_ref(black_box(&x), black_box(&k), h, w, ch, kh, kw))
    });
}

fn bench_depthwise_conv_fast(c: &mut Criterion) {
    let (x, k, h, w, ch, kh, kw) = depthwise_conv_inputs();
    let x_val = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![1, h as u32, w as u32, ch as u32],
            },
            x,
        )
        .unwrap(),
    );
    let k_val = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![kh as u32, kw as u32, 1, ch as u32],
            },
            k,
        )
        .unwrap(),
    );
    let mut p = no_params();
    p.insert("padding".to_owned(), "valid".to_owned());
    p.insert("strides".to_owned(), "1".to_owned());
    p.insert("feature_group_count".to_owned(), ch.to_string());
    c.bench_function("conv/depthwise_56x56x128_3x3_fast", |b| {
        b.iter(|| eval_primitive(Primitive::Conv, &[x_val.clone(), k_val.clone()], &p))
    });
}

// Grouped conv2d [1,28,28,256] 3x3 G=32 (cpg=8, rhs_cin=8) VALID — ResNeXt-style. A/B:
// AXPY fast path vs the pre-change per-output-channel channel-strided scalar loop.
fn grouped_conv_inputs() -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
) {
    let (h, w, cin, cout, kh, kw, g) = (
        28usize, 28usize, 256usize, 256usize, 3usize, 3usize, 32usize,
    );
    let rhs_cin = cin / g;
    let x: Vec<f64> = (0..h * w * cin)
        .map(|i| (i as f64 * 0.0011).sin())
        .collect();
    let k: Vec<f64> = (0..kh * kw * rhs_cin * cout)
        .map(|i| (i as f64 * 0.0023).cos())
        .collect();
    (x, k, h, w, cin, cout, kh, kw, g)
}

fn grouped_conv2d_general_ref(
    x: &[f64],
    k: &[f64],
    h: usize,
    w: usize,
    cin: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    g: usize,
) -> Vec<f64> {
    let (oh, ow) = (h - kh + 1, w - kw + 1);
    let rhs_cin = cin / g;
    let cpg = cout / g;
    let wc = w * cin;
    let mut out = vec![0.0f64; oh * ow * cout];
    let mut oi = 0usize;
    for o_h in 0..oh {
        for o_w in 0..ow {
            for co in 0..cout {
                let in_base = (co / cpg) * rhs_cin;
                let mut acc = 0.0f64;
                for a in 0..kh {
                    for b in 0..kw {
                        for ci in 0..rhs_cin {
                            acc += x[(o_h + a) * wc + (o_w + b) * cin + in_base + ci]
                                * k[((a * kw + b) * rhs_cin + ci) * cout + co];
                        }
                    }
                }
                out[oi] = acc;
                oi += 1;
            }
        }
    }
    out
}

fn bench_grouped_conv_general(c: &mut Criterion) {
    let (x, k, h, w, cin, cout, kh, kw, g) = grouped_conv_inputs();
    c.bench_function("conv/grouped_28x28x256_3x3_g32_general", |b| {
        b.iter(|| {
            grouped_conv2d_general_ref(black_box(&x), black_box(&k), h, w, cin, cout, kh, kw, g)
        })
    });
}

fn bench_grouped_conv_fast(c: &mut Criterion) {
    let (x, k, h, w, cin, cout, kh, kw, g) = grouped_conv_inputs();
    let rhs_cin = cin / g;
    let x_val = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![1, h as u32, w as u32, cin as u32],
            },
            x,
        )
        .unwrap(),
    );
    let k_val = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![kh as u32, kw as u32, rhs_cin as u32, cout as u32],
            },
            k,
        )
        .unwrap(),
    );
    let mut p = no_params();
    p.insert("padding".to_owned(), "valid".to_owned());
    p.insert("strides".to_owned(), "1".to_owned());
    p.insert("feature_group_count".to_owned(), g.to_string());
    c.bench_function("conv/grouped_28x28x256_3x3_g32_fast", |b| {
        b.iter(|| eval_primitive(Primitive::Conv, &[x_val.clone(), k_val.clone()], &p))
    });
}

// Grouped conv1d [1,512,256] k=3 G=32 (cpg=8, rhs_cin=8) VALID. A/B: AXPY fast path vs
// the pre-change per-output-channel channel-strided scalar loop.
fn grouped_conv1d_inputs() -> (Vec<f64>, Vec<f64>, usize, usize, usize, usize, usize) {
    let (w, cin, cout, kw, g) = (512usize, 256usize, 256usize, 3usize, 32usize);
    let rhs_cin = cin / g;
    let x: Vec<f64> = (0..w * cin).map(|i| (i as f64 * 0.0011).sin()).collect();
    let k: Vec<f64> = (0..kw * rhs_cin * cout)
        .map(|i| (i as f64 * 0.0023).cos())
        .collect();
    (x, k, w, cin, cout, kw, g)
}

fn grouped_conv1d_general_ref(
    x: &[f64],
    k: &[f64],
    w: usize,
    cin: usize,
    cout: usize,
    kw: usize,
    g: usize,
) -> Vec<f64> {
    let ow = w - kw + 1;
    let rhs_cin = cin / g;
    let cpg = cout / g;
    let mut out = vec![0.0f64; ow * cout];
    let mut oi = 0usize;
    for o_w in 0..ow {
        for co in 0..cout {
            let in_base = (co / cpg) * rhs_cin;
            let mut acc = 0.0f64;
            for a in 0..kw {
                for ci in 0..rhs_cin {
                    acc += x[(o_w + a) * cin + in_base + ci] * k[(a * rhs_cin + ci) * cout + co];
                }
            }
            out[oi] = acc;
            oi += 1;
        }
    }
    out
}

fn bench_grouped_conv1d_general(c: &mut Criterion) {
    let (x, k, w, cin, cout, kw, g) = grouped_conv1d_inputs();
    c.bench_function("conv/grouped1d_512x256_k3_g32_general", |b| {
        b.iter(|| grouped_conv1d_general_ref(black_box(&x), black_box(&k), w, cin, cout, kw, g))
    });
}

fn bench_grouped_conv1d_fast(c: &mut Criterion) {
    let (x, k, w, cin, cout, kw, g) = grouped_conv1d_inputs();
    let rhs_cin = cin / g;
    let x_val = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![1, w as u32, cin as u32],
            },
            x,
        )
        .unwrap(),
    );
    let k_val = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![kw as u32, rhs_cin as u32, cout as u32],
            },
            k,
        )
        .unwrap(),
    );
    let mut p = no_params();
    p.insert("padding".to_owned(), "valid".to_owned());
    p.insert("strides".to_owned(), "1".to_owned());
    p.insert("feature_group_count".to_owned(), g.to_string());
    c.bench_function("conv/grouped1d_512x256_k3_g32_fast", |b| {
        b.iter(|| eval_primitive(Primitive::Conv, &[x_val.clone(), k_val.clone()], &p))
    });
}

fn bench_complex_expm1_1k(c: &mut Criterion) {
    let input = complex_vector(1000);
    let p = no_params();
    c.bench_function("eval/expm1_1k_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Expm1, std::slice::from_ref(&input), &p))
    });
}

// 256k dense-complex128 Exp: complex_exp = exp + sin + cos per element (very
// compute-heavy) — exercises the dense + threaded complex unary map.
fn bench_complex_exp_256k_dense(c: &mut Criterion) {
    let input = complex_vector_dense(1 << 18);
    let p = no_params();
    c.bench_function("eval/exp_256k_complex128_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Exp, std::slice::from_ref(&input), &p))
    });
}

// 256k dense-complex128 Pow: complex_pow = log + mul + exp per element (very
// compute-heavy) — exercises the dense + threaded complex binary map.
fn bench_complex_pow_256k_dense(c: &mut Criterion) {
    let lhs = complex_vector_dense(1 << 18);
    let rhs = complex_vector_dense(1 << 18);
    let p = no_params();
    c.bench_function("eval/pow_256k_complex128_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Pow, &[lhs.clone(), rhs.clone()], &p))
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

// Batched power-of-two FFT along the last axis: 2048 rows of length 256.
// 256 is a power of two so the radix-2 transform is fast (O(n log n) with cheap
// butterflies) — making the serial complex<->Literal conversion the dominant
// cost. This is the scenario where dense Complex storage (as_complex_slice) wins.
fn bench_fft_batch_2048x256(c: &mut Criterion) {
    let input = complex_matrix(2048, 256);
    let p = no_params();
    c.bench_function("eval/fft_batch_2048x256_complex128", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Fft, std::slice::from_ref(&input), &p))
    });
}

// Same FFT as `bench_fft_batch_2048x256` but with a dense-complex-backed input,
// so `extract_tensor_complex` borrows the packed `(re, im)` slice instead of
// converting 524288 `Literal`s one by one. Isolates the dense-extract win.
fn bench_fft_batch_2048x256_dense_input(c: &mut Criterion) {
    let input = complex_matrix_dense(2048, 256);
    let p = no_params();
    c.bench_function(
        "eval/fft_batch_2048x256_complex128_dense_input",
        |bencher| bencher.iter(|| eval_primitive(Primitive::Fft, std::slice::from_ref(&input), &p)),
    );
}

// Same-binary head-to-head of the two complex-output build strategies for a
// 512k-element complex128 buffer: dense packed `(re, im)` storage vs the
// `Vec<Literal>` build. Isolates output-build cost from extract/transform.
fn bench_complex_build_dense_512k(c: &mut Criterion) {
    let values: Vec<(f64, f64)> = (0..2048 * 256)
        .map(|i| {
            let x = i as f64;
            ((x * 0.125).sin(), (x * 0.25).cos())
        })
        .collect();
    let shape = Shape {
        dims: vec![2048, 256],
    };
    c.bench_function("eval/complex_build_dense_512k", |bencher| {
        bencher.iter(|| {
            TensorValue::new_complex_values(DType::Complex128, shape.clone(), values.clone())
                .unwrap()
        })
    });
}

fn bench_complex_build_literal_512k(c: &mut Criterion) {
    let values: Vec<(f64, f64)> = (0..2048 * 256)
        .map(|i| {
            let x = i as f64;
            ((x * 0.125).sin(), (x * 0.25).cos())
        })
        .collect();
    let shape = Shape {
        dims: vec![2048, 256],
    };
    c.bench_function("eval/complex_build_literal_512k", |bencher| {
        bencher.iter(|| {
            let lits: Vec<Literal> = values
                .iter()
                .map(|&(re, im)| Literal::from_complex128(re, im))
                .collect();
            TensorValue::new(DType::Complex128, shape.clone(), lits).unwrap()
        })
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

// Same-invocation Bluestein control: 1003 = 17·59 has a prime factor > 13, so it
// stays on the Bluestein path while 64x1000 (2^3·5^3) takes mixed-radix.
fn bench_rfft_batch_64x1003_bluestein(c: &mut Criterion) {
    let input = real_matrix(64, 1003);
    let p = no_params();
    c.bench_function("eval/rfft_batch_64x1003_bluestein_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Rfft, std::slice::from_ref(&input), &p))
    });
}

// Padding-heavy case: 1500 = 2^2·3·5^3 is far above the previous power of two, so
// Bluestein would pad to 4096 (~2.7x). Mixed-radix transforms it natively. The
// 1499 (prime) control pads to 4096 too and stays on Bluestein — same invocation.
fn bench_rfft_batch_64x1500_mixed(c: &mut Criterion) {
    let input = real_matrix(64, 1500);
    let p = no_params();
    c.bench_function("eval/rfft_batch_64x1500_mixed_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Rfft, std::slice::from_ref(&input), &p))
    });
}

fn bench_rfft_batch_64x1499_bluestein(c: &mut Criterion) {
    let input = real_matrix(64, 1499);
    let p = no_params();
    c.bench_function("eval/rfft_batch_64x1499_bluestein_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Rfft, std::slice::from_ref(&input), &p))
    });
}

// Batched power-of-two real FFT: 2048 rows of length 256. The radix-2 transform
// is cheap so the per-row work fans out across threads (dense complex output).
fn bench_rfft_batch_2048x256(c: &mut Criterion) {
    let input = real_matrix(2048, 256);
    let p = no_params();
    c.bench_function("eval/rfft_batch_2048x256_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Rfft, std::slice::from_ref(&input), &p))
    });
}

// FFT of a real dense-f64 signal (fft(real)): exercises the dense-f64 extract
// fast path (bulk slice read -> (v,0.0)) + threaded transform + dense output.
fn bench_fft_batch_2048x256_real_dense(c: &mut Criterion) {
    let values: Vec<f64> = (0..2048 * 256)
        .map(|i| {
            let x = i as f64;
            (x * 0.125).sin() + (x * 0.03125).cos()
        })
        .collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![2048, 256],
            },
            values,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/fft_batch_2048x256_real_dense_input", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Fft, std::slice::from_ref(&input), &p))
    });
}

// Dense-f64-input variant: real input backed by packed f64 storage (new_f64_values),
// the representation upstream f64 ops produce — extract reads the bulk slice
// instead of converting 524288 Literals one by one.
fn bench_rfft_batch_2048x256_dense_input(c: &mut Criterion) {
    let values: Vec<f64> = (0..2048 * 256)
        .map(|i| {
            let x = i as f64;
            (x * 0.125).sin() + (x * 0.03125).cos()
        })
        .collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![2048, 256],
            },
            values,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/rfft_batch_2048x256_f64_dense_input", |bencher| {
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

// Batched IRFFT: 2048 rows, fft_length 256 (input half-spectrum 129 wide). Each
// row reconstructs the Hermitian spectrum + runs an inverse transform + real
// extraction independently — compute-heavy, threads cleanly into dense f64 output.
fn bench_irfft_batch_2048x256(c: &mut Criterion) {
    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "256".to_owned());
    let full = eval_primitive(Primitive::Rfft, &[real_matrix(2048, 256)], &params).unwrap();
    c.bench_function("eval/irfft_batch_2048x256_complex128", |bencher| {
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
    bench_random_uniform_1m,
    bench_random_normal_1m,
    bench_add_scalar,
    bench_add_1k_vector,
    bench_mul_1k_vector,
    bench_add_1k_f64_vector,
    bench_pow_broadcast_1024x1024_f64,
    bench_pow_1m_f64_vec,
    bench_pow_scalar_1m_f64_vec,
    bench_pow_scalar_1m_f64_literal_reference,
    bench_atan2_1m_f64_vec,
    bench_erf_1m_f64_vec,
    bench_polygamma_n2_256k_f64,
    bench_igamma_256k_f64,
    bench_cbrt_1m_f64_vec,
    bench_atan2_scalar_1m_f64_vec,
    bench_atan2_scalar_1m_f64_literal_reference,
    bench_div_1k_f64_vector,
    bench_add_64k_f64_vec,
    bench_add_broadcast_row_1024x1024_f64,
    bench_add_broadcast_col_1024x1024_f64,
    bench_add_broadcast_row_1024x1024_i64,
    bench_add_broadcast_col_1024x1024_i64,
    bench_bitand_same_shape_1024x1024_i64,
    bench_select_1024x1024_f64,
    bench_lt_same_shape_1024x1024_f64,
    bench_lt_broadcast_row_1024x1024_f64,
    bench_lt_broadcast_row_1024x1024_i64,
    bench_mul_broadcast_row_512x512_c128,
    bench_mul_broadcast_col_512x512_c128,
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
    bench_eigh_48,
    bench_cholesky_128_f64,
    bench_cholesky_512_f64,
    bench_cholesky_1024_f64,
    bench_qr_128_f64,
    bench_qr_1024_f64,
    bench_pinv_256x128_f64,
    bench_pinv_192x192_f64,
    bench_lu_128_f64,
    bench_lu_1024_f64,
    bench_svd_48_f64,
    bench_svd_48_complex_path,
    bench_svd_48_full_f64,
    bench_svd_48_full_complex_path,
    bench_matmul_2d_256,
    bench_matmul_2d_512,
    bench_matmul_2d_1024,
    bench_matmul_2d_2048,
    bench_conv2d_32x32x8_3x3x16_f64_vec,
    bench_conv2d_64x64x32_3x3x64_f64,
    bench_conv2d_32x32x8_3x3x16_f64_literal_reference,
    bench_conv1d_1024x16_5x32_f64_vec,
    bench_conv1d_1024x16_5x32_f64_literal_reference,
    bench_solve_24x24_24rhs,
    bench_concat_axis1_3x_f64,
    bench_concat_axis0_3x_f64,
    bench_transpose_256x256_f64,
    bench_reduce_sum_1k,
    bench_reduce_sum_4096x1024_axis1_f64,
    bench_reduce_sum_4096x1024_axis0_f64,
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
    bench_reduce_sum_4096x128_axis1_f64,
    bench_reduce_sum_64k_i64_literal_reference,
    bench_reduce_and_64k_bool_vec,
    bench_reduce_and_64k_bool_literal_reference,
    bench_reduce_and_256_axis1_bool_vec,
    bench_reduce_and_256_axis1_bool_literal_reference,
    bench_cumsum_64k_f64_vec,
    bench_cumsum_4m_f64_1d,
    bench_cumsum_4096x1024_f64_axis1,
    bench_cumsum_64k_f64_literal_reference,
    bench_sort_64k_i64,
    bench_sort_64k_f64,
    bench_argmax_64k_f64,
    bench_sort_64k_f64_descending,
    bench_convert_64k_f64_to_i64,
    bench_broadcast_256_to_256x256_f64,
    bench_pad_256x256_to_258x258_f64,
    bench_rev_256x256_f64,
    bench_one_hot_2048x512_f64,
    bench_broadcasted_iota_512x512_i64,
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
    bench_exp_1m,
    bench_exp_512k,
    bench_square_1k,
    bench_integer_pow_1k,
    bench_clamp_1k,
    bench_select_1k,
    bench_select_64k_i64_vec,
    bench_select_64k_i64_literal_reference,
    bench_complex_mul_1k,
    bench_complex_mul_1m_literal,
    bench_complex_mul_1m_dense,
    bench_complex_ctor_1k,
    bench_complex_conj_1k,
    bench_complex_neg_1k,
    bench_bf16_exp2_256k_dense,
    bench_bf16_exp2_256k_boxed,
    bench_bf16_mul_2m_dense,
    bench_bf16_mul_2m_boxed,
    bench_bf16_biasadd_dense,
    bench_bf16_biasadd_boxed,
    bench_depthwise_conv_general,
    bench_depthwise_conv_fast,
    bench_depthwise_conv1d_general,
    bench_depthwise_conv1d_fast,
    bench_grouped_conv_general,
    bench_grouped_conv_fast,
    bench_grouped_conv1d_general,
    bench_grouped_conv1d_fast,
    bench_bf16_scalarmul_2m_dense,
    bench_bf16_scalarmul_2m_boxed,
    bench_complex_expm1_1k,
    bench_complex_exp_256k_dense,
    bench_complex_pow_256k_dense,
    bench_complex_abs_1k,
    bench_complex_real_1k,
    bench_complex_imag_1k,
    bench_complex_is_finite_1k,
    bench_fft_256,
    bench_ifft_256,
    bench_fft_1000,
    bench_fft_1009_prime,
    bench_fft_batch_128x1000,
    bench_fft_batch_2048x256,
    bench_fft_batch_2048x256_dense_input,
    bench_complex_build_dense_512k,
    bench_complex_build_literal_512k,
    bench_rfft_256,
    bench_rfft_batch_64x1000,
    bench_rfft_batch_64x1003_bluestein,
    bench_rfft_batch_64x1500_mixed,
    bench_rfft_batch_64x1499_bluestein,
    bench_rfft_batch_2048x256,
    bench_rfft_batch_2048x256_dense_input,
    bench_fft_batch_2048x256_real_dense,
    bench_irfft_256,
    bench_irfft_batch_2048x256,
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
