use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
// [cc-temp] foreign WIP stub: use fj_lax::linalg::svd_jacobi_profile_counters;
use fj_lax::threefry::{random_key, random_normal, random_uniform};
use fj_lax::{eval_primitive, eval_primitive_multi, simd_exp::simd_poly_exp_into};
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

fn literal_backed_tensor(dtype: DType, shape: Shape, elements: Vec<Literal>) -> Value {
    Value::Tensor(
        TensorValue::new_with_literal_buffer(dtype, shape, fj_core::LiteralBuffer::new(elements))
            .unwrap(),
    )
}

fn literal_backed_vector(dtype: DType, elements: Vec<Literal>) -> Value {
    let len = elements.len() as u32;
    literal_backed_tensor(dtype, Shape { dims: vec![len] }, elements)
}

fn f64_to_u32_bitcast_chunks(value: f64) -> [u32; 2] {
    let bytes = value.to_bits().to_le_bytes();
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    ]
}

fn f32_to_u16_bitcast_chunks(value: f32) -> [u16; 2] {
    let bytes = value.to_bits().to_le_bytes();
    [
        u16::from_le_bytes([bytes[0], bytes[1]]),
        u16::from_le_bytes([bytes[2], bytes[3]]),
    ]
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
    c.bench_function("eval/cbrt_1m_f64_vec_libm_reference", |bencher| {
        bencher.iter(|| {
            let tensor = input.as_tensor().unwrap();
            let src = tensor.elements.as_f64_slice().unwrap();
            let mut out = vec![0.0; src.len()];
            let threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
                .min(2);
            let chunk = src.len().div_ceil(threads);
            std::thread::scope(|scope| {
                for (dst, src) in out.chunks_mut(chunk).zip(src.chunks(chunk)) {
                    scope.spawn(move || {
                        for (d, &v) in dst.iter_mut().zip(src.iter()) {
                            *d = v.cbrt();
                        }
                    });
                }
            });
            black_box(Value::Tensor(
                TensorValue::new_f64_values(tensor.shape.clone(), out).unwrap(),
            ))
        })
    });
    c.bench_function("eval/cbrt_1m_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cbrt, std::slice::from_ref(&input), &p))
    });
}

fn simd_poly_tanh_probe(src: &[f64]) -> Vec<f64> {
    let exponents: Vec<f64> = src.iter().map(|&x| -2.0 * x.abs()).collect();
    let mut out = vec![0.0; src.len()];
    simd_poly_exp_into(&exponents, &mut out);
    for (dst, &x) in out.iter_mut().zip(src) {
        let e = *dst;
        *dst = if x == 0.0 || x.is_nan() {
            x
        } else if x.is_infinite() || x.abs() > 20.0 {
            x.signum()
        } else {
            let y = (1.0 - e) / (1.0 + e);
            if x.is_sign_negative() { -y } else { y }
        };
    }
    out
}

fn bench_tanh_1m_f64_vec(c: &mut Criterion) {
    let a: Vec<f64> = (0..1 << 20)
        .map(|i| ((i % 4001) as f64 - 2000.0) * 0.001)
        .collect();
    let input = Value::vector_f64(&a).unwrap();
    let p = no_params();
    c.bench_function("eval/tanh_1m_f64_vec_libm_reference", |bencher| {
        bencher.iter(|| {
            let tensor = input.as_tensor().unwrap();
            let src = tensor.elements.as_f64_slice().unwrap();
            let mut out = vec![0.0; src.len()];
            let threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
                .min(2);
            let chunk = src.len().div_ceil(threads);
            std::thread::scope(|scope| {
                for (dst, src) in out.chunks_mut(chunk).zip(src.chunks(chunk)) {
                    scope.spawn(move || {
                        for (d, &v) in dst.iter_mut().zip(src.iter()) {
                            *d = v.tanh();
                        }
                    });
                }
            });
            black_box(Value::Tensor(
                TensorValue::new_f64_values(tensor.shape.clone(), out).unwrap(),
            ))
        })
    });
    c.bench_function("eval/tanh_1m_f64_vec_simd_poly_probe", |bencher| {
        bencher.iter(|| black_box(simd_poly_tanh_probe(&a)))
    });
    c.bench_function("eval/tanh_1m_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Tanh, std::slice::from_ref(&input), &p))
    });
}

fn bench_tan_1m_f64_vec(c: &mut Criterion) {
    let a: Vec<f64> = (0..1 << 20)
        .map(|i| ((i % 3001) as f64 - 1500.0) * 0.0005)
        .collect();
    let input = Value::vector_f64(&a).unwrap();
    let p = no_params();
    c.bench_function("eval/tan_1m_f64_vec_libm_reference", |bencher| {
        bencher.iter(|| {
            let tensor = input.as_tensor().unwrap();
            let src = tensor.elements.as_f64_slice().unwrap();
            let mut out = vec![0.0; src.len()];
            let threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
                .min(2);
            let chunk = src.len().div_ceil(threads);
            std::thread::scope(|scope| {
                for (dst, src) in out.chunks_mut(chunk).zip(src.chunks(chunk)) {
                    scope.spawn(move || {
                        for (d, &v) in dst.iter_mut().zip(src.iter()) {
                            *d = v.tan();
                        }
                    });
                }
            });
            black_box(Value::Tensor(
                TensorValue::new_f64_values(tensor.shape.clone(), out).unwrap(),
            ))
        })
    });
    c.bench_function("eval/tan_1m_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Tan, std::slice::from_ref(&input), &p))
    });
}

// 1M sqrt: f64::sqrt is a hardware intrinsic (vsqrtpd) — UNLIKE cbrt/exp which are libm
// libcalls. If the dense unary fast path autovectorizes the monomorphized f64::sqrt, sqrt
// should be MUCH faster than cbrt (same loop shape); if they're similar, sqrt is stuck
// scalar and an explicit std::simd sqrt (bit-identical, no fma) would be a free win.
fn bench_sqrt_1m_f64_vec(c: &mut Criterion) {
    let a: Vec<f64> = (0..1 << 20).map(|i| (i as f64) * 0.0007 + 1.0).collect();
    let input = Value::vector_f64(&a).unwrap();
    let p = no_params();
    c.bench_function("eval/sqrt_1m_f64_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sqrt, std::slice::from_ref(&input), &p))
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

// relu = max(x, 0.0) on 64k f32 (JAX default dtype): dense scalar-broadcast fast
// path (bead frankenjax-d15qd) vs the Vec<Literal> generic per-element broadcast.
// relu/clamp are lone ops (never fuse), so the generic path is what they hit today.
fn relu_f32_dense_64k() -> Value {
    let data: Vec<f32> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| i as f32 * 1e-3 - 30.0)
        .collect();
    Value::Tensor(
        TensorValue::new_f32_values(Shape::vector(LARGE_ELEMENTWISE_LEN as u32), data).unwrap(),
    )
}

fn relu_f32_literal_64k() -> Value {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::from_f32(i as f32 * 1e-3 - 30.0))
        .collect();
    Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    )
}

fn bench_relu_64k_f32_vec(c: &mut Criterion) {
    let tensor = relu_f32_dense_64k();
    let zero = Value::scalar_f32(0.0);
    let p = no_params();
    c.bench_function("eval/relu_64k_f32_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Max, &[tensor.clone(), zero.clone()], &p))
    });
}

fn bench_relu_64k_f32_literal_reference(c: &mut Criterion) {
    let tensor = relu_f32_literal_64k();
    let zero = Value::scalar_f32(0.0);
    let p = no_params();
    c.bench_function("eval/relu_64k_f32_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Max, &[tensor.clone(), zero.clone()], &p))
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

fn bench_concat_axis0_2x512x1024_then_add_f64(c: &mut Criterion) {
    // Hot pattern for frankenjax-dr67k: concatenate dense feature blocks, then
    // immediately feed a dense-reader elementwise op.
    let mk_dense = |rows: usize, cols: usize, base: f64| {
        let values: Vec<f64> = (0..rows * cols)
            .map(|i| base + (i as f64 * 0.000_001))
            .collect();
        Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                values,
            )
            .unwrap(),
        )
    };
    let a = mk_dense(512, 1024, 0.0);
    let b = mk_dense(512, 1024, 10_000.0);
    let zero = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![1024, 1024],
            },
            vec![0.0; 1024 * 1024],
        )
        .unwrap(),
    );
    let mut concat_params = no_params();
    concat_params.insert("dimension".to_owned(), "0".to_owned());
    let add_params = no_params();
    c.bench_function("eval/concat_axis0_2x512x1024_then_add_f64", |bencher| {
        bencher.iter(|| {
            let concat = eval_primitive(
                Primitive::Concatenate,
                &[a.clone(), b.clone()],
                &concat_params,
            )
            .unwrap();
            black_box(eval_primitive(
                Primitive::Add,
                &[concat, zero.clone()],
                &add_params,
            ))
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

// Complex128 [512,512] transpose: dense (f64,f64) cache-blocked path vs the boxed
// per-Literal odometer it used to fall to. Same-invocation A/B (dense storage hits
// the new path; Vec<Literal> storage hits the boxed walk).
fn bench_transpose_512x512_complex128_dense(c: &mut Criterion) {
    let n = 512usize;
    let vals: Vec<(f64, f64)> = (0..(n * n) as i64)
        .map(|i| (i as f64 * 0.001, i as f64 * -0.002))
        .collect();
    let m = Value::Tensor(
        TensorValue::new_complex_values(
            DType::Complex128,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            vals,
        )
        .unwrap(),
    );
    let mut p = no_params();
    p.insert("permutation".to_owned(), "1,0".to_owned());
    c.bench_function("eval/transpose_512x512_complex128_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Transpose, std::slice::from_ref(&m), &p))
    });
}

fn bench_transpose_512x512_complex128_literal_reference(c: &mut Criterion) {
    let n = 512usize;
    let lits: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::from_complex128(i as f64 * 0.001, i as f64 * -0.002))
        .collect();
    let m = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            lits,
        )
        .unwrap(),
    );
    let mut p = no_params();
    p.insert("permutation".to_owned(), "1,0".to_owned());
    c.bench_function("eval/transpose_512x512_complex128_literal_ref", |bencher| {
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

#[allow(dead_code)] // [cc-temp] foreign WIP stub kept to preserve the bench symbol.
fn bench_svd_48_f64_jacobi_counters(_c: &mut Criterion) {}

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

// BF16 GEMM (the dominant TRAINING matmul). Same-invocation A/B isolating the
// native-f32-accumulation lever: f32-accum SIMD (16 lanes + bare-shift widen, XLA
// parity) vs the prior f64-accum SIMD (8 lanes). Compute-bound O(mkn) kernel.
fn bf16_512_inputs() -> (Vec<u16>, Vec<u16>) {
    let (m, k, n) = (512usize, 512usize, 512usize);
    let a: Vec<u16> = (0..m * k)
        .map(|i| match Literal::from_bf16_f64((i as f64) * 1e-4 - 13.0) {
            Literal::BF16Bits(b) => b,
            _ => 0,
        })
        .collect();
    let b: Vec<u16> = (0..k * n)
        .map(|i| match Literal::from_bf16_f64((i as f64) * 2e-4 - 7.0) {
            Literal::BF16Bits(b) => b,
            _ => 0,
        })
        .collect();
    (a, b)
}

fn bench_bf16_matmul_512_f32accum(c: &mut Criterion) {
    let (m, k, n) = (512usize, 512usize, 512usize);
    let (a, b) = bf16_512_inputs();
    c.bench_function("linalg/bf16_matmul_512_f32accum", |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::bf16_matmul_bench(&a, m, k, &b, n, "f32simd"))
    });
}

fn bench_bf16_matmul_512_f64accum_reference(c: &mut Criterion) {
    let (m, k, n) = (512usize, 512usize, 512usize);
    let (a, b) = bf16_512_inputs();
    c.bench_function("linalg/bf16_matmul_512_f64accum_ref", |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::bf16_matmul_bench(&a, m, k, &b, n, "f64simd"))
    });
}

// bf16 GEMM register-blocking A/B at 1024³ (B=2MB, re-streamed per output row by the
// per-row kernel): F32_MR register-blocked (production "f32simd") vs the per-row
// reference ("f32rowref"). Same f32-accum, bit-identical; isolates the register win.
fn bf16_1024_inputs() -> (Vec<u16>, Vec<u16>) {
    let to_bf16 = |i: usize, s: f64| -> u16 {
        match Literal::from_bf16_f64((i as f64) * s) {
            Literal::BF16Bits(b) => b,
            _ => 0,
        }
    };
    let a: Vec<u16> = (0..1024 * 1024).map(|i| to_bf16(i, 1e-4)).collect();
    let b: Vec<u16> = (0..1024 * 1024).map(|i| to_bf16(i, 2e-4)).collect();
    (a, b)
}
fn bench_bf16_matmul_1024_blocked(c: &mut Criterion) {
    let (a, b) = bf16_1024_inputs();
    c.bench_function("linalg/bf16_matmul_1024_blocked", |bencher| {
        bencher.iter(|| {
            fj_lax::tensor_contraction::bf16_matmul_bench(&a, 1024, 1024, &b, 1024, "f32simd")
        })
    });
}
fn bench_bf16_matmul_1024_rowref(c: &mut Criterion) {
    let (a, b) = bf16_1024_inputs();
    c.bench_function("linalg/bf16_matmul_1024_rowref", |bencher| {
        bencher.iter(|| {
            fj_lax::tensor_contraction::bf16_matmul_bench(&a, 1024, 1024, &b, 1024, "f32rowref")
        })
    });
}

// F16 GEMM. Same-invocation A/B isolating the native-f32 lever: native (decode
// F16->f32 once + 16-lane f32-accum GEMM, XLA parity) vs the prior promote path
// (F16->f64 + 8-lane f64 GEMM + round). Compute-bound O(mkn) kernel.
fn f16_512_inputs() -> (Vec<u16>, Vec<u16>) {
    let (m, k, n) = (512usize, 512usize, 512usize);
    let a: Vec<u16> = (0..m * k)
        .map(|i| match Literal::from_f16_f64((i as f64) * 1e-3 - 13.0) {
            Literal::F16Bits(b) => b,
            _ => 0,
        })
        .collect();
    let b: Vec<u16> = (0..k * n)
        .map(|i| match Literal::from_f16_f64((i as f64) * 2e-3 - 7.0) {
            Literal::F16Bits(b) => b,
            _ => 0,
        })
        .collect();
    (a, b)
}

fn bench_f16_matmul_512_f32accum(c: &mut Criterion) {
    let (m, k, n) = (512usize, 512usize, 512usize);
    let (a, b) = f16_512_inputs();
    c.bench_function("linalg/f16_matmul_512_f32accum", |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::f16_matmul_bench(&a, m, k, &b, n, true))
    });
}

fn bench_f16_matmul_512_promote_reference(c: &mut Criterion) {
    let (m, k, n) = (512usize, 512usize, 512usize);
    let (a, b) = f16_512_inputs();
    c.bench_function("linalg/f16_matmul_512_promote_ref", |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::f16_matmul_bench(&a, m, k, &b, n, false))
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

// f32 GEMM cache-blocking A/B: F32_MR register-blocked (production) vs per-row reference.
// The win appears once B (k·n·4B) spills cache and the per-row kernel re-streams it from
// RAM once per output row; the MR-tile loads each B panel once and reuses it across rows.
fn f32_gemm_blocked_ab(c: &mut Criterion, m: usize, k: usize, n: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 1e-4).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 2e-4).cos()).collect();
    c.bench_function(&format!("linalg/f32_gemm_{m}_production"), |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::f32_matmul_bench(&a, m, k, &b, n, "production"))
    });
    c.bench_function(&format!("linalg/f32_gemm_{m}_packed"), |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::f32_matmul_bench(&a, m, k, &b, n, "packed"))
    });
    c.bench_function(&format!("linalg/f32_gemm_{m}_register"), |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::f32_matmul_bench(&a, m, k, &b, n, "register"))
    });
    c.bench_function(&format!("linalg/f32_gemm_{m}_kcblocked"), |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::f32_matmul_bench(&a, m, k, &b, n, "kcblocked"))
    });
}
fn bench_f32_gemm_1024(c: &mut Criterion) {
    f32_gemm_blocked_ab(c, 1024, 1024, 1024);
}
fn bench_f32_gemm_2048(c: &mut Criterion) {
    f32_gemm_blocked_ab(c, 2048, 2048, 2048);
}
fn bench_f32_gemm_4096(c: &mut Criterion) {
    f32_gemm_blocked_ab(c, 4096, 4096, 4096);
}

// f32 conv2d GEMM accumulation A/B at a ResNet-ish im2col shape (out=26x26=676 rows,
// kdim=3*3*32=288, c_out=64). The im2col GATHER is identical for both paths, so the
// GEMM accumulation ratio is the native-f32-conv lever's win: native f32 (16-lane
// f32-accum, XLA parity) vs the prior promote (f32->f64, cache-blocked f64 GEMM).
fn conv_gemm_shapes() -> (usize, usize, usize) {
    (676usize, 288usize, 64usize) // num_rows, kdim, c_out
}
fn bench_conv2d_gemm_f32_native(c: &mut Criterion) {
    let (rows, kdim, cout) = conv_gemm_shapes();
    let col: Vec<f32> = (0..rows * kdim).map(|i| (i as f32 * 1e-3).sin()).collect();
    let filt: Vec<f32> = (0..kdim * cout).map(|i| (i as f32 * 2e-3).cos()).collect();
    c.bench_function("linalg/conv2d_gemm_f32_native", |bencher| {
        bencher.iter(|| {
            fj_lax::tensor_contraction::batched_matmul_2d_f32_in(&col, 1, rows, kdim, &filt, cout)
        })
    });
}
fn bench_conv2d_gemm_f64_promote_reference(c: &mut Criterion) {
    let (rows, kdim, cout) = conv_gemm_shapes();
    let col: Vec<f64> = (0..rows * kdim).map(|i| (i as f64 * 1e-3).sin()).collect();
    let filt: Vec<f64> = (0..kdim * cout).map(|i| (i as f64 * 2e-3).cos()).collect();
    c.bench_function("linalg/conv2d_gemm_f64_promote_ref", |bencher| {
        bencher.iter(|| {
            let out = fj_lax::tensor_contraction::matmul_2d(&col, rows, kdim, &filt, cout);
            std::hint::black_box(out.iter().map(|&v| v as f32).collect::<Vec<f32>>())
        })
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

// Strassen vs matmul_2d same-invocation A/B (load-robust ratio). Strassen only wins
// if the GEMM is compute-bound; if matmul_2d is RAM-bound here, Strassen's extra
// submatrix traffic loses — this measures which regime we're in at n=1024/2048.
fn strassen_ab(c: &mut Criterion, n: usize) {
    let a: Vec<f64> = (0..n * n).map(|i| (i as f64 * 1e-5).sin()).collect();
    let b: Vec<f64> = (0..n * n).map(|i| (i as f64 * 2e-5).cos()).collect();
    c.bench_function(&format!("linalg/strassen_ab_{n}_matmul2d"), |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::matmul_2d(&a, n, n, &b, n))
    });
    c.bench_function(&format!("linalg/strassen_ab_{n}_strassen"), |bencher| {
        bencher.iter(|| fj_lax::tensor_contraction::strassen_matmul_2d(&a, n, n, &b, n))
    });
}
fn bench_strassen_ab_1024(c: &mut Criterion) {
    strassen_ab(c, 1024);
}
fn bench_strassen_ab_2048(c: &mut Criterion) {
    strassen_ab(c, 2048);
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

// Full reduce_sum over a 16M f64 1-D vector -> scalar. The eval is a strict
// single-accumulator sequential f64 fold (non-associative, bit-exact order), which
// is LATENCY-bound (each add depends on the previous), unlike JAX's pairwise sum.
// Measures the vs-JAX gap for the most fundamental reduction.
fn bench_reduce_sum_16m_f64_full(c: &mut Criterion) {
    const N: usize = 16_777_216;
    let data: Vec<f64> = (0..N).map(|i| ((i as f64) * 1.1e-7).sin() * 3.0).collect();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![N as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axes".to_string(), "0".to_string());
    c.bench_function("eval/reduce_sum_16m_f64_full", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// Full reduce_sum over a 16M i64 1-D vector -> scalar. Integer add is ASSOCIATIVE
// (wrapping, mod 2^64), so unlike the f64 sum this fold CAN be reassociated /
// vectorized bit-identically. This measures whether the monomorphized `int_op`
// (i64::wrapping_add) fold already autovectorizes, or whether it is a lever.
fn bench_reduce_sum_16m_i64_full(c: &mut Criterion) {
    const N: usize = 16_777_216;
    let data: Vec<i64> = (0..N as i64).map(|i| (i % 1000) - 500).collect();
    let input = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![N as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axes".to_string(), "0".to_string());
    c.bench_function("eval/reduce_sum_16m_i64_full", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// reduce_sum over a [16384, 1024] f64 tensor along axis 0 (the LEADING/batch axis)
// -> [1024]. The column-accumulation (kept = trailing suffix) path is SERIAL today;
// sum/mean over the batch axis is ubiquitous in ML. Each output column folds its rows
// in ascending order — column-stripe threading preserves that order bit-exactly.
fn bench_reduce_sum_16k_x_1k_axis0_f64(c: &mut Criterion) {
    let (rows, cols) = (16_384usize, 1_024usize);
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i as f64) * 1.3e-6).sin())
        .collect();
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
    p.insert("axes".to_string(), "0".to_string());
    c.bench_function("eval/reduce_sum_16kx1k_axis0_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// reduce_sum over a [16384, 1024] i64 tensor along axis 0 (leading) and axis 1
// (trailing). Integer reduce is ASSOCIATIVE, so these can thread bit-exactly (incl
// contiguous reduce-dimension chunking for the leading case).
fn bench_reduce_sum_16k_x_1k_i64_axes(c: &mut Criterion) {
    let (rows, cols) = (16_384usize, 1_024usize);
    let data: Vec<i64> = (0..(rows * cols) as i64).map(|i| (i % 977) - 488).collect();
    let input = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut p0 = BTreeMap::new();
    p0.insert("axes".to_string(), "0".to_string());
    c.bench_function("eval/reduce_sum_16kx1k_axis0_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p0))
    });
    let mut p1 = BTreeMap::new();
    p1.insert("axes".to_string(), "1".to_string());
    c.bench_function("eval/reduce_sum_16kx1k_axis1_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p1))
    });
}

// argmax over axis 1 of a [16384, 1024] f64 tensor -> [16384]. Each output row's
// argmax is INDEPENDENT, so the along-axis scan can thread over output rows bit-exactly.
fn bench_argmax_16k_x_1k_axis1_f64(c: &mut Criterion) {
    let (rows, cols) = (16_384usize, 1_024usize);
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i as f64) * 1.7e-5).sin())
        .collect();
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
    p.insert("axis".to_string(), "1".to_string());
    c.bench_function("eval/argmax_16kx1k_axis1_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Argmax, std::slice::from_ref(&input), &p))
    });
}

// i64 argmax over axis 1 (contiguous) and axis 0 (strided/leading) of a
// [16384, 1024] i64 tensor. Measures the serial integer paths (no fast path) vs JAX.
fn bench_argmax_16k_x_1k_axis1_i64(c: &mut Criterion) {
    let (rows, cols) = (16_384usize, 1_024usize);
    let data: Vec<i64> = (0..rows * cols).map(|i| (i as i64 % 8191) - 4095).collect();
    let input = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axis".to_string(), "1".to_string());
    c.bench_function("eval/argmax_16kx1k_axis1_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Argmax, std::slice::from_ref(&input), &p))
    });
}

fn bench_argmax_16k_x_1k_axis0_i64(c: &mut Criterion) {
    let (rows, cols) = (16_384usize, 1_024usize);
    let data: Vec<i64> = (0..rows * cols).map(|i| (i as i64 % 8191) - 4095).collect();
    let input = Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axis".to_string(), "0".to_string());
    c.bench_function("eval/argmax_16kx1k_axis0_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Argmax, std::slice::from_ref(&input), &p))
    });
}

// argmax over axis 0 (leading/strided) of a [16384, 1024] f64 tensor -> [1024].
// Each output column scans 16384 rows at stride 1024 (cache-hostile); measures the
// CURRENT serial strided path vs JAX to scope a possible column-block threaded lever.
fn bench_argmax_16k_x_1k_axis0_f64(c: &mut Criterion) {
    let (rows, cols) = (16_384usize, 1_024usize);
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i as f64) * 1.7e-5).sin())
        .collect();
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
    p.insert("axis".to_string(), "0".to_string());
    c.bench_function("eval/argmax_16kx1k_axis0_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Argmax, std::slice::from_ref(&input), &p))
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

// F32 (JAX's DEFAULT float) full reduce-sum over 64k elems: dense F32 storage (the
// new eval_dense_float_full_reduce: read 4-byte contiguous f32 + promote-to-f64 fold)
// vs Vec<Literal> boxed F32Bits (the generic per-element get().as_f64() + 24-byte
// stride). Both accumulate in a single f64 accumulator ascending, so bit-identical.
// Same-invocation A/B. This is the softmax-denominator / loss / norm hot path.
fn bench_reduce_sum_64k_f32_dense(c: &mut Criterion) {
    let data: Vec<f32> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| (i as f32) * 1e-3)
        .collect();
    let input = Value::Tensor(
        TensorValue::new_f32_values(Shape::vector(LARGE_ELEMENTWISE_LEN as u32), data).unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/reduce_sum_64k_f32_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_64k_f32_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::from_f32((i as f32) * 1e-3))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/reduce_sum_64k_f32_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// BF16 (the dominant TRAINING dtype) full reduce-sum over 64k elems: dense packed-u16
// storage (the new eval_dense_float_full_reduce half-float arm: decode u16->f64 fold)
// vs Vec<Literal> boxed BF16Bits (generic per-element as_f64() + 24-byte stride). Both
// fold a single f64 accumulator ascending, so bit-identical. Same-invocation A/B. This
// is the bf16 loss / grad-norm hot path.
// bf16 elementwise mul A/B (same binary): SIMD widen→f64→round-to-odd→RNE vs the scalar
// per-element map. 64k bf16 = 128KB/operand (L2-resident), so this isolates the
// compute (widen/round) floor the SIMD path attacks.
fn bench_bf16_elementwise_mul(c: &mut Criterion) {
    let n = LARGE_ELEMENTWISE_LEN;
    let a: Vec<u16> = (0..n)
        .map(|i| match Literal::from_bf16_f64((i as f64) * 1e-3 - 17.0) {
            Literal::BF16Bits(b) => b,
            _ => 0,
        })
        .collect();
    let b: Vec<u16> = (0..n)
        .map(|i| match Literal::from_bf16_f64((i as f64) * 2e-3 + 3.0) {
            Literal::BF16Bits(x) => x,
            _ => 0,
        })
        .collect();
    c.bench_function("eval/bf16_abs_64k_simd", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::bf16_neg_abs_bench(&a, true, true))
    });
    c.bench_function("eval/bf16_abs_64k_scalar", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::bf16_neg_abs_bench(&a, true, false))
    });
    c.bench_function("eval/bf16_mul_64k_simd", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::bf16_binary_bench(&a, &b, true))
    });
    c.bench_function("eval/bf16_mul_64k_scalar", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::bf16_binary_bench(&a, &b, false))
    });
    // f16 mul (normal-range inputs so the SIMD path, not the edge fallback, is exercised).
    let f16bits = |x: f64| -> u16 {
        match Literal::from_f16_f64(x) {
            Literal::F16Bits(b) => b,
            _ => 0,
        }
    };
    let af16: Vec<u16> = (0..n).map(|i| f16bits((i as f64) * 1e-3 + 1.0)).collect();
    let bf16v: Vec<u16> = (0..n).map(|i| f16bits((i as f64) * 7e-4 + 0.5)).collect();
    c.bench_function("eval/f16_mul_64k_simd", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::f16_binary_bench(&af16, &bf16v, true))
    });
    c.bench_function("eval/f16_mul_64k_scalar", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::f16_binary_bench(&af16, &bf16v, false))
    });
    // f16 relu: max(x, 0) — inputs span ± so ~half are clamped.
    let af16r: Vec<u16> = (0..n).map(|i| f16bits((i as f64) * 1e-3 - 30.0)).collect();
    c.bench_function("eval/f16_relu_64k_simd", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::f16_relu_bench(&af16r, true))
    });
    c.bench_function("eval/f16_relu_64k_scalar", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::f16_relu_bench(&af16r, false))
    });
    let s = a[3];
    c.bench_function("eval/bf16_scale_64k_simd", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::bf16_scalar_broadcast_bench(&a, s, true))
    });
    c.bench_function("eval/bf16_scale_64k_scalar", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::bf16_scalar_broadcast_bench(&a, s, false))
    });
    c.bench_function("eval/bf16_relu_64k_simd", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::bf16_relu_bench(&a, true))
    });
    c.bench_function("eval/bf16_relu_64k_scalar", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::bf16_relu_bench(&a, false))
    });
    // bias-add [N,C]+[C]: 512x128 = 64k, bias row reused across rows (the (1,1) case).
    let cols = 128usize;
    let bias: Vec<u16> = b[..cols].to_vec();
    c.bench_function("eval/bf16_bias_add_64k_simd", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::bf16_bias_add_bench(&a, &bias, cols, true))
    });
    c.bench_function("eval/bf16_bias_add_64k_scalar", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::bf16_bias_add_bench(&a, &bias, cols, false))
    });
}

// f16 reduce-sum A/B (same binary): SIMD IEEE-decode widen vs scalar per-element decode.
fn bench_f16_reduce_sum(c: &mut Criterion) {
    let n = LARGE_ELEMENTWISE_LEN;
    let f16bits = |x: f64| -> u16 {
        match Literal::from_f16_f64(x) {
            Literal::F16Bits(b) => b,
            _ => 0,
        }
    };
    let v: Vec<u16> = (0..n).map(|i| f16bits((i as f64) * 1e-3 + 1.0)).collect();
    c.bench_function("eval/f16_reduce_sum_64k_simd", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::f16_reduce_sum_bench(&v, true))
    });
    c.bench_function("eval/f16_reduce_sum_64k_scalar", |bencher| {
        bencher.iter(|| fj_lax::arithmetic::f16_reduce_sum_bench(&v, false))
    });
}

fn bench_reduce_sum_64k_bf16_dense(c: &mut Criterion) {
    let bits: Vec<u16> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| match Literal::from_bf16_f64((i as f64) * 1e-3) {
            Literal::BF16Bits(b) => b,
            _ => unreachable!(),
        })
        .collect();
    let input = Value::Tensor(
        TensorValue::new_half_float_values(
            DType::BF16,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            bits,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/reduce_sum_64k_bf16_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_64k_bf16_literal_reference(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| Literal::from_bf16_f64((i as f64) * 1e-3))
        .collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::BF16,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            elements,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/reduce_sum_64k_bf16_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

// BF16 ReduceMax (the dominant training dtype; max-norm grad clipping etc.): the
// dense path now widens u16->f32 and folds via a SIMD min/max reduce.
fn bench_reduce_max_64k_bf16_dense(c: &mut Criterion) {
    let bits: Vec<u16> = (0..LARGE_ELEMENTWISE_LEN)
        .map(
            |i| match Literal::from_bf16_f64(((i % 4099) as f64) * 1e-2 - 20.0) {
                Literal::BF16Bits(b) => b,
                _ => unreachable!(),
            },
        )
        .collect();
    let input = Value::Tensor(
        TensorValue::new_half_float_values(
            DType::BF16,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            bits,
        )
        .unwrap(),
    );
    let p = no_params();
    c.bench_function("eval/reduce_max_64k_bf16_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceMax, std::slice::from_ref(&input), &p))
    });
}

// BF16 `max(x, axis=-1)` over a [256,256] matrix (attention-score max in bf16): the
// inner==1 axis path folds each output cell via the bf16 SIMD min/max reduce.
fn bench_reduce_max_axis1_256_bf16(c: &mut Criterion) {
    let n = 256usize;
    let bits: Vec<u16> = (0..n * n)
        .map(
            |i| match Literal::from_bf16_f64(((i % 4099) as f64) * 1e-2 - 20.0) {
                Literal::BF16Bits(b) => b,
                _ => unreachable!(),
            },
        )
        .collect();
    let input = Value::Tensor(
        TensorValue::new_half_float_values(
            DType::BF16,
            Shape {
                dims: vec![n as u32, n as u32],
            },
            bits,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axes".to_owned(), "1".to_owned());
    c.bench_function("eval/reduce_max_axis1_256_bf16", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceMax, std::slice::from_ref(&input), &p))
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

// DIAGNOSTIC (CrimsonOtter 2026-06-21): tight direct-add f64 cumsum on the raw slice, no dispatch
// / no closure / no per-line chunk machinery. Isolates how much of eval/cumsum_4m_f64_1d's ~30ms
// is removable overhead vs the sequential dependency-chain floor (~3ms). If this is ~floor, the
// production scan path has ~10x overhead to reclaim (would beat JAX's 14.1ms).
fn bench_cumsum_4m_f64_1d_tight(c: &mut Criterion) {
    let data: Vec<f64> = (0..1 << 22).map(|i| (i as f64) * 0.001).collect();
    c.bench_function("eval/cumsum_4m_f64_1d_tight", |bencher| {
        bencher.iter(|| {
            let mut out = Vec::with_capacity(data.len());
            let mut acc = 0.0_f64;
            for &v in &data {
                acc += v;
                out.push(acc);
            }
            out
        })
    });
}

// CrimsonOtter 2026-06-22: cumprod/cummax at 4M-1D to verify the ledger's stale
// "cumprod/cummax = generic-serial LOSS" claim. Both route through the SAME generic
// `blocked_prefix_scan_to_vec` as cumsum_4m (op closure differs only), so they should be
// ~cumsum_4m, not ~20ms. Data kept near 1.0 so the cumprod stays finite.
fn bench_cumprod_4m_f64_1d(c: &mut Criterion) {
    let data: Vec<f64> = (0..1 << 22)
        .map(|i| 0.999_999_9 + ((i % 3) as f64) * 1e-7)
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("axis".to_owned(), "0".to_owned());
    c.bench_function("eval/cumprod_4m_f64_1d", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cumprod, std::slice::from_ref(&input), &p))
    });
}

fn bench_cummax_4m_f64_1d(c: &mut Criterion) {
    let data: Vec<f64> = (0_usize..(1_usize << 22))
        .map(|i| {
            let x = ((i.wrapping_mul(1_103_515_245_usize).wrapping_add(12_345)) & 0xffff) as f64;
            x - 32_768.0
        })
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("axis".to_owned(), "0".to_owned());
    c.bench_function("eval/cummax_4m_f64_1d", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cummax, std::slice::from_ref(&input), &p))
    });
}

fn bench_cummax_1m_f64_1d(c: &mut Criterion) {
    let data: Vec<f64> = (0_usize..(1_usize << 20))
        .map(|i| {
            let x = ((i.wrapping_mul(1_103_515_245_usize).wrapping_add(12_345)) & 0xffff) as f64;
            x - 32_768.0
        })
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("axis".to_owned(), "0".to_owned());
    c.bench_function("eval/cummax_1m_f64_1d", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cummax, std::slice::from_ref(&input), &p))
    });
}

fn bench_cummin_1m_f64_1d(c: &mut Criterion) {
    let data: Vec<f64> = (0_usize..(1_usize << 20))
        .map(|i| {
            let x = ((i.wrapping_mul(1_103_515_245_usize).wrapping_add(12_345)) & 0xffff) as f64;
            x - 32_768.0
        })
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("axis".to_owned(), "0".to_owned());
    c.bench_function("eval/cummin_1m_f64_1d", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cummin, std::slice::from_ref(&input), &p))
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

// 16384x1024 f32 Cumsum along the last axis (16384 independent lines): JAX's
// default float dtype. Each line is a sequential f64-accumulate scan rounded to
// f32; the lines are independent so the scan threads over the outer dimension
// bit-identically. Head-to-head: benchmarks/jax_comparison/cumsum_f32_axis1_gauntlet.py.
fn bench_cumsum_16k_x_1k_f32_axis1(c: &mut Criterion) {
    let (rows, cols) = (16_384usize, 1_024usize);
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 997) as f32 - 498.0) * 1e-3)
        .collect();
    let input = Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("axis".to_owned(), "1".to_owned());
    c.bench_function("eval/cumsum_16kx1k_f32_axis1", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Cumsum, std::slice::from_ref(&input), &p))
    });
}

// 16384x1024 f64 Cumsum along axis 0 (leading/strided): 1024 independent columns,
// each a 16384-deep scan read at stride 1024 by the current serial strided path.
// Measures the leading-axis cumulative gap vs JAX (head-to-head:
// benchmarks/jax_comparison/cumsum_axis0_gauntlet.py).
fn bench_cumsum_16k_x_1k_f64_axis0(c: &mut Criterion) {
    let (rows, cols) = (16_384usize, 1_024usize);
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i % 997) as f64 - 498.0) * 1e-3)
        .collect();
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
    p.insert("axis".to_owned(), "0".to_owned());
    c.bench_function("eval/cumsum_16kx1k_f64_axis0", |bencher| {
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

// Head-to-head vs JAX argsort (XLA CPU full bitonic sort): completes the order-statistics
// domination map alongside sort + top_k. CrimsonOtter 2026-06-21.
fn bench_argsort_64k_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| ((i as f64) * 1.000_173).sin() * 1e6 - (i as f64))
        .collect();
    let input = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("dimension".to_owned(), "0".to_owned());
    p.insert("descending".to_owned(), "false".to_owned());
    c.bench_function("eval/argsort_64k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Argsort, std::slice::from_ref(&input), &p))
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

// ConvertElementType downcast f32->bf16 over a 16M dense tensor (DRAM-bound):
// THE mixed-precision ML cast. Above CHEAP_BINARY_PARALLEL_MIN the dense path
// threads the per-element single-round convert into a calloc'd u16 output
// (parallel page-fault + read-bandwidth aggregation); bit-identical to serial.
const CONVERT_DRAM_LEN: usize = 16_777_216; // 1<<24, > CHEAP_BINARY_PARALLEL_MIN (1<<23)

fn bench_convert_16m_f32_to_bf16(c: &mut Criterion) {
    let data: Vec<f32> = (0..CONVERT_DRAM_LEN)
        .map(|i| ((i as f32) * 0.000_017_3).sin() + ((i % 97) as f32 * 0.125))
        .collect();
    let dense = Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![CONVERT_DRAM_LEN as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "bf16".to_owned());
    c.bench_function("eval/convert_16m_f32_to_bf16", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::ConvertElementType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
}

// ConvertElementType downcast f64->bf16 over a 16M dense tensor (DRAM-bound):
// the f64 sibling of the hot half cast (8-byte source read, u16 output).
fn bench_convert_16m_f64_to_bf16(c: &mut Criterion) {
    let data: Vec<f64> = (0..CONVERT_DRAM_LEN)
        .map(|i| ((i as f64) * 0.000_017_3).sin() + ((i % 97) as f64 * 0.125))
        .collect();
    let dense = Value::vector_f64(&data).unwrap();
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "bf16".to_owned());
    c.bench_function("eval/convert_16m_f64_to_bf16", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::ConvertElementType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
}

// Same-width bitcast over 1M dense f32 words: realistic RNG/mantissa workloads
// reinterpret contiguous XLA buffers. The literal reference keeps the old
// per-element Literal -> bytes -> Literal path in the same Criterion process.
fn bench_bitcast_f32_u32_dense_1m(c: &mut Criterion) {
    let data: Vec<f32> = (0..LARGE_RANDOM_LEN)
        .map(|i| ((i as f32) * 0.000_017_3).sin() + ((i % 97) as f32 * 0.125))
        .collect();
    let dense = Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![LARGE_RANDOM_LEN as u32],
            },
            data.clone(),
        )
        .unwrap(),
    );
    let literal = literal_backed_vector(
        DType::F32,
        data.iter().copied().map(Literal::from_f32).collect(),
    );
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "u32".to_owned());

    c.bench_function("eval/bitcast_f32_u32_dense_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
    c.bench_function("eval/bitcast_f32_u32_literal_ref_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&literal),
                &p,
            )
        })
    });
}

// Same-width bitcast over 1M dense f64 words: a wider version of the
// reinterpret path, covering i64-backed downstream bit manipulation.
fn bench_bitcast_f64_i64_dense_1m(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_RANDOM_LEN)
        .map(|i| ((i as f64) * 0.000_017_3).sin() + ((i % 97) as f64 * 0.125))
        .collect();
    let dense = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![LARGE_RANDOM_LEN as u32],
            },
            data.clone(),
        )
        .unwrap(),
    );
    let literal = literal_backed_vector(
        DType::F64,
        data.iter().copied().map(Literal::from_f64).collect(),
    );
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "i64".to_owned());

    c.bench_function("eval/bitcast_f64_i64_dense_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
    c.bench_function("eval/bitcast_f64_i64_literal_ref_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&literal),
                &p,
            )
        })
    });
}

// Same-width complement of f32->u32: signed integer output uses dense i32
// storage (`as_i64_slice`) but must preserve the identical 32 raw bits.
fn bench_bitcast_f32_i32_dense_1m(c: &mut Criterion) {
    let data: Vec<f32> = (0..LARGE_RANDOM_LEN)
        .map(|i| ((i as f32) * 0.000_017_3).sin() + ((i % 97) as f32 * 0.125))
        .collect();
    let dense = Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![LARGE_RANDOM_LEN as u32],
            },
            data.clone(),
        )
        .unwrap(),
    );
    let literal = literal_backed_vector(
        DType::F32,
        data.iter().copied().map(Literal::from_f32).collect(),
    );
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "i32".to_owned());

    c.bench_function("eval/bitcast_f32_i32_dense_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
    c.bench_function("eval/bitcast_f32_i32_literal_ref_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&literal),
                &p,
            )
        })
    });
}

// Same-width complement of u32->f32: signed i32 lanes are stored densely as i64
// values and reinterpreted back to f32 without per-Literal byte vectors.
fn bench_bitcast_i32_f32_dense_1m(c: &mut Criterion) {
    let data: Vec<i64> = (0..LARGE_RANDOM_LEN)
        .map(|i| {
            let value = ((i as f32) * 0.000_017_3).sin() + ((i % 97) as f32 * 0.125);
            i64::from(i32::from_le_bytes(value.to_bits().to_le_bytes()))
        })
        .collect();
    let dense = Value::Tensor(
        TensorValue::new_i32_values(
            Shape {
                dims: vec![LARGE_RANDOM_LEN as u32],
            },
            data.clone(),
        )
        .unwrap(),
    );
    let literal = literal_backed_vector(
        DType::I32,
        data.iter()
            .map(|&value| Literal::I32(i32::try_from(value).unwrap()))
            .collect(),
    );
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "f32".to_owned());

    c.bench_function("eval/bitcast_i32_f32_dense_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
    c.bench_function("eval/bitcast_i32_f32_literal_ref_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&literal),
                &p,
            )
        })
    });
}

// Same-width complement of f64->i64: unsigned integer output covers bit-level
// views used by hashing/serialization paths that need full 64-bit payloads.
fn bench_bitcast_f64_u64_dense_1m(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_RANDOM_LEN)
        .map(|i| ((i as f64) * 0.000_017_3).sin() + ((i % 97) as f64 * 0.125))
        .collect();
    let dense = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![LARGE_RANDOM_LEN as u32],
            },
            data.clone(),
        )
        .unwrap(),
    );
    let literal = literal_backed_vector(
        DType::F64,
        data.iter().copied().map(Literal::from_f64).collect(),
    );
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "u64".to_owned());

    c.bench_function("eval/bitcast_f64_u64_dense_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
    c.bench_function("eval/bitcast_f64_u64_literal_ref_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&literal),
                &p,
            )
        })
    });
}

// Same-width complement of i64->f64: packed u64 inputs are reinterpreted
// directly as f64, preserving NaN payloads and signed-zero bits.
fn bench_bitcast_u64_f64_dense_1m(c: &mut Criterion) {
    let data: Vec<u64> = (0..LARGE_RANDOM_LEN)
        .map(|i| {
            let value = ((i as f64) * 0.000_017_3).sin() + ((i % 97) as f64 * 0.125);
            value.to_bits()
        })
        .collect();
    let dense = Value::Tensor(
        TensorValue::new_u64_values(
            Shape {
                dims: vec![LARGE_RANDOM_LEN as u32],
            },
            data.clone(),
        )
        .unwrap(),
    );
    let literal =
        literal_backed_vector(DType::U64, data.iter().copied().map(Literal::U64).collect());
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "f64".to_owned());

    c.bench_function("eval/bitcast_u64_f64_dense_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
    c.bench_function("eval/bitcast_u64_f64_literal_ref_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&literal),
                &p,
            )
        })
    });
}

// Width-changing bitcast over 1M dense f64 words: splits each contiguous f64
// into two little-endian u32 chunks. This catches serialization/RNG-style
// reinterpret pipelines that used to materialize every Literal.
fn bench_bitcast_f64_u32_dense_1m(c: &mut Criterion) {
    let data: Vec<f64> = (0..LARGE_RANDOM_LEN)
        .map(|i| ((i as f64) * 0.000_017_3).sin() + ((i % 97) as f64 * 0.125))
        .collect();
    let dense = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![LARGE_RANDOM_LEN as u32],
            },
            data.clone(),
        )
        .unwrap(),
    );
    let literal = literal_backed_vector(
        DType::F64,
        data.iter().copied().map(Literal::from_f64).collect(),
    );
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "u32".to_owned());

    c.bench_function("eval/bitcast_f64_u32_dense_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
    c.bench_function("eval/bitcast_f64_u32_literal_ref_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&literal),
                &p,
            )
        })
    });
}

// Width-changing bitcast over 1M f64-worth of dense u32 chunks: consumes the
// trailing chunk dimension without per-element Literal byte grouping.
fn bench_bitcast_u32_f64_dense_1m(c: &mut Criterion) {
    let chunks: Vec<u32> = (0..LARGE_RANDOM_LEN)
        .map(|i| ((i as f64) * 0.000_017_3).sin() + ((i % 97) as f64 * 0.125))
        .flat_map(f64_to_u32_bitcast_chunks)
        .collect();
    let shape = Shape {
        dims: vec![LARGE_RANDOM_LEN as u32, 2],
    };
    let dense = Value::Tensor(TensorValue::new_u32_values(shape.clone(), chunks.clone()).unwrap());
    let literal = literal_backed_tensor(
        DType::U32,
        shape,
        chunks.iter().copied().map(Literal::U32).collect(),
    );
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "f64".to_owned());

    c.bench_function("eval/bitcast_u32_f64_dense_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
    c.bench_function("eval/bitcast_u32_f64_literal_ref_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&literal),
                &p,
            )
        })
    });
}

// Width-changing bitcast over 1M dense f32 words: splits each contiguous f32
// into two little-endian BF16 chunks without numeric conversion. This covers
// mixed-precision serialization/reinterpret pipelines that previously walked
// every element as a boxed Literal.
fn bench_bitcast_f32_bf16_dense_1m(c: &mut Criterion) {
    let data: Vec<f32> = (0..LARGE_RANDOM_LEN)
        .map(|i| ((i as f32) * 0.000_017_3).sin() + ((i % 97) as f32 * 0.125))
        .collect();
    let dense = Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![LARGE_RANDOM_LEN as u32],
            },
            data.clone(),
        )
        .unwrap(),
    );
    let literal = literal_backed_vector(
        DType::F32,
        data.iter().copied().map(Literal::from_f32).collect(),
    );
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "bf16".to_owned());

    c.bench_function("eval/bitcast_f32_bf16_dense_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
    c.bench_function("eval/bitcast_f32_bf16_literal_ref_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&literal),
                &p,
            )
        })
    });
}

// Width-changing bitcast over 1M f32-worth of dense BF16 chunks: consumes the
// trailing chunk dimension and rebuilds f32 words from packed u16 lanes.
fn bench_bitcast_bf16_f32_dense_1m(c: &mut Criterion) {
    let chunks: Vec<u16> = (0..LARGE_RANDOM_LEN)
        .map(|i| ((i as f32) * 0.000_017_3).sin() + ((i % 97) as f32 * 0.125))
        .flat_map(f32_to_u16_bitcast_chunks)
        .collect();
    let shape = Shape {
        dims: vec![LARGE_RANDOM_LEN as u32, 2],
    };
    let dense = Value::Tensor(
        TensorValue::new_half_float_values(DType::BF16, shape.clone(), chunks.clone()).unwrap(),
    );
    let literal = literal_backed_tensor(
        DType::BF16,
        shape,
        chunks.iter().copied().map(Literal::BF16Bits).collect(),
    );
    let mut p = BTreeMap::new();
    p.insert("new_dtype".to_owned(), "f32".to_owned());

    c.bench_function("eval/bitcast_bf16_f32_dense_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&dense),
                &p,
            )
        })
    });
    c.bench_function("eval/bitcast_bf16_f32_literal_ref_1m", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BitcastConvertType,
                std::slice::from_ref(&literal),
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

// Scalar BroadcastInDim over dense-capable non-float dtypes. These rows pin the
// direct typed-fill path for U32/U64/Complex instead of a Vec<Literal> fill.
fn bench_broadcast_scalar_dense_fill(c: &mut Criterion) {
    let mut p = BTreeMap::new();
    p.insert("shape".to_owned(), "1024,1024".to_owned());

    let u32_input = Value::Scalar(Literal::U32(0xfeed_beef));
    c.bench_function("eval/broadcast_scalar_u32_1024x1024", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BroadcastInDim,
                std::slice::from_ref(&u32_input),
                &p,
            )
        })
    });

    let u64_input = Value::Scalar(Literal::U64(0x0123_4567_89ab_cdef));
    c.bench_function("eval/broadcast_scalar_u64_1024x1024", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BroadcastInDim,
                std::slice::from_ref(&u64_input),
                &p,
            )
        })
    });

    let complex_input = Value::Scalar(Literal::Complex128Bits(
        1.25_f64.to_bits(),
        (-0.5_f64).to_bits(),
    ));
    c.bench_function("eval/broadcast_scalar_complex128_1024x1024", |bencher| {
        bencher.iter(|| {
            eval_primitive(
                Primitive::BroadcastInDim,
                std::slice::from_ref(&complex_input),
                &p,
            )
        })
    });
}

// Scalar Tile is another uniform-fill path: a rank-0 scalar tiled to a large
// tensor should allocate dense typed storage directly rather than a Vec<Literal>.
fn bench_tile_scalar_dense_fill(c: &mut Criterion) {
    let mut p = BTreeMap::new();
    p.insert("reps".to_owned(), "1024,1024".to_owned());

    let f32_input = Value::Scalar(Literal::from_f32(1.25));
    c.bench_function("eval/tile_scalar_f32_1024x1024", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Tile, std::slice::from_ref(&f32_input), &p))
    });

    let complex_input = Value::Scalar(Literal::Complex128Bits(
        1.25_f64.to_bits(),
        (-0.5_f64).to_bits(),
    ));
    c.bench_function("eval/tile_scalar_complex128_1024x1024", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Tile, std::slice::from_ref(&complex_input), &p))
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

fn sort_u32_data() -> Vec<u32> {
    (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| match i % 257 {
            0 => u32::MAX,
            1 => 0,
            2 => 1 << 31,
            3 | 4 => 19_999,
            _ => {
                (i as u32)
                    .wrapping_mul(2_654_435_761)
                    .rotate_left((i & 31) as u32)
                    ^ ((i as u32) >> 3)
            }
        })
        .collect()
}

// U32 sort: dense radix path over typed storage vs the Literal-backed radix
// reference, which must materialize boxed literals before sorting.
fn bench_sort_64k_u32(c: &mut Criterion) {
    let input = Value::Tensor(
        TensorValue::new_u32_values(Shape::vector(LARGE_ELEMENTWISE_LEN as u32), sort_u32_data())
            .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("dimension".to_owned(), "0".to_owned());
    p.insert("descending".to_owned(), "false".to_owned());
    c.bench_function("eval/sort_64k_u32", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &p))
    });
}

fn bench_sort_64k_u32_literal_reference(c: &mut Criterion) {
    let data = sort_u32_data();
    let input = Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::U32,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            fj_core::LiteralBuffer::new(data.iter().copied().map(Literal::U32).collect()),
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("dimension".to_owned(), "0".to_owned());
    p.insert("descending".to_owned(), "false".to_owned());
    c.bench_function("eval/sort_64k_u32_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &p))
    });
}

fn sort_u64_data() -> Vec<u64> {
    (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| match i % 263 {
            0 => u64::MAX,
            1 => 0,
            2 => 1 << 63,
            3 | 4 => 9_223_372_036_854_775_900,
            _ => {
                (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .rotate_left((i & 63) as u32)
                    ^ ((i as u64) << 32)
            }
        })
        .collect()
}

fn bench_sort_64k_u64(c: &mut Criterion) {
    let input = Value::Tensor(
        TensorValue::new_u64_values(Shape::vector(LARGE_ELEMENTWISE_LEN as u32), sort_u64_data())
            .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("dimension".to_owned(), "0".to_owned());
    p.insert("descending".to_owned(), "false".to_owned());
    c.bench_function("eval/sort_64k_u64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Sort, std::slice::from_ref(&input), &p))
    });
}

fn bench_sort_64k_u64_literal_reference(c: &mut Criterion) {
    let data = sort_u64_data();
    let input = Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::U64,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            fj_core::LiteralBuffer::new(data.iter().copied().map(Literal::U64).collect()),
        )
        .unwrap(),
    );
    let mut p = BTreeMap::new();
    p.insert("dimension".to_owned(), "0".to_owned());
    p.insert("descending".to_owned(), "false".to_owned());
    c.bench_function("eval/sort_64k_u64_literal_ref", |bencher| {
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

fn topk_u32_data() -> Vec<u32> {
    (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| match i % 257 {
            0 => u32::MAX,
            1 => 0,
            2 => 1 << 31,
            3 | 4 => 19_999,
            _ => {
                (i as u32)
                    .wrapping_mul(2_654_435_761)
                    .rotate_left((i & 31) as u32)
                    ^ ((i as u32) >> 3)
            }
        })
        .collect()
}

fn bench_topk_64k_k128_u32_vec(c: &mut Criterion) {
    let input = Value::Tensor(
        TensorValue::new_u32_values(Shape::vector(LARGE_ELEMENTWISE_LEN as u32), topk_u32_data())
            .unwrap(),
    );
    let p = topk_params();
    c.bench_function("eval/topk_64k_k128_u32_vec", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::TopK, std::slice::from_ref(&input), &p))
    });
}

fn bench_topk_64k_k128_u32_literal_reference(c: &mut Criterion) {
    let data = topk_u32_data();
    let input = Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::U32,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            fj_core::LiteralBuffer::new(data.iter().copied().map(Literal::U32).collect()),
        )
        .unwrap(),
    );
    let p = topk_params();
    c.bench_function("eval/topk_64k_k128_u32_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::TopK, std::slice::from_ref(&input), &p))
    });
}

fn topk_u64_data() -> Vec<u64> {
    (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| match i % 263 {
            0 => u64::MAX,
            1 => 0,
            2 => 1 << 63,
            3 | 4 => 9_223_372_036_854_775_900,
            _ => {
                (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .rotate_left((i & 63) as u32)
                    ^ ((i as u64) << 32)
            }
        })
        .collect()
}

fn bench_topk_64k_k128_u64_vec(c: &mut Criterion) {
    let input = Value::Tensor(
        TensorValue::new_u64_values(Shape::vector(LARGE_ELEMENTWISE_LEN as u32), topk_u64_data())
            .unwrap(),
    );
    let p = topk_params();
    c.bench_function("eval/topk_64k_k128_u64_vec", |bencher| {
        bencher.iter(|| eval_primitive_multi(Primitive::TopK, std::slice::from_ref(&input), &p))
    });
}

fn bench_topk_64k_k128_u64_literal_reference(c: &mut Criterion) {
    let data = topk_u64_data();
    let input = Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::U64,
            Shape::vector(LARGE_ELEMENTWISE_LEN as u32),
            fj_core::LiteralBuffer::new(data.iter().copied().map(Literal::U64).collect()),
        )
        .unwrap(),
    );
    let p = topk_params();
    c.bench_function("eval/topk_64k_k128_u64_literal_ref", |bencher| {
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

// `jnp.max(x, axis=-1)` over a [256,256] f64 matrix (65536 elems — the common
// single-threaded softmax/attention-stability regime, below the 2^18 thread gate):
// the inner==1 contiguous-block path now folds each output cell via a SIMD min/max
// reduce instead of the scalar jax_max fold.
fn bench_reduce_max_axis1_256_f64(c: &mut Criterion) {
    let n = 256usize;
    let data: Vec<f64> = (0..n * n)
        .map(|i| ((i % 4099) as f64) * 0.013 - 26.0)
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
    let mut p = BTreeMap::new();
    p.insert("axes".to_owned(), "1".to_owned());
    c.bench_function("eval/reduce_max_axis1_256_f64", |bencher| {
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

// 256x256 Complex128 ReduceSum along axis 1: dense (re,im) contiguous-block fold
// (bead frankenjax-wobjv) vs the Vec<Literal> per-element odometer reference.
// Same process for a same-worker ratio isolating the block fold vs the odometer.
fn bench_reduce_sum_256_axis1_complex_vec(c: &mut Criterion) {
    let input = complex_matrix_dense(256, 256);
    let p = axis_params("1");
    c.bench_function("eval/reduce_sum_256_axis1_complex_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p))
    });
}

fn bench_reduce_sum_256_axis1_complex_literal_reference(c: &mut Criterion) {
    let input = complex_matrix(256, 256);
    let p = axis_params("1");
    c.bench_function("eval/reduce_sum_256_axis1_complex_literal_ref", |bencher| {
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

// Large-window maxpool [256,256] 15x15 VALID: the separable max path (O(out·Σwin)) vs a
// direct full-window O(out·Πwin) max (the pre-separable dense algorithm), isolating the
// separability win (both dense f64, no per-Literal overhead).
fn maxpool_large_data() -> (Vec<f64>, usize, usize) {
    let n = 256usize;
    let data: Vec<f64> = (0..n * n)
        .map(|i| ((i as f64) * 0.123).sin() * 100.0)
        .collect();
    (data, n, 15)
}

fn maxpool_direct_ref(x: &[f64], n: usize, win: usize) -> Vec<f64> {
    let o = n - win + 1;
    let mut out = vec![f64::NEG_INFINITY; o * o];
    for oi in 0..o {
        for oj in 0..o {
            let mut acc = f64::NEG_INFINITY;
            for a in 0..win {
                for b in 0..win {
                    let v = x[(oi + a) * n + (oj + b)];
                    if v > acc {
                        acc = v;
                    }
                }
            }
            out[oi * o + oj] = acc;
        }
    }
    out
}

fn bench_maxpool_large_direct(c: &mut Criterion) {
    let (data, n, win) = maxpool_large_data();
    c.bench_function("eval/maxpool_256x256_15x15_direct", |b| {
        b.iter(|| maxpool_direct_ref(black_box(&data), n, win))
    });
}

fn bench_maxpool_large_separable(c: &mut Criterion) {
    let (data, n, win) = maxpool_large_data();
    let input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut p = no_params();
    p.insert("reduce_op".to_owned(), "max".to_owned());
    p.insert("window_dimensions".to_owned(), format!("{win},{win}"));
    p.insert("window_strides".to_owned(), "1,1".to_owned());
    p.insert("padding".to_owned(), "VALID".to_owned());
    c.bench_function("eval/maxpool_256x256_15x15_separable", |b| {
        b.iter(|| eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&input), &p))
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

fn select_64k_bool_inputs(dense: bool) -> [Value; 3] {
    let dims = vec![LARGE_ELEMENTWISE_LEN as u32];
    let cond: Vec<bool> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| (i.wrapping_mul(17).wrapping_add(3)) % 11 < 5)
        .collect();
    let t: Vec<bool> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| (i ^ (i >> 2)) & 1 == 0)
        .collect();
    let f: Vec<bool> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| i.is_multiple_of(3))
        .collect();
    let mk = |values: Vec<bool>| {
        if dense {
            Value::Tensor(
                TensorValue::new_bool_values(Shape { dims: dims.clone() }, values).unwrap(),
            )
        } else {
            Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    DType::Bool,
                    Shape { dims: dims.clone() },
                    fj_core::LiteralBuffer::new(values.into_iter().map(Literal::Bool).collect()),
                )
                .unwrap(),
            )
        }
    };
    [mk(cond), mk(t), mk(f)]
}

fn bench_select_64k_bool_vec(c: &mut Criterion) {
    let inputs = select_64k_bool_inputs(true);
    let p = no_params();
    c.bench_function("eval/select_64k_bool_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Select, &inputs, &p))
    });
}

fn bench_select_64k_bool_literal_reference(c: &mut Criterion) {
    let inputs = select_64k_bool_inputs(false);
    let p = no_params();
    c.bench_function("eval/select_64k_bool_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Select, &inputs, &p))
    });
}

fn select_n_64k_u32_inputs(dense: bool) -> [Value; 4] {
    let dims = vec![LARGE_ELEMENTWISE_LEN as u32];
    let idxv: Vec<i64> = (0..LARGE_ELEMENTWISE_LEN as i64).map(|i| i % 3).collect();
    let idx = if dense {
        Value::Tensor(TensorValue::new_i64_values(Shape { dims: dims.clone() }, idxv).unwrap())
    } else {
        Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: dims.clone() },
                idxv.into_iter().map(Literal::I64).collect(),
            )
            .unwrap(),
        )
    };
    let mk = |seed: u32| {
        let data: Vec<u32> = (0..LARGE_ELEMENTWISE_LEN)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed))
            .collect();
        if dense {
            Value::Tensor(TensorValue::new_u32_values(Shape { dims: dims.clone() }, data).unwrap())
        } else {
            Value::Tensor(
                TensorValue::new(
                    DType::U32,
                    Shape { dims: dims.clone() },
                    data.into_iter().map(Literal::U32).collect(),
                )
                .unwrap(),
            )
        }
    };
    [idx, mk(0), mk(11), mk(29)]
}

fn select_n_64k_u32_bool_index_inputs(dense: bool) -> [Value; 3] {
    let dims = vec![LARGE_ELEMENTWISE_LEN as u32];
    let shape = Shape { dims: dims.clone() };
    let mut words = vec![0u64; LARGE_ELEMENTWISE_LEN.div_ceil(64)];
    let bools: Vec<bool> = (0..LARGE_ELEMENTWISE_LEN)
        .map(|i| {
            let flag = (i.wrapping_mul(17).wrapping_add(11)) % 19 < 9;
            if flag {
                words[i / 64] |= 1_u64 << (i % 64);
            }
            flag
        })
        .collect();
    let idx = if dense {
        Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::Bool,
                shape,
                fj_core::LiteralBuffer::from_bool_words(words, LARGE_ELEMENTWISE_LEN).unwrap(),
            )
            .unwrap(),
        )
    } else {
        Value::Tensor(
            TensorValue::new(
                DType::Bool,
                shape,
                bools.into_iter().map(Literal::Bool).collect(),
            )
            .unwrap(),
        )
    };
    let mk = |seed: u32| {
        let data: Vec<u32> = (0..LARGE_ELEMENTWISE_LEN)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed))
            .collect();
        if dense {
            Value::Tensor(TensorValue::new_u32_values(Shape { dims: dims.clone() }, data).unwrap())
        } else {
            Value::Tensor(
                TensorValue::new(
                    DType::U32,
                    Shape { dims: dims.clone() },
                    data.into_iter().map(Literal::U32).collect(),
                )
                .unwrap(),
            )
        }
    };
    [idx, mk(0), mk(11)]
}

fn bench_select_n_64k_u32_vec(c: &mut Criterion) {
    let inputs = select_n_64k_u32_inputs(true);
    let p = no_params();
    c.bench_function("eval/select_n_64k_u32_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::SelectN, &inputs, &p))
    });
}

fn bench_select_n_64k_u32_literal_reference(c: &mut Criterion) {
    let inputs = select_n_64k_u32_inputs(false);
    let p = no_params();
    c.bench_function("eval/select_n_64k_u32_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::SelectN, &inputs, &p))
    });
}

fn bench_select_n_64k_u32_boolwords_index_vec(c: &mut Criterion) {
    let inputs = select_n_64k_u32_bool_index_inputs(true);
    let p = no_params();
    c.bench_function("eval/select_n_64k_u32_boolwords_index_vec", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::SelectN, &inputs, &p))
    });
}

fn bench_select_n_64k_u32_bool_index_literal_reference(c: &mut Criterion) {
    let inputs = select_n_64k_u32_bool_index_inputs(false);
    let p = no_params();
    c.bench_function("eval/select_n_64k_u32_bool_index_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::SelectN, &inputs, &p))
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

fn bench_complex_tensor_scalar_dense_fill(c: &mut Criterion) {
    let p = no_params();
    let n = 1usize << 20;

    let f32_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 500.0).collect();
    let f32_real =
        Value::Tensor(TensorValue::new_f32_values(Shape::vector(n as u32), f32_data).unwrap());
    let f32_inputs = [f32_real, Value::Scalar(Literal::from_f32(0.0))];
    c.bench_function("eval/complex_f32_tensor_scalar_1m", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Complex, &f32_inputs, &p))
    });

    let f64_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 500.0).collect();
    let f64_real =
        Value::Tensor(TensorValue::new_f64_values(Shape::vector(n as u32), f64_data).unwrap());
    let f64_inputs = [f64_real, Value::Scalar(Literal::from_f64(0.0))];
    c.bench_function("eval/complex_f64_tensor_scalar_1m", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Complex, &f64_inputs, &p))
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

#[allow(clippy::too_many_arguments)]
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

// i64 canonical [512,512]@[512,512] matmul: dense (contiguous rank2_i64_matmul
// fast path) vs Vec<Literal> boxed (the generic strided per-element reduction).
// Same-invocation A/B — the two run in one process so the ratio is trustworthy.
fn i64_matmul_params() -> BTreeMap<String, String> {
    let mut p = no_params();
    p.insert("lhs_contracting_dims".to_owned(), "1".to_owned());
    p.insert("rhs_contracting_dims".to_owned(), "0".to_owned());
    p
}

// Canonical batched matmul params: [batch,m,k]@[batch,k,n] (batch dim 0, contract 2/1).
fn batched_matmul_params() -> BTreeMap<String, String> {
    let mut p = no_params();
    p.insert("lhs_batch_dims".to_owned(), "0".to_owned());
    p.insert("rhs_batch_dims".to_owned(), "0".to_owned());
    p.insert("lhs_contracting_dims".to_owned(), "2".to_owned());
    p.insert("rhs_contracting_dims".to_owned(), "1".to_owned());
    p
}

// Batched i64 [64,64,64]@[64,64,64]: dense (contiguous batched_rank2_i64_matmul)
// vs Vec<Literal> boxed (the generic strided per-element loop batched i64 fell to
// before). Same-invocation A/B.
fn bench_batched_i64_matmul_dense(c: &mut Criterion) {
    let (bt, n) = (64usize, 64usize);
    let a: Vec<i64> = (0..(bt * n * n) as i64)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();
    let b: Vec<i64> = (0..(bt * n * n) as i64)
        .map(|i| i.wrapping_mul(40_503) ^ 0x5555)
        .collect();
    let d = vec![bt as u32, n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new_i64_values(Shape { dims: d.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new_i64_values(Shape { dims: d }, b).unwrap());
    let p = batched_matmul_params();
    c.bench_function("eval/matmul_batched64_64x64_i64_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_batched_i64_matmul_literal_reference(c: &mut Criterion) {
    let (bt, n) = (64usize, 64usize);
    let a: Vec<Literal> = (0..(bt * n * n) as i64)
        .map(|i| Literal::I64(i.wrapping_mul(2_654_435_761)))
        .collect();
    let b: Vec<Literal> = (0..(bt * n * n) as i64)
        .map(|i| Literal::I64(i.wrapping_mul(40_503) ^ 0x5555))
        .collect();
    let d = vec![bt as u32, n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new(DType::I64, Shape { dims: d.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new(DType::I64, Shape { dims: d }, b).unwrap());
    let p = batched_matmul_params();
    c.bench_function("eval/matmul_batched64_64x64_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Batched Complex128 [64,48,48]@[64,48,48]: dense (contiguous
// batched_rank2_complex_matmul) vs Vec<Literal> boxed. Same-invocation A/B.
fn bench_batched_complex_matmul_dense(c: &mut Criterion) {
    let (bt, n) = (64usize, 48usize);
    let vals: Vec<(f64, f64)> = (0..(bt * n * n) as i64)
        .map(|i| (i as f64 * 0.5 - 3.0, i as f64 * -0.25 + 1.0))
        .collect();
    let d = vec![bt as u32, n as u32, n as u32];
    let lhs = Value::Tensor(
        TensorValue::new_complex_values(DType::Complex128, Shape { dims: d.clone() }, vals.clone())
            .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_complex_values(DType::Complex128, Shape { dims: d }, vals).unwrap(),
    );
    let p = batched_matmul_params();
    c.bench_function("eval/matmul_batched64_48x48_complex128_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_batched_complex_matmul_literal_reference(c: &mut Criterion) {
    let (bt, n) = (64usize, 48usize);
    let elems: Vec<Literal> = (0..(bt * n * n) as i64)
        .map(|i| Literal::from_complex128(i as f64 * 0.5 - 3.0, i as f64 * -0.25 + 1.0))
        .collect();
    let d = vec![bt as u32, n as u32, n as u32];
    let lhs = Value::Tensor(
        TensorValue::new(DType::Complex128, Shape { dims: d.clone() }, elems.clone()).unwrap(),
    );
    let rhs = Value::Tensor(TensorValue::new(DType::Complex128, Shape { dims: d }, elems).unwrap());
    let p = batched_matmul_params();
    c.bench_function(
        "eval/matmul_batched64_48x48_complex128_literal_ref",
        |bencher| {
            bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
        },
    );
}

fn bench_i64_matmul_512_dense(c: &mut Criterion) {
    let n = 512usize;
    let a: Vec<i64> = (0..(n * n) as i64)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();
    let b: Vec<i64> = (0..(n * n) as i64)
        .map(|i| i.wrapping_mul(40_503) ^ 0x5555)
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new_i64_values(Shape { dims: dims.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new_i64_values(Shape { dims }, b).unwrap());
    let p = i64_matmul_params();
    c.bench_function("eval/matmul_512x512_i64_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_i64_matmul_512_literal_reference(c: &mut Criterion) {
    let n = 512usize;
    let a: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::I64(i.wrapping_mul(2_654_435_761)))
        .collect();
    let b: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::I64(i.wrapping_mul(40_503) ^ 0x5555))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new(DType::I64, Shape { dims: dims.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new(DType::I64, Shape { dims }, b).unwrap());
    let p = i64_matmul_params();
    c.bench_function("eval/matmul_512x512_i64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Canonical i32 [512,512]@[512,512] matmul: dense (the i32 branch of the
// contiguous rank2_i64_matmul fast path, tagged I32) vs Vec<Literal> boxed (the
// generic strided per-element reduction i32 matmul fell to before). i32 reuses
// the i64 kernel because i32 tensors are dense-i64-backed; the output stays I32
// and the narrow chokepoint wraps. Same-invocation A/B.
fn bench_i32_matmul_512_dense(c: &mut Criterion) {
    let n = 512usize;
    // Keep values in i32 range (cast through i32) so they are valid int32 inputs.
    let a: Vec<i64> = (0..(n * n) as i64)
        .map(|i| i64::from(i.wrapping_mul(2_654_435_761) as i32))
        .collect();
    let b: Vec<i64> = (0..(n * n) as i64)
        .map(|i| i64::from((i.wrapping_mul(40_503) ^ 0x5555) as i32))
        .collect();
    let dims = vec![n as u32, n as u32];
    let mut la = TensorValue::new_i64_values(Shape { dims: dims.clone() }, a).unwrap();
    la.dtype = DType::I32;
    let mut rb = TensorValue::new_i64_values(Shape { dims }, b).unwrap();
    rb.dtype = DType::I32;
    let lhs = Value::Tensor(la);
    let rhs = Value::Tensor(rb);
    let p = i64_matmul_params();
    c.bench_function("eval/matmul_512x512_i32_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_i32_matmul_512_literal_reference(c: &mut Criterion) {
    let n = 512usize;
    let a: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::I64(i64::from(i.wrapping_mul(2_654_435_761) as i32)))
        .collect();
    let b: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::I64(i64::from((i.wrapping_mul(40_503) ^ 0x5555) as i32)))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new(DType::I32, Shape { dims: dims.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new(DType::I32, Shape { dims }, b).unwrap());
    let p = i64_matmul_params();
    c.bench_function("eval/matmul_512x512_i32_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Canonical u32 [512,512]@[512,512] matmul: dense fast path (extract to Vec<u64> +
// contiguous rank2_u64_matmul) vs the generic strided per-element loop the SAME data
// hits in the transposed (A·Bᵀ) orientation (which the canonical-only fast path
// misses). Same-invocation A/B isolating fast-kernel vs generic-decode for u32.
fn bench_u32_matmul_512_canonical_fast(c: &mut Criterion) {
    let n = 512usize;
    let a: Vec<Literal> = (0..(n * n) as u64)
        .map(|i| Literal::U32((i.wrapping_mul(2_654_435_761) % 70_000) as u32))
        .collect();
    let b: Vec<Literal> = (0..(n * n) as u64)
        .map(|i| Literal::U32((i.wrapping_mul(40_503) % 70_000) as u32))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new(DType::U32, Shape { dims: dims.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new(DType::U32, Shape { dims }, b).unwrap());
    let mut p = no_params();
    p.insert("lhs_contracting_dims".to_owned(), "1".to_owned());
    p.insert("rhs_contracting_dims".to_owned(), "0".to_owned());
    c.bench_function("eval/matmul_512x512_u32_canonical_fast", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Transposed u32 A·Bᵀ — now a FAST path (rank2_u64_any_orientation_matmul: transpose
// to canonical + rank2_u64_matmul). Compared against the batched-generic reference
// below (batched u32 is still the generic strided loop), same 512^3 FLOPs.
fn bench_u32_matmul_512_transposed_fast(c: &mut Criterion) {
    let n = 512usize;
    let a: Vec<Literal> = (0..(n * n) as u64)
        .map(|i| Literal::U32((i.wrapping_mul(2_654_435_761) % 70_000) as u32))
        .collect();
    let b: Vec<Literal> = (0..(n * n) as u64)
        .map(|i| Literal::U32((i.wrapping_mul(40_503) % 70_000) as u32))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new(DType::U32, Shape { dims: dims.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new(DType::U32, Shape { dims }, b).unwrap());
    let mut p = no_params();
    p.insert("lhs_contracting_dims".to_owned(), "1".to_owned());
    p.insert("rhs_contracting_dims".to_owned(), "1".to_owned());
    c.bench_function("eval/matmul_512x512_u32_transposed_fast", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Batched u32 [1,512,512]@[1,512,512] — batched u32 is NOT in a fast path, so this is
// the generic strided per-element loop (512^3 FLOPs, directly comparable to the
// canonical/transposed fast benches above). Same-invocation A/B reference.
fn bench_u32_matmul_512_batched_generic(c: &mut Criterion) {
    let n = 512usize;
    let a: Vec<Literal> = (0..(n * n) as u64)
        .map(|i| Literal::U32((i.wrapping_mul(2_654_435_761) % 70_000) as u32))
        .collect();
    let b: Vec<Literal> = (0..(n * n) as u64)
        .map(|i| Literal::U32((i.wrapping_mul(40_503) % 70_000) as u32))
        .collect();
    let dims = vec![1u32, n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new(DType::U32, Shape { dims: dims.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new(DType::U32, Shape { dims }, b).unwrap());
    let mut p = no_params();
    p.insert("lhs_batch_dims".to_owned(), "0".to_owned());
    p.insert("rhs_batch_dims".to_owned(), "0".to_owned());
    p.insert("lhs_contracting_dims".to_owned(), "2".to_owned());
    p.insert("rhs_contracting_dims".to_owned(), "1".to_owned());
    c.bench_function("eval/matmul_512x512_u32_batched_generic", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Transposed i64 [512,512]·[512,512]ᵀ matmul (rhs_contracting=1, i.e. A·Bᵀ):
// dense (contiguous rank2_i64_any_orientation_matmul: transpose B then
// rank2_i64_matmul) vs Vec<Literal> boxed (the generic strided per-element
// reduction that this orientation fell to before). Same-invocation A/B.
fn i64_matmul_transposed_params() -> BTreeMap<String, String> {
    let mut p = no_params();
    p.insert("lhs_contracting_dims".to_owned(), "1".to_owned());
    p.insert("rhs_contracting_dims".to_owned(), "1".to_owned());
    p
}

fn bench_i64_matmul_512_transposed_dense(c: &mut Criterion) {
    let n = 512usize;
    let a: Vec<i64> = (0..(n * n) as i64)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();
    let b: Vec<i64> = (0..(n * n) as i64)
        .map(|i| i.wrapping_mul(40_503) ^ 0x5555)
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new_i64_values(Shape { dims: dims.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new_i64_values(Shape { dims }, b).unwrap());
    let p = i64_matmul_transposed_params();
    c.bench_function("eval/matmul_512x512_i64_transposed_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_i64_matmul_512_transposed_literal_reference(c: &mut Criterion) {
    let n = 512usize;
    let a: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::I64(i.wrapping_mul(2_654_435_761)))
        .collect();
    let b: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::I64(i.wrapping_mul(40_503) ^ 0x5555))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new(DType::I64, Shape { dims: dims.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new(DType::I64, Shape { dims }, b).unwrap());
    let p = i64_matmul_transposed_params();
    c.bench_function(
        "eval/matmul_512x512_i64_transposed_literal_ref",
        |bencher| {
            bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
        },
    );
}

// Transposed i32 [512,512]·[512,512]ᵀ matmul: dense (i32 branch of
// rank2_i64_any_orientation_matmul, re-tagged I32) vs Vec<Literal> boxed generic.
// Same-invocation A/B.
fn bench_i32_matmul_512_transposed_dense(c: &mut Criterion) {
    let n = 512usize;
    let a: Vec<i64> = (0..(n * n) as i64)
        .map(|i| i64::from(i.wrapping_mul(2_654_435_761) as i32))
        .collect();
    let b: Vec<i64> = (0..(n * n) as i64)
        .map(|i| i64::from((i.wrapping_mul(40_503) ^ 0x5555) as i32))
        .collect();
    let dims = vec![n as u32, n as u32];
    let mut la = TensorValue::new_i64_values(Shape { dims: dims.clone() }, a).unwrap();
    la.dtype = DType::I32;
    let mut rb = TensorValue::new_i64_values(Shape { dims }, b).unwrap();
    rb.dtype = DType::I32;
    let lhs = Value::Tensor(la);
    let rhs = Value::Tensor(rb);
    let p = i64_matmul_transposed_params();
    c.bench_function("eval/matmul_512x512_i32_transposed_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_i32_matmul_512_transposed_literal_reference(c: &mut Criterion) {
    let n = 512usize;
    let a: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::I64(i64::from(i.wrapping_mul(2_654_435_761) as i32)))
        .collect();
    let b: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::I64(i64::from((i.wrapping_mul(40_503) ^ 0x5555) as i32)))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new(DType::I32, Shape { dims: dims.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new(DType::I32, Shape { dims }, b).unwrap());
    let p = i64_matmul_transposed_params();
    c.bench_function(
        "eval/matmul_512x512_i32_transposed_literal_ref",
        |bencher| {
            bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
        },
    );
}

// Batched i32 [64,64,64]@[64,64,64]: dense (i32 branch of batched_rank2_i64_matmul,
// re-tagged I32) vs Vec<Literal> boxed generic. Same-invocation A/B.
fn bench_batched_i32_matmul_dense(c: &mut Criterion) {
    let (bt, n) = (64usize, 64usize);
    let a: Vec<i64> = (0..(bt * n * n) as i64)
        .map(|i| i64::from(i.wrapping_mul(2_654_435_761) as i32))
        .collect();
    let b: Vec<i64> = (0..(bt * n * n) as i64)
        .map(|i| i64::from((i.wrapping_mul(40_503) ^ 0x5555) as i32))
        .collect();
    let d = vec![bt as u32, n as u32, n as u32];
    let mut la = TensorValue::new_i64_values(Shape { dims: d.clone() }, a).unwrap();
    la.dtype = DType::I32;
    let mut rb = TensorValue::new_i64_values(Shape { dims: d }, b).unwrap();
    rb.dtype = DType::I32;
    let lhs = Value::Tensor(la);
    let rhs = Value::Tensor(rb);
    let p = batched_matmul_params();
    c.bench_function("eval/matmul_batched64_64x64_i32_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_batched_i32_matmul_literal_reference(c: &mut Criterion) {
    let (bt, n) = (64usize, 64usize);
    let a: Vec<Literal> = (0..(bt * n * n) as i64)
        .map(|i| Literal::I64(i64::from(i.wrapping_mul(2_654_435_761) as i32)))
        .collect();
    let b: Vec<Literal> = (0..(bt * n * n) as i64)
        .map(|i| Literal::I64(i64::from((i.wrapping_mul(40_503) ^ 0x5555) as i32)))
        .collect();
    let d = vec![bt as u32, n as u32, n as u32];
    let lhs = Value::Tensor(TensorValue::new(DType::I32, Shape { dims: d.clone() }, a).unwrap());
    let rhs = Value::Tensor(TensorValue::new(DType::I32, Shape { dims: d }, b).unwrap());
    let p = batched_matmul_params();
    c.bench_function("eval/matmul_batched64_64x64_i32_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Complex128 canonical [256,256]@[256,256] matmul: dense (contiguous
// rank2_complex_matmul fast path) vs Vec<Literal> boxed (the generic strided
// per-element complex reduction). Same-invocation A/B.
fn bench_complex_matmul_256_dense(c: &mut Criterion) {
    let n = 256usize;
    let vals: Vec<(f64, f64)> = (0..(n * n) as i64)
        .map(|i| (i as f64 * 0.5 - 3.0, i as f64 * -0.25 + 1.0))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(
        TensorValue::new_complex_values(
            DType::Complex128,
            Shape { dims: dims.clone() },
            vals.clone(),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_complex_values(DType::Complex128, Shape { dims }, vals).unwrap(),
    );
    let p = i64_matmul_params();
    c.bench_function("eval/matmul_256x256_complex128_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_complex_matmul_256_literal_reference(c: &mut Criterion) {
    let n = 256usize;
    let elems: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::from_complex128(i as f64 * 0.5 - 3.0, i as f64 * -0.25 + 1.0))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: dims.clone() },
            elems.clone(),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(TensorValue::new(DType::Complex128, Shape { dims }, elems).unwrap());
    let p = i64_matmul_params();
    c.bench_function("eval/matmul_256x256_complex128_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Complex64 (JAX's DEFAULT complex dtype) canonical [256,256]@[256,256] matmul: dense
// (now routed through the contiguous rank2_complex_matmul kernel + round-to-f32 output)
// vs Vec<Literal> boxed Complex64Bits (the generic strided per-element complex reduction
// it fell to before). Same-invocation A/B.
fn bench_complex64_matmul_256_dense(c: &mut Criterion) {
    let n = 256usize;
    let vals: Vec<(f64, f64)> = (0..(n * n) as i64)
        .map(|i| (i as f64 * 0.5 - 3.0, i as f64 * -0.25 + 1.0))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(
        TensorValue::new_complex_values(
            DType::Complex64,
            Shape { dims: dims.clone() },
            vals.clone(),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_complex_values(DType::Complex64, Shape { dims }, vals).unwrap(),
    );
    let p = i64_matmul_params();
    c.bench_function("eval/matmul_256x256_complex64_dense", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

fn bench_complex64_matmul_256_literal_reference(c: &mut Criterion) {
    let n = 256usize;
    let elems: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| {
            Literal::from_complex64(
                (i as f64 * 0.5 - 3.0) as f32,
                (i as f64 * -0.25 + 1.0) as f32,
            )
        })
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: dims.clone() },
            elems.clone(),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(TensorValue::new(DType::Complex64, Shape { dims }, elems).unwrap());
    let p = i64_matmul_params();
    c.bench_function("eval/matmul_256x256_complex64_literal_ref", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
    });
}

// Transposed Complex128 [256,256]·[256,256]ᵀ matmul (rhs_contracting=1, A·Bᵀ):
// dense (rank2_complex_any_orientation_matmul: transpose B then
// rank2_complex_matmul) vs Vec<Literal> boxed (the generic strided per-element
// complex reduction this orientation fell to before). Same-invocation A/B.
fn bench_complex_matmul_256_transposed_dense(c: &mut Criterion) {
    let n = 256usize;
    let vals: Vec<(f64, f64)> = (0..(n * n) as i64)
        .map(|i| (i as f64 * 0.5 - 3.0, i as f64 * -0.25 + 1.0))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(
        TensorValue::new_complex_values(
            DType::Complex128,
            Shape { dims: dims.clone() },
            vals.clone(),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(
        TensorValue::new_complex_values(DType::Complex128, Shape { dims }, vals).unwrap(),
    );
    let mut p = no_params();
    p.insert("lhs_contracting_dims".to_owned(), "1".to_owned());
    p.insert("rhs_contracting_dims".to_owned(), "1".to_owned());
    c.bench_function(
        "eval/matmul_256x256_complex128_transposed_dense",
        |bencher| {
            bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
        },
    );
}

fn bench_complex_matmul_256_transposed_literal_reference(c: &mut Criterion) {
    let n = 256usize;
    let elems: Vec<Literal> = (0..(n * n) as i64)
        .map(|i| Literal::from_complex128(i as f64 * 0.5 - 3.0, i as f64 * -0.25 + 1.0))
        .collect();
    let dims = vec![n as u32, n as u32];
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: dims.clone() },
            elems.clone(),
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(TensorValue::new(DType::Complex128, Shape { dims }, elems).unwrap());
    let mut p = no_params();
    p.insert("lhs_contracting_dims".to_owned(), "1".to_owned());
    p.insert("rhs_contracting_dims".to_owned(), "1".to_owned());
    c.bench_function(
        "eval/matmul_256x256_complex128_transposed_literal_ref",
        |bencher| {
            bencher.iter(|| eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &p))
        },
    );
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

// Batched prime-length FFT with dense-complex input. 1009 forces the Bluestein
// path; 256 rows crosses the threaded full-eval threshold and isolates the
// batched rough-length kernel family missing from the existing FFT bench set.
fn bench_fft_batch_256x1009_prime_dense_input(c: &mut Criterion) {
    let input = complex_matrix_dense(256, 1009);
    let p = no_params();
    c.bench_function(
        "eval/fft_batch_256x1009_prime_complex128_dense_input",
        |bencher| bencher.iter(|| eval_primitive(Primitive::Fft, std::slice::from_ref(&input), &p)),
    );
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

// Attention-score einsum with the head axis in the middle. This exercises the
// general einsum permute-copy path (`bqhd,bkhd->bhqk`), including the
// trailing-suffix memcpy in `permute_copy_f64`, before batched GEMM dominates.
fn bench_einsum2_general_attention_f64(c: &mut Criterion) {
    let (bsz, q, h, d, kk) = (4usize, 64usize, 8usize, 64usize, 64usize);
    let a_shape = [bsz, q, h, d];
    let b_shape = [bsz, kk, h, d];
    let a: Vec<f64> = (0..bsz * q * h * d)
        .map(|i| (i as f64 * 0.0011).sin())
        .collect();
    let b: Vec<f64> = (0..bsz * kk * h * d)
        .map(|i| (i as f64 * 0.0009).cos())
        .collect();

    c.bench_function("eval/einsum2_general_bqhd_bkhd_bhqk_f64", |bencher| {
        bencher.iter(|| {
            fj_lax::einsum::einsum2(
                "bqhd,bkhd->bhqk",
                black_box(&a),
                black_box(&a_shape),
                black_box(&b),
                black_box(&b_shape),
            )
        })
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

// Multi-output split over a dense f32 matrix. This pins the lazy concat-slice
// path so each output section can preserve dense backing without Vec<Literal>.
fn bench_split_multi_1024x1024_f32(c: &mut Criterion) {
    let rows = 1024usize;
    let cols = 1024usize;
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.001 - 100.0).collect();
    let input = Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            data,
        )
        .unwrap(),
    );
    let mut params = BTreeMap::new();
    params.insert("axis".into(), "1".into());
    params.insert("sizes".into(), "256,768".into());
    c.bench_function("eval/split_multi_1024x1024_f32_axis1", |bencher| {
        bencher
            .iter(|| eval_primitive_multi(Primitive::Split, std::slice::from_ref(&input), &params))
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
// CrimsonOtter 2026-06-22: scattered single-element gather (1M indices from a 4M f64
// operand, slice_sizes=1) — the ledger's documented ~18x JAX LOSS (Rust 34.6ms vs JAX
// 1.91ms). Pseudo-random indices so every read is a non-contiguous random access.
fn bench_gather_scatter_1m_f64(c: &mut Criterion) {
    let n = 1usize << 22; // 4M operand
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
    let operand = Value::Tensor(
        TensorValue::new_f64_values(Shape { dims: vec![n as u32] }, data).unwrap(),
    );
    let k = 1usize << 20; // 1M gathers
    let idx: Vec<i64> = (0..k)
        .map(|i| ((i.wrapping_mul(2_654_435_761) ^ (i >> 3)) % n) as i64)
        .collect();
    let indices = Value::vector_i64(&idx).unwrap();
    let mut p = BTreeMap::new();
    p.insert("slice_sizes".into(), "1".into());
    // Clone the 32MB operand ONCE (outside iter) so the measurement isolates the gather,
    // not a per-iteration 40MB memcpy. eval_primitive borrows its inputs.
    let args = [operand, indices];
    c.bench_function("eval/gather_scatter_1m_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Gather, &args, &p))
    });
}

// CrimsonOtter 2026-06-22: i32 scattered single-element gather (id/index lookup). The i32
// path previously had NO threaded/branchless fast path (serial dense_contiguous_gather only),
// so the branchless threaded gather_single_dense adds BOTH threading and MLP here.
fn bench_gather_scatter_1m_i32(c: &mut Criterion) {
    let n = 1usize << 22;
    let data: Vec<i64> = (0..n as i64).map(|i| i % 1000).collect();
    let operand = Value::Tensor(
        TensorValue::new_i32_values(Shape { dims: vec![n as u32] }, data).unwrap(),
    );
    let k = 1usize << 20;
    let idx: Vec<i64> = (0..k)
        .map(|i| ((i.wrapping_mul(2_654_435_761) ^ (i >> 3)) % n) as i64)
        .collect();
    let indices = Value::vector_i64(&idx).unwrap();
    let mut p = BTreeMap::new();
    p.insert("slice_sizes".into(), "1".into());
    let args = [operand, indices];
    c.bench_function("eval/gather_scatter_1m_i32", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Gather, &args, &p))
    });
}

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

fn bench_scatter_add_1m_f64_1d(c: &mut Criterion) {
    let n = 1_usize << 20;
    let operand = Value::vector_f64(&vec![0.0; n]).unwrap();
    let indices_data: Vec<i64> = (0..n)
        .map(|i| ((i.wrapping_mul(1_103_515_245_usize).wrapping_add(12_345)) & (n - 1)) as i64)
        .collect();
    let indices = Value::vector_i64(&indices_data).unwrap();
    let updates_data: Vec<f64> = (0..n)
        .map(|i| ((i % 4099) as f64 - 2049.0) * 0.001)
        .collect();
    let updates = Value::vector_f64(&updates_data).unwrap();
    let inputs = [operand, indices, updates];
    let mut p = BTreeMap::new();
    p.insert("mode".into(), "add".into());
    p.insert("index_mode".into(), "clip".into());
    c.bench_function("eval/scatter_add_1m_f64_1d", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Scatter, &inputs, &p))
    });
}

// CrimsonOtter 2026-06-22: scattered single-element OVERWRITE scatter (1M random writes into
// a 4M f64 operand). The general scatter_typed loop pays a per-element copy_from_slice(len 1)
// CALL + overflow/bounds checks; the branchless `out[idx]=upd[i]` fast path lets the store
// buffer overlap the random writes (the scatter dual of the gather MLP win).
fn bench_scatter_overwrite_1m_f64(c: &mut Criterion) {
    let n = 1_usize << 22; // 4M operand (DRAM-bound random scatter)
    let operand = Value::vector_f64(&vec![0.0; n]).unwrap();
    let k = 1_usize << 20; // 1M overwrites
    let indices_data: Vec<i64> = (0..k)
        .map(|i| ((i.wrapping_mul(2_654_435_761) ^ (i >> 3)) % n) as i64)
        .collect();
    let indices = Value::vector_i64(&indices_data).unwrap();
    let updates_data: Vec<f64> = (0..k).map(|i| (i % 4099) as f64 * 0.001).collect();
    let updates = Value::vector_f64(&updates_data).unwrap();
    let inputs = [operand, indices, updates];
    let mut p = BTreeMap::new();
    p.insert("mode".into(), "overwrite".into());
    p.insert("index_mode".into(), "clip".into());
    c.bench_function("eval/scatter_overwrite_1m_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Scatter, &inputs, &p))
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

// WY-blocked QR (bead wpjbg): same-binary A/B of the scalar reflector loop vs the
// WY-blocked GEMM-trailing-update path at large n, where the trailing matrix spills L3
// repeatedly under the BLAS-2 rank-1 update. `qr_real_bench` forces the path so this
// isolates the algorithm from the auto-gate. n=1024 probes the production gate;
// small sample count because each n=4096 QR is ~1s.
fn bench_qr_blocked_ab(c: &mut Criterion) {
    for &n in &[1024usize, 2048usize, 4096usize] {
        let a: Vec<f64> = (0..n * n)
            .map(|i| {
                let x = i as f64;
                (x * 0.125).sin() + (x * 0.03125).cos()
            })
            .collect();
        c.bench_function(&format!("linalg/qr_{n}_scalar"), |bencher| {
            bencher.iter(|| {
                black_box(fj_lax::linalg::qr_real_bench(
                    black_box(a.clone()),
                    n,
                    n,
                    false,
                ))
            })
        });
        c.bench_function(&format!("linalg/qr_{n}_blocked"), |bencher| {
            bencher.iter(|| {
                black_box(fj_lax::linalg::qr_real_bench(
                    black_box(a.clone()),
                    n,
                    n,
                    true,
                ))
            })
        });
    }
}

// Fused 2D softmax/log_softmax (no per-row Vec allocation): same-binary A/B.
// The "rowmap_ref" arm is the prior implementation (map the 1D helper over rows,
// allocating 2 Vecs + a copy per row); the "fused" arm writes each row in place.
// Many-rows/small-cols is the allocation-dominated batched regime where removing
// the per-row heap traffic wins. Both produce bit-identical output.
fn bench_softmax_2d_fused_ab(c: &mut Criterion) {
    let rows = 65_536usize;
    let cols = 16usize;
    let x: Vec<f64> = (0..rows * cols)
        .map(|k| ((k as f64) * 0.013).sin() * 4.0)
        .collect();

    c.bench_function("nn/softmax_2d_65536x16_rowmap_ref", |bencher| {
        bencher.iter(|| {
            let mut result = vec![0.0; rows * cols];
            for i in 0..rows {
                let row = &x[i * cols..(i + 1) * cols];
                let sm = fj_lax::nn::softmax(black_box(row));
                result[i * cols..(i + 1) * cols].copy_from_slice(&sm);
            }
            black_box(result)
        })
    });

    c.bench_function("nn/softmax_2d_65536x16_fused", |bencher| {
        bencher.iter(|| black_box(fj_lax::nn::softmax_2d(black_box(&x), rows, cols)))
    });

    c.bench_function("nn/log_softmax_2d_65536x16_rowmap_ref", |bencher| {
        bencher.iter(|| {
            let mut result = vec![0.0; rows * cols];
            for i in 0..rows {
                let row = &x[i * cols..(i + 1) * cols];
                let lsm = fj_lax::nn::log_softmax(black_box(row));
                result[i * cols..(i + 1) * cols].copy_from_slice(&lsm);
            }
            black_box(result)
        })
    });

    c.bench_function("nn/log_softmax_2d_65536x16_fused", |bencher| {
        bencher.iter(|| black_box(fj_lax::nn::log_softmax_2d(black_box(&x), rows, cols)))
    });
}

criterion_group!(
    name = qr_blocked_ab;
    config = Criterion::default().sample_size(10);
    targets = bench_qr_blocked_ab
);

criterion_group!(softmax_2d_fused_ab, bench_softmax_2d_fused_ab);

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
    bench_tanh_1m_f64_vec,
    bench_tan_1m_f64_vec,
    bench_sqrt_1m_f64_vec,
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
    bench_relu_64k_f32_vec,
    bench_relu_64k_f32_literal_reference,
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
    // bench_svd_48_f64_jacobi_counters, [cc-temp]
    bench_svd_48_complex_path,
    bench_svd_48_full_f64,
    bench_svd_48_full_complex_path,
    bench_matmul_2d_256,
    bench_bf16_matmul_512_f32accum,
    bench_bf16_matmul_512_f64accum_reference,
    bench_bf16_matmul_1024_blocked,
    bench_bf16_matmul_1024_rowref,
    bench_f16_matmul_512_f32accum,
    bench_f16_matmul_512_promote_reference,
    bench_matmul_2d_512,
    bench_f32_gemm_1024,
    bench_f32_gemm_2048,
    bench_f32_gemm_4096,
    bench_strassen_ab_1024,
    bench_strassen_ab_2048,
    bench_conv2d_gemm_f32_native,
    bench_conv2d_gemm_f64_promote_reference,
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
    bench_concat_axis0_2x512x1024_then_add_f64,
    bench_transpose_256x256_f64,
    bench_transpose_512x512_complex128_dense,
    bench_transpose_512x512_complex128_literal_reference,
    bench_reduce_sum_1k,
    bench_reduce_sum_4096x1024_axis1_f64,
    bench_reduce_sum_4096x1024_axis0_f64,
    bench_reduce_sum_64k_f64,
    bench_reduce_sum_64k_f64_literal_reference,
    bench_reduce_max_64k_f64,
    bench_reduce_max_64k_f64_literal_reference,
    bench_reduce_max_axis1_256_f64,
    bench_reduce_sum_256_axis1_f64,
    bench_reduce_sum_256_axis1_f64_literal_reference,
    bench_reduce_sum_256_axis0_f64,
    bench_reduce_sum_256_axis0_f64_literal_reference,
    bench_reduce_sum_256_axis1_i64,
    bench_reduce_sum_256_axis1_i64_literal_reference,
    bench_reduce_sum_256_axis0_i64,
    bench_reduce_sum_256_axis0_i64_literal_reference,
    bench_reduce_sum_256_axis1_complex_vec,
    bench_reduce_sum_256_axis1_complex_literal_reference,
    bench_reduce_sum_64k_i64,
    bench_reduce_sum_4096x128_axis1_f64,
    bench_reduce_sum_16m_f64_full,
    bench_reduce_sum_16m_i64_full,
    bench_reduce_sum_16k_x_1k_axis0_f64,
    bench_reduce_sum_16k_x_1k_i64_axes,
    bench_argmax_16k_x_1k_axis1_f64,
    bench_argmax_16k_x_1k_axis0_f64,
    bench_argmax_16k_x_1k_axis1_i64,
    bench_argmax_16k_x_1k_axis0_i64,
    bench_cumsum_16k_x_1k_f32_axis1,
    bench_cumsum_16k_x_1k_f64_axis0,
    bench_reduce_sum_64k_i64_literal_reference,
    bench_reduce_sum_64k_f32_dense,
    bench_reduce_sum_64k_f32_literal_reference,
    bench_bf16_elementwise_mul,
    bench_f16_reduce_sum,
    bench_reduce_sum_64k_bf16_dense,
    bench_reduce_max_64k_bf16_dense,
    bench_reduce_max_axis1_256_bf16,
    bench_reduce_sum_64k_bf16_literal_reference,
    bench_reduce_and_64k_bool_vec,
    bench_reduce_and_64k_bool_literal_reference,
    bench_reduce_and_256_axis1_bool_vec,
    bench_reduce_and_256_axis1_bool_literal_reference,
    bench_cumsum_64k_f64_vec,
    bench_cumsum_4m_f64_1d,
    bench_cumsum_4m_f64_1d_tight,
    bench_cumprod_4m_f64_1d,
    bench_cummax_4m_f64_1d,
    bench_cummax_1m_f64_1d,
    bench_cummin_1m_f64_1d,
    bench_cumsum_4096x1024_f64_axis1,
    bench_cumsum_64k_f64_literal_reference,
    bench_sort_64k_i64,
    bench_sort_64k_f64,
    bench_argsort_64k_f64,
    bench_argmax_64k_f64,
    bench_sort_64k_f64_descending,
    bench_convert_64k_f64_to_i64,
    bench_convert_16m_f32_to_bf16,
    bench_convert_16m_f64_to_bf16,
    bench_bitcast_f32_u32_dense_1m,
    bench_bitcast_f64_i64_dense_1m,
    bench_bitcast_f32_i32_dense_1m,
    bench_bitcast_i32_f32_dense_1m,
    bench_bitcast_f64_u64_dense_1m,
    bench_bitcast_u64_f64_dense_1m,
    bench_bitcast_f64_u32_dense_1m,
    bench_bitcast_u32_f64_dense_1m,
    bench_bitcast_f32_bf16_dense_1m,
    bench_bitcast_bf16_f32_dense_1m,
    bench_broadcast_256_to_256x256_f64,
    bench_broadcast_scalar_dense_fill,
    bench_tile_scalar_dense_fill,
    bench_pad_256x256_to_258x258_f64,
    bench_rev_256x256_f64,
    bench_one_hot_2048x512_f64,
    bench_broadcasted_iota_512x512_i64,
    bench_sort_64k_f32,
    bench_sort_64k_u32,
    bench_sort_64k_u32_literal_reference,
    bench_sort_64k_u64,
    bench_sort_64k_u64_literal_reference,
    bench_sort_calib_reduce_64k_i64,
    bench_topk_64k_k128_f64_vec,
    bench_topk_64k_k128_f32,
    bench_topk_64k_k128_f64_literal_reference,
    bench_topk_64k_k128_i64_vec,
    bench_topk_64k_k128_i64_literal_reference,
    bench_topk_64k_k128_u32_vec,
    bench_topk_64k_k128_u32_literal_reference,
    bench_topk_64k_k128_u64_vec,
    bench_topk_64k_k128_u64_literal_reference,
    bench_reduce_window_64x64,
    bench_maxpool_256x256_f64_vec,
    bench_maxpool_256x256_f64_literal_reference,
    bench_maxpool_large_direct,
    bench_maxpool_large_separable,
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
    bench_select_64k_bool_vec,
    bench_select_64k_bool_literal_reference,
    bench_select_n_64k_u32_vec,
    bench_select_n_64k_u32_literal_reference,
    bench_select_n_64k_u32_boolwords_index_vec,
    bench_select_n_64k_u32_bool_index_literal_reference,
    bench_complex_mul_1k,
    bench_complex_mul_1m_literal,
    bench_complex_mul_1m_dense,
    bench_complex_ctor_1k,
    bench_complex_tensor_scalar_dense_fill,
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
    bench_i64_matmul_512_dense,
    bench_i64_matmul_512_literal_reference,
    bench_i64_matmul_512_transposed_dense,
    bench_i64_matmul_512_transposed_literal_reference,
    bench_batched_i64_matmul_dense,
    bench_u32_matmul_512_canonical_fast,
    bench_u32_matmul_512_transposed_fast,
    bench_u32_matmul_512_batched_generic,
    bench_i32_matmul_512_dense,
    bench_i32_matmul_512_literal_reference,
    bench_i32_matmul_512_transposed_dense,
    bench_i32_matmul_512_transposed_literal_reference,
    bench_batched_i32_matmul_dense,
    bench_batched_i32_matmul_literal_reference,
    bench_batched_i64_matmul_literal_reference,
    bench_batched_complex_matmul_dense,
    bench_batched_complex_matmul_literal_reference,
    bench_complex_matmul_256_dense,
    bench_complex_matmul_256_literal_reference,
    bench_complex64_matmul_256_dense,
    bench_complex64_matmul_256_literal_reference,
    bench_complex_matmul_256_transposed_dense,
    bench_complex_matmul_256_transposed_literal_reference,
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
    bench_fft_batch_256x1009_prime_dense_input,
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
    bench_einsum2_general_attention_f64,
    bench_reshape,
    bench_split_multi_1024x1024_f32,
    bench_gather_128_rows_16_cols,
    bench_gather_scatter_1m_f64,
    bench_gather_scatter_1m_i32,
    bench_gather_256x256_f64_vec,
    bench_gather_256x256_f64_literal_reference,
    bench_scatter_128_rows_16_cols,
    bench_scatter_add_1m_f64_1d,
    bench_scatter_overwrite_1m_f64,
    bench_scatter_256x256_f64_vec,
    bench_scatter_256x256_f64_literal_reference,
    bench_slice_64_rows_16_cols,
    bench_dynamic_slice_64_rows_16_cols,
    bench_dynamic_update_slice_64_rows_16_cols,
    bench_eq_1k,
    bench_bitwise_and_1k,
);
criterion_main!(benches, qr_blocked_ab, softmax_2d_fused_ab);
