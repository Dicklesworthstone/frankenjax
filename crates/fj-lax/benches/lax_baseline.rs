use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

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
        elements,
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
        elements,
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
        elements,
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

fn bench_dot_100(c: &mut Criterion) {
    let data: Vec<i64> = (0..100).collect();
    let lhs = Value::vector_i64(&data).unwrap();
    let rhs = Value::vector_i64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/dot_100_i64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Dot, &[lhs.clone(), rhs.clone()], &p))
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

fn bench_sin_1k(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/sin_1k_f64", |bencher| {
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

fn bench_square_1k(c: &mut Criterion) {
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let input = Value::vector_f64(&data).unwrap();
    let p = no_params();
    c.bench_function("eval/square_1k_f64", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Square, std::slice::from_ref(&input), &p))
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
        elements: cond_elements,
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

fn bench_rfft_256(c: &mut Criterion) {
    let input = real_vector(256);
    let p = no_params();
    c.bench_function("eval/rfft_256_f64", |bencher| {
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
    bench_dot_100,
    bench_reduce_sum_1k,
    bench_reduce_window_64x64,
    bench_sin_1k,
    bench_exp_1k,
    bench_square_1k,
    bench_clamp_1k,
    bench_select_1k,
    bench_complex_mul_1k,
    bench_complex_ctor_1k,
    bench_complex_conj_1k,
    bench_complex_real_1k,
    bench_complex_imag_1k,
    bench_fft_256,
    bench_ifft_256,
    bench_rfft_256,
    bench_irfft_256,
    bench_reshape,
    bench_gather_128_rows_16_cols,
    bench_scatter_128_rows_16_cols,
    bench_slice_64_rows_16_cols,
    bench_dynamic_slice_64_rows_16_cols,
    bench_dynamic_update_slice_64_rows_16_cols,
    bench_eq_1k,
);
criterion_main!(benches);
