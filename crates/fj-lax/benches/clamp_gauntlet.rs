use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fj_core::{DType, Literal, LiteralBuffer, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

const N: usize = 1_048_576;

fn vector_shape(len: usize) -> Shape {
    Shape::vector(u32::try_from(len).expect("bench vector length fits u32"))
}

fn half_offset(i: usize) -> u16 {
    u16::try_from(i % 64).expect("bench half offset fits u16")
}

fn f32_dense(values: &[f32]) -> Value {
    Value::Tensor(TensorValue::new_f32_values(vector_shape(values.len()), values.to_vec()).unwrap())
}

fn f32_boxed(values: &[f32]) -> Value {
    Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::F32,
            vector_shape(values.len()),
            LiteralBuffer::new(values.iter().map(|&v| Literal::from_f32(v)).collect()),
        )
        .unwrap(),
    )
}

fn f64_dense(values: &[f64]) -> Value {
    Value::Tensor(TensorValue::new_f64_values(vector_shape(values.len()), values.to_vec()).unwrap())
}

fn f64_boxed(values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::F64,
            vector_shape(values.len()),
            LiteralBuffer::new(values.iter().map(|&v| Literal::from_f64(v)).collect()),
        )
        .unwrap(),
    )
}

fn i64_dense(values: &[i64]) -> Value {
    Value::Tensor(TensorValue::new_i64_values(vector_shape(values.len()), values.to_vec()).unwrap())
}

fn i64_boxed(values: &[i64]) -> Value {
    Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::I64,
            vector_shape(values.len()),
            LiteralBuffer::new(values.iter().map(|&v| Literal::I64(v)).collect()),
        )
        .unwrap(),
    )
}

fn half_literal(dtype: DType, bits: u16) -> Literal {
    if dtype == DType::BF16 {
        Literal::BF16Bits(bits)
    } else {
        Literal::F16Bits(bits)
    }
}

fn half_dense(dtype: DType, values: &[u16]) -> Value {
    Value::Tensor(
        TensorValue::new_half_float_values(dtype, vector_shape(values.len()), values.to_vec())
            .unwrap(),
    )
}

fn half_boxed(dtype: DType, values: &[u16]) -> Value {
    Value::Tensor(
        TensorValue::new_with_literal_buffer(
            dtype,
            vector_shape(values.len()),
            LiteralBuffer::new(values.iter().map(|&v| half_literal(dtype, v)).collect()),
        )
        .unwrap(),
    )
}

fn bench_inputs(label: &str, dense: [Value; 3], boxed: [Value; 3], c: &mut Criterion) {
    let params = BTreeMap::new();
    let mut group = c.benchmark_group(label);
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function(BenchmarkId::new("frankenjax_dense", N), |b| {
        b.iter(|| {
            let out =
                eval_primitive(Primitive::Clamp, black_box(&dense), black_box(&params)).unwrap();
            black_box(out);
        });
    });
    group.bench_function(BenchmarkId::new("frankenjax_boxed_reference", N), |b| {
        b.iter(|| {
            let out =
                eval_primitive(Primitive::Clamp, black_box(&boxed), black_box(&params)).unwrap();
            black_box(out);
        });
    });
    group.finish();
}

fn bench_f32_f64_mixed_scalar_tensor(c: &mut Criterion) {
    let x32: Vec<f32> = (0..N).map(|i| (i as f32) * 0.001 - 500.0).collect();
    let hi32: Vec<f32> = (0..N).map(|i| 2.0 + (i % 8192) as f32 * 0.0005).collect();
    bench_inputs(
        "f32_mixed_scalar_tensor_1m",
        [
            Value::Scalar(Literal::from_f32(0.0)),
            f32_dense(&x32),
            f32_dense(&hi32),
        ],
        [
            Value::Scalar(Literal::from_f32(0.0)),
            f32_boxed(&x32),
            f32_boxed(&hi32),
        ],
        c,
    );

    let x64: Vec<f64> = (0..N).map(|i| (i as f64) * 0.001 - 500.0).collect();
    let hi64: Vec<f64> = (0..N).map(|i| 2.0 + (i % 8192) as f64 * 0.0005).collect();
    bench_inputs(
        "f64_mixed_scalar_tensor_1m",
        [
            Value::Scalar(Literal::from_f64(0.0)),
            f64_dense(&x64),
            f64_dense(&hi64),
        ],
        [
            Value::Scalar(Literal::from_f64(0.0)),
            f64_boxed(&x64),
            f64_boxed(&hi64),
        ],
        c,
    );
}

fn bench_half_mixed_scalar_tensor(c: &mut Criterion) {
    for (dtype, label, one_bits, two_bits) in [
        (DType::BF16, "bf16_mixed_scalar_tensor_1m", 0x3f80, 0x4000),
        (DType::F16, "f16_mixed_scalar_tensor_1m", 0x3c00, 0x4000),
    ] {
        let x: Vec<u16> = (0..N).map(|i| one_bits + half_offset(i)).collect();
        let hi: Vec<u16> = (0..N).map(|i| two_bits + half_offset(i)).collect();
        bench_inputs(
            label,
            [
                Value::Scalar(half_literal(dtype, 0x0000)),
                half_dense(dtype, &x),
                half_dense(dtype, &hi),
            ],
            [
                Value::Scalar(half_literal(dtype, 0x0000)),
                half_boxed(dtype, &x),
                half_boxed(dtype, &hi),
            ],
            c,
        );
    }
}

fn bench_i64_tensor_tensor(c: &mut Criterion) {
    let lo: Vec<i64> = (0..N).map(|i| (i as i64 % 97) - 50).collect();
    let x: Vec<i64> = (0..N)
        .map(|i| (i as i64).wrapping_mul(2_654_435_761) % 10_000)
        .collect();
    let hi: Vec<i64> = lo.iter().map(|&v| v + 128).collect();
    bench_inputs(
        "i64_tensor_tensor_tensor_1m",
        [i64_dense(&lo), i64_dense(&x), i64_dense(&hi)],
        [i64_boxed(&lo), i64_boxed(&x), i64_boxed(&hi)],
        c,
    );
}

fn bench_i64_mixed_scalar_tensor(c: &mut Criterion) {
    let x: Vec<i64> = (0..N)
        .map(|i| (i as i64).wrapping_mul(1_103_515_245) % 100_000)
        .collect();
    let hi: Vec<i64> = (0..N).map(|i| 50 + (i as i64 % 211)).collect();
    bench_inputs(
        "i64_mixed_scalar_lo_tensor_hi_1m",
        [
            Value::Scalar(Literal::I64(0)),
            i64_dense(&x),
            i64_dense(&hi),
        ],
        [
            Value::Scalar(Literal::I64(0)),
            i64_boxed(&x),
            i64_boxed(&hi),
        ],
        c,
    );

    let lo: Vec<i64> = (0..N).map(|i| (i as i64 % 197) - 96).collect();
    let x_hi: Vec<i64> = (0..N)
        .map(|i| (i as i64).wrapping_mul(1_664_525) % 100_000)
        .collect();
    bench_inputs(
        "i64_mixed_tensor_lo_scalar_hi_1m",
        [
            i64_dense(&lo),
            i64_dense(&x_hi),
            Value::Scalar(Literal::I64(211)),
        ],
        [
            i64_boxed(&lo),
            i64_boxed(&x_hi),
            Value::Scalar(Literal::I64(211)),
        ],
        c,
    );
}

fn bench_half_tensor_tensor(c: &mut Criterion) {
    for (dtype, label, lo_bits, x_bits, hi_bits) in [
        (
            DType::BF16,
            "bf16_tensor_tensor_tensor_1m",
            0x3f00,
            0x3f80,
            0x4000,
        ),
        (
            DType::F16,
            "f16_tensor_tensor_tensor_1m",
            0x3800,
            0x3c00,
            0x4000,
        ),
    ] {
        let lo: Vec<u16> = (0..N).map(|i| lo_bits + half_offset(i)).collect();
        let x: Vec<u16> = (0..N).map(|i| x_bits + half_offset(i)).collect();
        let hi: Vec<u16> = (0..N).map(|i| hi_bits + half_offset(i)).collect();
        bench_inputs(
            label,
            [
                half_dense(dtype, &lo),
                half_dense(dtype, &x),
                half_dense(dtype, &hi),
            ],
            [
                half_boxed(dtype, &lo),
                half_boxed(dtype, &x),
                half_boxed(dtype, &hi),
            ],
            c,
        );
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets =
        bench_f32_f64_mixed_scalar_tensor,
        bench_i64_mixed_scalar_tensor,
        bench_i64_tensor_tensor,
        bench_half_mixed_scalar_tensor,
        bench_half_tensor_tensor
}
criterion_main!(benches);
