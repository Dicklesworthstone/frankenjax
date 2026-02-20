//! FJ-P2C-007-H: FFI performance baseline benchmarks.

#![allow(unsafe_code)]

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use fj_core::{DType, Literal, Shape, TensorValue, Value};
use fj_ffi::{FfiBuffer, FfiCall, FfiRegistry, buffer_to_value, value_to_buffer};

unsafe extern "C" fn ffi_double(
    inputs: *const *const u8,
    _input_count: usize,
    outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    unsafe {
        let src = *inputs as *const f64;
        let dst = *outputs as *mut f64;
        *dst = *src * 2.0;
    }
    0
}

unsafe extern "C" fn ffi_noop(
    _inputs: *const *const u8,
    _input_count: usize,
    _outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    0
}

unsafe extern "C" fn ffi_vec_negate(
    inputs: *const *const u8,
    _input_count: usize,
    outputs: *const *mut u8,
    _output_count: usize,
) -> i32 {
    unsafe {
        let src = *inputs as *const f64;
        let dst = *outputs as *mut f64;
        for i in 0..1000 {
            *dst.add(i) = -(*src.add(i));
        }
    }
    0
}

fn setup_registry() -> FfiRegistry {
    let reg = FfiRegistry::new();
    reg.register("double", ffi_double).unwrap();
    reg.register("noop", ffi_noop).unwrap();
    reg.register("negate_1k", ffi_vec_negate).unwrap();
    reg
}

fn bench_ffi_scalar_roundtrip(c: &mut Criterion) {
    let reg = setup_registry();
    let call = FfiCall::new("double");

    c.bench_function("ffi_roundtrip/scalar_f64", |b| {
        b.iter(|| {
            let input = FfiBuffer::new(
                black_box(42.0f64).to_ne_bytes().to_vec(),
                vec![],
                DType::F64,
            )
            .unwrap();
            let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];
            call.invoke(&reg, &[input], &mut outputs).unwrap();
            black_box(&outputs[0]);
        })
    });
}

fn bench_ffi_noop(c: &mut Criterion) {
    let reg = setup_registry();
    let call = FfiCall::new("noop");

    c.bench_function("ffi_roundtrip/noop", |b| {
        b.iter(|| {
            call.invoke(&reg, &[], &mut []).unwrap();
        })
    });
}

fn bench_ffi_1k_tensor(c: &mut Criterion) {
    let reg = setup_registry();
    let call = FfiCall::new("negate_1k");

    c.bench_function("ffi_roundtrip/1k_f64_vec", |b| {
        b.iter(|| {
            let mut data = Vec::with_capacity(8000);
            for i in 0..1000 {
                data.extend_from_slice(&(i as f64).to_ne_bytes());
            }
            let input = FfiBuffer::new(data, vec![1000], DType::F64).unwrap();
            let mut outputs = [FfiBuffer::zeroed(vec![1000], DType::F64).unwrap()];
            call.invoke(&reg, &[input], &mut outputs).unwrap();
            black_box(&outputs[0]);
        })
    });
}

fn bench_marshal_scalar(c: &mut Criterion) {
    let val = Value::Scalar(Literal::F64Bits(42.0f64.to_bits()));

    c.bench_function("marshal/scalar_to_buffer", |b| {
        b.iter(|| {
            let buf = value_to_buffer(black_box(&val)).unwrap();
            black_box(buf);
        })
    });

    let buf = value_to_buffer(&val).unwrap();
    c.bench_function("marshal/buffer_to_scalar", |b| {
        b.iter(|| {
            let v = buffer_to_value(black_box(&buf)).unwrap();
            black_box(v);
        })
    });
}

fn bench_marshal_tensor_1k(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..1000)
        .map(|i| Literal::F64Bits((i as f64).to_bits()))
        .collect();
    let val = Value::Tensor(TensorValue {
        dtype: DType::F64,
        shape: Shape { dims: vec![1000] },
        elements,
    });

    c.bench_function("marshal/1k_tensor_to_buffer", |b| {
        b.iter(|| {
            let buf = value_to_buffer(black_box(&val)).unwrap();
            black_box(buf);
        })
    });

    let buf = value_to_buffer(&val).unwrap();
    c.bench_function("marshal/1k_buffer_to_tensor", |b| {
        b.iter(|| {
            let v = buffer_to_value(black_box(&buf)).unwrap();
            black_box(v);
        })
    });
}

fn bench_registry_lookup(c: &mut Criterion) {
    let reg = setup_registry();

    c.bench_function("registry/lookup", |b| {
        b.iter(|| {
            let target = reg.get(black_box("double")).unwrap();
            black_box(target);
        })
    });
}

criterion_group!(
    benches,
    bench_ffi_scalar_roundtrip,
    bench_ffi_noop,
    bench_ffi_1k_tensor,
    bench_marshal_scalar,
    bench_marshal_tensor_1k,
    bench_registry_lookup,
);
criterion_main!(benches);
