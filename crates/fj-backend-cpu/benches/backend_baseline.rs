//! Baseline benchmarks for the CPU backend.

use criterion::{Criterion, criterion_group, criterion_main};
use fj_backend_cpu::CpuBackend;
use fj_core::{ProgramSpec, Value, build_program};
use fj_runtime::backend::Backend;
use fj_runtime::buffer::Buffer;
use fj_runtime::device::DeviceId;

fn bench_execute_add2(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let jaxpr = build_program(ProgramSpec::Add2);
    let args = vec![Value::scalar_i64(3), Value::scalar_i64(4)];
    c.bench_function("backend_execute/add2", |b| {
        b.iter(|| backend.execute(&jaxpr, &args, DeviceId(0)).unwrap())
    });
}

fn bench_execute_square(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let jaxpr = build_program(ProgramSpec::Square);
    let args = vec![Value::scalar_f64(5.0)];
    c.bench_function("backend_execute/square", |b| {
        b.iter(|| backend.execute(&jaxpr, &args, DeviceId(0)).unwrap())
    });
}

fn bench_execute_10eqn(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let jaxpr = build_program(ProgramSpec::Dot3);
    let args = vec![Value::scalar_f64(1.0), Value::scalar_f64(2.0)];
    c.bench_function("backend_execute/dot3_10eqn", |b| {
        b.iter(|| backend.execute(&jaxpr, &args, DeviceId(0)).unwrap())
    });
}

fn bench_allocate(c: &mut Criterion) {
    let backend = CpuBackend::new();
    c.bench_function("backend_allocate/256_bytes", |b| {
        b.iter(|| backend.allocate(256, DeviceId(0)).unwrap())
    });
}

fn bench_transfer(c: &mut Criterion) {
    let backend = CpuBackend::with_device_count(2);
    let buf = Buffer::new(vec![0u8; 1024], DeviceId(0));
    c.bench_function("backend_transfer/1kb", |b| {
        b.iter(|| backend.transfer(&buf, DeviceId(1)).unwrap())
    });
}

fn bench_device_discovery(c: &mut Criterion) {
    let backend = CpuBackend::new();
    c.bench_function("backend_discovery/devices", |b| {
        b.iter(|| backend.devices())
    });
}

criterion_group!(
    benches,
    bench_execute_add2,
    bench_execute_square,
    bench_execute_10eqn,
    bench_allocate,
    bench_transfer,
    bench_device_discovery,
);
criterion_main!(benches);
