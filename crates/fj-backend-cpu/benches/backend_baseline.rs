//! Baseline benchmarks for the CPU backend.

use criterion::{Criterion, criterion_group, criterion_main};
use fj_backend_cpu::CpuBackend;
use fj_core::{
    Atom, Equation, Jaxpr, Literal, Primitive, ProgramSpec, Value, VarId, build_program,
};
use fj_interpreters::eval_jaxpr;
use fj_runtime::backend::Backend;
use fj_runtime::buffer::Buffer;
use fj_runtime::device::DeviceId;
use std::collections::BTreeMap;

fn make_wide_parallel_jaxpr(width: usize) -> Jaxpr {
    assert!(width >= 2, "width must be at least 2");
    let input = VarId(1);
    let mut next_var = 2_u32;
    let mut equations = Vec::new();
    let mut active = Vec::new();

    for _ in 0..width {
        let out = VarId(next_var);
        next_var += 1;
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: vec![Atom::Var(input), Atom::Lit(Literal::I64(1))].into(),
            outputs: vec![out].into(),
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        active.push(out);
    }

    while active.len() > 1 {
        let mut next_level = Vec::with_capacity(active.len().div_ceil(2));
        for chunk in active.chunks(2) {
            if chunk.len() == 1 {
                next_level.push(chunk[0]);
                continue;
            }
            let out = VarId(next_var);
            next_var += 1;
            equations.push(Equation {
                primitive: Primitive::Add,
                inputs: vec![Atom::Var(chunk[0]), Atom::Var(chunk[1])].into(),
                outputs: vec![out].into(),
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            });
            next_level.push(out);
        }
        active = next_level;
    }

    Jaxpr::new(vec![input], vec![], vec![active[0]], equations)
}

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

fn bench_execute_wide_parallel(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let jaxpr = make_wide_parallel_jaxpr(64);
    let args = vec![Value::scalar_i64(7)];
    c.bench_function("backend_execute/wide_parallel_64", |b| {
        b.iter(|| backend.execute(&jaxpr, &args, DeviceId(0)).unwrap())
    });
}

fn bench_interpreter_wide_parallel(c: &mut Criterion) {
    let jaxpr = make_wide_parallel_jaxpr(64);
    let args = vec![Value::scalar_i64(7)];
    c.bench_function("interpreter_execute/wide_parallel_64", |b| {
        b.iter(|| eval_jaxpr(&jaxpr, &args).unwrap())
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
    bench_execute_wide_parallel,
    bench_interpreter_wide_parallel,
    bench_allocate,
    bench_transfer,
    bench_device_discovery,
);
criterion_main!(benches);
