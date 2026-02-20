use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{Primitive, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
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

fn bench_reshape(c: &mut Criterion) {
    let data: Vec<i64> = (0..1000).collect();
    let input = Value::vector_i64(&data).unwrap();
    let mut params = BTreeMap::new();
    params.insert("new_shape".into(), "10,100".into());
    c.bench_function("eval/reshape_1k_to_10x100", |bencher| {
        bencher.iter(|| eval_primitive(Primitive::Reshape, std::slice::from_ref(&input), &params))
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
    bench_sin_1k,
    bench_exp_1k,
    bench_reshape,
    bench_eq_1k,
);
criterion_main!(benches);
