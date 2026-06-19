use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{
    Atom, DType, Equation, Jaxpr, Literal, LiteralBuffer, Primitive, Shape, TensorValue, Value,
    VarId,
};
use smallvec::smallvec;
use std::collections::BTreeMap;
use std::hint::black_box;

fn build_simple_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0), VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    )
}

fn build_large_jaxpr() -> Jaxpr {
    let mut equations = Vec::new();
    let mut next_var = 2_u32;

    for i in 0..50 {
        equations.push(Equation {
            primitive: match i % 4 {
                0 => Primitive::Add,
                1 => Primitive::Mul,
                2 => Primitive::Sub,
                _ => Primitive::Div,
            },
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
            outputs: smallvec![VarId(next_var)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        });
        next_var += 1;
    }

    Jaxpr::new(
        vec![VarId(0), VarId(1)],
        vec![],
        vec![VarId(next_var - 1)],
        equations,
    )
}

fn bench_jaxpr_clone_simple(c: &mut Criterion) {
    let jaxpr = build_simple_jaxpr();
    c.bench_function("core/jaxpr_clone_simple", |b| b.iter(|| jaxpr.clone()));
}

fn bench_jaxpr_clone_large(c: &mut Criterion) {
    let jaxpr = build_large_jaxpr();
    c.bench_function("core/jaxpr_clone_large", |b| b.iter(|| jaxpr.clone()));
}

fn bench_jaxpr_fingerprint_simple(c: &mut Criterion) {
    let jaxpr = build_simple_jaxpr();
    c.bench_function("core/jaxpr_fingerprint_simple", |b| {
        b.iter(|| jaxpr.canonical_fingerprint())
    });
}

fn bench_jaxpr_fingerprint_large(c: &mut Criterion) {
    let jaxpr = build_large_jaxpr();
    c.bench_function("core/jaxpr_fingerprint_large", |b| {
        b.iter(|| jaxpr.canonical_fingerprint())
    });
}

fn bench_jaxpr_validate_simple(c: &mut Criterion) {
    let jaxpr = build_simple_jaxpr();
    c.bench_function("core/jaxpr_validate_simple", |b| {
        b.iter(|| jaxpr.validate_well_formed())
    });
}

fn bench_jaxpr_validate_large(c: &mut Criterion) {
    let jaxpr = build_large_jaxpr();
    c.bench_function("core/jaxpr_validate_large", |b| {
        b.iter(|| jaxpr.validate_well_formed())
    });
}

fn bench_tensor_value_new(c: &mut Criterion) {
    let elements: Vec<Literal> = (0..1000).map(|i| Literal::from_f64(i as f64)).collect();
    let shape = Shape {
        dims: vec![10, 100],
    };
    c.bench_function("core/tensor_value_new_1k_f64_generic_dense", |b| {
        b.iter(|| TensorValue::new(DType::F64, shape.clone(), elements.clone()))
    });
    c.bench_function("core/tensor_value_new_1k_f64_forced_literal", |b| {
        b.iter(|| {
            TensorValue::new_with_literal_buffer(
                DType::F64,
                shape.clone(),
                LiteralBuffer::new(elements.clone()),
            )
        })
    });
    c.bench_function("core/tensor_value_new_then_to_f64_vec_1k", |b| {
        b.iter(|| {
            let tensor = TensorValue::new(DType::F64, shape.clone(), elements.clone())
                .expect("valid f64 tensor");
            black_box(tensor.to_f64_vec())
        })
    });
    c.bench_function(
        "core/tensor_value_new_forced_literal_then_to_f64_vec_1k",
        |b| {
            b.iter(|| {
                let tensor = TensorValue::new_with_literal_buffer(
                    DType::F64,
                    shape.clone(),
                    LiteralBuffer::new(elements.clone()),
                )
                .expect("valid literal-backed f64 tensor");
                black_box(tensor.to_f64_vec())
            })
        },
    );
}

fn bench_tensor_to_i64_vec(c: &mut Criterion) {
    let values: Vec<i64> = (0..4096).map(|i| i as i64 - 2048).collect();
    let shape = Shape::vector(values.len() as u32);
    let dense =
        TensorValue::new_i64_values(shape.clone(), values.clone()).expect("valid dense i64 tensor");
    let literal_elements = values.iter().copied().map(Literal::I64).collect();
    let literal = TensorValue::new_with_literal_buffer(
        DType::I64,
        shape,
        LiteralBuffer::new(literal_elements),
    )
    .expect("valid literal-backed i64 tensor");

    c.bench_function("core/tensor_to_i64_vec_dense_4k", |b| {
        b.iter(|| black_box(&dense).to_i64_vec())
    });
    c.bench_function("core/tensor_to_i64_vec_literal_4k", |b| {
        b.iter(|| black_box(&literal).to_i64_vec())
    });
}

fn bench_literal_buffer_to_vec(c: &mut Criterion) {
    let values: Vec<f64> = (0..65_536).map(|i| i as f64 + 0.25).collect();
    let dense = LiteralBuffer::from_f64_values(values.clone());
    let literal = LiteralBuffer::new(values.into_iter().map(Literal::from_f64).collect());

    c.bench_function("core/literal_buffer_to_vec_dense_f64_64k", |b| {
        b.iter(|| black_box(&dense).to_vec())
    });
    c.bench_function("core/literal_buffer_to_vec_literal_f64_64k", |b| {
        b.iter(|| black_box(&literal).to_vec())
    });
}

fn bench_literal_buffer_serialize(c: &mut Criterion) {
    let values: Vec<f64> = (0..65_536).map(|i| i as f64 + 0.25).collect();
    let dense = LiteralBuffer::from_f64_values(values.clone());
    let literal = LiteralBuffer::new(values.into_iter().map(Literal::from_f64).collect());

    c.bench_function("core/literal_buffer_serialize_dense_f64_64k", |b| {
        b.iter(|| serde_json::to_vec(black_box(&dense)).expect("serialize dense buffer"))
    });
    c.bench_function("core/literal_buffer_serialize_literal_f64_64k", |b| {
        b.iter(|| serde_json::to_vec(black_box(&literal)).expect("serialize literal buffer"))
    });
}

fn bench_literal_buffer_eq(c: &mut Criterion) {
    let values: Vec<f64> = (0..65_536).map(|i| i as f64 + 0.25).collect();
    let dense = LiteralBuffer::from_f64_values(values.clone());
    let dense_same = LiteralBuffer::from_f64_values(values.clone());
    let mut mismatch_values = values.clone();
    mismatch_values[65_535] = -1.25;
    let dense_mismatch = LiteralBuffer::from_f64_values(mismatch_values);
    let literal = LiteralBuffer::new(values.iter().copied().map(Literal::from_f64).collect());
    let literal_same = LiteralBuffer::new(values.into_iter().map(Literal::from_f64).collect());

    c.bench_function("core/literal_buffer_eq_dense_f64_64k_equal", |b| {
        b.iter(|| black_box(&dense) == black_box(&dense_same))
    });
    c.bench_function("core/literal_buffer_eq_dense_f64_64k_mismatch", |b| {
        b.iter(|| black_box(&dense) == black_box(&dense_mismatch))
    });
    c.bench_function("core/literal_buffer_eq_literal_f64_64k_equal", |b| {
        b.iter(|| black_box(&literal) == black_box(&literal_same))
    });
}

fn bench_literal_buffer_index_mut(c: &mut Criterion) {
    let values: Vec<f64> = (0..65_536).map(|i| i as f64 + 0.25).collect();
    let dense = LiteralBuffer::from_f64_values(values.clone());
    let literal = LiteralBuffer::new(values.into_iter().map(Literal::from_f64).collect());

    c.bench_function("core/literal_buffer_index_mut_dense_f64_64k", |b| {
        b.iter(|| {
            let mut buffer = black_box(dense.clone());
            buffer[32_768] = Literal::from_f64(black_box(-1.5));
            black_box(buffer)
        })
    });
    c.bench_function("core/literal_buffer_index_mut_literal_f64_64k", |b| {
        b.iter(|| {
            let mut buffer = black_box(literal.clone());
            buffer[32_768] = Literal::from_f64(black_box(-1.5));
            black_box(buffer)
        })
    });
}

fn bench_tensor_repeat_axis0(c: &mut Criterion) {
    let values: Vec<f64> = (0..1024).map(|i| i as f64).collect();
    let shape = Shape::vector(values.len() as u32);
    let dense = Value::Tensor(
        TensorValue::new_f64_values(shape.clone(), values.clone()).expect("valid dense f64 tensor"),
    );
    let literal_elements = values.into_iter().map(Literal::from_f64).collect();
    let literal = Value::Tensor(
        TensorValue::new(DType::F64, shape, literal_elements)
            .expect("valid literal-backed f64 tensor"),
    );

    c.bench_function("core/tensor_repeat_axis0_dense_f64_1k_x64", |b| {
        b.iter(|| TensorValue::repeat_axis0(black_box(&dense), black_box(64)))
    });
    c.bench_function("core/tensor_repeat_axis0_literal_f64_1k_x64", |b| {
        b.iter(|| TensorValue::repeat_axis0(black_box(&literal), black_box(64)))
    });
}

fn bench_tensor_stack_axis0(c: &mut Criterion) {
    let shape = Shape::vector(1024);
    let dense_slices: Vec<Value> = (0..64)
        .map(|slice| {
            let values = (0..1024)
                .map(|index| f64::from(slice * 1024 + index))
                .collect();
            Value::Tensor(
                TensorValue::new_f64_values(shape.clone(), values).expect("valid dense f64 tensor"),
            )
        })
        .collect();
    let literal_slices: Vec<Value> = (0..64)
        .map(|slice| {
            let elements = (0..1024)
                .map(|index| Literal::from_f64(f64::from(slice * 1024 + index)))
                .collect();
            Value::Tensor(
                TensorValue::new(DType::F64, shape.clone(), elements)
                    .expect("valid literal-backed f64 tensor"),
            )
        })
        .collect();

    c.bench_function("core/tensor_stack_axis0_dense_f64_64x1k", |b| {
        b.iter(|| TensorValue::stack_axis0(black_box(dense_slices.as_slice())))
    });
    c.bench_function("core/tensor_stack_axis0_literal_f64_64x1k", |b| {
        b.iter(|| TensorValue::stack_axis0(black_box(literal_slices.as_slice())))
    });
}

fn bench_scalar_stack_axis0(c: &mut Criterion) {
    let scalars: Vec<Value> = (0..64)
        .map(|index| Value::scalar_f64(f64::from(index)))
        .collect();

    c.bench_function("core/scalar_stack_axis0_f64_64", |b| {
        b.iter(|| TensorValue::stack_axis0(black_box(scalars.as_slice())))
    });
}

fn bench_value_scalar_f64(c: &mut Criterion) {
    c.bench_function("core/value_scalar_f64", |b| {
        b.iter(|| Value::scalar_f64(std::f64::consts::PI))
    });
}

fn bench_value_vector_f64(c: &mut Criterion) {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    c.bench_function("core/value_vector_f64_100", |b| {
        b.iter(|| Value::vector_f64(&data))
    });
}

fn bench_shape_element_count(c: &mut Criterion) {
    let shape = Shape {
        dims: vec![10, 20, 30, 40],
    };
    c.bench_function("core/shape_element_count", |b| {
        b.iter(|| shape.element_count())
    });
}

criterion_group!(
    benches,
    bench_jaxpr_clone_simple,
    bench_jaxpr_clone_large,
    bench_jaxpr_fingerprint_simple,
    bench_jaxpr_fingerprint_large,
    bench_jaxpr_validate_simple,
    bench_jaxpr_validate_large,
    bench_tensor_value_new,
    bench_tensor_to_i64_vec,
    bench_literal_buffer_to_vec,
    bench_literal_buffer_serialize,
    bench_literal_buffer_eq,
    bench_literal_buffer_index_mut,
    bench_tensor_repeat_axis0,
    bench_tensor_stack_axis0,
    bench_scalar_stack_axis0,
    bench_value_scalar_f64,
    bench_value_vector_f64,
    bench_shape_element_count,
);
criterion_main!(benches);
