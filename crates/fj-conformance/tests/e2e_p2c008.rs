//! FJ-P2C-008 E2E Scenario Scripts: LAX Primitive First Wave
//!
//! End-to-end scenarios testing LAX primitive evaluation across scalar,
//! tensor, composition, edge cases, and error paths.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::{EvalError, eval_primitive};
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Scenario 1: All Primitives Scalar ========================

#[test]
fn e2e_all_primitives_scalar() {
    let p = no_params();

    // Binary arithmetic
    assert_eq!(
        eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
            &p
        )
        .unwrap(),
        Value::scalar_i64(7)
    );
    assert_eq!(
        eval_primitive(
            Primitive::Sub,
            &[Value::scalar_i64(10), Value::scalar_i64(3)],
            &p
        )
        .unwrap(),
        Value::scalar_i64(7)
    );
    assert_eq!(
        eval_primitive(
            Primitive::Mul,
            &[Value::scalar_i64(6), Value::scalar_i64(7)],
            &p
        )
        .unwrap(),
        Value::scalar_i64(42)
    );
    assert_eq!(
        eval_primitive(
            Primitive::Max,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &p
        )
        .unwrap(),
        Value::scalar_i64(7)
    );
    assert_eq!(
        eval_primitive(
            Primitive::Min,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &p
        )
        .unwrap(),
        Value::scalar_i64(3)
    );

    // Unary arithmetic
    assert_eq!(
        eval_primitive(Primitive::Neg, &[Value::scalar_i64(5)], &p).unwrap(),
        Value::scalar_i64(-5)
    );
    assert_eq!(
        eval_primitive(Primitive::Abs, &[Value::scalar_i64(-42)], &p).unwrap(),
        Value::scalar_i64(42)
    );

    // Transcendental (f64)
    let exp_1 = eval_primitive(Primitive::Exp, &[Value::scalar_f64(0.0)], &p)
        .unwrap()
        .as_f64_scalar()
        .unwrap();
    assert!((exp_1 - 1.0).abs() < 1e-14);

    let log_e = eval_primitive(
        Primitive::Log,
        &[Value::scalar_f64(std::f64::consts::E)],
        &p,
    )
    .unwrap()
    .as_f64_scalar()
    .unwrap();
    assert!((log_e - 1.0).abs() < 1e-14);

    let sqrt_4 = eval_primitive(Primitive::Sqrt, &[Value::scalar_f64(4.0)], &p)
        .unwrap()
        .as_f64_scalar()
        .unwrap();
    assert!((sqrt_4 - 2.0).abs() < 1e-14);

    let rsqrt_4 = eval_primitive(Primitive::Rsqrt, &[Value::scalar_f64(4.0)], &p)
        .unwrap()
        .as_f64_scalar()
        .unwrap();
    assert!((rsqrt_4 - 0.5).abs() < 1e-14);

    let floor_37 = eval_primitive(Primitive::Floor, &[Value::scalar_f64(3.7)], &p)
        .unwrap()
        .as_f64_scalar()
        .unwrap();
    assert!((floor_37 - 3.0).abs() < 1e-14);

    let ceil_32 = eval_primitive(Primitive::Ceil, &[Value::scalar_f64(3.2)], &p)
        .unwrap()
        .as_f64_scalar()
        .unwrap();
    assert!((ceil_32 - 4.0).abs() < 1e-14);

    let round_35 = eval_primitive(Primitive::Round, &[Value::scalar_f64(3.5)], &p)
        .unwrap()
        .as_f64_scalar()
        .unwrap();
    assert!((round_35 - 4.0).abs() < 1e-14);

    let sin_0 = eval_primitive(Primitive::Sin, &[Value::scalar_f64(0.0)], &p)
        .unwrap()
        .as_f64_scalar()
        .unwrap();
    assert!(sin_0.abs() < 1e-14);

    let cos_0 = eval_primitive(Primitive::Cos, &[Value::scalar_f64(0.0)], &p)
        .unwrap()
        .as_f64_scalar()
        .unwrap();
    assert!((cos_0 - 1.0).abs() < 1e-14);

    // Pow
    let pow_23 = eval_primitive(
        Primitive::Pow,
        &[Value::scalar_f64(2.0), Value::scalar_f64(3.0)],
        &p,
    )
    .unwrap()
    .as_f64_scalar()
    .unwrap();
    assert!((pow_23 - 8.0).abs() < 1e-10);

    // Comparison
    assert_eq!(
        eval_primitive(
            Primitive::Eq,
            &[Value::scalar_i64(5), Value::scalar_i64(5)],
            &p
        )
        .unwrap(),
        Value::scalar_bool(true)
    );
    assert_eq!(
        eval_primitive(
            Primitive::Lt,
            &[Value::scalar_i64(3), Value::scalar_i64(5)],
            &p
        )
        .unwrap(),
        Value::scalar_bool(true)
    );

    // Reduction (scalar passthrough)
    assert_eq!(
        eval_primitive(Primitive::ReduceSum, &[Value::scalar_i64(42)], &p).unwrap(),
        Value::scalar_i64(42)
    );

    // Dot scalar
    assert_eq!(
        eval_primitive(
            Primitive::Dot,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &p
        )
        .unwrap(),
        Value::scalar_i64(21)
    );
}

// ======================== Scenario 2: All Primitives Tensor ========================

#[test]
fn e2e_all_primitives_tensor_rank1() {
    let p = no_params();
    let a = Value::vector_i64(&[1, 2, 3]).unwrap();
    let b = Value::vector_i64(&[4, 5, 6]).unwrap();

    // Binary
    assert_eq!(
        eval_primitive(Primitive::Add, &[a.clone(), b.clone()], &p).unwrap(),
        Value::vector_i64(&[5, 7, 9]).unwrap()
    );
    assert_eq!(
        eval_primitive(Primitive::Sub, &[b.clone(), a.clone()], &p).unwrap(),
        Value::vector_i64(&[3, 3, 3]).unwrap()
    );
    assert_eq!(
        eval_primitive(Primitive::Mul, &[a.clone(), b.clone()], &p).unwrap(),
        Value::vector_i64(&[4, 10, 18]).unwrap()
    );

    // Unary
    assert_eq!(
        eval_primitive(Primitive::Neg, std::slice::from_ref(&a), &p).unwrap(),
        Value::vector_i64(&[-1, -2, -3]).unwrap()
    );

    // Comparison
    let lt = eval_primitive(Primitive::Lt, &[a.clone(), b.clone()], &p).unwrap();
    if let Value::Tensor(t) = &lt {
        assert!(t.elements.iter().all(|e| *e == Literal::Bool(true)));
    }

    // Reduction
    assert_eq!(
        eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&a), &p).unwrap(),
        Value::scalar_i64(6)
    );
    assert_eq!(
        eval_primitive(Primitive::ReduceMax, std::slice::from_ref(&a), &p).unwrap(),
        Value::scalar_i64(3)
    );

    // Dot
    assert_eq!(
        eval_primitive(Primitive::Dot, &[a, b], &p).unwrap(),
        Value::scalar_i64(32) // 1*4+2*5+3*6
    );
}

#[test]
fn e2e_tensor_rank2_operations() {
    // 2x3 matrix
    let mat = TensorValue::new(
        DType::I64,
        Shape { dims: vec![2, 3] },
        vec![
            Literal::I64(1),
            Literal::I64(2),
            Literal::I64(3),
            Literal::I64(4),
            Literal::I64(5),
            Literal::I64(6),
        ],
    )
    .unwrap();

    // Transpose 2x3 -> 3x2
    let transposed = eval_primitive(
        Primitive::Transpose,
        &[Value::Tensor(mat.clone())],
        &no_params(),
    )
    .unwrap();
    if let Value::Tensor(t) = &transposed {
        assert_eq!(t.shape.dims, vec![3, 2]);
    }

    // Reshape 2x3 -> 6
    let mut params = BTreeMap::new();
    params.insert("new_shape".into(), "6".into());
    let flat = eval_primitive(Primitive::Reshape, &[Value::Tensor(mat.clone())], &params).unwrap();
    if let Value::Tensor(t) = &flat {
        assert_eq!(t.shape.dims, vec![6]);
        assert_eq!(t.elements.len(), 6);
    }

    // Reduce sum (full tensor)
    let sum = eval_primitive(Primitive::ReduceSum, &[Value::Tensor(mat)], &no_params()).unwrap();
    assert_eq!(sum, Value::scalar_i64(21)); // 1+2+3+4+5+6
}

// ======================== Scenario 3: Primitive Composition ========================

#[test]
fn e2e_composition_add_mul_reduce() {
    // Compute: reduce_sum(add([1,2,3], [4,5,6]) * [2,2,2])
    // = reduce_sum(mul([5,7,9], [2,2,2]))
    // = reduce_sum([10,14,18])
    // = 42
    let a = Value::vector_i64(&[1, 2, 3]).unwrap();
    let b = Value::vector_i64(&[4, 5, 6]).unwrap();
    let c = Value::vector_i64(&[2, 2, 2]).unwrap();
    let p = no_params();

    let added = eval_primitive(Primitive::Add, &[a, b], &p).unwrap();
    let multiplied = eval_primitive(Primitive::Mul, &[added, c], &p).unwrap();
    let result = eval_primitive(Primitive::ReduceSum, &[multiplied], &p).unwrap();

    assert_eq!(result, Value::scalar_i64(42));
}

#[test]
fn e2e_composition_exp_log_roundtrip() {
    // For x > 0: exp(log(x)) ≈ x
    let x = Value::scalar_f64(42.0);
    let p = no_params();

    let log_x = eval_primitive(Primitive::Log, &[x], &p).unwrap();
    let exp_log_x = eval_primitive(Primitive::Exp, &[log_x], &p).unwrap();
    let result = exp_log_x.as_f64_scalar().unwrap();
    assert!((result - 42.0).abs() < 1e-10);
}

#[test]
fn e2e_composition_reshape_transpose_reshape() {
    // [1,2,3,4,5,6] -> 2x3 -> transpose -> 3x2 -> [1,4,2,5,3,6]
    let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();

    let mut params = BTreeMap::new();
    params.insert("new_shape".into(), "2,3".into());
    let mat = eval_primitive(Primitive::Reshape, &[input], &params).unwrap();

    let transposed = eval_primitive(Primitive::Transpose, &[mat], &no_params()).unwrap();

    let mut flatten_params = BTreeMap::new();
    flatten_params.insert("new_shape".into(), "6".into());
    let flat = eval_primitive(Primitive::Reshape, &[transposed], &flatten_params).unwrap();

    if let Value::Tensor(t) = &flat {
        let vals: Vec<i64> = t.elements.iter().map(|e| e.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![1, 4, 2, 5, 3, 6]);
    } else {
        panic!("expected tensor");
    }
}

// ======================== Scenario 4: Edge Cases ========================

#[test]
fn e2e_edge_cases_nan_inf() {
    let p = no_params();

    // NaN propagation through composition
    let nan_add = eval_primitive(
        Primitive::Add,
        &[Value::scalar_f64(f64::NAN), Value::scalar_f64(1.0)],
        &p,
    )
    .unwrap();
    let nan_mul = eval_primitive(Primitive::Mul, &[nan_add, Value::scalar_f64(2.0)], &p).unwrap();
    assert!(
        nan_mul.as_f64_scalar().unwrap().is_nan(),
        "NaN should propagate through add->mul"
    );

    // Inf arithmetic
    let inf_sub = eval_primitive(
        Primitive::Sub,
        &[
            Value::scalar_f64(f64::INFINITY),
            Value::scalar_f64(f64::INFINITY),
        ],
        &p,
    )
    .unwrap();
    assert!(
        inf_sub.as_f64_scalar().unwrap().is_nan(),
        "Inf - Inf should be NaN"
    );

    // Zero handling
    assert_eq!(
        eval_primitive(
            Primitive::Mul,
            &[Value::scalar_i64(0), Value::scalar_i64(i64::MAX)],
            &p
        )
        .unwrap(),
        Value::scalar_i64(0)
    );
}

#[test]
fn e2e_edge_cases_comparison_with_nan() {
    let p = no_params();
    let nan = Value::scalar_f64(f64::NAN);

    // All comparisons with NaN should return false (except !=)
    for prim in [
        Primitive::Eq,
        Primitive::Lt,
        Primitive::Le,
        Primitive::Gt,
        Primitive::Ge,
    ] {
        let result = eval_primitive(prim, &[nan.clone(), Value::scalar_f64(1.0)], &p).unwrap();
        assert_eq!(
            result,
            Value::scalar_bool(false),
            "{:?}(NaN, 1.0) should be false",
            prim
        );
    }
    // != with NaN is true
    let ne_result = eval_primitive(Primitive::Ne, &[nan, Value::scalar_f64(1.0)], &p).unwrap();
    assert_eq!(ne_result, Value::scalar_bool(true));
}

// ======================== Scenario 5: Error Paths ========================

#[test]
fn e2e_error_paths() {
    let p = no_params();

    // Wrong arity for binary op
    let err = eval_primitive(Primitive::Add, &[Value::scalar_i64(1)], &p).unwrap_err();
    assert!(matches!(err, EvalError::ArityMismatch { expected: 2, .. }));

    // Wrong arity for unary op
    let err = eval_primitive(
        Primitive::Neg,
        &[Value::scalar_i64(1), Value::scalar_i64(2)],
        &p,
    )
    .unwrap_err();
    assert!(matches!(err, EvalError::ArityMismatch { expected: 1, .. }));

    // Shape mismatch
    let a = Value::vector_i64(&[1, 2]).unwrap();
    let b = Value::vector_i64(&[1, 2, 3]).unwrap();
    let err = eval_primitive(Primitive::Add, &[a, b], &p).unwrap_err();
    assert!(matches!(err, EvalError::ShapeMismatch { .. }));

    // Unsupported primitive
    let err = eval_primitive(Primitive::Gather, &[Value::scalar_i64(1)], &p).unwrap_err();
    assert!(matches!(err, EvalError::Unsupported { .. }));
}

// ======================== Scenario 6: Broadcasting Pipeline ========================

#[test]
fn e2e_broadcasting_pipeline() {
    let p = no_params();

    // scalar + vector -> vector
    let vec = Value::vector_i64(&[10, 20, 30]).unwrap();
    let result = eval_primitive(Primitive::Add, &[Value::scalar_i64(5), vec], &p).unwrap();
    assert_eq!(result, Value::vector_i64(&[15, 25, 35]).unwrap());

    // vector + scalar -> vector
    let vec2 = Value::vector_i64(&[1, 2, 3]).unwrap();
    let result = eval_primitive(Primitive::Mul, &[vec2, Value::scalar_i64(10)], &p).unwrap();
    assert_eq!(result, Value::vector_i64(&[10, 20, 30]).unwrap());

    // Comparison with broadcast
    let vec3 = Value::vector_i64(&[1, 5, 3]).unwrap();
    let cmp = eval_primitive(Primitive::Gt, &[vec3, Value::scalar_i64(3)], &p).unwrap();
    if let Value::Tensor(t) = &cmp {
        assert_eq!(t.elements[0], Literal::Bool(false)); // 1 > 3
        assert_eq!(t.elements[1], Literal::Bool(true)); // 5 > 3
        assert_eq!(t.elements[2], Literal::Bool(false)); // 3 > 3
    }
}
