//! FJ-P2C-008 Differential Oracle + Metamorphic + Adversarial Validation
//!
//! Tests LAX primitive evaluation against expected mathematical behavior.
//! Since we don't have a live JAX process, we use hand-verified oracle values
//! matching JAX's documented behavior for each primitive.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::{EvalError, eval_primitive};
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn assert_f64_close(actual: f64, expected: f64, tol: f64, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: expected NaN, got {actual}");
    } else if expected.is_infinite() {
        assert_eq!(
            actual, expected,
            "{context}: expected {expected}, got {actual}"
        );
    } else {
        assert!(
            (actual - expected).abs() < tol,
            "{context}: expected {expected}, got {actual} (tol={tol})"
        );
    }
}

fn eval_f64(prim: Primitive, inputs: &[Value], params: &BTreeMap<String, String>) -> f64 {
    eval_primitive(prim, inputs, params)
        .unwrap()
        .as_f64_scalar()
        .unwrap()
}

fn eval_i64(prim: Primitive, inputs: &[Value], params: &BTreeMap<String, String>) -> i64 {
    match eval_primitive(prim, inputs, params).unwrap() {
        Value::Scalar(Literal::I64(v)) => v,
        other => panic!("expected i64 scalar, got {other:?}"),
    }
}

fn eval_bool(prim: Primitive, inputs: &[Value], params: &BTreeMap<String, String>) -> bool {
    match eval_primitive(prim, inputs, params).unwrap() {
        Value::Scalar(Literal::Bool(v)) => v,
        other => panic!("expected bool scalar, got {other:?}"),
    }
}

// ======================== Oracle: Arithmetic ========================

#[test]
fn oracle_add_i64() {
    assert_eq!(
        eval_i64(
            Primitive::Add,
            &[Value::scalar_i64(7), Value::scalar_i64(3)],
            &no_params()
        ),
        10
    );
    assert_eq!(
        eval_i64(
            Primitive::Add,
            &[Value::scalar_i64(-5), Value::scalar_i64(5)],
            &no_params()
        ),
        0
    );
    assert_eq!(
        eval_i64(
            Primitive::Add,
            &[Value::scalar_i64(0), Value::scalar_i64(0)],
            &no_params()
        ),
        0
    );
}

#[test]
fn oracle_add_f64() {
    assert_f64_close(
        eval_f64(
            Primitive::Add,
            &[Value::scalar_f64(1.5), Value::scalar_f64(2.5)],
            &no_params(),
        ),
        4.0,
        1e-14,
        "add(1.5, 2.5)",
    );
    assert_f64_close(
        eval_f64(
            Primitive::Add,
            &[Value::scalar_f64(f64::INFINITY), Value::scalar_f64(1.0)],
            &no_params(),
        ),
        f64::INFINITY,
        1e-14,
        "add(Inf, 1)",
    );
    assert_f64_close(
        eval_f64(
            Primitive::Add,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(1.0)],
            &no_params(),
        ),
        f64::NAN,
        1e-14,
        "add(NaN, 1)",
    );
}

#[test]
fn oracle_sub_i64() {
    assert_eq!(
        eval_i64(
            Primitive::Sub,
            &[Value::scalar_i64(10), Value::scalar_i64(3)],
            &no_params()
        ),
        7
    );
    assert_eq!(
        eval_i64(
            Primitive::Sub,
            &[Value::scalar_i64(3), Value::scalar_i64(10)],
            &no_params()
        ),
        -7
    );
}

#[test]
fn oracle_sub_f64_inf() {
    assert_f64_close(
        eval_f64(
            Primitive::Sub,
            &[
                Value::scalar_f64(f64::INFINITY),
                Value::scalar_f64(f64::INFINITY),
            ],
            &no_params(),
        ),
        f64::NAN,
        1e-14,
        "Inf - Inf = NaN",
    );
}

#[test]
fn oracle_mul_i64() {
    assert_eq!(
        eval_i64(
            Primitive::Mul,
            &[Value::scalar_i64(6), Value::scalar_i64(7)],
            &no_params()
        ),
        42
    );
    assert_eq!(
        eval_i64(
            Primitive::Mul,
            &[Value::scalar_i64(-3), Value::scalar_i64(4)],
            &no_params()
        ),
        -12
    );
    assert_eq!(
        eval_i64(
            Primitive::Mul,
            &[Value::scalar_i64(0), Value::scalar_i64(999)],
            &no_params()
        ),
        0
    );
}

#[test]
fn oracle_neg() {
    assert_eq!(
        eval_i64(Primitive::Neg, &[Value::scalar_i64(5)], &no_params()),
        -5
    );
    assert_eq!(
        eval_i64(Primitive::Neg, &[Value::scalar_i64(-5)], &no_params()),
        5
    );
    assert_eq!(
        eval_i64(Primitive::Neg, &[Value::scalar_i64(0)], &no_params()),
        0
    );
}

#[test]
fn oracle_abs() {
    assert_eq!(
        eval_i64(Primitive::Abs, &[Value::scalar_i64(-42)], &no_params()),
        42
    );
    assert_eq!(
        eval_i64(Primitive::Abs, &[Value::scalar_i64(42)], &no_params()),
        42
    );
    assert_eq!(
        eval_i64(Primitive::Abs, &[Value::scalar_i64(0)], &no_params()),
        0
    );
}

#[test]
fn oracle_max() {
    assert_eq!(
        eval_i64(
            Primitive::Max,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params()
        ),
        7
    );
    assert_eq!(
        eval_i64(
            Primitive::Max,
            &[Value::scalar_i64(7), Value::scalar_i64(3)],
            &no_params()
        ),
        7
    );
    assert_f64_close(
        eval_f64(
            Primitive::Max,
            &[Value::scalar_f64(f64::INFINITY), Value::scalar_f64(1.0)],
            &no_params(),
        ),
        f64::INFINITY,
        1e-14,
        "max(Inf, 1)",
    );
}

#[test]
fn oracle_min() {
    assert_eq!(
        eval_i64(
            Primitive::Min,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params()
        ),
        3
    );
    assert_f64_close(
        eval_f64(
            Primitive::Min,
            &[Value::scalar_f64(f64::NEG_INFINITY), Value::scalar_f64(1.0)],
            &no_params(),
        ),
        f64::NEG_INFINITY,
        1e-14,
        "min(-Inf, 1)",
    );
}

#[test]
fn oracle_pow() {
    assert_f64_close(
        eval_f64(
            Primitive::Pow,
            &[Value::scalar_f64(2.0), Value::scalar_f64(10.0)],
            &no_params(),
        ),
        1024.0,
        1e-10,
        "2^10",
    );
    assert_f64_close(
        eval_f64(
            Primitive::Pow,
            &[Value::scalar_f64(3.0), Value::scalar_f64(0.0)],
            &no_params(),
        ),
        1.0,
        1e-14,
        "3^0 = 1",
    );
    assert_f64_close(
        eval_f64(
            Primitive::Pow,
            &[Value::scalar_f64(0.0), Value::scalar_f64(0.0)],
            &no_params(),
        ),
        1.0,
        1e-14,
        "0^0 = 1",
    );
}

// ======================== Oracle: Transcendental ========================

#[test]
fn oracle_exp() {
    assert_f64_close(
        eval_f64(Primitive::Exp, &[Value::scalar_f64(0.0)], &no_params()),
        1.0,
        1e-14,
        "exp(0)",
    );
    assert_f64_close(
        eval_f64(Primitive::Exp, &[Value::scalar_f64(1.0)], &no_params()),
        std::f64::consts::E,
        1e-14,
        "exp(1)",
    );
    assert_f64_close(
        eval_f64(
            Primitive::Exp,
            &[Value::scalar_f64(f64::NEG_INFINITY)],
            &no_params(),
        ),
        0.0,
        1e-14,
        "exp(-Inf)",
    );
    assert_f64_close(
        eval_f64(
            Primitive::Exp,
            &[Value::scalar_f64(f64::INFINITY)],
            &no_params(),
        ),
        f64::INFINITY,
        1e-14,
        "exp(Inf)",
    );
}

#[test]
fn oracle_log() {
    assert_f64_close(
        eval_f64(Primitive::Log, &[Value::scalar_f64(1.0)], &no_params()),
        0.0,
        1e-14,
        "log(1)",
    );
    assert_f64_close(
        eval_f64(
            Primitive::Log,
            &[Value::scalar_f64(std::f64::consts::E)],
            &no_params(),
        ),
        1.0,
        1e-14,
        "log(e)",
    );
    assert_f64_close(
        eval_f64(Primitive::Log, &[Value::scalar_f64(0.0)], &no_params()),
        f64::NEG_INFINITY,
        1e-14,
        "log(0)",
    );
    assert_f64_close(
        eval_f64(Primitive::Log, &[Value::scalar_f64(-1.0)], &no_params()),
        f64::NAN,
        1e-14,
        "log(-1)",
    );
}

#[test]
fn oracle_sqrt() {
    assert_f64_close(
        eval_f64(Primitive::Sqrt, &[Value::scalar_f64(4.0)], &no_params()),
        2.0,
        1e-14,
        "sqrt(4)",
    );
    assert_f64_close(
        eval_f64(Primitive::Sqrt, &[Value::scalar_f64(0.0)], &no_params()),
        0.0,
        1e-14,
        "sqrt(0)",
    );
    assert_f64_close(
        eval_f64(Primitive::Sqrt, &[Value::scalar_f64(-1.0)], &no_params()),
        f64::NAN,
        1e-14,
        "sqrt(-1)",
    );
    assert_f64_close(
        eval_f64(
            Primitive::Sqrt,
            &[Value::scalar_f64(f64::INFINITY)],
            &no_params(),
        ),
        f64::INFINITY,
        1e-14,
        "sqrt(Inf)",
    );
}

#[test]
fn oracle_rsqrt() {
    assert_f64_close(
        eval_f64(Primitive::Rsqrt, &[Value::scalar_f64(4.0)], &no_params()),
        0.5,
        1e-14,
        "rsqrt(4)",
    );
    assert_f64_close(
        eval_f64(Primitive::Rsqrt, &[Value::scalar_f64(1.0)], &no_params()),
        1.0,
        1e-14,
        "rsqrt(1)",
    );
}

#[test]
fn oracle_sin_cos() {
    assert_f64_close(
        eval_f64(Primitive::Sin, &[Value::scalar_f64(0.0)], &no_params()),
        0.0,
        1e-14,
        "sin(0)",
    );
    assert_f64_close(
        eval_f64(Primitive::Cos, &[Value::scalar_f64(0.0)], &no_params()),
        1.0,
        1e-14,
        "cos(0)",
    );
    assert_f64_close(
        eval_f64(
            Primitive::Sin,
            &[Value::scalar_f64(std::f64::consts::FRAC_PI_2)],
            &no_params(),
        ),
        1.0,
        1e-14,
        "sin(pi/2)",
    );
    assert_f64_close(
        eval_f64(
            Primitive::Cos,
            &[Value::scalar_f64(std::f64::consts::PI)],
            &no_params(),
        ),
        -1.0,
        1e-14,
        "cos(pi)",
    );
}

#[test]
fn oracle_floor_ceil_round() {
    assert_f64_close(
        eval_f64(Primitive::Floor, &[Value::scalar_f64(3.7)], &no_params()),
        3.0,
        1e-14,
        "floor(3.7)",
    );
    assert_f64_close(
        eval_f64(Primitive::Floor, &[Value::scalar_f64(-1.2)], &no_params()),
        -2.0,
        1e-14,
        "floor(-1.2)",
    );
    assert_f64_close(
        eval_f64(Primitive::Ceil, &[Value::scalar_f64(3.2)], &no_params()),
        4.0,
        1e-14,
        "ceil(3.2)",
    );
    assert_f64_close(
        eval_f64(Primitive::Ceil, &[Value::scalar_f64(-1.7)], &no_params()),
        -1.0,
        1e-14,
        "ceil(-1.7)",
    );
    assert_f64_close(
        eval_f64(Primitive::Round, &[Value::scalar_f64(3.5)], &no_params()),
        4.0,
        1e-14,
        "round(3.5)",
    );
    // Rust f64::round rounds half away from zero (3.0), not banker's rounding (2.0)
    assert_f64_close(
        eval_f64(Primitive::Round, &[Value::scalar_f64(2.5)], &no_params()),
        3.0,
        1e-14,
        "round(2.5)",
    );
}

// ======================== Oracle: Comparison ========================

#[test]
fn oracle_eq() {
    assert!(eval_bool(
        Primitive::Eq,
        &[Value::scalar_i64(3), Value::scalar_i64(3)],
        &no_params()
    ));
    assert!(!eval_bool(
        Primitive::Eq,
        &[Value::scalar_i64(3), Value::scalar_i64(4)],
        &no_params()
    ));
    assert!(!eval_bool(
        Primitive::Eq,
        &[Value::scalar_f64(f64::NAN), Value::scalar_f64(f64::NAN)],
        &no_params()
    ));
}

#[test]
fn oracle_ne() {
    assert!(!eval_bool(
        Primitive::Ne,
        &[Value::scalar_i64(3), Value::scalar_i64(3)],
        &no_params()
    ));
    assert!(eval_bool(
        Primitive::Ne,
        &[Value::scalar_i64(3), Value::scalar_i64(4)],
        &no_params()
    ));
    assert!(eval_bool(
        Primitive::Ne,
        &[Value::scalar_f64(f64::NAN), Value::scalar_f64(f64::NAN)],
        &no_params()
    ));
}

#[test]
fn oracle_lt_le_gt_ge() {
    // lt
    assert!(eval_bool(
        Primitive::Lt,
        &[Value::scalar_i64(2), Value::scalar_i64(3)],
        &no_params()
    ));
    assert!(!eval_bool(
        Primitive::Lt,
        &[Value::scalar_i64(3), Value::scalar_i64(3)],
        &no_params()
    ));
    // le
    assert!(eval_bool(
        Primitive::Le,
        &[Value::scalar_i64(3), Value::scalar_i64(3)],
        &no_params()
    ));
    assert!(eval_bool(
        Primitive::Le,
        &[Value::scalar_i64(2), Value::scalar_i64(3)],
        &no_params()
    ));
    assert!(!eval_bool(
        Primitive::Le,
        &[Value::scalar_i64(4), Value::scalar_i64(3)],
        &no_params()
    ));
    // gt
    assert!(eval_bool(
        Primitive::Gt,
        &[Value::scalar_i64(4), Value::scalar_i64(3)],
        &no_params()
    ));
    assert!(!eval_bool(
        Primitive::Gt,
        &[Value::scalar_i64(3), Value::scalar_i64(3)],
        &no_params()
    ));
    // ge
    assert!(eval_bool(
        Primitive::Ge,
        &[Value::scalar_i64(3), Value::scalar_i64(3)],
        &no_params()
    ));
    assert!(eval_bool(
        Primitive::Ge,
        &[Value::scalar_i64(4), Value::scalar_i64(3)],
        &no_params()
    ));
    assert!(!eval_bool(
        Primitive::Ge,
        &[Value::scalar_i64(2), Value::scalar_i64(3)],
        &no_params()
    ));
}

#[test]
fn oracle_comparison_nan_always_false() {
    // NaN comparisons: everything except != is false
    let nan = Value::scalar_f64(f64::NAN);
    let one = Value::scalar_f64(1.0);
    assert!(!eval_bool(
        Primitive::Lt,
        &[nan.clone(), one.clone()],
        &no_params()
    ));
    assert!(!eval_bool(
        Primitive::Le,
        &[nan.clone(), one.clone()],
        &no_params()
    ));
    assert!(!eval_bool(
        Primitive::Gt,
        &[nan.clone(), one.clone()],
        &no_params()
    ));
    assert!(!eval_bool(
        Primitive::Ge,
        &[nan.clone(), one.clone()],
        &no_params()
    ));
    assert!(!eval_bool(
        Primitive::Eq,
        &[nan.clone(), one.clone()],
        &no_params()
    ));
    assert!(eval_bool(Primitive::Ne, &[nan.clone(), one], &no_params()));
}

// ======================== Oracle: Reductions ========================

#[test]
fn oracle_reduce_sum() {
    let input = Value::vector_i64(&[1, 2, 3, 4, 5]).unwrap();
    assert_eq!(eval_i64(Primitive::ReduceSum, &[input], &no_params()), 15);

    let f_input = Value::vector_f64(&[1.5, 2.5, 3.0]).unwrap();
    assert_f64_close(
        eval_f64(Primitive::ReduceSum, &[f_input], &no_params()),
        7.0,
        1e-14,
        "reduce_sum([1.5,2.5,3.0])",
    );
}

#[test]
fn oracle_reduce_max() {
    let input = Value::vector_i64(&[3, 1, 4, 1, 5, 9, 2, 6]).unwrap();
    assert_eq!(eval_i64(Primitive::ReduceMax, &[input], &no_params()), 9);
}

#[test]
fn oracle_reduce_min() {
    let input = Value::vector_i64(&[3, 1, 4, 1, 5, 9, 2, 6]).unwrap();
    assert_eq!(eval_i64(Primitive::ReduceMin, &[input], &no_params()), 1);
}

#[test]
fn oracle_reduce_prod() {
    let input = Value::vector_i64(&[1, 2, 3, 4, 5]).unwrap();
    assert_eq!(eval_i64(Primitive::ReduceProd, &[input], &no_params()), 120);
}

// ======================== Oracle: Shape Operations ========================

#[test]
fn oracle_reshape_6_to_2x3() {
    let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("new_shape".into(), "2,3".into());
    let out = eval_primitive(Primitive::Reshape, &[input], &params).unwrap();
    if let Value::Tensor(t) = &out {
        assert_eq!(t.shape.dims, vec![2, 3]);
        assert_eq!(t.elements.len(), 6);
        // Data unchanged, just shape
        assert_eq!(t.elements[0], Literal::I64(1));
        assert_eq!(t.elements[5], Literal::I64(6));
    } else {
        panic!("expected tensor");
    }
}

#[test]
fn oracle_transpose_2x3() {
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    let input = TensorValue::new(
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
    let out = eval_primitive(Primitive::Transpose, &[Value::Tensor(input)], &no_params()).unwrap();
    if let Value::Tensor(t) = &out {
        assert_eq!(t.shape.dims, vec![3, 2]);
        assert_eq!(t.elements[0], Literal::I64(1));
        assert_eq!(t.elements[1], Literal::I64(4));
        assert_eq!(t.elements[2], Literal::I64(2));
        assert_eq!(t.elements[3], Literal::I64(5));
    } else {
        panic!("expected tensor");
    }
}

#[test]
fn oracle_broadcast_scalar_to_3x2() {
    let mut params = BTreeMap::new();
    params.insert("shape".into(), "3,2".into());
    let out = eval_primitive(Primitive::BroadcastInDim, &[Value::scalar_i64(7)], &params).unwrap();
    if let Value::Tensor(t) = &out {
        assert_eq!(t.shape.dims, vec![3, 2]);
        assert_eq!(t.elements.len(), 6);
        for e in &t.elements {
            assert_eq!(*e, Literal::I64(7));
        }
    } else {
        panic!("expected tensor");
    }
}

#[test]
fn oracle_concatenate() {
    let a = Value::vector_i64(&[1, 2]).unwrap();
    let b = Value::vector_i64(&[3, 4, 5]).unwrap();
    let out = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    if let Value::Tensor(t) = &out {
        assert_eq!(t.shape.dims, vec![5]);
        let vals: Vec<i64> = t.elements.iter().map(|e| e.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![1, 2, 3, 4, 5]);
    } else {
        panic!("expected tensor");
    }
}

#[test]
fn oracle_slice() {
    let input = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("start_indices".into(), "1".into());
    params.insert("limit_indices".into(), "4".into());
    let out = eval_primitive(Primitive::Slice, &[input], &params).unwrap();
    if let Value::Tensor(t) = &out {
        let vals: Vec<i64> = t.elements.iter().map(|e| e.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![20, 30, 40]);
    } else {
        panic!("expected tensor");
    }
}

// ======================== Oracle: Dot ========================

#[test]
fn oracle_dot_i64_vectors() {
    let lhs = Value::vector_i64(&[1, 2, 3]).unwrap();
    let rhs = Value::vector_i64(&[4, 5, 6]).unwrap();
    assert_eq!(eval_i64(Primitive::Dot, &[lhs, rhs], &no_params()), 32); // 1*4+2*5+3*6
}

#[test]
fn oracle_dot_f64_vectors() {
    let lhs = Value::vector_f64(&[0.1, 0.2, 0.3]).unwrap();
    let rhs = Value::vector_f64(&[4.0, 5.0, 6.0]).unwrap();
    assert_f64_close(
        eval_f64(Primitive::Dot, &[lhs, rhs], &no_params()),
        3.2,
        1e-14,
        "dot([0.1,0.2,0.3],[4,5,6])",
    );
}

// ======================== Metamorphic: Identities ========================

#[test]
fn metamorphic_add_identity() {
    // add(x, 0) == x
    for x in [-100i64, -1, 0, 1, 42, i64::MAX, i64::MIN] {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(x), Value::scalar_i64(0)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_i64(x), "add({x}, 0) should be {x}");
    }
}

#[test]
fn metamorphic_mul_identity() {
    // mul(x, 1) == x
    for x in [-100i64, -1, 0, 1, 42] {
        let out = eval_primitive(
            Primitive::Mul,
            &[Value::scalar_i64(x), Value::scalar_i64(1)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_i64(x), "mul({x}, 1) should be {x}");
    }
}

#[test]
fn metamorphic_neg_involution() {
    // neg(neg(x)) == x
    for x in [-100i64, -1, 0, 1, 42] {
        let neg1 = eval_primitive(Primitive::Neg, &[Value::scalar_i64(x)], &no_params()).unwrap();
        let neg2 = eval_primitive(Primitive::Neg, &[neg1], &no_params()).unwrap();
        assert_eq!(neg2, Value::scalar_i64(x), "neg(neg({x})) should be {x}");
    }
}

#[test]
fn metamorphic_exp_log_inverse() {
    // exp(log(x)) ≈ x for x > 0 (relative tolerance for large values)
    for x in [0.1, 1.0, 2.7, 100.0, 1e-10, 1e10] {
        let log_x = eval_primitive(Primitive::Log, &[Value::scalar_f64(x)], &no_params()).unwrap();
        let exp_log_x = eval_f64(Primitive::Exp, &[log_x], &no_params());
        let tol = x.abs() * 1e-14 + 1e-14; // relative + absolute tolerance
        assert_f64_close(exp_log_x, x, tol, &format!("exp(log({x}))"));
    }
}

#[test]
fn metamorphic_transpose_involution_square() {
    // transpose(transpose(M)) == M for square matrices
    let input = TensorValue::new(
        DType::I64,
        Shape { dims: vec![3, 3] },
        (1..=9).map(Literal::I64).collect(),
    )
    .unwrap();
    let t1 = eval_primitive(
        Primitive::Transpose,
        &[Value::Tensor(input.clone())],
        &no_params(),
    )
    .unwrap();
    let t2 = eval_primitive(Primitive::Transpose, &[t1], &no_params()).unwrap();
    assert_eq!(t2, Value::Tensor(input));
}

#[test]
fn metamorphic_reduce_sum_all_equals_manual() {
    // reduce_sum(vec) == sum of all elements
    let elems = vec![3i64, 7, -2, 10, 0, -5];
    let expected: i64 = elems.iter().sum();
    let input = Value::vector_i64(&elems).unwrap();
    assert_eq!(
        eval_i64(Primitive::ReduceSum, &[input], &no_params()),
        expected
    );
}

#[test]
fn metamorphic_abs_idempotent() {
    // abs(abs(x)) == abs(x)
    for x in [-42i64, -1, 0, 1, 42] {
        let abs1 = eval_primitive(Primitive::Abs, &[Value::scalar_i64(x)], &no_params()).unwrap();
        let abs2 = eval_primitive(Primitive::Abs, std::slice::from_ref(&abs1), &no_params()).unwrap();
        assert_eq!(abs1, abs2, "abs(abs({x})) should be abs({x})");
    }
}

#[test]
fn metamorphic_max_min_agree_on_equal() {
    // max(x, x) == min(x, x) == x
    for x in [-5i64, 0, 42] {
        let max_v = eval_i64(
            Primitive::Max,
            &[Value::scalar_i64(x), Value::scalar_i64(x)],
            &no_params(),
        );
        let min_v = eval_i64(
            Primitive::Min,
            &[Value::scalar_i64(x), Value::scalar_i64(x)],
            &no_params(),
        );
        assert_eq!(max_v, x);
        assert_eq!(min_v, x);
    }
}

// ======================== Adversarial Cases ========================

#[test]
fn adversarial_reshape_incompatible_error() {
    let input = Value::vector_i64(&[1, 2, 3]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("new_shape".into(), "2,2".into());
    let err = eval_primitive(Primitive::Reshape, &[input], &params).unwrap_err();
    match err {
        EvalError::ShapeMismatch { .. } => {} // expected
        other => panic!("expected ShapeMismatch, got {other}"),
    }
}

#[test]
fn adversarial_broadcast_incompatible_shapes() {
    let a = Value::vector_i64(&[1, 2, 3]).unwrap();
    let b = Value::vector_i64(&[1, 2]).unwrap();
    let err = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap_err();
    assert!(matches!(err, EvalError::ShapeMismatch { .. }));
}

#[test]
fn adversarial_dot_dimension_mismatch() {
    let a = Value::vector_i64(&[1, 2, 3]).unwrap();
    let b = Value::vector_i64(&[1, 2]).unwrap();
    let err = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap_err();
    assert!(matches!(err, EvalError::ShapeMismatch { .. }));
}

#[test]
fn adversarial_arity_too_many_binary() {
    let err = eval_primitive(
        Primitive::Add,
        &[
            Value::scalar_i64(1),
            Value::scalar_i64(2),
            Value::scalar_i64(3),
        ],
        &no_params(),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvalError::ArityMismatch {
            expected: 2,
            actual: 3,
            ..
        }
    ));
}

#[test]
fn adversarial_arity_zero_unary() {
    let err = eval_primitive(Primitive::Neg, &[], &no_params()).unwrap_err();
    assert!(matches!(
        err,
        EvalError::ArityMismatch {
            expected: 1,
            actual: 0,
            ..
        }
    ));
}

#[test]
fn adversarial_gather_returns_unsupported() {
    let err = eval_primitive(Primitive::Gather, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
    assert!(matches!(err, EvalError::Unsupported { .. }));
}

#[test]
fn adversarial_scatter_returns_unsupported() {
    let err =
        eval_primitive(Primitive::Scatter, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
    assert!(matches!(err, EvalError::Unsupported { .. }));
}

#[test]
fn adversarial_slice_inverted_range() {
    let input = Value::vector_i64(&[10, 20, 30]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("start_indices".into(), "2".into());
    params.insert("limit_indices".into(), "1".into()); // start > limit
    let err = eval_primitive(Primitive::Slice, &[input], &params).unwrap_err();
    assert!(matches!(err, EvalError::Unsupported { .. }));
}

#[test]
fn adversarial_large_tensor_no_stack_overflow() {
    // 10K elements should work fine (no stack overflow)
    let big: Vec<i64> = (0..10_000).collect();
    let input = Value::vector_i64(&big).unwrap();
    let out = eval_primitive(Primitive::ReduceSum, &[input], &no_params()).unwrap();
    // Sum of 0..10000 = 10000*9999/2 = 49995000
    assert_eq!(
        eval_i64(Primitive::ReduceSum, &[Value::scalar_i64(0)], &no_params()),
        0
    ); // sanity
    if let Value::Scalar(Literal::I64(v)) = out {
        assert_eq!(v, 49_995_000);
    } else {
        panic!("expected i64 scalar from large reduction");
    }
}

#[test]
fn adversarial_concatenate_empty_error() {
    let err = eval_primitive(Primitive::Concatenate, &[], &no_params()).unwrap_err();
    assert!(matches!(err, EvalError::ArityMismatch { .. }));
}

#[test]
fn adversarial_reshape_two_inferred_dims_error() {
    let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
    let mut params = BTreeMap::new();
    params.insert("new_shape".into(), "-1,-1".into());
    let err = eval_primitive(Primitive::Reshape, &[input], &params).unwrap_err();
    assert!(matches!(err, EvalError::Unsupported { .. }));
}
