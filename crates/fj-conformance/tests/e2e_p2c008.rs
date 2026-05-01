//! FJ-P2C-008 E2E Scenario Scripts: LAX Primitive First Wave
//!
//! End-to-end scenarios testing LAX primitive evaluation across scalar,
//! tensor, composition, edge cases, and error paths.

use fj_ad::grad_jaxpr;
use fj_core::{
    CompatibilityMode, DType, Literal, Primitive, ProgramSpec, Shape, TensorValue,
    TraceTransformLedger, Transform, Value, build_program,
};
use fj_dispatch::{DispatchRequest, dispatch};
use fj_lax::{EvalError, eval_primitive};
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn tensor_f64(shape: &[u32], values: &[f64]) -> Value {
    let elements: Vec<Literal> = values.iter().copied().map(Literal::from_f64).collect();
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: shape.to_vec(),
            },
            elements,
        )
        .expect("tensor_f64 should be well-formed"),
    )
}

fn tensor_i64(shape: &[u32], values: &[i64]) -> Value {
    let elements: Vec<Literal> = values.iter().copied().map(Literal::I64).collect();
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: shape.to_vec(),
            },
            elements,
        )
        .expect("tensor_i64 should be well-formed"),
    )
}

fn tensor_bool(shape: &[u32], values: &[bool]) -> Value {
    let elements: Vec<Literal> = values.iter().copied().map(Literal::Bool).collect();
    Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape {
                dims: shape.to_vec(),
            },
            elements,
        )
        .expect("tensor_bool should be well-formed"),
    )
}

fn to_f64(value: &Value, context: &str) -> Result<f64, String> {
    value
        .as_f64_scalar()
        .ok_or_else(|| format!("{context}: expected scalar f64 output"))
}

fn to_i64(value: &Value, context: &str) -> Result<i64, String> {
    value
        .as_i64_scalar()
        .ok_or_else(|| format!("{context}: expected scalar i64 output"))
}

fn to_f64_vec(value: &Value, context: &str) -> Result<Vec<f64>, String> {
    value
        .as_tensor()
        .ok_or_else(|| format!("{context}: expected tensor output"))?
        .to_f64_vec()
        .ok_or_else(|| format!("{context}: expected f64 tensor elements"))
}

fn to_i64_vec(value: &Value, context: &str) -> Result<Vec<i64>, String> {
    value
        .as_tensor()
        .ok_or_else(|| format!("{context}: expected tensor output"))?
        .elements
        .iter()
        .map(|lit| {
            lit.as_i64()
                .ok_or_else(|| format!("{context}: expected i64 tensor elements"))
        })
        .collect()
}

fn to_bool_vec(value: &Value, context: &str) -> Result<Vec<bool>, String> {
    value
        .as_tensor()
        .ok_or_else(|| format!("{context}: expected tensor output"))?
        .elements
        .iter()
        .map(|lit| match lit {
            Literal::Bool(v) => Ok(*v),
            _ => Err(format!("{context}: expected bool tensor elements")),
        })
        .collect()
}

fn dispatch_with_transforms(
    program: ProgramSpec,
    transforms: &[Transform],
    args: Vec<Value>,
) -> Result<Value, String> {
    let mut ttl = TraceTransformLedger::new(build_program(program));
    for (index, transform) in transforms.iter().enumerate() {
        ttl.push_transform(
            *transform,
            format!("adversarial-{}-{index}", transform.as_str()),
        );
    }

    let response = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ttl,
        args,
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .map_err(|err| err.to_string())?;

    response
        .outputs
        .into_iter()
        .next()
        .ok_or_else(|| "dispatch returned no outputs".to_owned())
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

    // Wrong arity for gather (needs 2 inputs)
    let err = eval_primitive(Primitive::Gather, &[Value::scalar_i64(1)], &p).unwrap_err();
    assert!(matches!(err, EvalError::ArityMismatch { .. }));
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

// ======================== Scenario 7: Adversarial Fixture Family ========================

#[test]
fn test_adversarial_nan_input() {
    let p = no_params();

    let nan_add = eval_primitive(
        Primitive::Add,
        &[Value::scalar_f64(f64::NAN), Value::scalar_f64(1.0)],
        &p,
    )
    .expect("add(NaN, 1.0) should evaluate");
    assert!(
        nan_add
            .as_f64_scalar()
            .expect("add output should be scalar f64")
            .is_nan(),
        "NaN should propagate through add"
    );

    let grads = grad_jaxpr(
        &build_program(ProgramSpec::Square),
        &[Value::scalar_f64(f64::NAN)],
    )
    .expect("grad(square)(NaN) should evaluate");
    let grad = grads[0]
        .as_f64_scalar()
        .expect("gradient should be scalar f64");
    assert!(grad.is_nan(), "grad(square)(NaN) should be NaN");
}

#[test]
fn test_adversarial_inf_input() {
    let p = no_params();

    let inf_mul_zero = eval_primitive(
        Primitive::Mul,
        &[Value::scalar_f64(f64::INFINITY), Value::scalar_f64(0.0)],
        &p,
    )
    .expect("mul(Inf, 0.0) should evaluate");
    assert!(
        inf_mul_zero
            .as_f64_scalar()
            .expect("mul output should be scalar f64")
            .is_nan(),
        "Inf * 0.0 should be NaN"
    );

    let exp_inf = eval_primitive(Primitive::Exp, &[Value::scalar_f64(f64::INFINITY)], &p)
        .expect("exp(Inf) should evaluate");
    assert!(
        exp_inf
            .as_f64_scalar()
            .expect("exp output should be scalar f64")
            .is_infinite(),
        "exp(Inf) should be infinite"
    );
}

#[test]
fn test_adversarial_zero_division() {
    let p = no_params();
    let out = eval_primitive(
        Primitive::Div,
        &[Value::scalar_f64(1.0), Value::scalar_f64(0.0)],
        &p,
    )
    .expect("1.0 / 0.0 should evaluate");
    let v = out
        .as_f64_scalar()
        .expect("division output should be scalar f64");
    assert!(
        v.is_infinite() && v.is_sign_positive(),
        "1.0 / 0.0 should be +Inf, got {v}"
    );
}

#[test]
fn test_adversarial_empty_array() {
    let p = no_params();
    let empty = tensor_f64(&[0], &[]);

    let reduced = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&empty), &p)
        .expect("reduce_sum on empty should evaluate");
    assert_eq!(reduced, Value::scalar_f64(0.0));

    let added = eval_primitive(Primitive::Add, &[empty.clone(), empty], &p)
        .expect("add on empty tensors should evaluate");
    let tensor = added.as_tensor().expect("add output should be tensor");
    assert_eq!(tensor.shape.dims, vec![0]);
    assert!(
        tensor.elements.is_empty(),
        "empty add output should stay empty"
    );
}

#[test]
fn test_adversarial_huge_tensor() {
    let len: usize = 1_048_576;
    let shape_len = u32::try_from(len).expect("huge test length should fit in u32");
    let values = vec![1_i64; len];
    let huge = tensor_i64(&[shape_len], &values);

    let reduced = eval_primitive(Primitive::ReduceSum, &[huge], &no_params())
        .expect("reduce_sum on huge tensor should evaluate");
    assert_eq!(
        reduced,
        Value::scalar_i64(i64::try_from(len).expect("huge test length should fit in i64"))
    );
}

#[test]
fn test_adversarial_denormal_floats() {
    let p = no_params();
    let subnormal = f64::MIN_POSITIVE / 2.0;
    assert!(
        subnormal > 0.0 && subnormal < f64::MIN_POSITIVE,
        "constructed value should be subnormal"
    );

    let sum = eval_primitive(
        Primitive::Add,
        &[Value::scalar_f64(subnormal), Value::scalar_f64(0.0)],
        &p,
    )
    .expect("add(subnormal, 0.0) should evaluate");
    let sum_v = sum
        .as_f64_scalar()
        .expect("sum output should be scalar f64");
    assert!(
        sum_v.is_finite() && sum_v > 0.0,
        "subnormal should survive arithmetic, got {sum_v}"
    );

    let doubled = eval_primitive(
        Primitive::Mul,
        &[Value::scalar_f64(subnormal), Value::scalar_f64(2.0)],
        &p,
    )
    .expect("mul(subnormal, 2.0) should evaluate");
    let doubled_v = doubled
        .as_f64_scalar()
        .expect("doubled output should be scalar f64");
    assert!(
        doubled_v.is_finite() && doubled_v > 0.0,
        "doubled subnormal should remain finite positive, got {doubled_v}"
    );
}

#[test]
fn test_adversarial_max_rank() {
    let rank8 = tensor_f64(&[1, 1, 1, 1, 1, 1, 1, 1], &[3.0]);
    let out = eval_primitive(Primitive::Neg, &[rank8], &no_params())
        .expect("neg on rank-8 tensor should evaluate");

    let tensor = out.as_tensor().expect("rank-8 output should be tensor");
    assert_eq!(tensor.shape.dims.len(), 8, "output rank should remain 8");
    assert_eq!(
        tensor.elements,
        vec![Literal::from_f64(-3.0)],
        "rank-8 element should be negated"
    );
}

#[test]
fn test_adversarial_single_element() {
    let single = Value::vector_i64(&[41]).expect("single-element vector should build");
    let out = eval_primitive(
        Primitive::Add,
        &[single, Value::scalar_i64(1)],
        &no_params(),
    )
    .expect("single-element vector add should evaluate");
    assert_eq!(
        out,
        Value::vector_i64(&[42]).expect("single-element expected vector should build")
    );
}

#[test]
fn test_edge_case_negative_zero() {
    let p = no_params();

    let neg_zero = eval_primitive(Primitive::Neg, &[Value::scalar_f64(0.0)], &p)
        .expect("neg(0.0) should evaluate")
        .as_f64_scalar()
        .expect("neg output should be scalar f64");
    assert!(
        neg_zero == 0.0 && neg_zero.is_sign_negative(),
        "neg(0.0) should preserve negative zero sign bit"
    );

    let abs_neg_zero = eval_primitive(Primitive::Abs, &[Value::scalar_f64(-0.0)], &p)
        .expect("abs(-0.0) should evaluate")
        .as_f64_scalar()
        .expect("abs output should be scalar f64");
    assert!(
        abs_neg_zero == 0.0 && abs_neg_zero.is_sign_positive(),
        "abs(-0.0) should be +0.0"
    );
}

#[test]
fn test_edge_case_integer_overflow() {
    let p = no_params();

    let shifted_left = eval_primitive(
        Primitive::ShiftLeft,
        &[Value::scalar_i64(i64::MAX), Value::scalar_i64(1)],
        &p,
    )
    .expect("shift_left on max i64 should evaluate");
    assert_eq!(
        shifted_left,
        Value::scalar_i64(i64::MAX.wrapping_shl(1)),
        "shift-left should use wrapping semantics"
    );

    let shifted_right = eval_primitive(
        Primitive::ShiftRightArithmetic,
        &[Value::scalar_i64(i64::MIN), Value::scalar_i64(1)],
        &p,
    )
    .expect("shift_right_arithmetic on min i64 should evaluate");
    assert_eq!(
        shifted_right,
        Value::scalar_i64(i64::MIN.wrapping_shr(1)),
        "shift-right should follow wrapping/arithmetic semantics"
    );
}

#[test]
fn e2e_adversarial_fixture_suite() {
    let p = no_params();
    let mut entries = Vec::<serde_json::Value>::new();
    let mut log_entries = Vec::<serde_json::Value>::new();
    let mut total_cases = 0_usize;
    let mut passed_cases = 0_usize;

    macro_rules! record_case {
        ($fixture_id:expr, $edge_case_type:expr, $input_description:expr, $expected_behavior:expr, $actual_behavior:expr, $survived:expr, $error_type:expr, $oracle_match:expr, $pass:expr $(,)?) => {{
            total_cases += 1;
            if $pass {
                passed_cases += 1;
            }
            entries.push(json!({
                "fixture_id": $fixture_id,
                "edge_case_type": $edge_case_type,
                "survived": $survived,
                "error_type": $error_type,
                "oracle_match": $oracle_match,
                "pass": $pass
            }));
            log_entries.push(json!({
                "test_name": "e2e_adversarial_fixture_suite",
                "edge_case_type": $edge_case_type,
                "input_description": $input_description,
                "expected_behavior": $expected_behavior,
                "actual_behavior": $actual_behavior,
                "pass": $pass
            }));
        }};
    }

    let mut record_check = |fixture_id: &str,
                            edge_case_type: &str,
                            input_description: &str,
                            expected_behavior: &str,
                            outcome: Result<bool, String>| {
        match outcome {
            Ok(matched) => {
                let actual_behavior = if matched {
                    "matched expected behavior".to_owned()
                } else {
                    "behavior mismatch".to_owned()
                };
                record_case!(
                    fixture_id,
                    edge_case_type,
                    input_description,
                    expected_behavior,
                    actual_behavior,
                    true,
                    Option::<String>::None,
                    matched,
                    matched
                );
            }
            Err(err) => {
                record_case!(
                    fixture_id,
                    edge_case_type,
                    input_description,
                    expected_behavior,
                    err,
                    false,
                    Some("evaluation_error".to_owned()),
                    false,
                    false
                );
            }
        }
    };

    // ── NaN / Inf propagation (12 cases) ─────────────────────────────
    for (case_suffix, primitive) in [
        ("nan_add", Primitive::Add),
        ("nan_sub", Primitive::Sub),
        ("nan_mul", Primitive::Mul),
        ("nan_div", Primitive::Div),
    ] {
        let fixture_id = format!("adversarial_{case_suffix}");
        let outcome = eval_primitive(
            primitive,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(1.0)],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| to_f64(&value, &fixture_id).map(|actual| actual.is_nan()));
        record_check(
            &fixture_id,
            "nan_inf",
            "binary op with NaN and finite operand",
            "result should be NaN",
            outcome,
        );
    }

    record_check(
        "adversarial_inf_mul_zero",
        "nan_inf",
        "mul(Inf, 0.0)",
        "result should be NaN",
        eval_primitive(
            Primitive::Mul,
            &[Value::scalar_f64(f64::INFINITY), Value::scalar_f64(0.0)],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| to_f64(&value, "adversarial_inf_mul_zero").map(|actual| actual.is_nan())),
    );

    record_check(
        "adversarial_inf_add_finite",
        "nan_inf",
        "add(Inf, 1.0)",
        "result should remain +Inf",
        eval_primitive(
            Primitive::Add,
            &[Value::scalar_f64(f64::INFINITY), Value::scalar_f64(1.0)],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_f64(&value, "adversarial_inf_add_finite")
                .map(|actual| actual.is_infinite() && actual.is_sign_positive())
        }),
    );

    record_check(
        "adversarial_exp_inf",
        "nan_inf",
        "exp(Inf)",
        "result should be +Inf",
        eval_primitive(Primitive::Exp, &[Value::scalar_f64(f64::INFINITY)], &p)
            .map_err(|err| err.to_string())
            .and_then(|value| {
                to_f64(&value, "adversarial_exp_inf")
                    .map(|actual| actual.is_infinite() && actual.is_sign_positive())
            }),
    );

    record_check(
        "adversarial_exp_neg_inf",
        "nan_inf",
        "exp(-Inf)",
        "result should be +0.0",
        eval_primitive(Primitive::Exp, &[Value::scalar_f64(f64::NEG_INFINITY)], &p)
            .map_err(|err| err.to_string())
            .and_then(|value| {
                to_f64(&value, "adversarial_exp_neg_inf")
                    .map(|actual| actual == 0.0 && !actual.is_sign_negative())
            }),
    );

    record_check(
        "adversarial_log_zero",
        "nan_inf",
        "log(0.0)",
        "result should be -Inf",
        eval_primitive(Primitive::Log, &[Value::scalar_f64(0.0)], &p)
            .map_err(|err| err.to_string())
            .and_then(|value| {
                to_f64(&value, "adversarial_log_zero")
                    .map(|actual| actual.is_infinite() && actual.is_sign_negative())
            }),
    );

    record_check(
        "adversarial_log_negative",
        "nan_inf",
        "log(-1.0)",
        "result should be NaN",
        eval_primitive(Primitive::Log, &[Value::scalar_f64(-1.0)], &p)
            .map_err(|err| err.to_string())
            .and_then(|value| {
                to_f64(&value, "adversarial_log_negative").map(|actual| actual.is_nan())
            }),
    );

    record_check(
        "adversarial_sqrt_negative",
        "nan_inf",
        "sqrt(-1.0)",
        "result should be NaN",
        eval_primitive(Primitive::Sqrt, &[Value::scalar_f64(-1.0)], &p)
            .map_err(|err| err.to_string())
            .and_then(|value| {
                to_f64(&value, "adversarial_sqrt_negative").map(|actual| actual.is_nan())
            }),
    );

    record_check(
        "adversarial_reciprocal_zero",
        "nan_inf",
        "reciprocal(0.0)",
        "result should be +Inf",
        eval_primitive(Primitive::Reciprocal, &[Value::scalar_f64(0.0)], &p)
            .map_err(|err| err.to_string())
            .and_then(|value| {
                to_f64(&value, "adversarial_reciprocal_zero")
                    .map(|actual| actual.is_infinite() && actual.is_sign_positive())
            }),
    );

    // ── Shape / rank edge cases (11 cases) ───────────────────────────
    record_check(
        "adversarial_zero_size_reduce_sum",
        "shape_edge",
        "reduce_sum on tensor shape [0]",
        "result should be scalar zero",
        eval_primitive(Primitive::ReduceSum, &[tensor_f64(&[0], &[])], &p)
            .map_err(|err| err.to_string())
            .and_then(|value| {
                to_f64(&value, "adversarial_zero_size_reduce_sum").map(|actual| actual == 0.0)
            }),
    );

    record_check(
        "adversarial_zero_size_add",
        "shape_edge",
        "add two tensors shape [0]",
        "result should preserve empty shape",
        eval_primitive(
            Primitive::Add,
            &[tensor_f64(&[0], &[]), tensor_f64(&[0], &[])],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            value
                .as_tensor()
                .map(|tensor| tensor.shape.dims == vec![0] && tensor.elements.is_empty())
                .ok_or_else(|| "adversarial_zero_size_add: expected tensor output".to_owned())
        }),
    );

    record_check(
        "adversarial_rank8_neg",
        "shape_edge",
        "neg on rank-8 tensor",
        "rank and element count should remain stable",
        eval_primitive(
            Primitive::Neg,
            &[tensor_f64(&[1, 1, 1, 1, 1, 1, 1, 1], &[2.0])],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            value
                .as_tensor()
                .map(|tensor| tensor.shape.dims.len() == 8 && tensor.elements.len() == 1)
                .ok_or_else(|| "adversarial_rank8_neg: expected tensor output".to_owned())
        }),
    );

    record_check(
        "adversarial_scalar_tensor_add",
        "shape_edge",
        "scalar + tensor broadcast",
        "broadcasted add should match expected values",
        eval_primitive(
            Primitive::Add,
            &[
                Value::scalar_i64(5),
                Value::vector_i64(&[10, 20, 30]).expect("vector should build"),
            ],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_i64_vec(&value, "adversarial_scalar_tensor_add")
                .map(|actual| actual == vec![15, 25, 35])
        }),
    );

    record_check(
        "adversarial_tensor_scalar_mul",
        "shape_edge",
        "tensor * scalar broadcast",
        "broadcasted mul should match expected values",
        eval_primitive(
            Primitive::Mul,
            &[
                Value::vector_i64(&[1, 2, 3]).expect("vector should build"),
                Value::scalar_i64(10),
            ],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_i64_vec(&value, "adversarial_tensor_scalar_mul")
                .map(|actual| actual == vec![10, 20, 30])
        }),
    );

    for len in [1_usize, 8, 64, 256, 4096, 10_000] {
        let fixture_id = format!("adversarial_large_dim_reduce_sum_{len}");
        let len_u32 = u32::try_from(len).expect("large-dim test length should fit in u32");
        let values = vec![1_i64; len];
        let len_i64 = i64::try_from(len).expect("large-dim test length should fit in i64");
        let outcome = eval_primitive(Primitive::ReduceSum, &[tensor_i64(&[len_u32], &values)], &p)
            .map_err(|err| err.to_string())
            .and_then(|value| to_i64(&value, &fixture_id).map(|actual| actual == len_i64));
        record_check(
            &fixture_id,
            "shape_edge",
            "reduce_sum on large 1D tensor",
            "sum should equal tensor length for all-ones input",
            outcome,
        );
    }

    // ── Transform edge cases (8 cases) ───────────────────────────────
    for (label, x) in [
        ("adversarial_grad_floor_neg", -3.5),
        ("adversarial_grad_floor_zero", 0.0),
        ("adversarial_grad_floor_pos", 2.75),
    ] {
        let outcome = grad_jaxpr(
            &build_program(ProgramSpec::LaxFloor),
            &[Value::scalar_f64(x)],
        )
        .map_err(|err| err.to_string())
        .and_then(|grads| {
            grads
                .first()
                .ok_or_else(|| format!("{label}: expected one gradient output"))
                .and_then(|grad| to_f64(grad, label).map(|actual| actual.abs() <= 1e-12))
        });
        record_check(
            label,
            "transform_edge",
            "grad(floor(x))",
            "gradient should be zero for non-differentiable floor primitive",
            outcome,
        );
    }

    record_check(
        "adversarial_vmap_mismatched_batch",
        "transform_edge",
        "vmap(add2) with mismatched leading dims",
        "transform execution should return mismatched-dimension error",
        Ok(dispatch_with_transforms(
            ProgramSpec::Add2,
            &[Transform::Vmap],
            vec![
                Value::vector_i64(&[1, 2, 3]).expect("lhs vector should build"),
                Value::vector_i64(&[1, 2]).expect("rhs vector should build"),
            ],
        )
        .err()
        .is_some_and(|err| err.contains("vmap leading-dimension mismatch"))),
    );

    record_check(
        "adversarial_vmap_scalar_input",
        "transform_edge",
        "vmap(add_one) with scalar argument",
        "transform execution should reject rank-0 mapped input",
        Ok(dispatch_with_transforms(
            ProgramSpec::AddOne,
            &[Transform::Vmap],
            vec![Value::scalar_i64(7)],
        )
        .err()
        .is_some_and(|err| err.contains("rank >= 1"))),
    );

    record_check(
        "adversarial_double_grad_square",
        "transform_edge",
        "grad(grad(square))(3.0)",
        "second derivative should be approximately 2.0",
        dispatch_with_transforms(
            ProgramSpec::Square,
            &[Transform::Grad, Transform::Grad],
            vec![Value::scalar_f64(3.0)],
        )
        .and_then(|value| {
            to_f64(&value, "adversarial_double_grad_square")
                .map(|actual| (actual - 2.0).abs() <= 1e-3)
        }),
    );

    record_check(
        "adversarial_jit_of_jit",
        "transform_edge",
        "jit(jit(add_one))(41)",
        "double jit should preserve semantics",
        dispatch_with_transforms(
            ProgramSpec::AddOne,
            &[Transform::Jit, Transform::Jit],
            vec![Value::scalar_i64(41)],
        )
        .and_then(|value| to_i64(&value, "adversarial_jit_of_jit").map(|actual| actual == 42)),
    );

    record_check(
        "adversarial_jit_grad_composition",
        "transform_edge",
        "jit(grad(square))(4.0)",
        "composition should produce derivative 8.0",
        dispatch_with_transforms(
            ProgramSpec::Square,
            &[Transform::Jit, Transform::Grad],
            vec![Value::scalar_f64(4.0)],
        )
        .and_then(|value| {
            to_f64(&value, "adversarial_jit_grad_composition")
                .map(|actual| (actual - 8.0).abs() <= 1e-9)
        }),
    );

    // ── Type-coercion edge cases (10 cases) ──────────────────────────
    record_check(
        "adversarial_add_i64_f64_scalar",
        "type_coercion",
        "add(i64, f64)",
        "result should promote to f64 with correct value",
        eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(2), Value::scalar_f64(0.5)],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_f64(&value, "adversarial_add_i64_f64_scalar")
                .map(|actual| (actual - 2.5).abs() <= 1e-12)
        }),
    );

    record_check(
        "adversarial_add_i64_f64_vector",
        "type_coercion",
        "add(vector_i64, vector_f64)",
        "result should promote to f64 vector",
        eval_primitive(
            Primitive::Add,
            &[
                Value::vector_i64(&[1, 2, 3]).expect("lhs vector should build"),
                Value::vector_f64(&[0.5, 1.5, -1.0]).expect("rhs vector should build"),
            ],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_f64_vec(&value, "adversarial_add_i64_f64_vector")
                .map(|actual| actual == vec![1.5, 3.5, 2.0])
        }),
    );

    record_check(
        "adversarial_mul_bool_i64_vector",
        "type_coercion",
        "mul(vector_bool, vector_i64)",
        "bool should coerce through multiplication semantics",
        eval_primitive(
            Primitive::Mul,
            &[
                tensor_bool(&[4], &[true, false, true, false]),
                Value::vector_i64(&[10, 20, -3, 1]).expect("vector should build"),
            ],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_i64_vec(&value, "adversarial_mul_bool_i64_vector")
                .map(|actual| actual == vec![10, 0, -3, 0])
        }),
    );

    record_check(
        "adversarial_mul_bool_i64_scalar",
        "type_coercion",
        "mul(bool, i64)",
        "bool scalar should coerce as expected",
        eval_primitive(
            Primitive::Mul,
            &[Value::scalar_bool(true), Value::scalar_i64(9)],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_i64(&value, "adversarial_mul_bool_i64_scalar").map(|actual| actual == 9)
        }),
    );

    record_check(
        "adversarial_div_i64_f64_scalar",
        "type_coercion",
        "div(i64, f64)",
        "result should promote to f64",
        eval_primitive(
            Primitive::Div,
            &[Value::scalar_i64(9), Value::scalar_f64(2.0)],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_f64(&value, "adversarial_div_i64_f64_scalar")
                .map(|actual| (actual - 4.5).abs() <= 1e-12)
        }),
    );

    record_check(
        "adversarial_div_i64_f64_vector",
        "type_coercion",
        "div(vector_i64, scalar_f64)",
        "result should promote to f64 vector",
        eval_primitive(
            Primitive::Div,
            &[
                Value::vector_i64(&[9, 3, 1]).expect("vector should build"),
                Value::scalar_f64(2.0),
            ],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_f64_vec(&value, "adversarial_div_i64_f64_vector")
                .map(|actual| actual == vec![4.5, 1.5, 0.5])
        }),
    );

    record_check(
        "adversarial_gt_f64_i64_vector",
        "type_coercion",
        "gt(vector_f64, vector_i64)",
        "comparison should produce bool tensor",
        eval_primitive(
            Primitive::Gt,
            &[
                Value::vector_f64(&[0.5, 2.5, 3.0, -1.0]).expect("lhs vector should build"),
                Value::vector_i64(&[0, 3, 2, 0]).expect("rhs vector should build"),
            ],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_bool_vec(&value, "adversarial_gt_f64_i64_vector")
                .map(|actual| actual == vec![true, false, true, false])
        }),
    );

    record_check(
        "adversarial_eq_i64_f64_scalar",
        "type_coercion",
        "eq(i64, f64)",
        "comparison should succeed across numeric dtypes",
        eval_primitive(
            Primitive::Eq,
            &[Value::scalar_i64(7), Value::scalar_f64(7.0)],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            value
                .as_bool_scalar()
                .ok_or_else(|| "adversarial_eq_i64_f64_scalar: expected bool scalar".to_owned())
        }),
    );

    record_check(
        "adversarial_shift_left_wrapping",
        "type_coercion",
        "shift_left(i64::MAX, 1)",
        "operation should use wrapping semantics at integer boundary",
        eval_primitive(
            Primitive::ShiftLeft,
            &[Value::scalar_i64(i64::MAX), Value::scalar_i64(1)],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_i64(&value, "adversarial_shift_left_wrapping")
                .map(|actual| actual == i64::MAX.wrapping_shl(1))
        }),
    );

    record_check(
        "adversarial_shift_right_arithmetic_wrapping",
        "type_coercion",
        "shift_right_arithmetic(i64::MIN, 1)",
        "operation should preserve arithmetic/right-shift semantics at boundary",
        eval_primitive(
            Primitive::ShiftRightArithmetic,
            &[Value::scalar_i64(i64::MIN), Value::scalar_i64(1)],
            &p,
        )
        .map_err(|err| err.to_string())
        .and_then(|value| {
            to_i64(&value, "adversarial_shift_right_arithmetic_wrapping")
                .map(|actual| actual == i64::MIN.wrapping_shr(1))
        }),
    );

    assert!(
        total_cases >= 40,
        "expected >=40 adversarial fixture cases, got {total_cases}"
    );
    assert_eq!(
        passed_cases, total_cases,
        "all adversarial fixture cases should pass"
    );

    let e2e_payload = json!({
        "test_name": "e2e_adversarial_fixture_suite",
        "generated_at_unix_ms": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_millis(),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases.saturating_sub(passed_cases),
        "entries": entries
    });
    let e2e_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/e2e/e2e_adversarial_fixtures.e2e.json");
    fs::create_dir_all(
        e2e_path
            .parent()
            .expect("e2e adversarial artifact parent should exist"),
    )
    .expect("should create artifacts/e2e");
    fs::write(
        &e2e_path,
        serde_json::to_string_pretty(&e2e_payload).expect("serialize adversarial e2e payload"),
    )
    .expect("write adversarial e2e artifact");

    let log_payload = json!({
        "test_name": "e2e_adversarial_fixture_suite",
        "generated_at_unix_ms": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_millis(),
        "entries": log_entries
    });
    let log_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/testing/logs/fj-conformance/e2e_adversarial_fixture_suite.json");
    fs::create_dir_all(
        log_path
            .parent()
            .expect("adversarial fixture log parent should exist"),
    )
    .expect("should create artifacts/testing/logs/fj-conformance");
    fs::write(
        &log_path,
        serde_json::to_string_pretty(&log_payload).expect("serialize adversarial fixture log"),
    )
    .expect("write adversarial fixture log");

    // Emit machine-readable payloads for remote test execution environments where
    // untracked artifact files are not synchronized back to the caller.
    println!("__FJ_ADVERSARIAL_E2E_JSON_BEGIN__");
    println!(
        "{}",
        serde_json::to_string(&e2e_payload).expect("serialize adversarial e2e payload compact")
    );
    println!("__FJ_ADVERSARIAL_E2E_JSON_END__");
    println!("__FJ_ADVERSARIAL_LOG_JSON_BEGIN__");
    println!(
        "{}",
        serde_json::to_string(&log_payload).expect("serialize adversarial log payload compact")
    );
    println!("__FJ_ADVERSARIAL_LOG_JSON_END__");
}
