//! Oracle-backed dtype promotion parity tests.
//!
//! Validates that FrankenJAX type promotion rules match JAX's behavior
//! by performing binary operations on typed values and checking the result dtype.

use fj_core::{DType, Literal, Primitive, Value};
use fj_lax::eval_primitive;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Deserialize)]
struct DtypeBundle {
    cases: Vec<DtypeCase>,
}

#[derive(Deserialize)]
struct DtypeCase {
    case_id: String,
    operation: String,
    lhs_dtype: String,
    rhs_dtype: String,
    result_dtype: Option<String>,
    #[allow(dead_code)]
    result_value: Option<serde_json::Value>,
    #[allow(dead_code)]
    error: Option<String>,
}

fn dtype_from_name(name: &str) -> DType {
    match name {
        "bool" => DType::Bool,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "u32" => DType::U32,
        "u64" => DType::U64,
        "f16" => DType::F16,
        "f32" => DType::F32,
        "f64" => DType::F64,
        "bf16" => DType::BF16,
        _ => panic!("unknown dtype name: {name}"),
    }
}

fn jax_dtype_to_fj(jax_name: &str) -> DType {
    match jax_name {
        "bool" => DType::Bool,
        "int32" => DType::I32,
        "int64" => DType::I64,
        "uint32" => DType::U32,
        "uint64" => DType::U64,
        "float16" => DType::F16,
        "float32" => DType::F32,
        "float64" => DType::F64,
        "bfloat16" => DType::BF16,
        _ => panic!("unknown JAX dtype: {jax_name}"),
    }
}

fn make_typed_value(dtype: DType) -> Value {
    match dtype {
        DType::Bool => Value::Scalar(Literal::Bool(true)),
        DType::I32 => Value::Scalar(Literal::I64(7)), // I32 stored as I64 internally
        DType::I64 => Value::Scalar(Literal::I64(7)),
        DType::U32 => Value::Scalar(Literal::U32(7)),
        DType::U64 => Value::Scalar(Literal::U64(7)),
        DType::F16 => Value::Scalar(Literal::F16Bits(0x4100)), // f16 2.5 = 0x4100
        DType::F32 => Value::Scalar(Literal::from_f64(2.5)),   // stored as f64 bits internally
        DType::F64 => Value::Scalar(Literal::from_f64(2.5)),
        DType::BF16 => Value::Scalar(Literal::BF16Bits(0x4020)), // bf16 2.5 = 0x4020
        _ => panic!("unsupported dtype for test value: {dtype:?}"),
    }
}

fn make_typed_tensor(dtype: DType) -> Value {
    use fj_core::{Shape, TensorValue};
    let elements = match dtype {
        DType::Bool => vec![Literal::Bool(true)],
        DType::I32 => vec![Literal::I64(7)],
        DType::I64 => vec![Literal::I64(7)],
        DType::U32 => vec![Literal::U32(7)],
        DType::U64 => vec![Literal::U64(7)],
        DType::F16 => vec![Literal::F16Bits(0x4100)],
        DType::F32 => vec![Literal::from_f64(2.5)],
        DType::F64 => vec![Literal::from_f64(2.5)],
        DType::BF16 => vec![Literal::BF16Bits(0x4020)],
        _ => panic!("unsupported dtype for tensor test: {dtype:?}"),
    };
    Value::Tensor(
        TensorValue::new(dtype, Shape { dims: vec![1] }, elements)
            .expect("test tensor should be valid"),
    )
}

fn result_dtype(val: &Value) -> DType {
    match val {
        Value::Scalar(lit) => match lit {
            Literal::Bool(_) => DType::Bool,
            Literal::I64(_) => DType::I64,
            Literal::U32(_) => DType::U32,
            Literal::U64(_) => DType::U64,
            Literal::BF16Bits(_) => DType::BF16,
            Literal::F16Bits(_) => DType::F16,
            Literal::F64Bits(_) => DType::F64,
            Literal::Complex64Bits(..) => DType::Complex64,
            Literal::Complex128Bits(..) => DType::Complex128,
        },
        Value::Tensor(t) => t.dtype,
    }
}

fn load_bundle() -> DtypeBundle {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/dtype_promotion_oracle.v1.json");
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read dtype fixture: {e}"));
    serde_json::from_str(&data).expect("failed to parse dtype fixture")
}

#[test]
fn dtype_promotion_matches_jax() {
    let bundle = load_bundle();
    assert!(!bundle.cases.is_empty());

    // Core scalar types that FrankenJAX fully supports at scalar level.
    // Known gaps:
    // - i32: stored as i64 internally
    // - f32: no native F32 scalar literal (all floats stored as F64Bits)
    // - bf16/f16: half-precision scalar ops limited
    let core_dtypes = ["bool", "i64", "u32", "u64", "f64"];

    let mut mismatches = Vec::new();
    let mut tested = 0;

    for case in &bundle.cases {
        // Skip cases where JAX returned an error
        let Some(ref jax_result_dtype_str) = case.result_dtype else {
            continue;
        };

        // Only test core scalar dtype pairs
        if !core_dtypes.contains(&case.lhs_dtype.as_str())
            || !core_dtypes.contains(&case.rhs_dtype.as_str())
        {
            continue;
        }

        let jax_dtype = jax_dtype_to_fj(jax_result_dtype_str);
        let lhs_dtype = dtype_from_name(&case.lhs_dtype);
        let rhs_dtype = dtype_from_name(&case.rhs_dtype);

        let prim = match case.operation.as_str() {
            "add" => Primitive::Add,
            "mul" => Primitive::Mul,
            _ => continue,
        };

        let lhs = make_typed_value(lhs_dtype);
        let rhs = make_typed_value(rhs_dtype);

        match eval_primitive(prim, &[lhs, rhs], &BTreeMap::new()) {
            Ok(result) => {
                let fj_dtype = result_dtype(&result);
                if fj_dtype != jax_dtype {
                    mismatches.push(format!(
                        "{}: FrankenJAX={fj_dtype:?}, JAX={jax_dtype:?}",
                        case.case_id
                    ));
                }
                tested += 1;
            }
            Err(e) => {
                // FrankenJAX errored but JAX succeeded — this is a gap
                mismatches.push(format!("{}: FrankenJAX error: {e}", case.case_id));
                tested += 1;
            }
        }
    }

    println!("Tested {tested} dtype promotion cases");

    if !mismatches.is_empty() {
        println!(
            "Dtype promotion mismatches ({}/{tested}):",
            mismatches.len()
        );
        for m in &mismatches {
            println!("  {m}");
        }
        // For core numeric types, all promotions should match JAX exactly.
        assert!(
            mismatches.is_empty(),
            "dtype promotion mismatches for core types: {}/{tested}\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    }
}

/// Test dtype promotion at the tensor level, which supports F32, BF16, and F16 natively.
/// These types only have proper representation in TensorValue, not as scalar Literals.
#[test]
fn dtype_promotion_tensor_level() {
    let bundle = load_bundle();
    assert!(!bundle.cases.is_empty());

    // Expand to float types that need tensor representation
    let tensor_dtypes = ["bool", "i64", "u32", "u64", "f16", "f32", "f64", "bf16"];

    let mut mismatches = Vec::new();
    let mut tested = 0;

    for case in &bundle.cases {
        let Some(ref jax_result_dtype_str) = case.result_dtype else {
            continue;
        };

        // Test pairs where at least one operand is a half-precision type
        let lhs_is_half = matches!(case.lhs_dtype.as_str(), "f16" | "f32" | "bf16");
        let rhs_is_half = matches!(case.rhs_dtype.as_str(), "f16" | "f32" | "bf16");
        if !lhs_is_half && !rhs_is_half {
            continue;
        }
        if !tensor_dtypes.contains(&case.lhs_dtype.as_str())
            || !tensor_dtypes.contains(&case.rhs_dtype.as_str())
        {
            continue;
        }

        let jax_dtype = jax_dtype_to_fj(jax_result_dtype_str);
        let lhs_dtype = dtype_from_name(&case.lhs_dtype);
        let rhs_dtype = dtype_from_name(&case.rhs_dtype);

        let prim = match case.operation.as_str() {
            "add" => Primitive::Add,
            "mul" => Primitive::Mul,
            _ => continue,
        };

        let lhs = make_typed_tensor(lhs_dtype);
        let rhs = make_typed_tensor(rhs_dtype);

        match eval_primitive(prim, &[lhs, rhs], &BTreeMap::new()) {
            Ok(result) => {
                let fj_dtype = result_dtype(&result);
                if fj_dtype != jax_dtype {
                    mismatches.push(format!(
                        "{}: {}({:?}, {:?}) => FJ={fj_dtype:?}, JAX={jax_dtype:?}",
                        case.case_id, case.operation, lhs_dtype, rhs_dtype
                    ));
                }
                tested += 1;
            }
            Err(e) => {
                mismatches.push(format!(
                    "{}: {}({:?}, {:?}) => FJ error: {e}",
                    case.case_id, case.operation, lhs_dtype, rhs_dtype
                ));
                tested += 1;
            }
        }
    }

    println!("Tested {tested} tensor-level dtype promotion cases");
    if !mismatches.is_empty() {
        println!(
            "Tensor dtype promotion mismatches ({}/{tested}):",
            mismatches.len()
        );
        for m in &mismatches {
            println!("  {m}");
        }
    }

    println!(
        "Tensor dtype promotion: {}/{tested} passed",
        tested - mismatches.len()
    );
    assert!(
        mismatches.is_empty(),
        "tensor dtype promotion mismatches: {}/{tested}\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

// ============================================================================
// Complex dtype promotion coverage (frankenjax-pgs)
// ============================================================================

/// Verify promote_dtype() for all Complex64/Complex128 combinations.
/// These follow JAX's promotion lattice:
///   Complex128 + anything → Complex128
///   Complex64 + {F64, I64, U64} → Complex128 (component widening)
///   Complex64 + {F32, F16, BF16, I32, U32, Bool, Complex64} → Complex64
#[test]
fn complex_dtype_promotion_rules() {
    use fj_lax::promote_dtype_public as promote;

    let all_dtypes = [
        DType::Bool,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ];

    // Complex128 + anything → Complex128
    for &dt in &all_dtypes {
        assert_eq!(
            promote(DType::Complex128, dt),
            DType::Complex128,
            "Complex128 + {dt:?} should be Complex128"
        );
        assert_eq!(
            promote(dt, DType::Complex128),
            DType::Complex128,
            "{dt:?} + Complex128 should be Complex128"
        );
    }

    // Complex64 + types that widen to f64 → Complex128
    for &dt in &[DType::F64, DType::I64, DType::U64] {
        assert_eq!(
            promote(DType::Complex64, dt),
            DType::Complex128,
            "Complex64 + {dt:?} should be Complex128"
        );
        assert_eq!(
            promote(dt, DType::Complex64),
            DType::Complex128,
            "{dt:?} + Complex64 should be Complex128"
        );
    }

    // Complex64 + types within f32 → Complex64
    for &dt in &[
        DType::Bool,
        DType::I32,
        DType::U32,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::Complex64,
    ] {
        assert_eq!(
            promote(DType::Complex64, dt),
            DType::Complex64,
            "Complex64 + {dt:?} should be Complex64"
        );
        assert_eq!(
            promote(dt, DType::Complex64),
            DType::Complex64,
            "{dt:?} + Complex64 should be Complex64"
        );
    }
}

/// Verify that binary operations on complex values produce correctly promoted results.
#[test]
fn complex_binary_operation_dtype_output() {
    // Complex128 + F64 scalar → Complex128
    let c128 = Value::scalar_complex128(1.0, 2.0);
    let f64_val = Value::scalar_f64(3.0);
    let result = eval_primitive(Primitive::Add, &[c128, f64_val], &BTreeMap::new())
        .expect("Complex128 + F64 should succeed");
    assert_eq!(
        result.dtype(),
        DType::Complex128,
        "Complex128 + F64 should produce Complex128"
    );

    // Complex64 + F64 scalar → Complex128
    let c64 = Value::scalar_complex64(1.0, 2.0);
    let f64_val = Value::scalar_f64(3.0);
    let result = eval_primitive(Primitive::Add, &[c64.clone(), f64_val], &BTreeMap::new())
        .expect("Complex64 + F64 should succeed");
    assert_eq!(
        result.dtype(),
        DType::Complex128,
        "Complex64 + F64 should produce Complex128"
    );

    // Complex64 + I64 → Complex128
    let i64_val = Value::scalar_i64(5);
    let result = eval_primitive(Primitive::Add, &[c64.clone(), i64_val], &BTreeMap::new())
        .expect("Complex64 + I64 should succeed");
    assert_eq!(
        result.dtype(),
        DType::Complex128,
        "Complex64 + I64 should produce Complex128"
    );

    // Complex64 + Complex64 → Complex64
    let c64_2 = Value::scalar_complex64(3.0, 4.0);
    let result = eval_primitive(Primitive::Add, &[c64.clone(), c64_2], &BTreeMap::new())
        .expect("Complex64 + Complex64 should succeed");
    assert_eq!(
        result.dtype(),
        DType::Complex64,
        "Complex64 + Complex64 should produce Complex64"
    );

    // Complex64 + Complex128 → Complex128
    let c128 = Value::scalar_complex128(5.0, 6.0);
    let result = eval_primitive(Primitive::Add, &[c64, c128], &BTreeMap::new())
        .expect("Complex64 + Complex128 should succeed");
    assert_eq!(
        result.dtype(),
        DType::Complex128,
        "Complex64 + Complex128 should produce Complex128"
    );

    // Complex64 + Bool → Complex64, with Bool promoted as 0/1.
    let c64 = Value::scalar_complex64(1.0, 2.0);
    let result = eval_primitive(
        Primitive::Add,
        &[c64, Value::Scalar(Literal::Bool(true))],
        &BTreeMap::new(),
    )
    .expect("Complex64 + Bool should succeed");
    assert_eq!(
        result.dtype(),
        DType::Complex64,
        "Complex64 + Bool should produce Complex64"
    );
    assert_eq!(
        result.as_scalar_literal().and_then(Literal::as_complex64),
        Some((2.0, 2.0))
    );
}

/// Verify complex multiplication follows correct field arithmetic.
#[test]
fn complex_mul_correctness() {
    // (1+2i) * (3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
    let a = Value::scalar_complex128(1.0, 2.0);
    let b = Value::scalar_complex128(3.0, 4.0);
    let result = eval_primitive(Primitive::Mul, &[a, b], &BTreeMap::new())
        .expect("complex mul should succeed");
    let (re, im) = result.as_complex128_scalar().expect("should be complex128");
    assert!(
        (re - (-5.0)).abs() < 1e-10,
        "Re((1+2i)*(3+4i)) should be -5, got {re}"
    );
    assert!(
        (im - 10.0).abs() < 1e-10,
        "Im((1+2i)*(3+4i)) should be 10, got {im}"
    );
}

/// Verify complex division follows conjugate-denominator formula.
#[test]
fn complex_div_correctness() {
    // (1+2i) / (3+4i) = ((1*3+2*4) + (2*3-1*4)i) / (9+16) = (11 + 2i) / 25
    let a = Value::scalar_complex128(1.0, 2.0);
    let b = Value::scalar_complex128(3.0, 4.0);
    let result = eval_primitive(Primitive::Div, &[a, b], &BTreeMap::new())
        .expect("complex div should succeed");
    let (re, im) = result.as_complex128_scalar().expect("should be complex128");
    assert!(
        (re - 11.0 / 25.0).abs() < 1e-10,
        "Re((1+2i)/(3+4i)) should be 0.44, got {re}"
    );
    assert!(
        (im - 2.0 / 25.0).abs() < 1e-10,
        "Im((1+2i)/(3+4i)) should be 0.08, got {im}"
    );
}
