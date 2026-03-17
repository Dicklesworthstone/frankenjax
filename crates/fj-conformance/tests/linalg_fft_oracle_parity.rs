//! Oracle-backed linalg/FFT parity tests.
//!
//! Loads expected values captured from JAX 0.9.2 and compares them against
//! FrankenJAX's eval results. This validates that our linalg and FFT primitive
//! implementations produce outputs matching the JAX reference implementation.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::{eval_primitive, eval_primitive_multi};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Deserialize)]
struct OracleBundle {
    cases: Vec<OracleCase>,
}

#[derive(Deserialize)]
struct OracleCase {
    case_id: String,
    operation: String,
    inputs: Vec<OracleValue>,
    expected_outputs: Vec<OracleValue>,
    #[serde(default)]
    params: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum OracleValue {
    #[serde(rename = "matrix_f64")]
    MatrixF64 { shape: Vec<u32>, values: Vec<f64> },
    #[serde(rename = "vector_f64")]
    VectorF64 { shape: Vec<u32>, values: Vec<f64> },
    #[serde(rename = "complex_vector")]
    ComplexVector {
        shape: Vec<u32>,
        values: Vec<(f64, f64)>,
    },
}

fn oracle_value_to_fj(ov: &OracleValue) -> Value {
    match ov {
        OracleValue::MatrixF64 { shape, values } | OracleValue::VectorF64 { shape, values } => {
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: shape.clone(),
                    },
                    values.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            )
        }
        OracleValue::ComplexVector { shape, values } => Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape {
                    dims: shape.clone(),
                },
                values
                    .iter()
                    .map(|&(re, im)| Literal::from_complex128(re, im))
                    .collect(),
            )
            .unwrap(),
        ),
    }
}

fn extract_f64_vec(val: &Value) -> Vec<f64> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn extract_complex_vec(val: &Value) -> Vec<(f64, f64)> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| match l {
            Literal::Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
            _ => panic!("expected complex element"),
        })
        .collect()
}

fn assert_f64_close(actual: &[f64], expected: &[f64], tol: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{context}[{i}]: got {a}, expected {e}, diff={}",
            (a - e).abs()
        );
    }
}

fn assert_complex_close(actual: &[(f64, f64)], expected: &[(f64, f64)], tol: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (i, ((ar, ai), (er, ei))) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (ar - er).abs() < tol && (ai - ei).abs() < tol,
            "{context}[{i}]: got ({ar},{ai}), expected ({er},{ei})"
        );
    }
}

fn load_oracle_bundle() -> OracleBundle {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/linalg_fft_oracle.v1.json");
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read oracle fixture at {}: {e}", path.display()));
    serde_json::from_str(&data).expect("failed to parse oracle fixture JSON")
}

fn run_case(case: &OracleCase) {
    let inputs: Vec<Value> = case.inputs.iter().map(oracle_value_to_fj).collect();

    let prim = match case.operation.as_str() {
        "cholesky" => Primitive::Cholesky,
        "qr" => Primitive::Qr,
        "svd" => Primitive::Svd,
        "eigh" => Primitive::Eigh,
        "triangular_solve" => Primitive::TriangularSolve,
        "fft" => Primitive::Fft,
        "ifft" => Primitive::Ifft,
        "rfft" => Primitive::Rfft,
        "irfft" => Primitive::Irfft,
        other => panic!("unknown operation: {other}"),
    };

    let is_multi_output = matches!(
        prim,
        Primitive::Qr
            | Primitive::Svd
            | Primitive::Eigh
            | Primitive::Cholesky
            | Primitive::TriangularSolve
    );

    let tol = 1e-10;

    if is_multi_output {
        let outputs = eval_primitive_multi(prim, &inputs, &case.params).unwrap();
        assert_eq!(
            outputs.len(),
            case.expected_outputs.len(),
            "{}: output count mismatch",
            case.case_id
        );
        for (idx, (actual, expected_ov)) in
            outputs.iter().zip(case.expected_outputs.iter()).enumerate()
        {
            let context = format!("{} output[{idx}]", case.case_id);
            match expected_ov {
                OracleValue::MatrixF64 { values, .. } | OracleValue::VectorF64 { values, .. } => {
                    let actual_vals = extract_f64_vec(actual);
                    // Decompositions have sign ambiguity in eigenvectors/singular
                    // vectors. Only compare invariant outputs directly:
                    // - Eigh idx 0: eigenvalues (sorted ascending)
                    // - SVD idx 1: singular values (sorted descending)
                    // - Cholesky/TriangularSolve: fully determined
                    // Q, U, Vt, V, R: skip (sign ambiguity); reconstruction
                    // tests in linalg_oracle.rs verify correctness.
                    let should_compare = match prim {
                        Primitive::Cholesky | Primitive::TriangularSolve => true,
                        Primitive::Eigh => idx == 0,
                        Primitive::Svd => idx == 1,
                        Primitive::Qr => false,
                        _ => true,
                    };
                    if should_compare {
                        assert_f64_close(&actual_vals, values, tol, &context);
                    }
                }
                OracleValue::ComplexVector { values, .. } => {
                    let actual_vals = extract_complex_vec(actual);
                    assert_complex_close(&actual_vals, values, tol, &context);
                }
            }
        }
    } else {
        let output = eval_primitive(prim, &inputs, &case.params).unwrap();
        assert_eq!(
            case.expected_outputs.len(),
            1,
            "{}: expected 1 output",
            case.case_id
        );
        let context = &case.case_id;
        match &case.expected_outputs[0] {
            OracleValue::MatrixF64 { values, .. } | OracleValue::VectorF64 { values, .. } => {
                assert_f64_close(&extract_f64_vec(&output), values, tol, context);
            }
            OracleValue::ComplexVector { values, .. } => {
                assert_complex_close(&extract_complex_vec(&output), values, tol, context);
            }
        }
    }
}

#[test]
fn all_linalg_fft_oracle_cases_pass() {
    let bundle = load_oracle_bundle();
    assert!(
        !bundle.cases.is_empty(),
        "expected at least one oracle case"
    );

    let mut failures = Vec::new();
    for case in &bundle.cases {
        if let Err(e) = std::panic::catch_unwind(|| run_case(case)) {
            failures.push(format!(
                "{}: {:?}",
                case.case_id,
                e.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown panic".to_owned())
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "oracle parity failures ({}/{}):\n{}",
        failures.len(),
        bundle.cases.len(),
        failures.join("\n")
    );
}
