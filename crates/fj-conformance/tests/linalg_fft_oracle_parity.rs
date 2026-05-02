//! Oracle-backed linalg/FFT parity tests.
//!
//! Loads expected values from the linalg/FFT oracle fixture bundle and compares
//! them against FrankenJAX's eval results. This validates that our linalg and
//! FFT primitive implementations produce outputs matching the recorded reference
//! implementation.

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
    tolerance: Option<f64>,
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
    #[serde(rename = "tensor_f64")]
    TensorF64 { shape: Vec<u32>, values: Vec<f64> },
    #[serde(rename = "complex_vector")]
    ComplexVector {
        shape: Vec<u32>,
        values: Vec<(f64, f64)>,
    },
    #[serde(rename = "tensor_complex128")]
    TensorComplex128 {
        shape: Vec<u32>,
        reals: Vec<f64>,
        imags: Vec<f64>,
    },
    #[serde(rename = "tensor_complex64")]
    TensorComplex64 {
        shape: Vec<u32>,
        reals: Vec<f64>,
        imags: Vec<f64>,
    },
}

fn oracle_value_to_fj(ov: &OracleValue) -> Result<Value, String> {
    match ov {
        OracleValue::MatrixF64 { shape, values }
        | OracleValue::VectorF64 { shape, values }
        | OracleValue::TensorF64 { shape, values } => TensorValue::new(
            DType::F64,
            Shape {
                dims: shape.clone(),
            },
            values.iter().map(|&v| Literal::from_f64(v)).collect(),
        )
        .map(Value::Tensor)
        .map_err(|err| format!("invalid real-valued fixture tensor: {err}")),
        OracleValue::ComplexVector { shape, values } => {
            let (reals, imags): (Vec<f64>, Vec<f64>) = values.iter().copied().unzip();
            complex_tensor_value(shape, &reals, &imags, DType::Complex128)
        }
        OracleValue::TensorComplex128 {
            shape,
            reals,
            imags,
        } => complex_tensor_value(shape, reals, imags, DType::Complex128),
        OracleValue::TensorComplex64 {
            shape,
            reals,
            imags,
        } => complex_tensor_value(shape, reals, imags, DType::Complex64),
    }
}

fn complex_tensor_value(
    shape: &[u32],
    reals: &[f64],
    imags: &[f64],
    dtype: DType,
) -> Result<Value, String> {
    if reals.len() != imags.len() {
        return Err(format!(
            "complex tensor length mismatch: {} real values, {} imaginary values",
            reals.len(),
            imags.len()
        ));
    }

    let elements = match dtype {
        DType::Complex64 => reals
            .iter()
            .zip(imags.iter())
            .map(|(&re, &im)| Literal::from_complex64(re as f32, im as f32))
            .collect(),
        DType::Complex128 => reals
            .iter()
            .zip(imags.iter())
            .map(|(&re, &im)| Literal::from_complex128(re, im))
            .collect(),
        other => return Err(format!("unsupported complex fixture dtype: {other:?}")),
    };

    TensorValue::new(
        dtype,
        Shape {
            dims: shape.to_vec(),
        },
        elements,
    )
    .map(Value::Tensor)
    .map_err(|err| format!("invalid complex fixture tensor: {err}"))
}

fn extract_f64_vec(val: &Value) -> Vec<f64> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn extract_complex_vec(val: &Value) -> Result<Vec<(f64, f64)>, String> {
    val.as_tensor()
        .ok_or_else(|| "expected tensor output".to_owned())?
        .elements
        .iter()
        .enumerate()
        .map(|(idx, l)| match l {
            Literal::Complex64Bits(re, im) => {
                Ok((f32::from_bits(*re) as f64, f32::from_bits(*im) as f64))
            }
            Literal::Complex128Bits(re, im) => Ok((f64::from_bits(*re), f64::from_bits(*im))),
            other => Err(format!("expected complex element at {idx}, got {other:?}")),
        })
        .collect()
}

fn assert_tensor_dtype(actual: &Value, expected: DType, context: &str) -> Result<(), String> {
    let tensor = actual
        .as_tensor()
        .ok_or_else(|| format!("{context}: expected tensor output"))?;
    if tensor.dtype == expected {
        Ok(())
    } else {
        Err(format!(
            "{context}: expected dtype {expected:?}, got {:?}",
            tensor.dtype
        ))
    }
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

fn load_oracle_bundle() -> Result<OracleBundle, String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/linalg_fft_oracle.v1.json");
    let data = std::fs::read_to_string(&path)
        .map_err(|err| format!("failed to read oracle fixture {}: {err}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|err| format!("failed to parse oracle fixture {}: {err}", path.display()))
}

fn run_case(case: &OracleCase) -> Result<(), String> {
    let inputs: Vec<Value> = case
        .inputs
        .iter()
        .map(oracle_value_to_fj)
        .collect::<Result<_, _>>()?;

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
        other => return Err(format!("unknown operation: {other}")),
    };

    let is_multi_output = matches!(
        prim,
        Primitive::Qr
            | Primitive::Svd
            | Primitive::Eigh
            | Primitive::Cholesky
            | Primitive::TriangularSolve
    );

    let tol = case.tolerance.unwrap_or(1e-10);

    if is_multi_output {
        let outputs = eval_primitive_multi(prim, &inputs, &case.params)
            .map_err(|err| format!("{} multi-output eval failed: {err}", case.case_id))?;
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
                OracleValue::TensorF64 { values, .. } => {
                    let actual_vals = extract_f64_vec(actual);
                    assert_f64_close(&actual_vals, values, tol, &context);
                }
                OracleValue::ComplexVector { values, .. } => {
                    let actual_vals = extract_complex_vec(actual)?;
                    assert_complex_close(&actual_vals, values, tol, &context);
                }
                OracleValue::TensorComplex128 { reals, imags, .. } => {
                    assert_tensor_dtype(actual, DType::Complex128, &context)?;
                    let actual_vals = extract_complex_vec(actual)?;
                    let expected_vals: Vec<(f64, f64)> =
                        reals.iter().copied().zip(imags.iter().copied()).collect();
                    assert_complex_close(&actual_vals, &expected_vals, tol, &context);
                }
                OracleValue::TensorComplex64 { reals, imags, .. } => {
                    assert_tensor_dtype(actual, DType::Complex64, &context)?;
                    let actual_vals = extract_complex_vec(actual)?;
                    let expected_vals: Vec<(f64, f64)> =
                        reals.iter().copied().zip(imags.iter().copied()).collect();
                    assert_complex_close(&actual_vals, &expected_vals, tol, &context);
                }
            }
        }
    } else {
        let output = eval_primitive(prim, &inputs, &case.params)
            .map_err(|err| format!("{} eval failed: {err}", case.case_id))?;
        assert_eq!(
            case.expected_outputs.len(),
            1,
            "{}: expected 1 output",
            case.case_id
        );
        let context = &case.case_id;
        match &case.expected_outputs[0] {
            OracleValue::MatrixF64 { values, .. }
            | OracleValue::VectorF64 { values, .. }
            | OracleValue::TensorF64 { values, .. } => {
                assert_f64_close(&extract_f64_vec(&output), values, tol, context);
            }
            OracleValue::ComplexVector { values, .. } => {
                assert_complex_close(&extract_complex_vec(&output)?, values, tol, context);
            }
            OracleValue::TensorComplex128 { reals, imags, .. } => {
                assert_tensor_dtype(&output, DType::Complex128, context)?;
                let expected_vals: Vec<(f64, f64)> =
                    reals.iter().copied().zip(imags.iter().copied()).collect();
                assert_complex_close(&extract_complex_vec(&output)?, &expected_vals, tol, context);
            }
            OracleValue::TensorComplex64 { reals, imags, .. } => {
                assert_tensor_dtype(&output, DType::Complex64, context)?;
                let expected_vals: Vec<(f64, f64)> =
                    reals.iter().copied().zip(imags.iter().copied()).collect();
                assert_complex_close(&extract_complex_vec(&output)?, &expected_vals, tol, context);
            }
        }
    }
    Ok(())
}

#[test]
fn all_linalg_fft_oracle_cases_pass() -> Result<(), String> {
    let bundle = load_oracle_bundle()?;
    assert!(
        !bundle.cases.is_empty(),
        "expected at least one oracle case"
    );
    assert_eq!(bundle.cases.len(), 46, "unexpected linalg/FFT oracle count");
    for required_case in [
        "cholesky_5x5_spd",
        "qr_4x4_square",
        "svd_3x3_general",
        "svd_4x3_tall",
        "eigh_4x4_symmetric",
        "eigh_5x5_symmetric",
        "tsolve_lower_3x3",
        "tsolve_lower_3x3_multi_rhs",
        "fft_complex64_4point",
        "fft_complex128_8point",
        "rfft_8point_alternating",
        "irfft_8point_roundtrip",
    ] {
        assert!(
            bundle
                .cases
                .iter()
                .any(|case| case.case_id == required_case),
            "missing expanded oracle case {required_case}"
        );
    }

    let mut failures = Vec::new();
    for case in &bundle.cases {
        match std::panic::catch_unwind(|| run_case(case)) {
            Ok(Ok(())) => {}
            Ok(Err(err)) => failures.push(format!("{}: {err}", case.case_id)),
            Err(e) => {
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
    }

    assert!(
        failures.is_empty(),
        "oracle parity failures ({}/{}):\n{}",
        failures.len(),
        bundle.cases.len(),
        failures.join("\n")
    );
    Ok(())
}
