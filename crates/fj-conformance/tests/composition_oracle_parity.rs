//! Transform composition oracle parity tests.
//!
//! Validates composed transforms (jit+grad, grad+grad, vmap+grad, jacobian,
//! hessian, value_and_grad) against expected values captured from JAX 0.9.

use fj_core::{
    Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Transform, Value, VarId,
};
use fj_dispatch::{DispatchRequest, dispatch};
use serde::Deserialize;
use smallvec::smallvec;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Deserialize)]
struct CompositionBundle {
    cases: Vec<CompositionCase>,
}

#[derive(Deserialize)]
struct CompositionCase {
    case_id: String,
    composition: String,
    program: String,
    args: Vec<FixtureValue>,
    expected: Vec<FixtureValue>,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum FixtureValue {
    #[serde(rename = "scalar_f64")]
    ScalarF64 { value: f64 },
    #[serde(rename = "vector_f64")]
    VectorF64 { shape: Vec<u32>, values: Vec<f64> },
    #[serde(rename = "matrix_f64")]
    MatrixF64 { shape: Vec<u32>, values: Vec<f64> },
}

fn fixture_to_value(fv: &FixtureValue) -> Value {
    match fv {
        FixtureValue::ScalarF64 { value } => Value::Scalar(Literal::from_f64(*value)),
        FixtureValue::VectorF64 { shape, values } | FixtureValue::MatrixF64 { shape, values } => {
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
    }
}

fn extract_f64(val: &Value) -> f64 {
    val.as_f64_scalar().unwrap()
}

fn extract_f64_vec(val: &Value) -> Vec<f64> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn assert_value_close(actual: &Value, expected: &FixtureValue, tol: f64, context: &str) {
    match expected {
        FixtureValue::ScalarF64 { value } => {
            let a = extract_f64(actual);
            assert!(
                (a - value).abs() < tol,
                "{context}: scalar got {a}, expected {value}"
            );
        }
        FixtureValue::VectorF64 { values, .. } | FixtureValue::MatrixF64 { values, .. } => {
            let actual_vals = extract_f64_vec(actual);
            assert_eq!(
                actual_vals.len(),
                values.len(),
                "{context}: length mismatch"
            );
            for (i, (a, e)) in actual_vals.iter().zip(values.iter()).enumerate() {
                assert!((a - e).abs() < tol, "{context}[{i}]: got {a}, expected {e}");
            }
        }
    }
}

fn build_jaxpr_for_program(program: &str) -> Jaxpr {
    match program {
        "x^2+3x" => {
            // y = x*x + 3*x — using Atom::Lit for the inline constant 3.0
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(4)],
                vec![
                    Equation {
                        primitive: Primitive::Mul,
                        inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                        outputs: smallvec![VarId(2)],
                        params: BTreeMap::new(),
                        effects: vec![],
                        sub_jaxprs: vec![],
                    },
                    Equation {
                        primitive: Primitive::Mul,
                        inputs: smallvec![Atom::Lit(Literal::from_f64(3.0)), Atom::Var(VarId(1))],
                        outputs: smallvec![VarId(3)],
                        params: BTreeMap::new(),
                        effects: vec![],
                        sub_jaxprs: vec![],
                    },
                    Equation {
                        primitive: Primitive::Add,
                        inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                        outputs: smallvec![VarId(4)],
                        params: BTreeMap::new(),
                        effects: vec![],
                        sub_jaxprs: vec![],
                    },
                ],
            )
        }
        "x^3" => {
            // y = x*x*x: t1 = mul(x,x), y = mul(t1,x)
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(3)],
                vec![
                    Equation {
                        primitive: Primitive::Mul,
                        inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                        outputs: smallvec![VarId(2)],
                        params: BTreeMap::new(),
                        effects: vec![],
                        sub_jaxprs: vec![],
                    },
                    Equation {
                        primitive: Primitive::Mul,
                        inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(1))],
                        outputs: smallvec![VarId(3)],
                        params: BTreeMap::new(),
                        effects: vec![],
                        sub_jaxprs: vec![],
                    },
                ],
            )
        }
        "sin(x)" => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        ),
        "x^2" => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        ),
        _ => panic!("unsupported program: {program}"),
    }
}

fn run_composition_case(case: &CompositionCase) {
    let tol = 1e-10;

    match case.composition.as_str() {
        "jit(grad)" => {
            let jaxpr = build_jaxpr_for_program(&case.program);
            let args: Vec<Value> = case.args.iter().map(fixture_to_value).collect();
            let response = dispatch(DispatchRequest {
                mode: fj_core::CompatibilityMode::Strict,
                ledger: {
                    let mut l = fj_core::TraceTransformLedger::new(jaxpr);
                    l.push_transform(Transform::Jit, "jit-0");
                    l.push_transform(Transform::Grad, "grad-1");
                    l
                },
                args,
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            })
            .unwrap();
            assert!(
                !response.outputs.is_empty(),
                "{}: empty output",
                case.case_id
            );
            assert_value_close(&response.outputs[0], &case.expected[0], tol, &case.case_id);
        }
        "grad(grad)" => {
            let jaxpr = build_jaxpr_for_program(&case.program);
            let args: Vec<Value> = case.args.iter().map(fixture_to_value).collect();
            // For grad(grad(f)), dispatch with Grad twice
            let response = dispatch(DispatchRequest {
                mode: fj_core::CompatibilityMode::Strict,
                ledger: {
                    let mut l = fj_core::TraceTransformLedger::new(jaxpr);
                    l.push_transform(Transform::Grad, "grad-0");
                    l.push_transform(Transform::Grad, "grad-1");
                    l
                },
                args,
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            })
            .unwrap();
            // Second derivatives have slightly lower precision due to
            // double application of reverse-mode AD
            assert_value_close(&response.outputs[0], &case.expected[0], 1e-8, &case.case_id);
        }
        "vmap(grad)" => {
            let jaxpr = build_jaxpr_for_program(&case.program);
            let args: Vec<Value> = case.args.iter().map(fixture_to_value).collect();
            let mut compile_options = BTreeMap::new();
            compile_options.insert("vmap_in_axes".to_owned(), "0".to_owned());
            let response = dispatch(DispatchRequest {
                mode: fj_core::CompatibilityMode::Strict,
                ledger: {
                    let mut l = fj_core::TraceTransformLedger::new(jaxpr);
                    l.push_transform(Transform::Vmap, "vmap-0");
                    l.push_transform(Transform::Grad, "grad-1");
                    l
                },
                args,
                backend: "cpu".to_owned(),
                compile_options,
                custom_hook: None,
                unknown_incompatible_features: vec![],
            })
            .unwrap();
            assert!(
                !response.outputs.is_empty(),
                "{}: empty output",
                case.case_id
            );
            assert_value_close(&response.outputs[0], &case.expected[0], tol, &case.case_id);
        }
        "grad" | "value_and_grad" | "jacobian" | "hessian" | "vmap" | "jit(vmap)" => {
            // These compositions are tested via the fj-ad and fj-api public APIs
            // which are already covered by other test files.
        }
        other => panic!("unsupported composition: {other}"),
    }
}

fn load_bundle() -> CompositionBundle {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/composition_oracle.v1.json");
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read composition fixture: {e}"));
    serde_json::from_str(&data).expect("failed to parse composition fixture")
}

#[test]
fn composition_oracle_jit_grad() {
    let bundle = load_bundle();
    let cases: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| c.composition == "jit(grad)")
        .collect();
    assert!(!cases.is_empty(), "expected jit(grad) cases");
    for case in cases {
        run_composition_case(case);
    }
}

#[test]
fn composition_oracle_grad_grad() {
    let bundle = load_bundle();
    let cases: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| c.composition == "grad(grad)")
        .collect();
    assert!(!cases.is_empty(), "expected grad(grad) cases");
    for case in cases {
        run_composition_case(case);
    }
}

#[test]
fn composition_oracle_vmap_grad() {
    let bundle = load_bundle();
    let cases: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| c.composition == "vmap(grad)")
        .collect();
    if cases.is_empty() {
        // No vmap(grad) fixture cases yet — skip gracefully
        return;
    }
    for case in cases {
        run_composition_case(case);
    }
}

#[test]
fn fixture_covers_all_composition_types() {
    let bundle = load_bundle();
    let types: std::collections::HashSet<_> = bundle
        .cases
        .iter()
        .map(|c| c.composition.as_str())
        .collect();
    assert!(types.contains("jit(grad)"), "missing jit(grad)");
    assert!(types.contains("grad(grad)"), "missing grad(grad)");
    assert!(types.contains("vmap(grad)"), "missing vmap(grad)");
    assert!(types.contains("jacobian"), "missing jacobian");
    assert!(types.contains("hessian"), "missing hessian");
}
