#![forbid(unsafe_code)]

//! Comprehensive unit + property tests for dispatch/effects runtime (FJ-P2C-004-E).
//! Covers transform dispatch, gradient accuracy, effect tokens, cache integration,
//! evidence ledger, and property-based dispatch determinism.

use fj_core::{
    Atom, CompatibilityMode, DType, Equation, Jaxpr, Literal, Primitive, ProgramSpec, Shape,
    TensorValue, TraceTransformLedger, Transform, Value, VarId, build_program,
};
use fj_dispatch::{
    DispatchError, DispatchRequest, EffectContext, TransformExecutionError, dispatch,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn ledger(program: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(build_program(program));
    for (idx, transform) in transforms.iter().enumerate() {
        ledger.push_transform(
            *transform,
            format!("evidence-{}-{}", transform.as_str(), idx),
        );
    }
    ledger
}

fn make_request(
    program: ProgramSpec,
    transforms: &[Transform],
    args: Vec<Value>,
) -> DispatchRequest {
    DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ledger(program, transforms),
        args,
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    }
}

fn make_i64_matrix(rows: usize, cols: usize, data: &[i64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            data.iter().copied().map(Literal::I64).collect(),
        )
        .unwrap(),
    )
}

fn make_f64_tensor(dims: &[u32], data: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: dims.to_vec(),
            },
            data.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn make_vmap_request_with_axes(jaxpr: Jaxpr, args: Vec<Value>, in_axes: &str) -> DispatchRequest {
    let mut ttl = TraceTransformLedger::new(jaxpr);
    ttl.push_transform(Transform::Vmap, format!("vmap-{in_axes}"));
    let mut compile_options = BTreeMap::new();
    compile_options.insert("vmap_in_axes".to_owned(), in_axes.to_owned());
    DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ttl,
        args,
        backend: "cpu".to_owned(),
        compile_options,
        custom_hook: None,
        unknown_incompatible_features: vec![],
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn value_shape(value: &Value) -> Vec<u32> {
    match value {
        Value::Scalar(_) => vec![],
        Value::Tensor(tensor) => tensor.shape.dims.clone(),
    }
}

fn write_json_artifact(path: &PathBuf, payload: &serde_json::Value) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("artifact directory should be creatable");
    }
    let raw = serde_json::to_string_pretty(payload).expect("artifact payload should serialize");
    fs::write(path, raw).expect("artifact write should succeed");
}

// ── 1. Transform Dispatch Tests ────────────────────────────────────

#[test]
fn jit_scalar_add() {
    let r = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(10), Value::scalar_i64(20)],
    ))
    .unwrap();
    assert_eq!(r.outputs, vec![Value::scalar_i64(30)]);
}

#[test]
fn jit_vector_add_one() {
    let r = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Jit],
        vec![Value::vector_i64(&[1, 2, 3]).unwrap()],
    ))
    .unwrap();
    let out = r.outputs[0].as_tensor().unwrap();
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![2, 3, 4]);
}

#[test]
fn grad_polynomial_square_at_3() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::scalar_f64(3.0)],
    ))
    .unwrap();
    let derivative = r.outputs[0].as_f64_scalar().unwrap();
    assert!(
        (derivative - 6.0).abs() < 1e-3,
        "d/dx(x²) at 3 = 6, got {derivative}"
    );
}

#[test]
fn grad_square_plus_linear_at_2() {
    let r = dispatch(make_request(
        ProgramSpec::SquarePlusLinear,
        &[Transform::Grad],
        vec![Value::scalar_f64(2.0)],
    ))
    .unwrap();
    let derivative = r.outputs[0].as_f64_scalar().unwrap();
    // d/dx(x² + 2x) at x=2 = 2*2 + 2 = 6
    assert!((derivative - 6.0).abs() < 1e-3, "got {derivative}");
}

#[test]
fn vmap_varying_batch_sizes() {
    for size in [1, 5, 10] {
        let data: Vec<i64> = (1..=size).collect();
        let r = dispatch(make_request(
            ProgramSpec::AddOne,
            &[Transform::Vmap],
            vec![Value::vector_i64(&data).unwrap()],
        ))
        .unwrap();
        let out = r.outputs[0].as_tensor().unwrap();
        let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        let expected: Vec<i64> = data.iter().map(|x| x + 1).collect();
        assert_eq!(elems, expected, "batch_size={size}");
    }
}

#[test]
fn jit_grad_composition() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Grad],
        vec![Value::scalar_f64(5.0)],
    ))
    .unwrap();
    let derivative = r.outputs[0].as_f64_scalar().unwrap();
    assert!(
        (derivative - 10.0).abs() < 1e-3,
        "jit(grad(x²)) at 5 = 10, got {derivative}"
    );
}

#[test]
fn vmap_grad_batch_gradient() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Vmap, Transform::Grad],
        vec![Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap()],
    ))
    .unwrap();
    let out = r.outputs[0].as_tensor().unwrap().to_f64_vec().unwrap();
    assert_eq!(out.len(), 3);
    assert!((out[0] - 2.0).abs() < 1e-3);
    assert!((out[1] - 4.0).abs() < 1e-3);
    assert!((out[2] - 6.0).abs() < 1e-3);
}

#[test]
fn jit_vmap_batch_execution() {
    let r = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Jit, Transform::Vmap],
        vec![Value::vector_i64(&[10, 20, 30]).unwrap()],
    ))
    .unwrap();
    let out = r.outputs[0].as_tensor().unwrap();
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![11, 21, 31]);
}

#[test]
fn vmap_cond_batched_predicate() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2), VarId(3)],
        vec![],
        vec![VarId(4)],
        vec![Equation {
            primitive: Primitive::Cond,
            inputs: smallvec::smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)),
                Atom::Var(VarId(3))
            ],
            outputs: smallvec::smallvec![VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let mut ttl = TraceTransformLedger::new(jaxpr);
    ttl.push_transform(Transform::Vmap, "vmap-cond");

    let r = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ttl,
        args: vec![
            Value::vector_i64(&[1, 0, 1]).unwrap(),
            Value::vector_i64(&[10, 20, 30]).unwrap(),
            Value::vector_i64(&[100, 200, 300]).unwrap(),
        ],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![10, 200, 30]);
}

#[test]
fn vmap_scan_batched_carry_and_xs() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Scan,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([("body_op".to_owned(), "add".to_owned())]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let mut ttl = TraceTransformLedger::new(jaxpr);
    ttl.push_transform(Transform::Vmap, "vmap-scan");

    let r = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ttl,
        args: vec![
            Value::vector_i64(&[1, 100]).unwrap(),
            make_i64_matrix(2, 3, &[1, 2, 3, 1, 2, 3]),
        ],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![7, 106]);
}

#[test]
fn vmap_while_batched_init_and_threshold() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2), VarId(3)],
        vec![],
        vec![VarId(4)],
        vec![Equation {
            primitive: Primitive::While,
            inputs: smallvec::smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)),
                Atom::Var(VarId(3))
            ],
            outputs: smallvec::smallvec![VarId(4)],
            params: BTreeMap::from([
                ("body_op".to_owned(), "add".to_owned()),
                ("cond_op".to_owned(), "lt".to_owned()),
                ("max_iter".to_owned(), "64".to_owned()),
            ]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let mut ttl = TraceTransformLedger::new(jaxpr);
    ttl.push_transform(Transform::Vmap, "vmap-while");

    let r = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ttl,
        args: vec![
            Value::vector_i64(&[0, 10]).unwrap(),
            Value::vector_i64(&[2, 3]).unwrap(),
            Value::vector_i64(&[5, 25]).unwrap(),
        ],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![6, 25]);
}

#[test]
fn test_batch_rule_gather_batched_indices() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Gather,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([("slice_sizes".to_owned(), "1".to_owned())]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let r = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            Value::vector_i64(&[10, 20, 30, 40]).unwrap(),
            make_i64_matrix(2, 2, &[3, 1, 0, 2]),
        ],
        "none,0",
    ))
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    assert_eq!(out.shape.dims, vec![2, 2]);
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![40, 20, 10, 30]);
}

#[test]
fn test_batch_rule_gather_batched_operand() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Gather,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([("slice_sizes".to_owned(), "1".to_owned())]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let r = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            make_i64_matrix(2, 4, &[10, 20, 30, 40, 100, 200, 300, 400]),
            Value::vector_i64(&[3, 1]).unwrap(),
        ],
        "0,none",
    ))
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    assert_eq!(out.shape.dims, vec![2, 2]);
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![40, 20, 400, 200]);
}

#[test]
fn test_batch_rule_gather_both_batched() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Gather,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([("slice_sizes".to_owned(), "1".to_owned())]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let r = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            make_i64_matrix(2, 4, &[10, 20, 30, 40, 100, 200, 300, 400]),
            make_i64_matrix(2, 2, &[3, 1, 0, 2]),
        ],
        "0,0",
    ))
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    assert_eq!(out.shape.dims, vec![2, 2]);
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![40, 20, 100, 300]);
}

#[test]
fn test_batch_rule_scatter_batched_updates() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2), VarId(3)],
        vec![],
        vec![VarId(4)],
        vec![Equation {
            primitive: Primitive::Scatter,
            inputs: smallvec::smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)),
                Atom::Var(VarId(3))
            ],
            outputs: smallvec::smallvec![VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let r = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            Value::vector_i64(&[0, 0, 0, 0]).unwrap(),
            Value::vector_i64(&[1, 3]).unwrap(),
            make_i64_matrix(2, 2, &[10, 30, 100, 300]),
        ],
        "none,none,0",
    ))
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    assert_eq!(out.shape.dims, vec![2, 4]);
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![0, 10, 0, 30, 0, 100, 0, 300]);
}

#[test]
fn test_batch_rule_scatter_batched_operand() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2), VarId(3)],
        vec![],
        vec![VarId(4)],
        vec![Equation {
            primitive: Primitive::Scatter,
            inputs: smallvec::smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)),
                Atom::Var(VarId(3))
            ],
            outputs: smallvec::smallvec![VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let r = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            make_i64_matrix(2, 4, &[0, 0, 0, 0, 10, 20, 30, 40]),
            Value::vector_i64(&[1, 3]).unwrap(),
            Value::vector_i64(&[5, 7]).unwrap(),
        ],
        "0,none,none",
    ))
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    assert_eq!(out.shape.dims, vec![2, 4]);
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![0, 5, 0, 7, 10, 5, 30, 7]);
}

#[test]
fn test_batch_rule_scatter_both_batched() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2), VarId(3)],
        vec![],
        vec![VarId(4)],
        vec![Equation {
            primitive: Primitive::Scatter,
            inputs: smallvec::smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)),
                Atom::Var(VarId(3))
            ],
            outputs: smallvec::smallvec![VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let r = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            make_i64_matrix(2, 4, &[0, 0, 0, 0, 10, 20, 30, 40]),
            make_i64_matrix(2, 2, &[1, 3, 0, 2]),
            make_i64_matrix(2, 2, &[5, 7, 100, 300]),
        ],
        "0,0,0",
    ))
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    assert_eq!(out.shape.dims, vec![2, 4]);
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![0, 5, 0, 7, 100, 20, 300, 40]);
}

#[test]
fn test_batch_rule_conv_batched_kernel() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Conv,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([
                ("padding".to_owned(), "valid".to_owned()),
                ("strides".to_owned(), "1".to_owned()),
            ]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let r = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            make_f64_tensor(&[1, 4, 1], &[1.0, 2.0, 3.0, 4.0]),
            make_f64_tensor(&[2, 2, 1, 1], &[1.0, 1.0, 1.0, -1.0]),
        ],
        "none,0",
    ))
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    assert_eq!(out.shape.dims, vec![2, 1, 3, 1]);
    let elems: Vec<f64> = out.elements.iter().map(|l| l.as_f64().unwrap()).collect();
    assert_eq!(elems, vec![3.0, 5.0, 7.0, -1.0, -1.0, -1.0]);
}

#[test]
fn test_batch_rule_conv_both_batched() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Conv,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([
                ("padding".to_owned(), "valid".to_owned()),
                ("strides".to_owned(), "1".to_owned()),
            ]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let r = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            make_f64_tensor(&[2, 1, 4, 1], &[1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0]),
            make_f64_tensor(&[2, 2, 1, 1], &[1.0, 1.0, 1.0, -1.0]),
        ],
        "0,0",
    ))
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    assert_eq!(out.shape.dims, vec![2, 1, 3, 1]);
    let elems: Vec<f64> = out.elements.iter().map(|l| l.as_f64().unwrap()).collect();
    assert_eq!(elems, vec![3.0, 5.0, 7.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_batch_rule_pad_negative() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Pad,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([
                ("padding_low".to_owned(), "-1".to_owned()),
                ("padding_high".to_owned(), "0".to_owned()),
                ("padding_interior".to_owned(), "0".to_owned()),
            ]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let err = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            make_i64_matrix(2, 3, &[1, 2, 3, 4, 5, 6]),
            Value::scalar_i64(0),
        ],
        "0,none",
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(TransformExecutionError::TensorBuild(_))
    ));
}

#[test]
fn test_batch_rule_dynamic_slice_batched_starts() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::DynamicSlice,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([("slice_sizes".to_owned(), "2".to_owned())]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let r = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap(),
            Value::vector_i64(&[0, 2, 10]).unwrap(),
        ],
        "none,0",
    ))
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    assert_eq!(out.shape.dims, vec![3, 2]);
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![10, 20, 30, 40, 40, 50]);
}

#[test]
fn test_batch_rule_dynamic_update_slice_batched_update() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2), VarId(3)],
        vec![],
        vec![VarId(4)],
        vec![Equation {
            primitive: Primitive::DynamicUpdateSlice,
            inputs: smallvec::smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)),
                Atom::Var(VarId(3))
            ],
            outputs: smallvec::smallvec![VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let r = dispatch(make_vmap_request_with_axes(
        jaxpr,
        vec![
            Value::vector_i64(&[0, 0, 0, 0]).unwrap(),
            make_i64_matrix(2, 2, &[5, 6, 7, 8]),
            Value::scalar_i64(1),
        ],
        "none,0,none",
    ))
    .unwrap();

    let out = r.outputs[0].as_tensor().unwrap();
    assert_eq!(out.shape.dims, vec![2, 4]);
    let elems: Vec<i64> = out.elements.iter().map(|l| l.as_i64().unwrap()).collect();
    assert_eq!(elems, vec![0, 5, 6, 0, 0, 7, 8, 0]);
}

#[test]
fn e2e_vmap_tensor_ops_oracle() {
    let mut e2e_cases = Vec::new();
    let mut test_records = Vec::new();

    let gather_jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Gather,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([("slice_sizes".to_owned(), "1".to_owned())]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let gather_args = vec![
        Value::vector_i64(&[10, 20, 30, 40]).unwrap(),
        make_i64_matrix(2, 2, &[3, 1, 0, 2]),
    ];
    let gather_input_shapes: Vec<Vec<u32>> = gather_args.iter().map(value_shape).collect();
    let gather_result = dispatch(make_vmap_request_with_axes(
        gather_jaxpr,
        gather_args.clone(),
        "none,0",
    ))
    .expect("gather e2e dispatch should succeed");
    let gather_out = gather_result.outputs[0]
        .as_tensor()
        .expect("gather e2e output should be a tensor");
    let gather_output_shape = gather_out.shape.dims.clone();
    let gather_actual: Vec<i64> = gather_out
        .elements
        .iter()
        .map(|lit| lit.as_i64().unwrap())
        .collect();
    let gather_expected = vec![40, 20, 10, 30];
    let gather_pass = gather_actual == gather_expected;
    assert!(
        gather_pass,
        "gather oracle mismatch: actual={gather_actual:?}, expected={gather_expected:?}"
    );
    e2e_cases.push(json!({
        "primitive": "gather",
        "batch_size": 2,
        "input_shapes": gather_input_shapes,
        "batch_dims": ["none", "0"],
        "output_shape": gather_output_shape.clone(),
        "oracle_match": gather_pass,
        "pass": gather_pass,
    }));
    test_records.push(json!({
        "test_name": "e2e_vmap_tensor_ops_oracle",
        "primitive": "gather",
        "batch_dim": "none,0",
        "input_shapes": gather_args.iter().map(value_shape).collect::<Vec<_>>(),
        "output_shape": gather_output_shape,
        "pass": gather_pass,
    }));

    let scatter_jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2), VarId(3)],
        vec![],
        vec![VarId(4)],
        vec![Equation {
            primitive: Primitive::Scatter,
            inputs: smallvec::smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)),
                Atom::Var(VarId(3))
            ],
            outputs: smallvec::smallvec![VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let scatter_args = vec![
        Value::vector_i64(&[0, 0, 0, 0]).unwrap(),
        Value::vector_i64(&[1, 3]).unwrap(),
        make_i64_matrix(2, 2, &[10, 30, 100, 300]),
    ];
    let scatter_input_shapes: Vec<Vec<u32>> = scatter_args.iter().map(value_shape).collect();
    let scatter_result = dispatch(make_vmap_request_with_axes(
        scatter_jaxpr,
        scatter_args.clone(),
        "none,none,0",
    ))
    .expect("scatter e2e dispatch should succeed");
    let scatter_out = scatter_result.outputs[0]
        .as_tensor()
        .expect("scatter e2e output should be a tensor");
    let scatter_output_shape = scatter_out.shape.dims.clone();
    let scatter_actual: Vec<i64> = scatter_out
        .elements
        .iter()
        .map(|lit| lit.as_i64().unwrap())
        .collect();
    let scatter_expected = vec![0, 10, 0, 30, 0, 100, 0, 300];
    let scatter_pass = scatter_actual == scatter_expected;
    assert!(
        scatter_pass,
        "scatter oracle mismatch: actual={scatter_actual:?}, expected={scatter_expected:?}"
    );
    e2e_cases.push(json!({
        "primitive": "scatter",
        "batch_size": 2,
        "input_shapes": scatter_input_shapes,
        "batch_dims": ["none", "none", "0"],
        "output_shape": scatter_output_shape.clone(),
        "oracle_match": scatter_pass,
        "pass": scatter_pass,
    }));
    test_records.push(json!({
        "test_name": "e2e_vmap_tensor_ops_oracle",
        "primitive": "scatter",
        "batch_dim": "none,none,0",
        "input_shapes": scatter_args.iter().map(value_shape).collect::<Vec<_>>(),
        "output_shape": scatter_output_shape,
        "pass": scatter_pass,
    }));

    let conv_jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Conv,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([
                ("padding".to_owned(), "valid".to_owned()),
                ("strides".to_owned(), "1".to_owned()),
            ]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let conv_args = vec![
        make_f64_tensor(&[1, 4, 1], &[1.0, 2.0, 3.0, 4.0]),
        make_f64_tensor(&[2, 2, 1, 1], &[1.0, 1.0, 1.0, -1.0]),
    ];
    let conv_input_shapes: Vec<Vec<u32>> = conv_args.iter().map(value_shape).collect();
    let conv_result = dispatch(make_vmap_request_with_axes(
        conv_jaxpr,
        conv_args.clone(),
        "none,0",
    ))
    .expect("conv e2e dispatch should succeed");
    let conv_out = conv_result.outputs[0]
        .as_tensor()
        .expect("conv e2e output should be a tensor");
    let conv_output_shape = conv_out.shape.dims.clone();
    let conv_actual: Vec<f64> = conv_out
        .elements
        .iter()
        .map(|lit| lit.as_f64().unwrap())
        .collect();
    let conv_expected = [3.0, 5.0, 7.0, -1.0, -1.0, -1.0];
    let conv_pass = conv_actual
        .iter()
        .zip(conv_expected)
        .all(|(actual, expected)| (*actual - expected).abs() < 1e-9);
    assert!(
        conv_pass,
        "conv oracle mismatch: actual={conv_actual:?}, expected={conv_expected:?}"
    );
    e2e_cases.push(json!({
        "primitive": "conv",
        "batch_size": 2,
        "input_shapes": conv_input_shapes,
        "batch_dims": ["none", "0"],
        "output_shape": conv_output_shape.clone(),
        "oracle_match": conv_pass,
        "pass": conv_pass,
    }));
    test_records.push(json!({
        "test_name": "e2e_vmap_tensor_ops_oracle",
        "primitive": "conv",
        "batch_dim": "none,0",
        "input_shapes": conv_args.iter().map(value_shape).collect::<Vec<_>>(),
        "output_shape": conv_output_shape,
        "pass": conv_pass,
    }));

    let pad_jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Pad,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([
                ("padding_low".to_owned(), "-1".to_owned()),
                ("padding_high".to_owned(), "0".to_owned()),
                ("padding_interior".to_owned(), "0".to_owned()),
            ]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let pad_args = vec![
        make_i64_matrix(2, 3, &[1, 2, 3, 4, 5, 6]),
        Value::scalar_i64(0),
    ];
    let pad_input_shapes: Vec<Vec<u32>> = pad_args.iter().map(value_shape).collect();
    let pad_result = dispatch(make_vmap_request_with_axes(
        pad_jaxpr,
        pad_args.clone(),
        "0,none",
    ));
    let pad_pass = matches!(
        pad_result,
        Err(DispatchError::TransformExecution(
            TransformExecutionError::TensorBuild(_)
        ))
    );
    assert!(pad_pass, "pad oracle mismatch: expected TensorBuild error");
    e2e_cases.push(json!({
        "primitive": "pad",
        "batch_size": 2,
        "input_shapes": pad_input_shapes,
        "batch_dims": ["0", "none"],
        "output_shape": [],
        "oracle_match": pad_pass,
        "pass": pad_pass,
    }));
    test_records.push(json!({
        "test_name": "e2e_vmap_tensor_ops_oracle",
        "primitive": "pad",
        "batch_dim": "0,none",
        "input_shapes": pad_args.iter().map(value_shape).collect::<Vec<_>>(),
        "output_shape": [],
        "pass": pad_pass,
    }));

    let generated_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after unix epoch")
        .as_millis();
    let e2e_path = repo_root().join("artifacts/e2e/e2e_vmap_tensor_ops.e2e.json");
    let e2e_payload = json!({
        "schema_version": "frankenjax.e2e.vmap_tensor_ops.v1",
        "scenario": "e2e_vmap_tensor_ops_oracle",
        "generated_at_unix_ms": generated_at_unix_ms,
        "cases": e2e_cases,
    });
    write_json_artifact(&e2e_path, &e2e_payload);

    let test_log_path =
        repo_root().join("artifacts/testing/logs/fj-dispatch/e2e_vmap_tensor_ops_oracle.json");
    let test_log_payload = json!({
        "schema_version": "frankenjax.testing.log.v1",
        "generated_at_unix_ms": generated_at_unix_ms,
        "records": test_records,
    });
    write_json_artifact(&test_log_path, &test_log_payload);
}

#[test]
fn invalid_grad_vmap_rejected() {
    // grad(vmap(f)) should fail: grad requires scalar input, gets vector
    let err = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Grad, Transform::Vmap],
        vec![Value::vector_i64(&[1, 2, 3]).unwrap()],
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(TransformExecutionError::NonScalarGradientInput)
    ));
}

#[test]
fn invalid_empty_args_grad() {
    let err = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![],
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(TransformExecutionError::EmptyArgumentList { .. })
    ));
}

#[test]
fn invalid_empty_args_vmap() {
    let err = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Vmap],
        vec![],
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        DispatchError::TransformExecution(TransformExecutionError::EmptyArgumentList { .. })
    ));
}

// ── 2. Gradient Accuracy Tests ─────────────────────────────────────

#[test]
fn grad_sin_at_zero_is_one() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Sin,
            inputs: smallvec::smallvec![Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        }],
    );
    let mut ttl = TraceTransformLedger::new(jaxpr);
    ttl.push_transform(Transform::Grad, "grad-sin".to_owned());
    let r = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ttl,
        args: vec![Value::scalar_f64(0.0)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap();
    let deriv = r.outputs[0].as_f64_scalar().unwrap();
    assert!(
        (deriv - 1.0).abs() < 1e-3,
        "d/dx(sin(x)) at 0 = cos(0) = 1, got {deriv}"
    );
}

#[test]
fn grad_numerical_vs_analytical_comparison() {
    // grad(x²) at x=7 should be 14 (analytical)
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::scalar_f64(7.0)],
    ))
    .unwrap();
    let analytical = r.outputs[0].as_f64_scalar().unwrap();

    // Numerical: (f(x+eps) - f(x-eps)) / (2*eps)
    let eps = 1e-6;
    let x = 7.0;
    let plus: f64 = (x + eps) * (x + eps);
    let minus: f64 = (x - eps) * (x - eps);
    let numerical = (plus - minus) / (2.0 * eps);

    assert!(
        (analytical - numerical).abs() < 1e-4,
        "analytical={analytical}, numerical={numerical}"
    );
}

// ── 3. Effect Token Tests ──────────────────────────────────────────

#[test]
fn effect_context_empty() {
    let ctx = EffectContext::new();
    assert_eq!(ctx.effect_count(), 0);
    let tokens = ctx.finalize();
    assert!(tokens.is_empty());
}

#[test]
fn effect_context_duplicate_preserves_sequence() {
    let mut ctx = EffectContext::new();
    let t1 = ctx.thread_token("jit");
    let t2 = ctx.thread_token("jit"); // same name, new sequence
    assert_eq!(t1.sequence_number, 0);
    assert_eq!(t2.sequence_number, 1);
    assert_eq!(ctx.effect_count(), 2);
    let tokens = ctx.finalize();
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].sequence_number, 0);
    assert_eq!(tokens[1].sequence_number, 1);
}

#[test]
fn effect_tokens_in_dispatch_single_transform() {
    let r = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    let entry = &r.evidence_ledger.entries()[0];
    let signal = entry
        .signals
        .iter()
        .find(|s| s.signal_name == "effect_token_count")
        .unwrap();
    assert_eq!(signal.detail, "effect_tokens=0");
}

#[test]
fn effect_tokens_in_dispatch_triple_transform() {
    let mut request = make_request(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Vmap, Transform::Grad],
        vec![Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap()],
    );
    request.ledger.root_jaxpr.effects = vec!["Print".to_owned()];
    request.ledger.root_jaxpr.equations[0].effects = vec!["RngConsume".to_owned()];
    let r = dispatch(request).unwrap();
    let entry = &r.evidence_ledger.entries()[0];
    let count_signal = entry
        .signals
        .iter()
        .find(|s| s.signal_name == "effect_token_count")
        .unwrap();
    assert_eq!(count_signal.detail, "effect_tokens=2");

    let trace_signal = entry
        .signals
        .iter()
        .find(|s| s.signal_name == "effect_token_trace")
        .unwrap();
    assert_eq!(trace_signal.detail, "effect_tokens=[0:Print,1:RngConsume]");
}

// ── 4. Cache Integration Tests ─────────────────────────────────────

#[test]
fn same_input_produces_same_cache_key() {
    let r1 = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    let r2 = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    assert_eq!(r1.cache_key, r2.cache_key);
}

#[test]
fn different_program_produces_different_key() {
    let r1 = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    let r2 = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit],
        vec![Value::scalar_f64(1.0)],
    ))
    .unwrap();
    assert_ne!(r1.cache_key, r2.cache_key);
}

#[test]
fn different_transforms_produce_different_key() {
    let r1 = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit],
        vec![Value::scalar_f64(2.0)],
    ))
    .unwrap();
    let r2 = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::scalar_f64(2.0)],
    ))
    .unwrap();
    assert_ne!(r1.cache_key, r2.cache_key);
}

#[test]
fn strict_rejects_unknown_features() {
    let err = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
        args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec!["unknown.feature.v3".to_owned()],
    })
    .unwrap_err();
    assert!(matches!(err, DispatchError::Cache(_)));
}

#[test]
fn hardened_includes_features_in_key() {
    let r1 = dispatch(DispatchRequest {
        mode: CompatibilityMode::Hardened,
        ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
        args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .unwrap();
    let r2 = dispatch(DispatchRequest {
        mode: CompatibilityMode::Hardened,
        ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
        args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec!["feature.x".to_owned()],
    })
    .unwrap();
    assert_ne!(r1.cache_key, r2.cache_key, "features should affect key");
}

// ── 5. Evidence Ledger Tests ───────────────────────────────────────

#[test]
fn every_dispatch_produces_ledger_entry() {
    let r = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    assert_eq!(r.evidence_ledger.len(), 1);
}

#[test]
fn ledger_entry_contains_all_signals() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Grad],
        vec![Value::scalar_f64(3.0)],
    ))
    .unwrap();
    let entry = &r.evidence_ledger.entries()[0];
    assert_eq!(entry.signals.len(), 6);

    let signal_names: Vec<&str> = entry
        .signals
        .iter()
        .map(|s| s.signal_name.as_str())
        .collect();
    assert!(signal_names.contains(&"eqn_count"));
    assert!(signal_names.contains(&"transform_depth"));
    assert!(signal_names.contains(&"transform_stack_hash"));
    assert!(signal_names.contains(&"nested_trace_depth"));
    assert!(signal_names.contains(&"effect_token_count"));
    assert!(signal_names.contains(&"effect_token_trace"));
}

#[test]
fn ledger_decision_id_matches_cache_key() {
    let r = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(1), Value::scalar_i64(2)],
    ))
    .unwrap();
    let entry = &r.evidence_ledger.entries()[0];
    assert_eq!(entry.decision_id, r.cache_key);
}

#[test]
fn ledger_transform_depth_signal_correct() {
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Vmap, Transform::Grad],
        vec![Value::vector_f64(&[1.0, 2.0]).unwrap()],
    ))
    .unwrap();
    let entry = &r.evidence_ledger.entries()[0];
    let depth_signal = entry
        .signals
        .iter()
        .find(|s| s.signal_name == "transform_depth")
        .unwrap();
    assert_eq!(depth_signal.detail, "transform_depth=3");
}

// ── 6. Property Tests ──────────────────────────────────────────────

mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(
            fj_test_utils::property_test_case_count()
        ))]

        #[test]
        fn prop_dispatch_deterministic(x in prop::num::f64::NORMAL.prop_filter(
            "finite",
            |x| x.is_finite() && x.abs() < 1e6
        )) {
            let _seed = fj_test_utils::capture_proptest_seed();
            let r1 = dispatch(make_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(x)],
            )).expect("dispatch 1");
            let r2 = dispatch(make_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(x)],
            )).expect("dispatch 2");
            prop_assert_eq!(r1.cache_key, r2.cache_key);
            prop_assert_eq!(r1.outputs.len(), r2.outputs.len());
            let v1 = r1.outputs[0].as_f64_scalar().unwrap();
            let v2 = r2.outputs[0].as_f64_scalar().unwrap();
            prop_assert!((v1 - v2).abs() < 1e-12, "non-deterministic: {v1} vs {v2}");
        }

        #[test]
        fn prop_cache_key_stability(x in prop::num::f64::NORMAL.prop_filter(
            "finite",
            |x| x.is_finite() && x.abs() < 1e6
        )) {
            let _seed = fj_test_utils::capture_proptest_seed();
            // Cache key should be independent of argument values
            // (it depends on program structure, not runtime values)
            let r1 = dispatch(make_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(x)],
            )).expect("dispatch 1");
            let r2 = dispatch(make_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(x + 1.0)],
            )).expect("dispatch 2");
            // Same program + transforms = same cache key regardless of args
            prop_assert_eq!(r1.cache_key, r2.cache_key);
        }

        #[test]
        fn prop_ledger_always_populated(x in prop::num::i64::ANY.prop_filter(
            "not extreme",
            |x| x.abs() < i64::MAX / 2
        )) {
            let _seed = fj_test_utils::capture_proptest_seed();
            let r = dispatch(make_request(
                ProgramSpec::Add2,
                &[Transform::Jit],
                vec![Value::scalar_i64(x), Value::scalar_i64(1)],
            )).expect("dispatch");
            prop_assert_eq!(r.evidence_ledger.len(), 1);
            prop_assert_eq!(r.evidence_ledger.entries()[0].signals.len(), 6);
        }

        #[test]
        fn prop_grad_is_2x_for_square(x in prop::num::f64::NORMAL.prop_filter(
            "finite and moderate",
            |x| x.is_finite() && x.abs() < 1e6
        )) {
            let _seed = fj_test_utils::capture_proptest_seed();
            let r = dispatch(make_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(x)],
            )).expect("dispatch");
            let deriv = r.outputs[0].as_f64_scalar().unwrap();
            prop_assert!((deriv - 2.0 * x).abs() < 1e-3, "d/dx(x²) at {x}: got {deriv}");
        }

        #[test]
        fn prop_jit_is_identity(x in prop::num::i64::ANY.prop_filter(
            "not extreme",
            |x| x.abs() < i64::MAX / 2
        )) {
            let _seed = fj_test_utils::capture_proptest_seed();
            // jit(add2)(x, 1) should equal add2(x, 1) without jit
            let with_jit = dispatch(make_request(
                ProgramSpec::Add2,
                &[Transform::Jit],
                vec![Value::scalar_i64(x), Value::scalar_i64(1)],
            )).expect("with jit");
            let without_jit = dispatch(make_request(
                ProgramSpec::Add2,
                &[],
                vec![Value::scalar_i64(x), Value::scalar_i64(1)],
            )).expect("without jit");
            prop_assert_eq!(with_jit.outputs, without_jit.outputs);
        }
    }
}

// ── Higher-rank tensor tests ──────────────────────────────────────

#[test]
fn jit_rank2_add2() {
    let a = Value::vector_i64(&[1, 2, 3, 4]).unwrap();
    let b = Value::vector_i64(&[10, 20, 30, 40]).unwrap();
    let r = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![a, b],
    ))
    .unwrap();
    let expected = Value::vector_i64(&[11, 22, 33, 44]).unwrap();
    assert_eq!(r.outputs, vec![expected]);
}

#[test]
fn jit_add_one_vector() {
    let a = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
    let r = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Jit],
        vec![a],
    ))
    .unwrap();
    if let Value::Tensor(t) = &r.outputs[0] {
        let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
        assert!((vals[2] - 4.0).abs() < 1e-10);
    } else {
        panic!("expected tensor output");
    }
}

#[test]
fn vmap_square_over_vector() {
    let a = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Vmap],
        vec![a],
    ))
    .unwrap();
    if let Value::Tensor(t) = &r.outputs[0] {
        let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 4.0).abs() < 1e-10);
        assert!((vals[2] - 9.0).abs() < 1e-10);
    } else {
        panic!("expected tensor output");
    }
}

#[test]
fn vmap_add_one_over_vector() {
    let a = Value::vector_i64(&[10, 20, 30]).unwrap();
    let r = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Vmap],
        vec![a],
    ))
    .unwrap();
    if let Value::Tensor(t) = &r.outputs[0] {
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![11, 21, 31]);
    } else {
        panic!("expected tensor output");
    }
}

#[test]
fn grad_sin_vector_input() {
    // grad(sin)(x) = cos(x) for each element
    let r = dispatch(make_request(
        ProgramSpec::SinX,
        &[Transform::Grad],
        vec![Value::scalar_f64(0.0)],
    ))
    .unwrap();
    let deriv = r.outputs[0].as_f64_scalar().unwrap();
    // cos(0) = 1
    assert!(
        (deriv - 1.0).abs() < 1e-3,
        "d/dx sin(0) should be ~1.0, got {deriv}"
    );
}

#[test]
fn jit_grad_square() {
    // jit(grad(square))(3.0) = 6.0
    let r = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Grad],
        vec![Value::scalar_f64(3.0)],
    ))
    .unwrap();
    let deriv = r.outputs[0].as_f64_scalar().unwrap();
    assert!(
        (deriv - 6.0).abs() < 1e-3,
        "jit(grad(x²))(3) should be ~6.0, got {deriv}"
    );
}
