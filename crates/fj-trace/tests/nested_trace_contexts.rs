#![forbid(unsafe_code)]

use fj_core::{
    DType, ProgramSpec, Shape, TraceTransformLedger, Transform, Value, build_program,
    verify_transform_composition,
};
use fj_trace::{
    ShapedArray, SimpleTraceContext, TraceContext, TraceError, simulate_nested_trace_contexts,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn write_json_artifact(path: &PathBuf, payload: &serde_json::Value) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("artifact directory should be creatable");
    }
    let raw = serde_json::to_string_pretty(payload).expect("artifact payload should serialize");
    fs::write(path, raw).expect("artifact write should succeed");
}

fn scalar_f64_aval() -> ShapedArray {
    ShapedArray {
        dtype: DType::F64,
        shape: Shape::scalar(),
    }
}

fn vector_f64_aval(n: u32) -> ShapedArray {
    ShapedArray {
        dtype: DType::F64,
        shape: Shape::vector(n),
    }
}

fn transform_names(transforms: &[Transform]) -> Vec<String> {
    transforms.iter().map(|t| t.as_str().to_owned()).collect()
}

#[test]
fn test_nested_trace_jit_grad() {
    let summary = simulate_nested_trace_contexts(
        &[Transform::Jit, Transform::Grad],
        &[Value::scalar_f64(5.0)],
    )
    .expect("jit(grad) stack simulation should succeed");

    assert_eq!(summary.max_depth, 3);
    assert_eq!(summary.frames.len(), 2);
    assert_eq!(summary.frames[0].transform, Transform::Jit);
    assert_eq!(summary.frames[1].transform, Transform::Grad);
    assert_eq!(summary.frames[0].depth, 2);
    assert_eq!(summary.frames[1].depth, 3);
    assert!(summary.frames[0].trace_id < summary.frames[1].trace_id);
}

#[test]
fn test_nested_trace_vmap_jit() {
    let summary = simulate_nested_trace_contexts(
        &[Transform::Vmap, Transform::Jit],
        &[Value::vector_i64(&[1, 2, 3]).expect("vector should build")],
    )
    .expect("vmap(jit) stack simulation should succeed");

    assert_eq!(summary.max_depth, 3);
    assert_eq!(summary.frames.len(), 2);
    assert_eq!(summary.frames[0].transform, Transform::Vmap);
    assert_eq!(summary.frames[1].transform, Transform::Jit);
    assert_eq!(summary.frames[0].depth, 2);
    assert_eq!(summary.frames[1].depth, 3);
}

#[test]
fn test_nested_trace_grad_grad() {
    let summary = simulate_nested_trace_contexts(
        &[Transform::Grad, Transform::Grad],
        &[Value::scalar_f64(2.0)],
    )
    .expect("grad(grad) stack simulation should succeed");

    assert_eq!(summary.max_depth, 3);
    assert_eq!(summary.frames.len(), 2);
    assert_eq!(summary.frames[0].transform, Transform::Grad);
    assert_eq!(summary.frames[1].transform, Transform::Grad);
    assert_ne!(summary.frames[0].trace_id, summary.frames[1].trace_id);
}

#[test]
fn test_nested_trace_context_isolation() {
    let mut ctx = SimpleTraceContext::with_inputs(vec![scalar_f64_aval()]);
    let _trace_id = ctx.push_subtrace(vec![scalar_f64_aval()]);
    let nested_input = fj_trace::TracerId(2);
    let _ = ctx
        .process_primitive(
            fj_core::Primitive::Mul,
            &[nested_input, nested_input],
            BTreeMap::new(),
        )
        .expect("nested primitive should trace");
    let nested_closed = ctx
        .pop_subtrace_closed()
        .expect("nested subtrace should close into a ClosedJaxpr");
    assert_eq!(nested_closed.jaxpr.equations.len(), 1);

    let root_closed = ctx.finalize().expect("root finalize should succeed");
    assert!(
        root_closed.jaxpr.equations.is_empty(),
        "root frame should not inherit nested equations"
    );
}

#[test]
fn test_nested_trace_subtrace_open_close() {
    let mut ctx = SimpleTraceContext::with_inputs(vec![scalar_f64_aval()]);
    let err = ctx
        .pop_subtrace()
        .expect_err("popping root frame must fail");
    assert!(matches!(err, TraceError::CompositionViolation));

    let trace_id = ctx.push_subtrace(vec![scalar_f64_aval()]);
    assert_eq!(ctx.nesting_depth(), 2);
    let closed_id = ctx.pop_subtrace().expect("subtrace should close");
    assert_eq!(closed_id, trace_id);
    assert_eq!(ctx.nesting_depth(), 1);

    let mut unclosed = SimpleTraceContext::with_inputs(vec![scalar_f64_aval()]);
    let _ = unclosed.push_subtrace(vec![scalar_f64_aval()]);
    let err = unclosed
        .finalize()
        .expect_err("finalize with open subtrace must fail");
    assert!(matches!(err, TraceError::NestedTraceNotClosed));
}

#[test]
fn test_nested_trace_variable_numbering() {
    let mut ctx = SimpleTraceContext::with_inputs(vec![scalar_f64_aval()]);
    assert_eq!(ctx.next_tracer_id_hint(), 2);

    let first_trace = ctx.push_subtrace(vec![scalar_f64_aval()]);
    assert_eq!(ctx.next_tracer_id_hint(), 3);
    let out = ctx
        .process_primitive(
            fj_core::Primitive::Add,
            &[fj_trace::TracerId(2), fj_trace::TracerId(2)],
            BTreeMap::new(),
        )
        .expect("first subtrace add should trace");
    assert_eq!(out[0], fj_trace::TracerId(3));
    assert_eq!(ctx.next_tracer_id_hint(), 4);

    let second_trace = ctx.push_subtrace(vec![scalar_f64_aval()]);
    assert!(second_trace > first_trace);
    assert_eq!(ctx.next_tracer_id_hint(), 5);
    let inner_out = ctx
        .process_primitive(
            fj_core::Primitive::Neg,
            &[fj_trace::TracerId(4)],
            BTreeMap::new(),
        )
        .expect("second subtrace neg should trace");
    assert_eq!(inner_out[0], fj_trace::TracerId(5));
    assert_eq!(ctx.next_tracer_id_hint(), 6);
}

#[test]
fn test_nested_trace_transform_evidence() {
    let transforms = [Transform::Jit, Transform::Grad];
    let summary = simulate_nested_trace_contexts(&transforms, &[Value::scalar_f64(3.0)])
        .expect("nested summary should build");

    let mut ledger = TraceTransformLedger::new(build_program(ProgramSpec::Square));
    for (idx, transform) in transforms.iter().enumerate() {
        ledger.push_transform(
            *transform,
            format!("nested-evidence-{}-{idx}", transform.as_str()),
        );
    }
    let proof = verify_transform_composition(&ledger).expect("composition proof should validate");

    assert_eq!(proof.transform_count, summary.frames.len());
    assert_eq!(proof.evidence_count, transforms.len());
    assert!(proof.stack_signature.contains("jit>grad>"));
}

#[test]
fn e2e_nested_traces_oracle() {
    let scenarios = vec![
        (
            vec![Transform::Jit, Transform::Grad],
            vec![Value::scalar_f64(5.0)],
        ),
        (
            vec![Transform::Vmap, Transform::Grad],
            vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build")],
        ),
        (
            vec![Transform::Grad, Transform::Grad],
            vec![Value::scalar_f64(2.0)],
        ),
        (
            vec![Transform::Vmap, Transform::Jit],
            vec![Value::vector_f64(&[0.5, 1.5]).expect("vector should build")],
        ),
    ];

    let mut e2e_cases = Vec::new();
    let mut test_records = Vec::new();

    for (transforms, args) in scenarios {
        let summary = simulate_nested_trace_contexts(&transforms, &args)
            .expect("nested stack simulation should succeed");
        let expected_depth = transforms.len() + 1;
        let nesting_depth = summary.max_depth;
        let inner_frame = summary.frames.last();
        let outer_frame = summary.frames.first();
        let inner_jaxpr_eqns = inner_frame.map_or(0, |frame| frame.equation_count);
        let outer_jaxpr_eqns = outer_frame.map_or(0, |frame| frame.equation_count);
        let inner_vars = inner_frame.map_or(0, |frame| frame.invar_count);
        let outer_vars = outer_frame.map_or(0, |frame| frame.invar_count);
        let oracle_match =
            nesting_depth == expected_depth && summary.frames.len() == transforms.len();
        let pass = oracle_match;
        assert!(
            pass,
            "nested trace oracle mismatch: transforms={:?}, depth={}, expected_depth={}",
            transforms, nesting_depth, expected_depth
        );

        let names = transform_names(&transforms);
        e2e_cases.push(json!({
            "transform_stack": names,
            "nesting_depth": nesting_depth,
            "inner_jaxpr_eqns": inner_jaxpr_eqns,
            "outer_jaxpr_eqns": outer_jaxpr_eqns,
            "oracle_match": oracle_match,
            "pass": pass,
        }));
        test_records.push(json!({
            "test_name": "e2e_nested_traces_oracle",
            "nesting_depth": nesting_depth,
            "transforms": transform_names(&transforms),
            "inner_vars": inner_vars,
            "outer_vars": outer_vars,
            "pass": pass,
        }));
    }

    let generated_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after unix epoch")
        .as_millis();

    let e2e_path = repo_root().join("artifacts/e2e/e2e_nested_traces.e2e.json");
    let e2e_payload = json!({
        "schema_version": "frankenjax.e2e.nested_traces.v1",
        "scenario": "e2e_nested_traces_oracle",
        "generated_at_unix_ms": generated_at_unix_ms,
        "cases": e2e_cases,
    });
    write_json_artifact(&e2e_path, &e2e_payload);

    let test_log_path =
        repo_root().join("artifacts/testing/logs/fj-trace/e2e_nested_traces_oracle.json");
    let test_log_payload = json!({
        "schema_version": "frankenjax.testing.log.v1",
        "generated_at_unix_ms": generated_at_unix_ms,
        "records": test_records,
    });
    write_json_artifact(&test_log_path, &test_log_payload);
}

#[test]
fn test_nested_trace_vmap_input_shape_roundtrip() {
    let aval = vector_f64_aval(4);
    let mut ctx = SimpleTraceContext::with_inputs(vec![aval.clone()]);
    let _ = ctx.push_subtrace(vec![aval.clone()]);
    let closed = ctx
        .pop_subtrace_closed()
        .expect("closing vmap-like subtrace should succeed");
    assert_eq!(closed.jaxpr.invars.len(), 1);
    assert_eq!(closed.jaxpr.outvars.len(), 1);
}
