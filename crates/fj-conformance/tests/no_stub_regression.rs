use fj_ad::AdError;
use fj_api::ApiError;
use fj_core::{
    Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, ValueError, VarId,
};
use fj_dispatch::{
    batching::{batch_eval_jaxpr, BatchTracer},
    TransformExecutionError,
};
use fj_egraph::{jaxpr_to_egraph, EGraphLoweringError, ExclusionReason};
use fj_interpreters::eval_jaxpr;
use fj_lax::{eval_primitive, EvalError};
use smallvec::smallvec;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const BANNED_SUBSTRINGS: &[&str] = &[
    "not yet implemented",
    "not implemented",
    "not yet supported",
    "default placeholder",
    "stub",
    "placeholder",
    "mock",
];

const REPRESENTATIVE_DTYPES: &[DType] = &[
    DType::I32,
    DType::I64,
    DType::U32,
    DType::U64,
    DType::BF16,
    DType::F16,
    DType::F32,
    DType::F64,
    DType::Complex64,
    DType::Complex128,
    DType::Bool,
];

const PRODUCTION_SRC_DIRS: &[&str] = &[
    "crates/fj-ad/src",
    "crates/fj-api/src",
    "crates/fj-backend-cpu/src",
    "crates/fj-backend-gpu/src",
    "crates/fj-cache/src",
    "crates/fj-conformance/src",
    "crates/fj-core/src",
    "crates/fj-dispatch/src",
    "crates/fj-egraph/src",
    "crates/fj-ffi/src",
    "crates/fj-interpreters/src",
    "crates/fj-lax/src",
    "crates/fj-ledger/src",
    "crates/fj-py/src",
    "crates/fj-runtime/src",
    "crates/fj-test-utils/src",
    "crates/fj-trace/src",
];

const STATUS_CLAIM_AUDIT_ROOT_FILES: &[&str] = &[
    "README.md",
    "FEATURE_PARITY.md",
    "EXHAUSTIVE_LEGACY_ANALYSIS.md",
];

const STATUS_CLAIM_AUDIT_DIRS: &[&str] = &[
    "artifacts/conformance",
    "artifacts/durability",
    "artifacts/e2e",
    "artifacts/phase2c",
    "artifacts/testing/logs",
    "crates/fj-conformance/fixtures",
];

const STALE_STATUS_MARKERS: &[&str] = &[
    "not implemented",
    "not yet implemented",
    "placeholder",
    "stub",
    "todo",
    "mock",
];

const SUSPICIOUS_EMPTY_SUCCESS_PATTERNS: &[&str] =
    &["Ok(Default::default())", "Ok(vec![])", "Ok(Vec::new())"];

const REVIEWED_EMPTY_SUCCESS_FILES: &[&str] = &[
    "crates/fj-ad/src/lib.rs",
    // transforms.rs: parse_i64_list_param returns an empty Vec for explicitly empty
    // comma-list parameters; that is the parsed value, not an unimplemented path.
    "crates/fj-api/src/transforms.rs",
    "crates/fj-core/src/lib.rs",
    "crates/fj-dispatch/src/batching.rs",
    "crates/fj-interpreters/src/partial_eval.rs",
    "crates/fj-interpreters/src/staging.rs",
    // arithmetic.rs: parse_dim_list returns an empty Vec for dot_general's empty
    // dimension-number lists; that is a valid arity-zero dimension list.
    "crates/fj-lax/src/arithmetic.rs",
    "crates/fj-lax/src/tensor_ops.rs",
    // threefry.rs: random_choice returns Ok(Vec::new()) for the n_draws == 0 case
    // (drawing zero samples yields an empty index vector, matching JAX) — reviewed,
    // not a stub.
    "crates/fj-lax/src/threefry.rs",
    "crates/fj-py/src/lib.rs",
    "crates/fj-trace/src/lib.rs",
];

fn all_primitives() -> &'static [Primitive] {
    Primitive::ALL
}

fn primitive_arity(primitive: Primitive) -> usize {
    match primitive {
        // Binary operations (2 inputs)
        Primitive::Add
        | Primitive::Sub
        | Primitive::Mul
        | Primitive::Max
        | Primitive::Min
        | Primitive::Pow
        | Primitive::Hypot
        | Primitive::LogAddExp
        | Primitive::LogAddExp2
        | Primitive::Div
        | Primitive::Rem
        | Primitive::Gcd
        | Primitive::Lcm
        | Primitive::Atan2
        | Primitive::Complex
        | Primitive::Dot
        | Primitive::DotGeneral
        | Primitive::Gather
        | Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge
        | Primitive::Concatenate
        | Primitive::Pad
        | Primitive::BitwiseAnd
        | Primitive::BitwiseOr
        | Primitive::BitwiseXor
        | Primitive::ShiftLeft
        | Primitive::ShiftRightArithmetic
        | Primitive::ShiftRightLogical
        | Primitive::Nextafter
        | Primitive::TriangularSolve
        | Primitive::Polygamma
        | Primitive::Igamma
        | Primitive::Igammac
        | Primitive::Zeta
        | Primitive::Heaviside
        | Primitive::CopySign
        | Primitive::Ldexp
        | Primitive::XLogY
        | Primitive::XLog1PY => 2,
        // Ternary operations (3 inputs)
        Primitive::Select
        | Primitive::Scatter
        | Primitive::Clamp
        | Primitive::Cond
        | Primitive::Fma
        | Primitive::Betainc
        | Primitive::SelectN => 3,
        // Unary operations (1 input)
        Primitive::Neg
        | Primitive::Abs
        | Primitive::Exp
        | Primitive::Log
        | Primitive::Sqrt
        | Primitive::Rsqrt
        | Primitive::Floor
        | Primitive::Ceil
        | Primitive::Round
        | Primitive::Trunc
        | Primitive::Sin
        | Primitive::Cos
        | Primitive::Tan
        | Primitive::Asin
        | Primitive::Acos
        | Primitive::Atan
        | Primitive::Deg2Rad
        | Primitive::Rad2Deg
        | Primitive::Sinh
        | Primitive::Cosh
        | Primitive::Tanh
        | Primitive::Asinh
        | Primitive::Acosh
        | Primitive::Atanh
        | Primitive::Expm1
        | Primitive::Log1p
        | Primitive::Log2
        | Primitive::Exp2
        | Primitive::Sinc
        | Primitive::Sign
        | Primitive::Square
        | Primitive::Reciprocal
        | Primitive::Logistic
        | Primitive::Erf
        | Primitive::Erfc
        | Primitive::ReduceSum
        | Primitive::ReduceMax
        | Primitive::ReduceMin
        | Primitive::ReduceProd
        | Primitive::ReduceAnd
        | Primitive::ReduceOr
        | Primitive::ReduceXor
        | Primitive::Reshape
        | Primitive::Slice
        | Primitive::Transpose
        | Primitive::BroadcastInDim
        | Primitive::DynamicSlice
        | Primitive::BitwiseNot
        | Primitive::ReduceWindow
        | Primitive::PopulationCount
        | Primitive::CountLeadingZeros
        | Primitive::CountTrailingZeros
        | Primitive::Conj
        | Primitive::Real
        | Primitive::Imag
        | Primitive::Cbrt
        | Primitive::Lgamma
        | Primitive::Digamma
        | Primitive::ErfInv
        | Primitive::IsFinite
        | Primitive::IsNan
        | Primitive::IsInf
        | Primitive::Signbit
        | Primitive::BesselI0e
        | Primitive::BesselI1e
        | Primitive::Cholesky
        | Primitive::Qr
        | Primitive::Lu
        | Primitive::Svd
        | Primitive::Eigh
        | Primitive::Fft
        | Primitive::Ifft
        | Primitive::Rfft
        | Primitive::Irfft
        | Primitive::OneHot
        | Primitive::Cumsum
        | Primitive::Cumprod
        | Primitive::Cummax
        | Primitive::Cummin
        | Primitive::Sort
        | Primitive::Argsort
        | Primitive::TopK
        | Primitive::Argmin
        | Primitive::Argmax
        | Primitive::Psum
        | Primitive::Pmean
        | Primitive::AllGather
        | Primitive::AllToAll
        | Primitive::Copy
        | Primitive::Rev
        | Primitive::Squeeze
        | Primitive::Split
        | Primitive::ExpandDims
        | Primitive::Tile
        | Primitive::IntegerPow
        | Primitive::BitcastConvertType
        | Primitive::ConvertElementType
        | Primitive::StopGradient
        | Primitive::ReducePrecision => 1,
        // Zero-input operations
        Primitive::Iota | Primitive::BroadcastedIota | Primitive::AxisIndex => 0,
        // Variable-arity operations
        Primitive::DynamicUpdateSlice | Primitive::While => 3,
        Primitive::Conv => 2,
        Primitive::Scan => 2,
        Primitive::Switch => 3,
        Primitive::AssociativeScan => 2,
        Primitive::Eig | Primitive::Solve | Primitive::Det | Primitive::Slogdet => 1,
    }
}

#[test]
fn primitive_arity_matches_runtime_contract_for_reviewed_edges() {
    for primitive in [
        Primitive::Neg,
        Primitive::IntegerPow,
        Primitive::Copy,
        Primitive::Rev,
        Primitive::Squeeze,
        Primitive::Split,
        Primitive::ExpandDims,
        Primitive::Psum,
        Primitive::Pmean,
        Primitive::AllGather,
        Primitive::AllToAll,
    ] {
        assert_eq!(
            primitive_arity(primitive),
            1,
            "{primitive:?} should exercise its one-input runtime path"
        );
    }

    assert_eq!(primitive_arity(Primitive::AxisIndex), 0);
    assert_eq!(primitive_arity(Primitive::While), 3);
    assert_eq!(primitive_arity(Primitive::Switch), 3);
}

#[test]
fn primitive_inventory_comes_from_core_source_of_truth() {
    let local_primitive_count = all_primitives().len() - Primitive::PMAP_COLLECTIVES.len();

    assert_eq!(
        all_primitives().len(),
        162,
        "update Primitive::ALL and this audit count when the core enum changes"
    );
    assert_eq!(
        Primitive::PMAP_COLLECTIVES.len(),
        5,
        "pmap collective inventory should stay explicit while V1 fails closed"
    );
    assert_eq!(
        local_primitive_count, 157,
        "V1 local eval/AD docs should track non-pmap primitive scope"
    );

    for &primitive in Primitive::PMAP_COLLECTIVES {
        assert!(
            all_primitives().contains(&primitive),
            "{primitive:?} should remain part of the canonical primitive inventory"
        );
    }
}

#[test]
fn collective_primitives_fail_closed_with_pmap_context_error() {
    for &primitive in Primitive::PMAP_COLLECTIVES {
        let inputs = representative_inputs(DType::I64, primitive);
        let err = eval_primitive(primitive, &inputs, &BTreeMap::new())
            .expect_err("pmap-only collective should fail closed in V1");
        match err {
            EvalError::Unsupported {
                primitive: actual_primitive,
                detail,
            } => {
                assert_eq!(
                    actual_primitive, primitive,
                    "collective error should retain the primitive identity"
                );
                assert!(
                    detail.contains(
                        "collective operation requires pmap context with multi-device backend"
                    ),
                    "collective error should explain pmap context requirement: {detail}"
                );
                assert_no_banned_substrings(primitive.as_str(), &detail);
            }
            other => {
                assert!(
                    matches!(other, EvalError::Unsupported { .. }),
                    "collective {primitive:?} should use unsupported pmap-context error"
                );
            }
        }
    }
}

#[test]
fn public_docs_scope_primitive_coverage_claims() {
    let canonical_count = Primitive::ALL.len();
    let pmap_count = Primitive::PMAP_COLLECTIVES.len();
    let local_count = canonical_count - pmap_count;

    for path in ["README.md", "FEATURE_PARITY.md"] {
        let source =
            std::fs::read_to_string(workspace_root().join(path)).expect("doc should be readable");
        assert!(
            source.contains(&format!("{canonical_count} canonical primitive")),
            "{path} should state the canonical primitive inventory from Primitive::ALL"
        );
        assert!(
            source.contains(&format!("{local_count} V1 local")),
            "{path} should state the non-pmap V1 local primitive scope"
        );
        assert!(
            source.contains(&format!("{pmap_count} pmap collectives")),
            "{path} should state the pmap collective count"
        );
        assert!(
            source.contains("fail closed"),
            "{path} should describe pmap collectives as fail-closed, not fully implemented"
        );
    }
}

fn params_for(primitive: Primitive, dtype: DType) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    match primitive {
        Primitive::Reshape => {
            params.insert("new_shape".to_owned(), "1".to_owned());
        }
        Primitive::Slice => {
            params.insert("start_indices".to_owned(), "0".to_owned());
            params.insert("limit_indices".to_owned(), "1".to_owned());
        }
        Primitive::DynamicSlice => {
            params.insert("slice_sizes".to_owned(), "1".to_owned());
        }
        Primitive::Gather => {
            params.insert("slice_sizes".to_owned(), "1".to_owned());
        }
        Primitive::Transpose => {
            params.insert("permutation".to_owned(), "0".to_owned());
        }
        Primitive::BroadcastInDim => {
            params.insert("shape".to_owned(), "1".to_owned());
            params.insert("broadcast_dimensions".to_owned(), "0".to_owned());
        }
        Primitive::Concatenate => {
            params.insert("dimension".to_owned(), "0".to_owned());
        }
        Primitive::Pad => {
            params.insert("padding_low".to_owned(), "0".to_owned());
            params.insert("padding_high".to_owned(), "0".to_owned());
            params.insert("padding_interior".to_owned(), "0".to_owned());
        }
        Primitive::Rev => {
            params.insert("dimensions".to_owned(), "0".to_owned());
        }
        Primitive::Squeeze => {
            params.insert("dimensions".to_owned(), "0".to_owned());
        }
        Primitive::Split => {
            params.insert("axis".to_owned(), "0".to_owned());
            params.insert("sizes".to_owned(), "1".to_owned());
        }
        Primitive::ExpandDims => {
            params.insert("dimensions".to_owned(), "0".to_owned());
        }
        Primitive::IntegerPow => {
            params.insert("exponent".to_owned(), "2".to_owned());
        }
        Primitive::Iota => {
            params.insert("length".to_owned(), "1".to_owned());
            params.insert("dtype".to_owned(), dtype_name(dtype).to_owned());
        }
        Primitive::BroadcastedIota => {
            params.insert("shape".to_owned(), "1".to_owned());
            params.insert("dimension".to_owned(), "0".to_owned());
            params.insert("dtype".to_owned(), dtype_name(dtype).to_owned());
        }
        Primitive::BitcastConvertType => {
            params.insert("dtype".to_owned(), "u32".to_owned());
        }
        Primitive::ReducePrecision => {
            params.insert("exponent_bits".to_owned(), "5".to_owned());
            params.insert("mantissa_bits".to_owned(), "10".to_owned());
        }
        Primitive::Fft | Primitive::Ifft | Primitive::Rfft | Primitive::Irfft => {
            params.insert("fft_lengths".to_owned(), "1".to_owned());
        }
        Primitive::OneHot => {
            params.insert("num_classes".to_owned(), "3".to_owned());
            params.insert("dtype".to_owned(), dtype_name(dtype).to_owned());
        }
        Primitive::Cumsum | Primitive::Cumprod | Primitive::Sort | Primitive::Argsort => {
            params.insert("axis".to_owned(), "0".to_owned());
        }
        Primitive::Conv => {
            params.insert("padding".to_owned(), "valid".to_owned());
            params.insert("strides".to_owned(), "1".to_owned());
        }
        Primitive::Scan => {
            params.insert("body_op".to_owned(), "add".to_owned());
        }
        Primitive::While => {
            params.insert("body_op".to_owned(), "add".to_owned());
            params.insert("cond_op".to_owned(), "lt".to_owned());
            params.insert("max_iter".to_owned(), "4".to_owned());
        }
        Primitive::Switch => {
            params.insert("branches".to_owned(), "identity,neg,abs".to_owned());
        }
        Primitive::ReduceWindow => {
            params.insert("window_dimensions".to_owned(), "1".to_owned());
            params.insert("window_strides".to_owned(), "1".to_owned());
            params.insert("reduce_op".to_owned(), "sum".to_owned());
        }
        _ => {}
    }
    params
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::U32 => "u32",
        DType::U64 => "u64",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
        DType::Bool => "bool",
    }
}

#[test]
fn dtype_parameter_names_cover_representative_dtype_matrix() {
    let expected_names = [
        (DType::I32, "i32"),
        (DType::I64, "i64"),
        (DType::U32, "u32"),
        (DType::U64, "u64"),
        (DType::BF16, "bf16"),
        (DType::F16, "f16"),
        (DType::F32, "f32"),
        (DType::F64, "f64"),
        (DType::Complex64, "complex64"),
        (DType::Complex128, "complex128"),
        (DType::Bool, "bool"),
    ];
    let dtype_param_primitives = [
        Primitive::Iota,
        Primitive::BroadcastedIota,
        Primitive::OneHot,
    ];

    for (dtype, expected_name) in expected_names {
        assert_eq!(dtype_name(dtype), expected_name);
        for primitive in dtype_param_primitives {
            let params = params_for(primitive, dtype);
            assert_eq!(
                params.get("dtype").map(String::as_str),
                Some(expected_name),
                "{primitive:?}/{dtype:?} should preserve the representative dtype"
            );
        }
    }
}

fn representative_input(dtype: DType, slot: usize, primitive: Primitive) -> Value {
    if primitive == Primitive::While {
        return match (dtype, slot) {
            // I32 has no distinct Literal variant — it shares Literal::I64.
            (DType::I32 | DType::I64, 0) => Value::scalar_i64(3),
            (DType::I32 | DType::I64, 1) => Value::scalar_i64(1),
            (DType::I32 | DType::I64, 2) => Value::scalar_i64(0),
            (DType::U32, 0) => Value::scalar_u32(3),
            (DType::U32, 1) => Value::scalar_u32(1),
            (DType::U32, 2) => Value::scalar_u32(0),
            (DType::U64, 0) => Value::scalar_u64(3),
            (DType::U64, 1) => Value::scalar_u64(1),
            (DType::U64, 2) => Value::scalar_u64(0),
            (DType::BF16, 0) => Value::scalar_bf16(3.0),
            (DType::BF16, 1) => Value::scalar_bf16(1.0),
            (DType::BF16, 2) => Value::scalar_bf16(0.0),
            (DType::F16, 0) => Value::scalar_f16(3.0),
            (DType::F16, 1) => Value::scalar_f16(1.0),
            (DType::F16, 2) => Value::scalar_f16(0.0),
            (DType::F32, 0) => Value::scalar_f32(3.0),
            (DType::F32, 1) => Value::scalar_f32(1.0),
            (DType::F32, 2) => Value::scalar_f32(0.0),
            (DType::F64, 0) => Value::scalar_f64(3.0),
            (DType::F64, 1) => Value::scalar_f64(1.0),
            (DType::F64, 2) => Value::scalar_f64(0.0),
            (DType::Complex64, 0) => Value::scalar_complex64(3.0, 0.0),
            (DType::Complex64, 1) => Value::scalar_complex64(1.0, 0.0),
            (DType::Complex64, 2) => Value::scalar_complex64(0.0, 0.0),
            (DType::Complex128, 0) => Value::scalar_complex128(3.0, 0.0),
            (DType::Complex128, 1) => Value::scalar_complex128(1.0, 0.0),
            (DType::Complex128, 2) => Value::scalar_complex128(0.0, 0.0),
            (DType::Bool, 0) => Value::scalar_bool(true),
            (DType::Bool, 1) => Value::scalar_bool(false),
            (DType::Bool, 2) => Value::scalar_bool(false),
            _ => unreachable!(),
        };
    }

    match dtype {
        DType::I32 | DType::I64 => Value::scalar_i64(2 + slot as i64),
        DType::U32 => Value::scalar_u32(2 + slot as u32),
        DType::U64 => Value::scalar_u64(2 + slot as u64),
        DType::BF16 => Value::scalar_bf16(1.5 + slot as f32),
        DType::F16 => Value::scalar_f16(1.5 + slot as f32),
        DType::F32 => Value::scalar_f32(1.5 + slot as f32),
        DType::F64 => Value::scalar_f64(1.5 + slot as f64),
        DType::Complex64 => Value::scalar_complex64(1.0 + slot as f32, 0.25 + slot as f32),
        DType::Complex128 => Value::scalar_complex128(1.0 + slot as f64, 0.25 + slot as f64),
        DType::Bool => Value::scalar_bool(slot.is_multiple_of(2)),
    }
}

fn representative_inputs(dtype: DType, primitive: Primitive) -> Vec<Value> {
    (0..primitive_arity(primitive))
        .map(|slot| representative_input(dtype, slot, primitive))
        .collect()
}

fn make_lowering_jaxpr(primitive: Primitive) -> Jaxpr {
    let input_count = primitive_arity(primitive);
    let invars = (0..input_count)
        .map(|idx| VarId((idx + 1) as u32))
        .collect::<Vec<_>>();
    let inputs = invars
        .iter()
        .copied()
        .map(Atom::Var)
        .collect::<smallvec::SmallVec<[Atom; 4]>>();
    let outvar = VarId((input_count + 1) as u32);
    let params = if primitive == Primitive::IntegerPow {
        BTreeMap::from([("exponent".to_owned(), "2".to_owned())])
    } else {
        BTreeMap::new()
    };
    Jaxpr::new(
        invars,
        vec![],
        vec![outvar],
        vec![Equation {
            primitive,
            inputs,
            outputs: smallvec![outvar],
            params,
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

fn make_invalid_batchtrace_sub_jaxpr_carrier() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(3)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![make_lowering_jaxpr(Primitive::Neg)],
        }],
    )
}

fn assert_no_banned_substrings(context: &str, message: &str) {
    let normalized = message.to_ascii_lowercase();
    for banned in BANNED_SUBSTRINGS {
        assert!(
            !normalized.contains(banned),
            "{context} contains banned substring `{banned}`: {message}"
        );
    }
}

fn snippet(message: &str) -> String {
    const MAX_LEN: usize = 80;
    let mut out = message.replace('\n', " ");
    if out.len() > MAX_LEN {
        out.truncate(MAX_LEN);
        out.push_str("...");
    }
    out
}

fn workspace_source_files() -> Vec<PathBuf> {
    fn visit(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }

    let root = workspace_root();
    let mut files = Vec::new();
    for dir in PRODUCTION_SRC_DIRS {
        visit(&root.join(dir), &mut files);
    }
    files.sort();
    files
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("fj-conformance should live under crates/")
        .to_path_buf()
}

fn status_claim_audit_files() -> Vec<PathBuf> {
    fn visit(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(&path, out);
            } else if path
                .extension()
                .is_some_and(|ext| matches!(ext.to_str(), Some("json" | "log" | "md")))
            {
                out.push(path);
            }
        }
    }

    let root = workspace_root();
    let mut files = STATUS_CLAIM_AUDIT_ROOT_FILES
        .iter()
        .map(|file| root.join(file))
        .collect::<Vec<_>>();
    for dir in STATUS_CLAIM_AUDIT_DIRS {
        visit(&root.join(dir), &mut files);
    }
    files.sort();
    files
}

fn brace_delta(line: &str) -> i32 {
    let opens = line.chars().filter(|ch| *ch == '{').count() as i32;
    let closes = line.chars().filter(|ch| *ch == '}').count() as i32;
    opens - closes
}

fn production_lines(source: &str) -> Vec<(usize, &str)> {
    let mut lines = Vec::new();
    let mut pending_cfg_test = false;
    let mut skipped_test_module_depth: Option<i32> = None;

    for (idx, line) in source.lines().enumerate() {
        if let Some(depth) = skipped_test_module_depth.as_mut() {
            *depth += brace_delta(line);
            if *depth <= 0 {
                skipped_test_module_depth = None;
            }
            continue;
        }

        let trimmed = line.trim_start();
        if trimmed.starts_with("#[cfg(test)]") {
            pending_cfg_test = true;
            continue;
        }

        if pending_cfg_test {
            if trimmed.starts_with("mod ") && trimmed.contains('{') {
                let depth = brace_delta(line);
                if depth > 0 {
                    skipped_test_module_depth = Some(depth);
                }
                pending_cfg_test = false;
                continue;
            }
            pending_cfg_test = trimmed.starts_with("#[");
            if pending_cfg_test {
                continue;
            }
        }

        lines.push((idx + 1, line));
    }

    lines
}

fn source_marker_allowed(line: &str) -> bool {
    line.contains("user-supplied placeholder paths")
}

fn has_suspicious_empty_success(line: &str) -> bool {
    SUSPICIOUS_EMPTY_SUCCESS_PATTERNS
        .iter()
        .any(|pattern| line.contains(pattern))
}

fn suspicious_default_return_allowed(path: &Path, line: &str) -> bool {
    if line.contains("return Ok(Vec::new())")
        && (path.ends_with("crates/fj-api/src/transforms.rs")
            || path.ends_with("crates/fj-lax/src/arithmetic.rs"))
    {
        // Empty parameter lists are valid parser outputs for reshape/dot dimension specs.
        return true;
    }

    has_suspicious_empty_success(line)
        && REVIEWED_EMPTY_SUCCESS_FILES
            .iter()
            .any(|allowed| path.ends_with(allowed))
}

fn stale_status_claim_allowed(path: &Path, line: &str) -> bool {
    let path = path.to_string_lossy();
    let normalized = line.to_ascii_lowercase();

    if path.ends_with("EXHAUSTIVE_LEGACY_ANALYSIS.md") || path.contains("artifacts/phase2c/") {
        return true;
    }

    if path.contains("e2e_stub_errors") || path.contains("v2_stub_linalg_fft_shape_contracts") {
        return true;
    }

    [
        "closed",
        "concrete replay",
        "cpu-only",
        "durability",
        "explicit gap",
        "ffi",
        "fixture",
        "future",
        "generated",
        "graceful",
        "intentional",
        "legacy",
        "out-of-scope",
        "replay",
        "scenario",
        "sidecar_path",
        "test_id",
        "v1",
    ]
    .iter()
    .any(|allowed| normalized.contains(allowed))
}

#[test]
fn no_stub_regression_matrix() {
    let mut eval_rows = Vec::new();
    let mut eval_attempts = 0usize;

    for primitive in all_primitives() {
        for dtype in REPRESENTATIVE_DTYPES {
            eval_attempts += 1;
            let inputs = representative_inputs(*dtype, *primitive);
            let params = params_for(*primitive, *dtype);
            match eval_primitive(*primitive, &inputs, &params) {
                Ok(_) => eval_rows.push(format!("{primitive:?} | {dtype:?} | ok | -")),
                Err(err) => {
                    let rendered = err.to_string();
                    assert_no_banned_substrings(
                        &format!("eval {:?}/{:?}", primitive, dtype),
                        &rendered,
                    );
                    eval_rows.push(format!(
                        "{primitive:?} | {dtype:?} | error | {}",
                        snippet(&rendered)
                    ));
                }
            }
        }
    }

    assert_eq!(
        eval_attempts,
        all_primitives().len() * REPRESENTATIVE_DTYPES.len(),
        "expected full primitive x dtype eval sweep"
    );

    let mut lowering_rows = Vec::new();
    for primitive in all_primitives() {
        match jaxpr_to_egraph(&make_lowering_jaxpr(*primitive)) {
            Ok(_) => lowering_rows.push(format!("{primitive:?} | lowering | ok | -")),
            Err(err) => {
                let rendered = err.to_string();
                assert!(
                    rendered.contains("unsupported") || rendered.contains("excluded"),
                    "lowering error should explain unsupported/excluded status: {rendered}"
                );
                assert_no_banned_substrings(&format!("egraph {:?}", primitive), &rendered);
                lowering_rows.push(format!(
                    "{primitive:?} | lowering | error | {}",
                    snippet(&rendered)
                ));
            }
        }
    }

    let display_cases = [
        EvalError::ArityMismatch {
            primitive: Primitive::Add,
            expected: 2,
            actual: 1,
        }
        .to_string(),
        EvalError::TypeMismatch {
            primitive: Primitive::Mul,
            detail: "lhs and rhs must share dtype",
        }
        .to_string(),
        EvalError::ShapeMismatch {
            primitive: Primitive::Add,
            left: Shape { dims: vec![2] },
            right: Shape { dims: vec![3] },
        }
        .to_string(),
        EvalError::Unsupported {
            primitive: Primitive::Conv,
            detail: "kernel rank mismatch".to_owned(),
        }
        .to_string(),
        EvalError::InvalidTensor(ValueError::ElementCountMismatch {
            shape: Shape { dims: vec![2, 2] },
            expected_count: 4,
            actual_count: 3,
        })
        .to_string(),
        EvalError::MaxIterationsExceeded {
            primitive: Primitive::While,
            max_iterations: 4,
        }
        .to_string(),
        EvalError::ShapeChanged {
            primitive: Primitive::While,
            detail: "carry element 0 changed shape".to_owned(),
        }
        .to_string(),
        EGraphLoweringError::UnsupportedPrimitive {
            primitive: Primitive::Reshape,
            reason: ExclusionReason::ShapeManipulation,
        }
        .to_string(),
        AdError::UnsupportedPrimitive(Primitive::Scan).to_string(),
        TransformExecutionError::PmapUnavailable.to_string(),
        ApiError::UnsupportedFeature {
            feature: "pmap".to_owned(),
            reason: "multi-device backend infrastructure is unavailable in V1".to_owned(),
        }
        .to_string(),
    ];

    for rendered in display_cases {
        assert_no_banned_substrings("display", &rendered);
    }

    let batchtrace_error = batch_eval_jaxpr(
        &make_invalid_batchtrace_sub_jaxpr_carrier(),
        &[
            BatchTracer::batched(Value::vector_i64(&[1, 2]).unwrap(), 0),
            BatchTracer::batched(Value::vector_i64(&[3, 4]).unwrap(), 0),
        ],
    )
    .expect_err("non-control-flow primitive carrying sub_jaxprs should be rejected")
    .to_string();
    assert!(
        batchtrace_error.contains("invalid BatchTrace IR"),
        "BatchTrace error should use permanent validation wording: {batchtrace_error}"
    );
    assert_no_banned_substrings("BatchTrace sub_jaxpr carrier", &batchtrace_error);

    println!("primitive | dtype | result | error_msg_snippet");
    for row in &eval_rows {
        println!("{row}");
    }
    for row in &lowering_rows {
        println!("{row}");
    }
}

#[test]
fn no_stub_regression_handles_complex_tensor_value_error_display() {
    let err = EvalError::InvalidTensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![2] },
            vec![Literal::from_complex128(1.0, 0.0)],
        )
        .unwrap_err(),
    );
    assert_no_banned_substrings("complex tensor invalid display", &err.to_string());
}

#[test]
fn no_stub_regression_covers_interpreter_invalid_sub_jaxpr_error() {
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(3)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![Jaxpr::new(vec![], vec![], vec![], vec![])],
        }],
    );

    let err = eval_jaxpr(&jaxpr, &[Value::scalar_i64(1), Value::scalar_i64(2)])
        .expect_err("non-control-flow sub_jaxprs should be rejected");
    let rendered = err.to_string();
    assert_no_banned_substrings("interpreter invalid sub_jaxpr display", &rendered);
    assert!(
        rendered.contains("sub_jaxprs are only valid for cond, scan, while, or switch"),
        "interpreter error should describe the permanent validation rule: {rendered}"
    );
}

#[test]
fn no_stub_source_code_markers_cover_workspace_sources() {
    let stub_patterns = [
        "unimplemented!",
        "todo!",
        r#"panic!("not impl"#,
        "stub",
        "placeholder",
        "mock",
    ];

    let mut matches = Vec::new();

    for path in workspace_source_files() {
        let source = std::fs::read_to_string(&path).expect("source file should be readable");
        for (line_no, line) in production_lines(&source) {
            let normalized = line.to_ascii_lowercase();
            for pattern in stub_patterns {
                if normalized.contains(pattern) && !source_marker_allowed(line) {
                    matches.push(format!("{}:{line_no}: {line}", path.display()));
                }
            }
        }
    }

    assert!(
        matches.is_empty(),
        "Found source-code stub markers outside cfg(test) modules:\n{}",
        matches.join("\n")
    );
}

#[test]
fn no_stub_source_not_implemented_wording_cover_workspace_sources() {
    let banned_phrases = ["not implemented", "not yet implemented"];
    let mut matches = Vec::new();

    for path in workspace_source_files() {
        let source = std::fs::read_to_string(&path).expect("source file should be readable");
        for (line_no, line) in production_lines(&source) {
            let normalized = line.to_ascii_lowercase();
            for phrase in banned_phrases {
                if normalized.contains(phrase) {
                    matches.push(format!("{}:{line_no}: {line}", path.display()));
                }
            }
        }
    }

    assert!(
        matches.is_empty(),
        "Found production-source not-implemented wording outside cfg(test) modules:\n{}",
        matches.join("\n")
    );
}

#[test]
fn no_stub_suspicious_default_returns() {
    let mut matches = Vec::new();

    for path in workspace_source_files() {
        let source = std::fs::read_to_string(&path).expect("source file should be readable");
        for (line_no, line) in production_lines(&source) {
            for pattern in SUSPICIOUS_EMPTY_SUCCESS_PATTERNS {
                if line.contains(pattern) && !suspicious_default_return_allowed(&path, line) {
                    matches.push(format!("{}:{line_no}: {line}", path.display()));
                }
            }
        }
    }

    assert!(
        matches.is_empty(),
        "Found suspicious default returns outside cfg(test) modules:\n{}\n\
         These patterns often indicate stub implementations.",
        matches.join("\n")
    );
}

#[test]
fn no_stub_ffi_callback_empty_outputs_use_named_helpers() {
    let path = workspace_root().join("crates/fj-ffi/src/callback.rs");
    let source = std::fs::read_to_string(&path).expect("FFI callback source should be readable");
    let matches = source
        .lines()
        .enumerate()
        .filter(|(_, line)| line.contains("Ok(vec![])"))
        .map(|(line_no, line)| format!("{}:{}: {line}", path.display(), line_no + 1))
        .collect::<Vec<_>>();

    assert!(
        matches.is_empty(),
        "FFI callback tests should route intentional empty outputs through a named helper:\n{}",
        matches.join("\n")
    );
}

#[test]
fn no_stub_docs_and_artifacts_scope_stale_status_claims() {
    let mut matches = Vec::new();

    for path in status_claim_audit_files() {
        let source = std::fs::read_to_string(&path).expect("audit file should be readable");
        for (line_no, line) in source.lines().enumerate() {
            let normalized = line.to_ascii_lowercase();
            if STALE_STATUS_MARKERS
                .iter()
                .any(|marker| normalized.contains(marker))
                && !stale_status_claim_allowed(&path, line)
            {
                matches.push(format!("{}:{}: {line}", path.display(), line_no + 1));
            }
        }
    }

    assert!(
        matches.is_empty(),
        "Found stale status claims without explicit historical, excluded-scope, or replay rationale:\n{}",
        matches.join("\n")
    );
}
