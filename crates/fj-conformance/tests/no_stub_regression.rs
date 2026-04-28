use fj_ad::AdError;
use fj_core::{
    Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, ValueError, VarId,
};
use fj_dispatch::batching::{BatchTracer, batch_eval_jaxpr};
use fj_egraph::{EGraphLoweringError, ExclusionReason, jaxpr_to_egraph};
use fj_interpreters::eval_jaxpr;
use fj_lax::{EvalError, eval_primitive};
use smallvec::smallvec;
use std::collections::BTreeMap;

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
    DType::I64,
    DType::U32,
    DType::U64,
    DType::F64,
    DType::Complex128,
    DType::Bool,
];

fn all_primitives() -> &'static [Primitive] {
    &[
        Primitive::Add,
        Primitive::Sub,
        Primitive::Mul,
        Primitive::Neg,
        Primitive::Abs,
        Primitive::Max,
        Primitive::Min,
        Primitive::Pow,
        Primitive::Exp,
        Primitive::Log,
        Primitive::Sqrt,
        Primitive::Rsqrt,
        Primitive::Floor,
        Primitive::Ceil,
        Primitive::Round,
        Primitive::Sin,
        Primitive::Cos,
        Primitive::Tan,
        Primitive::Asin,
        Primitive::Acos,
        Primitive::Atan,
        Primitive::Sinh,
        Primitive::Cosh,
        Primitive::Tanh,
        Primitive::Expm1,
        Primitive::Log1p,
        Primitive::Sign,
        Primitive::Square,
        Primitive::Reciprocal,
        Primitive::Logistic,
        Primitive::Erf,
        Primitive::Erfc,
        Primitive::Div,
        Primitive::Rem,
        Primitive::Atan2,
        Primitive::Complex,
        Primitive::Conj,
        Primitive::Real,
        Primitive::Imag,
        Primitive::Select,
        Primitive::Dot,
        Primitive::Eq,
        Primitive::Ne,
        Primitive::Lt,
        Primitive::Le,
        Primitive::Gt,
        Primitive::Ge,
        Primitive::ReduceSum,
        Primitive::ReduceMax,
        Primitive::ReduceMin,
        Primitive::ReduceProd,
        Primitive::ReduceAnd,
        Primitive::ReduceOr,
        Primitive::ReduceXor,
        Primitive::Reshape,
        Primitive::Slice,
        Primitive::DynamicSlice,
        Primitive::DynamicUpdateSlice,
        Primitive::Gather,
        Primitive::Scatter,
        Primitive::Transpose,
        Primitive::BroadcastInDim,
        Primitive::Concatenate,
        Primitive::Pad,
        Primitive::Rev,
        Primitive::Squeeze,
        Primitive::Split,
        Primitive::ExpandDims,
        Primitive::Cbrt,
        Primitive::Lgamma,
        Primitive::Digamma,
        Primitive::ErfInv,
        Primitive::IsFinite,
        Primitive::IntegerPow,
        Primitive::Nextafter,
        Primitive::Clamp,
        Primitive::Iota,
        Primitive::BroadcastedIota,
        Primitive::Copy,
        Primitive::BitcastConvertType,
        Primitive::ReducePrecision,
        Primitive::Cholesky,
        Primitive::Qr,
        Primitive::Svd,
        Primitive::TriangularSolve,
        Primitive::Eigh,
        Primitive::Fft,
        Primitive::Ifft,
        Primitive::Rfft,
        Primitive::Irfft,
        Primitive::OneHot,
        Primitive::Cumsum,
        Primitive::Cumprod,
        Primitive::Sort,
        Primitive::Argsort,
        Primitive::Conv,
        Primitive::Cond,
        Primitive::Scan,
        Primitive::While,
        Primitive::Switch,
        Primitive::BitwiseAnd,
        Primitive::BitwiseOr,
        Primitive::BitwiseXor,
        Primitive::BitwiseNot,
        Primitive::ShiftLeft,
        Primitive::ShiftRightArithmetic,
        Primitive::ShiftRightLogical,
        Primitive::ReduceWindow,
        Primitive::PopulationCount,
        Primitive::CountLeadingZeros,
    ]
}

fn primitive_arity(primitive: Primitive) -> usize {
    match primitive {
        Primitive::Add
        | Primitive::Sub
        | Primitive::Mul
        | Primitive::Max
        | Primitive::Min
        | Primitive::Pow
        | Primitive::Div
        | Primitive::Rem
        | Primitive::Atan2
        | Primitive::Complex
        | Primitive::Dot
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
        | Primitive::IntegerPow
        | Primitive::Nextafter
        | Primitive::TriangularSolve => 2,
        Primitive::Select | Primitive::Scatter | Primitive::Clamp | Primitive::Cond => 3,
        Primitive::Neg
        | Primitive::Abs
        | Primitive::Exp
        | Primitive::Log
        | Primitive::Sqrt
        | Primitive::Rsqrt
        | Primitive::Floor
        | Primitive::Ceil
        | Primitive::Round
        | Primitive::Sin
        | Primitive::Cos
        | Primitive::Tan
        | Primitive::Asin
        | Primitive::Acos
        | Primitive::Atan
        | Primitive::Sinh
        | Primitive::Cosh
        | Primitive::Tanh
        | Primitive::Expm1
        | Primitive::Log1p
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
        | Primitive::Conj
        | Primitive::Real
        | Primitive::Imag
        | Primitive::Cbrt
        | Primitive::Lgamma
        | Primitive::Digamma
        | Primitive::ErfInv
        | Primitive::IsFinite
        | Primitive::Cholesky
        | Primitive::Qr
        | Primitive::Svd
        | Primitive::Eigh
        | Primitive::Fft
        | Primitive::Ifft
        | Primitive::Rfft
        | Primitive::Irfft
        | Primitive::OneHot
        | Primitive::Cumsum
        | Primitive::Cumprod
        | Primitive::Sort
        | Primitive::Argsort
        | Primitive::Copy
        | Primitive::Rev
        | Primitive::Squeeze
        | Primitive::Split
        | Primitive::ExpandDims
        | Primitive::While => 3,
        Primitive::Iota | Primitive::BroadcastedIota => 0,
        Primitive::DynamicUpdateSlice => 3,
        Primitive::Conv => 2,
        Primitive::Scan => 2,
        Primitive::Switch => 3,
        Primitive::BitcastConvertType | Primitive::ReducePrecision => 1,
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
        DType::I64 => "i64",
        DType::U32 => "u32",
        DType::U64 => "u64",
        DType::F64 => "f64",
        DType::Complex128 => "complex128",
        DType::Bool => "bool",
        _ => "f64",
    }
}

fn representative_input(dtype: DType, slot: usize, primitive: Primitive) -> Value {
    if primitive == Primitive::While {
        return match (dtype, slot) {
            (DType::I64, 0) => Value::scalar_i64(3),
            (DType::I64, 1) => Value::scalar_i64(1),
            (DType::I64, 2) => Value::scalar_i64(0),
            (DType::U32, 0) => Value::scalar_u32(3),
            (DType::U32, 1) => Value::scalar_u32(1),
            (DType::U32, 2) => Value::scalar_u32(0),
            (DType::U64, 0) => Value::scalar_u64(3),
            (DType::U64, 1) => Value::scalar_u64(1),
            (DType::U64, 2) => Value::scalar_u64(0),
            (DType::F64, 0) => Value::scalar_f64(3.0),
            (DType::F64, 1) => Value::scalar_f64(1.0),
            (DType::F64, 2) => Value::scalar_f64(0.0),
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
        DType::I64 => Value::scalar_i64(2 + slot as i64),
        DType::U32 => Value::scalar_u32(2 + slot as u32),
        DType::U64 => Value::scalar_u64(2 + slot as u64),
        DType::F64 => Value::scalar_f64(1.5 + slot as f64),
        DType::Complex128 => Value::scalar_complex128(1.0 + slot as f64, 0.25 + slot as f64),
        DType::Bool => Value::scalar_bool(slot.is_multiple_of(2)),
        _ => unreachable!(),
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
    Jaxpr::new(
        invars,
        vec![],
        vec![outvar],
        vec![Equation {
            primitive,
            inputs,
            outputs: smallvec![outvar],
            params: BTreeMap::new(),
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
