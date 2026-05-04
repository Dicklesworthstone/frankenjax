use crate::{Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, Transform, Value, VarId};
use proptest::prelude::*;
use smallvec::smallvec;
use std::collections::BTreeMap;

pub fn arb_literal() -> impl Strategy<Value = Literal> {
    prop_oneof![
        any::<i64>().prop_map(Literal::I64),
        any::<u32>().prop_map(Literal::U32),
        any::<u64>().prop_map(Literal::U64),
        any::<bool>().prop_map(Literal::Bool),
        any::<u16>().prop_map(Literal::BF16Bits),
        any::<u16>().prop_map(Literal::F16Bits),
        prop::num::f64::NORMAL.prop_map(Literal::from_f64),
        (prop::num::f32::NORMAL, prop::num::f32::NORMAL)
            .prop_map(|(re, im)| Literal::from_complex64(re, im)),
        (prop::num::f64::NORMAL, prop::num::f64::NORMAL)
            .prop_map(|(re, im)| Literal::from_complex128(re, im)),
    ]
}

pub fn arb_var_id() -> impl Strategy<Value = VarId> {
    (1..=1000u32).prop_map(VarId)
}

pub fn arb_atom() -> impl Strategy<Value = Atom> {
    prop_oneof![
        arb_var_id().prop_map(Atom::Var),
        arb_literal().prop_map(Atom::Lit),
    ]
}

pub fn arb_value() -> impl Strategy<Value = Value> {
    prop_oneof![
        any::<i64>().prop_map(Value::scalar_i64),
        any::<u32>().prop_map(Value::scalar_u32),
        any::<u64>().prop_map(Value::scalar_u64),
        prop::num::f32::NORMAL.prop_map(Value::scalar_bf16),
        prop::num::f32::NORMAL.prop_map(Value::scalar_f16),
        prop::num::f64::NORMAL.prop_map(Value::scalar_f64),
        any::<bool>().prop_map(Value::scalar_bool),
    ]
}

pub fn arb_shape() -> impl Strategy<Value = Shape> {
    prop_oneof![Just(Shape::scalar()), (1..=16u32).prop_map(Shape::vector),]
}

pub fn arb_primitive() -> impl Strategy<Value = Primitive> {
    prop_oneof![
        // Arithmetic
        Just(Primitive::Add),
        Just(Primitive::Sub),
        Just(Primitive::Mul),
        Just(Primitive::Neg),
        Just(Primitive::Abs),
        Just(Primitive::Max),
        Just(Primitive::Min),
        Just(Primitive::Pow),
        Just(Primitive::Exp),
        Just(Primitive::Log),
        Just(Primitive::Sqrt),
        Just(Primitive::Rsqrt),
        Just(Primitive::Floor),
        Just(Primitive::Ceil),
        Just(Primitive::Round),
        // Trigonometric
        Just(Primitive::Sin),
        Just(Primitive::Cos),
        Just(Primitive::Tan),
        Just(Primitive::Asin),
        Just(Primitive::Acos),
        Just(Primitive::Atan),
        // Hyperbolic
        Just(Primitive::Sinh),
        Just(Primitive::Cosh),
        Just(Primitive::Tanh),
        Just(Primitive::Asinh),
        Just(Primitive::Acosh),
        Just(Primitive::Atanh),
        // Additional math
        Just(Primitive::Expm1),
        Just(Primitive::Log1p),
        Just(Primitive::Sign),
        Just(Primitive::Square),
        Just(Primitive::Reciprocal),
        Just(Primitive::Logistic),
        Just(Primitive::Erf),
        Just(Primitive::Erfc),
        // Binary math
        Just(Primitive::Div),
        Just(Primitive::Rem),
        Just(Primitive::Atan2),
        // Complex number primitives
        Just(Primitive::Complex),
        Just(Primitive::Conj),
        Just(Primitive::Real),
        Just(Primitive::Imag),
        // Selection
        Just(Primitive::Select),
        // Dot product
        Just(Primitive::Dot),
        // Comparison
        Just(Primitive::Eq),
        Just(Primitive::Ne),
        Just(Primitive::Lt),
        Just(Primitive::Le),
        Just(Primitive::Gt),
        Just(Primitive::Ge),
        // Reduction
        Just(Primitive::ReduceSum),
        Just(Primitive::ReduceMax),
        Just(Primitive::ReduceMin),
        Just(Primitive::ReduceProd),
        Just(Primitive::ReduceAnd),
        Just(Primitive::ReduceOr),
        Just(Primitive::ReduceXor),
        // Shape manipulation
        Just(Primitive::Reshape),
        Just(Primitive::Slice),
        Just(Primitive::DynamicSlice),
        Just(Primitive::DynamicUpdateSlice),
        Just(Primitive::Gather),
        Just(Primitive::Scatter),
        Just(Primitive::Transpose),
        Just(Primitive::BroadcastInDim),
        Just(Primitive::Concatenate),
        Just(Primitive::Pad),
        Just(Primitive::Rev),
        Just(Primitive::Squeeze),
        Just(Primitive::Split),
        Just(Primitive::ExpandDims),
        // Special math
        Just(Primitive::Cbrt),
        Just(Primitive::Lgamma),
        Just(Primitive::Digamma),
        Just(Primitive::ErfInv),
        Just(Primitive::IsFinite),
        Just(Primitive::IntegerPow),
        Just(Primitive::Nextafter),
        // Clamping
        Just(Primitive::Clamp),
        // Index generation
        Just(Primitive::Iota),
        Just(Primitive::BroadcastedIota),
        // Utility operations
        Just(Primitive::Copy),
        Just(Primitive::BitcastConvertType),
        Just(Primitive::ReducePrecision),
        // Linear algebra
        Just(Primitive::Cholesky),
        Just(Primitive::Qr),
        Just(Primitive::Svd),
        Just(Primitive::TriangularSolve),
        Just(Primitive::Eigh),
        // FFT
        Just(Primitive::Fft),
        Just(Primitive::Ifft),
        Just(Primitive::Rfft),
        Just(Primitive::Irfft),
        // Encoding
        Just(Primitive::OneHot),
        // Cumulative
        Just(Primitive::Cumsum),
        Just(Primitive::Cumprod),
        // Sorting
        Just(Primitive::Sort),
        Just(Primitive::Argsort),
        // Convolution
        Just(Primitive::Conv),
        // Control flow
        Just(Primitive::Cond),
        Just(Primitive::Scan),
        Just(Primitive::While),
        Just(Primitive::Switch),
        // Bitwise
        Just(Primitive::BitwiseAnd),
        Just(Primitive::BitwiseOr),
        Just(Primitive::BitwiseXor),
        Just(Primitive::BitwiseNot),
        Just(Primitive::ShiftLeft),
        Just(Primitive::ShiftRightArithmetic),
        Just(Primitive::ShiftRightLogical),
        // Windowed reduction (pooling)
        Just(Primitive::ReduceWindow),
        // Integer intrinsics
        Just(Primitive::PopulationCount),
        Just(Primitive::CountLeadingZeros),
    ]
}

pub fn arb_transform() -> impl Strategy<Value = Transform> {
    prop_oneof![
        Just(Transform::Jit),
        Just(Transform::Grad),
        Just(Transform::Vmap),
    ]
}

pub fn arb_dtype() -> impl Strategy<Value = DType> {
    prop_oneof![
        Just(DType::F32),
        Just(DType::F64),
        Just(DType::I32),
        Just(DType::I64),
        Just(DType::U32),
        Just(DType::U64),
        Just(DType::BF16),
        Just(DType::F16),
        Just(DType::Bool),
        Just(DType::Complex64),
        Just(DType::Complex128),
    ]
}

/// Generate a valid single-equation Jaxpr for binary primitives (Add, Mul).
pub fn arb_binary_jaxpr() -> impl Strategy<Value = Jaxpr> {
    prop_oneof![Just(Primitive::Add), Just(Primitive::Mul),].prop_map(|prim| {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: prim,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TraceTransformLedger, verify_transform_composition};
    use proptest::strategy::ValueTree;

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(
            fj_test_utils::property_test_case_count()
        ))]
        #[test]
        fn prop_ir_fingerprint_determinism(prim in arb_primitive()) {
            let _seed = fj_test_utils::capture_proptest_seed();
            let jaxpr = Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: prim,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            );
            let fp1 = jaxpr.canonical_fingerprint().to_owned();
            let fp2 = jaxpr.canonical_fingerprint().to_owned();
            prop_assert_eq!(fp1, fp2);
        }

        #[test]
        fn prop_ttl_composition_signature_determinism(transform in arb_transform()) {
            let jaxpr = Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![]);
            let mut ttl = TraceTransformLedger::new(jaxpr);
            ttl.push_transform(transform, format!("evidence-{}", transform.as_str()));
            let sig1 = ttl.composition_signature();
            let sig2 = ttl.composition_signature();
            prop_assert_eq!(sig1, sig2);
        }

        #[test]
        fn prop_ttl_single_transform_composition_valid(transform in arb_transform()) {
            let jaxpr = Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![]);
            let mut ttl = TraceTransformLedger::new(jaxpr);
            ttl.push_transform(transform, format!("evidence-{}", transform.as_str()));
            let proof = verify_transform_composition(&ttl);
            prop_assert!(proof.is_ok());
        }

        #[test]
        fn prop_complex64_literal_roundtrip(re in prop::num::f32::NORMAL, im in prop::num::f32::NORMAL) {
            let lit = Literal::from_complex64(re, im);
            let (got_re, got_im) = lit.as_complex64().unwrap();
            prop_assert_eq!(re, got_re);
            prop_assert_eq!(im, got_im);
        }

        #[test]
        fn prop_complex128_literal_roundtrip(re in prop::num::f64::NORMAL, im in prop::num::f64::NORMAL) {
            let lit = Literal::from_complex128(re, im);
            let (got_re, got_im) = lit.as_complex128().unwrap();
            prop_assert_eq!(re, got_re);
            prop_assert_eq!(im, got_im);
        }

        #[test]
        fn prop_complex_serde_roundtrip(lit in arb_literal()) {
            let json = serde_json::to_string(&lit).unwrap();
            let deser: Literal = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(lit, deser);
        }

        #[test]
        fn prop_bfloat16_roundtrip_preserves_bits(bits in any::<u16>()) {
            let lit = Literal::BF16Bits(bits);
            let roundtrip = match Literal::from_bf16_f32(lit.as_bf16_f32().unwrap()) {
                Literal::BF16Bits(rt) => rt,
                _ => unreachable!("from_bf16_f32 must produce BF16Bits"),
            };
            let expected = half::bf16::from_f32(f32::from(half::bf16::from_bits(bits))).to_bits();
            prop_assert_eq!(expected, roundtrip);
        }

        #[test]
        fn prop_float16_finite_values_convert(value in any::<f32>()) {
            prop_assume!(value.is_finite() && value.abs() <= 65_504.0);
            let lit = Literal::from_f16_f32(value);
            let roundtrip = lit.as_f16_f32().unwrap();
            let expected = f32::from(half::f16::from_f32(value));
            prop_assert!(roundtrip.is_finite());
            prop_assert_eq!(roundtrip.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn test_shift_right_arithmetic_in_proptest() {
        let mut runner = proptest::test_runner::TestRunner::default();
        let strategy = arb_primitive();
        let found = (0..1_024).any(|_| {
            strategy
                .new_tree(&mut runner)
                .map(|tree| tree.current() == Primitive::ShiftRightArithmetic)
                .unwrap_or(false)
        });
        assert!(found, "arb_primitive should generate ShiftRightArithmetic");
    }

    #[test]
    fn test_shift_right_logical_in_proptest() {
        let mut runner = proptest::test_runner::TestRunner::default();
        let strategy = arb_primitive();
        let found = (0..1_024).any(|_| {
            strategy
                .new_tree(&mut runner)
                .map(|tree| tree.current() == Primitive::ShiftRightLogical)
                .unwrap_or(false)
        });
        assert!(found, "arb_primitive should generate ShiftRightLogical");
    }
}
