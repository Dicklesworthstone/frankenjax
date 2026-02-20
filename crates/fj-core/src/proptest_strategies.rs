use crate::{Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, Transform, Value, VarId};
use proptest::prelude::*;
use smallvec::smallvec;
use std::collections::BTreeMap;

pub fn arb_literal() -> impl Strategy<Value = Literal> {
    prop_oneof![
        any::<i64>().prop_map(Literal::I64),
        any::<bool>().prop_map(Literal::Bool),
        prop::num::f64::NORMAL.prop_map(Literal::from_f64),
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
        prop::num::f64::NORMAL.prop_map(Value::scalar_f64),
        any::<bool>().prop_map(Value::scalar_bool),
    ]
}

pub fn arb_shape() -> impl Strategy<Value = Shape> {
    prop_oneof![Just(Shape::scalar()), (1..=16u32).prop_map(Shape::vector),]
}

pub fn arb_primitive() -> impl Strategy<Value = Primitive> {
    prop_oneof![
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
        Just(Primitive::Sin),
        Just(Primitive::Cos),
        Just(Primitive::Tan),
        Just(Primitive::Asin),
        Just(Primitive::Acos),
        Just(Primitive::Atan),
        Just(Primitive::Sinh),
        Just(Primitive::Cosh),
        Just(Primitive::Tanh),
        Just(Primitive::Expm1),
        Just(Primitive::Log1p),
        Just(Primitive::Sign),
        Just(Primitive::Square),
        Just(Primitive::Reciprocal),
        Just(Primitive::Logistic),
        Just(Primitive::Erf),
        Just(Primitive::Erfc),
        Just(Primitive::Div),
        Just(Primitive::Rem),
        Just(Primitive::Atan2),
        Just(Primitive::Select),
        Just(Primitive::Dot),
        Just(Primitive::Eq),
        Just(Primitive::Ne),
        Just(Primitive::Lt),
        Just(Primitive::Le),
        Just(Primitive::Gt),
        Just(Primitive::Ge),
        Just(Primitive::ReduceSum),
        Just(Primitive::ReduceMax),
        Just(Primitive::ReduceMin),
        Just(Primitive::ReduceProd),
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
        Just(DType::Bool),
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
            }],
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TraceTransformLedger, verify_transform_composition};

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
            ttl.push_transform(transform, "evidence");
            let sig1 = ttl.composition_signature();
            let sig2 = ttl.composition_signature();
            prop_assert_eq!(sig1, sig2);
        }

        #[test]
        fn prop_ttl_single_transform_composition_valid(transform in arb_transform()) {
            let jaxpr = Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![]);
            let mut ttl = TraceTransformLedger::new(jaxpr);
            ttl.push_transform(transform, "evidence");
            let proof = verify_transform_composition(&ttl);
            prop_assert!(proof.is_ok());
        }
    }
}
