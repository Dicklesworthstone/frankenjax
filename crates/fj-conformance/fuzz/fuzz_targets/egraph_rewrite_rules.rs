#![no_main]

mod common;

use common::ByteCursor;
use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, VarId};
use fj_egraph::{OptimizationConfig, algebraic_rules, jaxpr_to_egraph, optimize_jaxpr_with_config};
use libfuzzer_sys::fuzz_target;
use smallvec::smallvec;
use std::collections::BTreeMap;

const EGRAPH_PRIMITIVES: &[Primitive] = &[
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
    Primitive::Eq,
    Primitive::Ne,
    Primitive::Lt,
    Primitive::Le,
    Primitive::Gt,
    Primitive::Ge,
    Primitive::Cbrt,
    Primitive::IsFinite,
    Primitive::IntegerPow,
    Primitive::Clamp,
    Primitive::Copy,
    Primitive::BitwiseAnd,
    Primitive::BitwiseOr,
    Primitive::BitwiseXor,
    Primitive::BitwiseNot,
    Primitive::ShiftLeft,
    Primitive::ShiftRightArithmetic,
    Primitive::ShiftRightLogical,
    Primitive::PopulationCount,
    Primitive::CountLeadingZeros,
];

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = ByteCursor::new(data);
    let jaxpr = sample_egraph_jaxpr(&mut cursor);

    if jaxpr.validate_well_formed().is_err() {
        return;
    }

    let _ = algebraic_rules();
    let _ = jaxpr_to_egraph(&jaxpr);

    for config in [OptimizationConfig::safe(), OptimizationConfig::aggressive()] {
        let optimized = optimize_jaxpr_with_config(&jaxpr, &config);
        assert!(
            optimized.validate_well_formed().is_ok(),
            "optimized jaxpr must stay well formed"
        );
        let _ = optimized.canonical_fingerprint();

        let reoptimized = optimize_jaxpr_with_config(&optimized, &config);
        assert!(
            reoptimized.validate_well_formed().is_ok(),
            "reoptimized jaxpr must stay well formed"
        );
    }
});

fn sample_egraph_jaxpr(cursor: &mut ByteCursor<'_>) -> Jaxpr {
    let invars = vec![VarId(1), VarId(2)];
    let mut available = vec![
        Atom::Var(VarId(1)),
        Atom::Var(VarId(2)),
        Atom::Lit(lit_f64(-1.0)),
        Atom::Lit(lit_f64(0.0)),
        Atom::Lit(lit_f64(1.0)),
        Atom::Lit(lit_f64(2.0)),
    ];
    let mut equations = Vec::new();
    let mut next_var = 3_u32;

    let steps = 1 + cursor.take_usize(11);
    for _ in 0..steps {
        let primitive = sample_egraph_primitive(cursor);
        let arity = egraph_arity(primitive);
        let mut inputs = smallvec![];
        for _ in 0..arity {
            inputs.push(sample_atom(cursor, &available));
        }

        let outvar = VarId(next_var);
        next_var += 1;

        equations.push(Equation {
            primitive,
            inputs,
            outputs: smallvec![outvar],
            params: sample_egraph_params(cursor, primitive),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        });
        available.push(Atom::Var(outvar));
    }

    Jaxpr::new(invars, Vec::new(), vec![VarId(next_var - 1)], equations)
}

fn sample_egraph_primitive(cursor: &mut ByteCursor<'_>) -> Primitive {
    let idx = cursor.take_usize(EGRAPH_PRIMITIVES.len().saturating_sub(1));
    EGRAPH_PRIMITIVES[idx]
}

fn egraph_arity(primitive: Primitive) -> usize {
    match primitive {
        Primitive::Select | Primitive::Clamp => 3,
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
        | Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge
        | Primitive::BitwiseAnd
        | Primitive::BitwiseOr
        | Primitive::BitwiseXor
        | Primitive::ShiftLeft
        | Primitive::ShiftRightArithmetic
        | Primitive::ShiftRightLogical => 2,
        _ => 1,
    }
}

fn sample_egraph_params(
    cursor: &mut ByteCursor<'_>,
    primitive: Primitive,
) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    if primitive == Primitive::IntegerPow {
        let exponent = i32::from(cursor.take_u8() % 11) - 5;
        params.insert("exponent".to_owned(), exponent.to_string());
    }
    params
}

fn sample_atom(cursor: &mut ByteCursor<'_>, available: &[Atom]) -> Atom {
    let idx = cursor.take_usize(available.len().saturating_sub(1));
    available[idx].clone()
}

fn lit_f64(value: f64) -> Literal {
    Literal::F64Bits(value.to_bits())
}
