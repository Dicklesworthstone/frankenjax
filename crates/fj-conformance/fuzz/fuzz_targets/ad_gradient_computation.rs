#![no_main]

mod common;

use common::ByteCursor;
use fj_ad::{grad_jaxpr, grad_jaxpr_with_cotangent, jvp, value_and_grad_jaxpr};
use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, Value, VarId};
use libfuzzer_sys::fuzz_target;
use smallvec::smallvec;
use std::collections::BTreeMap;

const AD_PRIMITIVES: &[Primitive] = &[
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
    Primitive::Sin,
    Primitive::Cos,
    Primitive::Tan,
    Primitive::Sinh,
    Primitive::Cosh,
    Primitive::Tanh,
    Primitive::Expm1,
    Primitive::Log1p,
    Primitive::Square,
    Primitive::Reciprocal,
    Primitive::Logistic,
    Primitive::Erf,
    Primitive::Erfc,
    Primitive::Div,
    Primitive::Atan2,
    Primitive::Cbrt,
    Primitive::IntegerPow,
];

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = ByteCursor::new(data);
    let jaxpr = sample_scalar_jaxpr(&mut cursor);
    if jaxpr.validate_well_formed().is_err() {
        return;
    }

    let primals = vec![
        Value::scalar_f64(sample_nonzero_scalar(&mut cursor)),
        Value::scalar_f64(sample_nonzero_scalar(&mut cursor)),
    ];
    let tangents = vec![
        Value::scalar_f64(sample_scalar(&mut cursor)),
        Value::scalar_f64(sample_scalar(&mut cursor)),
    ];
    let cotangent = Value::scalar_f64(sample_nonzero_scalar(&mut cursor));

    let _ = grad_jaxpr(&jaxpr, &primals);
    let _ = value_and_grad_jaxpr(&jaxpr, &primals);
    let _ = grad_jaxpr_with_cotangent(&jaxpr, &primals, &cotangent);
    let _ = jvp(&jaxpr, &primals, &tangents);
});

fn sample_scalar_jaxpr(cursor: &mut ByteCursor<'_>) -> Jaxpr {
    let invars = vec![VarId(1), VarId(2)];
    let mut available = vec![
        Atom::Var(VarId(1)),
        Atom::Var(VarId(2)),
        Atom::Lit(lit_f64(-1.0)),
        Atom::Lit(lit_f64(-0.5)),
        Atom::Lit(lit_f64(0.5)),
        Atom::Lit(lit_f64(1.0)),
        Atom::Lit(lit_f64(2.0)),
    ];
    let mut equations = Vec::new();
    let mut next_var = 3_u32;

    let steps = 1 + cursor.take_usize(9);
    for _ in 0..steps {
        let primitive = sample_ad_primitive(cursor);
        let arity = ad_arity(primitive);
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
            params: sample_ad_params(cursor, primitive),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        });
        available.push(Atom::Var(outvar));
    }

    Jaxpr::new(invars, Vec::new(), vec![VarId(next_var - 1)], equations)
}

fn sample_ad_primitive(cursor: &mut ByteCursor<'_>) -> Primitive {
    let idx = cursor.take_usize(AD_PRIMITIVES.len().saturating_sub(1));
    AD_PRIMITIVES[idx]
}

fn ad_arity(primitive: Primitive) -> usize {
    match primitive {
        Primitive::Add
        | Primitive::Sub
        | Primitive::Mul
        | Primitive::Max
        | Primitive::Min
        | Primitive::Pow
        | Primitive::Div
        | Primitive::Atan2 => 2,
        _ => 1,
    }
}

fn sample_ad_params(cursor: &mut ByteCursor<'_>, primitive: Primitive) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    if primitive == Primitive::IntegerPow {
        let exponent = i32::from(cursor.take_u8() % 8) - 2;
        params.insert("exponent".to_owned(), exponent.to_string());
    }
    params
}

fn sample_atom(cursor: &mut ByteCursor<'_>, available: &[Atom]) -> Atom {
    let idx = cursor.take_usize(available.len().saturating_sub(1));
    available[idx].clone()
}

fn sample_scalar(cursor: &mut ByteCursor<'_>) -> f64 {
    let raw = cursor.take_u32() % 20_001;
    f64::from(raw) / 1000.0 - 10.0
}

fn sample_nonzero_scalar(cursor: &mut ByteCursor<'_>) -> f64 {
    let value = sample_scalar(cursor);
    if value.abs() < 0.125 {
        if cursor.take_bool() { 0.125 } else { -0.125 }
    } else {
        value
    }
}

fn lit_f64(value: f64) -> Literal {
    Literal::F64Bits(value.to_bits())
}
