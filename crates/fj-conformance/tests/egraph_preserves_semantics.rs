//! E-graph optimization-preserving conformance gate.
//!
//! Verifies that e-graph algebraic rewrite rules preserve program semantics
//! by running Jaxpr programs both with and without optimization and comparing results.

use fj_core::{Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId};
use fj_egraph::{OptimizationConfig, optimize_jaxpr, optimize_jaxpr_with_config};
use fj_interpreters::eval_jaxpr;
use smallvec::smallvec;
use std::collections::BTreeMap;

fn make_unary_jaxpr(prim: Primitive) -> Jaxpr {
    Jaxpr::new(
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
    )
}

fn make_binary_jaxpr(prim: Primitive) -> Jaxpr {
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
}

/// Build: y = (x + x) which e-graph may rewrite to y = 2*x
fn make_add_self_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

/// Build: y = x * 1.0 (identity multiplication that e-graph should simplify)
fn make_mul_one_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![VarId(2)],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Mul,
            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(3)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

/// Build: y = x + 0.0 (identity addition)
fn make_add_zero_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![VarId(2)],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(3)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

/// Build: y = neg(neg(x)) which should simplify to y = x
fn make_double_neg_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(3)],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

/// Build: y = exp(log(x)) which should simplify to y = x
fn make_exp_log_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(3)],
        vec![
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn s_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

fn v_f64(data: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![data.len() as u32],
            },
            data.iter().map(|&v| Literal::from_f64(v)).collect(),
        )
        .unwrap(),
    )
}

fn assert_values_close(a: &Value, b: &Value, tol: f64, context: &str) {
    match (a, b) {
        (Value::Scalar(la), Value::Scalar(lb)) => {
            let va = la.as_f64().unwrap();
            let vb = lb.as_f64().unwrap();
            assert!(
                (va - vb).abs() < tol,
                "{context}: scalar mismatch: {va} vs {vb}"
            );
        }
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            assert_eq!(ta.shape.dims, tb.shape.dims, "{context}: shape mismatch");
            for (i, (ea, eb)) in ta.elements.iter().zip(tb.elements.iter()).enumerate() {
                let va = ea.as_f64().unwrap();
                let vb = eb.as_f64().unwrap();
                assert!(
                    (va - vb).abs() < tol,
                    "{context}[{i}]: element mismatch: {va} vs {vb}"
                );
            }
        }
        _ => {
            let same_kind = matches!(
                (a, b),
                (Value::Scalar(_), Value::Scalar(_)) | (Value::Tensor(_), Value::Tensor(_))
            );
            assert!(same_kind, "{context}: value kind mismatch");
        }
    }
}

/// Run a jaxpr with and without e-graph optimization and verify results match.
fn verify_optimization_preserves_semantics(
    jaxpr: &Jaxpr,
    args: &[Value],
    consts: &[Value],
    tol: f64,
    context: &str,
) {
    let original_result = if consts.is_empty() {
        eval_jaxpr(jaxpr, args).unwrap()
    } else {
        fj_interpreters::eval_jaxpr_with_consts(jaxpr, consts, args).unwrap()
    };

    let optimized = optimize_jaxpr(jaxpr);

    let optimized_result = if optimized.constvars.is_empty() {
        // Optimization may have eliminated constants (e.g., x*1 → x)
        eval_jaxpr(&optimized, args).unwrap()
    } else {
        fj_interpreters::eval_jaxpr_with_consts(&optimized, consts, args).unwrap()
    };

    assert_eq!(
        original_result.len(),
        optimized_result.len(),
        "{context}: output count mismatch"
    );
    for (i, (orig, opt)) in original_result
        .iter()
        .zip(optimized_result.iter())
        .enumerate()
    {
        assert_values_close(orig, opt, tol, &format!("{context} output[{i}]"));
    }
}

// ======================== Tests ========================

#[test]
fn egraph_preserves_add_scalar() {
    let jaxpr = make_binary_jaxpr(Primitive::Add);
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[s_f64(3.0), s_f64(4.0)],
        &[],
        1e-12,
        "add scalar",
    );
}

#[test]
fn egraph_preserves_mul_scalar() {
    let jaxpr = make_binary_jaxpr(Primitive::Mul);
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[s_f64(5.0), s_f64(6.0)],
        &[],
        1e-12,
        "mul scalar",
    );
}

#[test]
fn egraph_preserves_neg_scalar() {
    let jaxpr = make_unary_jaxpr(Primitive::Neg);
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(7.0)], &[], 1e-12, "neg scalar");
}

#[test]
fn egraph_preserves_exp_scalar() {
    let jaxpr = make_unary_jaxpr(Primitive::Exp);
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(2.0)], &[], 1e-12, "exp scalar");
}

#[test]
fn egraph_preserves_add_tensor() {
    let jaxpr = make_binary_jaxpr(Primitive::Add);
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[v_f64(&[1.0, 2.0, 3.0]), v_f64(&[4.0, 5.0, 6.0])],
        &[],
        1e-12,
        "add tensor",
    );
}

#[test]
fn egraph_preserves_add_self() {
    // x + x may be rewritten to 2*x
    let jaxpr = make_add_self_jaxpr();
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(42.0)], &[], 1e-12, "add self");
}

#[test]
fn egraph_preserves_mul_one_identity() {
    // x * 1 should produce same result as x
    let jaxpr = make_mul_one_jaxpr();
    let consts = vec![s_f64(1.0)];
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(99.0)], &consts, 1e-12, "mul one");
}

#[test]
fn egraph_preserves_add_zero_identity() {
    // x + 0 should produce same result as x
    let jaxpr = make_add_zero_jaxpr();
    let consts = vec![s_f64(0.0)];
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(123.0)], &consts, 1e-12, "add zero");
}

#[test]
fn egraph_preserves_double_neg() {
    // neg(neg(x)) should equal x
    let jaxpr = make_double_neg_jaxpr();
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(7.5)], &[], 1e-12, "double neg");
}

#[test]
fn egraph_preserves_exp_log_roundtrip() {
    // exp(log(x)) should equal x for positive x
    let jaxpr = make_exp_log_jaxpr();
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(3.5)], &[], 1e-10, "exp(log(x))");
}

#[test]
fn egraph_preserves_sin_scalar() {
    let jaxpr = make_unary_jaxpr(Primitive::Sin);
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(1.5)], &[], 1e-12, "sin scalar");
}

#[test]
fn egraph_preserves_cos_scalar() {
    let jaxpr = make_unary_jaxpr(Primitive::Cos);
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(1.5)], &[], 1e-12, "cos scalar");
}

// ======================== Multi-Equation Programs ========================

/// Build: y = sin(x); z = y * y (3 equations: sin, then square)
fn make_sin_squared_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(3)],
        vec![
            Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

/// Build: y = neg(x); z = neg(y); w = z + z (4 equations: cascade rewrites)
fn make_cascade_rewrite_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(3))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

/// Build: y = exp(x); z = log(y); w = z * z (exp(log) cancellation + squaring)
fn make_exp_log_square_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(3))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

/// Build: y = abs(x); z = abs(y) (abs idempotence)
fn make_abs_abs_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(3)],
        vec![
            Equation {
                primitive: Primitive::Abs,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Abs,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

/// Build: y = reciprocal(x); z = reciprocal(y) (reciprocal involution)
fn make_reciprocal_involution_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(3)],
        vec![
            Equation {
                primitive: Primitive::Reciprocal,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Reciprocal,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

#[test]
fn egraph_preserves_sin_squared() {
    // sin(x)^2: multi-equation with non-trivial computation
    let jaxpr = make_sin_squared_jaxpr();
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(1.2)], &[], 1e-12, "sin²(x)");
}

#[test]
fn egraph_preserves_sin_squared_tensor() {
    // sin(x)^2 on a tensor
    let jaxpr = make_sin_squared_jaxpr();
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[v_f64(&[0.5, 1.0, 1.5, 2.0])],
        &[],
        1e-12,
        "sin²(x) tensor",
    );
}

#[test]
fn egraph_preserves_cascade_rewrites() {
    // neg(neg(x)) + neg(neg(x)): should cascade to x + x = 2*x
    let jaxpr = make_cascade_rewrite_jaxpr();
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(7.0)], &[], 1e-12, "cascade rewrites");
}

#[test]
fn egraph_preserves_exp_log_square() {
    // (log(exp(x)))^2 = x^2: inverse pair cancellation + squaring
    let jaxpr = make_exp_log_square_jaxpr();
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(2.5)], &[], 1e-10, "exp-log-square");
}

#[test]
fn egraph_preserves_abs_idempotence() {
    // abs(abs(x)) = abs(x): idempotent rewrite
    let jaxpr = make_abs_abs_jaxpr();
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(-3.7)], &[], 1e-12, "abs idempotence");
}

#[test]
fn egraph_preserves_reciprocal_involution() {
    // reciprocal(reciprocal(x)) = x: involution rewrite
    let jaxpr = make_reciprocal_involution_jaxpr();
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[s_f64(4.0)],
        &[],
        1e-12,
        "reciprocal involution",
    );
}

#[test]
fn egraph_preserves_mul_tensor() {
    // Tensor multiplication preserves values through optimization
    let jaxpr = make_binary_jaxpr(Primitive::Mul);
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[v_f64(&[2.0, 3.0, 4.0]), v_f64(&[5.0, 6.0, 7.0])],
        &[],
        1e-12,
        "mul tensor",
    );
}

#[test]
fn egraph_preserves_neg_tensor() {
    // Tensor negation preserves values
    let jaxpr = make_unary_jaxpr(Primitive::Neg);
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[v_f64(&[1.0, -2.0, 3.0, -4.0])],
        &[],
        1e-12,
        "neg tensor",
    );
}

#[test]
fn egraph_preserves_sqrt_scalar() {
    let jaxpr = make_unary_jaxpr(Primitive::Sqrt);
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(16.0)], &[], 1e-12, "sqrt scalar");
}

#[test]
fn egraph_preserves_tanh_scalar() {
    let jaxpr = make_unary_jaxpr(Primitive::Tanh);
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(0.8)], &[], 1e-12, "tanh scalar");
}

// ======================== Multi-Output Programs ========================

/// Build: y1 = neg(x), y2 = abs(x) — two independent outputs from one input.
fn make_multi_output_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2), VarId(3)],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Abs,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

/// Build: t = neg(neg(x)), y1 = t, y2 = t + t  — shared intermediate, two outputs.
fn make_multi_output_shared_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(3), VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(3))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

/// Build: (q, y) where (q, r) = qr(x), t = r + 0, y = neg(neg(t)).
/// The QR itself is an opaque multi-output barrier, but the downstream
/// algebraic region should still optimize.
fn make_multi_output_barrier_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2), VarId(6)],
        vec![
            Equation {
                primitive: Primitive::Qr,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2), VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Lit(Literal::from_f64(0.0))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(4))],
                outputs: smallvec![VarId(5)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(5))],
                outputs: smallvec![VarId(6)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

#[test]
fn egraph_preserves_multi_output_independent() {
    let jaxpr = make_multi_output_jaxpr();
    let optimized = optimize_jaxpr(&jaxpr);
    assert_eq!(optimized.equations.len(), jaxpr.equations.len());
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[s_f64(-3.5)],
        &[],
        1e-12,
        "multi-output independent",
    );
}

#[test]
fn egraph_preserves_multi_output_shared() {
    let jaxpr = make_multi_output_shared_jaxpr();
    let optimized = optimize_jaxpr(&jaxpr);
    assert!(
        optimized.equations.len() < jaxpr.equations.len(),
        "shared multi-output jaxpr should shrink after optimization"
    );
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[s_f64(7.0)],
        &[],
        1e-12,
        "multi-output shared",
    );
}

#[test]
fn egraph_preserves_multi_output_barrier_regions() {
    let jaxpr = make_multi_output_barrier_jaxpr();
    let optimized = optimize_jaxpr(&jaxpr);
    assert!(
        optimized.equations.len() < jaxpr.equations.len(),
        "supported region after multi-output barrier should still optimize"
    );

    let matrix = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::from_f64(3.0),
                Literal::from_f64(1.0),
                Literal::from_f64(0.0),
                Literal::from_f64(2.0),
            ],
        )
        .expect("matrix"),
    );

    verify_optimization_preserves_semantics(
        &jaxpr,
        &[matrix],
        &[],
        1e-12,
        "multi-output barrier region",
    );
}

// ============================================================================
// Multi-equation chain optimization tests (frankenjax-fpe)
// ============================================================================

/// exp(sin(x) + cos(x)) — chain combining trig and exp
#[test]
fn egraph_preserves_exp_sin_plus_cos() {
    // v2 = sin(v1), v3 = cos(v1), v4 = add(v2, v3), v5 = exp(v4)
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(5)],
        vec![
            Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Cos,
                inputs: smallvec![Atom::Var(VarId(1))],
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
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec![Atom::Var(VarId(4))],
                outputs: smallvec![VarId(5)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    );
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[s_f64(1.0)],
        &[],
        1e-12,
        "exp(sin(x)+cos(x))",
    );
}

/// x * exp(-x^2) — Gaussian-like function
#[test]
fn egraph_preserves_gaussian_like() {
    // v2 = mul(v1,v1), v3 = neg(v2), v4 = exp(v3), v5 = mul(v1,v4)
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(5)],
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
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec![Atom::Var(VarId(3))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(4))],
                outputs: smallvec![VarId(5)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    );
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(1.5)], &[], 1e-12, "x*exp(-x^2)");
}

/// neg(neg(sin(x))) — double negation around a trig function (should cancel)
#[test]
fn egraph_preserves_double_neg_sin() {
    // v2 = sin(v1), v3 = neg(v2), v4 = neg(v3) => should simplify to sin(v1)
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(3))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    );
    let optimized = optimize_jaxpr(&jaxpr);
    assert!(
        optimized.equations.len() < jaxpr.equations.len(),
        "neg(neg(sin(x))) should simplify (got {} eqns, original {})",
        optimized.equations.len(),
        jaxpr.equations.len()
    );
    verify_optimization_preserves_semantics(&jaxpr, &[s_f64(2.0)], &[], 1e-12, "neg(neg(sin(x)))");
}

/// exp(log(x)) * exp(log(y)) — should simplify toward x*y
#[test]
fn egraph_preserves_exp_log_product() {
    // v3 = log(v1), v4 = exp(v3) => x; v5 = log(v2), v6 = exp(v5) => y; v7 = mul(v4,v6) => x*y
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(7)],
        vec![
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec![Atom::Var(VarId(3))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(5)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec![Atom::Var(VarId(5))],
                outputs: smallvec![VarId(6)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(6))],
                outputs: smallvec![VarId(7)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    );
    let optimized = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::aggressive());
    assert!(
        optimized.equations.len() < jaxpr.equations.len(),
        "aggressive exp(log(x))*exp(log(y)) should simplify (got {} eqns, original {})",
        optimized.equations.len(),
        jaxpr.equations.len()
    );
    let original_result = eval_jaxpr(&jaxpr, &[s_f64(3.0), s_f64(5.0)]).unwrap();
    let optimized_result = eval_jaxpr(&optimized, &[s_f64(3.0), s_f64(5.0)]).unwrap();
    assert_values_close(
        &original_result[0],
        &optimized_result[0],
        1e-10,
        "aggressive exp(log(x))*exp(log(y))",
    );
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[s_f64(3.0), s_f64(5.0)],
        &[],
        1e-10,
        "exp(log(x))*exp(log(y))",
    );
}

/// (a + b) - b — subtraction should cancel addition
#[test]
fn egraph_preserves_add_sub_cancel() {
    // v3 = add(v1,v2), v4 = sub(v3,v2) => should simplify to v1
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    );
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[s_f64(7.0), s_f64(3.0)],
        &[],
        1e-12,
        "(a+b)-b",
    );
}

/// sin(x)^2 * 1 + cos(x)^2 * 1 — Pythagorean identity components
#[test]
fn egraph_preserves_sin2_cos2_sum() {
    // v2=sin(v1), v3=mul(v2,v2), v4=cos(v1), v5=mul(v4,v4), v6=add(v3,v5)
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(6)],
        vec![
            Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Cos,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(4))],
                outputs: smallvec![VarId(5)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(5))],
                outputs: smallvec![VarId(6)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    );
    // sin^2(x) + cos^2(x) = 1, so optimization should simplify
    // Regardless of optimization, semantics must be preserved
    verify_optimization_preserves_semantics(
        &jaxpr,
        &[s_f64(1.23)],
        &[],
        1e-12,
        "sin^2(x)+cos^2(x)",
    );
    // Verify the result is 1.0
    let result = eval_jaxpr(&jaxpr, &[s_f64(1.23)]).unwrap();
    let val = result[0].as_f64_scalar().unwrap();
    assert!(
        (val - 1.0).abs() < 1e-12,
        "sin^2+cos^2 should equal 1.0, got {val}"
    );
}
