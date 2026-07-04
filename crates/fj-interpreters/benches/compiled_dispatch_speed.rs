//! Compiled-dispatch (CompiledJaxpr) vs eager `eval_jaxpr` on interpreter-bound workloads.
//!
//! Scalar `Add` chains are DISPATCH-bound — the per-equation kernel is trivial, so the
//! cost is the interpreter tax (slot env setup, per-equation `eval_primitive` dispatch,
//! and `BTreeMap<String,String>` param handling). This isolates exactly what the dense
//! compiled plan targets, so it quantifies the existing `compile_jaxpr_for_repeated_eval`
//! win over per-call `eval_jaxpr`, and BASELINES the tensor-param-prescan lever
//! (frankenjax-6dfew): re-run this bench before/after that change to measure it.
//!
//! Bit-exactness of compiled-vs-eager is guarded by the unit test
//! `compiled_jaxpr_eval_matches_eager_eval_jaxpr`; this file only measures speed.
use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId};
use fj_interpreters::{DENSE_F64_DONATION_DISABLE, compile_jaxpr_for_repeated_eval, eval_jaxpr};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::atomic::Ordering;

/// `x -> x+1 -> x+2 -> ... ` : an n-equation Add chain. The added literal is `lit`, so
/// passing an I64 lit + scalar arg gives a pure-scalar chain, and an F64 lit + f64-vector
/// arg gives a small-TENSOR elementwise-broadcast chain (dense binary — the op NOT yet
/// pre-scanned in DenseEvalPlan, so it profiles the remaining dispatch gap for 6dfew).
fn build_chain_jaxpr(n: usize, lit: Literal) -> Jaxpr {
    let mut equations = Vec::with_capacity(n);
    for i in 0..n {
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![Atom::Var(VarId((i + 1) as u32)), Atom::Lit(lit)],
            outputs: smallvec::smallvec![VarId((i + 2) as u32)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
    }
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId((n + 1) as u32)],
        equations,
    )
}

/// Alternating chain `x = unary(x + lit)` repeated `n` times. These unary ops
/// currently break the large-tensor cheap-op fusion path, so the rows below
/// measure the exact xjbvr target before and after adding dense unary CheapOps.
fn build_add_unary_chain_jaxpr(n: usize, unary: Primitive, lit: Literal) -> Jaxpr {
    let mut equations = Vec::with_capacity(n * 2);
    let mut current = VarId(1);
    let mut next = 2_u32;
    for _ in 0..n {
        let added = VarId(next);
        next += 1;
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![Atom::Var(current), Atom::Lit(lit)],
            outputs: smallvec::smallvec![added],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        let rounded = VarId(next);
        next += 1;
        equations.push(Equation {
            primitive: unary,
            inputs: smallvec::smallvec![Atom::Var(added)],
            outputs: smallvec::smallvec![rounded],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        current = rounded;
    }
    Jaxpr::new(vec![VarId(1)], vec![], vec![current], equations)
}

/// A rank-2 broadcast chain: `m -> m+v -> (m+v)+v -> ...` where `m` is [R,C] and `v` is
/// a [C] row-broadcast vector (the bias-add pattern). Exercises the arena's broadcast path.
fn build_bcast_chain_jaxpr(n: usize) -> Jaxpr {
    let mut equations = Vec::with_capacity(n);
    for i in 0..n {
        let lhs = if i == 0 {
            VarId(1)
        } else {
            VarId((i + 2) as u32)
        };
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![Atom::Var(lhs), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId((i + 3) as u32)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
    }
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId((n + 2) as u32)],
        equations,
    )
}

fn build_softmax_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    let x = VarId(1);
    let max = VarId(2);
    let max_b = VarId(3);
    let shifted = VarId(4);
    let exp = VarId(5);
    let sum = VarId(6);
    let sum_b = VarId(7);
    let out = VarId(8);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::ReduceMax,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![max],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(max)],
                outputs: smallvec::smallvec![max_b],
                params: bcast.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(max_b)],
                outputs: smallvec::smallvec![shifted],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec::smallvec![Atom::Var(shifted)],
                outputs: smallvec::smallvec![exp],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(exp)],
                outputs: smallvec::smallvec![sum],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(sum)],
                outputs: smallvec::smallvec![sum_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(exp), Atom::Var(sum_b)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_log_softmax_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    let x = VarId(1);
    let max = VarId(2);
    let max_b = VarId(3);
    let shifted = VarId(4);
    let exp = VarId(5);
    let sum = VarId(6);
    let logv = VarId(7);
    let log_b = VarId(8);
    let out = VarId(9);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::ReduceMax,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![max],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(max)],
                outputs: smallvec::smallvec![max_b],
                params: bcast.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(max_b)],
                outputs: smallvec::smallvec![shifted],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec::smallvec![Atom::Var(shifted)],
                outputs: smallvec::smallvec![exp],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(exp)],
                outputs: smallvec::smallvec![sum],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec::smallvec![Atom::Var(sum)],
                outputs: smallvec::smallvec![logv],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(logv)],
                outputs: smallvec::smallvec![log_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(shifted), Atom::Var(log_b)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_gelu_erf_jaxpr() -> Jaxpr {
    let x = VarId(1);
    let arg = VarId(2);
    let e = VarId(3);
    let one_plus = VarId(4);
    let half_x = VarId(5);
    let out = VarId(6);
    let sqrt2 = Literal::from_f64(2.0_f64.sqrt());
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Lit(sqrt2)],
                outputs: smallvec::smallvec![arg],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Erf,
                inputs: smallvec::smallvec![Atom::Var(arg)],
                outputs: smallvec::smallvec![e],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                outputs: smallvec::smallvec![one_plus],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Lit(Literal::from_f64(0.5))],
                outputs: smallvec::smallvec![half_x],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(half_x), Atom::Var(one_plus)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_log_sigmoid_jaxpr() -> Jaxpr {
    let x = VarId(1);
    let nx = VarId(2);
    let e = VarId(3);
    let one_plus = VarId(4);
    let sp = VarId(5);
    let out = VarId(6);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![nx],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec::smallvec![Atom::Var(nx)],
                outputs: smallvec::smallvec![e],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                outputs: smallvec::smallvec![one_plus],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec::smallvec![Atom::Var(one_plus)],
                outputs: smallvec::smallvec![sp],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec::smallvec![Atom::Var(sp)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_softplus_jaxpr() -> Jaxpr {
    let x = VarId(1);
    let e = VarId(2);
    let one_plus = VarId(3);
    let out = VarId(4);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![e],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                outputs: smallvec::smallvec![one_plus],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec::smallvec![Atom::Var(one_plus)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_geglu_jaxpr() -> Jaxpr {
    let a = VarId(1);
    let b = VarId(2);
    let arg = VarId(3);
    let e = VarId(4);
    let one_plus = VarId(5);
    let half_a = VarId(6);
    let gelu_a = VarId(7);
    let out = VarId(8);
    let sqrt2 = Literal::from_f64(2.0_f64.sqrt());
    Jaxpr::new(
        vec![a, b],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(a), Atom::Lit(sqrt2)],
                outputs: smallvec::smallvec![arg],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Erf,
                inputs: smallvec::smallvec![Atom::Var(arg)],
                outputs: smallvec::smallvec![e],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                outputs: smallvec::smallvec![one_plus],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(a), Atom::Lit(Literal::from_f64(0.5))],
                outputs: smallvec::smallvec![half_a],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(half_a), Atom::Var(one_plus)],
                outputs: smallvec::smallvec![gelu_a],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(gelu_a), Atom::Var(b)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_swiglu_jaxpr() -> Jaxpr {
    let a = VarId(1);
    let b = VarId(2);
    let na = VarId(3);
    let e = VarId(4);
    let one_plus = VarId(5);
    let silu_a = VarId(6);
    let out = VarId(7);
    Jaxpr::new(
        vec![a, b],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec::smallvec![Atom::Var(a)],
                outputs: smallvec::smallvec![na],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec::smallvec![Atom::Var(na)],
                outputs: smallvec::smallvec![e],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                outputs: smallvec::smallvec![one_plus],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(a), Atom::Var(one_plus)],
                outputs: smallvec::smallvec![silu_a],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(silu_a), Atom::Var(b)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_silu_jaxpr() -> Jaxpr {
    let x = VarId(1);
    let nx = VarId(2);
    let e = VarId(3);
    let one_plus = VarId(4);
    let out = VarId(5);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![nx],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec::smallvec![Atom::Var(nx)],
                outputs: smallvec::smallvec![e],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                outputs: smallvec::smallvec![one_plus],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(one_plus)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_euclidean_distance_2d_jaxpr() -> Jaxpr {
    let a = VarId(1);
    let b = VarId(2);
    let diff = VarId(3);
    let sq = VarId(4);
    let s = VarId(5);
    let out = VarId(6);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    Jaxpr::new(
        vec![a, b],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(a), Atom::Var(b)],
                outputs: smallvec::smallvec![diff],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(diff), Atom::Var(diff)],
                outputs: smallvec::smallvec![sq],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(sq)],
                outputs: smallvec::smallvec![s],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sqrt,
                inputs: smallvec::smallvec![Atom::Var(s)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_mean_squared_error_2d_jaxpr(cols: usize) -> Jaxpr {
    let a = VarId(1);
    let b = VarId(2);
    let diff = VarId(3);
    let sq = VarId(4);
    let s = VarId(5);
    let out = VarId(6);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    Jaxpr::new(
        vec![a, b],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(a), Atom::Var(b)],
                outputs: smallvec::smallvec![diff],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(diff), Atom::Var(diff)],
                outputs: smallvec::smallvec![sq],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(sq)],
                outputs: smallvec::smallvec![s],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![
                    Atom::Var(s),
                    Atom::Lit(Literal::from_f64(cols as f64))
                ],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_manhattan_distance_2d_jaxpr() -> Jaxpr {
    let a = VarId(1);
    let b = VarId(2);
    let diff = VarId(3);
    let abs = VarId(4);
    let out = VarId(5);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    Jaxpr::new(
        vec![a, b],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(a), Atom::Var(b)],
                outputs: smallvec::smallvec![diff],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Abs,
                inputs: smallvec::smallvec![Atom::Var(diff)],
                outputs: smallvec::smallvec![abs],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(abs)],
                outputs: smallvec::smallvec![out],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_entropy_2d_jaxpr() -> Jaxpr {
    let p = VarId(1);
    let log_p = VarId(2);
    let weighted = VarId(3);
    let summed = VarId(4);
    let out = VarId(5);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    Jaxpr::new(
        vec![p],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec::smallvec![Atom::Var(p)],
                outputs: smallvec::smallvec![log_p],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(p), Atom::Var(log_p)],
                outputs: smallvec::smallvec![weighted],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(weighted)],
                outputs: smallvec::smallvec![summed],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec::smallvec![Atom::Var(summed)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_kl_divergence_2d_jaxpr() -> Jaxpr {
    let p = VarId(1);
    let q = VarId(2);
    let ratio = VarId(3);
    let log_ratio = VarId(4);
    let weighted = VarId(5);
    let out = VarId(6);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    Jaxpr::new(
        vec![p, q],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(p), Atom::Var(q)],
                outputs: smallvec::smallvec![ratio],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec::smallvec![Atom::Var(ratio)],
                outputs: smallvec::smallvec![log_ratio],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(p), Atom::Var(log_ratio)],
                outputs: smallvec::smallvec![weighted],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(weighted)],
                outputs: smallvec::smallvec![out],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_pearson_correlation_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    let a = VarId(1);
    let b = VarId(2);
    let sa = VarId(3);
    let ma = VarId(4);
    let ma_b = VarId(5);
    let ca = VarId(6);
    let sb = VarId(7);
    let mb = VarId(8);
    let mb_b = VarId(9);
    let cb = VarId(10);
    let cov_prod = VarId(11);
    let cov = VarId(12);
    let va_prod = VarId(13);
    let va = VarId(14);
    let na = VarId(15);
    let vb_prod = VarId(16);
    let vb = VarId(17);
    let nb = VarId(18);
    let denom = VarId(19);
    let out = VarId(20);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let n = Literal::from_f64(cols as f64);
    let unary = |prim: Primitive, input: VarId, o: VarId| Equation {
        primitive: prim,
        inputs: smallvec::smallvec![Atom::Var(input)],
        outputs: smallvec::smallvec![o],
        params: BTreeMap::new(),
        effects: vec![],
        sub_jaxprs: vec![],
    };
    let binary = |prim: Primitive, l: Atom, r: Atom, o: VarId| Equation {
        primitive: prim,
        inputs: smallvec::smallvec![l, r],
        outputs: smallvec::smallvec![o],
        params: BTreeMap::new(),
        effects: vec![],
        sub_jaxprs: vec![],
    };
    let reduce = |input: VarId, o: VarId| Equation {
        primitive: Primitive::ReduceSum,
        inputs: smallvec::smallvec![Atom::Var(input)],
        outputs: smallvec::smallvec![o],
        params: reduce_axis1.clone(),
        effects: vec![],
        sub_jaxprs: vec![],
    };
    let broadcast = |input: VarId, o: VarId| Equation {
        primitive: Primitive::BroadcastInDim,
        inputs: smallvec::smallvec![Atom::Var(input)],
        outputs: smallvec::smallvec![o],
        params: bcast.clone(),
        effects: vec![],
        sub_jaxprs: vec![],
    };
    Jaxpr::new(
        vec![a, b],
        vec![],
        vec![out],
        vec![
            reduce(a, sa),
            binary(Primitive::Div, Atom::Var(sa), Atom::Lit(n), ma),
            broadcast(ma, ma_b),
            binary(Primitive::Sub, Atom::Var(a), Atom::Var(ma_b), ca),
            reduce(b, sb),
            binary(Primitive::Div, Atom::Var(sb), Atom::Lit(n), mb),
            broadcast(mb, mb_b),
            binary(Primitive::Sub, Atom::Var(b), Atom::Var(mb_b), cb),
            binary(Primitive::Mul, Atom::Var(ca), Atom::Var(cb), cov_prod),
            reduce(cov_prod, cov),
            binary(Primitive::Mul, Atom::Var(ca), Atom::Var(ca), va_prod),
            reduce(va_prod, va),
            unary(Primitive::Sqrt, va, na),
            binary(Primitive::Mul, Atom::Var(cb), Atom::Var(cb), vb_prod),
            reduce(vb_prod, vb),
            unary(Primitive::Sqrt, vb, nb),
            binary(Primitive::Mul, Atom::Var(na), Atom::Var(nb), denom),
            binary(Primitive::Div, Atom::Var(cov), Atom::Var(denom), out),
        ],
    )
}

fn build_cosine_similarity_2d_jaxpr() -> Jaxpr {
    let a = VarId(1);
    let b = VarId(2);
    let ab = VarId(3);
    let dot = VarId(4);
    let a2 = VarId(5);
    let sa = VarId(6);
    let na = VarId(7);
    let b2 = VarId(8);
    let sb = VarId(9);
    let nb = VarId(10);
    let denom = VarId(11);
    let out = VarId(12);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let mul = |lhs: VarId, rhs: VarId, o: VarId| Equation {
        primitive: Primitive::Mul,
        inputs: smallvec::smallvec![Atom::Var(lhs), Atom::Var(rhs)],
        outputs: smallvec::smallvec![o],
        params: BTreeMap::new(),
        effects: vec![],
        sub_jaxprs: vec![],
    };
    let reduce = |input: VarId, o: VarId| Equation {
        primitive: Primitive::ReduceSum,
        inputs: smallvec::smallvec![Atom::Var(input)],
        outputs: smallvec::smallvec![o],
        params: reduce_axis1.clone(),
        effects: vec![],
        sub_jaxprs: vec![],
    };
    let sqrt = |input: VarId, o: VarId| Equation {
        primitive: Primitive::Sqrt,
        inputs: smallvec::smallvec![Atom::Var(input)],
        outputs: smallvec::smallvec![o],
        params: BTreeMap::new(),
        effects: vec![],
        sub_jaxprs: vec![],
    };
    Jaxpr::new(
        vec![a, b],
        vec![],
        vec![out],
        vec![
            mul(a, b, ab),
            reduce(ab, dot),
            mul(a, a, a2),
            reduce(a2, sa),
            sqrt(sa, na),
            mul(b, b, b2),
            reduce(b2, sb),
            sqrt(sb, nb),
            mul(na, nb, denom),
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(dot), Atom::Var(denom)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_var_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    let x = VarId(1);
    let s1 = VarId(2);
    let mean = VarId(3);
    let mean_b = VarId(4);
    let centered = VarId(5);
    let sq = VarId(6);
    let s2 = VarId(7);
    let out = VarId(8);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let n = Literal::from_f64(cols as f64);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![s1],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(s1), Atom::Lit(n)],
                outputs: smallvec::smallvec![mean],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(mean)],
                outputs: smallvec::smallvec![mean_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(mean_b)],
                outputs: smallvec::smallvec![centered],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(centered), Atom::Var(centered)],
                outputs: smallvec::smallvec![sq],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(sq)],
                outputs: smallvec::smallvec![s2],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(s2), Atom::Lit(n)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_std_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    // Variance graph but the final Div lands in `var`, then a trailing Sqrt yields the output.
    let x = VarId(1);
    let s1 = VarId(2);
    let mean = VarId(3);
    let mean_b = VarId(4);
    let centered = VarId(5);
    let sq = VarId(6);
    let s2 = VarId(7);
    let var = VarId(8);
    let out = VarId(9);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let n = Literal::from_f64(cols as f64);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![s1],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(s1), Atom::Lit(n)],
                outputs: smallvec::smallvec![mean],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(mean)],
                outputs: smallvec::smallvec![mean_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(mean_b)],
                outputs: smallvec::smallvec![centered],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(centered), Atom::Var(centered)],
                outputs: smallvec::smallvec![sq],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(sq)],
                outputs: smallvec::smallvec![s2],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(s2), Atom::Lit(n)],
                outputs: smallvec::smallvec![var],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sqrt,
                inputs: smallvec::smallvec![Atom::Var(var)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_skewness_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    let x = VarId(1);
    let s1 = VarId(2);
    let mean = VarId(3);
    let mean_b = VarId(4);
    let d = VarId(5);
    let d2 = VarId(6);
    let d3 = VarId(7);
    let sum2 = VarId(8);
    let m2 = VarId(9);
    let sum3 = VarId(10);
    let m3 = VarId(11);
    let s = VarId(12);
    let denom = VarId(13);
    let out = VarId(14);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let n = Literal::from_f64(cols as f64);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![s1],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(s1), Atom::Lit(n)],
                outputs: smallvec::smallvec![mean],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(mean)],
                outputs: smallvec::smallvec![mean_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(mean_b)],
                outputs: smallvec::smallvec![d],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(d), Atom::Var(d)],
                outputs: smallvec::smallvec![d2],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(d2), Atom::Var(d)],
                outputs: smallvec::smallvec![d3],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(d2)],
                outputs: smallvec::smallvec![sum2],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(sum2), Atom::Lit(n)],
                outputs: smallvec::smallvec![m2],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(d3)],
                outputs: smallvec::smallvec![sum3],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(sum3), Atom::Lit(n)],
                outputs: smallvec::smallvec![m3],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sqrt,
                inputs: smallvec::smallvec![Atom::Var(m2)],
                outputs: smallvec::smallvec![s],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(m2), Atom::Var(s)],
                outputs: smallvec::smallvec![denom],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(m3), Atom::Var(denom)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_kurtosis_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    let x = VarId(1);
    let s1 = VarId(2);
    let mean = VarId(3);
    let mean_b = VarId(4);
    let d = VarId(5);
    let d2 = VarId(6);
    let d4 = VarId(7);
    let sum2 = VarId(8);
    let m2 = VarId(9);
    let sum4 = VarId(10);
    let m4 = VarId(11);
    let denom = VarId(12);
    let out = VarId(13);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let n = Literal::from_f64(cols as f64);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![s1],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(s1), Atom::Lit(n)],
                outputs: smallvec::smallvec![mean],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(mean)],
                outputs: smallvec::smallvec![mean_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(mean_b)],
                outputs: smallvec::smallvec![d],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(d), Atom::Var(d)],
                outputs: smallvec::smallvec![d2],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(d2), Atom::Var(d2)],
                outputs: smallvec::smallvec![d4],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(d2)],
                outputs: smallvec::smallvec![sum2],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(sum2), Atom::Lit(n)],
                outputs: smallvec::smallvec![m2],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(d4)],
                outputs: smallvec::smallvec![sum4],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(sum4), Atom::Lit(n)],
                outputs: smallvec::smallvec![m4],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(m2), Atom::Var(m2)],
                outputs: smallvec::smallvec![denom],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(m4), Atom::Var(denom)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_covariance_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    let a = VarId(1);
    let b = VarId(2);
    let suma = VarId(3);
    let mean_a = VarId(4);
    let mean_a_b = VarId(5);
    let sumb = VarId(6);
    let mean_b = VarId(7);
    let mean_b_b = VarId(8);
    let ca = VarId(9);
    let cb = VarId(10);
    let prod = VarId(11);
    let s = VarId(12);
    let out = VarId(13);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let n = Literal::from_f64(cols as f64);
    Jaxpr::new(
        vec![a, b],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(a)],
                outputs: smallvec::smallvec![suma],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(suma), Atom::Lit(n)],
                outputs: smallvec::smallvec![mean_a],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(mean_a)],
                outputs: smallvec::smallvec![mean_a_b],
                params: bcast.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(b)],
                outputs: smallvec::smallvec![sumb],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(sumb), Atom::Lit(n)],
                outputs: smallvec::smallvec![mean_b],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(mean_b)],
                outputs: smallvec::smallvec![mean_b_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(a), Atom::Var(mean_a_b)],
                outputs: smallvec::smallvec![ca],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(b), Atom::Var(mean_b_b)],
                outputs: smallvec::smallvec![cb],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(ca), Atom::Var(cb)],
                outputs: smallvec::smallvec![prod],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(prod)],
                outputs: smallvec::smallvec![s],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![Atom::Var(s), Atom::Lit(n)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_l2_norm_2d_jaxpr() -> Jaxpr {
    let x = VarId(1);
    let sq = VarId(2);
    let s = VarId(3);
    let out = VarId(4);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(x)],
                outputs: smallvec::smallvec![sq],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(sq)],
                outputs: smallvec::smallvec![s],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sqrt,
                inputs: smallvec::smallvec![Atom::Var(s)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_logsumexp_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    let x = VarId(1);
    let max = VarId(2);
    let max_b = VarId(3);
    let shifted = VarId(4);
    let exp = VarId(5);
    let sum = VarId(6);
    let logv = VarId(7);
    let out = VarId(8);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::ReduceMax,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![max],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(max)],
                outputs: smallvec::smallvec![max_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(max_b)],
                outputs: smallvec::smallvec![shifted],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec::smallvec![Atom::Var(shifted)],
                outputs: smallvec::smallvec![exp],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(exp)],
                outputs: smallvec::smallvec![sum],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec::smallvec![Atom::Var(sum)],
                outputs: smallvec::smallvec![logv],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![Atom::Var(max), Atom::Var(logv)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_softmax_cross_entropy_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    let logits = VarId(1);
    let labels = VarId(2);
    let max = VarId(3);
    let max_b = VarId(4);
    let shifted = VarId(5);
    let exp = VarId(6);
    let sum = VarId(7);
    let logv = VarId(8);
    let log_b = VarId(9);
    let ls = VarId(10);
    let prod = VarId(11);
    let ce_sum = VarId(12);
    let out = VarId(13);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    Jaxpr::new(
        vec![logits, labels],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::ReduceMax,
                inputs: smallvec::smallvec![Atom::Var(logits)],
                outputs: smallvec::smallvec![max],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(max)],
                outputs: smallvec::smallvec![max_b],
                params: bcast.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(logits), Atom::Var(max_b)],
                outputs: smallvec::smallvec![shifted],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec::smallvec![Atom::Var(shifted)],
                outputs: smallvec::smallvec![exp],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(exp)],
                outputs: smallvec::smallvec![sum],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec::smallvec![Atom::Var(sum)],
                outputs: smallvec::smallvec![logv],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(logv)],
                outputs: smallvec::smallvec![log_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(shifted), Atom::Var(log_b)],
                outputs: smallvec::smallvec![ls],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(labels), Atom::Var(ls)],
                outputs: smallvec::smallvec![prod],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(prod)],
                outputs: smallvec::smallvec![ce_sum],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec::smallvec![Atom::Var(ce_sum)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_rms_norm_2d_jaxpr(rows: usize, cols: usize, epsilon: f64) -> Jaxpr {
    let x = VarId(1);
    let squared = VarId(2);
    let sq_sum = VarId(3);
    let ms = VarId(4);
    let ms_eps = VarId(5);
    let inv = VarId(6);
    let inv_b = VarId(7);
    let out = VarId(8);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::Square,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![squared],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(squared)],
                outputs: smallvec::smallvec![sq_sum],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![
                    Atom::Var(sq_sum),
                    Atom::Lit(Literal::from_f64(cols as f64))
                ],
                outputs: smallvec::smallvec![ms],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![Atom::Var(ms), Atom::Lit(Literal::from_f64(epsilon))],
                outputs: smallvec::smallvec![ms_eps],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Rsqrt,
                inputs: smallvec::smallvec![Atom::Var(ms_eps)],
                outputs: smallvec::smallvec![inv],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(inv)],
                outputs: smallvec::smallvec![inv_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(inv_b)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn build_layer_norm_2d_jaxpr(rows: usize, cols: usize, epsilon: f64) -> Jaxpr {
    let x = VarId(1);
    let sum = VarId(2);
    let mean = VarId(3);
    let mean_b = VarId(4);
    let centered = VarId(5);
    let squared = VarId(6);
    let var_sum = VarId(7);
    let variance = VarId(8);
    let var_eps = VarId(9);
    let inv = VarId(10);
    let inv_b = VarId(11);
    let out = VarId(12);
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![sum],
                params: reduce_axis1.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![
                    Atom::Var(sum),
                    Atom::Lit(Literal::from_f64(cols as f64))
                ],
                outputs: smallvec::smallvec![mean],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(mean)],
                outputs: smallvec::smallvec![mean_b],
                params: bcast.clone(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Sub,
                inputs: smallvec::smallvec![Atom::Var(x), Atom::Var(mean_b)],
                outputs: smallvec::smallvec![centered],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Square,
                inputs: smallvec::smallvec![Atom::Var(centered)],
                outputs: smallvec::smallvec![squared],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec::smallvec![Atom::Var(squared)],
                outputs: smallvec::smallvec![var_sum],
                params: reduce_axis1,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Div,
                inputs: smallvec::smallvec![
                    Atom::Var(var_sum),
                    Atom::Lit(Literal::from_f64(cols as f64))
                ],
                outputs: smallvec::smallvec![variance],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec::smallvec![
                    Atom::Var(variance),
                    Atom::Lit(Literal::from_f64(epsilon))
                ],
                outputs: smallvec::smallvec![var_eps],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Rsqrt,
                inputs: smallvec::smallvec![Atom::Var(var_eps)],
                outputs: smallvec::smallvec![inv],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(inv)],
                outputs: smallvec::smallvec![inv_b],
                params: bcast,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec::smallvec![Atom::Var(centered), Atom::Var(inv_b)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn eval_softmax_2d_decomposed(input: &Value, rows: usize, cols: usize) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let max = eval_primitive(
        Primitive::ReduceMax,
        std::slice::from_ref(input),
        &reduce_axis1,
    )
    .expect("reduce max");
    let max_b = eval_primitive(Primitive::BroadcastInDim, &[max], &bcast).expect("broadcast max");
    let shifted =
        eval_primitive(Primitive::Sub, &[input.clone(), max_b], &empty).expect("subtract max");
    let exp = eval_primitive(Primitive::Exp, std::slice::from_ref(&shifted), &empty).expect("exp");
    let sum = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&exp),
        &reduce_axis1,
    )
    .expect("reduce sum");
    let sum_b = eval_primitive(Primitive::BroadcastInDim, &[sum], &bcast).expect("broadcast sum");
    eval_primitive(Primitive::Div, &[exp, sum_b], &empty).expect("divide")
}

fn eval_log_softmax_2d_decomposed(input: &Value, rows: usize, cols: usize) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let max = eval_primitive(
        Primitive::ReduceMax,
        std::slice::from_ref(input),
        &reduce_axis1,
    )
    .expect("reduce max");
    let max_b = eval_primitive(Primitive::BroadcastInDim, &[max], &bcast).expect("broadcast max");
    let shifted =
        eval_primitive(Primitive::Sub, &[input.clone(), max_b], &empty).expect("subtract max");
    let exp = eval_primitive(Primitive::Exp, std::slice::from_ref(&shifted), &empty).expect("exp");
    let sum = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&exp),
        &reduce_axis1,
    )
    .expect("reduce sum");
    let logv = eval_primitive(Primitive::Log, std::slice::from_ref(&sum), &empty).expect("log");
    let log_b = eval_primitive(Primitive::BroadcastInDim, &[logv], &bcast).expect("broadcast log");
    eval_primitive(Primitive::Sub, &[shifted, log_b], &empty).expect("subtract log")
}

fn eval_gelu_erf_decomposed(input: &Value) -> Value {
    let empty = BTreeMap::new();
    let arg = eval_primitive(
        Primitive::Div,
        &[input.clone(), Value::scalar_f64(2.0_f64.sqrt())],
        &empty,
    )
    .expect("div");
    let e = eval_primitive(Primitive::Erf, std::slice::from_ref(&arg), &empty).expect("erf");
    let one_plus =
        eval_primitive(Primitive::Add, &[Value::scalar_f64(1.0), e], &empty).expect("add one");
    let half_x = eval_primitive(
        Primitive::Mul,
        &[input.clone(), Value::scalar_f64(0.5)],
        &empty,
    )
    .expect("half");
    eval_primitive(Primitive::Mul, &[half_x, one_plus], &empty).expect("scale")
}

fn eval_log_sigmoid_decomposed(input: &Value) -> Value {
    let empty = BTreeMap::new();
    let nx = eval_primitive(Primitive::Neg, std::slice::from_ref(input), &empty).expect("neg");
    let e = eval_primitive(Primitive::Exp, std::slice::from_ref(&nx), &empty).expect("exp");
    let one_plus =
        eval_primitive(Primitive::Add, &[Value::scalar_f64(1.0), e], &empty).expect("add one");
    let sp = eval_primitive(Primitive::Log, std::slice::from_ref(&one_plus), &empty).expect("log");
    eval_primitive(Primitive::Neg, std::slice::from_ref(&sp), &empty).expect("neg out")
}

fn eval_silu_decomposed(input: &Value) -> Value {
    let empty = BTreeMap::new();
    let nx = eval_primitive(Primitive::Neg, std::slice::from_ref(input), &empty).expect("neg");
    let e = eval_primitive(Primitive::Exp, std::slice::from_ref(&nx), &empty).expect("exp");
    let one_plus =
        eval_primitive(Primitive::Add, &[Value::scalar_f64(1.0), e], &empty).expect("add one");
    eval_primitive(Primitive::Div, &[input.clone(), one_plus], &empty).expect("div")
}

fn eval_softplus_decomposed(input: &Value) -> Value {
    let empty = BTreeMap::new();
    let e = eval_primitive(Primitive::Exp, std::slice::from_ref(input), &empty).expect("exp");
    let one_plus =
        eval_primitive(Primitive::Add, &[Value::scalar_f64(1.0), e], &empty).expect("add one");
    eval_primitive(Primitive::Log, std::slice::from_ref(&one_plus), &empty).expect("log")
}

fn eval_swiglu_decomposed(a: &Value, b: &Value) -> Value {
    let empty = BTreeMap::new();
    let na = eval_primitive(Primitive::Neg, std::slice::from_ref(a), &empty).expect("neg");
    let e = eval_primitive(Primitive::Exp, std::slice::from_ref(&na), &empty).expect("exp");
    let one_plus =
        eval_primitive(Primitive::Add, &[Value::scalar_f64(1.0), e], &empty).expect("add one");
    let silu_a = eval_primitive(Primitive::Div, &[a.clone(), one_plus], &empty).expect("div");
    eval_primitive(Primitive::Mul, &[silu_a, b.clone()], &empty).expect("mul")
}

fn eval_geglu_decomposed(a: &Value, b: &Value) -> Value {
    let empty = BTreeMap::new();
    let arg = eval_primitive(
        Primitive::Div,
        &[a.clone(), Value::scalar_f64(2.0_f64.sqrt())],
        &empty,
    )
    .expect("div");
    let e = eval_primitive(Primitive::Erf, std::slice::from_ref(&arg), &empty).expect("erf");
    let one_plus =
        eval_primitive(Primitive::Add, &[Value::scalar_f64(1.0), e], &empty).expect("add one");
    let half_a =
        eval_primitive(Primitive::Mul, &[a.clone(), Value::scalar_f64(0.5)], &empty).expect("half");
    let gelu_a = eval_primitive(Primitive::Mul, &[half_a, one_plus], &empty).expect("gelu");
    eval_primitive(Primitive::Mul, &[gelu_a, b.clone()], &empty).expect("gate")
}

fn eval_euclidean_distance_2d_decomposed(a: &Value, b: &Value) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let empty = BTreeMap::new();
    let diff = eval_primitive(Primitive::Sub, &[a.clone(), b.clone()], &empty).expect("a-b");
    let sq = eval_primitive(Primitive::Mul, &[diff.clone(), diff], &empty).expect("square");
    let s = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&sq),
        &reduce_axis1,
    )
    .expect("sum sq");
    eval_primitive(Primitive::Sqrt, std::slice::from_ref(&s), &empty).expect("sqrt")
}

fn eval_mean_squared_error_2d_decomposed(a: &Value, b: &Value, cols: usize) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let empty = BTreeMap::new();
    let diff = eval_primitive(Primitive::Sub, &[a.clone(), b.clone()], &empty).expect("a-b");
    let sq = eval_primitive(Primitive::Mul, &[diff.clone(), diff], &empty).expect("square");
    let s = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&sq),
        &reduce_axis1,
    )
    .expect("sum sq");
    eval_primitive(Primitive::Div, &[s, Value::scalar_f64(cols as f64)], &empty).expect("mse")
}

fn eval_manhattan_distance_2d_decomposed(a: &Value, b: &Value) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let empty = BTreeMap::new();
    let diff = eval_primitive(Primitive::Sub, &[a.clone(), b.clone()], &empty).expect("a-b");
    let abs = eval_primitive(Primitive::Abs, std::slice::from_ref(&diff), &empty).expect("abs");
    eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&abs),
        &reduce_axis1,
    )
    .expect("sum abs")
}

fn eval_entropy_2d_decomposed(p: &Value) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let empty = BTreeMap::new();
    let log_p = eval_primitive(Primitive::Log, std::slice::from_ref(p), &empty).expect("log p");
    let weighted = eval_primitive(Primitive::Mul, &[p.clone(), log_p], &empty).expect("p*log");
    let summed = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&weighted), &reduce_axis1)
        .expect("sum");
    eval_primitive(Primitive::Neg, std::slice::from_ref(&summed), &empty).expect("neg")
}

fn eval_kl_divergence_2d_decomposed(p: &Value, q: &Value) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let empty = BTreeMap::new();
    let ratio = eval_primitive(Primitive::Div, &[p.clone(), q.clone()], &empty).expect("p/q");
    let log_ratio =
        eval_primitive(Primitive::Log, std::slice::from_ref(&ratio), &empty).expect("log");
    let weighted = eval_primitive(Primitive::Mul, &[p.clone(), log_ratio], &empty).expect("p*log");
    eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&weighted),
        &reduce_axis1,
    )
    .expect("kl sum")
}

fn eval_pearson_correlation_2d_decomposed(a: &Value, b: &Value, rows: usize, cols: usize) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let n = Value::scalar_f64(cols as f64);
    let sa = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(a), &reduce_axis1)
        .expect("sum a");
    let ma = eval_primitive(Primitive::Div, &[sa, n.clone()], &empty).expect("mean a");
    let ma_b = eval_primitive(Primitive::BroadcastInDim, &[ma], &bcast).expect("bcast ma");
    let ca = eval_primitive(Primitive::Sub, &[a.clone(), ma_b], &empty).expect("center a");
    let sb = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(b), &reduce_axis1)
        .expect("sum b");
    let mb = eval_primitive(Primitive::Div, &[sb, n], &empty).expect("mean b");
    let mb_b = eval_primitive(Primitive::BroadcastInDim, &[mb], &bcast).expect("bcast mb");
    let cb = eval_primitive(Primitive::Sub, &[b.clone(), mb_b], &empty).expect("center b");
    let cov_prod =
        eval_primitive(Primitive::Mul, &[ca.clone(), cb.clone()], &empty).expect("ca*cb");
    let cov = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&cov_prod),
        &reduce_axis1,
    )
    .expect("cov");
    let va_prod = eval_primitive(Primitive::Mul, &[ca.clone(), ca], &empty).expect("ca*ca");
    let va = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&va_prod),
        &reduce_axis1,
    )
    .expect("var a");
    let na = eval_primitive(Primitive::Sqrt, std::slice::from_ref(&va), &empty).expect("norm a");
    let vb_prod = eval_primitive(Primitive::Mul, &[cb.clone(), cb], &empty).expect("cb*cb");
    let vb = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&vb_prod),
        &reduce_axis1,
    )
    .expect("var b");
    let nb = eval_primitive(Primitive::Sqrt, std::slice::from_ref(&vb), &empty).expect("norm b");
    let denom = eval_primitive(Primitive::Mul, &[na, nb], &empty).expect("denom");
    eval_primitive(Primitive::Div, &[cov, denom], &empty).expect("pearson")
}

fn eval_cosine_similarity_2d_decomposed(a: &Value, b: &Value) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let empty = BTreeMap::new();
    let ab = eval_primitive(Primitive::Mul, &[a.clone(), b.clone()], &empty).expect("a*b");
    let dot = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&ab),
        &reduce_axis1,
    )
    .expect("dot");
    let a2 = eval_primitive(Primitive::Mul, &[a.clone(), a.clone()], &empty).expect("a*a");
    let sa = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&a2),
        &reduce_axis1,
    )
    .expect("sum a2");
    let na = eval_primitive(Primitive::Sqrt, std::slice::from_ref(&sa), &empty).expect("norm a");
    let b2 = eval_primitive(Primitive::Mul, &[b.clone(), b.clone()], &empty).expect("b*b");
    let sb = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&b2),
        &reduce_axis1,
    )
    .expect("sum b2");
    let nb = eval_primitive(Primitive::Sqrt, std::slice::from_ref(&sb), &empty).expect("norm b");
    let denom = eval_primitive(Primitive::Mul, &[na, nb], &empty).expect("denom");
    eval_primitive(Primitive::Div, &[dot, denom], &empty).expect("cosine")
}

fn eval_var_2d_decomposed(input: &Value, rows: usize, cols: usize) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let n = Value::scalar_f64(cols as f64);
    let s1 = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(input),
        &reduce_axis1,
    )
    .expect("reduce sum");
    let mean = eval_primitive(Primitive::Div, &[s1, n.clone()], &empty).expect("mean");
    let mean_b =
        eval_primitive(Primitive::BroadcastInDim, &[mean], &bcast).expect("broadcast mean");
    let centered =
        eval_primitive(Primitive::Sub, &[input.clone(), mean_b], &empty).expect("subtract mean");
    let sq = eval_primitive(Primitive::Mul, &[centered.clone(), centered], &empty).expect("square");
    let s2 = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&sq),
        &reduce_axis1,
    )
    .expect("reduce sum sq");
    eval_primitive(Primitive::Div, &[s2, n], &empty).expect("var")
}

fn eval_std_2d_decomposed(input: &Value, rows: usize, cols: usize) -> Value {
    let var = eval_var_2d_decomposed(input, rows, cols);
    let empty = BTreeMap::new();
    eval_primitive(Primitive::Sqrt, std::slice::from_ref(&var), &empty).expect("std")
}

fn eval_skewness_2d_decomposed(input: &Value, rows: usize, cols: usize) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let n = Value::scalar_f64(cols as f64);
    let s1 = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(input),
        &reduce_axis1,
    )
    .expect("reduce sum");
    let mean = eval_primitive(Primitive::Div, &[s1, n.clone()], &empty).expect("mean");
    let mean_b =
        eval_primitive(Primitive::BroadcastInDim, &[mean], &bcast).expect("broadcast mean");
    let d = eval_primitive(Primitive::Sub, &[input.clone(), mean_b], &empty).expect("center");
    let d2 = eval_primitive(Primitive::Mul, &[d.clone(), d.clone()], &empty).expect("square");
    let d3 = eval_primitive(Primitive::Mul, &[d2.clone(), d], &empty).expect("cube");
    let sum2 = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&d2), &reduce_axis1)
        .expect("reduce sum sq");
    let m2 = eval_primitive(Primitive::Div, &[sum2, n.clone()], &empty).expect("m2");
    let sum3 = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&d3), &reduce_axis1)
        .expect("reduce sum cube");
    let m3 = eval_primitive(Primitive::Div, &[sum3, n], &empty).expect("m3");
    let s = eval_primitive(Primitive::Sqrt, std::slice::from_ref(&m2), &empty).expect("sqrt m2");
    let denom = eval_primitive(Primitive::Mul, &[m2, s], &empty).expect("denom");
    eval_primitive(Primitive::Div, &[m3, denom], &empty).expect("skewness")
}

fn eval_kurtosis_2d_decomposed(input: &Value, rows: usize, cols: usize) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let n = Value::scalar_f64(cols as f64);
    let s1 = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(input),
        &reduce_axis1,
    )
    .expect("reduce sum");
    let mean = eval_primitive(Primitive::Div, &[s1, n.clone()], &empty).expect("mean");
    let mean_b =
        eval_primitive(Primitive::BroadcastInDim, &[mean], &bcast).expect("broadcast mean");
    let d = eval_primitive(Primitive::Sub, &[input.clone(), mean_b], &empty).expect("center");
    let d2 = eval_primitive(Primitive::Mul, &[d.clone(), d], &empty).expect("square");
    let d4 = eval_primitive(Primitive::Mul, &[d2.clone(), d2.clone()], &empty).expect("fourth");
    let sum2 = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&d2), &reduce_axis1)
        .expect("reduce sum sq");
    let m2 = eval_primitive(Primitive::Div, &[sum2, n.clone()], &empty).expect("m2");
    let sum4 = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&d4), &reduce_axis1)
        .expect("reduce sum fourth");
    let m4 = eval_primitive(Primitive::Div, &[sum4, n], &empty).expect("m4");
    let denom = eval_primitive(Primitive::Mul, &[m2.clone(), m2], &empty).expect("denom");
    eval_primitive(Primitive::Div, &[m4, denom], &empty).expect("kurtosis")
}

fn eval_covariance_2d_decomposed(a: &Value, b: &Value, rows: usize, cols: usize) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let n = Value::scalar_f64(cols as f64);
    let suma = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(a), &reduce_axis1)
        .expect("reduce sum a");
    let mean_a = eval_primitive(Primitive::Div, &[suma, n.clone()], &empty).expect("mean a");
    let mean_a_b =
        eval_primitive(Primitive::BroadcastInDim, &[mean_a], &bcast).expect("broadcast mean a");
    let sumb = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(b), &reduce_axis1)
        .expect("reduce sum b");
    let mean_b = eval_primitive(Primitive::Div, &[sumb, n.clone()], &empty).expect("mean b");
    let mean_b_b =
        eval_primitive(Primitive::BroadcastInDim, &[mean_b], &bcast).expect("broadcast mean b");
    let ca = eval_primitive(Primitive::Sub, &[a.clone(), mean_a_b], &empty).expect("center a");
    let cb = eval_primitive(Primitive::Sub, &[b.clone(), mean_b_b], &empty).expect("center b");
    let prod = eval_primitive(Primitive::Mul, &[ca, cb], &empty).expect("product");
    let s = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&prod), &reduce_axis1)
        .expect("reduce sum product");
    eval_primitive(Primitive::Div, &[s, n], &empty).expect("covariance")
}

fn eval_l2_norm_2d_decomposed(input: &Value) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let empty = BTreeMap::new();
    let sq = eval_primitive(Primitive::Mul, &[input.clone(), input.clone()], &empty)
        .expect("square");
    let s = eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&sq), &reduce_axis1)
        .expect("reduce sum sq");
    eval_primitive(Primitive::Sqrt, std::slice::from_ref(&s), &empty).expect("l2_norm")
}

fn eval_logsumexp_2d_decomposed(input: &Value, rows: usize, cols: usize) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let max = eval_primitive(
        Primitive::ReduceMax,
        std::slice::from_ref(input),
        &reduce_axis1,
    )
    .expect("reduce max");
    let max_b = eval_primitive(
        Primitive::BroadcastInDim,
        std::slice::from_ref(&max),
        &bcast,
    )
    .expect("broadcast max");
    let shifted =
        eval_primitive(Primitive::Sub, &[input.clone(), max_b], &empty).expect("subtract max");
    let exp = eval_primitive(Primitive::Exp, std::slice::from_ref(&shifted), &empty).expect("exp");
    let sum = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&exp),
        &reduce_axis1,
    )
    .expect("reduce sum");
    let logv = eval_primitive(Primitive::Log, std::slice::from_ref(&sum), &empty).expect("log");
    eval_primitive(Primitive::Add, &[max, logv], &empty).expect("add max")
}

fn eval_softmax_cross_entropy_2d_decomposed(
    logits: &Value,
    labels: &Value,
    rows: usize,
    cols: usize,
) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let max = eval_primitive(
        Primitive::ReduceMax,
        std::slice::from_ref(logits),
        &reduce_axis1,
    )
    .expect("reduce max");
    let max_b = eval_primitive(Primitive::BroadcastInDim, &[max], &bcast).expect("broadcast max");
    let shifted =
        eval_primitive(Primitive::Sub, &[logits.clone(), max_b], &empty).expect("subtract max");
    let exp = eval_primitive(Primitive::Exp, std::slice::from_ref(&shifted), &empty).expect("exp");
    let sum = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&exp),
        &reduce_axis1,
    )
    .expect("reduce sum");
    let logv = eval_primitive(Primitive::Log, std::slice::from_ref(&sum), &empty).expect("log");
    let log_b = eval_primitive(Primitive::BroadcastInDim, &[logv], &bcast).expect("broadcast log");
    let ls = eval_primitive(Primitive::Sub, &[shifted, log_b], &empty).expect("log_softmax");
    let prod = eval_primitive(Primitive::Mul, &[labels.clone(), ls], &empty).expect("weighted");
    let ce_sum = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&prod),
        &reduce_axis1,
    )
    .expect("cross entropy sum");
    eval_primitive(Primitive::Neg, std::slice::from_ref(&ce_sum), &empty).expect("negate")
}

fn eval_rms_norm_2d_decomposed(input: &Value, rows: usize, cols: usize, epsilon: f64) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let squared =
        eval_primitive(Primitive::Square, std::slice::from_ref(input), &empty).expect("square");
    let sq_sum = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&squared),
        &reduce_axis1,
    )
    .expect("reduce sum");
    let ms = eval_primitive(
        Primitive::Div,
        &[sq_sum, Value::scalar_f64(cols as f64)],
        &empty,
    )
    .expect("mean square");
    let ms_eps = eval_primitive(Primitive::Add, &[ms, Value::scalar_f64(epsilon)], &empty)
        .expect("mean square epsilon");
    let inv =
        eval_primitive(Primitive::Rsqrt, std::slice::from_ref(&ms_eps), &empty).expect("rsqrt");
    let inv_b = eval_primitive(Primitive::BroadcastInDim, &[inv], &bcast).expect("broadcast inv");
    eval_primitive(Primitive::Mul, &[input.clone(), inv_b], &empty).expect("scale")
}

fn eval_layer_norm_2d_decomposed(input: &Value, rows: usize, cols: usize, epsilon: f64) -> Value {
    let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
    let bcast = BTreeMap::from([
        ("shape".to_owned(), format!("{rows},{cols}")),
        ("broadcast_dimensions".to_owned(), "0".to_owned()),
    ]);
    let empty = BTreeMap::new();
    let sum = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(input),
        &reduce_axis1,
    )
    .expect("reduce sum");
    let mean = eval_primitive(
        Primitive::Div,
        &[sum, Value::scalar_f64(cols as f64)],
        &empty,
    )
    .expect("mean");
    let mean_b =
        eval_primitive(Primitive::BroadcastInDim, &[mean], &bcast).expect("broadcast mean");
    let centered =
        eval_primitive(Primitive::Sub, &[input.clone(), mean_b], &empty).expect("center");
    let squared =
        eval_primitive(Primitive::Square, std::slice::from_ref(&centered), &empty).expect("square");
    let var_sum = eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(&squared),
        &reduce_axis1,
    )
    .expect("variance sum");
    let variance = eval_primitive(
        Primitive::Div,
        &[var_sum, Value::scalar_f64(cols as f64)],
        &empty,
    )
    .expect("variance");
    let var_eps = eval_primitive(
        Primitive::Add,
        &[variance, Value::scalar_f64(epsilon)],
        &empty,
    )
    .expect("variance epsilon");
    let inv =
        eval_primitive(Primitive::Rsqrt, std::slice::from_ref(&var_eps), &empty).expect("rsqrt");
    let inv_b = eval_primitive(Primitive::BroadcastInDim, &[inv], &bcast).expect("broadcast inv");
    eval_primitive(Primitive::Mul, &[centered, inv_b], &empty).expect("scale")
}

fn build_broadcast_reciprocal_jaxpr(rows: usize, cols: usize) -> Jaxpr {
    let x = VarId(1);
    let bcast = VarId(2);
    let out = VarId(3);
    let mut broadcast_params = BTreeMap::new();
    broadcast_params.insert("shape".to_owned(), format!("{rows},{cols}"));
    broadcast_params.insert("broadcast_dimensions".to_owned(), "1".to_owned());
    Jaxpr::new(
        vec![x],
        vec![],
        vec![out],
        vec![
            Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec::smallvec![Atom::Var(x)],
                outputs: smallvec::smallvec![bcast],
                params: broadcast_params,
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Reciprocal,
                inputs: smallvec::smallvec![Atom::Var(bcast)],
                outputs: smallvec::smallvec![out],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn bench_one(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tag: &str,
    jaxpr: &Jaxpr,
    args: &[Value],
) {
    group.bench_function(format!("eager/{tag}"), |b| {
        b.iter(|| eval_jaxpr(black_box(jaxpr), black_box(args)).unwrap())
    });
    // Skip the compiled arm rather than panic if a workload is outside the dense subset.
    if let Some(compiled) = compile_jaxpr_for_repeated_eval(jaxpr) {
        group.bench_function(format!("compiled/{tag}"), |b| {
            b.iter(|| compiled.eval(black_box(args)).unwrap())
        });
        let mut runner = compiled.runner();
        group.bench_function(format!("compiled_runner/{tag}"), |b| {
            b.iter(|| {
                let out = runner.eval(black_box(args)).unwrap();
                black_box(out);
            })
        });
        // Same-invocation A/B control: identical runner with the dense-f64 inner-loop
        // vectorization DISABLED (generic per-element loop). Vectorized vs per-element in
        // ONE binary is the only worker-variance-immune signal on the contended host.
        let mut runner_scalar = compiled.runner();
        group.bench_function(format!("compiled_runner_scalar/{tag}"), |b| {
            b.iter(|| {
                let out = runner_scalar.eval_scalar_inner(black_box(args)).unwrap();
                black_box(out);
            })
        });
    }
}

fn bench_donation_one(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tag: &str,
    jaxpr: &Jaxpr,
    args: &[Value],
) {
    DENSE_F64_DONATION_DISABLE.store(1, Ordering::Relaxed);
    let original = eval_jaxpr(jaxpr, args).expect("original eval");
    DENSE_F64_DONATION_DISABLE.store(0, Ordering::Relaxed);
    let donated = eval_jaxpr(jaxpr, args).expect("donated eval");
    assert_eq!(original, donated, "{tag}: donation changed result");

    group.bench_function(format!("original_no_donation/{tag}"), |b| {
        b.iter(|| {
            DENSE_F64_DONATION_DISABLE.store(1, Ordering::Relaxed);
            black_box(eval_jaxpr(black_box(jaxpr), black_box(args)).unwrap());
        })
    });
    group.bench_function(format!("donated/{tag}"), |b| {
        b.iter(|| {
            DENSE_F64_DONATION_DISABLE.store(0, Ordering::Relaxed);
            black_box(eval_jaxpr(black_box(jaxpr), black_box(args)).unwrap());
        })
    });
    DENSE_F64_DONATION_DISABLE.store(0, Ordering::Relaxed);
}

fn bench_compiled_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_dispatch");
    // Scalar Add chains: dispatch-bound, trivial kernel — pure interpreter tax.
    let scalar_args = [Value::scalar_i64(0)];
    for &n in &[8usize, 32, 128] {
        let jaxpr = build_chain_jaxpr(n, Literal::I64(1));
        bench_one(&mut group, &format!("scalar/n={n}"), &jaxpr, &scalar_args);
    }
    // Small-tensor f64 elementwise-broadcast chains: dense binary is NOT pre-scanned in
    // DenseEvalPlan, so this profiles the remaining per-call dispatch gap (frankenjax-6dfew).
    let tensor_args = [Value::vector_f64(&[1.0_f64; 64]).expect("vector_f64")];
    for &n in &[8usize, 32] {
        let jaxpr = build_chain_jaxpr(n, Literal::from_f64(1.0));
        bench_one(&mut group, &format!("tensor64/n={n}"), &jaxpr, &tensor_args);
    }
    // Element-count sweep at a fixed short chain (n=4): confirms the vectorized inner
    // loop wins (or at worst ties) across sizes, never regresses.
    for &elems in &[8usize, 256, 1023] {
        let arg = vec![1.0_f64; elems];
        let args = [Value::vector_f64(&arg).expect("vector_f64")];
        let jaxpr = build_chain_jaxpr(4, Literal::from_f64(1.0));
        bench_one(&mut group, &format!("tensorE{elems}/n=4"), &jaxpr, &args);
    }
    // L3-resident f64 chains (>= FUSION_MIN_ELEMS): in-place chain (one buffer, 1-stream
    // traffic, no per-step alloc) vs the generic per-op path (N allocs, 2-stream). A/B is
    // compiled_runner (in-place, vectorize on) vs compiled_runner_scalar (generic per-op).
    for &elems in &[4096usize, 65536, 262144, 1048576, 16777216] {
        let arg = vec![1.0_f64; elems];
        let args = [Value::vector_f64(&arg).expect("vector_f64")];
        let jaxpr = build_chain_jaxpr(8, Literal::from_f64(1.0));
        bench_one(&mut group, &format!("bigchain{elems}/n=8"), &jaxpr, &args);
    }
    // xjbvr target: large dense unary chains that break cheap-op fusion before
    // floor/round/sign are admitted. Values are non-integral so floor/round do
    // real work; JAX jit fuses the full chain into one compiled kernel.
    let unary_f64_arg: Vec<f64> = (0..1_048_576)
        .map(|idx| idx as f64 * 0.000_001 - 0.5)
        .collect();
    let unary_f64_args = [Value::vector_f64(&unary_f64_arg).expect("vector_f64")];
    for &(name, primitive) in &[
        ("floor", Primitive::Floor),
        ("round", Primitive::Round),
        ("sign", Primitive::Sign),
    ] {
        let jaxpr = build_add_unary_chain_jaxpr(4, primitive, Literal::from_f64(0.125));
        bench_one(
            &mut group,
            &format!("{name}_f64_1m_add_unary_chain/n=4"),
            &jaxpr,
            &unary_f64_args,
        );
    }
    // f32 (JAX's DEFAULT tensor dtype): native-f32 vectorization (vaddps, 8-wide), bit-
    // exact vs eager's widen→f64→narrow for +/-/*/÷ (Figueroa). 256-lane chains.
    let f32_tensor = Value::Tensor(
        fj_core::TensorValue::new(
            fj_core::DType::F32,
            fj_core::Shape { dims: vec![256] },
            (0..256).map(|_| Literal::from_f32(1.0)).collect(),
        )
        .expect("f32 tensor"),
    );
    let f32_args = [f32_tensor];
    for &n in &[8usize, 32] {
        let jaxpr = build_chain_jaxpr(n, Literal::from_f32(1.0));
        bench_one(&mut group, &format!("f32E256/n={n}"), &jaxpr, &f32_args);
    }
    // L3-resident f32 chains (JAX's default dtype): in-place chain vs generic per-op.
    for &elems in &[4096usize, 65536] {
        let t = Value::Tensor(
            fj_core::TensorValue::new(
                fj_core::DType::F32,
                fj_core::Shape {
                    dims: vec![elems as u32],
                },
                (0..elems).map(|_| Literal::from_f32(1.0)).collect(),
            )
            .expect("f32 big tensor"),
        );
        let args = [t];
        let jaxpr = build_chain_jaxpr(8, Literal::from_f32(1.0));
        bench_one(&mut group, &format!("f32big{elems}/n=8"), &jaxpr, &args);
    }
    let unary_f32_tensor = Value::Tensor(
        fj_core::TensorValue::new(
            fj_core::DType::F32,
            fj_core::Shape {
                dims: vec![1_048_576],
            },
            (0..1_048_576)
                .map(|idx| Literal::from_f32(idx as f32 * 0.000_001 - 0.5))
                .collect(),
        )
        .expect("f32 unary tensor"),
    );
    let unary_f32_args = [unary_f32_tensor];
    for &(name, primitive) in &[
        ("floor", Primitive::Floor),
        ("round", Primitive::Round),
        ("sign", Primitive::Sign),
    ] {
        let jaxpr = build_add_unary_chain_jaxpr(4, primitive, Literal::from_f32(0.125));
        bench_one(
            &mut group,
            &format!("{name}_f32_1m_add_unary_chain/n=4"),
            &jaxpr,
            &unary_f32_args,
        );
    }
    // i64 (index/counter buffers): wrapping Add/Sub/Mul vectorize to vpaddq etc.
    let i64_args = [Value::vector_i64(&[1_i64; 256]).expect("vector_i64")];
    for &n in &[8usize, 32] {
        let jaxpr = build_chain_jaxpr(n, Literal::I64(1));
        bench_one(&mut group, &format!("i64E256/n={n}"), &jaxpr, &i64_args);
    }
    // f64 rank-2 ROW-BROADCAST bias-add chain: [16,16] matrix + [16] vector (the per-row
    // decomposition reuses the no-broadcast vectorized helper).
    let bcast_args = [
        Value::Tensor(
            fj_core::TensorValue::new(
                fj_core::DType::F64,
                fj_core::Shape { dims: vec![16, 16] },
                (0..256).map(|_| Literal::from_f64(1.0)).collect(),
            )
            .expect("matrix"),
        ),
        Value::vector_f64(&[0.5_f64; 16]).expect("row vector"),
    ];
    for &n in &[8usize, 32] {
        let jaxpr = build_bcast_chain_jaxpr(n);
        bench_one(
            &mut group,
            &format!("bcast16x16/n={n}"),
            &jaxpr,
            &bcast_args,
        );
    }
    let rows = 4096usize;
    let cols = 1024usize;
    let softmax_data: Vec<f64> = (0..rows * cols)
        .map(|idx| ((idx as f64) * 0.0007).sin() * 4.0)
        .collect();
    let softmax_input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            softmax_data,
        )
        .expect("softmax input"),
    );
    let softmax_jaxpr = build_softmax_2d_jaxpr(rows, cols);
    group.bench_function("softmax_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_softmax_2d_decomposed(
                black_box(&softmax_input),
                rows,
                cols,
            ))
        })
    });
    group.bench_function("softmax_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(
                    black_box(&softmax_jaxpr),
                    std::slice::from_ref(&softmax_input),
                )
                .unwrap(),
            )
        })
    });
    let log_softmax_jaxpr = build_log_softmax_2d_jaxpr(rows, cols);
    group.bench_function("log_softmax_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_log_softmax_2d_decomposed(
                black_box(&softmax_input),
                rows,
                cols,
            ))
        })
    });
    group.bench_function("log_softmax_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(
                    black_box(&log_softmax_jaxpr),
                    std::slice::from_ref(&softmax_input),
                )
                .unwrap(),
            )
        })
    });
    let epsilon = 1.0e-5;
    let layer_norm_data: Vec<f64> = (0..rows * cols)
        .map(|idx| ((idx as f64) * 0.0013).cos() * 3.0 + ((idx % cols) as f64) * 0.0002)
        .collect();
    let layer_norm_input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            layer_norm_data,
        )
        .expect("layer norm input"),
    );
    let layer_norm_jaxpr = build_layer_norm_2d_jaxpr(rows, cols, epsilon);
    group.bench_function("layer_norm_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_layer_norm_2d_decomposed(
                black_box(&layer_norm_input),
                rows,
                cols,
                epsilon,
            ))
        })
    });
    group.bench_function("layer_norm_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(
                    black_box(&layer_norm_jaxpr),
                    std::slice::from_ref(&layer_norm_input),
                )
                .unwrap(),
            )
        })
    });
    let rms_norm_jaxpr = build_rms_norm_2d_jaxpr(rows, cols, epsilon);
    group.bench_function("rms_norm_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_rms_norm_2d_decomposed(
                black_box(&layer_norm_input),
                rows,
                cols,
                epsilon,
            ))
        })
    });
    group.bench_function("rms_norm_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(
                    black_box(&rms_norm_jaxpr),
                    std::slice::from_ref(&layer_norm_input),
                )
                .unwrap(),
            )
        })
    });
    let ce_labels_data: Vec<f64> = (0..rows * cols)
        .map(|idx| ((idx as f64) * 0.0011).cos().abs() * 0.5)
        .collect();
    let ce_labels_input = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            ce_labels_data,
        )
        .expect("cross entropy labels"),
    );
    let cross_entropy_jaxpr = build_softmax_cross_entropy_2d_jaxpr(rows, cols);
    let ce_args = [softmax_input.clone(), ce_labels_input.clone()];
    group.bench_function("softmax_cross_entropy_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_softmax_cross_entropy_2d_decomposed(
                black_box(&ce_args[0]),
                black_box(&ce_args[1]),
                rows,
                cols,
            ))
        })
    });
    group.bench_function("softmax_cross_entropy_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_jaxpr(black_box(&cross_entropy_jaxpr), black_box(&ce_args)).unwrap())
        })
    });
    let euclidean_jaxpr = build_euclidean_distance_2d_jaxpr();
    let euclidean_args = [softmax_input.clone(), layer_norm_input.clone()];
    group.bench_function("euclidean_distance_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_euclidean_distance_2d_decomposed(
                black_box(&euclidean_args[0]),
                black_box(&euclidean_args[1]),
            ))
        })
    });
    group.bench_function("euclidean_distance_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_jaxpr(black_box(&euclidean_jaxpr), black_box(&euclidean_args)).unwrap())
        })
    });
    let mse_jaxpr = build_mean_squared_error_2d_jaxpr(cols);
    let mse_args = [softmax_input.clone(), layer_norm_input.clone()];
    group.bench_function("mean_squared_error_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_mean_squared_error_2d_decomposed(
                black_box(&mse_args[0]),
                black_box(&mse_args[1]),
                cols,
            ))
        })
    });
    group.bench_function("mean_squared_error_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| black_box(eval_jaxpr(black_box(&mse_jaxpr), black_box(&mse_args)).unwrap()))
    });
    let manhattan_jaxpr = build_manhattan_distance_2d_jaxpr();
    let manhattan_args = [softmax_input.clone(), layer_norm_input.clone()];
    group.bench_function("manhattan_distance_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_manhattan_distance_2d_decomposed(
                black_box(&manhattan_args[0]),
                black_box(&manhattan_args[1]),
            ))
        })
    });
    group.bench_function("manhattan_distance_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_jaxpr(black_box(&manhattan_jaxpr), black_box(&manhattan_args)).unwrap())
        })
    });
    let entropy_jaxpr = build_entropy_2d_jaxpr();
    group.bench_function("entropy_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| black_box(eval_entropy_2d_decomposed(black_box(&softmax_input))))
    });
    group.bench_function("entropy_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(black_box(&entropy_jaxpr), std::slice::from_ref(&softmax_input)).unwrap(),
            )
        })
    });
    let kl_jaxpr = build_kl_divergence_2d_jaxpr();
    let kl_args = [softmax_input.clone(), layer_norm_input.clone()];
    group.bench_function("kl_divergence_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_kl_divergence_2d_decomposed(
                black_box(&kl_args[0]),
                black_box(&kl_args[1]),
            ))
        })
    });
    group.bench_function("kl_divergence_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| black_box(eval_jaxpr(black_box(&kl_jaxpr), black_box(&kl_args)).unwrap()))
    });
    let pearson_jaxpr = build_pearson_correlation_2d_jaxpr(rows, cols);
    let pearson_args = [softmax_input.clone(), layer_norm_input.clone()];
    group.bench_function("pearson_correlation_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_pearson_correlation_2d_decomposed(
                black_box(&pearson_args[0]),
                black_box(&pearson_args[1]),
                rows,
                cols,
            ))
        })
    });
    group.bench_function("pearson_correlation_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_jaxpr(black_box(&pearson_jaxpr), black_box(&pearson_args)).unwrap())
        })
    });
    let cosine_jaxpr = build_cosine_similarity_2d_jaxpr();
    let cosine_args = [softmax_input.clone(), layer_norm_input.clone()];
    group.bench_function("cosine_similarity_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_cosine_similarity_2d_decomposed(
                black_box(&cosine_args[0]),
                black_box(&cosine_args[1]),
            ))
        })
    });
    group.bench_function("cosine_similarity_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| black_box(eval_jaxpr(black_box(&cosine_jaxpr), black_box(&cosine_args)).unwrap()))
    });
    let var_jaxpr = build_var_2d_jaxpr(rows, cols);
    group.bench_function("var_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_var_2d_decomposed(
                black_box(&softmax_input),
                rows,
                cols,
            ))
        })
    });
    group.bench_function("var_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(black_box(&var_jaxpr), std::slice::from_ref(&softmax_input)).unwrap(),
            )
        })
    });
    let std_jaxpr = build_std_2d_jaxpr(rows, cols);
    group.bench_function("std_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_std_2d_decomposed(
                black_box(&softmax_input),
                rows,
                cols,
            ))
        })
    });
    group.bench_function("std_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(black_box(&std_jaxpr), std::slice::from_ref(&softmax_input)).unwrap(),
            )
        })
    });
    let skewness_jaxpr = build_skewness_2d_jaxpr(rows, cols);
    group.bench_function("skewness_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_skewness_2d_decomposed(
                black_box(&softmax_input),
                rows,
                cols,
            ))
        })
    });
    group.bench_function("skewness_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(black_box(&skewness_jaxpr), std::slice::from_ref(&softmax_input))
                    .unwrap(),
            )
        })
    });
    let kurtosis_jaxpr = build_kurtosis_2d_jaxpr(rows, cols);
    group.bench_function("kurtosis_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_kurtosis_2d_decomposed(
                black_box(&softmax_input),
                rows,
                cols,
            ))
        })
    });
    group.bench_function("kurtosis_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(black_box(&kurtosis_jaxpr), std::slice::from_ref(&softmax_input))
                    .unwrap(),
            )
        })
    });
    let covariance_jaxpr = build_covariance_2d_jaxpr(rows, cols);
    let covariance_args = [softmax_input.clone(), layer_norm_input.clone()];
    group.bench_function("covariance_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_covariance_2d_decomposed(
                black_box(&covariance_args[0]),
                black_box(&covariance_args[1]),
                rows,
                cols,
            ))
        })
    });
    group.bench_function("covariance_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| black_box(eval_jaxpr(black_box(&covariance_jaxpr), black_box(&covariance_args)).unwrap()))
    });
    let l2_norm_jaxpr = build_l2_norm_2d_jaxpr();
    group.bench_function("l2_norm_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| black_box(eval_l2_norm_2d_decomposed(black_box(&softmax_input))))
    });
    group.bench_function("l2_norm_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(black_box(&l2_norm_jaxpr), std::slice::from_ref(&softmax_input)).unwrap(),
            )
        })
    });
    let logsumexp_jaxpr = build_logsumexp_2d_jaxpr(rows, cols);
    group.bench_function("logsumexp_2d/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_logsumexp_2d_decomposed(
                black_box(&softmax_input),
                rows,
                cols,
            ))
        })
    });
    group.bench_function("logsumexp_2d/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(
                    black_box(&logsumexp_jaxpr),
                    std::slice::from_ref(&softmax_input),
                )
                .unwrap(),
            )
        })
    });
    let gelu_jaxpr = build_gelu_erf_jaxpr();
    group.bench_function("gelu_erf/orig_decomposed_4096x1024", |b| {
        b.iter(|| black_box(eval_gelu_erf_decomposed(black_box(&softmax_input))))
    });
    group.bench_function("gelu_erf/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(black_box(&gelu_jaxpr), std::slice::from_ref(&softmax_input)).unwrap(),
            )
        })
    });
    let log_sigmoid_jaxpr = build_log_sigmoid_jaxpr();
    group.bench_function("log_sigmoid/orig_decomposed_4096x1024", |b| {
        b.iter(|| black_box(eval_log_sigmoid_decomposed(black_box(&softmax_input))))
    });
    group.bench_function("log_sigmoid/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(
                    black_box(&log_sigmoid_jaxpr),
                    std::slice::from_ref(&softmax_input),
                )
                .unwrap(),
            )
        })
    });
    let softplus_jaxpr = build_softplus_jaxpr();
    group.bench_function("softplus/orig_decomposed_4096x1024", |b| {
        b.iter(|| black_box(eval_softplus_decomposed(black_box(&softmax_input))))
    });
    group.bench_function("softplus/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(
                    black_box(&softplus_jaxpr),
                    std::slice::from_ref(&softmax_input),
                )
                .unwrap(),
            )
        })
    });
    let silu_jaxpr = build_silu_jaxpr();
    group.bench_function("silu/orig_decomposed_4096x1024", |b| {
        b.iter(|| black_box(eval_silu_decomposed(black_box(&softmax_input))))
    });
    group.bench_function("silu/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| {
            black_box(
                eval_jaxpr(black_box(&silu_jaxpr), std::slice::from_ref(&softmax_input)).unwrap(),
            )
        })
    });
    let swiglu_jaxpr = build_swiglu_jaxpr();
    let swiglu_args = [softmax_input.clone(), layer_norm_input.clone()];
    group.bench_function("swiglu/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_swiglu_decomposed(
                black_box(&swiglu_args[0]),
                black_box(&swiglu_args[1]),
            ))
        })
    });
    group.bench_function("swiglu/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| black_box(eval_jaxpr(black_box(&swiglu_jaxpr), black_box(&swiglu_args)).unwrap()))
    });
    let geglu_jaxpr = build_geglu_jaxpr();
    group.bench_function("geglu/orig_decomposed_4096x1024", |b| {
        b.iter(|| {
            black_box(eval_geglu_decomposed(
                black_box(&swiglu_args[0]),
                black_box(&swiglu_args[1]),
            ))
        })
    });
    group.bench_function("geglu/fast_eval_jaxpr_4096x1024", |b| {
        b.iter(|| black_box(eval_jaxpr(black_box(&geglu_jaxpr), black_box(&swiglu_args)).unwrap()))
    });
    for &(rows, cols) in &[(4096usize, 1024usize), (16384, 1024)] {
        let arg: Vec<f64> = (0..cols).map(|idx| idx as f64 * 0.001 + 1.0).collect();
        let args = [Value::vector_f64(&arg).expect("broadcast vector")];
        let jaxpr = build_broadcast_reciprocal_jaxpr(rows, cols);
        bench_donation_one(
            &mut group,
            &format!("donate_broadcast_reciprocal_f64_{rows}x{cols}"),
            &jaxpr,
            &args,
        );
    }
    group.finish();
}

criterion_group!(benches, bench_compiled_dispatch);
criterion_main!(benches);
