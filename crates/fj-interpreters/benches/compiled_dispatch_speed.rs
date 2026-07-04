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
