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
use fj_interpreters::{compile_jaxpr_for_repeated_eval, eval_jaxpr};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;

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
    group.finish();
}

criterion_group!(benches, bench_compiled_dispatch);
criterion_main!(benches);
