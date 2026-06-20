//! Speed A/B for elementwise fusion in eval_jaxpr (bead frankenjax-a8nbp).
//!
//! Same-invocation A/B: arm A evaluates a deep cheap-elementwise chain through
//! `eval_jaxpr` (which now FUSES the run into one chunked pass); arm B is the
//! UNFUSED reference — the same ops evaluated one at a time via `eval_primitive`,
//! materializing every intermediate (exactly what eval_jaxpr did before fusion).
//! Both run in one process so the ratio is trustworthy.
//! Run: `cargo bench -p fj-interpreters --bench eval_fusion_speed`.

use std::collections::BTreeMap;
use std::time::Instant;

use fj_core::{Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId};
use fj_interpreters::{FUSION_THREAD_CAP_OVERRIDE, eval_jaxpr};
use fj_lax::eval_primitive;
use smallvec::smallvec;

fn f64_tensor(vals: Vec<f64>) -> Value {
    let n = vals.len();
    Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32],
            },
            vals,
        )
        .unwrap(),
    )
}

fn f32_tensor(vals: Vec<f32>) -> Value {
    let n = vals.len();
    Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![n as u32],
            },
            vals,
        )
        .unwrap(),
    )
}

fn i64_tensor(vals: Vec<i64>) -> Value {
    let n = vals.len();
    Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![n as u32],
            },
            vals,
        )
        .unwrap(),
    )
}

fn bf16_bits_of(x: f64) -> u16 {
    match Literal::from_bf16_f64(x) {
        Literal::BF16Bits(b) => b,
        _ => 0,
    }
}

fn bf16_tensor(vals: Vec<u16>) -> Value {
    let n = vals.len();
    Value::Tensor(
        TensorValue::new_half_float_values(
            DType::BF16,
            Shape {
                dims: vec![n as u32],
            },
            vals,
        )
        .unwrap(),
    )
}

fn half_bits_at(t: &TensorValue, idx: usize) -> u16 {
    if let Some(values) = t.elements.as_half_float_slice() {
        return values[idx];
    }
    match t.elements[idx] {
        Literal::BF16Bits(bits) | Literal::F16Bits(bits) => bits,
        other => panic!("expected half-float element, got {other:?}"),
    }
}

fn f32_bits_at(t: &TensorValue, idx: usize) -> u32 {
    if let Some(values) = t.elements.as_f32_slice() {
        return values[idx].to_bits();
    }
    match t.elements[idx] {
        Literal::F32Bits(bits) => bits,
        other => panic!("expected f32 element, got {other:?}"),
    }
}

fn f64_bits_at(t: &TensorValue, idx: usize) -> u64 {
    if let Some(values) = t.elements.as_f64_slice() {
        return values[idx].to_bits();
    }
    t.elements[idx].as_f64().unwrap().to_bits()
}

fn run_f64() {
    let n = 1usize << 20; // 1M f64 = 8 MB per tensor (RAM-bound)
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 1e-6 - 0.5).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 3e-7).cos() + 1.2).collect();

    // Chain (8 cheap elementwise ops, single-use intermediates):
    //   v1 = mul(x,x); v2 = add(v1, 0.5); v3 = sub(v2, x); v4 = mul(v3, y);
    //   v5 = add(v4, 1.0); v6 = sub(v5, y); v7 = mul(v6, 2.0); out = add(v7, x)
    let xv = VarId(0);
    let yv = VarId(1);
    let v: Vec<VarId> = (2..=9).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f64| Atom::Lit(Literal::from_f64(c));
    let eqns = vec![
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(xv), Atom::Var(xv)],
            v[0],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[0]), lit(0.5)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(xv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[3]), lit(1.0)], v[4]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[4]), Atom::Var(yv)],
            v[5],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(xv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [f64_tensor(x.clone()), f64_tensor(y.clone())];

    // Unfused reference: evaluate each equation via eval_primitive, materializing
    // intermediates (the pre-fusion eval_jaxpr behavior), via a small env vec.
    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 10];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn
                .inputs
                .iter()
                .map(|a| match a {
                    Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                    Atom::Lit(l) => Value::Scalar(*l),
                })
                .collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };

    // Correctness sanity: fused == unfused, first element.
    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        let fv = ft.elements.iter().next().unwrap().as_f64().unwrap();
        let uv = ut.elements.iter().next().unwrap().as_f64().unwrap();
        assert_eq!(fv.to_bits(), uv.to_bits(), "fused != unfused");
    }

    let iters = 60;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;

    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(unfused());
    }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "EVAL_FUSION_SPEED_F64 n={n} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6,
        fused / 1e6,
        unf / fused,
    );
}

fn run_f32() {
    let n = 1usize << 20; // 1M f32 = 4 MB per tensor (JAX default float dtype)
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 1e-6 - 0.5).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32 * 3e-7).cos() + 1.2).collect();

    // Same chain as f64, but all tensor inputs and scalar literals are f32.
    let xv = VarId(0);
    let yv = VarId(1);
    let v: Vec<VarId> = (2..=9).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f32| Atom::Lit(Literal::from_f32(c));
    let eqns = vec![
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(xv), Atom::Var(xv)],
            v[0],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[0]), lit(0.5)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(xv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[3]), lit(1.0)], v[4]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[4]), Atom::Var(yv)],
            v[5],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(xv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [f32_tensor(x.clone()), f32_tensor(y.clone())];

    // Unfused reference: evaluate each equation via eval_primitive, materializing
    // intermediates (the pre-f32-fusion eval_jaxpr behavior), via a small env vec.
    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 10];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn
                .inputs
                .iter()
                .map(|a| match a {
                    Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                    Atom::Lit(l) => Value::Scalar(*l),
                })
                .collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };

    // Correctness sanity: fused == unfused for representative elements.
    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        for idx in [0, n / 2, n - 1] {
            assert_eq!(
                f32_bits_at(ft, idx),
                f32_bits_at(ut, idx),
                "fused f32 != unfused"
            );
        }
    }

    let iters = 60;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;

    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(unfused());
    }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "EVAL_FUSION_SPEED_F32 n={n} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6,
        fused / 1e6,
        unf / fused,
    );
}

fn run_f32_row_broadcast() {
    let rows = 1024usize;
    let cols = 1024usize;
    let n = rows * cols;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 1e-6 - 0.5).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32 * 7e-7).sin() + 1.1).collect();
    let bias: Vec<f32> = (0..cols).map(|i| i as f32 * 2e-4 - 0.25).collect();

    let tensor2 = |vals: Vec<f32>| {
        Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                vals,
            )
            .unwrap(),
        )
    };
    let row_tensor = |vals: Vec<f32>| {
        Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![cols as u32],
                },
                vals,
            )
            .unwrap(),
        )
    };

    // Bias-style row broadcast chain:
    //   v1 = add(x, b); v2 = mul(v1, 1.25); v3 = sub(v2, b); v4 = mul(v3, y);
    //   v5 = add(v4, b); v6 = sub(v5, 0.5); v7 = mul(v6, 2.0); out = add(v7, b)
    let xv = VarId(0);
    let bv = VarId(1);
    let yv = VarId(2);
    let v: Vec<VarId> = (3..=10).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f32| Atom::Lit(Literal::from_f32(c));
    let eqns = vec![
        mk(
            Primitive::Add,
            smallvec![Atom::Var(xv), Atom::Var(bv)],
            v[0],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[0]), lit(1.25)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(bv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[3]), Atom::Var(bv)],
            v[4],
        ),
        mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit(0.5)], v[5]),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(bv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [
        tensor2(x.clone()),
        row_tensor(bias.clone()),
        tensor2(y.clone()),
    ];

    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 11];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        env[2] = Some(args[2].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn
                .inputs
                .iter()
                .map(|a| match a {
                    Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                    Atom::Lit(l) => Value::Scalar(*l),
                })
                .collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };

    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        for idx in [0, cols - 1, n / 2, n - 1] {
            assert_eq!(
                f32_bits_at(ft, idx),
                f32_bits_at(ut, idx),
                "fused f32 row-broadcast != unfused"
            );
        }
    }

    let iters = 50;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;

    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(unfused());
    }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "EVAL_FUSION_SPEED_F32_ROW_BROADCAST rows={rows} cols={cols} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6,
        fused / 1e6,
        unf / fused,
    );
}

fn run_f32_col_broadcast() {
    let rows = 1024usize;
    let cols = 1024usize;
    let n = rows * cols;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 1e-6 - 0.5).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32 * 5e-7).cos() + 1.2).collect();
    let bias: Vec<f32> = (0..rows).map(|i| i as f32 * 1e-4 - 0.125).collect();

    let tensor2 = |vals: Vec<f32>| {
        Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                vals,
            )
            .unwrap(),
        )
    };
    let col_tensor = |vals: Vec<f32>| {
        let mut expanded = Vec::with_capacity(rows);
        expanded.extend(vals);
        Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![rows as u32, 1],
                },
                expanded,
            )
            .unwrap(),
        )
    };

    // Bias-style column broadcast chain:
    //   v1 = add(x, b); v2 = mul(v1, 1.25); v3 = sub(v2, b); v4 = mul(v3, y);
    //   v5 = add(v4, b); v6 = sub(v5, 0.5); v7 = mul(v6, 2.0); out = add(v7, b)
    let xv = VarId(0);
    let bv = VarId(1);
    let yv = VarId(2);
    let v: Vec<VarId> = (3..=10).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f32| Atom::Lit(Literal::from_f32(c));
    let eqns = vec![
        mk(
            Primitive::Add,
            smallvec![Atom::Var(xv), Atom::Var(bv)],
            v[0],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[0]), lit(1.25)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(bv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[3]), Atom::Var(bv)],
            v[4],
        ),
        mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit(0.5)], v[5]),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(bv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [
        tensor2(x.clone()),
        col_tensor(bias.clone()),
        tensor2(y.clone()),
    ];

    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 11];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        env[2] = Some(args[2].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn
                .inputs
                .iter()
                .map(|a| match a {
                    Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                    Atom::Lit(l) => Value::Scalar(*l),
                })
                .collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };

    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        for idx in [0, cols - 1, n / 2, n - 1] {
            assert_eq!(
                f32_bits_at(ft, idx),
                f32_bits_at(ut, idx),
                "fused f32 col-broadcast != unfused"
            );
        }
    }

    let iters = 50;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;

    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(unfused());
    }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "EVAL_FUSION_SPEED_F32_COL_BROADCAST rows={rows} cols={cols} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6,
        fused / 1e6,
        unf / fused,
    );
}

fn run_f64_row_broadcast() {
    let rows = 1024usize;
    let cols = 1024usize;
    let n = rows * cols;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 1e-6 - 0.5).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 7e-7).sin() + 1.1).collect();
    let bias: Vec<f64> = (0..cols).map(|i| i as f64 * 2e-4 - 0.25).collect();

    let tensor2 = |vals: Vec<f64>| {
        Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                vals,
            )
            .unwrap(),
        )
    };
    let row_tensor = |vals: Vec<f64>| {
        Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![cols as u32],
                },
                vals,
            )
            .unwrap(),
        )
    };

    // Bias-style row broadcast chain (f64 layernorm-shaped):
    //   v1 = add(x, b); v2 = mul(v1, 1.25); v3 = sub(v2, b); v4 = mul(v3, y);
    //   v5 = add(v4, b); v6 = sub(v5, 0.5); v7 = mul(v6, 2.0); out = add(v7, b)
    let xv = VarId(0);
    let bv = VarId(1);
    let yv = VarId(2);
    let v: Vec<VarId> = (3..=10).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f64| Atom::Lit(Literal::from_f64(c));
    let eqns = vec![
        mk(
            Primitive::Add,
            smallvec![Atom::Var(xv), Atom::Var(bv)],
            v[0],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[0]), lit(1.25)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(bv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[3]), Atom::Var(bv)],
            v[4],
        ),
        mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit(0.5)], v[5]),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(bv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [
        tensor2(x.clone()),
        row_tensor(bias.clone()),
        tensor2(y.clone()),
    ];

    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 11];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        env[2] = Some(args[2].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn
                .inputs
                .iter()
                .map(|a| match a {
                    Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                    Atom::Lit(l) => Value::Scalar(*l),
                })
                .collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };

    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        for idx in [0, cols - 1, n / 2, n - 1] {
            assert_eq!(
                f64_bits_at(ft, idx),
                f64_bits_at(ut, idx),
                "fused f64 row-broadcast != unfused"
            );
        }
    }

    let iters = 50;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;

    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(unfused());
    }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "EVAL_FUSION_SPEED_F64_ROW_BROADCAST rows={rows} cols={cols} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6,
        fused / 1e6,
        unf / fused,
    );
}

fn run_f64_col_broadcast() {
    let rows = 1024usize;
    let cols = 1024usize;
    let n = rows * cols;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 1e-6 - 0.5).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 5e-7).cos() + 1.2).collect();
    let bias: Vec<f64> = (0..rows).map(|i| i as f64 * 1e-4 - 0.125).collect();

    let tensor2 = |vals: Vec<f64>| {
        Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                vals,
            )
            .unwrap(),
        )
    };
    let col_tensor = |vals: Vec<f64>| {
        Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, 1],
                },
                vals,
            )
            .unwrap(),
        )
    };

    // Bias-style column broadcast chain (f64):
    //   v1 = add(x, b); v2 = mul(v1, 1.25); v3 = sub(v2, b); v4 = mul(v3, y);
    //   v5 = add(v4, b); v6 = sub(v5, 0.5); v7 = mul(v6, 2.0); out = add(v7, b)
    let xv = VarId(0);
    let bv = VarId(1);
    let yv = VarId(2);
    let v: Vec<VarId> = (3..=10).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f64| Atom::Lit(Literal::from_f64(c));
    let eqns = vec![
        mk(
            Primitive::Add,
            smallvec![Atom::Var(xv), Atom::Var(bv)],
            v[0],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[0]), lit(1.25)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(bv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[3]), Atom::Var(bv)],
            v[4],
        ),
        mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit(0.5)], v[5]),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(bv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [
        tensor2(x.clone()),
        col_tensor(bias.clone()),
        tensor2(y.clone()),
    ];

    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 11];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        env[2] = Some(args[2].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn
                .inputs
                .iter()
                .map(|a| match a {
                    Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                    Atom::Lit(l) => Value::Scalar(*l),
                })
                .collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };

    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        for idx in [0, cols - 1, n / 2, n - 1] {
            assert_eq!(
                f64_bits_at(ft, idx),
                f64_bits_at(ut, idx),
                "fused f64 col-broadcast != unfused"
            );
        }
    }

    let iters = 50;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;

    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(unfused());
    }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "EVAL_FUSION_SPEED_F64_COL_BROADCAST rows={rows} cols={cols} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6,
        fused / 1e6,
        unf / fused,
    );
}

fn run_i64() {
    let n = 1usize << 20; // 1M i64 = 8 MB per tensor (RAM-bound)
    let x: Vec<i64> = (0..n).map(|i| i as i64 - (n as i64 / 2)).collect();
    let y: Vec<i64> = (0..n).map(|i| (i as i64 % 13) - 6).collect();

    // Chain (8 cheap integer ops, single-use intermediates), no Div so the
    // wrapping arms autovectorize (matching the f64 bench's op mix):
    //   v1 = mul(x,x); v2 = add(v1, 3); v3 = sub(v2, x); v4 = mul(v3, y);
    //   v5 = add(v4, 11); v6 = sub(v5, y); v7 = mul(v6, 2); out = add(v7, x)
    let xv = VarId(0);
    let yv = VarId(1);
    let v: Vec<VarId> = (2..=9).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: i64| Atom::Lit(Literal::I64(c));
    let eqns = vec![
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(xv), Atom::Var(xv)],
            v[0],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[0]), lit(3)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(xv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[3]), lit(11)], v[4]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[4]), Atom::Var(yv)],
            v[5],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(xv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [i64_tensor(x.clone()), i64_tensor(y.clone())];

    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 10];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn
                .inputs
                .iter()
                .map(|a| match a {
                    Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                    Atom::Lit(l) => Value::Scalar(*l),
                })
                .collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };

    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        for idx in [0usize, n / 2, n - 1] {
            let fv = ft.elements[idx].as_i64().unwrap();
            let uv = ut.elements[idx].as_i64().unwrap();
            assert_eq!(fv, uv, "fused i64 != unfused");
        }
    }

    let iters = 60;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;

    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(unfused());
    }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "EVAL_FUSION_SPEED_I64 n={n} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6,
        fused / 1e6,
        unf / fused,
    );
}

/// bf16 elementwise chain (the dominant ML activation dtype). Each fused step
/// reproduces fj-lax's per-op bf16 contract (widen→f64→op→round-to-bf16), so the
/// win is the eliminated N-1 intermediate half tensors, same op mix as run_i64.
fn run_bf16() {
    let n = 1usize << 20; // 1M bf16 = 2 MB per tensor
    let x: Vec<u16> = (0..n)
        .map(|i| bf16_bits_of(i as f64 * 1e-6 - 0.5))
        .collect();
    let y: Vec<u16> = (0..n)
        .map(|i| bf16_bits_of((i as f64 * 3e-7).cos() + 1.2))
        .collect();

    let xv = VarId(0);
    let yv = VarId(1);
    let v: Vec<VarId> = (2..=9).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f64| Atom::Lit(Literal::from_bf16_f64(c));
    let eqns = vec![
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(xv), Atom::Var(xv)],
            v[0],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[0]), lit(0.5)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(xv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[3]), lit(1.0)], v[4]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[4]), Atom::Var(yv)],
            v[5],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(xv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [bf16_tensor(x.clone()), bf16_tensor(y.clone())];

    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 10];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn
                .inputs
                .iter()
                .map(|a| match a {
                    Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                    Atom::Lit(l) => Value::Scalar(*l),
                })
                .collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };

    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        for idx in [0usize, n / 2, n - 1] {
            assert_eq!(
                half_bits_at(ft, idx),
                half_bits_at(ut, idx),
                "fused bf16 != unfused"
            );
        }
    }

    let iters = 60;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;

    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(unfused());
    }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "EVAL_FUSION_SPEED_BF16 n={n} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6,
        fused / 1e6,
        unf / fused,
    );
}

/// Clamp/relu/abs activation chain (f32 — JAX's default inference dtype). Before
/// Max/Min/Abs were fusable, this chain fused NOTHING (the first Max broke the run
/// and the prefix was below FUSION_MIN_RUN), so eval_jaxpr ran 8 separate
/// materializing passes — identical to the unfused arm. Now it fuses into one pass.
fn run_f32_clamp() {
    let n = 1usize << 20; // 1M f32 = 4 MB per tensor
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 1e-5 - 5.0).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32 * 3e-7).cos() * 4.0).collect();

    // v1=abs(x); v2=max(v1,0); v3=mul(v2,y); v4=min(v3,6); v5=max(v4,0);
    // v6=sub(v5,x); v7=max(v6,y); out=mul(v7,0.5)  (8 ops, single-use chain)
    let xv = VarId(0);
    let yv = VarId(1);
    let v: Vec<VarId> = (2..=9).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f32| Atom::Lit(Literal::from_f32(c));
    let eqns = vec![
        mk(Primitive::Abs, smallvec![Atom::Var(xv)], v[0]),
        mk(Primitive::Max, smallvec![Atom::Var(v[0]), lit(0.0)], v[1]),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[1]), Atom::Var(yv)],
            v[2],
        ),
        mk(Primitive::Min, smallvec![Atom::Var(v[2]), lit(6.0)], v[3]),
        mk(Primitive::Max, smallvec![Atom::Var(v[3]), lit(0.0)], v[4]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[4]), Atom::Var(xv)],
            v[5],
        ),
        mk(
            Primitive::Max,
            smallvec![Atom::Var(v[5]), Atom::Var(yv)],
            v[6],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[6]), lit(0.5)], v[7]),
    ];
    let jaxpr = Jaxpr::new(vec![xv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [f32_tensor(x.clone()), f32_tensor(y.clone())];

    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 10];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn
                .inputs
                .iter()
                .map(|a| match a {
                    Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                    Atom::Lit(l) => Value::Scalar(*l),
                })
                .collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };

    // Correctness: fused == unfused, bit-for-bit across all elements.
    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        for i in 0..n {
            assert_eq!(
                f32_bits_at(ft, i),
                f32_bits_at(ut, i),
                "fused != unfused at {i}"
            );
        }
    }

    let iters = 60;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;

    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(unfused());
    }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;

    println!(
        "EVAL_FUSION_SPEED_F32_CLAMP n={n} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6,
        fused / 1e6,
        unf / fused,
    );
}

// Measures the bf16 row-broadcast fusion (bead bjqfr): [rows,cols] bf16 tensor +
// [cols] bf16 bias chain. The bf16 SAME-SHAPE fusion measured 0.93x (decode/encode
// overhead); this isolates whether the broadcast variant (less memory traffic) pays.
fn run_bf16_row_broadcast() {
    let rows = 1024usize;
    let cols = 1024usize;
    let n = rows * cols;
    let xb: Vec<u16> = (0..n).map(|i| bf16_bits_of(i as f64 * 1e-6 - 0.5)).collect();
    let yb: Vec<u16> = (0..n).map(|i| bf16_bits_of((i as f64 * 7e-7).sin() + 1.1)).collect();
    let bias: Vec<u16> = (0..cols).map(|i| bf16_bits_of(i as f64 * 2e-4 - 0.25)).collect();
    let t2 = |vals: Vec<u16>| {
        Value::Tensor(
            TensorValue::new_half_float_values(DType::BF16, Shape { dims: vec![rows as u32, cols as u32] }, vals).unwrap(),
        )
    };
    let row = |vals: Vec<u16>| {
        Value::Tensor(
            TensorValue::new_half_float_values(DType::BF16, Shape { dims: vec![cols as u32] }, vals).unwrap(),
        )
    };
    let xv = VarId(0);
    let bv = VarId(1);
    let yv = VarId(2);
    let v: Vec<VarId> = (3..=10).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p, inputs: ins, outputs: smallvec![o], params: BTreeMap::new(), sub_jaxprs: vec![], effects: vec![],
    };
    let lit = |c: f64| Atom::Lit(Literal::from_bf16_f64(c));
    let eqns = vec![
        mk(Primitive::Add, smallvec![Atom::Var(xv), Atom::Var(bv)], v[0]),
        mk(Primitive::Mul, smallvec![Atom::Var(v[0]), lit(1.25)], v[1]),
        mk(Primitive::Sub, smallvec![Atom::Var(v[1]), Atom::Var(bv)], v[2]),
        mk(Primitive::Mul, smallvec![Atom::Var(v[2]), Atom::Var(yv)], v[3]),
        mk(Primitive::Add, smallvec![Atom::Var(v[3]), Atom::Var(bv)], v[4]),
        mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit(0.5)], v[5]),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(Primitive::Add, smallvec![Atom::Var(v[6]), Atom::Var(bv)], v[7]),
    ];
    let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [t2(xb.clone()), row(bias.clone()), t2(yb.clone())];
    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 11];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        env[2] = Some(args[2].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn.inputs.iter().map(|a| match a {
                Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                Atom::Lit(l) => Value::Scalar(*l),
            }).collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };
    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        for idx in [0, cols - 1, n / 2, n - 1] {
            assert_eq!(half_bits_at(ft, idx), half_bits_at(ut, idx), "fused bf16 row-broadcast != unfused");
        }
    }
    let iters = 50;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters { std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap()); }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;
    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters { std::hint::black_box(unfused()); }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;
    println!(
        "EVAL_FUSION_SPEED_BF16_ROW_BROADCAST rows={rows} cols={cols} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6, fused / 1e6, unf / fused,
    );
}

// Measures the i64 row-broadcast fusion (bead rl9ha). i64 SAME-SHAPE measured 1.59x.
fn run_i64_row_broadcast() {
    let rows = 1024usize;
    let cols = 1024usize;
    let n = rows * cols;
    let xi: Vec<i64> = (0..n).map(|i| i as i64).collect();
    let yi: Vec<i64> = (0..n).map(|i| (i as i64 % 97) + 1).collect();
    let bias: Vec<i64> = (0..cols).map(|i| i as i64 - 128).collect();
    let t2 = |vals: Vec<i64>| {
        Value::Tensor(TensorValue::new_i64_values(Shape { dims: vec![rows as u32, cols as u32] }, vals).unwrap())
    };
    let row = |vals: Vec<i64>| {
        Value::Tensor(TensorValue::new_i64_values(Shape { dims: vec![cols as u32] }, vals).unwrap())
    };
    let xv = VarId(0);
    let bv = VarId(1);
    let yv = VarId(2);
    let v: Vec<VarId> = (3..=10).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p, inputs: ins, outputs: smallvec![o], params: BTreeMap::new(), sub_jaxprs: vec![], effects: vec![],
    };
    let lit = |c: i64| Atom::Lit(Literal::I64(c));
    let eqns = vec![
        mk(Primitive::Add, smallvec![Atom::Var(xv), Atom::Var(bv)], v[0]),
        mk(Primitive::Mul, smallvec![Atom::Var(v[0]), lit(3)], v[1]),
        mk(Primitive::Sub, smallvec![Atom::Var(v[1]), Atom::Var(bv)], v[2]),
        mk(Primitive::Mul, smallvec![Atom::Var(v[2]), Atom::Var(yv)], v[3]),
        mk(Primitive::Add, smallvec![Atom::Var(v[3]), Atom::Var(bv)], v[4]),
        mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit(5)], v[5]),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2)], v[6]),
        mk(Primitive::Add, smallvec![Atom::Var(v[6]), Atom::Var(bv)], v[7]),
    ];
    let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns.clone());
    let args = [t2(xi.clone()), row(bias.clone()), t2(yi.clone())];
    let unfused = || {
        let mut env: Vec<Option<Value>> = vec![None; 11];
        env[0] = Some(args[0].clone());
        env[1] = Some(args[1].clone());
        env[2] = Some(args[2].clone());
        for eqn in &eqns {
            let ins: Vec<Value> = eqn.inputs.iter().map(|a| match a {
                Atom::Var(vr) => env[vr.0 as usize].clone().unwrap(),
                Atom::Lit(l) => Value::Scalar(*l),
            }).collect();
            let out = eval_primitive(eqn.primitive, &ins, &eqn.params).unwrap();
            env[eqn.outputs[0].0 as usize] = Some(out);
        }
        env[v[7].0 as usize].clone().unwrap()
    };
    let f = eval_jaxpr(&jaxpr, &args).unwrap();
    let u = unfused();
    if let (Value::Tensor(ft), Value::Tensor(ut)) = (&f[0], &u) {
        for idx in [0usize, cols - 1, n / 2, n - 1] {
            assert_eq!(
                ft.elements.as_i64_slice().unwrap()[idx],
                ut.elements.as_i64_slice().unwrap()[idx],
                "fused i64 row-broadcast != unfused"
            );
        }
    }
    let iters = 50;
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters { std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap()); }
    let fused = t0.elapsed().as_nanos() as f64 / iters as f64;
    let _ = unfused();
    let t1 = Instant::now();
    for _ in 0..iters { std::hint::black_box(unfused()); }
    let unf = t1.elapsed().as_nanos() as f64 / iters as f64;
    println!(
        "EVAL_FUSION_SPEED_I64_ROW_BROADCAST rows={rows} cols={cols} ops=8 unfused={:.3}ms fused={:.3}ms speedup={:.2}x",
        unf / 1e6, fused / 1e6, unf / fused,
    );
}

/// Same-invocation A/B for the THREADED fusion driver: serial (worker cap = 1)
/// vs the production threaded gate, over identical data in ONE process. This is
/// the only worker-variance-immune signal on the contended rch host — absolute
/// ms drift 4-10x across rch invocations, so cross-run comparison is meaningless;
/// the in-process serial/threaded ratio is not.
fn run_f64_thread_ab(n: usize) {
    use std::sync::atomic::Ordering;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 1e-6 - 0.5).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 3e-7).cos() + 1.2).collect();
    let xv = VarId(0);
    let yv = VarId(1);
    let v: Vec<VarId> = (2..=9).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f64| Atom::Lit(Literal::from_f64(c));
    let eqns = vec![
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(xv), Atom::Var(xv)],
            v[0],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[0]), lit(0.5)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(xv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[3]), lit(1.0)], v[4]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[4]), Atom::Var(yv)],
            v[5],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(xv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, yv], vec![], vec![v[7]], eqns);
    let args = [f64_tensor(x), f64_tensor(y)];

    // Serial vs threaded must be bit-identical.
    FUSION_THREAD_CAP_OVERRIDE.store(1, Ordering::Relaxed);
    let serial_out = eval_jaxpr(&jaxpr, &args).unwrap();
    FUSION_THREAD_CAP_OVERRIDE.store(0, Ordering::Relaxed);
    let threaded_out = eval_jaxpr(&jaxpr, &args).unwrap();
    if let (Value::Tensor(s), Value::Tensor(t)) = (&serial_out[0], &threaded_out[0]) {
        for idx in [0, n / 2, n - 1] {
            assert_eq!(
                f64_bits_at(s, idx),
                f64_bits_at(t, idx),
                "serial != threaded fused at {idx}"
            );
        }
    }

    let iters = if n >= (1 << 23) { 30 } else { 80 };

    FUSION_THREAD_CAP_OVERRIDE.store(1, Ordering::Relaxed);
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let serial = t0.elapsed().as_nanos() as f64 / iters as f64;

    FUSION_THREAD_CAP_OVERRIDE.store(0, Ordering::Relaxed);
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let threaded = t1.elapsed().as_nanos() as f64 / iters as f64;
    FUSION_THREAD_CAP_OVERRIDE.store(0, Ordering::Relaxed);

    println!(
        "EVAL_FUSION_THREAD_AB_F64 n={n} ops=8 serial={:.3}ms threaded={:.3}ms speedup={:.2}x",
        serial / 1e6,
        threaded / 1e6,
        serial / threaded,
    );
}

/// bf16 same-invocation serial-vs-threaded A/B. bf16 elementwise is COMPUTE-bound
/// (every op decodes u16->f32, computes, re-rounds), so the fused chain is ~13x
/// slower than f64 and far from bandwidth-limited — threading should scale better
/// than the f64 chain. Mirrors `run_bf16`'s 8-op chain.
fn run_bf16_thread_ab(n: usize) {
    use std::sync::atomic::Ordering;
    let x: Vec<u16> = (0..n)
        .map(|i| bf16_bits_of(i as f64 * 1e-6 - 0.5))
        .collect();
    let y: Vec<u16> = (0..n)
        .map(|i| bf16_bits_of((i as f64 * 3e-7).cos() + 1.2))
        .collect();
    let xv = VarId(0);
    let yv = VarId(1);
    let v: Vec<VarId> = (2..=9).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f64| Atom::Lit(Literal::from_bf16_f64(c));
    let eqns = vec![
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(xv), Atom::Var(xv)],
            v[0],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[0]), lit(0.5)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(xv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[3]), lit(1.0)], v[4]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[4]), Atom::Var(yv)],
            v[5],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(xv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, yv], vec![], vec![v[7]], eqns);
    let args = [bf16_tensor(x), bf16_tensor(y)];

    FUSION_THREAD_CAP_OVERRIDE.store(1, Ordering::Relaxed);
    let serial_out = eval_jaxpr(&jaxpr, &args).unwrap();
    FUSION_THREAD_CAP_OVERRIDE.store(0, Ordering::Relaxed);
    let threaded_out = eval_jaxpr(&jaxpr, &args).unwrap();
    if let (Value::Tensor(s), Value::Tensor(t)) = (&serial_out[0], &threaded_out[0]) {
        for idx in [0, n / 2, n - 1] {
            assert_eq!(
                half_bits_at(s, idx),
                half_bits_at(t, idx),
                "serial != threaded bf16 at {idx}"
            );
        }
    }

    let iters = if n >= (1 << 23) { 30 } else { 80 };

    FUSION_THREAD_CAP_OVERRIDE.store(1, Ordering::Relaxed);
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let serial = t0.elapsed().as_nanos() as f64 / iters as f64;

    FUSION_THREAD_CAP_OVERRIDE.store(0, Ordering::Relaxed);
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let threaded = t1.elapsed().as_nanos() as f64 / iters as f64;
    FUSION_THREAD_CAP_OVERRIDE.store(0, Ordering::Relaxed);

    println!(
        "EVAL_FUSION_THREAD_AB_BF16 n={n} ops=8 serial={:.3}ms threaded={:.3}ms speedup={:.2}x",
        serial / 1e6,
        threaded / 1e6,
        serial / threaded,
    );
}

/// f32 same-invocation A/B (JAX's default dtype; bandwidth-bound like f64).
fn run_f32_thread_ab(n: usize) {
    use std::sync::atomic::Ordering;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 1e-6 - 0.5).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32 * 3e-7).cos() + 1.2).collect();
    let xv = VarId(0);
    let yv = VarId(1);
    let v: Vec<VarId> = (2..=9).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: f32| Atom::Lit(Literal::from_f32(c));
    let eqns = vec![
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(xv), Atom::Var(xv)],
            v[0],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[0]), lit(0.5)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(xv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[3]), lit(1.0)], v[4]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[4]), Atom::Var(yv)],
            v[5],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(xv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, yv], vec![], vec![v[7]], eqns);
    let args = [f32_tensor(x), f32_tensor(y)];

    FUSION_THREAD_CAP_OVERRIDE.store(1, Ordering::Relaxed);
    let so = eval_jaxpr(&jaxpr, &args).unwrap();
    FUSION_THREAD_CAP_OVERRIDE.store(0, Ordering::Relaxed);
    let to = eval_jaxpr(&jaxpr, &args).unwrap();
    if let (Value::Tensor(s), Value::Tensor(t)) = (&so[0], &to[0]) {
        for idx in [0, n / 2, n - 1] {
            assert_eq!(f32_bits_at(s, idx), f32_bits_at(t, idx), "serial != threaded f32");
        }
    }
    let iters = if n >= (1 << 23) { 30 } else { 80 };
    FUSION_THREAD_CAP_OVERRIDE.store(1, Ordering::Relaxed);
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let serial = t0.elapsed().as_nanos() as f64 / iters as f64;
    FUSION_THREAD_CAP_OVERRIDE.store(0, Ordering::Relaxed);
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let threaded = t1.elapsed().as_nanos() as f64 / iters as f64;
    println!(
        "EVAL_FUSION_THREAD_AB_F32 n={n} ops=8 serial={:.3}ms threaded={:.3}ms speedup={:.2}x",
        serial / 1e6,
        threaded / 1e6,
        serial / threaded,
    );
}

/// i64 same-invocation A/B (wrapping integer chain, no Div).
fn run_i64_thread_ab(n: usize) {
    use std::sync::atomic::Ordering;
    let x: Vec<i64> = (0..n).map(|i| i as i64 - (n as i64 / 2)).collect();
    let y: Vec<i64> = (0..n).map(|i| (i as i64 % 13) - 6).collect();
    let xv = VarId(0);
    let yv = VarId(1);
    let v: Vec<VarId> = (2..=9).map(VarId).collect();
    let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
        primitive: p,
        inputs: ins,
        outputs: smallvec![o],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
        effects: vec![],
    };
    let lit = |c: i64| Atom::Lit(Literal::I64(c));
    let eqns = vec![
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(xv), Atom::Var(xv)],
            v[0],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[0]), lit(3)], v[1]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[1]), Atom::Var(xv)],
            v[2],
        ),
        mk(
            Primitive::Mul,
            smallvec![Atom::Var(v[2]), Atom::Var(yv)],
            v[3],
        ),
        mk(Primitive::Add, smallvec![Atom::Var(v[3]), lit(11)], v[4]),
        mk(
            Primitive::Sub,
            smallvec![Atom::Var(v[4]), Atom::Var(yv)],
            v[5],
        ),
        mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2)], v[6]),
        mk(
            Primitive::Add,
            smallvec![Atom::Var(v[6]), Atom::Var(xv)],
            v[7],
        ),
    ];
    let jaxpr = Jaxpr::new(vec![xv, yv], vec![], vec![v[7]], eqns);
    let args = [i64_tensor(x), i64_tensor(y)];

    FUSION_THREAD_CAP_OVERRIDE.store(1, Ordering::Relaxed);
    let so = eval_jaxpr(&jaxpr, &args).unwrap();
    FUSION_THREAD_CAP_OVERRIDE.store(0, Ordering::Relaxed);
    let to = eval_jaxpr(&jaxpr, &args).unwrap();
    if let (Value::Tensor(s), Value::Tensor(t)) = (&so[0], &to[0]) {
        let ss = s.elements.as_i64_slice();
        let ts = t.elements.as_i64_slice();
        if let (Some(ss), Some(ts)) = (ss, ts) {
            for idx in [0, n / 2, n - 1] {
                assert_eq!(ss[idx], ts[idx], "serial != threaded i64");
            }
        }
    }
    let iters = if n >= (1 << 23) { 30 } else { 80 };
    FUSION_THREAD_CAP_OVERRIDE.store(1, Ordering::Relaxed);
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let serial = t0.elapsed().as_nanos() as f64 / iters as f64;
    FUSION_THREAD_CAP_OVERRIDE.store(0, Ordering::Relaxed);
    let _ = eval_jaxpr(&jaxpr, &args).unwrap();
    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(eval_jaxpr(&jaxpr, &args).unwrap());
    }
    let threaded = t1.elapsed().as_nanos() as f64 / iters as f64;
    println!(
        "EVAL_FUSION_THREAD_AB_I64 n={n} ops=8 serial={:.3}ms threaded={:.3}ms speedup={:.2}x",
        serial / 1e6,
        threaded / 1e6,
        serial / threaded,
    );
}

fn main() {
    run_f64_thread_ab(1usize << 20); // 1M  - L3-resident on the bench host
    run_f64_thread_ab(1usize << 22); // 4M  - L3 boundary
    run_f64_thread_ab(1usize << 24); // 16M - DRAM-bound, beyond L3
    run_f32_thread_ab(1usize << 20); // 1M  - JAX default dtype, BW-bound
    run_f32_thread_ab(1usize << 24); // 16M - DRAM-bound
    run_i64_thread_ab(1usize << 20); // 1M
    run_i64_thread_ab(1usize << 24); // 16M
    run_bf16_thread_ab(1usize << 20); // 1M  - compute-bound (decode/encode)
    run_bf16_thread_ab(1usize << 22); // 4M
    run_f64();
    run_f32();
    run_f32_clamp();
    run_f32_row_broadcast();
    run_f32_col_broadcast();
    run_f64_row_broadcast();
    run_f64_col_broadcast();
    run_i64();
    run_bf16();
    run_bf16_row_broadcast();
    run_i64_row_broadcast();
}
