use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fj_core::{
    Atom, CompatibilityMode, DType, Equation, Jaxpr, Literal, Primitive, ProgramSpec, Shape,
    TensorValue, TraceTransformLedger, Transform, Value, VarId, build_program,
    verify_transform_composition,
};
use fj_dispatch::{DispatchRequest, dispatch};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_ledger(spec: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(build_program(spec));
    for (idx, t) in transforms.iter().enumerate() {
        ledger.push_transform(*t, format!("ev-{idx}"));
    }
    ledger
}

fn dispatch_request(
    spec: ProgramSpec,
    transforms: &[Transform],
    args: Vec<Value>,
) -> DispatchRequest {
    DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: make_ledger(spec, transforms),
        args,
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    }
}

fn dispatch_jaxpr_request(
    jaxpr: Jaxpr,
    transforms: &[Transform],
    args: Vec<Value>,
) -> DispatchRequest {
    let mut ledger = TraceTransformLedger::new(jaxpr);
    for (idx, transform) in transforms.iter().enumerate() {
        ledger.push_transform(*transform, format!("ev-{idx}"));
    }
    DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger,
        args,
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    }
}

/// Build a synthetic Jaxpr with `n` equations chaining add operations.
fn build_chain_jaxpr(n: usize) -> Jaxpr {
    let mut equations = Vec::with_capacity(n);
    // in: v1, out: v(n+1)
    for i in 0..n {
        let input_var = VarId((i + 1) as u32);
        let output_var = VarId((i + 2) as u32);
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![Atom::Var(input_var), Atom::Lit(Literal::I64(1))],
            outputs: smallvec::smallvec![output_var],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        });
    }
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId((n + 1) as u32)],
        equations,
    )
}

fn switch_branch_identity_jaxpr() -> Jaxpr {
    Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![])
}

fn switch_branch_self_binary_jaxpr(primitive: Primitive) -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        }],
    )
}

fn switch_control_flow_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Switch,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([("num_branches".to_owned(), "3".to_owned())]),
            sub_jaxprs: vec![
                switch_branch_identity_jaxpr(),
                switch_branch_self_binary_jaxpr(Primitive::Add),
                switch_branch_self_binary_jaxpr(Primitive::Mul),
            ],
            effects: vec![],
        }],
    )
}

fn scan_control_flow_jaxpr(body_op: &str) -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Scan,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::from([("body_op".to_owned(), body_op.to_owned())]),
            sub_jaxprs: vec![],
            effects: vec![],
        }],
    )
}

// ---------------------------------------------------------------------------
// 1. Dispatch Latency — jit/grad/vmap x scalar/vector
// ---------------------------------------------------------------------------

fn bench_dispatch_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch_latency");

    // jit scalar add
    group.bench_function("jit/scalar_add", |b| {
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::Add2,
                &[Transform::Jit],
                vec![Value::scalar_i64(2), Value::scalar_i64(5)],
            ))
            .expect("jit scalar add should succeed");
        });
    });

    // jit scalar square_plus_linear (3 equations)
    group.bench_function("jit/scalar_square_plus_linear", |b| {
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::SquarePlusLinear,
                &[Transform::Jit],
                vec![Value::scalar_i64(7)],
            ))
            .expect("jit square_plus_linear should succeed");
        });
    });

    // jit vector add_one
    group.bench_function("jit/vector_add_one", |b| {
        let vec_arg = Value::vector_i64(&[1, 2, 3, 4, 5]).expect("vector should build");
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::AddOne,
                &[Transform::Jit],
                vec![vec_arg.clone()],
            ))
            .expect("jit vector add_one should succeed");
        });
    });

    // grad scalar square -> derivative = 2*x
    group.bench_function("grad/scalar_square", |b| {
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::Square,
                &[Transform::Grad],
                vec![Value::scalar_f64(3.0)],
            ))
            .expect("grad scalar square should succeed");
        });
    });

    // vmap vector add_one
    group.bench_function("vmap/vector_add_one", |b| {
        let vec_arg = Value::vector_i64(&[1, 2, 3, 4, 5]).expect("vector should build");
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::AddOne,
                &[Transform::Vmap],
                vec![vec_arg.clone()],
            ))
            .expect("vmap vector add_one should succeed");
        });
    });

    // vmap rank-2 add_one
    group.bench_function("vmap/rank2_add_one", |b| {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![4, 3] },
                (1..=12).map(Literal::I64).collect(),
            )
            .expect("matrix should build"),
        );
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::AddOne,
                &[Transform::Vmap],
                vec![matrix.clone()],
            ))
            .expect("vmap rank2 add_one should succeed");
        });
    });

    // jit(grad(f)) composed
    group.bench_function("jit_grad/scalar_square", |b| {
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::Square,
                &[Transform::Jit, Transform::Grad],
                vec![Value::scalar_f64(3.0)],
            ))
            .expect("jit(grad) should succeed");
        });
    });

    // vmap(grad(f)) composed
    group.bench_function("vmap_grad/vector_square", |b| {
        let vec_arg = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build");
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::Square,
                &[Transform::Vmap, Transform::Grad],
                vec![vec_arg.clone()],
            ))
            .expect("vmap(grad) should succeed");
        });
    });

    group.finish();
}

fn bench_vmap_gather(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmap_gather");

    let operand = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape::vector(4096),
            (0_i64..4096).map(Literal::I64).collect(),
        )
        .expect("operand should build"),
    );
    let indices = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 16],
            },
            (0..(128 * 16))
                .map(|idx| Literal::I64((idx % 4096) as i64))
                .collect(),
        )
        .expect("indices should build"),
    );

    group.bench_function("batched_indices", |b| {
        b.iter(|| {
            let mut request = dispatch_request(
                ProgramSpec::LaxGather1d,
                &[Transform::Vmap],
                vec![operand.clone(), indices.clone()],
            );
            request
                .compile_options
                .insert("vmap_in_axes".to_owned(), "none,0".to_owned());
            dispatch(request).expect("vmap gather should succeed");
        });
    });

    let batched_operand = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 4096],
            },
            (0..(128 * 4096))
                .map(|idx| Literal::I64((idx % 4096) as i64))
                .collect(),
        )
        .expect("batched operand should build"),
    );
    let shared_indices = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape::vector(16),
            (0..16).map(|idx| Literal::I64((idx * 17) as i64)).collect(),
        )
        .expect("shared indices should build"),
    );

    group.bench_function("batched_operand_shared_indices", |b| {
        b.iter(|| {
            let mut request = dispatch_request(
                ProgramSpec::LaxGather1d,
                &[Transform::Vmap],
                vec![batched_operand.clone(), shared_indices.clone()],
            );
            request
                .compile_options
                .insert("vmap_in_axes".to_owned(), "0,none".to_owned());
            dispatch(request).expect("vmap gather should succeed");
        });
    });

    let batched_operand_indices = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 16],
            },
            (0..(128 * 16))
                .map(|idx| Literal::I64(((idx * 17) % 4096) as i64))
                .collect(),
        )
        .expect("batched operand indices should build"),
    );

    group.bench_function("batched_operand_batched_indices", |b| {
        b.iter(|| {
            let mut request = dispatch_request(
                ProgramSpec::LaxGather1d,
                &[Transform::Vmap],
                vec![batched_operand.clone(), batched_operand_indices.clone()],
            );
            request
                .compile_options
                .insert("vmap_in_axes".to_owned(), "0,0".to_owned());
            dispatch(request).expect("vmap gather should succeed");
        });
    });

    group.finish();
}

fn bench_vmap_scatter(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmap_scatter");

    let operand = Value::Tensor(
        TensorValue::new(DType::I64, Shape::vector(4096), vec![Literal::I64(0); 4096])
            .expect("operand should build"),
    );
    let indices = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 16],
            },
            (0..(128 * 16))
                .map(|idx| Literal::I64((idx % 4096) as i64))
                .collect(),
        )
        .expect("indices should build"),
    );
    let updates = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 16],
            },
            (0_i64..(128 * 16)).map(Literal::I64).collect(),
        )
        .expect("updates should build"),
    );

    group.bench_function("batched_indices_updates", |b| {
        b.iter(|| {
            let mut request = dispatch_request(
                ProgramSpec::LaxScatterOverwrite,
                &[Transform::Vmap],
                vec![operand.clone(), indices.clone(), updates.clone()],
            );
            request
                .compile_options
                .insert("vmap_in_axes".to_owned(), "none,0,0".to_owned());
            dispatch(request).expect("vmap scatter should succeed");
        });
    });

    group.finish();
}

fn bench_vmap_dot_i64(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmap_dot_i64");

    let batch = 128_usize;
    let width = 64_usize;
    let lhs = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![batch as u32, width as u32],
            },
            (0..(batch * width))
                .map(|idx| Literal::I64((idx % 17) as i64))
                .collect(),
        )
        .expect("lhs should build"),
    );
    let rhs = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![batch as u32, width as u32],
            },
            (0..(batch * width))
                .map(|idx| Literal::I64(((idx * 3) % 19) as i64))
                .collect(),
        )
        .expect("rhs should build"),
    );

    group.bench_function("paired_vectors", |b| {
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::Dot3,
                &[Transform::Vmap],
                vec![lhs.clone(), rhs.clone()],
            ))
            .expect("vmap dot should succeed");
        });
    });

    group.finish();
}

fn bench_vmap_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmap_qr");

    let batched_matrix = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![512, 3, 2],
            },
            (0..512)
                .flat_map(|batch| {
                    let scale = 1.0 + f64::from(batch % 11) * 0.01;
                    [
                        Literal::F64Bits(scale.to_bits()),
                        Literal::F64Bits(0.0_f64.to_bits()),
                        Literal::F64Bits(0.0_f64.to_bits()),
                        Literal::F64Bits(scale.to_bits()),
                        Literal::F64Bits(scale.to_bits()),
                        Literal::F64Bits(scale.to_bits()),
                    ]
                })
                .collect(),
        )
        .expect("batched QR input should build"),
    );

    group.bench_function("batched_matrix_3x2", |b| {
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::LaxQr,
                &[Transform::Vmap],
                vec![batched_matrix.clone()],
            ))
            .expect("vmap QR should succeed");
        });
    });

    group.finish();
}

fn bench_vmap_eigh(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmap_eigh");

    let batched_matrix = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![512, 3, 3],
            },
            (0..512)
                .flat_map(|batch| {
                    let shift = f64::from(batch % 13) * 0.01;
                    [
                        Literal::F64Bits((2.0 + shift).to_bits()),
                        Literal::F64Bits(0.1_f64.to_bits()),
                        Literal::F64Bits(0.0_f64.to_bits()),
                        Literal::F64Bits(0.1_f64.to_bits()),
                        Literal::F64Bits((3.0 + shift).to_bits()),
                        Literal::F64Bits(0.2_f64.to_bits()),
                        Literal::F64Bits(0.0_f64.to_bits()),
                        Literal::F64Bits(0.2_f64.to_bits()),
                        Literal::F64Bits((4.0 + shift).to_bits()),
                    ]
                })
                .collect(),
        )
        .expect("batched Eigh input should build"),
    );

    group.bench_function("batched_matrix_3x3", |b| {
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::LaxEigh,
                &[Transform::Vmap],
                vec![batched_matrix.clone()],
            ))
            .expect("vmap Eigh should succeed");
        });
    });

    group.finish();
}

fn bench_vmap_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmap_svd");

    let batched_matrix = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![512, 3, 2],
            },
            (0..512)
                .flat_map(|batch| {
                    let scale = 1.0 + f64::from(batch % 11) * 0.01;
                    [
                        Literal::F64Bits(scale.to_bits()),
                        Literal::F64Bits(0.1_f64.to_bits()),
                        Literal::F64Bits(0.0_f64.to_bits()),
                        Literal::F64Bits((scale + 0.5).to_bits()),
                        Literal::F64Bits(0.2_f64.to_bits()),
                        Literal::F64Bits((scale + 1.0).to_bits()),
                    ]
                })
                .collect(),
        )
        .expect("batched SVD input should build"),
    );

    group.bench_function("batched_matrix_3x2", |b| {
        b.iter(|| {
            dispatch(dispatch_request(
                ProgramSpec::LaxSvd,
                &[Transform::Vmap],
                vec![batched_matrix.clone()],
            ))
            .expect("vmap SVD should succeed");
        });
    });

    group.finish();
}

fn bench_vmap_switch(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmap_switch");

    let jaxpr = switch_control_flow_jaxpr();
    let indices: Vec<i64> = (0..128)
        .map(|idx| match idx % 5 {
            0 => -1,
            1 => 0,
            2 => 1,
            3 => 2,
            _ => 99,
        })
        .collect();
    let operands: Vec<i64> = (1..=128).collect();
    let index_value = Value::vector_i64(&indices).expect("switch index vector should build");
    let operand_value = Value::vector_i64(&operands).expect("switch operand vector should build");

    group.bench_function("batched_index_128", |b| {
        b.iter(|| {
            dispatch(dispatch_jaxpr_request(
                jaxpr.clone(),
                &[Transform::Vmap],
                vec![index_value.clone(), operand_value.clone()],
            ))
            .expect("vmap(switch) should dispatch");
        });
    });

    group.finish();
}

fn bench_vmap_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmap_scan");

    let jaxpr = scan_control_flow_jaxpr("add");
    let xs = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![128, 64],
            },
            (0..(128 * 64))
                .map(|idx| Literal::I64((idx % 17) as i64))
                .collect(),
        )
        .expect("scan xs should build"),
    );

    group.bench_function("shared_init_batched_xs_128x64", |b| {
        b.iter(|| {
            let mut request = dispatch_jaxpr_request(
                jaxpr.clone(),
                &[Transform::Vmap],
                vec![Value::scalar_i64(0), xs.clone()],
            );
            request
                .compile_options
                .insert("vmap_in_axes".to_owned(), "none,0".to_owned());
            dispatch(request).expect("vmap(scan) should dispatch");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. eval_jaxpr Throughput — 10, 100, 1000 equation programs
// ---------------------------------------------------------------------------

fn bench_eval_jaxpr_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_jaxpr_throughput");

    for n in [10, 100, 1000] {
        let jaxpr = build_chain_jaxpr(n);
        group.bench_with_input(BenchmarkId::new("chain_add", n), &jaxpr, |b, jaxpr| {
            b.iter(|| {
                fj_interpreters::eval_jaxpr(jaxpr, &[Value::scalar_i64(0)])
                    .expect("chain eval should succeed");
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Transform Composition Overhead — depth 1..5
// ---------------------------------------------------------------------------

fn bench_transform_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_composition");

    // Single transforms
    for t in [Transform::Jit, Transform::Grad, Transform::Vmap] {
        let name = format!("single/{}", t.as_str());
        group.bench_function(&name, |b| {
            let ledger = make_ledger(ProgramSpec::Square, &[t]);
            b.iter(|| {
                verify_transform_composition(&ledger).expect("single should pass");
            });
        });
    }

    // Depth 2: jit+grad
    group.bench_function("depth2/jit_grad", |b| {
        let ledger = make_ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]);
        b.iter(|| {
            verify_transform_composition(&ledger).expect("depth-2 should pass");
        });
    });

    // Depth 3: jit+vmap+grad
    group.bench_function("depth3/jit_vmap_grad", |b| {
        let ledger = make_ledger(
            ProgramSpec::Square,
            &[Transform::Jit, Transform::Vmap, Transform::Grad],
        );
        b.iter(|| {
            verify_transform_composition(&ledger).expect("depth-3 should pass");
        });
    });

    // Empty stack
    group.bench_function("empty_stack", |b| {
        let ledger = make_ledger(ProgramSpec::Square, &[]);
        b.iter(|| {
            verify_transform_composition(&ledger).expect("empty should pass");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Cache Key Generation
// ---------------------------------------------------------------------------

fn bench_cache_key_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_key_generation");

    // Simple (1-equation jaxpr, 1 transform)
    group.bench_function("simple/1eq_1t", |b| {
        let jaxpr = build_program(ProgramSpec::Add2);
        let transforms = vec![Transform::Jit];
        let compile_options = BTreeMap::new();
        let unknown: Vec<String> = vec![];
        b.iter(|| {
            fj_cache::build_cache_key_ref(&fj_cache::CacheKeyInputRef {
                mode: CompatibilityMode::Strict,
                backend: "cpu",
                jaxpr: &jaxpr,
                transform_stack: &transforms,
                compile_options: &compile_options,
                custom_hook: None,
                unknown_incompatible_features: &unknown,
            })
            .expect("cache key should succeed");
        });
    });

    // Medium (3-equation jaxpr, 2 transforms)
    group.bench_function("medium/3eq_2t", |b| {
        let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
        let transforms = vec![Transform::Jit, Transform::Grad];
        let compile_options = BTreeMap::new();
        let unknown: Vec<String> = vec![];
        b.iter(|| {
            fj_cache::build_cache_key_ref(&fj_cache::CacheKeyInputRef {
                mode: CompatibilityMode::Strict,
                backend: "cpu",
                jaxpr: &jaxpr,
                transform_stack: &transforms,
                compile_options: &compile_options,
                custom_hook: None,
                unknown_incompatible_features: &unknown,
            })
            .expect("cache key should succeed");
        });
    });

    // Large (100-equation chain jaxpr)
    group.bench_function("large/100eq_1t", |b| {
        let jaxpr = build_chain_jaxpr(100);
        let transforms = vec![Transform::Jit];
        let compile_options = BTreeMap::new();
        let unknown: Vec<String> = vec![];
        b.iter(|| {
            fj_cache::build_cache_key_ref(&fj_cache::CacheKeyInputRef {
                mode: CompatibilityMode::Strict,
                backend: "cpu",
                jaxpr: &jaxpr,
                transform_stack: &transforms,
                compile_options: &compile_options,
                custom_hook: None,
                unknown_incompatible_features: &unknown,
            })
            .expect("cache key should succeed");
        });
    });

    // Hardened mode with unknown features
    group.bench_function("hardened/unknown_features", |b| {
        let jaxpr = build_program(ProgramSpec::Add2);
        let transforms = vec![Transform::Jit];
        let compile_options = BTreeMap::new();
        let unknown = vec!["future.protocol.v2".to_owned()];
        b.iter(|| {
            fj_cache::build_cache_key_ref(&fj_cache::CacheKeyInputRef {
                mode: CompatibilityMode::Hardened,
                backend: "cpu",
                jaxpr: &jaxpr,
                transform_stack: &transforms,
                compile_options: &compile_options,
                custom_hook: Some("custom-hook"),
                unknown_incompatible_features: &unknown,
            })
            .expect("hardened cache key should succeed");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Ledger Append Throughput
// ---------------------------------------------------------------------------

fn bench_ledger_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("ledger_append");

    group.bench_function("single_append", |b| {
        b.iter(|| {
            let mut ledger = fj_ledger::EvidenceLedger::new();
            let matrix = fj_ledger::LossMatrix::default();
            let record =
                fj_ledger::DecisionRecord::from_posterior(CompatibilityMode::Strict, 0.3, &matrix);
            ledger.append(fj_ledger::LedgerEntry {
                decision_id: "bench-key".to_owned(),
                record,
                signals: vec![fj_ledger::EvidenceSignal {
                    signal_name: "eqn_count".to_owned(),
                    log_likelihood_delta: 1.0_f64.ln(),
                    detail: "eqn_count=1".to_owned(),
                }],
            });
        });
    });

    group.bench_function("burst_100_appends", |b| {
        b.iter(|| {
            let mut ledger = fj_ledger::EvidenceLedger::new();
            let matrix = fj_ledger::LossMatrix::default();
            for i in 0..100 {
                let record = fj_ledger::DecisionRecord::from_posterior(
                    CompatibilityMode::Strict,
                    0.3,
                    &matrix,
                );
                ledger.append(fj_ledger::LedgerEntry {
                    decision_id: format!("bench-key-{i}"),
                    record,
                    signals: vec![fj_ledger::EvidenceSignal {
                        signal_name: "eqn_count".to_owned(),
                        log_likelihood_delta: (i as f64 + 1.0).ln(),
                        detail: format!("eqn_count={i}"),
                    }],
                });
            }
            assert_eq!(ledger.len(), 100);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. Jaxpr Fingerprint and Construction
// ---------------------------------------------------------------------------

fn bench_jaxpr_fingerprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("jaxpr_fingerprint");

    for n in [1, 10, 100] {
        let jaxpr = build_chain_jaxpr(n);
        group.bench_with_input(
            BenchmarkId::new("canonical_fingerprint", n),
            &jaxpr,
            |b, jaxpr| {
                b.iter(|| {
                    // Each clone gives a fresh OnceLock, measuring the actual computation
                    let fresh = jaxpr.clone();
                    let _fp = fresh.canonical_fingerprint();
                });
            },
        );
    }

    // Cached fingerprint (already computed)
    group.bench_function("cached_fingerprint/10eq", |b| {
        let jaxpr = build_chain_jaxpr(10);
        let _ = jaxpr.canonical_fingerprint(); // warm the cache
        b.iter(|| {
            let _fp = jaxpr.canonical_fingerprint();
        });
    });

    group.finish();
}

fn bench_jaxpr_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("jaxpr_validation");

    for n in [1, 10, 100] {
        let jaxpr = build_chain_jaxpr(n);
        group.bench_with_input(
            BenchmarkId::new("validate_well_formed", n),
            &jaxpr,
            |b, jaxpr| {
                b.iter(|| {
                    jaxpr.validate_well_formed().expect("should be valid");
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group registration
// ---------------------------------------------------------------------------

criterion_group!(
    dispatch_benches,
    bench_dispatch_latency,
    bench_vmap_gather,
    bench_vmap_scatter,
    bench_vmap_dot_i64,
    bench_vmap_qr,
    bench_vmap_eigh,
    bench_vmap_svd,
    bench_vmap_switch,
    bench_vmap_scan,
    bench_eval_jaxpr_throughput,
    bench_transform_composition,
    bench_cache_key_generation,
    bench_ledger_append,
    bench_jaxpr_fingerprint,
    bench_jaxpr_validation,
);
criterion_main!(dispatch_benches);
