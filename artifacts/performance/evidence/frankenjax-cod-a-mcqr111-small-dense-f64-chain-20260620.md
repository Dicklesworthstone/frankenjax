# frankenjax-mcqr.111 - in-place small dense f64 chain runner

Agent: WildForge / cod-a
Date: 2026-06-20

## Lever

`fj-interpreters` small dense-f64 tensor arena now recognizes the single-output
linear tensor-state chain case and moves the loaded input tensor buffer into the
runner output, mutating it in place across scalar/literal broadcast steps.

Fallbacks remain unchanged for multi-output plans, multiple tensor slots, row/col
broadcast tensors, non-f64 tensors, large tensors (`>= FUSION_MIN_ELEMS`), and
DAGs that read non-current tensor intermediates.

## Verification Commands

- Baseline: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-interpreters --bench compiled_dispatch_speed -- compiled_dispatch --warm-up-time 1 --measurement-time 5`
- Candidate: `RCH_WORKER=vmi1152480 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-interpreters --bench compiled_dispatch_speed -- compiled_dispatch --warm-up-time 1 --measurement-time 5`
- JAX comparator: `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/interpreter_compiled_dispatch_gauntlet.py --runs 80 --warmup 20 --inner-loops 200 --output artifacts/performance/evidence/frankenjax-cod-a-interpreters-jax-baseline-20260620T0105Z.json`
- Focused parity: `RCH_WORKER=vmi1152480 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-interpreters dense_f64_tensor_arena_bit_identical_to_generic --lib`

## Same-Worker Rust Results

Worker: `vmi1152480`.

| workload | baseline compiled_runner | candidate compiled_runner | speedup |
| --- | ---: | ---: | ---: |
| tensor64/n=8 | 2.2213 us | 1.1623 us | 1.91x |
| tensor64/n=32 | 8.3991 us | 4.4519 us | 1.89x |

## JAX Head-To-Head

JAX: `jax 0.10.1`, x64 enabled, CPU jit call + `block_until_ready`.

| workload | Rust candidate | JAX mean | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| scalar/n=8 | 28.642 ns | 5406.2266 ns | 0.0053 | Rust wins 188.7x |
| scalar/n=32 | 77.747 ns | 4724.7143 ns | 0.0165 | Rust wins 60.8x |
| scalar/n=128 | 264.19 ns | 4755.5858 ns | 0.0556 | Rust wins 18.0x |
| tensor64/n=8 | 1.1623 us | 4.6034999 us | 0.2525 | Rust wins 4.0x |
| tensor64/n=32 | 4.4519 us | 4.7659261 us | 0.9341 | Rust wins 1.07x |

Scorecard: 5 wins / 0 losses / 0 neutral vs JAX for current `compiled_runner`
rows. The important delta is the prior `tensor64/n=32` loss: it moved from
8.3991 us Rust vs 4.7659 us JAX (JAX 1.76x faster) to 4.4519 us Rust vs
4.7659 us JAX (Rust 1.07x faster by mean).

## Negative Evidence

The n=32 tensor win is narrow. This lane should not claim a large XLA-fusion win;
it mostly removes avoidable allocator traffic. Further work should target deeper
small-tensor algebra/codegen only if it preserves per-step floating-point order or
has an explicit relaxed-FP contract.
