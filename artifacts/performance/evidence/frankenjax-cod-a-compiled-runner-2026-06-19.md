# frankenjax-mcqr.110 - CompiledJaxpr runner arena

Date: 2026-06-19
Agent: WildForge / cod-a
Worktree: `/data/projects/.scratch/frankenjax-cod-a-runner-20260619140053`
Target dir: `/data/projects/.rch-targets/frankenjax-cod-a`

## Lever

`CompiledJaxpr::eval` rebuilt its slot environment and dense-plan scratch/output/scalar buffers on
every repeated eval. `run_dense_plan_into` already accepts caller-owned buffers, so this change adds
`CompiledJaxpr::runner()` and `CompiledJaxprRunner::{eval, eval_owned}` to keep those arenas warm
across calls. Primitive order and dense-plan semantics are unchanged.

## Rust Internal Benchmark

Baseline command:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo bench -p fj-interpreters --bench compiled_dispatch_speed -- \
  compiled_dispatch --warm-up-time 1 --measurement-time 5
```

Baseline worker: `vmi1153651`. Candidate command reused the same benchmark after adding the
`compiled_runner/*` arm; RCH selected `hz2`. Because the candidate run includes both `compiled/*`
and `compiled_runner/*` in the same Criterion process, the keep/reject proof uses same-process
`compiled` vs `compiled_runner` comparisons. Baseline rows establish the pre-change scalar
allocation gap only.

| workload | old compiled, candidate run | runner, candidate run | runner speedup |
| --- | ---: | ---: | ---: |
| scalar/n=8 | 142.38 ns | 34.076 ns | 4.18x |
| scalar/n=32 | 244.26 ns | 85.634 ns | 2.85x |
| scalar/n=128 | 671.69 ns | 297.46 ns | 2.26x |
| tensor64/n=8 | 1.9011 us | 1.7920 us | 1.06x |
| tensor64/n=32 | 7.1392 us | 7.0183 us | 1.02x |

Internal verdict: 5 wins / 0 losses / 0 neutral versus the previous compiled eval path.

## JAX Head-to-Head

JAX command used the existing comparison virtualenv:

```bash
/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python <inline add-chain harness>
```

JAX/JAXLIB: `jax 0.10.1`, `jax_enable_x64=true`, CPU backend. Each row used warmed `jax.jit`,
80 runs x 1000 inner loops.

| workload | runner | JAX p50 | Rust/JAX time | verdict |
| --- | ---: | ---: | ---: | --- |
| scalar/n=8 | 34.076 ns | 4857.883 ns | 0.0070 | Rust wins 142.6x |
| scalar/n=32 | 85.634 ns | 4653.0505 ns | 0.0184 | Rust wins 54.3x |
| scalar/n=128 | 297.46 ns | 4627.832 ns | 0.0643 | Rust wins 15.6x |
| tensor64/n=8 | 1.7920 us | 4.554228 us | 0.3935 | Rust wins 2.54x |
| tensor64/n=32 | 7.0183 us | 4.5211495 us | 1.5523 | JAX wins 1.55x |

Vs-JAX verdict: 4 wins / 1 loss / 0 neutral. The remaining loss is a real negative-evidence row:
JAX/XLA fuses the 32-step tensor chain while the dense interpreter still executes per-step dense
adds. Next target is dense elementwise-chain fusion / output reuse for tensor chains.

## Validation

- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-interpreters --all-targets`: pass.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-interpreters compiled_jaxpr_eval_matches_eager_eval_jaxpr --lib -- --nocapture`: pass.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a-local cargo test -p fj-conformance`: pass after repairing stale conformance expectations for cache lifecycle, complex cbrt JVP, and reviewed empty-list parser allowlisting.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo clippy -p fj-interpreters --lib --no-deps -- -D warnings`: pass.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a-local cargo clippy -p fj-conformance --tests --no-deps -- -D warnings`: pass.
- Full `cargo clippy -p fj-interpreters --all-targets -- -D warnings` remains blocked by pre-existing lints in `fj-trace` and `fj-lax`.
- Full-file rustfmt remains blocked by pre-existing drift in large interpreter files; scoped changed small files pass `rustfmt --edition 2024 --check`.

Generated conformance artifact churn was saved to `/tmp/frankenjax-cod-a-generated-artifacts.patch`
and reversed before commit.
