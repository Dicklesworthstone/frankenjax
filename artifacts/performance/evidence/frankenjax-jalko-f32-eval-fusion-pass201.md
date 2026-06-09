# frankenjax-jalko pass201: dense f32 eval_jaxpr fusion

Commit: `a08a761d`
Bead: `frankenjax-jalko`
Follow-up bead for remaining scope: `frankenjax-coax7`

## Target

Extend the shipped `a8nbp` chunked eval_jaxpr elementwise fusion from dense F64
to dense F32 same-shape `Add`/`Sub`/`Mul`/`Div`/`Neg` chains. This is the JAX
default float dtype and keeps the pass to one lever: no broadcast, half-float,
i64, max/min, or tolerance relaxation.

## Baseline And Benchmark

Pre-pass RCH control on `ovh-a` for the existing F64 fusion row:

```text
rch exec -- cargo bench -p fj-interpreters --bench eval_fusion_speed
EVAL_FUSION_SPEED n=1048576 ops=8 unfused=32.362ms fused=1.978ms speedup=16.37x
```

Final RCH same-invocation A/B on `vmi1227854`:

```text
EVAL_FUSION_SPEED_F32 n=1048576 ops=8 unfused=7.805ms fused=0.949ms speedup=8.22x
EVAL_FUSION_SPEED_F64 n=1048576 ops=8 unfused=25.893ms fused=2.296ms speedup=11.28x
```

Score: `12.5` (`Impact=5`, `Confidence=5`, `Effort=2`).

## Isomorphism

- Ordering preserved: the fused evaluator applies each primitive in equation
  order for each element.
- Tie-breaking unchanged: no comparison or tie surface exists in this pass.
- Floating point preserved: F32 operands are widened to F64, the primitive is
  applied, and the result is rounded back to F32 after every step, matching the
  existing fj-lax dense F32 arithmetic contract.
- RNG unchanged: no RNG surface.
- Fallback preserved: non-F32, non-dense, non-same-shape, broadcast, half/i64,
  max/min, params, effects, and sub-jaxpr cases stay on the normal interpreter path.

Golden output SHA:

```text
fef28624a52e5647abc35f0d388072b443cf081e5941243c6c58a8bd91f40a84
```

## Validation

```text
cargo fmt -p fj-interpreters --check
rch exec -- cargo check -p fj-interpreters --all-targets
rch exec -- cargo test -p fj-interpreters
rch exec -- cargo test -p fj-interpreters fusion_ -- --nocapture
```

Results: format passed, check passed, package tests passed (`147 passed`), and
focused fusion tests passed (`3 passed`). Standard clippy remains blocked by
pre-existing out-of-scope lints; package clippy passed with those known lint
classes allowed.
