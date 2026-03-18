# FrankenJAX

<div align="center">
  <img src="frankenjax_illustration.webp" alt="FrankenJAX - Clean-room Rust reimplementation of JAX transform semantics" width="600">

  **Clean-room Rust reimplementation of JAX's transform semantics.**

  Semantic fidelity. Mathematical rigor. Operational safety. Profile-proven performance.

  ![Rust](https://img.shields.io/badge/rust-nightly_2024-orange)
  ![Tests](https://img.shields.io/badge/tests-1724_passing-brightgreen)
  ![Primitives](https://img.shields.io/badge/primitives-110_ops-blue)
  ![AD Coverage](https://img.shields.io/badge/AD-110%2F110_VJP%2BJVP-brightgreen)
  ![Oracle Fixtures](https://img.shields.io/badge/oracle_fixtures-834_cases-purple)
</div>

---

## TL;DR

**The problem:** JAX's transform semantics (`jit`, `grad`, `vmap`) are deeply entangled with Python and XLA. There's no standalone, portable, verifiable implementation of the mathematical core.

**The solution:** FrankenJAX extracts and reimplements JAX's transform composition model in Rust with a canonical JAXPR-like IR, full automatic differentiation, and a differential conformance harness that validates every primitive against the real JAX oracle.

**Why FrankenJAX?**

| Feature | Status |
|---------|--------|
| 110 primitive operations with full eval | All green |
| Reverse-mode (VJP) + Forward-mode (JVP) AD for all 110 primitives | All green |
| Transform composition: `jit(grad(f))`, `vmap(grad(f))`, `grad(grad(f))` | All green |
| Linear algebra: Cholesky, QR, SVD, Eigh, TriangularSolve (eval + AD) | All green |
| FFT: Fft, Ifft, Rfft, Irfft (eval + AD) | All green |
| E-graph equality saturation optimizer (87 algebraic rewrite rules) | All green |
| 834 JAX oracle fixture cases for differential conformance | All green |
| RaptorQ erasure-coded durability for all long-lived artifacts | All green |
| Strict/Hardened compatibility-security mode split | All green |
| 1,724 `#[test]` cases + proptest suites | All passing |

## Comparison vs Alternatives

| | FrankenJAX | JAX (Google) | PyTorch | Enzyme (LLVM) | Autograd |
|---|---|---|---|---|---|
| **Language** | Rust | Python/C++ | Python/C++ | LLVM IR | Python |
| **Runtime dependency** | None (standalone) | Python + XLA + CUDA | Python + CUDA | LLVM toolchain | NumPy |
| **Transform composition** | Full (`jit`/`grad`/`vmap`) | Full | Limited (`torch.func`) | `grad` only | `grad` only |
| **Verifiable proofs** | Trace Transform Ledger | No | No | No | No |
| **Oracle conformance** | 834 JAX-verified cases | N/A (is the oracle) | No | No | No |
| **Artifact durability** | RaptorQ sidecars | No | No | No | No |
| **E-graph optimization** | 87 rules, equality saturation | XLA HLO passes | TorchScript/Inductor | LLVM passes | None |
| **Embeddable** | Yes (Rust library + C FFI) | No (Python required) | Partially (libtorch) | Yes (LLVM plugin) | No |

FrankenJAX is not trying to replace JAX for production ML training. It's a **reference implementation** of JAX's mathematical transform semantics that you can embed in Rust applications, use as a verification oracle, or study to understand how composable transforms actually work.

## Who Is This For?

- **Compiler researchers** studying how `jit`/`grad`/`vmap` compose and interact
- **Rust developers** who need automatic differentiation without Python
- **Verification engineers** who want auditable transform composition with proof artifacts
- **ML framework developers** who need a reference implementation of JAX semantics
- **Educators** teaching automatic differentiation, functional transforms, or IR design

## Quick Example

```rust
use fj_api::{jit, grad, vmap, jacobian, hessian, make_jaxpr};

// Trace a Rust closure into canonical IR
let jaxpr = make_jaxpr(|x| x * x + 3.0 * x)?;

// Apply transforms exactly like JAX
let result = jit(jaxpr.clone()).call(vec![Value::scalar_f64(5.0)])?;
let gradient = grad(jaxpr.clone()).call(vec![Value::scalar_f64(5.0)])?;

// Compose transforms
let batched_grad = jit(jaxpr.clone()).compose_grad();
let J = jacobian(jaxpr.clone()).call(vec![Value::scalar_f64(5.0)])?;
let H = hessian(jaxpr).call(vec![Value::scalar_f64(5.0)])?;
```

## Design Philosophy

**1. Transform composition semantics are non-negotiable.**
Every transform composition produces a verifiable proof artifact linking input IR, applied transforms, and output IR via the Trace Transform Ledger (TTL).

**2. Differential conformance, not reimplementation faith.**
Every primitive's behavior is validated against real JAX 0.9.2 oracle fixtures. We don't trust our implementations; we verify them with 834 oracle test cases spanning transforms, AD, linalg, FFT, RNG, dtype promotion, and transform composition.

**3. Strict/Hardened mode split.**
Strict mode maximizes observable compatibility. Hardened mode adds safety guards with bounded defensive recovery. You choose the tradeoff per invocation.

**4. RaptorQ-everywhere durability.**
Long-lived artifacts (fixtures, baselines, ledgers) get erasure-coded sidecars with scrub reports and decode proofs. Bit rot doesn't get to silently corrupt your conformance evidence.

**5. Correctness is measured, not assumed.**
Numerical AD rules are validated against finite-difference gradients. The Cholesky VJP/JVP bugs we found and fixed? Found by numerical verification tests, not by staring at formulas.

## Architecture

```
User API (fj-api)
    |
    v
Trace (fj-trace) ──> Canonical IR (fj-core: Jaxpr + 110 Primitives)
    |
    v
Transform Stack (fj-dispatch)
    |  jit    grad    vmap
    |    \      |      /
    v     v     v     v
    ├─ E-graph optimizer (fj-egraph: 87 rewrite rules)
    ├─ AD engine (fj-ad: VJP + JVP for all 110 primitives)
    ├─ Batch trace (fj-dispatch/batching: per-primitive vmap rules)
    └─ Evidence ledger (fj-ledger: transform composition proofs)
         |
         v
    Lowering + Eval (fj-lax: arithmetic, linalg, FFT, tensor ops)
         |
         v
    CPU Backend (fj-backend-cpu: dependency-wave parallel executor)
         |
         v
    Cache (fj-cache: SHA-256 deterministic keys, strict/hardened gates)
```

## Workspace Crates

| Crate | Purpose | Tests |
|-------|---------|-------|
| `fj-core` | Canonical IR (Jaxpr), 110 primitives, 11 dtypes, shapes, value model | Extensive |
| `fj-lax` | Primitive evaluation: arithmetic, linalg, FFT, reductions, tensor ops | 479 |
| `fj-ad` | Automatic differentiation: VJP + JVP for all 110 primitives | 179 |
| `fj-dispatch` | Transform dispatch, order-sensitive composition, batching | 55+ |
| `fj-trace` | `make_jaxpr` tracing from Rust closures, nested trace contexts | 50 |
| `fj-egraph` | E-graph equality saturation: 87 algebraic rewrite rules | 47 |
| `fj-api` | User-facing API: `jit`, `grad`, `vmap`, `jacobian`, `hessian` | 38 |
| `fj-cache` | Deterministic cache keys, strict/hardened gate behavior | Yes |
| `fj-ledger` | Decision/evidence ledger, loss-matrix actions, audit trail | Yes |
| `fj-runtime` | Tensor-aware runtime value model, optional async integration | Yes |
| `fj-interpreters` | Scoped primitive interpreter, partial evaluation, staging | Yes |
| `fj-conformance` | Differential conformance harness, 834 oracle fixtures, durability | 200+ |
| `fj-backend-cpu` | Dependency-wave parallel executor (rayon) | 40 |
| `fj-ffi` | C FFI surface (only crate permitted `unsafe`) | Yes |
| `fj-test-utils` | Shared test scaffolding, fixture helpers | Yes |

## Current Status

**80,872 lines of Rust** across 15 crates with end-to-end trace -> dispatch -> runtime pipeline:

- **110 primitive operations** covering arithmetic, trigonometric, hyperbolic, comparison, reduction, shape manipulation, linear algebra, FFT, bitwise, control flow, sorting, convolution, and special math functions
- **11 DTypes** (BF16, F16, F32, F64, I32, I64, U32, U64, Bool, Complex64, Complex128) with JAX-verified type promotion rules
- **Full AD coverage**: all 110 primitives have both VJP (reverse-mode) and JVP (forward-mode) rules, including multi-output decompositions (Cholesky, QR, SVD, Eigh) and FFT
- **Jacobian and Hessian** matrix computation via composable AD
- **`vmap`** with per-primitive batching rules, `in_axes`/`out_axes`, BatchTrace interpreter
- **E-graph optimizer**: 87 algebraic rewrite rules with equality saturation, verified to preserve program semantics
- **ThreeFry2x32 RNG**: key/split/fold_in/uniform/normal/bernoulli/categorical with JAX-matched determinism
- **Control flow**: `cond`, `scan`, `while_loop`, `fori_loop`, `switch` with AD support
- **834 JAX oracle fixture cases** captured from JAX 0.9.2 with x64 mode, covering transforms, AD, linalg, FFT, RNG, dtype promotion, and transform composition
- **1,724 `#[test]` cases** plus proptest/property-based suites
- **RaptorQ durability pipeline** for all long-lived evidence artifacts

## The Canonical IR: Jaxpr

At the center of FrankenJAX is the **Jaxpr** (JAX expression), a functional intermediate representation:

```rust
struct Jaxpr {
    invars: Vec<VarId>,          // Input variables (function parameters)
    constvars: Vec<VarId>,       // Constant bindings (captured values)
    outvars: Vec<VarId>,         // Output variables (return values)
    equations: Vec<Equation>,    // Sequence of primitive operations
}

struct Equation {
    primitive: Primitive,         // Which of the 110 operations
    inputs: SmallVec<[Atom; 4]>, // Input references (variables or literals)
    outputs: SmallVec<[VarId; 2]>, // Output variable bindings
    params: BTreeMap<String, String>,  // Operation parameters (axes, dtypes, etc.)
    effects: Vec<...>,           // Side-effect tokens
    sub_jaxprs: Vec<Jaxpr>,     // Nested Jaxprs (for control flow)
}

enum Atom {
    Var(VarId),    // Reference to a variable in the environment
    Lit(Literal),  // Inline constant value
}
```

A Jaxpr is a **straight-line program** — no branches, no loops at the top level. Control flow (`cond`, `scan`, `while_loop`, `switch`) is expressed via primitives that take sub-Jaxprs as arguments. This design makes transforms (grad, vmap) straightforward: they operate equation-by-equation, and each primitive has well-defined transform rules.

**Tracing**: The `make_jaxpr` function traces a Rust closure by running it with abstract tracer values that record operations instead of computing them. The result is a Jaxpr that represents the computation graph.

## DType Promotion

FrankenJAX implements JAX's type promotion lattice for mixed-type operations. When you add an `i64` and a `f64`, the result type follows this hierarchy:

```
Bool → I32 → I64 ─────────┐
                            ├──→ F64
U32 → U64 ────────────────┘
BF16 → F32 → F64
F16 → F32 → F64
Complex64 → Complex128
```

Key promotion rules verified against JAX oracle:
- Integer + Float → Float (e.g., `i64 + f64 → f64`)
- Unsigned + Signed → wider type or Float (e.g., `u64 + i64 → f64`)
- Any + Complex → Complex
- Same-type operations preserve type (e.g., `u32 + u32 → u32`)

The full 9x9 promotion matrix (162 cases for add and multiply) is captured from JAX 0.9.2 and validated in CI.

## Control Flow in the IR

FrankenJAX implements JAX's functional control flow primitives, which express branching and iteration within the Jaxpr IR:

| Primitive | Semantics | Sub-Jaxprs |
|-----------|-----------|------------|
| `cond` | `if pred then true_branch(args) else false_branch(args)` | 2 (true, false) |
| `scan` | Fold/scan with carry: `(carry, ys) = scan(body, init_carry, xs)` | 1 (body) |
| `while_loop` | `while cond(state): state = body(state)` | 2 (cond, body) |
| `fori_loop` | `for i in range(lo, hi): state = body(i, state)` | 1 (body) |
| `switch` | Multi-way branch: `branches[index](args)` | N (one per branch) |

These primitives compose with AD — `grad(scan(f))` differentiates through the scan by unrolling the tape. `vmap(cond(...))` batches the condition predicate and both branches.

## The Decision/Evidence Ledger

The `fj-ledger` crate implements a **decision-theoretic audit trail** for runtime choices:

Every non-trivial decision (cache hit vs recompute, strict vs hardened recovery, optimization level) is logged as a `DecisionRecord` with:
- **Action taken** and **alternatives considered**
- **Loss-matrix justification**: expected cost of each alternative under the current mode
- **Evidence signals**: what information was available at decision time
- **Conformal predictor**: statistical confidence bound on the decision quality

This is not a debugging log — it's a formal audit trail that answers "why did the system do X instead of Y?" for any execution. The ledger entries survive across sessions via the durability pipeline.

## Correctness Methodology

FrankenJAX uses **four layers of correctness assurance**, each catching different classes of bugs:

**Layer 1: Oracle conformance (834 cases)**
Capture expected outputs from real JAX, replay against FrankenJAX. Catches: wrong evaluation logic, incorrect primitive semantics, dtype mismatches.

**Layer 2: Numerical AD verification (9 tests)**
Compare analytical gradients (VJP/JVP rules) against finite-difference approximations. Catches: wrong derivative formulas, sign errors, missing terms. Found two real Cholesky bugs.

**Layer 3: Property-based testing (proptest)**
Generate random inputs and verify algebraic invariants: `exp(log(x)) ≈ x`, `Q^T Q ≈ I` for QR, `U Σ V^T ≈ A` for SVD. Catches: edge cases, numerical instability, overflow.

**Layer 4: E-graph semantics preservation (12 tests)**
Run programs with and without optimization, verify identical results. Catches: rewrite rules that change program meaning, cost model bugs that prefer wrong programs.

Combined, these layers provide defense-in-depth: oracle tests catch "wrong answer" bugs, numerical tests catch "wrong gradient" bugs, property tests catch "crashes on weird input" bugs, and e-graph tests catch "optimization broke something" bugs.

## How It Works: Deep Dive

### Automatic Differentiation Engine (fj-ad)

FrankenJAX uses **tape-based reverse-mode AD** (backpropagation) with full forward-mode support:

**Forward pass with tape recording:**
The AD engine evaluates the Jaxpr equation-by-equation, recording each operation as a `TapeEntry` that captures the primitive, input/output values, and parameters. Multi-output primitives (QR, SVD, Eigh) store all primal outputs explicitly — their VJP rules need them for the complex matrix calculus.

**Backward pass with cotangent threading:**
The tape is replayed in reverse. For each entry, the primitive's VJP rule computes input cotangents from output cotangents. When a variable appears in multiple equations, its gradients are accumulated (summed). An early-termination check skips entries where all output gradients are zero.

```
Forward:  x ──[mul]──> t1 ──[add]──> y     (record tape)
Backward: ḡ_y ──[add_vjp]──> ḡ_t1 ──[mul_vjp]──> ḡ_x   (replay reverse)
```

Each of the 110 primitives has hand-derived VJP and JVP rules. The linalg decomposition VJPs follow Murray 2016 ("Differentiation of the Cholesky decomposition") and related literature, with diagonal-halving corrections for the Phi operator and careful triangular-solve direction for L^{-T} vs L^{-1} terms.

### E-Graph Optimizer (fj-egraph)

The optimizer converts Jaxpr programs into an **e-graph** (equivalence graph) and applies **equality saturation** via the `egg` library. Instead of choosing one rewrite direction, all applicable rewrites fire simultaneously, and a cost function extracts the cheapest equivalent program.

87 algebraic rewrite rules, including:

| Category | Example Rule | Effect |
|----------|-------------|--------|
| Arithmetic identities | `x + 0 → x`, `x * 1 → x` | Eliminate no-ops |
| Strength reduction | `x + x → 2 * x` | Reduce operation count |
| Inverse pairs | `exp(log(x)) → x`, `neg(neg(x)) → x` | Cancel inverses |
| Trig identities | `sin²(x) + cos²(x) → 1` | Simplify trig expressions |
| Distributivity | `a*(b+c) ↔ a*b + a*c` | Factor or expand as needed |
| Complex field | `real(complex(r, i)) → r` | Simplify complex extraction |
| Comparison absorption | `max(a, min(a, b)) → a` | Eliminate nested comparisons |

The cost model (`OpCount`) counts the number of operations in each equivalent expression, ensuring the optimizer always extracts a program that's equal or smaller than the original.

### Transform Dispatch (fj-dispatch)

Transform composition is order-sensitive — `grad(vmap(f))` is not the same as `vmap(grad(f))`. The dispatcher processes a stack of transforms against a Jaxpr:

1. **Strip leading `jit`** (compile-time annotation, no-op in V1)
2. **Apply innermost transform first**: `grad` uses symbolic tape-based AD; `vmap` uses the BatchTrace interpreter with per-primitive batching rules
3. **Compose recursively**: for `grad(vmap(f))`, first vmap the Jaxpr, then grad the vectorized result
4. **Thread effect tokens** through execution to maintain side-effect ordering
5. **Record composition proof** in the Trace Transform Ledger for auditability

The dispatcher also supports **e-graph optimization** as a compile option — when `egraph_optimize=true`, the Jaxpr is optimized via equality saturation before evaluation.

### Linear Algebra Algorithms (fj-lax)

All linalg primitives are implemented in pure Rust with f64 arithmetic:

| Decomposition | Algorithm | Key Properties |
|--------------|-----------|----------------|
| **Cholesky** | Cholesky-Banachiewicz (row-by-row) | L where A = LL^T; requires SPD input |
| **QR** | Householder reflections | Q (orthogonal), R (upper triangular); sign-normalized diagonal |
| **SVD** | Jacobi rotations via A^T A eigendecomposition | U, S (descending), V^T; up to 100n^2 iterations |
| **Eigh** | Jacobi eigendecomposition | W (ascending eigenvalues), V (orthonormal eigenvectors) |
| **TriangularSolve** | Forward/back substitution | Exact for triangular systems; supports lower and upper |

The Cholesky VJP uses the formula bar_A = L^{-T} Phi(L^T bar_L) L^{-1} where Phi extracts the lower triangle with halved diagonal (Murray 2016). The JVP is dL = L Phi(L^{-1} dA L^{-T}). Both were numerically verified against finite-difference gradients.

### ThreeFry2x32 PRNG (fj-lax)

The RNG implements the ThreeFry2x32 counter-based PRNG from Salmon et al. (SC'11: "Parallel Random Numbers: As Easy as 1, 2, 3"):

- **Core cipher**: 20 rounds of rotation + XOR + key injection on 2-word (64-bit) state, using Skein rotation constants [13, 15, 26, 6, 17, 29, 16, 24]
- **Key splitting**: `split(key) = [threefry(key, [0,0]), threefry(key, [0,1])]` — produces two statistically independent child keys
- **Fold-in**: `fold_in(key, data) = threefry(key, [data, 0])` — mixes external data into the key
- **Sampling**: counter-based generation → uniform via division by 2^32 → normal via Box-Muller → bernoulli via threshold → categorical via Gumbel-max trick

The deterministic design means `random_key(42)` always produces the same sequence, matching JAX's ThreeFry implementation. This is verified against 25 JAX oracle fixtures covering key generation, splitting, fold-in, uniform, and normal distributions.

### Dependency-Wave Parallel Executor (fj-backend-cpu)

The CPU backend parallelizes Jaxpr execution via **dependency-wave scheduling**:

```
Wave 1:  a = f(x)    b = g(x)    ← parallel (both depend only on input)
Wave 2:  c = h(a, b)              ← sequential (depends on wave 1)
Wave 3:  d = k(c)                 ← sequential
```

The algorithm:
1. Find all equations whose inputs are available (the "ready wave")
2. Execute the wave in parallel via Rayon's thread pool
3. Store outputs in the environment
4. Repeat until all equations are executed

**Barrier detection**: equations with side effects, sub-Jaxprs, or multi-output primitives force sequential execution. This prevents reordering of effectful operations while maximizing parallelism for pure computations.

### Cache-Key Fingerprinting (fj-cache)

Every compilation/execution configuration gets a deterministic SHA-256 cache key:

```
fjx-<sha256hex>
  = SHA-256(mode|backend|transforms|compile_options|custom_hook|jaxpr_fingerprint)
```

The Jaxpr fingerprint recursively hashes the equation structure (primitives, arities, parameters, sub-Jaxprs). Transform ordering matters: `grad,vmap` and `vmap,grad` produce different keys. Compile options are sorted (BTreeMap) for deterministic ordering.

**Strict mode** rejects cache entries with unknown incompatible features. **Hardened mode** allows bounded recovery from unexpected cache states.

### RaptorQ Durability Sidecars (fj-conformance)

Long-lived artifacts (conformance fixtures, benchmark baselines, evidence ledgers) are protected against bit rot using RaptorQ erasure coding:

1. **Encode**: artifact → source symbols (256-byte chunks) + 10% repair symbols
2. **Sidecar**: JSON manifest with all symbols (Base64-encoded), SHA-256 hash, generation metadata
3. **Scrub**: decode from sidecar symbols, verify SHA-256 match against original artifact
4. **Decode proof**: intentionally drop N source symbols, verify recovery from remaining symbols + repair symbols

This means any fixture bundle can survive partial data loss and still be recoverable — critical for distributed CI environments where cache nodes can fail silently.

### Broadcasting Semantics

Binary operations in FrankenJAX support full **NumPy-style broadcasting** with four dispatch paths:

| LHS | RHS | Behavior |
|-----|-----|----------|
| Scalar | Scalar | Direct operation |
| Scalar | Tensor | Broadcast scalar across tensor shape |
| Tensor | Scalar | Broadcast scalar across tensor shape |
| Tensor | Tensor (same shape) | Elementwise operation |
| Tensor | Tensor (different shape) | Multi-dimensional broadcasting |

Multi-dimensional broadcasting follows NumPy rules: shapes are right-aligned, dimensions of size 1 are stretched, and incompatible dimensions cause errors. For example, a `[3, 1]` tensor can broadcast with a `[1, 4]` tensor to produce a `[3, 4]` result.

Complex number arithmetic is handled separately with proper (a+bi)(c+di) = (ac-bd) + (ad+bc)i multiplication and (a+bi)/(c+di) conjugate-denominator division.

### Security Model

FrankenJAX defends against several threat categories relevant to ML infrastructure:

**Cache confusion attacks**: Malicious or corrupted cache entries could cause a program to silently produce wrong results. The SHA-256 fingerprinting system binds cache keys to the exact computation (Jaxpr structure + transforms + mode + backend), and Strict mode rejects any unknown features.

**Transform-order vulnerabilities**: `grad(vmap(f))` and `vmap(grad(f))` produce different results. The Trace Transform Ledger records the exact transform composition order, preventing silent reordering. The dispatcher verifies composition compatibility before execution.

**Malformed graph attacks**: Adversarial or corrupted Jaxpr graphs could trigger panics or undefined behavior. All graph traversal validates arities, shapes, and dtypes at each equation. The fuzz testing infrastructure (`cargo fuzz`) continuously tests the IR deserializer and evaluator against malformed inputs.

**Silent data corruption**: Conformance fixtures and benchmark baselines could be corrupted on disk or in transit. RaptorQ sidecars provide erasure coding that detects and recovers from partial data loss, with decode proofs that verify recovery correctness.

## Oracle Conformance

FrankenJAX validates against real JAX output, not just hand-computed expected values:

```bash
# Set up JAX environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python jax jaxlib numpy

# Capture oracle fixtures from JAX (strict mode = real JAX, no fallback)
.venv/bin/python crates/fj-conformance/scripts/capture_legacy_fixtures.py \
  --legacy-root legacy_jax_code/jax \
  --output crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json \
  --rng-output crates/fj-conformance/fixtures/rng/rng_determinism.v1.json \
  --strict
```

| Fixture Family | Cases | Source |
|---------------|-------|--------|
| Transform (jit/grad/vmap/lax/control_flow/mixed_dtype) | 611 | JAX 0.9.2 |
| RNG determinism (key/split/fold_in/uniform/normal) | 25 | JAX 0.9.2 |
| Linear algebra + FFT oracle | 21 | JAX 0.9.2 (x64) |
| Transform composition (jit+grad, grad+grad, vmap+grad, jacobian, hessian) | 15 | JAX 0.9.2 (x64) |
| Dtype promotion (9x9 dtype matrix, add + mul) | 162 | JAX 0.9.2 (x64) |
| **Total** | **834** | |

## Verification Commands

```bash
# Format check
cargo fmt --check

# Compiler + lint
cargo check --all-targets
cargo clippy --all-targets -- -D warnings

# Full test suite
cargo test --workspace

# Conformance tests with output
cargo test -p fj-conformance -- --nocapture

# Benchmarks
cargo bench
```

## E2E Orchestration

```bash
# Run all E2E scenarios
./scripts/run_e2e.sh

# Run one scenario
./scripts/run_e2e.sh --scenario e2e_p2c001_full_dispatch_pipeline

# Each scenario emits forensic logs at artifacts/e2e/<scenario>.e2e.json
```

## Reliability Gates

```bash
# Full gates (coverage + flake + runtime + crash triage)
./scripts/enforce_quality_gates.sh

# Local iteration (fast)
./scripts/enforce_quality_gates.sh --skip-coverage --flake-runs 3

# Flake detector standalone
./scripts/detect_flakes.sh --runs 10
```

## Durability Pipeline

All long-lived artifacts get RaptorQ erasure-coded sidecars:

```bash
# Generate sidecar + scrub + decode proof (all-in-one)
cargo run -p fj-conformance --bin fj_durability -- \
  pipeline --artifact <path> --sidecar <sidecar_path> \
  --report <scrub_report_path> --proof <decode_proof_path>

# Batch process a directory
cargo run -p fj-conformance --bin fj_durability -- \
  batch --dir artifacts/e2e --output artifacts/durability --json
```

## Fuzzing

```bash
cd crates/fj-conformance/fuzz
cargo fuzz build
cargo fuzz run ir_deserializer corpus/seed/ir_deserializer
```

## Primitive Coverage

The 110 operations in the `Primitive` enum span the full breadth of JAX's LAX API:

| Category | Primitives | Count |
|----------|-----------|-------|
| **Arithmetic** | Add, Sub, Mul, Div, Neg, Abs, Rem, Pow, Max, Min, Exp, Log, Sqrt, Rsqrt, Floor, Ceil, Round, Expm1, Log1p, Sign, Square, Reciprocal, Logistic | 23 |
| **Trigonometric** | Sin, Cos, Tan, Asin, Acos, Atan, Atan2 | 7 |
| **Hyperbolic** | Sinh, Cosh, Tanh | 3 |
| **Special math** | Erf, Erfc, Cbrt, Lgamma, Digamma, ErfInv, IsFinite, IntegerPow, Nextafter | 9 |
| **Complex** | Complex, Conj, Real, Imag | 4 |
| **Comparison** | Eq, Ne, Lt, Le, Gt, Ge | 6 |
| **Reduction** | ReduceSum, ReduceMax, ReduceMin, ReduceProd, ReduceAnd, ReduceOr, ReduceXor, ReduceWindow | 8 |
| **Shape manipulation** | Reshape, Transpose, BroadcastInDim, Slice, DynamicSlice, DynamicUpdateSlice, Gather, Scatter, Concatenate, Pad, Rev, Squeeze, Split, ExpandDims | 14 |
| **Linear algebra** | Cholesky, QR, SVD, TriangularSolve, Eigh | 5 |
| **FFT** | Fft, Ifft, Rfft, Irfft | 4 |
| **Bitwise** | BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical, PopulationCount, CountLeadingZeros | 9 |
| **Control flow** | Cond, Scan, While, Switch | 4 |
| **Other** | Dot, Select, Clamp, Iota, BroadcastedIota, Copy, BitcastConvertType, ReducePrecision, OneHot, Cumsum, Cumprod, Sort, Argsort, Conv | 14 |

Every primitive has:
- Full evaluation in `fj-lax` (scalar and tensor paths)
- VJP rule (reverse-mode gradient) in `fj-ad`
- JVP rule (forward-mode tangent) in `fj-ad`
- NumPy-style broadcasting for binary operations

## Limitations

- **CPU-only backend.** GPU/TPU backends are not yet implemented. The CPU backend uses rayon for wave-parallel execution.
- **No F32 scalar literals.** All float scalars are stored as F64 internally. F32 tensor operations work via TensorValue, but scalar-level F32 promotion differs from JAX.
- **Bool arithmetic.** FrankenJAX does not treat Bool as numeric in arithmetic operations (JAX does: `True + True = 2`).
- **No XLA lowering.** FrankenJAX evaluates through its own interpreter, not through XLA. This means we match JAX's mathematical semantics but not its compilation/optimization pipeline.
- **Partial `vmap` + control flow composition.** `vmap(scan(...))` and similar compositions need further work.

## FAQ

**Q: Why not just use JAX directly?**
A: JAX requires Python + XLA + CUDA/TPU. FrankenJAX gives you the mathematical transform semantics in a standalone Rust library with no Python dependency.

**Q: How do you verify correctness without running JAX?**
A: We capture oracle fixtures from real JAX 0.9.2 (834 cases), then run our Rust implementation against those fixtures in CI. Every primitive, every transform, every dtype combination.

**Q: Is the AD (automatic differentiation) complete?**
A: Yes. All 110 primitives have both VJP (reverse-mode) and JVP (forward-mode) rules, including complex operations like Cholesky, QR, SVD, Eigh decompositions and FFT. Numerical verification tests confirm correctness via finite-difference comparison.

**Q: What's the Trace Transform Ledger?**
A: Every transform composition (`jit(grad(f))`, `vmap(grad(f))`, etc.) produces a verifiable proof artifact that records the input IR, applied transforms, and output IR. This is the crown-jewel innovation that makes transform composition auditable.

**Q: How fast is it?**
A: Performance optimization is ongoing. The CPU backend uses a dependency-wave parallel executor, and the e-graph optimizer applies 87 algebraic simplification rules. Profiling infrastructure is in place but systematic optimization hasn't started yet.

**Q: What's the difference between Strict and Hardened mode?**
A: Strict mode refuses to process anything it doesn't fully understand — unknown features, incompatible cache entries, or ambiguous inputs cause hard failures. Hardened mode allows bounded defensive recovery: it can handle some malformed inputs and degrade gracefully, but logs every recovery action in the decision ledger for audit.

**Q: How does the e-graph optimizer work?**
A: It converts your Jaxpr into an equivalence graph where every algebraically equivalent form exists simultaneously (e.g., `x+x` and `2*x` coexist as equivalent). The 87 rewrite rules fire until saturation (no new equivalences found), then a cost function extracts the cheapest program. This can discover simplifications that sequential rule application would miss.

**Q: Can I use FrankenJAX from Python/C?**
A: The `fj-ffi` crate provides a C FFI surface for calling FrankenJAX from any language with C interop. Python bindings are not yet implemented but would be straightforward via PyO3 or cffi on top of fj-ffi.

**Q: How are the linalg AD rules verified?**
A: Every linalg VJP and JVP rule is verified two ways: (1) numerical finite-difference comparison (perturb inputs, compare analytical vs numerical gradient), and (2) oracle comparison against JAX's output with x64 precision enabled. This caught two real bugs in the Cholesky decomposition AD during development — a missing diagonal-halving factor and a wrong triangular-solve direction.

**Q: What's RaptorQ and why is it in a math library?**
A: RaptorQ is a fountain code (erasure code) that can reconstruct data from any sufficient subset of encoded symbols. We use it to protect conformance fixtures and benchmark baselines against silent data corruption. In a distributed CI setup where cache nodes can lose data, this ensures your test evidence survives. It's admittedly unusual for a math library, but correctness evidence that can't be trusted is worthless.

**Q: How does the RNG match JAX?**
A: FrankenJAX implements the exact same ThreeFry2x32 cipher with the same rotation constants, key schedule, and sampling algorithms (Box-Muller for normal, Gumbel-max for categorical). The determinism is verified against 25 JAX oracle fixtures — `random_key(42)` produces bit-identical output to JAX's `jax.random.key(42)`.

## Key Documents

| Document | Purpose |
|----------|---------|
| `AGENTS.md` | AI agent development guidelines |
| `FEATURE_PARITY.md` | Feature-by-feature JAX parity status (all green) |
| `COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md` | Full V1 specification |
| `PLAN_TO_PORT_JAX_TO_RUST.md` | Original porting strategy |
| `EXISTING_JAX_STRUCTURE.md` | JAX architecture analysis |
| `PROPOSED_ARCHITECTURE.md` | FrankenJAX design decisions |

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.
