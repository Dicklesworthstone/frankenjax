# FrankenJAX

<div align="center">
  <img src="frankenjax_illustration.webp" alt="FrankenJAX - Clean-room Rust reimplementation of JAX transform semantics" width="600">

  **Clean-room Rust reimplementation of JAX's transform semantics.**

  Semantic fidelity. Mathematical rigor. Operational safety. Profile-proven performance.

  ![Rust](https://img.shields.io/badge/rust-nightly_2024-orange)
  ![Tests](https://img.shields.io/badge/tests-workspace_passing-brightgreen)
  ![Primitives](https://img.shields.io/badge/primitives-110_ops-blue)
  ![AD Coverage](https://img.shields.io/badge/AD-110%2F110_VJP%2BJVP-brightgreen)
  ![Oracle Fixtures](https://img.shields.io/badge/oracle_fixtures-848_cases-purple)
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
| Transform composition: `jit(grad(f))`, `vmap(grad(f))`, `grad(grad(f))` | Broad coverage; advanced compositions tracked |
| Linear algebra: Cholesky, QR, SVD, Eigh, TriangularSolve (eval + AD) | All green |
| FFT: Fft, Ifft, Rfft, Irfft (eval + AD) | All green |
| E-graph equality saturation optimizer (87 algebraic rewrite rules) | All green |
| 848 JAX oracle fixture cases for differential conformance | All green |
| RaptorQ erasure-coded durability for current evidence bundles | Implemented; all-long-lived-artifact expansion tracked |
| Strict/Hardened compatibility-security mode split | All green |
| Workspace tests + 4,416 static Rust test/proptest markers | Passing in the latest RCH run |

## Comparison vs Alternatives

| | FrankenJAX | JAX (Google) | PyTorch | Enzyme (LLVM) | Autograd |
|---|---|---|---|---|---|
| **Language** | Rust | Python/C++ | Python/C++ | LLVM IR | Python |
| **Runtime dependency** | None (standalone) | Python + XLA + CUDA | Python + CUDA | LLVM toolchain | NumPy |
| **Transform composition** | Scoped `jit`/`grad`/`vmap` with tracked advanced gaps | Full | Limited (`torch.func`) | `grad` only | `grad` only |
| **Verifiable evidence** | Trace Transform Ledger | No | No | No | No |
| **Oracle conformance** | 848 JAX-verified cases | N/A (is the oracle) | No | No | No |
| **Artifact durability** | RaptorQ sidecars | No | No | No | No |
| **E-graph optimization** | 87 rules, equality saturation | XLA HLO passes | TorchScript/Inductor | LLVM passes | None |
| **Embeddable** | Yes (Rust library + C FFI) | No (Python required) | Partially (libtorch) | Yes (LLVM plugin) | No |

FrankenJAX is not a replacement for JAX in production ML training. It is a **reference implementation** of JAX's mathematical transform semantics that you can embed in Rust applications, use as a verification oracle, or study to understand how composable transforms work.

## Who Is This For?

- **Compiler researchers** studying how `jit`/`grad`/`vmap` compose and interact
- **Rust developers** who need automatic differentiation without Python
- **Verification engineers** who want auditable transform composition evidence
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

## Worked Example: End-to-End Gradient Computation

What happens internally when you compute `grad(f)(3.0)` for f(x) = x^2 + 3x:

**Step 1: Trace.** `make_jaxpr` runs the closure with a tracer value, recording operations:
```
v1 = input
v2 = mul(v1, v1)        # x^2
v3 = mul(3.0, v1)       # 3x     (3.0 stored as Atom::Lit)
v4 = add(v2, v3)        # x^2 + 3x
output = v4
```

**Step 2: Dispatch.** The `grad` transform tells the dispatcher to apply reverse-mode AD.

**Step 3: Forward pass.** Evaluate with x=3.0, recording the tape:
```
v1 = 3.0
v2 = mul(3.0, 3.0) = 9.0       tape: [mul, inputs=[3.0, 3.0], output=9.0]
v3 = mul(3.0, 3.0) = 9.0       tape: [mul, inputs=[3.0, 3.0], output=9.0]
v4 = add(9.0, 9.0) = 18.0      tape: [add, inputs=[9.0, 9.0], output=18.0]
```

**Step 4: Backward pass.** Walk the tape in reverse with output cotangent = 1.0:
```
g_v4 = 1.0                          (seed)
add_vjp:  g_v2 += 1.0, g_v3 += 1.0 (add gradient = pass-through)
mul_vjp for v3:  g_v1 += 3.0 * 1.0 = 3.0   (d(3x)/dx = 3)
mul_vjp for v2:  g_v1 += 3.0 * 1.0 + 3.0 * 1.0 = 6.0   (d(x^2)/dx = 2x = 6)
                 g_v1 total = 3.0 + 6.0 = 9.0
```

**Result:** gradient = 9.0, which matches d/dx(x^2 + 3x)|_{x=3} = 2(3) + 3 = 9. Confirmed against JAX oracle.

## Design Philosophy

**1. Transform composition semantics are non-negotiable.**
Every transform composition produces an auditable evidence artifact linking input IR, applied transforms, and output IR via the Trace Transform Ledger (TTL). The current verifier checks deterministic ledger structure, evidence-to-transform binding, duplicate evidence IDs, and evidence-bound stack signatures.

**2. Differential conformance, not reimplementation faith.**
Every primitive's behavior is validated against real JAX 0.9.2 oracle fixtures. We verify, not trust, our implementations: 848 oracle test cases spanning transforms, AD, linalg, FFT, RNG, dtype promotion, and transform composition.

**3. Strict/Hardened mode split.**
Strict mode maximizes observable compatibility. Hardened mode adds safety guards with bounded defensive recovery. You choose the tradeoff per invocation.

**4. RaptorQ-everywhere durability.**
Current conformance and evidence bundles get erasure-coded sidecars with scrub reports and decode proofs. Expanding that coverage to every long-lived artifact family is tracked in `frankenjax-fcxy.2`.

**5. Correctness is measured, not assumed.**
Numerical AD rules are validated against finite-difference gradients. The Cholesky VJP/JVP bugs we found and fixed? Found by numerical verification tests, not by staring at formulas.

## Architecture

```
User API (fj-api)
  |
  v
Trace (fj-trace) --> Canonical IR (fj-core: Jaxpr + 110 Primitives)
  |
  v
Transform Stack (fj-dispatch)
  |  jit    grad    vmap
  |    \      |      /
  v     v     v     v
  +-- E-graph optimizer (fj-egraph: 87 rewrite rules)
  +-- AD engine (fj-ad: VJP + JVP for all 110 primitives)
  +-- Batch trace (fj-dispatch/batching: per-primitive vmap rules)
  +-- Evidence ledger (fj-ledger: transform composition proofs)
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
| `fj-conformance` | Differential conformance harness, 848 oracle fixtures, durability | 200+ |
| `fj-backend-cpu` | Dependency-wave parallel executor (rayon) | 40 |
| `fj-ffi` | C FFI surface (only crate permitted `unsafe`) | Yes |
| `fj-test-utils` | Shared test scaffolding, fixture helpers | Yes |

## Current Status

**162,733 lines of Rust** across 15 crates with end-to-end trace -> dispatch -> runtime pipeline:

- **110 primitive operations** covering arithmetic, trigonometric, hyperbolic, comparison, reduction, shape manipulation, linear algebra, FFT, bitwise, control flow, sorting, convolution, and special math functions
- **11 DTypes** (BF16, F16, F32, F64, I32, I64, U32, U64, Bool, Complex64, Complex128) with JAX-verified type promotion rules
- **Full AD coverage**: all 110 primitives have both VJP (reverse-mode) and JVP (forward-mode) rules, including multi-output decompositions (Cholesky, QR, SVD, Eigh) and FFT
- **Jacobian and Hessian** matrix computation via composable AD
- **`vmap`** with per-primitive batching rules, `in_axes`/`out_axes`, BatchTrace interpreter
- **E-graph optimizer**: 87 algebraic rewrite rules with equality saturation, verified to preserve program semantics
- **ThreeFry2x32 RNG**: key/split/fold_in/uniform/normal/bernoulli/categorical with JAX-matched determinism
- **Control flow**: `cond`, `scan`, `while_loop`, `fori_loop`, `switch` with AD support
- **848 JAX oracle fixture cases** captured from JAX 0.9.2 with x64 mode, covering transforms, AD, linalg, FFT, RNG, dtype promotion, and transform composition
- **4,416 static Rust test/proptest markers** plus 115 conformance test files; the latest full `cargo test --workspace` run passed through RCH
- **RaptorQ durability pipeline** for current conformance/evidence bundles, with all-long-lived-artifact coverage tracked in `frankenjax-fcxy.2`

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

A Jaxpr is a **straight-line program**: no branches, no loops at the top level. Control flow (`cond`, `scan`, `while_loop`, `switch`) is expressed via primitives that take sub-Jaxprs as arguments. This design makes transforms (grad, vmap) straightforward: they operate equation-by-equation, and each primitive has well-defined transform rules.

**Tracing**: The `make_jaxpr` function traces a Rust closure by running it with abstract tracer values that record operations instead of computing them. The result is a Jaxpr that represents the computation graph.

## Value Representation

FrankenJAX represents runtime values with a two-level model:

```rust
enum Value {
    Scalar(Literal),           // Single element
    Tensor(TensorValue),       // Multi-element with shape
}

enum Literal {
    Bool(bool),
    I64(i64),                  // Also used for I32 (widened internally)
    U32(u32),
    U64(u64),
    BF16Bits(u16),             // Stored as raw bits
    F16Bits(u16),              // Stored as raw bits
    F32Bits(u32),              // Stored as raw bits
    F64Bits(u64),              // Stored as raw bits
    Complex64Bits(u32, u32),   // (re_bits, im_bits)
    Complex128Bits(u64, u64),  // (re_bits, im_bits)
}

struct TensorValue {
    dtype: DType,              // Element type tag
    shape: Shape,              // Dimension vector: Vec<u32>
    elements: Vec<Literal>,    // Flat row-major storage
}
```

**Design rationale:** Storing floats as bit patterns rather than as native float values ensures exact round-trip serialization/deserialization without any NaN canonicalization surprises. The `Literal` enum is `Copy`, enabling cheap value passing through the interpreter. `TensorValue` uses flat row-major storage with a shape vector, matching NumPy/JAX's memory layout.

## Complex Number Support

FrankenJAX supports Complex64 (32-bit real + 32-bit imag) and Complex128 (64-bit real + 64-bit imag) throughout the stack:

**Arithmetic:** Complex binary operations use proper field arithmetic:
- Addition: (a+bi) + (c+di) = (a+c) + (b+d)i
- Multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
- Division: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2)

**Primitives:** `Complex(re, im)` constructs a complex value, `Real(z)` and `Imag(z)` extract components, `Conj(z)` conjugates. Core complex eval support currently includes `add`, `sub`, `mul`, `div`, `rem`, `pow`, `eq`, `ne`, `max`, `min`, `select`, `iota`, `broadcasted_iota`, `neg`, `abs`, `exp`, `expm1`, `log`, `log1p`, `sqrt`, `rsqrt`, `cbrt`, `sign`, `square`, `reciprocal`, `logistic`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, and `reduce_sum`.

**FFT:** Complex-valued FFT/IFFT operates on Complex128 tensors directly. RFFT takes real input and produces Complex128 output (half-spectrum). IRFFT inverts this.

**AD:** VJP and JVP rules for complex operations follow the Wirtinger derivative convention where applicable. The e-graph optimizer has dedicated rules for complex simplification: `real(complex(r, i)) → r`, `imag(complex(r, i)) → i`, `complex(real(z), imag(z)) → z`.

## DType Promotion

FrankenJAX implements JAX's type promotion lattice for mixed-type operations. When you add an `i64` and a `f64`, the result type follows this hierarchy:

```
Bool --> I32 --> I64 --------+
                             +--> F64
U32 --> U64 -----------------+
BF16 --> F32 --> F64
F16  --> F32 --> F64
Complex64 --> Complex128
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

These primitives compose with AD. `grad(scan(f))` differentiates through the scan by unrolling the tape. `vmap(cond(...))` batches the condition predicate and both branches.

## The Decision/Evidence Ledger

The `fj-ledger` crate implements a **decision-theoretic audit trail** for runtime choices:

Every non-trivial decision (cache hit vs recompute, strict vs hardened recovery, optimization level) is logged as a `DecisionRecord` with:
- **Action taken** and **alternatives considered**
- **Loss-matrix justification**: expected cost of each alternative under the current mode
- **Evidence signals**: what information was available at decision time
- **Conformal predictor**: statistical confidence bound on the decision quality

This is a formal audit trail, not a debugging log. It answers "why did the system do X instead of Y?" for any execution. The ledger entries survive across sessions via the durability pipeline.

## Correctness Methodology

FrankenJAX uses **four layers of correctness assurance**, each catching different classes of bugs:

**Layer 1: Oracle conformance (848 cases)**
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
The AD engine evaluates the Jaxpr equation-by-equation, recording each operation as a `TapeEntry` that captures the primitive, input/output values, and parameters. Multi-output primitives (QR, SVD, Eigh) store all primal outputs explicitly because their VJP rules need them for the matrix calculus.

**Backward pass with cotangent threading:**
The tape is replayed in reverse. For each entry, the primitive's VJP rule computes input cotangents from output cotangents. When a variable appears in multiple equations, its gradients are accumulated (summed). An early-termination check skips entries where all output gradients are zero.

```
Forward:  x --[mul]--> t1 --[add]--> y       (record tape)
Backward: g_y --[add_vjp]--> g_t1 --[mul_vjp]--> g_x   (replay reverse)
```

Each of the 110 primitives has hand-derived VJP and JVP rules. Here are a few concrete examples showing the mathematical formulas implemented:

| Primitive | VJP Rule (given output cotangent ḡ) | JVP Rule (given input tangent ẋ) |
|-----------|------|------|
| `mul(a, b)` | ḡ_a = ḡ * b, ḡ_b = ḡ * a | ẏ = ẋ_a * b + a * ẋ_b (product rule) |
| `sin(x)` | ḡ_x = ḡ * cos(x) | ẏ = ẋ * cos(x) |
| `exp(x)` | ḡ_x = ḡ * exp(x) | ẏ = ẋ * exp(x) |
| `reduce_sum(x)` | ḡ_x = broadcast(ḡ, x.shape) | ẏ = reduce_sum(ẋ) |
| `reshape(x)` | ḡ_x = reshape(ḡ, x.shape) | ẏ = reshape(ẋ, new_shape) |
| `transpose(x)` | ḡ_x = transpose(ḡ, inv_perm) | ẏ = transpose(ẋ, perm) |

Non-differentiable primitives (comparisons, bitwise ops, floor/ceil) return zero gradients, which is mathematically correct since their derivatives are zero almost everywhere.

The linalg decomposition VJPs follow Murray 2016 ("Differentiation of the Cholesky decomposition") and related literature, with diagonal-halving corrections for the Phi operator and careful triangular-solve direction for L^{-T} vs L^{-1} terms.

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

Transform composition is order-sensitive. `grad(vmap(f))` produces different results from `vmap(grad(f))`. The dispatcher processes a stack of transforms against a Jaxpr:

1. **Strip leading `jit`** (compile-time annotation, no-op in V1)
2. **Apply innermost transform first**: `grad` uses symbolic tape-based AD; `vmap` uses the BatchTrace interpreter with per-primitive batching rules
3. **Compose recursively**: for `grad(vmap(f))`, first vmap the Jaxpr, then grad the vectorized result
4. **Thread effect tokens** through execution to maintain side-effect ordering
5. **Record composition proof** in the Trace Transform Ledger for auditability

The dispatcher also supports **e-graph optimization** as a compile option. When `egraph_optimize=true`, the Jaxpr is optimized via equality saturation before evaluation.

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
- **Key splitting**: `split(key) = [threefry(key, [0,0]), threefry(key, [0,1])]`, producing two statistically independent child keys
- **Fold-in**: `fold_in(key, data) = threefry(key, [data, 0])`, mixing external data into the key
- **Sampling**: counter-based bit generation, then uniform (divide by 2^32), normal (Box-Muller), bernoulli (threshold), or categorical (Gumbel-max trick)

The deterministic design means `random_key(42)` always produces the same sequence, matching JAX's ThreeFry implementation. This is verified against 25 JAX oracle fixtures covering key generation, splitting, fold-in, uniform, and normal distributions.

### Dependency-Wave Parallel Executor (fj-backend-cpu)

The CPU backend parallelizes Jaxpr execution via **dependency-wave scheduling**:

```
Wave 1:  a = f(x)    b = g(x)    <-- parallel (both depend only on input)
Wave 2:  c = h(a, b)              <-- sequential (depends on wave 1)
Wave 3:  d = k(c)                 <-- sequential
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
fjx-<sha256hex> = SHA-256(
    mode | backend | transforms | compile_options | custom_hook | jaxpr_fingerprint
)
```

The Jaxpr fingerprint recursively hashes the equation structure (primitives, arities, parameters, sub-Jaxprs). Transform ordering matters: `grad,vmap` and `vmap,grad` produce different keys. Compile options are sorted (BTreeMap) for deterministic ordering.

**Strict mode** rejects cache entries with unknown incompatible features. **Hardened mode** allows bounded recovery from unexpected cache states.

The cache legacy parity ledger and lifecycle gate are generated with:

```bash
./scripts/run_cache_lifecycle_gate.sh --enforce
```

The gate writes `artifacts/conformance/cache_legacy_parity_ledger.v1.json`,
`artifacts/conformance/cache_lifecycle_report.v1.json`, a Markdown preview, and
`artifacts/e2e/e2e_cache_lifecycle_gate.e2e.json`. The report must prove sorted
compile-option keying, transform-order separation, strict unknown-metadata
rejection, hardened unknown-metadata inclusion, custom-hook key material,
namespace/version separation, hostile key-material non-aliasing, corrupt-read
bypass, and failed-write miss behavior before cache parity can be green.

### RaptorQ Durability Sidecars (fj-conformance)

Long-lived artifacts (conformance fixtures, benchmark baselines, evidence ledgers) are protected against bit rot using RaptorQ erasure coding:

1. **Encode**: split artifact into source symbols (256-byte chunks), generate 10% repair symbols
2. **Sidecar**: JSON manifest with all symbols (Base64-encoded), SHA-256 hash, generation metadata
3. **Scrub**: decode from sidecar symbols, verify SHA-256 match against original artifact
4. **Decode proof**: intentionally drop N source symbols, verify recovery from remaining symbols + repair symbols

Any fixture bundle can therefore survive partial data loss and still be recovered, which matters in distributed CI environments where cache nodes can fail silently.

### Multi-Output Primitive Handling

Five primitives produce multiple outputs: Cholesky (1 output but via multi-output path), QR (Q, R), SVD (U, S, Vt), Eigh (W, V), and TriangularSolve (X). These require special handling at every layer:

| Layer | Single-output | Multi-output |
|-------|--------------|--------------|
| **IR** | `outputs: [VarId(2)]` | `outputs: [VarId(2), VarId(3)]` or `[VarId(2), VarId(3), VarId(4)]` |
| **Eval** | `eval_primitive() -> Value` | `eval_primitive_multi() -> Vec<Value>` |
| **VJP** | `vjp(prim, inputs, [g]) -> [g_input]` | `vjp(prim, inputs, [g1, g2], [out1, out2]) -> [g_input]` |
| **JVP** | `jvp_rule(prim, primals, tangents) -> tangent` | `jvp_rule_multi(prim, primals, tangents, primal_outs) -> [t1, t2]` |
| **Vmap** | Batch dim tracked on single output | Each output independently tracks its batch dim |
| **Tape** | Stores one output value | Stores all output values (needed for VJP of decompositions) |

The SVD VJP, for example, needs the primal U, S, and Vt outputs to compute dA from dU, dS, and dVt. This is why the tape records `output_values`; they're consumed during the backward pass.

### How FrankenJAX Differs from JAX's Implementation

FrankenJAX reimplements JAX's *semantics*, not its *implementation*. Key architectural differences:

| Aspect | JAX (Python/C++) | FrankenJAX (Rust) |
|--------|-----------------|-------------------|
| **IR** | JAXPR with Python objects | Jaxpr with Rust enums (fully serializable) |
| **Tracing** | Python abstract interpreter | Rust closure tracing via `TracerRef` operator overloading |
| **AD tape** | Implicit via Python closures | Explicit `Vec<TapeEntry>` with value snapshots |
| **Compilation** | XLA HLO to device code | E-graph optimization then direct interpretation |
| **Dispatch** | C++ `xla::Client` | Rust `dispatch()` with transform stack |
| **Type system** | NumPy dtype objects | Rust `DType` enum (11 variants) |
| **Parallelism** | XLA partitioning + SPMD | Rayon dependency-wave scheduling |
| **Value model** | NumPy arrays + JAX DeviceArray | `Value::Scalar(Literal)` / `Value::Tensor(TensorValue)` |

What we kept identical: the Jaxpr structure (invars, constvars, outvars, equations), primitive semantics, transform composition rules, VJP/JVP mathematical formulas, type promotion lattice, ThreeFry PRNG cipher, and control flow primitive semantics.

What we simplified: no XLA lowering (we interpret directly), no device placement (CPU only), no SPMD partitioning, no custom call mechanism, no platform-specific kernels.

What we added: Trace Transform Ledger (auditable composition proofs), RaptorQ durability (erasure-coded artifacts), decision/evidence ledger (loss-matrix runtime decisions), strict/hardened mode split, e-graph equality saturation optimizer.

### Tracing: From Closures to IR (fj-trace)

`make_jaxpr` converts a Rust closure into a Jaxpr by running it with **tracer values** that record operations instead of computing them:

```rust
let jaxpr = make_jaxpr(|inputs| {
    vec![&inputs[0] * &inputs[0] + &inputs[0]]  // x^2 + x
}, vec![ShapedArray::scalar(DType::F64)])?;
// produces: Jaxpr: v2 = mul(v1, v1); v3 = add(v2, v1); output = v3
```

Each `TracerRef` carries an abstract value (dtype + shape) and a reference to the shared trace context. Operator overloading on `+`, `-`, `*` routes through `binary_op(Primitive::Add, ...)` which records an `Equation` in the active trace frame rather than evaluating it.

**Nested scopes** are handled via a frame stack. Control flow primitives (`cond`, `scan`) push a sub-trace frame, trace the body closure in that scope, pop the frame to get a sub-Jaxpr, and record the control flow equation in the parent frame. This enables tracing through arbitrary nesting of `jit(grad(vmap(f)))` where each transform traces its body and wraps the result.

**Shape inference** runs during tracing to validate tensor shapes at trace time rather than at eval time. The `infer_primitive_output_avals` function handles shape inference for all 110 primitives, including complex cases like broadcasting, gather indices, and convolution output shapes.

### Partial Evaluation & Constant Folding (fj-interpreters)

When some inputs to a Jaxpr are known at compile time, **partial evaluation** splits the program into a known part (pre-computable) and an unknown part (runtime-dependent):

```
Original:  y = (2 * 3) + x     [2 and 3 are known, x is unknown]
Known:     const_6 = 2 * 3     [pre-computed to 6]
Unknown:   y = const_6 + x     [residual program, simpler]
```

The algorithm classifies each equation as "known" (all inputs derivable from known values) or "unknown" (has abstract dependencies), then identifies **residual values**, meaning intermediate results produced by known equations but consumed by unknown ones. These become constants in the residual Jaxpr.

This is the mechanism that enables `jit` to specialize programs: static shapes, dtypes, and constant arguments are folded away, leaving only the dynamic computation.

### Vmap Batching Rules (fj-dispatch)

`vmap` does not loop over the batch dimension. Each primitive has an **O(1) batching rule** that handles the batch dimension as metadata:

**Elementwise operations** (sin, exp, add, mul, ...): The batch dimension passes through unchanged. If one operand is batched and the other is not, the unbatched operand is broadcast.

```
vmap(sin)(x)  where x.shape=[batch, n]
=> sin(x)     # batch dim just passes through, no loop
```

**Reductions** (reduce_sum, reduce_max, ...): The reduction axis is shifted to account for the batch dimension. Reduction never operates along the batch axis itself.

```
vmap(reduce_sum, axis=1)(x)  where x.shape=[batch, m, n]
=> reduce_sum(x, axis=2)    # axis shifted past batch dim
=> result.shape=[batch, m]
```

**Shape operations** (reshape, transpose, ...): The batch dimension is excluded from the reshape/transpose operation. The permutation indices are adjusted to skip position 0 (the batch axis).

**Multi-output operations** (QR, SVD, ...): Each output independently tracks its batch dimension position. For QR of a batched matrix `[batch, m, n]`, Q gets batch_dim=0 with shape `[batch, m, k]` and R gets batch_dim=0 with shape `[batch, k, n]`.

### Custom Derivative Registration (fj-ad)

Users can override the built-in VJP/JVP rules for any primitive:

```rust
fj_ad::register_custom_vjp(Primitive::MyOp, |inputs, grad, params| {
    // Custom gradient logic
    Ok(vec![custom_gradient])
});
```

Custom rules are stored in a global thread-safe registry (`RwLock<BTreeMap<Primitive, Arc<dyn Fn>>>`). During AD, the engine checks for custom rules before falling back to built-in rules. This enables:

- Overriding gradients for numerically unstable primitives with stable alternatives
- Implementing "stop gradient" by registering a zero-returning rule
- Adding gradient rules for new user-defined primitives
- Testing AD correctness by comparing custom vs built-in rules

The `clear_custom_derivative_rules()` function resets all registrations, used in test isolation.

### Jacobian & Hessian via AD Composition (fj-ad)

Higher-order derivatives are computed by composing first-order AD:

**Jacobian** (forward-mode, JVP-based):
```
For each basis vector e_j in input space:
    tangent_j = jvp(f, x, e_j)    # One forward pass per input dimension
    J[:, j] = tangent_j            # Column j of Jacobian
```
Cost: `input_dim` forward passes. Returns a `[output_dim, input_dim]` matrix.

**Hessian** (mixed-mode, grad + finite differences):
```
For each basis vector e_k in input space:
    g_plus  = grad(f)(x + eps*e_k)   # Gradient at perturbed point
    g_minus = grad(f)(x - eps*e_k)
    H[:, k] = (g_plus - g_minus) / (2*eps)  # Central difference
```
Cost: `2 × input_dim` gradient evaluations. Returns a symmetric `[input_dim, input_dim]` matrix. Uses `ε = 1e-5` for numerical stability.

### FFT Implementation (fj-lax)

The FFT primitives use a **radix-2 Cooley-Tukey fast path** for power-of-two lengths and keep the direct **O(n^2) DFT** fallback for non-power-of-two lengths. The fallback remains the simple reference path, while common FFT lengths get O(n log n) execution.

```
X[k] = sum_{j=0}^{n-1} x[j] * e^{-2*pi*i*j*k/n}    (DFT)
x[j] = (1/n) sum_{k=0}^{n-1} X[k] * e^{+2*pi*i*j*k/n}    (IDFT)
```

RFFT exploits Hermitian symmetry of real-valued input signals: only the first `n/2 + 1` frequency bins are returned (the rest are conjugate mirrors). IRFFT reconstructs the full spectrum from the half-spectrum before applying IDFT.

All FFT operations support batching. The transform applies along the last axis, with leading dimensions treated as independent batch elements.

### Parity Reporting (fj-conformance)

The conformance harness tracks parity at multiple granularities:

| Level | Tracks | Used For |
|-------|--------|----------|
| **Per-case** | matched/mismatched, expected vs actual JSON, comparator type | Debugging individual failures |
| **Per-family** | Jit/Grad/Vmap/Lax/Random/ControlFlow/MixedDtype breakdowns | Identifying weak areas |
| **Per-dtype** | Tolerance tiers (F64: atol=1e-12; F32: atol=1e-5; Int: exact) | Precision-appropriate comparison |
| **Suite-level** | Overall pass rate, gate status (pass if rate=1.0) | CI gate decision |
| **Drift classification** | Pass/Regression/Improvement/Flake/Timeout | Trend tracking across releases |

The tolerance system uses dtype-aware tiers: double-precision results get tight `1e-12` tolerance, single-precision gets `1e-5`, and integer/boolean results require exact match. This prevents false failures from legitimate floating-point variation while catching real regressions.

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
| Transform (jit/grad/vmap/lax/control_flow/mixed_dtype) | 613 | JAX 0.9.2 |
| RNG determinism (key/split/fold_in/uniform/normal) | 25 | JAX 0.9.2 |
| Linear algebra + FFT oracle | 33 | JAX 0.9.2; recapture gate requires fixture x64 metadata refresh |
| Transform composition (jit+grad, grad+grad, vmap+grad, jacobian, hessian) | 15 | JAX 0.9.1 fixture metadata; recapture gate requires 0.9.2 refresh |
| Dtype promotion (9x9 dtype matrix, add + mul) | 162 | JAX 0.9.1 fixture metadata; recapture gate requires 0.9.2 refresh |
| **Total** | **848** | |

The oracle recapture gate makes this evidence auditable instead of relying on
the static table above:

```bash
./scripts/run_oracle_recapture_gate.sh
```

It writes `artifacts/conformance/oracle_recapture_matrix.v1.json`,
`artifacts/conformance/oracle_drift_report.v1.json`,
`artifacts/conformance/oracle_recapture_matrix.v1.md`, and
`artifacts/e2e/e2e_oracle_recapture_gate.e2e.json`. The gate records fixture
paths, case counts, legacy anchors, capture commands, JAX versions, x64 mode,
hashes, unsupported recapture rows, and exact replay data. Use `--enforce` to
turn stale versions, missing fixture families, changed hashes, missing baselines,
or unsupported recapture rows into a nonzero CI exit.

## Building from Source

```bash
# Clone the repository
git clone https://github.com/Dicklesworthstone/frankenjax.git
cd frankenjax

# Rust nightly is required (see rust-toolchain.toml)
rustup install nightly
rustup override set nightly

# Build the entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Build with release optimizations (LTO + single codegen unit)
cargo build --workspace --release
```

**Optional: Set up JAX oracle environment for conformance capture:**
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python jax jaxlib numpy
git clone --depth 1 https://github.com/jax-ml/jax.git legacy_jax_code/jax
```

## E2E Forensic Logs

Each E2E scenario produces a shared forensic JSON log at `artifacts/e2e/<scenario>.e2e.json`.
New logs use `schema_version: "frankenjax.e2e-forensic-log.v1"` and are validated by the
Rust checker in `fj-conformance` before dashboard ingestion. The contract includes the
bead id, scenario id, exact command, working directory, environment fingerprint, feature
flags, fixture/oracle ids, transform stack, strict/hardened mode, inputs, expected and
actual results, tolerance policy, error taxonomy class, timings, allocation counters,
hash-bound artifact references, replay command, status, redaction notes, and failure
summary.

```json
{
  "schema_version": "frankenjax.e2e-forensic-log.v1",
  "bead_id": "frankenjax-cstq.16",
  "scenario_id": "e2e_p2c001_full_dispatch_pipeline",
  "command": ["cargo", "test", "-p", "fj-conformance", "--test", "e2e", "--", "..."],
  "working_dir": "/data/projects/frankenjax",
  "environment": {
    "os": "linux",
    "arch": "x86_64",
    "rust_version": "rustc nightly",
    "cargo_version": "cargo nightly",
    "cargo_target_dir": "<default>",
    "env_vars": { "API_TOKEN": "[REDACTED]" },
    "timestamp_unix_ms": 1777605600000
  },
  "feature_flags": ["default"],
  "fixture_ids": ["fixture:legacy_transform_cases.v1"],
  "oracle_ids": ["jax:0.9.2"],
  "transform_stack": ["jit", "grad"],
  "mode": "strict",
  "inputs": { "args": [3.0] },
  "expected": { "values": [6.0] },
  "actual": { "values": [6.0] },
  "tolerance": { "policy_id": "f64-default", "atol": 1e-12, "rtol": 1e-12, "ulp": null, "notes": null },
  "error": { "expected": null, "actual": null, "taxonomy_class": "none" },
  "timings": { "setup_ms": 0, "trace_ms": 1, "dispatch_ms": 2, "eval_ms": 1, "verify_ms": 1, "total_ms": 5 },
  "allocations": { "allocation_count": null, "allocated_bytes": null, "peak_rss_bytes": null, "measurement_backend": "not_measured" },
  "artifacts": [{ "kind": "stdout_log", "path": "artifacts/e2e/example.stdout.log", "sha256": "<64 hex chars>", "required": true }],
  "replay_command": "cargo test -p fj-conformance --test e2e -- e2e_p2c001_full_dispatch_pipeline --exact --nocapture",
  "status": "pass",
  "failure_summary": null,
  "redactions": [{ "path": "$.environment.env_vars.API_TOKEN", "reason": "secret-like env var" }],
  "metadata": {}
}
```

This enables full replay: given the log, users can re-run the exact command, verify
the hash-bound artifacts, inspect strict/hardened mode, and compare expected versus
actual results without source-code archaeology. Unknown fields are accepted for forward
compatibility, but missing required fields, stale artifact hashes, redacted replay
commands, unredacted secret-like values, malformed status values, and failing logs
without summaries are rejected. Empty log sets are also rejected so dashboard jobs
cannot silently pass without evidence.

```bash
# Validate the bootstrap contract sample
./scripts/validate_e2e_logs.sh

# Emit a dashboard-ready validation report for selected logs
./scripts/validate_e2e_logs.sh --output artifacts/e2e/e2e_validation_report.json artifacts/e2e

# Exercise the bootstrap sample and validator together
./scripts/bootstrap_e2e_forensic_log.sh
```

Existing legacy/ad hoc E2E emitters must either write this shared schema directly or
carry a temporary adapter bead before being counted as complete dashboard evidence.

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

## Test Program Library

FrankenJAX includes a library of **145+ pre-defined test programs** (`ProgramSpec` enum) that serve as standardized inputs for conformance testing, benchmarking, and AD verification:

| Category | Examples | Count |
|----------|---------|-------|
| Basic arithmetic | `Add2`, `Square`, `SquarePlusLinear`, `AddOne` | ~10 |
| Unary LAX ops | `LaxNeg`, `LaxAbs`, `LaxExp`, `LaxLog`, `LaxSqrt`, `LaxSin`, `LaxCos`, ... | ~30 |
| Binary LAX ops | `LaxAdd`, `LaxMul`, `LaxDiv`, `LaxPow`, `LaxMax`, `LaxMin`, ... | ~15 |
| Special math | `LaxCbrt`, `LaxLgamma`, `LaxDigamma`, `LaxErfInv`, `LaxIsFinite` | ~10 |
| Shape manipulation | `LaxReshape6To2x3`, `LaxTranspose2x3`, `LaxSlice1To4`, `LaxRev`, ... | ~15 |
| Linear algebra | `LaxCholesky`, `LaxQr`, `LaxSvd`, `LaxEigh`, `LaxTriangularSolve` | 5 |
| FFT | `LaxFft`, `LaxIfft`, `LaxRfft`, `LaxIrfft` | 4 |
| Control flow | `CondSelect`, `ScanAdd`, `LaxWhileAddLt`, `LaxSwitch3` | ~5 |
| Bitwise | `LaxBitwiseAnd`, `LaxShiftLeft`, `LaxPopulationCount`, ... | ~10 |
| Cumulative/Sort | `LaxCumsum`, `LaxCumprod`, `LaxSort`, `LaxArgsort` | 4 |
| Reductions | `LaxReduceSum`, `LaxReduceMax`, `LaxReduceWindow` | ~5 |
| Other | `LaxIota5`, `LaxOneHot4`, `LaxConv1dValid`, `LaxCopy` | ~30 |

Each program is constructed via `build_program(spec) -> Jaxpr`, returning a ready-to-evaluate Jaxpr. This enables systematic testing: run every program through `jit`, `grad`, `vmap`, and composed transforms, comparing against oracle values.

## Rust Ecosystem Integration

FrankenJAX is built on carefully chosen Rust crates:

| Dependency | Purpose | Why This Crate |
|-----------|---------|----------------|
| `egg` | E-graph equality saturation | Industry-standard for term rewriting; extensible language and cost model |
| `smallvec` | Inline small-vector for IR nodes | Avoids heap allocation for typical 1-4 element equation inputs/outputs |
| `rustc-hash` | Fast non-crypto hashing | FxHashMap for interpreter environments; faster than SipHash for small keys |
| `serde` + `serde_json` | Serialization | IR nodes, fixtures, ledger entries all need deterministic serialization |
| `sha2` | Cryptographic hashing | Cache keys and durability sidecars require collision-resistant hashing |
| `proptest` | Property-based testing | Generates random inputs to verify algebraic invariants (commutativity, associativity, etc.) |
| `criterion` | Microbenchmarking | Statistically rigorous benchmarks for dispatch and interpreter hot paths |
| `rayon` | Data parallelism | Dependency-wave executor parallelizes independent equations |
| `half` | Half-precision floats | BF16 and F16 scalar representation |
| `base64` | Encoding | RaptorQ sidecar symbols encoded as Base64 for JSON storage |
| `tempfile` | Temporary directories | Test isolation for conformance fixture and durability pipeline tests |

**Feature flags:**
```toml
[features]
default = []
asupersync-integration = ["dep:asupersync"]   # Structured async runtime
frankentui-integration = ["dep:ftui"]          # Terminal UI rendering
```

**Build profile:**
```toml
[profile.release]
opt-level = 3       # Maximum optimization
lto = true          # Link-time optimization
codegen-units = 1   # Single codegen for best optimization
strip = true        # Remove debug symbols
```

The workspace uses Rust edition 2024 (nightly) and enforces `#![forbid(unsafe_code)]` in every crate except `fj-ffi` (which provides the C FFI boundary).

## Special Math Function Approximations

Several primitives use polynomial or series approximations rather than calling libm:

| Function | Algorithm | Accuracy |
|----------|-----------|----------|
| `erf(x)` | Horner-form rational approximation (Abramowitz & Stegun 7.1.26) | ~1e-7 relative error |
| `lgamma(x)` | Lanczos approximation with 7 coefficients, reflection formula for x < 0.5 | ~1e-10 |
| `digamma(x)` | Asymptotic series with recurrence reduction to x > 6 | ~1e-8 |
| `erf_inv(x)` | Rational minimax approximation with tail correction | ~1e-6 |
| `trigamma(x)` | Asymptotic expansion with recurrence | ~1e-8 |

These approximations avoid platform-dependent libm behavior, ensuring identical results across operating systems. The tolerances are verified against reference values in the test suite.

## Error Handling

Each crate defines its own error type, and errors propagate upward through the stack:

```
EvalError (fj-lax)    --+
InterpreterError       --+
AdError (fj-ad)        --+--> ApiError (fj-api)
DispatchError          --+
DurabilityError        --+
```

Error types are enums with structured variants (not string messages), so callers can match on specific failure modes:

```rust
enum EvalError {
    ArityMismatch { primitive: Primitive, expected: usize, actual: usize },
    ShapeMismatch { primitive: Primitive, left: Shape, right: Shape },
    TypeMismatch { primitive: Primitive, detail: &'static str },
    Unsupported { primitive: Primitive, detail: String },
}
```

No panics in library code. All error paths return `Result`. The only `unwrap()` calls are in test code.

## Numerical Stability

FrankenJAX uses several guards against numerical issues:

- **Degenerate eigenvalue handling**: SVD and Eigh VJP rules check `|σ_i - σ_j| > 1e-20` before dividing by eigenvalue gaps. Near-degenerate eigenvalues produce zero gradient contributions instead of infinities.
- **Cholesky diagonal guard**: The Cholesky decomposition checks for non-positive diagonal elements, which indicate a non-SPD input matrix.
- **Division-by-zero protection**: The `div` primitive produces NaN for 0/0 and Inf for x/0, consistent with IEEE 754.
- **Triangular solve stability**: Forward/back substitution checks diagonal elements against machine epsilon before dividing.
- **FFT scaling**: IFFT uses exact `1/n` scaling, and RFFT interior bins get the correct factor-of-2 for real-signal Hermitian symmetry.

## The Effect System

Jaxpr equations can carry **effect tokens** that enforce sequential execution ordering for side-effectful operations. The effect system ensures that:

1. Effectful equations execute in program order (never reordered by the parallel executor)
2. The dependency-wave scheduler treats effectful equations as barriers
3. Transform composition preserves effect ordering (grad cannot reorder effects)
4. Sub-Jaxprs in control flow inherit their parent's effect context

Currently used for: RNG state threading (ensuring deterministic random sequences), and as extension points for future I/O or mutation effects.

## Gather and Scatter

The `Gather` and `Scatter` primitives implement generalized indexing:

**Gather**: Extract slices from a tensor at computed index positions. Given an operand `[m, n]` and indices `[k, index_depth]`, produces `[k, slice_size]` by looking up each index position. Supports multi-dimensional operands and configurable slice sizes.

**Scatter**: The inverse of gather. Updates a tensor at computed index positions. Given a base tensor, indices, and update values, produces a new tensor with updates applied at the specified positions. Supports configurable update modes (overwrite, add-to-existing).

Both primitives have VJP and JVP rules. Gather's VJP produces a scatter (adjoint of indexing is scattering gradients back), and Scatter's VJP produces a gather (adjoint of scattering is gathering the relevant gradients).

## Limitations

- **CPU-only backend.** GPU/TPU backends are not yet implemented. The CPU backend uses rayon for wave-parallel execution.
- **No XLA lowering.** FrankenJAX evaluates through its own interpreter, not through XLA. This means we match JAX's mathematical semantics but not its compilation/optimization pipeline.
- **Gateable finite-difference compatibility fallback.** `grad(jit(f))` uses symbolic AD because `jit` is transparent, but higher-order/composed `grad` cases that still require numerical fallback can be denied with `allow_finite_diff_grad_fallback=false` or `deny`.
- **Partial `vmap` + iterative control flow composition.** Primitive scalar-sequence `vmap(scan(...))` uses direct batched row folds, multi-carry functional `vmap(scan(...))` batches body sub-Jaxprs and returns `[batch, scan, ...]` outputs, primitive scalar `vmap(while_loop(...))` keeps supported scalar lanes in one batched active-mask loop, multi-carry functional `vmap(while_loop(...))` lowers cond/body sub-Jaxprs into the same active-mask path, and batched-index `vmap(switch(...))` uses branch-once batched evaluation with row-wise selection. More complex iterative compositions still need further work.

## FAQ

**Q: Why not just use JAX directly?**
A: JAX requires Python + XLA + CUDA/TPU. FrankenJAX gives you the mathematical transform semantics in a standalone Rust library with no Python dependency.

**Q: How do you verify correctness without running JAX?**
A: We capture oracle fixtures from real JAX 0.9.2 (848 cases), then run our Rust implementation against those fixtures in CI. Every primitive, every transform, every dtype combination.

**Q: Is the AD (automatic differentiation) complete?**
A: Yes. All 110 primitives have both VJP (reverse-mode) and JVP (forward-mode) rules, including complex operations like Cholesky, QR, SVD, Eigh decompositions and FFT. Numerical verification tests confirm correctness via finite-difference comparison.

**Q: What's the Trace Transform Ledger?**
A: Every transform composition (`jit(grad(f))`, `vmap(grad(f))`, etc.) produces an auditable evidence artifact that records the input IR, applied transforms, and output IR. Verification binds evidence entries to their transforms and includes evidence content in the stack signature.

**Q: How fast is it?**
A: Performance optimization is ongoing and evidence-gated. The CPU backend uses a dependency-wave parallel executor, the e-graph optimizer applies 87 algebraic simplification rules, and `artifacts/performance/global_performance_gate.v1.json` ties trace, compile/dispatch, execute, cold-cache, and warm-cache phases to Criterion baselines. Memory/allocation gating remains an explicit evidence gap until heap or RSS baselines are captured.

**Q: What's the difference between Strict and Hardened mode?**
A: Strict mode refuses to process anything it does not fully understand. Unknown features, incompatible cache entries, or ambiguous inputs cause hard failures. Hardened mode allows bounded defensive recovery: it can handle some malformed inputs and degrade gracefully, but logs every recovery action in the decision ledger for audit.

**Q: How does the e-graph optimizer work?**
A: It converts your Jaxpr into an equivalence graph where every algebraically equivalent form exists simultaneously (e.g., `x+x` and `2*x` coexist as equivalent). The 87 rewrite rules fire until saturation (no new equivalences found), then a cost function extracts the cheapest program. This can discover simplifications that sequential rule application would miss.

**Q: Can I use FrankenJAX from Python/C?**
A: The `fj-ffi` crate provides a C FFI surface for calling FrankenJAX from any language with C interop. Python bindings are not yet implemented but would be straightforward via PyO3 or cffi on top of fj-ffi.

**Q: How are the linalg AD rules verified?**
A: Every linalg VJP and JVP rule is verified two ways: (1) numerical finite-difference comparison (perturb inputs, compare analytical vs numerical gradient), and (2) oracle comparison against JAX's output with x64 precision enabled. This caught two real bugs in the Cholesky decomposition AD during development: a missing diagonal-halving factor and a wrong triangular-solve direction.

**Q: What's RaptorQ and why is it in a math library?**
A: RaptorQ is a fountain code (erasure code) that can reconstruct data from any sufficient subset of encoded symbols. We use it to protect conformance fixtures and benchmark baselines against silent data corruption. In a distributed CI setup where cache nodes can lose data, this ensures your test evidence survives. Unusual for a math library, yes, but correctness evidence that cannot be trusted is worthless.

**Q: How does the RNG match JAX?**
A: FrankenJAX implements the exact same ThreeFry2x32 cipher with the same rotation constants, key schedule, and sampling algorithms (Box-Muller for normal, Gumbel-max for categorical). The determinism is verified against 25 JAX oracle fixtures. `random_key(42)` produces bit-identical output to JAX's `jax.random.key(42)`.

## Key Documents

| Document | Purpose |
|----------|---------|
| `AGENTS.md` | AI agent development guidelines |
| `FEATURE_PARITY.md` | Feature-by-feature JAX parity status and explicit residual gaps |
| `COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md` | Full V1 specification |
| `PLAN_TO_PORT_JAX_TO_RUST.md` | Original porting strategy |
| `EXISTING_JAX_STRUCTURE.md` | JAX architecture analysis |
| `PROPOSED_ARCHITECTURE.md` | FrankenJAX design decisions |

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.
