# Changelog

All notable changes to [FrankenJAX](https://github.com/Dicklesworthstone/frankenjax) are documented in this file.

FrankenJAX is a clean-room Rust reimplementation of JAX's transform semantics. There are no formal releases, tags, or semver versions; the project is at `0.1.0` (workspace-wide) with continuous development on `main`. This changelog is organized by capability area within each development phase, derived from the complete 220-commit history.

---

## [Unreleased] — HEAD

Latest commit: [`0a7ec1b`](https://github.com/Dicklesworthstone/frankenjax/commit/0a7ec1b) (2026-05-01)

Current state: 110 primitive operations, VJP + JVP coverage for the declared V1 primitive set, 848 JAX oracle fixture cases, 87 e-graph rewrite rules, 15 workspace crates, 162,733 Rust source lines under `crates/`, 4,416 static Rust test/proptest markers, 115 conformance test files, and a passing `cargo test --workspace` run via RCH on 2026-05-01.

### 2026-05-01 Reality-Check Recalibration

- Reconciled public status docs with live evidence: 848 oracle fixtures, 4,416 static Rust test/proptest markers, 115 conformance test files, and the current HEAD commit.
- Downgraded over-broad "all green" language where the project still has tracked parity/evidence gaps: advanced transform/control-flow compositions, semantic TTL verification, all-long-lived-artifact durability coverage, and global performance gates.
- Created follow-up beads for remaining work: `frankenjax-fcxy.1` through `frankenjax-fcxy.5`.

---

## Phase 7 — Oracle Conformance, Linalg/FFT Autodiff, and Documentation (2026-03-12 to 2026-03-17)

Conformance coverage grew from 457 to 834 oracle fixture cases. Linear algebra decompositions (Cholesky, QR, SVD, Eigh, TriangularSolve) gained both eval kernels and full VJP/JVP rules. FFT primitives were implemented with autodiff. Two real Cholesky AD bugs were found and fixed by numerical verification. The README was rewritten to document all 110 primitives end-to-end.

### Linear Algebra Primitives

Pure-Rust eval kernels for five decompositions, plus full reverse-mode and forward-mode AD rules. Multi-output VJP/JVP handling for decompositions that return multiple matrices (QR returns Q and R, SVD returns U/S/Vt, Eigh returns W/V).

- Implement Cholesky, TriangularSolve, QR, SVD, and Eigh eval kernels ([`f58dc19`](https://github.com/Dicklesworthstone/frankenjax/commit/f58dc19d2d335676ebafe504186ef75cd9bf7285))
- Add multi-output VJP support for QR, SVD, Eigh decompositions ([`a07bc7b`](https://github.com/Dicklesworthstone/frankenjax/commit/a07bc7b3a71edaa59a9fb8898e3d87071e50d847))
- Add higher-order VJP rules for triangular solve, Cholesky, and einsum ([`bd3db9e`](https://github.com/Dicklesworthstone/frankenjax/commit/bd3db9e782532b9b38955a59ffc67c6398840288))
- Implement JVP rules for multi-output decompositions: Eigh, QR, SVD ([`e6eb996`](https://github.com/Dicklesworthstone/frankenjax/commit/e6eb996d729953ade8c50397ba32c04f25fbeacb))
- Remove unused SVD VJP transpose and update comment ([`b460a19`](https://github.com/Dicklesworthstone/frankenjax/commit/b460a1947c432bc9f691b94e87db319a1f6c2500))

### FFT Primitives

Naive O(n^2) DFT for correctness-first reference implementation. Supports batching along leading dimensions. RFFT exploits Hermitian symmetry; IRFFT reconstructs the full spectrum.

- Implement FFT primitives replacing V2 stubs in fj-lax ([`ee2c3bd`](https://github.com/Dicklesworthstone/frankenjax/commit/ee2c3bd39ff550627a7fb25ed96dd65e5b750330))
- Implement FFT/IFFT autodiff rules and harden FFT eval ([`4fc3062`](https://github.com/Dicklesworthstone/frankenjax/commit/4fc3062533b2290dfad142bfae1ab4f3c8a2f4e3))
- Implement RFFT/IRFFT autodiff rules and add TensorComplex128 conformance support ([`d970e4f`](https://github.com/Dicklesworthstone/frankenjax/commit/d970e4f3e122313c0f01580ed7fcc6b5ac6f2eab))

### Oracle Conformance Expansion (457 to 834 cases)

All fixtures captured from JAX 0.9.2 with x64 mode enabled. Coverage spans transforms, AD, linalg, FFT, RNG, dtype promotion, and transform composition.

- Re-capture all oracle fixtures from JAX 0.9.2 with fixed PRNG key extraction ([`4fc04bc`](https://github.com/Dicklesworthstone/frankenjax/commit/4fc04bc933cd44de243bf9de13a19ec5a37cc32c))
- Add dtype promotion oracle fixture: 162 JAX-verified cases covering the 9x9 dtype matrix ([`9ba620e`](https://github.com/Dicklesworthstone/frankenjax/commit/9ba620e2922d5ddadec219f45de7c18d520f92fc))
- Add transform composition oracle fixture and parity test ([`5e703dd`](https://github.com/Dicklesworthstone/frankenjax/commit/5e703ddcaecb1739d42c0b3960fc88969b3eab51))
- Add linalg/FFT oracle parity fixture bundle and test runner ([`03d60ec`](https://github.com/Dicklesworthstone/frankenjax/commit/03d60ec4403d880ab3da72e677ff29737596341d))
- Add conformance fixtures for 22 additional primitives reaching 596/596 parity ([`92250d6`](https://github.com/Dicklesworthstone/frankenjax/commit/92250d6927f92ec1f13e4f451214ac69b5ccd958))
- Add conformance fixtures for structural primitives: Iota, Copy, ExpandDims, Pad, BroadcastInDim ([`981fd11`](https://github.com/Dicklesworthstone/frankenjax/commit/981fd117d44b90510b18df15ab8b8d6e69515241))
- Add conformance fixtures for shape manipulation primitives ([`09d71e8`](https://github.com/Dicklesworthstone/frankenjax/commit/09d71e89b5a6f943dbcd9432bb2c9257c23cff12))
- Add conformance fixtures for sort, integer intrinsics, and reduction primitives ([`67d4ad0`](https://github.com/Dicklesworthstone/frankenjax/commit/67d4ad022f4b1e3a0766b127fc4d16c3f628cf7f))
- Expand lax conformance to special math, cumulative, and bitwise primitives ([`2f1ec76`](https://github.com/Dicklesworthstone/frankenjax/commit/2f1ec764932eb0de07d934a8d84cf1b31ad08aae))
- Add mixed_dtype conformance family for cross-dtype type promotion (457/457 pass) ([`3daf6fb`](https://github.com/Dicklesworthstone/frankenjax/commit/3daf6fb39a6a3c565eb8ac429133cdf9818b589b))
- Add complex number primitives and expand conformance capture tooling ([`c857a13`](https://github.com/Dicklesworthstone/frankenjax/commit/c857a13173722a3a419f1e67116986906c4f5f3d))
- Improve multirank conformance runner and test infrastructure ([`06f7ff3`](https://github.com/Dicklesworthstone/frankenjax/commit/06f7ff36f4ac5da610453d657b0d9aff9ff2aa6f))

### Correctness Verification

Numerical gradient verification caught two real Cholesky AD bugs. E-graph semantics preservation gate ensures optimization never changes program meaning.

- Add numerical gradient verification for AD VJP rules ([`22c02c4`](https://github.com/Dicklesworthstone/frankenjax/commit/22c02c4067d61c22b92283fb09f7f17858d53b78))
- Add e-graph optimization semantics-preserving gate ([`9d411d0`](https://github.com/Dicklesworthstone/frankenjax/commit/9d411d0654d092824adb644a7ae8cd6e5b4caeee))
- Add durability coverage tests and mark RaptorQ pipeline parity_green ([`0cd8363`](https://github.com/Dicklesworthstone/frankenjax/commit/0cd83630d9e6bdb1c0571d5963f309d46b5dfd05))
- Add transform API unit tests and JVP numerical verification ([`b5b0958`](https://github.com/Dicklesworthstone/frankenjax/commit/b5b0958984fb30704896d8d927c86700ca96e1b0))

### Bug Fixes (AD)

Both Cholesky bugs were discovered by the numerical gradient verification tests, not by manual inspection.

- Correct Cholesky VJP diagonal factor and back-substitution direction ([`6e79009`](https://github.com/Dicklesworthstone/frankenjax/commit/6e7900954c726a3b6de4a932147555f0aafb2153))
- Correct Cholesky JVP middle matrix solve direction ([`59fe202`](https://github.com/Dicklesworthstone/frankenjax/commit/59fe202a7d6517c9b35220ae0da3b5f58defcaca))
- Remove dead code, use `slice::from_ref`, and fix clippy lints ([`260b4ce`](https://github.com/Dicklesworthstone/frankenjax/commit/260b4ceaea11684a102a07b3c32bb82fa8744923))
- Use `slice::from_ref` instead of clone in tensor_ops tests ([`730edae`](https://github.com/Dicklesworthstone/frankenjax/commit/730edae8b98992d1c3a4e849c9a2e6d06d5a387f))
- Update capture script fixture path handling ([`920b90e`](https://github.com/Dicklesworthstone/frankenjax/commit/920b90e32bf03050fbbccb2ed86ab3d3304b1a6e))

### Tests

- Add comprehensive unit tests for arithmetic, comparison, and reduction ops ([`8158c37`](https://github.com/Dicklesworthstone/frankenjax/commit/8158c371a5c53a95bb1e156514c3d025abdc6cb6))
- Add unit tests for tensor reshape, transpose, slice, pad, concatenate, and broadcast ([`7609ac2`](https://github.com/Dicklesworthstone/frankenjax/commit/7609ac2d9696b43d12262bdc25b331170dc259bf))

### Documentation

- Rewrite README with 110 primitives, oracle conformance, and AD verification ([`9e6ecee`](https://github.com/Dicklesworthstone/frankenjax/commit/9e6eceea8155ebd719d1b4ec7b8d9da0f2f97208))
- Expand README with transform composition, conformance fixtures, and performance data ([`aa6fbf4`](https://github.com/Dicklesworthstone/frankenjax/commit/aa6fbf44d36656594c8ef34152cb1789fcba09af))
- Update parity to 110 ops and add linalg/FFT oracle tests ([`f7d8fcd`](https://github.com/Dicklesworthstone/frankenjax/commit/f7d8fcd6ce79bb2fa46fce51aa767698f9be57aa))
- Simplify Cholesky VJP step 3 comments to match corrected implementation ([`e65d143`](https://github.com/Dicklesworthstone/frankenjax/commit/e65d1430d6d17bd382c68b7f4bc9d574c03fd89d))
- Update e-graph feature parity entry with conformance test coverage ([`44f4459`](https://github.com/Dicklesworthstone/frankenjax/commit/44f4459b719701dfa8c22eeb6b4f87ef1de61b7c))
- Update linalg/FFT parity entry with oracle fixture coverage ([`29ad424`](https://github.com/Dicklesworthstone/frankenjax/commit/29ad424749081f56cc2c32c71dee8f41f98006ff))
- Replace Unicode combining macron with plain g in README gradient examples ([`0e62ff4`](https://github.com/Dicklesworthstone/frankenjax/commit/0e62ff4693a476edbdd2444279b8cd91c0feadc2))
- Rustfmt formatting for ShiftRightArithmetic match arm ([`f113078`](https://github.com/Dicklesworthstone/frankenjax/commit/f113078e496e47d848fd67800390417b77ba688e))

### Artifacts

- Update parity report artifacts with expanded conformance (546/546 pass) ([`5735888`](https://github.com/Dicklesworthstone/frankenjax/commit/57358884224589e28b69aa26fc40c78a3d733498))
- Update parity report artifacts with mixed_dtype results (457/457 pass) ([`4091da6`](https://github.com/Dicklesworthstone/frankenjax/commit/4091da6497d13ce4824d0312b42c7d22964089ae))
- Regenerate conformance artifacts and expand AD gradient fixtures ([`93ac343`](https://github.com/Dicklesworthstone/frankenjax/commit/93ac343c3d3f6d6d7b7f9fc9faa24862ea431052))
- Regenerate conformance artifacts and expand AD gradient coverage ([`1b44822`](https://github.com/Dicklesworthstone/frankenjax/commit/1b448229b4fe47b27914d74027aa83ef7a3bb95d))
- Update CI/conformance/e2e test artifacts after complex primitives expansion ([`4d2432f`](https://github.com/Dicklesworthstone/frankenjax/commit/4d2432f2ac74ef8baa2c0c918becb89baf9ad543))

---

## Phase 6 — E-graph Dispatch Integration, Control Flow, and Durability (2026-03-04 to 2026-03-12)

E-graph rewriting integrated into the dispatch pipeline so optimized programs can be executed end-to-end. Control flow primitives (cond, scan) added with RNG parity verification. ScalarBool type support. RaptorQ durability proof artifacts generated for long-lived evidence.

### E-graph and Dispatch Integration

- Integrate e-graph rewriting into dispatch and expand conformance ([`1d5fac7`](https://github.com/Dicklesworthstone/frankenjax/commit/1d5fac78d47099129b01bb862c14d6bdf210b91f))

### Control Flow and RNG

- Add control flow primitives (cond, scan) and RNG parity verification ([`aeb4648`](https://github.com/Dicklesworthstone/frankenjax/commit/aeb4648249f3da322caf90f9350ea07dd9365f23))

### Type System

- Add ScalarBool support, expand AD gradient coverage, and generate durability proof artifacts ([`7026442`](https://github.com/Dicklesworthstone/frankenjax/commit/702644231fbf4798164f2069d1e91b4d57c717ae))
- Expand AD engine, conformance suites, and core type system ([`21e435d`](https://github.com/Dicklesworthstone/frankenjax/commit/21e435da4b0e15226eaedf3500f89d57828538a1))

### Infrastructure

- Add shape inference tests and CI validation artifacts ([`6b6ff6c`](https://github.com/Dicklesworthstone/frankenjax/commit/6b6ff6c519a7866f9155ddca498c2ac7503d1aab))

---

## Phase 5 — Type System Expansion, Higher-Order AD, and Parallel Backend (2026-03-02 to 2026-03-04)

Major type system expansion: unsigned integers (U32/U64), half-precision floats (BF16/F16), complex number primitives. Higher-order derivatives via Jacobian and Hessian. Unified `value_and_grad` with shared forward pass. The CPU backend switched from sequential interpretation to dependency-wave parallel execution.

### Higher-Order Derivatives and Transform Composition

- Implement Jacobian and Hessian matrix computation via composable AD ([`4032637`](https://github.com/Dicklesworthstone/frankenjax/commit/4032637639bbf2a8279d3423ab802923b7c41929))
- Unify value_and_grad into a single shared forward pass ([`e72afd0`](https://github.com/Dicklesworthstone/frankenjax/commit/e72afd00d334886c600201fb3030d2e6fc2ec519))
- Expose custom VJP registration and add RNG determinism conformance suite ([`da05694`](https://github.com/Dicklesworthstone/frankenjax/commit/da05694efb83f8933e517d4ae0709782e308c7bf))
- Nested trace context infrastructure and BF16/F16 dtype support ([`3304b13`](https://github.com/Dicklesworthstone/frankenjax/commit/3304b13d222a688d4ffc6cf09822721304a8f03e))
- Add differentiation, batching, and tracing for BroadcastedIota, Copy, BitcastConvertType, and ReducePrecision ([`5fb6add`](https://github.com/Dicklesworthstone/frankenjax/commit/5fb6add8f5974760f172f7be73875243ce88038d))

### DType Expansion (5 to 11 dtypes)

- Add unsigned integers (U32/U64), half-precision floats (BF16/F16), and complex number primitives across core, lax, ad, and ffi ([`563285e`](https://github.com/Dicklesworthstone/frankenjax/commit/563285e451080277b89f85a493986193fe2178f5))
- Complete BF16/F16 half-precision support across all crates ([`1e688cd`](https://github.com/Dicklesworthstone/frankenjax/commit/1e688cd8dba71389c8e04cf0755eca87ee527e38))
- Add ReduceAnd, ReduceOr, ReduceXor bitwise reduction primitives ([`b2faf2d`](https://github.com/Dicklesworthstone/frankenjax/commit/b2faf2d5f2f8efb7351824d3c73db2d4eea7b60a))
- Split ShiftRight into ShiftRightArithmetic and ShiftRightLogical ([`3f10471`](https://github.com/Dicklesworthstone/frankenjax/commit/3f10471a4f6bb67e3ae6d3b8c3d130f772d065e7))
- Add effects field to Jaxpr/Equation IR and value_and_grad API ([`42026a6`](https://github.com/Dicklesworthstone/frankenjax/commit/42026a6b5972680fe68bdcd74acceb93c4d79bb6))

### Parallel CPU Backend

- Replace sequential interpreter with dependency-wave parallel executor using Rayon ([`92381c8`](https://github.com/Dicklesworthstone/frankenjax/commit/92381c84d250bb7cfc2c88063cd6bc9d6b4218f0))
- Dedicated batch rules for control flow and indexing ops ([`ec39fa8`](https://github.com/Dicklesworthstone/frankenjax/commit/ec39fa867f3209cd7010c5a157de5a2ea653e030))
- Add LAX tensor ops layer and tracing infrastructure ([`d8ccf71`](https://github.com/Dicklesworthstone/frankenjax/commit/d8ccf71cf0172349c963cee00e6383b34e6da436))

### Conformance

- Add control flow conformance suite and v1 parity golden data ([`7ff370d`](https://github.com/Dicklesworthstone/frankenjax/commit/7ff370dd8f742be6beebed8ce1950a8fee3ff383))

### Bug Fixes

- Symmetric subgradient tie-averaging in Sort VJP ([`596b9c2`](https://github.com/Dicklesworthstone/frankenjax/commit/596b9c22667a753845dbab6a4e1ce13b680f3ec6))

### Performance

- Add value_and_grad shared-forward-pass vs separate-calls benchmarks ([`204314b`](https://github.com/Dicklesworthstone/frankenjax/commit/204314b62f8b66a993e7a3562772b365e2e7522e))

### Documentation

- Overhaul feature parity matrix, expand security threat model, update README ([`7e8bcaa`](https://github.com/Dicklesworthstone/frankenjax/commit/7e8bcaa86527f9b3797e97c6aa1f7067f354093f))

### Code Quality

- Code quality cleanup: remove unused import, suppress dead_code warnings, use let-chains ([`d2c29b5`](https://github.com/Dicklesworthstone/frankenjax/commit/d2c29b503ea70a547ee2f5a2ef712b7083983f52))

---

## Phase 4 — Vmap, Control Flow Primitives, and PRNG (2026-02-25 to 2026-02-26)

BatchTrace interpreter for `vmap` with `in_axes`/`out_axes`. Full control flow suite: cond, scan, while_loop, fori_loop, switch -- all with VJP and JVP support. Eight new primitives. ThreeFry2x32 PRNG. Complex64/Complex128 dtype support introduced at the type level. `make_jaxpr` tracing API from Rust closures.

### Vmap (BatchTrace Interpreter)

O(1) per-primitive batching rules that avoid looping over the batch dimension. Elementwise ops pass the batch dim through; reductions shift the axis; shape ops exclude the batch dim.

- Implement BatchTrace interpreter and fix ReduceWindow VJP ([`74eeec3`](https://github.com/Dicklesworthstone/frankenjax/commit/74eeec3009b83651edb2c3dba74b50c894e0e504))
- Remove scalar-only output constraint and fix BatchTrace broadcasting ([`45297fd`](https://github.com/Dicklesworthstone/frankenjax/commit/45297fddd6fbb1c0ba91a13c974f25a5c3da9859))
- Add in_axes/out_axes to vmap and rewrite while_loop with functional API ([`3a7dcb5`](https://github.com/Dicklesworthstone/frankenjax/commit/3a7dcb56afb63ec570926fe48cc9d5c536faae01))
- Add per-dtype tolerance tiers with violation reporting and regenerate test artifacts ([`4a4986c`](https://github.com/Dicklesworthstone/frankenjax/commit/4a4986cca06eae59908ee8e02d4e7110084245c8))

### Control Flow and New Primitives

- Add Switch primitive with VJP support and fori_loop evaluator ([`845001e`](https://github.com/Dicklesworthstone/frankenjax/commit/845001e092c0da4acf0a1f16a28333bbd5f10b22))
- Add JVP (forward-mode) rule for Switch primitive ([`0aa23e9`](https://github.com/Dicklesworthstone/frankenjax/commit/0aa23e9b6c6ac500dd604e135b1b04ce167c06ed))
- Add functional scan API, ThreeFry2x32 PRNG, and vmap in_axes/out_axes tests ([`da8cb1a`](https://github.com/Dicklesworthstone/frankenjax/commit/da8cb1a1137b7d34de85eb11a84c0403a1b4d8d0))
- Add new primitives: Rev, Squeeze, Split, ExpandDims, Cbrt, IsFinite, IntegerPow, Nextafter; tensor-valued JVP; make_jaxpr API; PRNG sampling functions ([`074194b`](https://github.com/Dicklesworthstone/frankenjax/commit/074194b5e244ec917a0d814031b7ef12fc0ced45))
- Implement control flow VJPs, bitwise ops, and trace compilation ([`e182573`](https://github.com/Dicklesworthstone/frankenjax/commit/e182573114c5f96a411bfaa9088b89d6450b04e6))
- Implement full VJP suite for remaining primitives and expand LAX/tensor ops ([`c064058`](https://github.com/Dicklesworthstone/frankenjax/commit/c06405883734e9b93ba5ee43ba6d8f2ea81a9402))
- Implement pad/gather/scatter VJPs, expand LAX tensor ops, and add trace compilation ([`cb4d4a3`](https://github.com/Dicklesworthstone/frankenjax/commit/cb4d4a3262304ee85622b1dc053aecf55a4ba2f2))

### Complex Number Types

- Add Complex64 and Complex128 to DType and Literal enums ([`a8f7859`](https://github.com/Dicklesworthstone/frankenjax/commit/a8f78596b79a065e918aa40156ffddf4d2c294dd))

### Conformance

- Rewrite oracle capture script with lax primitive coverage ([`7612b3c`](https://github.com/Dicklesworthstone/frankenjax/commit/7612b3c182f498ad6b579ce2e3d2021dd7eecf0a))

### Bug Fixes

- Add sub_jaxprs field to Equation structs and extend primitive arity coverage ([`02d1e5c`](https://github.com/Dicklesworthstone/frankenjax/commit/02d1e5cb832b272cd99f20d5990f2d9d4c04c317))
- Rustfmt formatting in sort VJP and add conv2d kernel rank validation ([`7855d5e`](https://github.com/Dicklesworthstone/frankenjax/commit/7855d5effc7739473f45e32272d5df9564d0ad60))
- Add rank-3 tensor tests and fix pad expected dimensions ([`0f0d540`](https://github.com/Dicklesworthstone/frankenjax/commit/0f0d540719097b3bc39b69db6e945f9c9f80c943))

---

## Phase 3 — AD Hardening, E-graph Rules, and Typed PE (2026-02-21 to 2026-02-25)

Systematic AD correctness hardening with numerical gradient verification for 22 unary and 7 binary VJP rules plus composition tests. E-graph algebraic rewrite rules for trig/hyperbolic identities (growing toward 87 total rules). Typed partial evaluation preserving residual abstract value types. BackendRegistry with device-aware fallback. New primitives: Pad, Clamp, DynamicSlice, Iota. License updated to MIT with OpenAI/Anthropic Rider.

### AD Numerical Verification

Finite-difference gradient comparison for all differentiable primitives. Composition tests verify chain rule through nested functions.

- Add systematic numerical gradient verification for 22 unary VJP rules ([`dbce07c`](https://github.com/Dicklesworthstone/frankenjax/commit/dbce07c0862c5d42c8d272e164ca24ab29dd6803))
- Add binary VJP numerical gradient verification for Add, Sub, Mul, Div, Pow, Atan2, Rem ([`4620618`](https://github.com/Dicklesworthstone/frankenjax/commit/4620618b6f5c4120be8b87663448a53e846a2930))
- Add composition gradient tests: exp(sin), sin\*cos, log(x^2+1), tanh(2x) ([`f75dcdc`](https://github.com/Dicklesworthstone/frankenjax/commit/f75dcdc693e564878a47a5a8e2663a184e61d411))
- Add Max/Min VJP tie-breaking tests and higher-order gradient tests ([`baab370`](https://github.com/Dicklesworthstone/frankenjax/commit/baab37056d614ca68de33ef134935080825d04fc))

### AD Correctness Fixes

Scatter, slice, and reduction VJPs all required axis-aware corrections for multi-dimensional tensors.

- Add axis-aware gradient broadcasting for partial reductions in VJP backward pass ([`594289a`](https://github.com/Dicklesworthstone/frankenjax/commit/594289a403541389b4f9272e86e84a4e2eaa8a13))
- Differentiate scatter backward pass by mode: pass-through gradient for additive scatter ([`a5457e9`](https://github.com/Dicklesworthstone/frankenjax/commit/a5457e9144728fd9a5f0766579b7d5c190b373cf))
- Validate scatter VJP indices are non-negative and finite before usize cast ([`21c8afc`](https://github.com/Dicklesworthstone/frankenjax/commit/21c8afc784f28738e4801d3371fdc506d0bbb363))
- Fix axis-aware ReduceMax/ReduceMin VJP, prevent TTL cache memory leak, and materialize dangling e-graph literals ([`f74e223`](https://github.com/Dicklesworthstone/frankenjax/commit/f74e2236f2c33a87b8db3c39c993ff7c7f206758))
- Fix multi-dimensional VJP slice adjoint, harden dtype promotion in select ([`a890719`](https://github.com/Dicklesworthstone/frankenjax/commit/a890719acfab3c90514c3cd4bf2a5ff679247d60))
- Fix AD edge cases, add constvar support to e-graph, overhaul dtype promotion, and harden LAX ops ([`9495a63`](https://github.com/Dicklesworthstone/frankenjax/commit/9495a6339845cdcc982b027aa77d5a05f8c15c3c))
- Fix compilation errors, harden AD VJP rules, and add Select broadcasting for JAX parity ([`cfd2882`](https://github.com/Dicklesworthstone/frankenjax/commit/cfd2882bbf8b812aec6cf3e788db382f10f66faa))

### E-graph Rewrite Rules

Algebraic identity rules for trig/hyperbolic functions, reciprocal involution, and nested select simplification. Constvar support enables constant folding within the e-graph.

- Add e-graph algebraic rewrite rules for trig/hyperbolic identities, reciprocal involution, and nested select simplification ([`e282c93`](https://github.com/Dicklesworthstone/frankenjax/commit/e282c931a5b1ef127615a6e667361e4f13d9d211))
- Harden gather/scatter validation, add e-graph algebraic rules, and expand edge-case test coverage ([`38b51d2`](https://github.com/Dicklesworthstone/frankenjax/commit/38b51d289e9a6bf9740ccf27ae73ebe0813fd662))

### New Primitives

- Add Pad primitive with edge and interior padding support ([`4f3ed85`](https://github.com/Dicklesworthstone/frankenjax/commit/4f3ed853197f6c08d235d91dce56f725b8bf8d9f))
- Add Clamp/DynamicSlice/Iota primitives and wire dispatch through backend ([`5f41d20`](https://github.com/Dicklesworthstone/frankenjax/commit/5f41d20dc9f25ba448b55a977119c8b580c31299))
- Add DynamicSlice/Clamp/Iota support to trace, egraph, and PE type inference ([`e5b4b3a`](https://github.com/Dicklesworthstone/frankenjax/commit/e5b4b3ace992559af913042931682e79f7925e0b))

### Partial Evaluation and Interpreters

- Add typed partial evaluation to preserve residual abstract value types ([`dc2e311`](https://github.com/Dicklesworthstone/frankenjax/commit/dc2e311c08dbf406ef899175af671d21b027517f))
- Extend PE type inference for axis-aware reductions, Dot, and shape ops ([`a9795a1`](https://github.com/Dicklesworthstone/frankenjax/commit/a9795a1f5a5019bf46ce83033912dc9f871ea314))

### Backend and Runtime

- Refactor dispatch to use BackendRegistry with device-aware fallback ([`9856878`](https://github.com/Dicklesworthstone/frankenjax/commit/9856878a14688b360970c77742a1836fd6ad5c38))
- Add BackendExecution error variant mapping to ApiError ([`e079973`](https://github.com/Dicklesworthstone/frankenjax/commit/e07997389ccd516f1d19ed35830f2201a094c055))
- Add Value::as_i64_scalar, as_bool_scalar, dtype() and TensorValue::to_i64_vec ([`9b2568e`](https://github.com/Dicklesworthstone/frankenjax/commit/9b2568ec36f78d64faecd303c63b374426782825))

### Tests

- Add scatter mode="add" accumulation tests ([`1322ea0`](https://github.com/Dicklesworthstone/frankenjax/commit/1322ea0a7d92ac8941bd84832db68dc98673bd31))
- Strengthen Scatter validation and add Scatter/Gather trace tests ([`9263c24`](https://github.com/Dicklesworthstone/frankenjax/commit/9263c2411cab1b98984c68ee2e8b58049b405caa))
- Add higher-rank tensor and transform composition dispatch tests ([`9f65903`](https://github.com/Dicklesworthstone/frankenjax/commit/9f65903e3b0e1ee3e087355b24a0268da2303bcd))

### Chore

- Update license to MIT with OpenAI/Anthropic Rider ([`ebe06eb`](https://github.com/Dicklesworthstone/frankenjax/commit/ebe06eb1d7c629fcadb647d1108a2d8a9614c7ed))
- Switch asupersync + ftui from local paths to crates.io ([`0cbfe47`](https://github.com/Dicklesworthstone/frankenjax/commit/0cbfe47eeab94e5c48a0a7d6390345526fe356e2))
- Extend AD, egraph, cache eviction, LAX ops, and conformance tests ([`9a8924f`](https://github.com/Dicklesworthstone/frankenjax/commit/9a8924f2f1db1bcce2747715e4458a7dfdc2d4b7))

---

## Phase 2 — P2C Infrastructure Build-out (2026-02-20)

Systematic Phase-2C build-out establishing the five core subsystems. Each subsystem followed the same rigorous process: legacy anchor map, contract table, security threat model, implementation, unit/property tests, differential oracle + metamorphic + adversarial tests, E2E scenarios, performance benchmarks, and final evidence pack with RaptorQ durability sidecars. This phase created fj-api, fj-dispatch, fj-cache, fj-backend-cpu, fj-ffi, and the fj-lax module boundary skeleton.

### API Transform Front-Door (fj-api, P2C-002)

User-facing API surface: `jit`, `grad`, `vmap` with composable transform stacks.

- Implement API transform front-door in Rust ([`58dbffa`](https://github.com/Dicklesworthstone/frankenjax/commit/58dbffafe61b28ba59bc20bfb43d309e1ceb372d))
- Profile-driven optimization + isomorphism proof ([`7c04cf4`](https://github.com/Dicklesworthstone/frankenjax/commit/7c04cf4d862e21b4ccb63905dad8cf377aabe5f6))
- Legacy anchor map ([`060500f`](https://github.com/Dicklesworthstone/frankenjax/commit/060500f4230b35a8f56a6b9225bb69ed38bc3868)), contract table ([`1778eeb`](https://github.com/Dicklesworthstone/frankenjax/commit/1778eebf1ebf9ca62b0c617c721317a1fd065aef)), security threat model ([`ed6a304`](https://github.com/Dicklesworthstone/frankenjax/commit/ed6a304e02c45189ed30ea76184f6b7e446cc487)), crate skeleton ([`441c257`](https://github.com/Dicklesworthstone/frankenjax/commit/441c2579aadc36a70984b2d8387f49651d523cf1))
- Unit + property tests ([`d79c221`](https://github.com/Dicklesworthstone/frankenjax/commit/d79c221abec88c38d7a5d85bdccca59aa9881ec4)), differential oracle tests ([`23c3ae3`](https://github.com/Dicklesworthstone/frankenjax/commit/23c3ae39259db1d47aa0a74c3a1c0ab8c948a458)), E2E scenarios ([`19df570`](https://github.com/Dicklesworthstone/frankenjax/commit/19df5707136130dcc00e7e708e29fca0cb5d3184))
- Final evidence pack with parity gates + RaptorQ sidecars ([`75e07d9`](https://github.com/Dicklesworthstone/frankenjax/commit/75e07d98ac20e29f9570b0bba12f1914bec60ee7), [`8cbea29`](https://github.com/Dicklesworthstone/frankenjax/commit/8cbea29b7af6416906121a1600bd4966529ef9af))

### Dispatch/Effects Runtime (fj-dispatch, P2C-004)

Transform dispatch with order-sensitive composition, JVP support, and effect token threading.

- Implement forward-mode JVP + effect token system ([`d719da5`](https://github.com/Dicklesworthstone/frankenjax/commit/d719da52f0b2e13b7537c698ced608105b927a59))
- Dispatch optimization: iterative Jit-skip + Vec effect tokens ([`f3f7523`](https://github.com/Dicklesworthstone/frankenjax/commit/f3f75231521e6a9ccd9d5abd7779ffb19e40cc91))
- Legacy anchor map ([`9831a15`](https://github.com/Dicklesworthstone/frankenjax/commit/9831a1586d908056acdc317825adbede50ee9ff5)), contract table ([`80b14b5`](https://github.com/Dicklesworthstone/frankenjax/commit/80b14b5455c531a442dcafdb91de4f13f4a0ff38)), security threat model ([`32f7ea8`](https://github.com/Dicklesworthstone/frankenjax/commit/32f7ea8d99100979fa9ad0d957a3f928d126d054)), module skeleton ([`9196dc4`](https://github.com/Dicklesworthstone/frankenjax/commit/9196dc4f29b93e94199667e5616c13f7c36eae3b))
- Unit + property tests ([`a1c6356`](https://github.com/Dicklesworthstone/frankenjax/commit/a1c6356ae2d0ccd10616722dd2452b424ab6e61f)), differential oracle tests ([`649defd`](https://github.com/Dicklesworthstone/frankenjax/commit/649defd7e01317857e280b9e195d6a71d7ac243f)), E2E scenarios ([`7346769`](https://github.com/Dicklesworthstone/frankenjax/commit/73467696940ebebec53bd5dc025b71d0e4d36836))
- Final evidence pack with RaptorQ sidecars ([`a27f087`](https://github.com/Dicklesworthstone/frankenjax/commit/a27f08786aa188281c4d084c589047d44d4672b3))

### Compilation Cache/Keying (fj-cache, P2C-005)

SHA-256 deterministic cache keys with strict/hardened gate behavior. LRU eviction with persistence.

- CacheManager with persistence, eviction, and integrity verification ([`168f425`](https://github.com/Dicklesworthstone/frankenjax/commit/168f42540c4e1696de0b8bccb0ed60539797a3e7))
- Zero-alloc streaming hasher: 32.6% faster cache key generation ([`031f3d7`](https://github.com/Dicklesworthstone/frankenjax/commit/031f3d72788e6be357cce22bf6dc6c68fe6eecc9))
- Legacy anchor map ([`365ca3a`](https://github.com/Dicklesworthstone/frankenjax/commit/365ca3ab92be99e90e1afffdc4fe80abc7b7706c)), contract table ([`8dd3853`](https://github.com/Dicklesworthstone/frankenjax/commit/8dd3853721cf5a130ad1520458688be0af2402be)), security threat model ([`8c6dd47`](https://github.com/Dicklesworthstone/frankenjax/commit/8c6dd473ae3dfa1cca091092aebaa58aa87991e7)), module skeleton ([`3c575b2`](https://github.com/Dicklesworthstone/frankenjax/commit/3c575b24947730d33ddd343bd20a1a061e7445a2))
- Unit + property tests ([`9e2559e`](https://github.com/Dicklesworthstone/frankenjax/commit/9e2559e0f5286bc05a50c64c51069eb1fb11c001)), differential oracle tests ([`d91f1f2`](https://github.com/Dicklesworthstone/frankenjax/commit/d91f1f237b698c3534dc8b8d286a8b7ef1b3251c)), E2E scenarios ([`e7aee7b`](https://github.com/Dicklesworthstone/frankenjax/commit/e7aee7bfbb52cefa635ccfa7ba08582a5807994e))
- Final evidence pack with RaptorQ sidecars ([`6b65817`](https://github.com/Dicklesworthstone/frankenjax/commit/6b65817403bd848677ec3f1d2f2f9c1bc7bedc67))

### Backend Bridge (fj-backend-cpu, P2C-006)

CPU backend with device capability detection, registry pattern, and platform routing.

- Backend capabilities, registry, and platform routing ([`2eb290b`](https://github.com/Dicklesworthstone/frankenjax/commit/2eb290bf3f5a1390b31ed4bafba90098420dff2b))
- CPU backend baseline benchmarks ([`f6dabbf`](https://github.com/Dicklesworthstone/frankenjax/commit/f6dabbf42f97101275f37f80639c4934911f13bb))
- Legacy anchor map ([`d4dda74`](https://github.com/Dicklesworthstone/frankenjax/commit/d4dda74df0e1a50a64c658bbf8133950ff4981a3)), contract table ([`119f1a2`](https://github.com/Dicklesworthstone/frankenjax/commit/119f1a2cb2b368ad789a31b5c857c833244490a9)), security threat model ([`25f52d7`](https://github.com/Dicklesworthstone/frankenjax/commit/25f52d7742a984d8a958e9c3b0e255eaf3f57ae3)), module skeleton ([`000ffeb`](https://github.com/Dicklesworthstone/frankenjax/commit/000ffeb9b087e15815568c01e48103c48f986ddc))
- Unit + property tests ([`5841fb3`](https://github.com/Dicklesworthstone/frankenjax/commit/5841fb3ace939087f502c9b3d4e7ee8fe45ff6e6)), differential oracle tests ([`9108086`](https://github.com/Dicklesworthstone/frankenjax/commit/9108086a3ce288d7a7dabd9373b673b89c8a3f43)), E2E scenarios ([`b029660`](https://github.com/Dicklesworthstone/frankenjax/commit/b0296607180dd5f9cd78bdf58efe8a18c142ace6))
- Final evidence pack ([`a024672`](https://github.com/Dicklesworthstone/frankenjax/commit/a024672b2474b85d5a6b1ff99d8bf7e0f04aaea3))

### FFI Call Interface (fj-ffi, P2C-007)

C FFI surface -- the only crate in the workspace permitted to use `unsafe`.

- FFI call interface implementation ([`631ddb6`](https://github.com/Dicklesworthstone/frankenjax/commit/631ddb6524826dd23bbc02d659567520cfe7700a))
- FFI call interface baseline benchmarks ([`687960c`](https://github.com/Dicklesworthstone/frankenjax/commit/687960c78a4f7e1d313dac09f4af077d8c5dcb11))
- Legacy anchor map ([`f0a4530`](https://github.com/Dicklesworthstone/frankenjax/commit/f0a453066df0ea64d357e17331680a811d3d5bf0)), contract table ([`3b64b35`](https://github.com/Dicklesworthstone/frankenjax/commit/3b64b35d88d090b97b706e5ad4efe7a6de748c5d)), security threat matrix ([`acdd55c`](https://github.com/Dicklesworthstone/frankenjax/commit/acdd55c7344ccb9445be1d3ffd01f87b5034f6c5)), crate skeleton ([`b137297`](https://github.com/Dicklesworthstone/frankenjax/commit/b137297eaec575d061bb672e55d789c7f4200011))
- Unit + property tests ([`7a8f05b`](https://github.com/Dicklesworthstone/frankenjax/commit/7a8f05bfec86f73271a1c730bc3e2eb34509ebe7)), differential oracle tests ([`8786945`](https://github.com/Dicklesworthstone/frankenjax/commit/878694567f26ea5c8b6f63f3e44821cd05d2684b)), E2E scenarios ([`6a81c82`](https://github.com/Dicklesworthstone/frankenjax/commit/6a81c82f1fbd8f4fe6c1c1516820ab4422ebdc3a))
- Final evidence pack ([`31ce439`](https://github.com/Dicklesworthstone/frankenjax/commit/31ce43936030da8384e0cf4f9deaeb0fa8d424cf))

### LAX Primitive First Wave (fj-lax, P2C-008)

Module boundary skeleton for fj-lax with initial primitive set, comprehensive tests, and baseline benchmarks.

- Refactor fj-lax into module boundary skeleton ([`5e03cb3`](https://github.com/Dicklesworthstone/frankenjax/commit/5e03cb3b35dfa4c5f12e0376ffdc690321ce272f))
- LAX primitive baseline benchmarks ([`7cb1acd`](https://github.com/Dicklesworthstone/frankenjax/commit/7cb1acd7c1daa22232fdf7b8a65e0498755e7873))
- Legacy anchor map ([`22bdf69`](https://github.com/Dicklesworthstone/frankenjax/commit/22bdf69f6ef4555093d8e901374b25e73515a27e)), contract table ([`4688be4`](https://github.com/Dicklesworthstone/frankenjax/commit/4688be46176a3aaff4ea31021b25673e9b9d67e4)), security threat matrix ([`3a22d3a`](https://github.com/Dicklesworthstone/frankenjax/commit/3a22d3aa21dd029567556dc12edf3ab9b70abbe6))
- Unit + property tests ([`accbf8a`](https://github.com/Dicklesworthstone/frankenjax/commit/accbf8a55a18bbd380354f733f2367e4b1871982)), differential oracle tests ([`3a07cdc`](https://github.com/Dicklesworthstone/frankenjax/commit/3a07cdc2145a6794148b903592cde223c6e9fae8)), E2E scenarios ([`68fa108`](https://github.com/Dicklesworthstone/frankenjax/commit/68fa10862a2b3ff452e9452e778e62d4d2f8f21b))

### Cross-Cutting Work

- Add fj-backend-cpu and fj-ffi workspace crates ([`1663bf2`](https://github.com/Dicklesworthstone/frankenjax/commit/1663bf227880717fe4c86279ea6fd1b8274ce7fc))
- Phase-2C readiness drill and sign-off ([`683374d`](https://github.com/Dicklesworthstone/frankenjax/commit/683374ddec34ae201e26876e6bc82244d2fd42c7))
- Expand LAX primitive set with tensor-aware AD, new math ops, axis-aware reductions, and select ([`841ea88`](https://github.com/Dicklesworthstone/frankenjax/commit/841ea889cec6da82ed0c2bb0dacad16c8e1f1770))
- Broad improvements to AD, staging, dispatch, and conformance suite ([`31595e0`](https://github.com/Dicklesworthstone/frankenjax/commit/31595e0f5c1a53541efd3a42d3065e872d766409))
- Extend LAX lowering implementation ([`a6a76ea`](https://github.com/Dicklesworthstone/frankenjax/commit/a6a76ea00499fde9e97b7bb1cfb9f91554850b98))
- Expand fj-lax interpreter with additional operations ([`cd4c27c`](https://github.com/Dicklesworthstone/frankenjax/commit/cd4c27c7ccb7f01122dbfb5267bc048cc1d1285f))
- Resolve clippy warnings and apply formatting fixes ([`59f6eb3`](https://github.com/Dicklesworthstone/frankenjax/commit/59f6eb32ae949425710ecad4b47793b71f4b02b8))

### Documentation

- Complexity, performance, and memory characterization (DOC-PASS-05) ([`0978673`](https://github.com/Dicklesworthstone/frankenjax/commit/097867331a9b9f74e106905a4b48a748eac78270))
- EXHAUSTIVE_LEGACY_ANALYSIS.md Pass B expansion (DOC-PASS-11) ([`99eb6fb`](https://github.com/Dicklesworthstone/frankenjax/commit/99eb6fb0a9bfe8cdf0fd878ebd732afb1270848a))
- Red-team review fixes for analysis documents (DOC-PASS-12) ([`0da29b4`](https://github.com/Dicklesworthstone/frankenjax/commit/0da29b49ef5fd3f31efecc5c06942ff8cfd0f906))
- Final consistency sweep and sign-off (DOC-PASS-13) ([`c4fb516`](https://github.com/Dicklesworthstone/frankenjax/commit/c4fb516a3bbd925fa8b77d1231aa31469d5594d1))

### Dependencies

- Bump asupersync from v0.1.1 to v0.2.0 ([`5966a82`](https://github.com/Dicklesworthstone/frankenjax/commit/5966a82dfa53d7cea71b43c5ac229b9fa7b4ac29))
- Bump ftui from 0.1.1 to 0.2.0 ([`2da9509`](https://github.com/Dicklesworthstone/frankenjax/commit/2da9509b7f2c1e7fbcad39a7e86c5e3d44aa612d))
- Upgrade dependencies across all crates ([`647847c`](https://github.com/Dicklesworthstone/frankenjax/commit/647847cf97251fa09a05cdd5270e535ceb33788b))
- Workspace dependency tree refresh ([`d8915da`](https://github.com/Dicklesworthstone/frankenjax/commit/d8915dafa055d1829f9f57bd15646edd692d84be))

---

## Phase 1 — Partial Evaluation, Conformance Infrastructure, and Reliability Gates (2026-02-14 to 2026-02-15)

Partial evaluation and staging engine with bitset-indexed optimization (27% DCE + 56% PE improvement). Workspace-wide conformance infrastructure with artifact schema validation. Reliability gate reports with flake tracking. 24 new LAX primitive variants. Hardened IR transforms and trace replay engine.

### Partial Evaluation Engine (fj-interpreters, P2C-003)

Splits programs into known (pre-computable) and unknown (runtime-dependent) parts. Bitset-indexed optimization for fast known/unknown classification.

- Add partial evaluation and staging module skeletons ([`5059e77`](https://github.com/Dicklesworthstone/frankenjax/commit/5059e77da3eefd5dfdd7d414e0993f00dab3e622))
- Bitset-indexed PE with fast-path shortcuts: 27% DCE + 56% PE improvement ([`c1d15a1`](https://github.com/Dicklesworthstone/frankenjax/commit/c1d15a106f80f6a21b81f1081511f53f28d87bbe))
- P2C-003 partial eval evidence pack with 86-test conformance ([`f29ce35`](https://github.com/Dicklesworthstone/frankenjax/commit/f29ce35796dc2704ad40e9ec847af5a8d684bdae))
- Legacy anchor map ([`10f5752`](https://github.com/Dicklesworthstone/frankenjax/commit/10f5752a25eeb3a2b8e4cb52d747ad99896e1443)), contract table ([`4db4e01`](https://github.com/Dicklesworthstone/frankenjax/commit/4db4e01b801d12c6fdb404042acf20ae1b0444b7)), security threat matrix ([`5319a72`](https://github.com/Dicklesworthstone/frankenjax/commit/5319a72b0cf956549716e49a9942b6562651fada))
- Comprehensive PE/staging tests ([`6d98a4e`](https://github.com/Dicklesworthstone/frankenjax/commit/6d98a4efb85209de8fc052a3b703b35c8f3eb5d3)), differential oracle tests ([`05c0ef9`](https://github.com/Dicklesworthstone/frankenjax/commit/05c0ef9619657298d9d12f228c3694d86f35b7ce)), 6 E2E scenarios ([`bf48e80`](https://github.com/Dicklesworthstone/frankenjax/commit/bf48e809953357526301e4029b5f27bb3736b8d0))

### Conformance Infrastructure

- Add artifact schema validation and workspace-wide conformance infrastructure ([`0f035fa`](https://github.com/Dicklesworthstone/frankenjax/commit/0f035fa31f18bb9a156073667d7fa634b24a1f5b))
- Expand conformance harness, core transforms, and legacy parity infrastructure ([`d526f79`](https://github.com/Dicklesworthstone/frankenjax/commit/d526f797a22549628c8bf4c8662c270a395077d8))
- Add reliability gate reports, flake tracking, and quality gate enforcement ([`02ec12a`](https://github.com/Dicklesworthstone/frankenjax/commit/02ec12a78c9c19a26e2bde43d50680e4599eb922))
- Conformance lifecycle, error taxonomy, security, and closure crosswalk passes ([`5fbe325`](https://github.com/Dicklesworthstone/frankenjax/commit/5fbe32596ed5f918338dfcec87a2db045e1e191a))

### LAX Primitives and Transforms

- Add 24 new LAX primitive variants with eval kernels, AD rules, and e-graph support ([`a60189f`](https://github.com/Dicklesworthstone/frankenjax/commit/a60189f2104c654f6ea48eef0d937a5d1284e367))
- Add hardened IR transforms, trace replay engine, reliability budgets, and 8 new E2E tests ([`2769e57`](https://github.com/Dicklesworthstone/frankenjax/commit/2769e5764612a23c3515bf501ddadae87e8c7f91))
- Refactor staging output semantics and update quality gate infrastructure ([`22376bf`](https://github.com/Dicklesworthstone/frankenjax/commit/22376bf5c739b34a1c86fed10f23ede2eed516a0))

---

## Phase 0 — Foundation (2026-02-13)

Initial commit establishing the FrankenJAX workspace. Canonical Jaxpr IR with core primitive set. Automatic differentiation engine with VJP rules. E-graph optimizer using the `egg` library for equality saturation. Conformance harness skeleton. Conformal prediction infrastructure.

### Core

- Initial commit: FrankenJAX foundation (M0 complete) -- Jaxpr IR, core primitives, AD engine, e-graph optimizer, conformance harness ([`d5bb53f`](https://github.com/Dicklesworthstone/frankenjax/commit/d5bb53f18153040a9457ea50e787f237e6330d8c))
- AD, e-graph, conformal prediction, and performance optimizations ([`eccbee0`](https://github.com/Dicklesworthstone/frankenjax/commit/eccbee0c8c9adb9d7c1244d250df6af87ca2157a))

### Bug Fixes

- Correct round-trip variable mapping after equality saturation reordering ([`e94df54`](https://github.com/Dicklesworthstone/frankenjax/commit/e94df54ecea8e5de688a68841c416a49886ad64d))

---

## Capability Summary

| Capability | Phase Introduced | Current State |
|-----------|-----------------|---------------|
| Jaxpr IR (canonical intermediate representation) | 0 | 110 primitives, 11 dtypes |
| Automatic Differentiation (VJP reverse-mode) | 0 | All 110 primitives |
| Automatic Differentiation (JVP forward-mode) | 2 | All 110 primitives |
| E-graph optimizer (equality saturation) | 0 | 87 algebraic rewrite rules |
| Partial evaluation and staging | 1 | Bitset-indexed with 27% DCE improvement |
| API surface (`jit`, `grad`, `vmap`) | 2 | Broad composition support; advanced gaps tracked |
| Dispatch/effects runtime | 2 | Transform stack with effect tokens |
| Compilation cache (SHA-256 keying) | 2 | Strict/hardened modes, LRU eviction |
| CPU backend (parallel) | 2 | Dependency-wave scheduling via Rayon |
| C FFI surface | 2 | Only `unsafe` crate in workspace |
| Numerical AD verification | 3 | 22 unary + 7 binary + composition tests |
| Typed partial evaluation | 3 | Axis-aware reductions, Dot, shape ops |
| Vmap (BatchTrace) | 4 | in_axes/out_axes, per-primitive batch rules; advanced iterative compositions tracked |
| Control flow (cond/scan/while/fori/switch) | 4 | VJP + JVP support with remaining nested-composition gaps tracked |
| ThreeFry2x32 PRNG | 4 | JAX-matched determinism, 25 oracle fixtures |
| Complex numbers (Complex64/Complex128) | 4 | Full arithmetic + AD + e-graph rules |
| Unsigned integers (U32/U64) | 5 | Full stack support |
| Half-precision floats (BF16/F16) | 5 | Full stack support |
| Jacobian and Hessian | 5 | Via composable AD |
| `value_and_grad` (shared forward pass) | 5 | Single-pass optimization |
| Custom VJP registration | 5 | Thread-safe global registry |
| Linear algebra (Cholesky/QR/SVD/Eigh/TriSolve) | 7 | Eval + VJP + JVP, numerically verified |
| FFT (Fft/Ifft/Rfft/Irfft) | 7 | Eval + VJP + JVP |
| Oracle conformance | 1 | 848 JAX 0.9.2 fixture cases |
| RaptorQ durability | 2 | Erasure-coded sidecars for current evidence bundles; full long-lived coverage tracked |

---

## Timeline

| Phase | Date Range | Commits | Highlight |
|-------|-----------|---------|-----------|
| 0 | 2026-02-13 | 3 | Foundation: Jaxpr IR, AD, e-graph |
| 1 | 2026-02-14 to 02-15 | 16 | Partial evaluation, conformance infrastructure |
| 2 | 2026-02-20 | 77 | P2C build-out: 5 subsystem crates, documentation |
| 3 | 2026-02-21 to 02-25 | 28 | AD hardening, e-graph rules, typed PE, new primitives |
| 4 | 2026-02-25 to 02-26 | 17 | Vmap, control flow, PRNG, Complex types |
| 5 | 2026-03-02 to 03-04 | 22 | Type expansion, Jacobian/Hessian, parallel backend |
| 6 | 2026-03-04 to 03-12 | 8 | E-graph dispatch, ScalarBool, durability proofs |
| 7 | 2026-03-12 to 03-17 | 49 | 834 oracle fixtures, linalg/FFT AD, README rewrite |
| 8 | 2026-03-18 to 05-01 | ongoing | 848 oracle fixtures, performance slices, reality-check gap tracking |

---

*This changelog covers continuous development on `main`. FrankenJAX has no formal releases, tags, or semver milestones; all development is on the `main` branch at version 0.1.0.*
