# Changelog

All notable changes to [FrankenJAX](https://github.com/Dicklesworthstone/frankenjax) are documented in this file.

FrankenJAX is a clean-room Rust reimplementation of JAX's transform semantics. This project has no formal releases or tags; the changelog is organized by development phase, derived from the full commit history.

---

## [Unreleased] — HEAD

Latest commit: [`0e62ff4`](https://github.com/Dicklesworthstone/frankenjax/commit/0e62ff4693a476edbdd2444279b8cd91c0feadc2) (2026-03-17)

Current state: 110 primitive operations, full VJP + JVP AD coverage, 834 JAX oracle fixtures, 87 e-graph rewrite rules, 1,724 tests passing, 15 workspace crates, ~80,872 lines of Rust.

---

## Phase 7 — Oracle Conformance, Linalg/FFT AD, and Documentation (2026-03-12 to 2026-03-17)

Massive conformance expansion from 457 to 834 oracle fixture cases. Full linear algebra and FFT autodiff with numerical verification. README rewrite documenting all 110 primitives.

### Conformance

- Re-capture all oracle fixtures from JAX 0.9.2 with fixed PRNG key extraction ([`4fc04bc`](https://github.com/Dicklesworthstone/frankenjax/commit/4fc04bc933cd44de243bf9de13a19ec5a37cc32c))
- Add dtype promotion oracle fixture with 162 JAX-verified cases ([`9ba620e`](https://github.com/Dicklesworthstone/frankenjax/commit/9ba620e2922d5ddadec219f45de7c18d520f92fc))
- Add transform composition oracle fixture and parity test ([`5e703dd`](https://github.com/Dicklesworthstone/frankenjax/commit/5e703ddcaecb1739d42c0b3960fc88969b3eab51))
- Add linalg/FFT oracle parity fixture bundle and test runner ([`03d60ec`](https://github.com/Dicklesworthstone/frankenjax/commit/03d60ec4403d880ab3da72e677ff29737596341d))
- Add numerical gradient verification for AD VJP rules ([`22c02c4`](https://github.com/Dicklesworthstone/frankenjax/commit/22c02c4067d61c22b92283fb09f7f17858d53b78))
- Add e-graph optimization semantics-preserving gate ([`9d411d0`](https://github.com/Dicklesworthstone/frankenjax/commit/9d411d0654d092824adb644a7ae8cd6e5b4caeee))
- Add durability coverage tests and mark RaptorQ pipeline parity_green ([`0cd8363`](https://github.com/Dicklesworthstone/frankenjax/commit/0cd83630d9e6bdb1c0571d5963f309d46b5dfd05))
- Add conformance fixtures for 22 additional primitives reaching 596/596 parity ([`92250d6`](https://github.com/Dicklesworthstone/frankenjax/commit/92250d6927f92ec1f13e4f451214ac69b5ccd958))
- Add conformance fixtures for structural primitives (Iota, Copy, ExpandDims, Pad, BroadcastInDim) ([`981fd11`](https://github.com/Dicklesworthstone/frankenjax/commit/981fd117d44b90510b18df15ab8b8d6e69515241))
- Add conformance fixtures for shape manipulation primitives ([`09d71e8`](https://github.com/Dicklesworthstone/frankenjax/commit/09d71e89b5a6f943dbcd9432bb2c9257c23cff12))
- Add conformance fixtures for sort, integer intrinsics, and reduction primitives ([`67d4ad0`](https://github.com/Dicklesworthstone/frankenjax/commit/67d4ad022f4b1e3a0766b127fc4d16c3f628cf7f))
- Expand lax conformance to special math, cumulative, and bitwise primitives ([`2f1ec76`](https://github.com/Dicklesworthstone/frankenjax/commit/2f1ec764932eb0de07d934a8d84cf1b31ad08aae))
- Add mixed_dtype conformance family for cross-dtype type promotion (457/457 pass) ([`3daf6fb`](https://github.com/Dicklesworthstone/frankenjax/commit/3daf6fb39a6a3c565eb8ac429133cdf9818b589b))
- Add complex number primitives and expand conformance capture tooling ([`c857a13`](https://github.com/Dicklesworthstone/frankenjax/commit/c857a13173722a3a419f1e67116986906c4f5f3d))
- Improve multirank conformance runner and test infrastructure ([`06f7ff3`](https://github.com/Dicklesworthstone/frankenjax/commit/06f7ff36f4ac5da610453d657b0d9aff9ff2aa6f))
- Add transform API unit tests and JVP numerical verification ([`b5b0958`](https://github.com/Dicklesworthstone/frankenjax/commit/b5b0958984fb30704896d8d927c86700ca96e1b0))

### Linear Algebra

- Implement Cholesky, TriangularSolve, QR, SVD, and Eigh primitive evaluation kernels ([`f58dc19`](https://github.com/Dicklesworthstone/frankenjax/commit/f58dc19d2d335676ebafe504186ef75cd9bf7285))
- Add multi-output VJP support for QR, SVD, Eigh decompositions ([`a07bc7b`](https://github.com/Dicklesworthstone/frankenjax/commit/a07bc7b3a71edaa59a9fb8898e3d87071e50d847))
- Add higher-order VJP rules for triangular solve, Cholesky, and einsum ([`bd3db9e`](https://github.com/Dicklesworthstone/frankenjax/commit/bd3db9e782532b9b38955a59ffc67c6398840288))
- Implement JVP rules for multi-output decompositions (Eigh, QR, SVD) ([`e6eb996`](https://github.com/Dicklesworthstone/frankenjax/commit/e6eb996d729953ade8c50397ba32c04f25fbeacb))

### FFT

- Implement FFT primitives replacing V2 stubs in fj-lax ([`ee2c3bd`](https://github.com/Dicklesworthstone/frankenjax/commit/ee2c3bd39ff550627a7fb25ed96dd65e5b750330))
- Implement FFT/IFFT autodiff rules and harden FFT eval ([`4fc3062`](https://github.com/Dicklesworthstone/frankenjax/commit/4fc3062533b2290dfad142bfae1ab4f3c8a2f4e3))
- Implement RFFT/IRFFT autodiff rules and add TensorComplex128 conformance support ([`d970e4f`](https://github.com/Dicklesworthstone/frankenjax/commit/d970e4f3e122313c0f01580ed7fcc6b5ac6f2eab))

### Bug Fixes

- Correct Cholesky VJP diagonal factor and back-substitution direction ([`6e79009`](https://github.com/Dicklesworthstone/frankenjax/commit/6e7900954c726a3b6de4a932147555f0aafb2153))
- Correct Cholesky JVP middle matrix solve direction ([`59fe202`](https://github.com/Dicklesworthstone/frankenjax/commit/59fe202a7d6517c9b35220ae0da3b5f58defcaca))
- Remove unused SVD VJP transpose and update comment ([`b460a19`](https://github.com/Dicklesworthstone/frankenjax/commit/b460a1947c432bc9f691b94e87db319a1f6c2500))
- Remove dead code, use `slice::from_ref`, and fix clippy lints ([`260b4ce`](https://github.com/Dicklesworthstone/frankenjax/commit/260b4ceaea11684a102a07b3c32bb82fa8744923))
- Use `slice::from_ref` instead of clone in tensor_ops tests ([`730edae`](https://github.com/Dicklesworthstone/frankenjax/commit/730edae8b98992d1c3a4e849c9a2e6d06d5a387f))

### Tests

- Add comprehensive unit tests for arithmetic, comparison, and reduction ops ([`8158c37`](https://github.com/Dicklesworthstone/frankenjax/commit/8158c371a5c53a95bb1e156514c3d025abdc6cb6))
- Add unit tests for tensor reshape, transpose, slice, pad, concatenate, and broadcast ([`7609ac2`](https://github.com/Dicklesworthstone/frankenjax/commit/7609ac2d9696b43d12262bdc25b331170dc259bf))

### Documentation

- Rewrite README with 110 primitives, oracle conformance, and AD verification ([`9e6ecee`](https://github.com/Dicklesworthstone/frankenjax/commit/9e6eceea8155ebd719d1b4ec7b8d9da0f2f97208))
- Expand README with transform composition, conformance fixtures, and performance data ([`aa6fbf4`](https://github.com/Dicklesworthstone/frankenjax/commit/aa6fbf44d36656594c8ef34152cb1789fcba09af))
- Update parity to 110 ops and add linalg/FFT oracle tests ([`f7d8fcd`](https://github.com/Dicklesworthstone/frankenjax/commit/f7d8fcd6ce79bb2fa46fce51aa767698f9be57aa))
- Simplify Cholesky VJP step 3 comments to match corrected implementation ([`e65d143`](https://github.com/Dicklesworthstone/frankenjax/commit/e65d1430d6d17bd382c68b7f4bc9d574c03fd89d))
- Update e-graph feature parity entry with conformance test coverage ([`44f4459`](https://github.com/Dicklesworthstone/frankenjax/commit/44f4459b719701dfa8c22eeb6b4f87ef1de61b7c))
- Replace Unicode combining macron with plain g in README gradient examples ([`0e62ff4`](https://github.com/Dicklesworthstone/frankenjax/commit/0e62ff4693a476edbdd2444279b8cd91c0feadc2))

---

## Phase 6 — Control Flow, ScalarBool, and Durability Proofs (2026-03-04 to 2026-03-05)

E-graph rewriting integrated into dispatch pipeline. Control flow primitives (cond, scan) added with RNG parity verification. ScalarBool type support. RaptorQ durability proof artifacts generated.

### Features

- Integrate e-graph rewriting into dispatch and expand conformance ([`1d5fac7`](https://github.com/Dicklesworthstone/frankenjax/commit/1d5fac78d47099129b01bb862c14d6bdf210b91f))
- Add control flow primitives (cond, scan) and RNG parity verification ([`aeb4648`](https://github.com/Dicklesworthstone/frankenjax/commit/aeb4648249f3da322caf90f9350ea07dd9365f23))
- Add ScalarBool support, expand AD gradient coverage, and generate durability proof artifacts ([`7026442`](https://github.com/Dicklesworthstone/frankenjax/commit/702644231fbf4798164f2069d1e91b4d57c717ae))
- Expand AD engine, conformance suites, and core type system ([`21e435d`](https://github.com/Dicklesworthstone/frankenjax/commit/21e435da4b0e15226eaedf3500f89d57828538a1))
- Add shape inference tests and CI validation artifacts ([`6b6ff6c`](https://github.com/Dicklesworthstone/frankenjax/commit/6b6ff6c519a7866f9155ddca498c2ac7503d1aab))

---

## Phase 5 — Type System Expansion, Transforms, and Higher-Order AD (2026-03-02 to 2026-03-04)

Major type system expansion: unsigned integers (U32/U64), half-precision floats (BF16/F16), complex numbers. Full transform composition: Jacobian, Hessian, value_and_grad. Dependency-wave parallel CPU backend.

### Type System

- Add unsigned integers (U32/U64), half-precision floats (BF16/F16), and complex number primitives across core, lax, ad, and ffi ([`563285e`](https://github.com/Dicklesworthstone/frankenjax/commit/563285e451080277b89f85a493986193fe2178f5))
- Complete BF16/F16 half-precision support across all crates ([`1e688cd`](https://github.com/Dicklesworthstone/frankenjax/commit/1e688cd8dba71389c8e04cf0755eca87ee527e38))
- Add ReduceAnd, ReduceOr, ReduceXor bitwise reduction primitives ([`b2faf2d`](https://github.com/Dicklesworthstone/frankenjax/commit/b2faf2d5f2f8efb7351824d3c73db2d4eea7b60a))
- Split ShiftRight into ShiftRightArithmetic and ShiftRightLogical ([`3f10471`](https://github.com/Dicklesworthstone/frankenjax/commit/3f10471a4f6bb67e3ae6d3b8c3d130f772d065e7))
- Add effects field to Jaxpr/Equation IR and value_and_grad API ([`42026a6`](https://github.com/Dicklesworthstone/frankenjax/commit/42026a6b5972680fe68bdcd74acceb93c4d79bb6))

### Transforms and API

- Implement Jacobian and Hessian matrix computation ([`4032637`](https://github.com/Dicklesworthstone/frankenjax/commit/4032637639bbf2a8279d3423ab802923b7c41929))
- Unify value_and_grad into a single shared forward pass ([`e72afd0`](https://github.com/Dicklesworthstone/frankenjax/commit/e72afd00d334886c600201fb3030d2e6fc2ec519))
- Expose custom VJP registration and add RNG determinism conformance suite ([`da05694`](https://github.com/Dicklesworthstone/frankenjax/commit/da05694efb83f8933e517d4ae0709782e308c7bf))
- Nested trace context infrastructure and BF16/F16 dtype support ([`3304b13`](https://github.com/Dicklesworthstone/frankenjax/commit/3304b13d222a688d4ffc6cf09822721304a8f03e))
- Add differentiation, batching, and tracing for BroadcastedIota, Copy, BitcastConvertType, and ReducePrecision ([`5fb6add`](https://github.com/Dicklesworthstone/frankenjax/commit/5fb6add8f5974760f172f7be73875243ce88038d))

### Backend

- Replace sequential interpreter with dependency-wave parallel executor in fj-backend-cpu ([`92381c8`](https://github.com/Dicklesworthstone/frankenjax/commit/92381c84d250bb7cfc2c88063cd6bc9d6b4218f0))
- Dedicated batch rules for control flow and indexing ops ([`ec39fa8`](https://github.com/Dicklesworthstone/frankenjax/commit/ec39fa867f3209cd7010c5a157de5a2ea653e030))
- Add control flow conformance suite and v1 parity golden data ([`7ff370d`](https://github.com/Dicklesworthstone/frankenjax/commit/7ff370dd8f742be6beebed8ce1950a8fee3ff383))

### Bug Fixes

- Symmetric subgradient tie-averaging in Sort VJP ([`596b9c2`](https://github.com/Dicklesworthstone/frankenjax/commit/596b9c22667a753845dbab6a4e1ce13b680f3ec6))

### Performance

- Add value_and_grad shared-forward-pass vs separate-calls benchmarks ([`204314b`](https://github.com/Dicklesworthstone/frankenjax/commit/204314b62f8b66a993e7a3562772b365e2e7522e))

### Documentation

- Overhaul feature parity matrix, expand security threat model, update README ([`7e8bcaa`](https://github.com/Dicklesworthstone/frankenjax/commit/7e8bcaa86527f9b3797e97c6aa1f7067f354093f))

---

## Phase 4 — Vmap, Control Flow, and Primitive Expansion (2026-02-25 to 2026-02-26)

BatchTrace interpreter for vmap with in_axes/out_axes. Control flow VJPs (cond, scan, while_loop, switch). New primitives: Rev, Squeeze, Split, ExpandDims, Cbrt, IsFinite, IntegerPow, Nextafter. ThreeFry2x32 PRNG. Complex64/Complex128 dtype support.

### Vmap

- Implement BatchTrace interpreter and fix ReduceWindow VJP ([`74eeec3`](https://github.com/Dicklesworthstone/frankenjax/commit/74eeec3009b83651edb2c3dba74b50c894e0e504))
- Remove scalar-only output constraint and fix BatchTrace broadcasting ([`45297fd`](https://github.com/Dicklesworthstone/frankenjax/commit/45297fddd6fbb1c0ba91a13c974f25a5c3da9859))
- Add in_axes/out_axes to vmap and rewrite while_loop with functional API ([`3a7dcb5`](https://github.com/Dicklesworthstone/frankenjax/commit/3a7dcb56afb63ec570926fe48cc9d5c536faae01))
- Add per-dtype tolerance tiers with violation reporting and regenerate test artifacts ([`4a4986c`](https://github.com/Dicklesworthstone/frankenjax/commit/4a4986cca06eae59908ee8e02d4e7110084245c8))

### Primitives and Control Flow

- Add new primitives: Rev, Squeeze, Split, ExpandDims, Cbrt, IsFinite, IntegerPow, Nextafter; tensor-valued JVP; make_jaxpr API; PRNG sampling functions ([`074194b`](https://github.com/Dicklesworthstone/frankenjax/commit/074194b5e244ec917a0d814031b7ef12fc0ced45))
- Add Switch primitive with VJP support and fori_loop evaluator ([`845001e`](https://github.com/Dicklesworthstone/frankenjax/commit/845001e092c0da4acf0a1f16a28333bbd5f10b22))
- Add JVP (forward-mode) rule for Switch primitive ([`0aa23e9`](https://github.com/Dicklesworthstone/frankenjax/commit/0aa23e9b6c6ac500dd604e135b1b04ce167c06ed))
- Add functional scan API, ThreeFry2x32 PRNG, and vmap in_axes/out_axes tests ([`da8cb1a`](https://github.com/Dicklesworthstone/frankenjax/commit/da8cb1a1137b7d34de85eb11a84c0403a1b4d8d0))
- Add Complex64 and Complex128 to DType and Literal enums ([`a8f7859`](https://github.com/Dicklesworthstone/frankenjax/commit/a8f78596b79a065e918aa40156ffddf4d2c294dd))
- Implement control flow VJPs, bitwise ops, and trace compilation ([`e182573`](https://github.com/Dicklesworthstone/frankenjax/commit/e182573114c5f96a411bfaa9088b89d6450b04e6))
- Implement full VJP suite for remaining primitives and expand LAX/tensor ops ([`c064058`](https://github.com/Dicklesworthstone/frankenjax/commit/c06405883734e9b93ba5ee43ba6d8f2ea81a9402))
- Implement pad/gather/scatter VJPs, expand LAX tensor ops, and add trace compilation ([`cb4d4a3`](https://github.com/Dicklesworthstone/frankenjax/commit/cb4d4a3262304ee85622b1dc053aecf55a4ba2f2))
- Rewrite oracle capture script with lax primitive coverage ([`7612b3c`](https://github.com/Dicklesworthstone/frankenjax/commit/7612b3c182f498ad6b579ce2e3d2021dd7eecf0a))

### Bug Fixes

- Add sub_jaxprs field to Equation structs and extend primitive arity coverage ([`02d1e5c`](https://github.com/Dicklesworthstone/frankenjax/commit/02d1e5cb832b272cd99f20d5990f2d9d4c04c317))
- Rustfmt formatting in sort VJP and add conv2d kernel rank validation ([`7855d5e`](https://github.com/Dicklesworthstone/frankenjax/commit/7855d5effc7739473f45e32272d5df9564d0ad60))
- Add rank-3 tensor tests and fix pad expected dimensions ([`0f0d540`](https://github.com/Dicklesworthstone/frankenjax/commit/0f0d540719097b3bc39b69db6e945f9c9f80c943))

---

## Phase 3 — AD Hardening, E-graph Rules, and Numerical Verification (2026-02-21 to 2026-02-25)

Extensive AD correctness hardening with numerical gradient verification. E-graph algebraic rewrite rules for trig/hyperbolic identities. Typed partial evaluation. BackendRegistry with device-aware fallback. Pad primitive. License update.

### AD Hardening

- Add systematic numerical gradient verification for 22 unary VJP rules ([`dbce07c`](https://github.com/Dicklesworthstone/frankenjax/commit/dbce07c0862c5d42c8d272e164ca24ab29dd6803))
- Add binary VJP numerical gradient verification for Add, Sub, Mul, Div, Pow, Atan2, Rem ([`4620618`](https://github.com/Dicklesworthstone/frankenjax/commit/4620618b6f5c4120be8b87663448a53e846a2930))
- Add composition gradient tests: exp(sin), sin\*cos, log(x^2+1), tanh(2x) ([`f75dcdc`](https://github.com/Dicklesworthstone/frankenjax/commit/f75dcdc693e564878a47a5a8e2663a184e61d411))
- Add Max/Min VJP tie-breaking tests and higher-order gradient tests ([`baab370`](https://github.com/Dicklesworthstone/frankenjax/commit/baab37056d614ca68de33ef134935080825d04fc))
- Differentiate scatter backward pass by mode: pass-through gradient for additive scatter ([`a5457e9`](https://github.com/Dicklesworthstone/frankenjax/commit/a5457e9144728fd9a5f0766579b7d5c190b373cf))
- Add axis-aware gradient broadcasting for partial reductions in VJP backward pass ([`594289a`](https://github.com/Dicklesworthstone/frankenjax/commit/594289a403541389b4f9272e86e84a4e2eaa8a13))
- Validate scatter VJP indices are non-negative and finite before usize cast ([`21c8afc`](https://github.com/Dicklesworthstone/frankenjax/commit/21c8afc784f28738e4801d3371fdc506d0bbb363))

### E-graph

- Add e-graph algebraic rewrite rules for trig/hyperbolic identities, reciprocal involution, and nested select simplification ([`e282c93`](https://github.com/Dicklesworthstone/frankenjax/commit/e282c931a5b1ef127615a6e667361e4f13d9d211))
- Harden gather/scatter validation, add e-graph algebraic rules, and expand edge-case test coverage ([`38b51d2`](https://github.com/Dicklesworthstone/frankenjax/commit/38b51d289e9a6bf9740ccf27ae73ebe0813fd662))

### Primitives

- Add Pad primitive with edge and interior padding support ([`4f3ed85`](https://github.com/Dicklesworthstone/frankenjax/commit/4f3ed853197f6c08d235d91dce56f725b8bf8d9f))
- Add Clamp/DynamicSlice/Iota primitives and wire dispatch through backend ([`5f41d20`](https://github.com/Dicklesworthstone/frankenjax/commit/5f41d20dc9f25ba448b55a977119c8b580c31299))
- Add DynamicSlice/Clamp/Iota support to trace, egraph, and PE type inference ([`e5b4b3a`](https://github.com/Dicklesworthstone/frankenjax/commit/e5b4b3ace992559af913042931682e79f7925e0b))
- Add Value::as_i64_scalar, as_bool_scalar, dtype() and TensorValue::to_i64_vec ([`9b2568e`](https://github.com/Dicklesworthstone/frankenjax/commit/9b2568ec36f78d64faecd303c63b374426782825))

### Interpreters

- Add typed partial evaluation to preserve residual abstract value types ([`dc2e311`](https://github.com/Dicklesworthstone/frankenjax/commit/dc2e311c08dbf406ef899175af671d21b027517f))
- Extend PE type inference for axis-aware reductions, Dot, and shape ops ([`a9795a1`](https://github.com/Dicklesworthstone/frankenjax/commit/a9795a1f5a5019bf46ce83033912dc9f871ea314))

### Backend

- Refactor dispatch to use BackendRegistry with device-aware fallback ([`9856878`](https://github.com/Dicklesworthstone/frankenjax/commit/9856878a14688b360970c77742a1836fd6ad5c38))
- Add BackendExecution error variant mapping to ApiError ([`e079973`](https://github.com/Dicklesworthstone/frankenjax/commit/e07997389ccd516f1d19ed35830f2201a094c055))

### Tests

- Add scatter mode="add" accumulation tests ([`1322ea0`](https://github.com/Dicklesworthstone/frankenjax/commit/1322ea0a7d92ac8941bd84832db68dc98673bd31))
- Strengthen Scatter validation and add Scatter/Gather trace tests ([`9263c24`](https://github.com/Dicklesworthstone/frankenjax/commit/9263c2411cab1b98984c68ee2e8b58049b405caa))
- Add higher-rank tensor and transform composition dispatch tests ([`9f65903`](https://github.com/Dicklesworthstone/frankenjax/commit/9f65903e3b0e1ee3e087355b24a0268da2303bcd))

### Bug Fixes

- Fix axis-aware ReduceMax/ReduceMin VJP, prevent TTL cache memory leak, and materialize dangling e-graph literals ([`f74e223`](https://github.com/Dicklesworthstone/frankenjax/commit/f74e2236f2c33a87b8db3c39c993ff7c7f206758))
- Fix compilation errors, harden AD VJP rules, and add Select broadcasting for JAX parity ([`cfd2882`](https://github.com/Dicklesworthstone/frankenjax/commit/cfd2882bbf8b812aec6cf3e788db382f10f66faa))
- Fix multi-dimensional VJP slice adjoint, harden dtype promotion in select ([`a890719`](https://github.com/Dicklesworthstone/frankenjax/commit/a890719acfab3c90514c3cd4bf2a5ff679247d60))
- Fix AD edge cases, add constvar support to e-graph, overhaul dtype promotion, and harden LAX ops ([`9495a63`](https://github.com/Dicklesworthstone/frankenjax/commit/9495a6339845cdcc982b027aa77d5a05f8c15c3c))

### Chore

- Update license to MIT with OpenAI/Anthropic Rider ([`ebe06eb`](https://github.com/Dicklesworthstone/frankenjax/commit/ebe06eb1d7c629fcadb647d1108a2d8a9614c7ed))
- Switch asupersync + ftui from local paths to crates.io ([`0cbfe47`](https://github.com/Dicklesworthstone/frankenjax/commit/0cbfe47eeab94e5c48a0a7d6390345526fe356e2))

---

## Phase 2 — P2C Infrastructure Build-out (2026-02-20)

Systematic Phase-2C build-out: API transform front-door (fj-api), dispatch/effects runtime, compilation cache/keying, backend bridge, FFI call interface, and LAX primitive first wave. Each subsystem follows the same rigorous process: legacy anchor map, contract table, security threat model, implementation, unit/property tests, differential oracle tests, E2E scenarios, performance benchmarks, and final evidence pack with RaptorQ durability sidecars.

### API Transform Front-Door (P2C-002)

- Legacy anchor map for API transform front-door ([`060500f`](https://github.com/Dicklesworthstone/frankenjax/commit/060500f4230b35a8f56a6b9225bb69ed38bc3868))
- Contract table for API transform front-door ([`1778eeb`](https://github.com/Dicklesworthstone/frankenjax/commit/1778eebf1ebf9ca62b0c617c721317a1fd065aef))
- Security + compatibility threat model ([`ed6a304`](https://github.com/Dicklesworthstone/frankenjax/commit/ed6a304e02c45189ed30ea76184f6b7e446cc487))
- fj-api crate skeleton + implementation plan ([`441c257`](https://github.com/Dicklesworthstone/frankenjax/commit/441c2579aadc36a70984b2d8387f49651d523cf1))
- Implement API transform front-door in Rust ([`58dbffa`](https://github.com/Dicklesworthstone/frankenjax/commit/58dbffafe61b28ba59bc20bfb43d309e1ceb372d))
- Unit + property tests with structured logging ([`d79c221`](https://github.com/Dicklesworthstone/frankenjax/commit/d79c221abec88c38d7a5d85bdccca59aa9881ec4))
- Differential oracle + metamorphic + adversarial validation ([`23c3ae3`](https://github.com/Dicklesworthstone/frankenjax/commit/23c3ae39259db1d47aa0a74c3a1c0ab8c948a458))
- E2E scenario scripts + replay/forensics logging ([`19df570`](https://github.com/Dicklesworthstone/frankenjax/commit/19df5707136130dcc00e7e708e29fca0cb5d3184))
- Profile-driven optimization + isomorphism proof ([`7c04cf4`](https://github.com/Dicklesworthstone/frankenjax/commit/7c04cf4d862e21b4ccb63905dad8cf377aabe5f6))
- Final evidence pack with parity gates + RaptorQ sidecars ([`75e07d9`](https://github.com/Dicklesworthstone/frankenjax/commit/75e07d98ac20e29f9570b0bba12f1914bec60ee7))

### Dispatch/Effects Runtime (P2C-004)

- Legacy anchor map for dispatch/effects runtime ([`9831a15`](https://github.com/Dicklesworthstone/frankenjax/commit/9831a1586d908056acdc317825adbede50ee9ff5))
- Contract table with strict/hardened invariant spec ([`80b14b5`](https://github.com/Dicklesworthstone/frankenjax/commit/80b14b5455c531a442dcafdb91de4f13f4a0ff38))
- Security + compatibility threat model for dispatch/effects runtime ([`32f7ea8`](https://github.com/Dicklesworthstone/frankenjax/commit/32f7ea8d99100979fa9ad0d957a3f928d126d054))
- Rust implementation plan + module boundary skeleton ([`9196dc4`](https://github.com/Dicklesworthstone/frankenjax/commit/9196dc4f29b93e94199667e5616c13f7c36eae3b))
- Implement forward-mode JVP + effect token system ([`d719da5`](https://github.com/Dicklesworthstone/frankenjax/commit/d719da52f0b2e13b7537c698ced608105b927a59))
- Comprehensive unit + property test suite for dispatch runtime ([`a1c6356`](https://github.com/Dicklesworthstone/frankenjax/commit/a1c6356ae2d0ccd10616722dd2452b424ab6e61f))
- Differential oracle + metamorphic + adversarial validation ([`649defd`](https://github.com/Dicklesworthstone/frankenjax/commit/649defd7e01317857e280b9e195d6a71d7ac243f))
- E2E scenario scripts with forensic replay logging ([`7346769`](https://github.com/Dicklesworthstone/frankenjax/commit/73467696940ebebec53bd5dc025b71d0e4d36836))
- Dispatch optimization: iterative Jit-skip + Vec effect tokens ([`f3f7523`](https://github.com/Dicklesworthstone/frankenjax/commit/f3f75231521e6a9ccd9d5abd7779ffb19e40cc91))
- Final evidence pack with parity gates + risk note + RaptorQ sidecars ([`a27f087`](https://github.com/Dicklesworthstone/frankenjax/commit/a27f08786aa188281c4d084c589047d44d4672b3))

### Compilation Cache/Keying (P2C-005)

- Legacy anchor map for compilation cache/keying ([`365ca3a`](https://github.com/Dicklesworthstone/frankenjax/commit/365ca3ab92be99e90e1afffdc4fe80abc7b7706c))
- Contract table with strict/hardened invariant spec ([`8dd3853`](https://github.com/Dicklesworthstone/frankenjax/commit/8dd3853721cf5a130ad1520458688be0af2402be))
- Security + compatibility threat model for compilation cache/keying ([`8c6dd47`](https://github.com/Dicklesworthstone/frankenjax/commit/8c6dd473ae3dfa1cca091092aebaa58aa87991e7))
- Cache module boundary skeleton: backend trait, LRU eviction, persistence wire format, key stability harness ([`3c575b2`](https://github.com/Dicklesworthstone/frankenjax/commit/3c575b24947730d33ddd343bd20a1a061e7445a2))
- CacheManager with persistence, eviction, and integrity verification ([`168f425`](https://github.com/Dicklesworthstone/frankenjax/commit/168f42540c4e1696de0b8bccb0ed60539797a3e7))
- Unit + property tests for cache key sensitivity, compatibility, and persistence ([`9e2559e`](https://github.com/Dicklesworthstone/frankenjax/commit/9e2559e0f5286bc05a50c64c51069eb1fb11c001))
- Differential oracle + metamorphic + adversarial validation for cache/keying ([`d91f1f2`](https://github.com/Dicklesworthstone/frankenjax/commit/d91f1f237b698c3534dc8b8d286a8b7ef1b3251c))
- E2E scenario scripts for compilation cache/keying ([`e7aee7b`](https://github.com/Dicklesworthstone/frankenjax/commit/e7aee7bfbb52cefa635ccfa7ba08582a5807994e))
- Zero-alloc streaming hasher: 32.6% faster cache key generation ([`031f3d7`](https://github.com/Dicklesworthstone/frankenjax/commit/031f3d72788e6be357cce22bf6dc6c68fe6eecc9))
- Final evidence pack with parity gates + risk note + RaptorQ sidecars ([`6b65817`](https://github.com/Dicklesworthstone/frankenjax/commit/6b65817403bd848677ec3f1d2f2f9c1bc7bedc67))

### Backend Bridge (P2C-006)

- Legacy anchor map for backend bridge and platform routing ([`d4dda74`](https://github.com/Dicklesworthstone/frankenjax/commit/d4dda74df0e1a50a64c658bbf8133950ff4981a3))
- Contract table for backend bridge and platform routing ([`119f1a2`](https://github.com/Dicklesworthstone/frankenjax/commit/119f1a2cb2b368ad789a31b5c857c833244490a9))
- Security + compatibility threat model for backend bridge ([`25f52d7`](https://github.com/Dicklesworthstone/frankenjax/commit/25f52d7742a984d8a958e9c3b0e255eaf3f57ae3))
- Backend bridge module boundary skeleton ([`000ffeb`](https://github.com/Dicklesworthstone/frankenjax/commit/000ffeb9b087e15815568c01e48103c48f986ddc))
- Backend capabilities, registry, and platform routing ([`2eb290b`](https://github.com/Dicklesworthstone/frankenjax/commit/2eb290bf3f5a1390b31ed4bafba90098420dff2b))
- Unit + property tests for backend bridge ([`5841fb3`](https://github.com/Dicklesworthstone/frankenjax/commit/5841fb3ace939087f502c9b3d4e7ee8fe45ff6e6))
- Differential oracle + metamorphic + adversarial tests ([`9108086`](https://github.com/Dicklesworthstone/frankenjax/commit/9108086a3ce288d7a7dabd9373b673b89c8a3f43))
- E2E scenario scripts for backend bridge ([`b029660`](https://github.com/Dicklesworthstone/frankenjax/commit/b0296607180dd5f9cd78bdf58efe8a18c142ace6))
- CPU backend baseline benchmarks ([`f6dabbf`](https://github.com/Dicklesworthstone/frankenjax/commit/f6dabbf42f97101275f37f80639c4934911f13bb))
- Final evidence pack for backend bridge ([`a024672`](https://github.com/Dicklesworthstone/frankenjax/commit/a024672b2474b85d5a6b1ff99d8bf7e0f04aaea3))

### FFI Call Interface (P2C-007)

- Legacy anchor map for FFI call interface ([`f0a4530`](https://github.com/Dicklesworthstone/frankenjax/commit/f0a453066df0ea64d357e17331680a811d3d5bf0))
- Contract table for FFI call interface ([`3b64b35`](https://github.com/Dicklesworthstone/frankenjax/commit/3b64b35d88d090b97b706e5ad4efe7a6de748c5d))
- Security threat matrix for FFI call interface ([`acdd55c`](https://github.com/Dicklesworthstone/frankenjax/commit/acdd55c7344ccb9445be1d3ffd01f87b5034f6c5))
- fj-ffi crate skeleton with module boundaries ([`b137297`](https://github.com/Dicklesworthstone/frankenjax/commit/b137297eaec575d061bb672e55d789c7f4200011))
- FFI call interface implementation ([`631ddb6`](https://github.com/Dicklesworthstone/frankenjax/commit/631ddb6524826dd23bbc02d659567520cfe7700a))
- Unit + property tests for FFI call interface ([`7a8f05b`](https://github.com/Dicklesworthstone/frankenjax/commit/7a8f05bfec86f73271a1c730bc3e2eb34509ebe7))
- Differential oracle + metamorphic + adversarial tests ([`8786945`](https://github.com/Dicklesworthstone/frankenjax/commit/878694567f26ea5c8b6f63f3e44821cd05d2684b))
- E2E scenario scripts for FFI call interface ([`6a81c82`](https://github.com/Dicklesworthstone/frankenjax/commit/6a81c82f1fbd8f4fe6c1c1516820ab4422ebdc3a))
- FFI call interface baseline benchmarks ([`687960c`](https://github.com/Dicklesworthstone/frankenjax/commit/687960c78a4f7e1d313dac09f4af077d8c5dcb11))
- Final evidence pack for FFI call interface ([`31ce439`](https://github.com/Dicklesworthstone/frankenjax/commit/31ce43936030da8384e0cf4f9deaeb0fa8d424cf))

### LAX Primitive First Wave (P2C-008)

- Legacy anchor map for LAX primitive first wave ([`22bdf69`](https://github.com/Dicklesworthstone/frankenjax/commit/22bdf69f6ef4555093d8e901374b25e73515a27e))
- Contract table for LAX primitive first wave ([`4688be4`](https://github.com/Dicklesworthstone/frankenjax/commit/4688be46176a3aaff4ea31021b25673e9b9d67e4))
- Security threat matrix for LAX primitive first wave ([`3a22d3a`](https://github.com/Dicklesworthstone/frankenjax/commit/3a22d3aa21dd029567556dc12edf3ab9b70abbe6))
- Refactor fj-lax into module boundary skeleton ([`5e03cb3`](https://github.com/Dicklesworthstone/frankenjax/commit/5e03cb3b35dfa4c5f12e0376ffdc690321ce272f))
- Comprehensive unit + property tests for LAX primitives ([`accbf8a`](https://github.com/Dicklesworthstone/frankenjax/commit/accbf8a55a18bbd380354f733f2367e4b1871982))
- Differential oracle + metamorphic + adversarial for LAX primitives ([`3a07cdc`](https://github.com/Dicklesworthstone/frankenjax/commit/3a07cdc2145a6794148b903592cde223c6e9fae8))
- E2E scenario scripts for LAX primitive first wave ([`68fa108`](https://github.com/Dicklesworthstone/frankenjax/commit/68fa10862a2b3ff452e9452e778e62d4d2f8f21b))
- LAX primitive baseline benchmarks ([`7cb1acd`](https://github.com/Dicklesworthstone/frankenjax/commit/7cb1acd7c1daa22232fdf7b8a65e0498755e7873))

### Cross-Cutting

- Add fj-backend-cpu and fj-ffi workspace crates, refresh test artifacts ([`1663bf2`](https://github.com/Dicklesworthstone/frankenjax/commit/1663bf227880717fe4c86279ea6fd1b8274ce7fc))
- Refresh test artifacts and add backend/FFI conformance oracle tests ([`255b849`](https://github.com/Dicklesworthstone/frankenjax/commit/255b84931c0b3316dcc3f31dc8b07fe782cb4b3c))
- Phase-2C readiness drill and sign-off ([`683374d`](https://github.com/Dicklesworthstone/frankenjax/commit/683374ddec34ae201e26876e6bc82244d2fd42c7))
- Expand LAX primitive set with tensor-aware AD, new math ops, axis-aware reductions, and select ([`841ea88`](https://github.com/Dicklesworthstone/frankenjax/commit/841ea889cec6da82ed0c2bb0dacad16c8e1f1770))
- Broad improvements to AD, staging, dispatch, and conformance suite ([`31595e0`](https://github.com/Dicklesworthstone/frankenjax/commit/31595e0f5c1a53541efd3a42d3065e872d766409))
- Extend LAX lowering implementation ([`a6a76ea`](https://github.com/Dicklesworthstone/frankenjax/commit/a6a76ea00499fde9e97b7bb1cfb9f91554850b98))
- Extend AD, egraph, cache eviction, LAX ops, and conformance tests ([`9a8924f`](https://github.com/Dicklesworthstone/frankenjax/commit/9a8924f2f1db1bcce2747715e4458a7dfdc2d4b7))

### Documentation

- Complexity, performance, and memory characterization (DOC-PASS-05) ([`0978673`](https://github.com/Dicklesworthstone/frankenjax/commit/097867331a9b9f74e106905a4b48a748eac78270))
- EXHAUSTIVE_LEGACY_ANALYSIS.md Pass B expansion (DOC-PASS-11) ([`99eb6fb`](https://github.com/Dicklesworthstone/frankenjax/commit/99eb6fb0a9bfe8cdf0fd878ebd732afb1270848a))
- Red-team review fixes for both analysis documents (DOC-PASS-12) ([`0da29b4`](https://github.com/Dicklesworthstone/frankenjax/commit/0da29b49ef5fd3f31efecc5c06942ff8cfd0f906))
- Final consistency sweep and sign-off (DOC-PASS-13) ([`c4fb516`](https://github.com/Dicklesworthstone/frankenjax/commit/c4fb516a3bbd925fa8b77d1231aa31469d5594d1))

### Dependencies

- Bump asupersync from v0.1.1 to v0.2.0 across conformance and runtime crates ([`5966a82`](https://github.com/Dicklesworthstone/frankenjax/commit/5966a82dfa53d7cea71b43c5ac229b9fa7b4ac29))
- Bump ftui from 0.1.1 to 0.2.0 ([`2da9509`](https://github.com/Dicklesworthstone/frankenjax/commit/2da9509b7f2c1e7fbcad39a7e86c5e3d44aa612d))
- Upgrade dependencies across all crates and update conformance tests ([`647847c`](https://github.com/Dicklesworthstone/frankenjax/commit/647847cf97251fa09a05cdd5270e535ceb33788b))

---

## Phase 1 — Partial Evaluation, Conformance, and Infrastructure (2026-02-14 to 2026-02-15)

Partial evaluation and staging engine with bitset-indexed optimization. Conformance harness expansion with lifecycle documentation. Reliability gate infrastructure.

### Partial Evaluation (P2C-003)

- Legacy anchor map for partial evaluation and staging ([`10f5752`](https://github.com/Dicklesworthstone/frankenjax/commit/10f5752a25eeb3a2b8e4cb52d747ad99896e1443))
- Contract table with strict/hardened invariants for partial eval ([`4db4e01`](https://github.com/Dicklesworthstone/frankenjax/commit/4db4e01b801d12c6fdb404042acf20ae1b0444b7))
- Security threat matrix and compatibility envelope for partial eval ([`5319a72`](https://github.com/Dicklesworthstone/frankenjax/commit/5319a72b0cf956549716e49a9942b6562651fada))
- Add partial evaluation and staging module skeletons ([`5059e77`](https://github.com/Dicklesworthstone/frankenjax/commit/5059e77da3eefd5dfdd7d414e0993f00dab3e622))
- Comprehensive PE/staging tests with structured logging ([`6d98a4e`](https://github.com/Dicklesworthstone/frankenjax/commit/6d98a4efb85209de8fc052a3b703b35c8f3eb5d3))
- PE/staging differential oracle, metamorphic, and adversarial tests ([`05c0ef9`](https://github.com/Dicklesworthstone/frankenjax/commit/05c0ef9619657298d9d12f228c3694d86f35b7ce))
- 6 E2E scenarios for PE/staging with forensic logging ([`bf48e80`](https://github.com/Dicklesworthstone/frankenjax/commit/bf48e809953357526301e4029b5f27bb3736b8d0))
- Bitset-indexed PE with fast-path shortcuts: 27% DCE + 56% PE improvement ([`c1d15a1`](https://github.com/Dicklesworthstone/frankenjax/commit/c1d15a106f80f6a21b81f1081511f53f28d87bbe))
- P2C-003 partial eval evidence pack with 86-test conformance ([`f29ce35`](https://github.com/Dicklesworthstone/frankenjax/commit/f29ce35796dc2704ad40e9ec847af5a8d684bdae))

### Conformance and Infrastructure

- Add artifact schema validation and workspace-wide conformance infrastructure ([`0f035fa`](https://github.com/Dicklesworthstone/frankenjax/commit/0f035fa31f18bb9a156073667d7fa634b24a1f5b))
- Expand conformance harness, core transforms, and legacy parity infrastructure ([`d526f79`](https://github.com/Dicklesworthstone/frankenjax/commit/d526f797a22549628c8bf4c8662c270a395077d8))
- Add reliability gate reports, flake tracking, and quality gate enforcement ([`02ec12a`](https://github.com/Dicklesworthstone/frankenjax/commit/02ec12a78c9c19a26e2bde43d50680e4599eb922))
- Add hardened IR transforms, trace replay engine, reliability budgets, and 8 new E2E tests ([`2769e57`](https://github.com/Dicklesworthstone/frankenjax/commit/2769e5764612a23c3515bf501ddadae87e8c7f91))
- Add 24 new LAX primitive variants with eval kernels, AD rules, and e-graph support ([`a60189f`](https://github.com/Dicklesworthstone/frankenjax/commit/a60189f2104c654f6ea48eef0d937a5d1284e367))
- Refactor staging output semantics and update quality gate infrastructure ([`22376bf`](https://github.com/Dicklesworthstone/frankenjax/commit/22376bf5c739b34a1c86fed10f23ede2eed516a0))
- Conformance lifecycle, error taxonomy, security, and closure crosswalk passes ([`5fbe325`](https://github.com/Dicklesworthstone/frankenjax/commit/5fbe32596ed5f918338dfcec87a2db045e1e191a))

---

## Phase 0 — Foundation (2026-02-13)

Initial commit establishing the FrankenJAX workspace: canonical Jaxpr IR, core primitive set, automatic differentiation engine, e-graph optimizer, and conformance infrastructure.

### Features

- Initial commit: FrankenJAX foundation (M0 complete) — Jaxpr IR, core primitives, AD engine, e-graph optimizer, conformance harness ([`d5bb53f`](https://github.com/Dicklesworthstone/frankenjax/commit/d5bb53f18153040a9457ea50e787f237e6330d8c))
- AD, e-graph, conformal prediction, and performance optimizations ([`eccbee0`](https://github.com/Dicklesworthstone/frankenjax/commit/eccbee0c8c9adb9d7c1244d250df6af87ca2157a))

### Bug Fixes

- Correct round-trip variable mapping after equality saturation reordering ([`e94df54`](https://github.com/Dicklesworthstone/frankenjax/commit/e94df54ecea8e5de688a68841c416a49886ad64d))

---

## Summary of Capabilities by Phase

| Phase | Date Range | Key Capabilities Added |
|-------|-----------|----------------------|
| 0 | 2026-02-13 | Jaxpr IR, core primitives, AD engine, e-graph optimizer |
| 1 | 2026-02-14 to 02-15 | Partial evaluation, staging, conformance infrastructure, reliability gates |
| 2 | 2026-02-20 | fj-api, dispatch/effects, cache/keying, backend bridge, FFI, LAX primitives |
| 3 | 2026-02-21 to 02-25 | Numerical AD verification, e-graph trig rules, typed PE, Pad/Clamp/Iota, BackendRegistry |
| 4 | 2026-02-25 to 02-26 | Vmap (BatchTrace), control flow VJPs, Switch, PRNG, Complex64/128, 8 new primitives |
| 5 | 2026-03-02 to 03-04 | U32/U64, BF16/F16, Jacobian/Hessian, value_and_grad, parallel CPU backend |
| 6 | 2026-03-04 to 03-05 | E-graph dispatch integration, cond/scan control flow, ScalarBool, durability proofs |
| 7 | 2026-03-12 to 03-17 | 834 oracle fixtures, Cholesky/QR/SVD/Eigh AD, FFT AD, README rewrite |

---

*This changelog was built from the full 219-commit history. FrankenJAX has no formal releases or tags; all development is on the `main` branch.*
