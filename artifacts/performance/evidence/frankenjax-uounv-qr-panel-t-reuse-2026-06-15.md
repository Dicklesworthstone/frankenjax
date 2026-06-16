# frankenjax-uounv: QR compact-WY panel-T reuse

## Target

- Bead: `frankenjax-uounv`
- Hotspot: `fj-lax` Criterion row `linalg/qr_1024x1024_f64`
- Profile source: post-`frankenjax-mcqr.66` RCH linalg sweep on `vmi1227854` measured QR at `[30.703 ms 31.126 ms 31.545 ms]`, the top current production linalg row after rejected Cholesky detours.
- Prior rejected lever to avoid: QR row-thread fanout (`frankenjax-mcqr.64`) regressed same-worker QR from `[29.588 ms 29.995 ms 30.404 ms]` to `[61.934 ms 64.948 ms 68.124 ms]`.

## Lever

Cache each blocked QR panel's compact-WY `T` matrix during the R-factor pass and reuse that exact `T` during the later Q reconstruction pass.

Before this change, the blocked path computed `T` for each non-final panel before applying the trailing R update, then recomputed every panel's `T` during Q reconstruction. The final panel had no trailing R update, so it only computed `T` during Q reconstruction. The new path computes one `T` immediately after each panel's reflectors and taus are finalized, uses it for the trailing R update when needed, and stores it for the reverse-order Q pass.

## Push-Base Baseline

Command:

```text
rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'linalg/qr_1024x1024_f64'
```

Worker: `vmi1149989`
Worktree: clean `origin/main` at `9d7a1ac2` after peer commits landed.

Result:

```text
linalg/qr_1024x1024_f64 time: [49.319 ms 55.254 ms 62.400 ms]
```

## Re-benchmark

Command:

```text
rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'linalg/qr_1024x1024_f64'
```

Worker: `vmi1149989`
Commit: rebased candidate `68f4250a` on top of `9d7a1ac2`.

Result:

```text
linalg/qr_1024x1024_f64 time: [25.564 ms 25.917 ms 26.286 ms]
```

Midpoint ratio: `55.254 / 25.917 = 2.13x`.
Conservative interval ratio: `49.319 / 26.286 = 1.88x`.

Score: `3.80 = Impact 4.0 x Confidence 0.95 / Effort 1.0`.

Rationale: top current production QR row, same-worker RCH comparison against the actual push base, focused proof/gates passed, and a single low-risk reuse lever with no algorithmic contract expansion.

The original pre-rebase baseline before peer commits landed was also recorded on `vmi1149989` at `[26.832 ms 27.240 ms 27.657 ms]`; after rebasing onto `9d7a1ac2`, the clean push-base baseline above is the acceptance comparison.

## Behavior Proof

Focused proof command:

```text
rch exec -- cargo test -p fj-lax qr -- --nocapture
```

Worker: `vmi1149989`
Commit: rebased candidate `68f4250a`.

Result: 31 passed, 4 ignored. The proof includes:

- `qr_blocked_reconstructs_and_orthonormal`
- `qr_real_path_golden_output_digest`
- existing scalar/complex/shape/error QR coverage selected by the `qr` filter

Golden output SHA-256:

```text
6119fc5cf4759d8cdcd9c34d89a79de89d205203730814fc06aa52bf57ff262b
```

`fixture_id_from_json` serializes the output bits with `serde_json` and hashes them with SHA-256.

## Isomorphism

- Panel order is unchanged: panels are still factored from low column to high column.
- Reflector order is unchanged: R update still applies the panel reflector immediately after panel factorization; Q reconstruction still applies panels in reverse order.
- Floating-point operation order inside `qr_compact_wy_t` and `qr_block_apply` is unchanged. The change reuses the same `T` value instead of recomputing it later.
- Output ordering, QR sign convention, dtype/shape/error behavior, tie surface, and RNG absence are unchanged.
- The scalar/non-blocked QR path is unchanged.
- The implementation remains safe Rust with no `unsafe`.

## Validation

- `cargo fmt -p fj-lax`
- `cargo fmt --check -p fj-lax`
- `rch exec -- cargo test -p fj-lax qr -- --nocapture`
- post-rebase `rch exec -- cargo test -p fj-lax qr -- --nocapture`
- `rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'linalg/qr_1024x1024_f64'`
- clean push-base `origin/main` RCH Criterion for the same QR row on `vmi1149989`
- `rch exec -- cargo check -p fj-lax --all-targets`
- `rch exec -- cargo clippy -p fj-lax --all-targets -- -D warnings`
- `ubs crates/fj-lax/src/linalg.rs` was run. It exits nonzero on pre-existing file-wide inventories in `linalg.rs` (test unwraps/panics/asserts and direct indexing); its embedded formatter, clippy, cargo check, test-build, audit, and deny sections were clean.

## Next Primitive

Reprofile after landing. If QR remains dominant, do not retry row-thread fanout; the next QR lever should be a deeper communication-avoiding or panel-memory-layout primitive. If another linalg/special-function row rises above QR, attack that profile-backed target instead.
