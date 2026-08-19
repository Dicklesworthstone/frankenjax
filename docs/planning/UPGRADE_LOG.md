# Dependency Upgrade Log

**Date:** 2026-02-20  |  **Project:** FrankenJAX  |  **Language:** Rust  |  **Toolchain:** nightly-2026-02-19 (rustc 1.95.0-nightly)

## Summary

- **Updated:** 11  |  **Skipped:** 0  |  **Failed:** 0  |  **Already latest:** 1 (base64)

## Toolchain

- **Rust nightly:** Already on latest (`nightly-2026-02-19`, rustc 1.95.0-nightly)
- `rust-toolchain.toml` uses `channel = "nightly"` (floating, always latest)

## Workspace-Level Updates

### smallvec: 1.13 -> 1.15
- **Breaking:** None (minor version bump)
- **Tests:** All passed

### rustc-hash: 2.1 -> 2.1.1
- **Breaking:** None (patch bump)
- **Tests:** All passed

### proptest: 1.6 -> 1.9 (resolved to 1.10.0)
- **Breaking:** None (minor version bump)
- **Tests:** All passed

### egg: 0.9 -> 0.11
- **Breaking changes in library:** `Analysis::make` signature changed (added `&mut` and `id: Id`), `Language` trait requires new `Discriminant` associated type, `Searcher` trait method renamed
- **Impact on FrankenJAX:** None -- code uses `define_language!` macro (auto-generates Discriminant), default `()` analysis (no custom `make`), and no custom `Searcher`
- **Tests:** All 8 egraph tests passed

## Per-Crate Updates

### serde: 1.0.218 -> 1.0.228
- **Crates:** fj-core, fj-conformance, fj-ledger, fj-test-utils
- **Breaking:** None (patch bump)
- **Tests:** All passed

### serde_json: 1.0.139 -> 1.0.149
- **Crates:** fj-core (dev), fj-trace (dev), fj-conformance, fj-test-utils, fj-interpreters (dev)
- **Breaking:** None (patch bump)
- **Tests:** All passed

### sha2: 0.10.8 -> 0.10.9
- **Crates:** fj-cache, fj-conformance, fj-test-utils
- **Breaking:** None (patch bump)
- **Tests:** All passed

### criterion: 0.5/0.5.1 -> 0.8
- **Crates:** fj-interpreters (dev), fj-dispatch (dev)
- **Breaking changes in library:** `async_std` feature removed, MSRV bumped to 1.86, `real_blackbox` feature is now no-op
- **Impact on FrankenJAX:** None -- benchmark files use only `Criterion`, `criterion_group!`, `criterion_main!`, and `b.iter()` which are all unchanged
- **Tests:** All passed

### jsonschema: 0.41.0 -> 0.42
- **Crates:** fj-conformance (dev)
- **Breaking changes in library:** Default TLS provider switched to `aws-lc-rs`
- **Impact on FrankenJAX:** None -- only used for local schema validation, no remote HTTPS fetching
- **Tests:** All passed

### tempfile: 3.17.1 -> 3.25
- **Crates:** fj-conformance (dev)
- **Breaking:** None (minor version bump)
- **Tests:** All passed

## Already Latest

### base64: 0.22.1
- Already on latest stable version

## Code Fixes Required

### fj-lax: New clippy lints from nightly 1.95.0

1. **`manual_is_multiple_of`** (line 517): Replaced `elem_count % known_product != 0` with `!elem_count.is_multiple_of(known_product)`
2. **`needless_range_loop`** (line 606): Replaced `for flat_idx in 0..total` indexing pattern with `for (flat_idx, elem) in new_elements.iter_mut().enumerate()`
3. **`approx_constant`** (lines 1252, 1254): Changed test value from `3.14` (too close to `PI`) to `2.78` to avoid lint

## Path Dependencies (Not Updated)

- **asupersync** v0.2.0 -> v0.2.5 (resolved via path at `/dp/asupersync`)
- **ftui** v0.2.0 (path dep at `/dp/frankentui/crates/ftui`)

These are local path dependencies managed separately.

## Quality Gates

- `cargo check --workspace --all-targets`: Passed
- `cargo clippy --workspace --all-targets -- -D warnings`: Passed (only external `asupersync` warnings)
- `cargo fmt --check`: Passed
- `cargo test --workspace`: **253 tests passed, 0 failed**
