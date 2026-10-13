# Publishing the independent oracle packages

The crates.io package `fj-core` is an unrelated project. The oracle publication
names are `frankenjax-core`, `frankenjax-lax`, and `frankenjax-test-utils`, all
initially version 0.1.0. Rust library names remain `fj_core`, `fj_lax`, and
`fj_test_utils`; numerical implementations, conformance tests, and workspace
member directory names are unchanged. Existing consumers should use package
aliases rather than changing Rust imports or disabling their oracle feature.

```toml
fj-core = { package = "frankenjax-core", version = "=0.1.0" }
fj-lax = { package = "frankenjax-lax", version = "=0.1.0" }
```

These declarations prepare source for publication; they do not assert that
any of these versions is published or that any name is reserved. Do not
substitute crates.io `fj-core`, silently omit optional features, or claim a
registry-only build while using sibling paths. In-tree development retains
explicit path-plus-version dependencies; Cargo normalizes those to registry
dependencies when packaging. The repository LICENSE, including its rider,
is retained through `license-file`, not relabeled as plain MIT.

## Qualification and publication order

Run the existing workspace quality/conformance gates in the prescribed Rust
execution environment. Commands selecting a package now use its publication
name, for example `cargo test -p frankenjax-core` or
`cargo test -p frankenjax-lax`; `fj-conformance` retains its existing name.

Publish and independently verify in dependency order:

1. `frankenjax-test-utils` (no JAX dependencies).
2. `frankenjax-core` (uses the published helper for tests).
3. `frankenjax-lax` (uses published core and helper).

Before each publication, run `cargo package --locked -p <package>` with
verification enabled, inspect the resulting archive and LICENSE, and test
its extracted contents outside this workspace with an isolated Cargo home.
Do not use `--no-verify` as evidence of a working release. A package requiring
an unpublished predecessor must fail until that predecessor is actually
available. Preserve the packaged normalized manifest, registry lockfile,
archive checksum, source revision, toolchain, and test logs as evidence.

After publication, qualify the downstream `franken_whisper` package with
its `fj-oracle` feature and existing conformance assertions enabled. Registry
source identity checks do not replace numerical/behavioral tests.

## Scope of the metadata migration

The source base is 4f04dd82ace83f609eedd232839244ce872bb4ce. Metadata-only
changes preserve its numerical code; this is not a claim that it is identical
to Whisper's older ec74b7b release pin. A downstream pin advance requires
fresh oracle conformance qualification.

The root and fuzz lockfiles change local package identities and references
only, preserving existing third-party records and checksums. Static TOML,
path-alias, and lock identity checks were performed. Cargo resolution,
workspace tests, package verification, fuzz builds, and registry publication
have not been performed by this metadata change. In particular the existing
fuzz lock already predates some core dependency changes; refresh/qualify it
with Cargo before claiming a locked fuzz build.
