#![forbid(unsafe_code)]

use crate::CACHE_KEY_NAMESPACE;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CACHE_LEGACY_PARITY_LEDGER_SCHEMA_VERSION: &str =
    "frankenjax.cache-legacy-parity-ledger.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheParitySurface {
    CacheKey,
    CompilerOptions,
    CompilationCacheMetadata,
    BackendIdentity,
    TransformStack,
    VersioningInputs,
    UnknownMetadata,
    CacheHooks,
    CorruptReads,
    StaleWrites,
    HostileKeyMaterial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheParityStatus {
    Modeled,
    ModeledWithScopeDifference,
    ExplicitExclusion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLegacyParityRow {
    pub row_id: String,
    pub surface: CacheParitySurface,
    pub legacy_anchor: String,
    pub legacy_path: String,
    pub legacy_symbol: String,
    pub legacy_behavior: String,
    pub rust_key_material: Vec<String>,
    pub rust_behavior: String,
    pub strict_behavior: String,
    pub hardened_behavior: String,
    pub status: CacheParityStatus,
    pub exclusion_reason: Option<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLegacyParityLedger {
    pub schema_version: String,
    pub key_namespace: String,
    pub required_surfaces: Vec<CacheParitySurface>,
    pub rows: Vec<CacheLegacyParityRow>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheParityIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl CacheParityIssue {
    #[must_use]
    pub fn new(
        code: impl Into<String>,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            path: path.into(),
            message: message.into(),
        }
    }
}

#[must_use]
pub fn required_cache_surfaces() -> Vec<CacheParitySurface> {
    vec![
        CacheParitySurface::CacheKey,
        CacheParitySurface::CompilerOptions,
        CacheParitySurface::CompilationCacheMetadata,
        CacheParitySurface::BackendIdentity,
        CacheParitySurface::TransformStack,
        CacheParitySurface::VersioningInputs,
        CacheParitySurface::UnknownMetadata,
        CacheParitySurface::CacheHooks,
        CacheParitySurface::CorruptReads,
        CacheParitySurface::StaleWrites,
        CacheParitySurface::HostileKeyMaterial,
    ]
}

#[must_use]
pub fn cache_legacy_parity_ledger() -> CacheLegacyParityLedger {
    CacheLegacyParityLedger {
        schema_version: CACHE_LEGACY_PARITY_LEDGER_SCHEMA_VERSION.to_owned(),
        key_namespace: CACHE_KEY_NAMESPACE.to_owned(),
        required_surfaces: required_cache_surfaces(),
        rows: cache_legacy_parity_rows(),
    }
}

#[must_use]
pub fn cache_legacy_parity_rows() -> Vec<CacheLegacyParityRow> {
    macro_rules! row {
        (
            $row_id:expr,
            $surface:expr,
            $legacy_anchor:expr,
            $legacy_path:expr,
            $legacy_symbol:expr,
            $legacy_behavior:expr,
            $rust_key_material:expr,
            $rust_behavior:expr,
            $strict_behavior:expr,
            $hardened_behavior:expr,
            $status:expr,
            $exclusion_reason:expr,
            $evidence_refs:expr $(,)?
        ) => {
            CacheLegacyParityRow {
                row_id: $row_id.to_owned(),
                surface: $surface,
                legacy_anchor: $legacy_anchor.to_owned(),
                legacy_path: $legacy_path.to_owned(),
                legacy_symbol: $legacy_symbol.to_owned(),
                legacy_behavior: $legacy_behavior.to_owned(),
                rust_key_material: $rust_key_material
                    .iter()
                    .map(|value| (*value).to_owned())
                    .collect(),
                rust_behavior: $rust_behavior.to_owned(),
                strict_behavior: $strict_behavior.to_owned(),
                hardened_behavior: $hardened_behavior.to_owned(),
                status: $status,
                exclusion_reason: $exclusion_reason,
                evidence_refs: $evidence_refs
                    .iter()
                    .map(|value| (*value).to_owned())
                    .collect(),
            }
        };
    }

    vec![
        row!(
            "cache-key-hash-inputs",
            CacheParitySurface::CacheKey,
            "P2C005-A01,P2C005-A02",
            "jax/_src/cache_key.py",
            "get,_hash_computation",
            "Legacy JAX hashes compiled computation bytes plus devices, compile options, backend metadata, and version material.",
            &[
                "jaxpr.canonical_fingerprint",
                "mode",
                "backend",
                "transform_stack",
                "compile_options",
                "custom_hook",
                "unknown_incompatible_features",
            ],
            "FrankenJAX hashes canonical Jaxpr structure and all Rust-side execution configuration with SHA-256.",
            "Rejects unknown incompatible feature material before hashing.",
            "Hashes unknown feature material into the key so progress remains auditable.",
            CacheParityStatus::ModeledWithScopeDifference,
            None,
            &[
                "crates/fj-cache/src/lib.rs::build_cache_key_ref",
                "crates/fj-conformance/tests/cache_keying_oracle.rs",
            ],
        ),
        row!(
            "compile-options-sorted",
            CacheParitySurface::CompilerOptions,
            "P2C005-A10",
            "jax/_src/cache_key.py",
            "_hash_xla_flags",
            "Legacy JAX includes compiler flags and codegen-affecting options in the hash.",
            &["compile_options:BTreeMap<String,String>"],
            "Rust compile options are sorted by BTreeMap key order before hashing.",
            "Sorted options are hashed only when all material is recognized.",
            "Sorted options are hashed together with unknown feature material.",
            CacheParityStatus::Modeled,
            None,
            &[
                "crates/fj-cache/src/lib.rs::hash_canonical_payload_ref",
                "crates/fj-cache/src/legacy_parity.rs tests",
            ],
        ),
        row!(
            "backend-identity",
            CacheParitySurface::BackendIdentity,
            "P2C005-A11,P2C005-A16",
            "jax/_src/cache_key.py",
            "_hash_devices,_hash_platform",
            "Legacy JAX includes device assignment and platform identity to prevent cross-device artifact reuse.",
            &["backend"],
            "V1 Rust cache keys include backend identity. Current runtime scope is CPU, while non-CPU strings still separate keys.",
            "Unknown backend semantics must be represented as incompatible feature material by callers.",
            "Backend string still separates keys for defensive diagnostics.",
            CacheParityStatus::ModeledWithScopeDifference,
            None,
            &["crates/fj-conformance/tests/cache_keying_oracle.rs::oracle_cache_key_sensitivity_per_field"],
        ),
        row!(
            "transform-stack-order",
            CacheParitySurface::TransformStack,
            "P2C005-A12,P2C005-A13",
            "jax/_src/linear_util.py,jax/_src/compiler.py",
            "cache,get_executable",
            "Legacy JAX compilation caches distinguish transformed function state and abstract input state.",
            &["transform_stack"],
            "Rust cache keys preserve exact transform order; grad,vmap and vmap,grad hash differently.",
            "Transform stack is required key material.",
            "Transform stack is required key material.",
            CacheParityStatus::Modeled,
            None,
            &["crates/fj-cache/src/lib.rs tests"],
        ),
        row!(
            "namespace-version-material",
            CacheParitySurface::VersioningInputs,
            "P2C005-A03,P2C005-A18",
            "jax/_src/cache_key.py",
            "CacheKey,_version_hash",
            "Legacy JAX includes JAX version material in the hash and stores raw hex keys.",
            &[CACHE_KEY_NAMESPACE],
            "Rust uses the fjx namespace as explicit coarse version material for all cache keys.",
            "Namespace mismatch is a fail-closed cache miss across versions.",
            "Namespace mismatch is a diagnostic cache miss; no stale artifact is accepted.",
            CacheParityStatus::ModeledWithScopeDifference,
            None,
            &["crates/fj-cache/src/lib.rs::CacheKey::as_string"],
        ),
        row!(
            "unknown-metadata-policy",
            CacheParitySurface::UnknownMetadata,
            "P2C005-A01,P2C005-A10",
            "jax/_src/cache_key.py",
            "get,_hash_xla_flags",
            "Legacy JAX hashes known metadata and versioned compiler options.",
            &["unknown_incompatible_features"],
            "Rust makes unknown incompatible feature material explicit at the API boundary.",
            "Strict mode returns CacheKeyError::UnknownIncompatibleFeatures without producing a key.",
            "Hardened mode hashes the feature list so distinct unknowns cannot alias clean keys.",
            CacheParityStatus::Modeled,
            None,
            &["crates/fj-cache/src/lib.rs tests"],
        ),
        row!(
            "custom-cache-hook-material",
            CacheParitySurface::CacheHooks,
            "P2C005-A12,P2C005-A15",
            "jax/_src/linear_util.py,jax/_src/compiler.py",
            "cache,_cache_hit_logging",
            "Legacy JAX cache behavior includes function identity and hook-like wrapping state in memoization layers.",
            &["custom_hook"],
            "Rust exposes custom_hook as explicit key material instead of implicit function identity.",
            "Hook material changes the key only when the caller supplies recognized hook state.",
            "Hook material changes the key and can be diagnosed in lifecycle reports.",
            CacheParityStatus::ModeledWithScopeDifference,
            None,
            &["crates/fj-cache/src/lib.rs tests"],
        ),
        row!(
            "compilation-cache-metadata",
            CacheParitySurface::CompilationCacheMetadata,
            "P2C005-A04,P2C005-A06,P2C005-A13",
            "jax/_src/compiler.py,jax/_src/compilation_cache/compilation_cache.py",
            "_compile_and_write_cache,initialize_cache,get_executable",
            "Legacy JAX can read/write compiled executable artifacts through memory, file, and cloud cache layers.",
            &["CacheManager", "CacheBackend", "CachedArtifact.integrity_sha256_hex"],
            "Rust V1 has cache-manager primitives and deterministic keys, but dispatch still interprets directly instead of compiling XLA executables.",
            "Compiled-artifact cache misses fall back to recomputation/interpreted execution.",
            "Malformed cache state is reported as corrupted or missed without accepting stale bytes.",
            CacheParityStatus::ModeledWithScopeDifference,
            None,
            &[
                "crates/fj-cache/src/backend.rs",
                "crates/fj-conformance/tests/e2e_p2c005.rs",
            ],
        ),
        row!(
            "corrupt-read-bypass",
            CacheParitySurface::CorruptReads,
            "P2C005-A05,P2C005-A07",
            "jax/_src/compiler.py,jax/_src/compilation_cache/file_cache.py",
            "_read_from_cache,FileCache.get",
            "Legacy JAX treats missing or incompatible cached executables as cache misses and recompiles.",
            &["CachedArtifact.integrity_sha256_hex", "persistence::deserialize"],
            "Rust file cache embeds payload digests and surfaces corrupt reads as CacheLookup::Corrupted.",
            "Corrupt reads are not accepted as hits.",
            "Corrupt reads are not accepted as hits; recovery must be audited by callers.",
            CacheParityStatus::Modeled,
            None,
            &["crates/fj-cache/src/backend.rs tests"],
        ),
        row!(
            "stale-write-blocking",
            CacheParitySurface::StaleWrites,
            "P2C005-A04,P2C005-A07",
            "jax/_src/compiler.py,jax/_src/compilation_cache/file_cache.py",
            "_cache_write,FileCache.put",
            "Legacy JAX writes cache artifacts atomically and treats write failures as non-fatal cache misses.",
            &["FileCache::put atomic temp-write-rename"],
            "Rust file cache writes serialized artifacts through a temporary file and rename; failed writes do not create readable hits.",
            "A failed write cannot become a strict cache hit.",
            "A failed write cannot become a hardened cache hit.",
            CacheParityStatus::Modeled,
            None,
            &["crates/fj-cache/src/backend.rs"],
        ),
        row!(
            "hostile-key-material",
            CacheParitySurface::HostileKeyMaterial,
            "P2C005-A02",
            "jax/_src/cache_key.py",
            "_hash_computation",
            "Legacy JAX hashes raw serialized fields so delimiter-like bytes are part of the digest input.",
            &["backend", "compile_options", "custom_hook", "unknown_incompatible_features"],
            "Rust hashes delimiter-bearing strings as bytes in their field positions; lifecycle tests prove hostile material does not alias clean keys.",
            "Hostile unknown material is rejected when classified incompatible.",
            "Hostile unknown material is hashed and separated from clean key material.",
            CacheParityStatus::Modeled,
            None,
            &["crates/fj-conformance/tests/cache_keying_oracle.rs::adversarial_delimiter_in_backend_name"],
        ),
        row!(
            "gcs-cache-exclusion",
            CacheParitySurface::CompilationCacheMetadata,
            "P2C005-A08",
            "jax/_src/compilation_cache/gcs_cache.py",
            "GcsCache",
            "Legacy JAX can use Google Cloud Storage as a persistent compilation cache backend.",
            &[] as &[&str],
            "FrankenJAX V1 intentionally excludes cloud cache backends; local file and in-memory cache primitives cover current scope.",
            "Cloud cache URIs must not be silently accepted.",
            "Cloud cache URIs must not be silently accepted.",
            CacheParityStatus::ExplicitExclusion,
            Some("GCS compilation cache is outside V1 CPU/runtime scope and would require credentials, network policy, and artifact lifecycle controls.".to_owned()),
            &["artifacts/phase2c/FJ-P2C-005/security_threat_matrix.md"],
        ),
    ]
}

#[must_use]
pub fn validate_cache_legacy_parity_ledger(
    ledger: &CacheLegacyParityLedger,
) -> Vec<CacheParityIssue> {
    let mut issues = Vec::new();
    if ledger.schema_version != CACHE_LEGACY_PARITY_LEDGER_SCHEMA_VERSION {
        issues.push(CacheParityIssue::new(
            "unsupported_schema_version",
            "$.schema_version",
            format!(
                "expected `{}`, got `{}`",
                CACHE_LEGACY_PARITY_LEDGER_SCHEMA_VERSION, ledger.schema_version
            ),
        ));
    }
    if ledger.key_namespace != CACHE_KEY_NAMESPACE {
        issues.push(CacheParityIssue::new(
            "wrong_key_namespace",
            "$.key_namespace",
            format!(
                "expected namespace `{CACHE_KEY_NAMESPACE}`, got `{}`",
                ledger.key_namespace
            ),
        ));
    }

    let mut row_ids = BTreeSet::new();
    let mut covered = BTreeMap::<CacheParitySurface, usize>::new();
    for (idx, row) in ledger.rows.iter().enumerate() {
        let row_path = format!("$.rows[{idx}]");
        if !row_ids.insert(row.row_id.clone()) {
            issues.push(CacheParityIssue::new(
                "duplicate_row_id",
                format!("{row_path}.row_id"),
                format!("duplicate row id `{}`", row.row_id),
            ));
        }
        *covered.entry(row.surface).or_default() += 1;
        if row.legacy_anchor.trim().is_empty() {
            issues.push(CacheParityIssue::new(
                "missing_legacy_anchor",
                format!("{row_path}.legacy_anchor"),
                "row must name the legacy anchor",
            ));
        }
        if row.rust_behavior.trim().is_empty() {
            issues.push(CacheParityIssue::new(
                "missing_rust_behavior",
                format!("{row_path}.rust_behavior"),
                "row must describe Rust behavior",
            ));
        }
        if row.evidence_refs.is_empty() {
            issues.push(CacheParityIssue::new(
                "missing_evidence",
                format!("{row_path}.evidence_refs"),
                "row must link tests, code, or artifacts",
            ));
        }
        if row.status == CacheParityStatus::ExplicitExclusion && row.exclusion_reason.is_none() {
            issues.push(CacheParityIssue::new(
                "missing_exclusion_reason",
                format!("{row_path}.exclusion_reason"),
                "explicit exclusions require a reason",
            ));
        }
    }

    for surface in &ledger.required_surfaces {
        if !covered.contains_key(surface) {
            issues.push(CacheParityIssue::new(
                "missing_required_surface",
                "$.rows",
                format!("required surface `{surface:?}` is not covered"),
            ));
        }
    }

    issues
}

#[must_use]
pub fn cache_legacy_parity_markdown(ledger: &CacheLegacyParityLedger) -> String {
    let mut out = String::new();
    out.push_str("# Cache Legacy Parity Ledger\n\n");
    out.push_str(&format!("- schema: `{}`\n", ledger.schema_version));
    out.push_str(&format!("- key namespace: `{}`\n\n", ledger.key_namespace));
    out.push_str("| Row | Surface | Status | Legacy Anchor | Rust Behavior |\n");
    out.push_str("|-----|---------|--------|---------------|---------------|\n");
    for row in &ledger.rows {
        out.push_str(&format!(
            "| `{}` | `{:?}` | `{:?}` | `{}` | {} |\n",
            row.row_id,
            row.surface,
            row.status,
            row.legacy_anchor,
            row.rust_behavior.replace('|', "\\|")
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CacheKey;

    #[test]
    fn ledger_covers_required_surfaces() {
        let ledger = cache_legacy_parity_ledger();
        let issues = validate_cache_legacy_parity_ledger(&ledger);
        assert!(issues.is_empty(), "ledger issues: {issues:?}");
    }

    #[test]
    fn missing_required_surface_is_rejected() {
        let mut ledger = cache_legacy_parity_ledger();
        ledger
            .rows
            .retain(|row| row.surface != CacheParitySurface::CacheHooks);
        let issues = validate_cache_legacy_parity_ledger(&ledger);
        assert!(
            issues
                .iter()
                .any(|issue| issue.code == "missing_required_surface"),
            "missing surface should be reported: {issues:?}"
        );
    }

    #[test]
    fn duplicate_row_ids_are_rejected() {
        let mut ledger = cache_legacy_parity_ledger();
        ledger.rows[1].row_id = ledger.rows[0].row_id.clone();
        let issues = validate_cache_legacy_parity_ledger(&ledger);
        assert!(
            issues.iter().any(|issue| issue.code == "duplicate_row_id"),
            "duplicate row id should be reported: {issues:?}"
        );
    }

    #[test]
    fn explicit_exclusions_require_reasons() {
        let mut ledger = cache_legacy_parity_ledger();
        let exclusion = ledger
            .rows
            .iter_mut()
            .find(|row| row.status == CacheParityStatus::ExplicitExclusion)
            .expect("fixture should contain an explicit exclusion");
        exclusion.exclusion_reason = None;
        let issues = validate_cache_legacy_parity_ledger(&ledger);
        assert!(
            issues
                .iter()
                .any(|issue| issue.code == "missing_exclusion_reason"),
            "missing exclusion reason should be reported: {issues:?}"
        );
    }

    #[test]
    fn strict_and_hardened_unknown_metadata_rows_are_explicit() {
        let ledger = cache_legacy_parity_ledger();
        let row = ledger
            .rows
            .iter()
            .find(|row| row.surface == CacheParitySurface::UnknownMetadata)
            .expect("unknown metadata row should exist");
        assert!(row.strict_behavior.contains("UnknownIncompatibleFeatures"));
        assert!(row.hardened_behavior.contains("hash"));
    }

    #[test]
    fn cache_key_namespace_is_versioning_material() {
        let digest_hex = "0".repeat(64);
        let current = CacheKey {
            namespace: CACHE_KEY_NAMESPACE,
            digest_hex: digest_hex.clone(),
        };
        let next = CacheKey {
            namespace: "fjx-v2",
            digest_hex,
        };
        assert_ne!(current.as_string(), next.as_string());
    }
}
