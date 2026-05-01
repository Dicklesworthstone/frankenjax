#![forbid(unsafe_code)]

use fj_cache::{
    CACHE_KEY_NAMESPACE, CacheKey, CacheKeyInput, CacheLookup, CacheManager, build_cache_key,
    legacy_parity::{
        CacheLegacyParityLedger, CacheParityIssue, cache_legacy_parity_ledger,
        cache_legacy_parity_markdown, validate_cache_legacy_parity_ledger,
    },
};
#[cfg(test)]
use fj_core::Jaxpr;
use fj_core::{CompatibilityMode, ProgramSpec, Transform, build_program};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const CACHE_LIFECYCLE_REPORT_SCHEMA_VERSION: &str = "frankenjax.cache-lifecycle-report.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLifecycleScenario {
    pub scenario_id: String,
    pub status: String,
    pub mode: String,
    pub expected: String,
    pub actual: String,
    pub key_a: Option<String>,
    pub key_b: Option<String>,
    pub ledger_rows: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLifecycleReport {
    pub schema_version: String,
    pub bead_id: String,
    pub status: String,
    pub scenarios: Vec<CacheLifecycleScenario>,
    pub ledger_issues: Vec<CacheParityIssue>,
}

macro_rules! scenario {
    (
        $scenario_id:expr,
        $passed:expr,
        $mode:expr,
        $expected:expr,
        $actual:expr,
        $key_a:expr,
        $key_b:expr,
        $ledger_rows:expr $(,)?
    ) => {
        CacheLifecycleScenario {
            scenario_id: $scenario_id.to_owned(),
            status: if $passed { "pass" } else { "fail" }.to_owned(),
            mode: $mode.to_owned(),
            expected: $expected.to_owned(),
            actual: $actual,
            key_a: $key_a,
            key_b: $key_b,
            ledger_rows: $ledger_rows.iter().map(|row| (*row).to_owned()).collect(),
            replay_command: "./scripts/run_cache_lifecycle_gate.sh --enforce".to_owned(),
        }
    };
}

#[must_use]
pub fn build_cache_lifecycle_report(work_root: &Path) -> CacheLifecycleReport {
    let ledger = cache_legacy_parity_ledger();
    let ledger_issues = validate_cache_legacy_parity_ledger(&ledger);
    let mut scenarios = vec![
        scenario_compile_option_ordering(),
        scenario_transform_order_differentiation(),
        scenario_strict_unknown_metadata_rejection(),
        scenario_hardened_unknown_metadata_inclusion(),
        scenario_custom_hook_material(),
        scenario_namespace_version_material(),
        scenario_hostile_key_material(),
    ];
    scenarios.extend(file_cache_scenarios(work_root));

    let status = if ledger_issues.is_empty()
        && scenarios
            .iter()
            .all(|scenario| scenario.status.as_str() == "pass")
    {
        "pass"
    } else {
        "fail"
    }
    .to_owned();

    CacheLifecycleReport {
        schema_version: CACHE_LIFECYCLE_REPORT_SCHEMA_VERSION.to_owned(),
        bead_id: "frankenjax-cstq.6".to_owned(),
        status,
        scenarios,
        ledger_issues,
    }
}

pub fn write_cache_lifecycle_outputs(
    root: &Path,
    ledger_path: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> Result<(CacheLegacyParityLedger, CacheLifecycleReport), std::io::Error> {
    let ledger = cache_legacy_parity_ledger();
    let report = build_cache_lifecycle_report(root);
    let markdown = cache_legacy_parity_markdown(&ledger);

    write_json(ledger_path, &ledger)?;
    write_json(report_path, &report)?;
    if let Some(parent) = markdown_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(markdown_path, markdown)?;
    Ok((ledger, report))
}

#[must_use]
pub fn cache_lifecycle_summary_json(
    ledger: &CacheLegacyParityLedger,
    report: &CacheLifecycleReport,
) -> Value {
    json!({
        "ledger_rows": ledger.rows.len(),
        "required_surfaces": ledger.required_surfaces.len(),
        "ledger_issues": report.ledger_issues,
        "scenario_count": report.scenarios.len(),
        "scenario_status": report.scenarios.iter().map(|scenario| {
            json!({
                "scenario_id": scenario.scenario_id,
                "status": scenario.status,
                "actual": scenario.actual,
                "ledger_rows": scenario.ledger_rows,
            })
        }).collect::<Vec<_>>(),
        "gate_status": report.status,
    })
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    fs::write(path, format!("{raw}\n"))
}

fn scenario_compile_option_ordering() -> CacheLifecycleScenario {
    let mut input_a = baseline_input(CompatibilityMode::Strict);
    input_a
        .compile_options
        .insert("target".to_owned(), "x86_64".to_owned());
    input_a
        .compile_options
        .insert("opt_level".to_owned(), "3".to_owned());

    let mut input_b = baseline_input(CompatibilityMode::Strict);
    input_b
        .compile_options
        .insert("opt_level".to_owned(), "3".to_owned());
    input_b
        .compile_options
        .insert("target".to_owned(), "x86_64".to_owned());

    let key_a = build_cache_key(&input_a).expect("strict key should build");
    let key_b = build_cache_key(&input_b).expect("strict key should build");
    scenario!(
        "compile_option_ordering",
        key_a == key_b,
        "strict",
        "same key for equivalent sorted compile options",
        if key_a == key_b {
            "compile option insertion order ignored".to_owned()
        } else {
            "compile option insertion order changed the key".to_owned()
        },
        Some(key_a.as_string()),
        Some(key_b.as_string()),
        &["compile-options-sorted"],
    )
}

fn scenario_transform_order_differentiation() -> CacheLifecycleScenario {
    let mut grad_vmap = baseline_input(CompatibilityMode::Strict);
    grad_vmap.transform_stack = vec![Transform::Grad, Transform::Vmap];
    let mut vmap_grad = baseline_input(CompatibilityMode::Strict);
    vmap_grad.transform_stack = vec![Transform::Vmap, Transform::Grad];
    let key_a = build_cache_key(&grad_vmap).expect("strict key should build");
    let key_b = build_cache_key(&vmap_grad).expect("strict key should build");
    scenario!(
        "transform_order_differentiation",
        key_a != key_b,
        "strict",
        "different keys for grad,vmap and vmap,grad",
        if key_a != key_b {
            "transform order separated".to_owned()
        } else {
            "transform order aliased".to_owned()
        },
        Some(key_a.as_string()),
        Some(key_b.as_string()),
        &["transform-stack-order"],
    )
}

fn scenario_strict_unknown_metadata_rejection() -> CacheLifecycleScenario {
    let mut input = baseline_input(CompatibilityMode::Strict);
    input.unknown_incompatible_features = vec!["legacy_unknown_xla_flag".to_owned()];
    let err = build_cache_key(&input).expect_err("strict mode must reject unknown metadata");
    scenario!(
        "strict_unknown_metadata_rejection",
        format!("{err}").contains("legacy_unknown_xla_flag"),
        "strict",
        "CacheKeyError names rejected unknown metadata and no key is produced",
        format!("{err}"),
        None,
        None,
        &["unknown-metadata-policy"],
    )
}

fn scenario_hardened_unknown_metadata_inclusion() -> CacheLifecycleScenario {
    let clean = baseline_input(CompatibilityMode::Hardened);
    let mut unknown = baseline_input(CompatibilityMode::Hardened);
    unknown.unknown_incompatible_features = vec!["legacy_unknown_xla_flag".to_owned()];
    let key_a = build_cache_key(&clean).expect("hardened clean key should build");
    let key_b = build_cache_key(&unknown).expect("hardened unknown key should build");
    scenario!(
        "hardened_unknown_metadata_inclusion",
        key_a != key_b,
        "hardened",
        "unknown metadata is hash-bound and separated from clean key",
        if key_a != key_b {
            "unknown metadata changed key".to_owned()
        } else {
            "unknown metadata aliased clean key".to_owned()
        },
        Some(key_a.as_string()),
        Some(key_b.as_string()),
        &["unknown-metadata-policy"],
    )
}

fn scenario_custom_hook_material() -> CacheLifecycleScenario {
    let mut without_hook = baseline_input(CompatibilityMode::Strict);
    without_hook.custom_hook = None;
    let mut with_hook = baseline_input(CompatibilityMode::Strict);
    with_hook.custom_hook = Some("legacy-cache-hook".to_owned());
    let key_a = build_cache_key(&without_hook).expect("strict key should build");
    let key_b = build_cache_key(&with_hook).expect("strict key should build");
    scenario!(
        "custom_hook_key_material",
        key_a != key_b,
        "strict",
        "custom hook state changes cache key",
        if key_a != key_b {
            "custom hook separated".to_owned()
        } else {
            "custom hook aliased".to_owned()
        },
        Some(key_a.as_string()),
        Some(key_b.as_string()),
        &["custom-cache-hook-material"],
    )
}

fn scenario_namespace_version_material() -> CacheLifecycleScenario {
    let digest_hex = "0".repeat(64);
    let current = CacheKey {
        namespace: CACHE_KEY_NAMESPACE,
        digest_hex: digest_hex.clone(),
    };
    let next = CacheKey {
        namespace: "fjx-v2",
        digest_hex,
    };
    scenario!(
        "namespace_version_material",
        current.as_string() != next.as_string(),
        "strict",
        "namespace changes alter the public cache key string",
        if current.as_string() != next.as_string() {
            "namespace version separated".to_owned()
        } else {
            "namespace version aliased".to_owned()
        },
        Some(current.as_string()),
        Some(next.as_string()),
        &["namespace-version-material"],
    )
}

fn scenario_hostile_key_material() -> CacheLifecycleScenario {
    let mut hostile = baseline_input(CompatibilityMode::Hardened);
    hostile.backend = "cpu|backend=spoof".to_owned();
    hostile
        .compile_options
        .insert("target".to_owned(), "x86_64;unknown=spoof".to_owned());
    let mut clean = baseline_input(CompatibilityMode::Hardened);
    clean.backend = "cpubackend=spoof".to_owned();
    clean
        .compile_options
        .insert("target".to_owned(), "x86_64unknown=spoof".to_owned());
    let key_a = build_cache_key(&hostile).expect("hardened hostile key should build");
    let key_b = build_cache_key(&clean).expect("hardened clean key should build");
    scenario!(
        "hostile_key_material_no_alias",
        key_a != key_b,
        "hardened",
        "delimiter-like key material cannot alias clean material",
        if key_a != key_b {
            "hostile material separated".to_owned()
        } else {
            "hostile material aliased".to_owned()
        },
        Some(key_a.as_string()),
        Some(key_b.as_string()),
        &["hostile-key-material"],
    )
}

fn file_cache_scenarios(root: &Path) -> Vec<CacheLifecycleScenario> {
    let temp_root = unique_temp_root(root);
    let _ = fs::create_dir_all(&temp_root);
    vec![
        scenario_corrupt_read_bypass(&temp_root),
        scenario_failed_write_stays_miss(&temp_root),
    ]
}

fn unique_temp_root(root: &Path) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    root.join("target/cache-lifecycle-gate")
        .join(format!("run-{}-{nanos}", std::process::id()))
}

fn scenario_corrupt_read_bypass(temp_root: &Path) -> CacheLifecycleScenario {
    let dir = temp_root.join("corrupt-read");
    let _ = fs::create_dir_all(&dir);
    let key = build_cache_key(&baseline_input(CompatibilityMode::Strict))
        .expect("strict key should build");

    {
        let mut manager = CacheManager::file_backed(dir.clone());
        manager.put(&key, b"clean payload".to_vec());
    }

    let path = dir.join(format!("{}.bin", key.as_string()));
    let corrupt_result = fs::read(&path).and_then(|mut bytes| {
        if bytes.len() > 8 {
            bytes[8] ^= 0xff;
        }
        fs::write(&path, bytes)
    });
    let manager = CacheManager::file_backed(dir);
    let lookup = manager.get(&key);
    let passed = corrupt_result.is_ok() && matches!(lookup, CacheLookup::Corrupted { .. });
    scenario!(
        "corrupt_read_bypass",
        passed,
        "strict",
        "corrupt file-backed cache artifact is never returned as a hit",
        format!("{lookup:?}"),
        Some(key.as_string()),
        None,
        &["corrupt-read-bypass"],
    )
}

fn scenario_failed_write_stays_miss(temp_root: &Path) -> CacheLifecycleScenario {
    let missing_dir = temp_root.join("missing-write-dir").join("child");
    let key = build_cache_key(&baseline_input(CompatibilityMode::Strict))
        .expect("strict key should build");
    let mut manager = CacheManager::file_backed(missing_dir);
    manager.put(&key, b"blocked write payload".to_vec());
    let lookup = manager.get(&key);
    scenario!(
        "failed_write_stays_miss",
        matches!(lookup, CacheLookup::Miss),
        "strict",
        "failed file-cache write is not visible as a stale hit",
        format!("{lookup:?}"),
        Some(key.as_string()),
        None,
        &["stale-write-blocking"],
    )
}

fn baseline_input(mode: CompatibilityMode) -> CacheKeyInput {
    CacheKeyInput {
        mode,
        backend: "cpu".to_owned(),
        jaxpr: build_program(ProgramSpec::Add2),
        transform_stack: vec![Transform::Jit],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: Vec::new(),
    }
}

#[cfg(test)]
#[must_use]
fn empty_jaxpr() -> Jaxpr {
    Jaxpr::new(vec![], vec![], vec![], vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_lifecycle_report_passes_with_current_ledger() {
        let report = build_cache_lifecycle_report(Path::new("."));
        assert_eq!(report.schema_version, CACHE_LIFECYCLE_REPORT_SCHEMA_VERSION);
        assert_eq!(report.status, "pass", "report: {report:#?}");
        assert!(report.ledger_issues.is_empty());
        assert!(report.scenarios.len() >= 8);
    }

    #[test]
    fn cache_lifecycle_summary_names_all_scenarios() {
        let ledger = cache_legacy_parity_ledger();
        let report = build_cache_lifecycle_report(Path::new("."));
        let summary = cache_lifecycle_summary_json(&ledger, &report);
        assert_eq!(summary["gate_status"], "pass");
        assert_eq!(
            summary["scenario_count"].as_u64(),
            Some(report.scenarios.len() as u64)
        );
    }

    #[test]
    fn baseline_input_uses_nonempty_jaxpr() {
        let input = baseline_input(CompatibilityMode::Strict);
        assert_ne!(input.jaxpr, empty_jaxpr());
    }
}
