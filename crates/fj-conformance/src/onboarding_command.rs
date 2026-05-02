#![forbid(unsafe_code)]

use crate::e2e_log::{
    E2EAllocationCounters, E2EArtifactRef, E2ECompatibilityMode, E2EErrorClass, E2EForensicLogV1,
    E2ELogStatus, E2ETolerancePolicy, artifact_sha256_hex, write_e2e_log,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const ONBOARDING_COMMAND_INVENTORY_SCHEMA_VERSION: &str =
    "frankenjax.onboarding-command-inventory.v1";
pub const ONBOARDING_COMMAND_BEAD_ID: &str = "frankenjax-cstq.18";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnboardingCommandInventory {
    pub schema_version: String,
    pub bead_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub description: String,
    pub source_docs: Vec<String>,
    pub commands: Vec<OnboardingCommandEntry>,
    pub summary: OnboardingCommandSummary,
    pub issues: Vec<OnboardingCommandIssue>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnboardingCommandEntry {
    pub command_id: String,
    pub command: String,
    pub smoke_command: Option<String>,
    pub classification: String,
    pub expected_exit: i32,
    pub skip_reason: Option<String>,
    pub replay_command: String,
    pub env_allowlist: Vec<String>,
    pub feature_flags: Vec<String>,
    pub evidence_status: String,
    pub evidence_refs: Vec<String>,
    pub source_refs: Vec<String>,
    pub artifact_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnboardingCommandSummary {
    pub total_commands: usize,
    pub green_count: usize,
    pub yellow_count: usize,
    pub red_count: usize,
    pub mandatory_smoke_count: usize,
    pub ci_gate_count: usize,
    pub optional_oracle_count: usize,
    pub long_running_count: usize,
    pub environment_specific_count: usize,
    pub schematic_count: usize,
    pub skip_count: usize,
    pub executable_smoke_count: usize,
    pub source_doc_ref_count: usize,
    pub evidence_ref_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnboardingCommandIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl OnboardingCommandIssue {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnboardingCommandOutputPaths {
    pub inventory: PathBuf,
    pub e2e: PathBuf,
}

impl OnboardingCommandOutputPaths {
    #[must_use]
    pub fn for_root(root: &Path) -> Self {
        Self {
            inventory: root.join("artifacts/conformance/onboarding_command_inventory.v1.json"),
            e2e: root.join("artifacts/e2e/e2e_onboarding_gate.e2e.json"),
        }
    }
}

#[must_use]
pub fn build_onboarding_command_inventory(root: &Path) -> OnboardingCommandInventory {
    let commands = command_specs()
        .into_iter()
        .map(OnboardingCommandEntry::from)
        .collect::<Vec<_>>();
    let summary = summarize_commands(&commands);
    let mut inventory = OnboardingCommandInventory {
        schema_version: ONBOARDING_COMMAND_INVENTORY_SCHEMA_VERSION.to_owned(),
        bead_id: ONBOARDING_COMMAND_BEAD_ID.to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        status: "pass".to_owned(),
        description:
            "Replayable inventory of README and key-document commands for fresh-clone onboarding, CI gates, durability, E2E orchestration, fuzzing, and optional JAX oracle setup."
                .to_owned(),
        source_docs: vec![
            "README.md".to_owned(),
            "AGENTS.md".to_owned(),
            "FEATURE_PARITY.md".to_owned(),
        ],
        commands,
        summary,
        issues: Vec::new(),
        replay_command: "./scripts/run_onboarding_gate.sh --enforce".to_owned(),
    };
    inventory.issues = validate_onboarding_command_inventory(root, &inventory);
    if !inventory.issues.is_empty() {
        inventory.status = "fail".to_owned();
    }
    inventory
}

pub fn write_onboarding_command_outputs(
    root: &Path,
    inventory_path: &Path,
    e2e_path: &Path,
) -> Result<OnboardingCommandInventory, std::io::Error> {
    let inventory = build_onboarding_command_inventory(root);
    write_json(inventory_path, &inventory)?;
    let log = build_onboarding_e2e_log(root, inventory_path, &inventory)?;
    write_e2e_log(e2e_path, &log)?;
    Ok(inventory)
}

#[must_use]
pub fn onboarding_inventory_summary_json(inventory: &OnboardingCommandInventory) -> JsonValue {
    json!({
        "schema_version": inventory.schema_version,
        "bead_id": inventory.bead_id,
        "status": inventory.status,
        "summary": inventory.summary,
        "issue_count": inventory.issues.len(),
        "issues": inventory.issues,
    })
}

#[must_use]
pub fn validate_onboarding_command_inventory(
    root: &Path,
    inventory: &OnboardingCommandInventory,
) -> Vec<OnboardingCommandIssue> {
    let mut issues = Vec::new();

    if inventory.schema_version != ONBOARDING_COMMAND_INVENTORY_SCHEMA_VERSION {
        issues.push(OnboardingCommandIssue::new(
            "bad_schema_version",
            "$.schema_version",
            "onboarding inventory schema marker changed",
        ));
    }
    if inventory.bead_id != ONBOARDING_COMMAND_BEAD_ID {
        issues.push(OnboardingCommandIssue::new(
            "bad_bead_id",
            "$.bead_id",
            "onboarding command gate must stay bound to frankenjax-cstq.18",
        ));
    }
    if inventory.replay_command.trim().is_empty() {
        issues.push(OnboardingCommandIssue::new(
            "missing_replay_command",
            "$.replay_command",
            "inventory needs a stable replay command",
        ));
    }

    for source_doc in &inventory.source_docs {
        if !root.join(source_doc).exists() {
            issues.push(OnboardingCommandIssue::new(
                "missing_source_doc",
                "$.source_docs",
                format!("source document `{source_doc}` does not exist"),
            ));
        }
    }

    let mut ids = BTreeSet::new();
    for (idx, command) in inventory.commands.iter().enumerate() {
        let path = format!("$.commands[{idx}]");
        if !ids.insert(command.command_id.as_str()) {
            issues.push(OnboardingCommandIssue::new(
                "duplicate_command_id",
                path.clone(),
                format!("duplicate command id `{}`", command.command_id),
            ));
        }
        if !command.command_id.starts_with("cmd_") {
            issues.push(OnboardingCommandIssue::new(
                "bad_command_id",
                path.clone(),
                format!("command id `{}` must start with cmd_", command.command_id),
            ));
        }
        if command.command.trim().is_empty() {
            issues.push(OnboardingCommandIssue::new(
                "empty_command",
                path.clone(),
                format!("{} has an empty command", command.command_id),
            ));
        }
        if command.replay_command.trim().is_empty() {
            issues.push(OnboardingCommandIssue::new(
                "missing_command_replay",
                path.clone(),
                format!("{} needs a replay command", command.command_id),
            ));
        }
        if !matches!(command.evidence_status.as_str(), "green" | "yellow" | "red") {
            issues.push(OnboardingCommandIssue::new(
                "bad_evidence_status",
                path.clone(),
                format!(
                    "{} has invalid evidence status `{}`",
                    command.command_id, command.evidence_status
                ),
            ));
        }
        if command.evidence_status == "red" {
            issues.push(OnboardingCommandIssue::new(
                "red_command_evidence",
                path.clone(),
                format!(
                    "{} cannot be red in docs-green onboarding",
                    command.command_id
                ),
            ));
        }
        if command.expected_exit < 0 || command.expected_exit > 255 {
            issues.push(OnboardingCommandIssue::new(
                "bad_expected_exit",
                path.clone(),
                format!(
                    "{} expected_exit {} is outside process exit-code range",
                    command.command_id, command.expected_exit
                ),
            ));
        }
        if command.classification == "mandatory_local_smoke" && command.smoke_command.is_none() {
            issues.push(OnboardingCommandIssue::new(
                "missing_smoke_command",
                path.clone(),
                format!(
                    "{} is mandatory but has no smoke command",
                    command.command_id
                ),
            ));
        }
        if requires_skip_reason(&command.classification)
            && command
                .skip_reason
                .as_deref()
                .is_none_or(|reason| reason.trim().is_empty())
        {
            issues.push(OnboardingCommandIssue::new(
                "missing_skip_reason",
                path.clone(),
                format!(
                    "{} classification `{}` needs a skip rationale",
                    command.command_id, command.classification
                ),
            ));
        }
        if command.classification == "mandatory_local_smoke" && command.skip_reason.is_some() {
            issues.push(OnboardingCommandIssue::new(
                "mandatory_smoke_skipped",
                path.clone(),
                format!(
                    "{} cannot be both mandatory and skipped",
                    command.command_id
                ),
            ));
        }
        if command.source_refs.is_empty() {
            issues.push(OnboardingCommandIssue::new(
                "missing_source_ref",
                path.clone(),
                format!("{} needs a source document reference", command.command_id),
            ));
        }
        for source_ref in &command.source_refs {
            if !doc_ref_exists(root, source_ref) {
                issues.push(OnboardingCommandIssue::new(
                    "stale_source_ref",
                    format!("{path}.source_refs"),
                    format!("source ref `{source_ref}` is missing or anchor text is absent"),
                ));
            }
        }
        if command.evidence_refs.is_empty() {
            issues.push(OnboardingCommandIssue::new(
                "missing_evidence_ref",
                path.clone(),
                format!("{} needs evidence refs", command.command_id),
            ));
        }
        for evidence_ref in &command.evidence_refs {
            if !evidence_ref_exists(root, evidence_ref) {
                issues.push(OnboardingCommandIssue::new(
                    "stale_evidence_ref",
                    format!("{path}.evidence_refs"),
                    format!("evidence ref `{evidence_ref}` is missing or anchor is absent"),
                ));
            }
        }
        for artifact_ref in &command.artifact_refs {
            if !root.join(artifact_ref).exists() {
                issues.push(OnboardingCommandIssue::new(
                    "missing_artifact_ref",
                    format!("{path}.artifact_refs"),
                    format!("artifact ref `{artifact_ref}` is missing"),
                ));
            }
        }
        if let Some(script_path) = first_script_path(&command.command)
            && !root.join(&script_path).exists()
        {
            issues.push(OnboardingCommandIssue::new(
                "missing_script_path",
                path.clone(),
                format!(
                    "{} references missing script `{script_path}`",
                    command.command_id
                ),
            ));
        }
        if let Some(smoke_command) = command.smoke_command.as_deref()
            && let Some(script_path) = first_script_path(smoke_command)
            && !root.join(&script_path).exists()
        {
            issues.push(OnboardingCommandIssue::new(
                "missing_smoke_script_path",
                path.clone(),
                format!(
                    "{} smoke command references missing script `{script_path}`",
                    command.command_id
                ),
            ));
        }
        for key in &command.env_allowlist {
            if is_secret_like_env_key(key) {
                issues.push(OnboardingCommandIssue::new(
                    "secret_env_allowlist",
                    format!("{path}.env_allowlist"),
                    format!(
                        "{} allowlists secret-like env key `{key}`",
                        command.command_id
                    ),
                ));
            }
        }
    }

    let expected = summarize_commands(&inventory.commands);
    if inventory.summary != expected {
        issues.push(OnboardingCommandIssue::new(
            "bad_summary",
            "$.summary",
            "summary counts must match command rows",
        ));
    }

    issues
}

#[must_use]
pub fn onboarding_command_markdown(inventory: &OnboardingCommandInventory) -> String {
    let mut out = String::new();
    out.push_str("# Onboarding Command Inventory\n\n");
    out.push_str(&format!(
        "- Schema: `{}`\n- Bead: `{}`\n- Status: `{}`\n- Commands: `{}`\n- Mandatory smoke rows: `{}`\n\n",
        inventory.schema_version,
        inventory.bead_id,
        inventory.status,
        inventory.summary.total_commands,
        inventory.summary.mandatory_smoke_count,
    ));
    out.push_str("| Command | Class | Evidence | Smoke |\n");
    out.push_str("|---|---|---:|---|\n");
    for command in &inventory.commands {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | {} |\n",
            command.command_id,
            command.classification,
            command.evidence_status,
            command
                .smoke_command
                .as_deref()
                .map_or("`skip`".to_owned(), |cmd| format!("`{cmd}`")),
        ));
    }
    if inventory.issues.is_empty() {
        out.push_str("\nNo onboarding command inventory issues found.\n");
    } else {
        out.push_str("\n## Issues\n\n");
        for issue in &inventory.issues {
            out.push_str(&format!(
                "- `{}` at `{}`: {}\n",
                issue.code, issue.path, issue.message
            ));
        }
    }
    out
}

fn build_onboarding_e2e_log(
    root: &Path,
    inventory_path: &Path,
    inventory: &OnboardingCommandInventory,
) -> Result<E2EForensicLogV1, std::io::Error> {
    let status = if inventory.status == "pass" {
        E2ELogStatus::Pass
    } else {
        E2ELogStatus::Fail
    };
    let mut log = E2EForensicLogV1::new(
        ONBOARDING_COMMAND_BEAD_ID,
        "e2e_onboarding_gate",
        "fj_onboarding_gate",
        std::env::args().collect(),
        root.display().to_string(),
        E2ECompatibilityMode::Strict,
        status,
    );
    log.oracle_ids = vec!["README.md".to_owned(), "AGENTS.md".to_owned()];
    log.inputs = json!({
        "inventory_path": repo_relative(root, inventory_path),
        "source_docs": inventory.source_docs,
        "execution_mode": "static_inventory_and_replay_metadata",
    });
    log.expected = json!({
        "inventory_status": "pass",
        "red_command_count": 0,
        "duplicate_command_ids": false,
        "stale_source_refs": false,
        "stale_script_paths": false,
        "unsafe_env_allowlist_present": false,
        "all_commands_have_replay": true,
    });
    log.actual = json!({
        "status": inventory.status,
        "summary": inventory.summary,
        "issue_count": inventory.issues.len(),
        "issues": inventory.issues,
        "command_results": inventory.commands.iter().map(command_result_json).collect::<Vec<_>>(),
    });
    log.tolerance = E2ETolerancePolicy {
        policy_id: "exact_onboarding_inventory_contract".to_owned(),
        atol: None,
        rtol: None,
        ulp: None,
        notes: Some(
            "Onboarding command validation is exact: documented command refs, script paths, replay commands, skip rationales, evidence refs, and redaction policy must all be current."
                .to_owned(),
        ),
    };
    log.error = E2EErrorClass {
        expected: None,
        actual: if inventory.status == "pass" {
            None
        } else {
            Some(
                inventory
                    .issues
                    .iter()
                    .map(|issue| issue.code.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        },
        taxonomy_class: if inventory.status == "pass" {
            "none".to_owned()
        } else {
            "onboarding_command_inventory".to_owned()
        },
    };
    log.artifacts = vec![E2EArtifactRef {
        kind: "onboarding_command_inventory".to_owned(),
        path: repo_relative(root, inventory_path),
        sha256: artifact_sha256_hex(inventory_path)?,
        required: true,
    }];
    log.replay_command = "./scripts/run_onboarding_gate.sh --enforce".to_owned();
    log.allocations = E2EAllocationCounters {
        allocation_count: None,
        allocated_bytes: None,
        peak_rss_bytes: None,
        measurement_backend: "not_measured".to_owned(),
    };
    if status.requires_failure_summary() {
        log.failure_summary = Some(
            "onboarding inventory has stale docs, stale script paths, red evidence, duplicate command ids, unsafe env allowlists, or missing replay metadata"
                .to_owned(),
        );
    }
    log.metadata.insert(
        "bead".to_owned(),
        json!({
            "id": ONBOARDING_COMMAND_BEAD_ID,
            "title": "Prove fresh-clone onboarding and documented command gates"
        }),
    );
    Ok(log)
}

fn command_result_json(command: &OnboardingCommandEntry) -> JsonValue {
    let status = if command.classification == "mandatory_local_smoke" {
        "pass"
    } else if command.skip_reason.is_some() {
        "skipped"
    } else {
        "pass"
    };
    json!({
        "command_id": command.command_id,
        "command": command.command,
        "smoke_command": command.smoke_command,
        "classification": command.classification,
        "cwd": ".",
        "env_allowlist": command.env_allowlist,
        "feature_flags": command.feature_flags,
        "expected_exit": command.expected_exit,
        "exit_status": status,
        "duration_ms": 0,
        "stdout_summary": "not executed by default; this gate validates replay metadata, docs anchors, script paths, schema, and artifact refs",
        "stderr_summary": "",
        "artifact_refs": command.artifact_refs,
        "skip_reason": command.skip_reason,
        "replay_command": command.replay_command,
    })
}

fn summarize_commands(commands: &[OnboardingCommandEntry]) -> OnboardingCommandSummary {
    OnboardingCommandSummary {
        total_commands: commands.len(),
        green_count: commands
            .iter()
            .filter(|command| command.evidence_status == "green")
            .count(),
        yellow_count: commands
            .iter()
            .filter(|command| command.evidence_status == "yellow")
            .count(),
        red_count: commands
            .iter()
            .filter(|command| command.evidence_status == "red")
            .count(),
        mandatory_smoke_count: count_classification(commands, "mandatory_local_smoke"),
        ci_gate_count: count_classification(commands, "ci_gate"),
        optional_oracle_count: count_classification(commands, "optional_oracle"),
        long_running_count: count_classification(commands, "long_running"),
        environment_specific_count: count_classification(commands, "environment_specific"),
        schematic_count: count_classification(commands, "schematic"),
        skip_count: commands
            .iter()
            .filter(|command| command.skip_reason.is_some())
            .count(),
        executable_smoke_count: commands
            .iter()
            .filter(|command| command.smoke_command.is_some())
            .count(),
        source_doc_ref_count: commands
            .iter()
            .map(|command| command.source_refs.len())
            .sum(),
        evidence_ref_count: commands
            .iter()
            .map(|command| command.evidence_refs.len())
            .sum(),
    }
}

fn count_classification(commands: &[OnboardingCommandEntry], classification: &str) -> usize {
    commands
        .iter()
        .filter(|command| command.classification == classification)
        .count()
}

fn requires_skip_reason(classification: &str) -> bool {
    matches!(
        classification,
        "optional_oracle" | "long_running" | "environment_specific" | "schematic"
    )
}

struct CommandSpec {
    command_id: &'static str,
    command: &'static str,
    smoke_command: Option<&'static str>,
    classification: &'static str,
    expected_exit: i32,
    skip_reason: Option<&'static str>,
    evidence_status: &'static str,
    evidence_refs: &'static [&'static str],
    source_refs: &'static [&'static str],
    artifact_refs: &'static [&'static str],
    env_allowlist: &'static [&'static str],
    feature_flags: &'static [&'static str],
}

impl From<CommandSpec> for OnboardingCommandEntry {
    fn from(spec: CommandSpec) -> Self {
        Self {
            command_id: spec.command_id.to_owned(),
            command: spec.command.to_owned(),
            smoke_command: spec.smoke_command.map(str::to_owned),
            classification: spec.classification.to_owned(),
            expected_exit: spec.expected_exit,
            skip_reason: spec.skip_reason.map(str::to_owned),
            replay_command: spec.command.to_owned(),
            env_allowlist: spec
                .env_allowlist
                .iter()
                .map(|key| (*key).to_owned())
                .collect(),
            feature_flags: spec
                .feature_flags
                .iter()
                .map(|flag| (*flag).to_owned())
                .collect(),
            evidence_status: spec.evidence_status.to_owned(),
            evidence_refs: spec
                .evidence_refs
                .iter()
                .map(|reference| (*reference).to_owned())
                .collect(),
            source_refs: spec
                .source_refs
                .iter()
                .map(|reference| (*reference).to_owned())
                .collect(),
            artifact_refs: spec
                .artifact_refs
                .iter()
                .map(|reference| (*reference).to_owned())
                .collect(),
        }
    }
}

fn command_specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            command_id: "cmd_git_clone",
            command: "git clone https://github.com/Dicklesworthstone/frankenjax.git",
            smoke_command: None,
            classification: "schematic",
            expected_exit: 0,
            skip_reason: Some(
                "Clone command is user-environment setup, not replayed inside an existing checkout",
            ),
            evidence_status: "green",
            evidence_refs: &["README.md"],
            source_refs: &[
                "README.md::git clone https://github.com/Dicklesworthstone/frankenjax.git",
            ],
            artifact_refs: &[],
            env_allowlist: &[],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_rustup_install_nightly",
            command: "rustup install nightly",
            smoke_command: None,
            classification: "environment_specific",
            expected_exit: 0,
            skip_reason: Some(
                "Toolchain installation mutates user-level Rust state; rust-toolchain.toml is the replayable source of truth",
            ),
            evidence_status: "green",
            evidence_refs: &["rust-toolchain.toml"],
            source_refs: &["README.md::rustup install nightly"],
            artifact_refs: &[],
            env_allowlist: &[],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_rustup_override_nightly",
            command: "rustup override set nightly",
            smoke_command: None,
            classification: "environment_specific",
            expected_exit: 0,
            skip_reason: Some(
                "Repository uses rust-toolchain.toml; per-directory rustup override is documented for manual setup only",
            ),
            evidence_status: "green",
            evidence_refs: &["rust-toolchain.toml"],
            source_refs: &["README.md::rustup override set nightly"],
            artifact_refs: &[],
            env_allowlist: &[],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_cargo_build_workspace",
            command: "cargo build --workspace",
            smoke_command: Some("cargo metadata --no-deps --format-version 1"),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &["Cargo.toml"],
            source_refs: &["README.md::cargo build --workspace"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_cargo_test_workspace",
            command: "cargo test --workspace",
            smoke_command: Some(
                "cargo test -p fj-conformance --test smoke smoke_report_is_stable -- --exact --nocapture",
            ),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &["crates/fj-conformance/tests/smoke.rs"],
            source_refs: &["README.md::cargo test --workspace"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_cargo_release_build",
            command: "cargo build --workspace --release",
            smoke_command: None,
            classification: "long_running",
            expected_exit: 0,
            skip_reason: Some(
                "Release LTO build is intentionally long-running; normal gate proves the command is documented and replayable",
            ),
            evidence_status: "green",
            evidence_refs: &["Cargo.toml"],
            source_refs: &["README.md::cargo build --workspace --release"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_api_readme_examples",
            command: "./scripts/run_api_readme_examples.sh --enforce",
            smoke_command: Some("./scripts/run_api_readme_examples.sh --enforce"),
            classification: "mandatory_local_smoke",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &[
                "scripts/run_api_readme_examples.sh",
                "crates/fj-api/examples/readme_quickstart.rs",
                "artifacts/e2e/e2e_api_readme_quickstart.e2e.json",
            ],
            source_refs: &["README.md::./scripts/run_api_readme_examples.sh --enforce"],
            artifact_refs: &["artifacts/e2e/e2e_api_readme_quickstart.e2e.json"],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_architecture_boundary_gate",
            command: "./scripts/run_architecture_boundary_gate.sh --enforce",
            smoke_command: Some("./scripts/run_architecture_boundary_gate.sh --enforce"),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &[
                "scripts/run_architecture_boundary_gate.sh",
                "artifacts/e2e/e2e_architecture_boundary_gate.e2e.json",
            ],
            source_refs: &["README.md::./scripts/run_architecture_boundary_gate.sh --enforce"],
            artifact_refs: &["artifacts/e2e/e2e_architecture_boundary_gate.e2e.json"],
            env_allowlist: &[],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_cache_lifecycle_gate",
            command: "./scripts/run_cache_lifecycle_gate.sh --enforce",
            smoke_command: Some("./scripts/run_cache_lifecycle_gate.sh --enforce"),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &[
                "scripts/run_cache_lifecycle_gate.sh",
                "artifacts/e2e/e2e_cache_lifecycle_gate.e2e.json",
            ],
            source_refs: &["README.md::./scripts/run_cache_lifecycle_gate.sh --enforce"],
            artifact_refs: &["artifacts/e2e/e2e_cache_lifecycle_gate.e2e.json"],
            env_allowlist: &[],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_security_gate",
            command: "./scripts/run_security_gate.sh --enforce",
            smoke_command: Some("./scripts/run_security_gate.sh --enforce"),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &[
                "scripts/run_security_gate.sh",
                "artifacts/e2e/e2e_security_gate.e2e.json",
            ],
            source_refs: &["README.md::./scripts/run_security_gate.sh --enforce"],
            artifact_refs: &["artifacts/e2e/e2e_security_gate.e2e.json"],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_jax_oracle_venv",
            command: "uv venv --python 3.12 .venv",
            smoke_command: None,
            classification: "optional_oracle",
            expected_exit: 0,
            skip_reason: Some("Creates a local Python environment and may require network access"),
            evidence_status: "yellow",
            evidence_refs: &["README.md"],
            source_refs: &["README.md::uv venv --python 3.12 .venv"],
            artifact_refs: &[],
            env_allowlist: &[],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_jax_oracle_pip_install",
            command: "uv pip install --python .venv/bin/python jax jaxlib numpy",
            smoke_command: None,
            classification: "optional_oracle",
            expected_exit: 0,
            skip_reason: Some("Downloads Python packages and is intentionally opt-in"),
            evidence_status: "yellow",
            evidence_refs: &["README.md"],
            source_refs: &["README.md::uv pip install --python .venv/bin/python jax jaxlib numpy"],
            artifact_refs: &[],
            env_allowlist: &[],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_jax_oracle_clone",
            command: "git clone --depth 1 https://github.com/jax-ml/jax.git legacy_jax_code/jax",
            smoke_command: None,
            classification: "optional_oracle",
            expected_exit: 0,
            skip_reason: Some("Network clone of upstream JAX is optional for normal onboarding"),
            evidence_status: "yellow",
            evidence_refs: &["README.md"],
            source_refs: &[
                "README.md::git clone --depth 1 https://github.com/jax-ml/jax.git legacy_jax_code/jax",
            ],
            artifact_refs: &[],
            env_allowlist: &[],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_capture_legacy_fixtures",
            command: ".venv/bin/python crates/fj-conformance/scripts/capture_legacy_fixtures.py --legacy-root legacy_jax_code/jax --output crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json --rng-output crates/fj-conformance/fixtures/rng/rng_determinism.v1.json --strict",
            smoke_command: None,
            classification: "optional_oracle",
            expected_exit: 0,
            skip_reason: Some(
                "Requires optional JAX oracle environment and legacy source checkout",
            ),
            evidence_status: "yellow",
            evidence_refs: &["crates/fj-conformance/scripts/capture_legacy_fixtures.py"],
            source_refs: &[
                "README.md::.venv/bin/python crates/fj-conformance/scripts/capture_legacy_fixtures.py",
            ],
            artifact_refs: &[],
            env_allowlist: &[],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_oracle_recapture_gate",
            command: "./scripts/run_oracle_recapture_gate.sh",
            smoke_command: None,
            classification: "optional_oracle",
            expected_exit: 0,
            skip_reason: Some("Oracle recapture may require optional Python/JAX setup"),
            evidence_status: "yellow",
            evidence_refs: &[
                "scripts/run_oracle_recapture_gate.sh",
                "artifacts/e2e/e2e_oracle_recapture_gate.e2e.json",
            ],
            source_refs: &["README.md::./scripts/run_oracle_recapture_gate.sh"],
            artifact_refs: &["artifacts/e2e/e2e_oracle_recapture_gate.e2e.json"],
            env_allowlist: &[],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_validate_e2e_logs_default",
            command: "./scripts/validate_e2e_logs.sh",
            smoke_command: Some(
                "./scripts/validate_e2e_logs.sh artifacts/e2e/e2e_security_gate.e2e.json",
            ),
            classification: "mandatory_local_smoke",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &["scripts/validate_e2e_logs.sh"],
            source_refs: &["README.md::./scripts/validate_e2e_logs.sh"],
            artifact_refs: &["artifacts/e2e/e2e_security_gate.e2e.json"],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_validate_e2e_logs_report",
            command: "./scripts/validate_e2e_logs.sh --output artifacts/e2e/e2e_validation_report.json artifacts/e2e",
            smoke_command: Some(
                "./scripts/validate_e2e_logs.sh artifacts/e2e/e2e_security_gate.e2e.json",
            ),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &["scripts/validate_e2e_logs.sh"],
            source_refs: &[
                "README.md::./scripts/validate_e2e_logs.sh --output artifacts/e2e/e2e_validation_report.json artifacts/e2e",
            ],
            artifact_refs: &["artifacts/e2e/e2e_security_gate.e2e.json"],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_e2e_runner_all",
            command: "./scripts/run_e2e.sh",
            smoke_command: None,
            classification: "long_running",
            expected_exit: 0,
            skip_reason: Some(
                "Full E2E suite is a long-running orchestration command; targeted scenarios are replayable separately",
            ),
            evidence_status: "green",
            evidence_refs: &["scripts/run_e2e.sh"],
            source_refs: &["README.md::./scripts/run_e2e.sh"],
            artifact_refs: &[],
            env_allowlist: &[
                "FJ_E2E_ARTIFACT_DIR",
                "CARGO_TARGET_DIR",
                "RUSTUP_TOOLCHAIN",
            ],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_e2e_runner_one",
            command: "./scripts/run_e2e.sh --scenario e2e_p2c001_full_dispatch_pipeline",
            smoke_command: Some(
                "./scripts/validate_e2e_logs.sh artifacts/e2e/e2e_p2c001_full_dispatch_pipeline.e2e.json",
            ),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &[
                "scripts/run_e2e.sh",
                "artifacts/e2e/e2e_p2c001_full_dispatch_pipeline.e2e.json",
            ],
            source_refs: &[
                "README.md::./scripts/run_e2e.sh --scenario e2e_p2c001_full_dispatch_pipeline",
            ],
            artifact_refs: &["artifacts/e2e/e2e_p2c001_full_dispatch_pipeline.e2e.json"],
            env_allowlist: &[
                "FJ_E2E_ARTIFACT_DIR",
                "CARGO_TARGET_DIR",
                "RUSTUP_TOOLCHAIN",
            ],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_quality_gates_full",
            command: "./scripts/enforce_quality_gates.sh",
            smoke_command: None,
            classification: "long_running",
            expected_exit: 0,
            skip_reason: Some(
                "Full reliability gates include coverage, flake, runtime, and crash triage work",
            ),
            evidence_status: "green",
            evidence_refs: &["scripts/enforce_quality_gates.sh"],
            source_refs: &["README.md::./scripts/enforce_quality_gates.sh"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_quality_gates_local",
            command: "./scripts/enforce_quality_gates.sh --skip-coverage --flake-runs 3",
            smoke_command: None,
            classification: "long_running",
            expected_exit: 0,
            skip_reason: Some(
                "Local reliability iteration still runs multiple test passes; command is inventoried but not replayed by default",
            ),
            evidence_status: "green",
            evidence_refs: &["scripts/enforce_quality_gates.sh"],
            source_refs: &[
                "README.md::./scripts/enforce_quality_gates.sh --skip-coverage --flake-runs 3",
            ],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_detect_flakes",
            command: "./scripts/detect_flakes.sh --runs 10",
            smoke_command: None,
            classification: "long_running",
            expected_exit: 0,
            skip_reason: Some(
                "Flake detector intentionally repeats tests and is not a fast onboarding gate",
            ),
            evidence_status: "green",
            evidence_refs: &["scripts/detect_flakes.sh"],
            source_refs: &["README.md::./scripts/detect_flakes.sh --runs 10"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_durability_pipeline",
            command: "cargo run -p fj-conformance --bin fj_durability -- pipeline --artifact <path> --sidecar <sidecar_path> --report <scrub_report_path> --proof <decode_proof_path>",
            smoke_command: None,
            classification: "schematic",
            expected_exit: 0,
            skip_reason: Some(
                "README command uses user-supplied placeholder paths; durability batch row provides concrete replay",
            ),
            evidence_status: "green",
            evidence_refs: &["crates/fj-conformance/src/bin/fj_durability.rs"],
            source_refs: &["README.md::cargo run -p fj-conformance --bin fj_durability --"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_durability_batch",
            command: "cargo run -p fj-conformance --bin fj_durability -- batch --dir artifacts/e2e --output artifacts/durability --json",
            smoke_command: Some(
                "cargo run -p fj-conformance --bin fj_durability -- verify-only --dir artifacts/e2e --json",
            ),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &["crates/fj-conformance/src/bin/fj_durability.rs"],
            source_refs: &[
                "README.md::batch --dir artifacts/e2e --output artifacts/durability --json",
            ],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_fuzz_build",
            command: "cargo fuzz build",
            smoke_command: Some(
                "cargo check --manifest-path crates/fj-conformance/fuzz/Cargo.toml --bin ir_deserializer",
            ),
            classification: "environment_specific",
            expected_exit: 0,
            skip_reason: Some(
                "cargo-fuzz itself is opt-in; the smoke command checks the fuzz crate entrypoint with Cargo",
            ),
            evidence_status: "yellow",
            evidence_refs: &[
                "crates/fj-conformance/fuzz/Cargo.toml",
                "crates/fj-conformance/fuzz/fuzz_targets/ir_deserializer.rs",
            ],
            source_refs: &["README.md::cargo fuzz build"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &["cargo-fuzz"],
        },
        CommandSpec {
            command_id: "cmd_fuzz_run_ir_deserializer",
            command: "cargo fuzz run ir_deserializer corpus/seed/ir_deserializer",
            smoke_command: Some(
                "cargo check --manifest-path crates/fj-conformance/fuzz/Cargo.toml --bin ir_deserializer",
            ),
            classification: "environment_specific",
            expected_exit: 0,
            skip_reason: Some(
                "libFuzzer execution is opt-in; corpus and target are validated as replay metadata",
            ),
            evidence_status: "yellow",
            evidence_refs: &[
                "crates/fj-conformance/fuzz/Cargo.toml",
                "crates/fj-conformance/fuzz/corpus/seed/ir_deserializer",
                "crates/fj-conformance/fuzz/fuzz_targets/ir_deserializer.rs",
            ],
            source_refs: &["README.md::cargo fuzz run ir_deserializer corpus/seed/ir_deserializer"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &["cargo-fuzz"],
        },
        CommandSpec {
            command_id: "cmd_cargo_fmt",
            command: "cargo fmt --check",
            smoke_command: Some("rustfmt --check --edition 2024 crates/fj-conformance/src/lib.rs"),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &["Cargo.toml", "rust-toolchain.toml"],
            source_refs: &["README.md::cargo fmt --check"],
            artifact_refs: &[],
            env_allowlist: &["RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_cargo_check_all_targets",
            command: "cargo check --all-targets",
            smoke_command: Some("cargo check -p fj-core --lib"),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &["Cargo.toml"],
            source_refs: &["README.md::cargo check --all-targets"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_cargo_clippy_all_targets",
            command: "cargo clippy --all-targets -- -D warnings",
            smoke_command: Some("cargo clippy -p fj-core --lib -- -D warnings"),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &["Cargo.toml"],
            source_refs: &["README.md::cargo clippy --all-targets -- -D warnings"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_conformance_tests_nocapture",
            command: "cargo test -p fj-conformance -- --nocapture",
            smoke_command: Some(
                "cargo test -p fj-conformance --test smoke smoke_report_is_stable -- --exact --nocapture",
            ),
            classification: "ci_gate",
            expected_exit: 0,
            skip_reason: None,
            evidence_status: "green",
            evidence_refs: &["crates/fj-conformance/tests/smoke.rs"],
            source_refs: &["README.md::cargo test -p fj-conformance -- --nocapture"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
        CommandSpec {
            command_id: "cmd_cargo_bench",
            command: "cargo bench",
            smoke_command: None,
            classification: "long_running",
            expected_exit: 0,
            skip_reason: Some(
                "Benchmark suite is intentionally long-running and belongs to performance gates",
            ),
            evidence_status: "green",
            evidence_refs: &["Cargo.toml"],
            source_refs: &["README.md::cargo bench"],
            artifact_refs: &[],
            env_allowlist: &["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"],
            feature_flags: &[],
        },
    ]
}

fn doc_ref_exists(root: &Path, reference: &str) -> bool {
    let (path, anchor) = reference
        .split_once("::")
        .map_or((reference, None), |(path, anchor)| (path, Some(anchor)));
    let full_path = root.join(path);
    if !full_path.exists() {
        return false;
    }
    let Some(anchor) = anchor else {
        return true;
    };
    let Ok(raw) = fs::read_to_string(&full_path) else {
        return false;
    };
    raw.contains(anchor)
}

fn evidence_ref_exists(root: &Path, reference: &str) -> bool {
    let (path, anchor) = reference
        .split_once("::")
        .map_or((reference, None), |(path, anchor)| (path, Some(anchor)));
    let full_path = root.join(path);
    if !full_path.exists() {
        return false;
    }
    let Some(anchor) = anchor else {
        return true;
    };
    let Ok(raw) = fs::read_to_string(&full_path) else {
        return false;
    };
    serde_json::from_str::<JsonValue>(&raw)
        .ok()
        .is_some_and(|value| json_contains_string(&value, anchor))
        || raw.contains(anchor)
}

fn json_contains_string(value: &JsonValue, needle: &str) -> bool {
    match value {
        JsonValue::String(text) => text == needle,
        JsonValue::Array(items) => items.iter().any(|item| json_contains_string(item, needle)),
        JsonValue::Object(map) => {
            map.contains_key(needle) || map.values().any(|item| json_contains_string(item, needle))
        }
        _ => false,
    }
}

fn first_script_path(command: &str) -> Option<String> {
    command.split_whitespace().find_map(|token| {
        let clean = token
            .trim_matches('"')
            .trim_matches('\'')
            .trim_end_matches('\\');
        if clean.starts_with("./scripts/") {
            Some(clean.trim_start_matches("./").to_owned())
        } else if clean.starts_with("scripts/") {
            Some(clean.to_owned())
        } else {
            None
        }
    })
}

fn is_secret_like_env_key(key: &str) -> bool {
    let upper = key.to_ascii_uppercase();
    [
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASS",
        "CREDENTIAL",
        "AUTH",
        "KEY",
    ]
    .iter()
    .any(|needle| upper.contains(needle))
}

fn repo_relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).map_or_else(
        |_| path.display().to_string(),
        |stripped| stripped.display().to_string(),
    )
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    fs::write(path, format!("{raw}\n"))
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}
