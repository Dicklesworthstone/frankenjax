#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{E2ELogStatus, validate_e2e_log_path};
use fj_conformance::onboarding_command::{
    ONBOARDING_COMMAND_BEAD_ID, ONBOARDING_COMMAND_INVENTORY_SCHEMA_VERSION,
    OnboardingCommandEntry, build_onboarding_command_inventory, onboarding_command_markdown,
    validate_onboarding_command_inventory, write_onboarding_command_outputs,
};
use std::collections::BTreeSet;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn onboarding_inventory_covers_required_command_families() {
    let root = repo_root();
    let inventory = build_onboarding_command_inventory(&root);
    assert_eq!(
        inventory.schema_version,
        ONBOARDING_COMMAND_INVENTORY_SCHEMA_VERSION
    );
    assert_eq!(inventory.bead_id, ONBOARDING_COMMAND_BEAD_ID);
    assert_eq!(inventory.status, "pass", "issues: {:#?}", inventory.issues);
    assert!(inventory.issues.is_empty());

    let ids = inventory
        .commands
        .iter()
        .map(|command| command.command_id.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        "cmd_cargo_build_workspace",
        "cmd_cargo_test_workspace",
        "cmd_api_readme_examples",
        "cmd_e2e_runner_one",
        "cmd_durability_batch",
        "cmd_fuzz_build",
        "cmd_validate_e2e_logs_default",
    ] {
        assert!(ids.contains(required), "missing {required}");
    }
}

#[test]
fn onboarding_inventory_rejects_stale_paths_and_missing_replay() {
    let root = repo_root();
    let mut inventory = build_onboarding_command_inventory(&root);
    inventory.commands[0].command_id = inventory.commands[1].command_id.clone();
    inventory.commands[0].replay_command.clear();
    inventory.commands[0].evidence_refs = vec!["missing/evidence.json".to_owned()];
    inventory.commands[0].source_refs = vec!["README.md::not a real command anchor".to_owned()];

    let issue_codes = validate_onboarding_command_inventory(&root, &inventory)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("duplicate_command_id"));
    assert!(issue_codes.contains("missing_command_replay"));
    assert!(issue_codes.contains("stale_evidence_ref"));
    assert!(issue_codes.contains("stale_source_ref"));
}

#[test]
fn onboarding_inventory_rejects_unjustified_skips_and_secret_envs() {
    let root = repo_root();
    let mut inventory = build_onboarding_command_inventory(&root);
    let oracle_index = inventory
        .commands
        .iter()
        .position(|command| command.command_id == "cmd_jax_oracle_venv");
    assert!(oracle_index.is_some(), "oracle setup command exists");
    if let Some(index) = oracle_index {
        inventory.commands[index].skip_reason = None;
        inventory.commands[index].env_allowlist = vec!["JAX_API_TOKEN".to_owned()];
    }

    let issue_codes = validate_onboarding_command_inventory(&root, &inventory)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("missing_skip_reason"));
    assert!(issue_codes.contains("secret_env_allowlist"));
}

#[test]
fn onboarding_inventory_rejects_summary_drift() {
    let root = repo_root();
    let mut inventory = build_onboarding_command_inventory(&root);
    inventory.summary.total_commands += 1;

    let issue_codes = validate_onboarding_command_inventory(&root, &inventory)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("bad_summary"));
}

#[test]
fn onboarding_outputs_write_and_validate_e2e_log() -> Result<(), String> {
    let root = repo_root();
    let temp = tempfile::tempdir().map_err(|err| err.to_string())?;
    let inventory_path = temp.path().join("onboarding_command_inventory.v1.json");
    let e2e_path = temp.path().join("e2e_onboarding_gate.e2e.json");
    let inventory = write_onboarding_command_outputs(&root, &inventory_path, &e2e_path)
        .map_err(|err| err.to_string())?;
    assert_eq!(inventory.status, "pass");

    let parsed =
        validate_e2e_log_path(&e2e_path, temp.path()).map_err(|issues| format!("{issues:#?}"))?;
    assert_eq!(parsed.status, E2ELogStatus::Pass);
    assert_eq!(parsed.bead_id, ONBOARDING_COMMAND_BEAD_ID);
    assert!(
        parsed.artifacts[0]
            .path
            .ends_with("onboarding_command_inventory.v1.json")
    );
    Ok(())
}

#[test]
fn onboarding_markdown_is_dashboard_ready() {
    let root = repo_root();
    let inventory = build_onboarding_command_inventory(&root);
    let markdown = onboarding_command_markdown(&inventory);
    assert!(markdown.contains("Onboarding Command Inventory"));
    assert!(markdown.contains("cmd_api_readme_examples"));
    assert!(markdown.contains("No onboarding command inventory issues found."));
}

#[test]
fn onboarding_command_rows_have_replay_and_safe_env_allowlists() {
    let root = repo_root();
    let inventory = build_onboarding_command_inventory(&root);
    for command in &inventory.commands {
        assert_valid_command_shape(command);
    }
}

fn assert_valid_command_shape(command: &OnboardingCommandEntry) {
    assert!(command.command_id.starts_with("cmd_"));
    assert!(!command.command.trim().is_empty());
    assert!(!command.replay_command.trim().is_empty());
    assert!(!command.source_refs.is_empty());
    assert!(!command.evidence_refs.is_empty());
    for key in &command.env_allowlist {
        let upper = key.to_ascii_uppercase();
        assert!(!upper.contains("TOKEN"));
        assert!(!upper.contains("SECRET"));
        assert!(!upper.contains("PASSWORD"));
    }
}
