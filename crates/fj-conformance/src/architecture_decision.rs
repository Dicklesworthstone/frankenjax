#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::Command;

pub const ARCHITECTURE_BOUNDARY_REPORT_SCHEMA_VERSION: &str =
    "frankenjax.architecture-boundary-report.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrateLayer {
    Model,
    Semantics,
    Execution,
    FacadeBackend,
    OpsHarness,
}

impl CrateLayer {
    const fn rank(self) -> u8 {
        match self {
            Self::Model => 0,
            Self::Semantics => 1,
            Self::Execution => 2,
            Self::FacadeBackend => 3,
            Self::OpsHarness => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryDecisionKind {
    KeepCurrentBoundary,
    DeferExtraction,
    ExtractNow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceCrate {
    pub name: String,
    pub manifest_path: String,
    pub lib_path: Option<String>,
    pub layer: CrateLayer,
    pub normal_deps: Vec<String>,
    pub dev_deps: Vec<String>,
    pub target_kinds: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceEdge {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedBoundaryOption {
    pub option_id: String,
    pub reason: String,
    pub user_impact: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryDecision {
    pub boundary_id: String,
    pub title: String,
    pub owner_crates: Vec<String>,
    pub current_state: String,
    pub decision: BoundaryDecisionKind,
    pub user_outcome: String,
    pub dependency_graph_impact: String,
    pub compile_time_impact: String,
    pub public_api_impact: String,
    pub testability_impact: String,
    pub rejected_options: Vec<RejectedBoundaryOption>,
    pub guardrails: Vec<String>,
    pub follow_up_beads: Vec<String>,
    pub revisit_when: Vec<String>,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchitectureBoundaryIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl ArchitectureBoundaryIssue {
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchitectureBoundaryReport {
    pub schema_version: String,
    pub bead_id: String,
    pub status: String,
    pub workspace_root: String,
    pub crate_count: usize,
    pub workspace_crates: Vec<WorkspaceCrate>,
    pub normal_edges: Vec<WorkspaceEdge>,
    pub decisions: Vec<BoundaryDecision>,
    pub issues: Vec<ArchitectureBoundaryIssue>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceSnapshot {
    pub root: String,
    pub crates: Vec<WorkspaceCrate>,
    pub normal_edges: Vec<WorkspaceEdge>,
}

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    packages: Vec<CargoPackage>,
    workspace_members: Vec<String>,
    workspace_root: String,
}

#[derive(Debug, Deserialize)]
struct CargoPackage {
    name: String,
    id: String,
    manifest_path: String,
    dependencies: Vec<CargoDependency>,
    targets: Vec<CargoTarget>,
}

#[derive(Debug, Deserialize)]
struct CargoDependency {
    name: String,
    kind: Option<String>,
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CargoTarget {
    kind: Vec<String>,
    src_path: String,
}

pub fn capture_workspace_snapshot(root: &Path) -> Result<WorkspaceSnapshot, String> {
    let output = Command::new("cargo")
        .arg("metadata")
        .arg("--no-deps")
        .arg("--format-version")
        .arg("1")
        .current_dir(root)
        .output()
        .map_err(|err| format!("failed to run cargo metadata: {err}"))?;

    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed with status {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    snapshot_from_metadata_json(&String::from_utf8_lossy(&output.stdout))
}

pub fn snapshot_from_metadata_json(raw: &str) -> Result<WorkspaceSnapshot, String> {
    let metadata = serde_json::from_str::<CargoMetadata>(raw)
        .map_err(|err| format!("cargo metadata JSON did not parse: {err}"))?;
    let member_ids = metadata
        .workspace_members
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let member_names = metadata
        .packages
        .iter()
        .filter(|package| member_ids.contains(&package.id))
        .map(|package| package.name.clone())
        .collect::<BTreeSet<_>>();

    let mut crates = Vec::new();
    let mut normal_edges = Vec::new();
    for package in metadata
        .packages
        .iter()
        .filter(|package| member_ids.contains(&package.id))
    {
        let mut normal_deps = Vec::new();
        let mut dev_deps = Vec::new();
        for dep in &package.dependencies {
            if dep.path.is_none() || !member_names.contains(&dep.name) {
                continue;
            }
            if dep.kind.as_deref() == Some("dev") {
                dev_deps.push(dep.name.clone());
            } else {
                normal_edges.push(WorkspaceEdge {
                    from: package.name.clone(),
                    to: dep.name.clone(),
                });
                normal_deps.push(dep.name.clone());
            }
        }
        normal_deps.sort();
        normal_deps.dedup();
        dev_deps.sort();
        dev_deps.dedup();

        let lib_path = package
            .targets
            .iter()
            .find(|target| target.kind.iter().any(|kind| kind == "lib"))
            .map(|target| target.src_path.clone());
        let mut target_kinds = package
            .targets
            .iter()
            .flat_map(|target| target.kind.iter().cloned())
            .collect::<Vec<_>>();
        target_kinds.sort();
        target_kinds.dedup();

        crates.push(WorkspaceCrate {
            name: package.name.clone(),
            manifest_path: package.manifest_path.clone(),
            lib_path,
            layer: layer_for_crate(&package.name)
                .ok_or_else(|| format!("workspace crate `{}` has no layer policy", package.name))?,
            normal_deps,
            dev_deps,
            target_kinds,
        });
    }
    crates.sort_by(|left, right| left.name.cmp(&right.name));
    normal_edges.sort_by(|left, right| left.from.cmp(&right.from).then(left.to.cmp(&right.to)));

    Ok(WorkspaceSnapshot {
        root: metadata.workspace_root,
        crates,
        normal_edges,
    })
}

#[must_use]
pub fn build_architecture_boundary_report(
    snapshot: WorkspaceSnapshot,
) -> ArchitectureBoundaryReport {
    let mut report = ArchitectureBoundaryReport {
        schema_version: ARCHITECTURE_BOUNDARY_REPORT_SCHEMA_VERSION.to_owned(),
        bead_id: "frankenjax-cstq.12".to_owned(),
        status: "pass".to_owned(),
        workspace_root: snapshot.root,
        crate_count: snapshot.crates.len(),
        workspace_crates: snapshot.crates,
        normal_edges: snapshot.normal_edges,
        decisions: boundary_decisions(),
        issues: Vec::new(),
        replay_command: "./scripts/run_architecture_boundary_gate.sh --enforce".to_owned(),
    };
    report.issues = validate_architecture_boundary_report(&report);
    if !report.issues.is_empty() {
        report.status = "fail".to_owned();
    }
    report
}

pub fn build_architecture_boundary_report_from_root(
    root: &Path,
) -> Result<ArchitectureBoundaryReport, String> {
    capture_workspace_snapshot(root).map(build_architecture_boundary_report)
}

pub fn write_architecture_boundary_outputs(
    root: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> Result<ArchitectureBoundaryReport, std::io::Error> {
    let report =
        build_architecture_boundary_report_from_root(root).map_err(std::io::Error::other)?;
    write_json(report_path, &report)?;
    write_markdown(markdown_path, &architecture_boundary_markdown(&report))?;
    Ok(report)
}

#[must_use]
pub fn architecture_boundary_summary_json(report: &ArchitectureBoundaryReport) -> Value {
    json!({
        "status": report.status,
        "crate_count": report.crate_count,
        "normal_edge_count": report.normal_edges.len(),
        "decision_count": report.decisions.len(),
        "issue_count": report.issues.len(),
        "decisions": report.decisions.iter().map(|decision| {
            json!({
                "boundary_id": decision.boundary_id,
                "decision": decision.decision,
                "owners": decision.owner_crates,
                "status": decision.status,
            })
        }).collect::<Vec<_>>(),
    })
}

#[must_use]
pub fn architecture_boundary_markdown(report: &ArchitectureBoundaryReport) -> String {
    let mut out = String::new();
    out.push_str("# Architecture Boundary Decision\n\n");
    out.push_str(&format!(
        "- schema: `{}`\n- bead: `{}`\n- status: `{}`\n- crates: `{}`\n- normal edges: `{}`\n\n",
        report.schema_version,
        report.bead_id,
        report.status,
        report.crate_count,
        report.normal_edges.len()
    ));
    out.push_str("| Boundary | Decision | Owners | User outcome |\n");
    out.push_str("|---|---|---|---|\n");
    for decision in &report.decisions {
        out.push_str(&format!(
            "| `{}` | `{:?}` | `{}` | {} |\n",
            decision.boundary_id,
            decision.decision,
            decision.owner_crates.join(", "),
            decision.user_outcome.replace('|', "/")
        ));
    }
    out.push_str("\n## Issues\n\n");
    if report.issues.is_empty() {
        out.push_str("No architecture boundary issues found.\n");
    } else {
        for issue in &report.issues {
            out.push_str(&format!(
                "- `{}` at `{}`: {}\n",
                issue.code, issue.path, issue.message
            ));
        }
    }
    out
}

#[must_use]
pub fn validate_architecture_boundary_report(
    report: &ArchitectureBoundaryReport,
) -> Vec<ArchitectureBoundaryIssue> {
    let mut issues = Vec::new();
    validate_schema(report, &mut issues);
    validate_required_crates(report, &mut issues);
    validate_dependency_graph(report, &mut issues);
    validate_decisions(report, &mut issues);
    issues
}

fn validate_schema(
    report: &ArchitectureBoundaryReport,
    issues: &mut Vec<ArchitectureBoundaryIssue>,
) {
    if report.schema_version != ARCHITECTURE_BOUNDARY_REPORT_SCHEMA_VERSION {
        issues.push(ArchitectureBoundaryIssue::new(
            "unsupported_schema_version",
            "$.schema_version",
            format!(
                "expected {}, got {}",
                ARCHITECTURE_BOUNDARY_REPORT_SCHEMA_VERSION, report.schema_version
            ),
        ));
    }
    if report.bead_id != "frankenjax-cstq.12" {
        issues.push(ArchitectureBoundaryIssue::new(
            "wrong_bead_id",
            "$.bead_id",
            "architecture boundary gate must remain bound to frankenjax-cstq.12",
        ));
    }
}

fn validate_required_crates(
    report: &ArchitectureBoundaryReport,
    issues: &mut Vec<ArchitectureBoundaryIssue>,
) {
    let names = report
        .workspace_crates
        .iter()
        .map(|krate| krate.name.as_str())
        .collect::<BTreeSet<_>>();
    for required in REQUIRED_CRATES {
        if !names.contains(required) {
            issues.push(ArchitectureBoundaryIssue::new(
                "missing_required_crate",
                "$.workspace_crates",
                format!("required architecture crate `{required}` is absent"),
            ));
        }
    }
}

fn validate_dependency_graph(
    report: &ArchitectureBoundaryReport,
    issues: &mut Vec<ArchitectureBoundaryIssue>,
) {
    let layer_by_name = report
        .workspace_crates
        .iter()
        .map(|krate| (krate.name.as_str(), krate.layer))
        .collect::<BTreeMap<_, _>>();

    for edge in &report.normal_edges {
        if edge.to == "fj-conformance" && edge.from != "fj-conformance" {
            issues.push(ArchitectureBoundaryIssue::new(
                "production_depends_on_conformance",
                format!("$.normal_edges[{}->{}]", edge.from, edge.to),
                "production crates must not depend on the conformance harness",
            ));
        }

        let Some(from_layer) = layer_by_name.get(edge.from.as_str()) else {
            issues.push(ArchitectureBoundaryIssue::new(
                "edge_from_unknown_crate",
                "$.normal_edges",
                format!("edge starts from unknown crate `{}`", edge.from),
            ));
            continue;
        };
        let Some(to_layer) = layer_by_name.get(edge.to.as_str()) else {
            issues.push(ArchitectureBoundaryIssue::new(
                "edge_to_unknown_crate",
                "$.normal_edges",
                format!("edge points to unknown crate `{}`", edge.to),
            ));
            continue;
        };
        if from_layer.rank() < to_layer.rank() {
            issues.push(ArchitectureBoundaryIssue::new(
                "layer_violation",
                format!("$.normal_edges[{}->{}]", edge.from, edge.to),
                format!(
                    "lower layer {:?} crate `{}` depends on higher layer {:?} crate `{}`",
                    from_layer, edge.from, to_layer, edge.to
                ),
            ));
        }
    }

    if let Some(cycle) = first_cycle(&report.workspace_crates, &report.normal_edges) {
        issues.push(ArchitectureBoundaryIssue::new(
            "dependency_cycle",
            "$.normal_edges",
            format!(
                "normal dependency graph has a cycle: {}",
                cycle.join(" -> ")
            ),
        ));
    }
}

fn validate_decisions(
    report: &ArchitectureBoundaryReport,
    issues: &mut Vec<ArchitectureBoundaryIssue>,
) {
    let crate_names = report
        .workspace_crates
        .iter()
        .map(|krate| krate.name.as_str())
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();

    for (idx, decision) in report.decisions.iter().enumerate() {
        let path = format!("$.decisions[{idx}]");
        if !seen.insert(decision.boundary_id.as_str()) {
            issues.push(ArchitectureBoundaryIssue::new(
                "duplicate_boundary_id",
                format!("{path}.boundary_id"),
                format!(
                    "boundary id `{}` appears more than once",
                    decision.boundary_id
                ),
            ));
        }
        if decision.owner_crates.is_empty() {
            issues.push(ArchitectureBoundaryIssue::new(
                "missing_owner_crates",
                format!("{path}.owner_crates"),
                "each boundary decision needs at least one owner crate",
            ));
        }
        for owner in &decision.owner_crates {
            if !crate_names.contains(owner.as_str()) {
                issues.push(ArchitectureBoundaryIssue::new(
                    "owner_crate_missing",
                    format!("{path}.owner_crates"),
                    format!("owner crate `{owner}` is not present in the workspace"),
                ));
            }
        }
        if decision.rejected_options.is_empty() {
            issues.push(ArchitectureBoundaryIssue::new(
                "missing_rejected_options",
                format!("{path}.rejected_options"),
                "decision must record rejected alternatives so users can audit tradeoffs",
            ));
        }
        if decision.guardrails.is_empty() {
            issues.push(ArchitectureBoundaryIssue::new(
                "missing_guardrails",
                format!("{path}.guardrails"),
                "decision must include guardrails that keep the boundary stable",
            ));
        }
        if decision.decision == BoundaryDecisionKind::ExtractNow
            && decision.follow_up_beads.is_empty()
        {
            issues.push(ArchitectureBoundaryIssue::new(
                "extract_now_without_beads",
                format!("{path}.follow_up_beads"),
                "extract-now decisions require child beads for implementation slices",
            ));
        }
        if decision.decision == BoundaryDecisionKind::DeferExtraction
            && decision.revisit_when.is_empty()
        {
            issues.push(ArchitectureBoundaryIssue::new(
                "defer_without_revisit_condition",
                format!("{path}.revisit_when"),
                "deferred extractions need explicit revisit conditions",
            ));
        }
    }
}

fn first_cycle(crates: &[WorkspaceCrate], edges: &[WorkspaceEdge]) -> Option<Vec<String>> {
    let mut graph = BTreeMap::<&str, Vec<&str>>::new();
    for krate in crates {
        graph.entry(krate.name.as_str()).or_default();
    }
    for edge in edges {
        graph
            .entry(edge.from.as_str())
            .or_default()
            .push(edge.to.as_str());
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut stack = Vec::new();
    for node in graph.keys().copied() {
        if let Some(cycle) = dfs_cycle(node, &graph, &mut visiting, &mut visited, &mut stack) {
            return Some(cycle);
        }
    }
    None
}

fn dfs_cycle<'a>(
    node: &'a str,
    graph: &BTreeMap<&'a str, Vec<&'a str>>,
    visiting: &mut BTreeSet<&'a str>,
    visited: &mut BTreeSet<&'a str>,
    stack: &mut Vec<&'a str>,
) -> Option<Vec<String>> {
    if visited.contains(node) {
        return None;
    }
    if visiting.contains(node) {
        let start = stack.iter().position(|item| *item == node).unwrap_or(0);
        let mut cycle = stack[start..]
            .iter()
            .map(|item| (*item).to_owned())
            .collect::<Vec<_>>();
        cycle.push(node.to_owned());
        return Some(cycle);
    }

    visiting.insert(node);
    stack.push(node);
    if let Some(next_nodes) = graph.get(node) {
        for next in next_nodes {
            if let Some(cycle) = dfs_cycle(next, graph, visiting, visited, stack) {
                return Some(cycle);
            }
        }
    }
    stack.pop();
    visiting.remove(node);
    visited.insert(node);
    None
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    fs::write(path, format!("{raw}\n"))
}

fn write_markdown(path: &Path, markdown: &str) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, markdown)
}

fn layer_for_crate(name: &str) -> Option<CrateLayer> {
    Some(match name {
        "fj-core" | "fj-test-utils" => CrateLayer::Model,
        "fj-trace" | "fj-lax" | "fj-cache" | "fj-ledger" => CrateLayer::Semantics,
        "fj-interpreters" | "fj-ad" | "fj-runtime" | "fj-egraph" => CrateLayer::Execution,
        "fj-dispatch" | "fj-api" | "fj-backend-cpu" | "fj-ffi" => CrateLayer::FacadeBackend,
        "fj-conformance" => CrateLayer::OpsHarness,
        _ => return None,
    })
}

const REQUIRED_CRATES: &[&str] = &[
    "fj-core",
    "fj-trace",
    "fj-lax",
    "fj-interpreters",
    "fj-ad",
    "fj-cache",
    "fj-ledger",
    "fj-runtime",
    "fj-egraph",
    "fj-dispatch",
    "fj-backend-cpu",
    "fj-api",
    "fj-ffi",
    "fj-conformance",
    "fj-test-utils",
];

fn boundary_decisions() -> Vec<BoundaryDecision> {
    vec![
        BoundaryDecision {
            boundary_id: "user_api_facade".to_owned(),
            title: "User API facade".to_owned(),
            owner_crates: vec!["fj-api".to_owned(), "fj-trace".to_owned()],
            current_state:
                "fj-api is the V1 user-facing transform facade for jit, grad, vmap, value_and_grad, jacobian, hessian, and make_jaxpr."
                    .to_owned(),
            decision: BoundaryDecisionKind::KeepCurrentBoundary,
            user_outcome:
                "Users get one explicit Rust transform facade now; no extra compatibility wrapper or package rename is introduced in this bead."
                    .to_owned(),
            dependency_graph_impact:
                "fj-api depends only on trace/core/dispatch/ad plus dev-only test helpers, so it remains above semantic crates without reverse edges."
                    .to_owned(),
            compile_time_impact:
                "Keeping the facade avoids adding another workspace crate and keeps incremental compile churn lower while public examples are still being proven."
                    .to_owned(),
            public_api_impact:
                "The public import path remains fj_api::* for V1 evidence; frankenjax package naming is deferred until release packaging work."
                    .to_owned(),
            testability_impact:
                "Facade tests stay in fj-api and public example replay is owned by frankenjax-cstq.9."
                    .to_owned(),
            rejected_options: vec![RejectedBoundaryOption {
                option_id: "new_frankenjax_wrapper_crate".to_owned(),
                reason:
                    "A wrapper crate would add a second user import surface before README examples are fully replay-proven."
                        .to_owned(),
                user_impact:
                    "Avoids confusing users with two equivalent facade crates and no extra functionality."
                        .to_owned(),
            }],
            guardrails: vec![
                "fj-api must not be a normal dependency of lower-layer semantic crates".to_owned(),
                "public README examples must replay through fj-api under frankenjax-cstq.9".to_owned(),
            ],
            follow_up_beads: vec!["frankenjax-cstq.9".to_owned()],
            revisit_when: Vec::new(),
            status: "accepted".to_owned(),
        },
        BoundaryDecision {
            boundary_id: "transform_stack".to_owned(),
            title: "Transform stack ownership".to_owned(),
            owner_crates: vec![
                "fj-core".to_owned(),
                "fj-dispatch".to_owned(),
                "fj-ad".to_owned(),
                "fj-trace".to_owned(),
                "fj-api".to_owned(),
            ],
            current_state:
                "Transform proof types live in fj-core, tracing in fj-trace, AD in fj-ad, orchestration in fj-dispatch, and public wrappers in fj-api."
                    .to_owned(),
            decision: BoundaryDecisionKind::DeferExtraction,
            user_outcome:
                "Users keep current transform behavior while parity work closes advanced control-flow composition gaps."
                    .to_owned(),
            dependency_graph_impact:
                "A dedicated fj-transforms crate would currently pull together core, trace, AD, dispatch, and API responsibilities and risks new cycles."
                    .to_owned(),
            compile_time_impact:
                "Extraction would force broad rebuilds across the hottest crates without a measured performance or ergonomics win."
                    .to_owned(),
            public_api_impact:
                "No transform function moves until frankenjax-cstq.2 and frankenjax-cstq.9 prove the behavior and examples."
                    .to_owned(),
            testability_impact:
                "The safer near-term test seam is a conformance matrix over existing crates rather than a structural split."
                    .to_owned(),
            rejected_options: vec![RejectedBoundaryOption {
                option_id: "extract_fj_transforms_now".to_owned(),
                reason:
                    "The advanced transform/control-flow matrix is still open, so extraction would mix semantics work with crate churn."
                        .to_owned(),
                user_impact:
                    "Deferral protects existing jit/grad/vmap behavior and avoids moving APIs while users still need parity evidence."
                        .to_owned(),
            }],
            guardrails: vec![
                "transform-order tests must remain in dispatch/conformance gates".to_owned(),
                "fj-core proof types must not depend on fj-dispatch or fj-api".to_owned(),
            ],
            follow_up_beads: vec!["frankenjax-cstq.2".to_owned(), "frankenjax-cstq.9".to_owned()],
            revisit_when: vec![
                "advanced transform-control-flow parity is green".to_owned(),
                "public README examples replay through fj-api".to_owned(),
            ],
            status: "accepted".to_owned(),
        },
        BoundaryDecision {
            boundary_id: "lowering_execution".to_owned(),
            title: "Lowering and execution boundary".to_owned(),
            owner_crates: vec![
                "fj-interpreters".to_owned(),
                "fj-dispatch".to_owned(),
                "fj-backend-cpu".to_owned(),
                "fj-runtime".to_owned(),
            ],
            current_state:
                "V1 lowers to canonical IR evaluated by interpreter/dispatch paths; fj-backend-cpu owns dependency-wave CPU execution and fj-runtime owns admission policy."
                    .to_owned(),
            decision: BoundaryDecisionKind::DeferExtraction,
            user_outcome:
                "Users keep the documented CPU-only interpreter/backend semantics without an invented XLA-like lowering layer."
                    .to_owned(),
            dependency_graph_impact:
                "A new fj-lowering crate would be mostly policy glue today and would increase cross-crate coupling with dispatch/backend."
                    .to_owned(),
            compile_time_impact:
                "Deferral avoids moving interpreter/backend code while performance and memory gates are still being expanded."
                    .to_owned(),
            public_api_impact:
                "No user-facing lowering API is exposed in V1; claims remain CPU/interpreter/backend scoped."
                    .to_owned(),
            testability_impact:
                "Execution behavior remains covered by interpreter, dispatch, backend, and conformance tests with explicit E2E logs."
                    .to_owned(),
            rejected_options: vec![RejectedBoundaryOption {
                option_id: "extract_fj_lowering_now".to_owned(),
                reason:
                    "There is no separate XLA/compiler lowering contract in V1; extracting a crate would formalize a surface users cannot use yet."
                        .to_owned(),
                user_impact:
                    "Avoids overpromising a compiler pipeline and keeps the README limitation honest."
                        .to_owned(),
            }],
            guardrails: vec![
                "README must continue to state no XLA lowering for V1".to_owned(),
                "backend execution must stay covered by conformance and performance gates".to_owned(),
            ],
            follow_up_beads: vec!["frankenjax-cstq.4".to_owned(), "frankenjax-cstq.18".to_owned()],
            revisit_when: vec![
                "a real compiler/lowering IR is introduced".to_owned(),
                "backend parity requires multiple execution backends".to_owned(),
            ],
            status: "accepted".to_owned(),
        },
        BoundaryDecision {
            boundary_id: "cpu_backend".to_owned(),
            title: "CPU backend boundary".to_owned(),
            owner_crates: vec!["fj-backend-cpu".to_owned(), "fj-runtime".to_owned()],
            current_state:
                "fj-backend-cpu is a dedicated CPU backend crate and fj-runtime owns backend registry/admission concerns."
                    .to_owned(),
            decision: BoundaryDecisionKind::KeepCurrentBoundary,
            user_outcome:
                "Users get a concrete always-available CPU backend without waiting for GPU/TPU abstractions."
                    .to_owned(),
            dependency_graph_impact:
                "The backend stays above core/interpreter/lax/runtime and below dispatch/api, preserving layered flow."
                    .to_owned(),
            compile_time_impact:
                "Keeping the backend crate isolated localizes Rayon-heavy backend rebuilds.".to_owned(),
            public_api_impact:
                "Backend internals remain implementation detail unless explicitly promoted later."
                    .to_owned(),
            testability_impact:
                "Backend scheduling can be tested and benchmarked independently from facade examples."
                    .to_owned(),
            rejected_options: vec![RejectedBoundaryOption {
                option_id: "fold_cpu_backend_into_dispatch".to_owned(),
                reason:
                    "Dispatch already coordinates transforms/cache/ledger; folding backend scheduling into it would enlarge the integration choke point."
                        .to_owned(),
                user_impact:
                    "Keeping it separate preserves focused performance gates and simpler failure diagnosis."
                        .to_owned(),
            }],
            guardrails: vec![
                "fj-backend-cpu must not depend on fj-api or fj-conformance".to_owned(),
                "backend performance changes require benchmark evidence".to_owned(),
            ],
            follow_up_beads: vec!["frankenjax-cstq.4".to_owned()],
            revisit_when: Vec::new(),
            status: "accepted".to_owned(),
        },
        BoundaryDecision {
            boundary_id: "ffi_boundary".to_owned(),
            title: "FFI boundary".to_owned(),
            owner_crates: vec!["fj-ffi".to_owned()],
            current_state: "fj-ffi isolates C ABI registration, buffers, calls, callbacks, and error propagation."
                .to_owned(),
            decision: BoundaryDecisionKind::KeepCurrentBoundary,
            user_outcome:
                "Users get one auditable native interop boundary and the rest of the workspace stays unsafe-free."
                    .to_owned(),
            dependency_graph_impact:
                "fj-ffi depends only on fj-core in normal builds, so native interop cannot pull execution or harness crates downward."
                    .to_owned(),
            compile_time_impact:
                "The FFI crate compiles independently from dispatch/conformance-heavy paths.".to_owned(),
            public_api_impact:
                "FFI symbols remain opt-in and do not affect the Rust transform facade.".to_owned(),
            testability_impact:
                "FFI adversarial tests remain localized and can be fuzzed without changing pure Rust crates."
                    .to_owned(),
            rejected_options: vec![RejectedBoundaryOption {
                option_id: "spread_ffi_into_runtime".to_owned(),
                reason:
                    "Putting native calls into runtime would weaken the single unsafe-boundary doctrine."
                        .to_owned(),
                user_impact: "Users retain a simpler security story and clearer audit surface.".to_owned(),
            }],
            guardrails: vec![
                "unsafe code remains isolated to fj-ffi test/FFI implementation surfaces".to_owned(),
                "production crates must not depend on fj-ffi unless an explicit interop feature is added".to_owned(),
            ],
            follow_up_beads: vec!["frankenjax-cstq.17".to_owned()],
            revisit_when: Vec::new(),
            status: "accepted".to_owned(),
        },
        BoundaryDecision {
            boundary_id: "conformance_harness".to_owned(),
            title: "Conformance and evidence harness".to_owned(),
            owner_crates: vec!["fj-conformance".to_owned()],
            current_state:
                "fj-conformance owns oracle fixtures, parity reports, durability tools, E2E forensic logs, and architecture gates."
                    .to_owned(),
            decision: BoundaryDecisionKind::KeepCurrentBoundary,
            user_outcome:
                "Users and agents get one evidence plane for replaying claims without production crates depending on the harness."
                    .to_owned(),
            dependency_graph_impact:
                "Conformance stays at the top layer and may depend on implementation crates, never the reverse."
                    .to_owned(),
            compile_time_impact:
                "Evidence binaries and tests can build separately from core library consumers.".to_owned(),
            public_api_impact:
                "Harness APIs are evidence infrastructure, not user-facing transform APIs.".to_owned(),
            testability_impact:
                "All child evidence lanes can share forensic log validation and artifact hashing."
                    .to_owned(),
            rejected_options: vec![RejectedBoundaryOption {
                option_id: "move_gates_into_each_production_crate".to_owned(),
                reason:
                    "Distributed gate binaries would fragment log schema enforcement and replay commands."
                        .to_owned(),
                user_impact:
                    "Centralizing the evidence plane gives users one place to inspect red/yellow/green status."
                        .to_owned(),
            }],
            guardrails: vec![
                "no non-conformance crate may depend normally on fj-conformance".to_owned(),
                "new evidence scripts must emit shared E2E forensic logs or an explicit adapter".to_owned(),
            ],
            follow_up_beads: vec![
                "frankenjax-cstq.15".to_owned(),
                "frankenjax-cstq.16".to_owned(),
                "frankenjax-cstq.18".to_owned(),
            ],
            revisit_when: Vec::new(),
            status: "accepted".to_owned(),
        },
    ]
}
