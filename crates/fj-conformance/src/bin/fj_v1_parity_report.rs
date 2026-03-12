#![forbid(unsafe_code)]

use fj_conformance::{
    FamilyCaseEntry, FamilyReport, HarnessConfig, ParityReportSummary, ParityReportV1,
    read_transform_fixture_bundle, run_transform_fixture_bundle,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    fixtures: PathBuf,
    rng_fixtures: PathBuf,
    output_json: PathBuf,
    output_markdown: PathBuf,
    ci_json: PathBuf,
    e2e_json: PathBuf,
    mode: String,
    fj_version: String,
    oracle_version: String,
}

#[derive(Debug, Clone, Serialize)]
struct PrimitiveBreakdown {
    total: usize,
    matched: usize,
    mismatched: usize,
    cases: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ParityException {
    case_id: String,
    family: String,
    primitive: String,
    drift: String,
    expected: String,
    actual: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CoverageException {
    family: String,
    reason: String,
    justification: String,
}

#[derive(Debug, Clone, Serialize)]
struct ComprehensiveParityReport {
    version: String,
    timestamp: String,
    fj_version: String,
    jax_version: String,
    mode: String,
    summary: ParityReportSummary,
    families: BTreeMap<String, FamilyReport>,
    per_primitive: BTreeMap<String, PrimitiveBreakdown>,
    parity_exceptions: Vec<ParityException>,
    coverage_exceptions: Vec<CoverageException>,
    gate_status: String,
}

#[derive(Debug, Clone, Serialize)]
struct FamilyRollup {
    total: usize,
    matched: usize,
    mismatched: usize,
}

#[derive(Debug, Clone, Serialize)]
struct E2EFinalParityLog {
    total_cases: usize,
    matched: usize,
    mismatched: usize,
    pass_rate: f64,
    per_family: BTreeMap<String, FamilyRollup>,
    gate_status: String,
    pass: bool,
}

// ── RNG fixture types ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct RngFixtureBundle {
    #[allow(dead_code)]
    schema_version: String,
    cases: Vec<RngFixtureCase>,
}

#[derive(Debug, Deserialize)]
struct RngFixtureCase {
    case_id: String,
    operation: String,
    seed: u64,
    fold_in_data: Option<u32>,
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    expected_key_bits: Vec<u32>,
    #[serde(default)]
    expected_split_keys: Vec<Vec<u32>>,
    #[serde(default)]
    expected_values: Vec<f64>,
    #[serde(default)]
    minval: Option<f64>,
    #[serde(default)]
    maxval: Option<f64>,
    atol: f64,
    rtol: f64,
}

fn verify_rng_case(case: &RngFixtureCase) -> (bool, Option<String>) {
    use fj_lax::threefry::{
        random_fold_in, random_key, random_normal, random_split, random_uniform,
    };

    match case.operation.as_str() {
        "key" => {
            let actual = random_key(case.seed).0.to_vec();
            if actual == case.expected_key_bits {
                (true, None)
            } else {
                (
                    false,
                    Some(format!(
                        "key mismatch: expected {:?}, got {:?}",
                        case.expected_key_bits, actual
                    )),
                )
            }
        }
        "split" => {
            let (left, right) = random_split(random_key(case.seed));
            let actual = vec![left.0.to_vec(), right.0.to_vec()];
            if actual == case.expected_split_keys {
                (true, None)
            } else {
                (
                    false,
                    Some(format!(
                        "split mismatch: expected {:?}, got {:?}",
                        case.expected_split_keys, actual
                    )),
                )
            }
        }
        "fold_in" => {
            let data = case.fold_in_data.unwrap_or(0);
            let actual = random_fold_in(random_key(case.seed), data).0.to_vec();
            if actual == case.expected_key_bits {
                (true, None)
            } else {
                (
                    false,
                    Some(format!(
                        "fold_in mismatch: expected {:?}, got {:?}",
                        case.expected_key_bits, actual
                    )),
                )
            }
        }
        "uniform" => {
            let count = if case.shape.is_empty() {
                1
            } else {
                case.shape.iter().product()
            };
            let actual = random_uniform(
                random_key(case.seed),
                count,
                case.minval.unwrap_or(0.0),
                case.maxval.unwrap_or(1.0),
            );
            if actual.len() != case.expected_values.len() {
                return (
                    false,
                    Some(format!(
                        "uniform length mismatch: expected {}, got {}",
                        case.expected_values.len(),
                        actual.len()
                    )),
                );
            }
            for (i, (e, a)) in case.expected_values.iter().zip(actual.iter()).enumerate() {
                let diff = (e - a).abs();
                if diff > case.atol + case.rtol * e.abs() {
                    return (
                        false,
                        Some(format!(
                            "uniform[{i}] mismatch: expected {e}, got {a}, diff {diff}"
                        )),
                    );
                }
            }
            (true, None)
        }
        "normal" => {
            let count = if case.shape.is_empty() {
                1
            } else {
                case.shape.iter().product()
            };
            let actual = random_normal(random_key(case.seed), count);
            if actual.len() != case.expected_values.len() {
                return (
                    false,
                    Some(format!(
                        "normal length mismatch: expected {}, got {}",
                        case.expected_values.len(),
                        actual.len()
                    )),
                );
            }
            for (i, (e, a)) in case.expected_values.iter().zip(actual.iter()).enumerate() {
                let diff = (e - a).abs();
                if diff > case.atol + case.rtol * e.abs() {
                    return (
                        false,
                        Some(format!(
                            "normal[{i}] mismatch: expected {e}, got {a}, diff {diff}"
                        )),
                    );
                }
            }
            (true, None)
        }
        op => (false, Some(format!("unknown RNG operation: {op}"))),
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
        })
}

fn default_paths(root: &Path) -> Args {
    Args {
        fixtures: root
            .join("crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json"),
        rng_fixtures: root.join("crates/fj-conformance/fixtures/rng/rng_determinism.v1.json"),
        output_json: root.join("artifacts/conformance/v1_parity_report.json"),
        output_markdown: root.join("artifacts/conformance/v1_parity_report.md"),
        ci_json: root.join("artifacts/ci/runs/v1-parity-report/parity_report.v1.json"),
        e2e_json: root.join("artifacts/e2e/e2e_parity_report_final.e2e.json"),
        mode: "strict".to_owned(),
        fj_version: env!("CARGO_PKG_VERSION").to_owned(),
        oracle_version: "jax-0.9.0.1".to_owned(),
    }
}

fn usage() -> &'static str {
    "Usage:
  cargo run -p fj-conformance --bin fj_v1_parity_report -- [options]

Options:
  --fixtures <path>        Input transform fixture bundle JSON
  --rng-fixtures <path>    Input RNG determinism fixture bundle JSON
  --output-json <path>     Comprehensive parity JSON output
  --output-md <path>       Comprehensive parity markdown output
  --ci-json <path>         Spec Section 8.3 parity-report JSON output
  --e2e-json <path>        E2E forensic parity log output
  --mode <strict|hardened> Report mode label (default: strict)
  --fj-version <string>    FrankenJAX version in report metadata
  --oracle-version <str>   Oracle/JAX version in report metadata
  --help                   Show this help"
}

fn next_value<I>(args: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_args(root: &Path) -> Result<Args, String> {
    let mut parsed = default_paths(root);
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--fixtures" => parsed.fixtures = PathBuf::from(next_value(&mut it, "--fixtures")?),
            "--rng-fixtures" => {
                parsed.rng_fixtures = PathBuf::from(next_value(&mut it, "--rng-fixtures")?);
            }
            "--output-json" => {
                parsed.output_json = PathBuf::from(next_value(&mut it, "--output-json")?);
            }
            "--output-md" => {
                parsed.output_markdown = PathBuf::from(next_value(&mut it, "--output-md")?);
            }
            "--ci-json" => parsed.ci_json = PathBuf::from(next_value(&mut it, "--ci-json")?),
            "--e2e-json" => parsed.e2e_json = PathBuf::from(next_value(&mut it, "--e2e-json")?),
            "--mode" => parsed.mode = next_value(&mut it, "--mode")?,
            "--fj-version" => parsed.fj_version = next_value(&mut it, "--fj-version")?,
            "--oracle-version" => parsed.oracle_version = next_value(&mut it, "--oracle-version")?,
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unknown flag: {arg}\n\n{}", usage())),
        }
    }
    Ok(parsed)
}

fn family_name(name: &str) -> String {
    name.to_owned()
}

fn ensure_parent(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create directory {}: {err}", parent.display()))?;
    }
    Ok(())
}

fn write_text(path: &Path, content: &str) -> Result<(), String> {
    ensure_parent(path)?;
    fs::write(path, content).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn to_markdown(
    v1: &ParityReportV1,
    per_primitive: &BTreeMap<String, PrimitiveBreakdown>,
    parity_exceptions: &[ParityException],
    coverage_exceptions: &[CoverageException],
) -> String {
    let matched_total: usize = v1.families.values().map(|family| family.matched).sum();
    let mismatched_total: usize = v1.families.values().map(|family| family.mismatched).sum();

    let mut out = String::new();
    out.push_str("# V1 Parity Report\n\n");
    out.push_str(&format!("Mode: `{}`\n\n", v1.mode));
    out.push_str(&format!(
        "FrankenJAX: `{}` | Oracle: `{}`\n\n",
        v1.fj_version, v1.oracle_version
    ));
    out.push_str("## Summary\n\n");
    out.push_str("| Metric | Value |\n");
    out.push_str("|---|---|\n");
    out.push_str(&format!("| Total Cases | {} |\n", v1.summary.total));
    out.push_str(&format!("| Matched | {} |\n", matched_total));
    out.push_str(&format!("| Mismatched | {} |\n", mismatched_total));
    out.push_str(&format!(
        "| Pass Rate | {:.2}% |\n",
        v1.summary.pass_rate * 100.0
    ));
    out.push_str(&format!("| Gate | **{}** |\n\n", v1.summary.gate));

    out.push_str("## Per-Family Breakdown\n\n");
    out.push_str("| Family | Total | Matched | Mismatched |\n");
    out.push_str("|---|---|---|---|\n");
    for family in ["jit", "grad", "vmap", "lax", "random", "control_flow"] {
        let stats = v1.families.get(family).cloned().unwrap_or(FamilyReport {
            total: 0,
            matched: 0,
            mismatched: 0,
            cases: Vec::new(),
        });
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            family, stats.total, stats.matched, stats.mismatched
        ));
    }
    out.push('\n');

    out.push_str("## Per-Primitive Breakdown\n\n");
    out.push_str("| Primitive | Total | Matched | Mismatched |\n");
    out.push_str("|---|---|---|---|\n");
    for (primitive, stats) in per_primitive {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            primitive, stats.total, stats.matched, stats.mismatched
        ));
    }
    out.push('\n');

    out.push_str("## Coverage Exceptions\n\n");
    if coverage_exceptions.is_empty() {
        out.push_str("None.\n\n");
    } else {
        for exception in coverage_exceptions {
            out.push_str(&format!(
                "- `{}`: {} ({})\n",
                exception.family, exception.reason, exception.justification
            ));
        }
        out.push('\n');
    }

    out.push_str("## Parity Exceptions\n\n");
    if parity_exceptions.is_empty() {
        out.push_str("None.\n");
    } else {
        for exception in parity_exceptions {
            out.push_str(&format!(
                "- `{}` (`{}` / `{}`): {}\n",
                exception.case_id,
                exception.family,
                exception.primitive,
                exception
                    .error
                    .clone()
                    .unwrap_or_else(|| "mismatch without explicit error detail".to_owned())
            ));
        }
    }
    out
}

fn main() -> Result<(), String> {
    let root = repo_root();
    let args = parse_args(&root)?;

    let bundle = read_transform_fixture_bundle(&args.fixtures).map_err(|err| {
        format!(
            "failed reading fixture bundle {}: {err}",
            args.fixtures.display()
        )
    })?;

    let case_program: BTreeMap<String, String> = bundle
        .cases
        .iter()
        .map(|case| (case.case_id.clone(), format!("{:?}", case.program)))
        .collect();

    let config = HarnessConfig::default();
    let transform_report = run_transform_fixture_bundle(&config, &bundle);

    let mut v1 = ParityReportV1::from_transform_report(
        &transform_report,
        &args.mode,
        &args.fj_version,
        &args.oracle_version,
    );

    // Load and verify RNG fixtures
    let rng_results = if args.rng_fixtures.exists() {
        let rng_raw = fs::read_to_string(&args.rng_fixtures).map_err(|err| {
            format!(
                "failed reading RNG fixture bundle {}: {err}",
                args.rng_fixtures.display()
            )
        })?;
        let rng_bundle: RngFixtureBundle = serde_json::from_str(&rng_raw)
            .map_err(|err| format!("failed parsing RNG fixture bundle: {err}"))?;

        let mut rng_cases = Vec::new();
        for case in &rng_bundle.cases {
            let (matched, error) = verify_rng_case(case);
            rng_cases.push((case.case_id.clone(), matched, error));
        }
        rng_cases
    } else {
        Vec::new()
    };

    // Add RNG family to the report
    let rng_matched = rng_results.iter().filter(|(_, m, _)| *m).count();
    let rng_total = rng_results.len();
    let rng_family = FamilyReport {
        total: rng_total,
        matched: rng_matched,
        mismatched: rng_total - rng_matched,
        cases: rng_results
            .iter()
            .map(|(case_id, matched, error)| FamilyCaseEntry {
                case_id: case_id.clone(),
                matched: *matched,
                expected: None,
                actual: None,
                error: error.clone(),
            })
            .collect(),
    };
    if rng_total > 0 {
        v1.families.insert("random".to_owned(), rng_family);
        // Update summary totals to include RNG
        v1.summary.total += rng_total;
        let total_matched = v1.families.values().map(|f| f.matched).sum::<usize>();
        v1.summary.pass_rate = if v1.summary.total == 0 {
            1.0
        } else {
            total_matched as f64 / v1.summary.total as f64
        };
        v1.summary.gate = if v1.summary.pass_rate >= 1.0 {
            "pass".to_owned()
        } else {
            "fail".to_owned()
        };
    }

    for family in ["jit", "grad", "vmap", "lax", "random", "control_flow"] {
        v1.families
            .entry(family_name(family))
            .or_insert_with(|| FamilyReport {
                total: 0,
                matched: 0,
                mismatched: 0,
                cases: Vec::new(),
            });
    }

    let mut per_primitive: BTreeMap<String, PrimitiveBreakdown> = BTreeMap::new();
    let mut parity_exceptions = Vec::new();
    let mut coverage_exceptions = Vec::new();

    for case in &transform_report.reports {
        let primitive = case_program
            .get(&case.case_id)
            .cloned()
            .unwrap_or_else(|| "unknown".to_owned());
        let stats = per_primitive
            .entry(primitive.clone())
            .or_insert_with(|| PrimitiveBreakdown {
                total: 0,
                matched: 0,
                mismatched: 0,
                cases: Vec::new(),
            });
        stats.total += 1;
        if case.matched {
            stats.matched += 1;
        } else {
            stats.mismatched += 1;
            parity_exceptions.push(ParityException {
                case_id: case.case_id.clone(),
                family: format!("{:?}", case.family).to_lowercase(),
                primitive,
                drift: format!("{:?}", case.drift_classification),
                expected: case.expected_json.clone(),
                actual: case.actual_json.clone(),
                error: case.error.clone(),
            });
        }
        stats.cases.push(case.case_id.clone());
    }

    // Add RNG parity exceptions if any
    for (case_id, matched, error) in &rng_results {
        if !matched {
            parity_exceptions.push(ParityException {
                case_id: case_id.clone(),
                family: "random".to_owned(),
                primitive: "rng".to_owned(),
                drift: "Regression".to_owned(),
                expected: String::new(),
                actual: None,
                error: error.clone(),
            });
        }
    }

    for family in ["jit", "grad", "vmap", "lax", "random", "control_flow"] {
        if let Some(stats) = v1.families.get(family)
            && stats.total == 0
        {
            coverage_exceptions.push(CoverageException {
                family: family.to_owned(),
                reason: "no fixture cases captured for this family".to_owned(),
                justification: "tracked as known conformance gap pending fixture expansion"
                    .to_owned(),
            });
        }
    }

    let comprehensive = ComprehensiveParityReport {
        version: "frankenjax.v1-comprehensive-parity.v1".to_owned(),
        timestamp: v1.timestamp.clone(),
        fj_version: v1.fj_version.clone(),
        jax_version: v1.oracle_version.clone(),
        mode: v1.mode.clone(),
        summary: v1.summary.clone(),
        families: v1.families.clone(),
        per_primitive: per_primitive.clone(),
        parity_exceptions: parity_exceptions.clone(),
        coverage_exceptions: coverage_exceptions.clone(),
        gate_status: v1.summary.gate.clone(),
    };

    let markdown = to_markdown(
        &v1,
        &per_primitive,
        &parity_exceptions,
        &coverage_exceptions,
    );

    let per_family = ["jit", "grad", "vmap", "lax", "random", "control_flow"]
        .into_iter()
        .map(|name| {
            let family = v1.families.get(name).cloned().unwrap_or(FamilyReport {
                total: 0,
                matched: 0,
                mismatched: 0,
                cases: Vec::new(),
            });
            (
                name.to_owned(),
                FamilyRollup {
                    total: family.total,
                    matched: family.matched,
                    mismatched: family.mismatched,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    let e2e_matched: usize = v1.families.values().map(|f| f.matched).sum();
    let e2e_mismatched: usize = v1.families.values().map(|f| f.mismatched).sum();
    let e2e_log = E2EFinalParityLog {
        total_cases: v1.summary.total,
        matched: e2e_matched,
        mismatched: e2e_mismatched,
        pass_rate: v1.summary.pass_rate,
        per_family,
        gate_status: v1.summary.gate.clone(),
        pass: v1.gate_passes(),
    };

    let comprehensive_json = serde_json::to_string_pretty(&comprehensive)
        .map_err(|err| format!("failed to serialize comprehensive report: {err}"))?;
    let ci_json = v1
        .to_json()
        .map_err(|err| format!("failed to serialize V1 report: {err}"))?;
    let e2e_json = serde_json::to_string_pretty(&e2e_log)
        .map_err(|err| format!("failed to serialize e2e parity log: {err}"))?;

    write_text(&args.output_json, &comprehensive_json)?;
    write_text(&args.output_markdown, &markdown)?;
    write_text(&args.ci_json, &ci_json)?;
    write_text(&args.e2e_json, &e2e_json)?;

    println!("generated comprehensive parity artifacts:");
    println!("  {}", args.output_json.display());
    println!("  {}", args.output_markdown.display());
    println!("  {}", args.ci_json.display());
    println!("  {}", args.e2e_json.display());
    let total_matched: usize = v1.families.values().map(|f| f.matched).sum();
    let total_mismatched: usize = v1.families.values().map(|f| f.mismatched).sum();
    println!(
        "gate={} total={} matched={} mismatched={}",
        v1.summary.gate, v1.summary.total, total_matched, total_mismatched,
    );
    Ok(())
}
