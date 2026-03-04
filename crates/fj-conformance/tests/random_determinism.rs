use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use fj_conformance::HarnessConfig;
use fj_lax::threefry::{random_fold_in, random_key, random_normal, random_split, random_uniform};
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};

#[derive(Debug, Deserialize)]
struct RngFixtureBundle {
    schema_version: String,
    cases: Vec<RngFixtureCase>,
}

#[derive(Debug, Deserialize)]
struct RngFixtureCase {
    case_id: String,
    family: String,
    operation: String,
    seed: u64,
    fold_in_data: Option<u32>,
    minval: Option<f64>,
    maxval: Option<f64>,
    shape: Vec<usize>,
    comparator: String,
    atol: f64,
    rtol: f64,
    #[serde(default)]
    expected_key_bits: Vec<u32>,
    #[serde(default)]
    expected_split_keys: Vec<Vec<u32>>,
    #[serde(default)]
    expected_values: Vec<f64>,
}

fn approx_equal(expected: f64, actual: f64, atol: f64, rtol: f64) -> bool {
    if expected.is_nan() && actual.is_nan() {
        return true;
    }
    let diff = (expected - actual).abs();
    diff <= atol + rtol * expected.abs().max(actual.abs())
}

fn element_count(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

fn verify_case(case: &RngFixtureCase) -> Result<(), String> {
    if case.family != "random" {
        return Err(format!(
            "{} has unexpected family {}",
            case.case_id, case.family
        ));
    }

    match case.operation.as_str() {
        "key" => {
            let actual = random_key(case.seed).0.to_vec();
            if actual != case.expected_key_bits {
                return Err(format!(
                    "{} key mismatch: expected {:?}, actual {:?}",
                    case.case_id, case.expected_key_bits, actual
                ));
            }
        }
        "split" => {
            let (left, right) = random_split(random_key(case.seed));
            let actual = vec![left.0.to_vec(), right.0.to_vec()];
            if actual != case.expected_split_keys {
                return Err(format!(
                    "{} split mismatch: expected {:?}, actual {:?}",
                    case.case_id, case.expected_split_keys, actual
                ));
            }
        }
        "fold_in" => {
            let fold_in_data = case
                .fold_in_data
                .ok_or_else(|| format!("{} missing fold_in_data", case.case_id))?;
            let actual = random_fold_in(random_key(case.seed), fold_in_data)
                .0
                .to_vec();
            if actual != case.expected_key_bits {
                return Err(format!(
                    "{} fold_in mismatch: expected {:?}, actual {:?}",
                    case.case_id, case.expected_key_bits, actual
                ));
            }
        }
        "uniform" => {
            let minval = case
                .minval
                .ok_or_else(|| format!("{} missing minval", case.case_id))?;
            let maxval = case
                .maxval
                .ok_or_else(|| format!("{} missing maxval", case.case_id))?;
            let count = element_count(&case.shape);
            let actual = random_uniform(random_key(case.seed), count, minval, maxval);

            if actual.len() != case.expected_values.len() {
                return Err(format!(
                    "{} uniform length mismatch: expected {}, actual {}",
                    case.case_id,
                    case.expected_values.len(),
                    actual.len()
                ));
            }

            for (idx, (expected, got)) in case.expected_values.iter().zip(actual.iter()).enumerate()
            {
                if !approx_equal(*expected, *got, case.atol, case.rtol) {
                    return Err(format!(
                        "{} uniform mismatch at index {}: expected {:.16e}, actual {:.16e}",
                        case.case_id, idx, expected, got
                    ));
                }
            }
        }
        "normal" => {
            let count = element_count(&case.shape);
            let actual = random_normal(random_key(case.seed), count);

            if actual.len() != case.expected_values.len() {
                return Err(format!(
                    "{} normal length mismatch: expected {}, actual {}",
                    case.case_id,
                    case.expected_values.len(),
                    actual.len()
                ));
            }

            for (idx, (expected, got)) in case.expected_values.iter().zip(actual.iter()).enumerate()
            {
                if !approx_equal(*expected, *got, case.atol, case.rtol) {
                    return Err(format!(
                        "{} normal mismatch at index {}: expected {:.16e}, actual {:.16e}",
                        case.case_id, idx, expected, got
                    ));
                }
            }
        }
        other => return Err(format!("{} has unknown operation {}", case.case_id, other)),
    }

    // Keep comparator field validated so fixture schema stays intentional.
    if case.operation == "key" || case.operation == "split" || case.operation == "fold_in" {
        if case.comparator != "exact" {
            return Err(format!(
                "{} expected comparator=exact, got {}",
                case.case_id, case.comparator
            ));
        }
    } else if case.comparator != "approx_atol_rtol" {
        return Err(format!(
            "{} expected comparator=approx_atol_rtol, got {}",
            case.case_id, case.comparator
        ));
    }

    Ok(())
}

fn load_rng_fixture_bundle() -> RngFixtureBundle {
    let cfg = HarnessConfig::default_paths();
    let fixture_path = cfg.fixture_root.join("rng").join("rng_determinism.v1.json");
    assert!(
        Path::new(&fixture_path).exists(),
        "expected RNG fixture bundle at {}",
        fixture_path.display()
    );

    let raw =
        std::fs::read_to_string(&fixture_path).expect("rng fixture bundle should be readable");
    let bundle: RngFixtureBundle =
        serde_json::from_str(&raw).expect("rng fixture bundle should parse");

    assert_eq!(bundle.schema_version, "frankenjax.rng-fixtures.v1");
    assert!(
        bundle.cases.len() >= 20,
        "expected at least 20 RNG fixture cases, got {}",
        bundle.cases.len()
    );
    bundle
}

fn operation_cases<'a>(bundle: &'a RngFixtureBundle, operation: &str) -> Vec<&'a RngFixtureCase> {
    let cases: Vec<&RngFixtureCase> = bundle
        .cases
        .iter()
        .filter(|case| case.operation == operation)
        .collect();
    assert!(
        !cases.is_empty(),
        "expected at least one RNG fixture for operation {operation}"
    );
    cases
}

fn assert_cases_match(cases: &[&RngFixtureCase], label: &str) {
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|case| verify_case(case).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{label} mismatches ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn evaluate_case(case: &RngFixtureCase) -> (JsonValue, bool, bool) {
    match case.operation.as_str() {
        "key" => {
            let actual = random_key(case.seed).0.to_vec();
            let exact_match = actual == case.expected_key_bits;
            (json!(actual), exact_match, exact_match)
        }
        "split" => {
            let (left, right) = random_split(random_key(case.seed));
            let actual = vec![left.0.to_vec(), right.0.to_vec()];
            let exact_match = actual == case.expected_split_keys;
            (json!(actual), exact_match, exact_match)
        }
        "fold_in" => {
            let fold_in_data = case
                .fold_in_data
                .expect("fold_in fixtures must include fold_in_data");
            let actual = random_fold_in(random_key(case.seed), fold_in_data)
                .0
                .to_vec();
            let exact_match = actual == case.expected_key_bits;
            (json!(actual), exact_match, exact_match)
        }
        "uniform" => {
            let minval = case.minval.expect("uniform fixtures must include minval");
            let maxval = case.maxval.expect("uniform fixtures must include maxval");
            let count = element_count(&case.shape);
            let actual = random_uniform(random_key(case.seed), count, minval, maxval);
            let bitwise_match = actual.len() == case.expected_values.len()
                && actual
                    .iter()
                    .zip(case.expected_values.iter())
                    .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits());
            let pass = verify_case(case).is_ok();
            (json!(actual), bitwise_match, pass)
        }
        "normal" => {
            let count = element_count(&case.shape);
            let actual = random_normal(random_key(case.seed), count);
            let bitwise_match = actual.len() == case.expected_values.len()
                && actual
                    .iter()
                    .zip(case.expected_values.iter())
                    .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits());
            let pass = verify_case(case).is_ok();
            (json!(actual), bitwise_match, pass)
        }
        other => panic!("unsupported operation {other}"),
    }
}

fn expected_output(case: &RngFixtureCase) -> JsonValue {
    match case.operation.as_str() {
        "key" | "fold_in" => json!(case.expected_key_bits),
        "split" => json!(case.expected_split_keys),
        "uniform" | "normal" => json!(case.expected_values),
        other => panic!("unsupported operation {other}"),
    }
}

#[test]
fn rng_fixture_bundle_exists_and_parses() {
    let _ = load_rng_fixture_bundle();
}

#[test]
fn rng_fixture_bundle_matches_threefry_engine() {
    let bundle = load_rng_fixture_bundle();
    let cases: Vec<&RngFixtureCase> = bundle.cases.iter().collect();
    assert_cases_match(&cases, "rng fixture bundle");
}

#[test]
fn test_rng_fixture_threefry_vectors() {
    let bundle = load_rng_fixture_bundle();
    let cases = operation_cases(&bundle, "key");
    assert_cases_match(&cases, "threefry key vectors");
}

#[test]
fn test_rng_fixture_uniform_determinism() {
    let bundle = load_rng_fixture_bundle();
    let cases = operation_cases(&bundle, "uniform");
    assert_cases_match(&cases, "uniform determinism");
}

#[test]
fn test_rng_fixture_normal_determinism() {
    let bundle = load_rng_fixture_bundle();
    let cases = operation_cases(&bundle, "normal");
    assert_cases_match(&cases, "normal determinism");
}

#[test]
fn test_rng_fixture_key_split_matches() {
    let bundle = load_rng_fixture_bundle();
    let cases = operation_cases(&bundle, "split");
    assert_cases_match(&cases, "key split fixtures");
}

#[test]
fn test_rng_fixture_fold_in_matches() {
    let bundle = load_rng_fixture_bundle();
    let cases = operation_cases(&bundle, "fold_in");
    assert_cases_match(&cases, "fold_in fixtures");
}

#[test]
fn e2e_rng_determinism_full() {
    let bundle = load_rng_fixture_bundle();
    let mut records = Vec::with_capacity(bundle.cases.len());
    let mut forensic_rows = Vec::with_capacity(bundle.cases.len());
    let mut passed = 0usize;

    for case in &bundle.cases {
        let expected = expected_output(case);
        let (actual, bitwise_match, pass) = evaluate_case(case);
        if pass {
            passed += 1;
        }
        records.push(json!({
            "function": case.operation,
            "key": case.seed,
            "expected_output": expected,
            "actual_output": actual,
            "bitwise_match": bitwise_match,
            "pass": pass,
        }));
        forensic_rows.push(json!({
            "test_name": "e2e_rng_determinism_full",
            "function": case.operation,
            "key": case.seed,
            "fixture_source": "rng_determinism.v1.json",
            "bitwise_match": bitwise_match,
            "pass": pass,
        }));
    }

    let generated_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after UNIX_EPOCH")
        .as_millis();
    let generated_at_unix_ms =
        u64::try_from(generated_at_unix_ms).expect("unix millis should fit into u64");
    let e2e_payload = json!({
        "test_name": "e2e_rng_determinism_full",
        "generated_at_unix_ms": generated_at_unix_ms,
        "total_cases": bundle.cases.len(),
        "passed_cases": passed,
        "entries": records,
    });
    let log_payload = json!({
        "test_name": "e2e_rng_determinism_full",
        "generated_at_unix_ms": generated_at_unix_ms,
        "total_cases": bundle.cases.len(),
        "passed_cases": passed,
        "entries": forensic_rows,
    });

    let e2e_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/e2e/e2e_rng_determinism.e2e.json");
    std::fs::create_dir_all(
        e2e_path
            .parent()
            .expect("e2e artifact path should have a parent"),
    )
    .expect("should create artifacts/e2e");
    std::fs::write(
        &e2e_path,
        serde_json::to_string_pretty(&e2e_payload).expect("serialize e2e rng payload"),
    )
    .expect("should write e2e RNG artifact");

    let log_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/testing/logs/fj-conformance/e2e_rng_determinism_full.json");
    std::fs::create_dir_all(
        log_path
            .parent()
            .expect("forensic log path should have a parent"),
    )
    .expect("should create artifacts/testing/logs/fj-conformance");
    std::fs::write(
        &log_path,
        serde_json::to_string_pretty(&log_payload).expect("serialize e2e rng log payload"),
    )
    .expect("should write RNG forensic log");

    println!("__FJ_RNG_E2E_JSON_BEGIN__");
    println!(
        "{}",
        serde_json::to_string(&e2e_payload).expect("serialize e2e rng payload compact")
    );
    println!("__FJ_RNG_E2E_JSON_END__");
    println!("__FJ_RNG_LOG_JSON_BEGIN__");
    println!(
        "{}",
        serde_json::to_string(&log_payload).expect("serialize e2e rng log payload compact")
    );
    println!("__FJ_RNG_LOG_JSON_END__");

    assert!(
        passed == bundle.cases.len(),
        "rng determinism e2e mismatches: passed {} / {}",
        passed,
        bundle.cases.len()
    );
}
