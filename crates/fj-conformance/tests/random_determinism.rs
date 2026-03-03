use std::path::Path;

use fj_conformance::HarnessConfig;
use fj_lax::threefry::{random_fold_in, random_key, random_normal, random_split, random_uniform};
use serde::Deserialize;

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

#[test]
fn rng_fixture_bundle_exists_and_parses() {
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
}

#[test]
fn rng_fixture_bundle_matches_threefry_engine() {
    let cfg = HarnessConfig::default_paths();
    let fixture_path = cfg.fixture_root.join("rng").join("rng_determinism.v1.json");
    let raw =
        std::fs::read_to_string(&fixture_path).expect("rng fixture bundle should be readable");
    let bundle: RngFixtureBundle =
        serde_json::from_str(&raw).expect("rng fixture bundle should parse");

    let mut failures = Vec::new();
    for case in &bundle.cases {
        if let Err(err) = verify_case(case) {
            failures.push(err);
        }
    }

    assert!(
        failures.is_empty(),
        "rng fixture mismatches ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
