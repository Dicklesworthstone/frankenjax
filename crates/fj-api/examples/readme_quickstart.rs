#![forbid(unsafe_code)]

use fj_api::{DType, Shape, ShapedArray, Value, grad, jit, make_jaxpr, value_and_grad, vmap};
use serde_json::{Value as JsonValue, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_E2E_LOG: &str = "artifacts/e2e/e2e_api_readme_quickstart.e2e.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    let started = Instant::now();
    let current_dir = std::env::current_dir()?;

    let closed = make_jaxpr(
        |inputs| {
            let x = &inputs[0];
            let square = x * x;
            let double = x + x;
            let triple = &double + x;
            vec![square + triple]
        },
        vec![ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        }],
    )?;

    let scalar_x = 5.0;
    let jit_output = scalar_from_values(
        "jit",
        jit(closed.jaxpr.clone()).call(vec![Value::scalar_f64(scalar_x)])?,
    )?;
    let grad_output = scalar_from_values(
        "grad",
        grad(closed.jaxpr.clone()).call(vec![Value::scalar_f64(scalar_x)])?,
    )?;
    let (value_outputs, grad_outputs) =
        value_and_grad(closed.jaxpr.clone()).call(vec![Value::scalar_f64(scalar_x)])?;
    let value_and_grad_output = scalar_from_values("value_and_grad.value", value_outputs)?;
    let value_and_grad_gradient = scalar_from_values("value_and_grad.grad", grad_outputs)?;

    let batch = [1.0, 2.0, 3.0];
    let vmap_values = tensor_f64_values(
        "vmap",
        vmap(closed.jaxpr.clone()).call(vec![Value::vector_f64(&batch)?])?,
    )?;

    let expected_jit = scalar_x * scalar_x + 3.0 * scalar_x;
    let expected_grad = 2.0 * scalar_x + 3.0;
    let expected_vmap: Vec<f64> = batch.iter().map(|x| x * x + 3.0 * x).collect();

    assert_close("jit", jit_output, expected_jit, 1e-10)?;
    assert_close("grad", grad_output, expected_grad, 1e-3)?;
    assert_close(
        "value_and_grad.value",
        value_and_grad_output,
        expected_jit,
        1e-10,
    )?;
    assert_close(
        "value_and_grad.grad",
        value_and_grad_gradient,
        expected_grad,
        1e-3,
    )?;
    if vmap_values.len() != expected_vmap.len() {
        return Err(format!(
            "vmap returned {} values, expected {}",
            vmap_values.len(),
            expected_vmap.len()
        )
        .into());
    }
    for (idx, (actual, expected)) in vmap_values.iter().zip(expected_vmap.iter()).enumerate() {
        assert_close(&format!("vmap[{idx}]"), *actual, *expected, 1e-10)?;
    }

    let actual = json!({
        "jaxpr_equation_count": closed.jaxpr.equations.len(),
        "jit_value": jit_output,
        "grad_value": grad_output,
        "vmap_values": vmap_values,
        "value_and_grad_value": value_and_grad_output,
        "value_and_grad_grad": value_and_grad_gradient,
    });
    let expected = json!({
        "jaxpr_equation_count": 4,
        "jit_value": expected_jit,
        "grad_value": expected_grad,
        "vmap_values": expected_vmap,
        "value_and_grad_value": expected_jit,
        "value_and_grad_grad": expected_grad,
    });

    println!(
        "readme quickstart: status=pass equations={} jit={jit_output} grad={grad_output} vmap={}",
        closed.jaxpr.equations.len(),
        actual["vmap_values"]
    );

    if let Some(path) = args.json_path {
        let log = e2e_log(
            &current_dir,
            &args.raw_args,
            started.elapsed().as_millis(),
            expected,
            actual,
        )?;
        write_json(&path, &log)?;
    }

    Ok(())
}

#[derive(Debug)]
struct Args {
    raw_args: Vec<String>,
    json_path: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let raw_args: Vec<String> = std::env::args().collect();
        let mut json_path = None;
        let mut iter = raw_args.iter().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--json" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--json requires a path".to_owned())?;
                    if value.starts_with('-') {
                        return Err(format!("--json requires a path, got flag `{value}`"));
                    }
                    json_path = Some(PathBuf::from(value));
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument `{arg}`")),
            }
        }
        Ok(Self {
            raw_args,
            json_path,
        })
    }
}

fn scalar_from_values(name: &str, values: Vec<Value>) -> Result<f64, String> {
    if values.len() != 1 {
        return Err(format!(
            "{name} returned {} outputs, expected 1",
            values.len()
        ));
    }
    values[0]
        .as_f64_scalar()
        .ok_or_else(|| format!("{name} output is not an f64 scalar: {:?}", values[0]))
}

fn tensor_f64_values(name: &str, values: Vec<Value>) -> Result<Vec<f64>, String> {
    if values.len() != 1 {
        return Err(format!(
            "{name} returned {} outputs, expected 1",
            values.len()
        ));
    }
    values[0]
        .as_tensor()
        .ok_or_else(|| format!("{name} output is not a tensor: {:?}", values[0]))?
        .to_f64_vec()
        .ok_or_else(|| format!("{name} tensor is not f64-compatible"))
}

fn assert_close(name: &str, actual: f64, expected: f64, tolerance: f64) -> Result<(), String> {
    if (actual - expected).abs() <= tolerance {
        Ok(())
    } else {
        Err(format!(
            "{name} mismatch: actual {actual}, expected {expected}, tolerance {tolerance}"
        ))
    }
}

fn e2e_log(
    root: &Path,
    command: &[String],
    total_ms: u128,
    expected: JsonValue,
    actual: JsonValue,
) -> Result<JsonValue, Box<dyn std::error::Error>> {
    let mut env_vars = BTreeMap::new();
    for key in ["CARGO_TARGET_DIR", "RUSTUP_TOOLCHAIN"] {
        if let Ok(value) = std::env::var(key) {
            env_vars.insert(key.to_owned(), value);
        }
    }

    Ok(json!({
        "schema_version": "frankenjax.e2e-forensic-log.v1",
        "bead_id": "frankenjax-cstq.9",
        "scenario_id": "e2e_api_readme_quickstart",
        "test_id": "fj_api_readme_quickstart",
        "packet_id": null,
        "command": command,
        "working_dir": root.display().to_string(),
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "rust_version": command_version("rustc"),
            "cargo_version": command_version("cargo"),
            "cargo_target_dir": std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "<default>".to_owned()),
            "env_vars": env_vars,
            "timestamp_unix_ms": now_unix_ms(),
        },
        "feature_flags": [],
        "fixture_ids": ["README.md#quick-example"],
        "oracle_ids": ["README.md", "crates/fj-api/examples/readme_quickstart.rs"],
        "transform_stack": ["make_jaxpr", "jit", "grad", "vmap", "value_and_grad"],
        "mode": "strict",
        "inputs": {
            "scalar_x": 5.0,
            "batch": [1.0, 2.0, 3.0],
            "abstract_input": {"dtype": "f64", "shape": []},
        },
        "expected": expected,
        "actual": actual,
        "tolerance": {
            "policy_id": "readme_quickstart_exact_or_ad_tolerance",
            "atol": 0.001,
            "rtol": null,
            "ulp": null,
            "notes": "jit/vmap/value checks are exact within f64 arithmetic; grad/value_and_grad use the public AD tolerance already documented by fj-api tests",
        },
        "error": {
            "expected": null,
            "actual": null,
            "taxonomy_class": "none",
        },
        "timings": {
            "setup_ms": 0,
            "trace_ms": 0,
            "dispatch_ms": 0,
            "eval_ms": 0,
            "verify_ms": 0,
            "total_ms": u64::try_from(total_ms).unwrap_or(u64::MAX),
        },
        "allocations": {
            "allocation_count": null,
            "allocated_bytes": null,
            "peak_rss_bytes": null,
            "measurement_backend": "not_measured",
        },
        "artifacts": [
            artifact_ref(root, "README.md", "readme_quick_example")?,
            artifact_ref(root, "crates/fj-api/examples/readme_quickstart.rs", "executable_readme_quickstart")?,
        ],
        "replay_command": "./scripts/run_api_readme_examples.sh --enforce",
        "status": "pass",
        "failure_summary": null,
        "redactions": [],
        "metadata": {
            "bead": {
                "id": "frankenjax-cstq.9",
                "title": "Prove top-level user facade and README examples executable"
            },
            "public_facade_crate": "fj-api",
            "documented_import_path": "fj_api",
        },
    }))
}

fn artifact_ref(
    root: &Path,
    relative_path: &str,
    kind: &str,
) -> Result<JsonValue, Box<dyn std::error::Error>> {
    let path = root.join(relative_path);
    Ok(json!({
        "kind": kind,
        "path": relative_path,
        "sha256": sha256_file_hex(&path)?,
        "required": true,
    }))
}

fn sha256_file_hex(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn write_json(path: &Path, value: &JsonValue) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}

fn command_version(command: &str) -> String {
    match Command::new(command).arg("--version").output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }
        _ => format!("{command} <unknown>"),
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

fn print_usage() {
    println!("Usage: cargo run -p fj-api --example readme_quickstart -- [--json <path>]");
}
