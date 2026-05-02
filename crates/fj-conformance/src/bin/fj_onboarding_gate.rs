#![forbid(unsafe_code)]

use fj_conformance::onboarding_command::{
    OnboardingCommandOutputPaths, onboarding_inventory_summary_json,
    write_onboarding_command_outputs,
};
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct Args {
    root: PathBuf,
    inventory: PathBuf,
    e2e: PathBuf,
    enforce: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut root = std::env::current_dir().map_err(|err| format!("current dir: {err}"))?;
        let mut inventory = None;
        let mut e2e = None;
        let mut enforce = false;

        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--root" => root = PathBuf::from(next_value(&mut iter, "--root")?),
                "--inventory" => {
                    inventory = Some(PathBuf::from(next_value(&mut iter, "--inventory")?));
                }
                "--e2e" => e2e = Some(PathBuf::from(next_value(&mut iter, "--e2e")?)),
                "--enforce" => enforce = true,
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument `{arg}`")),
            }
        }

        let defaults = OnboardingCommandOutputPaths::for_root(&root);
        Ok(Self {
            root,
            inventory: inventory.unwrap_or(defaults.inventory),
            e2e: e2e.unwrap_or(defaults.e2e),
            enforce,
        })
    }
}

fn main() {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("error: {err}");
            print_usage();
            std::process::exit(2);
        }
    };

    let inventory = match write_onboarding_command_outputs(&args.root, &args.inventory, &args.e2e) {
        Ok(inventory) => inventory,
        Err(err) => {
            eprintln!("error: failed to write onboarding command gate outputs: {err}");
            std::process::exit(1);
        }
    };

    let summary = onboarding_inventory_summary_json(&inventory);
    println!(
        "onboarding command gate: status={} commands={} mandatory_smoke={} ci_gates={} optional_oracle={} long_running={} environment_specific={} schematic={} issues={}",
        inventory.status,
        inventory.summary.total_commands,
        inventory.summary.mandatory_smoke_count,
        inventory.summary.ci_gate_count,
        inventory.summary.optional_oracle_count,
        inventory.summary.long_running_count,
        inventory.summary.environment_specific_count,
        inventory.summary.schematic_count,
        inventory.issues.len(),
    );
    match serde_json::to_string_pretty(&summary) {
        Ok(raw) => println!("{raw}"),
        Err(err) => eprintln!("warning: failed to render summary json: {err}"),
    }

    if args.enforce && inventory.status != "pass" {
        std::process::exit(1);
    }
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    let value = iter
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?;
    if value.starts_with("--") {
        return Err(format!("{flag} requires a value, got flag `{value}`"));
    }
    Ok(value)
}

fn print_usage() {
    eprintln!(
        "Usage: fj_onboarding_gate [--root <dir>] [--inventory <json>] [--e2e <json>] [--enforce]"
    );
}
