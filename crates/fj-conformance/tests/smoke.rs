use std::path::Path;

use fj_conformance::{HarnessConfig, default_fixture_manifest, read_fixture_note, run_smoke};

#[test]
fn smoke_report_is_stable() {
    let cfg = HarnessConfig::default_paths();
    let report = run_smoke(&cfg);
    assert_eq!(report.suite, "smoke");
    assert!(report.fixture_count >= 1);
    if !cfg.oracle_root.exists() {
        eprintln!(
            "SKIP oracle assertions: oracle root not found at {}",
            cfg.oracle_root.display()
        );
    } else {
        assert!(report.oracle_present);
    }
    assert_eq!(
        report.manifest_family_count,
        default_fixture_manifest().len()
    );

    let fixture_path = cfg.fixture_root.join("smoke_case.json");
    assert!(Path::new(&fixture_path).exists());

    let note = read_fixture_note(&fixture_path).expect("fixture note should parse");
    assert_eq!(note.suite, "smoke");
}
