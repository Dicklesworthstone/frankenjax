#![no_main]

use fj_conformance::{
    FixtureNote, HarnessConfig, ParityCase, ParityReport, evaluate_parity, read_fixture_note,
    run_smoke,
};
use libfuzzer_sys::fuzz_target;
use std::fs;

const MAX_INPUT_BYTES: usize = 16 * 1024;
const MAX_PARITY_CASES: usize = 64;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    exercise_fixture_note_loader(data);
    exercise_parity_case_loader(data);
    exercise_parity_report_loader(data);
});

fn exercise_fixture_note_loader(data: &[u8]) {
    let Ok(tmp_dir) = tempfile::tempdir() else {
        return;
    };

    let note_path = tmp_dir.path().join("fixture_note.json");
    if fs::write(&note_path, data).is_err() {
        return;
    }

    let Ok(note) = read_fixture_note(&note_path) else {
        return;
    };

    let fixture_root = tmp_dir.path().join("fixtures");
    if fs::create_dir_all(&fixture_root).is_err() {
        return;
    }
    let _ = fs::write(fixture_root.join("note_copy.json"), data);

    let strict_mode = note
        .suite
        .len()
        .saturating_add(note.notes.len())
        .is_multiple_of(2);
    let config = HarnessConfig {
        oracle_root: tmp_dir.path().join("oracle"),
        fixture_root,
        strict_mode,
    };
    let report = run_smoke(&config);

    assert_eq!(report.suite, "smoke");
    assert_eq!(report.strict_mode, strict_mode);

    let Ok(encoded) = serde_json::to_vec(&note) else {
        return;
    };
    let Ok(decoded) = serde_json::from_slice::<FixtureNote>(&encoded) else {
        return;
    };
    assert_eq!(note, decoded);
}

fn exercise_parity_case_loader(data: &[u8]) {
    let Ok(cases) = serde_json::from_slice::<Vec<ParityCase>>(data) else {
        return;
    };
    if cases.len() > MAX_PARITY_CASES {
        return;
    }

    let strict_mode = data.first().is_some_and(|byte| byte.is_multiple_of(2));
    let report = evaluate_parity(&cases, strict_mode);

    assert_eq!(report.total_cases, cases.len());
    assert_eq!(
        report.total_cases,
        report.matched_cases.saturating_add(report.mismatched_cases)
    );
    assert_eq!(report.strict_mode, strict_mode);

    let Ok(encoded) = serde_json::to_vec(&report) else {
        return;
    };
    let Ok(decoded) = serde_json::from_slice::<ParityReport>(&encoded) else {
        return;
    };
    assert_eq!(report, decoded);
}

fn exercise_parity_report_loader(data: &[u8]) {
    let Ok(report) = serde_json::from_slice::<ParityReport>(data) else {
        return;
    };

    let _ = report
        .matched_cases
        .checked_add(report.mismatched_cases)
        .and_then(|count| count.checked_add(report.total_cases));

    let Ok(encoded) = serde_json::to_vec(&report) else {
        return;
    };
    let Ok(decoded) = serde_json::from_slice::<ParityReport>(&encoded) else {
        return;
    };
    assert_eq!(report, decoded);
}
