#![no_main]

mod common;

use fj_conformance::{
    BatchRunnerConfig, HarnessConfig, read_transform_fixture_bundle, run_transform_fixture_bundle,
    run_transform_fixture_bundle_batched,
};
use libfuzzer_sys::fuzz_target;
use std::fs;

fuzz_target!(|data: &[u8]| {
    let Ok(tmp_dir) = tempfile::tempdir() else {
        return;
    };

    let fixture_path = tmp_dir.path().join("bundle.json");
    if fs::write(&fixture_path, data).is_err() {
        return;
    }

    if let Ok(bundle) = read_transform_fixture_bundle(&fixture_path) {
        let config = HarnessConfig::default();
        let _ = run_transform_fixture_bundle(&config, &bundle);
        let _ =
            run_transform_fixture_bundle_batched(&config, &bundle, &BatchRunnerConfig::default());
    }
});
