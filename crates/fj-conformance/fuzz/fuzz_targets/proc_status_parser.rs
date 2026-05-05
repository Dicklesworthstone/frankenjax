#![no_main]

use fj_conformance::memory_performance::parse_proc_status_kib;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(raw) = std::str::from_utf8(data) else {
        return;
    };
    if raw.len() > 8192 {
        return;
    }

    for key in ["VmRSS:", "VmHWM:", "VmSize:", "VmPeak:", "RssAnon:", "RssShmem:"] {
        let result = parse_proc_status_kib(raw, key);
        if let Some(bytes) = result {
            assert!(
                bytes <= u64::MAX / 2,
                "parse_proc_status_kib returned suspiciously large value"
            );
        }
    }

    if !raw.is_empty() {
        let arbitrary_key = raw.lines().next().unwrap_or("X:");
        let _ = parse_proc_status_kib(raw, arbitrary_key);
    }
});
