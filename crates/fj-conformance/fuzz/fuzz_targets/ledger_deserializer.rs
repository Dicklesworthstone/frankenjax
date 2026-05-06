#![no_main]

//! Fuzz target for EvidenceLedger JSON deserialization.
//!
//! The EvidenceLedger is returned in dispatch responses and may be serialized
//! for logging, caching, or transmission. This fuzzer ensures deserialization
//! handles malformed JSON gracefully (no panics) and that round-trips are stable.

use fj_ledger::{EvidenceLedger, LedgerEntry};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }

    // Fuzz LedgerEntry deserialization
    if let Ok(entry) = serde_json::from_slice::<LedgerEntry>(data) {
        // Round-trip check
        let re_encoded = serde_json::to_vec(&entry).expect("re-encode must succeed");
        let re_decoded: LedgerEntry =
            serde_json::from_slice(&re_encoded).expect("round-trip must succeed");
        assert_eq!(entry, re_decoded, "round-trip must be stable");
    }

    // Fuzz EvidenceLedger deserialization
    if let Ok(ledger) = serde_json::from_slice::<EvidenceLedger>(data) {
        // Verify entries are accessible
        let entries = ledger.entries();
        let _ = entries.len();

        // Round-trip check
        let re_encoded = serde_json::to_vec(&ledger).expect("re-encode must succeed");
        let re_decoded: EvidenceLedger =
            serde_json::from_slice(&re_encoded).expect("round-trip must succeed");
        assert_eq!(ledger, re_decoded, "round-trip must be stable");
    }
});
