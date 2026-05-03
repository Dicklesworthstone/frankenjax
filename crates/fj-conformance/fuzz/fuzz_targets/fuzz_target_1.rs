#![no_main]

use fj_cache::backend::CachedArtifact;
use fj_cache::persistence::{deserialize, serialize};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz the cache persistence wire format deserializer.
    // All inputs must either parse successfully (round-trip check) or
    // return a typed error — never panic.
    match deserialize(data) {
        Ok(artifact) => {
            // If deserialization succeeds, verify round-trip stability:
            // re-serializing and deserializing must produce the same artifact.
            let re_serialized = serialize(&artifact);
            let re_deserialized =
                deserialize(&re_serialized).expect("round-trip re-deserialization must succeed");
            assert_eq!(artifact.data, re_deserialized.data);
            assert_eq!(
                artifact.integrity_sha256_hex,
                re_deserialized.integrity_sha256_hex
            );
        }
        Err(_) => {
            // Expected: most random inputs should fail cleanly.
        }
    }

    // Also fuzz serialization with arbitrary payload data to ensure
    // serialize() never panics, and the result always round-trips.
    if data.len() <= 4096 {
        let artifact = CachedArtifact {
            data: data.to_vec(),
            integrity_sha256_hex: String::new(), // ignored by serialize
        };
        let serialized = serialize(&artifact);
        let restored =
            deserialize(&serialized).expect("serialized artifact must always deserialize");
        assert_eq!(restored.data, data);
    }
});
