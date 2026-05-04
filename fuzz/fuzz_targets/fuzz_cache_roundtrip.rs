#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ArtifactInput {
    data: Vec<u8>,
}

fuzz_target!(|input: ArtifactInput| {
    if input.data.len() > 100_000 {
        return;
    }
    let artifact = fj_cache::CachedArtifact {
        data: input.data.clone(),
        integrity_sha256_hex: fj_cache::sha256_hex(&input.data),
    };
    let serialized = fj_cache::persistence::serialize(&artifact);
    let restored = fj_cache::persistence::deserialize(&serialized)
        .expect("round-trip must succeed");
    assert_eq!(restored.data, artifact.data);
    assert_eq!(restored.integrity_sha256_hex, artifact.integrity_sha256_hex);
});
