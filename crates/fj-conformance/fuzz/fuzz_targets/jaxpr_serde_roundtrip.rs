#![no_main]

//! Fuzz target verifying Jaxpr serialization roundtrip.
//!
//! Tests that: deserialize(serialize(jaxpr)) == jaxpr
//! This catches any serde implementation bugs where data is lost or
//! corrupted during the serialization/deserialization cycle.

use fj_core::Jaxpr;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }

    let Ok(original) = serde_json::from_slice::<Jaxpr>(data) else {
        return;
    };

    if original.validate_well_formed().is_err() {
        return;
    }

    let Ok(serialized) = serde_json::to_vec(&original) else {
        panic!("Jaxpr that deserialized successfully should serialize");
    };

    let Ok(roundtripped) = serde_json::from_slice::<Jaxpr>(&serialized) else {
        panic!("Jaxpr that serialized should deserialize back");
    };

    assert_eq!(
        original, roundtripped,
        "Jaxpr roundtrip should produce identical value"
    );

    assert_eq!(
        original.canonical_fingerprint(),
        roundtripped.canonical_fingerprint(),
        "Jaxpr fingerprint should be identical after roundtrip"
    );

    let original_equations = original.equations.len();
    let roundtripped_equations = roundtripped.equations.len();
    assert_eq!(
        original_equations, roundtripped_equations,
        "Equation count should match after roundtrip"
    );

    for (i, (orig_eq, rt_eq)) in original.equations.iter().zip(roundtripped.equations.iter()).enumerate() {
        assert_eq!(
            orig_eq.primitive, rt_eq.primitive,
            "Equation {} primitive mismatch after roundtrip",
            i
        );
        assert_eq!(
            orig_eq.inputs.len(),
            rt_eq.inputs.len(),
            "Equation {} input count mismatch after roundtrip",
            i
        );
        assert_eq!(
            orig_eq.outputs.len(),
            rt_eq.outputs.len(),
            "Equation {} output count mismatch after roundtrip",
            i
        );
        assert_eq!(
            orig_eq.params, rt_eq.params,
            "Equation {} params mismatch after roundtrip",
            i
        );
    }
});
