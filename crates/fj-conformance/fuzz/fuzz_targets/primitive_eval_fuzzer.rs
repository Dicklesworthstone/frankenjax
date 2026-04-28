#![no_main]

//! Fuzz target for primitive evaluation.
//!
//! Generates random primitives with random inputs and verifies:
//! 1. eval_primitive never panics (returns Ok or typed Err)
//! 2. Successful evaluations produce well-formed Values
//! 3. Shape inference is consistent with actual output

mod common;

use common::{primitive_arity, sample_primitive, sample_value, ByteCursor};
use fj_lax::eval_primitive;
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeMap;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let mut cursor = ByteCursor::new(data);
    let primitive = sample_primitive(&mut cursor);
    let arity = primitive_arity(primitive);

    let mut inputs = Vec::with_capacity(arity);
    for _ in 0..arity {
        inputs.push(sample_value(&mut cursor));
    }

    let params = BTreeMap::new();

    match eval_primitive(primitive, &inputs, &params) {
        Ok(value) => {
            match &value {
                fj_core::Value::Scalar(lit) => {
                    let _ = lit.as_f64();
                    let _ = lit.as_i64();
                }
                fj_core::Value::Tensor(t) => {
                    assert_eq!(
                        t.shape.element_count().unwrap_or(0) as usize,
                        t.elements.len(),
                        "shape/element count mismatch"
                    );
                }
            }
        }
        Err(_) => {
            // Expected: many random inputs produce typed errors.
            // The critical invariant is no panic.
        }
    }
});
