#![no_main]

//! Fuzz target for primitive evaluation WITH params.
//!
//! Unlike primitive_eval_fuzzer which uses empty params, this target generates
//! random parameter maps for primitives that use them (reduce axes, reshape
//! dimensions, slice bounds, etc.).

mod common;

use common::{
    ByteCursor, primitive_arity, sample_primitive, sample_primitive_params, sample_value,
};
use fj_lax::eval_primitive;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }

    let mut cursor = ByteCursor::new(data);
    let primitive = sample_primitive(&mut cursor);
    let arity = primitive_arity(primitive);

    let mut inputs = Vec::with_capacity(arity);
    for _ in 0..arity {
        inputs.push(sample_value(&mut cursor));
    }

    let params = sample_primitive_params(&mut cursor, primitive);

    if let Ok(value) = eval_primitive(primitive, &inputs, &params) {
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
});
