#![no_main]

use fj_core::Value;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 16 * 1024 {
        return;
    }

    if let Ok(value) = serde_json::from_slice::<Value>(data) {
        let encoded = serde_json::to_vec(&value).expect("Value serialization should not fail");
        let decoded: Value =
            serde_json::from_slice(&encoded).expect("serialized Value should deserialize");
        assert_eq!(value, decoded);

        let _ = value.dtype();
        let _ = value.as_scalar_literal();
        let _ = value.as_f64_scalar();
        let _ = value.as_i64_scalar();
        let _ = value.as_bool_scalar();
        let _ = value.as_complex128_scalar();

        if let Some(tensor) = value.as_tensor() {
            let _ = tensor.len();
            let _ = tensor.is_empty();
            let _ = tensor.rank();
            let _ = tensor.leading_dim();
            if tensor.shape.element_count() == Some(tensor.elements.len() as u64) {
                let max_axis0 = tensor.leading_dim().unwrap_or(0).min(8);
                for index in 0..max_axis0 {
                    let _ = tensor.slice_axis0(index as usize);
                }
            }
        }
    }
});
