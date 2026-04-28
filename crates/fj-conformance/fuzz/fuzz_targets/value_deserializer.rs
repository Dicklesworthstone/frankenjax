#![no_main]

use fj_core::Value;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 16 * 1024 {
        return;
    }

    if let Ok(value) = serde_json::from_slice::<Value>(data) {
        let Ok(encoded) = serde_json::to_vec(&value) else {
            return;
        };
        let Ok(decoded) = serde_json::from_slice::<Value>(&encoded) else {
            return;
        };
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
            let element_len = u64::try_from(tensor.elements.len()).ok();
            if tensor.shape.element_count() == element_len {
                let max_axis0 = tensor.leading_dim().unwrap_or(0).min(8);
                for index in 0..max_axis0 {
                    let Ok(index) = usize::try_from(index) else {
                        continue;
                    };
                    let _ = tensor.slice_axis0(index);
                }
            }
        }
    }
});
