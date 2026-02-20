//! FFI call dispatch — contains the ONLY unsafe code in the workspace.
//!
//! `FfiCall::invoke()` is the sole method that crosses the FFI boundary.
//! All preconditions are validated before the unsafe block is entered.

use crate::buffer::FfiBuffer;
use crate::error::FfiError;
use crate::registry::{FfiRegistry, FfiTarget};

/// Encapsulates an FFI call with pre-validated parameters.
#[derive(Debug)]
pub struct FfiCall {
    target_name: String,
}

impl FfiCall {
    /// Create a new FFI call for the named target.
    pub fn new(target_name: &str) -> Self {
        FfiCall {
            target_name: target_name.to_string(),
        }
    }

    /// Target name this call will dispatch to.
    pub fn target_name(&self) -> &str {
        &self.target_name
    }

    /// Invoke the FFI function with the given inputs, writing results to outputs.
    ///
    /// # Pre-call validation (all checked before entering unsafe):
    /// 1. Target exists in registry
    /// 2. All input buffer sizes match their declared dtype * shape
    /// 3. All output buffer sizes match their declared dtype * shape
    /// 4. Output buffers are pre-zeroed
    ///
    /// # Post-call:
    /// - Non-zero return code → `FfiError::CallFailed`
    /// - Zero return code → output buffers contain the result
    #[allow(unsafe_code)]
    pub fn invoke(
        &self,
        registry: &FfiRegistry,
        inputs: &[FfiBuffer],
        outputs: &mut [FfiBuffer],
    ) -> Result<(), FfiError> {
        // 1. Resolve target
        let target: FfiTarget = registry.get(&self.target_name)?;

        // 2. Validate input buffer sizes (redundant with construction, but defense-in-depth)
        for (i, buf) in inputs.iter().enumerate() {
            let expected = crate::buffer::checked_buffer_size(buf.shape(), buf.dtype())?;
            if buf.size() != expected {
                return Err(FfiError::BufferMismatch {
                    buffer_index: i,
                    expected_bytes: expected,
                    actual_bytes: buf.size(),
                });
            }
        }

        // 3. Validate output buffer sizes
        for (i, buf) in outputs.iter().enumerate() {
            let expected = crate::buffer::checked_buffer_size(buf.shape(), buf.dtype())?;
            if buf.size() != expected {
                return Err(FfiError::BufferMismatch {
                    buffer_index: inputs.len() + i,
                    expected_bytes: expected,
                    actual_bytes: buf.size(),
                });
            }
        }

        // 4. Build raw pointer arrays for the FFI boundary
        let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();
        let mut output_ptrs: Vec<*mut u8> = outputs.iter_mut().map(|b| b.as_mut_ptr()).collect();

        // 5. Call the external function
        // SAFETY:
        // - fn_ptr is non-null (validated at registration time by FfiRegistry.register())
        // - input_ptrs point to valid, immutable memory owned by `inputs` (alive for this scope)
        // - output_ptrs point to valid, mutable memory owned by `outputs` (alive for this scope)
        // - input_count and output_count match the actual array lengths
        // - The extern "C" function must not free, resize, or retain any buffer pointers
        // - The extern "C" function must write at most `buf.size()` bytes to each output
        let return_code = unsafe {
            (target.fn_ptr)(
                input_ptrs.as_ptr(),
                input_ptrs.len(),
                output_ptrs.as_mut_ptr(),
                output_ptrs.len(),
            )
        };

        // 6. Check return code
        if return_code != 0 {
            return Err(FfiError::CallFailed {
                target: self.target_name.clone(),
                code: return_code,
                message: format!("extern function returned code {return_code}"),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use crate::registry::FfiFnPtr;
    use fj_core::DType;

    /// Mock FFI function: copies first input to first output (8 bytes).
    unsafe extern "C" fn mock_copy(
        inputs: *const *const u8,
        input_count: usize,
        outputs: *const *mut u8,
        output_count: usize,
    ) -> i32 {
        if input_count == 0 || output_count == 0 {
            return 1;
        }
        unsafe {
            let src = *inputs;
            let dst = *outputs;
            std::ptr::copy_nonoverlapping(src, dst, 8);
        }
        0
    }

    /// Mock FFI function: always fails with code 99.
    unsafe extern "C" fn mock_fail(
        _inputs: *const *const u8,
        _input_count: usize,
        _outputs: *const *mut u8,
        _output_count: usize,
    ) -> i32 {
        99
    }

    /// Mock FFI function: doubles each f64 in the input.
    unsafe extern "C" fn mock_double(
        inputs: *const *const u8,
        _input_count: usize,
        outputs: *const *mut u8,
        _output_count: usize,
    ) -> i32 {
        unsafe {
            let src = *inputs as *const f64;
            let dst = *outputs as *mut f64;
            let val = *src;
            *dst = val * 2.0;
        }
        0
    }

    fn setup_registry(name: &str, fn_ptr: FfiFnPtr) -> FfiRegistry {
        let reg = FfiRegistry::new();
        reg.register(name, fn_ptr).unwrap();
        reg
    }

    #[test]
    fn invoke_copy_success() {
        let reg = setup_registry("copy", mock_copy);
        let call = FfiCall::new("copy");

        let input_data: f64 = 42.0;
        let input = FfiBuffer::new(input_data.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
        let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];

        call.invoke(&reg, &[input], &mut outputs).unwrap();

        let result_bytes: [u8; 8] = outputs[0].as_bytes().try_into().unwrap();
        let result = f64::from_ne_bytes(result_bytes);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn invoke_target_not_found() {
        let reg = FfiRegistry::new();
        let call = FfiCall::new("nonexistent");
        let err = call.invoke(&reg, &[], &mut []).unwrap_err();
        assert!(matches!(err, FfiError::TargetNotFound { .. }));
    }

    #[test]
    fn invoke_error_return_code() {
        let reg = setup_registry("fail", mock_fail);
        let call = FfiCall::new("fail");

        let input = FfiBuffer::new(vec![0u8; 8], vec![], DType::F64).unwrap();
        let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];

        let err = call.invoke(&reg, &[input], &mut outputs).unwrap_err();
        match err {
            FfiError::CallFailed { target, code, .. } => {
                assert_eq!(target, "fail");
                assert_eq!(code, 99);
            }
            other => panic!("expected CallFailed, got: {other}"),
        }
    }

    #[test]
    fn invoke_double_produces_correct_result() {
        let reg = setup_registry("double", mock_double);
        let call = FfiCall::new("double");

        let input_val: f64 = 21.0;
        let input = FfiBuffer::new(input_val.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
        let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];

        call.invoke(&reg, &[input], &mut outputs).unwrap();

        let result_bytes: [u8; 8] = outputs[0].as_bytes().try_into().unwrap();
        let result = f64::from_ne_bytes(result_bytes);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn invoke_no_inputs_no_outputs() {
        unsafe extern "C" fn mock_noop(
            _inputs: *const *const u8,
            _input_count: usize,
            _outputs: *const *mut u8,
            _output_count: usize,
        ) -> i32 {
            0
        }

        let reg = setup_registry("noop", mock_noop);
        let call = FfiCall::new("noop");
        call.invoke(&reg, &[], &mut []).unwrap();
    }

    #[test]
    fn invoke_vector_input_output() {
        unsafe extern "C" fn mock_negate_vec(
            inputs: *const *const u8,
            _input_count: usize,
            outputs: *const *mut u8,
            _output_count: usize,
        ) -> i32 {
            unsafe {
                let src = *inputs as *const f64;
                let dst = *outputs as *mut f64;
                for i in 0..3 {
                    *dst.add(i) = -(*src.add(i));
                }
            }
            0
        }

        let reg = setup_registry("negate3", mock_negate_vec);
        let call = FfiCall::new("negate3");

        let mut input_data = Vec::new();
        for &v in &[1.0f64, 2.0, 3.0] {
            input_data.extend_from_slice(&v.to_ne_bytes());
        }
        let input = FfiBuffer::new(input_data, vec![3], DType::F64).unwrap();
        let mut outputs = [FfiBuffer::zeroed(vec![3], DType::F64).unwrap()];

        call.invoke(&reg, &[input], &mut outputs).unwrap();

        let result_bytes = outputs[0].as_bytes();
        for (i, expected) in [-1.0f64, -2.0, -3.0].iter().enumerate() {
            let bytes: [u8; 8] = result_bytes[i * 8..(i + 1) * 8].try_into().unwrap();
            assert_eq!(f64::from_ne_bytes(bytes), *expected);
        }
    }
}
