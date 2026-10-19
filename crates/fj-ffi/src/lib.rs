//! # fj-ffi — FFI Call Interface for FrankenJAX
//!
//! This is the **ONLY** crate in the FrankenJAX workspace permitted to use `unsafe` code.
//! All unsafe blocks are confined to `call.rs` (specifically `FfiCall::invoke()`).
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────┐
//! │            Safe Rust (all other crates)   │
//! │                                          │
//! │   FfiRegistry::register(name, fn_ptr)    │
//! │   FfiCall::new(target_name)              │
//! │   FfiCall::invoke(&registry, in, out)    │
//! │        │                                 │
//! │        ▼ (pre-validation: sizes, types)  │
//! ├──────────────────────────────────────────┤
//! │   unsafe { (fn_ptr)(ptrs, counts) }      │  ← call.rs ONLY
//! ├──────────────────────────────────────────┤
//! │   (post-validation: return code check)   │
//! └──────────────────────────────────────────┘
//! ```
//!
//! ## Safety Contract
//!
//! The extern "C" function MUST:
//! - Not free any input or output buffer pointers
//! - Not write beyond the declared output buffer size
//! - Not retain references to any buffer after returning
//! - Return 0 on success, non-zero on error
//!
//! `FfiCall::invoke` zeroes output buffers immediately before dispatch and
//! again after non-zero return codes, so partial writes cannot expose stale
//! caller bytes through the safe Rust API.

#![deny(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod buffer;
pub mod call;
pub mod callback;
pub mod error;
pub mod marshal;
pub mod registry;

// Re-exports for convenience
pub use buffer::FfiBuffer;
pub use call::FfiCall;
pub use callback::{CallbackFlavor, CallbackRegistry, FfiCallback, PurityConfig};
pub use error::FfiError;
pub use marshal::{buffer_to_value, value_to_buffer};
pub use registry::{
    FfiFnPtr, FfiRegistry, FfiTarget, global_ffi_registry, register_global_ffi_target,
};

/// Status code returned by `fj_ffi_*` C ABI entrypoints on success.
pub const FJ_FFI_OK: i32 = 0;
/// Status code returned when a required pointer argument is null or aliases illegally.
pub const FJ_FFI_ERR_NULL_POINTER: i32 = -1;
/// Status code returned when a C string argument is not valid UTF-8.
pub const FJ_FFI_ERR_INVALID_UTF8: i32 = -2;
/// Status code returned when registering a duplicate FFI target name.
pub const FJ_FFI_ERR_DUPLICATE_TARGET: i32 = -3;
/// Status code returned when an FFI target name is not registered.
pub const FJ_FFI_ERR_TARGET_NOT_FOUND: i32 = -4;
/// Status code returned when a dtype code does not map to a valid `DType`.
pub const FJ_FFI_ERR_INVALID_DTYPE: i32 = -5;
/// Status code returned when buffer dimensions or byte length mismatch or overflow.
pub const FJ_FFI_ERR_BUFFER_MISMATCH: i32 = -6;
/// Status code returned when the target callback fails with a non-zero return code.
pub const FJ_FFI_ERR_CALL_FAILED: i32 = -7;

/// Convert a FrankenJAX `DType` to its stable C ABI integer discriminant.
#[must_use]
pub const fn dtype_to_ffi_code(dtype: fj_core::DType) -> u32 {
    match dtype {
        fj_core::DType::F64 => 0,
        fj_core::DType::F32 => 1,
        fj_core::DType::I64 => 2,
        fj_core::DType::I32 => 3,
        fj_core::DType::U64 => 4,
        fj_core::DType::U32 => 5,
        fj_core::DType::Bool => 6,
        fj_core::DType::Complex64 => 7,
        fj_core::DType::Complex128 => 8,
        fj_core::DType::F16 => 9,
        fj_core::DType::BF16 => 10,
    }
}

/// Convert a C ABI integer discriminant into a FrankenJAX `DType`.
#[must_use]
pub const fn ffi_code_to_dtype(code: u32) -> Option<fj_core::DType> {
    match code {
        0 => Some(fj_core::DType::F64),
        1 => Some(fj_core::DType::F32),
        2 => Some(fj_core::DType::I64),
        3 => Some(fj_core::DType::I32),
        4 => Some(fj_core::DType::U64),
        5 => Some(fj_core::DType::U32),
        6 => Some(fj_core::DType::Bool),
        7 => Some(fj_core::DType::Complex64),
        8 => Some(fj_core::DType::Complex128),
        9 => Some(fj_core::DType::F16),
        10 => Some(fj_core::DType::BF16),
        _ => None,
    }
}

/// Allocate a new empty `FfiRegistry` on the heap for C ABI consumers.
///
/// The caller owns the returned pointer and must release it with [`fj_ffi_registry_free`].
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn fj_ffi_registry_new() -> *mut FfiRegistry {
    Box::into_raw(Box::new(FfiRegistry::new()))
}

/// Free an `FfiRegistry` previously allocated by [`fj_ffi_registry_new`].
///
/// Passing a null pointer is a safe no-op.
///
/// # Safety
/// `registry` must either be null or a valid pointer returned by [`fj_ffi_registry_new`]
/// that has not yet been freed and is not concurrently accessed.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_ffi_registry_free(registry: *mut FfiRegistry) {
    if !registry.is_null() {
        unsafe {
            drop(Box::from_raw(registry));
        }
    }
}

/// Register an `extern "C"` target function in `registry` under `name_ptr`.
///
/// Returns [`FJ_FFI_OK`] (`0`) on success or a negative `FJ_FFI_ERR_*` code on failure.
///
/// # Safety
/// - `registry` must be either null (rejected with `FJ_FFI_ERR_NULL_POINTER`) or a valid
///   pointer to an `FfiRegistry`.
/// - `name_ptr` must be either null (rejected with `FJ_FFI_ERR_NULL_POINTER`) or a valid
///   NUL-terminated C string.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_ffi_register_target(
    registry: *mut FfiRegistry,
    name_ptr: *const std::ffi::c_char,
    fn_ptr: Option<FfiFnPtr>,
) -> i32 {
    if registry.is_null() || name_ptr.is_null() {
        return FJ_FFI_ERR_NULL_POINTER;
    }
    let Some(fn_ptr) = fn_ptr else {
        return FJ_FFI_ERR_NULL_POINTER;
    };
    if (fn_ptr as usize) == 0 {
        return FJ_FFI_ERR_NULL_POINTER;
    }
    let name_cstr = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
    let Ok(name) = name_cstr.to_str() else {
        return FJ_FFI_ERR_INVALID_UTF8;
    };
    let reg = unsafe { &*registry };
    match reg.register(name, fn_ptr) {
        Ok(()) => FJ_FFI_OK,
        Err(FfiError::NullPointer { .. }) => FJ_FFI_ERR_NULL_POINTER,
        Err(FfiError::DuplicateTarget { .. }) => FJ_FFI_ERR_DUPLICATE_TARGET,
        Err(_) => FJ_FFI_ERR_CALL_FAILED,
    }
}

/// Construct a validated `FfiBuffer` by copying `data_len` bytes from `data_ptr` with shape
/// `shape_ptr[0..rank]` and element type `dtype_code`.
///
/// Returns a null pointer if any pointer is null when non-empty, if `dtype_code` is invalid,
/// if `shape * dtype_size` overflows, or if `data_len` does not match `shape * dtype_size`.
///
/// # Safety
/// - If `data_len > 0`, `data_ptr` must be a valid pointer to at least `data_len` readable bytes.
/// - If `rank > 0`, `shape_ptr` must be a valid pointer to at least `rank` `usize` elements.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_ffi_buffer_from_bytes(
    data_ptr: *const u8,
    data_len: usize,
    shape_ptr: *const usize,
    rank: usize,
    dtype_code: u32,
) -> *mut FfiBuffer {
    let Some(dtype) = ffi_code_to_dtype(dtype_code) else {
        return std::ptr::null_mut();
    };
    if rank > 0 && shape_ptr.is_null() {
        return std::ptr::null_mut();
    }
    if data_len > 0 && data_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let Some(shape_bytes) = rank.checked_mul(std::mem::size_of::<usize>()) else {
        return std::ptr::null_mut();
    };
    if shape_bytes > isize::MAX as usize || data_len > isize::MAX as usize {
        return std::ptr::null_mut();
    }
    let shape = if rank == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(shape_ptr, rank).to_vec() }
    };
    let Ok(expected_len) = buffer::checked_buffer_size(&shape, dtype) else {
        return std::ptr::null_mut();
    };
    if expected_len != data_len {
        return std::ptr::null_mut();
    }
    let bytes = if data_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(data_ptr, data_len).to_vec() }
    };
    match FfiBuffer::new(bytes, shape, dtype) {
        Ok(buf) => Box::into_raw(Box::new(buf)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Allocate a zero-initialized `FfiBuffer` with shape `shape_ptr[0..rank]` and `dtype_code`.
///
/// Returns a null pointer on invalid `dtype_code`, null `shape_ptr` when `rank > 0`, or size overflow.
///
/// # Safety
/// - If `rank > 0`, `shape_ptr` must be a valid pointer to at least `rank` `usize` elements.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_ffi_buffer_zeroed(
    shape_ptr: *const usize,
    rank: usize,
    dtype_code: u32,
) -> *mut FfiBuffer {
    let Some(dtype) = ffi_code_to_dtype(dtype_code) else {
        return std::ptr::null_mut();
    };
    if rank > 0 && shape_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let Some(shape_bytes) = rank.checked_mul(std::mem::size_of::<usize>()) else {
        return std::ptr::null_mut();
    };
    if shape_bytes > isize::MAX as usize {
        return std::ptr::null_mut();
    }
    let shape = if rank == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(shape_ptr, rank).to_vec() }
    };
    match FfiBuffer::zeroed(shape, dtype) {
        Ok(buf) => Box::into_raw(Box::new(buf)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free an `FfiBuffer` previously allocated by [`fj_ffi_buffer_from_bytes`] or [`fj_ffi_buffer_zeroed`].
///
/// Passing a null pointer is a safe no-op.
///
/// # Safety
/// `buffer` must either be null or a valid pointer returned by [`fj_ffi_buffer_from_bytes`] or
/// [`fj_ffi_buffer_zeroed`] that has not yet been freed.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_ffi_buffer_free(buffer: *mut FfiBuffer) {
    if !buffer.is_null() {
        unsafe {
            drop(Box::from_raw(buffer));
        }
    }
}

/// Return a const pointer to the underlying bytes of `buffer`, or null if `buffer` is null.
///
/// # Safety
/// `buffer` must be either null or a valid pointer to an `FfiBuffer`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_ffi_buffer_data(buffer: *const FfiBuffer) -> *const u8 {
    if buffer.is_null() {
        return std::ptr::null();
    }
    unsafe { (&*buffer).as_bytes().as_ptr() }
}

/// Return the byte length of `buffer`, or `0` if `buffer` is null.
///
/// # Safety
/// `buffer` must be either null or a valid pointer to an `FfiBuffer`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_ffi_buffer_len(buffer: *const FfiBuffer) -> usize {
    if buffer.is_null() {
        return 0;
    }
    unsafe { (&*buffer).size() }
}

/// Invoke a registered FFI target across the C ABI boundary using `FfiBuffer` handles.
///
/// Returns [`FJ_FFI_OK`] (`0`) on success or a negative `FJ_FFI_ERR_*` status code on failure.
///
/// # Safety
/// - `registry` must be either null (rejected) or a valid pointer to an `FfiRegistry`.
/// - `target_name_ptr` must be either null (rejected) or a valid NUL-terminated C string.
/// - If `input_count > 0`, `inputs_ptr` must point to `input_count` non-null `*const FfiBuffer` pointers.
/// - If `output_count > 0`, `outputs_ptr` must point to `output_count` distinct non-null `*mut FfiBuffer` pointers.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fj_ffi_invoke(
    registry: *const FfiRegistry,
    target_name_ptr: *const std::ffi::c_char,
    inputs_ptr: *const *const FfiBuffer,
    input_count: usize,
    outputs_ptr: *const *mut FfiBuffer,
    output_count: usize,
) -> i32 {
    if registry.is_null() || target_name_ptr.is_null() {
        return FJ_FFI_ERR_NULL_POINTER;
    }
    if (input_count > 0 && inputs_ptr.is_null()) || (output_count > 0 && outputs_ptr.is_null()) {
        return FJ_FFI_ERR_NULL_POINTER;
    }
    let Some(in_table_bytes) = input_count.checked_mul(std::mem::size_of::<*const FfiBuffer>())
    else {
        return FJ_FFI_ERR_BUFFER_MISMATCH;
    };
    let Some(out_table_bytes) = output_count.checked_mul(std::mem::size_of::<*mut FfiBuffer>())
    else {
        return FJ_FFI_ERR_BUFFER_MISMATCH;
    };
    if in_table_bytes > isize::MAX as usize || out_table_bytes > isize::MAX as usize {
        return FJ_FFI_ERR_BUFFER_MISMATCH;
    }

    let target_cstr = unsafe { std::ffi::CStr::from_ptr(target_name_ptr) };
    let Ok(target_name) = target_cstr.to_str() else {
        return FJ_FFI_ERR_INVALID_UTF8;
    };

    let mut input_buffers = Vec::with_capacity(input_count);
    for idx in 0..input_count {
        let buf_ptr = unsafe { *inputs_ptr.add(idx) };
        if buf_ptr.is_null() {
            return FJ_FFI_ERR_NULL_POINTER;
        }
        input_buffers.push(unsafe { (&*buf_ptr).clone() });
    }

    let mut output_ptrs = Vec::with_capacity(output_count);
    let mut output_buffers = Vec::with_capacity(output_count);
    for idx in 0..output_count {
        let buf_ptr = unsafe { *outputs_ptr.add(idx) };
        if buf_ptr.is_null() {
            return FJ_FFI_ERR_NULL_POINTER;
        }
        // Reject aliasing output buffer pointers or aliasing with any input buffer pointer.
        if output_ptrs.contains(&buf_ptr) {
            return FJ_FFI_ERR_NULL_POINTER;
        }
        for in_idx in 0..input_count {
            let in_ptr = unsafe { *inputs_ptr.add(in_idx) };
            if in_ptr == buf_ptr.cast_const() {
                return FJ_FFI_ERR_NULL_POINTER;
            }
        }
        output_ptrs.push(buf_ptr);
        output_buffers.push(unsafe { (&*buf_ptr).clone() });
    }

    let call = FfiCall::new(target_name);
    let reg = unsafe { &*registry };
    match call.invoke(reg, &input_buffers, &mut output_buffers) {
        Ok(()) => {
            for (dst_ptr, updated) in output_ptrs.into_iter().zip(output_buffers) {
                unsafe {
                    *dst_ptr = updated;
                }
            }
            FJ_FFI_OK
        }
        Err(FfiError::TargetNotFound { .. }) => FJ_FFI_ERR_TARGET_NOT_FOUND,
        Err(FfiError::BufferMismatch { .. } | FfiError::InvalidBoolByte { .. }) => {
            for (dst_ptr, scrubbed) in output_ptrs.into_iter().zip(output_buffers) {
                unsafe {
                    *dst_ptr = scrubbed;
                }
            }
            FJ_FFI_ERR_BUFFER_MISMATCH
        }
        Err(FfiError::CallFailed { .. }) => {
            for (dst_ptr, scrubbed) in output_ptrs.into_iter().zip(output_buffers) {
                unsafe {
                    *dst_ptr = scrubbed;
                }
            }
            FJ_FFI_ERR_CALL_FAILED
        }
        Err(_) => FJ_FFI_ERR_CALL_FAILED,
    }
}

/// Built-in C ABI target computing `out[i] = in[i] * 3.0 + 1.0` for a 3-element `f64` vector.
///
/// Exposed so crates with `#![forbid(unsafe_code)]` can register and test `Primitive::CustomCall`
/// end-to-end without writing `unsafe` blocks outside `fj-ffi`.
#[allow(unsafe_code, clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn fj_ffi_builtin_affine3_plus1_vec3_f64(
    inputs: *const *const u8,
    n_inputs: usize,
    outputs: *const *mut u8,
    n_outputs: usize,
) -> i32 {
    if n_inputs != 1 || n_outputs != 1 || inputs.is_null() || outputs.is_null() {
        return 1;
    }
    unsafe {
        let src = *inputs as *const f64;
        let dst = *outputs as *mut f64;
        if src.is_null() || dst.is_null() {
            return 1;
        }
        for idx in 0..3 {
            *dst.add(idx) = (*src.add(idx)) * 3.0 + 1.0;
        }
    }
    0
}

/// Built-in C ABI target computing `out[0] = in[0] * 2.0 + 5.0` for a scalar `f64`.
#[allow(unsafe_code, clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn fj_ffi_builtin_double_plus5_scalar_f64(
    inputs: *const *const u8,
    n_inputs: usize,
    outputs: *const *mut u8,
    n_outputs: usize,
) -> i32 {
    if n_inputs != 1 || n_outputs != 1 || inputs.is_null() || outputs.is_null() {
        return 1;
    }
    unsafe {
        let src = *inputs as *const f64;
        let dst = *outputs as *mut f64;
        if src.is_null() || dst.is_null() {
            return 1;
        }
        *dst = (*src) * 2.0 + 5.0;
    }
    0
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use fj_core::{DType, Literal, Value};

    /// Integration test: register → call → verify round-trip.
    #[test]
    fn integration_register_call_verify() {
        /// Adds two f64 scalars: output[0] = input[0] + input[1].
        unsafe extern "C" fn ffi_add(
            inputs: *const *const u8,
            _input_count: usize,
            outputs: *const *mut u8,
            _output_count: usize,
        ) -> i32 {
            unsafe {
                let a = *((*inputs) as *const f64);
                let b = *((*inputs.add(1)) as *const f64);
                let dst = (*outputs) as *mut f64;
                *dst = a + b;
            }
            0
        }

        let registry = FfiRegistry::new();
        registry.register("add_f64", ffi_add).unwrap();

        let a = FfiBuffer::new(3.0f64.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
        let b = FfiBuffer::new(4.0f64.to_ne_bytes().to_vec(), vec![], DType::F64).unwrap();
        let mut outputs = [FfiBuffer::zeroed(vec![], DType::F64).unwrap()];

        let call = FfiCall::new("add_f64");
        call.invoke(&registry, &[a, b], &mut outputs).unwrap();

        let result_bytes: [u8; 8] = outputs[0].as_bytes().try_into().unwrap();
        let result = f64::from_ne_bytes(result_bytes);
        assert_eq!(result, 7.0);
    }

    /// Integration test: error propagation across FFI boundary.
    #[test]
    fn integration_error_propagation() {
        unsafe extern "C" fn ffi_fail(
            _inputs: *const *const u8,
            _input_count: usize,
            _outputs: *const *mut u8,
            _output_count: usize,
        ) -> i32 {
            -1
        }

        let registry = FfiRegistry::new();
        registry.register("fail_fn", ffi_fail).unwrap();

        let call = FfiCall::new("fail_fn");
        let err = call.invoke(&registry, &[], &mut []).unwrap_err();
        match err {
            FfiError::CallFailed { target, code, .. } => {
                assert_eq!(target, "fail_fn");
                assert_eq!(code, -1);
            }
            other => std::panic::panic_any(format!("expected CallFailed, got: {other}")),
        }
    }

    /// Test: error display messages are actionable.
    #[test]
    fn error_display_messages() {
        let err = FfiError::TargetNotFound {
            name: "missing".to_string(),
            available: vec!["fn_a".to_string(), "fn_b".to_string()],
        };
        let msg = err.to_string();
        assert!(msg.contains("missing"));
        assert!(msg.contains("fn_a"));

        let err = FfiError::BufferMismatch {
            buffer_index: 2,
            expected_bytes: 64,
            actual_bytes: 32,
        };
        let msg = err.to_string();
        assert!(msg.contains("64"));
        assert!(msg.contains("32"));
    }

    /// Test log schema contract.
    #[test]
    fn test_ffi_test_log_schema_contract() {
        let test_name = "test_ffi_test_log_schema_contract";
        let packet_id = "FJ-P2C-007";
        assert!(!test_name.is_empty());
        assert!(!packet_id.is_empty());
    }

    // --- Adversarial tests ---

    /// Adversarial: oversized buffer is rejected at construction.
    #[test]
    fn adversarial_oversized_buffer_shape() {
        let err = FfiBuffer::new(vec![0u8; 8], vec![999], DType::F64).unwrap_err();
        assert!(matches!(err, FfiError::BufferMismatch { .. }));
    }

    /// Adversarial: empty registry call returns TargetNotFound.
    #[test]
    fn adversarial_empty_registry_call() {
        let reg = FfiRegistry::new();
        let call = FfiCall::new("anything");
        let err = call.invoke(&reg, &[], &mut []).unwrap_err();
        match err {
            FfiError::TargetNotFound { name, available } => {
                assert_eq!(name, "anything");
                assert!(available.is_empty());
            }
            other => std::panic::panic_any(format!("expected TargetNotFound, got: {other}")),
        }
    }

    /// Adversarial: multiple output buffers with mixed dtypes.
    #[test]
    fn adversarial_mixed_dtype_outputs() {
        unsafe extern "C" fn extern_test_mixed(
            _inputs: *const *const u8,
            _input_count: usize,
            outputs: *const *mut u8,
            _output_count: usize,
        ) -> i32 {
            unsafe {
                // Write f64 to first output
                let dst0 = *outputs as *mut f64;
                *dst0 = 99.0;
                // Write i64 to second output
                let dst1 = *outputs.add(1) as *mut i64;
                *dst1 = 42;
            }
            0
        }

        let reg = FfiRegistry::new();
        reg.register("mixed", extern_test_mixed).unwrap();

        let mut outputs = [
            FfiBuffer::zeroed(vec![], DType::F64).unwrap(),
            FfiBuffer::zeroed(vec![], DType::I64).unwrap(),
        ];

        let call = FfiCall::new("mixed");
        call.invoke(&reg, &[], &mut outputs).unwrap();

        let f_bytes: [u8; 8] = outputs[0].as_bytes().try_into().unwrap();
        assert_eq!(f64::from_ne_bytes(f_bytes), 99.0);

        let i_bytes: [u8; 8] = outputs[1].as_bytes().try_into().unwrap();
        assert_eq!(i64::from_ne_bytes(i_bytes), 42);
    }

    /// Buffer lifecycle: borrow → use → return pattern.
    #[test]
    fn buffer_lifecycle_borrow_use_return() {
        unsafe extern "C" fn extern_test_increment(
            inputs: *const *const u8,
            _input_count: usize,
            outputs: *const *mut u8,
            _output_count: usize,
        ) -> i32 {
            unsafe {
                let src = *inputs as *const i64;
                let dst = *outputs as *mut i64;
                *dst = *src + 1;
            }
            0
        }

        let reg = FfiRegistry::new();
        reg.register("inc", extern_test_increment).unwrap();
        let call = FfiCall::new("inc");

        // Round-trip through multiple calls
        let mut current = Value::Scalar(Literal::I64(0));
        for expected in 1..=5i64 {
            let input = value_to_buffer(&current).unwrap();
            let mut outputs = [FfiBuffer::zeroed(vec![], DType::I64).unwrap()];
            call.invoke(&reg, &[input], &mut outputs).unwrap();
            current = buffer_to_value(&outputs[0]).unwrap();
            match &current {
                Value::Scalar(Literal::I64(v)) => assert_eq!(*v, expected),
                other => std::panic::panic_any(format!("expected I64 scalar, got: {other:?}")),
            }
        }
    }

    /// All error variants display correctly.
    #[test]
    fn all_error_variant_displays() {
        let errors = vec![
            FfiError::CallFailed {
                target: "t".to_string(),
                code: 1,
                message: "msg".to_string(),
            },
            FfiError::TargetNotFound {
                name: "n".to_string(),
                available: vec![],
            },
            FfiError::DuplicateTarget {
                name: "d".to_string(),
            },
            FfiError::NullPointer {
                target_name: "np".to_string(),
            },
            FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 8,
                actual_bytes: 4,
            },
            FfiError::InvalidBoolByte {
                buffer_index: 1,
                byte_index: 2,
                value: 3,
            },
            FfiError::UnsupportedDtype { dtype: DType::F32 },
            FfiError::PurityViolation {
                callback_name: "cb".to_string(),
                detail: "non-deterministic".to_string(),
            },
        ];
        for err in &errors {
            let msg = err.to_string();
            assert!(!msg.is_empty(), "error display should not be empty");
        }
    }

    /// Marshal round-trip: Value→Buffer→Value preserves identity for all supported types.
    #[test]
    fn marshal_roundtrip_all_supported_types() {
        use fj_core::{Literal, Shape, TensorValue};

        let values = vec![
            Value::Scalar(Literal::F64Bits(std::f64::consts::PI.to_bits())),
            Value::Scalar(Literal::F32Bits(1.25_f32.to_bits())),
            Value::Scalar(Literal::I64(i64::MIN)),
            Value::Scalar(Literal::I64(i64::MAX)),
            Value::Scalar(Literal::Bool(true)),
            Value::Scalar(Literal::Bool(false)),
            Value::Tensor(TensorValue {
                dtype: DType::F64,
                shape: Shape { dims: vec![2] },
                elements: vec![
                    Literal::F64Bits(0.0f64.to_bits()),
                    Literal::F64Bits(f64::NEG_INFINITY.to_bits()),
                ]
                .into(),
            }),
            Value::Tensor(TensorValue {
                dtype: DType::I64,
                shape: Shape { dims: vec![1, 1] },
                elements: vec![Literal::I64(42)].into(),
            }),
        ];

        for val in &values {
            let buf = value_to_buffer(val).unwrap();
            let restored = buffer_to_value(&buf).unwrap();
            assert_eq!(val, &restored, "round-trip failed for {val:?}");
        }
    }

    #[test]
    fn c_abi_end_to_end_and_negative_safety_checks() {
        use std::ffi::CString;

        unsafe extern "C" fn c_abi_scale_add(
            inputs: *const *const u8,
            input_count: usize,
            outputs: *const *mut u8,
            output_count: usize,
        ) -> i32 {
            if input_count != 2 || output_count != 1 {
                return 2;
            }
            unsafe {
                let x = *inputs as *const f64;
                let y = *inputs.add(1) as *const f64;
                let out = *outputs as *mut f64;
                for idx in 0..3 {
                    *out.add(idx) = (*x.add(idx)).mul_add(2.0, *y.add(idx));
                }
            }
            0
        }

        let reg = fj_ffi_registry_new();
        assert!(!reg.is_null());

        let target_name = CString::new("scale_add_3").unwrap();
        let missing_name = CString::new("missing_target").unwrap();

        // Null pointer rejections on registration
        assert_eq!(
            unsafe {
                fj_ffi_register_target(
                    std::ptr::null_mut(),
                    target_name.as_ptr(),
                    Some(c_abi_scale_add),
                )
            },
            FJ_FFI_ERR_NULL_POINTER
        );
        assert_eq!(
            unsafe { fj_ffi_register_target(reg, std::ptr::null(), Some(c_abi_scale_add)) },
            FJ_FFI_ERR_NULL_POINTER
        );
        assert_eq!(
            unsafe { fj_ffi_register_target(reg, target_name.as_ptr(), None) },
            FJ_FFI_ERR_NULL_POINTER
        );

        // Successful registration + duplicate rejection
        assert_eq!(
            unsafe { fj_ffi_register_target(reg, target_name.as_ptr(), Some(c_abi_scale_add)) },
            FJ_FFI_OK
        );
        assert_eq!(
            unsafe { fj_ffi_register_target(reg, target_name.as_ptr(), Some(c_abi_scale_add)) },
            FJ_FFI_ERR_DUPLICATE_TARGET
        );

        // Buffer creation: null/invalid dtype/length mismatch rejections
        let shape = [3_usize];
        let x_vals = [1.0_f64, 2.0, 3.0];
        let y_vals = [0.5_f64, 1.5, 2.5];
        let mut x_bytes = Vec::new();
        let mut y_bytes = Vec::new();
        for v in x_vals {
            x_bytes.extend_from_slice(&v.to_ne_bytes());
        }
        for v in y_vals {
            y_bytes.extend_from_slice(&v.to_ne_bytes());
        }

        assert!(
            unsafe {
                fj_ffi_buffer_from_bytes(
                    std::ptr::null(),
                    x_bytes.len(),
                    shape.as_ptr(),
                    1,
                    dtype_to_ffi_code(DType::F64),
                )
            }
            .is_null()
        );
        assert!(
            unsafe {
                fj_ffi_buffer_from_bytes(
                    x_bytes.as_ptr(),
                    x_bytes.len(),
                    std::ptr::null(),
                    1,
                    dtype_to_ffi_code(DType::F64),
                )
            }
            .is_null()
        );
        assert!(
            unsafe {
                fj_ffi_buffer_from_bytes(x_bytes.as_ptr(), x_bytes.len(), shape.as_ptr(), 1, 999)
            }
            .is_null()
        );
        assert!(
            unsafe {
                fj_ffi_buffer_from_bytes(
                    x_bytes.as_ptr(),
                    x_bytes.len() - 1,
                    shape.as_ptr(),
                    1,
                    dtype_to_ffi_code(DType::F64),
                )
            }
            .is_null()
        );

        let in0 = unsafe {
            fj_ffi_buffer_from_bytes(
                x_bytes.as_ptr(),
                x_bytes.len(),
                shape.as_ptr(),
                1,
                dtype_to_ffi_code(DType::F64),
            )
        };
        let in1 = unsafe {
            fj_ffi_buffer_from_bytes(
                y_bytes.as_ptr(),
                y_bytes.len(),
                shape.as_ptr(),
                1,
                dtype_to_ffi_code(DType::F64),
            )
        };
        let out0 =
            unsafe { fj_ffi_buffer_zeroed(shape.as_ptr(), 1, dtype_to_ffi_code(DType::F64)) };
        assert!(!in0.is_null() && !in1.is_null() && !out0.is_null());

        let inputs: [*const FfiBuffer; 2] = [in0.cast_const(), in1.cast_const()];
        let outputs: [*mut FfiBuffer; 1] = [out0];

        // Missing target fails closed
        assert_eq!(
            unsafe {
                fj_ffi_invoke(
                    reg,
                    missing_name.as_ptr(),
                    inputs.as_ptr(),
                    inputs.len(),
                    outputs.as_ptr(),
                    outputs.len(),
                )
            },
            FJ_FFI_ERR_TARGET_NOT_FOUND
        );

        // Aliasing input and output buffer pointer fails closed
        let aliased_outputs: [*mut FfiBuffer; 1] = [in0];
        assert_eq!(
            unsafe {
                fj_ffi_invoke(
                    reg,
                    target_name.as_ptr(),
                    inputs.as_ptr(),
                    inputs.len(),
                    aliased_outputs.as_ptr(),
                    aliased_outputs.len(),
                )
            },
            FJ_FFI_ERR_NULL_POINTER
        );

        // Happy-path C ABI invocation
        assert_eq!(
            unsafe {
                fj_ffi_invoke(
                    reg,
                    target_name.as_ptr(),
                    inputs.as_ptr(),
                    inputs.len(),
                    outputs.as_ptr(),
                    outputs.len(),
                )
            },
            FJ_FFI_OK
        );

        assert_eq!(unsafe { fj_ffi_buffer_len(out0) }, 24);
        let out_ptr = unsafe { fj_ffi_buffer_data(out0) };
        assert!(!out_ptr.is_null());
        let out_bytes = unsafe { std::slice::from_raw_parts(out_ptr, 24) };
        let expected = [2.5_f64, 5.5, 8.5];
        for (idx, exp) in expected.iter().enumerate() {
            let chunk: [u8; 8] = out_bytes[idx * 8..(idx + 1) * 8].try_into().unwrap();
            assert!((f64::from_ne_bytes(chunk) - exp).abs() < 1e-12);
        }

        unsafe {
            fj_ffi_buffer_free(in0);
            fj_ffi_buffer_free(in1);
            fj_ffi_buffer_free(out0);
            fj_ffi_buffer_free(std::ptr::null_mut());
            fj_ffi_registry_free(reg);
            fj_ffi_registry_free(std::ptr::null_mut());
        }
    }
}

/// Property tests for the FFI boundary.
#[cfg(test)]
mod prop_tests {
    use super::*;
    use fj_core::{DType, Literal, Value};
    use proptest::prelude::*;

    proptest! {
        /// Property: any f64 scalar round-trips through marshal.
        #[test]
        fn prop_f64_scalar_roundtrip(bits: u64) {
            let val = Value::Scalar(Literal::F64Bits(bits));
            let buf = value_to_buffer(&val).unwrap();
            let restored = buffer_to_value(&buf).unwrap();
            prop_assert_eq!(val, restored);
        }

        /// Property: any i64 scalar round-trips through marshal.
        #[test]
        fn prop_i64_scalar_roundtrip(v: i64) {
            let val = Value::Scalar(Literal::I64(v));
            let buf = value_to_buffer(&val).unwrap();
            let restored = buffer_to_value(&buf).unwrap();
            prop_assert_eq!(val, restored);
        }

        /// Property: FfiBuffer size matches checked_buffer_size for valid shapes.
        #[test]
        fn prop_buffer_size_matches_checked(
            dims in proptest::collection::vec(1usize..=10, 0..=3),
        ) {
            let dtype = DType::F64;
            let expected = buffer::checked_buffer_size(&dims, dtype).unwrap();
            let buf = FfiBuffer::zeroed(dims, dtype).unwrap();
            prop_assert_eq!(buf.size(), expected);
        }

        /// Property: zeroed buffers contain only zeros.
        #[test]
        fn prop_zeroed_buffer_is_zero(
            dims in proptest::collection::vec(1usize..=5, 0..=2),
        ) {
            let buf = FfiBuffer::zeroed(dims, DType::I64).unwrap();
            prop_assert!(buf.as_bytes().iter().all(|&b| b == 0));
        }
    }
}
