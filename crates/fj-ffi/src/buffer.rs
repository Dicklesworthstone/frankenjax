//! Safe buffer wrapper for FFI boundary crossing.
//!
//! `FfiBuffer` owns contiguous memory and tracks shape + dtype metadata.
//! It provides safe accessors for Rust code and raw pointer accessors
//! (confined to `call.rs`) for crossing the FFI boundary.

use fj_core::DType;

use crate::error::FfiError;

/// A contiguous buffer of bytes with shape and dtype metadata.
///
/// Memory is always owned by FrankenJAX. FFI functions receive borrows
/// via raw pointers that are valid only during `FfiCall::invoke()`.
#[derive(Debug, Clone)]
pub struct FfiBuffer {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
}

impl FfiBuffer {
    /// Create a new buffer from existing data.
    ///
    /// Validates that `data.len() == product(shape) * dtype_size_bytes(dtype)`.
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Result<Self, FfiError> {
        let expected = checked_buffer_size(&shape, dtype)?;
        if data.len() != expected {
            return Err(FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: expected,
                actual_bytes: data.len(),
            });
        }
        Ok(FfiBuffer { data, shape, dtype })
    }

    /// Create a zero-initialized output buffer of the declared size.
    pub fn zeroed(shape: Vec<usize>, dtype: DType) -> Result<Self, FfiError> {
        let size = checked_buffer_size(&shape, dtype)?;
        Ok(FfiBuffer {
            data: vec![0u8; size],
            shape,
            dtype,
        })
    }

    /// Read-only view of the buffer bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Mutable view of the buffer bytes (for output buffers).
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Consume the buffer, returning the raw bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Shape dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    // --- Raw pointer accessors (used only in call.rs unsafe block) ---

    /// Raw const pointer to buffer data. Only valid during FfiCall::invoke().
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Raw mutable pointer to buffer data. Only valid during FfiCall::invoke().
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
}

/// Returns the byte size for a single element of the given dtype.
pub fn dtype_size_bytes(dtype: DType) -> Result<usize, FfiError> {
    match dtype {
        DType::F64 => Ok(8),
        DType::I64 => Ok(8),
        DType::F32 => Ok(4),
        DType::I32 => Ok(4),
        DType::Bool => Ok(1),
    }
}

/// Compute `product(shape) * dtype_size` with overflow checking.
pub fn checked_buffer_size(shape: &[usize], dtype: DType) -> Result<usize, FfiError> {
    let elem_size = dtype_size_bytes(dtype)?;
    let num_elements = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
    match num_elements {
        Some(n) => n.checked_mul(elem_size).ok_or(FfiError::BufferMismatch {
            buffer_index: 0,
            expected_bytes: usize::MAX,
            actual_bytes: 0,
        }),
        None => Err(FfiError::BufferMismatch {
            buffer_index: 0,
            expected_bytes: usize::MAX,
            actual_bytes: 0,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_new_valid_f64_scalar() {
        let buf = FfiBuffer::new(vec![0u8; 8], vec![], DType::F64).unwrap();
        assert_eq!(buf.size(), 8);
        assert_eq!(buf.shape(), &[] as &[usize]);
        assert_eq!(buf.dtype(), DType::F64);
    }

    #[test]
    fn buffer_new_valid_f64_vector() {
        let buf = FfiBuffer::new(vec![0u8; 24], vec![3], DType::F64).unwrap();
        assert_eq!(buf.size(), 24);
        assert_eq!(buf.shape(), &[3]);
    }

    #[test]
    fn buffer_new_valid_f64_matrix() {
        let buf = FfiBuffer::new(vec![0u8; 48], vec![2, 3], DType::F64).unwrap();
        assert_eq!(buf.size(), 48);
    }

    #[test]
    fn buffer_new_size_mismatch() {
        let err = FfiBuffer::new(vec![0u8; 7], vec![], DType::F64).unwrap_err();
        assert!(matches!(err, FfiError::BufferMismatch { .. }));
    }

    #[test]
    fn buffer_zeroed_creates_correct_size() {
        let buf = FfiBuffer::zeroed(vec![4, 3], DType::I64).unwrap();
        assert_eq!(buf.size(), 4 * 3 * 8);
        assert!(buf.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn buffer_empty_shape_is_scalar() {
        // Empty shape = scalar = 1 element
        let buf = FfiBuffer::new(vec![0u8; 8], vec![], DType::F64).unwrap();
        assert_eq!(buf.size(), 8);
    }

    #[test]
    fn buffer_zero_dim_is_zero_size() {
        // Shape with a zero dimension = 0 elements = 0 bytes
        let buf = FfiBuffer::new(vec![], vec![0, 5], DType::F64).unwrap();
        assert_eq!(buf.size(), 0);
    }

    #[test]
    fn dtype_size_bytes_all_types() {
        assert_eq!(dtype_size_bytes(DType::F64).unwrap(), 8);
        assert_eq!(dtype_size_bytes(DType::I64).unwrap(), 8);
        assert_eq!(dtype_size_bytes(DType::F32).unwrap(), 4);
        assert_eq!(dtype_size_bytes(DType::I32).unwrap(), 4);
        assert_eq!(dtype_size_bytes(DType::Bool).unwrap(), 1);
    }

    #[test]
    fn checked_buffer_size_overflow() {
        let err = checked_buffer_size(&[usize::MAX, 2], DType::F64).unwrap_err();
        assert!(matches!(err, FfiError::BufferMismatch { .. }));
    }

    #[test]
    fn buffer_into_bytes_consumes() {
        let buf = FfiBuffer::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![], DType::F64).unwrap();
        let bytes = buf.into_bytes();
        assert_eq!(bytes, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
