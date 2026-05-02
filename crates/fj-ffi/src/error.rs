//! FFI error types.
//!
//! All errors that can occur at the FFI boundary are represented as `FfiError`
//! variants. These are always recoverable — the only unrecoverable FFI failure
//! is a segfault in external code, which aborts the process.

use std::fmt;

/// Errors arising from FFI operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfiError {
    /// FFI function returned a non-zero status code.
    CallFailed {
        target: String,
        code: i32,
        message: String,
    },

    /// Requested FFI target is not registered.
    TargetNotFound {
        name: String,
        available: Vec<String>,
    },

    /// Attempted to register a target name that already exists.
    DuplicateTarget { name: String },

    /// Function pointer is null.
    NullPointer { target_name: String },

    /// Buffer size does not match declared dtype * shape.
    BufferMismatch {
        buffer_index: usize,
        expected_bytes: usize,
        actual_bytes: usize,
    },

    /// Bool buffers must use the canonical one-byte encoding: 0 or 1.
    InvalidBoolByte {
        buffer_index: usize,
        byte_index: usize,
        value: u8,
    },

    /// Shape dimension cannot be represented by FrankenJAX core shape metadata.
    ShapeDimensionOutOfRange {
        dimension_index: usize,
        dimension: usize,
        max_supported: usize,
    },

    /// Literal payload cannot be represented as the declared dtype.
    UnrepresentableLiteral {
        dtype: fj_core::DType,
        literal: fj_core::Literal,
    },

    /// Dtype not supported at the FFI boundary.
    UnsupportedDtype { dtype: fj_core::DType },
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FfiError::CallFailed {
                target,
                code,
                message,
            } => write!(
                f,
                "FFI call to '{target}' failed with code {code}: {message}"
            ),
            FfiError::TargetNotFound { name, available } => {
                write!(
                    f,
                    "FFI target '{name}' not found. Registered targets: [{}]",
                    available.join(", ")
                )
            }
            FfiError::DuplicateTarget { name } => {
                write!(f, "FFI target '{name}' is already registered")
            }
            FfiError::NullPointer { target_name } => {
                write!(
                    f,
                    "Cannot register FFI target '{target_name}': function pointer is null"
                )
            }
            FfiError::BufferMismatch {
                buffer_index,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "FFI buffer {buffer_index} size mismatch: expected {expected_bytes} bytes, got {actual_bytes}"
            ),
            FfiError::InvalidBoolByte {
                buffer_index,
                byte_index,
                value,
            } => write!(
                f,
                "FFI bool buffer {buffer_index} contains non-canonical byte {value} at byte {byte_index}; expected 0 or 1"
            ),
            FfiError::ShapeDimensionOutOfRange {
                dimension_index,
                dimension,
                max_supported,
            } => write!(
                f,
                "FFI shape dimension {dimension_index}={dimension} exceeds maximum supported core dimension {max_supported}"
            ),
            FfiError::UnrepresentableLiteral { dtype, literal } => write!(
                f,
                "Literal {literal:?} cannot be represented as declared FFI dtype {dtype:?}"
            ),
            FfiError::UnsupportedDtype { dtype } => {
                write!(f, "Dtype {dtype:?} is not supported at the FFI boundary")
            }
        }
    }
}

impl std::error::Error for FfiError {}
