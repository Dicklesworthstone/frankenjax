#![forbid(unsafe_code)]

use fj_dispatch::{DispatchError, TransformExecutionError};

#[derive(Debug)]
pub enum ApiError {
    GradRequiresScalar { detail: String },
    VmapDimensionMismatch { expected: usize, actual: usize },
    InvalidComposition { detail: String },
    CacheKeyFailure { detail: String },
    EvalError { detail: String },
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GradRequiresScalar { detail } => {
                write!(f, "grad requires scalar input and output: {detail}")
            }
            Self::VmapDimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "vmap: all mapped arguments must have the same leading dimension (expected {expected}, got {actual})"
                )
            }
            Self::InvalidComposition { detail } => {
                write!(f, "invalid transform composition: {detail}")
            }
            Self::CacheKeyFailure { detail } => {
                write!(f, "cache key generation failed: {detail}")
            }
            Self::EvalError { detail } => {
                write!(f, "evaluation error: {detail}")
            }
        }
    }
}

impl std::error::Error for ApiError {}

impl From<DispatchError> for ApiError {
    fn from(err: DispatchError) -> Self {
        match err {
            DispatchError::Cache(e) => Self::CacheKeyFailure {
                detail: e.to_string(),
            },
            DispatchError::Interpreter(e) => Self::EvalError {
                detail: e.to_string(),
            },
            DispatchError::TransformInvariant(e) => Self::InvalidComposition {
                detail: e.to_string(),
            },
            DispatchError::BackendExecution(e) => Self::EvalError {
                detail: e.to_string(),
            },
            DispatchError::TransformExecution(ref e) => match e {
                TransformExecutionError::VmapMismatchedLeadingDimension { expected, actual } => {
                    Self::VmapDimensionMismatch {
                        expected: *expected,
                        actual: *actual,
                    }
                }
                TransformExecutionError::NonScalarGradientInput
                | TransformExecutionError::NonScalarGradientOutput => Self::GradRequiresScalar {
                    detail: e.to_string(),
                },
                _ => Self::EvalError {
                    detail: e.to_string(),
                },
            },
        }
    }
}
