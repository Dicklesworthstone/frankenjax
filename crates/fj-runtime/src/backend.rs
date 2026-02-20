//! Backend trait defining the uniform interface for compute backends.
//!
//! All backends (CPU, future GPU/TPU) implement this trait with identical
//! API surface. Platform-specific behavior is encapsulated behind the trait
//! boundary. See contract p2c006.strict.inv002.
//!
//! Legacy anchor: P2C006-A12 (Client), P2C006-A07 (backend_specific_translations).

use crate::buffer::Buffer;
use crate::device::{DeviceId, DeviceInfo, DevicePlacement};
use fj_core::{DType, Jaxpr, Value};

// ── Backend Errors ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// Requested backend is not available on this system.
    Unavailable { backend: String },
    /// Device allocation failed (e.g., OOM).
    AllocationFailed { device: DeviceId, detail: String },
    /// Cross-device transfer failed.
    TransferFailed {
        source: DeviceId,
        target: DeviceId,
        detail: String,
    },
    /// Execution error from the backend.
    ExecutionFailed { detail: String },
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable { backend } => write!(f, "backend unavailable: {backend}"),
            Self::AllocationFailed { device, detail } => {
                write!(f, "allocation failed on {device}: {detail}")
            }
            Self::TransferFailed {
                source,
                target,
                detail,
            } => {
                write!(f, "transfer {source} → {target} failed: {detail}")
            }
            Self::ExecutionFailed { detail } => write!(f, "execution failed: {detail}"),
        }
    }
}

impl std::error::Error for BackendError {}

// ── Backend Trait ──────────────────────────────────────────────────

/// Uniform interface for FrankenJAX compute backends.
///
/// Each backend provides device discovery, execution, and memory management.
/// V1 scope: CPU backend only. The trait surface is designed for future
/// GPU/TPU backends without breaking existing consumers.
pub trait Backend: Send + Sync {
    /// Human-readable backend name (e.g., "cpu", "gpu", "tpu").
    fn name(&self) -> &str;

    /// Discover available devices for this backend.
    /// CPU backend: returns one device per logical core (configurable).
    /// Legacy anchor: P2C006-A05 (local_devices).
    fn devices(&self) -> Vec<DeviceInfo>;

    /// Default device for this backend (first available).
    /// Legacy anchor: P2C006-A04 (default_backend).
    fn default_device(&self) -> DeviceId;

    /// Execute a Jaxpr program on the specified device with the given arguments.
    /// Returns output Values (host-resident for V1).
    /// Legacy anchor: P2C006-A11 (Executable.execute), P2C006-A19 (xla_primitive_callable).
    fn execute(
        &self,
        jaxpr: &Jaxpr,
        args: &[Value],
        device: DeviceId,
    ) -> Result<Vec<Value>, BackendError>;

    /// Allocate a buffer on the specified device.
    /// V1 (CPU): wraps a Vec<u8> on the host.
    /// Legacy anchor: P2C006-A10 (Buffer), P2C006-A25 (memory_stats).
    fn allocate(&self, size_bytes: usize, device: DeviceId) -> Result<Buffer, BackendError>;

    /// Transfer a buffer from one device to another.
    /// V1 (CPU): clone semantics (no cross-device transfer).
    /// Legacy anchor: P2C006-A23 (transfer_to_device).
    fn transfer(&self, buffer: &Buffer, target: DeviceId) -> Result<Buffer, BackendError>;

    /// Platform version string for cache key inclusion.
    /// Legacy anchor: P2C006-A20 (backend_xla_version).
    fn version(&self) -> &str;

    /// Query backend capabilities (supported dtypes, tensor rank limits, memory).
    fn capabilities(&self) -> BackendCapabilities;
}

// ── Backend Capabilities ───────────────────────────────────────────

/// Describes what a backend supports: dtypes, rank limits, memory budget.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendCapabilities {
    /// Set of supported element dtypes.
    pub supported_dtypes: Vec<DType>,
    /// Maximum tensor rank supported (0 = scalars only).
    pub max_tensor_rank: usize,
    /// Approximate memory budget in bytes (None = unlimited / unknown).
    pub memory_limit_bytes: Option<usize>,
    /// Whether the backend supports multi-device execution.
    pub multi_device: bool,
}

// ── Backend Registry ───────────────────────────────────────────────

/// Registry of available backends with priority-ordered selection.
///
/// Legacy anchor: P2C006-A01 (backends dict), P2C006-A06 (register_backend).
/// Contract: p2c006.strict.inv001 (CPU always available).
pub struct BackendRegistry {
    backends: Vec<Box<dyn Backend>>,
}

impl BackendRegistry {
    /// Create a registry with the given backends (first = highest priority).
    pub fn new(backends: Vec<Box<dyn Backend>>) -> Self {
        Self { backends }
    }

    /// Look up a backend by name. Returns None if not found.
    pub fn get(&self, name: &str) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .find(|b| b.name() == name)
            .map(|b| b.as_ref())
    }

    /// Return the highest-priority (first) backend.
    /// Legacy anchor: P2C006-A04 (default_backend).
    pub fn default_backend(&self) -> Option<&dyn Backend> {
        self.backends.first().map(|b| b.as_ref())
    }

    /// List all registered backend names.
    pub fn available_backends(&self) -> Vec<&str> {
        self.backends.iter().map(|b| b.name()).collect()
    }

    /// Resolve a DevicePlacement to a concrete (backend, DeviceId) pair.
    ///
    /// - `Default` → first backend's default device.
    /// - `Explicit(id)` → search all backends for a device with that id.
    ///
    /// Legacy anchor: P2C006-A03 (get_backend).
    pub fn resolve_placement(
        &self,
        placement: &DevicePlacement,
        requested_backend: Option<&str>,
    ) -> Result<(&dyn Backend, DeviceId), BackendError> {
        match (placement, requested_backend) {
            (DevicePlacement::Default, Some(name)) => {
                let backend = self.get(name).ok_or_else(|| BackendError::Unavailable {
                    backend: name.to_owned(),
                })?;
                Ok((backend, backend.default_device()))
            }
            (DevicePlacement::Default, None) => {
                let backend = self
                    .default_backend()
                    .ok_or_else(|| BackendError::Unavailable {
                        backend: "(none)".to_owned(),
                    })?;
                Ok((backend, backend.default_device()))
            }
            (DevicePlacement::Explicit(device_id), Some(name)) => {
                let backend = self.get(name).ok_or_else(|| BackendError::Unavailable {
                    backend: name.to_owned(),
                })?;
                Ok((backend, *device_id))
            }
            (DevicePlacement::Explicit(device_id), None) => {
                let backend = self
                    .default_backend()
                    .ok_or_else(|| BackendError::Unavailable {
                        backend: "(none)".to_owned(),
                    })?;
                Ok((backend, *device_id))
            }
        }
    }

    /// Resolve with fallback: if requested backend unavailable, fall back to
    /// the default backend. Returns the resolved backend, device, and whether
    /// a fallback occurred.
    ///
    /// Contract: p2c006.hardened.inv008 (missing backend → CPU fallback).
    pub fn resolve_with_fallback(
        &self,
        placement: &DevicePlacement,
        requested_backend: Option<&str>,
    ) -> Result<(&dyn Backend, DeviceId, bool), BackendError> {
        match self.resolve_placement(placement, requested_backend) {
            Ok((backend, device)) => Ok((backend, device, false)),
            Err(BackendError::Unavailable { .. }) => {
                let fallback = self
                    .default_backend()
                    .ok_or_else(|| BackendError::Unavailable {
                        backend: "(no fallback)".to_owned(),
                    })?;
                let device = match placement {
                    DevicePlacement::Default => fallback.default_device(),
                    DevicePlacement::Explicit(id) => *id,
                };
                Ok((fallback, device, true))
            }
            Err(other) => Err(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verify BackendError Display formatting
    #[test]
    fn backend_error_display() {
        let err = BackendError::Unavailable {
            backend: "tpu".to_owned(),
        };
        assert_eq!(err.to_string(), "backend unavailable: tpu");

        let err = BackendError::AllocationFailed {
            device: DeviceId(0),
            detail: "out of memory".to_owned(),
        };
        assert!(err.to_string().contains("allocation failed"));
        assert!(err.to_string().contains("out of memory"));
    }
}
