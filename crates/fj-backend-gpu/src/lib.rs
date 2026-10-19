//! GPU backend foundation for FrankenJAX.
//!
//! This crate provides the architectural foundation for GPU acceleration.
//! V1 reports unavailable because no GPU execution backend is shipped in this
//! crate. CUDA and wgpu integration are deferred contracts, not dormant probe
//! implementations.
//!
//! The Backend trait implementation is wired for:
//! - Device discovery infrastructure
//! - Backend registry integration
//! - A typed unavailable contract for non-CPU execution requests

#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use fj_core::{Jaxpr, Value};
use fj_runtime::backend::{Backend, BackendCapabilities, BackendError};
use fj_runtime::buffer::Buffer;
use fj_runtime::device::{DeviceId, DeviceInfo};

/// Permanent V1 reason that GPU execution cannot be used.
///
/// This is deliberately separate from hardware discovery. V1 does not probe
/// local CUDA or wgpu state, so callers get a deterministic unavailable result
/// instead of environment-dependent behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuUnavailableReason {
    /// FrankenJAX V1 has no shipped GPU execution backend.
    ExecutionBackendAbsent,
}

impl GpuUnavailableReason {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExecutionBackendAbsent => "gpu_execution_backend_absent",
        }
    }

    #[must_use]
    pub fn message(self) -> &'static str {
        match self {
            Self::ExecutionBackendAbsent => {
                "GPU backend unavailable in V1 (hardware execution backend absent)"
            }
        }
    }
}

/// GPU backend implementation.
///
/// V1 always returns a typed unavailable reason for operations and discovery.
#[derive(Debug, Clone)]
pub struct GpuBackend {
    /// Reason this backend is unavailable, or `None` once a real backend ships.
    unavailable_reason: Option<GpuUnavailableReason>,
    /// Detected GPU devices (empty if unavailable).
    devices: Vec<DeviceInfo>,
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend {
    /// Create a new GPU backend.
    ///
    /// V1 returns a deterministic unavailable contract because no GPU execution
    /// path is shipped.
    #[must_use]
    pub fn new() -> Self {
        let (unavailable_reason, devices) = Self::probe_gpu_devices();
        Self {
            unavailable_reason,
            devices,
        }
    }

    /// Return the V1 GPU availability contract.
    ///
    /// The `cuda` and `wgpu` feature flags are reserved names in V1. They do not
    /// enable hardware probing unless real execution dependencies are added.
    fn probe_gpu_devices() -> (Option<GpuUnavailableReason>, Vec<DeviceInfo>) {
        (
            Some(GpuUnavailableReason::ExecutionBackendAbsent),
            Vec::new(),
        )
    }

    /// Check if GPU backend is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.unavailable_reason.is_none()
    }

    /// Get the typed reason GPU is unavailable.
    #[must_use]
    pub fn unavailable_reason(&self) -> Option<GpuUnavailableReason> {
        self.unavailable_reason
    }

    /// Get a human-readable unavailable message.
    #[must_use]
    pub fn unavailable_message(&self) -> Option<&'static str> {
        self.unavailable_reason.map(GpuUnavailableReason::message)
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> &str {
        "gpu"
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        self.devices.clone()
    }

    fn default_device(&self) -> DeviceId {
        self.devices.first().map(|d| d.id).unwrap_or(DeviceId(0))
    }

    fn execute(
        &self,
        _jaxpr: &Jaxpr,
        _args: &[Value],
        _device: DeviceId,
    ) -> Result<Vec<Value>, BackendError> {
        Err(BackendError::Unavailable {
            backend: "gpu".to_owned(),
        })
    }

    fn allocate(&self, _size_bytes: usize, _device: DeviceId) -> Result<Buffer, BackendError> {
        Err(BackendError::Unavailable {
            backend: "gpu".to_owned(),
        })
    }

    fn transfer(&self, _buffer: &Buffer, _target: DeviceId) -> Result<Buffer, BackendError> {
        Err(BackendError::Unavailable {
            backend: "gpu".to_owned(),
        })
    }

    fn version(&self) -> &str {
        "gpu-v1-unavailable"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supported_dtypes: vec![],
            max_tensor_rank: 0,
            memory_limit_bytes: None,
            multi_device: false,
        }
    }
}

/// Check if any GPU is available on this system.
///
/// Convenience function for quick availability checks.
#[must_use]
pub fn gpu_available() -> bool {
    GpuBackend::new().is_available()
}

/// List available GPU devices.
///
/// Returns an empty vec if no GPUs are available.
#[must_use]
pub fn gpu_devices() -> Vec<DeviceInfo> {
    GpuBackend::new().devices()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_backend_reports_unavailable_in_v1() {
        let backend = GpuBackend::new();
        assert!(!backend.is_available());
        assert_eq!(backend.name(), "gpu");
        assert_eq!(
            backend.unavailable_reason(),
            Some(GpuUnavailableReason::ExecutionBackendAbsent)
        );
        assert_eq!(
            backend.unavailable_message(),
            Some("GPU backend unavailable in V1 (hardware execution backend absent)")
        );
        assert_eq!(
            GpuUnavailableReason::ExecutionBackendAbsent.as_str(),
            "gpu_execution_backend_absent"
        );
    }

    #[test]
    fn gpu_probe_contract_is_deterministic_and_empty() {
        let (reason, devices) = GpuBackend::probe_gpu_devices();
        assert_eq!(reason, Some(GpuUnavailableReason::ExecutionBackendAbsent));
        assert!(devices.is_empty());
    }

    #[test]
    fn gpu_execute_returns_unavailable_error() {
        let backend = GpuBackend::new();
        let jaxpr = fj_core::Jaxpr::new(vec![], vec![], vec![], vec![]);
        let result = backend.execute(&jaxpr, &[], DeviceId(0));
        assert!(matches!(result, Err(BackendError::Unavailable { .. })));
    }

    #[test]
    fn gpu_capabilities_empty_when_unavailable() {
        let backend = GpuBackend::new();
        let caps = backend.capabilities();
        assert!(caps.supported_dtypes.is_empty());
        assert_eq!(caps.max_tensor_rank, 0);
    }

    #[test]
    fn gpu_available_convenience_function() {
        assert!(!gpu_available());
        assert!(gpu_devices().is_empty());
    }
}
