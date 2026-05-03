//! GPU backend scaffold for FrankenJAX.
//!
//! This crate provides the architectural foundation for GPU acceleration.
//! V1 implementation returns "unavailable" - actual GPU support requires:
//! - CUDA backend via cudarc (NVIDIA GPUs)
//! - Cross-platform via wgpu (Vulkan/Metal/DX12)
//!
//! The Backend trait implementation is complete but stubbed, enabling:
//! - Device discovery infrastructure
//! - Backend registry integration
//! - Future GPU kernel compilation paths

#![forbid(unsafe_code)]

use fj_core::{DType, Jaxpr, Value};
use fj_runtime::backend::{Backend, BackendCapabilities, BackendError};
use fj_runtime::buffer::Buffer;
use fj_runtime::device::{DeviceId, DeviceInfo, Platform};

/// GPU backend implementation (stub).
///
/// Currently returns "unavailable" for all operations.
/// Future implementations will detect GPU hardware and provide:
/// - CUDA execution for NVIDIA GPUs
/// - wgpu execution for cross-platform GPU support
#[derive(Debug, Clone)]
pub struct GpuBackend {
    /// Whether GPU hardware was detected during initialization.
    available: bool,
    /// Detected GPU devices (empty if unavailable).
    devices: Vec<DeviceInfo>,
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend {
    /// Create a new GPU backend, probing for available hardware.
    ///
    /// V1: Always returns unavailable (no GPU implementation yet).
    /// Future: Will probe CUDA/wgpu for GPU devices.
    #[must_use]
    pub fn new() -> Self {
        let (available, devices) = Self::probe_gpu_devices();
        Self { available, devices }
    }

    /// Probe for GPU devices.
    ///
    /// V1: Returns (false, empty) - no GPU support yet.
    /// Future CUDA path: Query cudaGetDeviceCount, cudaGetDeviceProperties.
    /// Future wgpu path: Query wgpu::Instance::enumerate_adapters.
    fn probe_gpu_devices() -> (bool, Vec<DeviceInfo>) {
        // V1: No GPU support - return unavailable
        // Future: Probe CUDA or wgpu for devices
        #[cfg(feature = "cuda")]
        {
            // TODO: cudarc device enumeration
            // let device_count = cudarc::driver::CudaDevice::count()?;
        }

        #[cfg(feature = "wgpu")]
        {
            // TODO: wgpu adapter enumeration
            // let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            // let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        }

        (false, Vec::new())
    }

    /// Check if GPU backend is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Get the reason GPU is unavailable.
    #[must_use]
    pub fn unavailable_reason(&self) -> &'static str {
        if self.available {
            ""
        } else {
            "GPU backend not yet implemented (V1 scaffold only)"
        }
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
        self.devices
            .first()
            .map(|d| d.id)
            .unwrap_or(DeviceId(0))
    }

    fn execute(
        &self,
        _jaxpr: &Jaxpr,
        _args: &[Value],
        _device: DeviceId,
    ) -> Result<Vec<Value>, BackendError> {
        if !self.available {
            return Err(BackendError::Unavailable {
                backend: "gpu".to_owned(),
            });
        }

        // V1: Not implemented
        // Future: Compile jaxpr to GPU kernels, execute on device
        Err(BackendError::ExecutionFailed {
            detail: "GPU execution not yet implemented".to_owned(),
        })
    }

    fn allocate(&self, size_bytes: usize, device: DeviceId) -> Result<Buffer, BackendError> {
        if !self.available {
            return Err(BackendError::Unavailable {
                backend: "gpu".to_owned(),
            });
        }

        // V1: Not implemented
        // Future: Allocate GPU device memory
        Err(BackendError::AllocationFailed {
            device,
            detail: format!("GPU allocation of {size_bytes} bytes not yet implemented"),
        })
    }

    fn transfer(&self, _buffer: &Buffer, target: DeviceId) -> Result<Buffer, BackendError> {
        if !self.available {
            return Err(BackendError::Unavailable {
                backend: "gpu".to_owned(),
            });
        }

        // V1: Not implemented
        // Future: DMA transfer between host and device
        Err(BackendError::TransferFailed {
            source: DeviceId(0),
            target,
            detail: "GPU transfer not yet implemented".to_owned(),
        })
    }

    fn version(&self) -> &str {
        "gpu-v1-scaffold"
    }

    fn capabilities(&self) -> BackendCapabilities {
        if !self.available {
            return BackendCapabilities {
                supported_dtypes: vec![],
                max_tensor_rank: 0,
                memory_limit_bytes: None,
                multi_device: false,
            };
        }

        // Future: Query actual GPU capabilities
        BackendCapabilities {
            supported_dtypes: vec![
                DType::F32,
                DType::F64,
                DType::I32,
                DType::I64,
                DType::Bool,
            ],
            max_tensor_rank: 8,
            memory_limit_bytes: None, // Query from device
            multi_device: true,       // GPU backends typically support multi-device
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
