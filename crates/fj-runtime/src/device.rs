//! Device identification and placement types.
//!
//! V1 scope: single CPU device. The type system supports future multi-device
//! configurations without breaking existing code.
//!
//! Legacy anchor: P2C006-A05 (local_devices), P2C006-A15 (process_index).

/// Unique identifier for a device within a backend.
///
/// In V1, this is always DeviceId(0) for the single CPU device.
/// Future multi-GPU: DeviceId(0), DeviceId(1), etc. per visible GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId(pub u32);

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "device:{}", self.0)
    }
}

/// Platform type for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    Cpu,
    Gpu,
    Tpu,
}

impl Platform {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Tpu => "tpu",
        }
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metadata about a specific device.
///
/// Returned by Backend::devices() for discovery.
/// Legacy anchor: P2C006-A05 (local_devices: id, platform, host_id, process_index).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Unique device identifier.
    pub id: DeviceId,
    /// Platform type (CPU/GPU/TPU).
    pub platform: Platform,
    /// Host index (0 for single-host setups).
    pub host_id: u32,
    /// Process index within a multi-process setup (0 for single-process).
    pub process_index: u32,
}

/// Device placement specification for a computation.
///
/// V1: always `Default` (single CPU device).
/// Future: explicit device targeting or automatic placement.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DevicePlacement {
    /// Use the backend's default device.
    #[default]
    Default,
    /// Place on a specific device.
    Explicit(DeviceId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_id_display() {
        assert_eq!(DeviceId(0).to_string(), "device:0");
        assert_eq!(DeviceId(3).to_string(), "device:3");
    }

    #[test]
    fn platform_as_str() {
        assert_eq!(Platform::Cpu.as_str(), "cpu");
        assert_eq!(Platform::Gpu.as_str(), "gpu");
        assert_eq!(Platform::Tpu.as_str(), "tpu");
    }

    #[test]
    fn device_info_construction() {
        let info = DeviceInfo {
            id: DeviceId(0),
            platform: Platform::Cpu,
            host_id: 0,
            process_index: 0,
        };
        assert_eq!(info.id, DeviceId(0));
        assert_eq!(info.platform, Platform::Cpu);
    }

    #[test]
    fn device_placement_default_is_default() {
        let placement = DevicePlacement::default();
        assert_eq!(placement, DevicePlacement::Default);
    }
}
