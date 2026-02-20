//! CPU execution engine wrapping eval_jaxpr.
//!
//! The CpuBackend provides the Backend trait implementation for host-CPU
//! execution. All computation runs via fj-interpreters::eval_jaxpr.
//!
//! Contract: p2c006.strict.inv001 (CPU always available).

use fj_core::{DType, Jaxpr, Value};
use fj_runtime::backend::{Backend, BackendCapabilities, BackendError};
use fj_runtime::buffer::Buffer;
use fj_runtime::device::{DeviceId, DeviceInfo, Platform};

/// CPU backend: interprets Jaxpr programs on the host CPU.
///
/// V1 scope: single CPU device (DeviceId(0)). All computation is
/// synchronous and single-threaded.
pub struct CpuBackend {
    /// Number of logical CPU devices to expose.
    /// V1: always 1.
    device_count: u32,
    /// Version string for cache key inclusion.
    version_string: String,
}

impl CpuBackend {
    /// Create a CPU backend with a single device.
    #[must_use]
    pub fn new() -> Self {
        Self {
            device_count: 1,
            version_string: format!("fj-backend-cpu/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    /// Create a CPU backend exposing multiple logical devices.
    /// Useful for testing multi-device dispatch without GPU hardware.
    #[must_use]
    pub fn with_device_count(count: u32) -> Self {
        assert!(count > 0, "device count must be at least 1");
        Self {
            device_count: count,
            version_string: format!("fj-backend-cpu/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        (0..self.device_count)
            .map(|i| DeviceInfo {
                id: DeviceId(i),
                platform: Platform::Cpu,
                host_id: 0,
                process_index: 0,
            })
            .collect()
    }

    fn default_device(&self) -> DeviceId {
        DeviceId(0)
    }

    fn execute(
        &self,
        jaxpr: &Jaxpr,
        args: &[Value],
        _device: DeviceId,
    ) -> Result<Vec<Value>, BackendError> {
        // CPU backend ignores device ID — all execution is on the host.
        fj_interpreters::eval_jaxpr(jaxpr, args)
            .map_err(|e| BackendError::ExecutionFailed {
                detail: e.to_string(),
            })
    }

    fn allocate(&self, size_bytes: usize, device: DeviceId) -> Result<Buffer, BackendError> {
        if device.0 >= self.device_count {
            return Err(BackendError::AllocationFailed {
                device,
                detail: format!("device {} not available (have {})", device.0, self.device_count),
            });
        }
        Ok(Buffer::zeroed(size_bytes, device))
    }

    fn transfer(
        &self,
        buffer: &Buffer,
        target: DeviceId,
    ) -> Result<Buffer, BackendError> {
        if target.0 >= self.device_count {
            return Err(BackendError::TransferFailed {
                source: buffer.device(),
                target,
                detail: format!("target device {} not available", target.0),
            });
        }
        // CPU "transfer" is a clone (same memory space).
        Ok(Buffer::new(buffer.as_bytes().to_vec(), target))
    }

    fn version(&self) -> &str {
        &self.version_string
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supported_dtypes: vec![DType::F64, DType::I64],
            max_tensor_rank: 8,
            memory_limit_bytes: None, // host memory, effectively unlimited
            multi_device: self.device_count > 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{ProgramSpec, build_program};

    #[test]
    fn cpu_backend_name() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn cpu_backend_default_device() {
        let backend = CpuBackend::new();
        assert_eq!(backend.default_device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_single_device_discovery() {
        let backend = CpuBackend::new();
        let devices = backend.devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, DeviceId(0));
        assert_eq!(devices[0].platform, Platform::Cpu);
        assert_eq!(devices[0].host_id, 0);
        assert_eq!(devices[0].process_index, 0);
    }

    #[test]
    fn cpu_backend_multi_device_discovery() {
        let backend = CpuBackend::with_device_count(4);
        let devices = backend.devices();
        assert_eq!(devices.len(), 4);
        for (i, dev) in devices.iter().enumerate() {
            assert_eq!(dev.id, DeviceId(i as u32));
            assert_eq!(dev.platform, Platform::Cpu);
        }
    }

    #[test]
    fn cpu_backend_execute_add2() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::Add2);
        let result = backend
            .execute(
                &jaxpr,
                &[Value::scalar_i64(3), Value::scalar_i64(4)],
                DeviceId(0),
            )
            .expect("execution should succeed");
        assert_eq!(result, vec![Value::scalar_i64(7)]);
    }

    #[test]
    fn cpu_backend_execute_square() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::Square);
        let result = backend
            .execute(&jaxpr, &[Value::scalar_f64(5.0)], DeviceId(0))
            .expect("execution should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        assert!((val - 25.0).abs() < 1e-10);
    }

    #[test]
    fn cpu_backend_allocate_and_access() {
        let backend = CpuBackend::new();
        let buf = backend.allocate(256, DeviceId(0)).expect("alloc should succeed");
        assert_eq!(buf.size(), 256);
        assert_eq!(buf.device(), DeviceId(0));
        assert!(buf.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn cpu_backend_allocate_invalid_device() {
        let backend = CpuBackend::new();
        let err = backend.allocate(256, DeviceId(1)).expect_err("should fail");
        assert!(matches!(err, BackendError::AllocationFailed { .. }));
    }

    #[test]
    fn cpu_backend_transfer_same_device() {
        let backend = CpuBackend::new();
        let buf = Buffer::new(vec![1, 2, 3], DeviceId(0));
        let transferred = backend.transfer(&buf, DeviceId(0)).expect("transfer should succeed");
        assert_eq!(transferred.as_bytes(), &[1, 2, 3]);
        assert_eq!(transferred.device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_transfer_cross_device() {
        let backend = CpuBackend::with_device_count(2);
        let buf = Buffer::new(vec![10, 20, 30], DeviceId(0));
        let transferred = backend.transfer(&buf, DeviceId(1)).expect("cross-device transfer");
        assert_eq!(transferred.as_bytes(), &[10, 20, 30]);
        assert_eq!(transferred.device(), DeviceId(1));
        // Original buffer unchanged
        assert_eq!(buf.device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_transfer_invalid_target() {
        let backend = CpuBackend::new();
        let buf = Buffer::new(vec![1], DeviceId(0));
        let err = backend.transfer(&buf, DeviceId(5)).expect_err("should fail");
        assert!(matches!(err, BackendError::TransferFailed { .. }));
    }

    #[test]
    fn cpu_backend_version_string() {
        let backend = CpuBackend::new();
        assert!(backend.version().starts_with("fj-backend-cpu/"));
    }

    #[test]
    fn cpu_backend_buffer_roundtrip_preserves_data() {
        // Contract p2c006.strict.inv003: device_put/device_get round-trip
        let backend = CpuBackend::new();
        let original = vec![0xCA, 0xFE, 0xBA, 0xBE];
        let buf = Buffer::new(original.clone(), DeviceId(0));
        let data = buf.into_bytes();
        assert_eq!(original, data);

        // Through allocate + write
        let mut buf = backend.allocate(4, DeviceId(0)).expect("alloc");
        buf.as_bytes_mut().copy_from_slice(&original);
        assert_eq!(buf.as_bytes(), &original[..]);
    }

    #[test]
    fn cpu_backend_capabilities_supported_dtypes() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();
        assert!(caps.supported_dtypes.contains(&DType::F64));
        assert!(caps.supported_dtypes.contains(&DType::I64));
    }

    #[test]
    fn cpu_backend_capabilities_rank_limit() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();
        assert!(caps.max_tensor_rank >= 4);
    }

    #[test]
    fn cpu_backend_capabilities_memory_unlimited() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();
        assert!(caps.memory_limit_bytes.is_none());
    }

    #[test]
    fn cpu_backend_single_device_not_multi() {
        let backend = CpuBackend::new();
        assert!(!backend.capabilities().multi_device);
    }

    #[test]
    fn cpu_backend_multi_device_caps() {
        let backend = CpuBackend::with_device_count(2);
        assert!(backend.capabilities().multi_device);
    }

    // ── Registry tests ────────────────────────────────────────────

    use fj_runtime::backend::BackendRegistry;
    use fj_runtime::device::DevicePlacement;

    #[test]
    fn registry_get_by_name() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        assert!(registry.get("cpu").is_some());
        assert!(registry.get("gpu").is_none());
    }

    #[test]
    fn registry_default_backend() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let default = registry.default_backend().expect("should have default");
        assert_eq!(default.name(), "cpu");
    }

    #[test]
    fn registry_available_backends() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        assert_eq!(registry.available_backends(), vec!["cpu"]);
    }

    #[test]
    fn registry_resolve_default_placement() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, device) = registry
            .resolve_placement(&DevicePlacement::Default, None)
            .expect("should resolve");
        assert_eq!(backend.name(), "cpu");
        assert_eq!(device, DeviceId(0));
    }

    #[test]
    fn registry_resolve_explicit_backend() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, device) = registry
            .resolve_placement(&DevicePlacement::Default, Some("cpu"))
            .expect("should resolve");
        assert_eq!(backend.name(), "cpu");
        assert_eq!(device, DeviceId(0));
    }

    #[test]
    fn registry_resolve_unavailable_backend() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let result = registry.resolve_placement(&DevicePlacement::Default, Some("gpu"));
        match result {
            Err(BackendError::Unavailable { backend }) => assert_eq!(backend, "gpu"),
            Err(other) => panic!("expected Unavailable, got: {other}"),
            Ok(_) => panic!("expected error for unavailable gpu backend"),
        }
    }

    #[test]
    fn registry_resolve_with_fallback() {
        // Contract p2c006.hardened.inv008: missing backend → CPU fallback
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, device, fell_back) = registry
            .resolve_with_fallback(&DevicePlacement::Default, Some("gpu"))
            .expect("should fallback to CPU");
        assert_eq!(backend.name(), "cpu");
        assert_eq!(device, DeviceId(0));
        assert!(fell_back, "should report fallback occurred");
    }

    #[test]
    fn registry_resolve_no_fallback_needed() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, _, fell_back) = registry
            .resolve_with_fallback(&DevicePlacement::Default, Some("cpu"))
            .expect("should resolve directly");
        assert_eq!(backend.name(), "cpu");
        assert!(!fell_back, "no fallback should be needed");
    }

    #[test]
    fn registry_resolve_explicit_device_id() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::with_device_count(4))]);
        let (backend, device) = registry
            .resolve_placement(&DevicePlacement::Explicit(DeviceId(2)), Some("cpu"))
            .expect("should resolve");
        assert_eq!(backend.name(), "cpu");
        assert_eq!(device, DeviceId(2));
    }

    // ── Category 1: All primitives execute correctly on CPU ───────

    #[test]
    fn cpu_executes_add_one() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::AddOne);
        let result = backend
            .execute(&jaxpr, &[Value::scalar_i64(41)], DeviceId(0))
            .expect("should succeed");
        assert_eq!(result, vec![Value::scalar_i64(42)]);
    }

    #[test]
    fn cpu_executes_sin() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::SinX);
        let result = backend
            .execute(&jaxpr, &[Value::scalar_f64(0.0)], DeviceId(0))
            .expect("should succeed");
        let val = result[0].as_f64_scalar().expect("f64");
        assert!(val.abs() < 1e-10, "sin(0) should be 0");
    }

    #[test]
    fn cpu_executes_cos() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::CosX);
        let result = backend
            .execute(&jaxpr, &[Value::scalar_f64(0.0)], DeviceId(0))
            .expect("should succeed");
        let val = result[0].as_f64_scalar().expect("f64");
        assert!((val - 1.0).abs() < 1e-10, "cos(0) should be 1");
    }

    // ── Category 3: Backend selection correctness ─────────────────

    #[test]
    fn backend_selection_routes_to_named_backend() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, _) = registry
            .resolve_placement(&DevicePlacement::Default, Some("cpu"))
            .expect("cpu should resolve");
        assert_eq!(backend.name(), "cpu");
    }

    // ── Category 5: Memory layout ─────────────────────────────────

    #[test]
    fn buffer_data_is_contiguous() {
        let backend = CpuBackend::new();
        let mut buf = backend.allocate(16, DeviceId(0)).expect("alloc");
        // Write pattern and verify contiguity
        for (i, byte) in buf.as_bytes_mut().iter_mut().enumerate() {
            *byte = i as u8;
        }
        let data = buf.as_bytes();
        for i in 0..16 {
            assert_eq!(data[i], i as u8);
        }
    }

    // ── Category 6: Multi-backend isolation ───────────────────────

    #[test]
    fn two_cpu_backend_instances_are_independent() {
        let backend_a = CpuBackend::new();
        let backend_b = CpuBackend::with_device_count(2);

        // They should have different device counts
        assert_eq!(backend_a.devices().len(), 1);
        assert_eq!(backend_b.devices().len(), 2);

        // Execution on one doesn't affect the other
        let jaxpr = build_program(ProgramSpec::Add2);
        let result_a = backend_a
            .execute(&jaxpr, &[Value::scalar_i64(1), Value::scalar_i64(2)], DeviceId(0))
            .expect("backend_a");
        let result_b = backend_b
            .execute(&jaxpr, &[Value::scalar_i64(3), Value::scalar_i64(4)], DeviceId(0))
            .expect("backend_b");
        assert_eq!(result_a, vec![Value::scalar_i64(3)]);
        assert_eq!(result_b, vec![Value::scalar_i64(7)]);
    }

    // ── Structured logging contract ───────────────────────────────

    #[test]
    fn test_backend_cpu_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("backend-cpu", "execute")).expect("digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_backend_cpu_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ── Property tests ────────────────────────────────────────────

    proptest::proptest! {
        #[test]
        fn prop_cpu_backend_allocation_size_matches(size in 0_usize..4096) {
            let backend = CpuBackend::new();
            let buf = backend.allocate(size, DeviceId(0)).expect("alloc");
            proptest::prop_assert_eq!(buf.size(), size);
        }

        #[test]
        fn prop_cpu_backend_transfer_preserves_data(
            data in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..512)
        ) {
            let backend = CpuBackend::with_device_count(2);
            let buf = Buffer::new(data.clone(), DeviceId(0));
            let transferred = backend.transfer(&buf, DeviceId(1)).expect("transfer");
            proptest::prop_assert_eq!(transferred.as_bytes(), &data[..]);
            proptest::prop_assert_eq!(transferred.device(), DeviceId(1));
        }

        #[test]
        fn prop_cpu_backend_device_count_matches(count in 1_u32..16) {
            let backend = CpuBackend::with_device_count(count);
            proptest::prop_assert_eq!(backend.devices().len(), count as usize);
        }
    }
}
