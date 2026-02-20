//! Differential oracle, metamorphic, and adversarial tests for FJ-P2C-006
//! (Backend bridge and platform routing).

use fj_backend_cpu::CpuBackend;
use fj_core::{ProgramSpec, Value, build_program};
use fj_runtime::backend::{Backend, BackendError, BackendRegistry};
use fj_runtime::buffer::Buffer;
use fj_runtime::device::{DeviceId, DevicePlacement, Platform};

// ── Oracle Tests ───────────────────────────────────────────────────

/// Oracle: CPU backend execution matches direct eval_jaxpr for Add2.
#[test]
fn oracle_cpu_backend_matches_eval_jaxpr_add2() {
    let backend = CpuBackend::new();
    let jaxpr = build_program(ProgramSpec::Add2);
    let args = vec![Value::scalar_i64(10), Value::scalar_i64(20)];

    let backend_result = backend
        .execute(&jaxpr, &args, DeviceId(0))
        .expect("backend execute");
    let direct_result =
        fj_interpreters::eval_jaxpr(&jaxpr, &args).expect("direct eval");

    assert_eq!(backend_result, direct_result, "backend must match direct eval");
}

/// Oracle: CPU backend execution matches direct eval_jaxpr for Square (f64).
#[test]
fn oracle_cpu_backend_matches_eval_jaxpr_square() {
    let backend = CpuBackend::new();
    let jaxpr = build_program(ProgramSpec::Square);
    let args = vec![Value::scalar_f64(7.0)];

    let backend_result = backend
        .execute(&jaxpr, &args, DeviceId(0))
        .expect("backend execute");
    let direct_result =
        fj_interpreters::eval_jaxpr(&jaxpr, &args).expect("direct eval");

    assert_eq!(backend_result, direct_result);
}

/// Oracle: CPU backend matches direct eval for all available ProgramSpecs.
#[test]
fn oracle_cpu_backend_matches_all_programs() {
    let backend = CpuBackend::new();
    let specs: Vec<(ProgramSpec, Vec<Value>)> = vec![
        (ProgramSpec::Add2, vec![Value::scalar_i64(5), Value::scalar_i64(3)]),
        (ProgramSpec::Square, vec![Value::scalar_f64(4.0)]),
        (ProgramSpec::AddOne, vec![Value::scalar_i64(99)]),
        (ProgramSpec::SinX, vec![Value::scalar_f64(1.5707963)]),
        (ProgramSpec::CosX, vec![Value::scalar_f64(0.0)]),
    ];

    for (spec, args) in specs {
        let jaxpr = build_program(spec);
        let backend_result = backend
            .execute(&jaxpr, &args, DeviceId(0))
            .unwrap_or_else(|e| panic!("{spec:?} backend failed: {e}"));
        let direct_result = fj_interpreters::eval_jaxpr(&jaxpr, &args)
            .unwrap_or_else(|e| panic!("{spec:?} direct failed: {e}"));
        assert_eq!(
            backend_result, direct_result,
            "mismatch for {:?}",
            spec
        );
    }
}

/// Oracle: backend discovery reports CPU platform correctly.
#[test]
fn oracle_backend_discovery_platform() {
    let backend = CpuBackend::new();
    let devices = backend.devices();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].platform, Platform::Cpu);
    assert_eq!(devices[0].id, DeviceId(0));
}

/// Oracle: backend capabilities include F64 and I64 dtypes.
#[test]
fn oracle_backend_dtype_support() {
    let backend = CpuBackend::new();
    let caps = backend.capabilities();
    assert!(
        caps.supported_dtypes.contains(&fj_core::DType::F64),
        "CPU backend must support f64"
    );
    assert!(
        caps.supported_dtypes.contains(&fj_core::DType::I64),
        "CPU backend must support i64"
    );
}

// ── Metamorphic Tests ──────────────────────────────────────────────

/// Metamorphic: same program on CpuBackend(1 device) and CpuBackend(4 devices)
/// produces identical results (backend-independent semantics).
#[test]
fn metamorphic_same_program_different_device_counts() {
    let backend_1 = CpuBackend::new();
    let backend_4 = CpuBackend::with_device_count(4);
    let jaxpr = build_program(ProgramSpec::Square);
    let args = vec![Value::scalar_f64(3.0)];

    let result_1 = backend_1
        .execute(&jaxpr, &args, DeviceId(0))
        .expect("1-device");
    let result_4 = backend_4
        .execute(&jaxpr, &args, DeviceId(0))
        .expect("4-device");

    assert_eq!(result_1, result_4, "device count must not affect computation");
}

/// Metamorphic: transfer to same device is identity.
#[test]
fn metamorphic_transfer_same_device_is_identity() {
    let backend = CpuBackend::new();
    let original_data = vec![0xAB, 0xCD, 0xEF, 0x01];
    let buf = Buffer::new(original_data.clone(), DeviceId(0));

    let transferred = backend.transfer(&buf, DeviceId(0)).expect("transfer");
    assert_eq!(transferred.as_bytes(), &original_data[..]);
    assert_eq!(transferred.device(), buf.device());
}

/// Metamorphic: transfer preserves data across devices.
#[test]
fn metamorphic_transfer_cross_device_preserves_data() {
    let backend = CpuBackend::with_device_count(3);
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let buf = Buffer::new(data.clone(), DeviceId(0));

    // Transfer chain: 0 → 1 → 2
    let t1 = backend.transfer(&buf, DeviceId(1)).expect("0→1");
    let t2 = backend.transfer(&t1, DeviceId(2)).expect("1→2");

    assert_eq!(t2.as_bytes(), &data[..], "data must survive transfer chain");
    assert_eq!(t2.device(), DeviceId(2));
}

/// Metamorphic: capabilities reflect actual multi-device status.
#[test]
fn metamorphic_capabilities_reflect_device_count() {
    let single = CpuBackend::new();
    let multi = CpuBackend::with_device_count(2);

    assert!(!single.capabilities().multi_device);
    assert!(multi.capabilities().multi_device);
}

// ── Adversarial Tests ──────────────────────────────────────────────

/// Adversarial: request unsupported backend from registry.
#[test]
fn adversarial_unsupported_backend_request() {
    let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
    let result = registry.resolve_placement(&DevicePlacement::Default, Some("quantum"));
    match result {
        Err(BackendError::Unavailable { backend }) => {
            assert_eq!(backend, "quantum");
        }
        Err(other) => panic!("expected Unavailable, got: {other}"),
        Ok(_) => panic!("quantum backend should not exist"),
    }
}

/// Adversarial: allocate on non-existent device.
#[test]
fn adversarial_allocate_nonexistent_device() {
    let backend = CpuBackend::new(); // only device 0
    let err = backend
        .allocate(64, DeviceId(99))
        .expect_err("device 99 should not exist");
    match err {
        BackendError::AllocationFailed { device, .. } => {
            assert_eq!(device, DeviceId(99));
        }
        other => panic!("expected AllocationFailed, got: {other}"),
    }
}

/// Adversarial: transfer to non-existent device.
#[test]
fn adversarial_transfer_nonexistent_device() {
    let backend = CpuBackend::new();
    let buf = Buffer::new(vec![42], DeviceId(0));
    let err = backend
        .transfer(&buf, DeviceId(100))
        .expect_err("device 100 should not exist");
    match err {
        BackendError::TransferFailed { target, .. } => {
            assert_eq!(target, DeviceId(100));
        }
        other => panic!("expected TransferFailed, got: {other}"),
    }
}

/// Adversarial: zero-size buffer allocation.
#[test]
fn adversarial_zero_size_buffer() {
    let backend = CpuBackend::new();
    let buf = backend
        .allocate(0, DeviceId(0))
        .expect("zero-size alloc should succeed");
    assert_eq!(buf.size(), 0);
    assert!(buf.as_bytes().is_empty());
}

/// Adversarial: empty registry has no default backend.
#[test]
fn adversarial_empty_registry() {
    let registry = BackendRegistry::new(vec![]);
    assert!(registry.default_backend().is_none());
    assert!(registry.available_backends().is_empty());
    let result = registry.resolve_placement(&DevicePlacement::Default, None);
    assert!(result.is_err());
}

/// Adversarial: fallback from non-existent backend in registry with CPU present.
#[test]
fn adversarial_fallback_from_nonexistent_to_cpu() {
    let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
    let (backend, device, fell_back) = registry
        .resolve_with_fallback(&DevicePlacement::Default, Some("nonexistent"))
        .expect("should fallback to CPU");
    assert_eq!(backend.name(), "cpu");
    assert_eq!(device, DeviceId(0));
    assert!(fell_back);
}
