//! E2E scenario tests for FJ-P2C-006 (Backend bridge and platform routing).

use fj_backend_cpu::CpuBackend;
use fj_core::{ProgramSpec, Value, build_program};
use fj_runtime::backend::{Backend, BackendRegistry};
use fj_runtime::device::{DeviceId, DevicePlacement, Platform};

/// E2E Scenario 1: CPU backend executes all implemented primitives end-to-end.
#[test]
fn e2e_cpu_backend_all_primitives() {
    let backend = CpuBackend::new();
    let device = backend.default_device();

    // Add2: 3 + 4 = 7
    let jaxpr = build_program(ProgramSpec::Add2);
    let result = backend
        .execute(
            &jaxpr,
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
            device,
        )
        .expect("Add2");
    assert_eq!(result, vec![Value::scalar_i64(7)]);

    // Square: 5^2 = 25
    let jaxpr = build_program(ProgramSpec::Square);
    let result = backend
        .execute(&jaxpr, &[Value::scalar_f64(5.0)], device)
        .expect("Square");
    let val = result[0].as_f64_scalar().expect("f64");
    assert!((val - 25.0).abs() < 1e-10);

    // AddOne: 41 + 1 = 42
    let jaxpr = build_program(ProgramSpec::AddOne);
    let result = backend
        .execute(&jaxpr, &[Value::scalar_i64(41)], device)
        .expect("AddOne");
    assert_eq!(result, vec![Value::scalar_i64(42)]);

    // SinX: sin(pi/2) ≈ 1.0
    let jaxpr = build_program(ProgramSpec::SinX);
    let result = backend
        .execute(
            &jaxpr,
            &[Value::scalar_f64(std::f64::consts::FRAC_PI_2)],
            device,
        )
        .expect("SinX");
    let val = result[0].as_f64_scalar().expect("f64");
    assert!((val - 1.0).abs() < 1e-10);

    // CosX: cos(0) = 1.0
    let jaxpr = build_program(ProgramSpec::CosX);
    let result = backend
        .execute(&jaxpr, &[Value::scalar_f64(0.0)], device)
        .expect("CosX");
    let val = result[0].as_f64_scalar().expect("f64");
    assert!((val - 1.0).abs() < 1e-10);
}

/// E2E Scenario 2: Backend discovery returns correct capability information.
#[test]
fn e2e_backend_discovery_capabilities() {
    let backend = CpuBackend::new();

    // Device discovery
    let devices = backend.devices();
    assert!(!devices.is_empty(), "must have at least 1 device");
    assert_eq!(devices[0].platform, Platform::Cpu);

    // Capability query
    let caps = backend.capabilities();
    assert!(caps.supported_dtypes.contains(&fj_core::DType::F64));
    assert!(caps.supported_dtypes.contains(&fj_core::DType::I64));
    assert!(
        caps.max_tensor_rank >= 1,
        "must support at least rank-1 tensors"
    );

    // Version string
    assert!(!backend.version().is_empty());
    assert!(backend.version().starts_with("fj-backend-cpu/"));
}

/// E2E Scenario 3: Device routing via registry resolves correctly.
#[test]
fn e2e_device_routing_via_registry() {
    let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);

    // Default placement → CPU
    let (backend, device) = registry
        .resolve_placement(&DevicePlacement::Default, None)
        .expect("default should resolve");
    assert_eq!(backend.name(), "cpu");
    assert_eq!(device, DeviceId(0));

    // Explicit CPU → CPU
    let (backend, device) = registry
        .resolve_placement(&DevicePlacement::Default, Some("cpu"))
        .expect("explicit cpu should resolve");
    assert_eq!(backend.name(), "cpu");
    assert_eq!(device, DeviceId(0));

    // Execute through resolved backend
    let jaxpr = build_program(ProgramSpec::Add2);
    let result = backend
        .execute(
            &jaxpr,
            &[Value::scalar_i64(10), Value::scalar_i64(20)],
            device,
        )
        .expect("execute via resolved backend");
    assert_eq!(result, vec![Value::scalar_i64(30)]);
}

/// E2E Scenario 4: Unavailable backend triggers graceful CPU fallback.
#[test]
fn e2e_backend_fallback_to_cpu() {
    let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);

    // Request "gpu" which is unavailable
    let (backend, device, fell_back) = registry
        .resolve_with_fallback(&DevicePlacement::Default, Some("gpu"))
        .expect("should fallback to CPU");

    assert!(fell_back, "fallback should have occurred");
    assert_eq!(backend.name(), "cpu");
    assert_eq!(device, DeviceId(0));

    // Execution still works through fallback
    let jaxpr = build_program(ProgramSpec::Square);
    let result = backend
        .execute(&jaxpr, &[Value::scalar_f64(7.0)], device)
        .expect("execute on fallback");
    let val = result[0].as_f64_scalar().expect("f64");
    assert!((val - 49.0).abs() < 1e-10);
}

/// E2E Scenario 5: Same computation gives same result regardless of routing path.
#[test]
fn e2e_cross_backend_consistency() {
    let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
    let jaxpr = build_program(ProgramSpec::Square);
    let args = vec![Value::scalar_f64(3.0)];

    // Path 1: direct backend
    let (backend_direct, device_direct) = registry
        .resolve_placement(&DevicePlacement::Default, Some("cpu"))
        .expect("direct");
    let result_direct = backend_direct
        .execute(&jaxpr, &args, device_direct)
        .expect("direct execution");

    // Path 2: fallback path (requesting "gpu", falling back to "cpu")
    let (backend_fallback, device_fallback, _) = registry
        .resolve_with_fallback(&DevicePlacement::Default, Some("gpu"))
        .expect("fallback");
    let result_fallback = backend_fallback
        .execute(&jaxpr, &args, device_fallback)
        .expect("fallback execution");

    // Path 3: default (no backend specified)
    let (backend_default, device_default) = registry
        .resolve_placement(&DevicePlacement::Default, None)
        .expect("default");
    let result_default = backend_default
        .execute(&jaxpr, &args, device_default)
        .expect("default execution");

    assert_eq!(
        result_direct, result_fallback,
        "direct vs fallback must match"
    );
    assert_eq!(
        result_direct, result_default,
        "direct vs default must match"
    );
}

/// E2E Scenario 6: Buffer round-trip through backend allocate + transfer.
#[test]
fn e2e_buffer_roundtrip_through_backend() {
    let backend = CpuBackend::with_device_count(2);

    // Allocate on device 0
    let mut buf = backend.allocate(8, DeviceId(0)).expect("alloc");
    let payload = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
    buf.as_bytes_mut().copy_from_slice(&payload);

    // Transfer to device 1
    let transferred = backend.transfer(&buf, DeviceId(1)).expect("transfer");
    assert_eq!(transferred.device(), DeviceId(1));
    assert_eq!(transferred.as_bytes(), &payload);

    // Transfer back to device 0
    let round_tripped = backend
        .transfer(&transferred, DeviceId(0))
        .expect("round-trip");
    assert_eq!(round_tripped.device(), DeviceId(0));
    assert_eq!(round_tripped.as_bytes(), &payload);
}
