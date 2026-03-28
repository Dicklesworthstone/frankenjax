//! CPU execution engine with dependency-aware equation scheduling.
//!
//! The CpuBackend provides the Backend trait implementation for host-CPU
//! execution. Pure equations whose inputs are already available are evaluated
//! in dependency waves, with each wave evaluated in parallel.
//!
//! Contract: p2c006.strict.inv001 (CPU always available).

use fj_core::{Atom, DType, Jaxpr, Value, VarId};
use fj_interpreters::InterpreterError;
use fj_lax::{eval_primitive, eval_primitive_multi};
use fj_runtime::backend::{Backend, BackendCapabilities, BackendError};
use fj_runtime::buffer::Buffer;
use fj_runtime::device::{DeviceId, DeviceInfo, Platform};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

fn equation_inputs_ready(equation: &fj_core::Equation, env: &HashMap<VarId, Value>) -> bool {
    equation.inputs.iter().all(|atom| match atom {
        Atom::Var(var) => env.contains_key(var),
        Atom::Lit(_) => true,
    })
}

fn first_missing_input_var(
    equation: &fj_core::Equation,
    env: &HashMap<VarId, Value>,
) -> Option<VarId> {
    equation.inputs.iter().find_map(|atom| match atom {
        Atom::Var(var) if !env.contains_key(var) => Some(*var),
        _ => None,
    })
}

fn resolve_equation_inputs(
    equation: &fj_core::Equation,
    env: &HashMap<VarId, Value>,
) -> Result<Vec<Value>, InterpreterError> {
    let mut resolved = Vec::with_capacity(equation.inputs.len());
    for atom in &equation.inputs {
        match atom {
            Atom::Var(var) => {
                let value = env
                    .get(var)
                    .cloned()
                    .ok_or(InterpreterError::MissingVariable(*var))?;
                resolved.push(value);
            }
            Atom::Lit(lit) => resolved.push(Value::Scalar(*lit)),
        }
    }
    Ok(resolved)
}

fn evaluate_equation_multi(
    equation: &fj_core::Equation,
    env: &HashMap<VarId, Value>,
) -> Result<Vec<Value>, InterpreterError> {
    let resolved = resolve_equation_inputs(equation, env)?;
    let outputs = eval_primitive_multi(equation.primitive, &resolved, &equation.params)?;
    if outputs.len() != equation.outputs.len() {
        return Err(InterpreterError::UnexpectedOutputArity {
            primitive: equation.primitive,
            expected: equation.outputs.len(),
            actual: outputs.len(),
        });
    }
    Ok(outputs)
}

fn evaluate_equation(
    equation: &fj_core::Equation,
    env: &HashMap<VarId, Value>,
) -> Result<Value, InterpreterError> {
    if equation.outputs.len() != 1 {
        return Err(InterpreterError::UnexpectedOutputArity {
            primitive: equation.primitive,
            expected: 1,
            actual: equation.outputs.len(),
        });
    }
    let resolved = resolve_equation_inputs(equation, env)?;
    let output = eval_primitive(equation.primitive, &resolved, &equation.params)?;
    Ok(output)
}

fn evaluate_jaxpr_parallel_inner(
    jaxpr: &Jaxpr,
    args: &[Value],
    max_ready_wave: &mut usize,
) -> Result<Vec<Value>, InterpreterError> {
    if !jaxpr.constvars.is_empty() {
        return Err(InterpreterError::ConstArity {
            expected: jaxpr.constvars.len(),
            actual: 0,
        });
    }
    if args.len() != jaxpr.invars.len() {
        return Err(InterpreterError::InputArity {
            expected: jaxpr.invars.len(),
            actual: args.len(),
        });
    }

    let mut env: HashMap<VarId, Value> = HashMap::with_capacity_and_hasher(
        jaxpr.invars.len() + jaxpr.equations.len(),
        Default::default(),
    );
    for (index, var) in jaxpr.invars.iter().enumerate() {
        env.insert(*var, args[index].clone());
    }

    let mut executed = vec![false; jaxpr.equations.len()];
    let mut remaining = jaxpr.equations.len();

    while remaining > 0 {
        let first_pending = executed
            .iter()
            .position(|done| !*done)
            .expect("remaining > 0 guarantees at least one pending equation");
        let first_eqn = &jaxpr.equations[first_pending];

        let is_multi_output = first_eqn.outputs.len() > 1;
        let barrier =
            !first_eqn.effects.is_empty() || !first_eqn.sub_jaxprs.is_empty() || is_multi_output;
        if barrier {
            let outputs = evaluate_equation_multi(first_eqn, &env)?;
            for (out_var, out_val) in first_eqn.outputs.iter().zip(outputs) {
                env.insert(*out_var, out_val);
            }
            executed[first_pending] = true;
            remaining -= 1;
            continue;
        }

        let first_effect_barrier_index = (first_pending..jaxpr.equations.len())
            .find(|idx| {
                !executed[*idx]
                    && (!jaxpr.equations[*idx].effects.is_empty()
                        || !jaxpr.equations[*idx].sub_jaxprs.is_empty()
                        || jaxpr.equations[*idx].outputs.len() > 1)
            })
            .unwrap_or(jaxpr.equations.len());

        let mut ready_indices = Vec::new();
        for (idx, done) in executed
            .iter()
            .enumerate()
            .take(first_effect_barrier_index)
            .skip(first_pending)
        {
            if *done {
                continue;
            }
            let eqn = &jaxpr.equations[idx];
            if equation_inputs_ready(eqn, &env) {
                ready_indices.push(idx);
            }
        }

        if ready_indices.is_empty() {
            if let Some(missing) = first_missing_input_var(first_eqn, &env) {
                return Err(InterpreterError::MissingVariable(missing));
            }
            return Err(InterpreterError::MissingVariable(first_eqn.outputs[0]));
        }

        *max_ready_wave = (*max_ready_wave).max(ready_indices.len());

        let should_parallelize = ready_indices.len() > 1 && rayon::current_num_threads() > 1;
        if should_parallelize {
            // No env.clone() needed: the parallel phase only reads from env.
            // The shared borrow is released after collect() before we mutate env below.
            let env_ref = &env;
            let mut evaluated = ready_indices
                .par_iter()
                .map(|idx| {
                    let eqn = &jaxpr.equations[*idx];
                    let output = evaluate_equation(eqn, env_ref)?;
                    Ok((*idx, eqn.outputs[0], output))
                })
                .collect::<Result<Vec<_>, InterpreterError>>()?;

            evaluated.sort_by_key(|(idx, _, _)| *idx);
            for (idx, out_var, out_value) in evaluated {
                env.insert(out_var, out_value);
                executed[idx] = true;
                remaining -= 1;
            }
        } else {
            for idx in ready_indices {
                let eqn = &jaxpr.equations[idx];
                let output = evaluate_equation(eqn, &env)?;
                env.insert(eqn.outputs[0], output);
                executed[idx] = true;
                remaining -= 1;
            }
        }
    }

    jaxpr
        .outvars
        .iter()
        .map(|var| {
            env.get(var)
                .cloned()
                .ok_or(InterpreterError::MissingVariable(*var))
        })
        .collect()
}

fn evaluate_jaxpr_parallel(jaxpr: &Jaxpr, args: &[Value]) -> Result<Vec<Value>, InterpreterError> {
    let mut ignored_max_ready_wave = 0_usize;
    evaluate_jaxpr_parallel_inner(jaxpr, args, &mut ignored_max_ready_wave)
}

/// CPU backend: interprets Jaxpr programs on the host CPU.
///
/// V1 scope: single CPU device (DeviceId(0)). Execution is synchronous and
/// uses dependency-wave parallel scheduling for independent equations.
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
        evaluate_jaxpr_parallel(jaxpr, args).map_err(|e| BackendError::ExecutionFailed {
            detail: e.to_string(),
        })
    }

    fn allocate(&self, size_bytes: usize, device: DeviceId) -> Result<Buffer, BackendError> {
        if device.0 >= self.device_count {
            return Err(BackendError::AllocationFailed {
                device,
                detail: format!(
                    "device {} not available (have {})",
                    device.0, self.device_count
                ),
            });
        }
        Ok(Buffer::zeroed(size_bytes, device))
    }

    fn transfer(&self, buffer: &Buffer, target: DeviceId) -> Result<Buffer, BackendError> {
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
    use fj_core::{Atom, Equation, Jaxpr, Primitive, ProgramSpec, VarId, build_program};
    use std::collections::BTreeMap;

    fn make_parallel_independent_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(5)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: vec![Atom::Var(VarId(1))].into(),
                    outputs: vec![VarId(3)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: vec![Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(4)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(VarId(3)), Atom::Var(VarId(4))].into(),
                    outputs: vec![VarId(5)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

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
    fn test_parallel_independent_ops() {
        let jaxpr = make_parallel_independent_jaxpr();
        let mut max_ready_wave = 0_usize;
        let result = evaluate_jaxpr_parallel_inner(
            &jaxpr,
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
            &mut max_ready_wave,
        )
        .expect("parallel execution should succeed");

        assert_eq!(result, vec![Value::scalar_i64(-7)]);
        assert!(
            max_ready_wave >= 2,
            "expected at least one parallel-ready wave with width >=2, got {max_ready_wave}"
        );
    }

    #[test]
    fn test_parallel_correctness() {
        let backend = CpuBackend::new();
        let jaxpr = make_parallel_independent_jaxpr();
        let args = vec![Value::scalar_i64(11), Value::scalar_i64(-6)];

        let backend_outputs = backend
            .execute(&jaxpr, &args, DeviceId(0))
            .expect("backend execution should succeed");
        let interpreter_outputs = fj_interpreters::eval_jaxpr(&jaxpr, &args)
            .expect("interpreter execution should succeed");

        assert_eq!(backend_outputs, interpreter_outputs);
    }

    #[test]
    fn test_parallel_no_data_race() {
        use std::sync::Arc;
        use std::thread;

        let backend = Arc::new(CpuBackend::new());
        let jaxpr = Arc::new(make_parallel_independent_jaxpr());

        let mut workers = Vec::new();
        for worker_id in 0_i64..8 {
            let backend = Arc::clone(&backend);
            let jaxpr = Arc::clone(&jaxpr);
            workers.push(thread::spawn(move || {
                for offset in 0_i64..64 {
                    let a = worker_id * 10 + offset;
                    let b = -offset;
                    let outputs = backend
                        .execute(
                            &jaxpr,
                            &[Value::scalar_i64(a), Value::scalar_i64(b)],
                            DeviceId(0),
                        )
                        .expect("concurrent execution should succeed");
                    assert_eq!(outputs, vec![Value::scalar_i64(-(a + b))]);
                }
            }));
        }

        for worker in workers {
            worker.join().expect("worker thread should complete");
        }
    }

    #[test]
    fn cpu_backend_allocate_and_access() {
        let backend = CpuBackend::new();
        let buf = backend
            .allocate(256, DeviceId(0))
            .expect("alloc should succeed");
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
        let transferred = backend
            .transfer(&buf, DeviceId(0))
            .expect("transfer should succeed");
        assert_eq!(transferred.as_bytes(), &[1, 2, 3]);
        assert_eq!(transferred.device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_transfer_cross_device() {
        let backend = CpuBackend::with_device_count(2);
        let buf = Buffer::new(vec![10, 20, 30], DeviceId(0));
        let transferred = backend
            .transfer(&buf, DeviceId(1))
            .expect("cross-device transfer");
        assert_eq!(transferred.as_bytes(), &[10, 20, 30]);
        assert_eq!(transferred.device(), DeviceId(1));
        // Original buffer unchanged
        assert_eq!(buf.device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_transfer_invalid_target() {
        let backend = CpuBackend::new();
        let buf = Buffer::new(vec![1], DeviceId(0));
        let err = backend
            .transfer(&buf, DeviceId(5))
            .expect_err("should fail");
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
        for (i, &byte) in data.iter().enumerate().take(16) {
            assert_eq!(byte, i as u8);
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
            .execute(
                &jaxpr,
                &[Value::scalar_i64(1), Value::scalar_i64(2)],
                DeviceId(0),
            )
            .expect("backend_a");
        let result_b = backend_b
            .execute(
                &jaxpr,
                &[Value::scalar_i64(3), Value::scalar_i64(4)],
                DeviceId(0),
            )
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
