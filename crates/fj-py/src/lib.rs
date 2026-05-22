#![forbid(unsafe_code)]

use pyo3::prelude::*;

use fj_core::{Jaxpr, ProgramSpec, Value, build_program};

#[pyclass]
#[derive(Clone)]
struct PyJaxpr {
    inner: Jaxpr,
}

#[pyclass]
#[derive(Clone)]
struct PyValue {
    inner: Value,
}

#[pyclass]
#[derive(Clone)]
struct PyCheckpoint {
    inner: fj_api::CheckpointWrapped,
}

#[pyclass]
#[derive(Clone)]
struct PyVjpPullback {
    jaxpr: Jaxpr,
    primals: Vec<Value>,
}

#[pyclass]
#[derive(Clone)]
struct PyLinearizedJvp {
    jaxpr: Jaxpr,
    primals: Vec<Value>,
}

#[pyclass(name = "ShapeDtypeStruct")]
#[derive(Clone)]
struct PyShapeDtypeStruct {
    shape: Vec<u32>,
    dtype: String,
}

#[pyclass(name = "Device")]
#[derive(Clone)]
struct PyDevice {
    id: usize,
    process_index: usize,
}

#[pyclass]
#[derive(Clone)]
struct PyNamedScope {
    name: String,
}

#[pymethods]
impl PyValue {
    #[staticmethod]
    fn scalar_f64(v: f64) -> Self {
        PyValue {
            inner: Value::scalar_f64(v),
        }
    }

    #[staticmethod]
    fn scalar_i64(v: i64) -> Self {
        PyValue {
            inner: Value::scalar_i64(v),
        }
    }

    #[staticmethod]
    fn vector_i64(values: Vec<i64>) -> PyResult<Self> {
        Value::vector_i64(&values)
            .map(|v| PyValue { inner: v })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    #[staticmethod]
    fn vector_f64(values: Vec<f64>) -> PyResult<Self> {
        Value::vector_f64(&values)
            .map(|v| PyValue { inner: v })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn as_f64(&self) -> Option<f64> {
        self.inner.as_f64_scalar()
    }

    fn as_i64(&self) -> Option<i64> {
        self.inner.as_i64_scalar()
    }

    fn shape(&self) -> Vec<u32> {
        self.inner
            .as_tensor()
            .map_or_else(Vec::new, |tensor| tensor.shape.dims.clone())
    }

    fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype())
    }

    fn as_f64_list(&self) -> Option<Vec<f64>> {
        match &self.inner {
            Value::Scalar(_) => self.inner.as_f64_scalar().map(|value| vec![value]),
            Value::Tensor(tensor) => tensor.to_f64_vec(),
        }
    }

    fn as_i64_list(&self) -> Option<Vec<i64>> {
        match &self.inner {
            Value::Scalar(_) => self.inner.as_i64_scalar().map(|value| vec![value]),
            Value::Tensor(tensor) => tensor.to_i64_vec(),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pymethods]
impl PyCheckpoint {
    fn call(&self, args: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
        let rust_args = py_values_to_rust(args);
        self.inner
            .call(rust_args)
            .map(py_values_from_rust)
            .map_err(runtime_error)
    }

    fn grad(&self, args: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
        let rust_args = py_values_to_rust(args);
        self.inner
            .grad()
            .call(rust_args)
            .map(py_values_from_rust)
            .map_err(runtime_error)
    }

    fn value_and_grad(&self, args: Vec<PyValue>) -> PyResult<(Vec<PyValue>, Vec<PyValue>)> {
        let rust_args = py_values_to_rust(args);
        self.inner
            .value_and_grad()
            .call(rust_args)
            .map(|(values, grads)| (py_values_from_rust(values), py_values_from_rust(grads)))
            .map_err(runtime_error)
    }

    fn memory_savings_entries(&self) -> usize {
        self.inner.memory_savings_entries()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCheckpoint(memory_savings_entries={})",
            self.inner.memory_savings_entries()
        )
    }
}

#[pymethods]
impl PyVjpPullback {
    fn call(&self, cotangents: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
        let rust_cotangents = py_values_to_rust(cotangents);
        let [cotangent] = rust_cotangents.as_slice() else {
            return Err(runtime_error(format!(
                "vjp pullback expects exactly one output cotangent, got {}",
                rust_cotangents.len()
            )));
        };

        fj_ad::grad_jaxpr_with_cotangent(&self.jaxpr, &self.primals, cotangent)
            .map(py_values_from_rust)
            .map_err(runtime_error)
    }

    fn __call__(&self, cotangents: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
        self.call(cotangents)
    }

    fn __repr__(&self) -> String {
        format!("PyVjpPullback(num_primals={})", self.primals.len())
    }
}

#[pymethods]
impl PyLinearizedJvp {
    fn call(&self, tangents: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
        let rust_tangents = py_values_to_rust(tangents);
        fj_ad::jvp(&self.jaxpr, &self.primals, &rust_tangents)
            .map(|result| py_values_from_rust(result.tangents))
            .map_err(runtime_error)
    }

    fn __call__(&self, tangents: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
        self.call(tangents)
    }

    fn __repr__(&self) -> String {
        format!("PyLinearizedJvp(num_primals={})", self.primals.len())
    }
}

#[pymethods]
impl PyShapeDtypeStruct {
    fn shape(&self) -> Vec<u32> {
        self.shape.clone()
    }

    fn dtype(&self) -> String {
        self.dtype.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapeDtypeStruct(shape={:?}, dtype={})",
            self.shape, self.dtype
        )
    }
}

#[pymethods]
impl PyDevice {
    #[getter]
    fn id(&self) -> usize {
        self.id
    }

    #[getter]
    fn process_index(&self) -> usize {
        self.process_index
    }

    #[getter]
    fn platform(&self) -> &'static str {
        "cpu"
    }

    #[getter]
    fn device_kind(&self) -> &'static str {
        "cpu"
    }

    fn __repr__(&self) -> String {
        format!(
            "Device(id={}, process_index={}, platform='cpu')",
            self.id, self.process_index
        )
    }
}

#[pymethods]
impl PyNamedScope {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __enter__(&self) {}

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        false
    }

    fn __call__(&self, function: Py<PyAny>) -> Py<PyAny> {
        function
    }

    fn __repr__(&self) -> String {
        format!("PyNamedScope(name={:?})", self.name)
    }
}

fn py_values_to_rust(args: Vec<PyValue>) -> Vec<Value> {
    args.into_iter().map(|pv| pv.inner).collect()
}

fn py_values_from_rust(values: Vec<Value>) -> Vec<PyValue> {
    values.into_iter().map(|inner| PyValue { inner }).collect()
}

fn py_shape_dtype_from_rust(value: &Value) -> PyShapeDtypeStruct {
    PyShapeDtypeStruct {
        shape: value
            .as_tensor()
            .map_or_else(Vec::new, |tensor| tensor.shape.dims.clone()),
        dtype: format!("{:?}", value.dtype()),
    }
}

fn runtime_error(error: impl ToString) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
}

fn validate_cpu_backend(backend: Option<&str>) -> PyResult<()> {
    match backend {
        None | Some("cpu") => Ok(()),
        Some(backend) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported backend '{backend}'; fj-py currently exposes only the CPU backend"
        ))),
    }
}

fn validate_cpu_device(device: &PyDevice) -> PyResult<()> {
    if device.id == 0 && device.process_index == 0 {
        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "fj-py currently supports exactly one local CPU device",
        ))
    }
}

fn cpu_device() -> PyDevice {
    PyDevice {
        id: 0,
        process_index: 0,
    }
}

fn validate_single_cpu_device(devices: &[PyDevice]) -> PyResult<()> {
    match devices {
        [device] => validate_cpu_device(device),
        [] => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "devices must be a non-empty sequence",
        )),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "fj-py currently supports exactly one local CPU device",
        )),
    }
}

fn version_info() -> (u64, u64, u64) {
    (
        env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0),
        env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0),
        env!("CARGO_PKG_VERSION_PATCH").parse().unwrap_or(0),
    )
}

fn environment_info() -> String {
    let mut info = format!(
        "jax:    {}\n\
         jaxlib: unavailable\n\
         numpy:  unavailable\n\
         python: unavailable\n\
         device info: cpu-1, 1 local devices\n\
         process_count: 1\n\
         platform: {}-{}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );

    let mut jax_env: Vec<_> = std::env::vars()
        .filter(|(key, _)| key.starts_with("JAX_") || key.starts_with("XLA_"))
        .collect();
    jax_env.sort_by(|left, right| left.0.cmp(&right.0));
    for (key, value) in jax_env {
        info.push('\n');
        info.push_str(&key);
        info.push('=');
        info.push_str(&value);
    }

    info
}

#[pyfunction]
fn make_jaxpr_square() -> PyJaxpr {
    PyJaxpr {
        inner: build_program(ProgramSpec::Square),
    }
}

#[pyfunction]
fn make_jaxpr_add2() -> PyJaxpr {
    PyJaxpr {
        inner: build_program(ProgramSpec::Add2),
    }
}

#[pyfunction]
fn make_jaxpr_add_one() -> PyJaxpr {
    PyJaxpr {
        inner: build_program(ProgramSpec::AddOne),
    }
}

#[pyfunction]
fn jit(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
    let rust_args = py_values_to_rust(args);
    fj_api::jit(jaxpr.inner.clone())
        .call(rust_args)
        .map(py_values_from_rust)
        .map_err(runtime_error)
}

#[pyfunction]
fn grad(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
    let rust_args = py_values_to_rust(args);
    fj_api::grad(jaxpr.inner.clone())
        .call(rust_args)
        .map(py_values_from_rust)
        .map_err(runtime_error)
}

#[pyfunction]
fn jvp(
    jaxpr: &PyJaxpr,
    primals: Vec<PyValue>,
    tangents: Vec<PyValue>,
) -> PyResult<(Vec<PyValue>, Vec<PyValue>)> {
    let rust_primals = py_values_to_rust(primals);
    let rust_tangents = py_values_to_rust(tangents);
    fj_ad::jvp(&jaxpr.inner, &rust_primals, &rust_tangents)
        .map(|result| {
            (
                py_values_from_rust(result.primals),
                py_values_from_rust(result.tangents),
            )
        })
        .map_err(runtime_error)
}

#[pyfunction]
fn vjp(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<(Vec<PyValue>, PyVjpPullback)> {
    let rust_args = py_values_to_rust(args);
    fj_api::jit(jaxpr.inner.clone())
        .call(rust_args.clone())
        .map(|outputs| {
            (
                py_values_from_rust(outputs),
                PyVjpPullback {
                    jaxpr: jaxpr.inner.clone(),
                    primals: rust_args,
                },
            )
        })
        .map_err(runtime_error)
}

#[pyfunction]
fn linearize(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<(Vec<PyValue>, PyLinearizedJvp)> {
    let rust_args = py_values_to_rust(args);
    fj_api::jit(jaxpr.inner.clone())
        .call(rust_args.clone())
        .map(|outputs| {
            (
                py_values_from_rust(outputs),
                PyLinearizedJvp {
                    jaxpr: jaxpr.inner.clone(),
                    primals: rust_args,
                },
            )
        })
        .map_err(runtime_error)
}

#[pyfunction]
fn eval_shape(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<Vec<PyShapeDtypeStruct>> {
    let rust_args = py_values_to_rust(args);
    fj_api::jit(jaxpr.inner.clone())
        .call(rust_args)
        .map(|outputs| outputs.iter().map(py_shape_dtype_from_rust).collect())
        .map_err(runtime_error)
}

#[pyfunction(name = "typeof")]
fn typeof_value(value: &PyValue) -> PyShapeDtypeStruct {
    py_shape_dtype_from_rust(&value.inner)
}

#[pyfunction]
fn device_put(value: PyValue) -> PyValue {
    value
}

#[pyfunction]
fn device_put_replicated(value: PyValue, devices: Vec<PyDevice>) -> PyResult<PyValue> {
    validate_single_cpu_device(&devices)?;
    Ok(value)
}

#[pyfunction]
fn device_put_sharded(shards: Vec<PyValue>, devices: Vec<PyDevice>) -> PyResult<PyValue> {
    if shards.len() != devices.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "len(shards) = {} must equal len(devices) = {}",
            shards.len(),
            devices.len()
        )));
    }
    validate_single_cpu_device(&devices)?;

    match shards.into_iter().next() {
        Some(shard) => Ok(shard),
        None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "shards must be a non-empty sequence",
        )),
    }
}

#[pyfunction]
fn device_get(value: PyValue) -> PyValue {
    value
}

#[pyfunction]
fn block_until_ready(value: PyValue) -> PyValue {
    value
}

#[pyfunction]
fn copy_to_host_async(value: PyValue) -> PyValue {
    value
}

#[pyfunction]
fn effects_barrier() {}

#[pyfunction]
fn clear_caches() {}

#[pyfunction]
fn clear_backends() {}

#[pyfunction]
fn clean_up() {}

#[pyfunction(signature = (platform=None))]
fn live_arrays(platform: Option<String>) -> PyResult<Vec<PyValue>> {
    validate_cpu_backend(platform.as_deref())?;
    Ok(Vec::new())
}

#[pyfunction(signature = (device=None))]
fn default_device(py: Python<'_>, device: Option<Py<PyAny>>) -> PyResult<PyNamedScope> {
    let name = match device {
        None => "default_device(None)".to_owned(),
        Some(device) => {
            let device = device.bind(py);
            if device.is_none() {
                "default_device(None)".to_owned()
            } else if let Ok(platform) = device.extract::<String>() {
                validate_cpu_backend(Some(platform.as_str()))?;
                format!("default_device({platform:?})")
            } else if let Ok(device) = device.extract::<PyRef<'_, PyDevice>>() {
                validate_cpu_device(&device)?;
                format!(
                    "default_device(Device(id={}, process_index={}))",
                    device.id, device.process_index
                )
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "default_device expects None, 'cpu', or a fj-py Device",
                ));
            }
        }
    };

    Ok(PyNamedScope { name })
}

#[pyfunction]
fn default_backend() -> &'static str {
    "cpu"
}

#[pyfunction(signature = (backend=None))]
fn device_count(backend: Option<String>) -> PyResult<usize> {
    validate_cpu_backend(backend.as_deref())?;
    Ok(1)
}

#[pyfunction(signature = (backend=None))]
fn local_device_count(backend: Option<String>) -> PyResult<usize> {
    validate_cpu_backend(backend.as_deref())?;
    Ok(1)
}

#[pyfunction(signature = (backend=None))]
fn devices(backend: Option<String>) -> PyResult<Vec<PyDevice>> {
    validate_cpu_backend(backend.as_deref())?;
    Ok(vec![cpu_device()])
}

#[pyfunction(signature = (process_index=None, backend=None, host_id=None))]
fn local_devices(
    process_index: Option<usize>,
    backend: Option<String>,
    host_id: Option<usize>,
) -> PyResult<Vec<PyDevice>> {
    validate_cpu_backend(backend.as_deref())?;
    match host_id.or(process_index) {
        None | Some(0) => Ok(vec![cpu_device()]),
        Some(process_index) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown process_index {process_index}"
        ))),
    }
}

#[pyfunction(signature = (backend=None))]
fn process_index(backend: Option<String>) -> PyResult<usize> {
    validate_cpu_backend(backend.as_deref())?;
    Ok(0)
}

#[pyfunction(signature = (backend=None))]
fn process_count(backend: Option<String>) -> PyResult<usize> {
    validate_cpu_backend(backend.as_deref())?;
    Ok(1)
}

#[pyfunction(signature = (backend=None))]
fn process_indices(backend: Option<String>) -> PyResult<Vec<usize>> {
    validate_cpu_backend(backend.as_deref())?;
    Ok(vec![0])
}

#[pyfunction(signature = (backend=None))]
fn host_id(backend: Option<String>) -> PyResult<usize> {
    process_index(backend)
}

#[pyfunction(signature = (backend=None))]
fn host_count(backend: Option<String>) -> PyResult<usize> {
    process_count(backend)
}

#[pyfunction(signature = (backend=None))]
fn host_ids(backend: Option<String>) -> PyResult<Vec<usize>> {
    process_indices(backend)
}

#[pyfunction(signature = (function, name=None))]
fn named_call(function: Py<PyAny>, _name: Option<String>) -> Py<PyAny> {
    function
}

#[pyfunction]
fn named_scope(name: String) -> PyNamedScope {
    PyNamedScope { name }
}

#[pyfunction(signature = (checks=true))]
fn enable_checks(checks: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("enable_checks({checks})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn check_tracer_leaks(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("check_tracer_leaks({enabled})"),
    }
}

#[pyfunction]
fn checking_leaks() -> PyNamedScope {
    PyNamedScope {
        name: "checking_leaks".to_owned(),
    }
}

#[pyfunction(signature = (enabled=true))]
fn debug_nans(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("debug_nans({enabled})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn debug_infs(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("debug_infs({enabled})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn log_compiles(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("log_compiles({enabled})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn explain_cache_misses(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("explain_cache_misses({enabled})"),
    }
}

#[pyfunction(signature = (disable=true))]
fn disable_jit(disable: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("disable_jit({disable})"),
    }
}

#[pyfunction]
fn ensure_compile_time_eval() -> PyNamedScope {
    PyNamedScope {
        name: "ensure_compile_time_eval".to_owned(),
    }
}

#[pyfunction(signature = (return_string=false))]
fn print_environment_info(py: Python<'_>, return_string: bool) -> PyResult<Option<String>> {
    let info = environment_info();
    if return_string {
        Ok(Some(info))
    } else {
        py.import_bound("builtins")?
            .getattr("print")?
            .call1((info,))?;
        Ok(None)
    }
}

#[pyfunction]
fn vmap(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
    let rust_args = py_values_to_rust(args);
    fj_api::vmap(jaxpr.inner.clone())
        .call(rust_args)
        .map(py_values_from_rust)
        .map_err(runtime_error)
}

#[pyfunction]
fn pmap(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
    let rust_args = py_values_to_rust(args);
    fj_api::pmap(jaxpr.inner.clone())
        .call(rust_args)
        .map(py_values_from_rust)
        .map_err(runtime_error)
}

#[pyfunction]
fn value_and_grad(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<(Vec<PyValue>, Vec<PyValue>)> {
    let rust_args = py_values_to_rust(args);
    fj_api::value_and_grad(jaxpr.inner.clone())
        .call(rust_args)
        .map(|(values, grads)| (py_values_from_rust(values), py_values_from_rust(grads)))
        .map_err(runtime_error)
}

#[pyfunction]
fn jacobian(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<PyValue> {
    let rust_args = py_values_to_rust(args);
    fj_api::jacobian(jaxpr.inner.clone())
        .call(rust_args)
        .map(|inner| PyValue { inner })
        .map_err(runtime_error)
}

#[pyfunction]
fn jacrev(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<PyValue> {
    jacobian(jaxpr, args)
}

#[pyfunction]
fn jacfwd(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<PyValue> {
    jacobian(jaxpr, args)
}

#[pyfunction]
fn hessian(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<PyValue> {
    let rust_args = py_values_to_rust(args);
    fj_api::hessian(jaxpr.inner.clone())
        .call(rust_args)
        .map(|inner| PyValue { inner })
        .map_err(runtime_error)
}

#[pyfunction]
fn checkpoint(jaxpr: &PyJaxpr) -> PyCheckpoint {
    PyCheckpoint {
        inner: fj_api::checkpoint(jaxpr.inner.clone()),
    }
}

#[pyfunction]
fn remat(jaxpr: &PyJaxpr) -> PyCheckpoint {
    checkpoint(jaxpr)
}

#[pymodule]
fn frankenjax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__version_info__", version_info())?;
    m.add_class::<PyValue>()?;
    m.add("Array", m.getattr("PyValue")?)?;
    m.add_class::<PyJaxpr>()?;
    m.add_class::<PyCheckpoint>()?;
    m.add_class::<PyVjpPullback>()?;
    m.add_class::<PyLinearizedJvp>()?;
    m.add_class::<PyShapeDtypeStruct>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyNamedScope>()?;
    m.add_function(wrap_pyfunction!(make_jaxpr_square, m)?)?;
    m.add_function(wrap_pyfunction!(make_jaxpr_add2, m)?)?;
    m.add_function(wrap_pyfunction!(make_jaxpr_add_one, m)?)?;
    m.add_function(wrap_pyfunction!(jit, m)?)?;
    m.add_function(wrap_pyfunction!(grad, m)?)?;
    m.add_function(wrap_pyfunction!(jvp, m)?)?;
    m.add_function(wrap_pyfunction!(vjp, m)?)?;
    m.add_function(wrap_pyfunction!(linearize, m)?)?;
    m.add_function(wrap_pyfunction!(eval_shape, m)?)?;
    m.add_function(wrap_pyfunction!(typeof_value, m)?)?;
    m.add_function(wrap_pyfunction!(device_put, m)?)?;
    m.add_function(wrap_pyfunction!(device_put_replicated, m)?)?;
    m.add_function(wrap_pyfunction!(device_put_sharded, m)?)?;
    m.add_function(wrap_pyfunction!(device_get, m)?)?;
    m.add_function(wrap_pyfunction!(block_until_ready, m)?)?;
    m.add_function(wrap_pyfunction!(copy_to_host_async, m)?)?;
    m.add_function(wrap_pyfunction!(effects_barrier, m)?)?;
    m.add_function(wrap_pyfunction!(clear_caches, m)?)?;
    m.add_function(wrap_pyfunction!(clear_backends, m)?)?;
    m.add_function(wrap_pyfunction!(clean_up, m)?)?;
    m.add_function(wrap_pyfunction!(live_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(default_device, m)?)?;
    m.add_function(wrap_pyfunction!(default_backend, m)?)?;
    m.add_function(wrap_pyfunction!(device_count, m)?)?;
    m.add_function(wrap_pyfunction!(local_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_function(wrap_pyfunction!(local_devices, m)?)?;
    m.add_function(wrap_pyfunction!(process_index, m)?)?;
    m.add_function(wrap_pyfunction!(process_count, m)?)?;
    m.add_function(wrap_pyfunction!(process_indices, m)?)?;
    m.add_function(wrap_pyfunction!(host_id, m)?)?;
    m.add_function(wrap_pyfunction!(host_count, m)?)?;
    m.add_function(wrap_pyfunction!(host_ids, m)?)?;
    m.add_function(wrap_pyfunction!(named_call, m)?)?;
    m.add_function(wrap_pyfunction!(named_scope, m)?)?;
    m.add_function(wrap_pyfunction!(enable_checks, m)?)?;
    m.add_function(wrap_pyfunction!(check_tracer_leaks, m)?)?;
    m.add_function(wrap_pyfunction!(checking_leaks, m)?)?;
    m.add_function(wrap_pyfunction!(debug_nans, m)?)?;
    m.add_function(wrap_pyfunction!(debug_infs, m)?)?;
    m.add_function(wrap_pyfunction!(log_compiles, m)?)?;
    m.add_function(wrap_pyfunction!(explain_cache_misses, m)?)?;
    m.add_function(wrap_pyfunction!(disable_jit, m)?)?;
    m.add_function(wrap_pyfunction!(ensure_compile_time_eval, m)?)?;
    m.add_function(wrap_pyfunction!(print_environment_info, m)?)?;
    m.add_function(wrap_pyfunction!(vmap, m)?)?;
    m.add_function(wrap_pyfunction!(pmap, m)?)?;
    m.add_function(wrap_pyfunction!(value_and_grad, m)?)?;
    m.add_function(wrap_pyfunction!(jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(jacrev, m)?)?;
    m.add_function(wrap_pyfunction!(jacfwd, m)?)?;
    m.add_function(wrap_pyfunction!(hessian, m)?)?;
    m.add_function(wrap_pyfunction!(checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(remat, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_scalar_roundtrip() {
        let v = PyValue::scalar_f64(42.0);
        assert!((v.as_f64().unwrap() - 42.0).abs() < 1e-12);
    }

    #[test]
    fn value_vector_roundtrip() {
        let floats = PyValue::vector_f64(vec![1.0, 2.5, 4.0]).unwrap();
        assert_eq!(floats.shape(), vec![3]);
        assert_eq!(floats.dtype(), "F64");
        assert_eq!(floats.as_f64_list().unwrap(), vec![1.0, 2.5, 4.0]);
        assert_eq!(floats.as_i64_list(), None);

        let ints = PyValue::vector_i64(vec![1, 2, 3]).unwrap();
        assert_eq!(ints.shape(), vec![3]);
        assert_eq!(ints.dtype(), "I64");
        assert_eq!(ints.as_i64_list().unwrap(), vec![1, 2, 3]);
        assert_eq!(ints.as_f64_list().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn jaxpr_square_builds() {
        let jaxpr = make_jaxpr_square();
        assert!(!jaxpr.inner.equations.is_empty());
    }

    #[test]
    fn environment_info_reports_cpu_local_runtime() {
        let info = environment_info();
        assert!(info.contains("jax:    0.1.0"));
        assert!(info.contains("device info: cpu-1, 1 local devices"));
        assert!(info.contains("process_count: 1"));
    }

    #[test]
    fn checkpoint_wrapper_exposes_forward_grad_and_savings() {
        let jaxpr = make_jaxpr_square();
        let wrapped = checkpoint(&jaxpr);

        assert_eq!(
            wrapped.memory_savings_entries(),
            jaxpr.inner.equations.len()
        );

        let values = wrapped.call(vec![PyValue::scalar_f64(3.0)]).unwrap();
        assert!((values[0].as_f64().unwrap() - 9.0).abs() < 1e-12);

        let grads = wrapped.grad(vec![PyValue::scalar_f64(3.0)]).unwrap();
        assert!((grads[0].as_f64().unwrap() - 6.0).abs() < 1e-9);

        let (values, grads) = wrapped
            .value_and_grad(vec![PyValue::scalar_f64(4.0)])
            .unwrap();
        assert!((values[0].as_f64().unwrap() - 16.0).abs() < 1e-12);
        assert!((grads[0].as_f64().unwrap() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn remat_alias_exposes_checkpoint_wrapper() {
        let jaxpr = make_jaxpr_square();
        let wrapped = remat(&jaxpr);

        assert_eq!(
            wrapped.memory_savings_entries(),
            jaxpr.inner.equations.len()
        );

        let values = wrapped.call(vec![PyValue::scalar_f64(3.0)]).unwrap();
        assert!((values[0].as_f64().unwrap() - 9.0).abs() < 1e-12);

        let grads = wrapped.grad(vec![PyValue::scalar_f64(3.0)]).unwrap();
        assert!((grads[0].as_f64().unwrap() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn jvp_wrapper_returns_primal_and_tangent_outputs() {
        let jaxpr = make_jaxpr_square();
        let (primals, tangents) = jvp(
            &jaxpr,
            vec![PyValue::scalar_f64(3.0)],
            vec![PyValue::scalar_f64(1.0)],
        )
        .unwrap();

        assert_eq!(primals.len(), 1);
        assert_eq!(tangents.len(), 1);
        assert!((primals[0].as_f64().unwrap() - 9.0).abs() < 1e-12);
        assert!((tangents[0].as_f64().unwrap() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn vjp_wrapper_returns_outputs_and_pullback() {
        let jaxpr = make_jaxpr_square();
        let (values, pullback) = vjp(&jaxpr, vec![PyValue::scalar_f64(3.0)]).unwrap();
        assert_eq!(values.len(), 1);
        assert!((values[0].as_f64().unwrap() - 9.0).abs() < 1e-12);

        let grads = pullback.call(vec![PyValue::scalar_f64(1.0)]).unwrap();
        assert_eq!(grads.len(), 1);
        assert!((grads[0].as_f64().unwrap() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn linearize_wrapper_returns_outputs_and_linearized_jvp() {
        let jaxpr = make_jaxpr_square();
        let (values, linearized) = linearize(&jaxpr, vec![PyValue::scalar_f64(3.0)]).unwrap();
        assert_eq!(values.len(), 1);
        assert!((values[0].as_f64().unwrap() - 9.0).abs() < 1e-12);

        let tangents = linearized.call(vec![PyValue::scalar_f64(1.0)]).unwrap();
        assert_eq!(tangents.len(), 1);
        assert!((tangents[0].as_f64().unwrap() - 6.0).abs() < 1e-9);

        let scaled_tangents = linearized.call(vec![PyValue::scalar_f64(2.0)]).unwrap();
        assert_eq!(scaled_tangents.len(), 1);
        assert!((scaled_tangents[0].as_f64().unwrap() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn eval_shape_returns_output_shape_dtype_metadata() {
        let scalar_meta = eval_shape(&make_jaxpr_square(), vec![PyValue::scalar_f64(3.0)]).unwrap();
        assert_eq!(scalar_meta.len(), 1);
        assert_eq!(scalar_meta[0].shape(), Vec::<u32>::new());
        assert_eq!(scalar_meta[0].dtype(), "F64");

        let vector_meta = eval_shape(
            &make_jaxpr_add_one(),
            vec![PyValue::vector_f64(vec![1.0, 2.0, 3.0]).unwrap()],
        )
        .unwrap();
        assert_eq!(vector_meta.len(), 1);
        assert_eq!(vector_meta[0].shape(), vec![3]);
        assert_eq!(vector_meta[0].dtype(), "F64");
    }

    #[test]
    fn typeof_returns_value_shape_dtype_metadata() {
        let scalar_meta = typeof_value(&PyValue::scalar_i64(7));
        assert_eq!(scalar_meta.shape(), Vec::<u32>::new());
        assert_eq!(scalar_meta.dtype(), "I64");

        let vector = PyValue::vector_f64(vec![1.0, 2.0, 3.0]).unwrap();
        let vector_meta = typeof_value(&vector);
        assert_eq!(vector_meta.shape(), vec![3]);
        assert_eq!(vector_meta.dtype(), "F64");
    }

    #[test]
    fn cpu_local_device_helpers_preserve_values() {
        let scalar = PyValue::scalar_f64(3.5);
        let put_scalar = device_put(scalar.clone());
        assert!((put_scalar.as_f64().unwrap() - 3.5).abs() < 1e-12);
        let replicated_scalar = device_put_replicated(scalar.clone(), vec![cpu_device()]).unwrap();
        assert!((replicated_scalar.as_f64().unwrap() - 3.5).abs() < 1e-12);
        assert!(device_put_replicated(scalar.clone(), Vec::new()).is_err());
        let ready_scalar = block_until_ready(put_scalar);
        assert!((ready_scalar.as_f64().unwrap() - 3.5).abs() < 1e-12);
        let host_scalar = device_get(ready_scalar);
        assert!((host_scalar.as_f64().unwrap() - 3.5).abs() < 1e-12);

        let vector = PyValue::vector_i64(vec![1, 2, 3]).unwrap();
        let sharded_vector = device_put_sharded(vec![vector.clone()], vec![cpu_device()]).unwrap();
        assert_eq!(sharded_vector.as_i64_list().unwrap(), vec![1, 2, 3]);
        assert!(device_put_sharded(vec![vector.clone()], Vec::new()).is_err());
        assert!(device_put_sharded(Vec::new(), vec![cpu_device()]).is_err());
        let host_vector = device_get(block_until_ready(device_put(vector)));
        assert_eq!(host_vector.shape(), vec![3]);
        assert_eq!(host_vector.dtype(), "I64");
        assert_eq!(host_vector.as_i64_list().unwrap(), vec![1, 2, 3]);

        let copied_vector = copy_to_host_async(host_vector);
        assert_eq!(copied_vector.shape(), vec![3]);
        assert_eq!(copied_vector.dtype(), "I64");
        assert_eq!(copied_vector.as_i64_list().unwrap(), vec![1, 2, 3]);

        effects_barrier();
        clear_caches();
    }

    #[test]
    fn cpu_backend_topology_helpers_report_single_local_device() {
        assert_eq!(default_backend(), "cpu");
        assert_eq!(device_count(None::<String>).unwrap(), 1);
        assert_eq!(device_count(Some("cpu".to_owned())).unwrap(), 1);
        assert!(device_count(Some("gpu".to_owned())).is_err());
        assert_eq!(local_device_count(None::<String>).unwrap(), 1);
        assert_eq!(process_index(None::<String>).unwrap(), 0);
        assert_eq!(process_count(None::<String>).unwrap(), 1);
        assert_eq!(process_indices(None::<String>).unwrap(), vec![0]);
        assert_eq!(host_id(None::<String>).unwrap(), 0);
        assert_eq!(host_count(None::<String>).unwrap(), 1);
        assert_eq!(host_ids(None::<String>).unwrap(), vec![0]);

        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            assert_eq!(
                default_device(py, None).unwrap().name(),
                "default_device(None)"
            );
        });

        let all_devices = devices(None::<String>).unwrap();
        assert_eq!(all_devices.len(), 1);
        assert_eq!(all_devices[0].id(), 0);
        assert_eq!(all_devices[0].process_index(), 0);
        assert_eq!(all_devices[0].platform(), "cpu");
        assert_eq!(all_devices[0].device_kind(), "cpu");

        let local = local_devices(None, None::<String>, None).unwrap();
        assert_eq!(local.len(), 1);
        assert_eq!(local[0].id(), 0);
        assert_eq!(
            local_devices(Some(0), None::<String>, None).unwrap().len(),
            1
        );
        assert_eq!(
            local_devices(None, None::<String>, Some(0)).unwrap().len(),
            1
        );
        assert!(local_devices(Some(1), None::<String>, None).is_err());
        assert!(local_devices(None, None::<String>, Some(1)).is_err());
    }

    #[test]
    fn version_metadata_matches_crate_package_version() {
        assert_eq!(env!("CARGO_PKG_VERSION"), "0.1.0");
        assert_eq!(version_info(), (0, 1, 0));
    }

    #[test]
    fn named_scope_tracks_name() {
        let scope = named_scope("layer".to_owned());
        assert_eq!(scope.name(), "layer");
        let empty_scope = named_scope(String::new());
        assert_eq!(empty_scope.name(), "");
    }

    #[test]
    fn local_context_helpers_track_names() {
        assert_eq!(enable_checks(true).name(), "enable_checks(true)");
        assert_eq!(enable_checks(false).name(), "enable_checks(false)");
        assert_eq!(check_tracer_leaks(true).name(), "check_tracer_leaks(true)");
        assert_eq!(
            check_tracer_leaks(false).name(),
            "check_tracer_leaks(false)"
        );
        assert_eq!(checking_leaks().name(), "checking_leaks");
        assert_eq!(debug_nans(true).name(), "debug_nans(true)");
        assert_eq!(debug_nans(false).name(), "debug_nans(false)");
        assert_eq!(debug_infs(true).name(), "debug_infs(true)");
        assert_eq!(debug_infs(false).name(), "debug_infs(false)");
        assert_eq!(log_compiles(true).name(), "log_compiles(true)");
        assert_eq!(log_compiles(false).name(), "log_compiles(false)");
        assert_eq!(
            explain_cache_misses(true).name(),
            "explain_cache_misses(true)"
        );
        assert_eq!(
            explain_cache_misses(false).name(),
            "explain_cache_misses(false)"
        );
        assert_eq!(disable_jit(true).name(), "disable_jit(true)");
        assert_eq!(disable_jit(false).name(), "disable_jit(false)");
        assert_eq!(
            ensure_compile_time_eval().name(),
            "ensure_compile_time_eval"
        );
    }

    #[test]
    fn backend_cleanup_helpers_are_cpu_local() {
        clear_caches();
        clear_backends();
        clean_up();

        assert!(live_arrays(None).unwrap().is_empty());
        assert!(live_arrays(Some("cpu".to_owned())).unwrap().is_empty());
        assert!(live_arrays(Some("gpu".to_owned())).is_err());
    }

    #[test]
    fn jacobian_and_hessian_wrappers_return_python_values() {
        let jaxpr = make_jaxpr_square();
        let args = vec![PyValue::scalar_f64(3.0)];

        let jac = jacobian(&jaxpr, args.clone()).unwrap();
        assert_eq!(jac.shape(), vec![1, 1]);
        assert!((jac.as_f64_list().unwrap()[0] - 6.0).abs() < 1e-9);

        let jac_rev = jacrev(&jaxpr, args.clone()).unwrap();
        assert_eq!(jac_rev.shape(), vec![1, 1]);
        assert!((jac_rev.as_f64_list().unwrap()[0] - 6.0).abs() < 1e-9);

        let jac_fwd = jacfwd(&jaxpr, args.clone()).unwrap();
        assert_eq!(jac_fwd.shape(), vec![1, 1]);
        assert!((jac_fwd.as_f64_list().unwrap()[0] - 6.0).abs() < 1e-9);

        let hess = hessian(&jaxpr, args).unwrap();
        assert_eq!(hess.shape(), vec![1, 1]);
        assert!((hess.as_f64_list().unwrap()[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn pmap_wrapper_surfaces_fail_closed_error() {
        pyo3::prepare_freethreaded_python();
        let jaxpr = make_jaxpr_add_one();
        let result = pmap(&jaxpr, vec![PyValue::vector_f64(vec![1.0, 2.0]).unwrap()]);
        assert!(
            result.is_err(),
            "pmap should fail closed without multi-device context"
        );
        let err = result
            .err()
            .expect("pmap error should be available after is_err");
        let message = err.to_string().to_ascii_lowercase();
        assert!(message.contains("pmap unavailable"), "{message}");
        assert!(message.contains("multi-device"), "{message}");
    }
}
