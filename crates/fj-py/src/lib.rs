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

    fn as_f64_list(&self) -> Option<Vec<f64>> {
        self.inner.as_tensor().and_then(|tensor| tensor.to_f64_vec())
    }

    fn as_i64_list(&self) -> Option<Vec<i64>> {
        self.inner.as_tensor().and_then(|tensor| tensor.to_i64_vec())
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

fn py_values_to_rust(args: Vec<PyValue>) -> Vec<Value> {
    args.into_iter().map(|pv| pv.inner).collect()
}

fn py_values_from_rust(values: Vec<Value>) -> Vec<PyValue> {
    values.into_iter().map(|inner| PyValue { inner }).collect()
}

fn runtime_error(error: impl ToString) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
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
fn vmap(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
    let rust_args = py_values_to_rust(args);
    fj_api::vmap(jaxpr.inner.clone())
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
fn checkpoint(jaxpr: &PyJaxpr) -> PyCheckpoint {
    PyCheckpoint {
        inner: fj_api::checkpoint(jaxpr.inner.clone()),
    }
}

#[pymodule]
fn frankenjax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyValue>()?;
    m.add_class::<PyJaxpr>()?;
    m.add_class::<PyCheckpoint>()?;
    m.add_function(wrap_pyfunction!(make_jaxpr_square, m)?)?;
    m.add_function(wrap_pyfunction!(make_jaxpr_add2, m)?)?;
    m.add_function(wrap_pyfunction!(make_jaxpr_add_one, m)?)?;
    m.add_function(wrap_pyfunction!(jit, m)?)?;
    m.add_function(wrap_pyfunction!(grad, m)?)?;
    m.add_function(wrap_pyfunction!(vmap, m)?)?;
    m.add_function(wrap_pyfunction!(value_and_grad, m)?)?;
    m.add_function(wrap_pyfunction!(checkpoint, m)?)?;
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
        assert_eq!(floats.as_f64_list().unwrap(), vec![1.0, 2.5, 4.0]);
        assert_eq!(floats.as_i64_list(), None);

        let ints = PyValue::vector_i64(vec![1, 2, 3]).unwrap();
        assert_eq!(ints.as_i64_list().unwrap(), vec![1, 2, 3]);
        assert_eq!(ints.as_f64_list(), None);
    }

    #[test]
    fn jaxpr_square_builds() {
        let jaxpr = make_jaxpr_square();
        assert!(!jaxpr.inner.equations.is_empty());
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
}
