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

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
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
    let rust_args: Vec<Value> = args.iter().map(|pv| pv.inner.clone()).collect();
    fj_api::jit(jaxpr.inner.clone())
        .call(rust_args)
        .map(|outputs| outputs.into_iter().map(|v| PyValue { inner: v }).collect())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
fn grad(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
    let rust_args: Vec<Value> = args.iter().map(|pv| pv.inner.clone()).collect();
    fj_api::grad(jaxpr.inner.clone())
        .call(rust_args)
        .map(|outputs| outputs.into_iter().map(|v| PyValue { inner: v }).collect())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
fn vmap(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
    let rust_args: Vec<Value> = args.iter().map(|pv| pv.inner.clone()).collect();
    fj_api::vmap(jaxpr.inner.clone())
        .call(rust_args)
        .map(|outputs| outputs.into_iter().map(|v| PyValue { inner: v }).collect())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
fn value_and_grad(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyResult<(Vec<PyValue>, Vec<PyValue>)> {
    let rust_args: Vec<Value> = args.iter().map(|pv| pv.inner.clone()).collect();
    fj_api::value_and_grad(jaxpr.inner.clone())
        .call(rust_args)
        .map(|(values, grads)| {
            (
                values.into_iter().map(|v| PyValue { inner: v }).collect(),
                grads.into_iter().map(|v| PyValue { inner: v }).collect(),
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
fn checkpoint(jaxpr: &PyJaxpr) -> PyJaxpr {
    let _wrapped = fj_api::checkpoint(jaxpr.inner.clone());
    PyJaxpr {
        inner: jaxpr.inner.clone(),
    }
}

#[pymodule]
fn frankenjax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyValue>()?;
    m.add_class::<PyJaxpr>()?;
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
    fn jaxpr_square_builds() {
        let jaxpr = make_jaxpr_square();
        assert!(!jaxpr.inner.equations.is_empty());
    }
}
