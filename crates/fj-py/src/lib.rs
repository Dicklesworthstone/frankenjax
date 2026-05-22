#![forbid(unsafe_code)]

use pyo3::class::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyFrozenSet, PyList, PySet, PyTuple};

use fj_core::{DType, Jaxpr, Literal, ProgramSpec, Value, build_program};

#[pyclass]
#[derive(Clone)]
struct PyJaxpr {
    inner: Jaxpr,
}

#[pyclass]
#[derive(Clone)]
struct PyValue {
    inner: Value,
    deleted: bool,
}

impl PyValue {
    fn from_value(inner: Value) -> Self {
        Self {
            inner,
            deleted: false,
        }
    }

    fn shape_dims(&self) -> Vec<u32> {
        self.inner
            .as_tensor()
            .map_or_else(Vec::new, |tensor| tensor.shape.dims.clone())
    }
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

#[pyclass]
#[derive(Clone)]
struct PyForwardPass {
    jaxpr: Jaxpr,
}

#[pyclass]
#[derive(Clone)]
struct PyBackwardPass;

#[pyclass(name = "ShapeDtypeStruct")]
struct PyShapeDtypeStruct {
    shape: Vec<u32>,
    dtype: String,
    weak_type: bool,
    is_ref: bool,
    vma: Option<Py<PyAny>>,
    vma_hash: isize,
    vma_len: usize,
    vma_repr: Option<String>,
}

impl Clone for PyShapeDtypeStruct {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            weak_type: self.weak_type,
            is_ref: self.is_ref,
            vma: self.clone_vma(),
            vma_hash: self.vma_hash,
            vma_len: self.vma_len,
            vma_repr: self.vma_repr.clone(),
        }
    }
}

impl PyShapeDtypeStruct {
    fn require_sharding_none(sharding: Option<Py<PyAny>>) -> PyResult<()> {
        if sharding.is_some() {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ShapeDtypeStruct sharding is not supported by fj-py yet",
            ))
        } else {
            Ok(())
        }
    }

    fn normalize_vma(
        vma: Option<Py<PyAny>>,
    ) -> PyResult<(Option<Py<PyAny>>, isize, usize, Option<String>)> {
        let Some(vma) = vma else {
            return Ok((None, 0, 0, None));
        };

        Python::with_gil(|py| {
            let vma = vma.bind(py);
            if !vma.is_instance_of::<PySet>() && !vma.is_instance_of::<PyFrozenSet>() {
                let type_repr: String = vma.get_type().repr()?.extract()?;
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "`vma` argument passed to ShapeDtypeStruct should be of type `set` or `frozenset`. Got type {type_repr}"
                )));
            }

            let frozen = py.import("builtins")?.getattr("frozenset")?.call1((vma,))?;
            let vma_hash = frozen.hash()?;
            let vma_len = frozen.len()?;
            let vma_repr = frozen.repr()?.extract()?;
            Ok((Some(frozen.unbind()), vma_hash, vma_len, Some(vma_repr)))
        })
    }

    fn clone_vma(&self) -> Option<Py<PyAny>> {
        Python::with_gil(|py| self.vma.as_ref().map(|vma| vma.clone_ref(py)))
    }

    #[cfg(test)]
    fn shape_vec(&self) -> Vec<u32> {
        self.shape.clone()
    }

    fn same_metadata(&self, other: &Self) -> PyResult<bool> {
        let static_metadata_matches = self.shape == other.shape
            && self.dtype == other.dtype
            && self.weak_type == other.weak_type
            && self.is_ref == other.is_ref;
        if !static_metadata_matches {
            return Ok(false);
        }

        Python::with_gil(|py| match (&self.vma, &other.vma) {
            (None, None) => Ok(true),
            (Some(left), Some(right)) => left.bind(py).eq(right.bind(py)),
            _ => Ok(false),
        })
    }

    fn metadata_hash(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for dim in &self.shape {
            hash ^= u64::from(*dim);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for byte in self.dtype.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for flag in [self.weak_type, self.is_ref] {
            hash ^= u64::from(flag);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= u64::from(self.vma.is_some());
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        for byte in self.vma_hash.to_ne_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for byte in self.vma_len.to_ne_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }
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

#[pyclass]
#[derive(Clone)]
struct PyUserContext {
    default_repr: String,
}

#[pyclass(name = "_Float0DType")]
#[derive(Clone, Copy)]
struct PyFloat0DType;

#[pymethods]
impl PyValue {
    #[staticmethod]
    fn scalar_f64(v: f64) -> Self {
        PyValue::from_value(Value::scalar_f64(v))
    }

    #[staticmethod]
    fn scalar_i64(v: i64) -> Self {
        PyValue::from_value(Value::scalar_i64(v))
    }

    #[staticmethod]
    fn vector_i64(values: Vec<i64>) -> PyResult<Self> {
        Value::vector_i64(&values)
            .map(PyValue::from_value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    #[staticmethod]
    fn vector_f64(values: Vec<f64>) -> PyResult<Self> {
        Value::vector_f64(&values)
            .map(PyValue::from_value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn as_f64(&self) -> Option<f64> {
        self.inner.as_f64_scalar()
    }

    fn as_i64(&self) -> Option<i64> {
        self.inner.as_i64_scalar()
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        shape_to_py_tuple(py, &self.shape_dims())
    }

    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape_dims().len()
    }

    #[getter]
    fn size(&self) -> u64 {
        self.shape_dims()
            .iter()
            .fold(1_u64, |size, dim| size.saturating_mul(u64::from(*dim)))
    }

    #[getter]
    fn itemsize(&self) -> u64 {
        dtype_itemsize(self.inner.dtype())
    }

    #[getter]
    fn nbytes(&self) -> u64 {
        self.size().saturating_mul(self.itemsize())
    }

    #[getter]
    fn weak_type(&self) -> bool {
        false
    }

    #[getter]
    fn committed(&self) -> bool {
        false
    }

    #[getter]
    fn device(&self) -> PyDevice {
        cpu_device()
    }

    fn devices(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let devices = PySet::empty(py)?;
        devices.add(cpu_device())?;
        Ok(devices.into_any().unbind())
    }

    fn on_device_size_in_bytes(&self) -> u64 {
        self.nbytes()
    }

    #[getter]
    fn is_fully_addressable(&self) -> bool {
        true
    }

    #[getter]
    fn is_fully_replicated(&self) -> bool {
        true
    }

    fn __len__(&self) -> PyResult<usize> {
        let first_dim = self
            .inner
            .as_tensor()
            .and_then(|tensor| tensor.shape.dims.first().copied())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>("len() of unsized object")
            })?;

        usize::try_from(first_dim).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array length does not fit usize")
        })
    }

    fn block_until_ready(&self) -> Self {
        self.clone()
    }

    fn copy_to_host_async(&self) -> Self {
        self.clone()
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn delete(&mut self) {
        self.deleted = true;
    }

    fn is_deleted(&self) -> bool {
        self.deleted
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Value::Scalar(literal) => literal_to_py_object(py, *literal),
            Value::Tensor(tensor) => literals_to_py_list(py, &tensor.elements),
        }
    }

    #[pyo3(signature = (order = "C"))]
    fn tobytes(&self, py: Python<'_>, order: &str) -> PyResult<Py<PyAny>> {
        validate_tobytes_order(order)?;

        let capacity = usize::try_from(self.nbytes()).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "array byte length does not fit usize",
            )
        })?;
        let mut bytes = Vec::with_capacity(capacity);
        match &self.inner {
            Value::Scalar(literal) => literal_to_native_bytes(*literal, &mut bytes),
            Value::Tensor(tensor) => {
                for literal in &tensor.elements {
                    literal_to_native_bytes(*literal, &mut bytes);
                }
            }
        }

        Ok(PyBytes::new(py, &bytes).into_any().unbind())
    }

    fn __bool__(&self) -> PyResult<bool> {
        match self.size() {
            0 => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.",
            )),
            1 => {
                let literal = match &self.inner {
                    Value::Scalar(literal) => *literal,
                    Value::Tensor(tensor) => tensor.elements.first().copied().ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.",
                        )
                    })?,
                };
                Ok(literal_truth_value(literal))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()",
            )),
        }
    }

    fn __float__(&self) -> PyResult<f64> {
        let literal = self.inner.as_scalar_literal().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "only scalar arrays can be converted to Python scalars",
            )
        })?;

        literal
            .as_f64()
            .or_else(|| match literal {
                Literal::Bool(value) => Some(if value { 1.0 } else { 0.0 }),
                _ => None,
            })
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "only real scalar arrays can be converted to Python float",
                )
            })
    }

    fn __int__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let literal = self.inner.as_scalar_literal().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "only scalar arrays can be converted to Python scalars",
            )
        })?;

        let value = literal_to_py_object(py, literal)?;
        let builtins = py.import("builtins")?;
        Ok(builtins.getattr("int")?.call1((value.bind(py),))?.unbind())
    }

    fn __complex__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let literal = self.inner.as_scalar_literal().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "only scalar arrays can be converted to Python scalars",
            )
        })?;

        let value = literal_to_py_object(py, literal)?;
        let builtins = py.import("builtins")?;
        Ok(builtins
            .getattr("complex")?
            .call1((value.bind(py),))?
            .unbind())
    }

    fn __index__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let literal = self.inner.as_scalar_literal().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Only integer scalar arrays can be converted to a scalar index.",
            )
        })?;

        match literal {
            Literal::I64(_) | Literal::U32(_) | Literal::U64(_) => {
                literal_to_py_object(py, literal)
            }
            Literal::Bool(_)
            | Literal::BF16Bits(_)
            | Literal::F16Bits(_)
            | Literal::F32Bits(_)
            | Literal::F64Bits(_)
            | Literal::Complex64Bits(..)
            | Literal::Complex128Bits(..) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Only integer scalar arrays can be converted to a scalar index.",
            )),
        }
    }

    fn __hex__(&self, py: Python<'_>) -> PyResult<String> {
        let value = self.__index__(py)?;
        py.import("builtins")?
            .getattr("hex")?
            .call1((value.bind(py),))?
            .extract()
    }

    fn __oct__(&self, py: Python<'_>) -> PyResult<String> {
        let value = self.__index__(py)?;
        py.import("builtins")?
            .getattr("oct")?
            .call1((value.bind(py),))?
            .extract()
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
impl PyForwardPass {
    fn call(&self, args: Vec<PyValue>) -> PyResult<(Vec<PyValue>, PyVjpPullback)> {
        vjp_outputs_and_pullback(&self.jaxpr, args)
    }

    fn __call__(&self, args: Vec<PyValue>) -> PyResult<(Vec<PyValue>, PyVjpPullback)> {
        self.call(args)
    }

    fn __repr__(&self) -> String {
        "PyForwardPass()".to_owned()
    }
}

#[pymethods]
impl PyBackwardPass {
    fn call(&self, residuals: &PyVjpPullback, cotangents: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
        residuals.call(cotangents)
    }

    fn __call__(
        &self,
        residuals: &PyVjpPullback,
        cotangents: Vec<PyValue>,
    ) -> PyResult<Vec<PyValue>> {
        self.call(residuals, cotangents)
    }

    fn __repr__(&self) -> String {
        "PyBackwardPass()".to_owned()
    }
}

#[pymethods]
impl PyShapeDtypeStruct {
    #[new]
    #[pyo3(signature = (shape, dtype, *, sharding=None, weak_type=false, vma=None, is_ref=false))]
    fn new(
        shape: Vec<u32>,
        dtype: Option<String>,
        sharding: Option<Py<PyAny>>,
        weak_type: bool,
        vma: Option<Py<PyAny>>,
        is_ref: bool,
    ) -> PyResult<Self> {
        let dtype = dtype.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ShapeDtypeStruct: dtype must be specified.",
            )
        })?;
        Self::require_sharding_none(sharding)?;
        let (vma, vma_hash, vma_len, vma_repr) = Self::normalize_vma(vma)?;
        Ok(Self {
            shape,
            dtype,
            weak_type,
            is_ref,
            vma,
            vma_hash,
            vma_len,
            vma_repr,
        })
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        shape_to_py_tuple(py, &self.shape)
    }

    #[getter]
    fn dtype(&self) -> String {
        self.dtype.clone()
    }

    #[getter]
    fn sharding(&self) -> Option<Py<PyAny>> {
        None
    }

    #[getter]
    fn vma(&self) -> Option<Py<PyAny>> {
        self.clone_vma()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[getter]
    fn size(&self) -> u64 {
        self.shape
            .iter()
            .fold(1_u64, |size, dim| size.saturating_mul(u64::from(*dim)))
    }

    fn __len__(&self) -> PyResult<usize> {
        let first_dim = self.shape.first().copied().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>("len() of unsized object")
        })?;

        usize::try_from(first_dim).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "shape dtype struct length does not fit usize",
            )
        })
    }

    #[getter]
    fn weak_type(&self) -> bool {
        self.weak_type
    }

    #[getter]
    fn is_ref(&self) -> bool {
        self.is_ref
    }

    #[pyo3(signature = (*, shape=None, dtype=None, sharding=None, weak_type=None, vma=None, is_ref=None))]
    fn update(
        &self,
        shape: Option<Vec<u32>>,
        dtype: Option<String>,
        sharding: Option<Py<PyAny>>,
        weak_type: Option<bool>,
        vma: Option<Py<PyAny>>,
        is_ref: Option<bool>,
    ) -> PyResult<Self> {
        Self::require_sharding_none(sharding)?;
        let (vma, vma_hash, vma_len, vma_repr) = match vma {
            Some(vma) => Self::normalize_vma(Some(vma))?,
            None => (
                self.clone_vma(),
                self.vma_hash,
                self.vma_len,
                self.vma_repr.clone(),
            ),
        };
        Ok(Self {
            shape: shape.unwrap_or_else(|| self.shape.clone()),
            dtype: dtype.unwrap_or_else(|| self.dtype.clone()),
            weak_type: weak_type.unwrap_or(self.weak_type),
            is_ref: is_ref.unwrap_or(self.is_ref),
            vma,
            vma_hash,
            vma_len,
            vma_repr,
        })
    }

    fn __richcmp__(&self, other: PyRef<'_, Self>, op: CompareOp) -> PyResult<bool> {
        let equal = self.same_metadata(&other)?;
        match op {
            CompareOp::Eq => Ok(equal),
            CompareOp::Ne => Ok(!equal),
            _ => Ok(false),
        }
    }

    fn __hash__(&self) -> isize {
        let signed = i64::from_ne_bytes(self.metadata_hash().to_ne_bytes());
        let hash = isize::try_from(signed).unwrap_or_else(|_| {
            if signed.is_negative() {
                isize::MIN
            } else {
                isize::MAX
            }
        });
        if hash == -1 { -2 } else { hash }
    }

    fn __repr__(&self) -> String {
        let weak_type = if self.weak_type {
            ", weak_type=True"
        } else {
            ""
        };
        let vma = if self.vma_len > 0 {
            format!(
                ", vma={}",
                self.vma_repr.as_deref().unwrap_or("frozenset()")
            )
        } else {
            String::new()
        };
        let is_ref = if self.is_ref { ", is_ref=True" } else { "" };
        format!(
            "ShapeDtypeStruct(shape={}, dtype={}{}{}{})",
            format_shape_tuple(&self.shape),
            self.dtype,
            weak_type,
            vma,
            is_ref
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
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

#[pymethods]
impl PyUserContext {
    #[getter]
    fn value(&self) -> &str {
        &self.default_repr
    }

    fn __call__(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<PyNamedScope> {
        let value_repr = py_object_repr(py, Some(value))?;
        Ok(PyNamedScope {
            name: format!("user_context({value_repr})"),
        })
    }

    fn __repr__(&self) -> String {
        format!("PyUserContext(default_value={})", self.default_repr)
    }
}

#[pymethods]
impl PyFloat0DType {
    #[getter]
    fn name(&self) -> &'static str {
        "float0"
    }

    fn __repr__(&self) -> &'static str {
        "float0"
    }

    fn __str__(&self) -> &'static str {
        "float0"
    }
}

fn py_values_to_rust(args: Vec<PyValue>) -> Vec<Value> {
    args.into_iter().map(|pv| pv.inner).collect()
}

fn py_values_from_rust(values: Vec<Value>) -> Vec<PyValue> {
    values.into_iter().map(PyValue::from_value).collect()
}

fn dtype_itemsize(dtype: DType) -> u64 {
    match dtype {
        DType::Bool => 1,
        DType::BF16 | DType::F16 => 2,
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F64 | DType::I64 | DType::U64 | DType::Complex64 => 8,
        DType::Complex128 => 16,
    }
}

fn literal_truth_value(literal: Literal) -> bool {
    match literal {
        Literal::Bool(value) => value,
        Literal::I64(value) => value != 0,
        Literal::U32(value) => value != 0,
        Literal::U64(value) => value != 0,
        Literal::BF16Bits(_) | Literal::F16Bits(_) | Literal::F32Bits(_) | Literal::F64Bits(_) => {
            literal.as_f64().is_some_and(|value| value != 0.0)
        }
        Literal::Complex64Bits(re, im) => f32::from_bits(re) != 0.0 || f32::from_bits(im) != 0.0,
        Literal::Complex128Bits(re, im) => f64::from_bits(re) != 0.0 || f64::from_bits(im) != 0.0,
    }
}

fn py_shape_dtype_from_rust(value: &Value) -> PyShapeDtypeStruct {
    PyShapeDtypeStruct {
        shape: value
            .as_tensor()
            .map_or_else(Vec::new, |tensor| tensor.shape.dims.clone()),
        dtype: format!("{:?}", value.dtype()),
        weak_type: false,
        is_ref: false,
        vma: None,
        vma_hash: 0,
        vma_len: 0,
        vma_repr: None,
    }
}

fn vjp_outputs_and_pullback(
    jaxpr: &Jaxpr,
    args: Vec<PyValue>,
) -> PyResult<(Vec<PyValue>, PyVjpPullback)> {
    let rust_args = py_values_to_rust(args);
    fj_api::jit(jaxpr.clone())
        .call(rust_args.clone())
        .map(|outputs| {
            (
                py_values_from_rust(outputs),
                PyVjpPullback {
                    jaxpr: jaxpr.clone(),
                    primals: rust_args,
                },
            )
        })
        .map_err(runtime_error)
}

fn runtime_error(error: impl ToString) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
}

fn py_object_repr(py: Python<'_>, value: Option<Py<PyAny>>) -> PyResult<String> {
    match value {
        Some(value) => {
            let value = value.bind(py);
            if value.is_none() {
                Ok("None".to_owned())
            } else {
                value.repr()?.extract()
            }
        }
        None => Ok("None".to_owned()),
    }
}

fn literal_to_py_object(py: Python<'_>, literal: Literal) -> PyResult<Py<PyAny>> {
    match literal {
        Literal::I64(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
        Literal::U32(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
        Literal::U64(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
        Literal::Bool(value) => Ok(PyBool::new(py, value).to_owned().into_any().unbind()),
        Literal::BF16Bits(_) | Literal::F16Bits(_) | Literal::F32Bits(_) | Literal::F64Bits(_) => {
            Ok(literal
                .as_f64()
                .expect("float literal should convert to f64")
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
        Literal::Complex64Bits(..) => {
            let (re, im) = literal
                .as_complex64()
                .expect("complex64 literal should convert to parts");
            complex_to_py_object(py, f64::from(re), f64::from(im))
        }
        Literal::Complex128Bits(..) => {
            let (re, im) = literal
                .as_complex128()
                .expect("complex128 literal should convert to parts");
            complex_to_py_object(py, re, im)
        }
    }
}

fn complex_to_py_object(py: Python<'_>, re: f64, im: f64) -> PyResult<Py<PyAny>> {
    let builtins = py.import("builtins")?;
    Ok(builtins.getattr("complex")?.call1((re, im))?.unbind())
}

fn literals_to_py_list(py: Python<'_>, literals: &[Literal]) -> PyResult<Py<PyAny>> {
    let values = literals
        .iter()
        .copied()
        .map(|literal| literal_to_py_object(py, literal))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, values.iter().map(|value| value.bind(py)))?
        .into_any()
        .unbind())
}

fn shape_to_py_tuple(py: Python<'_>, shape: &[u32]) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, shape.iter().copied())?.into_any().unbind())
}

fn format_shape_tuple(shape: &[u32]) -> String {
    use std::fmt::Write as _;

    let mut formatted = String::from("(");
    for (index, dim) in shape.iter().enumerate() {
        if index > 0 {
            formatted.push_str(", ");
        }
        let _ = write!(&mut formatted, "{dim}");
    }
    if shape.len() == 1 {
        formatted.push(',');
    }
    formatted.push(')');
    formatted
}

fn validate_tobytes_order(order: &str) -> PyResult<()> {
    match order {
        "C" | "F" | "A" | "K" => Ok(()),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "order must be one of 'C', 'F', 'A', or 'K' (got {order:?})"
        ))),
    }
}

fn literal_to_native_bytes(literal: Literal, bytes: &mut Vec<u8>) {
    match literal {
        Literal::I64(value) => bytes.extend_from_slice(&value.to_ne_bytes()),
        Literal::U32(value) => bytes.extend_from_slice(&value.to_ne_bytes()),
        Literal::U64(value) => bytes.extend_from_slice(&value.to_ne_bytes()),
        Literal::Bool(value) => bytes.push(u8::from(value)),
        Literal::BF16Bits(bits) | Literal::F16Bits(bits) => {
            bytes.extend_from_slice(&bits.to_ne_bytes());
        }
        Literal::F32Bits(bits) => bytes.extend_from_slice(&bits.to_ne_bytes()),
        Literal::F64Bits(bits) => bytes.extend_from_slice(&bits.to_ne_bytes()),
        Literal::Complex64Bits(re, im) => {
            bytes.extend_from_slice(&re.to_ne_bytes());
            bytes.extend_from_slice(&im.to_ne_bytes());
        }
        Literal::Complex128Bits(re, im) => {
            bytes.extend_from_slice(&re.to_ne_bytes());
            bytes.extend_from_slice(&im.to_ne_bytes());
        }
    }
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

fn validate_enum_value(value: &str, allowed: &[&str], option_name: &str) -> PyResult<()> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported {option_name} value {value:?}; expected one of {}",
            allowed.join(", ")
        )))
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

const PROGRAM_SPECS: &[(&str, ProgramSpec)] = &[
    ("add2", ProgramSpec::Add2),
    ("square", ProgramSpec::Square),
    ("square_plus_linear", ProgramSpec::SquarePlusLinear),
    ("add_one", ProgramSpec::AddOne),
    ("sin_x", ProgramSpec::SinX),
    ("cos_x", ProgramSpec::CosX),
    ("dot3", ProgramSpec::Dot3),
    ("reduce_sum_vec", ProgramSpec::ReduceSumVec),
    ("lax_neg", ProgramSpec::LaxNeg),
    ("lax_abs", ProgramSpec::LaxAbs),
    ("lax_exp", ProgramSpec::LaxExp),
    ("lax_log", ProgramSpec::LaxLog),
    ("lax_sqrt", ProgramSpec::LaxSqrt),
    ("lax_rsqrt", ProgramSpec::LaxRsqrt),
    ("lax_floor", ProgramSpec::LaxFloor),
    ("lax_ceil", ProgramSpec::LaxCeil),
    ("lax_round", ProgramSpec::LaxRound),
    ("lax_tan", ProgramSpec::LaxTan),
    ("lax_asin", ProgramSpec::LaxAsin),
    ("lax_acos", ProgramSpec::LaxAcos),
    ("lax_atan", ProgramSpec::LaxAtan),
    ("lax_sinh", ProgramSpec::LaxSinh),
    ("lax_cosh", ProgramSpec::LaxCosh),
    ("lax_tanh", ProgramSpec::LaxTanh),
    ("lax_asinh", ProgramSpec::LaxAsinh),
    ("lax_acosh", ProgramSpec::LaxAcosh),
    ("lax_atanh", ProgramSpec::LaxAtanh),
    ("lax_expm1", ProgramSpec::LaxExpm1),
    ("lax_log1p", ProgramSpec::LaxLog1p),
    ("lax_sign", ProgramSpec::LaxSign),
    ("lax_square", ProgramSpec::LaxSquare),
    ("lax_reciprocal", ProgramSpec::LaxReciprocal),
    ("lax_logistic", ProgramSpec::LaxLogistic),
    ("lax_erf", ProgramSpec::LaxErf),
    ("lax_erfc", ProgramSpec::LaxErfc),
    ("lax_sub", ProgramSpec::LaxSub),
    ("lax_mul", ProgramSpec::LaxMul),
    ("lax_div", ProgramSpec::LaxDiv),
    ("lax_rem", ProgramSpec::LaxRem),
    ("lax_pow", ProgramSpec::LaxPow),
    ("lax_atan2", ProgramSpec::LaxAtan2),
    ("lax_max", ProgramSpec::LaxMax),
    ("lax_min", ProgramSpec::LaxMin),
    ("lax_eq", ProgramSpec::LaxEq),
    ("lax_ne", ProgramSpec::LaxNe),
    ("lax_lt", ProgramSpec::LaxLt),
    ("lax_le", ProgramSpec::LaxLe),
    ("lax_gt", ProgramSpec::LaxGt),
    ("lax_ge", ProgramSpec::LaxGe),
    ("lax_select", ProgramSpec::LaxSelect),
    ("lax_clamp", ProgramSpec::LaxClamp),
    ("lax_reduce_max", ProgramSpec::LaxReduceMax),
    ("lax_reduce_min", ProgramSpec::LaxReduceMin),
    ("lax_reduce_prod", ProgramSpec::LaxReduceProd),
    ("lax_cbrt", ProgramSpec::LaxCbrt),
    ("lax_lgamma", ProgramSpec::LaxLgamma),
    ("lax_digamma", ProgramSpec::LaxDigamma),
    ("lax_erf_inv", ProgramSpec::LaxErfInv),
    ("lax_is_finite", ProgramSpec::LaxIsFinite),
    ("lax_nextafter", ProgramSpec::LaxNextafter),
    ("lax_cumsum", ProgramSpec::LaxCumsum),
    ("lax_cumprod", ProgramSpec::LaxCumprod),
    ("lax_reduce_and", ProgramSpec::LaxReduceAnd),
    ("lax_reduce_or", ProgramSpec::LaxReduceOr),
    ("lax_bitwise_and", ProgramSpec::LaxBitwiseAnd),
    ("lax_bitwise_or", ProgramSpec::LaxBitwiseOr),
    ("lax_bitwise_xor", ProgramSpec::LaxBitwiseXor),
    ("lax_bitwise_not", ProgramSpec::LaxBitwiseNot),
    ("lax_population_count", ProgramSpec::LaxPopulationCount),
    ("lax_count_leading_zeros", ProgramSpec::LaxCountLeadingZeros),
    ("lax_reduce_xor", ProgramSpec::LaxReduceXor),
    ("lax_sort", ProgramSpec::LaxSort),
    ("lax_integer_pow2", ProgramSpec::LaxIntegerPow2),
    ("lax_integer_pow3", ProgramSpec::LaxIntegerPow3),
    ("lax_integer_pow_neg1", ProgramSpec::LaxIntegerPowNeg1),
    ("lax_reshape_6_to_2x3", ProgramSpec::LaxReshape6To2x3),
    ("lax_reshape_6_to_3x2", ProgramSpec::LaxReshape6To3x2),
    ("lax_slice_1_to_4", ProgramSpec::LaxSlice1To4),
    ("lax_transpose_2x3", ProgramSpec::LaxTranspose2x3),
    ("lax_rev", ProgramSpec::LaxRev),
    ("lax_squeeze", ProgramSpec::LaxSqueeze),
    ("lax_concatenate", ProgramSpec::LaxConcatenate),
    ("lax_iota5", ProgramSpec::LaxIota5),
    ("lax_copy", ProgramSpec::LaxCopy),
    ("lax_expand_dims_axis0", ProgramSpec::LaxExpandDimsAxis0),
    ("lax_pad_low1_high2", ProgramSpec::LaxPadLow1High2),
    (
        "lax_broadcast_in_dim_scalar3",
        ProgramSpec::LaxBroadcastInDimScalar3,
    ),
    ("lax_shift_left", ProgramSpec::LaxShiftLeft),
    (
        "lax_shift_right_arithmetic",
        ProgramSpec::LaxShiftRightArithmetic,
    ),
    ("lax_shift_right_logical", ProgramSpec::LaxShiftRightLogical),
    ("lax_dynamic_slice", ProgramSpec::LaxDynamicSlice),
    (
        "lax_dynamic_update_slice",
        ProgramSpec::LaxDynamicUpdateSlice,
    ),
    ("lax_split2", ProgramSpec::LaxSplit2),
    (
        "lax_broadcasted_iota_2x3",
        ProgramSpec::LaxBroadcastedIota2x3,
    ),
    ("lax_while_add_lt", ProgramSpec::LaxWhileAddLt),
    ("lax_switch3", ProgramSpec::LaxSwitch3),
    ("lax_argsort", ProgramSpec::LaxArgsort),
    ("lax_one_hot4", ProgramSpec::LaxOneHot4),
    ("lax_reduce_window_sum", ProgramSpec::LaxReduceWindowSum),
    ("lax_bitcast_f64_to_i64", ProgramSpec::LaxBitcastF64ToI64),
    (
        "lax_reduce_precision_f64",
        ProgramSpec::LaxReducePrecisionF64,
    ),
    ("lax_gather_1d", ProgramSpec::LaxGather1d),
    ("lax_conv_1d_valid", ProgramSpec::LaxConv1dValid),
    ("lax_scatter_overwrite", ProgramSpec::LaxScatterOverwrite),
    ("lax_complex", ProgramSpec::LaxComplex),
    ("lax_conj", ProgramSpec::LaxConj),
    ("lax_real", ProgramSpec::LaxReal),
    ("lax_imag", ProgramSpec::LaxImag),
    ("lax_cholesky", ProgramSpec::LaxCholesky),
    ("lax_triangular_solve", ProgramSpec::LaxTriangularSolve),
    ("lax_qr", ProgramSpec::LaxQr),
    ("lax_svd", ProgramSpec::LaxSvd),
    ("lax_eigh", ProgramSpec::LaxEigh),
    ("lax_fft", ProgramSpec::LaxFft),
    ("lax_ifft", ProgramSpec::LaxIfft),
    ("lax_rfft", ProgramSpec::LaxRfft),
    ("lax_irfft", ProgramSpec::LaxIrfft),
    ("identity", ProgramSpec::Identity),
    ("add_one_mul_two", ProgramSpec::AddOneMulTwo),
    ("cond_select", ProgramSpec::CondSelect),
    ("scan_add", ProgramSpec::ScanAdd),
];

fn program_spec_from_name(name: &str) -> Option<ProgramSpec> {
    PROGRAM_SPECS
        .iter()
        .find_map(|(spec_name, spec)| (*spec_name == name).then_some(*spec))
}

fn program_spec_names() -> String {
    PROGRAM_SPECS
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ")
}

#[pyfunction]
fn make_jaxpr(name: &str) -> PyResult<PyJaxpr> {
    let Some(spec) = program_spec_from_name(name) else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown ProgramSpec {name:?}; expected one of {}",
            program_spec_names()
        )));
    };

    Ok(PyJaxpr {
        inner: build_program(spec),
    })
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
    vjp_outputs_and_pullback(&jaxpr.inner, args)
}

#[pyfunction]
fn linear_transpose(jaxpr: &PyJaxpr, args: Vec<PyValue>) -> PyVjpPullback {
    PyVjpPullback {
        jaxpr: jaxpr.inner.clone(),
        primals: py_values_to_rust(args),
    }
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
fn fwd_and_bwd(jaxpr: &PyJaxpr) -> (PyForwardPass, PyBackwardPass) {
    (
        PyForwardPass {
            jaxpr: jaxpr.inner.clone(),
        },
        PyBackwardPass,
    )
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
fn named_call(function: Py<PyAny>, name: Option<String>) -> Py<PyAny> {
    drop(name);
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
fn debug_key_reuse(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("debug_key_reuse({enabled})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn enable_x64(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("enable_x64({enabled})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn enable_custom_prng(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("enable_custom_prng({enabled})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn softmax_custom_jvp(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("softmax_custom_jvp({enabled})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn enable_custom_vjp_by_custom_transpose(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("enable_custom_vjp_by_custom_transpose({enabled})"),
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

#[pyfunction(signature = (enabled=true))]
fn no_tracing(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("no_tracing({enabled})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn no_execution(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("no_execution({enabled})"),
    }
}

#[pyfunction(signature = (precision=None))]
fn default_matmul_precision(precision: Option<String>) -> PyResult<PyNamedScope> {
    const ALLOWED: &[&str] = &[
        "default",
        "high",
        "highest",
        "bfloat16",
        "tensorfloat32",
        "float32",
        "ANY_F8_ANY_F8_F32",
        "ANY_F8_ANY_F8_F32_FAST_ACCUM",
        "ANY_F8_ANY_F8_ANY",
        "ANY_F8_ANY_F8_ANY_FAST_ACCUM",
        "F16_F16_F16",
        "F16_F16_F32",
        "BF16_BF16_BF16",
        "BF16_BF16_F32",
        "BF16_BF16_F32_X3",
        "BF16_BF16_F32_X6",
        "BF16_BF16_F32_X9",
        "TF32_TF32_F32",
        "TF32_TF32_F32_X3",
        "F32_F32_F32",
        "F64_F64_F64",
    ];

    match precision {
        Some(precision) => {
            validate_enum_value(&precision, ALLOWED, "default_matmul_precision")?;
            Ok(PyNamedScope {
                name: format!("default_matmul_precision({precision:?})"),
            })
        }
        None => Ok(PyNamedScope {
            name: "default_matmul_precision(None)".to_owned(),
        }),
    }
}

#[pyfunction(signature = (impl_name))]
fn default_prng_impl(impl_name: String) -> PyResult<PyNamedScope> {
    const ALLOWED: &[&str] = &["threefry2x32", "rbg", "unsafe_rbg"];
    validate_enum_value(&impl_name, ALLOWED, "default_prng_impl")?;
    Ok(PyNamedScope {
        name: format!("default_prng_impl({impl_name:?})"),
    })
}

#[pyfunction(signature = (promotion))]
fn numpy_dtype_promotion(promotion: String) -> PyResult<PyNamedScope> {
    const ALLOWED: &[&str] = &["standard", "strict"];
    validate_enum_value(&promotion, ALLOWED, "numpy_dtype_promotion")?;
    Ok(PyNamedScope {
        name: format!("numpy_dtype_promotion({promotion:?})"),
    })
}

#[pyfunction(signature = (promotion))]
fn numpy_rank_promotion(promotion: String) -> PyResult<PyNamedScope> {
    const ALLOWED: &[&str] = &["allow", "warn", "raise"];
    validate_enum_value(&promotion, ALLOWED, "numpy_rank_promotion")?;
    Ok(PyNamedScope {
        name: format!("numpy_rank_promotion({promotion:?})"),
    })
}

#[pyfunction(signature = (enabled=true))]
fn allow_f16_reductions(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("allow_f16_reductions({enabled})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn jax2tf_associative_scan_reductions(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("jax2tf_associative_scan_reductions({enabled})"),
    }
}

#[pyfunction(signature = (mode))]
fn legacy_prng_key(mode: String) -> PyResult<PyNamedScope> {
    const ALLOWED: &[&str] = &["allow", "warn", "error"];
    validate_enum_value(&mode, ALLOWED, "legacy_prng_key")?;
    Ok(PyNamedScope {
        name: format!("legacy_prng_key({mode:?})"),
    })
}

#[pyfunction(signature = (enabled=true))]
fn threefry_partitionable(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("threefry_partitionable({enabled})"),
    }
}

#[pyfunction(signature = (level=None))]
fn array_garbage_collection_guard(level: Option<String>) -> PyResult<PyNamedScope> {
    const ALLOWED: &[&str] = &["allow", "log", "fatal"];

    match level {
        Some(level) => {
            validate_enum_value(&level, ALLOWED, "array_garbage_collection_guard")?;
            Ok(PyNamedScope {
                name: format!("array_garbage_collection_guard({level:?})"),
            })
        }
        None => Ok(PyNamedScope {
            name: "array_garbage_collection_guard(None)".to_owned(),
        }),
    }
}

#[pyfunction(signature = (enabled=true))]
fn remove_size_one_mesh_axis_from_type(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("remove_size_one_mesh_axis_from_type({enabled})"),
    }
}

#[pyfunction(signature = (enabled=true))]
fn thread_guard(enabled: bool) -> PyNamedScope {
    PyNamedScope {
        name: format!("thread_guard({enabled})"),
    }
}

#[pyfunction(signature = (default_value=None))]
fn make_user_context(py: Python<'_>, default_value: Option<Py<PyAny>>) -> PyResult<PyUserContext> {
    Ok(PyUserContext {
        default_repr: py_object_repr(py, default_value)?,
    })
}

fn transfer_guard_scope(name: &str, value: String) -> PyResult<PyNamedScope> {
    const ALLOWED: &[&str] = &[
        "allow",
        "log",
        "disallow",
        "log_explicit",
        "disallow_explicit",
    ];
    validate_enum_value(&value, ALLOWED, name)?;
    Ok(PyNamedScope {
        name: format!("{name}({value:?})"),
    })
}

#[pyfunction(signature = (new_val))]
fn transfer_guard(new_val: String) -> PyResult<PyNamedScope> {
    transfer_guard_scope("transfer_guard", new_val)
}

#[pyfunction(signature = (new_val))]
fn transfer_guard_host_to_device(new_val: String) -> PyResult<PyNamedScope> {
    transfer_guard_scope("transfer_guard_host_to_device", new_val)
}

#[pyfunction(signature = (new_val))]
fn transfer_guard_device_to_device(new_val: String) -> PyResult<PyNamedScope> {
    transfer_guard_scope("transfer_guard_device_to_device", new_val)
}

#[pyfunction(signature = (new_val))]
fn transfer_guard_device_to_host(new_val: String) -> PyResult<PyNamedScope> {
    transfer_guard_scope("transfer_guard_device_to_host", new_val)
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
        .map(PyValue::from_value)
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
        .map(PyValue::from_value)
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
    m.add_class::<PyForwardPass>()?;
    m.add_class::<PyBackwardPass>()?;
    m.add_class::<PyShapeDtypeStruct>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyNamedScope>()?;
    m.add_class::<PyUserContext>()?;
    m.add_class::<PyFloat0DType>()?;
    m.add("float0", Py::new(m.py(), PyFloat0DType)?)?;
    m.add_function(wrap_pyfunction!(make_jaxpr, m)?)?;
    m.add_function(wrap_pyfunction!(make_jaxpr_square, m)?)?;
    m.add_function(wrap_pyfunction!(make_jaxpr_add2, m)?)?;
    m.add_function(wrap_pyfunction!(make_jaxpr_add_one, m)?)?;
    m.add_function(wrap_pyfunction!(jit, m)?)?;
    m.add_function(wrap_pyfunction!(grad, m)?)?;
    m.add_function(wrap_pyfunction!(jvp, m)?)?;
    m.add_function(wrap_pyfunction!(vjp, m)?)?;
    m.add_function(wrap_pyfunction!(linear_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(linearize, m)?)?;
    m.add_function(wrap_pyfunction!(fwd_and_bwd, m)?)?;
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
    m.add_function(wrap_pyfunction!(debug_key_reuse, m)?)?;
    m.add_function(wrap_pyfunction!(enable_x64, m)?)?;
    m.add_function(wrap_pyfunction!(enable_custom_prng, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_custom_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(enable_custom_vjp_by_custom_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(check_tracer_leaks, m)?)?;
    m.add_function(wrap_pyfunction!(checking_leaks, m)?)?;
    m.add_function(wrap_pyfunction!(debug_nans, m)?)?;
    m.add_function(wrap_pyfunction!(debug_infs, m)?)?;
    m.add_function(wrap_pyfunction!(log_compiles, m)?)?;
    m.add_function(wrap_pyfunction!(explain_cache_misses, m)?)?;
    m.add_function(wrap_pyfunction!(no_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(no_execution, m)?)?;
    m.add_function(wrap_pyfunction!(default_matmul_precision, m)?)?;
    m.add_function(wrap_pyfunction!(default_prng_impl, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_dtype_promotion, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_rank_promotion, m)?)?;
    m.add_function(wrap_pyfunction!(allow_f16_reductions, m)?)?;
    m.add_function(wrap_pyfunction!(jax2tf_associative_scan_reductions, m)?)?;
    m.add_function(wrap_pyfunction!(legacy_prng_key, m)?)?;
    m.add_function(wrap_pyfunction!(threefry_partitionable, m)?)?;
    m.add_function(wrap_pyfunction!(array_garbage_collection_guard, m)?)?;
    m.add_function(wrap_pyfunction!(remove_size_one_mesh_axis_from_type, m)?)?;
    m.add_function(wrap_pyfunction!(thread_guard, m)?)?;
    m.add_function(wrap_pyfunction!(make_user_context, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_guard, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_guard_host_to_device, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_guard_device_to_device, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_guard_device_to_host, m)?)?;
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
        assert_eq!(v.shape_dims(), Vec::<u32>::new());
        assert_eq!(v.dtype(), "F64");
        assert_eq!(v.ndim(), 0);
        assert_eq!(v.size(), 1);
        assert_eq!(v.itemsize(), 8);
        assert_eq!(v.nbytes(), 8);
        assert!(!v.weak_type());
        assert!(!v.committed());
        let device = v.device();
        assert_eq!(device.id(), 0);
        assert_eq!(device.process_index(), 0);
        assert_eq!(device.platform(), "cpu");
        assert_eq!(v.on_device_size_in_bytes(), v.nbytes());
        assert!(v.is_fully_addressable());
        assert!(v.is_fully_replicated());
        assert!(v.__len__().is_err());
        assert!((v.block_until_ready().as_f64().unwrap() - 42.0).abs() < 1e-12);
        assert!((v.copy_to_host_async().as_f64().unwrap() - 42.0).abs() < 1e-12);
        assert!((v.copy().as_f64().unwrap() - 42.0).abs() < 1e-12);
        let mut deleted = v.copy();
        assert!(!deleted.is_deleted());
        deleted.delete();
        assert!(deleted.is_deleted());
        assert!(!v.is_deleted());
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let devices = v.devices(py).unwrap();
            assert_eq!(devices.bind(py).len().unwrap(), 1);
            let value = v.tolist(py).unwrap();
            assert!((value.bind(py).extract::<f64>().unwrap() - 42.0).abs() < 1e-12);
            let value = v.__int__(py).unwrap();
            assert_eq!(value.bind(py).extract::<i64>().unwrap(), 42);
            let value = v.tobytes(py, "C").unwrap();
            assert_eq!(
                value.bind(py).downcast::<PyBytes>().unwrap().as_bytes(),
                &42.0_f64.to_ne_bytes()
            );
            let value = v.__complex__(py).unwrap();
            let value = value.bind(py);
            assert!(
                (value.getattr("real").unwrap().extract::<f64>().unwrap() - 42.0).abs() < 1e-12
            );
            assert_eq!(
                value.getattr("imag").unwrap().extract::<f64>().unwrap(),
                0.0
            );
            assert!(v.__index__(py).is_err());
        });
        assert!((v.__float__().unwrap() - 42.0).abs() < 1e-12);
        assert!(v.__bool__().unwrap());
        assert!((v.as_f64().unwrap() - 42.0).abs() < 1e-12);

        let i = PyValue::scalar_i64(123);
        assert!(i.__bool__().unwrap());
        assert!(!PyValue::scalar_i64(0).__bool__().unwrap());
        Python::with_gil(|py| {
            let value = i.__index__(py).unwrap();
            assert_eq!(value.bind(py).extract::<i64>().unwrap(), 123);
            assert_eq!(i.__hex__(py).unwrap(), "0x7b");
            assert_eq!(i.__oct__(py).unwrap(), "0o173");
            let value = i.tobytes(py, "K").unwrap();
            assert_eq!(
                value.bind(py).downcast::<PyBytes>().unwrap().as_bytes(),
                &123_i64.to_ne_bytes()
            );
            assert!(i.tobytes(py, "bad").is_err());
            assert!(v.__hex__(py).is_err());
            assert!(v.__oct__(py).is_err());
        });
    }

    #[test]
    fn value_vector_roundtrip() {
        let floats = PyValue::vector_f64(vec![1.0, 2.5, 4.0]).unwrap();
        assert_eq!(floats.shape_dims(), vec![3]);
        assert_eq!(floats.dtype(), "F64");
        assert_eq!(floats.ndim(), 1);
        assert_eq!(floats.size(), 3);
        assert_eq!(floats.itemsize(), 8);
        assert_eq!(floats.nbytes(), 24);
        assert!(!floats.weak_type());
        assert!(!floats.committed());
        assert_eq!(floats.device().platform(), "cpu");
        assert!(floats.is_fully_addressable());
        assert!(floats.is_fully_replicated());
        assert_eq!(floats.__len__().unwrap(), 3);
        assert_eq!(
            floats.block_until_ready().as_f64_list().unwrap(),
            vec![1.0, 2.5, 4.0]
        );
        assert_eq!(
            floats.copy_to_host_async().as_f64_list().unwrap(),
            vec![1.0, 2.5, 4.0]
        );
        assert_eq!(floats.copy().as_f64_list().unwrap(), vec![1.0, 2.5, 4.0]);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let values = floats.tolist(py).unwrap();
            assert_eq!(
                values.bind(py).extract::<Vec<f64>>().unwrap(),
                vec![1.0, 2.5, 4.0]
            );
            let bytes = floats.tobytes(py, "A").unwrap();
            let mut expected = Vec::new();
            for value in [1.0_f64, 2.5, 4.0] {
                expected.extend_from_slice(&value.to_ne_bytes());
            }
            assert_eq!(
                bytes.bind(py).downcast::<PyBytes>().unwrap().as_bytes(),
                expected
            );
        });
        assert_eq!(floats.as_f64_list().unwrap(), vec![1.0, 2.5, 4.0]);
        assert_eq!(floats.as_i64_list(), None);

        let ints = PyValue::vector_i64(vec![1, 2, 3]).unwrap();
        assert_eq!(ints.shape_dims(), vec![3]);
        assert_eq!(ints.dtype(), "I64");
        assert_eq!(ints.ndim(), 1);
        assert_eq!(ints.size(), 3);
        assert_eq!(ints.itemsize(), 8);
        assert_eq!(ints.nbytes(), 24);
        assert!(!ints.weak_type());
        assert!(!ints.committed());
        assert!(ints.is_fully_addressable());
        assert!(ints.is_fully_replicated());
        assert_eq!(ints.__len__().unwrap(), 3);
        assert!(ints.__float__().is_err());
        assert!(ints.__bool__().is_err());
        Python::with_gil(|py| {
            assert!(ints.__int__(py).is_err());
            assert!(ints.__complex__(py).is_err());
            assert!(ints.__index__(py).is_err());
            let values = ints.tolist(py).unwrap();
            assert_eq!(
                values.bind(py).extract::<Vec<i64>>().unwrap(),
                vec![1, 2, 3]
            );
            let bytes = ints.tobytes(py, "F").unwrap();
            let mut expected = Vec::new();
            for value in [1_i64, 2, 3] {
                expected.extend_from_slice(&value.to_ne_bytes());
            }
            assert_eq!(
                bytes.bind(py).downcast::<PyBytes>().unwrap().as_bytes(),
                expected
            );
        });
        assert_eq!(ints.as_i64_list().unwrap(), vec![1, 2, 3]);
        assert_eq!(ints.as_f64_list().unwrap(), vec![1.0, 2.0, 3.0]);

        let one = PyValue::vector_i64(vec![0]).unwrap();
        assert!(!one.__bool__().unwrap());
        let empty = PyValue::vector_i64(Vec::new()).unwrap();
        assert!(empty.__bool__().is_err());
    }

    #[test]
    fn jaxpr_square_builds() {
        let jaxpr = make_jaxpr_square();
        assert!(!jaxpr.inner.equations.is_empty());
    }

    #[test]
    fn make_jaxpr_dispatches_program_specs_by_name() {
        let jaxpr = make_jaxpr("square").unwrap();
        let values = jit(&jaxpr, vec![PyValue::scalar_f64(3.0)]).unwrap();
        assert_eq!(values.len(), 1);
        assert!((values[0].as_f64().unwrap() - 9.0).abs() < 1e-12);

        let add_jaxpr = make_jaxpr("add2").unwrap();
        let values = jit(
            &add_jaxpr,
            vec![PyValue::scalar_i64(3), PyValue::scalar_i64(4)],
        )
        .unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].as_i64().unwrap(), 7);

        let result = make_jaxpr("missing_program");
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(err.to_string().contains("unknown ProgramSpec"));
            assert!(err.to_string().contains("square"));
        }
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
    fn linear_transpose_returns_callable_pullback() {
        let jaxpr = make_jaxpr_square();
        let pullback = linear_transpose(&jaxpr, vec![PyValue::scalar_f64(3.0)]);

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
    fn fwd_and_bwd_returns_reusable_forward_backward_pair() {
        let jaxpr = make_jaxpr_square();
        let (forward, backward) = fwd_and_bwd(&jaxpr);

        let (values, residuals) = forward.call(vec![PyValue::scalar_f64(3.0)]).unwrap();
        assert_eq!(values.len(), 1);
        assert!((values[0].as_f64().unwrap() - 9.0).abs() < 1e-12);

        let grads = backward
            .call(&residuals, vec![PyValue::scalar_f64(1.0)])
            .unwrap();
        assert_eq!(grads.len(), 1);
        assert!((grads[0].as_f64().unwrap() - 6.0).abs() < 1e-9);

        let scaled_grads = backward
            .call(&residuals, vec![PyValue::scalar_f64(2.0)])
            .unwrap();
        assert_eq!(scaled_grads.len(), 1);
        assert!((scaled_grads[0].as_f64().unwrap() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn eval_shape_returns_output_shape_dtype_metadata() {
        let scalar_meta = eval_shape(&make_jaxpr_square(), vec![PyValue::scalar_f64(3.0)]).unwrap();
        assert_eq!(scalar_meta.len(), 1);
        assert_eq!(scalar_meta[0].shape_vec(), Vec::<u32>::new());
        assert_eq!(scalar_meta[0].dtype(), "F64");

        let vector_meta = eval_shape(
            &make_jaxpr_add_one(),
            vec![PyValue::vector_f64(vec![1.0, 2.0, 3.0]).unwrap()],
        )
        .unwrap();
        assert_eq!(vector_meta.len(), 1);
        assert_eq!(vector_meta[0].shape_vec(), vec![3]);
        assert_eq!(vector_meta[0].dtype(), "F64");
    }

    #[test]
    fn shape_dtype_struct_constructor_roundtrips_metadata() {
        let meta =
            PyShapeDtypeStruct::new(vec![2, 3], Some("F64".to_owned()), None, false, None, false)
                .unwrap();
        assert_eq!(meta.shape_vec(), vec![2, 3]);
        assert_eq!(meta.dtype(), "F64");
        assert!(meta.sharding().is_none());
        assert!(meta.vma().is_none());
        assert_eq!(meta.ndim(), 2);
        assert_eq!(meta.size(), 6);
        assert_eq!(meta.__len__().unwrap(), 2);
        assert!(!meta.weak_type());
        assert!(!meta.is_ref());
        assert_eq!(meta.__repr__(), "ShapeDtypeStruct(shape=(2, 3), dtype=F64)");
        assert_eq!(meta.__str__(), meta.__repr__());

        let weak_meta =
            PyShapeDtypeStruct::new(vec![], Some("F64".to_owned()), None, true, None, true)
                .unwrap();
        assert!(weak_meta.__len__().is_err());
        assert!(weak_meta.sharding().is_none());
        assert!(weak_meta.vma().is_none());
        assert!(weak_meta.weak_type());
        assert!(weak_meta.is_ref());
        assert_eq!(
            weak_meta.__repr__(),
            "ShapeDtypeStruct(shape=(), dtype=F64, weak_type=True, is_ref=True)"
        );
        assert_eq!(weak_meta.__str__(), weak_meta.__repr__());

        let updated = meta
            .update(
                Some(vec![4]),
                Some("I64".to_owned()),
                None,
                Some(true),
                None,
                Some(true),
            )
            .unwrap();
        assert_eq!(updated.shape_vec(), vec![4]);
        assert_eq!(updated.dtype(), "I64");
        assert!(updated.sharding().is_none());
        assert!(updated.vma().is_none());
        assert_eq!(updated.__len__().unwrap(), 4);
        assert!(updated.weak_type());
        assert!(updated.is_ref());
        assert_eq!(
            updated.__repr__(),
            "ShapeDtypeStruct(shape=(4,), dtype=I64, weak_type=True, is_ref=True)"
        );
        assert_eq!(meta.shape_vec(), vec![2, 3]);
        assert!(!meta.weak_type());
        assert!(!meta.is_ref());

        let same_meta =
            PyShapeDtypeStruct::new(vec![2, 3], Some("F64".to_owned()), None, false, None, false)
                .unwrap();
        assert!(meta.same_metadata(&same_meta).unwrap());
        assert_eq!(meta.__hash__(), same_meta.__hash__());
        assert!(!meta.same_metadata(&updated).unwrap());
        assert_ne!(meta.__hash__(), updated.__hash__());

        Python::with_gil(|py| {
            let input_vma = PySet::new(py, ["data"]).unwrap().into_any().unbind();
            let vma_meta = PyShapeDtypeStruct::new(
                vec![2],
                Some("F64".to_owned()),
                None,
                false,
                Some(input_vma),
                false,
            )
            .unwrap();
            let vma = vma_meta.vma().unwrap();
            let vma = vma.bind(py);
            assert!(vma.is_instance_of::<PyFrozenSet>());
            assert!(vma.contains("data").unwrap());
            assert!(vma_meta.__repr__().contains("vma=frozenset("));

            let same_vma = PyFrozenSet::new(py, ["data"]).unwrap().into_any().unbind();
            let same_vma_meta = PyShapeDtypeStruct::new(
                vec![2],
                Some("F64".to_owned()),
                None,
                false,
                Some(same_vma),
                false,
            )
            .unwrap();
            assert!(vma_meta.same_metadata(&same_vma_meta).unwrap());
            assert_eq!(vma_meta.__hash__(), same_vma_meta.__hash__());

            let updated_vma = PySet::new(py, ["batch"]).unwrap().into_any().unbind();
            let updated_vma_meta = vma_meta
                .update(None, None, None, None, Some(updated_vma), None)
                .unwrap();
            let updated_vma = updated_vma_meta.vma().unwrap();
            let updated_vma = updated_vma.bind(py);
            assert!(updated_vma.contains("batch").unwrap());
            assert!(!updated_vma.contains("data").unwrap());

            let invalid_vma = PyList::empty(py).into_any().unbind();
            let invalid_result = PyShapeDtypeStruct::new(
                vec![2],
                Some("F64".to_owned()),
                None,
                false,
                Some(invalid_vma),
                false,
            );
            assert!(invalid_result.as_ref().err().is_some_and(|error| {
                error
                    .to_string()
                    .contains("should be of type `set` or `frozenset`")
            }));
        });

        let missing_dtype =
            PyShapeDtypeStruct::new(vec![2], Option::<String>::None, None, false, None, false);
        assert!(missing_dtype.as_ref().err().is_some_and(|error| {
            error
                .to_string()
                .contains("ShapeDtypeStruct: dtype must be specified.")
        }));
    }

    #[test]
    fn typeof_returns_value_shape_dtype_metadata() {
        let scalar_meta = typeof_value(&PyValue::scalar_i64(7));
        assert_eq!(scalar_meta.shape_vec(), Vec::<u32>::new());
        assert_eq!(scalar_meta.dtype(), "I64");

        let vector = PyValue::vector_f64(vec![1.0, 2.0, 3.0]).unwrap();
        let vector_meta = typeof_value(&vector);
        assert_eq!(vector_meta.shape_vec(), vec![3]);
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
        assert_eq!(host_vector.shape_dims(), vec![3]);
        assert_eq!(host_vector.dtype(), "I64");
        assert_eq!(host_vector.as_i64_list().unwrap(), vec![1, 2, 3]);

        let copied_vector = copy_to_host_async(host_vector);
        assert_eq!(copied_vector.shape_dims(), vec![3]);
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
    fn float0_sentinel_reports_stable_metadata() {
        let dtype = PyFloat0DType;
        assert_eq!(dtype.name(), "float0");
        assert_eq!(dtype.__repr__(), "float0");
        assert_eq!(dtype.__str__(), "float0");
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
        assert_eq!(jac.shape_dims(), vec![1, 1]);
        assert!((jac.as_f64_list().unwrap()[0] - 6.0).abs() < 1e-9);

        let jac_rev = jacrev(&jaxpr, args.clone()).unwrap();
        assert_eq!(jac_rev.shape_dims(), vec![1, 1]);
        assert!((jac_rev.as_f64_list().unwrap()[0] - 6.0).abs() < 1e-9);

        let jac_fwd = jacfwd(&jaxpr, args.clone()).unwrap();
        assert_eq!(jac_fwd.shape_dims(), vec![1, 1]);
        assert!((jac_fwd.as_f64_list().unwrap()[0] - 6.0).abs() < 1e-9);

        let hess = hessian(&jaxpr, args).unwrap();
        assert_eq!(hess.shape_dims(), vec![1, 1]);
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
