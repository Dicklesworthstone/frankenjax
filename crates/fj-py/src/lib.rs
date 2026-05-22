#![forbid(unsafe_code)]

use pyo3::class::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFrozenSet, PyList, PySet, PySlice, PySliceMethods, PyTuple,
};

use fj_core::{DType, Jaxpr, Literal, ProgramSpec, Shape, TensorValue, Value, build_program};

const ARRAY_API_VERSION: &str = "2024.12";

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

    fn ensure_not_deleted(&self) -> PyResult<()> {
        if self.deleted {
            Err(runtime_error("Array has been deleted."))
        } else {
            Ok(())
        }
    }

    fn shape_dims(&self) -> Vec<u32> {
        self.inner
            .as_tensor()
            .map_or_else(Vec::new, |tensor| tensor.shape.dims.clone())
    }

    fn leading_axis_values(&self) -> PyResult<Vec<Self>> {
        let Value::Tensor(tensor) = &self.inner else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "iteration over a 0-d array",
            ));
        };

        let first_dim = tensor.leading_dim().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>("iteration over a 0-d array")
        })?;
        let capacity = usize::try_from(first_dim).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array length does not fit usize")
        })?;

        let mut values = Vec::with_capacity(capacity);
        for index in 0..capacity {
            let value = tensor.slice_axis0(index).map_err(value_error)?;
            values.push(Self::from_value(value));
        }
        Ok(values)
    }

    fn axis0_value_at(&self, index: isize) -> PyResult<Self> {
        let Value::Tensor(tensor) = &self.inner else {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Too many indices: array is 0-dimensional, but 1 were indexed",
            ));
        };

        let first_dim = tensor.leading_dim().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Too many indices: array is 0-dimensional, but 1 were indexed",
            )
        })?;
        let axis_size = isize::try_from(first_dim).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array length does not fit isize")
        })?;
        let normalized_index = if index < 0 {
            index.checked_add(axis_size).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "index {index} is out of bounds for axis 0 with size {axis_size}"
                ))
            })?
        } else {
            index
        };
        if !(0..axis_size).contains(&normalized_index) {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "index {index} is out of bounds for axis 0 with size {axis_size}"
            )));
        }

        let index = usize::try_from(normalized_index).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array index does not fit usize")
        })?;
        let value = tensor.slice_axis0(index).map_err(value_error)?;
        Ok(Self::from_value(value))
    }

    fn axis0_slice(&self, slice: &Bound<'_, PySlice>) -> PyResult<Self> {
        let Value::Tensor(tensor) = &self.inner else {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Too many indices: array is 0-dimensional, but 1 were indexed",
            ));
        };

        let first_dim = tensor.leading_dim().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Too many indices: array is 0-dimensional, but 1 were indexed",
            )
        })?;
        let axis_size = isize::try_from(first_dim).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array length does not fit isize")
        })?;
        let indices = slice.indices(axis_size)?;
        if indices.slicelength == 0 {
            let mut dims = tensor.shape.dims.clone();
            let first_dim = dims.first_mut().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Too many indices: array is 0-dimensional, but 1 were indexed",
                )
            })?;
            *first_dim = 0;
            let tensor =
                TensorValue::new(tensor.dtype, Shape { dims }, Vec::new()).map_err(value_error)?;
            return Ok(Self::from_value(Value::Tensor(tensor)));
        }

        let mut values = Vec::with_capacity(indices.slicelength);
        let mut current = indices.start;
        for position in 0..indices.slicelength {
            if !(0..axis_size).contains(&current) {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "normalized slice index is out of bounds",
                ));
            }
            let index = usize::try_from(current).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyOverflowError, _>("slice index does not fit usize")
            })?;
            values.push(tensor.slice_axis0(index).map_err(value_error)?);
            if position + 1 < indices.slicelength {
                current = current.checked_add(indices.step).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                        "slice index progression overflowed",
                    )
                })?;
            }
        }

        let tensor = TensorValue::stack_axis0(&values).map_err(value_error)?;
        Ok(Self::from_value(Value::Tensor(tensor)))
    }

    fn transpose_all_axes(&self) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        let Value::Tensor(tensor) = &self.inner else {
            return Ok(self.clone());
        };
        let rank = tensor.rank();
        let permutation = default_transpose_permutation(rank);
        self.transpose_with_permutation(&permutation)
    }

    fn matrix_transpose(&self) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        let Some(tensor) = self.inner.as_tensor() else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x must be at least two-dimensional for matrix_transpose; got ndim=0",
            ));
        };
        let rank = tensor.rank();
        if rank < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "x must be at least two-dimensional for matrix_transpose; got ndim={rank}"
            )));
        }

        let mut permutation = (0..rank).collect::<Vec<_>>();
        permutation.swap(rank - 2, rank - 1);
        self.transpose_with_permutation(&permutation)
    }

    fn real_part(&self) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        value_real_part(&self.inner).map(Self::from_value)
    }

    fn imag_part(&self) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        value_imag_part(&self.inner).map(Self::from_value)
    }

    fn transpose_with_permutation(&self, permutation: &[usize]) -> PyResult<Self> {
        let Value::Tensor(tensor) = &self.inner else {
            if permutation.is_empty() {
                return Ok(self.clone());
            }
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "transpose permutation rank does not match scalar array rank",
            ));
        };
        let rank = tensor.rank();
        if permutation.len() != rank {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "transpose permutation rank does not match array rank",
            ));
        }
        validate_permutation(permutation, rank)?;

        if rank <= 1 {
            return Ok(self.clone());
        }

        let mut dims = Vec::with_capacity(rank);
        for axis in permutation {
            let dim = tensor.shape.dims.get(*axis).copied().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "transpose dimension index is out of bounds",
                )
            })?;
            dims.push(dim);
        }
        if tensor.elements.is_empty() {
            let tensor =
                TensorValue::new(tensor.dtype, Shape { dims }, Vec::new()).map_err(value_error)?;
            return Ok(Self::from_value(Value::Tensor(tensor)));
        }

        let old_strides = row_major_strides(&tensor.shape.dims)?;
        let new_strides = row_major_strides(&dims)?;
        let mut elements = Vec::with_capacity(tensor.elements.len());
        for new_flat_index in 0..tensor.elements.len() {
            let mut remainder = new_flat_index;
            let mut old_flat_index = 0_usize;
            for (axis, new_stride) in new_strides.iter().copied().enumerate() {
                if new_stride == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                        "transpose stride computation produced zero stride",
                    ));
                }
                let coord = remainder / new_stride;
                remainder %= new_stride;
                let old_axis = permutation.get(axis).copied().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        "transpose permutation index is out of bounds",
                    )
                })?;
                let old_stride = old_strides.get(old_axis).copied().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        "transpose stride index is out of bounds",
                    )
                })?;
                let offset = coord.checked_mul(old_stride).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                        "transpose index computation overflowed",
                    )
                })?;
                old_flat_index = old_flat_index.checked_add(offset).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                        "transpose index computation overflowed",
                    )
                })?;
            }
            let literal = tensor
                .elements
                .get(old_flat_index)
                .copied()
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        "transpose index is out of bounds",
                    )
                })?;
            elements.push(literal);
        }

        let tensor =
            TensorValue::new(tensor.dtype, Shape { dims }, elements).map_err(value_error)?;
        Ok(Self::from_value(Value::Tensor(tensor)))
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

    fn update_from_kwargs(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let shape = match py_dict_get_item(kwargs, "shape")? {
            Some(value) => value.extract::<Vec<u32>>()?,
            None => self.shape.clone(),
        };
        let dtype = match py_dict_get_item(kwargs, "dtype")? {
            Some(value) => {
                if value.is_none() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "ShapeDtypeStruct: dtype must be specified.",
                    ));
                }
                value.extract::<String>()?
            }
            None => self.dtype.clone(),
        };
        if let Some(sharding) = py_dict_get_item(kwargs, "sharding")?
            && !sharding.is_none()
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ShapeDtypeStruct sharding is not supported by fj-py yet",
            ));
        }
        let weak_type = match py_dict_get_item(kwargs, "weak_type")? {
            Some(value) if !value.is_none() => value.extract::<bool>()?,
            _ => self.weak_type,
        };
        let is_ref = match py_dict_get_item(kwargs, "is_ref")? {
            Some(value) if !value.is_none() => value.extract::<bool>()?,
            _ => self.is_ref,
        };
        let (vma, vma_hash, vma_len, vma_repr) = match py_dict_get_item(kwargs, "vma")? {
            Some(value) if value.is_none() => (None, 0, 0, None),
            Some(value) => Self::normalize_vma(Some(value.unbind()))?,
            None => (
                self.clone_vma(),
                self.vma_hash,
                self.vma_len,
                self.vma_repr.clone(),
            ),
        };

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

#[pyclass(name = "Shard")]
#[derive(Clone)]
struct PyShard {
    device: PyDevice,
    rank: usize,
    data: Option<PyValue>,
}

impl PyShard {
    fn from_array(value: &PyValue) -> Self {
        Self {
            device: cpu_device(),
            rank: value.shape_dims().len(),
            data: Some(value.clone()),
        }
    }
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

    #[getter(T)]
    fn transpose_property(&self) -> PyResult<Self> {
        self.transpose_all_axes()
    }

    #[getter(mT)]
    fn matrix_transpose_property(&self) -> PyResult<Self> {
        self.matrix_transpose()
    }

    #[pyo3(signature = (*args))]
    fn transpose(&self, args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        let rank = self.inner.as_tensor().map_or(0, |tensor| tensor.rank());
        let permutation = transpose_permutation_from_py_args(args, rank)?;
        self.transpose_with_permutation(&permutation)
    }

    fn swapaxes(&self, axis1: isize, axis2: isize) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        let rank = self.inner.as_tensor().map_or(0, |tensor| tensor.rank());
        let axis1 = normalize_axis(axis1, rank)?;
        let axis2 = normalize_axis(axis2, rank)?;
        let mut permutation = (0..rank).collect::<Vec<_>>();
        permutation.swap(axis1, axis2);
        self.transpose_with_permutation(&permutation)
    }

    #[getter]
    fn real(&self) -> PyResult<Self> {
        self.real_part()
    }

    #[getter]
    fn imag(&self) -> PyResult<Self> {
        self.imag_part()
    }

    fn conj(&self) -> PyResult<Self> {
        self.conjugate()
    }

    fn conjugate(&self) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        value_conjugate(&self.inner).map(Self::from_value)
    }

    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype())
    }

    #[getter]
    fn aval(&self) -> PyShapeDtypeStruct {
        py_shape_dtype_from_rust(&self.inner)
    }

    #[getter]
    fn __numpy_dtype__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        numpy_dtype_object(py, self.inner.dtype())
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

    fn platform(&self) -> &'static str {
        "cpu"
    }

    fn devices(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.ensure_not_deleted()?;
        let devices = PySet::empty(py)?;
        devices.add(cpu_device())?;
        Ok(devices.into_any().unbind())
    }

    #[pyo3(signature = (device, *, stream=None))]
    fn to_device(&self, device: PyRef<'_, PyDevice>, stream: Option<Py<PyAny>>) -> PyResult<Self> {
        let _ = stream;
        self.ensure_not_deleted()?;
        validate_cpu_device(&device)?;
        Ok(self.clone())
    }

    #[getter]
    fn device_buffer(&self) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
            "arr.device_buffer has been deprecated. Use arr.addressable_data(0)",
        ))
    }

    #[getter]
    fn device_buffers(&self) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
            "arr.device_buffers has been deprecated. Use [x.data for x in arr.addressable_shards]",
        ))
    }

    fn addressable_data(&self, index: isize) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        if index != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "addressable_data index {index} is out of bounds for 1 local shard"
            )));
        }
        Ok(self.clone())
    }

    #[getter]
    fn addressable_shards(&self) -> PyResult<Vec<PyShard>> {
        self.ensure_not_deleted()?;
        Ok(vec![PyShard::from_array(self)])
    }

    #[getter]
    fn global_shards(&self) -> PyResult<Vec<PyShard>> {
        self.addressable_shards()
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

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.ensure_not_deleted()?;
        let values = self.leading_axis_values()?;
        let mut py_values = Vec::with_capacity(values.len());
        for value in values {
            py_values.push(Py::new(py, value)?);
        }
        Ok(PyList::new(py, py_values)?
            .call_method0("__iter__")?
            .unbind())
    }

    fn __reversed__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.ensure_not_deleted()?;
        let values = self.leading_axis_values()?;
        let mut py_values = Vec::with_capacity(values.len());
        for value in values.into_iter().rev() {
            py_values.push(Py::new(py, value)?);
        }
        Ok(PyList::new(py, py_values)?
            .call_method0("__iter__")?
            .unbind())
    }

    #[pyo3(signature = (*, api_version=None))]
    fn __array_namespace__(
        &self,
        py: Python<'_>,
        api_version: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        self.ensure_not_deleted()?;
        array_namespace(py, api_version)
    }

    #[pyo3(signature = (dtype = None, copy = false, device = None))]
    fn astype(
        &self,
        py: Python<'_>,
        dtype: Option<Py<PyAny>>,
        copy: bool,
        device: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let _ = copy;
        self.ensure_not_deleted()?;
        validate_astype_device(py, device)?;
        let dtype = dtype_from_py_arg(py, dtype)?;
        cast_value_to_dtype(&self.inner, dtype).map(Self::from_value)
    }

    #[pyo3(signature = (*args, order = "C", out_sharding = None))]
    fn reshape(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        order: &str,
        out_sharding: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        validate_reshape_out_sharding(py, out_sharding)?;
        let raw_shape = reshape_shape_from_py_args(args)?;
        reshape_value(&self.inner, &raw_shape, order).map(Self::from_value)
    }

    #[pyo3(signature = (order = "C", *, out_sharding = None))]
    fn flatten(
        &self,
        py: Python<'_>,
        order: &str,
        out_sharding: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        validate_reshape_out_sharding(py, out_sharding)?;
        flatten_value(&self.inner, order).map(Self::from_value)
    }

    #[pyo3(signature = (order = "C", *, out_sharding = None))]
    fn ravel(
        &self,
        py: Python<'_>,
        order: &str,
        out_sharding: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        self.flatten(py, order, out_sharding)
    }

    #[pyo3(signature = (axis = None))]
    fn squeeze(&self, py: Python<'_>, axis: Option<Py<PyAny>>) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        let axes = squeeze_axes_from_py_arg(py, axis, self.shape_dims().len())?;
        squeeze_value(&self.inner, axes.as_deref()).map(Self::from_value)
    }

    #[pyo3(signature = (decimals = 0, out = None))]
    fn round(&self, decimals: i32, out: Option<Py<PyAny>>) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        if out.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "The 'out' argument to round is not supported.",
            ));
        }
        round_value(&self.inner, decimals, false).map(Self::from_value)
    }

    #[pyo3(signature = (ndigits = None))]
    fn __round__(&self, ndigits: Option<i32>) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        let decimals = ndigits.unwrap_or(0);
        round_value(&self.inner, decimals, ndigits.is_none()).map(Self::from_value)
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        if let Ok(index) = index.extract::<isize>() {
            return self.axis0_value_at(index);
        }
        if let Ok(slice) = index.downcast::<PySlice>() {
            return self.axis0_slice(slice);
        }

        let type_repr: String = index.get_type().repr()?.extract()?;
        Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
            "Unrecognized index type: {type_repr}"
        )))
    }

    fn __setitem__(&self, _index: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_not_deleted()?;
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "JAX arrays are immutable and do not support in-place item assignment. \
             Instead of x[idx] = y, use x = x.at[idx].set(y) or another .at[] method.",
        ))
    }

    fn block_until_ready(&self) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        Ok(self.clone())
    }

    fn is_ready(&self) -> PyResult<bool> {
        self.ensure_not_deleted()?;
        Ok(true)
    }

    fn copy_to_host_async(&self) -> PyResult<()> {
        self.ensure_not_deleted()?;
        Ok(())
    }

    fn copy(&self) -> PyResult<Self> {
        self.ensure_not_deleted()?;
        Ok(self.clone())
    }

    fn __copy__(&self) -> PyResult<Self> {
        self.copy()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.copy()
    }

    fn delete(&mut self) {
        self.deleted = true;
    }

    fn is_deleted(&self) -> bool {
        self.deleted
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.ensure_not_deleted()?;
        match &self.inner {
            Value::Scalar(literal) => literal_to_py_object(py, *literal),
            Value::Tensor(tensor) => literals_to_py_list(py, &tensor.elements),
        }
    }

    #[pyo3(signature = (*args))]
    fn item(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        self.ensure_not_deleted()?;
        let size = usize::try_from(self.size()).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array size does not fit usize")
        })?;

        let index = match args.len() {
            0 => {
                if size != 1 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "can only convert an array of size 1 to a Python scalar",
                    ));
                }
                0
            }
            1 => {
                let raw_index: isize = args.get_item(0)?.extract()?;
                let size_isize = isize::try_from(size).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                        "array size does not fit isize",
                    )
                })?;
                let normalized_index = if raw_index < 0 {
                    raw_index.checked_add(size_isize).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                            "item index is out of bounds",
                        )
                    })?
                } else {
                    raw_index
                };
                if !(0..size_isize).contains(&normalized_index) {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        "item index is out of bounds",
                    ));
                }
                usize::try_from(normalized_index).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                        "item index does not fit usize",
                    )
                })?
            }
            arg_count => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "item() takes at most 1 argument ({arg_count} given)"
                )));
            }
        };

        let literal = match &self.inner {
            Value::Scalar(literal) => *literal,
            Value::Tensor(tensor) => tensor.elements.get(index).copied().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>("item index is out of bounds")
            })?,
        };
        literal_to_py_object(py, literal)
    }

    #[pyo3(signature = (order = "C"))]
    fn tobytes(&self, py: Python<'_>, order: &str) -> PyResult<Py<PyAny>> {
        self.ensure_not_deleted()?;
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
        self.ensure_not_deleted()?;
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
        self.ensure_not_deleted()?;
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
        self.ensure_not_deleted()?;
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
        self.ensure_not_deleted()?;
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
        self.ensure_not_deleted()?;
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

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        self.tolist(py)?.bind(py).str()?.extract()
    }

    fn __format__(&self, py: Python<'_>, format_spec: &str) -> PyResult<String> {
        let value = self.tolist(py)?;
        py.import("builtins")?
            .getattr("format")?
            .call1((value.bind(py), format_spec))?
            .extract()
    }

    #[pyo3(signature = (dtype=None, context=None, copy=None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<Py<PyAny>>,
        context: Option<Py<PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let _ = context;
        let value = self.tolist(py)?;
        let kwargs = PyDict::new(py);
        if let Some(dtype) = dtype {
            let dtype = dtype.bind(py);
            if !dtype.is_none() {
                kwargs.set_item("dtype", dtype)?;
            }
        }
        if let Some(copy) = copy {
            kwargs.set_item("copy", copy)?;
        }
        Ok(py
            .import("numpy")?
            .getattr("asarray")?
            .call((value.bind(py),), Some(&kwargs))?
            .unbind())
    }

    fn __dlpack_device__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.ensure_not_deleted()?;
        Ok(PyTuple::new(py, [1_i32, 0_i32])?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __hash__(&self) -> PyResult<isize> {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "unhashable type: 'Array'",
        ))
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

    #[pyo3(signature = (**kwargs))]
    fn update(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        self.update_from_kwargs(kwargs)
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
impl PyShard {
    #[getter]
    fn device(&self) -> PyDevice {
        self.device.clone()
    }

    #[getter]
    fn index(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let slices: Vec<_> = (0..self.rank).map(|_| PySlice::full(py)).collect();
        Ok(PyTuple::new(py, slices)?.into_any().unbind())
    }

    #[getter]
    fn replica_id(&self) -> usize {
        0
    }

    #[getter]
    fn data(&self) -> Option<PyValue> {
        self.data.clone()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let index = self.index(py)?;
        let index = index.bind(py).repr()?.extract::<String>()?;
        let data = match &self.data {
            Some(data) => data.__repr__(),
            None => "None".to_string(),
        };
        Ok(format!(
            "Shard(device={}, index={index}, replica_id=0, data={data})",
            self.device.__repr__()
        ))
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

fn py_dict_get_item<'py>(
    kwargs: Option<&Bound<'py, PyDict>>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match kwargs {
        Some(kwargs) => kwargs.get_item(name),
        None => Ok(None),
    }
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

fn numpy_dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "bfloat16",
        DType::F16 => "float16",
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U32 => "uint32",
        DType::U64 => "uint64",
        DType::Bool => "bool",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
    }
}

fn numpy_dtype_object(py: Python<'_>, dtype: DType) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("numpy")?
        .getattr("dtype")?
        .call1((numpy_dtype_name(dtype),))?
        .unbind())
}

fn dtype_from_py_arg(py: Python<'_>, dtype: Option<Py<PyAny>>) -> PyResult<DType> {
    let Some(dtype) = dtype else {
        return Ok(DType::F64);
    };
    let dtype = dtype.bind(py);
    if dtype.is_none() {
        return Ok(DType::F64);
    }
    if let Ok(name) = dtype.extract::<String>() {
        return dtype_from_name(&name);
    }

    let name = py
        .import("numpy")?
        .getattr("dtype")?
        .call1((dtype,))?
        .getattr("name")?
        .extract::<String>()?;
    dtype_from_name(&name)
}

fn dtype_from_name(name: &str) -> PyResult<DType> {
    let name = name.to_ascii_lowercase();
    match name.as_str() {
        "bool" | "bool_" | "boolean" => Ok(DType::Bool),
        "i32" | "int32" => Ok(DType::I32),
        "i64" | "int" | "int64" | "long" => Ok(DType::I64),
        "u32" | "uint32" => Ok(DType::U32),
        "u64" | "uint64" => Ok(DType::U64),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f16" | "float16" | "half" => Ok(DType::F16),
        "f32" | "float32" | "single" => Ok(DType::F32),
        "f64" | "float" | "float64" | "double" => Ok(DType::F64),
        "c64" | "complex64" => Ok(DType::Complex64),
        "c128" | "complex" | "complex128" => Ok(DType::Complex128),
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "unsupported dtype {name:?}"
        ))),
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

fn value_error(error: impl ToString) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(error.to_string())
}

fn row_major_strides(dims: &[u32]) -> PyResult<Vec<usize>> {
    let mut strides = vec![1_usize; dims.len()];
    let mut stride = 1_usize;
    for (dim, stride_slot) in dims.iter().rev().zip(strides.iter_mut().rev()) {
        *stride_slot = stride;
        let dim = usize::try_from(*dim).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array dimension does not fit usize")
        })?;
        stride = stride.checked_mul(dim).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "array stride computation overflowed",
            )
        })?;
    }
    Ok(strides)
}

fn validate_permutation(permutation: &[usize], rank: usize) -> PyResult<()> {
    let mut seen = vec![false; rank];
    for axis in permutation {
        let Some(slot) = seen.get_mut(*axis) else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "transpose permutation axis is out of bounds",
            ));
        };
        if *slot {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "transpose permutation contains a repeated axis",
            ));
        }
        *slot = true;
    }
    Ok(())
}

fn default_transpose_permutation(rank: usize) -> Vec<usize> {
    (0..rank).rev().collect()
}

fn transpose_permutation_from_py_args(
    args: &Bound<'_, PyTuple>,
    rank: usize,
) -> PyResult<Vec<usize>> {
    let raw_axes = match args.len() {
        0 => return Ok(default_transpose_permutation(rank)),
        1 => {
            let value = args.get_item(0)?;
            if value.is_none() {
                return Ok(default_transpose_permutation(rank));
            }
            extract_axis_sequence(&value)?
        }
        _ => {
            let mut axes = Vec::with_capacity(args.len());
            for index in 0..args.len() {
                axes.push(args.get_item(index)?.extract::<isize>()?);
            }
            axes
        }
    };
    normalize_axis_permutation(&raw_axes, rank)
}

fn extract_axis_sequence(value: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(axis) = value.extract::<isize>() {
        return Ok(vec![axis]);
    }
    value.extract::<Vec<isize>>().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "transpose axes must be None, an integer, or a sequence of integers",
        )
    })
}

fn normalize_axis_permutation(raw_axes: &[isize], rank: usize) -> PyResult<Vec<usize>> {
    if raw_axes.len() != rank {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "transpose permutation rank does not match array rank",
        ));
    }

    let mut permutation = Vec::with_capacity(raw_axes.len());
    for axis in raw_axes {
        permutation.push(normalize_axis(*axis, rank)?);
    }
    validate_permutation(&permutation, rank)?;
    Ok(permutation)
}

fn normalize_axis(axis: isize, rank: usize) -> PyResult<usize> {
    let rank_isize = isize::try_from(rank).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array rank does not fit isize")
    })?;
    let normalized = if axis < 0 {
        axis.checked_add(rank_isize).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("axis normalization overflowed")
        })?
    } else {
        axis
    };
    if !(0..rank_isize).contains(&normalized) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "axis {axis} is out of bounds for array of dimension {rank}"
        )));
    }
    usize::try_from(normalized).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyOverflowError, _>("normalized axis does not fit usize")
    })
}

fn reshape_shape_from_py_args(args: &Bound<'_, PyTuple>) -> PyResult<Vec<i64>> {
    match args.len() {
        0 => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "reshape() missing required shape argument",
        )),
        1 => {
            let shape = args.get_item(0)?;
            extract_reshape_shape(&shape)
        }
        len => {
            let mut dims = Vec::with_capacity(len);
            for index in 0..len {
                dims.push(args.get_item(index)?.extract::<i64>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "reshape dimensions must be integers",
                    )
                })?);
            }
            Ok(dims)
        }
    }
}

fn extract_reshape_shape(shape: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(dim) = shape.extract::<i64>() {
        return Ok(vec![dim]);
    }
    shape.extract::<Vec<i64>>().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "reshape shape must be an integer or a sequence of integers",
        )
    })
}

fn validate_reshape_out_sharding(py: Python<'_>, out_sharding: Option<Py<PyAny>>) -> PyResult<()> {
    let Some(out_sharding) = out_sharding else {
        return Ok(());
    };
    if out_sharding.bind(py).is_none() {
        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Array.reshape out_sharding is not supported by the CPU-local fj-py backend",
        ))
    }
}

fn reshape_value(value: &Value, raw_shape: &[i64], order: &str) -> PyResult<Value> {
    let element_count = value_element_count(value)?;
    let dims = resolve_reshape_dims(raw_shape, element_count)?;

    match order {
        "C" => reshape_value_with_order(value, dims, ReshapeOrder::C),
        "F" => reshape_value_with_order(value, dims, ReshapeOrder::F),
        "A" => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "np.reshape order=A is not implemented.",
        )),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unexpected value for 'order' argument: {order}."
        ))),
    }
}

fn flatten_value(value: &Value, order: &str) -> PyResult<Value> {
    let element_count = value_element_count(value)?;
    let flattened_dim = i64::try_from(element_count).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array element count does not fit i64")
    })?;
    reshape_value(value, &[flattened_dim], order)
}

fn squeeze_axes_from_py_arg(
    py: Python<'_>,
    axis: Option<Py<PyAny>>,
    rank: usize,
) -> PyResult<Option<Vec<usize>>> {
    let Some(axis) = axis else {
        return Ok(None);
    };
    let axis = axis.bind(py);
    if axis.is_none() {
        return Ok(None);
    }
    let raw_axes = if let Ok(axis) = axis.extract::<isize>() {
        vec![axis]
    } else {
        axis.extract::<Vec<isize>>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "squeeze axis must be None, an integer, or a sequence of integers",
            )
        })?
    };

    let mut axes = Vec::with_capacity(raw_axes.len());
    let mut seen = vec![false; rank];
    for axis in raw_axes {
        let axis = normalize_axis(axis, rank)?;
        let slot = seen.get_mut(axis).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("squeeze axis is out of bounds")
        })?;
        if *slot {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "duplicate value in 'axis'",
            ));
        }
        *slot = true;
        axes.push(axis);
    }
    Ok(Some(axes))
}

fn squeeze_value(value: &Value, axes: Option<&[usize]>) -> PyResult<Value> {
    match value {
        Value::Scalar(_) => Ok(value.clone()),
        Value::Tensor(tensor) => {
            let dims = squeeze_dims(&tensor.shape.dims, axes)?;
            value_with_shape(tensor.dtype, &tensor.elements, dims)
        }
    }
}

fn squeeze_dims(dims: &[u32], axes: Option<&[usize]>) -> PyResult<Vec<u32>> {
    let Some(axes) = axes else {
        return Ok(dims.iter().copied().filter(|dim| *dim != 1).collect());
    };
    let mut remove = vec![false; dims.len()];
    for axis in axes {
        let dim = dims.get(*axis).copied().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("squeeze axis is out of bounds")
        })?;
        if dim != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cannot select an axis to squeeze out which has size not equal to one",
            ));
        }
        let slot = remove.get_mut(*axis).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("squeeze axis is out of bounds")
        })?;
        *slot = true;
    }

    Ok(dims
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(axis, dim)| {
            let should_remove = remove.get(axis).copied().unwrap_or(false);
            (!should_remove).then_some(dim)
        })
        .collect())
}

fn value_with_shape(dtype: DType, elements: &[Literal], dims: Vec<u32>) -> PyResult<Value> {
    if dims.is_empty() {
        let literal = elements.first().copied().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cannot reshape empty array into scalar shape",
            )
        })?;
        return Ok(Value::Scalar(literal));
    }

    TensorValue::new(dtype, Shape { dims }, elements.to_vec())
        .map(Value::Tensor)
        .map_err(value_error)
}

#[derive(Clone, Copy)]
enum ReshapeOrder {
    C,
    F,
}

fn reshape_value_with_order(value: &Value, dims: Vec<u32>, order: ReshapeOrder) -> PyResult<Value> {
    let dtype = value.dtype();
    let elements = match value {
        Value::Scalar(literal) => vec![*literal],
        Value::Tensor(tensor) => match order {
            ReshapeOrder::C => tensor.elements.clone(),
            ReshapeOrder::F => {
                fortran_reshape_elements(&tensor.elements, &tensor.shape.dims, &dims)?
            }
        },
    };

    if dims.is_empty() {
        let literal = elements.first().copied().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cannot reshape empty array into scalar shape",
            )
        })?;
        return Ok(Value::Scalar(literal));
    }

    TensorValue::new(dtype, Shape { dims }, elements)
        .map(Value::Tensor)
        .map_err(value_error)
}

fn value_element_count(value: &Value) -> PyResult<u64> {
    match value {
        Value::Scalar(_) => Ok(1),
        Value::Tensor(tensor) => u64::try_from(tensor.elements.len()).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "array element count does not fit u64",
            )
        }),
    }
}

fn resolve_reshape_dims(raw_shape: &[i64], element_count: u64) -> PyResult<Vec<u32>> {
    let mut dims = Vec::with_capacity(raw_shape.len());
    let mut inferred_axis = None;
    let mut known_product = 1_u64;

    for (axis, dim) in raw_shape.iter().copied().enumerate() {
        if dim == -1 {
            if inferred_axis.is_some() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "can only specify one unknown axis size with -1",
                ));
            }
            inferred_axis = Some(axis);
            dims.push(0);
        } else if dim < -1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "negative dimensions are not allowed",
            ));
        } else {
            let dim = u64::try_from(dim).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                    "reshape dimension does not fit u64",
                )
            })?;
            known_product = known_product.checked_mul(dim).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                    "reshape dimension product overflowed",
                )
            })?;
            dims.push(u32::try_from(dim).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                    "reshape dimension does not fit u32",
                )
            })?);
        }
    }

    if let Some(axis) = inferred_axis {
        if known_product == 0 || !element_count.is_multiple_of(known_product) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "cannot infer reshape dimension for size {element_count} and shape {raw_shape:?}"
            )));
        }
        let inferred = element_count / known_product;
        let inferred = u32::try_from(inferred).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "inferred reshape dimension does not fit u32",
            )
        })?;
        let slot = dims.get_mut(axis).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "inferred reshape axis is out of bounds",
            )
        })?;
        *slot = inferred;
    } else if known_product != element_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "cannot reshape array of size {element_count} into shape {raw_shape:?}"
        )));
    }

    Ok(dims)
}

fn fortran_reshape_elements(
    elements: &[Literal],
    old_dims: &[u32],
    new_dims: &[u32],
) -> PyResult<Vec<Literal>> {
    if elements.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(elements.len());
    for row_major_index in 0..elements.len() {
        let new_coord = unravel_row_major(row_major_index, new_dims)?;
        let fortran_position = column_major_flat_index(&new_coord, new_dims)?;
        let old_coord = unravel_column_major(fortran_position, old_dims)?;
        let old_row_major_index = row_major_flat_index(&old_coord, old_dims)?;
        let literal = elements.get(old_row_major_index).copied().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "reshape computed an out-of-bounds source index",
            )
        })?;
        output.push(literal);
    }
    Ok(output)
}

fn unravel_row_major(mut index: usize, dims: &[u32]) -> PyResult<Vec<usize>> {
    if dims.is_empty() {
        if index == 0 {
            return Ok(Vec::new());
        }
        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
            "scalar row-major index is out of bounds",
        ));
    }

    let strides = row_major_strides(dims)?;
    let mut coord = Vec::with_capacity(dims.len());
    for (dim, stride) in dims.iter().zip(strides) {
        let dim = usize::try_from(*dim).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array dimension does not fit usize")
        })?;
        if dim == 0 || stride == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "cannot unravel index for an empty reshape dimension",
            ));
        }
        let axis_coord = index / stride;
        if axis_coord >= dim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "row-major reshape coordinate is out of bounds",
            ));
        }
        coord.push(axis_coord);
        index %= stride;
    }
    Ok(coord)
}

fn row_major_flat_index(coord: &[usize], dims: &[u32]) -> PyResult<usize> {
    if coord.len() != dims.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "reshape coordinate rank does not match shape rank",
        ));
    }

    let strides = row_major_strides(dims)?;
    let mut index = 0_usize;
    for ((axis_coord, dim), stride) in coord.iter().zip(dims).zip(strides) {
        let dim = usize::try_from(*dim).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array dimension does not fit usize")
        })?;
        if *axis_coord >= dim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "reshape coordinate is out of bounds",
            ));
        }
        let offset = axis_coord.checked_mul(stride).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "reshape index computation overflowed",
            )
        })?;
        index = index.checked_add(offset).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "reshape index computation overflowed",
            )
        })?;
    }
    Ok(index)
}

fn column_major_flat_index(coord: &[usize], dims: &[u32]) -> PyResult<usize> {
    if coord.len() != dims.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "reshape coordinate rank does not match shape rank",
        ));
    }

    let mut index = 0_usize;
    let mut stride = 1_usize;
    for (axis_coord, dim) in coord.iter().zip(dims) {
        let dim = usize::try_from(*dim).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array dimension does not fit usize")
        })?;
        if *axis_coord >= dim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "reshape coordinate is out of bounds",
            ));
        }
        let offset = axis_coord.checked_mul(stride).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "reshape index computation overflowed",
            )
        })?;
        index = index.checked_add(offset).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "reshape index computation overflowed",
            )
        })?;
        stride = stride.checked_mul(dim).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "reshape stride computation overflowed",
            )
        })?;
    }
    Ok(index)
}

fn unravel_column_major(mut index: usize, dims: &[u32]) -> PyResult<Vec<usize>> {
    let mut coord = Vec::with_capacity(dims.len());
    for dim in dims {
        let dim = usize::try_from(*dim).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("array dimension does not fit usize")
        })?;
        if dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "cannot unravel index for an empty reshape dimension",
            ));
        }
        coord.push(index % dim);
        index /= dim;
    }
    if index == 0 {
        Ok(coord)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
            "column-major reshape index is out of bounds",
        ))
    }
}

fn value_real_part(value: &Value) -> PyResult<Value> {
    match value {
        Value::Scalar(literal) => Ok(Value::Scalar(literal_real_part(*literal))),
        Value::Tensor(tensor) if matches!(tensor.dtype, DType::Complex64 | DType::Complex128) => {
            let dtype = complex_component_dtype(tensor.dtype);
            let elements = tensor
                .elements
                .iter()
                .copied()
                .map(literal_real_part)
                .collect::<Vec<_>>();
            TensorValue::new(dtype, tensor.shape.clone(), elements)
                .map(Value::Tensor)
                .map_err(value_error)
        }
        Value::Tensor(_) => Ok(value.clone()),
    }
}

fn value_imag_part(value: &Value) -> PyResult<Value> {
    match value {
        Value::Scalar(literal) => Ok(Value::Scalar(literal_imag_part(*literal, value.dtype()))),
        Value::Tensor(tensor) if matches!(tensor.dtype, DType::Complex64 | DType::Complex128) => {
            let dtype = complex_component_dtype(tensor.dtype);
            let elements = tensor
                .elements
                .iter()
                .copied()
                .map(|literal| literal_imag_part(literal, tensor.dtype))
                .collect::<Vec<_>>();
            TensorValue::new(dtype, tensor.shape.clone(), elements)
                .map(Value::Tensor)
                .map_err(value_error)
        }
        Value::Tensor(tensor) => {
            let elements = vec![zero_literal_for_dtype(tensor.dtype); tensor.elements.len()];
            TensorValue::new(tensor.dtype, tensor.shape.clone(), elements)
                .map(Value::Tensor)
                .map_err(value_error)
        }
    }
}

fn value_conjugate(value: &Value) -> PyResult<Value> {
    match value {
        Value::Scalar(literal) => Ok(Value::Scalar(literal_conjugate(*literal))),
        Value::Tensor(tensor) if matches!(tensor.dtype, DType::Complex64 | DType::Complex128) => {
            let elements = tensor
                .elements
                .iter()
                .copied()
                .map(literal_conjugate)
                .collect::<Vec<_>>();
            TensorValue::new(tensor.dtype, tensor.shape.clone(), elements)
                .map(Value::Tensor)
                .map_err(value_error)
        }
        Value::Tensor(_) => Ok(value.clone()),
    }
}

fn literal_real_part(literal: Literal) -> Literal {
    match literal {
        Literal::Complex64Bits(re, _) => Literal::F32Bits(re),
        Literal::Complex128Bits(re, _) => Literal::F64Bits(re),
        _ => literal,
    }
}

fn literal_imag_part(literal: Literal, dtype: DType) -> Literal {
    match literal {
        Literal::Complex64Bits(_, im) => Literal::F32Bits(im),
        Literal::Complex128Bits(_, im) => Literal::F64Bits(im),
        _ => zero_literal_for_dtype(dtype),
    }
}

fn literal_conjugate(literal: Literal) -> Literal {
    match literal {
        Literal::Complex64Bits(re, im) => {
            Literal::from_complex64(f32::from_bits(re), -f32::from_bits(im))
        }
        Literal::Complex128Bits(re, im) => {
            Literal::from_complex128(f64::from_bits(re), -f64::from_bits(im))
        }
        _ => literal,
    }
}

fn complex_component_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Complex64 => DType::F32,
        DType::Complex128 => DType::F64,
        _ => dtype,
    }
}

fn zero_literal_for_dtype(dtype: DType) -> Literal {
    match dtype {
        DType::Bool => Literal::Bool(false),
        DType::I32 | DType::I64 => Literal::I64(0),
        DType::U32 => Literal::U32(0),
        DType::U64 => Literal::U64(0),
        DType::BF16 => Literal::BF16Bits(0),
        DType::F16 => Literal::F16Bits(0),
        DType::F32 => Literal::from_f32(0.0),
        DType::F64 => Literal::from_f64(0.0),
        DType::Complex64 => Literal::from_complex64(0.0, 0.0),
        DType::Complex128 => Literal::from_complex128(0.0, 0.0),
    }
}

fn cast_value_to_dtype(value: &Value, dtype: DType) -> PyResult<Value> {
    match value {
        Value::Scalar(literal) => cast_literal_to_dtype(*literal, dtype).map(Value::Scalar),
        Value::Tensor(tensor) => {
            let elements = tensor
                .elements
                .iter()
                .copied()
                .map(|literal| cast_literal_to_dtype(literal, dtype))
                .collect::<PyResult<Vec<_>>>()?;
            TensorValue::new(dtype, tensor.shape.clone(), elements)
                .map(Value::Tensor)
                .map_err(value_error)
        }
    }
}

fn cast_literal_to_dtype(literal: Literal, dtype: DType) -> PyResult<Literal> {
    match dtype {
        DType::Bool => Ok(Literal::Bool(literal_truth_value(literal))),
        DType::I32 => float_to_i64(
            literal_real_f64(literal)?,
            i64::from(i32::MIN),
            i64::from(i32::MAX),
        )
        .map(Literal::I64),
        DType::I64 => {
            float_to_i64(literal_real_f64(literal)?, i64::MIN, i64::MAX).map(Literal::I64)
        }
        DType::U32 => {
            float_to_u64(literal_real_f64(literal)?, u64::from(u32::MAX)).and_then(|value| {
                u32::try_from(value).map(Literal::U32).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                        "cast value does not fit uint32",
                    )
                })
            })
        }
        DType::U64 => float_to_u64(literal_real_f64(literal)?, u64::MAX).map(Literal::U64),
        DType::BF16 => Ok(Literal::from_bf16_f32(literal_real_f64(literal)? as f32)),
        DType::F16 => Ok(Literal::from_f16_f32(literal_real_f64(literal)? as f32)),
        DType::F32 => Ok(Literal::from_f32(literal_real_f64(literal)? as f32)),
        DType::F64 => Ok(Literal::from_f64(literal_real_f64(literal)?)),
        DType::Complex64 => {
            let (re, im) = literal_complex_f64(literal)?;
            Ok(Literal::from_complex64(re as f32, im as f32))
        }
        DType::Complex128 => {
            let (re, im) = literal_complex_f64(literal)?;
            Ok(Literal::from_complex128(re, im))
        }
    }
}

fn literal_real_f64(literal: Literal) -> PyResult<f64> {
    match literal {
        Literal::Bool(value) => Ok(if value { 1.0 } else { 0.0 }),
        Literal::Complex64Bits(re, _) => Ok(f64::from(f32::from_bits(re))),
        Literal::Complex128Bits(re, _) => Ok(f64::from_bits(re)),
        _ => literal.as_f64().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>("cannot cast literal to a real dtype")
        }),
    }
}

fn literal_complex_f64(literal: Literal) -> PyResult<(f64, f64)> {
    match literal {
        Literal::Complex64Bits(re, im) => {
            Ok((f64::from(f32::from_bits(re)), f64::from(f32::from_bits(im))))
        }
        Literal::Complex128Bits(re, im) => Ok((f64::from_bits(re), f64::from_bits(im))),
        _ => Ok((literal_real_f64(literal)?, 0.0)),
    }
}

fn float_to_i64(value: f64, min: i64, max: i64) -> PyResult<i64> {
    if !value.is_finite() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cannot cast non-finite value to int",
        ));
    }
    let value = value.trunc();
    if value < min as f64 || value > max as f64 {
        return Err(PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
            "cast value does not fit signed integer dtype",
        ));
    }
    Ok(value as i64)
}

fn float_to_u64(value: f64, max: u64) -> PyResult<u64> {
    if !value.is_finite() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cannot cast non-finite value to uint",
        ));
    }
    let value = value.trunc();
    if value < 0.0 || value > max as f64 {
        return Err(PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
            "cast value does not fit unsigned integer dtype",
        ));
    }
    Ok(value as u64)
}

fn round_value(value: &Value, decimals: i32, cast_to_int: bool) -> PyResult<Value> {
    match value {
        Value::Scalar(literal) => {
            round_literal(*literal, value.dtype(), decimals, cast_to_int).map(Value::Scalar)
        }
        Value::Tensor(tensor) => {
            let output_dtype = if cast_to_int {
                DType::I64
            } else {
                tensor.dtype
            };
            let elements = tensor
                .elements
                .iter()
                .copied()
                .map(|literal| round_literal(literal, tensor.dtype, decimals, cast_to_int))
                .collect::<PyResult<Vec<_>>>()?;
            TensorValue::new(output_dtype, tensor.shape.clone(), elements)
                .map(Value::Tensor)
                .map_err(value_error)
        }
    }
}

fn round_literal(
    literal: Literal,
    dtype: DType,
    decimals: i32,
    cast_to_int: bool,
) -> PyResult<Literal> {
    if cast_to_int {
        let rounded = round_literal(literal, dtype, decimals, false)?;
        return rounded_literal_to_i64(rounded);
    }

    match (dtype, literal) {
        (DType::I32 | DType::I64 | DType::U32 | DType::U64, _) => {
            if decimals < 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "integer round is not implemented for decimals < 0",
                ));
            }
            Ok(literal)
        }
        (DType::F32, Literal::F32Bits(bits)) => {
            Ok(Literal::from_f32(round_f32(f32::from_bits(bits), decimals)))
        }
        (DType::F64, Literal::F64Bits(bits)) => {
            Ok(Literal::from_f64(round_f64(f64::from_bits(bits), decimals)))
        }
        (DType::BF16, Literal::BF16Bits(bits)) => {
            let value = Literal::BF16Bits(bits)
                .as_f64()
                .expect("bf16 literal should convert to f64") as f32;
            Ok(Literal::from_bf16_f32(round_f32(value, decimals)))
        }
        (DType::F16, Literal::F16Bits(bits)) => {
            let value = Literal::F16Bits(bits)
                .as_f64()
                .expect("f16 literal should convert to f64") as f32;
            Ok(Literal::from_f16_f32(round_f32(value, decimals)))
        }
        (DType::Complex64, Literal::Complex64Bits(re, im)) => Ok(Literal::from_complex64(
            round_f32(f32::from_bits(re), decimals),
            round_f32(f32::from_bits(im), decimals),
        )),
        (DType::Complex128, Literal::Complex128Bits(re, im)) => Ok(Literal::from_complex128(
            round_f64(f64::from_bits(re), decimals),
            round_f64(f64::from_bits(im), decimals),
        )),
        (DType::Bool, Literal::Bool(_)) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "round is not defined for bool arrays",
        )),
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "round literal dtype mismatch",
        )),
    }
}

fn round_f32(value: f32, decimals: i32) -> f32 {
    if decimals == 0 {
        value.round_ties_even()
    } else {
        let factor = 10_f32.powi(decimals);
        if factor.is_infinite() {
            value
        } else {
            (value * factor).round_ties_even() / factor
        }
    }
}

fn round_f64(value: f64, decimals: i32) -> f64 {
    if decimals == 0 {
        value.round_ties_even()
    } else {
        let factor = 10_f64.powi(decimals);
        if factor.is_infinite() {
            value
        } else {
            (value * factor).round_ties_even() / factor
        }
    }
}

fn rounded_literal_to_i64(literal: Literal) -> PyResult<Literal> {
    match literal {
        Literal::I64(value) => Ok(Literal::I64(value)),
        Literal::U32(value) => Ok(Literal::I64(i64::from(value))),
        Literal::U64(value) => i64::try_from(value).map(Literal::I64).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>("rounded value does not fit int64")
        }),
        Literal::F32Bits(bits) => rounded_float_to_i64(f64::from(f32::from_bits(bits))),
        Literal::F64Bits(bits) => rounded_float_to_i64(f64::from_bits(bits)),
        Literal::BF16Bits(_) | Literal::F16Bits(_) => rounded_float_to_i64(
            literal
                .as_f64()
                .expect("low-precision float literal should convert to f64"),
        ),
        Literal::Bool(value) => Ok(Literal::I64(i64::from(value))),
        Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "round with ndigits=None cannot cast complex arrays to int",
            ))
        }
    }
}

fn rounded_float_to_i64(value: f64) -> PyResult<Literal> {
    if !value.is_finite() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cannot cast non-finite rounded value to int64",
        ));
    }
    if value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
            "rounded value does not fit int64",
        ));
    }
    Ok(Literal::I64(value as i64))
}

fn array_namespace(py: Python<'_>, api_version: Option<&str>) -> PyResult<Py<PyAny>> {
    if let Some(api_version) = api_version
        && api_version != ARRAY_API_VERSION
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "api_version={api_version:?} is not available; available versions are: [{ARRAY_API_VERSION:?}]"
        )));
    }
    Ok(py.import("frankenjax")?.into_any().unbind())
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

fn validate_astype_device(py: Python<'_>, device: Option<Py<PyAny>>) -> PyResult<()> {
    let Some(device) = device else {
        return Ok(());
    };
    let device = device.bind(py);
    if device.is_none() {
        return Ok(());
    }
    if let Ok(device) = device.extract::<PyRef<'_, PyDevice>>() {
        return validate_cpu_device(&device);
    }
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "astype device must be None or a fj-py CPU Device",
    ))
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
fn device_get(value: PyValue) -> PyResult<PyValue> {
    value.ensure_not_deleted()?;
    Ok(value)
}

#[pyfunction]
fn block_until_ready(value: PyValue) -> PyResult<PyValue> {
    value.block_until_ready()
}

#[pyfunction]
fn copy_to_host_async(value: PyValue) -> PyResult<PyValue> {
    value.copy_to_host_async()?;
    Ok(value)
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
    m.add("__array_api_version__", ARRAY_API_VERSION)?;
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
    m.add_class::<PyShard>()?;
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
        assert_eq!(
            v.transpose_all_axes().unwrap().shape_dims(),
            Vec::<u32>::new()
        );
        assert_eq!(v.dtype(), "F64");
        assert!((v.real_part().unwrap().as_f64().unwrap() - 42.0).abs() < 1e-12);
        assert!(v.imag_part().unwrap().as_f64().unwrap().abs() < 1e-12);
        assert_eq!(
            PyValue::scalar_f64(10.5)
                .round(0, None)
                .unwrap()
                .as_f64()
                .unwrap(),
            10.0
        );
        assert_eq!(
            PyValue::scalar_f64(21.5)
                .round(0, None)
                .unwrap()
                .as_f64()
                .unwrap(),
            22.0
        );
        assert_eq!(
            PyValue::scalar_f64(10.5)
                .__round__(None)
                .unwrap()
                .as_i64()
                .unwrap(),
            10
        );
        assert!(
            (PyValue::scalar_f64(1.25)
                .__round__(Some(1))
                .unwrap()
                .as_f64()
                .unwrap()
                - 1.2)
                .abs()
                < 1e-12
        );
        assert!(v.__hash__().is_err());
        assert_eq!(v.ndim(), 0);
        assert_eq!(v.size(), 1);
        assert_eq!(v.itemsize(), 8);
        assert_eq!(v.nbytes(), 8);
        assert!(v.leading_axis_values().is_err());
        assert!(v.axis0_value_at(0).is_err());
        assert!(!v.weak_type());
        assert!(!v.committed());
        let device = v.device();
        assert_eq!(device.id(), 0);
        assert_eq!(device.process_index(), 0);
        assert_eq!(device.platform(), "cpu");
        assert_eq!(v.platform(), "cpu");
        assert!(v.device_buffer().is_err());
        assert!(v.device_buffers().is_err());
        assert!((v.addressable_data(0).unwrap().as_f64().unwrap() - 42.0).abs() < 1e-12);
        assert!(v.addressable_data(1).is_err());
        assert_eq!(v.on_device_size_in_bytes(), v.nbytes());
        assert!(v.is_fully_addressable());
        assert!(v.is_fully_replicated());
        assert!(v.__len__().is_err());
        assert!(v.swapaxes(0, 0).is_err());
        assert!((v.block_until_ready().unwrap().as_f64().unwrap() - 42.0).abs() < 1e-12);
        assert!(v.is_ready().unwrap());
        assert!(v.copy_to_host_async().is_ok());
        assert!((v.copy().unwrap().as_f64().unwrap() - 42.0).abs() < 1e-12);
        assert!((v.__copy__().unwrap().as_f64().unwrap() - 42.0).abs() < 1e-12);
        let mut deleted = v.copy().unwrap();
        assert!(!deleted.is_deleted());
        deleted.delete();
        assert!(deleted.is_deleted());
        assert!(deleted.block_until_ready().is_err());
        assert!(deleted.is_ready().is_err());
        assert!(deleted.copy_to_host_async().is_err());
        assert!(deleted.copy().is_err());
        assert!(deleted.__copy__().is_err());
        assert!(deleted.transpose_all_axes().is_err());
        assert!(deleted.matrix_transpose().is_err());
        assert!(deleted.swapaxes(0, 0).is_err());
        assert!(deleted.real_part().is_err());
        assert!(deleted.imag_part().is_err());
        assert!(deleted.conjugate().is_err());
        assert!(deleted.round(0, None).is_err());
        assert!(deleted.__round__(None).is_err());
        assert!(deleted.addressable_data(0).is_err());
        assert!(deleted.addressable_shards().is_err());
        assert!(deleted.global_shards().is_err());
        assert!(!v.is_deleted());
        let shards = v.addressable_shards().unwrap();
        assert_eq!(shards.len(), 1);
        let shard = shards.first().unwrap();
        assert_eq!(shard.device().platform(), "cpu");
        assert_eq!(shard.replica_id(), 0);
        assert!((shard.data().unwrap().as_f64().unwrap() - 42.0).abs() < 1e-12);
        let global_shards = v.global_shards().unwrap();
        assert_eq!(global_shards.len(), 1);
        let global_shard = global_shards.first().unwrap();
        assert!((global_shard.data().unwrap().as_f64().unwrap() - 42.0).abs() < 1e-12);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let memo = PyDict::new(py);
            assert!(
                (v.__deepcopy__(memo.as_any()).unwrap().as_f64().unwrap() - 42.0).abs() < 1e-12
            );
            let memo = PyDict::new(py);
            assert!(deleted.__deepcopy__(memo.as_any()).is_err());
            assert!(deleted.devices(py).is_err());
            assert!(deleted.tolist(py).is_err());
            assert!(deleted.tobytes(py, "C").is_err());
            assert!(deleted.transpose(&PyTuple::empty(py)).is_err());
            assert!(deleted.astype(py, None, false, None).is_err());
            assert!(deleted.reshape(py, &PyTuple::empty(py), "C", None).is_err());
            assert!(deleted.flatten(py, "C", None).is_err());
            assert!(deleted.ravel(py, "C", None).is_err());
            assert!(deleted.squeeze(py, None).is_err());
            assert!(deleted.__array__(py, None, None, None).is_err());
            assert!(deleted.__dlpack_device__(py).is_err());
            assert_eq!(
                v.transpose(&PyTuple::empty(py)).unwrap().shape_dims(),
                Vec::<u32>::new()
            );
            let shard_index = shard.index(py).unwrap();
            assert_eq!(shard_index.bind(py).downcast::<PyTuple>().unwrap().len(), 0);
            let devices = v.devices(py).unwrap();
            assert_eq!(devices.bind(py).len().unwrap(), 1);
            let dlpack_device = v.__dlpack_device__(py).unwrap();
            let dlpack_device = dlpack_device.bind(py).downcast::<PyTuple>().unwrap();
            assert_eq!(dlpack_device.len(), 2);
            assert_eq!(
                dlpack_device.get_item(0).unwrap().extract::<i32>().unwrap(),
                1
            );
            assert_eq!(
                dlpack_device.get_item(1).unwrap().extract::<i32>().unwrap(),
                0
            );
            assert_eq!(v.__str__(py).unwrap(), "42.0");
            assert_eq!(v.__format__(py, ".1f").unwrap(), "42.0");
            let int_dtype = pyo3::types::PyString::new(py, "int64")
                .to_owned()
                .into_any()
                .unbind();
            assert_eq!(
                v.astype(py, Some(int_dtype), false, None)
                    .unwrap()
                    .as_i64()
                    .unwrap(),
                42
            );
            assert!(
                (v.astype(py, None, false, None).unwrap().as_f64().unwrap() - 42.0).abs() < 1e-12
            );
            let scalar_shape = PyTuple::empty(py);
            let scalar_args = PyTuple::new(py, [scalar_shape]).unwrap();
            assert!(
                (v.reshape(py, &scalar_args, "C", None)
                    .unwrap()
                    .as_f64()
                    .unwrap()
                    - 42.0)
                    .abs()
                    < 1e-12
            );
            let vector_args = PyTuple::new(py, [1_i64]).unwrap();
            let vector = v.reshape(py, &vector_args, "C", None).unwrap();
            assert_eq!(vector.shape_dims(), vec![1]);
            assert_eq!(vector.as_f64_list().unwrap(), vec![42.0]);
            let flattened = v.flatten(py, "C", None).unwrap();
            assert_eq!(flattened.shape_dims(), vec![1]);
            assert_eq!(flattened.as_f64_list().unwrap(), vec![42.0]);
            assert_eq!(
                v.ravel(py, "C", None).unwrap().as_f64_list().unwrap(),
                vec![42.0]
            );
            assert!((v.squeeze(py, None).unwrap().as_f64().unwrap() - 42.0).abs() < 1e-12);
            let bad_device = pyo3::types::PyString::new(py, "gpu")
                .to_owned()
                .into_any()
                .unbind();
            assert!(v.astype(py, None, false, Some(bad_device)).is_err());
            let out_sharding = PyBool::new(py, true).to_owned().into_any().unbind();
            assert!(
                v.reshape(py, &vector_args, "C", Some(out_sharding))
                    .is_err()
            );
            let out_sharding = PyBool::new(py, true).to_owned().into_any().unbind();
            assert!(v.flatten(py, "C", Some(out_sharding)).is_err());
            if py.import("numpy").is_ok() {
                let array = v.__array__(py, None, None, None).unwrap();
                let array = array.bind(py);
                assert_eq!(
                    array
                        .getattr("shape")
                        .unwrap()
                        .extract::<Vec<usize>>()
                        .unwrap(),
                    Vec::<usize>::new()
                );
                assert!(
                    (array
                        .call_method0("item")
                        .unwrap()
                        .extract::<f64>()
                        .unwrap()
                        - 42.0)
                        .abs()
                        < 1e-12
                );

                let dtype = "float32".into_pyobject(py).unwrap().into_any().unbind();
                let array = v.__array__(py, Some(dtype), None, Some(true)).unwrap();
                assert_eq!(
                    array
                        .bind(py)
                        .getattr("dtype")
                        .unwrap()
                        .str()
                        .unwrap()
                        .to_str()
                        .unwrap(),
                    "float32"
                );
            }
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
            let out = PyBool::new(py, true).to_owned().into_any().unbind();
            assert!(v.round(0, Some(out)).is_err());
        });
        assert!((v.__float__().unwrap() - 42.0).abs() < 1e-12);
        assert!(v.__bool__().unwrap());
        assert!((v.as_f64().unwrap() - 42.0).abs() < 1e-12);

        let i = PyValue::scalar_i64(123);
        assert!(i.__bool__().unwrap());
        assert_eq!(i.real_part().unwrap().as_i64().unwrap(), 123);
        assert_eq!(i.imag_part().unwrap().as_i64().unwrap(), 0);
        assert_eq!(i.conj().unwrap().as_i64().unwrap(), 123);
        assert_eq!(i.conjugate().unwrap().as_i64().unwrap(), 123);
        assert_eq!(i.round(0, None).unwrap().as_i64().unwrap(), 123);
        assert!(i.round(-1, None).is_err());
        assert!(!PyValue::scalar_i64(0).__bool__().unwrap());
        let complex = PyValue::from_value(Value::scalar_complex128(3.0, -2.0));
        assert!((complex.real_part().unwrap().as_f64().unwrap() - 3.0).abs() < 1e-12);
        assert!((complex.imag_part().unwrap().as_f64().unwrap() + 2.0).abs() < 1e-12);
        let conjugated = complex.conjugate().unwrap();
        assert!((conjugated.real_part().unwrap().as_f64().unwrap() - 3.0).abs() < 1e-12);
        assert!((conjugated.imag_part().unwrap().as_f64().unwrap() - 2.0).abs() < 1e-12);
        let conj_alias = complex.conj().unwrap();
        assert!((conj_alias.imag_part().unwrap().as_f64().unwrap() - 2.0).abs() < 1e-12);
        let complex_rounded = complex.round(0, None).unwrap();
        assert!((complex_rounded.real_part().unwrap().as_f64().unwrap() - 3.0).abs() < 1e-12);
        assert!((complex_rounded.imag_part().unwrap().as_f64().unwrap() + 2.0).abs() < 1e-12);
        Python::with_gil(|py| {
            let value = i.__index__(py).unwrap();
            assert_eq!(value.bind(py).extract::<i64>().unwrap(), 123);
            let float_dtype = pyo3::types::PyString::new(py, "float64")
                .to_owned()
                .into_any()
                .unbind();
            assert!(
                (i.astype(py, Some(float_dtype), false, None)
                    .unwrap()
                    .as_f64()
                    .unwrap()
                    - 123.0)
                    .abs()
                    < 1e-12
            );
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
            floats.block_until_ready().unwrap().as_f64_list().unwrap(),
            vec![1.0, 2.5, 4.0]
        );
        assert!(floats.copy_to_host_async().is_ok());
        assert_eq!(
            floats.copy().unwrap().as_f64_list().unwrap(),
            vec![1.0, 2.5, 4.0]
        );
        let rounded_ties = PyValue::vector_f64(vec![10.5, 21.5, 12.5, 31.5])
            .unwrap()
            .round(0, None)
            .unwrap();
        assert_eq!(
            rounded_ties.as_f64_list().unwrap(),
            vec![10.0, 22.0, 12.0, 32.0]
        );
        let rounded_decimals = PyValue::vector_f64(vec![1.25, 3.35]).unwrap();
        let rounded_decimals = rounded_decimals.__round__(Some(1)).unwrap();
        let rounded_decimals = rounded_decimals.as_f64_list().unwrap();
        assert!((rounded_decimals[0] - 1.2).abs() < 1e-12);
        assert!((rounded_decimals[1] - 3.4).abs() < 1e-12);
        let rounded_ints = PyValue::vector_f64(vec![10.5, 21.5])
            .unwrap()
            .__round__(None)
            .unwrap();
        assert_eq!(rounded_ints.as_i64_list().unwrap(), vec![10, 22]);
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let int_dtype = pyo3::types::PyString::new(py, "int64")
                .to_owned()
                .into_any()
                .unbind();
            assert_eq!(
                floats
                    .astype(py, Some(int_dtype), false, None)
                    .unwrap()
                    .as_i64_list()
                    .unwrap(),
                vec![1, 2, 4]
            );
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
        let ints_t = ints.transpose_all_axes().unwrap();
        assert_eq!(ints_t.shape_dims(), vec![3]);
        assert_eq!(ints_t.as_i64_list().unwrap(), vec![1, 2, 3]);
        let ints_swapped = ints.swapaxes(0, 0).unwrap();
        assert_eq!(ints_swapped.shape_dims(), vec![3]);
        assert_eq!(ints_swapped.as_i64_list().unwrap(), vec![1, 2, 3]);
        assert!(ints.matrix_transpose().is_err());
        assert_eq!(
            ints.real_part().unwrap().as_i64_list().unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            ints.imag_part().unwrap().as_i64_list().unwrap(),
            vec![0, 0, 0]
        );
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
        let iterated = ints.leading_axis_values().unwrap();
        assert_eq!(iterated.len(), 3);
        assert_eq!(
            iterated
                .iter()
                .map(|value| value.as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(ints.axis0_value_at(0).unwrap().as_i64().unwrap(), 1);
        assert_eq!(ints.axis0_value_at(-1).unwrap().as_i64().unwrap(), 3);
        assert!(ints.axis0_value_at(3).is_err());
        Python::with_gil(|py| {
            assert!(ints.__int__(py).is_err());
            assert!(ints.__complex__(py).is_err());
            assert!(ints.__index__(py).is_err());
            let float_dtype = pyo3::types::PyString::new(py, "float64")
                .to_owned()
                .into_any()
                .unbind();
            assert_eq!(
                ints.astype(py, Some(float_dtype), false, None)
                    .unwrap()
                    .as_f64_list()
                    .unwrap(),
                vec![1.0, 2.0, 3.0]
            );
            let reshape_args = PyTuple::new(py, [3_i64, 1_i64]).unwrap();
            let reshaped = ints.reshape(py, &reshape_args, "C", None).unwrap();
            assert_eq!(reshaped.shape_dims(), vec![3, 1]);
            assert_eq!(reshaped.as_i64_list().unwrap(), vec![1, 2, 3]);
            let inferred_args = PyTuple::new(py, [-1_i64]).unwrap();
            let inferred = ints.reshape(py, &inferred_args, "C", None).unwrap();
            assert_eq!(inferred.shape_dims(), vec![3]);
            assert_eq!(inferred.as_i64_list().unwrap(), vec![1, 2, 3]);
            let expanded_args = PyTuple::new(py, [1_i64, 3_i64, 1_i64]).unwrap();
            let expanded = ints.reshape(py, &expanded_args, "C", None).unwrap();
            let squeezed = expanded.squeeze(py, None).unwrap();
            assert_eq!(squeezed.shape_dims(), vec![3]);
            assert_eq!(squeezed.as_i64_list().unwrap(), vec![1, 2, 3]);
            let axis0 = 0_isize.into_pyobject(py).unwrap().into_any().unbind();
            let squeezed_axis0 = expanded.squeeze(py, Some(axis0)).unwrap();
            assert_eq!(squeezed_axis0.shape_dims(), vec![3, 1]);
            let axes = PyTuple::new(py, [0_isize, 2_isize])
                .unwrap()
                .into_any()
                .unbind();
            let squeezed_axes = expanded.squeeze(py, Some(axes)).unwrap();
            assert_eq!(squeezed_axes.shape_dims(), vec![3]);
            let bad_axis = 1_isize.into_pyobject(py).unwrap().into_any().unbind();
            assert!(expanded.squeeze(py, Some(bad_axis)).is_err());
            let duplicate_axes = PyTuple::new(py, [0_isize, 0_isize])
                .unwrap()
                .into_any()
                .unbind();
            assert!(expanded.squeeze(py, Some(duplicate_axes)).is_err());
            assert_eq!(
                ints.flatten(py, "C", None).unwrap().as_i64_list().unwrap(),
                vec![1, 2, 3]
            );
            assert_eq!(
                ints.ravel(py, "C", None).unwrap().as_i64_list().unwrap(),
                vec![1, 2, 3]
            );
            let shards = ints.addressable_shards().unwrap();
            assert_eq!(shards.len(), 1);
            let shard = shards.first().unwrap();
            assert_eq!(shard.data().unwrap().as_i64_list().unwrap(), vec![1, 2, 3]);
            let index = shard.index(py).unwrap();
            let index = index.bind(py).downcast::<PyTuple>().unwrap();
            assert_eq!(index.len(), 1);
            assert!(index.get_item(0).unwrap().downcast::<PySlice>().is_ok());
            let tail = ints.axis0_slice(&PySlice::new(py, 1, 3, 1)).unwrap();
            assert_eq!(tail.shape_dims(), vec![2]);
            assert_eq!(tail.as_i64_list().unwrap(), vec![2, 3]);
            let reversed = ints.axis0_slice(&PySlice::new(py, 2, -4, -1)).unwrap();
            assert_eq!(reversed.shape_dims(), vec![3]);
            assert_eq!(reversed.as_i64_list().unwrap(), vec![3, 2, 1]);
            let empty = ints.axis0_slice(&PySlice::new(py, 3, 3, 1)).unwrap();
            assert_eq!(empty.shape_dims(), vec![0]);
            assert_eq!(empty.as_i64_list().unwrap(), Vec::<i64>::new());
            let transpose_arg = PyTuple::new(py, [0_isize]).unwrap();
            let transpose_args = PyTuple::new(py, [transpose_arg]).unwrap();
            let ints_explicit_t = ints.transpose(&transpose_args).unwrap();
            assert_eq!(ints_explicit_t.shape_dims(), vec![3]);
            assert_eq!(ints_explicit_t.as_i64_list().unwrap(), vec![1, 2, 3]);
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

        let matrix = PyValue::from_value(Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                    Literal::I64(5),
                    Literal::I64(6),
                ],
            )
            .unwrap(),
        ));
        let matrix_t = matrix.transpose_all_axes().unwrap();
        assert_eq!(matrix_t.shape_dims(), vec![3, 2]);
        assert_eq!(matrix_t.as_i64_list().unwrap(), vec![1, 4, 2, 5, 3, 6]);
        let matrix_mt = matrix.matrix_transpose().unwrap();
        assert_eq!(matrix_mt.shape_dims(), vec![3, 2]);
        assert_eq!(matrix_mt.as_i64_list().unwrap(), vec![1, 4, 2, 5, 3, 6]);
        let matrix_swapped = matrix.swapaxes(0, 1).unwrap();
        assert_eq!(matrix_swapped.shape_dims(), vec![3, 2]);
        assert_eq!(
            matrix_swapped.as_i64_list().unwrap(),
            vec![1, 4, 2, 5, 3, 6]
        );
        let matrix_swapped_negative = matrix.swapaxes(-1, -2).unwrap();
        assert_eq!(matrix_swapped_negative.shape_dims(), vec![3, 2]);
        assert_eq!(
            matrix_swapped_negative.as_i64_list().unwrap(),
            vec![1, 4, 2, 5, 3, 6]
        );
        let matrix_swap_noop = matrix.swapaxes(0, 0).unwrap();
        assert_eq!(matrix_swap_noop.shape_dims(), vec![2, 3]);
        assert_eq!(
            matrix_swap_noop.as_i64_list().unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert!(matrix.swapaxes(0, 2).is_err());
        assert_eq!(
            matrix.real_part().unwrap().as_i64_list().unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert_eq!(
            matrix.imag_part().unwrap().as_i64_list().unwrap(),
            vec![0, 0, 0, 0, 0, 0]
        );
        let complex_vector = PyValue::from_value(Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(1.5, -2.0),
                    Literal::from_complex64(-3.25, 4.5),
                ],
            )
            .unwrap(),
        ));
        assert_eq!(
            complex_vector.real_part().unwrap().as_f64_list().unwrap(),
            vec![1.5, -3.25]
        );
        assert_eq!(
            complex_vector.imag_part().unwrap().as_f64_list().unwrap(),
            vec![-2.0, 4.5]
        );
        let complex_conjugated = complex_vector.conjugate().unwrap();
        assert_eq!(
            complex_conjugated
                .imag_part()
                .unwrap()
                .as_f64_list()
                .unwrap(),
            vec![2.0, -4.5]
        );
        assert_eq!(
            complex_vector
                .conj()
                .unwrap()
                .imag_part()
                .unwrap()
                .as_f64_list()
                .unwrap(),
            vec![2.0, -4.5]
        );
        Python::with_gil(|py| {
            let axes = PyTuple::new(py, [1_isize, 0_isize]).unwrap();
            let explicit_args = PyTuple::new(py, [axes]).unwrap();
            let matrix_explicit_t = matrix.transpose(&explicit_args).unwrap();
            assert_eq!(matrix_explicit_t.shape_dims(), vec![3, 2]);
            assert_eq!(
                matrix_explicit_t.as_i64_list().unwrap(),
                vec![1, 4, 2, 5, 3, 6]
            );

            let vararg_args = PyTuple::new(py, [1_isize, 0_isize]).unwrap();
            let matrix_vararg_t = matrix.transpose(&vararg_args).unwrap();
            assert_eq!(matrix_vararg_t.shape_dims(), vec![3, 2]);
            assert_eq!(
                matrix_vararg_t.as_i64_list().unwrap(),
                vec![1, 4, 2, 5, 3, 6]
            );

            let duplicate_args = PyTuple::new(py, [0_isize, 0_isize]).unwrap();
            assert!(matrix.transpose(&duplicate_args).is_err());
            let short_args = PyTuple::new(py, [0_isize]).unwrap();
            assert!(matrix.transpose(&short_args).is_err());

            let flat_args = PyTuple::new(py, [6_i64]).unwrap();
            let flat = matrix.reshape(py, &flat_args, "C", None).unwrap();
            assert_eq!(flat.shape_dims(), vec![6]);
            assert_eq!(flat.as_i64_list().unwrap(), vec![1, 2, 3, 4, 5, 6]);
            let f_shape = PyTuple::new(py, [3_i64, 2_i64]).unwrap();
            let f_args = PyTuple::new(py, [f_shape]).unwrap();
            let fortran = matrix.reshape(py, &f_args, "F", None).unwrap();
            assert_eq!(fortran.shape_dims(), vec![3, 2]);
            assert_eq!(fortran.as_i64_list().unwrap(), vec![1, 5, 4, 3, 2, 6]);
            assert_eq!(
                matrix
                    .flatten(py, "C", None)
                    .unwrap()
                    .as_i64_list()
                    .unwrap(),
                vec![1, 2, 3, 4, 5, 6]
            );
            assert_eq!(
                matrix
                    .flatten(py, "F", None)
                    .unwrap()
                    .as_i64_list()
                    .unwrap(),
                vec![1, 4, 2, 5, 3, 6]
            );
            assert_eq!(
                matrix.ravel(py, "F", None).unwrap().as_i64_list().unwrap(),
                vec![1, 4, 2, 5, 3, 6]
            );
            assert!(matrix.flatten(py, "A", None).is_err());
            assert!(matrix.ravel(py, "K", None).is_err());
            assert!(matrix.reshape(py, &f_args, "A", None).is_err());
            assert!(matrix.reshape(py, &f_args, "K", None).is_err());
            let incompatible = PyTuple::new(py, [4_i64]).unwrap();
            assert!(matrix.reshape(py, &incompatible, "C", None).is_err());
            let repeated_infer = PyTuple::new(py, [-1_i64, -1_i64]).unwrap();
            assert!(matrix.reshape(py, &repeated_infer, "C", None).is_err());
        });

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

        pyo3::prepare_freethreaded_python();
        let updated = Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("shape", vec![4_u32])?;
            kwargs.set_item("dtype", "I64")?;
            kwargs.set_item("weak_type", true)?;
            kwargs.set_item("is_ref", true)?;
            meta.update(Some(&kwargs))
        })
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

            let preserve_vma_meta = vma_meta.update(None).unwrap();
            let preserved_vma = preserve_vma_meta.vma().unwrap();
            assert!(preserved_vma.bind(py).contains("data").unwrap());

            let update_kwargs = PyDict::new(py);
            update_kwargs
                .set_item("vma", PySet::new(py, ["batch"]).unwrap())
                .unwrap();
            let updated_vma_meta = vma_meta.update(Some(&update_kwargs)).unwrap();
            let updated_vma = updated_vma_meta.vma().unwrap();
            let updated_vma = updated_vma.bind(py);
            assert!(updated_vma.contains("batch").unwrap());
            assert!(!updated_vma.contains("data").unwrap());

            let clear_kwargs = PyDict::new(py);
            clear_kwargs.set_item("vma", py.None()).unwrap();
            let cleared_vma_meta = vma_meta.update(Some(&clear_kwargs)).unwrap();
            assert!(cleared_vma_meta.vma().is_none());

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
        let ready_scalar = block_until_ready(put_scalar).unwrap();
        assert!((ready_scalar.as_f64().unwrap() - 3.5).abs() < 1e-12);
        let host_scalar = device_get(ready_scalar).unwrap();
        assert!((host_scalar.as_f64().unwrap() - 3.5).abs() < 1e-12);

        let vector = PyValue::vector_i64(vec![1, 2, 3]).unwrap();
        let sharded_vector = device_put_sharded(vec![vector.clone()], vec![cpu_device()]).unwrap();
        assert_eq!(sharded_vector.as_i64_list().unwrap(), vec![1, 2, 3]);
        assert!(device_put_sharded(vec![vector.clone()], Vec::new()).is_err());
        assert!(device_put_sharded(Vec::new(), vec![cpu_device()]).is_err());
        let host_vector = device_get(block_until_ready(device_put(vector)).unwrap()).unwrap();
        assert_eq!(host_vector.shape_dims(), vec![3]);
        assert_eq!(host_vector.dtype(), "I64");
        assert_eq!(host_vector.as_i64_list().unwrap(), vec![1, 2, 3]);

        let copied_vector = copy_to_host_async(host_vector).unwrap();
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
