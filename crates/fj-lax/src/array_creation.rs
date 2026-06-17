//! Array creation functions matching JAX's jnp module.
//!
//! These functions create new arrays with specified shapes and values.

use fj_core::{DType, Literal, Shape, TensorValue, Value, ValueError};

fn checked_shape_and_size(dims: &[u32]) -> Result<(Shape, usize), ValueError> {
    let shape = Shape {
        dims: dims.to_vec(),
    };
    let count = shape
        .element_count()
        .ok_or_else(|| ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;
    let size = usize::try_from(count).map_err(|_| ValueError::ShapeOverflow {
        shape: shape.clone(),
    })?;
    Ok((shape, size))
}

fn linspace_elements(start: f64, stop: f64, num: usize, endpoint: bool) -> Vec<Literal> {
    if num == 0 {
        return Vec::new();
    }

    let step = if num == 1 {
        0.0
    } else if endpoint {
        (stop - start) / (num - 1) as f64
    } else {
        (stop - start) / num as f64
    };

    let mut elements: Vec<Literal> = (0..num)
        .map(|i| Literal::from_f64(start + step * i as f64))
        .collect();

    // JAX/NumPy pin the final sample to exactly `stop` when `endpoint=True`,
    // overriding the `start + step*(num-1)` value (which drifts by up to an ULP).
    // `jnp.linspace` does `out = out.at[-1].set(stop)`; reproduce it bit-for-bit.
    if endpoint && num > 1 {
        elements[num - 1] = Literal::from_f64(stop);
    }

    elements
}

/// Create an array filled with zeros.
///
/// Matches `jnp.zeros(shape, dtype)`.
pub fn zeros(shape: &[u32], dtype: DType) -> Result<Value, ValueError> {
    let (shape, size) = checked_shape_and_size(shape)?;
    let elements = match dtype {
        DType::F32 => vec![Literal::from_f32(0.0); size],
        DType::F64 => vec![Literal::from_f64(0.0); size],
        DType::I32 => vec![Literal::I64(0); size],
        DType::I64 => vec![Literal::I64(0); size],
        DType::U32 => vec![Literal::U32(0); size],
        DType::U64 => vec![Literal::U64(0); size],
        DType::Bool => vec![Literal::Bool(false); size],
        DType::Complex64 => vec![Literal::from_complex64(0.0, 0.0); size],
        DType::Complex128 => vec![Literal::from_complex128(0.0, 0.0); size],
        DType::F16 => vec![Literal::from_f16_f32(0.0); size],
        DType::BF16 => vec![Literal::from_bf16_f32(0.0); size],
    };
    let tensor = TensorValue::new(dtype, shape, elements)?;
    Ok(Value::Tensor(tensor))
}

/// Create an array filled with ones.
///
/// Matches `jnp.ones(shape, dtype)`.
pub fn ones(shape: &[u32], dtype: DType) -> Result<Value, ValueError> {
    let (shape, size) = checked_shape_and_size(shape)?;
    let elements = match dtype {
        DType::F32 => vec![Literal::from_f32(1.0); size],
        DType::F64 => vec![Literal::from_f64(1.0); size],
        DType::I32 => vec![Literal::I64(1); size],
        DType::I64 => vec![Literal::I64(1); size],
        DType::U32 => vec![Literal::U32(1); size],
        DType::U64 => vec![Literal::U64(1); size],
        DType::Bool => vec![Literal::Bool(true); size],
        DType::Complex64 => vec![Literal::from_complex64(1.0, 0.0); size],
        DType::Complex128 => vec![Literal::from_complex128(1.0, 0.0); size],
        DType::F16 => vec![Literal::from_f16_f32(1.0); size],
        DType::BF16 => vec![Literal::from_bf16_f32(1.0); size],
    };
    let tensor = TensorValue::new(dtype, shape, elements)?;
    Ok(Value::Tensor(tensor))
}

/// Create an array filled with a specified value.
///
/// Matches `jnp.full(shape, fill_value, dtype)`.
pub fn full(shape: &[u32], fill_value: f64, dtype: DType) -> Result<Value, ValueError> {
    let (shape, size) = checked_shape_and_size(shape)?;
    let elements = match dtype {
        DType::F32 => vec![Literal::from_f32(fill_value as f32); size],
        DType::F64 => vec![Literal::from_f64(fill_value); size],
        DType::I32 => vec![Literal::I64(fill_value as i64); size],
        DType::I64 => vec![Literal::I64(fill_value as i64); size],
        DType::U32 => vec![Literal::U32(fill_value as u32); size],
        DType::U64 => vec![Literal::U64(fill_value as u64); size],
        DType::Bool => vec![Literal::Bool(fill_value != 0.0); size],
        DType::Complex64 => vec![Literal::from_complex64(fill_value as f32, 0.0); size],
        DType::Complex128 => vec![Literal::from_complex128(fill_value, 0.0); size],
        DType::F16 => vec![Literal::from_f16_f64(fill_value); size],
        DType::BF16 => vec![Literal::from_bf16_f64(fill_value); size],
    };
    let tensor = TensorValue::new(dtype, shape, elements)?;
    Ok(Value::Tensor(tensor))
}

/// Create a 2D identity matrix.
///
/// Matches `jnp.eye(n, m, k, dtype)` where k is the diagonal offset.
/// Supports F64, F32, I64, I32 dtypes.
pub fn eye(n: u32, m: Option<u32>, k: i32, dtype: DType) -> Result<Value, ValueError> {
    let m = m.unwrap_or(n);
    let size = (n as usize) * (m as usize);

    let (zero_lit, one_lit) = match dtype {
        DType::F64 => (Literal::from_f64(0.0), Literal::from_f64(1.0)),
        DType::F32 => (Literal::from_f32(0.0), Literal::from_f32(1.0)),
        DType::I64 => (Literal::I64(0), Literal::I64(1)),
        DType::I32 => (Literal::I64(0), Literal::I64(1)),
        _ => {
            // For other dtypes, use F64
            (Literal::from_f64(0.0), Literal::from_f64(1.0))
        }
    };

    let mut elements = vec![zero_lit; size];

    for i in 0..n as i32 {
        let j = i + k;
        if j >= 0 && j < m as i32 {
            let idx = (i as usize) * (m as usize) + (j as usize);
            elements[idx] = one_lit;
        }
    }

    let tensor = TensorValue::new(dtype, Shape { dims: vec![n, m] }, elements)?;
    Ok(Value::Tensor(tensor))
}

/// Create evenly spaced values within an interval.
///
/// Matches `jnp.linspace(start, stop, num, endpoint)`.
pub fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> Result<Value, ValueError> {
    if num == 0 {
        let tensor = TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![])?;
        return Ok(Value::Tensor(tensor));
    }

    let elements = linspace_elements(start, stop, num, endpoint);
    let tensor = TensorValue::new(
        DType::F64,
        Shape {
            dims: vec![num as u32],
        },
        elements,
    )?;
    Ok(Value::Tensor(tensor))
}

/// Create evenly spaced values within a half-open interval.
///
/// Matches `jnp.arange(start, stop, step)`.
/// Panics if step is zero.
pub fn arange(start: f64, stop: f64, step: f64) -> Result<Value, ValueError> {
    assert!(step != 0.0, "arange step cannot be zero");

    // JAX/NumPy size the result once as `ceil((stop - start) / step)` (clamped to
    // >= 0) and compute each value as `start + step*i`. The previous iterative
    // form (`current += step`) accumulates rounding error, so it could cross the
    // half-open boundary at a different index and return a DIFFERENT element
    // count than JAX — e.g. arange(0, 0.9, 0.3): 0.3+0.3+0.3 = 0.8999… < 0.9 so
    // the loop emitted 4 elements, whereas JAX's ceil(0.9/0.3)=ceil(2.999…)=3.
    let raw = ((stop - start) / step).ceil();
    let n = if raw.is_finite() && raw > 0.0 {
        raw as usize
    } else {
        0
    };

    let elements: Vec<Literal> = (0..n)
        .map(|i| Literal::from_f64(start + step * i as f64))
        .collect();

    let tensor = TensorValue::new(
        DType::F64,
        Shape {
            dims: vec![n as u32],
        },
        elements,
    )?;
    Ok(Value::Tensor(tensor))
}

/// Create evenly spaced values on a log scale.
///
/// Matches `jnp.logspace(start, stop, num, endpoint, base)`.
pub fn logspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
    base: f64,
) -> Result<Value, ValueError> {
    let source = linspace_elements(start, stop, num, endpoint);
    let elements: Vec<Literal> = source
        .into_iter()
        .map(|lit| {
            if let Some(v) = lit.as_f64() {
                Literal::from_f64(base.powf(v))
            } else {
                lit
            }
        })
        .collect();

    let tensor = TensorValue::new(
        DType::F64,
        Shape {
            dims: vec![num as u32],
        },
        elements,
    )?;
    Ok(Value::Tensor(tensor))
}

/// Create a diagonal matrix from a 1D array.
///
/// Matches `jnp.diag(v, k)` for 1D input.
pub fn diag(v: &[f64], k: i32) -> Result<Value, ValueError> {
    let n = v.len();
    let size = n as i32 + k.abs();
    let mat_size = (size as usize) * (size as usize);
    let mut elements = vec![Literal::from_f64(0.0); mat_size];

    for (i, &val) in v.iter().enumerate() {
        let row = if k >= 0 {
            i
        } else {
            i + k.unsigned_abs() as usize
        };
        let col = if k >= 0 { i + k as usize } else { i };
        if row < size as usize && col < size as usize {
            elements[row * (size as usize) + col] = Literal::from_f64(val);
        }
    }

    let tensor = TensorValue::new(
        DType::F64,
        Shape {
            dims: vec![size as u32, size as u32],
        },
        elements,
    )?;
    Ok(Value::Tensor(tensor))
}

/// Extract the upper triangle of a matrix.
///
/// Matches `jnp.triu(m, k)` where k is the diagonal offset.
/// Elements below the k-th diagonal are zeroed.
pub fn triu(a: &[f64], m: usize, n: usize, k: i32) -> Vec<f64> {
    let mut result = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            if (j as i32) >= (i as i32) + k {
                result[i * n + j] = a[i * n + j];
            }
        }
    }
    result
}

/// Extract the lower triangle of a matrix.
///
/// Matches `jnp.tril(m, k)` where k is the diagonal offset.
/// Elements above the k-th diagonal are zeroed.
pub fn tril(a: &[f64], m: usize, n: usize, k: i32) -> Vec<f64> {
    let mut result = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            if (j as i32) <= (i as i32) + k {
                result[i * n + j] = a[i * n + j];
            }
        }
    }
    result
}

/// Extract the trace (sum of diagonal elements) of a matrix.
///
/// Matches `jnp.trace(a, offset)`.
pub fn trace(a: &[f64], m: usize, n: usize, offset: i32) -> f64 {
    let mut sum = 0.0;
    for i in 0..m {
        let j = (i as i32) + offset;
        if j >= 0 && (j as usize) < n {
            sum += a[i * n + j as usize];
        }
    }
    sum
}

/// Extract a diagonal from a matrix.
///
/// Matches `jnp.diagonal(a, offset)`.
pub fn diagonal(a: &[f64], m: usize, n: usize, offset: i32) -> Vec<f64> {
    let mut diag = Vec::new();
    for i in 0..m {
        let j = (i as i32) + offset;
        if j >= 0 && (j as usize) < n {
            diag.push(a[i * n + j as usize]);
        }
    }
    diag
}

/// Flip array along an axis.
///
/// Matches `jnp.flip(m, axis)` for a 2D array.
pub fn flip_2d(a: &[f64], m: usize, n: usize, axis: usize) -> Vec<f64> {
    let mut result = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let (src_i, src_j) = if axis == 0 {
                (m - 1 - i, j)
            } else {
                (i, n - 1 - j)
            };
            result[i * n + j] = a[src_i * n + src_j];
        }
    }
    result
}

/// Roll array elements along an axis.
///
/// Matches `jnp.roll(a, shift, axis)` for a 1D array.
pub fn roll_1d(a: &[f64], shift: i32) -> Vec<f64> {
    if a.is_empty() {
        return Vec::new();
    }
    let n = a.len();
    let shift = ((shift % n as i32) + n as i32) as usize % n;
    let mut result = vec![0.0; n];
    for i in 0..n {
        result[(i + shift) % n] = a[i];
    }
    result
}

/// Create a triangular matrix (lower or upper) filled with ones.
///
/// Matches `jnp.tri(n, m, k)`.
pub fn tri(n: usize, m: usize, k: i32) -> Vec<f64> {
    let mut result = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            if (j as i32) <= (i as i32) + k {
                result[i * m + j] = 1.0;
            }
        }
    }
    result
}

/// Stack arrays along a new axis.
///
/// Matches `jnp.stack(arrays, axis)` for 1D arrays along axis 0.
/// Returns (result, new_shape) where new_shape is [count, len].
pub fn stack_1d(arrays: &[&[f64]]) -> (Vec<f64>, Vec<usize>) {
    if arrays.is_empty() {
        return (Vec::new(), vec![0, 0]);
    }
    let len = arrays[0].len();
    let count = arrays.len();
    let mut result = Vec::with_capacity(count * len);
    for arr in arrays {
        result.extend_from_slice(arr);
    }
    (result, vec![count, len])
}

/// Vertically stack (row-wise) 1D or 2D arrays.
///
/// Matches `jnp.vstack(arrays)` for 1D arrays.
pub fn vstack_1d(arrays: &[&[f64]]) -> Vec<f64> {
    let mut result = Vec::new();
    for arr in arrays {
        result.extend_from_slice(arr);
    }
    result
}

/// Horizontally stack (column-wise) 1D arrays.
///
/// Matches `jnp.hstack(arrays)` for 1D arrays.
pub fn hstack_1d(arrays: &[&[f64]]) -> Vec<f64> {
    let mut result = Vec::new();
    for arr in arrays {
        result.extend_from_slice(arr);
    }
    result
}

/// Repeat elements of an array.
///
/// Matches `jnp.repeat(a, repeats)` for a 1D array.
pub fn repeat_1d(a: &[f64], repeats: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(a.len() * repeats);
    for &val in a {
        for _ in 0..repeats {
            result.push(val);
        }
    }
    result
}

/// Tile an array.
///
/// Matches `jnp.tile(a, reps)` for a 1D array.
pub fn tile_1d(a: &[f64], reps: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(a.len() * reps);
    for _ in 0..reps {
        result.extend_from_slice(a);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_f64(v: &Value) -> Vec<f64> {
        match v {
            Value::Tensor(t) => t.elements.iter().filter_map(|lit| lit.as_f64()).collect(),
            Value::Scalar(lit) => lit.as_f64().into_iter().collect(),
        }
    }

    #[test]
    fn test_zeros_1d() {
        let v = zeros(&[5], DType::F64).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![0.0; 5]);
    }

    #[test]
    fn test_zeros_2d() {
        let v = zeros(&[2, 3], DType::F64).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals.len(), 6);
        assert!(vals.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_1d() {
        let v = ones(&[4], DType::F64).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![1.0; 4]);
    }

    #[test]
    fn test_full_value() {
        let v = full(&[3], 42.0, DType::F64).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![42.0; 3]);
    }

    #[test]
    fn array_creation_rejects_shape_overflow_before_allocation() {
        let huge = [u32::MAX, u32::MAX, 2];
        assert!(matches!(
            zeros(&huge, DType::F64),
            Err(ValueError::ShapeOverflow { .. })
        ));
        assert!(matches!(
            ones(&huge, DType::I32),
            Err(ValueError::ShapeOverflow { .. })
        ));
        assert!(matches!(
            full(&huge, 7.0, DType::Bool),
            Err(ValueError::ShapeOverflow { .. })
        ));
    }

    #[test]
    fn test_eye_square() {
        let v = eye(3, None, 0, DType::F64).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_eye_rectangular() {
        let v = eye(2, Some(3), 0, DType::F64).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_eye_offset_positive() {
        let v = eye(3, None, 1, DType::F64).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_eye_offset_negative() {
        let v = eye(3, None, -1, DType::F64).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_linspace_basic() {
        let v = linspace(0.0, 1.0, 5, true).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals.len(), 5);
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_no_endpoint() {
        let v = linspace(0.0, 1.0, 5, false).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals.len(), 5);
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[4] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_single() {
        let v = linspace(5.0, 10.0, 1, true).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![5.0]);
    }

    #[test]
    fn test_linspace_empty() {
        let v = linspace(0.0, 1.0, 0, true).unwrap();
        let vals = extract_f64(&v);
        assert!(vals.is_empty());
    }

    #[test]
    fn test_arange_basic() {
        let v = arange(0.0, 5.0, 1.0).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_arange_fractional() {
        let v = arange(0.0, 1.0, 0.25).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals.len(), 4);
    }

    #[test]
    fn test_arange_negative_step() {
        let v = arange(5.0, 0.0, -1.0).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "step cannot be zero")]
    fn test_arange_zero_step_panics() {
        let _ = arange(0.0, 5.0, 0.0);
    }

    #[test]
    fn test_linspace_endpoint_is_bit_exact_stop() {
        // JAX pins out[-1] to exactly `stop` (endpoint=True). For stops whose
        // step doesn't divide evenly, `start + step*(num-1)` drifts by an ULP;
        // the pin must make the final element bit-identical to `stop`.
        for &(start, stop, num) in &[(0.0, 1.0, 5usize), (0.0, 2.3, 7), (-1.0, 3.7, 13)] {
            let v = linspace(start, stop, num, true).unwrap();
            let vals = extract_f64(&v);
            assert_eq!(vals.len(), num);
            assert_eq!(
                vals[num - 1].to_bits(),
                stop.to_bits(),
                "linspace({start},{stop},{num}) endpoint not pinned to stop"
            );
        }
        // endpoint=false must NOT pin the last sample to stop.
        let open = extract_f64(&linspace(0.0, 1.0, 5, false).unwrap());
        assert!((open[4] - 0.8).abs() < 1e-12 && open[4] != 1.0);
        // num == 1 is just [start].
        assert_eq!(
            extract_f64(&linspace(5.0, 10.0, 1, true).unwrap()),
            vec![5.0]
        );
    }

    #[test]
    fn test_arange_count_matches_jax_ceil_formula() {
        // Element count is ceil((stop-start)/step), matching JAX/NumPy — NOT the
        // iterative-accumulation count. arange(0,0.9,0.3): ceil(0.9/0.3)=3 even
        // though 0.3+0.3+0.3 < 0.9 (which the old loop counted as 4).
        let v = extract_f64(&arange(0.0, 0.9, 0.3).unwrap());
        assert_eq!(
            v.len(),
            3,
            "arange(0,0.9,0.3) must have ceil(0.9/0.3)=3 elements"
        );
        // Values are start + step*i (exact for 0.25 stride), not accumulated.
        let q = extract_f64(&arange(0.0, 1.0, 0.25).unwrap());
        assert_eq!(q, vec![0.0, 0.25, 0.5, 0.75]);
        for (i, &val) in q.iter().enumerate() {
            assert_eq!(val.to_bits(), (0.0 + 0.25 * i as f64).to_bits());
        }
        // start == stop and overshoot-only ranges are empty.
        assert!(extract_f64(&arange(5.0, 5.0, 1.0).unwrap()).is_empty());
        assert!(extract_f64(&arange(5.0, 6.0, -1.0).unwrap()).is_empty());
    }

    #[test]
    fn test_logspace_basic() {
        let v = logspace(0.0, 2.0, 3, true, 10.0).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 10.0).abs() < 1e-10);
        assert!((vals[2] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_diag_basic() {
        let v = diag(&[1.0, 2.0, 3.0], 0).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_diag_positive_offset() {
        let v = diag(&[1.0, 2.0], 1).unwrap();
        let vals = extract_f64(&v);
        assert_eq!(vals.len(), 9);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        assert!((vals[5] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_triu_basic() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = triu(&a, 3, 3, 0);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn test_triu_offset() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = triu(&a, 3, 3, 1);
        assert_eq!(result, vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tril_basic() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = tril(&a, 3, 3, 0);
        assert_eq!(result, vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_tril_offset() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = tril(&a, 3, 3, -1);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]);
    }

    #[test]
    fn test_trace_basic() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tr = trace(&a, 3, 3, 0);
        assert!((tr - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_trace_offset() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tr = trace(&a, 3, 3, 1);
        assert!((tr - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_diagonal_basic() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let diag = diagonal(&a, 3, 3, 0);
        assert_eq!(diag, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_diagonal_offset() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let diag = diagonal(&a, 3, 3, 1);
        assert_eq!(diag, vec![2.0, 6.0]);
    }

    #[test]
    fn test_flip_2d_axis0() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = flip_2d(&a, 2, 3, 0);
        assert_eq!(result, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_flip_2d_axis1() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = flip_2d(&a, 2, 3, 1);
        assert_eq!(result, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_roll_1d_positive() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = roll_1d(&a, 2);
        assert_eq!(result, vec![4.0, 5.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_roll_1d_negative() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = roll_1d(&a, -2);
        assert_eq!(result, vec![3.0, 4.0, 5.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tri_basic() {
        let result = tri(3, 3, 0);
        assert_eq!(result, vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_tri_offset() {
        let result = tri(3, 3, 1);
        assert_eq!(result, vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_stack_1d() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let (result, shape) = stack_1d(&[&a, &b]);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, vec![2, 2]);
    }

    #[test]
    fn test_vstack_1d() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let result = vstack_1d(&[&a, &b]);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_repeat_1d() {
        let a = [1.0, 2.0];
        let result = repeat_1d(&a, 3);
        assert_eq!(result, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_tile_1d() {
        let a = [1.0, 2.0];
        let result = tile_1d(&a, 3);
        assert_eq!(result, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }
}
