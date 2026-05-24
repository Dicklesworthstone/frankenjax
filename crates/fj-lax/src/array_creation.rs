//! Array creation functions matching JAX's jnp module.
//!
//! These functions create new arrays with specified shapes and values.

use fj_core::{DType, Literal, Shape, TensorValue, Value, ValueError};

/// Create an array filled with zeros.
///
/// Matches `jnp.zeros(shape, dtype)`.
pub fn zeros(shape: &[u32], dtype: DType) -> Result<Value, ValueError> {
    let size = shape.iter().map(|&d| d as usize).product();
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
    let tensor = TensorValue::new(
        dtype,
        Shape {
            dims: shape.to_vec(),
        },
        elements,
    )?;
    Ok(Value::Tensor(tensor))
}

/// Create an array filled with ones.
///
/// Matches `jnp.ones(shape, dtype)`.
pub fn ones(shape: &[u32], dtype: DType) -> Result<Value, ValueError> {
    let size = shape.iter().map(|&d| d as usize).product();
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
    let tensor = TensorValue::new(
        dtype,
        Shape {
            dims: shape.to_vec(),
        },
        elements,
    )?;
    Ok(Value::Tensor(tensor))
}

/// Create an array filled with a specified value.
///
/// Matches `jnp.full(shape, fill_value, dtype)`.
pub fn full(shape: &[u32], fill_value: f64, dtype: DType) -> Result<Value, ValueError> {
    let size = shape.iter().map(|&d| d as usize).product();
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
        DType::F16 => vec![Literal::from_f16_f32(fill_value as f32); size],
        DType::BF16 => vec![Literal::from_bf16_f32(fill_value as f32); size],
    };
    let tensor = TensorValue::new(
        dtype,
        Shape {
            dims: shape.to_vec(),
        },
        elements,
    )?;
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

    let mut elements = vec![zero_lit.clone(); size];

    for i in 0..n as i32 {
        let j = i + k;
        if j >= 0 && j < m as i32 {
            let idx = (i as usize) * (m as usize) + (j as usize);
            elements[idx] = one_lit.clone();
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

    let step = if num == 1 {
        0.0
    } else if endpoint {
        (stop - start) / (num - 1) as f64
    } else {
        (stop - start) / num as f64
    };

    let elements: Vec<Literal> = (0..num)
        .map(|i| Literal::from_f64(start + step * i as f64))
        .collect();

    let tensor = TensorValue::new(DType::F64, Shape { dims: vec![num as u32] }, elements)?;
    Ok(Value::Tensor(tensor))
}

/// Create evenly spaced values within a half-open interval.
///
/// Matches `jnp.arange(start, stop, step)`.
/// Panics if step is zero.
pub fn arange(start: f64, stop: f64, step: f64) -> Result<Value, ValueError> {
    assert!(step != 0.0, "arange step cannot be zero");

    let mut elements = Vec::new();
    let mut current = start;

    if step > 0.0 {
        while current < stop {
            elements.push(Literal::from_f64(current));
            current += step;
        }
    } else {
        while current > stop {
            elements.push(Literal::from_f64(current));
            current += step;
        }
    }

    let n = elements.len() as u32;
    let tensor = TensorValue::new(DType::F64, Shape { dims: vec![n] }, elements)?;
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
    let lin = linspace(start, stop, num, endpoint)?;
    let Value::Tensor(tensor) = lin else {
        panic!("linspace returned non-tensor");
    };

    let elements: Vec<Literal> = tensor
        .elements
        .iter()
        .map(|lit| {
            if let Some(v) = lit.as_f64() {
                Literal::from_f64(base.powf(v))
            } else {
                lit.clone()
            }
        })
        .collect();

    let tensor = TensorValue::new(DType::F64, tensor.shape, elements)?;
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
        let row = if k >= 0 { i } else { i + k.unsigned_abs() as usize };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_f64(v: &Value) -> Vec<f64> {
        match v {
            Value::Tensor(t) => t
                .elements
                .iter()
                .filter_map(|lit| lit.as_f64())
                .collect(),
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
}
