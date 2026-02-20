//! Value ↔ FfiBuffer marshalling across the FFI boundary.
//!
//! Converts FrankenJAX `Value` types to contiguous byte buffers for FFI,
//! and unmarshals results back to `Value` types after the call.

use fj_core::{DType, Literal, Shape, TensorValue, Value};

use crate::buffer::{FfiBuffer, dtype_size_bytes};
use crate::error::FfiError;

/// Marshal a `Value` into an `FfiBuffer` for passing to an external function.
pub fn value_to_buffer(value: &Value) -> Result<FfiBuffer, FfiError> {
    match value {
        Value::Scalar(lit) => scalar_to_buffer(lit),
        Value::Tensor(tv) => tensor_to_buffer(tv),
    }
}

/// Unmarshal an `FfiBuffer` back into a `Value`.
pub fn buffer_to_value(buffer: &FfiBuffer) -> Result<Value, FfiError> {
    if buffer.shape().is_empty() {
        // Scalar
        let lit = bytes_to_literal(buffer.as_bytes(), buffer.dtype())?;
        Ok(Value::Scalar(lit))
    } else {
        // Tensor
        let shape = Shape {
            dims: buffer.shape().iter().map(|&d| d as u32).collect(),
        };
        let num_elements: usize = buffer.shape().iter().product();
        let elem_size = dtype_size_bytes(buffer.dtype())?;
        let mut elements = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let offset = i * elem_size;
            let lit = bytes_to_literal(
                &buffer.as_bytes()[offset..offset + elem_size],
                buffer.dtype(),
            )?;
            elements.push(lit);
        }
        Ok(Value::Tensor(TensorValue {
            dtype: buffer.dtype(),
            shape,
            elements,
        }))
    }
}

fn scalar_to_buffer(lit: &Literal) -> Result<FfiBuffer, FfiError> {
    match lit {
        Literal::F64Bits(bits) => FfiBuffer::new(bits.to_ne_bytes().to_vec(), vec![], DType::F64),
        Literal::I64(v) => FfiBuffer::new(v.to_ne_bytes().to_vec(), vec![], DType::I64),
        Literal::Bool(v) => FfiBuffer::new(vec![*v as u8], vec![], DType::Bool),
    }
}

fn tensor_to_buffer(tv: &TensorValue) -> Result<FfiBuffer, FfiError> {
    let elem_size = dtype_size_bytes(tv.dtype)?;
    let mut data = Vec::with_capacity(tv.elements.len() * elem_size);
    for lit in &tv.elements {
        match lit {
            Literal::F64Bits(bits) => data.extend_from_slice(&bits.to_ne_bytes()),
            Literal::I64(v) => data.extend_from_slice(&v.to_ne_bytes()),
            Literal::Bool(v) => data.push(*v as u8),
        }
    }
    let shape: Vec<usize> = tv.shape.dims.iter().map(|&d| d as usize).collect();
    FfiBuffer::new(data, shape, tv.dtype)
}

fn bytes_to_literal(bytes: &[u8], dtype: DType) -> Result<Literal, FfiError> {
    match dtype {
        DType::F64 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 8,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::F64Bits(u64::from_ne_bytes(arr)))
        }
        DType::I64 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 8,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::I64(i64::from_ne_bytes(arr)))
        }
        DType::F32 | DType::I32 => Err(FfiError::UnsupportedDtype { dtype }),
        DType::Bool => {
            if bytes.is_empty() {
                return Err(FfiError::BufferMismatch {
                    buffer_index: 0,
                    expected_bytes: 1,
                    actual_bytes: 0,
                });
            }
            Ok(Literal::Bool(bytes[0] != 0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_scalar_f64() {
        let val = Value::Scalar(Literal::F64Bits(42.0f64.to_bits()));
        let buf = value_to_buffer(&val).unwrap();
        let restored = buffer_to_value(&buf).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn roundtrip_scalar_i64() {
        let val = Value::Scalar(Literal::I64(99));
        let buf = value_to_buffer(&val).unwrap();
        let restored = buffer_to_value(&buf).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn roundtrip_scalar_bool() {
        for b in [true, false] {
            let val = Value::Scalar(Literal::Bool(b));
            let buf = value_to_buffer(&val).unwrap();
            let restored = buffer_to_value(&buf).unwrap();
            assert_eq!(val, restored);
        }
    }

    #[test]
    fn roundtrip_tensor_f64() {
        let val = Value::Tensor(TensorValue {
            dtype: DType::F64,
            shape: Shape { dims: vec![3] },
            elements: vec![
                Literal::F64Bits(1.0f64.to_bits()),
                Literal::F64Bits(2.0f64.to_bits()),
                Literal::F64Bits(3.0f64.to_bits()),
            ],
        });
        let buf = value_to_buffer(&val).unwrap();
        assert_eq!(buf.size(), 24);
        assert_eq!(buf.shape(), &[3]);
        let restored = buffer_to_value(&buf).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn roundtrip_tensor_i64_matrix() {
        let val = Value::Tensor(TensorValue {
            dtype: DType::I64,
            shape: Shape { dims: vec![2, 2] },
            elements: vec![
                Literal::I64(10),
                Literal::I64(20),
                Literal::I64(30),
                Literal::I64(40),
            ],
        });
        let buf = value_to_buffer(&val).unwrap();
        assert_eq!(buf.size(), 32);
        assert_eq!(buf.shape(), &[2, 2]);
        let restored = buffer_to_value(&buf).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn roundtrip_tensor_bool() {
        let val = Value::Tensor(TensorValue {
            dtype: DType::Bool,
            shape: Shape { dims: vec![4] },
            elements: vec![
                Literal::Bool(true),
                Literal::Bool(false),
                Literal::Bool(true),
                Literal::Bool(false),
            ],
        });
        let buf = value_to_buffer(&val).unwrap();
        assert_eq!(buf.size(), 4);
        let restored = buffer_to_value(&buf).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn buffer_to_value_f32_unsupported() {
        let buf = FfiBuffer::new(vec![0u8; 4], vec![], DType::F32).unwrap();
        let err = buffer_to_value(&buf).unwrap_err();
        assert!(matches!(
            err,
            FfiError::UnsupportedDtype { dtype: DType::F32 }
        ));
    }
}
