//! Value ↔ FfiBuffer marshalling across the FFI boundary.
//!
//! Converts FrankenJAX `Value` types to contiguous byte buffers for FFI,
//! and unmarshals results back to `Value` types after the call.

use fj_core::{DType, Literal, Shape, TensorValue, Value};

use crate::buffer::{FfiBuffer, dtype_size_bytes, validate_buffer_contents};
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
    validate_buffer_contents(0, buffer.as_bytes(), buffer.dtype())?;

    if buffer.shape().is_empty() {
        // Scalar
        let lit = bytes_to_literal(buffer.as_bytes(), buffer.dtype())?;
        Ok(Value::Scalar(lit))
    } else {
        // Tensor
        let shape = ffi_shape_to_core_shape(buffer.shape())?;
        let elem_size = dtype_size_bytes(buffer.dtype())?;
        let chunks = buffer.as_bytes().chunks_exact(elem_size);
        if !chunks.remainder().is_empty() {
            return Err(FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: buffer.as_bytes().len() - chunks.remainder().len(),
                actual_bytes: buffer.as_bytes().len(),
            });
        }
        let mut elements = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let lit = bytes_to_literal(chunk, buffer.dtype())?;
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
    let dtype = literal_storage_dtype(*lit);
    let elem_size = dtype_size_bytes(dtype)?;
    let mut data = Vec::with_capacity(elem_size);
    append_literal_as_dtype(&mut data, *lit, dtype)?;
    FfiBuffer::new(data, vec![], dtype)
}

fn tensor_to_buffer(tv: &TensorValue) -> Result<FfiBuffer, FfiError> {
    let elem_size = dtype_size_bytes(tv.dtype)?;
    let capacity =
        checked_data_capacity(tv.elements.len(), elem_size).ok_or(FfiError::BufferMismatch {
            buffer_index: 0,
            expected_bytes: usize::MAX,
            actual_bytes: 0,
        })?;
    let mut data = Vec::with_capacity(capacity);
    for lit in &tv.elements {
        append_literal_as_dtype(&mut data, *lit, tv.dtype)?;
    }
    let shape: Vec<usize> = tv.shape.dims.iter().map(|&d| d as usize).collect();
    FfiBuffer::new(data, shape, tv.dtype)
}

fn checked_data_capacity(element_count: usize, elem_size: usize) -> Option<usize> {
    element_count.checked_mul(elem_size)
}

fn ffi_shape_to_core_shape(shape: &[usize]) -> Result<Shape, FfiError> {
    let mut dims = Vec::with_capacity(shape.len());
    for (dimension_index, &dimension) in shape.iter().enumerate() {
        let dim = u32::try_from(dimension).map_err(|_| FfiError::ShapeDimensionOutOfRange {
            dimension_index,
            dimension,
            max_supported: u32::MAX as usize,
        })?;
        dims.push(dim);
    }
    Ok(Shape { dims })
}

fn literal_storage_dtype(lit: Literal) -> DType {
    match lit {
        Literal::BF16Bits(_) => DType::BF16,
        Literal::F16Bits(_) => DType::F16,
        Literal::F32Bits(_) => DType::F32,
        Literal::F64Bits(_) => DType::F64,
        Literal::I64(_) => DType::I64,
        Literal::U32(_) => DType::U32,
        Literal::U64(_) => DType::U64,
        Literal::Bool(_) => DType::Bool,
        Literal::Complex64Bits(..) => DType::Complex64,
        Literal::Complex128Bits(..) => DType::Complex128,
    }
}

fn append_literal_as_dtype(data: &mut Vec<u8>, lit: Literal, dtype: DType) -> Result<(), FfiError> {
    match (dtype, lit) {
        (DType::BF16, Literal::BF16Bits(bits)) => data.extend_from_slice(&bits.to_ne_bytes()),
        (DType::F16, Literal::F16Bits(bits)) => data.extend_from_slice(&bits.to_ne_bytes()),
        (DType::F32, Literal::F32Bits(bits)) => data.extend_from_slice(&bits.to_ne_bytes()),
        (DType::F64, Literal::F64Bits(bits)) => data.extend_from_slice(&bits.to_ne_bytes()),
        (DType::I64, Literal::I64(value)) => data.extend_from_slice(&value.to_ne_bytes()),
        (DType::I32, Literal::I64(value)) => {
            let value = i32::try_from(value).map_err(|_| FfiError::UnrepresentableLiteral {
                dtype,
                literal: lit,
            })?;
            data.extend_from_slice(&value.to_ne_bytes());
        }
        (DType::U32, Literal::U32(value)) => data.extend_from_slice(&value.to_ne_bytes()),
        (DType::U64, Literal::U64(value)) => data.extend_from_slice(&value.to_ne_bytes()),
        (DType::Bool, Literal::Bool(value)) => data.push(u8::from(value)),
        (DType::Complex64, Literal::Complex64Bits(re, im)) => {
            data.extend_from_slice(&re.to_ne_bytes());
            data.extend_from_slice(&im.to_ne_bytes());
        }
        (DType::Complex128, Literal::Complex128Bits(re, im)) => {
            data.extend_from_slice(&re.to_ne_bytes());
            data.extend_from_slice(&im.to_ne_bytes());
        }
        (dtype, literal) => return Err(FfiError::UnrepresentableLiteral { dtype, literal }),
    }
    Ok(())
}

fn bytes_to_literal(bytes: &[u8], dtype: DType) -> Result<Literal, FfiError> {
    match dtype {
        DType::BF16 => {
            let arr: [u8; 2] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 2,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::BF16Bits(u16::from_ne_bytes(arr)))
        }
        DType::F16 => {
            let arr: [u8; 2] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 2,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::F16Bits(u16::from_ne_bytes(arr)))
        }
        DType::F32 => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 4,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::F32Bits(u32::from_ne_bytes(arr)))
        }
        DType::F64 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 8,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::F64Bits(u64::from_ne_bytes(arr)))
        }
        DType::I32 => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 4,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::I64(i64::from(i32::from_ne_bytes(arr))))
        }
        DType::I64 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 8,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::I64(i64::from_ne_bytes(arr)))
        }
        DType::U32 => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 4,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::U32(u32::from_ne_bytes(arr)))
        }
        DType::U64 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 8,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::U64(u64::from_ne_bytes(arr)))
        }
        DType::Complex64 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 8,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::Complex64Bits(
                u32::from_ne_bytes([arr[0], arr[1], arr[2], arr[3]]),
                u32::from_ne_bytes([arr[4], arr[5], arr[6], arr[7]]),
            ))
        }
        DType::Complex128 => {
            let arr: [u8; 16] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 16,
                actual_bytes: bytes.len(),
            })?;
            Ok(Literal::Complex128Bits(
                u64::from_ne_bytes([
                    arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
                ]),
                u64::from_ne_bytes([
                    arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15],
                ]),
            ))
        }
        DType::Bool => {
            let arr: [u8; 1] = bytes.try_into().map_err(|_| FfiError::BufferMismatch {
                buffer_index: 0,
                expected_bytes: 1,
                actual_bytes: bytes.len(),
            })?;
            match arr[0] {
                0 => Ok(Literal::Bool(false)),
                1 => Ok(Literal::Bool(true)),
                value => Err(FfiError::InvalidBoolByte {
                    buffer_index: 0,
                    byte_index: 0,
                    value,
                }),
            }
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
    fn roundtrip_scalar_u32() {
        let val = Value::Scalar(Literal::U32(u32::MAX));
        let buf = value_to_buffer(&val).unwrap();
        let restored = buffer_to_value(&buf).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn roundtrip_scalar_u64() {
        let val = Value::Scalar(Literal::U64(u64::MAX));
        let buf = value_to_buffer(&val).unwrap();
        let restored = buffer_to_value(&buf).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn roundtrip_scalar_complex() {
        let values = [
            Value::Scalar(Literal::Complex64Bits(
                1.25_f32.to_bits(),
                (-3.5_f32).to_bits(),
            )),
            Value::Scalar(Literal::Complex128Bits(
                1.25_f64.to_bits(),
                (-3.5_f64).to_bits(),
            )),
        ];

        for val in values {
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
    fn roundtrip_tensor_i32() {
        let val = Value::Tensor(TensorValue {
            dtype: DType::I32,
            shape: Shape { dims: vec![3] },
            elements: vec![
                Literal::I64(i64::from(i32::MIN)),
                Literal::I64(0),
                Literal::I64(i64::from(i32::MAX)),
            ],
        });
        let buf = value_to_buffer(&val).unwrap();
        assert_eq!(buf.size(), 12);
        assert_eq!(buf.dtype(), DType::I32);
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
    fn roundtrip_tensor_u32() {
        let val = Value::Tensor(TensorValue {
            dtype: DType::U32,
            shape: Shape { dims: vec![3] },
            elements: vec![Literal::U32(0), Literal::U32(7), Literal::U32(u32::MAX)],
        });
        let buf = value_to_buffer(&val).unwrap();
        assert_eq!(buf.size(), 12);
        let restored = buffer_to_value(&buf).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn roundtrip_scalar_f32() -> Result<(), FfiError> {
        let val = Value::Scalar(Literal::F32Bits(1.25_f32.to_bits()));
        let buf = value_to_buffer(&val)?;
        assert_eq!(buf.size(), 4);
        let restored = buffer_to_value(&buf)?;
        assert_eq!(val, restored);
        Ok(())
    }

    #[test]
    fn buffer_to_value_f32() -> Result<(), FfiError> {
        let buf = FfiBuffer::new(vec![0u8; 4], vec![], DType::F32)?;
        assert_eq!(buffer_to_value(&buf)?, Value::Scalar(Literal::F32Bits(0)));
        Ok(())
    }

    #[test]
    fn buffer_to_value_i32_scalar_uses_i64_literal_storage() -> Result<(), FfiError> {
        let buf = FfiBuffer::new((-7_i32).to_ne_bytes().to_vec(), vec![], DType::I32)?;
        assert_eq!(buffer_to_value(&buf)?, Value::Scalar(Literal::I64(-7)));
        Ok(())
    }

    #[test]
    fn buffer_to_value_rejects_noncanonical_bool_byte() {
        let mut buf = FfiBuffer::zeroed(vec![3], DType::Bool).unwrap();
        buf.as_bytes_mut().copy_from_slice(&[0, 1, 2]);

        let err = buffer_to_value(&buf).unwrap_err();
        assert!(matches!(
            err,
            FfiError::InvalidBoolByte {
                buffer_index: 0,
                byte_index: 2,
                value: 2,
            }
        ));
    }

    #[test]
    fn buffer_to_value_rejects_shape_dimension_above_core_limit() {
        let buf = FfiBuffer::new(Vec::new(), vec![u32::MAX as usize + 1, 0], DType::F64).unwrap();
        let err = buffer_to_value(&buf).unwrap_err();
        assert!(matches!(
            err,
            FfiError::ShapeDimensionOutOfRange {
                dimension_index: 0,
                dimension,
                max_supported,
            } if dimension == u32::MAX as usize + 1 && max_supported == u32::MAX as usize
        ));
    }

    #[test]
    fn tensor_to_buffer_rejects_i32_literal_out_of_range() {
        let val = Value::Tensor(TensorValue {
            dtype: DType::I32,
            shape: Shape { dims: vec![1] },
            elements: vec![Literal::I64(i64::from(i32::MAX) + 1)],
        });
        let err = value_to_buffer(&val).unwrap_err();
        assert!(matches!(
            err,
            FfiError::UnrepresentableLiteral {
                dtype: DType::I32,
                literal: Literal::I64(_),
            }
        ));
    }

    #[test]
    fn tensor_to_buffer_rejects_declared_dtype_mismatch() {
        let val = Value::Tensor(TensorValue {
            dtype: DType::F32,
            shape: Shape { dims: vec![1] },
            elements: vec![Literal::U32(1)],
        });
        let err = value_to_buffer(&val).unwrap_err();
        assert!(matches!(
            err,
            FfiError::UnrepresentableLiteral {
                dtype: DType::F32,
                literal: Literal::U32(1),
            }
        ));
    }
}
