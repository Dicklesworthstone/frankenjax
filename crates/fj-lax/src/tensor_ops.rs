#![forbid(unsafe_code)]

use fj_core::{DType, Literal, LiteralBuffer, Primitive, Shape, TensorValue, Value};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::EvalError;
use crate::tensor_contraction::{batched_matmul_2d_f32_in, matmul_2d};
use crate::type_promotion::{binary_literal_op, promote_dtype};

/// Parse a comma-separated list of i64 values from a param string.
pub(crate) fn parse_i64_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
) -> Result<Vec<i64>, EvalError> {
    let raw = params.get(key).ok_or_else(|| EvalError::Unsupported {
        primitive,
        detail: format!("missing required param '{key}'"),
    })?;
    raw.split(',')
        .map(|s| {
            s.trim().parse::<i64>().map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("invalid integer in param '{key}': '{s}'"),
            })
        })
        .collect()
}

fn parse_pad_i64_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
    rank: usize,
    missing_default: Option<i64>,
) -> Result<Vec<i64>, EvalError> {
    let Some(raw) = params.get(key) else {
        if let Some(default) = missing_default {
            return Ok(vec![default; rank]);
        }
        if rank == 0 {
            return Ok(Vec::new());
        }
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("missing required param '{key}'"),
        });
    };

    if raw.trim().is_empty() {
        if rank == 0 {
            return Ok(Vec::new());
        }
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("invalid empty param '{key}' for rank {rank}"),
        });
    }

    raw.split(',')
        .map(|s| {
            s.trim().parse::<i64>().map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("invalid integer in param '{key}': '{s}'"),
            })
        })
        .collect()
}

/// Parse a comma-separated list of usize values from a param string.
pub(crate) fn parse_usize_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
) -> Result<Vec<usize>, EvalError> {
    let raw = params.get(key).ok_or_else(|| EvalError::Unsupported {
        primitive,
        detail: format!("missing required param '{key}'"),
    })?;
    raw.split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid usize in param '{key}': '{s}'"),
                })
        })
        .collect()
}

fn parse_axis_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
    rank: usize,
    default: usize,
) -> Result<usize, EvalError> {
    let Some(raw) = params.get(key) else {
        return Ok(default);
    };

    let axis = raw
        .trim()
        .parse::<i64>()
        .map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("invalid integer in param '{key}': '{raw}'"),
        })?;
    let normalized = if axis < 0 { rank as i64 + axis } else { axis };

    if normalized < 0 || normalized >= rank as i64 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("axis {axis} out of bounds for rank {rank}"),
        });
    }

    Ok(normalized as usize)
}

fn parse_axis_insert_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
    output_rank: usize,
    default: usize,
) -> Result<usize, EvalError> {
    let Some(raw) = params.get(key) else {
        return Ok(default);
    };

    let axis = raw
        .trim()
        .parse::<i64>()
        .map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("invalid integer in param '{key}': '{raw}'"),
        })?;
    let normalized = if axis < 0 {
        output_rank as i64 + axis
    } else {
        axis
    };

    if normalized < 0 || normalized >= output_rank as i64 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("axis {axis} out of bounds for output rank {output_rank}"),
        });
    }

    Ok(normalized as usize)
}

fn parse_dtype_param(
    primitive: Primitive,
    key: &str,
    params: &BTreeMap<String, String>,
) -> Result<DType, EvalError> {
    let raw = params.get(key).ok_or_else(|| EvalError::Unsupported {
        primitive,
        detail: format!("missing required param '{key}'"),
    })?;
    parse_dtype_name(primitive, key, raw)
}

fn parse_dtype_name(primitive: Primitive, key: &str, raw: &str) -> Result<DType, EvalError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f16" | "float16" => Ok(DType::F16),
        "f32" | "float32" => Ok(DType::F32),
        "f64" | "float64" => Ok(DType::F64),
        "i32" => Ok(DType::I32),
        "i64" => Ok(DType::I64),
        "u32" => Ok(DType::U32),
        "u64" => Ok(DType::U64),
        "bool" => Ok(DType::Bool),
        "complex64" => Ok(DType::Complex64),
        "complex128" => Ok(DType::Complex128),
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: format!("unsupported dtype in '{key}': '{raw}'"),
        }),
    }
}

const fn dtype_bit_width(dtype: DType) -> u32 {
    match dtype {
        DType::BF16 | DType::F16 => 16,
        DType::F32 | DType::I32 | DType::U32 => 32,
        DType::F64 | DType::I64 | DType::U64 | DType::Complex64 => 64,
        DType::Complex128 => 128,
        DType::Bool => 8,
    }
}

fn literal_to_bytes(
    primitive: Primitive,
    dtype: DType,
    literal: Literal,
) -> Result<Vec<u8>, EvalError> {
    let bytes = match dtype {
        DType::I64 => literal
            .as_i64()
            .ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "bitcast source literal is not representable as i64",
            })?
            .to_le_bytes()
            .to_vec(),
        DType::I32 => {
            let value = literal.as_i64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "bitcast source literal is not representable as i32",
            })?;
            i32::try_from(value)
                .map_err(|_| EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is out of i32 range",
                })?
                .to_le_bytes()
                .to_vec()
        }
        DType::U32 => u32::try_from(literal.as_u64().ok_or(EvalError::TypeMismatch {
            primitive,
            detail: "bitcast source literal is not representable as u32",
        })?)
        .map_err(|_| EvalError::TypeMismatch {
            primitive,
            detail: "bitcast source literal is out of u32 range",
        })?
        .to_le_bytes()
        .to_vec(),
        DType::U64 => literal
            .as_u64()
            .ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "bitcast source literal is not representable as u64",
            })?
            .to_le_bytes()
            .to_vec(),
        DType::Bool => match literal {
            Literal::Bool(value) => vec![u8::from(value)],
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not a bool",
                });
            }
        },
        DType::BF16 => match literal {
            Literal::BF16Bits(bits) => bits.to_le_bytes().to_vec(),
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not bf16",
                });
            }
        },
        DType::F16 => match literal {
            Literal::F16Bits(bits) => bits.to_le_bytes().to_vec(),
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not f16",
                });
            }
        },
        DType::F32 => match literal {
            Literal::F32Bits(bits) => bits.to_le_bytes().to_vec(),
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not f32",
                });
            }
        },
        DType::F64 => literal
            .as_f64()
            .ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "bitcast source literal is not representable as f64",
            })?
            .to_bits()
            .to_le_bytes()
            .to_vec(),
        DType::Complex64 => match literal {
            Literal::Complex64Bits(re, im) => {
                let mut out = Vec::with_capacity(8);
                out.extend_from_slice(&re.to_le_bytes());
                out.extend_from_slice(&im.to_le_bytes());
                out
            }
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not complex64",
                });
            }
        },
        DType::Complex128 => match literal {
            Literal::Complex128Bits(re, im) => {
                let mut out = Vec::with_capacity(16);
                out.extend_from_slice(&re.to_le_bytes());
                out.extend_from_slice(&im.to_le_bytes());
                out
            }
            _ => {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitcast source literal is not complex128",
                });
            }
        },
    };
    Ok(bytes)
}

fn bytes_to_literal(
    primitive: Primitive,
    dtype: DType,
    bytes: &[u8],
) -> Result<Literal, EvalError> {
    match dtype {
        DType::I64 => {
            let array = <[u8; 8]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for i64".to_owned(),
            })?;
            Ok(Literal::I64(i64::from_le_bytes(array)))
        }
        DType::I32 => {
            let array = <[u8; 4]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for i32".to_owned(),
            })?;
            Ok(Literal::I64(i64::from(i32::from_le_bytes(array))))
        }
        DType::U32 => {
            let array = <[u8; 4]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for u32".to_owned(),
            })?;
            Ok(Literal::U32(u32::from_le_bytes(array)))
        }
        DType::U64 => {
            let array = <[u8; 8]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for u64".to_owned(),
            })?;
            Ok(Literal::U64(u64::from_le_bytes(array)))
        }
        DType::Bool => {
            let array = <[u8; 1]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for bool".to_owned(),
            })?;
            Ok(Literal::Bool(array[0] != 0))
        }
        DType::BF16 => {
            let array = <[u8; 2]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for bf16".to_owned(),
            })?;
            Ok(Literal::BF16Bits(u16::from_le_bytes(array)))
        }
        DType::F16 => {
            let array = <[u8; 2]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for f16".to_owned(),
            })?;
            Ok(Literal::F16Bits(u16::from_le_bytes(array)))
        }
        DType::F32 => {
            let array = <[u8; 4]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for f32".to_owned(),
            })?;
            Ok(Literal::F32Bits(u32::from_le_bytes(array)))
        }
        DType::F64 => {
            let array = <[u8; 8]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for f64".to_owned(),
            })?;
            Ok(Literal::F64Bits(u64::from_le_bytes(array)))
        }
        DType::Complex64 => {
            let array = <[u8; 8]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for complex64".to_owned(),
            })?;
            let re = u32::from_le_bytes([array[0], array[1], array[2], array[3]]);
            let im = u32::from_le_bytes([array[4], array[5], array[6], array[7]]);
            Ok(Literal::Complex64Bits(re, im))
        }
        DType::Complex128 => {
            let array = <[u8; 16]>::try_from(bytes).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: "bitcast internal size mismatch for complex128".to_owned(),
            })?;
            let re = u64::from_le_bytes([
                array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7],
            ]);
            let im = u64::from_le_bytes([
                array[8], array[9], array[10], array[11], array[12], array[13], array[14],
                array[15],
            ]);
            Ok(Literal::Complex128Bits(re, im))
        }
    }
}

/// Reshape: change the shape of a tensor without changing its data.
/// Params: `new_shape` (comma-separated dims, -1 for a single inferred axis).
fn resolve_reshape_dims(
    primitive: Primitive,
    shape_spec: &[i64],
    elem_count: usize,
    input_shape: Shape,
) -> Result<Vec<u32>, EvalError> {
    let mut inferred_axis: Option<usize> = None;
    let mut known_product = 1_usize;
    let mut dims = Vec::with_capacity(shape_spec.len());

    for (idx, d) in shape_spec.iter().enumerate() {
        if *d == -1 {
            if inferred_axis.is_some() {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "only one -1 inferred axis allowed in new_shape".into(),
                });
            }
            inferred_axis = Some(idx);
            dims.push(0_u32); // temporary slot filled after the inferred extent is known
        } else if *d >= 0 {
            let du = u32::try_from(*d).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("new_shape dim {d} exceeds u32 range"),
            })?;
            known_product =
                known_product
                    .checked_mul(du as usize)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "reshape known dimension product overflows usize".to_owned(),
                    })?;
            dims.push(du);
        } else {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("invalid dim {d} in new_shape"),
            });
        }
    }

    if let Some(axis) = inferred_axis {
        if known_product == 0 || !elem_count.is_multiple_of(known_product) {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "cannot infer dim: elem_count={elem_count} known_product={known_product}"
                ),
            });
        }
        let inferred = elem_count / known_product;
        dims[axis] = u32::try_from(inferred).map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("inferred reshape dim {inferred} exceeds u32 range"),
        })?;
    } else if known_product != elem_count {
        return Err(EvalError::ShapeMismatch {
            primitive,
            left: input_shape,
            right: Shape { dims },
        });
    }

    Ok(dims)
}

pub(crate) fn eval_reshape(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Reshape;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let shape_spec = parse_i64_param(primitive, "new_shape", params)?;

    match &inputs[0] {
        Value::Scalar(lit) => {
            let dims = resolve_reshape_dims(primitive, &shape_spec, 1, Shape::scalar())?;
            Ok(Value::Tensor(TensorValue::new(
                match lit {
                    Literal::I64(_) => DType::I64,
                    Literal::U32(_) => DType::U32,
                    Literal::U64(_) => DType::U64,
                    Literal::BF16Bits(_) => DType::BF16,
                    Literal::F16Bits(_) => DType::F16,
                    Literal::F32Bits(_) => DType::F32,
                    Literal::F64Bits(_) => DType::F64,
                    Literal::Bool(_) => DType::Bool,
                    Literal::Complex64Bits(..) => DType::Complex64,
                    Literal::Complex128Bits(..) => DType::Complex128,
                },
                Shape { dims },
                vec![*lit],
            )?))
        }
        Value::Tensor(tensor) => {
            let dims = resolve_reshape_dims(
                primitive,
                &shape_spec,
                tensor.elements.len(),
                tensor.shape.clone(),
            )?;

            // Reshape is metadata-only — the element sequence is unchanged. Clone
            // the backing buffer (a cheap Arc bump; dense/concat storage preserved)
            // and re-tag the shape, instead of `to_vec()` which materializes the
            // whole buffer into a fresh `Vec<Literal>` (24B/elem) and re-analyzes.
            Ok(Value::Tensor(TensorValue::new_with_literal_buffer(
                tensor.dtype,
                Shape { dims },
                tensor.elements.clone(),
            )?))
        }
    }
}

/// Transpose: permute the axes of a tensor.
/// Params: `permutation` (comma-separated axis indices). If absent, reverses axes.
/// Rank-2 transpose ([1,0]) over a typed slice via a cache-blocked tile walk:
/// strided source reads + contiguous destination writes stay tile-local. Generic
/// over the element type (`T = f64/f32/u16/i64/Literal`) so the dense and Literal
/// paths share identical index math — pure data movement, bit-for-bit identical.
fn transpose_2d_blocked<T: Copy>(src: &[T], rows: usize, cols: usize, total: usize) -> Vec<T> {
    let mut out = vec![src[0]; total];
    const BLOCK: usize = 64;
    let mut bi = 0;
    while bi < rows {
        let i_end = (bi + BLOCK).min(rows);
        let mut bj = 0;
        while bj < cols {
            let j_end = (bj + BLOCK).min(cols);
            for i in bi..i_end {
                let src_row = i * cols;
                for j in bj..j_end {
                    out[j * rows + i] = src[src_row + j];
                }
            }
            bj += BLOCK;
        }
        bi += BLOCK;
    }
    out
}

/// General N-D transpose over a typed slice via a row-major odometer that
/// maintains the source flat index incrementally (`step[axis] =
/// old_strides[permutation[axis]]`). Generic over the element type; pure data
/// movement, bit-for-bit identical to the per-`Literal` walk.
fn transpose_general<T: Copy>(
    src: &[T],
    step: &[usize],
    new_extent: &[usize],
    rank: usize,
    total: usize,
) -> Vec<T> {
    let mut out = Vec::with_capacity(total);
    let mut coord = vec![0_usize; rank];
    let mut old_flat = 0_usize;
    for _ in 0..total {
        out.push(src[old_flat]);
        let mut axis = rank;
        while axis > 0 {
            axis -= 1;
            coord[axis] += 1;
            old_flat += step[axis];
            if coord[axis] < new_extent[axis] {
                break;
            }
            coord[axis] = 0;
            old_flat -= step[axis] * new_extent[axis];
        }
    }
    out
}

pub(crate) fn eval_transpose(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Transpose;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()), // scalar transpose is identity
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            let permutation = if params.contains_key("permutation") {
                parse_usize_param(primitive, "permutation", params)?
            } else {
                (0..rank).rev().collect()
            };

            if permutation.len() != rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "permutation length {} does not match rank {}",
                        permutation.len(),
                        rank
                    ),
                });
            }

            // Validate permutation is valid
            let mut seen = vec![false; rank];
            for &p in &permutation {
                if p >= rank || seen[p] {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("invalid permutation: {permutation:?} for rank {rank}"),
                    });
                }
                seen[p] = true;
            }

            let old_dims = &tensor.shape.dims;
            let new_dims: Vec<u32> = permutation.iter().map(|&p| old_dims[p]).collect();
            let total = tensor.elements.len();

            if total == 0 {
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: new_dims },
                    Vec::new(),
                )?));
            }

            let old_strides = checked_row_major_strides(primitive, "transpose", old_dims)?;
            // `new_strides` is validated (product == total fits usize); the
            // walks below stay within [0, total), so the prior per-element
            // checked div/mul/add are unnecessary.
            let _ = checked_row_major_strides(primitive, "transpose", &new_dims)?;

            // Rank-2 transpose (permutation [1,0]) is pure data movement; a
            // cache-blocked tile walk keeps both the strided source reads and
            // the contiguous destination writes local, instead of striding the
            // whole matrix per element. Output is identical (same permutation),
            // so it is bit-for-bit equivalent.
            // Dense dispatch shared by both transpose paths: copy the permuted
            // elements straight from the typed backing into dense output, avoiding
            // the full-buffer `Vec<Literal>` materialization that `tensor.elements[i]`
            // triggers AND the boxed output. Bit-identical (pure data movement). Falls
            // through (does nothing) for non-dense storage -> the Literal walk below.
            macro_rules! dense_transpose {
                ($kernel:ident ( $($arg:tt)* )) => {{
                    if let Some(s) = tensor.elements.as_f64_slice() {
                        return Ok(Value::Tensor(TensorValue::new_f64_values(
                            Shape { dims: new_dims.clone() }, $kernel(s, $($arg)*))?));
                    }
                    if let Some(s) = tensor.elements.as_f32_slice() {
                        return Ok(Value::Tensor(TensorValue::new_f32_values(
                            Shape { dims: new_dims.clone() }, $kernel(s, $($arg)*))?));
                    }
                    if let Some(s) = tensor.elements.as_half_float_slice() {
                        return Ok(Value::Tensor(TensorValue::new_half_float_values(
                            tensor.dtype, Shape { dims: new_dims.clone() }, $kernel(s, $($arg)*))?));
                    }
                    if let Some(s) = tensor.elements.as_i64_slice() {
                        // I64 backing also holds I32-dtype tensors; preserve the input
                        // integer width (i32 transpose must stay I32, not widen to I64).
                        let out = $kernel(s, $($arg)*);
                        let shape = Shape { dims: new_dims.clone() };
                        let tv = if tensor.dtype == DType::I32 {
                            TensorValue::new_i32_values(shape, out)
                        } else {
                            TensorValue::new_i64_values(shape, out)
                        };
                        return Ok(Value::Tensor(tv?));
                    }
                    // Complex (Complex64/Complex128) transpose over the dense
                    // (f64,f64) backing — FFT / complex-linalg transposes otherwise
                    // box every element through the per-`Literal` odometer. Pure data
                    // movement over Copy pairs; `tensor.dtype` preserves the width and
                    // `new_complex_values` round-trips the bits, so bit-for-bit identical.
                    if let Some(s) = tensor.elements.as_complex_slice() {
                        return Ok(Value::Tensor(TensorValue::new_complex_values(
                            tensor.dtype, Shape { dims: new_dims.clone() }, $kernel(s, $($arg)*))?));
                    }
                }};
            }

            if rank == 2 && permutation[0] == 1 && permutation[1] == 0 {
                let rows = old_dims[0] as usize; // source rows (M)
                let cols = old_dims[1] as usize; // source cols (N); output is N x M
                dense_transpose!(transpose_2d_blocked(rows, cols, total));
                let mut new_elements = vec![tensor.elements[0]; total];
                const BLOCK: usize = 64;
                let mut bi = 0;
                while bi < rows {
                    let i_end = (bi + BLOCK).min(rows);
                    let mut bj = 0;
                    while bj < cols {
                        let j_end = (bj + BLOCK).min(cols);
                        for i in bi..i_end {
                            let src_row = i * cols;
                            for j in bj..j_end {
                                // out[j, i] = in[i, j]  (output is cols x rows)
                                new_elements[j * rows + i] = tensor.elements[src_row + j];
                            }
                        }
                        bj += BLOCK;
                    }
                    bi += BLOCK;
                }
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: new_dims },
                    new_elements,
                )?));
            }

            // Per-axis source stride for the new layout: stepping new-axis k by
            // one moves the source index by `old_strides[permutation[k]]`.
            let step: Vec<usize> = permutation.iter().map(|&p| old_strides[p]).collect();
            let new_extent: Vec<usize> = new_dims.iter().map(|&d| d as usize).collect();

            dense_transpose!(transpose_general(&step, &new_extent, rank, total));

            let mut new_elements = Vec::with_capacity(total);

            // Odometer walk over the new layout in row-major order, maintaining
            // the source flat index incrementally instead of recomputing a full
            // multi-index (with division) for every element. Produces exactly
            // the same permutation as the prior code, so output bits are
            // identical (transpose is pure data movement, no arithmetic).
            let mut coord = vec![0_usize; rank];
            let mut old_flat = 0_usize;
            for _ in 0..total {
                new_elements.push(tensor.elements[old_flat]);
                let mut axis = rank;
                while axis > 0 {
                    axis -= 1;
                    coord[axis] += 1;
                    old_flat += step[axis];
                    if coord[axis] < new_extent[axis] {
                        break;
                    }
                    coord[axis] = 0;
                    old_flat -= step[axis] * new_extent[axis];
                }
            }

            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                Shape { dims: new_dims },
                new_elements,
            )?))
        }
    }
}

/// BroadcastInDim: broadcast a tensor to a larger shape.
/// Params: `shape` (target dims), `broadcast_dimensions` (mapping of input axes to output axes).
pub(crate) fn eval_broadcast_in_dim(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::BroadcastInDim;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let target_dims = parse_broadcast_target_dims(primitive, params)?;

    match &inputs[0] {
        Value::Scalar(lit) => {
            // Broadcast scalar to target shape. Emit DENSE storage for the common
            // numeric dtypes (a typed fill of the value — `jnp.zeros`/`full`/bias-init
            // hit this) instead of a `Vec<Literal>` of `total` 24-byte enums. Output
            // materializes to the same Literals, so it is bit-for-bit identical.
            let total = checked_shape_element_count(primitive, "broadcast_in_dim", &target_dims)?;
            let shape = Shape { dims: target_dims };
            match lit {
                Literal::F64Bits(b) => Ok(Value::Tensor(TensorValue::new_f64_values(
                    shape,
                    vec![f64::from_bits(*b); total],
                )?)),
                Literal::F32Bits(b) => Ok(Value::Tensor(TensorValue::new_f32_values(
                    shape,
                    vec![f32::from_bits(*b); total],
                )?)),
                Literal::BF16Bits(b) => Ok(Value::Tensor(TensorValue::new_half_float_values(
                    DType::BF16,
                    shape,
                    vec![*b; total],
                )?)),
                Literal::F16Bits(b) => Ok(Value::Tensor(TensorValue::new_half_float_values(
                    DType::F16,
                    shape,
                    vec![*b; total],
                )?)),
                Literal::I64(v) => Ok(Value::Tensor(TensorValue::new_i64_values(
                    shape,
                    vec![*v; total],
                )?)),
                Literal::Bool(v) => Ok(Value::Tensor(TensorValue::new_bool_values(
                    shape,
                    vec![*v; total],
                )?)),
                // U32/U64/Complex have no dense storage variant: boxed fill.
                Literal::U32(_) => Ok(Value::Tensor(TensorValue::new(
                    DType::U32,
                    shape,
                    vec![*lit; total],
                )?)),
                Literal::U64(_) => Ok(Value::Tensor(TensorValue::new(
                    DType::U64,
                    shape,
                    vec![*lit; total],
                )?)),
                Literal::Complex64Bits(..) => Ok(Value::Tensor(TensorValue::new(
                    DType::Complex64,
                    shape,
                    vec![*lit; total],
                )?)),
                Literal::Complex128Bits(..) => Ok(Value::Tensor(TensorValue::new(
                    DType::Complex128,
                    shape,
                    vec![*lit; total],
                )?)),
            }
        }
        Value::Tensor(tensor) => {
            let broadcast_dims = if params.contains_key("broadcast_dimensions") {
                parse_usize_param(primitive, "broadcast_dimensions", params)?
            } else {
                // Default: map input axes to trailing output axes.
                let out_rank = target_dims.len();
                let in_rank = tensor.shape.rank();
                if in_rank > out_rank {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("input rank {} exceeds output rank {}", in_rank, out_rank),
                    });
                }
                (out_rank - in_rank..out_rank).collect()
            };

            if broadcast_dims.len() != tensor.shape.rank() {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "broadcast_dimensions length {} must equal input rank {}",
                        broadcast_dims.len(),
                        tensor.shape.rank()
                    ),
                });
            }

            let out_rank = target_dims.len();
            let mut seen = BTreeSet::new();
            for (in_axis, &out_axis) in broadcast_dims.iter().enumerate() {
                if out_axis >= out_rank {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "broadcast_dimensions axis {} out of range for output rank {}",
                            out_axis, out_rank
                        ),
                    });
                }
                if !seen.insert(out_axis) {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "broadcast_dimensions must be unique".to_owned(),
                    });
                }
                let in_dim = tensor.shape.dims[in_axis];
                let target_dim = target_dims[out_axis];
                if in_dim != 1 && in_dim != target_dim {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "cannot broadcast input dim {} into target dim {} at axis {}",
                            in_dim, target_dim, out_axis
                        ),
                    });
                }
            }
            let total = checked_shape_element_count(primitive, "broadcast_in_dim", &target_dims)?;
            if total == 0 {
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: target_dims },
                    Vec::new(),
                )?));
            }

            // Build mapping: for each output axis, which input axis maps to it (if any).
            let mut out_to_in: Vec<Option<usize>> = vec![None; out_rank];
            for (in_axis, &out_axis) in broadcast_dims.iter().enumerate() {
                out_to_in[out_axis] = Some(in_axis);
            }

            // Compute input strides (row-major).
            let in_dims = &tensor.shape.dims;
            let in_strides = checked_row_major_strides(primitive, "broadcast_in_dim", in_dims)?;

            // Dense-output fast paths: replicate directly from the contiguous
            // typed slice into dense storage (new_*_values), bypassing the input
            // Literal materialization and the Vec<Literal> output build. The
            // replication index math is identical to the generic path
            // (broadcast_replicate), so the output is bit-for-bit unchanged.
            if let Some(src) = tensor.elements.as_f64_slice() {
                let out = broadcast_replicate(
                    total,
                    out_rank,
                    &target_dims,
                    &out_to_in,
                    in_dims,
                    &in_strides,
                    src,
                );
                return Ok(Value::Tensor(TensorValue::new_f64_values(
                    Shape { dims: target_dims },
                    out,
                )?));
            }
            if let Some(src) = tensor.elements.as_i64_slice() {
                let out = broadcast_replicate(
                    total,
                    out_rank,
                    &target_dims,
                    &out_to_in,
                    in_dims,
                    &in_strides,
                    src,
                );
                // Preserve the input integer width (i32 broadcast must stay I32).
                let shape = Shape { dims: target_dims };
                let tv = if tensor.dtype == DType::I32 {
                    TensorValue::new_i32_values(shape, out)
                } else {
                    TensorValue::new_i64_values(shape, out)
                };
                return Ok(Value::Tensor(tv?));
            }
            if let Some(src) = tensor.elements.as_bool_slice() {
                let out = broadcast_replicate(
                    total,
                    out_rank,
                    &target_dims,
                    &out_to_in,
                    in_dims,
                    &in_strides,
                    src,
                );
                return Ok(Value::Tensor(TensorValue::new_bool_values(
                    Shape { dims: target_dims },
                    out,
                )?));
            }
            // f32 (JAX's default dtype) and BF16/F16 reuse the same generic
            // `broadcast_replicate<T>` kernel over their typed backings, emitting
            // dense output — avoiding the input Literal materialization + boxed
            // output of the generic path below. Pure replication, bit-identical.
            if let Some(src) = tensor.elements.as_f32_slice() {
                let out = broadcast_replicate(
                    total,
                    out_rank,
                    &target_dims,
                    &out_to_in,
                    in_dims,
                    &in_strides,
                    src,
                );
                return Ok(Value::Tensor(TensorValue::new_f32_values(
                    Shape { dims: target_dims },
                    out,
                )?));
            }
            if let Some(src) = tensor.elements.as_half_float_slice() {
                let out = broadcast_replicate(
                    total,
                    out_rank,
                    &target_dims,
                    &out_to_in,
                    in_dims,
                    &in_strides,
                    src,
                );
                return Ok(Value::Tensor(TensorValue::new_half_float_values(
                    tensor.dtype,
                    Shape { dims: target_dims },
                    out,
                )?));
            }

            let src = tensor.elements.as_slice();
            let elements = broadcast_replicate(
                total,
                out_rank,
                &target_dims,
                &out_to_in,
                in_dims,
                &in_strides,
                src,
            );
            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                Shape { dims: target_dims },
                elements,
            )?))
        }
    }
}

/// Row-major broadcast replication shared by `eval_broadcast_in_dim`'s dense and
/// generic paths: for each of `total` output positions (iterated in row-major
/// order), map the output coordinate to the source flat index (input dims of
/// size 1 map to 0) and copy `src[in_flat]`. Generic over the element type so the
/// dense (f64/i64/bool) and Literal paths share identical index math.
fn broadcast_replicate<T: Copy>(
    total: usize,
    out_rank: usize,
    target_dims: &[u32],
    out_to_in: &[Option<usize>],
    in_dims: &[u32],
    in_strides: &[usize],
    src: &[T],
) -> Vec<T> {
    let mut out = Vec::with_capacity(total);
    let mut out_coords = vec![0_usize; out_rank];
    for _ in 0..total {
        let mut in_flat = 0_usize;
        for (out_axis, mapping) in out_to_in.iter().enumerate() {
            if let Some(in_axis) = mapping {
                let in_dim = in_dims[*in_axis] as usize;
                let coord = if in_dim == 1 { 0 } else { out_coords[out_axis] };
                in_flat += coord * in_strides[*in_axis];
            }
        }
        out.push(src[in_flat]);
        for axis in (0..out_rank).rev() {
            out_coords[axis] += 1;
            if out_coords[axis] < target_dims[axis] as usize {
                break;
            }
            out_coords[axis] = 0;
        }
    }
    out
}

fn parse_broadcast_target_dims(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
) -> Result<Vec<u32>, EvalError> {
    parse_i64_param(primitive, "shape", params)?
        .into_iter()
        .map(|d| {
            if d < 0 {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid target dim {d}"),
                });
            }
            u32::try_from(d).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("target dim {d} exceeds u32 range"),
            })
        })
        .collect()
}

fn checked_shape_element_count(
    primitive: Primitive,
    op_name: &str,
    dims: &[u32],
) -> Result<usize, EvalError> {
    if dims.contains(&0) {
        return Ok(0);
    }

    dims.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim as usize)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("{op_name} shape overflows usize"),
            })
    })
}

fn checked_row_major_strides(
    primitive: Primitive,
    op_name: &str,
    dims: &[u32],
) -> Result<Vec<usize>, EvalError> {
    let rank = dims.len();
    let mut strides = vec![1_usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(dims[i + 1] as usize)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("{op_name} input strides overflow usize"),
            })?;
    }
    Ok(strides)
}

/// Concatenate: join multiple tensors along an axis.
/// Params: `dimension` (axis index, default 0).
pub(crate) fn eval_concatenate(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Concatenate;
    if inputs.is_empty() {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: 0,
        });
    }

    let axis: usize = params
        .get("dimension")
        .map(|s| {
            s.parse::<usize>().map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("invalid dimension param: '{s}'"),
            })
        })
        .transpose()?
        .unwrap_or(0);

    // Collect all inputs as tensors. Scalars are invalid for concatenation.
    let tensors: Vec<&TensorValue> = inputs
        .iter()
        .map(|v| match v {
            Value::Tensor(t) => Ok(t),
            Value::Scalar(_) => Err(EvalError::Unsupported {
                primitive,
                detail: "cannot concatenate scalars".into(),
            }),
        })
        .collect::<Result<_, _>>()?;

    let rank = tensors[0].shape.rank();
    if axis >= rank {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("concat axis {axis} out of bounds for rank {rank}"),
        });
    }

    // JAX `lax.concatenate` requires uniform dtype across all operands;
    // it does not promote. Reject mismatches up front so we never end up
    // building an output tensor whose declared dtype disagrees with its
    // element literals.
    let expected_dtype = tensors[0].dtype;
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.dtype != expected_dtype {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "concatenate operand {i} dtype {:?} does not match operand 0 dtype {:?}",
                    t.dtype, expected_dtype
                ),
            });
        }
    }

    // Validate: all tensors must have the same rank and matching dims on non-concat axes.
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.shape.rank() != rank {
            return Err(EvalError::ShapeMismatch {
                primitive,
                left: tensors[0].shape.clone(),
                right: t.shape.clone(),
            });
        }
        for (ax, (d0, di)) in tensors[0]
            .shape
            .dims
            .iter()
            .zip(t.shape.dims.iter())
            .enumerate()
        {
            if ax != axis && d0 != di {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "tensor {i} dim mismatch on axis {ax}: expected {d0}, got {di}"
                    ),
                });
            }
        }
    }

    // Build output shape.
    let mut out_dims = tensors[0].shape.dims.clone();
    let concat_size = tensors.iter().try_fold(0_u32, |acc, tensor| {
        acc.checked_add(tensor.shape.dims[axis])
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "concatenate axis size overflows u32".to_owned(),
            })
    })?;
    out_dims[axis] = concat_size;

    let total = checked_shape_element_count(primitive, "concatenate", &out_dims)?;

    // Concatenation is pure data movement: along `axis`, the layout splits into
    // `outer` independent blocks (axes < axis) and, within each, a contiguous
    // run of `dim[axis] * inner` elements per operand (`inner` = product of axes
    // > axis, identical across operands). So the whole result is a sequence of
    // contiguous slice copies — same element order as the per-coordinate walk,
    // hence bit-for-bit identical, but via bulk `extend_from_slice`.
    let inner = checked_shape_element_count(primitive, "concatenate inner", &out_dims[axis + 1..])?;
    let outer = checked_shape_element_count(primitive, "concatenate outer", &out_dims[..axis])?;

    let mut parts = Vec::with_capacity(outer.saturating_mul(tensors.len()));
    for o in 0..outer {
        for t in &tensors {
            let block = t.shape.dims[axis] as usize * inner;
            let start = o * block;
            parts.push((t.elements.clone(), start, block));
        }
    }
    let elements =
        LiteralBuffer::from_concat_slices(parts).ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "concatenate slice spans overflow output bounds".to_owned(),
        })?;
    debug_assert_eq!(elements.len(), total);

    let dtype = tensors[0].dtype;
    Ok(Value::Tensor(TensorValue::new_with_literal_buffer(
        dtype,
        Shape { dims: out_dims },
        elements,
    )?))
}

/// Pad: add low/high edge padding and interior padding between elements.
///
/// Inputs: `[operand, pad_value]`
/// Params:
/// - `padding_low`: comma-separated integers (one per axis; negative crops low edge)
/// - `padding_high`: comma-separated integers (one per axis; negative crops high edge)
/// - `padding_interior`: comma-separated non-negative integers (one per axis, optional; defaults to 0)
pub(crate) fn eval_pad(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Pad;
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let operand_dtype = inputs[0].dtype();
    let pad_value_dtype = inputs[1].dtype();
    if pad_value_dtype != operand_dtype {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "pad value dtype {:?} must match operand dtype {:?}",
                pad_value_dtype, operand_dtype
            ),
        });
    }

    let pad_literal = match &inputs[1] {
        Value::Scalar(lit) => *lit,
        Value::Tensor(tensor) if tensor.elements.len() == 1 => tensor.elements[0],
        _ => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "pad value must be a scalar literal".into(),
            });
        }
    };

    let Value::Tensor(operand) = &inputs[0] else {
        let rank = 0;
        let lows_raw = parse_pad_i64_param(primitive, "padding_low", params, rank, None)?;
        let highs_raw = parse_pad_i64_param(primitive, "padding_high", params, rank, None)?;
        let interiors_raw =
            parse_pad_i64_param(primitive, "padding_interior", params, rank, Some(0))?;
        if !lows_raw.is_empty() || !highs_raw.is_empty() || !interiors_raw.is_empty() {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "padding params must match rank {rank}: low={} high={} interior={}",
                    lows_raw.len(),
                    highs_raw.len(),
                    interiors_raw.len()
                ),
            });
        }
        return Ok(inputs[0].clone());
    };

    let rank = operand.shape.rank();
    let lows_raw = parse_pad_i64_param(primitive, "padding_low", params, rank, None)?;
    let highs_raw = parse_pad_i64_param(primitive, "padding_high", params, rank, None)?;
    let interiors_raw =
        parse_pad_i64_param(primitive, "padding_interior", params, rank, Some(0_i64))?;

    if lows_raw.len() != rank || highs_raw.len() != rank || interiors_raw.len() != rank {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "padding params must match rank {rank}: low={} high={} interior={}",
                lows_raw.len(),
                highs_raw.len(),
                interiors_raw.len()
            ),
        });
    }

    let mut lows = Vec::with_capacity(rank);
    let mut interiors = Vec::with_capacity(rank);
    let mut out_dims = Vec::with_capacity(rank);

    for ax in 0..rank {
        let low = lows_raw[ax];
        let high = highs_raw[ax];
        let interior = interiors_raw[ax];
        if interior < 0 {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("interior padding must be non-negative on axis {ax}: {interior}"),
            });
        }

        let dim = i64::from(operand.shape.dims[ax]);
        let interior_span = if dim == 0 {
            0
        } else {
            (dim - 1)
                .checked_mul(interior)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!("padded dimension overflow on axis {ax}"),
                })?
        };
        let out_dim = low
            .checked_add(dim)
            .and_then(|value| value.checked_add(interior_span))
            .and_then(|value| value.checked_add(high))
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("padded dimension overflow on axis {ax}"),
            })?;
        if out_dim < 0 || out_dim > i64::from(u32::MAX) {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("padded dimension overflow on axis {ax}: {out_dim}"),
            });
        }

        lows.push(low);
        interiors.push(interior as usize);
        out_dims.push(out_dim as u32);
    }

    let out_total = checked_shape_element_count(primitive, "pad", &out_dims)?;

    // Edge cases with trivial / tiny output (rank-0 scalar, empty output): build
    // the Literal output directly.
    if rank == 0 {
        let mut out_elements = vec![pad_literal; out_total];
        if !out_elements.is_empty() {
            out_elements[0] = operand.elements[0];
        }
        return Ok(Value::Tensor(TensorValue::new(
            operand.dtype,
            Shape { dims: out_dims },
            out_elements,
        )?));
    }
    if out_total == 0 {
        return Ok(Value::Tensor(TensorValue::new(
            operand.dtype,
            Shape { dims: out_dims },
            Vec::new(),
        )?));
    }

    // Row-major strides.
    let in_dims = &operand.shape.dims;
    let out_strides = checked_row_major_strides(primitive, "pad", &out_dims)?;

    // Dense fast paths: fill the dense output with the pad value (fast typed fill
    // vs the per-element Literal clone) and place input elements into dense
    // storage — bypassing input materialization and the Vec<Literal> output.
    // When there is no interior dilation and no cropping (all low >= 0 and the
    // input fits within the output extent), each contiguous input row maps to a
    // contiguous output run, so whole rows are bulk-copied (copy_from_slice)
    // instead of placed element-by-element. Otherwise the per-element placement
    // (pad_fill_place) runs. Both are bit-identical to the generic path.
    let row_copyable = interiors.iter().all(|&i| i == 0)
        && (0..rank).all(|ax| {
            lows[ax] >= 0 && (lows[ax] as usize + in_dims[ax] as usize) <= out_dims[ax] as usize
        });
    if let (Some(src), Some(pad)) = (operand.elements.as_f64_slice(), pad_literal.as_f64()) {
        let out = if row_copyable {
            pad_copy_rows(src, pad, out_total, rank, in_dims, &lows, &out_strides)
        } else {
            pad_fill_place(
                src,
                pad,
                out_total,
                rank,
                in_dims,
                &lows,
                &interiors,
                &out_dims,
                &out_strides,
            )
        };
        return Ok(Value::Tensor(TensorValue::new_f64_values(
            Shape { dims: out_dims },
            out,
        )?));
    }
    if let (Some(src), Some(pad)) = (operand.elements.as_i64_slice(), pad_literal.as_i64()) {
        let out = if row_copyable {
            pad_copy_rows(src, pad, out_total, rank, in_dims, &lows, &out_strides)
        } else {
            pad_fill_place(
                src,
                pad,
                out_total,
                rank,
                in_dims,
                &lows,
                &interiors,
                &out_dims,
                &out_strides,
            )
        };
        // Preserve the input integer width (i32 pad must stay I32).
        let shape = Shape { dims: out_dims };
        let tv = if operand.dtype == DType::I32 {
            TensorValue::new_i32_values(shape, out)
        } else {
            TensorValue::new_i64_values(shape, out)
        };
        return Ok(Value::Tensor(tv?));
    }
    // f32 (JAX's default dtype) and BF16/F16 reuse the same generic
    // `pad_copy_rows<T>`/`pad_fill_place<T>` kernels over their typed backings —
    // a fast typed fill of the pad value + dense placement, avoiding the input
    // Literal materialization + boxed output. Bit-identical (the typed pad value
    // is the pad_literal's exact bit pattern: f32::from_bits / the raw u16).
    if let (Some(src), Literal::F32Bits(pb)) = (operand.elements.as_f32_slice(), pad_literal) {
        let pad = f32::from_bits(pb);
        let out = if row_copyable {
            pad_copy_rows(src, pad, out_total, rank, in_dims, &lows, &out_strides)
        } else {
            pad_fill_place(
                src,
                pad,
                out_total,
                rank,
                in_dims,
                &lows,
                &interiors,
                &out_dims,
                &out_strides,
            )
        };
        return Ok(Value::Tensor(TensorValue::new_f32_values(
            Shape { dims: out_dims },
            out,
        )?));
    }
    if let (Some(src), Literal::BF16Bits(pb) | Literal::F16Bits(pb)) =
        (operand.elements.as_half_float_slice(), pad_literal)
    {
        let out = if row_copyable {
            pad_copy_rows(src, pb, out_total, rank, in_dims, &lows, &out_strides)
        } else {
            pad_fill_place(
                src,
                pb,
                out_total,
                rank,
                in_dims,
                &lows,
                &interiors,
                &out_dims,
                &out_strides,
            )
        };
        return Ok(Value::Tensor(TensorValue::new_half_float_values(
            operand.dtype,
            Shape { dims: out_dims },
            out,
        )?));
    }

    // Generic Literal path.
    let mut out_elements = vec![pad_literal; out_total];
    if operand.elements.is_empty() {
        return Ok(Value::Tensor(TensorValue::new(
            operand.dtype,
            Shape { dims: out_dims },
            out_elements,
        )?));
    }

    let mut in_coords = vec![0_usize; rank];
    for element in &operand.elements {
        let mut out_flat = 0_usize;
        let mut place_element = true;
        for ax in 0..rank {
            let coord = lows[ax] + (in_coords[ax] * (interiors[ax] + 1)) as i64;
            if coord < 0 || coord >= i64::from(out_dims[ax]) {
                place_element = false;
                break;
            }
            out_flat += coord as usize * out_strides[ax];
        }
        if place_element {
            out_elements[out_flat] = *element;
        }

        for ax in (0..rank).rev() {
            in_coords[ax] += 1;
            if in_coords[ax] < in_dims[ax] as usize {
                break;
            }
            in_coords[ax] = 0;
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        operand.dtype,
        Shape { dims: out_dims },
        out_elements,
    )?))
}

/// Dense Pad kernel (rank >= 1): a dense output prefilled with `pad`, into which
/// each source element is placed at its row-major destination via the same index
/// math as the generic path (low offset + interior dilation, dropped if it falls
/// outside the output extent for negative low / high cropping). Generic over the
/// element type so the f64/i64 dense paths share identical placement with the
/// generic Literal loop — bit-for-bit identical output.
/// Dense Pad kernel for the no-interior, no-cropping case (all `low >= 0` and the
/// input fits within the output extent): each contiguous input row (the last
/// axis) maps to a contiguous output run, so whole rows are bulk-copied. Output
/// is row-major and `out_strides[last] == 1`, so the destination of a row is its
/// leading-axis base plus `low[last]`. Bit-identical to per-element placement.
fn pad_copy_rows<T: Copy>(
    src: &[T],
    pad: T,
    out_total: usize,
    rank: usize,
    in_dims: &[u32],
    lows: &[i64],
    out_strides: &[usize],
) -> Vec<T> {
    let mut out = vec![pad; out_total];
    let last = rank - 1;
    let in_last = in_dims[last] as usize;
    if in_last == 0 {
        return out;
    }
    let outer_count = src.len() / in_last;
    for r in 0..outer_count {
        // Decompose the row index into leading-axis coordinates (row-major: the
        // last leading axis varies fastest) and accumulate the output base.
        let mut rem = r;
        let mut out_base = lows[last] as usize * out_strides[last];
        for ax in (0..last).rev() {
            let d = in_dims[ax] as usize;
            let coord = rem % d;
            rem /= d;
            out_base += (lows[ax] as usize + coord) * out_strides[ax];
        }
        let in_start = r * in_last;
        out[out_base..out_base + in_last].copy_from_slice(&src[in_start..in_start + in_last]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn pad_fill_place<T: Copy>(
    src: &[T],
    pad: T,
    out_total: usize,
    rank: usize,
    in_dims: &[u32],
    lows: &[i64],
    interiors: &[usize],
    out_dims: &[u32],
    out_strides: &[usize],
) -> Vec<T> {
    let mut out = vec![pad; out_total];
    let mut in_coords = vec![0_usize; rank];
    for &element in src {
        let mut out_flat = 0_usize;
        let mut place_element = true;
        for ax in 0..rank {
            let coord = lows[ax] + (in_coords[ax] * (interiors[ax] + 1)) as i64;
            if coord < 0 || coord >= i64::from(out_dims[ax]) {
                place_element = false;
                break;
            }
            out_flat += coord as usize * out_strides[ax];
        }
        if place_element {
            out[out_flat] = element;
        }
        for ax in (0..rank).rev() {
            in_coords[ax] += 1;
            if in_coords[ax] < in_dims[ax] as usize {
                break;
            }
            in_coords[ax] = 0;
        }
    }
    out
}

/// Strided (general) slice gather over a typed slice: walk output positions in
/// row-major order with a reused coordinate odometer, mapping each to its input
/// flat index `sum_ax (start[ax] + out_coord[ax]*stride[ax]) * in_stride[ax]`, and
/// copy `src[in_flat]`. Generic over the element type so the dense (f64/f32/half/
/// i64) and `Literal` paths share identical index math — bit-for-bit identical.
/// No per-element overflow checks: for a valid input tensor every `in_flat` is
/// `< src.len() <= usize::MAX`, so the arithmetic cannot overflow.
fn slice_strided_gather<T: Copy>(
    src: &[T],
    rank: usize,
    total: usize,
    starts: &[usize],
    slice_strides: &[usize],
    in_strides: &[usize],
    out_dims: &[u32],
) -> Vec<T> {
    let mut out = Vec::with_capacity(total);
    let mut out_coords = vec![0_usize; rank];
    for _ in 0..total {
        let mut in_flat = 0_usize;
        for ax in 0..rank {
            let coord = starts[ax] + out_coords[ax] * slice_strides[ax];
            in_flat += coord * in_strides[ax];
        }
        out.push(src[in_flat]);
        for ax in (0..rank).rev() {
            out_coords[ax] += 1;
            if out_coords[ax] < out_dims[ax] as usize {
                break;
            }
            out_coords[ax] = 0;
        }
    }
    out
}

/// Slice: extract a sub-tensor.
/// Params: `start_indices` and `limit_indices` (comma-separated u32 lists).
pub(crate) fn eval_slice(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Slice;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(_) => Err(EvalError::Unsupported {
            primitive,
            detail: "cannot slice a scalar".into(),
        }),
        Value::Tensor(tensor) => {
            let starts = parse_usize_param(primitive, "start_indices", params)?;
            let limits = parse_usize_param(primitive, "limit_indices", params)?;

            let rank = tensor.shape.rank();
            let slice_strides = if params.contains_key("strides") {
                parse_usize_param(primitive, "strides", params)?
            } else {
                vec![1; rank]
            };
            if starts.len() != rank || limits.len() != rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "start/limit rank mismatch: starts={} limits={} rank={}",
                        starts.len(),
                        limits.len(),
                        rank
                    ),
                });
            }
            if slice_strides.len() != rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "strides rank mismatch: strides={} rank={}",
                        slice_strides.len(),
                        rank
                    ),
                });
            }

            // Validate and compute output dims.
            let mut out_dims = Vec::with_capacity(rank);
            for ax in 0..rank {
                let dim = tensor.shape.dims[ax] as usize;
                let stride = slice_strides[ax];
                if stride == 0 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("slice stride on axis {ax} must be positive"),
                    });
                }
                if starts[ax] > limits[ax] || limits[ax] > dim {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "invalid slice on axis {ax}: start={} limit={} dim={}",
                            starts[ax], limits[ax], dim
                        ),
                    });
                }
                let span = limits[ax] - starts[ax];
                out_dims.push(span.div_ceil(stride) as u32);
            }

            let total = checked_shape_element_count(primitive, "slice", &out_dims)?;
            if total == 0 {
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: out_dims },
                    Vec::new(),
                )?));
            }

            let in_dims = &tensor.shape.dims;

            let has_contiguous_trailing_slice = rank > 0
                && slice_strides.iter().all(|&stride| stride == 1)
                && starts.iter().skip(1).all(|&start| start == 0)
                && limits
                    .iter()
                    .skip(1)
                    .zip(in_dims.iter().skip(1))
                    .all(|(&limit, &dim)| limit == dim as usize);
            if has_contiguous_trailing_slice {
                let row_len = checked_shape_element_count(primitive, "slice row", &in_dims[1..])?;
                let start_offset =
                    starts[0]
                        .checked_mul(row_len)
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "slice start offset overflows usize".to_owned(),
                        })?;
                let end_offset =
                    start_offset
                        .checked_add(total)
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "slice end offset overflows usize".to_owned(),
                        })?;
                let shape = Shape { dims: out_dims };
                // Dense fast path: copy the contiguous element range straight from the
                // typed backing into a dense typed output. Avoids materializing the
                // ENTIRE input buffer to `Vec<Literal>` (which `tensor.elements[range]`
                // does via `Index`/`as_slice`) just to copy a sub-range, AND keeps the
                // output dense. Pure copy -> bit-for-bit identical to the boxed path.
                if let Some(src) = tensor.elements.as_f64_slice() {
                    return Ok(Value::Tensor(TensorValue::new_f64_values(
                        shape,
                        src[start_offset..end_offset].to_vec(),
                    )?));
                }
                if let Some(src) = tensor.elements.as_f32_slice() {
                    return Ok(Value::Tensor(TensorValue::new_f32_values(
                        shape,
                        src[start_offset..end_offset].to_vec(),
                    )?));
                }
                if let Some(src) = tensor.elements.as_half_float_slice() {
                    return Ok(Value::Tensor(TensorValue::new_half_float_values(
                        tensor.dtype,
                        shape,
                        src[start_offset..end_offset].to_vec(),
                    )?));
                }
                if let Some(src) = tensor.elements.as_i64_slice() {
                    // Preserve the input integer width (i32 slice must stay I32).
                    let vals = src[start_offset..end_offset].to_vec();
                    let tv = if tensor.dtype == DType::I32 {
                        TensorValue::new_i32_values(shape, vals)
                    } else {
                        TensorValue::new_i64_values(shape, vals)
                    };
                    return Ok(Value::Tensor(tv?));
                }
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    shape,
                    tensor.elements[start_offset..end_offset].to_vec(),
                )?));
            }

            let in_strides = checked_row_major_strides(primitive, "slice", in_dims)?;

            // Dense fast paths: gather the strided sub-tensor straight from the
            // typed backing into dense output, avoiding the full-input `Vec<Literal>`
            // materialization that `tensor.elements[in_flat]` triggers AND the boxed
            // output. Same odometer index math -> bit-for-bit identical. Falls
            // through to the Literal path for other dtypes.
            macro_rules! dense_strided_slice {
                ($slice:expr, $ctor:expr) => {{
                    let out = slice_strided_gather(
                        $slice,
                        rank,
                        total,
                        &starts,
                        &slice_strides,
                        &in_strides,
                        &out_dims,
                    );
                    return Ok(Value::Tensor($ctor(Shape { dims: out_dims }, out)?));
                }};
            }
            if let Some(s) = tensor.elements.as_f64_slice() {
                dense_strided_slice!(s, TensorValue::new_f64_values);
            }
            if let Some(s) = tensor.elements.as_f32_slice() {
                dense_strided_slice!(s, TensorValue::new_f32_values);
            }
            if let Some(s) = tensor.elements.as_half_float_slice() {
                let dt = tensor.dtype;
                dense_strided_slice!(s, |sh, out| TensorValue::new_half_float_values(dt, sh, out));
            }
            if let Some(s) = tensor.elements.as_i64_slice() {
                dense_strided_slice!(s, |sh, o| if tensor.dtype == DType::I32 {
                    TensorValue::new_i32_values(sh, o)
                } else {
                    TensorValue::new_i64_values(sh, o)
                });
            }

            // Literal fallback (boxed/other dtypes): the same gather over Literals.
            let elements = slice_strided_gather(
                tensor.elements.as_slice(),
                rank,
                total,
                &starts,
                &slice_strides,
                &in_strides,
                &out_dims,
            );
            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                Shape { dims: out_dims },
                elements,
            )?))
        }
    }
}

/// Gather: index into an operand tensor using an indices tensor.
///
/// Simplified semantics (axis-0 index gather):
///   operand: tensor of any rank
///   indices: integer tensor of gather indices (into axis 0 of operand)
///   params:  `slice_sizes` — comma-separated sizes for each axis of the gathered slice
///
/// For each index i in `indices`, extracts a slice of shape `slice_sizes` starting
/// at position `[indices[i], 0, 0, ...]` from `operand`.
/// Constraint: `slice_sizes[0]` must be 1 (axis-0 indexing only).
/// Output shape: `indices.shape ++ slice_sizes[1..]` (the leading axis is replaced by
/// the indices tensor shape, remaining axes keep their slice sizes).
///
/// Out-of-bounds indices follow JAX's `GatherScatterMode` (`index_mode` param), which
/// NEVER raises: `clip` clamps into range (the default, matching `jnp` integer
/// indexing), `fill_or_drop` substitutes a fill value for the affected gathered slice,
/// and `promise_in_bounds` assumes validity but clamps defensively rather than panic.
///
/// JAX reference: `jax/_src/lax/slicing.py` `GatherScatterMode`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexMode {
    /// Clamp out-of-bounds indices into `[0, dim-1]` (JAX `CLIP`).
    Clip,
    /// Out-of-bounds gather slices are filled / scatter updates are dropped
    /// (JAX `FILL_OR_DROP`).
    FillOrDrop,
    /// Caller promises indices are in bounds; out-of-bounds is undefined behavior in
    /// JAX, so we clamp defensively to stay panic-free (JAX `PROMISE_IN_BOUNDS`).
    PromiseInBounds,
}

/// Parse the JAX-style `index_mode` (out-of-bounds policy) param. Distinct from the
/// scatter combiner `mode` ("overwrite"/"add"/"mul"/"min"/"max"). Defaults to `default`
/// when absent.
fn parse_index_mode(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
    default: IndexMode,
) -> Result<IndexMode, EvalError> {
    match params.get("index_mode").map(String::as_str) {
        None => Ok(default),
        Some("clip") => Ok(IndexMode::Clip),
        Some("fill" | "drop" | "fill_or_drop") => Ok(IndexMode::FillOrDrop),
        Some("promise_in_bounds" | "promise") => Ok(IndexMode::PromiseInBounds),
        Some(other) => Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "unknown index_mode \"{other}\", expected \"clip\", \"fill_or_drop\", or \"promise_in_bounds\""
            ),
        }),
    }
}

/// Resolve a flat axis-0 index against `dim` under an [`IndexMode`]. Returns
/// `Some(in_bounds_index)` to read/write, or `None` to fill (gather) / drop (scatter).
/// `dim` is guaranteed `>= 1` by the slice-size checks in the callers.
fn resolve_axis0_index(idx: usize, dim: usize, mode: IndexMode) -> Option<usize> {
    if idx < dim {
        Some(idx)
    } else {
        match mode {
            IndexMode::FillOrDrop => None,
            IndexMode::Clip | IndexMode::PromiseInBounds => Some(dim - 1),
        }
    }
}

/// JAX's default `FILL_OR_DROP` gather fill value for `dtype`
/// (`jax/_src/lax/slicing.py`): NaN for inexact, `iinfo.min` for signed integers,
/// `iinfo.max`/`true` for unsigned/bool, NaN real component for complex.
fn gather_fill_literal(dtype: DType) -> Literal {
    match dtype {
        DType::F64 => Literal::F64Bits(f64::NAN.to_bits()),
        DType::F32 => Literal::F32Bits(f32::NAN.to_bits()),
        DType::F16 => Literal::from_f16_f64(f64::NAN),
        DType::BF16 => Literal::from_bf16_f64(f64::NAN),
        DType::I32 => Literal::I64(i64::from(i32::MIN)),
        DType::I64 => Literal::I64(i64::MIN),
        DType::U32 => Literal::U32(u32::MAX),
        DType::U64 => Literal::U64(u64::MAX),
        DType::Bool => Literal::Bool(true),
        DType::Complex64 => Literal::Complex64Bits(f32::NAN.to_bits(), 0),
        DType::Complex128 => Literal::Complex128Bits(f64::NAN.to_bits(), 0),
    }
}

pub(crate) fn eval_gather(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Gather;
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let operand = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "cannot gather from a scalar".into(),
            });
        }
    };

    if operand.shape.rank() == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "cannot gather from a rank-0 tensor".into(),
        });
    }

    let slice_sizes = parse_usize_param(primitive, "slice_sizes", params)?;
    if slice_sizes.len() != operand.shape.rank() {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "slice_sizes length {} must equal operand rank {}",
                slice_sizes.len(),
                operand.shape.rank()
            ),
        });
    }

    if slice_sizes[0] != 1 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "slice_sizes[0] must be 1 for gather, got {}",
                slice_sizes[0]
            ),
        });
    }

    // Extract indices as flat i64 values + capture indices shape
    let (index_vals, index_shape): (Vec<usize>, Vec<u32>) = match &inputs[1] {
        Value::Scalar(lit) => (vec![lit_to_usize(lit, primitive)?], Vec::new()),
        Value::Tensor(t) => (
            t.elements
                .iter()
                .map(|lit| lit_to_usize(lit, primitive))
                .collect::<Result<_, _>>()?,
            t.shape.dims.clone(),
        ),
    };

    let rank = operand.shape.rank();

    // Compute operand strides (row-major)
    let op_dims = &operand.shape.dims;

    for (ax, (&ss, &dim)) in slice_sizes.iter().zip(op_dims.iter()).enumerate() {
        if ss > dim as usize {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("slice_sizes[{ax}] = {ss} exceeds operand dimension {dim}"),
            });
        }
    }

    // Output shape: indices.shape ++ slice_sizes[1..]
    let mut out_dims: Vec<u32> = index_shape;
    let trailing_slice_dims: Vec<u32> = slice_sizes
        .iter()
        .skip(1)
        .map(|&s| {
            u32::try_from(s).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("slice size {s} exceeds u32 range"),
            })
        })
        .collect::<Result<_, _>>()?;
    out_dims.extend(trailing_slice_dims.iter().copied());

    // Number of elements per gathered slice (product of slice_sizes[1..])
    let slice_elems = checked_shape_element_count(primitive, "gather slice", &trailing_slice_dims)?;

    let total = checked_shape_element_count(primitive, "gather output", &out_dims)?;

    // Resolve out-of-bounds indices per JAX GatherScatterMode (never raises).
    // `Some(idx)` reads operand[idx, ..]; `None` fills the slice with `fill_lit`.
    let index_mode = parse_index_mode(primitive, params, IndexMode::Clip)?;
    let dim0 = op_dims[0] as usize;
    let fill_lit = gather_fill_literal(operand.dtype);
    let resolved: Vec<Option<usize>> = index_vals
        .iter()
        .map(|&idx| resolve_axis0_index(idx, dim0, index_mode))
        .collect();

    if total == 0 {
        return Ok(Value::Tensor(TensorValue::new(
            operand.dtype,
            Shape { dims: out_dims },
            Vec::new(),
        )?));
    }

    let mut elements = Vec::with_capacity(total);
    let trailing_slice_is_contiguous = slice_sizes
        .iter()
        .skip(1)
        .zip(op_dims.iter().skip(1))
        .all(|(&slice_size, &dim)| slice_size == dim as usize);

    if trailing_slice_is_contiguous {
        // Dense F64/I64 fast path: copy contiguous slices straight from the typed
        // backing into a dense typed output, bypassing the Vec<Literal>
        // materialization (8 vs 24 bytes/elem). Bit-identical to the generic copy
        // below — same resolved indices, same slice ranges, same per-dtype fill
        // (F64 -> NaN, I64 -> i64::MIN) and overflow/bounds errors.
        macro_rules! dense_contiguous_gather {
            ($slice:expr, $fill:expr, $ctor:path) => {{
                let src = $slice;
                let mut out = Vec::with_capacity(total);
                for &resolved_idx in &resolved {
                    let Some(idx) = resolved_idx else {
                        out.extend(std::iter::repeat_n($fill, slice_elems));
                        continue;
                    };
                    let base_offset =
                        idx.checked_mul(slice_elems)
                            .ok_or_else(|| EvalError::Unsupported {
                                primitive,
                                detail: "gather base offset overflows usize".to_owned(),
                            })?;
                    let end = base_offset.checked_add(slice_elems).ok_or_else(|| {
                        EvalError::Unsupported {
                            primitive,
                            detail: "gather contiguous slice end overflows usize".to_owned(),
                        }
                    })?;
                    if end > src.len() {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: "gather contiguous slice exceeds operand element count"
                                .to_owned(),
                        });
                    }
                    out.extend_from_slice(&src[base_offset..end]);
                }
                return Ok(Value::Tensor($ctor(Shape { dims: out_dims }, out)?));
            }};
        }
        if operand.dtype == DType::F64
            && let Some(src) = operand.elements.as_f64_slice()
        {
            dense_contiguous_gather!(src, f64::NAN, TensorValue::new_f64_values);
        }
        // Dense F32 gather (the embedding-lookup case: gather rows of an
        // [vocab, dim] f32 table). f32 is JAX's DEFAULT dtype. Pure contiguous
        // slice copy — bit-identical to the generic per-`Literal` copy (same
        // resolved indices, same ranges), and the FILL_OR_DROP OOB fill `f32::NAN`
        // matches `gather_fill_literal(F32)` == `F32Bits(f32::NAN.to_bits())`.
        if operand.dtype == DType::F32
            && let Some(src) = operand.elements.as_f32_slice()
        {
            dense_contiguous_gather!(src, f32::NAN, TensorValue::new_f32_values);
        }
        if operand.dtype == DType::I64
            && let Some(src) = operand.elements.as_i64_slice()
        {
            dense_contiguous_gather!(src, i64::MIN, TensorValue::new_i64_values);
        }
        // I32 (JAX's default int) shares the dense i64 backing; gather is a structural
        // copy (no arithmetic, no mod-2^32 wrap), so it reuses the same stencil with the
        // I32 OOB fill (`gather_fill_literal(I32)` == i32::MIN sign-extended) and the
        // I32 dense ctor. Bit-identical to the generic per-`Literal` copy.
        if operand.dtype == DType::I32
            && let Some(src) = operand.elements.as_i64_slice()
        {
            dense_contiguous_gather!(src, i64::from(i32::MIN), TensorValue::new_i32_values);
        }
        // Dense BF16/F16 gather (half-precision embedding lookup — bf16 is the
        // dominant training dtype). Pure contiguous u16-bit copy, bit-identical to
        // the generic per-`Literal` copy. The `new_half_float_values` ctor takes a
        // dtype, so this can't use the path-only `dense_contiguous_gather!` macro;
        // the loop mirrors it exactly. OOB fill = bits of `gather_fill_literal`.
        if matches!(operand.dtype, DType::BF16 | DType::F16)
            && let Some(src) = operand.elements.as_half_float_slice()
        {
            let fill: u16 = match gather_fill_literal(operand.dtype) {
                Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                _ => 0,
            };
            let mut out: Vec<u16> = Vec::with_capacity(total);
            for &resolved_idx in &resolved {
                let Some(idx) = resolved_idx else {
                    out.extend(std::iter::repeat_n(fill, slice_elems));
                    continue;
                };
                let base_offset =
                    idx.checked_mul(slice_elems)
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "gather base offset overflows usize".to_owned(),
                        })?;
                let end =
                    base_offset
                        .checked_add(slice_elems)
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "gather contiguous slice end overflows usize".to_owned(),
                        })?;
                if end > src.len() {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "gather contiguous slice exceeds operand element count".to_owned(),
                    });
                }
                out.extend_from_slice(&src[base_offset..end]);
            }
            return Ok(Value::Tensor(TensorValue::new_half_float_values(
                operand.dtype,
                Shape { dims: out_dims },
                out,
            )?));
        }
        // Dense complex gather (contiguous): copy `(re, im)` runs straight from the
        // dense complex backing. Bit-identical to the generic per-`Literal` copy; OOB
        // fill = the (re, im) of `gather_fill_literal` ((NaN, 0)). `new_complex_values`
        // re-rounds Complex64 to f32 exactly as the literal storage does.
        if matches!(operand.dtype, DType::Complex64 | DType::Complex128)
            && let Some(src) = operand.elements.as_complex_slice()
        {
            let fill: (f64, f64) = match gather_fill_literal(operand.dtype) {
                Literal::Complex64Bits(re, im) => {
                    (f64::from(f32::from_bits(re)), f64::from(f32::from_bits(im)))
                }
                Literal::Complex128Bits(re, im) => (f64::from_bits(re), f64::from_bits(im)),
                _ => (0.0, 0.0),
            };
            let mut out: Vec<(f64, f64)> = Vec::with_capacity(total);
            for &resolved_idx in &resolved {
                let Some(idx) = resolved_idx else {
                    out.extend(std::iter::repeat_n(fill, slice_elems));
                    continue;
                };
                let base_offset =
                    idx.checked_mul(slice_elems)
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "gather base offset overflows usize".to_owned(),
                        })?;
                let end = base_offset.checked_add(slice_elems).ok_or_else(|| {
                    EvalError::Unsupported {
                        primitive,
                        detail: "gather contiguous slice end overflows usize".to_owned(),
                    }
                })?;
                if end > src.len() {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "gather contiguous slice exceeds operand element count".to_owned(),
                    });
                }
                out.extend_from_slice(&src[base_offset..end]);
            }
            return Ok(Value::Tensor(TensorValue::new_complex_values(
                operand.dtype,
                Shape { dims: out_dims },
                out,
            )?));
        }

        for &resolved_idx in &resolved {
            let Some(idx) = resolved_idx else {
                // FILL_OR_DROP out-of-bounds slice: emit the fill value.
                elements.extend(std::iter::repeat_n(fill_lit, slice_elems));
                continue;
            };
            let base_offset =
                idx.checked_mul(slice_elems)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "gather base offset overflows usize".to_owned(),
                    })?;
            let end =
                base_offset
                    .checked_add(slice_elems)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "gather contiguous slice end overflows usize".to_owned(),
                    })?;
            if end > operand.elements.len() {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "gather contiguous slice exceeds operand element count".to_owned(),
                });
            }
            elements.extend_from_slice(&operand.elements[base_offset..end]);
        }

        return Ok(Value::Tensor(TensorValue::new(
            operand.dtype,
            Shape { dims: out_dims },
            elements,
        )?));
    }

    let op_strides = checked_row_major_strides(primitive, "gather", op_dims)?;

    // Dense fast path for the NON-contiguous (partial trailing slice / strided)
    // gather: read each gathered element straight from the typed backing using the
    // SAME odometer offset math as the generic loop, into dense typed output —
    // bypassing the 24-byte `Literal` reconstruction per access AND the boxed
    // output. Bit-for-bit identical (same offsets, same order, same OOB fill bits
    // = `gather_fill_literal`). Covers f64/f32/i64/bf16/f16; other dtypes fall to
    // the generic Literal loop below.
    macro_rules! dense_strided_gather {
        ($src:expr, $fill:expr, $ctor:expr) => {{
            let src = $src;
            let mut out = Vec::with_capacity(resolved.len().saturating_mul(slice_elems));
            for &resolved_idx in &resolved {
                let Some(idx) = resolved_idx else {
                    out.extend(std::iter::repeat_n($fill, slice_elems));
                    continue;
                };
                let base_offset =
                    idx.checked_mul(op_strides[0]).ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "gather base offset overflows usize".to_owned(),
                    })?;
                let mut slice_coords = vec![0_usize; rank.saturating_sub(1)];
                for _ in 0..slice_elems {
                    let mut flat = base_offset;
                    for (ax, &coord) in slice_coords.iter().enumerate() {
                        let offset =
                            coord.checked_mul(op_strides[ax + 1]).ok_or_else(|| {
                                EvalError::Unsupported {
                                    primitive,
                                    detail: format!(
                                        "gather offset overflows usize on axis {}",
                                        ax + 1
                                    ),
                                }
                            })?;
                        flat = flat.checked_add(offset).ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "gather flat offset overflows usize".to_owned(),
                        })?;
                    }
                    let v = *src.get(flat).ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "gather offset exceeds operand element count".to_owned(),
                    })?;
                    out.push(v);
                    if rank > 1 {
                        for ax in (0..rank - 1).rev() {
                            slice_coords[ax] += 1;
                            if slice_coords[ax] < slice_sizes[ax + 1] {
                                break;
                            }
                            slice_coords[ax] = 0;
                        }
                    }
                }
            }
            return Ok(Value::Tensor($ctor(Shape { dims: out_dims }, out)?));
        }};
    }
    if let Some(s) = operand.elements.as_f64_slice() {
        dense_strided_gather!(s, f64::NAN, TensorValue::new_f64_values);
    }
    if let Some(s) = operand.elements.as_f32_slice() {
        dense_strided_gather!(s, f32::NAN, TensorValue::new_f32_values);
    }
    // NOTE: `as_i64_slice()` is Some for BOTH I64 and I32 (i32 densifies into the same
    // i64 backing), so this MUST be dtype-gated — otherwise an i32 strided gather would
    // be emitted as I64 (widening) with the I64 OOB fill (i64::MIN, wrong). Gate I64 here
    // and handle I32 with its own dtype tag + fill below.
    if operand.dtype == DType::I64
        && let Some(s) = operand.elements.as_i64_slice()
    {
        dense_strided_gather!(s, i64::MIN, TensorValue::new_i64_values);
    }
    if operand.dtype == DType::I32
        && let Some(s) = operand.elements.as_i64_slice()
    {
        dense_strided_gather!(s, i64::from(i32::MIN), TensorValue::new_i32_values);
    }
    if let Some(s) = operand.elements.as_half_float_slice() {
        let dt = operand.dtype;
        let fill: u16 = match gather_fill_literal(dt) {
            Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
            _ => 0,
        };
        dense_strided_gather!(s, fill, |sh, o| TensorValue::new_half_float_values(dt, sh, o));
    }
    if let Some(s) = operand.elements.as_complex_slice() {
        let dt = operand.dtype;
        let fill: (f64, f64) = match gather_fill_literal(dt) {
            Literal::Complex64Bits(re, im) => {
                (f64::from(f32::from_bits(re)), f64::from(f32::from_bits(im)))
            }
            Literal::Complex128Bits(re, im) => (f64::from_bits(re), f64::from_bits(im)),
            _ => (0.0, 0.0),
        };
        dense_strided_gather!(s, fill, |sh, o| TensorValue::new_complex_values(dt, sh, o));
    }

    for &resolved_idx in &resolved {
        let Some(idx) = resolved_idx else {
            // FILL_OR_DROP out-of-bounds slice: emit the fill value.
            elements.extend(std::iter::repeat_n(fill_lit, slice_elems));
            continue;
        };
        // Base offset for this index along axis 0
        let base_offset = idx
            .checked_mul(op_strides[0])
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "gather base offset overflows usize".to_owned(),
            })?;

        // Iterate over all positions within the slice (axes 1..rank)
        let mut slice_coords = vec![0_usize; rank.saturating_sub(1)];
        for _ in 0..slice_elems {
            let mut flat = base_offset;
            for (ax, &coord) in slice_coords.iter().enumerate() {
                let offset = coord.checked_mul(op_strides[ax + 1]).ok_or_else(|| {
                    EvalError::Unsupported {
                        primitive,
                        detail: format!("gather offset overflows usize on axis {}", ax + 1),
                    }
                })?;
                flat = flat
                    .checked_add(offset)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "gather flat offset overflows usize".to_owned(),
                    })?;
            }
            let element = operand
                .elements
                .get(flat)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "gather offset exceeds operand element count".to_owned(),
                })?;
            elements.push(*element);

            // Increment slice coordinates
            if rank > 1 {
                for ax in (0..rank - 1).rev() {
                    slice_coords[ax] += 1;
                    if slice_coords[ax] < slice_sizes[ax + 1] {
                        break;
                    }
                    slice_coords[ax] = 0;
                }
            }
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        operand.dtype,
        Shape { dims: out_dims },
        elements,
    )?))
}

/// Dense F64/I64 scatter fast path (axis-0 slice scatter): scatter directly over
/// a clone of the contiguous typed operand backing, reading update slices from
/// the typed updates backing, bypassing the `Vec<Literal>` materialization and
/// per-element Literal dispatch. Bit-for-bit identical to the generic path —
/// same resolved indices, same slice ranges, overwrite via copy_from_slice and
/// add via `a + b` (F64) / `a.wrapping_add(b)` (I64), matching binary_literal_op
/// Add. Returns `None` unless both operand and updates are the same F64/I64 dense
/// storage.
/// How a scatter combines an incoming `update` with the `current` operand value
/// at a target index. Mirrors `jax.lax.scatter` / `scatter_add` / `scatter_mul` /
/// `scatter_min` / `scatter_max` (the `update_jaxpr` combiner): `Overwrite` is
/// plain `scatter` (replace), the rest fold via the matching primitive. All four
/// folds are associative, so repeated indices accumulate order-independently.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScatterCombine {
    Overwrite,
    Add,
    Mul,
    Min,
    Max,
}

/// Combine one `(current, update)` literal pair under `combine`. Routes through
/// the same `binary_literal_op` the elementwise ops use, so dtype dispatch,
/// integer wrapping, and float NaN semantics (via `jax_min_f64`/`jax_max_f64`,
/// matching `lax.min`/`lax.max`) are identical to `Add`/`Mul`/`Min`/`Max`
/// elementwise. (Complex operands never reach this Literal path: they always have
/// dense `(re, im)` storage, so `eval_scatter_dense` handles every complex combiner
/// — overwrite/add/mul/min/max — before this fallback is consulted.)
#[inline]
fn scatter_combine_literal(
    combine: ScatterCombine,
    current: Literal,
    update: Literal,
) -> Result<Literal, EvalError> {
    match combine {
        ScatterCombine::Overwrite => Ok(update),
        ScatterCombine::Add => binary_literal_op(
            current,
            update,
            Primitive::Add,
            &|a, b| a.wrapping_add(b),
            &|a, b| a + b,
        ),
        ScatterCombine::Mul => binary_literal_op(
            current,
            update,
            Primitive::Mul,
            &|a, b| a.wrapping_mul(b),
            &|a, b| a * b,
        ),
        ScatterCombine::Min => binary_literal_op(
            current,
            update,
            Primitive::Min,
            &|a, b| a.min(b),
            &crate::jax_min_f64,
        ),
        ScatterCombine::Max => binary_literal_op(
            current,
            update,
            Primitive::Max,
            &|a, b| a.max(b),
            &crate::jax_max_f64,
        ),
    }
}

/// Lexicographic `lhs >= rhs` on complex `(re, im)` pairs. Mirrors
/// `arithmetic::complex_lex_ge` (private there) so complex scatter min/max fold
/// bit-identically to elementwise complex `lax.max`/`lax.min`: real part first,
/// imaginary as the tie-breaker, raw IEEE `>`/`>=` (NOT `total_cmp`).
#[inline]
fn complex_lex_ge_scatter(lhs: (f64, f64), rhs: (f64, f64)) -> bool {
    lhs.0 > rhs.0 || (lhs.0 == rhs.0 && lhs.1 >= rhs.1)
}

#[allow(clippy::type_complexity)]
fn eval_scatter_dense(
    operand: &TensorValue,
    updates: &TensorValue,
    index_vals: &[usize],
    slice_elems: usize,
    dim0: usize,
    index_mode: IndexMode,
    combine: ScatterCombine,
) -> Result<Option<Value>, EvalError> {
    let primitive = Primitive::Scatter;
    // The dense typed fast paths handle Overwrite + all four reduce combiners
    // (Add/Mul/Min/Max), each computed identically to the generic
    // `scatter_combine_literal`/`binary_literal_op` path. Non-dense / other dtypes
    // (u32/u64/bool/complex) return None and fall to the generic path below.
    let is_overwrite = combine == ScatterCombine::Overwrite;
    macro_rules! scatter_typed {
        ($op:expr, $upd:expr, $ctor:expr, $combine_fn:expr) => {{
            let mut out = $op.to_vec();
            let upd_src = $upd;
            let cf = $combine_fn;
            for (i, &raw_idx) in index_vals.iter().enumerate() {
                let Some(idx) = resolve_axis0_index(raw_idx, dim0, index_mode) else {
                    continue;
                };
                let base = idx
                    .checked_mul(slice_elems)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "scatter base offset overflows usize".to_owned(),
                    })?;
                let uoff = i
                    .checked_mul(slice_elems)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "scatter update offset overflows usize".to_owned(),
                    })?;
                let rend = base
                    .checked_add(slice_elems)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "scatter result slice end overflows usize".to_owned(),
                    })?;
                let uend = uoff
                    .checked_add(slice_elems)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "scatter update slice end overflows usize".to_owned(),
                    })?;
                if rend > out.len() {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "scatter result slice exceeds operand element count".to_owned(),
                    });
                }
                if uend > upd_src.len() {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "scatter update slice exceeds update element count".to_owned(),
                    });
                }
                if is_overwrite {
                    out[base..rend].copy_from_slice(&upd_src[uoff..uend]);
                } else {
                    for j in 0..slice_elems {
                        out[base + j] = cf(out[base + j], upd_src[uoff + j]);
                    }
                }
            }
            return Ok(Some(Value::Tensor($ctor(operand.shape.clone(), out)?)));
        }};
    }
    if operand.dtype == DType::F64
        && let (Some(op), Some(upd)) = (
            operand.elements.as_f64_slice(),
            updates.elements.as_f64_slice(),
        )
    {
        // float ops match scatter_combine_literal's f64 branch exactly.
        let cf: fn(f64, f64) -> f64 = match combine {
            ScatterCombine::Add => |a, b| a + b,
            ScatterCombine::Mul => |a, b| a * b,
            ScatterCombine::Min => crate::jax_min_f64,
            ScatterCombine::Max => crate::jax_max_f64,
            ScatterCombine::Overwrite => |_, b| b,
        };
        scatter_typed!(op, upd, TensorValue::new_f64_values, cf);
    }
    // Dense F32 scatter (the embedding-gradient scatter-add case). f32 is JAX's
    // DEFAULT dtype. Overwrite is a contiguous copy; scatter-ADD computes
    // `(a as f64 + b as f64) as f32`, EXACTLY what the generic `binary_literal_op`
    // Add does for f32 operands (`literal_from_numeric_f64(F32, a_f64 + b_f64)`),
    // so it is bit-for-bit identical incl. repeated-index accumulation and NaN.
    if operand.dtype == DType::F32
        && let (Some(op), Some(upd)) = (
            operand.elements.as_f32_slice(),
            updates.elements.as_f32_slice(),
        )
    {
        // f32 promotes to f64, applies the op, rounds back — exactly
        // binary_literal_op's f32 branch for each combine.
        let cf: fn(f32, f32) -> f32 = match combine {
            ScatterCombine::Add => |a, b| (f64::from(a) + f64::from(b)) as f32,
            ScatterCombine::Mul => |a, b| (f64::from(a) * f64::from(b)) as f32,
            ScatterCombine::Min => |a, b| crate::jax_min_f64(f64::from(a), f64::from(b)) as f32,
            ScatterCombine::Max => |a, b| crate::jax_max_f64(f64::from(a), f64::from(b)) as f32,
            ScatterCombine::Overwrite => |_, b| b,
        };
        scatter_typed!(op, upd, TensorValue::new_f32_values, cf);
    }
    if operand.dtype == DType::I64
        && let (Some(op), Some(upd)) = (
            operand.elements.as_i64_slice(),
            updates.elements.as_i64_slice(),
        )
    {
        // i64 ops match scatter_combine_literal's I64 branch exactly.
        let cf: fn(i64, i64) -> i64 = match combine {
            ScatterCombine::Add => |a, b| a.wrapping_add(b),
            ScatterCombine::Mul => |a, b| a.wrapping_mul(b),
            ScatterCombine::Min => |a, b| a.min(b),
            ScatterCombine::Max => |a, b| a.max(b),
            ScatterCombine::Overwrite => |_, b| b,
        };
        scatter_typed!(op, upd, TensorValue::new_i64_values, cf);
    }
    // I32 (JAX's default int) shares the dense i64 backing; an i32 value is sign-extended
    // in its i64 slot, so the same i64 wrapping combiner is correct. The I32-tagged output
    // is wrapped mod 2^32 by the eval_primitive chokepoint (Add/Mul may overflow i32;
    // Min/Max/Overwrite select an in-range input). i32 otherwise ran the generic scatter.
    if operand.dtype == DType::I32
        && let (Some(op), Some(upd)) = (
            operand.elements.as_i64_slice(),
            updates.elements.as_i64_slice(),
        )
    {
        let cf: fn(i64, i64) -> i64 = match combine {
            ScatterCombine::Add => |a, b| a.wrapping_add(b),
            ScatterCombine::Mul => |a, b| a.wrapping_mul(b),
            ScatterCombine::Min => |a, b| a.min(b),
            ScatterCombine::Max => |a, b| a.max(b),
            ScatterCombine::Overwrite => |_, b| b,
        };
        scatter_typed!(op, upd, TensorValue::new_i32_values, cf);
    }
    // Dense BF16/F16 scatter (half-precision embedding update). bf16 is the dominant
    // training dtype. Overwrite is a contiguous u16-bit copy; scatter-ADD routes the
    // two half-float bit patterns through the SAME `binary_literal_op` Add the generic
    // path uses (widen u16 -> f64, add, round back to half), so it is bit-for-bit
    // identical including repeated-index accumulation and NaN. The `new_half_float_values`
    // ctor takes a dtype, so the macro's ctor arg is a closure.
    if matches!(operand.dtype, DType::BF16 | DType::F16)
        && let (Some(op), Some(upd)) = (
            operand.elements.as_half_float_slice(),
            updates.elements.as_half_float_slice(),
        )
    {
        let dt = operand.dtype;
        let half_lit = |bits: u16| -> Literal {
            if dt == DType::BF16 {
                Literal::BF16Bits(bits)
            } else {
                Literal::F16Bits(bits)
            }
        };
        // Each combiner routes the two half bit patterns through the SAME
        // binary_literal_op the generic scatter_combine_literal uses (widen u16->f64,
        // apply the op, round back to half), so it is bit-for-bit identical.
        let cf = move |a: u16, b: u16| -> u16 {
            let r = match combine {
                ScatterCombine::Add => binary_literal_op(
                    half_lit(a),
                    half_lit(b),
                    Primitive::Add,
                    &|x: i64, y: i64| x.wrapping_add(y),
                    &|x: f64, y: f64| x + y,
                ),
                ScatterCombine::Mul => binary_literal_op(
                    half_lit(a),
                    half_lit(b),
                    Primitive::Mul,
                    &|x: i64, y: i64| x.wrapping_mul(y),
                    &|x: f64, y: f64| x * y,
                ),
                ScatterCombine::Min => binary_literal_op(
                    half_lit(a),
                    half_lit(b),
                    Primitive::Min,
                    &|x: i64, y: i64| x.min(y),
                    &crate::jax_min_f64,
                ),
                ScatterCombine::Max => binary_literal_op(
                    half_lit(a),
                    half_lit(b),
                    Primitive::Max,
                    &|x: i64, y: i64| x.max(y),
                    &crate::jax_max_f64,
                ),
                ScatterCombine::Overwrite => Ok(half_lit(b)),
            };
            match r {
                Ok(Literal::BF16Bits(x) | Literal::F16Bits(x)) => x,
                _ => 0,
            }
        };
        scatter_typed!(
            op,
            upd,
            |shape, out| TensorValue::new_half_float_values(dt, shape, out),
            cf
        );
    }
    // Dense complex scatter: Overwrite (copy), Add (component-wise) and Mul (complex
    // multiply) over the dense `(re, im)` backing — closing a real coverage gap, the
    // generic per-`Literal` path ERRORS on complex (binary_literal_op has no complex
    // arm). Min/Max return None → generic (preserving the prior error, since complex
    // ordering for scatter min/max is intentionally unsupported). `new_complex_values`
    // rounds Complex64 to f32 exactly as the literal storage does → bit-exact.
    if matches!(operand.dtype, DType::Complex64 | DType::Complex128)
        && let (Some(op), Some(upd)) = (
            operand.elements.as_complex_slice(),
            updates.elements.as_complex_slice(),
        )
    {
        let dt = operand.dtype;
        let cf: fn((f64, f64), (f64, f64)) -> (f64, f64) = match combine {
            ScatterCombine::Overwrite => |_, b| b, // unused (is_overwrite copies)
            ScatterCombine::Add => |a, b| (a.0 + b.0, a.1 + b.1),
            ScatterCombine::Mul => |a, b| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0),
            // Complex min/max fold lexicographically on (re, im), bit-identically
            // to elementwise complex `lax.max`/`lax.min` (arithmetic::apply_complex_binary,
            // which selects via `complex_lex_ge`: `a >= b ? a : b` for Max, `b` for Min).
            // `current` is `a`, `update` is `b` — matching `binary_literal_op(current, update)`.
            ScatterCombine::Max => |a, b| if complex_lex_ge_scatter(a, b) { a } else { b },
            ScatterCombine::Min => |a, b| if complex_lex_ge_scatter(a, b) { b } else { a },
        };
        scatter_typed!(op, upd, |shape, out| TensorValue::new_complex_values(dt, shape, out), cf);
    }
    Ok(None)
}

/// Scatter: update positions in an operand tensor using indices and update values.
///
/// Simplified semantics (1-D index scatter):
///   operand: tensor to scatter into (cloned, not mutated)
///   indices: 1-D integer tensor of scatter indices (into axis 0 of operand)
///   updates: tensor of values to scatter; shape = [num_indices] ++ operand.shape[1..]
///
/// For each index i in `indices`, overwrites the slice at `operand[indices[i], ...]`
/// with the corresponding slice from `updates[i, ...]`.
pub(crate) fn eval_scatter(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Scatter;
    let _ = params; // No params needed for basic scatter
    if inputs.len() != 3 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }

    let operand = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "cannot scatter into a scalar".into(),
            });
        }
    };

    if operand.shape.rank() == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "cannot scatter into a rank-0 tensor".into(),
        });
    }

    let (index_vals, index_shape): (Vec<usize>, Vec<u32>) = match &inputs[1] {
        Value::Scalar(lit) => (vec![lit_to_usize(lit, primitive)?], Vec::new()),
        Value::Tensor(t) => (
            t.elements
                .iter()
                .map(|lit| lit_to_usize(lit, primitive))
                .collect::<Result<_, _>>()?,
            t.shape.dims.clone(),
        ),
    };

    let updates_dtype = inputs[2].dtype();
    if updates_dtype != operand.dtype {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "scatter updates dtype must match operand dtype",
        });
    }

    let mut expected_update_dims = index_shape;
    expected_update_dims.extend(operand.shape.dims.iter().skip(1).copied());
    let expected_update_shape = Shape {
        dims: expected_update_dims.clone(),
    };

    let scalar_updates_storage = match &inputs[2] {
        Value::Scalar(lit) => {
            if expected_update_dims.is_empty() {
                Some(TensorValue::new(
                    operand.dtype,
                    Shape::scalar(),
                    vec![*lit],
                )?)
            } else {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "updates must be a tensor for non-scalar scatter slices".into(),
                });
            }
        }
        Value::Tensor(_) => None,
    };
    let updates = match &inputs[2] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => match scalar_updates_storage.as_ref() {
            Some(tensor) => tensor,
            None => {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "scalar updates tensor construction failed".into(),
                });
            }
        },
    };

    let op_dims = &operand.shape.dims;

    // Number of elements per slice (product of dims[1..])
    let slice_elems = checked_shape_element_count(primitive, "scatter slice", &op_dims[1..])?;

    let mode = params
        .get("mode")
        .map(|s| s.as_str())
        .unwrap_or("overwrite");

    // Scatter combiner (`update_jaxpr`): `overwrite` = jax.lax.scatter (replace),
    // plus scatter_add/mul/min/max. Distinct from the OOB `index_mode` param below.
    let combine = match mode {
        "overwrite" => ScatterCombine::Overwrite,
        "add" => ScatterCombine::Add,
        "mul" => ScatterCombine::Mul,
        "min" => ScatterCombine::Min,
        "max" => ScatterCombine::Max,
        other => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "unknown scatter mode \"{other}\", expected \"overwrite\", \"add\", \"mul\", \"min\", or \"max\""
                ),
            });
        }
    };

    if updates.shape != expected_update_shape {
        return Err(EvalError::ShapeMismatch {
            primitive,
            left: updates.shape.clone(),
            right: expected_update_shape,
        });
    }

    let expected_update_elems =
        checked_shape_element_count(primitive, "scatter updates", &expected_update_dims)?;
    if updates.elements.len() != expected_update_elems {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "updates has {} elements but expected {} (num_indices={} * slice_elems={slice_elems})",
                updates.elements.len(),
                expected_update_elems,
                index_vals.len()
            ),
        });
    }

    // Out-of-bounds indices follow JAX GatherScatterMode (`index_mode` param; distinct
    // from the combiner `mode` above) and NEVER raise: `fill_or_drop` (the default,
    // matching `jnp` `.at[].set()`) silently drops out-of-bounds updates, `clip` clamps
    // them into range, `promise_in_bounds` clamps defensively.
    let index_mode = parse_index_mode(primitive, params, IndexMode::FillOrDrop)?;
    let dim0 = op_dims[0] as usize;

    if expected_update_elems == 0 {
        return Ok(Value::Tensor(operand.clone()));
    }

    // Dense F64/I64 fast path: scatter over the contiguous typed backing,
    // bypassing the Vec<Literal> materialization. Returns None for non-dense /
    // other dtypes -> generic below.
    if let Some(value) = eval_scatter_dense(
        operand,
        updates,
        &index_vals,
        slice_elems,
        dim0,
        index_mode,
        combine,
    )? {
        return Ok(value);
    }

    let mut result_elements = operand.elements.to_vec();

    for (i, &raw_idx) in index_vals.iter().enumerate() {
        let Some(idx) = resolve_axis0_index(raw_idx, dim0, index_mode) else {
            // FILL_OR_DROP: drop the out-of-bounds update slice.
            continue;
        };
        let base_offset = idx
            .checked_mul(slice_elems)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "scatter base offset overflows usize".to_owned(),
            })?;
        let update_offset = i
            .checked_mul(slice_elems)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "scatter update offset overflows usize".to_owned(),
            })?;

        if combine == ScatterCombine::Overwrite {
            let result_end =
                base_offset
                    .checked_add(slice_elems)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "scatter result slice end overflows usize".to_owned(),
                    })?;
            let update_end =
                update_offset
                    .checked_add(slice_elems)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "scatter update slice end overflows usize".to_owned(),
                    })?;
            if result_end > result_elements.len() {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "scatter result slice exceeds operand element count".to_owned(),
                });
            }
            if update_end > updates.elements.len() {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "scatter update slice exceeds update element count".to_owned(),
                });
            }
            result_elements[base_offset..result_end]
                .copy_from_slice(&updates.elements[update_offset..update_end]);
            continue;
        }

        for j in 0..slice_elems {
            let result_index =
                base_offset
                    .checked_add(j)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "scatter result index overflows usize".to_owned(),
                    })?;
            let update_index =
                update_offset
                    .checked_add(j)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "scatter update index overflows usize".to_owned(),
                    })?;
            let current =
                *result_elements
                    .get(result_index)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "scatter result index exceeds operand element count".to_owned(),
                    })?;
            let update =
                *updates
                    .elements
                    .get(update_index)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "scatter update index exceeds update element count".to_owned(),
                    })?;
            result_elements[result_index] = scatter_combine_literal(combine, current, update)?;
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        operand.dtype,
        operand.shape.clone(),
        result_elements,
    )?))
}

fn lit_to_usize(lit: &Literal, primitive: Primitive) -> Result<usize, EvalError> {
    match lit {
        Literal::I64(n) => {
            if *n >= 0 {
                Ok(*n as usize)
            } else {
                Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("negative index {n}"),
                })
            }
        }
        Literal::U32(n) => Ok(*n as usize),
        Literal::U64(n) => usize::try_from(*n).map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("index {n} exceeds usize range"),
        }),
        Literal::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Literal::BF16Bits(_) | Literal::F16Bits(_) | Literal::F32Bits(_) => {
            Err(EvalError::Unsupported {
                primitive,
                detail: "float indices not supported".into(),
            })
        }
        Literal::F64Bits(_) => Err(EvalError::Unsupported {
            primitive,
            detail: "float indices not supported".into(),
        }),
        Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => Err(EvalError::Unsupported {
            primitive,
            detail: "complex indices not supported".into(),
        }),
    }
}

fn normalize_dynamic_start(raw: i64, dim: i64, window: i64) -> usize {
    let adjusted = if raw < 0 { raw + dim } else { raw };
    // Defensive: callers (eval_dynamic_slice / eval_dynamic_update_slice)
    // already reject window > dim, but compute the upper bound so that an
    // accidental invariant break elsewhere can't cause the `as usize` cast
    // to wrap. With window > dim the upper bound would be negative; clamp
    // to 0 first.
    let max_start = (dim - window).max(0);
    adjusted.max(0).min(max_start) as usize
}

/// Dynamic slice: like slice but start indices are dynamic (runtime) values.
///
/// Inputs: [operand, start_0, start_1, ...] where start_i are scalar indices.
/// Params: `slice_sizes` — comma-separated sizes for the output slice along each axis.
///
/// JAX semantics: negative start indices are interpreted relative to the end,
/// then clamped to valid range [0, dim - size].
pub(crate) fn eval_dynamic_slice(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::DynamicSlice;
    if inputs.is_empty() {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: 0,
        });
    }

    let tensor = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "cannot dynamic_slice a scalar".into(),
            });
        }
    };

    let rank = tensor.shape.rank();
    let slice_sizes = parse_usize_param(primitive, "slice_sizes", params)?;

    if slice_sizes.len() != rank {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "slice_sizes length {} does not match operand rank {}",
                slice_sizes.len(),
                rank
            ),
        });
    }
    for (ax, &size) in slice_sizes.iter().enumerate() {
        let dim = tensor.shape.dims[ax] as usize;
        if size > dim {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("slice size {size} exceeds dimension {dim} on axis {ax}"),
            });
        }
    }

    // Start indices come from remaining inputs (one scalar per axis)
    if inputs.len() != 1 + rank {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1 + rank,
            actual: inputs.len(),
        });
    }

    let mut starts = Vec::with_capacity(rank);
    for ax in 0..rank {
        let start_val = match &inputs[1 + ax] {
            Value::Scalar(lit) => {
                let raw = match lit {
                    Literal::I64(v) => *v,
                    Literal::U32(v) => i64::from(*v),
                    Literal::U64(v) => i64::try_from(*v).unwrap_or(i64::MAX),
                    Literal::Bool(b) => {
                        if *b {
                            1
                        } else {
                            0
                        }
                    }
                    Literal::BF16Bits(_)
                    | Literal::F16Bits(_)
                    | Literal::F32Bits(_)
                    | Literal::F64Bits(_)
                    | Literal::Complex64Bits(..)
                    | Literal::Complex128Bits(..) => {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: format!("start index for axis {ax} must be integral dtype"),
                        });
                    }
                };
                let dim = tensor.shape.dims[ax] as i64;
                let size = slice_sizes[ax] as i64;
                normalize_dynamic_start(raw, dim, size)
            }
            Value::Tensor(_) => {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("start index for axis {ax} must be a scalar"),
                });
            }
        };
        starts.push(start_val);
    }

    // Validate slice sizes
    for ax in 0..rank {
        let dim = tensor.shape.dims[ax] as usize;
        let max_start = dim - slice_sizes[ax];
        if starts[ax] > max_start {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!(
                    "dynamic_slice out of bounds on axis {ax}: start={} max_start={}",
                    starts[ax], max_start
                ),
            });
        }
    }

    let out_dims: Vec<u32> = slice_sizes.iter().map(|&s| s as u32).collect();
    let total = checked_shape_element_count(primitive, "dynamic_slice", &out_dims)?;
    if total == 0 {
        return Ok(Value::Tensor(TensorValue::new(
            tensor.dtype,
            Shape { dims: out_dims },
            Vec::new(),
        )?));
    }

    // A trailing-contiguous slice (full extents + zero starts on every axis but
    // the first) is a single contiguous run [start_offset, end_offset); otherwise
    // we gather with an odometer. Computed once and shared by the dense and Literal
    // paths below.
    let has_contiguous_trailing_slice = rank > 0
        && slice_sizes
            .iter()
            .skip(1)
            .zip(tensor.shape.dims.iter().skip(1))
            .all(|(&slice_size, &dim)| slice_size == dim as usize)
        && starts.iter().skip(1).all(|&start| start == 0);
    let contig_range = if has_contiguous_trailing_slice {
        let row_len =
            checked_shape_element_count(primitive, "dynamic_slice row", &tensor.shape.dims[1..])?;
        let start_offset =
            starts[0]
                .checked_mul(row_len)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "dynamic_slice start offset overflows usize".to_owned(),
                })?;
        let end_offset = start_offset
            .checked_add(total)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "dynamic_slice end offset overflows usize".to_owned(),
            })?;
        Some((start_offset, end_offset))
    } else {
        None
    };

    let in_strides = checked_row_major_strides(primitive, "dynamic_slice", &tensor.shape.dims)?;

    // Dense fast paths: gather the slice straight from the typed backing into dense
    // output, avoiding the per-`Literal` materialization (`tensor.elements[..]`
    // reconstructs a 24-byte Literal each access) AND the boxed output. Same index
    // math/order -> bit-for-bit identical. Critical alongside dense
    // dynamic_update_slice for scan/decode loops (read the slice, compute, write it
    // back) — keeps the whole loop on dense storage. Falls through for other dtypes.
    macro_rules! dense_ds {
        ($src:expr, $ctor:expr) => {{
            let out = dynamic_slice_dense(
                $src,
                rank,
                total,
                &slice_sizes,
                &starts,
                &in_strides,
                contig_range,
            );
            return Ok(Value::Tensor($ctor(Shape { dims: out_dims }, out)?));
        }};
    }
    if let Some(s) = tensor.elements.as_f64_slice() {
        dense_ds!(s, TensorValue::new_f64_values);
    }
    if let Some(s) = tensor.elements.as_f32_slice() {
        dense_ds!(s, TensorValue::new_f32_values);
    }
    if let Some(s) = tensor.elements.as_half_float_slice() {
        let dt = tensor.dtype;
        dense_ds!(s, |sh, o| TensorValue::new_half_float_values(dt, sh, o));
    }
    if let Some(s) = tensor.elements.as_i64_slice() {
        dense_ds!(s, |sh, o| if tensor.dtype == DType::I32 {
            TensorValue::new_i32_values(sh, o)
        } else {
            TensorValue::new_i64_values(sh, o)
        });
    }

    // Literal fallback (boxed/other dtypes): the same gather over Literals.
    if let Some((start_offset, end_offset)) = contig_range {
        return Ok(Value::Tensor(TensorValue::new(
            tensor.dtype,
            Shape { dims: out_dims },
            tensor.elements[start_offset..end_offset].to_vec(),
        )?));
    }

    let mut elements = Vec::with_capacity(total);
    let mut out_coords = vec![0_usize; rank];
    for _ in 0..total {
        let mut in_flat = 0_usize;
        for ax in 0..rank {
            in_flat += (out_coords[ax] + starts[ax]) * in_strides[ax];
        }
        elements.push(tensor.elements[in_flat]);
        for ax in (0..rank).rev() {
            out_coords[ax] += 1;
            if out_coords[ax] < slice_sizes[ax] {
                break;
            }
            out_coords[ax] = 0;
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        tensor.dtype,
        Shape { dims: out_dims },
        elements,
    )?))
}

/// Dense, type-generic core of [`eval_dynamic_slice`]: gather the slice from a
/// contiguous typed backing into dense output. `contig_range` Some is a single
/// contiguous run (memcpy); None walks the output with an odometer. Identical
/// index math/order to the `Literal` path -> bit-for-bit identical.
fn dynamic_slice_dense<T: Copy>(
    src: &[T],
    rank: usize,
    total: usize,
    slice_sizes: &[usize],
    starts: &[usize],
    in_strides: &[usize],
    contig_range: Option<(usize, usize)>,
) -> Vec<T> {
    if let Some((start, end)) = contig_range {
        return src[start..end].to_vec();
    }
    let mut out = Vec::with_capacity(total);
    let mut out_coords = vec![0_usize; rank];
    for _ in 0..total {
        let mut in_flat = 0_usize;
        for ax in 0..rank {
            in_flat += (out_coords[ax] + starts[ax]) * in_strides[ax];
        }
        out.push(src[in_flat]);
        for ax in (0..rank).rev() {
            out_coords[ax] += 1;
            if out_coords[ax] < slice_sizes[ax] {
                break;
            }
            out_coords[ax] = 0;
        }
    }
    out
}

/// DynamicUpdateSlice: write `update` into `operand` at position given by start indices.
///
/// Inputs: [operand, update, start_0, start_1, ..., start_{rank-1}]
/// Output: tensor with same shape as operand, with the update region overwritten.
pub(crate) fn eval_dynamic_update_slice(
    inputs: &[Value],
    _params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::DynamicUpdateSlice;
    if inputs.len() < 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }

    let operand = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "cannot dynamic_update_slice a scalar".into(),
            });
        }
    };

    let update = match &inputs[1] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "update must be a tensor".into(),
            });
        }
    };
    if update.dtype != operand.dtype {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "update dtype must match operand dtype",
        });
    }

    let rank = operand.shape.rank();
    if update.shape.rank() != rank {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "update rank {} != operand rank {}",
                update.shape.rank(),
                rank
            ),
        });
    }
    for ax in 0..rank {
        let dim = operand.shape.dims[ax];
        let upd = update.shape.dims[ax];
        if upd > dim {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("update dim {upd} exceeds operand dim {dim} on axis {ax}"),
            });
        }
    }

    if inputs.len() != 2 + rank {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2 + rank,
            actual: inputs.len(),
        });
    }

    // Parse start indices (one scalar per axis), adjusted then clamped to valid range.
    let mut starts = Vec::with_capacity(rank);
    for ax in 0..rank {
        let start_val = match &inputs[2 + ax] {
            Value::Scalar(lit) => {
                let raw = match lit {
                    Literal::I64(v) => *v,
                    Literal::U32(v) => i64::from(*v),
                    Literal::U64(v) => i64::try_from(*v).unwrap_or(i64::MAX),
                    Literal::Bool(b) => i64::from(*b),
                    Literal::BF16Bits(_)
                    | Literal::F16Bits(_)
                    | Literal::F32Bits(_)
                    | Literal::F64Bits(_)
                    | Literal::Complex64Bits(..)
                    | Literal::Complex128Bits(..) => {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: format!("start index for axis {ax} must be integral dtype"),
                        });
                    }
                };
                let dim = operand.shape.dims[ax] as i64;
                let upd_size = update.shape.dims[ax] as i64;
                normalize_dynamic_start(raw, dim, upd_size)
            }
            Value::Tensor(_) => {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("start index for axis {ax} must be a scalar"),
                });
            }
        };
        starts.push(start_val);
    }

    let upd_total = update.elements.len();
    if upd_total == 0 {
        // Nothing to write — output equals the operand exactly. Zero-copy clone
        // (cheap Arc bump; dense/concat storage preserved).
        return Ok(Value::Tensor(TensorValue::new_with_literal_buffer(
            operand.dtype,
            operand.shape.clone(),
            operand.elements.clone(),
        )?));
    }

    // A trailing-contiguous update (full extents + zero starts on every axis but
    // the first) is a single contiguous run at `start_offset`; otherwise we walk
    // the update with an odometer. Computed once and shared by the dense and
    // Literal paths below.
    let has_contiguous_trailing_update = rank > 0
        && update
            .shape
            .dims
            .iter()
            .skip(1)
            .zip(operand.shape.dims.iter().skip(1))
            .all(|(&update_dim, &operand_dim)| update_dim == operand_dim)
        && starts.iter().skip(1).all(|&start| start == 0);
    let contig_start = if has_contiguous_trailing_update {
        let row_len = checked_shape_element_count(
            primitive,
            "dynamic_update_slice row",
            &operand.shape.dims[1..],
        )?;
        let start_offset =
            starts[0]
                .checked_mul(row_len)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "dynamic_update_slice start offset overflows usize".to_owned(),
                })?;
        // Validate the run stays in bounds (overflow + range), as before.
        let end_offset =
            start_offset
                .checked_add(upd_total)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "dynamic_update_slice end offset overflows usize".to_owned(),
                })?;
        if end_offset > operand.elements.len() {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "dynamic_update_slice update region exceeds operand".to_owned(),
            });
        }
        Some(start_offset)
    } else {
        None
    };

    let op_strides =
        checked_row_major_strides(primitive, "dynamic_update_slice", &operand.shape.dims)?;

    // Dense fast paths: clone the operand's typed backing and overwrite the update
    // region straight from the update's typed backing into dense output. This skips
    // materializing the (large) operand into a `Vec<Literal>` AND preserves dense
    // storage — critical for the autoregressive KV-cache loop, where the operand is
    // re-`dynamic_update_slice`d every decode step and a boxed result would demote
    // every downstream op. Identical writes/order -> bit-for-bit identical. Falls
    // through to the Literal path for boxed/other dtypes.
    macro_rules! dense_dus {
        ($op:expr, $upd:expr, $ctor:expr) => {{
            let out = dynamic_update_slice_dense(
                $op,
                $upd,
                rank,
                &update.shape.dims,
                &starts,
                &op_strides,
                contig_start,
            );
            return Ok(Value::Tensor($ctor(operand.shape.clone(), out)?));
        }};
    }
    if let (Some(o), Some(u)) = (
        operand.elements.as_f64_slice(),
        update.elements.as_f64_slice(),
    ) {
        dense_dus!(o, u, TensorValue::new_f64_values);
    }
    if let (Some(o), Some(u)) = (
        operand.elements.as_f32_slice(),
        update.elements.as_f32_slice(),
    ) {
        dense_dus!(o, u, TensorValue::new_f32_values);
    }
    if let (Some(o), Some(u)) = (
        operand.elements.as_half_float_slice(),
        update.elements.as_half_float_slice(),
    ) {
        let dt = operand.dtype;
        dense_dus!(o, u, |sh, out| TensorValue::new_half_float_values(
            dt, sh, out
        ));
    }
    if let (Some(o), Some(u)) = (
        operand.elements.as_i64_slice(),
        update.elements.as_i64_slice(),
    ) {
        dense_dus!(o, u, |sh, out| if operand.dtype == DType::I32 {
            TensorValue::new_i32_values(sh, out)
        } else {
            TensorValue::new_i64_values(sh, out)
        });
    }

    // Literal fallback (boxed/other dtypes): the same copy over Literals.
    let mut elements = operand.elements.to_vec();
    if let Some(start_offset) = contig_start {
        elements[start_offset..start_offset + upd_total].copy_from_slice(&update.elements);
        return Ok(Value::Tensor(TensorValue::new(
            operand.dtype,
            operand.shape.clone(),
            elements,
        )?));
    }

    let mut upd_coords = vec![0_usize; rank];
    for upd_flat in 0..upd_total {
        let mut op_flat = 0_usize;
        for ax in 0..rank {
            op_flat += (upd_coords[ax] + starts[ax]) * op_strides[ax];
        }
        if op_flat < elements.len() {
            elements[op_flat] = update.elements[upd_flat];
        }
        for ax in (0..rank).rev() {
            upd_coords[ax] += 1;
            if upd_coords[ax] < update.shape.dims[ax] as usize {
                break;
            }
            upd_coords[ax] = 0;
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        operand.dtype,
        operand.shape.clone(),
        elements,
    )?))
}

/// Dense, type-generic core of [`eval_dynamic_update_slice`]: clone the operand's
/// contiguous typed backing and overwrite the update region from the update's
/// typed backing, emitting dense output. `contig_start` Some means the update is a
/// single contiguous run at that offset; None means walk it with the odometer.
/// Identical writes and order to the `Literal` path -> bit-for-bit identical.
fn dynamic_update_slice_dense<T: Copy>(
    op_src: &[T],
    upd: &[T],
    rank: usize,
    update_dims: &[u32],
    starts: &[usize],
    op_strides: &[usize],
    contig_start: Option<usize>,
) -> Vec<T> {
    let mut out = op_src.to_vec();
    if let Some(start) = contig_start {
        out[start..start + upd.len()].copy_from_slice(upd);
        return out;
    }
    let mut upd_coords = vec![0_usize; rank];
    for &uv in upd.iter() {
        let mut op_flat = 0_usize;
        for ax in 0..rank {
            op_flat += (upd_coords[ax] + starts[ax]) * op_strides[ax];
        }
        if op_flat < out.len() {
            out[op_flat] = uv;
        }
        for ax in (0..rank).rev() {
            upd_coords[ax] += 1;
            if upd_coords[ax] < update_dims[ax] as usize {
                break;
            }
            upd_coords[ax] = 0;
        }
    }
    out
}

/// Copy: explicit identity operation that returns an independent cloned value.
pub(crate) fn eval_copy(inputs: &[Value]) -> Result<Value, EvalError> {
    let primitive = Primitive::Copy;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }
    Ok(inputs[0].clone())
}

/// BitcastConvertType: reinterpret element bit patterns as a new dtype.
///
/// Params:
/// - `new_dtype`: destination dtype string (e.g. `i32`, `f32`, `i64`, `f64`)
///
/// Constraint:
/// - Source and destination element widths must match.
pub(crate) fn eval_convert_element_type(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::ConvertElementType;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let target_dtype = parse_dtype_param(primitive, "new_dtype", params)?;

    match &inputs[0] {
        Value::Scalar(literal) => {
            let converted = convert_literal(*literal, target_dtype)?;
            Ok(Value::Scalar(converted))
        }
        Value::Tensor(tensor) => {
            // Dense-output fast path: a dense source slice (F64/I64) converted to
            // a dense-output target (F64/I64/Bool, exact dtype) is a tight typed
            // loop into dense storage — bypassing both the input Literal
            // materialization AND the per-element convert_literal + Vec<Literal>
            // output build. Values are exactly what convert_literal produces.
            let shape = tensor.shape.clone();
            if let Some(values) = tensor.elements.as_f64_slice() {
                let dense = match target_dtype {
                    // from_f64(v): identity in value.
                    DType::F64 => Some(TensorValue::new_f64_values(shape.clone(), values.to_vec())),
                    // from_f32(v as f32): mixed-precision downcast (the hot ML cast).
                    DType::F32 => Some(TensorValue::new_f32_values(
                        shape.clone(),
                        values.iter().map(|&v| v as f32).collect(),
                    )),
                    // from_f16_f64(v) / from_bf16_f64(v): single-rounded half-float downcast.
                    DType::F16 => Some(TensorValue::new_half_float_values(
                        DType::F16,
                        shape.clone(),
                        values.iter().map(|&v| convert_f16_bits(v)).collect(),
                    )),
                    DType::BF16 => Some(TensorValue::new_half_float_values(
                        DType::BF16,
                        shape.clone(),
                        values.iter().map(|&v| convert_bf16_bits(v)).collect(),
                    )),
                    // f64->complex (FFT input prep): re = v, im = 0. new_complex_values
                    // rounds Complex64's re to f32, matching convert_literal's
                    // from_complex64(v as f32, 0) / from_complex128(v, 0). Was per-Literal.
                    DType::Complex128 | DType::Complex64 => Some(TensorValue::new_complex_values(
                        target_dtype,
                        shape.clone(),
                        values.iter().map(|&v| (v, 0.0)).collect(),
                    )),
                    // i64_val(F64Bits) == v as i64 (NaN->0, +-inf saturate).
                    DType::I64 => Some(TensorValue::new_i64_values(
                        shape.clone(),
                        values.iter().map(|&v| v as i64).collect(),
                    )),
                    // f64->i32 (quantization / float-to-index): raw `v as i64` into the
                    // I32 tensor; the narrow chokepoint wraps mod 2^32, exactly as
                    // convert_literal (Literal::I64(v as i64) tagged I32). Was per-Literal.
                    DType::I32 => Some(TensorValue::new_i32_values(
                        shape.clone(),
                        values.iter().map(|&v| v as i64).collect(),
                    )),
                    // bool_val(F64Bits) == v != 0.0 (NaN -> true).
                    DType::Bool => Some(TensorValue::new_bool_values(
                        shape.clone(),
                        values.iter().map(|&v| v != 0.0).collect(),
                    )),
                    _ => None,
                };
                if let Some(t) = dense {
                    return Ok(Value::Tensor(t.map_err(EvalError::InvalidTensor)?));
                }
            } else if let Some(values) = tensor.elements.as_f32_slice() {
                // F32 source (dense-f32 backing) — the activation/mixed-precision
                // cast hot path. convert_literal routes F32 through f64_val =
                // f64::from(v) for float targets, but I64 uses `v as i64` on the
                // f32 directly and Bool uses `v != 0.0` — replicated exactly here.
                let dense = match target_dtype {
                    DType::F32 => Some(TensorValue::new_f32_values(shape.clone(), values.to_vec())),
                    DType::F64 => Some(TensorValue::new_f64_values(
                        shape.clone(),
                        values.iter().map(|&v| f64::from(v)).collect(),
                    )),
                    DType::F16 => Some(TensorValue::new_half_float_values(
                        DType::F16,
                        shape.clone(),
                        values
                            .iter()
                            .map(|&v| convert_f16_bits(f64::from(v)))
                            .collect(),
                    )),
                    DType::BF16 => Some(TensorValue::new_half_float_values(
                        DType::BF16,
                        shape.clone(),
                        values
                            .iter()
                            .map(|&v| convert_bf16_bits(f64::from(v)))
                            .collect(),
                    )),
                    // f32->complex: re = f64::from(v), im = 0 (new_complex_values rounds
                    // Complex64's re to f32 == convert_literal's from_complex64(v, 0)).
                    DType::Complex128 | DType::Complex64 => Some(TensorValue::new_complex_values(
                        target_dtype,
                        shape.clone(),
                        values.iter().map(|&v| (f64::from(v), 0.0)).collect(),
                    )),
                    // i64_val(F32Bits) == f32 as i64 (NOT via f64).
                    DType::I64 => Some(TensorValue::new_i64_values(
                        shape.clone(),
                        values.iter().map(|&v| v as i64).collect(),
                    )),
                    // f32->i32: raw `f32 as i64` into the I32 tensor (chokepoint wraps
                    // mod 2^32), matching convert_literal. Was per-Literal.
                    DType::I32 => Some(TensorValue::new_i32_values(
                        shape.clone(),
                        values.iter().map(|&v| v as i64).collect(),
                    )),
                    DType::Bool => Some(TensorValue::new_bool_values(
                        shape.clone(),
                        values.iter().map(|&v| v != 0.0).collect(),
                    )),
                    _ => None,
                };
                if let Some(t) = dense {
                    return Ok(Value::Tensor(t.map_err(EvalError::InvalidTensor)?));
                }
            } else if let Some(values) = tensor.elements.as_i64_slice() {
                // Source is dense I64 OR I32 (both back onto the i64 slice). For an I32
                // source each value is the i32 already sign-extended, so the same casts
                // apply. F32/I32 targets (int→float32 and i64→i32 narrowing — both common
                // ML casts, F32 is JAX's default float) previously fell to the per-Literal
                // path. F32 uses `(v as f64) as f32` to match convert_literal's
                // `from_f32(f64_val as f32)` (double-round-safe for |v| > 2^24). I32 stores
                // the raw value and lets the narrow chokepoint wrap mod 2^32, exactly as
                // convert_literal (`Literal::I64(i64_val)` tagged I32) does.
                let dense = match target_dtype {
                    DType::F64 => Some(TensorValue::new_f64_values(
                        shape.clone(),
                        values.iter().map(|&v| v as f64).collect(),
                    )),
                    DType::F32 => Some(TensorValue::new_f32_values(
                        shape.clone(),
                        values.iter().map(|&v| (v as f64) as f32).collect(),
                    )),
                    // int->half (mixed-precision): convert_bf16/f16_bits(v as f64) matches
                    // convert_literal (f64_val = v as f64, then single-round to half).
                    DType::F16 => Some(TensorValue::new_half_float_values(
                        DType::F16,
                        shape.clone(),
                        values.iter().map(|&v| convert_f16_bits(v as f64)).collect(),
                    )),
                    DType::BF16 => Some(TensorValue::new_half_float_values(
                        DType::BF16,
                        shape.clone(),
                        values.iter().map(|&v| convert_bf16_bits(v as f64)).collect(),
                    )),
                    DType::I64 => Some(TensorValue::new_i64_values(shape.clone(), values.to_vec())),
                    DType::I32 => Some(TensorValue::new_i32_values(shape.clone(), values.to_vec())),
                    DType::Bool => Some(TensorValue::new_bool_values(
                        shape.clone(),
                        values.iter().map(|&v| v != 0).collect(),
                    )),
                    _ => None,
                };
                if let Some(t) = dense {
                    return Ok(Value::Tensor(t.map_err(EvalError::InvalidTensor)?));
                }
            } else if let Some(values) = tensor.elements.as_half_float_slice() {
                // Half-float source (BF16/F16) — the mixed-precision UPCAST hot path
                // (decode + compute in f32/f64). convert_literal decodes each tap via
                // as_{bf16,f16}_f32 -> f32, then routes through f64_val = f64::from(v)
                // for float targets; I64 uses `f32 as i64` and Bool `f32 != 0.0`.
                // Decode here from the raw u16 backing (dtype-tagged), bit-identical.
                let is_bf16 = tensor.dtype == DType::BF16;
                let decode = |b: u16| -> f32 {
                    if is_bf16 {
                        Literal::BF16Bits(b).as_bf16_f32().unwrap_or(0.0)
                    } else {
                        Literal::F16Bits(b).as_f16_f32().unwrap_or(0.0)
                    }
                };
                let dense = match target_dtype {
                    // from_f32(f64::from(v) as f32) == from_f32(v): exact round-trip.
                    DType::F32 => Some(TensorValue::new_f32_values(
                        shape.clone(),
                        values.iter().map(|&b| decode(b)).collect(),
                    )),
                    // from_f64(f64::from(v)): exact widen.
                    DType::F64 => Some(TensorValue::new_f64_values(
                        shape.clone(),
                        values.iter().map(|&b| f64::from(decode(b))).collect(),
                    )),
                    // Cross/same half: from_{f16,bf16}_f64(f64::from(v)).
                    DType::F16 => Some(TensorValue::new_half_float_values(
                        DType::F16,
                        shape.clone(),
                        values
                            .iter()
                            .map(|&b| convert_f16_bits(f64::from(decode(b))))
                            .collect(),
                    )),
                    DType::BF16 => Some(TensorValue::new_half_float_values(
                        DType::BF16,
                        shape.clone(),
                        values
                            .iter()
                            .map(|&b| convert_bf16_bits(f64::from(decode(b))))
                            .collect(),
                    )),
                    // i64_val half branch == `f32 as i64`.
                    DType::I64 => Some(TensorValue::new_i64_values(
                        shape.clone(),
                        values.iter().map(|&b| decode(b) as i64).collect(),
                    )),
                    // half->i32: `decode(b) as i64` into the I32 tensor (chokepoint wraps),
                    // matching convert_literal's Literal::I64(decode as i64) tagged I32.
                    DType::I32 => Some(TensorValue::new_i32_values(
                        shape.clone(),
                        values.iter().map(|&b| decode(b) as i64).collect(),
                    )),
                    // bool_val half branch == `f32 != 0.0`.
                    DType::Bool => Some(TensorValue::new_bool_values(
                        shape.clone(),
                        values.iter().map(|&b| decode(b) != 0.0).collect(),
                    )),
                    _ => None,
                };
                if let Some(t) = dense {
                    return Ok(Value::Tensor(t.map_err(EvalError::InvalidTensor)?));
                }
            } else if let Some(values) = tensor.elements.as_complex_slice() {
                // Complex source (FFT/signal pipelines): real-part extraction for float
                // targets (im dropped, matching convert_literal's `re` read) and
                // complex->complex precision change (new_complex_values rounds a Complex64
                // target's components to f32, == convert_literal's from_complex64).
                let dense = match target_dtype {
                    DType::F64 => Some(TensorValue::new_f64_values(
                        shape.clone(),
                        values.iter().map(|&(re, _)| re).collect(),
                    )),
                    DType::F32 => Some(TensorValue::new_f32_values(
                        shape.clone(),
                        values.iter().map(|&(re, _)| re as f32).collect(),
                    )),
                    DType::Complex128 | DType::Complex64 => Some(TensorValue::new_complex_values(
                        target_dtype,
                        shape.clone(),
                        values.to_vec(),
                    )),
                    // complex->int/bool reads the real part (im dropped), matching
                    // convert_literal (i64_val/bool reads re; bool ORs im). I32 lets the
                    // narrow chokepoint wrap; Bool keeps the JAX `re!=0 || im!=0` rule.
                    DType::I64 => Some(TensorValue::new_i64_values(
                        shape.clone(),
                        values.iter().map(|&(re, _)| re as i64).collect(),
                    )),
                    DType::I32 => Some(TensorValue::new_i32_values(
                        shape.clone(),
                        values.iter().map(|&(re, _)| re as i64).collect(),
                    )),
                    DType::Bool => Some(TensorValue::new_bool_values(
                        shape.clone(),
                        values.iter().map(|&(re, im)| re != 0.0 || im != 0.0).collect(),
                    )),
                    _ => None,
                };
                if let Some(t) = dense {
                    return Ok(Value::Tensor(t.map_err(EvalError::InvalidTensor)?));
                }
            } else if let Some(values) = tensor.elements.as_u32_slice() {
                // u32 source (threefry RNG output -> float is the hot path). Mirrors
                // convert_literal exactly: f64_val = v as f64, i64_val = i64::from(v),
                // u64_val = u64::from(v), bool = v != 0. F32 uses (v as f64) as f32.
                let dense = match target_dtype {
                    DType::F64 => Some(TensorValue::new_f64_values(
                        shape.clone(),
                        values.iter().map(|&v| v as f64).collect(),
                    )),
                    DType::F32 => Some(TensorValue::new_f32_values(
                        shape.clone(),
                        values.iter().map(|&v| (v as f64) as f32).collect(),
                    )),
                    DType::F16 => Some(TensorValue::new_half_float_values(
                        DType::F16,
                        shape.clone(),
                        values.iter().map(|&v| convert_f16_bits(v as f64)).collect(),
                    )),
                    DType::BF16 => Some(TensorValue::new_half_float_values(
                        DType::BF16,
                        shape.clone(),
                        values.iter().map(|&v| convert_bf16_bits(v as f64)).collect(),
                    )),
                    DType::I64 => Some(TensorValue::new_i64_values(
                        shape.clone(),
                        values.iter().map(|&v| i64::from(v)).collect(),
                    )),
                    // u32->i32: raw i64::from(v) into I32 tensor; chokepoint wraps mod 2^32,
                    // matching convert_literal (Literal::I64(i64_val) tagged I32).
                    DType::I32 => Some(TensorValue::new_i32_values(
                        shape.clone(),
                        values.iter().map(|&v| i64::from(v)).collect(),
                    )),
                    DType::U64 => Some(TensorValue::new_u64_values(
                        shape.clone(),
                        values.iter().map(|&v| u64::from(v)).collect(),
                    )),
                    // u32->u32 identity (u64::from(v) as u32 == v).
                    DType::U32 => Some(TensorValue::new_u32_values(shape.clone(), values.to_vec())),
                    DType::Bool => Some(TensorValue::new_bool_values(
                        shape.clone(),
                        values.iter().map(|&v| v != 0).collect(),
                    )),
                    DType::Complex128 | DType::Complex64 => Some(TensorValue::new_complex_values(
                        target_dtype,
                        shape.clone(),
                        values.iter().map(|&v| (v as f64, 0.0)).collect(),
                    )),
                };
                if let Some(t) = dense {
                    return Ok(Value::Tensor(t.map_err(EvalError::InvalidTensor)?));
                }
            } else if let Some(values) = tensor.elements.as_u64_slice() {
                // u64 source. convert_literal: f64_val = v as f64; i64_val =
                // i64::try_from(v).ok() (>i64::MAX -> None -> 0); u64_val = v;
                // U32 target = v as u32; bool = v != 0.
                let dense = match target_dtype {
                    DType::F64 => Some(TensorValue::new_f64_values(
                        shape.clone(),
                        values.iter().map(|&v| v as f64).collect(),
                    )),
                    DType::F32 => Some(TensorValue::new_f32_values(
                        shape.clone(),
                        values.iter().map(|&v| (v as f64) as f32).collect(),
                    )),
                    DType::F16 => Some(TensorValue::new_half_float_values(
                        DType::F16,
                        shape.clone(),
                        values.iter().map(|&v| convert_f16_bits(v as f64)).collect(),
                    )),
                    DType::BF16 => Some(TensorValue::new_half_float_values(
                        DType::BF16,
                        shape.clone(),
                        values.iter().map(|&v| convert_bf16_bits(v as f64)).collect(),
                    )),
                    // i64_val for U64 saturates to 0 above i64::MAX (matches .ok().unwrap_or(0)).
                    DType::I64 => Some(TensorValue::new_i64_values(
                        shape.clone(),
                        values.iter().map(|&v| i64::try_from(v).unwrap_or(0)).collect(),
                    )),
                    DType::I32 => Some(TensorValue::new_i32_values(
                        shape.clone(),
                        values.iter().map(|&v| i64::try_from(v).unwrap_or(0)).collect(),
                    )),
                    DType::U64 => Some(TensorValue::new_u64_values(shape.clone(), values.to_vec())),
                    DType::U32 => Some(TensorValue::new_u32_values(
                        shape.clone(),
                        values.iter().map(|&v| v as u32).collect(),
                    )),
                    DType::Bool => Some(TensorValue::new_bool_values(
                        shape.clone(),
                        values.iter().map(|&v| v != 0).collect(),
                    )),
                    DType::Complex128 | DType::Complex64 => Some(TensorValue::new_complex_values(
                        target_dtype,
                        shape.clone(),
                        values.iter().map(|&v| (v as f64, 0.0)).collect(),
                    )),
                };
                if let Some(t) = dense {
                    return Ok(Value::Tensor(t.map_err(EvalError::InvalidTensor)?));
                }
            }

            let mut out = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                out.push(convert_literal(*literal, target_dtype)?);
            }
            Ok(Value::Tensor(TensorValue::new(target_dtype, shape, out)?))
        }
    }
}

/// f64 -> f16 raw bits, matching `Literal::from_f16_f64` exactly (single-rounded
/// via the round-to-odd f32 intermediate). Used by the dense convert fast paths
/// so a half-float output stays in dense `Vec<u16>` storage.
#[inline]
fn convert_f16_bits(v: f64) -> u16 {
    match Literal::from_f16_f64(v) {
        Literal::F16Bits(b) => b,
        _ => unreachable!("from_f16_f64 always yields F16Bits"),
    }
}

/// f64 -> bf16 raw bits, matching `Literal::from_bf16_f64` exactly.
#[inline]
fn convert_bf16_bits(v: f64) -> u16 {
    match Literal::from_bf16_f64(v) {
        Literal::BF16Bits(b) => b,
        _ => unreachable!("from_bf16_f64 always yields BF16Bits"),
    }
}

fn convert_literal(lit: Literal, target: DType) -> Result<Literal, EvalError> {
    let f64_val = || -> Option<f64> {
        match lit {
            Literal::F64Bits(bits) => Some(f64::from_bits(bits)),
            Literal::F32Bits(bits) => Some(f64::from(f32::from_bits(bits))),
            Literal::F16Bits(_) => lit.as_f16_f32().map(f64::from),
            Literal::BF16Bits(_) => lit.as_bf16_f32().map(f64::from),
            Literal::I64(v) => Some(v as f64),
            Literal::U32(v) => Some(v as f64),
            Literal::U64(v) => Some(v as f64),
            Literal::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
            Literal::Complex64Bits(re, _) => Some(f64::from(f32::from_bits(re))),
            Literal::Complex128Bits(re, _) => Some(f64::from_bits(re)),
        }
    };

    let i64_val = || -> Option<i64> {
        match lit {
            Literal::I64(v) => Some(v),
            Literal::U32(v) => Some(i64::from(v)),
            Literal::U64(v) => i64::try_from(v).ok(),
            Literal::Bool(b) => Some(if b { 1 } else { 0 }),
            Literal::F64Bits(bits) => Some(f64::from_bits(bits) as i64),
            Literal::F32Bits(bits) => Some(f32::from_bits(bits) as i64),
            Literal::F16Bits(_) => lit.as_f16_f32().map(|v| v as i64),
            Literal::BF16Bits(_) => lit.as_bf16_f32().map(|v| v as i64),
            Literal::Complex64Bits(re, _) => Some(f32::from_bits(re) as i64),
            Literal::Complex128Bits(re, _) => Some(f64::from_bits(re) as i64),
        }
    };

    let u64_val = || -> Option<u64> {
        match lit {
            Literal::U64(v) => Some(v),
            Literal::U32(v) => Some(u64::from(v)),
            Literal::I64(v) => u64::try_from(v).ok(),
            Literal::Bool(b) => Some(if b { 1 } else { 0 }),
            Literal::F64Bits(bits) => Some(f64::from_bits(bits) as u64),
            Literal::F32Bits(bits) => Some(f32::from_bits(bits) as u64),
            Literal::F16Bits(_) => lit.as_f16_f32().map(|v| v as u64),
            Literal::BF16Bits(_) => lit.as_bf16_f32().map(|v| v as u64),
            Literal::Complex64Bits(re, _) => Some(f32::from_bits(re) as u64),
            Literal::Complex128Bits(re, _) => Some(f64::from_bits(re) as u64),
        }
    };

    let bool_val = || -> bool {
        match lit {
            Literal::Bool(b) => b,
            Literal::I64(v) => v != 0,
            Literal::U32(v) => v != 0,
            Literal::U64(v) => v != 0,
            Literal::F64Bits(bits) => f64::from_bits(bits) != 0.0,
            Literal::F32Bits(bits) => f32::from_bits(bits) != 0.0,
            Literal::F16Bits(_) => lit.as_f16_f32().map(|v| v != 0.0).unwrap_or(false),
            Literal::BF16Bits(_) => lit.as_bf16_f32().map(|v| v != 0.0).unwrap_or(false),
            Literal::Complex64Bits(re, im) => {
                f32::from_bits(re) != 0.0 || f32::from_bits(im) != 0.0
            }
            Literal::Complex128Bits(re, im) => {
                f64::from_bits(re) != 0.0 || f64::from_bits(im) != 0.0
            }
        }
    };

    Ok(match target {
        DType::F64 => Literal::from_f64(f64_val().unwrap_or(0.0)),
        DType::F32 => Literal::from_f32(f64_val().unwrap_or(0.0) as f32),
        // Round f64 -> f16/bf16 in a single step (round-to-nearest-even) like
        // XLA's ConvertElementType. The round-to-odd f32 intermediate prevents
        // the double rounding that a plain `as f32` would cause near a tie.
        DType::F16 => Literal::from_f16_f64(f64_val().unwrap_or(0.0)),
        DType::BF16 => Literal::from_bf16_f64(f64_val().unwrap_or(0.0)),
        DType::I64 | DType::I32 => Literal::I64(i64_val().unwrap_or(0)),
        DType::U64 => Literal::U64(u64_val().unwrap_or(0)),
        DType::U32 => Literal::U32(u64_val().unwrap_or(0) as u32),
        DType::Bool => Literal::Bool(bool_val()),
        DType::Complex64 => {
            // Preserve the source's imaginary part (a complex->complex precision cast must
            // keep both components; real->complex has im=0). Previously dropped im.
            let re = f64_val().unwrap_or(0.0) as f32;
            let im = match lit {
                Literal::Complex64Bits(_, im) => f32::from_bits(im),
                Literal::Complex128Bits(_, im) => f64::from_bits(im) as f32,
                _ => 0.0,
            };
            Literal::from_complex64(re, im)
        }
        DType::Complex128 => {
            let re = f64_val().unwrap_or(0.0);
            let im = match lit {
                Literal::Complex64Bits(_, im) => f64::from(f32::from_bits(im)),
                Literal::Complex128Bits(_, im) => f64::from_bits(im),
                _ => 0.0,
            };
            Literal::from_complex128(re, im)
        }
    })
}

pub(crate) fn eval_bitcast_convert_type(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::BitcastConvertType;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let target_dtype = parse_dtype_param(primitive, "new_dtype", params)?;
    let source_dtype = inputs[0].dtype();

    if dtype_bit_width(source_dtype) != dtype_bit_width(target_dtype) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "bitcast requires equal element widths: source={source_dtype:?}({} bits) target={target_dtype:?}({} bits)",
                dtype_bit_width(source_dtype),
                dtype_bit_width(target_dtype)
            ),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => {
            let raw = literal_to_bytes(primitive, source_dtype, *literal)?;
            let converted = bytes_to_literal(primitive, target_dtype, &raw)?;
            Ok(Value::Scalar(converted))
        }
        Value::Tensor(tensor) => {
            let mut out = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                let raw = literal_to_bytes(primitive, source_dtype, *literal)?;
                out.push(bytes_to_literal(primitive, target_dtype, &raw)?);
            }
            Ok(Value::Tensor(TensorValue::new(
                target_dtype,
                tensor.shape.clone(),
                out,
            )?))
        }
    }
}

/// Iota: generate a 1-D index tensor of length `length` with dtype `dtype`.
///
/// Inputs: [] (no inputs — iota is a nullary operation)
/// Params: `length` — number of elements, `dtype` — element type (default: I64)
pub(crate) fn eval_iota(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Iota;
    if !inputs.is_empty() {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 0,
            actual: inputs.len(),
        });
    }

    let length_str = params.get("length").ok_or_else(|| EvalError::Unsupported {
        primitive,
        detail: "missing required param 'length'".into(),
    })?;
    let length: u32 = length_str
        .trim()
        .parse()
        .map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("invalid length: '{length_str}'"),
        })?;

    let dtype = if let Some(raw) = params.get("dtype") {
        parse_dtype_name(primitive, "dtype", raw)?
    } else {
        DType::I64
    };
    let elements: Vec<Literal> = (0..length as usize)
        .map(|index| literal_from_index_for_dtype(primitive, dtype, index))
        .collect::<Result<_, _>>()?;

    Ok(Value::Tensor(TensorValue::new(
        dtype,
        Shape::vector(length),
        elements,
    )?))
}

fn literal_from_index_for_dtype(
    primitive: Primitive,
    dtype: DType,
    index: usize,
) -> Result<Literal, EvalError> {
    match dtype {
        DType::I64 => Ok(Literal::I64(index as i64)),
        DType::I32 => {
            let value = i32::try_from(index).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("index {index} exceeds i32 range"),
            })?;
            Ok(Literal::I64(i64::from(value)))
        }
        DType::U32 => {
            let value = u32::try_from(index).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("index {index} exceeds u32 range"),
            })?;
            Ok(Literal::U32(value))
        }
        DType::U64 => Ok(Literal::U64(index as u64)),
        DType::F64 => Ok(Literal::from_f64(index as f64)),
        DType::F32 => Ok(Literal::from_f32(index as f32)),
        DType::BF16 => Ok(Literal::from_bf16_f64(index as f64)),
        DType::F16 => Ok(Literal::from_f16_f64(index as f64)),
        DType::Complex64 => Ok(Literal::from_complex64(index as f32, 0.0)),
        DType::Complex128 => Ok(Literal::from_complex128(index as f64, 0.0)),
        DType::Bool => Err(EvalError::Unsupported {
            primitive,
            detail: format!("{} does not accept bool dtype", primitive.as_str()),
        }),
    }
}

/// BroadcastedIota: iota over `dimension` and broadcast across full `shape`.
///
/// Inputs: none.
/// Params:
/// - `shape`: comma-separated output dimensions
/// - `dimension`: axis carrying monotonically increasing indices
/// - `dtype`: optional output dtype (default `i64`)
pub(crate) fn eval_broadcasted_iota(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::BroadcastedIota;
    if !inputs.is_empty() {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 0,
            actual: inputs.len(),
        });
    }

    let shape_usize = parse_usize_param(primitive, "shape", params)?;
    let dimension = params
        .get("dimension")
        .map(|raw| {
            raw.trim()
                .parse::<usize>()
                .map_err(|_| EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid dimension: '{raw}'"),
                })
        })
        .transpose()?
        .unwrap_or(0);
    let dtype = if let Some(raw) = params.get("dtype") {
        parse_dtype_name(primitive, "dtype", raw)?
    } else {
        DType::I64
    };

    let shape_u32: Vec<u32> = shape_usize
        .iter()
        .map(|&d| {
            u32::try_from(d).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("shape dimension {d} exceeds u32 range"),
            })
        })
        .collect::<Result<_, _>>()?;

    if shape_usize.is_empty() {
        return Ok(Value::Scalar(literal_from_index_for_dtype(
            primitive, dtype, 0,
        )?));
    }

    if dimension >= shape_usize.len() {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "dimension {dimension} out of bounds for rank {}",
                shape_usize.len()
            ),
        });
    }

    let total = checked_shape_element_count(primitive, "broadcasted_iota", &shape_u32)?;
    if total == 0 {
        return Ok(Value::Tensor(TensorValue::new(
            dtype,
            Shape { dims: shape_u32 },
            Vec::new(),
        )?));
    }

    let stride = shape_usize[(dimension + 1)..]
        .iter()
        .try_fold(1_usize, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "broadcasted_iota stride overflows usize".to_owned(),
            })
        })?;
    let axis_extent = shape_usize[dimension];

    // Structural construction: the iota value at a flat index is
    // `(flat / stride) % axis_extent`, i.e. the coordinate along `dimension`.
    // That value is constant across each contiguous run of `stride` elements;
    // the runs cycle `0..axis_extent` to form a block of length
    // `axis_extent * stride`; and that block repeats `outer` times, where
    // `outer` is the product of the dimensions before `dimension`. So the whole
    // tensor is built with bulk per-value fills + memcpy block repeats
    // (`extend_from_within`), eliminating the per-element division entirely. The
    // `axis_values` (only `axis_extent` of them, not one per output element)
    // carry the exact same values `literal_from_index_for_dtype` would produce,
    // so the output is bit-for-bit identical to the prior per-element loop.
    //
    // `outer` divides `total` exactly (total = outer * axis_extent * stride) and
    // both `stride` and `axis_extent` are >= 1 here (a zero trailing dim would
    // have made `total == 0`, handled above).
    let block_len = axis_extent * stride;
    let outer = total / block_len;

    // `iota_from_axis_values` fills `stride` copies of each axis value to build
    // one block, then repeats that block `outer` times via `extend_from_within`
    // (a memcpy), generic over the element type so the dense (i64/f64) and
    // Literal paths share identical layout logic.
    fn iota_from_axis_values<T: Copy>(
        axis_values: &[T],
        stride: usize,
        outer: usize,
        total: usize,
    ) -> Vec<T> {
        let block_len = axis_values.len() * stride;
        let mut out = Vec::with_capacity(total);
        for &v in axis_values {
            out.resize(out.len() + stride, v);
        }
        for _ in 1..outer {
            out.extend_from_within(0..block_len);
        }
        out
    }

    // Dense-output fast paths for the common iota dtypes (I64 is the default).
    // I64/F64 never overflow (unlike I32/U32), and `a as i64` / `a as f64`
    // reproduce `literal_from_index_for_dtype` exactly.
    match dtype {
        DType::I64 => {
            let axis_values: Vec<i64> = (0..axis_extent as i64).collect();
            let out = iota_from_axis_values(&axis_values, stride, outer, total);
            return Ok(Value::Tensor(TensorValue::new_i64_values(
                Shape { dims: shape_u32 },
                out,
            )?));
        }
        DType::F64 => {
            let axis_values: Vec<f64> = (0..axis_extent).map(|a| a as f64).collect();
            let out = iota_from_axis_values(&axis_values, stride, outer, total);
            return Ok(Value::Tensor(TensorValue::new_f64_values(
                Shape { dims: shape_u32 },
                out,
            )?));
        }
        _ => {}
    }

    // Generic (Literal) path: build the small per-axis-value `Literal` table
    // once (this is where I32/U32 range checks happen), then the same fill +
    // block-repeat layout. `block_len` element-wise clones for the first block,
    // the rest memcpy.
    let axis_values: Vec<Literal> = (0..axis_extent)
        .map(|a| literal_from_index_for_dtype(primitive, dtype, a))
        .collect::<Result<_, _>>()?;
    let elements = iota_from_axis_values(&axis_values, stride, outer, total);

    Ok(Value::Tensor(TensorValue::new(
        dtype,
        Shape { dims: shape_u32 },
        elements,
    )?))
}

fn quantize_f64(value: f64, exponent_bits: u32, mantissa_bits: u32) -> f64 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }

    let exp_bits = exponent_bits.clamp(1, 11);
    let mant_bits = mantissa_bits.min(52);
    if exp_bits == 11 && mant_bits == 52 {
        return value;
    }

    let bits = value.to_bits();
    let sign = bits & (1_u64 << 63);
    let mut exp = ((bits >> 52) & 0x7ff) as i32;
    let mut mant = bits & ((1_u64 << 52) - 1);

    if exp == 0 {
        return f64::from_bits(sign);
    }

    let unbiased = exp - 1023;
    if exp_bits < 11 {
        let bias_new = (1_i32 << (exp_bits - 1)) - 1;
        let min_unbiased = 1 - bias_new;
        let max_unbiased = bias_new;
        if unbiased > max_unbiased {
            return f64::from_bits(sign | (0x7ff_u64 << 52));
        }
        if unbiased < min_unbiased {
            return f64::from_bits(sign);
        }
    }

    if mant_bits < 52 {
        // Round the mantissa to `mant_bits` bits, to nearest with ties-to-even
        // (XLA reduce_precision semantics — NOT truncation).
        let drop = 52 - mant_bits;
        let truncated = (mant >> drop) << drop;
        let remainder = mant & ((1_u64 << drop) - 1);
        let half = 1_u64 << (drop - 1);
        let round_up = remainder > half || (remainder == half && (truncated >> drop) & 1 == 1);
        mant = truncated;
        if round_up {
            mant += 1_u64 << drop;
            if mant >> 52 != 0 {
                // Mantissa carried into the exponent (e.g. 1.111… → 10.0).
                mant = 0;
                exp += 1;
            }
        }
    }

    // A round-up carry can push the exponent past the reduced format's range
    // (or the f64 finite range) → ±infinity.
    let overflow = if exp_bits < 11 {
        exp - 1023 > (1_i32 << (exp_bits - 1)) - 1
    } else {
        exp >= 0x7ff
    };
    if overflow {
        return f64::from_bits(sign | (0x7ff_u64 << 52));
    }

    f64::from_bits(sign | ((exp as u64) << 52) | mant)
}

fn quantize_f32(value: f32, exponent_bits: u32, mantissa_bits: u32) -> f32 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }

    let exp_bits = exponent_bits.clamp(1, 8);
    let mant_bits = mantissa_bits.min(23);
    if exp_bits == 8 && mant_bits == 23 {
        return value;
    }

    let bits = value.to_bits();
    let sign = bits & (1_u32 << 31);
    let mut exp = ((bits >> 23) & 0xff) as i32;
    let mut mant = bits & ((1_u32 << 23) - 1);

    if exp == 0 {
        return f32::from_bits(sign);
    }

    let unbiased = exp - 127;
    if exp_bits < 8 {
        let bias_new = (1_i32 << (exp_bits - 1)) - 1;
        let min_unbiased = 1 - bias_new;
        let max_unbiased = bias_new;
        if unbiased > max_unbiased {
            return f32::from_bits(sign | (0xff_u32 << 23));
        }
        if unbiased < min_unbiased {
            return f32::from_bits(sign);
        }
    }

    if mant_bits < 23 {
        // Round to nearest, ties-to-even (XLA reduce_precision semantics).
        let drop = 23 - mant_bits;
        let truncated = (mant >> drop) << drop;
        let remainder = mant & ((1_u32 << drop) - 1);
        let half = 1_u32 << (drop - 1);
        let round_up = remainder > half || (remainder == half && (truncated >> drop) & 1 == 1);
        mant = truncated;
        if round_up {
            mant += 1_u32 << drop;
            if mant >> 23 != 0 {
                mant = 0;
                exp += 1;
            }
        }
    }

    let overflow = if exp_bits < 8 {
        exp - 127 > (1_i32 << (exp_bits - 1)) - 1
    } else {
        exp >= 0xff
    };
    if overflow {
        return f32::from_bits(sign | (0xff_u32 << 23));
    }

    f32::from_bits(sign | ((exp as u32) << 23) | mant)
}

fn reduce_precision_literal(
    primitive: Primitive,
    dtype: DType,
    literal: Literal,
    exponent_bits: u32,
    mantissa_bits: u32,
) -> Result<Literal, EvalError> {
    match dtype {
        DType::F64 => {
            let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "reduce_precision expected an f64 value",
            })?;
            Ok(Literal::from_f64(quantize_f64(
                value,
                exponent_bits,
                mantissa_bits,
            )))
        }
        DType::F32 => {
            let value = literal.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "reduce_precision expected an f32-compatible value",
            })? as f32;
            Ok(Literal::from_f64(f64::from(quantize_f32(
                value,
                exponent_bits,
                mantissa_bits,
            ))))
        }
        DType::BF16 => {
            let value = literal.as_bf16_f32().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "reduce_precision expected bf16 literal payload",
            })?;
            Ok(Literal::from_bf16_f32(quantize_f32(
                value,
                exponent_bits.min(8),
                mantissa_bits.min(23),
            )))
        }
        DType::F16 => {
            let value = literal.as_f16_f32().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "reduce_precision expected f16 literal payload",
            })?;
            Ok(Literal::from_f16_f32(quantize_f32(
                value,
                exponent_bits.min(8),
                mantissa_bits.min(23),
            )))
        }
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "reduce_precision supports floating-point dtypes only".to_owned(),
        }),
    }
}

/// ReducePrecision: simulate reduced floating-point exponent/mantissa precision.
///
/// Params:
/// - `exponent_bits` (default: native exponent width)
/// - `mantissa_bits` (default: native mantissa width)
pub(crate) fn eval_reduce_precision(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::ReducePrecision;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let source_dtype = inputs[0].dtype();
    let (default_exp_bits, default_mant_bits) = match source_dtype {
        DType::F64 => (11_u32, 52_u32),
        DType::F32 => (8_u32, 23_u32),
        DType::BF16 => (8_u32, 7_u32),
        DType::F16 => (5_u32, 10_u32),
        _ => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "reduce_precision supports floating-point dtypes only".to_owned(),
            });
        }
    };

    let exponent_bits = params
        .get("exponent_bits")
        .map(|raw| {
            raw.trim()
                .parse::<u32>()
                .map_err(|_| EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid exponent_bits: '{raw}'"),
                })
        })
        .transpose()?
        .unwrap_or(default_exp_bits);
    let mantissa_bits = params
        .get("mantissa_bits")
        .map(|raw| {
            raw.trim()
                .parse::<u32>()
                .map_err(|_| EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid mantissa_bits: '{raw}'"),
                })
        })
        .transpose()?
        .unwrap_or(default_mant_bits);

    match &inputs[0] {
        Value::Scalar(literal) => Ok(Value::Scalar(reduce_precision_literal(
            primitive,
            source_dtype,
            *literal,
            exponent_bits,
            mantissa_bits,
        )?)),
        Value::Tensor(tensor) => {
            let mut out = Vec::with_capacity(tensor.elements.len());
            for literal in &tensor.elements {
                out.push(reduce_precision_literal(
                    primitive,
                    source_dtype,
                    *literal,
                    exponent_bits,
                    mantissa_bits,
                )?);
            }
            Ok(Value::Tensor(TensorValue::new(
                source_dtype,
                tensor.shape.clone(),
                out,
            )?))
        }
    }
}

/// One-hot encoding: given integer indices, produces a tensor with an inserted
/// `num_classes` dimension where position `index` is `on_value` and the rest
/// are `off_value`.
pub(crate) fn eval_one_hot(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::OneHot;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let num_classes_str = params
        .get("num_classes")
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "missing required param 'num_classes'".into(),
        })?;
    let num_classes: u32 = num_classes_str
        .trim()
        .parse()
        .map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("invalid num_classes: '{num_classes_str}'"),
        })?;

    let on_value: f64 = params
        .get("on_value")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1.0);
    let off_value: f64 = params
        .get("off_value")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);

    let dtype = if let Some(raw) = params.get("dtype") {
        parse_dtype_name(primitive, "dtype", raw)?
    } else {
        DType::F64
    };

    // Collect flat index values from input
    let indices: Vec<i64> = match &inputs[0] {
        Value::Scalar(lit) => {
            vec![lit.as_i64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "one_hot requires integer indices",
            })?]
        }
        Value::Tensor(t) => {
            let mut idxs = Vec::with_capacity(t.elements.len());
            for lit in &t.elements {
                idxs.push(lit.as_i64().ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "one_hot requires integer indices",
                })?);
            }
            idxs
        }
    };

    let input_shape = match &inputs[0] {
        Value::Scalar(_) => vec![],
        Value::Tensor(t) => t.shape.dims.clone(),
    };
    let output_rank = input_shape.len() + 1;
    let axis = parse_axis_insert_param(primitive, "axis", params, output_rank, output_rank - 1)?;
    let mut out_dims = input_shape.clone();
    out_dims.insert(axis, num_classes);

    let nc = num_classes as usize;
    let total = indices
        .len()
        .checked_mul(nc)
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "one_hot output element count overflows usize".to_owned(),
        })?;
    let input_strides = checked_row_major_strides(primitive, "one_hot", &input_shape)?;
    let output_strides = checked_row_major_strides(primitive, "one_hot", &out_dims)?;

    let literal_for = |val: f64| -> Literal {
        match dtype {
            DType::I64 | DType::I32 => Literal::I64(val as i64),
            DType::U32 => Literal::U32(val as u32),
            DType::U64 => Literal::U64(val as u64),
            DType::BF16 => Literal::from_bf16_f64(val),
            DType::F16 => Literal::from_f16_f64(val),
            DType::F32 => Literal::from_f32(val as f32),
            DType::F64 => Literal::from_f64(val),
            DType::Bool => Literal::Bool(val != 0.0),
            DType::Complex64 => Literal::from_complex64(val as f32, 0.0),
            DType::Complex128 => Literal::from_complex128(val, 0.0),
        }
    };

    // One-hot is structurally a fill-with-off + scatter-on: the output is all
    // `off_value` except, for each input index `i` with `0 <= idx[i] < nc`, the
    // single position (input coords with the class axis = idx[i]) set to
    // `on_value`. This replaces the O(total * rank) per-element decode with an
    // O(total) fill + O(indices) scatter. The scatter destination is computed
    // from the input flat index `i` (decoded to input coords) mapped to the
    // output via the inserted class axis. Bit-identical to the per-element form.
    let input_rank = input_shape.len();
    let class_stride = output_strides[axis];
    let in_to_out_stride: Vec<usize> = (0..input_rank)
        .map(|j| output_strides[if j < axis { j } else { j + 1 }])
        .collect();

    let tensor = match dtype {
        DType::F64 => {
            let on = literal_for(on_value);
            let off = literal_for(off_value);
            if let Some(elements) = one_hot_repeated_patch_buffer(
                on,
                off,
                total,
                &indices,
                nc,
                input_rank,
                &input_strides,
                &in_to_out_stride,
                class_stride,
            ) {
                TensorValue::new_with_literal_buffer(DType::F64, Shape { dims: out_dims }, elements)
            } else {
                TensorValue::new_f64_values(
                    Shape { dims: out_dims },
                    one_hot_scatter(
                        on_value,
                        off_value,
                        total,
                        &indices,
                        nc,
                        input_rank,
                        &input_strides,
                        &in_to_out_stride,
                        class_stride,
                    ),
                )
            }
        }
        DType::I64 => TensorValue::new_i64_values(
            Shape { dims: out_dims },
            one_hot_scatter(
                on_value as i64,
                off_value as i64,
                total,
                &indices,
                nc,
                input_rank,
                &input_strides,
                &in_to_out_stride,
                class_stride,
            ),
        ),
        DType::Bool => TensorValue::new_bool_values(
            Shape { dims: out_dims },
            one_hot_scatter(
                on_value != 0.0,
                off_value != 0.0,
                total,
                &indices,
                nc,
                input_rank,
                &input_strides,
                &in_to_out_stride,
                class_stride,
            ),
        ),
        // f32 (JAX's default dtype) + BF16/F16 dense fill+scatter over the typed
        // backing -> dense output. The typed on/off values are the exact bits of
        // `literal_for(...)` for that dtype, so materialization is bit-identical.
        DType::F32 => TensorValue::new_f32_values(
            Shape { dims: out_dims },
            one_hot_scatter(
                on_value as f32,
                off_value as f32,
                total,
                &indices,
                nc,
                input_rank,
                &input_strides,
                &in_to_out_stride,
                class_stride,
            ),
        ),
        DType::BF16 | DType::F16 => {
            let to_bits = |v: f64| -> u16 {
                match if dtype == DType::BF16 {
                    Literal::from_bf16_f64(v)
                } else {
                    Literal::from_f16_f64(v)
                } {
                    Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                    _ => 0,
                }
            };
            TensorValue::new_half_float_values(
                dtype,
                Shape { dims: out_dims },
                one_hot_scatter(
                    to_bits(on_value),
                    to_bits(off_value),
                    total,
                    &indices,
                    nc,
                    input_rank,
                    &input_strides,
                    &in_to_out_stride,
                    class_stride,
                ),
            )
        }
        // Dtypes without dense storage (I32/U32/U64/Complex): still
        // fill+scatter, but over Literals (matching literal_for exactly).
        _ => TensorValue::new(
            dtype,
            Shape { dims: out_dims },
            one_hot_scatter(
                literal_for(on_value),
                literal_for(off_value),
                total,
                &indices,
                nc,
                input_rank,
                &input_strides,
                &in_to_out_stride,
                class_stride,
            ),
        ),
    }
    .map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

/// One-hot fill+scatter shared by `eval_one_hot`'s dense and Literal paths:
/// start from an all-`off` output and, for each input flat index `i` whose value
/// `idx[i]` is in `[0, nc)`, set the single on-position (input coords mapped into
/// the output with the class axis = `idx[i]`). Generic over the element type.
#[allow(clippy::too_many_arguments)]
fn one_hot_scatter<T: Copy>(
    on: T,
    off: T,
    total: usize,
    indices: &[i64],
    nc: usize,
    input_rank: usize,
    input_strides: &[usize],
    in_to_out_stride: &[usize],
    class_stride: usize,
) -> Vec<T> {
    let mut out = vec![off; total];
    for (i, &idx) in indices.iter().enumerate() {
        if idx < 0 || idx as usize >= nc {
            continue; // index outside [0, nc): the whole row stays `off`
        }
        let mut rem = i;
        let mut out_base = 0_usize;
        for j in 0..input_rank {
            let c = rem / input_strides[j];
            rem %= input_strides[j];
            out_base += c * in_to_out_stride[j];
        }
        out[out_base + idx as usize * class_stride] = on;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn one_hot_repeated_patch_buffer(
    on: Literal,
    off: Literal,
    total: usize,
    indices: &[i64],
    nc: usize,
    input_rank: usize,
    input_strides: &[usize],
    in_to_out_stride: &[usize],
    class_stride: usize,
) -> Option<LiteralBuffer> {
    let mut patches = Vec::with_capacity(indices.len());
    for (i, &idx) in indices.iter().enumerate() {
        if idx < 0 || idx as usize >= nc {
            continue;
        }
        let mut rem = i;
        let mut out_base = 0_usize;
        for j in 0..input_rank {
            let c = rem / input_strides[j];
            rem %= input_strides[j];
            out_base += c * in_to_out_stride[j];
        }
        patches.push((out_base + idx as usize * class_stride, on));
    }

    LiteralBuffer::from_repeated_with_patches(vec![off], total, patches)
}

/// Sort: sort elements along a specified axis (default: last axis).
pub(crate) fn eval_sort(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let descending = params
        .get("descending")
        .map(|s| s.trim() == "true")
        .unwrap_or(false);

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if rank == 0 {
                return Ok(Value::Scalar(tensor.elements[0]));
            }

            let axis = parse_axis_param(primitive, "axis", params, rank, rank - 1)?;

            sort_along_axis(primitive, tensor, axis, descending, false)
        }
    }
}

/// Argsort: return indices that would sort along a specified axis.
pub(crate) fn eval_argsort(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let descending = params
        .get("descending")
        .map(|s| s.trim() == "true")
        .unwrap_or(false);

    match &inputs[0] {
        Value::Scalar(_) => Ok(Value::scalar_i64(0)),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if rank == 0 {
                return Ok(Value::scalar_i64(0));
            }

            let axis = parse_axis_param(primitive, "axis", params, rank, rank - 1)?;

            sort_along_axis(primitive, tensor, axis, descending, true)
        }
    }
}

pub(crate) fn eval_argmin(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    eval_arg_extremum(primitive, inputs, params, false)
}

pub(crate) fn eval_argmax(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    eval_arg_extremum(primitive, inputs, params, true)
}

const TOP_K_PARTIAL_SELECT_MAX_DENOM: usize = 4;

#[inline]
fn top_k_pair_cmp(lhs: &(u64, u32), rhs: &(u64, u32)) -> Ordering {
    lhs.0.cmp(&rhs.0).then_with(|| lhs.1.cmp(&rhs.1))
}

#[inline]
fn order_top_k_pairs(pairs: &mut Vec<(u64, u32)>, scratch: &mut Vec<(u64, u32)>, k: usize) {
    if k == 0 {
        return;
    }
    if k <= pairs.len() / TOP_K_PARTIAL_SELECT_MAX_DENOM {
        let kth = k - 1;
        pairs.select_nth_unstable_by(kth, top_k_pair_cmp);
        pairs[..k].sort_unstable_by(top_k_pair_cmp);
    } else {
        if scratch.len() != pairs.len() {
            scratch.resize(pairs.len(), (0, 0));
        }
        radix_pairs_ascending(pairs, scratch);
    }
}

/// Process `n_slices` independent top-k slices — slice `s` writes its first `k`
/// `(value,index)` results to `out_vals[s*k..(s+1)*k]` / `out_idx[s*k..(s+1)*k]`
/// — by calling `per_slice(s, vals_chunk, idx_chunk, pairs, scratch)`. Each
/// slice's partial selection is cache-resident and compute-bound, and output
/// slice `s` depends only on input slice `s` (the `per_slice` reader closes over
/// a shared `&[_]`), so large workloads fan out across threads with each thread
/// owning disjoint `out_vals`/`out_idx` sub-slices plus its own pairs/scratch —
/// BIT-IDENTICAL to the serial loop for ANY partition (proven by
/// `threaded_top_k_matches_serial_reference_large_multislice`). Mirrors
/// [`for_each_contiguous_sort_slice`]; gated on the same
/// `SORT_PARALLEL_MIN_TOTAL_ELEMS` total-work threshold (`n_slices * stride`).
/// Below it (or with one core, or `k == 0`) it runs the prior single-threaded
/// loop with one shared scratch.
fn for_each_top_k_slice<V, F>(
    out_vals: &mut [V],
    out_idx: &mut [i64],
    k: usize,
    stride: usize,
    n_slices: usize,
    per_slice: F,
) where
    V: Send,
    F: Fn(usize, &mut [V], &mut [i64], &mut Vec<(u64, u32)>, &mut Vec<(u64, u32)>) + Sync,
{
    let total = n_slices.saturating_mul(stride);
    let hardware = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let threads = if k > 0 && total >= SORT_PARALLEL_MIN_TOTAL_ELEMS {
        hardware.min(total / SORT_PARALLEL_MIN_TOTAL_ELEMS).max(1)
    } else {
        1
    };

    if threads <= 1 {
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(stride);
        let mut scratch: Vec<(u64, u32)> = Vec::new();
        for (s, (vchunk, ichunk)) in out_vals
            .chunks_mut(k.max(1))
            .zip(out_idx.chunks_mut(k.max(1)))
            .enumerate()
        {
            per_slice(s, vchunk, ichunk, &mut pairs, &mut scratch);
        }
        return;
    }

    let chunk_slices = n_slices.div_ceil(threads);
    let per_slice = &per_slice;
    std::thread::scope(|scope| {
        let mut vrest: &mut [V] = out_vals;
        let mut irest: &mut [i64] = out_idx;
        let mut start_slice = 0usize;
        while start_slice < n_slices {
            let group = chunk_slices.min(n_slices - start_slice);
            let (vblock, vtail) = vrest.split_at_mut(group * k);
            let (iblock, itail) = irest.split_at_mut(group * k);
            vrest = vtail;
            irest = itail;
            let base_slice = start_slice;
            scope.spawn(move || {
                let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(stride);
                let mut scratch: Vec<(u64, u32)> = Vec::new();
                for (j, (vchunk, ichunk)) in
                    vblock.chunks_mut(k).zip(iblock.chunks_mut(k)).enumerate()
                {
                    per_slice(base_slice + j, vchunk, ichunk, &mut pairs, &mut scratch);
                }
            });
            start_slice += group;
        }
    });
}

/// Dense i64/f64 TopK fast path over the last (contiguous) axis: for each slice,
/// order `(complement key, in-slice index)` pairs and take the first `k`. The
/// complement (`!total_order_key`) turns ascending order into
/// descending-by-value, and the in-slice index keeps equal values in ascending
/// tie order — exactly the generic comparator
/// `compare_sort_keys(b, a).then(a_idx.cmp(b_idx))`. Small `k` uses exact
/// partial selection and sorts only the selected prefix; large `k` keeps the
/// stable LSD radix sort. Returns `None` (generic path) for non-dense /
/// non-i64/f64 storage or short axes.
fn eval_top_k_dense(
    tensor: &TensorValue,
    k: usize,
    stride: usize,
    n_slices: usize,
    output_dims: &[u32],
) -> Result<Option<Vec<Value>>, EvalError> {
    if stride < RADIX_SORT_MIN_AXIS {
        return Ok(None);
    }
    if let Some(values) = tensor.elements.as_i64_slice() {
        let mut out_vals = vec![0_i64; n_slices * k];
        let mut out_idx = vec![0_i64; n_slices * k];
        for_each_top_k_slice(
            &mut out_vals,
            &mut out_idx,
            k,
            stride,
            n_slices,
            |slice, vchunk, ichunk, pairs, scratch| {
                let base = slice * stride;
                pairs.clear();
                for (i, &v) in values[base..base + stride].iter().enumerate() {
                    pairs.push((!((v as u64) ^ (1_u64 << 63)), i as u32));
                }
                order_top_k_pairs(pairs, scratch, k);
                for (dst, &(_, orig)) in pairs.iter().take(k).enumerate() {
                    vchunk[dst] = values[base + orig as usize];
                    ichunk[dst] = i64::from(orig);
                }
            },
        );
        let values_t = TensorValue::new_i64_values(
            Shape {
                dims: output_dims.to_vec(),
            },
            out_vals,
        )
        .map_err(EvalError::InvalidTensor)?;
        let indices_t = TensorValue::new_i64_values(
            Shape {
                dims: output_dims.to_vec(),
            },
            out_idx,
        )
        .map_err(EvalError::InvalidTensor)?;
        return Ok(Some(vec![
            Value::Tensor(values_t),
            Value::Tensor(indices_t),
        ]));
    }

    if let Some(values) = tensor.elements.as_f64_slice() {
        let mut out_vals = vec![0.0_f64; n_slices * k];
        let mut out_idx = vec![0_i64; n_slices * k];
        for_each_top_k_slice(
            &mut out_vals,
            &mut out_idx,
            k,
            stride,
            n_slices,
            |slice, vchunk, ichunk, pairs, scratch| {
                let base = slice * stride;
                pairs.clear();
                for (i, &v) in values[base..base + stride].iter().enumerate() {
                    pairs.push((!f64_total_order_key(v), i as u32));
                }
                order_top_k_pairs(pairs, scratch, k);
                for (dst, &(_, orig)) in pairs.iter().take(k).enumerate() {
                    vchunk[dst] = values[base + orig as usize];
                    ichunk[dst] = i64::from(orig);
                }
            },
        );
        let values_t = TensorValue::new_f64_values(
            Shape {
                dims: output_dims.to_vec(),
            },
            out_vals,
        )
        .map_err(EvalError::InvalidTensor)?;
        let indices_t = TensorValue::new_i64_values(
            Shape {
                dims: output_dims.to_vec(),
            },
            out_idx,
        )
        .map_err(EvalError::InvalidTensor)?;
        return Ok(Some(vec![
            Value::Tensor(values_t),
            Value::Tensor(indices_t),
        ]));
    }

    // Literal-backed numeric dtypes with no dense storage (F32/F16/BF16, U32/U64,
    // I32): same complement-key radix as i64/f64, keyed per dtype family to match
    // the generic `compare_sort_keys` order — Float via f64_total_order_key(as_f64),
    // Unsigned via as_u64, Signed via (as_i64 as u64)^(1<<63). Mirrors the sort
    // path's sort_along_axis_literal_radix. F64/I64 are handled by the dense
    // branches above; Bool/Complex fall through to the generic path.
    enum TopKKeyKind {
        Float,
        Unsigned,
        Signed,
    }
    let kind = match tensor.dtype {
        DType::F32 | DType::F16 | DType::BF16 => TopKKeyKind::Float,
        DType::U32 | DType::U64 => TopKKeyKind::Unsigned,
        DType::I32 => TopKKeyKind::Signed,
        _ => return Ok(None),
    };
    let elems = tensor.elements.as_slice();
    let mut comp = Vec::with_capacity(elems.len());
    for lit in elems.iter() {
        let key = match kind {
            TopKKeyKind::Float => match lit.as_f64() {
                Some(v) => f64_total_order_key(v),
                None => return Ok(None),
            },
            TopKKeyKind::Unsigned => match lit.as_u64() {
                Some(v) => v,
                None => return Ok(None),
            },
            TopKKeyKind::Signed => match lit.as_i64() {
                Some(v) => (v as u64) ^ (1_u64 << 63),
                None => return Ok(None),
            },
        };
        comp.push(!key);
    }

    let mut out_lit = vec![elems[0]; n_slices * k];
    let mut out_idx = vec![0_i64; n_slices * k];
    for_each_top_k_slice(
        &mut out_lit,
        &mut out_idx,
        k,
        stride,
        n_slices,
        |slice, vchunk, ichunk, pairs, scratch| {
            let base = slice * stride;
            pairs.clear();
            for (i, &key) in comp[base..base + stride].iter().enumerate() {
                pairs.push((key, i as u32));
            }
            order_top_k_pairs(pairs, scratch, k);
            for (dst, &(_, orig)) in pairs.iter().take(k).enumerate() {
                vchunk[dst] = elems[base + orig as usize];
                ichunk[dst] = i64::from(orig);
            }
        },
    );
    let values_t = TensorValue::new(
        tensor.dtype,
        Shape {
            dims: output_dims.to_vec(),
        },
        out_lit,
    )
    .map_err(EvalError::InvalidTensor)?;
    let indices_t = TensorValue::new_i64_values(
        Shape {
            dims: output_dims.to_vec(),
        },
        out_idx,
    )
    .map_err(EvalError::InvalidTensor)?;
    Ok(Some(vec![
        Value::Tensor(values_t),
        Value::Tensor(indices_t),
    ]))
}

pub(crate) fn eval_top_k(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, EvalError> {
    let primitive = Primitive::TopK;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let k: usize =
        params
            .get("k")
            .and_then(|s| s.trim().parse().ok())
            .ok_or(EvalError::Unsupported {
                primitive,
                detail: "top_k requires 'k' param".to_owned(),
            })?;

    match &inputs[0] {
        Value::Scalar(_) => Err(EvalError::Unsupported {
            primitive,
            detail: "top_k operand must have >= 1 dimension".to_owned(),
        }),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if rank == 0 {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "top_k operand must have >= 1 dimension".to_owned(),
                });
            }

            let axis_size = tensor.shape.dims[rank - 1] as usize;
            if k > axis_size {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("k={k} exceeds axis size {axis_size}"),
                });
            }

            let mut output_dims = tensor.shape.dims.clone();
            output_dims[rank - 1] = k as u32;

            let stride = axis_size;
            let n_slices = tensor.elements.len() / stride;

            // Dense i64/f64 radix fast path (O(n) vs the comparison sort's
            // O(n log n)); returns None for non-dense / short axes -> generic.
            if let Some(result) = eval_top_k_dense(tensor, k, stride, n_slices, &output_dims)? {
                return Ok(result);
            }

            let mut values_elements = Vec::with_capacity(n_slices * k);
            let mut indices_elements = Vec::with_capacity(n_slices * k);

            for slice_idx in 0..n_slices {
                let base = slice_idx * stride;
                let mut indexed: Vec<(usize, Literal, SortKey)> = Vec::with_capacity(stride);
                for (orig_idx, literal) in tensor.elements[base..base + stride]
                    .iter()
                    .copied()
                    .enumerate()
                {
                    let key = sort_key(literal)
                        .map_err(|detail| EvalError::Unsupported { primitive, detail })?;
                    indexed.push((orig_idx, literal, key));
                }

                indexed.sort_by(|a, b| compare_sort_keys(b.2, a.2).then_with(|| a.0.cmp(&b.0)));

                for (orig_idx, lit, _) in indexed.iter().take(k) {
                    values_elements.push(*lit);
                    indices_elements.push(Literal::I64(*orig_idx as i64));
                }
            }

            let values = TensorValue::new(
                tensor.dtype,
                Shape {
                    dims: output_dims.clone(),
                },
                values_elements,
            )
            .map_err(|e| EvalError::Unsupported {
                primitive,
                detail: e.to_string(),
            })?;
            let indices =
                TensorValue::new(DType::I64, Shape { dims: output_dims }, indices_elements)
                    .map_err(|e| EvalError::Unsupported {
                        primitive,
                        detail: e.to_string(),
                    })?;

            Ok(vec![Value::Tensor(values), Value::Tensor(indices)])
        }
    }
}

fn eval_arg_extremum(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
    find_max: bool,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(_) => Ok(Value::scalar_i64(0)),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if rank == 0 {
                return Ok(Value::scalar_i64(0));
            }

            let axis = parse_axis_param(primitive, "axis", params, rank, rank - 1)?;
            extremum_along_axis(primitive, tensor, axis, find_max)
        }
    }
}

fn extremum_along_axis(
    primitive: Primitive,
    tensor: &TensorValue,
    axis: usize,
    find_max: bool,
) -> Result<Value, EvalError> {
    let rank = tensor.shape.rank();
    let axis_dim = tensor.shape.dims[axis] as usize;

    if axis_dim == 0 || tensor.elements.is_empty() {
        let mut result_dims: Vec<u32> = tensor.shape.dims.clone();
        result_dims.remove(axis);
        let result_shape = Shape { dims: result_dims };
        return Ok(Value::Tensor(
            TensorValue::new(DType::I64, result_shape, vec![]).map_err(EvalError::InvalidTensor)?,
        ));
    }

    let strides = checked_row_major_strides(primitive, "argmin/argmax", &tensor.shape.dims)?;
    let axis_stride = strides[axis];
    let total = tensor.elements.len();
    if !total.is_multiple_of(axis_dim) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "argmin/argmax axis dimension {axis_dim} does not divide {total} input elements"
            ),
        });
    }
    let outer_count = total / axis_dim;

    let mut result_dims: Vec<u32> = tensor.shape.dims.clone();
    result_dims.remove(axis);
    let result_shape = Shape { dims: result_dims };

    let mut result_elements = Vec::with_capacity(outer_count);

    // Row-major base offset (in-slice index 0) for output position `outer`,
    // shared by the dense fast paths below.
    let base_of = |outer: usize| -> Result<usize, EvalError> {
        let mut idx = outer;
        let mut flat = 0_usize;
        for ax in (0..rank).rev() {
            if ax == axis {
                continue;
            }
            let dim = tensor.shape.dims[ax] as usize;
            if dim == 0 {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "argmin/argmax encountered zero non-axis dimension".to_owned(),
                });
            }
            let offset =
                (idx % dim)
                    .checked_mul(strides[ax])
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "argmin/argmax flat offset multiplication overflowed".to_owned(),
                    })?;
            flat = flat
                .checked_add(offset)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "argmin/argmax flat offset addition overflowed".to_owned(),
                })?;
            idx /= dim;
        }
        Ok(flat)
    };

    // Dense fast paths: scan the contiguous typed slice without the per-element
    // Literal machinery. F64 uses the JAX float reducer (`arg_extreme_float`:
    // IEEE compare + sign-agnostic first-NaN); I64 uses strict cmp with
    // first-occurrence tie-break (no NaN for integers).
    if let Some(values) = tensor.elements.as_f64_slice() {
        for outer in 0..outer_count {
            let base = base_of(outer)?;
            let best_idx = if axis_stride == 1 {
                let end = base
                    .checked_add(axis_dim)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "argmin/argmax contiguous axis slice overflowed".to_owned(),
                    })?;
                let row = values.get(base..end).ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "argmin/argmax contiguous slice {base}..{end} out of bounds for {total} elements"
                    ),
                })?;
                arg_extreme_f64_contiguous_simd(row, find_max)
            } else {
                arg_extreme_float(axis_dim, find_max, |i| values[base + i * axis_stride])
            };
            result_elements.push(Literal::I64(best_idx as i64));
        }
    } else if let Some(values) = tensor.elements.as_i64_slice() {
        for outer in 0..outer_count {
            let base = base_of(outer)?;
            let mut best_idx = 0_usize;
            let mut best = values[base];
            for i in 1..axis_dim {
                let v = values[base + i * axis_stride];
                let better = if find_max { v > best } else { v < best };
                if better {
                    best_idx = i;
                    best = v;
                }
            }
            result_elements.push(Literal::I64(best_idx as i64));
        }
    } else if let Some(values) = tensor.elements.as_f32_slice() {
        // Dense F32 (JAX's default float, the decode-time argmax-over-logits hot
        // path): scan the contiguous f32 slice directly, widening each tap to f64
        // exactly. Identical to the generic float branch below, which reconstructs
        // a `Literal` via `tensor.elements.get(..)` per element then `as_f64()`
        // (= `f64::from(f32)`); same `arg_extreme_float` reducer -> identical
        // -NaN/±0.0 tie behavior.
        for outer in 0..outer_count {
            let base = base_of(outer)?;
            let best_idx = if axis_stride == 1 {
                let end = base
                    .checked_add(axis_dim)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "argmin/argmax contiguous axis slice overflowed".to_owned(),
                    })?;
                let row = values.get(base..end).ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "argmin/argmax contiguous slice {base}..{end} out of bounds for {total} elements"
                    ),
                })?;
                arg_extreme_f32_contiguous_simd(row, find_max)
            } else {
                arg_extreme_float(axis_dim, find_max, |i| {
                    f64::from(values[base + i * axis_stride])
                })
            };
            result_elements.push(Literal::I64(best_idx as i64));
        }
    } else if matches!(
        tensor.dtype,
        DType::BF16 | DType::F16 | DType::F32 | DType::F64
    ) {
        // Real floats not in dense f64 storage (F16/F32/BF16, or strided/non-dense
        // F64): gather each slice's strided values as f64 — widening BF16/F16/F32
        // is exact and order-preserving, NaN stays NaN — then apply the same JAX
        // float reducer as the dense path so -NaN selection and ±0.0 ties match.
        let mut slice_buf: Vec<f64> = Vec::with_capacity(axis_dim);
        for outer in 0..outer_count {
            let base = base_of(outer)?;
            slice_buf.clear();
            for i in 0..axis_dim {
                let flat_idx = i
                    .checked_mul(axis_stride)
                    .and_then(|offset| base.checked_add(offset))
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "argmin/argmax axis offset overflowed".to_owned(),
                    })?;
                let literal = *tensor.elements.get(flat_idx).ok_or_else(|| {
                    EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "argmin/argmax flat index {flat_idx} out of bounds for {total} elements"
                        ),
                    }
                })?;
                let v = literal.as_f64().ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!("argmin/argmax expected a float literal, got {literal:?}"),
                })?;
                slice_buf.push(v);
            }
            let best_idx = arg_extreme_float(axis_dim, find_max, |i| slice_buf[i]);
            result_elements.push(Literal::I64(best_idx as i64));
        }
    } else {
        for outer in 0..outer_count {
            let base = {
                let mut idx = outer;
                let mut flat = 0_usize;
                for ax in (0..rank).rev() {
                    if ax == axis {
                        continue;
                    }
                    let dim = tensor.shape.dims[ax] as usize;
                    if dim == 0 {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: "argmin/argmax encountered zero non-axis dimension".to_owned(),
                        });
                    }
                    let offset = (idx % dim).checked_mul(strides[ax]).ok_or_else(|| {
                        EvalError::Unsupported {
                            primitive,
                            detail: "argmin/argmax flat offset multiplication overflowed"
                                .to_owned(),
                        }
                    })?;
                    flat = flat
                        .checked_add(offset)
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "argmin/argmax flat offset addition overflowed".to_owned(),
                        })?;
                    idx /= dim;
                }
                flat
            };

            let mut best_idx = 0_usize;
            let first_flat = base;
            let first_literal = *tensor.elements.get(first_flat).ok_or_else(|| {
                EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "argmin/argmax flat index {first_flat} out of bounds for {total} elements"
                    ),
                }
            })?;
            let mut best_key = sort_key(first_literal)
                .map_err(|detail| EvalError::Unsupported { primitive, detail })?;

            for i in 1..axis_dim {
                let flat_idx = i
                    .checked_mul(axis_stride)
                    .and_then(|offset| base.checked_add(offset))
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "argmin/argmax axis offset overflowed".to_owned(),
                    })?;
                let literal = *tensor.elements.get(flat_idx).ok_or_else(|| {
                    EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "argmin/argmax flat index {flat_idx} out of bounds for {total} elements"
                        ),
                    }
                })?;
                let key = sort_key(literal)
                    .map_err(|detail| EvalError::Unsupported { primitive, detail })?;

                let is_better = if find_max {
                    compare_sort_keys(key, best_key) == std::cmp::Ordering::Greater
                } else {
                    compare_sort_keys(key, best_key) == std::cmp::Ordering::Less
                };

                if is_better {
                    best_idx = i;
                    best_key = key;
                }
            }

            result_elements.push(Literal::I64(best_idx as i64));
        }
    }

    if result_shape.dims.is_empty() {
        Ok(Value::Scalar(result_elements[0]))
    } else {
        Ok(Value::Tensor(
            TensorValue::new(DType::I64, result_shape, result_elements)
                .map_err(EvalError::InvalidTensor)?,
        ))
    }
}

/// Minimum axis length at which the dense i64 radix path is used (the per-slice
/// 8 byte passes + 256-bucket counting overhead is only amortized over a long
/// enough slice; below this the generic comparison path is fine).
const RADIX_SORT_MIN_AXIS: usize = 256;

/// Dense i64 ascending sort/argsort along `axis`, bypassing the generic
/// `SortKey`/`Literal` machinery entirely: gather the slice's i64 values
/// straight from the contiguous `as_i64_slice()` backing, stably order them with
/// a sign-flipped LSD radix (O(n) vs the comparison path's O(n log n)), and
/// write a dense i64 output. Returns `None` (caller uses the generic path) unless
/// the tensor is I64 dense storage, ascending, and every slice is long enough.
///
/// Bit-for-bit identical to the generic ascending path: for I64 the comparator
/// is `a.cmp(&b)`, which the sign-flipped byte order reproduces exactly; LSD
/// radix is stable and the per-slice gather visits original indices ascending,
/// so equal keys keep their original order (matching the stable `sort_by`). Sort
/// emits the reordered values; argsort emits the in-slice original indices —
/// the same fields the generic path writes.
fn sort_along_axis_dense_i64(
    tensor: &TensorValue,
    axis: usize,
    descending: bool,
    return_indices: bool,
) -> Result<Option<Value>, EvalError> {
    let Some(values) = tensor.elements.as_i64_slice() else {
        return Ok(None);
    };
    // Descending == ascending radix of the complement key (`key ^ u64::MAX`);
    // stable radix keeps equal keys in ascending index order, matching the
    // generic stable descending comparator.
    let key_mask: u64 = if descending { u64::MAX } else { 0 };
    let primitive = if return_indices {
        Primitive::Argsort
    } else {
        Primitive::Sort
    };
    let rank = tensor.shape.rank();
    let axis_dim = tensor.shape.dims[axis] as usize;
    if axis_dim < RADIX_SORT_MIN_AXIS {
        return Ok(None);
    }

    let strides = checked_row_major_strides(primitive, "sort", &tensor.shape.dims)?;
    let axis_stride = strides[axis];
    let total = tensor.elements.len();
    if !total.is_multiple_of(axis_dim) {
        return Ok(None);
    }
    let outer_count = total / axis_dim;

    let mut out = vec![0_i64; total];
    if axis_stride == 1 {
        // Contiguous last axis: slices are disjoint blocks -> fan out across threads.
        // A single large slice instead parallelizes its radix intra-slice.
        let parallel = use_parallel_radix(outer_count, axis_dim);
        for_each_contiguous_sort_slice(
            &mut out,
            axis_dim,
            outer_count,
            |s, out_slice, pairs, scratch| {
                let in_base = s * axis_dim;
                pairs.clear();
                for i in 0..axis_dim {
                    // Sign-flip so byte-wise unsigned order == signed i64 order.
                    let v = values[in_base + i];
                    pairs.push((((v as u64) ^ (1_u64 << 63)) ^ key_mask, i as u32));
                }
                radix_pairs_ascending_maybe_parallel(pairs, scratch, parallel);
                for (out_pos, &(_, orig)) in pairs.iter().enumerate() {
                    out_slice[out_pos] = if return_indices {
                        i64::from(orig)
                    } else {
                        values[in_base + orig as usize]
                    };
                }
            },
        );
    } else {
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(axis_dim);
        let mut scratch: Vec<(u64, u32)> = vec![(0, 0); axis_dim];
        for_each_sort_slice(
            rank,
            axis,
            &tensor.shape.dims,
            &strides,
            outer_count,
            |base| {
                pairs.clear();
                for i in 0..axis_dim {
                    // Sign-flip so byte-wise unsigned order == signed i64 order.
                    let v = values[base + i * axis_stride];
                    pairs.push((((v as u64) ^ (1_u64 << 63)) ^ key_mask, i as u32));
                }
                radix_pairs_ascending(&mut pairs, &mut scratch);
                for (out_pos, &(_, orig)) in pairs.iter().enumerate() {
                    let dst = base + out_pos * axis_stride;
                    out[dst] = if return_indices {
                        i64::from(orig)
                    } else {
                        values[base + orig as usize * axis_stride]
                    };
                }
            },
        );
    }

    let out_value =
        TensorValue::new_i64_values(tensor.shape.clone(), out).map_err(EvalError::InvalidTensor)?;
    Ok(Some(Value::Tensor(out_value)))
}

/// Map an f64 to a u64 whose ascending unsigned order equals `f64::total_cmp`
/// order (−NaN < −inf < … < −0 < +0 < … < +inf < +NaN). Reproduces std's
/// `total_cmp` transform (flip all bits but the sign for negatives) then flips
/// the sign bit so the result radix-sorts as plain unsigned. This matches the
/// generic sort path's `SortKey::Float` comparator (`lhs.total_cmp(&rhs)`).
#[inline]
fn f64_total_order_key(value: f64) -> u64 {
    let bits = value.to_bits() as i64;
    let transformed = bits ^ ((((bits >> 63) as u64) >> 1) as i64);
    (transformed as u64) ^ (1_u64 << 63)
}

/// Sort/argsort radix key matching JAX/numpy `sort` NaN handling, which differs
/// from `total_cmp` (and from `top_k`): every NaN — EITHER sign — ranks as the
/// single maximum, so ascending sort sends all NaN to the end (stable, by
/// original index since equal keys keep input order) and the descending
/// complement key (`!key`) sends them to the front, exactly as
/// `jnp.sort(descending=...)` does (verified against JAX 0.10.1). Non-NaN values
/// keep `f64_total_order_key`'s order, which already matches JAX for finite/inf
/// and the −0.0 < +0.0 tie. `u64::MAX` is collision-free: the only f64 bit
/// patterns whose key reaches `u64::MAX` are themselves NaN. Used ONLY by
/// sort/argsort; `top_k` keeps `f64_total_order_key` (it treats +NaN as max but
/// −NaN as min, per `lax.top_k`).
#[inline]
fn f64_sort_order_key(value: f64) -> u64 {
    if value.is_nan() {
        u64::MAX
    } else {
        f64_total_order_key(value)
    }
}

/// Dense F64 ascending sort/argsort along `axis`, mirroring
/// [`sort_along_axis_dense_i64`] but keyed by [`f64_sort_order_key`]. Sort emits
/// a dense f64 output; argsort emits dense i64 in-slice indices. Bit-for-bit
/// identical to the generic ascending path (whose F64 comparator is
/// `compare_sort_keys_nan_last`). Returns `None` (generic path) for non-F64-dense
/// or short axes.
fn sort_along_axis_dense_f64(
    tensor: &TensorValue,
    axis: usize,
    descending: bool,
    return_indices: bool,
) -> Result<Option<Value>, EvalError> {
    let Some(values) = tensor.elements.as_f64_slice() else {
        return Ok(None);
    };
    // Descending == ascending radix of the complement total-order key.
    let key_mask: u64 = if descending { u64::MAX } else { 0 };
    let primitive = if return_indices {
        Primitive::Argsort
    } else {
        Primitive::Sort
    };
    let rank = tensor.shape.rank();
    let axis_dim = tensor.shape.dims[axis] as usize;
    if axis_dim < RADIX_SORT_MIN_AXIS {
        return Ok(None);
    }

    let strides = checked_row_major_strides(primitive, "sort", &tensor.shape.dims)?;
    let axis_stride = strides[axis];
    let total = tensor.elements.len();
    if !total.is_multiple_of(axis_dim) {
        return Ok(None);
    }
    let outer_count = total / axis_dim;

    // Contiguous last axis (axis_stride == 1) fans its disjoint slices across
    // threads; argsort writes i64 indices, sort writes f64 values — each to its
    // own output buffer so only the needed one is allocated/filled.
    let parallel = use_parallel_radix(outer_count, axis_dim);
    let out_value = if axis_stride == 1 {
        if return_indices {
            let mut out_idx = vec![0_i64; total];
            for_each_contiguous_sort_slice(
                &mut out_idx,
                axis_dim,
                outer_count,
                |s, out_slice, pairs, scratch| {
                    let in_base = s * axis_dim;
                    pairs.clear();
                    for i in 0..axis_dim {
                        pairs.push((f64_sort_order_key(values[in_base + i]) ^ key_mask, i as u32));
                    }
                    radix_pairs_ascending_maybe_parallel(pairs, scratch, parallel);
                    for (out_pos, &(_, orig)) in pairs.iter().enumerate() {
                        out_slice[out_pos] = i64::from(orig);
                    }
                },
            );
            TensorValue::new_i64_values(tensor.shape.clone(), out_idx)
        } else {
            let mut out_val = vec![0.0_f64; total];
            for_each_contiguous_sort_slice(
                &mut out_val,
                axis_dim,
                outer_count,
                |s, out_slice, pairs, scratch| {
                    let in_base = s * axis_dim;
                    pairs.clear();
                    for i in 0..axis_dim {
                        pairs.push((f64_sort_order_key(values[in_base + i]) ^ key_mask, i as u32));
                    }
                    radix_pairs_ascending_maybe_parallel(pairs, scratch, parallel);
                    for (out_pos, &(_, orig)) in pairs.iter().enumerate() {
                        out_slice[out_pos] = values[in_base + orig as usize];
                    }
                },
            );
            TensorValue::new_f64_values(tensor.shape.clone(), out_val)
        }
    } else {
        let mut out_idx = vec![0_i64; total];
        let mut out_val = vec![0.0_f64; total];
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(axis_dim);
        let mut scratch: Vec<(u64, u32)> = vec![(0, 0); axis_dim];

        for_each_sort_slice(
            rank,
            axis,
            &tensor.shape.dims,
            &strides,
            outer_count,
            |base| {
                pairs.clear();
                for i in 0..axis_dim {
                    pairs.push((
                        f64_sort_order_key(values[base + i * axis_stride]) ^ key_mask,
                        i as u32,
                    ));
                }
                radix_pairs_ascending(&mut pairs, &mut scratch);
                for (out_pos, &(_, orig)) in pairs.iter().enumerate() {
                    let dst = base + out_pos * axis_stride;
                    if return_indices {
                        out_idx[dst] = i64::from(orig);
                    } else {
                        out_val[dst] = values[base + orig as usize * axis_stride];
                    }
                }
            },
        );

        if return_indices {
            TensorValue::new_i64_values(tensor.shape.clone(), out_idx)
        } else {
            TensorValue::new_f64_values(tensor.shape.clone(), out_val)
        }
    }
    .map_err(EvalError::InvalidTensor)?;
    Ok(Some(Value::Tensor(out_value)))
}

/// Ascending radix sort/argsort along `axis` for Literal-backed numeric tensors
/// that have no dense storage variant (F32/F16/BF16, U32/U64, I32). The generic
/// path keys these via `compare_sort_keys_nan_last` — `SortKey::Float(as_f64)`
/// with NaN-last (else `total_cmp`) for floats, unsigned `cmp` for U32/U64,
/// signed `cmp` for integers. Each maps to an ascending `u64` radix key that
/// reproduces that exact order:
/// `f64_sort_order_key(as_f64)` (float), `as_u64` (unsigned), and
/// `(as_i64 as u64) ^ (1<<63)` (signed). Both the generic `sort_by` and LSD radix
/// are stable, so the output permutation is bit-for-bit identical — in O(n) radix
/// passes instead of O(n log n) comparisons. Sort emits the reordered original
/// literals (exact bits); argsort emits dense i64 in-slice indices. F64/I64 are
/// excluded (their dedicated dense paths run first; Literal-backed F64/I64 keep
/// the generic path so their radix-vs-generic tests stay meaningful). Returns
/// `None` (generic path) for Bool/Complex, short axes, or non-orderable elements.
fn sort_along_axis_literal_radix(
    tensor: &TensorValue,
    axis: usize,
    descending: bool,
    return_indices: bool,
) -> Result<Option<Value>, EvalError> {
    // Descending == ascending radix of the complement key.
    let key_mask: u64 = if descending { u64::MAX } else { 0 };
    enum KeyKind {
        Float,
        Unsigned,
        Signed,
    }
    let kind = match tensor.dtype {
        DType::F32 | DType::F16 | DType::BF16 => KeyKind::Float,
        DType::U32 | DType::U64 => KeyKind::Unsigned,
        DType::I32 => KeyKind::Signed,
        _ => return Ok(None),
    };
    let axis_dim = tensor.shape.dims[axis] as usize;
    if axis_dim < RADIX_SORT_MIN_AXIS {
        return Ok(None);
    }
    let primitive = if return_indices {
        Primitive::Argsort
    } else {
        Primitive::Sort
    };
    let rank = tensor.shape.rank();
    let strides = checked_row_major_strides(primitive, "sort", &tensor.shape.dims)?;
    let axis_stride = strides[axis];
    let total = tensor.elements.len();
    if !total.is_multiple_of(axis_dim) {
        return Ok(None);
    }
    let outer_count = total / axis_dim;

    // Materialize the literals once and pre-extract ascending u64 keys; bail to
    // the generic path if any element is not orderable in the expected family
    // (matches the generic `sort_key` fallibility without diverging behavior).
    let elems = tensor.elements.as_slice();
    let mut keys = Vec::with_capacity(total);
    for lit in elems.iter() {
        let key = match kind {
            KeyKind::Float => match lit.as_f64() {
                Some(v) => f64_sort_order_key(v),
                None => return Ok(None),
            },
            KeyKind::Unsigned => match lit.as_u64() {
                Some(v) => v,
                None => return Ok(None),
            },
            KeyKind::Signed => match lit.as_i64() {
                Some(v) => (v as u64) ^ (1_u64 << 63),
                None => return Ok(None),
            },
        };
        keys.push(key);
    }

    let is_u32 = tensor.dtype == DType::U32;
    // u32 keys live in the low 32 bits, so the top-byte MSD partition is degenerate
    // (one bucket) — parallel radix only helps the full 8-byte (Float/I32) keys.
    let parallel = !is_u32 && use_parallel_radix(outer_count, axis_dim);
    let order_pairs = |pairs: &mut Vec<(u64, u32)>, scratch: &mut Vec<(u64, u32)>| {
        if is_u32 {
            radix_pairs_ascending_u32(pairs, scratch);
        } else {
            radix_pairs_ascending_maybe_parallel(pairs, scratch, parallel);
        }
    };

    // Contiguous last axis (axis_stride == 1) fans its disjoint slices across
    // threads; argsort writes i64 indices, sort writes the reordered literals —
    // each to its own output buffer.
    let out_value = if axis_stride == 1 {
        if return_indices {
            let mut out_idx = vec![0_i64; total];
            for_each_contiguous_sort_slice(
                &mut out_idx,
                axis_dim,
                outer_count,
                |s, out_slice, pairs, scratch| {
                    let in_base = s * axis_dim;
                    pairs.clear();
                    for i in 0..axis_dim {
                        pairs.push((keys[in_base + i] ^ key_mask, i as u32));
                    }
                    order_pairs(pairs, scratch);
                    for (out_pos, &(_, orig)) in pairs.iter().enumerate() {
                        out_slice[out_pos] = i64::from(orig);
                    }
                },
            );
            TensorValue::new_i64_values(tensor.shape.clone(), out_idx)
                .map_err(EvalError::InvalidTensor)?
        } else {
            let mut out_lit = elems.to_vec();
            for_each_contiguous_sort_slice(
                &mut out_lit,
                axis_dim,
                outer_count,
                |s, out_slice, pairs, scratch| {
                    let in_base = s * axis_dim;
                    pairs.clear();
                    for i in 0..axis_dim {
                        pairs.push((keys[in_base + i] ^ key_mask, i as u32));
                    }
                    order_pairs(pairs, scratch);
                    for (out_pos, &(_, orig)) in pairs.iter().enumerate() {
                        out_slice[out_pos] = elems[in_base + orig as usize];
                    }
                },
            );
            TensorValue::new(tensor.dtype, tensor.shape.clone(), out_lit)
                .map_err(EvalError::InvalidTensor)?
        }
    } else {
        let mut out_idx = vec![0_i64; total];
        let mut out_lit = elems.to_vec();
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(axis_dim);
        let mut scratch: Vec<(u64, u32)> = vec![(0, 0); axis_dim];

        for_each_sort_slice(
            rank,
            axis,
            &tensor.shape.dims,
            &strides,
            outer_count,
            |base| {
                pairs.clear();
                for i in 0..axis_dim {
                    pairs.push((keys[base + i * axis_stride] ^ key_mask, i as u32));
                }
                order_pairs(&mut pairs, &mut scratch);
                for (out_pos, &(_, orig)) in pairs.iter().enumerate() {
                    let dst = base + out_pos * axis_stride;
                    if return_indices {
                        out_idx[dst] = i64::from(orig);
                    } else {
                        out_lit[dst] = elems[base + orig as usize * axis_stride];
                    }
                }
            },
        );

        if return_indices {
            TensorValue::new_i64_values(tensor.shape.clone(), out_idx)
                .map_err(EvalError::InvalidTensor)?
        } else {
            TensorValue::new(tensor.dtype, tensor.shape.clone(), out_lit)
                .map_err(EvalError::InvalidTensor)?
        }
    };
    Ok(Some(Value::Tensor(out_value)))
}

/// Invoke `f(base)` for each sort slice's row-major base offset (the flat offset
/// of in-slice index 0, over all axes except `axis`), in ascending `outer`.
#[inline]
fn for_each_sort_slice(
    rank: usize,
    axis: usize,
    dims: &[u32],
    strides: &[usize],
    outer_count: usize,
    mut f: impl FnMut(usize),
) {
    for outer in 0..outer_count {
        let mut idx = outer;
        let mut base = 0_usize;
        for ax in (0..rank).rev() {
            if ax == axis {
                continue;
            }
            let dim = dims[ax] as usize;
            base += (idx % dim) * strides[ax];
            idx /= dim;
        }
        f(base);
    }
}

/// Minimum total element count before a multi-slice radix sort fans its
/// independent slices across threads. Each slice's radix is cache-resident (a few
/// KB of `(key,index)` pairs), so per-slice work is compute-bound and threading
/// scales near-linearly — but spawning for a small sort loses to thread overhead.
/// Matches the reduction/RNG threading regime (large workloads only); below it the
/// single-threaded loop runs unchanged.
const SORT_PARALLEL_MIN_TOTAL_ELEMS: usize = 1 << 18;

/// Process `outer_count` CONTIGUOUS sort slices — slice `s` occupies
/// `[s*axis_dim, (s+1)*axis_dim)` of `out` — by calling
/// `per_slice(s, out_slice, pairs, scratch)`. Used only when the sort axis is the
/// last/contiguous axis (`axis_stride == 1`), so the slices are disjoint
/// contiguous blocks in input AND output order.
///
/// Large workloads fan out across threads: the slices are partitioned into
/// contiguous groups, each thread owns a disjoint `out` sub-slice plus its own
/// `pairs`/`scratch` reused across its slices. Output slice `s` depends only on
/// input slice `s` (the `per_slice` reader closes over a shared `&[_]`), so the
/// result is BIT-IDENTICAL to the serial loop for ANY partition — proven by the
/// `threaded_*_sort_matches_serial` tests. Below the threshold (or with one core)
/// it runs a single-threaded loop with one shared scratch, identical to the prior
/// `for_each_sort_slice` path.
fn for_each_contiguous_sort_slice<Out, F>(
    out: &mut [Out],
    axis_dim: usize,
    outer_count: usize,
    per_slice: F,
) where
    Out: Send,
    F: Fn(usize, &mut [Out], &mut Vec<(u64, u32)>, &mut Vec<(u64, u32)>) + Sync,
{
    let total = outer_count.saturating_mul(axis_dim);
    let hardware = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let threads = if total >= SORT_PARALLEL_MIN_TOTAL_ELEMS {
        hardware.min(total / SORT_PARALLEL_MIN_TOTAL_ELEMS).max(1)
    } else {
        1
    };

    if threads <= 1 {
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(axis_dim);
        let mut scratch: Vec<(u64, u32)> = vec![(0, 0); axis_dim];
        for (s, out_slice) in out.chunks_mut(axis_dim).enumerate() {
            per_slice(s, out_slice, &mut pairs, &mut scratch);
        }
        return;
    }

    let chunk_slices = outer_count.div_ceil(threads);
    let per_slice = &per_slice;
    std::thread::scope(|scope| {
        let mut rest: &mut [Out] = out;
        let mut start_slice = 0usize;
        while start_slice < outer_count {
            let group = chunk_slices.min(outer_count - start_slice);
            let (block, tail) = rest.split_at_mut(group * axis_dim);
            rest = tail;
            let base_slice = start_slice;
            scope.spawn(move || {
                let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(axis_dim);
                let mut scratch: Vec<(u64, u32)> = vec![(0, 0); axis_dim];
                for (j, out_slice) in block.chunks_mut(axis_dim).enumerate() {
                    per_slice(base_slice + j, out_slice, &mut pairs, &mut scratch);
                }
            });
            start_slice += group;
        }
    });
}

/// Stable ascending LSD radix sort of `(key, index)` pairs by `key` (8 byte
/// passes; `scratch` is a reusable ping-pong buffer of the same length). Equal
/// keys keep their input order (stable). After 8 (even) swaps `pairs` holds the
/// sorted result.
#[inline]
fn radix_pairs_ascending(pairs: &mut Vec<(u64, u32)>, scratch: &mut Vec<(u64, u32)>) {
    radix_pairs_ascending_passes::<8>(pairs, scratch);
}

#[inline]
fn radix_pairs_ascending_u32(pairs: &mut Vec<(u64, u32)>, scratch: &mut Vec<(u64, u32)>) {
    radix_pairs_ascending_passes::<4>(pairs, scratch);
}

#[inline]
fn radix_pairs_ascending_passes<const PASSES: usize>(
    pairs: &mut Vec<(u64, u32)>,
    scratch: &mut Vec<(u64, u32)>,
) {
    for byte in 0..PASSES {
        let shift = byte * 8;
        let mut counts = [0_usize; 256];
        for &(key, _) in pairs.iter() {
            counts[((key >> shift) & 0xff) as usize] += 1;
        }
        let mut sum = 0_usize;
        for c in counts.iter_mut() {
            let count = *c;
            *c = sum;
            sum += count;
        }
        for &pair in pairs.iter() {
            let bucket = ((pair.0 >> shift) & 0xff) as usize;
            scratch[counts[bucket]] = pair;
            counts[bucket] += 1;
        }
        std::mem::swap(pairs, scratch);
    }
}

/// Minimum pairs in a SINGLE sort slice before its radix is parallelized
/// intra-slice (MSD partition + parallel per-bucket LSD). Inter-slice threading
/// ([`for_each_contiguous_sort_slice`]) handles many-small-slice sorts; this
/// covers the complementary case — few huge slices (the 1-D `jnp.sort(x)` over
/// millions of elements) where there is only one slice to hand a thread.
const PARALLEL_RADIX_MIN_PAIRS: usize = 1 << 19;

/// Stable ascending 8-pass LSD radix of a `(key,index)` SLICE, ping-ponging with
/// the equal-length `tmp`. Eight passes (even) leave the sorted result back in
/// `data`. Identical ordering to [`radix_pairs_ascending`]; the slice form lets a
/// thread sort one MSD bucket in place.
#[inline]
fn radix_sort_slice_8pass(data: &mut [(u64, u32)], tmp: &mut [(u64, u32)]) {
    let mut src: &mut [(u64, u32)] = data;
    let mut dst: &mut [(u64, u32)] = tmp;
    for byte in 0..8 {
        let shift = byte * 8;
        let mut counts = [0_usize; 256];
        for &(key, _) in src.iter() {
            counts[((key >> shift) & 0xff) as usize] += 1;
        }
        let mut sum = 0_usize;
        for c in counts.iter_mut() {
            let count = *c;
            *c = sum;
            sum += count;
        }
        for &pair in src.iter() {
            let bucket = ((pair.0 >> shift) & 0xff) as usize;
            dst[counts[bucket]] = pair;
            counts[bucket] += 1;
        }
        std::mem::swap(&mut src, &mut dst);
    }
    // 8 swaps -> `src` aliases the original `data`, which holds the result.
}

/// Parallel stable ascending radix of `(key,index)` pairs, for a single LARGE
/// slice. One serial MSD pass on the top key byte stably scatters the pairs into
/// 256 globally key-ordered, contiguous buckets (`scratch`); the buckets are then
/// LSD-sorted on the low 7 bytes IN PARALLEL (disjoint sub-slices, each thread its
/// own temp). The top-byte LSD pass within a bucket is a constant no-op, so
/// "MSD(byte7) then within-bucket LSD(bytes 0..7)" yields the SAME total order as
/// the flat 8-pass LSD — and both are stable, so the permutation is BIT-IDENTICAL
/// to [`radix_pairs_ascending`] (proven by
/// `parallel_radix_matches_serial_radix_with_ties`). The result is left in `pairs`.
#[allow(clippy::ptr_arg)]
fn radix_pairs_ascending_parallel(pairs: &mut Vec<(u64, u32)>, scratch: &mut Vec<(u64, u32)>) {
    let n = pairs.len();
    if n <= 1 {
        return;
    }
    if scratch.len() < n {
        scratch.resize(n, (0, 0));
    }
    const SHIFT: u32 = 56; // top byte

    // MSD histogram + exclusive prefix (bucket start offsets).
    let mut starts = [0_usize; 256];
    {
        let mut counts = [0_usize; 256];
        for &(key, _) in pairs.iter() {
            counts[(key >> SHIFT) as usize] += 1;
        }
        let mut sum = 0_usize;
        for b in 0..256 {
            starts[b] = sum;
            sum += counts[b];
        }
    }
    // Stable scatter pairs -> scratch[0..n] by top byte.
    let mut wpos = starts;
    for &pair in pairs.iter() {
        let b = (pair.0 >> SHIFT) as usize;
        scratch[wpos[b]] = pair;
        wpos[b] += 1;
    }

    // Sort each non-empty bucket (a disjoint contiguous span of scratch) on the
    // low bytes, in parallel. Buckets are already globally ordered by top byte, so
    // no merge is needed — after per-bucket sort, scratch[0..n] is fully sorted.
    let hardware = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let buf = &mut scratch[..n];
    if hardware <= 1 {
        let mut tmp = vec![(0_u64, 0_u32); 0];
        let mut start = 0usize;
        for b in 0..256 {
            let end = if b == 255 { n } else { starts[b + 1] };
            let len = end - start;
            if len > 1 {
                if tmp.len() < len {
                    tmp.resize(len, (0, 0));
                }
                radix_sort_slice_8pass(&mut buf[start..end], &mut tmp[..len]);
            }
            start = end;
        }
    } else {
        // Partition the 256 buckets into contiguous GROUPS of roughly equal element
        // count (target = n/threads), each a disjoint sub-slice of `buf`. A group is
        // closed once its accumulated element count reaches `target`. `bucket_end(b)`
        // is the exclusive element offset of bucket `b`.
        let target = n.div_ceil(hardware).max(1);
        let bucket_end = |b: usize| -> usize { if b + 1 >= 256 { n } else { starts[b + 1] } };
        // group_bucket_bounds: bucket indices marking group starts, ending with 256.
        let mut group_bucket_bounds: Vec<usize> = vec![0];
        let mut group_start_elem = 0usize;
        for b in 0..256 {
            if bucket_end(b) - group_start_elem >= target && b + 1 < 256 {
                group_bucket_bounds.push(b + 1);
                group_start_elem = starts[b + 1];
            }
        }
        group_bucket_bounds.push(256);

        std::thread::scope(|scope| {
            let mut rest: &mut [(u64, u32)] = buf;
            for w in 0..group_bucket_bounds.len() - 1 {
                let gb_start = group_bucket_bounds[w];
                let gb_end = group_bucket_bounds[w + 1];
                let elem_start = starts[gb_start];
                let elem_end = if gb_end >= 256 { n } else { starts[gb_end] };
                let group_len = elem_end - elem_start;
                let (group_slice, tail) = rest.split_at_mut(group_len);
                rest = tail;
                // Per-bucket boundaries within this group, rebased to the group start.
                let local_bounds: Vec<usize> = (gb_start..=gb_end)
                    .map(|b| (if b >= 256 { n } else { starts[b] }) - elem_start)
                    .collect();
                scope.spawn(move || {
                    let mut tmp = vec![(0_u64, 0_u32); 0];
                    for win in local_bounds.windows(2) {
                        let (s, e) = (win[0], win[1]);
                        let len = e - s;
                        if len > 1 {
                            if tmp.len() < len {
                                tmp.resize(len, (0, 0));
                            }
                            radix_sort_slice_8pass(&mut group_slice[s..e], &mut tmp[..len]);
                        }
                    }
                });
            }
        });
    }

    // Sorted result currently lives in scratch[0..n]; move it back into pairs.
    pairs[..n].copy_from_slice(&scratch[..n]);
}

/// Whether a sort slice of `axis_dim` pairs, in a tensor with `outer_count`
/// slices, should use the intra-slice parallel radix: only when there is a SINGLE
/// large slice (the 1-D `jnp.sort(x)` case), where inter-slice threading has no
/// slices to fan out and the per-slice radix would otherwise run single-threaded.
#[inline]
fn use_parallel_radix(outer_count: usize, axis_dim: usize) -> bool {
    outer_count == 1 && axis_dim >= PARALLEL_RADIX_MIN_PAIRS
}

/// 8-pass `(key,index)` radix, parallel intra-slice when `parallel` is set
/// (single large slice), else the serial LSD. Bit-identical either way.
#[inline]
fn radix_pairs_ascending_maybe_parallel(
    pairs: &mut Vec<(u64, u32)>,
    scratch: &mut Vec<(u64, u32)>,
    parallel: bool,
) {
    if parallel {
        radix_pairs_ascending_parallel(pairs, scratch);
    } else {
        radix_pairs_ascending(pairs, scratch);
    }
}

/// Sort or argsort a tensor along a given axis.
fn sort_along_axis(
    primitive: Primitive,
    tensor: &TensorValue,
    axis: usize,
    descending: bool,
    return_indices: bool,
) -> Result<Value, EvalError> {
    let rank = tensor.shape.rank();
    let axis_dim = tensor.shape.dims[axis] as usize;

    if axis_dim == 0 || tensor.elements.is_empty() {
        let result_dtype = if return_indices {
            DType::I64
        } else {
            tensor.dtype
        };
        return Ok(Value::Tensor(
            TensorValue::new(result_dtype, tensor.shape.clone(), vec![])
                .map_err(EvalError::InvalidTensor)?,
        ));
    }

    // Radix fast paths (no Literal machinery), for both ascending and descending:
    // descending uses the complement key (ascending radix of `!key` == stable
    // descending). Each returns None for wrong-dtype / non-dense / short axes ->
    // generic path below.
    if tensor.dtype == DType::I64
        && let Some(value) = sort_along_axis_dense_i64(tensor, axis, descending, return_indices)?
    {
        return Ok(value);
    }
    if tensor.dtype == DType::F64
        && let Some(value) = sort_along_axis_dense_f64(tensor, axis, descending, return_indices)?
    {
        return Ok(value);
    }
    if let Some(value) = sort_along_axis_literal_radix(tensor, axis, descending, return_indices)? {
        return Ok(value);
    }

    let strides = checked_row_major_strides(primitive, "sort", &tensor.shape.dims)?;
    let axis_stride = strides[axis];
    let total = tensor.elements.len();
    if !total.is_multiple_of(axis_dim) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "sort axis dimension {axis_dim} does not divide {total} input elements"
            ),
        });
    }
    let outer_count = total / axis_dim;

    let mut result_elements = if return_indices {
        vec![Literal::I64(0); total]
    } else {
        tensor.elements.to_vec()
    };

    let mut indexed = Vec::with_capacity(axis_dim);
    for outer in 0..outer_count {
        let base = {
            let mut idx = outer;
            let mut flat = 0_usize;
            for ax in (0..rank).rev() {
                if ax == axis {
                    continue;
                }
                let dim = tensor.shape.dims[ax] as usize;
                if dim == 0 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "sort encountered zero non-axis dimension in non-empty tensor"
                            .to_owned(),
                    });
                }
                let offset =
                    (idx % dim)
                        .checked_mul(strides[ax])
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "sort flat offset multiplication overflowed usize".to_owned(),
                        })?;
                flat = flat
                    .checked_add(offset)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "sort flat offset addition overflowed usize".to_owned(),
                    })?;
                idx /= dim;
            }
            flat
        };

        indexed.clear();
        for i in 0..axis_dim {
            let flat_idx = i
                .checked_mul(axis_stride)
                .and_then(|offset| base.checked_add(offset))
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "sort axis offset overflowed usize".to_owned(),
                })?;
            let literal = *tensor
                .elements
                .get(flat_idx)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "sort flat index {flat_idx} out of bounds for {total} elements"
                    ),
                })?;
            let key =
                sort_key(literal).map_err(|detail| EvalError::Unsupported { primitive, detail })?;
            indexed.push((i, key));
        }

        if descending {
            indexed.sort_by(|a, b| compare_sort_keys_nan_last(b.1, a.1));
        } else {
            indexed.sort_by(|a, b| compare_sort_keys_nan_last(a.1, b.1));
        }

        for (out_pos, &(orig_idx, _)) in indexed.iter().enumerate() {
            let flat_idx = out_pos
                .checked_mul(axis_stride)
                .and_then(|offset| base.checked_add(offset))
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "sort output offset overflowed usize".to_owned(),
                })?;
            if return_indices {
                *result_elements
                    .get_mut(flat_idx)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "sort output index {flat_idx} out of bounds for {total} elements"
                        ),
                    })? = Literal::I64(orig_idx as i64);
            } else {
                let source_idx = orig_idx
                    .checked_mul(axis_stride)
                    .and_then(|offset| base.checked_add(offset))
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "sort source offset overflowed usize".to_owned(),
                    })?;
                let source =
                    *tensor
                        .elements
                        .get(source_idx)
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: format!(
                                "sort source index {source_idx} out of bounds for {total} elements"
                            ),
                        })?;
                *result_elements
                    .get_mut(flat_idx)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "sort output index {flat_idx} out of bounds for {total} elements"
                        ),
                    })? = source;
            }
        }
    }

    let out_dtype = if return_indices {
        DType::I64
    } else {
        tensor.dtype
    };
    Ok(Value::Tensor(
        TensorValue::new(out_dtype, tensor.shape.clone(), result_elements)
            .map_err(EvalError::InvalidTensor)?,
    ))
}

#[derive(Clone, Copy)]
enum SortKey {
    Bool(bool),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
    /// Complex values order lexicographically by (real, imaginary), matching
    /// JAX's `lax.sort` comparator (`_operands_to_keys` splits complex into
    /// [real, imag] float keys) and NumPy's complex sort. Both components are
    /// widened to f64 so Complex64 and Complex128 share one comparison path.
    Complex(f64, f64),
}

/// First index of the extremum along a float slice, matching JAX's
/// `_ArgMinMaxReducer` (jax/_src/lax/lax.py): a candidate replaces the
/// accumulator when it is strictly better under IEEE `>` / `<` (so ±0.0 compare
/// equal and the first occurrence wins ties) OR when it is NaN (sign-agnostic,
/// via `v != v`). The first NaN therefore wins outright and is sticky — later
/// NaNs and finite values cannot displace it — which is exactly what
/// `jnp.argmax` / `jnp.argmin` return (verified against JAX CPU). This is the
/// real-float reducer; integer/bool/complex dtypes keep the total-order
/// `sort_key` path. It differs from `total_cmp` in two JAX-relevant ways:
/// total_cmp ranks -NaN below -inf (so a -NaN is missed by argmax / picked
/// wrongly by argmin), and ranks -0.0 < +0.0 (so a later +0.0 would beat an
/// earlier -0.0). `n` is the slice length (`axis_dim`, always ≥ 1 here).
fn arg_extreme_float<F: Fn(usize) -> f64>(n: usize, find_max: bool, get: F) -> usize {
    let mut best_idx = 0_usize;
    let mut best = get(0);
    let mut best_nan = best.is_nan();
    let mut i = 1;
    while i < n && !best_nan {
        let v = get(i);
        if v.is_nan() {
            best_idx = i;
            best_nan = true;
        } else if (find_max && v > best) || (!find_max && v < best) {
            best_idx = i;
            best = v;
        }
        i += 1;
    }
    best_idx
}

fn arg_extreme_f64_contiguous_simd(values: &[f64], find_max: bool) -> usize {
    use std::simd::cmp::SimdPartialOrd;
    use std::simd::num::SimdFloat;
    use std::simd::{Select, Simd};

    const LANES: usize = 8;
    debug_assert!(!values.is_empty());
    if values.len() < LANES {
        return arg_extreme_float(values.len(), find_max, |i| values[i]);
    }

    let (first, rest) = values.split_at(LANES);
    let first_values = Simd::<f64, LANES>::from_slice(first);
    if first_values.is_nan().any() {
        return arg_extreme_float(values.len(), find_max, |i| values[i]);
    }

    let mut best_values = first_values;
    let mut best_indices = Simd::<usize, LANES>::from_array(std::array::from_fn(|lane| lane));
    let mut offset = LANES;
    let mut chunks = rest.chunks_exact(LANES);

    for chunk in chunks.by_ref() {
        let row_values = Simd::<f64, LANES>::from_slice(chunk);
        if row_values.is_nan().any() {
            return arg_extreme_float(values.len(), find_max, |i| values[i]);
        }
        let row_indices =
            Simd::<usize, LANES>::from_array(std::array::from_fn(|lane| offset + lane));
        let better = if find_max {
            row_values.simd_gt(best_values)
        } else {
            row_values.simd_lt(best_values)
        };
        best_values = better.select(row_values, best_values);
        best_indices = better.select(row_indices, best_indices);
        offset += LANES;
    }

    let best_values_array = best_values.to_array();
    let best_indices_array = best_indices.to_array();
    let mut best = best_values_array[0];
    let mut best_idx = best_indices_array[0];
    for lane in 1..LANES {
        let value = best_values_array[lane];
        let idx = best_indices_array[lane];
        let better = if find_max { value > best } else { value < best };
        if better || (value == best && idx < best_idx) {
            best = value;
            best_idx = idx;
        }
    }

    for (tail_offset, &value) in chunks.remainder().iter().enumerate() {
        if value.is_nan() {
            return arg_extreme_float(values.len(), find_max, |i| values[i]);
        }
        let better = if find_max { value > best } else { value < best };
        if better {
            best = value;
            best_idx = offset + tail_offset;
        }
    }

    best_idx
}

fn arg_extreme_f32_contiguous_simd(values: &[f32], find_max: bool) -> usize {
    use std::simd::cmp::SimdPartialOrd;
    use std::simd::num::SimdFloat;
    use std::simd::{Select, Simd};

    const LANES: usize = 16;
    debug_assert!(!values.is_empty());
    if values.len() < LANES {
        return arg_extreme_float(values.len(), find_max, |i| f64::from(values[i]));
    }

    let (first, rest) = values.split_at(LANES);
    let first_values = Simd::<f32, LANES>::from_slice(first);
    if first_values.is_nan().any() {
        return arg_extreme_float(values.len(), find_max, |i| f64::from(values[i]));
    }

    let mut best_values = first_values;
    let mut best_indices = Simd::<usize, LANES>::from_array(std::array::from_fn(|lane| lane));
    let mut offset = LANES;
    let mut chunks = rest.chunks_exact(LANES);

    for chunk in chunks.by_ref() {
        let row_values = Simd::<f32, LANES>::from_slice(chunk);
        if row_values.is_nan().any() {
            return arg_extreme_float(values.len(), find_max, |i| f64::from(values[i]));
        }
        let row_indices =
            Simd::<usize, LANES>::from_array(std::array::from_fn(|lane| offset + lane));
        let better = if find_max {
            row_values.simd_gt(best_values)
        } else {
            row_values.simd_lt(best_values)
        };
        best_values = better.select(row_values, best_values);
        best_indices = better.select(row_indices, best_indices);
        offset += LANES;
    }

    let best_values_array = best_values.to_array();
    let best_indices_array = best_indices.to_array();
    let mut best = best_values_array[0];
    let mut best_idx = best_indices_array[0];
    for lane in 1..LANES {
        let value = best_values_array[lane];
        let idx = best_indices_array[lane];
        let better = if find_max { value > best } else { value < best };
        if better || (value == best && idx < best_idx) {
            best = value;
            best_idx = idx;
        }
    }

    for (tail_offset, &value) in chunks.remainder().iter().enumerate() {
        if value.is_nan() {
            return arg_extreme_float(values.len(), find_max, |i| f64::from(values[i]));
        }
        let better = if find_max { value > best } else { value < best };
        if better {
            best = value;
            best_idx = offset + tail_offset;
        }
    }

    best_idx
}

fn sort_key(literal: Literal) -> Result<SortKey, String> {
    match literal {
        Literal::Bool(value) => Ok(SortKey::Bool(value)),
        Literal::I64(value) => Ok(SortKey::Signed(value)),
        Literal::U32(value) => Ok(SortKey::Unsigned(u64::from(value))),
        Literal::U64(value) => Ok(SortKey::Unsigned(value)),
        Literal::BF16Bits(_) | Literal::F16Bits(_) | Literal::F32Bits(_) | Literal::F64Bits(_) => {
            literal
                .as_f64()
                .map(SortKey::Float)
                .ok_or_else(|| format!("sort requires orderable numeric literals, got {literal:?}"))
        }
        Literal::Complex64Bits(re, im) => Ok(SortKey::Complex(
            f64::from(f32::from_bits(re)),
            f64::from(f32::from_bits(im)),
        )),
        Literal::Complex128Bits(re, im) => {
            Ok(SortKey::Complex(f64::from_bits(re), f64::from_bits(im)))
        }
    }
}

fn compare_sort_keys(lhs: SortKey, rhs: SortKey) -> Ordering {
    match (lhs, rhs) {
        (SortKey::Bool(lhs), SortKey::Bool(rhs)) => lhs.cmp(&rhs),
        (SortKey::Signed(lhs), SortKey::Signed(rhs)) => lhs.cmp(&rhs),
        (SortKey::Unsigned(lhs), SortKey::Unsigned(rhs)) => lhs.cmp(&rhs),
        (SortKey::Float(lhs), SortKey::Float(rhs)) => lhs.total_cmp(&rhs),
        (SortKey::Complex(lhs_re, lhs_im), SortKey::Complex(rhs_re, rhs_im)) => lhs_re
            .total_cmp(&rhs_re)
            .then_with(|| lhs_im.total_cmp(&rhs_im)),
        (lhs, rhs) => sort_key_rank(lhs).cmp(&sort_key_rank(rhs)),
    }
}

fn sort_key_rank(key: SortKey) -> u8 {
    match key {
        SortKey::Bool(_) => 0,
        SortKey::Signed(_) => 1,
        SortKey::Unsigned(_) => 2,
        SortKey::Float(_) => 3,
        SortKey::Complex(..) => 4,
    }
}

/// `compare_sort_keys` for the generic sort/argsort path, matching JAX/numpy
/// `sort`: every float NaN (either sign) is the maximum and all NaN compare
/// EQUAL (so a stable sort keeps them in input order). This is the comparison
/// twin of [`f64_sort_order_key`] — together they keep the radix and generic
/// sort paths bit-identical while sending NaN to the end (ascending) / front
/// (descending). Non-NaN floats and every other key kind defer to
/// `compare_sort_keys` (plain `total_cmp`), which `top_k` and `argmin/argmax`
/// keep using unchanged.
fn compare_sort_keys_nan_last(lhs: SortKey, rhs: SortKey) -> Ordering {
    match (lhs, rhs) {
        (SortKey::Float(l), SortKey::Float(r)) => match (l.is_nan(), r.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => l.total_cmp(&r),
        },
        (lhs, rhs) => compare_sort_keys(lhs, rhs),
    }
}

/// Conv: N-dimensional convolution.
/// Layout: lhs=[batch, spatial..., in_channels], rhs=[kernel_spatial..., in_channels, out_channels]
/// Params: strides (comma-sep), padding ("VALID", "SAME", or "SAME_LOWER")
pub(crate) fn eval_conv(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let lhs = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "conv requires tensor inputs".into(),
            });
        }
    };
    let rhs = match &inputs[1] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "conv requires tensor kernel".into(),
            });
        }
    };

    let is_numeric = |dtype: DType| {
        matches!(
            dtype,
            DType::BF16
                | DType::F16
                | DType::F32
                | DType::F64
                | DType::Complex64
                | DType::Complex128
        )
    };
    if !is_numeric(lhs.dtype) || !is_numeric(rhs.dtype) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "conv requires floating or complex dtypes, got lhs {:?}, rhs {:?}",
                lhs.dtype, rhs.dtype
            ),
        });
    }

    let lhs_rank = lhs.shape.rank();
    if lhs_rank < 3 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("conv lhs must have rank >= 3, got {lhs_rank}"),
        });
    }

    if lhs_rank > 4 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("conv supports rank 3 (1D) and rank 4 (2D), got rank {lhs_rank}"),
        });
    }

    // lhs_dilation (input dilation = transposed / fractionally-strided conv):
    // conceptually inserts (L-1) zeros between input elements along each spatial axis,
    // then runs an ordinary conv. Implemented by materializing the zero-dilated input
    // and recursing with lhs_dilation stripped — exact by definition, and it composes
    // with stride / rhs_dilation / grouping / padding for free (the recursion runs the
    // full conv over the dilated input). Done before the param-reject guard.
    let num_spatial = lhs_rank - 2;
    let lhs_dils = parse_conv_lhs_dilation(primitive, params, num_spatial)?;
    if lhs_dils.iter().any(|&l| l > 1) {
        let dilated = dilate_conv_lhs(primitive, lhs, &lhs_dils)?;
        let mut inner = params.clone();
        inner.remove("lhs_dilation");
        return eval_conv(
            primitive,
            &[Value::Tensor(dilated), Value::Tensor(rhs.clone())],
            &inner,
        );
    }

    // batch_group_count (used in the gradient w.r.t. the kernel of a grouped conv):
    // the dual of feature_group_count. The batch splits into `g` group-major blocks
    // and the output channels into `g` group-major blocks; group `go` convolves batch
    // block `go` with out-channel block `go`. XLA forward signature `[gm,j][j,gk]->[m,gk]`
    // (jax convolution.py: the group is the MAJOR part of both batch and out-feature via
    // `_reshape_axis_out_of`), so output batch = N/g and output features = Cout. Decompose
    // into `g` independent plain convs (slice batch + slice out-channels, recurse with the
    // count stripped, concatenate over the channel axis) — exact by construction and reusing
    // the full conv machinery (1D and 2D). Done before the reject guard, like lhs_dilation.
    let batch_groups = parse_conv_named_count(primitive, params, "batch_group_count")?;
    if batch_groups > 1 {
        return eval_conv_batch_grouped(primitive, lhs, rhs, params, batch_groups);
    }

    let padding = parse_conv_padding(primitive, params)?;
    reject_unsupported_conv_params(primitive, params)?;

    if lhs_rank == 3 {
        eval_conv_1d(primitive, lhs, rhs, params, padding)
    } else {
        eval_conv_2d(primitive, lhs, rhs, params, padding)
    }
}

/// Convolution with `batch_group_count > 1`, decomposed into `g` independent plain
/// convolutions. The batch dimension (axis 0) and the output-channel dimension (the
/// last axis of `rhs`, fj layout `[(KH,)KW,Cin,Cout]`) each split into `g` group-major
/// blocks; group `go` convolves `lhs[go*m..(go+1)*m]` with `rhs[..,go*k..(go+1)*k]`,
/// and the per-group outputs concatenate along the output-channel axis. Bit-identical
/// to running each group as a standalone conv (no shared accumulation across groups).
fn eval_conv_batch_grouped(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    params: &BTreeMap<String, String>,
    batch_groups: usize,
) -> Result<Value, EvalError> {
    // JAX rejects combining the two groupings (convolution.py): "At most one of
    // batch_group_count and feature_group_count may be > 1".
    if parse_conv_group_count(primitive, params)? > 1 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "conv: at most one of batch_group_count and feature_group_count may be > 1"
                .to_owned(),
        });
    }

    let lhs_rank = lhs.shape.rank();
    let n = lhs.shape.dims[0] as usize;
    let c_out = *rhs.shape.dims.last().expect("rhs has rank >= 3") as usize;
    if !n.is_multiple_of(batch_groups) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("conv batch_group_count {batch_groups} must divide lhs batch {n}"),
        });
    }
    if !c_out.is_multiple_of(batch_groups) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "conv: rhs output-feature count {c_out} must be a multiple of batch_group_count {batch_groups}"
            ),
        });
    }
    let m = n / batch_groups;
    let k = c_out / batch_groups;

    // Inner params: strip batch_group_count so each per-group conv is ordinary.
    let mut inner = params.clone();
    inner.remove("batch_group_count");

    // Helper: static slice `tensor` to `[start, limit)` on `axis`, full extent elsewhere.
    let slice_axis = |tensor: &TensorValue, axis: usize, start: usize, limit: usize| {
        let rank = tensor.shape.rank();
        let starts: Vec<String> = (0..rank)
            .map(|d| if d == axis { start } else { 0 }.to_string())
            .collect();
        let limits: Vec<String> = (0..rank)
            .map(|d| {
                if d == axis {
                    limit
                } else {
                    tensor.shape.dims[d] as usize
                }
                .to_string()
            })
            .collect();
        let sp = BTreeMap::from([
            ("start_indices".to_owned(), starts.join(",")),
            ("limit_indices".to_owned(), limits.join(",")),
        ]);
        eval_slice(&[Value::Tensor(tensor.clone())], &sp)
    };

    let rhs_chan_axis = rhs.shape.rank() - 1;
    let mut group_outs: Vec<Value> = Vec::with_capacity(batch_groups);
    for go in 0..batch_groups {
        let lhs_g = slice_axis(lhs, 0, go * m, (go + 1) * m)?;
        let rhs_g = slice_axis(rhs, rhs_chan_axis, go * k, (go + 1) * k)?;
        let out_g = eval_conv(primitive, &[lhs_g, rhs_g], &inner)?;
        group_outs.push(out_g);
    }

    // Concatenate the per-group outputs along the output-channel axis (last axis of
    // the output, which has the same rank as lhs: `[N/g, spatial.., Cout]`).
    let out_chan_axis = lhs_rank - 1;
    let cat = BTreeMap::from([("dimension".to_owned(), out_chan_axis.to_string())]);
    eval_concatenate(&group_outs, &cat)
}

/// Parse `lhs_dilation` as `num_spatial` per-axis factors (>= 1). Absent → all 1s;
/// a single value broadcasts to every spatial axis; otherwise the list length must
/// equal `num_spatial`.
fn parse_conv_lhs_dilation(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
    num_spatial: usize,
) -> Result<Vec<usize>, EvalError> {
    let Some(raw) = params.get("lhs_dilation") else {
        return Ok(vec![1; num_spatial]);
    };
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
    let factors: Vec<usize> = parts
        .iter()
        .map(|p| parse_positive_stride(primitive, Some(p)))
        .collect::<Result<_, _>>()?;
    if factors.len() == 1 {
        Ok(vec![factors[0]; num_spatial])
    } else if factors.len() == num_spatial {
        Ok(factors)
    } else {
        Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "lhs_dilation {raw:?} must have 1 or {num_spatial} entries for a rank-{} conv",
                num_spatial + 2
            ),
        })
    }
}

/// Build the zero-dilated conv input for `lhs_dilation`: for layout `[N, S0.., C]`,
/// insert `(L_i - 1)` zeros between elements along spatial axis `i`, giving spatial
/// extent `(S_i - 1)*L_i + 1`. Real and complex dtypes both get a dtype-correct zero
/// fill; each original element is scattered to its dilated position.
fn dilate_conv_lhs(
    primitive: Primitive,
    lhs: &TensorValue,
    lhs_dils: &[usize],
) -> Result<TensorValue, EvalError> {
    let dims = &lhs.shape.dims;
    let rank = dims.len();
    let num_spatial = rank - 2;
    let mut new_dims = dims.clone();
    for (i, &l) in lhs_dils.iter().enumerate().take(num_spatial) {
        let s = dims[1 + i] as usize;
        let eff = if s == 0 { 0 } else { (s - 1) * l + 1 };
        new_dims[1 + i] = u32::try_from(eff).map_err(|_| EvalError::Unsupported {
            primitive,
            detail: "lhs_dilation output dimension exceeds u32::MAX".to_owned(),
        })?;
    }
    let new_total = new_dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "lhs_dilation output element count overflow".to_owned(),
        })?;

    let zero = if matches!(lhs.dtype, DType::Complex64 | DType::Complex128) {
        conv_literal_from_complex(lhs.dtype, 0.0, 0.0)
    } else {
        conv_float_literal_from_f64(lhs.dtype, 0.0)
    };
    let mut out = vec![zero; new_total];

    let out_strides = checked_row_major_strides(primitive, "lhs_dilation", &new_dims)?;
    let in_total = lhs.elements.len();
    let mut coord = vec![0usize; rank];
    for in_flat in 0..in_total {
        let mut out_flat = 0usize;
        for ax in 0..rank {
            let mapped = if ax >= 1 && ax <= num_spatial {
                coord[ax] * lhs_dils[ax - 1]
            } else {
                coord[ax]
            };
            out_flat += mapped * out_strides[ax];
        }
        out[out_flat] = lhs.elements[in_flat];
        // row-major odometer over the input dims (last axis fastest).
        for ax in (0..rank).rev() {
            coord[ax] += 1;
            if coord[ax] < dims[ax] as usize {
                break;
            }
            coord[ax] = 0;
        }
    }

    TensorValue::new(lhs.dtype, Shape { dims: new_dims }, out).map_err(EvalError::from)
}

/// fj-lax conv implements stride + padding (VALID/SAME/SAME_LOWER) on the default
/// `[N,(H,)W,Cin]` / `[(KH,)KW,Cin,Cout]` layout. The `conv_general_dilated` parameters
/// it implements via dilation, grouping, and (now) batch grouping. The remaining
/// `conv_general_dilated` parameters it does NOT implement must be REJECTED rather
/// than silently ignored — a silent parity violation vs `jax.lax.conv_general_dilated`.
/// `batch_group_count > 1` is handled in `eval_conv` (decomposed before this guard),
/// so it never reaches here; `feature_group_count` is handled by the grouped/depthwise
/// paths; `lhs_dilation`/`rhs_dilation` are supported. Nothing currently remains to
/// reject at this point, but the guard is retained as the single chokepoint where any
/// future unsupported param would fail loudly.
fn reject_unsupported_conv_params(
    _primitive: Primitive,
    _params: &BTreeMap<String, String>,
) -> Result<(), EvalError> {
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConvPadding {
    Valid,
    Same,
    SameLower,
}

fn parse_conv_padding(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
) -> Result<ConvPadding, EvalError> {
    let Some(raw) = params.get("padding") else {
        return Ok(ConvPadding::Valid);
    };
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("VALID") {
        Ok(ConvPadding::Valid)
    } else if trimmed.eq_ignore_ascii_case("SAME") {
        Ok(ConvPadding::Same)
    } else if trimmed.eq_ignore_ascii_case("SAME_LOWER") {
        Ok(ConvPadding::SameLower)
    } else {
        Err(EvalError::Unsupported {
            primitive,
            detail: format!("unsupported conv padding mode {raw:?}"),
        })
    }
}

fn conv_float_literal_from_f64(dtype: DType, value: f64) -> Literal {
    match dtype {
        DType::BF16 => Literal::from_bf16_f64(value),
        DType::F16 => Literal::from_f16_f64(value),
        DType::F32 => Literal::from_f32(value as f32),
        _ => Literal::from_f64(value),
    }
}

/// Extract a real-float conv operand's elements as f64 (F64 borrowed; F32/BF16/F16
/// promoted losslessly), or None for non-real-float literals. The generic conv
/// loop accumulates in f64 over these exact promotions and rounds to the output
/// dtype, so an im2col+GEMM on these values + rounding back is bit-identical.
fn conv_real_elements_as_f64(tensor: &TensorValue) -> Option<Cow<'_, [f64]>> {
    if let Some(values) = tensor.elements.as_f64_slice() {
        return Some(Cow::Borrowed(values));
    }
    let mut out = Vec::with_capacity(tensor.elements.len());
    for literal in &tensor.elements {
        match literal {
            Literal::F64Bits(_)
            | Literal::F32Bits(_)
            | Literal::BF16Bits(_)
            | Literal::F16Bits(_) => {
                out.push(literal.as_f64()?);
            }
            _ => return None,
        }
    }
    Some(Cow::Owned(out))
}

/// Decode an `F32` conv operand to `f32` (borrowing a dense f32 backing, else
/// unpacking boxed `F32Bits`). Used by the native-f32 conv path so f32 convolution
/// accumulates in f32 (XLA parity) regardless of storage — never widening to f64.
fn conv_f32_elements(tensor: &TensorValue) -> Option<Cow<'_, [f32]>> {
    if let Some(values) = tensor.elements.as_f32_slice() {
        return Some(Cow::Borrowed(values));
    }
    let mut out = Vec::with_capacity(tensor.elements.len());
    for literal in &tensor.elements {
        match literal {
            Literal::F32Bits(b) => out.push(f32::from_bits(*b)),
            _ => return None,
        }
    }
    Some(Cow::Owned(out))
}

/// Decode an `F32`/`BF16`/`F16` conv operand to `f32`. F32 borrows/unpacks directly;
/// BF16/F16 widen EXACTLY to f32 (`as_{bf16,f16}_f32` — bf16 is a 16-bit shift, f16
/// the canonical half decode) WITHOUT going through f64. This feeds the native-f32
/// conv accumulation (XLA parity: XLA accumulates half/f32 conv in f32). Returns
/// `None` for any other dtype / storage.
fn conv_decode_to_f32(tensor: &TensorValue) -> Option<Cow<'_, [f32]>> {
    if tensor.dtype == DType::F32 {
        return conv_f32_elements(tensor);
    }
    let is_bf16 = match tensor.dtype {
        DType::BF16 => true,
        DType::F16 => false,
        _ => return None,
    };
    let decode = |lit: Literal| -> Option<f32> {
        if is_bf16 {
            lit.as_bf16_f32()
        } else {
            lit.as_f16_f32()
        }
    };
    if let Some(bits) = tensor.elements.as_half_float_slice() {
        let mut out = Vec::with_capacity(bits.len());
        for &b in bits {
            let lit = if is_bf16 {
                Literal::BF16Bits(b)
            } else {
                Literal::F16Bits(b)
            };
            out.push(decode(lit)?);
        }
        return Some(Cow::Owned(out));
    }
    let mut out = Vec::with_capacity(tensor.elements.len());
    for &lit in &tensor.elements {
        match lit {
            Literal::BF16Bits(_) if is_bf16 => out.push(decode(lit)?),
            Literal::F16Bits(_) if !is_bf16 => out.push(decode(lit)?),
            _ => return None,
        }
    }
    Some(Cow::Owned(out))
}

/// Materialize a native-f32-accum conv result (`out`, one f32 per output element)
/// into an `out_dtype` tensor: F32 stays dense f32; BF16/F16 round each f32 to half
/// via `conv_float_literal_from_f64(out_dtype, f64::from(v))` (= round-to-nearest-even
/// f32->half, since `f64::from(f32)` is exact and the round-to-odd step is then the
/// identity — exactly XLA's half conv output rounding).
fn conv_f32_output_to_value(
    out_dtype: DType,
    dims: Vec<u32>,
    out: Vec<f32>,
) -> Result<Value, EvalError> {
    let shape = Shape { dims };
    match out_dtype {
        DType::F32 => Ok(Value::Tensor(TensorValue::new_f32_values(shape, out)?)),
        DType::BF16 | DType::F16 => {
            let bits: Vec<u16> = out
                .iter()
                .map(
                    |&v| match conv_float_literal_from_f64(out_dtype, f64::from(v)) {
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                        _ => 0,
                    },
                )
                .collect();
            Ok(Value::Tensor(TensorValue::new_half_float_values(
                out_dtype, shape, bits,
            )?))
        }
        _ => unreachable!("conv_f32_output_to_value handles only F32/BF16/F16"),
    }
}

/// Build a conv output tensor from f64 accumulators, rounding each to `out_dtype`
/// with `conv_float_literal_from_f64` (F64 keeps the dense new_f64_values path).
fn conv_real_output_from_f64(
    out_dtype: DType,
    dims: Vec<u32>,
    out: Vec<f64>,
) -> Result<Value, EvalError> {
    let shape = Shape { dims };
    if out_dtype == DType::F64 {
        return Ok(Value::Tensor(TensorValue::new_f64_values(shape, out)?));
    }
    // Dense f32 output: emit dense `f32` storage instead of boxed `Literal`s so
    // the conv result feeds downstream f32 elementwise (bias-add/activation)
    // densely without re-boxing. Bit-identical: `conv_float_literal_from_f64(F32,
    // v)` == `Literal::from_f32(v as f32)`, exactly what `new_f32_values` stores.
    if out_dtype == DType::F32 {
        let values: Vec<f32> = out.iter().map(|&v| v as f32).collect();
        return Ok(Value::Tensor(TensorValue::new_f32_values(shape, values)?));
    }
    let elements: Vec<Literal> = out
        .iter()
        .map(|&v| conv_float_literal_from_f64(out_dtype, v))
        .collect();
    Ok(Value::Tensor(TensorValue::new(out_dtype, shape, elements)?))
}

fn conv_literal_from_complex(dtype: DType, re: f64, im: f64) -> Literal {
    match dtype {
        DType::Complex64 => Literal::from_complex64(re as f32, im as f32),
        DType::Complex128 => Literal::from_complex128(re, im),
        // For real dtypes, ignore imaginary part (shouldn't happen in valid conv)
        DType::BF16 => Literal::from_bf16_f64(re),
        DType::F16 => Literal::from_f16_f64(re),
        DType::F32 => Literal::from_f32(re as f32),
        _ => Literal::from_f64(re),
    }
}

/// Extract complex value from literal, returning (re, im).
fn literal_as_complex(lit: &Literal) -> (f64, f64) {
    if let Some((re, im)) = lit.as_complex128() {
        (re, im)
    } else if let Some((re, im)) = lit.as_complex64() {
        (re as f64, im as f64)
    } else if let Some(v) = lit.as_f64() {
        (v, 0.0)
    } else {
        (0.0, 0.0)
    }
}

/// 1D convolution: lhs=[N, W, C_in], rhs=[K, C_in, C_out]
/// Grouped/depthwise 1D conv (feature_group_count > 1). Output channel `co` belongs
/// to group `g = co / (Cout/G)` and convolves only that group's `Cin/G` input
/// channels (`lhs` channels `[g*rhs_c_in .. (g+1)*rhs_c_in)`) with kernel slice
/// `rhs[:,:,co]` (kernel c_in dim is `rhs_c_in == Cin/G`). Direct loop (real
/// f64-accum or complex), same ascending (k,ci) order, zero-padding, and `k*dil`
/// atrous tap mapping as the ungrouped path — so it equals a per-group ungrouped
/// conv concatenated over output channels.
#[allow(clippy::too_many_arguments)]
fn eval_conv_1d_grouped(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    group_count: usize,
    batch: usize,
    width: usize,
    c_in: usize,
    kernel_w: usize,
    rhs_c_in: usize,
    c_out: usize,
    stride: usize,
    dil: usize,
    out_w: usize,
    pad_left: usize,
) -> Result<Value, EvalError> {
    let out_dtype = promote_dtype(lhs.dtype, rhs.dtype);
    let cout_per_group = c_out / group_count;
    let total = batch
        .checked_mul(out_w)
        .and_then(|v| v.checked_mul(c_out))
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "grouped conv output size overflow".into(),
        })?;
    let width_c_in = width * c_in;
    // Kernel layout [K, rhs_c_in, Cout].
    let rhs_c_in_c_out = rhs_c_in * c_out;
    let out_dims = vec![batch as u32, out_w as u32, c_out as u32];

    if matches!(out_dtype, DType::Complex64 | DType::Complex128) {
        let mut elements = Vec::with_capacity(total);
        for n in 0..batch {
            let n_off = n * width_c_in;
            for w in 0..out_w {
                for co in 0..c_out {
                    let in_ch_base = (co / cout_per_group) * rhs_c_in;
                    let (mut acc_re, mut acc_im) = (0.0_f64, 0.0_f64);
                    for k in 0..kernel_w {
                        let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                        let oob = in_pos < 0 || (in_pos as usize) >= width;
                        for ci in 0..rhs_c_in {
                            let rhs_idx = k * rhs_c_in_c_out + ci * c_out + co;
                            let (lre, lim) = if oob {
                                (0.0, 0.0)
                            } else {
                                literal_as_complex(
                                    &lhs.elements
                                        [n_off + (in_pos as usize) * c_in + in_ch_base + ci],
                                )
                            };
                            let (rre, rim) = literal_as_complex(&rhs.elements[rhs_idx]);
                            acc_re += lre * rre - lim * rim;
                            acc_im += lre * rim + lim * rre;
                        }
                    }
                    elements.push(conv_literal_from_complex(out_dtype, acc_re, acc_im));
                }
            }
        }
        return Ok(Value::Tensor(TensorValue::new(
            out_dtype,
            Shape { dims: out_dims },
            elements,
        )?));
    }

    // Native f32 accumulation for grouped/depthwise f32/bf16/f16 conv1d. This is
    // the grouped sibling of the ungrouped native path: XLA accumulates these
    // dtypes in f32, while the generic real path below widens to f64 and rounds
    // back, making the result too precise. The loop nests mirror the f64 fast
    // paths exactly, preserving output order and per-output ascending (k, ci)
    // accumulation. OOB taps still add 0.0*rhs in f32, preserving zero-padding
    // behavior for infinities and signed zeros.
    if matches!(out_dtype, DType::F32 | DType::BF16 | DType::F16)
        && lhs.dtype == out_dtype
        && rhs.dtype == out_dtype
        && let (Some(lhs_cow), Some(rhs_cow)) = (conv_decode_to_f32(lhs), conv_decode_to_f32(rhs))
    {
        let lhs_src: &[f32] = &lhs_cow;
        let rhs_src: &[f32] = &rhs_cow;

        if rhs_c_in == 1 && cout_per_group == 1 {
            let mut out = vec![0.0_f32; total];
            let mut oi = 0usize;
            for n in 0..batch {
                let n_off = n * width_c_in;
                for w in 0..out_w {
                    let acc = &mut out[oi..oi + c_out];
                    for k in 0..kernel_w {
                        let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                        let rbase = k * rhs_c_in_c_out;
                        let rhs_row = &rhs_src[rbase..rbase + c_out];
                        if in_pos < 0 || (in_pos as usize) >= width {
                            #[allow(
                                clippy::erasing_op,
                                reason = "preserve scalar f32 conv OOB 0.0*rhs behavior"
                            )]
                            for (a, &r) in acc.iter_mut().zip(rhs_row) {
                                *a += 0.0_f32 * r;
                            }
                        } else {
                            let base = n_off + (in_pos as usize) * c_in;
                            let lhs_row = &lhs_src[base..base + c_out];
                            for ((a, &l), &r) in acc.iter_mut().zip(lhs_row).zip(rhs_row) {
                                *a += l * r;
                            }
                        }
                    }
                    oi += c_out;
                }
            }
            return conv_f32_output_to_value(out_dtype, out_dims, out);
        }

        if cout_per_group > 1 {
            let mut out = vec![0.0_f32; total];
            let mut spatial_base = 0usize;
            for n in 0..batch {
                let n_off = n * width_c_in;
                for w in 0..out_w {
                    let acc = &mut out[spatial_base..spatial_base + c_out];
                    for k in 0..kernel_w {
                        let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                        let oob = in_pos < 0 || (in_pos as usize) >= width;
                        let lhs_base = if oob {
                            0
                        } else {
                            n_off + (in_pos as usize) * c_in
                        };
                        let rhs_k = k * rhs_c_in_c_out;
                        for ci in 0..rhs_c_in {
                            let rhs_ci = rhs_k + ci * c_out;
                            for g in 0..group_count {
                                let lhs_val = if oob {
                                    0.0
                                } else {
                                    lhs_src[lhs_base + g * rhs_c_in + ci]
                                };
                                let co_base = g * cout_per_group;
                                let rhs_row =
                                    &rhs_src[rhs_ci + co_base..rhs_ci + co_base + cout_per_group];
                                let acc_g = &mut acc[co_base..co_base + cout_per_group];
                                for (a, &r) in acc_g.iter_mut().zip(rhs_row) {
                                    *a += lhs_val * r;
                                }
                            }
                        }
                    }
                    spatial_base += c_out;
                }
            }
            return conv_f32_output_to_value(out_dtype, out_dims, out);
        }

        let mut out = Vec::with_capacity(total);
        for n in 0..batch {
            let n_off = n * width_c_in;
            for w in 0..out_w {
                for co in 0..c_out {
                    let in_ch_base = (co / cout_per_group) * rhs_c_in;
                    let mut acc = 0.0_f32;
                    for k in 0..kernel_w {
                        let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                        let oob = in_pos < 0 || (in_pos as usize) >= width;
                        for ci in 0..rhs_c_in {
                            let rhs_idx = k * rhs_c_in_c_out + ci * c_out + co;
                            let lhs_val = if oob {
                                0.0
                            } else {
                                lhs_src[n_off + (in_pos as usize) * c_in + in_ch_base + ci]
                            };
                            acc += lhs_val * rhs_src[rhs_idx];
                        }
                    }
                    out.push(acc);
                }
            }
        }
        return conv_f32_output_to_value(out_dtype, out_dims, out);
    }

    let (Some(lhs_cow), Some(rhs_cow)) = (
        conv_real_elements_as_f64(lhs),
        conv_real_elements_as_f64(rhs),
    ) else {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "grouped conv requires real-float or complex operands".into(),
        });
    };
    let lhs_src: &[f64] = &lhs_cow;
    let rhs_src: &[f64] = &rhs_cow;

    // Depthwise fast path (feature_group_count == c_in, multiplier 1: rhs_c_in == 1,
    // cout_per_group == 1, c_out == c_in) — the 1D sibling of eval_conv_2d_grouped's
    // fast path. Output channel co reads input channel co and kernel[k,0,co], BOTH
    // contiguous over co, so accumulate the whole channel vector per output position
    // (contiguous inner loop autovectorizes; each input position read ONCE) instead of
    // the general path's per-co channel-strided re-read. Per-channel k-ascending order
    // and the 0.0*rhs OOB contribution (NaN for inf kernel) are preserved -> BIT-FOR-BIT
    // identical. Verified by conv1d_depthwise_fast_path_matches_general.
    if rhs_c_in == 1 && cout_per_group == 1 {
        let mut out = vec![0.0_f64; total];
        let mut oi = 0usize;
        for n in 0..batch {
            let n_off = n * width_c_in;
            for w in 0..out_w {
                let acc = &mut out[oi..oi + c_out];
                for k in 0..kernel_w {
                    let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                    let rbase = k * rhs_c_in_c_out;
                    let rhs_row = &rhs_src[rbase..rbase + c_out];
                    if in_pos < 0 || (in_pos as usize) >= width {
                        #[allow(
                            clippy::erasing_op,
                            reason = "replicate the general loop's 0.0*rhs for bit-identity (inf kernel -> NaN)"
                        )]
                        for (a, &r) in acc.iter_mut().zip(rhs_row) {
                            *a += 0.0 * r;
                        }
                    } else {
                        let base = n_off + (in_pos as usize) * c_in;
                        let lhs_row = &lhs_src[base..base + c_out];
                        for ((a, &l), &r) in acc.iter_mut().zip(lhs_row).zip(rhs_row) {
                            *a += l * r;
                        }
                    }
                }
                oi += c_out;
            }
        }
        return conv_real_output_from_f64(out_dtype, out_dims, out);
    }

    // General grouped fast path (cout_per_group > 1; the depthwise multiplier-1 case
    // returned above) — the 1D sibling of eval_conv_2d_grouped's AXPY path. For a fixed
    // tap (k,ci) and group g, every output channel in the group shares the same input
    // value lhs[…,g·rhs_c_in+ci] and reads CONTIGUOUS kernel weights rhs[k,ci, g·cpg..],
    // so the inner work is an AXPY over the group's output channels (autovectorizes; lhs
    // read once per group). Per-channel (k/ci) order + 0.0·rhs OOB preserved -> BIT-FOR-BIT
    // identical. Verified by conv1d_grouped_axpy_matches_general.
    if cout_per_group > 1 {
        let mut out = vec![0.0_f64; total];
        let mut spatial_base = 0usize;
        for n in 0..batch {
            let n_off = n * width_c_in;
            for w in 0..out_w {
                let acc = &mut out[spatial_base..spatial_base + c_out];
                for k in 0..kernel_w {
                    let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                    let oob = in_pos < 0 || (in_pos as usize) >= width;
                    let lhs_base = if oob {
                        0
                    } else {
                        n_off + (in_pos as usize) * c_in
                    };
                    let rhs_k = k * rhs_c_in_c_out;
                    for ci in 0..rhs_c_in {
                        let rhs_ci = rhs_k + ci * c_out;
                        for g in 0..group_count {
                            let lhs_val = if oob {
                                0.0
                            } else {
                                lhs_src[lhs_base + g * rhs_c_in + ci]
                            };
                            let co_base = g * cout_per_group;
                            let rhs_row =
                                &rhs_src[rhs_ci + co_base..rhs_ci + co_base + cout_per_group];
                            let acc_g = &mut acc[co_base..co_base + cout_per_group];
                            for (a, &r) in acc_g.iter_mut().zip(rhs_row) {
                                *a += lhs_val * r;
                            }
                        }
                    }
                }
                spatial_base += c_out;
            }
        }
        return conv_real_output_from_f64(out_dtype, out_dims, out);
    }

    let mut out = Vec::with_capacity(total);
    for n in 0..batch {
        let n_off = n * width_c_in;
        for w in 0..out_w {
            for co in 0..c_out {
                let in_ch_base = (co / cout_per_group) * rhs_c_in;
                let mut acc = 0.0_f64;
                for k in 0..kernel_w {
                    let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                    let oob = in_pos < 0 || (in_pos as usize) >= width;
                    for ci in 0..rhs_c_in {
                        let rhs_idx = k * rhs_c_in_c_out + ci * c_out + co;
                        let lhs_val = if oob {
                            0.0
                        } else {
                            lhs_src[n_off + (in_pos as usize) * c_in + in_ch_base + ci]
                        };
                        acc += lhs_val * rhs_src[rhs_idx];
                    }
                }
                out.push(acc);
            }
        }
    }
    conv_real_output_from_f64(out_dtype, out_dims, out)
}

fn eval_conv_1d(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    params: &BTreeMap<String, String>,
    padding: ConvPadding,
) -> Result<Value, EvalError> {
    if rhs.shape.rank() != 3 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "1D conv kernel must have rank 3 [K, C_in, C_out], got rank {}",
                rhs.shape.rank()
            ),
        });
    }

    let batch = lhs.shape.dims[0] as usize;
    let width = lhs.shape.dims[1] as usize;
    let c_in = lhs.shape.dims[2] as usize;

    let kernel_w = rhs.shape.dims[0] as usize;
    let rhs_c_in = rhs.shape.dims[1] as usize;
    let c_out = rhs.shape.dims[2] as usize;

    // feature_group_count (grouped / depthwise conv): Cin splits into G groups; the
    // kernel's c_in dim is Cin/G, and output channel co belongs to group co/(Cout/G).
    // G==1 is ordinary conv (rhs_c_in == c_in).
    let group_count = parse_conv_group_count(primitive, params)?;
    if c_in != group_count * rhs_c_in {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "channel mismatch: lhs c_in={c_in} != feature_group_count={group_count} * rhs c_in={rhs_c_in}"
            ),
        });
    }
    if !c_out.is_multiple_of(group_count) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "conv c_out={c_out} must be divisible by feature_group_count={group_count}"
            ),
        });
    }

    let stride = parse_positive_stride(primitive, params.get("strides").map(String::as_str))?;
    // rhs_dilation (atrous): a dilation `d` spaces kernel taps `d` apart, so the
    // kernel's effective extent is `(k-1)*d+1` (used for the output size and SAME-pad
    // geometry) and tap `k` reads input position `w*stride + k*d - pad`.
    let dil = parse_conv_1d_dilation(primitive, params)?;
    let eff_kernel_w = (kernel_w - 1) * dil + 1;

    let (out_w, pad_left) = if width == 0 {
        // Empty input produces empty output with no padding
        (0, 0)
    } else {
        match padding {
            ConvPadding::Same | ConvPadding::SameLower => {
                let out_w = width.div_ceil(stride);
                // out_w >= 1 since width >= 1 and stride >= 1
                let pad_total = ((out_w - 1) * stride + eff_kernel_w).saturating_sub(width);
                let pad_low = if padding == ConvPadding::SameLower {
                    pad_total.div_ceil(2)
                } else {
                    pad_total / 2
                };
                (out_w, pad_low)
            }
            ConvPadding::Valid => (conv_valid_output_dim(width, eff_kernel_w, stride), 0),
        }
    };

    // Grouped/depthwise 1D conv (G>1): each group is an independent conv over its own
    // Cin/G input and Cout/G output channels. Dedicated direct path; the G==1 fast
    // paths below are left untouched.
    if group_count > 1 {
        return eval_conv_1d_grouped(
            primitive,
            lhs,
            rhs,
            group_count,
            batch,
            width,
            c_in,
            kernel_w,
            rhs_c_in,
            c_out,
            stride,
            dil,
            out_w,
            pad_left,
        );
    }

    let out_dtype = promote_dtype(lhs.dtype, rhs.dtype);
    let total = batch
        .checked_mul(out_w)
        .and_then(|v| v.checked_mul(c_out))
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "conv output size overflow".into(),
        })?;
    let mut elements = Vec::with_capacity(total);

    let width_c_in = width
        .checked_mul(c_in)
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "conv lhs stride overflow".into(),
        })?;
    let c_in_c_out = c_in
        .checked_mul(c_out)
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "conv rhs stride overflow".into(),
        })?;
    // Pre-check that kernel_w * c_in_c_out won't overflow (used in inner loop: k * c_in_c_out)
    kernel_w
        .checked_mul(c_in_c_out)
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "conv rhs kernel stride overflow".into(),
        })?;

    let is_complex = matches!(out_dtype, DType::Complex64 | DType::Complex128);

    // NATIVE f32-accum 1D conv path (XLA parity, the conv1d sibling of eval_conv_2d's
    // native path): XLA accumulates f32/bf16/f16 conv in f32, NOT f64 — fj's f64-promote
    // path was MORE precise than the reference. When the output is f32/bf16/f16 and both
    // operands share that dtype, decode operands to f32 (bf16/f16 widen EXACTLY, never
    // via f64), build the f32 im2col, run the native-f32 GEMM (which threads internally),
    // then round to out_dtype. Always uses the im2col GEMM (correct + threaded for every
    // size; the alloc is negligible for tiny convs). BIT-IDENTICAL to the scalar f32-accum
    // ascending-(k,ci) fold (zero-padded taps add 0·w; the GEMM matches the scalar f32
    // fold per the cz0g0 contract).
    if matches!(out_dtype, DType::F32 | DType::BF16 | DType::F16)
        && lhs.dtype == out_dtype
        && rhs.dtype == out_dtype
        && let (Some(lhs_f32), Some(rhs_f32)) = (conv_decode_to_f32(lhs), conv_decode_to_f32(rhs))
    {
        let kdim = kernel_w
            .checked_mul(c_in)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "conv1d im2col kdim overflow".into(),
            })?;
        let num_rows = total.checked_div(c_out).unwrap_or(0);
        let col_len = num_rows
            .checked_mul(kdim)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "conv1d im2col size overflow".into(),
            })?;
        let mut col = vec![0.0_f32; col_len];
        for n in 0..batch {
            let n_offset = n * width_c_in;
            for w in 0..out_w {
                let row_base = (n * out_w + w) * kdim;
                for k in 0..kernel_w {
                    let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                    if in_pos < 0 || (in_pos as usize) >= width {
                        continue;
                    }
                    let src_base = n_offset + (in_pos as usize) * c_in;
                    let col_base = row_base + k * c_in;
                    col[col_base..col_base + c_in]
                        .copy_from_slice(&lhs_f32[src_base..src_base + c_in]);
                }
            }
        }
        let out = batched_matmul_2d_f32_in(&col, 1, num_rows, kdim, rhs_f32.as_ref(), c_out);
        return conv_f32_output_to_value(
            out_dtype,
            vec![batch as u32, out_w as u32, c_out as u32],
            out,
        );
    }

    // Dense F64 fast path: read both operands straight from their contiguous f64
    // backings, bypassing the per-multiply Literal materialization + match in the
    // innermost conv loop. Bit-identical to the generic non-complex path — same
    // index math, same ascending k/ci accumulation order, same `*`/`+`, and the
    // same from_f64 output (out_dtype == F64; for dense f64,
    // src[idx] == as_f64().unwrap_or(0.0)). Large output spaces are split into
    // independent output morsels; each output element still performs its own
    // inner reduction in the same serial k/ci order. Reached by F64 outputs and
    // MIXED-dtype convs (same-dtype f32/bf16/f16 took the native path above).
    if !is_complex
        && matches!(
            out_dtype,
            DType::F64 | DType::F32 | DType::BF16 | DType::F16
        )
        && let (Some(lhs_cow), Some(rhs_cow)) = (
            conv_real_elements_as_f64(lhs),
            conv_real_elements_as_f64(rhs),
        )
    {
        // Re-borrow as &[f64] so the threaded morsel closures below can capture a
        // (Copy) slice reference rather than moving the Cow. The Cow owns the
        // (possibly promoted) data for the lifetime of this block.
        let lhs_src: &[f64] = &lhs_cow;
        let rhs_src: &[f64] = &rhs_cow;
        if batch > 0 {
            (batch - 1)
                .checked_mul(width_c_in)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "conv batch index overflow".into(),
                })?;
        }

        let conv_ops = total.saturating_mul(kernel_w).saturating_mul(c_in);

        // im2col + GEMM fast path (mirrors eval_conv_2d). The kernel rhs_src is
        // laid out [K,Cin,Cout] row-major == the [(K·Cin) × Cout] GEMM matrix;
        // gathering each output position's receptive field (zero-filled for
        // out-of-bounds/padding) into a row turns the conv into one cache-blocked,
        // auto-threaded matmul_2d. Bit-for-bit identical to the direct loop for
        // finite inputs: same ascending (k,ci) accumulation order, and the
        // zero-padded out-of-bounds taps add 0.0 (a no-op on a finite partial
        // sum, exactly as the direct loop's bounds check skips them). The GEMM
        // also vectorizes over Cout, which the direct scalar accumulate cannot.
        let kdim = kernel_w * c_in;
        if conv_ops >= CONV_IM2COL_MIN_OPS && kdim > 0 {
            let num_rows = total / c_out;
            let mut col = vec![0.0_f64; num_rows * kdim];
            for n in 0..batch {
                let n_offset = n * width_c_in;
                for w in 0..out_w {
                    let row_base = (n * out_w + w) * kdim;
                    for k in 0..kernel_w {
                        let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                        if in_pos < 0 || (in_pos as usize) >= width {
                            continue;
                        }
                        let src_base = n_offset + (in_pos as usize) * c_in;
                        let col_base = row_base + k * c_in;
                        col[col_base..col_base + c_in]
                            .copy_from_slice(&lhs_src[src_base..src_base + c_in]);
                    }
                }
            }
            let out = matmul_2d(&col, num_rows, kdim, rhs_src, c_out);
            return conv_real_output_from_f64(
                out_dtype,
                vec![batch as u32, out_w as u32, c_out as u32],
                out,
            );
        }

        let threads = conv_morsel_threads(total, conv_ops);
        let mut out = if threads > 1 {
            let mut out = vec![0.0_f64; total];
            let chunk = total.div_ceil(threads);
            std::thread::scope(|scope| {
                let mut rest = out.as_mut_slice();
                let mut start = 0usize;
                while start < total {
                    let len = chunk.min(total - start);
                    let (block, tail) = rest.split_at_mut(len);
                    rest = tail;
                    let block_start = start;
                    scope.spawn(move || {
                        for (offset, slot) in block.iter_mut().enumerate() {
                            let flat = block_start + offset;
                            let co = flat % c_out;
                            let t = flat / c_out;
                            let w = t % out_w;
                            let n = t / out_w;
                            let n_offset = n * width_c_in;

                            let mut acc = 0.0_f64;
                            for k in 0..kernel_w {
                                let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                                let oob = in_pos < 0 || (in_pos as usize) >= width;
                                let lhs_base = if oob {
                                    0
                                } else {
                                    n_offset + (in_pos as usize) * c_in
                                };
                                let rhs_base = k * c_in_c_out + co;
                                // Zero-padded (OOB) taps add 0·w, matching XLA zero-
                                // padding and conv2d; a no-op for finite data, fixes
                                // signed-zero parity vs the old in-bounds-only skip.
                                for ci in 0..c_in {
                                    let lhs_val = if oob { 0.0 } else { lhs_src[lhs_base + ci] };
                                    acc += lhs_val * rhs_src[rhs_base + ci * c_out];
                                }
                            }
                            *slot = acc;
                        }
                    });
                    start += len;
                }
            });
            out
        } else {
            Vec::with_capacity(total)
        };

        if threads <= 1 {
            for n in 0..batch {
                let n_offset = n * width_c_in;
                for w in 0..out_w {
                    for co in 0..c_out {
                        let mut acc = 0.0_f64;
                        for k in 0..kernel_w {
                            let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                            let oob = in_pos < 0 || (in_pos as usize) >= width;
                            for ci in 0..c_in {
                                let rhs_idx = k * c_in_c_out + ci * c_out + co;
                                // Zero-padded (OOB) taps add 0·w (XLA zero-padding).
                                let lhs_val = if oob {
                                    0.0
                                } else {
                                    lhs_src[n_offset + (in_pos as usize) * c_in + ci]
                                };
                                acc += lhs_val * rhs_src[rhs_idx];
                            }
                        }
                        out.push(acc);
                    }
                }
            }
        }

        return conv_real_output_from_f64(
            out_dtype,
            vec![batch as u32, out_w as u32, c_out as u32],
            out,
        );
    }

    for n in 0..batch {
        let n_offset = n
            .checked_mul(width_c_in)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "conv batch index overflow".into(),
            })?;
        for w in 0..out_w {
            for co in 0..c_out {
                if is_complex {
                    let mut acc_re = 0.0_f64;
                    let mut acc_im = 0.0_f64;
                    for k in 0..kernel_w {
                        let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                        let oob = in_pos < 0 || (in_pos as usize) >= width;
                        for ci in 0..c_in {
                            let rhs_idx = k * c_in_c_out + ci * c_out + co;
                            // Zero-padded (OOB) taps contribute 0·w, matching XLA's
                            // zero-padding; a no-op for finite data, but fixes signed-
                            // zero / non-finite parity vs the old in-bounds-only skip.
                            let (lhs_re, lhs_im) = if oob {
                                (0.0, 0.0)
                            } else {
                                let lhs_idx = n_offset + (in_pos as usize) * c_in + ci;
                                literal_as_complex(&lhs.elements[lhs_idx])
                            };
                            let (rhs_re, rhs_im) = literal_as_complex(&rhs.elements[rhs_idx]);
                            // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            acc_re += lhs_re * rhs_re - lhs_im * rhs_im;
                            acc_im += lhs_re * rhs_im + lhs_im * rhs_re;
                        }
                    }
                    elements.push(conv_literal_from_complex(out_dtype, acc_re, acc_im));
                } else {
                    let mut acc = 0.0_f64;
                    for k in 0..kernel_w {
                        let in_pos = (w * stride + k * dil) as isize - pad_left as isize;
                        let oob = in_pos < 0 || (in_pos as usize) >= width;
                        for ci in 0..c_in {
                            let rhs_idx = k * c_in_c_out + ci * c_out + co;
                            // Zero-padded (OOB) taps add 0·w (XLA zero-padding); a
                            // no-op for finite data, fixes signed-zero parity.
                            let lhs_val = if oob {
                                0.0
                            } else {
                                let lhs_idx = n_offset + (in_pos as usize) * c_in + ci;
                                lhs.elements[lhs_idx].as_f64().unwrap_or(0.0)
                            };
                            let rhs_val = rhs.elements[rhs_idx].as_f64().unwrap_or(0.0);
                            acc += lhs_val * rhs_val;
                        }
                    }
                    elements.push(conv_float_literal_from_f64(out_dtype, acc));
                }
            }
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        out_dtype,
        Shape {
            dims: vec![batch as u32, out_w as u32, c_out as u32],
        },
        elements,
    )?))
}

/// 2D convolution: lhs=[N, H, W, C_in], rhs=[KH, KW, C_in, C_out]
/// Minimum convolution FMA count (output_elems × KH·KW·Cin) above which the
/// dense F64 conv2d uses the im2col + GEMM path instead of the direct loop.
/// Below it the im2col buffer allocation + copy isn't worth it.
const CONV_IM2COL_MIN_OPS: usize = 1 << 16;

#[allow(
    clippy::too_many_arguments,
    reason = "conv2d im2col geometry is explicit for bounds/audit"
)]
fn fill_conv2d_im2col<T>(
    col: &mut [T],
    lhs_src: &[T],
    batch: usize,
    height: usize,
    width: usize,
    c_in: usize,
    out_h: usize,
    out_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    dil_h: usize,
    dil_w: usize,
    pad_top: usize,
    pad_left: usize,
    height_width_c_in: usize,
    width_c_in: usize,
    kdim: usize,
    kw_c_in: usize,
    threads: usize,
) where
    T: Copy + Send + Sync,
{
    let rows = batch * out_h * out_w;
    debug_assert_eq!(col.len(), rows * kdim);
    if rows == 0 || kdim == 0 {
        return;
    }

    let fill_rows = |row_start: usize, block: &mut [T]| {
        for (local_row, row_slice) in block.chunks_exact_mut(kdim).enumerate() {
            let row = row_start + local_row;
            let n = row / (out_h * out_w);
            let spatial = row % (out_h * out_w);
            let oh = spatial / out_w;
            let ow = spatial % out_w;
            let n_offset = n * height_width_c_in;
            for kh in 0..kernel_h {
                let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                if in_h < 0 || (in_h as usize) >= height {
                    continue;
                }
                let h_offset = (in_h as usize) * width_c_in;
                for kw in 0..kernel_w {
                    let in_w = (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                    if in_w < 0 || (in_w as usize) >= width {
                        continue;
                    }
                    let src_base = n_offset + h_offset + (in_w as usize) * c_in;
                    let col_base = kh * kw_c_in + kw * c_in;
                    row_slice[col_base..col_base + c_in]
                        .copy_from_slice(&lhs_src[src_base..src_base + c_in]);
                }
            }
        }
    };

    let workers = threads.min(rows);
    if workers <= 1 {
        fill_rows(0, col);
        return;
    }

    let rows_per = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut rest = col;
        let mut row_start = 0usize;
        while row_start < rows {
            let chunk_rows = rows_per.min(rows - row_start);
            let chunk_len = chunk_rows * kdim;
            let (block, tail) = rest.split_at_mut(chunk_len);
            rest = tail;
            let start = row_start;
            let fill = &fill_rows;
            scope.spawn(move || fill(start, block));
            row_start += chunk_rows;
        }
    });
}

/// Grouped/depthwise 2D conv (feature_group_count > 1). Each output channel `co`
/// belongs to group `g = co / (Cout/G)` and convolves only that group's `Cin/G`
/// input channels (`lhs` channels `[g*rhs_c_in .. (g+1)*rhs_c_in)`) with the kernel
/// slice `rhs[:,:,:,co]` (kernel c_in dim is `rhs_c_in == Cin/G`). Direct loop
/// (real f64-accum or complex), same ascending (kh,kw,ci) order and zero-padding as
/// the ungrouped path — so it equals a per-group ungrouped conv concatenated over
/// output channels (proved in conv2d_feature_group_count_matches_ungrouped_groups).
#[allow(clippy::too_many_arguments)]
fn eval_conv_2d_grouped(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    group_count: usize,
    batch: usize,
    height: usize,
    width: usize,
    c_in: usize,
    kernel_h: usize,
    kernel_w: usize,
    rhs_c_in: usize,
    c_out: usize,
    stride_h: usize,
    stride_w: usize,
    dil_h: usize,
    dil_w: usize,
    out_h: usize,
    out_w: usize,
    pad_top: usize,
    pad_left: usize,
) -> Result<Value, EvalError> {
    let out_dtype = promote_dtype(lhs.dtype, rhs.dtype);
    let cout_per_group = c_out / group_count;
    let total = batch
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(c_out))
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "grouped conv output size overflow".into(),
        })?;
    let width_c_in = width * c_in;
    let height_width_c_in = height * width_c_in;
    // Kernel layout [KH, KW, rhs_c_in, Cout].
    let rhs_c_in_c_out = rhs_c_in * c_out;
    let kw_rhs_c_in_c_out = kernel_w * rhs_c_in_c_out;
    let out_dims = vec![batch as u32, out_h as u32, out_w as u32, c_out as u32];

    // Native f32 accumulation for grouped/depthwise f32/bf16/f16 conv2d. XLA
    // accumulates these dtypes in f32; the generic real path below widens to f64
    // and then rounds back, making grouped/depthwise conv more precise than the reference.
    // The loop nests mirror the f64 fast paths exactly, preserving output order and
    // per-output ascending (kh, kw, ci) accumulation. OOB taps still add 0.0*rhs in
    // f32, preserving the scalar f32 reference behavior for infinities.
    if matches!(out_dtype, DType::F32 | DType::BF16 | DType::F16)
        && lhs.dtype == out_dtype
        && rhs.dtype == out_dtype
        && let (Some(lhs_cow), Some(rhs_cow)) = (conv_decode_to_f32(lhs), conv_decode_to_f32(rhs))
    {
        let lhs_src: &[f32] = &lhs_cow;
        let rhs_src: &[f32] = &rhs_cow;

        if rhs_c_in == 1 && cout_per_group == 1 {
            let mut out = vec![0.0_f32; total];
            let mut oi = 0usize;
            for n in 0..batch {
                let n_off = n * height_width_c_in;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let acc = &mut out[oi..oi + c_out];
                        for kh in 0..kernel_h {
                            let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                            let h_oob = in_h < 0 || (in_h as usize) >= height;
                            let h_off = if h_oob {
                                0
                            } else {
                                (in_h as usize) * width_c_in
                            };
                            let rhs_kh = kh * kw_rhs_c_in_c_out;
                            for kw in 0..kernel_w {
                                let in_w =
                                    (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                                let w_oob = in_w < 0 || (in_w as usize) >= width;
                                let rbase = rhs_kh + kw * rhs_c_in_c_out;
                                let rhs_row = &rhs_src[rbase..rbase + c_out];
                                if h_oob || w_oob {
                                    #[allow(
                                        clippy::erasing_op,
                                        reason = "preserve scalar f32 conv OOB 0.0*rhs behavior"
                                    )]
                                    for (a, &r) in acc.iter_mut().zip(rhs_row) {
                                        *a += 0.0_f32 * r;
                                    }
                                } else {
                                    let base = n_off + h_off + (in_w as usize) * c_in;
                                    let lhs_row = &lhs_src[base..base + c_out];
                                    for ((a, &l), &r) in acc.iter_mut().zip(lhs_row).zip(rhs_row) {
                                        *a += l * r;
                                    }
                                }
                            }
                        }
                        oi += c_out;
                    }
                }
            }
            return conv_f32_output_to_value(out_dtype, out_dims, out);
        }

        if cout_per_group > 1 {
            let mut out = vec![0.0_f32; total];
            let mut spatial_base = 0usize;
            for n in 0..batch {
                let n_off = n * height_width_c_in;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let acc = &mut out[spatial_base..spatial_base + c_out];
                        for kh in 0..kernel_h {
                            let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                            let h_oob = in_h < 0 || (in_h as usize) >= height;
                            let h_off = if h_oob {
                                0
                            } else {
                                (in_h as usize) * width_c_in
                            };
                            let rhs_kh = kh * kw_rhs_c_in_c_out;
                            for kw in 0..kernel_w {
                                let in_w =
                                    (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                                let w_oob = in_w < 0 || (in_w as usize) >= width;
                                let oob = h_oob || w_oob;
                                let in_w_off = if w_oob { 0 } else { (in_w as usize) * c_in };
                                let lhs_base = n_off + h_off + in_w_off;
                                let rhs_kw = rhs_kh + kw * rhs_c_in_c_out;
                                for ci in 0..rhs_c_in {
                                    let rhs_ci = rhs_kw + ci * c_out;
                                    for g in 0..group_count {
                                        let lhs_val = if oob {
                                            0.0
                                        } else {
                                            lhs_src[lhs_base + g * rhs_c_in + ci]
                                        };
                                        let co_base = g * cout_per_group;
                                        let rhs_row = &rhs_src
                                            [rhs_ci + co_base..rhs_ci + co_base + cout_per_group];
                                        let acc_g = &mut acc[co_base..co_base + cout_per_group];
                                        for (a, &r) in acc_g.iter_mut().zip(rhs_row) {
                                            *a += lhs_val * r;
                                        }
                                    }
                                }
                            }
                        }
                        spatial_base += c_out;
                    }
                }
            }
            return conv_f32_output_to_value(out_dtype, out_dims, out);
        }

        let mut out = Vec::with_capacity(total);
        for n in 0..batch {
            let n_off = n * height_width_c_in;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for co in 0..c_out {
                        let in_ch_base = (co / cout_per_group) * rhs_c_in;
                        let mut acc = 0.0_f32;
                        for kh in 0..kernel_h {
                            let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                            let h_oob = in_h < 0 || (in_h as usize) >= height;
                            let h_off = if h_oob {
                                0
                            } else {
                                (in_h as usize) * width_c_in
                            };
                            for kw in 0..kernel_w {
                                let in_w =
                                    (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                                let w_oob = in_w < 0 || (in_w as usize) >= width;
                                let oob = h_oob || w_oob;
                                let in_w_off = if w_oob { 0 } else { (in_w as usize) * c_in };
                                for ci in 0..rhs_c_in {
                                    let rhs_idx = kh * kw_rhs_c_in_c_out
                                        + kw * rhs_c_in_c_out
                                        + ci * c_out
                                        + co;
                                    let lhs_val = if oob {
                                        0.0
                                    } else {
                                        lhs_src[n_off + h_off + in_w_off + in_ch_base + ci]
                                    };
                                    acc += lhs_val * rhs_src[rhs_idx];
                                }
                            }
                        }
                        out.push(acc);
                    }
                }
            }
        }
        return conv_f32_output_to_value(out_dtype, out_dims, out);
    }

    if matches!(out_dtype, DType::Complex64 | DType::Complex128) {
        let mut elements = Vec::with_capacity(total);
        for n in 0..batch {
            let n_off = n * height_width_c_in;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for co in 0..c_out {
                        let in_ch_base = (co / cout_per_group) * rhs_c_in;
                        let (mut acc_re, mut acc_im) = (0.0_f64, 0.0_f64);
                        for kh in 0..kernel_h {
                            let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                            let h_oob = in_h < 0 || (in_h as usize) >= height;
                            let h_off = if h_oob {
                                0
                            } else {
                                (in_h as usize) * width_c_in
                            };
                            for kw in 0..kernel_w {
                                let in_w =
                                    (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                                let w_oob = in_w < 0 || (in_w as usize) >= width;
                                let oob = h_oob || w_oob;
                                let in_w_off = if w_oob { 0 } else { (in_w as usize) * c_in };
                                for ci in 0..rhs_c_in {
                                    let rhs_idx = kh * kw_rhs_c_in_c_out
                                        + kw * rhs_c_in_c_out
                                        + ci * c_out
                                        + co;
                                    let (lre, lim) = if oob {
                                        (0.0, 0.0)
                                    } else {
                                        literal_as_complex(
                                            &lhs.elements
                                                [n_off + h_off + in_w_off + in_ch_base + ci],
                                        )
                                    };
                                    let (rre, rim) = literal_as_complex(&rhs.elements[rhs_idx]);
                                    acc_re += lre * rre - lim * rim;
                                    acc_im += lre * rim + lim * rre;
                                }
                            }
                        }
                        elements.push(conv_literal_from_complex(out_dtype, acc_re, acc_im));
                    }
                }
            }
        }
        return Ok(Value::Tensor(TensorValue::new(
            out_dtype,
            Shape { dims: out_dims },
            elements,
        )?));
    }

    let (Some(lhs_cow), Some(rhs_cow)) = (
        conv_real_elements_as_f64(lhs),
        conv_real_elements_as_f64(rhs),
    ) else {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "grouped conv requires real-float or complex operands".into(),
        });
    };
    let lhs_src: &[f64] = &lhs_cow;
    let rhs_src: &[f64] = &rhs_cow;

    // Depthwise fast path (feature_group_count == c_in, multiplier 1: rhs_c_in == 1 and
    // cout_per_group == 1, so c_out == c_in). Output channel `co` reads input channel `co`
    // and kernel `[kh,kw,0,co]`, BOTH contiguous over `co`. So accumulate the whole
    // channel vector per spatial position with a contiguous inner loop (the compiler
    // autovectorizes it, and each input position is read ONCE) instead of the general
    // path's per-`co` channel-strided re-read of the window. Per-channel accumulation
    // order (kh outer, kw inner) and the `0.0 * rhs` contribution for OOB taps (which
    // yields NaN for an inf kernel value, exactly as the general loop does) are
    // preserved, so the output is BIT-FOR-BIT identical. Verified by
    // conv2d_depthwise_fast_path_matches_general.
    if rhs_c_in == 1 && cout_per_group == 1 {
        let mut out = vec![0.0_f64; total];
        let mut oi = 0usize;
        for n in 0..batch {
            let n_off = n * height_width_c_in;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let acc = &mut out[oi..oi + c_out];
                    for kh in 0..kernel_h {
                        let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                        let h_oob = in_h < 0 || (in_h as usize) >= height;
                        let h_off = if h_oob {
                            0
                        } else {
                            (in_h as usize) * width_c_in
                        };
                        let rhs_kh = kh * kw_rhs_c_in_c_out;
                        for kw in 0..kernel_w {
                            let in_w = (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                            let w_oob = in_w < 0 || (in_w as usize) >= width;
                            let rbase = rhs_kh + kw * rhs_c_in_c_out;
                            let rhs_row = &rhs_src[rbase..rbase + c_out];
                            if h_oob || w_oob {
                                // OOB tap: `0.0 * rhs` per channel — identical to the
                                // general loop (yields NaN for an inf kernel value).
                                #[allow(
                                    clippy::erasing_op,
                                    reason = "must replicate the general loop's 0.0*rhs for bit-identity (inf kernel -> NaN)"
                                )]
                                for (a, &r) in acc.iter_mut().zip(rhs_row) {
                                    *a += 0.0 * r;
                                }
                            } else {
                                let base = n_off + h_off + (in_w as usize) * c_in;
                                let lhs_row = &lhs_src[base..base + c_out];
                                for ((a, &l), &r) in acc.iter_mut().zip(lhs_row).zip(rhs_row) {
                                    *a += l * r;
                                }
                            }
                        }
                    }
                    oi += c_out;
                }
            }
        }
        return conv_real_output_from_f64(out_dtype, out_dims, out);
    }

    // General grouped fast path (cout_per_group > 1, e.g. ResNeXt; the depthwise
    // multiplier-1 case returned above). For a fixed input tap (kh,kw,ci) and group g,
    // every output channel in that group shares the SAME input value lhs[…, g·rhs_c_in+ci]
    // and reads CONTIGUOUS kernel weights rhs[kh,kw,ci, g·cpg .. g·cpg+cpg], so the inner
    // work is an AXPY (scalar·vector + vector) over the group's output channels — the
    // compiler autovectorizes it, and each lhs value is read ONCE per group instead of
    // the general loop's per-output-channel channel-strided re-read. Per-channel
    // (kh-outer/kw/ci) accumulation order and the 0.0·rhs OOB contribution are preserved,
    // so it is BIT-FOR-BIT identical. Verified by conv2d_grouped_axpy_matches_general.
    if cout_per_group > 1 {
        let mut out = vec![0.0_f64; total];
        let mut spatial_base = 0usize;
        for n in 0..batch {
            let n_off = n * height_width_c_in;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let acc = &mut out[spatial_base..spatial_base + c_out];
                    for kh in 0..kernel_h {
                        let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                        let h_oob = in_h < 0 || (in_h as usize) >= height;
                        let h_off = if h_oob {
                            0
                        } else {
                            (in_h as usize) * width_c_in
                        };
                        let rhs_kh = kh * kw_rhs_c_in_c_out;
                        for kw in 0..kernel_w {
                            let in_w = (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                            let w_oob = in_w < 0 || (in_w as usize) >= width;
                            let oob = h_oob || w_oob;
                            let in_w_off = if w_oob { 0 } else { (in_w as usize) * c_in };
                            let lhs_base = n_off + h_off + in_w_off;
                            let rhs_kw = rhs_kh + kw * rhs_c_in_c_out;
                            for ci in 0..rhs_c_in {
                                let rhs_ci = rhs_kw + ci * c_out;
                                for g in 0..group_count {
                                    let lhs_val = if oob {
                                        0.0
                                    } else {
                                        lhs_src[lhs_base + g * rhs_c_in + ci]
                                    };
                                    let co_base = g * cout_per_group;
                                    let rhs_row = &rhs_src
                                        [rhs_ci + co_base..rhs_ci + co_base + cout_per_group];
                                    let acc_g = &mut acc[co_base..co_base + cout_per_group];
                                    for (a, &r) in acc_g.iter_mut().zip(rhs_row) {
                                        *a += lhs_val * r;
                                    }
                                }
                            }
                        }
                    }
                    spatial_base += c_out;
                }
            }
        }
        return conv_real_output_from_f64(out_dtype, out_dims, out);
    }

    let mut out = Vec::with_capacity(total);
    for n in 0..batch {
        let n_off = n * height_width_c_in;
        for oh in 0..out_h {
            for ow in 0..out_w {
                for co in 0..c_out {
                    let in_ch_base = (co / cout_per_group) * rhs_c_in;
                    let mut acc = 0.0_f64;
                    for kh in 0..kernel_h {
                        let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                        let h_oob = in_h < 0 || (in_h as usize) >= height;
                        let h_off = if h_oob {
                            0
                        } else {
                            (in_h as usize) * width_c_in
                        };
                        for kw in 0..kernel_w {
                            let in_w = (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                            let w_oob = in_w < 0 || (in_w as usize) >= width;
                            let oob = h_oob || w_oob;
                            let in_w_off = if w_oob { 0 } else { (in_w as usize) * c_in };
                            for ci in 0..rhs_c_in {
                                let rhs_idx =
                                    kh * kw_rhs_c_in_c_out + kw * rhs_c_in_c_out + ci * c_out + co;
                                let lhs_val = if oob {
                                    0.0
                                } else {
                                    lhs_src[n_off + h_off + in_w_off + in_ch_base + ci]
                                };
                                acc += lhs_val * rhs_src[rhs_idx];
                            }
                        }
                    }
                    out.push(acc);
                }
            }
        }
    }
    conv_real_output_from_f64(out_dtype, out_dims, out)
}

fn eval_conv_2d(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    params: &BTreeMap<String, String>,
    padding: ConvPadding,
) -> Result<Value, EvalError> {
    if rhs.shape.rank() != 4 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "2D conv kernel must have rank 4 [KH, KW, C_in, C_out], got rank {}",
                rhs.shape.rank()
            ),
        });
    }

    let batch = lhs.shape.dims[0] as usize;
    let height = lhs.shape.dims[1] as usize;
    let width = lhs.shape.dims[2] as usize;
    let c_in = lhs.shape.dims[3] as usize;

    let kernel_h = rhs.shape.dims[0] as usize;
    let kernel_w = rhs.shape.dims[1] as usize;
    let rhs_c_in = rhs.shape.dims[2] as usize;
    let c_out = rhs.shape.dims[3] as usize;

    // feature_group_count (grouped / depthwise conv): the input channels split into
    // G groups; the kernel's c_in dim is Cin/G, and output channel `co` belongs to
    // group `co / (Cout/G)`, convolving only its group's input channels. G==1 is
    // ordinary conv (rhs_c_in == c_in).
    let group_count = parse_conv_group_count(primitive, params)?;
    if c_in != group_count * rhs_c_in {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "channel mismatch: lhs c_in={c_in} != feature_group_count={group_count} * rhs c_in={rhs_c_in}"
            ),
        });
    }
    if !c_out.is_multiple_of(group_count) {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "conv c_out={c_out} must be divisible by feature_group_count={group_count}"
            ),
        });
    }

    // Parse strides: either single value or "h,w" pair
    let (stride_h, stride_w) = parse_stride_pair(primitive, params)?;
    // rhs_dilation (atrous): a dilation `d` spaces kernel taps `d` apart, so the
    // kernel's effective spatial extent is `(k-1)*d+1` (used for the output size and
    // SAME-padding geometry) and tap `kh` reads input row `oh*stride + kh*d - pad`.
    let (dil_h, dil_w) = parse_dilation_pair(primitive, params)?;
    let eff_kernel_h = (kernel_h - 1) * dil_h + 1;
    let eff_kernel_w = (kernel_w - 1) * dil_w + 1;

    let (out_h, pad_top) = compute_output_and_pad(height, eff_kernel_h, stride_h, padding);
    let (out_w, pad_left) = compute_output_and_pad(width, eff_kernel_w, stride_w, padding);

    // Grouped/depthwise conv (G>1): each group is an independent conv over its own
    // Cin/G input channels and Cout/G output channels. Routed to a dedicated direct
    // path (the G==1 im2col matrix would have the wrong shape); the G==1 fast paths
    // below are left untouched.
    if group_count > 1 {
        return eval_conv_2d_grouped(
            primitive,
            lhs,
            rhs,
            group_count,
            batch,
            height,
            width,
            c_in,
            kernel_h,
            kernel_w,
            rhs_c_in,
            c_out,
            stride_h,
            stride_w,
            dil_h,
            dil_w,
            out_h,
            out_w,
            pad_top,
            pad_left,
        );
    }

    let out_dtype = promote_dtype(lhs.dtype, rhs.dtype);
    let total = batch
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(c_out))
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "conv output size overflow".into(),
        })?;
    let mut elements = Vec::with_capacity(total);

    let width_c_in = width
        .checked_mul(c_in)
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "conv lhs width*c_in overflow".into(),
        })?;
    let height_width_c_in =
        height
            .checked_mul(width_c_in)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "conv lhs stride overflow".into(),
            })?;
    let c_in_c_out = c_in
        .checked_mul(c_out)
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "conv rhs c_in*c_out overflow".into(),
        })?;
    let kw_c_in_c_out = kernel_w
        .checked_mul(c_in_c_out)
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "conv rhs stride overflow".into(),
        })?;

    let is_complex = matches!(out_dtype, DType::Complex64 | DType::Complex128);

    // NATIVE f32-accum conv path (XLA parity: XLA accumulates f32/bf16/f16 convolution
    // in f32, NOT f64 — fj's f64-promote path was MORE precise than the reference, the
    // same gap fixed for f32/bf16/f16 matmul). When the output is f32/bf16/f16 and BOTH
    // operands share that dtype, decode operands to f32 (bf16/f16 widen EXACTLY, never
    // via f64) and accumulate the im2col GEMM / direct conv in f32 (16-lane native-f32
    // GEMM for the large case; ascending-(kh,kw,ci) f32 fold for the small case), then
    // round to out_dtype. Matches XLA and halves/quarters the im2col bytes vs f64.
    // Storage-independent. Bit-identical to the scalar f32-accum reference (same index
    // math + ascending order; the native-f32 GEMM matches the scalar f32 fold per the
    // cz0g0 contract). Verified by conv2d_{f32,bf16,f16}_*_matches_reference.
    if matches!(out_dtype, DType::F32 | DType::BF16 | DType::F16)
        && lhs.dtype == out_dtype
        && rhs.dtype == out_dtype
        && let (Some(lhs_f32), Some(rhs_f32)) = (conv_decode_to_f32(lhs), conv_decode_to_f32(rhs))
    {
        let kdim = kernel_h
            .checked_mul(kernel_w)
            .and_then(|v| v.checked_mul(c_in))
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "conv im2col kdim overflow".into(),
            })?;
        let conv_ops = total.saturating_mul(kdim);
        if conv_ops >= CONV_IM2COL_MIN_OPS && kdim > 0 {
            let kw_c_in = kernel_w * c_in;
            let num_rows = total / c_out;
            let mut col = vec![0.0_f32; num_rows * kdim];
            fill_conv2d_im2col(
                &mut col,
                lhs_f32.as_ref(),
                batch,
                height,
                width,
                c_in,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                dil_h,
                dil_w,
                pad_top,
                pad_left,
                height_width_c_in,
                width_c_in,
                kdim,
                kw_c_in,
                conv_morsel_threads(num_rows, conv_ops),
            );
            let out = batched_matmul_2d_f32_in(&col, 1, num_rows, kdim, rhs_f32.as_ref(), c_out);
            return conv_f32_output_to_value(
                out_dtype,
                vec![batch as u32, out_h as u32, out_w as u32, c_out as u32],
                out,
            );
        }

        // Small conv: direct ascending-(kh,kw,ci) f32 fold (below CONV_IM2COL_MIN_OPS,
        // where the im2col buffer is not worth allocating). Same f32 accumulation.
        let mut out = Vec::with_capacity(total);
        for n in 0..batch {
            let n_offset = n * height_width_c_in;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for co in 0..c_out {
                        let mut acc = 0.0_f32;
                        for kh in 0..kernel_h {
                            let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                            let h_oob = in_h < 0 || (in_h as usize) >= height;
                            let h_offset = if h_oob {
                                0
                            } else {
                                (in_h as usize) * width_c_in
                            };
                            for kw in 0..kernel_w {
                                let in_w =
                                    (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                                let w_oob = in_w < 0 || (in_w as usize) >= width;
                                let oob = h_oob || w_oob;
                                let in_w_off = if w_oob { 0 } else { (in_w as usize) * c_in };
                                for ci in 0..c_in {
                                    let rhs_idx =
                                        kh * kw_c_in_c_out + kw * c_in_c_out + ci * c_out + co;
                                    let lhs_val = if oob {
                                        0.0
                                    } else {
                                        lhs_f32[n_offset + h_offset + in_w_off + ci]
                                    };
                                    acc += lhs_val * rhs_f32[rhs_idx];
                                }
                            }
                        }
                        out.push(acc);
                    }
                }
            }
        }
        return conv_f32_output_to_value(
            out_dtype,
            vec![batch as u32, out_h as u32, out_w as u32, c_out as u32],
            out,
        );
    }

    // Dense real-float fast path (F64/F32/BF16/F16): read both operands as f64
    // (F64 borrowed; f32/bf16/f16 promoted losslessly), bypassing the per-multiply
    // Literal materialization + match in the innermost conv loop. Bit-identical to
    // the generic non-complex path — same index math, same ascending kh/kw/ci
    // accumulation order in f64, same `*`/`+`, same `conv_float_literal_from_f64`
    // rounding to out_dtype. Reached by F64 outputs and MIXED-dtype convs (same-dtype
    // f32/bf16/f16 took the native-f32-accum path above).
    if !is_complex
        && matches!(
            out_dtype,
            DType::F64 | DType::F32 | DType::BF16 | DType::F16
        )
        && let (Some(lhs_src), Some(rhs_src)) = (
            conv_real_elements_as_f64(lhs),
            conv_real_elements_as_f64(rhs),
        )
    {
        // im2col + GEMM fast path. The kernel `rhs_src` is laid out
        // [KH,KW,Cin,Cout] row-major, which is exactly the [(KH·KW·Cin) × Cout]
        // matrix the GEMM needs. Gathering each output position's receptive
        // field into a row of an im2col matrix (KH·KW·Cin wide, zero-filled for
        // out-of-bounds/padding) turns the convolution into one cache-blocked,
        // auto-threaded `matmul_2d`. Bit-for-bit identical to the direct loop
        // below for finite inputs: the GEMM accumulates each output in ascending
        // (kh,kw,ci) order — the same order — and the zero-padded out-of-bounds
        // taps add `0.0` (a no-op on a finite partial sum, exactly as the direct
        // loop's `continue` skips them). The GEMM also vectorizes over Cout,
        // which the direct scalar-accumulate cannot.
        let kdim = kernel_h
            .checked_mul(kernel_w)
            .and_then(|v| v.checked_mul(c_in))
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "conv im2col kdim overflow".into(),
            })?;
        let conv_ops = total.saturating_mul(kdim);
        if conv_ops >= CONV_IM2COL_MIN_OPS && kdim > 0 {
            let kw_c_in = kernel_w * c_in;
            let num_rows = total / c_out;
            let mut col = vec![0.0_f64; num_rows * kdim];
            fill_conv2d_im2col(
                &mut col,
                lhs_src.as_ref(),
                batch,
                height,
                width,
                c_in,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                dil_h,
                dil_w,
                pad_top,
                pad_left,
                height_width_c_in,
                width_c_in,
                kdim,
                kw_c_in,
                conv_morsel_threads(num_rows, conv_ops),
            );
            let out = matmul_2d(&col, num_rows, kdim, rhs_src.as_ref(), c_out);
            return conv_real_output_from_f64(
                out_dtype,
                vec![batch as u32, out_h as u32, out_w as u32, c_out as u32],
                out,
            );
        }

        let mut out = Vec::with_capacity(total);
        for n in 0..batch {
            let n_offset =
                n.checked_mul(height_width_c_in)
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "conv batch index overflow".into(),
                    })?;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for co in 0..c_out {
                        let mut acc = 0.0_f64;
                        for kh in 0..kernel_h {
                            let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                            let h_oob = in_h < 0 || (in_h as usize) >= height;
                            let h_offset = if h_oob {
                                0
                            } else {
                                (in_h as usize) * width_c_in
                            };
                            for kw in 0..kernel_w {
                                let in_w =
                                    (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                                let w_oob = in_w < 0 || (in_w as usize) >= width;
                                let oob = h_oob || w_oob;
                                let in_w_off = if w_oob { 0 } else { (in_w as usize) * c_in };
                                for ci in 0..c_in {
                                    let rhs_idx =
                                        kh * kw_c_in_c_out + kw * c_in_c_out + ci * c_out + co;
                                    // Zero-padded taps add `0·w` (matching the im2col
                                    // GEMM and XLA), not a skip — keeps the small-conv
                                    // dense path bit-identical to the im2col path.
                                    let lhs_val = if oob {
                                        0.0
                                    } else {
                                        lhs_src[n_offset + h_offset + in_w_off + ci]
                                    };
                                    acc += lhs_val * rhs_src[rhs_idx];
                                }
                            }
                        }
                        out.push(acc);
                    }
                }
            }
        }
        return conv_real_output_from_f64(
            out_dtype,
            vec![batch as u32, out_h as u32, out_w as u32, c_out as u32],
            out,
        );
    }

    for n in 0..batch {
        let n_offset = n
            .checked_mul(height_width_c_in)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "conv batch index overflow".into(),
            })?;
        for oh in 0..out_h {
            for ow in 0..out_w {
                for co in 0..c_out {
                    if is_complex {
                        let mut acc_re = 0.0_f64;
                        let mut acc_im = 0.0_f64;
                        for kh in 0..kernel_h {
                            let in_h = (oh * stride_h + kh * dil_h) as isize - pad_top as isize;
                            let h_oob = in_h < 0 || (in_h as usize) >= height;
                            let h_offset = if h_oob {
                                0
                            } else {
                                (in_h as usize) * width_c_in
                            };
                            for kw in 0..kernel_w {
                                let in_w =
                                    (ow * stride_w + kw * dil_w) as isize - pad_left as isize;
                                let w_oob = in_w < 0 || (in_w as usize) >= width;
                                let oob = h_oob || w_oob;
                                let in_w_off = if w_oob { 0 } else { (in_w as usize) * c_in };
                                for ci in 0..c_in {
                                    let rhs_idx =
                                        kh * kw_c_in_c_out + kw * c_in_c_out + ci * c_out + co;
                                    // Zero-padded (OOB) taps contribute 0·w, matching
                                    // XLA's zero-padding (and the real conv2d path); a
                                    // no-op for finite data, but fixes signed-zero /
                                    // non-finite parity vs the old `continue`-skip.
                                    let (lhs_re, lhs_im) = if oob {
                                        (0.0, 0.0)
                                    } else {
                                        let lhs_idx = n_offset + h_offset + in_w_off + ci;
                                        literal_as_complex(&lhs.elements[lhs_idx])
                                    };
                                    let (rhs_re, rhs_im) =
                                        literal_as_complex(&rhs.elements[rhs_idx]);
                                    acc_re += lhs_re * rhs_re - lhs_im * rhs_im;
                                    acc_im += lhs_re * rhs_im + lhs_im * rhs_re;
                                }
                            }
                        }
                        elements.push(conv_literal_from_complex(out_dtype, acc_re, acc_im));
                    } else {
                        let mut acc = 0.0_f64;
                        for kh in 0..kernel_h {
                            let in_h = (oh * stride_h + kh) as isize - pad_top as isize;
                            let h_oob = in_h < 0 || (in_h as usize) >= height;
                            let h_offset = if h_oob {
                                0
                            } else {
                                (in_h as usize) * width_c_in
                            };
                            for kw in 0..kernel_w {
                                let in_w = (ow * stride_w + kw) as isize - pad_left as isize;
                                let w_oob = in_w < 0 || (in_w as usize) >= width;
                                let oob = h_oob || w_oob;
                                let in_w_off = if w_oob { 0 } else { (in_w as usize) * c_in };
                                for ci in 0..c_in {
                                    let rhs_idx =
                                        kh * kw_c_in_c_out + kw * c_in_c_out + ci * c_out + co;
                                    // Out-of-bounds (padding) taps contribute `0·w`,
                                    // exactly as XLA's zero-padding and the im2col GEMM
                                    // do — adding the term rather than skipping it keeps
                                    // the two paths bit-identical, including the signed-
                                    // zero accumulator case a `continue` would mishandle.
                                    let lhs_val = if oob {
                                        0.0
                                    } else {
                                        let lhs_idx = n_offset + h_offset + in_w_off + ci;
                                        lhs.elements[lhs_idx].as_f64().unwrap_or(0.0)
                                    };
                                    let rhs_val = rhs.elements[rhs_idx].as_f64().unwrap_or(0.0);
                                    acc += lhs_val * rhs_val;
                                }
                            }
                        }
                        elements.push(conv_float_literal_from_f64(out_dtype, acc));
                    }
                }
            }
        }
    }

    Ok(Value::Tensor(TensorValue::new(
        out_dtype,
        Shape {
            dims: vec![batch as u32, out_h as u32, out_w as u32, c_out as u32],
        },
        elements,
    )?))
}

fn parse_positive_stride(primitive: Primitive, stride: Option<&str>) -> Result<usize, EvalError> {
    let raw = stride.unwrap_or("1").trim();
    let parsed = raw.parse::<usize>().map_err(|_| EvalError::Unsupported {
        primitive,
        detail: format!("invalid conv stride {raw:?}"),
    })?;
    if parsed == 0 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "conv stride must be positive".to_owned(),
        });
    }
    Ok(parsed)
}

fn parse_stride_pair(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
) -> Result<(usize, usize), EvalError> {
    let strides_str = params.get("strides").map(String::as_str).unwrap_or("1");
    let mut parts = strides_str.split(',');
    let first = parts.next().unwrap_or("1");
    let second = parts.next();
    if parts.next().is_some() {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("invalid conv strides {strides_str:?}"),
        });
    }
    match second {
        Some(width) => Ok((
            parse_positive_stride(primitive, Some(first))?,
            parse_positive_stride(primitive, Some(width))?,
        )),
        None => {
            let stride = parse_positive_stride(primitive, Some(first))?;
            Ok((stride, stride))
        }
    }
}

/// Parse `rhs_dilation` (atrous/dilated kernel) for 2D conv as a per-spatial-dim
/// `(dilation_h, dilation_w)` pair. Absent or "1"/"1,1" means no dilation. A single
/// value applies to both dims; each factor must be >= 1 (`parse_positive_stride`).
/// A dilation of `d` spaces the kernel taps `d` apart: effective kernel extent is
/// `(k - 1) * d + 1`, and tap `kh` reads input row `oh*stride + kh*d - pad`.
fn parse_dilation_pair(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
) -> Result<(usize, usize), EvalError> {
    let Some(raw) = params.get("rhs_dilation") else {
        return Ok((1, 1));
    };
    let mut parts = raw.split(',');
    let first = parts.next().unwrap_or("1");
    let second = parts.next();
    if parts.next().is_some() {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("invalid conv rhs_dilation {raw:?}"),
        });
    }
    match second {
        Some(w) => Ok((
            parse_positive_stride(primitive, Some(first))?,
            parse_positive_stride(primitive, Some(w))?,
        )),
        None => {
            let d = parse_positive_stride(primitive, Some(first))?;
            Ok((d, d))
        }
    }
}

/// Parse `rhs_dilation` for 1D conv as a single per-spatial-dim factor (>= 1).
/// Absent or "1" means no dilation; a multi-value list is rejected (1D has one
/// spatial dim).
fn parse_conv_1d_dilation(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
) -> Result<usize, EvalError> {
    let Some(raw) = params.get("rhs_dilation") else {
        return Ok(1);
    };
    let mut parts = raw.split(',');
    let first = parts.next().unwrap_or("1");
    if parts.next().is_some() {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("invalid 1D conv rhs_dilation {raw:?} (expected a single factor)"),
        });
    }
    parse_positive_stride(primitive, Some(first))
}

/// Parse a positive-integer conv group count param (`feature_group_count` /
/// `batch_group_count`); absent, empty, or "1" means ordinary (ungrouped) conv.
fn parse_conv_named_count(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
    key: &str,
) -> Result<usize, EvalError> {
    let Some(raw) = params.get(key) else {
        return Ok(1);
    };
    let t = raw.trim();
    if t.is_empty() {
        return Ok(1);
    }
    t.parse::<usize>()
        .ok()
        .filter(|&g| g >= 1)
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: format!("invalid {key} {raw:?}"),
        })
}

/// Parse `feature_group_count` (grouped / depthwise conv) as a positive integer;
/// absent or "1" means ordinary (ungrouped) conv.
fn parse_conv_group_count(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
) -> Result<usize, EvalError> {
    parse_conv_named_count(primitive, params, "feature_group_count")
}

fn compute_output_and_pad(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: ConvPadding,
) -> (usize, usize) {
    // Empty input produces empty output with no padding
    if input_size == 0 {
        return (0, 0);
    }
    match padding {
        ConvPadding::Same | ConvPadding::SameLower => {
            let out = input_size.div_ceil(stride);
            // out >= 1 since input_size >= 1 and stride >= 1
            let pad_total = ((out - 1) * stride + kernel_size).saturating_sub(input_size);
            let pad_low = if padding == ConvPadding::SameLower {
                pad_total.div_ceil(2)
            } else {
                pad_total / 2
            };
            (out, pad_low)
        }
        ConvPadding::Valid => (conv_valid_output_dim(input_size, kernel_size, stride), 0),
    }
}

fn conv_valid_output_dim(input_size: usize, kernel_size: usize, stride: usize) -> usize {
    if input_size < kernel_size {
        0
    } else {
        (input_size - kernel_size) / stride + 1
    }
}

fn conv_morsel_threads(output_elems: usize, ops: usize) -> usize {
    const PARALLEL_MIN_OPS: usize = 1 << 21;
    const OPS_PER_THREAD: usize = 1 << 18;
    const MAX_THREADS: usize = 16;

    if output_elems <= 1 || ops < PARALLEL_MIN_OPS {
        return 1;
    }
    let by_work = (ops / OPS_PER_THREAD).max(1);
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(output_elems)
        .min(by_work)
        .clamp(1, MAX_THREADS)
}

// ── Rev: reverse elements along specified axes ─────────────────

pub(crate) fn eval_rev(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Rev;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let axes = parse_usize_param(primitive, "axes", params)?;

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()),
        Value::Tensor(tensor) => {
            let dims = &tensor.shape.dims;
            let rank = dims.len();

            for &a in &axes {
                if a >= rank {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("axis {a} out of range for rank {rank}"),
                    });
                }
            }

            let total = tensor.elements.len();
            if total == 0 {
                return Ok(Value::Tensor(tensor.clone()));
            }

            // Compute strides (row-major)
            let strides = checked_row_major_strides(primitive, "rev", dims)?;
            let reversed: Vec<bool> = (0..rank).map(|ax| axes.contains(&ax)).collect();

            // Dense fast paths: gather reversed elements straight from the typed
            // slice into dense storage via a shared reusable-odometer kernel
            // (rev_gather) — bypassing the per-element `vec![0; rank]` allocation
            // and division decode, the input Literal materialization, and the
            // Vec<Literal> output. Bit-identical to the generic path below.
            if let Some(src) = tensor.elements.as_f64_slice() {
                let out = rev_gather(src, dims, &strides, &reversed, total);
                return Ok(Value::Tensor(
                    TensorValue::new_f64_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            // f32 (JAX's default dtype) and BF16/F16 (half-precision) reuse the same
            // generic `rev_gather<T>` kernel over their typed backings — copying the
            // reversed elements into dense output, avoiding the full-buffer Literal
            // materialization + boxed output of the generic `as_slice` path below.
            if let Some(src) = tensor.elements.as_f32_slice() {
                let out = rev_gather(src, dims, &strides, &reversed, total);
                return Ok(Value::Tensor(
                    TensorValue::new_f32_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            if let Some(src) = tensor.elements.as_half_float_slice() {
                let out = rev_gather(src, dims, &strides, &reversed, total);
                return Ok(Value::Tensor(
                    TensorValue::new_half_float_values(tensor.dtype, tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            if let Some(src) = tensor.elements.as_i64_slice() {
                let out = rev_gather(src, dims, &strides, &reversed, total);
                // Preserve the input integer width (i32 reverse must stay I32).
                let tv = if tensor.dtype == DType::I32 {
                    TensorValue::new_i32_values(tensor.shape.clone(), out)
                } else {
                    TensorValue::new_i64_values(tensor.shape.clone(), out)
                };
                return Ok(Value::Tensor(tv.map_err(EvalError::InvalidTensor)?));
            }
            if let Some(src) = tensor.elements.as_bool_slice() {
                let out = rev_gather(src, dims, &strides, &reversed, total);
                return Ok(Value::Tensor(
                    TensorValue::new_bool_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }

            let src = tensor.elements.as_slice();
            let result = rev_gather(src, dims, &strides, &reversed, total);
            Ok(Value::Tensor(
                TensorValue::new(tensor.dtype, tensor.shape.clone(), result).map_err(|e| {
                    EvalError::Unsupported {
                        primitive,
                        detail: e.to_string(),
                    }
                })?,
            ))
        }
    }
}

/// Reverse-along-axes gather shared by `eval_rev`'s dense and generic paths: walk
/// output positions in row-major order with a reused coordinate odometer (no
/// per-element allocation), reflect the coordinate on each reversed axis, and
/// copy `src[src_flat]`. Generic over the element type so the dense (f64/i64/bool)
/// and Literal paths share identical index math.
fn rev_gather<T: Copy>(
    src: &[T],
    dims: &[u32],
    strides: &[usize],
    reversed: &[bool],
    total: usize,
) -> Vec<T> {
    let rank = dims.len();
    let mut out = Vec::with_capacity(total);
    let mut coords = vec![0_usize; rank];
    for _ in 0..total {
        let mut src_flat = 0_usize;
        for ax in 0..rank {
            let c = if reversed[ax] {
                dims[ax] as usize - 1 - coords[ax]
            } else {
                coords[ax]
            };
            src_flat += c * strides[ax];
        }
        out.push(src[src_flat]);
        for ax in (0..rank).rev() {
            coords[ax] += 1;
            if coords[ax] < dims[ax] as usize {
                break;
            }
            coords[ax] = 0;
        }
    }
    out
}

// ── Squeeze: remove singleton dimensions ───────────────────────

pub(crate) fn eval_squeeze(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Squeeze;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()),
        Value::Tensor(tensor) => {
            let dims = &tensor.shape.dims;

            let squeeze_dims: Vec<usize> = if params.contains_key("dimensions") {
                // Parse as i64 so a negative (end-relative) dimension is normalized
                // against the input rank, matching lax.squeeze's
                // `canonicalize_axis(i, ndim)`. A usize parse rejected e.g.
                // dimensions=-1 (squeeze the trailing dim), a common usage.
                let rank = dims.len() as i64;
                parse_i64_param(primitive, "dimensions", params)?
                    .into_iter()
                    .map(|d| {
                        let norm = if d < 0 { d + rank } else { d };
                        if norm < 0 || norm >= rank {
                            return Err(EvalError::Unsupported {
                                primitive,
                                detail: format!(
                                    "dimension {d} out of range for rank {}",
                                    dims.len()
                                ),
                            });
                        }
                        Ok(norm as usize)
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                // If no dimensions specified, squeeze all size-1 dims
                dims.iter()
                    .enumerate()
                    .filter(|&(_, &d)| d == 1)
                    .map(|(i, _)| i)
                    .collect()
            };

            for &d in &squeeze_dims {
                if d >= dims.len() {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!("dimension {d} out of range for rank {}", dims.len()),
                    });
                }
                if dims[d] != 1 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "cannot squeeze dimension {d} with size {} (must be 1)",
                            dims[d]
                        ),
                    });
                }
            }

            let new_dims: Vec<u32> = dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !squeeze_dims.contains(i))
                .map(|(_, &d)| d)
                .collect();

            if new_dims.is_empty() {
                // All dims squeezed — return scalar
                Ok(Value::Scalar(tensor.elements[0]))
            } else {
                // Squeeze is metadata-only: clone the backing buffer (cheap Arc
                // bump; dense/concat storage preserved) rather than materializing
                // the whole buffer via `to_vec()`.
                Ok(Value::Tensor(
                    TensorValue::new_with_literal_buffer(
                        tensor.dtype,
                        Shape { dims: new_dims },
                        tensor.elements.clone(),
                    )
                    .map_err(|e| EvalError::Unsupported {
                        primitive,
                        detail: e.to_string(),
                    })?,
                ))
            }
        }
    }
}

// ── Split: split array along an axis ───────────────────────────

pub(crate) fn eval_split(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Split;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(_) => Err(EvalError::Unsupported {
            primitive,
            detail: "cannot split a scalar".into(),
        }),
        Value::Tensor(tensor) => {
            // Parse axis as i64 so a negative (end-relative) axis is normalized
            // against the rank, matching lax.split's canonicalize_axis(axis, ndim).
            let raw_axis: i64 = params
                .get("axis")
                .map(|raw| {
                    raw.trim()
                        .parse::<i64>()
                        .map_err(|_| EvalError::Unsupported {
                            primitive,
                            detail: format!("invalid integer in param 'axis': '{raw}'"),
                        })
                })
                .transpose()?
                .unwrap_or(0);
            let dims = &tensor.shape.dims;
            let rank = dims.len();

            let norm = if raw_axis < 0 {
                raw_axis + rank as i64
            } else {
                raw_axis
            };
            if norm < 0 || norm >= rank as i64 {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("axis {raw_axis} out of range for rank {rank}"),
                });
            }
            let axis = norm as usize;

            let axis_size = dims[axis] as usize;

            // Determine split sizes
            let sizes: Vec<usize> = if params.contains_key("sizes") {
                parse_usize_param(primitive, "sizes", params)?
            } else if params.contains_key("num_sections") {
                let num_sections_vec = parse_usize_param(primitive, "num_sections", params)?;
                let num_sections = num_sections_vec[0];
                if num_sections == 0 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "num_sections must be positive".into(),
                    });
                }
                if !axis_size.is_multiple_of(num_sections) {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "axis size {axis_size} not evenly divisible by {num_sections}"
                        ),
                    });
                }
                let section_size = axis_size / num_sections;
                vec![section_size; num_sections]
            } else {
                vec![axis_size]
            };

            // Validate sizes sum to axis_size
            let total: usize = sizes.iter().sum();
            if total != axis_size {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("split sizes sum to {total} but axis size is {axis_size}"),
                });
            }

            // For now, Split returns the first section as a single Value.
            // Multi-output support: we return a Tensor containing concatenated sections.
            // Since our Value type doesn't support tuples, we return the first section only
            // (consistent with how Scan and other multi-output ops work — the caller
            // handles unpacking through the equation's output vars).

            // Actually, the simplest approach: return all sections concatenated as a single
            // tensor with an extra leading dimension = num_sections.
            // But the bead says "Returns multiple outputs" which our Value can't directly do.
            // For eval_primitive which returns a single Value, we'll return a reshaped tensor.
            // The correct approach: reshape to [num_sections, section_size, ...rest_dims].

            let num_sections = sizes.len();
            if sizes.windows(2).all(|w| w[0] == w[1]) || sizes.len() == 1 {
                // Equal split — reshape into [num_sections, section_size, ...rest]
                let section_size = sizes[0];
                let mut new_dims = Vec::with_capacity(rank + 1);
                for (i, &d) in dims.iter().enumerate() {
                    if i == axis {
                        new_dims.push(num_sections as u32);
                        new_dims.push(section_size as u32);
                    } else {
                        new_dims.push(d);
                    }
                }
                // Equal split is a pure reshape (adds a leading num_sections dim) —
                // metadata-only. Clone the backing buffer (cheap Arc bump;
                // dense/concat storage preserved) rather than `to_vec()`.
                Ok(Value::Tensor(
                    TensorValue::new_with_literal_buffer(
                        tensor.dtype,
                        Shape { dims: new_dims },
                        tensor.elements.clone(),
                    )
                    .map_err(|e| EvalError::Unsupported {
                        primitive,
                        detail: e.to_string(),
                    })?,
                ))
            } else {
                // Uneven split (explicit per-section `sizes` that are not all
                // equal) cannot be represented in V1's single-output packed
                // tensor model: the sections differ in extent along the split
                // axis, so they do not fit a single rectangular tensor with a
                // leading `num_sections` dimension. Fail closed rather than
                // silently returning only the first section — that diverged
                // from JAX's multi-array result and corrupted any transform
                // (grad/jit/vmap) flowing through the dropped sections.
                Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "uneven split with explicit sizes {sizes:?} is unsupported in V1; \
                         only equal-size splits (num_sections, or all-equal explicit sizes) \
                         are representable in the single-output packed-tensor model"
                    ),
                })
            }
        }
    }
}

/// Multi-output Split: slice the input into one tensor per section along `axis`,
/// returning `Vec<Value>` (matching `jnp.split`'s list-of-arrays semantics). Unlike
/// the single-output [`eval_split`] (which packs equal sections into one rectangular
/// tensor and fails closed on uneven sizes), each section is its own tensor, so
/// UNEVEN explicit `sizes` are fully supported. This is the path `eval_primitive_multi`
/// uses, so an N-section split equation yields N outputs (not a single packed tensor).
pub(crate) fn eval_split_multi(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, EvalError> {
    let primitive = Primitive::Split;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }
    let tensor = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "cannot split a scalar".into(),
            });
        }
    };

    // Parse axis as i64 so a negative (end-relative) axis is normalized against
    // the rank, matching lax.split's canonicalize_axis(axis, ndim).
    let raw_axis: i64 = params
        .get("axis")
        .map(|raw| {
            raw.trim()
                .parse::<i64>()
                .map_err(|_| EvalError::Unsupported {
                    primitive,
                    detail: format!("invalid integer in param 'axis': '{raw}'"),
                })
        })
        .transpose()?
        .unwrap_or(0);
    let dims = &tensor.shape.dims;
    let rank = dims.len();
    let norm = if raw_axis < 0 {
        raw_axis + rank as i64
    } else {
        raw_axis
    };
    if norm < 0 || norm >= rank as i64 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("axis {raw_axis} out of range for rank {rank}"),
        });
    }
    let axis = norm as usize;
    let axis_size = dims[axis] as usize;

    let sizes: Vec<usize> = if params.contains_key("sizes") {
        parse_usize_param(primitive, "sizes", params)?
    } else if params.contains_key("num_sections") {
        let num_sections = parse_usize_param(primitive, "num_sections", params)?[0];
        if num_sections == 0 {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "num_sections must be positive".into(),
            });
        }
        if !axis_size.is_multiple_of(num_sections) {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("axis size {axis_size} not evenly divisible by {num_sections}"),
            });
        }
        vec![axis_size / num_sections; num_sections]
    } else {
        vec![axis_size]
    };

    let total: usize = sizes.iter().sum();
    if total != axis_size {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("split sizes sum to {total} but axis size is {axis_size}"),
        });
    }

    // Row-major slice along `axis`: for each outer index the section is the
    // contiguous block `[start*inner, (start+len)*inner)` within that outer row.
    let outer: usize = dims[..axis].iter().map(|&d| d as usize).product();
    let inner: usize = dims[axis + 1..].iter().map(|&d| d as usize).product();
    let mut out = Vec::with_capacity(sizes.len());
    let mut start = 0usize;
    for &len in &sizes {
        let mut elements = Vec::with_capacity(outer * len * inner);
        for o in 0..outer {
            let base = o * axis_size * inner + start * inner;
            elements.extend_from_slice(&tensor.elements[base..base + len * inner]);
        }
        let mut new_dims = dims.clone();
        new_dims[axis] = len as u32;
        out.push(Value::Tensor(
            TensorValue::new(tensor.dtype, Shape { dims: new_dims }, elements).map_err(|e| {
                EvalError::Unsupported {
                    primitive,
                    detail: e.to_string(),
                }
            })?,
        ));
        start += len;
    }
    Ok(out)
}

// ── ExpandDims: add a singleton dimension ──────────────────────

pub(crate) fn eval_expand_dims(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::ExpandDims;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    // Parse as i64 so a negative (end-relative) axis is normalized against the
    // OUTPUT rank (input rank + 1), matching numpy/jnp expand_dims. A usize parse
    // rejected e.g. axis=-1 (insert a trailing dim), the most common usage.
    let raw_axis: i64 = params
        .get("axis")
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.trim().parse::<i64>().ok())
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "invalid axis param for expand_dims".to_owned(),
        })?;

    match &inputs[0] {
        Value::Scalar(lit) => {
            // Scalar -> rank-1 tensor of shape [1]
            Ok(Value::Tensor(
                TensorValue::new(
                    match lit {
                        Literal::I64(_) => DType::I64,
                        Literal::U32(_) => DType::U32,
                        Literal::U64(_) => DType::U64,
                        Literal::Bool(_) => DType::Bool,
                        Literal::BF16Bits(_) => DType::BF16,
                        Literal::F16Bits(_) => DType::F16,
                        Literal::F32Bits(_) => DType::F32,
                        Literal::F64Bits(_) => DType::F64,
                        Literal::Complex64Bits(..) => DType::Complex64,
                        Literal::Complex128Bits(..) => DType::Complex128,
                    },
                    Shape { dims: vec![1] },
                    vec![*lit],
                )
                .map_err(|e| EvalError::Unsupported {
                    primitive,
                    detail: e.to_string(),
                })?,
            ))
        }
        Value::Tensor(tensor) => {
            let rank = tensor.shape.dims.len();
            let norm = if raw_axis < 0 {
                raw_axis + rank as i64 + 1
            } else {
                raw_axis
            };
            if norm < 0 || norm > rank as i64 {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("axis {raw_axis} out of range for rank {rank} (max is {rank})"),
                });
            }
            let axis = norm as usize;

            let mut new_dims = tensor.shape.dims.clone();
            new_dims.insert(axis, 1);

            // Expand_dims is metadata-only: clone the backing buffer (cheap Arc
            // bump; dense/concat storage preserved) rather than `to_vec()`.
            Ok(Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    tensor.dtype,
                    Shape { dims: new_dims },
                    tensor.elements.clone(),
                )
                .map_err(|e| EvalError::Unsupported {
                    primitive,
                    detail: e.to_string(),
                })?,
            ))
        }
    }
}

pub(crate) fn eval_tile(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Tile;
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let reps = parse_usize_param(primitive, "reps", params)?;

    match &inputs[0] {
        Value::Scalar(lit) => {
            if reps.is_empty() || reps.len() > 1 {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("scalar tile requires 1 rep, got {}", reps.len()),
                });
            }
            let rep = reps[0];
            if rep == 0 {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "tile rep must be positive".into(),
                });
            }
            if rep == 1 {
                return Ok(inputs[0].clone());
            }
            let dtype = match lit {
                Literal::I64(_) => DType::I64,
                Literal::U32(_) => DType::U32,
                Literal::U64(_) => DType::U64,
                Literal::Bool(_) => DType::Bool,
                Literal::BF16Bits(_) => DType::BF16,
                Literal::F16Bits(_) => DType::F16,
                Literal::F32Bits(_) => DType::F32,
                Literal::F64Bits(_) => DType::F64,
                Literal::Complex64Bits(..) => DType::Complex64,
                Literal::Complex128Bits(..) => DType::Complex128,
            };
            Ok(Value::Tensor(
                TensorValue::new(dtype, Shape::vector(rep as u32), vec![*lit; rep])
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if reps.len() != rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "reps length {} does not match tensor rank {}",
                        reps.len(),
                        rank
                    ),
                });
            }

            if reps.contains(&0) {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: "tile reps must all be positive".into(),
                });
            }

            if reps.iter().all(|&r| r == 1) {
                return Ok(inputs[0].clone());
            }

            let new_dims: Vec<u32> = tensor
                .shape
                .dims
                .iter()
                .zip(reps.iter())
                .map(|(&d, &r)| {
                    d.checked_mul(r as u32)
                        .ok_or_else(|| EvalError::Unsupported {
                            primitive,
                            detail: "tile result dimension overflows u32".into(),
                        })
                })
                .collect::<Result<_, _>>()?;

            let new_count: u64 = new_dims
                .iter()
                .try_fold(1_u64, |acc, &d| acc.checked_mul(u64::from(d)))
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "tile result element count overflows u64".into(),
                })?;

            // Dense fast paths: tile straight off the typed backing with bulk
            // `extend_from_slice` (memcpy) into dense output, avoiding the full
            // input `Vec<Literal>` materialization (`tensor.elements` deref) AND
            // the boxed output. Same block-copy order as `tile_recursive` -> bit
            // -for-bit identical. Falls through to the Literal path for other dtypes.
            macro_rules! dense_tile {
                ($slice:expr, $ctor:expr) => {{
                    let mut out = Vec::with_capacity(new_count as usize);
                    tile_recursive_dense($slice, &tensor.shape.dims, &reps, 0, &mut out);
                    return Ok(Value::Tensor($ctor(
                        Shape {
                            dims: new_dims.clone(),
                        },
                        out,
                    )?));
                }};
            }
            if let Some(s) = tensor.elements.as_f64_slice() {
                dense_tile!(s, TensorValue::new_f64_values);
            }
            if let Some(s) = tensor.elements.as_f32_slice() {
                dense_tile!(s, TensorValue::new_f32_values);
            }
            if let Some(s) = tensor.elements.as_half_float_slice() {
                let dt = tensor.dtype;
                dense_tile!(s, |sh, o| TensorValue::new_half_float_values(dt, sh, o));
            }
            if let Some(s) = tensor.elements.as_i64_slice() {
                dense_tile!(s, |sh, o| if tensor.dtype == DType::I32 {
                    TensorValue::new_i32_values(sh, o)
                } else {
                    TensorValue::new_i64_values(sh, o)
                });
            }

            let mut result = Vec::with_capacity(new_count as usize);
            tile_recursive(&tensor.elements, &tensor.shape.dims, &reps, 0, &mut result);

            Ok(Value::Tensor(
                TensorValue::new(tensor.dtype, Shape { dims: new_dims }, result)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
    }
}

/// Dense, type-generic sibling of [`tile_recursive`]: tiles a contiguous typed
/// slice into `out` via bulk `extend_from_slice` (memcpy for `Copy` elements),
/// using the identical block-copy traversal so the result is bit-for-bit the same
/// element sequence as the `Literal` path — for any backing type (f64/f32/half/i64).
fn tile_recursive_dense<T: Copy>(
    src: &[T],
    dims: &[u32],
    reps: &[usize],
    depth: usize,
    out: &mut Vec<T>,
) {
    if depth == dims.len() {
        return;
    }
    let dim = dims[depth] as usize;
    let rep = reps[depth];
    let stride: usize = dims[depth + 1..].iter().map(|&d| d as usize).product();
    if depth == dims.len() - 1 {
        for _ in 0..rep {
            out.extend_from_slice(&src[..dim]);
        }
    } else {
        for _ in 0..rep {
            for i in 0..dim {
                let start = i * stride;
                tile_recursive_dense(
                    &src[start..start + stride],
                    &dims[depth + 1..],
                    &reps[depth + 1..],
                    0,
                    out,
                );
            }
        }
    }
}

fn tile_recursive(
    elements: &[Literal],
    dims: &[u32],
    reps: &[usize],
    depth: usize,
    result: &mut Vec<Literal>,
) {
    if depth == dims.len() {
        return;
    }

    let dim = dims[depth] as usize;
    let rep = reps[depth];
    let stride: usize = dims[depth + 1..].iter().map(|&d| d as usize).product();

    if depth == dims.len() - 1 {
        for _ in 0..rep {
            result.extend(elements.iter().take(dim).copied());
        }
    } else {
        for _ in 0..rep {
            for i in 0..dim {
                let start = i * stride;
                let sub_elements = &elements[start..start + stride];
                tile_recursive(
                    sub_elements,
                    &dims[depth + 1..],
                    &reps[depth + 1..],
                    0,
                    result,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn v_f64(data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn mat_f64(rows: u32, cols: u32, data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![rows, cols],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn extract_f64_vec(val: &Value) -> Vec<f64> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect()
    }
    fn extract_i64_vec(val: &Value) -> Vec<i64> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect()
    }
    fn extract_shape(val: &Value) -> Vec<u32> {
        val.as_tensor().unwrap().shape.dims.clone()
    }
    fn params(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|&(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }

    #[test]
    fn conv2d_f32_im2col_gemm_bit_identical_to_reference() {
        // f32 conv2d (default ML dtype) routes through the NATIVE-f32 im2col + GEMM
        // path (f32 accumulation, matching XLA's f32 conv — not the prior f64 promote).
        // Sized above CONV_IM2COL_MIN_OPS to exercise the GEMM path. Must be bit-for-bit
        // identical to the textbook ascending-(kh,kw,ci) reference: f32 accumulation of
        // f32 products.
        let (h, w, c_in, c_out, kh, kw) = (20usize, 20usize, 4usize, 8usize, 3usize, 3usize);
        let xf: Vec<f32> = (0..h * w * c_in)
            .map(|i| (i as f32 * 0.011).sin() * 1.3 - 0.2)
            .collect();
        let kf: Vec<f32> = (0..kh * kw * c_in * c_out)
            .map(|i| (i as f32 * 0.017).cos() * 0.9 + 0.1)
            .collect();
        let mk32 = |dims: Vec<u32>, data: &[f32]| {
            Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape { dims },
                    data.iter().map(|&v| Literal::from_f32(v)).collect(),
                )
                .unwrap(),
            )
        };
        let out = eval_conv(
            Primitive::Conv,
            &[
                mk32(vec![1, h as u32, w as u32, c_in as u32], &xf),
                mk32(vec![kh as u32, kw as u32, c_in as u32, c_out as u32], &kf),
            ],
            &params(&[("padding", "valid"), ("strides", "1")]),
        )
        .unwrap();
        let (out_h, out_w) = (h - kh + 1, w - kw + 1);
        let Value::Tensor(t) = out else {
            panic!("expected tensor")
        };
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(
            t.shape.dims,
            vec![1, out_h as u32, out_w as u32, c_out as u32]
        );
        let got: Vec<u32> = t
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(b) => *b,
                o => panic!("unexpected {o:?}"),
            })
            .collect();
        let mut want = Vec::new();
        for oh in 0..out_h {
            for ow in 0..out_w {
                for co in 0..c_out {
                    let mut acc = 0.0f32;
                    for dh in 0..kh {
                        for dw in 0..kw {
                            for ci in 0..c_in {
                                let lv = xf[((oh + dh) * w + (ow + dw)) * c_in + ci];
                                let kv = kf[((dh * kw + dw) * c_in + ci) * c_out + co];
                                acc += lv * kv;
                            }
                        }
                    }
                    want.push(acc.to_bits());
                }
            }
        }
        assert_eq!(
            got, want,
            "f32 conv2d must be bit-identical to the f32-accum reference"
        );
    }

    #[test]
    fn conv2d_half_native_f32_accum_matches_reference() {
        // BF16/F16 conv2d now accumulates in f32 (XLA parity — the prior path promoted
        // to f64). Must be bit-for-bit identical to the textbook reference: decode the
        // half operands to f32, fold each output ascending-(kh,kw,ci) in f32, then round
        // f32->half (round-to-nearest-even). Sized above CONV_IM2COL_MIN_OPS.
        let (h, w, c_in, c_out, kh, kw) = (20usize, 20usize, 4usize, 8usize, 3usize, 3usize);
        for half_dt in [DType::BF16, DType::F16] {
            let to_half = |v: f64| -> u16 {
                match conv_float_literal_from_f64(half_dt, v) {
                    Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                    _ => 0,
                }
            };
            let decode = |bits: u16| -> f32 {
                if half_dt == DType::BF16 {
                    Literal::BF16Bits(bits).as_bf16_f32().unwrap()
                } else {
                    Literal::F16Bits(bits).as_f16_f32().unwrap()
                }
            };
            let x16: Vec<u16> = (0..h * w * c_in)
                .map(|i| to_half((i as f64 * 0.011).sin() * 1.3 - 0.2))
                .collect();
            let k16: Vec<u16> = (0..kh * kw * c_in * c_out)
                .map(|i| to_half((i as f64 * 0.017).cos() * 0.9 + 0.1))
                .collect();
            let mk = |dims: Vec<u32>, data: &[u16]| {
                Value::Tensor(
                    TensorValue::new_half_float_values(half_dt, Shape { dims }, data.to_vec())
                        .unwrap(),
                )
            };
            let out = eval_conv(
                Primitive::Conv,
                &[
                    mk(vec![1, h as u32, w as u32, c_in as u32], &x16),
                    mk(vec![kh as u32, kw as u32, c_in as u32, c_out as u32], &k16),
                ],
                &params(&[("padding", "valid"), ("strides", "1")]),
            )
            .unwrap();
            let (out_h, out_w) = (h - kh + 1, w - kw + 1);
            let Value::Tensor(t) = out else {
                panic!("expected tensor")
            };
            assert_eq!(t.dtype, half_dt);
            let got = t
                .elements
                .as_half_float_slice()
                .expect("dense half output")
                .to_vec();
            let mut want = Vec::new();
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for co in 0..c_out {
                        let mut acc = 0.0f32;
                        for dh in 0..kh {
                            for dw in 0..kw {
                                for ci in 0..c_in {
                                    let lv = decode(x16[((oh + dh) * w + (ow + dw)) * c_in + ci]);
                                    let kv = decode(k16[((dh * kw + dw) * c_in + ci) * c_out + co]);
                                    acc += lv * kv;
                                }
                            }
                        }
                        want.push(to_half(f64::from(acc)));
                    }
                }
            }
            assert_eq!(
                got, want,
                "{half_dt:?} conv2d must match the f32-accum reference"
            );
        }
    }

    #[test]
    fn conv1d_native_f32_accum_matches_reference() {
        // f32/bf16/f16 conv1d now accumulates in f32 (XLA parity — the prior path
        // promoted to f64). Bit-for-bit identical to the textbook reference: decode the
        // operands to f32, fold each output ascending-(k,ci) in f32, then round to
        // out_dtype. Sized above CONV_IM2COL_MIN_OPS to exercise the im2col GEMM.
        let (width, c_in, c_out, kw) = (200usize, 6usize, 12usize, 5usize);
        for dt in [DType::F32, DType::BF16, DType::F16] {
            let to_dt = |v: f64| -> Literal { conv_float_literal_from_f64(dt, v) };
            let bits = |l: Literal| -> u16 {
                match l {
                    Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                    _ => 0,
                }
            };
            let decode = |l: &Literal| -> f32 {
                match dt {
                    DType::F32 => match l {
                        Literal::F32Bits(b) => f32::from_bits(*b),
                        _ => 0.0,
                    },
                    DType::BF16 => l.as_bf16_f32().unwrap(),
                    _ => l.as_f16_f32().unwrap(),
                }
            };
            let xv: Vec<Literal> = (0..width * c_in)
                .map(|i| to_dt((i as f64 * 0.013).sin() * 1.2 - 0.3))
                .collect();
            let kv: Vec<Literal> = (0..kw * c_in * c_out)
                .map(|i| to_dt((i as f64 * 0.019).cos() * 0.8 + 0.2))
                .collect();
            let mk = |dims: Vec<u32>, data: &[Literal]| {
                Value::Tensor(TensorValue::new(dt, Shape { dims }, data.to_vec()).unwrap())
            };
            let out = eval_conv(
                Primitive::Conv,
                &[
                    mk(vec![1, width as u32, c_in as u32], &xv),
                    mk(vec![kw as u32, c_in as u32, c_out as u32], &kv),
                ],
                &params(&[("padding", "valid"), ("strides", "1")]),
            )
            .unwrap();
            let out_w = width - kw + 1;
            let Value::Tensor(t) = out else {
                panic!("expected tensor")
            };
            assert_eq!(t.dtype, dt);
            assert_eq!(t.shape.dims, vec![1, out_w as u32, c_out as u32]);
            // Reference: decode -> f32, ascending-(k,ci) f32 fold, round to out_dtype.
            let mut want: Vec<Literal> = Vec::new();
            for ow in 0..out_w {
                for co in 0..c_out {
                    let mut acc = 0.0f32;
                    for k in 0..kw {
                        for ci in 0..c_in {
                            let lv = decode(&xv[(ow + k) * c_in + ci]);
                            let wv = decode(&kv[(k * c_in + ci) * c_out + co]);
                            acc += lv * wv;
                        }
                    }
                    want.push(to_dt(f64::from(acc)));
                }
            }
            let got: Vec<Literal> = t.elements.iter().copied().collect();
            if dt == DType::F32 {
                let gb: Vec<u32> = got
                    .iter()
                    .map(|l| match l {
                        Literal::F32Bits(b) => *b,
                        _ => panic!("expected f32"),
                    })
                    .collect();
                let wb: Vec<u32> = want
                    .iter()
                    .map(|l| match l {
                        Literal::F32Bits(b) => *b,
                        _ => 0,
                    })
                    .collect();
                assert_eq!(gb, wb, "f32 conv1d must match the f32-accum reference");
            } else {
                let gb: Vec<u16> = got.iter().map(|l| bits(*l)).collect();
                let wb: Vec<u16> = want.iter().map(|l| bits(*l)).collect();
                assert_eq!(gb, wb, "{dt:?} conv1d must match the f32-accum reference");
            }
        }
    }

    #[test]
    fn grouped_conv_native_f32_accum_matches_reference_and_golden_sha256()
    -> Result<(), Box<dyn std::error::Error>> {
        fn decode(dtype: DType, literal: Literal) -> f32 {
            match dtype {
                DType::F32 => match literal {
                    Literal::F32Bits(bits) => f32::from_bits(bits),
                    other => panic!("expected f32 literal, got {other:?}"),
                },
                DType::BF16 => literal.as_bf16_f32().expect("bf16 literal"),
                DType::F16 => literal.as_f16_f32().expect("f16 literal"),
                other => panic!("unexpected dtype {other:?}"),
            }
        }

        fn rounded_bits(dtype: DType, value: f32) -> u64 {
            match conv_float_literal_from_f64(dtype, f64::from(value)) {
                Literal::F32Bits(bits) => u64::from(bits),
                Literal::BF16Bits(bits) | Literal::F16Bits(bits) => u64::from(bits),
                other => panic!("unexpected rounded literal {other:?}"),
            }
        }

        fn tensor_bits(dtype: DType, value: &Value) -> Vec<u64> {
            let tensor = value.as_tensor().expect("tensor output");
            assert_eq!(tensor.dtype, dtype);
            match dtype {
                DType::F32 => tensor
                    .elements
                    .as_f32_slice()
                    .map(|values| values.iter().map(|v| u64::from(v.to_bits())).collect())
                    .unwrap_or_else(|| {
                        tensor
                            .elements
                            .iter()
                            .map(|literal| match literal {
                                Literal::F32Bits(bits) => u64::from(*bits),
                                other => panic!("expected f32 literal, got {other:?}"),
                            })
                            .collect()
                    }),
                DType::BF16 | DType::F16 => tensor
                    .elements
                    .as_half_float_slice()
                    .map(|values| values.iter().map(|&bits| u64::from(bits)).collect())
                    .unwrap_or_else(|| {
                        tensor
                            .elements
                            .iter()
                            .map(|literal| match literal {
                                Literal::BF16Bits(bits) | Literal::F16Bits(bits) => {
                                    u64::from(*bits)
                                }
                                other => panic!("expected half literal, got {other:?}"),
                            })
                            .collect()
                    }),
                other => panic!("unexpected dtype {other:?}"),
            }
        }

        fn values(dtype: DType, len: usize, scale: f64, bias: f64) -> Vec<Literal> {
            (0..len)
                .map(|i| conv_float_literal_from_f64(dtype, (i as f64 * scale).sin() + bias))
                .collect()
        }

        let mut fixtures: Vec<(String, Vec<u32>, Vec<u64>)> = Vec::new();

        for dtype in [DType::F32, DType::BF16, DType::F16] {
            for &(label, width, c_in, c_out, kw, groups) in &[
                ("depthwise1d", 9usize, 4usize, 4usize, 3usize, 4usize),
                ("grouped1d", 10, 6, 9, 3, 3),
            ] {
                let rhs_c_in = c_in / groups;
                let cpg = c_out / groups;
                let lhs = values(dtype, width * c_in, 0.017, -0.25);
                let rhs = values(dtype, kw * rhs_c_in * c_out, 0.023, 0.125);
                let mk = |dims: Vec<u32>, data: &[Literal]| {
                    Value::Tensor(TensorValue::new(dtype, Shape { dims }, data.to_vec()).unwrap())
                };
                let group_count = groups.to_string();
                let output = eval_conv(
                    Primitive::Conv,
                    &[
                        mk(vec![1, width as u32, c_in as u32], &lhs),
                        mk(vec![kw as u32, rhs_c_in as u32, c_out as u32], &rhs),
                    ],
                    &params(&[
                        ("padding", "same"),
                        ("strides", "1"),
                        ("feature_group_count", &group_count),
                    ]),
                )
                .unwrap();
                let shape = output.as_tensor().unwrap().shape.dims.clone();
                let got = tensor_bits(dtype, &output);
                let out_w = width;
                let pad_left = (kw - 1) / 2;
                let mut want = Vec::new();
                for ow in 0..out_w {
                    for co in 0..c_out {
                        let in_ch_base = (co / cpg) * rhs_c_in;
                        let mut acc = 0.0f32;
                        for k in 0..kw {
                            let in_pos = ow as isize + k as isize - pad_left as isize;
                            let oob = in_pos < 0 || (in_pos as usize) >= width;
                            for ci in 0..rhs_c_in {
                                let lhs_val = if oob {
                                    0.0
                                } else {
                                    decode(dtype, lhs[(in_pos as usize) * c_in + in_ch_base + ci])
                                };
                                let rhs_val = decode(dtype, rhs[(k * rhs_c_in + ci) * c_out + co]);
                                acc += lhs_val * rhs_val;
                            }
                        }
                        want.push(rounded_bits(dtype, acc));
                    }
                }
                assert_eq!(got, want, "{dtype:?} {label} native grouped conv1d");
                fixtures.push((format!("{dtype:?}_{label}"), shape, got));
            }

            for &(label, h, w, c_in, c_out, kh, kw, groups) in &[
                (
                    "depthwise2d",
                    6usize,
                    5usize,
                    4usize,
                    4usize,
                    3usize,
                    3usize,
                    4usize,
                ),
                ("grouped2d", 6, 5, 6, 9, 2, 3, 3),
            ] {
                let rhs_c_in = c_in / groups;
                let cpg = c_out / groups;
                let lhs = values(dtype, h * w * c_in, 0.011, -0.375);
                let rhs = values(dtype, kh * kw * rhs_c_in * c_out, 0.019, 0.25);
                let mk = |dims: Vec<u32>, data: &[Literal]| {
                    Value::Tensor(TensorValue::new(dtype, Shape { dims }, data.to_vec()).unwrap())
                };
                let group_count = groups.to_string();
                let output = eval_conv(
                    Primitive::Conv,
                    &[
                        mk(vec![1, h as u32, w as u32, c_in as u32], &lhs),
                        mk(
                            vec![kh as u32, kw as u32, rhs_c_in as u32, c_out as u32],
                            &rhs,
                        ),
                    ],
                    &params(&[
                        ("padding", "same"),
                        ("strides", "1"),
                        ("feature_group_count", &group_count),
                    ]),
                )
                .unwrap();
                let shape = output.as_tensor().unwrap().shape.dims.clone();
                let got = tensor_bits(dtype, &output);
                let pad_top = (kh - 1) / 2;
                let pad_left = (kw - 1) / 2;
                let mut want = Vec::new();
                for oh in 0..h {
                    for ow in 0..w {
                        for co in 0..c_out {
                            let in_ch_base = (co / cpg) * rhs_c_in;
                            let mut acc = 0.0f32;
                            for a in 0..kh {
                                let ih = oh as isize + a as isize - pad_top as isize;
                                let h_oob = ih < 0 || (ih as usize) >= h;
                                for b in 0..kw {
                                    let iw = ow as isize + b as isize - pad_left as isize;
                                    let w_oob = iw < 0 || (iw as usize) >= w;
                                    let oob = h_oob || w_oob;
                                    for ci in 0..rhs_c_in {
                                        let lhs_val = if oob {
                                            0.0
                                        } else {
                                            decode(
                                                dtype,
                                                lhs[((ih as usize * w + iw as usize) * c_in)
                                                    + in_ch_base
                                                    + ci],
                                            )
                                        };
                                        let rhs_idx = ((a * kw + b) * rhs_c_in + ci) * c_out + co;
                                        acc += lhs_val * decode(dtype, rhs[rhs_idx]);
                                    }
                                }
                            }
                            want.push(rounded_bits(dtype, acc));
                        }
                    }
                }
                assert_eq!(got, want, "{dtype:?} {label} native grouped conv2d");
                fixtures.push((format!("{dtype:?}_{label}"), shape, got));
            }
        }

        let digest = fj_test_utils::fixture_id_from_json(&fixtures)?;
        assert_eq!(
            digest, "dfe98ccb7192d4ce86072dccafae4f8717d7c5abec0ccd16367070c05811f8b3",
            "native grouped f32 conv digest changed"
        );
        Ok(())
    }

    #[test]
    fn conv2d_f32_native_accum_golden_sha256() -> Result<(), Box<dyn std::error::Error>> {
        fn run_case(
            h: usize,
            w: usize,
            c_in: usize,
            c_out: usize,
            kh: usize,
            kw: usize,
            params: &[(&str, &str)],
        ) -> Vec<u32> {
            let xf: Vec<f32> = (0..h * w * c_in)
                .map(|i| (i as f32 * 0.007).sin() * 0.75 - 0.125)
                .collect();
            let kf: Vec<f32> = (0..kh * kw * c_in * c_out)
                .map(|i| (i as f32 * 0.013).cos() * 1.25 + 0.0625)
                .collect();
            let lhs = Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: vec![1, h as u32, w as u32, c_in as u32],
                    },
                    xf,
                )
                .unwrap(),
            );
            let rhs = Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: vec![kh as u32, kw as u32, c_in as u32, c_out as u32],
                    },
                    kf,
                )
                .unwrap(),
            );
            let out = eval_conv(Primitive::Conv, &[lhs, rhs], &self::params(params)).unwrap();
            let Value::Tensor(tensor) = out else {
                panic!("expected tensor")
            };
            assert_eq!(tensor.dtype, DType::F32);
            tensor
                .elements
                .as_f32_slice()
                .expect("f32 conv output should stay dense")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        }

        let fixtures = [
            (
                "im2col_valid",
                run_case(
                    20,
                    20,
                    4,
                    8,
                    3,
                    3,
                    &[("padding", "valid"), ("strides", "1")],
                ),
            ),
            (
                "direct_same_stride2",
                run_case(5, 6, 2, 3, 3, 2, &[("padding", "same"), ("strides", "2")]),
            ),
        ];
        let digest = fj_test_utils::fixture_id_from_json(&fixtures)?;
        assert_eq!(
            digest, "4642006de6ba3f3a608d30fb5a7904647f37a9a8d0277894fb7c45b1c8491490",
            "native-f32 conv2d golden output digest changed"
        );
        Ok(())
    }

    #[test]
    fn conv2d_rhs_dilation_matches_reference_and_inflated_kernel() {
        // Atrous conv (rhs_dilation): tap (kh,kw) reads input (oh*s + kh*d, ow*s + kw*d).
        // (1) VALID: bit-identical to a direct dilated f64 reference.
        // (2) SAME : equal (to fp) to a plain conv with the kernel "inflated" by
        //     inserting (d-1) zero rows/cols between taps — the trusted undilated path.
        let (h, w, c_in, c_out, kh, kw) = (12usize, 11usize, 3usize, 5usize, 3usize, 2usize);
        let (dh, dw) = (2usize, 3usize);
        let xf: Vec<f64> = (0..h * w * c_in)
            .map(|i| (i as f64 * 0.013).sin() * 1.7 - 0.3)
            .collect();
        let kf: Vec<f64> = (0..kh * kw * c_in * c_out)
            .map(|i| (i as f64 * 0.019).cos() * 0.8 + 0.2)
            .collect();
        let mk = |dims: Vec<u32>, d: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, d.to_vec()).unwrap())
        };
        let lhs = mk(vec![1, h as u32, w as u32, c_in as u32], &xf);
        let kernel = mk(vec![kh as u32, kw as u32, c_in as u32, c_out as u32], &kf);

        // (1) VALID dilated vs direct reference.
        let eff_kh = (kh - 1) * dh + 1;
        let eff_kw = (kw - 1) * dw + 1;
        let (out_h, out_w) = (h - eff_kh + 1, w - eff_kw + 1);
        let out = eval_conv(
            Primitive::Conv,
            &[lhs.clone(), kernel.clone()],
            &params(&[
                ("padding", "valid"),
                ("strides", "1"),
                ("rhs_dilation", "2,3"),
            ]),
        )
        .unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(
            t.shape.dims,
            vec![1, out_h as u32, out_w as u32, c_out as u32],
            "dilated VALID shape"
        );
        let got: Vec<u64> = t
            .elements
            .iter()
            .map(|l| match l {
                Literal::F64Bits(b) => *b,
                o => panic!("unexpected {o:?}"),
            })
            .collect();
        let mut want = Vec::new();
        for oh in 0..out_h {
            for ow in 0..out_w {
                for co in 0..c_out {
                    let mut acc = 0.0f64;
                    for a in 0..kh {
                        for b in 0..kw {
                            for ci in 0..c_in {
                                let lv = xf[((oh + a * dh) * w + (ow + b * dw)) * c_in + ci];
                                let kv = kf[((a * kw + b) * c_in + ci) * c_out + co];
                                acc += lv * kv;
                            }
                        }
                    }
                    want.push(acc.to_bits());
                }
            }
        }
        assert_eq!(
            got, want,
            "dilated VALID conv must match direct dilated f64 reference"
        );

        // (2) SAME dilated vs inflated-kernel plain conv.
        let mut inflated = vec![0.0f64; eff_kh * eff_kw * c_in * c_out];
        for a in 0..kh {
            for b in 0..kw {
                for ci in 0..c_in {
                    for co in 0..c_out {
                        let src = ((a * kw + b) * c_in + ci) * c_out + co;
                        let dst = (((a * dh) * eff_kw + (b * dw)) * c_in + ci) * c_out + co;
                        inflated[dst] = kf[src];
                    }
                }
            }
        }
        let infl_kernel = mk(
            vec![eff_kh as u32, eff_kw as u32, c_in as u32, c_out as u32],
            &inflated,
        );
        let dilated_same = eval_conv(
            Primitive::Conv,
            &[lhs.clone(), kernel.clone()],
            &params(&[
                ("padding", "same"),
                ("strides", "1"),
                ("rhs_dilation", "2,3"),
            ]),
        )
        .unwrap();
        let plain_same = eval_conv(
            Primitive::Conv,
            &[lhs.clone(), infl_kernel],
            &params(&[("padding", "same"), ("strides", "1")]),
        )
        .unwrap();
        let da = extract_f64_vec(&dilated_same);
        let pa = extract_f64_vec(&plain_same);
        assert_eq!(
            extract_shape(&dilated_same),
            extract_shape(&plain_same),
            "dilated SAME shape == inflated SAME shape"
        );
        assert_eq!(da.len(), pa.len());
        for (i, (&x, &y)) in da.iter().zip(pa.iter()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-12,
                "dilated SAME != inflated at {i}: {x} vs {y}"
            );
        }

        // 1D conv rhs_dilation: bit-identical to a direct dilated 1D reference.
        let d1 = 2usize;
        let x1: Vec<f64> = xf[..w * c_in].to_vec();
        let k1d: Vec<f64> = kf[..kw * c_in * c_out].to_vec();
        let lhs1 = mk(vec![1, w as u32, c_in as u32], &x1);
        let k1 = mk(vec![kw as u32, c_in as u32, c_out as u32], &k1d);
        let out1 = eval_conv(
            Primitive::Conv,
            &[lhs1, k1],
            &params(&[
                ("padding", "valid"),
                ("strides", "1"),
                ("rhs_dilation", "2"),
            ]),
        )
        .unwrap();
        let eff_kw1 = (kw - 1) * d1 + 1;
        let ow1 = w - eff_kw1 + 1;
        let t1 = out1.as_tensor().unwrap();
        assert_eq!(
            t1.shape.dims,
            vec![1, ow1 as u32, c_out as u32],
            "1D dilated shape"
        );
        let got1: Vec<u64> = t1
            .elements
            .iter()
            .map(|l| match l {
                Literal::F64Bits(b) => *b,
                o => panic!("unexpected {o:?}"),
            })
            .collect();
        let mut want1 = Vec::new();
        for ow in 0..ow1 {
            for co in 0..c_out {
                let mut acc = 0.0f64;
                for k in 0..kw {
                    for ci in 0..c_in {
                        acc += x1[(ow + k * d1) * c_in + ci] * k1d[(k * c_in + ci) * c_out + co];
                    }
                }
                want1.push(acc.to_bits());
            }
        }
        assert_eq!(
            got1, want1,
            "1D dilated conv must match direct dilated reference"
        );
    }

    #[test]
    fn conv2d_depthwise_fast_path_matches_general() {
        // Depthwise conv (feature_group_count == c_in, multiplier 1) takes the channel-
        // vectorized fast path. Validate it BIT-FOR-BIT against an INDEPENDENT oracle:
        // a depthwise conv is C single-channel non-grouped convs (im2col path), which
        // accumulate the same kh-outer/kw-inner products in f64. Covers SAME padding
        // (border OOB) and stride 2.
        let mk = |dims: Vec<u32>, d: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, d.to_vec()).unwrap())
        };
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F64Bits(b) => *b,
                    o => panic!("unexpected {o:?}"),
                })
                .collect()
        };
        for &(h, w, c, kh, kw, pad, stride) in &[
            (8usize, 7usize, 6usize, 3usize, 3usize, "valid", 1usize),
            (8, 7, 6, 3, 3, "same", 1),
            (9, 9, 5, 3, 3, "same", 2),
            (6, 6, 4, 2, 2, "valid", 1),
        ] {
            let xf: Vec<f64> = (0..h * w * c)
                .map(|i| (i as f64 * 0.021).sin() * 1.4 - 0.25)
                .collect();
            let kf: Vec<f64> = (0..kh * kw * c)
                .map(|i| (i as f64 * 0.017).cos() * 0.7 + 0.15)
                .collect();
            let strides = stride.to_string();
            let got = eval_conv(
                Primitive::Conv,
                &[
                    mk(vec![1, h as u32, w as u32, c as u32], &xf),
                    mk(vec![kh as u32, kw as u32, 1, c as u32], &kf),
                ],
                &params(&[
                    ("padding", pad),
                    ("strides", &strides),
                    ("feature_group_count", &c.to_string()),
                ]),
            )
            .unwrap();
            let gt = got.as_tensor().unwrap();
            let out_h = gt.shape.dims[1] as usize;
            let out_w = gt.shape.dims[2] as usize;
            let got_bits = bits(&got);
            // Per-channel oracle: single-channel non-grouped conv (im2col path).
            for ch in 0..c {
                let x_ch: Vec<f64> = (0..h * w).map(|p| xf[p * c + ch]).collect();
                let k_ch: Vec<f64> = (0..kh * kw).map(|p| kf[p * c + ch]).collect();
                let ref_out = eval_conv(
                    Primitive::Conv,
                    &[
                        mk(vec![1, h as u32, w as u32, 1], &x_ch),
                        mk(vec![kh as u32, kw as u32, 1, 1], &k_ch),
                    ],
                    &params(&[("padding", pad), ("strides", &strides)]),
                )
                .unwrap();
                let ref_bits = bits(&ref_out);
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let g = got_bits[(oh * out_w + ow) * c + ch];
                        let r = ref_bits[oh * out_w + ow];
                        assert_eq!(
                            g, r,
                            "depthwise fast path != per-channel ref at \
                             ({oh},{ow},ch={ch}) cfg=({h},{w},{c},{kh},{kw},{pad},s{stride})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn conv2d_grouped_dilated_matches_inflated_kernel() {
        // Atrous (rhs_dilation) grouped/depthwise conv2d goes through the SAME fast paths
        // (depthwise + AXPY), which thread the dilation into the tap formula. Validate that
        // dilation handling against the inflation isomorphism: an rhs_dilation=d conv equals
        // a plain conv with the kernel inflated by inserting (d-1) zeros between taps. Covers
        // depthwise (G=Cin) and general grouped (G=3); both go through my fast paths.
        let mk = |dims: Vec<u32>, d: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, d.to_vec()).unwrap())
        };
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F64Bits(b) => *b,
                    o => panic!("unexpected {o:?}"),
                })
                .collect()
        };
        let dil = 2usize;
        // (h, w, cin, cout, kh, kw, G)
        for &(h, w, cin, cout, kh, kw, g) in &[
            (12usize, 11usize, 4usize, 4usize, 3usize, 3usize, 4usize), // depthwise G=Cin
            (12, 11, 6, 9, 3, 2, 3),                                    // general grouped G=3
        ] {
            let rhs_cin = cin / g;
            let xf: Vec<f64> = (0..h * w * cin)
                .map(|i| (i as f64 * 0.021).sin() * 1.4 - 0.25)
                .collect();
            let kf: Vec<f64> = (0..kh * kw * rhs_cin * cout)
                .map(|i| (i as f64 * 0.017).cos() * 0.7 + 0.15)
                .collect();
            // Inflated kernel [kh', kw', rhs_cin, cout], taps at multiples of `dil`, else 0.
            let (kh2, kw2) = ((kh - 1) * dil + 1, (kw - 1) * dil + 1);
            let mut kinf = vec![0.0_f64; kh2 * kw2 * rhs_cin * cout];
            for a in 0..kh {
                for b in 0..kw {
                    for ci in 0..rhs_cin {
                        for co in 0..cout {
                            let src = ((a * kw + b) * rhs_cin + ci) * cout + co;
                            let dst = (((a * dil) * kw2 + (b * dil)) * rhs_cin + ci) * cout + co;
                            kinf[dst] = kf[src];
                        }
                    }
                }
            }
            let x = mk(vec![1, h as u32, w as u32, cin as u32], &xf);
            let dilated = eval_conv(
                Primitive::Conv,
                &[
                    x.clone(),
                    mk(vec![kh as u32, kw as u32, rhs_cin as u32, cout as u32], &kf),
                ],
                &params(&[
                    ("padding", "valid"),
                    ("strides", "1"),
                    ("feature_group_count", &g.to_string()),
                    ("rhs_dilation", &format!("{dil},{dil}")),
                ]),
            )
            .unwrap();
            let inflated = eval_conv(
                Primitive::Conv,
                &[
                    x,
                    mk(
                        vec![kh2 as u32, kw2 as u32, rhs_cin as u32, cout as u32],
                        &kinf,
                    ),
                ],
                &params(&[
                    ("padding", "valid"),
                    ("strides", "1"),
                    ("feature_group_count", &g.to_string()),
                ]),
            )
            .unwrap();
            assert_eq!(
                bits(&dilated),
                bits(&inflated),
                "grouped dilated conv != inflated-kernel reference (G={g})"
            );
        }
    }

    #[test]
    fn conv2d_grouped_axpy_matches_general() {
        // General grouped conv2d (cout_per_group > 1) takes the AXPY fast path. Validate
        // BIT-FOR-BIT against an independent oracle: group g is a non-grouped conv of input
        // channels [g·rhs_c_in..) with kernel slice [:,:,:, g·cpg..), placed into output
        // channels [g·cpg..). Covers SAME padding (border OOB) + stride 2.
        let mk = |dims: Vec<u32>, d: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, d.to_vec()).unwrap())
        };
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F64Bits(b) => *b,
                    o => panic!("unexpected {o:?}"),
                })
                .collect()
        };
        // (h, w, c_in, c_out, kh, kw, G, padding, stride)
        for &(h, w, cin, cout, kh, kw, g, pad, stride) in &[
            (
                8usize, 7usize, 6usize, 9usize, 3usize, 2usize, 3usize, "same", 1usize,
            ),
            (9, 9, 8, 8, 3, 3, 2, "same", 2),
            (7, 8, 6, 12, 3, 3, 3, "valid", 1),
        ] {
            let rhs_cin = cin / g;
            let cpg = cout / g;
            let xf: Vec<f64> = (0..h * w * cin)
                .map(|i| (i as f64 * 0.021).sin() * 1.4 - 0.25)
                .collect();
            // kernel [kh, kw, rhs_cin, cout]
            let kf: Vec<f64> = (0..kh * kw * rhs_cin * cout)
                .map(|i| (i as f64 * 0.017).cos() * 0.7 + 0.15)
                .collect();
            let strides = stride.to_string();
            let got = eval_conv(
                Primitive::Conv,
                &[
                    mk(vec![1, h as u32, w as u32, cin as u32], &xf),
                    mk(vec![kh as u32, kw as u32, rhs_cin as u32, cout as u32], &kf),
                ],
                &params(&[
                    ("padding", pad),
                    ("strides", &strides),
                    ("feature_group_count", &g.to_string()),
                ]),
            )
            .unwrap();
            let gt = got.as_tensor().unwrap();
            let out_h = gt.shape.dims[1] as usize;
            let out_w = gt.shape.dims[2] as usize;
            let got_bits = bits(&got);
            for grp in 0..g {
                // input channels of this group
                let mut x_g = Vec::with_capacity(h * w * rhs_cin);
                for p in 0..h * w {
                    for ci in 0..rhs_cin {
                        x_g.push(xf[p * cin + grp * rhs_cin + ci]);
                    }
                }
                // kernel slice [kh,kw,rhs_cin, cpg] = kf[..., grp*cpg .. grp*cpg+cpg]
                let mut k_g = Vec::with_capacity(kh * kw * rhs_cin * cpg);
                for t in 0..kh * kw * rhs_cin {
                    for j in 0..cpg {
                        k_g.push(kf[t * cout + grp * cpg + j]);
                    }
                }
                let ref_out = eval_conv(
                    Primitive::Conv,
                    &[
                        mk(vec![1, h as u32, w as u32, rhs_cin as u32], &x_g),
                        mk(vec![kh as u32, kw as u32, rhs_cin as u32, cpg as u32], &k_g),
                    ],
                    &params(&[("padding", pad), ("strides", &strides)]),
                )
                .unwrap();
                let ref_bits = bits(&ref_out);
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        for j in 0..cpg {
                            let go = got_bits[(oh * out_w + ow) * cout + grp * cpg + j];
                            let re = ref_bits[(oh * out_w + ow) * cpg + j];
                            assert_eq!(
                                go, re,
                                "grouped AXPY != per-group ref at (oh={oh},ow={ow},g={grp},j={j}) \
                                 cfg=({h},{w},{cin},{cout},G={g},{pad},s{stride})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn conv2d_grouped_native_f32_accum_matches_reference() {
        // The grouped/depthwise F32 path must accumulate in native f32 (XLA parity),
        // not via the older f64-promote path. Compare dense-F32 outputs bit-for-bit
        // against an independent scalar f32 reference in the same output/channel order.
        let mk32 = |dims: Vec<u32>, data: &[f32]| {
            Value::Tensor(TensorValue::new_f32_values(Shape { dims }, data.to_vec()).unwrap())
        };
        let got_bits = |v: &Value| -> Vec<u32> {
            v.as_tensor()
                .unwrap()
                .elements
                .as_f32_slice()
                .expect("grouped f32 conv should return dense f32")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };

        // (h, w, c_in, c_out, kh, kw, G)
        for &(h, w, cin, cout, kh, kw, g) in &[
            (6usize, 5usize, 4usize, 4usize, 3usize, 3usize, 4usize), // depthwise
            (7, 7, 6, 9, 3, 3, 3),                                    // grouped AXPY
            (7, 5, 6, 3, 3, 3, 3),                                    // grouped direct
        ] {
            let rhs_cin = cin / g;
            let cpg = cout / g;
            let xf: Vec<f32> = (0..h * w * cin)
                .map(|i| (i as f32 * 0.031).sin() * 1.25 - 0.4)
                .collect();
            let mut kf: Vec<f32> = (0..kh * kw * rhs_cin * cout)
                .map(|i| (i as f32 * 0.019).cos() * 0.75 + 0.2)
                .collect();
            kf[0] = f32::INFINITY;
            let got = eval_conv(
                Primitive::Conv,
                &[
                    mk32(vec![1, h as u32, w as u32, cin as u32], &xf),
                    mk32(vec![kh as u32, kw as u32, rhs_cin as u32, cout as u32], &kf),
                ],
                &params(&[
                    ("padding", "same"),
                    ("strides", "1"),
                    ("feature_group_count", &g.to_string()),
                ]),
            )
            .unwrap();
            let gt = got.as_tensor().unwrap();
            let (out_h, out_w) = (gt.shape.dims[1] as usize, gt.shape.dims[2] as usize);
            let (pad_top, pad_left) = (kh / 2, kw / 2);
            let mut want = Vec::with_capacity(out_h * out_w * cout);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for co in 0..cout {
                        let in_ch_base = (co / cpg) * rhs_cin;
                        let mut acc = 0.0_f32;
                        for a in 0..kh {
                            let in_h = (oh + a) as isize - pad_top as isize;
                            let h_oob = in_h < 0 || (in_h as usize) >= h;
                            for b in 0..kw {
                                let in_w = (ow + b) as isize - pad_left as isize;
                                let w_oob = in_w < 0 || (in_w as usize) >= w;
                                let oob = h_oob || w_oob;
                                for ci in 0..rhs_cin {
                                    let lhs_val = if oob {
                                        0.0
                                    } else {
                                        xf[((in_h as usize * w + in_w as usize) * cin)
                                            + in_ch_base
                                            + ci]
                                    };
                                    let rhs_val = kf[((a * kw + b) * rhs_cin + ci) * cout + co];
                                    acc += lhs_val * rhs_val;
                                }
                            }
                        }
                        want.push(acc.to_bits());
                    }
                }
            }
            assert_eq!(
                got_bits(&got),
                want,
                "grouped/depthwise F32 conv2d must match scalar f32 accumulation (G={g})"
            );
        }
    }

    #[test]
    fn conv1d_grouped_axpy_matches_general() {
        // General grouped conv1d (cout_per_group > 1) AXPY fast path vs an independent
        // per-group non-grouped 1D conv oracle. Covers SAME padding (border OOB) + stride 2.
        let mk = |dims: Vec<u32>, d: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, d.to_vec()).unwrap())
        };
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F64Bits(b) => *b,
                    o => panic!("unexpected {o:?}"),
                })
                .collect()
        };
        for &(w, cin, cout, kw, g, pad, stride) in &[
            (16usize, 6usize, 9usize, 3usize, 3usize, "same", 1usize),
            (17, 8, 8, 3, 2, "same", 2),
            (14, 6, 12, 3, 3, "valid", 1),
        ] {
            let rhs_cin = cin / g;
            let cpg = cout / g;
            let xf: Vec<f64> = (0..w * cin)
                .map(|i| (i as f64 * 0.021).sin() * 1.4 - 0.25)
                .collect();
            let kf: Vec<f64> = (0..kw * rhs_cin * cout)
                .map(|i| (i as f64 * 0.017).cos() * 0.7 + 0.15)
                .collect();
            let strides = stride.to_string();
            let got = eval_conv(
                Primitive::Conv,
                &[
                    mk(vec![1, w as u32, cin as u32], &xf),
                    mk(vec![kw as u32, rhs_cin as u32, cout as u32], &kf),
                ],
                &params(&[
                    ("padding", pad),
                    ("strides", &strides),
                    ("feature_group_count", &g.to_string()),
                ]),
            )
            .unwrap();
            let gt = got.as_tensor().unwrap();
            let out_w = gt.shape.dims[1] as usize;
            let got_bits = bits(&got);
            for grp in 0..g {
                let mut x_g = Vec::with_capacity(w * rhs_cin);
                for p in 0..w {
                    for ci in 0..rhs_cin {
                        x_g.push(xf[p * cin + grp * rhs_cin + ci]);
                    }
                }
                let mut k_g = Vec::with_capacity(kw * rhs_cin * cpg);
                for t in 0..kw * rhs_cin {
                    for j in 0..cpg {
                        k_g.push(kf[t * cout + grp * cpg + j]);
                    }
                }
                let ref_out = eval_conv(
                    Primitive::Conv,
                    &[
                        mk(vec![1, w as u32, rhs_cin as u32], &x_g),
                        mk(vec![kw as u32, rhs_cin as u32, cpg as u32], &k_g),
                    ],
                    &params(&[("padding", pad), ("strides", &strides)]),
                )
                .unwrap();
                let ref_bits = bits(&ref_out);
                for ow in 0..out_w {
                    for j in 0..cpg {
                        assert_eq!(
                            got_bits[ow * cout + grp * cpg + j],
                            ref_bits[ow * cpg + j],
                            "conv1d grouped AXPY != per-group ref at (ow={ow},g={grp},j={j}) \
                             cfg=({w},{cin},{cout},G={g},{pad},s{stride})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn conv1d_depthwise_fast_path_matches_general() {
        // 1D depthwise (feature_group_count == c_in, mult 1) fast path vs an independent
        // per-channel single-channel non-grouped 1D conv oracle. Covers SAME padding and
        // stride 2.
        let mk = |dims: Vec<u32>, d: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, d.to_vec()).unwrap())
        };
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F64Bits(b) => *b,
                    o => panic!("unexpected {o:?}"),
                })
                .collect()
        };
        for &(w, c, kw, pad, stride) in &[
            (16usize, 6usize, 3usize, "valid", 1usize),
            (16, 6, 3, "same", 1),
            (17, 5, 3, "same", 2),
            (12, 4, 2, "valid", 1),
        ] {
            let xf: Vec<f64> = (0..w * c)
                .map(|i| (i as f64 * 0.021).sin() * 1.4 - 0.25)
                .collect();
            let kf: Vec<f64> = (0..kw * c)
                .map(|i| (i as f64 * 0.017).cos() * 0.7 + 0.15)
                .collect();
            let strides = stride.to_string();
            let got = eval_conv(
                Primitive::Conv,
                &[
                    mk(vec![1, w as u32, c as u32], &xf),
                    mk(vec![kw as u32, 1, c as u32], &kf),
                ],
                &params(&[
                    ("padding", pad),
                    ("strides", &strides),
                    ("feature_group_count", &c.to_string()),
                ]),
            )
            .unwrap();
            let gt = got.as_tensor().unwrap();
            let out_w = gt.shape.dims[1] as usize;
            let got_bits = bits(&got);
            for ch in 0..c {
                let x_ch: Vec<f64> = (0..w).map(|p| xf[p * c + ch]).collect();
                let k_ch: Vec<f64> = (0..kw).map(|p| kf[p * c + ch]).collect();
                let ref_out = eval_conv(
                    Primitive::Conv,
                    &[
                        mk(vec![1, w as u32, 1], &x_ch),
                        mk(vec![kw as u32, 1, 1], &k_ch),
                    ],
                    &params(&[("padding", pad), ("strides", &strides)]),
                )
                .unwrap();
                let ref_bits = bits(&ref_out);
                for ow in 0..out_w {
                    assert_eq!(
                        got_bits[ow * c + ch],
                        ref_bits[ow],
                        "conv1d depthwise fast != per-channel ref at (ow={ow},ch={ch}) \
                         cfg=({w},{c},{kw},{pad},s{stride})"
                    );
                }
            }
        }
    }

    #[test]
    fn conv2d_feature_group_count_matches_ungrouped_groups() {
        // Grouped conv: output channel co (group g = co/(Cout/G)) convolves only
        // input channels [g*Cin/G .. (g+1)*Cin/G) with kernel[:,:,:,co]. Validate
        // against a direct grouped f64 reference (bit-exact). Covers a general group
        // (G=3) and the depthwise case (G=Cin=Cout).
        let mk = |dims: Vec<u32>, d: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, d.to_vec()).unwrap())
        };
        // (h, w, c_in, c_out, kh, kw, G)
        for &(h, w, c_in, c_out, kh, kw, g) in &[
            (8usize, 7usize, 6usize, 9usize, 3usize, 2usize, 3usize), // general grouped
            (6, 6, 4, 4, 3, 3, 4),                                    // depthwise (G=Cin=Cout)
        ] {
            let rhs_c_in = c_in / g;
            let cpg = c_out / g;
            let xf: Vec<f64> = (0..h * w * c_in)
                .map(|i| (i as f64 * 0.021).sin() * 1.4 - 0.25)
                .collect();
            let kf: Vec<f64> = (0..kh * kw * rhs_c_in * c_out)
                .map(|i| (i as f64 * 0.017).cos() * 0.7 + 0.15)
                .collect();
            let out = eval_conv(
                Primitive::Conv,
                &[
                    mk(vec![1, h as u32, w as u32, c_in as u32], &xf),
                    mk(
                        vec![kh as u32, kw as u32, rhs_c_in as u32, c_out as u32],
                        &kf,
                    ),
                ],
                &params(&[
                    ("padding", "valid"),
                    ("strides", "1"),
                    ("feature_group_count", &g.to_string()),
                ]),
            )
            .unwrap();
            let (out_h, out_w) = (h - kh + 1, w - kw + 1);
            let t = out.as_tensor().unwrap();
            assert_eq!(
                t.shape.dims,
                vec![1, out_h as u32, out_w as u32, c_out as u32],
                "grouped conv shape G={g}"
            );
            let got: Vec<u64> = t
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F64Bits(b) => *b,
                    o => panic!("unexpected {o:?}"),
                })
                .collect();
            let mut want = Vec::new();
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for co in 0..c_out {
                        let in_ch_base = (co / cpg) * rhs_c_in;
                        let mut acc = 0.0f64;
                        for a in 0..kh {
                            for b in 0..kw {
                                for ci in 0..rhs_c_in {
                                    let lv =
                                        xf[((oh + a) * w + (ow + b)) * c_in + (in_ch_base + ci)];
                                    let kv = kf[((a * kw + b) * rhs_c_in + ci) * c_out + co];
                                    acc += lv * kv;
                                }
                            }
                        }
                        want.push(acc.to_bits());
                    }
                }
            }
            assert_eq!(
                got, want,
                "grouped conv G={g} must match direct grouped reference"
            );
        }

        // 1D grouped conv: bit-identical to a direct grouped 1D reference, for a
        // general group (G=3) and the depthwise case (G=Cin=Cout).
        for &(w, c_in, c_out, kw, g) in &[(8usize, 6usize, 9usize, 3usize, 3usize), (6, 4, 4, 3, 4)]
        {
            let rhs_c_in = c_in / g;
            let cpg = c_out / g;
            let x1: Vec<f64> = (0..w * c_in)
                .map(|i| (i as f64 * 0.023).sin() * 1.1 - 0.2)
                .collect();
            let k1d: Vec<f64> = (0..kw * rhs_c_in * c_out)
                .map(|i| (i as f64 * 0.019).cos() * 0.6 + 0.1)
                .collect();
            let out = eval_conv(
                Primitive::Conv,
                &[
                    mk(vec![1, w as u32, c_in as u32], &x1),
                    mk(vec![kw as u32, rhs_c_in as u32, c_out as u32], &k1d),
                ],
                &params(&[
                    ("padding", "valid"),
                    ("strides", "1"),
                    ("feature_group_count", &g.to_string()),
                ]),
            )
            .unwrap();
            let ow = w - kw + 1;
            let t = out.as_tensor().unwrap();
            assert_eq!(
                t.shape.dims,
                vec![1, ow as u32, c_out as u32],
                "1D grouped shape G={g}"
            );
            let got: Vec<u64> = t
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F64Bits(b) => *b,
                    o => panic!("unexpected {o:?}"),
                })
                .collect();
            let mut want = Vec::new();
            for ow_i in 0..ow {
                for co in 0..c_out {
                    let in_ch_base = (co / cpg) * rhs_c_in;
                    let mut acc = 0.0f64;
                    for k in 0..kw {
                        for ci in 0..rhs_c_in {
                            acc += x1[(ow_i + k) * c_in + (in_ch_base + ci)]
                                * k1d[(k * rhs_c_in + ci) * c_out + co];
                        }
                    }
                    want.push(acc.to_bits());
                }
            }
            assert_eq!(
                got, want,
                "1D grouped conv G={g} must match direct reference"
            );
        }
    }

    /// f32 conv2d now emits DENSE f32 storage so the output feeds downstream f32
    /// elementwise (bias-add/activation) densely. Verify `as_f32_slice`-backed.
    #[test]
    fn f32_conv2d_emits_dense_f32_storage() {
        let (h, w, c_in, c_out, kh, kw) = (20usize, 20usize, 4usize, 8usize, 3usize, 3usize);
        let xf: Vec<f32> = (0..h * w * c_in)
            .map(|i| (i as f32 * 0.011).sin())
            .collect();
        let kf: Vec<f32> = (0..kh * kw * c_in * c_out)
            .map(|i| (i as f32 * 0.017).cos())
            .collect();
        let mk32 = |dims: Vec<u32>, data: &[f32]| {
            Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape { dims },
                    data.iter().map(|&v| Literal::from_f32(v)).collect(),
                )
                .unwrap(),
            )
        };
        let out = eval_conv(
            Primitive::Conv,
            &[
                mk32(vec![1, h as u32, w as u32, c_in as u32], &xf),
                mk32(vec![kh as u32, kw as u32, c_in as u32, c_out as u32], &kf),
            ],
            &params(&[("padding", "valid"), ("strides", "1")]),
        )
        .unwrap();
        let Value::Tensor(t) = out else {
            panic!("expected tensor")
        };
        assert_eq!(t.dtype, DType::F32);
        assert!(
            t.elements.as_f32_slice().is_some(),
            "f32 conv2d output must be dense-f32-backed (not boxed Literals)"
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_f32_conv2d() {
        use std::time::Instant;
        let run = |b: usize, h: usize, c_in: usize, c_out: usize, k: usize| {
            let x: Vec<f32> = (0..b * h * h * c_in)
                .map(|i| (i % 7) as f32 * 0.5)
                .collect();
            let ker: Vec<f32> = (0..k * k * c_in * c_out)
                .map(|i| (i % 5) as f32 * 0.25)
                .collect();
            let mk = |dims: Vec<u32>, data: &[f32]| {
                Value::Tensor(
                    TensorValue::new(
                        DType::F32,
                        Shape { dims },
                        data.iter().map(|&v| Literal::from_f32(v)).collect(),
                    )
                    .unwrap(),
                )
            };
            let inputs = [
                mk(vec![b as u32, h as u32, h as u32, c_in as u32], &x),
                mk(vec![k as u32, k as u32, c_in as u32, c_out as u32], &ker),
            ];
            let p = params(&[("padding", "valid"), ("strides", "1")]);
            let _ = eval_conv(Primitive::Conv, &inputs, &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..15 {
                let t = Instant::now();
                let _ = eval_conv(Primitive::Conv, &inputs, &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!(
                "BENCH f32 conv2d [{b},{h},{h},{c_in}]*[{k},{k},{c_in},{c_out}]: {:.4}ms",
                best * 1e3
            );
        };
        run(8, 32, 16, 32, 3);
        run(4, 28, 32, 64, 3);
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_f32_grouped_conv_accum() {
        use std::time::Instant;

        fn best_ms(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..9 {
                let start = Instant::now();
                f();
                best = best.min(start.elapsed().as_secs_f64());
            }
            best * 1e3
        }

        let run_1d = |label: &str, width: usize, c_in: usize, c_out: usize, kw: usize, g: usize| {
            let rhs_c_in = c_in / g;
            let x: Vec<f32> = (0..width * c_in)
                .map(|i| (i as f32 * 0.0011).sin())
                .collect();
            let ker: Vec<f32> = (0..kw * rhs_c_in * c_out)
                .map(|i| (i as f32 * 0.0023).cos())
                .collect();
            let lhs = Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: vec![1, width as u32, c_in as u32],
                    },
                    x,
                )
                .unwrap(),
            );
            let rhs = Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: vec![kw as u32, rhs_c_in as u32, c_out as u32],
                    },
                    ker,
                )
                .unwrap(),
            );
            let p = params(&[
                ("padding", "valid"),
                ("strides", "1"),
                ("feature_group_count", &g.to_string()),
            ]);
            let ms = best_ms(|| {
                let _ = eval_conv(Primitive::Conv, &[lhs.clone(), rhs.clone()], &p).unwrap();
            });
            println!(
                "BENCH f32 {label} conv1d [1,{width},{c_in}]*[{kw},{rhs_c_in},{c_out}] g={g}: {ms:.4}ms"
            );
        };

        let run_2d = |label: &str,
                      h: usize,
                      w: usize,
                      c_in: usize,
                      c_out: usize,
                      k: usize,
                      g: usize| {
            let rhs_c_in = c_in / g;
            let x: Vec<f32> = (0..h * w * c_in)
                .map(|i| (i as f32 * 0.0011).sin())
                .collect();
            let ker: Vec<f32> = (0..k * k * rhs_c_in * c_out)
                .map(|i| (i as f32 * 0.0023).cos())
                .collect();
            let lhs = Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: vec![1, h as u32, w as u32, c_in as u32],
                    },
                    x,
                )
                .unwrap(),
            );
            let rhs = Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: vec![k as u32, k as u32, rhs_c_in as u32, c_out as u32],
                    },
                    ker,
                )
                .unwrap(),
            );
            let p = params(&[
                ("padding", "valid"),
                ("strides", "1"),
                ("feature_group_count", &g.to_string()),
            ]);
            let ms = best_ms(|| {
                let _ = eval_conv(Primitive::Conv, &[lhs.clone(), rhs.clone()], &p).unwrap();
            });
            println!(
                "BENCH f32 {label} conv2d [1,{h},{w},{c_in}]*[{k},{k},{rhs_c_in},{c_out}] g={g}: {ms:.4}ms"
            );
        };

        run_1d("depthwise", 1024, 256, 256, 5, 256);
        run_1d("grouped", 512, 256, 256, 3, 32);
        run_2d("depthwise", 56, 56, 128, 128, 3, 128);
        run_2d("grouped", 28, 28, 256, 256, 3, 32);
    }

    #[test]
    fn conv1d_real_same_padding_zero_pads_like_valid_on_padded_input() {
        // The real conv1d paths must treat out-of-bounds 'same'-padding taps as
        // 0·w (XLA zero-padding), matching conv2d. Metamorphic proof: 'same'
        // padding on X is bit-identical to 'valid' padding on X explicitly
        // zero-bordered — both sum every (k,ci) tap (pad positions = 0) by the
        // same kernel in the same order. Signed-zero / infinity inputs make the
        // skip-vs-zero-pad difference observable (-0.0 vs +0.0 accumulator).
        let (width, c_in, c_out) = (3usize, 2usize, 2usize);
        let kw = 3usize; // odd ⇒ symmetric pad of 1 each side @ stride 1

        let x: Vec<f64> = vec![-0.0, 1.5, f64::INFINITY, -2.0, -0.0, 3.0]; // width*c_in
        let kdata: Vec<f64> = (0..kw * c_in * c_out)
            .map(|i| ((i as f64) * 0.3).cos() - 0.5)
            .collect();
        let mk = |dims: Vec<u32>, data: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, data.to_vec()).unwrap())
        };
        let kernel = mk(vec![kw as u32, c_in as u32, c_out as u32], &kdata);

        let same = eval_conv(
            Primitive::Conv,
            &[mk(vec![1, width as u32, c_in as u32], &x), kernel.clone()],
            &params(&[("padding", "same"), ("strides", "1")]),
        )
        .unwrap();

        let pw = width + 2;
        let mut xp = vec![0.0_f64; pw * c_in];
        for col in 0..width {
            for ci in 0..c_in {
                xp[(col + 1) * c_in + ci] = x[col * c_in + ci];
            }
        }
        let valid = eval_conv(
            Primitive::Conv,
            &[mk(vec![1, pw as u32, c_in as u32], &xp), kernel],
            &params(&[("padding", "valid"), ("strides", "1")]),
        )
        .unwrap();

        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };
        assert_eq!(
            same.as_tensor().unwrap().shape.dims,
            valid.as_tensor().unwrap().shape.dims,
            "same(X) and valid(zero-padded X) must share shape"
        );
        assert_eq!(
            bits(&same),
            bits(&valid),
            "conv1d 'same' must zero-pad OOB taps bit-for-bit like valid-on-padded"
        );
    }

    #[test]
    fn conv_lhs_dilation_matches_explicitly_dilated_input() {
        // lhs_dilation (transposed conv): inserting (L-1) zeros between input elements
        // then convolving must equal a plain conv over an INDEPENDENTLY hand-built
        // zero-dilated input (bit-for-bit, finite f64). Covers 2D (VALID + SAME) and 1D.
        let mk = |dims: Vec<u32>, d: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, d.to_vec()).unwrap())
        };
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F64Bits(b) => *b,
                    o => panic!("unexpected {o:?}"),
                })
                .collect()
        };

        // ── 2D ──
        let (h, w, c_in, c_out, kh, kw, lh, lw) = (5usize, 4, 2, 3, 2, 2, 2, 3);
        let xf: Vec<f64> = (0..h * w * c_in)
            .map(|i| (i as f64 * 0.031).sin() * 1.3 - 0.2)
            .collect();
        let kf: Vec<f64> = (0..kh * kw * c_in * c_out)
            .map(|i| (i as f64 * 0.013).cos() * 0.7 + 0.1)
            .collect();
        let kernel = mk(vec![kh as u32, kw as u32, c_in as u32, c_out as u32], &kf);
        // Hand-built dilated input [1, eff_h, eff_w, c_in].
        let (eff_h, eff_w) = ((h - 1) * lh + 1, (w - 1) * lw + 1);
        let mut dilated = vec![0.0f64; eff_h * eff_w * c_in];
        for ih in 0..h {
            for iw in 0..w {
                for ci in 0..c_in {
                    dilated[((ih * lh) * eff_w + (iw * lw)) * c_in + ci] =
                        xf[(ih * w + iw) * c_in + ci];
                }
            }
        }
        let lhs2 = mk(vec![1, h as u32, w as u32, c_in as u32], &xf);
        let dlhs2 = mk(vec![1, eff_h as u32, eff_w as u32, c_in as u32], &dilated);
        for pad in ["valid", "same"] {
            let via_dil = eval_conv(
                Primitive::Conv,
                &[lhs2.clone(), kernel.clone()],
                &params(&[("padding", pad), ("strides", "1"), ("lhs_dilation", "2,3")]),
            )
            .unwrap();
            let via_plain = eval_conv(
                Primitive::Conv,
                &[dlhs2.clone(), kernel.clone()],
                &params(&[("padding", pad), ("strides", "1")]),
            )
            .unwrap();
            assert_eq!(
                extract_shape(&via_dil),
                extract_shape(&via_plain),
                "2D lhs_dilation {pad} shape"
            );
            assert_eq!(
                bits(&via_dil),
                bits(&via_plain),
                "2D lhs_dilation {pad} values"
            );
        }

        // ── 1D ──
        let (w1, ci1, co1, kw1, l1) = (6usize, 2, 3, 2, 3);
        let x1: Vec<f64> = (0..w1 * ci1)
            .map(|i| (i as f64 * 0.027).cos() * 0.9 - 0.1)
            .collect();
        let k1: Vec<f64> = (0..kw1 * ci1 * co1)
            .map(|i| (i as f64 * 0.019).sin() * 0.5 + 0.2)
            .collect();
        let kern1 = mk(vec![kw1 as u32, ci1 as u32, co1 as u32], &k1);
        let eff_w1 = (w1 - 1) * l1 + 1;
        let mut dil1 = vec![0.0f64; eff_w1 * ci1];
        for iw in 0..w1 {
            for ci in 0..ci1 {
                dil1[(iw * l1) * ci1 + ci] = x1[iw * ci1 + ci];
            }
        }
        let via_dil = eval_conv(
            Primitive::Conv,
            &[mk(vec![1, w1 as u32, ci1 as u32], &x1), kern1.clone()],
            &params(&[
                ("padding", "valid"),
                ("strides", "1"),
                ("lhs_dilation", "3"),
            ]),
        )
        .unwrap();
        let via_plain = eval_conv(
            Primitive::Conv,
            &[mk(vec![1, eff_w1 as u32, ci1 as u32], &dil1), kern1],
            &params(&[("padding", "valid"), ("strides", "1")]),
        )
        .unwrap();
        assert_eq!(
            extract_shape(&via_dil),
            extract_shape(&via_plain),
            "1D lhs_dilation shape"
        );
        assert_eq!(bits(&via_dil), bits(&via_plain), "1D lhs_dilation values");
    }

    #[test]
    fn conv_rejects_unimplemented_dilation_and_grouping() {
        // Unsupported conv_general_dilated params must fail loudly rather than be
        // silently ignored. rhs_dilation, lhs_dilation, feature_group_count (1D+2D),
        // and batch_group_count are all supported now; what stays rejected here is a
        // multi-value rhs_dilation for 1D (one spatial dim) and a 1D feature_group_count
        // with an inconsistent kernel (channel mismatch). (batch_group_count is covered
        // by conv_batch_group_count_matches_per_group_decomposition.)
        let mk = |dims: Vec<u32>, data: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, data.to_vec()).unwrap())
        };
        let lhs = mk(vec![1, 5, 2], &[0.0; 10]); // [N=1, W=5, Cin=2]
        let rhs = mk(vec![3, 2, 2], &[0.0; 12]); // [K=3, Cin=2, Cout=2]
        for (key, val) in [
            ("rhs_dilation", "1,2"),
            ("feature_group_count", "2"),
        ] {
            let err = eval_conv(
                Primitive::Conv,
                &[lhs.clone(), rhs.clone()],
                &params(&[("padding", "valid"), (key, val)]),
            )
            .expect_err(&format!("conv must reject {key}={val}"));
            assert!(
                matches!(err, EvalError::Unsupported { .. }),
                "conv {key}={val} should be Unsupported, got {err:?}"
            );
        }
        // Sanity: the default (all no-op) params still succeed.
        eval_conv(
            Primitive::Conv,
            &[lhs, rhs],
            &params(&[
                ("padding", "valid"),
                ("strides", "1"),
                ("rhs_dilation", "1"),
                ("feature_group_count", "1"),
            ]),
        )
        .expect("conv with default dilation/grouping must still succeed");
    }

    #[test]
    fn conv_batch_group_count_matches_per_group_decomposition() {
        let mkf = |dims: Vec<u32>, data: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, data.to_vec()).unwrap())
        };
        let vals = |v: &Value| -> Vec<f64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap())
                .collect()
        };

        // --- Hand-computed 1D oracle (independent of the slice/concat impl) ---
        // batch_group_count=2: group g uses batch block g and out-channel block g
        // (group-major), output batch = N/2 = 1, output features = Cout = 2.
        // lhs [N=2, W=4, Cin=1]: batch0=[1,2,3,4], batch1=[10,20,30,40].
        // rhs [K=2, Cin=1, Cout=2]: ch0 kernel=[1,1], ch1 kernel=[1,-1].
        let lhs1 = mkf(vec![2, 4, 1], &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let rhs1 = mkf(vec![2, 1, 2], &[1.0, 1.0, 1.0, -1.0]);
        let got = eval_conv(
            Primitive::Conv,
            &[lhs1.clone(), rhs1.clone()],
            &params(&[("padding", "valid"), ("strides", "1"), ("batch_group_count", "2")]),
        )
        .expect("batch_group_count=2 must be supported");
        assert_eq!(got.as_tensor().unwrap().shape.dims, vec![1, 3, 2], "bgc out shape");
        // out[0,w,0] = batch0 ⊛ [1,1]; out[0,w,1] = batch1 ⊛ [1,-1]. Row-major [m,Wout,Cout].
        let expect = vec![3.0, -10.0, 5.0, -10.0, 7.0, -10.0];
        assert_eq!(vals(&got), expect, "batch_group_count 1D hand reference");

        // --- 2D: param path must equal explicit per-group conv + channel concat ---
        // lhs [N=4, H=3, W=3, Cin=2], rhs [KH=2, KW=2, Cin=2, Cout=6], G=2 → m=2, k=3.
        let (n, h, w, ci, kh, kw, cout, g) = (4, 3, 3, 2, 2, 2, 6, 2usize);
        let lhs_data: Vec<f64> =
            (0..n * h * w * ci).map(|i| (i % 13) as f64 - 6.0).collect();
        let rhs_data: Vec<f64> =
            (0..kh * kw * ci * cout).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let lhs2 = mkf(vec![n as u32, h as u32, w as u32, ci as u32], &lhs_data);
        let rhs2 = mkf(vec![kh as u32, kw as u32, ci as u32, cout as u32], &rhs_data);
        let p2 = params(&[("padding", "valid"), ("strides", "1"), ("batch_group_count", "2")]);
        let via_param = eval_conv(Primitive::Conv, &[lhs2.clone(), rhs2.clone()], &p2).unwrap();

        let m = n / g;
        let k = cout / g;
        let slice = |t: &Value, axis: usize, s: usize, l: usize| {
            let tv = t.as_tensor().unwrap();
            let rank = tv.shape.rank();
            let st: Vec<String> = (0..rank).map(|d| if d == axis { s } else { 0 }.to_string()).collect();
            let li: Vec<String> = (0..rank)
                .map(|d| if d == axis { l } else { tv.shape.dims[d] as usize }.to_string())
                .collect();
            eval_slice(
                &[t.clone()],
                &BTreeMap::from([
                    ("start_indices".to_owned(), st.join(",")),
                    ("limit_indices".to_owned(), li.join(",")),
                ]),
            )
            .unwrap()
        };
        let plain = params(&[("padding", "valid"), ("strides", "1")]);
        let mut groups = Vec::new();
        for go in 0..g {
            let lg = slice(&lhs2, 0, go * m, (go + 1) * m);
            let rg = slice(&rhs2, 3, go * k, (go + 1) * k); // rhs channel axis = last (3)
            groups.push(eval_conv(Primitive::Conv, &[lg, rg], &plain).unwrap());
        }
        let via_decomp =
            eval_concatenate(&groups, &BTreeMap::from([("dimension".to_owned(), "3".to_owned())]))
                .unwrap();
        assert_eq!(
            via_param.as_tensor().unwrap().shape.dims,
            vec![m as u32, (h - kh + 1) as u32, (w - kw + 1) as u32, cout as u32],
            "bgc 2D out shape"
        );
        assert_eq!(vals(&via_param), vals(&via_decomp), "batch_group_count 2D == per-group decomp");

        // --- Divisibility errors (mirror jax) ---
        // N=3 not divisible by G=2.
        let bad_n = mkf(vec![3, 4, 1], &[0.0; 12]);
        assert!(matches!(
            eval_conv(
                Primitive::Conv,
                &[bad_n, rhs1.clone()],
                &params(&[("padding", "valid"), ("batch_group_count", "2")]),
            ),
            Err(EvalError::Unsupported { .. })
        ));
        // batch_group_count and feature_group_count both > 1 is rejected.
        assert!(matches!(
            eval_conv(
                Primitive::Conv,
                &[lhs1, rhs1],
                &params(&[
                    ("padding", "valid"),
                    ("batch_group_count", "2"),
                    ("feature_group_count", "2"),
                ]),
            ),
            Err(EvalError::Unsupported { .. })
        ));
    }

    #[test]
    fn conv_1d_parallel_dense_matches_literal_bits() {
        let (batch, width, c_in) = (1usize, 1024usize, 16usize);
        let (kernel_w, c_out) = (5usize, 32usize);
        let output_elems = batch * width * c_out;
        let ops = output_elems * kernel_w * c_in;
        assert!(
            conv_morsel_threads(output_elems, ops) > 1,
            "test must exercise the threaded dense conv path"
        );

        let lhs_data: Vec<f64> = (0..batch * width * c_in)
            .map(|i| match i % 257 {
                0 => -0.0,
                1 => 0.0,
                2 => f64::INFINITY,
                3 => f64::NEG_INFINITY,
                _ => ((i as f64) * 0.011).sin(),
            })
            .collect();
        let rhs_data: Vec<f64> = (0..kernel_w * c_in * c_out)
            .map(|i| match i % 193 {
                0 => -0.0,
                1 => f64::from_bits(0x7ff8_0000_0000_0001),
                _ => ((i as f64) * 0.019).cos(),
            })
            .collect();
        let mk = |data: &[f64], dims: Vec<u32>, dense: bool| {
            if dense {
                Value::Tensor(TensorValue::new_f64_values(Shape { dims }, data.to_vec()).unwrap())
            } else {
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        Shape { dims },
                        data.iter().copied().map(Literal::from_f64).collect(),
                    )
                    .unwrap(),
                )
            }
        };
        // Canonicalize NaN: the im2col+GEMM path and the direct-loop path both
        // produce NaN for Inf−Inf / 0·Inf taps but with different IEEE payloads
        // (e.g. x86 GEMM → 0xfff8…, the scalar loop → 0x7ff8…1). NaN payloads are
        // unspecified by IEEE and not part of JAX/XLA parity, so compare them as a
        // single canonical NaN; every finite value still compares bit-for-bit.
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| {
                    let x = l.as_f64().unwrap();
                    if x.is_nan() {
                        0x7ff8_0000_0000_0000
                    } else {
                        x.to_bits()
                    }
                })
                .collect()
        };

        let lhs_dims = vec![batch as u32, width as u32, c_in as u32];
        let rhs_dims = vec![kernel_w as u32, c_in as u32, c_out as u32];
        for (padding, stride) in [("same", "1"), ("same_lower", "1"), ("valid", "2")] {
            let p = params(&[("padding", padding), ("strides", stride)]);
            let dense = eval_conv(
                Primitive::Conv,
                &[
                    mk(&lhs_data, lhs_dims.clone(), true),
                    mk(&rhs_data, rhs_dims.clone(), true),
                ],
                &p,
            )
            .unwrap();
            let literal = eval_conv(
                Primitive::Conv,
                &[
                    mk(&lhs_data, lhs_dims.clone(), false),
                    mk(&rhs_data, rhs_dims.clone(), false),
                ],
                &p,
            )
            .unwrap();
            assert_eq!(
                dense.as_tensor().unwrap().shape.dims,
                literal.as_tensor().unwrap().shape.dims,
                "conv1d shape for {padding}/{stride}"
            );
            assert_eq!(
                bits(&dense),
                bits(&literal),
                "conv1d bits for {padding}/{stride}"
            );
        }
    }

    #[test]
    fn conv_2d_im2col_dense_matches_literal_bits() {
        // Size chosen so the dense F64 path takes im2col + GEMM (ops >=
        // CONV_IM2COL_MIN_OPS) while the Vec<Literal> path takes the direct
        // loop — comparing their output bits proves im2col == direct.
        let (batch, height, width, c_in) = (2usize, 16usize, 16usize, 8usize);
        let (kernel_h, kernel_w, c_out) = (3usize, 3usize, 16usize);
        let out_elems = batch * height * width * c_out; // SAME padding keeps H,W
        assert!(
            out_elems * kernel_h * kernel_w * c_in >= CONV_IM2COL_MIN_OPS,
            "test must exercise the im2col path"
        );

        // Edge values in LHS (signed zero / inf / NaN); kernel stays finite so
        // the out-of-bounds zero taps remain exact no-ops (0·w == 0).
        let lhs_data: Vec<f64> = (0..batch * height * width * c_in)
            .map(|i| match i % 211 {
                0 => -0.0,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => f64::from_bits(0x7ff8_0000_0000_0001),
                _ => ((i as f64) * 0.013).sin() * 2.0,
            })
            .collect();
        let rhs_data: Vec<f64> = (0..kernel_h * kernel_w * c_in * c_out)
            .map(|i| ((i as f64) * 0.017).cos())
            .collect();

        let mk = |data: &[f64], dims: Vec<u32>, dense: bool| {
            if dense {
                Value::Tensor(TensorValue::new_f64_values(Shape { dims }, data.to_vec()).unwrap())
            } else {
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        Shape { dims },
                        data.iter().copied().map(Literal::from_f64).collect(),
                    )
                    .unwrap(),
                )
            }
        };
        // Canonicalize NaN — see conv_1d_parallel_dense_matches_literal_bits: the
        // im2col+GEMM and direct-loop paths produce NaN with different IEEE payloads
        // for Inf−Inf / 0·Inf taps; payloads are not part of JAX/XLA parity. Finite
        // values still compare bit-for-bit.
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| {
                    let x = l.as_f64().unwrap();
                    if x.is_nan() {
                        0x7ff8_0000_0000_0000
                    } else {
                        x.to_bits()
                    }
                })
                .collect()
        };

        let lhs_dims = vec![batch as u32, height as u32, width as u32, c_in as u32];
        let rhs_dims = vec![kernel_h as u32, kernel_w as u32, c_in as u32, c_out as u32];
        for (padding, stride) in [("same", "1"), ("valid", "1"), ("valid", "2")] {
            let p = params(&[("padding", padding), ("strides", stride)]);
            let dense = eval_conv(
                Primitive::Conv,
                &[
                    mk(&lhs_data, lhs_dims.clone(), true),
                    mk(&rhs_data, rhs_dims.clone(), true),
                ],
                &p,
            )
            .unwrap();
            let literal = eval_conv(
                Primitive::Conv,
                &[
                    mk(&lhs_data, lhs_dims.clone(), false),
                    mk(&rhs_data, rhs_dims.clone(), false),
                ],
                &p,
            )
            .unwrap();
            assert_eq!(
                dense.as_tensor().unwrap().shape.dims,
                literal.as_tensor().unwrap().shape.dims,
                "conv2d shape for {padding}/{stride}"
            );
            assert_eq!(
                bits(&dense),
                bits(&literal),
                "conv2d im2col vs direct bits for {padding}/{stride}"
            );
        }
    }

    #[test]
    fn conv2d_complex_same_padding_zero_pads_like_valid_on_padded_input() {
        // The complex conv2d path must treat out-of-bounds 'same'-padding taps as
        // 0+0i (XLA zero-padding), not skip them. Metamorphic proof: 'same' padding
        // on X must be bit-identical to 'valid' padding on X explicitly zero-bordered
        // — both multiply every (kh,kw,ci) tap (pad positions = 0+0i) by the same
        // kernel in the same order. The signed-zero / infinity inputs make the
        // skip-vs-zero-pad difference observable (a `continue`-skip leaves the
        // accumulator at -0.0 where zero-padding lands +0.0).
        let (h, w, c_in, c_out) = (2usize, 2usize, 1usize, 1usize);
        let (kh, kw) = (3usize, 3usize); // odd ⇒ symmetric pad of 1 each side @ stride 1

        let x: Vec<(f64, f64)> = vec![(-0.0, 0.0), (1.5, -2.0), (f64::INFINITY, -0.0), (-3.0, 0.5)];
        // Negative real/imag kernel parts so 0·w yields signed-zero sub-terms.
        let kdata: Vec<(f64, f64)> = (0..kh * kw * c_in * c_out)
            .map(|i| {
                let s = i as f64;
                ((s * 0.3).cos() - 0.5, (s * 0.7).sin() - 0.5)
            })
            .collect();

        let mk_c = |dims: Vec<u32>, data: &[(f64, f64)]| {
            Value::Tensor(
                TensorValue::new(
                    DType::Complex128,
                    Shape { dims },
                    data.iter()
                        .map(|&(re, im)| Literal::Complex128Bits(re.to_bits(), im.to_bits()))
                        .collect(),
                )
                .unwrap(),
            )
        };
        let kernel = mk_c(
            vec![kh as u32, kw as u32, c_in as u32, c_out as u32],
            &kdata,
        );

        let same = eval_conv(
            Primitive::Conv,
            &[
                mk_c(vec![1, h as u32, w as u32, c_in as u32], &x),
                kernel.clone(),
            ],
            &params(&[("padding", "same"), ("strides", "1")]),
        )
        .unwrap();

        // X explicitly zero-bordered by 1 on every side, then 'valid'.
        let (ph, pw) = (h + 2, w + 2);
        let mut xp = vec![(0.0_f64, 0.0_f64); ph * pw * c_in];
        for r in 0..h {
            for col in 0..w {
                for ci in 0..c_in {
                    xp[((r + 1) * pw + (col + 1)) * c_in + ci] = x[(r * w + col) * c_in + ci];
                }
            }
        }
        let valid = eval_conv(
            Primitive::Conv,
            &[
                mk_c(vec![1, ph as u32, pw as u32, c_in as u32], &xp),
                kernel,
            ],
            &params(&[("padding", "valid"), ("strides", "1")]),
        )
        .unwrap();

        let bits = |v: &Value| -> Vec<(u64, u64)> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Complex128Bits(re, im) => (*re, *im),
                    other => panic!("expected complex literal, got {other:?}"),
                })
                .collect()
        };
        assert_eq!(
            same.as_tensor().unwrap().shape.dims,
            valid.as_tensor().unwrap().shape.dims,
            "same(X) and valid(zero-padded X) must share shape"
        );
        assert_eq!(
            bits(&same),
            bits(&valid),
            "complex conv2d 'same' must zero-pad OOB taps bit-for-bit like valid-on-padded"
        );
    }

    // ── Reshape ──

    #[test]
    fn reshape_1d_to_2d() {
        let x = v_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = params(&[("new_shape", "2,3")]);
        let result = eval_reshape(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![2, 3]);
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reshape_2d_to_1d() {
        let x = mat_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = params(&[("new_shape", "6")]);
        let result = eval_reshape(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![6]);
    }

    #[test]
    fn metadata_reshape_ops_preserve_dense_storage_and_values() {
        // reshape/squeeze/expand_dims/split are metadata-only — they must keep the
        // dense backing (NOT re-box into Vec<Literal>) and leave element bits
        // unchanged. Verify against dense f32 and i64 inputs for each op.
        let lits = |v: &Value| v.as_tensor().unwrap().elements.to_vec();

        let f32d: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
        let i64d: Vec<i64> = (0..24).map(|i| i as i64 - 7).collect();
        let f32_src = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: vec![4, 6] }, f32d.clone()).unwrap(),
        );
        let i64_src = Value::Tensor(
            TensorValue::new_i64_values(Shape { dims: vec![4, 6] }, i64d.clone()).unwrap(),
        );
        let expect_f32: Vec<Literal> = f32d.iter().copied().map(Literal::from_f32).collect();
        let expect_i64: Vec<Literal> = i64d.iter().copied().map(Literal::I64).collect();

        // reshape [4,6] -> [2,12]
        let r = eval_reshape(
            std::slice::from_ref(&f32_src),
            &params(&[("new_shape", "2,12")]),
        )
        .unwrap();
        assert_eq!(extract_shape(&r), vec![2, 12]);
        assert!(
            r.as_tensor().unwrap().elements.as_f32_slice().is_some(),
            "reshape stays dense f32"
        );
        assert_eq!(lits(&r), expect_f32, "reshape values");
        let r = eval_reshape(
            std::slice::from_ref(&i64_src),
            &params(&[("new_shape", "24")]),
        )
        .unwrap();
        assert!(
            r.as_tensor().unwrap().elements.as_i64_slice().is_some(),
            "reshape stays dense i64"
        );
        assert_eq!(lits(&r), expect_i64, "reshape i64 values");

        // squeeze: [1,24] -> [24]
        let f32_1x24 = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: vec![1, 24] }, f32d.clone()).unwrap(),
        );
        let s = eval_squeeze(
            std::slice::from_ref(&f32_1x24),
            &params(&[("dimensions", "0")]),
        )
        .unwrap();
        assert_eq!(extract_shape(&s), vec![24]);
        assert!(
            s.as_tensor().unwrap().elements.as_f32_slice().is_some(),
            "squeeze stays dense"
        );
        assert_eq!(lits(&s), expect_f32, "squeeze values");

        // expand_dims: [4,6] -> [4,1,6]
        let e =
            eval_expand_dims(std::slice::from_ref(&f32_src), &params(&[("axis", "1")])).unwrap();
        assert_eq!(extract_shape(&e), vec![4, 1, 6]);
        assert!(
            e.as_tensor().unwrap().elements.as_f32_slice().is_some(),
            "expand_dims stays dense"
        );
        assert_eq!(lits(&e), expect_f32, "expand_dims values");

        // split [4,6] into 2 sections on axis 0 -> [2,2,6]
        let sp = eval_split(
            std::slice::from_ref(&f32_src),
            &params(&[("axis", "0"), ("num_sections", "2")]),
        )
        .unwrap();
        assert_eq!(extract_shape(&sp), vec![2, 2, 6]);
        assert!(
            sp.as_tensor().unwrap().elements.as_f32_slice().is_some(),
            "split stays dense"
        );
        assert_eq!(lits(&sp), expect_f32, "split values");
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reshape_metadata_clone_vs_to_vec() {
        use std::time::Instant;
        // Direct same-binary A/B of the lever: the OLD path materialized the whole
        // buffer (to_vec -> Vec<Literal>) and re-analyzed via TensorValue::new; the
        // NEW path clones the backing buffer (Arc bump) via new_with_literal_buffer.
        let n = 1usize << 22; // 4M f32
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-3 - 2000.0).collect();
        let src = TensorValue::new_f32_values(
            Shape {
                dims: vec![n as u32],
            },
            data,
        )
        .unwrap();
        let new_shape = Shape {
            dims: vec![2048, 2048],
        };
        // OLD path: materialize to Vec<Literal> then re-analyze.
        let mut old = f64::MAX;
        for _ in 0..20 {
            let t = Instant::now();
            let out =
                TensorValue::new(DType::F32, new_shape.clone(), src.elements.to_vec()).unwrap();
            old = old.min(t.elapsed().as_secs_f64());
            std::hint::black_box(&out);
        }
        // NEW path: clone the backing buffer (Arc bump).
        let mut new = f64::MAX;
        for _ in 0..20 {
            let t = Instant::now();
            let out = TensorValue::new_with_literal_buffer(
                DType::F32,
                new_shape.clone(),
                src.elements.clone(),
            )
            .unwrap();
            new = new.min(t.elapsed().as_secs_f64());
            std::hint::black_box(&out);
        }
        println!(
            "BENCH reshape metadata n={n}: old(to_vec+new)={:.4}ms new(clone)={:.4}ms speedup={:.2}x",
            old * 1e3,
            new * 1e3,
            old / new
        );
    }

    #[test]
    fn dense_dynamic_update_slice_matches_literal_and_stays_dense() {
        // Dense DUS over f64/f32/bf16/f16/i64 must be BIT-FOR-BIT identical to the
        // boxed per-`Literal` path AND keep dense output, for both the contiguous
        // -trailing fast path and the general odometer path.
        let (rows, cols) = (6usize, 4usize);
        let lits = |v: &Value| v.as_tensor().unwrap().elements.to_vec();
        // (update dims, start indices) — contiguous-trailing then general.
        let cases: [(&[u32], &[i64]); 2] = [(&[2, 4], &[3, 0]), (&[2, 3], &[1, 1])];
        for (udims, starts) in cases {
            let usz = (udims[0] * udims[1]) as usize;
            let start_lits: Vec<Value> = starts
                .iter()
                .map(|&s| Value::Scalar(Literal::I64(s)))
                .collect();
            let mk_inputs = |op: Value, up: Value| -> Vec<Value> {
                let mut v = vec![op, up];
                v.extend(start_lits.iter().cloned());
                v
            };
            let odims = vec![rows as u32, cols as u32];
            let udimsv = udims.to_vec();

            // f64
            let opd: Vec<f64> = (0..rows * cols).map(|i| i as f64 * 0.5 - 3.0).collect();
            let upd: Vec<f64> = (0..usz).map(|i| 100.0 + i as f64).collect();
            let dense_op = Value::Tensor(
                TensorValue::new_f64_values(
                    Shape {
                        dims: odims.clone(),
                    },
                    opd.clone(),
                )
                .unwrap(),
            );
            let dense_up = Value::Tensor(
                TensorValue::new_f64_values(
                    Shape {
                        dims: udimsv.clone(),
                    },
                    upd.clone(),
                )
                .unwrap(),
            );
            let box_op = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: odims.clone(),
                    },
                    opd.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            );
            let box_up = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: udimsv.clone(),
                    },
                    upd.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            );
            let d =
                eval_dynamic_update_slice(&mk_inputs(dense_op, dense_up), &params(&[])).unwrap();
            let l = eval_dynamic_update_slice(&mk_inputs(box_op, box_up), &params(&[])).unwrap();
            assert_eq!(extract_shape(&d), extract_shape(&l), "f64 DUS shape");
            assert_eq!(lits(&d), lits(&l), "f64 DUS {starts:?}");
            assert!(
                d.as_tensor().unwrap().elements.as_f64_slice().is_some(),
                "f64 DUS dense"
            );

            // f32
            let opd: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.25 - 1.0).collect();
            let upd: Vec<f32> = (0..usz).map(|i| 50.0 + i as f32).collect();
            let dense_op = Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: odims.clone(),
                    },
                    opd.clone(),
                )
                .unwrap(),
            );
            let dense_up = Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: udimsv.clone(),
                    },
                    upd.clone(),
                )
                .unwrap(),
            );
            let box_op = Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape {
                        dims: odims.clone(),
                    },
                    opd.iter().copied().map(Literal::from_f32).collect(),
                )
                .unwrap(),
            );
            let box_up = Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape {
                        dims: udimsv.clone(),
                    },
                    upd.iter().copied().map(Literal::from_f32).collect(),
                )
                .unwrap(),
            );
            let d =
                eval_dynamic_update_slice(&mk_inputs(dense_op, dense_up), &params(&[])).unwrap();
            let l = eval_dynamic_update_slice(&mk_inputs(box_op, box_up), &params(&[])).unwrap();
            assert_eq!(lits(&d), lits(&l), "f32 DUS {starts:?}");
            assert!(
                d.as_tensor().unwrap().elements.as_f32_slice().is_some(),
                "f32 DUS dense"
            );

            for dtype in [DType::BF16, DType::F16] {
                let opr: Vec<u16> = (0..rows * cols)
                    .map(|i| (i as u16).wrapping_mul(67).wrapping_add(5))
                    .collect();
                let upr: Vec<u16> = (0..usz)
                    .map(|i| (i as u16).wrapping_mul(131).wrapping_add(9))
                    .collect();
                let mk = move |b: u16| {
                    if dtype == DType::BF16 {
                        Literal::BF16Bits(b)
                    } else {
                        Literal::F16Bits(b)
                    }
                };
                let dense_op = Value::Tensor(
                    TensorValue::new_half_float_values(
                        dtype,
                        Shape {
                            dims: odims.clone(),
                        },
                        opr.clone(),
                    )
                    .unwrap(),
                );
                let dense_up = Value::Tensor(
                    TensorValue::new_half_float_values(
                        dtype,
                        Shape {
                            dims: udimsv.clone(),
                        },
                        upr.clone(),
                    )
                    .unwrap(),
                );
                let box_op = Value::Tensor(
                    TensorValue::new(
                        dtype,
                        Shape {
                            dims: odims.clone(),
                        },
                        opr.iter().copied().map(mk).collect(),
                    )
                    .unwrap(),
                );
                let box_up = Value::Tensor(
                    TensorValue::new(
                        dtype,
                        Shape {
                            dims: udimsv.clone(),
                        },
                        upr.iter().copied().map(mk).collect(),
                    )
                    .unwrap(),
                );
                let d = eval_dynamic_update_slice(&mk_inputs(dense_op, dense_up), &params(&[]))
                    .unwrap();
                let l =
                    eval_dynamic_update_slice(&mk_inputs(box_op, box_up), &params(&[])).unwrap();
                assert_eq!(lits(&d), lits(&l), "{dtype:?} DUS {starts:?}");
                assert!(
                    d.as_tensor()
                        .unwrap()
                        .elements
                        .as_half_float_slice()
                        .is_some(),
                    "{dtype:?} DUS dense"
                );
            }

            // i64
            let opi: Vec<i64> = (0..(rows * cols) as i64).map(|i| i - 6).collect();
            let upi: Vec<i64> = (0..usz as i64).map(|i| 1000 + i).collect();
            let dense_op = Value::Tensor(
                TensorValue::new_i64_values(
                    Shape {
                        dims: odims.clone(),
                    },
                    opi.clone(),
                )
                .unwrap(),
            );
            let dense_up = Value::Tensor(
                TensorValue::new_i64_values(
                    Shape {
                        dims: udimsv.clone(),
                    },
                    upi.clone(),
                )
                .unwrap(),
            );
            let box_op = Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape {
                        dims: odims.clone(),
                    },
                    opi.iter().copied().map(Literal::I64).collect(),
                )
                .unwrap(),
            );
            let box_up = Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape {
                        dims: udimsv.clone(),
                    },
                    upi.iter().copied().map(Literal::I64).collect(),
                )
                .unwrap(),
            );
            let d =
                eval_dynamic_update_slice(&mk_inputs(dense_op, dense_up), &params(&[])).unwrap();
            let l = eval_dynamic_update_slice(&mk_inputs(box_op, box_up), &params(&[])).unwrap();
            assert_eq!(lits(&d), lits(&l), "i64 DUS {starts:?}");
            assert!(
                d.as_tensor().unwrap().elements.as_i64_slice().is_some(),
                "i64 DUS dense"
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_dynamic_update_slice_f32_dense_vs_boxed() {
        use std::time::Instant;
        // KV-cache style: write one row into a large [seq, dim] cache.
        let (seq, dim) = (4096usize, 1024usize);
        let opd: Vec<f32> = (0..seq * dim)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let upd: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let odims = vec![seq as u32, dim as u32];
        let udims = vec![1u32, dim as u32];
        let dense_op = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: odims.clone(),
                },
                opd.clone(),
            )
            .unwrap(),
        );
        let dense_up = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: udims.clone(),
                },
                upd.clone(),
            )
            .unwrap(),
        );
        let box_op = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: odims.clone(),
                },
                opd.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let box_up = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: udims.clone(),
                },
                upd.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let start = [
            Value::Scalar(Literal::I64(2000)),
            Value::Scalar(Literal::I64(0)),
        ];
        let mk = |op: &Value, up: &Value| {
            let mut v = vec![op.clone(), up.clone()];
            v.extend(start.iter().cloned());
            v
        };
        let time = |op: &Value, up: &Value| {
            let _ = eval_dynamic_update_slice(&mk(op, up), &params(&[])).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = eval_dynamic_update_slice(&mk(op, up), &params(&[])).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&box_op, &box_up);
        let dense_t = time(&dense_op, &dense_up);
        println!(
            "BENCH dynamic_update_slice f32 [{seq},{dim}] write 1 row: boxed(materialize+box)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn dense_dynamic_slice_matches_literal_and_stays_dense() {
        // Dense dynamic_slice over f64/f32/bf16/f16/i64 must be BIT-FOR-BIT
        // identical to the boxed per-`Literal` path AND keep dense output, for both
        // the contiguous-trailing fast path and the general odometer path.
        let (rows, cols) = (6usize, 4usize);
        let lits = |v: &Value| v.as_tensor().unwrap().elements.to_vec();
        // (slice_sizes, start indices) — contiguous-trailing then general.
        let cases: [(&[&str], &[i64]); 2] = [(&["2", "4"], &[3, 0]), (&["2", "3"], &[1, 1])];
        for (ssizes, starts) in cases {
            let start_lits: Vec<Value> = starts
                .iter()
                .map(|&s| Value::Scalar(Literal::I64(s)))
                .collect();
            let p = params(&[("slice_sizes", &ssizes.join(","))]);
            let mk_inputs = |op: Value| -> Vec<Value> {
                let mut v = vec![op];
                v.extend(start_lits.iter().cloned());
                v
            };
            let odims = vec![rows as u32, cols as u32];

            let opd: Vec<f64> = (0..rows * cols).map(|i| i as f64 * 0.5 - 3.0).collect();
            let dense = Value::Tensor(
                TensorValue::new_f64_values(
                    Shape {
                        dims: odims.clone(),
                    },
                    opd.clone(),
                )
                .unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: odims.clone(),
                    },
                    opd.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            );
            let d = eval_dynamic_slice(&mk_inputs(dense), &p).unwrap();
            let l = eval_dynamic_slice(&mk_inputs(boxed), &p).unwrap();
            assert_eq!(extract_shape(&d), extract_shape(&l), "f64 DS shape");
            assert_eq!(lits(&d), lits(&l), "f64 DS {starts:?}");
            assert!(
                d.as_tensor().unwrap().elements.as_f64_slice().is_some(),
                "f64 DS dense"
            );

            let opf: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.25 - 1.0).collect();
            let dense = Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: odims.clone(),
                    },
                    opf.clone(),
                )
                .unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape {
                        dims: odims.clone(),
                    },
                    opf.iter().copied().map(Literal::from_f32).collect(),
                )
                .unwrap(),
            );
            let d = eval_dynamic_slice(&mk_inputs(dense), &p).unwrap();
            let l = eval_dynamic_slice(&mk_inputs(boxed), &p).unwrap();
            assert_eq!(lits(&d), lits(&l), "f32 DS {starts:?}");
            assert!(
                d.as_tensor().unwrap().elements.as_f32_slice().is_some(),
                "f32 DS dense"
            );

            for dtype in [DType::BF16, DType::F16] {
                let opr: Vec<u16> = (0..rows * cols)
                    .map(|i| (i as u16).wrapping_mul(67).wrapping_add(5))
                    .collect();
                let mk = move |b: u16| {
                    if dtype == DType::BF16 {
                        Literal::BF16Bits(b)
                    } else {
                        Literal::F16Bits(b)
                    }
                };
                let dense = Value::Tensor(
                    TensorValue::new_half_float_values(
                        dtype,
                        Shape {
                            dims: odims.clone(),
                        },
                        opr.clone(),
                    )
                    .unwrap(),
                );
                let boxed = Value::Tensor(
                    TensorValue::new(
                        dtype,
                        Shape {
                            dims: odims.clone(),
                        },
                        opr.iter().copied().map(mk).collect(),
                    )
                    .unwrap(),
                );
                let d = eval_dynamic_slice(&mk_inputs(dense), &p).unwrap();
                let l = eval_dynamic_slice(&mk_inputs(boxed), &p).unwrap();
                assert_eq!(lits(&d), lits(&l), "{dtype:?} DS {starts:?}");
                assert!(
                    d.as_tensor()
                        .unwrap()
                        .elements
                        .as_half_float_slice()
                        .is_some(),
                    "{dtype:?} DS dense"
                );
            }

            let opi: Vec<i64> = (0..(rows * cols) as i64).map(|i| i - 6).collect();
            let dense = Value::Tensor(
                TensorValue::new_i64_values(
                    Shape {
                        dims: odims.clone(),
                    },
                    opi.clone(),
                )
                .unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape {
                        dims: odims.clone(),
                    },
                    opi.iter().copied().map(Literal::I64).collect(),
                )
                .unwrap(),
            );
            let d = eval_dynamic_slice(&mk_inputs(dense), &p).unwrap();
            let l = eval_dynamic_slice(&mk_inputs(boxed), &p).unwrap();
            assert_eq!(lits(&d), lits(&l), "i64 DS {starts:?}");
            assert!(
                d.as_tensor().unwrap().elements.as_i64_slice().is_some(),
                "i64 DS dense"
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_dynamic_slice_f32_dense_vs_boxed() {
        use std::time::Instant;
        // Read a large contiguous block of rows out of a [rows, cols] tensor — the
        // contiguous-trailing fast path (KV-cache / scan "read a window of rows").
        // Boxed `to_vec()` clones N 24-byte Literals; dense clones N 4-byte f32 and
        // keeps dense storage. Memory-bandwidth bound -> the per-Literal width wins.
        let (rows, cols) = (8192usize, 2048usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let odims = vec![rows as u32, cols as u32];
        let dense = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: odims.clone(),
                },
                data.clone(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: odims.clone(),
                },
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let take = rows / 2;
        let p = params(&[("slice_sizes", &format!("{take},{cols}"))]);
        let mk = |op: &Value| {
            vec![
                op.clone(),
                Value::Scalar(Literal::I64(1024)),
                Value::Scalar(Literal::I64(0)),
            ]
        };
        let time = |op: &Value| {
            let _ = eval_dynamic_slice(&mk(op), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..50 {
                let t = Instant::now();
                let _ = eval_dynamic_slice(&mk(op), &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH dynamic_slice f32 [{rows},{cols}] read {take} rows (contiguous): boxed(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn reshape_rejects_tensor_dim_above_u32() {
        let x =
            Value::Tensor(TensorValue::new(DType::I64, Shape { dims: vec![0] }, vec![]).unwrap());
        let p = params(&[("new_shape", "4294967296")]);
        let err = eval_reshape(&[x], &p).unwrap_err().to_string();

        assert!(
            err.contains("new_shape dim 4294967296 exceeds u32 range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reshape_rejects_scalar_dim_above_u32() {
        let p = params(&[("new_shape", "4294967296")]);
        let err = eval_reshape(&[Value::scalar_i64(1)], &p)
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("new_shape dim 4294967296 exceeds u32 range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reshape_rejects_shape_product_overflow() {
        let x =
            Value::Tensor(TensorValue::new(DType::I64, Shape { dims: vec![0] }, vec![]).unwrap());
        let p = params(&[("new_shape", "4294967295,4294967295,4294967295")]);
        let err = eval_reshape(&[x], &p).unwrap_err().to_string();

        assert!(
            err.contains("reshape known dimension product overflows usize"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn i32_structural_ops_preserve_dtype_not_widen_to_i64() {
        // i32 densifies into the i64 backing, so the dense structural paths gated on
        // `as_i64_slice()` (transpose / slice contiguous+strided / reverse / tile /
        // broadcast_in_dim / pad) silently emitted an I64 result — a widening parity bug
        // (these ops must preserve dtype). Each dense (densified i32) result must keep
        // I32 AND match the boxed generic path (forced via new_with_literal_buffer).
        let mk_d = |dims: &[u32], d: &[i64]| {
            Value::Tensor(
                TensorValue::new(
                    DType::I32,
                    Shape { dims: dims.to_vec() },
                    d.iter().map(|&v| Literal::I64(v)).collect(),
                )
                .unwrap(),
            )
        };
        let mk_b = |dims: &[u32], d: &[i64]| {
            Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    DType::I32,
                    Shape { dims: dims.to_vec() },
                    fj_core::LiteralBuffer::new(d.iter().map(|&v| Literal::I64(v)).collect()),
                )
                .unwrap(),
            )
        };
        let geti = |v: &Value| -> (DType, Vec<i64>) {
            let t = v.as_tensor().unwrap();
            (
                t.dtype,
                t.elements
                    .iter()
                    .map(|l| match l {
                        Literal::I64(x) => *x,
                        o => panic!("expected I64-backed i32, got {o:?}"),
                    })
                    .collect(),
            )
        };
        let data: Vec<i64> = (0..24).map(|i| i64::from((i as i32).wrapping_mul(7) - 11)).collect();

        // Single-operand structural ops over a [4,6] i32 tensor.
        let single: &[(fn(&[Value], &BTreeMap<String, String>) -> Result<Value, EvalError>, BTreeMap<String, String>, &str)] = &[
            (eval_transpose, params(&[("permutation", "1,0")]), "transpose"),
            (eval_slice, params(&[("start_indices", "0,0"), ("limit_indices", "4,6")]), "slice-contig"),
            (eval_slice, params(&[("start_indices", "0,0"), ("limit_indices", "4,6"), ("strides", "2,2")]), "slice-strided"),
            (eval_rev, params(&[("axes", "0,1")]), "reverse"),
            (eval_tile, params(&[("reps", "2,1")]), "tile"),
        ];
        for (f, p, label) in single {
            let d = f(std::slice::from_ref(&mk_d(&[4, 6], &data)), p).unwrap();
            let b = f(std::slice::from_ref(&mk_b(&[4, 6], &data)), p).unwrap();
            let (dt_d, v_d) = geti(&d);
            let (dt_b, v_b) = geti(&b);
            assert_eq!(dt_d, DType::I32, "{label}: dense must stay I32 (was widening to I64)");
            assert_eq!(dt_b, DType::I32, "{label}: generic dtype");
            assert_eq!(v_d, v_b, "{label}: dense values != generic");
        }

        // broadcast_in_dim: [3] -> [4,3] (broadcast_dimensions=1).
        let bd = data[..3].to_vec();
        let pbc = params(&[("shape", "4,3"), ("broadcast_dimensions", "1")]);
        let d = eval_broadcast_in_dim(std::slice::from_ref(&mk_d(&[3], &bd)), &pbc).unwrap();
        let b = eval_broadcast_in_dim(std::slice::from_ref(&mk_b(&[3], &bd)), &pbc).unwrap();
        assert_eq!(geti(&d).0, DType::I32, "broadcast: dense must stay I32");
        assert_eq!(geti(&d), geti(&b), "broadcast: dense != generic");

        // pad: operand [4,6] + 0-d I32 pad value (must match operand dtype), interior 1.
        let padv = Value::Tensor(
            TensorValue::new(DType::I32, Shape { dims: vec![] }, vec![Literal::I64(-99)]).unwrap(),
        );
        let pp = params(&[
            ("padding_low", "1,1"),
            ("padding_high", "1,1"),
            ("padding_interior", "1,0"),
        ]);
        let d = eval_pad(&[mk_d(&[4, 6], &data), padv.clone()], &pp).unwrap();
        let b = eval_pad(&[mk_b(&[4, 6], &data), padv.clone()], &pp).unwrap();
        assert_eq!(geti(&d).0, DType::I32, "pad: dense must stay I32");
        assert_eq!(geti(&d), geti(&b), "pad: dense != generic");
    }

    // ── Transpose ──

    #[test]
    fn transpose_2x3() {
        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        let x = mat_f64(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = params(&[("permutation", "1,0")]);
        let result = eval_transpose(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![3, 2]);
        assert_eq!(extract_f64_vec(&result), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_identity() {
        let x = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let p = params(&[("permutation", "0,1")]);
        let result = eval_transpose(std::slice::from_ref(&x), &p).unwrap();
        assert_eq!(extract_f64_vec(&result), extract_f64_vec(&x));
    }

    #[test]
    fn transpose_rank3_matches_naive_reference() {
        // Rank-3 tensor 2x3x4 with a non-trivial permutation [2,0,1]; the
        // incremental ("odometer") walk must reproduce the textbook
        // multi-index mapping exactly (transpose is pure data movement).
        let (d0, d1, d2) = (2usize, 3usize, 4usize);
        let total = d0 * d1 * d2;
        let data: Vec<f64> = (0..total).map(|i| i as f64 * 0.5 - 3.0).collect();
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![d0 as u32, d1 as u32, d2 as u32],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        );
        let perm = [2usize, 0, 1];
        let p = params(&[("permutation", "2,0,1")]);
        let got = extract_f64_vec(&eval_transpose(std::slice::from_ref(&x), &p).unwrap());

        // Naive reference: new layout row-major over permuted dims.
        let old_dims = [d0, d1, d2];
        let new_dims: Vec<usize> = perm.iter().map(|&a| old_dims[a]).collect();
        let old_strides = [d1 * d2, d2, 1usize];
        let mut new_strides = [0usize; 3];
        new_strides[2] = 1;
        new_strides[1] = new_dims[2];
        new_strides[0] = new_dims[1] * new_dims[2];
        let mut want = Vec::with_capacity(total);
        for flat in 0..total {
            let mut rem = flat;
            let mut old_flat = 0usize;
            for new_axis in 0..3 {
                let coord = rem / new_strides[new_axis];
                rem %= new_strides[new_axis];
                old_flat += coord * old_strides[perm[new_axis]];
            }
            want.push(data[old_flat]);
        }
        assert_eq!(got, want);
        assert_eq!(
            extract_shape(&eval_transpose(&[x], &p).unwrap()),
            vec![4, 2, 3]
        );
    }

    #[test]
    fn transpose_complex_dense_matches_naive_reference() {
        // Complex transpose now uses the dense (f64,f64) path. It must equal the
        // textbook permuted mapping bit-for-bit for both Complex128 and Complex64,
        // across rank-2 ([1,0]) and rank-3 ([2,0,1]) permutations.
        for &dtype in &[DType::Complex128, DType::Complex64] {
            // rank-2 [3,5] -> [5,3]
            let (m, n) = (3usize, 5usize);
            let total = m * n;
            let parts: Vec<(f64, f64)> = (0..total)
                .map(|i| (i as f64 * 0.5 - 2.0, i as f64 * -0.25 + 1.0))
                .collect();
            let lits: Vec<Literal> = parts
                .iter()
                .map(|&(re, im)| match dtype {
                    DType::Complex64 => Literal::from_complex64(re as f32, im as f32),
                    _ => Literal::from_complex128(re, im),
                })
                .collect();
            let x = Value::Tensor(
                TensorValue::new(
                    dtype,
                    Shape {
                        dims: vec![m as u32, n as u32],
                    },
                    lits.clone(),
                )
                .unwrap(),
            );
            let p = params(&[("permutation", "1,0")]);
            let Value::Tensor(out) = eval_transpose(std::slice::from_ref(&x), &p).unwrap() else {
                panic!("expected tensor");
            };
            assert_eq!(out.dtype, dtype);
            assert_eq!(out.shape.dims, vec![n as u32, m as u32]);
            // out[j,i] == in[i,j]
            for j in 0..n {
                for i in 0..m {
                    assert_eq!(
                        out.elements[j * m + i],
                        lits[i * n + j],
                        "complex {dtype:?} transpose mismatch at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn transpose_rank2_tiled_non_block_aligned() {
        // Non-square dims that cross the 64-element tile boundary, exercising
        // the cache-blocked rank-2 path's partial tiles.
        let (rows, cols) = (70usize, 130usize);
        let data: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        );
        let p = params(&[("permutation", "1,0")]);
        let result = eval_transpose(std::slice::from_ref(&x), &p).unwrap();
        assert_eq!(extract_shape(&result), vec![cols as u32, rows as u32]);
        let got = extract_f64_vec(&result);
        // out[j, i] == in[i, j]
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(got[j * rows + i], data[i * cols + j], "({i},{j})");
            }
        }
    }

    /// Dense transpose (f64/f32/bf16/f16/i64) must be BIT-FOR-BIT identical to the
    /// boxed per-`Literal` path AND keep dense output, for both the rank-2 tiled
    /// path (perm [1,0], non-block-aligned) and the general rank-3 odometer path.
    #[test]
    fn dense_transpose_matches_literal_path_and_stays_dense() {
        for (dims, perm) in [
            (vec![70u32, 130u32], "1,0"),      // rank-2 tiled, partial tiles
            (vec![5u32, 7u32, 4u32], "2,0,1"), // general rank-3
        ] {
            let n: usize = dims.iter().map(|&d| d as usize).product();
            let lits = |v: &Value| v.as_tensor().unwrap().elements.to_vec();
            let p = params(&[("permutation", perm)]);

            macro_rules! check {
                ($mk_dense:expr, $mk_boxed:expr, $is_dense:expr) => {{
                    let dense = $mk_dense;
                    let boxed = $mk_boxed;
                    let d = eval_transpose(std::slice::from_ref(&dense), &p).unwrap();
                    let l = eval_transpose(std::slice::from_ref(&boxed), &p).unwrap();
                    assert_eq!(
                        extract_shape(&d),
                        extract_shape(&l),
                        "{dims:?} {perm} shape"
                    );
                    assert_eq!(lits(&d), lits(&l), "{dims:?} {perm} values");
                    assert!(
                        $is_dense(d.as_tensor().unwrap()),
                        "{dims:?} {perm} output dense"
                    );
                }};
            }
            let sh = || Shape { dims: dims.clone() };

            let f64d: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 3.0).collect();
            check!(
                Value::Tensor(TensorValue::new_f64_values(sh(), f64d.clone()).unwrap()),
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        sh(),
                        f64d.iter().copied().map(Literal::from_f64).collect()
                    )
                    .unwrap()
                ),
                |t: &TensorValue| t.elements.as_f64_slice().is_some()
            );
            let f32d: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 1.0).collect();
            check!(
                Value::Tensor(TensorValue::new_f32_values(sh(), f32d.clone()).unwrap()),
                Value::Tensor(
                    TensorValue::new(
                        DType::F32,
                        sh(),
                        f32d.iter().copied().map(Literal::from_f32).collect()
                    )
                    .unwrap()
                ),
                |t: &TensorValue| t.elements.as_f32_slice().is_some()
            );
            for dtype in [DType::BF16, DType::F16] {
                let raw: Vec<u16> = (0..n)
                    .map(|i| (i as u16).wrapping_mul(83).wrapping_add(9))
                    .collect();
                let mk_lit = move |b: u16| {
                    if dtype == DType::BF16 {
                        Literal::BF16Bits(b)
                    } else {
                        Literal::F16Bits(b)
                    }
                };
                check!(
                    Value::Tensor(
                        TensorValue::new_half_float_values(dtype, sh(), raw.clone()).unwrap()
                    ),
                    Value::Tensor(
                        TensorValue::new(dtype, sh(), raw.iter().copied().map(mk_lit).collect())
                            .unwrap()
                    ),
                    |t: &TensorValue| t.elements.as_half_float_slice().is_some()
                );
            }
            let i64d: Vec<i64> = (0..n as i64).map(|i| i - 7).collect();
            check!(
                Value::Tensor(TensorValue::new_i64_values(sh(), i64d.clone()).unwrap()),
                Value::Tensor(
                    TensorValue::new(
                        DType::I64,
                        sh(),
                        i64d.iter().copied().map(Literal::I64).collect()
                    )
                    .unwrap()
                ),
                |t: &TensorValue| t.elements.as_i64_slice().is_some()
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_transpose_dense_vs_boxed() {
        use std::time::Instant;
        let (rows, cols) = (3000usize, 3000usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let sh = Shape {
            dims: vec![rows as u32, cols as u32],
        };
        let dense = Value::Tensor(TensorValue::new_f32_values(sh.clone(), data.clone()).unwrap());
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                sh.clone(),
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let p = params(&[("permutation", "1,0")]);
        let time = |x: &Value| {
            let _ = eval_transpose(std::slice::from_ref(x), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..10 {
                let t = Instant::now();
                let _ = eval_transpose(std::slice::from_ref(x), &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 transpose [{rows},{cols}]: boxed(materialize+box)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    // ── Slice ──

    #[test]
    fn slice_1d() {
        let x = v_f64(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let p = params(&[("start_indices", "1"), ("limit_indices", "4")]);
        let result = eval_slice(&[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn slice_1d_from_start() {
        let x = v_f64(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let p = params(&[("start_indices", "0"), ("limit_indices", "2")]);
        let result = eval_slice(&[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![10.0, 20.0]);
    }

    /// Dense broadcast_in_dim (f32/bf16/f16) — both the tensor-replication path and
    /// the scalar fill — must be BIT-FOR-BIT identical to the boxed per-`Literal`
    /// path AND keep dense output.
    #[test]
    fn dense_broadcast_in_dim_matches_literal_path_and_stays_dense() {
        let lits = |v: &Value| v.as_tensor().unwrap().elements.to_vec();
        // Tensor broadcast: [3] -> [4,3] (broadcast_dimensions=[1]).
        let p_tensor = params(&[("shape", "4,3"), ("broadcast_dimensions", "1")]);
        // Scalar broadcast: scalar -> [5,6].
        let p_scalar = params(&[("shape", "5,6")]);

        // f32
        let f32d: Vec<f32> = vec![1.5, -0.0, 3.25];
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: vec![3] }, f32d.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: vec![3] },
                f32d.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let d = eval_broadcast_in_dim(std::slice::from_ref(&dense), &p_tensor).unwrap();
        let l = eval_broadcast_in_dim(std::slice::from_ref(&boxed), &p_tensor).unwrap();
        assert_eq!(lits(&d), lits(&l), "f32 tensor broadcast");
        assert!(
            d.as_tensor().unwrap().elements.as_f32_slice().is_some(),
            "f32 broadcast output dense"
        );
        // f32 scalar fill
        let ds =
            eval_broadcast_in_dim(&[Value::Scalar(Literal::from_f32(-2.5))], &p_scalar).unwrap();
        assert_eq!(ds.as_tensor().unwrap().dtype, DType::F32);
        assert!(
            ds.as_tensor().unwrap().elements.as_f32_slice().is_some(),
            "f32 scalar broadcast dense"
        );
        assert_eq!(lits(&ds), vec![Literal::from_f32(-2.5); 30]);

        // bf16 + f16
        for dtype in [DType::BF16, DType::F16] {
            let raw: Vec<u16> = vec![0x3f80, 0x8000, 0x7fc1];
            let mk_lit = move |b: u16| {
                if dtype == DType::BF16 {
                    Literal::BF16Bits(b)
                } else {
                    Literal::F16Bits(b)
                }
            };
            let dense = Value::Tensor(
                TensorValue::new_half_float_values(dtype, Shape { dims: vec![3] }, raw.clone())
                    .unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    dtype,
                    Shape { dims: vec![3] },
                    raw.iter().copied().map(mk_lit).collect(),
                )
                .unwrap(),
            );
            let d = eval_broadcast_in_dim(std::slice::from_ref(&dense), &p_tensor).unwrap();
            let l = eval_broadcast_in_dim(std::slice::from_ref(&boxed), &p_tensor).unwrap();
            assert_eq!(lits(&d), lits(&l), "{dtype:?} tensor broadcast");
            assert!(
                d.as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_some(),
                "{dtype:?} broadcast output dense"
            );
            let ds = eval_broadcast_in_dim(&[Value::Scalar(mk_lit(0x4000))], &p_scalar).unwrap();
            assert_eq!(ds.as_tensor().unwrap().dtype, dtype);
            assert!(
                ds.as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_some(),
                "{dtype:?} scalar broadcast dense"
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_broadcast_dense_vs_boxed() {
        use std::time::Instant;
        let (rows, cols) = (8192usize, 2048usize); // [cols] -> [rows, cols]
        let data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let dense = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: vec![cols as u32],
                },
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let p = params(&[
            ("shape", &format!("{rows},{cols}")),
            ("broadcast_dimensions", "1"),
        ]);
        let time = |x: &Value| {
            let _ = eval_broadcast_in_dim(std::slice::from_ref(x), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..15 {
                let t = Instant::now();
                let _ = eval_broadcast_in_dim(std::slice::from_ref(x), &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 broadcast [{cols}]->[{rows},{cols}]: boxed(materialize+box)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    /// Dense pad (f32/bf16/f16) must be BIT-FOR-BIT identical to the boxed
    /// per-`Literal` path AND keep dense output, for both the row-copyable case
    /// (low padding only) and the per-element placement case (interior dilation).
    #[test]
    fn dense_pad_matches_literal_path_and_stays_dense() {
        let (rows, cols) = (6usize, 4usize);
        let dims = vec![rows as u32, cols as u32];
        let lits = |v: &Value| v.as_tensor().unwrap().elements.to_vec();
        for p in [
            params(&[
                ("padding_low", "1,2"),
                ("padding_high", "2,1"),
                ("padding_interior", "0,0"),
            ]), // row-copyable
            params(&[
                ("padding_low", "1,0"),
                ("padding_high", "0,1"),
                ("padding_interior", "1,2"),
            ]), // interior dilation
            params(&[
                ("padding_low", "-1,0"),
                ("padding_high", "0,-1"),
                ("padding_interior", "0,0"),
            ]), // crop
        ] {
            // f32
            let f32d: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.5 - 2.0).collect();
            let dense = Value::Tensor(
                TensorValue::new_f32_values(Shape { dims: dims.clone() }, f32d.clone()).unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape { dims: dims.clone() },
                    f32d.iter().copied().map(Literal::from_f32).collect(),
                )
                .unwrap(),
            );
            let pv = Value::Scalar(Literal::from_f32(-9.5));
            let d = eval_pad(&[dense, pv.clone()], &p).unwrap();
            let l = eval_pad(&[boxed, pv], &p).unwrap();
            assert_eq!(extract_shape(&d), extract_shape(&l), "f32 pad shape");
            assert_eq!(lits(&d), lits(&l), "f32 pad values");
            assert!(
                d.as_tensor().unwrap().elements.as_f32_slice().is_some(),
                "f32 pad output dense"
            );

            // bf16 + f16
            for dtype in [DType::BF16, DType::F16] {
                let raw: Vec<u16> = (0..rows * cols)
                    .map(|i| (i as u16).wrapping_mul(91).wrapping_add(3))
                    .collect();
                let mk_lit = move |b: u16| {
                    if dtype == DType::BF16 {
                        Literal::BF16Bits(b)
                    } else {
                        Literal::F16Bits(b)
                    }
                };
                let dense = Value::Tensor(
                    TensorValue::new_half_float_values(
                        dtype,
                        Shape { dims: dims.clone() },
                        raw.clone(),
                    )
                    .unwrap(),
                );
                let boxed = Value::Tensor(
                    TensorValue::new(
                        dtype,
                        Shape { dims: dims.clone() },
                        raw.iter().copied().map(mk_lit).collect(),
                    )
                    .unwrap(),
                );
                let pv = Value::Scalar(mk_lit(0x7fc0));
                let d = eval_pad(&[dense, pv.clone()], &p).unwrap();
                let l = eval_pad(&[boxed, pv], &p).unwrap();
                assert_eq!(lits(&d), lits(&l), "{dtype:?} pad values");
                assert!(
                    d.as_tensor()
                        .unwrap()
                        .elements
                        .as_half_float_slice()
                        .is_some(),
                    "{dtype:?} pad output dense"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_pad_dense_vs_boxed() {
        use std::time::Instant;
        let (rows, cols) = (4000usize, 1000usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: dims.clone() },
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let p = params(&[
            ("padding_low", "1,1"),
            ("padding_high", "1,1"),
            ("padding_interior", "0,0"),
        ]);
        let pv = Value::Scalar(Literal::from_f32(0.0));
        let time = |x: &Value| {
            let _ = eval_pad(&[x.clone(), pv.clone()], &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = eval_pad(&[x.clone(), pv.clone()], &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 pad [{rows},{cols}]+1 border: boxed(materialize+box)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    /// Dense STRIDED slice (non-leading-axis slice + stride>1) must be BIT-FOR-BIT
    /// identical to the boxed per-`Literal` path AND keep dense output, for
    /// f64/f32/bf16/f16/i64.
    #[test]
    fn dense_strided_slice_matches_literal_path_and_stays_dense() {
        let (rows, cols) = (10usize, 12usize);
        let dims = vec![rows as u32, cols as u32];
        let lits = |v: &Value| v.as_tensor().unwrap().elements.to_vec();
        for p in [
            params(&[("start_indices", "0,2"), ("limit_indices", "10,8")]), // non-leading axis crop
            params(&[
                ("start_indices", "1,1"),
                ("limit_indices", "9,11"),
                ("strides", "2,3"),
            ]), // stride>1
        ] {
            let f64d: Vec<f64> = (0..rows * cols).map(|i| i as f64 * 0.5 - 5.0).collect();
            let dense = Value::Tensor(
                TensorValue::new_f64_values(Shape { dims: dims.clone() }, f64d.clone()).unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: dims.clone() },
                    f64d.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            );
            let d = eval_slice(std::slice::from_ref(&dense), &p).unwrap();
            let l = eval_slice(std::slice::from_ref(&boxed), &p).unwrap();
            assert_eq!(
                extract_shape(&d),
                extract_shape(&l),
                "f64 strided slice shape"
            );
            assert_eq!(lits(&d), lits(&l), "f64 strided slice");
            assert!(
                d.as_tensor().unwrap().elements.as_f64_slice().is_some(),
                "f64 strided slice dense"
            );

            let f32d: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.25 - 1.0).collect();
            let dense = Value::Tensor(
                TensorValue::new_f32_values(Shape { dims: dims.clone() }, f32d.clone()).unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape { dims: dims.clone() },
                    f32d.iter().copied().map(Literal::from_f32).collect(),
                )
                .unwrap(),
            );
            let d = eval_slice(std::slice::from_ref(&dense), &p).unwrap();
            let l = eval_slice(std::slice::from_ref(&boxed), &p).unwrap();
            assert_eq!(lits(&d), lits(&l), "f32 strided slice");
            assert!(
                d.as_tensor().unwrap().elements.as_f32_slice().is_some(),
                "f32 strided slice dense"
            );

            for dtype in [DType::BF16, DType::F16] {
                let raw: Vec<u16> = (0..rows * cols)
                    .map(|i| (i as u16).wrapping_mul(67).wrapping_add(5))
                    .collect();
                let mk_lit = move |b: u16| {
                    if dtype == DType::BF16 {
                        Literal::BF16Bits(b)
                    } else {
                        Literal::F16Bits(b)
                    }
                };
                let dense = Value::Tensor(
                    TensorValue::new_half_float_values(
                        dtype,
                        Shape { dims: dims.clone() },
                        raw.clone(),
                    )
                    .unwrap(),
                );
                let boxed = Value::Tensor(
                    TensorValue::new(
                        dtype,
                        Shape { dims: dims.clone() },
                        raw.iter().copied().map(mk_lit).collect(),
                    )
                    .unwrap(),
                );
                let d = eval_slice(std::slice::from_ref(&dense), &p).unwrap();
                let l = eval_slice(std::slice::from_ref(&boxed), &p).unwrap();
                assert_eq!(lits(&d), lits(&l), "{dtype:?} strided slice");
                assert!(
                    d.as_tensor()
                        .unwrap()
                        .elements
                        .as_half_float_slice()
                        .is_some(),
                    "{dtype:?} strided slice dense"
                );
            }

            let i64d: Vec<i64> = (0..(rows * cols) as i64).map(|i| i - 30).collect();
            let dense = Value::Tensor(
                TensorValue::new_i64_values(Shape { dims: dims.clone() }, i64d.clone()).unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape { dims: dims.clone() },
                    i64d.iter().copied().map(Literal::I64).collect(),
                )
                .unwrap(),
            );
            let d = eval_slice(std::slice::from_ref(&dense), &p).unwrap();
            let l = eval_slice(std::slice::from_ref(&boxed), &p).unwrap();
            assert_eq!(lits(&d), lits(&l), "i64 strided slice");
            assert!(
                d.as_tensor().unwrap().elements.as_i64_slice().is_some(),
                "i64 strided slice dense"
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_strided_slice_dense_vs_boxed() {
        use std::time::Instant;
        let (rows, cols) = (4096usize, 2048usize); // crop feature axis: [:, 256:1792]
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: dims.clone() },
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let p = params(&[("start_indices", "0,256"), ("limit_indices", "4096,1792")]);
        let time = |x: &Value| {
            let _ = eval_slice(std::slice::from_ref(x), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = eval_slice(std::slice::from_ref(x), &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 strided slice [{rows},{cols}]->[:,256:1792]: boxed(materialize+box)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    /// Dense contiguous slice (f64/f32/bf16/f16/i64) must be BIT-FOR-BIT identical
    /// to the boxed per-`Literal` path AND keep dense output storage (without
    /// materializing the whole input buffer).
    #[test]
    fn dense_contiguous_slice_matches_literal_path_and_stays_dense() {
        let (rows, cols) = (8usize, 5usize);
        let dims = vec![rows as u32, cols as u32];
        let p = params(&[("start_indices", "2,0"), ("limit_indices", "6,5")]); // rows [2:6]
        let lits = |v: &Value| v.as_tensor().unwrap().elements.to_vec();

        // f64
        let f64d: Vec<f64> = (0..rows * cols).map(|i| i as f64 * 1.5 - 3.0).collect();
        let dense = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, f64d.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: dims.clone() },
                f64d.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let d = eval_slice(&[dense], &p).unwrap();
        let l = eval_slice(&[boxed], &p).unwrap();
        assert_eq!(lits(&d), lits(&l), "f64 slice");
        assert!(
            d.as_tensor().unwrap().elements.as_f64_slice().is_some(),
            "f64 slice output dense"
        );

        // f32
        let f32d: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.25 - 1.0).collect();
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, f32d.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: dims.clone() },
                f32d.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let d = eval_slice(&[dense], &p).unwrap();
        let l = eval_slice(&[boxed], &p).unwrap();
        assert_eq!(lits(&d), lits(&l), "f32 slice");
        assert!(
            d.as_tensor().unwrap().elements.as_f32_slice().is_some(),
            "f32 slice output dense"
        );

        // bf16 + f16
        for dtype in [DType::BF16, DType::F16] {
            let raw: Vec<u16> = (0..rows * cols)
                .map(|i| (i as u16).wrapping_mul(61).wrapping_add(7))
                .collect();
            let mk_lit = |b: u16| {
                if dtype == DType::BF16 {
                    Literal::BF16Bits(b)
                } else {
                    Literal::F16Bits(b)
                }
            };
            let dense = Value::Tensor(
                TensorValue::new_half_float_values(
                    dtype,
                    Shape { dims: dims.clone() },
                    raw.clone(),
                )
                .unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    dtype,
                    Shape { dims: dims.clone() },
                    raw.iter().copied().map(mk_lit).collect(),
                )
                .unwrap(),
            );
            let d = eval_slice(&[dense], &p).unwrap();
            let l = eval_slice(&[boxed], &p).unwrap();
            assert_eq!(lits(&d), lits(&l), "{dtype:?} slice");
            assert!(
                d.as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_some(),
                "{dtype:?} slice output dense"
            );
        }

        // i64
        let i64d: Vec<i64> = (0..(rows * cols) as i64).map(|i| i - 12).collect();
        let dense = Value::Tensor(
            TensorValue::new_i64_values(Shape { dims: dims.clone() }, i64d.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: dims.clone() },
                i64d.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let d = eval_slice(&[dense], &p).unwrap();
        let l = eval_slice(&[boxed], &p).unwrap();
        assert_eq!(lits(&d), lits(&l), "i64 slice");
        assert!(
            d.as_tensor().unwrap().elements.as_i64_slice().is_some(),
            "i64 slice output dense"
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_slice_dense_vs_boxed() {
        use std::time::Instant;
        let (rows, cols) = (4096usize, 1024usize); // slice [0:2048] -> half
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: dims.clone() },
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let p = params(&[("start_indices", "0,0"), ("limit_indices", "2048,1024")]);
        let time = |x: &Value| {
            let _ = eval_slice(std::slice::from_ref(x), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = eval_slice(std::slice::from_ref(x), &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 slice [{rows},{cols}]->[2048,{cols}]: boxed(materialize+box)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn slice_empty_huge_trailing_shape_returns_empty() {
        let huge = u32::MAX;
        let x = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![0, huge, huge, huge, huge],
                },
                Vec::new(),
            )
            .unwrap(),
        );
        let p = params(&[
            ("start_indices", "0,0,0,0,0"),
            (
                "limit_indices",
                "0,4294967295,4294967295,4294967295,4294967295",
            ),
        ]);

        let result = eval_slice(&[x], &p).unwrap();

        assert_eq!(extract_shape(&result), vec![0, huge, huge, huge, huge]);
        assert!(result.as_tensor().unwrap().elements.is_empty());
    }

    // ── Concatenate ──

    #[test]
    fn concatenate_1d() {
        let a = v_f64(&[1.0, 2.0]);
        let b = v_f64(&[3.0, 4.0, 5.0]);
        let p = params(&[("dimension", "0")]);
        let result = eval_concatenate(&[a, b], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn concatenate_rank3_middle_axis_matches_naive() {
        // Three operands concatenated on the middle axis of a rank-3 tensor;
        // the block-copy result must equal the per-coordinate reference.
        let outer = 2usize;
        let inner = 4usize;
        let dims_axis = [3usize, 1usize, 2usize]; // concat-axis size per operand
        let axis = 1usize;
        let mk = |d1: usize, base: f64| -> (Value, Vec<f64>) {
            let data: Vec<f64> = (0..outer * d1 * inner).map(|i| base + i as f64).collect();
            (
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        Shape {
                            dims: vec![outer as u32, d1 as u32, inner as u32],
                        },
                        data.iter().map(|&v| Literal::from_f64(v)).collect(),
                    )
                    .unwrap(),
                ),
                data,
            )
        };
        let (t0, d0) = mk(dims_axis[0], 0.0);
        let (t1, d1) = mk(dims_axis[1], 1000.0);
        let (t2, d2) = mk(dims_axis[2], 2000.0);
        let p = params(&[("dimension", "1")]);
        let got = extract_f64_vec(&eval_concatenate(&[t0, t1, t2], &p).unwrap());

        // Naive per-coordinate reference.
        let total_axis: usize = dims_axis.iter().sum();
        let srcs = [
            (&d0, dims_axis[0]),
            (&d1, dims_axis[1]),
            (&d2, dims_axis[2]),
        ];
        let mut want = Vec::with_capacity(outer * total_axis * inner);
        for o in 0..outer {
            for a in 0..total_axis {
                // find operand + local axis coord
                let mut rem = a;
                let mut sel = 0;
                while rem >= srcs[sel].1 {
                    rem -= srcs[sel].1;
                    sel += 1;
                }
                let (data, d_axis) = srcs[sel];
                for k in 0..inner {
                    want.push(data[(o * d_axis + rem) * inner + k]);
                }
            }
        }
        let _ = axis;
        assert_eq!(got, want);
    }

    #[test]
    fn concatenate_pass65_lazy_output_preserves_tensor_contract() -> Result<(), String> {
        let a = v_f64(&[1.0, 2.0]);
        let b = v_f64(&[3.0, 4.0, 5.0]);
        let p = params(&[("dimension", "0")]);
        let result = eval_concatenate(&[a, b], &p).map_err(|err| err.to_string())?;
        let Value::Tensor(mut tensor) = result else {
            return Err("expected tensor".to_owned());
        };

        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.shape.dims, vec![5]);
        assert!(
            tensor.elements.as_f64_slice().is_none(),
            "concat output remains literal-observable storage"
        );
        assert_eq!(
            tensor.elements.to_vec(),
            vec![
                Literal::from_f64(1.0),
                Literal::from_f64(2.0),
                Literal::from_f64(3.0),
                Literal::from_f64(4.0),
                Literal::from_f64(5.0),
            ]
        );

        tensor.elements[0] = Literal::from_f64(9.0);
        assert_eq!(tensor.elements[0], Literal::from_f64(9.0));
        assert_eq!(tensor.elements[1], Literal::from_f64(2.0));
        Ok(())
    }

    // ── Reverse ──

    #[test]
    fn rev_1d() {
        let x = v_f64(&[1.0, 2.0, 3.0, 4.0]);
        let p = params(&[("axes", "0")]);
        let result = eval_rev(&[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn rev_dense_matches_generic() {
        // Dense (F64/I64) rev fast path vs Literal-backed generic path on a rank-3
        // tensor reversing axes 0 and 2 — must match exactly.
        let shape = Shape {
            dims: vec![4, 3, 5],
        };
        let n = 60usize;
        let p = params(&[("axes", "0,2")]);

        let fdata: Vec<f64> = (0..n).map(|i| (i as f64) * 1.5 - 7.0).collect();
        let f_dense =
            Value::Tensor(TensorValue::new_f64_values(shape.clone(), fdata.clone()).unwrap());
        let f_lit = Value::Tensor(
            TensorValue::new(
                DType::F64,
                shape.clone(),
                fdata.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        assert!(
            f_dense
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_some()
        );
        assert!(f_lit.as_tensor().unwrap().elements.as_f64_slice().is_none());
        assert_eq!(
            extract_f64_vec(&eval_rev(std::slice::from_ref(&f_dense), &p).unwrap()),
            extract_f64_vec(&eval_rev(std::slice::from_ref(&f_lit), &p).unwrap()),
            "f64 rev dense vs generic"
        );

        let idata: Vec<i64> = (0..n as i64).map(|i| i * 3 - 11).collect();
        let i_dense =
            Value::Tensor(TensorValue::new_i64_values(shape.clone(), idata.clone()).unwrap());
        let i_lit = Value::Tensor(
            TensorValue::new(
                DType::I64,
                shape.clone(),
                idata.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let ivals = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect()
        };
        assert_eq!(
            ivals(&eval_rev(std::slice::from_ref(&i_dense), &p).unwrap()),
            ivals(&eval_rev(std::slice::from_ref(&i_lit), &p).unwrap()),
            "i64 rev dense vs generic"
        );

        // f32 dense path vs generic (the default ML dtype), bit-identical + dense out.
        let f32d: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 2.0).collect();
        let f32_dense =
            Value::Tensor(TensorValue::new_f32_values(shape.clone(), f32d.clone()).unwrap());
        let f32_lit = Value::Tensor(
            TensorValue::new(
                DType::F32,
                shape.clone(),
                f32d.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let f32bits = |v: &Value| -> Vec<u32> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F32Bits(b) => *b,
                    o => panic!("{o:?}"),
                })
                .collect()
        };
        let d = eval_rev(std::slice::from_ref(&f32_dense), &p).unwrap();
        assert_eq!(
            f32bits(&d),
            f32bits(&eval_rev(std::slice::from_ref(&f32_lit), &p).unwrap()),
            "f32 rev"
        );
        assert!(
            d.as_tensor().unwrap().elements.as_f32_slice().is_some(),
            "f32 rev output dense"
        );

        // bf16 + f16 dense path vs generic.
        for dtype in [DType::BF16, DType::F16] {
            let raw: Vec<u16> = (0..n)
                .map(|i| (i as u16).wrapping_mul(71).wrapping_add(5))
                .collect();
            let mk_lit = |b: u16| {
                if dtype == DType::BF16 {
                    Literal::BF16Bits(b)
                } else {
                    Literal::F16Bits(b)
                }
            };
            let hf_dense = Value::Tensor(
                TensorValue::new_half_float_values(dtype, shape.clone(), raw.clone()).unwrap(),
            );
            let hf_lit = Value::Tensor(
                TensorValue::new(
                    dtype,
                    shape.clone(),
                    raw.iter().copied().map(mk_lit).collect(),
                )
                .unwrap(),
            );
            let hfbits = |v: &Value| -> Vec<u16> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => *b,
                        o => panic!("{o:?}"),
                    })
                    .collect()
            };
            let d = eval_rev(std::slice::from_ref(&hf_dense), &p).unwrap();
            assert_eq!(
                hfbits(&d),
                hfbits(&eval_rev(std::slice::from_ref(&hf_lit), &p).unwrap()),
                "{dtype:?} rev"
            );
            assert!(
                d.as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_some(),
                "{dtype:?} rev output dense"
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_rev_dense_vs_boxed() {
        use std::time::Instant;
        let (rows, cols) = (2048usize, 2048usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let shape = Shape {
            dims: vec![rows as u32, cols as u32],
        };
        let dense =
            Value::Tensor(TensorValue::new_f32_values(shape.clone(), data.clone()).unwrap());
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                shape.clone(),
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let p = params(&[("axes", "1")]);
        let time = |x: &Value| {
            let _ = eval_rev(std::slice::from_ref(x), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = eval_rev(std::slice::from_ref(x), &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 rev axis1 [{rows},{cols}]: boxed(materialize+box)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    // ── Iota ──

    #[test]
    fn iota_1d() {
        let p = params(&[("length", "5"), ("dtype", "F64")]);
        let result = eval_iota(&[], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    // ── One-hot ──

    #[test]
    fn one_hot_basic() {
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3] },
                vec![Literal::I64(0), Literal::I64(1), Literal::I64(2)],
            )
            .unwrap(),
        );
        let p = params(&[("num_classes", "4"), ("dtype", "f64")]);
        let result = eval_one_hot(&[indices], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![3, 4]);
        let vals = extract_f64_vec(&result);
        // Row 0: [1,0,0,0], Row 1: [0,1,0,0], Row 2: [0,0,1,0]
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 0.0);
        assert_eq!(vals[4], 0.0);
        assert_eq!(vals[5], 1.0);
        assert_eq!(vals[10], 1.0);
    }

    /// Dense f32/bf16/f16 one_hot must keep dense output and equal the f64 reference
    /// (default on/off 1.0/0.0 are exact in every float dtype, so `as_f64` matches).
    /// Covers a 2D index input + an out-of-range index (that row stays all-off).
    #[test]
    fn dense_one_hot_f32_half_matches_f64_and_stays_dense() {
        let indices = Value::Tensor(
            TensorValue::new_i64_values(Shape { dims: vec![2, 3] }, vec![0, 1, 5, 2, 3, 1])
                .unwrap(),
        );
        let f64bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };
        let reference = f64bits(
            &eval_one_hot(
                std::slice::from_ref(&indices),
                &params(&[("num_classes", "4"), ("dtype", "f64")]),
            )
            .unwrap(),
        );
        for dt in ["f32", "bf16", "f16"] {
            let r = eval_one_hot(
                std::slice::from_ref(&indices),
                &params(&[("num_classes", "4"), ("dtype", dt)]),
            )
            .unwrap();
            assert_eq!(f64bits(&r), reference, "{dt} one_hot values");
            let t = r.as_tensor().unwrap();
            if dt == "f32" {
                assert!(t.elements.as_f32_slice().is_some(), "f32 one_hot dense");
            } else {
                assert!(
                    t.elements.as_half_float_slice().is_some(),
                    "{dt} one_hot dense"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_one_hot_dense_vs_boxed() {
        use std::time::Instant;
        let (batch, nc) = (8192usize, 1024usize); // [batch] -> [batch, nc]
        let idxs: Vec<i64> = (0..batch as i64).map(|i| (i * 7919) % nc as i64).collect();
        let indices = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![batch as u32],
                },
                idxs.clone(),
            )
            .unwrap(),
        );
        let p = params(&[("num_classes", &nc.to_string()), ("dtype", "f32")]);
        let total = batch * nc;
        let shape = Shape {
            dims: vec![batch as u32, nc as u32],
        };
        let timeit = |f: &dyn Fn()| {
            f();
            let mut best = f64::MAX;
            for _ in 0..15 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        // boxed (old path): Literal fill-off + scatter-on, boxed output.
        let boxed = timeit(&|| {
            let mut out = vec![Literal::from_f32(0.0); total];
            for (i, &idx) in idxs.iter().enumerate() {
                if idx >= 0 && (idx as usize) < nc {
                    out[i * nc + idx as usize] = Literal::from_f32(1.0);
                }
            }
            let _ = TensorValue::new(DType::F32, shape.clone(), out).unwrap();
        });
        let dense = timeit(&|| {
            let _ = eval_one_hot(std::slice::from_ref(&indices), &p).unwrap();
        });
        println!(
            "BENCH f32 one_hot [{batch}]->[{batch},{nc}]: boxed(Literal fill+box)={:.4}ms dense={:.4}ms speedup={:.2}x",
            boxed * 1e3,
            dense * 1e3,
            boxed / dense
        );
    }

    #[test]
    fn one_hot_fill_scatter_matches_reference() {
        // Validate the fill+scatter one_hot (non-last axis, out-of-range indices)
        // against an independent reference built from the spec, for the dense f64
        // path; and confirm a Literal-dtype (i32) result has the same on/off
        // pattern (same shape, on at the right positions).
        let in_rows = 2usize;
        let in_cols = 3usize;
        // indices incl out-of-range (-1 and 5) which must yield all-off rows.
        let idata = [0_i64, 2, -1, 1, 5, 3];
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![in_rows as u32, in_cols as u32],
                },
                idata.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let nc = 4usize; // class axis inserted at axis 1 -> out [2,4,3]
        let p = params(&[
            ("num_classes", "4"),
            ("axis", "1"),
            ("dtype", "f64"),
            ("on_value", "1"),
            ("off_value", "0"),
        ]);
        let out = eval_one_hot(std::slice::from_ref(&indices), &p).unwrap();
        assert_eq!(
            extract_shape(&out),
            vec![in_rows as u32, nc as u32, in_cols as u32]
        );
        let got = extract_f64_vec(&out);

        // Reference: out[r, c, k] = 1 if idata[r*in_cols + k] == c else 0.
        let mut expected = vec![0.0_f64; in_rows * nc * in_cols];
        for r in 0..in_rows {
            for k in 0..in_cols {
                let idx = idata[r * in_cols + k];
                if idx >= 0 && (idx as usize) < nc {
                    expected[r * (nc * in_cols) + (idx as usize) * in_cols + k] = 1.0;
                }
            }
        }
        assert_eq!(got, expected, "one_hot f64 fill+scatter vs reference");

        // i32 (Literal path, exact dtype): on/off pattern + dtype preserved.
        let p32 = params(&[("num_classes", "4"), ("axis", "1"), ("dtype", "i32")]);
        let out32 = eval_one_hot(std::slice::from_ref(&indices), &p32).unwrap();
        assert_eq!(out32.as_tensor().unwrap().dtype, DType::I32);
        let vals32: Vec<i64> = out32
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let expected32: Vec<i64> = expected.iter().map(|&v| v as i64).collect();
        assert_eq!(vals32, expected32, "one_hot i32 fill+scatter vs reference");
    }

    #[test]
    fn one_hot_f64_repeated_patches_materialize_like_dense_scatter() {
        let in_rows = 2usize;
        let in_cols = 3usize;
        let idata = [0_i64, 2, -1, 1, 5, 3];
        let indices = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![in_rows as u32, in_cols as u32],
                },
                idata.to_vec(),
            )
            .unwrap(),
        );
        let p = params(&[
            ("num_classes", "4"),
            ("axis", "1"),
            ("dtype", "f64"),
            ("on_value", "1.5"),
            ("off_value", "-0.0"),
        ]);
        let out = eval_one_hot(std::slice::from_ref(&indices), &p).unwrap();
        let tensor = out.as_tensor().unwrap();
        assert_eq!(
            tensor.shape.dims,
            vec![in_rows as u32, 4_u32, in_cols as u32]
        );
        assert_eq!(tensor.dtype, DType::F64);
        assert!(
            tensor.elements.as_f64_slice().is_none(),
            "F64 one_hot should stay compressed until materialized"
        );

        let got: Vec<u64> = tensor
            .elements
            .iter()
            .map(|literal| literal.as_f64().unwrap().to_bits())
            .collect();
        let on = 1.5_f64.to_bits();
        let off = (-0.0_f64).to_bits();
        let expected = vec![
            on, off, off, off, off, off, off, on, off, off, off, off, off, off, off, on, off, off,
            off, off, off, off, off, on,
        ];
        assert_eq!(got, expected);
    }

    // ── Scatter ──

    #[test]
    fn scatter_scalar_index_accepts_scalar_update() {
        let operand = v_f64(&[1.0, 2.0, 3.0]);
        let index = Value::Scalar(Literal::I64(1));
        let update = Value::Scalar(Literal::from_f64(9.0));
        let result = eval_scatter(&[operand, index, update], &BTreeMap::new()).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![1.0, 9.0, 3.0]);
    }

    #[test]
    fn scatter_scalar_update_rejects_non_scalar_slice() {
        let operand = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let index = Value::Scalar(Literal::I64(1));
        let update = Value::Scalar(Literal::from_f64(9.0));
        let err = eval_scatter(&[operand, index, update], &BTreeMap::new())
            .expect_err("scalar update should not fill non-scalar scatter slices");
        assert!(
            err.to_string().contains("updates must be a tensor"),
            "unexpected error: {err}"
        );
    }

    // ── scatter combiner modes: scatter_max / scatter_min / scatter_mul (jax.lax) ──

    #[test]
    fn scatter_max_f64_duplicate_indices() {
        // operand [1,2,3,4]; scatter-max indices [0,0,2] updates [5,3,1]:
        //   out[0]=max(1,5,3)=5, out[2]=max(3,1)=3 -> [5,2,3,4]. Duplicate indices
        //   accumulate order-independently (max is associative).
        let operand = v_f64(&[1.0, 2.0, 3.0, 4.0]);
        let idx = Value::vector_i64(&[0, 0, 2]).unwrap();
        let upd = v_f64(&[5.0, 3.0, 1.0]);
        let p = params(&[("mode", "max")]);
        let out = eval_scatter(&[operand, idx, upd], &p).unwrap();
        assert_eq!(extract_f64_vec(&out), vec![5.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn scatter_min_f64_duplicate_indices() {
        // operand [5,5,5]; scatter-min indices [0,1,0] updates [2,7,8]:
        //   out[0]=min(5,2,8)=2, out[1]=min(5,7)=5 -> [2,5,5].
        let operand = v_f64(&[5.0, 5.0, 5.0]);
        let idx = Value::vector_i64(&[0, 1, 0]).unwrap();
        let upd = v_f64(&[2.0, 7.0, 8.0]);
        let p = params(&[("mode", "min")]);
        let out = eval_scatter(&[operand, idx, upd], &p).unwrap();
        assert_eq!(extract_f64_vec(&out), vec![2.0, 5.0, 5.0]);
    }

    #[test]
    fn scatter_max_propagates_nan_like_lax_max() {
        // jax.lax.max(x, NaN) == NaN; scatter_max uses the SAME jax_max_f64 the
        // elementwise Max uses, so a NaN update poisons the target.
        let operand = v_f64(&[1.0]);
        let idx = Value::vector_i64(&[0]).unwrap();
        let upd = v_f64(&[f64::NAN]);
        let p = params(&[("mode", "max")]);
        let out = eval_scatter(&[operand, idx, upd], &p).unwrap();
        assert!(
            extract_f64_vec(&out)[0].is_nan(),
            "scatter_max must propagate NaN like lax.max"
        );
    }

    #[test]
    fn scatter_mul_i64_accumulates_via_generic_path() {
        // i64 (boxed): operand [2,3,4]; scatter-mul indices [0,0] updates [3,5]:
        //   out[0]=2*3*5=30 -> [30,3,4]. Exercises the GENERIC combine path (the
        //   dense fast path returns None for mul) and integer wrapping_mul, and the
        //   output stays I64.
        let operand = Value::vector_i64(&[2, 3, 4]).unwrap();
        let idx = Value::vector_i64(&[0, 0]).unwrap();
        let upd = Value::vector_i64(&[3, 5]).unwrap();
        let p = params(&[("mode", "mul")]);
        let out = eval_scatter(&[operand, idx, upd], &p).unwrap();
        let Value::Tensor(t) = &out else {
            panic!("expected tensor")
        };
        assert_eq!(t.dtype, DType::I64);
        let got: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
        assert_eq!(got, vec![30.0, 3.0, 4.0]);
    }

    #[test]
    fn scatter_rejects_unknown_combine_mode() {
        let operand = v_f64(&[1.0, 2.0]);
        let idx = Value::vector_i64(&[0]).unwrap();
        let upd = v_f64(&[9.0]);
        let p = params(&[("mode", "bogus")]);
        let err =
            eval_scatter(&[operand, idx, upd], &p).expect_err("unknown scatter mode must error");
        assert!(
            err.to_string().contains("unknown scatter mode"),
            "unexpected error: {err}"
        );
    }

    // ── Sort ──

    #[test]
    fn sort_1d_ascending() {
        let x = v_f64(&[3.0, 1.0, 4.0, 1.0, 5.0]);
        let p = params(&[
            ("dimension", "0"),
            ("is_stable", "true"),
            ("descending", "false"),
        ]);
        let result = eval_sort(Primitive::Sort, &[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sort_1d_descending() {
        let x = v_f64(&[3.0, 1.0, 4.0]);
        let p = params(&[
            ("dimension", "0"),
            ("is_stable", "true"),
            ("descending", "true"),
        ]);
        let result = eval_sort(Primitive::Sort, &[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![4.0, 3.0, 1.0]);
    }

    #[test]
    fn radix_sort_i64_ascending_matches_comparison_sort() {
        // Large enough (>= RADIX_SORT_MIN_AXIS) to exercise the LSD radix path,
        // with negatives, duplicates, and i64::MIN/MAX. Sort + argsort must match
        // a stable comparison reference exactly (including tie order).
        let n = 1000usize;
        let data: Vec<i64> = (0..n)
            .map(|i| match i % 11 {
                0 => i64::MIN,
                1 => i64::MAX,
                2 => 0,
                3 => -1,
                4 => 7,
                5 => -((i as i64) % 50),
                _ => ((i as i64) * 2654435761).wrapping_rem(100_003) - 50_000,
            })
            .collect();

        let asc = params(&[("dimension", "0"), ("descending", "false")]);

        // Sort parity vs Rust stable sort.
        let sorted = extract_i64_vec(
            &eval_sort(Primitive::Sort, &[Value::vector_i64(&data).unwrap()], &asc).unwrap(),
        );
        let mut want = data.clone();
        want.sort();
        assert_eq!(sorted, want, "radix sort values");

        // Argsort parity vs stable argsort by (value, original index).
        let got_idx = extract_i64_vec(
            &eval_argsort(
                Primitive::Argsort,
                &[Value::vector_i64(&data).unwrap()],
                &asc,
            )
            .unwrap(),
        );
        let mut want_idx: Vec<i64> = (0..n as i64).collect();
        want_idx.sort_by(|&a, &b| {
            data[a as usize]
                .cmp(&data[b as usize])
                .then_with(|| a.cmp(&b))
        });
        assert_eq!(got_idx, want_idx, "radix argsort indices");

        // Multi-slice: [4, 300] sort along last (contiguous) axis, each row
        // independently, all rows long enough for radix.
        let rows = 4usize;
        let cols = 300usize;
        let mat: Vec<i64> = (0..rows * cols)
            .map(|i| ((i as i64) * 48271).wrapping_rem(7919) - 4000)
            .collect();
        let mat_tensor = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                mat.clone(),
            )
            .unwrap(),
        );
        let mat_sorted = extract_i64_vec(
            &eval_sort(
                Primitive::Sort,
                &[mat_tensor],
                &params(&[("dimension", "1")]),
            )
            .unwrap(),
        );
        let mut mat_want = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let mut row = mat[r * cols..(r + 1) * cols].to_vec();
            row.sort();
            mat_want.extend_from_slice(&row);
        }
        assert_eq!(mat_sorted, mat_want, "radix per-row sort");
    }

    #[test]
    fn parallel_radix_matches_serial_radix_with_ties() {
        // The MSD-partition + parallel-per-bucket radix must produce the EXACT same
        // (key,index) permutation as the serial 8-pass LSD radix — both are stable
        // sorts by the same key. Fuzz many key distributions: heavy top-byte ties
        // (most pairs in one MSD bucket), full-range keys, and small-range keys with
        // many equal-key ties (stability), plus boundary sizes around the bucketing.
        for &n in &[2usize, 17, 256, 1000, 70_000, 262_145] {
            for seed in 0u64..6 {
                let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                let mut next = || {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    s
                };
                let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(n);
                for i in 0..n {
                    let r = next();
                    let key = match seed % 3 {
                        0 => r,                            // full range
                        1 => r & 0xff,                     // heavy ties, all in MSD bucket 0
                        _ => (r % 50) | ((r & 0x3) << 56), // few top bytes, many key ties
                    };
                    pairs.push((key, i as u32));
                }
                let mut want = pairs.clone();
                let mut scratch_a: Vec<(u64, u32)> = vec![(0, 0); n];
                radix_pairs_ascending(&mut want, &mut scratch_a);

                let mut got = pairs.clone();
                let mut scratch_b: Vec<(u64, u32)> = Vec::new();
                radix_pairs_ascending_parallel(&mut got, &mut scratch_b);

                assert_eq!(
                    got, want,
                    "parallel radix != serial radix (n={n}, seed={seed})"
                );
            }
        }
    }

    #[test]
    fn parallel_radix_1d_sort_matches_reference() {
        // A single 1-D sort over >= PARALLEL_RADIX_MIN_PAIRS elements routes through
        // the intra-slice parallel radix (outer_count == 1). f64 sort + argsort must
        // equal a stable comparison reference, including ties.
        let n = (1usize << 19) + 123; // > threshold, not a round number
        let raw: Vec<i64> = (0..n)
            .map(|i| (((i as i64) * 2_654_435_761).rem_euclid(401)) - 200) // many ties
            .collect();
        let data: Vec<f64> = raw.iter().map(|&v| v as f64).collect();
        let tensor = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let p = params(&[("dimension", "0"), ("descending", "false")]);

        let sorted = extract_f64_vec(
            &eval_sort(Primitive::Sort, std::slice::from_ref(&tensor), &p).unwrap(),
        );
        let mut want = data.clone();
        want.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(
            sorted.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "parallel-radix 1-D sort values"
        );

        let idx = extract_i64_vec(
            &eval_argsort(Primitive::Argsort, std::slice::from_ref(&tensor), &p).unwrap(),
        );
        let mut want_idx: Vec<i64> = (0..n as i64).collect();
        want_idx.sort_by(|&a, &b| {
            data[a as usize]
                .total_cmp(&data[b as usize])
                .then(a.cmp(&b))
        });
        assert_eq!(idx, want_idx, "parallel-radix 1-D argsort indices");
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_parallel_vs_serial_radix_1d() {
        use std::time::Instant;
        // Single large 1-D radix: serial 8-pass LSD vs MSD-partition + parallel
        // per-bucket. Bit-identical (asserted); isolates the intra-slice parallelism.
        let n = 1usize << 22; // 4.19M pairs
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let pairs0: Vec<(u64, u32)> = (0..n).map(|i| (next(), i as u32)).collect();
        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut b = f64::MAX;
            let mut d = 0u64;
            for _ in 0..5 {
                let t = Instant::now();
                d = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, d)
        };
        let digest = |v: &[(u64, u32)]| v.iter().fold(0u64, |a, &(k, i)| a ^ k ^ (i as u64));

        let p1 = pairs0.clone();
        let (t_serial, d_serial) = best(Box::new(move || {
            let mut p = p1.clone();
            let mut sc = vec![(0u64, 0u32); p.len()];
            radix_pairs_ascending(&mut p, &mut sc);
            digest(&p)
        }));
        let p2 = pairs0.clone();
        let (t_par, d_par) = best(Box::new(move || {
            let mut p = p2.clone();
            let mut sc: Vec<(u64, u32)> = Vec::new();
            radix_pairs_ascending_parallel(&mut p, &mut sc);
            digest(&p)
        }));
        assert_eq!(d_serial, d_par, "parallel radix digest must match serial");
        println!(
            "BENCH 1-D radix sort ({n} pairs): serial={:.4}ms parallel={:.4}ms speedup={:.2}x",
            t_serial * 1e3,
            t_par * 1e3,
            t_serial / t_par,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_threaded_vs_serial_radix_sort() {
        use std::time::Instant;
        // f64 sort(x[R,C], axis=-1): the threaded contiguous-slice driver (eval_sort)
        // vs the identical single-threaded radix loop (same kernel, one shared
        // scratch). Isolates the threading win; both produce bit-identical output.
        let (rows, cols) = (4096usize, 1024usize);
        let total = rows * cols;
        let vals: Vec<f64> = (0..total)
            .map(|i| (((i as i64) * 2654435761).wrapping_rem(1_000_003) - 500_000) as f64 * 0.5)
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                vals.clone(),
            )
            .unwrap(),
        );
        let p = params(&[("dimension", "1"), ("descending", "false")]);
        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut b = f64::MAX;
            let mut digest = 0u64;
            for _ in 0..5 {
                let t = Instant::now();
                digest = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, digest)
        };

        // Serial baseline: the same radix kernel, single-threaded, one scratch.
        let vs = vals.clone();
        let (t_serial, d_serial) = best(Box::new(move || {
            let mut out = vec![0.0_f64; total];
            let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(cols);
            let mut scratch: Vec<(u64, u32)> = vec![(0, 0); cols];
            for (s, out_slice) in out.chunks_mut(cols).enumerate() {
                let in_base = s * cols;
                pairs.clear();
                for i in 0..cols {
                    pairs.push((f64_sort_order_key(vs[in_base + i]), i as u32));
                }
                radix_pairs_ascending(&mut pairs, &mut scratch);
                for (pos, &(_, orig)) in pairs.iter().enumerate() {
                    out_slice[pos] = vs[in_base + orig as usize];
                }
            }
            out.iter().fold(0u64, |a, &v| a ^ v.to_bits())
        }));

        // End-to-end eval_sort (threaded radix + output tensor build), digested
        // zero-copy over the output's f64 slice so the fold doesn't bias timing.
        let (t_threaded, d_threaded) = best(Box::new(move || {
            let out = eval_sort(Primitive::Sort, std::slice::from_ref(&tensor), &p).unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .unwrap()
                .iter()
                .fold(0u64, |a, &v| a ^ v.to_bits())
        }));

        assert_eq!(
            d_serial, d_threaded,
            "threaded sort digest must match serial"
        );
        println!(
            "BENCH f64 sort(x[{rows},{cols}],axis=-1): serial={:.4}ms threaded={:.4}ms speedup={:.2}x digest={d_serial:016x}",
            t_serial * 1e3,
            t_threaded * 1e3,
            t_serial / t_threaded,
        );
    }

    #[test]
    fn threaded_radix_sort_matches_serial_reference_large_multislice() {
        // A multi-slice sort over the contiguous last axis whose total element
        // count exceeds SORT_PARALLEL_MIN_TOTAL_ELEMS (1<<18), so on a multi-core
        // box `for_each_contiguous_sort_slice` fans the slices across threads. Each
        // output slice depends only on its input slice, so the threaded result must
        // be bit-identical to a per-row stable comparison reference for EVERY
        // partition — covering the dense-i64, dense-f64, and literal (f32) radix
        // paths, both sort and argsort, ascending and descending.
        let rows = 2048usize;
        let cols = 256usize; // == RADIX_SORT_MIN_AXIS; rows*cols = 524288 > 1<<18.
        let total = rows * cols;

        // Deterministic spread with duplicates (to exercise stable tie order) and
        // negatives; kept well within i32/f32 exact-integer range.
        let raw: Vec<i64> = (0..total)
            .map(|i| (((i as i64) * 2654435761).wrapping_rem(701)) - 350)
            .collect();

        for descending in [false, true] {
            let p = params(&[
                ("dimension", "1"),
                ("descending", if descending { "true" } else { "false" }),
            ]);

            // Per-row stable reference permutation (value order, ties by index).
            let row_perm = |r: usize, key: &dyn Fn(usize) -> i64| -> Vec<usize> {
                let mut idx: Vec<usize> = (0..cols).collect();
                idx.sort_by(|&a, &b| {
                    let (ka, kb) = (key(r * cols + a), key(r * cols + b));
                    let ord = ka.cmp(&kb);
                    let ord = if descending { ord.reverse() } else { ord };
                    ord.then(a.cmp(&b))
                });
                idx
            };

            // ── dense i64 path ──
            let i64_tensor = Value::Tensor(
                TensorValue::new_i64_values(
                    Shape {
                        dims: vec![rows as u32, cols as u32],
                    },
                    raw.clone(),
                )
                .unwrap(),
            );
            let got_sort = extract_i64_vec(
                &eval_sort(Primitive::Sort, std::slice::from_ref(&i64_tensor), &p).unwrap(),
            );
            let got_arg = extract_i64_vec(
                &eval_argsort(Primitive::Argsort, std::slice::from_ref(&i64_tensor), &p).unwrap(),
            );
            for r in 0..rows {
                let perm = row_perm(r, &|f| raw[f]);
                for (pos, &orig) in perm.iter().enumerate() {
                    assert_eq!(
                        got_sort[r * cols + pos],
                        raw[r * cols + orig],
                        "i64 sort r{r}"
                    );
                    assert_eq!(got_arg[r * cols + pos], orig as i64, "i64 argsort r{r}");
                }
            }

            // ── dense f64 path ──
            let f64_raw: Vec<f64> = raw.iter().map(|&v| v as f64).collect();
            let f64_tensor = Value::Tensor(
                TensorValue::new_f64_values(
                    Shape {
                        dims: vec![rows as u32, cols as u32],
                    },
                    f64_raw.clone(),
                )
                .unwrap(),
            );
            let got_f64 = extract_f64_vec(
                &eval_sort(Primitive::Sort, std::slice::from_ref(&f64_tensor), &p).unwrap(),
            );
            for r in 0..rows {
                let perm = row_perm(r, &|f| raw[f]);
                for (pos, &orig) in perm.iter().enumerate() {
                    assert_eq!(
                        got_f64[r * cols + pos].to_bits(),
                        f64_raw[r * cols + orig].to_bits(),
                        "f64 sort r{r}"
                    );
                }
            }

            // ── literal f32 path ──
            let f32_lits: Vec<Literal> = raw.iter().map(|&v| Literal::from_f32(v as f32)).collect();
            let f32_tensor = Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape {
                        dims: vec![rows as u32, cols as u32],
                    },
                    f32_lits.clone(),
                )
                .unwrap(),
            );
            let got_f32 =
                eval_sort(Primitive::Sort, std::slice::from_ref(&f32_tensor), &p).unwrap();
            let got_f32_bits: Vec<u32> = got_f32
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F32Bits(b) => *b,
                    other => panic!("expected f32, got {other:?}"),
                })
                .collect();
            for r in 0..rows {
                let perm = row_perm(r, &|f| raw[f]);
                for (pos, &orig) in perm.iter().enumerate() {
                    let want = match f32_lits[r * cols + orig] {
                        Literal::F32Bits(b) => b,
                        _ => unreachable!(),
                    };
                    assert_eq!(got_f32_bits[r * cols + pos], want, "f32 sort r{r}");
                }
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_threaded_vs_serial_top_k() {
        use std::time::Instant;
        // f64 top_k(x[R,C], k): threaded eval_top_k vs the identical single-threaded
        // per-slice select loop (same order_top_k_pairs kernel). Both bit-identical;
        // digested zero-copy so the fold doesn't bias timing.
        let (rows, cols, k) = (4096usize, 1024usize, 128usize);
        let total = rows * cols;
        let vals: Vec<f64> = (0..total)
            .map(|i| (((i as i64) * 2_654_435_761).wrapping_rem(1_000_003) - 500_000) as f64 * 0.5)
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                vals.clone(),
            )
            .unwrap(),
        );
        let p = params(&[("k", "128")]);
        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut b = f64::MAX;
            let mut digest = 0u64;
            for _ in 0..5 {
                let t = Instant::now();
                digest = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, digest)
        };

        let vs = vals.clone();
        let (t_serial, d_serial) = best(Box::new(move || {
            let mut out = vec![0.0_f64; rows * k];
            let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(cols);
            let mut scratch: Vec<(u64, u32)> = Vec::new();
            for (s, ochunk) in out.chunks_mut(k).enumerate() {
                let base = s * cols;
                pairs.clear();
                for (i, &v) in vs[base..base + cols].iter().enumerate() {
                    pairs.push((!f64_total_order_key(v), i as u32));
                }
                order_top_k_pairs(&mut pairs, &mut scratch, k);
                for (dst, &(_, orig)) in pairs.iter().take(k).enumerate() {
                    ochunk[dst] = vs[base + orig as usize];
                }
            }
            out.iter().fold(0u64, |a, &v| a ^ v.to_bits())
        }));

        let (t_threaded, d_threaded) = best(Box::new(move || {
            let out = super::eval_top_k(std::slice::from_ref(&tensor), &p).unwrap();
            out[0]
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .unwrap()
                .iter()
                .fold(0u64, |a, &v| a ^ v.to_bits())
        }));

        assert_eq!(
            d_serial, d_threaded,
            "threaded top_k digest must match serial"
        );
        println!(
            "BENCH f64 top_k(x[{rows},{cols}],k={k}): serial={:.4}ms threaded={:.4}ms speedup={:.2}x digest={d_serial:016x}",
            t_serial * 1e3,
            t_threaded * 1e3,
            t_serial / t_threaded,
        );
    }

    #[test]
    fn threaded_top_k_matches_serial_reference_large_multislice() {
        // Multi-slice top-k over the contiguous last axis whose total element count
        // exceeds SORT_PARALLEL_MIN_TOTAL_ELEMS (1<<18), so for_each_top_k_slice
        // fans the slices across threads. Each output slice depends only on its
        // input slice, so the threaded result must equal a per-slice stable
        // reference (k largest by value, ties by ascending index) for EVERY
        // partition — across the dense-i64, dense-f64, and literal (f32) paths.
        let rows = 2048usize;
        let cols = 256usize; // rows*cols = 524288 > 1<<18; cols >= RADIX_SORT_MIN_AXIS.
        let k = 64usize;
        let total = rows * cols;

        // Finite values with frequent ties (to exercise ascending-index tie order),
        // exactly representable in i32/f32 so the three dtype views agree on order.
        let raw: Vec<i64> = (0..total)
            .map(|i| (((i as i64) * 2_654_435_761).rem_euclid(193)) - 96)
            .collect();
        let p = params(&[("k", "64")]);

        // Reference top-k indices for row r: k largest by value, ties by index asc.
        let ref_idx = |r: usize| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..cols).collect();
            idx.sort_by(|&a, &b| {
                raw[r * cols + b]
                    .cmp(&raw[r * cols + a]) // value descending
                    .then(a.cmp(&b)) // tie: index ascending
            });
            idx.truncate(k);
            idx
        };

        // ── dense i64 ──
        let i64_t = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                raw.clone(),
            )
            .unwrap(),
        );
        let oi = super::eval_top_k(std::slice::from_ref(&i64_t), &p).unwrap();
        let (vi, ii) = (extract_i64_vec(&oi[0]), extract_i64_vec(&oi[1]));
        for r in 0..rows {
            let refr = ref_idx(r);
            for (pos, &orig) in refr.iter().enumerate() {
                assert_eq!(vi[r * k + pos], raw[r * cols + orig], "i64 topk val r{r}");
                assert_eq!(ii[r * k + pos], orig as i64, "i64 topk idx r{r}");
            }
        }

        // ── dense f64 ──
        let f64_raw: Vec<f64> = raw.iter().map(|&v| v as f64).collect();
        let f64_t = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                f64_raw.clone(),
            )
            .unwrap(),
        );
        let of = super::eval_top_k(std::slice::from_ref(&f64_t), &p).unwrap();
        let (vf, iff) = (extract_f64_vec(&of[0]), extract_i64_vec(&of[1]));
        for r in 0..rows {
            let refr = ref_idx(r);
            for (pos, &orig) in refr.iter().enumerate() {
                assert_eq!(
                    vf[r * k + pos].to_bits(),
                    f64_raw[r * cols + orig].to_bits(),
                    "f64 topk val r{r}"
                );
                assert_eq!(iff[r * k + pos], orig as i64, "f64 topk idx r{r}");
            }
        }

        // ── literal f32 ──
        let f32_lits: Vec<Literal> = raw.iter().map(|&v| Literal::from_f32(v as f32)).collect();
        let f32_t = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                f32_lits.clone(),
            )
            .unwrap(),
        );
        let o32 = super::eval_top_k(std::slice::from_ref(&f32_t), &p).unwrap();
        let v32_bits: Vec<u32> = o32[0]
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(b) => *b,
                other => panic!("expected f32, got {other:?}"),
            })
            .collect();
        let i32idx = extract_i64_vec(&o32[1]);
        for r in 0..rows {
            let refr = ref_idx(r);
            for (pos, &orig) in refr.iter().enumerate() {
                let want = match f32_lits[r * cols + orig] {
                    Literal::F32Bits(b) => b,
                    _ => unreachable!(),
                };
                assert_eq!(v32_bits[r * k + pos], want, "f32 topk val r{r}");
                assert_eq!(i32idx[r * k + pos], orig as i64, "f32 topk idx r{r}");
            }
        }
    }

    #[test]
    fn radix_sort_f64_ascending_matches_comparison_sort() {
        // Dense (radix) vs Literal-backed (generic total_cmp) ascending sort +
        // argsort over the SAME data including NaN / +-inf / +-0 / dups, compared
        // by bits so distinct NaN payloads and signed zeros are checked exactly.
        let n = 1000usize;
        let data: Vec<f64> = (0..n)
            .map(|i| match i % 13 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => -0.0,
                4 => 0.0,
                5 => f64::from_bits(0x7ff8_0000_0000_0001), // NaN with payload
                6 => -1.5,
                _ => ((i as f64) * 1.000_173).sin() * 1e6 - (i as f64),
            })
            .collect();

        let dense = || {
            Value::Tensor(
                TensorValue::new_f64_values(
                    Shape {
                        dims: vec![n as u32],
                    },
                    data.clone(),
                )
                .unwrap(),
            )
        };
        let literal = || {
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![n as u32],
                    },
                    data.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            )
        };
        assert!(
            dense()
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_some()
        );
        assert!(
            literal()
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_none()
        );
        let asc = params(&[("dimension", "0"), ("descending", "false")]);

        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };

        let d_sort = eval_sort(Primitive::Sort, &[dense()], &asc).unwrap();
        let l_sort = eval_sort(Primitive::Sort, &[literal()], &asc).unwrap();
        assert_eq!(bits(&d_sort), bits(&l_sort), "radix f64 sort vs generic");

        let d_arg = extract_i64_vec(&eval_argsort(Primitive::Argsort, &[dense()], &asc).unwrap());
        let l_arg = extract_i64_vec(&eval_argsort(Primitive::Argsort, &[literal()], &asc).unwrap());
        assert_eq!(d_arg, l_arg, "radix f64 argsort vs generic");
    }

    #[test]
    fn radix_sort_descending_matches_generic() {
        // Descending radix (complement key) must match the generic STABLE
        // descending comparison sort exactly — same order, ascending-index ties.
        let n = 1000usize;
        let desc = params(&[("descending", "true")]); // 1D -> axis 0

        // f64: dense (radix) vs Literal (generic), by bits, incl NaN/inf/-0/dups.
        let fdata: Vec<f64> = (0..n)
            .map(|i| match i % 9 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => -0.0,
                4 => 0.0,
                5 => 3.0,
                _ => ((i as f64) * 1.000_173).sin() * 1e6 - (i as f64),
            })
            .collect();
        let sh = Shape {
            dims: vec![n as u32],
        };
        let f_dense =
            Value::Tensor(TensorValue::new_f64_values(sh.clone(), fdata.clone()).unwrap());
        let f_lit = Value::Tensor(
            TensorValue::new(
                DType::F64,
                sh.clone(),
                fdata.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        assert!(
            f_dense
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_some()
        );
        assert!(f_lit.as_tensor().unwrap().elements.as_f64_slice().is_none());
        let fbits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };
        assert_eq!(
            fbits(&eval_sort(Primitive::Sort, std::slice::from_ref(&f_dense), &desc).unwrap()),
            fbits(&eval_sort(Primitive::Sort, std::slice::from_ref(&f_lit), &desc).unwrap()),
            "f64 desc sort dense vs generic"
        );
        assert_eq!(
            extract_i64_vec(
                &eval_argsort(Primitive::Argsort, std::slice::from_ref(&f_dense), &desc).unwrap()
            ),
            extract_i64_vec(
                &eval_argsort(Primitive::Argsort, std::slice::from_ref(&f_lit), &desc).unwrap()
            ),
            "f64 desc argsort dense vs generic"
        );

        // i64: dense vs Literal.
        let idata: Vec<i64> = (0..n as i64)
            .map(|i| i.wrapping_mul(2_654_435_761) ^ (i % 7))
            .collect();
        let i_dense =
            Value::Tensor(TensorValue::new_i64_values(sh.clone(), idata.clone()).unwrap());
        let i_lit = Value::Tensor(
            TensorValue::new(
                DType::I64,
                sh.clone(),
                idata.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let ivals = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect()
        };
        assert_eq!(
            ivals(&eval_sort(Primitive::Sort, std::slice::from_ref(&i_dense), &desc).unwrap()),
            ivals(&eval_sort(Primitive::Sort, std::slice::from_ref(&i_lit), &desc).unwrap()),
            "i64 desc sort dense vs generic"
        );

        // f32 (Literal-backed radix) vs an independent descending-total_cmp /
        // ascending-index reference.
        let f32data: Vec<f32> = (0..n)
            .map(|i| match i % 7 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => -0.0,
                3 => 5.0,
                _ => ((i as f32) * 0.37).cos() * 100.0,
            })
            .collect();
        let f32t = Value::Tensor(
            TensorValue::new(
                DType::F32,
                sh.clone(),
                f32data
                    .iter()
                    .map(|&v| Literal::F32Bits(v.to_bits()))
                    .collect(),
            )
            .unwrap(),
        );
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            (f32data[b] as f64)
                .total_cmp(&(f32data[a] as f64))
                .then(a.cmp(&b))
        });
        let exp_f32: Vec<u32> = idx.iter().map(|&i| f32data[i].to_bits()).collect();
        let got_f32: Vec<u32> = eval_sort(Primitive::Sort, std::slice::from_ref(&f32t), &desc)
            .unwrap()
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(b) => *b,
                o => panic!("expected F32Bits, got {o:?}"),
            })
            .collect();
        assert_eq!(got_f32, exp_f32, "f32 desc sort vs reference");
    }

    #[test]
    fn radix_sort_f32_ascending_matches_jax_nan_last_reference() {
        // F32 sort/argsort now uses the LSD radix path (axis >= 256). Validate it
        // against an independent stable JAX/numpy reference — finite/inf by
        // total_cmp, ALL NaN (either sign) equal-and-last — over data including
        // NaN / +-inf / +-0 / NaN-payload / dups, compared by bits. Note the two
        // distinct +NaN payloads (0x7fc00000, 0x7fc00001) must stay in INPUT
        // order at the end (NaN-equal), not payload order (which total_cmp gives).
        let n = 1000usize;
        let data: Vec<f32> = (0..n)
            .map(|i| match i % 11 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => -0.0,
                4 => 0.0,
                5 => f32::from_bits(0x7fc0_0001), // NaN with payload
                6 => -1.5,
                _ => ((i as f32) * 1.000_173).sin() * 1e3 - (i as f32),
            })
            .collect();

        let tensor = || {
            Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape {
                        dims: vec![n as u32],
                    },
                    data.iter()
                        .map(|&v| Literal::F32Bits(v.to_bits()))
                        .collect(),
                )
                .unwrap(),
            )
        };
        let asc = params(&[("dimension", "0"), ("descending", "false")]);

        // Reference: stable sort by the JAX/numpy sort order — NaN (either sign)
        // equal-and-last, finite/inf by total_cmp — on the f64 promotion.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            let (x, y) = (data[a] as f64, data[b] as f64);
            match (x.is_nan(), y.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => x.total_cmp(&y),
            }
        });
        let expected_bits: Vec<u32> = idx.iter().map(|&i| data[i].to_bits()).collect();
        let expected_arg: Vec<i64> = idx.iter().map(|&i| i as i64).collect();

        let sorted = eval_sort(Primitive::Sort, &[tensor()], &asc).unwrap();
        let got_bits: Vec<u32> = sorted
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(b) => *b,
                other => panic!("expected F32Bits, got {other:?}"),
            })
            .collect();
        assert_eq!(
            got_bits, expected_bits,
            "radix f32 sort vs JAX nan-last reference"
        );

        let arg = extract_i64_vec(&eval_argsort(Primitive::Argsort, &[tensor()], &asc).unwrap());
        assert_eq!(
            arg, expected_arg,
            "radix f32 argsort vs JAX nan-last reference"
        );
    }

    #[test]
    fn sort_argsort_nan_placement_matches_jax() {
        // Pins fj sort/argsort to JAX 0.10.1 jnp.sort/argsort (CPU): NaN of EITHER
        // sign is the maximum and all NaN compare EQUAL — ascending sends them to
        // the END, descending to the FRONT, both in input (stable) order — while
        // finite/inf keep IEEE order with -0.0 < +0.0. Sort-specific: top_k keeps
        // +NaN-max / -NaN-min (f64_total_order_key), unchanged. Sibling of ds6ny.
        let neg_nan = f32::from_bits(0xFFC0_0000);
        let pos_nan = f32::NAN; // 0x7FC0_0000
        let asc = params(&[("dimension", "0"), ("descending", "false")]);
        let desc = params(&[("dimension", "0"), ("descending", "true")]);
        let f32t = |vals: &[f32]| {
            Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape {
                        dims: vec![vals.len() as u32],
                    },
                    vals.iter()
                        .map(|&v| Literal::F32Bits(v.to_bits()))
                        .collect(),
                )
                .unwrap(),
            )
        };
        let args = |t: &Value, p| {
            extract_i64_vec(&eval_argsort(Primitive::Argsort, std::slice::from_ref(t), p).unwrap())
        };

        // arr indices:   0        1    2        3    4
        let arr = [1.0_f32, pos_nan, 2.0, neg_nan, 0.0];
        let t = f32t(&arr);
        // jnp.argsort: asc [4,0,2,1,3]; desc [1,3,2,0,4] (NaN kept in input order)
        assert_eq!(args(&t, &asc), vec![4, 0, 2, 1, 3], "asc argsort");
        assert_eq!(args(&t, &desc), vec![1, 3, 2, 0, 4], "desc argsort");
        let asc_bits: Vec<u32> = eval_sort(Primitive::Sort, std::slice::from_ref(&t), &asc)
            .unwrap()
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(b) => *b,
                o => panic!("expected F32Bits, got {o:?}"),
            })
            .collect();
        assert_eq!(
            asc_bits,
            vec![
                0.0_f32.to_bits(),
                1.0_f32.to_bits(),
                2.0_f32.to_bits(),
                pos_nan.to_bits(),
                neg_nan.to_bits(),
            ],
            "asc sorted bits: finite then NaN in input order"
        );

        // ±0.0: -0.0 < +0.0 ascending; descending reverses to +0.0 before -0.0.
        let z = f32t(&[5.0, -0.0, 0.0, -5.0]);
        assert_eq!(args(&z, &asc), vec![3, 1, 2, 0], "asc ±0: -5,-0,+0,5");
        assert_eq!(args(&z, &desc), vec![0, 2, 1, 3], "desc ±0: 5,+0,-0,-5");

        // Large (>=256) dense f64 to exercise the radix path: both NaN land last
        // (asc) / first (desc), in input order (-NaN@100 before +NaN@400).
        let mut big: Vec<f64> = (0..512).map(|i| (i as f64).sin()).collect();
        big[100] = f64::from_bits(0xFFF8_0000_0000_0000); // -NaN
        big[400] = f64::NAN; // +NaN
        let bt =
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![512] }, big).unwrap());
        let asc_idx = args(&bt, &asc);
        assert_eq!(
            &asc_idx[510..],
            &[100, 400],
            "asc f64 radix: NaN last, input order"
        );
        let desc_idx = args(&bt, &desc);
        assert_eq!(
            &desc_idx[..2],
            &[100, 400],
            "desc f64 radix: NaN first, input order"
        );
    }

    #[test]
    fn radix_sort_u32_ascending_matches_unsigned_reference() {
        // U32 sort/argsort now uses the LSD radix path. Validate against an
        // independent stable unsigned-value reference, including 0 / u32::MAX /
        // dups, over an axis >= 256 so the radix path is taken.
        let n = 1000usize;
        let data: Vec<u32> = (0..n)
            .map(|i| match i % 7 {
                0 => 0,
                1 => u32::MAX,
                2 => 1,
                3 => (i as u32) % 5, // duplicates
                _ => ((i as u32).wrapping_mul(2_654_435_761)) ^ (i as u32),
            })
            .collect();

        let tensor = || {
            Value::Tensor(
                TensorValue::new(
                    DType::U32,
                    Shape {
                        dims: vec![n as u32],
                    },
                    data.iter().map(|&v| Literal::U32(v)).collect(),
                )
                .unwrap(),
            )
        };
        let asc = params(&[("dimension", "0"), ("descending", "false")]);

        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| data[a].cmp(&data[b]));
        let expected_vals: Vec<u32> = idx.iter().map(|&i| data[i]).collect();
        let expected_arg: Vec<i64> = idx.iter().map(|&i| i as i64).collect();

        let sorted = eval_sort(Primitive::Sort, &[tensor()], &asc).unwrap();
        let got_vals: Option<Vec<u32>> = sorted
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::U32(v) => Some(*v),
                _ => None,
            })
            .collect();
        assert_eq!(
            got_vals.as_deref(),
            Some(expected_vals.as_slice()),
            "radix u32 sort vs unsigned reference"
        );

        let arg = extract_i64_vec(&eval_argsort(Primitive::Argsort, &[tensor()], &asc).unwrap());
        assert_eq!(arg, expected_arg, "radix u32 argsort vs unsigned reference");

        let desc = params(&[("dimension", "0"), ("descending", "true")]);
        let mut desc_idx: Vec<usize> = (0..n).collect();
        desc_idx.sort_by(|&a, &b| data[b].cmp(&data[a]).then(a.cmp(&b)));
        let expected_desc_vals: Vec<u32> = desc_idx.iter().map(|&i| data[i]).collect();
        let expected_desc_arg: Vec<i64> = desc_idx.iter().map(|&i| i as i64).collect();
        let sorted_desc = eval_sort(Primitive::Sort, &[tensor()], &desc).unwrap();
        let got_desc_vals: Option<Vec<u32>> = sorted_desc
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::U32(v) => Some(*v),
                _ => None,
            })
            .collect();
        assert_eq!(
            got_desc_vals.as_deref(),
            Some(expected_desc_vals.as_slice()),
            "descending radix u32 sort vs unsigned reference"
        );
        let desc_arg =
            extract_i64_vec(&eval_argsort(Primitive::Argsort, &[tensor()], &desc).unwrap());
        assert_eq!(
            desc_arg, expected_desc_arg,
            "descending radix u32 argsort vs unsigned reference"
        );
    }

    // ── Argsort ──

    #[test]
    fn argsort_1d() {
        let x = v_f64(&[30.0, 10.0, 20.0]);
        let p = params(&[
            ("dimension", "0"),
            ("is_stable", "true"),
            ("descending", "false"),
        ]);
        let result = eval_argsort(Primitive::Argsort, &[x], &p).unwrap();
        let indices: Vec<i64> = result
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_i64().expect("expected integer index"))
            .collect();
        assert_eq!(indices, vec![1, 2, 0]);
    }

    #[test]
    fn sort_and_argsort_empty_huge_shape_return_empty_tensor() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![u32::MAX, 0, u32::MAX],
                },
                vec![],
            )
            .unwrap(),
        );
        let p = params(&[("dimension", "2")]);

        let sorted = eval_sort(Primitive::Sort, std::slice::from_ref(&input), &p).unwrap();
        let sorted_tensor = sorted.as_tensor().unwrap();
        assert_eq!(sorted_tensor.dtype, DType::F64);
        assert_eq!(sorted_tensor.shape.dims, vec![u32::MAX, 0, u32::MAX]);
        assert!(sorted_tensor.elements.is_empty());

        let argsorted = eval_argsort(Primitive::Argsort, &[input], &p).unwrap();
        let argsorted_tensor = argsorted.as_tensor().unwrap();
        assert_eq!(argsorted_tensor.dtype, DType::I64);
        assert_eq!(argsorted_tensor.shape.dims, vec![u32::MAX, 0, u32::MAX]);
        assert!(argsorted_tensor.elements.is_empty());
    }

    #[test]
    fn sort_and_argsort_reject_malformed_huge_shapes_without_panic() {
        let input = Value::Tensor(TensorValue {
            dtype: DType::F64,
            shape: Shape {
                dims: vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            },
            elements: vec![Literal::from_f64(1.0)].into(),
        });
        let p = params(&[("dimension", "3")]);

        let sort_err = eval_sort(Primitive::Sort, std::slice::from_ref(&input), &p)
            .expect_err("sort should reject overflowing adversarial strides");
        assert!(
            sort_err
                .to_string()
                .contains("sort input strides overflow usize"),
            "unexpected sort error: {sort_err}"
        );

        let argsort_err = eval_argsort(Primitive::Argsort, &[input], &p)
            .expect_err("argsort should reject overflowing adversarial strides");
        assert!(
            argsort_err
                .to_string()
                .contains("sort input strides overflow usize"),
            "unexpected argsort error: {argsort_err}"
        );

        let mismatched = Value::Tensor(TensorValue {
            dtype: DType::F64,
            shape: Shape {
                dims: vec![u32::MAX],
            },
            elements: vec![Literal::from_f64(1.0)].into(),
        });
        let mismatch_err = eval_sort(
            Primitive::Sort,
            &[mismatched],
            &params(&[("dimension", "0")]),
        )
        .expect_err("sort should reject impossible axis lengths before allocation");
        assert!(
            mismatch_err
                .to_string()
                .contains("does not divide 1 input elements"),
            "unexpected mismatch error: {mismatch_err}"
        );
    }

    // ── Argmin / Argmax ──

    #[test]
    fn argmin_1d_finds_minimum_index() {
        let x = v_f64(&[3.0, 1.0, 4.0, 1.0, 5.0]);
        let result = eval_argmin(Primitive::Argmin, &[x], &BTreeMap::new()).unwrap();
        let idx = result.as_i64_scalar().expect("expected scalar i64 result");
        assert_eq!(idx, 1);
    }

    #[test]
    fn argmax_1d_finds_maximum_index() {
        let x = v_f64(&[3.0, 1.0, 4.0, 1.0, 5.0]);
        let result = eval_argmax(Primitive::Argmax, &[x], &BTreeMap::new()).unwrap();
        let idx = result.as_i64_scalar().expect("expected scalar i64 result");
        assert_eq!(idx, 4);
    }

    #[test]
    fn argmax_argmin_dense_matches_generic() {
        // Dense f64 (as_f64_slice) takes the dense fast path; the same data as a
        // Literal-backed tensor takes the generic sort_key/compare_sort_keys path.
        // Indices must match exactly over NaN / +-inf / +-0 / dups (incl. tie
        // first-occurrence) for both argmax and argmin, on a 2D tensor (axis 1).
        let rows = 7usize;
        let cols = 300usize; // >= a few so ties/extrema vary per row
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| match i % 17 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => -0.0,
                4 => 0.0,
                5 => 7.0, // duplicated extremum to exercise first-occurrence ties
                _ => ((i as f64) * 1.000_173).sin() * 10.0 - (i as f64) * 0.001,
            })
            .collect();
        let shape = Shape {
            dims: vec![rows as u32, cols as u32],
        };
        let dense =
            || Value::Tensor(TensorValue::new_f64_values(shape.clone(), data.clone()).unwrap());
        let literal = || {
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    shape.clone(),
                    data.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            )
        };
        assert!(
            dense()
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_some()
        );
        assert!(
            literal()
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_none()
        );
        let p = params(&[("axis", "1")]);

        for prim in [Primitive::Argmax, Primitive::Argmin] {
            let eval = |v: Value| -> Vec<i64> {
                let r = if prim == Primitive::Argmax {
                    eval_argmax(prim, &[v], &p).unwrap()
                } else {
                    eval_argmin(prim, &[v], &p).unwrap()
                };
                extract_i64_vec(&r)
            };
            assert_eq!(eval(dense()), eval(literal()), "{prim:?} dense vs generic");
        }

        // I64 dense path too.
        let idata: Vec<i64> = (0..rows * cols)
            .map(|i| ((i as i64).wrapping_mul(2_654_435_761)) ^ (i as i64 % 9))
            .collect();
        let idense =
            Value::Tensor(TensorValue::new_i64_values(shape.clone(), idata.clone()).unwrap());
        let iliteral = Value::Tensor(
            TensorValue::new(
                DType::I64,
                shape.clone(),
                idata.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        assert_eq!(
            extract_i64_vec(
                &eval_argmax(Primitive::Argmax, std::slice::from_ref(&idense), &p).unwrap()
            ),
            extract_i64_vec(
                &eval_argmax(Primitive::Argmax, std::slice::from_ref(&iliteral), &p).unwrap()
            ),
            "i64 argmax dense vs generic"
        );
        assert_eq!(
            extract_i64_vec(
                &eval_argmin(Primitive::Argmin, std::slice::from_ref(&idense), &p).unwrap()
            ),
            extract_i64_vec(
                &eval_argmin(Primitive::Argmin, std::slice::from_ref(&iliteral), &p).unwrap()
            ),
            "i64 argmin dense vs generic"
        );
    }

    #[test]
    fn argmax_argmin_dense_f32_matches_generic() {
        // Dense F32 (as_f32_slice) takes the new direct fast path; the same data as
        // a boxed-Literal F32 tensor takes the generic per-element `get().as_f64()`
        // float branch. Indices must match exactly over NaN/+-inf/+-0/dups (incl.
        // first-occurrence ties) for both argmax and argmin, axis 1.
        let rows = 7usize;
        let cols = 300usize;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| match i % 17 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => -0.0,
                4 => 0.0,
                5 => 7.0,
                _ => ((i as f32) * 1.000_173).sin() * 10.0 - (i as f32) * 0.001,
            })
            .collect();
        let shape = Shape {
            dims: vec![rows as u32, cols as u32],
        };
        let dense =
            Value::Tensor(TensorValue::new_f32_values(shape.clone(), data.clone()).unwrap());
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                shape.clone(),
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_f32_slice().is_some());
        assert!(boxed.as_tensor().unwrap().elements.as_f32_slice().is_none());
        let p = params(&[("axis", "1")]);
        for prim in [Primitive::Argmax, Primitive::Argmin] {
            let eval = |v: &Value| -> Vec<i64> {
                let r = if prim == Primitive::Argmax {
                    eval_argmax(prim, std::slice::from_ref(v), &p).unwrap()
                } else {
                    eval_argmin(prim, std::slice::from_ref(v), &p).unwrap()
                };
                extract_i64_vec(&r)
            };
            assert_eq!(eval(&dense), eval(&boxed), "f32 {prim:?} dense vs generic");
        }
    }

    #[test]
    fn argmax_argmin_contiguous_simd_float_matches_generic_and_golden()
    -> Result<(), Box<dyn std::error::Error>> {
        fn f32_case() -> (Value, Value) {
            let rows = 5usize;
            let cols = 73usize;
            let mut data: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as f32) * 0.017).sin() * 31.0 - ((i % 19) as f32))
                .collect();
            data[3] = 250.0;
            data[11] = 250.0;
            data[cols + 7] = -250.0;
            data[cols + 29] = -250.0;
            for value in &mut data[2 * cols..3 * cols] {
                *value = 0.0;
            }
            data[2 * cols] = -0.0;
            data[2 * cols + 1] = 0.0;
            data[3 * cols + 13] = f32::from_bits(0xffc0_0001);
            data[3 * cols + 41] = 1_000.0;
            data[3 * cols + 59] = f32::NAN;

            let shape = Shape {
                dims: vec![rows as u32, cols as u32],
            };
            let dense =
                Value::Tensor(TensorValue::new_f32_values(shape.clone(), data.clone()).unwrap());
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    shape,
                    data.into_iter().map(Literal::from_f32).collect(),
                )
                .unwrap(),
            );
            (dense, boxed)
        }

        fn f64_case() -> (Value, Value) {
            let rows = 4usize;
            let cols = 67usize;
            let mut data: Vec<f64> = (0..rows * cols)
                .map(|i| ((i as f64) * 0.023).cos() * 41.0 - ((i % 23) as f64))
                .collect();
            data[5] = 500.0;
            data[17] = 500.0;
            data[cols + 4] = -500.0;
            data[cols + 33] = -500.0;
            for value in &mut data[2 * cols..3 * cols] {
                *value = 0.0;
            }
            data[2 * cols] = 0.0;
            data[2 * cols + 1] = -0.0;
            data[3 * cols + 12] = f64::from_bits(0xfff8_0000_0000_0001);
            data[3 * cols + 30] = -1_000.0;
            data[3 * cols + 52] = f64::NAN;

            let shape = Shape {
                dims: vec![rows as u32, cols as u32],
            };
            let dense =
                Value::Tensor(TensorValue::new_f64_values(shape.clone(), data.clone()).unwrap());
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    shape,
                    data.into_iter().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            );
            (dense, boxed)
        }

        let p = params(&[("axis", "1")]);
        let mut fixtures: Vec<(&str, Vec<i64>)> = Vec::new();
        let (f32_dense, f32_boxed) = f32_case();
        let (f64_dense, f64_boxed) = f64_case();
        for (label, dense, boxed) in [("f32", f32_dense, f32_boxed), ("f64", f64_dense, f64_boxed)]
        {
            for prim in [Primitive::Argmax, Primitive::Argmin] {
                let dense_out = if prim == Primitive::Argmax {
                    eval_argmax(prim, std::slice::from_ref(&dense), &p).unwrap()
                } else {
                    eval_argmin(prim, std::slice::from_ref(&dense), &p).unwrap()
                };
                let boxed_out = if prim == Primitive::Argmax {
                    eval_argmax(prim, std::slice::from_ref(&boxed), &p).unwrap()
                } else {
                    eval_argmin(prim, std::slice::from_ref(&boxed), &p).unwrap()
                };
                let dense_indices = extract_i64_vec(&dense_out);
                assert_eq!(
                    dense_indices,
                    extract_i64_vec(&boxed_out),
                    "{label} {prim:?} dense SIMD vs boxed generic"
                );
                fixtures.push((
                    if prim == Primitive::Argmax {
                        match label {
                            "f32" => "f32_argmax",
                            "f64" => "f64_argmax",
                            _ => unreachable!(),
                        }
                    } else {
                        match label {
                            "f32" => "f32_argmin",
                            "f64" => "f64_argmin",
                            _ => unreachable!(),
                        }
                    },
                    dense_indices,
                ));
            }
        }

        let digest = fj_test_utils::fixture_id_from_json(&fixtures)?;
        assert_eq!(
            digest, "9d58e890fff3ceeba4962617ae365536954e31e42aab7ba4d05f5160e9698e2f",
            "contiguous argmin/argmax digest changed"
        );
        Ok(())
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_argmax_f32_dense_vs_boxed() {
        use std::time::Instant;
        // Decode-time argmax over logits: [batch, vocab] argmax on axis 1.
        let (batch, vocab) = (256usize, 32768usize);
        let data: Vec<f32> = (0..batch * vocab)
            .map(|i| ((i.wrapping_mul(2_654_435_761) % 100_003) as f32) * 0.001 - 50.0)
            .collect();
        let shape = Shape {
            dims: vec![batch as u32, vocab as u32],
        };
        let dense =
            Value::Tensor(TensorValue::new_f32_values(shape.clone(), data.clone()).unwrap());
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                shape.clone(),
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let p = params(&[("axis", "1")]);
        let time = |v: &Value| {
            let _ = eval_argmax(Primitive::Argmax, std::slice::from_ref(v), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = eval_argmax(Primitive::Argmax, std::slice::from_ref(v), &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH argmax f32 [{batch},{vocab}] axis1: boxed(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn argmax_argmin_nan_and_signed_zero_match_jax() {
        // Pins fj to JAX's _ArgMinMaxReducer (jax/_src/lax/lax.py): a NaN of EITHER
        // sign is selected (op_val != op_val), the FIRST NaN wins and is sticky,
        // and ±0.0 compare equal under IEEE so first-occurrence wins ties. Indices
        // below were verified directly against jnp.argmax/argmin (CPU). The prior
        // total_cmp scan missed -NaN (ranked below -inf) and split ±0.0.
        let neg_nan = f64::from_bits(0xFFF8_0000_0000_0000); // sign bit set
        let pos_nan = f64::NAN; // 0x7FF8...
        // (input, expected argmax, expected argmin)
        let cases: &[(&[f64], i64, i64)] = &[
            (&[3.0, neg_nan, 5.0], 1, 1),
            (&[3.0, pos_nan, 5.0], 1, 1),
            (&[3.0, neg_nan, 5.0, neg_nan, 1.0], 1, 1), // first NaN sticky
            (&[neg_nan, 3.0, 5.0], 0, 0),
            (&[-0.0, 0.0], 0, 0),
            (&[0.0, -0.0], 0, 0),
            (&[5.0, -0.0, 0.0, -5.0], 0, 3),
            (&[1.0, 2.0, 3.0], 2, 0), // NaN-free control
        ];
        let p = BTreeMap::new();
        for &(data, want_max, want_min) in cases {
            // Dense f64 storage (as_f64_slice fast path) and Literal-backed storage
            // (generic path) must BOTH match JAX.
            let dense = Value::Tensor(
                TensorValue::new_f64_values(Shape::vector(data.len() as u32), data.to_vec())
                    .unwrap(),
            );
            let literal = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape::vector(data.len() as u32),
                    data.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            );
            assert!(
                literal
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_f64_slice()
                    .is_none()
            );
            for v in [&dense, &literal] {
                let got_max = eval_argmax(Primitive::Argmax, std::slice::from_ref(v), &p)
                    .unwrap()
                    .as_i64_scalar()
                    .unwrap();
                let got_min = eval_argmin(Primitive::Argmin, std::slice::from_ref(v), &p)
                    .unwrap()
                    .as_i64_scalar()
                    .unwrap();
                assert_eq!(got_max, want_max, "argmax {data:?}");
                assert_eq!(got_min, want_min, "argmin {data:?}");
            }
        }
    }

    #[test]
    fn argmin_2d_axis0() {
        let x = mat_f64(2, 3, &[1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
        let p = params(&[("axis", "0")]);
        let result = eval_argmin(Primitive::Argmin, &[x], &p).unwrap();
        let indices: Vec<i64> = result
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_i64().expect("expected integer index"))
            .collect();
        assert_eq!(indices, vec![0, 1, 0]);
    }

    #[test]
    fn argmax_2d_axis1() {
        let x = mat_f64(2, 3, &[1.0, 4.0, 2.0, 3.0, 0.0, 5.0]);
        let p = params(&[("axis", "1")]);
        let result = eval_argmax(Primitive::Argmax, &[x], &p).unwrap();
        let indices: Vec<i64> = result
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_i64().expect("expected integer index"))
            .collect();
        assert_eq!(indices, vec![1, 2]);
    }

    #[test]
    fn argmin_scalar_returns_zero() {
        let x = Value::Scalar(Literal::from_f64(42.0));
        let result = eval_argmin(Primitive::Argmin, &[x], &BTreeMap::new()).unwrap();
        assert_eq!(result.as_i64_scalar(), Some(0));
    }

    // ── Copy ──

    #[test]
    fn copy_preserves_value() {
        let x = v_f64(&[1.0, 2.0, 3.0]);
        let result = eval_copy(std::slice::from_ref(&x)).unwrap();
        assert_eq!(extract_f64_vec(&result), extract_f64_vec(&x));
    }

    // ── ConvertElementType ──

    #[test]
    fn convert_element_type_f64_to_f32() {
        let x = v_f64(&[1.5, 2.5, 3.5]);
        let p = params(&[("new_dtype", "f32")]);
        let result = eval_convert_element_type(&[x], &p).unwrap();
        assert_eq!(result.dtype(), DType::F32);
        match result {
            Value::Tensor(t) => {
                let vals: Vec<f32> = t
                    .elements
                    .iter()
                    .map(|literal| {
                        let Literal::F32Bits(bits) = literal else {
                            return None;
                        };
                        Some(f32::from_bits(*bits))
                    })
                    .collect::<Option<Vec<_>>>()
                    .expect("expected F32Bits");
                assert_eq!(vals, vec![1.5_f32, 2.5_f32, 3.5_f32]);
            }
            other => assert!(
                matches!(other, Value::Tensor(_)),
                "expected tensor, got {other:?}"
            ),
        }
    }

    #[test]
    fn convert_element_type_f64_to_f16_single_rounds() {
        // f64 -> f16 must round once (to-nearest-even), not via f32. This value
        // sits just above the f16 half-ULP tie at 1.0 + 2^-11, by less than the
        // f32 ULP, so a double rounding (f64->f32->f16) collapses it onto the tie
        // and rounds DOWN to f16(1.0) = 0x3C00, while a single direct rounding
        // goes UP to f16(1.0 + 2^-10) = 1.0009765625 = 0x3C01 (matching XLA).
        let x = v_f64(&[1.00048828125 + 2f64.powi(-30)]);
        let p = params(&[("new_dtype", "f16")]);
        let result = eval_convert_element_type(&[x], &p).unwrap();
        assert_eq!(result.dtype(), DType::F16);
        let Value::Tensor(t) = result else {
            panic!("expected tensor");
        };
        let Literal::F16Bits(bits) = t.elements[0] else {
            panic!("expected F16Bits");
        };
        assert_eq!(
            bits, 0x3C01,
            "f64->f16 must round up once (single rounding), got {bits:#06x}"
        );
    }

    #[test]
    fn convert_element_type_f64_to_i64() {
        let x = v_f64(&[1.9, 2.1, -3.7]);
        let p = params(&[("new_dtype", "i64")]);
        let result = eval_convert_element_type(&[x], &p).unwrap();
        assert_eq!(result.dtype(), DType::I64);
        match result {
            Value::Tensor(t) => {
                let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
                assert_eq!(vals, vec![1, 2, -3]);
            }
            other => assert!(
                matches!(other, Value::Tensor(_)),
                "expected tensor, got {other:?}"
            ),
        }
    }

    #[test]
    fn convert_element_type_i64_to_f64() {
        let x = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(3),
                vec![Literal::I64(1), Literal::I64(2), Literal::I64(3)],
            )
            .unwrap(),
        );
        let p = params(&[("new_dtype", "f64")]);
        let result = eval_convert_element_type(&[x], &p).unwrap();
        assert_eq!(result.dtype(), DType::F64);
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn convert_element_type_dense_matches_generic_all_targets() {
        // The dense f64/i64 fast path reuses convert_literal on the reconstructed
        // literal, so it must equal the generic (Literal-backed) path element-by
        // -element (by bits) for every target dtype, incl tricky casts.
        let targets = [
            "f64",
            "f32",
            "f16",
            "bf16",
            "i64",
            "i32",
            "u64",
            "u32",
            "bool",
            "complex64",
            "complex128",
        ];
        let lits = |v: &Value| {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .copied()
                .collect::<Vec<Literal>>()
        };

        // f64 source incl NaN / +-inf / +-0 / negative / fractional / large.
        let fdata = [
            1.9_f64,
            -3.7,
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e19,
            -1e19,
            2.5,
        ];
        let f_dense = Value::Tensor(
            TensorValue::new_f64_values(Shape::vector(fdata.len() as u32), fdata.to_vec()).unwrap(),
        );
        let f_lit = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(fdata.len() as u32),
                fdata.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        assert!(
            f_dense
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_some()
        );
        assert!(f_lit.as_tensor().unwrap().elements.as_f64_slice().is_none());

        // i64 source incl negatives / extremes.
        let idata = [1_i64, -3, 0, i64::MIN, i64::MAX, -1, 42, 7];
        let i_dense = Value::Tensor(
            TensorValue::new_i64_values(Shape::vector(idata.len() as u32), idata.to_vec()).unwrap(),
        );
        let i_lit = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(idata.len() as u32),
                idata.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        assert!(
            i_dense
                .as_tensor()
                .unwrap()
                .elements
                .as_i64_slice()
                .is_some()
        );

        for t in targets {
            let p = params(&[("new_dtype", t)]);
            assert_eq!(
                lits(&eval_convert_element_type(std::slice::from_ref(&f_dense), &p).unwrap()),
                lits(&eval_convert_element_type(std::slice::from_ref(&f_lit), &p).unwrap()),
                "f64 -> {t} dense vs generic"
            );
            assert_eq!(
                lits(&eval_convert_element_type(std::slice::from_ref(&i_dense), &p).unwrap()),
                lits(&eval_convert_element_type(std::slice::from_ref(&i_lit), &p).unwrap()),
                "i64 -> {t} dense vs generic"
            );
        }

        // f32 source incl NaN / +-inf / +-0 / negative / fractional / large.
        let f32data = [
            1.9_f32,
            -3.7,
            0.0,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e19,
            -1e19,
            2.5,
        ];
        let f32_dense = Value::Tensor(
            TensorValue::new_f32_values(Shape::vector(f32data.len() as u32), f32data.to_vec())
                .unwrap(),
        );
        let f32_lit = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape::vector(f32data.len() as u32),
                f32data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        assert!(
            f32_dense
                .as_tensor()
                .unwrap()
                .elements
                .as_f32_slice()
                .is_some()
        );
        assert!(
            f32_lit
                .as_tensor()
                .unwrap()
                .elements
                .as_f32_slice()
                .is_none()
        );
        for t in targets {
            let p = params(&[("new_dtype", t)]);
            assert_eq!(
                lits(&eval_convert_element_type(std::slice::from_ref(&f32_dense), &p).unwrap()),
                lits(&eval_convert_element_type(std::slice::from_ref(&f32_lit), &p).unwrap()),
                "f32 -> {t} dense vs generic"
            );
        }

        // Half-float source (BF16/F16) — the mixed-precision upcast hot path.
        // Hand-built representative bit patterns incl +-0, NaN, +-inf, small/large.
        for src_dt in [DType::BF16, DType::F16] {
            let raw: Vec<u16> = vec![
                0x0000,                                             // +0
                0x8000,                                             // -0
                if src_dt == DType::F16 { 0x3c00 } else { 0x3f80 }, // 1.0
                0x4000,                                             // 2.0 (both layouts)
                0xc000,                                             // -2.0
                if src_dt == DType::F16 { 0x7c00 } else { 0x7f80 }, // +inf
                if src_dt == DType::F16 { 0xfc00 } else { 0xff80 }, // -inf
                if src_dt == DType::F16 { 0x7e00 } else { 0x7fc0 }, // NaN
                0x0001,                                             // smallest subnormal
                0x1234,                                             // arbitrary
            ];
            let h_dense = Value::Tensor(
                TensorValue::new_half_float_values(
                    src_dt,
                    Shape::vector(raw.len() as u32),
                    raw.clone(),
                )
                .unwrap(),
            );
            let mk_lit = |b: u16| {
                if src_dt == DType::BF16 {
                    Literal::BF16Bits(b)
                } else {
                    Literal::F16Bits(b)
                }
            };
            let h_lit = Value::Tensor(
                TensorValue::new(
                    src_dt,
                    Shape::vector(raw.len() as u32),
                    raw.iter().copied().map(mk_lit).collect(),
                )
                .unwrap(),
            );
            assert!(
                h_dense
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_some()
            );
            assert!(
                h_lit
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_none()
            );
            for t in targets {
                let p = params(&[("new_dtype", t)]);
                assert_eq!(
                    lits(&eval_convert_element_type(std::slice::from_ref(&h_dense), &p).unwrap()),
                    lits(&eval_convert_element_type(std::slice::from_ref(&h_lit), &p).unwrap()),
                    "{src_dt:?} -> {t} dense vs generic"
                );
            }
        }

        // u32 source (threefry RNG bits) incl values above i32::MAX (sign edge).
        let u32data = [0u32, 1, 7, u32::MAX, 3_000_000_000, 1 << 31, 42, u32::MAX - 1];
        let u32_dense = Value::Tensor(
            TensorValue::new_u32_values(Shape::vector(u32data.len() as u32), u32data.to_vec())
                .unwrap(),
        );
        let u32_lit = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::U32,
                Shape::vector(u32data.len() as u32),
                fj_core::LiteralBuffer::new(u32data.iter().copied().map(Literal::U32).collect()),
            )
            .unwrap(),
        );
        assert!(u32_dense.as_tensor().unwrap().elements.as_u32_slice().is_some());
        assert!(u32_lit.as_tensor().unwrap().elements.as_u32_slice().is_none());

        // u64 source incl values above i64::MAX (i64 target saturates to 0).
        let u64data = [0u64, 1, u64::MAX, 1 << 63, u32::MAX as u64 + 1, 7, u64::MAX - 1, 42];
        let u64_dense = Value::Tensor(
            TensorValue::new_u64_values(Shape::vector(u64data.len() as u32), u64data.to_vec())
                .unwrap(),
        );
        let u64_lit = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::U64,
                Shape::vector(u64data.len() as u32),
                fj_core::LiteralBuffer::new(u64data.iter().copied().map(Literal::U64).collect()),
            )
            .unwrap(),
        );
        assert!(u64_dense.as_tensor().unwrap().elements.as_u64_slice().is_some());

        for t in targets {
            let p = params(&[("new_dtype", t)]);
            assert_eq!(
                lits(&eval_convert_element_type(std::slice::from_ref(&u32_dense), &p).unwrap()),
                lits(&eval_convert_element_type(std::slice::from_ref(&u32_lit), &p).unwrap()),
                "u32 -> {t} dense vs generic"
            );
            assert_eq!(
                lits(&eval_convert_element_type(std::slice::from_ref(&u64_dense), &p).unwrap()),
                lits(&eval_convert_element_type(std::slice::from_ref(&u64_lit), &p).unwrap()),
                "u64 -> {t} dense vs generic"
            );
        }

        // The hot mixed-precision casts must keep DENSE output storage (no boxing).
        let is_dense = |v: &Value| -> bool {
            let e = &v.as_tensor().unwrap().elements;
            e.as_f64_slice().is_some()
                || e.as_f32_slice().is_some()
                || e.as_half_float_slice().is_some()
        };
        let bf16_dense = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::BF16,
                Shape::vector(3),
                vec![0x3f80, 0x4000, 0xc000],
            )
            .unwrap(),
        );
        for (src, outs) in [
            (&f_dense, ["f32", "f16", "bf16"].as_slice()),
            (&f32_dense, ["f64", "f16", "bf16"].as_slice()),
            (&bf16_dense, ["f32", "f64", "f16"].as_slice()),
        ] {
            for &t in outs {
                let p = params(&[("new_dtype", t)]);
                let got = eval_convert_element_type(std::slice::from_ref(src), &p).unwrap();
                assert!(is_dense(&got), "convert -> {t} must stay dense");
            }
        }
    }

    #[test]
    fn dense_int_to_f32_and_i32_convert_matches_generic() {
        // int→f32 (F32 is JAX's default float) and i64→i32 narrowing previously fell to
        // the per-Literal convert path. Dense (densified i64/i32 source) must match the
        // boxed generic path (forced via new_with_literal_buffer), incl large values
        // exercising f32 rounding (>2^24), i64→f32 double-round (>2^53), i32 wrap (>2^31).
        let data: Vec<i64> = vec![
            0, 1, -1, 7, -7, 100,
            i64::from(i32::MAX), i64::from(i32::MIN),
            16_777_217, 16_777_219, // > 2^24: f32 rounds
            1 << 53, (1 << 53) + 1, // > 2^53: i64->f64 rounds
            5_000_000_000, -123_456_789, 9_007_199_254_740_993,
        ];
        let lits = |v: &Value| v.as_tensor().unwrap().elements.iter().copied().collect::<Vec<Literal>>();
        let n = data.len() as u32;
        let i64_dense = Value::Tensor(TensorValue::new_i64_values(Shape::vector(n), data.clone()).unwrap());
        let i64_boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I64,
                Shape::vector(n),
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::I64(v)).collect()),
            )
            .unwrap(),
        );
        assert!(i64_dense.as_tensor().unwrap().elements.as_i64_slice().is_some());
        assert!(i64_boxed.as_tensor().unwrap().elements.as_i64_slice().is_none());
        let i32data: Vec<i64> = data.iter().map(|&v| i64::from(v as i32)).collect();
        let i32_dense = Value::Tensor(
            TensorValue::new(DType::I32, Shape::vector(n), i32data.iter().map(|&v| Literal::I64(v)).collect()).unwrap(),
        );
        let i32_boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I32,
                Shape::vector(n),
                fj_core::LiteralBuffer::new(i32data.iter().map(|&v| Literal::I64(v)).collect()),
            )
            .unwrap(),
        );
        for t in ["f32", "i32", "f64", "i64", "bf16", "f16"] {
            let p = params(&[("new_dtype", t)]);
            assert_eq!(
                lits(&eval_convert_element_type(std::slice::from_ref(&i64_dense), &p).unwrap()),
                lits(&eval_convert_element_type(std::slice::from_ref(&i64_boxed), &p).unwrap()),
                "i64 -> {t} dense != generic"
            );
            assert_eq!(
                lits(&eval_convert_element_type(std::slice::from_ref(&i32_dense), &p).unwrap()),
                lits(&eval_convert_element_type(std::slice::from_ref(&i32_boxed), &p).unwrap()),
                "i32 -> {t} dense != generic"
            );
        }
    }

    #[test]
    fn dense_complex_float_convert_matches_generic() {
        // float<->complex convert (FFT/signal pipelines) previously ran per-Literal.
        // float->complex: re=v, im=0. complex->float: real-part extraction. complex->
        // complex: precision change. Dense must match the boxed generic path bit-for-bit.
        let lits = |v: &Value| v.as_tensor().unwrap().elements.iter().copied().collect::<Vec<Literal>>();
        // float source -> complex.
        let fd = [1.5_f64, -2.0, 0.0, -0.0, 3.25, -7.0, 1e10];
        let f64_dense = Value::Tensor(TensorValue::new_f64_values(Shape::vector(fd.len() as u32), fd.to_vec()).unwrap());
        let f64_boxed = Value::Tensor(TensorValue::new(DType::F64, Shape::vector(fd.len() as u32), fd.iter().map(|&v| Literal::from_f64(v)).collect()).unwrap());
        assert!(f64_dense.as_tensor().unwrap().elements.as_f64_slice().is_some());
        assert!(f64_boxed.as_tensor().unwrap().elements.as_f64_slice().is_none());
        let f32v: Vec<f32> = fd.iter().map(|&v| v as f32).collect();
        let f32_dense = Value::Tensor(TensorValue::new_f32_values(Shape::vector(f32v.len() as u32), f32v.clone()).unwrap());
        let f32_boxed = Value::Tensor(TensorValue::new(DType::F32, Shape::vector(f32v.len() as u32), f32v.iter().map(|&v| Literal::from_f32(v)).collect()).unwrap());
        for t in ["complex128", "complex64"] {
            let p = params(&[("new_dtype", t)]);
            assert_eq!(lits(&eval_convert_element_type(std::slice::from_ref(&f64_dense), &p).unwrap()), lits(&eval_convert_element_type(std::slice::from_ref(&f64_boxed), &p).unwrap()), "f64 -> {t}");
            assert_eq!(lits(&eval_convert_element_type(std::slice::from_ref(&f32_dense), &p).unwrap()), lits(&eval_convert_element_type(std::slice::from_ref(&f32_boxed), &p).unwrap()), "f32 -> {t}");
        }
        // complex source -> float / complex.
        let cd: Vec<(f64, f64)> = vec![(1.5, -0.5), (2.0, 3.0), (-7.25, 0.0), (0.0, 1.0), (1e10, -1e10)];
        for src_dt in [DType::Complex128, DType::Complex64] {
            let c_dense = Value::Tensor(TensorValue::new_complex_values(src_dt, Shape::vector(cd.len() as u32), cd.clone()).unwrap());
            let c_boxed = Value::Tensor(TensorValue::new_with_literal_buffer(src_dt, Shape::vector(cd.len() as u32), fj_core::LiteralBuffer::new(cd.iter().map(|&(re, im)| if src_dt == DType::Complex64 { Literal::from_complex64(re as f32, im as f32) } else { Literal::from_complex128(re, im) }).collect())).unwrap());
            assert!(c_dense.as_tensor().unwrap().elements.as_complex_slice().is_some());
            assert!(c_boxed.as_tensor().unwrap().elements.as_complex_slice().is_none());
            for t in ["f64", "f32", "complex64", "complex128", "i64", "i32", "bool"] {
                let p = params(&[("new_dtype", t)]);
                assert_eq!(lits(&eval_convert_element_type(std::slice::from_ref(&c_dense), &p).unwrap()), lits(&eval_convert_element_type(std::slice::from_ref(&c_boxed), &p).unwrap()), "{src_dt:?} -> {t}");
            }
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_convert_f64_to_complex128_dense_vs_generic() {
        use std::time::Instant;
        let n = 1_000_000usize;
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 500.0).collect();
        let dense = Value::Tensor(TensorValue::new_f64_values(Shape::vector(n as u32), data.clone()).unwrap());
        let boxed = Value::Tensor(TensorValue::new(DType::F64, Shape::vector(n as u32), data.iter().map(|&v| Literal::from_f64(v)).collect()).unwrap());
        let p = params(&[("new_dtype", "complex128")]);
        let best = |v: &Value| {
            let _ = eval_convert_element_type(std::slice::from_ref(v), &p).unwrap();
            let mut b = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                let o = eval_convert_element_type(std::slice::from_ref(v), &p).unwrap();
                std::hint::black_box(o.as_tensor().unwrap().elements.len());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let generic = best(&boxed);
        let dense_t = best(&dense);
        println!(
            "BENCH convert f64->complex128 [{n}]: generic={:.2}ms dense={:.2}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_convert_u32_to_f32_dense_vs_generic() {
        use std::time::Instant;
        let n = 1_000_000usize;
        let data: Vec<u32> = (0..n).map(|i| (i as u32).wrapping_mul(2_654_435_761)).collect();
        let dense = Value::Tensor(TensorValue::new_u32_values(Shape::vector(n as u32), data.clone()).unwrap());
        let boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::U32,
                Shape::vector(n as u32),
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::U32(v)).collect()),
            )
            .unwrap(),
        );
        let p = params(&[("new_dtype", "f32")]);
        let best = |v: &Value| {
            let _ = eval_convert_element_type(std::slice::from_ref(v), &p).unwrap();
            let mut b = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                let o = eval_convert_element_type(std::slice::from_ref(v), &p).unwrap();
                std::hint::black_box(o.as_tensor().unwrap().elements.len());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let generic = best(&boxed);
        let dense_t = best(&dense);
        println!(
            "BENCH convert u32->f32 [{n}]: generic={:.2}ms dense={:.2}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn dense_float_to_i32_convert_matches_generic() {
        // float->i32 (quantization / float-to-index) previously fell to the per-Literal
        // convert path. Dense (f64/f32 source) must match the boxed generic path, incl
        // NaN (->0), ±inf (saturate i64 then wrap i32), and out-of-i32-range magnitudes.
        let lits = |v: &Value| v.as_tensor().unwrap().elements.iter().copied().collect::<Vec<Literal>>();
        let f64d = [1.9_f64, -3.7, 0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e10, -1e10, 3e9, -2.5, 2_147_483_647.0, 2_147_483_648.0];
        let f64_dense = Value::Tensor(TensorValue::new_f64_values(Shape::vector(f64d.len() as u32), f64d.to_vec()).unwrap());
        let f64_boxed = Value::Tensor(TensorValue::new(DType::F64, Shape::vector(f64d.len() as u32), f64d.iter().copied().map(Literal::from_f64).collect()).unwrap());
        assert!(f64_dense.as_tensor().unwrap().elements.as_f64_slice().is_some());
        assert!(f64_boxed.as_tensor().unwrap().elements.as_f64_slice().is_none());
        let f32d = [1.9_f32, -3.7, 0.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e10, -1e10, 3e9, -2.5];
        let f32_dense = Value::Tensor(TensorValue::new_f32_values(Shape::vector(f32d.len() as u32), f32d.to_vec()).unwrap());
        let f32_boxed = Value::Tensor(TensorValue::new(DType::F32, Shape::vector(f32d.len() as u32), f32d.iter().copied().map(Literal::from_f32).collect()).unwrap());
        assert!(f32_dense.as_tensor().unwrap().elements.as_f32_slice().is_some());
        assert!(f32_boxed.as_tensor().unwrap().elements.as_f32_slice().is_none());
        let p = params(&[("new_dtype", "i32")]);
        assert_eq!(
            lits(&eval_convert_element_type(std::slice::from_ref(&f64_dense), &p).unwrap()),
            lits(&eval_convert_element_type(std::slice::from_ref(&f64_boxed), &p).unwrap()),
            "f64 -> i32 dense != generic"
        );
        assert_eq!(
            lits(&eval_convert_element_type(std::slice::from_ref(&f32_dense), &p).unwrap()),
            lits(&eval_convert_element_type(std::slice::from_ref(&f32_boxed), &p).unwrap()),
            "f32 -> i32 dense != generic"
        );
        // half (bf16/f16) source -> i32.
        let hd = [1.9_f32, -3.7, 0.0, f32::NAN, f32::INFINITY, 100.5, -7.2, 1e6];
        for dt in [DType::BF16, DType::F16] {
            let bits: Vec<u16> = hd
                .iter()
                .map(|&v| match if dt == DType::BF16 { Literal::from_bf16_f64(f64::from(v)) } else { Literal::from_f16_f64(f64::from(v)) } {
                    Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                    _ => 0,
                })
                .collect();
            let h_dense = Value::Tensor(TensorValue::new_half_float_values(dt, Shape::vector(bits.len() as u32), bits.clone()).unwrap());
            let h_boxed = Value::Tensor(TensorValue::new(dt, Shape::vector(bits.len() as u32), bits.iter().map(|&b| if dt == DType::BF16 { Literal::BF16Bits(b) } else { Literal::F16Bits(b) }).collect()).unwrap());
            assert!(h_dense.as_tensor().unwrap().elements.as_half_float_slice().is_some());
            assert!(h_boxed.as_tensor().unwrap().elements.as_half_float_slice().is_none());
            assert_eq!(
                lits(&eval_convert_element_type(std::slice::from_ref(&h_dense), &p).unwrap()),
                lits(&eval_convert_element_type(std::slice::from_ref(&h_boxed), &p).unwrap()),
                "{dt:?} -> i32 dense != generic"
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_convert_int_to_f32_dense_vs_generic() {
        use std::time::Instant;
        let n = 2_000_000usize;
        let data: Vec<i64> = (0..n).map(|i| i64::from((i as i32).wrapping_mul(2_654_435_761u32 as i32))).collect();
        let dense = Value::Tensor(TensorValue::new_i64_values(Shape::vector(n as u32), data.clone()).unwrap());
        let boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I64,
                Shape::vector(n as u32),
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::I64(v)).collect()),
            )
            .unwrap(),
        );
        let p = params(&[("new_dtype", "f32")]);
        let best = |v: &Value| {
            let _ = eval_convert_element_type(std::slice::from_ref(v), &p).unwrap();
            let mut b = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                let o = eval_convert_element_type(std::slice::from_ref(v), &p).unwrap();
                std::hint::black_box(o.as_tensor().unwrap().elements.len());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let generic = best(&boxed);
        let dense_t = best(&dense);
        println!(
            "BENCH convert i64->f32 [{n}]: generic={:.2}ms dense={:.2}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_convert_bf16_to_f32_dense_vs_boxed() {
        use std::time::Instant;
        let n = 1usize << 22; // 4M bf16 -> f32 upcast (mixed-precision compute prologue)
        let raw: Vec<u16> = (0..n)
            .map(|i| ((i as u16).wrapping_mul(37)).wrapping_add(0x3f00))
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_half_float_values(DType::BF16, Shape::vector(n as u32), raw.clone())
                .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::BF16,
                Shape::vector(n as u32),
                raw.iter().copied().map(Literal::BF16Bits).collect(),
            )
            .unwrap(),
        );
        let p = params(&[("new_dtype", "f32")]);
        let time = |x: &Value| {
            let _ = eval_convert_element_type(std::slice::from_ref(x), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = eval_convert_element_type(std::slice::from_ref(x), &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH convert bf16->f32 n={n}: boxed(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_convert_f64_to_f32_dense_vs_boxed() {
        use std::time::Instant;
        let n = 1usize << 22; // 4M elements — mixed-precision activation downcast
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-3 - 2000.0).collect();
        let dense = Value::Tensor(
            TensorValue::new_f64_values(Shape::vector(n as u32), data.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(n as u32),
                data.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let p = params(&[("new_dtype", "f32")]);
        let time = |x: &Value| {
            let _ = eval_convert_element_type(std::slice::from_ref(x), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = eval_convert_element_type(std::slice::from_ref(x), &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH convert f64->f32 n={n}: boxed(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn convert_element_type_scalar() {
        let x = Value::Scalar(Literal::from_f64(std::f64::consts::PI));
        let p = params(&[("new_dtype", "i64")]);
        let result = eval_convert_element_type(&[x], &p).unwrap();
        assert_eq!(result.as_i64_scalar(), Some(3));
    }

    // ── Tile ──

    #[test]
    fn tile_1d_repeat_twice() {
        let x = v_f64(&[1.0, 2.0, 3.0]);
        let p = params(&[("reps", "2")]);
        let result = eval_tile(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![6]);
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn tile_2d_repeat() {
        let x = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let p = params(&[("reps", "2,3")]);
        let result = eval_tile(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![4, 6]);
    }

    #[test]
    fn tile_scalar_repeat() {
        let x = Value::Scalar(Literal::from_f64(5.0));
        let p = params(&[("reps", "3")]);
        let result = eval_tile(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![3]);
        assert_eq!(extract_f64_vec(&result), vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn tile_identity_reps() {
        let x = v_f64(&[1.0, 2.0]);
        let p = params(&[("reps", "1")]);
        let result = eval_tile(&[x], &p).unwrap();
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0]);
    }

    #[test]
    fn dense_tile_matches_literal_path_and_stays_dense() {
        // Dense f64/f32/bf16/f16/i64 tile must be BIT-FOR-BIT identical to the
        // boxed per-`Literal` path AND keep dense output. Multi-axis reps exercise
        // the recursive block-copy traversal (both leaf and interior depths).
        let (rows, cols) = (3usize, 4usize);
        let dims = vec![rows as u32, cols as u32];
        let lits = |v: &Value| v.as_tensor().unwrap().elements.to_vec();
        for p in [
            params(&[("reps", "2,3")]),
            params(&[("reps", "1,4")]),
            params(&[("reps", "5,1")]),
        ] {
            let f64d: Vec<f64> = (0..rows * cols).map(|i| i as f64 * 0.5 - 3.0).collect();
            let dense = Value::Tensor(
                TensorValue::new_f64_values(Shape { dims: dims.clone() }, f64d.clone()).unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: dims.clone() },
                    f64d.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            );
            let d = eval_tile(std::slice::from_ref(&dense), &p).unwrap();
            let l = eval_tile(std::slice::from_ref(&boxed), &p).unwrap();
            assert_eq!(extract_shape(&d), extract_shape(&l), "f64 tile shape");
            assert_eq!(lits(&d), lits(&l), "f64 tile");
            assert!(
                d.as_tensor().unwrap().elements.as_f64_slice().is_some(),
                "f64 tile dense"
            );

            let f32d: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.25 - 1.0).collect();
            let dense = Value::Tensor(
                TensorValue::new_f32_values(Shape { dims: dims.clone() }, f32d.clone()).unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape { dims: dims.clone() },
                    f32d.iter().copied().map(Literal::from_f32).collect(),
                )
                .unwrap(),
            );
            let d = eval_tile(std::slice::from_ref(&dense), &p).unwrap();
            let l = eval_tile(std::slice::from_ref(&boxed), &p).unwrap();
            assert_eq!(lits(&d), lits(&l), "f32 tile");
            assert!(
                d.as_tensor().unwrap().elements.as_f32_slice().is_some(),
                "f32 tile dense"
            );

            for dtype in [DType::BF16, DType::F16] {
                let raw: Vec<u16> = (0..rows * cols)
                    .map(|i| (i as u16).wrapping_mul(67).wrapping_add(5))
                    .collect();
                let mk_lit = move |b: u16| {
                    if dtype == DType::BF16 {
                        Literal::BF16Bits(b)
                    } else {
                        Literal::F16Bits(b)
                    }
                };
                let dense = Value::Tensor(
                    TensorValue::new_half_float_values(
                        dtype,
                        Shape { dims: dims.clone() },
                        raw.clone(),
                    )
                    .unwrap(),
                );
                let boxed = Value::Tensor(
                    TensorValue::new(
                        dtype,
                        Shape { dims: dims.clone() },
                        raw.iter().copied().map(mk_lit).collect(),
                    )
                    .unwrap(),
                );
                let d = eval_tile(std::slice::from_ref(&dense), &p).unwrap();
                let l = eval_tile(std::slice::from_ref(&boxed), &p).unwrap();
                assert_eq!(lits(&d), lits(&l), "{dtype:?} tile");
                assert!(
                    d.as_tensor()
                        .unwrap()
                        .elements
                        .as_half_float_slice()
                        .is_some(),
                    "{dtype:?} tile dense"
                );
            }

            let i64d: Vec<i64> = (0..(rows * cols) as i64).map(|i| i - 6).collect();
            let dense = Value::Tensor(
                TensorValue::new_i64_values(Shape { dims: dims.clone() }, i64d.clone()).unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape { dims: dims.clone() },
                    i64d.iter().copied().map(Literal::I64).collect(),
                )
                .unwrap(),
            );
            let d = eval_tile(std::slice::from_ref(&dense), &p).unwrap();
            let l = eval_tile(std::slice::from_ref(&boxed), &p).unwrap();
            assert_eq!(lits(&d), lits(&l), "i64 tile");
            assert!(
                d.as_tensor().unwrap().elements.as_i64_slice().is_some(),
                "i64 tile dense"
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_tile_f32_dense_vs_boxed() {
        use std::time::Instant;
        let (rows, cols) = (2048usize, 512usize); // tile to [2048, 2048] (reps 1,4)
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: dims.clone() },
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let p = params(&[("reps", "1,4")]);
        let time = |x: &Value| {
            let _ = eval_tile(std::slice::from_ref(x), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = eval_tile(std::slice::from_ref(x), &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH tile f32 [{rows},{cols}]x(1,4): boxed(materialize+box)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    // ── Squeeze ──

    #[test]
    fn squeeze_removes_unit_dim() {
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![1, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );
        let p = params(&[("dimensions", "0")]);
        let result = eval_squeeze(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![3]);
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0]);
    }

    // ── ExpandDims ──

    #[test]
    fn expand_dims_adds_unit_dim() {
        let x = v_f64(&[1.0, 2.0, 3.0]);
        let p = params(&[("axis", "0")]);
        let result = eval_expand_dims(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![1, 3]);
        assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn expand_dims_trailing() {
        let x = v_f64(&[1.0, 2.0]);
        let p = params(&[("axis", "1")]);
        let result = eval_expand_dims(&[x], &p).unwrap();
        assert_eq!(extract_shape(&result), vec![2, 1]);
    }

    // ── normalize_dynamic_start defensive bounds ──

    #[test]
    fn normalize_dynamic_start_clamps_when_window_exceeds_dim() {
        // dim=4, window=10 — caller violates the contract. The function
        // must NOT cast a negative i64 to usize. Result should clamp to 0.
        assert_eq!(super::normalize_dynamic_start(0, 4, 10), 0);
        assert_eq!(super::normalize_dynamic_start(7, 4, 10), 0);
        assert_eq!(super::normalize_dynamic_start(-3, 4, 10), 0);
    }

    #[test]
    fn normalize_dynamic_start_keeps_existing_behavior_when_window_le_dim() {
        // dim=10, window=4 -> max_start = 6
        assert_eq!(super::normalize_dynamic_start(0, 10, 4), 0);
        assert_eq!(super::normalize_dynamic_start(5, 10, 4), 5);
        assert_eq!(super::normalize_dynamic_start(6, 10, 4), 6);
        assert_eq!(super::normalize_dynamic_start(7, 10, 4), 6); // clamp
        assert_eq!(super::normalize_dynamic_start(100, 10, 4), 6); // clamp
        // Negative start interpreted relative to dim: -2 -> dim - 2 = 8 -> clamp to 6
        assert_eq!(super::normalize_dynamic_start(-2, 10, 4), 6);
        // -8 -> dim - 8 = 2
        assert_eq!(super::normalize_dynamic_start(-8, 10, 4), 2);
    }

    #[test]
    fn normalize_dynamic_start_zero_window_pins_to_start() {
        // window = 0 -> max_start = dim
        assert_eq!(super::normalize_dynamic_start(5, 10, 0), 5);
        assert_eq!(super::normalize_dynamic_start(10, 10, 0), 10);
        assert_eq!(super::normalize_dynamic_start(15, 10, 0), 10); // clamp
    }

    // ── TopK ──

    #[test]
    fn top_k_basic() {
        let x = v_f64(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
        let p = params(&[("k", "3")]);
        let outputs = super::eval_top_k(&[x], &p).unwrap();
        assert_eq!(outputs.len(), 2);
        let values = extract_f64_vec(&outputs[0]);
        assert_eq!(values, vec![9.0, 5.0, 4.0]);
        let indices = extract_i64_vec(&outputs[1]);
        assert_eq!(indices, vec![5, 4, 2]);
    }

    #[test]
    fn radix_top_k_dense_matches_generic() {
        // Dense (radix) vs Literal-backed (generic comparison) TopK over a long
        // last axis, including duplicates/ties (tie order = ascending index) and,
        // for f64, NaN / +-inf / +-0. Multi-slice [3, 800]. Compared by bits.
        let rows = 3usize;
        let cols = 800usize;
        let k = 50usize;

        // i64
        let di: Vec<i64> = (0..rows * cols)
            .map(|i| ((i as i64) * 2_654_435_761).rem_euclid(97) - 40) // many ties
            .collect();
        let dense_i = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                di.clone(),
            )
            .unwrap(),
        );
        // Boxed (Vec<Literal>) reference: TensorValue::new now densifies all-I64 inputs
        // (fj-core i64-densify), so build the boxed buffer explicitly to keep exercising
        // the generic path against the dense top_k fast path.
        let lit_i = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I64,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                fj_core::LiteralBuffer::new(di.iter().copied().map(Literal::I64).collect()),
            )
            .unwrap(),
        );
        assert!(
            dense_i
                .as_tensor()
                .unwrap()
                .elements
                .as_i64_slice()
                .is_some()
        );
        assert!(lit_i.as_tensor().unwrap().elements.as_i64_slice().is_none());
        let p = params(&[("k", "50")]);
        let _ = k;
        let dout = super::eval_top_k(&[dense_i], &p).unwrap();
        let lout = super::eval_top_k(&[lit_i], &p).unwrap();
        assert_eq!(
            extract_i64_vec(&dout[0]),
            extract_i64_vec(&lout[0]),
            "i64 topk values"
        );
        assert_eq!(
            extract_i64_vec(&dout[1]),
            extract_i64_vec(&lout[1]),
            "i64 topk indices"
        );

        // f64 with special values + ties
        let df: Vec<f64> = (0..rows * cols)
            .map(|i| match i % 17 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => -0.0,
                4 => 0.0,
                5 => 7.0,
                6 => 7.0, // tie
                _ => ((i as f64) * 0.5).sin() * 100.0,
            })
            .collect();
        let dense_f = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                df.clone(),
            )
            .unwrap(),
        );
        let lit_f = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                df.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let dof = super::eval_top_k(&[dense_f], &p).unwrap();
        let lof = super::eval_top_k(&[lit_f], &p).unwrap();
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };
        assert_eq!(bits(&dof[0]), bits(&lof[0]), "f64 topk values");
        assert_eq!(
            extract_i64_vec(&dof[1]),
            extract_i64_vec(&lof[1]),
            "f64 topk indices"
        );
    }

    /// Bit-exact parity for the dense F64/I64 contiguous-gather fast path vs the
    /// Vec<Literal> copy: gather full rows of a [rows, cols] operand by index
    /// (slice_sizes = [1, cols]), including out-of-bounds indices under clip and
    /// fill_or_drop. Dense operand via new_*_values, literal via new.
    /// Bit-exact parity for the dense F64/I64 scatter fast path vs the
    /// Vec<Literal> path: overwrite and add modes, full-row updates into a
    /// [rows, cols] operand, including out-of-bounds indices (fill_or_drop drops,
    /// clip clamps). Dense via new_*_values, literal via new.
    #[test]
    fn dense_scatter_matches_literal_path() {
        let rows = 16usize;
        let cols = 24usize;
        let dims = vec![rows as u32, cols as u32];
        let idxs = [0_i64, 3, 15, 99, 1, 15]; // 99 OOB, 15 repeated (add accumulates)
        let n_upd = idxs.len();
        let idx = Value::vector_i64(&idxs).unwrap();
        let upd_dims = vec![n_upd as u32, cols as u32];

        for mode in ["overwrite", "add", "mul", "min", "max"] {
            for imode in ["fill_or_drop", "clip"] {
                let p = params(&[("mode", mode), ("index_mode", imode)]);

                // f64
                let opf: Vec<f64> = (0..rows * cols).map(|i| i as f64 * 0.5 - 10.0).collect();
                let updf: Vec<f64> = (0..n_upd * cols).map(|i| i as f64 * 0.25 + 1.0).collect();
                let mk_f = |d: &[f64], dm: &[u32], dense: bool| {
                    if dense {
                        Value::Tensor(
                            TensorValue::new_f64_values(Shape { dims: dm.to_vec() }, d.to_vec())
                                .unwrap(),
                        )
                    } else {
                        Value::Tensor(
                            TensorValue::new(
                                DType::F64,
                                Shape { dims: dm.to_vec() },
                                d.iter().copied().map(Literal::from_f64).collect(),
                            )
                            .unwrap(),
                        )
                    }
                };
                let bits = |v: &Value| -> Vec<u64> {
                    v.as_tensor()
                        .unwrap()
                        .elements
                        .iter()
                        .map(|l| l.as_f64().unwrap().to_bits())
                        .collect()
                };
                let d = super::eval_scatter(
                    &[
                        mk_f(&opf, &dims, true),
                        idx.clone(),
                        mk_f(&updf, &upd_dims, true),
                    ],
                    &p,
                )
                .unwrap();
                let l = super::eval_scatter(
                    &[
                        mk_f(&opf, &dims, false),
                        idx.clone(),
                        mk_f(&updf, &upd_dims, false),
                    ],
                    &p,
                )
                .unwrap();
                assert_eq!(bits(&d), bits(&l), "f64 scatter mode={mode} imode={imode}");

                // i64
                let opi: Vec<i64> = (0..(rows * cols) as i64).map(|i| i - 30).collect();
                let updi: Vec<i64> = (0..(n_upd * cols) as i64).map(|i| i * 2 + 1).collect();
                let mk_i = |d: &[i64], dm: &[u32], dense: bool| {
                    if dense {
                        Value::Tensor(
                            TensorValue::new_i64_values(Shape { dims: dm.to_vec() }, d.to_vec())
                                .unwrap(),
                        )
                    } else {
                        Value::Tensor(
                            TensorValue::new(
                                DType::I64,
                                Shape { dims: dm.to_vec() },
                                d.iter().copied().map(Literal::I64).collect(),
                            )
                            .unwrap(),
                        )
                    }
                };
                let di = super::eval_scatter(
                    &[
                        mk_i(&opi, &dims, true),
                        idx.clone(),
                        mk_i(&updi, &upd_dims, true),
                    ],
                    &p,
                )
                .unwrap();
                let li = super::eval_scatter(
                    &[
                        mk_i(&opi, &dims, false),
                        idx.clone(),
                        mk_i(&updi, &upd_dims, false),
                    ],
                    &p,
                )
                .unwrap();
                assert_eq!(
                    extract_i64_vec(&di),
                    extract_i64_vec(&li),
                    "i64 scatter mode={mode} imode={imode}"
                );
            }
        }
    }

    /// Dense f32 scatter (overwrite + scatter-ADD) must be BIT-FOR-BIT identical to
    /// the generic per-`Literal` path, incl. repeated-index accumulation (index 15)
    /// and OOB (index 99), across fill_or_drop/clip and NaN/±inf/-0.0. scatter-add
    /// matches because the dense `(a as f64 + b as f64) as f32` is exactly what the
    /// generic `binary_literal_op` Add computes for f32.
    #[test]
    fn dense_f32_scatter_matches_literal_path() {
        let (rows, cols) = (16usize, 24usize);
        let dims = vec![rows as u32, cols as u32];
        let idxs = [0_i64, 3, 15, 99, 1, 15]; // 99 OOB, 15 repeated -> add accumulates
        let n_upd = idxs.len();
        let idx = Value::vector_i64(&idxs).unwrap();
        let upd_dims = vec![n_upd as u32, cols as u32];
        let opf: Vec<f32> = (0..rows * cols)
            .map(|i| match i % 13 {
                0 => f32::INFINITY,
                1 => f32::NEG_INFINITY,
                2 => -0.0,
                3 => f32::from_bits(0x7fc0_0001),
                _ => i as f32 * 0.5 - 10.0,
            })
            .collect();
        let updf: Vec<f32> = (0..n_upd * cols).map(|i| i as f32 * 0.25 + 1.0).collect();
        let mk = |d: &[f32], dm: &[u32], dense: bool| {
            if dense {
                Value::Tensor(
                    TensorValue::new_f32_values(Shape { dims: dm.to_vec() }, d.to_vec()).unwrap(),
                )
            } else {
                Value::Tensor(
                    TensorValue::new(
                        DType::F32,
                        Shape { dims: dm.to_vec() },
                        d.iter().copied().map(Literal::from_f32).collect(),
                    )
                    .unwrap(),
                )
            }
        };
        let bits = |v: &Value| -> Vec<u32> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F32Bits(b) => *b,
                    o => panic!("expected f32, got {o:?}"),
                })
                .collect()
        };
        for mode in ["overwrite", "add", "mul", "min", "max"] {
            for imode in ["fill_or_drop", "clip"] {
                let p = params(&[("mode", mode), ("index_mode", imode)]);
                let d = super::eval_scatter(
                    &[
                        mk(&opf, &dims, true),
                        idx.clone(),
                        mk(&updf, &upd_dims, true),
                    ],
                    &p,
                )
                .unwrap();
                let l = super::eval_scatter(
                    &[
                        mk(&opf, &dims, false),
                        idx.clone(),
                        mk(&updf, &upd_dims, false),
                    ],
                    &p,
                )
                .unwrap();
                assert_eq!(
                    d.as_tensor().unwrap().dtype,
                    DType::F32,
                    "f32 scatter dtype {mode} {imode}"
                );
                assert_eq!(bits(&d), bits(&l), "f32 scatter mode={mode} imode={imode}");
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_embedding_scatter_add_dense_vs_generic() {
        use std::time::Instant;
        let (vocab, dim) = (50000usize, 256usize);
        let batch = 8192usize;
        let zeros: Vec<f32> = vec![0.0; vocab * dim];
        let updates: Vec<f32> = (0..batch * dim)
            .map(|i| ((i % 1009) as f32) * 0.001 - 0.5)
            .collect();
        let op_dims = vec![vocab as u32, dim as u32];
        let upd_dims = vec![batch as u32, dim as u32];
        let dense_op = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: op_dims.clone(),
                },
                zeros.clone(),
            )
            .unwrap(),
        );
        let boxed_op = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: op_dims.clone(),
                },
                zeros.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let dense_upd = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: upd_dims.clone(),
                },
                updates.clone(),
            )
            .unwrap(),
        );
        let boxed_upd = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: upd_dims.clone(),
                },
                updates.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let idx: Vec<i64> = (0..batch as i64)
            .map(|i| (i * 7919) % vocab as i64)
            .collect();
        let idx_v = Value::vector_i64(&idx).unwrap();
        let p = params(&[("mode", "add"), ("index_mode", "clip")]);
        let time = |op: &Value, upd: &Value| {
            let _ = super::eval_scatter(&[op.clone(), idx_v.clone(), upd.clone()], &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..15 {
                let t = Instant::now();
                let _ = super::eval_scatter(&[op.clone(), idx_v.clone(), upd.clone()], &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed_op, &boxed_upd);
        let dense_t = time(&dense_op, &dense_upd);
        println!(
            "BENCH f32 embedding scatter-add [{vocab},{dim}] x {batch}: generic(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_scatter_max_dense_vs_generic() {
        use std::time::Instant;
        // scatter-MAX (segment_max / maxpool-backward shape) — now dense for the
        // Min/Max/Mul combiners (previously generic per-Literal). f32 boxed storage
        // (TensorValue::new with F32Bits literals) stays on the generic path, so this
        // is a genuine dense-vs-generic comparison (unlike i64, which densifies).
        let (rows, dim) = (50000usize, 256usize);
        let batch = 8192usize;
        let base: Vec<f32> = vec![f32::NEG_INFINITY; rows * dim];
        let updates: Vec<f32> = (0..batch * dim).map(|i| ((i % 1009) as f32) * 0.001 - 0.5).collect();
        let op_dims = vec![rows as u32, dim as u32];
        let upd_dims = vec![batch as u32, dim as u32];
        let dense_op = Value::Tensor(TensorValue::new_f32_values(Shape { dims: op_dims.clone() }, base.clone()).unwrap());
        let boxed_op = Value::Tensor(TensorValue::new(DType::F32, Shape { dims: op_dims.clone() }, base.iter().copied().map(Literal::from_f32).collect()).unwrap());
        let dense_upd = Value::Tensor(TensorValue::new_f32_values(Shape { dims: upd_dims.clone() }, updates.clone()).unwrap());
        let boxed_upd = Value::Tensor(TensorValue::new(DType::F32, Shape { dims: upd_dims.clone() }, updates.iter().copied().map(Literal::from_f32).collect()).unwrap());
        let idx: Vec<i64> = (0..batch as i64).map(|i| (i * 7919) % rows as i64).collect();
        let idx_v = Value::vector_i64(&idx).unwrap();
        let p = params(&[("mode", "max"), ("index_mode", "clip")]);
        let bits = |v: Value| -> Vec<u32> { v.as_tensor().unwrap().elements.iter().map(|l| match l { Literal::F32Bits(b)=>*b, _=>0 }).collect() };
        let time = |op: &Value, upd: &Value| {
            let _ = super::eval_scatter(&[op.clone(), idx_v.clone(), upd.clone()], &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..10 { let t = Instant::now(); let _ = super::eval_scatter(&[op.clone(), idx_v.clone(), upd.clone()], &p).unwrap(); best = best.min(t.elapsed().as_secs_f64()); }
            best
        };
        assert_eq!(bits(super::eval_scatter(&[dense_op.clone(), idx_v.clone(), dense_upd.clone()], &p).unwrap()),
                   bits(super::eval_scatter(&[boxed_op.clone(), idx_v.clone(), boxed_upd.clone()], &p).unwrap()), "parity");
        let generic = time(&boxed_op, &boxed_upd);
        let dense_t = time(&dense_op, &dense_upd);
        println!(
            "BENCH f32 scatter-max [{rows},{dim}] x {batch}: generic(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3, dense_t * 1e3, generic / dense_t
        );
    }

    #[test]
    fn dense_i32_scatter_matches_generic() {
        // i32 (JAX's default int) scatter (overwrite/add/mul/min/max) now uses the dense
        // i64 path (was generic per-Literal). i32 shares the i64 backing and sign-extends,
        // so the i64 wrapping combiner is identical. Dense (densified) must match the boxed
        // generic path bit-for-bit AND stay I32, incl repeated index (15) + OOB (99).
        let (rows, cols) = (16usize, 24usize);
        let dims = vec![rows as u32, cols as u32];
        let idxs = [0_i64, 3, 15, 99, 1, 15];
        let n = idxs.len();
        let idx = Value::vector_i64(&idxs).unwrap();
        let updims = vec![n as u32, cols as u32];
        // magnitudes large enough that add/mul overflow i32 (exercises the i64-wrap path)
        let opd: Vec<i64> = (0..rows * cols)
            .map(|i| i64::from((i as i32).wrapping_mul(50_021).wrapping_sub(100)))
            .collect();
        let upd: Vec<i64> = (0..n * cols)
            .map(|i| i64::from((i as i32).wrapping_mul(40_009).wrapping_add(7)))
            .collect();
        let mk_d = |d: &[i64], dm: &[u32]| {
            Value::Tensor(
                TensorValue::new(DType::I32, Shape { dims: dm.to_vec() }, d.iter().map(|&v| Literal::I64(v)).collect())
                    .unwrap(),
            )
        };
        let mk_b = |d: &[i64], dm: &[u32]| {
            Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    DType::I32,
                    Shape { dims: dm.to_vec() },
                    fj_core::LiteralBuffer::new(d.iter().map(|&v| Literal::I64(v)).collect()),
                )
                .unwrap(),
            )
        };
        let geti = |v: &Value| -> Vec<i64> {
            v.as_tensor().unwrap().elements.iter().map(|l| match l {
                Literal::I64(x) => *x,
                o => panic!("expected I64-backed i32, got {o:?}"),
            }).collect()
        };
        for mode in ["overwrite", "add", "mul", "min", "max"] {
            for imode in ["fill_or_drop", "clip"] {
                let p = params(&[("mode", mode), ("index_mode", imode)]);
                let d = super::eval_scatter(&[mk_d(&opd, &dims), idx.clone(), mk_d(&upd, &updims)], &p).unwrap();
                let b = super::eval_scatter(&[mk_b(&opd, &dims), idx.clone(), mk_b(&upd, &updims)], &p).unwrap();
                assert_eq!(d.as_tensor().unwrap().dtype, DType::I32, "{mode}/{imode}: dtype must stay I32");
                assert_eq!(geti(&d), geti(&b), "{mode}/{imode}: dense != generic");
            }
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_i32_scatter_dense_vs_generic() {
        use std::time::Instant;
        let (rows, cols) = (20000usize, 128usize);
        let n_upd = 8192usize;
        let dims = vec![rows as u32, cols as u32];
        let updims = vec![n_upd as u32, cols as u32];
        let idxs: Vec<i64> = (0..n_upd).map(|i| ((i * 7919) % rows) as i64).collect();
        let idx = Value::vector_i64(&idxs).unwrap();
        let opd: Vec<i64> = (0..rows * cols).map(|i| i64::from(i as i32)).collect();
        let upd: Vec<i64> = (0..n_upd * cols).map(|i| i64::from(i as i32)).collect();
        let mk_d = |d: &[i64], dm: &[u32]| {
            Value::Tensor(TensorValue::new(DType::I32, Shape { dims: dm.to_vec() }, d.iter().map(|&v| Literal::I64(v)).collect()).unwrap())
        };
        let mk_b = |d: &[i64], dm: &[u32]| {
            Value::Tensor(TensorValue::new_with_literal_buffer(DType::I32, Shape { dims: dm.to_vec() }, fj_core::LiteralBuffer::new(d.iter().map(|&v| Literal::I64(v)).collect())).unwrap())
        };
        let p = params(&[("mode", "add"), ("index_mode", "clip")]);
        let best = |op: &Value, up: &Value| {
            let _ = super::eval_scatter(&[op.clone(), idx.clone(), up.clone()], &p).unwrap();
            let mut t = f64::MAX;
            for _ in 0..5 {
                let s = Instant::now();
                let o = super::eval_scatter(&[op.clone(), idx.clone(), up.clone()], &p).unwrap();
                std::hint::black_box(o.as_tensor().unwrap().elements.len());
                t = t.min(s.elapsed().as_secs_f64());
            }
            t
        };
        let generic = best(&mk_b(&opd, &dims), &mk_b(&upd, &updims));
        let dense = best(&mk_d(&opd, &dims), &mk_d(&upd, &updims));
        println!(
            "BENCH i32 scatter-add [{rows}x{cols}] <- {n_upd} rows: generic={:.2}ms dense={:.2}ms speedup={:.2}x",
            generic * 1e3,
            dense * 1e3,
            generic / dense
        );
    }

    /// Dense BF16/F16 scatter (overwrite + scatter-ADD) must be BIT-FOR-BIT identical
    /// to the generic per-`Literal` path, incl. repeated-index accumulation (15) and
    /// OOB (99), across fill_or_drop/clip and NaN/±inf/±0 bit patterns. scatter-add
    #[test]
    fn complex_scatter_add_mul_overwrite_works() {
        // Complex scatter (add/mul/overwrite) previously ERRORED (binary_literal_op
        // has no complex arm). Now handled by the dense path; verify against a hand
        // reference (component-add / complex-multiply / overwrite, fill_or_drop OOB).
        let (rows, cols) = (8usize, 4usize);
        let dims = vec![rows as u32, cols as u32];
        let idxs = [0_i64, 3, 7, 99, 1, 3]; // 99 OOB (dropped), 3 repeated (accumulates)
        let n = idxs.len();
        let idx = Value::vector_i64(&idxs).unwrap();
        let base: Vec<(f64, f64)> = (0..rows * cols).map(|i| (i as f64 * 0.5, (i % 5) as f64 - 2.0)).collect();
        let upd: Vec<(f64, f64)> = (0..n * cols).map(|i| ((i % 7) as f64 - 3.0, (i % 3) as f64)).collect();
        let mk = |v: &[(f64, f64)], d: &[u32]| Value::Tensor(TensorValue::new_complex_values(DType::Complex128, Shape { dims: d.to_vec() }, v.to_vec()).unwrap());
        let opv = mk(&base, &dims);
        let updv = mk(&upd, &[n as u32, cols as u32]);
        let getc = |v: &Value| -> Vec<(u64, u64)> {
            v.as_tensor().unwrap().elements.iter().map(|l| match l { Literal::Complex128Bits(re, im) => (*re, *im), o => panic!("{o:?}") }).collect()
        };
        for mode in ["overwrite", "add", "mul"] {
            let p = params(&[("mode", mode), ("index_mode", "fill_or_drop")]);
            let got = getc(&super::eval_scatter(&[opv.clone(), idx.clone(), updv.clone()], &p).unwrap());
            // Hand reference.
            let mut r = base.clone();
            for (i, &raw) in idxs.iter().enumerate() {
                if raw < 0 || raw as usize >= rows { continue; }
                let bi = raw as usize * cols;
                let ui = i * cols;
                for j in 0..cols {
                    let (a, b) = r[bi + j];
                    let (c, d) = upd[ui + j];
                    r[bi + j] = match mode {
                        "overwrite" => (c, d),
                        "add" => (a + c, b + d),
                        _ => (a * c - b * d, a * d + b * c),
                    };
                }
            }
            let expect: Vec<(u64, u64)> = r.iter().map(|&(re, im)| (re.to_bits(), im.to_bits())).collect();
            assert_eq!(got, expect, "complex scatter {mode} mismatch");
        }
    }

    #[test]
    fn complex_scatter_min_max_lexicographic() {
        // Complex scatter min/max previously ERRORED (dense path bailed to the
        // Literal path, which has no complex arm). Now folded lexicographically on
        // (re, im) — bit-identically to elementwise complex lax.max/lax.min
        // (`complex_lex_ge`: a >= b ? a : b for Max, b for Min). Includes a real-part
        // TIE (idx 3 update re == base re) to exercise the imaginary tie-breaker, a
        // repeated index, and an OOB drop.
        let (rows, cols) = (8usize, 4usize);
        let dims = vec![rows as u32, cols as u32];
        let idxs = [0_i64, 3, 7, 99, 1, 3]; // 99 OOB (dropped), 3 repeated (folds twice)
        let n = idxs.len();
        let idx = Value::vector_i64(&idxs).unwrap();
        let base: Vec<(f64, f64)> =
            (0..rows * cols).map(|i| (i as f64 * 0.5, (i % 5) as f64 - 2.0)).collect();
        // Make some updates tie the base real part (i*0.5) so the imaginary tie-break fires.
        let upd: Vec<(f64, f64)> = (0..n * cols)
            .map(|i| {
                if i % 4 == 0 {
                    // tie a plausible base real value, vary imaginary
                    ((i % rows * cols) as f64 * 0.5, (i % 7) as f64 - 3.0)
                } else {
                    ((i % 9) as f64 - 4.0, (i % 3) as f64 - 1.0)
                }
            })
            .collect();
        let mk = |v: &[(f64, f64)], d: &[u32]| {
            Value::Tensor(
                TensorValue::new_complex_values(DType::Complex128, Shape { dims: d.to_vec() }, v.to_vec())
                    .unwrap(),
            )
        };
        let opv = mk(&base, &dims);
        let updv = mk(&upd, &[n as u32, cols as u32]);
        let getc = |v: &Value| -> Vec<(u64, u64)> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Complex128Bits(re, im) => (*re, *im),
                    o => panic!("{o:?}"),
                })
                .collect()
        };
        let lex_ge = |a: (f64, f64), b: (f64, f64)| a.0 > b.0 || (a.0 == b.0 && a.1 >= b.1);
        for mode in ["min", "max"] {
            let p = params(&[("mode", mode), ("index_mode", "fill_or_drop")]);
            let got = getc(&super::eval_scatter(&[opv.clone(), idx.clone(), updv.clone()], &p).unwrap());
            let mut r = base.clone();
            for (i, &raw) in idxs.iter().enumerate() {
                if raw < 0 || raw as usize >= rows {
                    continue;
                }
                let bi = raw as usize * cols;
                let ui = i * cols;
                for j in 0..cols {
                    let a = r[bi + j];
                    let b = upd[ui + j];
                    r[bi + j] = if mode == "max" {
                        if lex_ge(a, b) { a } else { b }
                    } else if lex_ge(a, b) {
                        b
                    } else {
                        a
                    };
                }
            }
            let expect: Vec<(u64, u64)> =
                r.iter().map(|&(re, im)| (re.to_bits(), im.to_bits())).collect();
            assert_eq!(got, expect, "complex scatter {mode} mismatch");
        }
    }

    /// matches because the dense path routes through the same `binary_literal_op` Add.
    #[test]
    fn dense_half_float_scatter_matches_literal_path() {
        let (rows, cols) = (16usize, 24usize);
        let dims = vec![rows as u32, cols as u32];
        let idxs = [0_i64, 3, 15, 99, 1, 15]; // 99 OOB, 15 repeated -> add accumulates
        let n_upd = idxs.len();
        let idx = Value::vector_i64(&idxs).unwrap();
        let upd_dims = vec![n_upd as u32, cols as u32];
        for dtype in [DType::BF16, DType::F16] {
            let mk_lit = |b: u16| {
                if dtype == DType::BF16 {
                    Literal::BF16Bits(b)
                } else {
                    Literal::F16Bits(b)
                }
            };
            let opr: Vec<u16> = (0..rows * cols)
                .map(|i| match i % 11 {
                    0 => 0x7f80, // +inf-ish (bf16 +inf=0x7f80)
                    1 => 0xff80,
                    2 => 0x8000, // -0
                    3 => 0x7fc1, // NaN
                    _ => (i as u16).wrapping_mul(53).wrapping_add(17),
                })
                .collect();
            let updr: Vec<u16> = (0..n_upd * cols)
                .map(|i| (i as u16).wrapping_mul(97).wrapping_add(3))
                .collect();
            let mk = |d: &[u16], dm: &[u32], dense: bool| {
                if dense {
                    Value::Tensor(
                        TensorValue::new_half_float_values(
                            dtype,
                            Shape { dims: dm.to_vec() },
                            d.to_vec(),
                        )
                        .unwrap(),
                    )
                } else {
                    Value::Tensor(
                        TensorValue::new(
                            dtype,
                            Shape { dims: dm.to_vec() },
                            d.iter().copied().map(mk_lit).collect(),
                        )
                        .unwrap(),
                    )
                }
            };
            let bits = |v: &Value| -> Vec<u16> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => *b,
                        o => panic!("expected half-float, got {o:?}"),
                    })
                    .collect()
            };
            for mode in ["overwrite", "add", "mul", "min", "max"] {
                for imode in ["fill_or_drop", "clip"] {
                    let p = params(&[("mode", mode), ("index_mode", imode)]);
                    let d = super::eval_scatter(
                        &[
                            mk(&opr, &dims, true),
                            idx.clone(),
                            mk(&updr, &upd_dims, true),
                        ],
                        &p,
                    )
                    .unwrap();
                    let l = super::eval_scatter(
                        &[
                            mk(&opr, &dims, false),
                            idx.clone(),
                            mk(&updr, &upd_dims, false),
                        ],
                        &p,
                    )
                    .unwrap();
                    assert_eq!(
                        d.as_tensor().unwrap().dtype,
                        dtype,
                        "{dtype:?} scatter dtype {mode} {imode}"
                    );
                    assert_eq!(
                        bits(&d),
                        bits(&l),
                        "{dtype:?} scatter mode={mode} imode={imode}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_bf16_embedding_scatter_overwrite_dense_vs_generic() {
        use std::time::Instant;
        let (vocab, dim) = (50000usize, 256usize);
        let batch = 8192usize;
        let zeros: Vec<u16> = vec![0u16; vocab * dim];
        let updates: Vec<u16> = (0..batch * dim)
            .map(|i| (i as u16).wrapping_mul(40503).wrapping_add(7))
            .collect();
        let op_dims = vec![vocab as u32, dim as u32];
        let upd_dims = vec![batch as u32, dim as u32];
        let dense_op = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::BF16,
                Shape {
                    dims: op_dims.clone(),
                },
                zeros.clone(),
            )
            .unwrap(),
        );
        let boxed_op = Value::Tensor(
            TensorValue::new(
                DType::BF16,
                Shape {
                    dims: op_dims.clone(),
                },
                zeros.iter().copied().map(Literal::BF16Bits).collect(),
            )
            .unwrap(),
        );
        let dense_upd = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::BF16,
                Shape {
                    dims: upd_dims.clone(),
                },
                updates.clone(),
            )
            .unwrap(),
        );
        let boxed_upd = Value::Tensor(
            TensorValue::new(
                DType::BF16,
                Shape {
                    dims: upd_dims.clone(),
                },
                updates.iter().copied().map(Literal::BF16Bits).collect(),
            )
            .unwrap(),
        );
        let idx: Vec<i64> = (0..batch as i64)
            .map(|i| (i * 7919) % vocab as i64)
            .collect();
        let idx_v = Value::vector_i64(&idx).unwrap();
        let p = params(&[("mode", "overwrite"), ("index_mode", "clip")]);
        let time = |op: &Value, upd: &Value| {
            let _ = super::eval_scatter(&[op.clone(), idx_v.clone(), upd.clone()], &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..15 {
                let t = Instant::now();
                let _ = super::eval_scatter(&[op.clone(), idx_v.clone(), upd.clone()], &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed_op, &boxed_upd);
        let dense_t = time(&dense_op, &dense_upd);
        println!(
            "BENCH bf16 embedding scatter-overwrite [{vocab},{dim}] x {batch}: generic(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn dense_gather_strided_matches_literal_path() {
        // Non-contiguous gather (partial trailing slice: slice_sizes[1..] < dims[1..])
        // takes the strided dense path. Compare dense (new_*_values) vs boxed
        // (new with literals → generic) across dtypes, clip + fill_or_drop.
        let (rows, cols, pc) = (24usize, 40usize, 13usize); // pc < cols ⇒ strided
        let dims = vec![rows as u32, cols as u32];
        let idx = Value::vector_i64(&[0, 5, 23, 999, 1, 7, 23, 0]).unwrap();
        for imode in ["clip", "fill_or_drop"] {
            let p = params(&[("slice_sizes", &format!("1,{pc}")), ("index_mode", imode)]);
            // f64
            let f: Vec<f64> = (0..rows * cols).map(|i| i as f64 * 0.5 - 7.0).collect();
            let df = Value::Tensor(TensorValue::new_f64_values(Shape { dims: dims.clone() }, f.clone()).unwrap());
            let bf = Value::Tensor(TensorValue::new(DType::F64, Shape { dims: dims.clone() }, f.iter().copied().map(Literal::from_f64).collect()).unwrap());
            let getf = |v: &Value| -> Vec<u64> { v.as_tensor().unwrap().elements.iter().map(|l| l.as_f64().unwrap_or(0.0).to_bits()).collect() };
            assert_eq!(getf(&super::eval_gather(&[df.clone(), idx.clone()], &p).unwrap()),
                       getf(&super::eval_gather(&[bf.clone(), idx.clone()], &p).unwrap()), "f64 strided gather imode={imode}");
            // i64
            let n: Vec<i64> = (0..rows * cols).map(|i| (i as i64).wrapping_mul(7) - 3).collect();
            let dn = Value::Tensor(TensorValue::new_i64_values(Shape { dims: dims.clone() }, n.clone()).unwrap());
            let bn = Value::Tensor(TensorValue::new(DType::I64, Shape { dims: dims.clone() }, n.iter().map(|&v| Literal::I64(v)).collect()).unwrap());
            let geti = |v: &Value| -> Vec<i64> { v.as_tensor().unwrap().elements.iter().map(|l| match l { Literal::I64(x)=>*x, _=>0 }).collect() };
            assert_eq!(geti(&super::eval_gather(&[dn.clone(), idx.clone()], &p).unwrap()),
                       geti(&super::eval_gather(&[bn.clone(), idx.clone()], &p).unwrap()), "i64 strided gather imode={imode}");
            // bf16
            let bits = |x: f64| match Literal::from_bf16_f64(x) { Literal::BF16Bits(b)=>b, _=>0 };
            let h: Vec<u16> = (0..rows * cols).map(|i| bits((i % 19) as f64 * 0.3 - 2.0)).collect();
            let dh = Value::Tensor(TensorValue::new_half_float_values(DType::BF16, Shape { dims: dims.clone() }, h.clone()).unwrap());
            let bh = Value::Tensor(TensorValue::new(DType::BF16, Shape { dims: dims.clone() }, h.iter().map(|&b| Literal::BF16Bits(b)).collect()).unwrap());
            let geth = |v: &Value| -> Vec<u16> { v.as_tensor().unwrap().elements.iter().map(|l| match l { Literal::BF16Bits(b)|Literal::F16Bits(b)=>*b, _=>0 }).collect() };
            assert_eq!(geth(&super::eval_gather(&[dh.clone(), idx.clone()], &p).unwrap()),
                       geth(&super::eval_gather(&[bh.clone(), idx.clone()], &p).unwrap()), "bf16 strided gather imode={imode}");
        }
    }

    #[test]
    fn dense_i32_gather_matches_generic_and_preserves_dtype() {
        // i32 (JAX's default int) gather, contiguous + strided. i32 densifies into the
        // i64 backing, so the strided path (gated on as_i64_slice) USED to emit I64 with
        // the i64::MIN OOB fill — a widening + wrong-fill bug; now dtype-gated. The dense
        // (densified) path must match the boxed generic path bit-for-bit AND stay I32,
        // including the i32::MIN OOB fill under fill_or_drop.
        let (rows, cols) = (20usize, 24usize);
        let dims = vec![rows as u32, cols as u32];
        let idx = Value::vector_i64(&[0, 5, 19, 999, 1, 7, 19, 0]).unwrap(); // 999 is OOB
        let data: Vec<i64> = (0..rows * cols)
            .map(|i| i64::from((i as i32).wrapping_mul(7).wrapping_sub(3)))
            .collect();
        let dense = Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape { dims: dims.clone() },
                data.iter().map(|&v| Literal::I64(v)).collect(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I32,
                Shape { dims: dims.clone() },
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::I64(v)).collect()),
            )
            .unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_i64_slice().is_some());
        assert!(boxed.as_tensor().unwrap().elements.as_i64_slice().is_none());
        let geti = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::I64(x) => *x,
                    o => panic!("expected I64-backed i32, got {o:?}"),
                })
                .collect()
        };
        for (ss, label) in [(format!("1,{cols}"), "contiguous"), ("1,13".to_owned(), "strided")] {
            for imode in ["clip", "fill_or_drop"] {
                let p = params(&[("slice_sizes", &ss), ("index_mode", imode)]);
                let d = super::eval_gather(&[dense.clone(), idx.clone()], &p).unwrap();
                let b = super::eval_gather(&[boxed.clone(), idx.clone()], &p).unwrap();
                assert_eq!(
                    d.as_tensor().unwrap().dtype,
                    DType::I32,
                    "i32 {label} gather must stay I32 (imode={imode})"
                );
                assert_eq!(
                    b.as_tensor().unwrap().dtype,
                    DType::I32,
                    "i32 {label} gather generic dtype (imode={imode})"
                );
                assert_eq!(geti(&d), geti(&b), "i32 {label} gather imode={imode}");
            }
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_i32_strided_gather_dense_vs_generic() {
        use std::time::Instant;
        let (rows, cols, pc) = (50000usize, 256usize, 96usize); // pc < cols ⇒ strided
        let dims = vec![rows as u32, cols as u32];
        let idxs: Vec<i64> = (0..8192).map(|i| ((i * 7919) % rows) as i64).collect();
        let idx = Value::vector_i64(&idxs).unwrap();
        let data: Vec<i64> = (0..rows * cols).map(|i| i64::from(i as i32)).collect();
        let dense = Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape { dims: dims.clone() },
                data.iter().map(|&v| Literal::I64(v)).collect(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I32,
                Shape { dims: dims.clone() },
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::I64(v)).collect()),
            )
            .unwrap(),
        );
        let p = params(&[("slice_sizes", &format!("1,{pc}")), ("index_mode", "clip")]);
        let best = |v: &Value| {
            let _ = super::eval_gather(&[v.clone(), idx.clone()], &p).unwrap();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                let o = super::eval_gather(&[v.clone(), idx.clone()], &p).unwrap();
                std::hint::black_box(o.as_tensor().unwrap().elements.len());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let generic = best(&boxed);
        let dense_t = best(&dense);
        println!(
            "BENCH i32 strided gather [{rows},{cols}]->[8192,{pc}]: generic={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn dense_complex_gather_matches_literal_path() {
        // Complex gather (contiguous full-row AND strided partial-row) dense path
        // vs boxed-Literal generic. Bit-identical (same indices/order/OOB fill).
        let (rows, cols) = (24usize, 16usize);
        let dims = vec![rows as u32, cols as u32];
        let idx = Value::vector_i64(&[0, 5, 23, 999, 1, 23, 0]).unwrap();
        let cplx: Vec<(f64, f64)> = (0..rows * cols)
            .map(|i| ((i as f64) * 0.5 - 7.0, ((i * 3 % 11) as f64) - 5.0))
            .collect();
        let dense = Value::Tensor(TensorValue::new_complex_values(DType::Complex128, Shape { dims: dims.clone() }, cplx.clone()).unwrap());
        let boxed = Value::Tensor(TensorValue::new(DType::Complex128, Shape { dims: dims.clone() }, cplx.iter().map(|&(re,im)| Literal::from_complex128(re,im)).collect()).unwrap());
        let bits = |v: &Value| -> Vec<(u64, u64)> {
            v.as_tensor().unwrap().elements.iter().map(|l| match l {
                Literal::Complex128Bits(re, im) => (*re, *im),
                o => panic!("expected Complex128Bits, got {o:?}"),
            }).collect()
        };
        for (ss, label) in [("1,16", "contiguous"), ("1,9", "strided")] {
            for imode in ["clip", "fill_or_drop"] {
                let p = params(&[("slice_sizes", ss), ("index_mode", imode)]);
                assert_eq!(
                    bits(&super::eval_gather(&[dense.clone(), idx.clone()], &p).unwrap()),
                    bits(&super::eval_gather(&[boxed.clone(), idx.clone()], &p).unwrap()),
                    "complex {label} gather imode={imode}"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_complex_gather_dense_vs_generic() {
        use std::time::Instant;
        let (vocab, dim) = (50000usize, 128usize);
        let batch = 8192usize;
        let cplx: Vec<(f64, f64)> = (0..vocab * dim).map(|i| ((i % 1009) as f64 * 0.001, (i % 503) as f64 * 0.002)).collect();
        let dims = vec![vocab as u32, dim as u32];
        let dense = Value::Tensor(TensorValue::new_complex_values(DType::Complex128, Shape { dims: dims.clone() }, cplx.clone()).unwrap());
        let boxed = Value::Tensor(TensorValue::new(DType::Complex128, Shape { dims: dims.clone() }, cplx.iter().map(|&(re,im)| Literal::from_complex128(re,im)).collect()).unwrap());
        let idx: Vec<i64> = (0..batch as i64).map(|i| (i * 7919) % vocab as i64).collect();
        let idx_v = Value::vector_i64(&idx).unwrap();
        let pc = 48usize; // partial trailing slice ⇒ strided path (per-element get/push)
        let p = params(&[("slice_sizes", &format!("1,{pc}")), ("index_mode", "clip")]);
        let csum = |v: Value| -> u64 { v.as_tensor().unwrap().elements.iter().fold(0u64, |a, l| match l { Literal::Complex128Bits(re,_)=>a.wrapping_add(*re), _=>a }) };
        let time = |op: &Value| {
            let _ = super::eval_gather(&[op.clone(), idx_v.clone()], &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..10 { let t = Instant::now(); let _ = super::eval_gather(&[op.clone(), idx_v.clone()], &p).unwrap(); best = best.min(t.elapsed().as_secs_f64()); }
            best
        };
        assert_eq!(csum(super::eval_gather(&[dense.clone(), idx_v.clone()], &p).unwrap()),
                   csum(super::eval_gather(&[boxed.clone(), idx_v.clone()], &p).unwrap()), "parity");
        let g = time(&boxed); let d = time(&dense);
        println!("BENCH complex128 strided gather [{vocab},{dim}]->[{batch},{pc}]: generic={:.4}ms dense={:.4}ms speedup={:.2}x", g*1e3, d*1e3, g/d);
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_strided_gather_dense_vs_generic() {
        use std::time::Instant;
        // Partial-column gather of a large [vocab, dim] f32 table (gather first `pc`
        // columns of each selected row) — strided dense path vs generic per-Literal.
        let (vocab, dim, pc) = (50000usize, 256usize, 96usize);
        let batch = 8192usize;
        let data: Vec<f32> = (0..vocab * dim).map(|i| ((i % 1009) as f32) * 0.001 - 0.5).collect();
        let dims = vec![vocab as u32, dim as u32];
        let dense = Value::Tensor(TensorValue::new_f32_values(Shape { dims: dims.clone() }, data.clone()).unwrap());
        let boxed = Value::Tensor(TensorValue::new(DType::F32, Shape { dims: dims.clone() }, data.iter().copied().map(Literal::from_f32).collect()).unwrap());
        let idx: Vec<i64> = (0..batch as i64).map(|i| (i * 7919) % vocab as i64).collect();
        let idx_v = Value::vector_i64(&idx).unwrap();
        let p = params(&[("slice_sizes", &format!("1,{pc}")), ("index_mode", "clip")]);
        let bits = |v: Value| -> Vec<u32> { v.as_tensor().unwrap().elements.iter().map(|l| match l { Literal::F32Bits(b)=>*b, _=>0 }).collect() };
        let time = |op: &Value| {
            let _ = eval_gather(&[op.clone(), idx_v.clone()], &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..10 { let t = Instant::now(); let _ = eval_gather(&[op.clone(), idx_v.clone()], &p).unwrap(); best = best.min(t.elapsed().as_secs_f64()); }
            best
        };
        assert_eq!(bits(super::eval_gather(&[dense.clone(), idx_v.clone()], &p).unwrap()),
                   bits(super::eval_gather(&[boxed.clone(), idx_v.clone()], &p).unwrap()), "parity");
        let g = time(&boxed); let d = time(&dense);
        println!("BENCH f32 strided gather [{vocab},{dim}]->[{batch},{pc}]: generic={:.4}ms dense={:.4}ms speedup={:.2}x", g*1e3, d*1e3, g/d);
    }

    #[test]
    fn dense_gather_contiguous_matches_literal_path() {
        let rows = 32usize;
        let cols = 40usize;
        let dims = vec![rows as u32, cols as u32];
        let idx = Value::vector_i64(&[0, 5, 31, 999, 1, 7, 31, 0, 12, 999]).unwrap();
        for mode in ["clip", "fill_or_drop"] {
            let params = params(&[("slice_sizes", "1,40"), ("index_mode", mode)]);

            let f: Vec<f64> = (0..rows * cols).map(|i| (i as f64 - 100.0) * 0.5).collect();
            let dense_f = Value::Tensor(
                TensorValue::new_f64_values(Shape { dims: dims.clone() }, f.clone()).unwrap(),
            );
            assert!(
                dense_f
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_f64_slice()
                    .is_some()
            );
            let lit_f = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: dims.clone() },
                    f.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            );
            let bits = |v: &Value| -> Vec<u64> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap().to_bits())
                    .collect()
            };
            let d = super::eval_gather(&[dense_f, idx.clone()], &params).unwrap();
            let l = super::eval_gather(&[lit_f, idx.clone()], &params).unwrap();
            assert_eq!(
                d.as_tensor().unwrap().shape.dims,
                l.as_tensor().unwrap().shape.dims
            );
            assert_eq!(bits(&d), bits(&l), "f64 gather mode={mode}");

            let n: Vec<i64> = (0..(rows * cols) as i64).map(|i| i - 50).collect();
            let dense_i = Value::Tensor(
                TensorValue::new_i64_values(Shape { dims: dims.clone() }, n.clone()).unwrap(),
            );
            let lit_i = Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape { dims: dims.clone() },
                    n.iter().copied().map(Literal::I64).collect(),
                )
                .unwrap(),
            );
            let di = super::eval_gather(&[dense_i, idx.clone()], &params).unwrap();
            let li = super::eval_gather(&[lit_i, idx.clone()], &params).unwrap();
            assert_eq!(
                extract_i64_vec(&di),
                extract_i64_vec(&li),
                "i64 gather mode={mode}"
            );
        }
    }

    /// Dense f32 gather (embedding-lookup case) must be BIT-FOR-BIT identical to
    /// the generic per-`Literal` copy, across clip + fill_or_drop (the OOB index
    /// 999 exercises the `f32::NAN` fill, which must match gather_fill_literal(F32)).
    #[test]
    fn dense_f32_gather_contiguous_matches_literal_path() {
        let (rows, cols) = (32usize, 40usize);
        let dims = vec![rows as u32, cols as u32];
        let idx = Value::vector_i64(&[0, 5, 31, 999, 1, 7, 31, 0, 12, 999]).unwrap();
        for mode in ["clip", "fill_or_drop"] {
            let params = params(&[("slice_sizes", "1,40"), ("index_mode", mode)]);
            let f: Vec<f32> = (0..rows * cols).map(|i| (i as f32 - 100.0) * 0.5).collect();
            let dense = Value::Tensor(
                TensorValue::new_f32_values(Shape { dims: dims.clone() }, f.clone()).unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape { dims: dims.clone() },
                    f.iter().copied().map(Literal::from_f32).collect(),
                )
                .unwrap(),
            );
            assert!(dense.as_tensor().unwrap().elements.as_f32_slice().is_some());
            assert!(boxed.as_tensor().unwrap().elements.as_f32_slice().is_none());
            let bits = |v: &Value| -> Vec<u32> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::F32Bits(b) => *b,
                        o => panic!("expected f32, got {o:?}"),
                    })
                    .collect()
            };
            let d = super::eval_gather(&[dense, idx.clone()], &params).unwrap();
            let l = super::eval_gather(&[boxed, idx.clone()], &params).unwrap();
            assert_eq!(
                d.as_tensor().unwrap().dtype,
                DType::F32,
                "f32 gather dtype mode={mode}"
            );
            assert_eq!(
                d.as_tensor().unwrap().shape.dims,
                l.as_tensor().unwrap().shape.dims
            );
            assert_eq!(bits(&d), bits(&l), "f32 gather mode={mode}");
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_embedding_gather_dense_vs_generic() {
        use std::time::Instant;
        let (vocab, dim) = (50000usize, 256usize); // [50000,256] f32 table
        let batch = 8192usize;
        let table: Vec<f32> = (0..vocab * dim)
            .map(|i| ((i % 1009) as f32) * 0.001 - 0.5)
            .collect();
        let dims = vec![vocab as u32, dim as u32];
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, table.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: dims.clone() },
                table.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let idx: Vec<i64> = (0..batch as i64)
            .map(|i| (i * 7919) % vocab as i64)
            .collect();
        let idx_v = Value::vector_i64(&idx).unwrap();
        let p = params(&[("slice_sizes", "1,256"), ("index_mode", "clip")]);
        let time = |operand: &Value| {
            let _ = super::eval_gather(&[operand.clone(), idx_v.clone()], &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = super::eval_gather(&[operand.clone(), idx_v.clone()], &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 embedding gather [{vocab},{dim}] x {batch} rows: generic(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    /// Dense BF16/F16 gather (half-precision embedding lookup) must be BIT-FOR-BIT
    /// identical to the generic per-`Literal` copy, across clip + fill_or_drop (OOB
    /// index 999 exercises the half-float NaN fill = gather_fill_literal bits).
    #[test]
    fn dense_half_float_gather_contiguous_matches_literal_path() {
        let (rows, cols) = (32usize, 40usize);
        let dims = vec![rows as u32, cols as u32];
        let idx = Value::vector_i64(&[0, 5, 31, 999, 1, 7, 31, 0, 12, 999]).unwrap();
        // raw u16 bit patterns spanning specials (0,-0,~1,NaN,inf) and varied values.
        let raw: Vec<u16> = (0..rows * cols)
            .map(|i| match i % 7 {
                0 => 0x0000,
                1 => 0x8000,
                2 => 0x3f80,
                3 => 0x7fc1,
                4 => 0x7f80,
                _ => (i as u16).wrapping_mul(37).wrapping_add(11),
            })
            .collect();
        for dtype in [DType::BF16, DType::F16] {
            let mk_lit = |b: u16| {
                if dtype == DType::BF16 {
                    Literal::BF16Bits(b)
                } else {
                    Literal::F16Bits(b)
                }
            };
            for mode in ["clip", "fill_or_drop"] {
                let params = params(&[("slice_sizes", "1,40"), ("index_mode", mode)]);
                let dense = Value::Tensor(
                    TensorValue::new_half_float_values(
                        dtype,
                        Shape { dims: dims.clone() },
                        raw.clone(),
                    )
                    .unwrap(),
                );
                let boxed = Value::Tensor(
                    TensorValue::new(
                        dtype,
                        Shape { dims: dims.clone() },
                        raw.iter().copied().map(mk_lit).collect(),
                    )
                    .unwrap(),
                );
                assert!(
                    dense
                        .as_tensor()
                        .unwrap()
                        .elements
                        .as_half_float_slice()
                        .is_some()
                );
                assert!(
                    boxed
                        .as_tensor()
                        .unwrap()
                        .elements
                        .as_half_float_slice()
                        .is_none()
                );
                let bits = |v: &Value| -> Vec<u16> {
                    v.as_tensor()
                        .unwrap()
                        .elements
                        .iter()
                        .map(|l| match l {
                            Literal::BF16Bits(b) | Literal::F16Bits(b) => *b,
                            o => panic!("expected half-float, got {o:?}"),
                        })
                        .collect()
                };
                let d = super::eval_gather(&[dense, idx.clone()], &params).unwrap();
                let l = super::eval_gather(&[boxed, idx.clone()], &params).unwrap();
                assert_eq!(
                    d.as_tensor().unwrap().dtype,
                    dtype,
                    "{dtype:?} gather dtype mode={mode}"
                );
                assert_eq!(bits(&d), bits(&l), "{dtype:?} gather mode={mode}");
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_bf16_embedding_gather_dense_vs_generic() {
        use std::time::Instant;
        let (vocab, dim) = (50000usize, 256usize);
        let batch = 8192usize;
        let raw: Vec<u16> = (0..vocab * dim)
            .map(|i| (i as u16).wrapping_mul(40503).wrapping_add(7))
            .collect();
        let dims = vec![vocab as u32, dim as u32];
        let dense = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::BF16,
                Shape { dims: dims.clone() },
                raw.clone(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::BF16,
                Shape { dims: dims.clone() },
                raw.iter().copied().map(Literal::BF16Bits).collect(),
            )
            .unwrap(),
        );
        let idx: Vec<i64> = (0..batch as i64)
            .map(|i| (i * 7919) % vocab as i64)
            .collect();
        let idx_v = Value::vector_i64(&idx).unwrap();
        let p = params(&[("slice_sizes", "1,256"), ("index_mode", "clip")]);
        let time = |operand: &Value| {
            let _ = super::eval_gather(&[operand.clone(), idx_v.clone()], &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = super::eval_gather(&[operand.clone(), idx_v.clone()], &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH bf16 embedding gather [{vocab},{dim}] x {batch} rows: generic(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn top_k_k_equals_len() {
        let x = v_f64(&[2.0, 1.0, 3.0]);
        let p = params(&[("k", "3")]);
        let outputs = super::eval_top_k(&[x], &p).unwrap();
        let values = extract_f64_vec(&outputs[0]);
        assert_eq!(values, vec![3.0, 2.0, 1.0]);
        let indices = extract_i64_vec(&outputs[1]);
        assert_eq!(indices, vec![2, 0, 1]);
    }

    #[test]
    fn top_k_k_one() {
        let x = v_f64(&[5.0, 10.0, 3.0, 8.0]);
        let p = params(&[("k", "1")]);
        let outputs = super::eval_top_k(&[x], &p).unwrap();
        let values = extract_f64_vec(&outputs[0]);
        assert_eq!(values, vec![10.0]);
        let indices = extract_i64_vec(&outputs[1]);
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn top_k_scalar_rejects() {
        let x = Value::Scalar(fj_core::Literal::from_f64(42.0));
        let p = params(&[("k", "1")]);
        let err = super::eval_top_k(&[x], &p).expect_err("scalar should be rejected");
        assert!(
            err.to_string().contains(">= 1 dimension"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn top_k_f32_radix_matches_reference() {
        // F32 top_k now uses the Literal-backed complement-key radix (axis >= 256).
        // Validate against an independent descending-total_cmp / ascending-index
        // reference over data incl NaN / +-inf / +-0 / NaN-payload / dups.
        let n = 1000usize;
        let data: Vec<f32> = (0..n)
            .map(|i| match i % 11 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => -0.0,
                4 => 0.0,
                5 => f32::from_bits(0x7fc0_0001),
                6 => 2.5,
                _ => ((i as f32) * 1.000_173).sin() * 1e3 - (i as f32),
            })
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: vec![n as u32],
                },
                data.iter()
                    .map(|&v| Literal::F32Bits(v.to_bits()))
                    .collect(),
            )
            .unwrap(),
        );

        for k in [1usize, 32, 257] {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| {
                (data[b] as f64)
                    .total_cmp(&(data[a] as f64))
                    .then(a.cmp(&b))
            });
            let expected_bits: Vec<u32> = idx[..k].iter().map(|&i| data[i].to_bits()).collect();
            let expected_idx: Vec<i64> = idx[..k].iter().map(|&i| i as i64).collect();

            let p = params(&[("k", &k.to_string())]);
            let outputs = super::eval_top_k(std::slice::from_ref(&tensor), &p).unwrap();
            let got_bits: Vec<u32> = outputs[0]
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F32Bits(b) => *b,
                    other => panic!("expected F32Bits, got {other:?}"),
                })
                .collect();
            assert_eq!(got_bits, expected_bits, "top_k f32 values, k={k}");
            assert_eq!(
                extract_i64_vec(&outputs[1]),
                expected_idx,
                "top_k f32 indices, k={k}"
            );
        }
    }
}
