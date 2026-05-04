#![no_main]

//! Fuzz target for primitive evaluation WITH params.
//!
//! Unlike primitive_eval_fuzzer which uses empty params, this target generates
//! random parameter maps for primitives that use them (reduce axes, reshape
//! dimensions, slice bounds, etc.).

mod common;

use common::{primitive_arity, sample_primitive, sample_value, ByteCursor};
use fj_core::Primitive;
use fj_lax::eval_primitive;
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeMap;

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }

    let mut cursor = ByteCursor::new(data);
    let primitive = sample_primitive(&mut cursor);
    let arity = primitive_arity(primitive);

    let mut inputs = Vec::with_capacity(arity);
    for _ in 0..arity {
        inputs.push(sample_value(&mut cursor));
    }

    let params = sample_primitive_params(&mut cursor, primitive);

    match eval_primitive(primitive, &inputs, &params) {
        Ok(value) => match &value {
            fj_core::Value::Scalar(lit) => {
                let _ = lit.as_f64();
                let _ = lit.as_i64();
            }
            fj_core::Value::Tensor(t) => {
                assert_eq!(
                    t.shape.element_count().unwrap_or(0) as usize,
                    t.elements.len(),
                    "shape/element count mismatch"
                );
            }
        },
        Err(_) => {}
    }
});

fn sample_primitive_params(
    cursor: &mut ByteCursor<'_>,
    primitive: Primitive,
) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();

    match primitive {
        Primitive::ReduceSum
        | Primitive::ReduceMax
        | Primitive::ReduceMin
        | Primitive::ReduceProd
        | Primitive::ReduceAnd
        | Primitive::ReduceOr
        | Primitive::ReduceXor => {
            if cursor.take_bool() {
                let axes = sample_axes_string(cursor);
                params.insert("axes".to_owned(), axes);
            }
            if cursor.take_bool() {
                params.insert("keepdims".to_owned(), cursor.take_bool().to_string());
            }
        }

        Primitive::Reshape => {
            let dims = sample_shape_string(cursor);
            params.insert("new_sizes".to_owned(), dims);
        }

        Primitive::Slice => {
            params.insert("start_indices".to_owned(), sample_indices_string(cursor));
            params.insert("limit_indices".to_owned(), sample_indices_string(cursor));
            if cursor.take_bool() {
                params.insert("strides".to_owned(), sample_strides_string(cursor));
            }
        }

        Primitive::DynamicSlice => {
            params.insert("slice_sizes".to_owned(), sample_sizes_string(cursor));
        }

        Primitive::Gather => {
            params.insert("slice_sizes".to_owned(), sample_sizes_string(cursor));
        }

        Primitive::Transpose => {
            params.insert("permutation".to_owned(), sample_permutation_string(cursor));
        }

        Primitive::BroadcastInDim => {
            params.insert("shape".to_owned(), sample_shape_string(cursor));
            params.insert(
                "broadcast_dimensions".to_owned(),
                sample_indices_string(cursor),
            );
        }

        Primitive::Pad => {
            params.insert("padding_config".to_owned(), sample_pad_config(cursor));
        }

        Primitive::Concatenate => {
            let axis = sample_axis_i64(cursor);
            params.insert("dimension".to_owned(), axis.to_string());
        }

        Primitive::Rev => {
            params.insert("dimensions".to_owned(), sample_indices_string(cursor));
        }

        Primitive::Squeeze => {
            params.insert("dimensions".to_owned(), sample_indices_string(cursor));
        }

        Primitive::ExpandDims => {
            params.insert("dimensions".to_owned(), sample_indices_string(cursor));
        }

        Primitive::Split => {
            let axis = sample_axis_i64(cursor);
            params.insert("axis".to_owned(), axis.to_string());
            params.insert("num_sections".to_owned(), sample_positive_int(cursor));
        }

        Primitive::IntegerPow => {
            let exponent = i32::from(cursor.take_u8() % 21) - 10;
            params.insert("exponent".to_owned(), exponent.to_string());
        }

        Primitive::ReducePrecision => {
            let exp = 1 + cursor.take_usize(11);
            let mant = cursor.take_usize(52);
            params.insert("exponent_bits".to_owned(), exp.to_string());
            params.insert("mantissa_bits".to_owned(), mant.to_string());
        }

        Primitive::Iota | Primitive::BroadcastedIota => {
            params.insert("shape".to_owned(), sample_shape_string(cursor));
            params.insert("dimension".to_owned(), sample_axis_i64(cursor).to_string());
            params.insert("dtype".to_owned(), sample_dtype_string(cursor));
        }

        Primitive::BitcastConvertType => {
            params.insert("new_dtype".to_owned(), sample_dtype_string(cursor));
        }

        Primitive::OneHot => {
            params.insert("depth".to_owned(), sample_positive_int(cursor));
            params.insert("axis".to_owned(), sample_axis_i64(cursor).to_string());
        }

        Primitive::Cumsum | Primitive::Cumprod => {
            params.insert("axis".to_owned(), sample_axis_i64(cursor).to_string());
            if cursor.take_bool() {
                params.insert("reverse".to_owned(), cursor.take_bool().to_string());
            }
        }

        Primitive::Sort | Primitive::Argsort => {
            params.insert("dimension".to_owned(), sample_axis_i64(cursor).to_string());
            if cursor.take_bool() {
                params.insert("is_stable".to_owned(), cursor.take_bool().to_string());
            }
        }

        Primitive::Fft | Primitive::Ifft | Primitive::Rfft | Primitive::Irfft => {
            if cursor.take_bool() {
                let len = 1 + cursor.take_usize(1023);
                params.insert("fft_length".to_owned(), len.to_string());
            }
        }

        Primitive::Cholesky => {
            if cursor.take_bool() {
                params.insert("lower".to_owned(), cursor.take_bool().to_string());
            }
        }

        Primitive::TriangularSolve => {
            if cursor.take_bool() {
                params.insert("left_side".to_owned(), cursor.take_bool().to_string());
            }
            if cursor.take_bool() {
                params.insert("lower".to_owned(), cursor.take_bool().to_string());
            }
            if cursor.take_bool() {
                params.insert("transpose_a".to_owned(), cursor.take_bool().to_string());
            }
        }

        Primitive::Qr => {
            if cursor.take_bool() {
                params.insert("full_matrices".to_owned(), cursor.take_bool().to_string());
            }
        }

        Primitive::Svd => {
            if cursor.take_bool() {
                params.insert("full_matrices".to_owned(), cursor.take_bool().to_string());
            }
            if cursor.take_bool() {
                params.insert("compute_uv".to_owned(), cursor.take_bool().to_string());
            }
        }

        _ => {}
    }

    if cursor.take_bool() {
        let garbage_key = cursor.take_string(8);
        let garbage_val = cursor.take_string(12);
        if !garbage_key.is_empty() {
            params.insert(garbage_key, garbage_val);
        }
    }

    params
}

fn sample_axes_string(cursor: &mut ByteCursor<'_>) -> String {
    let count = 1 + cursor.take_usize(3);
    let mut axes = Vec::with_capacity(count);
    for _ in 0..count {
        axes.push(sample_axis_i64(cursor).to_string());
    }
    axes.join(",")
}

fn sample_shape_string(cursor: &mut ByteCursor<'_>) -> String {
    let rank = cursor.take_usize(4);
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        let dim = cursor.take_usize(16);
        dims.push(dim.to_string());
    }
    dims.join(",")
}

fn sample_indices_string(cursor: &mut ByteCursor<'_>) -> String {
    let count = cursor.take_usize(4);
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        let idx = cursor.take_usize(16);
        indices.push(idx.to_string());
    }
    indices.join(",")
}

fn sample_sizes_string(cursor: &mut ByteCursor<'_>) -> String {
    let count = 1 + cursor.take_usize(3);
    let mut sizes = Vec::with_capacity(count);
    for _ in 0..count {
        let size = 1 + cursor.take_usize(8);
        sizes.push(size.to_string());
    }
    sizes.join(",")
}

fn sample_strides_string(cursor: &mut ByteCursor<'_>) -> String {
    let count = cursor.take_usize(4);
    let mut strides = Vec::with_capacity(count);
    for _ in 0..count {
        let stride = 1 + cursor.take_usize(3);
        strides.push(stride.to_string());
    }
    strides.join(",")
}

fn sample_permutation_string(cursor: &mut ByteCursor<'_>) -> String {
    let rank = cursor.take_usize(4);
    let mut perm: Vec<usize> = (0..rank).collect();
    for i in (1..rank).rev() {
        let j = cursor.take_usize(i);
        perm.swap(i, j);
    }
    perm.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn sample_pad_config(cursor: &mut ByteCursor<'_>) -> String {
    let rank = cursor.take_usize(3);
    let mut parts = Vec::with_capacity(rank);
    for _ in 0..rank {
        let lo = cursor.take_usize(4);
        let hi = cursor.take_usize(4);
        let interior = cursor.take_usize(2);
        parts.push(format!("({lo},{hi},{interior})"));
    }
    parts.join(",")
}

fn sample_axis_i64(cursor: &mut ByteCursor<'_>) -> i64 {
    let raw = cursor.take_u8();
    if raw < 128 {
        i64::from(raw % 8)
    } else {
        -1 - i64::from((raw - 128) % 4)
    }
}

fn sample_positive_int(cursor: &mut ByteCursor<'_>) -> String {
    let val = 1 + cursor.take_usize(15);
    val.to_string()
}

fn sample_dtype_string(cursor: &mut ByteCursor<'_>) -> String {
    const DTYPES: &[&str] = &[
        "bool", "i32", "i64", "u32", "u64", "bf16", "f16", "f32", "f64", "c64", "c128",
    ];
    let idx = cursor.take_usize(DTYPES.len().saturating_sub(1));
    DTYPES[idx].to_owned()
}
