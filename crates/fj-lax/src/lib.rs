#![forbid(unsafe_code)]
#![feature(portable_simd)]

pub mod arithmetic;
pub mod array_creation;
mod comparison;
pub mod cz0g0_f32accum_evidence;
pub mod cz0g0_fma_evidence;
pub mod dense;
pub mod einsum;
mod fft;
pub mod linalg;
pub mod nn;
mod reduction;
pub mod simd_exp;
pub mod tensor_contraction;
mod tensor_ops;
pub mod threefry;
pub mod tree_util;
mod type_promotion;

/// Public access to `promote_dtype` for conformance tests.
pub use type_promotion::promote_dtype as promote_dtype_public;

/// Public access to ∂/∂a of igamma for the AD layer's igamma/igammac rules.
pub use arithmetic::eval_igamma_grad_a;

use fj_core::{Literal, Primitive, Shape, TensorValue, Value, ValueError};
use std::borrow::Cow;
use std::collections::BTreeMap;

/// Build a `TensorValue` with guaranteed BOXED (`Literal`-backed) storage, for
/// the dense-fast-path-vs-generic guard tests. `TensorValue::new` densifies
/// homogeneous literal vectors (see `dense_buffer_for_declared_dtype`,
/// cbea72b3), which would silently turn these *reference* inputs dense and
/// defeat the dense-vs-boxed comparison (and trip the `as_*_slice().is_none()`
/// preconditions). Routing those reference builders here keeps them boxed, so
/// each guard test again exercises the boxed generic path against the dense
/// fast path. Bit-for-bit identical to `TensorValue::new` in value.
#[cfg(test)]
pub(crate) fn new_boxed(
    dtype: fj_core::DType,
    shape: Shape,
    elements: Vec<Literal>,
) -> Result<TensorValue, ValueError> {
    TensorValue::new_with_literal_buffer(dtype, shape, fj_core::LiteralBuffer::new(elements))
}

use arithmetic::{
    ensure_float_or_complex_operands, eval_abs, eval_acosh, eval_asinh, eval_atan, eval_atanh,
    eval_bessel_i0e, eval_bessel_i1e, eval_betainc, eval_binary_elementwise, eval_cbrt, eval_clamp,
    eval_complex, eval_conj, eval_cos, eval_cosh, eval_digamma, eval_dot, eval_dot_general,
    eval_erf, eval_erf_inv, eval_erfc, eval_exp, eval_expm1, eval_float_complex_unary, eval_fma,
    eval_igamma,
    eval_igammac, eval_imag, eval_integer_pow, eval_is_finite, eval_is_inf, eval_is_nan,
    eval_lgamma, eval_log, eval_log1p, eval_logistic, eval_neg, eval_nextafter, eval_polygamma,
    eval_real,
    eval_round,
    eval_rsqrt,
    eval_select, eval_select_n, eval_signbit, eval_sin, eval_sinh, eval_tan, eval_tanh,
    eval_unary_elementwise_parallel, eval_unary_int_or_float, eval_zeta,
};

use comparison::eval_comparison;
use fft::{eval_fft, eval_ifft, eval_irfft, eval_rfft};
use linalg::{
    eval_cholesky, eval_det, eval_eig, eval_eigh, eval_lu, eval_qr, eval_slogdet, eval_solve,
    eval_svd, eval_triangular_solve,
};
use reduction::{eval_cumulative, eval_reduce_axes, eval_reduce_bitwise_axes};
use tensor_ops::{
    eval_argmax, eval_argmin, eval_argsort, eval_bitcast_convert_type, eval_broadcast_in_dim,
    eval_broadcasted_iota, eval_concatenate, eval_conv, eval_convert_element_type, eval_copy,
    eval_dynamic_slice, eval_dynamic_update_slice, eval_expand_dims, eval_gather, eval_iota,
    eval_one_hot, eval_pad, eval_reduce_precision, eval_reshape, eval_rev, eval_scatter,
    eval_slice, eval_sort, eval_sort_multi, eval_split, eval_squeeze, eval_tile, eval_top_k,
    eval_transpose,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    ArityMismatch {
        primitive: Primitive,
        expected: usize,
        actual: usize,
    },
    TypeMismatch {
        primitive: Primitive,
        detail: &'static str,
    },
    ShapeMismatch {
        primitive: Primitive,
        left: Shape,
        right: Shape,
    },
    Unsupported {
        primitive: Primitive,
        detail: String,
    },
    InvalidTensor(ValueError),
    MaxIterationsExceeded {
        primitive: Primitive,
        max_iterations: usize,
    },
    ShapeChanged {
        primitive: Primitive,
        detail: String,
    },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArityMismatch {
                primitive,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "arity mismatch for {}: expected {}, got {}",
                    primitive.as_str(),
                    expected,
                    actual
                )
            }
            Self::TypeMismatch { primitive, detail } => {
                write!(f, "type mismatch for {}: {}", primitive.as_str(), detail)
            }
            Self::ShapeMismatch {
                primitive,
                left,
                right,
            } => {
                write!(
                    f,
                    "shape mismatch for {}: left={:?} right={:?}",
                    primitive.as_str(),
                    left.dims,
                    right.dims
                )
            }
            Self::Unsupported { primitive, detail } => {
                write!(f, "unsupported {} behavior: {}", primitive.as_str(), detail)
            }
            Self::InvalidTensor(err) => write!(f, "invalid tensor: {err}"),
            Self::MaxIterationsExceeded {
                primitive,
                max_iterations,
            } => {
                write!(
                    f,
                    "{} exceeded max iterations ({})",
                    primitive.as_str(),
                    max_iterations
                )
            }
            Self::ShapeChanged { primitive, detail } => {
                write!(
                    f,
                    "{} body changed carry shape: {}",
                    primitive.as_str(),
                    detail
                )
            }
        }
    }
}

impl std::error::Error for EvalError {}

impl From<ValueError> for EvalError {
    fn from(value: ValueError) -> Self {
        Self::InvalidTensor(value)
    }
}

#[inline]
fn jax_max_f64(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else {
        left.max(right)
    }
}

#[inline]
fn jax_min_f64(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else {
        left.min(right)
    }
}

#[inline]
/// jnp/numpy `int32` ops wrap mod 2^32 (two's complement): e.g.
/// `int32(2147483647) + int32(1) == -2147483648`. fj-lax has no `Literal::I32` — i32
/// values are stored as `Literal::I64` with the dtype tracked only on the `TensorValue`,
/// and the shared i64 wrapping ops accumulate/produce in i64 WITHOUT narrowing (only the
/// `U32` literal arm and `integer_pow` narrow). So an arithmetic result on an I32 tensor
/// (elementwise add/sub/mul, matmul, reduce, …) could land outside i32 range and diverge
/// from JAX. This single chokepoint narrows every I32 tensor result back into i32 range.
/// In-range results (the overwhelmingly common case — indices, shapes, small ints) are
/// returned untouched with NO realloc (only a cheap scan), so it cannot change any
/// currently-correct value; only genuinely-overflowing results are corrected.
///
/// I32 scalars now use `Literal::I32`; this remains the tensor-output chokepoint for
/// dense I32 paths that share the I64 storage lane.
fn narrow_i32_tensor_result(value: Value) -> Value {
    let Value::Tensor(t) = &value else {
        return value;
    };
    if t.dtype != fj_core::DType::I32 {
        return value;
    }
    // Dense i64-backed I32 (the common dense path): scan AND narrow directly on the
    // typed i64 slice, with no per-element 24-byte `Literal` reconstruction. This
    // chokepoint runs after EVERY i32 primitive eval, so the detection scan alone
    // (the no-overflow common case) otherwise materialized a Literal per element
    // every time. Bit-identical: the boxed condition is `Literal::I64(v) if v !=
    // i64::from(v as i32)` and dense I32 elements are all I64(v), so the slice scan
    // and `i64::from(v as i32)` narrowing produce the same I32 dense storage.
    if let Some(slice) = t.elements.as_i64_slice() {
        if !slice.iter().any(|&v| v != i64::from(v as i32)) {
            return value;
        }
        let shape = t.shape.clone();
        let narrowed: Vec<i64> = slice.iter().map(|&v| i64::from(v as i32)).collect();
        return TensorValue::new_i32_values(shape, narrowed)
            .map(Value::Tensor)
            .unwrap_or(value);
    }
    // Boxed/other backing fallback: scan + narrow over the materialized Literals.
    if !t
        .elements
        .iter()
        .any(|l| matches!(l, Literal::I64(v) if *v != i64::from(*v as i32)))
    {
        return value;
    }
    let shape = t.shape.clone();
    let narrowed: Vec<Literal> = t
        .elements
        .iter()
        .map(|l| match l {
            Literal::I64(v) => Literal::I64(i64::from(*v as i32)),
            other => *other,
        })
        .collect();
    TensorValue::new(fj_core::DType::I32, shape, narrowed)
        .map(Value::Tensor)
        .unwrap_or(value)
}

pub fn eval_primitive(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    eval_primitive_inner(primitive, inputs, params).map(narrow_i32_tensor_result)
}

fn eval_primitive_inner(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    match primitive {
        // Binary arithmetic
        Primitive::Add => {
            eval_binary_elementwise(primitive, inputs, |a, b| a.wrapping_add(b), |a, b| a + b)
        }
        Primitive::Sub => {
            eval_binary_elementwise(primitive, inputs, |a, b| a.wrapping_sub(b), |a, b| a - b)
        }
        Primitive::Mul => {
            eval_binary_elementwise(primitive, inputs, |a, b| a.wrapping_mul(b), |a, b| a * b)
        }
        Primitive::Max => eval_binary_elementwise(primitive, inputs, |a, b| a.max(b), jax_max_f64),
        Primitive::Min => eval_binary_elementwise(primitive, inputs, |a, b| a.min(b), jax_min_f64),
        Primitive::Fma => eval_fma(primitive, inputs),
        Primitive::Pow => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| (a as f64).powf(b as f64) as i64,
            f64::powf,
        ),
        Primitive::Hypot => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| f64::hypot(a as f64, b as f64) as i64,
            f64::hypot,
        ),
        Primitive::LogAddExp => {
            // log(exp(x) + exp(y)) = max(x,y) + log1p(exp(min(x,y) - max(x,y)))
            eval_binary_elementwise(
                primitive,
                inputs,
                |a, b| {
                    let (a, b) = (a as f64, b as f64);
                    (a.max(b) + (-(a - b).abs()).exp().ln_1p()) as i64
                },
                |a, b| a.max(b) + (-(a - b).abs()).exp().ln_1p(),
            )
        }
        Primitive::LogAddExp2 => {
            // log2(2^x + 2^y) = max(x,y) + log2(1 + 2^(min - max))
            //                = max(x,y) + log2(1 + 2^(-|x-y|))
            eval_binary_elementwise(
                primitive,
                inputs,
                |a, b| {
                    let (a, b) = (a as f64, b as f64);
                    let diff = -(a - b).abs();
                    (a.max(b) + (1.0 + 2f64.powf(diff)).log2()) as i64
                },
                |a, b| {
                    let diff = -(a - b).abs();
                    a.max(b) + (1.0 + 2f64.powf(diff)).log2()
                },
            )
        }
        // Unary arithmetic
        Primitive::Neg => eval_neg(primitive, inputs),
        Primitive::Abs => eval_abs(primitive, inputs),
        Primitive::Exp => eval_exp(primitive, inputs),
        Primitive::Log => eval_log(primitive, inputs),
        Primitive::Log2 => eval_float_complex_unary(primitive, inputs, f64::log2),
        Primitive::Exp2 => eval_float_complex_unary(primitive, inputs, f64::exp2),
        // Sinc carries a sin + div per element (compute-bound), so thread it like
        // the other transcendentals instead of the serial path.
        Primitive::Sinc => eval_unary_elementwise_parallel(primitive, inputs, |x| {
            if x == 0.0 {
                1.0
            } else {
                let pi_x = std::f64::consts::PI * x;
                pi_x.sin() / pi_x
            }
        }),
        // sqrt / rsqrt are div-/sqrt-unit-bound (NOT pipelined like add/mul), so they are
        // compute-bound and thread like the other transcendentals; the parallel path falls
        // back to the identical serial dense map below the threshold (bit-for-bit identical).
        Primitive::Sqrt => eval_float_complex_unary(primitive, inputs, f64::sqrt),
        Primitive::Rsqrt => eval_rsqrt(primitive, inputs),
        Primitive::Floor => eval_unary_elementwise_parallel(primitive, inputs, f64::floor),
        Primitive::Ceil => eval_unary_elementwise_parallel(primitive, inputs, f64::ceil),
        Primitive::Round => eval_round(primitive, inputs, params),
        Primitive::Trunc => eval_unary_elementwise_parallel(primitive, inputs, f64::trunc),
        // Trigonometric
        Primitive::Sin => eval_sin(primitive, inputs),
        Primitive::Cos => eval_cos(primitive, inputs),
        Primitive::Tan => eval_tan(primitive, inputs),
        Primitive::Asin => eval_float_complex_unary(primitive, inputs, f64::asin),
        Primitive::Acos => eval_float_complex_unary(primitive, inputs, f64::acos),
        Primitive::Atan => eval_atan(primitive, inputs),
        Primitive::Deg2Rad => eval_unary_elementwise_parallel(primitive, inputs, f64::to_radians),
        Primitive::Rad2Deg => eval_unary_elementwise_parallel(primitive, inputs, f64::to_degrees),
        // Hyperbolic
        Primitive::Sinh => eval_sinh(primitive, inputs),
        Primitive::Cosh => eval_cosh(primitive, inputs),
        Primitive::Tanh => eval_tanh(primitive, inputs),
        Primitive::Asinh => eval_asinh(primitive, inputs),
        Primitive::Acosh => eval_acosh(primitive, inputs),
        Primitive::Atanh => eval_atanh(primitive, inputs),
        // Additional math
        Primitive::Expm1 => eval_expm1(primitive, inputs),
        Primitive::Log1p => eval_log1p(primitive, inputs),
        Primitive::Sign => eval_unary_int_or_float(
            primitive,
            inputs,
            |x| x.signum(),
            |x| u32::from(x != 0),
            |x| u64::from(x != 0),
            |x| {
                if x.is_nan() {
                    f64::NAN
                } else if x == 0.0 {
                    x
                } else {
                    x.signum()
                }
            },
        ),
        Primitive::Square => eval_unary_int_or_float(
            primitive,
            inputs,
            |x| x * x,
            |x| x.wrapping_mul(x),
            |x| x.wrapping_mul(x),
            |x| x * x,
        ),
        Primitive::Reciprocal => eval_unary_elementwise_parallel(primitive, inputs, |x| 1.0 / x),
        Primitive::Logistic => eval_logistic(primitive, inputs),
        Primitive::Erf => eval_erf(primitive, inputs),
        Primitive::Erfc => eval_erfc(primitive, inputs),
        Primitive::Lgamma => eval_lgamma(primitive, inputs),
        Primitive::Digamma => eval_digamma(primitive, inputs),
        Primitive::Polygamma => eval_polygamma(primitive, inputs),
        Primitive::ErfInv => eval_erf_inv(primitive, inputs),
        Primitive::Igamma => eval_igamma(primitive, inputs),
        Primitive::Igammac => eval_igammac(primitive, inputs),
        Primitive::Betainc => eval_betainc(primitive, inputs),
        Primitive::Zeta => eval_zeta(primitive, inputs),
        Primitive::BesselI0e => eval_bessel_i0e(primitive, inputs),
        Primitive::BesselI1e => eval_bessel_i1e(primitive, inputs),
        Primitive::Conj => eval_conj(primitive, inputs),
        Primitive::Real => eval_real(primitive, inputs),
        Primitive::Imag => eval_imag(primitive, inputs),
        // Binary math
        Primitive::Div => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| a.checked_div(b).unwrap_or(0),
            |a, b| a / b,
        ),
        Primitive::Rem => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| a.checked_rem(b).unwrap_or(0),
            |a, b| a % b,
        ),
        Primitive::Gcd => eval_binary_elementwise(
            primitive,
            inputs,
            |mut a, mut b| {
                while b != 0 {
                    let t = b;
                    b = a % b;
                    a = t;
                }
                a.abs()
            },
            |a, b| {
                let (mut a, mut b) = (a.abs() as i64, b.abs() as i64);
                while b != 0 {
                    let t = b;
                    b = a % b;
                    a = t;
                }
                a as f64
            },
        ),
        Primitive::Lcm => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| {
                if a == 0 || b == 0 {
                    return 0;
                }
                let (mut ga, mut gb) = (a.abs(), b.abs());
                let prod = ga * gb;
                while gb != 0 {
                    let t = gb;
                    gb = ga % gb;
                    ga = t;
                }
                prod / ga
            },
            |a, b| {
                if a == 0.0 || b == 0.0 {
                    return 0.0;
                }
                let (mut ga, mut gb) = (a.abs() as i64, b.abs() as i64);
                let prod = ga * gb;
                while gb != 0 {
                    let t = gb;
                    gb = ga % gb;
                    ga = t;
                }
                (prod / ga) as f64
            },
        ),
        Primitive::Atan2 => {
            // JAX atan2_p = standard_naryop([_float | _complex, _float | _complex]):
            // integer operands are rejected at the lax level (the generic path would
            // otherwise compute atan2 in f64 and truncate back to i64).
            ensure_float_or_complex_operands(primitive, inputs)?;
            eval_binary_elementwise(
                primitive,
                inputs,
                |a, b| (a as f64).atan2(b as f64) as i64,
                f64::atan2,
            )
        }
        Primitive::Complex => eval_complex(primitive, inputs),
        // Selection
        Primitive::Select => eval_select(primitive, inputs),
        Primitive::SelectN => eval_select_n(primitive, inputs),
        // Dot product
        Primitive::Dot => eval_dot(inputs),
        Primitive::DotGeneral => eval_dot_general(inputs, params),
        // Comparison
        Primitive::Eq => eval_comparison(primitive, inputs, |a, b| a == b, |a, b| a == b),
        Primitive::Ne => eval_comparison(primitive, inputs, |a, b| a != b, |a, b| a != b),
        Primitive::Lt => eval_comparison(primitive, inputs, |a, b| a < b, |a, b| a < b),
        Primitive::Le => eval_comparison(primitive, inputs, |a, b| a <= b, |a, b| a <= b),
        Primitive::Gt => eval_comparison(primitive, inputs, |a, b| a > b, |a, b| a > b),
        Primitive::Ge => eval_comparison(primitive, inputs, |a, b| a >= b, |a, b| a >= b),
        // Reductions (axis-aware)
        Primitive::ReduceSum => eval_reduce_axes(
            primitive,
            inputs,
            params,
            0_i64,
            0.0,
            // Explicit two's-complement wrapping on integer overflow, matching
            // XLA's `Add` (and the elementwise `Add` / `integer_pow`). Plain `+`
            // would panic in debug builds and only happened to wrap in release.
            i64::wrapping_add,
            |a, b| a + b,
        ),
        Primitive::ReduceMax => eval_reduce_axes(
            primitive,
            inputs,
            params,
            i64::MIN,
            f64::NEG_INFINITY,
            i64::max,
            jax_max_f64,
        ),
        Primitive::ReduceMin => eval_reduce_axes(
            primitive,
            inputs,
            params,
            i64::MAX,
            f64::INFINITY,
            i64::min,
            jax_min_f64,
        ),
        Primitive::ReduceProd => eval_reduce_axes(
            primitive,
            inputs,
            params,
            1_i64,
            1.0,
            i64::wrapping_mul,
            |a, b| a * b,
        ),
        Primitive::ReduceAnd => eval_reduce_bitwise_axes(
            primitive,
            inputs,
            params,
            -1_i64,
            true,
            |a, b| a & b,
            |a, b| a && b,
        ),
        Primitive::ReduceOr => eval_reduce_bitwise_axes(
            primitive,
            inputs,
            params,
            0_i64,
            false,
            |a, b| a | b,
            |a, b| a || b,
        ),
        Primitive::ReduceXor => eval_reduce_bitwise_axes(
            primitive,
            inputs,
            params,
            0_i64,
            false,
            |a, b| a ^ b,
            |a, b| a ^ b,
        ),
        // Clamp: clamp(x, lo, hi) = min(max(x, lo), hi)
        Primitive::Clamp => eval_clamp(primitive, inputs),
        // Shape manipulation
        Primitive::Reshape => eval_reshape(inputs, params),
        Primitive::Transpose => eval_transpose(inputs, params),
        Primitive::BroadcastInDim => eval_broadcast_in_dim(inputs, params),
        Primitive::Concatenate => eval_concatenate(inputs, params),
        Primitive::Pad => eval_pad(inputs, params),
        Primitive::Rev => eval_rev(inputs, params),
        Primitive::Squeeze => eval_squeeze(inputs, params),
        Primitive::Split => eval_split(inputs, params),
        Primitive::ExpandDims => eval_expand_dims(inputs, params),
        Primitive::Tile => eval_tile(inputs, params),
        // Special math
        Primitive::Cbrt => eval_cbrt(primitive, inputs),
        Primitive::IsFinite => eval_is_finite(primitive, inputs),
        Primitive::IsNan => eval_is_nan(primitive, inputs),
        Primitive::IsInf => eval_is_inf(primitive, inputs),
        Primitive::Signbit => eval_signbit(primitive, inputs),
        Primitive::Heaviside => {
            // Match JAX/Numpy: NaN compares neither < 0 nor > 0, so it returns h0.
            eval_binary_elementwise(
                primitive,
                inputs,
                |x, h0| {
                    if x < 0 {
                        0
                    } else if x == 0 {
                        h0
                    } else {
                        1
                    }
                },
                |x, h0| {
                    if x < 0.0 {
                        0.0
                    } else if x > 0.0 {
                        1.0
                    } else {
                        h0
                    }
                },
            )
        }
        Primitive::CopySign => eval_binary_elementwise(
            primitive,
            inputs,
            |a, b| f64::copysign(a as f64, b as f64) as i64,
            f64::copysign,
        ),
        Primitive::Ldexp => {
            // ldexp(x, n) = x * 2^n
            eval_binary_elementwise(
                primitive,
                inputs,
                |a, b| (a as f64 * 2f64.powi(b as i32)) as i64,
                |a, b| a * 2f64.powi(b as i32),
            )
        }
        Primitive::XLogY => {
            // xlogy(x, y) = x * log(y), with 0 * log(anything) = 0
            eval_binary_elementwise(
                primitive,
                inputs,
                |a, b| {
                    let x = a as f64;
                    let y = b as f64;
                    if x == 0.0 { 0 } else { (x * y.ln()) as i64 }
                },
                |x, y| if x == 0.0 { 0.0 } else { x * y.ln() },
            )
        }
        Primitive::XLog1PY => {
            // xlog1py(x, y) = x * log1p(y), with 0 * log1p(anything) = 0
            eval_binary_elementwise(
                primitive,
                inputs,
                |a, b| {
                    let x = a as f64;
                    let y = b as f64;
                    if x == 0.0 { 0 } else { (x * y.ln_1p()) as i64 }
                },
                |x, y| if x == 0.0 { 0.0 } else { x * y.ln_1p() },
            )
        }
        Primitive::IntegerPow => eval_integer_pow(primitive, inputs, params),
        Primitive::Nextafter => eval_nextafter(primitive, inputs),
        Primitive::Slice => eval_slice(inputs, params),
        Primitive::DynamicSlice => eval_dynamic_slice(inputs, params),
        Primitive::Gather => eval_gather(inputs, params),
        Primitive::Scatter => eval_scatter(inputs, params),
        // Iota: generate index sequence
        Primitive::Iota => eval_iota(inputs, params),
        Primitive::BroadcastedIota => eval_broadcasted_iota(inputs, params),
        Primitive::Copy => eval_copy(inputs),
        Primitive::StopGradient => eval_copy(inputs),
        Primitive::ConvertElementType => eval_convert_element_type(inputs, params),
        Primitive::BitcastConvertType => eval_bitcast_convert_type(inputs, params),
        Primitive::ReducePrecision => eval_reduce_precision(inputs, params),
        Primitive::Cholesky => eval_cholesky(inputs, params),
        Primitive::TriangularSolve => eval_triangular_solve(inputs, params),
        Primitive::Qr => {
            // QR is multi-output; return first output (Q) for single-value API.
            // Use eval_primitive_multi for both Q and R.
            let mut outputs = eval_qr(inputs, params)?;
            Ok(outputs.remove(0))
        }
        Primitive::Lu => {
            // LU is multi-output; return first output (lu) for single-value API.
            // Use eval_primitive_multi for lu, pivots, and permutation.
            let mut outputs = eval_lu(inputs, params)?;
            Ok(outputs.remove(0))
        }
        Primitive::Svd => {
            // SVD is multi-output; return first output (U) for single-value API.
            let mut outputs = eval_svd(inputs, params)?;
            Ok(outputs.remove(0))
        }
        Primitive::Eigh => {
            // Eigh is multi-output; return first output (W) for single-value API.
            let mut outputs = eval_eigh(inputs, params)?;
            Ok(outputs.remove(0))
        }
        Primitive::Eig => {
            // Eig is multi-output; return first output (eigenvalues) for single-value API.
            let mut outputs = eval_eig(inputs, params)?;
            Ok(outputs.remove(0))
        }
        Primitive::Solve => eval_solve(inputs, params),
        Primitive::Det => eval_det(inputs, params),
        Primitive::Slogdet => {
            let outputs = eval_slogdet(inputs, params)?;
            Ok(outputs.into_iter().next().unwrap_or(Value::scalar_f64(0.0)))
        }
        Primitive::Fft => eval_fft(inputs, params),
        Primitive::Ifft => eval_ifft(inputs, params),
        Primitive::Rfft => eval_rfft(inputs, params),
        Primitive::Irfft => eval_irfft(inputs, params),
        // One-hot encoding
        Primitive::OneHot => eval_one_hot(inputs, params),
        // Dynamic update slice
        Primitive::DynamicUpdateSlice => eval_dynamic_update_slice(inputs, params),
        // Cumulative operations
        Primitive::Cumsum => eval_cumulative(
            primitive,
            inputs,
            params,
            0_i64,
            0.0,
            i64::wrapping_add,
            |a, b| a + b,
        ),
        Primitive::Cumprod => eval_cumulative(
            primitive,
            inputs,
            params,
            1_i64,
            1.0,
            i64::wrapping_mul,
            |a, b| a * b,
        ),
        Primitive::Cummax => eval_cumulative(
            primitive,
            inputs,
            params,
            i64::MIN,
            f64::NEG_INFINITY,
            i64::max,
            // NaN-propagating max (XLA `Max` / lax.cummax), NOT Rust `f64::max`
            // which drops NaN. Matches ReduceMax and elementwise Max so a NaN in
            // the prefix poisons the running maximum, as JAX does.
            jax_max_f64,
        ),
        Primitive::Cummin => eval_cumulative(
            primitive,
            inputs,
            params,
            i64::MAX,
            f64::INFINITY,
            i64::min,
            jax_min_f64,
        ),
        // Sorting
        Primitive::Sort => eval_sort(primitive, inputs, params),
        Primitive::Argsort => eval_argsort(primitive, inputs, params),
        Primitive::TopK => {
            // TopK is multi-output; return first output (values) for single-value API.
            let mut outputs = eval_top_k(inputs, params)?;
            Ok(outputs.remove(0))
        }
        // Index-of-extremum
        Primitive::Argmin => eval_argmin(primitive, inputs, params),
        Primitive::Argmax => eval_argmax(primitive, inputs, params),
        // Convolution
        Primitive::Conv => eval_conv(primitive, inputs, params),
        // Control flow
        Primitive::Cond => eval_cond(primitive, inputs),
        Primitive::Scan => eval_scan(primitive, inputs, params),
        Primitive::AssociativeScan => eval_associative_scan(primitive, inputs, params),
        Primitive::While => eval_while_loop(primitive, inputs, params),
        Primitive::Switch => eval_switch(primitive, inputs, params),
        // Collective operations (pmap-only, require multi-device runtime)
        Primitive::Psum
        | Primitive::Pmean
        | Primitive::AllGather
        | Primitive::AllToAll
        | Primitive::AxisIndex => Err(EvalError::Unsupported {
            primitive,
            detail: "collective operation requires pmap context with multi-device backend"
                .to_owned(),
        }),
        // Bitwise
        Primitive::BitwiseAnd
        | Primitive::BitwiseOr
        | Primitive::BitwiseXor
        | Primitive::ShiftLeft
        | Primitive::ShiftRightArithmetic
        | Primitive::ShiftRightLogical => eval_bitwise_binary(primitive, inputs),
        Primitive::BitwiseNot
        | Primitive::PopulationCount
        | Primitive::CountLeadingZeros
        | Primitive::CountTrailingZeros => eval_bitwise_unary(primitive, inputs),
        // Windowed reduction (pooling)
        Primitive::ReduceWindow => eval_reduce_window(primitive, inputs, params),
    }
}

/// Evaluate a primitive that may produce multiple outputs.
///
/// Multi-output primitives (Qr, Svd, Eigh) return `Vec<Value>` directly.
/// Single-output primitives delegate to `eval_primitive` and wrap the result.
pub fn eval_primitive_multi(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, EvalError> {
    match primitive {
        Primitive::Qr => eval_qr(inputs, params),
        Primitive::Lu => eval_lu(inputs, params),
        Primitive::Svd => eval_svd(inputs, params),
        Primitive::Eigh => eval_eigh(inputs, params),
        Primitive::TopK => eval_top_k(inputs, params),
        Primitive::Sort => eval_sort_multi(primitive, inputs, params),
        // Slogdet → (sign, logabsdet); Eig → (eigenvalues, eigenvectors).
        // The single-output `eval_primitive` path keeps only the first of
        // each, so route the multi-output evaluator to the real evals instead
        // of falling through (which silently truncated the second output and
        // would trip the interpreter's output-arity check on a 2-output
        // equation).
        Primitive::Slogdet => eval_slogdet(inputs, params),
        Primitive::Eig => eval_eig(inputs, params),
        // Split → one array per section (jnp.split's list-of-arrays). The single-
        // output path packs equal sections into one tensor and fails closed on
        // uneven sizes; the multi-output path returns N tensors, uneven supported.
        Primitive::Split => crate::tensor_ops::eval_split_multi(inputs, params),
        _ => eval_primitive(primitive, inputs, params).map(|v| vec![v]),
    }
}

/// Evaluate Cond: select between two operands based on a boolean predicate.
///
/// inputs: [predicate, true_value, false_value]
/// Returns true_value if predicate is true, false_value otherwise.
fn eval_cond(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 3 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }
    let true_shape = value_shape(&inputs[1]);
    let false_shape = value_shape(&inputs[2]);
    if true_shape != false_shape {
        return Err(EvalError::ShapeMismatch {
            primitive,
            left: true_shape,
            right: false_shape,
        });
    }
    let true_dtype = inputs[1].dtype();
    let false_dtype = inputs[2].dtype();
    if true_dtype != false_dtype {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "cond branches must have same dtype",
        });
    }
    let pred = value_to_bool(primitive, &inputs[0])?;
    if pred {
        Ok(inputs[1].clone())
    } else {
        Ok(inputs[2].clone())
    }
}

fn value_shape(value: &Value) -> Shape {
    match value {
        Value::Scalar(_) => Shape::scalar(),
        Value::Tensor(t) => t.shape.clone(),
    }
}

/// Evaluate Scan: iterate a body operation over slices of a tensor, threading carry state.
///
/// inputs: [init_carry, xs_tensor]
///   - init_carry: initial carry value (scalar or tensor)
///   - xs_tensor: tensor whose leading axis is scanned over
///
/// params:
///   - "body_op": the primitive to apply per iteration, e.g. "add", "mul", "div", "pow"
///     The body computes: new_carry = body_op(carry, x_slice)
///   - "length": optional explicit scan length (inferred from xs if absent)
///   - "reverse": "true" to scan in reverse order (default: "false")
///
/// Returns the final carry value after all iterations (legacy single-value API).
fn eval_scan(
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

    let init_carry = &inputs[0];
    let xs = &inputs[1];

    // Determine body operation (defaults to Add if not specified)
    let body_op_name = params.get("body_op").map(|s| s.as_str()).unwrap_or("add");
    let body_op = match body_op_name {
        "add" => Primitive::Add,
        "sub" => Primitive::Sub,
        "mul" => Primitive::Mul,
        "div" => Primitive::Div,
        "pow" => Primitive::Pow,
        "max" => Primitive::Max,
        "min" => Primitive::Min,
        other => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("unsupported scan body_op: {other}"),
            });
        }
    };

    let reverse = params.get("reverse").map(|s| s == "true").unwrap_or(false);

    // Extract slices from xs along leading axis
    match xs {
        Value::Scalar(_) => {
            // Single element scan — just apply body_op to carry and xs
            eval_primitive(body_op, &[init_carry.clone(), xs.clone()], &BTreeMap::new())
        }
        Value::Tensor(t) => {
            let leading_dim = scan_leading_dim(t)?;
            if leading_dim == 0 {
                return Ok(init_carry.clone());
            }

            let mut carry = init_carry.clone();
            if reverse {
                for i in (0..leading_dim).rev() {
                    let x_slice = t.slice_axis0(i).map_err(EvalError::InvalidTensor)?;
                    carry = eval_primitive(body_op, &[carry, x_slice], &BTreeMap::new())?;
                }
            } else {
                for i in 0..leading_dim {
                    let x_slice = t.slice_axis0(i).map_err(EvalError::InvalidTensor)?;
                    carry = eval_primitive(body_op, &[carry, x_slice], &BTreeMap::new())?;
                }
            }

            Ok(carry)
        }
    }
}

/// Evaluate associative_scan: parallel prefix scan using an associative binary operator.
///
/// Unlike regular scan, associative_scan:
/// 1. Takes only xs (no initial carry)
/// 2. Returns all prefix scan values, not just the final carry
/// 3. First output is xs[0], not fn(init, xs[0])
///
/// For an associative binary op ⊕:
///   out[0] = xs[0]
///   out[i] = xs[0] ⊕ xs[1] ⊕ ... ⊕ xs[i]
///
/// V1: Sequential implementation for correctness. Parallel optimization deferred.
fn eval_associative_scan(
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

    let xs = &inputs[0];

    let body_op_name = params.get("body_op").map(|s| s.as_str()).unwrap_or("add");
    let body_op = match body_op_name {
        "add" => Primitive::Add,
        "mul" => Primitive::Mul,
        "max" => Primitive::Max,
        "min" => Primitive::Min,
        "and" => Primitive::BitwiseAnd,
        "or" => Primitive::BitwiseOr,
        "xor" => Primitive::BitwiseXor,
        other => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("unsupported associative_scan body_op: {other}"),
            });
        }
    };

    let reverse = params.get("reverse").map(|s| s == "true").unwrap_or(false);

    match xs {
        Value::Scalar(lit) => Ok(Value::Scalar(*lit)),
        Value::Tensor(t) => {
            let leading_dim = scan_leading_dim(t)?;
            if leading_dim == 0 {
                return Ok(xs.clone());
            }
            if leading_dim == 1 {
                return Ok(xs.clone());
            }

            // Dense fast path for the scalar body_ops. The generic path below does
            // `leading_dim` separate `eval_primitive` dispatches, each allocating
            // slice/result `Value`s plus two clones, then a final `stack_axis0` —
            // O(leading_dim) heap churn dominated by dispatch/alloc, not the
            // elementwise op itself (13x on 1-D scans where inner==1). Here we scan
            // the typed buffer in place: out[i] = op(out[i-1], x[i]) (forward) /
            // op(x[i], out[i+1]) (reverse), matching the generic path's EXACT operand
            // order (so order-sensitive max/min on signed zero stay identical) and
            // the SAME scalar fn the elementwise primitive uses for that dtype:
            //   f64  -> IEEE +/* and jax_max_f64/jax_min_f64
            //   f32  -> op(a as f64, b as f64) as f32  (mirrors eval_same_shape_f32_map)
            //   i64  -> wrapping add/mul, signed max/min, bitwise and/or/xor
            //   i32  -> same integer ops, then narrow every computed prefix step
            //           through the public mod-2^32 chokepoint.
            // Other dtypes / body_ops fall through to the generic path unchanged.
            macro_rules! dense_scan_return {
                ($values:expr, $op:expr, $ctor:path) => {{
                    let values = $values;
                    let op = $op;
                    let inner = values.len() / leading_dim;
                    let mut out = values.to_vec(); // seeds every row; non-seed rows overwritten
                    if reverse {
                        for i in (0..leading_dim - 1).rev() {
                            let (base, next) = (i * inner, (i + 1) * inner);
                            for j in 0..inner {
                                out[base + j] = op(values[base + j], out[next + j]);
                            }
                        }
                    } else {
                        for i in 1..leading_dim {
                            let (base, prev) = (i * inner, (i - 1) * inner);
                            for j in 0..inner {
                                out[base + j] = op(out[prev + j], values[base + j]);
                            }
                        }
                    }
                    return $ctor(t.shape.clone(), out)
                        .map(Value::Tensor)
                        .map_err(EvalError::InvalidTensor);
                }};
            }
            if let Some(values) = t.elements.as_f64_slice() {
                match body_op {
                    Primitive::Add => {
                        dense_scan_return!(
                            values,
                            |a: f64, b: f64| a + b,
                            TensorValue::new_f64_values
                        )
                    }
                    Primitive::Mul => {
                        dense_scan_return!(
                            values,
                            |a: f64, b: f64| a * b,
                            TensorValue::new_f64_values
                        )
                    }
                    Primitive::Max => {
                        dense_scan_return!(values, jax_max_f64, TensorValue::new_f64_values)
                    }
                    Primitive::Min => {
                        dense_scan_return!(values, jax_min_f64, TensorValue::new_f64_values)
                    }
                    _ => {}
                }
            }
            if let Some(values) = t.elements.as_f32_slice() {
                match body_op {
                    Primitive::Add => dense_scan_return!(
                        values,
                        |a: f32, b: f32| (a as f64 + b as f64) as f32,
                        TensorValue::new_f32_values
                    ),
                    Primitive::Mul => dense_scan_return!(
                        values,
                        |a: f32, b: f32| (a as f64 * b as f64) as f32,
                        TensorValue::new_f32_values
                    ),
                    Primitive::Max => dense_scan_return!(
                        values,
                        |a: f32, b: f32| jax_max_f64(a as f64, b as f64) as f32,
                        TensorValue::new_f32_values
                    ),
                    Primitive::Min => dense_scan_return!(
                        values,
                        |a: f32, b: f32| jax_min_f64(a as f64, b as f64) as f32,
                        TensorValue::new_f32_values
                    ),
                    _ => {}
                }
            }
            if let Some(values) = t.elements.as_i64_slice() {
                match (t.dtype, body_op) {
                    (fj_core::DType::I64, Primitive::Add) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| a.wrapping_add(b),
                        TensorValue::new_i64_values
                    ),
                    (fj_core::DType::I64, Primitive::Mul) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| a.wrapping_mul(b),
                        TensorValue::new_i64_values
                    ),
                    (fj_core::DType::I64, Primitive::Max) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| a.max(b),
                        TensorValue::new_i64_values
                    ),
                    (fj_core::DType::I64, Primitive::Min) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| a.min(b),
                        TensorValue::new_i64_values
                    ),
                    (fj_core::DType::I64, Primitive::BitwiseAnd) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| a & b,
                        TensorValue::new_i64_values
                    ),
                    (fj_core::DType::I64, Primitive::BitwiseOr) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| a | b,
                        TensorValue::new_i64_values
                    ),
                    (fj_core::DType::I64, Primitive::BitwiseXor) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| a ^ b,
                        TensorValue::new_i64_values
                    ),
                    (fj_core::DType::I32, Primitive::Add) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| i64::from(a.wrapping_add(b) as i32),
                        TensorValue::new_i32_values
                    ),
                    (fj_core::DType::I32, Primitive::Mul) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| i64::from(a.wrapping_mul(b) as i32),
                        TensorValue::new_i32_values
                    ),
                    (fj_core::DType::I32, Primitive::Max) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| i64::from(a.max(b) as i32),
                        TensorValue::new_i32_values
                    ),
                    (fj_core::DType::I32, Primitive::Min) => dense_scan_return!(
                        values,
                        |a: i64, b: i64| i64::from(a.min(b) as i32),
                        TensorValue::new_i32_values
                    ),
                    _ => {}
                }
            }

            let mut results: Vec<Value> = Vec::with_capacity(leading_dim);

            if reverse {
                let mut acc = t
                    .slice_axis0(leading_dim - 1)
                    .map_err(EvalError::InvalidTensor)?;
                results.push(acc.clone());
                for i in (0..leading_dim - 1).rev() {
                    let x_slice = t.slice_axis0(i).map_err(EvalError::InvalidTensor)?;
                    acc = eval_primitive(body_op, &[x_slice, acc], &BTreeMap::new())?;
                    results.push(acc.clone());
                }
                results.reverse();
            } else {
                let mut acc = t.slice_axis0(0).map_err(EvalError::InvalidTensor)?;
                results.push(acc.clone());
                for i in 1..leading_dim {
                    let x_slice = t.slice_axis0(i).map_err(EvalError::InvalidTensor)?;
                    acc = eval_primitive(body_op, &[acc, x_slice], &BTreeMap::new())?;
                    results.push(acc.clone());
                }
            }

            TensorValue::stack_axis0(&results)
                .map(Value::Tensor)
                .map_err(EvalError::InvalidTensor)
        }
    }
}

/// Evaluate scan with a functional body: `body_fn(carry, x_i) -> (new_carry, y_i)`.
///
/// This is the full JAX-compatible scan API:
///   `scan(body_fn, init_carry, xs) -> (final_carry, stacked_ys)`
///
/// The body function receives the current carry and one slice of xs, and
/// returns a new carry and an output value. Outputs are collected and stacked.
///
/// - `init_carry`: initial carry values (one or more)
/// - `xs`: tensor to scan over (leading axis is iterated)
/// - `body_fn`: `(carry_values, x_slice) -> (new_carry_values, output_values)`
/// - `reverse`: if true, iterate xs in reverse order while returning `ys` in input order
///
/// Returns `(final_carry_values, stacked_ys)` where `stacked_ys` are stacked
/// along a new leading axis.
pub fn eval_scan_functional<B>(
    init_carry: Vec<Value>,
    xs: &Value,
    mut body_fn: B,
    reverse: bool,
) -> Result<(Vec<Value>, Vec<Value>), EvalError>
where
    B: FnMut(Vec<Value>, Value) -> Result<(Vec<Value>, Vec<Value>), EvalError>,
{
    let scan_len = scan_input_len(xs)?;

    if scan_len == 0 {
        // Empty scan: return init carry, no outputs
        return Ok((init_carry, vec![]));
    }

    let mut carry = init_carry;
    let mut per_output_values: Vec<Vec<Value>> = Vec::new();

    if reverse {
        for i in (0..scan_len).rev() {
            carry = eval_scan_functional_step(
                xs,
                i,
                carry,
                &mut body_fn,
                &mut per_output_values,
                scan_len,
            )?;
        }
    } else {
        for i in 0..scan_len {
            carry = eval_scan_functional_step(
                xs,
                i,
                carry,
                &mut body_fn,
                &mut per_output_values,
                scan_len,
            )?;
        }
    }

    if reverse {
        for values in &mut per_output_values {
            values.reverse();
        }
    }

    // Stack each output along a new leading axis
    let stacked_ys: Vec<Value> = per_output_values
        .into_iter()
        .map(|values| {
            TensorValue::stack_axis0(&values)
                .map(Value::Tensor)
                .map_err(EvalError::InvalidTensor)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((carry, stacked_ys))
}

fn eval_scan_functional_step<B>(
    xs: &Value,
    index: usize,
    carry: Vec<Value>,
    body_fn: &mut B,
    per_output_values: &mut Vec<Vec<Value>>,
    output_capacity: usize,
) -> Result<Vec<Value>, EvalError>
where
    B: FnMut(Vec<Value>, Value) -> Result<(Vec<Value>, Vec<Value>), EvalError>,
{
    let x_slice = scan_slice_at(xs, index)?;
    let (new_carry, ys) = body_fn(carry, x_slice)?;

    // Initialize per-output collectors on first iteration.
    if per_output_values.is_empty() {
        *per_output_values = vec![Vec::with_capacity(output_capacity); ys.len()];
    }

    for (out_idx, y) in ys.into_iter().enumerate() {
        if out_idx < per_output_values.len() {
            per_output_values[out_idx].push(y);
        }
    }

    Ok(new_carry)
}

fn scan_input_len(xs: &Value) -> Result<usize, EvalError> {
    match xs {
        Value::Scalar(_) => Ok(1),
        Value::Tensor(t) => scan_leading_dim(t),
    }
}

fn scan_leading_dim(tensor: &TensorValue) -> Result<usize, EvalError> {
    tensor
        .shape
        .dims
        .first()
        .map(|dim| *dim as usize)
        .ok_or(EvalError::TypeMismatch {
            primitive: Primitive::Scan,
            detail: "scan tensor xs must have a leading axis",
        })
}

fn scan_slice_at(xs: &Value, index: usize) -> Result<Value, EvalError> {
    match xs {
        Value::Scalar(_) => Ok(xs.clone()),
        Value::Tensor(t) => t.slice_axis0(index).map_err(EvalError::InvalidTensor),
    }
}

/// Evaluate While: iterate a body operation on carry while a condition holds.
///
/// inputs: [init_carry, step_value, threshold]
///   - init_carry: initial carry value (scalar or tensor)
///   - step_value: value applied via body_op each iteration
///   - threshold: value compared against via cond_op
///
/// params:
///   - "body_op": the primitive to apply per iteration, e.g. "add", "mul", "div", "pow"
///     The body computes: new_carry = body_op(carry, step_value)
///   - "cond_op": comparison primitive, e.g. "lt", "le", "gt", "ge", "ne", "eq"
///     Loop continues while: cond_op(carry, threshold) is true
///   - "max_iter": safety limit on iterations (default: 1000)
///
/// Returns the final carry value when the condition becomes false.
/// Returns `MaxIterationsExceeded` if the limit is reached without the condition
/// becoming false.
fn eval_while_loop(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.len() != 3 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 3,
            actual: inputs.len(),
        });
    }

    let init_carry = &inputs[0];
    let step_value = &inputs[1];
    let threshold = &inputs[2];

    let body_op_name = params.get("body_op").map(|s| s.as_str()).unwrap_or("add");
    let body_op = parse_while_body_op(primitive, body_op_name)?;

    let cond_op_name = params.get("cond_op").map(|s| s.as_str()).unwrap_or("lt");
    let cond_op = parse_while_cond_op(primitive, cond_op_name)?;

    let max_iter: usize = params
        .get("max_iter")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let init_shape = value_shape_fingerprint(init_carry);

    let cond_fn = |carry: &Value| -> Result<bool, EvalError> {
        let cond_result = eval_primitive(
            cond_op,
            &[carry.clone(), threshold.clone()],
            &BTreeMap::new(),
        )?;
        value_to_bool(primitive, &cond_result)
    };

    let body_fn = |carry: Value| -> Result<Value, EvalError> {
        eval_primitive(body_op, &[carry, step_value.clone()], &BTreeMap::new())
    };

    eval_while_loop_core(
        primitive,
        init_carry.clone(),
        &init_shape,
        max_iter,
        cond_fn,
        body_fn,
    )
}

/// Evaluate a while loop with arbitrary condition and body functions.
///
/// This is the core implementation used by both the param-based `eval_while_loop`
/// and can be called directly by higher-level evaluators (e.g., fj-dispatch)
/// that provide sub_jaxpr-based condition and body functions.
pub fn eval_while_loop_functional<C, B>(
    init_carry: Vec<Value>,
    max_iterations: usize,
    mut cond_fn: C,
    mut body_fn: B,
) -> Result<Vec<Value>, EvalError>
where
    C: FnMut(&[Value]) -> Result<bool, EvalError>,
    B: FnMut(Vec<Value>) -> Result<Vec<Value>, EvalError>,
{
    let init_shapes: Vec<String> = init_carry.iter().map(value_shape_fingerprint).collect();
    let mut carry = init_carry;

    for _ in 0..max_iterations {
        if !cond_fn(&carry)? {
            return Ok(carry);
        }
        carry = body_fn(carry)?;
        // Verify carry shape is preserved
        for (i, (new_shape, orig_shape)) in carry
            .iter()
            .map(value_shape_fingerprint)
            .zip(init_shapes.iter())
            .enumerate()
        {
            if new_shape != *orig_shape {
                return Err(EvalError::ShapeChanged {
                    primitive: Primitive::While,
                    detail: format!(
                        "carry element {i} changed shape from {orig_shape} to {new_shape}"
                    ),
                });
            }
        }
    }
    Err(EvalError::MaxIterationsExceeded {
        primitive: Primitive::While,
        max_iterations,
    })
}

/// Internal while_loop core that handles single-value carry.
fn eval_while_loop_core<C, B>(
    primitive: Primitive,
    init: Value,
    init_shape: &str,
    max_iter: usize,
    mut cond_fn: C,
    mut body_fn: B,
) -> Result<Value, EvalError>
where
    C: FnMut(&Value) -> Result<bool, EvalError>,
    B: FnMut(Value) -> Result<Value, EvalError>,
{
    let mut carry = init;

    for _ in 0..max_iter {
        if !cond_fn(&carry)? {
            return Ok(carry);
        }
        carry = body_fn(carry)?;
        // Verify shape preservation
        let new_shape = value_shape_fingerprint(&carry);
        if new_shape != init_shape {
            return Err(EvalError::ShapeChanged {
                primitive,
                detail: format!("carry changed from {init_shape} to {new_shape}"),
            });
        }
    }
    Err(EvalError::MaxIterationsExceeded {
        primitive,
        max_iterations: max_iter,
    })
}

fn parse_while_body_op(primitive: Primitive, name: &str) -> Result<Primitive, EvalError> {
    match name {
        "add" => Ok(Primitive::Add),
        "sub" => Ok(Primitive::Sub),
        "mul" => Ok(Primitive::Mul),
        "div" => Ok(Primitive::Div),
        "pow" => Ok(Primitive::Pow),
        other => Err(EvalError::Unsupported {
            primitive,
            detail: format!("unsupported while body_op: {other}"),
        }),
    }
}

fn parse_while_cond_op(primitive: Primitive, name: &str) -> Result<Primitive, EvalError> {
    match name {
        "lt" => Ok(Primitive::Lt),
        "le" => Ok(Primitive::Le),
        "gt" => Ok(Primitive::Gt),
        "ge" => Ok(Primitive::Ge),
        "ne" => Ok(Primitive::Ne),
        "eq" => Ok(Primitive::Eq),
        other => Err(EvalError::Unsupported {
            primitive,
            detail: format!("unsupported while cond_op: {other}"),
        }),
    }
}

fn scalar_literal(primitive: Primitive, value: &Value) -> Result<Literal, EvalError> {
    match value {
        Value::Scalar(lit) => Ok(*lit),
        Value::Tensor(tensor) => {
            if tensor.shape != Shape::scalar() {
                return Err(EvalError::ShapeMismatch {
                    primitive,
                    left: tensor.shape.clone(),
                    right: Shape::scalar(),
                });
            }
            if tensor.elements.len() != 1 {
                return Err(EvalError::InvalidTensor(ValueError::ElementCountMismatch {
                    shape: tensor.shape.clone(),
                    expected_count: 1,
                    actual_count: tensor.elements.len(),
                }));
            }
            Ok(tensor.elements[0])
        }
    }
}

fn literal_to_bool(primitive: Primitive, literal: Literal) -> Result<bool, EvalError> {
    match literal {
        Literal::Bool(b) => Ok(b),
        Literal::I32(v) => Ok(v != 0),
        Literal::I64(v) => Ok(v != 0),
        Literal::U32(v) => Ok(v != 0),
        Literal::U64(v) => Ok(v != 0),
        Literal::BF16Bits(bits) => Ok(Literal::BF16Bits(bits).as_f64().is_some_and(|v| v != 0.0)),
        Literal::F16Bits(bits) => Ok(Literal::F16Bits(bits).as_f64().is_some_and(|v| v != 0.0)),
        Literal::F32Bits(bits) => Ok(f32::from_bits(bits) != 0.0),
        Literal::F64Bits(bits) => Ok(f64::from_bits(bits) != 0.0),
        Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => Err(EvalError::TypeMismatch {
            primitive,
            detail: "predicate must be boolean or numeric",
        }),
    }
}

/// Extract a boolean-ish value from a comparison result.
fn value_to_bool(primitive: Primitive, value: &Value) -> Result<bool, EvalError> {
    let literal = scalar_literal(primitive, value)?;
    literal_to_bool(primitive, literal)
}

fn literal_to_switch_index(
    primitive: Primitive,
    literal: Literal,
    num_branches: usize,
) -> Result<usize, EvalError> {
    let last_branch = num_branches.saturating_sub(1);
    match literal {
        Literal::Bool(b) => Ok(usize::from(b).min(last_branch)),
        Literal::I32(v) => {
            let clamped = if v <= 0 {
                0
            } else {
                (v as u32).min(last_branch as u32) as usize
            };
            Ok(clamped)
        }
        Literal::I64(v) => {
            let clamped = if v <= 0 {
                0
            } else {
                (v as u64).min(last_branch as u64) as usize
            };
            Ok(clamped)
        }
        Literal::U32(v) => Ok((v as usize).min(last_branch)),
        Literal::U64(v) => Ok(v.min(last_branch as u64) as usize),
        Literal::BF16Bits(..)
        | Literal::F16Bits(..)
        | Literal::F32Bits(..)
        | Literal::F64Bits(..)
        | Literal::Complex64Bits(..)
        | Literal::Complex128Bits(..) => Err(EvalError::TypeMismatch {
            primitive,
            detail: "switch index must be integer or bool",
        }),
    }
}

fn value_to_switch_index(
    primitive: Primitive,
    value: &Value,
    num_branches: usize,
) -> Result<usize, EvalError> {
    let literal = scalar_literal(primitive, value)?;
    literal_to_switch_index(primitive, literal, num_branches)
}

/// Compute a simple shape fingerprint for shape-preservation checks.
fn value_shape_fingerprint(v: &Value) -> String {
    match v {
        Value::Scalar(lit) => {
            let kind = match lit {
                fj_core::Literal::I32(_) => "i32",
                fj_core::Literal::I64(_) => "i64",
                fj_core::Literal::U32(_) => "u32",
                fj_core::Literal::U64(_) => "u64",
                fj_core::Literal::Bool(_) => "bool",
                fj_core::Literal::BF16Bits(_) => "bf16",
                fj_core::Literal::F16Bits(_) => "f16",
                fj_core::Literal::F32Bits(_) => "f32",
                fj_core::Literal::F64Bits(_) => "f64",
                fj_core::Literal::Complex64Bits(_, _) => "c64",
                fj_core::Literal::Complex128Bits(_, _) => "c128",
            };
            format!("scalar:{kind}")
        }
        Value::Tensor(t) => format!("tensor:{:?}:{:?}", t.dtype, t.shape.dims),
    }
}

/// Evaluate the primitive-form `Switch`: select among precomputed branch values.
///
/// inputs: [index, branch0_result, branch1_result, ...]
///   - index: integer selecting which branch to take
///   - branch results: pre-computed results for each branch
///
/// Equation-level control-flow execution for `Switch` with `sub_jaxprs`
/// happens in the Jaxpr interpreter/backend, not here.
///
/// params:
///   - "num_branches": number of branches (optional; defaults to provided branch values)
fn eval_switch(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.len() < 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let expected_shape = value_shape(&inputs[1]);
    let expected_dtype = inputs[1].dtype();
    for (_idx, branch) in inputs.iter().enumerate().skip(2) {
        let other_shape = value_shape(branch);
        if other_shape != expected_shape {
            return Err(EvalError::ShapeMismatch {
                primitive,
                left: expected_shape,
                right: other_shape,
            });
        }
        if branch.dtype() != expected_dtype {
            return Err(EvalError::TypeMismatch {
                primitive,
                detail: "switch branches must have same dtype",
            });
        }
    }

    let provided_branches = inputs.len().saturating_sub(1);
    let num_branches = if let Some(raw) = params.get("num_branches") {
        raw.parse::<usize>().map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("invalid num_branches value: {raw}"),
        })?
    } else {
        provided_branches
    };
    if num_branches != provided_branches {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "switch expected {num_branches} branch values but got {provided_branches}"
            ),
        });
    }

    // Branch values start at inputs[1]
    let branch_idx = value_to_switch_index(primitive, &inputs[0], num_branches)?;
    Ok(inputs[branch_idx + 1].clone())
}

/// Evaluate fori_loop: `fori_loop(lower, upper, body_fn, init_val) -> final_val`.
///
/// Desugars to a while_loop with an explicit counter:
///   carry = (counter=lower, val=init_val)
///   while counter < upper:
///     val = body_fn(counter, val)
///     counter += 1
///   return val
pub fn eval_fori_loop<B>(
    lower: i64,
    upper: i64,
    init_val: Value,
    mut body_fn: B,
) -> Result<Value, EvalError>
where
    B: FnMut(i64, Value) -> Result<Value, EvalError>,
{
    let mut val = init_val;
    for i in lower..upper {
        val = body_fn(i, val)?;
    }
    Ok(val)
}

/// Evaluate a binary bitwise operation on integer values.
fn apply_bitwise_binary_i64(primitive: Primitive, lhs: i64, rhs: i64) -> i64 {
    match primitive {
        Primitive::BitwiseAnd => lhs & rhs,
        Primitive::BitwiseOr => lhs | rhs,
        Primitive::BitwiseXor => lhs ^ rhs,
        Primitive::ShiftLeft => lhs.wrapping_shl(rhs as u32),
        Primitive::ShiftRightArithmetic => lhs.wrapping_shr(rhs as u32),
        Primitive::ShiftRightLogical => ((lhs as u64).wrapping_shr(rhs as u32)) as i64,
        _ => lhs,
    }
}

/// Bitwise binary on i32 operands (stored sign-extended in i64). Done at i32 WIDTH —
/// crucially ShiftRightLogical must zero-fill from bit 31 (mask the sign extension via
/// `as u32`), NOT bit 63, and ShiftLeft/arithmetic shr mask the count to <32 (i32 width).
/// AND/OR/XOR are bit-preserving and stay valid i32. The i32 result is sign-extended back
/// into i64 storage. JAX/XLA leave shift-by-≥width undefined; Rust's `wrapping_sh*`
/// (count % 32) is a consistent definition.
fn apply_bitwise_binary_i32(primitive: Primitive, lhs: i64, rhs: i64) -> i64 {
    let a = lhs as i32;
    let b = rhs as i32;
    let r: i32 = match primitive {
        Primitive::BitwiseAnd => a & b,
        Primitive::BitwiseOr => a | b,
        Primitive::BitwiseXor => a ^ b,
        Primitive::ShiftLeft => a.wrapping_shl(b as u32),
        Primitive::ShiftRightArithmetic => a.wrapping_shr(b as u32),
        Primitive::ShiftRightLogical => ((a as u32).wrapping_shr(b as u32)) as i32,
        _ => a,
    };
    i64::from(r)
}

/// Bitwise (= logical) `and`/`or`/`xor` on `Bool`. JAX lowers `logical_and`,
/// bool `&`, and even bool `*` to `lax.bitwise_and` on the bool (PRED) dtype
/// (jax/_src/numpy/ufuncs.py), so the eval layer must accept Bool operands and
/// return a Bool result rather than erroring. Shifts are undefined on Bool
/// (JAX promotes bool→int before a shift), so they return `None`.
fn apply_bitwise_binary_bool(primitive: Primitive, lhs: bool, rhs: bool) -> Option<bool> {
    match primitive {
        Primitive::BitwiseAnd => Some(lhs & rhs),
        Primitive::BitwiseOr => Some(lhs | rhs),
        Primitive::BitwiseXor => Some(lhs ^ rhs),
        _ => None,
    }
}

fn bool_shift_unsupported(primitive: Primitive) -> EvalError {
    EvalError::TypeMismatch {
        primitive,
        detail: "shift ops are undefined on bool",
    }
}

/// `Bool` scalar ⊗ `Bool` tensor bitwise op. and/or/xor are commutative, so the
/// scalar position is irrelevant. Reads the dense Bool backing when available and
/// emits a dense Bool result.
fn eval_bitwise_bool_scalar_tensor(
    primitive: Primitive,
    scalar: bool,
    tensor: &TensorValue,
) -> Result<Value, EvalError> {
    if !matches!(
        primitive,
        Primitive::BitwiseAnd | Primitive::BitwiseOr | Primitive::BitwiseXor
    ) {
        return Err(bool_shift_unsupported(primitive));
    }
    if let Some(values) = tensor.elements.as_bool_slice() {
        let out: Vec<bool> = values
            .iter()
            .map(|&v| apply_bitwise_binary_bool(primitive, scalar, v).unwrap_or(false))
            .collect();
        return Ok(Value::Tensor(
            TensorValue::new_bool_values(tensor.shape.clone(), out)
                .map_err(EvalError::InvalidTensor)?,
        ));
    }
    let mut out = Vec::with_capacity(tensor.elements.len());
    for el in tensor.elements.iter() {
        let fj_core::Literal::Bool(v) = el else {
            return Err(EvalError::TypeMismatch {
                primitive,
                detail: "bitwise bool op requires bool tensor",
            });
        };
        out.push(apply_bitwise_binary_bool(primitive, scalar, *v).unwrap_or(false));
    }
    Ok(Value::Tensor(
        TensorValue::new_bool_values(tensor.shape.clone(), out)
            .map_err(EvalError::InvalidTensor)?,
    ))
}

fn apply_bitwise_binary_u32(primitive: Primitive, lhs: u32, rhs: u32) -> u32 {
    match primitive {
        Primitive::BitwiseAnd => lhs & rhs,
        Primitive::BitwiseOr => lhs | rhs,
        Primitive::BitwiseXor => lhs ^ rhs,
        Primitive::ShiftLeft => lhs.wrapping_shl(rhs),
        Primitive::ShiftRightArithmetic | Primitive::ShiftRightLogical => lhs.wrapping_shr(rhs),
        _ => lhs,
    }
}

fn apply_bitwise_binary_u64(primitive: Primitive, lhs: u64, rhs: u64) -> u64 {
    match primitive {
        Primitive::BitwiseAnd => lhs & rhs,
        Primitive::BitwiseOr => lhs | rhs,
        Primitive::BitwiseXor => lhs ^ rhs,
        Primitive::ShiftLeft => lhs.wrapping_shl(rhs as u32),
        Primitive::ShiftRightArithmetic | Primitive::ShiftRightLogical => {
            lhs.wrapping_shr(rhs as u32)
        }
        _ => lhs,
    }
}

// Broadcast helpers for bitwise operations
fn bitwise_broadcast_shape(lhs: &Shape, rhs: &Shape) -> Option<Shape> {
    let max_rank = lhs.rank().max(rhs.rank());
    let mut dims = Vec::with_capacity(max_rank);

    for offset in 0..max_rank {
        let lhs_dim = if offset < lhs.rank() {
            lhs.dims[lhs.rank() - 1 - offset]
        } else {
            1
        };
        let rhs_dim = if offset < rhs.rank() {
            rhs.dims[rhs.rank() - 1 - offset]
        } else {
            1
        };

        let out_dim = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            return None;
        };
        dims.push(out_dim);
    }

    dims.reverse();
    Some(Shape { dims })
}

fn bitwise_compute_strides(dims: &[u32]) -> Vec<usize> {
    let mut strides = vec![1_usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as usize;
    }
    strides
}

fn bitwise_flat_to_multi(flat: usize, strides: &[usize]) -> Vec<usize> {
    let mut multi = Vec::with_capacity(strides.len());
    let mut remainder = flat;
    for &stride in strides {
        multi.push(remainder / stride);
        remainder %= stride;
    }
    multi
}

fn bitwise_broadcast_strides(shape: &Shape, out_shape: &Shape) -> Vec<usize> {
    let rank = shape.rank();
    let out_rank = out_shape.rank();
    let real_strides = bitwise_compute_strides(&shape.dims);

    let mut result = vec![0_usize; out_rank];
    for (i, &stride) in real_strides.iter().enumerate().take(rank) {
        let out_axis = out_rank - rank + i;
        if shape.dims[i] == 1 {
            result[out_axis] = 0;
        } else {
            result[out_axis] = stride;
        }
    }
    result
}

fn bitwise_broadcast_flat_index(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides.iter()).map(|(&m, &s)| m * s).sum()
}

fn eval_bitwise_binary(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }
    match (&inputs[0], &inputs[1]) {
        // Scalar-Scalar
        (Value::Scalar(fj_core::Literal::Bool(a)), Value::Scalar(fj_core::Literal::Bool(b))) => {
            apply_bitwise_binary_bool(primitive, *a, *b)
                .map(Value::scalar_bool)
                .ok_or(EvalError::TypeMismatch {
                    primitive,
                    detail: "shift ops are undefined on bool",
                })
        }
        (Value::Scalar(fj_core::Literal::I64(a)), Value::Scalar(fj_core::Literal::I64(b))) => Ok(
            Value::scalar_i64(apply_bitwise_binary_i64(primitive, *a, *b)),
        ),
        (Value::Scalar(fj_core::Literal::I32(a)), Value::Scalar(fj_core::Literal::I32(b))) => {
            Ok(Value::scalar_i32(
                apply_bitwise_binary_i32(primitive, i64::from(*a), i64::from(*b)) as i32,
            ))
        }
        (Value::Scalar(fj_core::Literal::U32(a)), Value::Scalar(fj_core::Literal::U32(b))) => Ok(
            Value::scalar_u32(apply_bitwise_binary_u32(primitive, *a, *b)),
        ),
        (Value::Scalar(fj_core::Literal::U64(a)), Value::Scalar(fj_core::Literal::U64(b))) => Ok(
            Value::scalar_u64(apply_bitwise_binary_u64(primitive, *a, *b)),
        ),

        // Scalar-Tensor broadcast (Bool)
        (Value::Scalar(fj_core::Literal::Bool(scalar)), Value::Tensor(tensor))
            if tensor.dtype == fj_core::DType::Bool =>
        {
            eval_bitwise_bool_scalar_tensor(primitive, *scalar, tensor)
        }
        (Value::Tensor(tensor), Value::Scalar(fj_core::Literal::Bool(scalar)))
            if tensor.dtype == fj_core::DType::Bool =>
        {
            eval_bitwise_bool_scalar_tensor(primitive, *scalar, tensor)
        }

        // Scalar-Tensor broadcast
        (Value::Scalar(fj_core::Literal::I64(scalar)), Value::Tensor(tensor))
            if tensor.dtype == fj_core::DType::I64 =>
        {
            // Dense I64 fast path (sibling of the I32/U32/U64 arms): map the typed
            // i64 slice straight into dense output, skipping the per-element 24-byte
            // Literal reconstruction + boxed build + densify rescan. `as_i64_slice`
            // is Some for I64 and I32, but this arm is gated to I64. Bit-identical:
            // same apply_bitwise_binary_i64(scalar, v), same I64 dense storage.
            if let Some(vals) = tensor.elements.as_i64_slice() {
                let out: Vec<i64> = vals
                    .iter()
                    .map(|&v| apply_bitwise_binary_i64(primitive, *scalar, v))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_i64_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for el in tensor.elements.iter() {
                match el {
                    fj_core::Literal::I64(v) => {
                        elements.push(fj_core::Literal::I64(apply_bitwise_binary_i64(
                            primitive, *scalar, *v,
                        )));
                    }
                    _ => {
                        return Err(EvalError::TypeMismatch {
                            primitive,
                            detail: "bitwise ops require integer tensors",
                        });
                    }
                }
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::I64, tensor.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        // i32 (JAX's default int) scalar⊗tensor.
        (Value::Scalar(fj_core::Literal::I32(scalar)), Value::Tensor(tensor))
            if tensor.dtype == fj_core::DType::I32 =>
        {
            if let Some(vals) = tensor.elements.as_i64_slice() {
                let out: Vec<i64> = vals
                    .iter()
                    .map(|&v| apply_bitwise_binary_i32(primitive, i64::from(*scalar), v))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_i32_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for el in tensor.elements.iter() {
                let fj_core::Literal::I64(v) = el else {
                    return Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "bitwise ops require integer tensors",
                    });
                };
                elements.push(fj_core::Literal::I64(apply_bitwise_binary_i32(
                    primitive,
                    i64::from(*scalar),
                    *v,
                )));
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::I32, tensor.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        (Value::Tensor(tensor), Value::Scalar(fj_core::Literal::I32(scalar)))
            if tensor.dtype == fj_core::DType::I32 =>
        {
            if let Some(vals) = tensor.elements.as_i64_slice() {
                let out: Vec<i64> = vals
                    .iter()
                    .map(|&v| apply_bitwise_binary_i32(primitive, v, i64::from(*scalar)))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_i32_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for el in tensor.elements.iter() {
                let fj_core::Literal::I64(v) = el else {
                    return Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "bitwise ops require integer tensors",
                    });
                };
                elements.push(fj_core::Literal::I64(apply_bitwise_binary_i32(
                    primitive,
                    *v,
                    i64::from(*scalar),
                )));
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::I32, tensor.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        // Legacy dtype-less Literal::I64 scalar is still accepted with I32 tensors.
        (Value::Scalar(fj_core::Literal::I64(scalar)), Value::Tensor(tensor))
            if tensor.dtype == fj_core::DType::I32 =>
        {
            if let Some(vals) = tensor.elements.as_i64_slice() {
                let out: Vec<i64> = vals
                    .iter()
                    .map(|&v| apply_bitwise_binary_i32(primitive, *scalar, v))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_i32_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for el in tensor.elements.iter() {
                let fj_core::Literal::I64(v) = el else {
                    return Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "bitwise ops require integer tensors",
                    });
                };
                elements.push(fj_core::Literal::I64(apply_bitwise_binary_i32(
                    primitive, *scalar, *v,
                )));
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::I32, tensor.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        (Value::Tensor(tensor), Value::Scalar(fj_core::Literal::I64(scalar)))
            if tensor.dtype == fj_core::DType::I32 =>
        {
            if let Some(vals) = tensor.elements.as_i64_slice() {
                let out: Vec<i64> = vals
                    .iter()
                    .map(|&v| apply_bitwise_binary_i32(primitive, v, *scalar))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_i32_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for el in tensor.elements.iter() {
                let fj_core::Literal::I64(v) = el else {
                    return Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "bitwise ops require integer tensors",
                    });
                };
                elements.push(fj_core::Literal::I64(apply_bitwise_binary_i32(
                    primitive, *v, *scalar,
                )));
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::I32, tensor.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        (Value::Scalar(fj_core::Literal::U32(scalar)), Value::Tensor(tensor))
            if tensor.dtype == fj_core::DType::U32 =>
        {
            if let Some(vals) = tensor.elements.as_u32_slice() {
                let out: Vec<u32> = vals
                    .iter()
                    .map(|&v| apply_bitwise_binary_u32(primitive, *scalar, v))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_u32_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for el in tensor.elements.iter() {
                match el {
                    fj_core::Literal::U32(v) => {
                        elements.push(fj_core::Literal::U32(apply_bitwise_binary_u32(
                            primitive, *scalar, *v,
                        )));
                    }
                    _ => {
                        return Err(EvalError::TypeMismatch {
                            primitive,
                            detail: "bitwise ops require integer tensors",
                        });
                    }
                }
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::U32, tensor.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        (Value::Scalar(fj_core::Literal::U64(scalar)), Value::Tensor(tensor))
            if tensor.dtype == fj_core::DType::U64 =>
        {
            if let Some(vals) = tensor.elements.as_u64_slice() {
                let out: Vec<u64> = vals
                    .iter()
                    .map(|&v| apply_bitwise_binary_u64(primitive, *scalar, v))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_u64_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for el in tensor.elements.iter() {
                match el {
                    fj_core::Literal::U64(v) => {
                        elements.push(fj_core::Literal::U64(apply_bitwise_binary_u64(
                            primitive, *scalar, *v,
                        )));
                    }
                    _ => {
                        return Err(EvalError::TypeMismatch {
                            primitive,
                            detail: "bitwise ops require integer tensors",
                        });
                    }
                }
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::U64, tensor.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }

        // Tensor-Scalar broadcast
        (Value::Tensor(tensor), Value::Scalar(fj_core::Literal::I64(scalar)))
            if tensor.dtype == fj_core::DType::I64 =>
        {
            // Dense I64 fast path (tensor⊗scalar direction): same as above with the
            // scalar as the rhs of apply_bitwise_binary_i64. Bit-identical.
            if let Some(vals) = tensor.elements.as_i64_slice() {
                let out: Vec<i64> = vals
                    .iter()
                    .map(|&v| apply_bitwise_binary_i64(primitive, v, *scalar))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_i64_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for el in tensor.elements.iter() {
                match el {
                    fj_core::Literal::I64(v) => {
                        elements.push(fj_core::Literal::I64(apply_bitwise_binary_i64(
                            primitive, *v, *scalar,
                        )));
                    }
                    _ => {
                        return Err(EvalError::TypeMismatch {
                            primitive,
                            detail: "bitwise ops require integer tensors",
                        });
                    }
                }
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::I64, tensor.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        (Value::Tensor(tensor), Value::Scalar(fj_core::Literal::U32(scalar)))
            if tensor.dtype == fj_core::DType::U32 =>
        {
            if let Some(vals) = tensor.elements.as_u32_slice() {
                let out: Vec<u32> = vals
                    .iter()
                    .map(|&v| apply_bitwise_binary_u32(primitive, v, *scalar))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_u32_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for el in tensor.elements.iter() {
                match el {
                    fj_core::Literal::U32(v) => {
                        elements.push(fj_core::Literal::U32(apply_bitwise_binary_u32(
                            primitive, *v, *scalar,
                        )));
                    }
                    _ => {
                        return Err(EvalError::TypeMismatch {
                            primitive,
                            detail: "bitwise ops require integer tensors",
                        });
                    }
                }
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::U32, tensor.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        (Value::Tensor(tensor), Value::Scalar(fj_core::Literal::U64(scalar)))
            if tensor.dtype == fj_core::DType::U64 =>
        {
            if let Some(vals) = tensor.elements.as_u64_slice() {
                let out: Vec<u64> = vals
                    .iter()
                    .map(|&v| apply_bitwise_binary_u64(primitive, v, *scalar))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_u64_values(tensor.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(tensor.elements.len());
            for el in tensor.elements.iter() {
                match el {
                    fj_core::Literal::U64(v) => {
                        elements.push(fj_core::Literal::U64(apply_bitwise_binary_u64(
                            primitive, *v, *scalar,
                        )));
                    }
                    _ => {
                        return Err(EvalError::TypeMismatch {
                            primitive,
                            detail: "bitwise ops require integer tensors",
                        });
                    }
                }
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::U64, tensor.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }

        // Tensor-Tensor (same shape or broadcast)
        (Value::Tensor(a), Value::Tensor(b)) => {
            if a.dtype != b.dtype {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitwise tensor operands must share dtype",
                });
            }

            // Same shape: fast path
            if a.shape == b.shape {
                return eval_bitwise_tensor_same_shape(primitive, a, b);
            }

            // Different shapes: broadcast
            let out_shape =
                bitwise_broadcast_shape(&a.shape, &b.shape).ok_or(EvalError::ShapeMismatch {
                    primitive,
                    left: a.shape.clone(),
                    right: b.shape.clone(),
                })?;

            let out_count = out_shape.element_count().unwrap_or(0) as usize;
            let out_strides = bitwise_compute_strides(&out_shape.dims);
            let a_strides = bitwise_broadcast_strides(&a.shape, &out_shape);
            let b_strides = bitwise_broadcast_strides(&b.shape, &out_shape);

            match a.dtype {
                fj_core::DType::I64 => {
                    // Dense I64 broadcast: gather via the shared contiguous-inner
                    // traversal (no per-element flat→multi decode) straight from
                    // the i64 backings into a dense i64 output. `bitwise_broadcast_
                    // strides` is output-aligned (0 for broadcast dims), exactly
                    // what broadcast_visit_row_major expects, so the (a_idx, b_idx)
                    // sequence — and the result — is bit-for-bit identical.
                    if let (Some(av), Some(bv)) =
                        (a.elements.as_i64_slice(), b.elements.as_i64_slice())
                    {
                        let mut out = Vec::with_capacity(out_count);
                        crate::arithmetic::broadcast_visit_row_major(
                            &out_shape.dims,
                            &a_strides,
                            &b_strides,
                            |ai, bi| {
                                out.push(apply_bitwise_binary_i64(primitive, av[ai], bv[bi]));
                            },
                        );
                        return Ok(Value::Tensor(
                            TensorValue::new_i64_values(out_shape, out)
                                .map_err(EvalError::InvalidTensor)?,
                        ));
                    }
                    let mut elements = Vec::with_capacity(out_count);
                    for flat_idx in 0..out_count {
                        let multi = bitwise_flat_to_multi(flat_idx, &out_strides);
                        let a_idx = bitwise_broadcast_flat_index(&multi, &a_strides);
                        let b_idx = bitwise_broadcast_flat_index(&multi, &b_strides);

                        match (&a.elements[a_idx], &b.elements[b_idx]) {
                            (fj_core::Literal::I64(va), fj_core::Literal::I64(vb)) => {
                                elements.push(fj_core::Literal::I64(apply_bitwise_binary_i64(
                                    primitive, *va, *vb,
                                )));
                            }
                            _ => {
                                return Err(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "bitwise ops require integer tensors",
                                });
                            }
                        }
                    }
                    Ok(Value::Tensor(
                        TensorValue::new(fj_core::DType::I64, out_shape, elements)
                            .map_err(EvalError::InvalidTensor)?,
                    ))
                }
                fj_core::DType::I32 => {
                    // i32 broadcast bitwise (was unsupported). Same dense broadcast as
                    // I64 but applied at i32 width via apply_bitwise_binary_i32.
                    if let (Some(av), Some(bv)) =
                        (a.elements.as_i64_slice(), b.elements.as_i64_slice())
                    {
                        let mut out = Vec::with_capacity(out_count);
                        crate::arithmetic::broadcast_visit_row_major(
                            &out_shape.dims,
                            &a_strides,
                            &b_strides,
                            |ai, bi| {
                                out.push(apply_bitwise_binary_i32(primitive, av[ai], bv[bi]));
                            },
                        );
                        return Ok(Value::Tensor(
                            TensorValue::new_i32_values(out_shape, out)
                                .map_err(EvalError::InvalidTensor)?,
                        ));
                    }
                    let mut elements = Vec::with_capacity(out_count);
                    for flat_idx in 0..out_count {
                        let multi = bitwise_flat_to_multi(flat_idx, &out_strides);
                        let a_idx = bitwise_broadcast_flat_index(&multi, &a_strides);
                        let b_idx = bitwise_broadcast_flat_index(&multi, &b_strides);
                        let (fj_core::Literal::I64(va), fj_core::Literal::I64(vb)) =
                            (&a.elements[a_idx], &b.elements[b_idx])
                        else {
                            return Err(EvalError::TypeMismatch {
                                primitive,
                                detail: "bitwise ops require integer tensors",
                            });
                        };
                        elements.push(fj_core::Literal::I64(apply_bitwise_binary_i32(
                            primitive, *va, *vb,
                        )));
                    }
                    Ok(Value::Tensor(
                        TensorValue::new(fj_core::DType::I32, out_shape, elements)
                            .map_err(EvalError::InvalidTensor)?,
                    ))
                }
                fj_core::DType::U32 => {
                    // Dense u32 broadcast via the shared contiguous-inner traversal
                    // (same output-aligned strides as the I64/I32 arms → identical
                    // (a_idx, b_idx) sequence and result).
                    if let (Some(av), Some(bv)) =
                        (a.elements.as_u32_slice(), b.elements.as_u32_slice())
                    {
                        let mut out = Vec::with_capacity(out_count);
                        crate::arithmetic::broadcast_visit_row_major(
                            &out_shape.dims,
                            &a_strides,
                            &b_strides,
                            |ai, bi| {
                                out.push(apply_bitwise_binary_u32(primitive, av[ai], bv[bi]));
                            },
                        );
                        return Ok(Value::Tensor(
                            TensorValue::new_u32_values(out_shape, out)
                                .map_err(EvalError::InvalidTensor)?,
                        ));
                    }
                    let mut elements = Vec::with_capacity(out_count);
                    for flat_idx in 0..out_count {
                        let multi = bitwise_flat_to_multi(flat_idx, &out_strides);
                        let a_idx = bitwise_broadcast_flat_index(&multi, &a_strides);
                        let b_idx = bitwise_broadcast_flat_index(&multi, &b_strides);

                        match (&a.elements[a_idx], &b.elements[b_idx]) {
                            (fj_core::Literal::U32(va), fj_core::Literal::U32(vb)) => {
                                elements.push(fj_core::Literal::U32(apply_bitwise_binary_u32(
                                    primitive, *va, *vb,
                                )));
                            }
                            _ => {
                                return Err(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "bitwise ops require integer tensors",
                                });
                            }
                        }
                    }
                    Ok(Value::Tensor(
                        TensorValue::new(fj_core::DType::U32, out_shape, elements)
                            .map_err(EvalError::InvalidTensor)?,
                    ))
                }
                fj_core::DType::U64 => {
                    if let (Some(av), Some(bv)) =
                        (a.elements.as_u64_slice(), b.elements.as_u64_slice())
                    {
                        let mut out = Vec::with_capacity(out_count);
                        crate::arithmetic::broadcast_visit_row_major(
                            &out_shape.dims,
                            &a_strides,
                            &b_strides,
                            |ai, bi| {
                                out.push(apply_bitwise_binary_u64(primitive, av[ai], bv[bi]));
                            },
                        );
                        return Ok(Value::Tensor(
                            TensorValue::new_u64_values(out_shape, out)
                                .map_err(EvalError::InvalidTensor)?,
                        ));
                    }
                    let mut elements = Vec::with_capacity(out_count);
                    for flat_idx in 0..out_count {
                        let multi = bitwise_flat_to_multi(flat_idx, &out_strides);
                        let a_idx = bitwise_broadcast_flat_index(&multi, &a_strides);
                        let b_idx = bitwise_broadcast_flat_index(&multi, &b_strides);

                        match (&a.elements[a_idx], &b.elements[b_idx]) {
                            (fj_core::Literal::U64(va), fj_core::Literal::U64(vb)) => {
                                elements.push(fj_core::Literal::U64(apply_bitwise_binary_u64(
                                    primitive, *va, *vb,
                                )));
                            }
                            _ => {
                                return Err(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "bitwise ops require integer tensors",
                                });
                            }
                        }
                    }
                    Ok(Value::Tensor(
                        TensorValue::new(fj_core::DType::U64, out_shape, elements)
                            .map_err(EvalError::InvalidTensor)?,
                    ))
                }
                fj_core::DType::Bool => {
                    if !matches!(
                        primitive,
                        Primitive::BitwiseAnd | Primitive::BitwiseOr | Primitive::BitwiseXor
                    ) {
                        return Err(bool_shift_unsupported(primitive));
                    }
                    // Dense Bool broadcast via the shared contiguous-inner traversal.
                    if let (Some(av), Some(bv)) =
                        (a.elements.as_bool_slice(), b.elements.as_bool_slice())
                    {
                        let mut out = Vec::with_capacity(out_count);
                        crate::arithmetic::broadcast_visit_row_major(
                            &out_shape.dims,
                            &a_strides,
                            &b_strides,
                            |ai, bi| {
                                out.push(
                                    apply_bitwise_binary_bool(primitive, av[ai], bv[bi])
                                        .unwrap_or(false),
                                );
                            },
                        );
                        return Ok(Value::Tensor(
                            TensorValue::new_bool_values(out_shape, out)
                                .map_err(EvalError::InvalidTensor)?,
                        ));
                    }
                    let mut elements = Vec::with_capacity(out_count);
                    for flat_idx in 0..out_count {
                        let multi = bitwise_flat_to_multi(flat_idx, &out_strides);
                        let a_idx = bitwise_broadcast_flat_index(&multi, &a_strides);
                        let b_idx = bitwise_broadcast_flat_index(&multi, &b_strides);
                        let (fj_core::Literal::Bool(x), fj_core::Literal::Bool(y)) =
                            (&a.elements[a_idx], &b.elements[b_idx])
                        else {
                            return Err(EvalError::TypeMismatch {
                                primitive,
                                detail: "bitwise bool op requires bool tensors",
                            });
                        };
                        elements.push(fj_core::Literal::Bool(
                            apply_bitwise_binary_bool(primitive, *x, *y).unwrap_or(false),
                        ));
                    }
                    Ok(Value::Tensor(
                        TensorValue::new(fj_core::DType::Bool, out_shape, elements)
                            .map_err(EvalError::InvalidTensor)?,
                    ))
                }
                _ => Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitwise ops require integer types",
                }),
            }
        }
        _ => Err(EvalError::TypeMismatch {
            primitive,
            detail: "bitwise ops require integer types",
        }),
    }
}

fn eval_bitwise_tensor_same_shape(
    primitive: Primitive,
    a: &TensorValue,
    b: &TensorValue,
) -> Result<Value, EvalError> {
    match a.dtype {
        fj_core::DType::I64 => {
            // Dense I64 path: a contiguous `apply_bitwise_binary_i64` map straight
            // into a dense i64 backing, skipping the per-element Literal::I64 match
            // and 24-byte enum stride on both inputs and the output. Bit-for-bit
            // identical to the loop below (same op, same order). `as_i64_slice` is
            // `Some` only for dense I64 storage.
            if let (Some(av), Some(bv)) = (a.elements.as_i64_slice(), b.elements.as_i64_slice()) {
                let n = av.len();
                // Thread the dense bitwise map above the DRAM-bound gate (calloc'd output +
                // parallel page-fault); bit-identical (per-element apply_bitwise_binary_i64).
                let out: Vec<i64> = if n >= crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN
                    && crate::arithmetic::work_scaled_threads(n) > 1
                {
                    let mut o = vec![0i64; n];
                    crate::arithmetic::threaded_index_fill_into(
                        &mut o,
                        crate::arithmetic::work_scaled_threads(n),
                        |i| apply_bitwise_binary_i64(primitive, av[i], bv[i]),
                    );
                    o
                } else {
                    av.iter()
                        .zip(bv)
                        .map(|(&va, &vb)| apply_bitwise_binary_i64(primitive, va, vb))
                        .collect()
                };
                return Ok(Value::Tensor(
                    TensorValue::new_i64_values(a.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(a.elements.len());
            for (ea, eb) in a.elements.iter().zip(b.elements.iter()) {
                match (ea, eb) {
                    (fj_core::Literal::I64(va), fj_core::Literal::I64(vb)) => {
                        elements.push(fj_core::Literal::I64(apply_bitwise_binary_i64(
                            primitive, *va, *vb,
                        )));
                    }
                    _ => {
                        return Err(EvalError::TypeMismatch {
                            primitive,
                            detail: "bitwise ops require integer tensors",
                        });
                    }
                }
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::I64, a.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        fj_core::DType::I32 => {
            // i32 (JAX's default int) bitwise previously hit the `_ => Err` arm and was
            // UNSUPPORTED. i32 shares the i64 backing; apply at i32 width (see
            // apply_bitwise_binary_i32 — ShiftRightLogical zero-fills from bit 31).
            // Dense as_i64_slice path → dense I32 output; boxed Literal::I64 fallback.
            if let (Some(av), Some(bv)) = (a.elements.as_i64_slice(), b.elements.as_i64_slice()) {
                let n = av.len();
                let out: Vec<i64> = if n >= crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN
                    && crate::arithmetic::work_scaled_threads(n) > 1
                {
                    let mut o = vec![0i64; n];
                    crate::arithmetic::threaded_index_fill_into(
                        &mut o,
                        crate::arithmetic::work_scaled_threads(n),
                        |i| apply_bitwise_binary_i32(primitive, av[i], bv[i]),
                    );
                    o
                } else {
                    av.iter()
                        .zip(bv)
                        .map(|(&va, &vb)| apply_bitwise_binary_i32(primitive, va, vb))
                        .collect()
                };
                return Ok(Value::Tensor(
                    TensorValue::new_i32_values(a.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(a.elements.len());
            for (ea, eb) in a.elements.iter().zip(b.elements.iter()) {
                let (fj_core::Literal::I64(va), fj_core::Literal::I64(vb)) = (ea, eb) else {
                    return Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "bitwise ops require integer tensors",
                    });
                };
                elements.push(fj_core::Literal::I64(apply_bitwise_binary_i32(
                    primitive, *va, *vb,
                )));
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::I32, a.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        fj_core::DType::U32 => {
            // Dense u32 path: contiguous apply_bitwise_binary_u32 map straight into a
            // dense u32 backing (skips per-Literal::U32 match + 24-byte enum stride on
            // both inputs and output). Bit-identical to the per-Literal fallback below
            // (same op, same order). `as_u32_slice` is `Some` only for dense u32.
            if let (Some(av), Some(bv)) = (a.elements.as_u32_slice(), b.elements.as_u32_slice()) {
                let n = av.len();
                let out: Vec<u32> = if n >= crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN
                    && crate::arithmetic::work_scaled_threads(n) > 1
                {
                    let mut o = vec![0u32; n];
                    crate::arithmetic::threaded_index_fill_into(
                        &mut o,
                        crate::arithmetic::work_scaled_threads(n),
                        |i| apply_bitwise_binary_u32(primitive, av[i], bv[i]),
                    );
                    o
                } else {
                    av.iter()
                        .zip(bv)
                        .map(|(&va, &vb)| apply_bitwise_binary_u32(primitive, va, vb))
                        .collect()
                };
                return Ok(Value::Tensor(
                    TensorValue::new_u32_values(a.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(a.elements.len());
            for (ea, eb) in a.elements.iter().zip(b.elements.iter()) {
                match (ea, eb) {
                    (fj_core::Literal::U32(va), fj_core::Literal::U32(vb)) => {
                        elements.push(fj_core::Literal::U32(apply_bitwise_binary_u32(
                            primitive, *va, *vb,
                        )));
                    }
                    _ => {
                        return Err(EvalError::TypeMismatch {
                            primitive,
                            detail: "bitwise ops require integer tensors",
                        });
                    }
                }
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::U32, a.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        fj_core::DType::U64 => {
            // Dense u64 path (mirror of the u32 arm above).
            if let (Some(av), Some(bv)) = (a.elements.as_u64_slice(), b.elements.as_u64_slice()) {
                let n = av.len();
                let out: Vec<u64> = if n >= crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN
                    && crate::arithmetic::work_scaled_threads(n) > 1
                {
                    let mut o = vec![0u64; n];
                    crate::arithmetic::threaded_index_fill_into(
                        &mut o,
                        crate::arithmetic::work_scaled_threads(n),
                        |i| apply_bitwise_binary_u64(primitive, av[i], bv[i]),
                    );
                    o
                } else {
                    av.iter()
                        .zip(bv)
                        .map(|(&va, &vb)| apply_bitwise_binary_u64(primitive, va, vb))
                        .collect()
                };
                return Ok(Value::Tensor(
                    TensorValue::new_u64_values(a.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut elements = Vec::with_capacity(a.elements.len());
            for (ea, eb) in a.elements.iter().zip(b.elements.iter()) {
                match (ea, eb) {
                    (fj_core::Literal::U64(va), fj_core::Literal::U64(vb)) => {
                        elements.push(fj_core::Literal::U64(apply_bitwise_binary_u64(
                            primitive, *va, *vb,
                        )));
                    }
                    _ => {
                        return Err(EvalError::TypeMismatch {
                            primitive,
                            detail: "bitwise ops require integer tensors",
                        });
                    }
                }
            }
            Ok(Value::Tensor(
                TensorValue::new(fj_core::DType::U64, a.shape.clone(), elements)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        fj_core::DType::Bool => {
            // JAX lowers logical_and/or/xor and bool `&|^*` to lax.bitwise_* on
            // Bool, so accept Bool operands and emit a dense Bool result. Reads
            // the contiguous bool backing when available.
            if !matches!(
                primitive,
                Primitive::BitwiseAnd | Primitive::BitwiseOr | Primitive::BitwiseXor
            ) {
                return Err(bool_shift_unsupported(primitive));
            }
            // SWAR fast path: two bit-packed BoolWords operands — the common
            // mask-combine case (`(a<b) & (c<d)`, since f64 same-shape comparisons
            // emit BoolWords) — combine 64 bools per u64 word instead of per-element
            // (and BoolWords otherwise misses as_bool_slice → the per-Literal path).
            // Bit-identical: unused tail bits are canonicalized to 0 and &/|/^ of
            // canonical tails stay 0, plus from_bool_words re-canonicalizes. Requires
            // equal logical lengths (same shape ⇒ same word layout).
            if let (Some((aw, alen)), Some((bw, blen))) =
                (a.elements.as_bool_words(), b.elements.as_bool_words())
                && alen == blen
            {
                let words: Vec<u64> = aw
                    .iter()
                    .zip(bw)
                    .map(|(&x, &y)| match primitive {
                        Primitive::BitwiseAnd => x & y,
                        Primitive::BitwiseOr => x | y,
                        _ => x ^ y, // BitwiseXor (only and/or/xor reach here)
                    })
                    .collect();
                let buf = fj_core::LiteralBuffer::from_bool_words(words, alen).ok_or(
                    EvalError::TypeMismatch {
                        primitive,
                        detail: "bool word mask length mismatch",
                    },
                )?;
                return Ok(Value::Tensor(
                    TensorValue::new_with_literal_buffer(
                        fj_core::DType::Bool,
                        a.shape.clone(),
                        buf,
                    )
                    .map_err(EvalError::InvalidTensor)?,
                ));
            }
            if let (Some(av), Some(bv)) = (a.elements.as_bool_slice(), b.elements.as_bool_slice()) {
                let out: Vec<bool> = av
                    .iter()
                    .zip(bv)
                    .map(|(&x, &y)| apply_bitwise_binary_bool(primitive, x, y).unwrap_or(false))
                    .collect();
                return Ok(Value::Tensor(
                    TensorValue::new_bool_values(a.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            let mut out = Vec::with_capacity(a.elements.len());
            for (ea, eb) in a.elements.iter().zip(b.elements.iter()) {
                let (fj_core::Literal::Bool(x), fj_core::Literal::Bool(y)) = (ea, eb) else {
                    return Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "bitwise bool op requires bool tensors",
                    });
                };
                out.push(apply_bitwise_binary_bool(primitive, *x, *y).unwrap_or(false));
            }
            Ok(Value::Tensor(
                TensorValue::new_bool_values(a.shape.clone(), out)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
        _ => Err(EvalError::TypeMismatch {
            primitive,
            detail: "bitwise ops require integer types",
        }),
    }
}

/// Evaluate a unary bitwise operation on integer values.
fn apply_bitwise_unary_literal(
    primitive: Primitive,
    literal: fj_core::Literal,
) -> Option<fj_core::Literal> {
    apply_bitwise_unary_literal_with_dtype(primitive, literal, None)
}

/// Evaluate a unary bitwise operation with explicit dtype for proper I32 handling.
/// When dtype is I32, PopulationCount/CountLeadingZeros operate on the lower 32 bits.
fn apply_bitwise_unary_literal_with_dtype(
    primitive: Primitive,
    literal: fj_core::Literal,
    dtype: Option<fj_core::DType>,
) -> Option<fj_core::Literal> {
    match (primitive, literal) {
        (Primitive::BitwiseNot, fj_core::Literal::Bool(value)) => {
            // JAX lowers logical_not / bool `~` to lax.bitwise_not on Bool.
            Some(fj_core::Literal::Bool(!value))
        }
        (Primitive::BitwiseNot, fj_core::Literal::I64(value)) => {
            Some(fj_core::Literal::I64(!value))
        }
        (Primitive::BitwiseNot, fj_core::Literal::U32(value)) => {
            Some(fj_core::Literal::U32(!value))
        }
        (Primitive::BitwiseNot, fj_core::Literal::U64(value)) => {
            Some(fj_core::Literal::U64(!value))
        }
        (Primitive::PopulationCount, fj_core::Literal::I64(value)) => {
            let count = if dtype == Some(fj_core::DType::I32) {
                (value as i32).count_ones()
            } else {
                value.count_ones()
            };
            Some(fj_core::Literal::I64(i64::from(count)))
        }
        (Primitive::PopulationCount, fj_core::Literal::U32(value)) => {
            Some(fj_core::Literal::I64(i64::from(value.count_ones())))
        }
        (Primitive::PopulationCount, fj_core::Literal::U64(value)) => {
            Some(fj_core::Literal::I64(i64::from(value.count_ones())))
        }
        (Primitive::CountLeadingZeros, fj_core::Literal::I64(value)) => {
            let count = if dtype == Some(fj_core::DType::I32) {
                (value as i32).leading_zeros()
            } else {
                value.leading_zeros()
            };
            Some(fj_core::Literal::I64(i64::from(count)))
        }
        (Primitive::CountLeadingZeros, fj_core::Literal::U32(value)) => {
            Some(fj_core::Literal::I64(i64::from(value.leading_zeros())))
        }
        (Primitive::CountLeadingZeros, fj_core::Literal::U64(value)) => {
            Some(fj_core::Literal::I64(i64::from(value.leading_zeros())))
        }
        (Primitive::CountTrailingZeros, fj_core::Literal::I64(value)) => {
            let count = if dtype == Some(fj_core::DType::I32) {
                (value as i32).trailing_zeros()
            } else {
                value.trailing_zeros()
            };
            Some(fj_core::Literal::I64(i64::from(count)))
        }
        (Primitive::CountTrailingZeros, fj_core::Literal::U32(value)) => {
            Some(fj_core::Literal::I64(i64::from(value.trailing_zeros())))
        }
        (Primitive::CountTrailingZeros, fj_core::Literal::U64(value)) => {
            Some(fj_core::Literal::I64(i64::from(value.trailing_zeros())))
        }
        _ => None,
    }
}

fn unary_bitwise_output_dtype(primitive: Primitive, input_dtype: fj_core::DType) -> fj_core::DType {
    match primitive {
        Primitive::PopulationCount
        | Primitive::CountLeadingZeros
        | Primitive::CountTrailingZeros => fj_core::DType::I64,
        Primitive::BitwiseNot => input_dtype,
        _ => input_dtype,
    }
}

fn eval_bitwise_unary(primitive: Primitive, inputs: &[Value]) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }
    match &inputs[0] {
        Value::Scalar(literal) => {
            let out = apply_bitwise_unary_literal(primitive, *literal).ok_or(
                EvalError::TypeMismatch {
                    primitive,
                    detail: "bitwise ops require integer types",
                },
            )?;
            Ok(Value::Scalar(out))
        }
        Value::Tensor(t) => {
            // SWAR Bool BitwiseNot on a bit-packed BoolWords mask (e.g. `~(a<b)` on an
            // f64-comparison mask): flip 64 bools per u64 word. Bit-identical — `!word`
            // also flips the unused tail bits, but from_bool_words re-canonicalizes them
            // to 0. Otherwise a BoolWords `~mask` falls to the per-Literal loop below.
            if primitive == Primitive::BitwiseNot
                && t.dtype == fj_core::DType::Bool
                && let Some((words, len)) = t.elements.as_bool_words()
            {
                let flipped: Vec<u64> = words.iter().map(|&w| !w).collect();
                let buf = fj_core::LiteralBuffer::from_bool_words(flipped, len).ok_or(
                    EvalError::TypeMismatch {
                        primitive,
                        detail: "bool word mask length mismatch",
                    },
                )?;
                return Ok(Value::Tensor(
                    TensorValue::new_with_literal_buffer(
                        fj_core::DType::Bool,
                        t.shape.clone(),
                        buf,
                    )
                    .map_err(EvalError::InvalidTensor)?,
                ));
            }
            // Dense Bool BitwiseNot (= logical_not / `~mask`): read the contiguous
            // bool backing and emit a dense Bool result, avoiding 24-byte Literals.
            if primitive == Primitive::BitwiseNot
                && t.dtype == fj_core::DType::Bool
                && let Some(values) = t.elements.as_bool_slice()
            {
                let out: Vec<bool> = values.iter().map(|&v| !v).collect();
                return Ok(Value::Tensor(
                    TensorValue::new_bool_values(t.shape.clone(), out)
                        .map_err(EvalError::InvalidTensor)?,
                ));
            }
            // Dense + threaded integer BitwiseNot (`~x` on i64/i32/u32/u64): the boxed
            // per-Literal path below is slow AND single-thread; read the packed backing and
            // map `!v` (i32: sign-extend `!(v as i32)`) into dense storage across cores. JAX
            // is slow here (~24ms). Bit-identical to apply_bitwise_unary_literal (`!v`).
            if primitive == Primitive::BitwiseNot {
                if t.dtype == fj_core::DType::I64
                    && let Some(src) = t.elements.as_i64_slice()
                {
                    let out = crate::arithmetic::threaded_unary_typed_map(src, &|v: i64| !v);
                    return Ok(Value::Tensor(
                        TensorValue::new_i64_values(t.shape.clone(), out)
                            .map_err(EvalError::InvalidTensor)?,
                    ));
                }
                if t.dtype == fj_core::DType::I32
                    && let Some(src) = t.elements.as_i64_slice()
                {
                    let out = crate::arithmetic::threaded_unary_typed_map(src, &|v: i64| {
                        i64::from(!(v as i32))
                    });
                    return Ok(Value::Tensor(
                        TensorValue::new_i32_values(t.shape.clone(), out)
                            .map_err(EvalError::InvalidTensor)?,
                    ));
                }
                if let Some(src) = t.elements.as_u32_slice() {
                    let out = crate::arithmetic::threaded_unary_typed_map(src, &|v: u32| !v);
                    return Ok(Value::Tensor(
                        TensorValue::new_u32_values(t.shape.clone(), out)
                            .map_err(EvalError::InvalidTensor)?,
                    ));
                }
                if let Some(src) = t.elements.as_u64_slice() {
                    let out = crate::arithmetic::threaded_unary_typed_map(src, &|v: u64| !v);
                    return Ok(Value::Tensor(
                        TensorValue::new_u64_values(t.shape.clone(), out)
                            .map_err(EvalError::InvalidTensor)?,
                    ));
                }
            }
            // Dense threaded fast path for the bit-count ops (PopulationCount/CLZ/CTZ): read the
            // typed integer backing and map the count intrinsic into a dense I64 output, escaping
            // the per-`Literal` generic loop below (24-byte Literal + enum dispatch per element).
            // Bit-identical to `apply_bitwise_unary_literal_with_dtype` — same per-dtype bit width
            // (I32 operates on the low 32 bits) and the same `i64::from(count)` result.
            if matches!(
                primitive,
                Primitive::PopulationCount
                    | Primitive::CountLeadingZeros
                    | Primitive::CountTrailingZeros
            ) {
                use crate::arithmetic::threaded_unary_typed_map_to as tmap;
                // Gated to I64: its de-box win is robust (1.9-5.6x vs the per-Literal loop). The
                // cheaper-payload unsigned/i32 cases were MEASURED to reverse under load (U32 0.80x)
                // because the threaded map's contention can outweigh the small de-box gain there —
                // they keep the generic path.
                let counts: Option<Vec<i64>> = match t.dtype {
                    fj_core::DType::I64 => t.elements.as_i64_slice().map(|s| match primitive {
                        Primitive::PopulationCount => tmap(s, &|v: i64| i64::from(v.count_ones())),
                        Primitive::CountLeadingZeros => {
                            tmap(s, &|v: i64| i64::from(v.leading_zeros()))
                        }
                        _ => tmap(s, &|v: i64| i64::from(v.trailing_zeros())),
                    }),
                    _ => None,
                };
                if let Some(out) = counts {
                    return Ok(Value::Tensor(
                        TensorValue::new_i64_values(t.shape.clone(), out)
                            .map_err(EvalError::InvalidTensor)?,
                    ));
                }
            }
            let out_dtype = unary_bitwise_output_dtype(primitive, t.dtype);
            let elements: Result<Vec<_>, _> = t
                .elements
                .iter()
                .map(|e| {
                    apply_bitwise_unary_literal_with_dtype(primitive, *e, Some(t.dtype)).ok_or(
                        EvalError::TypeMismatch {
                            primitive,
                            detail: "bitwise ops require integer types",
                        },
                    )
                })
                .collect();
            Ok(Value::Tensor(
                TensorValue::new(out_dtype, t.shape.clone(), elements?)
                    .map_err(EvalError::InvalidTensor)?,
            ))
        }
    }
}

fn parse_reduce_window_param(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
    key: &str,
    rank: usize,
    default: usize,
) -> Result<Vec<usize>, EvalError> {
    let values = if let Some(raw) = params.get(key) {
        if raw.trim().is_empty() {
            if rank == 0 {
                Vec::new()
            } else {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("{key} must match tensor rank {rank}"),
                });
            }
        } else {
            raw.split(',')
                .enumerate()
                .map(|(idx, item)| {
                    let trimmed = item.trim();
                    let value = trimmed
                        .parse::<usize>()
                        .map_err(|_| EvalError::Unsupported {
                            primitive,
                            detail: format!("invalid {key}[{idx}]: '{trimmed}'"),
                        })?;
                    if value == 0 {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: format!("{key}[{idx}] must be positive"),
                        });
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    } else {
        vec![default; rank]
    };

    if values.len() != rank {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "{key} length {} doesn't match tensor rank {rank}",
                values.len()
            ),
        });
    }

    Ok(values)
}

fn reduce_window_same_geometry(
    primitive: Primitive,
    input_dim: usize,
    window_dim: usize,
    stride: usize,
    lower: bool,
) -> Result<(usize, usize), EvalError> {
    let out_dim = input_dim.div_ceil(stride);
    if out_dim == 0 {
        return Ok((0, 0));
    }

    let covered = (out_dim - 1)
        .checked_mul(stride)
        .and_then(|base| base.checked_add(window_dim))
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "reduce_window SAME padding geometry overflow".to_owned(),
        })?;
    let total_padding = covered.saturating_sub(input_dim);
    let pad_low = if lower {
        total_padding.div_ceil(2)
    } else {
        total_padding / 2
    };
    Ok((out_dim, pad_low))
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ReduceWindowPadding {
    Valid,
    Same,
    SameLower,
    Explicit(Vec<(usize, usize)>),
}

fn parse_reduce_window_padding(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
) -> Result<ReduceWindowPadding, EvalError> {
    let Some(raw) = params.get("padding") else {
        return Ok(ReduceWindowPadding::Valid);
    };
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("VALID") {
        Ok(ReduceWindowPadding::Valid)
    } else if trimmed.eq_ignore_ascii_case("SAME") {
        Ok(ReduceWindowPadding::Same)
    } else if trimmed.eq_ignore_ascii_case("SAME_LOWER") {
        Ok(ReduceWindowPadding::SameLower)
    } else {
        parse_reduce_window_explicit_padding(primitive, raw).map(ReduceWindowPadding::Explicit)
    }
}

fn parse_reduce_window_explicit_padding(
    primitive: Primitive,
    raw: &str,
) -> Result<Vec<(usize, usize)>, EvalError> {
    let trimmed = raw
        .trim()
        .strip_prefix("EXPLICIT:")
        .or_else(|| raw.trim().strip_prefix("explicit:"))
        .unwrap_or(raw.trim());
    let numbers: Result<Vec<isize>, _> = trimmed
        .split(|ch: char| !(ch.is_ascii_digit() || ch == '-'))
        .filter(|part| !part.is_empty())
        .map(str::parse::<isize>)
        .collect();
    let numbers = numbers.map_err(|_| EvalError::Unsupported {
        primitive,
        detail: format!("unsupported reduce_window padding mode {raw:?}"),
    })?;
    if numbers.is_empty() || !numbers.len().is_multiple_of(2) {
        Err(EvalError::Unsupported {
            primitive,
            detail: format!("unsupported reduce_window padding mode {raw:?}"),
        })
    } else {
        let mut pairs = Vec::with_capacity(numbers.len() / 2);
        for pair in numbers.chunks_exact(2) {
            let low = usize::try_from(pair[0]).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("reduce_window explicit padding must be non-negative: {raw:?}"),
            })?;
            let high = usize::try_from(pair[1]).map_err(|_| EvalError::Unsupported {
                primitive,
                detail: format!("reduce_window explicit padding must be non-negative: {raw:?}"),
            })?;
            pairs.push((low, high));
        }
        Ok(pairs)
    }
}

fn reduce_window_explicit_geometry(
    primitive: Primitive,
    input_dim: usize,
    window_dim: usize,
    stride: usize,
    pad_low: usize,
    pad_high: usize,
) -> Result<(usize, usize), EvalError> {
    let padded = input_dim
        .checked_add(pad_low)
        .and_then(|value| value.checked_add(pad_high))
        .ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "reduce_window explicit padded input size overflow".to_owned(),
        })?;
    let out_dim = if padded >= window_dim {
        (padded - window_dim) / stride + 1
    } else {
        0
    };
    Ok((out_dim, pad_low))
}

/// ReduceWindow preserves the input dtype across all variants — every arm of
/// the prior exhaustive match returned `input_dtype` (including the Bool arm
/// returning `DType::Bool == input_dtype`), so it was effectively the
/// identity. The function is retained as a named helper so call sites read
/// like a contract ("this primitive preserves dtype") rather than passing the
/// raw input field through.
const fn reduce_window_output_dtype(input_dtype: fj_core::DType) -> fj_core::DType {
    input_dtype
}

fn reduce_window_literal_from_f64(dtype: fj_core::DType, value: f64) -> fj_core::Literal {
    match dtype {
        fj_core::DType::BF16 => fj_core::Literal::from_bf16_f64(value),
        fj_core::DType::F16 => fj_core::Literal::from_f16_f64(value),
        fj_core::DType::F32 => fj_core::Literal::from_f32(value as f32),
        _ => fj_core::Literal::from_f64(value),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ReduceWindowAccumulator {
    I64(i64),
    U32(u32),
    U64(u64),
    Bool(bool),
    Complex(f64, f64),
    F64(f64),
}

fn reduce_window_complex_initial(reduce_op: &str) -> (f64, f64) {
    match reduce_op {
        "max" => (f64::NEG_INFINITY, f64::NEG_INFINITY),
        "min" => (f64::INFINITY, f64::INFINITY),
        _ => (0.0, 0.0),
    }
}

fn reduce_window_complex_ge(lhs: (f64, f64), rhs: (f64, f64)) -> bool {
    lhs.0 > rhs.0 || (lhs.0 == rhs.0 && lhs.1 >= rhs.1)
}

fn reduce_window_complex_parts(
    primitive: Primitive,
    literal: fj_core::Literal,
) -> Result<(f64, f64), EvalError> {
    match literal {
        fj_core::Literal::Complex64Bits(re, im) => {
            Ok((f32::from_bits(re) as f64, f32::from_bits(im) as f64))
        }
        fj_core::Literal::Complex128Bits(re, im) => Ok((f64::from_bits(re), f64::from_bits(im))),
        _ => Err(EvalError::TypeMismatch {
            primitive,
            detail: "reduce_window complex tensors require complex literals",
        }),
    }
}

fn reduce_window_initial_accumulator(
    output_dtype: fj_core::DType,
    reduce_op: &str,
) -> ReduceWindowAccumulator {
    match output_dtype {
        fj_core::DType::I32 | fj_core::DType::I64 => {
            ReduceWindowAccumulator::I64(match reduce_op {
                "max" => i64::MIN,
                "min" => i64::MAX,
                _ => 0,
            })
        }
        fj_core::DType::U32 => ReduceWindowAccumulator::U32(match reduce_op {
            "max" => u32::MIN,
            "min" => u32::MAX,
            _ => 0,
        }),
        fj_core::DType::U64 => ReduceWindowAccumulator::U64(match reduce_op {
            "max" => u64::MIN,
            "min" => u64::MAX,
            _ => 0,
        }),
        fj_core::DType::Bool => ReduceWindowAccumulator::Bool(matches!(reduce_op, "min")),
        fj_core::DType::Complex64 | fj_core::DType::Complex128 => {
            let (re, im) = reduce_window_complex_initial(reduce_op);
            ReduceWindowAccumulator::Complex(re, im)
        }
        _ => ReduceWindowAccumulator::F64(match reduce_op {
            "max" => f64::NEG_INFINITY,
            "min" => f64::INFINITY,
            _ => 0.0,
        }),
    }
}

fn reduce_window_accumulate_literal(
    primitive: Primitive,
    reduce_op: &str,
    accumulator: &mut ReduceWindowAccumulator,
    literal: fj_core::Literal,
) -> Result<(), EvalError> {
    match accumulator {
        ReduceWindowAccumulator::I64(value) => {
            let fj_core::Literal::I64(input) = literal else {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "reduce_window I64 tensors require I64 literals",
                });
            };
            match reduce_op {
                "max" => *value = (*value).max(input),
                "min" => *value = (*value).min(input),
                _ => *value = value.wrapping_add(input),
            }
        }
        ReduceWindowAccumulator::U32(value) => {
            let fj_core::Literal::U32(input) = literal else {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "reduce_window U32 tensors require U32 literals",
                });
            };
            match reduce_op {
                "max" => *value = (*value).max(input),
                "min" => *value = (*value).min(input),
                _ => *value = value.wrapping_add(input),
            }
        }
        ReduceWindowAccumulator::U64(value) => {
            let fj_core::Literal::U64(input) = literal else {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "reduce_window U64 tensors require U64 literals",
                });
            };
            match reduce_op {
                "max" => *value = (*value).max(input),
                "min" => *value = (*value).min(input),
                _ => *value = value.wrapping_add(input),
            }
        }
        ReduceWindowAccumulator::Bool(value) => {
            let fj_core::Literal::Bool(input) = literal else {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "reduce_window Bool tensors require Bool literals",
                });
            };
            match reduce_op {
                "min" => *value &= input,
                _ => *value |= input,
            }
        }
        ReduceWindowAccumulator::Complex(re, im) => {
            let input = reduce_window_complex_parts(primitive, literal)?;
            match reduce_op {
                "max" => {
                    if reduce_window_complex_ge(input, (*re, *im)) {
                        *re = input.0;
                        *im = input.1;
                    }
                }
                "min" => {
                    if !reduce_window_complex_ge(input, (*re, *im)) {
                        *re = input.0;
                        *im = input.1;
                    }
                }
                _ => {
                    *re += input.0;
                    *im += input.1;
                }
            }
        }
        ReduceWindowAccumulator::F64(value) => {
            let input = literal.as_f64().unwrap_or(0.0);
            match reduce_op {
                "max" => *value = jax_max_f64(*value, input),
                "min" => *value = jax_min_f64(*value, input),
                _ => *value += input,
            }
        }
    }
    Ok(())
}

fn reduce_window_accumulator_literal(
    primitive: Primitive,
    output_dtype: fj_core::DType,
    accumulator: ReduceWindowAccumulator,
) -> Result<fj_core::Literal, EvalError> {
    match (output_dtype, accumulator) {
        (fj_core::DType::I32 | fj_core::DType::I64, ReduceWindowAccumulator::I64(value)) => {
            Ok(fj_core::Literal::I64(value))
        }
        (fj_core::DType::U32, ReduceWindowAccumulator::U32(value)) => {
            Ok(fj_core::Literal::U32(value))
        }
        (fj_core::DType::U64, ReduceWindowAccumulator::U64(value)) => {
            Ok(fj_core::Literal::U64(value))
        }
        (fj_core::DType::Bool, ReduceWindowAccumulator::Bool(value)) => {
            Ok(fj_core::Literal::Bool(value))
        }
        (fj_core::DType::Complex64, ReduceWindowAccumulator::Complex(re, im)) => {
            Ok(fj_core::Literal::from_complex64(re as f32, im as f32))
        }
        (fj_core::DType::Complex128, ReduceWindowAccumulator::Complex(re, im)) => {
            Ok(fj_core::Literal::from_complex128(re, im))
        }
        (_, ReduceWindowAccumulator::F64(value)) => {
            Ok(reduce_window_literal_from_f64(output_dtype, value))
        }
        _ => Err(EvalError::Unsupported {
            primitive,
            detail: "reduce_window accumulator/output dtype mismatch".to_owned(),
        }),
    }
}

fn reduce_window_sum_like(reduce_op: &str) -> bool {
    !matches!(reduce_op, "max" | "min")
}

fn reduce_window_unsupported(primitive: Primitive, detail: &'static str) -> EvalError {
    EvalError::Unsupported {
        primitive,
        detail: detail.to_owned(),
    }
}

#[inline]
fn reduce_window_f64_sum_value(literal: Literal) -> f64 {
    literal.as_f64().unwrap_or(0.0)
}

#[inline]
fn reduce_window_rank2_f64_sum_3x3_border(
    tensor: &TensorValue,
    input_rows: usize,
    input_cols: usize,
    out_row: usize,
    out_col: usize,
) -> f64 {
    // 3x3 'same' window (pad 1) covers input positions [out-1, out, out+1].
    // Out-of-bounds (padding) taps contribute 0.0 — XLA pads reduce_window with
    // the reduction init (0.0 for sum) and reduces the FULL window, so the pad
    // taps must be summed (a no-op for finite data, but it fixes the signed-zero
    // case where a -0.0 partial sum must land +0.0). Full-window row-major order
    // keeps this bit-identical to the generic loop and to valid-on-zero-padded.
    let mut accum = 0.0;
    for dr in 0..3usize {
        let in_row = out_row as isize + dr as isize - 1;
        let row_oob = in_row < 0 || (in_row as usize) >= input_rows;
        let row_offset = if row_oob {
            0
        } else {
            (in_row as usize) * input_cols
        };
        for dc in 0..3usize {
            let in_col = out_col as isize + dc as isize - 1;
            let v = if row_oob || in_col < 0 || (in_col as usize) >= input_cols {
                0.0
            } else {
                reduce_window_f64_sum_value(tensor.elements[row_offset + in_col as usize])
            };
            accum += v;
        }
    }
    accum
}

fn reduce_window_f64_values(tensor: &TensorValue) -> Option<Cow<'_, [f64]>> {
    if let Some(values) = tensor.elements.as_f64_slice() {
        return Some(Cow::Borrowed(values));
    }

    let mut values = Vec::with_capacity(tensor.elements.len());
    for literal in tensor.elements.iter().copied() {
        let Literal::F64Bits(bits) = literal else {
            return None;
        };
        values.push(f64::from_bits(bits));
    }
    Some(Cow::Owned(values))
}

#[inline]
fn reduce_window_rank2_f64_sum_3x3_border_values(
    values: &[f64],
    input_rows: usize,
    input_cols: usize,
    out_row: usize,
    out_col: usize,
) -> f64 {
    // See reduce_window_rank2_f64_sum_3x3_border: OOB padding taps add 0.0
    // (XLA zero-init padding), iterated full-window row-major.
    let mut accum = 0.0;
    for dr in 0..3usize {
        let in_row = out_row as isize + dr as isize - 1;
        let row_oob = in_row < 0 || (in_row as usize) >= input_rows;
        let row_offset = if row_oob {
            0
        } else {
            (in_row as usize) * input_cols
        };
        for dc in 0..3usize {
            let in_col = out_col as isize + dc as isize - 1;
            let v = if row_oob || in_col < 0 || (in_col as usize) >= input_cols {
                0.0
            } else {
                values[row_offset + in_col as usize]
            };
            accum += v;
        }
    }
    accum
}

fn eval_reduce_window_rank2_f64_sum_3x3_same_values(
    values: &[f64],
    tensor: &TensorValue,
    out_dims: &[u32],
    total_output: usize,
) -> Result<Value, EvalError> {
    let input_rows = tensor.shape.dims[0] as usize;
    let input_cols = tensor.shape.dims[1] as usize;

    let mut output_values = Vec::with_capacity(total_output);
    if input_rows <= 2 || input_cols <= 2 {
        for out_row in 0..input_rows {
            for out_col in 0..input_cols {
                output_values.push(reduce_window_rank2_f64_sum_3x3_border_values(
                    values, input_rows, input_cols, out_row, out_col,
                ));
            }
        }
    } else {
        for out_col in 0..input_cols {
            output_values.push(reduce_window_rank2_f64_sum_3x3_border_values(
                values, input_rows, input_cols, 0, out_col,
            ));
        }

        for out_row in 1..(input_rows - 1) {
            output_values.push(reduce_window_rank2_f64_sum_3x3_border_values(
                values, input_rows, input_cols, out_row, 0,
            ));

            let top_row_offset = (out_row - 1) * input_cols;
            let center_row_offset = out_row * input_cols;
            let bottom_row_offset = (out_row + 1) * input_cols;
            for out_col in 1..(input_cols - 1) {
                let left_col = out_col - 1;
                let top_left = top_row_offset + left_col;
                let center_left = center_row_offset + left_col;
                let bottom_left = bottom_row_offset + left_col;
                let mut accum = 0.0;
                accum += values[top_left];
                accum += values[top_left + 1];
                accum += values[top_left + 2];
                accum += values[center_left];
                accum += values[center_left + 1];
                accum += values[center_left + 2];
                accum += values[bottom_left];
                accum += values[bottom_left + 1];
                accum += values[bottom_left + 2];
                output_values.push(accum);
            }

            output_values.push(reduce_window_rank2_f64_sum_3x3_border_values(
                values,
                input_rows,
                input_cols,
                out_row,
                input_cols - 1,
            ));
        }

        for out_col in 0..input_cols {
            output_values.push(reduce_window_rank2_f64_sum_3x3_border_values(
                values,
                input_rows,
                input_cols,
                input_rows - 1,
                out_col,
            ));
        }
    }

    Ok(Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: out_dims.to_vec(),
            },
            output_values,
        )
        .map_err(EvalError::InvalidTensor)?,
    ))
}

fn eval_reduce_window_rank2_f64_sum_3x3_same(
    tensor: &TensorValue,
    out_dims: &[u32],
    total_output: usize,
) -> Result<Value, EvalError> {
    if let Some(values) = reduce_window_f64_values(tensor) {
        return eval_reduce_window_rank2_f64_sum_3x3_same_values(
            values.as_ref(),
            tensor,
            out_dims,
            total_output,
        );
    }

    let input_rows = tensor.shape.dims[0] as usize;
    let input_cols = tensor.shape.dims[1] as usize;
    let elements = &tensor.elements;

    let mut output_elements = Vec::with_capacity(total_output);
    if input_rows <= 2 || input_cols <= 2 {
        for out_row in 0..input_rows {
            for out_col in 0..input_cols {
                output_elements.push(Literal::from_f64(reduce_window_rank2_f64_sum_3x3_border(
                    tensor, input_rows, input_cols, out_row, out_col,
                )));
            }
        }
    } else {
        for out_col in 0..input_cols {
            output_elements.push(Literal::from_f64(reduce_window_rank2_f64_sum_3x3_border(
                tensor, input_rows, input_cols, 0, out_col,
            )));
        }

        for out_row in 1..(input_rows - 1) {
            output_elements.push(Literal::from_f64(reduce_window_rank2_f64_sum_3x3_border(
                tensor, input_rows, input_cols, out_row, 0,
            )));

            let top_row_offset = (out_row - 1) * input_cols;
            let center_row_offset = out_row * input_cols;
            let bottom_row_offset = (out_row + 1) * input_cols;
            for out_col in 1..(input_cols - 1) {
                let left_col = out_col - 1;
                let top_left = top_row_offset + left_col;
                let center_left = center_row_offset + left_col;
                let bottom_left = bottom_row_offset + left_col;
                let mut accum = 0.0;
                accum += reduce_window_f64_sum_value(elements[top_left]);
                accum += reduce_window_f64_sum_value(elements[top_left + 1]);
                accum += reduce_window_f64_sum_value(elements[top_left + 2]);
                accum += reduce_window_f64_sum_value(elements[center_left]);
                accum += reduce_window_f64_sum_value(elements[center_left + 1]);
                accum += reduce_window_f64_sum_value(elements[center_left + 2]);
                accum += reduce_window_f64_sum_value(elements[bottom_left]);
                accum += reduce_window_f64_sum_value(elements[bottom_left + 1]);
                accum += reduce_window_f64_sum_value(elements[bottom_left + 2]);
                output_elements.push(Literal::from_f64(accum));
            }

            output_elements.push(Literal::from_f64(reduce_window_rank2_f64_sum_3x3_border(
                tensor,
                input_rows,
                input_cols,
                out_row,
                input_cols - 1,
            )));
        }

        for out_col in 0..input_cols {
            output_elements.push(Literal::from_f64(reduce_window_rank2_f64_sum_3x3_border(
                tensor,
                input_rows,
                input_cols,
                input_rows - 1,
                out_col,
            )));
        }
    }

    Ok(Value::Tensor(
        TensorValue::new(
            tensor.dtype,
            Shape {
                dims: out_dims.to_vec(),
            },
            output_elements,
        )
        .map_err(EvalError::InvalidTensor)?,
    ))
}

// Separable O(input) running-sum for LARGE-window rank-2 f64 SUM pooling (stride 1, finite inputs).
// Returns None when not applicable -> caller falls back to the naive O(out*wr*wc) fold. SUM is separable,
// so two running-sum passes replace the nested per-window fold (177-363x slower than JAX at 11/31). It
// materializes the zero-padded input for pad>0 (matching naive OOB=0), so same-padding and valid-on-padded
// produce the SAME padded array -> bit-identical to each other (preserving the metamorphic invariant). The
// running subtract-old would do inf-inf=NaN, hence the finite-input guard (falls back to naive on inf/nan).
// Reassociates vs the flat fold (tolerance ~1e-13 f64); no test compares large-window sum-pool bit-exactly
// to the naive — only the same-vs-valid metamorphic equality, which this preserves.
#[allow(clippy::too_many_arguments)]
fn separable_reduce_window_sum_f64(
    tensor: &TensorValue,
    input_rows: usize,
    input_cols: usize,
    out_rows: usize,
    out_cols: usize,
    window_rows: usize,
    window_cols: usize,
    stride_rows: usize,
    stride_cols: usize,
    pad_rows: usize,
    pad_cols: usize,
) -> Option<Vec<f64>> {
    if stride_rows != 1
        || stride_cols != 1
        || window_rows < 2
        || window_cols < 2
        || window_rows.saturating_mul(window_cols) < 25
    {
        return None;
    }
    let padded_rows = out_rows + window_rows - 1;
    let padded_cols = out_cols + window_cols - 1;
    if pad_rows + input_rows > padded_rows || pad_cols + input_cols > padded_cols {
        return None;
    }
    let src = tensor.elements.as_f64_slice()?;
    if src.len() != input_rows.saturating_mul(input_cols) || !src.iter().all(|v| v.is_finite()) {
        return None;
    }
    // VALID (pad 0) reads `src` directly; padded cases materialize a zero-bordered copy.
    let owned_padded: Vec<f64>;
    let (data, d_cols): (&[f64], usize) = if pad_rows == 0 && pad_cols == 0 {
        (src, input_cols)
    } else {
        let mut padded = vec![0.0f64; padded_rows * padded_cols];
        for r in 0..input_rows {
            let dst_base = (pad_rows + r) * padded_cols + pad_cols;
            padded[dst_base..dst_base + input_cols]
                .copy_from_slice(&src[r * input_cols..(r + 1) * input_cols]);
        }
        owned_padded = padded;
        (&owned_padded, padded_cols)
    };
    let d_rows = padded_rows;
    // Phase 1: per-row horizontal running window-sum -> hsum[d_rows * out_cols].
    let mut hsum = vec![0.0f64; d_rows * out_cols];
    for r in 0..d_rows {
        let row = &data[r * d_cols..(r + 1) * d_cols];
        let hr = &mut hsum[r * out_cols..(r + 1) * out_cols];
        let mut s = 0.0;
        for &v in &row[0..window_cols] {
            s += v;
        }
        hr[0] = s;
        for oc in 1..out_cols {
            s += row[oc + window_cols - 1] - row[oc - 1];
            hr[oc] = s;
        }
    }
    // Phase 2: vertical cols-wide running window-sum over rows -> output[out_rows * out_cols].
    let mut output = vec![0.0f64; out_rows * out_cols];
    let mut vsum = vec![0.0f64; out_cols];
    for wr in 0..window_rows {
        let hrow = &hsum[wr * out_cols..(wr + 1) * out_cols];
        for (v, &h) in vsum.iter_mut().zip(hrow) {
            *v += h;
        }
    }
    output[0..out_cols].copy_from_slice(&vsum);
    for or in 1..out_rows {
        let add = &hsum[(or + window_rows - 1) * out_cols..(or + window_rows) * out_cols];
        let sub = &hsum[(or - 1) * out_cols..or * out_cols];
        for ((v, &a), &s) in vsum.iter_mut().zip(add).zip(sub) {
            *v += a - s;
        }
        output[or * out_cols..(or + 1) * out_cols].copy_from_slice(&vsum);
    }
    Some(output)
}

// f32 sibling of `separable_reduce_window_sum_f64` (f32 is JAX's default dtype). Matches the naive f32 sum
// contract (widen each tap to f64, accumulate in f64, round each output to f32) — the running sum stays in
// f64 and only the output is narrowed, so it is tolerance-close to the naive fold. Same padded-input
// materialization + finite-input guard, so the same-vs-valid metamorphic invariant holds.
#[allow(clippy::too_many_arguments)]
fn separable_reduce_window_sum_f32(
    tensor: &TensorValue,
    input_rows: usize,
    input_cols: usize,
    out_rows: usize,
    out_cols: usize,
    window_rows: usize,
    window_cols: usize,
    stride_rows: usize,
    stride_cols: usize,
    pad_rows: usize,
    pad_cols: usize,
) -> Option<Vec<f32>> {
    if stride_rows != 1
        || stride_cols != 1
        || window_rows < 2
        || window_cols < 2
        || window_rows.saturating_mul(window_cols) < 25
    {
        return None;
    }
    let padded_rows = out_rows + window_rows - 1;
    let padded_cols = out_cols + window_cols - 1;
    if pad_rows + input_rows > padded_rows || pad_cols + input_cols > padded_cols {
        return None;
    }
    let src = tensor.elements.as_f32_slice()?;
    if src.len() != input_rows.saturating_mul(input_cols) || !src.iter().all(|v| v.is_finite()) {
        return None;
    }
    let owned_padded: Vec<f32>;
    let (data, d_cols): (&[f32], usize) = if pad_rows == 0 && pad_cols == 0 {
        (src, input_cols)
    } else {
        let mut padded = vec![0.0f32; padded_rows * padded_cols];
        for r in 0..input_rows {
            let dst_base = (pad_rows + r) * padded_cols + pad_cols;
            padded[dst_base..dst_base + input_cols]
                .copy_from_slice(&src[r * input_cols..(r + 1) * input_cols]);
        }
        owned_padded = padded;
        (&owned_padded, padded_cols)
    };
    let d_rows = padded_rows;
    // Phase 1: per-row horizontal running window-sum in f64 -> hsum[d_rows * out_cols].
    let mut hsum = vec![0.0f64; d_rows * out_cols];
    for r in 0..d_rows {
        let row = &data[r * d_cols..(r + 1) * d_cols];
        let hr = &mut hsum[r * out_cols..(r + 1) * out_cols];
        let mut s = 0.0f64;
        for &v in &row[0..window_cols] {
            s += f64::from(v);
        }
        hr[0] = s;
        for oc in 1..out_cols {
            s += f64::from(row[oc + window_cols - 1]) - f64::from(row[oc - 1]);
            hr[oc] = s;
        }
    }
    // Phase 2: vertical cols-wide running window-sum (f64), narrow each output to f32.
    let mut output = vec![0.0f32; out_rows * out_cols];
    let mut vsum = vec![0.0f64; out_cols];
    for wr in 0..window_rows {
        let hrow = &hsum[wr * out_cols..(wr + 1) * out_cols];
        for (v, &h) in vsum.iter_mut().zip(hrow) {
            *v += h;
        }
    }
    for (o, &v) in output[0..out_cols].iter_mut().zip(&vsum) {
        *o = v as f32;
    }
    for or in 1..out_rows {
        let add = &hsum[(or + window_rows - 1) * out_cols..(or + window_rows) * out_cols];
        let sub = &hsum[(or - 1) * out_cols..or * out_cols];
        for ((v, &a), &s) in vsum.iter_mut().zip(add).zip(sub) {
            *v += a - s;
        }
        let orow = &mut output[or * out_cols..(or + 1) * out_cols];
        for (o, &v) in orow.iter_mut().zip(&vsum) {
            *o = v as f32;
        }
    }
    Some(output)
}

/// Rank-3 separable running-window-SUM for VALID (pad 0), unit-stride f64 sum pooling
/// (volumetric/3D `reduce_window` add). Replaces the naive O(out·∏window) tap loop in
/// [`reduce_window_rank3_f64_sum_xlane`] with three O(input) add-new-subtract-old running
/// sums — one per axis (x innermost, then y, then z), each phase reading the prior phase's
/// contiguous output. Op count drops from `∏window` taps/cell to ~3 adds/cell (win5³: 125→3,
/// win9³: 729→3). Tolerance-equal (not bit-identical) to the naive fold — the running-sum
/// reassociates the additions — so it is gated to the same all-finite inputs as the rank-2
/// separable (a NaN/±inf window would make the subtract-old diverge: ∞−∞ = NaN). Returns
/// `None` (fall back to the bit-identical xlane path) on non-finite input or a size mismatch.
fn separable_reduce_window_sum_3d_f64(
    src: &[f64],
    input_dims: [usize; 3],
    window_dims: [usize; 3],
    out_dims: [usize; 3],
) -> Option<Vec<f64>> {
    let [in_z, in_y, in_x] = input_dims;
    let [win_z, win_y, win_x] = window_dims;
    let [out_z, out_y, out_x] = out_dims;
    if src.len() != in_z.checked_mul(in_y)?.checked_mul(in_x)? {
        return None;
    }
    // VALID unit-stride geometry: out_ax = in_ax − win_ax + 1 (caller already checks the
    // halo fits, but re-derive defensively so the running-sum index math can't go OOB).
    if win_x == 0
        || win_y == 0
        || win_z == 0
        || out_x + win_x - 1 != in_x
        || out_y + win_y - 1 != in_y
        || out_z + win_z - 1 != in_z
    {
        return None;
    }
    if !src.iter().all(|v| v.is_finite()) {
        return None;
    }

    // Phase 1: horizontal (x) running window-sum -> hx[in_z, in_y, out_x].
    let zy_rows = in_z * in_y;
    let mut hx = vec![0.0f64; zy_rows * out_x];
    for zy in 0..zy_rows {
        let row = &src[zy * in_x..zy * in_x + in_x];
        let hr = &mut hx[zy * out_x..(zy + 1) * out_x];
        let mut s = 0.0;
        for &v in &row[0..win_x] {
            s += v;
        }
        hr[0] = s;
        for ox in 1..out_x {
            s += row[ox + win_x - 1] - row[ox - 1];
            hr[ox] = s;
        }
    }

    // Phase 2: vertical (y) running window-sum within each z-plane -> hy[in_z, out_y, out_x].
    let in_plane = in_y * out_x;
    let out_plane = out_y * out_x;
    let mut hy = vec![0.0f64; in_z * out_plane];
    let mut vsum = vec![0.0f64; out_x];
    for z in 0..in_z {
        let plane = &hx[z * in_plane..(z + 1) * in_plane];
        let outp = &mut hy[z * out_plane..(z + 1) * out_plane];
        for v in vsum.iter_mut() {
            *v = 0.0;
        }
        for wy in 0..win_y {
            for (v, &h) in vsum.iter_mut().zip(&plane[wy * out_x..(wy + 1) * out_x]) {
                *v += h;
            }
        }
        outp[0..out_x].copy_from_slice(&vsum);
        for oy in 1..out_y {
            let add = &plane[(oy + win_y - 1) * out_x..(oy + win_y) * out_x];
            let sub = &plane[(oy - 1) * out_x..oy * out_x];
            for ((v, &a), &s) in vsum.iter_mut().zip(add).zip(sub) {
                *v += a - s;
            }
            outp[oy * out_x..(oy + 1) * out_x].copy_from_slice(&vsum);
        }
    }

    // Phase 3: depth (z) running window-sum over planes -> output[out_z, out_y, out_x].
    let mut output = vec![0.0f64; out_z * out_plane];
    let mut zsum = vec![0.0f64; out_plane];
    for wz in 0..win_z {
        for (v, &h) in zsum
            .iter_mut()
            .zip(&hy[wz * out_plane..(wz + 1) * out_plane])
        {
            *v += h;
        }
    }
    output[0..out_plane].copy_from_slice(&zsum);
    for oz in 1..out_z {
        let add = &hy[(oz + win_z - 1) * out_plane..(oz + win_z) * out_plane];
        let sub = &hy[(oz - 1) * out_plane..oz * out_plane];
        for ((v, &a), &s) in zsum.iter_mut().zip(add).zip(sub) {
            *v += a - s;
        }
        output[oz * out_plane..(oz + 1) * out_plane].copy_from_slice(&zsum);
    }
    Some(output)
}

/// f32 sibling of [`separable_reduce_window_sum_3d_f64`] (f32 is JAX's default float, so f32
/// volumetric sum/avg pooling is the common signal case). The running sums stay in f64 and only
/// the final output is narrowed to f32 — matching the naive f32 sum contract (widen each tap to
/// f64, accumulate in f64, round each output to f32), so it is tolerance-close to the naive fold.
/// Same all-finite guard + VALID unit-stride geometry as the f64 version. f32 rank-3 volumetric
/// sum has NO dedicated kernel today (it falls to the generic `eval_reduce_window_dense_float`
/// per-tap loop), so the win is larger here than for f64.
fn separable_reduce_window_sum_3d_f32(
    src: &[f32],
    input_dims: [usize; 3],
    window_dims: [usize; 3],
    out_dims: [usize; 3],
) -> Option<Vec<f32>> {
    let [in_z, in_y, in_x] = input_dims;
    let [win_z, win_y, win_x] = window_dims;
    let [out_z, out_y, out_x] = out_dims;
    if src.len() != in_z.checked_mul(in_y)?.checked_mul(in_x)? {
        return None;
    }
    if win_x == 0
        || win_y == 0
        || win_z == 0
        || out_x + win_x - 1 != in_x
        || out_y + win_y - 1 != in_y
        || out_z + win_z - 1 != in_z
    {
        return None;
    }
    if !src.iter().all(|v| v.is_finite()) {
        return None;
    }

    // Phase 1: horizontal (x) running window-sum in f64 -> hx[in_z, in_y, out_x].
    let zy_rows = in_z * in_y;
    let mut hx = vec![0.0f64; zy_rows * out_x];
    for zy in 0..zy_rows {
        let row = &src[zy * in_x..zy * in_x + in_x];
        let hr = &mut hx[zy * out_x..(zy + 1) * out_x];
        let mut s = 0.0f64;
        for &v in &row[0..win_x] {
            s += f64::from(v);
        }
        hr[0] = s;
        for ox in 1..out_x {
            s += f64::from(row[ox + win_x - 1]) - f64::from(row[ox - 1]);
            hr[ox] = s;
        }
    }

    // Phase 2: vertical (y) running window-sum within each z-plane -> hy[in_z, out_y, out_x].
    let in_plane = in_y * out_x;
    let out_plane = out_y * out_x;
    let mut hy = vec![0.0f64; in_z * out_plane];
    let mut vsum = vec![0.0f64; out_x];
    for z in 0..in_z {
        let plane = &hx[z * in_plane..(z + 1) * in_plane];
        let outp = &mut hy[z * out_plane..(z + 1) * out_plane];
        for v in vsum.iter_mut() {
            *v = 0.0;
        }
        for wy in 0..win_y {
            for (v, &h) in vsum.iter_mut().zip(&plane[wy * out_x..(wy + 1) * out_x]) {
                *v += h;
            }
        }
        outp[0..out_x].copy_from_slice(&vsum);
        for oy in 1..out_y {
            let add = &plane[(oy + win_y - 1) * out_x..(oy + win_y) * out_x];
            let sub = &plane[(oy - 1) * out_x..oy * out_x];
            for ((v, &a), &s) in vsum.iter_mut().zip(add).zip(sub) {
                *v += a - s;
            }
            outp[oy * out_x..(oy + 1) * out_x].copy_from_slice(&vsum);
        }
    }

    // Phase 3: depth (z) running window-sum over planes, narrowing output to f32.
    let mut output = vec![0.0f32; out_z * out_plane];
    let mut zsum = vec![0.0f64; out_plane];
    for wz in 0..win_z {
        for (v, &h) in zsum
            .iter_mut()
            .zip(&hy[wz * out_plane..(wz + 1) * out_plane])
        {
            *v += h;
        }
    }
    for (o, &v) in output[0..out_plane].iter_mut().zip(&zsum) {
        *o = v as f32;
    }
    for oz in 1..out_z {
        let add = &hy[(oz + win_z - 1) * out_plane..(oz + win_z) * out_plane];
        let sub = &hy[(oz - 1) * out_plane..oz * out_plane];
        for ((v, &a), &s) in zsum.iter_mut().zip(add).zip(sub) {
            *v += a - s;
        }
        let orow = &mut output[oz * out_plane..(oz + 1) * out_plane];
        for (o, &v) in orow.iter_mut().zip(&zsum) {
            *o = v as f32;
        }
    }
    Some(output)
}

/// i64 sibling of [`separable_reduce_window_sum_3d_f64`] for VALID unit-stride integer 3D sum
/// pooling. Integer addition is a RING (associative + commutative even on overflow), so the
/// add-new-subtract-old running sum with `wrapping_add`/`wrapping_sub` is BIT-IDENTICAL to the
/// naive `wrapping_add` fold in [`eval_reduce_window_dense_i64`] — no tolerance, no finite guard
/// needed. Replaces the rank-3 naive O(out·∏window) loop (the SAT fast path is rank-2 only).
fn separable_reduce_window_sum_3d_i64(
    src: &[i64],
    input_dims: [usize; 3],
    window_dims: [usize; 3],
    out_dims: [usize; 3],
) -> Option<Vec<i64>> {
    let [in_z, in_y, in_x] = input_dims;
    let [win_z, win_y, win_x] = window_dims;
    let [out_z, out_y, out_x] = out_dims;
    if src.len() != in_z.checked_mul(in_y)?.checked_mul(in_x)? {
        return None;
    }
    if win_x == 0
        || win_y == 0
        || win_z == 0
        || out_x + win_x - 1 != in_x
        || out_y + win_y - 1 != in_y
        || out_z + win_z - 1 != in_z
    {
        return None;
    }

    // Phase 1: horizontal (x) running window-sum -> hx[in_z, in_y, out_x].
    let zy_rows = in_z * in_y;
    let mut hx = vec![0i64; zy_rows * out_x];
    for zy in 0..zy_rows {
        let row = &src[zy * in_x..zy * in_x + in_x];
        let hr = &mut hx[zy * out_x..(zy + 1) * out_x];
        let mut s = 0i64;
        for &v in &row[0..win_x] {
            s = s.wrapping_add(v);
        }
        hr[0] = s;
        for ox in 1..out_x {
            s = s
                .wrapping_add(row[ox + win_x - 1])
                .wrapping_sub(row[ox - 1]);
            hr[ox] = s;
        }
    }

    // Phase 2: vertical (y) running window-sum within each z-plane -> hy[in_z, out_y, out_x].
    let in_plane = in_y * out_x;
    let out_plane = out_y * out_x;
    let mut hy = vec![0i64; in_z * out_plane];
    let mut vsum = vec![0i64; out_x];
    for z in 0..in_z {
        let plane = &hx[z * in_plane..(z + 1) * in_plane];
        let outp = &mut hy[z * out_plane..(z + 1) * out_plane];
        for v in vsum.iter_mut() {
            *v = 0;
        }
        for wy in 0..win_y {
            for (v, &h) in vsum.iter_mut().zip(&plane[wy * out_x..(wy + 1) * out_x]) {
                *v = v.wrapping_add(h);
            }
        }
        outp[0..out_x].copy_from_slice(&vsum);
        for oy in 1..out_y {
            let add = &plane[(oy + win_y - 1) * out_x..(oy + win_y) * out_x];
            let sub = &plane[(oy - 1) * out_x..oy * out_x];
            for ((v, &a), &s) in vsum.iter_mut().zip(add).zip(sub) {
                *v = v.wrapping_add(a).wrapping_sub(s);
            }
            outp[oy * out_x..(oy + 1) * out_x].copy_from_slice(&vsum);
        }
    }

    // Phase 3: depth (z) running window-sum over planes -> output[out_z, out_y, out_x].
    let mut output = vec![0i64; out_z * out_plane];
    let mut zsum = vec![0i64; out_plane];
    for wz in 0..win_z {
        for (v, &h) in zsum
            .iter_mut()
            .zip(&hy[wz * out_plane..(wz + 1) * out_plane])
        {
            *v = v.wrapping_add(h);
        }
    }
    output[0..out_plane].copy_from_slice(&zsum);
    for oz in 1..out_z {
        let add = &hy[(oz + win_z - 1) * out_plane..(oz + win_z) * out_plane];
        let sub = &hy[(oz - 1) * out_plane..oz * out_plane];
        for ((v, &a), &s) in zsum.iter_mut().zip(add).zip(sub) {
            *v = v.wrapping_add(a).wrapping_sub(s);
        }
        output[oz * out_plane..(oz + 1) * out_plane].copy_from_slice(&zsum);
    }
    Some(output)
}

fn eval_reduce_window_rank2_f64_sum(
    primitive: Primitive,
    tensor: &TensorValue,
    window_dims: &[usize],
    strides: &[usize],
    out_dims: &[u32],
    pad_lows: &[usize],
    total_output: usize,
) -> Result<Value, EvalError> {
    let input_rows = tensor.shape.dims[0] as usize;
    let input_cols = tensor.shape.dims[1] as usize;
    let out_rows = out_dims[0] as usize;
    let out_cols = out_dims[1] as usize;
    let window_rows = window_dims[0];
    let window_cols = window_dims[1];
    let stride_rows = strides[0];
    let stride_cols = strides[1];
    let pad_rows = pad_lows[0];
    let pad_cols = pad_lows[1];

    if window_rows == 3
        && window_cols == 3
        && stride_rows == 1
        && stride_cols == 1
        && pad_rows == 1
        && pad_cols == 1
        && out_rows == input_rows
        && out_cols == input_cols
        && input_rows.checked_mul(input_cols) == Some(tensor.elements.len())
    {
        return eval_reduce_window_rank2_f64_sum_3x3_same(tensor, out_dims, total_output);
    }

    // Separable running-sum fast path for LARGE windows (stride 1, FINITE inputs): SUM is separable, so two
    // O(input) running-sum passes replace the naive O(out*wr*wc) fold (177-363x slower than JAX at 11/31).
    // It materializes the zero-padded input (matching the naive OOB=0 semantics for ANY pad_low), so
    // same-padding and valid-on-zero-padded produce the SAME padded array -> bit-identical to each other
    // (the metamorphic invariant holds within the separable). The running-sum subtract-old would do
    // inf-inf=NaN, so a finite-input guard falls back to the naive fold on inf/nan. The result reassociates
    // vs the flat fold (tolerance, ~1e-13 f64) but no test compares large-window sum-pool to a bit-exact
    // naive value — only the same-vs-valid metamorphic equality, which this preserves.
    if let Some(output) = separable_reduce_window_sum_f64(
        tensor,
        input_rows,
        input_cols,
        out_rows,
        out_cols,
        window_rows,
        window_cols,
        stride_rows,
        stride_cols,
        pad_rows,
        pad_cols,
    ) {
        return Ok(Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: out_dims.to_vec(),
                },
                output,
            )
            .map_err(EvalError::InvalidTensor)?,
        ));
    }

    // Dense fast path: read the operand straight from its contiguous f64 backing
    // and write a dense f64 output, bypassing the per-window-element Literal
    // materialization. Bit-identical to the generic loop below — same in-bounds
    // window iteration, same ascending `+` accumulation from 0.0, same from_f64
    // output (for dense f64, src[idx] == as_f64().unwrap_or(0.0)).
    if let Some(src) = tensor.elements.as_f64_slice() {
        let mut output = Vec::with_capacity(total_output);
        for out_row in 0..out_rows {
            let row_base = out_row.checked_mul(stride_rows).ok_or_else(|| {
                reduce_window_unsupported(primitive, "reduce_window window index overflow")
            })?;
            for out_col in 0..out_cols {
                let col_base = out_col.checked_mul(stride_cols).ok_or_else(|| {
                    reduce_window_unsupported(primitive, "reduce_window window index overflow")
                })?;
                let mut accum = 0.0;
                for window_row in 0..window_rows {
                    let padded_row = row_base.checked_add(window_row).ok_or_else(|| {
                        reduce_window_unsupported(primitive, "reduce_window window index overflow")
                    })?;
                    let row_oob = padded_row < pad_rows || (padded_row - pad_rows) >= input_rows;
                    let row_offset = if row_oob {
                        0
                    } else {
                        (padded_row - pad_rows)
                            .checked_mul(input_cols)
                            .ok_or_else(|| {
                                reduce_window_unsupported(
                                    primitive,
                                    "reduce_window flat index overflow",
                                )
                            })?
                    };
                    for window_col in 0..window_cols {
                        let padded_col = col_base.checked_add(window_col).ok_or_else(|| {
                            reduce_window_unsupported(
                                primitive,
                                "reduce_window window index overflow",
                            )
                        })?;
                        // Out-of-bounds (padding) taps add 0.0 — XLA pads
                        // reduce_window with the sum init (0.0) and reduces the
                        // full window; a no-op for finite data, fixes signed-zero.
                        let v = if row_oob
                            || padded_col < pad_cols
                            || (padded_col - pad_cols) >= input_cols
                        {
                            0.0
                        } else {
                            src[row_offset + (padded_col - pad_cols)]
                        };
                        accum += v;
                    }
                }
                output.push(accum);
            }
        }
        return Ok(Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: out_dims.to_vec(),
                },
                output,
            )
            .map_err(EvalError::InvalidTensor)?,
        ));
    }

    let mut output_elements = Vec::with_capacity(total_output);
    for out_row in 0..out_rows {
        let row_base = out_row.checked_mul(stride_rows).ok_or_else(|| {
            reduce_window_unsupported(primitive, "reduce_window window index overflow")
        })?;
        for out_col in 0..out_cols {
            let col_base = out_col.checked_mul(stride_cols).ok_or_else(|| {
                reduce_window_unsupported(primitive, "reduce_window window index overflow")
            })?;
            let mut accum = 0.0;

            for window_row in 0..window_rows {
                let padded_row = row_base.checked_add(window_row).ok_or_else(|| {
                    reduce_window_unsupported(primitive, "reduce_window window index overflow")
                })?;
                let row_oob = padded_row < pad_rows || (padded_row - pad_rows) >= input_rows;
                let row_offset = if row_oob {
                    0
                } else {
                    (padded_row - pad_rows)
                        .checked_mul(input_cols)
                        .ok_or_else(|| {
                            reduce_window_unsupported(
                                primitive,
                                "reduce_window flat index overflow",
                            )
                        })?
                };

                for window_col in 0..window_cols {
                    let padded_col = col_base.checked_add(window_col).ok_or_else(|| {
                        reduce_window_unsupported(primitive, "reduce_window window index overflow")
                    })?;
                    // OOB (padding) taps add 0.0 — XLA reduce_window init=0 over
                    // the full window (no-op for finite data, fixes signed-zero).
                    let v = if row_oob
                        || padded_col < pad_cols
                        || (padded_col - pad_cols) >= input_cols
                    {
                        0.0
                    } else {
                        tensor.elements[row_offset + (padded_col - pad_cols)]
                            .as_f64()
                            .unwrap_or(0.0)
                    };
                    accum += v;
                }
            }

            output_elements.push(Literal::from_f64(accum));
        }
    }

    Ok(Value::Tensor(
        TensorValue::new(
            tensor.dtype,
            Shape {
                dims: out_dims.to_vec(),
            },
            output_elements,
        )
        .map_err(EvalError::InvalidTensor)?,
    ))
}

/// Rank-2 F64 max/min reduce_window (maxpool/minpool) over contiguous f64
/// values. Dense tensors borrow their backing storage; literal-backed F64
/// tensors are packed once by the caller. Bit-for-bit identical to the generic
/// path for F64: same in-bounds-only window iteration and padding skip, same
/// `-inf`/`+inf` initial accumulator, and the same `jax_max_f64`/`jax_min_f64`
/// (NaN-propagating) combiner. `is_max` selects max vs min.
#[allow(clippy::too_many_arguments)]
fn eval_reduce_window_rank2_f64_max_min(
    primitive: Primitive,
    tensor: &TensorValue,
    src: &[f64],
    window_dims: &[usize],
    strides: &[usize],
    out_dims: &[u32],
    pad_lows: &[usize],
    total_output: usize,
    is_max: bool,
) -> Result<Value, EvalError> {
    let input_rows = tensor.shape.dims[0] as usize;
    let input_cols = tensor.shape.dims[1] as usize;
    let out_rows = out_dims[0] as usize;
    let out_cols = out_dims[1] as usize;
    let window_rows = window_dims[0];
    let window_cols = window_dims[1];
    let stride_rows = strides[0];
    let stride_cols = strides[1];
    let pad_rows = pad_lows[0];
    let pad_cols = pad_lows[1];
    let init = if is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };

    let mut output = Vec::with_capacity(total_output);
    for out_row in 0..out_rows {
        let row_base = out_row.checked_mul(stride_rows).ok_or_else(|| {
            reduce_window_unsupported(primitive, "reduce_window window index overflow")
        })?;
        for out_col in 0..out_cols {
            let col_base = out_col.checked_mul(stride_cols).ok_or_else(|| {
                reduce_window_unsupported(primitive, "reduce_window window index overflow")
            })?;
            let mut accum = init;
            for window_row in 0..window_rows {
                let padded_row = row_base.checked_add(window_row).ok_or_else(|| {
                    reduce_window_unsupported(primitive, "reduce_window window index overflow")
                })?;
                if padded_row < pad_rows {
                    continue;
                }
                let input_row = padded_row - pad_rows;
                if input_row >= input_rows {
                    continue;
                }
                let row_offset = input_row.checked_mul(input_cols).ok_or_else(|| {
                    reduce_window_unsupported(primitive, "reduce_window flat index overflow")
                })?;
                for window_col in 0..window_cols {
                    let padded_col = col_base.checked_add(window_col).ok_or_else(|| {
                        reduce_window_unsupported(primitive, "reduce_window window index overflow")
                    })?;
                    if padded_col < pad_cols {
                        continue;
                    }
                    let input_col = padded_col - pad_cols;
                    if input_col >= input_cols {
                        continue;
                    }
                    let flat_input_idx = row_offset.checked_add(input_col).ok_or_else(|| {
                        reduce_window_unsupported(primitive, "reduce_window flat index overflow")
                    })?;
                    let v = src[flat_input_idx];
                    accum = if is_max {
                        jax_max_f64(accum, v)
                    } else {
                        jax_min_f64(accum, v)
                    };
                }
            }
            output.push(accum);
        }
    }

    Ok(Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: out_dims.to_vec(),
            },
            output,
        )
        .map_err(EvalError::InvalidTensor)?,
    ))
}

/// Borrow a dense `f64` view of a reduce_window input for the dense float fast
/// path: F64 storage is borrowed directly; dense-F32 storage is widened once to an
/// owned `Vec<f64>` (exact, lossless — the same `as_f64()` the generic loop applies
/// per tap, but hoisted out of the O(output·window) tap loop). Returns `None` for
/// boxed/non-float storage so the caller falls back to the generic per-`Literal`
/// loop.
fn reduce_window_dense_f64_view(tensor: &TensorValue) -> Option<Cow<'_, [f64]>> {
    match tensor.dtype {
        fj_core::DType::F64 => tensor.elements.as_f64_slice().map(Cow::Borrowed),
        fj_core::DType::F32 => tensor
            .elements
            .as_f32_slice()
            .map(|s| Cow::Owned(s.iter().map(|&v| f64::from(v)).collect())),
        // BF16/F16 widen EXACTLY as `Literal::as_f64` does (the same conversion the
        // generic per-`Literal` path applies per tap), so the dense f64-accumulator
        // path is bit-identical: same widen, same f64 accumulate order, same final
        // round (`reduce_window_literal_from_f64`). Half pooling (bf16/f16) is JAX's
        // common ML dtype and otherwise ran the slow generic path.
        // BF16 widens via SIMD (`widen_bf16_slice_to_f64`, bit-identical to the scalar
        // `as_f64` lane-wise) — the scalar per-element widen was the ~30x bottleneck for bf16
        // pooling. F16 stays scalar (its widen needs per-lane special-case handling).
        fj_core::DType::BF16 => tensor
            .elements
            .as_half_float_slice()
            .map(|s| Cow::Owned(crate::arithmetic::widen_bf16_slice_to_f64(s))),
        fj_core::DType::F16 => tensor
            .elements
            .as_half_float_slice()
            .map(|s| Cow::Owned(crate::arithmetic::widen_f16_slice_to_f64(s))),
        _ => None,
    }
}

/// Dense float reduce_window (any rank, sum/max/min, F64 or dense-F32 input).
///
/// The F64-accumulator float case of the generic loop, specialized for a dense
/// source: the per-tap `Literal` gather + string-dispatched accumulate is replaced
/// by a direct `f64` read and a single op selected ONCE. For each output position
/// the window is classified interior (all taps in bounds) vs border; the interior
/// runs a tight loop over a precomputed flat tap-offset stencil (no per-tap bounds
/// check — vectorizes for sum), the border keeps the per-tap bounds check with
/// OOB taps contributing the init value, exactly as the generic loop does.
///
/// BIT-FOR-BIT identical to the generic per-`Literal` loop: same row-major tap
/// order, same `f64` init (max=-inf/min=+inf/sum=0), same `jax_max_f64`/`jax_min_f64`/
/// `+=` ops, same OOB→init contribution, and the same final round
/// (`new_f64_values` == `from_f64`, `new_f32_values(v as f32)` == `from_f32(v as f32)`
/// == `reduce_window_literal_from_f64`).
#[allow(clippy::too_many_arguments)]
fn eval_reduce_window_dense_float(
    output_dtype: fj_core::DType,
    reduce_op: &str,
    src: &[f64],
    window_dims: &[usize],
    strides: &[usize],
    window_dilation: &[usize],
    out_dims: &[u32],
    pad_lows: &[usize],
    input_dims: &[usize],
    input_strides: &[usize],
    total_output: usize,
) -> Result<Value, EvalError> {
    let rank = window_dims.len();
    let win_total: usize = window_dims.iter().product();

    // Precompute, in row-major window order (last dim fastest — matching the
    // generic odometer), each tap's per-dim coord and its flat offset from the
    // window's input top-left corner.
    let mut tap_coord = vec![0usize; rank];
    let mut tap_offsets: Vec<usize> = Vec::with_capacity(win_total);
    let mut tap_coords: Vec<Vec<usize>> = Vec::with_capacity(win_total);
    for _ in 0..win_total {
        let mut off = 0usize;
        for d in 0..rank {
            // window_dilation spaces taps `wd` apart (atrous pooling); wd==1 (the
            // contiguous case) is bit-identical to the undilated stencil.
            off += tap_coord[d] * window_dilation[d] * input_strides[d];
        }
        tap_offsets.push(off);
        tap_coords.push(tap_coord.clone());
        let mut carry = true;
        for d in (0..rank).rev() {
            if carry {
                tap_coord[d] += 1;
                if tap_coord[d] >= window_dims[d] {
                    tap_coord[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Op {
        Sum,
        Max,
        Min,
    }
    let op = match reduce_op {
        "max" => Op::Max,
        "min" => Op::Min,
        _ => Op::Sum,
    };
    let init = match op {
        Op::Max => f64::NEG_INFINITY,
        Op::Min => f64::INFINITY,
        Op::Sum => 0.0,
    };

    let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
    // Per-output-cell window reduction; cells are independent and each scans its window in the
    // SAME fixed tap order, so this threads bit-identically across the output (sum + dilated max/min;
    // the non-dilated max/min large-window case takes the separable deque path before reaching here).
    let cell = |out_idx: &[usize]| -> f64 {
        // Window's input top-left corner per dim (may be negative under padding).
        // Interior iff every tap lands in bounds: corner >= 0 and
        // corner + (win-1) < input_dim for all dims.
        let mut interior = true;
        let mut base: isize = 0;
        for d in 0..rank {
            let corner = out_idx[d] as isize * strides[d] as isize - pad_lows[d] as isize;
            if corner < 0
                || corner + (window_dims[d] as isize - 1) * window_dilation[d] as isize
                    >= input_dims[d] as isize
            {
                interior = false;
                break;
            }
            base += corner * input_strides[d] as isize;
        }

        let mut acc = init;
        if interior {
            let base = base as usize;
            match op {
                Op::Sum => {
                    for &off in &tap_offsets {
                        acc += src[base + off];
                    }
                }
                Op::Max => {
                    for &off in &tap_offsets {
                        acc = jax_max_f64(acc, src[base + off]);
                    }
                }
                Op::Min => {
                    for &off in &tap_offsets {
                        acc = jax_min_f64(acc, src[base + off]);
                    }
                }
            }
        } else {
            for coords in &tap_coords {
                let mut in_bounds = true;
                let mut flat = 0usize;
                for d in (0..rank).rev() {
                    let padded = out_idx[d] * strides[d] + coords[d] * window_dilation[d];
                    if padded < pad_lows[d] {
                        in_bounds = false;
                        break;
                    }
                    let ip = padded - pad_lows[d];
                    if ip >= input_dims[d] {
                        in_bounds = false;
                        break;
                    }
                    flat += ip * input_strides[d];
                }
                let tap = if in_bounds { src[flat] } else { init };
                match op {
                    Op::Sum => acc += tap,
                    Op::Max => acc = jax_max_f64(acc, tap),
                    Op::Min => acc = jax_min_f64(acc, tap),
                }
            }
        }
        acc
    };
    let mut values: Vec<f64> = vec![0.0; total_output];
    // Fill a contiguous output range [start, start+slots.len()), recovering the starting odometer
    // coords from the flat `start`. Each range is disjoint → safe split_at_mut, no cross-thread races.
    let fill = |start: usize, slots: &mut [f64]| {
        let mut out_idx = vec![0usize; rank];
        let mut rem = start;
        for d in (0..rank).rev() {
            out_idx[d] = rem % out_dims_usize[d];
            rem /= out_dims_usize[d];
        }
        for slot in slots.iter_mut() {
            *slot = cell(&out_idx);
            let mut carry = true;
            for d in (0..rank).rev() {
                if carry {
                    out_idx[d] += 1;
                    if out_idx[d] >= out_dims_usize[d] {
                        out_idx[d] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }
    };
    let work = total_output.saturating_mul(win_total);
    let threads = if work >= (1 << 18) && total_output >= 2 {
        crate::arithmetic::work_scaled_threads(work).min(total_output)
    } else {
        1
    };
    if threads <= 1 {
        fill(0, &mut values);
    } else {
        let per = total_output.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut rest: &mut [f64] = values.as_mut_slice();
            let mut s = 0usize;
            while s < total_output {
                let cnt = per.min(total_output - s);
                let (blk, tail) = rest.split_at_mut(cnt);
                rest = tail;
                let start = s;
                s += cnt;
                let f = &fill;
                scope.spawn(move || f(start, blk));
            }
        });
    }

    let shape = Shape {
        dims: out_dims.to_vec(),
    };
    match output_dtype {
        fj_core::DType::F32 => {
            let f32_values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            Ok(Value::Tensor(
                TensorValue::new_f32_values(shape, f32_values).map_err(EvalError::from)?,
            ))
        }
        fj_core::DType::BF16 | fj_core::DType::F16 => {
            // Round each f64 accumulator to the half dtype via the SAME
            // reduce_window_literal_from_f64 the generic path uses, and emit dense
            // half storage. (Sum accumulates in f64 then rounds once — identical to
            // the generic F64-accumulator path; max/min select an exactly-widened
            // input value so the round-trip is exact.)
            let bits: Vec<u16> = values
                .iter()
                .map(|&v| match reduce_window_literal_from_f64(output_dtype, v) {
                    fj_core::Literal::BF16Bits(b) | fj_core::Literal::F16Bits(b) => b,
                    _ => 0,
                })
                .collect();
            Ok(Value::Tensor(
                TensorValue::new_half_float_values(output_dtype, shape, bits)
                    .map_err(EvalError::from)?,
            ))
        }
        _ => Ok(Value::Tensor(
            TensorValue::new_f64_values(shape, values).map_err(EvalError::from)?,
        )),
    }
}

fn reduce_window_rank3_f64_sum_xlane(
    src: &[f64],
    input_dims: [usize; 3],
    window_dims: [usize; 3],
    out_dims: [usize; 3],
) -> Vec<f64> {
    use std::simd::Simd;
    type F = Simd<f64, 8>;

    let [_, in_y, in_x] = input_dims;
    let [win_z, win_y, win_x] = window_dims;
    let [out_z, out_y, out_x] = out_dims;
    let in_z_stride = in_y * in_x;
    let x8 = out_x - out_x % 8;
    let rows = out_z * out_y;
    let mut out = vec![0.0_f64; rows * out_x];

    let fill_rows = |row_start: usize, rows_out: &mut [f64]| {
        let n_rows = rows_out.len() / out_x;
        for local_row in 0..n_rows {
            let row = row_start + local_row;
            let oz = row / out_y;
            let oy = row % out_y;
            let out_base = local_row * out_x;
            let mut ox = 0usize;
            while ox < x8 {
                let mut acc = F::splat(0.0);
                for wz in 0..win_z {
                    let z_base = (oz + wz) * in_z_stride;
                    for wy in 0..win_y {
                        let row_base = z_base + (oy + wy) * in_x + ox;
                        for wx in 0..win_x {
                            acc += F::from_slice(&src[row_base + wx..row_base + wx + 8]);
                        }
                    }
                }
                acc.copy_to_slice(&mut rows_out[out_base + ox..out_base + ox + 8]);
                ox += 8;
            }
            while ox < out_x {
                let mut acc = 0.0_f64;
                for wz in 0..win_z {
                    let z_base = (oz + wz) * in_z_stride;
                    for wy in 0..win_y {
                        let row_base = z_base + (oy + wy) * in_x + ox;
                        for wx in 0..win_x {
                            acc += src[row_base + wx];
                        }
                    }
                }
                rows_out[out_base + ox] = acc;
                ox += 1;
            }
        }
    };

    let work = rows
        .saturating_mul(out_x)
        .saturating_mul(win_z * win_y * win_x);
    let threads = if work >= (1 << 18) && rows >= 2 {
        crate::arithmetic::work_scaled_threads(work).min(rows)
    } else {
        1
    };
    if threads <= 1 {
        fill_rows(0, &mut out);
        return out;
    }

    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = out.as_mut_slice();
        let mut row0 = 0usize;
        while row0 < rows {
            let cnt = per.min(rows - row0);
            let (blk, tail) = rest.split_at_mut(cnt * out_x);
            rest = tail;
            let start = row0;
            row0 += cnt;
            let f = &fill_rows;
            scope.spawn(move || f(start, blk));
        }
    });
    out
}

struct ReduceWindowExtremumAxis {
    axis: usize,
    window: usize,
    stride: usize,
    pad_low: usize,
    out_ax: usize,
    is_max: bool,
}

/// One 1-D sliding-window max/min pass along `axis` of a row-major dense f64
/// tensor `cur` (shape `dims`), via a MONOTONIC DEQUE: O(axis_len) per line,
/// WINDOW-INDEPENDENT (vs the naive O(axis_len·window) tap loop). A separate
/// `nan_count` over the window makes any NaN-containing window emit `f64::NAN`,
/// matching `jax_max_f64`/`jax_min_f64` (which canonicalize NaN to `f64::NAN`),
/// so the result is bit-for-bit identical to the naive accumulator — finite
/// windows reduce to the unique max/min, NaN windows to canonical NaN. Out-of-
/// bounds (padded) taps are the op identity (-inf max / +inf min) and so are
/// simply never visited, exactly as the naive halo treats them. Returns the new
/// buffer with `out_ax` positions along `axis`.
fn reduce_window_extremum_1d_axis_f64(
    cur: &[f64],
    dims: &[usize],
    spec: ReduceWindowExtremumAxis,
) -> Vec<f64> {
    let axis_len = dims[spec.axis];
    let inner: usize = dims[spec.axis + 1..].iter().product();
    let outer: usize = dims[..spec.axis].iter().product();
    let ident = if spec.is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let mut out = vec![0.0_f64; outer * spec.out_ax * inner];
    // One line (o, ii)'s deque scan: WINDOW-INDEPENDENT, writes the line's `out_ax` outputs at
    // stride `inner` within this `o`-block. Pure-scratch deque, no cross-line state.
    let scan_o_block = |o_lo: usize, blk: &mut [f64]| {
        let mut dq: std::collections::VecDeque<(f64, usize)> = std::collections::VecDeque::new();
        for (bo, o) in (o_lo..).take(blk.len() / (spec.out_ax * inner)).enumerate() {
            let o_out_base = bo * spec.out_ax * inner;
            for ii in 0..inner {
                let line_base = o * axis_len * inner + ii;
                let get = |k: usize| cur[line_base + k * inner];
                dq.clear();
                let mut nan_count = 0usize;
                let mut added = 0usize; // indices [0, added) entered
                let mut left = 0usize; // indices [0, left) removed
                let out_base = o_out_base + ii;
                for oa in 0..spec.out_ax {
                    let start = oa * spec.stride;
                    let lo = start.saturating_sub(spec.pad_low);
                    let hi = (start + spec.window)
                        .saturating_sub(spec.pad_low)
                        .min(axis_len);
                    while added < hi {
                        let v = get(added);
                        if v.is_nan() {
                            nan_count += 1;
                        } else {
                            while let Some(&(bv, _)) = dq.back() {
                                let dominated = if spec.is_max { bv <= v } else { bv >= v };
                                if dominated {
                                    dq.pop_back();
                                } else {
                                    break;
                                }
                            }
                            dq.push_back((v, added));
                        }
                        added += 1;
                    }
                    while left < lo {
                        if get(left).is_nan() {
                            nan_count -= 1;
                        }
                        if dq.front().map(|&(_, i)| i) == Some(left) {
                            dq.pop_front();
                        }
                        left += 1;
                    }
                    let val = if hi <= lo {
                        ident
                    } else if nan_count > 0 {
                        f64::NAN
                    } else {
                        dq.front().map(|&(v, _)| v).unwrap_or(ident)
                    };
                    blk[out_base + oa * inner] = val;
                }
            }
        }
    };
    // Thread over the independent OUTER blocks (each `o`-block of `out_ax*inner` outputs is a
    // contiguous, disjoint slice — no strided cross-thread writes). Bit-identical to the serial scan.
    let work = outer * axis_len * inner;
    let threads = if work >= (1 << 18) && outer >= 2 {
        crate::arithmetic::work_scaled_threads(work).min(outer)
    } else {
        1
    };
    if threads <= 1 {
        scan_o_block(0, &mut out);
        return out;
    }
    let per = outer.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = out.as_mut_slice();
        let mut o0 = 0usize;
        while o0 < outer {
            let cnt = per.min(outer - o0);
            let (blk, tail) = rest.split_at_mut(cnt * spec.out_ax * inner);
            rest = tail;
            let start = o0;
            o0 += cnt;
            let f = &scan_o_block;
            scope.spawn(move || f(start, blk));
        }
    });
    out
}

/// Separable monotonic-deque max/min `reduce_window` (any rank, no dilation): a box
/// max/min over a window is the composition of per-axis 1-D sliding max/min (max and
/// min are separable and order-independent), so we run [`reduce_window_extremum_1d_axis_f64`]
/// along each windowed axis in turn — O(total) overall, WINDOW-INDEPENDENT, vs the
/// naive O(out·∏window). Bit-for-bit identical to `eval_reduce_window_dense_float`'s
/// max/min path. Output dtype handling mirrors that function (max/min select an
/// exactly-widened input value, so F32/half downcast is exact).
#[allow(clippy::too_many_arguments)]
fn eval_reduce_window_separable_extremum(
    output_dtype: fj_core::DType,
    is_max: bool,
    src: &[f64],
    window_dims: &[usize],
    strides: &[usize],
    out_dims: &[u32],
    pad_lows: &[usize],
    input_dims: &[usize],
) -> Result<Value, EvalError> {
    let rank = window_dims.len();
    let mut cur = src.to_vec();
    let mut cur_dims = input_dims.to_vec();
    for ax in 0..rank {
        let w = window_dims[ax];
        let s = strides[ax];
        let p = pad_lows[ax];
        let out_ax = out_dims[ax] as usize;
        // Pure pass-through axis (no window, unit stride, no pad, unchanged length).
        if w == 1 && s == 1 && p == 0 && out_ax == cur_dims[ax] {
            continue;
        }
        cur = reduce_window_extremum_1d_axis_f64(
            &cur,
            &cur_dims,
            ReduceWindowExtremumAxis {
                axis: ax,
                window: w,
                stride: s,
                pad_low: p,
                out_ax,
                is_max,
            },
        );
        cur_dims[ax] = out_ax;
    }
    let shape = Shape {
        dims: out_dims.to_vec(),
    };
    match output_dtype {
        fj_core::DType::F32 => Ok(Value::Tensor(
            TensorValue::new_f32_values(shape, cur.iter().map(|&v| v as f32).collect())
                .map_err(EvalError::from)?,
        )),
        fj_core::DType::BF16 | fj_core::DType::F16 => {
            let bits: Vec<u16> = cur
                .iter()
                .map(|&v| match reduce_window_literal_from_f64(output_dtype, v) {
                    fj_core::Literal::BF16Bits(b) | fj_core::Literal::F16Bits(b) => b,
                    _ => 0,
                })
                .collect();
            Ok(Value::Tensor(
                TensorValue::new_half_float_values(output_dtype, shape, bits)
                    .map_err(EvalError::from)?,
            ))
        }
        _ => Ok(Value::Tensor(
            TensorValue::new_f64_values(shape, cur).map_err(EvalError::from)?,
        )),
    }
}

/// i64 sibling of [`reduce_window_extremum_1d_axis_f64`]: 1-D sliding max/min along
/// `axis` via a monotonic deque, O(axis_len) per line. Simpler than the f64 version —
/// integers have no NaN, so no `nan_count`. Out-of-bounds (padded) taps are the op
/// identity (`i64::MIN` max / `i64::MAX` min) and are never visited, matching the
/// naive i64 halo. Returns the new buffer with `out_ax` positions along `axis`.
fn reduce_window_extremum_1d_axis_i64(
    cur: &[i64],
    dims: &[usize],
    spec: ReduceWindowExtremumAxis,
) -> Vec<i64> {
    let axis_len = dims[spec.axis];
    let inner: usize = dims[spec.axis + 1..].iter().product();
    let outer: usize = dims[..spec.axis].iter().product();
    let ident = if spec.is_max { i64::MIN } else { i64::MAX };
    let mut out = vec![0_i64; outer * spec.out_ax * inner];
    let mut dq: std::collections::VecDeque<(i64, usize)> = std::collections::VecDeque::new();
    for o in 0..outer {
        for ii in 0..inner {
            let line_base = o * axis_len * inner + ii;
            let get = |k: usize| cur[line_base + k * inner];
            dq.clear();
            let mut added = 0usize;
            let mut left = 0usize;
            let out_base = o * spec.out_ax * inner + ii;
            for oa in 0..spec.out_ax {
                let start = oa * spec.stride;
                let lo = start.saturating_sub(spec.pad_low);
                let hi = (start + spec.window)
                    .saturating_sub(spec.pad_low)
                    .min(axis_len);
                while added < hi {
                    let v = get(added);
                    while let Some(&(bv, _)) = dq.back() {
                        let dominated = if spec.is_max { bv <= v } else { bv >= v };
                        if dominated {
                            dq.pop_back();
                        } else {
                            break;
                        }
                    }
                    dq.push_back((v, added));
                    added += 1;
                }
                while left < lo {
                    if dq.front().map(|&(_, i)| i) == Some(left) {
                        dq.pop_front();
                    }
                    left += 1;
                }
                let val = if hi <= lo {
                    ident
                } else {
                    dq.front().map(|&(v, _)| v).unwrap_or(ident)
                };
                out[out_base + oa * inner] = val;
            }
        }
    }
    out
}

/// Separable monotonic-deque max/min `reduce_window` for dense i64/i32 (no dilation):
/// the integer sibling of [`eval_reduce_window_separable_extremum`]. Runs one O(axis_len)
/// deque pass per windowed axis (max/min separable + order-independent), O(total) vs the
/// naive O(out·∏window) tap loop. Bit-for-bit identical to `eval_reduce_window_dense_i64`'s
/// max/min path (integer max/min is exact). I32 output is wrapped/tagged exactly as that
/// function does (max/min of in-range i32 values stay in range, so the wrap is a no-op).
#[allow(clippy::too_many_arguments)]
fn eval_reduce_window_separable_extremum_i64(
    out_dtype: fj_core::DType,
    is_max: bool,
    src: &[i64],
    window_dims: &[usize],
    strides: &[usize],
    out_dims: &[u32],
    pad_lows: &[usize],
    input_dims: &[usize],
) -> Result<Value, EvalError> {
    let rank = window_dims.len();
    let mut cur = src.to_vec();
    let mut cur_dims = input_dims.to_vec();
    for ax in 0..rank {
        let w = window_dims[ax];
        let s = strides[ax];
        let p = pad_lows[ax];
        let out_ax = out_dims[ax] as usize;
        if w == 1 && s == 1 && p == 0 && out_ax == cur_dims[ax] {
            continue;
        }
        cur = reduce_window_extremum_1d_axis_i64(
            &cur,
            &cur_dims,
            ReduceWindowExtremumAxis {
                axis: ax,
                window: w,
                stride: s,
                pad_low: p,
                out_ax,
                is_max,
            },
        );
        cur_dims[ax] = out_ax;
    }
    let shape = Shape {
        dims: out_dims.to_vec(),
    };
    if out_dtype == fj_core::DType::I32 {
        let wrapped: Vec<i64> = cur.into_iter().map(|v| i64::from(v as i32)).collect();
        return Ok(Value::Tensor(
            TensorValue::new_i32_values(shape, wrapped).map_err(EvalError::from)?,
        ));
    }
    Ok(Value::Tensor(
        TensorValue::new_i64_values(shape, cur).map_err(EvalError::from)?,
    ))
}

/// Dense i64 reduce_window (any rank, sum/max/min, I64 input).
///
/// The I64-accumulator integer case of the generic loop, specialized for a dense
/// `i64` source: the per-tap `Literal` gather + string-dispatched accumulate is
/// replaced by a direct `i64` read and a single op selected ONCE, with the same
/// interior-vs-border classification and precomputed flat tap-offset stencil used
/// by `eval_reduce_window_dense_float`.
///
/// BIT-FOR-BIT identical to the generic per-`Literal` loop for I64 input: same
/// row-major tap order, same i64 init (max=`i64::MIN`/min=`i64::MAX`/sum=0), the
/// same `i64::max`/`i64::min`/`wrapping_add` ops the I64 accumulator uses, and the
/// same OOB→init contribution (init is the identity for every op: `wrapping_add(0)`,
/// `max(_, i64::MIN)`, `min(_, i64::MAX)` are all no-ops). Output is `Literal::I64`,
/// matching `reduce_window_accumulator_literal` for the I64 dtype.
#[allow(clippy::too_many_arguments)]
/// Summed-area-table (integral image) i64 sum reduce_window for rank 2.
///
/// Replaces the per-window O(out·∏window) tap sum with an O(input + out)
/// window-INDEPENDENT computation: build the integral image once, then each
/// output window sum is a 4-corner inclusion-exclusion. Integer addition is
/// associative AND commutative (i64 `wrapping_add` is exact in the ring mod
/// 2^64), so the rectangle sum equals the per-window ascending tap sum
/// bit-for-bit regardless of accumulation order — OOB/padding taps contribute 0
/// in BOTH (the clamped in-bounds rectangle is exactly the set of non-pad taps).
/// `wrapping_sub` inverts `wrapping_add` in the same ring, so the corner formula
/// is exact even under i64 overflow. Output is `Literal::I64`, matching the
/// dense/generic i64 path. Wins for large windows (small 2×2/3×3 stay on the
/// per-window dense path, which has no integral-image build cost).
/// Re-tag an i64 summed-area-table result to I32 for an i32 input. The SAT emits the
/// (i64-ring) windowed sums; tagging them I32 lets the eval_primitive chokepoint wrap
/// each mod 2^32 — congruent to the per-window i32 sum since the 4-corner difference is
/// computed in the same ring (mod 2^32 is a +/− homomorphism). Identity for I64.
fn reduce_window_sat_retag_i32(result: Value, dtype: fj_core::DType) -> Result<Value, EvalError> {
    if dtype == fj_core::DType::I32
        && let Value::Tensor(t) = &result
        && let Some(v) = t.elements.as_i64_slice()
    {
        return Ok(Value::Tensor(
            TensorValue::new_i32_values(t.shape.clone(), v.to_vec()).map_err(EvalError::from)?,
        ));
    }
    Ok(result)
}

fn eval_reduce_window_rank2_i64_sum_sat(
    src: &[i64],
    window_dims: &[usize],
    strides: &[usize],
    out_dims: &[u32],
    pad_lows: &[usize],
    input_dims: &[usize],
    total_output: usize,
) -> Result<Value, EvalError> {
    let input_rows = input_dims[0];
    let input_cols = input_dims[1];
    let out_rows = out_dims[0] as usize;
    let out_cols = out_dims[1] as usize;
    let (win_r, win_c) = (window_dims[0], window_dims[1]);
    let (stride_r, stride_c) = (strides[0], strides[1]);
    let (pad_r, pad_c) = (pad_lows[0], pad_lows[1]);

    // Integral image `sat` of shape (input_rows+1) x (input_cols+1), with a zero
    // first row/col so a sub-rectangle sum is sat[r1][c1] - sat[r0][c1]
    // - sat[r1][c0] + sat[r0][c0]. All ops wrap in the i64 ring.
    let stride = input_cols + 1;
    let mut sat = vec![0i64; (input_rows + 1) * stride];
    for r in 0..input_rows {
        let in_row = r * input_cols;
        let sat_row = (r + 1) * stride;
        let sat_prev = r * stride;
        let mut row_run = 0i64;
        for c in 0..input_cols {
            row_run = row_run.wrapping_add(src[in_row + c]);
            sat[sat_row + c + 1] = row_run.wrapping_add(sat[sat_prev + c + 1]);
        }
    }

    let mut values: Vec<i64> = Vec::with_capacity(total_output);
    for or in 0..out_rows {
        // Padded window rows [base_r, base_r+win_r); in-bounds input rows are
        // those minus pad_r, clamped to [0, input_rows).
        let base_r = or * stride_r;
        let lo_r = base_r as isize - pad_r as isize;
        let r0 = lo_r.max(0) as usize;
        let r1 = (lo_r + win_r as isize).clamp(0, input_rows as isize) as usize;
        for oc in 0..out_cols {
            let base_c = oc * stride_c;
            let lo_c = base_c as isize - pad_c as isize;
            let c0 = lo_c.max(0) as usize;
            let c1 = (lo_c + win_c as isize).clamp(0, input_cols as isize) as usize;
            let sum = if r0 >= r1 || c0 >= c1 {
                0
            } else {
                sat[r1 * stride + c1]
                    .wrapping_sub(sat[r0 * stride + c1])
                    .wrapping_sub(sat[r1 * stride + c0])
                    .wrapping_add(sat[r0 * stride + c0])
            };
            values.push(sum);
        }
    }

    let shape = Shape {
        dims: out_dims.to_vec(),
    };
    Ok(Value::Tensor(
        TensorValue::new_i64_values(shape, values).map_err(EvalError::from)?,
    ))
}

/// General-rank summed-area-table (N-dim integral image) i64 sum reduce_window.
///
/// The rank-N generalization of `eval_reduce_window_rank2_i64_sum_sat`: build an
/// N-dim integral image once (prefix-sum along every axis), then each output
/// window is a 2^rank-corner inclusion-exclusion — O(rank·input + 2^rank·output),
/// WINDOW-INDEPENDENT. Covers rank-1 sliding-window integer sums and rank-3/4
/// NHWC integer sum-pooling. Same bit-exactness as the rank-2 path: i64 add is
/// associative+commutative in ℤ/2^64 and `wrapping_sub` inverts it, so the box
/// sum equals the per-window ascending wrapping tap sum bit-for-bit (OOB/pad taps
/// contribute 0 in both = the clamped in-bounds box). Caller gates `rank ≤ 6` so
/// the 2^rank corner loop stays bounded.
fn eval_reduce_window_in_sum_sat(
    src: &[i64],
    window_dims: &[usize],
    strides: &[usize],
    out_dims: &[u32],
    pad_lows: &[usize],
    input_dims: &[usize],
    total_output: usize,
) -> Result<Value, EvalError> {
    let primitive = Primitive::ReduceWindow;
    let rank = input_dims.len();

    // Padded integral-image dims (n_d + 1) and row-major strides.
    let p_dims: Vec<usize> = input_dims.iter().map(|&n| n + 1).collect();
    let mut p_strides = vec![1usize; rank];
    let mut acc = 1usize;
    for d in (0..rank).rev() {
        p_strides[d] = acc;
        acc = acc.checked_mul(p_dims[d]).ok_or_else(|| {
            reduce_window_unsupported(primitive, "reduce_window integral-image size overflow")
        })?;
    }
    let p_len = acc;
    let mut sat = vec![0i64; p_len];

    // Scatter input into the +1-shifted interior of the padded array.
    let in_strides = {
        let mut s = vec![1usize; rank];
        let mut a = 1usize;
        for d in (0..rank).rev() {
            s[d] = a;
            a = a.checked_mul(input_dims[d]).ok_or_else(|| {
                reduce_window_unsupported(primitive, "reduce_window input size overflow")
            })?;
        }
        s
    };
    for (flat, &v) in src.iter().enumerate() {
        let mut p_idx = 0usize;
        let mut rem = flat;
        for d in 0..rank {
            let coord = rem / in_strides[d];
            rem %= in_strides[d];
            p_idx += (coord + 1) * p_strides[d];
        }
        sat[p_idx] = v;
    }

    // Prefix-sum along each axis (in-place; ascending flat order ⇒ the predecessor
    // along axis d at i - p_strides[d] is already accumulated this pass).
    for d in 0..rank {
        let stride = p_strides[d];
        let dim = p_dims[d];
        for i in 0..p_len {
            if (i / stride) % dim >= 1 {
                sat[i] = sat[i].wrapping_add(sat[i - stride]);
            }
        }
    }

    let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
    // Per-output low/high in-bounds box endpoints (r0_d, r1_d), recomputed each step.
    let mut out_idx = vec![0usize; rank];
    let mut r0 = vec![0usize; rank];
    let mut r1 = vec![0usize; rank];
    let corners = 1usize << rank;
    let mut values: Vec<i64> = Vec::with_capacity(total_output);

    for _ in 0..total_output {
        let mut empty = false;
        for d in 0..rank {
            let lo = out_idx[d] as isize * strides[d] as isize - pad_lows[d] as isize;
            let a = lo.max(0) as usize;
            let b = (lo + window_dims[d] as isize).clamp(0, input_dims[d] as isize) as usize;
            r0[d] = a;
            r1[d] = b;
            if a >= b {
                empty = true;
            }
        }
        let sum = if empty {
            0
        } else {
            let mut s = 0i64;
            for corner in 0..corners {
                let mut p_idx = 0usize;
                let mut lows = 0u32;
                for d in 0..rank {
                    if (corner >> d) & 1 == 1 {
                        p_idx += r1[d] * p_strides[d];
                    } else {
                        p_idx += r0[d] * p_strides[d];
                        lows += 1;
                    }
                }
                if lows.is_multiple_of(2) {
                    s = s.wrapping_add(sat[p_idx]);
                } else {
                    s = s.wrapping_sub(sat[p_idx]);
                }
            }
            s
        };
        values.push(sum);

        // Row-major odometer over output dims.
        let mut carry = true;
        for d in (0..rank).rev() {
            if carry {
                out_idx[d] += 1;
                if out_idx[d] >= out_dims_usize[d] {
                    out_idx[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    let shape = Shape {
        dims: out_dims.to_vec(),
    };
    Ok(Value::Tensor(
        TensorValue::new_i64_values(shape, values).map_err(EvalError::from)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn eval_reduce_window_dense_i64(
    out_dtype: fj_core::DType,
    reduce_op: &str,
    src: &[i64],
    window_dims: &[usize],
    strides: &[usize],
    window_dilation: &[usize],
    out_dims: &[u32],
    pad_lows: &[usize],
    input_dims: &[usize],
    input_strides: &[usize],
    total_output: usize,
) -> Result<Value, EvalError> {
    let rank = window_dims.len();
    let win_total: usize = window_dims.iter().product();

    // Precompute, in row-major window order (last dim fastest — matching the
    // generic odometer), each tap's per-dim coord and its flat offset from the
    // window's input top-left corner.
    let mut tap_coord = vec![0usize; rank];
    let mut tap_offsets: Vec<usize> = Vec::with_capacity(win_total);
    let mut tap_coords: Vec<Vec<usize>> = Vec::with_capacity(win_total);
    for _ in 0..win_total {
        let mut off = 0usize;
        for d in 0..rank {
            // window_dilation spaces taps `wd` apart (atrous pooling); wd==1 (the
            // contiguous case) is bit-identical to the undilated stencil.
            off += tap_coord[d] * window_dilation[d] * input_strides[d];
        }
        tap_offsets.push(off);
        tap_coords.push(tap_coord.clone());
        let mut carry = true;
        for d in (0..rank).rev() {
            if carry {
                tap_coord[d] += 1;
                if tap_coord[d] >= window_dims[d] {
                    tap_coord[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Op {
        Sum,
        Max,
        Min,
    }
    let op = match reduce_op {
        "max" => Op::Max,
        "min" => Op::Min,
        _ => Op::Sum,
    };
    let init = match op {
        Op::Max => i64::MIN,
        Op::Min => i64::MAX,
        Op::Sum => 0,
    };

    let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
    let mut out_idx = vec![0usize; rank];
    let mut values: Vec<i64> = Vec::with_capacity(total_output);

    for _ in 0..total_output {
        // Window's input top-left corner per dim (may be negative under padding).
        // Interior iff every tap lands in bounds.
        let mut interior = true;
        let mut base: isize = 0;
        for d in 0..rank {
            let corner = out_idx[d] as isize * strides[d] as isize - pad_lows[d] as isize;
            if corner < 0
                || corner + (window_dims[d] as isize - 1) * window_dilation[d] as isize
                    >= input_dims[d] as isize
            {
                interior = false;
                break;
            }
            base += corner * input_strides[d] as isize;
        }

        let mut acc = init;
        if interior {
            let base = base as usize;
            match op {
                Op::Sum => {
                    for &off in &tap_offsets {
                        acc = acc.wrapping_add(src[base + off]);
                    }
                }
                Op::Max => {
                    for &off in &tap_offsets {
                        acc = acc.max(src[base + off]);
                    }
                }
                Op::Min => {
                    for &off in &tap_offsets {
                        acc = acc.min(src[base + off]);
                    }
                }
            }
        } else {
            for coords in &tap_coords {
                let mut in_bounds = true;
                let mut flat = 0usize;
                for d in (0..rank).rev() {
                    let padded = out_idx[d] * strides[d] + coords[d] * window_dilation[d];
                    if padded < pad_lows[d] {
                        in_bounds = false;
                        break;
                    }
                    let ip = padded - pad_lows[d];
                    if ip >= input_dims[d] {
                        in_bounds = false;
                        break;
                    }
                    flat += ip * input_strides[d];
                }
                let tap = if in_bounds { src[flat] } else { init };
                match op {
                    Op::Sum => acc = acc.wrapping_add(tap),
                    Op::Max => acc = acc.max(tap),
                    Op::Min => acc = acc.min(tap),
                }
            }
        }
        values.push(acc);

        let mut carry = true;
        for d in (0..rank).rev() {
            if carry {
                out_idx[d] += 1;
                if out_idx[d] >= out_dims_usize[d] {
                    out_idx[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    let shape = Shape {
        dims: out_dims.to_vec(),
    };
    // Preserve the input integer dtype. I32 (JAX's default int) shares the dense i64
    // backing (`as_i64_slice`), so it reuses this exact stencil; the output is tagged
    // I32 and the eval_primitive chokepoint (`narrow_i32_tensor_result`) wraps it
    // mod 2^32 — identical to the generic path, which also flows through that chokepoint
    // (sum: wrap-once == per-step wrap since + is a ring homomorphism; max/min select an
    // in-range input so no wrap occurs).
    if out_dtype == fj_core::DType::I32 {
        // Wrap mod 2^32 to the i32 range here (sum can overflow; max/min are already in
        // range) via the direct dense i32 constructor, so the result is canonical and the
        // narrow chokepoint no-ops instead of rescanning + rebuilding.
        let wrapped: Vec<i64> = values.into_iter().map(|v| i64::from(v as i32)).collect();
        return Ok(Value::Tensor(
            TensorValue::new_i32_values(shape, wrapped).map_err(EvalError::from)?,
        ));
    }
    Ok(Value::Tensor(
        TensorValue::new_i64_values(shape, values).map_err(EvalError::from)?,
    ))
}

/// Dense bool reduce_window (any rank, logical and/or pooling, Bool input).
///
/// The Bool-accumulator case of the generic loop, specialized for a dense `bool`
/// source: the per-tap `Literal` gather + string-dispatched accumulate is
/// Dense complex reduce_window (any rank, sum/max/min, Complex64/Complex128).
///
/// The Complex case of the generic loop, specialized for a dense `(re, im)`
/// source: replaces the per-tap `Literal` gather + string-dispatched accumulate
/// with a direct `(f64, f64)` read and a single op over the same row-major
/// tap-offset stencil and interior/border split as `eval_reduce_window_dense_i64`.
///
/// BIT-FOR-BIT identical to the generic Complex accumulator: same window order
/// (so the f64 component sums round identically), the SAME lexicographic
/// `reduce_window_complex_ge` for max/min, the SAME init
/// (`reduce_window_complex_initial`: max=(−∞,−∞), min=(+∞,+∞), sum=(0,0)) — and
/// OOB/border taps contribute that init, a no-op for every op. Output preserves
/// the input complex dtype.
#[allow(clippy::too_many_arguments)]
fn eval_reduce_window_dense_complex(
    dtype: fj_core::DType,
    reduce_op: &str,
    src: &[(f64, f64)],
    window_dims: &[usize],
    strides: &[usize],
    window_dilation: &[usize],
    out_dims: &[u32],
    pad_lows: &[usize],
    input_dims: &[usize],
    input_strides: &[usize],
    total_output: usize,
) -> Result<Value, EvalError> {
    let rank = window_dims.len();
    let win_total: usize = window_dims.iter().product();

    let mut tap_coord = vec![0usize; rank];
    let mut tap_offsets: Vec<usize> = Vec::with_capacity(win_total);
    let mut tap_coords: Vec<Vec<usize>> = Vec::with_capacity(win_total);
    for _ in 0..win_total {
        let mut off = 0usize;
        for d in 0..rank {
            // window_dilation spaces taps `wd` apart (atrous pooling); wd==1 (the
            // contiguous case) is bit-identical to the undilated stencil.
            off += tap_coord[d] * window_dilation[d] * input_strides[d];
        }
        tap_offsets.push(off);
        tap_coords.push(tap_coord.clone());
        let mut carry = true;
        for d in (0..rank).rev() {
            if carry {
                tap_coord[d] += 1;
                if tap_coord[d] >= window_dims[d] {
                    tap_coord[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Op {
        Sum,
        Max,
        Min,
    }
    let op = match reduce_op {
        "max" => Op::Max,
        "min" => Op::Min,
        _ => Op::Sum,
    };
    let init = reduce_window_complex_initial(reduce_op);
    let combine = |acc: (f64, f64), tap: (f64, f64)| -> (f64, f64) {
        match op {
            Op::Sum => (acc.0 + tap.0, acc.1 + tap.1),
            Op::Max => {
                if reduce_window_complex_ge(tap, acc) {
                    tap
                } else {
                    acc
                }
            }
            Op::Min => {
                if !reduce_window_complex_ge(tap, acc) {
                    tap
                } else {
                    acc
                }
            }
        }
    };

    let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
    let mut out_idx = vec![0usize; rank];
    let mut values: Vec<(f64, f64)> = Vec::with_capacity(total_output);

    for _ in 0..total_output {
        let mut interior = true;
        let mut base: isize = 0;
        for d in 0..rank {
            let corner = out_idx[d] as isize * strides[d] as isize - pad_lows[d] as isize;
            if corner < 0
                || corner + (window_dims[d] as isize - 1) * window_dilation[d] as isize
                    >= input_dims[d] as isize
            {
                interior = false;
                break;
            }
            base += corner * input_strides[d] as isize;
        }

        let mut acc = init;
        if interior {
            let base = base as usize;
            for &off in &tap_offsets {
                acc = combine(acc, src[base + off]);
            }
        } else {
            for coords in &tap_coords {
                let mut in_bounds = true;
                let mut flat = 0usize;
                for d in (0..rank).rev() {
                    let padded = out_idx[d] * strides[d] + coords[d] * window_dilation[d];
                    if padded < pad_lows[d] {
                        in_bounds = false;
                        break;
                    }
                    let ip = padded - pad_lows[d];
                    if ip >= input_dims[d] {
                        in_bounds = false;
                        break;
                    }
                    flat += ip * input_strides[d];
                }
                let tap = if in_bounds { src[flat] } else { init };
                acc = combine(acc, tap);
            }
        }
        values.push(acc);

        let mut carry = true;
        for d in (0..rank).rev() {
            if carry {
                out_idx[d] += 1;
                if out_idx[d] >= out_dims_usize[d] {
                    out_idx[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    let shape = Shape {
        dims: out_dims.to_vec(),
    };
    Ok(Value::Tensor(
        TensorValue::new_complex_values(dtype, shape, values).map_err(EvalError::from)?,
    ))
}

/// replaced by a direct `bool` read and a single op selected ONCE, over the same
/// row-major tap-offset stencil with the same interior/border split as
/// `eval_reduce_window_dense_i64`.
///
/// BIT-FOR-BIT identical to the generic Bool accumulator: `min` reduces with `&`
/// (init `true`), every other op with `|` (init `false`) — exactly the Bool arm
/// of `reduce_window_accumulate_literal`. OOB/border taps contribute `init`,
/// which is the identity for each op (`acc & true == acc`, `acc | false == acc`),
/// matching the generic `pad_literal` contribution. Output is `Literal::Bool`.
#[allow(clippy::too_many_arguments)]
fn eval_reduce_window_dense_bool(
    reduce_op: &str,
    src: &[bool],
    window_dims: &[usize],
    strides: &[usize],
    window_dilation: &[usize],
    out_dims: &[u32],
    pad_lows: &[usize],
    input_dims: &[usize],
    input_strides: &[usize],
    total_output: usize,
) -> Result<Value, EvalError> {
    let rank = window_dims.len();
    let win_total: usize = window_dims.iter().product();

    let mut tap_coord = vec![0usize; rank];
    let mut tap_offsets: Vec<usize> = Vec::with_capacity(win_total);
    let mut tap_coords: Vec<Vec<usize>> = Vec::with_capacity(win_total);
    for _ in 0..win_total {
        let mut off = 0usize;
        for d in 0..rank {
            // window_dilation spaces taps `wd` apart (atrous pooling); wd==1 (the
            // contiguous case) is bit-identical to the undilated stencil.
            off += tap_coord[d] * window_dilation[d] * input_strides[d];
        }
        tap_offsets.push(off);
        tap_coords.push(tap_coord.clone());
        let mut carry = true;
        for d in (0..rank).rev() {
            if carry {
                tap_coord[d] += 1;
                if tap_coord[d] >= window_dims[d] {
                    tap_coord[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    // `min` => logical AND (init true); every other op => logical OR (init false).
    let is_and = reduce_op == "min";
    let init = is_and;

    let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
    let mut out_idx = vec![0usize; rank];
    let mut values: Vec<bool> = Vec::with_capacity(total_output);

    for _ in 0..total_output {
        let mut interior = true;
        let mut base: isize = 0;
        for d in 0..rank {
            let corner = out_idx[d] as isize * strides[d] as isize - pad_lows[d] as isize;
            if corner < 0
                || corner + (window_dims[d] as isize - 1) * window_dilation[d] as isize
                    >= input_dims[d] as isize
            {
                interior = false;
                break;
            }
            base += corner * input_strides[d] as isize;
        }

        let mut acc = init;
        if interior {
            let base = base as usize;
            if is_and {
                for &off in &tap_offsets {
                    acc &= src[base + off];
                }
            } else {
                for &off in &tap_offsets {
                    acc |= src[base + off];
                }
            }
        } else {
            for coords in &tap_coords {
                let mut in_bounds = true;
                let mut flat = 0usize;
                for d in (0..rank).rev() {
                    let padded = out_idx[d] * strides[d] + coords[d] * window_dilation[d];
                    if padded < pad_lows[d] {
                        in_bounds = false;
                        break;
                    }
                    let ip = padded - pad_lows[d];
                    if ip >= input_dims[d] {
                        in_bounds = false;
                        break;
                    }
                    flat += ip * input_strides[d];
                }
                let tap = if in_bounds { src[flat] } else { init };
                if is_and {
                    acc &= tap;
                } else {
                    acc |= tap;
                }
            }
        }
        values.push(acc);

        let mut carry = true;
        for d in (0..rank).rev() {
            if carry {
                out_idx[d] += 1;
                if out_idx[d] >= out_dims_usize[d] {
                    out_idx[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    let shape = Shape {
        dims: out_dims.to_vec(),
    };
    Ok(Value::Tensor(
        TensorValue::new_bool_values(shape, values).map_err(EvalError::from)?,
    ))
}

#[inline]
fn reduce_window_new_extreme_dominates(old: f64, new: f64, is_max: bool) -> bool {
    debug_assert!(!old.is_nan());
    debug_assert!(!new.is_nan());

    if is_max {
        if new > old {
            return true;
        }
        if new < old {
            return false;
        }
        // `f64::max` selects +0 over -0. For other equal values, either bit
        // pattern is identical, so replacing the older index is safe.
        if new == 0.0 && old == 0.0 {
            return (old.is_sign_negative() && new.is_sign_positive())
                || old.to_bits() == new.to_bits();
        }
    } else {
        if new < old {
            return true;
        }
        if new > old {
            return false;
        }
        // `f64::min` selects -0 over +0.
        if new == 0.0 && old == 0.0 {
            return (old.is_sign_positive() && new.is_sign_negative())
                || old.to_bits() == new.to_bits();
        }
    }

    old.to_bits() == new.to_bits()
}

/// SIMD-over-channel max/min `reduce_window` for a channel-last layout: the LAST axis is a
/// contiguous feature/channel axis with window 1 (not pooled), the leading axes are pooled.
/// For each output position we max/min over the window taps, vectorized across the C channels
/// (f64x8, NaN-propagating to canonical `f64::NAN` to match `lax.max`/`lax.min`). This replaces
/// the scalar `eval_reduce_window_dense_float` odometer for the common CNN pooling case (2x2/3x3
/// windows that miss the separable-deque gate `win_total > 2·win_sum`), turning a ~6-9x JAX loss
/// into a win. Bit-identical to the odometer: max/min is order-independent, NaN propagates to the
/// same canonical NaN, OOB taps contribute the init (∓∞, a no-op). No base/window dilation.
/// One contiguous range of output positions [b_start, b_start + out_chunk.len()/c) for the f64
/// SIMD-channel maxpool: decodes each flat position to its pooled coords, maxes/mins over the
/// window taps (f64x8 across C, NaN→canonical), writing into the thread-local `out_chunk` (already
/// init-filled). Self-contained (no cross-position state) so it runs on an independent output slice.
#[allow(clippy::too_many_arguments)]
fn simd_channel_block_f64(
    src: &[f64],
    in_strides: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    input_dims: &[usize],
    c: usize,
    win_total: usize,
    is_max: bool,
    b_start: usize,
    out_chunk: &mut [f64],
) {
    use std::simd::cmp::SimdPartialEq;
    use std::simd::num::SimdFloat;
    use std::simd::{Mask, Select, Simd};
    type F = Simd<f64, 8>;
    let pooled = out_dims.len() - 1;
    let c8 = c - c % 8;
    let n_blocks = out_chunk.len() / c;
    let mut out_coord = vec![0usize; pooled];
    let mut tap = vec![0usize; pooled];
    for bi in 0..n_blocks {
        let mut rem = b_start + bi;
        for d in (0..pooled).rev() {
            out_coord[d] = rem % out_dims[d];
            rem /= out_dims[d];
        }
        let ob = bi * c;
        tap.iter_mut().for_each(|t| *t = 0);
        for _ in 0..win_total {
            let mut in_base = 0usize;
            let mut oob = false;
            for d in 0..pooled {
                let padded = out_coord[d] * strides[d] + tap[d];
                if padded < pad_lows[d] {
                    oob = true;
                    break;
                }
                let ic = padded - pad_lows[d];
                if ic >= input_dims[d] {
                    oob = true;
                    break;
                }
                in_base += ic * in_strides[d];
            }
            if !oob {
                let mut ci = 0;
                while ci < c8 {
                    let a = F::from_slice(&out_chunk[ob + ci..ob + ci + 8]);
                    let t = F::from_slice(&src[in_base + ci..in_base + ci + 8]);
                    let m = if is_max { a.simd_max(t) } else { a.simd_min(t) };
                    let nanm: Mask<i64, 8> = a.simd_ne(a) | t.simd_ne(t);
                    let res = nanm.select(F::splat(f64::NAN), m);
                    res.copy_to_slice(&mut out_chunk[ob + ci..ob + ci + 8]);
                    ci += 8;
                }
                while ci < c {
                    let a = out_chunk[ob + ci];
                    let t = src[in_base + ci];
                    out_chunk[ob + ci] = if a.is_nan() || t.is_nan() {
                        f64::NAN
                    } else if is_max {
                        a.max(t)
                    } else {
                        a.min(t)
                    };
                    ci += 1;
                }
            }
            for d in (0..pooled).rev() {
                tap[d] += 1;
                if tap[d] < window_dims[d] {
                    break;
                }
                tap[d] = 0;
            }
        }
    }
}

fn reduce_window_simd_channel_maxmin_f64(
    src: &[f64],
    input_dims: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    is_max: bool,
) -> Vec<f64> {
    let rank = input_dims.len();
    let pooled = rank - 1; // axes 0..pooled are pooled; the last axis is the channel
    let c = input_dims[pooled];
    let init = if is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let out_total: usize = out_dims.iter().product();
    let mut out = vec![init; out_total];
    let mut in_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        in_strides[d] = in_strides[d + 1] * input_dims[d + 1];
    }
    let out_spatial: usize = out_dims[..pooled].iter().product();
    let win_total: usize = window_dims[..pooled].iter().product();
    // The output positions are independent (disjoint contiguous c-blocks) → thread across them.
    // The per-position work (win_total taps × c/8 SIMD ops) is substantial, so unlike the deque
    // this amortizes thread-spawn for the large CNN feature maps. work_scaled_threads → 1 for small.
    let threads = crate::arithmetic::work_scaled_threads(out_total).min(out_spatial.max(1));
    if threads <= 1 {
        simd_channel_block_f64(
            src,
            &in_strides,
            window_dims,
            strides,
            pad_lows,
            out_dims,
            input_dims,
            c,
            win_total,
            is_max,
            0,
            &mut out,
        );
        return out;
    }
    let in_strides: &[usize] = &in_strides;
    let per = out_spatial.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut b0 = 0usize;
        let mut rest: &mut [f64] = out.as_mut_slice();
        while b0 < out_spatial {
            let cnt = per.min(out_spatial - b0);
            let (chunk, tail) = rest.split_at_mut(cnt * c);
            rest = tail;
            scope.spawn(move || {
                simd_channel_block_f64(
                    src,
                    in_strides,
                    window_dims,
                    strides,
                    pad_lows,
                    out_dims,
                    input_dims,
                    c,
                    win_total,
                    is_max,
                    b0,
                    chunk,
                );
            });
            b0 += cnt;
        }
    });
    out
}

/// f32 sibling of [`reduce_window_simd_channel_maxmin_f64`] — JAX's DEFAULT CNN dtype. Same
/// channel-last SIMD reduction, f32x16 across the contiguous feature axis (2× the f64 width).
/// NaN-propagating to canonical `f32::NAN`; signed-zero matches `f32::max`/`f32::min`.
/// f32 sibling of [`simd_channel_block_f64`] (f32x16). One independent output range.
#[allow(clippy::too_many_arguments)]
fn simd_channel_block_f32(
    src: &[f32],
    in_strides: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    input_dims: &[usize],
    c: usize,
    win_total: usize,
    is_max: bool,
    b_start: usize,
    out_chunk: &mut [f32],
) {
    use std::simd::cmp::SimdPartialEq;
    use std::simd::num::SimdFloat;
    use std::simd::{Mask, Select, Simd};
    type F = Simd<f32, 16>;
    let pooled = out_dims.len() - 1;
    let c16 = c - c % 16;
    let n_blocks = out_chunk.len() / c;
    let mut out_coord = vec![0usize; pooled];
    let mut tap = vec![0usize; pooled];
    for bi in 0..n_blocks {
        let mut rem = b_start + bi;
        for d in (0..pooled).rev() {
            out_coord[d] = rem % out_dims[d];
            rem /= out_dims[d];
        }
        let ob = bi * c;
        tap.iter_mut().for_each(|t| *t = 0);
        for _ in 0..win_total {
            let mut in_base = 0usize;
            let mut oob = false;
            for d in 0..pooled {
                let padded = out_coord[d] * strides[d] + tap[d];
                if padded < pad_lows[d] {
                    oob = true;
                    break;
                }
                let ic = padded - pad_lows[d];
                if ic >= input_dims[d] {
                    oob = true;
                    break;
                }
                in_base += ic * in_strides[d];
            }
            if !oob {
                let mut ci = 0;
                while ci < c16 {
                    let a = F::from_slice(&out_chunk[ob + ci..ob + ci + 16]);
                    let t = F::from_slice(&src[in_base + ci..in_base + ci + 16]);
                    let m = if is_max { a.simd_max(t) } else { a.simd_min(t) };
                    let nanm: Mask<i32, 16> = a.simd_ne(a) | t.simd_ne(t);
                    let res = nanm.select(F::splat(f32::NAN), m);
                    res.copy_to_slice(&mut out_chunk[ob + ci..ob + ci + 16]);
                    ci += 16;
                }
                while ci < c {
                    let a = out_chunk[ob + ci];
                    let t = src[in_base + ci];
                    out_chunk[ob + ci] = if a.is_nan() || t.is_nan() {
                        f32::NAN
                    } else if is_max {
                        a.max(t)
                    } else {
                        a.min(t)
                    };
                    ci += 1;
                }
            }
            for d in (0..pooled).rev() {
                tap[d] += 1;
                if tap[d] < window_dims[d] {
                    break;
                }
                tap[d] = 0;
            }
        }
    }
}

fn reduce_window_simd_channel_maxmin_f32(
    src: &[f32],
    input_dims: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    is_max: bool,
) -> Vec<f32> {
    let rank = input_dims.len();
    let pooled = rank - 1;
    let c = input_dims[pooled];
    let init = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let out_total: usize = out_dims.iter().product();
    let mut out = vec![init; out_total];
    let mut in_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        in_strides[d] = in_strides[d + 1] * input_dims[d + 1];
    }
    let out_spatial: usize = out_dims[..pooled].iter().product();
    let win_total: usize = window_dims[..pooled].iter().product();
    let threads = crate::arithmetic::work_scaled_threads(out_total).min(out_spatial.max(1));
    if threads <= 1 {
        simd_channel_block_f32(
            src,
            &in_strides,
            window_dims,
            strides,
            pad_lows,
            out_dims,
            input_dims,
            c,
            win_total,
            is_max,
            0,
            &mut out,
        );
        return out;
    }
    let in_strides: &[usize] = &in_strides;
    let per = out_spatial.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut b0 = 0usize;
        let mut rest: &mut [f32] = out.as_mut_slice();
        while b0 < out_spatial {
            let cnt = per.min(out_spatial - b0);
            let (chunk, tail) = rest.split_at_mut(cnt * c);
            rest = tail;
            scope.spawn(move || {
                simd_channel_block_f32(
                    src,
                    in_strides,
                    window_dims,
                    strides,
                    pad_lows,
                    out_dims,
                    input_dims,
                    c,
                    win_total,
                    is_max,
                    b0,
                    chunk,
                );
            });
            b0 += cnt;
        }
    });
    out
}

/// One output range for the f64 channel-last SUM reduce_window: accumulates the window taps in
/// ROW-MAJOR order across C channels (f64x8) into `out_chunk` (caller pre-fills 0.0). SIMD is across
/// channels (independent) so each channel's tap-sum order is unchanged — bit-identical to the scalar
/// `eval_reduce_window_dense_float` f64 accumulator despite float non-associativity.
#[allow(clippy::too_many_arguments)]
fn simd_channel_sum_block_f64(
    src: &[f64],
    in_strides: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    input_dims: &[usize],
    c: usize,
    win_total: usize,
    b_start: usize,
    out_chunk: &mut [f64],
) {
    use std::simd::Simd;
    type F = Simd<f64, 8>;
    let pooled = out_dims.len() - 1;
    let c8 = c - c % 8;
    let n_blocks = out_chunk.len() / c;
    let mut out_coord = vec![0usize; pooled];
    let mut tap = vec![0usize; pooled];
    for bi in 0..n_blocks {
        let mut rem = b_start + bi;
        for d in (0..pooled).rev() {
            out_coord[d] = rem % out_dims[d];
            rem /= out_dims[d];
        }
        let ob = bi * c;
        tap.iter_mut().for_each(|t| *t = 0);
        for _ in 0..win_total {
            let mut in_base = 0usize;
            let mut oob = false;
            for d in 0..pooled {
                let padded = out_coord[d] * strides[d] + tap[d];
                if padded < pad_lows[d] {
                    oob = true;
                    break;
                }
                let ic = padded - pad_lows[d];
                if ic >= input_dims[d] {
                    oob = true;
                    break;
                }
                in_base += ic * in_strides[d];
            }
            if !oob {
                let mut ci = 0;
                while ci < c8 {
                    let a = F::from_slice(&out_chunk[ob + ci..ob + ci + 8]);
                    let t = F::from_slice(&src[in_base + ci..in_base + ci + 8]);
                    (a + t).copy_to_slice(&mut out_chunk[ob + ci..ob + ci + 8]);
                    ci += 8;
                }
                while ci < c {
                    out_chunk[ob + ci] += src[in_base + ci];
                    ci += 1;
                }
            }
            for d in (0..pooled).rev() {
                tap[d] += 1;
                if tap[d] < window_dims[d] {
                    break;
                }
                tap[d] = 0;
            }
        }
    }
}

/// f32 channel-last SUM: accumulates in f64 (widening f32→f64 per tap, EXACTLY as dense_float's
/// f64-view path does), then rounds `as f32` — bit-identical to the scalar odometer. f64x8 across
/// channels; the per-channel row-major tap order is preserved.
#[allow(clippy::too_many_arguments)]
fn simd_channel_sum_block_f32(
    src: &[f32],
    in_strides: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    input_dims: &[usize],
    c: usize,
    win_total: usize,
    b_start: usize,
    out_chunk: &mut [f32],
) {
    use std::simd::Simd;
    use std::simd::num::SimdFloat;
    type F = Simd<f64, 8>;
    let pooled = out_dims.len() - 1;
    let c8 = c - c % 8;
    let n_blocks = out_chunk.len() / c;
    let mut out_coord = vec![0usize; pooled];
    let mut tap = vec![0usize; pooled];
    let mut acc = vec![0.0f64; c];
    for bi in 0..n_blocks {
        let mut rem = b_start + bi;
        for d in (0..pooled).rev() {
            out_coord[d] = rem % out_dims[d];
            rem /= out_dims[d];
        }
        let ob = bi * c;
        acc.iter_mut().for_each(|a| *a = 0.0);
        tap.iter_mut().for_each(|t| *t = 0);
        for _ in 0..win_total {
            let mut in_base = 0usize;
            let mut oob = false;
            for d in 0..pooled {
                let padded = out_coord[d] * strides[d] + tap[d];
                if padded < pad_lows[d] {
                    oob = true;
                    break;
                }
                let ic = padded - pad_lows[d];
                if ic >= input_dims[d] {
                    oob = true;
                    break;
                }
                in_base += ic * in_strides[d];
            }
            if !oob {
                let mut ci = 0;
                while ci < c8 {
                    let a = F::from_slice(&acc[ci..ci + 8]);
                    let t = Simd::<f32, 8>::from_slice(&src[in_base + ci..in_base + ci + 8])
                        .cast::<f64>();
                    (a + t).copy_to_slice(&mut acc[ci..ci + 8]);
                    ci += 8;
                }
                while ci < c {
                    acc[ci] += f64::from(src[in_base + ci]);
                    ci += 1;
                }
            }
            for d in (0..pooled).rev() {
                tap[d] += 1;
                if tap[d] < window_dims[d] {
                    break;
                }
                tap[d] = 0;
            }
        }
        for ci in 0..c {
            out_chunk[ob + ci] = acc[ci] as f32;
        }
    }
}

fn reduce_window_simd_channel_sum_f64(
    src: &[f64],
    input_dims: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
) -> Vec<f64> {
    let rank = input_dims.len();
    let pooled = rank - 1;
    let c = input_dims[pooled];
    let out_total: usize = out_dims.iter().product();
    let mut out = vec![0.0f64; out_total];
    let mut in_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        in_strides[d] = in_strides[d + 1] * input_dims[d + 1];
    }
    let out_spatial: usize = out_dims[..pooled].iter().product();
    let win_total: usize = window_dims[..pooled].iter().product();
    let threads = crate::arithmetic::work_scaled_threads(out_total).min(out_spatial.max(1));
    if threads <= 1 {
        simd_channel_sum_block_f64(
            src,
            &in_strides,
            window_dims,
            strides,
            pad_lows,
            out_dims,
            input_dims,
            c,
            win_total,
            0,
            &mut out,
        );
        return out;
    }
    let in_strides: &[usize] = &in_strides;
    let per = out_spatial.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut b0 = 0usize;
        let mut rest: &mut [f64] = out.as_mut_slice();
        while b0 < out_spatial {
            let cnt = per.min(out_spatial - b0);
            let (chunk, tail) = rest.split_at_mut(cnt * c);
            rest = tail;
            scope.spawn(move || {
                simd_channel_sum_block_f64(
                    src,
                    in_strides,
                    window_dims,
                    strides,
                    pad_lows,
                    out_dims,
                    input_dims,
                    c,
                    win_total,
                    b0,
                    chunk,
                );
            });
            b0 += cnt;
        }
    });
    out
}

fn reduce_window_simd_channel_sum_f32(
    src: &[f32],
    input_dims: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
) -> Vec<f32> {
    let rank = input_dims.len();
    let pooled = rank - 1;
    let c = input_dims[pooled];
    let out_total: usize = out_dims.iter().product();
    let mut out = vec![0.0f32; out_total];
    let mut in_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        in_strides[d] = in_strides[d + 1] * input_dims[d + 1];
    }
    let out_spatial: usize = out_dims[..pooled].iter().product();
    let win_total: usize = window_dims[..pooled].iter().product();
    let threads = crate::arithmetic::work_scaled_threads(out_total).min(out_spatial.max(1));
    if threads <= 1 {
        simd_channel_sum_block_f32(
            src,
            &in_strides,
            window_dims,
            strides,
            pad_lows,
            out_dims,
            input_dims,
            c,
            win_total,
            0,
            &mut out,
        );
        return out;
    }
    let in_strides: &[usize] = &in_strides;
    let per = out_spatial.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut b0 = 0usize;
        let mut rest: &mut [f32] = out.as_mut_slice();
        while b0 < out_spatial {
            let cnt = per.min(out_spatial - b0);
            let (chunk, tail) = rest.split_at_mut(cnt * c);
            rest = tail;
            scope.spawn(move || {
                simd_channel_sum_block_f32(
                    src,
                    in_strides,
                    window_dims,
                    strides,
                    pad_lows,
                    out_dims,
                    input_dims,
                    c,
                    win_total,
                    b0,
                    chunk,
                );
            });
            b0 += cnt;
        }
    });
    out
}

/// FUSED bf16 channel-last max/min pooling: widen bf16→f32 INLINE per tap (no f64 intermediate),
/// f32x16 NaN-aware max/min, running extremum kept AS bf16 bits (exact — max/min of bf16 is bf16).
/// One output range. Bit-identical to the widen→f64-pool→round odometer (max/min selects an exact
/// value; the canonical-NaN bf16 from `(f32_nan>>16)` matches `from_bf16_f64`, verified by guard).
#[allow(clippy::too_many_arguments)]
fn simd_channel_block_bf16_maxmin(
    bits: &[u16],
    in_strides: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    input_dims: &[usize],
    c: usize,
    win_total: usize,
    is_max: bool,
    b_start: usize,
    out_chunk: &mut [u16],
) {
    use std::simd::cmp::SimdPartialEq;
    use std::simd::num::{SimdFloat, SimdUint};
    use std::simd::{Mask, Select, Simd};
    type F = Simd<f32, 16>;
    type U16 = Simd<u16, 16>;
    type U32 = Simd<u32, 16>;
    let pooled = out_dims.len() - 1;
    let c16 = c - c % 16;
    let init_bits: u16 = if is_max { 0xFF80 } else { 0x7F80 }; // bf16 -inf / +inf
    out_chunk.fill(init_bits);
    let widen = |u: U16| F::from_bits(u.cast::<u32>() << U32::splat(16));
    let mut out_coord = vec![0usize; pooled];
    let mut tap = vec![0usize; pooled];
    for bi in 0..out_chunk.len() / c {
        let mut rem = b_start + bi;
        for d in (0..pooled).rev() {
            out_coord[d] = rem % out_dims[d];
            rem /= out_dims[d];
        }
        let ob = bi * c;
        tap.iter_mut().for_each(|t| *t = 0);
        for _ in 0..win_total {
            let mut in_base = 0usize;
            let mut oob = false;
            for d in 0..pooled {
                let padded = out_coord[d] * strides[d] + tap[d];
                if padded < pad_lows[d] {
                    oob = true;
                    break;
                }
                let ic = padded - pad_lows[d];
                if ic >= input_dims[d] {
                    oob = true;
                    break;
                }
                in_base += ic * in_strides[d];
            }
            if !oob {
                let mut ci = 0;
                while ci < c16 {
                    let a = widen(U16::from_slice(&out_chunk[ob + ci..ob + ci + 16]));
                    let t = widen(U16::from_slice(&bits[in_base + ci..in_base + ci + 16]));
                    let m = if is_max { a.simd_max(t) } else { a.simd_min(t) };
                    let nanm: Mask<i32, 16> = a.simd_ne(a) | t.simd_ne(t);
                    let res = nanm.select(F::splat(f32::NAN), m);
                    let ob2: U16 = (res.to_bits() >> U32::splat(16)).cast::<u16>();
                    ob2.copy_to_slice(&mut out_chunk[ob + ci..ob + ci + 16]);
                    ci += 16;
                }
                while ci < c {
                    let af = f32::from_bits((out_chunk[ob + ci] as u32) << 16);
                    let tf = f32::from_bits((bits[in_base + ci] as u32) << 16);
                    let r = if af.is_nan() || tf.is_nan() {
                        f32::NAN
                    } else if is_max {
                        af.max(tf)
                    } else {
                        af.min(tf)
                    };
                    out_chunk[ob + ci] = (r.to_bits() >> 16) as u16;
                    ci += 1;
                }
            }
            for d in (0..pooled).rev() {
                tap[d] += 1;
                if tap[d] < window_dims[d] {
                    break;
                }
                tap[d] = 0;
            }
        }
    }
}

/// FUSED bf16 channel-last SUM pooling: widen bf16→f64 INLINE per tap (f64x8), accumulate in f64
/// (matching the odometer's f64 accumulator + row-major tap order), round f64→bf16 once per output
/// via [`crate::arithmetic::round_f64_slice_to_bf16`]. Bit-identical to the odometer.
#[allow(clippy::too_many_arguments)]
fn simd_channel_block_bf16_sum(
    bits: &[u16],
    in_strides: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    input_dims: &[usize],
    c: usize,
    win_total: usize,
    b_start: usize,
    out_chunk: &mut [u16],
) {
    use std::simd::Simd;
    use std::simd::num::{SimdFloat, SimdUint};
    type F = Simd<f64, 8>;
    type U16 = Simd<u16, 8>;
    type U32 = Simd<u32, 8>;
    let pooled = out_dims.len() - 1;
    let c8 = c - c % 8;
    let widen = |u: U16| -> F {
        Simd::<f32, 8>::from_bits(u.cast::<u32>() << U32::splat(16)).cast::<f64>()
    };
    let mut out_coord = vec![0usize; pooled];
    let mut tap = vec![0usize; pooled];
    let mut acc = vec![0.0f64; c];
    for bi in 0..out_chunk.len() / c {
        let mut rem = b_start + bi;
        for d in (0..pooled).rev() {
            out_coord[d] = rem % out_dims[d];
            rem /= out_dims[d];
        }
        acc.iter_mut().for_each(|a| *a = 0.0);
        tap.iter_mut().for_each(|t| *t = 0);
        for _ in 0..win_total {
            let mut in_base = 0usize;
            let mut oob = false;
            for d in 0..pooled {
                let padded = out_coord[d] * strides[d] + tap[d];
                if padded < pad_lows[d] {
                    oob = true;
                    break;
                }
                let ic = padded - pad_lows[d];
                if ic >= input_dims[d] {
                    oob = true;
                    break;
                }
                in_base += ic * in_strides[d];
            }
            if !oob {
                let mut ci = 0;
                while ci < c8 {
                    let a = F::from_slice(&acc[ci..ci + 8]);
                    let t = widen(U16::from_slice(&bits[in_base + ci..in_base + ci + 8]));
                    (a + t).copy_to_slice(&mut acc[ci..ci + 8]);
                    ci += 8;
                }
                while ci < c {
                    acc[ci] += f32::from_bits((bits[in_base + ci] as u32) << 16) as f64;
                    ci += 1;
                }
            }
            for d in (0..pooled).rev() {
                tap[d] += 1;
                if tap[d] < window_dims[d] {
                    break;
                }
                tap[d] = 0;
            }
        }
        let rounded = crate::arithmetic::round_f64_slice_to_bf16(&acc);
        out_chunk[bi * c..bi * c + c].copy_from_slice(&rounded);
    }
}

/// FUSED f16 channel-last max/min pooling. f16 needs the IEEE decode (unlike bf16's shift), so it
/// widens each tap via the branchless [`crate::arithmetic::f16_widen8_full_f32`] (all of
/// normal/subnormal/±0/inf/NaN, no per-chunk fast/slow split) into an f32 running-max SCRATCH, then
/// narrows ONCE per output to f16 bits (the max of f16 values IS an f16 value → exact, no SIMD
/// f32→f16 narrow). Bit-identical to the widen-to-f64 odometer (verified vs it incl ±0/NaN).
#[allow(clippy::too_many_arguments)]
fn simd_channel_block_f16_maxmin(
    bits: &[u16],
    in_strides: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    input_dims: &[usize],
    c: usize,
    win_total: usize,
    is_max: bool,
    b_start: usize,
    out_chunk: &mut [u16],
) {
    use std::simd::cmp::SimdPartialEq;
    use std::simd::num::SimdFloat;
    use std::simd::{Select, Simd};
    type F = Simd<f32, 8>;
    let pooled = out_dims.len() - 1;
    let c8 = c - c % 8;
    let init = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let narrow = |v: f32| -> u16 {
        match Literal::from_f16_f64(v as f64) {
            Literal::F16Bits(b) => b,
            _ => 0,
        }
    };
    let mut scratch = vec![init; c];
    let mut out_coord = vec![0usize; pooled];
    let mut tap = vec![0usize; pooled];
    for bi in 0..out_chunk.len() / c {
        let mut rem = b_start + bi;
        for d in (0..pooled).rev() {
            out_coord[d] = rem % out_dims[d];
            rem /= out_dims[d];
        }
        scratch.iter_mut().for_each(|s| *s = init);
        tap.iter_mut().for_each(|t| *t = 0);
        for _ in 0..win_total {
            let mut in_base = 0usize;
            let mut oob = false;
            for d in 0..pooled {
                let padded = out_coord[d] * strides[d] + tap[d];
                if padded < pad_lows[d] {
                    oob = true;
                    break;
                }
                let ic = padded - pad_lows[d];
                if ic >= input_dims[d] {
                    oob = true;
                    break;
                }
                in_base += ic * in_strides[d];
            }
            if !oob {
                let mut ci = 0;
                while ci < c8 {
                    let a = F::from_slice(&scratch[ci..ci + 8]);
                    let t = crate::arithmetic::f16_widen8_full_f32(Simd::<u16, 8>::from_slice(
                        &bits[in_base + ci..in_base + ci + 8],
                    ));
                    let m = if is_max { a.simd_max(t) } else { a.simd_min(t) };
                    let nanm = a.simd_ne(a) | t.simd_ne(t);
                    nanm.select(F::splat(f32::NAN), m)
                        .copy_to_slice(&mut scratch[ci..ci + 8]);
                    ci += 8;
                }
                while ci < c {
                    let af = scratch[ci];
                    let tf = Literal::F16Bits(bits[in_base + ci]).as_f64().unwrap_or(0.0) as f32;
                    scratch[ci] = if af.is_nan() || tf.is_nan() {
                        f32::NAN
                    } else if is_max {
                        af.max(tf)
                    } else {
                        af.min(tf)
                    };
                    ci += 1;
                }
            }
            for d in (0..pooled).rev() {
                tap[d] += 1;
                if tap[d] < window_dims[d] {
                    break;
                }
                tap[d] = 0;
            }
        }
        let ob = bi * c;
        for ci in 0..c {
            out_chunk[ob + ci] = narrow(scratch[ci]);
        }
    }
}

/// FUSED f16 channel-last SUM pooling: widen each tap f16→f64 via the branchless
/// [`crate::arithmetic::f16_widen8_full_f32`] (then `f32→f64`, exact), accumulate in f64 in the
/// odometer's row-major tap order, round f64→f16 ONCE per output via `Literal::from_f16_f64` —
/// EXACTLY the odometer's widen+accumulate+round, so bit-identical.
#[allow(clippy::too_many_arguments)]
fn simd_channel_block_f16_sum(
    bits: &[u16],
    in_strides: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    input_dims: &[usize],
    c: usize,
    win_total: usize,
    b_start: usize,
    out_chunk: &mut [u16],
) {
    use std::simd::Simd;
    use std::simd::num::SimdFloat;
    type F = Simd<f64, 8>;
    let pooled = out_dims.len() - 1;
    let c8 = c - c % 8;
    let mut out_coord = vec![0usize; pooled];
    let mut tap = vec![0usize; pooled];
    let mut acc = vec![0.0f64; c];
    for bi in 0..out_chunk.len() / c {
        let mut rem = b_start + bi;
        for d in (0..pooled).rev() {
            out_coord[d] = rem % out_dims[d];
            rem /= out_dims[d];
        }
        acc.iter_mut().for_each(|a| *a = 0.0);
        tap.iter_mut().for_each(|t| *t = 0);
        for _ in 0..win_total {
            let mut in_base = 0usize;
            let mut oob = false;
            for d in 0..pooled {
                let padded = out_coord[d] * strides[d] + tap[d];
                if padded < pad_lows[d] {
                    oob = true;
                    break;
                }
                let ic = padded - pad_lows[d];
                if ic >= input_dims[d] {
                    oob = true;
                    break;
                }
                in_base += ic * in_strides[d];
            }
            if !oob {
                let mut ci = 0;
                while ci < c8 {
                    let a = F::from_slice(&acc[ci..ci + 8]);
                    let t = crate::arithmetic::f16_widen8_full_f32(Simd::<u16, 8>::from_slice(
                        &bits[in_base + ci..in_base + ci + 8],
                    ))
                    .cast::<f64>();
                    (a + t).copy_to_slice(&mut acc[ci..ci + 8]);
                    ci += 8;
                }
                while ci < c {
                    acc[ci] += Literal::F16Bits(bits[in_base + ci]).as_f64().unwrap_or(0.0);
                    ci += 1;
                }
            }
            for d in (0..pooled).rev() {
                tap[d] += 1;
                if tap[d] < window_dims[d] {
                    break;
                }
                tap[d] = 0;
            }
        }
        let ob = bi * c;
        for ci in 0..c {
            out_chunk[ob + ci] = match Literal::from_f16_f64(acc[ci]) {
                Literal::F16Bits(b) => b,
                _ => 0,
            };
        }
    }
}

/// Threaded fused f16 channel-last pooling (max/min/sum), threading independent output blocks.
fn reduce_window_simd_channel_f16(
    bits: &[u16],
    input_dims: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    reduce_op: &str,
) -> Vec<u16> {
    let is_sum = reduce_window_sum_like(reduce_op);
    let is_max = reduce_op == "max";
    let rank = input_dims.len();
    let pooled = rank - 1;
    let c = input_dims[pooled];
    let out_total: usize = out_dims.iter().product();
    let mut out = vec![0u16; out_total];
    let mut in_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        in_strides[d] = in_strides[d + 1] * input_dims[d + 1];
    }
    let win_total: usize = window_dims[..pooled].iter().product();
    let out_spatial = out_total / c;
    let run = |b0: usize, chunk: &mut [u16]| {
        if is_sum {
            simd_channel_block_f16_sum(
                bits,
                &in_strides,
                window_dims,
                strides,
                pad_lows,
                out_dims,
                input_dims,
                c,
                win_total,
                b0,
                chunk,
            );
        } else {
            simd_channel_block_f16_maxmin(
                bits,
                &in_strides,
                window_dims,
                strides,
                pad_lows,
                out_dims,
                input_dims,
                c,
                win_total,
                is_max,
                b0,
                chunk,
            );
        }
    };
    let threads = crate::arithmetic::work_scaled_threads(out_total).min(out_spatial.max(1));
    if threads <= 1 || out_spatial < 2 {
        run(0, &mut out);
        return out;
    }
    let per = out_spatial.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [u16] = out.as_mut_slice();
        let mut b0 = 0usize;
        while b0 < out_spatial {
            let rows = per.min(out_spatial - b0);
            let (blk, tail) = rest.split_at_mut(rows * c);
            rest = tail;
            let start = b0;
            b0 += rows;
            scope.spawn(move || run(start, blk));
        }
    });
    out
}

/// Threaded fused bf16 channel-last pooling (max/min/sum), threading independent output blocks.
fn reduce_window_simd_channel_bf16(
    bits: &[u16],
    input_dims: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    reduce_op: &str,
) -> Vec<u16> {
    let rank = input_dims.len();
    let pooled = rank - 1;
    let c = input_dims[pooled];
    let out_total: usize = out_dims.iter().product();
    let mut out = vec![0u16; out_total];
    let mut in_strides = vec![1usize; rank];
    for d in (0..rank - 1).rev() {
        in_strides[d] = in_strides[d + 1] * input_dims[d + 1];
    }
    let out_spatial: usize = out_dims[..pooled].iter().product();
    let win_total: usize = window_dims[..pooled].iter().product();
    let is_sum = reduce_window_sum_like(reduce_op);
    let is_max = reduce_op == "max";
    let run = |b0: usize, chunk: &mut [u16]| {
        if is_sum {
            simd_channel_block_bf16_sum(
                bits,
                &in_strides,
                window_dims,
                strides,
                pad_lows,
                out_dims,
                input_dims,
                c,
                win_total,
                b0,
                chunk,
            );
        } else {
            simd_channel_block_bf16_maxmin(
                bits,
                &in_strides,
                window_dims,
                strides,
                pad_lows,
                out_dims,
                input_dims,
                c,
                win_total,
                is_max,
                b0,
                chunk,
            );
        }
    };
    let threads = crate::arithmetic::work_scaled_threads(out_total).min(out_spatial.max(1));
    if threads <= 1 {
        run(0, out.as_mut_slice());
        return out;
    }
    let run_ref = &run;
    let per = out_spatial.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut b0 = 0usize;
        let mut rest: &mut [u16] = out.as_mut_slice();
        while b0 < out_spatial {
            let cnt = per.min(out_spatial - b0);
            let (chunk, tail) = rest.split_at_mut(cnt * c);
            rest = tail;
            scope.spawn(move || run_ref(b0, chunk));
            b0 += cnt;
        }
    });
    out
}

/// Separable sliding-window max/min: reduce each window axis with an independent
/// monotonic-deque 1D pass. This turns the O(output·∏window) full-window scan and
/// the older O(output·∑window) per-axis scan into O(input+output) work per axis.
/// Out-of-bounds taps contribute the reduction init (∓∞), a no-op for max/min.
/// NaNs are tracked separately so every window containing at least one NaN yields
/// the same canonical NaN as `jax_max_f64`/`jax_min_f64`; signed zero preference
/// matches `f64::max`/`f64::min`.
fn reduce_window_separable_maxmin(
    src: &[f64],
    input_dims: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    is_max: bool,
) -> Vec<f64> {
    let init = if is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let rank = input_dims.len();
    let mut cur = src.to_vec();
    let mut cur_dims = input_dims.to_vec();
    for d in 0..rank {
        // Identity axis (window 1, stride 1, no padding): output == input along d.
        if window_dims[d] == 1 && strides[d] == 1 && pad_lows[d] == 0 && cur_dims[d] == out_dims[d]
        {
            continue;
        }
        let l = cur_dims[d];
        let out_l = out_dims[d];
        let inner: usize = cur_dims[d + 1..].iter().product();
        let outer: usize = cur_dims[..d].iter().product();
        let (wd, sd, pd) = (window_dims[d], strides[d], pad_lows[d]);
        let mut next = vec![init; outer * out_l * inner];
        // NOTE: threading the outer blocks here was a measured NO-SHIP (0.62-0.75x regression,
        // `bench_maxpool_rank2_direct_vs_deque`): the per-pass work (~58k elems / ~0.9ms for a
        // 256x256/15x15 pool) is too small to amortize thread-spawn, and the leading axis has
        // outer==1 (no parallelism). The real ~2x JAX gap is SIMD-vectorization of the window
        // reduction, not parallelism — left serial.
        for o in 0..outer {
            let cur_o = o * l * inner;
            let next_o = o * out_l * inner;
            for lane in 0..inner {
                let line_base = cur_o + lane;
                let out_base = next_o + lane;
                let mut deque: Vec<usize> = Vec::with_capacity(l);
                let mut nan_positions: Vec<usize> = Vec::new();
                let mut head = 0usize;
                let mut nan_head = 0usize;
                let mut pushed_until = 0usize;
                for ol in 0..out_l {
                    let raw_start = ol as isize * sd as isize - pd as isize;
                    let raw_end = raw_start + wd as isize;
                    let start = raw_start.clamp(0, l as isize) as usize;
                    let end = raw_end.clamp(0, l as isize) as usize;

                    while pushed_until < end {
                        let v = cur[line_base + pushed_until * inner];
                        if v.is_nan() {
                            nan_positions.push(pushed_until);
                        } else {
                            while deque.len() > head {
                                let Some(&back) = deque.last() else {
                                    break;
                                };
                                let back_value = cur[line_base + back * inner];
                                if reduce_window_new_extreme_dominates(back_value, v, is_max) {
                                    deque.pop();
                                } else {
                                    break;
                                }
                            }
                            if deque.len() == head {
                                deque.clear();
                                head = 0;
                            }
                            deque.push(pushed_until);
                        }
                        pushed_until += 1;
                    }

                    while deque.get(head).is_some_and(|&idx| idx < start) {
                        head += 1;
                    }
                    while nan_positions.get(nan_head).is_some_and(|&idx| idx < start) {
                        nan_head += 1;
                    }

                    next[out_base + ol * inner] = if nan_head < nan_positions.len() {
                        f64::NAN
                    } else if let Some(&idx) = deque.get(head) {
                        cur[line_base + idx * inner]
                    } else {
                        init
                    }
                }
            }
        }
        cur = next;
        cur_dims[d] = out_l;
    }
    cur
}

/// N-D separable sliding-window SUM for dense float (f64 view; caller narrows to the output
/// dtype). The float sibling of `reduce_window_separable_maxmin`: reduces each WINDOWED axis with
/// an independent 1-D pass — a per-line PREFIX SUM then `pre[end]-pre[start]` per output — so it
/// handles ANY stride/padding (OOB taps are the sum identity 0 ⇒ clamp the window to `[0,l)`) at
/// O(input+output) per axis vs the per-window O(output·∏window) dense path. Generalises the
/// rank-3 separable to any rank (the realistic 3-D-CNN avg-pool is rank-5 `[N,D,H,W,C]`, windows
/// on D/H/W). Tolerance-equal (running/prefix sum reassociates the adds); accumulates in f64.
fn reduce_window_separable_sum_nd_f64(
    src: &[f64],
    input_dims: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
) -> Vec<f64> {
    let rank = input_dims.len();
    let mut cur = src.to_vec();
    let mut cur_dims = input_dims.to_vec();
    for d in 0..rank {
        // Identity axis (window 1, stride 1, no padding): output == input along d.
        if window_dims[d] == 1 && strides[d] == 1 && pad_lows[d] == 0 && cur_dims[d] == out_dims[d]
        {
            continue;
        }
        let l = cur_dims[d];
        let out_l = out_dims[d];
        let inner: usize = cur_dims[d + 1..].iter().product();
        let outer: usize = cur_dims[..d].iter().product();
        let (wd, sd, pd) = (window_dims[d], strides[d], pad_lows[d]);
        let mut next = vec![0.0f64; outer * out_l * inner];
        // Block-prefix over `inner`-wide CONTIGUOUS blocks: prefix[k][..] = Σ_{j<k} cur[block j].
        // Operating on the contiguous inner block (rather than one strided scalar lane at a time)
        // keeps the hot loop sequential + autovectorizable — critical for channel-last layouts
        // where `inner` (e.g. C·…) is large and per-lane striding would thrash cache.
        let mut prefix = vec![0.0f64; (l + 1) * inner];
        for o in 0..outer {
            let cur_o = o * l * inner;
            let next_o = o * out_l * inner;
            // prefix[0] block is zero; accumulate forward.
            for v in prefix[0..inner].iter_mut() {
                *v = 0.0;
            }
            for k in 0..l {
                let (lo, hi) = prefix.split_at_mut((k + 1) * inner);
                let prev = &lo[k * inner..(k + 1) * inner];
                let nextp = &mut hi[0..inner];
                let row = &cur[cur_o + k * inner..cur_o + (k + 1) * inner];
                for ((p, &pv), &rv) in nextp.iter_mut().zip(prev).zip(row) {
                    *p = pv + rv;
                }
            }
            for ol in 0..out_l {
                let raw_start = ol as isize * sd as isize - pd as isize;
                let raw_end = raw_start + wd as isize;
                let start = raw_start.clamp(0, l as isize) as usize;
                let end = raw_end.clamp(0, l as isize) as usize;
                let lo = &prefix[start * inner..(start + 1) * inner];
                let hi = &prefix[end * inner..(end + 1) * inner];
                let out_blk = &mut next[next_o + ol * inner..next_o + (ol + 1) * inner];
                for ((o_, &h), &l_) in out_blk.iter_mut().zip(hi).zip(lo) {
                    *o_ = h - l_;
                }
            }
        }
        cur = next;
        cur_dims[d] = out_l;
    }
    cur
}

/// i64 sibling of [`reduce_window_separable_sum_nd_f64`] for N-D integer sum pooling. Integer add
/// is a ring (associative even on overflow), so the block-prefix with `wrapping_add`/`wrapping_sub`
/// is BIT-IDENTICAL to the naive `wrapping_add` fold in `eval_reduce_window_dense_i64` — no
/// tolerance. Generalizes the rank-3 i64 separable to any rank (e.g. rank-5 integer volumetric
/// pooling), handling any stride/padding (OOB taps = 0).
fn reduce_window_separable_sum_nd_i64(
    src: &[i64],
    input_dims: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
) -> Vec<i64> {
    let rank = input_dims.len();
    let mut cur = src.to_vec();
    let mut cur_dims = input_dims.to_vec();
    for d in 0..rank {
        if window_dims[d] == 1 && strides[d] == 1 && pad_lows[d] == 0 && cur_dims[d] == out_dims[d]
        {
            continue;
        }
        let l = cur_dims[d];
        let out_l = out_dims[d];
        let inner: usize = cur_dims[d + 1..].iter().product();
        let outer: usize = cur_dims[..d].iter().product();
        let (wd, sd, pd) = (window_dims[d], strides[d], pad_lows[d]);
        let mut next = vec![0i64; outer * out_l * inner];
        let mut prefix = vec![0i64; (l + 1) * inner];
        for o in 0..outer {
            let cur_o = o * l * inner;
            let next_o = o * out_l * inner;
            for v in prefix[0..inner].iter_mut() {
                *v = 0;
            }
            for k in 0..l {
                let (lo, hi) = prefix.split_at_mut((k + 1) * inner);
                let prev = &lo[k * inner..(k + 1) * inner];
                let nextp = &mut hi[0..inner];
                let row = &cur[cur_o + k * inner..cur_o + (k + 1) * inner];
                for ((p, &pv), &rv) in nextp.iter_mut().zip(prev).zip(row) {
                    *p = pv.wrapping_add(rv);
                }
            }
            for ol in 0..out_l {
                let raw_start = ol as isize * sd as isize - pd as isize;
                let raw_end = raw_start + wd as isize;
                let start = raw_start.clamp(0, l as isize) as usize;
                let end = raw_end.clamp(0, l as isize) as usize;
                let lo = &prefix[start * inner..(start + 1) * inner];
                let hi = &prefix[end * inner..(end + 1) * inner];
                let out_blk = &mut next[next_o + ol * inner..next_o + (ol + 1) * inner];
                for ((o_, &h), &l_) in out_blk.iter_mut().zip(hi).zip(lo) {
                    *o_ = h.wrapping_sub(l_);
                }
            }
        }
        cur = next;
        cur_dims[d] = out_l;
    }
    cur
}

/// Separable sliding-window max/min for dense i64 — the integer sibling of
/// `reduce_window_separable_maxmin`. Reduces each window axis with an independent
/// monotonic-deque 1D pass: O(input+output) per axis, WINDOW-INDEPENDENT, vs the
/// per-window O(output·∏window) dense path. i64 max/min is a total order (no NaN /
/// signed-zero), so the window extremum VALUE is unique and the deque result is
/// bit-identical to the ascending per-window `i64::max`/`i64::min` fold regardless
/// of order; OOB taps contribute the init (`i64::MIN`/`i64::MAX`), a no-op.
fn reduce_window_separable_maxmin_i64(
    src: &[i64],
    input_dims: &[usize],
    window_dims: &[usize],
    strides: &[usize],
    pad_lows: &[usize],
    out_dims: &[usize],
    is_max: bool,
) -> Vec<i64> {
    let init = if is_max { i64::MIN } else { i64::MAX };
    let rank = input_dims.len();
    let mut cur = src.to_vec();
    let mut cur_dims = input_dims.to_vec();
    for d in 0..rank {
        if window_dims[d] == 1 && strides[d] == 1 && pad_lows[d] == 0 && cur_dims[d] == out_dims[d]
        {
            continue;
        }
        let l = cur_dims[d];
        let out_l = out_dims[d];
        let inner: usize = cur_dims[d + 1..].iter().product();
        let outer: usize = cur_dims[..d].iter().product();
        let (wd, sd, pd) = (window_dims[d], strides[d], pad_lows[d]);
        let mut next = vec![init; outer * out_l * inner];
        for o in 0..outer {
            let cur_o = o * l * inner;
            let next_o = o * out_l * inner;
            for lane in 0..inner {
                let line_base = cur_o + lane;
                let out_base = next_o + lane;
                let mut deque: Vec<usize> = Vec::with_capacity(l);
                let mut head = 0usize;
                let mut pushed_until = 0usize;
                for ol in 0..out_l {
                    let raw_start = ol as isize * sd as isize - pd as isize;
                    let raw_end = raw_start + wd as isize;
                    let start = raw_start.clamp(0, l as isize) as usize;
                    let end = raw_end.clamp(0, l as isize) as usize;

                    while pushed_until < end {
                        let v = cur[line_base + pushed_until * inner];
                        while deque.len() > head {
                            let Some(&back) = deque.last() else {
                                break;
                            };
                            let back_value = cur[line_base + back * inner];
                            let dominated = if is_max {
                                back_value <= v
                            } else {
                                back_value >= v
                            };
                            if dominated {
                                deque.pop();
                            } else {
                                break;
                            }
                        }
                        if deque.len() == head {
                            deque.clear();
                            head = 0;
                        }
                        deque.push(pushed_until);
                        pushed_until += 1;
                    }

                    while deque.get(head).is_some_and(|&idx| idx < start) {
                        head += 1;
                    }

                    next[out_base + ol * inner] = if let Some(&idx) = deque.get(head) {
                        cur[line_base + idx * inner]
                    } else {
                        init
                    };
                }
            }
        }
        cur = next;
        cur_dims[d] = out_l;
    }
    cur
}

/// Evaluate ReduceWindow: apply a reduction over sliding windows of a tensor.
///
/// inputs: [tensor]
/// params:
///   - "reduce_op": "sum", "max", "min" (default: "sum")
///   - "window_dimensions": comma-separated window sizes per dimension, e.g. "2,2"
///   - "window_strides": comma-separated strides, e.g. "1,1" (default: all 1s)
///   - "padding": "VALID", "SAME", or "SAME_LOWER" (default: "VALID")
///
/// Returns the reduced tensor with output shape determined by window/stride/padding.
fn eval_reduce_window(
    primitive: Primitive,
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    if inputs.is_empty() {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: 0,
        });
    }

    let tensor = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => return Ok(inputs[0].clone()), // scalar passthrough
    };

    let rank = tensor.shape.rank();

    let window_dims = parse_reduce_window_param(primitive, params, "window_dimensions", rank, 2)?;
    let strides = parse_reduce_window_param(primitive, params, "window_strides", rank, 1)?;

    // Determine reduction operation
    let reduce_op = params.get("reduce_op").map(|s| s.as_str()).unwrap_or("sum");
    // Fail closed on unknown reducers: only sum/max/min are implemented. Without this an
    // unrecognized reduce_op (e.g. "mul"/"prod" or a typo) silently falls through to the
    // sum accumulator (init 0, wrapping_add), computing a wrong result with no error.
    if !matches!(reduce_op, "sum" | "max" | "min") {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "unsupported reduce_window reduce_op {reduce_op:?}; expected \"sum\", \"max\", or \"min\""
            ),
        });
    }

    let padding = parse_reduce_window_padding(primitive, params)?;

    // window_dilation (atrous/dilated pooling) IS supported: a dilation `d` spaces the
    // window taps `d` apart, so the window's effective extent is `(win-1)*d+1` (used for
    // the output size and SAME padding) and tap `t` reads input `out*stride + t*d - pad`.
    // base_dilation (operand/input dilation) IS ALSO supported: a dilation `db` spaces the
    // BASE (operand) elements `db` apart, so the dilated base extent is `(n-1)*db+1` (used
    // for the output size + SAME padding). A position `p` in the dilated base maps to real
    // element `p/db` when `p % db == 0`, otherwise it is a HOLE filled with the reduction
    // init value — exactly as XLA dilates the base for `reduce_window` (and identical to how
    // the generic loop below already treats out-of-bounds/padding taps). The dense/rank-2
    // fast paths assume `db == 1`, so they are each gated on `no_base_dilation` below; a
    // dilated base falls through to the generic windowing loop.
    let base_dilation = parse_reduce_window_param(primitive, params, "base_dilation", rank, 1)?;
    let no_base_dilation = base_dilation.iter().all(|&d| d == 1);
    let window_dilation = parse_reduce_window_param(primitive, params, "window_dilation", rank, 1)?;
    let no_window_dilation = window_dilation.iter().all(|&d| d == 1);
    // Effective (dilated) window extent per dim, for output-size + SAME-pad geometry.
    let eff_window_dims: Vec<usize> = window_dims
        .iter()
        .zip(&window_dilation)
        .map(|(&w, &d)| (w.max(1) - 1) * d + 1)
        .collect();
    // Effective (dilated) BASE extent per dim: `(n-1)*db + 1` (= `n` when `db == 1`, so the
    // geometry is bit-identical to the undilated path in that case).
    let dilated_input_dims: Vec<usize> = (0..rank)
        .map(|d| {
            let n = tensor.shape.dims[d] as usize;
            if n == 0 {
                0
            } else {
                (n - 1) * base_dilation[d] + 1
            }
        })
        .collect();

    let output_dtype = reduce_window_output_dtype(tensor.dtype);

    if let ReduceWindowPadding::Explicit(pairs) = &padding
        && pairs.len() != rank
    {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "reduce_window explicit padding rank {} does not match tensor rank {rank}",
                pairs.len()
            ),
        });
    }

    // Calculate output dimensions
    let mut out_dims: Vec<u32> = Vec::with_capacity(rank);
    let mut pad_lows: Vec<usize> = Vec::with_capacity(rank);
    for d in 0..rank {
        let input_dim = dilated_input_dims[d];
        let win = eff_window_dims[d];
        let stride = strides[d];
        let (out_dim, pad_low) = match &padding {
            ReduceWindowPadding::Same => {
                reduce_window_same_geometry(primitive, input_dim, win, stride, false)?
            }
            ReduceWindowPadding::SameLower => {
                reduce_window_same_geometry(primitive, input_dim, win, stride, true)?
            }
            ReduceWindowPadding::Explicit(pairs) => {
                let &(pad_low, pad_high) = pairs.get(d).ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!("reduce_window explicit padding missing axis {d}"),
                })?;
                reduce_window_explicit_geometry(
                    primitive, input_dim, win, stride, pad_low, pad_high,
                )?
            }
            ReduceWindowPadding::Valid => {
                let out_dim = if input_dim >= win {
                    (input_dim - win) / stride + 1
                } else {
                    0
                };
                (out_dim, 0)
            }
        };
        let out_dim = u32::try_from(out_dim).map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("reduce_window output dimension {out_dim} exceeds u32::MAX"),
        })?;
        out_dims.push(out_dim);
        pad_lows.push(pad_low);
    }

    let total_output: usize = out_dims.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim as usize)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: "reduce_window output element count overflow".to_owned(),
            })
    })?;
    if total_output == 0 {
        return Ok(Value::Tensor(
            TensorValue::new(
                tensor.dtype,
                Shape {
                    dims: out_dims.clone(),
                },
                vec![],
            )
            .map_err(EvalError::InvalidTensor)?,
        ));
    }

    // Separable max/min fast path (ANY rank): when ∏window is meaningfully larger than
    // ∑window, reduce each window axis with an independent 1D pass — O(output·∑window)
    // instead of O(output·∏window). Bit-identical for max/min (see
    // reduce_window_separable_maxmin). Gated above 2·∑window so the pass/buffer overhead
    // pays (small windows like 2×2/3×3 stay on the single-pass stencil paths below).
    if no_base_dilation
        && no_window_dilation
        && matches!(reduce_op, "max" | "min")
        && matches!(
            output_dtype,
            fj_core::DType::F64 | fj_core::DType::F32 | fj_core::DType::BF16 | fj_core::DType::F16
        )
    {
        let win_total: usize = window_dims.iter().product();
        let win_sum: usize = window_dims.iter().sum();
        if win_total > 2 * win_sum
            && let Some(src) = reduce_window_dense_f64_view(tensor)
        {
            let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
            let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
            let values = reduce_window_separable_maxmin(
                src.as_ref(),
                &input_dims,
                &window_dims,
                &strides,
                &pad_lows,
                &out_dims_usize,
                reduce_op == "max",
            );
            let shape = Shape {
                dims: out_dims.clone(),
            };
            return match output_dtype {
                fj_core::DType::F32 => {
                    let f32_values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                    Ok(Value::Tensor(
                        TensorValue::new_f32_values(shape, f32_values).map_err(EvalError::from)?,
                    ))
                }
                // bf16/f16 max/min SELECT an exactly-widened input value (or the
                // ∓∞/NaN init for all-pad/NaN windows), so rounding back via the
                // SAME reduce_window_literal_from_f64 is exact — bit-identical to the
                // generic half path. Window-independent (deque) vs the dense_float
                // per-window O(∏window) path for large windows.
                fj_core::DType::BF16 | fj_core::DType::F16 => {
                    let bits: Vec<u16> = values
                        .iter()
                        .map(|&v| match reduce_window_literal_from_f64(output_dtype, v) {
                            fj_core::Literal::BF16Bits(b) | fj_core::Literal::F16Bits(b) => b,
                            _ => 0,
                        })
                        .collect();
                    Ok(Value::Tensor(
                        TensorValue::new_half_float_values(output_dtype, shape, bits)
                            .map_err(EvalError::from)?,
                    ))
                }
                _ => Ok(Value::Tensor(
                    TensorValue::new_f64_values(shape, values).map_err(EvalError::from)?,
                )),
            };
        }
    }

    // SIMD-over-channel max/min fast path: channel-last layout (last axis = contiguous feature
    // axis with window 1, not pooled), small pooled windows that miss the deque gate above.
    // Replaces the scalar dense_float odometer with f64x8 SIMD across channels — the common CNN
    // 2x2/3x3 maxpool case (was ~6-9x JAX loss → win). max/min order-independent + NaN→canonical,
    // bit-identical to the odometer.
    if no_base_dilation
        && no_window_dilation
        && tensor.dtype == fj_core::DType::F64
        && matches!(reduce_op, "max" | "min")
        && rank >= 2
        && window_dims[rank - 1] == 1
        && strides[rank - 1] == 1
        && pad_lows[rank - 1] == 0
        && out_dims[rank - 1] as usize == tensor.shape.dims[rank - 1] as usize
        && let Some(src) = reduce_window_dense_f64_view(tensor)
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
        let values = reduce_window_simd_channel_maxmin_f64(
            src.as_ref(),
            &input_dims,
            &window_dims,
            &strides,
            &pad_lows,
            &out_dims_usize,
            reduce_op == "max",
        );
        let shape = Shape {
            dims: out_dims.clone(),
        };
        return Ok(Value::Tensor(
            TensorValue::new_f64_values(shape, values).map_err(EvalError::from)?,
        ));
    }

    // f32 sibling of the SIMD-over-channel path above — JAX's DEFAULT CNN dtype (was ~57x JAX loss
    // on the scalar odometer; JAX f32 maxpool is 16-wide). f32x16 across the contiguous channel.
    if no_base_dilation
        && no_window_dilation
        && tensor.dtype == fj_core::DType::F32
        && matches!(reduce_op, "max" | "min")
        && rank >= 2
        && window_dims[rank - 1] == 1
        && strides[rank - 1] == 1
        && pad_lows[rank - 1] == 0
        && out_dims[rank - 1] as usize == tensor.shape.dims[rank - 1] as usize
        && let Some(src) = tensor.elements.as_f32_slice()
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
        let values = reduce_window_simd_channel_maxmin_f32(
            src,
            &input_dims,
            &window_dims,
            &strides,
            &pad_lows,
            &out_dims_usize,
            reduce_op == "max",
        );
        let shape = Shape {
            dims: out_dims.clone(),
        };
        return Ok(Value::Tensor(
            TensorValue::new_f32_values(shape, values).map_err(EvalError::from)?,
        ));
    }

    // SIMD-over-channel SUM fast path (channel-last f64/f32) — sum/average pooling. The scalar
    // dense_float odometer was ~58-200x JAX. SIMD across C with an f64 accumulator preserving the
    // row-major tap order (bit-identical to the odometer despite float non-associativity, since
    // channels are summed independently; f32 widens→f64→rounds exactly as dense_float does).
    // Matches the rank-2 sum path's `reduce_window_sum_like` contract. Threaded outer loop.
    if no_base_dilation
        && no_window_dilation
        && reduce_window_sum_like(reduce_op)
        && rank >= 2
        && window_dims[rank - 1] == 1
        && strides[rank - 1] == 1
        && pad_lows[rank - 1] == 0
        && out_dims[rank - 1] as usize == tensor.shape.dims[rank - 1] as usize
        && output_dtype == tensor.dtype
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
        let shape = Shape {
            dims: out_dims.clone(),
        };
        if tensor.dtype == fj_core::DType::F64
            && let Some(src) = tensor.elements.as_f64_slice()
        {
            let values = reduce_window_simd_channel_sum_f64(
                src,
                &input_dims,
                &window_dims,
                &strides,
                &pad_lows,
                &out_dims_usize,
            );
            return Ok(Value::Tensor(
                TensorValue::new_f64_values(shape, values).map_err(EvalError::from)?,
            ));
        } else if tensor.dtype == fj_core::DType::F32
            && let Some(src) = tensor.elements.as_f32_slice()
        {
            let values = reduce_window_simd_channel_sum_f32(
                src,
                &input_dims,
                &window_dims,
                &strides,
                &pad_lows,
                &out_dims_usize,
            );
            return Ok(Value::Tensor(
                TensorValue::new_f32_values(shape, values).map_err(EvalError::from)?,
            ));
        }
    }

    // FUSED bf16 channel-last pooling (max/min/sum): widen bf16→f32/f64 INLINE per tap (no f64
    // intermediate), threaded across output blocks. The materialized-widen path below was bounded
    // by the 51MB intermediate; this reads only the half-size bf16 input → ~8x (26ms→~3ms single
    // thread, ~1ms threaded), beating JAX. Bit-identical (half_pool guard).
    if no_base_dilation
        && no_window_dilation
        && output_dtype == fj_core::DType::BF16
        && tensor.dtype == fj_core::DType::BF16
        && (matches!(reduce_op, "max" | "min") || reduce_window_sum_like(reduce_op))
        && rank >= 2
        && window_dims[rank - 1] == 1
        && strides[rank - 1] == 1
        && pad_lows[rank - 1] == 0
        && out_dims[rank - 1] as usize == tensor.shape.dims[rank - 1] as usize
        && let Some(bits) = tensor.elements.as_half_float_slice()
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
        let obits = reduce_window_simd_channel_bf16(
            bits,
            &input_dims,
            &window_dims,
            &strides,
            &pad_lows,
            &out_dims_usize,
            reduce_op,
        );
        let shape = Shape {
            dims: out_dims.clone(),
        };
        return Ok(Value::Tensor(
            TensorValue::new_half_float_values(fj_core::DType::BF16, shape, obits)
                .map_err(EvalError::from)?,
        ));
    }

    // FUSED f16 channel-last pooling (max/min/sum): widen each tap via the branchless
    // f16_widen8_full_f32. MAX/MIN use an f32 running-max scratch (narrow once per output); SUM uses
    // an f64 accumulator in the odometer's tap order (rounds f64→f16 once). Reads the half-size f16
    // input (vs the materialized-f64 odometer below) → bf16-like speedup. Bit-identical (half_pool guard).
    if no_base_dilation
        && no_window_dilation
        && output_dtype == fj_core::DType::F16
        && tensor.dtype == fj_core::DType::F16
        && (matches!(reduce_op, "max" | "min") || reduce_window_sum_like(reduce_op))
        && rank >= 2
        && window_dims[rank - 1] == 1
        && strides[rank - 1] == 1
        && pad_lows[rank - 1] == 0
        && out_dims[rank - 1] as usize == tensor.shape.dims[rank - 1] as usize
        && let Some(bits) = tensor.elements.as_half_float_slice()
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
        let obits = reduce_window_simd_channel_f16(
            bits,
            &input_dims,
            &window_dims,
            &strides,
            &pad_lows,
            &out_dims_usize,
            reduce_op,
        );
        let shape = Shape {
            dims: out_dims.clone(),
        };
        return Ok(Value::Tensor(
            TensorValue::new_half_float_values(fj_core::DType::F16, shape, obits)
                .map_err(EvalError::from)?,
        ));
    }

    // f16 channel-last pooling (max/min/sum) — REUSES the f64 SIMD fns on the widened f64 view
    // (EXACTLY the widen the dense_float odometer applies), then rounds f64→half via
    // reduce_window_literal_from_f64. Bit-identical to the odometer (same widen, same f64 op/order,
    // same round). (bf16 takes the fused path above; this now serves f16.)
    if no_base_dilation
        && no_window_dilation
        && matches!(output_dtype, fj_core::DType::BF16 | fj_core::DType::F16)
        && output_dtype == tensor.dtype
        && (matches!(reduce_op, "max" | "min") || reduce_window_sum_like(reduce_op))
        && rank >= 2
        && window_dims[rank - 1] == 1
        && strides[rank - 1] == 1
        && pad_lows[rank - 1] == 0
        && out_dims[rank - 1] as usize == tensor.shape.dims[rank - 1] as usize
        && let Some(src) = reduce_window_dense_f64_view(tensor)
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
        let values = if matches!(reduce_op, "max" | "min") {
            reduce_window_simd_channel_maxmin_f64(
                src.as_ref(),
                &input_dims,
                &window_dims,
                &strides,
                &pad_lows,
                &out_dims_usize,
                reduce_op == "max",
            )
        } else {
            reduce_window_simd_channel_sum_f64(
                src.as_ref(),
                &input_dims,
                &window_dims,
                &strides,
                &pad_lows,
                &out_dims_usize,
            )
        };
        // BF16 rounds via SIMD (`round_f64_slice_to_bf16`, bit-identical to from_bf16_f64); F16
        // stays scalar. With the SIMD widen above, this removes the round bottleneck for bf16.
        let bits: Vec<u16> = if output_dtype == fj_core::DType::BF16 {
            crate::arithmetic::round_f64_slice_to_bf16(&values)
        } else {
            values
                .iter()
                .map(|&v| match reduce_window_literal_from_f64(output_dtype, v) {
                    fj_core::Literal::BF16Bits(b) | fj_core::Literal::F16Bits(b) => b,
                    _ => 0,
                })
                .collect()
        };
        let shape = Shape {
            dims: out_dims.clone(),
        };
        return Ok(Value::Tensor(
            TensorValue::new_half_float_values(output_dtype, shape, bits)
                .map_err(EvalError::from)?,
        ));
    }

    // Separable i64 max/min: integer sibling of the float deque above. Large-window
    // integer max/min pooling otherwise ran the per-window O(output·∏window) dense
    // path; the monotonic-deque per-axis pass is window-independent and bit-identical
    // (i64 max/min is a unique total order). Same ∏window > 2·∑window gate.
    // I64 AND I32 (JAX's default int) share the i64 backing. max/min select an input
    // value (valid i32, no mod-2^32 wrap), so i32 reuses the deque and just preserves the
    // I32 dtype — a window-INDEPENDENT win over the O(output·∏window) general dense path.
    if no_base_dilation
        && no_window_dilation
        && matches!(tensor.dtype, fj_core::DType::I64 | fj_core::DType::I32)
        && matches!(reduce_op, "max" | "min")
    {
        let win_total: usize = window_dims.iter().product();
        let win_sum: usize = window_dims.iter().sum();
        if win_total > 2 * win_sum
            && let Some(src) = tensor.elements.as_i64_slice()
        {
            let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
            let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
            let values = reduce_window_separable_maxmin_i64(
                src,
                &input_dims,
                &window_dims,
                &strides,
                &pad_lows,
                &out_dims_usize,
                reduce_op == "max",
            );
            let shape = Shape {
                dims: out_dims.clone(),
            };
            let tv = if tensor.dtype == fj_core::DType::I32 {
                TensorValue::new_i32_values(shape, values)
            } else {
                TensorValue::new_i64_values(shape, values)
            };
            return Ok(Value::Tensor(tv.map_err(EvalError::from)?));
        }
    }

    if no_base_dilation
        && no_window_dilation
        && tensor.dtype == fj_core::DType::F64
        && rank == 2
        && matches!(reduce_op, "max" | "min")
        && let Some(src) = reduce_window_f64_values(tensor)
    {
        return eval_reduce_window_rank2_f64_max_min(
            primitive,
            tensor,
            src.as_ref(),
            &window_dims,
            &strides,
            &out_dims,
            &pad_lows,
            total_output,
            reduce_op == "max",
        );
    }

    if no_base_dilation
        && no_window_dilation
        && tensor.dtype == fj_core::DType::F64
        && rank == 2
        && reduce_window_sum_like(reduce_op)
    {
        return eval_reduce_window_rank2_f64_sum(
            primitive,
            tensor,
            &window_dims,
            &strides,
            &out_dims,
            &pad_lows,
            total_output,
        );
    }

    // f32 rank-2 SUM: separable O(input) running-sum fast path for large windows (else falls through to the
    // naive dense path below). f32 is JAX's default dtype.
    if no_base_dilation
        && no_window_dilation
        && tensor.dtype == fj_core::DType::F32
        && rank == 2
        && reduce_window_sum_like(reduce_op)
        && let Some(output) = separable_reduce_window_sum_f32(
            tensor,
            tensor.shape.dims[0] as usize,
            tensor.shape.dims[1] as usize,
            out_dims[0] as usize,
            out_dims[1] as usize,
            window_dims[0],
            window_dims[1],
            strides[0],
            strides[1],
            pad_lows[0],
            pad_lows[1],
        )
    {
        return Ok(Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: out_dims.to_vec(),
                },
                output,
            )
            .map_err(EvalError::InvalidTensor)?,
        ));
    }

    // Separable monotonic-deque MAX/MIN fast path: a box max/min is the composition
    // of per-axis 1-D sliding extrema (max/min are separable + order-independent),
    // each computable WINDOW-INDEPENDENTLY in O(axis_len) by a monotonic deque —
    // O(total) vs the naive O(out·∏window) tap loop below. Gated to OVERLAPPING
    // windows (some windowed axis has stride < window); non-overlapping pooling
    // (stride == window) is already O(input) on the dense_float path, where the
    // deque's bookkeeping would only add overhead. No (base/window) dilation.
    // Bit-identical to eval_reduce_window_dense_float for max/min (finite → unique
    // extremum; NaN → canonical f64::NAN, since jax_max/min_f64 canonicalize too).
    if no_base_dilation
        && no_window_dilation
        && matches!(reduce_op, "max" | "min")
        && window_dims
            .iter()
            .zip(strides.iter())
            .any(|(&w, &s)| w > 1 && s < w)
        && let Some(src) = reduce_window_dense_f64_view(tensor)
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        return eval_reduce_window_separable_extremum(
            output_dtype,
            reduce_op == "max",
            src.as_ref(),
            &window_dims,
            &strides,
            &out_dims,
            &pad_lows,
            &input_dims,
        );
    }

    // Rank-3 VALID f64 SUM: SIMD across adjacent output-x cells. Each lane is an
    // independent output cell and receives taps in the exact row-major order used
    // by eval_reduce_window_dense_float, preserving floating-point bit identity.
    if no_base_dilation
        && no_window_dilation
        && tensor.dtype == fj_core::DType::F64
        && output_dtype == fj_core::DType::F64
        && rank == 3
        && reduce_window_sum_like(reduce_op)
        && strides.iter().all(|&s| s == 1)
        && pad_lows.iter().all(|&p| p == 0)
        && window_dims.iter().all(|&w| w > 0)
        && window_dims.iter().product::<usize>() >= 64
        && let Some(src) = tensor.elements.as_f64_slice()
    {
        let input_dims = [
            tensor.shape.dims[0] as usize,
            tensor.shape.dims[1] as usize,
            tensor.shape.dims[2] as usize,
        ];
        let window_dims_rank3 = [window_dims[0], window_dims[1], window_dims[2]];
        let out_dims_rank3 = [
            out_dims[0] as usize,
            out_dims[1] as usize,
            out_dims[2] as usize,
        ];
        if (0..3).all(|d| out_dims_rank3[d] + window_dims_rank3[d] - 1 <= input_dims[d]) {
            // Large windows: the separable 3-pass running-sum is window-INDEPENDENT, so it
            // beats the per-tap xlane path once ∏window crosses the measured break-even
            // (~win6³=216 on this host: xlane scales with ∏window, separable stays ~flat —
            // win9³ is 3.48x faster). Tolerance-equal (running-sum reassoc); falls back to
            // the bit-identical xlane path on non-finite input or below the gate.
            const SEPARABLE_3D_SUM_MIN_TAPS: usize = 216;
            if window_dims_rank3.iter().product::<usize>() >= SEPARABLE_3D_SUM_MIN_TAPS
                && let Some(values) = separable_reduce_window_sum_3d_f64(
                    src,
                    input_dims,
                    window_dims_rank3,
                    out_dims_rank3,
                )
            {
                return Ok(Value::Tensor(
                    TensorValue::new_f64_values(
                        Shape {
                            dims: out_dims.clone(),
                        },
                        values,
                    )
                    .map_err(EvalError::from)?,
                ));
            }
            let values = reduce_window_rank3_f64_sum_xlane(
                src,
                input_dims,
                window_dims_rank3,
                out_dims_rank3,
            );
            return Ok(Value::Tensor(
                TensorValue::new_f64_values(
                    Shape {
                        dims: out_dims.clone(),
                    },
                    values,
                )
                .map_err(EvalError::from)?,
            ));
        }
    }

    // Rank-3 VALID f32 SUM: large-window separable running-sum. f32 volumetric sum has no
    // dedicated kernel (it falls to the generic dense-float per-tap loop below), so the
    // window-independent 3-pass separable wins above the same ∏window break-even as f64.
    // Tolerance-equal (running-sum reassoc, f64 accumulate, narrow output); falls back to the
    // generic loop on non-finite input or below the gate.
    if no_base_dilation
        && no_window_dilation
        && tensor.dtype == fj_core::DType::F32
        && output_dtype == fj_core::DType::F32
        && rank == 3
        && reduce_window_sum_like(reduce_op)
        && strides.iter().all(|&s| s == 1)
        && pad_lows.iter().all(|&p| p == 0)
        && window_dims.iter().all(|&w| w > 0)
        && window_dims.iter().product::<usize>() >= 216
        && let Some(src) = tensor.elements.as_f32_slice()
    {
        let input_dims = [
            tensor.shape.dims[0] as usize,
            tensor.shape.dims[1] as usize,
            tensor.shape.dims[2] as usize,
        ];
        let window_dims_rank3 = [window_dims[0], window_dims[1], window_dims[2]];
        let out_dims_rank3 = [
            out_dims[0] as usize,
            out_dims[1] as usize,
            out_dims[2] as usize,
        ];
        if (0..3).all(|d| out_dims_rank3[d] + window_dims_rank3[d] - 1 <= input_dims[d])
            && let Some(values) = separable_reduce_window_sum_3d_f32(
                src,
                input_dims,
                window_dims_rank3,
                out_dims_rank3,
            )
        {
            return Ok(Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: out_dims.clone(),
                    },
                    values,
                )
                .map_err(EvalError::from)?,
            ));
        }
    }

    // Rank>=4 float SUM with large windows: the N-D separable prefix-sum (per-axis 1-D pass)
    // beats the per-tap dense loop once ∏window outgrows the per-axis overhead — the realistic
    // 3-D-CNN avg-pool is rank-5 [N,D,H,W,C] (windows on D/H/W). Same `∏window > 2·∑window`
    // break-even the separable max/min path uses; rank<=3 keeps its dedicated paths above.
    // Tolerance-equal (prefix-sum reassoc, f64 accumulate, narrow to output dtype).
    if no_base_dilation
        && no_window_dilation
        && rank >= 4
        && reduce_window_sum_like(reduce_op)
        && matches!(
            output_dtype,
            fj_core::DType::F64 | fj_core::DType::F32 | fj_core::DType::BF16 | fj_core::DType::F16
        )
        && window_dims.iter().product::<usize>() > 2 * window_dims.iter().sum::<usize>()
        && let Some(src) = reduce_window_dense_f64_view(tensor)
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
        let values = reduce_window_separable_sum_nd_f64(
            src.as_ref(),
            &input_dims,
            &window_dims,
            &strides,
            &pad_lows,
            &out_dims_usize,
        );
        let shape = Shape {
            dims: out_dims.clone(),
        };
        return match output_dtype {
            fj_core::DType::F32 => Ok(Value::Tensor(
                TensorValue::new_f32_values(shape, values.iter().map(|&v| v as f32).collect())
                    .map_err(EvalError::from)?,
            )),
            fj_core::DType::BF16 | fj_core::DType::F16 => {
                let bits: Vec<u16> = values
                    .iter()
                    .map(|&v| match reduce_window_literal_from_f64(output_dtype, v) {
                        fj_core::Literal::BF16Bits(b) | fj_core::Literal::F16Bits(b) => b,
                        _ => 0,
                    })
                    .collect();
                Ok(Value::Tensor(
                    TensorValue::new_half_float_values(output_dtype, shape, bits)
                        .map_err(EvalError::from)?,
                ))
            }
            _ => Ok(Value::Tensor(
                TensorValue::new_f64_values(shape, values).map_err(EvalError::from)?,
            )),
        };
    }

    // Dense float fast path (ANY rank, sum/max/min): the F64-accumulator float case
    // for F64 or dense-F32 input. Replaces the generic per-`Literal` gather +
    // string-dispatched accumulate with a dense `f64` read and a hoisted op, plus an
    // interior tap-offset stencil. Covers real 2D pooling (rank-4 NHWC) and ALL f32
    // pooling — neither of which the rank-2-only F64 paths above ever reach. Falls
    // through (returns None view) for boxed/non-float input. Bit-identical to the
    // generic loop below (see eval_reduce_window_dense_float).
    if no_base_dilation
        && matches!(
            output_dtype,
            fj_core::DType::F64 | fj_core::DType::F32 | fj_core::DType::BF16 | fj_core::DType::F16
        )
        && (matches!(reduce_op, "max" | "min") || reduce_window_sum_like(reduce_op))
        && let Some(src) = reduce_window_dense_f64_view(tensor)
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let mut input_strides = vec![1usize; rank];
        let mut stride_mult = 1usize;
        for d in (0..rank).rev() {
            input_strides[d] = stride_mult;
            stride_mult = stride_mult.checked_mul(input_dims[d]).ok_or_else(|| {
                reduce_window_unsupported(primitive, "reduce_window stride multiplier overflow")
            })?;
        }
        return eval_reduce_window_dense_float(
            output_dtype,
            reduce_op,
            src.as_ref(),
            &window_dims,
            &strides,
            &window_dilation,
            &out_dims,
            &pad_lows,
            &input_dims,
            &input_strides,
            total_output,
        );
    }

    // Summed-area-table (integral image) i64 SUM fast path for rank-2 pooling.
    // O(input + output) and WINDOW-INDEPENDENT vs the per-window O(output·∏window)
    // tap sum — a big win for large windows. Integer add is associative+commutative
    // so the 4-corner rectangle sum is bit-identical to the per-window wrapping
    // tap sum (see eval_reduce_window_rank2_i64_sum_sat). Gated to windows large
    // enough (∏window ≥ 16) that the integral-image build pays; small 2×2/3×3
    // windows stay on the per-window dense path below.
    if no_base_dilation
        && no_window_dilation
        && rank == 2
        && matches!(tensor.dtype, fj_core::DType::I64 | fj_core::DType::I32)
        && reduce_window_sum_like(reduce_op)
        && window_dims.iter().product::<usize>() >= 16
        && let Some(src) = tensor.elements.as_i64_slice()
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let result = eval_reduce_window_rank2_i64_sum_sat(
            src,
            &window_dims,
            &strides,
            &out_dims,
            &pad_lows,
            &input_dims,
            total_output,
        )?;
        return reduce_window_sat_retag_i32(result, tensor.dtype);
    }

    // General-rank summed-area-table i64 SUM fast path (rank 1 and 3..=6; rank 2
    // is handled by the specialized path above). Same window-independent
    // O(input + output) integral-image + 2^rank-corner inclusion-exclusion,
    // bit-identical to the per-window wrapping tap sum. Gated ∏window ≥ 16 and
    // rank ≤ 6 (bounds the corner loop); rank 1 = sliding-window integer sum,
    // rank 3/4 = NHWC integer pooling.
    if no_base_dilation
        && no_window_dilation
        && (rank == 1 || (3..=6).contains(&rank))
        && matches!(tensor.dtype, fj_core::DType::I64 | fj_core::DType::I32)
        && reduce_window_sum_like(reduce_op)
        && window_dims.iter().product::<usize>() >= 16
        && let Some(src) = tensor.elements.as_i64_slice()
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let result = eval_reduce_window_in_sum_sat(
            src,
            &window_dims,
            &strides,
            &out_dims,
            &pad_lows,
            &input_dims,
            total_output,
        )?;
        return reduce_window_sat_retag_i32(result, tensor.dtype);
    }

    // Dense i64 fast path (ANY rank, sum/max/min): integer pooling/windowed
    // reductions ran the generic per-`Literal` gather + string-dispatched
    // accumulate — the slowest path. I64 tensors densify, so read the i64 buffer
    // directly and run the hoisted op over the same tap-offset stencil the float
    // path uses. Bit-identical to the generic I64 accumulator (see
    // eval_reduce_window_dense_i64). I32 stays on the generic path (i32 is stored
    // boxed → no i64 slice view) and is narrowed mod-2^32 by the eval_primitive
    // chokepoint regardless.
    // I64 AND I32: both share the dense i64 backing (`as_i64_slice`). I32 is JAX's
    // default integer dtype (so integer pooling is common) but was excluded here and
    // fell to the generic per-`Literal` loop; route it through the same stencil and let
    // the eval_primitive chokepoint wrap the I32 output mod 2^32.
    // Separable monotonic-deque MAX/MIN for dense i64/i32 OVERLAPPING windows: the
    // integer sibling of the float deque path above. Same O(input) window-independent
    // win, simpler (integers have no NaN). Gated to overlapping (stride < window) +
    // no dilation; non-overlapping pooling is already O(input) on the dense_i64 path.
    if no_base_dilation
        && no_window_dilation
        && matches!(tensor.dtype, fj_core::DType::I64 | fj_core::DType::I32)
        && matches!(reduce_op, "max" | "min")
        && window_dims
            .iter()
            .zip(strides.iter())
            .any(|(&w, &s)| w > 1 && s < w)
        && let Some(src) = tensor.elements.as_i64_slice()
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        return eval_reduce_window_separable_extremum_i64(
            tensor.dtype,
            reduce_op == "max",
            src,
            &window_dims,
            &strides,
            &out_dims,
            &pad_lows,
            &input_dims,
        );
    }

    // Rank-3 VALID I64 SUM: large-window separable running-sum. Integer add is a ring, so this is
    // BIT-IDENTICAL to the naive `wrapping_add` fold (no SAT for rank>2; the per-tap loop is
    // O(out·∏window)). Same ∏window>=216 break-even as the float separable.
    if no_base_dilation
        && no_window_dilation
        && tensor.dtype == fj_core::DType::I64
        && output_dtype == fj_core::DType::I64
        && rank == 3
        && reduce_window_sum_like(reduce_op)
        && strides.iter().all(|&s| s == 1)
        && pad_lows.iter().all(|&p| p == 0)
        && window_dims.iter().all(|&w| w > 0)
        && window_dims.iter().product::<usize>() >= 216
        && let Some(src) = tensor.elements.as_i64_slice()
    {
        let input_dims = [
            tensor.shape.dims[0] as usize,
            tensor.shape.dims[1] as usize,
            tensor.shape.dims[2] as usize,
        ];
        let window_dims_rank3 = [window_dims[0], window_dims[1], window_dims[2]];
        let out_dims_rank3 = [
            out_dims[0] as usize,
            out_dims[1] as usize,
            out_dims[2] as usize,
        ];
        if (0..3).all(|d| out_dims_rank3[d] + window_dims_rank3[d] - 1 <= input_dims[d])
            && let Some(values) = separable_reduce_window_sum_3d_i64(
                src,
                input_dims,
                window_dims_rank3,
                out_dims_rank3,
            )
        {
            return Ok(Value::Tensor(
                TensorValue::new_i64_values(
                    Shape {
                        dims: out_dims.clone(),
                    },
                    values,
                )
                .map_err(EvalError::from)?,
            ));
        }
    }

    // Rank>=4 I64 SUM: N-D separable block-prefix (the integer analogue of the rank>=4 float
    // path). Bit-identical (integer ring); covers integer volumetric pooling at any stride/pad.
    if no_base_dilation
        && no_window_dilation
        && tensor.dtype == fj_core::DType::I64
        && output_dtype == fj_core::DType::I64
        && rank >= 4
        && reduce_window_sum_like(reduce_op)
        && window_dims.iter().product::<usize>() > 2 * window_dims.iter().sum::<usize>()
        && let Some(src) = tensor.elements.as_i64_slice()
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let out_dims_usize: Vec<usize> = out_dims.iter().map(|&d| d as usize).collect();
        let values = reduce_window_separable_sum_nd_i64(
            src,
            &input_dims,
            &window_dims,
            &strides,
            &pad_lows,
            &out_dims_usize,
        );
        return Ok(Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: out_dims.clone(),
                },
                values,
            )
            .map_err(EvalError::from)?,
        ));
    }

    if no_base_dilation
        && matches!(tensor.dtype, fj_core::DType::I64 | fj_core::DType::I32)
        && (matches!(reduce_op, "max" | "min") || reduce_window_sum_like(reduce_op))
        && let Some(src) = tensor.elements.as_i64_slice()
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let mut input_strides = vec![1usize; rank];
        let mut stride_mult = 1usize;
        for d in (0..rank).rev() {
            input_strides[d] = stride_mult;
            stride_mult = stride_mult.checked_mul(input_dims[d]).ok_or_else(|| {
                reduce_window_unsupported(primitive, "reduce_window stride multiplier overflow")
            })?;
        }
        return eval_reduce_window_dense_i64(
            output_dtype,
            reduce_op,
            src,
            &window_dims,
            &strides,
            &window_dilation,
            &out_dims,
            &pad_lows,
            &input_dims,
            &input_strides,
            total_output,
        );
    }

    // Dense bool fast path (ANY rank, logical and/or pooling): boolean windowed
    // reductions (binary morphology dilation/erosion, validity-mask pooling) ran
    // the generic per-`Literal` gather. Dense Bool storage exposes a `bool` slice,
    // so read it directly and run the hoisted op. Bit-identical to the generic
    // Bool accumulator (see eval_reduce_window_dense_bool). Bit-packed BoolWords
    // storage returns no slice view → stays on the generic path.
    if no_base_dilation
        && tensor.dtype == fj_core::DType::Bool
        && matches!(reduce_op, "max" | "min" | "sum")
        && let Some(src) = tensor.elements.as_bool_slice()
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let mut input_strides = vec![1usize; rank];
        let mut stride_mult = 1usize;
        for d in (0..rank).rev() {
            input_strides[d] = stride_mult;
            stride_mult = stride_mult.checked_mul(input_dims[d]).ok_or_else(|| {
                reduce_window_unsupported(primitive, "reduce_window stride multiplier overflow")
            })?;
        }
        return eval_reduce_window_dense_bool(
            reduce_op,
            src,
            &window_dims,
            &strides,
            &window_dilation,
            &out_dims,
            &pad_lows,
            &input_dims,
            &input_strides,
            total_output,
        );
    }

    // Dense complex fast path (ANY rank, sum/max/min): complex windowed reductions
    // (FFT-domain pooling/integration) ran the generic per-`Literal` gather. Dense
    // Complex storage exposes a `(re, im)` slice; read it directly and run the
    // hoisted op. Bit-identical to the generic Complex accumulator (same window
    // order for the component sums, same lexicographic complex_ge for max/min,
    // same ∓∞/(0,0) init — see eval_reduce_window_dense_complex).
    if no_base_dilation
        && matches!(
            tensor.dtype,
            fj_core::DType::Complex64 | fj_core::DType::Complex128
        )
        && matches!(reduce_op, "max" | "min" | "sum")
        && let Some(src) = tensor.elements.as_complex_slice()
    {
        let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
        let mut input_strides = vec![1usize; rank];
        let mut stride_mult = 1usize;
        for d in (0..rank).rev() {
            input_strides[d] = stride_mult;
            stride_mult = stride_mult.checked_mul(input_dims[d]).ok_or_else(|| {
                reduce_window_unsupported(primitive, "reduce_window stride multiplier overflow")
            })?;
        }
        return eval_reduce_window_dense_complex(
            tensor.dtype,
            reduce_op,
            src,
            &window_dims,
            &strides,
            &window_dilation,
            &out_dims,
            &pad_lows,
            &input_dims,
            &input_strides,
            total_output,
        );
    }

    // For 1D case: straightforward sliding window
    // For N-D: use multi-dimensional indexing
    let input_dims: Vec<usize> = tensor.shape.dims.iter().map(|d| *d as usize).collect();
    let mut input_strides = vec![1usize; rank];
    let mut stride_mult = 1usize;
    for d in (0..rank).rev() {
        input_strides[d] = stride_mult;
        stride_mult =
            stride_mult
                .checked_mul(input_dims[d])
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: "reduce_window stride multiplier overflow".to_owned(),
                })?;
    }

    let mut output_elements = Vec::with_capacity(total_output);

    // Out-of-bounds (padding) taps contribute the reduction init value: XLA pads
    // reduce_window with `init` and reduces the FULL window. For sum that is 0
    // (so a -0.0 partial sum lands +0.0, as XLA does); for max/min the init is
    // ∓∞, so combining it is a no-op — making this correct for every op.
    let pad_literal = reduce_window_accumulator_literal(
        primitive,
        output_dtype,
        reduce_window_initial_accumulator(output_dtype, reduce_op),
    )?;

    // Iterate over all output positions using multi-dimensional index
    let out_dims_usize: Vec<usize> = out_dims.iter().map(|d| *d as usize).collect();
    let mut out_idx = vec![0usize; rank];
    let win_total: usize = window_dims.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "reduce_window window element count overflow".to_owned(),
        })
    })?;
    let mut win_idx = vec![0usize; rank];

    for _ in 0..total_output {
        // For this output position, compute the window
        let mut accum = reduce_window_initial_accumulator(output_dtype, reduce_op);

        // Iterate over all positions within the window
        win_idx.fill(0);

        for _ in 0..win_total {
            // Compute input index for this window position
            let mut in_bounds = true;
            let mut flat_input_idx = 0usize;

            for d in (0..rank).rev() {
                // Dilated window tap: position = out*stride + win_idx*window_dilation.
                let padded_pos = win_idx[d]
                    .checked_mul(window_dilation[d])
                    .and_then(|tap| out_idx[d].checked_mul(strides[d]).map(|base| base + tap))
                    .ok_or_else(|| EvalError::Unsupported {
                        primitive,
                        detail: "reduce_window window index overflow".to_owned(),
                    })?;
                if padded_pos < pad_lows[d] {
                    in_bounds = false;
                    break;
                }
                // `input_pos` is a coordinate in the base-DILATED grid (size
                // `dilated_input_dims[d] = (n-1)*db+1`).
                let input_pos = padded_pos - pad_lows[d];
                if input_pos >= dilated_input_dims[d] {
                    in_bounds = false;
                    break;
                }
                // Base-dilation hole: a position not on a `base_dilation` multiple has no
                // real operand element, so it contributes the reduction init value — handled
                // identically to an out-of-bounds/padding tap (`pad_literal`). For `db == 1`
                // this never triggers and `real_pos == input_pos` (bit-identical).
                if !input_pos.is_multiple_of(base_dilation[d]) {
                    in_bounds = false;
                    break;
                }
                let real_pos = input_pos / base_dilation[d];
                let flat_increment = real_pos.checked_mul(input_strides[d]).ok_or_else(|| {
                    EvalError::Unsupported {
                        primitive,
                        detail: "reduce_window flat index overflow".to_owned(),
                    }
                })?;
                flat_input_idx = flat_input_idx.checked_add(flat_increment).ok_or_else(|| {
                    EvalError::Unsupported {
                        primitive,
                        detail: "reduce_window flat index overflow".to_owned(),
                    }
                })?;
            }

            let tap = if in_bounds {
                tensor.elements[flat_input_idx]
            } else {
                pad_literal
            };
            reduce_window_accumulate_literal(primitive, reduce_op, &mut accum, tap)?;

            // Increment window index
            let mut carry = true;
            for d in (0..rank).rev() {
                if carry {
                    win_idx[d] += 1;
                    if win_idx[d] >= window_dims[d] {
                        win_idx[d] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }

        output_elements.push(reduce_window_accumulator_literal(
            primitive,
            output_dtype,
            accum,
        )?);

        // Increment output index
        let mut carry = true;
        for d in (0..rank).rev() {
            if carry {
                out_idx[d] += 1;
                if out_idx[d] >= out_dims_usize[d] {
                    out_idx[d] = 0;
                } else {
                    carry = false;
                }
            }
        }
    }

    Ok(Value::Tensor(
        TensorValue::new(output_dtype, Shape { dims: out_dims }, output_elements)
            .map_err(EvalError::InvalidTensor)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        EvalError, eval_primitive, reduce_window_accumulate_literal,
        reduce_window_accumulator_literal, reduce_window_initial_accumulator,
        reduce_window_same_geometry,
    };
    use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value, ValueError};
    use std::collections::BTreeMap;

    // The THREADED generic dense-float reduce_window (eval_reduce_window_dense_float, threaded over
    // independent output cells) must be bit-identical to a naive single-thread per-cell window sum.
    // Sized to cross the threading threshold (total_output·win_total ≥ 1<<18).
    #[test]
    fn reduce_window_dense_float_threaded_matches_naive() {
        let (rows, cols, win) = (256usize, 4096usize, 9usize);
        let src: Vec<f64> = (0..rows * cols)
            .map(|i| ((i % 4099) as f64) * 0.017 - 30.0)
            .collect();
        let out_cols = cols - win + 1;
        let got = super::eval_reduce_window_dense_float(
            fj_core::DType::F64,
            "sum",
            &src,
            &[1, win],
            &[1, 1],
            &[1, 1],
            &[rows as u32, out_cols as u32],
            &[0, 0],
            &[rows, cols],
            &[cols, 1],
            rows * out_cols,
        )
        .unwrap();
        let gv = match &got {
            Value::Tensor(t) => t.elements.as_f64_slice().unwrap().to_vec(),
            _ => panic!(),
        };
        let mut want = vec![0.0f64; rows * out_cols];
        for r in 0..rows {
            for oc in 0..out_cols {
                let mut acc = 0.0;
                for k in 0..win {
                    acc += src[r * cols + oc + k];
                }
                want[r * out_cols + oc] = acc;
            }
        }
        assert_eq!(gv.len(), want.len());
        for i in 0..want.len() {
            assert_eq!(
                gv[i].to_bits(),
                want[i].to_bits(),
                "dense_float threaded sum mismatch i={i}"
            );
        }
    }

    // A/B: naive single-thread per-cell sum vs the THREADED eval_reduce_window_dense_float.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_dense_float_threaded_ab() {
        use std::time::Instant;
        let (rows, cols, win) = (512usize, 4096usize, 17usize);
        let src: Vec<f64> = (0..rows * cols)
            .map(|i| ((i % 4099) as f64) * 0.017 - 30.0)
            .collect();
        let out_cols = cols - win + 1;
        let threaded = || {
            let r = super::eval_reduce_window_dense_float(
                fj_core::DType::F64,
                "sum",
                &src,
                &[1, win],
                &[1, 1],
                &[1, 1],
                &[rows as u32, out_cols as u32],
                &[0, 0],
                &[rows, cols],
                &[cols, 1],
                rows * out_cols,
            )
            .unwrap();
            std::hint::black_box(&r);
        };
        let naive = || {
            let mut out = vec![0.0f64; rows * out_cols];
            for r in 0..rows {
                for oc in 0..out_cols {
                    let mut acc = 0.0;
                    for k in 0..win {
                        acc += src[r * cols + oc + k];
                    }
                    out[r * out_cols + oc] = acc;
                }
            }
            std::hint::black_box(&out);
        };
        let bench = |f: &dyn Fn()| {
            f();
            let mut b = f64::MAX;
            for _ in 0..10 {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b * 1e3
        };
        let n = bench(&naive);
        let t = bench(&threaded);
        println!(
            "AB reduce_window dense_float sum [512,4096] w=17: naive-1thread={n:.4}ms threaded={t:.4}ms ({:.2}x)",
            n / t
        );
    }

    // The THREADED separable monotonic-deque 1-D sliding max/min (reduce_window_extremum_1d_axis_f64,
    // threaded over the independent outer blocks) must be bit-identical to a naive O(out·window) tap
    // reference. Sized to cross the threading threshold (outer=128 ≥ 2, work ≥ 1<<18).
    #[test]
    fn reduce_window_deque_threaded_matches_naive() {
        let (outer, axis_len) = (128usize, 2048usize); // [outer, axis_len], window on the last axis
        let src: Vec<f64> = (0..outer * axis_len)
            .map(|i| {
                let v = ((i % 4099) as f64) * 0.031 - 60.0;
                if i % 911 == 0 { f64::NAN } else { v }
            })
            .collect();
        for is_max in [true, false] {
            for (window, stride, pad_low) in [(17usize, 1usize, 0usize), (33, 3, 5), (64, 7, 0)] {
                let out_ax = {
                    let padded = axis_len + pad_low; // pad_high handled by min(axis_len) in the deque
                    (padded.saturating_sub(window)) / stride + 1
                };
                let got = super::reduce_window_extremum_1d_axis_f64(
                    &src,
                    &[outer, axis_len],
                    super::ReduceWindowExtremumAxis {
                        axis: 1,
                        window,
                        stride,
                        pad_low,
                        out_ax,
                        is_max,
                    },
                );
                // Naive reference: per line, per output, scan the window (OOB padded taps skipped;
                // NaN in window → canonical NaN), bit-for-bit.
                let mut want = vec![0.0f64; outer * out_ax];
                for o in 0..outer {
                    for oa in 0..out_ax {
                        let start = oa * stride;
                        let lo = start.saturating_sub(pad_low);
                        let hi = (start + window).saturating_sub(pad_low).min(axis_len);
                        let mut acc = if is_max {
                            f64::NEG_INFINITY
                        } else {
                            f64::INFINITY
                        };
                        let mut any_nan = false;
                        for k in lo..hi {
                            let v = src[o * axis_len + k];
                            if v.is_nan() {
                                any_nan = true;
                            } else if is_max {
                                acc = acc.max(v);
                            } else {
                                acc = acc.min(v);
                            }
                        }
                        want[o * out_ax + oa] = if any_nan { f64::NAN } else { acc };
                    }
                }
                assert_eq!(got.len(), want.len());
                for i in 0..want.len() {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "deque vs naive mismatch is_max={is_max} w={window} s={stride} p={pad_low} i={i}"
                    );
                }
            }
        }
    }

    // A/B: threaded deque (large, crosses threshold) vs the naive O(out·window) odometer it replaces.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_deque_threaded_ab() {
        use std::time::Instant;
        let (outer, axis_len, window, stride) = (256usize, 4096usize, 65usize, 1usize);
        let src: Vec<f64> = (0..outer * axis_len)
            .map(|i| ((i % 4099) as f64) * 0.031 - 60.0)
            .collect();
        let out_ax = (axis_len - window) / stride + 1;
        let deque = || {
            let r = super::reduce_window_extremum_1d_axis_f64(
                &src,
                &[outer, axis_len],
                super::ReduceWindowExtremumAxis {
                    axis: 1,
                    window,
                    stride,
                    pad_low: 0,
                    out_ax,
                    is_max: true,
                },
            );
            std::hint::black_box(&r);
        };
        // SERIAL monotonic deque (single-thread) — the pre-threading baseline; isolates the threading win.
        let serial_deque = || {
            let mut out = vec![0.0f64; outer * out_ax];
            let mut dq: std::collections::VecDeque<(f64, usize)> =
                std::collections::VecDeque::new();
            for o in 0..outer {
                dq.clear();
                let mut added = 0usize;
                let mut left = 0usize;
                for oa in 0..out_ax {
                    let start = oa * stride;
                    let hi = (start + window).min(axis_len);
                    while added < hi {
                        let v = src[o * axis_len + added];
                        while let Some(&(bv, _)) = dq.back() {
                            if bv <= v {
                                dq.pop_back();
                            } else {
                                break;
                            }
                        }
                        dq.push_back((v, added));
                        added += 1;
                    }
                    while left < start {
                        if dq.front().map(|&(_, i)| i) == Some(left) {
                            dq.pop_front();
                        }
                        left += 1;
                    }
                    out[o * out_ax + oa] = dq.front().map(|&(v, _)| v).unwrap_or(f64::NEG_INFINITY);
                }
            }
            std::hint::black_box(&out);
        };
        let bench = |f: &dyn Fn()| {
            f();
            let mut b = f64::MAX;
            for _ in 0..10 {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            b * 1e3
        };
        let s = bench(&serial_deque);
        let d = bench(&deque);
        println!(
            "AB reduce_window max [256,4096] w=65 s=1: serial-deque={s:.4}ms threaded-deque={d:.4}ms ({:.2}x)",
            s / d
        );
    }

    // SIMD-over-channel maxpool/minpool bit-identity guard: the production
    // reduce_window_simd_channel_maxmin_f64 (f64x8 across channels, NaN-propagating) must match a
    // naive scalar channel-last reduction bit-for-bit across max/min, NaN, signed-zero, VALID and
    // padded windows, rank-3 and rank-4. (eval_primitive now ROUTES through the SIMD fn for this
    // shape class, so this pins it against an independent scalar reference.)
    #[test]
    fn maxpool_simd_channel_bit_identical() {
        fn naive(
            src: &[f64],
            dims: &[usize],
            win: &[usize],
            strides: &[usize],
            pad: &[usize],
            out_dims: &[usize],
            is_max: bool,
        ) -> Vec<f64> {
            let rank = dims.len();
            let pooled = rank - 1;
            let c = dims[pooled];
            let init = if is_max {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            let mut in_strides = vec![1usize; rank];
            for d in (0..rank - 1).rev() {
                in_strides[d] = in_strides[d + 1] * dims[d + 1];
            }
            let out_total: usize = out_dims.iter().product();
            let mut out = vec![init; out_total];
            let out_spatial: usize = out_dims[..pooled].iter().product();
            let win_total: usize = win[..pooled].iter().product();
            let mut oc = vec![0usize; pooled];
            for _ in 0..out_spatial {
                let mut ob = 0usize;
                for d in 0..pooled {
                    ob = ob * out_dims[d] + oc[d];
                }
                ob *= c;
                let mut tap = vec![0usize; pooled];
                for _ in 0..win_total {
                    let mut ib = 0usize;
                    let mut oob = false;
                    for d in 0..pooled {
                        let p = oc[d] * strides[d] + tap[d];
                        if p < pad[d] {
                            oob = true;
                            break;
                        }
                        let ic = p - pad[d];
                        if ic >= dims[d] {
                            oob = true;
                            break;
                        }
                        ib += ic * in_strides[d];
                    }
                    if !oob {
                        for ci in 0..c {
                            let a = out[ob + ci];
                            let t = src[ib + ci];
                            out[ob + ci] = if a.is_nan() || t.is_nan() {
                                f64::NAN
                            } else if is_max {
                                a.max(t)
                            } else {
                                a.min(t)
                            };
                        }
                    }
                    for d in (0..pooled).rev() {
                        tap[d] += 1;
                        if tap[d] < win[d] {
                            break;
                        }
                        tap[d] = 0;
                    }
                }
                for d in (0..pooled).rev() {
                    oc[d] += 1;
                    if oc[d] < out_dims[d] {
                        break;
                    }
                    oc[d] = 0;
                }
            }
            out
        }
        let out_dim = |i: usize, w: usize, s: usize, p: usize| (i + 2 * p - w) / s + 1;
        // (dims, window, strides, pad_low) for the channel-last shape under test.
        type Cfg = (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>);
        let configs: &[Cfg] = &[
            (
                vec![2, 16, 16, 12],
                vec![1, 3, 3, 1],
                vec![1, 2, 2, 1],
                vec![0, 0, 0, 0],
            ),
            (
                vec![2, 16, 16, 12],
                vec![1, 3, 3, 1],
                vec![1, 1, 1, 1],
                vec![0, 1, 1, 0],
            ),
            (
                vec![3, 17, 13, 8],
                vec![1, 2, 2, 1],
                vec![1, 2, 2, 1],
                vec![0, 0, 0, 0],
            ),
            (vec![4, 32, 9], vec![1, 3, 1], vec![1, 2, 1], vec![0, 0, 0]),
            // Large enough (out_total > 2^17) to exercise the THREADED partition vs the scalar naive.
            (
                vec![8, 56, 56, 64],
                vec![1, 3, 3, 1],
                vec![1, 2, 2, 1],
                vec![0, 0, 0, 0],
            ),
        ];
        for (dims, win, strides, pad) in configs {
            let rank = dims.len();
            let out_dims: Vec<usize> = (0..rank)
                .map(|d| out_dim(dims[d], win[d], strides[d], pad[d]))
                .collect();
            let total: usize = dims.iter().product();
            for is_max in [true, false] {
                for variant in 0..3 {
                    let mut src: Vec<f64> = (0..total)
                        .map(|i| ((i % 101) as f64) * 0.37 - 18.5)
                        .collect();
                    if variant == 1 {
                        src[total / 3] = f64::NAN;
                        src[total / 2] = f64::NAN;
                    } else if variant == 2 {
                        for (i, v) in src.iter_mut().enumerate() {
                            *v = if i % 2 == 0 { 0.0 } else { -0.0 };
                        }
                    }
                    let got = super::reduce_window_simd_channel_maxmin_f64(
                        src.as_slice(),
                        dims,
                        win,
                        strides,
                        pad,
                        &out_dims,
                        is_max,
                    );
                    let want = naive(src.as_slice(), dims, win, strides, pad, &out_dims, is_max);
                    assert_eq!(
                        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "SIMD-channel maxpool mismatch (dims={dims:?} is_max={is_max} variant={variant})"
                    );
                }
            }
        }
    }

    // f32 sibling guard: the f32x16 SIMD-channel maxpool must match a naive f32 scalar reference
    // bit-for-bit (max/min × finite/NaN/signed-zero), pinning the f32 simd_max/simd_ne/select.
    #[test]
    fn maxpool_simd_channel_f32_bit_identical() {
        fn naive_f32(
            src: &[f32],
            dims: &[usize],
            win: &[usize],
            strides: &[usize],
            pad: &[usize],
            out_dims: &[usize],
            is_max: bool,
        ) -> Vec<f32> {
            let rank = dims.len();
            let pooled = rank - 1;
            let c = dims[pooled];
            let init = if is_max {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
            let mut in_strides = vec![1usize; rank];
            for d in (0..rank - 1).rev() {
                in_strides[d] = in_strides[d + 1] * dims[d + 1];
            }
            let mut out = vec![init; out_dims.iter().product()];
            let out_spatial: usize = out_dims[..pooled].iter().product();
            let win_total: usize = win[..pooled].iter().product();
            let mut oc = vec![0usize; pooled];
            for _ in 0..out_spatial {
                let mut ob = 0usize;
                for d in 0..pooled {
                    ob = ob * out_dims[d] + oc[d];
                }
                ob *= c;
                let mut tap = vec![0usize; pooled];
                for _ in 0..win_total {
                    let mut ib = 0usize;
                    let mut oob = false;
                    for d in 0..pooled {
                        let p = oc[d] * strides[d] + tap[d];
                        if p < pad[d] || p - pad[d] >= dims[d] {
                            oob = true;
                            break;
                        }
                        ib += (p - pad[d]) * in_strides[d];
                    }
                    if !oob {
                        for ci in 0..c {
                            let a = out[ob + ci];
                            let t = src[ib + ci];
                            out[ob + ci] = if a.is_nan() || t.is_nan() {
                                f32::NAN
                            } else if is_max {
                                a.max(t)
                            } else {
                                a.min(t)
                            };
                        }
                    }
                    for d in (0..pooled).rev() {
                        tap[d] += 1;
                        if tap[d] < win[d] {
                            break;
                        }
                        tap[d] = 0;
                    }
                }
                for d in (0..pooled).rev() {
                    oc[d] += 1;
                    if oc[d] < out_dims[d] {
                        break;
                    }
                    oc[d] = 0;
                }
            }
            out
        }
        // Small (edge-case heavy, single-thread) + large (exercises the THREADED partition).
        // (dims, window, strides, pad_low) — each a rank-4 [N,H,W,C] descriptor.
        type Cfg4 = ([usize; 4], [usize; 4], [usize; 4], [usize; 4]);
        let configs: &[Cfg4] = &[
            ([2, 15, 13, 20], [1, 3, 3, 1], [1, 2, 2, 1], [0, 1, 1, 0]),
            ([8, 56, 56, 64], [1, 3, 3, 1], [1, 2, 2, 1], [0, 0, 0, 0]),
        ];
        for (dims, win, strides, pad) in configs {
            let out_dims: Vec<usize> = (0..4)
                .map(|d| (dims[d] + 2 * pad[d] - win[d]) / strides[d] + 1)
                .collect();
            let total: usize = dims.iter().product();
            for is_max in [true, false] {
                for variant in 0..3 {
                    let mut src: Vec<f32> = (0..total)
                        .map(|i| ((i % 103) as f32) * 0.41 - 21.0)
                        .collect();
                    if variant == 1 {
                        src[total / 3] = f32::NAN;
                        src[total / 2] = f32::NAN;
                    } else if variant == 2 {
                        for (i, v) in src.iter_mut().enumerate() {
                            *v = if i % 2 == 0 { 0.0 } else { -0.0 };
                        }
                    }
                    let got = super::reduce_window_simd_channel_maxmin_f32(
                        &src, dims, win, strides, pad, &out_dims, is_max,
                    );
                    let want = naive_f32(&src, dims, win, strides, pad, &out_dims, is_max);
                    assert_eq!(
                        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "f32 SIMD-channel maxpool mismatch (dims={dims:?} is_max={is_max} variant={variant})"
                    );
                }
            }
        }
    }

    // SIMD-over-channel SUM (avg/sum pooling) bit-identity guard: the f64/f32 SIMD sum fns must
    // match the REAL scalar odometer (eval_reduce_window_dense_float) bit-for-bit, incl. NaN, padded
    // windows, and large (threaded) shapes. f32 verifies the f64-accumulate-then-round path.
    #[test]
    fn sumpool_simd_channel_bit_identical() {
        let in_strides_of = |dims: &[usize]| {
            let r = dims.len();
            let mut s = vec![1usize; r];
            for d in (0..r - 1).rev() {
                s[d] = s[d + 1] * dims[d + 1];
            }
            s
        };
        type Cfg = (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>);
        let configs: &[Cfg] = &[
            (
                vec![2, 16, 16, 12],
                vec![1, 3, 3, 1],
                vec![1, 2, 2, 1],
                vec![0, 0, 0, 0],
            ),
            (
                vec![2, 16, 16, 12],
                vec![1, 3, 3, 1],
                vec![1, 1, 1, 1],
                vec![0, 1, 1, 0],
            ),
            (
                vec![8, 56, 56, 64],
                vec![1, 3, 3, 1],
                vec![1, 2, 2, 1],
                vec![0, 0, 0, 0],
            ),
        ];
        for (dims, win, strides, pad) in configs {
            let rank = dims.len();
            let out_dims: Vec<usize> = (0..rank)
                .map(|d| (dims[d] + 2 * pad[d] - win[d]) / strides[d] + 1)
                .collect();
            let out_dims_u32: Vec<u32> = out_dims.iter().map(|&d| d as u32).collect();
            let total_out: usize = out_dims.iter().product();
            let in_strides = in_strides_of(dims);
            let win_dil = vec![1usize; rank];
            let total: usize = dims.iter().product();
            for variant in 0..2 {
                let mut s64: Vec<f64> = (0..total)
                    .map(|i| ((i % 113) as f64) * 0.29 - 16.0)
                    .collect();
                if variant == 1 {
                    s64[total / 3] = f64::NAN;
                }
                let s32: Vec<f32> = s64.iter().map(|&v| v as f32).collect();
                // f64 vs the real odometer.
                let want64 = match super::eval_reduce_window_dense_float(
                    DType::F64,
                    "add",
                    &s64,
                    win,
                    strides,
                    &win_dil,
                    &out_dims_u32,
                    pad,
                    dims,
                    &in_strides,
                    total_out,
                )
                .unwrap()
                {
                    Value::Tensor(t) => t.elements.as_f64_slice().unwrap().to_vec(),
                    _ => panic!(),
                };
                let got64 = super::reduce_window_simd_channel_sum_f64(
                    &s64, dims, win, strides, pad, &out_dims,
                );
                assert_eq!(
                    got64.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want64.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "f64 SIMD-channel sumpool mismatch (dims={dims:?} variant={variant})"
                );
                // f32: odometer widens s32→f64 then rounds; SIMD does the same.
                let widened: Vec<f64> = s32.iter().map(|&v| f64::from(v)).collect();
                let want32 = match super::eval_reduce_window_dense_float(
                    DType::F32,
                    "add",
                    &widened,
                    win,
                    strides,
                    &win_dil,
                    &out_dims_u32,
                    pad,
                    dims,
                    &in_strides,
                    total_out,
                )
                .unwrap()
                {
                    Value::Tensor(t) => t.elements.as_f32_slice().unwrap().to_vec(),
                    _ => panic!(),
                };
                let got32 = super::reduce_window_simd_channel_sum_f32(
                    &s32, dims, win, strides, pad, &out_dims,
                );
                assert_eq!(
                    got32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want32.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "f32 SIMD-channel sumpool mismatch (dims={dims:?} variant={variant})"
                );
            }
        }
    }

    #[test]
    fn rank3_sum_pool_xlane_simd_matches_dense_float() {
        for (dims, win) in [
            ([9usize, 10usize, 17usize], [3usize, 2usize, 5usize]),
            ([4usize, 5usize, 6usize], [2usize, 3usize, 4usize]),
        ] {
            let out_dims = [
                dims[0] - win[0] + 1,
                dims[1] - win[1] + 1,
                dims[2] - win[2] + 1,
            ];
            let total: usize = dims.iter().product();
            let mut src: Vec<f64> = (0..total)
                .map(|i| match i % 17 {
                    0 => -0.0,
                    1 => f64::INFINITY,
                    2 => f64::NEG_INFINITY,
                    _ => ((i as f64) * 0.037).sin() * 11.0 - (i % 7) as f64,
                })
                .collect();
            src[total / 3] = f64::NAN;
            let input_strides = [dims[1] * dims[2], dims[2], 1usize];
            let want = match super::eval_reduce_window_dense_float(
                DType::F64,
                "sum",
                &src,
                &win,
                &[1, 1, 1],
                &[1, 1, 1],
                &[out_dims[0] as u32, out_dims[1] as u32, out_dims[2] as u32],
                &[0, 0, 0],
                &dims,
                &input_strides,
                out_dims.iter().product(),
            )
            .unwrap()
            {
                Value::Tensor(t) => t.elements.as_f64_slice().unwrap().to_vec(),
                _ => panic!(),
            };
            let got = super::reduce_window_rank3_f64_sum_xlane(&src, dims, win, out_dims);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "rank3 x-lane sumpool mismatch dims={dims:?} win={win:?}"
            );
        }
    }

    // BOLD-VERIFY prototype: FUSED bf16 maxpool — widen bf16→f32 INLINE per tap (no f64
    // intermediate; the running max is kept as bf16 bits, exact since max-of-bf16 is bf16),
    // f32x16 NaN-aware max, store (bits>>16). Reads only the half-size bf16 input. Verifies
    // bit-identity vs eval_primitive (the widen→f64-pool→round path) + measures vs JAX bf16 2.80ms.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_fused_bf16_maxpool_proto() {
        use std::time::Instant;

        #[allow(clippy::too_many_arguments)]
        fn fused(
            bits: &[u16],
            nb: usize,
            h: usize,
            w: usize,
            c: usize,
            kh: usize,
            kw: usize,
            sh: usize,
            sw: usize,
        ) -> Vec<u16> {
            use std::simd::cmp::SimdPartialEq;
            use std::simd::num::{SimdFloat, SimdUint};
            use std::simd::{Mask, Select, Simd};
            type F = Simd<f32, 16>;
            type U16 = Simd<u16, 16>;
            type U32 = Simd<u32, 16>;
            let oh = (h - kh) / sh + 1;
            let ow = (w - kw) / sw + 1;
            let mut out = vec![0xFF80u16; nb * oh * ow * c]; // bf16 -inf
            let c16 = c - c % 16;
            let widen = |u: U16| F::from_bits(u.cast::<u32>() << U32::splat(16));
            for ni in 0..nb {
                for ohi in 0..oh {
                    for owi in 0..ow {
                        let ob = ((ni * oh + ohi) * ow + owi) * c;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = ohi * sh + khi;
                                let iw = owi * sw + kwi;
                                let ib = ((ni * h + ih) * w + iw) * c;
                                let mut ci = 0;
                                while ci < c16 {
                                    let a = widen(U16::from_slice(&out[ob + ci..ob + ci + 16]));
                                    let t = widen(U16::from_slice(&bits[ib + ci..ib + ci + 16]));
                                    let m = a.simd_max(t);
                                    let nanm: Mask<i32, 16> = a.simd_ne(a) | t.simd_ne(t);
                                    let res = nanm.select(F::splat(f32::NAN), m);
                                    let obits: U16 =
                                        (res.to_bits() >> U32::splat(16)).cast::<u16>();
                                    obits.copy_to_slice(&mut out[ob + ci..ob + ci + 16]);
                                    ci += 16;
                                }
                                while ci < c {
                                    let af = f32::from_bits((out[ob + ci] as u32) << 16);
                                    let tf = f32::from_bits((bits[ib + ci] as u32) << 16);
                                    let r = if af.is_nan() || tf.is_nan() {
                                        f32::NAN
                                    } else {
                                        af.max(tf)
                                    };
                                    out[ob + ci] = (r.to_bits() >> 16) as u16;
                                    ci += 1;
                                }
                            }
                        }
                    }
                }
            }
            out
        }

        // Correctness vs eval_primitive (the current widen→f64-pool→round path) incl NaN.
        for nan_case in [false, true] {
            let (nb, h, w, c, win, stride) = (2usize, 16usize, 16usize, 32usize, 3usize, 2usize);
            let total = nb * h * w * c;
            let mut data: Vec<f64> = (0..total).map(|i| ((i % 53) as f64) * 0.4 - 10.0).collect();
            if nan_case {
                data[101] = f64::NAN;
                data[500] = f64::NAN;
            }
            let bits: Vec<u16> = data
                .iter()
                .map(|&v| match fj_core::Literal::from_bf16_f64(v) {
                    fj_core::Literal::BF16Bits(b) => b,
                    _ => 0,
                })
                .collect();
            let tensor = Value::Tensor(
                TensorValue::new_half_float_values(
                    DType::BF16,
                    Shape {
                        dims: vec![nb as u32, h as u32, w as u32, c as u32],
                    },
                    bits.clone(),
                )
                .unwrap(),
            );
            let p = rw_params("max", "1,3,3,1", "1,2,2,1");
            let want =
                match eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&tensor), &p)
                    .unwrap()
                {
                    Value::Tensor(t) => super::reduce_window_dense_f64_view(&t)
                        .unwrap()
                        .into_owned(),
                    _ => panic!(),
                };
            let got_bits = fused(&bits, nb, h, w, c, win, win, stride, stride);
            let got: Vec<f64> = got_bits
                .iter()
                .map(|&b| fj_core::Literal::BF16Bits(b).as_f64().unwrap())
                .collect();
            assert_eq!(
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "fused bf16 maxpool diverges (nan_case={nan_case})"
            );
        }

        // Speed (CNN shape).
        let (nb, h, w, c, win, stride) = (8usize, 112usize, 112usize, 64usize, 3usize, 2usize);
        let bits: Vec<u16> = (0..nb * h * w * c)
            .map(
                |i| match fj_core::Literal::from_bf16_f64(((i % 9973) as f64) * 0.013 - 50.0) {
                    fj_core::Literal::BF16Bits(b) => b,
                    _ => 0,
                },
            )
            .collect();
        let _ = fused(&bits, nb, h, w, c, win, win, stride, stride);
        let mut best = f64::MAX;
        for _ in 0..15 {
            let t = Instant::now();
            let r = fused(&bits, nb, h, w, c, win, win, stride, stride);
            best = best.min(t.elapsed().as_secs_f64());
            std::hint::black_box(&r);
        }
        println!(
            "BENCH fused-bf16-maxpool 112x112x64 3x3/s2: {:.4}ms (current 26ms, JAX 2.80ms)",
            best * 1e3
        );
    }

    // bf16/f16 channel-last pooling: the fast path (eval_primitive, widen→f64-SIMD→round) must equal
    // the REAL scalar odometer (eval_reduce_window_dense_float) for max + sum, bit-for-bit.
    #[test]
    fn half_pool_simd_channel_bit_identical() {
        // C=20: exercises the SIMD body (2×8 lanes) AND the c%8/c%16 scalar tail (the f16 fused
        // maxmin path's tail + the bf16 path's tail).
        let dims = vec![2usize, 15, 13, 20];
        let win = vec![1usize, 3, 3, 1];
        let strides = vec![1usize, 2, 2, 1];
        let pad = vec![0usize, 0, 0, 0];
        let rank = 4;
        let out_dims: Vec<usize> = (0..rank)
            .map(|d| (dims[d] - win[d]) / strides[d] + 1)
            .collect();
        let out_u32: Vec<u32> = out_dims.iter().map(|&d| d as u32).collect();
        let total_out: usize = out_dims.iter().product();
        let mut in_strides = vec![1usize; rank];
        for d in (0..rank - 1).rev() {
            in_strides[d] = in_strides[d + 1] * dims[d + 1];
        }
        let dil = vec![1usize; rank];
        let total: usize = dims.iter().product();
        let data: Vec<f64> = (0..total).map(|i| ((i % 37) as f64) * 0.5 - 9.0).collect();
        for dt in [DType::BF16, DType::F16] {
            let bits: Vec<u16> = data
                .iter()
                .map(|&v| {
                    let lit = if dt == DType::BF16 {
                        Literal::from_bf16_f64(v)
                    } else {
                        Literal::from_f16_f64(v)
                    };
                    match lit {
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                        _ => 0,
                    }
                })
                .collect();
            let tensor = Value::Tensor(
                TensorValue::new_half_float_values(
                    dt,
                    Shape {
                        dims: dims.iter().map(|&d| d as u32).collect(),
                    },
                    bits,
                )
                .unwrap(),
            );
            let tref = match &tensor {
                Value::Tensor(t) => t,
                _ => unreachable!(),
            };
            let widened = super::reduce_window_dense_f64_view(tref).unwrap();
            // eval_primitive's sum reduce_op is "sum" (not "add"); dense_float accepts either.
            for op in ["max", "sum"] {
                let want = super::eval_reduce_window_dense_float(
                    dt,
                    op,
                    widened.as_ref(),
                    &win,
                    &strides,
                    &dil,
                    &out_u32,
                    &pad,
                    &dims,
                    &in_strides,
                    total_out,
                )
                .unwrap();
                let p = rw_params(op, "1,3,3,1", "1,2,2,1");
                let got =
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&tensor), &p)
                        .unwrap();
                // Compare VALUES (widen each half output→f64, exact) — not Value structs, which
                // can differ by dense-vs-boxed representation while value-equal (Arc-COW gotcha).
                let to_f64 = |v: &Value| match v {
                    Value::Tensor(t) => {
                        super::reduce_window_dense_f64_view(t).unwrap().into_owned()
                    }
                    _ => panic!(),
                };
                let wf = to_f64(&want);
                let gf = to_f64(&got);
                for i in 0..wf.len().min(gf.len()) {
                    if wf[i].to_bits() != gf[i].to_bits() {
                        panic!(
                            "half mismatch dt={dt:?} op={op} len(w={} g={}) idx={i} want={} got={}",
                            wf.len(),
                            gf.len(),
                            wf[i],
                            gf[i]
                        );
                    }
                }
                assert_eq!(
                    wf.len(),
                    gf.len(),
                    "half pooling len mismatch (dt={dt:?} op={op})"
                );
            }
        }
    }

    // Parity gate for the separable running-sum sum-pooling fast path: at a window large enough to trigger
    // it (11x11, VALID, stride 1) the separable result must match the naive nested-fold reference within
    // tolerance (running-sum reassociation, ~1e-14 relative for f64).
    #[test]
    fn reduce_window_sum_separable_matches_naive() {
        let (rows, cols, w) = (256usize, 256usize, 11usize);
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| ((i % 9973) as f64) * 1e-2 - 50.0)
            .collect();
        let x = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let mut p = BTreeMap::new();
        p.insert("reduce_op".to_owned(), "sum".to_owned());
        p.insert("window_dimensions".to_owned(), format!("{w},{w}"));
        p.insert("window_strides".to_owned(), "1,1".to_owned());
        let Value::Tensor(out) =
            eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&x), &p).unwrap()
        else {
            panic!("tensor");
        };
        let got = out.elements.as_f64_slice().expect("dense f64");
        let (orow, ocol) = (rows - w + 1, cols - w + 1);
        assert_eq!(got.len(), orow * ocol);
        let mut max_rel = 0.0f64;
        for r in 0..orow {
            for c in 0..ocol {
                let mut accum = 0.0;
                for wr in 0..w {
                    for wc in 0..w {
                        accum += data[(r + wr) * cols + (c + wc)];
                    }
                }
                let g = got[r * ocol + c];
                let denom = accum.abs().max(1.0);
                max_rel = max_rel.max((g - accum).abs() / denom);
            }
        }
        assert!(
            max_rel < 1e-9,
            "separable sum-pool drifted: max relative error {max_rel:e}"
        );
    }

    // f32 parity gate for separable_reduce_window_sum_f32 (widen-to-f64 accumulate, narrow output): within
    // tolerance of the naive f64-accumulate-then-round fold at a window that triggers the fast path.
    #[test]
    fn reduce_window_sum_separable_f32_matches_naive() {
        let (rows, cols, w) = (256usize, 256usize, 11usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 9973) as f32) * 1e-2 - 50.0)
            .collect();
        let x = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let mut p = BTreeMap::new();
        p.insert("reduce_op".to_owned(), "sum".to_owned());
        p.insert("window_dimensions".to_owned(), format!("{w},{w}"));
        p.insert("window_strides".to_owned(), "1,1".to_owned());
        let Value::Tensor(out) =
            eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&x), &p).unwrap()
        else {
            panic!("tensor");
        };
        let got = out.elements.as_f32_slice().expect("dense f32");
        let (orow, ocol) = (rows - w + 1, cols - w + 1);
        assert_eq!(got.len(), orow * ocol);
        let mut max_rel = 0.0f64;
        for r in 0..orow {
            for c in 0..ocol {
                let mut accum = 0.0f64;
                for wr in 0..w {
                    for wc in 0..w {
                        accum += f64::from(data[(r + wr) * cols + (c + wc)]);
                    }
                }
                let reference = accum as f32;
                let denom = f64::from(reference).abs().max(1.0);
                max_rel = max_rel
                    .max((f64::from(got[r * ocol + c]) - f64::from(reference)).abs() / denom);
            }
        }
        assert!(
            max_rel < 1e-5,
            "separable f32 sum-pool drifted: {max_rel:e}"
        );
    }

    // Large-window SUM pooling vs JAX (measured JAX f64 [2048,2048] VALID: win11x11 8.18ms, win31x31
    // 35.3ms; f32 win11 6.98ms, win31 34.2ms). The general rank-2 sum path is naive O(out*wr*wc); sum is
    // SEPARABLE (two 1-D running-sum passes = O(input)) — check the gap.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_sum_f32_vs_jax() {
        use std::time::Instant;
        let n = 2048usize;
        let data: Vec<f32> = (0..n * n)
            .map(|i| ((i % 9973) as f32) * 1e-3 - 5.0)
            .collect();
        let x = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![n as u32, n as u32],
                },
                data,
            )
            .unwrap(),
        );
        for (w, jax) in [(11usize, 6.98), (31usize, 34.2)] {
            let mut p = BTreeMap::new();
            p.insert("reduce_op".to_owned(), "sum".to_owned());
            p.insert("window_dimensions".to_owned(), format!("{w},{w}"));
            p.insert("window_strides".to_owned(), "1,1".to_owned());
            let f = || {
                std::hint::black_box(
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&x), &p).unwrap(),
                );
            };
            f();
            let mut b = f64::MAX;
            for _ in 0..6 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!(
                "fj-lax reduce_window sum f32 [2048,2048] win{w}x{w}: {:.3}ms | JAX={jax}ms",
                b * 1e3
            );
        }
    }

    // Large-window SUM pooling vs JAX (measured JAX f64 [2048,2048] VALID: win11x11 8.18ms, win31x31
    // 35.3ms). The general rank-2 f64 sum path is naive O(out*wr*wc); sum is SEPARABLE (two 1-D running-sum
    // passes = O(input)) — check the gap.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_sum_vs_jax() {
        use std::time::Instant;
        let n = 2048usize;
        let data: Vec<f64> = (0..n * n)
            .map(|i| ((i % 9973) as f64) * 1e-3 - 5.0)
            .collect();
        let x = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32, n as u32],
                },
                data,
            )
            .unwrap(),
        );
        for (w, jax) in [(11usize, 8.18), (31usize, 35.3)] {
            let mut p = BTreeMap::new();
            p.insert("reduce_op".to_owned(), "sum".to_owned());
            p.insert("window_dimensions".to_owned(), format!("{w},{w}"));
            p.insert("window_strides".to_owned(), "1,1".to_owned());
            let f = || {
                std::hint::black_box(
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&x), &p).unwrap(),
                );
            };
            f();
            let mut b = f64::MAX;
            for _ in 0..6 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!(
                "fj-lax reduce_window sum f64 [2048,2048] win{w}x{w}: {:.3}ms | JAX={jax}ms",
                b * 1e3
            );
        }
    }

    // bf16 pooling production timing (eval_primitive → fast path) vs JAX bf16 (max 2.80ms, sum 2.96ms).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_bf16_pool_cnn() {
        use std::time::Instant;
        let (nb, hw, c, win, stride) = (8usize, 112usize, 64usize, 3usize, 2usize);
        let total = nb * hw * hw * c;
        let bits: Vec<u16> = (0..total)
            .map(
                |i| match Literal::from_bf16_f64(((i % 9973) as f64) * 0.013 - 50.0) {
                    Literal::BF16Bits(b) => b,
                    _ => 0,
                },
            )
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::BF16,
                Shape {
                    dims: vec![nb as u32, hw as u32, hw as u32, c as u32],
                },
                bits,
            )
            .unwrap(),
        );
        for op in ["max", "sum"] {
            let p = rw_params(
                op,
                &format!("1,{win},{win},1"),
                &format!("1,{stride},{stride},1"),
            );
            let run = || {
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&tensor), &p).unwrap()
            };
            let _ = run();
            let mut best = f64::MAX;
            for _ in 0..15 {
                let t = Instant::now();
                let r = run();
                best = best.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&r);
            }
            println!(
                "BENCH bf16 {op}pool 112x112x64 3x3/s2: fj-lax={:.4}ms",
                best * 1e3
            );
        }
        // f16 (widen→f64 path with the SIMD f16 widen)
        let f16_bits: Vec<u16> = (0..total)
            .map(
                |i| match Literal::from_f16_f64(((i % 9973) as f64) * 0.013 - 50.0) {
                    Literal::F16Bits(b) => b,
                    _ => 0,
                },
            )
            .collect();
        let t16 = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::F16,
                Shape {
                    dims: vec![nb as u32, hw as u32, hw as u32, c as u32],
                },
                f16_bits,
            )
            .unwrap(),
        );
        for op in ["max", "sum"] {
            let p = rw_params(
                op,
                &format!("1,{win},{win},1"),
                &format!("1,{stride},{stride},1"),
            );
            let run =
                || eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&t16), &p).unwrap();
            let _ = run();
            let mut best = f64::MAX;
            for _ in 0..15 {
                let t = Instant::now();
                let r = run();
                best = best.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&r);
            }
            println!(
                "BENCH f16 {op}pool 112x112x64 3x3/s2: fj-lax={:.4}ms",
                best * 1e3
            );
        }
        // Same-binary A/B for f16 MAXpool: FUSED (eval_primitive, current) vs the prior widen-to-f64
        // odometer path (reduce_window_dense_f64_view + eval_reduce_window_dense_float).
        {
            let tref = match &t16 {
                Value::Tensor(t) => t,
                _ => unreachable!(),
            };
            let dims_u: Vec<usize> = vec![nb, hw, hw, c];
            let win_v = vec![1usize, win, win, 1];
            let str_v = vec![1usize, stride, stride, 1];
            let dil_v = vec![1usize; 4];
            let pad_v = vec![0usize; 4];
            let out_u32: Vec<u32> = (0..4)
                .map(|d| ((dims_u[d] - win_v[d]) / str_v[d] + 1) as u32)
                .collect();
            let total_out: usize = out_u32.iter().map(|&d| d as usize).product();
            let mut in_strides = vec![1usize; 4];
            for d in (0..3).rev() {
                in_strides[d] = in_strides[d + 1] * dims_u[d + 1];
            }
            let widened = super::reduce_window_dense_f64_view(tref).unwrap();
            let bench = |f: &dyn Fn()| {
                f();
                let mut b = f64::MAX;
                for _ in 0..15 {
                    let t = Instant::now();
                    f();
                    b = b.min(t.elapsed().as_secs_f64());
                }
                b * 1e3
            };
            for op in ["max", "sum"] {
                let p = rw_params(
                    op,
                    &format!("1,{win},{win},1"),
                    &format!("1,{stride},{stride},1"),
                );
                let fused = bench(&|| {
                    std::hint::black_box(
                        eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&t16), &p)
                            .unwrap(),
                    );
                });
                let odo = bench(&|| {
                    std::hint::black_box(
                        super::eval_reduce_window_dense_float(
                            DType::F16,
                            op,
                            widened.as_ref(),
                            &win_v,
                            &str_v,
                            &dil_v,
                            &out_u32,
                            &pad_v,
                            &dims_u,
                            &in_strides,
                            total_out,
                        )
                        .unwrap(),
                    );
                });
                println!(
                    "AB f16 {op}pool 112x112x64 3x3/s2: odometer(widen-f64)={odo:.4}ms fused={fused:.4}ms ({:.2}x)",
                    odo / fused
                );
            }
        }
    }

    // f32 rank-3 separable sum vs the generic dense-float per-tap loop (the current f32 volumetric
    // baseline). Tolerance match (f64 accumulate, narrow output) + same-binary A/B ratio.
    #[test]
    fn separable_sum_3d_f32_matches_generic() {
        for &(n, win) in &[(96usize, 6usize), (96, 9), (40, 7)] {
            let out = n - win + 1;
            let s32: Vec<f32> = (0..n * n * n)
                .map(|i| (((i % 9973) as f64) * 0.013 - 50.0) as f32)
                .collect();
            let sep = super::separable_reduce_window_sum_3d_f32(
                &s32,
                [n, n, n],
                [win, win, win],
                [out, out, out],
            )
            .expect("f32 separable should accept finite VALID input");
            // Generic dense-float baseline (f64 accumulate, narrow to f32) — same contract.
            let widened: Vec<f64> = s32.iter().map(|&v| f64::from(v)).collect();
            let dims = [n, n, n];
            let wd = [win, win, win];
            let st = [1usize, 1, 1];
            let dil = [1usize; 3];
            let pad = [0usize; 3];
            let out_u32 = [out as u32, out as u32, out as u32];
            let mut in_strides = [1usize; 3];
            for d in (0..2).rev() {
                in_strides[d] = in_strides[d + 1] * dims[d + 1];
            }
            let generic = super::eval_reduce_window_dense_float(
                DType::F32,
                "add",
                &widened,
                &wd,
                &st,
                &dil,
                &out_u32,
                &pad,
                &dims,
                &in_strides,
                out * out * out,
            )
            .unwrap();
            let gvals: Vec<f32> = generic
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    fj_core::Literal::F32Bits(b) => f32::from_bits(*b),
                    other => panic!("expected f32, got {other:?}"),
                })
                .collect();
            assert_eq!(sep.len(), gvals.len());
            let mut max_rel = 0.0f64;
            for (&a, &b) in sep.iter().zip(&gvals) {
                let denom = (b.abs() as f64).max(1.0);
                max_rel = max_rel.max((f64::from(a) - f64::from(b)).abs() / denom);
            }
            assert!(
                max_rel < 1e-5,
                "f32 separable 3D sum drifted at n={n} win={win}: max rel {max_rel:e}"
            );
        }
    }

    // i64 rank-3 separable sum is BIT-IDENTICAL to the naive wrapping fold (integer ring), incl.
    // a window large/dense enough to overflow i64 (verifies wrapping consistency).
    #[test]
    fn separable_sum_3d_i64_bit_identical_to_dense() {
        for &(n, win, scale) in &[(40usize, 6usize, 1i64), (40, 9, 1), (24, 7, 1i64 << 60)] {
            let out = n - win + 1;
            let s64: Vec<i64> = (0..n * n * n)
                .map(|i| ((i as i64 % 97) - 48).wrapping_mul(scale))
                .collect();
            let sep = super::separable_reduce_window_sum_3d_i64(
                &s64,
                [n, n, n],
                [win, win, win],
                [out, out, out],
            )
            .expect("i64 separable should accept VALID input");
            let dims = [n, n, n];
            let wd = [win, win, win];
            let st = [1usize, 1, 1];
            let dil = [1usize; 3];
            let pad = [0usize; 3];
            let out_u32 = [out as u32; 3];
            let mut in_strides = [1usize; 3];
            for d in (0..2).rev() {
                in_strides[d] = in_strides[d + 1] * dims[d + 1];
            }
            let naive = super::eval_reduce_window_dense_i64(
                DType::I64,
                "add",
                &s64,
                &wd,
                &st,
                &dil,
                &out_u32,
                &pad,
                &dims,
                &in_strides,
                out * out * out,
            )
            .unwrap();
            let nvals: Vec<i64> = naive
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    fj_core::Literal::I64(v) => *v,
                    other => panic!("expected i64, got {other:?}"),
                })
                .collect();
            assert_eq!(
                sep, nvals,
                "i64 separable 3D sum != naive at n={n} win={win}"
            );
        }
    }

    // N-D (rank-5 3D-CNN [N,D,H,W,C]) separable sum vs the generic dense-float per-tap loop.
    // Tolerance match (prefix-sum reassoc) + verifies general stride/pad on a 4-D case.
    #[test]
    fn separable_sum_nd_matches_generic() {
        // (input_dims, window, strides, pad)
        let cases: &[(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>)] = &[
            // rank-5 3D-CNN avg-pool, VALID unit stride.
            (
                vec![2, 16, 16, 16, 8],
                vec![1, 5, 5, 5, 1],
                vec![1, 1, 1, 1, 1],
                vec![0, 0, 0, 0, 0],
            ),
            // rank-4 strided + padded volumetric pool.
            (
                vec![3, 20, 20, 20],
                vec![1, 6, 6, 6],
                vec![1, 2, 2, 2],
                vec![0, 1, 1, 1],
            ),
        ];
        for (input_dims, window, strides, pad) in cases {
            let rank = input_dims.len();
            let total: usize = input_dims.iter().product();
            let src: Vec<f64> = (0..total)
                .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
                .collect();
            let out_dims: Vec<usize> = (0..rank)
                .map(|d| (input_dims[d] + 2 * pad[d] - window[d]) / strides[d] + 1)
                .collect();
            let sep = super::reduce_window_separable_sum_nd_f64(
                &src, input_dims, window, strides, pad, &out_dims,
            );
            let out_u32: Vec<u32> = out_dims.iter().map(|&d| d as u32).collect();
            let dil = vec![1usize; rank];
            let mut in_strides = vec![1usize; rank];
            for d in (0..rank - 1).rev() {
                in_strides[d] = in_strides[d + 1] * input_dims[d + 1];
            }
            let total_out: usize = out_dims.iter().product();
            let generic = super::eval_reduce_window_dense_float(
                DType::F64,
                "add",
                &src,
                window,
                strides,
                &dil,
                &out_u32,
                pad,
                input_dims,
                &in_strides,
                total_out,
            )
            .unwrap();
            let gvals: Vec<f64> = generic
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    fj_core::Literal::F64Bits(b) => f64::from_bits(*b),
                    other => panic!("expected f64, got {other:?}"),
                })
                .collect();
            assert_eq!(sep.len(), gvals.len());
            let mut max_rel = 0.0f64;
            for (&a, &b) in sep.iter().zip(&gvals) {
                max_rel = max_rel.max((a - b).abs() / b.abs().max(1.0));
            }
            assert!(
                max_rel < 1e-10,
                "N-D separable sum drifted (rank={rank}): max rel {max_rel:e}"
            );
        }
    }

    // Dense bit-count fast path (PopulationCount/CLZ/CTZ) is BIT-IDENTICAL to the per-Literal
    // generic reference across all integer dtypes, and an A/B shows the de-box+thread win.
    #[test]
    fn bitcount_dense_matches_generic_and_ab() {
        use std::time::Instant;
        let p = BTreeMap::new();
        let n = 1usize << 20;
        // Build each integer tensor via its proper typed constructor (typed tensors densify to the
        // backing `as_*_slice` the fast path reads); the per-Literal reference uses the SAME storage
        // literals (`as_slice`) the generic loop would walk.
        for dtype in [DType::I64, DType::I32, DType::U32, DType::U64] {
            let shape = Shape {
                dims: vec![n as u32],
            };
            let t = Value::Tensor(match dtype {
                DType::I64 => TensorValue::new_i64_values(
                    shape.clone(),
                    (0..n)
                        .map(|i| (i as i64).wrapping_mul(2_654_435_761) ^ ((i as i64) << 17))
                        .collect(),
                )
                .unwrap(),
                DType::I32 => TensorValue::new_i32_values(
                    shape.clone(),
                    (0..n)
                        .map(|i| i64::from((i as i32).wrapping_mul(40_503)))
                        .collect(),
                )
                .unwrap(),
                DType::U32 => TensorValue::new_u32_values(
                    shape.clone(),
                    (0..n)
                        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
                        .collect(),
                )
                .unwrap(),
                _ => TensorValue::new_u64_values(
                    shape.clone(),
                    (0..n)
                        .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        .collect(),
                )
                .unwrap(),
            });
            let lits: Vec<Literal> = t.as_tensor().unwrap().elements.iter().copied().collect();
            for prim in [
                Primitive::PopulationCount,
                Primitive::CountLeadingZeros,
                Primitive::CountTrailingZeros,
            ] {
                let got = eval_primitive(prim, std::slice::from_ref(&t), &p).unwrap();
                let gvals: Vec<i64> = got
                    .as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::I64(v) => *v,
                        other => panic!("expected i64 count, got {other:?}"),
                    })
                    .collect();
                // Reference: the per-Literal generic op.
                let want: Vec<i64> = lits
                    .iter()
                    .map(|&l| {
                        match super::apply_bitwise_unary_literal_with_dtype(prim, l, Some(dtype)) {
                            Some(Literal::I64(v)) => v,
                            other => panic!("ref produced {other:?}"),
                        }
                    })
                    .collect();
                assert_eq!(
                    gvals, want,
                    "bitcount {prim:?} {dtype:?} fast path != generic"
                );
            }
            // A/B: dense fast path (eval_primitive) vs the per-Literal generic loop.
            let best = |f: &dyn Fn() -> i64| -> u64 {
                let mut bb = u64::MAX;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    let acc = f();
                    std::hint::black_box(acc);
                    bb = bb.min(t0.elapsed().as_nanos() as u64);
                }
                bb
            };
            let fast = best(&|| {
                let v = eval_primitive(Primitive::PopulationCount, std::slice::from_ref(&t), &p)
                    .unwrap();
                match &v.as_tensor().unwrap().elements.as_slice()[0] {
                    Literal::I64(x) => *x,
                    _ => 0,
                }
            });
            let generic = best(&|| {
                let out: Vec<Literal> = lits
                    .iter()
                    .map(|&l| {
                        super::apply_bitwise_unary_literal_with_dtype(
                            Primitive::PopulationCount,
                            l,
                            Some(dtype),
                        )
                        .unwrap()
                    })
                    .collect();
                match out[0] {
                    Literal::I64(x) => x,
                    _ => 0,
                }
            });
            eprintln!(
                "[popcount {dtype:?} {n}] generic={:.3}ms dense={:.3}ms ratio={:.2}x",
                generic as f64 / 1e6,
                fast as f64 / 1e6,
                generic as f64 / fast as f64,
            );
        }
    }

    // N-D i64 separable sum is BIT-IDENTICAL to the naive wrapping fold (rank-5 + rank-4 strided).
    #[test]
    fn separable_sum_nd_i64_bit_identical() {
        let cases: &[(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>)] = &[
            (
                vec![2, 12, 12, 12, 4],
                vec![1, 5, 5, 5, 1],
                vec![1, 1, 1, 1, 1],
                vec![0, 0, 0, 0, 0],
            ),
            (
                vec![3, 16, 16, 16],
                vec![1, 6, 6, 6],
                vec![1, 2, 2, 2],
                vec![0, 1, 1, 1],
            ),
        ];
        for (input_dims, window, strides, pad) in cases {
            let rank = input_dims.len();
            let total: usize = input_dims.iter().product();
            let src: Vec<i64> = (0..total).map(|i| (i as i64 % 131) - 65).collect();
            let out_dims: Vec<usize> = (0..rank)
                .map(|d| (input_dims[d] + 2 * pad[d] - window[d]) / strides[d] + 1)
                .collect();
            let sep = super::reduce_window_separable_sum_nd_i64(
                &src, input_dims, window, strides, pad, &out_dims,
            );
            let out_u32: Vec<u32> = out_dims.iter().map(|&d| d as u32).collect();
            let dil = vec![1usize; rank];
            let mut in_strides = vec![1usize; rank];
            for d in (0..rank - 1).rev() {
                in_strides[d] = in_strides[d + 1] * input_dims[d + 1];
            }
            let total_out: usize = out_dims.iter().product();
            let naive = super::eval_reduce_window_dense_i64(
                DType::I64,
                "add",
                &src,
                window,
                strides,
                &dil,
                &out_u32,
                pad,
                input_dims,
                &in_strides,
                total_out,
            )
            .unwrap();
            let nvals: Vec<i64> = naive
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    fj_core::Literal::I64(v) => *v,
                    other => panic!("expected i64, got {other:?}"),
                })
                .collect();
            assert_eq!(sep, nvals, "N-D i64 separable != naive (rank={rank})");
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_sumpool_nd_i64_separable_ab() {
        use std::time::Instant;
        let input_dims = vec![2usize, 48, 48, 48, 16];
        let rank = input_dims.len();
        let total: usize = input_dims.iter().product();
        let src: Vec<i64> = (0..total).map(|i| (i as i64 % 131) - 65).collect();
        let best = |f: &dyn Fn() -> i64| -> u64 {
            let mut b = u64::MAX;
            for _ in 0..5 {
                let t0 = Instant::now();
                let acc = f();
                std::hint::black_box(acc);
                b = b.min(t0.elapsed().as_nanos() as u64);
            }
            b
        };
        for k in [3usize, 5] {
            let window = vec![1, k, k, k, 1];
            let strides = vec![1usize; rank];
            let pad = vec![0usize; rank];
            let out_dims: Vec<usize> = (0..rank).map(|d| (input_dims[d] - window[d]) + 1).collect();
            let out_u32: Vec<u32> = out_dims.iter().map(|&d| d as u32).collect();
            let dil = vec![1usize; rank];
            let mut in_strides = vec![1usize; rank];
            for d in (0..rank - 1).rev() {
                in_strides[d] = in_strides[d + 1] * input_dims[d + 1];
            }
            let total_out: usize = out_dims.iter().product();
            let naive = best(&|| {
                let v = super::eval_reduce_window_dense_i64(
                    DType::I64,
                    "add",
                    &src,
                    &window,
                    &strides,
                    &dil,
                    &out_u32,
                    &pad,
                    &input_dims,
                    &in_strides,
                    total_out,
                )
                .unwrap();
                match &v.as_tensor().unwrap().elements.as_slice()[0] {
                    fj_core::Literal::I64(b) => *b,
                    _ => 0,
                }
            });
            let sep = best(&|| {
                let v = super::reduce_window_separable_sum_nd_i64(
                    &src,
                    &input_dims,
                    &window,
                    &strides,
                    &pad,
                    &out_dims,
                );
                v[0].wrapping_add(v[v.len() / 2])
            });
            eprintln!(
                "[sumpool-nd-i64 rank5 48^3 win{k}] naive={:.3}ms separable={:.3}ms ratio={:.2}x (min of 5)",
                naive as f64 / 1e6,
                sep as f64 / 1e6,
                naive as f64 / sep as f64,
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_sumpool_nd_separable_ab() {
        use std::time::Instant;
        // rank-5 3D-CNN avg-pool: [N,D,H,W,C], windows on D/H/W.
        let input_dims = vec![2usize, 48, 48, 48, 16];
        let rank = input_dims.len();
        let total: usize = input_dims.iter().product();
        let src: Vec<f64> = (0..total)
            .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
            .collect();
        let best = |f: &dyn Fn() -> f64| -> u64 {
            let mut b = u64::MAX;
            for _ in 0..5 {
                let t0 = Instant::now();
                let acc = f();
                let dt = t0.elapsed().as_nanos() as u64;
                std::hint::black_box(acc);
                b = b.min(dt);
            }
            b
        };
        for k in [3usize, 5] {
            let window = vec![1, k, k, k, 1];
            let strides = vec![1usize; rank];
            let pad = vec![0usize; rank];
            let out_dims: Vec<usize> = (0..rank)
                .map(|d| (input_dims[d] - window[d]) / strides[d] + 1)
                .collect();
            let out_u32: Vec<u32> = out_dims.iter().map(|&d| d as u32).collect();
            let dil = vec![1usize; rank];
            let mut in_strides = vec![1usize; rank];
            for d in (0..rank - 1).rev() {
                in_strides[d] = in_strides[d + 1] * input_dims[d + 1];
            }
            let total_out: usize = out_dims.iter().product();
            let generic = best(&|| {
                let v = super::eval_reduce_window_dense_float(
                    DType::F64,
                    "add",
                    &src,
                    &window,
                    &strides,
                    &dil,
                    &out_u32,
                    &pad,
                    &input_dims,
                    &in_strides,
                    total_out,
                )
                .unwrap();
                match &v.as_tensor().unwrap().elements.as_slice()[0] {
                    fj_core::Literal::F64Bits(b) => f64::from_bits(*b),
                    _ => 0.0,
                }
            });
            let sep = best(&|| {
                let v = super::reduce_window_separable_sum_nd_f64(
                    &src,
                    &input_dims,
                    &window,
                    &strides,
                    &pad,
                    &out_dims,
                );
                v[0] + v[v.len() / 2]
            });
            eprintln!(
                "[sumpool-nd rank5 48^3 win{k}] generic={:.3}ms separable={:.3}ms ratio={:.2}x (min of 5)",
                generic as f64 / 1e6,
                sep as f64 / 1e6,
                generic as f64 / sep as f64,
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_sumpool_3d_i64_separable_ab() {
        use std::time::Instant;
        let n = 96usize;
        let s64: Vec<i64> = (0..n * n * n).map(|i| (i as i64 % 97) - 48).collect();
        let best = |f: &dyn Fn() -> i64| -> u64 {
            let mut b = u64::MAX;
            for _ in 0..7 {
                let t0 = Instant::now();
                let acc = f();
                let dt = t0.elapsed().as_nanos() as u64;
                std::hint::black_box(acc);
                b = b.min(dt);
            }
            b
        };
        for win in [6usize, 9] {
            let out = n - win + 1;
            let dims = [n, n, n];
            let wd = [win, win, win];
            let st = [1usize, 1, 1];
            let dil = [1usize; 3];
            let pad = [0usize; 3];
            let out_u32 = [out as u32; 3];
            let mut in_strides = [1usize; 3];
            for d in (0..2).rev() {
                in_strides[d] = in_strides[d + 1] * dims[d + 1];
            }
            let naive = best(&|| {
                let v = super::eval_reduce_window_dense_i64(
                    DType::I64,
                    "add",
                    &s64,
                    &wd,
                    &st,
                    &dil,
                    &out_u32,
                    &pad,
                    &dims,
                    &in_strides,
                    out * out * out,
                )
                .unwrap();
                match &v.as_tensor().unwrap().elements.as_slice()[0] {
                    fj_core::Literal::I64(b) => *b,
                    _ => 0,
                }
            });
            let sep = best(&|| {
                let v = super::separable_reduce_window_sum_3d_i64(&s64, dims, wd, [out, out, out])
                    .unwrap();
                v[0].wrapping_add(v[v.len() / 2])
            });
            eprintln!(
                "[sumpool3d-i64 {n}^3 win{win}] naive={:.3}ms separable={:.3}ms ratio={:.2}x (min of 7)",
                naive as f64 / 1e6,
                sep as f64 / 1e6,
                naive as f64 / sep as f64,
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_sumpool_3d_f32_separable_ab() {
        use std::time::Instant;
        let n = 96usize;
        let s32: Vec<f32> = (0..n * n * n)
            .map(|i| (((i % 9973) as f64) * 0.013 - 50.0) as f32)
            .collect();
        let widened: Vec<f64> = s32.iter().map(|&v| f64::from(v)).collect();
        let best = |f: &dyn Fn() -> f32| -> u64 {
            let mut b = u64::MAX;
            for _ in 0..7 {
                let t0 = Instant::now();
                let acc = f();
                let dt = t0.elapsed().as_nanos() as u64;
                std::hint::black_box(acc);
                b = b.min(dt);
            }
            b
        };
        for win in [6usize, 9] {
            let out = n - win + 1;
            let dims = [n, n, n];
            let wd = [win, win, win];
            let st = [1usize, 1, 1];
            let dil = [1usize; 3];
            let pad = [0usize; 3];
            let out_u32 = [out as u32; 3];
            let mut in_strides = [1usize; 3];
            for d in (0..2).rev() {
                in_strides[d] = in_strides[d + 1] * dims[d + 1];
            }
            let generic = best(&|| {
                let v = super::eval_reduce_window_dense_float(
                    DType::F32,
                    "add",
                    &widened,
                    &wd,
                    &st,
                    &dil,
                    &out_u32,
                    &pad,
                    &dims,
                    &in_strides,
                    out * out * out,
                )
                .unwrap();
                match &v.as_tensor().unwrap().elements.as_slice()[0] {
                    fj_core::Literal::F32Bits(b) => f32::from_bits(*b),
                    _ => 0.0,
                }
            });
            let sep = best(&|| {
                let v = super::separable_reduce_window_sum_3d_f32(&s32, dims, wd, [out, out, out])
                    .unwrap();
                v[0] + v[v.len() / 2]
            });
            eprintln!(
                "[sumpool3d-f32 {n}^3 win{win}] generic={:.3}ms separable={:.3}ms ratio={:.2}x (min of 7)",
                generic as f64 / 1e6,
                sep as f64 / 1e6,
                generic as f64 / sep as f64,
            );
        }
    }

    // Correctness + same-binary A/B for the rank-3 separable running-window-SUM (3D volumetric
    // sum pooling) vs the shipped naive SIMD-x-lane path. Tolerance match (running-sum reassoc).
    #[test]
    fn separable_sum_3d_matches_xlane() {
        for &(n, win) in &[(96usize, 5usize), (96, 9), (40, 3), (33, 7)] {
            let out = n - win + 1;
            let src: Vec<f64> = (0..n * n * n)
                .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
                .collect();
            let sep = super::separable_reduce_window_sum_3d_f64(
                &src,
                [n, n, n],
                [win, win, win],
                [out, out, out],
            )
            .expect("separable should accept finite VALID input");
            let naive = super::reduce_window_rank3_f64_sum_xlane(
                &src,
                [n, n, n],
                [win, win, win],
                [out, out, out],
            );
            assert_eq!(sep.len(), naive.len());
            let mut max_rel = 0.0f64;
            for (&a, &b) in sep.iter().zip(&naive) {
                let denom = b.abs().max(1.0);
                max_rel = max_rel.max((a - b).abs() / denom);
            }
            assert!(
                max_rel < 1e-10,
                "separable 3D sum drifted at n={n} win={win}: max rel {max_rel:e}"
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_sumpool_3d_separable_ab() {
        use std::time::Instant;
        let n = 96usize;
        let src: Vec<f64> = (0..n * n * n)
            .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
            .collect();
        let best = |f: &dyn Fn() -> f64| -> u64 {
            let mut b = u64::MAX;
            for _ in 0..7 {
                let t0 = Instant::now();
                let acc = f();
                let dt = t0.elapsed().as_nanos() as u64;
                std::hint::black_box(acc);
                b = b.min(dt);
            }
            b
        };
        for win in [5usize, 6, 7, 8, 9] {
            let out = n - win + 1;
            let naive = best(&|| {
                let v = super::reduce_window_rank3_f64_sum_xlane(
                    &src,
                    [n, n, n],
                    [win, win, win],
                    [out, out, out],
                );
                v[0] + v[v.len() / 2]
            });
            let sep = best(&|| {
                let v = super::separable_reduce_window_sum_3d_f64(
                    &src,
                    [n, n, n],
                    [win, win, win],
                    [out, out, out],
                )
                .unwrap();
                v[0] + v[v.len() / 2]
            });
            eprintln!(
                "[sumpool3d {n}^3 win{win}] xlane={:.3}ms separable={:.3}ms ratio={:.2}x (min of 7)",
                naive as f64 / 1e6,
                sep as f64 / 1e6,
                naive as f64 / sep as f64,
            );
        }
    }

    // Same-binary A/B: scalar odometer (eval_reduce_window_dense_float) vs the SIMD-channel sum, per
    // CNN shape/dtype. Sum/avg pooling baseline vs the new fast path.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_sumpool_simd_ab() {
        use std::time::Instant;
        for (nb, hw, c, win, stride) in [
            (8usize, 112usize, 64usize, 3usize, 2usize),
            (8, 56, 128, 2, 2),
        ] {
            let dims = [nb, hw, hw, c];
            let wd = [1usize, win, win, 1];
            let st = [1usize, stride, stride, 1];
            let pad = [0usize, 0, 0, 0];
            let dil = [1usize; 4];
            let out_dims: Vec<usize> = (0..4).map(|d| (dims[d] - wd[d]) / st[d] + 1).collect();
            let out_u32: Vec<u32> = out_dims.iter().map(|&d| d as u32).collect();
            let total_out: usize = out_dims.iter().product();
            let mut in_strides = [1usize; 4];
            for d in (0..3).rev() {
                in_strides[d] = in_strides[d + 1] * dims[d + 1];
            }
            let total = nb * hw * hw * c;
            let s64: Vec<f64> = (0..total)
                .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
                .collect();
            let s32: Vec<f32> = s64.iter().map(|&v| v as f32).collect();
            let w64: Vec<f64> = s32.iter().map(|&v| f64::from(v)).collect();
            let odo64 = || {
                super::eval_reduce_window_dense_float(
                    DType::F64,
                    "add",
                    &s64,
                    &wd,
                    &st,
                    &dil,
                    &out_u32,
                    &pad,
                    &dims,
                    &in_strides,
                    total_out,
                )
                .unwrap()
            };
            let simd64 = || {
                super::reduce_window_simd_channel_sum_f64(&s64, &dims, &wd, &st, &pad, &out_dims)
            };
            let odo32 = || {
                super::eval_reduce_window_dense_float(
                    DType::F32,
                    "add",
                    &w64,
                    &wd,
                    &st,
                    &dil,
                    &out_u32,
                    &pad,
                    &dims,
                    &in_strides,
                    total_out,
                )
                .unwrap()
            };
            let simd32 = || {
                super::reduce_window_simd_channel_sum_f32(&s32, &dims, &wd, &st, &pad, &out_dims)
            };
            let bench = |f: &dyn Fn()| {
                f();
                let mut b = f64::MAX;
                for _ in 0..15 {
                    let t = Instant::now();
                    f();
                    b = b.min(t.elapsed().as_secs_f64());
                }
                b * 1e3
            };
            let o64 = bench(&|| {
                std::hint::black_box(odo64());
            });
            let m64 = bench(&|| {
                std::hint::black_box(simd64());
            });
            let o32 = bench(&|| {
                std::hint::black_box(odo32());
            });
            let m32 = bench(&|| {
                std::hint::black_box(simd32());
            });
            println!(
                "AB sumpool N{nb} {hw}x{hw}x{c} {win}x{win}/s{stride}: f64 odo={o64:.4} simd={m64:.4} ({:.1}x) | f32 odo={o32:.4} simd={m32:.4} ({:.1}x)",
                o64 / m64,
                o32 / m32
            );
        }
    }

    // BOLD-VERIFY: COMMON CNN maxpool (small windows 2x2/3x3) does NOT hit the deque
    // (win_total <= 2*win_sum) — rank-4 [N,H,W,C] routes to the scalar dense_float odometer.
    // Measure it vs JAX (3x3/s2 = 9.34ms, 2x2/s2 = 2.11ms) — the contiguous channel axis (C) is
    // a clean SIMD target. Production eval_primitive head-to-head.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_maxpool_simd_thread_ab() {
        // Same-binary A/B: single-thread block fn vs the threaded wrapper, per CNN shape/dtype.
        use std::time::Instant;
        let cases = [
            (8usize, 112usize, 64usize, 3usize, 2usize),
            (8, 56, 128, 2, 2),
        ];
        for (nb, hw, c, win, stride) in cases {
            let dims = [nb, hw, hw, c];
            let wd = [1usize, win, win, 1];
            let st = [1usize, stride, stride, 1];
            let pad = [0usize, 0, 0, 0];
            let out_dims: Vec<usize> = (0..4).map(|d| (dims[d] - wd[d]) / st[d] + 1).collect();
            let mut in_strides = [1usize; 4];
            for d in (0..3).rev() {
                in_strides[d] = in_strides[d + 1] * dims[d + 1];
            }
            let win_total = win * win;
            let total = nb * hw * hw * c;
            let s64: Vec<f64> = (0..total)
                .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
                .collect();
            let s32: Vec<f32> = (0..total)
                .map(|i| ((i % 9973) as f32) * 0.013 - 50.0)
                .collect();
            let ot: usize = out_dims.iter().product();
            let serial64 = || {
                let mut o = vec![f64::NEG_INFINITY; ot];
                super::simd_channel_block_f64(
                    &s64,
                    &in_strides,
                    &wd,
                    &st,
                    &pad,
                    &out_dims,
                    &dims,
                    c,
                    win_total,
                    true,
                    0,
                    &mut o,
                );
                o
            };
            let thr64 = || {
                super::reduce_window_simd_channel_maxmin_f64(
                    &s64, &dims, &wd, &st, &pad, &out_dims, true,
                )
            };
            let serial32 = || {
                let mut o = vec![f32::NEG_INFINITY; ot];
                super::simd_channel_block_f32(
                    &s32,
                    &in_strides,
                    &wd,
                    &st,
                    &pad,
                    &out_dims,
                    &dims,
                    c,
                    win_total,
                    true,
                    0,
                    &mut o,
                );
                o
            };
            let thr32 = || {
                super::reduce_window_simd_channel_maxmin_f32(
                    &s32, &dims, &wd, &st, &pad, &out_dims, true,
                )
            };
            assert_eq!(
                serial64().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                thr64().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                serial32().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                thr32().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let bench = |f: &dyn Fn()| {
                f();
                let mut b = f64::MAX;
                for _ in 0..20 {
                    let t = Instant::now();
                    f();
                    b = b.min(t.elapsed().as_secs_f64());
                }
                b * 1e3
            };
            let s64b = bench(&|| {
                std::hint::black_box(serial64());
            });
            let t64b = bench(&|| {
                std::hint::black_box(thr64());
            });
            let s32b = bench(&|| {
                std::hint::black_box(serial32());
            });
            let t32b = bench(&|| {
                std::hint::black_box(thr32());
            });
            println!(
                "AB maxpool N{nb} {hw}x{hw}x{c} {win}x{win}/s{stride}: f64 serial={s64b:.4} thr={t64b:.4} ({:.2}x) | f32 serial={s32b:.4} thr={t32b:.4} ({:.2}x)",
                s64b / t64b,
                s32b / t32b
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_maxpool_cnn_small_window() {
        use std::time::Instant;
        let cases = [
            (8usize, 112usize, 64usize, 3usize, 2usize),
            (8, 56, 128, 2, 2),
        ];
        for (nb, hw, c, win, stride) in cases {
            let total = nb * hw * hw * c;
            let src: Vec<f64> = (0..total)
                .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
                .collect();
            let tensor = Value::Tensor(
                TensorValue::new_f64_values(
                    Shape {
                        dims: vec![nb as u32, hw as u32, hw as u32, c as u32],
                    },
                    src,
                )
                .unwrap(),
            );
            let p = rw_params(
                "max",
                &format!("1,{win},{win},1"),
                &format!("1,{stride},{stride},1"),
            );
            let run = || {
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&tensor), &p).unwrap()
            };
            let _ = run();
            let mut best = f64::MAX;
            for _ in 0..15 {
                let t = Instant::now();
                let r = run();
                best = best.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&r);
            }
            // f32 (JAX's default CNN dtype)
            let src32: Vec<f32> = (0..total)
                .map(|i| ((i % 9973) as f32) * 0.013 - 50.0)
                .collect();
            let tensor32 = Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: vec![nb as u32, hw as u32, hw as u32, c as u32],
                    },
                    src32,
                )
                .unwrap(),
            );
            let run32 = || {
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&tensor32), &p)
                    .unwrap()
            };
            let _ = run32();
            let mut best32 = f64::MAX;
            for _ in 0..15 {
                let t = Instant::now();
                let r = run32();
                best32 = best32.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&r);
            }
            println!(
                "BENCH maxpool-cnn N{nb} {hw}x{hw}x{c} {win}x{win}/s{stride}: fj-lax-f64={:.4}ms fj-lax-f32={:.4}ms",
                best * 1e3,
                best32 * 1e3
            );
        }
    }

    // BOLD-VERIFY: split the maxpool deque cost by axis to decide if a "transpose the strided
    // axis to contiguous" lever is worth it. Axis-0 reads down columns (stride = inner = 256,
    // L2-bound); axis-1 reads contiguous. Window [15,1] isolates the axis-0 pass, [1,15] the axis-1.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_maxpool_axis_split() {
        use std::time::Instant;
        let (n, win) = (256usize, 15usize);
        let out = n - win + 1;
        let src: Vec<f64> = (0..n * n)
            .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
            .collect();
        let timeit = |wd: [usize; 2], od: [usize; 2]| {
            let f = || {
                super::reduce_window_separable_maxmin(
                    &src,
                    &[n, n],
                    &wd,
                    &[1, 1],
                    &[0, 0],
                    &od,
                    true,
                )
            };
            let _ = f();
            let mut b = f64::MAX;
            for _ in 0..30 {
                let t = Instant::now();
                let r = f();
                b = b.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&r);
            }
            b * 1e3
        };
        let ax0 = timeit([win, 1], [out, n]); // strided axis-0 only
        let ax1 = timeit([1, win], [n, out]); // contiguous axis-1 only
        let both = timeit([win, win], [out, out]);
        println!(
            "BENCH maxpool axis-split 256x256/15: axis0-strided={ax0:.4}ms axis1-contig={ax1:.4}ms both={both:.4}ms  ax0/ax1={:.2}x",
            ax0 / ax1
        );
    }

    // BOLD-VERIFY (HYPOTHESIS DISPROVEN, kept as routing guard): I suspected the rank-2 f64
    // maxpool dispatch hit the DIRECT O(out·window) loop for 15x15. It does NOT — the separable
    // deque check (`win_total > 2·win_sum`, lib.rs ~5531) precedes the rank-2 direct path, and
    // 15x15 (225 > 60) routes to the O(n) deque already. This A/B confirms that routing matters:
    // for a large overlapping window the direct loop is ~56x SLOWER than the deque, so the
    // `win_total > 2·win_sum` gate MUST keep large windows on the deque (regression guard). The
    // real maxpool 2.3x JAX loss is the deque-vs-XLA-SIMD gap (XLA vectorizes the window reduction;
    // the safe-Rust deque is scalar) — a hard SIMD lever, NOT a dispatch bug. Bit-identity asserted.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_maxpool_rank2_direct_vs_deque() {
        use std::time::Instant;
        let (n, win) = (256usize, 15usize);
        let out = n - win + 1; // VALID, stride 1
        let src: Vec<f64> = (0..n * n)
            .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
            .collect();
        let tensor = TensorValue::new_f64_values(
            Shape {
                dims: vec![n as u32, n as u32],
            },
            src.clone(),
        )
        .unwrap();
        let total_output = out * out;
        let direct = || {
            super::eval_reduce_window_rank2_f64_max_min(
                Primitive::ReduceWindow,
                &tensor,
                &src,
                &[win, win],
                &[1, 1],
                &[out as u32, out as u32],
                &[0, 0],
                total_output,
                true,
            )
            .unwrap()
        };
        let deque = || {
            super::eval_reduce_window_separable_extremum(
                DType::F64,
                true,
                &src,
                &[win, win],
                &[1, 1],
                &[out as u32, out as u32],
                &[0, 0],
                &[n, n],
            )
            .unwrap()
        };
        // Production dispatch path (reduce_window_separable_maxmin, serial — threading it was a
        // measured no-ship). Verified bit-identical to the direct loop + timed as a guard.
        let prod = || {
            super::reduce_window_separable_maxmin(
                &src,
                &[n, n],
                &[win, win],
                &[1, 1],
                &[0, 0],
                &[out, out],
                true,
            )
        };
        let dv = match direct() {
            Value::Tensor(t) => t.elements.as_f64_slice().unwrap().to_vec(),
            _ => panic!(),
        };
        let qv = match deque() {
            Value::Tensor(t) => t.elements.as_f64_slice().unwrap().to_vec(),
            _ => panic!(),
        };
        assert_eq!(dv, qv, "separable deque maxpool diverges from direct loop");
        assert_eq!(
            dv,
            prod(),
            "production separable maxpool diverges from direct loop"
        );
        let _ = direct();
        let _ = deque();
        let _ = prod();
        let (mut db, mut qb, mut pb) = (f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..30 {
            let t = Instant::now();
            let a = direct();
            db = db.min(t.elapsed().as_secs_f64());
            std::hint::black_box(&a);
            let t = Instant::now();
            let b = deque();
            qb = qb.min(t.elapsed().as_secs_f64());
            std::hint::black_box(&b);
            let t = Instant::now();
            let c = prod();
            pb = pb.min(t.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        println!(
            "BENCH maxpool 256x256 15x15 VALID: direct={:.4}ms deque(4256)={:.4}ms prod-separable(5185)={:.4}ms  deque-vs-direct={:.2}x",
            db * 1e3,
            qb * 1e3,
            pb * 1e3,
            db / qb
        );
    }

    #[test]
    fn threaded_bitwise_bit_identical_to_serial() {
        // i64/u64 and/or/xor at >= gate engage the threaded dense map; compare bit-for-bit
        // to a serial reference.
        let n = crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN + 511;
        let shape = Shape {
            dims: vec![n as u32],
        };
        for prim in [
            Primitive::BitwiseAnd,
            Primitive::BitwiseOr,
            Primitive::BitwiseXor,
        ] {
            let a: Vec<i64> = (0..n).map(|i| i as i64 * 3 - 7).collect();
            let b: Vec<i64> = (0..n).map(|i| (i as i64) ^ 0x5555).collect();
            let mki = |v: &[i64]| {
                Value::Tensor(TensorValue::new_i64_values(shape.clone(), v.to_vec()).unwrap())
            };
            let got = eval_primitive(prim, &[mki(&a), mki(&b)], &no_params()).unwrap();
            let gi = got.as_tensor().unwrap().elements.as_i64_slice().unwrap();
            let want: Vec<i64> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| match prim {
                    Primitive::BitwiseAnd => x & y,
                    Primitive::BitwiseOr => x | y,
                    _ => x ^ y,
                })
                .collect();
            assert!(gi == want.as_slice(), "{prim:?} i64 threaded != serial");
            let au: Vec<u64> = (0..n).map(|i| i as u64 * 13 + 1).collect();
            let bu: Vec<u64> = (0..n).map(|i| (i as u64) ^ 0xF0F0).collect();
            let mku = |v: &[u64]| {
                Value::Tensor(TensorValue::new_u64_values(shape.clone(), v.to_vec()).unwrap())
            };
            let gotu = eval_primitive(prim, &[mku(&au), mku(&bu)], &no_params()).unwrap();
            let gu = gotu.as_tensor().unwrap().elements.as_u64_slice().unwrap();
            let wantu: Vec<u64> = au
                .iter()
                .zip(&bu)
                .map(|(&x, &y)| match prim {
                    Primitive::BitwiseAnd => x & y,
                    Primitive::BitwiseOr => x | y,
                    _ => x ^ y,
                })
                .collect();
            assert!(gu == wantu.as_slice(), "{prim:?} u64 threaded != serial");
        }
    }

    fn no_params() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    #[test]
    fn i32_arithmetic_wraps_mod_2pow32_like_jax() {
        // jnp/numpy int32 ops wrap mod 2^32. fj-lax stores i32 as Literal::I64 and the
        // shared i64 wrapping ops do NOT narrow, so without the eval_primitive narrowing
        // chokepoint an I32 result could exceed i32 range. Verify add/sub/mul/matmul/
        // reduce all land in i32 range and equal the two's-complement reference.
        let i32t = |dims: Vec<u32>, vals: &[i64]| -> Value {
            Value::Tensor(
                TensorValue::new(
                    DType::I32,
                    Shape { dims },
                    vals.iter().map(|&v| Literal::I64(v)).collect(),
                )
                .unwrap(),
            )
        };
        let out_i64s = |v: &Value| -> Vec<i64> {
            let Value::Tensor(t) = v else {
                panic!("tensor")
            };
            assert_eq!(t.dtype, DType::I32, "result must stay I32");
            t.elements
                .iter()
                .map(|l| match l {
                    Literal::I64(x) => *x,
                    o => panic!("unexpected {o:?}"),
                })
                .collect()
        };
        let max = i64::from(i32::MAX); // 2147483647

        // Add overflow: MAX + 1 -> i32::MIN.
        let r = eval_primitive(
            Primitive::Add,
            &[i32t(vec![1], &[max]), i32t(vec![1], &[1])],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out_i64s(&r), vec![i64::from(i32::MIN)]);

        // Mul overflow: 100000 * 100000 = 1e10 -> wraps to (1e10 as i32).
        let r = eval_primitive(
            Primitive::Mul,
            &[i32t(vec![1], &[100_000]), i32t(vec![1], &[100_000])],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out_i64s(&r), vec![i64::from(10_000_000_000_i64 as i32)]);

        // Sub underflow: i32::MIN - 1 -> i32::MAX.
        let r = eval_primitive(
            Primitive::Sub,
            &[i32t(vec![1], &[i64::from(i32::MIN)]), i32t(vec![1], &[1])],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out_i64s(&r), vec![max]);

        // In-range results are untouched.
        let r = eval_primitive(
            Primitive::Add,
            &[i32t(vec![2], &[3, -4]), i32t(vec![2], &[5, 6])],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out_i64s(&r), vec![8, 2]);

        // matmul [1,2]@[2,1] = 100000*100000 + 100000*100000 = 2e10 -> wraps to i32.
        let mut dot_p = BTreeMap::new();
        dot_p.insert("lhs_contracting_dims".to_owned(), "1".to_owned());
        dot_p.insert("rhs_contracting_dims".to_owned(), "0".to_owned());
        let r = eval_primitive(
            Primitive::DotGeneral,
            &[
                i32t(vec![1, 2], &[100_000, 100_000]),
                i32t(vec![2, 1], &[100_000, 100_000]),
            ],
            &dot_p,
        )
        .unwrap();
        assert_eq!(out_i64s(&r), vec![i64::from(20_000_000_000_i64 as i32)]);
        // NOTE: integer REDUCE over i32 currently widens the result dtype to I64 (a
        // separate dtype-promotion divergence from JAX, which keeps int32) — so it is NOT
        // covered by this I32-output narrowing chokepoint and is tracked in its own bead.
    }

    fn pad_params(low: &str, high: &str, interior: &str) -> BTreeMap<String, String> {
        let mut params = BTreeMap::new();
        params.insert("padding_low".to_owned(), low.to_owned());
        params.insert("padding_high".to_owned(), high.to_owned());
        params.insert("padding_interior".to_owned(), interior.to_owned());
        params
    }

    fn square_2x2(data: [f64; 4]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn eval_primitive_multi_slogdet_returns_both_outputs() {
        // Previously eval_primitive_multi fell through to the single-output
        // path and silently dropped logabsdet, returning only [sign].
        let a = square_2x2([2.0, 0.0, 0.0, 3.0]); // det = 6
        let outputs = super::eval_primitive_multi(Primitive::Slogdet, &[a], &no_params()).unwrap();
        assert_eq!(outputs.len(), 2, "slogdet must yield (sign, logabsdet)");
        let sign = outputs[0].as_f64_scalar().unwrap();
        let logabsdet = outputs[1].as_f64_scalar().unwrap();
        assert!((sign - 1.0).abs() < 1e-12, "sign={sign}");
        assert!(
            (logabsdet - 6.0_f64.ln()).abs() < 1e-9,
            "logabsdet={logabsdet}"
        );
    }

    #[test]
    fn eval_primitive_multi_eig_returns_both_outputs() {
        // Eig must yield (eigenvalues, eigenvectors), not just eigenvalues.
        let a = square_2x2([2.0, 0.0, 0.0, 3.0]);
        let outputs = super::eval_primitive_multi(Primitive::Eig, &[a], &no_params()).unwrap();
        assert_eq!(
            outputs.len(),
            2,
            "eig must yield (eigenvalues, eigenvectors)"
        );
        // eigenvalues: length-2 vector; eigenvectors: 2x2 matrix.
        assert_eq!(
            outputs[0].as_tensor().unwrap().shape.dims,
            vec![2],
            "eigenvalues shape"
        );
        assert_eq!(
            outputs[1].as_tensor().unwrap().shape.dims,
            vec![2, 2],
            "eigenvectors shape"
        );
    }

    #[test]
    fn test_cholesky_rejects_scalar_input() {
        let err = eval_primitive(Primitive::Cholesky, &[Value::scalar_f64(1.0)], &no_params())
            .expect_err("cholesky should reject scalar input");
        if let EvalError::Unsupported { detail, .. } = &err {
            assert!(detail.contains("scalar"), "detail: {detail}");
        } else {
            assert!(
                matches!(err, EvalError::Unsupported { .. }),
                "expected unsupported error, got {err:?}"
            );
        }
    }

    #[test]
    fn test_svd_rejects_scalar_input() {
        let result = eval_primitive(Primitive::Svd, &[Value::scalar_f64(1.0)], &no_params());
        assert!(result.is_err(), "Svd should reject scalar input");
    }

    #[test]
    fn test_eigh_rejects_scalar_input() {
        let result = eval_primitive(Primitive::Eigh, &[Value::scalar_f64(1.0)], &no_params());
        assert!(result.is_err(), "Eigh should reject scalar input");
    }

    #[test]
    fn test_fft_rejects_scalar_input() {
        let result = eval_primitive(Primitive::Fft, &[Value::scalar_f64(1.0)], &no_params());
        assert!(result.is_err(), "Fft should reject scalar input");
    }

    #[test]
    fn test_ifft_rejects_scalar_input() {
        let result = eval_primitive(Primitive::Ifft, &[Value::scalar_f64(1.0)], &no_params());
        assert!(result.is_err(), "Ifft should reject scalar input");
    }

    #[test]
    fn test_rfft_rejects_scalar_input() {
        let result = eval_primitive(Primitive::Rfft, &[Value::scalar_f64(1.0)], &no_params());
        assert!(result.is_err(), "Rfft should reject scalar input");
    }

    #[test]
    fn test_irfft_rejects_scalar_input() {
        let result = eval_primitive(Primitive::Irfft, &[Value::scalar_f64(1.0)], &no_params());
        assert!(result.is_err(), "Irfft should reject scalar input");
    }

    #[test]
    fn add_i64_scalars() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(2), Value::scalar_i64(5)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn add_i64_overflow_wraps() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(i64::MAX), Value::scalar_i64(1)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_i64(i64::MIN));
    }

    #[test]
    fn add_vector_and_scalar_broadcasts() {
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector value should build");
        let out = eval_primitive(Primitive::Add, &[input, Value::scalar_i64(2)], &no_params())
            .expect("broadcasted add should succeed");

        let expected = Value::vector_i64(&[3, 4, 5]).expect("vector value should build");
        assert_eq!(out, expected);
    }

    #[test]
    fn sub_i64_scalars() {
        let out = eval_primitive(
            Primitive::Sub,
            &[Value::scalar_i64(10), Value::scalar_i64(3)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn sub_i64_overflow_wraps() {
        let out = eval_primitive(
            Primitive::Sub,
            &[Value::scalar_i64(i64::MIN), Value::scalar_i64(1)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_i64(i64::MAX));
    }

    #[test]
    fn sub_f64_scalars() {
        let out = eval_primitive(
            Primitive::Sub,
            &[Value::scalar_f64(5.5), Value::scalar_f64(2.0)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.5).abs() < 1e-10);
    }

    #[test]
    fn neg_i64_scalar() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_i64(7)], &no_params());
        assert_eq!(out, Ok(Value::scalar_i64(-7)));
    }

    #[test]
    fn neg_i64_min_wraps() {
        let out =
            eval_primitive(Primitive::Neg, &[Value::scalar_i64(i64::MIN)], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(i64::MIN));
    }

    #[test]
    fn neg_f64_scalar() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_f64(3.5)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - (-3.5)).abs() < 1e-10);
    }

    #[test]
    fn neg_u32_scalar_wraps() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_u32(5)], &no_params());
        assert_eq!(out, Ok(Value::scalar_u32(5u32.wrapping_neg())));
    }

    #[test]
    fn neg_u64_scalar_wraps() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_u64(1)], &no_params());
        assert_eq!(out, Ok(Value::scalar_u64(1u64.wrapping_neg())));
    }

    #[test]
    fn abs_negative_i64() {
        let out = eval_primitive(Primitive::Abs, &[Value::scalar_i64(-42)], &no_params());
        assert_eq!(out, Ok(Value::scalar_i64(42)));
    }

    #[test]
    fn abs_i64_min_wraps() {
        let out =
            eval_primitive(Primitive::Abs, &[Value::scalar_i64(i64::MIN)], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(i64::MIN));
    }

    #[test]
    fn abs_negative_f64() {
        let out =
            eval_primitive(Primitive::Abs, &[Value::scalar_f64(-2.78)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 2.78).abs() < 1e-10);
    }

    #[test]
    fn abs_unsigned_scalars_are_identity() {
        let u32_out = eval_primitive(Primitive::Abs, &[Value::scalar_u32(42)], &no_params());
        let u64_out = eval_primitive(Primitive::Abs, &[Value::scalar_u64(99)], &no_params());
        assert_eq!(u32_out, Ok(Value::scalar_u32(42)));
        assert_eq!(u64_out, Ok(Value::scalar_u64(99)));
    }

    #[test]
    fn sign_unsigned_scalars_match_zero_or_one() {
        let zero_u32 = eval_primitive(Primitive::Sign, &[Value::scalar_u32(0)], &no_params());
        let pos_u32 = eval_primitive(Primitive::Sign, &[Value::scalar_u32(7)], &no_params());
        let zero_u64 = eval_primitive(Primitive::Sign, &[Value::scalar_u64(0)], &no_params());
        let pos_u64 = eval_primitive(Primitive::Sign, &[Value::scalar_u64(11)], &no_params());

        assert_eq!(zero_u32, Ok(Value::scalar_u32(0)));
        assert_eq!(pos_u32, Ok(Value::scalar_u32(1)));
        assert_eq!(zero_u64, Ok(Value::scalar_u64(0)));
        assert_eq!(pos_u64, Ok(Value::scalar_u64(1)));
    }

    #[test]
    fn unsigned_unary_tensor_ops_preserve_dtype() {
        let neg_input = Value::Tensor(
            TensorValue::new(
                DType::U32,
                Shape::vector(2),
                vec![Literal::U32(0), Literal::U32(5)],
            )
            .unwrap(),
        );
        let sign_input = Value::Tensor(
            TensorValue::new(
                DType::U64,
                Shape::vector(3),
                vec![Literal::U64(0), Literal::U64(2), Literal::U64(9)],
            )
            .unwrap(),
        );

        let neg_out = eval_primitive(Primitive::Neg, &[neg_input], &no_params()).unwrap();
        let sign_out = eval_primitive(Primitive::Sign, &[sign_input], &no_params()).unwrap();

        assert_eq!(
            neg_out,
            Value::Tensor(
                TensorValue::new(
                    DType::U32,
                    Shape::vector(2),
                    vec![Literal::U32(0), Literal::U32(5u32.wrapping_neg())],
                )
                .unwrap(),
            )
        );
        assert_eq!(
            sign_out,
            Value::Tensor(
                TensorValue::new(
                    DType::U64,
                    Shape::vector(3),
                    vec![Literal::U64(0), Literal::U64(1), Literal::U64(1)],
                )
                .unwrap(),
            )
        );
    }

    #[test]
    fn max_i64_scalars() {
        let out = eval_primitive(
            Primitive::Max,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(7)));
    }

    #[test]
    fn min_i64_scalars() {
        let out = eval_primitive(
            Primitive::Min,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_i64(3)));
    }

    #[test]
    fn exp_scalar() {
        let out = eval_primitive(Primitive::Exp, &[Value::scalar_f64(1.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn log_scalar() {
        let out = eval_primitive(
            Primitive::Log,
            &[Value::scalar_f64(std::f64::consts::E)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sqrt_scalar() {
        let out = eval_primitive(Primitive::Sqrt, &[Value::scalar_f64(9.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn rsqrt_scalar() {
        let out =
            eval_primitive(Primitive::Rsqrt, &[Value::scalar_f64(4.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn floor_scalar() {
        let out =
            eval_primitive(Primitive::Floor, &[Value::scalar_f64(3.7)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn ceil_scalar() {
        let out = eval_primitive(Primitive::Ceil, &[Value::scalar_f64(3.2)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 4.0).abs() < 1e-10);
    }

    #[test]
    fn round_scalar() {
        let out =
            eval_primitive(Primitive::Round, &[Value::scalar_f64(3.5)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 4.0).abs() < 1e-10);
    }

    #[test]
    fn round_to_nearest_even_scalar() {
        let mut params = BTreeMap::new();
        params.insert("rounding_method".to_owned(), "TO_NEAREST_EVEN".to_owned());
        let out = eval_primitive(Primitive::Round, &[Value::scalar_f64(2.5)], &params).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 2.0).abs() < 1e-10);
    }

    #[test]
    fn round_to_nearest_even_rejects_unknown_method() {
        let mut params = BTreeMap::new();
        params.insert("rounding_method".to_owned(), "HALF_UP".to_owned());
        let err = eval_primitive(Primitive::Round, &[Value::scalar_f64(2.5)], &params)
            .expect_err("unknown rounding methods should fail closed");
        assert!(
            err.to_string()
                .contains("unsupported rounding_method 'HALF_UP'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn pow_f64_scalars() {
        let out = eval_primitive(
            Primitive::Pow,
            &[Value::scalar_f64(2.0), Value::scalar_f64(3.0)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 8.0).abs() < 1e-10);
    }

    #[test]
    fn eq_i64_scalars() {
        let p = no_params();
        let out = eval_primitive(
            Primitive::Eq,
            &[Value::scalar_i64(3), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(
            Primitive::Eq,
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn ne_i64_scalars() {
        let out = eval_primitive(
            Primitive::Ne,
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn lt_i64_scalars() {
        let p = no_params();
        let out = eval_primitive(
            Primitive::Lt,
            &[Value::scalar_i64(3), Value::scalar_i64(5)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(
            Primitive::Lt,
            &[Value::scalar_i64(5), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn le_ge_i64_scalars() {
        let p = no_params();
        let out = eval_primitive(
            Primitive::Le,
            &[Value::scalar_i64(3), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));

        let out = eval_primitive(
            Primitive::Ge,
            &[Value::scalar_i64(3), Value::scalar_i64(3)],
            &p,
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn gt_f64_scalars() {
        let out = eval_primitive(
            Primitive::Gt,
            &[Value::scalar_f64(3.5), Value::scalar_f64(2.0)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn comparison_on_vectors() {
        let lhs = Value::vector_i64(&[1, 2, 3]).unwrap();
        let rhs = Value::vector_i64(&[2, 2, 1]).unwrap();
        let out = eval_primitive(Primitive::Lt, &[lhs, rhs], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements.len(), 3);
            assert_eq!(t.elements[0], fj_core::Literal::Bool(true));
            assert_eq!(t.elements[1], fj_core::Literal::Bool(false));
            assert_eq!(t.elements[2], fj_core::Literal::Bool(false));
        } else {
            assert!(
                matches!(out, Value::Tensor(_)),
                "expected tensor output for vector comparison"
            );
        }
    }

    #[test]
    fn reduce_max_vector() {
        let input = Value::vector_i64(&[3, 7, 2, 9, 1]).unwrap();
        let out = eval_primitive(Primitive::ReduceMax, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(9));
    }

    #[test]
    fn reduce_min_vector() {
        let input = Value::vector_i64(&[3, 7, 2, 9, 1]).unwrap();
        let out = eval_primitive(Primitive::ReduceMin, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(1));
    }

    #[test]
    fn reduce_prod_vector() {
        let input = Value::vector_i64(&[2, 3, 4]).unwrap();
        let out = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(24));
    }

    #[test]
    fn neg_vector() {
        let input = Value::vector_i64(&[1, -2, 3]).unwrap();
        let out = eval_primitive(Primitive::Neg, &[input], &no_params()).unwrap();
        let expected = Value::vector_i64(&[-1, 2, -3]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dot_vector_works() {
        let lhs = Value::vector_i64(&[1, 2, 3]).expect("vector value should build");
        let rhs = Value::vector_i64(&[4, 5, 6]).expect("vector value should build");
        let out =
            eval_primitive(Primitive::Dot, &[lhs, rhs], &no_params()).expect("dot should succeed");
        assert_eq!(out, Value::scalar_i64(32));
    }

    #[test]
    fn reduce_sum_requires_single_argument() {
        let err = eval_primitive(Primitive::ReduceSum, &[], &no_params()).expect_err("should fail");
        assert_eq!(
            err,
            EvalError::ArityMismatch {
                primitive: Primitive::ReduceSum,
                expected: 1,
                actual: 0,
            }
        );
    }

    // --- Shape manipulation tests ---

    #[test]
    fn reshape_vector_to_matrix() {
        let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("new_shape".into(), "2,3".into());
        let out = eval_primitive(Primitive::Reshape, &[input], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
            assert_eq!(t.elements.len(), 6);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reshape_with_inferred_dim() {
        let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("new_shape".into(), "2,-1".into());
        let out = eval_primitive(Primitive::Reshape, &[input], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn transpose_2d() {
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let input = fj_core::TensorValue::new(
            DType::I64,
            fj_core::Shape { dims: vec![2, 3] },
            vec![
                fj_core::Literal::I64(1),
                fj_core::Literal::I64(2),
                fj_core::Literal::I64(3),
                fj_core::Literal::I64(4),
                fj_core::Literal::I64(5),
                fj_core::Literal::I64(6),
            ],
        )
        .unwrap();
        let out =
            eval_primitive(Primitive::Transpose, &[Value::Tensor(input)], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3, 2]);
            assert_eq!(t.elements[0], fj_core::Literal::I64(1));
            assert_eq!(t.elements[1], fj_core::Literal::I64(4));
            assert_eq!(t.elements[2], fj_core::Literal::I64(2));
            assert_eq!(t.elements[3], fj_core::Literal::I64(5));
            assert_eq!(t.elements[4], fj_core::Literal::I64(3));
            assert_eq!(t.elements[5], fj_core::Literal::I64(6));
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn transpose_empty_huge_shape_returns_empty() {
        let huge = u32::MAX;
        let input = fj_core::TensorValue::new(
            DType::I64,
            fj_core::Shape {
                dims: vec![0, huge, huge, huge],
            },
            Vec::new(),
        )
        .unwrap();
        let mut params = BTreeMap::new();
        params.insert("permutation".into(), "1,2,3,0".into());

        let out = eval_primitive(Primitive::Transpose, &[Value::Tensor(input)], &params).unwrap();

        let tensor = out.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![huge, huge, huge, 0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn broadcast_in_dim_scalar_to_vector() {
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "3".into());
        let out =
            eval_primitive(Primitive::BroadcastInDim, &[Value::scalar_i64(5)], &params).unwrap();
        let expected = Value::vector_i64(&[5, 5, 5]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn broadcast_in_dim_vector_to_matrix() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "2,3".into());
        params.insert("broadcast_dimensions".into(), "1".into());
        let out = eval_primitive(Primitive::BroadcastInDim, &[input], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
            // Row 0: [1,2,3], Row 1: [1,2,3]
            assert_eq!(t.elements[0], fj_core::Literal::I64(1));
            assert_eq!(t.elements[1], fj_core::Literal::I64(2));
            assert_eq!(t.elements[2], fj_core::Literal::I64(3));
            assert_eq!(t.elements[3], fj_core::Literal::I64(1));
            assert_eq!(t.elements[4], fj_core::Literal::I64(2));
            assert_eq!(t.elements[5], fj_core::Literal::I64(3));
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn broadcast_in_dim_dense_matches_literal_path() {
        // The dense (F64/I64) broadcast fast path and the Literal-backed generic
        // path must produce identical elements (same replication index math).
        use fj_core::{DType, Shape};
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "4,3,5".into()); // broadcast [3] over axes 0,2
        params.insert("broadcast_dimensions".into(), "1".into());

        // f64: dense vs Literal-backed.
        let fdata = [1.5_f64, -2.0, 3.25];
        let f_dense = Value::vector_f64(&fdata).unwrap();
        let f_lit = Value::Tensor(
            crate::new_boxed(
                DType::F64,
                Shape::vector(3),
                fdata
                    .iter()
                    .copied()
                    .map(fj_core::Literal::from_f64)
                    .collect(),
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
        let dense_out = eval_primitive(
            Primitive::BroadcastInDim,
            std::slice::from_ref(&f_dense),
            &params,
        )
        .unwrap();
        let lit_out = eval_primitive(
            Primitive::BroadcastInDim,
            std::slice::from_ref(&f_lit),
            &params,
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
        assert_eq!(dense_out.as_tensor().unwrap().shape.dims, vec![4, 3, 5]);
        assert_eq!(
            bits(&dense_out),
            bits(&lit_out),
            "f64 broadcast dense vs generic"
        );

        // i64: dense vs Literal-backed.
        let idata = [7_i64, -8, 9];
        let i_dense = Value::vector_i64(&idata).unwrap();
        let i_lit = Value::Tensor(
            crate::new_boxed(
                DType::I64,
                Shape::vector(3),
                idata.iter().copied().map(fj_core::Literal::I64).collect(),
            )
            .unwrap(),
        );
        let i_dense_out = eval_primitive(
            Primitive::BroadcastInDim,
            std::slice::from_ref(&i_dense),
            &params,
        )
        .unwrap();
        let i_lit_out = eval_primitive(
            Primitive::BroadcastInDim,
            std::slice::from_ref(&i_lit),
            &params,
        )
        .unwrap();
        let ivals = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect()
        };
        assert_eq!(
            ivals(&i_dense_out),
            ivals(&i_lit_out),
            "i64 broadcast dense vs generic"
        );
    }

    #[test]
    fn broadcasted_iota_dense_matches_reference() {
        // The dense I64/F64 BroadcastedIota fast path must produce dense storage
        // whose values equal an independent reference computed with the same
        // `(flat / stride) % axis_extent` index formula. Exercise a rank-3 shape
        // with a non-leading iota dimension.
        let dims = [3_usize, 4, 5];
        let dimension = 1_usize; // axis carrying the iota
        let total: usize = dims.iter().product();
        let stride: usize = dims[(dimension + 1)..].iter().product();
        let axis_extent = dims[dimension];
        let expected: Vec<i64> = (0..total)
            .map(|flat| ((flat / stride) % axis_extent) as i64)
            .collect();

        let mut params = BTreeMap::new();
        params.insert("shape".into(), "3,4,5".to_string());
        params.insert("dimension".into(), "1".to_string());

        // I64 (default dtype): dense storage + value-equal to reference.
        params.insert("dtype".into(), "i64".to_string());
        let i_out = eval_primitive(Primitive::BroadcastedIota, &[], &params).unwrap();
        let i_tensor = i_out.as_tensor().unwrap();
        assert_eq!(i_tensor.shape.dims, vec![3, 4, 5]);
        assert!(
            i_tensor.elements.as_i64_slice().is_some(),
            "i64 iota should be dense"
        );
        assert_eq!(
            i_tensor.elements.as_i64_slice().unwrap(),
            expected.as_slice()
        );

        // F64: dense storage + bit-equal to `index as f64`.
        params.insert("dtype".into(), "f64".to_string());
        let f_out = eval_primitive(Primitive::BroadcastedIota, &[], &params).unwrap();
        let f_tensor = f_out.as_tensor().unwrap();
        assert!(
            f_tensor.elements.as_f64_slice().is_some(),
            "f64 iota should be dense"
        );
        let f_bits: Vec<u64> = f_tensor
            .elements
            .as_f64_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let exp_bits: Vec<u64> = expected.iter().map(|&v| (v as f64).to_bits()).collect();
        assert_eq!(f_bits, exp_bits, "f64 iota dense vs reference");
    }

    #[test]
    fn broadcast_in_dim_empty_huge_tensor_returns_empty_tensor() {
        let huge = u32::MAX;
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![0, huge, huge, huge],
                },
                Vec::new(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("shape".into(), format!("0,{huge},{huge},{huge}"));
        params.insert("broadcast_dimensions".into(), "0,1,2,3".into());

        let out = eval_primitive(Primitive::BroadcastInDim, &[input], &params).unwrap();
        let tensor = out.as_tensor().unwrap();

        assert_eq!(tensor.shape.dims, vec![0, huge, huge, huge]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn broadcast_in_dim_rejects_duplicate_axes() {
        let input = Value::vector_i64(&[1, 2]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "2,2".into());
        params.insert("broadcast_dimensions".into(), "1,1".into());
        let result = eval_primitive(Primitive::BroadcastInDim, &[input], &params);
        assert!(result.is_err());
    }

    #[test]
    fn broadcast_in_dim_rejects_out_of_range_axis() {
        let input = Value::vector_i64(&[1, 2]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "2,2".into());
        params.insert("broadcast_dimensions".into(), "2".into());
        let result = eval_primitive(Primitive::BroadcastInDim, &[input], &params);
        assert!(result.is_err());
    }

    #[test]
    fn broadcast_in_dim_rejects_non_strictly_increasing_dims() {
        // [1,1] -> [2,2] with broadcast_dimensions=[1,0]: unique, in range, and size
        // compatible (both input dims are 1), but NOT strictly increasing. JAX
        // (lax._broadcast_in_dim_shape_rule) requires strictly increasing, so this must
        // fail closed (frankenjax-gnie5).
        let input = Value::Tensor(
            fj_core::TensorValue::new(
                DType::I64,
                fj_core::Shape { dims: vec![1, 1] },
                vec![fj_core::Literal::I64(5)],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "2,2".into());
        params.insert("broadcast_dimensions".into(), "1,0".into());
        let result = eval_primitive(Primitive::BroadcastInDim, &[input], &params);
        assert!(
            result.is_err(),
            "non-strictly-increasing broadcast_dimensions must be rejected"
        );
    }

    #[test]
    fn squeeze_rejects_duplicate_dimensions() {
        // [1,1] squeeze dimensions=[0,0]: both axes are size 1 and in range, but the
        // dimension list is duplicated. JAX (lax._compute_squeeze_shape) rejects duplicate
        // dimensions, so this must fail closed (frankenjax-wsuwr).
        let input = Value::Tensor(
            fj_core::TensorValue::new(
                DType::I64,
                fj_core::Shape { dims: vec![1, 1] },
                vec![fj_core::Literal::I64(5)],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("dimensions".into(), "0,0".into());
        let result = eval_primitive(Primitive::Squeeze, &[input], &params);
        assert!(
            result.is_err(),
            "duplicate squeeze dimensions must be rejected"
        );
    }

    #[test]
    fn rev_rejects_duplicate_axes() {
        // rev axes=[0,0]: axis 0 is in range but duplicated. JAX (lax._rev_shape_rule)
        // requires rev dimensions unique, so this must fail closed (frankenjax-wsuwr
        // sibling). Without the check the contains()-mask would silently reverse once.
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0,0".into());
        let result = eval_primitive(Primitive::Rev, &[input], &params);
        assert!(result.is_err(), "duplicate rev axes must be rejected");
    }

    #[test]
    fn broadcast_in_dim_rejects_incompatible_dim() {
        let input = Value::vector_i64(&[1, 2]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "3,2".into());
        params.insert("broadcast_dimensions".into(), "0".into());
        let result = eval_primitive(Primitive::BroadcastInDim, &[input], &params);
        assert!(result.is_err());
    }

    #[test]
    fn broadcast_in_dim_default_mapping_rejects_incompatible_dim() {
        let input = Value::vector_i64(&[1, 2]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "3".into());
        let result = eval_primitive(Primitive::BroadcastInDim, &[input], &params);
        assert!(result.is_err());
    }

    #[test]
    fn broadcast_in_dim_rejects_dim_above_u32_range() {
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "4294967296".into());
        let result = eval_primitive(Primitive::BroadcastInDim, &[Value::scalar_i64(1)], &params);

        let err = result.unwrap_err().to_string();
        assert!(err.contains("exceeds u32 range"), "unexpected error: {err}");
    }

    #[test]
    fn broadcast_in_dim_rejects_shape_product_overflow() {
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "4294967295,4294967295,4294967295".into());
        let result = eval_primitive(Primitive::BroadcastInDim, &[Value::scalar_i64(1)], &params);

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("shape overflows usize"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn concatenate_vectors() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[3, 4, 5]).unwrap();
        let out = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
        let expected = Value::vector_i64(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn concatenate_rejects_output_axis_overflow() {
        let a = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![u32::MAX, 0],
                },
                vec![],
            )
            .unwrap(),
        );
        let b = Value::Tensor(
            TensorValue::new(DType::I64, Shape { dims: vec![1, 0] }, vec![]).unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("dimension".to_owned(), "0".to_owned());
        let result = eval_primitive(Primitive::Concatenate, &[a, b], &params);

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("axis size overflows u32"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn pad_vector_with_edge_padding() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let params = pad_params("1", "2", "0");
        let out = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params).unwrap();
        let expected = Value::vector_i64(&[0, 1, 2, 3, 0, 0]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn pad_vector_with_interior_and_edge_padding() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let params = pad_params("1", "1", "1");
        let out = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params).unwrap();
        let expected = Value::vector_i64(&[0, 1, 0, 2, 0, 3, 0]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn pad_rejects_output_dimension_arithmetic_overflow() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![u32::MAX, 0],
                },
                vec![],
            )
            .unwrap(),
        );
        let params = pad_params("0,0", "0,0", &format!("{},0", i64::MAX));
        let result = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params);

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("padded dimension overflow on axis 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn pad_rejects_output_shape_product_overflow() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![0, 0, 0],
                },
                vec![],
            )
            .unwrap(),
        );
        let params = pad_params("4294967295,4294967295,4294967295", "0,0,0", "0,0,0");
        let result = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params);

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("shape overflows usize"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn pad_empty_huge_output_returns_empty_tensor() {
        let huge = u32::MAX;
        let low_huge = huge - 1;
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![1, 1, 1, 1],
                },
                vec![Literal::I64(1)],
            )
            .unwrap(),
        );
        let params = pad_params(
            &format!("-1,{low_huge},{low_huge},{low_huge}"),
            "0,0,0,0",
            "0,0,0,0",
        );

        let out = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params).unwrap();
        let tensor = out.as_tensor().unwrap();

        assert_eq!(tensor.shape.dims, vec![0, huge, huge, huge]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn pad_rank2_tensor_preserves_layout() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let params = pad_params("1,0", "0,1", "0,1");
        let out = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params).unwrap();
        let out_tensor = out.as_tensor().expect("pad output should be tensor");
        assert_eq!(out_tensor.shape.dims, vec![3, 4]);
        assert_eq!(
            out_tensor.elements,
            vec![
                Literal::I64(0),
                Literal::I64(0),
                Literal::I64(0),
                Literal::I64(0),
                Literal::I64(1),
                Literal::I64(0),
                Literal::I64(2),
                Literal::I64(0),
                Literal::I64(3),
                Literal::I64(0),
                Literal::I64(4),
                Literal::I64(0),
            ]
        );
    }

    #[test]
    fn pad_dense_matches_literal_path() {
        // The dense (F64/I64) pad fast path must equal the generic Literal path
        // element-by-element under edge + interior + negative (cropping) padding.
        let shape = Shape { dims: vec![3, 4] };
        let params = pad_params("1,-1", "2,1", "1,0"); // low/high incl crop + interior

        // f64: dense (new_f64_values) vs Literal-backed.
        let fdata: Vec<f64> = (0..12).map(|i| (i as f64) * 1.5 - 2.0).collect();
        let f_dense =
            Value::Tensor(TensorValue::new_f64_values(shape.clone(), fdata.clone()).unwrap());
        let f_lit = Value::Tensor(
            crate::new_boxed(
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
        let pv = Value::scalar_f64(9.0);
        let d = eval_primitive(Primitive::Pad, &[f_dense, pv.clone()], &params).unwrap();
        let l = eval_primitive(Primitive::Pad, &[f_lit, pv], &params).unwrap();
        let fbits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|x| x.as_f64().unwrap().to_bits())
                .collect()
        };
        assert_eq!(
            d.as_tensor().unwrap().shape.dims,
            l.as_tensor().unwrap().shape.dims
        );
        assert_eq!(fbits(&d), fbits(&l), "f64 pad dense vs generic");

        // i64: dense vs Literal-backed.
        let idata: Vec<i64> = (0..12).map(|i| i as i64 * 7 - 5).collect();
        let i_dense =
            Value::Tensor(TensorValue::new_i64_values(shape.clone(), idata.clone()).unwrap());
        let i_lit = Value::Tensor(
            crate::new_boxed(
                DType::I64,
                shape.clone(),
                idata.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let di = eval_primitive(Primitive::Pad, &[i_dense, Value::scalar_i64(0)], &params).unwrap();
        let li = eval_primitive(Primitive::Pad, &[i_lit, Value::scalar_i64(0)], &params).unwrap();
        let ivals = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect()
        };
        assert_eq!(ivals(&di), ivals(&li), "i64 pad dense vs generic");

        // No-interior, no-crop config exercises the dense ROW-COPY path; it must
        // also match the generic per-element placement.
        let nc = pad_params("1,2", "2,1", "0,0");
        let f_dense2 =
            Value::Tensor(TensorValue::new_f64_values(shape.clone(), fdata.clone()).unwrap());
        let f_lit2 = Value::Tensor(
            crate::new_boxed(
                DType::F64,
                shape.clone(),
                fdata.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let dnc = eval_primitive(Primitive::Pad, &[f_dense2, Value::scalar_f64(9.0)], &nc).unwrap();
        let lnc = eval_primitive(Primitive::Pad, &[f_lit2, Value::scalar_f64(9.0)], &nc).unwrap();
        assert_eq!(
            dnc.as_tensor().unwrap().shape.dims,
            lnc.as_tensor().unwrap().shape.dims
        );
        assert_eq!(fbits(&dnc), fbits(&lnc), "f64 pad row-copy vs generic");
    }

    #[test]
    fn pad_scalar_with_empty_padding_config_passes_through() {
        let out = eval_primitive(
            Primitive::Pad,
            &[Value::scalar_f64(3.5), Value::scalar_f64(0.0)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_f64(3.5));
    }

    #[test]
    fn pad_rank_zero_tensor_accepts_empty_padding_config() {
        let input = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(7)]).unwrap(),
        );
        let params = pad_params("", "", "");
        let out = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params).unwrap();
        let out_tensor = out.as_tensor().expect("pad output should be tensor");
        assert_eq!(out_tensor.shape, Shape::scalar());
        assert_eq!(out_tensor.elements, vec![Literal::I64(7)]);
    }

    #[test]
    fn pad_scalar_rejects_nonempty_padding_config() {
        let params = pad_params("1", "0", "0");
        let err = eval_primitive(
            Primitive::Pad,
            &[Value::scalar_i64(7), Value::scalar_i64(0)],
            &params,
        )
        .unwrap_err();
        let detail = match err {
            EvalError::Unsupported { detail, .. } => detail,
            other => format!("expected unsupported error, got {other:?}"),
        };
        assert!(detail.contains("rank 0"), "detail: {detail}");
    }

    #[test]
    fn pad_rejects_dtype_mismatch() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let params = pad_params("1", "0", "0");
        let err =
            eval_primitive(Primitive::Pad, &[input, Value::scalar_f64(0.0)], &params).unwrap_err();
        let detail = match err {
            EvalError::Unsupported { detail, .. } => detail,
            other => format!("expected unsupported error, got {other:?}"),
        };
        assert!(detail.contains("dtype"), "detail: {detail}");
    }

    #[test]
    fn pad_negative_edge_padding_crops_operand() {
        let input = Value::vector_i64(&[1, 2, 3, 4, 5]).unwrap();
        let params = pad_params("-1", "-2", "0");
        let out = eval_primitive(Primitive::Pad, &[input, Value::scalar_i64(0)], &params).unwrap();
        let expected = Value::vector_i64(&[2, 3]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn slice_vector() {
        let input = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "1".into());
        params.insert("limit_indices".into(), "4".into());
        let out = eval_primitive(Primitive::Slice, &[input], &params).unwrap();
        let expected = Value::vector_i64(&[20, 30, 40]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn slice_full_trailing_rows() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![4, 3] },
                (0..12).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "1,0".into());
        params.insert("limit_indices".into(), "3,3".into());

        let out = eval_primitive(Primitive::Slice, &[input], &params).unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![3, 4, 5, 6, 7, 8]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn slice_with_stride_1d() {
        let input = Value::vector_i64(&[10, 20, 30, 40, 50, 60]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "1".into());
        params.insert("limit_indices".into(), "6".into());
        params.insert("strides".into(), "2".into());

        let out = eval_primitive(Primitive::Slice, &[input], &params).unwrap();
        let expected = Value::vector_i64(&[20, 40, 60]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn slice_with_strides_2d() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![4, 5] },
                (0..20).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "0,1".into());
        params.insert("limit_indices".into(), "4,5".into());
        params.insert("strides".into(), "2,2".into());

        let out = eval_primitive(Primitive::Slice, &[input], &params).unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 2]);
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 3, 11, 13]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn slice_zero_stride_errors() {
        let input = Value::vector_i64(&[10, 20, 30]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "0".into());
        params.insert("limit_indices".into(), "3".into());
        params.insert("strides".into(), "0".into());

        let result = eval_primitive(Primitive::Slice, &[input], &params);
        assert!(result.is_err(), "zero stride should error");
    }

    #[test]
    fn test_lax_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("lax", "add")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_lax_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ===================================================================
    // NaN / Inf edge cases (IEEE 754 compliance)
    // ===================================================================

    #[test]
    fn add_nan_propagates() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(1.0)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn sub_inf_inf_is_nan() {
        let out = eval_primitive(
            Primitive::Sub,
            &[
                Value::scalar_f64(f64::INFINITY),
                Value::scalar_f64(f64::INFINITY),
            ],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn mul_nan_propagates() {
        let out = eval_primitive(
            Primitive::Mul,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(0.0)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn log_zero_is_neg_inf() {
        let out = eval_primitive(Primitive::Log, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v.is_infinite() && v < 0.0);
    }

    #[test]
    fn log_negative_is_nan() {
        let out = eval_primitive(Primitive::Log, &[Value::scalar_f64(-1.0)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn sqrt_negative_is_nan() {
        let out =
            eval_primitive(Primitive::Sqrt, &[Value::scalar_f64(-1.0)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn exp_neg_inf_is_zero() {
        let out = eval_primitive(
            Primitive::Exp,
            &[Value::scalar_f64(f64::NEG_INFINITY)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 0.0).abs() < 1e-15);
    }

    #[test]
    fn exp_inf_is_inf() {
        let out = eval_primitive(
            Primitive::Exp,
            &[Value::scalar_f64(f64::INFINITY)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_infinite());
    }

    #[test]
    fn neg_zero_is_neg_zero() {
        let out = eval_primitive(Primitive::Neg, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v == 0.0 && v.is_sign_negative());
    }

    #[test]
    fn abs_neg_zero_is_pos_zero() {
        let out = eval_primitive(Primitive::Abs, &[Value::scalar_f64(-0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v == 0.0 && v.is_sign_positive());
    }

    #[test]
    fn abs_nan_is_nan() {
        let out =
            eval_primitive(Primitive::Abs, &[Value::scalar_f64(f64::NAN)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn abs_neg_inf_is_inf() {
        let out = eval_primitive(
            Primitive::Abs,
            &[Value::scalar_f64(f64::NEG_INFINITY)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), f64::INFINITY);
    }

    #[test]
    fn reduce_sum_with_nan() {
        let input = Value::vector_f64(&[1.0, f64::NAN, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceSum, &[input], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn reduce_max_with_nan() {
        let input = Value::vector_f64(&[1.0, f64::NAN, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceMax, &[input], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn reduce_min_with_nan() {
        let input = Value::vector_f64(&[1.0, f64::NAN, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceMin, &[input], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn max_nan_propagates() {
        let out = eval_primitive(
            Primitive::Max,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(5.0)],
            &no_params(),
        )
        .unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    #[test]
    fn eq_nan_nan_is_false() {
        let out = eval_primitive(
            Primitive::Eq,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(f64::NAN)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    #[test]
    fn ne_nan_nan_is_true() {
        let out = eval_primitive(
            Primitive::Ne,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(f64::NAN)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(true)));
    }

    #[test]
    fn lt_nan_always_false() {
        let out = eval_primitive(
            Primitive::Lt,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(1.0)],
            &no_params(),
        );
        assert_eq!(out, Ok(Value::scalar_bool(false)));
    }

    // ===================================================================
    // Type promotion tests
    // ===================================================================

    #[test]
    fn add_i64_f64_promotes_to_f64() {
        let out = eval_primitive(
            Primitive::Add,
            &[Value::scalar_i64(2), Value::scalar_f64(3.5)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 5.5).abs() < 1e-10);
    }

    #[test]
    fn sub_vector_broadcasts_scalar() {
        let vec = Value::vector_i64(&[10, 20, 30]).unwrap();
        let out =
            eval_primitive(Primitive::Sub, &[vec, Value::scalar_i64(5)], &no_params()).unwrap();
        let expected = Value::vector_i64(&[5, 15, 25]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn mul_scalar_broadcasts_to_vector() {
        let vec = Value::vector_i64(&[1, 2, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Mul, &[Value::scalar_i64(10), vec], &no_params()).unwrap();
        let expected = Value::vector_i64(&[10, 20, 30]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn pow_int_int_cast_through_float() {
        // pow(2, 10) should give 1024 (cast through f64)
        let out = eval_primitive(
            Primitive::Pow,
            &[Value::scalar_i64(2), Value::scalar_i64(10)],
            &no_params(),
        )
        .unwrap();
        // i64 pow goes through f64 path: (2 as f64).powf(10 as f64) as i64 = 1024
        if let Value::Scalar(fj_core::Literal::I64(v)) = out {
            assert_eq!(v, 1024);
        } else {
            assert!(
                matches!(out, Value::Scalar(fj_core::Literal::I64(_))),
                "expected i64 scalar from int pow"
            );
        }
    }

    // ===================================================================
    // Broadcasting and shape mismatch error tests
    // ===================================================================

    #[test]
    fn add_shape_mismatch_error() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::Add, &[a, b], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    #[test]
    fn binary_wrong_arity_error() {
        let err =
            eval_primitive(Primitive::Add, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(
            err,
            EvalError::ArityMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));
    }

    #[test]
    fn unary_wrong_arity_error() {
        let err = eval_primitive(
            Primitive::Neg,
            &[Value::scalar_i64(1), Value::scalar_i64(2)],
            &no_params(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvalError::ArityMismatch {
                expected: 1,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn dot_wrong_arity_error() {
        let err =
            eval_primitive(Primitive::Dot, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(
            err,
            EvalError::ArityMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));
    }

    #[test]
    fn comparison_shape_mismatch_error() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::Eq, &[a, b], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    // ===================================================================
    // Reduction edge cases
    // ===================================================================

    #[test]
    fn reduce_sum_scalar_identity() {
        let out =
            eval_primitive(Primitive::ReduceSum, &[Value::scalar_i64(42)], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(42));
    }

    #[test]
    fn reduce_prod_f64_vector() {
        let input = Value::vector_f64(&[2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceProd, &[input], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 24.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_min_f64_with_neg_inf() {
        let input = Value::vector_f64(&[1.0, f64::NEG_INFINITY, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceMin, &[input], &no_params()).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn reduce_max_f64_with_inf() {
        let input = Value::vector_f64(&[1.0, f64::INFINITY, 3.0]).unwrap();
        let out = eval_primitive(Primitive::ReduceMax, &[input], &no_params()).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), f64::INFINITY);
    }

    // ===================================================================
    // Axis-aware reduction tests
    // ===================================================================

    fn axes_params(axes: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("axes".into(), axes.into());
        p
    }

    fn bool_tensor(dims: &[u32], elements: &[bool]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape {
                    dims: dims.to_vec(),
                },
                elements.iter().copied().map(Literal::Bool).collect(),
            )
            .unwrap(),
        )
    }

    fn i64_tensor(dims: &[u32], elements: &[i64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: dims.to_vec(),
                },
                elements.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn test_reduce_and_all_true() {
        let input = bool_tensor(&[3], &[true, true, true]);
        let out = eval_primitive(Primitive::ReduceAnd, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_bool(true));
    }

    #[test]
    fn test_reduce_and_one_false() {
        let input = bool_tensor(&[3], &[true, false, true]);
        let out = eval_primitive(Primitive::ReduceAnd, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_bool(false));
    }

    #[test]
    fn test_reduce_or_all_false() {
        let input = bool_tensor(&[3], &[false, false, false]);
        let out = eval_primitive(Primitive::ReduceOr, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_bool(false));
    }

    #[test]
    fn test_reduce_or_one_true() {
        let input = bool_tensor(&[3], &[false, true, false]);
        let out = eval_primitive(Primitive::ReduceOr, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_bool(true));
    }

    #[test]
    fn test_reduce_xor_even() {
        let input = bool_tensor(&[4], &[true, false, true, false]);
        let out = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_bool(false));
    }

    #[test]
    fn test_reduce_xor_odd() {
        let input = bool_tensor(&[4], &[true, false, true, true]);
        let out = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_bool(true));
    }

    #[test]
    fn test_reduce_and_axis() {
        let input = bool_tensor(&[2, 3], &[true, true, false, false, true, true]);
        let out = eval_primitive(Primitive::ReduceAnd, &[input], &axes_params("0")).unwrap();
        let expected = bool_tensor(&[3], &[false, true, false]);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_reduce_or_axis() {
        let input = bool_tensor(&[2, 3], &[false, false, true, false, false, false]);
        let out = eval_primitive(Primitive::ReduceOr, &[input], &axes_params("1")).unwrap();
        let expected = bool_tensor(&[2], &[true, false]);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_reduce_and_middle_axis_rank3_contiguous_block() {
        // Rank-3 [2,2,2] reducing the MIDDLE axis exercises the general
        // contiguous-block fast path with outer>1 AND inner>1 (outer=2, reduce=2,
        // inner=2) — the per-row vectorized fold, not the inner==1 scalar fold.
        // out[o,i] = AND over r of in[o,r,i]; layout in[((o*2+r)*2)+i].
        let input = bool_tensor(
            &[2, 2, 2],
            &[
                true, false, // o0 r0
                true, true, // o0 r1
                false, true, // o1 r0
                true, true, // o1 r1
            ],
        );
        let out = eval_primitive(Primitive::ReduceAnd, &[input], &axes_params("1")).unwrap();
        // o0: [t&t, f&t]=[t,f]; o1: [f&t, t&t]=[f,t]
        let expected = bool_tensor(&[2, 2], &[true, false, false, true]);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_reduce_or_empty_axes_identity() {
        let input = bool_tensor(&[2, 2], &[true, false, false, true]);
        let out = eval_primitive(
            Primitive::ReduceOr,
            std::slice::from_ref(&input),
            &axes_params(""),
        )
        .unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn test_reduce_xor_integer() {
        let input = i64_tensor(&[3], &[1, 3, 2]);
        let out = eval_primitive(Primitive::ReduceXor, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(0));
    }

    #[test]
    fn reduce_sum_axis0_rank2() {
        // [[1,2,3],[4,5,6]] shape [2,3] -> reduce axis 0 -> [5,7,9] shape [3]
        let input = Value::Tensor(
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
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("0")).unwrap();
        let expected = Value::vector_i64(&[5, 7, 9]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn reduce_sum_axis1_rank2() {
        // [[1,2,3],[4,5,6]] shape [2,3] -> reduce axis 1 -> [6,15] shape [2]
        let input = Value::Tensor(
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
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("1")).unwrap();
        let expected = Value::vector_i64(&[6, 15]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn reduce_sum_empty_axes_identity() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceSum,
            std::slice::from_ref(&input),
            &axes_params(""),
        )
        .unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn reduce_sum_both_axes_rank2() {
        // reducing both axes should give full reduction (scalar)
        let input = Value::Tensor(
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
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("0,1")).unwrap();
        assert_eq!(out, Value::scalar_i64(21));
    }

    #[test]
    fn reduce_max_axis0_rank2() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(1),
                    Literal::I64(5),
                    Literal::I64(3),
                    Literal::I64(4),
                    Literal::I64(2),
                    Literal::I64(6),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceMax, &[input], &axes_params("0")).unwrap();
        let expected = Value::vector_i64(&[4, 5, 6]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn reduce_min_axis1_rank2() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(3),
                    Literal::I64(1),
                    Literal::I64(5),
                    Literal::I64(6),
                    Literal::I64(2),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceMin, &[input], &axes_params("1")).unwrap();
        let expected = Value::vector_i64(&[1, 2]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn reduce_prod_axis0_rank2() {
        let input = Value::Tensor(
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
        );
        let out = eval_primitive(Primitive::ReduceProd, &[input], &axes_params("0")).unwrap();
        let expected = Value::vector_i64(&[4, 10, 18]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn reduce_sum_axis_out_of_bounds() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("1")).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn reduce_sum_axis0_rank3() {
        // shape [2,2,2] with values [1..8], reduce axis 0 -> shape [2,2]
        // [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 2, 2],
                },
                (1..=8).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("0")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 2]);
            let values: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(values, vec![6, 8, 10, 12]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor output");
        }
    }

    #[test]
    fn reduce_sum_f64_axis0() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("0")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3]);
            let values: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert!((values[0] - 5.0).abs() < 1e-10);
            assert!((values[1] - 7.0).abs() < 1e-10);
            assert!((values[2] - 9.0).abs() < 1e-10);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor output");
        }
    }

    // ===================================================================
    // Tensor manipulation edge cases
    // ===================================================================

    #[test]
    fn reshape_incompatible_size_error() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("new_shape".into(), "2,2".into());
        let err = eval_primitive(Primitive::Reshape, &[input], &params).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    #[test]
    fn transpose_invalid_permutation_error() {
        let input = fj_core::TensorValue::new(
            DType::I64,
            fj_core::Shape { dims: vec![2, 3] },
            vec![
                fj_core::Literal::I64(1),
                fj_core::Literal::I64(2),
                fj_core::Literal::I64(3),
                fj_core::Literal::I64(4),
                fj_core::Literal::I64(5),
                fj_core::Literal::I64(6),
            ],
        )
        .unwrap();
        let mut params = BTreeMap::new();
        params.insert("permutation".into(), "0".into()); // wrong length
        let err =
            eval_primitive(Primitive::Transpose, &[Value::Tensor(input)], &params).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn transpose_scalar_is_identity() {
        let out =
            eval_primitive(Primitive::Transpose, &[Value::scalar_i64(42)], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_i64(42));
    }

    #[test]
    fn slice_out_of_bounds_error() {
        let input = Value::vector_i64(&[10, 20, 30]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "1".into());
        params.insert("limit_indices".into(), "5".into()); // exceeds dim
        let err = eval_primitive(Primitive::Slice, &[input], &params).unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn concatenate_scalar_error() {
        let err = eval_primitive(
            Primitive::Concatenate,
            &[Value::scalar_i64(1), Value::scalar_i64(2)],
            &no_params(),
        )
        .unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn gather_wrong_arity_rejected() {
        let err =
            eval_primitive(Primitive::Gather, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ArityMismatch { .. }));
    }

    #[test]
    fn gather_scalar_operand_rejected() {
        let indices = Value::Tensor(
            TensorValue::new(DType::I64, Shape::vector(1), vec![Literal::I64(0)]).unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1".into());
        let err = eval_primitive(Primitive::Gather, &[Value::scalar_i64(1), indices], &params)
            .unwrap_err();
        assert!(matches!(err, EvalError::Unsupported { .. }));
    }

    #[test]
    fn gather_1d_indices_from_2d() {
        // operand: [[10,20],[30,40],[50,60]] (shape [3,2])
        // indices: [2, 0] — gather rows 2 and 0
        // slice_sizes: 1,2
        // result: [[50,60],[10,20]] (shape [2,2])
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                    Literal::I64(50),
                    Literal::I64(60),
                ],
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(2),
                vec![Literal::I64(2), Literal::I64(0)],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1,2".into());

        let out = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 2]);
            let vals: Vec<i64> = t
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(n) = l {
                        *n
                    } else {
                        assert!(matches!(l, Literal::I64(_)), "expected i64 literal");
                        0
                    }
                })
                .collect();
            assert_eq!(vals, vec![50, 60, 10, 20]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn gather_partial_trailing_slice_from_2d() {
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                    Literal::I64(50),
                    Literal::I64(60),
                ],
            )
            .unwrap(),
        );
        let indices = Value::vector_i64(&[1, 0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1,2".into());

        let out = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap();
        let tensor = out.as_tensor().expect("gather output should be tensor");
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        let vals: Vec<i64> = tensor
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("i64 gather output"))
            .collect();
        assert_eq!(vals, vec![40, 50, 10, 20]);
    }

    #[test]
    fn gather_out_of_bounds_clips_by_default() {
        // JAX parity: gather never raises on OOB; the default mode (CLIP) clamps the
        // index into [0, dim-1], so index 5 against dim 3 clamps to 2 -> element 3.
        let operand = Value::vector_i64(&[1, 2, 3]).unwrap();
        let indices = Value::Tensor(
            TensorValue::new(DType::I64, Shape::vector(1), vec![Literal::I64(5)]).unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1".into());
        let result = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap();
        match result {
            Value::Tensor(t) => assert_eq!(t.elements, vec![Literal::I64(3)]),
            Value::Scalar(_) => panic!("expected tensor"),
        }
    }

    #[test]
    fn scatter_1d_indices_into_2d() {
        // operand: [[0,0],[0,0],[0,0]] (shape [3,2])
        // indices: [1, 0]
        // updates: [[10,20],[30,40]] (shape [2,2])
        // result:  [[30,40],[10,20],[0,0]]
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
                vec![Literal::I64(0); 6],
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(2),
                vec![Literal::I64(1), Literal::I64(0)],
            )
            .unwrap(),
        );
        let updates = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3, 2]);
            let vals: Vec<i64> = t
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(n) = l {
                        *n
                    } else {
                        assert!(matches!(l, Literal::I64(_)), "expected i64 literal");
                        0
                    }
                })
                .collect();
            assert_eq!(vals, vec![30, 40, 10, 20, 0, 0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn scatter_scalar_operand_rejected() {
        let err =
            eval_primitive(Primitive::Scatter, &[Value::scalar_i64(1)], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ArityMismatch { .. }));
    }

    #[test]
    fn gather_1d_simple() {
        // operand: [10, 20, 30, 40, 50] (shape [5])
        // indices: [3, 1, 4]
        // slice_sizes: 1
        // result: [40, 20, 50]
        let operand = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(3),
                vec![Literal::I64(3), Literal::I64(1), Literal::I64(4)],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1".into());

        let out = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3]);
            let vals: Vec<i64> = t
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(n) = l {
                        *n
                    } else {
                        assert!(matches!(l, Literal::I64(_)), "expected i64 literal");
                        0
                    }
                })
                .collect();
            assert_eq!(vals, vec![40, 20, 50]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    // ===================================================================
    // Trigonometric tests
    // ===================================================================

    #[test]
    fn sin_zero_is_zero() {
        let out = eval_primitive(Primitive::Sin, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 0.0).abs() < 1e-15);
    }

    #[test]
    fn cos_zero_is_one() {
        let out = eval_primitive(Primitive::Cos, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 1.0).abs() < 1e-15);
    }

    #[test]
    fn sin_pi_is_zero() {
        let out = eval_primitive(
            Primitive::Sin,
            &[Value::scalar_f64(std::f64::consts::PI)],
            &no_params(),
        )
        .unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!(v.abs() < 1e-14);
    }

    #[test]
    fn sin_nan_is_nan() {
        let out =
            eval_primitive(Primitive::Sin, &[Value::scalar_f64(f64::NAN)], &no_params()).unwrap();
        assert!(out.as_f64_scalar().unwrap().is_nan());
    }

    // ===================================================================
    // Comparison with broadcast (scalar + tensor)
    // ===================================================================

    #[test]
    fn gt_scalar_tensor_broadcast() {
        let vec = Value::vector_i64(&[1, 5, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Gt, &[Value::scalar_i64(3), vec], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements[0], fj_core::Literal::Bool(true)); // 3 > 1
            assert_eq!(t.elements[1], fj_core::Literal::Bool(false)); // 3 > 5
            assert_eq!(t.elements[2], fj_core::Literal::Bool(false)); // 3 > 3
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn le_tensor_scalar_broadcast() {
        let vec = Value::vector_i64(&[1, 5, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Le, &[vec, Value::scalar_i64(3)], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements[0], fj_core::Literal::Bool(true)); // 1 <= 3
            assert_eq!(t.elements[1], fj_core::Literal::Bool(false)); // 5 <= 3
            assert_eq!(t.elements[2], fj_core::Literal::Bool(true)); // 3 <= 3
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    // ===================================================================
    // Dot product edge cases
    // ===================================================================

    #[test]
    fn dot_f64_vectors() {
        let lhs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let rhs = Value::vector_f64(&[4.0, 5.0, 6.0]).unwrap();
        let out = eval_primitive(Primitive::Dot, &[lhs, rhs], &no_params()).unwrap();
        let v = out.as_f64_scalar().unwrap();
        assert!((v - 32.0).abs() < 1e-10);
    }

    #[test]
    fn dot_scalar_multiply() {
        let out = eval_primitive(
            Primitive::Dot,
            &[Value::scalar_i64(3), Value::scalar_i64(7)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_i64(21));
    }

    #[test]
    fn dot_shape_mismatch_error() {
        let a = Value::vector_i64(&[1, 2]).unwrap();
        let b = Value::vector_i64(&[1, 2, 3]).unwrap();
        let err = eval_primitive(Primitive::Dot, &[a, b], &no_params()).unwrap_err();
        assert!(matches!(err, EvalError::ShapeMismatch { .. }));
    }

    // ===================================================================
    // EvalError display formatting
    // ===================================================================

    #[test]
    fn eval_error_display_formatting() {
        let err = EvalError::ArityMismatch {
            primitive: Primitive::Add,
            expected: 2,
            actual: 1,
        };
        let msg = format!("{err}");
        assert!(msg.contains("add"));
        assert!(msg.contains("expected 2"));
        assert!(msg.contains("got 1"));
    }

    // ===================================================================
    // Vector-level elementwise tests for transcendentals
    // ===================================================================

    #[test]
    fn exp_vector() {
        let input = Value::vector_f64(&[0.0, 1.0]).unwrap();
        let out = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let e0 = t.elements[0].as_f64().unwrap();
            let e1 = t.elements[1].as_f64().unwrap();
            assert!((e0 - 1.0).abs() < 1e-10);
            assert!((e1 - std::f64::consts::E).abs() < 1e-10);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn floor_vector() {
        let input = Value::vector_f64(&[1.9, 2.1, -0.5]).unwrap();
        let out = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert!((t.elements[0].as_f64().unwrap() - 1.0).abs() < 1e-10);
            assert!((t.elements[1].as_f64().unwrap() - 2.0).abs() < 1e-10);
            assert!((t.elements[2].as_f64().unwrap() - (-1.0)).abs() < 1e-10);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn rsqrt_vector() {
        let input = Value::vector_f64(&[1.0, 4.0, 16.0]).unwrap();
        let out = eval_primitive(Primitive::Rsqrt, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert!((t.elements[0].as_f64().unwrap() - 1.0).abs() < 1e-10);
            assert!((t.elements[1].as_f64().unwrap() - 0.5).abs() < 1e-10);
            assert!((t.elements[2].as_f64().unwrap() - 0.25).abs() < 1e-10);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    // ── Select broadcasting tests ─────────────────────────────────

    #[test]
    fn select_scalar_all_scalars() {
        let out = eval_primitive(
            Primitive::Select,
            &[
                Value::scalar_bool(true),
                Value::scalar_f64(1.0),
                Value::scalar_f64(0.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::Scalar(Literal::from_f64(1.0)));
    }

    #[test]
    fn select_tensor_cond_scalar_values() {
        let cond = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape::vector(3),
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::Select,
            &[cond, Value::scalar_f64(10.0), Value::scalar_f64(-1.0)],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![10.0, -1.0, 10.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor output");
        }
    }

    #[test]
    fn select_scalar_cond_tensor_values() {
        let on_true = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let on_false = Value::vector_f64(&[4.0, 5.0, 6.0]).unwrap();
        let out = eval_primitive(
            Primitive::Select,
            &[Value::scalar_bool(false), on_true, on_false.clone()],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, on_false);
    }

    // ===================================================================
    // Scatter mode="add" tests
    // ===================================================================

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_scatter_add_vs_jax() {
        use std::time::Instant;
        let n_tab = 1_000_000usize;
        let n_upd = 1_000_000usize;
        let operand = Value::Tensor(
            TensorValue::new_f64_values(Shape::vector(n_tab as u32), vec![0.0; n_tab]).unwrap(),
        );
        let idx: Vec<i64> = (0..n_upd)
            .map(|i| (i.wrapping_mul(2654435761) % n_tab) as i64)
            .collect();
        let indices =
            Value::Tensor(TensorValue::new_i64_values(Shape::vector(n_upd as u32), idx).unwrap());
        let upd: Vec<f64> = (0..n_upd).map(|i| ((i % 9973) as f64) * 1e-3).collect();
        let updates =
            Value::Tensor(TensorValue::new_f64_values(Shape::vector(n_upd as u32), upd).unwrap());
        let p = BTreeMap::from([("mode".to_owned(), "add".to_owned())]);
        let f = || {
            std::hint::black_box(
                eval_primitive(
                    Primitive::Scatter,
                    &[operand.clone(), indices.clone(), updates.clone()],
                    &p,
                )
                .unwrap(),
            );
        };
        f();
        let mut b = f64::MAX;
        for _ in 0..6 {
            let s = Instant::now();
            f();
            b = b.min(s.elapsed().as_secs_f64());
        }
        println!(
            "fj-lax scatter-add f64 (table 1M, 1M updates): {:.3}ms | JAX=4.78ms",
            b * 1e3
        );
    }

    #[test]
    fn scatter_add_mode_accumulates() {
        // operand: [0.0, 0.0, 0.0] (shape [3])
        // indices: [1, 1]  (duplicate index)
        // updates: [10.0, 20.0]
        // With mode="add", index 1 should accumulate: 0 + 10 + 20 = 30
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(3),
                vec![
                    Literal::from_f64(0.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(0.0),
                ],
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(2),
                vec![Literal::I64(1), Literal::I64(1)],
            )
            .unwrap(),
        );
        let updates = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(2),
                vec![Literal::from_f64(10.0), Literal::from_f64(20.0)],
            )
            .unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("mode".into(), "add".into());

        let out =
            eval_primitive(Primitive::Scatter, &[operand, indices, updates], &params).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals[0], 0.0);
            assert_eq!(vals[1], 30.0); // 10 + 20 accumulated
            assert_eq!(vals[2], 0.0);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn scatter_add_mode_i64_accumulates() {
        let operand = Value::vector_i64(&[0, 0, 0]).unwrap();
        let indices = Value::vector_i64(&[1, 1]).unwrap();
        let updates = Value::vector_i64(&[10, 20]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("mode".into(), "add".into());

        let out =
            eval_primitive(Primitive::Scatter, &[operand, indices, updates], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements[0], Literal::I64(0));
            assert_eq!(t.elements[1], Literal::I64(30));
            assert_eq!(t.elements[2], Literal::I64(0));
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn scatter_add_preserves_existing_values() {
        // operand: [100.0, 200.0, 300.0]
        // indices: [0]
        // updates: [5.0]
        // mode="add": result[0] = 100 + 5 = 105
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(3),
                vec![
                    Literal::from_f64(100.0),
                    Literal::from_f64(200.0),
                    Literal::from_f64(300.0),
                ],
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(DType::I64, Shape::vector(1), vec![Literal::I64(0)]).unwrap(),
        );
        let updates = Value::Tensor(
            TensorValue::new(DType::F64, Shape::vector(1), vec![Literal::from_f64(5.0)]).unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("mode".into(), "add".into());

        let out =
            eval_primitive(Primitive::Scatter, &[operand, indices, updates], &params).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![105.0, 200.0, 300.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    // ===================================================================
    // Concatenate edge cases
    // ===================================================================

    // ===================================================================
    // Gather edge cases
    // ===================================================================

    #[test]
    fn gather_slice_sizes_exceed_operand_dims_rejected() {
        // operand shape [3, 2], slice_sizes [1, 5] — 5 > 2 should fail
        let operand = Value::Tensor(
            fj_core::TensorValue::new(
                fj_core::DType::F64,
                fj_core::Shape { dims: vec![3, 2] },
                vec![
                    fj_core::Literal::from_f64(1.0),
                    fj_core::Literal::from_f64(2.0),
                    fj_core::Literal::from_f64(3.0),
                    fj_core::Literal::from_f64(4.0),
                    fj_core::Literal::from_f64(5.0),
                    fj_core::Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );
        let indices = Value::scalar_i64(0);
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1,5".into());
        let result = eval_primitive(Primitive::Gather, &[operand, indices], &params);
        assert!(result.is_err(), "slice_sizes[1]=5 exceeds dim=2");
    }

    #[test]
    fn gather_axis0_slice_size_must_be_one() {
        let operand = Value::vector_i64(&[10, 20, 30, 40]).unwrap();
        let indices = Value::vector_i64(&[1]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "2".into());
        let result = eval_primitive(Primitive::Gather, &[operand, indices], &params);
        assert!(result.is_err(), "slice_sizes[0] != 1 should be rejected");
    }

    #[test]
    fn gather_empty_indices() {
        let operand = Value::vector_f64(&[10.0, 20.0, 30.0]).unwrap();
        let indices = Value::Tensor(
            fj_core::TensorValue::new(
                fj_core::DType::I64,
                fj_core::Shape { dims: vec![0] },
                vec![],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1".into());
        let result = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap();
        if let Value::Tensor(t) = &result {
            assert_eq!(t.elements.len(), 0);
            assert_eq!(t.shape.dims[0], 0);
        } else {
            assert!(matches!(result, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn gather_empty_huge_trailing_slice_returns_empty() {
        let huge = u32::MAX;
        let operand = Value::Tensor(
            fj_core::TensorValue::new(
                fj_core::DType::I64,
                fj_core::Shape {
                    dims: vec![1, 0, huge, huge, huge],
                },
                Vec::new(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), format!("1,0,{huge},{huge},{huge}"));

        let result =
            eval_primitive(Primitive::Gather, &[operand, Value::scalar_i64(0)], &params).unwrap();

        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![0, huge, huge, huge]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn gather_empty_slice_out_of_bounds_returns_empty() {
        // JAX parity: gather never raises on OOB. With a zero-size trailing slice the
        // output is empty regardless of the index, so an OOB index yields an empty
        // tensor rather than an error.
        let operand = Value::Tensor(
            fj_core::TensorValue::new(
                fj_core::DType::I64,
                fj_core::Shape { dims: vec![1, 0] },
                Vec::new(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1,0".into());

        let result =
            eval_primitive(Primitive::Gather, &[operand, Value::scalar_i64(2)], &params).unwrap();
        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![0]);
        assert!(tensor.elements.is_empty());
    }

    // ===================================================================
    // Scatter edge cases
    // ===================================================================

    #[test]
    fn scatter_duplicate_indices_overwrite_last_wins() {
        // indices [0, 0] with updates [10, 20] — last write wins
        let operand = Value::vector_f64(&[0.0, 0.0, 0.0]).unwrap();
        let indices = Value::vector_i64(&[0, 0]).unwrap();
        let updates = Value::vector_f64(&[10.0, 20.0]).unwrap();
        let out = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals[0], 20.0, "last write should win for duplicate indices");
            assert_eq!(vals[1], 0.0);
            assert_eq!(vals[2], 0.0);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn scatter_empty_huge_trailing_slice_returns_operand() {
        let huge = u32::MAX;
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![1, 0, huge, huge, huge],
                },
                Vec::new(),
            )
            .unwrap(),
        );
        let updates = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![0, huge, huge, huge],
                },
                Vec::new(),
            )
            .unwrap(),
        );

        let result = eval_primitive(
            Primitive::Scatter,
            &[operand, Value::scalar_i64(0), updates],
            &no_params(),
        )
        .unwrap();

        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![1, 0, huge, huge, huge]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn scatter_empty_slice_out_of_bounds_returns_operand() {
        // JAX parity: scatter never raises on OOB. With a zero-size update slice there
        // is nothing to write, so an OOB index leaves the operand unchanged rather than
        // raising.
        let operand = Value::Tensor(
            TensorValue::new(DType::I64, Shape { dims: vec![1, 0] }, Vec::new()).unwrap(),
        );
        let updates = Value::Tensor(
            TensorValue::new(DType::I64, Shape { dims: vec![0] }, Vec::new()).unwrap(),
        );

        let result = eval_primitive(
            Primitive::Scatter,
            &[operand, Value::scalar_i64(2), updates],
            &no_params(),
        )
        .unwrap();
        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![1, 0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn scatter_duplicate_row_overwrite_last_wins() {
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
                vec![Literal::I64(0); 6],
            )
            .unwrap(),
        );
        let indices = Value::vector_i64(&[1, 1]).unwrap();
        let updates = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        )
        .unwrap();
        let tensor = out.as_tensor().expect("scatter output should be tensor");
        let values: Vec<i64> = tensor
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("i64 scatter output"))
            .collect();
        assert_eq!(values, vec![0, 0, 30, 40, 0, 0]);
    }

    #[test]
    fn scatter_unknown_mode_rejected() {
        let operand = Value::vector_f64(&[1.0, 2.0]).unwrap();
        let indices = Value::vector_i64(&[0]).unwrap();
        let updates = Value::vector_f64(&[9.0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("mode".into(), "invalid_mode".into());
        let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &params);
        assert!(result.is_err(), "unknown mode should be rejected");
    }

    #[test]
    fn scatter_updates_shape_mismatch_rejected() {
        // operand [3], indices [1], updates [2] — updates has 2 elems but expected 1
        let operand = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let indices = Value::vector_i64(&[0]).unwrap();
        let updates = Value::vector_f64(&[10.0, 20.0]).unwrap();
        let result = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        );
        assert!(
            result.is_err(),
            "updates element count mismatch should error"
        );
    }

    #[test]
    fn scatter_updates_shape_dims_mismatch_rejected() {
        // operand [2,2], indices [2], updates [4] (same element count, wrong shape)
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::from_f64(0.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(0.0),
                ],
            )
            .unwrap(),
        );
        let indices = Value::vector_i64(&[0, 1]).unwrap();
        let updates = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(4),
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                ],
            )
            .unwrap(),
        );
        let result = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        );
        assert!(matches!(result, Err(EvalError::ShapeMismatch { .. })));
    }

    #[test]
    fn scatter_scalar_update_for_scalar_index() {
        let operand = Value::vector_i64(&[0, 0, 0]).unwrap();
        let indices = Value::scalar_i64(1);
        let updates = Value::scalar_i64(7);
        let out = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(
                t.elements,
                vec![Literal::I64(0), Literal::I64(7), Literal::I64(0)]
            );
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    // ===================================================================
    // Slice edge cases
    // ===================================================================

    #[test]
    fn slice_empty_result() {
        // slice with start == limit should produce empty tensor
        let v = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "2".into());
        params.insert("limit_indices".into(), "2".into());
        let out = eval_primitive(Primitive::Slice, &[v], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements.len(), 0);
            assert_eq!(t.shape.dims[0], 0);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn slice_single_element() {
        let v = Value::vector_f64(&[10.0, 20.0, 30.0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "1".into());
        params.insert("limit_indices".into(), "2".into());
        let out = eval_primitive(Primitive::Slice, &[v], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.elements.len(), 1);
            assert_eq!(t.elements[0].as_f64().unwrap(), 20.0);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn slice_start_exceeds_limit_rejected() {
        let v = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("start_indices".into(), "2".into());
        params.insert("limit_indices".into(), "1".into());
        let result = eval_primitive(Primitive::Slice, &[v], &params);
        assert!(result.is_err(), "start > limit should error");
    }

    #[test]
    fn concatenate_single_input() {
        let a = Value::vector_i64(&[1, 2, 3]).unwrap();
        let out = eval_primitive(
            Primitive::Concatenate,
            std::slice::from_ref(&a),
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, a);
    }

    #[test]
    fn concatenate_three_inputs() {
        let a = Value::vector_i64(&[1]).unwrap();
        let b = Value::vector_i64(&[2, 3]).unwrap();
        let c = Value::vector_i64(&[4, 5, 6]).unwrap();
        let out = eval_primitive(Primitive::Concatenate, &[a, b, c], &no_params()).unwrap();
        let expected = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(out, expected);
    }

    // ══════════════════════════════════════════════════════════════
    // Clamp tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn clamp_scalar_within_range() {
        let out = eval_primitive(
            Primitive::Clamp,
            &[
                Value::scalar_f64(3.0),
                Value::scalar_f64(1.0),
                Value::scalar_f64(5.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_f64(3.0));
    }

    #[test]
    fn clamp_scalar_below_min() {
        let out = eval_primitive(
            Primitive::Clamp,
            &[
                Value::scalar_f64(-2.0),
                Value::scalar_f64(1.0),
                Value::scalar_f64(5.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_f64(1.0));
    }

    #[test]
    fn clamp_scalar_above_max() {
        let out = eval_primitive(
            Primitive::Clamp,
            &[
                Value::scalar_f64(10.0),
                Value::scalar_f64(1.0),
                Value::scalar_f64(5.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_f64(5.0));
    }

    #[test]
    fn clamp_i64_scalar() {
        let out = eval_primitive(
            Primitive::Clamp,
            &[
                Value::scalar_i64(10),
                Value::scalar_i64(0),
                Value::scalar_i64(5),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_i64(5));
    }

    #[test]
    fn clamp_tensor_with_scalar_bounds() {
        let x = Value::vector_f64(&[-1.0, 2.0, 5.0, 8.0]).unwrap();
        let out = eval_primitive(
            Primitive::Clamp,
            &[x, Value::scalar_f64(0.0), Value::scalar_f64(6.0)],
            &no_params(),
        )
        .unwrap();
        let expected = Value::vector_f64(&[0.0, 2.0, 5.0, 6.0]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn clamp_arity_error() {
        let result = eval_primitive(
            Primitive::Clamp,
            &[Value::scalar_f64(1.0), Value::scalar_f64(0.0)],
            &no_params(),
        );
        assert!(result.is_err());
    }

    // ══════════════════════════════════════════════════════════════
    // Iota tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn iota_i64_length_5() {
        let mut params = BTreeMap::new();
        params.insert("length".to_owned(), "5".to_owned());
        let out = eval_primitive(Primitive::Iota, &[], &params).unwrap();
        let expected = Value::vector_i64(&[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn iota_f64() {
        let mut params = BTreeMap::new();
        params.insert("length".to_owned(), "3".to_owned());
        params.insert("dtype".to_owned(), "F64".to_owned());
        let out = eval_primitive(Primitive::Iota, &[], &params).unwrap();
        let expected = Value::vector_f64(&[0.0, 1.0, 2.0]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn iota_supports_unsigned_half_and_complex_numeric_dtypes() {
        let mut u64_params = BTreeMap::new();
        u64_params.insert("length".to_owned(), "3".to_owned());
        u64_params.insert("dtype".to_owned(), "u64".to_owned());
        let u64_out = eval_primitive(Primitive::Iota, &[], &u64_params).unwrap();
        let u64_tensor = u64_out.as_tensor().expect("tensor output expected");
        assert_eq!(u64_tensor.dtype, DType::U64);
        assert_eq!(
            u64_tensor.elements,
            vec![Literal::U64(0), Literal::U64(1), Literal::U64(2)]
        );

        let mut bf16_params = BTreeMap::new();
        bf16_params.insert("length".to_owned(), "2".to_owned());
        bf16_params.insert("dtype".to_owned(), "bf16".to_owned());
        let bf16_out = eval_primitive(Primitive::Iota, &[], &bf16_params).unwrap();
        let bf16_tensor = bf16_out.as_tensor().expect("tensor output expected");
        assert_eq!(bf16_tensor.dtype, DType::BF16);
        assert_eq!(bf16_tensor.elements[0].as_f64(), Some(0.0));
        assert_eq!(bf16_tensor.elements[1].as_f64(), Some(1.0));

        let mut complex_params = BTreeMap::new();
        complex_params.insert("length".to_owned(), "3".to_owned());
        complex_params.insert("dtype".to_owned(), "complex64".to_owned());
        let complex_out = eval_primitive(Primitive::Iota, &[], &complex_params).unwrap();
        let complex_tensor = complex_out.as_tensor().expect("tensor output expected");
        assert_eq!(complex_tensor.dtype, DType::Complex64);
        let complex_values: Vec<(f32, f32)> = complex_tensor
            .elements
            .iter()
            .map(|lit| lit.as_complex64().expect("complex64 element"))
            .collect();
        assert_eq!(complex_values, vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]);
    }

    #[test]
    fn iota_rejects_bool_dtype() {
        let mut params = BTreeMap::new();
        params.insert("length".to_owned(), "3".to_owned());
        params.insert("dtype".to_owned(), "bool".to_owned());
        let err =
            eval_primitive(Primitive::Iota, &[], &params).expect_err("JAX iota rejects bool dtype");
        assert!(
            err.to_string().contains("iota does not accept bool dtype"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn iota_zero_length() {
        let mut params = BTreeMap::new();
        params.insert("length".to_owned(), "0".to_owned());
        let out = eval_primitive(Primitive::Iota, &[], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape, Shape::vector(0));
            assert!(t.elements.is_empty());
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn iota_arity_error_with_input() {
        let mut params = BTreeMap::new();
        params.insert("length".to_owned(), "3".to_owned());
        let result = eval_primitive(Primitive::Iota, &[Value::scalar_i64(1)], &params);
        assert!(result.is_err());
    }

    // ══════════════════════════════════════════════════════════════
    // DynamicSlice tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn dynamic_slice_1d() {
        let x = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "3".to_owned());
        let out =
            eval_primitive(Primitive::DynamicSlice, &[x, Value::scalar_i64(1)], &params).unwrap();
        let expected = Value::vector_i64(&[20, 30, 40]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dynamic_slice_start_clamping() {
        // JAX clamps start indices to valid range
        let x = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "3".to_owned());
        // Start index 10 should be clamped to 2 (5 - 3 = 2)
        let out = eval_primitive(
            Primitive::DynamicSlice,
            &[x, Value::scalar_i64(10)],
            &params,
        )
        .unwrap();
        let expected = Value::vector_i64(&[30, 40, 50]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dynamic_slice_negative_start_clamped() {
        let x = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "2".to_owned());
        // -5 is relative to the end, then lands at 0.
        let out = eval_primitive(
            Primitive::DynamicSlice,
            &[x, Value::scalar_i64(-5)],
            &params,
        )
        .unwrap();
        let expected = Value::vector_i64(&[10, 20]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dynamic_slice_negative_start_relative_to_end() {
        let x = Value::vector_i64(&[10, 20, 30, 40, 50]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "2".to_owned());
        let out = eval_primitive(
            Primitive::DynamicSlice,
            &[x, Value::scalar_i64(-3)],
            &params,
        )
        .unwrap();
        let expected = Value::vector_i64(&[30, 40]).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn dynamic_slice_rejects_float_start() {
        let x = Value::vector_i64(&[10, 20, 30, 40]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "2".to_owned());
        let result = eval_primitive(
            Primitive::DynamicSlice,
            &[x, Value::scalar_f64(1.0)],
            &params,
        );
        assert!(result.is_err());
    }

    #[test]
    fn dynamic_slice_2d() {
        let t = TensorValue::new(
            DType::I64,
            Shape { dims: vec![3, 4] },
            (0..12).map(Literal::I64).collect(),
        )
        .unwrap();
        let x = Value::Tensor(t);
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "2,2".to_owned());
        let out = eval_primitive(
            Primitive::DynamicSlice,
            &[x, Value::scalar_i64(1), Value::scalar_i64(1)],
            &params,
        )
        .unwrap();
        // Extracting a 2x2 block starting at (1,1) from a 3x4 matrix
        // Matrix: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
        // Expected: [[5,6],[9,10]]
        let expected = TensorValue::new(
            DType::I64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::I64(5),
                Literal::I64(6),
                Literal::I64(9),
                Literal::I64(10),
            ],
        )
        .unwrap();
        assert_eq!(out, Value::Tensor(expected));
    }

    #[test]
    fn dynamic_slice_full_trailing_rows() {
        let t = TensorValue::new(
            DType::I64,
            Shape { dims: vec![4, 3] },
            (0..12).map(Literal::I64).collect(),
        )
        .unwrap();
        let x = Value::Tensor(t);
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "2,3".to_owned());
        let out = eval_primitive(
            Primitive::DynamicSlice,
            &[x, Value::scalar_i64(1), Value::scalar_i64(0)],
            &params,
        )
        .unwrap();
        let expected = TensorValue::new(
            DType::I64,
            Shape { dims: vec![2, 3] },
            vec![
                Literal::I64(3),
                Literal::I64(4),
                Literal::I64(5),
                Literal::I64(6),
                Literal::I64(7),
                Literal::I64(8),
            ],
        )
        .unwrap();
        assert_eq!(out, Value::Tensor(expected));
    }

    #[test]
    fn dynamic_slice_empty_huge_trailing_shape_returns_empty() {
        let huge = u32::MAX;
        let t = TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![0, huge, huge, huge, huge],
            },
            Vec::new(),
        )
        .unwrap();
        let x = Value::Tensor(t);
        let mut params = BTreeMap::new();
        params.insert(
            "slice_sizes".to_owned(),
            "0,4294967295,4294967295,4294967295,4294967295".to_owned(),
        );
        let out = eval_primitive(
            Primitive::DynamicSlice,
            &[
                x,
                Value::scalar_i64(0),
                Value::scalar_i64(0),
                Value::scalar_i64(0),
                Value::scalar_i64(0),
                Value::scalar_i64(0),
            ],
            &params,
        )
        .unwrap();

        let tensor = out.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![0, huge, huge, huge, huge]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn dynamic_slice_scalar_error() {
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "1".to_owned());
        let result = eval_primitive(
            Primitive::DynamicSlice,
            &[Value::scalar_i64(42), Value::scalar_i64(0)],
            &params,
        );
        assert!(result.is_err());
    }

    #[test]
    fn dynamic_slice_oversize_errors() {
        let x = Value::vector_i64(&[1, 2, 3]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "5".to_owned());
        let result = eval_primitive(Primitive::DynamicSlice, &[x, Value::scalar_i64(0)], &params);
        assert!(result.is_err());
    }

    // ── Higher-rank gather/scatter tests ─────────────────────────

    #[test]
    fn gather_rank3_operand() {
        // operand: shape [3, 2, 2] — 3 matrices of 2x2
        // indices: [2, 0]
        // slice_sizes: 1, 2, 2
        // result: shape [2, 2, 2] — matrices 2 and 0
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![3, 2, 2],
                },
                (1..=12).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(2),
                vec![Literal::I64(2), Literal::I64(0)],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".into(), "1,2,2".into());

        let out = eval_primitive(Primitive::Gather, &[operand, indices], &params).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 2, 2]);
            let vals: Vec<i64> = t
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(n) = l {
                        *n
                    } else {
                        assert!(matches!(l, Literal::I64(_)), "expected i64 literal");
                        0
                    }
                })
                .collect();
            // index 2: elements 9,10,11,12; index 0: elements 1,2,3,4
            assert_eq!(vals, vec![9, 10, 11, 12, 1, 2, 3, 4]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn scatter_into_rank3_operand() {
        // operand: shape [3, 2, 2] all zeros
        // indices: [1]
        // updates: shape [1, 2, 2] = [[10, 20], [30, 40]]
        // result: slot 1 of operand replaced with updates
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![3, 2, 2],
                },
                vec![Literal::I64(0); 12],
            )
            .unwrap(),
        );
        let indices = Value::Tensor(
            TensorValue::new(DType::I64, Shape::vector(1), vec![Literal::I64(1)]).unwrap(),
        );
        let updates = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![1, 2, 2],
                },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(
            Primitive::Scatter,
            &[operand, indices, updates],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3, 2, 2]);
            let vals: Vec<i64> = t
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(n) = l {
                        *n
                    } else {
                        assert!(matches!(l, Literal::I64(_)), "expected i64 literal");
                        0
                    }
                })
                .collect();
            // Slot 0: [0,0,0,0], Slot 1: [10,20,30,40], Slot 2: [0,0,0,0]
            assert_eq!(vals, vec![0, 0, 0, 0, 10, 20, 30, 40, 0, 0, 0, 0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    // ── Select tensor tests ────────────────────────────────────────────

    #[test]
    fn select_tensor_condition_picks_elementwise() {
        // cond=[true,false,true], on_true=[10,20,30], on_false=[1,2,3]
        // expected: [10,2,30]
        let cond = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape::vector(3),
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .unwrap(),
        );
        let on_true = Value::vector_i64(&[10, 20, 30]).unwrap();
        let on_false = Value::vector_i64(&[1, 2, 3]).unwrap();
        let out =
            eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3]);
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![10, 2, 30]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn select_rank2_tensor() {
        // 2x2 tensors
        let cond = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .unwrap(),
        );
        let on_true = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let on_false = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                ],
            )
            .unwrap(),
        );
        let out =
            eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 2]);
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 20, 30, 4]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn select_tensor_shape_mismatch_errors() {
        let cond = Value::Tensor(
            TensorValue::new(DType::Bool, Shape::vector(3), vec![Literal::Bool(true); 3]).unwrap(),
        );
        let on_true = Value::vector_i64(&[1, 2]).unwrap(); // shape [2] != [3]
        let on_false = Value::vector_i64(&[10, 20]).unwrap();
        let result = eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params());
        assert!(result.is_err());
    }

    // ── OneHot tests ──────────────────────────────────────────────────

    fn one_hot_params(num_classes: u32) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("num_classes".to_owned(), num_classes.to_string());
        p
    }

    // one_hot vs JAX (measured JAX f32 [200000,512] = 48.0ms). fj-lax is fill-off + scatter-on (O(total)
    // fill + O(idx) scatter) — should beat JAX's broadcast-compare (O(total) compare).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_one_hot_vs_jax() {
        use std::time::Instant;
        let (n_idx, n_cls) = (200_000usize, 512u32);
        let idx: Vec<i64> = (0..n_idx)
            .map(|i| (i.wrapping_mul(2654435761) % 512) as i64)
            .collect();
        let indices =
            Value::Tensor(TensorValue::new_i64_values(Shape::vector(n_idx as u32), idx).unwrap());
        let mut p = BTreeMap::new();
        p.insert("num_classes".to_owned(), n_cls.to_string());
        p.insert("dtype".to_owned(), "f32".to_owned());
        let f = || {
            std::hint::black_box(
                eval_primitive(Primitive::OneHot, std::slice::from_ref(&indices), &p).unwrap(),
            );
        };
        f();
        let mut b = f64::MAX;
        for _ in 0..6 {
            let s = Instant::now();
            f();
            b = b.min(s.elapsed().as_secs_f64());
        }
        println!(
            "fj-lax one_hot f32 [200000,512]: {:.3}ms | JAX=48.0ms",
            b * 1e3
        );
    }

    #[test]
    fn one_hot_vector_indices() {
        // one_hot([0, 2, 1], num_classes=3) → [[1,0,0],[0,0,1],[0,1,0]]
        let indices = Value::vector_i64(&[0, 2, 1]).unwrap();
        let out = eval_primitive(Primitive::OneHot, &[indices], &one_hot_params(3)).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3, 3]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn one_hot_scalar_index() {
        // one_hot(2, num_classes=4) → [0,0,1,0]
        let indices = Value::scalar_i64(2);
        let out = eval_primitive(Primitive::OneHot, &[indices], &one_hot_params(4)).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![4]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![0.0, 0.0, 1.0, 0.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn one_hot_out_of_range_index() {
        // Negative index → all off_value
        let indices = Value::scalar_i64(-1);
        let out = eval_primitive(Primitive::OneHot, &[indices], &one_hot_params(3)).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![0.0, 0.0, 0.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn one_hot_custom_on_off_values() {
        let mut p = one_hot_params(3);
        p.insert("on_value".to_owned(), "5.0".to_owned());
        p.insert("off_value".to_owned(), "-1.0".to_owned());
        let indices = Value::scalar_i64(1);
        let out = eval_primitive(Primitive::OneHot, &[indices], &p).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![-1.0, 5.0, -1.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn one_hot_missing_num_classes_errors() {
        let indices = Value::scalar_i64(0);
        let result = eval_primitive(Primitive::OneHot, &[indices], &no_params());
        assert!(result.is_err());
    }

    // ── DynamicUpdateSlice tests ──────────────────────────────────────

    #[test]
    fn dynamic_update_slice_1d() {
        // operand: [0, 0, 0, 0, 0], update: [10, 20], start: 2
        // result: [0, 0, 10, 20, 0]
        let operand = Value::vector_i64(&[0, 0, 0, 0, 0]).unwrap();
        let update = Value::vector_i64(&[10, 20]).unwrap();
        let start = Value::scalar_i64(2);
        let out = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, start],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![5]);
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![0, 0, 10, 20, 0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn dynamic_update_slice_2d() {
        // operand: [[0,0,0],[0,0,0]], update: [[7,8]], start: (1,1)
        // result: [[0,0,0],[0,7,8]]
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![Literal::I64(0); 6],
            )
            .unwrap(),
        );
        let update = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![1, 2] },
                vec![Literal::I64(7), Literal::I64(8)],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, Value::scalar_i64(1), Value::scalar_i64(1)],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 3]);
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![0, 0, 0, 0, 7, 8]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn dynamic_update_slice_full_trailing_rows() {
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![4, 3] },
                (0..12).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let update = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(90),
                    Literal::I64(91),
                    Literal::I64(92),
                    Literal::I64(93),
                    Literal::I64(94),
                    Literal::I64(95),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, Value::scalar_i64(1), Value::scalar_i64(0)],
            &no_params(),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![0, 1, 2, 90, 91, 92, 93, 94, 95, 9, 10, 11]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn dynamic_update_slice_empty_huge_trailing_shape_returns_operand() {
        let huge = u32::MAX;
        let operand = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![0, huge, huge, huge, huge],
                },
                Vec::new(),
            )
            .unwrap(),
        );
        let update = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![0, huge, huge, huge, huge],
                },
                Vec::new(),
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[
                operand,
                update,
                Value::scalar_i64(0),
                Value::scalar_i64(0),
                Value::scalar_i64(0),
                Value::scalar_i64(0),
                Value::scalar_i64(0),
            ],
            &no_params(),
        )
        .unwrap();

        let tensor = out.as_tensor().unwrap();
        assert_eq!(tensor.shape.dims, vec![0, huge, huge, huge, huge]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn dynamic_update_slice_clamped_start() {
        // Start index out of range should be clamped
        // operand: [1, 2, 3], update: [99, 88], start: 10 → clamped to 1
        let operand = Value::vector_i64(&[1, 2, 3]).unwrap();
        let update = Value::vector_i64(&[99, 88]).unwrap();
        let start = Value::scalar_i64(10);
        let out = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, start],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 99, 88]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn dynamic_update_slice_negative_start_relative_to_end() {
        let operand = Value::vector_i64(&[1, 2, 3, 4, 5]).unwrap();
        let update = Value::vector_i64(&[99, 88]).unwrap();
        let start = Value::scalar_i64(-3);
        let out = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, start],
            &no_params(),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 2, 99, 88, 5]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn dynamic_update_slice_rejects_float_start() {
        let operand = Value::vector_i64(&[1, 2, 3]).unwrap();
        let update = Value::vector_i64(&[9, 8]).unwrap();
        let result = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, Value::scalar_f64(1.0)],
            &no_params(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn dynamic_update_slice_dtype_mismatch_errors() {
        let operand = Value::vector_i64(&[1, 2, 3]).unwrap();
        let update = Value::vector_f64(&[9.0, 8.0]).unwrap();
        let result = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, Value::scalar_i64(0)],
            &no_params(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn dynamic_update_slice_arity_error() {
        let operand = Value::vector_i64(&[1, 2]).unwrap();
        let result = eval_primitive(Primitive::DynamicUpdateSlice, &[operand], &no_params());
        assert!(result.is_err());
    }

    #[test]
    fn dynamic_update_slice_oversize_update_errors() {
        let operand = Value::vector_i64(&[1, 2, 3]).unwrap();
        let update = Value::vector_i64(&[9, 8, 7, 6]).unwrap();
        let result = eval_primitive(
            Primitive::DynamicUpdateSlice,
            &[operand, update, Value::scalar_i64(0)],
            &no_params(),
        );
        assert!(result.is_err());
    }

    // ── Cumsum / Cumprod tests ────────────────────────────────────────

    fn axis_params(axis: usize) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("axis".to_owned(), axis.to_string());
        p
    }

    fn raw_axis_params(axis: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("axis".to_owned(), axis.to_owned());
        p
    }

    fn reverse_axis_params(axis: usize) -> BTreeMap<String, String> {
        let mut p = axis_params(axis);
        p.insert("reverse".to_owned(), "true".to_owned());
        p
    }

    #[test]
    fn integer_reduce_and_cumulative_wrap_on_overflow() {
        // XLA integer Add/Mul wrap two's-complement; the reduce/scan reducers
        // must too (and must not panic in debug builds). i64::MAX + 1 == i64::MIN.
        let sum = eval_primitive(
            Primitive::ReduceSum,
            &[Value::vector_i64(&[i64::MAX, 1]).unwrap()],
            &no_params(),
        )
        .unwrap();
        assert_eq!(sum.as_i64_scalar().unwrap(), i64::MIN);

        let prod = eval_primitive(
            Primitive::ReduceProd,
            &[Value::vector_i64(&[i64::MAX, 4]).unwrap()],
            &no_params(),
        )
        .unwrap();
        assert_eq!(prod.as_i64_scalar().unwrap(), i64::MAX.wrapping_mul(4));

        let cumsum = eval_primitive(
            Primitive::Cumsum,
            &[Value::vector_i64(&[i64::MAX, 1, 1]).unwrap()],
            &no_params(),
        )
        .unwrap();
        let vals: Vec<i64> = cumsum
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(vals, vec![i64::MAX, i64::MIN, i64::MIN + 1]);
    }

    #[test]
    fn cumsum_1d() {
        // cumsum([1, 2, 3, 4]) = [1, 3, 6, 10]
        let input = Value::vector_i64(&[1, 2, 3, 4]).unwrap();
        let out = eval_primitive(Primitive::Cumsum, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 3, 6, 10]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn cummax_cummin_propagate_nan_like_jax() {
        // jnp.cummax/cummin use the NaN-propagating XLA max/min: once a NaN appears
        // in the prefix it poisons the running extremum. Rust f64::max/min would
        // wrongly drop the NaN.
        let nan_bits = |v: &Value| -> Vec<bool> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().is_nan())
                .collect()
        };
        let cummax = eval_primitive(
            Primitive::Cummax,
            &[Value::vector_f64(&[1.0, f64::NAN, 2.0, 0.5]).unwrap()],
            &no_params(),
        )
        .unwrap();
        assert_eq!(nan_bits(&cummax), vec![false, true, true, true], "cummax");

        let cummin = eval_primitive(
            Primitive::Cummin,
            &[Value::vector_f64(&[3.0, f64::NAN, 1.0, 5.0]).unwrap()],
            &no_params(),
        )
        .unwrap();
        assert_eq!(nan_bits(&cummin), vec![false, true, true, true], "cummin");
    }

    #[test]
    fn cumsum_2d_axis0() {
        // [[1, 2], [3, 4]] cumsum axis=0 → [[1, 2], [4, 6]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Cumsum, &[input], &axis_params(0)).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 2, 4, 6]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn cumsum_2d_axis1() {
        // [[1, 2], [3, 4]] cumsum axis=1 → [[1, 3], [3, 7]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Cumsum, &[input], &axis_params(1)).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 3, 3, 7]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn cumsum_reverse_1d() {
        let input = Value::vector_i64(&[1, 2, 3, 4]).unwrap();
        let out = eval_primitive(Primitive::Cumsum, &[input], &reverse_axis_params(0)).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![10, 9, 7, 4]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn cumsum_empty_vector_negative_axis_returns_empty_tensor() {
        let input =
            Value::Tensor(TensorValue::new(DType::I64, Shape { dims: vec![0] }, vec![]).unwrap());
        let out = eval_primitive(Primitive::Cumsum, &[input], &raw_axis_params("-1")).unwrap();

        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape, Shape { dims: vec![0] });
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn cumprod_empty_selected_axis_returns_empty_tensor() {
        let input = Value::Tensor(
            TensorValue::new(DType::I64, Shape { dims: vec![2, 0] }, vec![]).unwrap(),
        );
        let out = eval_primitive(Primitive::Cumprod, &[input], &axis_params(1)).unwrap();

        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape, Shape { dims: vec![2, 0] });
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn cumprod_1d() {
        // cumprod([1, 2, 3, 4]) = [1, 2, 6, 24]
        let input = Value::vector_i64(&[1, 2, 3, 4]).unwrap();
        let out = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 2, 6, 24]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn cumprod_f64() {
        // cumprod([1.0, 2.0, 3.0]) = [1.0, 2.0, 6.0]
        let input = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let out = eval_primitive(Primitive::Cumprod, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![1.0, 2.0, 6.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn cumprod_reverse_2d_axis1() {
        let input = Value::Tensor(
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
        );
        let out = eval_primitive(Primitive::Cumprod, &[input], &reverse_axis_params(1)).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![6, 6, 3, 120, 30, 6]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    // ── Sort / Argsort tests ──────────────────────────────────────────

    #[test]
    fn sort_1d_ascending() {
        let input = Value::vector_i64(&[3, 1, 4, 1, 5]).unwrap();
        let out = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 1, 3, 4, 5]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn sort_1d_descending() {
        let mut p = BTreeMap::new();
        p.insert("descending".to_owned(), "true".to_owned());
        let input = Value::vector_i64(&[3, 1, 4, 1, 5]).unwrap();
        let out = eval_primitive(Primitive::Sort, &[input], &p).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![5, 4, 3, 1, 1]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn argsort_1d() {
        let input = Value::vector_i64(&[30, 10, 20]).unwrap();
        let out = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 2, 0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn sort_2d_axis1() {
        // [[3, 1], [4, 2]] sorted along axis 1 → [[1, 3], [2, 4]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(3),
                    Literal::I64(1),
                    Literal::I64(4),
                    Literal::I64(2),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Sort, &[input], &axis_params(1)).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(vals, vec![1, 3, 2, 4]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    // ── Conv tests ────────────────────────────────────────────────────

    fn conv_params(padding: &str, strides: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("padding".to_owned(), padding.to_owned());
        p.insert("strides".to_owned(), strides.to_owned());
        p
    }

    fn conv_1d_single_channel_values(lhs_values: &[f64], rhs_values: &[f64]) -> [Value; 2] {
        [
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![1, lhs_values.len() as u32, 1],
                    },
                    lhs_values.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            ),
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![rhs_values.len() as u32, 1, 1],
                    },
                    rhs_values.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            ),
        ]
    }

    fn conv_2d_single_channel_values(side: u32, lhs_values: &[f64]) -> [Value; 2] {
        [
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![1, side, side, 1],
                    },
                    lhs_values.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            ),
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![1, 1, 1, 1],
                    },
                    vec![Literal::from_f64(1.0)],
                )
                .unwrap(),
            ),
        ]
    }

    #[test]
    fn dense_f64_conv_2d_matches_generic() {
        // Dense f64 2D conv (the new fast path) must be bit-identical to the
        // generic Literal path. lhs=[N,H,W,Cin], rhs=[KH,KW,Cin,Cout]; SAME +
        // VALID padding, multi-channel.
        let (n, h, w, cin) = (2usize, 9usize, 7usize, 3usize);
        let (kh, kw, cout) = (3usize, 3usize, 4usize);
        let lhs_data: Vec<f64> = (0..n * h * w * cin)
            .map(|i| ((i as f64) * 0.137).sin() * 3.0 - 1.0)
            .collect();
        let rhs_data: Vec<f64> = (0..kh * kw * cin * cout)
            .map(|i| ((i as f64) * 0.211).cos() * 0.5)
            .collect();
        let mk = |d: &[f64], dims: Vec<u32>, dense: bool| {
            if dense {
                Value::Tensor(TensorValue::new_f64_values(Shape { dims }, d.to_vec()).unwrap())
            } else {
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        Shape { dims },
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
        let ld = vec![n as u32, h as u32, w as u32, cin as u32];
        let rd = vec![kh as u32, kw as u32, cin as u32, cout as u32];
        assert!(
            mk(&lhs_data, ld.clone(), true)
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_some()
        );
        for (pad, stride) in [("valid", "1"), ("same", "1"), ("valid", "2")] {
            let p = conv_params(pad, stride);
            let d = eval_primitive(
                Primitive::Conv,
                &[
                    mk(&lhs_data, ld.clone(), true),
                    mk(&rhs_data, rd.clone(), true),
                ],
                &p,
            )
            .unwrap();
            let l = eval_primitive(
                Primitive::Conv,
                &[
                    mk(&lhs_data, ld.clone(), false),
                    mk(&rhs_data, rd.clone(), false),
                ],
                &p,
            )
            .unwrap();
            assert_eq!(
                d.as_tensor().unwrap().shape.dims,
                l.as_tensor().unwrap().shape.dims,
                "conv2d {pad} {stride} shape"
            );
            assert_eq!(bits(&d), bits(&l), "conv2d {pad} stride={stride}");
        }
    }

    #[test]
    fn dense_f64_conv_1d_matches_generic() {
        // Dense f64 1D conv (the new fast path) must be bit-identical to the
        // generic Literal path. lhs=[N,W,Cin], rhs=[KW,Cin,Cout]; SAME + VALID
        // padding, strides, multi-channel.
        let (n, w, cin) = (2usize, 16usize, 4usize);
        let (kw, cout) = (3usize, 5usize);
        let lhs_data: Vec<f64> = (0..n * w * cin)
            .map(|i| ((i as f64) * 0.151).sin() * 2.0 - 0.5)
            .collect();
        let rhs_data: Vec<f64> = (0..kw * cin * cout)
            .map(|i| ((i as f64) * 0.223).cos() * 0.7)
            .collect();
        let mk = |d: &[f64], dims: Vec<u32>, dense: bool| {
            if dense {
                Value::Tensor(TensorValue::new_f64_values(Shape { dims }, d.to_vec()).unwrap())
            } else {
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        Shape { dims },
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
        let ld = vec![n as u32, w as u32, cin as u32];
        let rd = vec![kw as u32, cin as u32, cout as u32];
        assert!(
            mk(&lhs_data, ld.clone(), true)
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_some()
        );
        for (pad, stride) in [("valid", "1"), ("same", "1"), ("valid", "2")] {
            let p = conv_params(pad, stride);
            let d = eval_primitive(
                Primitive::Conv,
                &[
                    mk(&lhs_data, ld.clone(), true),
                    mk(&rhs_data, rd.clone(), true),
                ],
                &p,
            )
            .unwrap();
            let l = eval_primitive(
                Primitive::Conv,
                &[
                    mk(&lhs_data, ld.clone(), false),
                    mk(&rhs_data, rd.clone(), false),
                ],
                &p,
            )
            .unwrap();
            assert_eq!(
                d.as_tensor().unwrap().shape.dims,
                l.as_tensor().unwrap().shape.dims,
                "conv1d {pad} {stride} shape"
            );
            assert_eq!(bits(&d), bits(&l), "conv1d {pad} stride={stride}");
        }
    }

    #[test]
    fn conv_1d_valid_single_channel() {
        // lhs=[1, 4, 1] (batch=1, width=4, channels=1)
        // rhs=[2, 1, 1] (kernel=2, c_in=1, c_out=1)
        // valid padding, stride=1 → output=[1, 3, 1]
        // input: [1, 2, 3, 4], kernel: [1, 1]
        // out: [1*1+2*1, 2*1+3*1, 3*1+4*1] = [3, 5, 7]
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 4, 1],
                },
                vec![1.0, 2.0, 3.0, 4.0]
                    .into_iter()
                    .map(Literal::from_f64)
                    .collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 1, 1],
                },
                vec![1.0, 1.0].into_iter().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 3, 1]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 5.0, 7.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn conv_1d_valid_rejects_zero_stride() {
        let inputs = conv_1d_single_channel_values(&[1.0, 2.0, 3.0], &[1.0]);
        let result = eval_primitive(Primitive::Conv, &inputs, &conv_params("valid", "0"));

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("conv stride must be positive"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn conv_1d_same_rejects_zero_stride() {
        let inputs = conv_1d_single_channel_values(&[1.0, 2.0, 3.0], &[1.0]);
        let result = eval_primitive(Primitive::Conv, &inputs, &conv_params("same", "0"));

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("conv stride must be positive"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn conv_1d_preserves_f32_dtype() {
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: vec![1, 3, 1],
                },
                vec![1.0, 2.0, 3.0]
                    .into_iter()
                    .map(Literal::from_f64)
                    .collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: vec![2, 1, 1],
                },
                vec![1.0, 1.0].into_iter().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::F32);
            assert_eq!(t.shape.dims, vec![1, 2, 1]);
            t.validate_dtype_consistency()
                .expect("conv F32 output dtype/element invariant");
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn conv_1d_preserves_bf16_literal_dtype() {
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::BF16,
                Shape {
                    dims: vec![1, 3, 1],
                },
                vec![
                    Literal::from_bf16_f32(1.0),
                    Literal::from_bf16_f32(2.0),
                    Literal::from_bf16_f32(3.0),
                ],
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::BF16,
                Shape {
                    dims: vec![2, 1, 1],
                },
                vec![Literal::from_bf16_f32(1.0), Literal::from_bf16_f32(1.0)],
            )
            .unwrap(),
        );

        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::BF16);
            assert_eq!(t.shape.dims, vec![1, 2, 1]);
            t.validate_dtype_consistency()
                .expect("conv BF16 output dtype/element invariant");
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn conv_1d_same_padding() {
        // lhs=[1, 3, 1], rhs=[3, 1, 1], same padding, stride=1
        // input: [1, 2, 3], kernel: [1, 1, 1]
        // same → output width=3
        // pad_left=1: padded=[0, 1, 2, 3, 0]
        // out: [0+1+2, 1+2+3, 2+3+0] = [3, 6, 5]
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 1],
                },
                vec![1.0, 2.0, 3.0]
                    .into_iter()
                    .map(Literal::from_f64)
                    .collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![3, 1, 1],
                },
                vec![1.0, 1.0, 1.0]
                    .into_iter()
                    .map(Literal::from_f64)
                    .collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("same", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 3, 1]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 6.0, 5.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn conv_1d_uppercase_same_padding() {
        let inputs = conv_1d_single_channel_values(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0]);
        let out = eval_primitive(Primitive::Conv, &inputs, &conv_params("SAME", "1")).unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 4, 1]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 5.0, 7.0, 4.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn conv_1d_same_lower_padding_puts_extra_pad_low() {
        let inputs = conv_1d_single_channel_values(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0]);
        let out =
            eval_primitive(Primitive::Conv, &inputs, &conv_params("SAME_LOWER", "1")).unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 4, 1]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![1.0, 3.0, 5.0, 7.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn conv_rejects_unknown_padding() {
        let inputs = conv_1d_single_channel_values(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0]);
        let result = eval_primitive(Primitive::Conv, &inputs, &conv_params("mirror", "1"));

        let err = result.expect_err("unknown padding should fail").to_string();
        assert!(
            err.contains("unsupported conv padding mode"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn conv_1d_stride2() {
        // lhs=[1, 6, 1], rhs=[2, 1, 1], valid, stride=2
        // input: [1,2,3,4,5,6], kernel: [1,1]
        // output width = (6-2)/2+1 = 3
        // positions: 0,2,4 → [1+2, 3+4, 5+6] = [3, 7, 11]
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 6, 1],
                },
                (1..=6).map(|i| Literal::from_f64(i as f64)).collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 1, 1],
                },
                vec![Literal::from_f64(1.0), Literal::from_f64(1.0)],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "2")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 3, 1]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 7.0, 11.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn conv_1d_valid_kernel_larger_than_input_returns_empty() {
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 1, 1],
                },
                vec![Literal::from_f64(2.0)],
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 1, 1],
                },
                vec![Literal::from_f64(1.0), Literal::from_f64(1.0)],
            )
            .unwrap(),
        );

        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::F64);
            assert_eq!(t.shape.dims, vec![1, 0, 1]);
            assert!(t.elements.is_empty());
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    // ── Conv 2D tests ────────────────────────────────────────────

    #[test]
    fn conv_2d_valid_single_channel() {
        // lhs=[1, 3, 3, 1], rhs=[2, 2, 1, 1], valid, stride=1
        // Input 3x3 image:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        // Kernel 2x2 (all ones):
        // 1 1
        // 1 1
        // Output 2x2: [1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9] = [12, 16, 24, 28]
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 3, 1],
                },
                (1..=9).map(|i| Literal::from_f64(i as f64)).collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 2, 1, 1],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 2, 2, 1]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![12.0, 16.0, 24.0, 28.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn conv_2d_valid_kernel_larger_than_height_returns_empty() {
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 1, 3, 1],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 1, 1, 1],
                },
                vec![Literal::from_f64(1.0), Literal::from_f64(1.0)],
            )
            .unwrap(),
        );

        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::F64);
            assert_eq!(t.shape.dims, vec![1, 0, 3, 1]);
            assert!(t.elements.is_empty());
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn conv_2d_rejects_zero_stride() {
        let inputs =
            conv_2d_single_channel_values(3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = eval_primitive(Primitive::Conv, &inputs, &conv_params("valid", "0"));

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("conv stride must be positive"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn conv_2d_same_padding() {
        // lhs=[1, 3, 3, 1], rhs=[3, 3, 1, 1], same padding
        // With same padding, output should have same spatial dims as input: 3x3
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 3, 1],
                },
                (1..=9).map(|i| Literal::from_f64(i as f64)).collect(),
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![3, 3, 1, 1],
                },
                vec![Literal::from_f64(1.0); 9],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("same", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 3, 3, 1]);
            // Center element: sum of all 9 values = 45
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert!((vals[4] - 45.0).abs() < 1e-10, "center = {}", vals[4]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn conv_2d_multi_channel() {
        // lhs=[1, 2, 2, 2] (2x2 image, 2 channels)
        // rhs=[1, 1, 2, 3] (1x1 kernel, 2 c_in, 3 c_out) -- pointwise conv
        // This is effectively a dense transform of each pixel's channel vector
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 2, 2, 2],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0), // pixel (0,0): [1,2]
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0), // pixel (0,1): [3,4]
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0), // pixel (1,0): [5,6]
                    Literal::from_f64(7.0),
                    Literal::from_f64(8.0), // pixel (1,1): [7,8]
                ],
            )
            .unwrap(),
        );
        // kernel: 1x1, c_in=2, c_out=3
        // rhs layout: [KH=1, KW=1, C_in=2, C_out=3]
        // W = [[1,0,1], [0,1,1]] -> output channels: ch0=ci0, ch1=ci1, ch2=ci0+ci1
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 1, 2, 3],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(1.0), // ci=0: [1,0,1]
                    Literal::from_f64(0.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0), // ci=1: [0,1,1]
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Conv, &[lhs, rhs], &conv_params("valid", "1")).unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![1, 2, 2, 3]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            // pixel (0,0): [1,2] -> [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
            assert_eq!(vals[0..3], [1.0, 2.0, 3.0]);
            // pixel (0,1): [3,4] -> [3, 4, 7]
            assert_eq!(vals[3..6], [3.0, 4.0, 7.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn cond_true_returns_true_branch() {
        let pred = Value::scalar_bool(true);
        let true_val = Value::scalar_f64(42.0);
        let false_val = Value::scalar_f64(99.0);
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 42.0);
    }

    #[test]
    fn cond_false_returns_false_branch() {
        let pred = Value::scalar_bool(false);
        let true_val = Value::scalar_f64(42.0);
        let false_val = Value::scalar_f64(99.0);
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 99.0);
    }

    #[test]
    fn cond_with_tensor_branches() {
        let pred = Value::scalar_bool(true);
        let true_val = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let false_val = Value::vector_f64(&[4.0, 5.0, 6.0]).unwrap();
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![1.0, 2.0, 3.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn cond_branch_dtype_mismatch_errors() {
        let pred = Value::scalar_bool(true);
        let true_val = Value::scalar_f64(1.0);
        let false_val = Value::scalar_i64(2);
        let params = BTreeMap::new();
        let result = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params);
        assert!(matches!(
            result,
            Err(EvalError::TypeMismatch {
                primitive: Primitive::Cond,
                ..
            })
        ));
    }

    #[test]
    fn cond_branch_shape_mismatch_errors() {
        let pred = Value::scalar_bool(true);
        let true_val = Value::vector_f64(&[1.0, 2.0]).unwrap();
        let false_val = Value::vector_f64(&[1.0]).unwrap();
        let params = BTreeMap::new();
        let result = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params);
        assert!(matches!(
            result,
            Err(EvalError::ShapeMismatch {
                primitive: Primitive::Cond,
                ..
            })
        ));
    }

    #[test]
    fn cond_i64_pred_nonzero_is_true() {
        let pred = Value::scalar_i64(1);
        let true_val = Value::scalar_f64(10.0);
        let false_val = Value::scalar_f64(20.0);
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 10.0);
    }

    #[test]
    fn cond_f16_predicate_zero_is_false() {
        let pred = Value::scalar_f16(0.0);
        let true_val = Value::scalar_f64(1.0);
        let false_val = Value::scalar_f64(2.0);
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 2.0);
    }

    #[test]
    fn cond_bf16_predicate_nonzero_is_true() {
        let pred = Value::scalar_bf16(1.0);
        let true_val = Value::scalar_f64(3.0);
        let false_val = Value::scalar_f64(4.0);
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 3.0);
    }

    #[test]
    fn cond_tensor_bool_predicate_selects_true_branch() {
        let pred = Value::Tensor(
            TensorValue::new(DType::Bool, Shape::scalar(), vec![Literal::Bool(true)]).unwrap(),
        );
        let true_val = Value::scalar_f64(5.0);
        let false_val = Value::scalar_f64(6.0);
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Cond, &[pred, true_val, false_val], &params).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 5.0);
    }

    #[test]
    fn cond_arity_error() {
        let params = BTreeMap::new();
        let result = eval_primitive(Primitive::Cond, &[Value::scalar_bool(true)], &params);
        assert!(result.is_err());
    }

    // ── Scan tests ──────────────────────────────────────────────────

    fn scan_params(body_op: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("body_op".to_owned(), body_op.to_owned());
        p
    }

    fn scan_params_reverse(body_op: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("body_op".to_owned(), body_op.to_owned());
        p.insert("reverse".to_owned(), "true".to_owned());
        p
    }

    #[test]
    fn scan_add_vector() {
        // scan(add, 0.0, [1,2,3,4]) => 0+1+2+3+4 = 10
        let init = Value::scalar_f64(0.0);
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("add")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 10.0);
    }

    #[test]
    fn scan_mul_vector() {
        // scan(mul, 1.0, [2,3,4]) => 1*2*3*4 = 24
        let init = Value::scalar_f64(1.0);
        let xs = Value::vector_f64(&[2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("mul")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 24.0);
    }

    #[test]
    fn scan_div_vector() {
        // scan(div, 100.0, [2,5]) => (100/2)/5 = 10
        let init = Value::scalar_f64(100.0);
        let xs = Value::vector_f64(&[2.0, 5.0]).unwrap();
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("div")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 10.0);
    }

    #[test]
    fn scan_pow_vector() {
        // scan(pow, 2.0, [3,2]) => (2^3)^2 = 64
        let init = Value::scalar_f64(2.0);
        let xs = Value::vector_f64(&[3.0, 2.0]).unwrap();
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("pow")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 64.0);
    }

    #[test]
    fn scan_add_reverse() {
        // scan(add, 0.0, [1,2,3], reverse=true) => 0+3+2+1 = 6 (same as forward for add)
        let init = Value::scalar_f64(0.0);
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let out =
            eval_primitive(Primitive::Scan, &[init, xs], &scan_params_reverse("add")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 6.0);
    }

    #[test]
    fn scan_sub_reverse() {
        // scan(sub, 10.0, [1,2,3], reverse=true) => ((10-3)-2)-1 = 4
        let init = Value::scalar_f64(10.0);
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let out =
            eval_primitive(Primitive::Scan, &[init, xs], &scan_params_reverse("sub")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 4.0);
    }

    #[test]
    fn scan_max_vector() {
        // scan(max, -inf, [3,1,4,1,5]) => 5
        let init = Value::scalar_f64(f64::NEG_INFINITY);
        let xs = Value::vector_f64(&[3.0, 1.0, 4.0, 1.0, 5.0]).unwrap();
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("max")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 5.0);
    }

    #[test]
    fn scan_with_tensor_slices() {
        // xs shape [2, 3]: scan(add, [0,0,0], [[1,2,3],[4,5,6]]) => [5,7,9]
        let init = Value::vector_f64(&[0.0, 0.0, 0.0]).unwrap();
        let xs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("add")).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![5.0, 7.0, 9.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor output");
        }
    }

    #[test]
    fn scan_arity_error() {
        let result = eval_primitive(Primitive::Scan, &[Value::scalar_f64(0.0)], &no_params());
        assert!(result.is_err());
    }

    #[test]
    fn scan_empty_tensor() {
        // Scan over empty leading axis returns init_carry unchanged
        let init = Value::scalar_f64(42.0);
        let xs =
            Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap());
        let out = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("add")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 42.0);
    }

    #[test]
    fn scan_scalar_tensor_xs_returns_error() {
        let init = Value::scalar_f64(0.0);
        let xs = Value::Tensor(
            TensorValue::new(DType::F64, Shape::scalar(), vec![Literal::from_f64(1.0)]).unwrap(),
        );

        let err = eval_primitive(Primitive::Scan, &[init, xs], &scan_params("add")).unwrap_err();
        assert_eq!(
            err,
            EvalError::TypeMismatch {
                primitive: Primitive::Scan,
                detail: "scan tensor xs must have a leading axis"
            }
        );
    }

    // ── Associative scan tests ──────────────────────────────

    fn assoc_scan_params(op: &str) -> BTreeMap<String, String> {
        let mut params = BTreeMap::new();
        params.insert("body_op".to_owned(), op.to_owned());
        params
    }

    fn assoc_scan_params_reverse(op: &str) -> BTreeMap<String, String> {
        let mut params = BTreeMap::new();
        params.insert("body_op".to_owned(), op.to_owned());
        params.insert("reverse".to_owned(), "true".to_owned());
        params
    }

    #[test]
    fn associative_scan_add_prefix_sum() {
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out =
            eval_primitive(Primitive::AssociativeScan, &[xs], &assoc_scan_params("add")).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![1.0, 3.0, 6.0, 10.0]);
        } else {
            panic!("expected tensor output");
        }
    }

    #[test]
    fn associative_scan_mul_prefix_prod() {
        let xs = Value::vector_f64(&[2.0, 3.0, 4.0]).unwrap();
        let out =
            eval_primitive(Primitive::AssociativeScan, &[xs], &assoc_scan_params("mul")).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![2.0, 6.0, 24.0]);
        } else {
            panic!("expected tensor output");
        }
    }

    #[test]
    fn associative_scan_max_running_max() {
        let xs = Value::vector_f64(&[3.0, 1.0, 4.0, 1.0, 5.0]).unwrap();
        let out =
            eval_primitive(Primitive::AssociativeScan, &[xs], &assoc_scan_params("max")).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 3.0, 4.0, 4.0, 5.0]);
        } else {
            panic!("expected tensor output");
        }
    }

    #[test]
    fn associative_scan_reverse() {
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(
            Primitive::AssociativeScan,
            &[xs],
            &assoc_scan_params_reverse("add"),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![10.0, 9.0, 7.0, 4.0]);
        } else {
            panic!("expected tensor output");
        }
    }

    #[test]
    fn associative_scan_single_element() {
        let xs = Value::vector_f64(&[42.0]).unwrap();
        let out =
            eval_primitive(Primitive::AssociativeScan, &[xs], &assoc_scan_params("add")).unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![42.0]);
        } else {
            panic!("expected tensor output");
        }
    }

    #[test]
    fn associative_scan_scalar() {
        let xs = Value::scalar_f64(7.0);
        let out =
            eval_primitive(Primitive::AssociativeScan, &[xs], &assoc_scan_params("add")).unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 7.0);
    }

    #[test]
    fn associative_scan_dense_f64_bit_identical_to_slice_dispatch() {
        // The dense f64 fast path must be bit-for-bit identical to the generic
        // per-slice `eval_primitive` dispatch it replaces — including operand order
        // (matters for order-sensitive max/min on signed zero) and special values
        // (±0, NaN, ±inf). Cover add/mul/max/min x forward/reverse over a 2-D
        // [L, D] tensor (inner = D > 1) so the inner-stride indexing is exercised.
        let l = 6usize;
        let d = 4usize;
        let raw: Vec<f64> = vec![
            1.0,
            -0.0,
            f64::NAN,
            -3.25,
            f64::INFINITY,
            0.0,
            2.5,
            -1.0,
            f64::NEG_INFINITY,
            7.0,
            -0.0,
            4.0,
            0.5,
            -2.0,
            3.0,
            f64::NAN,
            -8.0,
            6.0,
            -6.0,
            1.5,
            9.0,
            -9.0,
            0.25,
            -0.25,
        ];
        let raw: Vec<f64> = raw.into_iter().take(l * d).collect();
        let tensor = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![l as u32, d as u32],
                },
                raw.clone(),
            )
            .unwrap(),
        );

        // Reference: the exact generic slice-dispatch algorithm (pre-fast-path).
        let reference = |body_op: Primitive, reverse: bool| -> Vec<u64> {
            let t = tensor.as_tensor().unwrap();
            let mut results: Vec<Value> = Vec::new();
            if reverse {
                let mut acc = t.slice_axis0(l - 1).unwrap();
                results.push(acc.clone());
                for i in (0..l - 1).rev() {
                    let x = t.slice_axis0(i).unwrap();
                    acc = eval_primitive(body_op, &[x, acc], &BTreeMap::new()).unwrap();
                    results.push(acc.clone());
                }
                results.reverse();
            } else {
                let mut acc = t.slice_axis0(0).unwrap();
                results.push(acc.clone());
                for i in 1..l {
                    let x = t.slice_axis0(i).unwrap();
                    acc = eval_primitive(body_op, &[acc, x], &BTreeMap::new()).unwrap();
                    results.push(acc.clone());
                }
            }
            let stacked = TensorValue::stack_axis0(&results).unwrap();
            stacked
                .elements
                .iter()
                .map(|x| x.as_f64().unwrap().to_bits())
                .collect()
        };

        for (name, prim) in [
            ("add", Primitive::Add),
            ("mul", Primitive::Mul),
            ("max", Primitive::Max),
            ("min", Primitive::Min),
        ] {
            for reverse in [false, true] {
                let params = if reverse {
                    assoc_scan_params_reverse(name)
                } else {
                    assoc_scan_params(name)
                };
                let out = eval_primitive(
                    Primitive::AssociativeScan,
                    std::slice::from_ref(&tensor),
                    &params,
                )
                .unwrap();
                let got: Vec<u64> = out
                    .as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|x| x.as_f64().unwrap().to_bits())
                    .collect();
                assert_eq!(
                    got,
                    reference(prim, reverse),
                    "associative_scan {name} reverse={reverse}: fast path != slice-dispatch"
                );
            }
        }
    }

    #[test]
    fn associative_scan_dense_f32_bit_identical_to_slice_dispatch() {
        // f32 (JAX's default float): the fast path mirrors eval_same_shape_f32_map
        // (op in f64, narrow to f32). Must equal the per-slice dispatch bit-for-bit
        // across add/mul/max/min x forward/reverse with NaN/±0/±inf.
        let (l, d) = (6usize, 4usize);
        let raw: Vec<f32> = vec![
            1.0,
            -0.0,
            f32::NAN,
            -3.25,
            f32::INFINITY,
            0.0,
            2.5,
            -1.0,
            f32::NEG_INFINITY,
            7.0,
            -0.0,
            4.0,
            0.5,
            -2.0,
            3.0,
            f32::NAN,
            -8.0,
            6.0,
            -6.0,
            1.5,
            9.0,
            -9.0,
            0.25,
            -0.25,
        ];
        let tensor = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![l as u32, d as u32],
                },
                raw,
            )
            .unwrap(),
        );
        let f32_bits = |value: &Value| -> Vec<u32> {
            value
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|x| match x {
                    Literal::F32Bits(bits) => *bits,
                    other => {
                        panic!("expected F32Bits in f32 associative_scan proof, got {other:?}")
                    }
                })
                .collect()
        };
        let reference = |body_op: Primitive, reverse: bool| -> Vec<u32> {
            let t = tensor.as_tensor().unwrap();
            let mut results: Vec<Value> = Vec::new();
            if reverse {
                let mut acc = t.slice_axis0(l - 1).unwrap();
                results.push(acc.clone());
                for i in (0..l - 1).rev() {
                    let x = t.slice_axis0(i).unwrap();
                    acc = eval_primitive(body_op, &[x, acc], &BTreeMap::new()).unwrap();
                    results.push(acc.clone());
                }
                results.reverse();
            } else {
                let mut acc = t.slice_axis0(0).unwrap();
                results.push(acc.clone());
                for i in 1..l {
                    let x = t.slice_axis0(i).unwrap();
                    acc = eval_primitive(body_op, &[acc, x], &BTreeMap::new()).unwrap();
                    results.push(acc.clone());
                }
            }
            f32_bits(&Value::Tensor(TensorValue::stack_axis0(&results).unwrap()))
        };
        let mut fixtures = Vec::new();
        for (name, prim) in [
            ("add", Primitive::Add),
            ("mul", Primitive::Mul),
            ("max", Primitive::Max),
            ("min", Primitive::Min),
        ] {
            for reverse in [false, true] {
                let params = if reverse {
                    assoc_scan_params_reverse(name)
                } else {
                    assoc_scan_params(name)
                };
                let out = eval_primitive(
                    Primitive::AssociativeScan,
                    std::slice::from_ref(&tensor),
                    &params,
                )
                .unwrap();
                let got = f32_bits(&out);
                assert_eq!(
                    got,
                    reference(prim, reverse),
                    "f32 associative_scan {name} reverse={reverse}: fast path != slice-dispatch"
                );
                fixtures.push((name.to_owned(), reverse, got));
            }
        }
        let digest =
            fj_test_utils::fixture_id_from_json(&fixtures).expect("f32 associative_scan digest");
        eprintln!("f32 associative_scan golden digest: {digest}");
        assert_eq!(
            digest, "922b9e035eec116a5dc10da15bcc68dc45ebacdddb3b1f4a95908e6621966f80",
            "f32 associative_scan golden output digest changed"
        );
    }

    #[test]
    fn associative_scan_dense_integer_bit_identical_to_slice_dispatch() {
        // Integer dense scan must mirror the old per-slice dispatcher exactly:
        // i64 accumulates in wrapping two's-complement i64, while i32 narrows
        // after each dispatcher step through the public narrow_i32 chokepoint.
        let (l, d) = (6usize, 4usize);
        let raw_i64: Vec<i64> = vec![
            1,
            i64::MAX,
            i64::MIN,
            -3,
            2_654_435_761,
            -9,
            17,
            i64::MAX - 5,
            -11,
            0x0f0f_0f0f_0f0f_0f0f,
            -0x0101_0101_0101_0101,
            4,
            i64::MIN + 7,
            13,
            -21,
            0x5555_5555_5555_5555,
            -8,
            6,
            -6,
            1_234_567_890_123_456_789,
            9,
            -9,
            0x3333_3333_3333_3333,
            -0x2222_2222_2222_2222,
        ];
        let raw_i32: Vec<i64> = vec![
            i64::from(i32::MAX),
            1,
            -1,
            i64::from(i32::MIN),
            65_537,
            -65_539,
            12_345_678,
            -98_765_432,
            0x7fff_0001,
            0x0000_ffff,
            -0x0001_0000,
            17,
            -3,
            5,
            i64::from(i32::MAX - 7),
            -11,
            0x5555_5555,
            0x3333_3333,
            -0x2222_2222,
            9,
            -9,
            0,
            42,
            -42,
        ];

        let tensor_i64 = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![l as u32, d as u32],
                },
                raw_i64,
            )
            .unwrap(),
        );
        let tensor_i32 = Value::Tensor(
            TensorValue::new_i32_values(
                Shape {
                    dims: vec![l as u32, d as u32],
                },
                raw_i32,
            )
            .unwrap(),
        );
        let bits = |value: &Value| -> Vec<i64> {
            value
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|x| match x {
                    Literal::I64(v) => *v,
                    other => panic!("expected integer literal, got {other:?}"),
                })
                .collect()
        };
        let reference = |tensor: &Value, body_op: Primitive, reverse: bool| -> Vec<i64> {
            let t = tensor.as_tensor().unwrap();
            let mut results: Vec<Value> = Vec::new();
            if reverse {
                let mut acc = t.slice_axis0(l - 1).unwrap();
                results.push(acc.clone());
                for i in (0..l - 1).rev() {
                    let x = t.slice_axis0(i).unwrap();
                    acc = eval_primitive(body_op, &[x, acc], &BTreeMap::new()).unwrap();
                    results.push(acc.clone());
                }
                results.reverse();
            } else {
                let mut acc = t.slice_axis0(0).unwrap();
                results.push(acc.clone());
                for i in 1..l {
                    let x = t.slice_axis0(i).unwrap();
                    acc = eval_primitive(body_op, &[acc, x], &BTreeMap::new()).unwrap();
                    results.push(acc.clone());
                }
            }
            bits(&Value::Tensor(TensorValue::stack_axis0(&results).unwrap()))
        };

        let mut fixtures = Vec::new();
        for (name, prim) in [
            ("add", Primitive::Add),
            ("mul", Primitive::Mul),
            ("max", Primitive::Max),
            ("min", Primitive::Min),
            ("and", Primitive::BitwiseAnd),
            ("or", Primitive::BitwiseOr),
            ("xor", Primitive::BitwiseXor),
        ] {
            for (dtype_name, tensor) in [("i64", &tensor_i64), ("i32", &tensor_i32)] {
                if dtype_name == "i32"
                    && matches!(
                        prim,
                        Primitive::BitwiseAnd | Primitive::BitwiseOr | Primitive::BitwiseXor
                    )
                {
                    continue;
                }
                for reverse in [false, true] {
                    let params = if reverse {
                        assoc_scan_params_reverse(name)
                    } else {
                        assoc_scan_params(name)
                    };
                    let out = eval_primitive(
                        Primitive::AssociativeScan,
                        std::slice::from_ref(tensor),
                        &params,
                    )
                    .unwrap();
                    let got = bits(&out);
                    assert_eq!(
                        got,
                        reference(tensor, prim, reverse),
                        "{dtype_name} associative_scan {name} reverse={reverse}: fast path != slice-dispatch"
                    );
                    fixtures.push((dtype_name.to_owned(), name.to_owned(), reverse, got));
                }
            }
        }
        let digest = fj_test_utils::fixture_id_from_json(&fixtures)
            .expect("integer associative_scan digest");
        eprintln!("integer associative_scan golden digest: {digest}");
        assert_eq!(
            digest, "83ffbd381da3466a8a439360580be405087f409ceb95af001cc3488889aface3",
            "integer associative_scan golden output digest changed"
        );
    }

    #[test]
    fn associative_scan_i32_rank1_wraps_mod_2e32() {
        // Rank-1 (inner==1) i32 associative_scan: the dense integer fast path
        // (convergent commit 9933139b) routes i32 add/mul/max/min through
        // `i64::from(op(a, b) as i32)`, narrowing every prefix step mod-2^32 —
        // the JAX-correct i32 contract. This INTENTIONALLY differs from the
        // generic per-slice dispatch path, whose rank-1 i32 slices are
        // dtype-less `Literal::I64` scalars that wrap mod-2^64 (the bug tracked
        // by frankenjax-b6w3l). The dense path is now the production path for
        // these ops, so this test pins mod-2^32 to guard against a silent
        // regression back to mod-2^64 scalar semantics. The existing
        // bit-identical golden only exercises inner>1 (where slices are I32
        // tensors that the dispatcher already narrows), so rank-1 was untested.
        let raw: Vec<i64> = vec![i64::from(i32::MAX), 1, 1, 1];
        let tensor = Value::Tensor(
            TensorValue::new_i32_values(Shape { dims: vec![4] }, raw.clone()).unwrap(),
        );
        let out = eval_primitive(
            Primitive::AssociativeScan,
            std::slice::from_ref(&tensor),
            &assoc_scan_params("add"),
        )
        .unwrap();
        let got: Vec<i64> = out
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::I64(v) => *v,
                other => panic!("expected I64, got {other:?}"),
            })
            .collect();

        // Independent mod-2^32 prefix-sum reference (plain i32 wrapping add).
        let mut expected = Vec::with_capacity(raw.len());
        let mut acc = raw[0] as i32;
        expected.push(i64::from(acc));
        for &x in &raw[1..] {
            acc = acc.wrapping_add(x as i32);
            expected.push(i64::from(acc));
        }
        assert_eq!(
            got, expected,
            "rank-1 i32 scan must wrap mod-2^32 (JAX semantics)"
        );

        // Index 1 is exactly where mod-2^32 diverges from the legacy mod-2^64
        // generic path: i32::MAX + 1 wraps to i32::MIN, not 2^31.
        assert_eq!(got[1], i64::from(i32::MIN));
        assert_ne!(got[1], i64::from(i32::MAX) + 1);
        // Output dtype stays I32.
        assert_eq!(out.as_tensor().unwrap().dtype, fj_core::DType::I32);
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_associative_scan_dense_vs_slice_dispatch() {
        use std::time::Instant;
        // Dense f64 fast path vs the generic per-slice eval_primitive dispatch loop
        // (the prior behaviour, replicated here). Same scan; isolates the
        // dispatch/alloc elimination.
        // 1-D scan (inner == 1): the common lax.associative_scan over a sequence,
        // where the per-step slice/clone/dispatch is almost pure overhead.
        let l = 1usize << 20;
        let raw: Vec<f64> = (0..l).map(|i| ((i % 1000) as f64) * 0.001 - 0.5).collect();
        let tensor = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![l as u32],
                },
                raw,
            )
            .unwrap(),
        );
        let params = assoc_scan_params("add");
        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut b = f64::MAX;
            let mut dg = 0u64;
            for _ in 0..5 {
                let t = Instant::now();
                dg = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, dg)
        };
        let t_ref = tensor.clone();
        let (t_slice, d_slice) = best(Box::new(move || {
            // Generic slice-dispatch reference (pre-fast-path algorithm).
            let t = t_ref.as_tensor().unwrap();
            let mut results: Vec<Value> = Vec::with_capacity(l);
            let mut acc = t.slice_axis0(0).unwrap();
            results.push(acc.clone());
            for i in 1..l {
                let x = t.slice_axis0(i).unwrap();
                acc = eval_primitive(Primitive::Add, &[acc, x], &BTreeMap::new()).unwrap();
                results.push(acc.clone());
            }
            let stacked = TensorValue::stack_axis0(&results).unwrap();
            stacked
                .elements
                .iter()
                .fold(0u64, |a, x| a ^ x.as_f64().unwrap().to_bits())
        }));
        let t_fast = tensor.clone();
        let (t_dense, d_dense) = best(Box::new(move || {
            let out = eval_primitive(
                Primitive::AssociativeScan,
                std::slice::from_ref(&t_fast),
                &params,
            )
            .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |a, x| a ^ x.as_f64().unwrap().to_bits())
        }));
        assert_eq!(
            d_slice, d_dense,
            "fast-path digest must match slice-dispatch"
        );
        println!(
            "BENCH associative_scan f64 add(x[{l}],axis=0): slice-dispatch={:.4}ms dense={:.4}ms speedup={:.2}x",
            t_slice * 1e3,
            t_dense * 1e3,
            t_slice / t_dense,
        );

        let raw: Vec<f32> = (0..l).map(|i| ((i % 1000) as f32) * 0.001 - 0.5).collect();
        let tensor = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![l as u32],
                },
                raw,
            )
            .unwrap(),
        );
        let params = assoc_scan_params("add");
        let raw_f32_digest = |v: &Value| -> u64 {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |a, x| match x {
                    Literal::F32Bits(bits) => a ^ u64::from(*bits),
                    other => panic!("expected F32Bits, got {other:?}"),
                })
        };
        let t_ref = tensor.clone();
        let (t_slice, d_slice) = best(Box::new(move || {
            let t = t_ref.as_tensor().unwrap();
            let mut results: Vec<Value> = Vec::with_capacity(l);
            let mut acc = t.slice_axis0(0).unwrap();
            results.push(acc.clone());
            for i in 1..l {
                let x = t.slice_axis0(i).unwrap();
                acc = eval_primitive(Primitive::Add, &[acc, x], &BTreeMap::new()).unwrap();
                results.push(acc.clone());
            }
            let stacked = Value::Tensor(TensorValue::stack_axis0(&results).unwrap());
            raw_f32_digest(&stacked)
        }));
        let t_fast = tensor.clone();
        let (t_dense, d_dense) = best(Box::new(move || {
            let out = eval_primitive(
                Primitive::AssociativeScan,
                std::slice::from_ref(&t_fast),
                &params,
            )
            .unwrap();
            raw_f32_digest(&out)
        }));
        assert_eq!(
            d_slice, d_dense,
            "f32 fast-path digest must match slice-dispatch"
        );
        println!(
            "BENCH associative_scan f32 add(x[{l}],axis=0): slice-dispatch={:.4}ms dense={:.4}ms speedup={:.2}x",
            t_slice * 1e3,
            t_dense * 1e3,
            t_slice / t_dense,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_associative_scan_dense_integer_vs_slice_dispatch() {
        use std::time::Instant;

        let l = 1usize << 20;
        let mk_i64 = |i: usize| {
            (i as i64)
                .wrapping_mul(2_654_435_761)
                .wrapping_sub(0x4000_0000_0000 * ((i & 1) as i64))
        };
        let raw_i64: Vec<i64> = (0..l).map(mk_i64).collect();
        let raw_i32: Vec<i64> = (0..l)
            .map(|i| i64::from((mk_i64(i) as i32).wrapping_add(17)))
            .collect();
        let tensor_i64 = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![l as u32],
                },
                raw_i64,
            )
            .unwrap(),
        );
        let tensor_i32 = Value::Tensor(
            TensorValue::new_i32_values(
                Shape {
                    dims: vec![l as u32],
                },
                raw_i32,
            )
            .unwrap(),
        );
        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut best = f64::MAX;
            let mut digest = 0u64;
            for _ in 0..3 {
                let t = Instant::now();
                digest = std::hint::black_box(f());
                best = best.min(t.elapsed().as_secs_f64());
            }
            (best, digest)
        };
        let digest_i64 = |value: &Value| -> u64 {
            value
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |acc, x| acc ^ (x.as_i64().unwrap() as u64))
        };
        let run_slice = |tensor: &Value, body_op: Primitive| -> u64 {
            let t = tensor.as_tensor().unwrap();
            let mut results: Vec<Value> = Vec::with_capacity(l);
            let mut acc = t.slice_axis0(0).unwrap();
            results.push(acc.clone());
            for i in 1..l {
                let x = t.slice_axis0(i).unwrap();
                acc = eval_primitive(body_op, &[acc, x], &BTreeMap::new()).unwrap();
                results.push(acc.clone());
            }
            digest_i64(&Value::Tensor(TensorValue::stack_axis0(&results).unwrap()))
        };
        let run_assoc = |tensor: &Value, name: &str| -> u64 {
            let params = assoc_scan_params(name);
            let out = eval_primitive(
                Primitive::AssociativeScan,
                std::slice::from_ref(tensor),
                &params,
            )
            .unwrap();
            digest_i64(&out)
        };

        for (label, tensor) in [("i64", tensor_i64), ("i32", tensor_i32)] {
            let t_ref = tensor.clone();
            let (t_slice, d_slice) = best(Box::new(move || run_slice(&t_ref, Primitive::Add)));
            let t_fast = tensor.clone();
            let (t_assoc, d_assoc) = best(Box::new(move || run_assoc(&t_fast, "add")));
            assert_eq!(
                d_slice, d_assoc,
                "{label} fast-path digest must match slice-dispatch"
            );
            println!(
                "BENCH associative_scan {label} add(x[{l}],axis=0): slice-dispatch={:.4}ms assoc-eval={:.4}ms speedup={:.2}x digest={d_assoc:016x}",
                t_slice * 1e3,
                t_assoc * 1e3,
                t_slice / t_assoc,
            );
        }
    }

    // ── Scan functional tests (bd-3eyv) ──────────────────────────────

    #[test]
    fn test_scan_accumulate_sum() {
        // scan(add, init=0, xs=[1,2,3,4]) → carry=10, ys=[1,3,6,10]
        let init = vec![Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 10.0);
        assert_eq!(ys.len(), 1);
        let ys_tensor = ys[0].as_tensor().expect("ys should be tensor");
        let ys_vals = ys_tensor.to_f64_vec().unwrap();
        assert_eq!(ys_vals, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_scan_accumulate_product() {
        // scan(mul, init=1, xs=[1,2,3,4]) → carry=24, ys=[1,2,6,24]
        let init = vec![Value::scalar_f64(1.0)];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Mul, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 24.0);
        let ys_vals = ys[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(ys_vals, vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_scan_functional_scalar_outputs_use_shared_stack_axis0() {
        let init = vec![Value::scalar_i64(0)];
        let xs = Value::vector_i64(&[0, 1]).unwrap();
        let (_, ys) = super::eval_scan_functional(
            init,
            &xs,
            |carry, x| {
                let carry_value = carry[0].as_i64_scalar().expect("carry should be i64");
                let x_value = x.as_i64_scalar().expect("scan slice should be i64");
                let next_carry = Value::scalar_i64(carry_value + x_value);
                let y = if x_value == 0 {
                    Value::scalar_i64(1)
                } else {
                    Value::scalar_f64(2.5)
                };
                Ok((vec![next_carry], vec![y]))
            },
            false,
        )
        .expect("functional scan with mixed scalar outputs should succeed");

        let ys_tensor = ys[0].as_tensor().expect("ys should be tensor");
        assert_eq!(ys_tensor.shape, Shape::vector(2));
        assert_eq!(ys_tensor.dtype, DType::F64);
        assert_eq!(ys_tensor.to_f64_vec().unwrap(), vec![1.0, 2.5]);
    }

    #[test]
    fn test_scan_custom_body() {
        // Custom body: carry = carry + x * 2
        let init = vec![Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let doubled = eval_primitive(
                    Primitive::Mul,
                    &[x, Value::scalar_f64(2.0)],
                    &BTreeMap::new(),
                )?;
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), doubled], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            false,
        )
        .unwrap();
        // 0 + 1*2 = 2, 2 + 2*2 = 6, 6 + 3*2 = 12
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 12.0);
        let ys_vals = ys[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(ys_vals, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_scan_multi_carry() {
        // Multi-carry: (count, sum). Body: count += 1, sum += x
        let init = vec![Value::scalar_f64(0.0), Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[10.0, 20.0, 30.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_count = eval_primitive(
                    Primitive::Add,
                    &[c[0].clone(), Value::scalar_f64(1.0)],
                    &BTreeMap::new(),
                )?;
                let new_sum =
                    eval_primitive(Primitive::Add, &[c[1].clone(), x.clone()], &BTreeMap::new())?;
                Ok((vec![new_count, new_sum], vec![x]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 3.0); // count
        assert_eq!(carry[1].as_f64_scalar().unwrap(), 60.0); // sum
        let ys_vals = ys[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(ys_vals, vec![10.0, 20.0, 30.0]); // identity output
    }

    #[test]
    fn test_scan_multi_carry_matrix_shape_witness() {
        let init = vec![Value::scalar_f64(1.0), Value::scalar_f64(10.0)];
        let xs = Value::vector_f64(&[2.0, 3.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let next_product =
                    eval_primitive(Primitive::Mul, &[c[0].clone(), x.clone()], &BTreeMap::new())?;
                let next_sum =
                    eval_primitive(Primitive::Add, &[c[1].clone(), x.clone()], &BTreeMap::new())?;
                Ok((
                    vec![next_product.clone(), next_sum.clone()],
                    vec![next_product, next_sum],
                ))
            },
            false,
        )
        .unwrap();

        assert_eq!(carry.len(), 2);
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 6.0);
        assert_eq!(carry[1].as_f64_scalar().unwrap(), 15.0);
        assert_eq!(ys.len(), 2);
        assert_eq!(ys[0].as_tensor().unwrap().shape.dims, vec![2]);
        assert_eq!(ys[1].as_tensor().unwrap().shape.dims, vec![2]);
        assert_eq!(
            ys[0].as_tensor().unwrap().to_f64_vec().unwrap(),
            vec![2.0, 6.0]
        );
        assert_eq!(
            ys[1].as_tensor().unwrap().to_f64_vec().unwrap(),
            vec![12.0, 15.0]
        );
    }

    #[test]
    fn test_scan_no_output() {
        // Scan that only accumulates carry, no per-step output
        let init = vec![Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry], vec![]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 6.0);
        assert!(ys.is_empty());
    }

    #[test]
    fn test_scan_mixed_output_kinds_returns_error() {
        let init = vec![Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[1.0, 2.0]).unwrap();
        let mut emit_tensor = false;
        let result = super::eval_scan_functional(
            init,
            &xs,
            |carry, _x| {
                let y = if emit_tensor {
                    Value::vector_f64(&[1.0]).unwrap()
                } else {
                    emit_tensor = true;
                    Value::scalar_f64(1.0)
                };
                Ok((carry, vec![y]))
            },
            false,
        );
        assert!(
            matches!(
                result,
                Err(EvalError::InvalidTensor(ValueError::MixedAxisStackKinds))
            ),
            "mixed scan outputs should return a structured error, got: {result:?}"
        );
    }

    #[test]
    fn test_scan_empty_xs() {
        // Scan over empty array → carry = init, ys = empty
        let init = vec![Value::scalar_f64(42.0)];
        let xs =
            Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap());
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 42.0);
        assert!(ys.is_empty());
    }

    #[test]
    fn test_scan_functional_scalar_tensor_xs_returns_error() {
        let init = vec![Value::scalar_f64(0.0)];
        let xs = Value::Tensor(
            TensorValue::new(DType::F64, Shape::scalar(), vec![Literal::from_f64(1.0)]).unwrap(),
        );

        let mut body_calls = 0_usize;
        let result = super::eval_scan_functional(
            init,
            &xs,
            |carry, _x| {
                body_calls += 1;
                Ok((carry, vec![]))
            },
            false,
        );
        assert!(
            matches!(
                result,
                Err(EvalError::TypeMismatch {
                    primitive: Primitive::Scan,
                    detail: "scan tensor xs must have a leading axis"
                })
            ),
            "expected structured scan rank error, got {result:?}"
        );
        assert_eq!(body_calls, 0);
    }

    #[test]
    fn test_scan_single_element() {
        // Scan over single-element array
        let init = vec![Value::scalar_f64(5.0)];
        let xs = Value::vector_f64(&[3.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            false,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 8.0);
        let ys_vals = ys[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(ys_vals, vec![8.0]);
    }

    #[test]
    fn test_scan_tensor_carry() {
        // Scan with rank-2 tensor as carry
        // carry is [2] vector, xs is [3] vector, body: carry = carry + [x, x]
        let init = vec![Value::vector_f64(&[0.0, 0.0]).unwrap()];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let (carry, _ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                // Build a [2] tensor from the scalar x
                let x_val = x.as_f64_scalar().unwrap();
                let x_vec = Value::vector_f64(&[x_val, x_val]).unwrap();
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x_vec], &BTreeMap::new())?;
                Ok((vec![new_carry], vec![]))
            },
            false,
        )
        .unwrap();
        // carry should be [0+1+2+3, 0+1+2+3] = [6, 6]
        let carry_tensor = carry[0].as_tensor().unwrap();
        let carry_vals = carry_tensor.to_f64_vec().unwrap();
        assert_eq!(carry_vals, vec![6.0, 6.0]);
    }

    #[test]
    fn test_scan_reverse() {
        // scan(add, init=0, xs=[1,2,3], reverse=true) → iterate 3,2,1
        // JAX reverse scan returns ys in the original leading-axis order.
        let init = vec![Value::scalar_f64(0.0)];
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let (carry, ys) = super::eval_scan_functional(
            init,
            &xs,
            |c, x| {
                let new_carry =
                    eval_primitive(Primitive::Add, &[c[0].clone(), x], &BTreeMap::new())?;
                Ok((vec![new_carry.clone()], vec![new_carry]))
            },
            true,
        )
        .unwrap();
        assert_eq!(carry[0].as_f64_scalar().unwrap(), 6.0);
        let ys_vals = ys[0].as_tensor().unwrap().to_f64_vec().unwrap();
        // Execution-order partial sums are [3, 5, 6]; output order is reversed back.
        assert_eq!(ys_vals, vec![6.0, 5.0, 3.0]);
    }

    // ── While loop tests ────────────────────────────────────────────

    fn while_params(body_op: &str, cond_op: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("body_op".to_owned(), body_op.to_owned());
        p.insert("cond_op".to_owned(), cond_op.to_owned());
        p
    }

    fn while_params_max(body_op: &str, cond_op: &str, max: usize) -> BTreeMap<String, String> {
        let mut p = while_params(body_op, cond_op);
        p.insert("max_iter".to_owned(), max.to_string());
        p
    }

    #[test]
    fn while_add_until_ge_threshold() {
        // while carry < 10: carry += 3 => 0, 3, 6, 9, 12 => stops at 12
        let init = Value::scalar_f64(0.0);
        let step = Value::scalar_f64(3.0);
        let threshold = Value::scalar_f64(10.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "lt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 12.0);
    }

    #[test]
    fn while_mul_until_ge_threshold() {
        // while carry < 100: carry *= 2 => 1, 2, 4, 8, 16, 32, 64, 128 => stops at 128
        let init = Value::scalar_f64(1.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(100.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("mul", "lt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 128.0);
    }

    #[test]
    fn while_sub_until_le_zero() {
        // while carry > 0: carry -= 2 => 10, 8, 6, 4, 2, 0 => stops at 0
        let init = Value::scalar_f64(10.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(0.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("sub", "gt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn while_condition_false_immediately() {
        // carry = 10, while carry < 5: carry += 1 => condition false, returns 10
        let init = Value::scalar_f64(10.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(5.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "lt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 10.0);
    }

    #[test]
    fn while_tensor_scalar_predicate_works() {
        let init = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(0)]).unwrap(),
        );
        let step = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(1)]).unwrap(),
        );
        let threshold = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(3)]).unwrap(),
        );
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "lt"),
        )
        .unwrap();
        let out_tensor = out.as_tensor().expect("while output should be tensor");
        assert_eq!(out_tensor.shape, Shape::scalar());
        assert_eq!(out_tensor.elements[0].as_i64().unwrap(), 3);
    }

    #[test]
    fn while_arity_error() {
        let result = eval_primitive(
            Primitive::While,
            &[Value::scalar_f64(0.0)],
            &while_params("add", "lt"),
        );
        assert!(result.is_err());
    }

    // ── New while_loop tests (bd-2807) ─────────────────────────────

    #[test]
    fn test_while_loop_countdown() {
        // while_loop(|x| x > 0, |x| x - 1, init=10) → 0
        let init = Value::scalar_f64(10.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(0.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("sub", "gt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn test_while_loop_convergence() {
        // Newton's method for sqrt(2): x_{n+1} = (x + 2/x) / 2
        // We approximate by iterating: carry = (carry + 2/carry) / 2
        // Using the functional API
        let init = vec![Value::scalar_f64(2.0)];
        let result = super::eval_while_loop_functional(
            init,
            100,
            |carry| {
                let x = carry[0].as_f64_scalar().unwrap();
                // Continue while |x² - 2| > 1e-10
                Ok((x * x - 2.0).abs() > 1e-10)
            },
            |carry| {
                let x = carry[0].as_f64_scalar().unwrap();
                let new_x = (x + 2.0 / x) / 2.0;
                Ok(vec![Value::scalar_f64(new_x)])
            },
        )
        .unwrap();
        let sqrt2 = result[0].as_f64_scalar().unwrap();
        assert!(
            (sqrt2 - std::f64::consts::SQRT_2).abs() < 1e-10,
            "Newton's method should converge to sqrt(2), got {sqrt2}"
        );
    }

    #[test]
    fn test_while_loop_zero_iterations() {
        // Condition false initially → returns init unchanged
        let init = Value::scalar_f64(42.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(100.0);
        // carry > 100 is false for carry=42
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "gt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 42.0);
    }

    #[test]
    fn test_while_loop_single_iteration() {
        // Condition true once, then false
        // carry=5, while carry < 6: carry += 1 → 6, then 6 < 6 is false
        let init = Value::scalar_f64(5.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(6.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "lt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 6.0);
    }

    #[test]
    fn test_while_loop_max_iterations() {
        // Loop that would never terminate, but max_iter caps it
        let init = Value::scalar_f64(0.0);
        let step = Value::scalar_f64(0.0); // Adding zero never changes carry
        let threshold = Value::scalar_f64(10.0);
        let result = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params_max("add", "lt", 5),
        );
        if let Err(super::EvalError::MaxIterationsExceeded { max_iterations, .. }) = result {
            assert_eq!(max_iterations, 5);
        } else {
            assert!(
                matches!(result, Err(super::EvalError::MaxIterationsExceeded { .. })),
                "expected MaxIterationsExceeded, got {result:?}"
            );
        }
    }

    #[test]
    fn test_while_loop_functional_tuple_carry() {
        // Carry state is (i, accumulator): count from 0 to 5, accumulating squares
        let init = vec![Value::scalar_f64(0.0), Value::scalar_f64(0.0)];
        let result = super::eval_while_loop_functional(
            init,
            100,
            |carry| {
                let i = carry[0].as_f64_scalar().unwrap();
                Ok(i < 5.0)
            },
            |carry| {
                let i = carry[0].as_f64_scalar().unwrap();
                let acc = carry[1].as_f64_scalar().unwrap();
                Ok(vec![
                    Value::scalar_f64(i + 1.0),
                    Value::scalar_f64(acc + i * i),
                ])
            },
        )
        .unwrap();
        let final_i = result[0].as_f64_scalar().unwrap();
        let final_acc = result[1].as_f64_scalar().unwrap();
        assert_eq!(final_i, 5.0);
        // 0² + 1² + 2² + 3² + 4² = 30
        assert_eq!(final_acc, 30.0);
    }

    #[test]
    fn test_while_loop_functional_shape_mismatch() {
        // Body changes carry shape → error
        let init = vec![Value::scalar_f64(0.0)];
        let result = super::eval_while_loop_functional(
            init,
            100,
            |_carry| Ok(true),
            |_carry| {
                // Return a vector instead of scalar
                Ok(vec![Value::vector_f64(&[1.0, 2.0]).unwrap()])
            },
        );
        if let Err(super::EvalError::ShapeChanged { .. }) = result {
        } else {
            assert!(
                matches!(result, Err(super::EvalError::ShapeChanged { .. })),
                "expected ShapeChanged, got {result:?}"
            );
        }
    }

    #[test]
    fn test_while_loop_div_body_op() {
        // while carry > 1: carry /= 2 → 16, 8, 4, 2, 1 → stops at 1
        let init = Value::scalar_f64(16.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(1.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("div", "gt"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 1.0);
    }

    #[test]
    fn test_while_loop_ne_cond() {
        // while carry != 10: carry += 2 → 0, 2, 4, 6, 8, 10 → stops at 10
        let init = Value::scalar_f64(0.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(10.0);
        let out = eval_primitive(
            Primitive::While,
            &[init, step, threshold],
            &while_params("add", "ne"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 10.0);
    }

    // ── Bitwise tests ───────────────────────────────────────────────

    #[test]
    // Expected-value arrays below mirror the bitwise-AND operation element-wise
    // (e.g. `3 & 0`, `2 & -1`) so each entry reads as the op under test; clippy's
    // erasing_op/identity_op suggestions would obscure that intent.
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn i32_bitwise_ops_supported_and_correct() {
        // i32 (JAX's default int) bitwise previously ERRORED ("bitwise ops require
        // integer types"). Verify every op now works at i32 width — especially
        // ShiftRightLogical, which must zero-fill from bit 31 (negative i32), not bit 63.
        use fj_core::Primitive::{
            BitwiseAnd, BitwiseOr, BitwiseXor, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical,
        };
        let mk = |d: &[i32]| {
            Value::Tensor(
                TensorValue::new(
                    fj_core::DType::I32,
                    Shape::vector(d.len() as u32),
                    d.iter()
                        .map(|&v| fj_core::Literal::I64(i64::from(v)))
                        .collect(),
                )
                .unwrap(),
            )
        };
        let geti = |v: &Value| -> Vec<i32> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    fj_core::Literal::I64(x) => *x as i32,
                    o => panic!("expected I64-backed i32, got {o:?}"),
                })
                .collect()
        };
        let refop = |op: fj_core::Primitive, x: i32, y: i32| -> i32 {
            match op {
                BitwiseAnd => x & y,
                BitwiseOr => x | y,
                BitwiseXor => x ^ y,
                ShiftLeft => x.wrapping_shl(y as u32),
                ShiftRightArithmetic => x.wrapping_shr(y as u32),
                ShiftRightLogical => ((x as u32).wrapping_shr(y as u32)) as i32,
                _ => x,
            }
        };
        let a = [-1i32, 0x1234_5678, i32::MIN, 5, -7, 0];
        let b = [0x0f0f_0f0f, -1, 3, 2, 1, 31];
        let p = BTreeMap::new();
        for op in [
            BitwiseAnd,
            BitwiseOr,
            BitwiseXor,
            ShiftLeft,
            ShiftRightArithmetic,
            ShiftRightLogical,
        ] {
            let r = eval_primitive(op, &[mk(&a), mk(&b)], &p).unwrap();
            assert_eq!(
                r.as_tensor().unwrap().dtype,
                fj_core::DType::I32,
                "{op:?} dtype"
            );
            assert_eq!(
                geti(&r),
                a.iter()
                    .zip(&b)
                    .map(|(&x, &y)| refop(op, x, y))
                    .collect::<Vec<_>>(),
                "{op:?} same-shape"
            );
            let s = Value::Scalar(fj_core::Literal::I64(i64::from(b[0])));
            assert_eq!(
                geti(&eval_primitive(op, &[mk(&a), s.clone()], &p).unwrap()),
                a.iter().map(|&x| refop(op, x, b[0])).collect::<Vec<_>>(),
                "{op:?} tensor⊗scalar"
            );
            assert_eq!(
                geti(&eval_primitive(op, &[s.clone(), mk(&a)], &p).unwrap()),
                a.iter().map(|&x| refop(op, b[0], x)).collect::<Vec<_>>(),
                "{op:?} scalar⊗tensor"
            );
        }
        // Broadcast [2,3] & [3].
        let m = Value::Tensor(
            TensorValue::new(
                fj_core::DType::I32,
                Shape { dims: vec![2, 3] },
                [-1i32, 2, 3, 4, 5, 6]
                    .iter()
                    .map(|&v| fj_core::Literal::I64(i64::from(v)))
                    .collect(),
            )
            .unwrap(),
        );
        let r = eval_primitive(BitwiseAnd, &[m, mk(&[7i32, -1, 0])], &p).unwrap();
        assert_eq!(
            r.as_tensor().unwrap().dtype,
            fj_core::DType::I32,
            "broadcast dtype"
        );
        assert_eq!(
            geti(&r),
            vec![-1 & 7, 2 & -1, 3 & 0, 4 & 7, 5 & -1, 6 & 0],
            "broadcast and"
        );
    }

    // u32 dense (densified) vs boxed (as_u32_slice None → per-Literal) for A/B + iso.
    fn u32_bw_dense(d: &[u32]) -> Value {
        Value::Tensor(
            TensorValue::new(
                fj_core::DType::U32,
                Shape::vector(d.len() as u32),
                d.iter().map(|&v| fj_core::Literal::U32(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn u32_bw_boxed(d: &[u32]) -> Value {
        Value::Tensor(
            TensorValue::new_with_literal_buffer(
                fj_core::DType::U32,
                Shape::vector(d.len() as u32),
                fj_core::LiteralBuffer::new(d.iter().map(|&v| fj_core::Literal::U32(v)).collect()),
            )
            .unwrap(),
        )
    }

    #[test]
    fn u32_u64_bitwise_dense_matches_generic() {
        // The dense as_u32_slice/as_u64_slice bitwise paths (same-shape, scalar both
        // orders, broadcast) must be bit-identical to the boxed per-Literal fallback,
        // for all six ops, incl high-bit-set values and ShiftRightLogical (unsigned
        // zero-fill is the whole-width shift for u32/u64 — no sign-extension).
        use fj_core::Primitive::{
            BitwiseAnd, BitwiseOr, BitwiseXor, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical,
        };
        let geti = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    fj_core::Literal::U32(x) => u64::from(*x),
                    fj_core::Literal::U64(x) => *x,
                    o => panic!("expected unsigned literal, got {o:?}"),
                })
                .collect()
        };
        let p = BTreeMap::new();
        let a32 = [0xFFFF_FFFFu32, 0x1234_5678, 0x8000_0000, 5, 7, 0];
        let b32 = [0x0F0F_0F0Fu32, 0xFFFF_FFFF, 3, 2, 1, 31];
        let u64d = |d: &[u64]| {
            Value::Tensor(
                TensorValue::new(
                    fj_core::DType::U64,
                    Shape::vector(d.len() as u32),
                    d.iter().map(|&v| fj_core::Literal::U64(v)).collect(),
                )
                .unwrap(),
            )
        };
        let u64b = |d: &[u64]| {
            Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    fj_core::DType::U64,
                    Shape::vector(d.len() as u32),
                    fj_core::LiteralBuffer::new(
                        d.iter().map(|&v| fj_core::Literal::U64(v)).collect(),
                    ),
                )
                .unwrap(),
            )
        };
        let a64 = [u64::MAX, 1u64 << 63, 0x1234_5678_9ABC_DEF0, 5, 7, 0];
        let b64 = [0x0F0F_0F0F_0F0F_0F0Fu64, u64::MAX, 4, 2, 1, 63];

        for op in [
            BitwiseAnd,
            BitwiseOr,
            BitwiseXor,
            ShiftLeft,
            ShiftRightArithmetic,
            ShiftRightLogical,
        ] {
            // u32 same-shape
            let d =
                geti(&eval_primitive(op, &[u32_bw_dense(&a32), u32_bw_dense(&b32)], &p).unwrap());
            let g =
                geti(&eval_primitive(op, &[u32_bw_boxed(&a32), u32_bw_boxed(&b32)], &p).unwrap());
            assert_eq!(d, g, "{op:?} u32 same-shape dense!=generic");
            // u32 scalar both orders
            let s = Value::Scalar(fj_core::Literal::U32(b32[0]));
            let d = geti(&eval_primitive(op, &[u32_bw_dense(&a32), s.clone()], &p).unwrap());
            let g = geti(&eval_primitive(op, &[u32_bw_boxed(&a32), s.clone()], &p).unwrap());
            assert_eq!(d, g, "{op:?} u32 tensor⊗scalar dense!=generic");
            let d = geti(&eval_primitive(op, &[s.clone(), u32_bw_dense(&a32)], &p).unwrap());
            let g = geti(&eval_primitive(op, &[s.clone(), u32_bw_boxed(&a32)], &p).unwrap());
            assert_eq!(d, g, "{op:?} u32 scalar⊗tensor dense!=generic");
            // u64 same-shape
            let d = geti(&eval_primitive(op, &[u64d(&a64), u64d(&b64)], &p).unwrap());
            let g = geti(&eval_primitive(op, &[u64b(&a64), u64b(&b64)], &p).unwrap());
            assert_eq!(d, g, "{op:?} u64 same-shape dense!=generic");
        }

        // u32 broadcast [2,3] op [3], dense vs boxed.
        let m_dense = Value::Tensor(
            TensorValue::new(
                fj_core::DType::U32,
                Shape { dims: vec![2, 3] },
                [1u32, 2, 3, 4, 5, 6]
                    .iter()
                    .map(|&v| fj_core::Literal::U32(v))
                    .collect(),
            )
            .unwrap(),
        );
        let m_boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                fj_core::DType::U32,
                Shape { dims: vec![2, 3] },
                fj_core::LiteralBuffer::new(
                    [1u32, 2, 3, 4, 5, 6]
                        .iter()
                        .map(|&v| fj_core::Literal::U32(v))
                        .collect(),
                ),
            )
            .unwrap(),
        );
        let d =
            geti(&eval_primitive(BitwiseXor, &[m_dense, u32_bw_dense(&[7, 0xFF, 0])], &p).unwrap());
        let g =
            geti(&eval_primitive(BitwiseXor, &[m_boxed, u32_bw_boxed(&[7, 0xFF, 0])], &p).unwrap());
        assert_eq!(d, g, "u32 broadcast xor dense!=generic");
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_u32_bitwise_dense_vs_generic() {
        use std::time::Instant;
        let n = 1_000_000usize;
        let a: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let b: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(40_503) ^ 0x9E37_79B9)
            .collect();
        let (ad, ab) = (u32_bw_dense(&a), u32_bw_boxed(&a));
        let (bd, bb) = (u32_bw_dense(&b), u32_bw_boxed(&b));
        let p = BTreeMap::new();
        let best = |mut f: Box<dyn FnMut() -> usize>| {
            f();
            let mut t = f64::MAX;
            for _ in 0..7 {
                let s = Instant::now();
                std::hint::black_box(f());
                t = t.min(s.elapsed().as_secs_f64());
            }
            t
        };
        let (abc, bbc) = (ab.clone(), bb.clone());
        let generic = best(Box::new(move || {
            eval_primitive(Primitive::BitwiseXor, &[abc.clone(), bbc.clone()], &p)
                .unwrap()
                .as_tensor()
                .unwrap()
                .elements
                .len()
        }));
        let p2 = BTreeMap::new();
        let (adc, bdc) = (ad.clone(), bd.clone());
        let dense = best(Box::new(move || {
            eval_primitive(Primitive::BitwiseXor, &[adc.clone(), bdc.clone()], &p2)
                .unwrap()
                .as_tensor()
                .unwrap()
                .elements
                .len()
        }));
        println!(
            "BENCH u32 same-shape BitwiseXor [1e6]: generic={:.2}ms dense={:.2}ms speedup={:.2}x",
            generic * 1e3,
            dense * 1e3,
            generic / dense
        );
    }

    #[test]
    fn bitwise_and_scalars() {
        let a = Value::scalar_i64(0b1100);
        let b = Value::scalar_i64(0b1010);
        let out = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 0b1000);
    }

    #[test]
    fn bool_words_swar_bitwise_matches_per_element() {
        // f64 same-shape comparisons emit bit-packed BoolWords. Combining two such
        // masks with bitwise and/or/xor must take the SWAR word-level path and be
        // bit-identical to the per-element (Bool storage) reference. n=200 spans
        // multiple u64 words + a partial tail (200 % 64 = 8).
        let n = 200usize;
        let a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.123).sin()).collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.077).cos()).collect();
        let va = Value::Tensor(TensorValue::new_f64_values(Shape::vector(n as u32), a).unwrap());
        let vb = Value::Tensor(TensorValue::new_f64_values(Shape::vector(n as u32), b).unwrap());
        let p = no_params();
        // Two BoolWords masks (f64 same-shape ⇒ BoolWords storage).
        let m1 = eval_primitive(Primitive::Lt, &[va.clone(), vb.clone()], &p).unwrap();
        let m2 = eval_primitive(Primitive::Ge, &[va.clone(), vb.clone()], &p).unwrap();
        assert!(
            m1.as_tensor().unwrap().elements.as_bool_words().is_some(),
            "m1 must be BoolWords"
        );
        assert!(
            m2.as_tensor().unwrap().elements.as_bool_words().is_some(),
            "m2 must be BoolWords"
        );
        let bits = |v: &Value| -> Vec<bool> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| matches!(l, Literal::Bool(true)))
                .collect()
        };
        let m1b = bits(&m1);
        let m2b = bits(&m2);
        // Per-element reference via Bool (Vec<bool>) storage.
        let ref_mk = |d: &[bool]| {
            Value::Tensor(
                TensorValue::new_bool_values(Shape::vector(n as u32), d.to_vec()).unwrap(),
            )
        };
        for prim in [
            Primitive::BitwiseAnd,
            Primitive::BitwiseOr,
            Primitive::BitwiseXor,
        ] {
            let swar = eval_primitive(prim, &[m1.clone(), m2.clone()], &p).unwrap();
            assert_eq!(
                swar.as_tensor().unwrap().dtype,
                DType::Bool,
                "{prim:?} dtype"
            );
            let want: Vec<bool> = m1b
                .iter()
                .zip(&m2b)
                .map(|(&x, &y)| match prim {
                    Primitive::BitwiseAnd => x & y,
                    Primitive::BitwiseOr => x | y,
                    _ => x ^ y,
                })
                .collect();
            assert_eq!(bits(&swar), want, "{prim:?} SWAR != per-element reference");
            // Also equals the Bool-storage path bit-for-bit.
            let perelem = eval_primitive(prim, &[ref_mk(&m1b), ref_mk(&m2b)], &p).unwrap();
            assert_eq!(
                bits(&swar),
                bits(&perelem),
                "{prim:?} SWAR != Bool-storage path"
            );
        }
        // Unary NOT on a BoolWords mask (SWAR word-flip) must match per-element !.
        let not_swar =
            eval_primitive(Primitive::BitwiseNot, std::slice::from_ref(&m1), &p).unwrap();
        assert_eq!(
            not_swar.as_tensor().unwrap().dtype,
            DType::Bool,
            "NOT dtype"
        );
        let not_want: Vec<bool> = m1b.iter().map(|&x| !x).collect();
        assert_eq!(
            bits(&not_swar),
            not_want,
            "NOT SWAR != per-element reference"
        );
        assert_eq!(
            bits(&not_swar),
            bits(&eval_primitive(Primitive::BitwiseNot, &[ref_mk(&m1b)], &p).unwrap()),
            "NOT SWAR != Bool-storage path"
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_bool_words_swar_bitwise() {
        use std::time::Instant;
        let n = 1_000_000usize;
        let a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.002).cos()).collect();
        let va = Value::Tensor(TensorValue::new_f64_values(Shape::vector(n as u32), a).unwrap());
        let vb = Value::Tensor(TensorValue::new_f64_values(Shape::vector(n as u32), b).unwrap());
        let p = no_params();
        let m1 = eval_primitive(Primitive::Lt, &[va.clone(), vb.clone()], &p).unwrap(); // BoolWords
        let m2 = eval_primitive(Primitive::Ge, &[va.clone(), vb.clone()], &p).unwrap(); // BoolWords
        let m1b: Vec<bool> = m1
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| matches!(l, Literal::Bool(true)))
            .collect();
        let m2b: Vec<bool> = m2
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| matches!(l, Literal::Bool(true)))
            .collect();
        let b1 = Value::Tensor(TensorValue::new_bool_values(Shape::vector(n as u32), m1b).unwrap());
        let b2 = Value::Tensor(TensorValue::new_bool_values(Shape::vector(n as u32), m2b).unwrap());
        let best = |x: &Value, y: &Value| {
            let _ = eval_primitive(Primitive::BitwiseAnd, &[x.clone(), y.clone()], &p).unwrap();
            let mut t = f64::MAX;
            for _ in 0..7 {
                let s = Instant::now();
                let o = eval_primitive(Primitive::BitwiseAnd, &[x.clone(), y.clone()], &p).unwrap();
                std::hint::black_box(o.as_tensor().unwrap().elements.len());
                t = t.min(s.elapsed().as_secs_f64());
            }
            t
        };
        let perelem = best(&b1, &b2); // Bool (Vec<bool>) per-element path
        let swar = best(&m1, &m2); // BoolWords SWAR word path
        println!(
            "BENCH bool bitwise-and [1e6]: per-element(Bool)={:.3}ms SWAR(BoolWords)={:.3}ms speedup={:.2}x",
            perelem * 1e3,
            swar * 1e3,
            perelem / swar
        );
    }

    #[test]
    fn bitwise_bool_and_or_xor() {
        // JAX lowers logical_and/or/xor (and bool `&|^*`) to lax.bitwise_* on
        // Bool, so these must produce a Bool result, not error.
        let mk = |data: &[bool], dims: Vec<u32>| {
            Value::Tensor(TensorValue::new_bool_values(Shape { dims }, data.to_vec()).unwrap())
        };
        let bits = |v: &Value| -> Vec<bool> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| matches!(l, Literal::Bool(true)))
                .collect()
        };
        let a = [true, true, false, false];
        let b = [true, false, true, false];

        // Same-shape.
        for (prim, expected) in [
            (Primitive::BitwiseAnd, vec![true, false, false, false]),
            (Primitive::BitwiseOr, vec![true, true, true, false]),
            (Primitive::BitwiseXor, vec![false, true, true, false]),
        ] {
            let out =
                eval_primitive(prim, &[mk(&a, vec![4]), mk(&b, vec![4])], &no_params()).unwrap();
            assert_eq!(
                out.as_tensor().unwrap().dtype,
                DType::Bool,
                "{prim:?} dtype"
            );
            assert_eq!(bits(&out), expected, "{prim:?} same-shape");
        }

        // Scalar ⊗ tensor.
        let out = eval_primitive(
            Primitive::BitwiseAnd,
            &[Value::scalar_bool(true), mk(&a, vec![4])],
            &no_params(),
        )
        .unwrap();
        assert_eq!(bits(&out), vec![true, true, false, false], "scalar&tensor");

        // Broadcast [2,2] | [2].
        let out = eval_primitive(
            Primitive::BitwiseOr,
            &[
                mk(&[true, false, false, false], vec![2, 2]),
                mk(&[false, true], vec![2]),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(bits(&out), vec![true, true, false, true], "broadcast or");

        // Shifts remain undefined on bool.
        assert!(
            eval_primitive(
                Primitive::ShiftLeft,
                &[mk(&a, vec![4]), mk(&b, vec![4])],
                &no_params()
            )
            .is_err()
        );
    }

    #[test]
    fn bitwise_or_scalars() {
        let a = Value::scalar_i64(0b1100);
        let b = Value::scalar_i64(0b1010);
        let out = eval_primitive(Primitive::BitwiseOr, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 0b1110);
    }

    #[test]
    fn bitwise_xor_scalars() {
        let a = Value::scalar_i64(0b1100);
        let b = Value::scalar_i64(0b1010);
        let out = eval_primitive(Primitive::BitwiseXor, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 0b0110);
    }

    #[test]
    fn bitwise_not_bool_is_logical_not() {
        // JAX lowers logical_not / bool `~` to lax.bitwise_not on Bool, which must
        // return the negated bool, not error.
        let m = Value::Tensor(
            TensorValue::new_bool_values(Shape { dims: vec![4] }, vec![true, false, true, false])
                .unwrap(),
        );
        let out = eval_primitive(Primitive::BitwiseNot, &[m], &no_params()).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.dtype, DType::Bool);
        let vals: Vec<bool> = t
            .elements
            .iter()
            .map(|l| matches!(l, Literal::Bool(true)))
            .collect();
        assert_eq!(vals, vec![false, true, false, true]);

        // Scalar bool too.
        let s = eval_primitive(
            Primitive::BitwiseNot,
            &[Value::scalar_bool(true)],
            &no_params(),
        )
        .unwrap();
        assert!(matches!(s, Value::Scalar(Literal::Bool(false))));
    }

    #[test]
    fn bitwise_not_scalar() {
        let a = Value::scalar_i64(0);
        let out = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), -1); // !0 = all ones = -1 in two's complement
    }

    #[test]
    fn bitwise_and_u32_scalars() {
        let a = Value::scalar_u32(0b1111_0000);
        let b = Value::scalar_u32(0b1010_1010);
        let out = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_u32(0b1010_0000));
    }

    #[test]
    fn bitwise_not_u64_scalar() {
        let a = Value::scalar_u64(0);
        let out = eval_primitive(Primitive::BitwiseNot, &[a], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_u64(u64::MAX));
    }

    #[test]
    fn shift_left_scalar() {
        let a = Value::scalar_i64(1);
        let b = Value::scalar_i64(4);
        let out = eval_primitive(Primitive::ShiftLeft, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 16);
    }

    #[test]
    fn shift_right_arithmetic_scalar() {
        let a = Value::scalar_i64(16);
        let b = Value::scalar_i64(2);
        let out = eval_primitive(Primitive::ShiftRightArithmetic, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 4);
    }

    #[test]
    fn shift_right_logical_scalar() {
        let a = Value::scalar_i64(16);
        let b = Value::scalar_i64(2);
        let out = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 4);
    }

    #[test]
    fn shift_right_arithmetic_negative_preserves_sign() {
        let a = Value::scalar_i64(-8);
        let b = Value::scalar_i64(2);
        let out = eval_primitive(Primitive::ShiftRightArithmetic, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), -2);
    }

    #[test]
    fn shift_right_logical_negative_zero_fills() {
        let a = Value::scalar_i64(-8);
        let b = Value::scalar_i64(2);
        let out = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 4_611_686_018_427_387_902);
    }

    #[test]
    fn shift_right_logical_u32() {
        let a = Value::scalar_u32(0b1111_0000);
        let b = Value::scalar_u32(4);
        let out = eval_primitive(Primitive::ShiftRightLogical, &[a, b], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_u32(0b0000_1111));
    }

    #[test]
    fn unsigned_division_truncating() {
        let out = eval_primitive(
            Primitive::Div,
            &[Value::scalar_u32(7), Value::scalar_u32(2)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_u32(3));
    }

    #[test]
    fn unsigned_comparison_no_sign_extension() {
        let out = eval_primitive(
            Primitive::Gt,
            &[Value::scalar_u64(u64::MAX), Value::scalar_i64(-1)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_bool(true));
    }

    #[test]
    fn e2e_unsigned_int_ops() {
        let add = eval_primitive(
            Primitive::Add,
            &[Value::scalar_u32(5), Value::scalar_u32(6)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(add, Value::scalar_u32(11));

        let rem = eval_primitive(
            Primitive::Rem,
            &[Value::scalar_u64(17), Value::scalar_u64(5)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(rem, Value::scalar_u64(2));

        let popcnt = eval_primitive(
            Primitive::PopulationCount,
            &[Value::scalar_u64(0b1011)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(popcnt, Value::scalar_i64(3));
    }

    #[test]
    fn e2e_bitwise_reductions_oracle() {
        let reduce_and = eval_primitive(
            Primitive::ReduceAnd,
            &[bool_tensor(&[3], &[true, true, true])],
            &no_params(),
        )
        .unwrap();
        let reduce_or = eval_primitive(
            Primitive::ReduceOr,
            &[bool_tensor(&[3], &[false, true, false])],
            &no_params(),
        )
        .unwrap();
        let reduce_xor = eval_primitive(
            Primitive::ReduceXor,
            &[i64_tensor(&[3], &[1, 3, 2])],
            &no_params(),
        )
        .unwrap();

        let and_actual = reduce_and.as_bool_scalar().expect("bool output");
        let or_actual = reduce_or.as_bool_scalar().expect("bool output");
        let xor_actual = reduce_xor.as_i64_scalar().expect("i64 output");

        let and_expected = true;
        let or_expected = true;
        let xor_expected = 0_i64;

        let and_pass = and_actual == and_expected;
        let or_pass = or_actual == or_expected;
        let xor_pass = xor_actual == xor_expected;
        let all_passed = and_pass && or_pass && xor_pass;

        let case_logs = format!(
            concat!(
                "[",
                "{{\"test_name\":\"test_reduce_and_all_true\",\"reduction\":\"reduce_and\",",
                "\"input_dtype\":\"Bool\",\"input_shape\":[3],\"axis\":null,",
                "\"expected\":{},\"actual\":{},\"pass\":{}}},",
                "{{\"test_name\":\"test_reduce_or_one_true\",\"reduction\":\"reduce_or\",",
                "\"input_dtype\":\"Bool\",\"input_shape\":[3],\"axis\":null,",
                "\"expected\":{},\"actual\":{},\"pass\":{}}},",
                "{{\"test_name\":\"test_reduce_xor_integer\",\"reduction\":\"reduce_xor\",",
                "\"input_dtype\":\"I64\",\"input_shape\":[3],\"axis\":null,",
                "\"expected\":{},\"actual\":{},\"pass\":{}}}",
                "]"
            ),
            and_expected,
            and_actual,
            and_pass,
            or_expected,
            or_actual,
            or_pass,
            xor_expected,
            xor_actual,
            xor_pass
        );

        let generated_at_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let forensic_log = format!(
            concat!(
                "{{",
                "\"scenario\":\"e2e_bitwise_reductions_oracle\",",
                "\"generated_at_unix\":{},",
                "\"all_passed\":{},",
                "\"cases\":{}",
                "}}"
            ),
            generated_at_unix, all_passed, case_logs
        );

        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts");
        let e2e_path = root.join("e2e/e2e_bitwise_reductions.e2e.json");
        let test_log_path = root.join("testing/logs/fj-lax/e2e_bitwise_reductions_oracle.json");

        if let Some(parent) = e2e_path.parent() {
            std::fs::create_dir_all(parent).expect("create e2e artifact dir");
        }
        if let Some(parent) = test_log_path.parent() {
            std::fs::create_dir_all(parent).expect("create test log dir");
        }
        std::fs::write(&e2e_path, forensic_log).expect("write e2e forensic log");
        std::fs::write(&test_log_path, case_logs).expect("write test case logs");

        assert!(all_passed);
    }

    #[test]
    fn shift_right_arithmetic_vs_logical() {
        let sra_positive = eval_primitive(
            Primitive::ShiftRightArithmetic,
            &[Value::scalar_i64(16), Value::scalar_i64(2)],
            &no_params(),
        )
        .unwrap();
        let srl_positive = eval_primitive(
            Primitive::ShiftRightLogical,
            &[Value::scalar_i64(16), Value::scalar_i64(2)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(
            sra_positive.as_i64_scalar().unwrap(),
            srl_positive.as_i64_scalar().unwrap()
        );

        let sra_negative = eval_primitive(
            Primitive::ShiftRightArithmetic,
            &[Value::scalar_i64(-8), Value::scalar_i64(2)],
            &no_params(),
        )
        .unwrap();
        let srl_negative = eval_primitive(
            Primitive::ShiftRightLogical,
            &[Value::scalar_i64(-8), Value::scalar_i64(2)],
            &no_params(),
        )
        .unwrap();
        assert_ne!(
            sra_negative.as_i64_scalar().unwrap(),
            srl_negative.as_i64_scalar().unwrap()
        );
    }

    #[test]
    fn shift_right_by_zero_is_identity() {
        let sra = eval_primitive(
            Primitive::ShiftRightArithmetic,
            &[Value::scalar_i64(-8), Value::scalar_i64(0)],
            &no_params(),
        )
        .unwrap();
        let srl = eval_primitive(
            Primitive::ShiftRightLogical,
            &[Value::scalar_i64(-8), Value::scalar_i64(0)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(sra.as_i64_scalar().unwrap(), -8);
        assert_eq!(srl.as_i64_scalar().unwrap(), -8);
    }

    #[test]
    fn shift_right_full_width() {
        let sra = eval_primitive(
            Primitive::ShiftRightArithmetic,
            &[Value::scalar_i64(-8), Value::scalar_i64(32)],
            &no_params(),
        )
        .unwrap();
        let srl = eval_primitive(
            Primitive::ShiftRightLogical,
            &[Value::scalar_i64(-8), Value::scalar_i64(32)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(sra.as_i64_scalar().unwrap(), -1);
        assert_eq!(srl.as_i64_scalar().unwrap(), 4_294_967_295);
    }

    #[test]
    fn bitwise_type_error_f64() {
        let a = Value::scalar_f64(1.0);
        let b = Value::scalar_f64(2.0);
        let result = eval_primitive(Primitive::BitwiseAnd, &[a, b], &no_params());
        assert!(result.is_err());
    }

    // ── PopulationCount / CountLeadingZeros tests ─────────────────────

    #[test]
    fn population_count_scalar() {
        let a = Value::scalar_i64(0b1010_1100);
        let out = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 4);
    }

    #[test]
    fn population_count_zero() {
        let a = Value::scalar_i64(0);
        let out = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 0);
    }

    #[test]
    fn population_count_all_ones() {
        let a = Value::scalar_i64(-1); // all bits set
        let out = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 64);
    }

    #[test]
    fn count_leading_zeros_scalar() {
        let a = Value::scalar_i64(1);
        let out = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 63);
    }

    #[test]
    fn count_leading_zeros_zero() {
        let a = Value::scalar_i64(0);
        let out = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 64);
    }

    #[test]
    fn count_leading_zeros_negative() {
        let a = Value::scalar_i64(-1); // all ones, no leading zeros
        let out = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
        assert_eq!(out.as_i64_scalar().unwrap(), 0);
    }

    #[test]
    fn population_count_i32_tensor_negative() {
        let tensor = TensorValue::new(
            fj_core::DType::I32,
            fj_core::Shape::vector(2),
            vec![
                fj_core::Literal::I64(-1), // all 32 bits set when viewed as i32
                fj_core::Literal::I64(0b1010_1100),
            ],
        )
        .unwrap();
        let a = Value::Tensor(tensor);
        let out = eval_primitive(Primitive::PopulationCount, &[a], &no_params()).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.elements[0], fj_core::Literal::I64(32)); // -1 as i32 has 32 bits set, not 64
        assert_eq!(t.elements[1], fj_core::Literal::I64(4));
    }

    #[test]
    fn count_leading_zeros_i32_tensor() {
        let tensor = TensorValue::new(
            fj_core::DType::I32,
            fj_core::Shape::vector(3),
            vec![
                fj_core::Literal::I64(1),  // 31 leading zeros in i32
                fj_core::Literal::I64(0),  // 32 leading zeros in i32
                fj_core::Literal::I64(-1), // 0 leading zeros (all ones)
            ],
        )
        .unwrap();
        let a = Value::Tensor(tensor);
        let out = eval_primitive(Primitive::CountLeadingZeros, &[a], &no_params()).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.elements[0], fj_core::Literal::I64(31)); // not 63
        assert_eq!(t.elements[1], fj_core::Literal::I64(32)); // not 64
        assert_eq!(t.elements[2], fj_core::Literal::I64(0));
    }

    #[test]
    fn population_count_type_error() {
        let a = Value::scalar_f64(1.0);
        let result = eval_primitive(Primitive::PopulationCount, &[a], &no_params());
        assert!(result.is_err());
    }

    // ── ReduceWindow tests ──────────────────────────────────────────

    fn rw_params(reduce_op: &str, window: &str, strides: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("reduce_op".to_owned(), reduce_op.to_owned());
        p.insert("window_dimensions".to_owned(), window.to_owned());
        p.insert("window_strides".to_owned(), strides.to_owned());
        p
    }

    fn rw_params_with_padding(
        reduce_op: &str,
        window: &str,
        strides: &str,
        padding: &str,
    ) -> BTreeMap<String, String> {
        let mut p = rw_params(reduce_op, window, strides);
        p.insert("padding".to_owned(), padding.to_owned());
        p
    }

    #[test]
    fn reduce_window_base_dilation_1d_sum_matches_hand_computation() {
        // base_dilation=2 dilates [1,2,3] -> [1,0,2,0,3] (holes = sum init 0), then a
        // window=3, stride=1, VALID sum slides over the length-5 dilated base:
        //   [1,0,2]=3, [0,2,0]=2, [2,0,3]=5  => [3,2,5].
        let input = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let mut p = rw_params("sum", "3", "1");
        p.insert("base_dilation".to_owned(), "2".to_owned());
        let out =
            eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&input), &p).unwrap();
        let Value::Tensor(t) = &out else {
            panic!("expected tensor")
        };
        let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
        assert_eq!(vals, vec![3.0, 2.0, 5.0]);
    }

    #[test]
    fn reduce_window_base_dilation_matches_explicit_dilated_operand() {
        // GENERAL reference: base_dilation on the operand must equal running the SAME
        // reduce_window (base_dilation=1) on an operand that has been explicitly dilated
        // with the reduction init value in the holes. Cover sum (init 0) and max (init
        // -inf) over a 2-D operand with asymmetric per-axis dilation and a stride.
        let dims = [3usize, 4usize];
        let data: Vec<f64> = (0..(dims[0] * dims[1]))
            .map(|i| (i as f64) * 0.5 - 2.0)
            .collect();
        let operand = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![dims[0] as u32, dims[1] as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let base_dil = [2usize, 3usize];
        for (op, init) in [("sum", 0.0_f64), ("max", f64::NEG_INFINITY)] {
            // Build the explicitly base-dilated operand: size (n-1)*db+1 per axis, holes=init.
            let dd = [
                (dims[0] - 1) * base_dil[0] + 1,
                (dims[1] - 1) * base_dil[1] + 1,
            ];
            let mut dilated = vec![init; dd[0] * dd[1]];
            for r in 0..dims[0] {
                for c in 0..dims[1] {
                    dilated[(r * base_dil[0]) * dd[1] + c * base_dil[1]] = data[r * dims[1] + c];
                }
            }
            let dilated_operand = Value::Tensor(
                TensorValue::new_f64_values(
                    Shape {
                        dims: vec![dd[0] as u32, dd[1] as u32],
                    },
                    dilated,
                )
                .unwrap(),
            );

            let mut p_dil = rw_params(op, "2,2", "2,1");
            p_dil.insert("base_dilation".to_owned(), "2,3".to_owned());
            let got = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&operand),
                &p_dil,
            )
            .unwrap();

            let p_ref = rw_params(op, "2,2", "2,1"); // base_dilation defaults to 1
            let want = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&dilated_operand),
                &p_ref,
            )
            .unwrap();

            let (Value::Tensor(gt), Value::Tensor(wt)) = (&got, &want) else {
                panic!("expected tensors")
            };
            assert_eq!(gt.shape.dims, wt.shape.dims, "{op}: shape mismatch");
            let gv: Vec<u64> = gt
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect();
            let wv: Vec<u64> = wt
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect();
            assert_eq!(gv, wv, "{op}: base_dilation != explicit dilated operand");
        }
    }

    #[test]
    fn reduce_window_default_base_dilation_still_succeeds() {
        // Default (all-1) base_dilation is the unchanged path and must still succeed.
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims: vec![4, 4] },
                (0..16).map(|i| i as f64).collect(),
            )
            .unwrap(),
        );
        let mut p = rw_params("max", "2,2", "1,1");
        p.insert("base_dilation".to_owned(), "1,1".to_owned());
        p.insert("window_dilation".to_owned(), "1,1".to_owned());
        eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&input), &p)
            .expect("reduce_window with default (all-1) dilation must still succeed");
    }

    #[test]
    fn reduce_window_window_dilation_matches_reference() {
        // Atrous pooling (window_dilation): window tap t reads input out*stride + t*d.
        // VALID, bit-identical to a direct dilated f64 reference, for max and sum over a
        // 2D input. Effective window extent (w-1)*d+1 -> out = (in - eff)/stride + 1.
        let (h, w) = (7usize, 8usize);
        let (wh, ww, dh, dw) = (2usize, 3usize, 3usize, 2usize);
        let data: Vec<f64> = (0..h * w).map(|i| ((i * 7) % 13) as f64 - 5.0).collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![h as u32, w as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let (eff_h, eff_w) = ((wh - 1) * dh + 1, (ww - 1) * dw + 1);
        let (out_h, out_w) = (h - eff_h + 1, w - eff_w + 1);
        for op in ["max", "sum"] {
            let mut p = rw_params_with_padding(op, &format!("{wh},{ww}"), "1,1", "VALID");
            p.insert("window_dilation".to_owned(), format!("{dh},{dw}"));
            let got =
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&input), &p).unwrap();
            let t = got.as_tensor().unwrap();
            assert_eq!(
                t.shape.dims,
                vec![out_h as u32, out_w as u32],
                "{op} dilated shape"
            );
            let got_v: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            let mut want = Vec::new();
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = if op == "max" { f64::NEG_INFINITY } else { 0.0 };
                    for a in 0..wh {
                        for b in 0..ww {
                            let v = data[(oh + a * dh) * w + (ow + b * dw)];
                            acc = if op == "max" { acc.max(v) } else { acc + v };
                        }
                    }
                    want.push(acc);
                }
            }
            assert_eq!(
                got_v, want,
                "{op} dilated pooling must match direct reference"
            );
        }
    }

    #[test]
    fn reduce_window_sum_same_padding_zero_pads_like_valid_on_padded_input() {
        // XLA pads reduce_window with the sum init (0.0) and reduces the FULL
        // window. Metamorphic proof exercising the 3x3 fast path, the general
        // rank-2 f64 sum path, and the generic N-D loop: 'same' padding on X must
        // be bit-identical to 'valid' on X explicitly zero-bordered. Signed-zero /
        // infinity inputs make the skip-vs-zero-pad difference observable.
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };
        let mk2 = |rows: usize, cols: usize, data: Vec<f64>| {
            Value::Tensor(
                TensorValue::new_f64_values(
                    Shape {
                        dims: vec![rows as u32, cols as u32],
                    },
                    data,
                )
                .unwrap(),
            )
        };
        let check_2d = |rows: usize, cols: usize, data: Vec<f64>, win: usize, pad: usize| {
            let win_s = format!("{win},{win}");
            let same = eval_primitive(
                Primitive::ReduceWindow,
                &[mk2(rows, cols, data.clone())],
                &rw_params_with_padding("sum", &win_s, "1,1", "same"),
            )
            .unwrap();
            let (pr, pc) = (rows + 2 * pad, cols + 2 * pad);
            let mut padded = vec![0.0_f64; pr * pc];
            for r in 0..rows {
                for c in 0..cols {
                    padded[(r + pad) * pc + (c + pad)] = data[r * cols + c];
                }
            }
            let valid = eval_primitive(
                Primitive::ReduceWindow,
                &[mk2(pr, pc, padded)],
                &rw_params_with_padding("sum", &win_s, "1,1", "valid"),
            )
            .unwrap();
            assert_eq!(
                same.as_tensor().unwrap().shape.dims,
                valid.as_tensor().unwrap().shape.dims,
                "2d shape {rows}x{cols} win {win}"
            );
            assert_eq!(
                bits(&same),
                bits(&valid),
                "2d sum same vs valid-on-padded {rows}x{cols} win {win}"
            );
        };

        // 4x4, 3x3 window → the 3x3-same fast path (border + interior).
        let d4: Vec<f64> = vec![
            -0.0,
            1.0,
            -2.0,
            0.5,
            3.0,
            -0.0,
            f64::INFINITY,
            -1.0,
            -0.0,
            -0.0,
            2.0,
            -3.0,
            4.0,
            -1.5,
            -0.0,
            0.0,
        ];
        check_2d(4, 4, d4, 3, 1);

        // 6x6, 5x5 window → the general rank-2 f64 sum path (not the 3x3 special).
        let d6: Vec<f64> = (0..36)
            .map(|i| match i % 7 {
                0 => -0.0,
                3 => f64::INFINITY,
                _ => ((i as f64) * 0.1).sin() - 0.3,
            })
            .collect();
        check_2d(6, 6, d6, 5, 2);

        // 1D window-3 'same' → the generic N-D loop (rank != 2).
        let x1: Vec<f64> = vec![-0.0, 1.0, -0.0, f64::INFINITY, -2.0, -0.0];
        let same1 = eval_primitive(
            Primitive::ReduceWindow,
            &[Value::vector_f64(&x1).unwrap()],
            &rw_params_with_padding("sum", "3", "1", "same"),
        )
        .unwrap();
        let mut p1 = vec![0.0_f64; x1.len() + 2];
        p1[1..1 + x1.len()].copy_from_slice(&x1);
        let valid1 = eval_primitive(
            Primitive::ReduceWindow,
            &[Value::vector_f64(&p1).unwrap()],
            &rw_params_with_padding("sum", "3", "1", "valid"),
        )
        .unwrap();
        assert_eq!(
            bits(&same1),
            bits(&valid1),
            "1d generic sum same vs valid-on-padded"
        );
    }

    #[test]
    fn reduce_window_sum_1d() {
        // [1, 2, 3, 4, 5], window=3, stride=1, valid => [6, 9, 12]
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "3", "1"),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![6.0, 9.0, 12.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_sum_preserves_f32_literal_dtype() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: vec![4] },
                vec![
                    Literal::from_f32(1.0),
                    Literal::from_f32(2.0),
                    Literal::from_f32(3.0),
                    Literal::from_f32(4.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "2", "1"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::F32);
            assert_eq!(t.shape.dims, vec![3]);
            t.validate_dtype_consistency()
                .expect("reduce_window F32 output dtype/element invariant");
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 5.0, 7.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_sum_preserves_i32_declared_dtype() {
        let input = Value::Tensor(
            TensorValue::new(
                fj_core::DType::I32,
                Shape { dims: vec![4] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "2", "1"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, fj_core::DType::I32);
            assert_eq!(t.shape.dims, vec![3]);
            t.validate_dtype_consistency()
                .expect("reduce_window I32 output dtype/element invariant");
            assert_eq!(
                t.elements,
                vec![Literal::I64(3), Literal::I64(5), Literal::I64(7)]
            );
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn dense_f64_reduce_window_max_min_matches_generic() {
        // Dense and literal-backed f64 rank-2 max/min reduce_window paths must
        // be bit-identical to the generic Literal accumulator semantics, incl
        // NaN/+-inf and SAME/VALID padding.
        let rows = 7usize;
        let cols = 9usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| match i % 11 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => -0.0,
                _ => ((i as f64) * 0.37).sin() * 10.0 - (i as f64),
            })
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_f64_slice().is_some());
        let literal = Value::Tensor(
            crate::new_boxed(
                DType::F64,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
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
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };

        let reference_bits =
            |op: &str, window: (usize, usize), stride: (usize, usize), pad: &str| {
                let geometry = |input_dim: usize, win: usize, stride: usize| match pad {
                    "SAME" => reduce_window_same_geometry(
                        Primitive::ReduceWindow,
                        input_dim,
                        win,
                        stride,
                        false,
                    )
                    .unwrap(),
                    "VALID" => {
                        if input_dim >= win {
                            ((input_dim - win) / stride + 1, 0)
                        } else {
                            (0, 0)
                        }
                    }
                    _ => unreachable!("test only uses SAME/VALID padding"),
                };
                let (out_rows, pad_rows) = geometry(rows, window.0, stride.0);
                let (out_cols, pad_cols) = geometry(cols, window.1, stride.1);
                let mut out = Vec::with_capacity(out_rows * out_cols);
                for out_row in 0..out_rows {
                    let row_base = out_row * stride.0;
                    for out_col in 0..out_cols {
                        let col_base = out_col * stride.1;
                        let mut accum = reduce_window_initial_accumulator(DType::F64, op);
                        for window_row in 0..window.0 {
                            let padded_row = row_base + window_row;
                            if padded_row < pad_rows {
                                continue;
                            }
                            let input_row = padded_row - pad_rows;
                            if input_row >= rows {
                                continue;
                            }
                            let row_offset = input_row * cols;
                            for window_col in 0..window.1 {
                                let padded_col = col_base + window_col;
                                if padded_col < pad_cols {
                                    continue;
                                }
                                let input_col = padded_col - pad_cols;
                                if input_col >= cols {
                                    continue;
                                }
                                reduce_window_accumulate_literal(
                                    Primitive::ReduceWindow,
                                    op,
                                    &mut accum,
                                    Literal::from_f64(data[row_offset + input_col]),
                                )
                                .unwrap();
                            }
                        }
                        let literal = reduce_window_accumulator_literal(
                            Primitive::ReduceWindow,
                            DType::F64,
                            accum,
                        )
                        .unwrap();
                        out.push(literal.as_f64().unwrap().to_bits());
                    }
                }
                (vec![out_rows as u32, out_cols as u32], out)
            };

        for op in ["max", "min"] {
            for (win, stride, pad) in [
                ((3usize, 3usize), (1usize, 1usize), "SAME"),
                ((2, 2), (2, 2), "VALID"),
                ((3, 2), (1, 1), "VALID"),
            ] {
                let win_param = format!("{},{}", win.0, win.1);
                let stride_param = format!("{},{}", stride.0, stride.1);
                let p = rw_params_with_padding(op, &win_param, &stride_param, pad);
                let (reference_shape, reference) = reference_bits(op, win, stride, pad);
                let d = eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense), &p)
                    .unwrap();
                let l = eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&literal), &p)
                    .unwrap();
                assert_eq!(
                    d.as_tensor().unwrap().shape.dims,
                    reference_shape,
                    "{op} {win:?} {pad} shape"
                );
                assert_eq!(
                    l.as_tensor().unwrap().shape.dims,
                    reference_shape,
                    "{op} {win:?} {pad} literal shape"
                );
                assert_eq!(bits(&d), reference, "{op} {win:?} {stride:?} {pad}");
                assert_eq!(bits(&l), reference, "{op} {win:?} {stride:?} {pad} literal");
            }
        }
    }

    #[test]
    fn separable_deque_maxpool_matches_naive_dense_float_large_window() {
        // Large OVERLAPPING window (stride 1) — the deque's regime. Bit-identical to
        // the naive tap-loop dense_float path across max/min, incl NaN/±inf, for a
        // window big enough that the naive O(out·window²) clearly differs in cost.
        let (rows, cols) = (40usize, 48usize);
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| match i % 13 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => -0.0,
                _ => ((i as f64) * 0.21).cos() * 7.0 - (i as f64) * 0.01,
            })
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let canon = |bits: u64| {
            if f64::from_bits(bits).is_nan() {
                0x7ff8_0000_0000_0000
            } else {
                bits
            }
        };
        for op in ["max", "min"] {
            for (wr, wc) in [(5usize, 7usize), (8, 8), (6, 3)] {
                let out_r = rows - wr + 1;
                let out_c = cols - wc + 1;
                let p = rw_params(op, &format!("{wr},{wc}"), "1,1"); // VALID, stride 1
                let deque =
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense), &p)
                        .unwrap();
                // Naive baseline: call the dense_float tap-loop directly.
                let naive = super::eval_reduce_window_dense_float(
                    DType::F64,
                    op,
                    &data,
                    &[wr, wc],
                    &[1, 1],
                    &[1, 1],
                    &[out_r as u32, out_c as u32],
                    &[0, 0],
                    &[rows, cols],
                    &[cols, 1],
                    out_r * out_c,
                )
                .unwrap();
                let db: Vec<u64> = deque
                    .as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| canon(l.as_f64().unwrap().to_bits()))
                    .collect();
                let nb: Vec<u64> = naive
                    .as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| canon(l.as_f64().unwrap().to_bits()))
                    .collect();
                assert_eq!(
                    db, nb,
                    "{op} window {wr}x{wc} deque vs naive (NaN-canonicalized)"
                );
            }
        }
    }

    #[test]
    fn separable_deque_maxpool_i64_matches_naive_dense_i64() {
        // i64 OVERLAPPING max/min pooling: the new i64 deque path must be bit-identical
        // to the naive eval_reduce_window_dense_i64 tap loop (integer max/min is exact),
        // across windows/ops, incl i64::MIN/MAX and negatives.
        let (rows, cols) = (37usize, 41usize);
        let data: Vec<i64> = (0..rows * cols)
            .map(|i| match i % 9 {
                0 => i64::MIN,
                1 => i64::MAX,
                2 => -((i as i64) * 7),
                _ => (i as i64).wrapping_mul(2_654_435_761) ^ (i as i64),
            })
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        for op in ["max", "min"] {
            for (wr, wc) in [(4usize, 6usize), (5, 5), (7, 2)] {
                let out_r = rows - wr + 1;
                let out_c = cols - wc + 1;
                let p = rw_params(op, &format!("{wr},{wc}"), "1,1"); // VALID, stride 1 (overlapping)
                let deque =
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense), &p)
                        .unwrap();
                let naive = super::eval_reduce_window_dense_i64(
                    DType::I64,
                    op,
                    &data,
                    &[wr, wc],
                    &[1, 1],
                    &[1, 1],
                    &[out_r as u32, out_c as u32],
                    &[0, 0],
                    &[rows, cols],
                    &[cols, 1],
                    out_r * out_c,
                )
                .unwrap();
                let iv = |v: &Value| -> Vec<i64> {
                    v.as_tensor()
                        .unwrap()
                        .elements
                        .iter()
                        .map(|l| l.as_i64().unwrap())
                        .collect()
                };
                assert_eq!(
                    iv(&deque),
                    iv(&naive),
                    "i64 {op} window {wr}x{wc} deque vs naive"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_separable_deque_maxpool_i64_vs_naive() {
        use std::time::Instant;
        let (rows, cols, w) = (256usize, 256usize, 8usize);
        let data: Vec<i64> = (0..rows * cols)
            .map(|i| (i as i64).wrapping_mul(2_654_435_761))
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let out_r = rows - w + 1;
        let out_c = cols - w + 1;
        let p = rw_params("max", &format!("{w},{w}"), "1,1");
        let best = |f: &dyn Fn() -> i64| {
            let _ = f();
            let mut t = f64::MAX;
            let mut d = 0i64;
            for _ in 0..5 {
                let s = Instant::now();
                d = f();
                t = t.min(s.elapsed().as_secs_f64());
            }
            (t, d)
        };
        let (t_new, d_new) = best(&|| {
            let r =
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense), &p).unwrap();
            r.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0i64, |a, l| a ^ l.as_i64().unwrap())
        });
        let (t_old, d_old) = best(&|| {
            let r = super::eval_reduce_window_dense_i64(
                DType::I64,
                "max",
                &data,
                &[w, w],
                &[1, 1],
                &[1, 1],
                &[out_r as u32, out_c as u32],
                &[0, 0],
                &[rows, cols],
                &[cols, 1],
                out_r * out_c,
            )
            .unwrap();
            r.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0i64, |a, l| a ^ l.as_i64().unwrap())
        });
        assert_eq!(d_new, d_old, "i64 deque vs naive maxpool digest");
        println!(
            "BENCH i64 maxpool [{rows},{cols}] {w}x{w} stride1 VALID: naive={:.2}ms deque={:.2}ms speedup={:.2}x",
            t_old * 1e3,
            t_new * 1e3,
            t_old / t_new
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_separable_deque_maxpool_vs_naive() {
        use std::time::Instant;
        // Large overlapping maxpool [256,256] window 8x8 stride 1 VALID: the NEW
        // separable monotonic-deque path (O(input)) vs the naive O(out·window²)
        // tap loop (eval_reduce_window_dense_float).
        let (rows, cols, w) = (256usize, 256usize, 8usize);
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| ((i as f64) * 0.0013).sin())
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let out_r = rows - w + 1;
        let out_c = cols - w + 1;
        let p = rw_params("max", &format!("{w},{w}"), "1,1");
        let best = |f: &dyn Fn() -> u64| {
            let _ = f();
            let mut t = f64::MAX;
            let mut d = 0u64;
            for _ in 0..5 {
                let s = Instant::now();
                d = f();
                t = t.min(s.elapsed().as_secs_f64());
            }
            (t, d)
        };
        let (t_new, d_new) = best(&|| {
            let r =
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense), &p).unwrap();
            r.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |a, l| a ^ l.as_f64().unwrap().to_bits())
        });
        let (t_old, d_old) = best(&|| {
            let r = super::eval_reduce_window_dense_float(
                DType::F64,
                "max",
                &data,
                &[w, w],
                &[1, 1],
                &[1, 1],
                &[out_r as u32, out_c as u32],
                &[0, 0],
                &[rows, cols],
                &[cols, 1],
                out_r * out_c,
            )
            .unwrap();
            r.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |a, l| a ^ l.as_f64().unwrap().to_bits())
        });
        assert_eq!(d_new, d_old, "deque vs naive maxpool digest");
        println!(
            "BENCH maxpool [{rows},{cols}] {w}x{w} stride1 VALID: naive={:.2}ms deque={:.2}ms speedup={:.2}x digest={d_new:016x}",
            t_old * 1e3,
            t_new * 1e3,
            t_old / t_new
        );
    }

    /// Atrous pooling (window_dilation > 1) now takes the dense f64/i64 stencil path
    /// (gated on no_base_dilation only). The dilated-tap dense path must be BIT-FOR-BIT
    /// identical to the generic per-`Literal` loop, which still handles window_dilation
    /// for the boxed input (whose `as_*_slice` is None). Covers max/min/sum, f64 + i64,
    /// SAME + VALID, single and per-axis dilation, incl NaN/±inf/-0.0.
    #[test]
    #[allow(clippy::identity_op)] // `/1`/`*1` keep the stride-general formula explicit
    fn dense_reduce_window_window_dilation_matches_generic() {
        let (rows, cols) = (9usize, 11usize);
        let data64: Vec<f64> = (0..rows * cols)
            .map(|i| match i % 11 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => -0.0,
                _ => ((i as f64) * 0.41).cos() * 7.0 - (i as f64) * 0.5,
            })
            .collect();
        // F64 forces the generic path via a boxed input (`new` with F64Bits literals
        // keeps `as_f64_slice` == None); I64 cannot (I64 `new` densifies), so the i64
        // dilated path — which shares the IDENTICAL tap stencil — is covered against a
        // hand-computed reference below.
        let f64_dense = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data64.clone(),
            )
            .unwrap(),
        );
        let f64_boxed = Value::Tensor(
            crate::new_boxed(
                DType::F64,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data64.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        assert!(
            f64_boxed
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_none()
        );
        assert!(
            f64_dense
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_some()
        );

        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };

        for op in ["max", "min", "sum"] {
            for (win, dil, pad) in [
                ("3,3", "2,2", "VALID"),
                ("2,3", "3,2", "VALID"),
                ("3,3", "2,2", "SAME"),
                ("2,2", "2,1", "SAME"),
            ] {
                let mut p = rw_params_with_padding(op, win, "1,1", pad);
                p.insert("window_dilation".to_owned(), dil.to_owned());
                let d = eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(&f64_dense),
                    &p,
                )
                .unwrap();
                let l = eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(&f64_boxed),
                    &p,
                )
                .unwrap();
                assert_eq!(
                    d.as_tensor().unwrap().shape.dims,
                    l.as_tensor().unwrap().shape.dims,
                    "{op} win={win} dil={dil} {pad} shape"
                );
                assert_eq!(
                    bits(&d),
                    bits(&l),
                    "{op} win={win} dil={dil} {pad} dense!=generic"
                );
            }
        }

        // I64 dilated path vs a hand-computed reference (VALID, window 3x3, dilation 2x2).
        let datai: Vec<i64> = (0..rows * cols)
            .map(|i| (i as i64 * 37 - 211) % 97)
            .collect();
        let i64_dense = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                datai.clone(),
            )
            .unwrap(),
        );
        let (wr, wc, dr, dc) = (3usize, 3usize, 2usize, 2usize);
        let eff_r = (wr - 1) * dr + 1;
        let eff_c = (wc - 1) * dc + 1;
        let (out_r, out_c) = ((rows - eff_r) / 1 + 1, (cols - eff_c) / 1 + 1);
        for op in ["max", "min", "sum"] {
            let mut p = rw_params_with_padding(op, "3,3", "1,1", "VALID");
            p.insert("window_dilation".to_owned(), "2,2".to_owned());
            let got = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&i64_dense),
                &p,
            )
            .unwrap();
            let got_vals: Vec<i64> = got
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| {
                    if let Literal::I64(x) = l {
                        *x
                    } else {
                        panic!()
                    }
                })
                .collect();
            let mut want = Vec::with_capacity(out_r * out_c);
            for orow in 0..out_r {
                for ocol in 0..out_c {
                    let mut acc: i64 = match op {
                        "max" => i64::MIN,
                        "min" => i64::MAX,
                        _ => 0,
                    };
                    for tr in 0..wr {
                        for tc in 0..wc {
                            let v = datai[(orow + tr * dr) * cols + (ocol + tc * dc)];
                            acc = match op {
                                "max" => acc.max(v),
                                "min" => acc.min(v),
                                _ => acc.wrapping_add(v),
                            };
                        }
                    }
                    want.push(acc);
                }
            }
            assert_eq!(
                got_vals, want,
                "i64 {op} dilated reduce_window vs hand reference"
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_reduce_window_window_dilation_dense_vs_generic() {
        use std::time::Instant;
        let (rows, cols) = (512usize, 512usize);
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| ((i as f64) * 0.013).sin())
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let mut p = rw_params_with_padding("max", "5,5", "1,1", "VALID");
        p.insert("window_dilation".to_owned(), "3,3".to_owned());
        let run = |v: &Value| {
            let mut acc = 0u64;
            for _ in 0..20 {
                let o =
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(v), &p).unwrap();
                acc = acc.wrapping_add(o.as_tensor().unwrap().elements.len() as u64);
            }
            acc
        };
        run(&dense);
        run(&boxed); // warm
        let t = Instant::now();
        let _ = run(&boxed);
        let generic = t.elapsed();
        let t = Instant::now();
        let _ = run(&dense);
        let dense_t = t.elapsed();
        println!(
            "BENCH reduce_window window_dilation [512x512] win5x5 dil3: generic={:.2}ms dense={:.2}ms speedup={:.2}x",
            generic.as_secs_f64() * 1e3 / 20.0,
            dense_t.as_secs_f64() * 1e3 / 20.0,
            generic.as_secs_f64() / dense_t.as_secs_f64()
        );
    }

    /// Real 2D pooling on rank-4 NHWC (and all f32 pooling) routes through the new
    /// dense float path — neither is reached by the rank-2-only F64 paths. It must
    /// be BIT-FOR-BIT identical to the generic per-`Literal` loop (forced via a
    /// boxed input whose `as_*_slice` is None), across max/min/sum, f64 AND f32,
    /// SAME (border-heavy) + VALID padding, incl NaN/±inf/-0.0.
    #[test]
    fn dense_reduce_window_rank4_pooling_matches_generic() {
        let (n, h, w, c) = (2usize, 7usize, 9usize, 3usize);
        let total = n * h * w * c;
        let data64: Vec<f64> = (0..total)
            .map(|i| match i % 13 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => -0.0,
                _ => ((i as f64) * 0.19).sin() * 8.0 - (i as f64) * 0.1,
            })
            .collect();
        let dims = vec![n as u32, h as u32, w as u32, c as u32];
        let dense64 = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, data64.clone()).unwrap(),
        );
        let boxed64 = Value::Tensor(
            crate::new_boxed(
                DType::F64,
                Shape { dims: dims.clone() },
                data64.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        assert!(
            dense64
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_some()
        );
        assert!(
            boxed64
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_none()
        );
        let data32: Vec<f32> = data64.iter().map(|&v| v as f32).collect();
        let dense32 = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, data32.clone()).unwrap(),
        );
        let boxed32 = Value::Tensor(
            crate::new_boxed(
                DType::F32,
                Shape { dims: dims.clone() },
                data32.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        assert!(
            dense32
                .as_tensor()
                .unwrap()
                .elements
                .as_f32_slice()
                .is_some()
        );
        assert!(
            boxed32
                .as_tensor()
                .unwrap()
                .elements
                .as_f32_slice()
                .is_none()
        );
        // Canonicalize NaN: `inf + (-inf)` (and friends) produces a NaN whose SIGN
        // BIT is IEEE-754-unspecified and differs between two valid summation orders
        // (the dense tight loop vs the generic per-Literal fold) — not a parity
        // concern (JAX/XLA don't specify it either, and reduce_window has no golden
        // digests). Map every NaN to one pattern before comparing, matching the
        // project's conv/const-fold NaN-bit-test precedent.
        let bits64 = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| {
                    let f = l.as_f64().unwrap();
                    if f.is_nan() {
                        0x7ff8_0000_0000_0000
                    } else {
                        f.to_bits()
                    }
                })
                .collect()
        };
        let bits32 = |v: &Value| -> Vec<u32> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F32Bits(b) => {
                        if f32::from_bits(*b).is_nan() {
                            0x7fc0_0000
                        } else {
                            *b
                        }
                    }
                    o => panic!("expected f32, got {o:?}"),
                })
                .collect()
        };
        for op in ["max", "min", "sum"] {
            for (win, stride, pad) in [
                ("1,3,3,1", "1,1,1,1", "SAME"), // border-heavy (exercises border path)
                ("1,2,2,1", "1,2,2,1", "VALID"), // standard 2x2 stride-2 pool
                ("1,2,3,1", "1,1,1,1", "VALID"), // asymmetric window
            ] {
                let p = rw_params_with_padding(op, win, stride, pad);
                let d64 =
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense64), &p)
                        .unwrap();
                let l64 =
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&boxed64), &p)
                        .unwrap();
                assert_eq!(
                    d64.as_tensor().unwrap().shape.dims,
                    l64.as_tensor().unwrap().shape.dims,
                    "f64 {op} {win} {pad} shape"
                );
                assert_eq!(bits64(&d64), bits64(&l64), "f64 {op} {win} {stride} {pad}");
                assert!(
                    d64.as_tensor().unwrap().elements.as_f64_slice().is_some(),
                    "f64 output must be dense"
                );
                let d32 =
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense32), &p)
                        .unwrap();
                let l32 =
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&boxed32), &p)
                        .unwrap();
                assert_eq!(bits32(&d32), bits32(&l32), "f32 {op} {win} {stride} {pad}");
                assert!(
                    d32.as_tensor().unwrap().elements.as_f32_slice().is_some(),
                    "f32 output must be dense"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_maxpool_dense_vs_generic() {
        use std::time::Instant;
        let (n, h, w, c) = (16usize, 64usize, 64usize, 32usize); // [16,64,64,32] -> 2x2 s2 pool
        let total = n * h * w * c;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let dims = vec![n as u32, h as u32, w as u32, c as u32];
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
        let p = rw_params_with_padding("max", "1,2,2,1", "1,2,2,1", "VALID");
        let time = |input: &Value| {
            let _ =
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(input), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(input), &p)
                    .unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 maxpool [{n},{h},{w},{c}] 2x2 s2: generic(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn dense_f64_reduce_window_sum_matches_generic() {
        // Dense f64 rank-2 SUM reduce_window general path (non-3x3-SAME shapes,
        // which hit the dense loop) must be bit-identical to the generic Literal
        // path, incl NaN/+-inf and SAME/VALID padding.
        let rows = 7usize;
        let cols = 9usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| match i % 11 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                3 => -0.0,
                _ => ((i as f64) * 0.37).sin() * 10.0 - (i as f64),
            })
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_f64_slice().is_some());
        let literal = Value::Tensor(
            crate::new_boxed(
                DType::F64,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
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
        let bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };
        for (win, stride, pad) in [
            ("2,2", "2,2", "VALID"),
            ("3,2", "1,1", "VALID"),
            ("2,3", "1,1", "SAME"),
        ] {
            let p = rw_params_with_padding("sum", win, stride, pad);
            let d =
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense), &p).unwrap();
            let l = eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&literal), &p)
                .unwrap();
            assert_eq!(
                d.as_tensor().unwrap().shape.dims,
                l.as_tensor().unwrap().shape.dims,
                "sum {win} {pad} shape"
            );
            assert_eq!(bits(&d), bits(&l), "sum {win} {stride} {pad}");
        }
    }

    #[test]
    fn reduce_window_max_preserves_bf16_literal_dtype() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::BF16,
                Shape { dims: vec![4] },
                vec![
                    Literal::from_bf16_f32(1.0),
                    Literal::from_bf16_f32(3.0),
                    Literal::from_bf16_f32(2.0),
                    Literal::from_bf16_f32(4.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("max", "2", "1"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::BF16);
            assert_eq!(t.shape.dims, vec![3]);
            t.validate_dtype_consistency()
                .expect("reduce_window BF16 output dtype/element invariant");
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 3.0, 4.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_sum_preserves_i64_literal_dtype_and_wraps() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3] },
                vec![Literal::I64(i64::MAX), Literal::I64(1), Literal::I64(2)],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "2", "1"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::I64);
            assert_eq!(t.shape.dims, vec![2]);
            assert_eq!(t.elements, vec![Literal::I64(i64::MIN), Literal::I64(3)]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_min_preserves_u32_literal_dtype() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::U32,
                Shape { dims: vec![4] },
                vec![
                    Literal::U32(9),
                    Literal::U32(4),
                    Literal::U32(7),
                    Literal::U32(2),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("min", "2", "1"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::U32);
            assert_eq!(t.shape.dims, vec![3]);
            assert_eq!(
                t.elements,
                vec![Literal::U32(4), Literal::U32(4), Literal::U32(2)]
            );
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_max_preserves_u64_literal_dtype() {
        let high = 9_007_199_254_740_995_u64;
        let input = Value::Tensor(
            TensorValue::new(
                DType::U64,
                Shape { dims: vec![3] },
                vec![
                    Literal::U64(9_007_199_254_740_993),
                    Literal::U64(high),
                    Literal::U64(5),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("max", "2", "1"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::U64);
            assert_eq!(t.shape.dims, vec![2]);
            assert_eq!(t.elements, vec![Literal::U64(high), Literal::U64(high)]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_sum_preserves_bool_literal_dtype_as_or() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![4] },
                vec![
                    Literal::Bool(false),
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(false),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "2", "1"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::Bool);
            assert_eq!(t.shape.dims, vec![3]);
            assert_eq!(
                t.elements,
                vec![
                    Literal::Bool(true),
                    Literal::Bool(true),
                    Literal::Bool(false),
                ]
            );
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_min_preserves_bool_literal_dtype_as_and() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![4] },
                vec![
                    Literal::Bool(true),
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("min", "2", "1"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::Bool);
            assert_eq!(t.shape.dims, vec![3]);
            assert_eq!(
                t.elements,
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(false),
                ]
            );
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_sum_preserves_complex128_literal_dtype() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape { dims: vec![4] },
                vec![
                    Literal::from_complex128(1.0, 2.0),
                    Literal::from_complex128(3.0, -4.0),
                    Literal::from_complex128(-2.0, 0.5),
                    Literal::from_complex128(0.0, 1.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "2", "1"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::Complex128);
            assert_eq!(t.shape.dims, vec![3]);
            assert_eq!(t.elements[0].as_complex128(), Some((4.0, -2.0)));
            assert_eq!(t.elements[1].as_complex128(), Some((1.0, -3.5)));
            assert_eq!(t.elements[2].as_complex128(), Some((-2.0, 1.5)));
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_max_preserves_complex64_literal_dtype() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_complex64(1.0, 2.0),
                    Literal::from_complex64(3.0, 0.0),
                    Literal::from_complex64(3.0, 4.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("max", "2", "1"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.dtype, DType::Complex64);
            assert_eq!(t.shape.dims, vec![2]);
            assert_eq!(t.elements[0].as_complex64(), Some((3.0, 0.0)));
            assert_eq!(t.elements[1].as_complex64(), Some((3.0, 4.0)));
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_max_1d() {
        // [1, 3, 2, 5, 4], window=2, stride=1, valid => [3, 3, 5, 5]
        let input = Value::vector_f64(&[1.0, 3.0, 2.0, 5.0, 4.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("max", "2", "1"),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 3.0, 5.0, 5.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_max_1d_stride2() {
        // [1, 3, 2, 5, 4, 6], window=2, stride=2, valid => [3, 5, 6]
        let input = Value::vector_f64(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("max", "2", "2"),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 5.0, 6.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_sum_2d() {
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [7, 8, 9]]
        // window=2x2, stride=1x1 => [[12, 16], [24, 28]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3, 3] },
                (1..=9).map(|v| Literal::from_f64(v as f64)).collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "2,2", "1,1"),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![2, 2]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![12.0, 16.0, 24.0, 28.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn dense_i32_separable_maxmin_matches_generic() {
        // Large-window i32 max/min reduce_window now uses the window-independent deque
        // (was the O(output·∏window) general dense). max/min select an input value →
        // valid i32, no wrap; dtype must stay I32 and match the boxed generic path.
        let (rows, cols) = (40usize, 48usize);
        let data: Vec<i64> = (0..rows * cols)
            .map(|i| i64::from((i as i32).wrapping_mul(2_654_435).wrapping_sub(7)))
            .collect();
        let dense = Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.iter().map(|&v| Literal::I64(v)).collect(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I32,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::I64(v)).collect()),
            )
            .unwrap(),
        );
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
        for op in ["max", "min"] {
            // window 9x9 (∏=81 > 2·∑=36) ⇒ deque path.
            let p = rw_params_with_padding(op, "9,9", "1,1", "VALID");
            let d =
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense), &p).unwrap();
            let b =
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&boxed), &p).unwrap();
            assert_eq!(
                d.as_tensor().unwrap().dtype,
                DType::I32,
                "i32 {op} deque dtype"
            );
            assert_eq!(geti(&d), geti(&b), "i32 {op} deque != generic");
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_i32_reduce_window_deque_vs_dense() {
        use std::time::Instant;
        // Honest deque-vs-general-dense A/B for large-window i32 max pooling, calling
        // both internal kernels directly on identical i32 data (the dispatch picks the
        // deque, so this isolates the window-independence win over the O(window) dense).
        let (rows, cols, win) = (512usize, 512usize, 17usize);
        let src: Vec<i64> = (0..rows * cols)
            .map(|i| i64::from((i as i64).wrapping_mul(2_654_435_761) as i32))
            .collect();
        let in_dims = vec![rows, cols];
        let in_strides = vec![cols, 1usize];
        let wdims = vec![win, win];
        let strides = vec![1usize, 1];
        let out = vec![(rows - win + 1) as u32, (cols - win + 1) as u32];
        let pad = vec![0usize, 0];
        let total = (rows - win + 1) * (cols - win + 1);
        let best = |mut f: Box<dyn FnMut() -> usize>| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let (s1, w1, st1, o1, p1, i1, is1) = (
            src.clone(),
            wdims.clone(),
            strides.clone(),
            out.clone(),
            pad.clone(),
            in_dims.clone(),
            in_strides.clone(),
        );
        let od = vec![1usize, 1];
        let dense_t = best(Box::new(move || {
            super::eval_reduce_window_dense_i64(
                fj_core::DType::I32,
                "max",
                &s1,
                &w1,
                &st1,
                &od,
                &o1,
                &p1,
                &i1,
                &is1,
                total,
            )
            .unwrap();
            total
        }));
        let (s2, i2, w2, st2, p2) = (
            src.clone(),
            in_dims.clone(),
            wdims.clone(),
            strides.clone(),
            pad.clone(),
        );
        let ou = vec![(rows - win + 1), (cols - win + 1)];
        let deque_t = best(Box::new(move || {
            let _ = super::reduce_window_separable_maxmin_i64(&s2, &i2, &w2, &st2, &p2, &ou, true);
            total
        }));
        println!(
            "BENCH i32 reduce_window max [{rows}x{cols}] win{win}x{win}: dense={:.2}ms deque={:.2}ms speedup={:.2}x",
            dense_t * 1e3,
            deque_t * 1e3,
            dense_t / deque_t
        );
    }

    #[test]
    fn dense_i32_sum_sat_matches_generic() {
        // Large-window i32 SUM reduce_window now uses the window-independent SAT (was the
        // O(window) general dense). The i64-ring SAT sums are re-tagged I32 and wrapped
        // mod 2^32 by the chokepoint. Dense (densified → SAT) must match boxed-generic,
        // rank-2 AND rank-1, with overflowing values, dtype preserved.
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
        let mk_d = |dims: Vec<u32>, d: &[i64]| {
            Value::Tensor(
                TensorValue::new(
                    DType::I32,
                    Shape { dims },
                    d.iter().map(|&v| Literal::I64(v)).collect(),
                )
                .unwrap(),
            )
        };
        let mk_b = |dims: Vec<u32>, d: &[i64]| {
            Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    DType::I32,
                    Shape { dims },
                    fj_core::LiteralBuffer::new(d.iter().map(|&v| Literal::I64(v)).collect()),
                )
                .unwrap(),
            )
        };
        // rank-2: 40x40, window 8x8 (∏=64 ≥ 16 ⇒ SAT). Big magnitudes ⇒ sums overflow i32.
        let (rows, cols) = (40usize, 40usize);
        let d2: Vec<i64> = (0..rows * cols)
            .map(|i| i64::from((i as i32).wrapping_mul(5_000_003)))
            .collect();
        let p2 = rw_params_with_padding("sum", "8,8", "1,1", "VALID");
        let got = eval_primitive(
            Primitive::ReduceWindow,
            std::slice::from_ref(&mk_d(vec![rows as u32, cols as u32], &d2)),
            &p2,
        )
        .unwrap();
        let exp = eval_primitive(
            Primitive::ReduceWindow,
            std::slice::from_ref(&mk_b(vec![rows as u32, cols as u32], &d2)),
            &p2,
        )
        .unwrap();
        assert_eq!(
            got.as_tensor().unwrap().dtype,
            DType::I32,
            "rank2 SAT dtype I32"
        );
        assert_eq!(geti(&got), geti(&exp), "rank2 i32 SAT != generic");
        // rank-1: 4000, window 32 (≥16 ⇒ in_sum_sat).
        let n = 4000usize;
        let d1: Vec<i64> = (0..n)
            .map(|i| i64::from((i as i32).wrapping_mul(6_000_011)))
            .collect();
        let p1 = rw_params_with_padding("sum", "32", "1", "VALID");
        let got = eval_primitive(
            Primitive::ReduceWindow,
            std::slice::from_ref(&mk_d(vec![n as u32], &d1)),
            &p1,
        )
        .unwrap();
        let exp = eval_primitive(
            Primitive::ReduceWindow,
            std::slice::from_ref(&mk_b(vec![n as u32], &d1)),
            &p1,
        )
        .unwrap();
        assert_eq!(
            got.as_tensor().unwrap().dtype,
            DType::I32,
            "rank1 SAT dtype I32"
        );
        assert_eq!(geti(&got), geti(&exp), "rank1 i32 SAT != generic");
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_i32_sum_sat_vs_dense() {
        use std::time::Instant;
        // Honest SAT-vs-general-dense A/B for large-window i32 sum pooling (both kernels
        // called directly on identical i32 data), isolating the window-independence win.
        let (rows, cols, win) = (512usize, 512usize, 17usize);
        let src: Vec<i64> = (0..rows * cols)
            .map(|i| i64::from((i as i64).wrapping_mul(2_654_435_761) as i32))
            .collect();
        let in_dims = vec![rows, cols];
        let in_strides = vec![cols, 1usize];
        let wdims = vec![win, win];
        let strides = vec![1usize, 1];
        let out = vec![(rows - win + 1) as u32, (cols - win + 1) as u32];
        let pad = vec![0usize, 0];
        let total = (rows - win + 1) * (cols - win + 1);
        let best = |mut f: Box<dyn FnMut() -> usize>| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let (s1, w1, st1, o1, p1, i1, is1) = (
            src.clone(),
            wdims.clone(),
            strides.clone(),
            out.clone(),
            pad.clone(),
            in_dims.clone(),
            in_strides.clone(),
        );
        let od = vec![1usize, 1];
        let dense_t = best(Box::new(move || {
            super::eval_reduce_window_dense_i64(
                fj_core::DType::I32,
                "sum",
                &s1,
                &w1,
                &st1,
                &od,
                &o1,
                &p1,
                &i1,
                &is1,
                total,
            )
            .unwrap();
            total
        }));
        let (s2, w2, st2, o2, p2b, i2) = (
            src.clone(),
            wdims.clone(),
            strides.clone(),
            out.clone(),
            pad.clone(),
            in_dims.clone(),
        );
        let sat_t = best(Box::new(move || {
            super::eval_reduce_window_rank2_i64_sum_sat(&s2, &w2, &st2, &o2, &p2b, &i2, total)
                .unwrap();
            total
        }));
        println!(
            "BENCH i32 reduce_window sum [{rows}x{cols}] win{win}x{win}: dense={:.2}ms sat={:.2}ms speedup={:.2}x",
            dense_t * 1e3,
            sat_t * 1e3,
            dense_t / sat_t
        );
    }

    #[test]
    #[allow(clippy::identity_op)] // `/1`/`*1` keep the stride-general formula explicit
    fn dense_reduce_window_i32_matches_reference() {
        // i32 (JAX's default int) now routes through the dense i64 stencil (was the
        // generic per-`Literal` path). i32 densifies (`as_i64_slice` is Some) so it can't
        // be forced onto the generic path by boxing — verify against a hand-computed i32
        // reference instead. Values straddle i32::MAX so the SUM overflows and must wrap
        // mod 2^32 (via the eval_primitive narrow chokepoint); max/min stay in range.
        let (rows, cols) = (8usize, 10usize);
        let datai: Vec<i64> = (0..rows * cols)
            .map(|i| {
                // large magnitudes near i32::MAX with sign variation → 9-tap sums overflow
                let v = (i as i64 * 521_000_003 - 1_900_000_000) % (i64::from(i32::MAX) + 1);
                i64::from(v as i32)
            })
            .collect();
        let dense = Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                datai.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        assert_eq!(dense.as_tensor().unwrap().dtype, DType::I32);
        assert!(dense.as_tensor().unwrap().elements.as_i64_slice().is_some());

        for (win, dil, pad) in [
            ("3,3", "1,1", "VALID"),
            ("3,3", "2,2", "VALID"),
            ("2,2", "1,1", "SAME"),
        ] {
            let (wr, wc): (usize, usize) = {
                let mut it = win.split(',').map(|s| s.parse().unwrap());
                (it.next().unwrap(), it.next().unwrap())
            };
            let (dr, dc): (usize, usize) = {
                let mut it = dil.split(',').map(|s| s.parse().unwrap());
                (it.next().unwrap(), it.next().unwrap())
            };
            for op in ["sum", "max", "min"] {
                let mut p = rw_params_with_padding(op, win, "1,1", pad);
                p.insert("window_dilation".to_owned(), dil.to_owned());
                let got = eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense), &p)
                    .unwrap();
                let gt = got.as_tensor().unwrap();
                assert_eq!(
                    gt.dtype,
                    DType::I32,
                    "{op} {win} {dil} {pad}: dtype preserved"
                );
                let got_vals: Vec<i32> = gt
                    .elements
                    .iter()
                    .map(|l| {
                        if let Literal::I64(x) = l {
                            *x as i32
                        } else {
                            panic!()
                        }
                    })
                    .collect();

                // Hand reference (SAME pad_low from the shared geometry helper).
                let eff_r = (wr - 1) * dr + 1;
                let eff_c = (wc - 1) * dc + 1;
                let (out_r, plr) = match pad {
                    "VALID" => ((rows - eff_r) / 1 + 1, 0usize),
                    _ => {
                        reduce_window_same_geometry(Primitive::ReduceWindow, rows, eff_r, 1, false)
                            .unwrap()
                    }
                };
                let (out_c, plc) = match pad {
                    "VALID" => ((cols - eff_c) / 1 + 1, 0usize),
                    _ => {
                        reduce_window_same_geometry(Primitive::ReduceWindow, cols, eff_c, 1, false)
                            .unwrap()
                    }
                };
                let mut want = Vec::with_capacity(out_r * out_c);
                for orow in 0..out_r {
                    for ocol in 0..out_c {
                        let mut acc: i32 = match op {
                            "max" => i32::MIN,
                            "min" => i32::MAX,
                            _ => 0,
                        };
                        for tr in 0..wr {
                            for tc in 0..wc {
                                let pr = orow * 1 + tr * dr;
                                let pc = ocol * 1 + tc * dc;
                                if pr < plr || pc < plc {
                                    continue;
                                }
                                let (ir, ic) = (pr - plr, pc - plc);
                                if ir >= rows || ic >= cols {
                                    continue;
                                }
                                let v = datai[ir * cols + ic] as i32;
                                acc = match op {
                                    "max" => acc.max(v),
                                    "min" => acc.min(v),
                                    _ => acc.wrapping_add(v),
                                };
                            }
                        }
                        want.push(acc);
                    }
                }
                assert_eq!(
                    got_vals, want,
                    "i32 {op} win={win} dil={dil} {pad} vs reference"
                );
            }
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_reduce_window_i32_dense_vs_generic() {
        use std::time::Instant;
        // i32 sum-pool [512,512] window 5x5 stride 1 VALID. Dense i64 stencil (production,
        // now reached for I32) vs the generic per-`Literal` gather + accumulate loop.
        let (n, w) = (512usize, 5usize);
        let raw: Vec<i64> = (0..n * n)
            .map(|i| i64::from(((i as i64).wrapping_mul(2_654_435_761) ^ 0x5555) as i32))
            .collect();
        let dense = Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape {
                    dims: vec![n as u32, n as u32],
                },
                raw.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let params = rw_params("sum", "5,5", "1,1");
        let out_n = n - w + 1;
        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let t_ref = dense.clone();
        let generic = best(Box::new(move || {
            let t = t_ref.as_tensor().unwrap();
            let mut out: Vec<Literal> = Vec::with_capacity(out_n * out_n);
            for oy in 0..out_n {
                for ox in 0..out_n {
                    let mut accum = reduce_window_initial_accumulator(DType::I32, "sum");
                    for wy in 0..w {
                        for wx in 0..w {
                            reduce_window_accumulate_literal(
                                Primitive::ReduceWindow,
                                "sum",
                                &mut accum,
                                t.elements[(oy + wy) * n + (ox + wx)],
                            )
                            .unwrap();
                        }
                    }
                    out.push(
                        reduce_window_accumulator_literal(
                            Primitive::ReduceWindow,
                            DType::I32,
                            accum,
                        )
                        .unwrap(),
                    );
                }
            }
            out.len() as u64
        }));
        let t_dense = dense.clone();
        let dense_t = best(Box::new(move || {
            let o = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&t_dense),
                &params,
            )
            .unwrap();
            o.as_tensor().unwrap().elements.len() as u64
        }));
        println!(
            "BENCH reduce_window i32 sum [512x512] win5x5: generic={:.2}ms dense={:.2}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_dense_i64_vs_generic() {
        use std::time::Instant;
        // 2D i64 sum-pool [512,512] window 3x3 stride 1 (VALID -> [510,510], all
        // interior). Compares the dense i64 fast path (production) against the
        // generic per-`Literal` gather + reduce_window_accumulate_literal loop it
        // replaces (both build an output buffer, isolating the Literal
        // materialization + string-dispatched accumulate).
        let (n, w) = (512usize, 3usize);
        let raw: Vec<i64> = (0..n * n)
            .map(|i| (i as i64).wrapping_mul(2_654_435_761) ^ 0x5555)
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![n as u32, n as u32],
                },
                raw,
            )
            .unwrap(),
        );
        let params = rw_params("sum", "3,3", "1,1");
        let out_n = n - w + 1;

        let best = |mut f: Box<dyn FnMut() -> i64>| {
            f();
            let mut b = f64::MAX;
            let mut dg = 0i64;
            for _ in 0..5 {
                let t = Instant::now();
                dg = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, dg)
        };

        // Generic reference: per-tap Literal gather + the SAME accumulate helpers
        // the generic loop calls, building a Vec<Literal> output like the real path.
        let t_ref = tensor.clone();
        let (t_gen, d_gen) = best(Box::new(move || {
            let t = t_ref.as_tensor().unwrap();
            let mut out_elems: Vec<Literal> = Vec::with_capacity(out_n * out_n);
            for oy in 0..out_n {
                for ox in 0..out_n {
                    let mut accum = reduce_window_initial_accumulator(DType::I64, "sum");
                    for wy in 0..w {
                        for wx in 0..w {
                            let flat = (oy + wy) * n + (ox + wx);
                            reduce_window_accumulate_literal(
                                Primitive::ReduceWindow,
                                "sum",
                                &mut accum,
                                t.elements[flat],
                            )
                            .unwrap();
                        }
                    }
                    out_elems.push(
                        reduce_window_accumulator_literal(
                            Primitive::ReduceWindow,
                            DType::I64,
                            accum,
                        )
                        .unwrap(),
                    );
                }
            }
            out_elems.iter().fold(0i64, |acc, l| match l {
                Literal::I64(v) => acc.wrapping_add(*v),
                _ => acc,
            })
        }));

        let t_fast = tensor.clone();
        let (t_dense, d_dense) = best(Box::new(move || {
            let out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&t_fast),
                &params,
            )
            .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0i64, |acc, l| match l {
                    Literal::I64(v) => acc.wrapping_add(*v),
                    _ => acc,
                })
        }));

        assert_eq!(
            d_gen, d_dense,
            "dense i64 reduce_window checksum must match generic"
        );
        println!(
            "BENCH reduce_window i64 sum([{n},{n}],win=3x3,stride=1): generic={:.4}ms dense={:.4}ms speedup={:.2}x",
            t_gen * 1e3,
            t_dense * 1e3,
            t_gen / t_dense,
        );
    }

    #[test]
    fn reduce_window_i64_sum_sat_matches_per_window() {
        // Large-window i64 sum reduce_window (SAT path) must equal an independent
        // per-window in-bounds wrapping-sum reference, across VALID/SAME padding
        // and strides. Exercises the integral-image + 4-corner inclusion-exclusion
        // including boundary clamping.
        let (rows, cols) = (23usize, 19usize);
        let raw: Vec<i64> = (0..rows * cols)
            .map(|i| (i as i64).wrapping_mul(2_654_435_761) ^ 0x5a5a)
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                raw.clone(),
            )
            .unwrap(),
        );
        // (win_r, win_c, stride_r, stride_c, padding) — all with ∏window ≥ 16 to hit SAT.
        let cases = [
            (4usize, 4usize, 1usize, 1usize, "VALID"),
            (5, 5, 2, 2, "VALID"),
            (5, 5, 1, 1, "SAME"),
            (7, 3, 2, 1, "SAME"),
            (4, 5, 3, 2, "VALID"),
        ];
        for (wr, wc, sr, sc, pad) in cases {
            let params =
                rw_params_with_padding("sum", &format!("{wr},{wc}"), &format!("{sr},{sc}"), pad);
            let out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&tensor),
                &params,
            )
            .unwrap();
            let out_t = out.as_tensor().unwrap();
            let (out_rows, out_cols) = (out_t.shape.dims[0] as usize, out_t.shape.dims[1] as usize);
            // SAME pad_low = floor(((out-1)*stride + win - in)/2); VALID = 0.
            let pad_r = if pad == "SAME" {
                (((out_rows - 1) * sr + wr).saturating_sub(rows)) / 2
            } else {
                0
            };
            let pad_c = if pad == "SAME" {
                (((out_cols - 1) * sc + wc).saturating_sub(cols)) / 2
            } else {
                0
            };
            let got: Vec<i64> = out_t
                .elements
                .iter()
                .map(|l| match l {
                    Literal::I64(v) => *v,
                    o => panic!("expected I64, got {o:?}"),
                })
                .collect();
            // Independent per-window in-bounds wrapping-sum reference.
            let mut expect = Vec::with_capacity(out_rows * out_cols);
            for or in 0..out_rows {
                for oc in 0..out_cols {
                    let mut acc = 0i64;
                    for dr in 0..wr {
                        let pr = or * sr + dr;
                        if pr < pad_r {
                            continue;
                        }
                        let ir = pr - pad_r;
                        if ir >= rows {
                            continue;
                        }
                        for dc in 0..wc {
                            let pc = oc * sc + dc;
                            if pc < pad_c {
                                continue;
                            }
                            let ic = pc - pad_c;
                            if ic >= cols {
                                continue;
                            }
                            acc = acc.wrapping_add(raw[ir * cols + ic]);
                        }
                    }
                    expect.push(acc);
                }
            }
            assert_eq!(
                got, expect,
                "SAT i64 sum win={wr}x{wc} stride={sr}x{sc} pad={pad} != per-window reference"
            );
        }
    }

    #[test]
    #[allow(clippy::type_complexity)] // explicit case-table tuple reads clearer inline
    fn reduce_window_in_sum_sat_matches_generic() {
        // General-rank i64 SAT sum (rank 1, 3, 4) must match the generic per-window
        // path element-for-element across padding/strides. The SAT path is gated
        // ∏window ≥ 16; the generic eval (which these compare against via a second
        // tensor that forces the slow path) is the ground truth. Here both go
        // through eval_primitive; the SAT output is checked vs a hand reference.
        let ref_sum = |raw: &[i64],
                       in_dims: &[usize],
                       win: &[usize],
                       strd: &[usize],
                       pad: &[usize],
                       out_dims: &[usize]|
         -> Vec<i64> {
            let rank = in_dims.len();
            let mut in_strides = vec![1usize; rank];
            let mut acc = 1;
            for d in (0..rank).rev() {
                in_strides[d] = acc;
                acc *= in_dims[d];
            }
            let total_out: usize = out_dims.iter().product();
            let mut out = Vec::with_capacity(total_out);
            let mut oi = vec![0usize; rank];
            for _ in 0..total_out {
                let mut s = 0i64;
                let win_total: usize = win.iter().product();
                let mut wi = vec![0usize; rank];
                for _ in 0..win_total {
                    let mut ok = true;
                    let mut idx = 0usize;
                    for d in 0..rank {
                        let p = oi[d] * strd[d] + wi[d];
                        if p < pad[d] {
                            ok = false;
                            break;
                        }
                        let ic = p - pad[d];
                        if ic >= in_dims[d] {
                            ok = false;
                            break;
                        }
                        idx += ic * in_strides[d];
                    }
                    if ok {
                        s = s.wrapping_add(raw[idx]);
                    }
                    // odometer over window
                    let mut c = true;
                    for d in (0..rank).rev() {
                        if c {
                            wi[d] += 1;
                            if wi[d] >= win[d] {
                                wi[d] = 0;
                            } else {
                                c = false;
                            }
                        }
                    }
                }
                out.push(s);
                let mut c = true;
                for d in (0..rank).rev() {
                    if c {
                        oi[d] += 1;
                        if oi[d] >= out_dims[d] {
                            oi[d] = 0;
                        } else {
                            c = false;
                        }
                    }
                }
            }
            out
        };
        // (in_dims, win, stride, padding-str)
        let cases: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, &str)> = vec![
            (vec![64], vec![16], vec![1], "VALID"),
            (vec![50], vec![20], vec![3], "SAME"),
            (vec![9, 9, 3], vec![4, 4, 1], vec![1, 1, 1], "VALID"),
            (
                vec![2, 8, 8, 3],
                vec![1, 4, 4, 1],
                vec![1, 2, 2, 1],
                "VALID",
            ),
        ];
        for (in_dims, win, strd, pad) in cases {
            let rank = in_dims.len();
            let total: usize = in_dims.iter().product();
            let raw: Vec<i64> = (0..total)
                .map(|i| (i as i64).wrapping_mul(2_654_435_761) ^ 0x77)
                .collect();
            let dims_u32: Vec<u32> = in_dims.iter().map(|&d| d as u32).collect();
            let tensor = Value::Tensor(
                TensorValue::new_i64_values(Shape { dims: dims_u32 }, raw.clone()).unwrap(),
            );
            let win_s = win
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let str_s = strd
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let params = rw_params_with_padding("sum", &win_s, &str_s, pad);
            let out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&tensor),
                &params,
            )
            .unwrap();
            let out_t = out.as_tensor().unwrap();
            let out_dims: Vec<usize> = out_t.shape.dims.iter().map(|&d| d as usize).collect();
            // pad_lows: SAME = floor(((out-1)*stride+win-in)/2), VALID = 0
            let pad_lows: Vec<usize> = (0..rank)
                .map(|d| {
                    if pad == "SAME" {
                        (((out_dims[d] - 1) * strd[d] + win[d]).saturating_sub(in_dims[d])) / 2
                    } else {
                        0
                    }
                })
                .collect();
            let got: Vec<i64> = out_t
                .elements
                .iter()
                .map(|l| match l {
                    Literal::I64(v) => *v,
                    o => panic!("{o:?}"),
                })
                .collect();
            let expect = ref_sum(&raw, &in_dims, &win, &strd, &pad_lows, &out_dims);
            assert_eq!(
                got, expect,
                "iN SAT rank {rank} win={win:?} stride={strd:?} pad={pad} mismatch"
            );
        }
    }

    #[test]
    #[allow(clippy::type_complexity)] // explicit case-table tuple reads clearer inline
    fn reduce_window_i64_maxmin_separable_matches_per_window() {
        // Large-window i64 max/min (separable deque path) must equal an independent
        // per-window in-bounds extremum reference across VALID/SAME, strides, ranks.
        let ref_ext = |raw: &[i64],
                       in_dims: &[usize],
                       win: &[usize],
                       strd: &[usize],
                       pad: &[usize],
                       out_dims: &[usize],
                       is_max: bool|
         -> Vec<i64> {
            let rank = in_dims.len();
            let mut in_strides = vec![1usize; rank];
            let mut a = 1;
            for d in (0..rank).rev() {
                in_strides[d] = a;
                a *= in_dims[d];
            }
            let total_out: usize = out_dims.iter().product();
            let init = if is_max { i64::MIN } else { i64::MAX };
            let mut out = Vec::with_capacity(total_out);
            let mut oi = vec![0usize; rank];
            for _ in 0..total_out {
                let mut acc = init;
                let win_total: usize = win.iter().product();
                let mut wi = vec![0usize; rank];
                for _ in 0..win_total {
                    let mut ok = true;
                    let mut idx = 0usize;
                    for d in 0..rank {
                        let p = oi[d] * strd[d] + wi[d];
                        if p < pad[d] {
                            ok = false;
                            break;
                        }
                        let ic = p - pad[d];
                        if ic >= in_dims[d] {
                            ok = false;
                            break;
                        }
                        idx += ic * in_strides[d];
                    }
                    if ok {
                        acc = if is_max {
                            acc.max(raw[idx])
                        } else {
                            acc.min(raw[idx])
                        };
                    }
                    let mut c = true;
                    for d in (0..rank).rev() {
                        if c {
                            wi[d] += 1;
                            if wi[d] >= win[d] {
                                wi[d] = 0;
                            } else {
                                c = false;
                            }
                        }
                    }
                }
                out.push(acc);
                let mut c = true;
                for d in (0..rank).rev() {
                    if c {
                        oi[d] += 1;
                        if oi[d] >= out_dims[d] {
                            oi[d] = 0;
                        } else {
                            c = false;
                        }
                    }
                }
            }
            out
        };
        let cases: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, &str)> = vec![
            (vec![64], vec![16], vec![1], "VALID"),
            (vec![50], vec![20], vec![3], "SAME"),
            (vec![23, 19], vec![7, 5], vec![2, 2], "VALID"),
            (vec![20, 20], vec![6, 6], vec![1, 1], "SAME"),
            (vec![9, 9, 3], vec![4, 4, 1], vec![1, 1, 1], "VALID"),
        ];
        for is_max in [true, false] {
            for (in_dims, win, strd, pad) in &cases {
                let rank = in_dims.len();
                let total: usize = in_dims.iter().product();
                let raw: Vec<i64> = (0..total)
                    .map(|i| ((i as i64).wrapping_mul(2_654_435_761) >> 3) ^ 0x2b)
                    .collect();
                let dims_u32: Vec<u32> = in_dims.iter().map(|&d| d as u32).collect();
                let tensor = Value::Tensor(
                    TensorValue::new_i64_values(Shape { dims: dims_u32 }, raw.clone()).unwrap(),
                );
                let win_s = win
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let str_s = strd
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let op = if is_max { "max" } else { "min" };
                let params = rw_params_with_padding(op, &win_s, &str_s, pad);
                let out = eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(&tensor),
                    &params,
                )
                .unwrap();
                let out_t = out.as_tensor().unwrap();
                let out_dims: Vec<usize> = out_t.shape.dims.iter().map(|&d| d as usize).collect();
                let pad_lows: Vec<usize> = (0..rank)
                    .map(|d| {
                        if *pad == "SAME" {
                            (((out_dims[d] - 1) * strd[d] + win[d]).saturating_sub(in_dims[d])) / 2
                        } else {
                            0
                        }
                    })
                    .collect();
                let got: Vec<i64> = out_t
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::I64(v) => *v,
                        o => panic!("{o:?}"),
                    })
                    .collect();
                let expect = ref_ext(&raw, in_dims, win, strd, &pad_lows, &out_dims, is_max);
                assert_eq!(
                    got, expect,
                    "i64 {op} separable win={win:?} stride={strd:?} pad={pad} mismatch"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_i64_maxmin_separable_vs_dense() {
        use std::time::Instant;
        let (rows, cols) = (512usize, 512usize);
        let win = 15usize;
        let raw: Vec<i64> = (0..rows * cols)
            .map(|i| (i as i64).wrapping_mul(40_503) ^ 0x5)
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                raw.clone(),
            )
            .unwrap(),
        );
        let params = rw_params_with_padding("max", &format!("{win},{win}"), "1,1", "VALID");
        let out_n = rows - win + 1;
        let best = |mut f: Box<dyn FnMut() -> i64>| {
            f();
            let mut b = f64::MAX;
            let mut dg = 0i64;
            for _ in 0..5 {
                let t = Instant::now();
                dg = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, dg)
        };
        let raw_ref = raw.clone();
        let (t_win, d_win) = best(Box::new(move || {
            let mut s = 0i64;
            for or in 0..out_n {
                for oc in 0..out_n {
                    let mut m = i64::MIN;
                    for dr in 0..win {
                        let base = (or + dr) * cols + oc;
                        for dc in 0..win {
                            m = m.max(raw_ref[base + dc]);
                        }
                    }
                    s = s.wrapping_add(m);
                }
            }
            s
        }));
        let t_ref = tensor.clone();
        let (t_sep, d_sep) = best(Box::new(move || {
            let out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&t_ref),
                &params,
            )
            .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0i64, |a, l| match l {
                    Literal::I64(v) => a.wrapping_add(*v),
                    _ => a,
                })
        }));
        assert_eq!(
            d_win, d_sep,
            "separable i64 max checksum must match per-window"
        );
        println!(
            "BENCH reduce_window i64 max([{rows},{cols}],win={win}x{win},s=1): per-window={:.4}ms separable={:.4}ms speedup={:.2}x",
            t_win * 1e3,
            t_sep * 1e3,
            t_win / t_sep,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_in_sum_sat_vs_dense() {
        use std::time::Instant;
        // Rank-1 sliding-window i64 sum (box filter): general-rank SAT
        // (window-independent) vs the per-window O(out·win) tap sum.
        let n = 1_048_576usize;
        let win = 4096usize;
        let raw: Vec<i64> = (0..n)
            .map(|i| (i as i64).wrapping_mul(40_503) ^ 0x9)
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![n as u32],
                },
                raw.clone(),
            )
            .unwrap(),
        );
        let params = rw_params_with_padding("sum", &win.to_string(), "1", "VALID");
        let out_n = n - win + 1;
        let best = |mut f: Box<dyn FnMut() -> i64>| {
            f();
            let mut b = f64::MAX;
            let mut dg = 0i64;
            for _ in 0..5 {
                let t = Instant::now();
                dg = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, dg)
        };
        let raw_ref = raw.clone();
        let (t_win, d_win) = best(Box::new(move || {
            let mut sum = 0i64;
            for o in 0..out_n {
                let mut acc = 0i64;
                for w in 0..win {
                    acc = acc.wrapping_add(raw_ref[o + w]);
                }
                sum = sum.wrapping_add(acc);
            }
            sum
        }));
        let t_ref = tensor.clone();
        let (t_sat, d_sat) = best(Box::new(move || {
            let out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&t_ref),
                &params,
            )
            .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0i64, |a, l| match l {
                    Literal::I64(v) => a.wrapping_add(*v),
                    _ => a,
                })
        }));
        assert_eq!(d_win, d_sat, "rank-1 SAT checksum must match per-window");
        println!(
            "BENCH reduce_window i64 sum 1-D([{n}],win={win},s=1): per-window={:.4}ms SAT={:.4}ms speedup={:.2}x",
            t_win * 1e3,
            t_sat * 1e3,
            t_win / t_sat,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_i64_sum_sat_vs_dense() {
        use std::time::Instant;
        // Large-window i64 sum-pool: SAT (window-independent) vs the per-window
        // dense tap sum. Same primitive output; A/B by toggling the SAT gate via a
        // direct call to each path through eval_primitive is not possible, so we
        // compare the SAT path (production, window≥16) against a replicated dense
        // per-window reference of the same shape.
        let (rows, cols) = (512usize, 512usize);
        let (win, stride) = (15usize, 1usize);
        let raw: Vec<i64> = (0..rows * cols)
            .map(|i| (i as i64).wrapping_mul(40_503) ^ 0x33)
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                raw.clone(),
            )
            .unwrap(),
        );
        let params = rw_params_with_padding(
            "sum",
            &format!("{win},{win}"),
            &format!("{stride},{stride}"),
            "VALID",
        );
        let out_n = (rows - win) / stride + 1;
        let best = |mut f: Box<dyn FnMut() -> i64>| {
            f();
            let mut b = f64::MAX;
            let mut dg = 0i64;
            for _ in 0..5 {
                let t = Instant::now();
                dg = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, dg)
        };
        // Per-window dense reference (the prior algorithm: O(out·win²)).
        let raw_ref = raw.clone();
        let (t_win, d_win) = best(Box::new(move || {
            let mut sum = 0i64;
            for or in 0..out_n {
                for oc in 0..out_n {
                    let mut acc = 0i64;
                    for dr in 0..win {
                        let base = (or * stride + dr) * cols + oc * stride;
                        for dc in 0..win {
                            acc = acc.wrapping_add(raw_ref[base + dc]);
                        }
                    }
                    sum = sum.wrapping_add(acc);
                }
            }
            sum
        }));
        let t_ref = tensor.clone();
        let (t_sat, d_sat) = best(Box::new(move || {
            let out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&t_ref),
                &params,
            )
            .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0i64, |a, l| match l {
                    Literal::I64(v) => a.wrapping_add(*v),
                    _ => a,
                })
        }));
        assert_eq!(d_win, d_sat, "SAT checksum must match per-window reference");
        println!(
            "BENCH reduce_window i64 sum([{rows},{cols}],win={win}x{win},s={stride}): per-window={:.4}ms SAT={:.4}ms speedup={:.2}x",
            t_win * 1e3,
            t_sat * 1e3,
            t_win / t_sat,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_half_dense_vs_generic() {
        use std::time::Instant;
        // bf16/f16 2D pooling: dense f64-accumulator path (HalfFloat storage) vs the
        // generic per-`Literal` path (boxed Literal storage). Parity: half bits match.
        let (n, w) = (512usize, 3usize);
        let out_n = n - w + 1;
        for (dt, op) in [
            (DType::BF16, "sum"),
            (DType::BF16, "max"),
            (DType::F16, "sum"),
            (DType::F16, "max"),
        ] {
            let bits = |v: f64| -> u16 {
                let l = if dt == DType::BF16 {
                    Literal::from_bf16_f64(v)
                } else {
                    Literal::from_f16_f64(v)
                };
                match l {
                    Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                    _ => 0,
                }
            };
            let lit = |b: u16| -> Literal {
                if dt == DType::BF16 {
                    Literal::BF16Bits(b)
                } else {
                    Literal::F16Bits(b)
                }
            };
            let raw: Vec<u16> = (0..n * n)
                .map(|i| bits(((i % 31) as f64) * 0.1 - 1.5))
                .collect();
            let dims = vec![n as u32, n as u32];
            let dense = Value::Tensor(
                TensorValue::new_half_float_values(dt, Shape { dims: dims.clone() }, raw.clone())
                    .unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    dt,
                    Shape { dims: dims.clone() },
                    raw.iter().copied().map(lit).collect(),
                )
                .unwrap(),
            );
            let params = rw_params(op, "3,3", "1,1");
            let half_bits = |v: &Value| -> Vec<u16> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => *b,
                        _ => 0,
                    })
                    .collect()
            };
            let d_out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&dense),
                &params,
            )
            .unwrap();
            let g_out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&boxed),
                &params,
            )
            .unwrap();
            assert_eq!(
                half_bits(&d_out),
                half_bits(&g_out),
                "{dt:?} {op} dense != generic half bits"
            );
            let time = |input: &Value| {
                let _ = eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(input),
                    &params,
                )
                .unwrap();
                let mut best = f64::MAX;
                for _ in 0..10 {
                    let t = Instant::now();
                    let _ = eval_primitive(
                        Primitive::ReduceWindow,
                        std::slice::from_ref(input),
                        &params,
                    )
                    .unwrap();
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best
            };
            let g = time(&boxed);
            let d = time(&dense);
            println!(
                "BENCH reduce_window {dt:?} {op}([{n},{n}],win=3x3): generic={:.4}ms dense={:.4}ms speedup={:.2}x (out {out_n}x{out_n})",
                g * 1e3,
                d * 1e3,
                g / d,
            );
        }
    }

    #[test]
    fn reduce_window_complex_dense_matches_generic() {
        // Dense complex reduce_window (HalfFloat... Complex storage) must equal the
        // generic per-Literal path (boxed Complex literals), for sum/max/min, across
        // VALID/SAME padding and strides. complex max/min use lexicographic
        // complex_ge; sum accumulates (re,im) in window order.
        let (rows, cols) = (13usize, 11usize);
        let raw: Vec<(f64, f64)> = (0..rows * cols)
            .map(|i| {
                let a = (i as f64) * 0.3 - 5.0;
                let b = ((i * 7 % 23) as f64) - 9.0;
                (a, b)
            })
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let dense = Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape { dims: dims.clone() },
                raw.clone(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape { dims: dims.clone() },
                raw.iter()
                    .map(|&(re, im)| Literal::from_complex128(re, im))
                    .collect(),
            )
            .unwrap(),
        );
        let bits = |v: &Value| -> Vec<(u64, u64)> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Complex128Bits(re, im) => (*re, *im),
                    o => panic!("expected Complex128Bits, got {o:?}"),
                })
                .collect()
        };
        for op in ["sum", "max", "min"] {
            for (win, strd, pad) in [
                ("3,3", "1,1", "VALID"),
                ("4,4", "2,2", "VALID"),
                ("3,3", "1,1", "SAME"),
            ] {
                let params = rw_params_with_padding(op, win, strd, pad);
                let d = eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(&dense),
                    &params,
                )
                .unwrap();
                let g = eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(&boxed),
                    &params,
                )
                .unwrap();
                assert_eq!(
                    bits(&d),
                    bits(&g),
                    "complex {op} win={win} pad={pad}: dense != generic"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_complex_dense_vs_generic() {
        use std::time::Instant;
        let (n, _w) = (384usize, 3usize);
        let raw: Vec<(f64, f64)> = (0..n * n)
            .map(|i| ((i % 97) as f64 * 0.1, (i % 53) as f64 * 0.2))
            .collect();
        let dims = vec![n as u32, n as u32];
        let dense = Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape { dims: dims.clone() },
                raw.clone(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape { dims: dims.clone() },
                raw.iter()
                    .map(|&(re, im)| Literal::from_complex128(re, im))
                    .collect(),
            )
            .unwrap(),
        );
        let params = rw_params("sum", "3,3", "1,1");
        let sumbits = |v: &Value| -> u64 {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |a, l| match l {
                    Literal::Complex128Bits(re, _) => a.wrapping_add(*re),
                    _ => a,
                })
        };
        let time = |input: &Value| {
            let _ = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(input),
                &params,
            )
            .unwrap();
            let mut best = f64::MAX;
            for _ in 0..8 {
                let t = Instant::now();
                let _ = eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(input),
                    &params,
                )
                .unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        assert_eq!(
            sumbits(
                &eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(&dense),
                    &params
                )
                .unwrap()
            ),
            sumbits(
                &eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(&boxed),
                    &params
                )
                .unwrap()
            ),
            "parity"
        );
        let g = time(&boxed);
        let d = time(&dense);
        println!(
            "BENCH reduce_window Complex128 sum([{n},{n}],win=3x3): generic={:.4}ms dense={:.4}ms speedup={:.2}x",
            g * 1e3,
            d * 1e3,
            g / d
        );
    }

    #[test]
    fn reduce_window_half_maxmin_separable_matches_generic() {
        // Large-window bf16/f16 max/min uses the separable monotonic-deque path
        // (dense HalfFloat storage); a boxed Literal tensor takes the generic
        // per-window path. Half bits must match (max/min select an exactly-widened
        // value, rounded back identically).
        let (n, w) = (40usize, 15usize); // ∏window=225 > 2·∑window=60 ⇒ separable
        for dt in [DType::BF16, DType::F16] {
            let bits = |v: f64| -> u16 {
                let l = if dt == DType::BF16 {
                    Literal::from_bf16_f64(v)
                } else {
                    Literal::from_f16_f64(v)
                };
                match l {
                    Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                    _ => 0,
                }
            };
            let lit = |b: u16| {
                if dt == DType::BF16 {
                    Literal::BF16Bits(b)
                } else {
                    Literal::F16Bits(b)
                }
            };
            let raw: Vec<u16> = (0..n * n)
                .map(|i| bits(((i * 7 % 53) as f64) * 0.1 - 2.5))
                .collect();
            let dims = vec![n as u32, n as u32];
            let dense = Value::Tensor(
                TensorValue::new_half_float_values(dt, Shape { dims: dims.clone() }, raw.clone())
                    .unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    dt,
                    Shape { dims: dims.clone() },
                    raw.iter().copied().map(lit).collect(),
                )
                .unwrap(),
            );
            let half_bits = |v: &Value| -> Vec<u16> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => *b,
                        _ => 0,
                    })
                    .collect()
            };
            for op in ["max", "min"] {
                let params = rw_params_with_padding(op, &format!("{w},{w}"), "1,1", "VALID");
                let d = eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(&dense),
                    &params,
                )
                .unwrap();
                let g = eval_primitive(
                    Primitive::ReduceWindow,
                    std::slice::from_ref(&boxed),
                    &params,
                )
                .unwrap();
                assert_eq!(
                    half_bits(&d),
                    half_bits(&g),
                    "{dt:?} {op} separable != generic"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_half_maxmin_separable_vs_dense() {
        use std::time::Instant;
        let (n, w) = (512usize, 15usize);
        let raw: Vec<u16> = (0..n * n)
            .map(
                |i| match Literal::from_bf16_f64(((i % 251) as f64) * 0.05 - 6.0) {
                    Literal::BF16Bits(b) => b,
                    _ => 0,
                },
            )
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::BF16,
                Shape {
                    dims: vec![n as u32, n as u32],
                },
                raw.clone(),
            )
            .unwrap(),
        );
        let params = rw_params_with_padding("max", &format!("{w},{w}"), "1,1", "VALID");
        let out_n = n - w + 1;
        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut b = f64::MAX;
            let mut dg = 0u64;
            for _ in 0..5 {
                let t = Instant::now();
                dg = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, dg)
        };
        // Per-window dense reference (the dense_float path emulated): f64-accumulate jax_max over the window.
        let raw_ref = raw.clone();
        let (t_win, d_win) = best(Box::new(move || {
            let wid = |u: u16| Literal::BF16Bits(u).as_f64().unwrap_or(0.0);
            let mut sum = 0u64;
            for or in 0..out_n {
                for oc in 0..out_n {
                    let mut m = f64::NEG_INFINITY;
                    for dr in 0..w {
                        let base = (or + dr) * n + oc;
                        for dc in 0..w {
                            m = m.max(wid(raw_ref[base + dc]));
                        }
                    }
                    sum = sum.wrapping_add(m.to_bits());
                }
            }
            sum
        }));
        let t_ref = tensor.clone();
        let (t_sep, d_sep) = best(Box::new(move || {
            let out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&t_ref),
                &params,
            )
            .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |a, l| match l {
                    Literal::BF16Bits(b) | Literal::F16Bits(b) => a.wrapping_add(u64::from(*b)),
                    _ => a,
                })
        }));
        let _ = (d_win, d_sep);
        println!(
            "BENCH reduce_window BF16 max([{n},{n}],win={w}x{w},s=1): per-window={:.4}ms separable={:.4}ms speedup={:.2}x",
            t_win * 1e3,
            t_sep * 1e3,
            t_win / t_sep,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_reduce_window_dense_bool_vs_generic() {
        use std::time::Instant;
        // 2D bool OR-pool [512,512] window 3x3 stride 1 (VALID -> [510,510]).
        // Dense bool fast path (production) vs the generic per-`Literal` gather +
        // reduce_window_accumulate_literal loop it replaces. Also a parity proof:
        // the checksum (count of `true`) must match. Built via new_bool_values so
        // the dense Bool storage path (as_bool_slice) is exercised.
        let (n, w) = (512usize, 3usize);
        let raw: Vec<bool> = (0..n * n)
            .map(|i| ((i.wrapping_mul(2_654_435_761) >> 7) & 1) == 0)
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_bool_values(
                Shape {
                    dims: vec![n as u32, n as u32],
                },
                raw,
            )
            .unwrap(),
        );
        let params = rw_params("max", "3,3", "1,1");
        let out_n = n - w + 1;

        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut b = f64::MAX;
            let mut dg = 0u64;
            for _ in 0..5 {
                let t = Instant::now();
                dg = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, dg)
        };

        let t_ref = tensor.clone();
        let (t_gen, d_gen) = best(Box::new(move || {
            let t = t_ref.as_tensor().unwrap();
            let mut out_elems: Vec<Literal> = Vec::with_capacity(out_n * out_n);
            for oy in 0..out_n {
                for ox in 0..out_n {
                    let mut accum = reduce_window_initial_accumulator(DType::Bool, "max");
                    for wy in 0..w {
                        for wx in 0..w {
                            let flat = (oy + wy) * n + (ox + wx);
                            reduce_window_accumulate_literal(
                                Primitive::ReduceWindow,
                                "max",
                                &mut accum,
                                t.elements[flat],
                            )
                            .unwrap();
                        }
                    }
                    out_elems.push(
                        reduce_window_accumulator_literal(
                            Primitive::ReduceWindow,
                            DType::Bool,
                            accum,
                        )
                        .unwrap(),
                    );
                }
            }
            out_elems.iter().fold(0u64, |acc, l| match l {
                Literal::Bool(v) => acc + u64::from(*v),
                _ => acc,
            })
        }));

        let t_fast = tensor.clone();
        let (t_dense, d_dense) = best(Box::new(move || {
            let out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&t_fast),
                &params,
            )
            .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |acc, l| match l {
                    Literal::Bool(v) => acc + u64::from(*v),
                    _ => acc,
                })
        }));

        assert_eq!(
            d_gen, d_dense,
            "dense bool reduce_window checksum must match generic"
        );
        println!(
            "BENCH reduce_window bool or([{n},{n}],win=3x3,stride=1): generic={:.4}ms dense={:.4}ms speedup={:.2}x",
            t_gen * 1e3,
            t_dense * 1e3,
            t_gen / t_dense,
        );
    }

    #[test]
    fn reduce_window_min_1d() {
        // [5, 3, 7, 1, 4], window=3, stride=1 => [3, 1, 1]
        let input = Value::vector_f64(&[5.0, 3.0, 7.0, 1.0, 4.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("min", "3", "1"),
        )
        .unwrap();
        if let Value::Tensor(t) = &out {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 1.0, 1.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_same_padding_centers_odd_window_1d() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params_with_padding("sum", "3", "1", "same"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![5]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 6.0, 9.0, 12.0, 9.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_same_padding_centers_even_window_1d() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params_with_padding("sum", "2", "1", "same"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![4]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals, vec![3.0, 5.0, 7.0, 4.0]);
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_uppercase_same_padding() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params_with_padding("sum", "2", "1", "SAME"),
        )
        .unwrap();

        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.shape.dims, vec![4]);
        let vals: Vec<f64> = tensor
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect();
        assert_eq!(vals, vec![3.0, 5.0, 7.0, 4.0]);
    }

    #[test]
    fn reduce_window_same_lower_padding_puts_extra_pad_low() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params_with_padding("sum", "2", "1", "SAME_LOWER"),
        )
        .unwrap();

        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.shape.dims, vec![4]);
        let vals: Vec<f64> = tensor
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect();
        assert_eq!(vals, vec![1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn reduce_window_rejects_unknown_padding() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params_with_padding("sum", "2", "1", "mirror"),
        );

        let err = result.expect_err("unknown padding should fail").to_string();
        assert!(
            err.contains("unsupported reduce_window padding mode"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reduce_window_separable_maxmin_matches_generic() {
        // The separable max/min fast path (dense input, ∏window > 2·∑window) must equal the
        // generic per-`Literal` loop (boxed input bypasses every dense path) bit-for-bit.
        // Covers rank-4 NHWC 7×7 + rank-2 5×5, max & min, VALID + SAME padding + stride 2.
        let data = |n: usize| -> Vec<f64> {
            (0..n)
                .map(|i| (i as f64 * 0.137).sin() * 3.0 - (i as f64 * 0.0411).cos())
                .collect()
        };
        let out_bits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };
        let cases: &[(Vec<u32>, &str, &str, &str)] = &[
            (vec![1, 16, 16, 4], "1,7,7,1", "1,1,1,1", "VALID"),
            (vec![1, 16, 16, 4], "1,7,7,1", "1,2,2,1", "SAME"),
            (vec![18, 18], "5,5", "1,1", "VALID"),
            (vec![18, 18], "5,5", "2,2", "SAME"),
        ];
        for (dims, window, strides, pad) in cases {
            let n: usize = dims.iter().map(|&d| d as usize).product();
            let raw = data(n);
            let dense = Value::Tensor(
                TensorValue::new_f64_values(Shape { dims: dims.clone() }, raw.clone()).unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: dims.clone() },
                    raw.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            );
            assert!(
                dense.as_tensor().unwrap().elements.as_f64_slice().is_some(),
                "dense input must be dense-backed to hit the separable path"
            );
            for op in ["max", "min"] {
                let p = rw_params_with_padding(op, window, strides, pad);
                let got = eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense), &p)
                    .unwrap();
                let refr =
                    eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&boxed), &p)
                        .unwrap();
                assert_eq!(
                    out_bits(&got),
                    out_bits(&refr),
                    "{op} window={window} pad={pad} dims={dims:?}"
                );
            }
        }

        let mut special = data(8 * 8);
        special[0] = -0.0;
        special[1] = 0.0;
        special[9] = f64::NAN;
        special[18] = f64::INFINITY;
        special[27] = f64::NEG_INFINITY;
        special[36] = -0.0;
        special[45] = 0.0;
        let dense = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: vec![8, 8] }, special.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![8, 8] },
                special.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        );
        for op in ["max", "min"] {
            let p = rw_params_with_padding(op, "5,5", "1,1", "SAME");
            let got =
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&dense), &p).unwrap();
            let refr =
                eval_primitive(Primitive::ReduceWindow, std::slice::from_ref(&boxed), &p).unwrap();
            assert_eq!(out_bits(&got), out_bits(&refr), "{op} special-value bits");
        }
    }

    #[test]
    fn reduce_window_same_padding_centers_rank2_window() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3, 3] },
                (1..=9).map(|v| Literal::from_f64(v as f64)).collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params_with_padding("sum", "2,2", "1,1", "same"),
        )
        .unwrap();

        if let Value::Tensor(t) = &out {
            assert_eq!(t.shape.dims, vec![3, 3]);
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(
                vals,
                vec![12.0, 16.0, 9.0, 24.0, 28.0, 15.0, 15.0, 17.0, 9.0]
            );
        } else {
            assert!(matches!(out, Value::Tensor(_)), "expected tensor");
        }
    }

    #[test]
    fn reduce_window_rank2_f64_3x3_same_sum_row_major_golden() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3, 3] },
                (1..=9).map(|v| Literal::from_f64(v as f64)).collect(),
            )
            .unwrap(),
        );
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params_with_padding("sum", "3,3", "1,1", "SAME"),
        )
        .unwrap();

        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.shape.dims, vec![3, 3]);
        let vals: Vec<f64> = tensor
            .elements
            .iter()
            .map(|literal| literal.as_f64().expect("F64 output"))
            .collect();
        assert_eq!(
            vals,
            vec![12.0, 21.0, 16.0, 27.0, 45.0, 33.0, 24.0, 39.0, 28.0]
        );
    }

    #[test]
    fn reduce_window_rank2_f64_same_sum_golden_hash() {
        let elements: Vec<Literal> = (0..64 * 64)
            .map(|i| {
                let x = i as f64;
                Literal::from_f64((x * 0.125).sin() + (x * 0.03125).cos())
            })
            .collect();
        let input = Value::Tensor(
            TensorValue::new(DType::F64, Shape { dims: vec![64, 64] }, elements).unwrap(),
        );

        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params_with_padding("sum", "3,3", "1,1", "SAME"),
        )
        .unwrap();
        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.shape.dims, vec![64, 64]);
        assert!(
            tensor.elements.as_f64_slice().is_some(),
            "well-formed F64 reduce_window output should stay dense"
        );
        let output_bits: Vec<u64> = tensor
            .elements
            .iter()
            .map(|literal| literal.as_f64().expect("F64 output").to_bits())
            .collect();
        let digest =
            fj_test_utils::fixture_id_from_json(&output_bits).expect("output digest should build");
        assert_eq!(
            digest,
            "693388d434dacc2e3367b7853eb9c5da6ea1e03db144ef64138087dc971e3aee"
        );
    }

    #[test]
    fn reduce_window_rank2_f64_same_sum_malformed_literal_fallback() {
        let input = Value::Tensor(TensorValue {
            dtype: DType::F64,
            shape: Shape { dims: vec![1, 1] },
            elements: vec![Literal::I64(7)].into(),
        });

        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params_with_padding("sum", "3,3", "1,1", "SAME"),
        )
        .unwrap();

        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.shape.dims, vec![1, 1]);
        // The malformed I64-in-declared-F64 input forces the literal fallback:
        // the dense fast path cannot run (the boxed input has no `as_f64_slice`),
        // so the I64(7) is read via `as_f64()` -> 7.0 rather than bit-reinterpreted
        // (which would yield ~3.5e-323). The fallback's *value* is the guarantee;
        // its output storage may now be densified by `TensorValue::new` (cbea72b3).
        let got = tensor.elements[0].as_f64().expect("F64 fallback output");
        assert_eq!(got.to_bits(), 7.0_f64.to_bits());
    }

    #[test]
    fn reduce_window_scalar_passthrough() {
        let input = Value::scalar_f64(42.0);
        let out = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "1", "1"),
        )
        .unwrap();
        assert_eq!(out.as_f64_scalar().unwrap(), 42.0);
    }

    #[test]
    fn reduce_window_rejects_zero_stride() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let result = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "2", "0"),
        );

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("window_strides[0] must be positive"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reduce_window_rejects_short_stride_list() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                (1..=4).map(|v| Literal::from_f64(v as f64)).collect(),
            )
            .unwrap(),
        );
        let result = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "1,1", "1"),
        );

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("window_strides length 1 doesn't match tensor rank 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reduce_window_rejects_malformed_window_dimension() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let result = eval_primitive(
            Primitive::ReduceWindow,
            &[input],
            &rw_params("sum", "two", "1"),
        );

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("invalid window_dimensions[0]: 'two'"),
            "unexpected error: {err}"
        );
    }

    // ── Rev tests ────────────────────────────────────────────────

    #[test]
    fn test_rev_1d() {
        let input = Value::vector_i64(&[1, 2, 3]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0".into());
        let out = eval_primitive(Primitive::Rev, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![3, 2, 1]);
    }

    #[test]
    fn test_rev_2d_axis0() {
        // [[1,2],[3,4],[5,6]] reversed along axis 0 => [[5,6],[3,4],[1,2]]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
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
        );
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0".into());
        let out = eval_primitive(Primitive::Rev, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![5, 6, 3, 4, 1, 2]);
    }

    #[test]
    fn test_rev_2d_axis1() {
        // [[1,2,3],[4,5,6]] reversed along axis 1 => [[3,2,1],[6,5,4]]
        let input = Value::Tensor(
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
        );
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "1".into());
        let out = eval_primitive(Primitive::Rev, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn test_rev_empty_huge_shape_returns_empty_tensor() {
        let huge = u32::MAX;
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![0, huge, huge, huge],
                },
                Vec::new(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "1,2,3".into());

        let out = eval_primitive(Primitive::Rev, &[input], &params).unwrap();
        let tensor = out.as_tensor().unwrap();

        assert_eq!(tensor.shape.dims, vec![0, huge, huge, huge]);
        assert!(tensor.elements.is_empty());
    }

    // ── Squeeze tests ────────────────────────────────────────────

    #[test]
    fn test_squeeze_remove_leading() {
        // [1, 4, 1] → [4, 1] removing dim 0
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![1, 4, 1],
                },
                vec![
                    Literal::I64(10),
                    Literal::I64(20),
                    Literal::I64(30),
                    Literal::I64(40),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("dimensions".into(), "0".into());
        let out = eval_primitive(Primitive::Squeeze, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![4, 1]);
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_squeeze_remove_trailing() {
        // [4, 1] → [4] removing dim 1
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![4, 1] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("dimensions".into(), "1".into());
        let out = eval_primitive(Primitive::Squeeze, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![4]);
    }

    #[test]
    fn test_squeeze_remove_multiple() {
        // [1, 4, 1, 3, 1] → [4, 3] removing dims 0, 2, 4
        let mut elems = Vec::new();
        for i in 0..12 {
            elems.push(Literal::I64(i));
        }
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![1, 4, 1, 3, 1],
                },
                elems,
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("dimensions".into(), "0,2,4".into());
        let out = eval_primitive(Primitive::Squeeze, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![4, 3]);
        assert_eq!(t.elements.len(), 12);
    }

    #[test]
    fn test_squeeze_no_op() {
        // [4, 3] with no size-1 dims — identity
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![4, 3] },
                (0..12).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let params = BTreeMap::new(); // no dimensions param — squeeze all size-1, which is none
        let out =
            eval_primitive(Primitive::Squeeze, std::slice::from_ref(&input), &params).unwrap();
        assert_eq!(out, input);
    }

    // ── Split tests ──────────────────────────────────────────────

    #[test]
    fn test_split_equal() {
        // split [1,2,3,4,5,6] into 3 equal parts: [[1,2],[3,4],[5,6]]
        let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "0".into());
        params.insert("num_sections".into(), "3".into());
        let out = eval_primitive(Primitive::Split, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        // Result shape: [3, 2]
        assert_eq!(t.shape.dims, vec![3, 2]);
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_split_multi_uneven_sizes() {
        // 2×4 matrix, split axis=1 into uneven sizes [1, 3] → two arrays of shapes
        // [2,1] and [2,3] (jnp.split semantics). The single-output path fails closed
        // on uneven sizes; eval_primitive_multi must return the N sections.
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 4] },
                (0..8).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "1".into());
        params.insert("sizes".into(), "1,3".into());
        let out = super::eval_primitive_multi(Primitive::Split, &[input], &params).unwrap();
        assert_eq!(out.len(), 2, "uneven split must yield 2 sections");

        let s0 = out[0].as_tensor().unwrap();
        assert_eq!(s0.shape.dims, vec![2, 1]);
        let v0: Vec<i64> = s0.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(v0, vec![0, 4]); // column 0 of each row

        let s1 = out[1].as_tensor().unwrap();
        assert_eq!(s1.shape.dims, vec![2, 3]);
        let v1: Vec<i64> = s1.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(v1, vec![1, 2, 3, 5, 6, 7]); // columns 1..4 of each row
    }

    #[test]
    fn test_split_multi_equal_returns_separate_sections() {
        // Equal split via the multi-output path returns N separate tensors (not the
        // single packed [num_sections, …] tensor of the single-output path).
        let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("num_sections".into(), "3".into());
        let out = super::eval_primitive_multi(Primitive::Split, &[input], &params).unwrap();
        assert_eq!(out.len(), 3);
        for (i, expect) in [[1, 2], [3, 4], [5, 6]].iter().enumerate() {
            let t = out[i].as_tensor().unwrap();
            assert_eq!(t.shape.dims, vec![2]);
            let v: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
            assert_eq!(v, expect.to_vec());
        }
    }

    #[test]
    fn test_split_defaults_axis_to_zero() {
        let input = Value::vector_i64(&[1, 2, 3, 4, 5, 6]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("num_sections".into(), "3".into());
        let out = eval_primitive(Primitive::Split, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![3, 2]);
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_split_defaults_to_single_section_without_sizes_or_num_sections() {
        let input = Value::vector_i64(&[1, 2, 3, 4]).unwrap();
        let out = eval_primitive(Primitive::Split, &[input], &BTreeMap::new()).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![1, 4]);
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_split_empty_huge_equal_section_returns_empty_tensor() {
        let huge = u32::MAX;
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![0, huge, huge, huge],
                },
                Vec::new(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "0".into());
        params.insert("num_sections".into(), "1".into());

        let out = eval_primitive(Primitive::Split, &[input], &params).unwrap();
        let tensor = out.as_tensor().unwrap();

        assert_eq!(tensor.shape.dims, vec![1, 0, huge, huge, huge]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn test_split_uneven_sizes_fails_closed() {
        // Uneven split sizes [2,3] cannot be packed into the single-output
        // rectangular tensor model, so Split must fail closed rather than
        // silently returning only the first section (the prior behaviour,
        // which corrupted any transform flowing through the dropped sections).
        let input = Value::vector_i64(&[1, 2, 3, 4, 5]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "0".into());
        params.insert("sizes".into(), "2,3".into());
        let err = eval_primitive(Primitive::Split, &[input], &params)
            .expect_err("uneven split must be rejected, not silently truncated");
        assert!(
            matches!(
                err,
                EvalError::Unsupported {
                    primitive: Primitive::Split,
                    ..
                }
            ),
            "expected Unsupported for uneven split, got {err:?}"
        );
    }

    #[test]
    fn test_split_all_equal_explicit_sizes_packs() {
        // Explicit `sizes` that happen to be all equal still take the packed
        // equal-split path: [1,2,3,4] with sizes [2,2] -> shape [2, 2].
        let input = Value::vector_i64(&[1, 2, 3, 4]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "0".into());
        params.insert("sizes".into(), "2,2".into());
        let out = eval_primitive(Primitive::Split, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![2, 2]);
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_split_axis1() {
        // 2x4 matrix split along axis 1 into 2 parts → shape [2, 2, 2]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 4] },
                (1..=8).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "1".into());
        params.insert("num_sections".into(), "2".into());
        let out = eval_primitive(Primitive::Split, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![2, 2, 2]);
    }

    // ── ExpandDims tests ─────────────────────────────────────────

    #[test]
    fn test_expand_dims_leading() {
        // expand_dims [4, 3] axis=0 → [1, 4, 3]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![4, 3] },
                (0..12).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "0".into());
        let out = eval_primitive(Primitive::ExpandDims, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![1, 4, 3]);
        assert_eq!(t.elements.len(), 12);
    }

    #[test]
    fn test_expand_dims_trailing() {
        // expand_dims [4, 3] axis=2 → [4, 3, 1]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![4, 3] },
                (0..12).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "2".into());
        let out = eval_primitive(Primitive::ExpandDims, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![4, 3, 1]);
        assert_eq!(t.elements.len(), 12);
    }

    #[test]
    fn test_expand_dims_middle() {
        // expand_dims [4, 3] axis=1 → [4, 1, 3]
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![4, 3] },
                (0..12).map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "1".into());
        let out = eval_primitive(Primitive::ExpandDims, &[input], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![4, 1, 3]);
        assert_eq!(t.elements.len(), 12);
    }

    #[test]
    fn test_copy_is_value_identity() {
        // `Copy` is a value-level identity. The element buffer is backed by an
        // immutable `Arc`-shared `LiteralBuffer` (a copy-on-write storage
        // optimization), so cloning shares the backing buffer rather than allocating
        // a new one. Because the buffer is never mutated in place, sharing it is
        // observationally identical to a deep copy — the only contract `Copy` owes is
        // value equality of the result.
        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3] },
                vec![Literal::I64(1), Literal::I64(2), Literal::I64(3)],
            )
            .unwrap(),
        );

        let copied = eval_primitive(Primitive::Copy, std::slice::from_ref(&input), &no_params())
            .expect("copy should succeed");

        assert_eq!(copied, input);
        let copied_tensor = copied.as_tensor().unwrap();
        assert_eq!(copied_tensor.shape.dims, vec![3]);
        assert_eq!(
            copied_tensor.elements,
            vec![Literal::I64(1), Literal::I64(2), Literal::I64(3)]
        );
    }

    #[test]
    fn test_bitcast_f64_to_i64_and_back_preserves_bits() {
        let input = Value::scalar_f64(-3.5);

        let mut to_i64 = BTreeMap::new();
        to_i64.insert("new_dtype".to_owned(), "i64".to_owned());
        let bitcast_i64 = eval_primitive(
            Primitive::BitcastConvertType,
            std::slice::from_ref(&input),
            &to_i64,
        )
        .expect("f64 -> i64 bitcast should succeed");

        let mut to_f64 = BTreeMap::new();
        to_f64.insert("new_dtype".to_owned(), "f64".to_owned());
        let round_trip = eval_primitive(
            Primitive::BitcastConvertType,
            std::slice::from_ref(&bitcast_i64),
            &to_f64,
        )
        .expect("i64 -> f64 bitcast should succeed");

        match (input, round_trip) {
            (
                Value::Scalar(Literal::F64Bits(expected)),
                Value::Scalar(Literal::F64Bits(actual)),
            ) => {
                assert_eq!(
                    actual, expected,
                    "bitcast round trip must preserve exact bits"
                );
            }
            other => {
                assert!(
                    matches!(
                        other,
                        (
                            Value::Scalar(Literal::F64Bits(_)),
                            Value::Scalar(Literal::F64Bits(_))
                        )
                    ),
                    "unexpected round-trip payload: {other:?}"
                );
            }
        }
    }

    #[test]
    fn test_bitcast_narrowing_appends_trailing_dimension() {
        let input = Value::scalar_f64(1.25);
        let mut params = BTreeMap::new();
        params.insert("new_dtype".to_owned(), "u32".to_owned());
        let out = eval_primitive(Primitive::BitcastConvertType, &[input], &params)
            .expect("f64 -> u32 bitcast should split into two trailing elements");
        let tensor = out.as_tensor().expect("narrowing should produce a tensor");
        assert_eq!(tensor.dtype, DType::U32);
        assert_eq!(tensor.shape.dims, vec![2]);

        let bytes = 1.25_f64.to_bits().to_le_bytes();
        assert_eq!(
            tensor.elements,
            vec![
                Literal::U32(u32::from_le_bytes(
                    [bytes[0], bytes[1], bytes[2], bytes[3],]
                )),
                Literal::U32(u32::from_le_bytes(
                    [bytes[4], bytes[5], bytes[6], bytes[7],]
                )),
            ]
        );
    }

    #[test]
    fn test_bitcast_widening_removes_trailing_dimension() {
        let bytes = 1.25_f64.to_bits().to_le_bytes();
        let input = Value::Tensor(
            TensorValue::new_u32_values(
                Shape { dims: vec![2] },
                vec![
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("new_dtype".to_owned(), "f64".to_owned());
        let out = eval_primitive(Primitive::BitcastConvertType, &[input], &params)
            .expect("u32 -> f64 bitcast should consume the trailing dimension");
        let tensor = out.as_tensor().expect("tensor input stays tensor output");
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(tensor.shape.dims, Vec::<u32>::new());
        assert_eq!(tensor.elements, vec![Literal::from_f64(1.25)]);
    }

    #[test]
    fn test_bitcast_widening_rejects_wrong_trailing_dimension() {
        let input = Value::Tensor(
            TensorValue::new_u32_values(Shape { dims: vec![3] }, vec![1, 2, 3]).unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("new_dtype".to_owned(), "f64".to_owned());
        let err = eval_primitive(Primitive::BitcastConvertType, &[input], &params)
            .expect_err("u32 -> f64 requires a trailing dimension of size two");
        match err {
            EvalError::Unsupported { detail, .. } => {
                assert!(detail.contains("trailing dimension size 2"));
            }
            other => panic!("expected unsupported error, got {other:?}"),
        }
    }

    #[test]
    fn test_broadcasted_iota_2d_axis_one() {
        let mut params = BTreeMap::new();
        params.insert("shape".to_owned(), "2,3".to_owned());
        params.insert("dimension".to_owned(), "1".to_owned());
        params.insert("dtype".to_owned(), "i64".to_owned());

        let out = eval_primitive(Primitive::BroadcastedIota, &[], &params)
            .expect("broadcasted_iota should succeed");
        let tensor = out.as_tensor().expect("tensor output expected");
        assert_eq!(tensor.shape.dims, vec![2, 3]);

        let values: Vec<i64> = tensor
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64 element"))
            .collect();
        assert_eq!(values, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_broadcasted_iota_supports_complex_numeric_dtype() {
        let mut params = BTreeMap::new();
        params.insert("shape".to_owned(), "2,3".to_owned());
        params.insert("dimension".to_owned(), "0".to_owned());
        params.insert("dtype".to_owned(), "complex128".to_owned());

        let out = eval_primitive(Primitive::BroadcastedIota, &[], &params)
            .expect("complex broadcasted_iota should follow JAX numeric dtype rule");
        let tensor = out.as_tensor().expect("tensor output expected");
        assert_eq!(tensor.dtype, DType::Complex128);
        assert_eq!(tensor.shape.dims, vec![2, 3]);

        let values: Vec<(f64, f64)> = tensor
            .elements
            .iter()
            .map(|lit| lit.as_complex128().expect("complex128 element"))
            .collect();
        assert_eq!(
            values,
            vec![
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
            ]
        );
    }

    #[test]
    fn test_broadcasted_iota_rejects_bool_dtype() {
        let mut params = BTreeMap::new();
        params.insert("shape".to_owned(), "2,3".to_owned());
        params.insert("dimension".to_owned(), "1".to_owned());
        params.insert("dtype".to_owned(), "bool".to_owned());

        let err = eval_primitive(Primitive::BroadcastedIota, &[], &params)
            .expect_err("JAX broadcasted_iota rejects bool dtype");
        assert!(
            err.to_string()
                .contains("broadcasted_iota does not accept bool dtype"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_broadcasted_iota_empty_huge_shape_returns_empty_tensor() {
        let huge = u32::MAX;
        let mut params = BTreeMap::new();
        params.insert("shape".to_owned(), format!("{huge},{huge},{huge},0"));
        params.insert("dimension".to_owned(), "2".to_owned());
        params.insert("dtype".to_owned(), "i64".to_owned());

        let out = eval_primitive(Primitive::BroadcastedIota, &[], &params)
            .expect("empty broadcasted_iota should succeed before huge products overflow");
        let tensor = out.as_tensor().expect("tensor output expected");

        assert_eq!(tensor.shape.dims, vec![huge, huge, huge, 0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn test_reduce_precision_identity_with_full_bits() {
        let input = Value::scalar_f64(1.0000001);
        let mut params = BTreeMap::new();
        params.insert("exponent_bits".to_owned(), "11".to_owned());
        params.insert("mantissa_bits".to_owned(), "52".to_owned());

        let out = eval_primitive(
            Primitive::ReducePrecision,
            std::slice::from_ref(&input),
            &params,
        )
        .expect("reduce_precision should succeed");
        assert_eq!(out, input);
    }

    #[test]
    fn test_reduce_precision_truncates_mantissa_bits() {
        let input = Value::scalar_f64(1.0000001);
        let mut params = BTreeMap::new();
        params.insert("exponent_bits".to_owned(), "8".to_owned());
        params.insert("mantissa_bits".to_owned(), "7".to_owned());

        let out = eval_primitive(
            Primitive::ReducePrecision,
            std::slice::from_ref(&input),
            &params,
        )
        .expect("reduce_precision should succeed");

        let input_val = input.as_f64_scalar().unwrap();
        let out_val = out.as_f64_scalar().unwrap();
        assert_ne!(out_val.to_bits(), input_val.to_bits());
    }
}

#[cfg(test)]
mod prop_tests {
    use super::arithmetic::trigamma_approx;
    use super::{EvalError, eval_fori_loop, eval_primitive};
    use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    fn no_params() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    fn axes_params(axes: &str) -> BTreeMap<String, String> {
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), axes.to_owned());
        params
    }

    fn assert_complex128_close(value: &Value, expected_re: f64, expected_im: f64, tol: f64) {
        let (re, im) = match value {
            Value::Scalar(Literal::Complex128Bits(re, im)) => {
                (f64::from_bits(*re), f64::from_bits(*im))
            }
            Value::Scalar(Literal::Complex64Bits(re, im)) => {
                (f32::from_bits(*re) as f64, f32::from_bits(*im) as f64)
            }
            _ => {
                assert!(
                    matches!(
                        value,
                        Value::Scalar(Literal::Complex128Bits(_, _))
                            | Value::Scalar(Literal::Complex64Bits(_, _))
                    ),
                    "expected complex scalar, got {value:?}"
                );
                (0.0, 0.0)
            }
        };
        assert!(
            (re - expected_re).abs() <= tol,
            "real mismatch: got {re}, expected {expected_re}"
        );
        assert!(
            (im - expected_im).abs() <= tol,
            "imag mismatch: got {im}, expected {expected_im}"
        );
    }

    proptest! {
        #[test]
        fn prop_add_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Add,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Add,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_mul_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Mul,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Mul,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_max_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Max,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Max,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_min_commutative(a in -1000i64..1000, b in -1000i64..1000) {
            let ab = eval_primitive(
                Primitive::Min,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let ba = eval_primitive(
                Primitive::Min,
                &[Value::scalar_i64(b), Value::scalar_i64(a)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn prop_neg_involution(x in -1000i64..1000) {
            let neg1 = eval_primitive(
                Primitive::Neg,
                &[Value::scalar_i64(x)],
                &no_params(),
            ).unwrap();
            let neg2 = eval_primitive(
                Primitive::Neg,
                &[neg1],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(neg2, Value::scalar_i64(x));
        }

        #[test]
        fn prop_abs_non_negative(x in -1000i64..1000) {
            let out = eval_primitive(
                Primitive::Abs,
                &[Value::scalar_i64(x)],
                &no_params(),
            ).unwrap();
            if let Value::Scalar(fj_core::Literal::I64(v)) = out {
                prop_assert!(v >= 0, "abs({x}) = {v} should be non-negative");
            }
        }

        #[test]
        fn prop_reshape_roundtrip(a in -100i64..100, b in -100i64..100, c in -100i64..100) {
            let input = Value::vector_i64(&[a, b, c]).unwrap();

            let mut to_3x1 = BTreeMap::new();
            to_3x1.insert("new_shape".into(), "3,1".into());
            let reshaped = eval_primitive(Primitive::Reshape, std::slice::from_ref(&input), &to_3x1).unwrap();

            let mut to_3 = BTreeMap::new();
            to_3.insert("new_shape".into(), "3".into());
            let restored = eval_primitive(Primitive::Reshape, &[reshaped], &to_3).unwrap();

            prop_assert_eq!(restored, input);
        }

        #[test]
        fn prop_reduce_sum_matches_manual(
            a in -100i64..100,
            b in -100i64..100,
            c in -100i64..100
        ) {
            let input = Value::vector_i64(&[a, b, c]).unwrap();
            let out = eval_primitive(
                Primitive::ReduceSum,
                &[input],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(out, Value::scalar_i64(a + b + c));
        }

        #[test]
        fn prop_reduce_prod_matches_manual(
            a in -10i64..10,
            b in -10i64..10,
            c in -10i64..10
        ) {
            let input = Value::vector_i64(&[a, b, c]).unwrap();
            let out = eval_primitive(
                Primitive::ReduceProd,
                &[input],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(out, Value::scalar_i64(a * b * c));
        }

        #[test]
        fn prop_eq_reflexive(x in -1000i64..1000) {
            let out = eval_primitive(
                Primitive::Eq,
                &[Value::scalar_i64(x), Value::scalar_i64(x)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(out, Value::scalar_bool(true));
        }

        #[test]
        fn prop_add_sub_inverse(a in -1000i64..1000, b in -1000i64..1000) {
            let sum = eval_primitive(
                Primitive::Add,
                &[Value::scalar_i64(a), Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            let result = eval_primitive(
                Primitive::Sub,
                &[sum, Value::scalar_i64(b)],
                &no_params(),
            ).unwrap();
            prop_assert_eq!(result, Value::scalar_i64(a));
        }
    }

    #[test]
    fn test_complex_constructor() {
        let out = eval_primitive(
            Primitive::Complex,
            &[Value::scalar_f64(1.0), Value::scalar_f64(2.0)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_complex128(1.0, 2.0));
    }

    #[test]
    fn test_complex_add() {
        let out = eval_primitive(
            Primitive::Add,
            &[
                Value::scalar_complex128(1.0, 2.0),
                Value::scalar_complex128(3.0, 4.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_complex128(4.0, 6.0));
    }

    #[test]
    fn test_complex_mul() {
        let out = eval_primitive(
            Primitive::Mul,
            &[
                Value::scalar_complex128(1.0, 2.0),
                Value::scalar_complex128(3.0, 4.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_complex128(-5.0, 10.0));
    }

    #[test]
    fn test_complex_div() {
        let out = eval_primitive(
            Primitive::Div,
            &[
                Value::scalar_complex128(1.0, 2.0),
                Value::scalar_complex128(3.0, 4.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.44, 0.08, 1e-12);
    }

    #[test]
    fn test_complex_pow_integer_exponent() {
        let out = eval_primitive(
            Primitive::Pow,
            &[
                Value::scalar_complex128(1.0, 1.0),
                Value::scalar_complex128(2.0, 0.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, 2.0, 1e-12);
    }

    #[test]
    fn test_complex_pow_square_root_branch() {
        let out = eval_primitive(
            Primitive::Pow,
            &[
                Value::scalar_complex128(1.0, 1.0),
                Value::scalar_complex128(0.5, 0.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 1.09868411346781, 0.45508986056222733, 1e-12);
    }

    #[test]
    fn test_complex_pow_real_inputs_match_real_pow() {
        let complex_out = eval_primitive(
            Primitive::Pow,
            &[
                Value::scalar_complex128(3.0, 0.0),
                Value::scalar_complex128(0.5, 0.0),
            ],
            &no_params(),
        )
        .unwrap();
        let real_out = eval_primitive(
            Primitive::Pow,
            &[Value::scalar_f64(3.0), Value::scalar_f64(0.5)],
            &no_params(),
        )
        .unwrap();
        let real_value = real_out.as_f64_scalar().unwrap();
        assert_complex128_close(&complex_out, real_value, 0.0, 1e-12);
    }

    #[test]
    fn test_complex_max_lexicographic_order() {
        let out = eval_primitive(
            Primitive::Max,
            &[
                Value::scalar_complex128(1.0, 2.0),
                Value::scalar_complex128(3.0, 0.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_complex128(3.0, 0.0));
    }

    #[test]
    fn test_complex_min_lexicographic_tiebreaks_on_imag() {
        let out = eval_primitive(
            Primitive::Min,
            &[
                Value::scalar_complex128(1.0, 2.0),
                Value::scalar_complex128(1.0, 1.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_complex128(1.0, 1.0));
    }

    #[test]
    fn test_complex_rem_gaussian_integer_rounding() {
        let out = eval_primitive(
            Primitive::Rem,
            &[
                Value::scalar_complex128(4.0, 3.0),
                Value::scalar_complex128(2.0, 1.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, 1.0, 1e-12);
    }

    #[test]
    fn test_complex_sign_normalizes_value() {
        let out = eval_primitive(
            Primitive::Sign,
            &[Value::scalar_complex128(1.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        let inv_sqrt2 = 2.0_f64.sqrt().recip();
        assert_complex128_close(&out, inv_sqrt2, inv_sqrt2, 1e-12);
    }

    #[test]
    fn test_complex_square_multiplies_in_field() {
        let out = eval_primitive(
            Primitive::Square,
            &[Value::scalar_complex128(1.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, 2.0, 1e-12);
    }

    #[test]
    fn test_complex_reciprocal_inverts_value() {
        let out = eval_primitive(
            Primitive::Reciprocal,
            &[Value::scalar_complex128(1.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.5, -0.5, 1e-12);
    }

    #[test]
    fn test_complex_sqrt_uses_principal_branch() {
        let out = eval_primitive(
            Primitive::Sqrt,
            &[Value::scalar_complex128(-4.0, 0.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, 2.0, 1e-12);
    }

    #[test]
    fn test_complex_rsqrt_inverts_principal_sqrt() {
        let out = eval_primitive(
            Primitive::Rsqrt,
            &[Value::scalar_complex128(-4.0, 0.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, -0.5, 1e-12);
    }

    #[test]
    fn test_complex64_tensor_sqrt_preserves_dtype_and_shape() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(3.0, 4.0),
                    Literal::from_complex64(-4.0, 0.0),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::Complex64);
        assert_eq!(tensor.shape, Shape { dims: vec![2] });

        let (re0, im0) = tensor.elements[0]
            .as_complex64()
            .expect("complex64 element");
        assert!((re0 - 2.0).abs() < 1e-6);
        assert!((im0 - 1.0).abs() < 1e-6);

        let (re1, im1) = tensor.elements[1]
            .as_complex64()
            .expect("complex64 element");
        assert!(re1.abs() < 1e-6);
        assert!((im1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_is_finite_checks_both_parts() {
        let out = eval_primitive(
            Primitive::IsFinite,
            &[Value::scalar_complex128(1.0, f64::INFINITY)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_bool(false));
    }

    #[test]
    fn test_complex_atan2_reports_unsupported_dtype() {
        let err = eval_primitive(
            Primitive::Atan2,
            &[
                Value::scalar_complex128(1.0, 1.0),
                Value::scalar_complex128(2.0, -1.0),
            ],
            &no_params(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("complex operands"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_complex64_tensor_atan2_reports_unsupported_dtype() {
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(1.0, 0.0),
                    Literal::from_complex64(-1.0, 0.0),
                ],
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(2.0, 0.0),
                    Literal::from_complex64(0.0, 0.0),
                ],
            )
            .unwrap(),
        );

        let err = eval_primitive(Primitive::Atan2, &[lhs, rhs], &no_params()).unwrap_err();
        assert!(
            err.to_string().contains("complex operands"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_floor_complex_reports_unsupported_dtype() {
        let err = eval_primitive(
            Primitive::Floor,
            &[Value::scalar_complex128(1.0, 1.0)],
            &no_params(),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("floor is not supported for complex dtypes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_select_complex_condition_reports_boolean_requirement() {
        let err = eval_primitive(
            Primitive::Select,
            &[
                Value::scalar_complex128(1.0, 0.0),
                Value::scalar_f64(2.0),
                Value::scalar_f64(3.0),
            ],
            &no_params(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("select condition must be boolean"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_select_complex_scalar_values_preserve_selected_branch() {
        let out = eval_primitive(
            Primitive::Select,
            &[
                Value::scalar_bool(false),
                Value::scalar_complex128(1.0, 2.0),
                Value::scalar_complex128(3.0, -4.0),
            ],
            &no_params(),
        )
        .unwrap();

        assert_complex128_close(&out, 3.0, -4.0, 1e-12);
    }

    #[test]
    fn test_select_complex64_tensor_values_preserve_dtype_and_shape() {
        let cond = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![2] },
                vec![Literal::Bool(true), Literal::Bool(false)],
            )
            .unwrap(),
        );
        let on_true = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(1.0, 2.0),
                    Literal::from_complex64(3.0, 4.0),
                ],
            )
            .unwrap(),
        );
        let on_false = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(5.0, 6.0),
                    Literal::from_complex64(7.0, 8.0),
                ],
            )
            .unwrap(),
        );

        let out =
            eval_primitive(Primitive::Select, &[cond, on_true, on_false], &no_params()).unwrap();
        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::Complex64);
        assert_eq!(tensor.shape, Shape { dims: vec![2] });

        let (re0, im0) = tensor.elements[0]
            .as_complex64()
            .expect("complex64 element");
        assert!((re0 - 1.0).abs() < 1e-6);
        assert!((im0 - 2.0).abs() < 1e-6);

        let (re1, im1) = tensor.elements[1]
            .as_complex64()
            .expect("complex64 element");
        assert!((re1 - 7.0).abs() < 1e-6);
        assert!((im1 - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_clamp_complex_value_reports_unsupported_dtype() {
        let err = eval_primitive(
            Primitive::Clamp,
            &[
                Value::scalar_complex128(1.0, 0.0),
                Value::scalar_f64(0.0),
                Value::scalar_f64(2.0),
            ],
            &no_params(),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("clamp is not supported for complex dtypes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_complex_neg() {
        let out = eval_primitive(
            Primitive::Neg,
            &[Value::scalar_complex128(1.0, 2.0)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_complex128(-1.0, -2.0));
    }

    #[test]
    fn test_complex_abs() {
        let out = eval_primitive(
            Primitive::Abs,
            &[Value::scalar_complex128(3.0, 4.0)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_f64(5.0));
    }

    #[test]
    fn test_complex_exp() {
        let out = eval_primitive(
            Primitive::Exp,
            &[Value::scalar_complex128(0.0, std::f64::consts::PI)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, -1.0, 0.0, 1e-10);
    }

    #[test]
    fn test_complex_log() {
        let exp_out = eval_primitive(
            Primitive::Exp,
            &[Value::scalar_complex128(2.0, 0.0)],
            &no_params(),
        )
        .unwrap();
        let out = eval_primitive(Primitive::Log, &[exp_out], &no_params()).unwrap();
        assert_complex128_close(&out, 2.0, 0.0, 1e-12);
    }

    #[test]
    fn test_complex_expm1() {
        let out = eval_primitive(
            Primitive::Expm1,
            &[Value::scalar_complex128(0.0, std::f64::consts::PI)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, -2.0, 0.0, 1e-10);
    }

    #[test]
    fn test_complex_log1p() {
        let out = eval_primitive(
            Primitive::Log1p,
            &[Value::scalar_complex128(0.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(
            &out,
            2.0_f64.sqrt().ln(),
            std::f64::consts::FRAC_PI_4,
            1e-12,
        );
    }

    #[test]
    fn test_complex64_tensor_log1p_preserves_dtype_and_shape() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(0.0, 0.0),
                    Literal::from_complex64(0.0, 1.0),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::Complex64);
        assert_eq!(tensor.shape, Shape { dims: vec![2] });

        let (re0, im0) = tensor.elements[0]
            .as_complex64()
            .expect("complex64 element");
        assert!(re0.abs() < 1e-6);
        assert!(im0.abs() < 1e-6);

        let (re1, im1) = tensor.elements[1]
            .as_complex64()
            .expect("complex64 element");
        assert!((f64::from(re1) - 2.0_f64.sqrt().ln()).abs() < 1e-6);
        assert!((f64::from(im1) - std::f64::consts::FRAC_PI_4).abs() < 1e-6);
    }

    #[test]
    fn test_complex_asin_uses_principal_branch() {
        let out = eval_primitive(
            Primitive::Asin,
            &[Value::scalar_complex128(1.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.6662394324925153, 1.0612750619050357, 1e-12);
    }

    #[test]
    fn test_complex_acos_uses_principal_branch() {
        let out = eval_primitive(
            Primitive::Acos,
            &[Value::scalar_complex128(1.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.9045568943023814, -1.0612750619050357, 1e-12);
    }

    #[test]
    fn test_complex_atan_uses_principal_branch() {
        let out = eval_primitive(
            Primitive::Atan,
            &[Value::scalar_complex128(1.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 1.0172219678978514, 0.40235947810852507, 1e-12);
    }

    #[test]
    fn test_complex64_tensor_atan_preserves_dtype_and_shape() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(1.0, 0.0),
                    Literal::from_complex64(1.0, 1.0),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(Primitive::Atan, &[input], &no_params()).unwrap();
        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::Complex64);
        assert_eq!(tensor.shape, Shape { dims: vec![2] });

        let (re0, im0) = tensor.elements[0]
            .as_complex64()
            .expect("complex64 element");
        assert!((f64::from(re0) - std::f64::consts::FRAC_PI_4).abs() < 1e-6);
        assert!(im0.abs() < 1e-6);

        let (re1, im1) = tensor.elements[1]
            .as_complex64()
            .expect("complex64 element");
        assert!((f64::from(re1) - 1.0172219678978514).abs() < 1e-6);
        assert!((f64::from(im1) - 0.40235947810852507).abs() < 1e-6);
    }

    #[test]
    fn test_complex_logistic_scalar() {
        let out = eval_primitive(
            Primitive::Logistic,
            &[Value::scalar_complex128(1.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.7820415706337492, 0.2019482276580129, 1e-12);
    }

    #[test]
    fn test_complex64_tensor_logistic_preserves_dtype_and_shape() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(0.0, 0.0),
                    Literal::from_complex64(-1.0, 0.5),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(Primitive::Logistic, &[input], &no_params()).unwrap();
        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::Complex64);
        assert_eq!(tensor.shape, Shape { dims: vec![2] });

        let (re0, im0) = tensor.elements[0]
            .as_complex64()
            .expect("complex64 element");
        assert!((f64::from(re0) - 0.5).abs() < 1e-6);
        assert!(im0.abs() < 1e-6);

        let (re1, im1) = tensor.elements[1]
            .as_complex64()
            .expect("complex64 element");
        assert!((f64::from(re1) - 0.25725635948793235).abs() < 1e-6);
        assert!((f64::from(im1) - 0.09902772497567477).abs() < 1e-6);
    }

    #[test]
    fn test_complex_cbrt_fails_closed() {
        // JAX `cbrt_p = standard_unop(_float)` rejects complex input; fj-lax matches
        // (frankenjax-w8u0a, commit eb5ad225 fails closed on the float-only complex
        // unary ops). So complex Cbrt must error rather than return a principal-branch
        // approximation. (Updated from the old principal-branch assertion left stale by
        // the w8u0a fail-close.)
        let result = eval_primitive(
            Primitive::Cbrt,
            &[Value::scalar_complex128(1.0, 1.0)],
            &no_params(),
        );
        assert!(
            result.is_err(),
            "complex Cbrt must fail closed (JAX float-only), got {result:?}"
        );
    }

    #[test]
    fn test_complex64_tensor_cbrt_fails_closed() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(0.0, 0.0),
                    Literal::from_complex64(-8.0, 0.0),
                ],
            )
            .unwrap(),
        );

        let result = eval_primitive(Primitive::Cbrt, &[input], &no_params());
        assert!(
            result.is_err(),
            "complex64 Cbrt must fail closed (JAX float-only), got {result:?}"
        );
    }

    #[test]
    fn test_complex_sin() {
        let out = eval_primitive(
            Primitive::Sin,
            &[Value::scalar_complex128(0.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, 1.0_f64.sinh(), 1e-12);
    }

    #[test]
    fn test_complex_cos() {
        let out = eval_primitive(
            Primitive::Cos,
            &[Value::scalar_complex128(0.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 1.0_f64.cosh(), 0.0, 1e-12);
    }

    #[test]
    fn test_complex_tan() {
        let out = eval_primitive(
            Primitive::Tan,
            &[Value::scalar_complex128(0.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, 1.0_f64.tanh(), 1e-12);
    }

    #[test]
    fn test_complex_sinh() {
        let out = eval_primitive(
            Primitive::Sinh,
            &[Value::scalar_complex128(0.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, 1.0_f64.sin(), 1e-12);
    }

    #[test]
    fn test_complex_cosh() {
        let out = eval_primitive(
            Primitive::Cosh,
            &[Value::scalar_complex128(0.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 1.0_f64.cos(), 0.0, 1e-12);
    }

    #[test]
    fn test_complex_tanh() {
        let out = eval_primitive(
            Primitive::Tanh,
            &[Value::scalar_complex128(0.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, 1.0_f64.tan(), 1e-12);
    }

    #[test]
    fn test_asinh_real() {
        let out =
            eval_primitive(Primitive::Asinh, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        assert!((out.as_f64_scalar().unwrap() - 0.0).abs() < 1e-12);
        let out =
            eval_primitive(Primitive::Asinh, &[Value::scalar_f64(1.0)], &no_params()).unwrap();
        assert!((out.as_f64_scalar().unwrap() - 1.0_f64.asinh()).abs() < 1e-12);
    }

    #[test]
    fn test_acosh_real() {
        let out =
            eval_primitive(Primitive::Acosh, &[Value::scalar_f64(1.0)], &no_params()).unwrap();
        assert!((out.as_f64_scalar().unwrap() - 0.0).abs() < 1e-12);
        let out =
            eval_primitive(Primitive::Acosh, &[Value::scalar_f64(2.0)], &no_params()).unwrap();
        assert!((out.as_f64_scalar().unwrap() - 2.0_f64.acosh()).abs() < 1e-12);
    }

    #[test]
    fn test_atanh_real() {
        let out =
            eval_primitive(Primitive::Atanh, &[Value::scalar_f64(0.0)], &no_params()).unwrap();
        assert!((out.as_f64_scalar().unwrap() - 0.0).abs() < 1e-12);
        let out =
            eval_primitive(Primitive::Atanh, &[Value::scalar_f64(0.5)], &no_params()).unwrap();
        assert!((out.as_f64_scalar().unwrap() - 0.5_f64.atanh()).abs() < 1e-12);
    }

    #[test]
    fn test_complex_asinh() {
        let out = eval_primitive(
            Primitive::Asinh,
            &[Value::scalar_complex128(0.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, std::f64::consts::FRAC_PI_2, 1e-12);
    }

    #[test]
    fn test_complex_acosh() {
        let out = eval_primitive(
            Primitive::Acosh,
            &[Value::scalar_complex128(0.0, 0.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, std::f64::consts::FRAC_PI_2, 1e-12);
    }

    #[test]
    fn test_complex_atanh() {
        let out = eval_primitive(
            Primitive::Atanh,
            &[Value::scalar_complex128(0.0, 1.0)],
            &no_params(),
        )
        .unwrap();
        assert_complex128_close(&out, 0.0, std::f64::consts::FRAC_PI_4, 1e-12);
    }

    #[test]
    fn test_complex_conj_through_arithmetic() {
        let a = Value::scalar_complex128(1.0, 2.0);
        let b = Value::scalar_complex128(3.0, -0.5);
        let a_plus_b =
            eval_primitive(Primitive::Add, &[a.clone(), b.clone()], &no_params()).unwrap();
        let lhs = eval_primitive(Primitive::Conj, &[a_plus_b], &no_params()).unwrap();

        let conj_a = eval_primitive(Primitive::Conj, &[a], &no_params()).unwrap();
        let conj_b = eval_primitive(Primitive::Conj, &[b], &no_params()).unwrap();
        let rhs = eval_primitive(Primitive::Add, &[conj_a, conj_b], &no_params()).unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_complex_batch_operations() {
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape::vector(2),
                vec![
                    Literal::from_complex128(1.0, 2.0),
                    Literal::from_complex128(-1.0, 0.5),
                ],
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape::vector(2),
                vec![
                    Literal::from_complex128(3.0, -4.0),
                    Literal::from_complex128(2.0, 1.5),
                ],
            )
            .unwrap(),
        );

        let add =
            eval_primitive(Primitive::Add, &[lhs.clone(), rhs.clone()], &no_params()).unwrap();
        let add_tensor = add.as_tensor().unwrap();
        assert_eq!(add_tensor.elements[0].as_complex128(), Some((4.0, -2.0)));
        assert_eq!(add_tensor.elements[1].as_complex128(), Some((1.0, 2.0)));

        let mul = eval_primitive(Primitive::Mul, &[lhs, rhs], &no_params()).unwrap();
        let mul_tensor = mul.as_tensor().unwrap();
        assert_eq!(mul_tensor.elements[0].as_complex128(), Some((11.0, 2.0)));
        assert_eq!(mul_tensor.elements[1].as_complex128(), Some((-2.75, -0.5)));
    }

    #[test]
    fn test_conj_basic() {
        let out = eval_primitive(
            Primitive::Conj,
            &[Value::scalar_complex128(3.0, 4.0)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_complex128(3.0, -4.0));
    }

    #[test]
    fn test_conj_real() {
        let out = eval_primitive(
            Primitive::Conj,
            &[Value::scalar_complex128(5.0, 0.0)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_complex128(5.0, -0.0));
    }

    #[test]
    fn test_real_extraction() {
        let out = eval_primitive(
            Primitive::Real,
            &[Value::scalar_complex128(3.0, 4.0)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_f64(3.0));
    }

    #[test]
    fn test_imag_extraction() {
        let out = eval_primitive(
            Primitive::Imag,
            &[Value::scalar_complex128(3.0, 4.0)],
            &no_params(),
        )
        .unwrap();
        assert_eq!(out, Value::scalar_f64(4.0));
    }

    #[test]
    fn test_complex_tensor() {
        let real = Value::vector_f64(&[1.0, -2.0, 3.5]).unwrap();
        let imag = Value::vector_f64(&[0.5, 4.0, -1.5]).unwrap();

        let z = eval_primitive(
            Primitive::Complex,
            &[real.clone(), imag.clone()],
            &no_params(),
        )
        .expect("complex should construct tensor");
        let z_tensor = z.as_tensor().expect("complex should return tensor");
        assert_eq!(z_tensor.dtype, DType::Complex128);
        assert_eq!(z_tensor.shape, Shape::vector(3));
        assert_eq!(z_tensor.elements[0].as_complex128(), Some((1.0, 0.5)));
        assert_eq!(z_tensor.elements[1].as_complex128(), Some((-2.0, 4.0)));
        assert_eq!(z_tensor.elements[2].as_complex128(), Some((3.5, -1.5)));

        let conj = eval_primitive(Primitive::Conj, std::slice::from_ref(&z), &no_params())
            .expect("conj should work on complex tensor");
        let conj_tensor = conj.as_tensor().expect("conj should return tensor");
        assert_eq!(conj_tensor.dtype, DType::Complex128);
        assert_eq!(conj_tensor.elements[0].as_complex128(), Some((1.0, -0.5)));
        assert_eq!(conj_tensor.elements[1].as_complex128(), Some((-2.0, -4.0)));
        assert_eq!(conj_tensor.elements[2].as_complex128(), Some((3.5, 1.5)));

        let real_out = eval_primitive(Primitive::Real, std::slice::from_ref(&z), &no_params())
            .expect("real extraction should work");
        let imag_out = eval_primitive(Primitive::Imag, &[z], &no_params())
            .expect("imag extraction should work");
        assert_eq!(real_out, real);
        assert_eq!(imag_out, imag);
    }

    #[test]
    fn test_conj_involution() {
        let z = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape::vector(3),
                vec![
                    Literal::from_complex128(1.0, 2.0),
                    Literal::from_complex128(-3.5, -0.25),
                    Literal::from_complex128(0.0, 4.0),
                ],
            )
            .unwrap(),
        );
        let conj_once =
            eval_primitive(Primitive::Conj, std::slice::from_ref(&z), &no_params()).unwrap();
        let conj_twice = eval_primitive(Primitive::Conj, &[conj_once], &no_params()).unwrap();
        assert_eq!(conj_twice, z);
    }

    #[test]
    fn test_reduce_sum_complex_full() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape::vector(3),
                vec![
                    Literal::from_complex128(1.0, 2.0),
                    Literal::from_complex128(3.0, -4.0),
                    Literal::from_complex128(-2.0, 0.5),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &no_params()).unwrap();
        assert_eq!(out, Value::scalar_complex128(2.0, -1.5));
    }

    #[test]
    fn test_reduce_sum_complex_axis0_rank2() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::from_complex128(1.0, 1.0),
                    Literal::from_complex128(2.0, 2.0),
                    Literal::from_complex128(3.0, -1.0),
                    Literal::from_complex128(-4.0, 0.5),
                ],
            )
            .unwrap(),
        );
        let out = eval_primitive(Primitive::ReduceSum, &[input], &axes_params("0")).unwrap();
        let out_tensor = out.as_tensor().unwrap();
        assert_eq!(out_tensor.dtype, DType::Complex128);
        assert_eq!(out_tensor.shape, Shape::vector(2));
        assert_eq!(out_tensor.elements[0].as_complex128(), Some((4.0, 0.0)));
        assert_eq!(out_tensor.elements[1].as_complex128(), Some((-2.0, 2.5)));
    }

    #[test]
    fn test_complex_comparison_errors() {
        let err = eval_primitive(
            Primitive::Lt,
            &[
                Value::scalar_complex128(1.0, 1.0),
                Value::scalar_complex128(1.0, -1.0),
            ],
            &no_params(),
        )
        .expect_err("complex comparison should fail");
        assert!(matches!(
            err,
            EvalError::TypeMismatch {
                primitive: Primitive::Lt,
                ..
            }
        ));
    }

    #[test]
    fn test_complex_eq_and_ne_scalars() {
        let eq = eval_primitive(
            Primitive::Eq,
            &[
                Value::scalar_complex128(1.0, -2.0),
                Value::scalar_complex128(1.0, -2.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(eq, Value::scalar_bool(true));

        let ne = eval_primitive(
            Primitive::Ne,
            &[
                Value::scalar_complex128(1.0, -2.0),
                Value::scalar_complex128(1.0, 2.0),
            ],
            &no_params(),
        )
        .unwrap();
        assert_eq!(ne, Value::scalar_bool(true));
    }

    #[test]
    fn test_complex64_tensor_eq_is_elementwise_bool() {
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_complex64(1.0, 2.0),
                    Literal::from_complex64(3.0, 4.0),
                    Literal::from_complex64(5.0, 6.0),
                ],
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_complex64(1.0, 2.0),
                    Literal::from_complex64(3.0, -4.0),
                    Literal::from_complex64(5.0, 6.0),
                ],
            )
            .unwrap(),
        );

        let out = eval_primitive(Primitive::Eq, &[lhs, rhs], &no_params()).unwrap();
        let tensor = out.as_tensor().expect("expected tensor");
        assert_eq!(tensor.dtype, DType::Bool);
        assert_eq!(tensor.shape, Shape { dims: vec![3] });
        assert_eq!(
            tensor.elements,
            vec![
                Literal::Bool(true),
                Literal::Bool(false),
                Literal::Bool(true)
            ]
        );
    }

    proptest! {
        #[test]
        fn prop_conj_involution(re in -1.0e6f64..1.0e6f64, im in -1.0e6f64..1.0e6f64) {
            let z = Value::scalar_complex128(re, im);
            let conj_once = eval_primitive(Primitive::Conj, std::slice::from_ref(&z), &no_params()).unwrap();
            let conj_twice = eval_primitive(Primitive::Conj, &[conj_once], &no_params()).unwrap();
            prop_assert_eq!(conj_twice, z);
        }

        #[test]
        fn prop_real_imag_reconstruct(re in -1.0e6f64..1.0e6f64, im in -1.0e6f64..1.0e6f64) {
            let z = Value::scalar_complex128(re, im);
            let re_part = eval_primitive(Primitive::Real, std::slice::from_ref(&z), &no_params()).unwrap();
            let im_part = eval_primitive(Primitive::Imag, &[z], &no_params()).unwrap();
            let rebuilt = eval_primitive(Primitive::Complex, &[re_part, im_part], &no_params()).unwrap();
            prop_assert_eq!(rebuilt, Value::scalar_complex128(re, im));
        }
    }

    // ── Switch tests ─────────────────────────────────────────────────
    #[test]
    fn test_switch_two_branches() {
        let mut params = no_params();
        params.insert("num_branches".into(), "2".into());

        // Select branch 0
        let result = eval_primitive(
            Primitive::Switch,
            &[
                Value::scalar_i64(0),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &params,
        )
        .unwrap();
        assert_eq!(result, Value::scalar_f64(10.0));

        // Select branch 1
        let result = eval_primitive(
            Primitive::Switch,
            &[
                Value::scalar_i64(1),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &params,
        )
        .unwrap();
        assert_eq!(result, Value::scalar_f64(20.0));
    }

    #[test]
    fn test_switch_three_branches() {
        let mut params = no_params();
        params.insert("num_branches".into(), "3".into());

        for idx in 0..3 {
            let branches: Vec<Value> = vec![
                Value::scalar_f64(100.0),
                Value::scalar_f64(200.0),
                Value::scalar_f64(300.0),
            ];
            let mut inputs = vec![Value::scalar_i64(idx)];
            inputs.extend(branches);

            let result = eval_primitive(Primitive::Switch, &inputs, &params).unwrap();
            let expected = (idx + 1) as f64 * 100.0;
            assert_eq!(result, Value::scalar_f64(expected));
        }
    }

    #[test]
    fn test_switch_high_index_clamps_to_last_branch() {
        let mut params = no_params();
        params.insert("num_branches".into(), "2".into());

        let result = eval_primitive(
            Primitive::Switch,
            &[
                Value::scalar_i64(2),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &params,
        )
        .unwrap();
        assert_eq!(result, Value::scalar_f64(20.0));
    }

    #[test]
    fn test_switch_negative_index_clamps_to_first_branch() {
        let mut params = no_params();
        params.insert("num_branches".into(), "2".into());

        let result = eval_primitive(
            Primitive::Switch,
            &[
                Value::scalar_i64(-1),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &params,
        )
        .unwrap();
        assert_eq!(result, Value::scalar_f64(10.0));
    }

    #[test]
    fn test_switch_u64_max_index_clamps_to_last_branch() {
        let mut params = no_params();
        params.insert("num_branches".into(), "2".into());

        let result = eval_primitive(
            Primitive::Switch,
            &[
                Value::scalar_u64(u64::MAX),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &params,
        )
        .unwrap();
        assert_eq!(result, Value::scalar_f64(20.0));
    }

    #[test]
    fn test_switch_bool_index() {
        let mut params = no_params();
        params.insert("num_branches".into(), "2".into());

        // false => branch 0
        let result = eval_primitive(
            Primitive::Switch,
            &[
                Value::Scalar(Literal::Bool(false)),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &params,
        )
        .unwrap();
        assert_eq!(result, Value::scalar_f64(10.0));

        // true => branch 1
        let result = eval_primitive(
            Primitive::Switch,
            &[
                Value::Scalar(Literal::Bool(true)),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &params,
        )
        .unwrap();
        assert_eq!(result, Value::scalar_f64(20.0));
    }

    #[test]
    fn test_switch_tensor_index_selects_branch() {
        let mut params = no_params();
        params.insert("num_branches".into(), "2".into());

        let index = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(1)]).unwrap(),
        );
        let result = eval_primitive(
            Primitive::Switch,
            &[index, Value::scalar_f64(10.0), Value::scalar_f64(20.0)],
            &params,
        )
        .unwrap();
        assert_eq!(result, Value::scalar_f64(20.0));
    }

    #[test]
    fn test_switch_tensor_branches() {
        let mut params = no_params();
        params.insert("num_branches".into(), "2".into());

        let t0 = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let t1 = Value::vector_f64(&[4.0, 5.0, 6.0]).unwrap();

        let result = eval_primitive(
            Primitive::Switch,
            &[Value::scalar_i64(1), t0, t1.clone()],
            &params,
        )
        .unwrap();
        assert_eq!(result, t1);
    }

    #[test]
    fn test_switch_u32_index_selects_branch() {
        let mut params = no_params();
        params.insert("num_branches".into(), "2".into());

        let result = eval_primitive(
            Primitive::Switch,
            &[
                Value::scalar_u32(0),
                Value::scalar_f64(1.0),
                Value::scalar_f64(2.0),
            ],
            &params,
        )
        .unwrap();
        assert_eq!(result, Value::scalar_f64(1.0));
    }

    #[test]
    fn test_switch_rejects_mismatched_num_branches() {
        let mut params = no_params();
        params.insert("num_branches".into(), "3".into());

        let result = eval_primitive(
            Primitive::Switch,
            &[
                Value::scalar_i64(0),
                Value::scalar_f64(1.0),
                Value::scalar_f64(2.0),
            ],
            &params,
        );
        assert!(matches!(
            result,
            Err(EvalError::Unsupported {
                primitive: Primitive::Switch,
                ..
            })
        ));
    }

    #[test]
    fn test_switch_branch_dtype_mismatch_errors() {
        let mut params = no_params();
        params.insert("num_branches".into(), "2".into());

        let result = eval_primitive(
            Primitive::Switch,
            &[
                Value::scalar_i64(0),
                Value::scalar_f64(10.0),
                Value::scalar_i64(20),
            ],
            &params,
        );
        assert!(matches!(
            result,
            Err(EvalError::TypeMismatch {
                primitive: Primitive::Switch,
                ..
            })
        ));
    }

    #[test]
    fn test_switch_branch_shape_mismatch_errors() {
        let mut params = no_params();
        params.insert("num_branches".into(), "2".into());

        let t0 = Value::vector_f64(&[1.0]).unwrap();
        let t1 = Value::vector_f64(&[1.0, 2.0]).unwrap();
        let result = eval_primitive(Primitive::Switch, &[Value::scalar_i64(0), t0, t1], &params);
        assert!(matches!(
            result,
            Err(EvalError::ShapeMismatch {
                primitive: Primitive::Switch,
                ..
            })
        ));
    }

    // ── fori_loop tests ──────────────────────────────────────────────
    #[test]
    fn test_fori_loop_sum() {
        // Sum 0..10 = 45
        let result = eval_fori_loop(0, 10, Value::scalar_i64(0), |i, val| {
            let current = val.as_i64_scalar().unwrap();
            Ok(Value::scalar_i64(current + i))
        })
        .unwrap();
        assert_eq!(result, Value::scalar_i64(45));
    }

    #[test]
    fn test_fori_loop_zero_range() {
        // lower == upper => no iterations, return init_val unchanged
        let init = Value::scalar_f64(42.0);
        let body_called = std::cell::Cell::new(false);
        let result = eval_fori_loop(5, 5, init.clone(), |_, val| {
            body_called.set(true);
            Ok(val)
        })
        .unwrap();
        assert!(
            !body_called.get(),
            "body should not be called for empty range"
        );
        assert_eq!(result, init);
    }

    #[test]
    fn test_fori_loop_negative_range() {
        // upper < lower => no iterations
        let init = Value::scalar_f64(99.0);
        let body_called = std::cell::Cell::new(false);
        let result = eval_fori_loop(10, 5, init.clone(), |_, val| {
            body_called.set(true);
            Ok(val)
        })
        .unwrap();
        assert!(
            !body_called.get(),
            "body should not be called for negative range"
        );
        assert_eq!(result, init);
    }

    #[test]
    fn test_fori_loop_factorial() {
        // Compute 5! = 120
        let result = eval_fori_loop(1, 6, Value::scalar_i64(1), |i, val| {
            let current = val.as_i64_scalar().unwrap();
            Ok(Value::scalar_i64(current * i))
        })
        .unwrap();
        assert_eq!(result, Value::scalar_i64(120));
    }

    #[test]
    fn test_fori_loop_tensor_accumulation() {
        // Accumulate into a tensor: add i to each element
        let init = Value::vector_f64(&[0.0, 0.0, 0.0]).unwrap();
        let result = eval_fori_loop(0, 3, init, |i, val| {
            let offset = Value::vector_f64(&[i as f64, i as f64, i as f64]).unwrap();
            eval_primitive(Primitive::Add, &[val, offset], &no_params())
        })
        .unwrap();
        // Each element gets 0+1+2 = 3
        assert_eq!(result, Value::vector_f64(&[3.0, 3.0, 3.0]).unwrap());
    }

    #[test]
    fn test_fori_loop_body_error_propagation() {
        let result = eval_fori_loop(0, 5, Value::scalar_i64(0), |i, _| {
            if i == 3 {
                Err(EvalError::Unsupported {
                    primitive: Primitive::While,
                    detail: "test error at i=3".into(),
                })
            } else {
                Ok(Value::scalar_i64(i))
            }
        });
        assert!(result.is_err(), "Error in body should propagate");
    }

    // ── Cbrt tests ─────────────────────────────────────────────

    #[test]
    fn test_cbrt_perfect_cube() {
        let result = eval_primitive(
            Primitive::Cbrt,
            &[Value::scalar_f64(27.0)],
            &BTreeMap::new(),
        )
        .unwrap();
        assert!((result.as_f64_scalar().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cbrt_negative() {
        let result = eval_primitive(
            Primitive::Cbrt,
            &[Value::scalar_f64(-8.0)],
            &BTreeMap::new(),
        )
        .unwrap();
        assert!((result.as_f64_scalar().unwrap() - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cbrt_zero() {
        let result =
            eval_primitive(Primitive::Cbrt, &[Value::scalar_f64(0.0)], &BTreeMap::new()).unwrap();
        assert!((result.as_f64_scalar().unwrap()).abs() < 1e-10);
    }

    // ── IsFinite tests ─────────────────────────────────────────

    #[test]
    fn test_is_finite_normal() {
        let result = eval_primitive(
            Primitive::IsFinite,
            &[Value::scalar_f64(1.0)],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(result, Value::Scalar(Literal::Bool(true)));
    }

    #[test]
    fn test_is_finite_inf() {
        let result = eval_primitive(
            Primitive::IsFinite,
            &[Value::scalar_f64(f64::INFINITY)],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(result, Value::Scalar(Literal::Bool(false)));
    }

    #[test]
    fn test_is_finite_nan() {
        let result = eval_primitive(
            Primitive::IsFinite,
            &[Value::scalar_f64(f64::NAN)],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(result, Value::Scalar(Literal::Bool(false)));
    }

    #[test]
    fn test_is_finite_tensor() {
        let tensor = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(4),
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(f64::INFINITY),
                    Literal::from_f64(f64::NAN),
                    Literal::from_f64(-42.0),
                ],
            )
            .unwrap(),
        );
        let result = eval_primitive(Primitive::IsFinite, &[tensor], &BTreeMap::new()).unwrap();
        if let Value::Tensor(t) = &result {
            assert_eq!(t.dtype, DType::Bool);
            assert_eq!(
                t.elements,
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ]
            );
        } else {
            assert!(matches!(result, Value::Tensor(_)), "expected tensor");
        }
    }

    // ── IntegerPow tests ───────────────────────────────────────

    #[test]
    fn test_integer_pow_positive() {
        let mut params = BTreeMap::new();
        params.insert("exponent".into(), "4".into());
        let result =
            eval_primitive(Primitive::IntegerPow, &[Value::scalar_f64(3.0)], &params).unwrap();
        assert!((result.as_f64_scalar().unwrap() - 81.0).abs() < 1e-10);
    }

    #[test]
    fn test_integer_pow_zero() {
        let mut params = BTreeMap::new();
        params.insert("exponent".into(), "0".into());
        let result =
            eval_primitive(Primitive::IntegerPow, &[Value::scalar_f64(5.0)], &params).unwrap();
        assert!((result.as_f64_scalar().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_integer_pow_negative() {
        let mut params = BTreeMap::new();
        params.insert("exponent".into(), "-3".into());
        let result =
            eval_primitive(Primitive::IntegerPow, &[Value::scalar_f64(2.0)], &params).unwrap();
        assert!((result.as_f64_scalar().unwrap() - 0.125).abs() < 1e-10);
    }

    // ── Special function tests ────────────────────────────────

    #[test]
    fn test_lgamma_positive_int() {
        let out = eval_primitive(Primitive::Lgamma, &[Value::scalar_f64(5.0)], &no_params())
            .expect("lgamma should succeed");
        let actual = out.as_f64_scalar().expect("scalar");
        let expected = 24.0_f64.ln();
        assert!(
            (actual - expected).abs() < 1e-9,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn test_lgamma_half_int() {
        let out = eval_primitive(Primitive::Lgamma, &[Value::scalar_f64(0.5)], &no_params())
            .expect("lgamma should succeed");
        let actual = out.as_f64_scalar().expect("scalar");
        let expected = std::f64::consts::PI.sqrt().ln();
        assert!(
            (actual - expected).abs() < 1e-9,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn test_lgamma_negative() {
        let out = eval_primitive(Primitive::Lgamma, &[Value::scalar_f64(-0.5)], &no_params())
            .expect("lgamma should succeed");
        let actual = out.as_f64_scalar().expect("scalar");
        let expected = (2.0 * std::f64::consts::PI.sqrt()).ln();
        assert!(
            (actual - expected).abs() < 1e-9,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn test_digamma_positive_int() {
        let out = eval_primitive(Primitive::Digamma, &[Value::scalar_f64(1.0)], &no_params())
            .expect("digamma should succeed");
        let actual = out.as_f64_scalar().expect("scalar");
        let expected = -0.577_215_664_901_532_9_f64;
        assert!(
            (actual - expected).abs() < 1e-9,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn test_digamma_large() {
        let out = eval_primitive(
            Primitive::Digamma,
            &[Value::scalar_f64(100.0)],
            &no_params(),
        )
        .expect("digamma should succeed");
        let actual = out.as_f64_scalar().expect("scalar");
        let expected = 100.0_f64.ln() - 1.0 / (2.0 * 100.0);
        assert!(
            (actual - expected).abs() < 2e-5,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn test_erf_inv_roundtrip() {
        for x in [-2.5_f64, -1.0, -0.1, 0.0, 0.1, 0.9, 2.5] {
            let erf_x = eval_primitive(Primitive::Erf, &[Value::scalar_f64(x)], &no_params())
                .expect("erf should succeed")
                .as_f64_scalar()
                .expect("scalar");
            let roundtrip =
                eval_primitive(Primitive::ErfInv, &[Value::scalar_f64(erf_x)], &no_params())
                    .expect("erf_inv should succeed")
                    .as_f64_scalar()
                    .expect("scalar");
            assert!(
                (roundtrip - x).abs() < 2e-3,
                "x={x}, erf(x)={erf_x}, erf_inv(erf(x))={roundtrip}"
            );
        }
    }

    #[test]
    fn test_erf_inv_boundaries() {
        let at_zero = eval_primitive(Primitive::ErfInv, &[Value::scalar_f64(0.0)], &no_params())
            .expect("erf_inv should succeed");
        assert_eq!(at_zero.as_f64_scalar().expect("scalar"), 0.0);

        let at_pos_one = eval_primitive(Primitive::ErfInv, &[Value::scalar_f64(1.0)], &no_params())
            .expect("erf_inv should succeed")
            .as_f64_scalar()
            .expect("scalar");
        assert!(at_pos_one.is_infinite() && at_pos_one.is_sign_positive());

        let at_neg_one =
            eval_primitive(Primitive::ErfInv, &[Value::scalar_f64(-1.0)], &no_params())
                .expect("erf_inv should succeed")
                .as_f64_scalar()
                .expect("scalar");
        assert!(at_neg_one.is_infinite() && at_neg_one.is_sign_negative());
    }

    #[test]
    fn test_trigamma_positive() {
        let actual = trigamma_approx(1.0);
        let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!(
            (actual - expected).abs() < 1e-9,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn test_trigamma_large() {
        let actual = trigamma_approx(100.0);
        let expected = 1.0 / 100.0 + 1.0 / (2.0 * 100.0_f64.powi(2));
        assert!(
            (actual - expected).abs() < 5e-7,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn e2e_special_functions_oracle() {
        let cases = [
            ("lgamma", 0.5_f64, 0.572_364_942_924_700_1_f64),
            ("lgamma", 5.0_f64, 3.178_053_830_347_945_8_f64),
            ("digamma", 1.0_f64, -0.577_215_664_901_532_9_f64),
            ("digamma", 5.0_f64, 1.506_117_668_431_800_3_f64),
            ("erf_inv", 0.5_f64, 0.476_936_276_204_469_9_f64),
            ("erf_inv", -0.9_f64, -1.163_087_153_676_674_3_f64),
        ];

        let atol = 1e-6_f64;
        let mut rows = Vec::with_capacity(cases.len());
        let mut all_passed = true;

        for (function, input, expected) in cases {
            let primitive = match function {
                "lgamma" => Primitive::Lgamma,
                "digamma" => Primitive::Digamma,
                "erf_inv" => Primitive::ErfInv,
                _ => unreachable!("unknown function"),
            };

            let actual = eval_primitive(primitive, &[Value::scalar_f64(input)], &no_params())
                .expect("oracle eval should succeed")
                .as_f64_scalar()
                .expect("scalar");
            let abs_error = (actual - expected).abs();
            let pass = abs_error <= atol;
            all_passed &= pass;
            rows.push((function, input, expected, actual, abs_error, pass));
        }

        let case_logs = format!(
            "[{}]",
            rows.iter()
                .map(
                    |(function, input, expected, actual, abs_error, pass)| format!(
                        "{{\"test_name\":\"e2e_special_functions_oracle\",\"function\":\"{}\",\"input\":{},\"expected\":{},\"actual\":{},\"error\":{},\"pass\":{}}}",
                        function, input, expected, actual, abs_error, pass
                    )
                )
                .collect::<Vec<_>>()
                .join(",")
        );

        let forensic_log = format!(
            concat!(
                "{{",
                "\"scenario\":\"e2e_special_functions_oracle\",",
                "\"all_passed\":{},",
                "\"cases\":[{}]",
                "}}"
            ),
            all_passed,
            rows.iter()
                .map(
                    |(function, input, expected, actual, abs_error, pass)| format!(
                        "{{\"function\":\"{}\",\"input\":{},\"expected\":{},\"actual\":{},\"abs_error\":{},\"pass\":{}}}",
                        function, input, expected, actual, abs_error, pass
                    )
                )
                .collect::<Vec<_>>()
                .join(",")
        );

        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts");
        let e2e_path = root.join("e2e/e2e_special_functions.e2e.json");
        let test_log_path = root.join("testing/logs/fj-lax/e2e_special_functions_oracle.json");

        if let Some(parent) = e2e_path.parent() {
            std::fs::create_dir_all(parent).expect("create e2e artifact dir");
        }
        if let Some(parent) = test_log_path.parent() {
            std::fs::create_dir_all(parent).expect("create test log dir");
        }

        std::fs::write(&e2e_path, forensic_log).expect("write special e2e forensic log");
        std::fs::write(&test_log_path, case_logs).expect("write special test logs");

        assert!(all_passed, "special function oracle mismatch");
    }

    #[test]
    fn e2e_special_function_error_paths_graceful() {
        // These primitives are implemented and should reject invalid scalar input cleanly.
        // Verify they all reject invalid (scalar) input gracefully.
        let implemented = [
            ("cholesky", Primitive::Cholesky),
            ("triangular_solve", Primitive::TriangularSolve),
            ("qr", Primitive::Qr),
            ("svd", Primitive::Svd),
            ("eigh", Primitive::Eigh),
            ("fft", Primitive::Fft),
            ("ifft", Primitive::Ifft),
            ("rfft", Primitive::Rfft),
            ("irfft", Primitive::Irfft),
        ];

        let mut all_passed = true;
        let mut rows: Vec<(&str, bool)> = Vec::new();
        for (function, primitive) in implemented {
            let result = eval_primitive(primitive, &[Value::scalar_f64(1.0)], &no_params());
            let pass = result.is_err();
            all_passed &= pass;
            rows.push((function, pass));
        }

        let case_logs = format!(
            "[{}]",
            rows.iter()
                .map(|(function, pass)| format!(
                    "{{\"test_name\":\"e2e_special_function_error_paths_graceful\",\"function\":{:?},\"pass\":{}}}",
                    function, pass
                ))
                .collect::<Vec<_>>()
                .join(",")
        );

        let forensic_log = format!(
            "{{\"scenario\":\"e2e_special_function_error_paths_graceful\",\"all_passed\":{},\"cases\":[{}]}}",
            all_passed,
            rows.iter()
                .map(|(function, pass)| format!(
                    "{{\"function\":{:?},\"pass\":{}}}",
                    function, pass
                ))
                .collect::<Vec<_>>()
                .join(",")
        );

        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts");
        let e2e_path = root.join("e2e/e2e_special_function_error_paths.e2e.json");
        let test_log_path =
            root.join("testing/logs/fj-lax/e2e_special_function_error_paths_graceful.json");

        if let Some(parent) = e2e_path.parent() {
            std::fs::create_dir_all(parent).expect("create e2e artifact dir");
        }
        if let Some(parent) = test_log_path.parent() {
            std::fs::create_dir_all(parent).expect("create test log dir");
        }

        std::fs::write(&e2e_path, forensic_log).expect("write special-function e2e forensic log");
        std::fs::write(&test_log_path, case_logs).expect("write special-function test logs");

        assert!(
            all_passed,
            "implemented primitives should reject scalar input"
        );
    }

    // ── Nextafter tests ────────────────────────────────────────

    #[test]
    fn test_nextafter_up() {
        let result = eval_primitive(
            Primitive::Nextafter,
            &[Value::scalar_f64(1.0), Value::scalar_f64(2.0)],
            &BTreeMap::new(),
        )
        .unwrap();
        let v = result.as_f64_scalar().unwrap();
        assert!(v > 1.0, "nextafter(1.0, 2.0) should be > 1.0");
        assert!(
            v - 1.0 < 1e-15,
            "nextafter(1.0, 2.0) should be very close to 1.0"
        );
    }

    #[test]
    fn test_nextafter_down() {
        let result = eval_primitive(
            Primitive::Nextafter,
            &[Value::scalar_f64(1.0), Value::scalar_f64(0.0)],
            &BTreeMap::new(),
        )
        .unwrap();
        let v = result.as_f64_scalar().unwrap();
        assert!(v < 1.0, "nextafter(1.0, 0.0) should be < 1.0");
        assert!(
            1.0 - v < 1e-15,
            "nextafter(1.0, 0.0) should be very close to 1.0"
        );
    }

    // ── Metamorphic property tests ────────────────────────────────────────
    // These verify algebraic invariants that must hold across all inputs.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn metamorphic_exp_sum_is_product(a in -10.0f64..10.0, b in -10.0f64..10.0) {
            // exp(a + b) == exp(a) * exp(b)
            let sum = eval_primitive(
                Primitive::Add,
                &[Value::scalar_f64(a), Value::scalar_f64(b)],
                &BTreeMap::new(),
            ).unwrap();
            let exp_sum = eval_primitive(Primitive::Exp, &[sum], &BTreeMap::new()).unwrap();

            let exp_a = eval_primitive(Primitive::Exp, &[Value::scalar_f64(a)], &BTreeMap::new()).unwrap();
            let exp_b = eval_primitive(Primitive::Exp, &[Value::scalar_f64(b)], &BTreeMap::new()).unwrap();
            let product = eval_primitive(Primitive::Mul, &[exp_a, exp_b], &BTreeMap::new()).unwrap();

            let lhs = exp_sum.as_f64_scalar().unwrap();
            let rhs = product.as_f64_scalar().unwrap();
            let rel_err = if rhs.abs() > 1e-10 { (lhs - rhs).abs() / rhs.abs() } else { (lhs - rhs).abs() };
            prop_assert!(rel_err < 1e-10, "exp(a+b) != exp(a)*exp(b): {} != {}", lhs, rhs);
        }

        #[test]
        fn metamorphic_log_product_is_sum(a in 0.1f64..100.0, b in 0.1f64..100.0) {
            // log(a * b) == log(a) + log(b) for positive a, b
            let product = eval_primitive(
                Primitive::Mul,
                &[Value::scalar_f64(a), Value::scalar_f64(b)],
                &BTreeMap::new(),
            ).unwrap();
            let log_product = eval_primitive(Primitive::Log, &[product], &BTreeMap::new()).unwrap();

            let log_a = eval_primitive(Primitive::Log, &[Value::scalar_f64(a)], &BTreeMap::new()).unwrap();
            let log_b = eval_primitive(Primitive::Log, &[Value::scalar_f64(b)], &BTreeMap::new()).unwrap();
            let sum = eval_primitive(Primitive::Add, &[log_a, log_b], &BTreeMap::new()).unwrap();

            let lhs = log_product.as_f64_scalar().unwrap();
            let rhs = sum.as_f64_scalar().unwrap();
            prop_assert!((lhs - rhs).abs() < 1e-10, "log(a*b) != log(a)+log(b): {} != {}", lhs, rhs);
        }

        #[test]
        fn metamorphic_sin_odd_function(x in -10.0f64..10.0) {
            // sin(-x) == -sin(x)
            let neg_x = eval_primitive(Primitive::Neg, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let sin_neg_x = eval_primitive(Primitive::Sin, &[neg_x], &BTreeMap::new()).unwrap();

            let sin_x = eval_primitive(Primitive::Sin, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let neg_sin_x = eval_primitive(Primitive::Neg, &[sin_x], &BTreeMap::new()).unwrap();

            let lhs = sin_neg_x.as_f64_scalar().unwrap();
            let rhs = neg_sin_x.as_f64_scalar().unwrap();
            prop_assert!((lhs - rhs).abs() < 1e-14, "sin(-x) != -sin(x): {} != {}", lhs, rhs);
        }

        #[test]
        fn metamorphic_cos_even_function(x in -10.0f64..10.0) {
            // cos(-x) == cos(x)
            let neg_x = eval_primitive(Primitive::Neg, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let cos_neg_x = eval_primitive(Primitive::Cos, &[neg_x], &BTreeMap::new()).unwrap();
            let cos_x = eval_primitive(Primitive::Cos, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();

            let lhs = cos_neg_x.as_f64_scalar().unwrap();
            let rhs = cos_x.as_f64_scalar().unwrap();
            prop_assert!((lhs - rhs).abs() < 1e-14, "cos(-x) != cos(x): {} != {}", lhs, rhs);
        }

        #[test]
        fn metamorphic_sinh_odd_function(x in -5.0f64..5.0) {
            // sinh(-x) == -sinh(x)
            let neg_x = eval_primitive(Primitive::Neg, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let sinh_neg_x = eval_primitive(Primitive::Sinh, &[neg_x], &BTreeMap::new()).unwrap();

            let sinh_x = eval_primitive(Primitive::Sinh, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let neg_sinh_x = eval_primitive(Primitive::Neg, &[sinh_x], &BTreeMap::new()).unwrap();

            let lhs = sinh_neg_x.as_f64_scalar().unwrap();
            let rhs = neg_sinh_x.as_f64_scalar().unwrap();
            prop_assert!((lhs - rhs).abs() < 1e-12, "sinh(-x) != -sinh(x): {} != {}", lhs, rhs);
        }

        #[test]
        fn metamorphic_cosh_even_function(x in -5.0f64..5.0) {
            // cosh(-x) == cosh(x)
            let neg_x = eval_primitive(Primitive::Neg, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let cosh_neg_x = eval_primitive(Primitive::Cosh, &[neg_x], &BTreeMap::new()).unwrap();
            let cosh_x = eval_primitive(Primitive::Cosh, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();

            let lhs = cosh_neg_x.as_f64_scalar().unwrap();
            let rhs = cosh_x.as_f64_scalar().unwrap();
            prop_assert!((lhs - rhs).abs() < 1e-12, "cosh(-x) != cosh(x): {} != {}", lhs, rhs);
        }

        #[test]
        fn metamorphic_pythagorean_identity(x in -10.0f64..10.0) {
            // sin^2(x) + cos^2(x) == 1
            let sin_x = eval_primitive(Primitive::Sin, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let cos_x = eval_primitive(Primitive::Cos, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let sin2 = eval_primitive(Primitive::Mul, &[sin_x.clone(), sin_x], &BTreeMap::new()).unwrap();
            let cos2 = eval_primitive(Primitive::Mul, &[cos_x.clone(), cos_x], &BTreeMap::new()).unwrap();
            let sum = eval_primitive(Primitive::Add, &[sin2, cos2], &BTreeMap::new()).unwrap();

            let result = sum.as_f64_scalar().unwrap();
            prop_assert!((result - 1.0).abs() < 1e-14, "sin^2(x) + cos^2(x) != 1: got {}", result);
        }

        #[test]
        fn metamorphic_hyperbolic_identity(x in -5.0f64..5.0) {
            // cosh^2(x) - sinh^2(x) == 1
            let sinh_x = eval_primitive(Primitive::Sinh, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let cosh_x = eval_primitive(Primitive::Cosh, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let sinh2 = eval_primitive(Primitive::Mul, &[sinh_x.clone(), sinh_x], &BTreeMap::new()).unwrap();
            let cosh2 = eval_primitive(Primitive::Mul, &[cosh_x.clone(), cosh_x], &BTreeMap::new()).unwrap();
            let diff = eval_primitive(Primitive::Sub, &[cosh2, sinh2], &BTreeMap::new()).unwrap();

            let result = diff.as_f64_scalar().unwrap();
            prop_assert!((result - 1.0).abs() < 1e-10, "cosh^2(x) - sinh^2(x) != 1: got {}", result);
        }

        #[test]
        fn metamorphic_exp_log_inverse(x in 0.01f64..100.0) {
            // exp(log(x)) == x for positive x
            let log_x = eval_primitive(Primitive::Log, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let exp_log_x = eval_primitive(Primitive::Exp, &[log_x], &BTreeMap::new()).unwrap();

            let result = exp_log_x.as_f64_scalar().unwrap();
            let rel_err = (result - x).abs() / x.abs();
            prop_assert!(rel_err < 1e-14, "exp(log(x)) != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_sqrt_square_inverse(x in 0.0f64..1000.0) {
            // sqrt(x)^2 == x for non-negative x
            let sqrt_x = eval_primitive(Primitive::Sqrt, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let sq = eval_primitive(Primitive::Mul, &[sqrt_x.clone(), sqrt_x], &BTreeMap::new()).unwrap();

            let result = sq.as_f64_scalar().unwrap();
            let rel_err = if x > 1e-10 { (result - x).abs() / x } else { (result - x).abs() };
            prop_assert!(rel_err < 1e-14, "sqrt(x)^2 != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_tan_odd_function(x in -1.5f64..1.5) {
            // tan(-x) == -tan(x), avoiding poles near ±π/2
            let neg_x = eval_primitive(Primitive::Neg, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let tan_neg_x = eval_primitive(Primitive::Tan, &[neg_x], &BTreeMap::new()).unwrap();

            let tan_x = eval_primitive(Primitive::Tan, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let neg_tan_x = eval_primitive(Primitive::Neg, &[tan_x], &BTreeMap::new()).unwrap();

            let lhs = tan_neg_x.as_f64_scalar().unwrap();
            let rhs = neg_tan_x.as_f64_scalar().unwrap();
            prop_assert!((lhs - rhs).abs() < 1e-12, "tan(-x) != -tan(x): {} != {}", lhs, rhs);
        }

        #[test]
        fn metamorphic_tanh_odd_function(x in -5.0f64..5.0) {
            // tanh(-x) == -tanh(x)
            let neg_x = eval_primitive(Primitive::Neg, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let tanh_neg_x = eval_primitive(Primitive::Tanh, &[neg_x], &BTreeMap::new()).unwrap();

            let tanh_x = eval_primitive(Primitive::Tanh, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let neg_tanh_x = eval_primitive(Primitive::Neg, &[tanh_x], &BTreeMap::new()).unwrap();

            let lhs = tanh_neg_x.as_f64_scalar().unwrap();
            let rhs = neg_tanh_x.as_f64_scalar().unwrap();
            prop_assert!((lhs - rhs).abs() < 1e-14, "tanh(-x) != -tanh(x): {} != {}", lhs, rhs);
        }

        #[test]
        fn metamorphic_rsqrt_identity(x in 0.01f64..1000.0) {
            // rsqrt(x)^2 * x == 1 for positive x
            let rsqrt_x = eval_primitive(Primitive::Rsqrt, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let rsqrt2 = eval_primitive(Primitive::Mul, &[rsqrt_x.clone(), rsqrt_x], &BTreeMap::new()).unwrap();
            let product = eval_primitive(Primitive::Mul, &[rsqrt2, Value::scalar_f64(x)], &BTreeMap::new()).unwrap();

            let result = product.as_f64_scalar().unwrap();
            prop_assert!((result - 1.0).abs() < 1e-12, "rsqrt(x)^2 * x != 1: got {}", result);
        }

        #[test]
        fn metamorphic_abs_even_function(x in -100.0f64..100.0) {
            // abs(-x) == abs(x)
            let neg_x = eval_primitive(Primitive::Neg, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let abs_neg_x = eval_primitive(Primitive::Abs, &[neg_x], &BTreeMap::new()).unwrap();
            let abs_x = eval_primitive(Primitive::Abs, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();

            let lhs = abs_neg_x.as_f64_scalar().unwrap();
            let rhs = abs_x.as_f64_scalar().unwrap();
            prop_assert!((lhs - rhs).abs() < 1e-15, "abs(-x) != abs(x): {} != {}", lhs, rhs);
        }

        #[test]
        fn metamorphic_cbrt_cube_inverse(x in -100.0f64..100.0) {
            // cbrt(x)^3 == x
            let cbrt_x = eval_primitive(Primitive::Cbrt, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let cbrt2 = eval_primitive(Primitive::Mul, &[cbrt_x.clone(), cbrt_x.clone()], &BTreeMap::new()).unwrap();
            let cube = eval_primitive(Primitive::Mul, &[cbrt2, cbrt_x], &BTreeMap::new()).unwrap();

            let result = cube.as_f64_scalar().unwrap();
            let rel_err = if x.abs() > 1e-10 { (result - x).abs() / x.abs() } else { (result - x).abs() };
            prop_assert!(rel_err < 1e-12, "cbrt(x)^3 != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_max_plus_min_equals_sum(a in -100.0f64..100.0, b in -100.0f64..100.0) {
            // max(a,b) + min(a,b) == a + b
            let va = Value::scalar_f64(a);
            let vb = Value::scalar_f64(b);
            let max_ab = eval_primitive(Primitive::Max, &[va.clone(), vb.clone()], &BTreeMap::new()).unwrap();
            let min_ab = eval_primitive(Primitive::Min, &[va.clone(), vb.clone()], &BTreeMap::new()).unwrap();
            let lhs = eval_primitive(Primitive::Add, &[max_ab, min_ab], &BTreeMap::new()).unwrap();
            let rhs = eval_primitive(Primitive::Add, &[va, vb], &BTreeMap::new()).unwrap();

            let lhs_val = lhs.as_f64_scalar().unwrap();
            let rhs_val = rhs.as_f64_scalar().unwrap();
            prop_assert!((lhs_val - rhs_val).abs() < 1e-12, "max(a,b)+min(a,b) != a+b: {} != {}", lhs_val, rhs_val);
        }

        #[test]
        fn metamorphic_max_min_ordering(a in -100.0f64..100.0, b in -100.0f64..100.0) {
            // min(a,b) <= max(a,b) always
            let va = Value::scalar_f64(a);
            let vb = Value::scalar_f64(b);
            let max_ab = eval_primitive(Primitive::Max, &[va.clone(), vb.clone()], &BTreeMap::new()).unwrap();
            let min_ab = eval_primitive(Primitive::Min, &[va, vb], &BTreeMap::new()).unwrap();

            let max_val = max_ab.as_f64_scalar().unwrap();
            let min_val = min_ab.as_f64_scalar().unwrap();
            prop_assert!(min_val <= max_val, "min(a,b) > max(a,b): {} > {}", min_val, max_val);
        }

        #[test]
        fn metamorphic_select_same_branches(cond in proptest::bool::ANY, a in -100.0f64..100.0) {
            // select(cond, a, a) == a for any condition
            let vcond = Value::scalar_bool(cond);
            let va = Value::scalar_f64(a);
            let result = eval_primitive(Primitive::Select, &[vcond, va.clone(), va.clone()], &BTreeMap::new()).unwrap();
            let result_val = result.as_f64_scalar().unwrap();
            prop_assert!((result_val - a).abs() < 1e-15, "select(cond, a, a) != a: {} != {}", result_val, a);
        }

        #[test]
        fn metamorphic_select_integer_cond_consistency(cond in -100i64..100, a in -100.0f64..100.0, b in -100.0f64..100.0) {
            // select with integer condition should match boolean interpretation
            let bool_cond = cond != 0;
            let vint_cond = Value::scalar_i64(cond);
            let vbool_cond = Value::scalar_bool(bool_cond);
            let va = Value::scalar_f64(a);
            let vb = Value::scalar_f64(b);

            let result_int = eval_primitive(Primitive::Select, &[vint_cond, va.clone(), vb.clone()], &BTreeMap::new()).unwrap();
            let result_bool = eval_primitive(Primitive::Select, &[vbool_cond, va, vb], &BTreeMap::new()).unwrap();

            let val_int = result_int.as_f64_scalar().unwrap();
            let val_bool = result_bool.as_f64_scalar().unwrap();
            prop_assert!((val_int - val_bool).abs() < 1e-15,
                "select(int, a, b) != select(bool, a, b): {} != {} for cond={}", val_int, val_bool, cond);
        }

        #[test]
        fn metamorphic_asin_sin_inverse(x in -0.99f64..0.99) {
            // asin(x) should satisfy sin(asin(x)) == x for x in (-1, 1)
            let asin_x = eval_primitive(Primitive::Asin, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let sin_asin_x = eval_primitive(Primitive::Sin, &[asin_x], &BTreeMap::new()).unwrap();

            let result = sin_asin_x.as_f64_scalar().unwrap();
            prop_assert!((result - x).abs() < 1e-14, "sin(asin(x)) != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_acos_cos_inverse(x in -0.99f64..0.99) {
            // acos(x) should satisfy cos(acos(x)) == x for x in (-1, 1)
            let acos_x = eval_primitive(Primitive::Acos, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let cos_acos_x = eval_primitive(Primitive::Cos, &[acos_x], &BTreeMap::new()).unwrap();

            let result = cos_acos_x.as_f64_scalar().unwrap();
            prop_assert!((result - x).abs() < 1e-14, "cos(acos(x)) != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_atan_tan_inverse(x in -10.0f64..10.0) {
            // tan(atan(x)) == x for any finite x
            let atan_x = eval_primitive(Primitive::Atan, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let tan_atan_x = eval_primitive(Primitive::Tan, &[atan_x], &BTreeMap::new()).unwrap();

            let result = tan_atan_x.as_f64_scalar().unwrap();
            let rel_err = if x.abs() > 1e-10 { (result - x).abs() / x.abs() } else { (result - x).abs() };
            prop_assert!(rel_err < 1e-12, "tan(atan(x)) != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_asinh_sinh_inverse(x in -10.0f64..10.0) {
            // sinh(asinh(x)) == x for any finite x
            let asinh_x = eval_primitive(Primitive::Asinh, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let sinh_asinh_x = eval_primitive(Primitive::Sinh, &[asinh_x], &BTreeMap::new()).unwrap();

            let result = sinh_asinh_x.as_f64_scalar().unwrap();
            let rel_err = if x.abs() > 1e-10 { (result - x).abs() / x.abs() } else { (result - x).abs() };
            prop_assert!(rel_err < 1e-12, "sinh(asinh(x)) != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_atanh_tanh_inverse(x in -0.99f64..0.99) {
            // tanh(atanh(x)) == x for x in (-1, 1)
            let atanh_x = eval_primitive(Primitive::Atanh, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let tanh_atanh_x = eval_primitive(Primitive::Tanh, &[atanh_x], &BTreeMap::new()).unwrap();

            let result = tanh_atanh_x.as_f64_scalar().unwrap();
            prop_assert!((result - x).abs() < 1e-14, "tanh(atanh(x)) != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_acosh_cosh_inverse(x in 1.01f64..100.0) {
            // cosh(acosh(x)) == x for x >= 1
            let acosh_x = eval_primitive(Primitive::Acosh, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let cosh_acosh_x = eval_primitive(Primitive::Cosh, &[acosh_x], &BTreeMap::new()).unwrap();

            let result = cosh_acosh_x.as_f64_scalar().unwrap();
            let rel_err = (result - x).abs() / x;
            prop_assert!(rel_err < 1e-12, "cosh(acosh(x)) != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_erf_erfinv_inverse(x in -0.99f64..0.99) {
            // erfinv(erf(x)) should approximately equal x for x in (-1, 1)
            // Note: due to numerical precision limits, we use looser tolerance
            let erf_x = eval_primitive(Primitive::Erf, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let erfinv_erf_x = eval_primitive(Primitive::ErfInv, &[erf_x], &BTreeMap::new()).unwrap();

            let result = erfinv_erf_x.as_f64_scalar().unwrap();
            let abs_err = (result - x).abs();
            prop_assert!(abs_err < 1e-10, "erfinv(erf(x)) != x: {} != {} (err={})", result, x, abs_err);
        }

        #[test]
        fn metamorphic_reciprocal_involution(x in -100.0f64..100.0) {
            // 1/(1/x) == x for non-zero x
            prop_assume!(x.abs() > 1e-10);
            let recip_x = eval_primitive(Primitive::Reciprocal, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let recip_recip_x = eval_primitive(Primitive::Reciprocal, &[recip_x], &BTreeMap::new()).unwrap();

            let result = recip_recip_x.as_f64_scalar().unwrap();
            let rel_err = (result - x).abs() / x.abs();
            prop_assert!(rel_err < 1e-14, "1/(1/x) != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_neg_involution(x in -100.0f64..100.0) {
            // -(-x) == x
            let neg_x = eval_primitive(Primitive::Neg, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let neg_neg_x = eval_primitive(Primitive::Neg, &[neg_x], &BTreeMap::new()).unwrap();

            let result = neg_neg_x.as_f64_scalar().unwrap();
            prop_assert!((result - x).abs() < 1e-15, "-(-x) != x: {} != {}", result, x);
        }

        #[test]
        fn metamorphic_reduce_sum_ones(len in 1usize..20) {
            // sum of len ones should equal len
            let ones: Vec<f64> = vec![1.0; len];
            let tensor = Value::vector_f64(&ones).unwrap();
            let result = eval_primitive(Primitive::ReduceSum, &[tensor], &BTreeMap::new()).unwrap();
            let sum = result.as_f64_scalar().unwrap();
            prop_assert!((sum - len as f64).abs() < 1e-12, "sum(ones) != len: {} != {}", sum, len);
        }

        #[test]
        fn metamorphic_reduce_prod_ones(len in 1usize..20) {
            // product of len ones should equal 1
            let ones: Vec<f64> = vec![1.0; len];
            let tensor = Value::vector_f64(&ones).unwrap();
            let result = eval_primitive(Primitive::ReduceProd, &[tensor], &BTreeMap::new()).unwrap();
            let prod = result.as_f64_scalar().unwrap();
            prop_assert!((prod - 1.0).abs() < 1e-14, "prod(ones) != 1: {}", prod);
        }

        #[test]
        fn metamorphic_reduce_max_min_ordering(a in -100.0f64..100.0, b in -100.0f64..100.0, c in -100.0f64..100.0) {
            // min(vec) <= max(vec) always
            let tensor = Value::vector_f64(&[a, b, c]).unwrap();
            let max_result = eval_primitive(Primitive::ReduceMax, std::slice::from_ref(&tensor), &BTreeMap::new()).unwrap();
            let min_result = eval_primitive(Primitive::ReduceMin, &[tensor], &BTreeMap::new()).unwrap();
            let max_val = max_result.as_f64_scalar().unwrap();
            let min_val = min_result.as_f64_scalar().unwrap();
            prop_assert!(min_val <= max_val, "min > max: {} > {}", min_val, max_val);
        }

        #[test]
        fn metamorphic_reduce_sum_linearity(a in -10.0f64..10.0, b in -10.0f64..10.0, c in -10.0f64..10.0) {
            // sum([a,b,c]) == a + b + c
            let tensor = Value::vector_f64(&[a, b, c]).unwrap();
            let sum_result = eval_primitive(Primitive::ReduceSum, &[tensor], &BTreeMap::new()).unwrap();
            let sum_val = sum_result.as_f64_scalar().unwrap();
            let expected = a + b + c;
            prop_assert!((sum_val - expected).abs() < 1e-12, "sum([a,b,c]) != a+b+c: {} != {}", sum_val, expected);
        }

        #[test]
        fn metamorphic_square_equals_mul_self(x in -100.0f64..100.0) {
            // square(x) == x * x
            let square_x = eval_primitive(Primitive::Square, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let mul_xx = eval_primitive(Primitive::Mul, &[Value::scalar_f64(x), Value::scalar_f64(x)], &BTreeMap::new()).unwrap();

            let sq_val = square_x.as_f64_scalar().unwrap();
            let mul_val = mul_xx.as_f64_scalar().unwrap();
            prop_assert!((sq_val - mul_val).abs() < 1e-14, "square(x) != x*x: {} != {}", sq_val, mul_val);
        }

        #[test]
        fn metamorphic_floor_ceil_ordering(x in -100.0f64..100.0) {
            // floor(x) <= x <= ceil(x) always
            let floor_x = eval_primitive(Primitive::Floor, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let ceil_x = eval_primitive(Primitive::Ceil, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();

            let floor_val = floor_x.as_f64_scalar().unwrap();
            let ceil_val = ceil_x.as_f64_scalar().unwrap();
            prop_assert!(floor_val <= x, "floor(x) > x: {} > {}", floor_val, x);
            prop_assert!(x <= ceil_val, "x > ceil(x): {} > {}", x, ceil_val);
            prop_assert!(floor_val <= ceil_val, "floor > ceil: {} > {}", floor_val, ceil_val);
        }

        #[test]
        fn metamorphic_logistic_range(x in -20.0f64..20.0) {
            // logistic(x) is always in (0, 1)
            let logistic_x = eval_primitive(Primitive::Logistic, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let val = logistic_x.as_f64_scalar().unwrap();
            prop_assert!(val > 0.0 && val < 1.0, "logistic(x) not in (0,1): {}", val);
        }

        #[test]
        fn metamorphic_sign_values(x in -100.0f64..100.0) {
            // sign(x) is in {-1, 0, 1}
            prop_assume!(x != 0.0); // skip zero to avoid sign(0) ambiguity
            let sign_x = eval_primitive(Primitive::Sign, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let val = sign_x.as_f64_scalar().unwrap();
            prop_assert!(val == -1.0 || val == 1.0, "sign(x) not in {{-1, 1}}: {}", val);
            prop_assert!((val > 0.0) == (x > 0.0), "sign(x) wrong sign: sign({}) = {}", x, val);
        }

        #[test]
        fn metamorphic_pow_zero_exponent(x in 0.1f64..100.0) {
            // x^0 = 1 for x > 0
            let result = eval_primitive(Primitive::Pow, &[Value::scalar_f64(x), Value::scalar_f64(0.0)], &BTreeMap::new()).unwrap();
            let val = result.as_f64_scalar().unwrap();
            prop_assert!((val - 1.0).abs() < 1e-14, "x^0 != 1: {}^0 = {}", x, val);
        }

        #[test]
        fn metamorphic_pow_one_exponent(x in -100.0f64..100.0) {
            // x^1 = x
            let result = eval_primitive(Primitive::Pow, &[Value::scalar_f64(x), Value::scalar_f64(1.0)], &BTreeMap::new()).unwrap();
            let val = result.as_f64_scalar().unwrap();
            prop_assert!((val - x).abs() < 1e-14, "x^1 != x: {}^1 = {}", x, val);
        }

        #[test]
        fn metamorphic_pow_two_equals_square(x in -10.0f64..10.0) {
            // x^2 = square(x)
            let pow_result = eval_primitive(Primitive::Pow, &[Value::scalar_f64(x), Value::scalar_f64(2.0)], &BTreeMap::new()).unwrap();
            let square_result = eval_primitive(Primitive::Square, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let pow_val = pow_result.as_f64_scalar().unwrap();
            let square_val = square_result.as_f64_scalar().unwrap();
            prop_assert!((pow_val - square_val).abs() < 1e-12, "x^2 != square(x): {} vs {}", pow_val, square_val);
        }

        #[test]
        fn metamorphic_expm1_equals_exp_minus_one(x in -2.0f64..2.0) {
            // expm1(x) = exp(x) - 1
            let expm1_result = eval_primitive(Primitive::Expm1, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let exp_result = eval_primitive(Primitive::Exp, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let expm1_val = expm1_result.as_f64_scalar().unwrap();
            let exp_val = exp_result.as_f64_scalar().unwrap();
            let diff = (expm1_val - (exp_val - 1.0)).abs();
            prop_assert!(diff < 1e-14, "expm1(x) != exp(x)-1: {} vs {}", expm1_val, exp_val - 1.0);
        }

        #[test]
        fn metamorphic_log1p_equals_log_one_plus(x in 0.01f64..10.0) {
            // log1p(x) = log(1 + x) for x > 0
            let log1p_result = eval_primitive(Primitive::Log1p, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let one_plus_x = 1.0 + x;
            let log_result = eval_primitive(Primitive::Log, &[Value::scalar_f64(one_plus_x)], &BTreeMap::new()).unwrap();
            let log1p_val = log1p_result.as_f64_scalar().unwrap();
            let log_val = log_result.as_f64_scalar().unwrap();
            let diff = (log1p_val - log_val).abs();
            prop_assert!(diff < 1e-14, "log1p(x) != log(1+x): {} vs {}", log1p_val, log_val);
        }

        #[test]
        fn metamorphic_erfc_equals_one_minus_erf(x in -3.0f64..3.0) {
            // erfc(x) = 1 - erf(x)
            let erfc_result = eval_primitive(Primitive::Erfc, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let erf_result = eval_primitive(Primitive::Erf, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let erfc_val = erfc_result.as_f64_scalar().unwrap();
            let erf_val = erf_result.as_f64_scalar().unwrap();
            let diff = (erfc_val - (1.0 - erf_val)).abs();
            prop_assert!(diff < 1e-14, "erfc(x) != 1-erf(x): {} vs {}", erfc_val, 1.0 - erf_val);
        }

        #[test]
        fn metamorphic_round_close_to_input(x in -100.0f64..100.0) {
            // |round(x) - x| <= 0.5
            let round_result = eval_primitive(Primitive::Round, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let round_val = round_result.as_f64_scalar().unwrap();
            let diff = (round_val - x).abs();
            prop_assert!(diff <= 0.5 + 1e-10, "|round(x) - x| > 0.5: |{} - {}| = {}", round_val, x, diff);
        }

        #[test]
        fn metamorphic_round_is_integer(x in -100.0f64..100.0) {
            // round(x) is an integer
            let round_result = eval_primitive(Primitive::Round, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let round_val = round_result.as_f64_scalar().unwrap();
            let frac = round_val - round_val.trunc();
            prop_assert!(frac.abs() < 1e-10, "round(x) not an integer: round({}) = {}", x, round_val);
        }

        #[test]
        fn metamorphic_div_mul_identity(a in 1.0f64..100.0, b in 1.0f64..100.0) {
            // (a / b) * b ≈ a
            let div_result = eval_primitive(Primitive::Div, &[Value::scalar_f64(a), Value::scalar_f64(b)], &BTreeMap::new()).unwrap();
            let div_val = div_result.as_f64_scalar().unwrap();
            let mul_result = eval_primitive(Primitive::Mul, &[Value::scalar_f64(div_val), Value::scalar_f64(b)], &BTreeMap::new()).unwrap();
            let mul_val = mul_result.as_f64_scalar().unwrap();
            let diff = (mul_val - a).abs();
            prop_assert!(diff < 1e-12, "(a/b)*b != a: ({}/{})*{} = {} != {}", a, b, b, mul_val, a);
        }

        #[test]
        fn metamorphic_lgamma_one_equals_zero(_dummy in 0..1i32) {
            // lgamma(1) = 0
            let result = eval_primitive(Primitive::Lgamma, &[Value::scalar_f64(1.0)], &BTreeMap::new()).unwrap();
            let val = result.as_f64_scalar().unwrap();
            prop_assert!(val.abs() < 1e-14, "lgamma(1) != 0: {}", val);
        }

        #[test]
        fn metamorphic_lgamma_two_equals_zero(_dummy in 0..1i32) {
            // lgamma(2) = 0 (since gamma(2) = 1! = 1)
            let result = eval_primitive(Primitive::Lgamma, &[Value::scalar_f64(2.0)], &BTreeMap::new()).unwrap();
            let val = result.as_f64_scalar().unwrap();
            prop_assert!(val.abs() < 1e-14, "lgamma(2) != 0: {}", val);
        }

        #[test]
        fn metamorphic_bitwise_and_idempotent(a in 0i64..1000) {
            // a & a = a
            let result = eval_primitive(Primitive::BitwiseAnd, &[Value::scalar_i64(a), Value::scalar_i64(a)], &BTreeMap::new()).unwrap();
            let val = result.as_i64_scalar().unwrap();
            prop_assert_eq!(val, a, "a & a != a: {} & {} = {}", a, a, val);
        }

        #[test]
        fn metamorphic_bitwise_or_idempotent(a in 0i64..1000) {
            // a | a = a
            let result = eval_primitive(Primitive::BitwiseOr, &[Value::scalar_i64(a), Value::scalar_i64(a)], &BTreeMap::new()).unwrap();
            let val = result.as_i64_scalar().unwrap();
            prop_assert_eq!(val, a, "a | a != a: {} | {} = {}", a, a, val);
        }

        #[test]
        fn metamorphic_bitwise_xor_self_is_zero(a in 0i64..1000) {
            // a ^ a = 0
            let result = eval_primitive(Primitive::BitwiseXor, &[Value::scalar_i64(a), Value::scalar_i64(a)], &BTreeMap::new()).unwrap();
            let val = result.as_i64_scalar().unwrap();
            prop_assert_eq!(val, 0, "a ^ a != 0: {} ^ {} = {}", a, a, val);
        }

        #[test]
        fn metamorphic_bitwise_not_involution(a in 0i64..1000) {
            // !!a = a (double negation)
            let not_a = eval_primitive(Primitive::BitwiseNot, &[Value::scalar_i64(a)], &BTreeMap::new()).unwrap();
            let not_not_a = eval_primitive(Primitive::BitwiseNot, &[not_a], &BTreeMap::new()).unwrap();
            let val = not_not_a.as_i64_scalar().unwrap();
            prop_assert_eq!(val, a, "!!a != a: !!{} = {}", a, val);
        }

        #[test]
        fn metamorphic_atan2_vs_atan(y in 0.1f64..10.0, x in 0.1f64..10.0) {
            // atan2(y, x) = atan(y/x) for x > 0, y > 0
            let atan2_result = eval_primitive(Primitive::Atan2, &[Value::scalar_f64(y), Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let ratio = y / x;
            let atan_result = eval_primitive(Primitive::Atan, &[Value::scalar_f64(ratio)], &BTreeMap::new()).unwrap();
            let atan2_val = atan2_result.as_f64_scalar().unwrap();
            let atan_val = atan_result.as_f64_scalar().unwrap();
            let diff = (atan2_val - atan_val).abs();
            prop_assert!(diff < 1e-14, "atan2(y,x) != atan(y/x): {} vs {}", atan2_val, atan_val);
        }

        #[test]
        fn metamorphic_is_finite_on_regular_floats(x in -1e100f64..1e100) {
            // is_finite should return true for regular (non-infinite, non-NaN) floats
            let result = eval_primitive(Primitive::IsFinite, &[Value::scalar_f64(x)], &BTreeMap::new()).unwrap();
            let val = result.as_bool_scalar().unwrap();
            prop_assert!(val, "is_finite({}) returned false", x);
        }

    }
}
