//! Dense contiguous-`f64` elementwise kernels — the safe-Rust portable-SIMD
//! foundation for the dense tensor-storage migration (frankenjax-mcqr.30).
//!
//! `TensorValue` currently stores `Vec<Literal>` (a 24-byte enum, sized by the
//! `Complex128Bits(u64, u64)` variant), so an F64 tensor carries ~3x the bytes
//! of native `f64` and the per-element `Literal::F64Bits` match blocks
//! autovectorization. The `eval/add_64k_f64_vec` vs `eval/add_64k_f64_dense_ref`
//! benches measure this at ~51x slower than a contiguous `f64` add.
//!
//! These kernels operate on `&[f64]`, which the optimizer autovectorizes, and
//! apply the same per-lane IEEE-754 operations the `Vec<Literal>` path uses
//! (`Primitive::Add` => `a + b`, etc.). They are therefore bit-for-bit identical
//! drop-in replacements once a dense storage representation is wired in — see
//! the parity tests below, which check each kernel against `eval_primitive`.
//!
//! Routing tensor ops through these (and storing F64/F32 tensors densely) is the
//! remaining work in mcqr.30; until then these have no production caller.
#![allow(dead_code)]

/// Elementwise `a[i] + b[i]`. Bit-identical to `Primitive::Add` over F64.
#[inline]
#[must_use]
pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "dense::add length mismatch");
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

/// Elementwise `a[i] - b[i]`. Bit-identical to `Primitive::Sub` over F64.
#[inline]
#[must_use]
pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "dense::sub length mismatch");
    a.iter().zip(b).map(|(&x, &y)| x - y).collect()
}

/// Elementwise `a[i] * b[i]`. Bit-identical to `Primitive::Mul` over F64.
#[inline]
#[must_use]
pub fn mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "dense::mul length mismatch");
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

/// Elementwise `a[i] / b[i]`. Bit-identical to `Primitive::Div` over F64.
#[inline]
#[must_use]
pub fn div(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "dense::div length mismatch");
    a.iter().zip(b).map(|(&x, &y)| x / y).collect()
}

/// Elementwise `-a[i]`. Bit-identical to `Primitive::Neg` over F64.
#[inline]
#[must_use]
pub fn neg(a: &[f64]) -> Vec<f64> {
    a.iter().map(|&x| -x).collect()
}

/// Elementwise scalar-broadcast `op(a[i], scalar)` for the four arithmetic ops,
/// matching the `Tensor ⊗ Scalar` path. `scalar_on_left` swaps operand order
/// for the non-commutative `-`/`/`.
#[inline]
#[must_use]
pub fn scalar_op(a: &[f64], scalar: f64, op: ArithOp, scalar_on_left: bool) -> Vec<f64> {
    a.iter()
        .map(|&x| {
            let (l, r) = if scalar_on_left {
                (scalar, x)
            } else {
                (x, scalar)
            };
            op.apply(l, r)
        })
        .collect()
}

/// The four IEEE-754 arithmetic ops, used to drive `scalar_op` without a closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ArithOp {
    #[inline]
    #[must_use]
    pub fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            ArithOp::Add => a + b,
            ArithOp::Sub => a - b,
            ArithOp::Mul => a * b,
            ArithOp::Div => a / b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_primitive;
    use fj_core::{Literal, Primitive, Value};
    use std::collections::BTreeMap;

    fn eval_bits(p: Primitive, a: &[f64], b: &[f64]) -> Vec<u64> {
        let lhs = Value::vector_f64(a).unwrap();
        let rhs = Value::vector_f64(b).unwrap();
        let out = eval_primitive(p, &[lhs, rhs], &BTreeMap::new()).unwrap();
        let Value::Tensor(t) = out else {
            panic!("expected tensor");
        };
        t.elements
            .iter()
            .map(|e| match e {
                Literal::F64Bits(bits) => *bits,
                other => panic!("expected F64Bits, got {other:?}"),
            })
            .collect()
    }

    fn dense_bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn assert_length_mismatch_panic<F>(f: F, expected: &str)
    where
        F: FnOnce() + std::panic::UnwindSafe,
    {
        let err = std::panic::catch_unwind(f).expect_err("expected length mismatch panic");
        let msg = if let Some(s) = err.downcast_ref::<&'static str>() {
            (*s).to_string()
        } else if let Some(s) = err.downcast_ref::<String>() {
            s.clone()
        } else {
            String::new()
        };
        assert!(
            msg.contains(expected),
            "panic message {msg:?} did not contain {expected:?}"
        );
    }

    #[test]
    fn dense_binary_kernels_reject_length_mismatch() {
        let a = [1.0, 2.0];
        let b = [3.0];
        assert_length_mismatch_panic(
            || {
                let _ = add(&a, &b);
            },
            "dense::add length mismatch",
        );
        assert_length_mismatch_panic(
            || {
                let _ = sub(&a, &b);
            },
            "dense::sub length mismatch",
        );
        assert_length_mismatch_panic(
            || {
                let _ = mul(&a, &b);
            },
            "dense::mul length mismatch",
        );
        assert_length_mismatch_panic(
            || {
                let _ = div(&a, &b);
            },
            "dense::div length mismatch",
        );
    }

    #[test]
    fn dense_arith_bit_identical_to_eval_primitive() {
        // Adversarial values incl. signed zero, inf, NaN, div-by-zero.
        // `black_box` so the compiler cannot const-fold the dense kernels (whose
        // inputs are literals) to the compiler's +NaN for `0.0/0.0` while the
        // dynamic `eval_primitive` path computes the CPU's -NaN at runtime — a
        // spurious NaN-sign mismatch under some nightlies (the e7ej unpinned-
        // nightly fragility). Both paths are bit-identical at runtime.
        let a = std::hint::black_box([1.5, -0.0, f64::INFINITY, f64::NAN, 7.0, -3.25, 0.0, 2.0]);
        let b = std::hint::black_box([2.0, 3.0, -4.0, 5.0, 0.0, f64::NEG_INFINITY, 0.0, -8.5]);

        assert_eq!(
            dense_bits(&add(&a, &b)),
            eval_bits(Primitive::Add, &a, &b),
            "add"
        );
        assert_eq!(
            dense_bits(&sub(&a, &b)),
            eval_bits(Primitive::Sub, &a, &b),
            "sub"
        );
        assert_eq!(
            dense_bits(&mul(&a, &b)),
            eval_bits(Primitive::Mul, &a, &b),
            "mul"
        );
        assert_eq!(
            dense_bits(&div(&a, &b)),
            eval_bits(Primitive::Div, &a, &b),
            "div"
        );
    }

    #[test]
    fn dense_scalar_op_matches_arith() {
        let a = [1.0, -2.0, 3.5, -0.0, f64::INFINITY];
        let s = 2.5;
        for op in [ArithOp::Add, ArithOp::Sub, ArithOp::Mul, ArithOp::Div] {
            let left = scalar_op(&a, s, op, true);
            let right = scalar_op(&a, s, op, false);
            for (i, &x) in a.iter().enumerate() {
                assert_eq!(left[i].to_bits(), op.apply(s, x).to_bits(), "{op:?} L {i}");
                assert_eq!(right[i].to_bits(), op.apply(x, s).to_bits(), "{op:?} R {i}");
            }
        }
    }

    #[test]
    fn dense_neg_matches() {
        let a = [1.0, -2.0, 0.0, -0.0, f64::INFINITY];
        let n = neg(&a);
        for (i, &x) in a.iter().enumerate() {
            assert_eq!(n[i].to_bits(), (-x).to_bits(), "neg {i}");
        }
    }
}
