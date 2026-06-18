//! Oracle conformance tests for jax.nn softmax/log_softmax/logsumexp.
//!
//! These tests verify that our implementations match JAX's behavior.
//! Reference values were computed with JAX 0.4.x using:
//!
//! ```python
//! import jax.numpy as jnp
//! import jax.nn as nn
//! from jax.scipy.special import logsumexp
//! ```

use fj_lax::nn::{log_softmax, logsumexp, softmax, standardize};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    (a - b).abs() <= tol
}

fn vec_approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y, tol))
}

// JAX reference values for logsumexp([1.0, 2.0, 3.0])
// >>> logsumexp(jnp.array([1.0, 2.0, 3.0]))
// Array(3.4076059, dtype=float32)
#[test]
fn test_logsumexp_jax_reference_basic() {
    let x = [1.0, 2.0, 3.0];
    let result = logsumexp(&x);
    // JAX gives 3.4076059 (float32), we use f64 so slightly more precise
    let expected = 3.4076059644443806;
    assert!(
        approx_eq(result, expected, 1e-6),
        "logsumexp mismatch: got {result}, expected {expected}"
    );
}

// JAX reference: logsumexp([0.0, 0.0]) = ln(2) ≈ 0.693147
#[test]
fn test_logsumexp_zeros() {
    let x = [0.0, 0.0];
    let result = logsumexp(&x);
    let expected = 2.0_f64.ln();
    assert!(
        approx_eq(result, expected, 1e-10),
        "logsumexp zeros mismatch: got {result}, expected {expected}"
    );
}

// A +inf term dominates: logsumexp([inf, x]) = log(exp(inf)+exp(x)) = inf.
#[test]
fn test_logsumexp_positive_infinity_is_infinity() {
    let result = logsumexp(&[f64::INFINITY, 1.0, 2.0]);
    assert!(
        result.is_infinite() && result.is_sign_positive(),
        "logsumexp with a +inf term must be +inf, got {result}"
    );
}

// A -inf term contributes exp(-inf) = 0, so it is masked out: logsumexp([-inf, 5])
// = 5. The logsumexp analog of softmax's -inf masking (masked-attention pattern).
#[test]
fn test_logsumexp_masks_negative_infinity() {
    let result = logsumexp(&[f64::NEG_INFINITY, 5.0, f64::NEG_INFINITY]);
    assert!(
        approx_eq(result, 5.0, 1e-12),
        "logsumexp must mask -inf terms (exp(-inf)=0); got {result}, expected 5.0"
    );
}

// JAX reference: logsumexp([1000.0, 1000.0]) = 1000 + ln(2)
// Tests numerical stability with large values
#[test]
fn test_logsumexp_large_values_stable() {
    let x = [1000.0, 1000.0];
    let result = logsumexp(&x);
    let expected = 1000.0 + 2.0_f64.ln();
    assert!(
        approx_eq(result, expected, 1e-10),
        "logsumexp large values mismatch: got {result}, expected {expected}"
    );
}

// JAX reference: softmax([1.0, 2.0, 3.0])
// Array([0.09003057, 0.24472848, 0.66524094], dtype=float32)
#[test]
fn test_softmax_jax_reference() {
    let x = [1.0, 2.0, 3.0];
    let result = softmax(&x);
    let expected = [0.09003057317038046, 0.24472847105479767, 0.6652409557748218];
    assert!(
        vec_approx_eq(&result, &expected, 1e-6),
        "softmax mismatch: got {:?}, expected {:?}",
        result,
        expected
    );
}

// Verify softmax sums to 1
#[test]
fn test_softmax_sums_to_one() {
    let inputs = [
        vec![1.0, 2.0, 3.0],
        vec![-1.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![100.0, 100.0, 100.0],
    ];
    for x in inputs {
        let result = softmax(&x);
        let sum: f64 = result.iter().sum();
        assert!(
            approx_eq(sum, 1.0, 1e-10),
            "softmax sum should be 1, got {sum} for input {:?}",
            x
        );
    }
}

// Numerical stability: equal large logits must give a uniform distribution, not
// NaN from exp overflow. softmax([1000,1000]) = [0.5, 0.5] (max-subtraction). The
// sums-to-one test exercises large values but not the exact stable values.
#[test]
fn test_softmax_large_equal_values_stable() {
    let result = softmax(&[1000.0, 1000.0]);
    assert!(
        vec_approx_eq(&result, &[0.5, 0.5], 1e-12),
        "softmax of equal large logits must be uniform (stable), got {:?}",
        result
    );
}

// Masked-attention pattern: a -inf logit contributes exp(-inf) = 0, so it is fully
// masked out and the remaining finite logits share the mass. JAX/numpy parity.
#[test]
fn test_softmax_masks_negative_infinity() {
    let result = softmax(&[0.0, f64::NEG_INFINITY, 0.0]);
    assert!(
        vec_approx_eq(&result, &[0.5, 0.0, 0.5], 1e-12),
        "softmax must mask -inf logits to 0 and split mass over finite ones, got {:?}",
        result
    );
}

// JAX reference: log_softmax([1.0, 2.0, 3.0])
// Array([-2.4076061, -1.4076061, -0.4076061], dtype=float32)
#[test]
fn test_log_softmax_jax_reference() {
    let x = [1.0, 2.0, 3.0];
    let result = log_softmax(&x);
    let expected = [
        -2.4076059644443806,
        -1.4076059644443806,
        -0.4076059644443806,
    ];
    assert!(
        vec_approx_eq(&result, &expected, 1e-6),
        "log_softmax mismatch: got {:?}, expected {:?}",
        result,
        expected
    );
}

// Verify log_softmax = log(softmax(x))
#[test]
fn test_log_softmax_equals_log_softmax() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let log_sm = log_softmax(&x);
    let sm = softmax(&x);
    let log_of_sm: Vec<f64> = sm.iter().map(|&v| v.ln()).collect();
    assert!(
        vec_approx_eq(&log_sm, &log_of_sm, 1e-10),
        "log_softmax should equal log(softmax): got {:?} vs {:?}",
        log_sm,
        log_of_sm
    );
}

// Test numerical stability of log_softmax with large values
#[test]
fn test_log_softmax_large_values_stable() {
    let x = [1000.0, 1001.0, 1002.0];
    let result = log_softmax(&x);
    // Should be finite and sum of exp should be 1
    assert!(
        result.iter().all(|&v| v.is_finite()),
        "log_softmax should be finite for large inputs: got {:?}",
        result
    );
    let sum_exp: f64 = result.iter().map(|&v| v.exp()).sum();
    assert!(
        approx_eq(sum_exp, 1.0, 1e-10),
        "exp(log_softmax) should sum to 1: got {sum_exp}"
    );
}

// Test standardize (jax.nn.standardize)
// standardize([1.0, 2.0, 3.0, 4.0, 5.0]) should have mean ~0, var ~1
#[test]
fn test_standardize_zero_mean_unit_variance() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let result = standardize(&x, 1e-10);
    let n = result.len() as f64;

    // Check mean ≈ 0
    let mean: f64 = result.iter().sum::<f64>() / n;
    assert!(
        approx_eq(mean, 0.0, 1e-10),
        "standardize mean should be 0: got {mean}"
    );

    // Check variance ≈ 1
    let var: f64 = result.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    assert!(
        approx_eq(var, 1.0, 1e-10),
        "standardize variance should be 1: got {var}"
    );
}

// Test that softmax preserves ordering
#[test]
fn test_softmax_preserves_order() {
    let x = [1.0, 2.0, 3.0, 0.5, 2.5];
    let result = softmax(&x);
    // Create pairs of (original index, softmax value) and verify order
    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            if x[i] < x[j] {
                assert!(
                    result[i] < result[j],
                    "softmax should preserve order: x[{i}]={} < x[{j}]={}, but sm[{i}]={} >= sm[{j}]={}",
                    x[i],
                    x[j],
                    result[i],
                    result[j]
                );
            }
        }
    }
}

// Edge case: single element
#[test]
fn test_softmax_single_element() {
    let x = [5.0];
    let result = softmax(&x);
    assert_eq!(result.len(), 1);
    assert!(
        approx_eq(result[0], 1.0, 1e-10),
        "softmax of single element should be 1.0"
    );
}

// Edge case: empty input
#[test]
fn test_softmax_empty() {
    let x: [f64; 0] = [];
    let result = softmax(&x);
    assert!(result.is_empty());
}

// Test uniform input
#[test]
fn test_softmax_uniform() {
    let x = [1.0, 1.0, 1.0, 1.0];
    let result = softmax(&x);
    let expected = 0.25;
    for &v in &result {
        assert!(
            approx_eq(v, expected, 1e-10),
            "softmax of uniform input should be uniform: got {v}, expected {expected}"
        );
    }
}
