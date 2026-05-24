//! Neural network activation functions matching JAX's jax.nn module.
//!
//! These are standalone functions that operate on f64 slices, matching JAX semantics.

use std::f64::consts::PI;

/// ReLU: max(x, 0)
///
/// Matches `jax.nn.relu(x)`.
#[must_use]
pub fn relu(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// Leaky ReLU: x if x >= 0 else negative_slope * x
///
/// Matches `jax.nn.leaky_relu(x, negative_slope)`.
#[must_use]
pub fn leaky_relu(x: &[f64], negative_slope: f64) -> Vec<f64> {
    x.iter()
        .map(|&v| if v >= 0.0 { v } else { negative_slope * v })
        .collect()
}

/// ReLU6: min(max(x, 0), 6)
///
/// Matches `jax.nn.relu6(x)`.
#[must_use]
pub fn relu6(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| v.max(0.0).min(6.0)).collect()
}

/// Sigmoid: 1 / (1 + exp(-x))
///
/// Matches `jax.nn.sigmoid(x)` and `jax.lax.logistic(x)`.
#[must_use]
pub fn sigmoid(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

/// Hard sigmoid: clip((x + 3) / 6, 0, 1)
///
/// Matches `jax.nn.hard_sigmoid(x)`.
#[must_use]
pub fn hard_sigmoid(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&v| ((v + 3.0) / 6.0).clamp(0.0, 1.0))
        .collect()
}

/// Hard tanh: clip(x, -1, 1)
///
/// Matches `jax.nn.hard_tanh(x)`.
#[must_use]
pub fn hard_tanh(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| v.clamp(-1.0, 1.0)).collect()
}

/// SiLU / Swish: x * sigmoid(x)
///
/// Matches `jax.nn.silu(x)` and `jax.nn.swish(x)`.
#[must_use]
pub fn silu(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&v| v / (1.0 + (-v).exp()))
        .collect()
}

/// Alias for silu
#[must_use]
pub fn swish(x: &[f64]) -> Vec<f64> {
    silu(x)
}

/// Hard SiLU / Hard Swish: x * hard_sigmoid(x)
///
/// Matches `jax.nn.hard_silu(x)` and `jax.nn.hard_swish(x)`.
#[must_use]
pub fn hard_silu(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&v| v * ((v + 3.0) / 6.0).clamp(0.0, 1.0))
        .collect()
}

/// Alias for hard_silu
#[must_use]
pub fn hard_swish(x: &[f64]) -> Vec<f64> {
    hard_silu(x)
}

/// GELU (Gaussian Error Linear Unit): x * Phi(x)
///
/// Uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// Matches `jax.nn.gelu(x, approximate=True)`.
#[must_use]
pub fn gelu(x: &[f64]) -> Vec<f64> {
    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    x.iter()
        .map(|&v| {
            let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect()
}

/// ELU (Exponential Linear Unit): x if x > 0 else alpha * (exp(x) - 1)
///
/// Matches `jax.nn.elu(x, alpha)`.
#[must_use]
pub fn elu(x: &[f64], alpha: f64) -> Vec<f64> {
    x.iter()
        .map(|&v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
        .collect()
}

/// CELU (Continuously-differentiable ELU): max(x, 0) + min(0, alpha * (exp(x/alpha) - 1))
///
/// Matches `jax.nn.celu(x, alpha)`.
#[must_use]
pub fn celu(x: &[f64], alpha: f64) -> Vec<f64> {
    x.iter()
        .map(|&v| v.max(0.0) + (alpha * ((v / alpha).exp() - 1.0)).min(0.0))
        .collect()
}

/// SELU (Scaled ELU): scale * (max(x, 0) + alpha * min(0, exp(x) - 1))
///
/// Uses the self-normalizing constants:
/// - alpha ≈ 1.6732632423543772
/// - scale ≈ 1.0507009873554805
///
/// Matches `jax.nn.selu(x)`.
#[must_use]
pub fn selu(x: &[f64]) -> Vec<f64> {
    const ALPHA: f64 = 1.6732632423543772;
    const SCALE: f64 = 1.0507009873554805;
    x.iter()
        .map(|&v| {
            SCALE * if v > 0.0 { v } else { ALPHA * (v.exp() - 1.0) }
        })
        .collect()
}

/// Softplus: log(1 + exp(x))
///
/// Uses numerically stable formulation to avoid overflow.
/// Matches `jax.nn.softplus(x)`.
#[must_use]
pub fn softplus(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&v| {
            if v > 20.0 {
                // For large x, softplus(x) ≈ x
                v
            } else if v < -20.0 {
                // For large negative x, softplus(x) ≈ exp(x)
                v.exp()
            } else {
                (1.0 + v.exp()).ln()
            }
        })
        .collect()
}

/// Softsign: x / (1 + |x|)
///
/// Matches `jax.nn.soft_sign(x)`.
#[must_use]
pub fn softsign(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| v / (1.0 + v.abs())).collect()
}

/// Mish: x * tanh(softplus(x))
///
/// Matches `jax.nn.mish(x)`.
#[must_use]
pub fn mish(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&v| {
            let sp = if v > 20.0 {
                v
            } else if v < -20.0 {
                v.exp()
            } else {
                (1.0 + v.exp()).ln()
            };
            v * sp.tanh()
        })
        .collect()
}

/// Log sigmoid: log(sigmoid(x)) = -softplus(-x)
///
/// Uses numerically stable formulation.
/// Matches `jax.nn.log_sigmoid(x)`.
#[must_use]
pub fn log_sigmoid(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&v| {
            if v > 20.0 {
                // For large x, log_sigmoid(x) ≈ 0
                0.0
            } else if v < -20.0 {
                // For large negative x, log_sigmoid(x) ≈ x
                v
            } else {
                -(1.0 + (-v).exp()).ln()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_relu_basic() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = relu(&x);
        assert_eq!(y, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu_basic() {
        let x = vec![-2.0, 0.0, 2.0];
        let y = leaky_relu(&x, 0.1);
        assert!(approx_eq(y[0], -0.2, 1e-10));
        assert!(approx_eq(y[1], 0.0, 1e-10));
        assert!(approx_eq(y[2], 2.0, 1e-10));
    }

    #[test]
    fn test_relu6_clips() {
        let x = vec![-1.0, 0.0, 3.0, 6.0, 10.0];
        let y = relu6(&x);
        assert_eq!(y, vec![0.0, 0.0, 3.0, 6.0, 6.0]);
    }

    #[test]
    fn test_sigmoid_bounds() {
        let x = vec![-100.0, 0.0, 100.0];
        let y = sigmoid(&x);
        assert!(y[0] < 0.001);
        assert!(approx_eq(y[1], 0.5, 1e-10));
        assert!(y[2] > 0.999);
    }

    #[test]
    fn test_hard_sigmoid_piecewise() {
        let x = vec![-10.0, -3.0, 0.0, 3.0, 10.0];
        let y = hard_sigmoid(&x);
        assert!(approx_eq(y[0], 0.0, 1e-10));
        assert!(approx_eq(y[1], 0.0, 1e-10));
        assert!(approx_eq(y[2], 0.5, 1e-10));
        assert!(approx_eq(y[3], 1.0, 1e-10));
        assert!(approx_eq(y[4], 1.0, 1e-10));
    }

    #[test]
    fn test_hard_tanh_clips() {
        let x = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let y = hard_tanh(&x);
        assert_eq!(y, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_silu_at_zero() {
        let x = vec![0.0];
        let y = silu(&x);
        assert!(approx_eq(y[0], 0.0, 1e-10));
    }

    #[test]
    fn test_swish_is_silu() {
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        assert_eq!(silu(&x), swish(&x));
    }

    #[test]
    fn test_gelu_symmetry() {
        let x = vec![-1.0, 1.0];
        let y = gelu(&x);
        // GELU is not symmetric, but gelu(-x) + gelu(x) ≈ x for small x
        // Just check it computes something reasonable
        assert!(y[0] < 0.0);
        assert!(y[1] > 0.0);
    }

    #[test]
    fn test_elu_at_zero() {
        let x = vec![0.0];
        let y = elu(&x, 1.0);
        assert!(approx_eq(y[0], 0.0, 1e-10));
    }

    #[test]
    fn test_elu_negative() {
        let x = vec![-1.0];
        let y = elu(&x, 1.0);
        // elu(-1) = exp(-1) - 1 ≈ -0.632
        assert!(approx_eq(y[0], (-1.0_f64).exp() - 1.0, 1e-10));
    }

    #[test]
    fn test_selu_self_normalizing_constants() {
        // SELU should preserve mean 0 and variance 1 for standard normal inputs
        // Just check the constants are applied
        let x = vec![1.0];
        let y = selu(&x);
        assert!(approx_eq(y[0], 1.0507009873554805, 1e-10));
    }

    #[test]
    fn test_softplus_positive() {
        let x = vec![-2.0, 0.0, 2.0];
        let y = softplus(&x);
        assert!(y.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_softplus_large_input() {
        let x = vec![100.0];
        let y = softplus(&x);
        assert!(approx_eq(y[0], 100.0, 1e-10));
    }

    #[test]
    fn test_softsign_bounds() {
        let x = vec![-100.0, 0.0, 100.0];
        let y = softsign(&x);
        assert!(y[0] > -1.0 && y[0] < -0.99);
        assert!(approx_eq(y[1], 0.0, 1e-10));
        assert!(y[2] < 1.0 && y[2] > 0.99);
    }

    #[test]
    fn test_mish_at_zero() {
        let x = vec![0.0];
        let y = mish(&x);
        assert!(approx_eq(y[0], 0.0, 1e-10));
    }

    #[test]
    fn test_log_sigmoid_bounds() {
        let x = vec![-100.0, 0.0, 100.0];
        let y = log_sigmoid(&x);
        assert!(approx_eq(y[0], -100.0, 1.0)); // approximately -100
        assert!(approx_eq(y[1], -0.693147, 1e-5)); // ln(0.5)
        assert!(approx_eq(y[2], 0.0, 1e-10));
    }

    #[test]
    fn test_celu_continuous() {
        let x = vec![-0.01, 0.0, 0.01];
        let y = celu(&x, 1.0);
        // Should be continuous at 0
        assert!(approx_eq(y[1], 0.0, 1e-10));
        assert!((y[2] - y[0]).abs() < 0.03);
    }

    #[test]
    fn test_hard_silu_at_boundaries() {
        let x = vec![-3.0, 0.0, 3.0];
        let y = hard_silu(&x);
        assert!(approx_eq(y[0], 0.0, 1e-10)); // -3 * 0 = 0
        assert!(approx_eq(y[1], 0.0, 1e-10)); // 0 * 0.5 = 0
        assert!(approx_eq(y[2], 3.0, 1e-10)); // 3 * 1 = 3
    }

    #[test]
    fn test_hard_swish_is_hard_silu() {
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        assert_eq!(hard_silu(&x), hard_swish(&x));
    }
}
