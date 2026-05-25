//! Oracle conformance tests for jax.random distribution functions.
//!
//! Tests statistical properties of random distributions rather than exact values
//! since PRNG outputs depend on implementation details.
//!
//! Reference: JAX random module (jax.random.*)
//! ```python
//! import jax.random as random
//! key = random.key(42)
//! random.weibull(key, scale=1.0, concentration=2.0, shape=(1000,))
//! random.chi2(key, df=5.0, shape=(1000,))
//! ```

use fj_lax::threefry::{
    random_beta, random_cauchy, random_chi2, random_dirichlet, random_exponential, random_gamma,
    random_geometric, random_key, random_laplace, random_normal, random_pareto, random_poisson,
    random_rayleigh, random_t, random_truncated_normal, random_uniform, random_weibull,
};

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    let m = mean(values);
    values.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / values.len() as f64
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

const SAMPLE_SIZE: usize = 10000;
const STAT_TOL: f64 = 0.1;

#[test]
fn test_uniform_mean_variance() {
    let key = random_key(42);
    let samples = random_uniform(key, SAMPLE_SIZE, 0.0, 1.0);

    let m = mean(&samples);
    let v = variance(&samples);

    assert!(
        approx_eq(m, 0.5, STAT_TOL),
        "uniform mean should be ~0.5, got {}",
        m
    );
    assert!(
        approx_eq(v, 1.0 / 12.0, STAT_TOL),
        "uniform variance should be ~1/12, got {}",
        v
    );
}

#[test]
fn test_normal_mean_variance() {
    let key = random_key(42);
    let samples = random_normal(key, SAMPLE_SIZE);

    let m = mean(&samples);
    let v = variance(&samples);

    assert!(
        approx_eq(m, 0.0, STAT_TOL),
        "normal mean should be ~0, got {}",
        m
    );
    assert!(
        approx_eq(v, 1.0, STAT_TOL),
        "normal variance should be ~1, got {}",
        v
    );
}

#[test]
fn test_exponential_mean() {
    let key = random_key(42);
    let rate = 2.0;
    let samples = random_exponential(key, SAMPLE_SIZE, rate);

    let m = mean(&samples);
    let expected_mean = 1.0 / rate;

    assert!(
        approx_eq(m, expected_mean, STAT_TOL),
        "exponential(rate={}) mean should be ~{}, got {}",
        rate,
        expected_mean,
        m
    );
    assert!(
        samples.iter().all(|&x| x >= 0.0),
        "exponential samples should be non-negative"
    );
}

#[test]
fn test_gamma_mean() {
    let key = random_key(42);
    let shape = 3.0;
    let samples = random_gamma(key, SAMPLE_SIZE, shape);

    let m = mean(&samples);

    assert!(
        approx_eq(m, shape, shape * 0.1),
        "gamma(shape={}) mean should be ~{}, got {}",
        shape,
        shape,
        m
    );
    assert!(
        samples.iter().all(|&x| x >= 0.0),
        "gamma samples should be non-negative"
    );
}

#[test]
fn test_beta_range() {
    let key = random_key(42);
    let samples = random_beta(key, SAMPLE_SIZE, 2.0, 5.0);

    assert!(
        samples.iter().all(|&x| x >= 0.0 && x <= 1.0),
        "beta samples should be in [0, 1]"
    );
}

#[test]
fn test_poisson_mean() {
    let key = random_key(42);
    let lam = 5.0;
    let samples: Vec<f64> = random_poisson(key, SAMPLE_SIZE, lam)
        .iter()
        .map(|&x| x as f64)
        .collect();

    let m = mean(&samples);

    assert!(
        approx_eq(m, lam, 0.5),
        "poisson(lam={}) mean should be ~{}, got {}",
        lam,
        lam,
        m
    );
}

#[test]
fn test_weibull_mean() {
    let key = random_key(42);
    let scale = 1.0;
    let concentration = 2.0;
    let samples = random_weibull(key, SAMPLE_SIZE, scale, concentration);

    let m = mean(&samples);
    // Weibull mean = scale * Gamma(1 + 1/k) where k is concentration
    // For k=2, Gamma(1.5) ≈ 0.886, so mean ≈ 0.886
    // Use simple approximation: Gamma(1.5) ≈ 0.886
    let expected_mean = scale * 0.886;

    assert!(
        approx_eq(m, expected_mean, expected_mean * 0.15),
        "weibull mean should be ~{}, got {}",
        expected_mean,
        m
    );
    assert!(
        samples.iter().all(|&x| x >= 0.0),
        "weibull samples should be non-negative"
    );
}

#[test]
fn test_rayleigh_mean() {
    let key = random_key(42);
    let scale = 2.0;
    let samples = random_rayleigh(key, SAMPLE_SIZE, scale);

    let m = mean(&samples);
    // Rayleigh mean = scale * sqrt(pi/2)
    let expected_mean = scale * (std::f64::consts::PI / 2.0).sqrt();

    assert!(
        approx_eq(m, expected_mean, expected_mean * 0.1),
        "rayleigh mean should be ~{}, got {}",
        expected_mean,
        m
    );
    assert!(
        samples.iter().all(|&x| x >= 0.0),
        "rayleigh samples should be non-negative"
    );
}

#[test]
fn test_chi2_mean() {
    let key = random_key(42);
    let df = 5.0;
    let samples = random_chi2(key, SAMPLE_SIZE, df);

    let m = mean(&samples);
    // Chi-square mean = df
    assert!(
        approx_eq(m, df, df * 0.1),
        "chi2(df={}) mean should be ~{}, got {}",
        df,
        df,
        m
    );
    assert!(
        samples.iter().all(|&x| x >= 0.0),
        "chi2 samples should be non-negative"
    );
}

#[test]
fn test_t_distribution_mean() {
    let key = random_key(42);
    let df = 10.0;
    let samples = random_t(key, SAMPLE_SIZE, df);

    let m = mean(&samples);
    // Student's t mean = 0 for df > 1
    assert!(
        approx_eq(m, 0.0, STAT_TOL),
        "t(df={}) mean should be ~0, got {}",
        df,
        m
    );
}

#[test]
fn test_laplace_mean() {
    let key = random_key(42);
    let loc = 2.0;
    let scale = 1.0;
    let samples = random_laplace(key, SAMPLE_SIZE, loc, scale);

    let m = mean(&samples);

    assert!(
        approx_eq(m, loc, STAT_TOL),
        "laplace(loc={}) mean should be ~{}, got {}",
        loc,
        loc,
        m
    );
}

#[test]
fn test_cauchy_median() {
    let key = random_key(42);
    let samples = random_cauchy(key, SAMPLE_SIZE);

    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[SAMPLE_SIZE / 2];

    assert!(
        approx_eq(median, 0.0, 1.0),
        "cauchy median should be ~0, got {}",
        median
    );
}

#[test]
fn test_pareto_positive() {
    let key = random_key(42);
    let b = 2.0;
    let samples = random_pareto(key, SAMPLE_SIZE, b);

    assert!(
        samples.iter().all(|&x| x >= 0.0),
        "pareto samples should be non-negative"
    );
}

#[test]
fn test_truncated_normal_bounds() {
    let key = random_key(42);
    let lower = -1.0;
    let upper = 1.0;
    let samples = random_truncated_normal(key, SAMPLE_SIZE, lower, upper);

    assert!(
        samples.iter().all(|&x| x >= lower && x <= upper),
        "truncated normal samples should be in [{}, {}]",
        lower,
        upper
    );
}

#[test]
fn test_dirichlet_simplex() {
    let key = random_key(42);
    let alpha = [1.0, 1.0, 1.0];
    let samples = random_dirichlet(key, 100, &alpha);

    let k = alpha.len();
    for i in 0..100 {
        let row: Vec<f64> = (0..k).map(|j| samples[i * k + j]).collect();
        let sum: f64 = row.iter().sum();
        assert!(
            approx_eq(sum, 1.0, 1e-8),
            "dirichlet sample should sum to 1, got {}",
            sum
        );
        assert!(
            row.iter().all(|&x| x >= 0.0 && x <= 1.0),
            "dirichlet components should be in [0, 1]"
        );
    }
}

#[test]
fn test_geometric_positive_integers() {
    let key = random_key(42);
    let p = 0.3;
    let samples: Vec<f64> = random_geometric(key, SAMPLE_SIZE, p)
        .iter()
        .map(|&x| x as f64)
        .collect();

    let m = mean(&samples);
    // JAX uses the number of trials convention: mean = 1/p
    let expected_mean = 1.0 / p;

    assert!(
        approx_eq(m, expected_mean, expected_mean * 0.2),
        "geometric(p={}) mean should be ~{}, got {}",
        p,
        expected_mean,
        m
    );
}

#[test]
fn test_gamma_shape_param() {
    let key = random_key(42);
    let shapes = [0.5, 1.0, 2.0, 5.0];

    for &shape in &shapes {
        let samples = random_gamma(key, 1000, shape);
        let m = mean(&samples);
        assert!(
            approx_eq(m, shape, shape * 0.2),
            "gamma(shape={}) mean should be ~{}, got {}",
            shape,
            shape,
            m
        );
    }
}

#[test]
fn test_beta_symmetric() {
    let key = random_key(42);
    let samples = random_beta(key, SAMPLE_SIZE, 2.0, 2.0);

    let m = mean(&samples);

    assert!(
        approx_eq(m, 0.5, STAT_TOL),
        "beta(2,2) mean should be ~0.5, got {}",
        m
    );
}
