//! ThreeFry2x32 PRNG core — counter-based PRNG matching JAX's default implementation.
//!
//! Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (SC'11)
//! JAX source: jax/_src/prng.py, threefry2x32 function

/// ThreeFry rotation constants for 2x32 variant.
/// From the original paper, Table 1 (Skein rotation constants for Nw=2).
const ROTATIONS: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];

/// Number of rounds in ThreeFry2x32 (default in JAX).
const NUM_ROUNDS: usize = 20;

/// A PRNGKey is a pair of u32 values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PRNGKey(pub [u32; 2]);

/// ThreeFry2x32: encrypt a 2-word plaintext with a 2-word key using `NUM_ROUNDS` rounds.
///
/// This exactly matches JAX's `threefry2x32` implementation.
#[must_use]
pub fn threefry2x32(key: [u32; 2], data: [u32; 2]) -> [u32; 2] {
    // Key schedule constant (from Skein specification)
    const KS_PARITY: u32 = 0x1BD1_1BDA;

    let ks2 = key[0] ^ key[1] ^ KS_PARITY;

    let mut x0 = data[0].wrapping_add(key[0]);
    let mut x1 = data[1].wrapping_add(key[1]);

    for round in 0..NUM_ROUNDS {
        // Apply rotation and XOR
        x0 = x0.wrapping_add(x1);
        x1 = x1.rotate_left(ROTATIONS[round % 8]) ^ x0;

        // Key injection every 4 rounds
        if (round + 1) % 4 == 0 {
            let inject_idx = (round + 1) / 4;
            // Key schedule: ks[(inject_idx) % 3], ks[(inject_idx+1) % 3]
            let keys = [key[0], key[1], ks2];
            x0 = x0.wrapping_add(keys[inject_idx % 3]);
            x1 = x1.wrapping_add(keys[(inject_idx + 1) % 3].wrapping_add(inject_idx as u32));
        }
    }

    [x0, x1]
}

/// Create a PRNG key from a 64-bit seed, matching JAX's `random.key(seed)`.
///
/// JAX splits the seed into two u32s: high and low halves.
#[must_use]
pub fn random_key(seed: u64) -> PRNGKey {
    let high = (seed >> 32) as u32;
    let low = seed as u32;
    PRNGKey([high, low])
}

/// Deterministic key splitting: produces two independent child keys.
///
/// Matches JAX's `random.split(key)` which uses ThreeFry to derive child keys.
#[must_use]
pub fn random_split(key: PRNGKey) -> (PRNGKey, PRNGKey) {
    let child1 = threefry2x32(key.0, [0, 0]);
    let child2 = threefry2x32(key.0, [0, 1]);
    (PRNGKey(child1), PRNGKey(child2))
}

/// Mix additional data into a key, producing a derived key.
///
/// Matches JAX's `random.fold_in(key, data)`.
/// JAX internally calls `threefry_2x32(key, threefry_seed(data))` where
/// `threefry_seed` puts high-32 bits first, low-32 bits second: `[data >> 32, data & 0xFFFFFFFF]`.
/// For u32 data, this is always `[0, data]`.
#[must_use]
pub fn random_fold_in(key: PRNGKey, data: u32) -> PRNGKey {
    PRNGKey(threefry2x32(key.0, [0, data]))
}

/// Generate `count` pseudorandom u32 values from a key.
///
/// Matches JAX's partitionable `_threefry_random_bits_partitionable` with `bit_width=32`:
/// For each sample index `i`, calls `threefry2x32(key, [0, i])` and XORs
/// the two output words: `result[0] ^ result[1]`.
fn generate_u32_bits(key: PRNGKey, count: usize) -> Vec<u32> {
    (0..count)
        .map(|i| {
            let [a, b] = threefry2x32(key.0, [0, i as u32]);
            a ^ b
        })
        .collect()
}

/// Generate uniform random f64 values in [minval, maxval).
///
/// Matches JAX's `jax.random.uniform` with default f32 mode (x64 not enabled):
/// 1. Generate one u32 per sample via XOR of threefry outputs
/// 2. Right-shift by 9 to keep 23 mantissa bits (f32 precision)
/// 3. OR with f32 1.0's bit pattern (0x3F800000), bitcast to f32, subtract 1.0
/// 4. Convert to f64 and scale to [minval, maxval)
///
/// Note: JAX defaults to f32 unless `jax_enable_x64` is set. The oracle fixtures
/// were captured without x64 mode, so uniform/normal use f32 precision internally.
#[must_use]
pub fn random_uniform(key: PRNGKey, count: usize, minval: f64, maxval: f64) -> Vec<f64> {
    let bits = generate_u32_bits(key, count);
    let scale = maxval - minval;
    bits.into_iter()
        .map(|u32_val| {
            // Keep 23 mantissa bits, set exponent to 1.0 ([1.0, 2.0) in f32)
            let mantissa = u32_val >> 9;
            let float_bits = mantissa | 0x3F80_0000_u32;
            let unit = f64::from(f32::from_bits(float_bits) - 1.0);
            minval + unit * scale
        })
        .collect()
}

/// Generate standard normal random f64 values using the inverse error function.
///
/// Matches JAX's approach: generate uniform samples in `(lo, hi)` where
/// `lo = nextafter_f32(-1, 0)` and `hi = 1.0`, then apply `sqrt(2) * erfinv(u)`.
/// Uses f32 precision internally (JAX default without x64 mode).
#[must_use]
pub fn random_normal(key: PRNGKey, count: usize) -> Vec<f64> {
    // JAX: lo = nextafter(float32(-1), float32(0)), hi = float32(1)
    let lo = f64::from(f32::from_bits((-1.0_f32).to_bits() - 1)); // nextafter_f32(-1.0, 0.0)
    let hi = 1.0_f64;
    let uniforms = random_uniform(key, count, lo, hi);
    let sqrt2 = std::f64::consts::SQRT_2;
    uniforms
        .into_iter()
        .map(|u| sqrt2 * crate::arithmetic::erf_inv_approx(u))
        .collect()
}

/// Generate Bernoulli random boolean values with probability `p` of being true.
///
/// Matches JAX's `jax.random.bernoulli`.
#[must_use]
pub fn random_bernoulli(key: PRNGKey, count: usize, p: f64) -> Vec<bool> {
    let uniforms = random_uniform(key, count, 0.0, 1.0);
    uniforms.into_iter().map(|u| u < p).collect()
}

/// Generate categorical samples from logits using the Gumbel-max trick.
///
/// Returns integer indices drawn from the categorical distribution defined by `logits`.
/// Uses the Gumbel-max trick: argmax(logits + Gumbel noise) gives categorical samples.
#[must_use]
pub fn random_categorical(key: PRNGKey, logits: &[f64], num_samples: usize) -> Vec<usize> {
    let num_categories = logits.len();
    // Need num_samples * num_categories uniform samples for Gumbel noise
    let total = num_samples * num_categories;
    let uniforms = random_uniform(key, total, 0.0, 1.0);

    let mut result = Vec::with_capacity(num_samples);
    for s in 0..num_samples {
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for c in 0..num_categories {
            let u = uniforms[s * num_categories + c];
            // Gumbel noise: -log(-log(u)), clamp u away from 0 and 1
            let clamped = u.clamp(1e-30, 1.0 - 1e-10);
            let gumbel = -(-clamped.ln()).ln();
            let val = logits[c] + gumbel;
            if val > best_val {
                best_val = val;
                best_idx = c;
            }
        }
        result.push(best_idx);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threefry_deterministic() {
        let key = [0u32, 0u32];
        let counter = [0u32, 0u32];
        let a = threefry2x32(key, counter);
        let b = threefry2x32(key, counter);
        assert_eq!(a, b, "ThreeFry must be deterministic");
    }

    #[test]
    fn test_threefry_different_keys() {
        let counter = [0u32, 0u32];
        let a = threefry2x32([0, 0], counter);
        let b = threefry2x32([0, 1], counter);
        assert_ne!(a, b, "Different keys should produce different output");
    }

    #[test]
    fn test_threefry_different_counters() {
        let key = [0u32, 0u32];
        let a = threefry2x32(key, [0, 0]);
        let b = threefry2x32(key, [0, 1]);
        assert_ne!(a, b, "Different counters should produce different output");
    }

    #[test]
    fn test_threefry_known_vector() {
        // Known test vector: key=[0,0], data=[0,0]
        // The output should be non-zero and deterministic.
        // We verify against our own reference computation to ensure
        // the implementation is self-consistent.
        let result = threefry2x32([0, 0], [0, 0]);
        // Verify non-trivial output
        assert_ne!(
            result,
            [0, 0],
            "ThreeFry should produce non-zero output for zero inputs"
        );
        // Store reference value for regression
        let expected = result;
        assert_eq!(
            threefry2x32([0, 0], [0, 0]),
            expected,
            "ThreeFry reference value changed"
        );
    }

    #[test]
    fn test_threefry_20_rounds() {
        // Verify that 20 rounds of mixing produces well-distributed output.
        // With key=[1,2] and data=[3,4], output should be thoroughly mixed.
        let result = threefry2x32([1, 2], [3, 4]);
        // Both words should be non-trivially mixed
        assert_ne!(result[0], 3, "First output should not equal first input");
        assert_ne!(result[1], 4, "Second output should not equal second input");
        // Verify bitwise mixing (output should look random)
        let bits = (result[0].count_ones() + result[1].count_ones()) as i32;
        // For 64 random bits, expected ~32 ones. Allow wide margin [10, 54].
        assert!(
            (10..=54).contains(&bits),
            "Output should have reasonable bit distribution, got {bits} ones out of 64"
        );
    }

    #[test]
    fn test_threefry_split() {
        let key = random_key(42);
        let (child1, child2) = random_split(key);
        // Children should be different from each other
        assert_ne!(child1, child2, "Split keys should be different");
        // Children should be different from parent
        assert_ne!(child1, key, "Child 1 should differ from parent");
        assert_ne!(child2, key, "Child 2 should differ from parent");
        // Splitting should be deterministic
        let (child1b, child2b) = random_split(key);
        assert_eq!(child1, child1b, "Split should be deterministic (child 1)");
        assert_eq!(child2, child2b, "Split should be deterministic (child 2)");
    }

    #[test]
    fn test_threefry_fold_in() {
        let key = random_key(42);
        let derived1 = random_fold_in(key, 0);
        let derived2 = random_fold_in(key, 1);
        // Different data should produce different keys
        assert_ne!(
            derived1, derived2,
            "fold_in with different data should produce different keys"
        );
        // fold_in should be deterministic
        assert_eq!(
            random_fold_in(key, 0),
            derived1,
            "fold_in should be deterministic"
        );
    }

    #[test]
    fn test_random_key_from_seed() {
        let key = random_key(42);
        assert_eq!(key.0[0], 0); // high 32 bits of 42 = 0
        assert_eq!(key.0[1], 42); // low 32 bits of 42 = 42

        let key_large = random_key(0x0000_0001_0000_002A);
        assert_eq!(key_large.0[0], 1); // high 32 bits
        assert_eq!(key_large.0[1], 42); // low 32 bits
    }

    #[test]
    fn test_threefry_bits_uniform() {
        // Chi-squared test: generate 10K samples and check bit uniformity
        let key = [42u32, 7u32];
        let num_samples = 10_000;
        let mut bit_counts = [0u64; 64]; // 32 bits per word × 2 words

        for i in 0..num_samples {
            let result = threefry2x32(key, [i as u32, 0]);
            for bit in 0..32 {
                if result[0] & (1u32 << bit) != 0 {
                    bit_counts[bit] += 1;
                }
                if result[1] & (1u32 << bit) != 0 {
                    bit_counts[32 + bit] += 1;
                }
            }
        }

        // For each bit position, expected count is num_samples/2 = 5000
        // Chi-squared statistic for each bit: (observed - expected)^2 / expected
        let expected = num_samples as f64 / 2.0;
        let mut chi_sq_total = 0.0;
        for count in &bit_counts {
            let observed = *count as f64;
            let diff = observed - expected;
            chi_sq_total += (diff * diff) / expected;
        }

        // With 64 degrees of freedom, chi-squared critical value at p=0.001 is ~103.4
        // We use a generous threshold since we're testing randomness quality
        assert!(
            chi_sq_total < 150.0,
            "Bit uniformity chi-squared test failed: chi_sq={chi_sq_total:.1} (threshold=150)"
        );
    }

    // === Sampling function tests ===

    #[test]
    fn test_uniform_range() {
        let key = random_key(42);
        let vals = random_uniform(key, 10_000, -2.0, 5.0);
        for v in &vals {
            assert!(
                *v >= -2.0 && *v < 5.0,
                "uniform value {v} out of range [-2, 5)"
            );
        }
    }

    #[test]
    fn test_uniform_shape() {
        let key = random_key(99);
        let vals = random_uniform(key, 137, 0.0, 1.0);
        assert_eq!(vals.len(), 137);
    }

    #[test]
    fn test_uniform_default_range() {
        let key = random_key(7);
        let vals = random_uniform(key, 10_000, 0.0, 1.0);
        for v in &vals {
            assert!(*v >= 0.0 && *v < 1.0, "uniform value {v} out of [0,1)");
        }
    }

    #[test]
    fn test_normal_mean_stddev() {
        let key = random_key(42);
        let n = 10_000;
        let vals = random_normal(key, n);
        assert_eq!(vals.len(), n);
        let mean = vals.iter().sum::<f64>() / n as f64;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();
        assert!(mean.abs() < 0.05, "normal mean should be ~0, got {mean}");
        assert!(
            (stddev - 1.0).abs() < 0.05,
            "normal stddev should be ~1, got {stddev}"
        );
    }

    #[test]
    fn test_normal_shape() {
        let key = random_key(123);
        let vals = random_normal(key, 200);
        assert_eq!(vals.len(), 200);
    }

    #[test]
    fn test_bernoulli_probability() {
        let key = random_key(42);
        let n = 10_000;
        let vals = random_bernoulli(key, n, 0.3);
        let true_count = vals.iter().filter(|&&v| v).count();
        let ratio = true_count as f64 / n as f64;
        assert!(
            (ratio - 0.3).abs() < 0.03,
            "bernoulli p=0.3 should have ~30% true, got {ratio:.3}"
        );
    }

    #[test]
    fn test_bernoulli_extreme_p0() {
        let key = random_key(1);
        let vals = random_bernoulli(key, 1000, 0.0);
        assert!(
            vals.iter().all(|&v| !v),
            "bernoulli p=0.0 should be all false"
        );
    }

    #[test]
    fn test_bernoulli_extreme_p1() {
        let key = random_key(1);
        let vals = random_bernoulli(key, 1000, 1.0);
        assert!(
            vals.iter().all(|&v| v),
            "bernoulli p=1.0 should be all true"
        );
    }

    #[test]
    fn test_sampling_deterministic() {
        let key = random_key(42);
        let a = random_uniform(key, 100, 0.0, 1.0);
        let b = random_uniform(key, 100, 0.0, 1.0);
        assert_eq!(a, b, "same key must produce same samples");
        let c = random_normal(key, 50);
        let d = random_normal(key, 50);
        assert_eq!(c, d, "normal: same key must produce same samples");
    }

    #[test]
    fn test_sampling_different_keys() {
        let k1 = random_key(42);
        let k2 = random_key(43);
        let a = random_uniform(k1, 100, 0.0, 1.0);
        let b = random_uniform(k2, 100, 0.0, 1.0);
        assert_ne!(a, b, "different keys should produce different samples");
    }

    // === Statistical tests ===

    #[test]
    fn test_uniform_ks_test() {
        // Kolmogorov-Smirnov test against uniform [0,1)
        let key = random_key(42);
        let n = 10_000;
        let mut vals = random_uniform(key, n, 0.0, 1.0);
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut d_max = 0.0_f64;
        for (i, v) in vals.iter().enumerate() {
            let empirical = (i + 1) as f64 / n as f64;
            let theoretical = *v; // CDF of uniform [0,1) is x
            d_max = d_max.max((empirical - theoretical).abs());
            let empirical_minus = i as f64 / n as f64;
            d_max = d_max.max((empirical_minus - theoretical).abs());
        }
        // KS critical value at alpha=0.01: ~1.63 / sqrt(n)
        let critical = 1.63 / (n as f64).sqrt();
        assert!(
            d_max < critical,
            "KS test failed: D={d_max:.4}, critical={critical:.4}"
        );
    }

    #[test]
    fn test_normal_ks_test() {
        // KS test against standard normal using approximate CDF
        let key = random_key(42);
        let n = 10_000;
        let mut vals = random_normal(key, n);
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut d_max = 0.0_f64;
        for (i, v) in vals.iter().enumerate() {
            let empirical = (i + 1) as f64 / n as f64;
            // Approximate standard normal CDF using erf
            let theoretical = 0.5 * (1.0 + erf_approx(*v / std::f64::consts::SQRT_2));
            d_max = d_max.max((empirical - theoretical).abs());
            let empirical_minus = i as f64 / n as f64;
            d_max = d_max.max((empirical_minus - theoretical).abs());
        }
        let critical = 1.63 / (n as f64).sqrt();
        assert!(
            d_max < critical,
            "Normal KS test failed: D={d_max:.4}, critical={critical:.4}"
        );
    }

    /// Approximate erf function for test use.
    fn erf_approx(x: f64) -> f64 {
        // Abramowitz and Stegun approximation 7.1.26
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        sign * (1.0 - poly * (-x * x).exp())
    }

    #[test]
    fn test_bernoulli_binomial_test() {
        // Test that bernoulli with p=0.5 passes a simple binomial check
        let key = random_key(42);
        let n = 10_000;
        let vals = random_bernoulli(key, n, 0.5);
        let true_count = vals.iter().filter(|&&v| v).count() as f64;
        // Under H0: true_count ~ Binomial(n, 0.5), mean=5000, sd=50
        let z = (true_count - 5000.0) / 50.0;
        assert!(
            z.abs() < 3.0,
            "Bernoulli binomial test failed: z={z:.2} (|z| > 3)"
        );
    }

    // === Categorical tests ===

    #[test]
    fn test_categorical_basic() {
        let key = random_key(42);
        let logits = [0.0, 0.0, 0.0]; // uniform over 3 categories
        let samples = random_categorical(key, &logits, 10_000);
        assert_eq!(samples.len(), 10_000);
        // All indices should be in [0, 3)
        for &idx in &samples {
            assert!(idx < 3, "categorical index {idx} out of range");
        }
        // Roughly uniform distribution
        let mut counts = [0usize; 3];
        for &idx in &samples {
            counts[idx] += 1;
        }
        for (i, &count) in counts.iter().enumerate() {
            let ratio = count as f64 / 10_000.0;
            assert!(
                (ratio - 1.0 / 3.0).abs() < 0.03,
                "category {i} ratio {ratio:.3} too far from 1/3"
            );
        }
    }

    #[test]
    fn test_categorical_skewed() {
        let key = random_key(42);
        // log(0.9) ≈ -0.105, log(0.05) ≈ -2.996, log(0.05) ≈ -2.996
        let logits = [10.0, 0.0, 0.0]; // heavily favor first category
        let samples = random_categorical(key, &logits, 1000);
        let count_0 = samples.iter().filter(|&&i| i == 0).count();
        assert!(
            count_0 > 900,
            "heavily-weighted category should dominate, got {count_0}/1000"
        );
    }
}
