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

/// Errors returned by [`random_categorical`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoricalError {
    EmptyLogits,
    SampleCountOverflow {
        num_samples: usize,
        num_categories: usize,
    },
}

impl std::fmt::Display for CategoricalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLogits => {
                write!(f, "categorical sampling requires at least one logit")
            }
            Self::SampleCountOverflow {
                num_samples,
                num_categories,
            } => write!(
                f,
                "categorical sampling size overflow: samples={num_samples} categories={num_categories}"
            ),
        }
    }
}

impl std::error::Error for CategoricalError {}

/// Errors returned by [`random_choice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceError {
    /// Samples were requested from an empty population.
    EmptyPopulation,
    /// `replace = false` but more samples were requested than the population size.
    SampleLargerThanPopulation { n_draws: usize, n_inputs: usize },
    /// Probability vector length did not match the population size.
    WeightsLengthMismatch { weights_len: usize, n_inputs: usize },
}

impl std::fmt::Display for ChoiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPopulation => write!(f, "choice requires a non-empty population"),
            Self::SampleLargerThanPopulation { n_draws, n_inputs } => write!(
                f,
                "cannot take a sample of size {n_draws} from population {n_inputs} without replacement"
            ),
            Self::WeightsLengthMismatch {
                weights_len,
                n_inputs,
            } => write!(
                f,
                "p must be a 1-D vector matching the population size: p.len()={weights_len}, population={n_inputs}"
            ),
        }
    }
}

impl std::error::Error for ChoiceError {}

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

/// N-way key split: derive `n` independent child keys, matching JAX's
/// `random.split(key, n)`. JAX's threefry split is prefix-stable (child `i` depends
/// only on `i`, so `split(key,n)[i] == split(key,m)[i]`), which the 2-way
/// [`random_split`] already exploits via counter `[0, i]`; this generalizes it to
/// arbitrary `n`. Verified against `jax.random.split(PRNGKey(0), {2,3})` bit-for-bit
/// (see `random_split_n_matches_jax`).
#[must_use]
pub fn random_split_n(key: PRNGKey, n: usize) -> Vec<PRNGKey> {
    (0..n)
        .map(|i| PRNGKey(threefry2x32(key.0, [0, i as u32])))
        .collect()
}

/// 2^-23, the mantissa-bits uniform step (see `random_uniform_serial_simd`).
const INV_2POW23_SCALAR: f64 = 1.0 / 8_388_608.0;

/// Allocation-free single uniform in `[0,1)` from `key` (the index-0 draw), bit-identical
/// to `random_uniform(key,1,0.0,1.0)[0]`: `bits = threefry(key,[0,0])[0]^[1]`, then
/// `(bits>>9)·2^-23`. The per-element rejection samplers (`gamma_one`) draw one value at a
/// time; routing each through the batch `random_uniform` (Vec alloc + threading setup)
/// made `random_gamma` ~145x slower than JAX. These scalar helpers keep the EXACT bits.
fn scalar_unit_uniform(key: PRNGKey) -> f64 {
    let [w0, w1] = threefry2x32(key.0, [0, 0]);
    let bits = w0 ^ w1;
    ((bits >> 9) as f64) * INV_2POW23_SCALAR
}

/// Allocation-free single uniform in `[minval, maxval)`.
fn scalar_uniform(key: PRNGKey, minval: f64, maxval: f64) -> f64 {
    minval + scalar_unit_uniform(key) * (maxval - minval)
}

/// Allocation-free single standard normal, bit-identical to `random_normal(key,1)[0]`
/// (uniform over `[nextafter_f32(-1,0), 1)` then `√2·erf_inv`).
fn scalar_normal(key: PRNGKey) -> f64 {
    let lo = f64::from(f32::from_bits((-1.0_f32).to_bits() - 1));
    let u = scalar_uniform(key, lo, 1.0);
    std::f64::consts::SQRT_2 * crate::arithmetic::erf_inv_approx(u)
}

/// 3-way split with no heap allocation, matching `random_split_n(key,3)`.
fn scalar_split3(key: PRNGKey) -> [PRNGKey; 3] {
    [
        PRNGKey(threefry2x32(key.0, [0, 0])),
        PRNGKey(threefry2x32(key.0, [0, 1])),
        PRNGKey(threefry2x32(key.0, [0, 2])),
    ]
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
///
/// ThreeFry is counter-based — sample `i` only depends on `i` — so the 20-round ARX
/// permutation is run on [`LANES`] consecutive counters at once with portable SIMD
/// (`std::simd`, 100% safe). Each lane reproduces the scalar [`threefry2x32`]
/// `[0, i]` exactly (integer add/rotate/xor/key-injection are bit-exact element-wise),
/// so the output is bit-for-bit identical to the scalar map — JAX RNG parity is
/// preserved (see `simd_generate_u32_bits_matches_scalar`). A scalar tail handles
/// the `count % LANES` remainder.
/// Minimum elements per worker before [`generate_u32_bits`] fans the raw
/// counter-based bit generation across threads. Same regime as the uniform
/// generator (a compute-bound 20-round ARX permutation that is its own producer,
/// not stacked on a threaded upstream), so it shares the `1<<18` cap that keeps
/// each worker's slice large enough to amortize the spawn.
const RANDOM_BITS_MIN_ELEMS_PER_THREAD: usize = 1 << 18;

/// Generate `count` raw u32 random words, matching JAX's partitionable
/// `_threefry_random_bits_partitionable` with `bit_width=32`: word `i` is
/// `threefry2x32(key, [0, i])[0] ^ [1]`.
///
/// ThreeFry is counter-based — word `i` depends only on its absolute counter `i`
/// — so generation partitions into independent contiguous ranges that fan out
/// across threads with NO cross-element dependency and BIT-IDENTICAL output for
/// any partition (proven by `threaded_u32_bits_matches_serial`). Small draws stay
/// on the single-threaded SIMD path; large draws fan out (the permutation is
/// compute-bound). Underpins randint (two draws), random_bits, and the
/// sort/argsort key path.
#[must_use]
pub fn generate_u32_bits(key: PRNGKey, count: usize) -> Vec<u32> {
    let hardware = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let threads = hardware.min(count / RANDOM_BITS_MIN_ELEMS_PER_THREAD);
    if threads <= 1 {
        return generate_u32_bits_serial(key, count);
    }

    let mut out = vec![0_u32; count];
    let chunk = count.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [u32] = out.as_mut_slice();
        let mut start = 0usize;
        while start < count {
            let len = chunk.min(count - start);
            let (block, tail) = rest.split_at_mut(len);
            rest = tail;
            scope.spawn(move || fill_u32_bits(block, start, key));
            start += len;
        }
    });
    out
}

/// Fill `out[j]` with the raw u32 word for absolute ThreeFry counter
/// `global_start + j`. Counter-based, so correct for ANY contiguous sub-range —
/// the shared kernel behind both the serial and threaded `generate_u32_bits`
/// paths. Bit-identical to the scalar `threefry2x32([0, i])` map per word
/// (`simd_generate_u32_bits_matches_scalar`).
fn fill_u32_bits(out: &mut [u32], global_start: usize, key: PRNGKey) {
    use std::simd::Simd;

    const LANES: usize = 8;
    const KS_PARITY: u32 = 0x1BD1_1BDA;

    let [k0, k1] = key.0;
    let ks2 = k0 ^ k1 ^ KS_PARITY;
    let ksched = [k0, k1, ks2];

    let k0v: Simd<u32, LANES> = Simd::splat(k0);
    let k1v: Simd<u32, LANES> = Simd::splat(k1);
    let lane_off: Simd<u32, LANES> = Simd::from_array(std::array::from_fn(|r| r as u32));

    let n = out.len();
    let chunks = n / LANES;
    for c in 0..chunks {
        let base = (global_start + c * LANES) as u32;
        // data = [0, counter]; x0 = 0 + k0, x1 = counter + k1.
        let mut x0 = k0v;
        let mut x1 = (Simd::splat(base) + lane_off) + k1v;

        for round in 0..NUM_ROUNDS {
            x0 += x1;
            let r = ROTATIONS[round % 8];
            // rotate_left(r) lane-wise: r ∈ {6,13,15,16,17,24,26,29} so 32-r is in 1..=31.
            let rotated = (x1 << Simd::splat(r)) | (x1 >> Simd::splat(32 - r));
            x1 = rotated ^ x0;

            if (round + 1) % 4 == 0 {
                let inject_idx = (round + 1) / 4;
                x0 += Simd::splat(ksched[inject_idx % 3]);
                x1 += Simd::splat(ksched[(inject_idx + 1) % 3].wrapping_add(inject_idx as u32));
            }
        }

        out[c * LANES..c * LANES + LANES].copy_from_slice((x0 ^ x1).as_array());
    }

    // Scalar tail (range length % LANES): same absolute counters, bit-identical
    // to the SIMD lanes per word.
    for (j, slot) in out.iter_mut().enumerate().skip(chunks * LANES) {
        let i = (global_start + j) as u32;
        let [a, b] = threefry2x32(key.0, [0, i]);
        *slot = a ^ b;
    }
}

/// Single-threaded SIMD generation of `count` raw u32 random words. Retained as
/// the threaded path's bit-identity oracle and A/B baseline.
#[must_use]
pub fn generate_u32_bits_serial(key: PRNGKey, count: usize) -> Vec<u32> {
    use std::simd::Simd;

    const LANES: usize = 8;
    const KS_PARITY: u32 = 0x1BD1_1BDA;

    let [k0, k1] = key.0;
    let ks2 = k0 ^ k1 ^ KS_PARITY;
    let ksched = [k0, k1, ks2];

    let mut out: Vec<u32> = Vec::with_capacity(count);
    let chunks = count / LANES;

    let k0v: Simd<u32, LANES> = Simd::splat(k0);
    let k1v: Simd<u32, LANES> = Simd::splat(k1);
    // Lane offsets [0, 1, …, LANES-1]; added to each chunk's base counter.
    let lane_off: Simd<u32, LANES> = Simd::from_array(std::array::from_fn(|r| r as u32));

    for c in 0..chunks {
        let base = (c * LANES) as u32;
        // data = [0, counter]; x0 = 0 + k0, x1 = counter + k1.
        let mut x0 = k0v;
        let mut x1 = (Simd::splat(base) + lane_off) + k1v;

        for round in 0..NUM_ROUNDS {
            x0 += x1;
            let r = ROTATIONS[round % 8];
            // rotate_left(r) lane-wise: r ∈ {6,13,15,16,17,24,26,29} so 32-r is in 1..=31.
            let rotated = (x1 << Simd::splat(r)) | (x1 >> Simd::splat(32 - r));
            x1 = rotated ^ x0;

            if (round + 1) % 4 == 0 {
                let inject_idx = (round + 1) / 4;
                x0 += Simd::splat(ksched[inject_idx % 3]);
                x1 += Simd::splat(ksched[(inject_idx + 1) % 3].wrapping_add(inject_idx as u32));
            }
        }

        out.extend_from_slice((x0 ^ x1).as_array());
    }

    // Scalar tail for the final partial chunk.
    for i in (chunks * LANES)..count {
        let [a, b] = threefry2x32(key.0, [0, i as u32]);
        out.push(a ^ b);
    }

    out
}

/// Minimum elements assigned to each worker before [`random_uniform`] adds
/// another thread. ThreeFry's 20-round ARX permutation is compute-bound, but the
/// generated bytes still hit memory, so over-spawning tiny slices loses to thread
/// overhead (measured: 64 threads on a 64K draw regress to 0.24x). Capping the
/// worker count at `count / this` keeps each thread's slice large enough to
/// amortize the spawn — the threaded path only engages once at least two such
/// slices exist (≈ 512K elements), and below that the single-threaded SIMD path
/// runs unchanged.
const RANDOM_UNIFORM_MIN_ELEMS_PER_THREAD: usize = 1 << 18;

/// Generate uniform random f64 values in `[minval, maxval)`, matching JAX's
/// `jax.random.uniform` (default f32 mode).
///
/// ThreeFry is counter-based — sample `i` depends only on counter `i` — so the
/// output partitions into independent contiguous ranges that fan out across
/// threads with NO cross-element dependency. Each thread fills its slice with the
/// SAME absolute counters it would see in the serial pass, so the result is
/// bit-for-bit identical to [`random_uniform_serial_simd`] for any partition
/// (proven by `threaded_uniform_matches_serial_bits`). Small draws stay on the
/// single-threaded SIMD path; large draws fan out (the permutation is
/// compute-bound, the ideal threading regime).
#[must_use]
pub fn random_uniform(key: PRNGKey, count: usize, minval: f64, maxval: f64) -> Vec<f64> {
    let hardware = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let threads = hardware.min(count / RANDOM_UNIFORM_MIN_ELEMS_PER_THREAD);
    if threads <= 1 {
        return random_uniform_serial_simd(key, count, minval, maxval);
    }

    let scale = maxval - minval;
    let mut out = vec![0.0_f64; count];
    let chunk = count.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = out.as_mut_slice();
        let mut start = 0usize;
        while start < count {
            let len = chunk.min(count - start);
            let (block, tail) = rest.split_at_mut(len);
            rest = tail;
            scope.spawn(move || fill_uniform_simd(block, start, key, minval, scale));
            start += len;
        }
    });
    out
}

/// Fill `out[j]` with the uniform sample for absolute ThreeFry counter
/// `global_start + j`, scaled into `[minval, minval + scale)`. Counter-based, so
/// this is correct for ANY contiguous sub-range — the shared kernel behind both
/// the serial and threaded `random_uniform` paths. Bit-identical to the scalar
/// formula (the SIMD lanes reproduce the per-counter `threefry2x32([0, i])` map
/// and the exact-f64 mantissa unit, see `simd_uniform_matches_scalar_bits`).
fn fill_uniform_simd(out: &mut [f64], global_start: usize, key: PRNGKey, minval: f64, scale: f64) {
    use std::simd::{Simd, num::SimdUint};

    const LANES: usize = 8;
    const KS_PARITY: u32 = 0x1BD1_1BDA;
    const INV_2POW23: f64 = 1.0 / 8_388_608.0;

    let [k0, k1] = key.0;
    let ks2 = k0 ^ k1 ^ KS_PARITY;
    let ksched = [k0, k1, ks2];

    let k0v: Simd<u32, LANES> = Simd::splat(k0);
    let k1v: Simd<u32, LANES> = Simd::splat(k1);
    let lane_off: Simd<u32, LANES> = Simd::from_array(std::array::from_fn(|r| r as u32));
    let inv2_23: Simd<f64, LANES> = Simd::splat(INV_2POW23);
    let scalev: Simd<f64, LANES> = Simd::splat(scale);
    let minv: Simd<f64, LANES> = Simd::splat(minval);

    let n = out.len();
    let chunks = n / LANES;
    for c in 0..chunks {
        let base = (global_start + c * LANES) as u32;
        let mut x0 = k0v;
        let mut x1 = (Simd::splat(base) + lane_off) + k1v;
        for round in 0..NUM_ROUNDS {
            x0 += x1;
            let r = ROTATIONS[round % 8];
            let rotated = (x1 << Simd::splat(r)) | (x1 >> Simd::splat(32 - r));
            x1 = rotated ^ x0;
            if (round + 1) % 4 == 0 {
                let inject_idx = (round + 1) / 4;
                x0 += Simd::splat(ksched[inject_idx % 3]);
                x1 += Simd::splat(ksched[(inject_idx + 1) % 3].wrapping_add(inject_idx as u32));
            }
        }
        let mantissa = (x0 ^ x1) >> Simd::splat(9_u32);
        let unit = mantissa.cast::<f64>() * inv2_23;
        let res = minv + unit * scalev;
        out[c * LANES..c * LANES + LANES].copy_from_slice(res.as_array());
    }

    // Scalar tail (range length % LANES): same exact-f64 unit formula, same
    // absolute counters — bit-identical to the SIMD lanes per element.
    for (j, slot) in out.iter_mut().enumerate().skip(chunks * LANES) {
        let i = (global_start + j) as u32;
        let [a, b] = threefry2x32(key.0, [0, i]);
        let mantissa = (a ^ b) >> 9;
        *slot = minval + f64::from(mantissa) * INV_2POW23 * scale;
    }
}

/// Single-threaded SIMD generation of uniform f64 values in `[minval, maxval)`,
/// matching JAX's `jax.random.uniform` with default f32 mode (x64 not enabled):
/// 1. Generate one u32 per sample via XOR of threefry outputs
/// 2. Right-shift by 9 to keep 23 mantissa bits (f32 precision)
/// 3. OR with f32 1.0's bit pattern (0x3F800000), bitcast to f32, subtract 1.0
/// 4. Convert to f64 and scale to [minval, maxval)
///
/// Note: JAX defaults to f32 unless `jax_enable_x64` is set. The oracle fixtures
/// were captured without x64 mode, so uniform/normal use f32 precision internally.
/// Retained as the threaded path's bit-identity oracle and A/B baseline.
#[must_use]
pub fn random_uniform_serial_simd(
    key: PRNGKey,
    count: usize,
    minval: f64,
    maxval: f64,
) -> Vec<f64> {
    use std::simd::{Simd, num::SimdUint};

    const LANES: usize = 8;
    const KS_PARITY: u32 = 0x1BD1_1BDA;
    // 2^-23. The scalar mantissa step `f32::from_bits((m>>9)|0x3F800000) - 1.0`
    // (a value in [1,2) minus 1) equals `(m>>9) as f64 * 2^-23` EXACTLY: `m>>9`
    // is 23 bits, so both `1 + (m>>9)·2^-23` (f32) and `(m>>9)·2^-23` (f64) are
    // representable and the subtraction is exact. So the SIMD path computes the
    // unit directly in f64 (no f32 round-trip), bit-for-bit identical to the
    // scalar formula — proven by `simd_uniform_matches_scalar_bits`.
    const INV_2POW23: f64 = 1.0 / 8_388_608.0;

    let [k0, k1] = key.0;
    let ks2 = k0 ^ k1 ^ KS_PARITY;
    let ksched = [k0, k1, ks2];
    let scale = maxval - minval;

    let mut out: Vec<f64> = Vec::with_capacity(count);
    let chunks = count / LANES;

    let k0v: Simd<u32, LANES> = Simd::splat(k0);
    let k1v: Simd<u32, LANES> = Simd::splat(k1);
    let lane_off: Simd<u32, LANES> = Simd::from_array(std::array::from_fn(|r| r as u32));
    let inv2_23: Simd<f64, LANES> = Simd::splat(INV_2POW23);
    let scalev: Simd<f64, LANES> = Simd::splat(scale);
    let minv: Simd<f64, LANES> = Simd::splat(minval);

    // Fused threefry → f64 uniform straight into the output: no intermediate
    // Vec<u32>, mantissa conversion vectorized. Same lane stream + exact-f64 unit
    // formula as the scalar path, so the bits are identical.
    for c in 0..chunks {
        let base = (c * LANES) as u32;
        let mut x0 = k0v;
        let mut x1 = (Simd::splat(base) + lane_off) + k1v;
        for round in 0..NUM_ROUNDS {
            x0 += x1;
            let r = ROTATIONS[round % 8];
            let rotated = (x1 << Simd::splat(r)) | (x1 >> Simd::splat(32 - r));
            x1 = rotated ^ x0;
            if (round + 1) % 4 == 0 {
                let inject_idx = (round + 1) / 4;
                x0 += Simd::splat(ksched[inject_idx % 3]);
                x1 += Simd::splat(ksched[(inject_idx + 1) % 3].wrapping_add(inject_idx as u32));
            }
        }
        let mantissa = (x0 ^ x1) >> Simd::splat(9_u32);
        let unit = mantissa.cast::<f64>() * inv2_23;
        let res = minv + unit * scalev;
        out.extend_from_slice(res.as_array());
    }

    // Scalar tail (count % LANES): same exact-f64 unit formula.
    for i in (chunks * LANES)..count {
        let [a, b] = threefry2x32(key.0, [0, i as u32]);
        let mantissa = (a ^ b) >> 9;
        out.push(minval + f64::from(mantissa) * INV_2POW23 * scale);
    }

    out
}

/// Scalar reference for [`random_uniform`] using the original f32-bitcast mantissa
/// formula, over the scalar bit generator. Retained as the bit-identity oracle
/// (`simd_uniform_matches_scalar_bits`) and the A/B benchmark baseline.
#[must_use]
pub fn random_uniform_scalar(key: PRNGKey, count: usize, minval: f64, maxval: f64) -> Vec<f64> {
    let scale = maxval - minval;
    (0..count)
        .map(|i| {
            let [a, b] = threefry2x32(key.0, [0, i as u32]);
            let mantissa = (a ^ b) >> 9;
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
    random_normal_from_uniforms(&uniforms)
}

const RANDOM_NORMAL_PARALLEL_MIN_ELEMS: usize = 1 << 14;

fn random_normal_from_uniforms(uniforms: &[f64]) -> Vec<f64> {
    let count = uniforms.len();
    let threads = if count >= RANDOM_NORMAL_PARALLEL_MIN_ELEMS {
        std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1)
    } else {
        1
    };
    random_normal_from_uniforms_with_threads(uniforms, threads)
}

fn random_normal_from_uniforms_serial(uniforms: &[f64]) -> Vec<f64> {
    let sqrt2 = std::f64::consts::SQRT_2;
    uniforms
        .iter()
        .map(|&u| sqrt2 * crate::arithmetic::erf_inv_approx(u))
        .collect()
}

fn random_normal_from_uniforms_with_threads(uniforms: &[f64], threads: usize) -> Vec<f64> {
    let count = uniforms.len();
    let threads = threads.min(count);
    if threads <= 1 {
        return random_normal_from_uniforms_serial(uniforms);
    }

    let mut out = vec![0.0_f64; count];
    let chunk = count.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = out.as_mut_slice();
        let mut start = 0usize;
        while start < count {
            let len = chunk.min(count - start);
            let (out_block, tail) = rest.split_at_mut(len);
            rest = tail;
            let input_block = &uniforms[start..start + len];
            scope.spawn(move || {
                let sqrt2 = std::f64::consts::SQRT_2;
                for (slot, &u) in out_block.iter_mut().zip(input_block) {
                    *slot = sqrt2 * crate::arithmetic::erf_inv_approx(u);
                }
            });
            start += len;
        }
    });
    out
}

/// Minimum elements per worker before a distribution's per-element transcendental
/// transform fans out across threads. The inverse-transform maps (`ln`/`log`/
/// `tan`/`erf_inv` over the drawn uniforms) are compute-bound, so threading pays
/// off on large draws — but the uniforms are themselves drawn on the threaded
/// path, so a mid-size transform competes with that fan-out and an aggressive
/// threshold REGRESSES (measured on a 64-core box: a 131072-element draw fell to
/// 0.71x at `1<<16`). Matching the uniform generator's `1<<18` keeps each worker's
/// slice large enough that threading is a clear win (≥2x from ~1M up) and leaves
/// mid-size draws on the unchanged serial map.
const DIST_TRANSFORM_MIN_ELEMS_PER_THREAD: usize = 1 << 18;

/// Apply an elementwise transform `f` to drawn `uniforms`, fanning out across
/// threads for large counts. Output `i` depends only on `uniforms[i]`, so
/// threading preserves per-element order and is BIT-IDENTICAL to
/// `uniforms.into_iter().map(f).collect()` for ANY partition (proven by
/// `map_uniforms_parallel_matches_serial_bits`). The transcendental
/// inverse-transform distributions route their now-bottleneck `ln`/`tan`/`erf_inv`
/// maps through here so the transform keeps pace with the threaded uniform draw.
fn map_uniforms_parallel<F>(uniforms: Vec<f64>, f: F) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Sync,
{
    let count = uniforms.len();
    let hardware = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let threads = hardware.min(count / DIST_TRANSFORM_MIN_ELEMS_PER_THREAD);
    if threads <= 1 {
        return uniforms.into_iter().map(f).collect();
    }

    let mut out = vec![0.0_f64; count];
    let chunk = count.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut out_rest: &mut [f64] = out.as_mut_slice();
        let mut in_rest: &[f64] = uniforms.as_slice();
        let f_ref = &f;
        while !out_rest.is_empty() {
            let len = chunk.min(out_rest.len());
            let (out_block, out_tail) = out_rest.split_at_mut(len);
            let (in_block, in_tail) = in_rest.split_at(len);
            out_rest = out_tail;
            in_rest = in_tail;
            scope.spawn(move || {
                for (slot, &u) in out_block.iter_mut().zip(in_block) {
                    *slot = f_ref(u);
                }
            });
        }
    });
    out
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
pub fn random_categorical(
    key: PRNGKey,
    logits: &[f64],
    num_samples: usize,
) -> Result<Vec<usize>, CategoricalError> {
    let num_categories = logits.len();
    if num_categories == 0 {
        return Err(CategoricalError::EmptyLogits);
    }

    // Need num_samples * num_categories uniform samples for Gumbel noise
    let total =
        num_samples
            .checked_mul(num_categories)
            .ok_or(CategoricalError::SampleCountOverflow {
                num_samples,
                num_categories,
            })?;
    // Gumbel-max trick with JAX's gumbel noise: -log(-log(uniform(finfo.tiny, 1)))
    // (matches `random_gumbel` / jax.random.gumbel — minval=tiny, no upper clamp).
    let tiny = f64::from(f32::MIN_POSITIVE);
    let uniforms = random_uniform(key, total, tiny, 1.0);

    let mut result = Vec::with_capacity(num_samples);
    for s in 0..num_samples {
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for c in 0..num_categories {
            let u = uniforms[s * num_categories + c];
            let gumbel = -(-u.ln()).ln();
            let val = logits[c] + gumbel;
            if val > best_val {
                best_val = val;
                best_idx = c;
            }
        }
        result.push(best_idx);
    }
    Ok(result)
}

/// Generate exponentially distributed samples with rate parameter `rate` (λ).
///
/// Matches JAX's `jax.random.exponential(key, shape)` (rate 1) scaled by
/// `1/rate`. JAX computes `-log1p(-u)` rather than `-log(u)`, deliberately
/// taking `1 - u` so the log domain is `(0, 1]` instead of `[0, 1)`: at `u = 0`
/// the result is exactly `0` and, since `uniform` is half-open `[0, 1)`, `u < 1`
/// keeps it finite — so no ad-hoc clamp is needed. The previous `-ln(max(u,
/// 1e-30))` both paired `u -> X` on the opposite tail from JAX and bolted on a
/// clamp hack, diverging from JAX bit-for-bit.
#[must_use]
pub fn random_exponential(key: PRNGKey, count: usize, rate: f64) -> Vec<f64> {
    let uniforms = random_uniform(key, count, 0.0, 1.0);
    map_uniforms_parallel(uniforms, move |u| -(-u).ln_1p() / rate)
}

/// Generate Gumbel-distributed samples with location `loc` and scale `scale`.
///
/// Matches JAX's `jax.random.gumbel(key, shape)` with optional loc/scale.
/// Uses inverse transform: X = loc - scale * ln(-ln(U)) where U ~ Uniform(0,1).
#[must_use]
pub fn random_gumbel(key: PRNGKey, count: usize, loc: f64, scale: f64) -> Vec<f64> {
    // JAX's `_gumbel`: -log(-log(uniform(finfo.tiny, 1))), generalized here with
    // loc/scale (loc + scale*standard). The RNG core is JAX-f32-mode bit-exact,
    // so use the f32 `tiny` as the uniform lower bound. The previous form drew
    // uniform(0,1) and clamped to [1e-30, 1-1e-10] — JAX uses minval=tiny (not
    // 1e-30, which only matters at the rare u==0 sample) and has no upper clamp.
    let tiny = f64::from(f32::MIN_POSITIVE);
    let uniforms = random_uniform(key, count, tiny, 1.0);
    map_uniforms_parallel(uniforms, move |u| loc - scale * (-u.ln()).ln())
}

/// Generate Laplace (double exponential) distributed samples.
///
/// Matches JAX's `jax.random.laplace(key, shape)` with loc and scale.
/// Uses inverse transform on U ~ Uniform(-0.5, 0.5).
#[must_use]
pub fn random_laplace(key: PRNGKey, count: usize, loc: f64, scale: f64) -> Vec<f64> {
    // JAX's `_laplace`: u ~ uniform(finfo.eps - 1, 1); laplace = sign(u)·log1p(-|u|),
    // generalized here with loc/scale (loc + scale·standard_laplace). The RNG core
    // is JAX-f32-mode bit-exact, so use the f32 eps for the lower bound.
    //
    // The previous form drew uniform(0,1), shifted to (-0.5, 0.5), and computed
    // `-sign(·)·ln(1 - 2|·|)` with a 1e-10 clamp — that FLIPPED the sign versus
    // JAX on every sample, used a different uniform domain, and added a clamp JAX
    // does not have. (Both are valid Laplace, so the statistical tests passed, but
    // it diverged per-sample.)
    let minval = f64::from(f32::EPSILON - 1.0);
    let uniforms = random_uniform(key, count, minval, 1.0);
    map_uniforms_parallel(uniforms, move |u| {
        loc + scale * u.signum() * (-u.abs()).ln_1p()
    })
}

/// Sample from the logistic distribution.
///
/// Matches JAX's `_logistic`: `x ~ uniform(finfo.tiny, 1); logistic = log(x) -
/// log1p(-x)` (the numerically stable form of `logit(x)`). Generalized here with
/// loc/scale (`loc + scale·standard_logistic`); `loc=0, scale=1` reproduces
/// `jax.random.logistic` exactly. The RNG core is JAX-f32-mode bit-exact, so the
/// uniform lower bound is the f32 `tiny` (matching `_logistic`'s
/// `minval=finfo(dtype).tiny`, with dtype=float32 when x64 is disabled) and there
/// is no upper clamp.
#[must_use]
pub fn random_logistic(key: PRNGKey, count: usize, loc: f64, scale: f64) -> Vec<f64> {
    let tiny = f64::from(f32::MIN_POSITIVE);
    let uniforms = random_uniform(key, count, tiny, 1.0);
    // log(x) - log1p(-x); JAX computes it in exactly this two-term form rather
    // than as a single log(x/(1-x)) for accuracy as x -> 1.
    map_uniforms_parallel(uniforms, move |x| loc + scale * (x.ln() - (-x).ln_1p()))
}

/// Generate random integers uniformly in [minval, maxval).
///
/// Matches JAX's `jax.random.randint(key, shape, minval, maxval)` with the
/// default `int32` dtype (x64 disabled), bit-for-bit.
///
/// JAX deliberately avoids the "uniform float, cast to int" approach (which
/// biases large ranges because many integers are never sampled). Instead it
/// samples `2 * nbits` random bits per value (two `nbits`-wide words from a
/// split key) and reduces them modulo `span = maxval - minval`, which only
/// leaves an `O(span^2 / 2^(2·nbits))` bias. For `int32`, `nbits = 32`:
///
/// ```text
/// k1, k2          = split(key)
/// higher, lower   = random_bits(k1), random_bits(k2)      // u32 words
/// span            = (maxval - minval) as u32
/// multiplier      = (2^16 mod span)^2 mod span            // == 2^32 mod span
/// offset          = ((higher mod span) * multiplier + (lower mod span)) mod span
/// result          = minval + offset
/// ```
///
/// The `*`/`+` are u32 operations that wrap at 2^32 (matching XLA's unsigned
/// integer arithmetic), using the identity `(a·b) mod N = ((a mod N)·(b mod N))
/// mod N` to fold the high word in without a 64-bit intermediate. The previous
/// implementation used `minval + (u * range).floor()` over an f64 uniform,
/// which is exactly the biased float-cast approach JAX rejects and diverged
/// from JAX per-sample. `random_split` and `random_bits` are both oracle-pinned
/// to JAX (see `random_determinism` fixtures), so this is bit-exact for int32.
#[must_use]
pub fn random_randint(key: PRNGKey, count: usize, minval: i64, maxval: i64) -> Vec<i64> {
    // JAX: span = 1 when maxval <= minval, so every sample reduces to minval.
    if maxval <= minval {
        return vec![minval; count];
    }
    let (k1, k2) = random_split(key);
    let higher = generate_u32_bits(k1, count);
    let lower = generate_u32_bits(k2, count);

    // span fits u32: maxval/minval are int32-range, so 0 < maxval - minval < 2^32.
    let span = (maxval - minval) as u32;
    // multiplier = 2^32 mod span, computed as (2^16 mod span)^2 mod span with a
    // wrapping u32 multiply (the square overflows u32 for span > 2^16).
    let half = (1u32 << 16) % span;
    let multiplier = half.wrapping_mul(half) % span;

    higher
        .into_iter()
        .zip(lower)
        .map(|(hi, lo)| {
            let offset = (hi % span).wrapping_mul(multiplier).wrapping_add(lo % span) % span;
            minval + i64::from(offset)
        })
        .collect()
}

/// Randomly permute `0..n`, matching `jax.random.permutation(key, n)` /
/// `jax.random.shuffle`.
///
/// JAX does NOT use Fisher-Yates (the previous implementation here, which drew
/// `n-1` uniforms and swapped — a valid permutation but a DIFFERENT one than JAX
/// for the same key). JAX's `_shuffle` (tjablin's parallel-friendly algorithm)
/// runs several rounds of "sort the array by freshly-drawn random u32 keys":
///
/// ```text
/// num_rounds = ceil(3 * ln(max(1, n)) / ln(u32::MAX))
/// x = [0, 1, ..., n-1]
/// repeat num_rounds:
///     key, subkey = split(key)
///     sort_keys = random_bits(subkey, 32, [n])   # one u32 per element
///     x = stable_argsort(sort_keys) applied to x  # lax.sort_key_val(sort_keys, x)
/// ```
///
/// Each round permutes by a stable sort on independent u32 keys; repeating until
/// the concatenated per-element key strings are (almost surely) distinct yields a
/// uniform permutation. We reuse the JAX-oracle-pinned `random_split` and
/// `generate_u32_bits` (== JAX `_random_bits` bit_width=32) and a stable ascending
/// sort, so this follows JAX's algorithm rather than diverging via Fisher-Yates.
/// `n <= 1` needs zero rounds (the permutation is the identity), exactly as JAX's
/// `ceil(...)` yields 0 rounds for `n == 1`.
#[must_use]
pub fn random_permutation(key: PRNGKey, n: usize) -> Vec<usize> {
    if n <= 1 {
        return (0..n).collect();
    }

    // num_rounds = ceil(3 * ln(n) / ln(u32::MAX)); matches JAX's static stop.
    let exponent = 3.0_f64;
    let uint32max = f64::from(u32::MAX);
    let num_rounds = (exponent * (n as f64).ln() / uint32max.ln()).ceil() as usize;

    let mut x: Vec<usize> = (0..n).collect();
    let mut key = key;
    for _ in 0..num_rounds {
        // JAX: `key, subkey = split(key)` — split[0] continues, split[1] draws bits.
        let (next_key, subkey) = random_split(key);
        key = next_key;
        let sort_keys = generate_u32_bits(subkey, n);
        // Stable ascending argsort by the u32 sort keys (== lax.sort_key_val,
        // is_stable=True), then reorder x by it.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| sort_keys[i]);
        x = order.into_iter().map(|i| x[i]).collect();
    }
    x
}

/// Sample `n_draws` indices from `0..n_inputs`, matching `jax.random.choice` in
/// its integer-population form (`choice(key, n_inputs, shape, replace, p)`
/// returning indices into `arange(n_inputs)`).
///
/// All four JAX branches are reproduced verbatim, each drawing once from the raw
/// `key` (JAX does no key splitting inside `choice`):
/// - `p=None, replace=true`  → `randint(key, n_draws, 0, n_inputs)`
/// - `p=None, replace=false` → `permutation(key, n_inputs)[:n_draws]`
/// - `p=Some, replace=true`  → inverse-CDF: `cuml=cumsum(p); r=cuml[-1]*(1-uniform);
///   ind=searchsorted_left(cuml, r)`
/// - `p=Some, replace=false` → Gumbel-top-k: `g=gumbel(key, n_inputs)+ln(p);
///   ind=top_k(g, n_draws)` (indices of the largest `n_draws`, descending)
///
/// Errors mirror JAX: empty population, sampling more than the population without
/// replacement, and a probability vector whose length ≠ `n_inputs`.
pub fn random_choice(
    key: PRNGKey,
    n_inputs: usize,
    n_draws: usize,
    replace: bool,
    p: Option<&[f64]>,
) -> Result<Vec<usize>, ChoiceError> {
    if n_draws == 0 {
        return Ok(Vec::new());
    }
    if n_inputs == 0 {
        return Err(ChoiceError::EmptyPopulation);
    }
    if !replace && n_draws > n_inputs {
        return Err(ChoiceError::SampleLargerThanPopulation { n_draws, n_inputs });
    }
    if let Some(weights) = p
        && weights.len() != n_inputs
    {
        return Err(ChoiceError::WeightsLengthMismatch {
            weights_len: weights.len(),
            n_inputs,
        });
    }

    match (p, replace) {
        (None, true) => Ok(random_randint(key, n_draws, 0, n_inputs as i64)
            .into_iter()
            .map(|i| i as usize)
            .collect()),
        (None, false) => {
            let mut perm = random_permutation(key, n_inputs);
            perm.truncate(n_draws);
            Ok(perm)
        }
        (Some(weights), true) => {
            // Inverse-CDF sampling against the cumulative weights. JAX:
            // r = p_cuml[-1] * (1 - uniform); ind = searchsorted(p_cuml, r, 'left').
            let mut cuml = Vec::with_capacity(n_inputs);
            let mut acc = 0.0_f64;
            for &w in weights {
                acc += w;
                cuml.push(acc);
            }
            let total = *cuml.last().expect("n_inputs > 0");
            let uniforms = random_uniform(key, n_draws, 0.0, 1.0);
            Ok(uniforms
                .into_iter()
                .map(|u| {
                    let r = total * (1.0 - u);
                    // searchsorted side='left' == count of elements strictly < r.
                    // r <= total == cuml[last], so this is always <= n_inputs-1.
                    cuml.partition_point(|&x| x < r).min(n_inputs - 1)
                })
                .collect())
        }
        (Some(weights), false) => {
            // Gumbel top-k trick: argsort of (gumbel + log p), take the top n_draws.
            let gumbel = random_gumbel(key, n_inputs, 0.0, 1.0);
            let mut scored: Vec<(f64, usize)> = (0..n_inputs)
                .map(|i| (gumbel[i] + weights[i].ln(), i))
                .collect();
            // top_k returns the largest values descending; break ties by lower
            // index (gumbel values are continuous, so ties are measure-zero).
            scored.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            Ok(scored.into_iter().take(n_draws).map(|(_, i)| i).collect())
        }
    }
}

/// Generate gamma-distributed samples using Marsaglia & Tsang's method.
///
/// Matches `jax.random.gamma(key, a, shape)` where `a` is the shape parameter.
/// Uses the transformation method for a >= 1 and rejection sampling otherwise.
#[must_use]
pub fn random_gamma(key: PRNGKey, count: usize, shape_param: f64) -> Vec<f64> {
    if shape_param <= 0.0 {
        return vec![f64::NAN; count];
    }
    // JAX `random.gamma`: split the base key into `count` per-element keys and run an
    // INDEPENDENT Marsaglia-Tsang stream per element (jax `_gamma_impl` vmaps
    // `_gamma_one` over `split(key, count)`). This replaces the previous non-JAX
    // shared-pool heuristic (10x-oversample + global reject loop) that diverged
    // element-wise. Draws use fj's JAX-f32-matching uniform/normal + `random_split_n`.
    gamma_one_map_parallel(random_split_n(key, count), shape_param, false)
}

/// Minimum element count before the per-element gamma/beta rejection sampler fans out
/// across threads. `gamma_one` runs a Marsaglia-Tsang accept/reject loop (each iteration
/// draws a fresh normal — itself a threefry block + `erf_inv` — plus a uniform), so it is
/// ~1-2 orders of magnitude costlier than the inverse-transform maps gated at `1<<18`; a
/// much lower threshold still keeps every worker busy. Below this the serial map wins.
const GAMMA_PARALLEL_MIN_ELEMS: usize = 1 << 13;

/// Thread the independent-per-element `gamma_one` draw over its already-split subkeys.
/// Output `i` depends ONLY on `keys[i]` (JAX splits one subkey per element, no shared
/// state), so any partition is BIT-IDENTICAL to
/// `keys.into_iter().map(|k| gamma_one(k, alpha, log_space)).collect()`
/// (guarded by `gamma_map_parallel_matches_serial_bits`). `FJ_GAMMA_SERIAL` forces the
/// serial reference for a same-invocation A/B.
fn gamma_one_map_parallel(keys: Vec<PRNGKey>, alpha: f64, log_space: bool) -> Vec<f64> {
    gamma_one_map_parallel_indexed(&keys, log_space, |_| alpha)
}

/// Generalization of [`gamma_one_map_parallel`] where element `i` draws with shape
/// `alpha_of(i)` — for `random_dirichlet`, whose element `i*k+j` uses `alpha[j]`. Output
/// `i` depends ONLY on `keys[i]` and `alpha_of(i)` (no shared state), so any thread
/// partition is BIT-IDENTICAL to the serial index map (guarded by
/// `gamma_map_parallel_matches_serial_bits` for the constant case and
/// `dirichlet_threaded_matches_serial_bits` for the indexed case). `FJ_GAMMA_SERIAL`
/// forces the serial reference for a same-invocation A/B.
fn gamma_one_map_parallel_indexed<F>(keys: &[PRNGKey], log_space: bool, alpha_of: F) -> Vec<f64>
where
    F: Fn(usize) -> f64 + Sync,
{
    let count = keys.len();
    let threads =
        if count >= GAMMA_PARALLEL_MIN_ELEMS && std::env::var_os("FJ_GAMMA_SERIAL").is_none() {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
                .min(count)
        } else {
            1
        };
    if threads <= 1 {
        return (0..count)
            .map(|i| gamma_one(keys[i], alpha_of(i), log_space))
            .collect();
    }
    let mut out = vec![0.0_f64; count];
    let chunk = count.div_ceil(threads);
    std::thread::scope(|scope| {
        let alpha_of = &alpha_of;
        let mut rest: &mut [f64] = out.as_mut_slice();
        let mut start = 0usize;
        while start < count {
            let len = chunk.min(count - start);
            let (blk, tail) = rest.split_at_mut(len);
            rest = tail;
            let s = start;
            scope.spawn(move || {
                for (j, slot) in blk.iter_mut().enumerate() {
                    let idx = s + j;
                    *slot = gamma_one(keys[idx], alpha_of(idx), log_space);
                }
            });
            start += len;
        }
    });
    out
}

/// Faithful port of JAX's `_gamma_one` (jax/_src/random.py): Marsaglia-Tsang with
/// per-iteration key evolution. For `alpha < 1` it boosts to `alpha+1` and scales by
/// `Uniform^(1/alpha)`. `log_space` returns `log(sample)` via `-Exponential` boost (used
/// by `random_beta`/loggamma) — matching JAX's numerically-stable log path. The
/// accept/reject math is the (already-correct) squeeze test; the fidelity fix is the JAX
/// key schedule — each element/iteration draws from its own split subkey, so results
/// follow JAX element-for-element (f32 tolerance; fj arith is f64 on f32-matching draws).
fn gamma_one(key: PRNGKey, alpha_orig: f64, log_space: bool) -> f64 {
    let boost = alpha_orig < 1.0;
    let alpha = if boost { alpha_orig + 1.0 } else { alpha_orig };
    let d = alpha - 1.0 / 3.0;
    let c = (1.0 / 3.0) / d.sqrt();

    let (mut k, subkey) = random_split(key);

    // while_loop over state (key, X, V, U); the condition is JAX's REJECT test, so the
    // initial (X=0, V=1, U=2) enters the loop, and it exits on the first ACCEPT.
    let (mut xx, mut vv, mut uu) = (0.0f64, 1.0f64, 2.0f64);
    while uu >= 1.0 - 0.0331 * xx * xx && uu.ln() >= 0.5 * xx + d * (1.0 - vv + vv.ln()) {
        let s3 = scalar_split3(k);
        k = s3[0];
        // inner loop: redraw the normal until v = 1 + x*c > 0.
        let mut ik = s3[1];
        let mut x = 0.0f64;
        let mut v = -1.0f64;
        while v <= 0.0 {
            let (nik, subk) = random_split(ik);
            ik = nik;
            x = scalar_normal(subk);
            v = 1.0 + x * c;
        }
        xx = x * x;
        vv = v * v * v;
        uu = scalar_uniform(s3[2], 0.0, 1.0);
    }

    if log_space {
        // log_samples = -Exponential(subkey) = log1p(-uniform(subkey)); boost only for a<1.
        let log_samples = (-scalar_uniform(subkey, 0.0, 1.0)).ln_1p();
        let log_boost = if !boost || log_samples == 0.0 {
            0.0
        } else {
            log_samples / alpha_orig
        };
        d.ln() + vv.ln() + log_boost
    } else if boost {
        let s = 1.0 - scalar_uniform(subkey, 0.0, 1.0);
        d * vv * s.powf(1.0 / alpha_orig)
    } else {
        d * vv
    }
}

/// Generate beta-distributed samples.
///
/// Matches `jax.random.beta(key, a, b, shape)`.
/// Uses the relationship: Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
#[must_use]
pub fn random_beta(key: PRNGKey, count: usize, a: f64, b: f64) -> Vec<f64> {
    // JAX `_beta`: split the key, draw LOG-gamma for each parameter (per-element keys),
    // then combine gamma_a/(gamma_a+gamma_b) via the numerically-stable log-max trick
    // (exp(log_g - log_max)). Using loggamma (not plain gamma) + log-max is what makes
    // this match JAX element-for-element, incl. small a/b where plain gamma underflows.
    let (k1, k2) = random_split(key);
    let log_ga: Vec<f64> = gamma_one_map_parallel(random_split_n(k1, count), a, true);
    let log_gb: Vec<f64> = gamma_one_map_parallel(random_split_n(k2, count), b, true);
    (0..count)
        .map(|i| {
            let log_max = log_ga[i].max(log_gb[i]);
            let ga_s = (log_ga[i] - log_max).exp();
            let gb_s = (log_gb[i] - log_max).exp();
            ga_s / (ga_s + gb_s)
        })
        .collect()
}

/// Generate Poisson-distributed samples.
///
/// Matches `jax.random.poisson(key, lam, shape)`.
/// Matches `jax.random.poisson`: Knuth's algorithm for `lam < 10`, Hörmann's
/// transformed-rejection (PTRS) for `lam >= 10` (jax `_poisson`). Both paths are
/// VECTORIZED over the whole array sharing one evolving key (JAX draws a count-sized
/// `uniform(subkey, shape)` block per iteration — NOT a per-element key split, unlike
/// gamma). Replaces the prior non-JAX shared-pool Knuth + normal-approximation.
#[must_use]
pub fn random_poisson(key: PRNGKey, count: usize, lam: f64) -> Vec<u64> {
    if lam <= 0.0 {
        return vec![0; count];
    }
    if lam < 10.0 {
        poisson_knuth(key, count, lam)
    } else {
        poisson_rejection(key, count, lam)
    }
}

/// Vectorized Knuth (jax `_poisson_knuth`), log-space. Each iteration splits the key,
/// draws a count-sized uniform block, and for every element still accumulating
/// (`log_prod > -lam`) increments `k`; `log_prod += log(u)` unconditionally. Result is
/// `k - 1`. Loops until no element is still accumulating.
fn poisson_knuth(key: PRNGKey, count: usize, lam: f64) -> Vec<u64> {
    let mut rng = key;
    let mut k = vec![0i64; count];
    let mut log_prod = vec![0.0f64; count];
    // JAX max_iters = int-max; small lam converges in a handful of iterations. This cap
    // only bounds the (astronomically unlikely) tail and never triggers for lam < 10.
    for _ in 0..10_000 {
        if !log_prod.iter().any(|&lp| lp > -lam) {
            break;
        }
        let s = random_split_n(rng, 2);
        rng = s[0];
        let u = random_uniform(s[1], count, 0.0, 1.0);
        for i in 0..count {
            if log_prod[i] > -lam {
                k[i] += 1;
            }
            log_prod[i] += u[i].ln();
        }
    }
    k.iter().map(|&ki| (ki - 1).max(0) as u64).collect()
}

/// Vectorized Hörmann transformed-rejection (jax `_poisson_rejection`). Each iteration
/// splits the key 3-way, draws two count-sized uniform blocks, and accepts per element
/// via the squeeze/log tests; loops until every element is accepted.
fn poisson_rejection(key: PRNGKey, count: usize, lam: f64) -> Vec<u64> {
    let log_lam = lam.ln();
    let b = 0.931 + 2.53 * lam.sqrt();
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let v_r = 0.9277 - 3.6224 / (b - 2.0);

    let mut rng = key;
    let mut k_out = vec![-1.0f64; count];
    let mut accepted = vec![false; count];
    for _ in 0..10_000 {
        if accepted.iter().all(|&x| x) {
            break;
        }
        let s = random_split_n(rng, 3);
        rng = s[0];
        let u_blk = random_uniform(s[1], count, 0.0, 1.0);
        let v_blk = random_uniform(s[2], count, 0.0, 1.0);
        for i in 0..count {
            let u = u_blk[i] - 0.5;
            let v = v_blk[i];
            let u_shifted = 0.5 - u.abs();
            let k = ((2.0 * a / u_shifted + b) * u + lam + 0.43).floor();
            let s_val = (v * inv_alpha / (a / (u_shifted * u_shifted) + b)).ln();
            let t = -lam + k * log_lam - crate::arithmetic::lgamma_approx(k + 1.0);
            let accept1 = u_shifted >= 0.07 && v <= v_r;
            let reject = k < 0.0 || (u_shifted < 0.013 && v > u_shifted);
            let accept2 = s_val <= t;
            if accept1 || (!reject && accept2) {
                k_out[i] = k;
                accepted[i] = true;
            }
        }
    }
    k_out.iter().map(|&k| k.max(0.0) as u64).collect()
}

/// Generate truncated normal samples in (lower, upper).
///
/// Matches `jax.random.truncated_normal(key, lower, upper, shape)`: the
/// inverse-CDF transform `√2·erf_inv(uniform(erf(lower/√2), erf(upper/√2)))`,
/// then clamped into the OPEN interval (random.py `_truncated_normal`). This
/// consumes exactly ONE uniform per sample over the same threefry stream as
/// `random_normal`, so it tracks JAX's RNG sequence.
///
/// The previous implementation used REJECTION sampling on `random_normal`,
/// which (a) consumed a different number of draws than JAX (count*20, with a
/// data-dependent count), so it never matched JAX's sequence, and (b)
/// degenerated to a constant midpoint fill for any narrow or off-center range
/// (e.g. [2.0, 2.5], where almost every standard-normal draw is rejected) —
/// producing a near-degenerate distribution instead of a truncated normal.
#[must_use]
pub fn random_truncated_normal(key: PRNGKey, count: usize, lower: f64, upper: f64) -> Vec<f64> {
    if lower >= upper {
        return vec![f64::NAN; count];
    }

    let sqrt2 = std::f64::consts::SQRT_2;
    let a = crate::arithmetic::erf_approx(lower / sqrt2);
    let b = crate::arithmetic::erf_approx(upper / sqrt2);

    // JAX clamps into the OPEN interval so rounding (or drawing u == a) can't
    // land exactly on a bound: clip(out, nextafter(lower, +inf), nextafter(upper, -inf)).
    let lo_clamp = next_after_f64(lower, f64::INFINITY);
    let hi_clamp = next_after_f64(upper, f64::NEG_INFINITY);

    map_uniforms_parallel(random_uniform(key, count, a, b), move |u| {
        let out = sqrt2 * crate::arithmetic::erf_inv_approx(u);
        // numpy/jnp clip semantics (max then min — never panics).
        out.max(lo_clamp).min(hi_clamp)
    })
}

/// `nextafter(x, toward)` for f64 — the adjacent representable value stepping
/// from `x` toward `toward`. Mirrors `lax.nextafter`; used to clamp
/// truncated_normal into its open interval exactly as JAX does.
fn next_after_f64(x: f64, toward: f64) -> f64 {
    if x.is_nan() || toward.is_nan() {
        return f64::NAN;
    }
    if x == toward {
        return toward;
    }
    if x == 0.0 {
        let tiny = f64::from_bits(1);
        return if toward > 0.0 { tiny } else { -tiny };
    }
    let bits = x.to_bits();
    // Moving away from zero (increasing magnitude) increments the bit pattern;
    // moving toward zero decrements it.
    let next = if (toward > x) == (x > 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next)
}

/// Generate samples from a Cauchy distribution.
///
/// Matches `jax.random.cauchy(key, shape)`.
///
/// JAX `_cauchy`: `tan(pi * (U - 0.5))` over `U ~ uniform(minval=finfo.eps,
/// maxval=1)` — the eps lower bound (f32 eps in JAX's default f32 mode) keeps U
/// off 0 so `tan` never hits the asymptote at U=0. The previous code sampled
/// `uniform(0, 1)`, a different (and slightly wider) domain than JAX.
#[must_use]
pub fn random_cauchy(key: PRNGKey, count: usize) -> Vec<f64> {
    let eps = f64::from(f32::EPSILON);
    map_uniforms_parallel(random_uniform(key, count, eps, 1.0), |u| {
        (std::f64::consts::PI * (u - 0.5)).tan()
    })
}

/// Generate samples from a Pareto distribution.
///
/// Matches `jax.random.pareto(key, b, shape)` where b is the shape parameter.
///
/// JAX `_pareto`: `exp(exponential(key, shape) / b)`. The previous
/// `U^(-1/b)` = `exp(-ln(U)/b)` sampled the OPPOSITE tail of JAX's
/// `exp(-log1p(-U)/b)` = `(1-U)^(-1/b)` (statistically valid but a different
/// RNG sequence — the same class as the exponential/laplace parity fixes), and
/// its `U.max(1e-30)` clamp was an ad-hoc guard JAX does not have. Reusing
/// `random_exponential` (already JAX's `-log1p(-U)`, finite at U=0) is exact.
#[must_use]
pub fn random_pareto(key: PRNGKey, count: usize, b: f64) -> Vec<f64> {
    random_exponential(key, count, 1.0)
        .into_iter()
        .map(|e| (e / b).exp())
        .collect()
}

/// Generate Weibull-distributed random samples.
///
/// Matches `jax.random.weibull_min(key, scale, concentration, shape)`.
///
/// JAX `_weibull_min`: `power(-log1p(-U), 1/concentration) * scale` over
/// `U ~ uniform(0, 1)`. `-log1p(-U)` is exactly an Exponential(1) sample, so we
/// reuse `random_exponential` (same uniform domain) for an exact match. The
/// previous `(-ln(U))^(1/c)` used the opposite tail and a `U.max(1e-30)` clamp
/// JAX lacks (log1p is finite at U=0, so no clamp is needed).
pub fn random_weibull(key: PRNGKey, count: usize, scale: f64, concentration: f64) -> Vec<f64> {
    random_exponential(key, count, 1.0)
        .into_iter()
        .map(|e| scale * e.powf(1.0 / concentration))
        .collect()
}

/// Generate Rayleigh-distributed random samples.
///
/// Matches `jax.random.rayleigh(key, scale, shape)`.
/// Uses inverse transform: X = scale * sqrt(-2 * ln(U))
pub fn random_rayleigh(key: PRNGKey, count: usize, scale: f64) -> Vec<f64> {
    let uniforms = random_uniform(key, count, 0.0, 1.0);
    map_uniforms_parallel(uniforms, move |u| {
        let clamped = u.max(1e-30);
        scale * (-2.0 * clamped.ln()).sqrt()
    })
}

/// Generate chi-squared distributed random samples.
///
/// Matches `jax.random.chisquare(key, df, shape)`.
/// Uses the fact that chi2(df) = 2 * Gamma(df/2, 1).
pub fn random_chi2(key: PRNGKey, count: usize, df: f64) -> Vec<f64> {
    let gamma_samples = random_gamma(key, count, df / 2.0);
    gamma_samples.into_iter().map(|g| 2.0 * g).collect()
}

/// Sample from the Maxwell-Boltzmann speed distribution.
///
/// Matches JAX's `_maxwell` verbatim: draw `normal(key, shape + (3,))` and take
/// the Euclidean norm over the trailing length-3 axis — i.e. each output is the
/// magnitude of a 3-D standard-normal vector, `sqrt(n0² + n1² + n2²)`. Because
/// `random_normal` reproduces JAX's flat normal stream, drawing `3·count` normals
/// and grouping them in row-major triples matches JAX's `(count, 3)` layout
/// exactly. `jnp.linalg.norm(..., axis=-1)` is the plain `sqrt(Σ x²)` for a real
/// vector (no hypot rescaling), so the per-sample arithmetic matches too.
pub fn random_maxwell(key: PRNGKey, count: usize) -> Vec<f64> {
    let normal_count = checked_random_output_len("random_maxwell normal draw", count, 3);
    let normals = random_normal(key, normal_count);
    (0..count)
        .map(|i| {
            let a = normals[3 * i];
            let b = normals[3 * i + 1];
            let c = normals[3 * i + 2];
            (a * a + b * b + c * c).sqrt()
        })
        .collect()
}

#[inline]
fn checked_random_output_len(context: &str, count: usize, factor: usize) -> usize {
    let Some(len) = count.checked_mul(factor) else {
        panic!("{context} output size overflow: count={count} factor={factor}");
    };
    len
}

/// Generate Student's t-distributed random samples.
///
/// Matches `jax.random.t(key, df, shape)`.
/// Uses the representation: t = Z / sqrt(chi2 / df) where Z ~ N(0,1), chi2 ~ chi2(df).
pub fn random_t(key: PRNGKey, count: usize, df: f64) -> Vec<f64> {
    let (key1, key2) = random_split(key);
    let normals = random_normal(key1, count);
    let chi2_samples = random_chi2(key2, count, df);

    normals
        .into_iter()
        .zip(chi2_samples)
        .map(|(z, c)| z / (c / df).sqrt())
        .collect()
}

/// Generate Dirichlet-distributed random samples.
///
/// Matches `jax.random.dirichlet(key, alpha, shape)`.
/// Uses the property that if X_i ~ Gamma(alpha_i, 1), then (X_1/S, ..., X_k/S) ~ Dirichlet(alpha)
/// where S = sum(X_i).
///
/// Returns a vector of length `count * alpha.len()`, representing `count` samples from Dirichlet(alpha).
pub fn random_dirichlet(key: PRNGKey, count: usize, alpha: &[f64]) -> Vec<f64> {
    let k = alpha.len();
    if k == 0 || count == 0 {
        return Vec::new();
    }

    let output_len = checked_random_output_len("random_dirichlet", count, k);
    // JAX `_dirichlet`: draw LOG-gamma over the full (count, K) shape from ONE key
    // (`loggamma(key, alpha, (count,K))` splits into count*K per-element keys, element
    // (i,j) uses alpha[j]), then `softmax(log_gamma, axis=-1)` per row — the stable
    // log-space normalize. Replaces the prior plain-gamma + sequential-split + direct
    // ratio, which matched neither the algorithm nor JAX's key schedule.
    let keys = random_split_n(key, output_len);
    // The per-element loggamma draw (element i*k+j uses alpha[j]) is the cost; each is an
    // independent `gamma_one` rejection loop, so thread it — bit-identical because idx%k == j
    // for idx = i*k+j (guard `dirichlet_threaded_matches_serial_bits`). The per-row softmax
    // below is cheap and stays serial.
    let log_g_all = gamma_one_map_parallel_indexed(&keys, true, |idx| alpha[idx % k]);
    let mut result = Vec::with_capacity(output_len);
    for i in 0..count {
        let base = i * k;
        let row = &log_g_all[base..base + k];
        // softmax over the row: exp(x - max) / sum(exp(x - max)).
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for e in exps {
            result.push(e / sum);
        }
    }
    result
}

/// Generate geometrically-distributed random samples.
///
/// Matches `jax.random.geometric(key, p, shape)`.
///
/// JAX `_geometric`: `floor(log(U) / log1p(-p)) + 1`. The previous code used
/// `ceil(ln(U) / ln(1-p))`, which equals `floor(..)+1` for non-integer ratios
/// but diverges by one whenever the ratio lands exactly on an integer. `ln(1-p)`
/// equals JAX's `log1p(-p)`. The `U.max(1e-30)` guard only affects the
/// measure-zero `U==0` draw (the smallest f32 uniform is ~2^-23), keeping the
/// `u64` cast finite; JAX would overflow there.
pub fn random_geometric(key: PRNGKey, count: usize, p: f64) -> Vec<u64> {
    if p <= 0.0 || p >= 1.0 {
        return vec![1; count];
    }
    let uniforms = random_uniform(key, count, 0.0, 1.0);
    // JAX uses log1p(-p); (1.0 - p).ln() loses up to ~|p| relative precision for
    // small p (1-p rounds near 1.0 before ln), while (-p).ln_1p() is exact.
    let log_1_minus_p = (-p).ln_1p();
    uniforms
        .into_iter()
        .map(|u| {
            let clamped = u.max(1e-30);
            ((clamped.ln() / log_1_minus_p).floor() + 1.0).max(1.0) as u64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    // Test-only `[(&str, fn(f64) -> f64); N]` transform tables.
    #![allow(clippy::type_complexity)]
    use super::*;

    // RNG vs JAX (16M, measured JAX random.uniform 112.0ms / normal 109.7ms — XLA CPU threefry is slow).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_random_uniform_vs_jax() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let key = random_key(0);
        let f = || {
            std::hint::black_box(random_uniform(key, n, 0.0, 1.0));
        };
        f();
        let mut b = f64::MAX;
        for _ in 0..6 {
            let s = Instant::now();
            f();
            b = b.min(s.elapsed().as_secs_f64());
        }
        println!(
            "fj-lax random_uniform f64 16M: {:.3}ms | JAX(f32)=112.0ms",
            b * 1e3
        );
    }

    // Threaded gamma/beta rejection sampler must be BIT-for-bit identical to the serial map.
    // Each element draws from its own split subkey, so any thread partition preserves per-element
    // results; RNG parity with JAX is absolute, so any divergence is a hard failure.
    #[test]
    fn gamma_map_parallel_matches_serial_bits() {
        let count = GAMMA_PARALLEL_MIN_ELEMS + 777; // above threshold, not a multiple of thread count
        for &alpha in &[0.4_f64, 1.0, 2.5, 7.3] {
            for &log_space in &[false, true] {
                let keys = random_split_n(random_key((alpha * 10.0) as u64 + 1), count);
                let serial: Vec<f64> = keys
                    .iter()
                    .map(|&k| gamma_one(k, alpha, log_space))
                    .collect();
                let parallel = gamma_one_map_parallel(keys, alpha, log_space);
                assert_eq!(parallel.len(), count);
                for (i, (p, s)) in parallel.iter().zip(&serial).enumerate() {
                    assert_eq!(
                        p.to_bits(),
                        s.to_bits(),
                        "gamma_one mismatch at {i} alpha={alpha} log_space={log_space}"
                    );
                }
            }
        }
    }

    // Rejection samplers vs JAX 0.10.2 CPU (jaxvenv, jit'd, min-of-7, 1M): gamma(2.5)=924ms,
    // gamma(0.4)=1226ms, beta(2,3)=1341ms — XLA-CPU cannot vectorize NOR thread the per-element
    // Marsaglia-Tsang accept/reject loop. fj threads it (bit-identical). Same-invocation serial A/B.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_random_gamma_beta_vs_jax() {
        use std::time::Instant;
        let n = 1_usize << 20;
        let key = random_key(7);
        let best = |f: &dyn Fn()| -> f64 {
            f();
            let mut b = f64::MAX;
            for _ in 0..7 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            b
        };
        // Serial reference: identical work to random_gamma minus the thread fan-out.
        let g25_ser = best(&|| {
            let out: Vec<f64> = random_split_n(key, n)
                .into_iter()
                .map(|k| gamma_one(k, 2.5, false))
                .collect();
            std::hint::black_box(out);
        });
        let g25_thr = best(&|| {
            std::hint::black_box(random_gamma(key, n, 2.5));
        });
        let g04_thr = best(&|| {
            std::hint::black_box(random_gamma(key, n, 0.4));
        });
        let beta_thr = best(&|| {
            std::hint::black_box(random_beta(key, n, 2.0, 3.0));
        });
        println!(
            "[rejection samplers 1M vs JAX] gamma(2.5): serial={:.1}ms threaded={:.1}ms ({:.2}x self | JAX 924, {:.1}x) | gamma(0.4) threaded={:.1}ms (JAX 1226, {:.1}x) | beta(2,3) threaded={:.1}ms (JAX 1341, {:.1}x)",
            g25_ser * 1e3,
            g25_thr * 1e3,
            g25_ser / g25_thr,
            924.0 / (g25_thr * 1e3),
            g04_thr * 1e3,
            1226.0 / (g04_thr * 1e3),
            beta_thr * 1e3,
            1341.0 / (beta_thr * 1e3),
        );
    }

    // Threaded dirichlet loggamma draw must be BIT-for-bit identical to the serial index map.
    // Element i*k+j draws gamma_one(keys[i*k+j], alpha[j]) with j == (i*k+j)%k, from its own subkey.
    #[test]
    fn dirichlet_threaded_matches_serial_bits() {
        let k = 4usize;
        let count = (GAMMA_PARALLEL_MIN_ELEMS / k) + 123; // output_len above threshold, non-aligned
        let output_len = count * k;
        let alpha = [0.3_f64, 1.0, 2.5, 5.0];
        let keys = random_split_n(random_key(99), output_len);
        let serial: Vec<f64> = (0..output_len)
            .map(|i| gamma_one(keys[i], alpha[i % k], true))
            .collect();
        let parallel = gamma_one_map_parallel_indexed(&keys, true, |idx| alpha[idx % k]);
        assert_eq!(parallel.len(), output_len);
        for (i, (p, s)) in parallel.iter().zip(&serial).enumerate() {
            assert_eq!(
                p.to_bits(),
                s.to_bits(),
                "dirichlet loggamma mismatch at {i}"
            );
        }
    }

    // Dirichlet vs JAX 0.10.2 CPU (jaxvenv, jit'd, min-of-7): count=262144, k=4 (1M loggamma draws)
    // = 920ms. Per-element loggamma is an independent Marsaglia-Tsang rejection loop XLA can't
    // vectorize/thread; fj threads it. Same-invocation serial A/B (serial = split + serial gamma map).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_random_dirichlet_vs_jax() {
        use std::time::Instant;
        let count = 1_usize << 18; // 262144 samples
        let k = 4usize;
        let output_len = count * k;
        let alpha = [0.3_f64, 1.0, 2.5, 5.0];
        let key = random_key(7);
        let best = |f: &dyn Fn()| -> f64 {
            f();
            let mut b = f64::MAX;
            for _ in 0..7 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            b
        };
        // Serial reference: the dominant loggamma draw without the thread fan-out.
        let ser = best(&|| {
            let keys = random_split_n(key, output_len);
            let out: Vec<f64> = (0..output_len)
                .map(|i| gamma_one(keys[i], alpha[i % k], true))
                .collect();
            std::hint::black_box(out);
        });
        let thr = best(&|| {
            std::hint::black_box(random_dirichlet(key, count, &alpha));
        });
        println!(
            "[dirichlet 262144x4 (1M draws) vs JAX] serial-gamma={:.1}ms threaded-full={:.1}ms ({:.2}x self | JAX 920, {:.1}x)",
            ser * 1e3,
            thr * 1e3,
            ser / thr,
            920.0 / (thr * 1e3),
        );
    }

    #[test]
    fn test_threefry_deterministic() {
        let key = [0u32, 0u32];
        let counter = [0u32, 0u32];
        let a = threefry2x32(key, counter);
        let b = threefry2x32(key, counter);
        assert_eq!(a, b, "ThreeFry must be deterministic");
    }

    #[test]
    fn simd_generate_u32_bits_matches_scalar() {
        // The portable-SIMD generate_u32_bits must be BIT-for-bit identical to the
        // scalar per-counter threefry2x32([0, i]) ^ — across keys and across counts
        // that span full SIMD chunks AND every tail length (0..LANES). RNG parity
        // with JAX is absolute, so any lane/tail divergence is a hard failure.
        for key in [
            PRNGKey([0, 0]),
            PRNGKey([1, 0]),
            PRNGKey([0x9E37_79B9, 0x1234_5678]),
            PRNGKey([u32::MAX, u32::MAX]),
        ] {
            // 0..=20 covers chunk=0 (tail-only) through 2.5 chunks at LANES=8.
            for count in 0..=20usize {
                let simd = generate_u32_bits(key, count);
                let scalar: Vec<u32> = (0..count)
                    .map(|i| {
                        let [a, b] = threefry2x32(key.0, [0, i as u32]);
                        a ^ b
                    })
                    .collect();
                assert_eq!(simd, scalar, "key={key:?} count={count}");
            }
            // A larger count to exercise many full chunks.
            let count = 10_007;
            let simd = generate_u32_bits(key, count);
            let scalar: Vec<u32> = (0..count)
                .map(|i| {
                    let [a, b] = threefry2x32(key.0, [0, i as u32]);
                    a ^ b
                })
                .collect();
            assert_eq!(simd, scalar, "key={key:?} large count={count}");
        }
    }

    #[test]
    fn simd_uniform_matches_scalar_bits() {
        // The fused SIMD-f64 random_uniform must be BIT-for-bit identical to the
        // original f32-bitcast scalar formula (random_uniform_scalar) — the f64
        // mantissa shortcut (m·2^-23) only holds if it reproduces every bit. Cover
        // keys, ranges, and tail lengths; JAX RNG parity is absolute.
        for key in [
            PRNGKey([0, 0]),
            PRNGKey([7, 0]),
            PRNGKey([0x9E37_79B9, 0x1234_5678]),
            PRNGKey([u32::MAX, u32::MAX]),
        ] {
            for &(lo, hi) in &[
                (0.0_f64, 1.0_f64),
                (-1.0, 1.0),
                (-3.5, 7.25),
                (100.0, 100.5),
            ] {
                for count in 0..=19usize {
                    let simd = random_uniform(key, count, lo, hi);
                    let scalar = random_uniform_scalar(key, count, lo, hi);
                    assert_eq!(simd.len(), scalar.len());
                    for (s, r) in simd.iter().zip(scalar.iter()) {
                        assert_eq!(s.to_bits(), r.to_bits(), "key={key:?} range=({lo},{hi})");
                    }
                }
                let count = 9_973;
                let simd = random_uniform(key, count, lo, hi);
                let scalar = random_uniform_scalar(key, count, lo, hi);
                for (s, r) in simd.iter().zip(scalar.iter()) {
                    assert_eq!(
                        s.to_bits(),
                        r.to_bits(),
                        "key={key:?} big range=({lo},{hi})"
                    );
                }
            }
        }
    }

    #[test]
    fn threaded_uniform_matches_serial_bits() {
        // The threaded random_uniform must be BIT-for-bit identical to the
        // single-threaded SIMD baseline for ANY partition. ThreeFry is
        // counter-based, so element i always uses absolute counter i regardless of
        // which thread (or SIMD-vs-tail boundary) computes it. Cover counts well
        // above the parallel threshold, including non-multiple-of-LANES totals that
        // force ragged thread/tail boundaries. JAX RNG parity is absolute.
        for key in [
            PRNGKey([0, 0]),
            PRNGKey([7, 0]),
            PRNGKey([0x9E37_79B9, 0x1234_5678]),
            PRNGKey([u32::MAX, u32::MAX]),
        ] {
            for &(lo, hi) in &[(0.0_f64, 1.0_f64), (-1.0, 1.0), (-3.5, 7.25)] {
                for count in [
                    2 * RANDOM_UNIFORM_MIN_ELEMS_PER_THREAD,
                    2 * RANDOM_UNIFORM_MIN_ELEMS_PER_THREAD + 1,
                    2 * RANDOM_UNIFORM_MIN_ELEMS_PER_THREAD + 7,
                    1_000_003,
                    4_194_304,
                    5_000_007,
                ] {
                    let threaded = random_uniform(key, count, lo, hi);
                    let serial = random_uniform_serial_simd(key, count, lo, hi);
                    assert_eq!(threaded.len(), serial.len());
                    for (idx, (t, s)) in threaded.iter().zip(serial.iter()).enumerate() {
                        assert_eq!(
                            t.to_bits(),
                            s.to_bits(),
                            "key={key:?} range=({lo},{hi}) count={count} idx={idx}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn threaded_u32_bits_matches_serial() {
        // The threaded generate_u32_bits must be word-for-word identical to the
        // single-threaded SIMD baseline for ANY partition — ThreeFry is
        // counter-based, so word i always uses absolute counter i regardless of
        // which thread (or SIMD-vs-tail boundary) computes it. Cover counts above
        // the parallel threshold, including non-multiple-of-LANES totals that force
        // ragged thread/tail boundaries. JAX RNG parity is absolute.
        for key in [
            PRNGKey([0, 0]),
            PRNGKey([7, 0]),
            PRNGKey([0x9E37_79B9, 0x1234_5678]),
            PRNGKey([u32::MAX, u32::MAX]),
        ] {
            for count in [
                2 * RANDOM_BITS_MIN_ELEMS_PER_THREAD,
                2 * RANDOM_BITS_MIN_ELEMS_PER_THREAD + 1,
                2 * RANDOM_BITS_MIN_ELEMS_PER_THREAD + 7,
                1_000_003,
                4_194_304,
                5_000_007,
            ] {
                let threaded = generate_u32_bits(key, count);
                let serial = generate_u32_bits_serial(key, count);
                assert_eq!(threaded.len(), serial.len());
                assert!(
                    threaded == serial,
                    "key={key:?} count={count}: threaded u32 bits diverged from serial"
                );
            }
        }
    }

    #[test]
    fn map_uniforms_parallel_matches_serial_bits() {
        // The threaded distribution-transform map must be BIT-for-bit identical to
        // the serial `uniforms.into_iter().map(f)` for ANY partition — the map is
        // elementwise (output i depends only on uniforms[i]). Cover counts above
        // the parallel threshold, ragged thread boundaries, and the actual
        // transcendental transforms the distributions use.
        let key = random_key(0x0BAD_F00D_DEAD_BEEF);
        let transforms: [(&str, fn(f64) -> f64); 4] = [
            ("exponential", |u| -(-u).ln_1p()),
            ("gumbel", |u| -(-u.max(f64::MIN_POSITIVE).ln()).ln()),
            ("cauchy", |u| (std::f64::consts::PI * (u - 0.5)).tan()),
            ("rayleigh", |u| (-2.0 * u.max(1e-30).ln()).sqrt()),
        ];
        for count in [
            2 * DIST_TRANSFORM_MIN_ELEMS_PER_THREAD,
            2 * DIST_TRANSFORM_MIN_ELEMS_PER_THREAD + 1,
            2 * DIST_TRANSFORM_MIN_ELEMS_PER_THREAD + 13,
            500_003,
            1_000_000,
        ] {
            let uniforms = random_uniform(key, count, 0.0, 1.0);
            for (name, f) in transforms {
                let serial: Vec<f64> = uniforms.iter().copied().map(f).collect();
                let parallel = map_uniforms_parallel(uniforms.clone(), f);
                assert_eq!(parallel.len(), serial.len());
                for (idx, (p, s)) in parallel.iter().zip(serial.iter()).enumerate() {
                    assert_eq!(
                        p.to_bits(),
                        s.to_bits(),
                        "transform={name} count={count} idx={idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn fill_uniform_simd_subrange_matches_whole() {
        // fill_uniform_simd over an arbitrary sub-range [start, start+len) must
        // equal the corresponding window of the whole-array serial stream — the
        // invariant the threaded path relies on for any chunk boundary.
        let key = PRNGKey([0x1357_9BDF, 0x2468_ACE0]);
        let (lo, scale) = (-2.0_f64, 5.0_f64);
        let whole = random_uniform_serial_simd(key, 50_000, lo, lo + scale);
        for &(start, len) in &[
            (0usize, 13usize),
            (1, 8),
            (7, 9),
            (12_345, 6_789),
            (40_000, 9_999),
        ] {
            let mut block = vec![0.0_f64; len];
            fill_uniform_simd(&mut block, start, key, lo, scale);
            for (j, b) in block.iter().enumerate() {
                assert_eq!(
                    b.to_bits(),
                    whole[start + j].to_bits(),
                    "start={start} len={len} j={j}"
                );
            }
        }
    }

    #[test]
    fn threaded_random_normal_transform_matches_serial() {
        let key = random_key(0xCAFE_BABE_F00D_F00D);
        let lo = f64::from(f32::from_bits((-1.0_f32).to_bits() - 1));
        let uniforms = random_uniform(key, 70_003, lo, 1.0);
        let serial = random_normal_from_uniforms_serial(&uniforms);
        for threads in [2usize, 3, 7, 16, 64] {
            let threaded = random_normal_from_uniforms_with_threads(&uniforms, threads);
            assert_eq!(threaded, serial, "threads={threads}");
        }
    }

    #[test]
    fn random_normal_threaded_golden_sha256() {
        let mut streams = Vec::new();
        for seed in [0_u64, 1, 0x1234_5678_9ABC_DEF0, u64::MAX] {
            let key = random_key(seed);
            for count in [0_usize, 1, 7, 8, 9, 64, 257, 4096] {
                streams.push((seed, count, random_normal(key, count)));
            }
        }
        let digest = fj_test_utils::fixture_id_from_json(&streams).expect("normal digest");
        assert_eq!(
            digest, "982d444c8f93dd7331fff1e2141e40ee967f6698a7a12cbeef104d38b6c6c29b",
            "random_normal golden SHA-256 changed"
        );
    }

    #[test]
    fn threefry_generated_bits_golden_sha256() {
        let mut streams = Vec::new();
        for key in [
            PRNGKey([0, 0]),
            PRNGKey([1, 0]),
            PRNGKey([0x9E37_79B9, 0x1234_5678]),
            PRNGKey([u32::MAX, u32::MAX]),
        ] {
            for count in [0_usize, 1, 7, 8, 9, 64, 257, 4096] {
                streams.push((key.0, count, generate_u32_bits(key, count)));
            }
        }
        let digest = fj_test_utils::fixture_id_from_json(&streams).expect("rng digest");
        assert_eq!(
            digest, "e96748ed0520e1648d412ffe3ccddbf966b23fd0828824e0d924e2fa7108c8f5",
            "ThreeFry generated-bits golden SHA-256 changed"
        );
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
    fn test_categorical_basic() -> Result<(), CategoricalError> {
        let key = random_key(42);
        let logits = [0.0, 0.0, 0.0]; // uniform over 3 categories
        let samples = random_categorical(key, &logits, 10_000)?;
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
        Ok(())
    }

    #[test]
    fn test_categorical_skewed() -> Result<(), CategoricalError> {
        let key = random_key(42);
        // log(0.9) ≈ -0.105, log(0.05) ≈ -2.996, log(0.05) ≈ -2.996
        let logits = [10.0, 0.0, 0.0]; // heavily favor first category
        let samples = random_categorical(key, &logits, 1000)?;
        let count_0 = samples.iter().filter(|&&i| i == 0).count();
        assert!(
            count_0 > 900,
            "heavily-weighted category should dominate, got {count_0}/1000"
        );
        Ok(())
    }

    #[test]
    fn test_categorical_rejects_empty_logits() -> Result<(), String> {
        let Err(err) = random_categorical(random_key(42), &[], 3) else {
            return Err("empty logits should be rejected".to_owned());
        };
        assert_eq!(err, CategoricalError::EmptyLogits);
        assert_eq!(
            err.to_string(),
            "categorical sampling requires at least one logit"
        );
        Ok(())
    }

    #[test]
    fn test_categorical_rejects_sample_count_overflow() -> Result<(), String> {
        let Err(err) = random_categorical(random_key(42), &[0.0, 1.0], usize::MAX) else {
            return Err("overflowed sample count should be rejected".to_owned());
        };
        assert_eq!(
            err,
            CategoricalError::SampleCountOverflow {
                num_samples: usize::MAX,
                num_categories: 2
            }
        );
        Ok(())
    }

    // === Extended RNG tests (frankenjax-tr5) ===

    #[test]
    fn test_multi_level_split_independence() {
        // Split a key multiple times and verify all descendants are unique
        let root = random_key(42);
        let (a, b) = random_split(root);
        let (a1, a2) = random_split(a);
        let (b1, b2) = random_split(b);
        let (a1a, a1b) = random_split(a1);

        let all_keys = [root, a, b, a1, a2, b1, b2, a1a, a1b];
        for i in 0..all_keys.len() {
            for j in (i + 1)..all_keys.len() {
                assert_ne!(
                    all_keys[i], all_keys[j],
                    "keys at positions {i} and {j} should be different"
                );
            }
        }
    }

    #[test]
    fn test_multi_level_split_determinism() {
        // Splitting the same key tree twice should produce identical results
        let root = random_key(123);
        let (a1, b1) = random_split(root);
        let (a1a1, a1b1) = random_split(a1);

        let (a2, b2) = random_split(root);
        let (a1a2, a1b2) = random_split(a2);

        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
        assert_eq!(a1a1, a1a2);
        assert_eq!(a1b1, a1b2);
    }

    #[test]
    fn test_fold_in_composition() {
        let key = random_key(42);
        // fold_in(fold_in(key, 0), 1) should differ from fold_in(key, 0) and fold_in(key, 1)
        let f0 = random_fold_in(key, 0);
        let f1 = random_fold_in(key, 1);
        let f01 = random_fold_in(f0, 1);
        let f10 = random_fold_in(f1, 0);

        assert_ne!(f0, f1, "fold_in(key,0) != fold_in(key,1)");
        assert_ne!(f01, f10, "fold_in composition should not commute");
        assert_ne!(f01, f0, "double fold should differ from single");
        assert_ne!(f01, f1, "double fold should differ from single");
    }

    #[test]
    fn test_fold_in_sequence_samples_independent() {
        // Fold in a sequence of indices and verify samples are independent
        let root = random_key(42);
        let n = 1000;
        let mut all_means = Vec::new();
        for i in 0..10 {
            let derived = random_fold_in(root, i);
            let samples = random_uniform(derived, n, 0.0, 1.0);
            let mean = samples.iter().sum::<f64>() / n as f64;
            all_means.push(mean);
        }
        // Each mean should be near 0.5
        for (i, &mean) in all_means.iter().enumerate() {
            assert!(
                (mean - 0.5).abs() < 0.05,
                "fold_in({i}) uniform mean should be ~0.5, got {mean}"
            );
        }
    }

    #[test]
    fn test_large_seed_values() {
        // Seeds near u64::MAX should work correctly
        let key_max = random_key(u64::MAX);
        assert_eq!(key_max.0[0], u32::MAX);
        assert_eq!(key_max.0[1], u32::MAX);

        let key_large = random_key(u64::MAX - 1);
        assert_ne!(key_max, key_large);

        // Both should produce valid samples
        let samples_max = random_uniform(key_max, 100, 0.0, 1.0);
        let samples_large = random_uniform(key_large, 100, 0.0, 1.0);
        assert_ne!(samples_max, samples_large);
        for v in samples_max.iter().chain(samples_large.iter()) {
            assert!(*v >= 0.0 && *v < 1.0, "large seed sample {v} out of range");
        }
    }

    #[test]
    fn test_exponential_via_uniform() {
        // Exponential distribution via -log(uniform): verify mean ≈ 1/λ
        let key = random_key(42);
        let n = 10_000;
        let uniforms = random_uniform(key, n, 0.0, 1.0);
        let lambda = 2.0;
        let exponentials: Vec<f64> = uniforms
            .iter()
            .map(|&u| -u.max(1e-30).ln() / lambda)
            .collect();

        let mean = exponentials.iter().sum::<f64>() / n as f64;
        let expected_mean = 1.0 / lambda;
        assert!(
            (mean - expected_mean).abs() < 0.05,
            "exponential mean should be ~{expected_mean}, got {mean}"
        );

        // All values should be non-negative
        assert!(
            exponentials.iter().all(|&v| v >= 0.0),
            "exponential values should be non-negative"
        );
    }

    #[test]
    fn test_exponential_matches_jax_log1p_transform() {
        // JAX: exponential(key) = -log1p(-uniform(key)); fj scales by 1/rate.
        // Pin the exact transform against fj's (JAX-matched) uniform stream so a
        // regression to -log(u) or a clamp hack is caught bit-for-bit.
        let key = random_key(7);
        let n = 256;
        let u = random_uniform(key, n, 0.0, 1.0);
        let e1 = random_exponential(key, n, 1.0);
        let e2 = random_exponential(key, n, 2.5);
        assert_eq!(e1.len(), n);
        for i in 0..n {
            let base = -(-u[i]).ln_1p();
            assert_eq!(
                e1[i].to_bits(),
                base.to_bits(),
                "rate=1 transform must equal -log1p(-u) at {i}"
            );
            assert_eq!(
                e2[i].to_bits(),
                (base / 2.5).to_bits(),
                "rate=2.5 must be the rate=1 base divided by rate at {i}"
            );
            assert!(e1[i] >= 0.0 && e1[i].is_finite());
        }
    }

    #[test]
    fn test_truncated_normal_via_rejection() {
        // Truncated normal in [-2, 2] via rejection sampling
        let key = random_key(42);
        let n = 20_000;
        let normals = random_normal(key, n);
        let truncated: Vec<f64> = normals
            .into_iter()
            .filter(|&v| (-2.0..=2.0).contains(&v))
            .collect();

        // Should retain ~95.4% of samples (2-sigma)
        let retention = truncated.len() as f64 / n as f64;
        assert!(
            (retention - 0.954).abs() < 0.02,
            "truncated normal retention should be ~95.4%, got {:.1}%",
            retention * 100.0
        );

        // Mean should be ~0 (symmetric truncation)
        let mean = truncated.iter().sum::<f64>() / truncated.len() as f64;
        assert!(
            mean.abs() < 0.05,
            "truncated normal mean should be ~0, got {mean}"
        );
    }

    #[test]
    fn test_multi_seed_ks_uniform() {
        // Run KS test across 5 different seeds to verify consistency
        for seed in [0, 42, 100, 999, 65535] {
            let key = random_key(seed);
            let n = 5_000;
            let mut vals = random_uniform(key, n, 0.0, 1.0);
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut d_max = 0.0_f64;
            for (i, v) in vals.iter().enumerate() {
                let empirical = (i + 1) as f64 / n as f64;
                d_max = d_max.max((empirical - *v).abs());
                let empirical_minus = i as f64 / n as f64;
                d_max = d_max.max((empirical_minus - *v).abs());
            }
            let critical = 1.63 / (n as f64).sqrt();
            assert!(
                d_max < critical,
                "KS test failed for seed {seed}: D={d_max:.4}, critical={critical:.4}"
            );
        }
    }

    #[test]
    fn test_multi_seed_normal_mean_stddev() {
        // Verify normal distribution statistics across multiple seeds
        for seed in [7, 42, 256, 1000, 50000] {
            let key = random_key(seed);
            let n = 5_000;
            let vals = random_normal(key, n);
            let mean = vals.iter().sum::<f64>() / n as f64;
            let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            let stddev = variance.sqrt();
            assert!(
                mean.abs() < 0.08,
                "seed {seed}: normal mean should be ~0, got {mean}"
            );
            assert!(
                (stddev - 1.0).abs() < 0.08,
                "seed {seed}: normal stddev should be ~1, got {stddev}"
            );
        }
    }

    #[test]
    fn test_split_samples_uncorrelated() {
        // Samples from split keys should be uncorrelated
        let root = random_key(42);
        let (k1, k2) = random_split(root);
        let n = 1000;
        let samples1 = random_uniform(k1, n, 0.0, 1.0);
        let samples2 = random_uniform(k2, n, 0.0, 1.0);

        // Compute Pearson correlation coefficient
        let mean1 = samples1.iter().sum::<f64>() / n as f64;
        let mean2 = samples2.iter().sum::<f64>() / n as f64;
        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;
        for i in 0..n {
            let d1 = samples1[i] - mean1;
            let d2 = samples2[i] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }
        let corr = cov / (var1.sqrt() * var2.sqrt());
        assert!(
            corr.abs() < 0.1,
            "split key samples should be uncorrelated, got r={corr:.4}"
        );
    }

    // === Tests for new distributions ===

    #[test]
    fn test_exponential_positive() {
        let key = random_key(42);
        let vals = random_exponential(key, 1000, 1.0);
        assert!(vals.iter().all(|&v| v > 0.0), "exponential should be > 0");
    }

    #[test]
    fn test_exponential_mean() {
        let key = random_key(42);
        let rate = 2.0;
        let vals = random_exponential(key, 10_000, rate);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let expected = 1.0 / rate;
        assert!(
            (mean - expected).abs() < 0.05,
            "exponential mean={mean:.4} expected={expected:.4}"
        );
    }

    #[test]
    fn test_gumbel_deterministic() {
        let key = random_key(42);
        let a = random_gumbel(key, 100, 0.0, 1.0);
        let b = random_gumbel(key, 100, 0.0, 1.0);
        assert_eq!(a, b, "gumbel: same key must produce same samples");
    }

    #[test]
    fn test_laplace_symmetric() {
        let key = random_key(42);
        let vals = random_laplace(key, 10_000, 0.0, 1.0);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(
            mean.abs() < 0.1,
            "laplace(0,1) should have mean ~0, got {mean:.4}"
        );
    }

    #[test]
    fn test_laplace_matches_jax_sign_log1p_transform() {
        // JAX: u ~ uniform(finfo.eps - 1, 1); laplace = sign(u)*log1p(-|u|).
        // Pin the exact transform (and the loc/scale generalization) over the same
        // key, guarding against the old sign-flipped -sign(u-0.5)*ln(1-2|u-0.5|).
        let key = random_key(11);
        let n = 256;
        let minval = f64::from(f32::EPSILON - 1.0);
        let u = random_uniform(key, n, minval, 1.0);
        let l = random_laplace(key, n, 0.0, 1.0);
        let l2 = random_laplace(key, n, 3.0, 2.0);
        assert_eq!(l.len(), n);
        for i in 0..n {
            let base = u[i].signum() * (-u[i].abs()).ln_1p();
            assert_eq!(
                l[i].to_bits(),
                base.to_bits(),
                "standard laplace must equal sign(u)*log1p(-|u|) at {i}"
            );
            assert_eq!(
                l2[i].to_bits(),
                (3.0 + 2.0 * base).to_bits(),
                "loc/scale generalization mismatch at {i}"
            );
            assert!(l[i].is_finite(), "laplace sample must be finite at {i}");
        }
    }

    #[test]
    fn test_logistic_deterministic() {
        let key = random_key(42);
        let a = random_logistic(key, 100, 0.0, 1.0);
        let b = random_logistic(key, 100, 0.0, 1.0);
        assert_eq!(a, b, "logistic: same key must produce same samples");
    }

    #[test]
    fn test_logistic_symmetric() {
        // Standard logistic is symmetric about 0 (median 0, mean 0).
        let key = random_key(42);
        let vals = random_logistic(key, 10_000, 0.0, 1.0);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(
            mean.abs() < 0.1,
            "logistic(0,1) should have mean ~0, got {mean:.4}"
        );
    }

    #[test]
    fn test_logistic_matches_jax_log_minus_log1p_transform() {
        // JAX `_logistic`: x ~ uniform(finfo.tiny, 1); logistic = log(x) - log1p(-x).
        // Pin the exact two-term transform over the same key (and the loc/scale
        // generalization), bit-for-bit — guarding against an ad-hoc logit(x) =
        // log(x/(1-x)) form, which differs in the last ulp as x -> 1.
        let key = random_key(7);
        let n = 256;
        let tiny = f64::from(f32::MIN_POSITIVE);
        let x = random_uniform(key, n, tiny, 1.0);
        let s = random_logistic(key, n, 0.0, 1.0);
        let s2 = random_logistic(key, n, 3.0, 2.0);
        assert_eq!(s.len(), n);
        for i in 0..n {
            let base = x[i].ln() - (-x[i]).ln_1p();
            assert_eq!(
                s[i].to_bits(),
                base.to_bits(),
                "standard logistic must equal log(x) - log1p(-x) at {i}"
            );
            assert_eq!(
                s2[i].to_bits(),
                (3.0 + 2.0 * base).to_bits(),
                "loc/scale generalization mismatch at {i}"
            );
            assert!(s[i].is_finite(), "logistic sample must be finite at {i}");
        }
    }

    #[test]
    fn test_maxwell_deterministic() {
        let key = random_key(42);
        let a = random_maxwell(key, 100);
        let b = random_maxwell(key, 100);
        assert_eq!(a, b, "maxwell: same key must produce same samples");
    }

    #[test]
    #[should_panic(expected = "random_maxwell normal draw output size overflow")]
    fn maxwell_rejects_normal_draw_count_overflow() {
        let _ = random_maxwell(random_key(42), usize::MAX / 3 + 1);
    }

    #[test]
    fn test_maxwell_matches_jax_norm_of_three_normals() {
        // JAX `_maxwell`: norm(normal(key, shape+(3,)), axis=-1). Pin the exact
        // transform bit-for-bit over the SAME normal stream: sample i is the L2
        // norm of the i-th row-major triple (matching JAX's (count,3) layout).
        let key = random_key(7);
        let n = 128;
        let normals = random_normal(key, n * 3);
        let m = random_maxwell(key, n);
        assert_eq!(m.len(), n);
        for i in 0..n {
            let (a, b, c) = (normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]);
            let expected = (a * a + b * b + c * c).sqrt();
            assert_eq!(
                m[i].to_bits(),
                expected.to_bits(),
                "maxwell[{i}] must equal sqrt(n0^2+n1^2+n2^2) of the i-th normal triple"
            );
            assert!(m[i] >= 0.0, "maxwell magnitude must be non-negative at {i}");
        }
    }

    #[test]
    fn test_maxwell_mean_matches_distribution() {
        // Maxwell-Boltzmann (sigma=1) has mean 2*sqrt(2/pi) ~ 1.5958.
        let key = random_key(123);
        let vals = random_maxwell(key, 50_000);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let expected = 2.0 * (2.0 / std::f64::consts::PI).sqrt();
        assert!(
            (mean - expected).abs() < 0.05,
            "maxwell mean={mean:.4} expected={expected:.4}"
        );
    }

    #[test]
    fn test_randint_range() {
        let key = random_key(42);
        let vals = random_randint(key, 1000, 0, 10);
        assert!(
            vals.iter().all(|&v| (0..10).contains(&v)),
            "randint should be in [minval, maxval)"
        );
    }

    #[test]
    fn test_randint_all_values_possible() {
        let key = random_key(42);
        let vals = random_randint(key, 10_000, 0, 10);
        for i in 0..10 {
            assert!(vals.contains(&i), "randint should produce value {i}");
        }
    }

    #[test]
    fn test_randint_matches_two_word_modular_reduction() {
        // Independently verify JAX's multiplier trick against the value it is an
        // identity for: for span <= 2^16 the u32 multiply does not overflow, so
        // `((hi % span)*multiplier + (lo % span)) % span` must equal the full
        // 64-bit `(hi*2^32 + lo) % span`. This pins the reduction math separately
        // from the (oracle-validated) bit source, without needing a live JAX.
        for &(minval, maxval) in &[(0i64, 10i64), (-5, 5), (3, 1000), (100, 165)] {
            let span = (maxval - minval) as u64; // <= 2^16 for these cases
            let key = random_key(7);
            let count = 256;
            let (k1, k2) = random_split(key);
            let higher = generate_u32_bits(k1, count);
            let lower = generate_u32_bits(k2, count);
            let got = random_randint(key, count, minval, maxval);
            for i in 0..count {
                let full = (u64::from(higher[i]) << 32 | u64::from(lower[i])) % span;
                assert_eq!(
                    got[i],
                    minval + full as i64,
                    "randint must equal (hi*2^32+lo) mod span for [{minval},{maxval}) at {i}"
                );
            }
        }
    }

    #[test]
    fn test_randint_deterministic() {
        // Same key/args reproduce; different keys differ.
        let a = random_randint(random_key(1), 64, 0, 1000);
        let b = random_randint(random_key(1), 64, 0, 1000);
        let c = random_randint(random_key(2), 64, 0, 1000);
        assert_eq!(a, b, "randint must be deterministic per key");
        assert_ne!(a, c, "different keys should produce different draws");
    }

    #[test]
    fn test_permutation_is_permutation() {
        let key = random_key(42);
        let n = 100;
        let perm = random_permutation(key, n);
        assert_eq!(perm.len(), n);
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(sorted, expected, "permutation should contain 0..n exactly");
    }

    #[test]
    fn test_permutation_empty() {
        let key = random_key(42);
        let perm = random_permutation(key, 0);
        assert!(perm.is_empty());
    }

    #[test]
    fn test_permutation_shuffles() {
        let key = random_key(42);
        let perm = random_permutation(key, 100);
        let identity: Vec<usize> = (0..100).collect();
        assert_ne!(perm, identity, "permutation should shuffle");
    }

    #[test]
    fn test_permutation_deterministic() {
        let key = random_key(7);
        assert_eq!(
            random_permutation(key, 50),
            random_permutation(key, 50),
            "same key must produce same permutation"
        );
    }

    #[test]
    fn test_permutation_n1_is_identity() {
        // JAX: ceil(3*ln(1)/ln(u32max)) == 0 rounds -> identity for n=1.
        let key = random_key(99);
        assert_eq!(random_permutation(key, 1), vec![0]);
    }

    #[test]
    fn test_permutation_matches_jax_sort_by_random_bits() {
        // Pin the JAX `_shuffle` algorithm exactly for the single-round regime
        // (n small enough that num_rounds == 1): permutation == stable argsort of
        // random_bits(split(key).1, n). This guards against regressing to the old
        // ad-hoc Fisher-Yates, which is a DIFFERENT permutation for the same key.
        let key = random_key(42);
        let n = 64; // ceil(3*ln(64)/ln(u32max)) = ceil(0.56) = 1 round
        let exponent = 3.0_f64;
        let num_rounds = (exponent * (n as f64).ln() / f64::from(u32::MAX).ln()).ceil() as usize;
        assert_eq!(num_rounds, 1, "this test assumes the single-round regime");

        let (_next, subkey) = random_split(key);
        let sort_keys = generate_u32_bits(subkey, n);
        let mut expected: Vec<usize> = (0..n).collect();
        expected.sort_by_key(|&i| sort_keys[i]);

        assert_eq!(
            random_permutation(key, n),
            expected,
            "permutation must equal stable argsort of JAX random_bits"
        );
    }

    #[test]
    fn test_choice_uniform_replace_matches_randint() {
        // JAX p=None, replace=True: ind = randint(key, shape, 0, n).
        let key = random_key(42);
        let (n, d) = (10usize, 25usize);
        let got = random_choice(key, n, d, true, None).unwrap();
        let expected: Vec<usize> = random_randint(key, d, 0, n as i64)
            .into_iter()
            .map(|i| i as usize)
            .collect();
        assert_eq!(got, expected);
        assert!(got.iter().all(|&i| i < n));
    }

    #[test]
    fn test_choice_uniform_noreplace_matches_permutation_prefix() {
        // JAX p=None, replace=False: permutation(key, n)[:n_draws].
        let key = random_key(42);
        let (n, d) = (50usize, 12usize);
        let got = random_choice(key, n, d, false, None).unwrap();
        let mut expected = random_permutation(key, n);
        expected.truncate(d);
        assert_eq!(got, expected);
        // Without replacement → all indices distinct.
        let mut sorted = got.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), d, "no-replacement draws must be distinct");
    }

    #[test]
    fn test_choice_weighted_replace_matches_inverse_cdf() {
        // JAX p given, replace=True: r = total*(1-u); ind = searchsorted_left(cuml, r).
        let key = random_key(7);
        let weights = [0.1f64, 0.2, 0.3, 0.4];
        let d = 20usize;
        let got = random_choice(key, weights.len(), d, true, Some(&weights)).unwrap();

        let mut cuml = Vec::new();
        let mut acc = 0.0;
        for &w in &weights {
            acc += w;
            cuml.push(acc);
        }
        let total = *cuml.last().unwrap();
        let uniforms = random_uniform(key, d, 0.0, 1.0);
        let expected: Vec<usize> = uniforms
            .into_iter()
            .map(|u| {
                let r = total * (1.0 - u);
                cuml.partition_point(|&x| x < r).min(weights.len() - 1)
            })
            .collect();
        assert_eq!(got, expected);
        assert!(got.iter().all(|&i| i < weights.len()));
    }

    #[test]
    fn test_choice_weighted_noreplace_matches_gumbel_topk() {
        // JAX p given, replace=False: ind = top_k(gumbel(key,n)+log(p), n_draws).
        let key = random_key(7);
        let weights = [0.4f64, 0.1, 0.2, 0.15, 0.15];
        let d = 3usize;
        let got = random_choice(key, weights.len(), d, false, Some(&weights)).unwrap();

        let gumbel = random_gumbel(key, weights.len(), 0.0, 1.0);
        let mut scored: Vec<(f64, usize)> = (0..weights.len())
            .map(|i| (gumbel[i] + weights[i].ln(), i))
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let expected: Vec<usize> = scored.into_iter().take(d).map(|(_, i)| i).collect();
        assert_eq!(got, expected);
        let mut sorted = got.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), d, "no-replacement draws must be distinct");
    }

    #[test]
    fn test_choice_deterministic_and_errors() {
        let key = random_key(1);
        assert_eq!(
            random_choice(key, 8, 4, true, None),
            random_choice(key, 8, 4, true, None)
        );
        assert_eq!(
            random_choice(key, 5, 0, false, None).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            random_choice(key, 0, 3, true, None),
            Err(ChoiceError::EmptyPopulation)
        );
        assert_eq!(
            random_choice(key, 3, 5, false, None),
            Err(ChoiceError::SampleLargerThanPopulation {
                n_draws: 5,
                n_inputs: 3
            })
        );
        let w = [0.5f64, 0.5];
        assert_eq!(
            random_choice(key, 3, 2, true, Some(&w)),
            Err(ChoiceError::WeightsLengthMismatch {
                weights_len: 2,
                n_inputs: 3
            })
        );
    }

    // === New distribution tests ===

    #[test]
    fn test_gamma_positive() {
        let key = random_key(42);
        let vals = random_gamma(key, 100, 2.0);
        assert!(vals.iter().all(|&v| v > 0.0 || v.is_nan()));
    }

    #[test]
    fn random_chi2_and_t_match_jax_reference_f32() {
        // jax.random.chisquare/t(PRNGKey(0), df, (6,)) in DEFAULT f32 mode. Both compose
        // the (now-faithful) gamma: chi2 = 2*gamma(df/2); t = z*sqrt((df/2)/gamma).
        let chi2_df4: [f64; 6] = [
            3.1795756816864014,
            3.5198826789855957,
            2.25801420211792,
            7.948668003082275,
            5.909200191497803,
            3.9576170444488525,
        ];
        let got_chi2 = random_chi2(random_key(0), 6, 4.0);
        for (i, (&w, &g)) in chi2_df4.iter().zip(got_chi2.iter()).enumerate() {
            assert!(
                (w - g).abs() <= 1e-4 * w.abs().max(1.0),
                "chi2(df=4) elem {i}: JAX-f32 {w} vs fj {g}"
            );
        }
        let t_df5: [f64; 6] = [
            0.758949875831604,
            -2.055917739868164,
            -1.281833529472351,
            -1.5719773769378662,
            -0.555972695350647,
            0.6316303610801697,
        ];
        let got_t = random_t(random_key(0), 6, 5.0);
        for (i, (&w, &g)) in t_df5.iter().zip(got_t.iter()).enumerate() {
            assert!(
                (w - g).abs() <= 1e-4 * w.abs().max(1.0),
                "t(df=5) elem {i}: JAX-f32 {w} vs fj {g}"
            );
        }
    }

    #[test]
    fn random_dirichlet_matches_jax_reference_f32() {
        // jax.random.dirichlet(PRNGKey(0), [1,2,3], (4,)) in DEFAULT f32 mode (4 rows x 3).
        let want: [f64; 12] = [
            0.14228560030460358,
            0.40488260984420776,
            0.45283177495002747,
            0.28478530049324036,
            0.351531058549881,
            0.36368367075920105,
            0.1310892552137375,
            0.10866259038448334,
            0.7602481842041016,
            0.10009737312793732,
            0.2562600374221802,
            0.6436426043510437,
        ];
        let got = random_dirichlet(random_key(0), 4, &[1.0, 2.0, 3.0]);
        assert_eq!(got.len(), want.len());
        for (i, (&w, &g)) in want.iter().zip(got.iter()).enumerate() {
            assert!(
                (w - g).abs() <= 1e-4 * w.abs().max(1.0),
                "dirichlet elem {i}: JAX-f32 {w} vs fj {g}"
            );
        }
    }

    #[test]
    fn random_beta_matches_jax_reference_f32() {
        // jax.random.beta(random.PRNGKey(0), a, b, (8,)) in DEFAULT f32 mode.
        let cases: [(f64, f64, [f64; 8]); 2] = [
            (
                2.0,
                3.0,
                [
                    0.2362736165523529,
                    0.4904595613479614,
                    0.6837947964668274,
                    0.5423442125320435,
                    0.4902370274066925,
                    0.3394175171852112,
                    0.07781261205673218,
                    0.27173343300819397,
                ],
            ),
            (
                0.5,
                0.5,
                [
                    0.774951696395874,
                    0.9974318146705627,
                    0.961112380027771,
                    0.35893872380256653,
                    0.1144595518708229,
                    0.9511231184005737,
                    0.8620636463165283,
                    0.463880330324173,
                ],
            ),
        ];
        for (a, b, want) in cases {
            let got = random_beta(random_key(0), 8, a, b);
            for (i, (&w, &g)) in want.iter().zip(got.iter()).enumerate() {
                assert!(
                    (w - g).abs() <= 1e-4 * w.abs().max(1.0),
                    "beta(a={a},b={b}) elem {i}: JAX-f32 {w} vs fj {g}"
                );
            }
        }
    }

    #[test]
    fn random_poisson_matches_jax_reference_f32() {
        // jax.random.poisson(random.PRNGKey(0), lam, (8,)) in DEFAULT f32 mode.
        // lam=3 -> Knuth path, lam=30 -> PTRS rejection path.
        let cases: [(f64, [u64; 8]); 2] = [
            (3.0, [0, 0, 4, 4, 1, 2, 2, 6]),
            (30.0, [22, 27, 23, 36, 25, 32, 36, 29]),
        ];
        for (lam, want) in cases {
            let got = random_poisson(random_key(0), 8, lam);
            assert_eq!(got, want.to_vec(), "poisson(lam={lam}) vs JAX-f32");
        }
    }

    #[test]
    fn random_split_n_matches_jax() {
        // jax.random.split(random.PRNGKey(0), n) — bit-for-bit reference from the venv.
        let s2 = random_split_n(random_key(0), 2);
        assert_eq!(s2[0].0, [1797259609, 2579123966]);
        assert_eq!(s2[1].0, [928981903, 3453687069]);
        let s3 = random_split_n(random_key(0), 3);
        // Prefix-stable: first two keys equal split(key,2).
        assert_eq!(s3[0].0, s2[0].0);
        assert_eq!(s3[1].0, s2[1].0);
        assert_eq!(s3[2].0, [4146024105, 2718843009]);
        // Consistent with the 2-way random_split.
        let (a, b) = random_split(random_key(0));
        assert_eq!((s2[0].0, s2[1].0), (a.0, b.0));
    }

    /// STRICT element-wise parity oracle for `random_gamma` vs `jax.random.gamma`.
    ///
    /// `random_gamma` is now the faithful per-element-key `_gamma_one` port
    /// (jax/_src/random.py), so it matches JAX ELEMENT-FOR-ELEMENT. References are JAX in
    /// its DEFAULT **f32** mode (fj's RNG is f32-based — mantissa-bits uniform matches JAX-f32
    /// exactly, normal ~3e-7). fj does the Marsaglia-Tsang arithmetic in f64 on those
    /// f32-matching draws, so parity is to ~f32 tolerance (1e-4), not bit-exact. Covers the
    /// non-boost path (a>=1: a=2, a=5) AND the boost path (a<1: a=0.5, which scales by
    /// `Uniform^(1/a)`). See docs/NEGATIVE_EVIDENCE.md 2026-07-02.
    #[test]
    fn random_gamma_matches_jax_reference_f32() {
        // jax.random.gamma(random.PRNGKey(0), a, (8,)) in DEFAULT f32 mode.
        let cases: [(f64, [f64; 8]); 3] = [
            (
                2.0,
                [
                    1.5897877216339111,
                    1.7599413394927979,
                    1.12900710105896,
                    3.9743340015411377,
                    2.9546000957489014,
                    1.9788084030151367,
                    1.4843521118164062,
                    0.4591488242149353,
                ],
            ),
            (
                5.0,
                [
                    4.537209510803223,
                    4.821604251861572,
                    3.7203195095062256,
                    8.080060005187988,
                    6.656269073486328,
                    5.176936149597168,
                    4.356862545013428,
                    2.311718702316284,
                ],
            ),
            (
                0.5,
                [
                    0.8845648169517517,
                    1.0290775299072266,
                    0.04707420989871025,
                    0.07322501391172409,
                    0.3485542833805084,
                    1.3415878129308112e-05,
                    0.3963169455528259,
                    0.22419296205043793,
                ],
            ),
        ];
        for (a, want) in cases {
            let got = random_gamma(random_key(0), 8, a);
            for (i, (&w, &g)) in want.iter().zip(got.iter()).enumerate() {
                assert!(
                    (w - g).abs() <= 1e-4 * w.abs().max(1.0),
                    "gamma(a={a}) elem {i}: JAX-f32 {w} vs fj {g}"
                );
            }
        }
    }

    #[test]
    fn test_gamma_shape_less_than_one() {
        let key = random_key(42);
        let vals = random_gamma(key, 100, 0.5);
        let valid: Vec<_> = vals.iter().filter(|v| !v.is_nan()).collect();
        assert!(!valid.is_empty(), "should produce some valid samples");
    }

    #[test]
    fn test_beta_bounds() {
        let key = random_key(42);
        let vals = random_beta(key, 100, 2.0, 5.0);
        assert!(vals.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_beta_symmetric() {
        let key = random_key(42);
        let vals = random_beta(key, 1000, 1.0, 1.0);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(
            (mean - 0.5).abs() < 0.1,
            "Beta(1,1) should have mean ~0.5, got {mean}"
        );
    }

    #[test]
    fn test_poisson_nonnegative() {
        let key = random_key(42);
        let vals = random_poisson(key, 100, 5.0);
        assert!(vals.iter().all(|&v| v < 1000), "poisson should be bounded");
    }

    #[test]
    fn test_poisson_mean() {
        let key = random_key(42);
        let lam = 10.0;
        let vals = random_poisson(key, 1000, lam);
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        assert!(
            (mean - lam).abs() < 1.5,
            "Poisson({lam}) mean should be ~{lam}, got {mean}"
        );
    }

    #[test]
    fn test_truncated_normal_bounds() {
        let key = random_key(42);
        let vals = random_truncated_normal(key, 100, -1.0, 1.0);
        assert!(vals.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_truncated_normal_narrow_offcenter_is_proper_distribution() {
        // Regression for the old rejection sampler: for a narrow, off-center
        // range like [2.0, 2.5] it rejected ~98% of standard-normal draws and
        // padded the rest with the EXACT midpoint 2.25 — a near-degenerate
        // distribution. The inverse-CDF method (matching JAX) yields a proper
        // truncated normal: strictly inside the open interval, full spread, and
        // the analytic truncated-normal mean (~2.2045), NOT the midpoint 2.25.
        let key = random_key(7);
        let n = 20_000;
        let (lower, upper) = (2.0_f64, 2.5_f64);
        let vals = random_truncated_normal(key, n, lower, upper);

        assert!(
            vals.iter().all(|&v| v > lower && v < upper),
            "all samples must be strictly inside ({lower}, {upper})"
        );
        let distinct = vals
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct > n / 2,
            "distribution must be non-degenerate (old midpoint-fill bug), got {distinct} distinct of {n}"
        );
        let mean = vals.iter().sum::<f64>() / n as f64;
        assert!(
            (mean - 2.2045).abs() < 0.02,
            "truncated-normal mean over [2,2.5] should be ~2.2045 (analytic), got {mean}"
        );
    }

    #[test]
    fn test_cauchy_produces_values() {
        let key = random_key(42);
        let vals = random_cauchy(key, 100);
        assert_eq!(vals.len(), 100);
        // Cauchy has heavy tails, just check we get finite values mostly
        let finite_count = vals.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count > 90);
    }

    #[test]
    fn test_pareto_greater_than_one() {
        let key = random_key(42);
        let vals = random_pareto(key, 100, 1.0);
        assert!(vals.iter().all(|&v| v >= 1.0));
    }

    #[test]
    fn test_weibull_positive() {
        let key = random_key(42);
        let vals = random_weibull(key, 100, 1.0, 2.0);
        assert!(vals.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_weibull_scale() {
        let key = random_key(42);
        let vals = random_weibull(key, 1000, 5.0, 1.0);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(
            (mean - 5.0).abs() < 1.0,
            "Weibull(5, 1) should have mean ~5, got {mean}"
        );
    }

    #[test]
    fn test_rayleigh_positive() {
        let key = random_key(42);
        let vals = random_rayleigh(key, 100, 1.0);
        assert!(vals.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_chi2_positive() {
        let key = random_key(42);
        let vals = random_chi2(key, 100, 5.0);
        assert!(vals.iter().all(|&v| v > 0.0 || v.is_nan()));
    }

    #[test]
    fn test_chi2_mean() {
        let key = random_key(42);
        let df = 10.0;
        let vals = random_chi2(key, 1000, df);
        let valid: Vec<_> = vals.into_iter().filter(|v| !v.is_nan()).collect();
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        assert!(
            (mean - df).abs() < 2.0,
            "Chi2({df}) should have mean ~{df}, got {mean}"
        );
    }

    #[test]
    fn test_t_produces_values() {
        let key = random_key(42);
        let vals = random_t(key, 100, 5.0);
        assert_eq!(vals.len(), 100);
        let finite_count = vals.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count > 90);
    }

    #[test]
    fn test_t_symmetric() {
        let key = random_key(42);
        let vals = random_t(key, 1000, 10.0);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(mean.abs() < 0.5, "t(10) should have mean ~0, got {mean}");
    }

    #[test]
    fn test_dirichlet_sums_to_one() {
        let key = random_key(42);
        let alpha = [1.0, 1.0, 1.0];
        let vals = random_dirichlet(key, 10, &alpha);
        assert_eq!(vals.len(), 30); // 10 samples * 3 dimensions
        for i in 0..10 {
            let sum: f64 = vals[i * 3..(i + 1) * 3].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Dirichlet sample {} should sum to 1, got {sum}",
                i
            );
        }
    }

    #[test]
    #[should_panic(expected = "random_dirichlet output size overflow")]
    fn dirichlet_rejects_output_size_overflow() {
        let alpha = [1.0, 1.0];
        let _ = random_dirichlet(random_key(42), usize::MAX / 2 + 1, &alpha);
    }

    #[test]
    fn test_dirichlet_uniform() {
        let key = random_key(42);
        let alpha = [1.0, 1.0, 1.0];
        let vals = random_dirichlet(key, 100, &alpha);
        let means: Vec<f64> = (0..3)
            .map(|j| (0..100).map(|i| vals[i * 3 + j]).sum::<f64>() / 100.0)
            .collect();
        for (j, &m) in means.iter().enumerate() {
            assert!(
                (m - 1.0 / 3.0).abs() < 0.1,
                "Dirichlet(1,1,1) dim {} should have mean ~0.333, got {}",
                j,
                m
            );
        }
    }

    #[test]
    fn test_geometric_positive() {
        let key = random_key(42);
        let vals = random_geometric(key, 100, 0.5);
        assert!(vals.iter().all(|&v| v >= 1));
    }

    #[test]
    fn test_geometric_mean() {
        let key = random_key(42);
        let p = 0.5;
        let vals = random_geometric(key, 1000, p);
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        let expected_mean = 1.0 / p;
        assert!(
            (mean - expected_mean).abs() < 0.5,
            "Geometric({p}) should have mean ~{expected_mean}, got {mean}"
        );
    }
}
