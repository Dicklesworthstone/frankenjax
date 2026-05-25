//! Tensor contraction operations matching JAX's jnp module.
//!
//! Provides tensordot and related operations.

#![allow(dead_code)]

/// Compute tensor dot product along specified axes.
///
/// Matches `jnp.tensordot(a, b, axes)`.
///
/// # Arguments
/// * `a` - First tensor as flat f64 array
/// * `a_shape` - Shape of first tensor
/// * `b` - Second tensor as flat f64 array
/// * `b_shape` - Shape of second tensor
/// * `axes_a` - Axes of `a` to contract
/// * `axes_b` - Axes of `b` to contract (must have same length as axes_a)
///
/// # Returns
/// Result tensor and its shape
pub fn tensordot(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    axes_a: &[usize],
    axes_b: &[usize],
) -> (Vec<f64>, Vec<usize>) {
    assert_eq!(
        axes_a.len(),
        axes_b.len(),
        "axes_a and axes_b must have same length"
    );

    // Validate contraction dimensions match
    for (&ax_a, &ax_b) in axes_a.iter().zip(axes_b.iter()) {
        assert_eq!(
            a_shape[ax_a], b_shape[ax_b],
            "contracted dimensions must match"
        );
    }

    // Compute output shape: non-contracted axes of a, then non-contracted axes of b
    let mut out_shape = Vec::new();
    for (i, &dim) in a_shape.iter().enumerate() {
        if !axes_a.contains(&i) {
            out_shape.push(dim);
        }
    }
    for (i, &dim) in b_shape.iter().enumerate() {
        if !axes_b.contains(&i) {
            out_shape.push(dim);
        }
    }

    if out_shape.is_empty() {
        // Scalar result (full contraction)
        let mut sum = 0.0;
        for (i, &av) in a.iter().enumerate() {
            sum += av * b[i];
        }
        return (vec![sum], vec![]);
    }

    // Compute strides for a and b
    let a_strides = compute_strides(a_shape);
    let b_strides = compute_strides(b_shape);
    let out_strides = compute_strides(&out_shape);

    // Get non-contracted axes
    let free_axes_a: Vec<usize> = (0..a_shape.len()).filter(|i| !axes_a.contains(i)).collect();
    let _free_axes_b: Vec<usize> = (0..b_shape.len()).filter(|i| !axes_b.contains(i)).collect();

    // Contracted dimension size
    let contracted_size: usize = axes_a.iter().map(|&ax| a_shape[ax]).product();

    // Output size
    let out_size: usize = out_shape.iter().product();
    let mut result = vec![0.0; out_size];

    // For each output element
    for out_idx in 0..out_size {
        // Decode output index into coordinates
        let out_coords = index_to_coords(out_idx, &out_strides, out_shape.len());

        // Split output coords: first part for a's free axes, second for b's free axes
        let a_free_coords = &out_coords[..free_axes_a.len()];
        let b_free_coords = &out_coords[free_axes_a.len()..];

        // Sum over contracted dimensions
        let mut sum = 0.0;
        for c_idx in 0..contracted_size {
            // Decode contracted index into coordinates
            let c_coords = index_to_contracted_coords(c_idx, axes_a, a_shape);

            // Build full a index
            let mut a_flat_idx = 0;
            let mut free_i = 0;
            let mut contr_i = 0;
            for ax in 0..a_shape.len() {
                let coord = if axes_a.contains(&ax) {
                    let c = c_coords[contr_i];
                    contr_i += 1;
                    c
                } else {
                    let c = a_free_coords[free_i];
                    free_i += 1;
                    c
                };
                a_flat_idx += coord * a_strides[ax];
            }

            // Build full b index
            let mut b_flat_idx = 0;
            let mut free_i = 0;
            let mut contr_i = 0;
            for ax in 0..b_shape.len() {
                let coord = if axes_b.contains(&ax) {
                    let c = c_coords[contr_i];
                    contr_i += 1;
                    c
                } else {
                    let c = b_free_coords[free_i];
                    free_i += 1;
                    c
                };
                b_flat_idx += coord * b_strides[ax];
            }

            sum += a[a_flat_idx] * b[b_flat_idx];
        }
        result[out_idx] = sum;
    }

    (result, out_shape)
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn index_to_coords(mut idx: usize, strides: &[usize], ndim: usize) -> Vec<usize> {
    let mut coords = vec![0; ndim];
    for i in 0..ndim {
        if strides[i] > 0 {
            coords[i] = idx / strides[i];
            idx %= strides[i];
        }
    }
    coords
}

fn index_to_contracted_coords(idx: usize, axes: &[usize], shape: &[usize]) -> Vec<usize> {
    let contracted_shape: Vec<usize> = axes.iter().map(|&ax| shape[ax]).collect();
    let contracted_strides = compute_strides(&contracted_shape);
    index_to_coords(idx, &contracted_strides, axes.len())
}

/// Matrix multiplication as a special case of tensordot.
///
/// Matches `jnp.matmul(a, b)` for 2D arrays.
pub fn matmul_2d(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

/// Outer product of two vectors.
///
/// Matches `jnp.outer(a, b)`.
pub fn outer(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; a.len() * b.len()];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            result[i * b.len() + j] = av * bv;
        }
    }
    result
}

/// Inner product (dot product) of two vectors.
///
/// Matches `jnp.inner(a, b)` for 1D arrays.
pub fn inner(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Kronecker product of two matrices.
///
/// Matches `jnp.kron(a, b)`.
pub fn kron(a: &[f64], a_m: usize, a_n: usize, b: &[f64], b_m: usize, b_n: usize) -> Vec<f64> {
    let out_m = a_m * b_m;
    let out_n = a_n * b_n;
    let mut result = vec![0.0; out_m * out_n];

    for i in 0..a_m {
        for j in 0..a_n {
            let a_val = a[i * a_n + j];
            for k in 0..b_m {
                for l in 0..b_n {
                    let out_row = i * b_m + k;
                    let out_col = j * b_n + l;
                    result[out_row * out_n + out_col] = a_val * b[k * b_n + l];
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensordot_matrix_mul() {
        // 2x3 @ 3x2 = 2x2 via tensordot
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (result, shape) = tensordot(&a, &[2, 3], &b, &[3, 2], &[1], &[0]);
        assert_eq!(shape, vec![2, 2]);
        // [1,2,3] @ [1,3,5]^T = 1+6+15 = 22
        // [1,2,3] @ [2,4,6]^T = 2+8+18 = 28
        // [4,5,6] @ [1,3,5]^T = 4+15+30 = 49
        // [4,5,6] @ [2,4,6]^T = 8+20+36 = 64
        assert!((result[0] - 22.0).abs() < 1e-10);
        assert!((result[1] - 28.0).abs() < 1e-10);
        assert!((result[2] - 49.0).abs() < 1e-10);
        assert!((result[3] - 64.0).abs() < 1e-10);
    }

    #[test]
    fn tensordot_vector_dot() {
        // Vector dot product: sum(a * b)
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let (result, shape) = tensordot(&a, &[3], &b, &[3], &[0], &[0]);
        assert!(shape.is_empty()); // Scalar result
        assert!((result[0] - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn tensordot_outer_product() {
        // Outer product: tensordot with no axes contracted
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0];
        let (result, shape) = tensordot(&a, &[2], &b, &[3], &[], &[]);
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn matmul_2d_basic() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [5.0, 6.0, 7.0, 8.0]; // 2x2
        let result = matmul_2d(&a, 2, 2, &b, 2);
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert!((result[0] - 19.0).abs() < 1e-10);
        assert!((result[1] - 22.0).abs() < 1e-10);
        assert!((result[2] - 43.0).abs() < 1e-10);
        assert!((result[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn outer_basic() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let result = outer(&a, &b);
        assert_eq!(result, vec![3.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn inner_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = inner(&a, &b);
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn kron_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [0.0, 5.0, 6.0, 7.0]; // 2x2
        let result = kron(&a, 2, 2, &b, 2, 2);
        // Result is 4x4
        assert_eq!(result.len(), 16);
        // First 2x2 block is 1*b
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 5.0).abs() < 1e-10);
    }
}
