//! Oracle conformance tests for jnp.linalg.solve and lstsq.
//!
//! Reference values computed with JAX/NumPy:
//! ```python
//! import jax.numpy as jnp
//! x = jnp.linalg.solve(A, b)
//! x = jnp.linalg.lstsq(A, b)
//! ```

use fj_lax::linalg::{lstsq, solve, solve_multi_rhs};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

fn vec_approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y, tol))
}

// JAX reference: solve([[1,0],[0,1]], [3,4]) = [3,4]
#[test]
fn test_solve_identity() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [3.0, 4.0];
    let x = solve(&a, &b, 2).expect("should solve identity");
    assert!(
        vec_approx_eq(&x, &b, 1e-10),
        "identity solve: got {:?}, expected {:?}",
        x,
        b
    );
}

// JAX reference: solve([[2,1],[1,3]], [5,7]) = [1.6, 1.8]
#[test]
fn test_solve_2x2_jax_reference() {
    let a = [2.0, 1.0, 1.0, 3.0];
    let b = [5.0, 7.0];
    let x = solve(&a, &b, 2).expect("should solve 2x2");
    let expected = [1.6, 1.8];
    assert!(
        vec_approx_eq(&x, &expected, 1e-10),
        "2x2 solve: got {:?}, expected {:?}",
        x,
        expected
    );
}

// Verify A @ x = b
#[test]
fn test_solve_verifies_solution() {
    let a = [3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0];
    let b = [1.0, 2.0, 3.0];
    let x = solve(&a, &b, 3).expect("should solve 3x3");

    // Compute A @ x
    for i in 0..3 {
        let mut row_sum = 0.0;
        for j in 0..3 {
            row_sum += a[i * 3 + j] * x[j];
        }
        assert!(
            approx_eq(row_sum, b[i], 1e-8),
            "A@x[{}] = {}, expected {}",
            i,
            row_sum,
            b[i]
        );
    }
}

// Singular matrix: JAX's solve divides through the zero pivot and returns
// inf/NaN (NumPy raises LinAlgError); it does not fail.
#[test]
fn test_solve_singular_returns_non_finite() {
    let a = [1.0, 2.0, 2.0, 4.0]; // rank 1
    let b = [3.0, 6.0];
    let x = solve(&a, &b, 2).expect("singular solve must not fail (matches JAX, not NumPy)");
    assert!(
        x.iter().any(|v| !v.is_finite()),
        "singular solve must be non-finite, got {x:?}"
    );
}

// Test solve_multi_rhs
#[test]
fn test_solve_multi_rhs() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [1.0, 2.0, 3.0, 4.0]; // 2x2, two RHS columns
    let x = solve_multi_rhs(&a, &b, 2, 2).expect("should solve multi-rhs");
    assert!(
        vec_approx_eq(&x, &b, 1e-10),
        "identity multi-rhs: got {:?}, expected {:?}",
        x,
        b
    );
}

// JAX reference: lstsq for square system equals solve
#[test]
fn test_lstsq_square_system() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [3.0, 4.0];
    let x = lstsq(&a, 2, 2, &b).expect("should lstsq square");
    let expected = [3.0, 4.0];
    assert!(
        vec_approx_eq(&x, &expected, 1e-10),
        "lstsq square: got {:?}, expected {:?}",
        x,
        expected
    );
}

// JAX reference: lstsq for overdetermined system (linear regression)
// Points: (0,0), (1,1), (2,2) on y=x line
// A = [[1,0],[1,1],[1,2]], b = [0,1,2]
// lstsq gives intercept=0, slope=1
#[test]
fn test_lstsq_overdetermined_line() {
    let a = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0];
    let b = [0.0, 1.0, 2.0];
    let x = lstsq(&a, 3, 2, &b).expect("should lstsq overdetermined");

    // x[0] = intercept ≈ 0, x[1] = slope ≈ 1
    assert!(
        approx_eq(x[0], 0.0, 1e-8),
        "intercept should be ~0, got {}",
        x[0]
    );
    assert!(
        approx_eq(x[1], 1.0, 1e-8),
        "slope should be ~1, got {}",
        x[1]
    );
}

// Verify normal equations: A^T @ A @ x = A^T @ b
#[test]
fn test_lstsq_normal_equations() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
    let b = [1.0, 2.0, 3.0];
    let x = lstsq(&a, 3, 2, &b).expect("should lstsq");

    // Compute A^T @ A @ x
    let mut ata_x = [0.0; 2];
    let mut atb = [0.0; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut ata_ij = 0.0;
            for k in 0..3 {
                ata_ij += a[k * 2 + i] * a[k * 2 + j];
            }
            ata_x[i] += ata_ij * x[j];
        }
        for k in 0..3 {
            atb[i] += a[k * 2 + i] * b[k];
        }
    }

    assert!(
        vec_approx_eq(&ata_x, &atb, 1e-8),
        "A^T A x = A^T b: got {:?}, expected {:?}",
        ata_x,
        atb
    );
}

// Test with ill-conditioned but solvable matrix
#[test]
fn test_solve_ill_conditioned() {
    // Hilbert-like matrix (mildly ill-conditioned)
    let a = [1.0, 0.5, 0.5, 0.333333333];
    let b = [1.5, 0.833333333];
    let x = solve(&a, &b, 2).expect("should solve");

    // Verify solution
    for i in 0..2 {
        let mut row_sum = 0.0;
        for j in 0..2 {
            row_sum += a[i * 2 + j] * x[j];
        }
        assert!(
            approx_eq(row_sum, b[i], 1e-6),
            "ill-conditioned A@x[{}] = {}, expected {}",
            i,
            row_sum,
            b[i]
        );
    }
}
