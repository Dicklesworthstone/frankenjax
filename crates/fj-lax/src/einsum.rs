//! Einstein summation convention matching JAX's jnp.einsum.
//!
//! Supports subscript notation like "ij,jk->ik" for matrix multiplication.

use std::collections::{HashMap, HashSet};

/// Error type for einsum parsing and execution.
#[derive(Debug, Clone, PartialEq)]
pub enum EinsumError {
    InvalidSubscript(String),
    ShapeMismatch {
        index: char,
        expected: usize,
        got: usize,
    },
    MissingOperand {
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for EinsumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSubscript(s) => write!(f, "invalid einsum subscript: {s}"),
            Self::ShapeMismatch {
                index,
                expected,
                got,
            } => {
                write!(
                    f,
                    "dimension mismatch for index '{index}': expected {expected}, got {got}"
                )
            }
            Self::MissingOperand { expected, got } => {
                write!(f, "expected {expected} operands, got {got}")
            }
        }
    }
}

impl std::error::Error for EinsumError {}

/// Parse an einsum subscript string into input subscripts and output subscript.
///
/// Examples:
/// - "ij,jk->ik" -> (["ij", "jk"], "ik")
/// - "ij,jk" -> (["ij", "jk"], None) - implicit output
fn parse_subscripts(subscripts: &str) -> Result<(Vec<String>, Option<String>), EinsumError> {
    let subscripts = subscripts.replace(' ', "");

    let (inputs_str, output_str) = if let Some(idx) = subscripts.find("->") {
        (&subscripts[..idx], Some(&subscripts[idx + 2..]))
    } else {
        (subscripts.as_str(), None)
    };

    let input_subs: Vec<String> = inputs_str.split(',').map(|s| s.to_string()).collect();

    // Validate subscripts contain only lowercase letters
    for sub in &input_subs {
        if !sub.chars().all(|c| c.is_ascii_lowercase() || c == '.') {
            return Err(EinsumError::InvalidSubscript(format!(
                "subscript '{sub}' contains invalid characters"
            )));
        }
    }

    if let Some(out) = output_str
        && !out.chars().all(|c| c.is_ascii_lowercase() || c == '.')
    {
        return Err(EinsumError::InvalidSubscript(format!(
            "output subscript '{out}' contains invalid characters"
        )));
    }

    Ok((input_subs, output_str.map(|s| s.to_string())))
}

/// Compute implicit output subscript (indices that appear exactly once).
fn compute_implicit_output(input_subs: &[String]) -> String {
    let mut counts: HashMap<char, usize> = HashMap::new();
    for sub in input_subs {
        for c in sub.chars() {
            *counts.entry(c).or_insert(0) += 1;
        }
    }

    // Output indices are those that appear exactly once, in alphabetical order
    let mut output_chars: Vec<char> = counts
        .iter()
        .filter(|&(_, &count)| count == 1)
        .map(|(&c, _)| c)
        .collect();
    output_chars.sort();
    output_chars.into_iter().collect()
}

/// Execute einsum with two operands (most common case).
///
/// Matches `jnp.einsum(subscripts, a, b)`.
pub fn einsum2(
    subscripts: &str,
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), EinsumError> {
    let (input_subs, output_sub) = parse_subscripts(subscripts)?;

    if input_subs.len() != 2 {
        return Err(EinsumError::MissingOperand {
            expected: 2,
            got: input_subs.len(),
        });
    }

    let sub_a = &input_subs[0];
    let sub_b = &input_subs[1];
    let sub_out = output_sub.unwrap_or_else(|| compute_implicit_output(&input_subs));

    // Validate shapes match subscripts
    if sub_a.len() != a_shape.len() {
        return Err(EinsumError::InvalidSubscript(format!(
            "subscript '{}' has {} indices but operand has {} dimensions",
            sub_a,
            sub_a.len(),
            a_shape.len()
        )));
    }
    if sub_b.len() != b_shape.len() {
        return Err(EinsumError::InvalidSubscript(format!(
            "subscript '{}' has {} indices but operand has {} dimensions",
            sub_b,
            sub_b.len(),
            b_shape.len()
        )));
    }

    // Build index->dimension mapping
    let mut index_dims: HashMap<char, usize> = HashMap::new();
    for (i, c) in sub_a.chars().enumerate() {
        index_dims.insert(c, a_shape[i]);
    }
    for (i, c) in sub_b.chars().enumerate() {
        if let Some(&existing) = index_dims.get(&c) {
            if existing != b_shape[i] {
                return Err(EinsumError::ShapeMismatch {
                    index: c,
                    expected: existing,
                    got: b_shape[i],
                });
            }
        } else {
            index_dims.insert(c, b_shape[i]);
        }
    }

    // Compute output shape
    let out_shape: Vec<usize> = sub_out.chars().map(|c| index_dims[&c]).collect();

    // Find summed (contracted) indices
    let all_indices: HashSet<char> = sub_a.chars().chain(sub_b.chars()).collect();
    let out_indices: HashSet<char> = sub_out.chars().collect();
    let sum_indices: Vec<char> = all_indices.difference(&out_indices).copied().collect();

    // Compute sum dimensions
    let sum_dims: Vec<usize> = sum_indices.iter().map(|&c| index_dims[&c]).collect();
    let sum_size: usize = sum_dims.iter().product();

    // Output size
    let out_size: usize = out_shape.iter().product();
    if out_size == 0 {
        return Ok((vec![], out_shape));
    }

    let mut result = vec![0.0; out_size.max(1)];

    // Iterate over output indices
    for out_idx in 0..out_size.max(1) {
        let out_coords = idx_to_coords(out_idx, &out_shape);

        // Sum over contracted indices
        let mut sum = 0.0;
        for sum_idx in 0..sum_size.max(1) {
            let sum_coords = idx_to_coords(sum_idx, &sum_dims);

            // Build full index assignment
            let mut assignment: HashMap<char, usize> = HashMap::new();
            for (i, c) in sub_out.chars().enumerate() {
                if i < out_coords.len() {
                    assignment.insert(c, out_coords[i]);
                }
            }
            for (i, &c) in sum_indices.iter().enumerate() {
                if i < sum_coords.len() {
                    assignment.insert(c, sum_coords[i]);
                }
            }

            // Get a element
            let a_idx = subscript_to_flat_idx(sub_a, a_shape, &assignment);
            // Get b element
            let b_idx = subscript_to_flat_idx(sub_b, b_shape, &assignment);

            if a_idx < a.len() && b_idx < b.len() {
                sum += a[a_idx] * b[b_idx];
            }
        }
        if out_idx < result.len() {
            result[out_idx] = sum;
        }
    }

    Ok((result, out_shape))
}

/// Execute einsum with a single operand (trace, diagonal, transpose, etc.).
pub fn einsum1(
    subscripts: &str,
    a: &[f64],
    a_shape: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), EinsumError> {
    let (input_subs, output_sub) = parse_subscripts(subscripts)?;

    if input_subs.len() != 1 {
        return Err(EinsumError::MissingOperand {
            expected: 1,
            got: input_subs.len(),
        });
    }

    let sub_a = &input_subs[0];
    let sub_out = output_sub.unwrap_or_else(|| {
        // For single operand, implicit output keeps indices appearing once
        let mut counts: HashMap<char, usize> = HashMap::new();
        for c in sub_a.chars() {
            *counts.entry(c).or_insert(0) += 1;
        }
        let mut output_chars: Vec<char> = counts
            .iter()
            .filter(|&(_, &count)| count == 1)
            .map(|(&c, _)| c)
            .collect();
        output_chars.sort();
        output_chars.into_iter().collect()
    });

    if sub_a.len() != a_shape.len() {
        return Err(EinsumError::InvalidSubscript(format!(
            "subscript '{}' has {} indices but operand has {} dimensions",
            sub_a,
            sub_a.len(),
            a_shape.len()
        )));
    }

    // Build index->dimension mapping
    let mut index_dims: HashMap<char, usize> = HashMap::new();
    for (i, c) in sub_a.chars().enumerate() {
        if let Some(&existing) = index_dims.get(&c) {
            if existing != a_shape[i] {
                return Err(EinsumError::ShapeMismatch {
                    index: c,
                    expected: existing,
                    got: a_shape[i],
                });
            }
        } else {
            index_dims.insert(c, a_shape[i]);
        }
    }

    // Compute output shape
    let out_shape: Vec<usize> = sub_out.chars().map(|c| index_dims[&c]).collect();

    // Find summed indices (those not in output)
    let out_indices: HashSet<char> = sub_out.chars().collect();
    let sum_indices: Vec<char> = sub_a.chars().filter(|c| !out_indices.contains(c)).collect();
    let unique_sum: HashSet<char> = sum_indices.iter().copied().collect();
    let sum_dims: Vec<usize> = unique_sum.iter().map(|&c| index_dims[&c]).collect();
    let sum_size: usize = sum_dims.iter().product();

    let out_size: usize = out_shape.iter().product();
    if out_size == 0 {
        return Ok((vec![], out_shape));
    }

    let mut result = vec![0.0; out_size.max(1)];

    for out_idx in 0..out_size.max(1) {
        let out_coords = idx_to_coords(out_idx, &out_shape);

        let mut sum = 0.0;
        for sum_idx in 0..sum_size.max(1) {
            let sum_coords = idx_to_coords(sum_idx, &sum_dims);

            let mut assignment: HashMap<char, usize> = HashMap::new();
            for (i, c) in sub_out.chars().enumerate() {
                if i < out_coords.len() {
                    assignment.insert(c, out_coords[i]);
                }
            }
            let unique_sum_vec: Vec<char> = unique_sum.iter().copied().collect();
            for (i, &c) in unique_sum_vec.iter().enumerate() {
                if i < sum_coords.len() {
                    assignment.insert(c, sum_coords[i]);
                }
            }

            let a_idx = subscript_to_flat_idx(sub_a, a_shape, &assignment);
            if a_idx < a.len() {
                sum += a[a_idx];
            }
        }
        if out_idx < result.len() {
            result[out_idx] = sum;
        }
    }

    Ok((result, out_shape))
}

fn idx_to_coords(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut coords = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        if shape[i] > 0 {
            coords[i] = idx % shape[i];
            idx /= shape[i];
        }
    }
    coords
}

fn subscript_to_flat_idx(
    subscript: &str,
    shape: &[usize],
    assignment: &HashMap<char, usize>,
) -> usize {
    // Compute strides for row-major layout
    let chars: Vec<char> = subscript.chars().collect();
    let ndim = chars.len().min(shape.len());
    if ndim == 0 {
        return 0;
    }

    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut idx = 0;
    for (i, &c) in chars.iter().enumerate() {
        if i < ndim
            && let Some(&coord) = assignment.get(&c)
        {
            idx += coord * strides[i];
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn einsum_matrix_mul() {
        // "ij,jk->ik" is matrix multiplication
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [5.0, 6.0, 7.0, 8.0]; // 2x2
        let (result, shape) = einsum2("ij,jk->ik", &a, &[2, 2], &b, &[2, 2]).unwrap();
        assert_eq!(shape, vec![2, 2]);
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert!((result[0] - 19.0).abs() < 1e-10);
        assert!((result[1] - 22.0).abs() < 1e-10);
        assert!((result[2] - 43.0).abs() < 1e-10);
        assert!((result[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_dot_product() {
        // "i,i->" is dot product
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let (result, shape) = einsum2("i,i->", &a, &[3], &b, &[3]).unwrap();
        assert!(shape.is_empty());
        assert!((result[0] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_outer_product() {
        // "i,j->ij" is outer product
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0];
        let (result, shape) = einsum2("i,j->ij", &a, &[2], &b, &[3]).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn einsum_trace() {
        // "ii->" is trace
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let (result, shape) = einsum1("ii->", &a, &[2, 2]).unwrap();
        assert!(shape.is_empty());
        assert!((result[0] - 5.0).abs() < 1e-10); // 1 + 4 = 5
    }

    #[test]
    fn einsum_transpose() {
        // "ij->ji" is transpose
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let (result, shape) = einsum1("ij->ji", &a, &[2, 3]).unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn einsum_diagonal() {
        // "ii->i" extracts diagonal
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let (result, shape) = einsum1("ii->i", &a, &[2, 2]).unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(result, vec![1.0, 4.0]);
    }

    #[test]
    fn einsum_batch_matmul() {
        // "bij,bjk->bik" is batched matrix multiplication
        // 2 batches of 2x2 matrices
        let a = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]; // shape [2, 2, 2]
        let b = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]; // shape [2, 2, 2]
        let (result, shape) = einsum2("bij,bjk->bik", &a, &[2, 2, 2], &b, &[2, 2, 2]).unwrap();
        assert_eq!(shape, vec![2, 2, 2]);
        // First batch: I @ [[1,2],[3,4]] = [[1,2],[3,4]]
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_invalid_subscript() {
        let a = [1.0, 2.0];
        let result = einsum1("AB->", &a, &[2]);
        assert!(result.is_err());
    }

    #[test]
    fn einsum_shape_mismatch() {
        let a = [1.0, 2.0, 3.0]; // 3 elements
        let b = [1.0, 2.0]; // 2 elements
        let result = einsum2("i,i->", &a, &[3], &b, &[2]);
        assert!(matches!(result, Err(EinsumError::ShapeMismatch { .. })));
    }
}
