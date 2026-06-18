//! Oracle conformance tests for jax.tree_util functions.
//!
//! Reference values computed with JAX:
//! ```python
//! import jax
//! from jax import tree_util
//! tree_util.tree_flatten({'a': 1.0, 'b': [2.0, 3.0]})
//! tree_util.tree_leaves({'a': 1.0, 'b': [2.0, 3.0]})
//! tree_util.tree_map(lambda x: x * 2, {'a': 1.0, 'b': 2.0})
//! ```

use fj_lax::tree_util::{
    TreeDef, TreeNode, tree_add, tree_all, tree_any, tree_flatten, tree_leaf_count, tree_leaves,
    tree_map, tree_map2, tree_mul, tree_ones_like, tree_reduce, tree_scalar_mul, tree_structure,
    tree_sub, tree_unflatten, tree_zeros_like,
};
use std::collections::HashMap;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn vec_approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y, tol))
}

fn make_simple_tree() -> TreeNode {
    TreeNode::Tuple(vec![
        TreeNode::Leaf(1.0),
        TreeNode::Leaf(2.0),
        TreeNode::Leaf(3.0),
    ])
}

fn make_nested_tree() -> TreeNode {
    TreeNode::Tuple(vec![
        TreeNode::Leaf(1.0),
        TreeNode::Tuple(vec![TreeNode::Leaf(2.0), TreeNode::Leaf(3.0)]),
        TreeNode::Leaf(4.0),
    ])
}

fn make_dict_tree() -> TreeNode {
    let mut map = HashMap::new();
    map.insert("a".to_string(), TreeNode::Leaf(1.0));
    map.insert("b".to_string(), TreeNode::Leaf(2.0));
    TreeNode::Dict(map)
}

#[test]
fn test_tree_flatten_simple() {
    let tree = make_simple_tree();
    let (leaves, _def) = tree_flatten(&tree);
    assert_eq!(leaves.len(), 3);
    assert!(vec_approx_eq(&leaves, &[1.0, 2.0, 3.0], 1e-10));
}

#[test]
fn test_tree_flatten_nested() {
    let tree = make_nested_tree();
    let (leaves, _def) = tree_flatten(&tree);
    assert_eq!(leaves.len(), 4);
    assert!(vec_approx_eq(&leaves, &[1.0, 2.0, 3.0, 4.0], 1e-10));
}

#[test]
fn test_tree_flatten_dict() {
    let mut map = HashMap::new();
    map.insert("delta".to_string(), TreeNode::Leaf(4.0));
    map.insert("alpha".to_string(), TreeNode::Leaf(1.0));
    map.insert("charlie".to_string(), TreeNode::Leaf(3.0));
    map.insert("bravo".to_string(), TreeNode::Leaf(2.0));
    let tree = TreeNode::Dict(map);

    let (leaves, def) = tree_flatten(&tree);
    let TreeDef::Dict(items) = &def else {
        panic!("expected dict treedef");
    };
    let keys: Vec<&str> = items.iter().map(|(key, _)| key.as_str()).collect();
    assert_eq!(keys, vec!["alpha", "bravo", "charlie", "delta"]);
    assert!(vec_approx_eq(&leaves, &[1.0, 2.0, 3.0, 4.0], 1e-10));

    let (reconstructed, consumed) = tree_unflatten(&def, &leaves);
    assert_eq!(consumed, leaves.len());
    assert_eq!(reconstructed, tree);
}

#[test]
fn test_tree_unflatten_roundtrip() {
    let tree = make_nested_tree();
    let (leaves, def) = tree_flatten(&tree);
    let (reconstructed, _consumed) = tree_unflatten(&def, &leaves);
    assert_eq!(tree, reconstructed);
}

#[test]
fn test_tree_leaves_simple() {
    let tree = make_simple_tree();
    let leaves = tree_leaves(&tree);
    assert!(vec_approx_eq(&leaves, &[1.0, 2.0, 3.0], 1e-10));
}

#[test]
fn test_tree_leaves_nested() {
    let tree = make_nested_tree();
    let leaves = tree_leaves(&tree);
    assert!(vec_approx_eq(&leaves, &[1.0, 2.0, 3.0, 4.0], 1e-10));
}

#[test]
fn test_tree_structure_simple() {
    let tree = make_simple_tree();
    let def = tree_structure(&tree);
    match def {
        TreeDef::Tuple(children) => {
            assert_eq!(children.len(), 3);
            assert!(matches!(children[0], TreeDef::Leaf));
            assert!(matches!(children[1], TreeDef::Leaf));
            assert!(matches!(children[2], TreeDef::Leaf));
        }
        _ => panic!("expected Tuple structure"),
    }
}

#[test]
fn test_tree_map_double() {
    let tree = make_simple_tree();
    let doubled = tree_map(|x| x * 2.0, &tree);
    let leaves = tree_leaves(&doubled);
    assert!(vec_approx_eq(&leaves, &[2.0, 4.0, 6.0], 1e-10));
}

#[test]
fn test_tree_map_nested() {
    let tree = make_nested_tree();
    let squared = tree_map(|x| x * x, &tree);
    let leaves = tree_leaves(&squared);
    assert!(vec_approx_eq(&leaves, &[1.0, 4.0, 9.0, 16.0], 1e-10));
}

#[test]
fn test_tree_map2_add() {
    let tree1 = make_simple_tree();
    let tree2 = make_simple_tree();
    let sum = tree_map2(|a, b| a + b, &tree1, &tree2);
    let leaves = tree_leaves(&sum);
    assert!(vec_approx_eq(&leaves, &[2.0, 4.0, 6.0], 1e-10));
}

#[test]
fn test_tree_reduce_sum() {
    let tree = make_simple_tree();
    let sum = tree_reduce(|acc, x| acc + x, &tree, 0.0);
    assert!(approx_eq(sum, 6.0, 1e-10));
}

#[test]
fn test_tree_reduce_product() {
    let tree = make_simple_tree();
    let prod = tree_reduce(|acc, x| acc * x, &tree, 1.0);
    assert!(approx_eq(prod, 6.0, 1e-10));
}

#[test]
fn test_tree_all_positive() {
    let tree = make_simple_tree();
    assert!(tree_all(&tree, |x| x > 0.0));
}

#[test]
fn test_tree_all_greater_than_two() {
    let tree = make_simple_tree();
    assert!(!tree_all(&tree, |x| x > 2.0));
}

#[test]
fn test_tree_any_greater_than_two() {
    let tree = make_simple_tree();
    assert!(tree_any(&tree, |x| x > 2.0));
}

#[test]
fn test_tree_any_negative() {
    let tree = make_simple_tree();
    assert!(!tree_any(&tree, |x| x < 0.0));
}

#[test]
fn test_tree_leaf_count() {
    let tree = make_simple_tree();
    assert_eq!(tree_leaf_count(&tree), 3);
}

#[test]
fn test_tree_leaf_count_nested() {
    let tree = make_nested_tree();
    assert_eq!(tree_leaf_count(&tree), 4);
}

#[test]
fn test_tree_zeros_like() {
    let tree = make_simple_tree();
    let zeros = tree_zeros_like(&tree);
    let leaves = tree_leaves(&zeros);
    assert!(vec_approx_eq(&leaves, &[0.0, 0.0, 0.0], 1e-10));
}

#[test]
fn test_tree_ones_like() {
    let tree = make_simple_tree();
    let ones = tree_ones_like(&tree);
    let leaves = tree_leaves(&ones);
    assert!(vec_approx_eq(&leaves, &[1.0, 1.0, 1.0], 1e-10));
}

#[test]
fn test_tree_add() {
    let tree1 = make_simple_tree();
    let tree2 = make_simple_tree();
    let sum = tree_add(&tree1, &tree2);
    let leaves = tree_leaves(&sum);
    assert!(vec_approx_eq(&leaves, &[2.0, 4.0, 6.0], 1e-10));
}

#[test]
fn test_tree_sub() {
    let tree1 = make_simple_tree();
    let tree2 = make_simple_tree();
    let diff = tree_sub(&tree1, &tree2);
    let leaves = tree_leaves(&diff);
    assert!(vec_approx_eq(&leaves, &[0.0, 0.0, 0.0], 1e-10));
}

#[test]
fn test_tree_mul() {
    let tree1 = make_simple_tree();
    let tree2 = make_simple_tree();
    let prod = tree_mul(&tree1, &tree2);
    let leaves = tree_leaves(&prod);
    assert!(vec_approx_eq(&leaves, &[1.0, 4.0, 9.0], 1e-10));
}

#[test]
fn test_tree_scalar_mul() {
    let tree = make_simple_tree();
    let scaled = tree_scalar_mul(2.0, &tree);
    let leaves = tree_leaves(&scaled);
    assert!(vec_approx_eq(&leaves, &[2.0, 4.0, 6.0], 1e-10));
}

#[test]
fn test_tree_none_handling() {
    let tree = TreeNode::Tuple(vec![
        TreeNode::Leaf(1.0),
        TreeNode::None,
        TreeNode::Leaf(2.0),
    ]);
    let leaves = tree_leaves(&tree);
    assert!(vec_approx_eq(&leaves, &[1.0, 2.0], 1e-10));
}

#[test]
fn test_tree_empty_tuple() {
    let tree = TreeNode::Tuple(vec![]);
    let (leaves, def) = tree_flatten(&tree);
    assert!(leaves.is_empty());
    match def {
        TreeDef::Tuple(children) => assert!(children.is_empty()),
        _ => panic!("expected empty Tuple"),
    }
}

#[test]
fn test_tree_single_leaf() {
    let tree = TreeNode::Leaf(42.0);
    let (leaves, def) = tree_flatten(&tree);
    assert!(vec_approx_eq(&leaves, &[42.0], 1e-10));
    assert!(matches!(def, TreeDef::Leaf));
}

#[test]
fn test_tree_deeply_nested() {
    let tree = TreeNode::Tuple(vec![TreeNode::Tuple(vec![TreeNode::Tuple(vec![
        TreeNode::Leaf(1.0),
    ])])]);
    let leaves = tree_leaves(&tree);
    assert!(vec_approx_eq(&leaves, &[1.0], 1e-10));
}
