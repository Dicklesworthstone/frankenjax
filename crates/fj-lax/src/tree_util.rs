//! PyTree utilities matching JAX's jax.tree_util module.
//!
//! Provides tree traversal and manipulation for nested data structures.
//! In this implementation, trees are represented as nested Vec<TreeNode> structures.

use std::collections::HashMap;

/// A node in a PyTree structure.
#[derive(Debug, Clone, PartialEq)]
pub enum TreeNode {
    /// A leaf node containing a value (represented as f64 for simplicity)
    Leaf(f64),
    /// A tuple/list node containing children
    Tuple(Vec<TreeNode>),
    /// A dict node with string keys
    Dict(HashMap<String, TreeNode>),
    /// None/null value
    None,
}

/// Structure information for a tree (for tree_unflatten).
#[derive(Debug, Clone, PartialEq)]
pub enum TreeDef {
    Leaf,
    Tuple(Vec<TreeDef>),
    Dict(Vec<(String, TreeDef)>),
    None,
}

/// Flatten a tree into a list of leaves and a tree structure definition.
///
/// Matches `jax.tree_util.tree_flatten(tree)`.
pub fn tree_flatten(tree: &TreeNode) -> (Vec<f64>, TreeDef) {
    let mut leaves = Vec::new();
    let def = flatten_recursive(tree, &mut leaves);
    (leaves, def)
}

fn flatten_recursive(node: &TreeNode, leaves: &mut Vec<f64>) -> TreeDef {
    match node {
        TreeNode::Leaf(v) => {
            leaves.push(*v);
            TreeDef::Leaf
        }
        TreeNode::Tuple(children) => {
            let child_defs: Vec<TreeDef> = children
                .iter()
                .map(|c| flatten_recursive(c, leaves))
                .collect();
            TreeDef::Tuple(child_defs)
        }
        TreeNode::Dict(map) => {
            let mut items: Vec<(String, TreeDef)> = map
                .iter()
                .map(|(k, v)| (k.clone(), flatten_recursive(v, leaves)))
                .collect();
            // Sort by key for deterministic order
            items.sort_by(|a, b| a.0.cmp(&b.0));
            TreeDef::Dict(items)
        }
        TreeNode::None => TreeDef::None,
    }
}

/// Reconstruct a tree from leaves and structure definition.
///
/// Matches `jax.tree_util.tree_unflatten(treedef, leaves)`.
pub fn tree_unflatten(treedef: &TreeDef, leaves: &[f64]) -> (TreeNode, usize) {
    let (tree, consumed) = unflatten_recursive(treedef, leaves, 0);
    assert!(
        consumed == leaves.len(),
        "tree_unflatten leaf overflow: treedef consumed {consumed} leaves but {} provided",
        leaves.len()
    );
    (tree, consumed)
}

fn unflatten_recursive(def: &TreeDef, leaves: &[f64], idx: usize) -> (TreeNode, usize) {
    match def {
        TreeDef::Leaf => {
            let value = leaves.get(idx).copied().unwrap_or_else(|| {
                panic!("tree_unflatten leaf underflow: treedef requires more leaves than provided")
            });
            (TreeNode::Leaf(value), idx + 1)
        }
        TreeDef::Tuple(child_defs) => {
            let mut children = Vec::with_capacity(child_defs.len());
            let mut current_idx = idx;
            for child_def in child_defs {
                let (child, new_idx) = unflatten_recursive(child_def, leaves, current_idx);
                children.push(child);
                current_idx = new_idx;
            }
            (TreeNode::Tuple(children), current_idx)
        }
        TreeDef::Dict(items) => {
            let mut map = HashMap::new();
            let mut current_idx = idx;
            for (key, child_def) in items {
                let (child, new_idx) = unflatten_recursive(child_def, leaves, current_idx);
                map.insert(key.clone(), child);
                current_idx = new_idx;
            }
            (TreeNode::Dict(map), current_idx)
        }
        TreeDef::None => (TreeNode::None, idx),
    }
}

/// Get all leaves from a tree.
///
/// Matches `jax.tree_util.tree_leaves(tree)`.
pub fn tree_leaves(tree: &TreeNode) -> Vec<f64> {
    let (leaves, _) = tree_flatten(tree);
    leaves
}

/// Get the structure definition of a tree.
///
/// Matches `jax.tree_util.tree_structure(tree)`.
pub fn tree_structure(tree: &TreeNode) -> TreeDef {
    let (_, def) = tree_flatten(tree);
    def
}

/// Apply a function to each leaf of a tree.
///
/// Matches `jax.tree_util.tree_map(f, tree)`.
pub fn tree_map<F>(f: F, tree: &TreeNode) -> TreeNode
where
    F: Fn(f64) -> f64 + Copy,
{
    match tree {
        TreeNode::Leaf(v) => TreeNode::Leaf(f(*v)),
        TreeNode::Tuple(children) => {
            TreeNode::Tuple(children.iter().map(|c| tree_map(f, c)).collect())
        }
        TreeNode::Dict(map) => TreeNode::Dict(
            map.iter()
                .map(|(k, v)| (k.clone(), tree_map(f, v)))
                .collect(),
        ),
        TreeNode::None => TreeNode::None,
    }
}

/// Apply a function to corresponding leaves of multiple trees.
///
/// Matches `jax.tree_util.tree_map(f, tree, *rest)` for two trees.
pub fn tree_map2<F>(f: F, tree1: &TreeNode, tree2: &TreeNode) -> TreeNode
where
    F: Fn(f64, f64) -> f64 + Copy,
{
    match (tree1, tree2) {
        (TreeNode::Leaf(v1), TreeNode::Leaf(v2)) => TreeNode::Leaf(f(*v1, *v2)),
        (TreeNode::Tuple(c1), TreeNode::Tuple(c2)) => {
            assert!(
                c1.len() == c2.len(),
                "tree_map2 tuple arity mismatch: left has {} children, right has {}",
                c1.len(),
                c2.len()
            );
            let children: Vec<TreeNode> = c1
                .iter()
                .zip(c2.iter())
                .map(|(a, b)| tree_map2(f, a, b))
                .collect();
            TreeNode::Tuple(children)
        }
        (TreeNode::Dict(m1), TreeNode::Dict(m2)) => {
            assert!(
                m1.len() == m2.len(),
                "tree_map2 dict key mismatch: left has {} keys, right has {}",
                m1.len(),
                m2.len()
            );
            let mut map = HashMap::new();
            for (k, v1) in m1 {
                let v2 = m2.get(k).unwrap_or_else(|| {
                    panic!("tree_map2 dict key mismatch: right tree missing key {k:?}")
                });
                map.insert(k.clone(), tree_map2(f, v1, v2));
            }
            for k in m2.keys() {
                assert!(
                    m1.contains_key(k),
                    "tree_map2 dict key mismatch: right tree has extra key {k:?}"
                );
            }
            TreeNode::Dict(map)
        }
        (TreeNode::None, TreeNode::None) => TreeNode::None,
        _ => panic!(
            "tree_map2 structure mismatch: left is {}, right is {}",
            tree_node_kind(tree1),
            tree_node_kind(tree2)
        ),
    }
}

fn tree_node_kind(node: &TreeNode) -> &'static str {
    match node {
        TreeNode::Leaf(_) => "leaf",
        TreeNode::Tuple(_) => "tuple",
        TreeNode::Dict(_) => "dict",
        TreeNode::None => "none",
    }
}

/// Reduce a tree to a single value.
///
/// Matches `jax.tree_util.tree_reduce(f, tree, initializer)`.
pub fn tree_reduce<F>(f: F, tree: &TreeNode, initializer: f64) -> f64
where
    F: Fn(f64, f64) -> f64 + Copy,
{
    let leaves = tree_leaves(tree);
    leaves.into_iter().fold(initializer, f)
}

/// Check if all leaves satisfy a predicate.
///
/// Matches `jax.tree_util.tree_all(tree, is_leaf)`.
pub fn tree_all<F>(tree: &TreeNode, predicate: F) -> bool
where
    F: Fn(f64) -> bool,
{
    let leaves = tree_leaves(tree);
    leaves.into_iter().all(predicate)
}

/// Check if any leaf satisfies a predicate.
pub fn tree_any<F>(tree: &TreeNode, predicate: F) -> bool
where
    F: Fn(f64) -> bool,
{
    let leaves = tree_leaves(tree);
    leaves.into_iter().any(predicate)
}

/// Count the number of leaves in a tree.
pub fn tree_leaf_count(tree: &TreeNode) -> usize {
    tree_leaves(tree).len()
}

/// Create a tree filled with zeros matching the structure of another tree.
pub fn tree_zeros_like(tree: &TreeNode) -> TreeNode {
    tree_map(|_| 0.0, tree)
}

/// Create a tree filled with ones matching the structure of another tree.
pub fn tree_ones_like(tree: &TreeNode) -> TreeNode {
    tree_map(|_| 1.0, tree)
}

/// Add two trees element-wise.
pub fn tree_add(tree1: &TreeNode, tree2: &TreeNode) -> TreeNode {
    tree_map2(|a, b| a + b, tree1, tree2)
}

/// Subtract two trees element-wise.
pub fn tree_sub(tree1: &TreeNode, tree2: &TreeNode) -> TreeNode {
    tree_map2(|a, b| a - b, tree1, tree2)
}

/// Multiply two trees element-wise.
pub fn tree_mul(tree1: &TreeNode, tree2: &TreeNode) -> TreeNode {
    tree_map2(|a, b| a * b, tree1, tree2)
}

/// Scale a tree by a scalar.
pub fn tree_scalar_mul(scalar: f64, tree: &TreeNode) -> TreeNode {
    tree_map(|v| scalar * v, tree)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_tree() -> TreeNode {
        TreeNode::Tuple(vec![
            TreeNode::Leaf(1.0),
            TreeNode::Leaf(2.0),
            TreeNode::Tuple(vec![TreeNode::Leaf(3.0), TreeNode::Leaf(4.0)]),
        ])
    }

    #[test]
    fn test_tree_flatten_leaves() {
        let tree = make_simple_tree();
        let (leaves, _) = tree_flatten(&tree);
        assert_eq!(leaves, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tree_unflatten_roundtrip() {
        let tree = make_simple_tree();
        let (leaves, def) = tree_flatten(&tree);
        let (reconstructed, _) = tree_unflatten(&def, &leaves);
        assert_eq!(reconstructed, tree);
    }

    #[test]
    #[should_panic(expected = "tree_unflatten leaf underflow")]
    fn test_tree_unflatten_rejects_missing_leaves() {
        let def = TreeDef::Tuple(vec![TreeDef::Leaf, TreeDef::Leaf]);
        let _ = tree_unflatten(&def, &[1.0]);
    }

    #[test]
    #[should_panic(expected = "tree_unflatten leaf overflow")]
    fn test_tree_unflatten_rejects_extra_leaves() {
        let def = TreeDef::Leaf;
        let _ = tree_unflatten(&def, &[1.0, 2.0]);
    }

    #[test]
    fn test_tree_leaves() {
        let tree = make_simple_tree();
        let leaves = tree_leaves(&tree);
        assert_eq!(leaves, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tree_map() {
        let tree = TreeNode::Tuple(vec![TreeNode::Leaf(1.0), TreeNode::Leaf(2.0)]);
        let doubled = tree_map(|x| x * 2.0, &tree);
        let leaves = tree_leaves(&doubled);
        assert_eq!(leaves, vec![2.0, 4.0]);
    }

    #[test]
    fn test_tree_map2() {
        let tree1 = TreeNode::Tuple(vec![TreeNode::Leaf(1.0), TreeNode::Leaf(2.0)]);
        let tree2 = TreeNode::Tuple(vec![TreeNode::Leaf(3.0), TreeNode::Leaf(4.0)]);
        let sum = tree_map2(|a, b| a + b, &tree1, &tree2);
        let leaves = tree_leaves(&sum);
        assert_eq!(leaves, vec![4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "tree_map2 tuple arity mismatch")]
    fn test_tree_map2_rejects_tuple_arity_mismatch() {
        let tree1 = TreeNode::Tuple(vec![TreeNode::Leaf(1.0), TreeNode::Leaf(2.0)]);
        let tree2 = TreeNode::Tuple(vec![TreeNode::Leaf(3.0)]);
        let _ = tree_map2(|a, b| a + b, &tree1, &tree2);
    }

    #[test]
    #[should_panic(expected = "tree_map2 dict key mismatch")]
    fn test_tree_map2_rejects_dict_key_mismatch() {
        let mut left = HashMap::new();
        left.insert("a".to_string(), TreeNode::Leaf(1.0));
        let mut right = HashMap::new();
        right.insert("b".to_string(), TreeNode::Leaf(2.0));
        let _ = tree_map2(
            |a, b| a + b,
            &TreeNode::Dict(left),
            &TreeNode::Dict(right),
        );
    }

    #[test]
    #[should_panic(expected = "tree_map2 structure mismatch")]
    fn test_tree_map2_rejects_node_kind_mismatch() {
        let _ = tree_map2(|a, b| a + b, &TreeNode::Leaf(1.0), &TreeNode::None);
    }

    #[test]
    fn test_tree_reduce() {
        let tree = make_simple_tree();
        let sum = tree_reduce(|acc, x| acc + x, &tree, 0.0);
        assert!((sum - 10.0).abs() < 1e-10); // 1 + 2 + 3 + 4 = 10
    }

    #[test]
    fn test_tree_all() {
        let tree = TreeNode::Tuple(vec![TreeNode::Leaf(1.0), TreeNode::Leaf(2.0)]);
        assert!(tree_all(&tree, |x| x > 0.0));
        assert!(!tree_all(&tree, |x| x > 1.5));
    }

    #[test]
    fn test_tree_any() {
        let tree = TreeNode::Tuple(vec![TreeNode::Leaf(1.0), TreeNode::Leaf(2.0)]);
        assert!(tree_any(&tree, |x| x > 1.5));
        assert!(!tree_any(&tree, |x| x > 10.0));
    }

    #[test]
    fn test_tree_leaf_count() {
        let tree = make_simple_tree();
        assert_eq!(tree_leaf_count(&tree), 4);
    }

    #[test]
    fn test_tree_zeros_like() {
        let tree = make_simple_tree();
        let zeros = tree_zeros_like(&tree);
        let leaves = tree_leaves(&zeros);
        assert!(leaves.iter().all(|&v| v == 0.0));
        assert_eq!(leaves.len(), 4);
    }

    #[test]
    fn test_tree_ones_like() {
        let tree = make_simple_tree();
        let ones = tree_ones_like(&tree);
        let leaves = tree_leaves(&ones);
        assert!(leaves.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_tree_add() {
        let tree1 = TreeNode::Tuple(vec![TreeNode::Leaf(1.0), TreeNode::Leaf(2.0)]);
        let tree2 = TreeNode::Tuple(vec![TreeNode::Leaf(3.0), TreeNode::Leaf(4.0)]);
        let sum = tree_add(&tree1, &tree2);
        let leaves = tree_leaves(&sum);
        assert_eq!(leaves, vec![4.0, 6.0]);
    }

    #[test]
    fn test_tree_scalar_mul() {
        let tree = TreeNode::Tuple(vec![TreeNode::Leaf(1.0), TreeNode::Leaf(2.0)]);
        let scaled = tree_scalar_mul(3.0, &tree);
        let leaves = tree_leaves(&scaled);
        assert_eq!(leaves, vec![3.0, 6.0]);
    }

    #[test]
    fn test_tree_with_dict() {
        let mut map = HashMap::new();
        map.insert("a".to_string(), TreeNode::Leaf(1.0));
        map.insert("b".to_string(), TreeNode::Leaf(2.0));
        let tree = TreeNode::Dict(map);

        let (leaves, def) = tree_flatten(&tree);
        assert_eq!(leaves.len(), 2);

        let (reconstructed, _) = tree_unflatten(&def, &leaves);
        if let TreeNode::Dict(rmap) = &reconstructed {
            assert_eq!(rmap.len(), 2);
        } else {
            panic!("expected dict");
        }
    }

    #[test]
    fn test_tree_none() {
        let tree = TreeNode::None;
        let leaves = tree_leaves(&tree);
        assert!(leaves.is_empty());

        let (_, def) = tree_flatten(&tree);
        let (reconstructed, _) = tree_unflatten(&def, &[]);
        assert_eq!(reconstructed, TreeNode::None);
    }
}
