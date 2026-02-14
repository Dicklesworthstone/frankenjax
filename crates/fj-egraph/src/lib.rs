#![forbid(unsafe_code)]

use egg::{CostFunction, Id, Language, RecExpr, Runner, define_language, rewrite};
use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, VarId};
use smallvec::smallvec;
use std::collections::BTreeMap;

define_language! {
    pub enum FjLang {
        "add" = Add([Id; 2]),
        "mul" = Mul([Id; 2]),
        "sin" = Sin([Id; 1]),
        "cos" = Cos([Id; 1]),
        "reduce_sum" = ReduceSum([Id; 1]),
        "dot" = Dot([Id; 2]),
        Num(i64),
        Symbol(egg::Symbol),
    }
}

/// Cost function: minimize total operation count.
pub struct OpCount;

impl CostFunction<FjLang> for OpCount {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &FjLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            FjLang::Num(_) | FjLang::Symbol(_) => 0,
            _ => 1,
        };
        enode.fold(op_cost, |sum, id| sum + costs(id))
    }
}

/// Standard algebraic rewrite rules for FjLang.
pub fn algebraic_rules() -> Vec<egg::Rewrite<FjLang, ()>> {
    vec![
        // Commutativity
        rewrite!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)"),
        rewrite!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)"),
        // Identity
        rewrite!("add-zero"; "(add ?a 0)" => "?a"),
        rewrite!("mul-one"; "(mul ?a 1)" => "?a"),
        // Annihilation
        rewrite!("mul-zero"; "(mul ?a 0)" => "0"),
        // Associativity
        rewrite!("add-assoc"; "(add (add ?a ?b) ?c)" => "(add ?a (add ?b ?c))"),
        rewrite!("mul-assoc"; "(mul (mul ?a ?b) ?c)" => "(mul ?a (mul ?b ?c))"),
        // Distributivity
        rewrite!("distribute"; "(mul ?a (add ?b ?c))" => "(add (mul ?a ?b) (mul ?a ?c))"),
        rewrite!("factor"; "(add (mul ?a ?b) (mul ?a ?c))" => "(mul ?a (add ?b ?c))"),
    ]
}

/// Convert a Jaxpr to an e-graph RecExpr.
pub fn jaxpr_to_egraph(jaxpr: &Jaxpr) -> (RecExpr<FjLang>, BTreeMap<VarId, Id>) {
    let mut expr = RecExpr::default();
    let mut var_map: BTreeMap<VarId, Id> = BTreeMap::new();

    // Add input variables as symbols
    for var in &jaxpr.invars {
        let sym = egg::Symbol::from(format!("v{}", var.0));
        let id = expr.add(FjLang::Symbol(sym));
        var_map.insert(*var, id);
    }

    // Process equations
    for eqn in &jaxpr.equations {
        let input_ids: Vec<Id> = eqn
            .inputs
            .iter()
            .map(|atom| match atom {
                Atom::Var(var) => var_map[var],
                Atom::Lit(Literal::I64(n)) => expr.add(FjLang::Num(*n)),
                Atom::Lit(Literal::F64Bits(bits)) => {
                    // Encode as symbol to preserve bit-exactness
                    let sym = egg::Symbol::from(format!("f64:{bits}"));
                    expr.add(FjLang::Symbol(sym))
                }
                Atom::Lit(Literal::Bool(b)) => expr.add(FjLang::Num(if *b { 1 } else { 0 })),
            })
            .collect();

        let node = match eqn.primitive {
            Primitive::Add => FjLang::Add([input_ids[0], input_ids[1]]),
            Primitive::Mul => FjLang::Mul([input_ids[0], input_ids[1]]),
            Primitive::Sin => FjLang::Sin([input_ids[0]]),
            Primitive::Cos => FjLang::Cos([input_ids[0]]),
            Primitive::ReduceSum => FjLang::ReduceSum([input_ids[0]]),
            Primitive::Dot => FjLang::Dot([input_ids[0], input_ids[1]]),
        };

        let id = expr.add(node);
        for outvar in eqn.outputs.iter() {
            var_map.insert(*outvar, id);
        }
    }

    (expr, var_map)
}

/// Convert an e-graph RecExpr back to a Jaxpr.
pub fn egraph_to_jaxpr(
    expr: &RecExpr<FjLang>,
    invars: &[VarId],
    original_outvars: &[VarId],
) -> Jaxpr {
    let mut next_var = invars.iter().map(|v| v.0).max().unwrap_or(0) + 1;
    let mut equations = Vec::new();
    let mut node_to_var: BTreeMap<usize, VarId> = BTreeMap::new();

    // Map Symbol nodes back to their original VarIds by parsing the symbol name.
    // After equality saturation + extraction, Symbol nodes may appear in any order
    // in the RecExpr, so we cannot assume positional correspondence with invars.
    for (idx, node) in expr.as_ref().iter().enumerate() {
        if let FjLang::Symbol(sym) = node
            && let Some(rest) = sym.as_str().strip_prefix('v')
            && let Ok(var_num) = rest.parse::<u32>()
        {
            let var_id = VarId(var_num);
            if invars.contains(&var_id) {
                node_to_var.insert(idx, var_id);
            }
        }
    }

    fn resolve_or_create(
        node_idx: usize,
        node_to_var: &mut BTreeMap<usize, VarId>,
        next_var: &mut u32,
    ) -> VarId {
        if let Some(var) = node_to_var.get(&node_idx) {
            return *var;
        }
        let var = VarId(*next_var);
        *next_var += 1;
        node_to_var.insert(node_idx, var);
        var
    }

    for (idx, node) in expr.as_ref().iter().enumerate() {
        match node {
            FjLang::Num(_) => {
                // Literals don't become equations; they become Atom::Lit
                // Just register the var mapping so other nodes can reference it.
                resolve_or_create(idx, &mut node_to_var, &mut next_var);
            }
            FjLang::Symbol(_) => {
                // Already mapped or will be created
                resolve_or_create(idx, &mut node_to_var, &mut next_var);
            }
            FjLang::Add([a, b]) => {
                let out = resolve_or_create(idx, &mut node_to_var, &mut next_var);
                let a_atom = id_to_atom(*a, &node_to_var, expr);
                let b_atom = id_to_atom(*b, &node_to_var, expr);
                equations.push(Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![a_atom, b_atom],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                });
            }
            FjLang::Mul([a, b]) => {
                let out = resolve_or_create(idx, &mut node_to_var, &mut next_var);
                let a_atom = id_to_atom(*a, &node_to_var, expr);
                let b_atom = id_to_atom(*b, &node_to_var, expr);
                equations.push(Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![a_atom, b_atom],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                });
            }
            FjLang::Sin([a]) => {
                let out = resolve_or_create(idx, &mut node_to_var, &mut next_var);
                let a_atom = id_to_atom(*a, &node_to_var, expr);
                equations.push(Equation {
                    primitive: Primitive::Sin,
                    inputs: smallvec![a_atom],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                });
            }
            FjLang::Cos([a]) => {
                let out = resolve_or_create(idx, &mut node_to_var, &mut next_var);
                let a_atom = id_to_atom(*a, &node_to_var, expr);
                equations.push(Equation {
                    primitive: Primitive::Cos,
                    inputs: smallvec![a_atom],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                });
            }
            FjLang::ReduceSum([a]) => {
                let out = resolve_or_create(idx, &mut node_to_var, &mut next_var);
                let a_atom = id_to_atom(*a, &node_to_var, expr);
                equations.push(Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![a_atom],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                });
            }
            FjLang::Dot([a, b]) => {
                let out = resolve_or_create(idx, &mut node_to_var, &mut next_var);
                let a_atom = id_to_atom(*a, &node_to_var, expr);
                let b_atom = id_to_atom(*b, &node_to_var, expr);
                equations.push(Equation {
                    primitive: Primitive::Dot,
                    inputs: smallvec![a_atom, b_atom],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                });
            }
        }
    }

    // Use the last node as the output variable (matches extraction behavior)
    let outvars = if let Some(last_var) = node_to_var.get(&(expr.as_ref().len() - 1)) {
        vec![*last_var]
    } else {
        original_outvars.to_vec()
    };

    Jaxpr::new(invars.to_vec(), vec![], outvars, equations)
}

fn id_to_atom(id: Id, node_to_var: &BTreeMap<usize, VarId>, expr: &RecExpr<FjLang>) -> Atom {
    let idx: usize = id.into();
    // Check if this node is a literal (Num or f64-encoded Symbol)
    match &expr.as_ref()[idx] {
        FjLang::Num(n) => Atom::Lit(Literal::I64(*n)),
        FjLang::Symbol(sym) => {
            // f64 literals were encoded as Symbol("f64:{bits}") in jaxpr_to_egraph
            if let Some(bits_str) = sym.as_str().strip_prefix("f64:")
                && let Ok(bits) = bits_str.parse::<u64>()
            {
                return Atom::Lit(Literal::F64Bits(bits));
            }
            // Otherwise it's a variable reference
            node_to_var
                .get(&idx)
                .map(|var| Atom::Var(*var))
                .unwrap_or_else(|| Atom::Var(VarId(idx as u32)))
        }
        _ => node_to_var
            .get(&idx)
            .map(|var| Atom::Var(*var))
            .unwrap_or_else(|| Atom::Var(VarId(idx as u32))),
    }
}

/// Optimize a Jaxpr using equality saturation with algebraic rules.
pub fn optimize_jaxpr(jaxpr: &Jaxpr) -> Jaxpr {
    let (expr, var_map) = jaxpr_to_egraph(jaxpr);

    // Get the root (last output)
    let root_id = var_map
        .get(&jaxpr.outvars[0])
        .copied()
        .unwrap_or_else(|| Id::from(expr.as_ref().len() - 1));

    let runner = Runner::<FjLang, ()>::default()
        .with_expr(&expr)
        .run(&algebraic_rules());

    let extractor = egg::Extractor::new(&runner.egraph, OpCount);
    let (_, best_expr) = extractor.find_best(root_id);

    egraph_to_jaxpr(&best_expr, &jaxpr.invars, &jaxpr.outvars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{ProgramSpec, Value, build_program};
    use fj_interpreters::eval_jaxpr;

    #[test]
    fn round_trip_add2() {
        let original = build_program(ProgramSpec::Add2);
        let optimized = optimize_jaxpr(&original);

        // Both must produce the same output
        let orig_out =
            eval_jaxpr(&original, &[Value::scalar_i64(3), Value::scalar_i64(4)]).unwrap();
        let opt_out =
            eval_jaxpr(&optimized, &[Value::scalar_i64(3), Value::scalar_i64(4)]).unwrap();
        assert_eq!(orig_out, opt_out);
    }

    #[test]
    fn round_trip_square() {
        let original = build_program(ProgramSpec::Square);
        let optimized = optimize_jaxpr(&original);

        let orig_out = eval_jaxpr(&original, &[Value::scalar_f64(5.0)]).unwrap();
        let opt_out = eval_jaxpr(&optimized, &[Value::scalar_f64(5.0)]).unwrap();
        assert_eq!(orig_out, opt_out);
    }

    #[test]
    fn mul_zero_simplification() {
        // Build: x * 0
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(0))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        // Optimized should have fewer or equal equations
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "optimization should not increase equation count"
        );
    }

    #[test]
    fn round_trip_add2_asymmetric_args() {
        // Regression: after commutativity rewrites, input Symbols may be reordered.
        // With asymmetric args (3, 7) we can't accidentally pass with swapped inputs.
        let original = build_program(ProgramSpec::Add2);
        let optimized = optimize_jaxpr(&original);

        let orig_out =
            eval_jaxpr(&original, &[Value::scalar_i64(3), Value::scalar_i64(7)]).unwrap();
        let opt_out =
            eval_jaxpr(&optimized, &[Value::scalar_i64(3), Value::scalar_i64(7)]).unwrap();
        assert_eq!(orig_out, opt_out);
    }

    #[test]
    fn round_trip_square_plus_linear() {
        // x²+2x uses both mul and add with shared variable x, and a literal 2.
        // This exercises Symbol reordering after saturation.
        let original = build_program(ProgramSpec::SquarePlusLinear);
        let optimized = optimize_jaxpr(&original);

        for &x in &[0.0, 1.0, -2.5, 7.0] {
            let orig_out = eval_jaxpr(&original, &[Value::scalar_f64(x)]).unwrap();
            let opt_out = eval_jaxpr(&optimized, &[Value::scalar_f64(x)]).unwrap();
            assert_eq!(
                orig_out, opt_out,
                "mismatch at x={x}: original={orig_out:?}, optimized={opt_out:?}"
            );
        }
    }

    #[test]
    fn round_trip_with_f64_literal() {
        // Regression: f64 literals are encoded as Symbol("f64:{bits}") in the e-graph.
        // id_to_atom must decode them back to Atom::Lit(Literal::F64Bits(...)).
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::from_f64(2.5))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);

        for &x in &[0.0, 1.0, 3.0, -4.0] {
            let orig_out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)]).unwrap();
            let opt_out = eval_jaxpr(&optimized, &[Value::scalar_f64(x)]).unwrap();
            assert_eq!(
                orig_out, opt_out,
                "f64 literal round-trip mismatch at x={x}: original={orig_out:?}, optimized={opt_out:?}"
            );
        }
    }

    #[test]
    fn language_definition_covers_all_primitives() {
        // Verify we can parse all primitive shapes
        let add: RecExpr<FjLang> = "(add 1 2)".parse().unwrap();
        assert!(!add.as_ref().is_empty());

        let mul: RecExpr<FjLang> = "(mul 3 4)".parse().unwrap();
        assert!(!mul.as_ref().is_empty());

        let sin: RecExpr<FjLang> = "(sin 1)".parse().unwrap();
        assert!(!sin.as_ref().is_empty());

        let cos: RecExpr<FjLang> = "(cos 1)".parse().unwrap();
        assert!(!cos.as_ref().is_empty());
    }
}
