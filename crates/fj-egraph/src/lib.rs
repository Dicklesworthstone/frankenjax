#![forbid(unsafe_code)]

use egg::{CostFunction, Id, Language, RecExpr, Runner, define_language, rewrite};
use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, VarId};
use smallvec::smallvec;
use std::collections::BTreeMap;

define_language! {
    pub enum FjLang {
        // Arithmetic (binary)
        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "mul" = Mul([Id; 2]),
        "div" = Div([Id; 2]),
        "rem" = Rem([Id; 2]),
        "pow" = Pow([Id; 2]),
        "max" = Max([Id; 2]),
        "min" = Min([Id; 2]),
        "atan2" = Atan2([Id; 2]),
        "dot" = Dot([Id; 2]),
        // Arithmetic (unary)
        "neg" = Neg([Id; 1]),
        "abs" = Abs([Id; 1]),
        "sign" = Sign([Id; 1]),
        "square" = Square([Id; 1]),
        "reciprocal" = Reciprocal([Id; 1]),
        // Exponential / logarithmic
        "exp" = Exp([Id; 1]),
        "expm1" = Expm1([Id; 1]),
        "log" = Log([Id; 1]),
        "log1p" = Log1p([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "rsqrt" = Rsqrt([Id; 1]),
        // Trigonometric
        "sin" = Sin([Id; 1]),
        "cos" = Cos([Id; 1]),
        "tan" = Tan([Id; 1]),
        "asin" = Asin([Id; 1]),
        "acos" = Acos([Id; 1]),
        "atan" = Atan([Id; 1]),
        // Hyperbolic
        "sinh" = Sinh([Id; 1]),
        "cosh" = Cosh([Id; 1]),
        "tanh" = Tanh([Id; 1]),
        // Special functions
        "logistic" = Logistic([Id; 1]),
        "erf" = Erf([Id; 1]),
        "erfc" = Erfc([Id; 1]),
        // Rounding
        "floor" = Floor([Id; 1]),
        "ceil" = Ceil([Id; 1]),
        "round" = Round([Id; 1]),
        // Reductions
        "reduce_sum" = ReduceSum([Id; 1]),
        "reduce_max" = ReduceMax([Id; 1]),
        "reduce_min" = ReduceMin([Id; 1]),
        "reduce_prod" = ReduceProd([Id; 1]),
        // Comparisons
        "eq" = Eq([Id; 2]),
        "ne" = Ne([Id; 2]),
        "lt" = Lt([Id; 2]),
        "le" = Le([Id; 2]),
        "gt" = Gt([Id; 2]),
        "ge" = Ge([Id; 2]),
        // Select (ternary)
        "select" = Select([Id; 3]),
        // Leaves
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
#[must_use]
pub fn algebraic_rules() -> Vec<egg::Rewrite<FjLang, ()>> {
    vec![
        // ── Commutativity ────────────────────────────────────────────
        rewrite!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)"),
        rewrite!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)"),
        rewrite!("max-comm"; "(max ?a ?b)" => "(max ?b ?a)"),
        rewrite!("min-comm"; "(min ?a ?b)" => "(min ?b ?a)"),
        // ── Associativity ────────────────────────────────────────────
        rewrite!("add-assoc"; "(add (add ?a ?b) ?c)" => "(add ?a (add ?b ?c))"),
        rewrite!("mul-assoc"; "(mul (mul ?a ?b) ?c)" => "(mul ?a (mul ?b ?c))"),
        rewrite!("max-assoc"; "(max (max ?a ?b) ?c)" => "(max ?a (max ?b ?c))"),
        rewrite!("min-assoc"; "(min (min ?a ?b) ?c)" => "(min ?a (min ?b ?c))"),
        // ── Additive identity / annihilation ─────────────────────────
        rewrite!("add-zero"; "(add ?a 0)" => "?a"),
        rewrite!("sub-zero"; "(sub ?a 0)" => "?a"),
        rewrite!("sub-self"; "(sub ?a ?a)" => "0"),
        rewrite!("sub-to-add-neg"; "(sub ?a ?b)" => "(add ?a (neg ?b))"),
        // ── Multiplicative identity / annihilation ───────────────────
        rewrite!("mul-one"; "(mul ?a 1)" => "?a"),
        rewrite!("mul-zero"; "(mul ?a 0)" => "0"),
        rewrite!("mul-neg-one"; "(mul ?a (neg 1))" => "(neg ?a)"),
        // ── Distributivity ───────────────────────────────────────────
        rewrite!("distribute"; "(mul ?a (add ?b ?c))" => "(add (mul ?a ?b) (mul ?a ?c))"),
        rewrite!("factor"; "(add (mul ?a ?b) (mul ?a ?c))" => "(mul ?a (add ?b ?c))"),
        // ── Negation ─────────────────────────────────────────────────
        rewrite!("neg-neg"; "(neg (neg ?a))" => "?a"),
        rewrite!("neg-zero"; "(neg 0)" => "0"),
        rewrite!("add-neg-self"; "(add ?a (neg ?a))" => "0"),
        // ── Abs idempotence ──────────────────────────────────────────
        rewrite!("abs-abs"; "(abs (abs ?a))" => "(abs ?a)"),
        rewrite!("abs-neg"; "(abs (neg ?a))" => "(abs ?a)"),
        // ── Max / Min idempotence ────────────────────────────────────
        rewrite!("max-self"; "(max ?a ?a)" => "?a"),
        rewrite!("min-self"; "(min ?a ?a)" => "?a"),
        // ── Power rules ──────────────────────────────────────────────
        rewrite!("pow-zero"; "(pow ?a 0)" => "1"),
        rewrite!("pow-one"; "(pow ?a 1)" => "?a"),
        // ── Exp / Log inverse pair ───────────────────────────────────
        rewrite!("exp-log"; "(exp (log ?a))" => "?a"),
        rewrite!("log-exp"; "(log (exp ?a))" => "?a"),
        // ── Sqrt / Rsqrt relationships ───────────────────────────────
        rewrite!("rsqrt-to-sqrt"; "(rsqrt ?a)" => "(pow (sqrt ?a) (neg 1))"),
        // ── Floor / Ceil / Round idempotence ─────────────────────────
        rewrite!("floor-floor"; "(floor (floor ?a))" => "(floor ?a)"),
        rewrite!("ceil-ceil"; "(ceil (ceil ?a))" => "(ceil ?a)"),
        rewrite!("round-round"; "(round (round ?a))" => "(round ?a)"),
        // ── Reduction idempotence (scalar results) ───────────────────
        rewrite!("reduce-sum-sum"; "(reduce_sum (reduce_sum ?a))" => "(reduce_sum ?a)"),
        rewrite!("reduce-max-max"; "(reduce_max (reduce_max ?a))" => "(reduce_max ?a)"),
        rewrite!("reduce-min-min"; "(reduce_min (reduce_min ?a))" => "(reduce_min ?a)"),
        rewrite!("reduce-prod-prod"; "(reduce_prod (reduce_prod ?a))" => "(reduce_prod ?a)"),
        // ── Trig negation rules ──────────────────────────────────────
        rewrite!("sin-neg"; "(sin (neg ?a))" => "(neg (sin ?a))"),
        rewrite!("cos-neg"; "(cos (neg ?a))" => "(cos ?a)"),
        rewrite!("tan-neg"; "(tan (neg ?a))" => "(neg (tan ?a))"),
        // ── Hyperbolic negation rules ──────────────────────────────────
        rewrite!("sinh-neg"; "(sinh (neg ?a))" => "(neg (sinh ?a))"),
        rewrite!("cosh-neg"; "(cosh (neg ?a))" => "(cosh ?a)"),
        rewrite!("tanh-neg"; "(tanh (neg ?a))" => "(neg (tanh ?a))"),
        // ── Division rules ─────────────────────────────────────────────
        rewrite!("div-one"; "(div ?a 1)" => "?a"),
        rewrite!("div-self"; "(div ?a ?a)" => "1"),
        // ── Square / Reciprocal rewrites ───────────────────────────────
        rewrite!("square-as-mul"; "(square ?a)" => "(mul ?a ?a)"),
        rewrite!("reciprocal-as-div"; "(reciprocal ?a)" => "(div 1 ?a)"),
        // ── Expm1 / Log1p inverses ─────────────────────────────────────
        rewrite!("expm1-log1p"; "(expm1 (log1p ?a))" => "?a"),
        rewrite!("log1p-expm1"; "(log1p (expm1 ?a))" => "?a"),
        // ── Select with constant condition ─────────────────────────────
        rewrite!("select-true"; "(select 1 ?a ?b)" => "?a"),
        rewrite!("select-false"; "(select 0 ?a ?b)" => "?b"),
        rewrite!("select-same"; "(select ?c ?a ?a)" => "?a"),
    ]
}

/// Convert a Jaxpr to an e-graph RecExpr.
pub fn jaxpr_to_egraph(jaxpr: &Jaxpr) -> (RecExpr<FjLang>, BTreeMap<VarId, Id>) {
    let mut expr = RecExpr::default();
    let mut var_map: BTreeMap<VarId, Id> = BTreeMap::new();

    // Add input variables and constant variables as symbols
    for var in jaxpr.invars.iter().chain(jaxpr.constvars.iter()) {
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
                Atom::Lit(Literal::Bool(b)) => {
                    let sym = egg::Symbol::from(format!("bool:{}", if *b { 1 } else { 0 }));
                    expr.add(FjLang::Symbol(sym))
                }
            })
            .collect();

        let node = match eqn.primitive {
            Primitive::Add => FjLang::Add([input_ids[0], input_ids[1]]),
            Primitive::Sub => FjLang::Sub([input_ids[0], input_ids[1]]),
            Primitive::Mul => FjLang::Mul([input_ids[0], input_ids[1]]),
            Primitive::Neg => FjLang::Neg([input_ids[0]]),
            Primitive::Abs => FjLang::Abs([input_ids[0]]),
            Primitive::Max => FjLang::Max([input_ids[0], input_ids[1]]),
            Primitive::Min => FjLang::Min([input_ids[0], input_ids[1]]),
            Primitive::Pow => FjLang::Pow([input_ids[0], input_ids[1]]),
            Primitive::Exp => FjLang::Exp([input_ids[0]]),
            Primitive::Log => FjLang::Log([input_ids[0]]),
            Primitive::Sqrt => FjLang::Sqrt([input_ids[0]]),
            Primitive::Rsqrt => FjLang::Rsqrt([input_ids[0]]),
            Primitive::Floor => FjLang::Floor([input_ids[0]]),
            Primitive::Ceil => FjLang::Ceil([input_ids[0]]),
            Primitive::Round => FjLang::Round([input_ids[0]]),
            Primitive::Sin => FjLang::Sin([input_ids[0]]),
            Primitive::Cos => FjLang::Cos([input_ids[0]]),
            Primitive::ReduceSum => FjLang::ReduceSum([input_ids[0]]),
            Primitive::ReduceMax => FjLang::ReduceMax([input_ids[0]]),
            Primitive::ReduceMin => FjLang::ReduceMin([input_ids[0]]),
            Primitive::ReduceProd => FjLang::ReduceProd([input_ids[0]]),
            Primitive::Dot => FjLang::Dot([input_ids[0], input_ids[1]]),
            Primitive::Eq => FjLang::Eq([input_ids[0], input_ids[1]]),
            Primitive::Ne => FjLang::Ne([input_ids[0], input_ids[1]]),
            Primitive::Lt => FjLang::Lt([input_ids[0], input_ids[1]]),
            Primitive::Le => FjLang::Le([input_ids[0], input_ids[1]]),
            Primitive::Gt => FjLang::Gt([input_ids[0], input_ids[1]]),
            Primitive::Ge => FjLang::Ge([input_ids[0], input_ids[1]]),
            // New binary ops
            Primitive::Div => FjLang::Div([input_ids[0], input_ids[1]]),
            Primitive::Rem => FjLang::Rem([input_ids[0], input_ids[1]]),
            Primitive::Atan2 => FjLang::Atan2([input_ids[0], input_ids[1]]),
            // New unary ops
            Primitive::Sign => FjLang::Sign([input_ids[0]]),
            Primitive::Square => FjLang::Square([input_ids[0]]),
            Primitive::Reciprocal => FjLang::Reciprocal([input_ids[0]]),
            Primitive::Expm1 => FjLang::Expm1([input_ids[0]]),
            Primitive::Log1p => FjLang::Log1p([input_ids[0]]),
            Primitive::Tan => FjLang::Tan([input_ids[0]]),
            Primitive::Asin => FjLang::Asin([input_ids[0]]),
            Primitive::Acos => FjLang::Acos([input_ids[0]]),
            Primitive::Atan => FjLang::Atan([input_ids[0]]),
            Primitive::Sinh => FjLang::Sinh([input_ids[0]]),
            Primitive::Cosh => FjLang::Cosh([input_ids[0]]),
            Primitive::Tanh => FjLang::Tanh([input_ids[0]]),
            Primitive::Logistic => FjLang::Logistic([input_ids[0]]),
            Primitive::Erf => FjLang::Erf([input_ids[0]]),
            Primitive::Erfc => FjLang::Erfc([input_ids[0]]),
            // Ternary
            Primitive::Select => FjLang::Select([input_ids[0], input_ids[1], input_ids[2]]),
            // Shape ops require params – not yet supported
            Primitive::Reshape
            | Primitive::Slice
            | Primitive::Gather
            | Primitive::Scatter
            | Primitive::Transpose
            | Primitive::BroadcastInDim
            | Primitive::Concatenate => {
                panic!(
                    "primitive {} not supported by egraph lowering",
                    eqn.primitive.as_str()
                )
            }
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
    constvars: &[VarId],
    original_outvars: &[VarId],
) -> Jaxpr {
    let max_in_const = invars.iter().chain(constvars.iter()).map(|v| v.0).max().unwrap_or(0);
    let mut next_var = max_in_const + 1;
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
            if invars.contains(&var_id) || constvars.contains(&var_id) {
                node_to_var.insert(idx, var_id);
            }
        }
    }

    for (idx, node) in expr.as_ref().iter().enumerate() {
        match node {
            FjLang::Num(_) | FjLang::Symbol(_) => {
                resolve_or_create(idx, &mut node_to_var, &mut next_var);
            }
            // Binary ops
            FjLang::Add([a, b]) => push_binary(
                idx,
                Primitive::Add,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Sub([a, b]) => push_binary(
                idx,
                Primitive::Sub,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Mul([a, b]) => push_binary(
                idx,
                Primitive::Mul,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Max([a, b]) => push_binary(
                idx,
                Primitive::Max,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Min([a, b]) => push_binary(
                idx,
                Primitive::Min,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Pow([a, b]) => push_binary(
                idx,
                Primitive::Pow,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Dot([a, b]) => push_binary(
                idx,
                Primitive::Dot,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Eq([a, b]) => push_binary(
                idx,
                Primitive::Eq,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Ne([a, b]) => push_binary(
                idx,
                Primitive::Ne,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Lt([a, b]) => push_binary(
                idx,
                Primitive::Lt,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Le([a, b]) => push_binary(
                idx,
                Primitive::Le,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Gt([a, b]) => push_binary(
                idx,
                Primitive::Gt,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Ge([a, b]) => push_binary(
                idx,
                Primitive::Ge,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Div([a, b]) => push_binary(
                idx,
                Primitive::Div,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Rem([a, b]) => push_binary(
                idx,
                Primitive::Rem,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Atan2([a, b]) => push_binary(
                idx,
                Primitive::Atan2,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            // Ternary ops
            FjLang::Select([cond, a, b]) => push_ternary(
                idx,
                Primitive::Select,
                *cond,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            // Unary ops
            FjLang::Neg([a]) => push_unary(
                idx,
                Primitive::Neg,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Abs([a]) => push_unary(
                idx,
                Primitive::Abs,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Exp([a]) => push_unary(
                idx,
                Primitive::Exp,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Log([a]) => push_unary(
                idx,
                Primitive::Log,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Sqrt([a]) => push_unary(
                idx,
                Primitive::Sqrt,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Rsqrt([a]) => push_unary(
                idx,
                Primitive::Rsqrt,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Floor([a]) => push_unary(
                idx,
                Primitive::Floor,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Ceil([a]) => push_unary(
                idx,
                Primitive::Ceil,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Round([a]) => push_unary(
                idx,
                Primitive::Round,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Sin([a]) => push_unary(
                idx,
                Primitive::Sin,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Cos([a]) => push_unary(
                idx,
                Primitive::Cos,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Tan([a]) => push_unary(
                idx,
                Primitive::Tan,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Asin([a]) => push_unary(
                idx,
                Primitive::Asin,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Acos([a]) => push_unary(
                idx,
                Primitive::Acos,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Atan([a]) => push_unary(
                idx,
                Primitive::Atan,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Sinh([a]) => push_unary(
                idx,
                Primitive::Sinh,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Cosh([a]) => push_unary(
                idx,
                Primitive::Cosh,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Tanh([a]) => push_unary(
                idx,
                Primitive::Tanh,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Sign([a]) => push_unary(
                idx,
                Primitive::Sign,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Square([a]) => push_unary(
                idx,
                Primitive::Square,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Reciprocal([a]) => push_unary(
                idx,
                Primitive::Reciprocal,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Expm1([a]) => push_unary(
                idx,
                Primitive::Expm1,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Log1p([a]) => push_unary(
                idx,
                Primitive::Log1p,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Logistic([a]) => push_unary(
                idx,
                Primitive::Logistic,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Erf([a]) => push_unary(
                idx,
                Primitive::Erf,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Erfc([a]) => push_unary(
                idx,
                Primitive::Erfc,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ReduceSum([a]) => push_unary(
                idx,
                Primitive::ReduceSum,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ReduceMax([a]) => push_unary(
                idx,
                Primitive::ReduceMax,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ReduceMin([a]) => push_unary(
                idx,
                Primitive::ReduceMin,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ReduceProd([a]) => push_unary(
                idx,
                Primitive::ReduceProd,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
        }
    }

    // Use the last node as the output variable (matches extraction behavior)
    let outvars = if let Some(last_var) = node_to_var.get(&(expr.as_ref().len() - 1)) {
        vec![*last_var]
    } else {
        original_outvars.to_vec()
    };

    Jaxpr::new(invars.to_vec(), constvars.to_vec(), outvars, equations)
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

#[allow(clippy::too_many_arguments)]
fn push_binary(
    idx: usize,
    prim: Primitive,
    a: Id,
    b: Id,
    node_to_var: &mut BTreeMap<usize, VarId>,
    next_var: &mut u32,
    equations: &mut Vec<Equation>,
    expr: &RecExpr<FjLang>,
) {
    let out = resolve_or_create(idx, node_to_var, next_var);
    let a_atom = id_to_atom(a, node_to_var, expr);
    let b_atom = id_to_atom(b, node_to_var, expr);
    equations.push(Equation {
        primitive: prim,
        inputs: smallvec![a_atom, b_atom],
        outputs: smallvec![out],
        params: BTreeMap::new(),
    });
}

fn push_unary(
    idx: usize,
    prim: Primitive,
    a: Id,
    node_to_var: &mut BTreeMap<usize, VarId>,
    next_var: &mut u32,
    equations: &mut Vec<Equation>,
    expr: &RecExpr<FjLang>,
) {
    let out = resolve_or_create(idx, node_to_var, next_var);
    let a_atom = id_to_atom(a, node_to_var, expr);
    equations.push(Equation {
        primitive: prim,
        inputs: smallvec![a_atom],
        outputs: smallvec![out],
        params: BTreeMap::new(),
    });
}

#[allow(clippy::too_many_arguments)]
fn push_ternary(
    idx: usize,
    prim: Primitive,
    a: Id,
    b: Id,
    c: Id,
    node_to_var: &mut BTreeMap<usize, VarId>,
    next_var: &mut u32,
    equations: &mut Vec<Equation>,
    expr: &RecExpr<FjLang>,
) {
    let out = resolve_or_create(idx, node_to_var, next_var);
    let a_atom = id_to_atom(a, node_to_var, expr);
    let b_atom = id_to_atom(b, node_to_var, expr);
    let c_atom = id_to_atom(c, node_to_var, expr);
    equations.push(Equation {
        primitive: prim,
        inputs: smallvec![a_atom, b_atom, c_atom],
        outputs: smallvec![out],
        params: BTreeMap::new(),
    });
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
            if let Some(bool_str) = sym.as_str().strip_prefix("bool:") {
                return Atom::Lit(Literal::Bool(bool_str == "1"));
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
#[must_use]
pub fn optimize_jaxpr(jaxpr: &Jaxpr) -> Jaxpr {
    if jaxpr
        .equations
        .iter()
        .any(|eqn| !is_egraph_supported_primitive(eqn.primitive))
    {
        // Keep behavior unchanged when e-graph cannot represent an operation yet.
        return jaxpr.clone();
    }

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

    egraph_to_jaxpr(&best_expr, &jaxpr.invars, &jaxpr.constvars, &jaxpr.outvars)
}

fn is_egraph_supported_primitive(primitive: Primitive) -> bool {
    !matches!(
        primitive,
        Primitive::Reshape
            | Primitive::Slice
            | Primitive::Gather
            | Primitive::Scatter
            | Primitive::Transpose
            | Primitive::BroadcastInDim
            | Primitive::Concatenate
    )
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

    #[test]
    fn neg_neg_simplification() {
        // neg(neg(x)) should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        // Double negation should be eliminated
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "neg(neg(x)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );

        // Verify correctness
        let orig_out = eval_jaxpr(&jaxpr, &[Value::scalar_i64(7)]).unwrap();
        let opt_out = eval_jaxpr(&optimized, &[Value::scalar_i64(7)]).unwrap();
        assert_eq!(orig_out, opt_out);
    }

    #[test]
    fn sub_self_simplification() {
        // x - x should simplify (fewer or equal equations)
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Sub,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        // The optimizer should recognize sub(x,x) = 0 and simplify
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "sub(x,x) should not increase equation count"
        );
    }

    #[test]
    fn exp_log_inverse() {
        // exp(log(x)) should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "exp(log(x)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn abs_idempotent() {
        // abs(abs(x)) should simplify to abs(x)
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Abs,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Abs,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "abs(abs(x)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );

        // Verify correctness for negative input
        let orig_out = eval_jaxpr(&jaxpr, &[Value::scalar_i64(-5)]).unwrap();
        let opt_out = eval_jaxpr(&optimized, &[Value::scalar_i64(-5)]).unwrap();
        assert_eq!(orig_out, opt_out);
    }

    #[test]
    fn max_min_self_simplification() {
        // max(x, x) should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Max,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        let orig_out = eval_jaxpr(&jaxpr, &[Value::scalar_i64(42)]).unwrap();
        let opt_out = eval_jaxpr(&optimized, &[Value::scalar_i64(42)]).unwrap();
        assert_eq!(orig_out, opt_out);
    }

    #[test]
    fn floor_idempotent() {
        // floor(floor(x)) should simplify to floor(x)
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Floor,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Floor,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "floor(floor(x)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn add_zero_identity() {
        // x + 0 should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(0))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        // After add-zero rule, should have no equations (just pass-through)
        assert!(
            optimized.equations.is_empty(),
            "x+0 should simplify to x: got {} eqns",
            optimized.equations.len(),
        );
    }

    #[test]
    fn algebraic_rules_count() {
        let rules = algebraic_rules();
        assert!(
            rules.len() >= 50,
            "expected at least 50 rewrite rules, got {}",
            rules.len(),
        );
    }

    #[test]
    fn div_one_identity() {
        // x / 1 should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Div,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.is_empty(),
            "x/1 should simplify to x: got {} eqns",
            optimized.equations.len(),
        );
    }

    #[test]
    fn square_as_mul_roundtrip() {
        // square(x) should be rewritable to mul(x, x)
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Square,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        // Should still produce the same result (either square or mul(x,x))
        assert!(
            !optimized.equations.is_empty(),
            "square(x) optimization should produce at least one equation"
        );
    }

    #[test]
    fn expm1_log1p_inverse() {
        // expm1(log1p(x)) should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Log1p,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Expm1,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "expm1(log1p(x)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn select_same_simplification() {
        // select(c, a, a) should simplify to a
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Select,
                inputs: smallvec![
                    Atom::Var(VarId(1)),
                    Atom::Var(VarId(2)),
                    Atom::Var(VarId(2))
                ],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "select(c, a, a) should not increase equation count"
        );
    }

    #[test]
    fn trig_roundtrip_tan() {
        // tan(x) should round-trip through e-graph
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Tan,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert_eq!(
            optimized.equations.len(),
            1,
            "single tan should not be simplified away"
        );
        assert_eq!(optimized.equations[0].primitive, Primitive::Tan);
    }

    #[test]
    fn hyperbolic_roundtrip_tanh() {
        // tanh(x) should round-trip through e-graph
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Tanh,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert_eq!(
            optimized.equations.len(),
            1,
            "single tanh should not be simplified away"
        );
        assert_eq!(optimized.equations[0].primitive, Primitive::Tanh);
    }

    #[test]
    fn new_primitives_parse_in_sexpr() {
        // Verify all new FjLang variants can be parsed from s-expressions
        let _: RecExpr<FjLang> = "(div 1 2)".parse().unwrap();
        let _: RecExpr<FjLang> = "(rem 1 2)".parse().unwrap();
        let _: RecExpr<FjLang> = "(atan2 1 2)".parse().unwrap();
        let _: RecExpr<FjLang> = "(sign 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(square 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(reciprocal 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(expm1 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(log1p 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(tan 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(asin 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(acos 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(atan 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(sinh 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(cosh 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(tanh 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(logistic 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(erf 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(erfc 1)".parse().unwrap();
        let _: RecExpr<FjLang> = "(select 1 2 3)".parse().unwrap();
    }

    #[test]
    fn test_egraph_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("egraph", "optimize")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_egraph_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }
}
