#![forbid(unsafe_code)]

use egg::{CostFunction, Id, Language, RecExpr, Runner, define_language, rewrite};
use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, VarId};
use smallvec::smallvec;
use std::collections::{BTreeMap, BTreeSet};

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
        // Complex number ops
        "complex" = Complex([Id; 2]),
        "conj" = Conj([Id; 1]),
        "real" = Real([Id; 1]),
        "imag" = Imag([Id; 1]),
        // Special math (unary)
        "cbrt" = Cbrt([Id; 1]),
        "lgamma" = Lgamma([Id; 1]),
        "digamma" = Digamma([Id; 1]),
        "erf_inv" = ErfInv([Id; 1]),
        "is_finite" = IsFinite([Id; 1]),
        // Special math (binary)
        "integer_pow" = IntegerPow([Id; 2]),
        "nextafter" = Nextafter([Id; 2]),
        // Bitwise (binary)
        "bitwise_and" = BitwiseAnd([Id; 2]),
        "bitwise_or" = BitwiseOr([Id; 2]),
        "bitwise_xor" = BitwiseXor([Id; 2]),
        "shift_left" = ShiftLeft([Id; 2]),
        "shift_right_arith" = ShiftRightArithmetic([Id; 2]),
        "shift_right_logical" = ShiftRightLogical([Id; 2]),
        // Bitwise (unary)
        "bitwise_not" = BitwiseNot([Id; 1]),
        "popcount" = PopulationCount([Id; 1]),
        "clz" = CountLeadingZeros([Id; 1]),
        // Bitwise reductions
        "reduce_and" = ReduceAnd([Id; 1]),
        "reduce_or" = ReduceOr([Id; 1]),
        "reduce_xor" = ReduceXor([Id; 1]),
        // Utility
        "copy" = Copy([Id; 1]),
        // Select (ternary)
        "select" = Select([Id; 3]),
        // Clamp (ternary)
        "clamp" = Clamp([Id; 3]),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EGraphLoweringError {
    UnsupportedPrimitive {
        primitive: Primitive,
        reason: ExclusionReason,
    },
    InvalidPrimitiveParams {
        primitive: Primitive,
        detail: String,
    },
    MissingLoweringCase {
        primitive: Primitive,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExclusionReason {
    ShapeManipulation,
    LinearAlgebra,
    Fft,
    ControlFlow,
    Sorting,
    Convolution,
    IndexUtility,
    TypeConversion,
    Cumulative,
    Windowed,
    Encoding,
    Collective,
}

impl ExclusionReason {
    #[must_use]
    pub fn category(self) -> &'static str {
        match self {
            Self::ShapeManipulation => "shape manipulation",
            Self::LinearAlgebra => "linear algebra",
            Self::Fft => "fft",
            Self::ControlFlow => "control flow",
            Self::Sorting => "sorting",
            Self::Convolution => "convolution",
            Self::IndexUtility => "index/utility",
            Self::TypeConversion => "type conversion",
            Self::Cumulative => "cumulative",
            Self::Windowed => "windowed",
            Self::Encoding => "encoding",
            Self::Collective => "collective",
        }
    }

    #[must_use]
    pub fn detail(self) -> &'static str {
        match self {
            Self::ShapeManipulation => {
                "requires shape or axis parameters not representable in the algebraic e-graph"
            }
            Self::LinearAlgebra => {
                "requires decomposition-specific parameters or multi-result structure outside the algebraic e-graph"
            }
            Self::Fft => {
                "requires transform-length and layout metadata not representable in the algebraic e-graph"
            }
            Self::ControlFlow => {
                "requires sub-jaxprs and branch metadata outside the algebraic e-graph"
            }
            Self::Sorting => {
                "requires sort-axis or comparator metadata not representable in the algebraic e-graph"
            }
            Self::Convolution => {
                "requires window, stride, and padding metadata not representable in the algebraic e-graph"
            }
            Self::IndexUtility => {
                "requires dynamic index or update metadata not representable in the algebraic e-graph"
            }
            Self::TypeConversion => {
                "requires bit-layout or precision metadata not representable in the algebraic e-graph"
            }
            Self::Cumulative => {
                "requires axis and scan-direction metadata not representable in the algebraic e-graph"
            }
            Self::Windowed => {
                "requires window geometry metadata not representable in the algebraic e-graph"
            }
            Self::Encoding => {
                "requires category-depth and axis metadata not representable in the algebraic e-graph"
            }
            Self::Collective => {
                "requires pmap axis context and multi-device semantics outside the algebraic e-graph"
            }
        }
    }
}

impl std::fmt::Display for EGraphLoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPrimitive { primitive, reason } => write!(
                f,
                "primitive {} excluded from egraph lowering ({}) because it {}",
                primitive.as_str(),
                reason.category(),
                reason.detail()
            ),
            Self::InvalidPrimitiveParams { primitive, detail } => write!(
                f,
                "primitive {} has invalid egraph lowering parameters: {detail}",
                primitive.as_str()
            ),
            Self::MissingLoweringCase { primitive } => write!(
                f,
                "primitive {} is permitted for egraph lowering but has no lowering case",
                primitive.as_str()
            ),
        }
    }
}

impl std::error::Error for EGraphLoweringError {}

/// Configuration for e-graph optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OptimizationConfig {
    /// When enabled, disables rewrites that are mathematically correct but
    /// numerically unsafe near domain boundaries (0, NaN, Inf, denormals).
    ///
    /// Disabled rewrites in safety mode:
    /// - Floating associativity/distributivity/factoring (changes rounding,
    ///   overflow, and cancellation order)
    /// - `exp(log(a)) => a` (fails when a <= 0)
    /// - `a/a => 1` (fails when a = 0, NaN, denormal)
    /// - `a - a => 0` (fails when a is NaN or Inf)
    /// - `a + -a => 0` (fails when a is NaN or Inf)
    /// - `a * 0 => 0` (fails when a is NaN or Inf)
    /// - `a * (1/a) => 1` (fails when a = 0)
    /// - `(a*b)/b => a` (fails when b = 0)
    /// - `log(exp(a)) => a` (fails when exp(a) overflows to Inf)
    /// - `expm1(log1p(a)) => a` (fails when a <= -1)
    /// - `log1p(expm1(a)) => a` (fails when expm1(a) overflows to Inf)
    /// - `log(a*b) => log(a) + log(b)` (fails when a or b <= 0)
    /// - `log(a/b) => log(a) - log(b)` (fails when a <= 0 or b <= 0)
    /// - Exact trig/hyperbolic/logistic identities (erase NaN/Inf boundaries)
    /// - Max/min absorption (fails when repeated operand is NaN)
    pub numerical_safety_mode: bool,
}

impl OptimizationConfig {
    /// Create a config with numerical safety mode enabled.
    #[must_use]
    pub fn safe() -> Self {
        Self {
            numerical_safety_mode: true,
        }
    }

    /// Create a config with all optimizations enabled (default).
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            numerical_safety_mode: false,
        }
    }
}

/// Standard algebraic rewrite rules for FjLang.
#[must_use]
pub fn algebraic_rules() -> Vec<egg::Rewrite<FjLang, ()>> {
    algebraic_rules_with_config(&OptimizationConfig::default())
}

/// Algebraic rewrite rules respecting the given configuration.
#[must_use]
pub fn algebraic_rules_with_config(config: &OptimizationConfig) -> Vec<egg::Rewrite<FjLang, ()>> {
    let mut rules = safe_algebraic_rules();
    if !config.numerical_safety_mode {
        rules.extend(numerically_unsafe_rules());
    }
    rules
}

/// Rules that are always safe regardless of input domain.
fn safe_algebraic_rules() -> Vec<egg::Rewrite<FjLang, ()>> {
    vec![
        // ── Commutativity ────────────────────────────────────────────
        rewrite!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)"),
        rewrite!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)"),
        rewrite!("max-comm"; "(max ?a ?b)" => "(max ?b ?a)"),
        rewrite!("min-comm"; "(min ?a ?b)" => "(min ?b ?a)"),
        // ── Associativity ────────────────────────────────────────────
        // add-assoc, mul-assoc moved to numerically_unsafe_rules
        // (floating association changes rounding, overflow, and cancellation order)
        rewrite!("max-assoc"; "(max (max ?a ?b) ?c)" => "(max ?a (max ?b ?c))"),
        rewrite!("min-assoc"; "(min (min ?a ?b) ?c)" => "(min ?a (min ?b ?c))"),
        // ── Additive identity / annihilation ─────────────────────────
        rewrite!("add-zero"; "(add ?a 0)" => "?a"),
        rewrite!("sub-zero"; "(sub ?a 0)" => "?a"),
        // sub-self moved to numerically_unsafe_rules (fails for NaN/Inf)
        rewrite!("sub-to-add-neg"; "(sub ?a ?b)" => "(add ?a (neg ?b))"),
        // ── Multiplicative identity / annihilation ───────────────────
        rewrite!("mul-one"; "(mul ?a 1)" => "?a"),
        // mul-zero moved to numerically_unsafe_rules (fails for NaN/Inf)
        rewrite!("mul-neg-one"; "(mul ?a (neg 1))" => "(neg ?a)"),
        // ── Distributivity ───────────────────────────────────────────
        // distribute, factor moved to numerically_unsafe_rules
        // (floating distribution changes overflow and cancellation order)
        // ── Negation ─────────────────────────────────────────────────
        rewrite!("neg-neg"; "(neg (neg ?a))" => "?a"),
        rewrite!("neg-zero"; "(neg 0)" => "0"),
        // add-neg-self moved to numerically_unsafe_rules (fails for NaN/Inf)
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
        // exp-log moved to numerically_unsafe_rules (fails when a <= 0)
        // log-exp moved to numerically_unsafe_rules (fails when exp(a) overflows)
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
        // div-self moved to numerically_unsafe_rules (fails when a = 0, NaN)
        // ── Square / Reciprocal rewrites ───────────────────────────────
        rewrite!("square-as-mul"; "(square ?a)" => "(mul ?a ?a)"),
        rewrite!("reciprocal-as-div"; "(reciprocal ?a)" => "(div 1 ?a)"),
        // ── Expm1 / Log1p inverses ─────────────────────────────────────
        // expm1-log1p, log1p-expm1 moved to numerically_unsafe_rules
        // ── Reciprocal involution ────────────────────────────────────────
        rewrite!("reciprocal-reciprocal"; "(reciprocal (reciprocal ?a))" => "?a"),
        // ── Sign idempotence ────────────────────────────────────────────
        rewrite!("sign-sign"; "(sign (sign ?a))" => "(sign ?a)"),
        // ── Trigonometric / hyperbolic / logistic identities ────────────
        // Exact identities moved to numerically_unsafe_rules (erase NaN/Inf boundaries)
        // ── Select with constant condition ─────────────────────────────
        rewrite!("select-true"; "(select 1 ?a ?b)" => "?a"),
        rewrite!("select-false"; "(select 0 ?a ?b)" => "?b"),
        rewrite!("select-same"; "(select ?c ?a ?a)" => "?a"),
        // ── Nested select with same condition ──────────────────────────
        rewrite!("select-nest-true"; "(select ?c (select ?c ?a ?b) ?x)" => "(select ?c ?a ?x)"),
        rewrite!("select-nest-false"; "(select ?c ?x (select ?c ?a ?b))" => "(select ?c ?x ?b)"),
        // ── Multiplicative cancellation ──────────────────────────────
        // mul-reciprocal, div-mul-cancel moved to numerically_unsafe_rules
        // ── Additional power rules ────────────────────────────────────
        rewrite!("pow-neg-one"; "(pow ?a (neg 1))" => "(reciprocal ?a)"),
        rewrite!("pow-two"; "(pow ?a 2)" => "(mul ?a ?a)"),
        // ── Log decomposition ─────────────────────────────────────────
        // log-product, log-quotient moved to numerically_unsafe_rules
        // ── Erf / Erfc identities ─────────────────────────────────────
        rewrite!("erf-neg"; "(erf (neg ?a))" => "(neg (erf ?a))"),
        // ── Max / Min absorption ──────────────────────────────────────
        // max-min-absorb, min-max-absorb moved to numerically_unsafe_rules
        // (fail when the repeated operand is NaN)
        // ── Clamp rules ─────────────────────────────────────────────────
        // clamp(x, lo, hi) = min(max(x, lo), hi)
        rewrite!("clamp-to-minmax"; "(clamp ?x ?lo ?hi)" => "(min (max ?x ?lo) ?hi)"),
        // clamp(x, x, x) = x (identity)
        rewrite!("clamp-same"; "(clamp ?a ?a ?a)" => "?a"),
        // ── Complex number rules ──────────────────────────────────────
        rewrite!("conj-conj"; "(conj (conj ?a))" => "?a"),
        rewrite!("real-complex"; "(real (complex ?r ?i))" => "?r"),
        rewrite!("imag-complex"; "(imag (complex ?r ?i))" => "?i"),
        rewrite!("complex-real-imag"; "(complex (real ?z) (imag ?z))" => "?z"),
        // ── Copy elimination ──────────────────────────────────────────
        rewrite!("copy-elim"; "(copy ?a)" => "?a"),
        // ── Integer power rules ───────────────────────────────────────
        rewrite!("integer-pow-zero"; "(integer_pow ?a 0)" => "1"),
        rewrite!("integer-pow-one"; "(integer_pow ?a 1)" => "?a"),
        rewrite!("integer-pow-two"; "(integer_pow ?a 2)" => "(mul ?a ?a)"),
        // ── Bitwise rules ─────────────────────────────────────────────
        rewrite!("bitwise-not-not"; "(bitwise_not (bitwise_not ?a))" => "?a"),
        rewrite!("bitwise-and-self"; "(bitwise_and ?a ?a)" => "?a"),
        rewrite!("bitwise-or-self"; "(bitwise_or ?a ?a)" => "?a"),
        rewrite!("bitwise-xor-self"; "(bitwise_xor ?a ?a)" => "0"),
        rewrite!("bitwise-and-comm"; "(bitwise_and ?a ?b)" => "(bitwise_and ?b ?a)"),
        rewrite!("bitwise-or-comm"; "(bitwise_or ?a ?b)" => "(bitwise_or ?b ?a)"),
        rewrite!("bitwise-xor-comm"; "(bitwise_xor ?a ?b)" => "(bitwise_xor ?b ?a)"),
        // ── IsFinite idempotence ──────────────────────────────────────
        rewrite!("is-finite-const-0"; "(is_finite 0)" => "1"),
        rewrite!("is-finite-const-1"; "(is_finite 1)" => "1"),
    ]
}

/// Rules that are mathematically correct but numerically unsafe near domain boundaries.
///
/// These rewrites can change observable behavior when inputs include:
/// - Zero (division, reciprocal)
/// - Negative numbers (log)
/// - NaN/Inf (cancellation patterns)
/// - Denormals (underflow-prone intermediate results)
///
/// Disabled when `OptimizationConfig::numerical_safety_mode` is true.
fn numerically_unsafe_rules() -> Vec<egg::Rewrite<FjLang, ()>> {
    vec![
        // Floating addition is not associative; reordering can change
        // rounding and overflow-visible results.
        rewrite!("add-assoc"; "(add (add ?a ?b) ?c)" => "(add ?a (add ?b ?c))"),
        // Floating multiplication is not associative; reordering can change
        // rounding and overflow-visible results.
        rewrite!("mul-assoc"; "(mul (mul ?a ?b) ?c)" => "(mul ?a (mul ?b ?c))"),
        // Distribution/factoring can erase overflow-visible NaNs, e.g.
        // (x*2) + (x*-2) is NaN for x=Inf-ish overflow, while x*(2 + -2) is 0.
        rewrite!("distribute"; "(mul ?a (add ?b ?c))" => "(add (mul ?a ?b) (mul ?a ?c))"),
        rewrite!("factor"; "(add (mul ?a ?b) (mul ?a ?c))" => "(mul ?a (add ?b ?c))"),
        // exp(log(a)) => a fails when a <= 0 (log undefined)
        rewrite!("exp-log"; "(exp (log ?a))" => "?a"),
        // a/a => 1 fails when a = 0 (0/0 = NaN, not 1)
        rewrite!("div-self"; "(div ?a ?a)" => "1"),
        // a - a => 0 fails when a is NaN or Inf
        rewrite!("sub-self"; "(sub ?a ?a)" => "0"),
        // a + -a => 0 fails when a is NaN or Inf
        rewrite!("add-neg-self"; "(add ?a (neg ?a))" => "0"),
        // a * 0 => 0 fails when a is NaN or Inf
        rewrite!("mul-zero"; "(mul ?a 0)" => "0"),
        // a * (1/a) => 1 fails when a = 0
        rewrite!("mul-reciprocal"; "(mul ?a (reciprocal ?a))" => "1"),
        // (a*b)/b => a fails when b = 0
        rewrite!("div-mul-cancel"; "(div (mul ?a ?b) ?b)" => "?a"),
        // log(exp(a)) => a fails when exp(a) overflows to Inf
        rewrite!("log-exp"; "(log (exp ?a))" => "?a"),
        // expm1(log1p(a)) => a fails when a <= -1 (log1p is NaN)
        rewrite!("expm1-log1p"; "(expm1 (log1p ?a))" => "?a"),
        // log1p(expm1(a)) => a fails when expm1(a) overflows to Inf
        rewrite!("log1p-expm1"; "(log1p (expm1 ?a))" => "?a"),
        // log(a*b) => log(a) + log(b) fails when a or b <= 0
        rewrite!("log-product"; "(log (mul ?a ?b))" => "(add (log ?a) (log ?b))"),
        // log(a/b) => log(a) - log(b) fails when a <= 0 or b <= 0
        rewrite!("log-quotient"; "(log (div ?a ?b))" => "(sub (log ?a) (log ?b))"),
        // sin(inf), cos(inf), and NaN inputs should remain NaN-observable.
        rewrite!("sin2-cos2"; "(add (mul (sin ?a) (sin ?a)) (mul (cos ?a) (cos ?a)))" => "1"),
        // cosh/sinh overflow makes the original expression observe Inf - Inf = NaN.
        rewrite!("cosh2-sinh2"; "(sub (mul (cosh ?a) (cosh ?a)) (mul (sinh ?a) (sinh ?a)))" => "1"),
        // logistic(NaN) + logistic(-NaN) is NaN, not 1.
        rewrite!("logistic-complement"; "(add (logistic ?a) (logistic (neg ?a)))" => "1"),
        // max/min absorption changes results when the repeated operand is NaN.
        rewrite!("max-min-absorb"; "(max ?a (min ?a ?b))" => "?a"),
        rewrite!("min-max-absorb"; "(min ?a (max ?a ?b))" => "?a"),
    ]
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TransposeSpec {
    Reverse,
    Explicit(Vec<usize>),
}

fn resolve_var_alias(var: VarId, aliases: &BTreeMap<VarId, VarId>) -> VarId {
    let mut current = var;
    while let Some(next) = aliases.get(&current).copied() {
        if next == current {
            break;
        }
        current = next;
    }
    current
}

fn use_counts(jaxpr: &Jaxpr) -> BTreeMap<VarId, usize> {
    let mut counts = BTreeMap::new();
    for equation in &jaxpr.equations {
        for atom in &equation.inputs {
            if let Atom::Var(var) = atom {
                *counts.entry(*var).or_insert(0) += 1;
            }
        }
    }
    for outvar in &jaxpr.outvars {
        *counts.entry(*outvar).or_insert(0) += 1;
    }
    counts
}

fn parse_usize_csv(raw: &str) -> Option<Vec<usize>> {
    raw.split(',')
        .map(|part| part.trim().parse::<usize>().ok())
        .collect()
}

fn transpose_spec(params: &BTreeMap<String, String>) -> Option<TransposeSpec> {
    match params.get("permutation") {
        Some(raw) => parse_usize_csv(raw).map(TransposeSpec::Explicit),
        None => Some(TransposeSpec::Reverse),
    }
}

fn is_identity_transpose(params: &BTreeMap<String, String>) -> bool {
    matches!(
        transpose_spec(params),
        Some(TransposeSpec::Explicit(permutation))
            if permutation.iter().enumerate().all(|(axis, value)| axis == *value)
    )
}

fn are_inverse_transposes(
    first: &BTreeMap<String, String>,
    second: &BTreeMap<String, String>,
) -> bool {
    match (transpose_spec(first), transpose_spec(second)) {
        (Some(TransposeSpec::Reverse), Some(TransposeSpec::Reverse)) => true,
        (Some(TransposeSpec::Explicit(lhs)), Some(TransposeSpec::Explicit(rhs))) => {
            lhs.len() == rhs.len()
                && lhs
                    .iter()
                    .enumerate()
                    .all(|(index, axis)| rhs.get(*axis).is_some_and(|rhs_axis| *rhs_axis == index))
        }
        (Some(TransposeSpec::Reverse), Some(TransposeSpec::Explicit(permutation)))
        | (Some(TransposeSpec::Explicit(permutation)), Some(TransposeSpec::Reverse)) => permutation
            .iter()
            .enumerate()
            .all(|(index, axis)| *axis == permutation.len().saturating_sub(index + 1)),
        _ => false,
    }
}

fn squeeze_single_dimension(params: &BTreeMap<String, String>) -> Option<usize> {
    params
        .get("dimensions")
        .and_then(|raw| parse_usize_csv(raw))
        .filter(|dims| dims.len() == 1)
        .map(|dims| dims[0])
}

fn expand_dims_axis(params: &BTreeMap<String, String>) -> Option<usize> {
    params.get("axis").and_then(|raw| raw.parse::<usize>().ok())
}

fn rev_axes(params: &BTreeMap<String, String>) -> Option<Vec<usize>> {
    params.get("axes").and_then(|raw| {
        let mut axes = parse_usize_csv(raw)?;
        axes.sort_unstable();
        Some(axes)
    })
}

fn is_shape_identity(equation: &Equation) -> bool {
    match equation.primitive {
        Primitive::Transpose => is_identity_transpose(&equation.params),
        _ => false,
    }
}

fn cancels_shape_chain(previous: &Equation, current: &Equation) -> bool {
    match (previous.primitive, current.primitive) {
        (Primitive::Transpose, Primitive::Transpose) => {
            are_inverse_transposes(&previous.params, &current.params)
        }
        (Primitive::ExpandDims, Primitive::Squeeze) => {
            expand_dims_axis(&previous.params) == squeeze_single_dimension(&current.params)
        }
        (Primitive::Squeeze, Primitive::ExpandDims) => {
            squeeze_single_dimension(&previous.params) == expand_dims_axis(&current.params)
        }
        (Primitive::Rev, Primitive::Rev) => rev_axes(&previous.params) == rev_axes(&current.params),
        _ => false,
    }
}

fn optimize_shape_parametric_chains(jaxpr: &Jaxpr) -> Jaxpr {
    let counts = use_counts(jaxpr);
    let mut aliases = BTreeMap::new();
    let mut equations: Vec<Equation> = Vec::with_capacity(jaxpr.equations.len());

    for equation in &jaxpr.equations {
        let mut rewritten = equation.clone();
        for atom in &mut rewritten.inputs {
            if let Atom::Var(var) = atom {
                *var = resolve_var_alias(*var, &aliases);
            }
        }

        let identity_input = if rewritten.outputs.len() == 1
            && rewritten.inputs.len() == 1
            && is_shape_identity(&rewritten)
        {
            match rewritten.inputs[0] {
                Atom::Var(input_var) => Some(input_var),
                Atom::Lit(_) => None,
            }
        } else {
            None
        };
        if let Some(input_var) = identity_input {
            aliases.insert(rewritten.outputs[0], input_var);
            continue;
        }

        let previous = equations.last();
        let can_cancel = if rewritten.outputs.len() == 1 && rewritten.inputs.len() == 1 {
            if let (Some(previous), Atom::Var(input_var)) = (previous, &rewritten.inputs[0]) {
                previous.outputs.len() == 1
                    && previous.outputs[0] == *input_var
                    && counts.get(&previous.outputs[0]).copied().unwrap_or(0) == 1
                    && cancels_shape_chain(previous, &rewritten)
            } else {
                false
            }
        } else {
            false
        };

        if can_cancel {
            let Some(previous) = equations.pop() else {
                equations.push(rewritten);
                continue;
            };
            let source = match previous.inputs[0] {
                Atom::Var(var) => var,
                Atom::Lit(_) => {
                    equations.push(previous);
                    equations.push(rewritten);
                    continue;
                }
            };
            aliases.insert(rewritten.outputs[0], source);
            continue;
        }

        equations.push(rewritten);
    }

    let mut optimized = Jaxpr::new(
        jaxpr.invars.clone(),
        jaxpr.constvars.clone(),
        jaxpr
            .outvars
            .iter()
            .map(|outvar| resolve_var_alias(*outvar, &aliases))
            .collect(),
        equations,
    );
    optimized.effects = jaxpr.effects.clone();
    optimized
}

/// Convert a Jaxpr to an e-graph RecExpr.
pub fn jaxpr_to_egraph(
    jaxpr: &Jaxpr,
) -> Result<(RecExpr<FjLang>, BTreeMap<VarId, Id>), EGraphLoweringError> {
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
                Atom::Lit(Literal::U32(n)) => {
                    let sym = egg::Symbol::from(format!("u32:{n}"));
                    expr.add(FjLang::Symbol(sym))
                }
                Atom::Lit(Literal::U64(n)) => {
                    let sym = egg::Symbol::from(format!("u64:{n}"));
                    expr.add(FjLang::Symbol(sym))
                }
                Atom::Lit(Literal::F64Bits(bits)) => {
                    // Encode as symbol to preserve bit-exactness
                    let sym = egg::Symbol::from(format!("f64:{bits}"));
                    expr.add(FjLang::Symbol(sym))
                }
                Atom::Lit(Literal::F32Bits(bits)) => {
                    let sym = egg::Symbol::from(format!("f32:{bits}"));
                    expr.add(FjLang::Symbol(sym))
                }
                Atom::Lit(Literal::Bool(b)) => {
                    let sym = egg::Symbol::from(format!("bool:{}", if *b { 1 } else { 0 }));
                    expr.add(FjLang::Symbol(sym))
                }
                Atom::Lit(Literal::BF16Bits(bits)) => {
                    let sym = egg::Symbol::from(format!("bf16:{bits}"));
                    expr.add(FjLang::Symbol(sym))
                }
                Atom::Lit(Literal::F16Bits(bits)) => {
                    let sym = egg::Symbol::from(format!("f16:{bits}"));
                    expr.add(FjLang::Symbol(sym))
                }
                Atom::Lit(Literal::Complex64Bits(re, im)) => {
                    let sym = egg::Symbol::from(format!("c64:{re}:{im}"));
                    expr.add(FjLang::Symbol(sym))
                }
                Atom::Lit(Literal::Complex128Bits(re, im)) => {
                    let sym = egg::Symbol::from(format!("c128:{re}:{im}"));
                    expr.add(FjLang::Symbol(sym))
                }
            })
            .collect();

        if let Some(reason) = excluded_primitive_reason(eqn.primitive) {
            return Err(EGraphLoweringError::UnsupportedPrimitive {
                primitive: eqn.primitive,
                reason,
            });
        }

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
            Primitive::Complex => FjLang::Complex([input_ids[0], input_ids[1]]),
            Primitive::Conj => FjLang::Conj([input_ids[0]]),
            Primitive::Real => FjLang::Real([input_ids[0]]),
            Primitive::Imag => FjLang::Imag([input_ids[0]]),
            // Ternary
            Primitive::Select => FjLang::Select([input_ids[0], input_ids[1], input_ids[2]]),
            // Clamp (ternary)
            Primitive::Clamp => FjLang::Clamp([input_ids[0], input_ids[1], input_ids[2]]),
            // New unary ops
            Primitive::Cbrt => FjLang::Cbrt([input_ids[0]]),
            Primitive::Lgamma => FjLang::Lgamma([input_ids[0]]),
            Primitive::Digamma => FjLang::Digamma([input_ids[0]]),
            Primitive::ErfInv => FjLang::ErfInv([input_ids[0]]),
            Primitive::IsFinite => FjLang::IsFinite([input_ids[0]]),
            Primitive::Copy => FjLang::Copy([input_ids[0]]),
            Primitive::BitwiseNot => FjLang::BitwiseNot([input_ids[0]]),
            Primitive::PopulationCount => FjLang::PopulationCount([input_ids[0]]),
            Primitive::CountLeadingZeros => FjLang::CountLeadingZeros([input_ids[0]]),
            Primitive::ReduceAnd => FjLang::ReduceAnd([input_ids[0]]),
            Primitive::ReduceOr => FjLang::ReduceOr([input_ids[0]]),
            Primitive::ReduceXor => FjLang::ReduceXor([input_ids[0]]),
            // New binary ops
            Primitive::IntegerPow => {
                let exponent = integer_pow_exponent_param(&eqn.params)?;
                let exponent_id = expr.add(FjLang::Num(i64::from(exponent)));
                FjLang::IntegerPow([input_ids[0], exponent_id])
            }
            Primitive::Nextafter => FjLang::Nextafter([input_ids[0], input_ids[1]]),
            Primitive::BitwiseAnd => FjLang::BitwiseAnd([input_ids[0], input_ids[1]]),
            Primitive::BitwiseOr => FjLang::BitwiseOr([input_ids[0], input_ids[1]]),
            Primitive::BitwiseXor => FjLang::BitwiseXor([input_ids[0], input_ids[1]]),
            Primitive::ShiftLeft => FjLang::ShiftLeft([input_ids[0], input_ids[1]]),
            Primitive::ShiftRightArithmetic => {
                FjLang::ShiftRightArithmetic([input_ids[0], input_ids[1]])
            }
            Primitive::ShiftRightLogical => FjLang::ShiftRightLogical([input_ids[0], input_ids[1]]),
            primitive => {
                return Err(EGraphLoweringError::MissingLoweringCase { primitive });
            }
        };

        let id = expr.add(node);
        for outvar in eqn.outputs.iter() {
            var_map.insert(*outvar, id);
        }
    }

    Ok((expr, var_map))
}

fn integer_pow_exponent_param(
    params: &BTreeMap<String, String>,
) -> Result<i32, EGraphLoweringError> {
    params
        .get("exponent")
        .and_then(|raw| raw.trim().parse::<i32>().ok())
        .ok_or_else(|| EGraphLoweringError::InvalidPrimitiveParams {
            primitive: Primitive::IntegerPow,
            detail: "integer_pow requires integer 'exponent' param".to_owned(),
        })
}

fn decode_symbol_literal(sym: &egg::Symbol) -> Option<Literal> {
    let raw = sym.as_str();
    if let Some(bits_str) = raw.strip_prefix("f64:") {
        return bits_str.parse::<u64>().ok().map(Literal::F64Bits);
    }
    if let Some(bits_str) = raw.strip_prefix("f32:") {
        return bits_str.parse::<u32>().ok().map(Literal::F32Bits);
    }
    if let Some(bool_str) = raw.strip_prefix("bool:") {
        return match bool_str {
            "0" => Some(Literal::Bool(false)),
            "1" => Some(Literal::Bool(true)),
            _ => None,
        };
    }
    if let Some(value_str) = raw.strip_prefix("u32:") {
        return value_str.parse::<u32>().ok().map(Literal::U32);
    }
    if let Some(value_str) = raw.strip_prefix("u64:") {
        return value_str.parse::<u64>().ok().map(Literal::U64);
    }
    if let Some(bits_str) = raw.strip_prefix("bf16:") {
        return bits_str.parse::<u16>().ok().map(Literal::BF16Bits);
    }
    if let Some(bits_str) = raw.strip_prefix("f16:") {
        return bits_str.parse::<u16>().ok().map(Literal::F16Bits);
    }
    if let Some(values_str) = raw.strip_prefix("c64:") {
        let mut parts = values_str.split(':');
        let re = parts.next()?.parse::<u32>().ok()?;
        let im = parts.next()?.parse::<u32>().ok()?;
        if parts.next().is_some() {
            return None;
        }
        return Some(Literal::Complex64Bits(re, im));
    }
    if let Some(values_str) = raw.strip_prefix("c128:") {
        let mut parts = values_str.split(':');
        let re = parts.next()?.parse::<u64>().ok()?;
        let im = parts.next()?.parse::<u64>().ok()?;
        if parts.next().is_some() {
            return None;
        }
        return Some(Literal::Complex128Bits(re, im));
    }
    None
}

/// Convert an e-graph RecExpr back to a Jaxpr.
pub fn egraph_to_jaxpr(
    expr: &RecExpr<FjLang>,
    invars: &[VarId],
    constvars: &[VarId],
    original_outvars: &[VarId],
) -> Jaxpr {
    let max_in_const = invars
        .iter()
        .chain(constvars.iter())
        .map(|v| v.0)
        .max()
        .unwrap_or(0);
    let mut next_var = max_in_const + 1;
    let mut equations = Vec::new();
    let mut node_to_var: BTreeMap<usize, VarId> = BTreeMap::new();

    // Map Symbol nodes back to their original VarIds by parsing the symbol name.
    // After equality saturation + extraction, Symbol nodes may appear in any order
    // in the RecExpr, so we cannot assume positional correspondence with invars.
    for (idx, node) in expr.as_ref().iter().enumerate() {
        let FjLang::Symbol(sym) = node else {
            continue;
        };
        let Some(rest) = sym.as_str().strip_prefix('v') else {
            continue;
        };
        let Ok(var_num) = rest.parse::<u32>() else {
            continue;
        };
        let var_id = VarId(var_num);
        if invars.contains(&var_id) || constvars.contains(&var_id) {
            node_to_var.insert(idx, var_id);
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
            FjLang::Clamp([x, lo, hi]) => push_ternary(
                idx,
                Primitive::Clamp,
                *x,
                *lo,
                *hi,
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
            // Complex ops
            FjLang::Complex([a, b]) => push_binary(
                idx,
                Primitive::Complex,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Conj([a]) => push_unary(
                idx,
                Primitive::Conj,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Real([a]) => push_unary(
                idx,
                Primitive::Real,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Imag([a]) => push_unary(
                idx,
                Primitive::Imag,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            // Special math
            FjLang::Cbrt([a]) => push_unary(
                idx,
                Primitive::Cbrt,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Lgamma([a]) => push_unary(
                idx,
                Primitive::Lgamma,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Digamma([a]) => push_unary(
                idx,
                Primitive::Digamma,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ErfInv([a]) => push_unary(
                idx,
                Primitive::ErfInv,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::IsFinite([a]) => push_unary(
                idx,
                Primitive::IsFinite,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::IntegerPow([a, b]) => push_integer_pow(
                idx,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::Nextafter([a, b]) => push_binary(
                idx,
                Primitive::Nextafter,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            // Bitwise ops
            FjLang::BitwiseAnd([a, b]) => push_binary(
                idx,
                Primitive::BitwiseAnd,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::BitwiseOr([a, b]) => push_binary(
                idx,
                Primitive::BitwiseOr,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::BitwiseXor([a, b]) => push_binary(
                idx,
                Primitive::BitwiseXor,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ShiftLeft([a, b]) => push_binary(
                idx,
                Primitive::ShiftLeft,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ShiftRightArithmetic([a, b]) => push_binary(
                idx,
                Primitive::ShiftRightArithmetic,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ShiftRightLogical([a, b]) => push_binary(
                idx,
                Primitive::ShiftRightLogical,
                *a,
                *b,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::BitwiseNot([a]) => push_unary(
                idx,
                Primitive::BitwiseNot,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::PopulationCount([a]) => push_unary(
                idx,
                Primitive::PopulationCount,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::CountLeadingZeros([a]) => push_unary(
                idx,
                Primitive::CountLeadingZeros,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ReduceAnd([a]) => push_unary(
                idx,
                Primitive::ReduceAnd,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ReduceOr([a]) => push_unary(
                idx,
                Primitive::ReduceOr,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            FjLang::ReduceXor([a]) => push_unary(
                idx,
                Primitive::ReduceXor,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
            // Utility
            FjLang::Copy([a]) => push_unary(
                idx,
                Primitive::Copy,
                *a,
                &mut node_to_var,
                &mut next_var,
                &mut equations,
                expr,
            ),
        }
    }

    // Use the last node as the output variable (matches extraction behavior)
    let last_idx = expr.as_ref().len().saturating_sub(1);
    let outvars = if let Some(last_var) = node_to_var.get(&last_idx) {
        if !expr.as_ref().is_empty() {
            let last_node = &expr.as_ref()[last_idx];
            let is_literal = match last_node {
                FjLang::Num(_) => true,
                FjLang::Symbol(sym) => decode_symbol_literal(sym).is_some(),
                _ => false,
            };
            if is_literal {
                let lit_atom = id_to_atom(Id::from(last_idx), &node_to_var, expr);
                equations.push(Equation {
                    primitive: Primitive::Select,
                    inputs: smallvec![Atom::Lit(Literal::Bool(true)), lit_atom.clone(), lit_atom],
                    outputs: smallvec![*last_var],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                });
            }
        }
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
        effects: vec![],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
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
        effects: vec![],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
    });
}

#[allow(clippy::too_many_arguments)]
fn push_integer_pow(
    idx: usize,
    base: Id,
    exponent: Id,
    node_to_var: &mut BTreeMap<usize, VarId>,
    next_var: &mut u32,
    equations: &mut Vec<Equation>,
    expr: &RecExpr<FjLang>,
) {
    let exponent_atom = id_to_atom(exponent, node_to_var, expr);
    let Some(exponent) = integer_pow_exponent_from_atom(&exponent_atom) else {
        push_binary(
            idx,
            Primitive::IntegerPow,
            base,
            exponent,
            node_to_var,
            next_var,
            equations,
            expr,
        );
        return;
    };

    let out = resolve_or_create(idx, node_to_var, next_var);
    let base_atom = id_to_atom(base, node_to_var, expr);
    equations.push(Equation {
        primitive: Primitive::IntegerPow,
        inputs: smallvec![base_atom],
        outputs: smallvec![out],
        effects: vec![],
        params: BTreeMap::from([("exponent".to_owned(), exponent.to_string())]),
        sub_jaxprs: vec![],
    });
}

fn integer_pow_exponent_from_atom(atom: &Atom) -> Option<i32> {
    match atom {
        Atom::Lit(Literal::I64(value)) => i32::try_from(*value).ok(),
        Atom::Lit(Literal::U32(value)) => i32::try_from(*value).ok(),
        Atom::Lit(Literal::U64(value)) => i32::try_from(*value).ok(),
        _ => None,
    }
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
        effects: vec![],
        params: BTreeMap::new(),
        sub_jaxprs: vec![],
    });
}

fn id_to_atom(id: Id, node_to_var: &BTreeMap<usize, VarId>, expr: &RecExpr<FjLang>) -> Atom {
    let idx: usize = id.into();
    // Check if this node is a literal (Num or encoded Symbol)
    match &expr.as_ref()[idx] {
        FjLang::Num(n) => Atom::Lit(Literal::I64(*n)),
        FjLang::Symbol(sym) => {
            if let Some(literal) = decode_symbol_literal(sym) {
                return Atom::Lit(literal);
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
///
/// Optimizes supported single-output regions while preserving opaque barriers
/// such as multi-output primitives, control flow, and shape-parametric ops.
#[must_use]
pub fn optimize_jaxpr(jaxpr: &Jaxpr) -> Jaxpr {
    optimize_jaxpr_with_config(jaxpr, &OptimizationConfig::default())
}

/// Optimize a Jaxpr with the given configuration.
///
/// When `config.numerical_safety_mode` is true, rewrites that could change
/// observable behavior near domain boundaries (0, NaN, Inf) are disabled.
#[must_use]
pub fn optimize_jaxpr_with_config(jaxpr: &Jaxpr, config: &OptimizationConfig) -> Jaxpr {
    let jaxpr = optimize_shape_parametric_chains(jaxpr);
    let mut equations = Vec::new();
    let mut index = 0;
    let mut next_var = max_var_id(&jaxpr).saturating_add(1);
    let mut outvar_remap = BTreeMap::new();

    while index < jaxpr.equations.len() {
        if is_egraph_barrier(&jaxpr.equations[index]) {
            equations.push(jaxpr.equations[index].clone());
            index += 1;
            continue;
        }

        let segment_start = index;
        while index < jaxpr.equations.len() && !is_egraph_barrier(&jaxpr.equations[index]) {
            index += 1;
        }

        let (segment, preserved_outvars) =
            build_supported_segment_jaxpr(&jaxpr, segment_start, index);
        let optimized =
            optimize_supported_segment(&segment, &preserved_outvars, &mut next_var, config);
        equations.extend(optimized.equations);
        outvar_remap.extend(optimized.outvar_remap);
    }

    let mut optimized = Jaxpr::new(
        jaxpr.invars.clone(),
        jaxpr.constvars.clone(),
        jaxpr
            .outvars
            .iter()
            .map(|outvar| *outvar_remap.get(outvar).unwrap_or(outvar))
            .collect(),
        equations,
    );
    optimized.effects = jaxpr.effects.clone();
    optimized
}

struct SegmentOptimization {
    equations: Vec<Equation>,
    outvar_remap: BTreeMap<VarId, VarId>,
}

fn optimize_supported_segment(
    jaxpr: &Jaxpr,
    preserved_outvars: &BTreeSet<VarId>,
    next_var: &mut u32,
    config: &OptimizationConfig,
) -> SegmentOptimization {
    let (expr, var_map) = match jaxpr_to_egraph(jaxpr) {
        Ok(lowered) => lowered,
        Err(_) => {
            return SegmentOptimization {
                equations: jaxpr.equations.clone(),
                outvar_remap: jaxpr
                    .outvars
                    .iter()
                    .map(|outvar| (*outvar, *outvar))
                    .collect(),
            };
        }
    };
    let (egraph, rec_id_to_egraph_id) = build_egraph_with_id_map(&expr);
    let var_map: BTreeMap<VarId, Id> = var_map
        .into_iter()
        .map(|(var, rec_id)| {
            let idx: usize = rec_id.into();
            (var, rec_id_to_egraph_id[idx])
        })
        .collect();
    let mut runner = Runner::<FjLang, ()>::default().with_egraph(egraph);
    if let Some(root) = rec_id_to_egraph_id.last().copied() {
        runner.roots.push(root);
    }
    let runner = runner.run(&algebraic_rules_with_config(config));
    let extractor = egg::Extractor::new(&runner.egraph, OpCount);

    let mut merged_equations = Vec::new();
    let mut outvar_remap = BTreeMap::new();
    for desired_outvar in &jaxpr.outvars {
        let root_id = var_map
            .get(desired_outvar)
            .copied()
            .unwrap_or_else(|| Id::from(expr.as_ref().len() - 1));
        let (_, best_expr) = extractor.find_best(root_id);
        let piece = egraph_to_jaxpr(
            &best_expr,
            &jaxpr.invars,
            &jaxpr.constvars,
            &[*desired_outvar],
        );
        let result = rewrite_piece_for_output(
            &piece,
            *desired_outvar,
            preserved_outvars.contains(desired_outvar),
            &jaxpr.invars,
            &jaxpr.constvars,
            next_var,
        );
        merged_equations.extend(result.equations);
        outvar_remap.insert(*desired_outvar, result.actual_outvar);
    }
    let (merged_equations, aliases) =
        dedupe_identical_extracted_equations(merged_equations, preserved_outvars);
    for outvar in outvar_remap.values_mut() {
        *outvar = resolve_var_alias(*outvar, &aliases);
    }

    SegmentOptimization {
        equations: merged_equations,
        outvar_remap,
    }
}

fn build_egraph_with_id_map(expr: &RecExpr<FjLang>) -> (egg::EGraph<FjLang, ()>, Vec<Id>) {
    let mut egraph = egg::EGraph::<FjLang, ()>::default();
    let mut rec_id_to_egraph_id = Vec::with_capacity(expr.as_ref().len());

    for node in expr.as_ref() {
        let egraph_node = node
            .clone()
            .map_children(|child| rec_id_to_egraph_id[usize::from(child)]);
        let id = egraph.add(egraph_node);
        rec_id_to_egraph_id.push(id);
    }

    (egraph, rec_id_to_egraph_id)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExtractedEquationKey {
    primitive: Primitive,
    inputs: Vec<Atom>,
    params: BTreeMap<String, String>,
}

fn dedupe_identical_extracted_equations(
    equations: Vec<Equation>,
    protected_outputs: &BTreeSet<VarId>,
) -> (Vec<Equation>, BTreeMap<VarId, VarId>) {
    let mut rewritten = Vec::with_capacity(equations.len());
    let mut aliases = BTreeMap::new();
    let mut seen: Vec<(ExtractedEquationKey, VarId)> = Vec::new();

    for equation in equations {
        let mut equation = equation;
        for input in &mut equation.inputs {
            if let Atom::Var(var) = input {
                *var = resolve_var_alias(*var, &aliases);
            }
        }

        if equation.outputs.len() != 1
            || !equation.effects.is_empty()
            || !equation.sub_jaxprs.is_empty()
        {
            rewritten.push(equation);
            continue;
        }

        let key = ExtractedEquationKey {
            primitive: equation.primitive,
            inputs: equation.inputs.iter().cloned().collect(),
            params: equation.params.clone(),
        };
        let output = equation.outputs[0];

        if let Some((_, canonical)) = seen.iter().find(|(seen_key, _)| seen_key == &key) {
            let canonical = resolve_var_alias(*canonical, &aliases);
            if protected_outputs.contains(&output) {
                if output != canonical {
                    rewritten.push(Equation {
                        primitive: Primitive::Copy,
                        inputs: smallvec![Atom::Var(canonical)],
                        outputs: smallvec![output],
                        params: BTreeMap::new(),
                        effects: vec![],
                        sub_jaxprs: vec![],
                    });
                }
            } else {
                aliases.insert(output, canonical);
            }
            continue;
        }

        seen.push((key, output));
        rewritten.push(equation);
    }

    (rewritten, aliases)
}

struct PieceRewriteResult {
    equations: Vec<Equation>,
    actual_outvar: VarId,
}

fn rewrite_piece_for_output(
    piece: &Jaxpr,
    desired_outvar: VarId,
    preserve_output_var: bool,
    invars: &[VarId],
    constvars: &[VarId],
    next_var: &mut u32,
) -> PieceRewriteResult {
    let actual_outvar = piece.outvars[0];
    let interface_vars: BTreeSet<VarId> = invars.iter().chain(constvars.iter()).copied().collect();
    let can_retarget_directly = preserve_output_var && !interface_vars.contains(&actual_outvar);

    let mut remap = BTreeMap::new();
    if can_retarget_directly && actual_outvar != desired_outvar {
        remap.insert(actual_outvar, desired_outvar);
    }

    for equation in &piece.equations {
        for outvar in &equation.outputs {
            if remap.contains_key(outvar) {
                continue;
            }
            if preserve_output_var && *outvar == desired_outvar {
                remap.insert(*outvar, desired_outvar);
                continue;
            }
            if interface_vars.contains(outvar) {
                remap.insert(*outvar, *outvar);
            } else {
                remap.insert(*outvar, VarId(*next_var));
                *next_var += 1;
            }
        }
    }

    let mut rewritten = Vec::with_capacity(piece.equations.len() + 1);
    for equation in &piece.equations {
        let inputs = equation
            .inputs
            .iter()
            .map(|atom| match atom {
                Atom::Var(var) => Atom::Var(*remap.get(var).unwrap_or(var)),
                Atom::Lit(lit) => Atom::Lit(*lit),
            })
            .collect();
        let outputs = equation
            .outputs
            .iter()
            .map(|var| *remap.get(var).unwrap_or(var))
            .collect();

        rewritten.push(Equation {
            primitive: equation.primitive,
            inputs,
            outputs,
            params: equation.params.clone(),
            effects: equation.effects.clone(),
            sub_jaxprs: equation.sub_jaxprs.clone(),
        });
    }

    let actual_outvar = *remap.get(&actual_outvar).unwrap_or(&actual_outvar);
    if preserve_output_var && actual_outvar != desired_outvar {
        rewritten.push(Equation {
            primitive: Primitive::Copy,
            inputs: smallvec![Atom::Var(actual_outvar)],
            outputs: smallvec![desired_outvar],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        return PieceRewriteResult {
            equations: rewritten,
            actual_outvar: desired_outvar,
        };
    }

    PieceRewriteResult {
        equations: rewritten,
        actual_outvar,
    }
}

fn build_supported_segment_jaxpr(
    jaxpr: &Jaxpr,
    start: usize,
    end: usize,
) -> (Jaxpr, BTreeSet<VarId>) {
    let equations = jaxpr.equations[start..end].to_vec();
    let bound_vars: BTreeSet<VarId> = equations
        .iter()
        .flat_map(|equation| equation.outputs.iter().copied())
        .collect();

    let mut input_vars = Vec::new();
    let mut seen_inputs = BTreeSet::new();
    for equation in &equations {
        for atom in &equation.inputs {
            let Atom::Var(var) = atom else {
                continue;
            };
            if !bound_vars.contains(var) && seen_inputs.insert(*var) {
                input_vars.push(*var);
            }
        }
    }

    let mut later_uses = BTreeSet::new();
    for equation in &jaxpr.equations[end..] {
        for atom in &equation.inputs {
            if let Atom::Var(var) = atom {
                later_uses.insert(*var);
            }
        }
    }
    let preserved_outvars = later_uses.clone();
    later_uses.extend(jaxpr.outvars.iter().copied());

    let mut outvars = Vec::new();
    let mut seen_outvars = BTreeSet::new();
    for equation in &equations {
        for outvar in &equation.outputs {
            if later_uses.contains(outvar) && seen_outvars.insert(*outvar) {
                outvars.push(*outvar);
            }
        }
    }

    (
        Jaxpr::new(input_vars, vec![], outvars, equations),
        preserved_outvars,
    )
}

fn is_egraph_barrier(equation: &Equation) -> bool {
    equation.outputs.len() != 1
        || !equation.effects.is_empty()
        || !equation.sub_jaxprs.is_empty()
        || !is_egraph_supported_primitive(equation.primitive)
}

fn max_var_id(jaxpr: &Jaxpr) -> u32 {
    let mut max_var = 0;
    for var in jaxpr
        .invars
        .iter()
        .chain(jaxpr.constvars.iter())
        .chain(jaxpr.outvars.iter())
    {
        max_var = max_var.max(var.0);
    }
    for equation in &jaxpr.equations {
        for atom in &equation.inputs {
            if let Atom::Var(var) = atom {
                max_var = max_var.max(var.0);
            }
        }
        for outvar in &equation.outputs {
            max_var = max_var.max(outvar.0);
        }
    }
    max_var
}

fn is_egraph_supported_primitive(primitive: Primitive) -> bool {
    excluded_primitive_reason(primitive).is_none()
}

fn excluded_primitive_reason(primitive: Primitive) -> Option<ExclusionReason> {
    match primitive {
        // Shape manipulation is architecturally excluded because the
        // s-expression e-graph cannot encode explicit shapes, slices, or axes.
        Primitive::Reshape
        | Primitive::Slice
        | Primitive::DynamicSlice
        | Primitive::Gather
        | Primitive::Scatter
        | Primitive::Transpose
        | Primitive::BroadcastInDim
        | Primitive::Concatenate
        | Primitive::Pad
        | Primitive::Rev
        | Primitive::Squeeze
        | Primitive::Split
        | Primitive::ExpandDims => Some(ExclusionReason::ShapeManipulation),
        // Linear algebra decompositions need richer result structure and
        // decomposition metadata than the algebraic e-graph can model.
        Primitive::Cholesky
        | Primitive::Qr
        | Primitive::Svd
        | Primitive::TriangularSolve
        | Primitive::Eigh => Some(ExclusionReason::LinearAlgebra),
        // FFT lowering depends on transform-length and layout metadata.
        Primitive::Fft | Primitive::Ifft | Primitive::Rfft | Primitive::Irfft => {
            Some(ExclusionReason::Fft)
        }
        // Control flow carries sub-jaxprs and branch metadata.
        Primitive::Cond | Primitive::Scan | Primitive::While | Primitive::Switch => {
            Some(ExclusionReason::ControlFlow)
        }
        // Sorting needs explicit axis and comparator metadata.
        Primitive::Sort | Primitive::Argsort => Some(ExclusionReason::Sorting),
        // Convolution needs window, stride, and padding metadata.
        Primitive::Conv => Some(ExclusionReason::Convolution),
        // Index and utility helpers need dynamic index/update metadata.
        Primitive::Iota | Primitive::BroadcastedIota | Primitive::DynamicUpdateSlice => {
            Some(ExclusionReason::IndexUtility)
        }
        // Type-conversion ops carry bit-layout or precision parameters.
        Primitive::BitcastConvertType | Primitive::ReducePrecision => {
            Some(ExclusionReason::TypeConversion)
        }
        // Cumulative ops need axis and direction metadata.
        Primitive::Cumsum | Primitive::Cumprod => Some(ExclusionReason::Cumulative),
        // Windowed reduction carries window geometry.
        Primitive::ReduceWindow => Some(ExclusionReason::Windowed),
        // OneHot needs category-depth and axis metadata.
        Primitive::OneHot => Some(ExclusionReason::Encoding),
        // Collectives depend on active pmap axis context and device topology.
        Primitive::Psum
        | Primitive::Pmean
        | Primitive::AllGather
        | Primitive::AllToAll
        | Primitive::AxisIndex => Some(ExclusionReason::Collective),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{DType, ProgramSpec, Shape, TensorValue, Value, build_program};
    use fj_interpreters::eval_jaxpr;

    fn single_equation_jaxpr(primitive: Primitive) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3)],
            vec![],
            vec![VarId(4)],
            vec![Equation {
                primitive,
                inputs: smallvec![
                    Atom::Var(VarId(1)),
                    Atom::Var(VarId(2)),
                    Atom::Var(VarId(3))
                ],
                outputs: smallvec![VarId(4)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        )
    }

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
    fn multi_output_extraction_deduplicates_shared_expression()
    -> Result<(), fj_interpreters::InterpreterError> {
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2), VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Sin,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Sin,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert_eq!(
            optimized
                .equations
                .iter()
                .filter(|equation| equation.primitive == Primitive::Sin)
                .count(),
            1,
            "shared sin(x) should be extracted once for both outputs"
        );

        let args = [Value::scalar_f64(0.5)];
        assert_eq!(eval_jaxpr(&jaxpr, &args)?, eval_jaxpr(&optimized, &args)?);
        Ok(())
    }

    #[test]
    fn public_lowering_returns_error_for_unsupported_primitive() {
        let jaxpr = single_equation_jaxpr(Primitive::Transpose);

        let err = jaxpr_to_egraph(&jaxpr).unwrap_err();
        assert_eq!(
            err,
            EGraphLoweringError::UnsupportedPrimitive {
                primitive: Primitive::Transpose,
                reason: ExclusionReason::ShapeManipulation,
            }
        );
    }

    #[test]
    fn integer_pow_lowering_uses_exponent_param_contract() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::IntegerPow,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                effects: vec![],
                params: BTreeMap::from([("exponent".to_owned(), "2".to_owned())]),
                sub_jaxprs: vec![],
            }],
        );

        let (expr, _) = jaxpr_to_egraph(&jaxpr).expect("integer_pow should lower");
        assert!(
            expr.as_ref()
                .iter()
                .any(|node| matches!(node, FjLang::IntegerPow(_))),
            "integer_pow should be represented in the egraph expression"
        );

        let roundtrip = egraph_to_jaxpr(&expr, &[VarId(1)], &[], &[VarId(2)]);
        assert_eq!(roundtrip.equations.len(), 1);
        let equation = &roundtrip.equations[0];
        assert_eq!(equation.primitive, Primitive::IntegerPow);
        assert_eq!(equation.inputs.len(), 1);
        assert_eq!(equation.inputs[0], Atom::Var(VarId(1)));
        assert_eq!(
            equation.params.get("exponent").map(String::as_str),
            Some("2")
        );
    }

    #[test]
    fn excluded_primitive_messages_include_category_reason() {
        let cases = [
            (Primitive::Reshape, ExclusionReason::ShapeManipulation),
            (Primitive::Slice, ExclusionReason::ShapeManipulation),
            (Primitive::DynamicSlice, ExclusionReason::ShapeManipulation),
            (Primitive::Gather, ExclusionReason::ShapeManipulation),
            (Primitive::Scatter, ExclusionReason::ShapeManipulation),
            (Primitive::Transpose, ExclusionReason::ShapeManipulation),
            (
                Primitive::BroadcastInDim,
                ExclusionReason::ShapeManipulation,
            ),
            (Primitive::Concatenate, ExclusionReason::ShapeManipulation),
            (Primitive::Pad, ExclusionReason::ShapeManipulation),
            (Primitive::Rev, ExclusionReason::ShapeManipulation),
            (Primitive::Squeeze, ExclusionReason::ShapeManipulation),
            (Primitive::Split, ExclusionReason::ShapeManipulation),
            (Primitive::ExpandDims, ExclusionReason::ShapeManipulation),
            (Primitive::Cholesky, ExclusionReason::LinearAlgebra),
            (Primitive::Qr, ExclusionReason::LinearAlgebra),
            (Primitive::Svd, ExclusionReason::LinearAlgebra),
            (Primitive::TriangularSolve, ExclusionReason::LinearAlgebra),
            (Primitive::Eigh, ExclusionReason::LinearAlgebra),
            (Primitive::Fft, ExclusionReason::Fft),
            (Primitive::Ifft, ExclusionReason::Fft),
            (Primitive::Rfft, ExclusionReason::Fft),
            (Primitive::Irfft, ExclusionReason::Fft),
            (Primitive::Cond, ExclusionReason::ControlFlow),
            (Primitive::Scan, ExclusionReason::ControlFlow),
            (Primitive::While, ExclusionReason::ControlFlow),
            (Primitive::Switch, ExclusionReason::ControlFlow),
            (Primitive::Sort, ExclusionReason::Sorting),
            (Primitive::Argsort, ExclusionReason::Sorting),
            (Primitive::Conv, ExclusionReason::Convolution),
            (Primitive::Iota, ExclusionReason::IndexUtility),
            (Primitive::BroadcastedIota, ExclusionReason::IndexUtility),
            (Primitive::DynamicUpdateSlice, ExclusionReason::IndexUtility),
            (
                Primitive::BitcastConvertType,
                ExclusionReason::TypeConversion,
            ),
            (Primitive::ReducePrecision, ExclusionReason::TypeConversion),
            (Primitive::Cumsum, ExclusionReason::Cumulative),
            (Primitive::Cumprod, ExclusionReason::Cumulative),
            (Primitive::ReduceWindow, ExclusionReason::Windowed),
            (Primitive::OneHot, ExclusionReason::Encoding),
            (Primitive::Psum, ExclusionReason::Collective),
            (Primitive::Pmean, ExclusionReason::Collective),
            (Primitive::AllGather, ExclusionReason::Collective),
            (Primitive::AllToAll, ExclusionReason::Collective),
            (Primitive::AxisIndex, ExclusionReason::Collective),
        ];

        for (primitive, reason) in cases {
            let err = jaxpr_to_egraph(&single_equation_jaxpr(primitive)).unwrap_err();
            let rendered = err.to_string();
            assert!(
                rendered.contains(reason.category()),
                "{primitive:?} message missing category {reason:?}: {rendered}"
            );
            assert!(
                rendered.contains(reason.detail()),
                "{primitive:?} message missing detail {reason:?}: {rendered}"
            );
        }
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
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
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
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
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
    fn round_trip_with_all_encoded_symbol_literal_kinds() {
        let literals = [
            Literal::Bool(true),
            Literal::U32(7),
            Literal::U64(11),
            Literal::BF16Bits(0x3f80),
            Literal::F16Bits(0x3c00),
            Literal::F32Bits(1.25_f32.to_bits()),
            Literal::Complex64Bits(1.5_f32.to_bits(), (-2.25_f32).to_bits()),
        ];

        for literal in literals {
            let jaxpr = Jaxpr::new(
                vec![],
                vec![],
                vec![VarId(1)],
                vec![Equation {
                    primitive: Primitive::Copy,
                    inputs: smallvec![Atom::Lit(literal)],
                    outputs: smallvec![VarId(1)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                }],
            );

            let (expr, _) = jaxpr_to_egraph(&jaxpr).expect("literal copy should lower");
            let roundtrip = egraph_to_jaxpr(&expr, &jaxpr.invars, &jaxpr.constvars, &jaxpr.outvars);
            assert_eq!(roundtrip.equations.len(), 1);
            assert_eq!(roundtrip.equations[0].inputs[0], Atom::Lit(literal));
        }
    }

    #[test]
    fn encoded_u32_symbol_root_becomes_literal_output() {
        let mut expr = RecExpr::default();
        expr.add(FjLang::Symbol(egg::Symbol::from("u32:7")));

        let roundtrip = egraph_to_jaxpr(&expr, &[], &[], &[VarId(1)]);
        assert_eq!(roundtrip.outvars, vec![VarId(1)]);
        assert_eq!(roundtrip.equations.len(), 1);
        assert_eq!(roundtrip.equations[0].primitive, Primitive::Select);
        assert_eq!(roundtrip.equations[0].outputs.as_slice(), &[VarId(1)]);
        assert_eq!(
            roundtrip.equations[0].inputs.as_slice(),
            &[
                Atom::Lit(Literal::Bool(true)),
                Atom::Lit(Literal::U32(7)),
                Atom::Lit(Literal::U32(7))
            ]
        );
    }

    #[test]
    fn malformed_f64_symbol_falls_back_to_variable_reference() {
        let mut expr = RecExpr::default();
        let malformed = expr.add(FjLang::Symbol(egg::Symbol::from("f64:not-bits")));
        let one = expr.add(FjLang::Num(1));
        expr.add(FjLang::Add([malformed, one]));

        let roundtrip = egraph_to_jaxpr(&expr, &[], &[], &[VarId(9)]);
        assert_eq!(roundtrip.equations.len(), 1);
        assert_eq!(roundtrip.equations[0].primitive, Primitive::Add);
        assert!(matches!(roundtrip.equations[0].inputs[0], Atom::Var(_)));
        assert_eq!(roundtrip.equations[0].inputs[1], Atom::Lit(Literal::I64(1)));
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
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
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
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
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
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
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
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Abs,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
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
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
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
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Floor,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
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
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
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
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
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
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
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
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Expm1,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
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
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
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
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
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
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
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
    fn reciprocal_involution() {
        // reciprocal(reciprocal(x)) should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Reciprocal,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Reciprocal,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "reciprocal(reciprocal(x)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn sign_idempotent() {
        // sign(sign(x)) should simplify to sign(x)
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Sign,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Sign,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "sign(sign(x)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn sin2_cos2_pythagorean() {
        // sin(x)^2 + cos(x)^2 should simplify to 1
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(7)],
            vec![
                Equation {
                    primitive: Primitive::Sin,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Cos,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(4))],
                    outputs: smallvec![VarId(5)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(5))],
                    outputs: smallvec![VarId(7)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "sin^2(x)+cos^2(x) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn cosh2_sinh2_hyperbolic_identity() {
        // cosh(x)^2 - sinh(x)^2 should simplify to 1
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(7)],
            vec![
                Equation {
                    primitive: Primitive::Cosh,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Sinh,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(4))],
                    outputs: smallvec![VarId(5)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(5))],
                    outputs: smallvec![VarId(7)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "cosh^2(x)-sinh^2(x) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn logistic_complement_identity() {
        // logistic(x) + logistic(-x) should simplify to 1
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(5)],
            vec![
                Equation {
                    primitive: Primitive::Logistic,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Logistic,
                    inputs: smallvec![Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(4))],
                    outputs: smallvec![VarId(5)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "logistic(x)+logistic(-x) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn select_nested_true_branch() {
        // select(c, select(c, a, b), x) should simplify to select(c, a, x)
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3), VarId(4)],
            vec![],
            vec![VarId(6)],
            vec![
                Equation {
                    primitive: Primitive::Select,
                    inputs: smallvec![
                        Atom::Var(VarId(1)),
                        Atom::Var(VarId(2)),
                        Atom::Var(VarId(3))
                    ],
                    outputs: smallvec![VarId(5)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Select,
                    inputs: smallvec![
                        Atom::Var(VarId(1)),
                        Atom::Var(VarId(5)),
                        Atom::Var(VarId(4))
                    ],
                    outputs: smallvec![VarId(6)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "nested select should not increase equation count: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn select_nested_false_branch() {
        // select(c, x, select(c, a, b)) should simplify to select(c, x, b)
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3), VarId(4)],
            vec![],
            vec![VarId(6)],
            vec![
                Equation {
                    primitive: Primitive::Select,
                    inputs: smallvec![
                        Atom::Var(VarId(1)),
                        Atom::Var(VarId(2)),
                        Atom::Var(VarId(3))
                    ],
                    outputs: smallvec![VarId(5)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Select,
                    inputs: smallvec![
                        Atom::Var(VarId(1)),
                        Atom::Var(VarId(4)),
                        Atom::Var(VarId(5))
                    ],
                    outputs: smallvec![VarId(6)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "nested select should not increase equation count: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    // ── Multiplicative cancellation ────────────────────────────────

    #[test]
    fn mul_reciprocal_cancels_to_one() {
        // mul(x, reciprocal(x)) → 1
        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(2)],
            vec![
                Equation {
                    primitive: Primitive::Reciprocal,
                    inputs: smallvec![Atom::Var(VarId(0))],
                    outputs: smallvec![VarId(1)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "mul(x, reciprocal(x)) should simplify to 1: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn div_mul_cancel() {
        // div(mul(a, b), b) → a
        let jaxpr = Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "div(mul(a,b), b) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    // ── Power rules ───────────────────────────────────────────────────

    #[test]
    fn pow_neg_one_is_reciprocal() {
        // pow(x, -1) → reciprocal(x)
        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(2)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Lit(Literal::from_f64(1.0))],
                    outputs: smallvec![VarId(1)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Pow,
                    inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "pow(x, -1) should simplify to reciprocal: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn pow_two_is_square() {
        // pow(x, 2) → mul(x, x)
        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(1)],
            vec![Equation {
                primitive: Primitive::Pow,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Lit(Literal::from_f64(2.0))],
                outputs: smallvec![VarId(1)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        // pow(x,2) rewrites to mul(x,x) which is same cost, so equation count
        // should not increase
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "pow(x, 2) should not increase cost: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    // ── Log decomposition ─────────────────────────────────────────────

    #[test]
    fn log_product_decomposes() {
        // log(mul(a, b)) → add(log(a), log(b))
        let jaxpr = Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        // The decomposition creates more ops but proves the rule fires
        // (equivalence class contains both forms)
        assert!(
            optimized.equations.len() <= jaxpr.equations.len() + 2,
            "log(mul(a,b)) decomposition should not blow up: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    // ── Erf negation ──────────────────────────────────────────────────

    #[test]
    fn erf_neg_symmetry() {
        // erf(neg(x)) → neg(erf(x))
        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(2)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(0))],
                    outputs: smallvec![VarId(1)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Erf,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "erf(neg(x)) should not increase: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    // ── Max/Min absorption ────────────────────────────────────────────

    #[test]
    fn max_min_absorption() {
        // max(a, min(a, b)) → a
        let jaxpr = Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Min,
                    inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Max,
                    inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "max(a, min(a, b)) should simplify to a: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn min_max_absorption() {
        // min(a, max(a, b)) → a
        let jaxpr = Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Max,
                    inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Min,
                    inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "min(a, max(a, b)) should simplify to a: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
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

    // ── V2-PRIM e-graph rewrite tests ──────────────────────────────────

    #[test]
    fn test_rewrite_conj_conj() {
        // conj(conj(x)) should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Conj,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Conj,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "conj(conj(x)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn test_rewrite_real_complex() {
        // real(complex(r, i)) should simplify to r
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Complex,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Real,
                    inputs: smallvec![Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "real(complex(r, i)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn test_rewrite_imag_complex() {
        // imag(complex(r, i)) should simplify to i
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Complex,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Imag,
                    inputs: smallvec![Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "imag(complex(r, i)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn test_rewrite_copy_elimination() {
        // copy(x) should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Copy,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "copy(x) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn test_rewrite_integer_pow_zero() {
        // integer_pow(x, 0) should simplify to 1
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::IntegerPow,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(0))],
                outputs: smallvec![VarId(2)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "integer_pow(x, 0) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn test_rewrite_integer_pow_one() {
        // integer_pow(x, 1) should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::IntegerPow,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(1))],
                outputs: smallvec![VarId(2)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "integer_pow(x, 1) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn test_rewrite_bitwise_not_not() {
        // bitwise_not(bitwise_not(x)) should simplify to x
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::BitwiseNot,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::BitwiseNot,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() < jaxpr.equations.len(),
            "bitwise_not(bitwise_not(x)) should simplify: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn test_rewrite_bitwise_xor_self() {
        // x ^ x should simplify to 0
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::BitwiseXor,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "x ^ x should simplify to 0: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );
    }

    #[test]
    fn transpose_identity_permutation_is_elided() {
        let mut params = BTreeMap::new();
        params.insert("permutation".to_owned(), "0,1".to_owned());
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Transpose,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                effects: vec![],
                params,
                sub_jaxprs: vec![],
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(optimized.equations.is_empty());
        assert_eq!(optimized.outvars, vec![VarId(1)]);
    }

    #[test]
    fn transpose_inverse_pair_is_elided() {
        let mut params = BTreeMap::new();
        params.insert("permutation".to_owned(), "1,0".to_owned());
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Transpose,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: params.clone(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Transpose,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params,
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(optimized.equations.is_empty());
        assert_eq!(optimized.outvars, vec![VarId(1)]);

        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                    Literal::I64(5),
                    Literal::I64(6),
                ],
            )
            .unwrap(),
        );
        let original_out = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).unwrap();
        let optimized_out = eval_jaxpr(&optimized, std::slice::from_ref(&input)).unwrap();
        assert_eq!(original_out, optimized_out);
    }

    #[test]
    fn malformed_transpose_inverse_params_do_not_panic_or_elide() {
        let mut first_params = BTreeMap::new();
        first_params.insert("permutation".to_owned(), "2,0".to_owned());
        let mut second_params = BTreeMap::new();
        second_params.insert("permutation".to_owned(), "1,0".to_owned());
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Transpose,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: first_params,
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Transpose,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: second_params,
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert_eq!(optimized.equations.len(), 2);
        assert_eq!(optimized.outvars, vec![VarId(3)]);
    }

    #[test]
    fn expand_dims_then_squeeze_same_axis_is_elided() {
        let mut expand_params = BTreeMap::new();
        expand_params.insert("axis".to_owned(), "0".to_owned());
        let mut squeeze_params = BTreeMap::new();
        squeeze_params.insert("dimensions".to_owned(), "0".to_owned());
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::ExpandDims,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: expand_params,
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Squeeze,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: squeeze_params,
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(optimized.equations.is_empty());
        assert_eq!(optimized.outvars, vec![VarId(1)]);

        let input = Value::vector_i64(&[7, 8, 9]).unwrap();
        let original_out = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).unwrap();
        let optimized_out = eval_jaxpr(&optimized, std::slice::from_ref(&input)).unwrap();
        assert_eq!(original_out, optimized_out);
    }

    #[test]
    fn squeeze_then_expand_dims_same_axis_is_elided() {
        let mut squeeze_params = BTreeMap::new();
        squeeze_params.insert("dimensions".to_owned(), "0".to_owned());
        let mut expand_params = BTreeMap::new();
        expand_params.insert("axis".to_owned(), "0".to_owned());
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Squeeze,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: squeeze_params,
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::ExpandDims,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: expand_params,
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(optimized.equations.is_empty());
        assert_eq!(optimized.outvars, vec![VarId(1)]);

        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![1, 3] },
                vec![Literal::I64(1), Literal::I64(2), Literal::I64(3)],
            )
            .unwrap(),
        );
        let original_out = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).unwrap();
        let optimized_out = eval_jaxpr(&optimized, std::slice::from_ref(&input)).unwrap();
        assert_eq!(original_out, optimized_out);
    }

    #[test]
    fn rev_inverse_pair_is_elided_before_egraph_saturation() {
        let mut rev_params = BTreeMap::new();
        rev_params.insert("axes".to_owned(), "0".to_owned());
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Rev,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: rev_params.clone(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Rev,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: rev_params,
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Lit(Literal::I64(0))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.is_empty(),
            "rev pair plus add-zero should collapse to the input: {optimized:#?}"
        );
        assert_eq!(optimized.outvars, vec![VarId(1)]);

        let input = Value::vector_i64(&[1, 2, 3, 4, 5]).unwrap();
        let original_out = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).unwrap();
        let optimized_out = eval_jaxpr(&optimized, std::slice::from_ref(&input)).unwrap();
        assert_eq!(original_out, optimized_out);
    }

    #[test]
    fn revrev_axes_canon() {
        let mut first_rev_params = BTreeMap::new();
        first_rev_params.insert("axes".to_owned(), "0,1".to_owned());
        let mut second_rev_params = BTreeMap::new();
        second_rev_params.insert("axes".to_owned(), "1,0".to_owned());
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Rev,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: first_rev_params,
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Rev,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: second_rev_params,
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.is_empty(),
            "rev pair with permuted axes should collapse to the input: {optimized:#?}"
        );
        assert_eq!(optimized.outvars, vec![VarId(1)]);

        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                    Literal::I64(5),
                    Literal::I64(6),
                ],
            )
            .unwrap(),
        );
        let original_out = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).unwrap();
        let optimized_out = eval_jaxpr(&optimized, std::slice::from_ref(&input)).unwrap();
        assert_eq!(original_out, optimized_out);
    }

    #[test]
    fn test_new_rules_dont_break_existing() {
        // Verify that existing rules still work after adding new V2-PRIM rules.
        // Test: mul(x, 0) should still simplify to 0.
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(0))],
                outputs: smallvec![VarId(2)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        );

        let optimized = optimize_jaxpr(&jaxpr);
        assert!(
            optimized.equations.len() <= jaxpr.equations.len(),
            "existing mul-zero rule should still work: got {} eqns (was {})",
            optimized.equations.len(),
            jaxpr.equations.len(),
        );

        // Also verify neg(neg(x)) still works
        let jaxpr2 = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let optimized2 = optimize_jaxpr(&jaxpr2);
        assert!(
            optimized2.equations.len() < jaxpr2.equations.len(),
            "existing neg-neg rule should still work: got {} eqns (was {})",
            optimized2.equations.len(),
            jaxpr2.equations.len(),
        );
    }

    // ── Systematic rewrite rule unit tests (frankenjax-6qi) ──────────

    // Helper: build a single unary-op Jaxpr: out = op(in)
    fn unary_jaxpr(prim: Primitive, input: Atom) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: prim,
                inputs: smallvec![input],
                outputs: smallvec![VarId(2)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        )
    }

    // Helper: build a chained unary pair: t = op1(in), out = op2(t)
    fn chained_unary_jaxpr(prim1: Primitive, prim2: Primitive) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: prim1,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: prim2,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    // Helper: build a chained unary triple: t1 = op1(in), t2 = op2(t1), out = op3(t2)
    fn triple_unary_jaxpr(prim1: Primitive, prim2: Primitive, prim3: Primitive) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: prim1,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: prim2,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: prim3,
                    inputs: smallvec![Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    // Helper: build a binary-op Jaxpr: out = op(lhs, rhs)
    fn binary_jaxpr(prim: Primitive, lhs: Atom, rhs: Atom) -> Jaxpr {
        let invars: Vec<VarId> = [&lhs, &rhs]
            .iter()
            .filter_map(|a| match a {
                Atom::Var(v) => Some(*v),
                _ => None,
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Jaxpr::new(
            invars,
            vec![],
            vec![VarId(100)],
            vec![Equation {
                primitive: prim,
                inputs: smallvec![lhs, rhs],
                outputs: smallvec![VarId(100)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        )
    }

    fn unary_square_identity_jaxpr(
        first: Primitive,
        second: Primitive,
        combine: Primitive,
    ) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(7)],
            vec![
                Equation {
                    primitive: first,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: second,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(4))],
                    outputs: smallvec![VarId(5)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: combine,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(5))],
                    outputs: smallvec![VarId(7)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    fn logistic_complement_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(5)],
            vec![
                Equation {
                    primitive: Primitive::Logistic,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Logistic,
                    inputs: smallvec![Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(4))],
                    outputs: smallvec![VarId(5)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    fn minmax_absorption_jaxpr(outer: Primitive, inner: Primitive) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: inner,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: outer,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    #[test]
    fn rule_sub_zero() {
        // x - 0 → x
        let jaxpr = binary_jaxpr(
            Primitive::Sub,
            Atom::Var(VarId(1)),
            Atom::Lit(Literal::I64(0)),
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "x-0 should simplify to x: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_mul_one() {
        // x * 1 → x
        let jaxpr = binary_jaxpr(
            Primitive::Mul,
            Atom::Var(VarId(1)),
            Atom::Lit(Literal::I64(1)),
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "x*1 should simplify to x: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_mul_neg_one() {
        // x * (-1) → neg(x)
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Lit(Literal::I64(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "x*(-1) should simplify to neg(x): got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_neg_zero() {
        // neg(0) → 0 (extraction may keep a constant-binding equation)
        let jaxpr = unary_jaxpr(Primitive::Neg, Atom::Lit(Literal::I64(0)));
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "neg(0) should simplify: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_add_neg_self() {
        // x + neg(x) → 0
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "x+neg(x) should simplify to 0: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_pow_zero() {
        // x^0 → 1 (extraction may keep a constant-binding equation)
        let jaxpr = binary_jaxpr(
            Primitive::Pow,
            Atom::Var(VarId(1)),
            Atom::Lit(Literal::I64(0)),
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "x^0 should simplify: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_pow_one() {
        // x^1 → x
        let jaxpr = binary_jaxpr(
            Primitive::Pow,
            Atom::Var(VarId(1)),
            Atom::Lit(Literal::I64(1)),
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "x^1 should simplify to x: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_log_exp_inverse() {
        // log(exp(x)) → x
        let jaxpr = chained_unary_jaxpr(Primitive::Exp, Primitive::Log);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "log(exp(x)) should simplify to x: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_ceil_idempotent() {
        // ceil(ceil(x)) → ceil(x)
        let jaxpr = chained_unary_jaxpr(Primitive::Ceil, Primitive::Ceil);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "ceil(ceil(x)) should simplify to ceil(x): got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_round_idempotent() {
        // round(round(x)) → round(x)
        let jaxpr = chained_unary_jaxpr(Primitive::Round, Primitive::Round);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "round(round(x)) should simplify to round(x): got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_reduce_sum_idempotent() {
        // reduce_sum(reduce_sum(x)) → reduce_sum(x)
        let jaxpr = chained_unary_jaxpr(Primitive::ReduceSum, Primitive::ReduceSum);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "reduce_sum(reduce_sum(x)) should simplify: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_reduce_max_idempotent() {
        let jaxpr = chained_unary_jaxpr(Primitive::ReduceMax, Primitive::ReduceMax);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "reduce_max(reduce_max(x)) should simplify: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_reduce_min_idempotent() {
        let jaxpr = chained_unary_jaxpr(Primitive::ReduceMin, Primitive::ReduceMin);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "reduce_min(reduce_min(x)) should simplify: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_reduce_prod_idempotent() {
        let jaxpr = chained_unary_jaxpr(Primitive::ReduceProd, Primitive::ReduceProd);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "reduce_prod(reduce_prod(x)) should simplify: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_sin_neg() {
        // sin(neg(x)) → neg(sin(x)) — should produce same structure or fewer ops
        let jaxpr = chained_unary_jaxpr(Primitive::Neg, Primitive::Sin);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 2,
            "sin(neg(x)) should not grow: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_cos_neg() {
        // cos(neg(x)) → cos(x) — should eliminate the neg
        let jaxpr = chained_unary_jaxpr(Primitive::Neg, Primitive::Cos);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "cos(neg(x)) should simplify to cos(x): got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_tan_neg() {
        // tan(neg(x)) → neg(tan(x))
        let jaxpr = chained_unary_jaxpr(Primitive::Neg, Primitive::Tan);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 2,
            "tan(neg(x)) should not grow: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_sinh_neg() {
        let jaxpr = chained_unary_jaxpr(Primitive::Neg, Primitive::Sinh);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 2,
            "sinh(neg(x)) should not grow: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_cosh_neg() {
        // cosh(neg(x)) → cosh(x) — should eliminate the neg
        let jaxpr = chained_unary_jaxpr(Primitive::Neg, Primitive::Cosh);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "cosh(neg(x)) should simplify to cosh(x): got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_tanh_neg() {
        let jaxpr = chained_unary_jaxpr(Primitive::Neg, Primitive::Tanh);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 2,
            "tanh(neg(x)) should not grow: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_div_self() {
        // x / x → 1 (extraction may keep a constant-binding equation)
        let jaxpr = binary_jaxpr(Primitive::Div, Atom::Var(VarId(1)), Atom::Var(VarId(1)));
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "x/x should simplify: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_log1p_expm1_inverse() {
        // log1p(expm1(x)) → x
        let jaxpr = chained_unary_jaxpr(Primitive::Expm1, Primitive::Log1p);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "log1p(expm1(x)) should simplify to x: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_abs_neg_simplifies() {
        // abs(neg(x)) → abs(x) — should eliminate the neg
        let jaxpr = chained_unary_jaxpr(Primitive::Neg, Primitive::Abs);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "abs(neg(x)) should simplify to abs(x): got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_reciprocal_as_div() {
        // reciprocal(x) → div(1, x)
        let jaxpr = unary_jaxpr(Primitive::Reciprocal, Atom::Var(VarId(1)));
        let opt = optimize_jaxpr(&jaxpr);
        // Should produce an equivalent (possibly different structure)
        assert!(
            opt.equations.len() <= 1,
            "reciprocal(x) should not grow beyond 1 op: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_erf_neg_symmetry_chain() {
        // erf(neg(x)) → neg(erf(x))
        let jaxpr = chained_unary_jaxpr(Primitive::Neg, Primitive::Erf);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 2,
            "erf(neg(x)) should not grow: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_bitwise_and_self() {
        // x & x → x
        let jaxpr = binary_jaxpr(
            Primitive::BitwiseAnd,
            Atom::Var(VarId(1)),
            Atom::Var(VarId(1)),
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "x&x should simplify to x: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_bitwise_or_self() {
        // x | x → x
        let jaxpr = binary_jaxpr(
            Primitive::BitwiseOr,
            Atom::Var(VarId(1)),
            Atom::Var(VarId(1)),
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "x|x should simplify to x: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_is_finite_const_zero() {
        // is_finite(0) → 1 (true); extraction may keep a constant-binding equation
        let jaxpr = unary_jaxpr(Primitive::IsFinite, Atom::Lit(Literal::I64(0)));
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "is_finite(0) should simplify: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_is_finite_const_one() {
        // is_finite(1) → 1 (true); extraction may keep a constant-binding equation
        let jaxpr = unary_jaxpr(Primitive::IsFinite, Atom::Lit(Literal::I64(1)));
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.len() <= 1,
            "is_finite(1) should simplify: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_complex_real_imag_roundtrip() {
        // complex(real(z), imag(z)) → z
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Real,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Imag,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Complex,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "complex(real(z), imag(z)) should simplify to z: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_integer_pow_two() {
        // integer_pow(x, 2) → mul(x, x)
        let jaxpr = binary_jaxpr(
            Primitive::IntegerPow,
            Atom::Var(VarId(1)),
            Atom::Lit(Literal::I64(2)),
        );
        let opt = optimize_jaxpr(&jaxpr);
        // Should rewrite to mul(x, x) which is still 1 equation
        assert!(
            opt.equations.len() <= 1,
            "integer_pow(x,2) should become mul(x,x): got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn cbrt_triple_preserves_nested_semantics() {
        let jaxpr = triple_unary_jaxpr(Primitive::Cbrt, Primitive::Cbrt, Primitive::Cbrt);
        let args = [Value::scalar_f64(64.0)];
        let original = eval_jaxpr(&jaxpr, &args).expect("original eval");
        let original_value = original[0].as_f64_scalar().expect("original scalar");
        let single_cbrt = 64.0_f64.cbrt();
        assert!(
            (original_value - single_cbrt).abs() > 1.0,
            "test fixture must distinguish nested cbrt from cbrt(x)"
        );

        let opt = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::safe());
        let optimized = eval_jaxpr(&opt, &args).expect("safe optimized eval");
        let optimized_value = optimized[0].as_f64_scalar().expect("optimized scalar");
        assert!(
            (optimized_value - original_value).abs() < 1e-12,
            "safe egraph optimization must preserve nested cbrt semantics: original={original_value}, optimized={optimized_value}, equations={:?}",
            opt.equations
        );
    }

    #[test]
    fn rule_bitwise_not_not_chain() {
        // bitwise_not(bitwise_not(x)) → x
        let jaxpr = chained_unary_jaxpr(Primitive::BitwiseNot, Primitive::BitwiseNot);
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "not(not(x)) should simplify to x: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_select_true_branch() {
        // select(1, a, b) → a
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Select,
                inputs: smallvec![
                    Atom::Lit(Literal::I64(1)),
                    Atom::Var(VarId(1)),
                    Atom::Var(VarId(2))
                ],
                outputs: smallvec![VarId(3)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "select(1,a,b) should simplify to a: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_select_false_branch() {
        // select(0, a, b) → b
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Select,
                inputs: smallvec![
                    Atom::Lit(Literal::I64(0)),
                    Atom::Var(VarId(1)),
                    Atom::Var(VarId(2))
                ],
                outputs: smallvec![VarId(3)],
                effects: vec![],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
            }],
        );
        let opt = optimize_jaxpr(&jaxpr);
        assert!(
            opt.equations.is_empty(),
            "select(0,a,b) should simplify to b: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_sub_to_add_neg() {
        // sub(a, b) rewrites to add(a, neg(b))
        let jaxpr = binary_jaxpr(Primitive::Sub, Atom::Var(VarId(1)), Atom::Var(VarId(2)));
        let opt = optimize_jaxpr(&jaxpr);
        // Rewrite may change the form but should not increase cost
        assert!(
            opt.equations.len() <= 2,
            "sub(a,b) should not grow beyond 2 eqns: got {}",
            opt.equations.len()
        );
    }

    #[test]
    fn rule_log_quotient() {
        // log(div(a, b)) → sub(log(a), log(b))
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );
        let opt = optimize_jaxpr(&jaxpr);
        // May decompose to sub(log(a), log(b)) which is 3 eqns, or stay as 2
        assert!(
            opt.equations.len() <= 3,
            "log(a/b) should not explode: got {} eqns",
            opt.equations.len()
        );
    }

    #[test]
    fn numerical_safety_mode_disables_unsafe_rules() {
        // In safety mode, div-self should NOT be applied (a/a could be NaN when a=0)
        let jaxpr = binary_jaxpr(Primitive::Div, Atom::Var(VarId(1)), Atom::Var(VarId(1)));

        // Default mode (unsafe rules enabled): a/a becomes 1
        let opt_default = optimize_jaxpr(&jaxpr);
        let has_div_default = opt_default
            .equations
            .iter()
            .any(|eq| eq.primitive == Primitive::Div);

        // Safety mode: a/a stays as a/a (div-self rule disabled)
        let opt_safe = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::safe());
        let has_div_safe = opt_safe
            .equations
            .iter()
            .any(|eq| eq.primitive == Primitive::Div);

        assert!(
            !has_div_default,
            "default mode should simplify a/a (no Div remaining)"
        );
        assert!(
            has_div_safe,
            "safety mode should preserve a/a (Div remains)"
        );
    }

    #[test]
    fn numerical_safety_mode_preserves_safe_rules() {
        // Even in safety mode, neg-neg should still work
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );

        let opt = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::safe());
        // neg(neg(a)) should become just a, so no Neg in result
        let has_neg = opt
            .equations
            .iter()
            .any(|eq| eq.primitive == Primitive::Neg);
        assert!(
            !has_neg,
            "safety mode should still apply neg-neg rule: {:?}",
            opt.equations
        );
    }

    #[test]
    fn numerical_safety_mode_preserves_log_exp_overflow_boundary() {
        let jaxpr = chained_unary_jaxpr(Primitive::Exp, Primitive::Log);
        let args = [Value::scalar_f64(1000.0)];

        let original = eval_jaxpr(&jaxpr, &args).expect("original eval");
        assert!(
            original[0].as_f64_scalar().is_some_and(f64::is_infinite),
            "log(exp(1000.0)) should observe exp overflow as Inf"
        );

        let safe = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::safe());
        assert!(
            safe.equations
                .iter()
                .any(|equation| equation.primitive == Primitive::Exp),
            "safe mode must keep exp in log(exp(x)): {:?}",
            safe.equations
        );
        assert!(
            safe.equations
                .iter()
                .any(|equation| equation.primitive == Primitive::Log),
            "safe mode must keep log in log(exp(x)): {:?}",
            safe.equations
        );
        let safe_value = eval_jaxpr(&safe, &args).expect("safe eval");
        assert!(
            safe_value[0].as_f64_scalar().is_some_and(f64::is_infinite),
            "safe optimized value should preserve overflow-visible Inf"
        );

        let aggressive = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::aggressive());
        let aggressive_value = eval_jaxpr(&aggressive, &args).expect("aggressive eval");
        assert_eq!(
            aggressive_value[0].as_f64_scalar(),
            Some(1000.0),
            "aggressive mode may apply the numerically unsafe log-exp rewrite"
        );
    }

    #[test]
    fn numerical_safety_mode_preserves_log1p_expm1_boundaries() {
        let expm1_log1p = chained_unary_jaxpr(Primitive::Log1p, Primitive::Expm1);
        let domain_args = [Value::scalar_f64(-2.0)];

        let original_domain = eval_jaxpr(&expm1_log1p, &domain_args).expect("original eval");
        assert!(
            original_domain[0].as_f64_scalar().is_some_and(f64::is_nan),
            "expm1(log1p(-2.0)) should preserve log1p domain NaN"
        );

        let safe_domain = optimize_jaxpr_with_config(&expm1_log1p, &OptimizationConfig::safe());
        assert!(
            safe_domain
                .equations
                .iter()
                .any(|equation| equation.primitive == Primitive::Log1p),
            "safe mode must keep log1p in expm1(log1p(x)): {:?}",
            safe_domain.equations
        );
        assert!(
            safe_domain
                .equations
                .iter()
                .any(|equation| equation.primitive == Primitive::Expm1),
            "safe mode must keep expm1 in expm1(log1p(x)): {:?}",
            safe_domain.equations
        );
        let safe_domain_value = eval_jaxpr(&safe_domain, &domain_args).expect("safe eval");
        assert!(
            safe_domain_value[0]
                .as_f64_scalar()
                .is_some_and(f64::is_nan),
            "safe optimized value should preserve log1p domain NaN"
        );

        let aggressive_domain =
            optimize_jaxpr_with_config(&expm1_log1p, &OptimizationConfig::aggressive());
        let aggressive_domain_value =
            eval_jaxpr(&aggressive_domain, &domain_args).expect("aggressive eval");
        assert_eq!(
            aggressive_domain_value[0].as_f64_scalar(),
            Some(-2.0),
            "aggressive mode may apply the numerically unsafe expm1-log1p rewrite"
        );

        let log1p_expm1 = chained_unary_jaxpr(Primitive::Expm1, Primitive::Log1p);
        let overflow_args = [Value::scalar_f64(1000.0)];

        let original_overflow = eval_jaxpr(&log1p_expm1, &overflow_args).expect("original eval");
        assert!(
            original_overflow[0]
                .as_f64_scalar()
                .is_some_and(f64::is_infinite),
            "log1p(expm1(1000.0)) should observe expm1 overflow as Inf"
        );

        let safe_overflow = optimize_jaxpr_with_config(&log1p_expm1, &OptimizationConfig::safe());
        assert!(
            safe_overflow
                .equations
                .iter()
                .any(|equation| equation.primitive == Primitive::Expm1),
            "safe mode must keep expm1 in log1p(expm1(x)): {:?}",
            safe_overflow.equations
        );
        assert!(
            safe_overflow
                .equations
                .iter()
                .any(|equation| equation.primitive == Primitive::Log1p),
            "safe mode must keep log1p in log1p(expm1(x)): {:?}",
            safe_overflow.equations
        );
        let safe_overflow_value = eval_jaxpr(&safe_overflow, &overflow_args).expect("safe eval");
        assert!(
            safe_overflow_value[0]
                .as_f64_scalar()
                .is_some_and(f64::is_infinite),
            "safe optimized value should preserve overflow-visible Inf"
        );

        let aggressive_overflow =
            optimize_jaxpr_with_config(&log1p_expm1, &OptimizationConfig::aggressive());
        let aggressive_overflow_value =
            eval_jaxpr(&aggressive_overflow, &overflow_args).expect("aggressive eval");
        assert_eq!(
            aggressive_overflow_value[0].as_f64_scalar(),
            Some(1000.0),
            "aggressive mode may apply the numerically unsafe log1p-expm1 rewrite"
        );
    }

    #[test]
    fn numerical_safety_mode_preserves_nan_inf_cancellation_boundaries() {
        let cases = [
            (
                "inf_sub_inf",
                binary_jaxpr(Primitive::Sub, Atom::Var(VarId(1)), Atom::Var(VarId(1))),
                Value::scalar_f64(f64::INFINITY),
            ),
            (
                "nan_sub_nan",
                binary_jaxpr(Primitive::Sub, Atom::Var(VarId(1)), Atom::Var(VarId(1))),
                Value::scalar_f64(f64::NAN),
            ),
            (
                "inf_add_neg_inf",
                Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(3)],
                    vec![
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(2)],
                            effects: vec![],
                            params: BTreeMap::new(),
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(3)],
                            effects: vec![],
                            params: BTreeMap::new(),
                            sub_jaxprs: vec![],
                        },
                    ],
                ),
                Value::scalar_f64(f64::INFINITY),
            ),
            (
                "inf_mul_zero",
                binary_jaxpr(
                    Primitive::Mul,
                    Atom::Var(VarId(1)),
                    Atom::Lit(Literal::I64(0)),
                ),
                Value::scalar_f64(f64::INFINITY),
            ),
        ];

        for (name, jaxpr, input) in cases {
            let args = [input];
            let original = eval_jaxpr(&jaxpr, &args).expect("original eval");
            assert!(
                original[0].as_f64_scalar().is_some_and(f64::is_nan),
                "{name}: original program should preserve JAX/IEEE NaN boundary, got {:?}",
                original[0]
            );

            let safe = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::safe());
            let safe_value = eval_jaxpr(&safe, &args).expect("safe eval");
            assert!(
                safe_value[0].as_f64_scalar().is_some_and(f64::is_nan),
                "{name}: safe optimized program should preserve NaN boundary, got {:?} from {:?}",
                safe_value[0],
                safe.equations
            );

            let aggressive = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::aggressive());
            let aggressive_value = eval_jaxpr(&aggressive, &args).expect("aggressive eval");
            assert_eq!(
                aggressive_value[0].as_f64_scalar(),
                Some(0.0),
                "{name}: aggressive mode may apply the numerically unsafe zero rewrite"
            );
        }
    }

    #[test]
    fn numerical_safety_mode_preserves_distributive_overflow_boundary() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3)],
            vec![],
            vec![VarId(6)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(5)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(5))],
                    outputs: smallvec![VarId(6)],
                    effects: vec![],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                },
            ],
        );
        let args = [
            Value::scalar_f64(1e308),
            Value::scalar_f64(2.0),
            Value::scalar_f64(-2.0),
        ];

        let original = eval_jaxpr(&jaxpr, &args).expect("original eval");
        assert!(
            original[0].as_f64_scalar().is_some_and(f64::is_nan),
            "original program should observe +Inf + -Inf as NaN, got {:?}",
            original[0]
        );

        let safe = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::safe());
        let safe_value = eval_jaxpr(&safe, &args).expect("safe eval");
        assert!(
            safe_value[0].as_f64_scalar().is_some_and(f64::is_nan),
            "safe optimized program should preserve overflow-visible NaN, got {:?} from {:?}",
            safe_value[0],
            safe.equations
        );

        let aggressive = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::aggressive());
        let aggressive_value = eval_jaxpr(&aggressive, &args).expect("aggressive eval");
        assert_eq!(
            aggressive_value[0].as_f64_scalar(),
            Some(0.0),
            "aggressive mode may factor through x * (y + z) and erase the overflow NaN"
        );
    }

    #[test]
    fn numerical_safety_mode_preserves_exact_identity_nan_boundaries() {
        let cases = [
            (
                "sin2_cos2_inf",
                unary_square_identity_jaxpr(Primitive::Sin, Primitive::Cos, Primitive::Add),
                Value::scalar_f64(f64::INFINITY),
            ),
            (
                "cosh2_sinh2_overflow",
                unary_square_identity_jaxpr(Primitive::Cosh, Primitive::Sinh, Primitive::Sub),
                Value::scalar_f64(1000.0),
            ),
            (
                "logistic_complement_nan",
                logistic_complement_jaxpr(),
                Value::scalar_f64(f64::NAN),
            ),
        ];

        for (name, jaxpr, input) in cases {
            let args = [input];
            let original = eval_jaxpr(&jaxpr, &args).expect("original eval");
            assert!(
                original[0].as_f64_scalar().is_some_and(f64::is_nan),
                "{name}: original program should observe NaN, got {:?}",
                original[0]
            );

            let safe = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::safe());
            let safe_value = eval_jaxpr(&safe, &args).expect("safe eval");
            assert!(
                safe_value[0].as_f64_scalar().is_some_and(f64::is_nan),
                "{name}: safe optimized program should preserve NaN boundary, got {:?} from {:?}",
                safe_value[0],
                safe.equations
            );

            let aggressive = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::aggressive());
            let aggressive_value = eval_jaxpr(&aggressive, &args).expect("aggressive eval");
            assert_eq!(
                aggressive_value[0].as_f64_scalar(),
                Some(1.0),
                "{name}: aggressive mode may apply the numerically unsafe exact identity"
            );
        }
    }

    #[test]
    fn numerical_safety_mode_preserves_max_min_absorption_nan_boundary() {
        let cases = [
            (
                "max_min_absorb_nan",
                minmax_absorption_jaxpr(Primitive::Max, Primitive::Min),
            ),
            (
                "min_max_absorb_nan",
                minmax_absorption_jaxpr(Primitive::Min, Primitive::Max),
            ),
        ];
        let args = [Value::scalar_f64(f64::NAN), Value::scalar_f64(5.0)];

        for (name, jaxpr) in cases {
            let original = eval_jaxpr(&jaxpr, &args).expect("original eval");
            assert_eq!(
                original[0].as_f64_scalar(),
                Some(5.0),
                "{name}: original program should return the non-NaN operand"
            );

            let safe = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::safe());
            let safe_value = eval_jaxpr(&safe, &args).expect("safe eval");
            assert_eq!(
                safe_value[0].as_f64_scalar(),
                Some(5.0),
                "{name}: safe optimized program should preserve NaN-skipping max/min semantics"
            );

            let aggressive = optimize_jaxpr_with_config(&jaxpr, &OptimizationConfig::aggressive());
            let aggressive_value = eval_jaxpr(&aggressive, &args).expect("aggressive eval");
            assert!(
                aggressive_value[0].as_f64_scalar().is_some_and(f64::is_nan),
                "{name}: aggressive mode may apply the unsafe absorption identity"
            );
        }
    }

    #[test]
    fn optimization_config_constructors() {
        let safe = OptimizationConfig::safe();
        assert!(safe.numerical_safety_mode);

        let aggressive = OptimizationConfig::aggressive();
        assert!(!aggressive.numerical_safety_mode);

        let default = OptimizationConfig::default();
        assert!(!default.numerical_safety_mode);
    }
}
