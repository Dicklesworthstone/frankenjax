#![forbid(unsafe_code)]

pub mod batching;

use fj_cache::{CacheKey, CacheKeyError, CacheKeyInputRef, build_cache_key_ref};
use fj_core::{
    AbstractValue, Atom, CompatibilityMode, DType, GradWitness, Jaxpr, JitWitness, Literal,
    Primitive, Shape, TensorValue, TraceTransformLedger, Transform, TransformCompositionError,
    TransformCompositionProof, TransformProofStep, TransformRewriteWitness, Value, VmapWitness,
    jaxpr_fingerprint_sha256, verify_transform_composition, verify_transform_composition_parts,
    verify_transform_composition_steps,
};
use fj_interpreters::InterpreterError;
use fj_ledger::{
    CalibratedEstimate, CalibrationReport, ConformalPredictor, EProcess, EvidenceLedger,
    EvidenceSignal, LedgerEntry, LogDomainPosterior,
};
use fj_runtime::backend::{Backend, BackendError, BackendRegistry};
use fj_runtime::device::{DeviceId, DevicePlacement};
use fj_runtime::{
    AdmissionRejection, CancellationToken, ExecutionRegime, MemoryBudget, RuntimeAdmissionModel,
};
use fj_trace::simulate_nested_trace_contexts;
use std::collections::BTreeMap;

// ── Effect Token System ────────────────────────────────────────────

/// Per-effect runtime token for ordered side-effect tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectToken {
    pub effect_name: String,
    pub sequence_number: u64,
}

/// Context for threading effect tokens through a dispatch execution.
///
/// V1 scope: tracking only — records which effects were observed and in what
/// order while preserving every observation.
///
/// Uses a Vec to preserve insertion order for deterministic sequence threading.
#[derive(Debug, Clone)]
pub struct EffectContext {
    tokens: Vec<EffectToken>,
    next_sequence: u64,
}

impl EffectContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            next_sequence: 0,
        }
    }

    /// Record observation of a named effect and return its sequence token.
    pub fn thread_token(&mut self, effect_name: &str) -> EffectToken {
        let name = effect_name.to_owned();
        #[rustfmt::skip]
        let effect_token = EffectToken { // ubs:ignore — dispatch effect sequencing token, not a credential or secret value
            effect_name: name,
            sequence_number: self.next_sequence,
        };
        self.next_sequence += 1;
        self.tokens.push(effect_token.clone());
        effect_token
    }

    /// Finalize and return all observed effect tokens in sequence order.
    /// Tokens are already in insertion order, so no sorting needed.
    #[must_use]
    pub fn finalize(self) -> Vec<EffectToken> {
        self.tokens
    }

    /// Number of effect observations.
    #[must_use]
    pub fn effect_count(&self) -> usize {
        self.tokens.len()
    }
}

impl Default for EffectContext {
    fn default() -> Self {
        Self::new()
    }
}

fn thread_jaxpr_effect_tokens(effect_ctx: &mut EffectContext, jaxpr: &Jaxpr) {
    for effect in &jaxpr.effects {
        effect_ctx.thread_token(effect);
    }

    for equation in &jaxpr.equations {
        for effect in &equation.effects {
            effect_ctx.thread_token(effect);
        }
        for sub_jaxpr in &equation.sub_jaxprs {
            thread_jaxpr_effect_tokens(effect_ctx, sub_jaxpr);
        }
    }
}

/// Specifies which axis of an input/output is the batch axis for vmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisSpec {
    /// This input/output is batched along the given axis.
    Batched(i32),
    /// This input/output is not batched (broadcast to all batch elements).
    NotBatched,
}

impl AxisSpec {
    /// Resolve negative axis index to a positive one, given the tensor rank.
    fn resolve(self, rank: usize) -> Option<usize> {
        match self {
            Self::NotBatched => None,
            Self::Batched(axis) => {
                if axis >= 0 {
                    Some(axis as usize)
                } else {
                    let resolved = rank as i32 + axis;
                    if resolved >= 0 {
                        Some(resolved as usize)
                    } else {
                        None
                    }
                }
            }
        }
    }
}

/// Specifies in_axes/out_axes for vmap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VmapAxes {
    /// Same axis for all inputs/outputs.
    Uniform(AxisSpec),
    /// Per-input/output axis specification.
    PerArg(Vec<AxisSpec>),
}

impl Default for VmapAxes {
    fn default() -> Self {
        Self::Uniform(AxisSpec::Batched(0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchRequest {
    pub mode: CompatibilityMode,
    pub ledger: TraceTransformLedger,
    pub args: Vec<Value>,
    pub backend: String,
    pub compile_options: BTreeMap<String, String>,
    pub custom_hook: Option<String>,
    pub unknown_incompatible_features: Vec<String>,
}

#[derive(Debug)]
pub struct DispatchRequestRef<'a> {
    pub mode: CompatibilityMode,
    pub root_jaxpr: &'a Jaxpr,
    pub transform_stack: &'a [Transform],
    pub transform_evidence: &'a [String],
    pub args: Vec<Value>,
    pub backend: &'a str,
    pub compile_options: BTreeMap<String, String>,
    pub custom_hook: Option<&'a str>,
    pub unknown_incompatible_features: &'a [String],
    /// Optional precomputed dispatch metadata (composition proof + cache key).
    ///
    /// Both values are a pure function of the *static* dispatch inputs (jaxpr,
    /// transform stack/evidence, mode, backend, compile options, custom hook,
    /// incompatible features) — they never depend on the runtime `args`. When a
    /// caller invokes the same wrapped function repeatedly it can compute this
    /// once via [`prepare_dispatch_meta`] and pass it here, skipping the
    /// per-call fingerprint signature allocation + FNV/SHA-256 hashing.
    pub prepared: Option<&'a PreparedDispatchMeta>,
}

/// Precomputed, args-independent dispatch metadata.
///
/// Produced by [`prepare_dispatch_meta`] and threaded through
/// [`DispatchRequestRef::prepared`] so repeated calls of the same wrapped
/// function avoid recomputing the transform-composition proof and cache key
/// (both dominated by hashing the canonical Jaxpr fingerprint).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDispatchMeta {
    pub composition_proof: TransformCompositionProof,
    pub cache_key: CacheKey,
    static_inputs: PreparedDispatchStaticInputs,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedDispatchStaticInputs {
    mode: CompatibilityMode,
    root_jaxpr_fingerprint: String,
    transform_stack: Vec<Transform>,
    transform_evidence: Vec<String>,
    backend: String,
    compile_options: BTreeMap<String, String>,
    custom_hook: Option<String>,
    unknown_incompatible_features: Vec<String>,
}

impl PreparedDispatchStaticInputs {
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        mode: CompatibilityMode,
        root_jaxpr: &Jaxpr,
        transform_stack: &[Transform],
        transform_evidence: &[String],
        backend: &str,
        compile_options: &BTreeMap<String, String>,
        custom_hook: Option<&str>,
        unknown_incompatible_features: &[String],
    ) -> Self {
        Self {
            mode,
            root_jaxpr_fingerprint: root_jaxpr.canonical_fingerprint().to_owned(),
            transform_stack: transform_stack.to_vec(),
            transform_evidence: transform_evidence.to_vec(),
            backend: backend.to_owned(),
            compile_options: compile_options.clone(),
            custom_hook: custom_hook.map(str::to_owned),
            unknown_incompatible_features: unknown_incompatible_features.to_vec(),
        }
    }

    fn matches_request(&self, request: &DispatchRequestRef<'_>) -> bool {
        self.mode == request.mode
            && self.root_jaxpr_fingerprint == request.root_jaxpr.canonical_fingerprint()
            && self.transform_stack == request.transform_stack
            && self.transform_evidence == request.transform_evidence
            && self.backend == request.backend
            && self.compile_options == request.compile_options
            && self.custom_hook.as_deref() == request.custom_hook
            && self.unknown_incompatible_features == request.unknown_incompatible_features
    }
}

/// Compute the args-independent dispatch metadata for a wrapped function.
///
/// This performs exactly the same composition verification and cache-key
/// derivation that [`dispatch_ref`] does when `prepared` is `None`, so passing
/// the result back in produces byte-identical proofs, cache keys, and evidence
/// ledgers. Returns an error if composition verification or cache-key
/// construction would fail (e.g. strict-mode unknown incompatible features).
#[allow(clippy::too_many_arguments)]
pub fn prepare_dispatch_meta(
    mode: CompatibilityMode,
    root_jaxpr: &Jaxpr,
    transform_stack: &[Transform],
    transform_evidence: &[String],
    backend: &str,
    compile_options: &BTreeMap<String, String>,
    custom_hook: Option<&str>,
    unknown_incompatible_features: &[String],
) -> Result<PreparedDispatchMeta, DispatchError> {
    let composition_proof =
        verify_transform_composition_parts(root_jaxpr, transform_stack, transform_evidence)?;
    let cache_key = build_cache_key_ref(&CacheKeyInputRef {
        mode,
        backend,
        jaxpr: root_jaxpr,
        transform_stack,
        compile_options,
        custom_hook,
        unknown_incompatible_features,
    })?;
    Ok(PreparedDispatchMeta {
        composition_proof,
        cache_key,
        static_inputs: PreparedDispatchStaticInputs::from_parts(
            mode,
            root_jaxpr,
            transform_stack,
            transform_evidence,
            backend,
            compile_options,
            custom_hook,
            unknown_incompatible_features,
        ),
    })
}

/// Parse in_axes from compile_options.
/// Format: comma-separated list of axis specs, e.g. "0,none,1" or just "0" for uniform.
fn parse_axis_spec(raw: &str, option_name: &'static str) -> Result<AxisSpec, DispatchError> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("none") {
        return Ok(AxisSpec::NotBatched);
    }

    trimmed.parse::<i32>().map(AxisSpec::Batched).map_err(|_| {
        TransformExecutionError::InvalidVmapAxisSpec {
            option: option_name,
            value: trimmed.to_owned(),
        }
        .into()
    })
}

fn parse_vmap_in_axes(
    opts: &BTreeMap<String, String>,
    num_args: usize,
) -> Result<Vec<AxisSpec>, DispatchError> {
    match opts.get("vmap_in_axes") {
        None => Ok(vec![AxisSpec::Batched(0); num_args]),
        Some(s) if s.trim().is_empty() => Ok(vec![AxisSpec::Batched(0); num_args]),
        Some(s) => {
            let specs = s
                .split(',')
                .map(|part| parse_axis_spec(part, "vmap_in_axes"))
                .collect::<Result<Vec<_>, _>>()?;
            // If only one spec provided, broadcast to all args
            if specs.len() == 1 {
                Ok(vec![specs[0]; num_args])
            } else {
                Ok(specs)
            }
        }
    }
}

/// Parse out_axes from compile_options.
fn parse_vmap_out_axes(
    opts: &BTreeMap<String, String>,
    num_outputs: usize,
) -> Result<Vec<AxisSpec>, DispatchError> {
    match opts.get("vmap_out_axes") {
        None => Ok(vec![AxisSpec::Batched(0); num_outputs]),
        Some(s) if s.trim().is_empty() => Ok(vec![AxisSpec::Batched(0); num_outputs]),
        Some(s) => {
            let specs = s
                .split(',')
                .map(|part| parse_axis_spec(part, "vmap_out_axes"))
                .collect::<Result<Vec<_>, _>>()?;
            if specs.len() == 1 {
                Ok(vec![specs[0]; num_outputs])
            } else {
                Ok(specs)
            }
        }
    }
}

fn wants_value_and_grad(opts: &BTreeMap<String, String>) -> bool {
    opts.get("value_and_grad").is_some_and(|raw| {
        raw.eq_ignore_ascii_case("true")
            || raw.eq_ignore_ascii_case("1")
            || raw.eq_ignore_ascii_case("yes")
    })
}

fn custom_vjp_rule_key(opts: &BTreeMap<String, String>) -> Option<&str> {
    opts.get("custom_vjp_rule_key")
        .map(String::as_str)
        .filter(|key| !key.is_empty())
}

fn wants_egraph_optimize(opts: &BTreeMap<String, String>) -> bool {
    opts.get("egraph_optimize").is_some_and(|raw| {
        raw.eq_ignore_ascii_case("true")
            || raw.eq_ignore_ascii_case("1")
            || raw.eq_ignore_ascii_case("yes")
    })
}

fn allows_finite_diff_grad_fallback(opts: &BTreeMap<String, String>) -> bool {
    opts.get("allow_finite_diff_grad_fallback")
        .is_none_or(|raw| {
            !(raw.eq_ignore_ascii_case("false")
                || raw.eq_ignore_ascii_case("0")
                || raw.eq_ignore_ascii_case("no")
                || raw.eq_ignore_ascii_case("deny"))
        })
}

fn parse_grad_argnums(
    opts: &BTreeMap<String, String>,
    num_args: usize,
) -> Result<(Vec<usize>, bool), DispatchError> {
    match opts.get("grad_argnums") {
        None => Ok((vec![0], false)),
        Some(raw) if raw.trim().is_empty() => Ok((vec![0], false)),
        Some(raw) => {
            let mut argnums = Vec::new();
            for part in raw.split(',') {
                let idx = part.trim().parse::<usize>().map_err(|_| {
                    TransformExecutionError::TensorBuild(format!(
                        "invalid grad_argnums entry {:?} in {raw:?}",
                        part.trim()
                    ))
                })?;
                if idx >= num_args {
                    return Err(TransformExecutionError::TensorBuild(format!(
                        "grad_argnums index {idx} out of bounds for {num_args} arguments"
                    ))
                    .into());
                }
                argnums.push(idx);
            }
            if argnums.is_empty() {
                return Err(TransformExecutionError::TensorBuild(
                    "grad_argnums must specify at least one argument index".to_owned(),
                )
                .into());
            }
            Ok((argnums, true))
        }
    }
}

fn parse_static_argnums(
    opts: &BTreeMap<String, String>,
    num_invars: usize,
) -> Result<Vec<usize>, DispatchError> {
    match opts.get("static_argnums") {
        None => {
            let no_static_args = Vec::new();
            Ok(no_static_args)
        }
        Some(raw) if raw.trim().is_empty() => {
            let no_static_args = Vec::new();
            Ok(no_static_args)
        }
        Some(raw) => {
            let mut argnums = Vec::new();
            for part in raw.split(',') {
                let idx = part.trim().parse::<usize>().map_err(|_| {
                    TransformExecutionError::TensorBuild(format!(
                        "invalid static_argnums entry {:?} in {raw:?}",
                        part.trim()
                    ))
                })?;
                if idx >= num_invars {
                    return Err(TransformExecutionError::TensorBuild(format!(
                        "static_argnums index {idx} out of bounds for {num_invars} inputs"
                    ))
                    .into());
                }
                if !argnums.contains(&idx) {
                    argnums.push(idx);
                }
            }
            argnums.sort_unstable();
            Ok(argnums)
        }
    }
}

fn format_static_arg_value(val: &Value) -> String {
    match val {
        Value::Scalar(lit) => format!("S:{lit:?}"),
        Value::Tensor(tensor) => format!(
            "T:{:?}:{:?}:{:?}",
            tensor.dtype, tensor.shape.dims, tensor.elements
        ),
    }
}

fn compile_options_with_static_args(
    opts: &BTreeMap<String, String>,
    static_argnums: &[usize],
    args: &[Value],
) -> BTreeMap<String, String> {
    let mut out = opts.clone();
    let encoded = static_argnums
        .iter()
        .map(|&idx| {
            let val_repr = args
                .get(idx)
                .map_or_else(|| "missing".to_owned(), format_static_arg_value);
            format!("{idx}={val_repr}")
        })
        .collect::<Vec<_>>()
        .join("|");
    out.insert("__static_arg_values".to_owned(), encoded);
    out
}

fn compile_options_for_transform_tail(opts: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    opts.iter()
        .filter(|(key, _)| key.as_str() != "vmap_in_axes" && key.as_str() != "vmap_out_axes")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn transform_tail_summary(tail: &[Transform]) -> String {
    tail.iter()
        .map(|transform| transform.as_str())
        .collect::<Vec<_>>()
        .join(">")
}

/// Telemetry from the `fj-cache` compilation cache, `fj-interpreters::staging`,
/// and `fj-runtime` / `fj-ledger` pre-execution governance attached to a [`DispatchResponse`].
///
/// Excluded from [`PartialEq`] comparisons so memoized warm calls remain
/// logically equal to cold calls while still exposing hit/miss, staging, and
/// governance metadata to callers and tests.
#[derive(Debug, Clone)]
pub struct DispatchCacheTelemetry {
    pub cache_hit: bool,
    pub cache_stats: fj_cache::backend::CacheStats,
    pub staged_program: Option<std::sync::Arc<fj_interpreters::staging::StagedProgram>>,
    pub execution_regime: ExecutionRegime,
    pub estimated_working_set_bytes: usize,
    pub calibrated_posterior: CalibratedEstimate,
    pub e_value: f64,
    pub ece: f64,
}

impl Default for DispatchCacheTelemetry {
    fn default() -> Self {
        Self {
            cache_hit: false,
            cache_stats: fj_cache::backend::CacheStats::default(),
            staged_program: None,
            execution_regime: ExecutionRegime::CompiledSlotFastPath,
            estimated_working_set_bytes: 0,
            calibrated_posterior: CalibratedEstimate {
                point: 0.05,
                lower: 0.025,
                upper: 0.075,
                coverage: 0.0,
                used_conformal: false,
            },
            e_value: 1.0,
            ece: 0.0,
        }
    }
}

impl PartialEq for DispatchCacheTelemetry {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for DispatchCacheTelemetry {}

fn wants_wave_parallel(opts: &BTreeMap<String, String>) -> Option<bool> {
    opts.get("wave_parallel").and_then(|raw| {
        if raw.eq_ignore_ascii_case("true")
            || raw.eq_ignore_ascii_case("1")
            || raw.eq_ignore_ascii_case("yes")
        {
            Some(true)
        } else if raw.eq_ignore_ascii_case("false")
            || raw.eq_ignore_ascii_case("0")
            || raw.eq_ignore_ascii_case("no")
        {
            Some(false)
        } else {
            None
        }
    })
}

fn allows_fast_canonicalize(opts: &BTreeMap<String, String>) -> bool {
    opts.get("egraph_optimize").is_none_or(|raw| {
        !(raw.eq_ignore_ascii_case("false")
            || raw.eq_ignore_ascii_case("0")
            || raw.eq_ignore_ascii_case("no"))
    })
}

fn is_heavy_parallel_primitive(p: fj_core::Primitive) -> bool {
    use fj_core::Primitive;
    matches!(
        p,
        Primitive::Dot
            | Primitive::Conv
            | Primitive::Cholesky
            | Primitive::Qr
            | Primitive::Svd
            | Primitive::Eigh
            | Primitive::TriangularSolve
            | Primitive::Fft
            | Primitive::Ifft
            | Primitive::Rfft
            | Primitive::Irfft
    )
}

/// Return `true` if `jaxpr` contains at least one dependency wave with 2 or more
/// independent heavy operations (`Dot`, `Conv`, linalg decompositions, or FFTs)
/// or 2 or more independent equations with explicit tensor shapes >= 4096 elements.
#[must_use]
pub fn jaxpr_has_parallel_heavy_wave(jaxpr: &Jaxpr) -> bool {
    if jaxpr.equations.len() < 2 {
        return false;
    }
    let waves = fj_backend_cpu::build_dependency_waves(jaxpr);
    waves.iter().any(|wave| {
        if wave.len() < 2 {
            return false;
        }
        let heavy_count = wave
            .iter()
            .filter(|&&idx| {
                jaxpr
                    .equations
                    .get(idx)
                    .is_some_and(|eqn| is_heavy_parallel_primitive(eqn.primitive))
            })
            .count();
        heavy_count >= 2
    })
}

#[derive(Clone)]
pub struct CachedDispatchPlan {
    pub cache_key: CacheKey,
    pub effective_jaxpr: std::sync::Arc<Jaxpr>,
    pub compiled_jaxpr: Option<std::sync::Arc<fj_interpreters::CompiledJaxpr>>,
    pub staged_program: Option<std::sync::Arc<fj_interpreters::staging::StagedProgram>>,
    pub compiled_unknown_jaxpr: Option<std::sync::Arc<fj_interpreters::CompiledJaxpr>>,
    pub dependency_waves: Vec<Vec<usize>>,
    pub use_wave_parallel: bool,
}

impl std::fmt::Debug for CachedDispatchPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedDispatchPlan")
            .field("cache_key", &self.cache_key)
            .field("effective_jaxpr", &self.effective_jaxpr)
            .field("has_compiled_jaxpr", &self.compiled_jaxpr.is_some())
            .field("staged_program", &self.staged_program)
            .field(
                "has_compiled_unknown_jaxpr",
                &self.compiled_unknown_jaxpr.is_some(),
            )
            .field("dependency_waves", &self.dependency_waves)
            .field("use_wave_parallel", &self.use_wave_parallel)
            .finish()
    }
}

struct DispatchCompilationCache {
    lru_store: fj_cache::eviction::LruCache<fj_cache::backend::InMemoryCache>,
    plans: std::collections::HashMap<CacheKey, std::sync::Arc<CachedDispatchPlan>>,
    hits: u64,
    misses: u64,
}

impl DispatchCompilationCache {
    fn new() -> Self {
        Self {
            lru_store: fj_cache::eviction::LruCache::new(
                fj_cache::backend::InMemoryCache::new(),
                fj_cache::eviction::LruConfig::default(),
            ),
            plans: std::collections::HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }
}

static DISPATCH_COMPILATION_CACHE: std::sync::OnceLock<std::sync::Mutex<DispatchCompilationCache>> =
    std::sync::OnceLock::new();

fn compilation_cache() -> &'static std::sync::Mutex<DispatchCompilationCache> {
    DISPATCH_COMPILATION_CACHE
        .get_or_init(|| std::sync::Mutex::new(DispatchCompilationCache::new()))
}

/// Return current `LruCache` statistics (`entry_count`, `total_bytes`) for the
/// process-wide dispatch compilation cache.
#[must_use]
pub fn dispatch_cache_stats() -> fj_cache::backend::CacheStats {
    use fj_cache::backend::CacheBackend;
    let guard = compilation_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.lru_store.stats()
}

/// Return cumulative `(hits, misses)` recorded by the dispatch compilation cache.
#[must_use]
pub fn dispatch_cache_hit_miss_counts() -> (u64, u64) {
    let guard = compilation_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    (guard.hits, guard.misses)
}

/// Clear all entries and reset hit/miss counters in the dispatch compilation cache.
pub fn clear_dispatch_cache() {
    use fj_cache::backend::CacheBackend;
    let mut guard = compilation_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.lru_store.clear();
    guard.plans.clear();
    guard.hits = 0;
    guard.misses = 0;
}

/// Register a pre-compiled `JitWrapped` plan in the dispatch `LruCache` on cold
/// compilation.
pub fn register_compiled_dispatch_plan(
    cache_key: CacheKey,
    root_jaxpr: &Jaxpr,
    compiled: Option<std::sync::Arc<fj_interpreters::CompiledJaxpr>>,
) {
    use fj_cache::backend::CacheBackend;
    let payload = root_jaxpr.canonical_fingerprint().as_bytes().to_vec();
    let artifact = fj_cache::CachedArtifact {
        integrity_sha256_hex: fj_cache::sha256_hex(&payload),
        data: payload,
    };
    let dependency_waves = fj_backend_cpu::build_dependency_waves(root_jaxpr);
    let use_wave_parallel = jaxpr_has_parallel_heavy_wave(root_jaxpr);
    let plan = std::sync::Arc::new(CachedDispatchPlan {
        cache_key: cache_key.clone(),
        effective_jaxpr: std::sync::Arc::new(root_jaxpr.clone()),
        compiled_jaxpr: compiled,
        staged_program: None,
        compiled_unknown_jaxpr: None,
        dependency_waves,
        use_wave_parallel,
    });
    let mut guard = compilation_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.lru_store.get(&cache_key).is_some() {
        guard.hits += 1;
    } else {
        guard.misses += 1;
        guard.lru_store.put(&cache_key, artifact);
        guard.plans.insert(cache_key, plan);
    }
}

/// Record a warm cache hit in the dispatch compilation cache.
pub fn record_dispatch_cache_hit(cache_key: &CacheKey) {
    use fj_cache::backend::CacheBackend;
    let mut guard = compilation_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.lru_store.get(cache_key).is_some() {
        guard.hits += 1;
    }
}

/// Snapshot of the process-wide pre-execution statistical governance state
/// (`ConformalPredictor`, `CalibrationReport`, `EProcess`, `MemoryBudget`).
#[derive(Debug, Clone)]
pub struct DispatchGovernanceSnapshot {
    pub calibration_size: usize,
    pub is_calibrated: bool,
    pub prediction_threshold: Option<f64>,
    pub ece: f64,
    pub e_value: f64,
    pub e_process_observations: usize,
    pub e_process_rejected: bool,
    pub memory_budget: MemoryBudget,
    pub total_admissions: u64,
    pub total_rejections: u64,
}

struct DispatchGovernanceState {
    conformal: ConformalPredictor,
    calibration: CalibrationReport,
    e_process: EProcess,
    memory_budget: MemoryBudget,
    cancellation_token: CancellationToken,
    total_admissions: u64,
    total_rejections: u64,
}

static GLOBAL_CANCELLATION_TOKEN: std::sync::OnceLock<CancellationToken> =
    std::sync::OnceLock::new();
static GOVERNANCE_NONDEFAULT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static FAST_PATH_ADMISSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn global_cancellation_token() -> &'static CancellationToken {
    GLOBAL_CANCELLATION_TOKEN.get_or_init(CancellationToken::new)
}

impl DispatchGovernanceState {
    fn new() -> Self {
        Self {
            conformal: ConformalPredictor::new(0.9, 5),
            calibration: CalibrationReport::new(10),
            e_process: EProcess::new(20.0),
            memory_budget: MemoryBudget::default(),
            cancellation_token: global_cancellation_token().clone(),
            total_admissions: 0,
            total_rejections: 0,
        }
    }

    fn snapshot(&self) -> DispatchGovernanceSnapshot {
        DispatchGovernanceSnapshot {
            calibration_size: self.conformal.calibration_size(),
            is_calibrated: self.conformal.is_calibrated(),
            prediction_threshold: self.conformal.prediction_threshold(),
            ece: self.calibration.compute_ece(),
            e_value: self.e_process.e_value(),
            e_process_observations: self.e_process.observations(),
            e_process_rejected: self.e_process.rejected(),
            memory_budget: self.memory_budget,
            total_admissions: self.total_admissions
                + FAST_PATH_ADMISSIONS.load(std::sync::atomic::Ordering::Relaxed),
            total_rejections: self.total_rejections,
        }
    }
}

static DISPATCH_GOVERNANCE_STATE: std::sync::OnceLock<std::sync::Mutex<DispatchGovernanceState>> =
    std::sync::OnceLock::new();

fn governance_state() -> &'static std::sync::Mutex<DispatchGovernanceState> {
    DISPATCH_GOVERNANCE_STATE.get_or_init(|| std::sync::Mutex::new(DispatchGovernanceState::new()))
}

/// Return a snapshot of the process-wide `fj-ledger` + `fj-runtime` governance state.
#[must_use]
pub fn dispatch_governance_stats() -> DispatchGovernanceSnapshot {
    let guard = governance_state().lock().unwrap_or_else(|e| e.into_inner());
    guard.snapshot()
}

/// Reset the process-wide `ConformalPredictor`, `CalibrationReport`, `EProcess`,
/// `MemoryBudget`, and `CancellationToken` to their clean initial state.
pub fn reset_dispatch_governance() {
    global_cancellation_token().reset();
    GOVERNANCE_NONDEFAULT.store(false, std::sync::atomic::Ordering::SeqCst);
    FAST_PATH_ADMISSIONS.store(0, std::sync::atomic::Ordering::SeqCst);
    let mut guard = governance_state().lock().unwrap_or_else(|e| e.into_inner());
    *guard = DispatchGovernanceState::new();
}

/// Override the process-wide [`MemoryBudget`] enforced before dispatch execution.
pub fn set_dispatch_memory_budget(budget: MemoryBudget) {
    let mut guard = governance_state().lock().unwrap_or_else(|e| e.into_inner());
    guard.memory_budget = budget;
    GOVERNANCE_NONDEFAULT.store(
        budget != MemoryBudget::default() || guard.e_process.rejected(),
        std::sync::atomic::Ordering::SeqCst,
    );
}

/// Feed a likelihood ratio observation into the process-wide sequential drift [`EProcess`].
pub fn observe_dispatch_drift_likelihood_ratio(likelihood_ratio: f64) {
    let mut guard = governance_state().lock().unwrap_or_else(|e| e.into_inner());
    guard.e_process.update(likelihood_ratio);
    if guard.e_process.rejected() {
        GOVERNANCE_NONDEFAULT.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Return a handle to the process-wide [`CancellationToken`] checked before dispatch execution.
#[must_use]
pub fn dispatch_cancellation_token() -> CancellationToken {
    global_cancellation_token().clone()
}

/// Evaluate pre-execution admission for fast-path callers (`JitWrapped` / `GradWrapped` /
/// `ValueAndGradWrapped`) so that `MemoryBudget`, `CancellationToken`, and `EProcess`
/// sequential rejection are enforced before any fast-path evaluation runs.
pub fn check_fast_path_admission(
    jaxpr: &Jaxpr,
    transform_stack: &[Transform],
    args: &[Value],
    mode: CompatibilityMode,
) -> Result<(), DispatchError> {
    if mode == CompatibilityMode::Strict
        && !GOVERNANCE_NONDEFAULT.load(std::sync::atomic::Ordering::Relaxed)
        && !global_cancellation_token().is_cancelled()
        && jaxpr.equations.len() <= 256
        && transform_stack.len() <= 16
        && args.iter().all(|v| match v {
            Value::Scalar(_) => true,
            Value::Tensor(t) => t.elements.len() <= 4096,
        })
    {
        FAST_PATH_ADMISSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(());
    }

    let posterior_abandoned = heuristic_posterior_abandoned_parts(jaxpr, transform_stack);
    let has_parallel_wave = jaxpr_has_parallel_heavy_wave(jaxpr);
    let model = RuntimeAdmissionModel::new(mode);
    let mut guard = governance_state().lock().unwrap_or_else(|e| e.into_inner());
    match model.evaluate_admission(
        jaxpr,
        args,
        transform_stack.len(),
        posterior_abandoned,
        &guard.memory_budget,
        Some(&guard.cancellation_token),
        &guard.conformal,
        &guard.e_process,
        has_parallel_wave,
        false,
        false,
    ) {
        Ok(_decision) => {
            guard.total_admissions += 1;
            Ok(())
        }
        Err(rejection) => {
            guard.total_rejections += 1;
            Err(DispatchError::AdmissionRejected(rejection))
        }
    }
}

#[derive(Debug, Clone)]
pub struct DispatchResponse {
    pub outputs: Vec<Value>,
    pub cache_key: String,
    pub evidence_ledger: EvidenceLedger,
    pub composition_proof: TransformCompositionProof,
    pub telemetry: DispatchCacheTelemetry,
}

impl PartialEq for DispatchResponse {
    fn eq(&self, other: &Self) -> bool {
        self.outputs == other.outputs
            && self.cache_key == other.cache_key
            && self.composition_proof == other.composition_proof
            && self.evidence_ledger.entries().len() == other.evidence_ledger.entries().len()
            && self
                .evidence_ledger
                .entries()
                .iter()
                .zip(other.evidence_ledger.entries().iter())
                .all(|(a, b)| {
                    a.decision_id == b.decision_id
                        && a.signals == b.signals
                        && a.record.mode == b.record.mode
                        && a.record.posterior_abandoned == b.record.posterior_abandoned
                        && a.record.expected_loss_keep == b.record.expected_loss_keep
                        && a.record.expected_loss_kill == b.record.expected_loss_kill
                        && a.record.action == b.record.action
                })
    }
}

impl DispatchResponse {
    #[must_use]
    pub fn cache_hit(&self) -> bool {
        self.telemetry.cache_hit
    }

    #[must_use]
    pub fn cache_stats(&self) -> &fj_cache::backend::CacheStats {
        &self.telemetry.cache_stats
    }

    #[must_use]
    pub fn staged_program(&self) -> Option<&fj_interpreters::staging::StagedProgram> {
        self.telemetry.staged_program.as_deref()
    }

    #[must_use]
    pub fn execution_regime(&self) -> ExecutionRegime {
        self.telemetry.execution_regime
    }

    #[must_use]
    pub fn estimated_working_set_bytes(&self) -> usize {
        self.telemetry.estimated_working_set_bytes
    }

    #[must_use]
    pub fn calibrated_posterior(&self) -> &CalibratedEstimate {
        &self.telemetry.calibrated_posterior
    }

    #[must_use]
    pub fn e_value(&self) -> f64 {
        self.telemetry.e_value
    }

    #[must_use]
    pub fn calibration_ece(&self) -> f64 {
        self.telemetry.ece
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformExecutionError {
    EmptyArgumentList { transform: Transform },
    NonScalarGradientInput,
    NonScalarGradientOutput,
    VmapRequiresRankOneLeadingArgument,
    VmapMismatchedLeadingDimension { expected: usize, actual: usize },
    VmapInconsistentOutputArity { expected: usize, actual: usize },
    VmapAxesOutOfBounds { axis: i32, ndim: usize },
    VmapAxesCountMismatch { expected: usize, actual: usize },
    InvalidVmapAxisSpec { option: &'static str, value: String },
    VmapUnmappedOutputMismatch,
    FiniteDiffGradFallbackDisabled { tail: String },
    EmptyVmapOutput,
    TensorBuild(String),
    PmapUnavailable,
}

impl std::fmt::Display for TransformExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyArgumentList { transform } => {
                write!(f, "{} requires at least one argument", transform.as_str())
            }
            Self::NonScalarGradientInput => {
                write!(f, "grad currently requires scalar first input")
            }
            Self::NonScalarGradientOutput => {
                write!(f, "grad currently requires scalar first output")
            }
            Self::VmapRequiresRankOneLeadingArgument => {
                write!(f, "vmap currently requires first argument with rank >= 1")
            }
            Self::VmapMismatchedLeadingDimension { expected, actual } => {
                write!(
                    f,
                    "vmap leading-dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::VmapInconsistentOutputArity { expected, actual } => {
                write!(
                    f,
                    "vmap inner output arity mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::EmptyVmapOutput => {
                write!(f, "vmap received no mapped elements")
            }
            Self::VmapAxesOutOfBounds { axis, ndim } => {
                write!(
                    f,
                    "vmap axis {axis} is out of bounds for tensor with {ndim} dimensions"
                )
            }
            Self::VmapAxesCountMismatch { expected, actual } => {
                write!(
                    f,
                    "vmap in_axes/out_axes length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidVmapAxisSpec { option, value } => {
                write!(
                    f,
                    "invalid {option} axis spec {value:?}; expected an integer or \"none\""
                )
            }
            Self::VmapUnmappedOutputMismatch => {
                write!(
                    f,
                    "vmap out_axes=none requires the output to be identical across mapped elements"
                )
            }
            Self::FiniteDiffGradFallbackDisabled { tail } => {
                write!(
                    f,
                    "finite-difference grad fallback is disabled for remaining transform tail [{tail}]"
                )
            }
            Self::TensorBuild(detail) => write!(f, "tensor build error: {detail}"),
            Self::PmapUnavailable => write!(
                f,
                "pmap unsupported: multi-device backend infrastructure (GPU/TPU) is unavailable in V1"
            ),
        }
    }
}

impl std::error::Error for TransformExecutionError {}

#[derive(Debug)]
pub enum DispatchError {
    Cache(CacheKeyError),
    Interpreter(InterpreterError),
    BackendExecution(BackendError),
    TransformInvariant(TransformCompositionError),
    TransformExecution(TransformExecutionError),
    AdmissionRejected(AdmissionRejection),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cache(err) => write!(f, "cache key error: {err}"),
            Self::Interpreter(err) => write!(f, "interpreter error: {err}"),
            Self::BackendExecution(err) => write!(f, "backend execution error: {err}"),
            Self::TransformInvariant(err) => write!(f, "transform invariant error: {err}"),
            Self::TransformExecution(err) => write!(f, "transform execution error: {err}"),
            Self::AdmissionRejected(err) => write!(f, "admission rejected: {err}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<CacheKeyError> for DispatchError {
    fn from(value: CacheKeyError) -> Self {
        Self::Cache(value)
    }
}

impl From<InterpreterError> for DispatchError {
    fn from(value: InterpreterError) -> Self {
        Self::Interpreter(value)
    }
}

impl From<BackendError> for DispatchError {
    fn from(value: BackendError) -> Self {
        Self::BackendExecution(value)
    }
}

impl From<TransformCompositionError> for DispatchError {
    fn from(value: TransformCompositionError) -> Self {
        Self::TransformInvariant(value)
    }
}

impl From<TransformExecutionError> for DispatchError {
    fn from(value: TransformExecutionError) -> Self {
        Self::TransformExecution(value)
    }
}

impl From<AdmissionRejection> for DispatchError {
    fn from(value: AdmissionRejection) -> Self {
        Self::AdmissionRejected(value)
    }
}

pub fn dispatch(request: DispatchRequest) -> Result<DispatchResponse, DispatchError> {
    let DispatchRequest {
        mode,
        ledger,
        args,
        backend,
        compile_options,
        custom_hook,
        unknown_incompatible_features,
    } = request;
    let custom_ledger_proof = if !ledger.steps.is_empty()
        || !ledger.root_input_avals.is_empty()
        || !ledger.root_output_avals.is_empty()
    {
        Some(verify_transform_composition(&ledger)?)
    } else {
        None
    };
    let mut response = dispatch_ref(DispatchRequestRef {
        mode,
        root_jaxpr: &ledger.root_jaxpr,
        transform_stack: &ledger.transform_stack,
        transform_evidence: &ledger.transform_evidence,
        args,
        backend: &backend,
        compile_options,
        custom_hook: custom_hook.as_deref(),
        unknown_incompatible_features: &unknown_incompatible_features,
        prepared: None,
    })?;
    if !ledger.steps.is_empty()
        && let Some(proof) = custom_ledger_proof
    {
        response.composition_proof = proof;
    }
    Ok(response)
}

/// Execute `dispatch` with an explicit `FfiRegistry` installed for `Primitive::CustomCall` equations.
pub fn dispatch_with_ffi(
    mut request: DispatchRequest,
    registry: &fj_ffi::FfiRegistry,
) -> Result<DispatchResponse, DispatchError> {
    let mut targets = registry.registered_names();
    targets.sort();
    request
        .compile_options
        .insert("ffi_registered_targets".to_owned(), targets.join(","));
    fj_interpreters::with_ffi_registry(registry, || dispatch(request))
}

pub fn dispatch_ref(request: DispatchRequestRef<'_>) -> Result<DispatchResponse, DispatchError> {
    use fj_cache::backend::CacheBackend;
    use std::borrow::Cow;

    let static_argnums =
        parse_static_argnums(&request.compile_options, request.root_jaxpr.invars.len())?;
    let effective_key_opts: Cow<'_, BTreeMap<String, String>> = if static_argnums.is_empty() {
        Cow::Borrowed(&request.compile_options)
    } else {
        Cow::Owned(compile_options_with_static_args(
            &request.compile_options,
            &static_argnums,
            &request.args,
        ))
    };

    let (composition_proof, cache_key): (Cow<'_, _>, Cow<'_, _>) =
        match (request.prepared, static_argnums.is_empty()) {
            (Some(meta), true) => {
                if request.mode == CompatibilityMode::Strict
                    && !request.unknown_incompatible_features.is_empty()
                {
                    return Err(CacheKeyError::UnknownIncompatibleFeatures {
                        features: request.unknown_incompatible_features.to_vec(),
                    }
                    .into());
                }
                if !meta.static_inputs.matches_request(&request) {
                    return Err(DispatchError::TransformExecution(
                        TransformExecutionError::TensorBuild(
                            "prepared dispatch metadata does not match request static inputs"
                                .to_owned(),
                        ),
                    ));
                }
                (
                    Cow::Borrowed(&meta.composition_proof),
                    Cow::Borrowed(&meta.cache_key),
                )
            }
            _ => {
                let composition_proof = verify_transform_composition_parts(
                    request.root_jaxpr,
                    request.transform_stack,
                    request.transform_evidence,
                )?;
                let cache_key = build_cache_key_ref(&CacheKeyInputRef {
                    mode: request.mode,
                    backend: request.backend,
                    jaxpr: request.root_jaxpr,
                    transform_stack: request.transform_stack,
                    compile_options: &effective_key_opts,
                    custom_hook: request.custom_hook,
                    unknown_incompatible_features: request.unknown_incompatible_features,
                })?;
                (Cow::Owned(composition_proof), Cow::Owned(cache_key))
            }
        };
    let nested_trace_summary =
        simulate_nested_trace_contexts(request.transform_stack, &request.args).map_err(|err| {
            TransformExecutionError::TensorBuild(format!("nested trace simulation failed: {err}"))
        })?;

    // Thread effect context through Jaxpr/equation effect ordering.
    let mut effect_ctx = EffectContext::new();
    thread_jaxpr_effect_tokens(&mut effect_ctx, request.root_jaxpr);
    let effect_tokens = effect_ctx.finalize();
    let effect_count = effect_tokens.len();
    let effect_trace = effect_tokens
        .iter()
        .map(|token| format!("{}:{}", token.sequence_number, token.effect_name))
        .collect::<Vec<_>>()
        .join(",");
    let nested_trace_frame_trace = nested_trace_summary
        .frames
        .iter()
        .map(|frame| {
            format!(
                "{}:{}@{}",
                frame.transform.as_str(),
                frame.trace_id,
                frame.depth
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    // ── Pre-Execution Governance (`fj-ledger` + `fj-runtime`) ──────
    let signals = vec![
        EvidenceSignal {
            signal_name: "eqn_count".to_owned(),
            log_likelihood_delta: (request.root_jaxpr.equations.len() as f64 + 1.0).ln(),
            detail: format!("eqn_count={}", request.root_jaxpr.equations.len()),
        },
        EvidenceSignal {
            signal_name: "transform_depth".to_owned(),
            log_likelihood_delta: request.transform_stack.len() as f64 * 0.1,
            detail: format!("transform_depth={}", request.transform_stack.len()),
        },
        EvidenceSignal {
            signal_name: "transform_stack_hash".to_owned(),
            log_likelihood_delta: (composition_proof.transform_count as f64 + 1.0).ln(),
            detail: composition_proof.stack_hash_hex.clone(),
        },
        EvidenceSignal {
            signal_name: "nested_trace_depth".to_owned(),
            log_likelihood_delta: nested_trace_summary.max_depth as f64 * 0.05,
            detail: format!(
                "max_depth={},frames={}",
                nested_trace_summary.max_depth, nested_trace_frame_trace
            ),
        },
        EvidenceSignal {
            signal_name: "effect_token_count".to_owned(),
            log_likelihood_delta: (effect_count as f64 + 1.0).ln(),
            detail: format!("effect_tokens={effect_count}"),
        },
        EvidenceSignal {
            signal_name: "effect_token_trace".to_owned(),
            log_likelihood_delta: (effect_count as f64 + 1.0).ln() * 0.1,
            detail: format!("effect_tokens=[{effect_trace}]"),
        },
    ];

    let posterior_abandoned = compute_posterior_abandoned_with_signals(
        request.root_jaxpr,
        request.transform_stack,
        &request.compile_options,
        &signals,
    );
    let has_parallel_heavy_wave = wants_wave_parallel(&request.compile_options)
        .unwrap_or_else(|| jaxpr_has_parallel_heavy_wave(request.root_jaxpr));
    let wants_egraph = wants_egraph_optimize(&request.compile_options);
    let reject_on_kill = request
        .compile_options
        .get("reject_on_kill")
        .is_some_and(|v| v.eq_ignore_ascii_case("true") || v == "1");

    let admission_decision = {
        let mut gov = governance_state().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(raw_lr) = request.compile_options.get("drift_likelihood_ratio")
            && let Ok(lr) = raw_lr.trim().parse::<f64>()
        {
            gov.e_process.update(lr);
        }
        let mut effective_budget = gov.memory_budget;
        if let Some(raw_bytes) = request.compile_options.get("max_working_set_bytes")
            && let Ok(bytes) = raw_bytes.trim().parse::<usize>()
        {
            effective_budget.max_working_set_bytes = bytes;
        }
        if let Some(raw_eqns) = request.compile_options.get("max_equation_count")
            && let Ok(eqns) = raw_eqns.trim().parse::<usize>()
        {
            effective_budget.max_equation_count = eqns;
        }
        if let Some(raw_depth) = request.compile_options.get("max_transform_depth")
            && let Ok(depth) = raw_depth.trim().parse::<usize>()
        {
            effective_budget.max_transform_depth = depth;
        }

        let admission_model = RuntimeAdmissionModel::new(request.mode);
        match admission_model.evaluate_admission(
            request.root_jaxpr,
            &request.args,
            request.transform_stack.len(),
            posterior_abandoned,
            &effective_budget,
            Some(&gov.cancellation_token),
            &gov.conformal,
            &gov.e_process,
            has_parallel_heavy_wave,
            wants_egraph,
            reject_on_kill,
        ) {
            Ok(decision) => {
                gov.total_admissions += 1;
                decision
            }
            Err(rejection) => {
                gov.total_rejections += 1;
                return Err(DispatchError::AdmissionRejected(rejection));
            }
        }
    };

    let mut evidence_ledger = EvidenceLedger::new();
    evidence_ledger.append(LedgerEntry {
        decision_id: cache_key.as_string(),
        record: admission_decision.record.clone(),
        signals,
    });

    // Look up or compile the execution plan in the LruCache-backed compilation cache.
    let cached_hit = {
        let mut guard = compilation_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if guard.lru_store.get(&cache_key).is_some()
            && let Some(plan) = guard.plans.get(&cache_key).cloned()
        {
            guard.hits += 1;
            let stats = guard.lru_store.stats();
            Some((plan, stats))
        } else {
            None
        }
    };

    let (plan, cache_hit, cache_stats) = if let Some((plan, stats)) = cached_hit {
        (plan, true, stats)
    } else {
        let effective_jaxpr = match request
            .compile_options
            .get("egraph_optimize")
            .map(|s| s.trim())
        {
            Some(opt) if opt.eq_ignore_ascii_case("finite_fast_math") => {
                std::sync::Arc::new(fj_egraph::optimize_jaxpr_with_config(
                    request.root_jaxpr,
                    &fj_egraph::OptimizationConfig::finite_fast_math(),
                ))
            }
            Some(opt) if opt.eq_ignore_ascii_case("ieee_strict") => {
                std::sync::Arc::new(fj_egraph::optimize_jaxpr_with_config(
                    request.root_jaxpr,
                    &fj_egraph::OptimizationConfig::ieee_strict(),
                ))
            }
            Some(_) if wants_egraph => std::sync::Arc::new(fj_egraph::optimize_jaxpr_with_config(
                request.root_jaxpr,
                &fj_egraph::OptimizationConfig::safe(),
            )),
            _ if admission_decision.regime == ExecutionRegime::EGraphSaturatedCompile
                && allows_fast_canonicalize(&request.compile_options) =>
            {
                std::sync::Arc::new(fj_egraph::optimize_jaxpr_with_config(
                    request.root_jaxpr,
                    &fj_egraph::OptimizationConfig::ieee_strict(),
                ))
            }
            _ if request.transform_stack.contains(&Transform::Jit)
                && allows_fast_canonicalize(&request.compile_options) =>
            {
                std::sync::Arc::new(fj_egraph::canonicalize_jaxpr_fast(
                    request.root_jaxpr,
                    &fj_egraph::OptimizationConfig::ieee_strict(),
                ))
            }
            _ => std::sync::Arc::new(request.root_jaxpr.clone()),
        };

        let (staged_program, compiled_unknown_jaxpr) =
            if !static_argnums.is_empty() && request.args.len() == effective_jaxpr.invars.len() {
                let unknowns: Vec<bool> = (0..effective_jaxpr.invars.len())
                    .map(|idx| !static_argnums.contains(&idx))
                    .collect();
                let known_values: Vec<Value> = static_argnums
                    .iter()
                    .map(|&idx| request.args[idx].clone())
                    .collect();
                let staged = fj_interpreters::staging::stage_jaxpr(
                    &effective_jaxpr,
                    &unknowns,
                    &known_values,
                )
                .map_err(|e| TransformExecutionError::TensorBuild(format!("staging error: {e}")))?;
                let compiled_unk =
                    fj_interpreters::compile_jaxpr_for_repeated_eval(&staged.jaxpr_unknown)
                        .map(std::sync::Arc::new);
                (Some(std::sync::Arc::new(staged)), compiled_unk)
            } else {
                (None, None)
            };

        let active_jaxpr_for_waves: &Jaxpr = match &staged_program {
            Some(staged) => &staged.jaxpr_unknown,
            None => &effective_jaxpr,
        };
        let dependency_waves = fj_backend_cpu::build_dependency_waves(active_jaxpr_for_waves);
        let use_wave_parallel = wants_wave_parallel(&request.compile_options)
            .unwrap_or_else(|| jaxpr_has_parallel_heavy_wave(active_jaxpr_for_waves));

        let compiled_jaxpr = if staged_program.is_none()
            && request.transform_stack.iter().all(|t| *t == Transform::Jit)
        {
            fj_interpreters::compile_jaxpr_for_repeated_eval(&effective_jaxpr)
                .map(std::sync::Arc::new)
        } else {
            None
        };

        let payload_str = match &staged_program {
            Some(staged) => format!(
                "{}|{}",
                effective_jaxpr.canonical_fingerprint(),
                staged.jaxpr_unknown.canonical_fingerprint()
            ),
            None => effective_jaxpr.canonical_fingerprint().to_owned(),
        };
        let payload_bytes = payload_str.into_bytes();
        let artifact = fj_cache::CachedArtifact {
            integrity_sha256_hex: fj_cache::sha256_hex(&payload_bytes),
            data: payload_bytes,
        };
        if let Some(cache_dir) = request.compile_options.get("cache_dir") {
            let mut file_cache =
                fj_cache::backend::FileCache::new(std::path::PathBuf::from(cache_dir));
            file_cache.put(&cache_key, artifact.clone());
        }

        let new_plan = std::sync::Arc::new(CachedDispatchPlan {
            cache_key: cache_key.as_ref().clone(),
            effective_jaxpr,
            compiled_jaxpr,
            staged_program,
            compiled_unknown_jaxpr,
            dependency_waves,
            use_wave_parallel,
        });

        let stats = {
            let mut guard = compilation_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            guard.misses += 1;
            guard.lru_store.put(&cache_key, artifact);
            guard
                .plans
                .insert(cache_key.as_ref().clone(), new_plan.clone());
            guard.lru_store.stats()
        };
        (new_plan, false, stats)
    };

    let cpu_backend = fj_backend_cpu::CpuBackend::new();
    let backend_registry = BackendRegistry::new(vec![Box::new(fj_backend_cpu::CpuBackend::new())]);
    let requested_backend = (!request.backend.is_empty()).then_some(request.backend);
    let (backend, device, _fell_back) =
        backend_registry.resolve_with_fallback(&DevicePlacement::Default, requested_backend)?;

    let exec_start = std::time::Instant::now();
    let outputs = if let Some(staged) = &plan.staged_program {
        let dynamic_args: Vec<Value> = request
            .args
            .iter()
            .enumerate()
            .filter_map(|(idx, val)| (!static_argnums.contains(&idx)).then_some(val.clone()))
            .collect();
        if request.transform_stack.iter().all(|t| *t == Transform::Jit) {
            if plan.use_wave_parallel {
                cpu_backend.execute_with_consts(
                    &staged.jaxpr_unknown,
                    &staged.residuals,
                    &dynamic_args,
                    device,
                )?
            } else {
                fj_interpreters::staging::execute_staged(staged, &dynamic_args).map_err(|e| {
                    DispatchError::BackendExecution(BackendError::ExecutionFailed {
                        detail: e.to_string(),
                    })
                })?
            }
        } else {
            let mut full_unk_args = staged.residuals.clone();
            full_unk_args.extend(dynamic_args);
            execute_with_transforms(
                &staged.jaxpr_unknown,
                request.transform_stack,
                &full_unk_args,
                backend,
                device,
                &request.compile_options,
            )?
        }
    } else if request.transform_stack.iter().all(|t| *t == Transform::Jit)
        && !plan.use_wave_parallel
        && let Some(compiled) = &plan.compiled_jaxpr
    {
        compiled.eval(&request.args).map_err(|e| {
            DispatchError::BackendExecution(BackendError::ExecutionFailed {
                detail: e.to_string(),
            })
        })?
    } else {
        execute_with_transforms(
            &plan.effective_jaxpr,
            request.transform_stack,
            &request.args,
            backend,
            device,
            &request.compile_options,
        )?
    };
    let exec_elapsed_secs = exec_start.elapsed().as_secs_f64();

    // Post-execution calibration feedback into ConformalPredictor and CalibrationReport.
    let (post_e_value, post_ece) = {
        let mut gov = governance_state().lock().unwrap_or_else(|e| e.into_inner());
        let observed_latency_ratio = (exec_elapsed_secs / 0.05).clamp(0.0, 1.0);
        let nonconformity = (posterior_abandoned - observed_latency_ratio).abs();
        gov.conformal.observe(nonconformity);
        let observed_abandoned = exec_elapsed_secs >= 0.05;
        gov.calibration
            .observe(posterior_abandoned, observed_abandoned);
        if !request
            .compile_options
            .contains_key("drift_likelihood_ratio")
        {
            let benign_lr = if cache_hit { 0.95 } else { 0.99 };
            gov.e_process.update(benign_lr);
        }
        (gov.e_process.e_value(), gov.calibration.compute_ece())
    };

    let lowered_proof = build_lowered_composition_proof(&request, &plan, &static_argnums, &outputs)
        .unwrap_or_else(|| composition_proof.into_owned());

    Ok(DispatchResponse {
        outputs,
        cache_key: cache_key.as_string(),
        evidence_ledger,
        composition_proof: lowered_proof,
        telemetry: DispatchCacheTelemetry {
            cache_hit,
            cache_stats,
            staged_program: plan.staged_program.clone(),
            execution_regime: if plan.use_wave_parallel {
                ExecutionRegime::WaveParallelCpu
            } else {
                admission_decision.regime
            },
            estimated_working_set_bytes: admission_decision.estimated_working_set_bytes,
            calibrated_posterior: admission_decision.calibrated_posterior,
            e_value: post_e_value,
            ece: post_ece,
        },
    })
}

fn value_to_aval(val: &Value) -> AbstractValue {
    match val {
        Value::Scalar(lit) => {
            let dtype = match lit {
                Literal::I64(_) => DType::I64,
                Literal::I32(_) => DType::I32,
                Literal::U32(_) => DType::U32,
                Literal::U64(_) => DType::U64,
                Literal::Bool(_) => DType::Bool,
                Literal::BF16Bits(_) => DType::BF16,
                Literal::F16Bits(_) => DType::F16,
                Literal::F32Bits(_) => DType::F32,
                Literal::F64Bits(_) => DType::F64,
                Literal::Complex64Bits(..) => DType::Complex64,
                Literal::Complex128Bits(..) => DType::Complex128,
            };
            AbstractValue {
                dtype,
                shape: Shape::scalar(),
            }
        }
        Value::Tensor(t) => AbstractValue {
            dtype: t.dtype,
            shape: t.shape.clone(),
        },
    }
}

fn insert_aval_dim(aval: &AbstractValue, axis: usize, dim: u32) -> AbstractValue {
    let mut dims = aval.shape.dims.clone();
    let idx = axis.min(dims.len());
    dims.insert(idx, dim);
    AbstractValue {
        dtype: aval.dtype,
        shape: Shape { dims },
    }
}

fn build_lowered_composition_proof(
    request: &DispatchRequestRef<'_>,
    plan: &CachedDispatchPlan,
    static_argnums: &[usize],
    outputs: &[Value],
) -> Option<TransformCompositionProof> {
    use std::borrow::Cow;

    if wants_value_and_grad(&request.compile_options) {
        return None;
    }

    let n = request.transform_stack.len();
    let outer_in_avals: Vec<AbstractValue> = request.args.iter().map(value_to_aval).collect();
    let outer_out_avals: Vec<AbstractValue> = outputs.iter().map(value_to_aval).collect();

    if n == 0 {
        return verify_transform_composition_steps(
            request.root_jaxpr,
            request.transform_stack,
            request.transform_evidence,
            &outer_in_avals,
            &outer_out_avals,
            &[],
        )
        .ok();
    }

    // Peel outer Vmap/Pmap dimensions from outer_in_avals to recover root_input_avals.
    let mut vmap_meta: Vec<Option<(u32, Vec<Option<usize>>)>> = vec![None; n];
    let mut current_peeled_in = outer_in_avals;
    for (stack_idx, &transform) in request.transform_stack.iter().enumerate() {
        if transform == Transform::Vmap {
            let specs =
                parse_vmap_in_axes(&request.compile_options, current_peeled_in.len()).ok()?;
            if specs.len() != current_peeled_in.len() {
                return None;
            }
            let mut axis_size_opt: Option<u32> = None;
            let mut next_peeled = Vec::with_capacity(current_peeled_in.len());
            let mut resolved_axes = Vec::with_capacity(current_peeled_in.len());
            for (aval, spec) in current_peeled_in.iter().zip(specs.iter()) {
                match spec {
                    AxisSpec::NotBatched => {
                        resolved_axes.push(None);
                        next_peeled.push(aval.clone());
                    }
                    AxisSpec::Batched(_) => {
                        let ndim = aval.shape.rank();
                        let ax = spec.resolve(ndim)?;
                        if ax >= ndim {
                            return None;
                        }
                        let dim = aval.shape.dims[ax];
                        if dim == 0 {
                            return None;
                        }
                        if let Some(expected) = axis_size_opt {
                            if expected != dim {
                                return None;
                            }
                        } else {
                            axis_size_opt = Some(dim);
                        }
                        let mut stripped_dims = aval.shape.dims.clone();
                        stripped_dims.remove(ax);
                        resolved_axes.push(Some(ax));
                        next_peeled.push(AbstractValue {
                            dtype: aval.dtype,
                            shape: Shape {
                                dims: stripped_dims,
                            },
                        });
                    }
                }
            }
            let axis_size = axis_size_opt?;
            vmap_meta[stack_idx] = Some((axis_size, resolved_axes));
            current_peeled_in = next_peeled;
        }
    }

    let root_in_avals = current_peeled_in;

    // Determine root_output_avals.
    let has_grad = request.transform_stack.contains(&Transform::Grad);
    let root_out_avals = if has_grad {
        let dtype = root_in_avals.first().map_or(DType::F64, |aval| aval.dtype);
        vec![AbstractValue {
            dtype,
            shape: Shape::scalar(),
        }]
    } else {
        let mut current_peeled_out = outer_out_avals;
        for (stack_idx, &transform) in request.transform_stack.iter().enumerate() {
            if transform == Transform::Vmap {
                let (axis_size, _) = vmap_meta[stack_idx].as_ref()?;
                let out_specs =
                    parse_vmap_out_axes(&request.compile_options, current_peeled_out.len()).ok()?;
                if out_specs.len() != current_peeled_out.len() {
                    return None;
                }
                let mut next_out = Vec::with_capacity(current_peeled_out.len());
                for (aval, spec) in current_peeled_out.iter().zip(out_specs.iter()) {
                    match spec {
                        AxisSpec::NotBatched => next_out.push(aval.clone()),
                        AxisSpec::Batched(_) => {
                            let ndim = aval.shape.rank();
                            let ax = spec.resolve(ndim)?;
                            if ax >= ndim || aval.shape.dims[ax] != *axis_size {
                                return None;
                            }
                            let mut dims = aval.shape.dims.clone();
                            dims.remove(ax);
                            next_out.push(AbstractValue {
                                dtype: aval.dtype,
                                shape: Shape { dims },
                            });
                        }
                    }
                }
                current_peeled_out = next_out;
            }
        }
        current_peeled_out
    };

    let mut steps = Vec::with_capacity(n);
    let mut current_jaxpr: Cow<'_, Jaxpr> = Cow::Borrowed(request.root_jaxpr);
    let mut current_in_avals = root_in_avals.clone();
    let mut current_out_avals = root_out_avals.clone();

    for step_index in 0..n {
        let stack_idx = n - 1 - step_index;
        let transform = request.transform_stack[stack_idx];
        let evidence = request.transform_evidence.get(stack_idx)?.trim().to_owned();
        let input_sha256 = jaxpr_fingerprint_sha256(&current_jaxpr);
        let has_outer_grad =
            (0..stack_idx).any(|outer_i| request.transform_stack[outer_i] == Transform::Grad);

        match transform {
            Transform::Jit => {
                let (next_jaxpr, next_in_avals, witness) = if stack_idx == 0
                    && !static_argnums.is_empty()
                    && let Some(staged) = &plan.staged_program
                {
                    let dyn_in: Vec<AbstractValue> = current_in_avals
                        .iter()
                        .enumerate()
                        .filter_map(|(i, a)| (!static_argnums.contains(&i)).then_some(a.clone()))
                        .collect();
                    let w = TransformRewriteWitness::Jit(JitWitness {
                        static_argnums: static_argnums.to_vec(),
                        folded_eqn_count: staged.jaxpr_known.equations.len(),
                        residual_eqn_count: staged.jaxpr_unknown.equations.len(),
                    });
                    (Cow::Borrowed(&staged.jaxpr_unknown), dyn_in, w)
                } else if current_jaxpr.as_ref() == request.root_jaxpr {
                    let w = TransformRewriteWitness::Jit(JitWitness {
                        static_argnums: Vec::new(),
                        folded_eqn_count: 0,
                        residual_eqn_count: plan.effective_jaxpr.equations.len(),
                    });
                    (
                        Cow::Borrowed(plan.effective_jaxpr.as_ref()),
                        current_in_avals.clone(),
                        w,
                    )
                } else {
                    let w = TransformRewriteWitness::Jit(JitWitness {
                        static_argnums: Vec::new(),
                        folded_eqn_count: 0,
                        residual_eqn_count: current_jaxpr.equations.len(),
                    });
                    (current_jaxpr.clone(), current_in_avals.clone(), w)
                };
                let output_sha256 = jaxpr_fingerprint_sha256(&next_jaxpr);
                let next_out_avals = current_out_avals.clone();
                steps.push(TransformProofStep {
                    step_index,
                    transform,
                    evidence,
                    input_jaxpr_fingerprint_sha256: input_sha256,
                    output_jaxpr_fingerprint_sha256: output_sha256,
                    input_avals: next_in_avals.clone(),
                    output_avals: next_out_avals.clone(),
                    witness,
                });
                current_jaxpr = next_jaxpr;
                current_in_avals = next_in_avals;
                current_out_avals = next_out_avals;
            }
            Transform::Grad => {
                if current_out_avals.is_empty() || current_out_avals[0].shape.rank() != 0 {
                    return None;
                }
                let (argnums, _) =
                    parse_grad_argnums(&request.compile_options, current_in_avals.len()).ok()?;
                let mut target_for_grad = current_jaxpr.clone().into_owned();
                if has_outer_grad && current_in_avals.first().is_some_and(|a| a.shape.rank() > 0) {
                    append_reduce_sum_to_scalar(&mut target_for_grad);
                }
                let primal_eqn_count = target_for_grad.equations.len();
                let mut grad_jaxpr = fj_ad::build_grad_jaxpr(&target_for_grad, &argnums).ok()?;
                if has_outer_grad && current_in_avals.first().is_some_and(|a| a.shape.rank() > 0) {
                    append_reduce_sum_to_scalar(&mut grad_jaxpr);
                }
                let total_eqn_count = grad_jaxpr.equations.len();
                let adjoint_eqn_count = total_eqn_count.saturating_sub(primal_eqn_count);
                let pruned_dead_cotangents =
                    (primal_eqn_count.saturating_mul(2)).saturating_sub(total_eqn_count);
                let output_sha256 = jaxpr_fingerprint_sha256(&grad_jaxpr);
                let next_in_avals = current_in_avals.clone();
                let next_out_avals: Vec<AbstractValue> = if has_outer_grad
                    && current_in_avals.first().is_some_and(|a| a.shape.rank() > 0)
                {
                    return None;
                } else {
                    argnums
                        .iter()
                        .map(|&idx| current_in_avals.get(idx).cloned())
                        .collect::<Option<Vec<_>>>()?
                };
                let witness = TransformRewriteWitness::Grad(GradWitness {
                    argnums,
                    primal_eqn_count,
                    adjoint_eqn_count,
                    pruned_dead_cotangents,
                });
                steps.push(TransformProofStep {
                    step_index,
                    transform,
                    evidence,
                    input_jaxpr_fingerprint_sha256: input_sha256,
                    output_jaxpr_fingerprint_sha256: output_sha256,
                    input_avals: next_in_avals.clone(),
                    output_avals: next_out_avals.clone(),
                    witness,
                });
                current_jaxpr = Cow::Owned(grad_jaxpr);
                current_in_avals = next_in_avals;
                current_out_avals = next_out_avals;
            }
            Transform::Vmap => {
                let (axis_size, in_axes_opt) = vmap_meta.get(stack_idx)?.clone()?;
                let next_in_avals: Vec<AbstractValue> = current_in_avals
                    .iter()
                    .zip(in_axes_opt.iter())
                    .map(|(aval, ax)| match ax {
                        Some(a) => insert_aval_dim(aval, *a, axis_size),
                        None => aval.clone(),
                    })
                    .collect();
                let out_specs =
                    parse_vmap_out_axes(&request.compile_options, current_out_avals.len()).ok()?;
                if out_specs.len() != current_out_avals.len() {
                    return None;
                }
                let mut out_axes_opt = Vec::with_capacity(current_out_avals.len());
                let mut next_out_avals = Vec::with_capacity(current_out_avals.len());
                for (aval, spec) in current_out_avals.iter().zip(out_specs.iter()) {
                    if has_outer_grad {
                        out_axes_opt.push(None);
                        next_out_avals.push(aval.clone());
                    } else {
                        match spec {
                            AxisSpec::NotBatched => {
                                out_axes_opt.push(None);
                                next_out_avals.push(aval.clone());
                            }
                            AxisSpec::Batched(_) => {
                                let out_ndim = aval.shape.rank() + 1;
                                let ax = spec.resolve(out_ndim)?;
                                if ax >= out_ndim {
                                    return None;
                                }
                                out_axes_opt.push(Some(ax));
                                next_out_avals.push(insert_aval_dim(aval, ax, axis_size));
                            }
                        }
                    }
                }
                let mut batched_jaxpr = Jaxpr::new(
                    current_jaxpr.invars.clone(),
                    current_jaxpr.constvars.clone(),
                    current_jaxpr.outvars.clone(),
                    current_jaxpr.equations.clone(),
                );
                batched_jaxpr.effects = current_jaxpr.effects.clone();
                if !batched_jaxpr.equations.is_empty() {
                    batched_jaxpr.equations[0].params.insert(
                        "__ttl_vmap_axis_size".to_owned(),
                        format!("{axis_size}:{in_axes_opt:?}:{out_axes_opt:?}"),
                    );
                }
                let output_sha256 = jaxpr_fingerprint_sha256(&batched_jaxpr);
                let witness = TransformRewriteWitness::Vmap(VmapWitness {
                    in_axes: in_axes_opt,
                    out_axes: out_axes_opt,
                    axis_size,
                });
                steps.push(TransformProofStep {
                    step_index,
                    transform,
                    evidence,
                    input_jaxpr_fingerprint_sha256: input_sha256,
                    output_jaxpr_fingerprint_sha256: output_sha256,
                    input_avals: next_in_avals.clone(),
                    output_avals: next_out_avals.clone(),
                    witness,
                });
                current_jaxpr = Cow::Owned(batched_jaxpr);
                current_in_avals = next_in_avals;
                current_out_avals = next_out_avals;
            }
            Transform::Pmap => {
                return None;
            }
        }
    }

    verify_transform_composition_steps(
        request.root_jaxpr,
        request.transform_stack,
        request.transform_evidence,
        &root_in_avals,
        &root_out_avals,
        &steps,
    )
    .ok()
}

fn execute_with_transforms(
    root_jaxpr: &Jaxpr,
    transforms: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    // Skip leading Jit transforms (no-op pass-through) to find the first
    // non-Jit transform, avoiding recursive stack frames for Jit chains.
    let non_jit_start = transforms
        .iter()
        .position(|t| *t != Transform::Jit)
        .unwrap_or(transforms.len());

    let remaining = &transforms[non_jit_start..];
    let Some((head, tail)) = remaining.split_first() else {
        return backend
            .execute(root_jaxpr, args, device)
            .map_err(DispatchError::from);
    };

    match head {
        Transform::Jit => {
            execute_with_transforms(root_jaxpr, tail, args, backend, device, compile_options)
        }
        Transform::Grad => execute_grad(root_jaxpr, tail, args, backend, device, compile_options),
        Transform::Vmap => execute_vmap(root_jaxpr, tail, args, backend, device, compile_options),
        Transform::Pmap => Err(DispatchError::TransformExecution(
            TransformExecutionError::PmapUnavailable,
        )),
    }
}

fn execute_grad(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    if args.is_empty() {
        return Err(TransformExecutionError::EmptyArgumentList {
            transform: Transform::Grad,
        }
        .into());
    }

    // A trailing Jit is semantically transparent for symbolic AD. Other
    // remaining transforms use symbolic Jaxpr-to-Jaxpr AD lowering when
    // available, or fall back to finite-difference if enabled.
    if !tail.is_empty() {
        let non_jit_start = tail
            .iter()
            .position(|t| *t != Transform::Jit)
            .unwrap_or(tail.len());
        let effective_tail = &tail[non_jit_start..];
        if effective_tail.is_empty() {
            return execute_symbolic_grad(root_jaxpr, args, compile_options);
        }
        if !allows_finite_diff_grad_fallback(compile_options) {
            return Err(TransformExecutionError::FiniteDiffGradFallbackDisabled {
                tail: transform_tail_summary(tail),
            }
            .into());
        }
        if effective_tail
            .iter()
            .all(|transform| matches!(transform, Transform::Grad | Transform::Jit))
            && custom_vjp_rule_key(compile_options).is_none()
        {
            let (argnums, _) = parse_grad_argnums(compile_options, args.len())?;
            let first_argnum = argnums.first().copied().unwrap_or(0);
            let reduce_tensor_steps = matches!(args.get(first_argnum), Some(Value::Tensor(_)));
            if let Some(lowered) = try_lower_symbolic_grad_chain(
                root_jaxpr,
                effective_tail,
                &argnums,
                false,
                reduce_tensor_steps,
                true,
            ) && let Ok(outputs) = execute_symbolic_grad(&lowered, args, compile_options)
            {
                return Ok(outputs);
            }
        }
        if let Some((Transform::Vmap, vmap_tail)) = effective_tail.split_first()
            && vmap_tail
                .iter()
                .all(|transform| matches!(transform, Transform::Grad | Transform::Jit))
            && custom_vjp_rule_key(compile_options).is_none()
            && can_execute_symbolic_grad_vmap(root_jaxpr, compile_options)
        {
            return execute_symbolic_grad_vmap(
                root_jaxpr,
                vmap_tail,
                args,
                backend,
                device,
                compile_options,
            );
        }
        args[0]
            .as_f64_scalar()
            .ok_or(TransformExecutionError::NonScalarGradientInput)?;
        return execute_grad_finite_diff(root_jaxpr, tail, args, backend, device, compile_options);
    }

    execute_symbolic_grad(root_jaxpr, args, compile_options)
}

fn append_reduce_sum_to_scalar(jaxpr: &mut Jaxpr) {
    if jaxpr.outvars.len() != 1 {
        return;
    }
    let out_var = jaxpr.outvars[0];
    let max_var = jaxpr
        .invars
        .iter()
        .chain(jaxpr.constvars.iter())
        .chain(jaxpr.outvars.iter())
        .chain(jaxpr.equations.iter().flat_map(|eqn| eqn.outputs.iter()))
        .map(|v| v.0)
        .max()
        .unwrap_or(0);
    let reduced_var = fj_core::VarId(max_var + 1);
    jaxpr.equations.push(fj_core::Equation {
        primitive: Primitive::ReduceSum,
        inputs: [Atom::Var(out_var)].into_iter().collect(),
        outputs: [reduced_var].into_iter().collect(),
        params: BTreeMap::new(),
        effects: Vec::new(),
        sub_jaxprs: Vec::new(),
    });
    jaxpr.outvars[0] = reduced_var;
}

fn try_lower_symbolic_grad_chain(
    root_jaxpr: &Jaxpr,
    grad_transforms: &[Transform],
    outer_argnums: &[usize],
    outer_include_value: bool,
    reduce_tensor_steps: bool,
    reduce_final_step: bool,
) -> Option<Jaxpr> {
    if !grad_transforms
        .iter()
        .all(|t| matches!(t, Transform::Grad | Transform::Jit))
    {
        return None;
    }
    let grad_count = grad_transforms
        .iter()
        .filter(|&&t| t == Transform::Grad)
        .count();
    if grad_count == 0 {
        return None;
    }
    let inner_argnum = outer_argnums.first().copied().unwrap_or(0);
    let mut current = root_jaxpr.clone();
    for step in 0..grad_count {
        let is_outer = step + 1 == grad_count;
        current = if is_outer && outer_include_value {
            fj_ad::build_value_and_grad_jaxpr(&current, outer_argnums).ok()?
        } else if is_outer {
            fj_ad::build_grad_jaxpr(&current, outer_argnums).ok()?
        } else {
            fj_ad::build_grad_jaxpr(&current, &[inner_argnum]).ok()?
        };
        if reduce_tensor_steps && (!is_outer || reduce_final_step) {
            append_reduce_sum_to_scalar(&mut current);
        }
    }
    Some(current)
}

fn jaxpr_ends_with_reduction(root_jaxpr: &Jaxpr) -> bool {
    if root_jaxpr.outvars.len() != 1 || !root_jaxpr.effects.is_empty() {
        return false;
    }
    let out_var = root_jaxpr.outvars[0];
    root_jaxpr.equations.iter().any(|eqn| {
        eqn.outputs.as_slice() == [out_var]
            && matches!(
                eqn.primitive,
                Primitive::ReduceSum
                    | Primitive::ReduceMax
                    | Primitive::ReduceMin
                    | Primitive::ReduceProd
                    | Primitive::Dot
            )
    })
}

fn can_execute_symbolic_grad_vmap(
    root_jaxpr: &Jaxpr,
    compile_options: &BTreeMap<String, String>,
) -> bool {
    let explicit_reduce = compile_options
        .get("grad_vmap_reduce")
        .is_some_and(|v| v.eq_ignore_ascii_case("sum"));
    let out_axes_none = compile_options
        .get("vmap_out_axes")
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("none"));
    explicit_reduce || out_axes_none || jaxpr_ends_with_reduction(root_jaxpr)
}

fn execute_symbolic_grad_vmap(
    root_jaxpr: &Jaxpr,
    vmap_tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    let (argnums, _explicit) = parse_grad_argnums(compile_options, args.len())?;
    let in_axes = parse_vmap_in_axes(compile_options, args.len())?;
    let include_value = wants_value_and_grad(compile_options);

    let mut current = root_jaxpr.clone();
    let tail_grads = vmap_tail.iter().filter(|&&t| t == Transform::Grad).count();
    let first_argnum = argnums.first().copied().unwrap_or(0);
    let per_slice_is_tensor = match (args.get(first_argnum), in_axes.get(first_argnum)) {
        (Some(Value::Tensor(t)), Some(AxisSpec::Batched(_))) => t.rank() > 1,
        (Some(Value::Tensor(_)), _) => true,
        _ => false,
    };
    for _ in 0..tail_grads {
        current = fj_ad::build_grad_jaxpr(&current, &[first_argnum])
            .map_err(|e| TransformExecutionError::TensorBuild(format!("AD lowering error: {e}")))?;
        if per_slice_is_tensor {
            append_reduce_sum_to_scalar(&mut current);
        }
    }

    let lowered = if include_value {
        fj_ad::build_value_and_grad_jaxpr(&current, &argnums)
    } else {
        fj_ad::build_grad_jaxpr(&current, &argnums)
    }
    .map_err(|e| match e {
        fj_ad::AdError::NonScalarGradientOutput => TransformExecutionError::NonScalarGradientOutput,
        other => TransformExecutionError::TensorBuild(format!("AD error: {other}")),
    })?;

    let mut vmap_opts = compile_options.clone();
    vmap_opts.remove("vmap_out_axes");
    let mut mapped_outputs = execute_vmap(&lowered, &[], args, backend, device, &vmap_opts)?;

    let mut reduce_axis0_params = BTreeMap::new();
    reduce_axis0_params.insert("axes".to_owned(), "0".to_owned());

    if include_value && !mapped_outputs.is_empty() {
        mapped_outputs[0] = fj_lax::eval_primitive(
            Primitive::ReduceSum,
            std::slice::from_ref(&mapped_outputs[0]),
            &reduce_axis0_params,
        )
        .map_err(|e| TransformExecutionError::TensorBuild(e.to_string()))?;
    }

    let grad_offset = usize::from(include_value);
    for (k, &argnum) in argnums.iter().enumerate() {
        if matches!(in_axes.get(argnum), Some(AxisSpec::NotBatched))
            && let Some(slot) = mapped_outputs.get_mut(grad_offset + k)
        {
            *slot = fj_lax::eval_primitive(
                Primitive::ReduceSum,
                std::slice::from_ref(slot),
                &reduce_axis0_params,
            )
            .map_err(|e| TransformExecutionError::TensorBuild(e.to_string()))?;
        }
    }

    Ok(mapped_outputs)
}

fn vmap_per_slice_output_guaranteed_scalar(
    root_jaxpr: &Jaxpr,
    args: &[Value],
    in_axes: &[AxisSpec],
) -> bool {
    if root_jaxpr.outvars.len() != 1 || !root_jaxpr.effects.is_empty() {
        return false;
    }
    let all_per_slice_scalars = args
        .iter()
        .zip(in_axes.iter())
        .all(|(arg, axis)| match axis {
            AxisSpec::NotBatched => matches!(arg, Value::Scalar(_)),
            AxisSpec::Batched(_) => matches!(arg, Value::Tensor(t) if t.rank() == 1),
        });
    if all_per_slice_scalars && can_use_nonzero_axis_batch_trace(root_jaxpr) {
        return true;
    }
    let out_var = root_jaxpr.outvars[0];
    root_jaxpr.equations.iter().any(|eqn| {
        eqn.outputs.as_slice() == [out_var]
            && eqn.primitive == Primitive::ReduceSum
            && eqn
                .params
                .get("axes")
                .is_none_or(|axes| axes.trim().is_empty())
    })
}

fn execute_symbolic_grad(
    root_jaxpr: &Jaxpr,
    args: &[Value],
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    let (argnums, explicit_argnums) = parse_grad_argnums(compile_options, args.len())?;
    if wants_value_and_grad(compile_options) {
        // Shared forward pass for value_and_grad mode.
        let (mut values, grads) = if let Some(rule_key) = custom_vjp_rule_key(compile_options) {
            fj_ad::value_and_grad_jaxpr_with_custom_vjp_key(root_jaxpr, args, rule_key)
        } else {
            fj_ad::value_and_grad_jaxpr(root_jaxpr, args)
        }
        .map_err(|e| match e {
            fj_ad::AdError::NonScalarGradientOutput => {
                TransformExecutionError::NonScalarGradientOutput
            }
            other => TransformExecutionError::TensorBuild(format!("AD error: {other}")),
        })?;
        if explicit_argnums {
            for &idx in &argnums {
                values.push(grads.get(idx).cloned().unwrap_or(Value::scalar_f64(0.0)));
            }
        } else {
            // value_and_grad returns the primal outputs followed by one gradient
            // per input, matching fj_ad::value_and_grad_jaxpr.
            values.extend(grads);
        }
        return Ok(values);
    }

    // Tensor-aware AD: grad_jaxpr returns Value gradients for all inputs.
    let grads = if let Some(rule_key) = custom_vjp_rule_key(compile_options) {
        fj_ad::grad_jaxpr_with_custom_vjp_key(root_jaxpr, args, rule_key)
    } else {
        fj_ad::grad_jaxpr(root_jaxpr, args)
    }
    .map_err(|e| match e {
        fj_ad::AdError::NonScalarGradientOutput => TransformExecutionError::NonScalarGradientOutput,
        other => TransformExecutionError::TensorBuild(format!("AD error: {other}")),
    })?;
    if explicit_argnums {
        return Ok(argnums
            .into_iter()
            .map(|idx| grads.get(idx).cloned().unwrap_or(Value::scalar_f64(0.0)))
            .collect());
    }
    // Return gradient of first input (matches JAX's default grad behavior).
    Ok(vec![
        grads.into_iter().next().unwrap_or(Value::scalar_f64(0.0)),
    ])
}

fn execute_grad_finite_diff(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    let input_value = args[0]
        .as_f64_scalar()
        .ok_or(TransformExecutionError::NonScalarGradientInput)?;

    let epsilon = 1e-6_f64;
    let mut plus_args = args.to_vec();
    let mut minus_args = args.to_vec();
    plus_args[0] = Value::scalar_f64(input_value + epsilon);
    minus_args[0] = Value::scalar_f64(input_value - epsilon);

    let plus_out = execute_with_transforms(
        root_jaxpr,
        tail,
        &plus_args,
        backend,
        device,
        compile_options,
    )?;
    let minus_out = execute_with_transforms(
        root_jaxpr,
        tail,
        &minus_args,
        backend,
        device,
        compile_options,
    )?;

    let plus_value = plus_out
        .first()
        .and_then(Value::as_f64_scalar)
        .ok_or(TransformExecutionError::NonScalarGradientOutput)?;
    let minus_value = minus_out
        .first()
        .and_then(Value::as_f64_scalar)
        .ok_or(TransformExecutionError::NonScalarGradientOutput)?;

    let derivative = (plus_value - minus_value) / (2.0 * epsilon);
    Ok(vec![Value::scalar_f64(derivative)])
}

fn execute_vmap(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    if args.is_empty() {
        return Err(TransformExecutionError::EmptyArgumentList {
            transform: Transform::Vmap,
        }
        .into());
    }

    // Parse in_axes from compile_options
    let in_axes = parse_vmap_in_axes(compile_options, args.len())?;

    // Validate in_axes length matches args
    if in_axes.len() != args.len() {
        return Err(TransformExecutionError::VmapAxesCountMismatch {
            expected: args.len(),
            actual: in_axes.len(),
        }
        .into());
    }

    // Determine batch size from the first batched argument
    let mut batch_size: Option<usize> = None;
    for (arg, axis_spec) in args.iter().zip(in_axes.iter()) {
        if let AxisSpec::Batched(axis) = axis_spec {
            match arg {
                Value::Tensor(tensor) => {
                    let rank = tensor.rank();
                    let resolved = axis_spec.resolve(rank).ok_or(
                        TransformExecutionError::VmapAxesOutOfBounds {
                            axis: *axis,
                            ndim: rank,
                        },
                    )?;
                    if resolved >= rank {
                        return Err(TransformExecutionError::VmapAxesOutOfBounds {
                            axis: *axis,
                            ndim: rank,
                        }
                        .into());
                    }
                    let dim_size = tensor.shape.dims[resolved] as usize;
                    if let Some(expected) = batch_size {
                        if dim_size != expected {
                            return Err(TransformExecutionError::VmapMismatchedLeadingDimension {
                                expected,
                                actual: dim_size,
                            }
                            .into());
                        }
                    } else {
                        batch_size = Some(dim_size);
                    }
                }
                Value::Scalar(_) => {
                    // Scalar with batched axis is invalid
                    return Err(TransformExecutionError::VmapRequiresRankOneLeadingArgument.into());
                }
            }
        }
    }

    let lead_len = batch_size.ok_or(TransformExecutionError::EmptyVmapOutput)?;
    if lead_len == 0 {
        return Err(TransformExecutionError::EmptyVmapOutput.into());
    }

    // Use BatchTrace interpreter when no tail transforms exist. Axis-0 remains
    // the broad default path; non-zero axes are currently limited to effect-free
    // elementwise Jaxprs whose batching rules preserve the mapped axis.
    let all_axis_zero = in_axes
        .iter()
        .all(|spec| matches!(spec, AxisSpec::Batched(0) | AxisSpec::NotBatched));
    let out_axes = if compile_options.contains_key("vmap_out_axes") {
        Some(parse_vmap_out_axes(
            compile_options,
            root_jaxpr.outvars.len(),
        )?)
    } else {
        None
    };
    let out_axes_none_batch_independent = match out_axes.as_deref() {
        None => false,
        Some(specs) => can_use_out_axes_none_batch_trace(root_jaxpr, &in_axes, specs)?,
    };

    if matches!(tail, [Transform::Jit]) && out_axes_none_batch_independent {
        let mapped_args = map_vmap_args_at_index(args, &in_axes, 0)?;
        let tail_compile_options = compile_options_for_transform_tail(compile_options);
        return execute_with_transforms(
            root_jaxpr,
            tail,
            &mapped_args,
            backend,
            device,
            &tail_compile_options,
        );
    }

    if tail.is_empty()
        && out_axes.is_none()
        && all_axis_zero
        && let Some(outputs) = execute_vmap_paired_i64_dot_direct(root_jaxpr, args, &in_axes)?
    {
        return Ok(outputs);
    }

    let can_use_batch_trace_out_axes =
        out_axes.is_none() || (tail.is_empty() && out_axes_none_batch_independent);
    if tail.is_empty()
        && can_use_batch_trace_out_axes
        && (all_axis_zero || can_use_nonzero_axis_batch_trace(root_jaxpr))
    {
        return execute_vmap_batch_trace(root_jaxpr, args, &in_axes, lead_len, out_axes.is_some());
    }

    let grad_only_tail = !tail.is_empty()
        && tail
            .iter()
            .all(|t| matches!(t, Transform::Grad | Transform::Jit))
        && tail.contains(&Transform::Grad);
    if grad_only_tail {
        let grad_count = tail.iter().filter(|&&t| t == Transform::Grad).count();
        let (argnums, explicit_argnums) = parse_grad_argnums(compile_options, args.len())?;
        let include_value = wants_value_and_grad(compile_options);
        let outer_argnums: Vec<usize> = if include_value && !explicit_argnums {
            (0..root_jaxpr.invars.len()).collect()
        } else {
            argnums
        };
        let first_argnum = outer_argnums.first().copied().unwrap_or(0);
        let per_slice_is_tensor = match (args.get(first_argnum), in_axes.get(first_argnum)) {
            (Some(Value::Tensor(t)), Some(AxisSpec::Batched(_))) => t.rank() > 1,
            (Some(Value::Tensor(_)), _) => true,
            _ => false,
        };
        if (grad_count == 1 || allows_finite_diff_grad_fallback(compile_options))
            && custom_vjp_rule_key(compile_options).is_none()
            && out_axes.is_none()
            && vmap_per_slice_output_guaranteed_scalar(root_jaxpr, args, &in_axes)
            && let Some(lowered) = try_lower_symbolic_grad_chain(
                root_jaxpr,
                tail,
                &outer_argnums,
                include_value,
                per_slice_is_tensor,
                false,
            )
            && (all_axis_zero || can_use_nonzero_axis_batch_trace(&lowered))
            && let Ok(outputs) = execute_vmap_batch_trace(&lowered, args, &in_axes, lead_len, false)
        {
            return Ok(outputs);
        }
    }

    // Fall back to loop-and-stack for non-trivial axes or composed transforms
    execute_vmap_loop_and_stack(
        root_jaxpr,
        tail,
        args,
        backend,
        device,
        lead_len,
        &in_axes,
        compile_options,
    )
}

fn execute_vmap_paired_i64_dot_direct(
    root_jaxpr: &Jaxpr,
    args: &[Value],
    in_axes: &[AxisSpec],
) -> Result<Option<Vec<Value>>, DispatchError> {
    if !is_binary_dot_jaxpr(root_jaxpr)
        || args.len() != 2
        || in_axes.len() != 2
        || !in_axes
            .iter()
            .all(|axis| matches!(axis, AxisSpec::Batched(0)))
    {
        return Ok(None);
    }

    let (Value::Tensor(lhs), Value::Tensor(rhs)) = (&args[0], &args[1]) else {
        return Ok(None);
    };
    if lhs.dtype != DType::I64
        || rhs.dtype != DType::I64
        || lhs.rank() != 2
        || rhs.rank() != 2
        || lhs.shape.dims != rhs.shape.dims
    {
        return Ok(None);
    }

    let batch = lhs.shape.dims[0] as usize;
    let width = lhs.shape.dims[1] as usize;
    let Some(expected_len) = batch.checked_mul(width) else {
        return Ok(None);
    };
    if lhs.elements.len() != expected_len || rhs.elements.len() != expected_len {
        return Ok(None);
    }

    let mut elements = Vec::with_capacity(batch);
    if let (Some(lhs_values), Some(rhs_values)) =
        (lhs.elements.as_i64_slice(), rhs.elements.as_i64_slice())
    {
        for (lhs_row, rhs_row) in lhs_values
            .chunks_exact(width)
            .zip(rhs_values.chunks_exact(width))
        {
            let mut sum = 0_i64;
            for (&left, &right) in lhs_row.iter().zip(rhs_row) {
                sum = sum.wrapping_add(left.wrapping_mul(right));
            }
            elements.push(Literal::I64(sum));
        }
    } else {
        for batch_idx in 0..batch {
            let offset = batch_idx * width;
            let mut sum = 0_i64;
            for kk in 0..width {
                let Literal::I64(left) = lhs.elements[offset + kk] else {
                    return Ok(None);
                };
                let Literal::I64(right) = rhs.elements[offset + kk] else {
                    return Ok(None);
                };
                sum = sum.wrapping_add(left.wrapping_mul(right));
            }
            elements.push(Literal::I64(sum));
        }
    }

    let tensor = TensorValue::new(DType::I64, Shape::vector(lhs.shape.dims[0]), elements).map_err(
        |error| {
            DispatchError::TransformExecution(TransformExecutionError::TensorBuild(
                error.to_string(),
            ))
        },
    )?;
    Ok(Some(vec![Value::Tensor(tensor)]))
}

fn is_binary_dot_jaxpr(root_jaxpr: &Jaxpr) -> bool {
    let [input_0, input_1] = root_jaxpr.invars.as_slice() else {
        return false;
    };
    let [output] = root_jaxpr.outvars.as_slice() else {
        return false;
    };
    let [equation] = root_jaxpr.equations.as_slice() else {
        return false;
    };

    root_jaxpr.constvars.is_empty()
        && root_jaxpr.effects.is_empty()
        && equation.primitive == Primitive::Dot
        && equation.params.is_empty()
        && equation.sub_jaxprs.is_empty()
        && equation.effects.is_empty()
        && equation.inputs.as_slice() == [Atom::Var(*input_0), Atom::Var(*input_1)]
        && equation.outputs.as_slice() == [*output]
}

fn can_use_out_axes_none_batch_trace(
    root_jaxpr: &Jaxpr,
    in_axes: &[AxisSpec],
    out_axes: &[AxisSpec],
) -> Result<bool, DispatchError> {
    if out_axes.len() != root_jaxpr.outvars.len() {
        return Err(TransformExecutionError::VmapAxesCountMismatch {
            expected: root_jaxpr.outvars.len(),
            actual: out_axes.len(),
        }
        .into());
    }
    if !out_axes
        .iter()
        .all(|spec| matches!(spec, AxisSpec::NotBatched))
    {
        return Ok(false);
    }
    Ok(jaxpr_outputs_are_batch_independent(root_jaxpr, in_axes))
}

fn jaxpr_outputs_are_batch_independent(root_jaxpr: &Jaxpr, in_axes: &[AxisSpec]) -> bool {
    if !root_jaxpr.effects.is_empty() {
        return false;
    }

    let mut batched_vars = BTreeMap::new();
    for (var, axis) in root_jaxpr.invars.iter().zip(in_axes.iter()) {
        batched_vars.insert(*var, matches!(axis, AxisSpec::Batched(_)));
    }
    for var in &root_jaxpr.constvars {
        batched_vars.insert(*var, false);
    }

    for equation in &root_jaxpr.equations {
        if !equation.effects.is_empty()
            || !equation.sub_jaxprs.is_empty()
            || !is_elementwise_batch_trace_primitive(equation.primitive)
        {
            return false;
        }
        let output_is_batched = equation.inputs.iter().any(|atom| match atom {
            Atom::Var(var) => *batched_vars.get(var).unwrap_or(&true),
            Atom::Lit(_) => false,
        });
        for output in &equation.outputs {
            batched_vars.insert(*output, output_is_batched);
        }
    }

    root_jaxpr
        .outvars
        .iter()
        .all(|var| !*batched_vars.get(var).unwrap_or(&true))
}

fn can_use_nonzero_axis_batch_trace(root_jaxpr: &Jaxpr) -> bool {
    root_jaxpr.effects.is_empty()
        && root_jaxpr.equations.iter().all(|equation| {
            equation.effects.is_empty()
                && equation.sub_jaxprs.is_empty()
                && is_elementwise_batch_trace_primitive(equation.primitive)
        })
}

fn is_elementwise_batch_trace_primitive(primitive: Primitive) -> bool {
    matches!(
        primitive,
        Primitive::Neg
            | Primitive::Abs
            | Primitive::Exp
            | Primitive::Log
            | Primitive::Sqrt
            | Primitive::Rsqrt
            | Primitive::Floor
            | Primitive::Ceil
            | Primitive::Round
            | Primitive::Sin
            | Primitive::Cos
            | Primitive::Tan
            | Primitive::Asin
            | Primitive::Acos
            | Primitive::Atan
            | Primitive::Sinh
            | Primitive::Cosh
            | Primitive::Tanh
            | Primitive::Asinh
            | Primitive::Acosh
            | Primitive::Atanh
            | Primitive::Expm1
            | Primitive::Log1p
            | Primitive::Sign
            | Primitive::Square
            | Primitive::Reciprocal
            | Primitive::Logistic
            | Primitive::Erf
            | Primitive::Erfc
            | Primitive::Lgamma
            | Primitive::Digamma
            | Primitive::ErfInv
            | Primitive::Conj
            | Primitive::Real
            | Primitive::Imag
            | Primitive::Cbrt
            | Primitive::IsFinite
            | Primitive::IntegerPow
            | Primitive::Copy
            | Primitive::ConvertElementType
            | Primitive::BitcastConvertType
            | Primitive::ReducePrecision
            | Primitive::Trunc
            | Primitive::Deg2Rad
            | Primitive::Rad2Deg
            | Primitive::Log2
            | Primitive::Exp2
            | Primitive::Sinc
            | Primitive::BesselI0e
            | Primitive::BesselI1e
            | Primitive::IsNan
            | Primitive::IsInf
            | Primitive::Signbit
            | Primitive::StopGradient
            | Primitive::Add
            | Primitive::Sub
            | Primitive::Mul
            | Primitive::Max
            | Primitive::Min
            | Primitive::Pow
            | Primitive::Div
            | Primitive::Rem
            | Primitive::Atan2
            | Primitive::Complex
            | Primitive::Nextafter
            | Primitive::Hypot
            | Primitive::LogAddExp
            | Primitive::LogAddExp2
            | Primitive::Gcd
            | Primitive::Lcm
            | Primitive::Polygamma
            | Primitive::Igamma
            | Primitive::Igammac
            | Primitive::Zeta
            | Primitive::Heaviside
            | Primitive::CopySign
            | Primitive::Ldexp
            | Primitive::XLogY
            | Primitive::XLog1PY
            | Primitive::Eq
            | Primitive::Ne
            | Primitive::Lt
            | Primitive::Le
            | Primitive::Gt
            | Primitive::Ge
            | Primitive::BitwiseAnd
            | Primitive::BitwiseOr
            | Primitive::BitwiseXor
            | Primitive::ShiftLeft
            | Primitive::ShiftRightArithmetic
            | Primitive::ShiftRightLogical
            | Primitive::BitwiseNot
            | Primitive::PopulationCount
            | Primitive::CountLeadingZeros
            | Primitive::CountTrailingZeros
            | Primitive::Select
            | Primitive::Clamp
    )
}

/// BatchTrace-based vmap execution: O(1) vectorized dispatch via per-primitive
/// batching rules. Each arg becomes a BatchTracer with the specified batch_dim
/// or None (scalars/unbatched), and the Jaxpr is evaluated equation-by-equation.
fn execute_vmap_batch_trace(
    root_jaxpr: &Jaxpr,
    args: &[Value],
    in_axes: &[AxisSpec],
    lead_len: usize,
    out_axes_none: bool,
) -> Result<Vec<Value>, DispatchError> {
    use batching::{BatchTracer, batch_eval_jaxpr};

    let batch_inputs: Vec<BatchTracer> = args
        .iter()
        .zip(in_axes.iter())
        .map(|(arg, axis_spec)| match axis_spec {
            AxisSpec::NotBatched => BatchTracer::unbatched(arg.clone()),
            AxisSpec::Batched(_) => {
                let resolved = match arg {
                    Value::Tensor(t) => axis_spec.resolve(t.rank()).unwrap_or(0),
                    Value::Scalar(_) => 0,
                };
                if matches!(arg, Value::Scalar(_)) {
                    BatchTracer::unbatched(arg.clone())
                } else {
                    BatchTracer::batched(arg.clone(), resolved)
                }
            }
        })
        .collect();

    let results = batch_eval_jaxpr(root_jaxpr, &batch_inputs)
        .map_err(|e| TransformExecutionError::TensorBuild(format!("BatchTrace error: {e}")))?;

    if out_axes_none {
        let mut outputs = Vec::with_capacity(results.len());
        for tracer in results {
            if tracer.batch_dim.is_some() {
                return Err(TransformExecutionError::VmapUnmappedOutputMismatch.into());
            }
            outputs.push(tracer.value);
        }
        return Ok(outputs);
    }

    // Extract output values, ensuring batch dim is at position 0
    let mut outputs = Vec::with_capacity(results.len());
    for tracer in results {
        match tracer.batch_dim {
            Some(0) => outputs.push(tracer.value),
            Some(bd) => {
                // Move batch dim to front for consistent output
                let moved = batching::move_batch_dim_to_front(&tracer.value, bd)
                    .map_err(|e| TransformExecutionError::TensorBuild(e.to_string()))?;
                outputs.push(moved);
            }
            None => {
                // Unbatched output — broadcast it across the mapped batch axis.
                // This happens when the function output doesn't depend on the input.
                let broadcast = TensorValue::repeat_axis0(&tracer.value, lead_len)
                    .map(Value::Tensor)
                    .map_err(|e| TransformExecutionError::TensorBuild(e.to_string()))?;
                outputs.push(broadcast);
            }
        }
    }

    Ok(outputs)
}

fn map_vmap_args_at_index(
    args: &[Value],
    in_axes: &[AxisSpec],
    index: usize,
) -> Result<Vec<Value>, DispatchError> {
    let mut mapped_args = Vec::with_capacity(args.len());
    for (arg, axis_spec) in args.iter().zip(in_axes.iter()) {
        match axis_spec {
            AxisSpec::NotBatched => {
                mapped_args.push(arg.clone());
            }
            AxisSpec::Batched(_) => match arg {
                Value::Scalar(lit) => mapped_args.push(Value::Scalar(*lit)),
                Value::Tensor(tensor) => {
                    let resolved_axis = axis_spec.resolve(tensor.rank()).unwrap_or(0);
                    if resolved_axis == 0 {
                        mapped_args.push(tensor.slice_axis0(index).map_err(|err| {
                            TransformExecutionError::TensorBuild(err.to_string())
                        })?);
                    } else {
                        let sliced = slice_along_axis(tensor, resolved_axis, index)
                            .map_err(TransformExecutionError::TensorBuild)?;
                        mapped_args.push(sliced);
                    }
                }
            },
        }
    }
    Ok(mapped_args)
}

/// Loop-and-stack vmap fallback for composed transforms (e.g., vmap(grad(f))).
#[allow(clippy::too_many_arguments)]
fn execute_vmap_loop_and_stack(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
    backend: &dyn Backend,
    device: DeviceId,
    lead_len: usize,
    in_axes: &[AxisSpec],
    compile_options: &BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    let mut per_output_values: Vec<Vec<Value>> = Vec::new();
    let tail_compile_options = compile_options_for_transform_tail(compile_options);

    for index in 0..lead_len {
        let mapped_args = map_vmap_args_at_index(args, in_axes, index)?;

        let mapped_output = execute_with_transforms(
            root_jaxpr,
            tail,
            &mapped_args,
            backend,
            device,
            &tail_compile_options,
        )?;
        if index == 0 {
            per_output_values = vec![Vec::with_capacity(lead_len); mapped_output.len()];
        } else if mapped_output.len() != per_output_values.len() {
            return Err(TransformExecutionError::VmapInconsistentOutputArity {
                expected: per_output_values.len(),
                actual: mapped_output.len(),
            }
            .into());
        }

        for (output_idx, value) in mapped_output.iter().enumerate() {
            per_output_values[output_idx].push(value.clone());
        }
    }

    let out_axes = parse_vmap_out_axes(compile_options, per_output_values.len())?;
    if out_axes.len() != per_output_values.len() {
        return Err(TransformExecutionError::VmapAxesCountMismatch {
            expected: per_output_values.len(),
            actual: out_axes.len(),
        }
        .into());
    }

    let mut outputs = Vec::with_capacity(per_output_values.len());
    for (out_idx, values) in per_output_values.iter().enumerate() {
        let out_axis = out_axes
            .get(out_idx)
            .copied()
            .unwrap_or(AxisSpec::Batched(0));
        match out_axis {
            AxisSpec::Batched(0) => {
                let tensor = TensorValue::stack_axis0(values)
                    .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?;
                outputs.push(Value::Tensor(tensor));
            }
            AxisSpec::NotBatched => {
                let Some(first) = values.first() else {
                    return Err(TransformExecutionError::EmptyVmapOutput.into());
                };
                if values.iter().skip(1).any(|value| value != first) {
                    return Err(TransformExecutionError::VmapUnmappedOutputMismatch.into());
                }
                outputs.push(first.clone());
            }
            AxisSpec::Batched(target_axis) => {
                // Move batch dim from 0 to target_axis via Transpose primitive
                let tensor = TensorValue::stack_axis0(values)
                    .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?;
                let rank = tensor.rank();
                let resolved = AxisSpec::Batched(target_axis).resolve(rank).ok_or(
                    TransformExecutionError::VmapAxesOutOfBounds {
                        axis: target_axis,
                        ndim: rank,
                    },
                )?;
                if resolved >= rank {
                    return Err(TransformExecutionError::VmapAxesOutOfBounds {
                        axis: target_axis,
                        ndim: rank,
                    }
                    .into());
                }
                if resolved == 0 {
                    outputs.push(Value::Tensor(tensor));
                } else {
                    // Build permutation: move axis 0 to position `resolved`
                    let mut perm: Vec<usize> = (0..rank).collect();
                    perm.remove(0);
                    perm.insert(resolved, 0);
                    let perm_str = perm
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    let mut params = BTreeMap::new();
                    params.insert("permutation".to_owned(), perm_str);
                    let transposed = fj_lax::eval_primitive(
                        fj_core::Primitive::Transpose,
                        &[Value::Tensor(tensor)],
                        &params,
                    )
                    .map_err(|err| TransformExecutionError::TensorBuild(err.to_string()))?;
                    outputs.push(transposed);
                }
            }
        }
    }

    Ok(outputs)
}

/// Slice a tensor along an arbitrary axis, extracting the `index`-th slice.
fn slice_along_axis(tensor: &TensorValue, axis: usize, index: usize) -> Result<Value, String> {
    let rank = tensor.rank();
    if axis >= rank {
        return Err(format!(
            "slice_along_axis: axis {axis} out of bounds for rank {rank}"
        ));
    }
    if axis == 0 {
        return tensor.slice_axis0(index).map_err(|e| e.to_string());
    }

    let axis_size = tensor.shape.dims[axis] as usize;
    if index >= axis_size {
        return Err(format!(
            "slice_along_axis: index {index} out of bounds for axis {axis} with size {axis_size}"
        ));
    }

    let out_dims: Vec<u32> = tensor
        .shape
        .dims
        .iter()
        .enumerate()
        .filter_map(|(dim_idx, dim)| (dim_idx != axis).then_some(*dim))
        .collect();
    let out_shape = Shape { dims: out_dims };
    let out_count = out_shape
        .element_count()
        .ok_or_else(|| {
            format!(
                "slice_along_axis: output shape overflowed: {:?}",
                out_shape.dims
            )
        })
        .and_then(|count| {
            usize::try_from(count)
                .map_err(|_| format!("slice_along_axis: output shape too large: {count}"))
        })?;

    if out_count == 0 {
        return Ok(Value::Tensor(
            TensorValue::new(tensor.dtype, out_shape, Vec::new()).map_err(|e| e.to_string())?,
        ));
    }

    let input_strides = row_major_strides(&tensor.shape.dims)?;
    let output_strides = row_major_strides(&out_shape.dims)?;
    let kept_axes: Vec<usize> = (0..rank).filter(|dim_idx| *dim_idx != axis).collect();
    let mut elements = Vec::with_capacity(out_count);

    for out_flat in 0..out_count {
        let mut source_flat = index
            .checked_mul(input_strides[axis])
            .ok_or_else(|| "slice_along_axis: source index overflowed".to_owned())?;

        for (out_axis, source_axis) in kept_axes.iter().copied().enumerate() {
            let dim = out_shape.dims[out_axis] as usize;
            let output_stride = output_strides[out_axis];
            if dim == 0 || output_stride == 0 {
                return Err("slice_along_axis: zero output dimension or stride".to_owned());
            }
            let coord = (out_flat / output_stride) % dim;
            let offset = coord
                .checked_mul(input_strides[source_axis])
                .ok_or_else(|| "slice_along_axis: source offset overflowed".to_owned())?;
            source_flat = source_flat
                .checked_add(offset)
                .ok_or_else(|| "slice_along_axis: source index overflowed".to_owned())?;
        }

        elements.push(tensor.elements[source_flat]);
    }

    Ok(Value::Tensor(
        TensorValue::new(tensor.dtype, out_shape, elements).map_err(|e| e.to_string())?,
    ))
}

fn row_major_strides(dims: &[u32]) -> Result<Vec<usize>, String> {
    let mut strides = vec![1; dims.len()];
    let mut stride = 1_usize;
    for (idx, dim) in dims.iter().enumerate().rev() {
        strides[idx] = stride;
        stride = stride
            .checked_mul(*dim as usize)
            .ok_or_else(|| format!("row-major stride overflowed for dims {dims:?}"))?;
    }
    Ok(strides)
}

#[inline]
fn heuristic_posterior_abandoned(ledger: &TraceTransformLedger) -> f64 {
    heuristic_posterior_abandoned_parts(&ledger.root_jaxpr, &ledger.transform_stack)
}

#[inline]
fn heuristic_posterior_abandoned_parts(root_jaxpr: &Jaxpr, transform_stack: &[Transform]) -> f64 {
    let eqn_factor = root_jaxpr.equations.len() as f64;
    let depth_factor = transform_stack.len() as f64;
    let workload_odds = (eqn_factor + 2.0 * depth_factor).max(1e-9);
    let mut posterior = LogDomainPosterior::new(1.0 / 21.0);
    posterior.update(workload_odds.ln());
    posterior.posterior_abandoned().clamp(0.05, 0.95)
}

fn compute_posterior_abandoned_with_signals(
    root_jaxpr: &Jaxpr,
    transform_stack: &[Transform],
    compile_options: &BTreeMap<String, String>,
    signals: &[EvidenceSignal],
) -> f64 {
    if let Some(raw_prior) = compile_options.get("prior_abandoned")
        && let Ok(prior) = raw_prior.trim().parse::<f64>()
    {
        let mut posterior = LogDomainPosterior::new(prior);
        for signal in signals {
            posterior.update(signal.log_likelihood_delta);
        }
        return posterior.posterior_abandoned().clamp(0.05, 0.99);
    }
    heuristic_posterior_abandoned_parts(root_jaxpr, transform_stack)
}

/// Compute posterior with conformal calibration when available.
#[must_use]
pub fn calibrated_posterior_abandoned(
    ledger: &TraceTransformLedger,
    conformal: Option<&ConformalPredictor>,
) -> f64 {
    let heuristic = heuristic_posterior_abandoned(ledger);
    match conformal {
        Some(cp) if cp.is_calibrated() => {
            let estimate = cp.calibrated_posterior(heuristic);
            estimate.point
        }
        _ => heuristic,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DispatchError, DispatchRequest, DispatchRequestRef, TransformExecutionError, dispatch,
        dispatch_ref, prepare_dispatch_meta,
    };
    use fj_core::{
        Atom, CompatibilityMode, DType, Equation, Jaxpr, Primitive, ProgramSpec, Shape,
        TensorValue, TraceTransformLedger, Transform, Value, VarId, build_program,
    };
    use smallvec::smallvec;
    use std::collections::BTreeMap;

    fn ledger(program: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
        let mut ledger = TraceTransformLedger::new(build_program(program));
        for (idx, transform) in transforms.iter().enumerate() {
            ledger.push_transform(
                *transform,
                format!("evidence-{}-{}", transform.as_str(), idx),
            );
        }
        ledger
    }

    #[test]
    fn dispatch_jit_add_scalar() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch should succeed");

        assert_eq!(response.outputs, vec![Value::scalar_i64(6)]);
        assert!(response.cache_key.starts_with("fjx-"));
        assert_eq!(response.evidence_ledger.len(), 1);
    }

    #[test]
    fn dispatch_grad_square_scalar() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch grad should succeed");

        let derivative = response.outputs[0]
            .as_f64_scalar()
            .expect("grad output should be scalar f64");
        assert!((derivative - 6.0).abs() < 1e-3);
    }

    #[test]
    fn dispatch_grad_jit_uses_symbolic_ad_when_fallback_is_disabled() {
        let mut compile_options = BTreeMap::new();
        compile_options.insert(
            "allow_finite_diff_grad_fallback".to_owned(),
            "false".to_owned(),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Jit]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("grad(jit(f)) should use symbolic AD because jit is transparent");

        let derivative = response.outputs[0]
            .as_f64_scalar()
            .expect("grad output should be scalar f64");
        assert!((derivative - 6.0).abs() < 1e-3);
    }

    #[test]
    fn dispatch_grad_grad_can_deny_finite_diff_fallback() {
        let mut compile_options = BTreeMap::new();
        compile_options.insert(
            "allow_finite_diff_grad_fallback".to_owned(),
            "deny".to_owned(),
        );
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("finite-difference fallback should be gateable");

        assert!(matches!(
            err,
            DispatchError::TransformExecution(
                TransformExecutionError::FiniteDiffGradFallbackDisabled { .. }
            )
        ));
    }

    #[test]
    fn dispatch_vmap_grad_grad_preserves_finite_diff_fallback_denial() {
        let mut compile_options = BTreeMap::new();
        compile_options.insert(
            "allow_finite_diff_grad_fallback".to_owned(),
            "false".to_owned(),
        );
        compile_options.insert("vmap_in_axes".to_owned(), "0".to_owned());

        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::Square,
                &[Transform::Vmap, Transform::Grad, Transform::Grad],
            ),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("vmap tail must preserve finite-difference fallback denial");

        assert!(matches!(
            err,
            DispatchError::TransformExecution(
                TransformExecutionError::FiniteDiffGradFallbackDisabled { .. }
            )
        ));
    }

    #[test]
    fn dispatch_value_and_grad_mode_square_scalar() {
        let mut compile_options = BTreeMap::new();
        compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch value_and_grad should succeed");

        assert_eq!(response.outputs.len(), 2, "expected [value, grad]");
        let value = response.outputs[0]
            .as_f64_scalar()
            .expect("value output should be scalar f64");
        let derivative = response.outputs[1]
            .as_f64_scalar()
            .expect("grad output should be scalar f64");
        assert!((value - 9.0).abs() < 1e-6);
        assert!((derivative - 6.0).abs() < 1e-3);
    }

    #[test]
    fn dispatch_value_and_grad_mode_multi_output_preserves_all_outputs() {
        let mut compile_options = BTreeMap::new();
        compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOneMulTwo, &[Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch value_and_grad should succeed");

        assert_eq!(
            response.outputs.len(),
            3,
            "expected both primal outputs plus the first-input gradient"
        );
        let values = response
            .outputs
            .iter()
            .map(|value| value.as_f64_scalar().expect("output should be f64"))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![4.0, 6.0, 1.0]);
    }

    #[test]
    fn dispatch_grad_vmap_vector_input_fails_closed_with_typed_error() {
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Grad, Transform::Vmap]),
            args: vec![Value::vector_i64(&[1, 2, 3]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("grad(vmap(...)) over a vector first input must fail closed");

        assert!(matches!(
            err,
            DispatchError::TransformExecution(TransformExecutionError::NonScalarGradientInput)
        ));
    }

    #[test]
    fn dispatch_vmap_empty_batch_fails_closed_with_typed_error() {
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![Value::vector_i64(&[]).expect("empty vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("empty vmap batch must fail closed");

        assert!(matches!(
            err,
            DispatchError::TransformExecution(TransformExecutionError::EmptyVmapOutput)
        ));
    }

    #[test]
    fn dispatch_grad_of_grad_square_scalar() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch grad-of-grad should succeed");

        let second_derivative = response.outputs[0]
            .as_f64_scalar()
            .expect("second derivative output should be scalar f64");
        assert!((second_derivative - 2.0).abs() < 1e-3);
    }

    #[test]
    fn dispatch_vmap_add_one_vector() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![Value::vector_i64(&[1, 2, 3]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap output should be tensor");
        let as_i64 = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(as_i64, vec![2, 3, 4]);
    }

    #[test]
    fn dispatch_vmap_dot_i64_paired_vectors_wraps_like_lax() {
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    fj_core::Literal::I64(i64::MAX),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(3),
                    fj_core::Literal::I64(4),
                ],
            )
            .expect("lhs should build"),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(5),
                    fj_core::Literal::I64(6),
                ],
            )
            .expect("rhs should build"),
        );

        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Dot3, &[Transform::Vmap]),
            args: vec![lhs, rhs],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap dot should dispatch");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap dot output should be a tensor");
        assert_eq!(output.dtype, DType::I64);
        assert_eq!(output.shape.dims, vec![2]);
        let values = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 output"))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![2, 39]);
        assert!(response.cache_key.starts_with("fjx-"));
        assert_eq!(response.evidence_ledger.len(), 1);
    }

    #[test]
    fn dispatch_vmap_broadcasts_constant_batch_trace_outputs() {
        use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, VarId};
        use smallvec::smallvec;

        let constant_jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Lit(Literal::I64(1)), Atom::Lit(Literal::I64(2))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            }],
        );
        let mut ttl = TraceTransformLedger::new(constant_jaxpr);
        ttl.push_transform(Transform::Vmap, "constant-vmap".to_owned());

        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl,
            args: vec![Value::vector_i64(&[10, 20, 30]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap with constant body should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("constant vmap output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![3] });
        let values = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![3, 3, 3]);
    }

    #[test]
    fn dispatch_vmap_add_one_rank2_tensor() {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    fj_core::Literal::I64(1),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(3),
                    fj_core::Literal::I64(4),
                ],
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap rank2 should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![2, 2] });
        let as_i64 = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(as_i64, vec![2, 3, 4, 5]);
    }

    #[test]
    fn dispatch_vmap_of_vmap_add_one_rank2_tensor() {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    fj_core::Literal::I64(1),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(3),
                    fj_core::Literal::I64(4),
                    fj_core::Literal::I64(5),
                    fj_core::Literal::I64(6),
                ],
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch nested vmap should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("nested vmap output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![2, 3] });
        let as_i64 = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect::<Vec<_>>();
        assert_eq!(as_i64, vec![2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn transform_order_is_explicit() {
        let good = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Vmap, Transform::Grad]),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(grad(f)) should be supported");

        let good_out = good.outputs[0]
            .as_tensor()
            .expect("output should be tensor")
            .to_f64_vec()
            .expect("output should be numeric");
        assert_eq!(good_out.len(), 3);
        assert!((good_out[0] - 2.0).abs() < 1e-3);
        assert!((good_out[1] - 4.0).abs() < 1e-3);
        assert!((good_out[2] - 6.0).abs() < 1e-3);

        let bad = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Vmap]),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("grad(vmap(f)) should fail with current constraints");

        assert!(matches!(
            bad,
            DispatchError::TransformExecution(
                super::TransformExecutionError::NonScalarGradientInput
            )
        ));
    }

    #[test]
    fn strict_mode_rejects_unknown_features_fail_closed() {
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["future.backend.protocol.v2".to_owned()],
        })
        .expect_err("strict mode must reject unknown incompatible features");

        assert!(
            matches!(
                &err,
                DispatchError::Cache(fj_cache::CacheKeyError::UnknownIncompatibleFeatures { .. })
            ),
            "expected fail-closed cache-key rejection, got: {err:?}"
        );
        if let DispatchError::Cache(fj_cache::CacheKeyError::UnknownIncompatibleFeatures {
            features,
        }) = &err
        {
            assert_eq!(features, &vec!["future.backend.protocol.v2".to_owned()]);
        }
    }

    #[test]
    fn hardened_mode_allowlists_unknown_features_for_auditable_progress() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Hardened,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["future.backend.protocol.v2".to_owned()],
        })
        .expect("hardened mode should permit allowlisted unknown features");

        assert_eq!(response.outputs, vec![Value::scalar_i64(6)]);
        assert!(response.cache_key.starts_with("fjx-"));
        assert_eq!(response.evidence_ledger.len(), 1);
    }

    #[test]
    fn test_dispatch_test_log_schema_contract() {
        let fixture_id = fj_test_utils::fixture_id_from_json(&("dispatch", "transform-order"))
            .expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_dispatch_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ── Effect token tests ─────────────────────────────────────

    #[test]
    fn effect_context_preserves_observation_sequence() {
        use super::EffectContext;
        let mut ctx = EffectContext::new();
        let t1 = ctx.thread_token("jit");
        let t2 = ctx.thread_token("jit");
        let t3 = ctx.thread_token("grad");

        assert_eq!(t1.sequence_number, 0);
        assert_eq!(t2.sequence_number, 1);
        assert_eq!(t3.sequence_number, 2);
        assert_eq!(ctx.effect_count(), 3);

        let tokens = ctx.finalize();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].effect_name, "jit");
        assert_eq!(tokens[1].effect_name, "jit");
        assert_eq!(tokens[2].effect_name, "grad");
    }

    #[test]
    fn effect_threading_walks_jaxpr_and_subjaxprs_in_order() {
        use super::{EffectContext, thread_jaxpr_effect_tokens};

        let mut root = build_program(ProgramSpec::Add2);
        root.effects = vec!["Print".to_owned()];
        root.equations[0].effects = vec!["RngConsume".to_owned()];

        let mut sub = build_program(ProgramSpec::AddOne);
        sub.effects = vec!["SubTrace".to_owned()];
        sub.equations[0].effects = vec!["SubPrint".to_owned()];
        root.equations[0].sub_jaxprs.push(sub);

        let mut ctx = EffectContext::new();
        thread_jaxpr_effect_tokens(&mut ctx, &root);
        let tokens = ctx.finalize();
        let names = tokens
            .iter()
            .map(|token| token.effect_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["Print", "RngConsume", "SubTrace", "SubPrint"]);
    }

    #[test]
    fn dispatch_effect_signals_reflect_jaxpr_effects() {
        let mut ttl = ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]);
        ttl.root_jaxpr.effects = vec!["Print".to_owned()];
        ttl.root_jaxpr.equations[0].effects = vec!["RngConsume".to_owned()];

        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl,
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch should succeed");

        // Evidence ledger should include count + trace for effect token observations.
        assert_eq!(response.evidence_ledger.len(), 1);
        let entry = &response.evidence_ledger.entries()[0];
        assert_eq!(entry.signals.len(), 6);

        let count_signal = entry
            .signals
            .iter()
            .find(|s| s.signal_name == "effect_token_count")
            .expect("should have effect_token_count signal");
        assert_eq!(count_signal.detail, "effect_tokens=2");

        let trace_signal = entry
            .signals
            .iter()
            .find(|s| s.signal_name == "effect_token_trace")
            .expect("should have effect_token_trace signal");
        assert_eq!(trace_signal.detail, "effect_tokens=[0:Print,1:RngConsume]");
    }

    #[test]
    fn dispatch_with_no_effects_has_zero_effect_tokens() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch should succeed");

        let entry = &response.evidence_ledger.entries()[0];
        let count_signal = entry
            .signals
            .iter()
            .find(|s| s.signal_name == "effect_token_count")
            .expect("should have effect_token_count signal");
        assert_eq!(count_signal.detail, "effect_tokens=0");
    }

    #[test]
    fn dispatch_cache_hit_miss_determinism() {
        // Two identical requests should produce identical cache keys
        let make_request = || DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        };

        let r1 = dispatch(make_request()).expect("dispatch 1");
        let r2 = dispatch(make_request()).expect("dispatch 2");
        assert_eq!(r1.cache_key, r2.cache_key, "same request = same cache key");
        assert_eq!(r1.outputs, r2.outputs, "same request = same outputs");

        // Different program should produce different cache key
        let r3 = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Jit]),
            args: vec![Value::scalar_f64(2.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch 3");
        assert_ne!(
            r1.cache_key, r3.cache_key,
            "different program = different key"
        );
    }

    #[test]
    fn prepared_metadata_yields_identical_response() {
        // The precomputed-metadata fast path must be byte-for-byte isomorphic to
        // the recompute path: same cache key, outputs, and evidence ledger.
        let led = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let mut compile_options = BTreeMap::new();
        compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
        let args = vec![Value::scalar_f64(3.0)];
        let features: Vec<String> = vec![];

        let recompute = dispatch_ref(DispatchRequestRef {
            mode: CompatibilityMode::Strict,
            root_jaxpr: &led.root_jaxpr,
            transform_stack: &led.transform_stack,
            transform_evidence: &led.transform_evidence,
            args: args.clone(),
            backend: "cpu",
            compile_options: compile_options.clone(),
            custom_hook: None,
            unknown_incompatible_features: &features,
            prepared: None,
        })
        .expect("recompute dispatch");

        let meta = prepare_dispatch_meta(
            CompatibilityMode::Strict,
            &led.root_jaxpr,
            &led.transform_stack,
            &led.transform_evidence,
            "cpu",
            &compile_options,
            None,
            &features,
        )
        .expect("prepare meta");

        let prepared = dispatch_ref(DispatchRequestRef {
            mode: CompatibilityMode::Strict,
            root_jaxpr: &led.root_jaxpr,
            transform_stack: &led.transform_stack,
            transform_evidence: &led.transform_evidence,
            args,
            backend: "cpu",
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: &features,
            prepared: Some(&meta),
        })
        .expect("prepared dispatch");

        assert_eq!(
            recompute, prepared,
            "prepared metadata must produce an identical DispatchResponse"
        );
    }

    #[test]
    fn prepared_metadata_rejects_mismatched_static_inputs() {
        let led = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let mut compile_options = BTreeMap::new();
        compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
        let features: Vec<String> = vec![];

        let meta = prepare_dispatch_meta(
            CompatibilityMode::Strict,
            &led.root_jaxpr,
            &led.transform_stack,
            &led.transform_evidence,
            "cpu",
            &compile_options,
            None,
            &features,
        )
        .expect("prepare meta");

        let err = dispatch_ref(DispatchRequestRef {
            mode: CompatibilityMode::Strict,
            root_jaxpr: &led.root_jaxpr,
            transform_stack: &led.transform_stack,
            transform_evidence: &led.transform_evidence,
            args: vec![Value::scalar_f64(3.0)],
            backend: "gpu",
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: &features,
            prepared: Some(&meta),
        })
        .expect_err("stale prepared metadata must be rejected");

        assert!(
            err.to_string()
                .contains("prepared dispatch metadata does not match request static inputs"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn prepared_metadata_does_not_bypass_strict_unknown_features() {
        let led = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let mut compile_options = BTreeMap::new();
        compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
        let features: Vec<String> = vec![];
        let unknown_features = vec!["experimental-transform".to_owned()];

        let meta = prepare_dispatch_meta(
            CompatibilityMode::Strict,
            &led.root_jaxpr,
            &led.transform_stack,
            &led.transform_evidence,
            "cpu",
            &compile_options,
            None,
            &features,
        )
        .expect("prepare meta");

        let err = dispatch_ref(DispatchRequestRef {
            mode: CompatibilityMode::Strict,
            root_jaxpr: &led.root_jaxpr,
            transform_stack: &led.transform_stack,
            transform_evidence: &led.transform_evidence,
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu",
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: &unknown_features,
            prepared: Some(&meta),
        })
        .expect_err("strict mode must reject unknown incompatible features");

        assert!(matches!(
            err,
            DispatchError::Cache(fj_cache::CacheKeyError::UnknownIncompatibleFeatures { .. })
        ));
    }

    #[test]
    fn dispatch_value_and_grad_mode_changes_cache_key() {
        let make_request = |value_and_grad: bool| {
            let mut compile_options = BTreeMap::new();
            if value_and_grad {
                compile_options.insert("value_and_grad".to_owned(), "true".to_owned());
            }
            DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
                args: vec![Value::scalar_f64(3.0)],
                backend: "cpu".to_owned(),
                compile_options,
                custom_hook: None,
                unknown_incompatible_features: vec![],
            }
        };

        let plain_grad = dispatch(make_request(false)).expect("plain grad dispatch");
        let value_and_grad = dispatch(make_request(true)).expect("value_and_grad dispatch");

        assert_ne!(
            plain_grad.cache_key, value_and_grad.cache_key,
            "value_and_grad mode must use a distinct cache identity from plain grad"
        );
    }

    #[test]
    fn dispatch_unknown_backend_falls_back_to_cpu_execution() {
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: "quantum".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("unknown backend should fall back to cpu backend");

        assert_eq!(response.outputs, vec![Value::scalar_i64(6)]);
    }

    #[test]
    fn dispatch_backend_name_still_changes_cache_key_under_fallback() {
        let make_request = |backend: &str| DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
            args: vec![Value::scalar_i64(2), Value::scalar_i64(4)],
            backend: backend.to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        };

        let cpu = dispatch(make_request("cpu")).expect("cpu dispatch should succeed");
        let fallback = dispatch(make_request("quantum")).expect("fallback dispatch should succeed");

        assert_eq!(cpu.outputs, fallback.outputs);
        assert_ne!(
            cpu.cache_key, fallback.cache_key,
            "requested backend remains part of cache identity even when runtime execution falls back"
        );
    }

    // ── Calibrated posterior tests ────────────────────────────

    #[test]
    fn calibrated_posterior_falls_back_to_heuristic_without_conformal() {
        let ttl = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let result = super::calibrated_posterior_abandoned(&ttl, None);
        let heuristic = super::heuristic_posterior_abandoned(&ttl);
        assert!(
            (result - heuristic).abs() < 1e-12,
            "without conformal predictor, should return heuristic: {result} vs {heuristic}"
        );
    }

    #[test]
    fn calibrated_posterior_falls_back_when_uncalibrated() {
        use fj_ledger::ConformalPredictor;
        let ttl = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let cp = ConformalPredictor::new(0.9, 10); // needs 10 scores, has 0
        let result = super::calibrated_posterior_abandoned(&ttl, Some(&cp));
        let heuristic = super::heuristic_posterior_abandoned(&ttl);
        assert!(
            (result - heuristic).abs() < 1e-12,
            "uncalibrated predictor should fall back to heuristic"
        );
    }

    #[test]
    fn calibrated_posterior_uses_conformal_when_calibrated() {
        use fj_ledger::ConformalPredictor;
        let ttl = ledger(ProgramSpec::Square, &[Transform::Grad]);
        let mut cp = ConformalPredictor::new(0.9, 5);
        for score in &[0.01, 0.02, 0.03, 0.04, 0.05] {
            cp.observe(*score);
        }
        assert!(cp.is_calibrated());

        let result = super::calibrated_posterior_abandoned(&ttl, Some(&cp));
        let heuristic = super::heuristic_posterior_abandoned(&ttl);
        // Result should equal the conformal point estimate (which equals heuristic)
        // because calibrated_posterior returns point = heuristic
        assert!(
            (result - heuristic).abs() < 1e-12,
            "calibrated conformal point estimate should equal heuristic input"
        );
    }

    #[test]
    fn heuristic_posterior_increases_with_transform_depth() {
        let shallow = ledger(ProgramSpec::Add2, &[Transform::Jit]);
        let deep = ledger(
            ProgramSpec::Add2,
            &[Transform::Jit, Transform::Grad, Transform::Vmap],
        );
        let h_shallow = super::heuristic_posterior_abandoned(&shallow);
        let h_deep = super::heuristic_posterior_abandoned(&deep);
        assert!(
            h_deep > h_shallow,
            "deeper transform stack should have higher abandoned posterior: {h_deep} vs {h_shallow}"
        );
    }

    #[test]
    fn heuristic_posterior_is_bounded() {
        let minimal = ledger(ProgramSpec::AddOne, &[]);
        let result = super::heuristic_posterior_abandoned(&minimal);
        assert!(result >= 0.05, "posterior should be >= 0.05, got {result}");
        assert!(result <= 0.95, "posterior should be <= 0.95, got {result}");
    }

    // ══════════════════════════════════════════════════════════════
    // Higher-rank tensor tests (3D+)
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn dispatch_vmap_add_one_rank3_tensor() {
        // 3D tensor [2, 3, 2] — vmap should process each leading-axis slice
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 3, 2],
                },
                (1..=12).map(fj_core::Literal::I64).collect(),
            )
            .expect("3d tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap rank3 should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap rank3 output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 3, 2]
            }
        );
        let as_i64: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64"))
            .collect();
        let expected: Vec<i64> = (2..=13).collect();
        assert_eq!(as_i64, expected);
    }

    #[test]
    fn dispatch_triple_vmap_add_one_rank3_tensor() {
        // 3D tensor with triple vmap (vmap(vmap(vmap(f))))
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 2, 3],
                },
                (0..12).map(fj_core::Literal::I64).collect(),
            )
            .expect("3d tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::AddOne,
                &[Transform::Vmap, Transform::Vmap, Transform::Vmap],
            ),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch triple vmap should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("triple vmap output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 2, 3]
            }
        );
        let as_i64: Vec<i64> = output
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64"))
            .collect();
        let expected: Vec<i64> = (1..=12).collect();
        assert_eq!(as_i64, expected);
    }

    #[test]
    fn dispatch_triple_vmap_grad_square_rank3_tensor() {
        let input_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 2, 2],
                },
                input_vals
                    .iter()
                    .copied()
                    .map(fj_core::Literal::from_f64)
                    .collect(),
            )
            .expect("3d tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::Square,
                &[
                    Transform::Vmap,
                    Transform::Vmap,
                    Transform::Vmap,
                    Transform::Grad,
                ],
            ),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch triple vmap(grad(square)) should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("triple vmap grad output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 2, 2]
            }
        );
        let grads = output.to_f64_vec().expect("output should be f64 tensor");
        let expected: Vec<f64> = input_vals.iter().map(|x| 2.0 * x).collect();
        assert_eq!(grads.len(), expected.len());
        for (idx, (actual, expected)) in grads.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-3,
                "index {idx}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn dispatch_grad_square_rank3_tensor_rejects_non_scalar_input() {
        let tensor_3d = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 2, 2],
                },
                (1..=8)
                    .map(|v| fj_core::Literal::from_f64(v as f64))
                    .collect(),
            )
            .expect("3d tensor should build"),
        );
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
            args: vec![tensor_3d],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("grad(square) on rank3 input should fail");

        let err_msg = err.to_string();
        assert!(
            err_msg.contains("scalar"),
            "rank3 grad failure should mention scalar requirement, got: {err_msg}"
        );
    }

    #[test]
    fn dispatch_grad_sin_with_jit() {
        // grad(jit(sin(x))) = cos(x) at x=0 => 1.0
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::SinX, &[Transform::Jit, Transform::Grad]),
            args: vec![Value::scalar_f64(0.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch grad(jit(sin)) should succeed");

        let grad_val = response.outputs[0]
            .as_f64_scalar()
            .expect("grad should be scalar f64");
        assert!(
            (grad_val - 1.0).abs() < 1e-10,
            "grad(sin)(0) should be cos(0) = 1.0, got {grad_val}"
        );
    }

    #[test]
    fn dispatch_vmap_sin_f64_vector() {
        // vmap(sin)([0, pi/2, pi]) = [0, 1, ~0]
        use std::f64::consts::PI;
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::SinX, &[Transform::Vmap]),
            args: vec![Value::vector_f64(&[0.0, PI / 2.0, PI]).unwrap()],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("dispatch vmap(sin) should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("vmap output should be tensor");
        let vals: Vec<f64> = output.to_f64_vec().expect("should convert to f64");
        assert!(vals[0].abs() < 1e-10, "sin(0) should be 0");
        assert!((vals[1] - 1.0).abs() < 1e-10, "sin(pi/2) should be 1");
        assert!(vals[2].abs() < 1e-10, "sin(pi) should be ~0");
    }

    // ── VMAP tensor output tests (bd-22lm) ───────────────────────

    #[test]
    fn test_vmap_tensor_output_rank1() {
        // vmap(AddOne) over a [3, 4] matrix: inner function receives rank-1 vectors,
        // returns rank-1 vectors, stacked back to [3, 4].
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 4] },
                (1..=12).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap with tensor output should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be tensor");
        assert_eq!(output.shape, Shape { dims: vec![3, 4] });
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let expected: Vec<i64> = (2..=13).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn test_vmap_tensor_output_rank2() {
        // vmap(AddOne) over a [2, 3, 4] rank-3 tensor: inner receives rank-2 matrices,
        // returns rank-2 matrices, stacked back to [2, 3, 4].
        let tensor = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![2, 3, 4],
                },
                (0..24).map(fj_core::Literal::I64).collect(),
            )
            .expect("rank-3 tensor should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![tensor],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap with rank-2 inner output should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![2, 3, 4]
            }
        );
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let expected: Vec<i64> = (1..=24).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn test_vmap_multi_output() {
        // vmap(AddOneMulTwo) over [1, 2, 3]: inner returns (x+1, x*2) per element.
        // Output: two tensors, each of shape [3].
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector should build");
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOneMulTwo, &[Transform::Vmap]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap multi-output should succeed");

        assert_eq!(response.outputs.len(), 2, "should have two outputs");

        let out0 = response.outputs[0]
            .as_tensor()
            .expect("first output should be tensor");
        let out1 = response.outputs[1]
            .as_tensor()
            .expect("second output should be tensor");

        let vals0: Vec<i64> = out0.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        let vals1: Vec<i64> = out1.elements.iter().map(|l| l.as_i64().unwrap()).collect();

        assert_eq!(vals0, vec![2, 3, 4], "x+1 for [1,2,3]");
        assert_eq!(vals1, vec![2, 4, 6], "x*2 for [1,2,3]");
    }

    #[test]
    fn test_vmap_multi_output_loop_and_stack() {
        // Same as above but force the loop-and-stack path by adding a Jit tail transform.
        let input = Value::vector_i64(&[10, 20, 30]).expect("vector should build");
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::AddOneMulTwo,
                &[Transform::Vmap, Transform::Jit],
            ),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap multi-output loop-and-stack should succeed");

        assert_eq!(response.outputs.len(), 2, "should have two outputs");

        let vals0: Vec<i64> = response.outputs[0]
            .as_tensor()
            .expect("first output tensor")
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        let vals1: Vec<i64> = response.outputs[1]
            .as_tensor()
            .expect("second output tensor")
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();

        assert_eq!(vals0, vec![11, 21, 31], "x+1 for [10,20,30]");
        assert_eq!(vals1, vec![20, 40, 60], "x*2 for [10,20,30]");
    }

    #[test]
    fn test_vmap_output_shape_batch_prepend() {
        // Verify batch dimension is correctly prepended to inner output shape.
        // vmap(AddOne) on [5, 3] matrix: inner returns [3] vectors → stacked to [5, 3].
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5, 3] },
                (0..15)
                    .map(|i| fj_core::Literal::from_f64(i as f64))
                    .collect(),
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap shape prepend should succeed");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be tensor");
        // Output shape should be [5, 3] = [batch_size, ...inner_output_shape]
        assert_eq!(
            output.shape,
            Shape { dims: vec![5, 3] },
            "batch dim (5) prepended to inner output shape (3)"
        );
    }

    #[test]
    fn test_vmap_identity_preserves_shape() {
        // vmap(identity) should return the same tensor as the input.
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 4] },
                (1..=12).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix should build"),
        );
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Identity, &[Transform::Vmap]),
            args: vec![matrix.clone()],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(identity) should succeed");

        assert_eq!(response.outputs.len(), 1);
        let output = &response.outputs[0];
        let out_tensor = output.as_tensor().expect("output should be tensor");
        let in_tensor = matrix.as_tensor().unwrap();
        assert_eq!(
            out_tensor.shape, in_tensor.shape,
            "shape should be preserved"
        );
        assert_eq!(
            out_tensor.elements, in_tensor.elements,
            "elements should be identical"
        );
    }

    #[test]
    fn test_vmap_scalar_output_still_works() {
        // Regression: existing scalar-output vmap behavior should be preserved.
        let input = Value::vector_f64(&[1.0, 4.0, 9.0]).expect("vector should build");
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Vmap]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(square) with scalar outputs should still work");

        let output = response.outputs[0]
            .as_tensor()
            .expect("output should be stacked tensor");
        let vals = output.to_f64_vec().expect("output should be f64");
        assert!((vals[0] - 1.0).abs() < 1e-10, "1^2 = 1");
        assert!((vals[1] - 16.0).abs() < 1e-10, "4^2 = 16");
        assert!((vals[2] - 81.0).abs() < 1e-10, "9^2 = 81");
    }

    // ── in_axes / out_axes tests (bd-3edr) ───────────────────────

    #[test]
    fn test_in_axes_0_default() {
        // in_axes=0 (default) batches along leading dim — same as no compile_options
        let input = Value::vector_i64(&[10, 20, 30]).expect("vector");
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "0".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap in_axes=0 should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(vals, vec![11, 21, 31]);
    }

    #[test]
    fn test_in_axes_none_broadcasts() {
        // in_axes=None means input is not batched — broadcast to all batch elements.
        // We use Add2 with two args: first batched (axis 0), second not batched.
        let batched = Value::vector_i64(&[1, 2, 3]).expect("vector");
        let scalar = Value::scalar_i64(100);
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "0,none".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Vmap]),
            args: vec![batched, scalar],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap with in_axes=none broadcast should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(vals, vec![101, 102, 103]);
    }

    #[test]
    fn test_in_axes_1_second_dim_batch_trace() {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    fj_core::Literal::I64(1),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(3),
                    fj_core::Literal::I64(4),
                    fj_core::Literal::I64(5),
                    fj_core::Literal::I64(6),
                ],
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "1".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap in_axes=1 should succeed through BatchTrace");

        let output = response.outputs[0].as_tensor().expect("tensor");
        assert_eq!(output.shape, Shape { dims: vec![3, 2] });
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(vals, vec![2, 5, 3, 6, 4, 7]);
    }

    #[test]
    fn test_in_axes_1_i64_add_one_bench_shape_golden_sha256() {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![64, 128],
                },
                (0_i64..(64 * 128)).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "1".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap in_axes=1 should succeed through BatchTrace");

        let output = response.outputs[0].as_tensor().expect("tensor");
        assert_eq!(
            output.shape,
            Shape {
                dims: vec![128, 64],
            }
        );
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|literal| literal.as_i64().expect("expected i64 element"))
            .collect();
        assert_eq!(&vals[..8], &[1, 129, 257, 385, 513, 641, 769, 897]);
        assert_eq!(
            &vals[vals.len() - 8..],
            &[7296, 7424, 7552, 7680, 7808, 7936, 8064, 8192]
        );
        let digest =
            fj_test_utils::fixture_id_from_json(&(output.shape.dims.clone(), vals)).unwrap();
        assert_eq!(
            digest,
            "29d48dbffb8ba3bc95e29416bbe470cd18432ffd471b8ec7ef068c3b7a61b3ef"
        );
    }

    #[test]
    fn test_in_axes_1_second_dim() {
        // in_axes=1 batches along second dimension.
        // Input shape [2, 3]: batch along axis 1 gives batch_size=3, each slice is [2].
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    fj_core::Literal::I64(1),
                    fj_core::Literal::I64(2),
                    fj_core::Literal::I64(3),
                    fj_core::Literal::I64(4),
                    fj_core::Literal::I64(5),
                    fj_core::Literal::I64(6),
                ],
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "1".to_owned());
        // Force loop-and-stack path by adding Jit tail
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap in_axes=1 should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        // Batch size 3, each inner result is [2], stacked along axis 0 → [3, 2]
        assert_eq!(output.shape, Shape { dims: vec![3, 2] });
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        // Column 0: [1, 4]+1 = [2, 5], Column 1: [2, 5]+1 = [3, 6], Column 2: [3, 6]+1 = [4, 7]
        assert_eq!(vals, vec![2, 5, 3, 6, 4, 7]);
    }

    #[test]
    fn test_slice_along_axis_rank3_preserves_row_major_order() {
        let tensor = TensorValue::new(
            DType::I64,
            Shape {
                dims: vec![2, 3, 2],
            },
            (0_i64..12).map(fj_core::Literal::I64).collect(),
        )
        .expect("rank-3 tensor");

        let slice = super::slice_along_axis(&tensor, 1, 2)
            .expect("axis-1 slice should succeed")
            .as_tensor()
            .expect("rank-3 axis-1 slice should be rank-2")
            .clone();

        assert_eq!(slice.shape, Shape { dims: vec![2, 2] });
        let vals: Vec<i64> = slice
            .elements
            .iter()
            .map(|lit| lit.as_i64().expect("i64 literal"))
            .collect();
        assert_eq!(vals, vec![4, 5, 10, 11]);
    }

    #[test]
    fn test_in_axes_tuple_mixed() {
        // in_axes=(0, none) for two-arg Add2: first arg batched, second broadcast
        let batched = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let broadcast = Value::scalar_f64(10.0);
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "0,none".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Vmap]),
            args: vec![batched, broadcast],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap mixed in_axes should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        let vals = output.to_f64_vec().expect("f64");
        assert!((vals[0] - 11.0).abs() < 1e-10);
        assert!((vals[1] - 12.0).abs() < 1e-10);
        assert!((vals[2] - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_out_axes_0_default() {
        // out_axes=0 (default) places batch dim at position 0 — same as normal
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector");
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "0".to_owned());
        // Force loop-and-stack by adding Jit tail
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap out_axes=0 should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(vals, vec![2, 3, 4]);
    }

    #[test]
    fn test_out_axes_1_trailing() {
        // out_axes=1: batch dim moved from position 0 to position 1.
        // Use a matrix input so the stacked output rank is 2 and axis 1 is valid:
        // Input [3, 2], in_axes=0 → inner gets [2], outputs [2], stacked → [3, 2]
        // out_axes=1 → transpose to [2, 3]
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
                (1..=6).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "1".to_owned());
        // Force loop-and-stack
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap out_axes=1 should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        // Original stacked: [3, 2] with batch dim at 0
        // out_axes=1 → transpose [0,1] → [1,0] → shape [2, 3]
        assert_eq!(output.shape, Shape { dims: vec![2, 3] });
        // Original elements row-major: [2,3, 4,5, 6,7]
        // Transposed [2,3] row-major: col0=[2,4,6], col1=[3,5,7]
        let vals: Vec<i64> = output
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(vals, vec![2, 4, 6, 3, 5, 7]);
    }

    #[test]
    fn test_in_axes_negative_index() {
        // in_axes=-1 on a [2, 3] matrix means batch along last axis (axis 1), batch_size=3
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                (1..=6).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "-1".to_owned());
        // Force loop-and-stack
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap in_axes=-1 should succeed");

        let output = response.outputs[0].as_tensor().expect("tensor");
        // in_axes=-1 on [2,3] resolves to axis 1, batch_size=3
        // Each slice is column [2], stacked → [3, 2]
        assert_eq!(output.shape, Shape { dims: vec![3, 2] });
    }

    #[test]
    fn test_in_axes_error_out_of_bounds() {
        // in_axes=5 on a rank-2 tensor should error
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                (1..=6).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "5".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("vmap with out-of-bounds in_axes should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("out of bounds"),
            "error should mention out of bounds, got: {msg}"
        );
    }

    #[test]
    fn test_out_axes_error_positive_out_of_bounds() {
        let input = Value::vector_i64(&[1, 2, 3]).expect("vector");
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "1".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![input],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("rank-1 outputs should reject out_axes=1");

        assert!(matches!(
            err,
            DispatchError::TransformExecution(TransformExecutionError::VmapAxesOutOfBounds {
                axis: 1,
                ndim: 1,
            })
        ));
    }

    #[test]
    fn test_out_axes_error_negative_out_of_bounds() {
        let matrix = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
                (1..=6).map(fj_core::Literal::I64).collect(),
            )
            .expect("matrix"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "-3".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![matrix],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("rank-2 outputs should reject out_axes=-3");

        assert!(matches!(
            err,
            DispatchError::TransformExecution(TransformExecutionError::VmapAxesOutOfBounds {
                axis: -3,
                ndim: 2,
            })
        ));
    }

    #[test]
    fn test_in_axes_error_mismatched_batch_size() {
        // Two batched args with different batch sizes along their respective axes
        let a = Value::vector_i64(&[1, 2, 3]).expect("vec3");
        let b = Value::vector_i64(&[1, 2]).expect("vec2");
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "0,0".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Add2, &[Transform::Vmap]),
            args: vec![a, b],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("mismatched batch sizes should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("mismatch"),
            "error should mention mismatch, got: {msg}"
        );
    }

    #[test]
    fn test_parse_vmap_in_axes_uniform() {
        let opts = BTreeMap::new();
        let axes = super::parse_vmap_in_axes(&opts, 3).expect("default axes");
        assert_eq!(axes, vec![super::AxisSpec::Batched(0); 3]);
    }

    #[test]
    fn test_parse_vmap_in_axes_per_arg() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "0,none,1".to_owned());
        let axes = super::parse_vmap_in_axes(&opts, 3).expect("per-arg axes");
        assert_eq!(
            axes,
            vec![
                super::AxisSpec::Batched(0),
                super::AxisSpec::NotBatched,
                super::AxisSpec::Batched(1),
            ]
        );
    }

    #[test]
    fn test_parse_vmap_out_axes_broadcast_single() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "1".to_owned());
        let axes = super::parse_vmap_out_axes(&opts, 3).expect("broadcast out axes");
        assert_eq!(axes, vec![super::AxisSpec::Batched(1); 3]);
    }

    #[test]
    fn test_parse_vmap_in_axes_rejects_invalid_token() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "oops".to_owned());
        let err = super::parse_vmap_in_axes(&opts, 1).expect_err("invalid axis token");
        let msg = err.to_string();
        assert!(msg.contains("vmap_in_axes"), "error should identify option");
        assert!(msg.contains("oops"), "error should include invalid token");
    }

    #[test]
    fn test_parse_vmap_out_axes_rejects_invalid_token() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "bad".to_owned());
        let err = super::parse_vmap_out_axes(&opts, 1).expect_err("invalid axis token");
        let msg = err.to_string();
        assert!(
            msg.contains("vmap_out_axes"),
            "error should identify option"
        );
        assert!(msg.contains("bad"), "error should include invalid token");
    }

    #[test]
    fn test_dispatch_rejects_invalid_vmap_in_axes_option() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_in_axes".to_owned(), "bad".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![Value::vector_i64(&[1, 2, 3]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("invalid axis token should fail dispatch");

        let msg = err.to_string();
        assert!(msg.contains("vmap_in_axes"), "error should identify option");
        assert!(msg.contains("bad"), "error should include invalid token");
    }

    #[test]
    fn test_out_axes_error_mismatched_output_arity() {
        let batched_qr_input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 3, 2],
                },
                [
                    1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, // batch element 0
                    2.0, 0.0, 0.0, 2.0, 2.0, 2.0, // batch element 1
                ]
                .into_iter()
                .map(|x| fj_core::Literal::F64Bits(x.to_bits()))
                .collect(),
            )
            .expect("batched qr input"),
        );
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "0,0,0".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::LaxQr, &[Transform::Vmap, Transform::Jit]),
            args: vec![batched_qr_input],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("mismatched out_axes length should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("length mismatch"),
            "error should mention length mismatch, got: {msg}"
        );
        assert!(
            msg.contains("expected 2"),
            "error should report expected arity"
        );
        assert!(msg.contains("got 3"), "error should report provided arity");
    }

    #[test]
    fn test_out_axes_none_returns_unmapped_constant_output() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![
                    Atom::Lit(fj_core::Literal::I64(3)),
                    Atom::Lit(fj_core::Literal::I64(4))
                ],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );
        let mut ledger = TraceTransformLedger::new(jaxpr);
        ledger.push_transform(Transform::Vmap, "test-vmap".to_owned());
        ledger.push_transform(Transform::Jit, "test-jit".to_owned());

        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "none".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger,
            args: vec![Value::vector_i64(&[10, 20, 30]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("constant unmapped output should succeed");

        assert_eq!(response.outputs, vec![Value::scalar_i64(7)]);
    }

    #[test]
    fn test_out_axes_none_tail_free_constant_uses_batch_trace() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![
                    Atom::Lit(fj_core::Literal::I64(3)),
                    Atom::Lit(fj_core::Literal::I64(4))
                ],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );
        let mut ledger = TraceTransformLedger::new(jaxpr);
        ledger.push_transform(Transform::Vmap, "test-vmap".to_owned());

        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "none".to_owned());
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger,
            args: vec![Value::vector_i64(&[10, 20, 30]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("tail-free constant unmapped output should succeed");

        assert_eq!(response.outputs, vec![Value::scalar_i64(7)]);
    }

    #[test]
    fn test_out_axes_none_rejects_varying_outputs() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "none".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![Value::vector_i64(&[1, 2, 3]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("varying unmapped output should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("out_axes=none"),
            "error should mention unmapped out_axes"
        );
        assert!(
            msg.contains("identical"),
            "error should explain equality requirement"
        );
    }

    #[test]
    fn test_out_axes_none_tail_free_rejects_varying_outputs() {
        let mut opts = BTreeMap::new();
        opts.insert("vmap_out_axes".to_owned(), "none".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
            args: vec![Value::vector_i64(&[1, 2, 3]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("tail-free varying unmapped output should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("out_axes=none"),
            "error should mention unmapped out_axes"
        );
        assert!(
            msg.contains("identical"),
            "error should explain equality requirement"
        );
    }

    // ── Depth-3+ transform composition tests (frankenjax-yxv) ──────

    #[test]
    fn dispatch_jit_grad_grad_square() {
        // jit(grad(grad(x^2))) at x=3: d²/dx²(x²)=2
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::Square,
                &[Transform::Grad, Transform::Grad, Transform::Jit],
            ),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("jit(grad(grad(x^2))) should succeed");

        let d2 = response.outputs[0]
            .as_f64_scalar()
            .expect("should be scalar");
        assert!(
            (d2 - 2.0).abs() < 1e-2,
            "d²/dx²(x²) should be ≈2.0, got {d2}"
        );
    }

    #[test]
    fn dispatch_grad_grad_sin() {
        // grad(grad(sin))(x) = -sin(x); at x=1.0
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::SinX, &[Transform::Grad, Transform::Grad]),
            args: vec![Value::scalar_f64(1.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("grad(grad(sin)) should succeed");

        let d2 = response.outputs[0]
            .as_f64_scalar()
            .expect("should be scalar");
        let expected = -(1.0_f64.sin());
        assert!(
            (d2 - expected).abs() < 1e-4,
            "d²/dx²(sin(x)) at x=1 should be {expected}, got {d2}"
        );
    }

    #[test]
    fn dispatch_vmap_grad_requires_scalar_inputs() {
        // vmap(grad(x^2)) with vector input should fail because grad needs scalar inputs
        let result = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Vmap]),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        });
        // Grad requires scalar inputs — this composition through dispatch correctly rejects
        assert!(
            result.is_err(),
            "vmap(grad(f)) with vector input should fail through dispatch"
        );
    }

    #[test]
    fn dispatch_grad_vmap_add_one() {
        // grad(vmap(f)) where f(x) = x+1 (constant derivative)
        // grad of vmap(add_one) at each element should be 1.0
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Grad]),
            args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        });
        // This may succeed or fail depending on how grad handles batched outputs.
        // The key assertion: it shouldn't panic or produce garbage.
        match response {
            Ok(r) => {
                // If it succeeds, outputs should be finite
                for (i, out) in r.outputs.iter().enumerate() {
                    if let Some(v) = out.as_f64_scalar() {
                        assert!(v.is_finite(), "output[{i}] should be finite, got {v}");
                    }
                }
            }
            Err(e) => {
                // Some composition orders may legitimately fail
                let msg = e.to_string();
                assert!(!msg.is_empty(), "error should have a descriptive message");
            }
        }
    }

    #[test]
    fn dispatch_jit_vmap_add_one() {
        // jit(vmap(f)) — JIT wrapping a vmapped function
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap, Transform::Jit]),
            args: vec![Value::vector_i64(&[10, 20, 30]).expect("vector")],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("jit(vmap(add_one)) should succeed");

        let t = response.outputs[0].as_tensor().expect("tensor result");
        let vals = t.to_i64_vec().expect("i64 vec");
        assert_eq!(vals, vec![11, 21, 31]);
    }

    #[test]
    fn dispatch_composition_order_affects_cache_key() {
        // Verify that [Grad, Jit] and [Jit, Grad] produce different cache keys
        let response_gj = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Jit]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("grad then jit should succeed");

        let response_jg = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]),
            args: vec![Value::scalar_f64(3.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("jit then grad should succeed");

        assert_ne!(
            response_gj.cache_key, response_jg.cache_key,
            "different transform orders must produce different cache keys"
        );
    }

    #[test]
    fn dispatch_grad_grad_grad_square() {
        // Third derivative of x^2 should be 0
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ledger(
                ProgramSpec::Square,
                &[Transform::Grad, Transform::Grad, Transform::Grad],
            ),
            args: vec![Value::scalar_f64(5.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("grad(grad(grad(x^2))) should succeed");

        let d3 = response.outputs[0]
            .as_f64_scalar()
            .expect("should be scalar");
        assert!(d3.abs() < 1e-4, "d³/dx³(x²) should be 0, got {d3}");
    }

    #[test]
    fn dispatch_hardened_mode_grad_grad() {
        // Same as above but in hardened mode — should still work correctly
        let response = dispatch(DispatchRequest {
            mode: CompatibilityMode::Hardened,
            ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Grad]),
            args: vec![Value::scalar_f64(4.0)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("hardened grad(grad(x^2)) should succeed");

        let d2 = response.outputs[0]
            .as_f64_scalar()
            .expect("should be scalar");
        assert!(
            (d2 - 2.0).abs() < 1e-4,
            "hardened d²/dx²(x²) should be 2.0, got {d2}"
        );
    }

    #[test]
    fn rc26_9_symbolic_grad_grad_and_vmap_grad_batch_trace() {
        // 1. f(x) = exp(10.0 * x) at x = 0.1:
        //    f'(x) = 10 * exp(10*x), f''(x) = 100 * exp(10*x), f'''(x) = 1000 * exp(10*x)
        //    Finite differences with eps=1e-6 suffer ~1e-2 (2nd order) and >1.0 (3rd order) error.
        let exp10x = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![
                        Atom::Lit(fj_core::Literal::from_f64(10.0)),
                        Atom::Var(VarId(1))
                    ],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        );

        let mut ttl_exp_d2 = TraceTransformLedger::new(exp10x.clone());
        ttl_exp_d2.push_transform(Transform::Grad, "grad-0");
        ttl_exp_d2.push_transform(Transform::Grad, "grad-1");
        let resp_exp_d2 = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl_exp_d2,
            args: vec![Value::scalar_f64(0.1)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("symbolic grad(grad(exp(10x))) should succeed");
        let exp_d2 = resp_exp_d2.outputs[0].as_f64_scalar().unwrap();
        let expected_exp_d2 = 100.0 * 1.0_f64.exp();
        assert!(
            (exp_d2 - expected_exp_d2).abs() / expected_exp_d2 < 1e-13,
            "exact symbolic d²/dx²(exp(10x)) at 0.1 should be {expected_exp_d2}, got {exp_d2}"
        );

        let mut ttl_exp_d3 = TraceTransformLedger::new(exp10x);
        ttl_exp_d3.push_transform(Transform::Grad, "grad-0");
        ttl_exp_d3.push_transform(Transform::Grad, "grad-1");
        ttl_exp_d3.push_transform(Transform::Grad, "grad-2");
        let resp_exp_d3 = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl_exp_d3,
            args: vec![Value::scalar_f64(0.1)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("symbolic grad(grad(grad(exp(10x)))) should succeed");
        let exp_d3 = resp_exp_d3.outputs[0].as_f64_scalar().unwrap();
        let expected_exp_d3 = 1000.0 * 1.0_f64.exp();
        assert!(
            (exp_d3 - expected_exp_d3).abs() / expected_exp_d3 < 1e-13,
            "exact symbolic d³/dx³(exp(10x)) at 0.1 should be {expected_exp_d3}, got {exp_d3}"
        );

        // 2. f(x) = x^3: at x = 1e5, f(x) = 1e15 where nested finite differences with
        // eps=1e-6 would suffer catastrophic cancellation, whereas symbolic
        // d²/dx²(x^3) = 6*x = 6e5 and d³/dx³(x^3) = 6 are exact to machine precision.
        let cubic = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        );

        let mut ttl_d2 = TraceTransformLedger::new(cubic.clone());
        ttl_d2.push_transform(Transform::Grad, "grad-0");
        ttl_d2.push_transform(Transform::Grad, "grad-1");
        let resp_d2 = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl_d2,
            args: vec![Value::scalar_f64(1e5)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("symbolic grad(grad(x^3)) should succeed");
        let d2 = resp_d2.outputs[0].as_f64_scalar().unwrap();
        assert!(
            (d2 - 6e5).abs() < 1e-9,
            "exact symbolic d²/dx²(x^3) at 1e5 should be 6e5, got {d2}"
        );

        let mut ttl_d3 = TraceTransformLedger::new(cubic.clone());
        ttl_d3.push_transform(Transform::Grad, "grad-0");
        ttl_d3.push_transform(Transform::Grad, "grad-1");
        ttl_d3.push_transform(Transform::Grad, "grad-2");
        let resp_d3 = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl_d3,
            args: vec![Value::scalar_f64(1e5)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("symbolic grad(grad(grad(x^3))) should succeed");
        let d3 = resp_d3.outputs[0].as_f64_scalar().unwrap();
        assert!(
            (d3 - 6.0).abs() < 1e-13,
            "exact symbolic d³/dx³(x^3) should be 6.0, got {d3}"
        );

        // 3. Tensor [Grad, Grad] and [Jit, Grad, Vmap] on f(x) = sum(x^3)
        let cubic_sum = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        );
        let batch_x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(4),
                vec![
                    fj_core::Literal::from_f64(1.0),
                    fj_core::Literal::from_f64(2.0),
                    fj_core::Literal::from_f64(3.0),
                    fj_core::Literal::from_f64(4.0),
                ],
            )
            .unwrap(),
        );
        let mut ttl_tensor_gg = TraceTransformLedger::new(cubic_sum.clone());
        ttl_tensor_gg.push_transform(Transform::Grad, "grad-0");
        ttl_tensor_gg.push_transform(Transform::Grad, "grad-1");
        let resp_tensor_gg = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl_tensor_gg,
            args: vec![batch_x.clone()],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("tensor [Grad, Grad] on sum(x^3) should succeed without NonScalarGradientInput");
        assert_eq!(
            resp_tensor_gg.outputs[0]
                .as_tensor()
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            vec![6.0, 12.0, 18.0, 24.0]
        );

        let mut ttl_jgv = TraceTransformLedger::new(cubic_sum);
        ttl_jgv.push_transform(Transform::Jit, "jit-0");
        ttl_jgv.push_transform(Transform::Grad, "grad-0");
        ttl_jgv.push_transform(Transform::Vmap, "vmap-0");
        let resp_jgv = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl_jgv,
            args: vec![batch_x.clone()],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("[Jit, Grad, Vmap] on reduced Jaxpr should succeed without NonScalarGradientInput");
        assert_eq!(
            resp_jgv.outputs[0]
                .as_tensor()
                .unwrap()
                .to_f64_vec()
                .unwrap(),
            vec![3.0, 12.0, 27.0, 48.0]
        );

        // 4. grad_argnums support: f(x, y) = x * y^2, grad w.r.t. arg 1 = 2 * x * y
        let xy2 = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        );
        let mut ttl_argnums = TraceTransformLedger::new(xy2);
        ttl_argnums.push_transform(Transform::Grad, "grad-0");
        let mut argnums_opts = BTreeMap::new();
        argnums_opts.insert("grad_argnums".to_owned(), "1,0".to_owned());
        let resp_argnums = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl_argnums,
            args: vec![Value::scalar_f64(3.0), Value::scalar_f64(5.0)],
            backend: "cpu".to_owned(),
            compile_options: argnums_opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("grad with grad_argnums=1,0 should succeed");
        assert_eq!(resp_argnums.outputs.len(), 2);
        assert!((resp_argnums.outputs[0].as_f64_scalar().unwrap() - 30.0).abs() < 1e-13);
        assert!((resp_argnums.outputs[1].as_f64_scalar().unwrap() - 25.0).abs() < 1e-13);

        // 5. vmap(grad(x^3)) and vmap(value_and_grad(x^3)) over batch [1.0, 2.0, 3.0, 4.0]
        let mut ttl_vg = TraceTransformLedger::new(cubic);
        ttl_vg.push_transform(Transform::Vmap, "vmap-0");
        ttl_vg.push_transform(Transform::Grad, "grad-0");
        let resp_vg = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl_vg.clone(),
            args: vec![batch_x.clone()],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(grad(x^3)) should succeed");
        let grads = resp_vg.outputs[0]
            .as_tensor()
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(grads, vec![3.0, 12.0, 27.0, 48.0]);

        let mut vag_opts = BTreeMap::new();
        vag_opts.insert("value_and_grad".to_owned(), "true".to_owned());
        let resp_vag = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl_vg,
            args: vec![batch_x],
            backend: "cpu".to_owned(),
            compile_options: vag_opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("vmap(value_and_grad(x^3)) should succeed");
        assert_eq!(resp_vag.outputs.len(), 2);
        let vals = resp_vag.outputs[0]
            .as_tensor()
            .unwrap()
            .to_f64_vec()
            .unwrap();
        let vag_grads = resp_vag.outputs[1]
            .as_tensor()
            .unwrap()
            .to_f64_vec()
            .unwrap();
        assert_eq!(vals, vec![1.0, 8.0, 27.0, 64.0]);
        assert_eq!(vag_grads, vec![3.0, 12.0, 27.0, 48.0]);
    }

    #[test]
    fn e2e_symbolic_higher_order_ad() {
        // Forensic e2e test emitting frankenjax.e2e-forensic-log.v1 proving
        // symbolic higher-order AD and symbolic Hessian parity (< 1e-13 error).
        let x4 = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        );

        let mut ttl = TraceTransformLedger::new(x4.clone());
        ttl.push_transform(Transform::Grad, "grad-0");
        ttl.push_transform(Transform::Grad, "grad-1");
        ttl.push_transform(Transform::Grad, "grad-2");
        let resp = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl,
            args: vec![Value::scalar_f64(2.5)],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("d³/dx³(x^4) should succeed");
        let d3 = resp.outputs[0].as_f64_scalar().unwrap();
        let expected_d3 = 24.0 * 2.5;
        let rel_err = (d3 - expected_d3).abs() / expected_d3;
        assert!(rel_err < 1e-13, "rel_err={rel_err}");

        let hess = fj_ad::symbolic_hessian_jaxpr(&x4, &[Value::scalar_f64(2.5)])
            .expect("symbolic_hessian_jaxpr should succeed");
        let h00 = hess.as_tensor().unwrap().to_f64_vec().unwrap()[0];
        let expected_h00 = 12.0 * 2.5 * 2.5;
        assert!((h00 - expected_h00).abs() / expected_h00 < 1e-13);

        let log = serde_json::json!({
            "schema_version": "frankenjax.e2e-forensic-log.v1",
            "scenario_id": "e2e_symbolic_higher_order_ad",
            "test_id": "fj_dispatch::tests::e2e_symbolic_higher_order_ad",
            "result": "PASS",
            "d3_x4_at_2_5": d3,
            "expected_d3": expected_d3,
            "hessian_x4_at_2_5": h00,
            "expected_hessian": expected_h00,
        });
        assert_eq!(log["schema_version"], "frankenjax.e2e-forensic-log.v1");
    }

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(proptest::test_runner::Config::with_cases(
                fj_test_utils::property_test_case_count()
            ))]

            #[test]
            fn metamorphic_dispatch_deterministic(
                a in -1000i64..1000,
                b in -1000i64..1000
            ) {
                let request = || DispatchRequest {
                    mode: CompatibilityMode::Strict,
                    ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
                    args: vec![Value::scalar_i64(a), Value::scalar_i64(b)],
                    backend: "cpu".to_owned(),
                    compile_options: BTreeMap::new(),
                    custom_hook: None,
                    unknown_incompatible_features: vec![],
                };
                let r1 = dispatch(request()).expect("dispatch 1");
                let r2 = dispatch(request()).expect("dispatch 2");
                prop_assert_eq!(r1.outputs, r2.outputs, "dispatch not deterministic for a={}, b={}", a, b);
            }

            #[test]
            fn metamorphic_jit_transparent(x in -100.0f64..100.0) {
                prop_assume!(x.is_finite());
                let no_jit = dispatch(DispatchRequest {
                    mode: CompatibilityMode::Strict,
                    ledger: ledger(ProgramSpec::Square, &[]),
                    args: vec![Value::scalar_f64(x)],
                    backend: "cpu".to_owned(),
                    compile_options: BTreeMap::new(),
                    custom_hook: None,
                    unknown_incompatible_features: vec![],
                }).expect("no jit dispatch");
                let with_jit = dispatch(DispatchRequest {
                    mode: CompatibilityMode::Strict,
                    ledger: ledger(ProgramSpec::Square, &[Transform::Jit]),
                    args: vec![Value::scalar_f64(x)],
                    backend: "cpu".to_owned(),
                    compile_options: BTreeMap::new(),
                    custom_hook: None,
                    unknown_incompatible_features: vec![],
                }).expect("jit dispatch");
                let no_jit_val = no_jit.outputs[0].as_f64_scalar().unwrap();
                let jit_val = with_jit.outputs[0].as_f64_scalar().unwrap();
                prop_assert!((no_jit_val - jit_val).abs() < 1e-14, "jit not transparent: {} vs {} at x={}", no_jit_val, jit_val, x);
            }

            #[test]
            fn metamorphic_grad_linearity_constant_factor(
                x in 1.0f64..10.0,
                c in 1.0f64..5.0
            ) {
                let grad_x = dispatch(DispatchRequest {
                    mode: CompatibilityMode::Strict,
                    ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
                    args: vec![Value::scalar_f64(x)],
                    backend: "cpu".to_owned(),
                    compile_options: BTreeMap::new(),
                    custom_hook: None,
                    unknown_incompatible_features: vec![],
                }).expect("grad at x");
                let grad_cx = dispatch(DispatchRequest {
                    mode: CompatibilityMode::Strict,
                    ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
                    args: vec![Value::scalar_f64(c * x)],
                    backend: "cpu".to_owned(),
                    compile_options: BTreeMap::new(),
                    custom_hook: None,
                    unknown_incompatible_features: vec![],
                }).expect("grad at c*x");
                let g_x = grad_x.outputs[0].as_f64_scalar().unwrap();
                let g_cx = grad_cx.outputs[0].as_f64_scalar().unwrap();
                let expected = 2.0 * x;
                let expected_cx = 2.0 * c * x;
                prop_assert!((g_x - expected).abs() < 1e-10, "grad(square) at x={}: {} vs {}", x, g_x, expected);
                prop_assert!((g_cx - expected_cx).abs() < 1e-10, "grad(square) at c*x={}: {} vs {}", c*x, g_cx, expected_cx);
            }
        }
    }

    #[test]
    fn e2e_wave_parallel_dispatch_and_fast_canonicalization() {
        crate::clear_dispatch_cache();

        // Build a 2-branch independent Dot DAG with a redundant double-negation on the output:
        // Branch 1: v5 = dot(v1, v2)
        // Branch 2: v6 = dot(v3, v4)
        // Merge:    v7 = add(v5, v6)
        // Canonicalize target: v8 = neg(v7), v9 = neg(v8) -> collapses to v7 under ieee_strict!
        let dag_jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3), VarId(4)],
            vec![],
            vec![VarId(9)],
            vec![
                Equation {
                    primitive: Primitive::Dot,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(5)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Dot,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(4))],
                    outputs: smallvec![VarId(6)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(5)), Atom::Var(VarId(6))],
                    outputs: smallvec![VarId(7)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(7))],
                    outputs: smallvec![VarId(8)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(8))],
                    outputs: smallvec![VarId(9)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        );

        assert!(crate::jaxpr_has_parallel_heavy_wave(&dag_jaxpr));
        let waves = fj_backend_cpu::build_dependency_waves(&dag_jaxpr);
        assert_eq!(waves[0], vec![0, 1]);

        let n = 64usize;
        let make_vec = |seed: f64| {
            let data: Vec<f64> = (0..n).map(|i| seed + (i as f64) * 0.125).collect();
            Value::Tensor(TensorValue::new_f64_values(Shape::vector(n as u32), data).unwrap())
        };
        let args = vec![make_vec(1.0), make_vec(2.0), make_vec(0.5), make_vec(1.5)];

        let mut ttl = TraceTransformLedger::new(dag_jaxpr.clone());
        ttl.push_transform(Transform::Jit, "jit-wave-parallel");
        let resp = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl,
            args: args.clone(),
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("wave-parallel JIT dispatch should succeed");

        let seq = fj_interpreters::eval_jaxpr(&dag_jaxpr, &args).expect("sequential eval");
        assert_eq!(resp.outputs, seq);
    }

    #[test]
    fn e2e_ttl_v2_dispatch_composition_proof_and_tamper_rejection() {
        use fj_core::{
            TransformCompositionError, TransformRewriteWitness, jaxpr_fingerprint_sha256,
        };

        let root = build_program(ProgramSpec::Square);
        let mut ttl = TraceTransformLedger::new(root.clone());
        ttl.push_transform(Transform::Jit, "jit-outer");
        ttl.push_transform(Transform::Vmap, "vmap-mid");
        ttl.push_transform(Transform::Grad, "grad-inner");

        let batch = Value::vector_f64(&[2.0, 3.0, 5.0]).unwrap();
        let resp = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: ttl.clone(),
            args: vec![batch.clone()],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect("triple composition dispatch should succeed");

        let proof = &resp.composition_proof;
        assert_eq!(proof.steps.len(), 3);
        assert_eq!(proof.root_jaxpr_sha256, jaxpr_fingerprint_sha256(&root));
        assert_eq!(proof.steps[0].transform, Transform::Grad);
        assert_eq!(proof.steps[1].transform, Transform::Vmap);
        assert_eq!(proof.steps[2].transform, Transform::Jit);
        assert!(matches!(
            proof.steps[0].witness,
            TransformRewriteWitness::Grad(_)
        ));
        assert!(matches!(
            &proof.steps[1].witness,
            TransformRewriteWitness::Vmap(w) if w.axis_size == 3
        ));
        assert!(matches!(
            proof.steps[2].witness,
            TransformRewriteWitness::Jit(_)
        ));
        assert_eq!(
            proof.steps[0].output_jaxpr_fingerprint_sha256,
            proof.steps[1].input_jaxpr_fingerprint_sha256
        );
        assert_eq!(
            proof.steps[1].output_jaxpr_fingerprint_sha256,
            proof.steps[2].input_jaxpr_fingerprint_sha256
        );
        assert_eq!(
            proof.steps[2].output_jaxpr_fingerprint_sha256,
            proof.final_output_jaxpr_sha256
        );
        assert!(proof.merkle_commitment.starts_with("ttl-proof-sha256-"));
        proof
            .verify()
            .expect("lowered DispatchResponse composition_proof must verify");

        // Negative: tampered intermediate hash in ledger steps is rejected by dispatch
        let mut tampered_steps = proof.steps.clone();
        tampered_steps[0].output_jaxpr_fingerprint_sha256[0] ^= 0xaa;
        let tampered_ttl = ttl.with_steps(tampered_steps);
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Strict,
            ledger: tampered_ttl,
            args: vec![batch],
            backend: "cpu".to_owned(),
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("dispatch must reject tampered intermediate step chain");
        assert!(matches!(
            err,
            DispatchError::TransformInvariant(TransformCompositionError::StepChainDiscontinuity {
                step_index: 1,
                ..
            })
        ));
    }

    #[test]
    fn e2e_rc26_14_pre_execution_admission_and_ledger_governance() {
        use fj_runtime::{AdmissionRejection, ExecutionRegime};

        super::reset_dispatch_governance();

        // Use a well-typedJaxpr (`Square`, 1 input) but pass 2 runtime `args` so that if
        // `eval_jaxpr` were ever reached it would fail with `InterpreterError::InputArityMismatch`.
        // Pre-execution admission must reject with `DispatchError::AdmissionRejected` BEFORE
        // compilation or `eval_jaxpr` runs.
        let trap_ttl = ledger(ProgramSpec::Square, &[Transform::Jit]);
        let trap_args = vec![Value::scalar_f64(1.0), Value::scalar_f64(2.0)];

        // 1. MemoryBudget rejection blocks BEFORE eval_jaxpr runs.
        let mut tight_mem_opts = BTreeMap::new();
        tight_mem_opts.insert("max_working_set_bytes".to_owned(), "4".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Hardened,
            ledger: trap_ttl.clone(),
            args: trap_args.clone(),
            backend: "cpu".to_owned(),
            compile_options: tight_mem_opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("oversized working set must be rejected before eval_jaxpr");
        assert!(
            matches!(
                err,
                DispatchError::AdmissionRejected(AdmissionRejection::MemoryBudgetExceeded { .. })
            ),
            "expected MemoryBudgetExceeded, got {err:?}"
        );

        // 2. EquationCountExceeded rejection blocks BEFORE eval_jaxpr runs.
        let mut tight_eqn_opts = BTreeMap::new();
        tight_eqn_opts.insert("max_equation_count".to_owned(), "0".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Hardened,
            ledger: trap_ttl.clone(),
            args: trap_args.clone(),
            backend: "cpu".to_owned(),
            compile_options: tight_eqn_opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("equation budget overflow must be rejected before eval_jaxpr");
        assert!(
            matches!(
                err,
                DispatchError::AdmissionRejected(AdmissionRejection::EquationCountExceeded { .. })
            ),
            "expected EquationCountExceeded, got {err:?}"
        );

        // 3. Hardened loss-matrix Kill rejection on high prior_abandoned blocks BEFORE eval_jaxpr runs.
        let mut kill_opts = BTreeMap::new();
        kill_opts.insert("prior_abandoned".to_owned(), "0.95".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Hardened,
            ledger: trap_ttl.clone(),
            args: trap_args.clone(),
            backend: "cpu".to_owned(),
            compile_options: kill_opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("hardened Kill decision must reject before eval_jaxpr");
        assert!(
            matches!(
                err,
                DispatchError::AdmissionRejected(AdmissionRejection::AbandonedWorkKilled { .. })
            ),
            "expected AbandonedWorkKilled, got {err:?}"
        );

        // 4. Sequential EProcess drift rejection blocks BEFORE eval_jaxpr runs, and reset restores admission.
        super::reset_dispatch_governance();
        let mut drift_opts = BTreeMap::new();
        drift_opts.insert("drift_likelihood_ratio".to_owned(), "25.0".to_owned());
        let err = dispatch(DispatchRequest {
            mode: CompatibilityMode::Hardened,
            ledger: trap_ttl,
            args: trap_args,
            backend: "cpu".to_owned(),
            compile_options: drift_opts,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })
        .expect_err("sequential EProcess drift must reject before eval_jaxpr");
        assert!(
            matches!(
                err,
                DispatchError::AdmissionRejected(
                    AdmissionRejection::SequentialDriftRejected { .. }
                )
            ),
            "expected SequentialDriftRejected, got {err:?}"
        );
        super::reset_dispatch_governance();

        // 5. Normal executions calibrate ConformalPredictor and populate telemetry.
        for i in 1..=6 {
            let resp = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Square, &[Transform::Jit]),
                args: vec![Value::scalar_f64(i as f64)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            })
            .expect("normal dispatch should succeed");
            assert_eq!(
                resp.execution_regime(),
                ExecutionRegime::CompiledSlotFastPath
            );
            assert!(resp.estimated_working_set_bytes() >= 16);
            if i >= 6 {
                assert!(resp.calibrated_posterior().used_conformal);
            }
        }
        let stats = super::dispatch_governance_stats();
        assert!(stats.is_calibrated);
        assert!(stats.calibration_size >= 5);
        assert!(stats.ece.is_finite());
    }

    #[test]
    fn dispatch_with_ffi_executes_custom_call_and_rejects_unregistered_target() {
        let registry = fj_ffi::FfiRegistry::new();
        registry
            .register(
                "double_plus_5",
                fj_ffi::fj_ffi_builtin_double_plus5_scalar_f64,
            )
            .unwrap();

        let mut params = BTreeMap::new();
        params.insert("target".to_owned(), "double_plus_5".to_owned());
        params.insert("out_dtype".to_owned(), "f64".to_owned());
        params.insert("out_shape".to_owned(), String::new());

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::CustomCall,
                inputs: smallvec::smallvec![Atom::Var(VarId(1))],
                outputs: smallvec::smallvec![VarId(2)],
                params,
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let mut ttl = TraceTransformLedger::new(jaxpr);
        ttl.push_transform(Transform::Jit, "jit-custom-call".to_owned());

        let response = super::dispatch_with_ffi(
            DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ttl.clone(),
                args: vec![Value::scalar_f64(6.0)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            },
            &registry,
        )
        .expect("custom_call dispatch with registered target should succeed");
        assert_eq!(response.outputs, vec![Value::scalar_f64(17.0)]);

        let empty_registry = fj_ffi::FfiRegistry::new();
        let err = super::dispatch_with_ffi(
            DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ttl,
                args: vec![Value::scalar_f64(6.0)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            },
            &empty_registry,
        )
        .expect_err("unregistered custom_call target must fail closed");
        assert!(err.to_string().contains("not found"));
    }
}
