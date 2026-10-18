#![forbid(unsafe_code)]

pub mod backend;
pub mod buffer;
pub mod device;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use fj_core::{CompatibilityMode, DType, Jaxpr, Value};
use fj_ledger::{
    CalibratedEstimate, ConformalPredictor, DecisionAction, DecisionRecord, EProcess, LossMatrix,
    recommend_action,
};

/// Resource and graph complexity budget enforced before dispatch execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Maximum estimated working-set bytes across input arguments and graph intermediates.
    pub max_working_set_bytes: usize,
    /// Maximum equation count (including nested `sub_jaxprs`) permitted for a single dispatch.
    pub max_equation_count: usize,
    /// Maximum transform composition depth permitted for a single dispatch.
    pub max_transform_depth: usize,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            max_working_set_bytes: 256 * 1024 * 1024,
            max_equation_count: 65_536,
            max_transform_depth: 32,
        }
    }
}

impl MemoryBudget {
    #[must_use]
    pub const fn new(
        max_working_set_bytes: usize,
        max_equation_count: usize,
        max_transform_depth: usize,
    ) -> Self {
        Self {
            max_working_set_bytes,
            max_equation_count,
            max_transform_depth,
        }
    }
}

/// Cooperative cancellation token checked by pre-execution runtime admission.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    reason: Arc<Mutex<Option<String>>>,
}

impl CancellationToken {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self, reason: impl Into<String>) {
        if let Ok(mut slot) = self.reason.lock() {
            *slot = Some(reason.into());
        }
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
        if let Ok(mut slot) = self.reason.lock() {
            *slot = None;
        }
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    #[must_use]
    pub fn reason(&self) -> Option<String> {
        self.reason.lock().ok().and_then(|guard| guard.clone())
    }
}

/// Execution strategy selected by pre-execution runtime admission and ledger governance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionRegime {
    CompiledSlotFastPath,
    WaveParallelCpu,
    EGraphSaturatedCompile,
    FailClosedHardenedReject,
}

impl ExecutionRegime {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::CompiledSlotFastPath => "compiled_slot_fast_path",
            Self::WaveParallelCpu => "wave_parallel_cpu",
            Self::EGraphSaturatedCompile => "egraph_saturated_compile",
            Self::FailClosedHardenedReject => "fail_closed_hardened_reject",
        }
    }
}

/// Reason why pre-execution runtime admission blocked a dispatch request before evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionRejection {
    Cancelled {
        reason: String,
    },
    MemoryBudgetExceeded {
        estimated_bytes: usize,
        max_bytes: usize,
    },
    EquationCountExceeded {
        equation_count: usize,
        max_equations: usize,
    },
    TransformDepthExceeded {
        transform_depth: usize,
        max_depth: usize,
    },
    SequentialDriftRejected {
        e_value_milli: u64,
        threshold_milli: u64,
    },
    AbandonedWorkKilled {
        mode: CompatibilityMode,
        posterior_milli: u64,
    },
}

impl std::fmt::Display for AdmissionRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled { reason } => {
                write!(f, "runtime admission rejected: cancelled ({reason})")
            }
            Self::MemoryBudgetExceeded {
                estimated_bytes,
                max_bytes,
            } => write!(
                f,
                "runtime admission rejected: estimated working set {estimated_bytes} bytes exceeds budget {max_bytes} bytes"
            ),
            Self::EquationCountExceeded {
                equation_count,
                max_equations,
            } => write!(
                f,
                "runtime admission rejected: equation count {equation_count} exceeds budget {max_equations}"
            ),
            Self::TransformDepthExceeded {
                transform_depth,
                max_depth,
            } => write!(
                f,
                "runtime admission rejected: transform depth {transform_depth} exceeds budget {max_depth}"
            ),
            Self::SequentialDriftRejected {
                e_value_milli,
                threshold_milli,
            } => write!(
                f,
                "runtime admission rejected: e-process sequential drift e_value={:.3} >= threshold={:.3}",
                *e_value_milli as f64 / 1000.0,
                *threshold_milli as f64 / 1000.0
            ),
            Self::AbandonedWorkKilled {
                mode,
                posterior_milli,
            } => write!(
                f,
                "runtime admission rejected in {mode:?} mode: loss-matrix action=Kill at posterior={:.3}",
                *posterior_milli as f64 / 1000.0
            ),
        }
    }
}

impl std::error::Error for AdmissionRejection {}

/// Pre-execution admission outcome produced by [`RuntimeAdmissionModel::evaluate_admission`].
#[derive(Debug, Clone, PartialEq)]
pub struct AdmissionDecision {
    pub record: DecisionRecord,
    pub calibrated_posterior: CalibratedEstimate,
    pub estimated_working_set_bytes: usize,
    pub total_equation_count: usize,
    pub e_value: f64,
    pub regime: ExecutionRegime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAdmissionModel {
    pub mode: CompatibilityMode,
    pub loss_matrix: LossMatrix,
}

impl RuntimeAdmissionModel {
    #[must_use]
    pub fn new(mode: CompatibilityMode) -> Self {
        Self {
            mode,
            loss_matrix: LossMatrix::default(),
        }
    }

    #[must_use]
    pub fn decide(&self, posterior_abandoned: f64) -> DecisionAction {
        let matrix = self.effective_loss_matrix();
        recommend_action(posterior_abandoned, &matrix)
    }

    /// Return the loss matrix adjusted for the compatibility mode.
    /// - Strict: uses the configured matrix as-is (conservative, high kill_if_useful cost).
    /// - Hardened: reduces kill_if_useful and increases keep_if_abandoned, making the
    ///   model more aggressive about reclaiming resources from suspected-abandoned work.
    #[must_use]
    pub fn effective_loss_matrix(&self) -> LossMatrix {
        match self.mode {
            CompatibilityMode::Strict => self.loss_matrix.clone(),
            CompatibilityMode::Hardened => LossMatrix {
                keep_if_useful: self.loss_matrix.keep_if_useful,
                kill_if_useful: (self.loss_matrix.kill_if_useful / 2).max(1),
                keep_if_abandoned: self
                    .loss_matrix
                    .keep_if_abandoned
                    .saturating_mul(2)
                    .min(100),
                kill_if_abandoned: self.loss_matrix.kill_if_abandoned,
            },
        }
    }

    /// Count total equations across `jaxpr` and all nested `sub_jaxprs`.
    #[must_use]
    pub fn count_total_equations(jaxpr: &Jaxpr) -> usize {
        let mut total = jaxpr.equations.len();
        for eqn in &jaxpr.equations {
            for sub in &eqn.sub_jaxprs {
                total = total.saturating_add(Self::count_total_equations(sub));
            }
        }
        total
    }

    /// Estimate the working-set memory footprint in bytes for `jaxpr` executed on `args`.
    #[must_use]
    pub fn estimate_working_set_bytes(jaxpr: &Jaxpr, args: &[Value]) -> usize {
        let mut arg_bytes: usize = 0;
        let mut max_arg_bytes: usize = 8;
        for arg in args {
            let bytes = value_byte_footprint(arg);
            arg_bytes = arg_bytes.saturating_add(bytes);
            max_arg_bytes = max_arg_bytes.max(bytes);
        }
        let const_bytes = jaxpr.constvars.len().saturating_mul(8);
        let total_eqns = Self::count_total_equations(jaxpr);
        let intermediate_bytes = total_eqns.saturating_mul(max_arg_bytes.max(8));
        arg_bytes
            .saturating_add(const_bytes)
            .saturating_add(intermediate_bytes)
    }

    /// Run pre-execution admission checks and statistical decision governance.
    ///
    /// Evaluates cancellation state, `MemoryBudget` limits, sequential drift (`EProcess`),
    /// conformal uncertainty calibration (`ConformalPredictor`), and loss-matrix expected
    /// loss (`DecisionRecord`), returning the selected [`ExecutionRegime`] or failing closed
    /// with an [`AdmissionRejection`] *before* any IR evaluation occurs.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_admission(
        &self,
        jaxpr: &Jaxpr,
        args: &[Value],
        transform_depth: usize,
        posterior_abandoned: f64,
        budget: &MemoryBudget,
        cancellation: Option<&CancellationToken>,
        conformal: &ConformalPredictor,
        e_process: &EProcess,
        has_parallel_heavy_wave: bool,
        wants_egraph: bool,
        reject_on_kill: bool,
    ) -> Result<AdmissionDecision, AdmissionRejection> {
        if let Some(tok) = cancellation
            && tok.is_cancelled()
        {
            return Err(AdmissionRejection::Cancelled {
                reason: tok
                    .reason()
                    .unwrap_or_else(|| "cancellation requested".to_owned()),
            });
        }

        #[cfg(feature = "asupersync-integration")]
        if let Some(cx) = asupersync::Cx::current()
            && asupersync_bridge::run_admission_probe(&cx, "fj-runtime pre-execution admission")
                .is_err()
        {
            return Err(AdmissionRejection::Cancelled {
                reason: "asupersync caller context cancelled".to_owned(),
            });
        }

        let estimated_working_set_bytes = Self::estimate_working_set_bytes(jaxpr, args);
        if estimated_working_set_bytes > budget.max_working_set_bytes {
            return Err(AdmissionRejection::MemoryBudgetExceeded {
                estimated_bytes: estimated_working_set_bytes,
                max_bytes: budget.max_working_set_bytes,
            });
        }

        let total_equation_count = Self::count_total_equations(jaxpr);
        if total_equation_count > budget.max_equation_count {
            return Err(AdmissionRejection::EquationCountExceeded {
                equation_count: total_equation_count,
                max_equations: budget.max_equation_count,
            });
        }

        if transform_depth > budget.max_transform_depth {
            return Err(AdmissionRejection::TransformDepthExceeded {
                transform_depth,
                max_depth: budget.max_transform_depth,
            });
        }

        if e_process.rejected() {
            return Err(AdmissionRejection::SequentialDriftRejected {
                e_value_milli: (e_process.e_value() * 1000.0).round() as u64,
                threshold_milli: 20_000,
            });
        }

        let calibrated_posterior = conformal.calibrated_posterior(posterior_abandoned);
        let point_posterior = calibrated_posterior.point;

        let matrix = self.effective_loss_matrix();
        let record = DecisionRecord::from_posterior(self.mode, point_posterior, &matrix);

        if record.action == DecisionAction::Kill
            && (reject_on_kill
                || (self.mode == CompatibilityMode::Hardened && point_posterior >= 0.85))
        {
            return Err(AdmissionRejection::AbandonedWorkKilled {
                mode: self.mode,
                posterior_milli: (point_posterior * 1000.0).round() as u64,
            });
        }

        let regime = if has_parallel_heavy_wave {
            ExecutionRegime::WaveParallelCpu
        } else if wants_egraph || record.action == DecisionAction::Reprofile {
            ExecutionRegime::EGraphSaturatedCompile
        } else {
            ExecutionRegime::CompiledSlotFastPath
        };

        Ok(AdmissionDecision {
            record,
            calibrated_posterior,
            estimated_working_set_bytes,
            total_equation_count,
            e_value: e_process.e_value(),
            regime,
        })
    }
}

fn dtype_byte_width(dtype: DType) -> usize {
    match dtype {
        DType::Bool => 1,
        DType::F16 | DType::BF16 => 2,
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::I64 | DType::U64 | DType::F64 | DType::Complex64 => 8,
        DType::Complex128 => 16,
    }
}

fn value_byte_footprint(value: &Value) -> usize {
    match value {
        Value::Scalar(lit) => match lit {
            fj_core::Literal::Bool(_) => 1,
            fj_core::Literal::F16Bits(_) | fj_core::Literal::BF16Bits(_) => 2,
            fj_core::Literal::I32(_) | fj_core::Literal::U32(_) | fj_core::Literal::F32Bits(_) => 4,
            fj_core::Literal::I64(_)
            | fj_core::Literal::U64(_)
            | fj_core::Literal::F64Bits(_)
            | fj_core::Literal::Complex64Bits(..) => 8,
            fj_core::Literal::Complex128Bits(..) => 16,
        },
        Value::Tensor(tensor) => tensor
            .elements
            .len()
            .saturating_mul(dtype_byte_width(tensor.dtype)),
    }
}

#[cfg(feature = "asupersync-integration")]
pub mod asupersync_bridge {
    use asupersync::{Cx, Error};

    pub fn emit_checkpoint(cx: &Cx, message: impl Into<String>) -> Result<(), Box<Error>> {
        cx.checkpoint_with(message.into()).map_err(Box::new)
    }

    #[must_use]
    pub fn cancellation_requested(cx: &Cx) -> bool {
        cx.is_cancel_requested()
    }

    /// Run an async-context admission probe before executing a dispatch workload.
    pub fn run_admission_probe(cx: &Cx, label: impl Into<String>) -> Result<(), Box<Error>> {
        if cancellation_requested(cx) {
            return emit_checkpoint(cx, label);
        }
        emit_checkpoint(cx, label)
    }
}

#[cfg(feature = "frankentui-integration")]
pub mod frankentui_bridge {
    use ftui::text::{Line, Span, Text};
    use ftui::{Buffer, Cell, CellAttrs, PackedRgba, Style, StyleFlags};

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct StatusCard {
        pub title: String,
        pub mode: String,
        pub confidence_percent: u8,
    }

    /// Construct a styled [`ftui::text::Text`] representation of a runtime [`StatusCard`]
    /// using real [`ftui::Style`] colors and attribute flags.
    #[must_use]
    pub fn build_styled_status_text(card: &StatusCard) -> Text {
        let title_style = Style::new()
            .fg(PackedRgba::rgb(80, 200, 255))
            .bg(PackedRgba::rgb(16, 24, 32))
            .bold();
        let label_style = Style::new().fg(PackedRgba::rgb(180, 190, 200));
        let mode_style = if card.mode.eq_ignore_ascii_case("hardened") {
            Style::new().fg(PackedRgba::rgb(255, 180, 60)).bold()
        } else {
            Style::new().fg(PackedRgba::rgb(100, 220, 140))
        };
        let conf_style = if card.confidence_percent >= 80 {
            Style::new().fg(PackedRgba::rgb(80, 230, 120)).bold()
        } else if card.confidence_percent >= 50 {
            Style::new().fg(PackedRgba::rgb(240, 200, 80))
        } else {
            Style::new()
                .fg(PackedRgba::rgb(240, 90, 90))
                .bold()
                .underline()
        };

        let header_line =
            Line::from_spans([Span::styled(format!("[{}]", card.title), title_style)]);
        let mode_line = Line::from_spans([
            Span::styled("mode: ", label_style),
            Span::styled(card.mode.clone(), mode_style),
        ]);
        let confidence_line = Line::from_spans([
            Span::styled("confidence: ", label_style),
            Span::styled(format!("{}%", card.confidence_percent), conf_style),
        ]);

        Text::from_lines([header_line, mode_line, confidence_line])
    }

    /// Rasterize a [`StatusCard`] into a 2D [`ftui::Buffer`] with cell-level [`ftui::Style`]
    /// foreground/background colors and attribute flags.
    #[must_use]
    pub fn render_status_buffer(card: &StatusCard) -> Buffer {
        let styled_text = build_styled_status_text(card);
        let width = styled_text.width().max(1).min(u16::MAX as usize) as u16;
        let height = styled_text.height().max(1).min(u16::MAX as usize) as u16;
        let mut buffer = Buffer::new(width, height);

        for (row_idx, line) in styled_text.lines().iter().enumerate() {
            let y = row_idx as u16;
            let mut x: u16 = 0;
            for span in line.spans() {
                let style = span.style.unwrap_or_default();
                let cell_flags: ftui::render::cell::StyleFlags =
                    style.attrs.unwrap_or(StyleFlags::NONE).into();
                let attrs = CellAttrs::new(cell_flags, 0);
                for ch in span.as_str().chars() {
                    if x >= width {
                        break;
                    }
                    let mut cell = Cell::from_char(ch);
                    if let Some(fg) = style.fg {
                        cell.fg = fg;
                    }
                    if let Some(bg) = style.bg {
                        cell.bg = bg;
                    }
                    cell.attrs = attrs;
                    buffer.set(x, y, cell);
                    x = x.saturating_add(1);
                }
            }
        }
        buffer
    }

    /// Render a [`StatusCard`] via [`ftui::text::Text`] and [`ftui::Buffer`] and return its
    /// canonical text representation.
    #[must_use]
    pub fn render_status_card(card: &StatusCard) -> String {
        let styled_text = build_styled_status_text(card);
        let _rendered_buffer = render_status_buffer(card);
        styled_text.to_plain_text()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdmissionRejection, CancellationToken, ExecutionRegime, MemoryBudget, RuntimeAdmissionModel,
    };
    use fj_core::{CompatibilityMode, ProgramSpec, Value, build_program};
    use fj_ledger::{ConformalPredictor, DecisionAction, EProcess};

    #[test]
    fn runtime_admission_is_conservative_at_low_abandoned_probability() {
        let model = RuntimeAdmissionModel::new(CompatibilityMode::Strict);
        assert_eq!(model.decide(0.1), DecisionAction::Keep);
    }

    #[test]
    fn runtime_admission_kills_at_high_abandoned_probability() {
        let model = RuntimeAdmissionModel::new(CompatibilityMode::Strict);
        assert_eq!(model.decide(0.95), DecisionAction::Kill);
    }

    #[test]
    fn hardened_mode_kills_at_lower_threshold_than_strict() {
        // Find a posterior value where Strict keeps but Hardened kills,
        // demonstrating that mode influences the decision.
        let strict = RuntimeAdmissionModel::new(CompatibilityMode::Strict);
        let hardened = RuntimeAdmissionModel::new(CompatibilityMode::Hardened);

        // At moderate posterior (around 0.6), hardened should be more aggressive.
        // Strict default: kill_if_useful=100, keep_if_abandoned=30
        // Hardened:        kill_if_useful=50,  keep_if_abandoned=60
        // expected_loss_keep(0.6, strict) = 0.4*0 + 0.6*30 = 18.0
        // expected_loss_kill(0.6, strict) = 0.4*100 + 0.6*1 = 40.6 → Keep
        // expected_loss_keep(0.6, hardened) = 0.4*0 + 0.6*60 = 36.0
        // expected_loss_kill(0.6, hardened) = 0.4*50 + 0.6*1 = 20.6 → Kill
        assert_eq!(strict.decide(0.6), DecisionAction::Keep);
        assert_eq!(hardened.decide(0.6), DecisionAction::Kill);
    }

    #[test]
    fn test_runtime_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("runtime", "admission")).expect("digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_runtime_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    #[cfg(feature = "asupersync-integration")]
    #[test]
    fn asupersync_bridge_checkpoint_and_cancellation_roundtrip() {
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
            .build()
            .expect("runtime should build");
        let budget = asupersync::Budget::INFINITE.with_poll_quota(8);
        let request = runtime.request_cx_with_budget(budget);
        assert!(asupersync::Cx::current().is_none());
        let cx = {
            let _restriction = request
                .restrict::<asupersync::cx::cap::None>()
                .set_current_restricted();
            asupersync::Cx::current().expect("restricted request is installed")
        };
        assert!(asupersync::Cx::current().is_none());
        let capabilities = cx.capabilities();
        assert_eq!(
            [
                capabilities.spawn,
                capabilities.time,
                capabilities.entropy,
                capabilities.io,
                capabilities.remote,
            ],
            [false; 5]
        );
        assert_eq!(cx.budget(), budget);

        assert!(!super::asupersync_bridge::cancellation_requested(&cx));
        super::asupersync_bridge::emit_checkpoint(&cx, "frankenjax runtime checkpoint")
            .expect("live caller context should accept checkpoints");
        super::asupersync_bridge::run_admission_probe(&cx, "pre-execution probe")
            .expect("admission probe should pass when not cancelled");

        request.cancel_with(asupersync::CancelKind::User, Some("caller cancelled"));
        assert!(super::asupersync_bridge::cancellation_requested(&cx));
        let error = super::asupersync_bridge::emit_checkpoint(&cx, "cancelled checkpoint")
            .expect_err("bridge must observe caller cancellation");
        assert_eq!(error.kind(), asupersync::error::ErrorKind::Cancelled);
        assert_eq!(cx.capabilities(), capabilities);
        assert_eq!(cx.budget(), budget);
        assert!(asupersync::Cx::current().is_none());
    }

    // ── Admission model boundary tests ────────────────────────

    #[test]
    fn admission_keep_at_zero_posterior() {
        let model = RuntimeAdmissionModel::new(CompatibilityMode::Strict);
        assert_eq!(model.decide(0.0), DecisionAction::Keep);
    }

    #[test]
    fn admission_kill_at_one_posterior() {
        let model = RuntimeAdmissionModel::new(CompatibilityMode::Strict);
        assert_eq!(model.decide(1.0), DecisionAction::Kill);
    }

    #[test]
    fn admission_strict_vs_hardened_monotonic() {
        // Hardened should be at least as aggressive as strict at every posterior
        let strict = RuntimeAdmissionModel::new(CompatibilityMode::Strict);
        let hardened = RuntimeAdmissionModel::new(CompatibilityMode::Hardened);

        let mut strict_kills_at = None;
        let mut hardened_kills_at = None;
        for i in 0..=100 {
            let posterior = i as f64 / 100.0;
            if strict.decide(posterior) == DecisionAction::Kill && strict_kills_at.is_none() {
                strict_kills_at = Some(posterior);
            }
            if hardened.decide(posterior) == DecisionAction::Kill && hardened_kills_at.is_none() {
                hardened_kills_at = Some(posterior);
            }
        }

        let s = strict_kills_at.unwrap_or(1.0);
        let h = hardened_kills_at.unwrap_or(1.0);
        assert!(
            h <= s,
            "hardened should kill at same or lower posterior than strict: hardened={h}, strict={s}"
        );
    }

    #[test]
    fn admission_effective_loss_matrix_strict_unchanged() {
        let model = RuntimeAdmissionModel::new(CompatibilityMode::Strict);
        let effective = model.effective_loss_matrix();
        assert_eq!(effective, model.loss_matrix);
    }

    #[test]
    fn admission_effective_loss_matrix_hardened_more_aggressive() {
        let model = RuntimeAdmissionModel::new(CompatibilityMode::Hardened);
        let effective = model.effective_loss_matrix();
        let default = fj_ledger::LossMatrix::default();
        // Hardened should have lower kill_if_useful (less penalty for killing useful work)
        assert!(effective.kill_if_useful <= default.kill_if_useful);
        // Hardened should have higher keep_if_abandoned (more penalty for keeping abandoned)
        assert!(effective.keep_if_abandoned >= default.keep_if_abandoned);
    }

    #[test]
    fn evaluate_admission_enforces_memory_budget_and_eprocess() {
        let model = RuntimeAdmissionModel::new(CompatibilityMode::Hardened);
        let jaxpr = build_program(ProgramSpec::Add2);
        let args = vec![Value::scalar_i64(1), Value::scalar_i64(2)];
        let conformal = ConformalPredictor::new(0.9, 5);
        let e_process = EProcess::new(20.0);

        // Normal budget admits
        let decision = model
            .evaluate_admission(
                &jaxpr,
                &args,
                1,
                0.1,
                &MemoryBudget::default(),
                None,
                &conformal,
                &e_process,
                false,
                false,
                false,
            )
            .expect("normal budget should admit");
        assert_eq!(decision.regime, ExecutionRegime::CompiledSlotFastPath);

        // Tight byte budget rejects
        let tight_budget = MemoryBudget::new(4, 100, 10);
        let err = model
            .evaluate_admission(
                &jaxpr,
                &args,
                1,
                0.1,
                &tight_budget,
                None,
                &conformal,
                &e_process,
                false,
                false,
                false,
            )
            .expect_err("tight memory budget must reject");
        assert!(matches!(
            err,
            AdmissionRejection::MemoryBudgetExceeded { .. }
        ));

        // Triggered EProcess rejects
        let mut drifted_e = EProcess::new(10.0);
        drifted_e.update(5.0);
        drifted_e.update(3.0);
        assert!(drifted_e.rejected());
        let err = model
            .evaluate_admission(
                &jaxpr,
                &args,
                1,
                0.1,
                &MemoryBudget::default(),
                None,
                &conformal,
                &drifted_e,
                false,
                false,
                false,
            )
            .expect_err("rejected e-process must block admission");
        assert!(matches!(
            err,
            AdmissionRejection::SequentialDriftRejected { .. }
        ));

        // Cancelled token rejects
        let token = CancellationToken::new();
        token.cancel("operator abort");
        let err = model
            .evaluate_admission(
                &jaxpr,
                &args,
                1,
                0.1,
                &MemoryBudget::default(),
                Some(&token),
                &conformal,
                &e_process,
                false,
                false,
                false,
            )
            .expect_err("cancelled token must block admission");
        assert!(matches!(err, AdmissionRejection::Cancelled { .. }));
    }

    #[cfg(feature = "frankentui-integration")]
    #[test]
    fn frankentui_status_card_rendering_matches_snapshot() {
        let card = super::frankentui_bridge::StatusCard {
            title: "FrankenJAX Runtime".to_owned(),
            mode: "hardened".to_owned(),
            confidence_percent: 87,
        };

        let styled_text = super::frankentui_bridge::build_styled_status_text(&card);
        assert_eq!(styled_text.height(), 3);
        let header_span = &styled_text.lines()[0].spans()[0];
        let header_style = header_span
            .style
            .expect("header span must have ftui::Style");
        assert!(header_style.fg.is_some());
        assert!(
            header_style
                .attrs
                .is_some_and(|flags| flags.contains(ftui::StyleFlags::BOLD))
        );

        let buffer = super::frankentui_bridge::render_status_buffer(&card);
        assert_eq!(buffer.height(), 3);
        assert!(buffer.content_height() == 3);
        let first_cell = buffer.get(0, 0).expect("top-left cell");
        assert_eq!(first_cell.fg, ftui::PackedRgba::rgb(80, 200, 255));

        insta::assert_snapshot!(
            "frankentui_status_card",
            super::frankentui_bridge::render_status_card(&card)
        );
    }
}
