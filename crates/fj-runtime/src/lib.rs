#![forbid(unsafe_code)]

pub mod backend;
pub mod buffer;
pub mod device;

use fj_core::CompatibilityMode;
use fj_ledger::{DecisionAction, LossMatrix, recommend_action};

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
    fn effective_loss_matrix(&self) -> LossMatrix {
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
}

#[cfg(feature = "asupersync-integration")]
pub mod asupersync_bridge {
    use asupersync::{Cx, Error};

    pub fn emit_checkpoint(cx: &Cx, message: impl Into<String>) -> Result<(), Error> {
        cx.checkpoint_with(message.into())
    }

    #[must_use]
    pub fn cancellation_requested(cx: &Cx) -> bool {
        cx.is_cancel_requested()
    }
}

#[cfg(feature = "frankentui-integration")]
pub mod frankentui_bridge {
    use std::fmt::Write;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct StatusCard {
        pub title: String,
        pub mode: String,
        pub confidence_percent: u8,
    }

    #[must_use]
    pub fn render_status_card(card: &StatusCard) -> String {
        let _type_check = std::any::type_name::<ftui::Style>();

        let mut out = String::new();
        let _ = writeln!(&mut out, "[{}]", card.title);
        let _ = writeln!(&mut out, "mode: {}", card.mode);
        let _ = write!(&mut out, "confidence: {}%", card.confidence_percent);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeAdmissionModel;
    use fj_core::CompatibilityMode;
    use fj_ledger::DecisionAction;

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
}
