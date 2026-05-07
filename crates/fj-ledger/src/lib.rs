#![forbid(unsafe_code)]

use fj_core::CompatibilityMode;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionAction {
    Keep,
    Kill,
    Reprofile,
    Fallback,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSignal {
    pub signal_name: String,
    pub log_likelihood_delta: f64,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LossMatrix {
    pub keep_if_useful: u32,
    pub kill_if_useful: u32,
    pub keep_if_abandoned: u32,
    pub kill_if_abandoned: u32,
}

impl Default for LossMatrix {
    fn default() -> Self {
        Self {
            keep_if_useful: 0,
            kill_if_useful: 100,
            keep_if_abandoned: 30,
            kill_if_abandoned: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub mode: CompatibilityMode,
    pub posterior_abandoned: f64,
    pub expected_loss_keep: f64,
    pub expected_loss_kill: f64,
    pub action: DecisionAction,
    pub timestamp_unix_ms: u128,
}

impl DecisionRecord {
    #[must_use]
    pub fn from_posterior(
        mode: CompatibilityMode,
        posterior_abandoned: f64,
        matrix: &LossMatrix,
    ) -> Self {
        let expected_loss_keep = expected_loss_keep(posterior_abandoned, matrix);
        let expected_loss_kill = expected_loss_kill(posterior_abandoned, matrix);
        let action = if expected_loss_keep < expected_loss_kill {
            DecisionAction::Keep
        } else if expected_loss_kill < expected_loss_keep {
            DecisionAction::Kill
        } else {
            DecisionAction::Reprofile
        };

        Self {
            mode,
            posterior_abandoned,
            expected_loss_keep,
            expected_loss_kill,
            action,
            timestamp_unix_ms: now_unix_ms(),
        }
    }
}

#[must_use]
pub fn recommend_action(posterior_abandoned: f64, matrix: &LossMatrix) -> DecisionAction {
    DecisionRecord::from_posterior(CompatibilityMode::Strict, posterior_abandoned, matrix).action
}

#[must_use]
pub fn expected_loss_keep(posterior_abandoned: f64, matrix: &LossMatrix) -> f64 {
    let useful_prob = 1.0 - posterior_abandoned;
    useful_prob * f64::from(matrix.keep_if_useful)
        + posterior_abandoned * f64::from(matrix.keep_if_abandoned)
}

#[must_use]
pub fn expected_loss_kill(posterior_abandoned: f64, matrix: &LossMatrix) -> f64 {
    let useful_prob = 1.0 - posterior_abandoned;
    useful_prob * f64::from(matrix.kill_if_useful)
        + posterior_abandoned * f64::from(matrix.kill_if_abandoned)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub decision_id: String,
    pub record: DecisionRecord,
    pub signals: Vec<EvidenceSignal>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvidenceLedger {
    entries: Vec<LedgerEntry>,
}

impl EvidenceLedger {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn append(&mut self, entry: LedgerEntry) {
        self.entries.push(entry);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }
}

/// Distribution-free conformal predictor for calibrated uncertainty.
///
/// Maintains a calibration set of nonconformity scores and produces
/// prediction intervals with finite-sample coverage guarantees.
#[derive(Debug, Clone)]
pub struct ConformalPredictor {
    calibration_scores: Vec<f64>,
    target_coverage: f64,
    min_calibration_size: usize,
}

/// A calibrated estimate combining point estimate with conformal bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedEstimate {
    pub point: f64,
    pub lower: f64,
    pub upper: f64,
    pub coverage: f64,
    pub used_conformal: bool,
}

impl ConformalPredictor {
    #[must_use]
    pub fn new(target_coverage: f64, min_calibration_size: usize) -> Self {
        Self {
            calibration_scores: Vec::new(),
            target_coverage: target_coverage.clamp(0.5, 0.999),
            min_calibration_size: min_calibration_size.max(2),
        }
    }

    pub fn observe(&mut self, score: f64) {
        self.calibration_scores.push(score);
    }

    #[must_use]
    pub fn calibration_size(&self) -> usize {
        self.calibration_scores.len()
    }

    #[must_use]
    pub fn is_calibrated(&self) -> bool {
        self.calibration_scores.len() >= self.min_calibration_size
    }

    /// Returns the conformal prediction threshold (quantile) if calibrated.
    #[must_use]
    pub fn prediction_threshold(&self) -> Option<f64> {
        if !self.is_calibrated() {
            return None;
        }

        let mut sorted = self.calibration_scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        // Finite-sample correction: ceil((n+1) * coverage) / n
        let idx = ((n as f64 + 1.0) * self.target_coverage).ceil() as usize;
        let idx = idx.min(n).saturating_sub(1);
        Some(sorted[idx])
    }

    /// Returns a calibrated posterior estimate with conformal bounds.
    /// Falls back to heuristic when calibration set is too small.
    #[must_use]
    pub fn calibrated_posterior(&self, point_estimate: f64) -> CalibratedEstimate {
        match self.prediction_threshold() {
            Some(threshold) => {
                let half_width = threshold;
                CalibratedEstimate {
                    point: point_estimate,
                    lower: (point_estimate - half_width).clamp(0.0, 1.0),
                    upper: (point_estimate + half_width).clamp(0.0, 1.0),
                    coverage: self.target_coverage,
                    used_conformal: true,
                }
            }
            None => CalibratedEstimate {
                point: point_estimate,
                lower: (point_estimate * 0.5).clamp(0.0, 1.0),
                upper: (point_estimate * 1.5).clamp(0.0, 1.0),
                coverage: 0.0,
                used_conformal: false,
            },
        }
    }
}

/// Numerically stable log-domain posterior computation.
#[derive(Debug, Clone)]
pub struct LogDomainPosterior {
    log_prior_abandoned: f64,
    log_prior_useful: f64,
    accumulated_log_likelihood: f64,
}

impl LogDomainPosterior {
    #[must_use]
    pub fn new(prior_abandoned: f64) -> Self {
        let prior = prior_abandoned.clamp(1e-10, 1.0 - 1e-10);
        Self {
            log_prior_abandoned: prior.ln(),
            log_prior_useful: (1.0 - prior).ln(),
            accumulated_log_likelihood: 0.0,
        }
    }

    pub fn update(&mut self, log_likelihood_delta: f64) {
        self.accumulated_log_likelihood += log_likelihood_delta;
    }

    #[must_use]
    pub fn posterior_abandoned(&self) -> f64 {
        let log_numerator = self.log_prior_abandoned + self.accumulated_log_likelihood;
        let log_denominator = log_sum_exp(
            log_numerator,
            self.log_prior_useful, // no accumulated evidence for "useful" (log-likelihood = 0)
        );
        (log_numerator - log_denominator).exp().clamp(0.0, 1.0)
    }

    #[must_use]
    pub fn bayes_factor(&self) -> f64 {
        self.accumulated_log_likelihood.exp()
    }
}

/// Numerically stable log-sum-exp: log(exp(a) + exp(b))
#[must_use]
pub fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Expected Calibration Error tracking.
#[derive(Debug, Clone)]
pub struct CalibrationReport {
    bins: Vec<CalibrationBin>,
    num_bins: usize,
}

#[derive(Debug, Clone)]
struct CalibrationBin {
    predicted_sum: f64,
    observed_sum: f64,
    count: usize,
}

impl CalibrationReport {
    #[must_use]
    pub fn new(num_bins: usize) -> Self {
        let num_bins = num_bins.max(2);
        Self {
            bins: (0..num_bins)
                .map(|_| CalibrationBin {
                    predicted_sum: 0.0,
                    observed_sum: 0.0,
                    count: 0,
                })
                .collect(),
            num_bins,
        }
    }

    pub fn observe(&mut self, predicted: f64, observed: bool) {
        let clamped = predicted.clamp(0.0, 1.0);
        let bin_idx = ((clamped * self.num_bins as f64) as usize).min(self.num_bins - 1);
        let bin = &mut self.bins[bin_idx];
        bin.predicted_sum += clamped;
        bin.observed_sum += if observed { 1.0 } else { 0.0 };
        bin.count += 1;
    }

    /// Compute Expected Calibration Error.
    #[must_use]
    pub fn compute_ece(&self) -> f64 {
        let total: usize = self.bins.iter().map(|b| b.count).sum();
        if total == 0 {
            return 0.0;
        }

        self.bins
            .iter()
            .filter(|b| b.count > 0)
            .map(|b| {
                let avg_predicted = b.predicted_sum / b.count as f64;
                let avg_observed = b.observed_sum / b.count as f64;
                let weight = b.count as f64 / total as f64;
                weight * (avg_predicted - avg_observed).abs()
            })
            .sum()
    }
}

/// E-Process for anytime-valid sequential hypothesis testing.
///
/// Accumulates evidence via likelihood ratios; rejection is valid at any
/// stopping time (no p-hacking risk).
#[derive(Debug, Clone)]
pub struct EProcess {
    e_value: f64,
    observations: usize,
    rejection_threshold: f64,
}

impl EProcess {
    #[must_use]
    pub fn new(rejection_threshold: f64) -> Self {
        Self {
            e_value: 1.0,
            observations: 0,
            rejection_threshold: rejection_threshold.max(1.0),
        }
    }

    pub fn update(&mut self, likelihood_ratio: f64) {
        self.e_value *= likelihood_ratio.max(0.0);
        self.observations += 1;
    }

    #[must_use]
    pub fn e_value(&self) -> f64 {
        self.e_value
    }

    #[must_use]
    pub fn observations(&self) -> usize {
        self.observations
    }

    /// Returns true if the accumulated evidence exceeds the rejection threshold.
    /// This is anytime-valid: can be checked after any number of observations.
    #[must_use]
    pub fn rejected(&self) -> bool {
        self.e_value >= self.rejection_threshold
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use super::{
        DecisionAction, DecisionRecord, EvidenceLedger, LossMatrix, expected_loss_keep,
        expected_loss_kill, recommend_action,
    };
    use fj_core::CompatibilityMode;

    #[test]
    fn loss_matrix_prefers_keep_when_useful_probability_is_high() {
        let matrix = LossMatrix::default();
        let keep = expected_loss_keep(0.1, &matrix);
        let kill = expected_loss_kill(0.1, &matrix);
        assert!(keep < kill);
        assert_eq!(recommend_action(0.1, &matrix), DecisionAction::Keep);
    }

    #[test]
    fn loss_matrix_prefers_kill_when_abandoned_probability_is_high() {
        let matrix = LossMatrix::default();
        let keep = expected_loss_keep(0.95, &matrix);
        let kill = expected_loss_kill(0.95, &matrix);
        assert!(kill < keep);
        assert_eq!(recommend_action(0.95, &matrix), DecisionAction::Kill);
    }

    #[test]
    fn decision_record_includes_timestamp() {
        let matrix = LossMatrix::default();
        let record = DecisionRecord::from_posterior(CompatibilityMode::Hardened, 0.5, &matrix);
        assert!(record.timestamp_unix_ms > 0);
    }

    #[test]
    fn ledger_append_increases_length() {
        let mut ledger = EvidenceLedger::new();
        assert!(ledger.is_empty());
        ledger.append(super::LedgerEntry {
            decision_id: "d1".to_owned(),
            record: DecisionRecord::from_posterior(
                CompatibilityMode::Strict,
                0.3,
                &LossMatrix::default(),
            ),
            signals: Vec::new(),
        });
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn conformal_predictor_uncalibrated_fallback() {
        let cp = super::ConformalPredictor::new(0.9, 10);
        assert!(!cp.is_calibrated());
        assert!(cp.prediction_threshold().is_none());

        let estimate = cp.calibrated_posterior(0.5);
        assert!(!estimate.used_conformal);
        assert!((estimate.point - 0.5).abs() < 1e-10);
    }

    #[test]
    fn conformal_predictor_calibrated() {
        let mut cp = super::ConformalPredictor::new(0.9, 5);
        for i in 1..=10 {
            cp.observe(i as f64 * 0.1);
        }
        assert!(cp.is_calibrated());
        let threshold = cp.prediction_threshold().expect("should be calibrated");
        assert!(threshold > 0.0);

        let estimate = cp.calibrated_posterior(0.5);
        assert!(estimate.used_conformal);
        assert!((estimate.coverage - 0.9).abs() < 1e-10);
    }

    #[test]
    fn log_domain_posterior_basic() {
        let mut ldp = super::LogDomainPosterior::new(0.5);
        let initial = ldp.posterior_abandoned();
        assert!((initial - 0.5).abs() < 0.01);

        // Strong evidence toward abandoned
        ldp.update(2.0);
        let updated = ldp.posterior_abandoned();
        assert!(updated > initial);
    }

    #[test]
    fn log_sum_exp_basic() {
        let result = super::log_sum_exp(1.0, 2.0);
        let expected = (1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn calibration_report_perfect_calibration() {
        let mut report = super::CalibrationReport::new(10);
        // All predictions at 0.9, all observed true -> well-calibrated for that bin
        for _ in 0..100 {
            report.observe(0.9, true);
        }
        let ece = report.compute_ece();
        assert!(ece < 0.15, "ECE should be low for well-calibrated: {ece}");
    }

    #[test]
    fn e_process_accumulates_evidence() {
        let mut ep = super::EProcess::new(20.0);
        assert!(!ep.rejected());
        assert_eq!(ep.observations(), 0);

        // Feed strong evidence
        for _ in 0..5 {
            ep.update(2.0);
        }
        assert_eq!(ep.observations(), 5);
        assert!((ep.e_value() - 32.0).abs() < 1e-10);
        assert!(ep.rejected()); // 32 >= 20
    }

    #[test]
    fn e_process_weak_evidence_no_rejection() {
        let mut ep = super::EProcess::new(20.0);
        for _ in 0..10 {
            ep.update(1.0); // neutral evidence
        }
        assert!(!ep.rejected());
        assert!((ep.e_value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn log_domain_posterior_bayes_factor_starts_at_one() {
        let ldp = super::LogDomainPosterior::new(0.5);
        assert!((ldp.bayes_factor() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn log_domain_posterior_bayes_factor_grows_with_evidence() {
        let mut ldp = super::LogDomainPosterior::new(0.5);
        ldp.update(1.0); // log-likelihood = 1.0
        let bf = ldp.bayes_factor();
        assert!((bf - 1.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn log_domain_posterior_extreme_prior_near_zero() {
        let ldp = super::LogDomainPosterior::new(0.001);
        let p = ldp.posterior_abandoned();
        assert!(p < 0.01, "near-zero prior should yield low posterior: {p}");
    }

    #[test]
    fn log_domain_posterior_extreme_prior_near_one() {
        let ldp = super::LogDomainPosterior::new(0.999);
        let p = ldp.posterior_abandoned();
        assert!(p > 0.99, "near-one prior should yield high posterior: {p}");
    }

    #[test]
    fn calibration_report_empty_returns_zero_ece() {
        let report = super::CalibrationReport::new(10);
        assert!((report.compute_ece()).abs() < 1e-10);
    }

    #[test]
    fn calibration_report_poor_calibration_has_high_ece() {
        let mut report = super::CalibrationReport::new(10);
        // Predict 0.9 but never observe true → poor calibration
        for _ in 0..100 {
            report.observe(0.9, false);
        }
        let ece = report.compute_ece();
        assert!(ece > 0.5, "ECE should be high for poor calibration: {ece}");
    }

    #[test]
    fn conformal_predictor_exact_min_size_calibrates() {
        let mut cp = super::ConformalPredictor::new(0.9, 5);
        for i in 1..=5 {
            cp.observe(i as f64 * 0.1);
        }
        assert!(cp.is_calibrated());
        assert!(cp.prediction_threshold().is_some());
    }

    #[test]
    fn log_sum_exp_handles_neg_infinity() {
        let result = super::log_sum_exp(f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert_eq!(result, f64::NEG_INFINITY);
    }

    #[test]
    fn log_sum_exp_one_neg_infinity() {
        let result = super::log_sum_exp(f64::NEG_INFINITY, 0.0);
        assert!((result - 0.0).abs() < 1e-10, "log(0 + 1) = 0: got {result}");
    }

    // ── Extended ledger tests (frankenjax-o9j) ──────────────────

    #[test]
    fn log_domain_posterior_strong_evidence_shifts_belief() {
        let mut posterior = super::LogDomainPosterior::new(0.5);
        assert!((posterior.posterior_abandoned() - 0.5).abs() < 0.01);
        // Positive log-likelihood shifts toward "abandoned"
        posterior.update(2.0);
        assert!(
            posterior.posterior_abandoned() > 0.7,
            "strong positive evidence should increase abandoned posterior: {}",
            posterior.posterior_abandoned()
        );
    }

    #[test]
    fn log_domain_posterior_negative_evidence_shifts_to_useful() {
        let mut posterior = super::LogDomainPosterior::new(0.5);
        // Negative log-likelihood shifts toward "useful"
        posterior.update(-3.0);
        assert!(
            posterior.posterior_abandoned() < 0.1,
            "strong negative evidence should decrease abandoned posterior: {}",
            posterior.posterior_abandoned()
        );
    }

    #[test]
    fn log_domain_posterior_extreme_prior() {
        // Very low prior for abandoned
        let mut posterior = super::LogDomainPosterior::new(0.001);
        assert!(posterior.posterior_abandoned() < 0.01);
        // Even with moderate evidence, prior dominates
        posterior.update(1.0);
        assert!(
            posterior.posterior_abandoned() < 0.01,
            "low prior should resist moderate evidence: {}",
            posterior.posterior_abandoned()
        );
    }

    #[test]
    fn bayes_factor_matches_accumulated_evidence() {
        let mut posterior = super::LogDomainPosterior::new(0.5);
        posterior.update(1.0);
        posterior.update(0.5);
        let bf = posterior.bayes_factor();
        let expected = (1.0_f64 + 0.5).exp();
        assert!(
            (bf - expected).abs() < 1e-10,
            "Bayes factor should be exp(1.5)={expected}, got {bf}"
        );
    }

    #[test]
    fn e_process_rejects_after_strong_evidence() {
        let mut ep = super::EProcess::new(20.0);
        assert!(!ep.rejected());
        assert_eq!(ep.observations(), 0);
        // Each observation with LR=3 accumulates: 1*3=3, 3*3=9, 9*3=27>20
        ep.update(3.0);
        assert_eq!(ep.observations(), 1);
        assert!(!ep.rejected());
        ep.update(3.0);
        assert!(!ep.rejected());
        ep.update(3.0);
        assert!(
            ep.rejected(),
            "e-value={} should exceed threshold 20",
            ep.e_value()
        );
    }

    #[test]
    fn e_process_never_negative() {
        let mut ep = super::EProcess::new(10.0);
        ep.update(0.0); // LR=0 should make e-value 0, not negative
        assert!(ep.e_value() >= 0.0);
        ep.update(-1.0); // Negative LR clamped to 0
        assert!(ep.e_value() >= 0.0);
    }

    #[test]
    fn conformal_predictor_uncalibrated_uses_heuristic() {
        let cp = super::ConformalPredictor::new(0.9, 10);
        assert!(!cp.is_calibrated());
        let est = cp.calibrated_posterior(0.6);
        assert!(!est.used_conformal);
        assert_eq!(est.coverage, 0.0);
        // Heuristic: lower=0.3, upper=0.9
        assert!((est.lower - 0.3).abs() < 1e-10);
        assert!((est.upper - 0.9).abs() < 1e-10);
    }

    #[test]
    fn conformal_predictor_calibrated_uses_conformal_bounds() {
        let mut cp = super::ConformalPredictor::new(0.9, 5);
        for i in 1..=10 {
            cp.observe(i as f64 * 0.01);
        }
        assert!(cp.is_calibrated());
        let est = cp.calibrated_posterior(0.5);
        assert!(est.used_conformal);
        assert!(est.coverage > 0.0);
        assert!(est.lower <= est.point);
        assert!(est.upper >= est.point);
    }

    #[test]
    fn ledger_entry_serialization_roundtrip() {
        use super::LedgerEntry;

        let entry = LedgerEntry {
            decision_id: "test-001".to_owned(),
            record: DecisionRecord::from_posterior(
                CompatibilityMode::Strict,
                0.3,
                &LossMatrix::default(),
            ),
            signals: vec![super::EvidenceSignal {
                signal_name: "cache_hit_rate".to_owned(),
                log_likelihood_delta: -0.5,
                detail: "80% hit rate".to_owned(),
            }],
        };

        let json = serde_json::to_string(&entry).expect("serialize");
        let deserialized: LedgerEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(entry.decision_id, deserialized.decision_id);
        assert_eq!(entry.record.action, deserialized.record.action);
        assert_eq!(entry.signals.len(), deserialized.signals.len());
    }

    #[test]
    fn ledger_audit_trail_ordering() {
        let mut ledger = EvidenceLedger::new();
        assert!(ledger.is_empty());

        for i in 0..5 {
            ledger.append(super::LedgerEntry {
                decision_id: format!("decision-{i}"),
                record: DecisionRecord::from_posterior(
                    CompatibilityMode::Strict,
                    i as f64 * 0.2,
                    &LossMatrix::default(),
                ),
                signals: vec![],
            });
        }

        assert_eq!(ledger.len(), 5);
        // Entries should be in insertion order
        for (i, entry) in ledger.entries().iter().enumerate() {
            assert_eq!(entry.decision_id, format!("decision-{i}"));
        }
    }

    #[test]
    fn loss_matrix_boundary_posterior_zero() {
        // p(abandoned) = 0: always useful → Keep
        let matrix = LossMatrix::default();
        let action = recommend_action(0.0, &matrix);
        assert_eq!(action, DecisionAction::Keep);
    }

    #[test]
    fn loss_matrix_boundary_posterior_one() {
        // p(abandoned) = 1: certainly abandoned → Kill
        let matrix = LossMatrix::default();
        let action = recommend_action(1.0, &matrix);
        assert_eq!(action, DecisionAction::Kill);
    }

    #[test]
    fn decision_record_uses_reprofile_on_exact_loss_tie() {
        let matrix = LossMatrix {
            keep_if_useful: 2,
            kill_if_useful: 8,
            keep_if_abandoned: 8,
            kill_if_abandoned: 2,
        };

        let record = DecisionRecord::from_posterior(CompatibilityMode::Hardened, 0.5, &matrix);

        assert!((record.expected_loss_keep - 5.0).abs() < 1e-10);
        assert!((record.expected_loss_kill - 5.0).abs() < 1e-10);
        assert_eq!(record.action, DecisionAction::Reprofile);
        assert_eq!(record.mode, CompatibilityMode::Hardened);
    }

    #[test]
    fn ledger_roundtrip_preserves_full_audit_trail() {
        use super::{EvidenceSignal, LedgerEntry};

        let mut ledger = EvidenceLedger::new();
        for (idx, posterior) in [0.2, 0.6].into_iter().enumerate() {
            ledger.append(LedgerEntry {
                decision_id: format!("audit-{idx}"),
                record: DecisionRecord::from_posterior(
                    CompatibilityMode::Strict,
                    posterior,
                    &LossMatrix::default(),
                ),
                signals: vec![
                    EvidenceSignal {
                        signal_name: format!("signal-{idx}"),
                        log_likelihood_delta: posterior - 0.5,
                        detail: format!("detail-{idx}"),
                    },
                    EvidenceSignal {
                        signal_name: format!("signal-{idx}-confirm"),
                        log_likelihood_delta: 0.25,
                        detail: "secondary corroboration".to_owned(),
                    },
                ],
            });
        }

        let json = serde_json::to_string(&ledger).expect("serialize ledger");
        let decoded: EvidenceLedger = serde_json::from_str(&json).expect("deserialize ledger");

        assert_eq!(decoded.entries(), ledger.entries());
        assert_eq!(
            decoded.entries()[1].signals[1].detail,
            "secondary corroboration"
        );
    }

    #[test]
    fn ledger_entries_expose_signal_payloads_without_reordering() {
        use super::{EvidenceSignal, LedgerEntry};

        let mut ledger = EvidenceLedger::new();
        ledger.append(LedgerEntry {
            decision_id: "audit-primary".to_owned(),
            record: DecisionRecord::from_posterior(
                CompatibilityMode::Strict,
                0.15,
                &LossMatrix::default(),
            ),
            signals: vec![
                EvidenceSignal {
                    signal_name: "cache_pressure".to_owned(),
                    log_likelihood_delta: 0.4,
                    detail: "cache pressure rising".to_owned(),
                },
                EvidenceSignal {
                    signal_name: "manual_override".to_owned(),
                    log_likelihood_delta: -0.2,
                    detail: "operator inspected shard".to_owned(),
                },
            ],
        });

        let entry = &ledger.entries()[0];
        assert_eq!(entry.decision_id, "audit-primary");
        assert_eq!(entry.signals[0].signal_name, "cache_pressure");
        assert_eq!(entry.signals[1].detail, "operator inspected shard");
    }

    #[test]
    fn loss_matrix_custom_values_shift_decision_boundary() {
        // Custom matrix where Kill is cheaper even at moderate abandonment
        let aggressive_matrix = LossMatrix {
            keep_if_useful: 10,
            kill_if_useful: 20,
            keep_if_abandoned: 80,
            kill_if_abandoned: 5,
        };
        // At p(abandoned)=0.3: keep_loss = 0.7*10 + 0.3*80 = 31, kill_loss = 0.7*20 + 0.3*5 = 15.5
        assert_eq!(
            recommend_action(0.3, &aggressive_matrix),
            DecisionAction::Kill
        );
        // At p(abandoned)=0.05: keep_loss = 0.95*10 + 0.05*80 = 13.5, kill_loss = 0.95*20 + 0.05*5 = 19.25
        assert_eq!(
            recommend_action(0.05, &aggressive_matrix),
            DecisionAction::Keep
        );
    }

    #[test]
    fn loss_matrix_symmetric_always_reprofiles() {
        let symmetric = LossMatrix {
            keep_if_useful: 10,
            kill_if_useful: 10,
            keep_if_abandoned: 10,
            kill_if_abandoned: 10,
        };
        // With symmetric costs, expected losses are always equal → Reprofile
        assert_eq!(recommend_action(0.0, &symmetric), DecisionAction::Reprofile);
        assert_eq!(recommend_action(0.5, &symmetric), DecisionAction::Reprofile);
        assert_eq!(recommend_action(1.0, &symmetric), DecisionAction::Reprofile);
    }

    #[test]
    fn decision_record_strict_vs_hardened_mode_preserved() {
        let matrix = LossMatrix::default();
        let strict = DecisionRecord::from_posterior(CompatibilityMode::Strict, 0.3, &matrix);
        let hardened = DecisionRecord::from_posterior(CompatibilityMode::Hardened, 0.3, &matrix);

        assert_eq!(strict.mode, CompatibilityMode::Strict);
        assert_eq!(hardened.mode, CompatibilityMode::Hardened);
        // Same posterior + matrix → same action regardless of mode
        assert_eq!(strict.action, hardened.action);
        assert!((strict.expected_loss_keep - hardened.expected_loss_keep).abs() < 1e-10);
    }

    #[test]
    fn log_domain_posterior_multiple_updates_accumulate() {
        let mut posterior = super::LogDomainPosterior::new(0.5);
        // Multiple small updates should accumulate
        for _ in 0..10 {
            posterior.update(0.5);
        }
        let bf = posterior.bayes_factor();
        let expected = (5.0_f64).exp(); // 10 * 0.5 = 5.0
        assert!(
            (bf - expected).abs() < 1e-6,
            "10 updates of 0.5 should give exp(5.0), got {bf}"
        );
        // Posterior should be heavily shifted toward abandoned
        assert!(posterior.posterior_abandoned() > 0.99);
    }

    #[test]
    fn evidence_signal_negative_delta_favors_useful() {
        use super::EvidenceSignal;
        let signal = EvidenceSignal {
            signal_name: "high_cache_hit".to_owned(),
            log_likelihood_delta: -2.0,
            detail: "95% cache hit rate suggests active usage".to_owned(),
        };
        assert!(signal.log_likelihood_delta < 0.0);

        let mut posterior = super::LogDomainPosterior::new(0.5);
        posterior.update(signal.log_likelihood_delta);
        assert!(
            posterior.posterior_abandoned() < 0.2,
            "negative evidence should shift toward useful: {}",
            posterior.posterior_abandoned()
        );
    }

    #[test]
    fn e_process_zero_lr_absorbs() {
        let mut ep = super::EProcess::new(10.0);
        ep.update(5.0); // e_value = 5
        ep.update(0.0); // LR=0 → e_value = 0 (absorbing state)
        assert!((ep.e_value()).abs() < 1e-10);
        // Once absorbed, can never reject
        ep.update(1000.0);
        assert!((ep.e_value()).abs() < 1e-10);
        assert!(!ep.rejected());
    }

    #[test]
    fn calibration_report_mixed_predictions() {
        let mut report = super::CalibrationReport::new(10);
        // Well-calibrated: predict 0.8, observe true 80% of the time
        for i in 0..100 {
            report.observe(0.8, i < 80);
        }
        let ece = report.compute_ece();
        assert!(
            ece < 0.05,
            "well-calibrated predictions should have low ECE: {ece}"
        );
    }

    #[test]
    fn test_ledger_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("ledger", "loss-matrix")).expect("digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_ledger_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ── Metamorphic tests (frankenjax-meta) ──────────────────

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn metamorphic_log_sum_exp_commutativity(a in -10.0_f64..10.0, b in -10.0_f64..10.0) {
            let ab = super::log_sum_exp(a, b);
            let ba = super::log_sum_exp(b, a);
            prop_assert!((ab - ba).abs() < 1e-10, "log_sum_exp should be commutative");
        }

        #[test]
        fn metamorphic_log_sum_exp_matches_naive(a in -5.0_f64..5.0, b in -5.0_f64..5.0) {
            let stable = super::log_sum_exp(a, b);
            let naive = (a.exp() + b.exp()).ln();
            prop_assert!((stable - naive).abs() < 1e-9, "log_sum_exp should match naive: stable={}, naive={}", stable, naive);
        }

        #[test]
        fn metamorphic_expected_loss_boundary_zero(
            keep_if_useful in 0u32..100,
            kill_if_useful in 0u32..100,
            keep_if_abandoned in 0u32..100,
            kill_if_abandoned in 0u32..100,
        ) {
            let matrix = LossMatrix { keep_if_useful, kill_if_useful, keep_if_abandoned, kill_if_abandoned };
            let loss_keep = super::expected_loss_keep(0.0, &matrix);
            let loss_kill = super::expected_loss_kill(0.0, &matrix);
            prop_assert!((loss_keep - f64::from(keep_if_useful)).abs() < 1e-10);
            prop_assert!((loss_kill - f64::from(kill_if_useful)).abs() < 1e-10);
        }

        #[test]
        fn metamorphic_expected_loss_boundary_one(
            keep_if_useful in 0u32..100,
            kill_if_useful in 0u32..100,
            keep_if_abandoned in 0u32..100,
            kill_if_abandoned in 0u32..100,
        ) {
            let matrix = LossMatrix { keep_if_useful, kill_if_useful, keep_if_abandoned, kill_if_abandoned };
            let loss_keep = super::expected_loss_keep(1.0, &matrix);
            let loss_kill = super::expected_loss_kill(1.0, &matrix);
            prop_assert!((loss_keep - f64::from(keep_if_abandoned)).abs() < 1e-10);
            prop_assert!((loss_kill - f64::from(kill_if_abandoned)).abs() < 1e-10);
        }

        #[test]
        fn metamorphic_e_process_product(ratios in prop::collection::vec(0.5_f64..3.0, 1..10)) {
            let mut ep = super::EProcess::new(1e10);
            let expected_product: f64 = ratios.iter().copied().product();
            for lr in &ratios {
                ep.update(*lr);
            }
            prop_assert!((ep.e_value() - expected_product).abs() < 1e-6, "e_value should equal product of LRs");
        }

        #[test]
        fn metamorphic_conformal_bounds_containment(
            point in 0.0_f64..1.0,
            scores in prop::collection::vec(0.01_f64..0.5, 10..20),
        ) {
            let mut cp = super::ConformalPredictor::new(0.9, 5);
            for s in scores {
                cp.observe(s);
            }
            let est = cp.calibrated_posterior(point);
            prop_assert!(est.lower <= est.point, "lower bound should not exceed point");
            prop_assert!(est.upper >= est.point, "upper bound should not be below point");
            prop_assert!(est.lower >= 0.0 && est.upper <= 1.0, "bounds should be in [0,1]");
        }

        #[test]
        fn metamorphic_posterior_prior_recovery(prior in 0.01_f64..0.99) {
            let ldp = super::LogDomainPosterior::new(prior);
            let recovered = ldp.posterior_abandoned();
            prop_assert!((recovered - prior).abs() < 0.01, "no evidence should recover prior: expected {}, got {}", prior, recovered);
        }
    }
}
