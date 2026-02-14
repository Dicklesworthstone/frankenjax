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
        let idx = idx.min(n) - 1;
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
}
