#![forbid(unsafe_code)]

pub mod architecture_decision;
pub mod cache_lifecycle;
pub mod durability;
pub mod e2e_log;
pub mod error_taxonomy;
pub mod memory_performance;
pub mod onboarding_command;
pub mod oracle_recapture;
pub mod security_adversarial;
pub mod transform_control_flow;
pub mod ttl_semantic;

use fj_core::{
    CompatibilityMode, ProgramSpec, TraceTransformLedger, Transform, Value, build_program,
};
use fj_dispatch::{DispatchRequest, dispatch};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub oracle_root: PathBuf,
    pub fixture_root: PathBuf,
    pub strict_mode: bool,
}

impl HarnessConfig {
    #[must_use]
    pub fn default_paths() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            oracle_root: repo_root.join("legacy_jax_code/jax"),
            fixture_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"),
            strict_mode: true,
        }
    }

    #[must_use]
    pub fn mode(&self) -> CompatibilityMode {
        if self.strict_mode {
            CompatibilityMode::Strict
        } else {
            CompatibilityMode::Hardened
        }
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self::default_paths()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureFamily {
    Jit,
    Grad,
    Vmap,
    Lax,
    Random,
    ControlFlow,
    MixedDtype,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureManifestEntry {
    pub family: FixtureFamily,
    pub path_pattern: String,
    pub representative_test: String,
}

#[must_use]
pub fn default_fixture_manifest() -> Vec<FixtureManifestEntry> {
    vec![
        FixtureManifestEntry {
            family: FixtureFamily::Jit,
            path_pattern: "tests/jax_jit_test.py".to_owned(),
            representative_test: "JaxJitTest".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Grad,
            path_pattern: "tests/lax_autodiff_test.py".to_owned(),
            representative_test: "LAX_GRAD_OPS".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Vmap,
            path_pattern: "tests/lax_vmap_test.py".to_owned(),
            representative_test: "LaxVmapTest::_CheckBatching".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Lax,
            path_pattern: "tests/lax_test.py".to_owned(),
            representative_test: "LaxTest::testOp".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Random,
            path_pattern: "tests/random_test.py".to_owned(),
            representative_test: "RandomValuesCase".to_owned(),
        },
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessReport {
    pub suite: &'static str,
    pub oracle_present: bool,
    pub fixture_count: usize,
    pub manifest_family_count: usize,
    pub strict_mode: bool,
}

#[must_use]
pub fn run_smoke(config: &HarnessConfig) -> HarnessReport {
    let fixture_count = collect_json_fixtures(&config.fixture_root).len();
    let _mode = config.mode();

    HarnessReport {
        suite: "smoke",
        oracle_present: config.oracle_root.exists(),
        fixture_count,
        manifest_family_count: default_fixture_manifest().len(),
        strict_mode: config.strict_mode,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityCase {
    pub case_id: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityReport {
    pub total_cases: usize,
    pub matched_cases: usize,
    pub mismatched_cases: usize,
    pub strict_mode: bool,
}

#[must_use]
pub fn evaluate_parity(cases: &[ParityCase], strict_mode: bool) -> ParityReport {
    let matched_cases = cases
        .iter()
        .filter(|case| case.expected == case.actual)
        .count();

    ParityReport {
        total_cases: cases.len(),
        matched_cases,
        mismatched_cases: cases.len().saturating_sub(matched_cases),
        strict_mode,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureNote {
    pub suite: String,
    pub oracle: String,
    pub notes: String,
}

pub fn read_fixture_note(path: &Path) -> Result<FixtureNote, std::io::Error> {
    let raw = fs::read_to_string(path)?;
    let parsed = serde_json::from_str::<FixtureNote>(&raw).map_err(std::io::Error::other)?;
    Ok(parsed)
}

#[must_use]
pub fn collect_json_fixtures(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_json_fixtures_recursive(root, &mut out);
    out.sort();
    out
}

fn collect_json_fixtures_recursive(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_json_fixtures_recursive(&path, out);
            continue;
        }

        if path.extension().is_some_and(|ext| ext == "json") {
            out.push(path);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureMode {
    Strict,
    Hardened,
}

impl FixtureMode {
    #[must_use]
    pub fn as_runtime_mode(self) -> CompatibilityMode {
        match self {
            Self::Strict => CompatibilityMode::Strict,
            Self::Hardened => CompatibilityMode::Hardened,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureProgram {
    Add2,
    Square,
    SquarePlusLinear,
    AddOne,
    SinX,
    CosX,
    Dot3,
    ReduceSumVec,
    // Lax unary primitives
    LaxNeg,
    LaxAbs,
    LaxExp,
    LaxLog,
    LaxSqrt,
    LaxRsqrt,
    LaxFloor,
    LaxCeil,
    LaxRound,
    LaxTan,
    LaxAsin,
    LaxAcos,
    LaxAtan,
    LaxSinh,
    LaxCosh,
    LaxTanh,
    LaxExpm1,
    LaxLog1p,
    LaxSign,
    LaxSquare,
    LaxReciprocal,
    LaxLogistic,
    LaxErf,
    LaxErfc,
    // Lax binary primitives
    LaxSub,
    LaxMul,
    LaxDiv,
    LaxRem,
    LaxPow,
    LaxAtan2,
    LaxMax,
    LaxMin,
    LaxEq,
    LaxNe,
    LaxLt,
    LaxLe,
    LaxGt,
    LaxGe,
    // Lax ternary primitives
    LaxSelect,
    LaxClamp,
    // Lax reduction primitives
    LaxReduceMax,
    LaxReduceMin,
    LaxReduceProd,
    // Lax special math unary primitives
    LaxCbrt,
    LaxLgamma,
    LaxDigamma,
    LaxErfInv,
    LaxIsFinite,
    LaxNextafter,
    // Lax cumulative primitives
    LaxCumsum,
    LaxCumprod,
    // Lax boolean reduction primitives
    LaxReduceAnd,
    LaxReduceOr,
    // Lax bitwise primitives
    LaxBitwiseAnd,
    LaxBitwiseOr,
    LaxBitwiseXor,
    LaxBitwiseNot,
    LaxPopulationCount,
    LaxCountLeadingZeros,
    LaxReduceXor,
    LaxSort,
    LaxIntegerPow2,
    LaxIntegerPow3,
    LaxIntegerPowNeg1,
    LaxReshape6To2x3,
    LaxReshape6To3x2,
    LaxSlice1To4,
    LaxTranspose2x3,
    LaxRev,
    LaxSqueeze,
    LaxConcatenate,
    // Lax structural primitives
    LaxIota5,
    LaxCopy,
    LaxExpandDimsAxis0,
    LaxPadLow1High2,
    LaxBroadcastInDimScalar3,
    // Lax bitwise shift primitives
    LaxShiftLeft,
    LaxShiftRightArithmetic,
    LaxShiftRightLogical,
    // Lax advanced shape/index primitives
    LaxDynamicSlice,
    LaxDynamicUpdateSlice,
    LaxSplit2,
    LaxBroadcastedIota2x3,
    // Lax advanced primitives
    LaxWhileAddLt,
    LaxSwitch3,
    LaxArgsort,
    LaxOneHot4,
    LaxReduceWindowSum,
    // Lax utility/data primitives
    LaxBitcastF64ToI64,
    LaxReducePrecisionF64,
    LaxGather1d,
    // Lax convolution and scatter
    LaxConv1dValid,
    LaxScatterOverwrite,
    // Lax complex number primitives
    LaxComplex,
    LaxConj,
    LaxReal,
    LaxImag,
    // Linalg primitives
    LaxCholesky,
    LaxTriangularSolve,
    LaxQr,
    LaxSvd,
    LaxEigh,
    // FFT primitives
    LaxFft,
    LaxIfft,
    LaxRfft,
    LaxIrfft,
    // Utility programs
    Identity,
    AddOneMulTwo,
    // Control flow programs
    CondSelect,
    ScanAdd,
}

impl FixtureProgram {
    #[must_use]
    pub fn as_program_spec(self) -> ProgramSpec {
        match self {
            Self::Add2 => ProgramSpec::Add2,
            Self::Square => ProgramSpec::Square,
            Self::SquarePlusLinear => ProgramSpec::SquarePlusLinear,
            Self::AddOne => ProgramSpec::AddOne,
            Self::SinX => ProgramSpec::SinX,
            Self::CosX => ProgramSpec::CosX,
            Self::Dot3 => ProgramSpec::Dot3,
            Self::ReduceSumVec => ProgramSpec::ReduceSumVec,
            Self::LaxNeg => ProgramSpec::LaxNeg,
            Self::LaxAbs => ProgramSpec::LaxAbs,
            Self::LaxExp => ProgramSpec::LaxExp,
            Self::LaxLog => ProgramSpec::LaxLog,
            Self::LaxSqrt => ProgramSpec::LaxSqrt,
            Self::LaxRsqrt => ProgramSpec::LaxRsqrt,
            Self::LaxFloor => ProgramSpec::LaxFloor,
            Self::LaxCeil => ProgramSpec::LaxCeil,
            Self::LaxRound => ProgramSpec::LaxRound,
            Self::LaxTan => ProgramSpec::LaxTan,
            Self::LaxAsin => ProgramSpec::LaxAsin,
            Self::LaxAcos => ProgramSpec::LaxAcos,
            Self::LaxAtan => ProgramSpec::LaxAtan,
            Self::LaxSinh => ProgramSpec::LaxSinh,
            Self::LaxCosh => ProgramSpec::LaxCosh,
            Self::LaxTanh => ProgramSpec::LaxTanh,
            Self::LaxExpm1 => ProgramSpec::LaxExpm1,
            Self::LaxLog1p => ProgramSpec::LaxLog1p,
            Self::LaxSign => ProgramSpec::LaxSign,
            Self::LaxSquare => ProgramSpec::LaxSquare,
            Self::LaxReciprocal => ProgramSpec::LaxReciprocal,
            Self::LaxLogistic => ProgramSpec::LaxLogistic,
            Self::LaxErf => ProgramSpec::LaxErf,
            Self::LaxErfc => ProgramSpec::LaxErfc,
            Self::LaxSub => ProgramSpec::LaxSub,
            Self::LaxMul => ProgramSpec::LaxMul,
            Self::LaxDiv => ProgramSpec::LaxDiv,
            Self::LaxRem => ProgramSpec::LaxRem,
            Self::LaxPow => ProgramSpec::LaxPow,
            Self::LaxAtan2 => ProgramSpec::LaxAtan2,
            Self::LaxMax => ProgramSpec::LaxMax,
            Self::LaxMin => ProgramSpec::LaxMin,
            Self::LaxEq => ProgramSpec::LaxEq,
            Self::LaxNe => ProgramSpec::LaxNe,
            Self::LaxLt => ProgramSpec::LaxLt,
            Self::LaxLe => ProgramSpec::LaxLe,
            Self::LaxGt => ProgramSpec::LaxGt,
            Self::LaxGe => ProgramSpec::LaxGe,
            Self::LaxSelect => ProgramSpec::LaxSelect,
            Self::LaxClamp => ProgramSpec::LaxClamp,
            Self::LaxReduceMax => ProgramSpec::LaxReduceMax,
            Self::LaxReduceMin => ProgramSpec::LaxReduceMin,
            Self::LaxReduceProd => ProgramSpec::LaxReduceProd,
            Self::LaxCbrt => ProgramSpec::LaxCbrt,
            Self::LaxLgamma => ProgramSpec::LaxLgamma,
            Self::LaxDigamma => ProgramSpec::LaxDigamma,
            Self::LaxErfInv => ProgramSpec::LaxErfInv,
            Self::LaxIsFinite => ProgramSpec::LaxIsFinite,
            Self::LaxNextafter => ProgramSpec::LaxNextafter,
            Self::LaxCumsum => ProgramSpec::LaxCumsum,
            Self::LaxCumprod => ProgramSpec::LaxCumprod,
            Self::LaxReduceAnd => ProgramSpec::LaxReduceAnd,
            Self::LaxReduceOr => ProgramSpec::LaxReduceOr,
            Self::LaxBitwiseAnd => ProgramSpec::LaxBitwiseAnd,
            Self::LaxBitwiseOr => ProgramSpec::LaxBitwiseOr,
            Self::LaxBitwiseXor => ProgramSpec::LaxBitwiseXor,
            Self::LaxBitwiseNot => ProgramSpec::LaxBitwiseNot,
            Self::LaxPopulationCount => ProgramSpec::LaxPopulationCount,
            Self::LaxCountLeadingZeros => ProgramSpec::LaxCountLeadingZeros,
            Self::LaxReduceXor => ProgramSpec::LaxReduceXor,
            Self::LaxSort => ProgramSpec::LaxSort,
            Self::LaxIntegerPow2 => ProgramSpec::LaxIntegerPow2,
            Self::LaxIntegerPow3 => ProgramSpec::LaxIntegerPow3,
            Self::LaxIntegerPowNeg1 => ProgramSpec::LaxIntegerPowNeg1,
            Self::LaxReshape6To2x3 => ProgramSpec::LaxReshape6To2x3,
            Self::LaxReshape6To3x2 => ProgramSpec::LaxReshape6To3x2,
            Self::LaxSlice1To4 => ProgramSpec::LaxSlice1To4,
            Self::LaxTranspose2x3 => ProgramSpec::LaxTranspose2x3,
            Self::LaxRev => ProgramSpec::LaxRev,
            Self::LaxSqueeze => ProgramSpec::LaxSqueeze,
            Self::LaxConcatenate => ProgramSpec::LaxConcatenate,
            Self::LaxIota5 => ProgramSpec::LaxIota5,
            Self::LaxCopy => ProgramSpec::LaxCopy,
            Self::LaxExpandDimsAxis0 => ProgramSpec::LaxExpandDimsAxis0,
            Self::LaxPadLow1High2 => ProgramSpec::LaxPadLow1High2,
            Self::LaxBroadcastInDimScalar3 => ProgramSpec::LaxBroadcastInDimScalar3,
            Self::LaxShiftLeft => ProgramSpec::LaxShiftLeft,
            Self::LaxShiftRightArithmetic => ProgramSpec::LaxShiftRightArithmetic,
            Self::LaxShiftRightLogical => ProgramSpec::LaxShiftRightLogical,
            Self::LaxDynamicSlice => ProgramSpec::LaxDynamicSlice,
            Self::LaxDynamicUpdateSlice => ProgramSpec::LaxDynamicUpdateSlice,
            Self::LaxSplit2 => ProgramSpec::LaxSplit2,
            Self::LaxBroadcastedIota2x3 => ProgramSpec::LaxBroadcastedIota2x3,
            Self::LaxWhileAddLt => ProgramSpec::LaxWhileAddLt,
            Self::LaxSwitch3 => ProgramSpec::LaxSwitch3,
            Self::LaxArgsort => ProgramSpec::LaxArgsort,
            Self::LaxOneHot4 => ProgramSpec::LaxOneHot4,
            Self::LaxReduceWindowSum => ProgramSpec::LaxReduceWindowSum,
            Self::LaxBitcastF64ToI64 => ProgramSpec::LaxBitcastF64ToI64,
            Self::LaxReducePrecisionF64 => ProgramSpec::LaxReducePrecisionF64,
            Self::LaxGather1d => ProgramSpec::LaxGather1d,
            Self::LaxConv1dValid => ProgramSpec::LaxConv1dValid,
            Self::LaxScatterOverwrite => ProgramSpec::LaxScatterOverwrite,
            Self::LaxComplex => ProgramSpec::LaxComplex,
            Self::LaxConj => ProgramSpec::LaxConj,
            Self::LaxReal => ProgramSpec::LaxReal,
            Self::LaxImag => ProgramSpec::LaxImag,
            Self::LaxCholesky => ProgramSpec::LaxCholesky,
            Self::LaxTriangularSolve => ProgramSpec::LaxTriangularSolve,
            Self::LaxQr => ProgramSpec::LaxQr,
            Self::LaxSvd => ProgramSpec::LaxSvd,
            Self::LaxEigh => ProgramSpec::LaxEigh,
            Self::LaxFft => ProgramSpec::LaxFft,
            Self::LaxIfft => ProgramSpec::LaxIfft,
            Self::LaxRfft => ProgramSpec::LaxRfft,
            Self::LaxIrfft => ProgramSpec::LaxIrfft,
            Self::Identity => ProgramSpec::Identity,
            Self::AddOneMulTwo => ProgramSpec::AddOneMulTwo,
            Self::CondSelect => ProgramSpec::CondSelect,
            Self::ScanAdd => ProgramSpec::ScanAdd,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureTransform {
    Jit,
    Grad,
    Vmap,
}

impl FixtureTransform {
    #[must_use]
    pub fn as_runtime_transform(self) -> Transform {
        match self {
            Self::Jit => Transform::Jit,
            Self::Grad => Transform::Grad,
            Self::Vmap => Transform::Vmap,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FixtureValue {
    ScalarF64 {
        value: f64,
    },
    ScalarI64 {
        value: i64,
    },
    ScalarBool {
        value: bool,
    },
    ScalarComplex128 {
        real: f64,
        imag: f64,
    },
    VectorF64 {
        values: Vec<f64>,
    },
    VectorI64 {
        values: Vec<i64>,
    },
    TensorF64 {
        shape: Vec<u32>,
        values: Vec<f64>,
    },
    TensorI64 {
        shape: Vec<u32>,
        values: Vec<i64>,
    },
    TensorBool {
        shape: Vec<u32>,
        values: Vec<bool>,
    },
    TensorComplex128 {
        shape: Vec<u32>,
        reals: Vec<f64>,
        imags: Vec<f64>,
    },
}

impl FixtureValue {
    pub fn to_runtime_value(&self) -> Result<Value, String> {
        match self {
            Self::ScalarF64 { value } => Ok(Value::scalar_f64(*value)),
            Self::ScalarI64 { value } => Ok(Value::scalar_i64(*value)),
            Self::ScalarBool { value } => Ok(Value::scalar_bool(*value)),
            Self::ScalarComplex128 { real, imag } => Ok(Value::scalar_complex128(*real, *imag)),
            Self::VectorF64 { values } => Value::vector_f64(values)
                .map_err(|err| format!("vector_f64 conversion failed: {err}")),
            Self::VectorI64 { values } => Value::vector_i64(values)
                .map_err(|err| format!("vector_i64 conversion failed: {err}")),
            Self::TensorF64 { shape, values } => {
                let elements: Vec<fj_core::Literal> = values
                    .iter()
                    .copied()
                    .map(fj_core::Literal::from_f64)
                    .collect();
                let s = fj_core::Shape {
                    dims: shape.clone(),
                };
                fj_core::TensorValue::new(fj_core::DType::F64, s, elements)
                    .map(Value::Tensor)
                    .map_err(|e| format!("tensor_f64 conversion failed: {e}"))
            }
            Self::TensorI64 { shape, values } => {
                let elements: Vec<fj_core::Literal> =
                    values.iter().copied().map(fj_core::Literal::I64).collect();
                let s = fj_core::Shape {
                    dims: shape.clone(),
                };
                fj_core::TensorValue::new(fj_core::DType::I64, s, elements)
                    .map(Value::Tensor)
                    .map_err(|e| format!("tensor_i64 conversion failed: {e}"))
            }
            Self::TensorBool { shape, values } => {
                let elements: Vec<fj_core::Literal> =
                    values.iter().copied().map(fj_core::Literal::Bool).collect();
                let s = fj_core::Shape {
                    dims: shape.clone(),
                };
                fj_core::TensorValue::new(fj_core::DType::Bool, s, elements)
                    .map(Value::Tensor)
                    .map_err(|e| format!("tensor_bool conversion failed: {e}"))
            }
            Self::TensorComplex128 {
                shape,
                reals,
                imags,
            } => {
                if reals.len() != imags.len() {
                    return Err(format!(
                        "tensor_complex128: reals ({}) and imags ({}) length mismatch",
                        reals.len(),
                        imags.len()
                    ));
                }
                let elements: Vec<fj_core::Literal> = reals
                    .iter()
                    .zip(imags.iter())
                    .map(|(&re, &im)| fj_core::Literal::from_complex128(re, im))
                    .collect();
                let s = fj_core::Shape {
                    dims: shape.clone(),
                };
                fj_core::TensorValue::new(fj_core::DType::Complex128, s, elements)
                    .map(Value::Tensor)
                    .map_err(|e| format!("tensor_complex128 conversion failed: {e}"))
            }
        }
    }

    pub fn approx_matches(&self, actual: &Value, atol: f64, rtol: f64) -> bool {
        match self {
            Self::ScalarF64 { value } => actual
                .as_f64_scalar()
                .is_some_and(|actual_value| approx_equal(*value, actual_value, atol, rtol)),
            Self::ScalarI64 { value } => actual
                .as_scalar_literal()
                .and_then(fj_core::Literal::as_i64)
                .is_some_and(|actual_value| actual_value == *value),
            Self::ScalarBool { value } => actual
                .as_bool_scalar()
                .is_some_and(|actual_value| actual_value == *value),
            Self::ScalarComplex128 { real, imag } => actual
                .as_scalar_literal()
                .and_then(fj_core::Literal::as_complex128)
                .is_some_and(|(re, im)| {
                    approx_equal(*real, re, atol, rtol) && approx_equal(*imag, im, atol, rtol)
                }),
            Self::VectorF64 { values } => actual
                .as_tensor()
                .and_then(|tensor| tensor.to_f64_vec())
                .is_some_and(|actual_values| {
                    values.len() == actual_values.len()
                        && values
                            .iter()
                            .zip(actual_values.iter())
                            .all(|(expected, actual)| approx_equal(*expected, *actual, atol, rtol))
                }),
            Self::VectorI64 { values } => actual.as_tensor().is_some_and(|tensor| {
                if tensor.elements.len() != values.len() {
                    return false;
                }
                tensor
                    .elements
                    .iter()
                    .zip(values.iter())
                    .all(|(actual, expected)| actual.as_i64().is_some_and(|v| v == *expected))
            }),
            Self::TensorF64 { shape, values } => actual.as_tensor().is_some_and(|tensor| {
                tensor.shape.dims == *shape
                    && tensor.to_f64_vec().is_some_and(|actual_values| {
                        values.len() == actual_values.len()
                            && values
                                .iter()
                                .zip(actual_values.iter())
                                .all(|(e, a)| approx_equal(*e, *a, atol, rtol))
                    })
            }),
            Self::TensorI64 { shape, values } => actual.as_tensor().is_some_and(|tensor| {
                tensor.shape.dims == *shape
                    && tensor.elements.len() == values.len()
                    && tensor
                        .elements
                        .iter()
                        .zip(values.iter())
                        .all(|(a, e)| a.as_i64().is_some_and(|v| v == *e))
            }),
            Self::TensorBool { shape, values } => actual.as_tensor().is_some_and(|tensor| {
                tensor.shape.dims == *shape
                    && tensor.elements.len() == values.len()
                    && tensor
                        .elements
                        .iter()
                        .zip(values.iter())
                        .all(|(a, e)| matches!(a, fj_core::Literal::Bool(b) if *b == *e))
            }),
            Self::TensorComplex128 {
                shape,
                reals,
                imags,
            } => actual.as_tensor().is_some_and(|tensor| {
                tensor.shape.dims == *shape
                    && tensor.elements.len() == reals.len()
                    && tensor
                        .elements
                        .iter()
                        .zip(reals.iter().zip(imags.iter()))
                        .all(|(a, (&exp_re, &exp_im))| {
                            a.as_complex128().is_some_and(|(act_re, act_im)| {
                                approx_equal(exp_re, act_re, atol, rtol)
                                    && approx_equal(exp_im, act_im, atol, rtol)
                            })
                        })
            }),
        }
    }

    /// Return the rank of this fixture value.
    #[must_use]
    pub fn rank(&self) -> usize {
        match self {
            Self::ScalarF64 { .. }
            | Self::ScalarI64 { .. }
            | Self::ScalarBool { .. }
            | Self::ScalarComplex128 { .. } => 0,
            Self::VectorF64 { .. } | Self::VectorI64 { .. } => 1,
            Self::TensorF64 { shape, .. }
            | Self::TensorI64 { shape, .. }
            | Self::TensorBool { shape, .. }
            | Self::TensorComplex128 { shape, .. } => shape.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformFixtureCase {
    pub case_id: String,
    pub family: FixtureFamily,
    pub mode: FixtureMode,
    pub program: FixtureProgram,
    pub transforms: Vec<FixtureTransform>,
    #[serde(default = "default_comparator_kind")]
    pub comparator: ComparatorKind,
    #[serde(default)]
    pub baseline_mismatch: bool,
    #[serde(default)]
    pub flaky: bool,
    #[serde(default)]
    pub simulated_delay_ms: u64,
    pub args: Vec<FixtureValue>,
    pub expected: Vec<FixtureValue>,
    pub atol: f64,
    pub rtol: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformFixtureBundle {
    pub schema_version: String,
    pub generated_by: String,
    pub generated_at_unix_ms: u128,
    pub cases: Vec<TransformFixtureCase>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformCaseReport {
    pub case_id: String,
    pub family: FixtureFamily,
    pub mode: FixtureMode,
    pub comparator: ComparatorKind,
    pub drift_classification: DriftClassification,
    pub matched: bool,
    pub expected_json: String,
    pub actual_json: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformParityReport {
    pub schema_version: String,
    pub total_cases: usize,
    pub matched_cases: usize,
    pub mismatched_cases: usize,
    pub reports: Vec<TransformCaseReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchRunnerConfig {
    pub case_timeout: Duration,
}

impl Default for BatchRunnerConfig {
    fn default() -> Self {
        Self {
            case_timeout: default_case_timeout(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleCaptureRequest {
    pub legacy_root: PathBuf,
    pub output_path: PathBuf,
    pub strict: bool,
    pub python_bin: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OracleCaptureResult {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub bundle: TransformFixtureBundle,
}

#[derive(Debug)]
pub enum OracleCaptureError {
    Io(std::io::Error),
    ScriptFailed {
        status: Option<i32>,
        stdout: String,
        stderr: String,
    },
}

impl std::fmt::Display for OracleCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "oracle capture io error: {err}"),
            Self::ScriptFailed {
                status,
                stdout,
                stderr,
            } => write!(
                f,
                "oracle capture script failed (status={status:?})\nstdout:\n{stdout}\nstderr:\n{stderr}"
            ),
        }
    }
}

impl std::error::Error for OracleCaptureError {}

impl From<std::io::Error> for OracleCaptureError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparatorKind {
    Exact,
    ApproxAtolRtol,
    ShapeOnly,
    TypeOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftClassification {
    Pass,
    Regression,
    Improvement,
    Flake,
    Timeout,
}

const fn default_comparator_kind() -> ComparatorKind {
    ComparatorKind::ApproxAtolRtol
}

// ── Per-dtype tolerance tiers ──────────────────────────────────────

/// Per-dtype tolerance tier for numerical comparisons.
///
/// Follows JAX's tolerance conventions:
/// - F64: tight (1e-12 / 1e-10) since double-precision is highly reproducible
/// - F32: moderate (1e-5 / 1e-5) matching JAX's default for single-precision
/// - I32/I64/U32/U64/Bool: exact (0.0 / 0.0) since integer ops are deterministic
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ToleranceTier {
    pub atol: f64,
    pub rtol: f64,
}

impl ToleranceTier {
    #[must_use]
    pub const fn exact() -> Self {
        Self {
            atol: 0.0,
            rtol: 0.0,
        }
    }

    #[must_use]
    pub const fn for_dtype(dtype: fj_core::DType) -> Self {
        use fj_core::DType;
        match dtype {
            DType::F64 | DType::Complex128 => Self {
                atol: 1e-12,
                rtol: 1e-10,
            },
            DType::F32 | DType::BF16 | DType::F16 | DType::Complex64 => Self {
                atol: 1e-5,
                rtol: 1e-5,
            },
            DType::I32 | DType::I64 | DType::U32 | DType::U64 | DType::Bool => Self::exact(),
        }
    }

    /// Returns the effective tolerance, taking the wider of the dtype default
    /// and the fixture-specified override.
    #[must_use]
    pub fn resolve(dtype: fj_core::DType, fixture_atol: f64, fixture_rtol: f64) -> Self {
        let tier = Self::for_dtype(dtype);
        Self {
            atol: tier.atol.max(fixture_atol),
            rtol: tier.rtol.max(fixture_rtol),
        }
    }
}

// ── Tolerance violation reporting ──────────────────────────────────

/// A single tolerance violation for reporting purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceViolation {
    pub case_id: String,
    pub output_index: usize,
    pub element_index: usize,
    pub expected: f64,
    pub actual: f64,
    pub abs_diff: f64,
    pub tolerance_used: f64,
}

/// Error distribution histogram bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub count: usize,
}

/// Summary of tolerance violations for a fixture bundle run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceReport {
    pub total_comparisons: usize,
    pub violations: Vec<ToleranceViolation>,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub error_histogram: Vec<HistogramBucket>,
    error_sum: f64,
}

impl ToleranceReport {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_comparisons: 0,
            violations: Vec::new(),
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            error_histogram: Vec::new(),
            error_sum: 0.0,
        }
    }

    /// Record a comparison result.
    pub fn record(
        &mut self,
        case_id: &str,
        output_idx: usize,
        elem_idx: usize,
        expected: f64,
        actual: f64,
        tolerance: f64,
    ) {
        self.total_comparisons += 1;
        let abs_diff = (expected - actual).abs();
        self.error_sum += abs_diff;
        if abs_diff > self.max_abs_error {
            self.max_abs_error = abs_diff;
        }
        if abs_diff > tolerance {
            self.violations.push(ToleranceViolation {
                case_id: case_id.to_owned(),
                output_index: output_idx,
                element_index: elem_idx,
                expected,
                actual,
                abs_diff,
                tolerance_used: tolerance,
            });
        }
    }

    /// Finalize the report: compute mean error and build histogram.
    pub fn finalize(&mut self, error_sum: f64) {
        let total_sum = if error_sum > 0.0 {
            error_sum
        } else {
            self.error_sum
        };
        if self.total_comparisons > 0 {
            self.mean_abs_error = total_sum / self.total_comparisons as f64;
        }
    }

    /// Build the error distribution histogram from recorded violations
    /// and all comparisons. Call after all `record()` calls are done.
    pub fn build_histogram(&mut self) {
        // Log-scale buckets: [0, 1e-15), [1e-15, 1e-12), [1e-12, 1e-9),
        // [1e-9, 1e-6), [1e-6, 1e-3), [1e-3, 1e0), [1e0, inf)
        let boundaries: &[(f64, f64)] = &[
            (0.0, 1e-15),
            (1e-15, 1e-12),
            (1e-12, 1e-9),
            (1e-9, 1e-6),
            (1e-6, 1e-3),
            (1e-3, 1.0),
            (1.0, f64::INFINITY),
        ];

        self.error_histogram = boundaries
            .iter()
            .map(|&(lo, hi)| HistogramBucket {
                lower_bound: lo,
                upper_bound: hi,
                count: 0,
            })
            .collect();

        // Count violations into buckets
        for v in &self.violations {
            for bucket in &mut self.error_histogram {
                if v.abs_diff >= bucket.lower_bound && v.abs_diff < bucket.upper_bound {
                    bucket.count += 1;
                    break;
                }
            }
        }
    }

    /// Whether all comparisons passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.violations.is_empty()
    }
}

impl Default for ToleranceReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Infer the FixtureValue's dtype for tolerance tier selection.
#[must_use]
pub fn fixture_value_dtype(value: &FixtureValue) -> fj_core::DType {
    match value {
        FixtureValue::ScalarF64 { .. }
        | FixtureValue::VectorF64 { .. }
        | FixtureValue::TensorF64 { .. } => fj_core::DType::F64,
        FixtureValue::ScalarI64 { .. }
        | FixtureValue::VectorI64 { .. }
        | FixtureValue::TensorI64 { .. } => fj_core::DType::I64,
        FixtureValue::ScalarBool { .. } | FixtureValue::TensorBool { .. } => fj_core::DType::Bool,
        FixtureValue::ScalarComplex128 { .. } | FixtureValue::TensorComplex128 { .. } => {
            fj_core::DType::Complex128
        }
    }
}

const fn default_case_timeout() -> Duration {
    Duration::from_secs(2)
}

const MAX_EXTERNAL_TRANSFORM_FIXTURE_CASES: usize = 4096;

pub fn read_transform_fixture_bundle(
    path: &Path,
) -> Result<TransformFixtureBundle, std::io::Error> {
    let raw = fs::read_to_string(path)?;
    let parsed =
        serde_json::from_str::<TransformFixtureBundle>(&raw).map_err(std::io::Error::other)?;
    validate_external_transform_fixture_bundle(&parsed)?;
    Ok(parsed)
}

fn validate_external_transform_fixture_bundle(
    bundle: &TransformFixtureBundle,
) -> Result<(), std::io::Error> {
    if bundle.cases.len() > MAX_EXTERNAL_TRANSFORM_FIXTURE_CASES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "too many transform fixture cases: {} > {}",
                bundle.cases.len(),
                MAX_EXTERNAL_TRANSFORM_FIXTURE_CASES
            ),
        ));
    }

    if let Some(case) = bundle
        .cases
        .iter()
        .find(|case| case.simulated_delay_ms != 0)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "external transform fixture case {} sets simulated_delay_ms={}; simulated delays are test-only",
                case.case_id, case.simulated_delay_ms
            ),
        ));
    }

    Ok(())
}

#[must_use]
pub fn run_transform_fixture_bundle(
    _config: &HarnessConfig,
    bundle: &TransformFixtureBundle,
) -> TransformParityReport {
    let mut reports = Vec::with_capacity(bundle.cases.len());

    for case in &bundle.cases {
        reports.push(run_transform_fixture_case(case));
    }

    let matched_cases = reports.iter().filter(|report| report.matched).count();

    TransformParityReport {
        schema_version: "frankenjax.transform-parity-report.v1".to_owned(),
        total_cases: reports.len(),
        matched_cases,
        mismatched_cases: reports.len().saturating_sub(matched_cases),
        reports,
    }
}

pub fn capture_transform_fixture_bundle_with_oracle(
    request: &OracleCaptureRequest,
) -> Result<OracleCaptureResult, OracleCaptureError> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("capture_legacy_fixtures.py");
    let python = request
        .python_bin
        .clone()
        .or_else(default_python_for_repo)
        .unwrap_or_else(|| PathBuf::from("python3"));

    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .arg("--legacy-root")
        .arg(&request.legacy_root)
        .arg("--output")
        .arg(&request.output_path);
    if request.strict {
        cmd.arg("--strict");
    }

    let rendered_cmd = format!(
        "{} {} --legacy-root {} --output {}{}",
        python.display(),
        script.display(),
        request.legacy_root.display(),
        request.output_path.display(),
        if request.strict { " --strict" } else { "" }
    );

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(OracleCaptureError::ScriptFailed {
            status: output.status.code(),
            stdout,
            stderr,
        });
    }

    let bundle = read_transform_fixture_bundle(&request.output_path)?;
    Ok(OracleCaptureResult {
        command: rendered_cmd,
        stdout,
        stderr,
        bundle,
    })
}

#[must_use]
pub fn run_transform_fixture_bundle_batched(
    _config: &HarnessConfig,
    bundle: &TransformFixtureBundle,
    batch: &BatchRunnerConfig,
) -> TransformParityReport {
    let mut reports = Vec::with_capacity(bundle.cases.len());

    for case in &bundle.cases {
        let expected_json = serde_json::to_string(&case.expected)
            .unwrap_or_else(|err| format!("<expected serialization error: {err}>"));
        if simulated_delay_exceeds_timeout(case, batch.case_timeout) {
            reports.push(timeout_case_report(
                case.clone(),
                expected_json,
                batch.case_timeout,
            ));
            continue;
        }

        let case_for_timeout = case.clone();
        let (tx, rx) = mpsc::channel::<TransformCaseReport>();
        let case_for_worker = case.clone();
        let handle = std::thread::spawn(move || {
            let report = run_transform_fixture_case(&case_for_worker);
            let _ = tx.send(report);
        });

        match rx.recv_timeout(batch.case_timeout) {
            Ok(report) => {
                let _ = handle.join();
                reports.push(report);
            }
            Err(_) => {
                // Detach the worker so a timeout remains a wall-clock bound
                // even if the underlying fixture code is slow or stuck.
                drop(handle);
                reports.push(timeout_case_report(
                    case_for_timeout,
                    expected_json,
                    batch.case_timeout,
                ));
            }
        }
    }

    let matched_cases = reports.iter().filter(|report| report.matched).count();
    TransformParityReport {
        schema_version: "frankenjax.transform-parity-report.v1".to_owned(),
        total_cases: reports.len(),
        matched_cases,
        mismatched_cases: reports.len().saturating_sub(matched_cases),
        reports,
    }
}

fn simulated_delay_exceeds_timeout(case: &TransformFixtureCase, case_timeout: Duration) -> bool {
    case.simulated_delay_ms > 0 && Duration::from_millis(case.simulated_delay_ms) >= case_timeout
}

fn timeout_case_report(
    case: TransformFixtureCase,
    expected_json: String,
    case_timeout: Duration,
) -> TransformCaseReport {
    TransformCaseReport {
        case_id: case.case_id,
        family: case.family,
        mode: case.mode,
        comparator: case.comparator,
        drift_classification: DriftClassification::Timeout,
        matched: false,
        expected_json,
        actual_json: None,
        error: Some(format!(
            "timeout waiting for case result after {}ms",
            case_timeout.as_millis()
        )),
    }
}

pub fn emit_parity_json(report: &TransformParityReport) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

#[must_use]
pub fn emit_parity_markdown(report: &TransformParityReport) -> String {
    let mut out = String::new();
    out.push_str("# Transform Parity Report\n\n");
    out.push_str("| Metric | Value |\n");
    out.push_str("|---|---|\n");
    out.push_str(&format!("| Schema | `{}` |\n", report.schema_version));
    out.push_str(&format!("| Total Cases | {} |\n", report.total_cases));
    out.push_str(&format!("| Matched Cases | {} |\n", report.matched_cases));
    out.push_str(&format!(
        "| Mismatched Cases | {} |\n\n",
        report.mismatched_cases
    ));

    out.push_str("| Case ID | Family | Mode | Comparator | Drift | Matched |\n");
    out.push_str("|---|---|---|---|---|---|\n");
    for case in &report.reports {
        out.push_str(&format!(
            "| {} | {:?} | {:?} | {:?} | {:?} | {} |\n",
            case.case_id,
            case.family,
            case.mode,
            case.comparator,
            case.drift_classification,
            case.matched
        ));
    }

    out
}

// ── V1 Parity Report (spec Section 8.3) ─────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyReport {
    pub total: usize,
    pub matched: usize,
    pub mismatched: usize,
    pub cases: Vec<FamilyCaseEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyCaseEntry {
    pub case_id: String,
    pub matched: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityReportSummary {
    pub total: usize,
    pub pass_rate: f64,
    pub gate: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityReportV1 {
    pub schema_version: String,
    pub timestamp: String,
    pub fj_version: String,
    pub oracle_version: String,
    pub mode: String,
    pub families: std::collections::BTreeMap<String, FamilyReport>,
    pub summary: ParityReportSummary,
}

impl ParityReportV1 {
    /// Build a V1 parity report from a TransformParityReport.
    #[must_use]
    pub fn from_transform_report(
        report: &TransformParityReport,
        mode: &str,
        fj_version: &str,
        oracle_version: &str,
    ) -> Self {
        use std::collections::BTreeMap;

        let timestamp = {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            format!("{now}")
        };

        // Group cases by family
        let mut families: BTreeMap<String, Vec<&TransformCaseReport>> = BTreeMap::new();
        for case in &report.reports {
            let family_name = match case.family {
                FixtureFamily::Jit => "jit",
                FixtureFamily::Grad => "grad",
                FixtureFamily::Vmap => "vmap",
                FixtureFamily::Lax => "lax",
                FixtureFamily::Random => "random",
                FixtureFamily::ControlFlow => "control_flow",
                FixtureFamily::MixedDtype => "mixed_dtype",
            };
            families
                .entry(family_name.to_owned())
                .or_default()
                .push(case);
        }

        let family_reports: BTreeMap<String, FamilyReport> = families
            .into_iter()
            .map(|(name, cases)| {
                let total = cases.len();
                let matched = cases.iter().filter(|c| c.matched).count();
                let mismatched = total - matched;
                let entries = cases
                    .iter()
                    .map(|c| FamilyCaseEntry {
                        case_id: c.case_id.clone(),
                        matched: c.matched,
                        expected: if c.matched {
                            None
                        } else {
                            Some(c.expected_json.clone())
                        },
                        actual: if c.matched {
                            None
                        } else {
                            c.actual_json.clone()
                        },
                        error: c.error.clone(),
                    })
                    .collect();
                (
                    name,
                    FamilyReport {
                        total,
                        matched,
                        mismatched,
                        cases: entries,
                    },
                )
            })
            .collect();

        let total = report.total_cases;
        let pass_rate = if total == 0 {
            1.0
        } else {
            report.matched_cases as f64 / total as f64
        };
        let gate = if pass_rate >= 1.0 {
            "pass".to_owned()
        } else {
            "fail".to_owned()
        };

        Self {
            schema_version: "frankenjax.parity-report.v1".to_owned(),
            timestamp,
            fj_version: fj_version.to_owned(),
            oracle_version: oracle_version.to_owned(),
            mode: mode.to_owned(),
            families: family_reports,
            summary: ParityReportSummary {
                total,
                pass_rate,
                gate,
            },
        }
    }

    /// Emit the report as pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Emit a markdown summary of the report.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# Parity Report V1\n\n");
        out.push_str(&format!("**Mode**: {}\n\n", self.mode));
        out.push_str(&format!(
            "**FrankenJAX**: {} | **Oracle**: {}\n\n",
            self.fj_version, self.oracle_version
        ));
        out.push_str("## Summary\n\n");
        out.push_str("| Metric | Value |\n");
        out.push_str("|---|---|\n");
        out.push_str(&format!("| Total Cases | {} |\n", self.summary.total));
        out.push_str(&format!(
            "| Pass Rate | {:.2}% |\n",
            self.summary.pass_rate * 100.0
        ));
        out.push_str(&format!("| Gate | **{}** |\n\n", self.summary.gate));
        out.push_str("## Per-Family Breakdown\n\n");
        out.push_str("| Family | Total | Matched | Mismatched |\n");
        out.push_str("|---|---|---|---|\n");
        for (name, family) in &self.families {
            out.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                name, family.total, family.matched, family.mismatched
            ));
        }
        out
    }

    /// Returns true if the gate passes (100% parity).
    #[must_use]
    pub fn gate_passes(&self) -> bool {
        self.summary.gate == "pass"
    }

    /// CI exit code: 0 if gate passes, 1 if it fails.
    #[must_use]
    pub fn ci_exit_code(&self) -> i32 {
        if self.gate_passes() { 0 } else { 1 }
    }
}

fn default_python_for_repo() -> Option<PathBuf> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let venv_python = repo_root.join(".venv").join("bin").join("python");
    if venv_python.exists() {
        return Some(venv_python);
    }
    None
}

fn run_transform_fixture_case(case: &TransformFixtureCase) -> TransformCaseReport {
    if case.simulated_delay_ms > 0 {
        std::thread::sleep(Duration::from_millis(case.simulated_delay_ms));
    }

    let runtime_args = case
        .args
        .iter()
        .map(FixtureValue::to_runtime_value)
        .collect::<Result<Vec<_>, _>>();

    let expected_json = serde_json::to_string(&case.expected)
        .unwrap_or_else(|err| format!("<expected serialization error: {err}>"));

    let runtime_args = match runtime_args {
        Ok(args) => args,
        Err(err) => {
            return TransformCaseReport {
                case_id: case.case_id.clone(),
                family: case.family,
                mode: case.mode,
                comparator: case.comparator,
                drift_classification: classify_drift(
                    false,
                    Some("fixture argument conversion failed"),
                    case.baseline_mismatch,
                    case.flaky,
                ),
                matched: false,
                expected_json,
                actual_json: None,
                error: Some(format!("fixture argument conversion failed: {err}")),
            };
        }
    };

    let mut ledger = TraceTransformLedger::new(build_program(case.program.as_program_spec()));
    for (idx, transform) in case.transforms.iter().enumerate() {
        let runtime_transform = transform.as_runtime_transform();
        ledger.push_transform(
            runtime_transform,
            format!(
                "fixture:{}:{}:{}",
                case.case_id,
                runtime_transform.as_str(),
                idx
            ),
        );
    }

    let result = dispatch(DispatchRequest {
        mode: case.mode.as_runtime_mode(),
        ledger,
        args: runtime_args,
        backend: "cpu".to_owned(),
        compile_options: std::collections::BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: Vec::new(),
    });

    match result {
        Ok(response) => {
            let matched = compare_outputs(case, &response.outputs);

            let actual_json = serde_json::to_string(&response.outputs)
                .unwrap_or_else(|err| format!("<actual serialization error: {err}>"));

            TransformCaseReport {
                case_id: case.case_id.clone(),
                family: case.family,
                mode: case.mode,
                comparator: case.comparator,
                drift_classification: classify_drift(
                    matched,
                    if matched {
                        None
                    } else {
                        Some("output mismatch")
                    },
                    case.baseline_mismatch,
                    case.flaky,
                ),
                matched,
                expected_json,
                actual_json: Some(actual_json),
                error: if matched {
                    None
                } else {
                    Some("output mismatch".to_owned())
                },
            }
        }
        Err(err) => TransformCaseReport {
            case_id: case.case_id.clone(),
            family: case.family,
            mode: case.mode,
            comparator: case.comparator,
            drift_classification: classify_drift(
                false,
                Some(&err.to_string()),
                case.baseline_mismatch,
                case.flaky,
            ),
            matched: false,
            expected_json,
            actual_json: None,
            error: Some(err.to_string()),
        },
    }
}

fn approx_equal(expected: f64, actual: f64, atol: f64, rtol: f64) -> bool {
    if expected.is_nan() && actual.is_nan() {
        return true;
    }
    if expected.is_infinite() && actual.is_infinite() && expected.signum() == actual.signum() {
        return true;
    }
    let tolerance = atol + rtol * expected.abs();
    (expected - actual).abs() <= tolerance
}

fn compare_outputs(case: &TransformFixtureCase, actual: &[Value]) -> bool {
    compare_outputs_with_report(case, actual, None)
}

/// Compare outputs with optional tolerance reporting.
/// When `report` is Some, records every numeric comparison for analysis.
fn compare_outputs_with_report(
    case: &TransformFixtureCase,
    actual: &[Value],
    report: Option<&mut ToleranceReport>,
) -> bool {
    if case.expected.len() != actual.len() {
        return false;
    }

    match case.comparator {
        ComparatorKind::Exact => {
            case.expected
                .iter()
                .zip(actual.iter())
                .all(|(expected, actual_value)| {
                    expected
                        .to_runtime_value()
                        .is_ok_and(|expected_value| expected_value == *actual_value)
                })
        }
        ComparatorKind::ApproxAtolRtol => {
            let mut all_match = true;
            for (expected, actual_value) in case.expected.iter().zip(actual.iter()) {
                let dtype = fixture_value_dtype(expected);
                let tier = ToleranceTier::resolve(dtype, case.atol, case.rtol);
                let matched = expected.approx_matches(actual_value, tier.atol, tier.rtol);
                if !matched {
                    all_match = false;
                }
            }
            // Record detailed comparisons if reporting is enabled
            if let Some(rep) = report {
                for (out_idx, (expected, actual_value)) in
                    case.expected.iter().zip(actual.iter()).enumerate()
                {
                    let dtype = fixture_value_dtype(expected);
                    let tier = ToleranceTier::resolve(dtype, case.atol, case.rtol);
                    record_fixture_comparison(
                        rep,
                        &case.case_id,
                        out_idx,
                        expected,
                        actual_value,
                        tier.atol + tier.rtol,
                    );
                }
            }
            all_match
        }
        ComparatorKind::ShapeOnly => {
            case.expected
                .iter()
                .zip(actual.iter())
                .all(|(expected, actual_value)| {
                    value_shape_fingerprint(expected) == value_shape_runtime(actual_value)
                })
        }
        ComparatorKind::TypeOnly => {
            case.expected
                .iter()
                .zip(actual.iter())
                .all(|(expected, actual_value)| {
                    value_type_fingerprint(expected) == value_type_runtime(actual_value)
                })
        }
    }
}

/// Record detailed per-element comparisons into the tolerance report.
fn record_fixture_comparison(
    report: &mut ToleranceReport,
    case_id: &str,
    output_idx: usize,
    expected: &FixtureValue,
    actual: &Value,
    tolerance: f64,
) {
    match expected {
        FixtureValue::ScalarF64 { value } => {
            if let Some(actual_val) = actual.as_f64_scalar() {
                report.record(case_id, output_idx, 0, *value, actual_val, tolerance);
            }
        }
        FixtureValue::VectorF64 { values } => {
            if let Some(actual_vals) = actual.as_tensor().and_then(|t| t.to_f64_vec()) {
                for (elem_idx, (exp, act)) in values.iter().zip(actual_vals.iter()).enumerate() {
                    report.record(case_id, output_idx, elem_idx, *exp, *act, tolerance);
                }
            }
        }
        FixtureValue::TensorF64 { values, .. } => {
            if let Some(actual_vals) = actual.as_tensor().and_then(|t| t.to_f64_vec()) {
                for (elem_idx, (exp, act)) in values.iter().zip(actual_vals.iter()).enumerate() {
                    report.record(case_id, output_idx, elem_idx, *exp, *act, tolerance);
                }
            }
        }
        FixtureValue::ScalarComplex128 { real, imag } => {
            if let Some((act_re, act_im)) = actual
                .as_scalar_literal()
                .and_then(fj_core::Literal::as_complex128)
            {
                report.record(case_id, output_idx, 0, *real, act_re, tolerance);
                report.record(case_id, output_idx, 1, *imag, act_im, tolerance);
            }
        }
        FixtureValue::TensorComplex128 { reals, imags, .. } => {
            if let Some(tensor) = actual.as_tensor() {
                for (elem_idx, ((&exp_re, &exp_im), lit)) in reals
                    .iter()
                    .zip(imags.iter())
                    .zip(tensor.elements.iter())
                    .enumerate()
                {
                    if let Some((act_re, act_im)) = lit.as_complex128() {
                        report.record(case_id, output_idx, elem_idx * 2, exp_re, act_re, tolerance);
                        report.record(
                            case_id,
                            output_idx,
                            elem_idx * 2 + 1,
                            exp_im,
                            act_im,
                            tolerance,
                        );
                    }
                }
            }
        }
        FixtureValue::ScalarI64 { .. }
        | FixtureValue::ScalarBool { .. }
        | FixtureValue::VectorI64 { .. }
        | FixtureValue::TensorI64 { .. }
        | FixtureValue::TensorBool { .. } => {
            // Integer/bool comparisons are exact, no tolerance reporting needed
        }
    }
}

fn value_shape_fingerprint(expected: &FixtureValue) -> String {
    match expected {
        FixtureValue::ScalarF64 { .. }
        | FixtureValue::ScalarI64 { .. }
        | FixtureValue::ScalarBool { .. }
        | FixtureValue::ScalarComplex128 { .. } => "scalar".to_owned(),
        FixtureValue::VectorF64 { values } => format!("vector:{}", values.len()),
        FixtureValue::VectorI64 { values } => format!("vector:{}", values.len()),
        FixtureValue::TensorF64 { shape, .. }
        | FixtureValue::TensorI64 { shape, .. }
        | FixtureValue::TensorComplex128 { shape, .. } => {
            format!("tensor:{shape:?}")
        }
        FixtureValue::TensorBool { shape, .. } => format!("tensor:{shape:?}"),
    }
}

fn value_shape_runtime(actual: &Value) -> String {
    if let Some(tensor) = actual.as_tensor() {
        if tensor.shape.dims.is_empty() {
            "scalar".to_owned()
        } else {
            format!("vector:{}", tensor.elements.len())
        }
    } else {
        "scalar".to_owned()
    }
}

fn value_type_fingerprint(expected: &FixtureValue) -> &'static str {
    match expected {
        FixtureValue::ScalarF64 { .. }
        | FixtureValue::VectorF64 { .. }
        | FixtureValue::TensorF64 { .. } => "f64",
        FixtureValue::ScalarI64 { .. }
        | FixtureValue::VectorI64 { .. }
        | FixtureValue::TensorI64 { .. } => "i64",
        FixtureValue::ScalarBool { .. } | FixtureValue::TensorBool { .. } => "bool",
        FixtureValue::ScalarComplex128 { .. } | FixtureValue::TensorComplex128 { .. } => {
            "complex128"
        }
    }
}

fn value_type_runtime(actual: &Value) -> &'static str {
    if let Some(scalar) = actual.as_scalar_literal() {
        return match scalar {
            fj_core::Literal::I64(_) => "i64",
            fj_core::Literal::U32(_) => "u32",
            fj_core::Literal::U64(_) => "u64",
            fj_core::Literal::BF16Bits(_) => "bf16",
            fj_core::Literal::F16Bits(_) => "f16",
            fj_core::Literal::F32Bits(_) => "f32",
            fj_core::Literal::F64Bits(_) => "f64",
            fj_core::Literal::Bool(_) => "bool",
            fj_core::Literal::Complex64Bits(..) => "complex64",
            fj_core::Literal::Complex128Bits(..) => "complex128",
        };
    }

    if let Some(tensor) = actual.as_tensor() {
        return match tensor.dtype {
            fj_core::DType::I64 | fj_core::DType::I32 => "i64",
            fj_core::DType::U32 => "u32",
            fj_core::DType::U64 => "u64",
            fj_core::DType::BF16 => "bf16",
            fj_core::DType::F16 => "f16",
            fj_core::DType::F64 | fj_core::DType::F32 => "f64",
            fj_core::DType::Bool => "bool",
            fj_core::DType::Complex64 => "complex64",
            fj_core::DType::Complex128 => "complex128",
        };
    }

    "unknown"
}

fn classify_drift(
    matched: bool,
    error: Option<&str>,
    baseline_mismatch: bool,
    flaky: bool,
) -> DriftClassification {
    if matched && baseline_mismatch {
        return DriftClassification::Improvement;
    }

    if matched {
        return DriftClassification::Pass;
    }

    if error.is_some_and(|detail| detail.to_ascii_lowercase().contains("timeout")) {
        return DriftClassification::Timeout;
    }

    if flaky {
        return DriftClassification::Flake;
    }

    DriftClassification::Regression
}

#[cfg(test)]
mod tests {
    use super::{
        BatchRunnerConfig, ComparatorKind, DriftClassification, FixtureFamily, FixtureMode,
        FixtureProgram, FixtureTransform, FixtureValue, HarnessConfig, OracleCaptureRequest,
        ParityCase, TransformCaseReport, TransformFixtureBundle, TransformFixtureCase,
        TransformParityReport, capture_transform_fixture_bundle_with_oracle, classify_drift,
        collect_json_fixtures, default_fixture_manifest, emit_parity_json, emit_parity_markdown,
        evaluate_parity, read_fixture_note, run_smoke, run_transform_fixture_bundle,
        run_transform_fixture_bundle_batched,
    };
    use tempfile::tempdir;

    #[test]
    fn smoke_harness_finds_oracle_and_fixtures() {
        let cfg = HarnessConfig::default_paths();
        if !cfg.oracle_root.exists() {
            eprintln!(
                "SKIP: oracle root not found at {}",
                cfg.oracle_root.display()
            );
            return;
        }
        let report = run_smoke(&cfg);
        assert!(report.oracle_present, "oracle repo should be present");
        assert!(report.fixture_count >= 1, "expected at least one fixture");
        assert_eq!(report.manifest_family_count, 5);
        assert!(report.strict_mode);
    }

    #[test]
    fn manifest_covers_core_families() {
        let manifest = default_fixture_manifest();
        assert_eq!(manifest.len(), 5);
    }

    #[test]
    fn parity_report_counts_matches() {
        let report = evaluate_parity(
            &[
                ParityCase {
                    case_id: "a".to_owned(),
                    expected: "1".to_owned(),
                    actual: "1".to_owned(),
                },
                ParityCase {
                    case_id: "b".to_owned(),
                    expected: "2".to_owned(),
                    actual: "3".to_owned(),
                },
            ],
            true,
        );

        assert_eq!(report.total_cases, 2);
        assert_eq!(report.matched_cases, 1);
        assert_eq!(report.mismatched_cases, 1);
        assert!(report.strict_mode);
    }

    #[test]
    fn fixture_note_round_trip() {
        let cfg = HarnessConfig::default_paths();
        let note_path = cfg.fixture_root.join("smoke_case.json");
        let note = read_fixture_note(&note_path).expect("fixture note should parse");
        assert_eq!(note.suite, "smoke");
    }

    #[test]
    fn fixture_collection_finds_json() {
        let cfg = HarnessConfig::default_paths();
        let fixtures = collect_json_fixtures(&cfg.fixture_root);
        assert!(!fixtures.is_empty());
    }

    #[test]
    fn transform_bundle_runs_and_matches() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![TransformFixtureCase {
                case_id: "jit_add_scalar".to_owned(),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                comparator: ComparatorKind::ApproxAtolRtol,
                baseline_mismatch: false,
                flaky: false,
                simulated_delay_ms: 0,
                args: vec![
                    FixtureValue::ScalarI64 { value: 2 },
                    FixtureValue::ScalarI64 { value: 5 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 7 }],
                atol: 1e-6,
                rtol: 1e-6,
            }],
        };

        let report = run_transform_fixture_bundle(&cfg, &bundle);
        assert_eq!(report.total_cases, 1);
        assert_eq!(report.matched_cases, 1);
        assert_eq!(report.mismatched_cases, 0);
    }

    #[test]
    fn test_conformance_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("conformance", "smoke")).expect("digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_conformance_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    #[test]
    fn test_drift_classification_timeout_and_regression() {
        assert_eq!(
            classify_drift(false, Some("timeout waiting for oracle"), false, false),
            DriftClassification::Timeout
        );
        assert_eq!(
            classify_drift(false, Some("shape mismatch"), false, false),
            DriftClassification::Regression
        );
        assert_eq!(
            classify_drift(true, None, true, false),
            DriftClassification::Improvement
        );
        assert_eq!(
            classify_drift(false, Some("intermittent mismatch"), false, true),
            DriftClassification::Flake
        );
        assert_eq!(
            classify_drift(true, None, false, false),
            DriftClassification::Pass
        );
    }

    #[test]
    fn test_parity_markdown_emitter_includes_summary() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![TransformFixtureCase {
                case_id: "jit_add_scalar_markdown".to_owned(),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                comparator: ComparatorKind::ApproxAtolRtol,
                baseline_mismatch: false,
                flaky: false,
                simulated_delay_ms: 0,
                args: vec![
                    FixtureValue::ScalarI64 { value: 1 },
                    FixtureValue::ScalarI64 { value: 2 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 3 }],
                atol: 1e-6,
                rtol: 1e-6,
            }],
        };
        let report = run_transform_fixture_bundle(&cfg, &bundle);
        let markdown = emit_parity_markdown(&report);
        assert!(markdown.contains("Transform Parity Report"));
        assert!(markdown.contains("jit_add_scalar_markdown"));
    }

    #[test]
    fn test_parity_markdown_snapshot() {
        let report = TransformParityReport {
            schema_version: "frankenjax.transform-parity-report.v1".to_owned(),
            total_cases: 3,
            matched_cases: 2,
            mismatched_cases: 1,
            reports: vec![
                TransformCaseReport {
                    case_id: "add_scalars".to_owned(),
                    family: FixtureFamily::Jit,
                    mode: FixtureMode::Strict,
                    comparator: ComparatorKind::Exact,
                    drift_classification: DriftClassification::Pass,
                    matched: true,
                    expected_json: "[3]".to_owned(),
                    actual_json: Some("[3]".to_owned()),
                    error: None,
                },
                TransformCaseReport {
                    case_id: "mul_vectors".to_owned(),
                    family: FixtureFamily::Grad,
                    mode: FixtureMode::Hardened,
                    comparator: ComparatorKind::ApproxAtolRtol,
                    drift_classification: DriftClassification::Pass,
                    matched: true,
                    expected_json: "[6.0]".to_owned(),
                    actual_json: Some("[6.0]".to_owned()),
                    error: None,
                },
                TransformCaseReport {
                    case_id: "div_by_zero".to_owned(),
                    family: FixtureFamily::Vmap,
                    mode: FixtureMode::Strict,
                    comparator: ComparatorKind::ApproxAtolRtol,
                    drift_classification: DriftClassification::Regression,
                    matched: false,
                    expected_json: "[Inf]".to_owned(),
                    actual_json: Some("[NaN]".to_owned()),
                    error: Some("value mismatch".to_owned()),
                },
            ],
        };
        let markdown = emit_parity_markdown(&report);
        insta::assert_snapshot!(markdown);
    }

    #[test]
    fn test_parity_json_emitter_round_trip() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![TransformFixtureCase {
                case_id: "jit_add_scalar_json".to_owned(),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                comparator: ComparatorKind::ApproxAtolRtol,
                baseline_mismatch: false,
                flaky: false,
                simulated_delay_ms: 0,
                args: vec![
                    FixtureValue::ScalarI64 { value: 1 },
                    FixtureValue::ScalarI64 { value: 1 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 2 }],
                atol: 1e-6,
                rtol: 1e-6,
            }],
        };
        let report = run_transform_fixture_bundle(&cfg, &bundle);
        let json = emit_parity_json(&report).expect("parity json should serialize");
        let decoded: super::TransformParityReport =
            serde_json::from_str(&json).expect("parity json should parse");
        assert_eq!(decoded.total_cases, 1);
    }

    #[test]
    fn test_comparator_taxonomy_exact_shape_type_and_approx() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![
                TransformFixtureCase {
                    case_id: "cmp_exact".to_owned(),
                    family: FixtureFamily::Jit,
                    mode: FixtureMode::Strict,
                    program: FixtureProgram::Add2,
                    transforms: vec![FixtureTransform::Jit],
                    comparator: ComparatorKind::Exact,
                    baseline_mismatch: false,
                    flaky: false,
                    simulated_delay_ms: 0,
                    args: vec![
                        FixtureValue::ScalarI64 { value: 2 },
                        FixtureValue::ScalarI64 { value: 2 },
                    ],
                    expected: vec![FixtureValue::ScalarI64 { value: 4 }],
                    atol: 0.0,
                    rtol: 0.0,
                },
                TransformFixtureCase {
                    case_id: "cmp_shape_only".to_owned(),
                    family: FixtureFamily::Vmap,
                    mode: FixtureMode::Strict,
                    program: FixtureProgram::AddOne,
                    transforms: vec![FixtureTransform::Vmap],
                    comparator: ComparatorKind::ShapeOnly,
                    baseline_mismatch: false,
                    flaky: false,
                    simulated_delay_ms: 0,
                    args: vec![FixtureValue::VectorI64 {
                        values: vec![10, 20, 30],
                    }],
                    expected: vec![FixtureValue::VectorI64 {
                        values: vec![999, 999, 999],
                    }],
                    atol: 1e-6,
                    rtol: 1e-6,
                },
                TransformFixtureCase {
                    case_id: "cmp_type_only".to_owned(),
                    family: FixtureFamily::Jit,
                    mode: FixtureMode::Strict,
                    program: FixtureProgram::Square,
                    transforms: vec![FixtureTransform::Jit],
                    comparator: ComparatorKind::TypeOnly,
                    baseline_mismatch: false,
                    flaky: false,
                    simulated_delay_ms: 0,
                    args: vec![FixtureValue::ScalarF64 { value: 2.0 }],
                    expected: vec![FixtureValue::ScalarF64 { value: -1234.0 }],
                    atol: 1e-6,
                    rtol: 1e-6,
                },
                TransformFixtureCase {
                    case_id: "cmp_approx".to_owned(),
                    family: FixtureFamily::Grad,
                    mode: FixtureMode::Strict,
                    program: FixtureProgram::Square,
                    transforms: vec![FixtureTransform::Grad],
                    comparator: ComparatorKind::ApproxAtolRtol,
                    baseline_mismatch: false,
                    flaky: false,
                    simulated_delay_ms: 0,
                    args: vec![FixtureValue::ScalarF64 { value: 3.0 }],
                    expected: vec![FixtureValue::ScalarF64 { value: 6.0 }],
                    atol: 1e-3,
                    rtol: 1e-3,
                },
            ],
        };
        let report = run_transform_fixture_bundle(&cfg, &bundle);
        assert_eq!(report.mismatched_cases, 0);
        let comparators = report
            .reports
            .iter()
            .map(|case| case.comparator)
            .collect::<Vec<_>>();
        assert!(comparators.contains(&ComparatorKind::Exact));
        assert!(comparators.contains(&ComparatorKind::ShapeOnly));
        assert!(comparators.contains(&ComparatorKind::TypeOnly));
        assert!(comparators.contains(&ComparatorKind::ApproxAtolRtol));
    }

    #[test]
    fn test_batch_runner_marks_timeout() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![
                TransformFixtureCase {
                    case_id: "batch_timeout".to_owned(),
                    family: FixtureFamily::Jit,
                    mode: FixtureMode::Strict,
                    program: FixtureProgram::Add2,
                    transforms: vec![FixtureTransform::Jit],
                    comparator: ComparatorKind::ApproxAtolRtol,
                    baseline_mismatch: false,
                    flaky: false,
                    simulated_delay_ms: 500,
                    args: vec![
                        FixtureValue::ScalarI64 { value: 1 },
                        FixtureValue::ScalarI64 { value: 2 },
                    ],
                    expected: vec![FixtureValue::ScalarI64 { value: 3 }],
                    atol: 1e-6,
                    rtol: 1e-6,
                },
                TransformFixtureCase {
                    case_id: "batch_fast_after_timeout".to_owned(),
                    family: FixtureFamily::Jit,
                    mode: FixtureMode::Strict,
                    program: FixtureProgram::Add2,
                    transforms: vec![FixtureTransform::Jit],
                    comparator: ComparatorKind::ApproxAtolRtol,
                    baseline_mismatch: false,
                    flaky: false,
                    simulated_delay_ms: 0,
                    args: vec![
                        FixtureValue::ScalarI64 { value: 2 },
                        FixtureValue::ScalarI64 { value: 5 },
                    ],
                    expected: vec![FixtureValue::ScalarI64 { value: 7 }],
                    atol: 1e-6,
                    rtol: 1e-6,
                },
            ],
        };
        let report = run_transform_fixture_bundle_batched(
            &cfg,
            &bundle,
            &BatchRunnerConfig {
                case_timeout: std::time::Duration::from_millis(200),
            },
        );
        assert_eq!(report.total_cases, 2);
        assert_eq!(report.reports[0].case_id, "batch_timeout");
        assert_eq!(
            report.reports[0].drift_classification,
            DriftClassification::Timeout
        );
        assert_eq!(report.reports[1].case_id, "batch_fast_after_timeout");
        assert_eq!(
            report.reports[1].drift_classification,
            DriftClassification::Pass
        );
    }

    #[test]
    fn test_batch_runner_timeout_returns_before_delayed_worker_finishes() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![TransformFixtureCase {
                case_id: "batch_long_timeout".to_owned(),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                comparator: ComparatorKind::ApproxAtolRtol,
                baseline_mismatch: false,
                flaky: false,
                simulated_delay_ms: 5_000,
                args: vec![
                    FixtureValue::ScalarI64 { value: 1 },
                    FixtureValue::ScalarI64 { value: 2 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 3 }],
                atol: 1e-6,
                rtol: 1e-6,
            }],
        };

        let started = std::time::Instant::now();
        let report = run_transform_fixture_bundle_batched(
            &cfg,
            &bundle,
            &BatchRunnerConfig {
                case_timeout: std::time::Duration::from_millis(1),
            },
        );
        let elapsed = started.elapsed();

        assert_eq!(report.total_cases, 1);
        assert_eq!(
            report.reports[0].drift_classification,
            DriftClassification::Timeout
        );
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "batch timeout waited for delayed worker: {elapsed:?}"
        );
    }

    #[test]
    fn test_batch_runner_preclassifies_impossible_simulated_delays() {
        let cfg = HarnessConfig::default_paths();
        let cases = (0..128)
            .map(|i| TransformFixtureCase {
                case_id: format!("batch_timeout_{i}"),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                comparator: ComparatorKind::ApproxAtolRtol,
                baseline_mismatch: false,
                flaky: false,
                simulated_delay_ms: 60_000,
                args: vec![
                    FixtureValue::ScalarI64 { value: 1 },
                    FixtureValue::ScalarI64 { value: 2 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 3 }],
                atol: 1e-6,
                rtol: 1e-6,
            })
            .collect();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases,
        };

        let started = std::time::Instant::now();
        let report = run_transform_fixture_bundle_batched(
            &cfg,
            &bundle,
            &BatchRunnerConfig {
                case_timeout: std::time::Duration::from_millis(5),
            },
        );
        let elapsed = started.elapsed();

        assert_eq!(report.total_cases, 128);
        assert!(report.reports.iter().all(|report| {
            report.drift_classification == DriftClassification::Timeout && !report.matched
        }));
        assert!(
            elapsed < std::time::Duration::from_millis(250),
            "preclassified simulated delays should not spawn sleeping workers: {elapsed:?}"
        );
    }

    fn add2_case(case_id: String, simulated_delay_ms: u64) -> TransformFixtureCase {
        TransformFixtureCase {
            case_id,
            family: FixtureFamily::Jit,
            mode: FixtureMode::Strict,
            program: FixtureProgram::Add2,
            transforms: vec![FixtureTransform::Jit],
            comparator: ComparatorKind::ApproxAtolRtol,
            baseline_mismatch: false,
            flaky: false,
            simulated_delay_ms,
            args: vec![
                FixtureValue::ScalarI64 { value: 1 },
                FixtureValue::ScalarI64 { value: 2 },
            ],
            expected: vec![FixtureValue::ScalarI64 { value: 3 }],
            atol: 1e-6,
            rtol: 1e-6,
        }
    }

    #[test]
    fn test_read_transform_fixture_bundle_rejects_external_simulated_delay() {
        let tmp = tempdir().expect("tempdir should build");
        let fixture_path = tmp.path().join("delayed-bundle.json");
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![add2_case("delayed_case".to_owned(), 1)],
        };
        std::fs::write(&fixture_path, serde_json::to_vec(&bundle).unwrap())
            .expect("fixture should write");

        let err = super::read_transform_fixture_bundle(&fixture_path).unwrap_err();

        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("simulated_delay_ms"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_read_transform_fixture_bundle_rejects_excessive_case_count() {
        let tmp = tempdir().expect("tempdir should build");
        let fixture_path = tmp.path().join("large-bundle.json");
        let cases = (0..=super::MAX_EXTERNAL_TRANSFORM_FIXTURE_CASES)
            .map(|i| add2_case(format!("case_{i}"), 0))
            .collect();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases,
        };
        std::fs::write(&fixture_path, serde_json::to_vec(&bundle).unwrap())
            .expect("fixture should write");

        let err = super::read_transform_fixture_bundle(&fixture_path).unwrap_err();

        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("too many transform fixture cases"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_oracle_capture_invocation_produces_large_bundle() {
        let cfg = HarnessConfig::default_paths();
        if !cfg.oracle_root.exists() {
            eprintln!(
                "SKIP: oracle root not found at {}",
                cfg.oracle_root.display()
            );
            return;
        }
        let tmp = tempdir().expect("tempdir should build");
        let output = tmp.path().join("oracle-capture.json");
        let result = capture_transform_fixture_bundle_with_oracle(&OracleCaptureRequest {
            legacy_root: cfg.oracle_root.clone(),
            output_path: output,
            strict: false,
            python_bin: None,
        })
        .expect("oracle capture script should run");
        assert!(
            result.bundle.cases.len() >= 50,
            "expected expanded fixture corpus, got {}",
            result.bundle.cases.len()
        );
    }

    // ── Tolerance tier tests (bd-2l5q) ───────────────────────────

    #[test]
    fn test_tolerance_f64_default() {
        let tier = super::ToleranceTier::for_dtype(fj_core::DType::F64);
        assert_eq!(tier.atol, 1e-12, "F64 atol should be 1e-12");
        assert_eq!(tier.rtol, 1e-10, "F64 rtol should be 1e-10");
    }

    #[test]
    fn test_tolerance_f32_default() {
        let tier = super::ToleranceTier::for_dtype(fj_core::DType::F32);
        assert_eq!(tier.atol, 1e-5, "F32 atol should be 1e-5");
        assert_eq!(tier.rtol, 1e-5, "F32 rtol should be 1e-5");
    }

    #[test]
    fn test_tolerance_i32_exact() {
        let tier = super::ToleranceTier::for_dtype(fj_core::DType::I32);
        assert_eq!(tier.atol, 0.0, "I32 should use exact comparison");
        assert_eq!(tier.rtol, 0.0, "I32 should use exact comparison");
    }

    #[test]
    fn test_tolerance_i64_exact() {
        let tier = super::ToleranceTier::for_dtype(fj_core::DType::I64);
        assert_eq!(tier.atol, 0.0);
        assert_eq!(tier.rtol, 0.0);
    }

    #[test]
    fn test_tolerance_bool_exact() {
        let tier = super::ToleranceTier::for_dtype(fj_core::DType::Bool);
        assert_eq!(tier.atol, 0.0);
        assert_eq!(tier.rtol, 0.0);
    }

    #[test]
    fn test_tolerance_per_fixture_override() {
        // Fixture specifies wider tolerance → resolve takes the wider
        let tier = super::ToleranceTier::resolve(fj_core::DType::F64, 1e-6, 1e-6);
        assert!(
            tier.atol >= 1e-6,
            "resolve should take wider: got {}",
            tier.atol
        );
        assert!(
            tier.rtol >= 1e-6,
            "resolve should take wider: got {}",
            tier.rtol
        );
    }

    #[test]
    fn test_tolerance_erf_wider() {
        // erf/erfc fixtures should be allowed wider tolerance
        // When fixture specifies 1e-4, resolve should use that over F64's 1e-12
        let tier = super::ToleranceTier::resolve(fj_core::DType::F64, 1e-4, 1e-4);
        assert_eq!(tier.atol, 1e-4, "erf override should use 1e-4");
        assert_eq!(tier.rtol, 1e-4, "erf override should use 1e-4");
    }

    #[test]
    fn test_tolerance_violation_report() {
        let mut report = super::ToleranceReport::new();
        // Record a passing comparison
        report.record("case_ok", 0, 0, 1.0, 1.0 + 1e-13, 1e-12);
        assert!(report.all_passed(), "within tolerance should pass");

        // Record a failing comparison
        report.record("case_fail", 0, 0, 1.0, 1.5, 1e-12);
        assert!(
            !report.all_passed(),
            "exceeded tolerance should produce violation"
        );
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].case_id, "case_fail");
        assert!((report.violations[0].abs_diff - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_report_statistics() {
        let mut report = super::ToleranceReport::new();
        report.record("c1", 0, 0, 1.0, 1.1, 1e-6);
        report.record("c2", 0, 0, 2.0, 2.0001, 1e-3);
        report.finalize(0.1 + 0.0001);

        assert_eq!(report.total_comparisons, 2);
        assert!((report.max_abs_error - 0.1).abs() < 1e-10);
        assert!(report.mean_abs_error > 0.0);
    }

    #[test]
    fn test_existing_fixtures_still_pass_with_tight_tolerances() {
        // Run the full fixture bundle with the tightened tolerances
        let cfg = HarnessConfig::default_paths();
        let fixture_path = cfg
            .fixture_root
            .join("transforms/legacy_transform_cases.v1.json");
        let bundle =
            super::read_transform_fixture_bundle(&fixture_path).expect("bundle should load");
        let report = run_transform_fixture_bundle(&cfg, &bundle);

        assert_eq!(
            report.mismatched_cases, 0,
            "all fixtures should pass with tightened tolerances (got {} mismatches out of {})",
            report.mismatched_cases, report.total_cases
        );
    }

    #[test]
    fn test_tolerance_histogram() {
        let mut report = super::ToleranceReport::new();
        // Record violations at different magnitudes
        report.record("small", 0, 0, 1.0, 1.0 + 1e-14, 0.0); // ~1e-14 error
        report.record("medium", 0, 0, 1.0, 1.0 + 1e-7, 0.0); // ~1e-7 error
        report.record("large", 0, 0, 1.0, 1.5, 0.0); // 0.5 error
        report.build_histogram();

        assert_eq!(
            report.error_histogram.len(),
            7,
            "histogram should have 7 buckets"
        );
        // 1e-14 falls in [1e-15, 1e-12)
        assert_eq!(
            report.error_histogram[1].count, 1,
            "1e-14 error should be in bucket [1e-15, 1e-12)"
        );
        // 1e-7 falls in [1e-9, 1e-6)
        assert_eq!(
            report.error_histogram[3].count, 1,
            "1e-7 error should be in bucket [1e-9, 1e-6)"
        );
        // 0.5 falls in [1e-3, 1.0)
        assert_eq!(
            report.error_histogram[5].count, 1,
            "0.5 error should be in bucket [1e-3, 1.0)"
        );
    }

    #[test]
    fn test_known_imprecise_ops_flagged() {
        // erf/erfc fixtures should specify wider tolerance (1e-4) in the fixture itself
        // When resolved against F64 defaults, the wider fixture tolerance wins
        let tier = super::ToleranceTier::resolve(fj_core::DType::F64, 1e-4, 1e-4);
        assert!(
            tier.atol > 1e-6,
            "erf/erfc should use wider tolerance than 1e-6, got {}",
            tier.atol
        );

        // Standard F64 ops should use tight tolerance
        let tight = super::ToleranceTier::resolve(fj_core::DType::F64, 0.0, 0.0);
        assert!(
            tight.atol <= 1e-12,
            "standard F64 should use tight tolerance, got {}",
            tight.atol
        );
    }

    #[test]
    fn test_e2e_tolerance_tightening_forensic_log() {
        // Run the full fixture bundle and emit an E2E forensic log
        let cfg = HarnessConfig::default_paths();
        let fixture_path = cfg
            .fixture_root
            .join("transforms/legacy_transform_cases.v1.json");
        let bundle =
            super::read_transform_fixture_bundle(&fixture_path).expect("bundle should load");
        let report = run_transform_fixture_bundle(&cfg, &bundle);

        // Build forensic log entry
        let log_entry = serde_json::json!({
            "scenario": "e2e_tolerance_tightening",
            "fixture_family": "transforms",
            "total_cases": report.total_cases,
            "passed": report.matched_cases,
            "failed": report.mismatched_cases,
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        });

        let log_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/e2e/e2e_tolerance_tightening.e2e.json");
        if let Some(parent) = log_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&log_path, serde_json::to_string_pretty(&log_entry).unwrap())
            .expect("should write forensic log");

        assert_eq!(
            report.mismatched_cases, 0,
            "all fixtures should pass with tight tolerances"
        );
    }

    #[test]
    fn test_fixture_value_dtype_inference() {
        assert_eq!(
            super::fixture_value_dtype(&FixtureValue::ScalarF64 { value: 1.0 }),
            fj_core::DType::F64
        );
        assert_eq!(
            super::fixture_value_dtype(&FixtureValue::ScalarI64 { value: 1 }),
            fj_core::DType::I64
        );
        assert_eq!(
            super::fixture_value_dtype(&FixtureValue::VectorF64 { values: vec![1.0] }),
            fj_core::DType::F64
        );
        assert_eq!(
            super::fixture_value_dtype(&FixtureValue::VectorI64 { values: vec![1] }),
            fj_core::DType::I64
        );
    }
}
