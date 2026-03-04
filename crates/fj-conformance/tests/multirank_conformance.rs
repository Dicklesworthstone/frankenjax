//! Multi-rank tensor fixture families (V2-CONFORM-03)
//!
//! Tests fixture construction and structural validation for tensors of rank 0-4,
//! covering broadcasting, reduction along axes, transpose, and edge cases.

use fj_ad::grad_jaxpr;
use fj_conformance::{
    ComparatorKind, FixtureFamily, FixtureMode, FixtureProgram, FixtureTransform, FixtureValue,
    HarnessConfig, TransformFixtureBundle, TransformFixtureCase, run_transform_fixture_bundle,
};
use fj_core::{DType, Literal, Primitive, ProgramSpec, Shape, TensorValue, Value, build_program};
use fj_lax::eval_primitive;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn make_lax_case(
    case_id: &str,
    program: FixtureProgram,
    transforms: Vec<FixtureTransform>,
    family: FixtureFamily,
    args: Vec<FixtureValue>,
    expected: Vec<FixtureValue>,
) -> TransformFixtureCase {
    TransformFixtureCase {
        case_id: case_id.to_owned(),
        family,
        mode: FixtureMode::Strict,
        program,
        transforms,
        comparator: ComparatorKind::ApproxAtolRtol,
        baseline_mismatch: false,
        flaky: false,
        simulated_delay_ms: 0,
        args,
        expected,
        atol: 1e-6,
        rtol: 1e-6,
    }
}

fn build_multirank_fixture_bundle() -> TransformFixtureBundle {
    let cases = vec![
        // ════════════════════════════════════════════════════════
        // Rank 0: Scalar operations
        // ════════════════════════════════════════════════════════
        // 1. Scalar add
        make_lax_case(
            "r0_add_scalars",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::ScalarF64 { value: 3.0 },
                FixtureValue::ScalarF64 { value: 4.0 },
            ],
            vec![FixtureValue::ScalarF64 { value: 7.0 }],
        ),
        // 2. Scalar square
        make_lax_case(
            "r0_square_scalar",
            FixtureProgram::Square,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::ScalarF64 { value: 5.0 }],
            vec![FixtureValue::ScalarF64 { value: 25.0 }],
        ),
        // 3. Scalar neg
        make_lax_case(
            "r0_neg_scalar",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::ScalarF64 { value: 7.0 }],
            vec![FixtureValue::ScalarF64 { value: -7.0 }],
        ),
        // 4. Scalar exp
        make_lax_case(
            "r0_exp_scalar",
            FixtureProgram::LaxExp,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::ScalarF64 { value: 0.0 }],
            vec![FixtureValue::ScalarF64 { value: 1.0 }],
        ),
        // 5. Scalar grad
        make_lax_case(
            "r0_grad_square",
            FixtureProgram::Square,
            vec![FixtureTransform::Grad],
            FixtureFamily::Grad,
            vec![FixtureValue::ScalarF64 { value: 3.0 }],
            vec![FixtureValue::ScalarF64 { value: 6.0 }],
        ),
        // ════════════════════════════════════════════════════════
        // Rank 1: Vector operations
        // ════════════════════════════════════════════════════════
        // 6. Vector add
        make_lax_case(
            "r1_add_vectors",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::VectorF64 {
                    values: vec![1.0, 2.0, 3.0],
                },
                FixtureValue::VectorF64 {
                    values: vec![4.0, 5.0, 6.0],
                },
            ],
            vec![FixtureValue::VectorF64 {
                values: vec![5.0, 7.0, 9.0],
            }],
        ),
        // 7. Single-element vector
        make_lax_case(
            "r1_add_single",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::VectorF64 { values: vec![42.0] },
                FixtureValue::VectorF64 { values: vec![8.0] },
            ],
            vec![FixtureValue::VectorF64 { values: vec![50.0] }],
        ),
        // 8. Large vector (1024 elements)
        make_lax_case(
            "r1_neg_large",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::VectorF64 {
                values: (0..1024).map(|i| i as f64).collect(),
            }],
            vec![FixtureValue::VectorF64 {
                values: (0..1024).map(|i| -(i as f64)).collect(),
            }],
        ),
        // 9. Vector dot product
        make_lax_case(
            "r1_dot_vectors",
            FixtureProgram::Dot3,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::VectorF64 {
                    values: vec![1.0, 2.0, 3.0],
                },
                FixtureValue::VectorF64 {
                    values: vec![4.0, 5.0, 6.0],
                },
            ],
            vec![FixtureValue::ScalarF64 { value: 32.0 }],
        ),
        // 10. Vector reduce_sum
        make_lax_case(
            "r1_reduce_sum",
            FixtureProgram::ReduceSumVec,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
            vec![FixtureValue::ScalarF64 { value: 10.0 }],
        ),
        // ════════════════════════════════════════════════════════
        // Rank 2: Matrix operations (using TensorF64)
        // ════════════════════════════════════════════════════════
        // 11. 2x2 matrix negation
        make_lax_case(
            "r2_neg_2x2",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![-1.0, -2.0, -3.0, -4.0],
            }],
        ),
        // 12. 3x3 matrix abs
        make_lax_case(
            "r2_abs_3x3",
            FixtureProgram::LaxAbs,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 3],
                values: vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 3],
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            }],
        ),
        // 13. 4x4 matrix floor
        make_lax_case(
            "r2_floor_4x4",
            FixtureProgram::LaxFloor,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![4, 4],
                values: vec![
                    1.1, 2.9, 3.5, 4.0, -1.1, -2.9, -3.5, -4.0, 0.0, 0.5, -0.5, 99.99, -99.99,
                    0.001, -0.001, 1e10,
                ],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![4, 4],
                values: vec![
                    1.0, 2.0, 3.0, 4.0, -2.0, -3.0, -4.0, -4.0, 0.0, 0.0, -1.0, 99.0, -100.0, 0.0,
                    -1.0, 1e10,
                ],
            }],
        ),
        // 14. 2x3 matrix exp
        make_lax_case(
            "r2_exp_2x3",
            FixtureProgram::LaxExp,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![
                    1.0,
                    std::f64::consts::E,
                    (-1.0_f64).exp(),
                    (2.0_f64).exp(),
                    (-2.0_f64).exp(),
                    (0.5_f64).exp(),
                ],
            }],
        ),
        // 15. 1x1 matrix (singleton)
        make_lax_case(
            "r2_neg_1x1",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1],
                values: vec![42.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1],
                values: vec![-42.0],
            }],
        ),
        // 16. 8x16 matrix square
        make_lax_case(
            "r2_square_8x16",
            FixtureProgram::LaxSquare,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![8, 16],
                values: (0..128).map(|i| i as f64).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![8, 16],
                values: (0..128).map(|i| (i * i) as f64).collect(),
            }],
        ),
        // 17. 2x2 matrix add (elementwise)
        make_lax_case(
            "r2_add_2x2",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![10.0, 20.0, 30.0, 40.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![11.0, 22.0, 33.0, 44.0],
            }],
        ),
        // 18. 3x2 matrix sin
        make_lax_case(
            "r2_sin_3x2",
            FixtureProgram::SinX,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 2],
                values: vec![
                    0.0,
                    std::f64::consts::FRAC_PI_2,
                    std::f64::consts::PI,
                    0.0,
                    1.0,
                    2.0,
                ],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 2],
                values: vec![
                    0.0,
                    1.0,
                    std::f64::consts::PI.sin(),
                    0.0,
                    1.0_f64.sin(),
                    2.0_f64.sin(),
                ],
            }],
        ),
        // 19. 2x2 matrix reciprocal
        make_lax_case(
            "r2_reciprocal_2x2",
            FixtureProgram::LaxReciprocal,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 2.0, 4.0, 5.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 0.5, 0.25, 0.2],
            }],
        ),
        // 20. 2x2 matrix mul (elementwise)
        make_lax_case(
            "r2_mul_2x2",
            FixtureProgram::LaxMul,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![5.0, 6.0, 7.0, 8.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![5.0, 12.0, 21.0, 32.0],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Rank 3: 3D tensor operations
        // ════════════════════════════════════════════════════════
        // 21. 2x2x2 tensor negation
        make_lax_case(
            "r3_neg_2x2x2",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2, 2],
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2, 2],
                values: vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
            }],
        ),
        // 22. 2x3x4 tensor abs
        make_lax_case(
            "r3_abs_2x3x4",
            FixtureProgram::LaxAbs,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3, 4],
                values: (0..24)
                    .map(|i| if i % 2 == 0 { i as f64 } else { -(i as f64) })
                    .collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3, 4],
                values: (0..24).map(|i| i as f64).collect(),
            }],
        ),
        // 23. 1x1x1 tensor (singleton 3D)
        make_lax_case(
            "r3_neg_1x1x1",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1, 1],
                values: vec![99.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1, 1],
                values: vec![-99.0],
            }],
        ),
        // 24. 3x2x2 tensor exp
        make_lax_case(
            "r3_exp_3x2x2",
            FixtureProgram::LaxExp,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 2, 2],
                values: vec![
                    0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 1.0, -1.0, 0.0, 0.5,
                ],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 2, 2],
                values: vec![
                    1.0,
                    std::f64::consts::E,
                    (-1.0_f64).exp(),
                    (0.5_f64).exp(),
                    (-0.5_f64).exp(),
                    (2.0_f64).exp(),
                    (-2.0_f64).exp(),
                    1.0,
                    std::f64::consts::E,
                    (-1.0_f64).exp(),
                    1.0,
                    (0.5_f64).exp(),
                ],
            }],
        ),
        // 25. 2x2x3 tensor add (elementwise)
        make_lax_case(
            "r3_add_2x2x3",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2, 3],
                    values: (0..12).map(|i| i as f64).collect(),
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2, 3],
                    values: (0..12).map(|i| (i * 10) as f64).collect(),
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2, 3],
                values: (0..12).map(|i| (i + i * 10) as f64).collect(),
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Rank 4: 4D tensor operations
        // ════════════════════════════════════════════════════════
        // 26. 2x2x2x2 tensor neg
        make_lax_case(
            "r4_neg_2x2x2x2",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2, 2, 2],
                values: (0..16).map(|i| (i + 1) as f64).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2, 2, 2],
                values: (0..16).map(|i| -((i + 1) as f64)).collect(),
            }],
        ),
        // 27. 1x2x3x4 tensor square
        make_lax_case(
            "r4_square_1x2x3x4",
            FixtureProgram::LaxSquare,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 2, 3, 4],
                values: (0..24).map(|i| i as f64).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 2, 3, 4],
                values: (0..24).map(|i| (i * i) as f64).collect(),
            }],
        ),
        // 28. 1x1x1x1 tensor (singleton 4D)
        make_lax_case(
            "r4_abs_1x1x1x1",
            FixtureProgram::LaxAbs,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1, 1, 1],
                values: vec![-7.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1, 1, 1],
                values: vec![7.0],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Trig functions on matrices
        // ════════════════════════════════════════════════════════
        // 29. 2x2 cos
        make_lax_case(
            "r2_cos_2x2",
            FixtureProgram::CosX,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, std::f64::consts::PI, std::f64::consts::FRAC_PI_2, 1.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    1.0,
                    -1.0,
                    (std::f64::consts::FRAC_PI_2).cos(),
                    1.0_f64.cos(),
                ],
            }],
        ),
        // 30. 2x3 tanh
        make_lax_case(
            "r2_tanh_2x3",
            FixtureProgram::LaxTanh,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![0.0, 1.0, -1.0, 100.0, -100.0, 0.5],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![
                    0.0,
                    1.0_f64.tanh(),
                    (-1.0_f64).tanh(),
                    100.0_f64.tanh(),
                    (-100.0_f64).tanh(),
                    0.5_f64.tanh(),
                ],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Log/special functions on matrices
        // ════════════════════════════════════════════════════════
        // 31. 2x2 log
        make_lax_case(
            "r2_log_2x2",
            FixtureProgram::LaxLog,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, std::f64::consts::E, 10.0, 0.5],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, 10.0_f64.ln(), 0.5_f64.ln()],
            }],
        ),
        // 32. 2x2 sqrt
        make_lax_case(
            "r2_sqrt_2x2",
            FixtureProgram::LaxSqrt,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, 4.0, 9.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, 2.0, 3.0],
            }],
        ),
        // 33. 2x2 logistic
        make_lax_case(
            "r2_logistic_2x2",
            FixtureProgram::LaxLogistic,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 100.0, -100.0, 1.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    0.5,
                    1.0, // logistic(100) ≈ 1
                    0.0, // logistic(-100) ≈ 0
                    1.0 / (1.0 + (-1.0_f64).exp()),
                ],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Comparison on matrices
        // ════════════════════════════════════════════════════════
        // 34. 2x2 eq
        make_lax_case(
            "r2_eq_2x2",
            FixtureProgram::LaxEq,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 0.0, 3.0, 5.0],
                },
            ],
            vec![FixtureValue::TensorBool {
                shape: vec![2, 2],
                values: vec![true, false, true, false],
            }],
        ),
        // 35. 2x2 lt
        make_lax_case(
            "r2_lt_2x2",
            FixtureProgram::LaxLt,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 5.0, 3.0, 4.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![2.0, 3.0, 3.0, 5.0],
                },
            ],
            vec![FixtureValue::TensorBool {
                shape: vec![2, 2],
                values: vec![true, false, false, true],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Binary operations on matrices
        // ════════════════════════════════════════════════════════
        // 36. 2x2 sub
        make_lax_case(
            "r2_sub_2x2",
            FixtureProgram::LaxSub,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![10.0, 20.0, 30.0, 40.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![9.0, 18.0, 27.0, 36.0],
            }],
        ),
        // 37. 2x2 div
        make_lax_case(
            "r2_div_2x2",
            FixtureProgram::LaxDiv,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![10.0, 20.0, 30.0, 40.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![2.0, 5.0, 6.0, 8.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![5.0, 4.0, 5.0, 5.0],
            }],
        ),
        // 38. 2x2 max
        make_lax_case(
            "r2_max_2x2",
            FixtureProgram::LaxMax,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 5.0, 3.0, 7.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![4.0, 2.0, 6.0, 0.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![4.0, 5.0, 6.0, 7.0],
            }],
        ),
        // 39. 2x2 min
        make_lax_case(
            "r2_min_2x2",
            FixtureProgram::LaxMin,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 5.0, 3.0, 7.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![4.0, 2.0, 6.0, 0.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 2.0, 3.0, 0.0],
            }],
        ),
        // 40. 2x2 pow
        make_lax_case(
            "r2_pow_2x2",
            FixtureProgram::LaxPow,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![2.0, 3.0, 4.0, 5.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![3.0, 2.0, 0.5, 1.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![8.0, 9.0, 2.0, 5.0],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Vmap over matrices
        // ════════════════════════════════════════════════════════
        // 41. vmap(neg) over batch of vectors (rank 2 input)
        make_lax_case(
            "r2_vmap_neg",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Vmap],
            FixtureFamily::Vmap,
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 4],
                values: (0..12).map(|i| (i + 1) as f64).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 4],
                values: (0..12).map(|i| -((i + 1) as f64)).collect(),
            }],
        ),
        // 42. vmap(square) over batch of vectors (rank 2 input)
        make_lax_case(
            "r2_vmap_square",
            FixtureProgram::LaxSquare,
            vec![FixtureTransform::Vmap],
            FixtureFamily::Vmap,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Grad on matrices (scalar output)
        // ════════════════════════════════════════════════════════
        // 43. grad(reduce_sum) on vector
        make_lax_case(
            "r1_grad_reduce_sum",
            FixtureProgram::ReduceSumVec,
            vec![FixtureTransform::Grad],
            FixtureFamily::Grad,
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 3.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 1.0, 1.0],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Additional unary ops on rank-2
        // ════════════════════════════════════════════════════════
        // 44. 2x2 ceil
        make_lax_case(
            "r2_ceil_2x2",
            FixtureProgram::LaxCeil,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.1, 2.9, -1.1, -2.9],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![2.0, 3.0, -1.0, -2.0],
            }],
        ),
        // 45. 2x2 round
        make_lax_case(
            "r2_round_2x2",
            FixtureProgram::LaxRound,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.4, 1.5, 2.5, 3.6],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 2.0, 2.0, 4.0],
            }],
        ),
        // 46. 2x2 sign
        make_lax_case(
            "r2_sign_2x2",
            FixtureProgram::LaxSign,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![-5.0, 0.0, 3.0, -0.001],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![-1.0, 0.0, 1.0, -1.0],
            }],
        ),
        // 47. 2x2 rsqrt
        make_lax_case(
            "r2_rsqrt_2x2",
            FixtureProgram::LaxRsqrt,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 4.0, 9.0, 16.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 0.5, 1.0 / 3.0, 0.25],
            }],
        ),
        // 48. 2x2 erf
        make_lax_case(
            "r2_erf_2x2",
            FixtureProgram::LaxErf,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, -1.0, 3.0],
            }],
            // erf values computed by reference
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    0.0,
                    0.842700792949715,
                    -0.842700792949715,
                    0.9999779095030014,
                ],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Jit composition on tensors
        // ════════════════════════════════════════════════════════
        // 49. jit(add_one) on 2x2 matrix
        make_lax_case(
            "r2_jit_add_one",
            FixtureProgram::AddOne,
            vec![FixtureTransform::Jit],
            FixtureFamily::Jit,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![10.0, 20.0, 30.0, 40.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![11.0, 21.0, 31.0, 41.0],
            }],
        ),
        // 50. jit(square) on 2x3 matrix
        make_lax_case(
            "r2_jit_square",
            FixtureProgram::Square,
            vec![FixtureTransform::Jit],
            FixtureFamily::Jit,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
            }],
        ),
        // 51. Identity on various ranks
        make_lax_case(
            "r3_identity",
            FixtureProgram::Identity,
            vec![FixtureTransform::Jit],
            FixtureFamily::Jit,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3, 2],
                values: (0..12).map(|i| i as f64).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3, 2],
                values: (0..12).map(|i| i as f64).collect(),
            }],
        ),
        // 52. log1p on 2x2
        make_lax_case(
            "r2_log1p_2x2",
            FixtureProgram::LaxLog1p,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, std::f64::consts::E - 1.0, 99.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    0.0,
                    2.0_f64.ln(),
                    1.0, // ln(e) = 1
                    100.0_f64.ln(),
                ],
            }],
        ),
        // 53. expm1 on 2x2
        make_lax_case(
            "r2_expm1_2x2",
            FixtureProgram::LaxExpm1,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, -1.0, 0.001],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    0.0,
                    std::f64::consts::E - 1.0,
                    (-1.0_f64).exp() - 1.0,
                    (0.001_f64).exp_m1(),
                ],
            }],
        ),
        // 54. erfc on 2x2
        make_lax_case(
            "r2_erfc_2x2",
            FixtureProgram::LaxErfc,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, -1.0, 3.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    1.0,
                    1.0 - 0.842700792949715,
                    1.0 + 0.842700792949715,
                    1.0 - 0.9999779095030014,
                ],
            }],
        ),
        // 55. Matrix broadcasting [4,1] + [1,4] -> [4,4]
        make_lax_case(
            "r2_add_broadcast_4x1_1x4",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![4, 1],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![1, 4],
                    values: vec![10.0, 20.0, 30.0, 40.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![4, 4],
                values: vec![
                    11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0, 14.0,
                    24.0, 34.0, 44.0,
                ],
            }],
        ),
        // 56. Empty vector edge case (shape [0])
        make_lax_case(
            "r1_neg_empty",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::VectorF64 { values: vec![] }],
            vec![FixtureValue::VectorF64 { values: vec![] }],
        ),
    ];

    TransformFixtureBundle {
        schema_version: "frankenjax.transform-fixture-bundle.v1".to_owned(),
        generated_by: "multirank_conformance".to_owned(),
        generated_at_unix_ms: 0,
        cases,
    }
}

fn fixture_shape(value: &FixtureValue) -> Vec<u32> {
    match value {
        FixtureValue::ScalarF64 { .. } | FixtureValue::ScalarI64 { .. } => Vec::new(),
        FixtureValue::VectorF64 { values } => vec![values.len() as u32],
        FixtureValue::VectorI64 { values } => vec![values.len() as u32],
        FixtureValue::TensorF64 { shape, .. }
        | FixtureValue::TensorI64 { shape, .. }
        | FixtureValue::TensorBool { shape, .. } => shape.clone(),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn make_tensor_value(dtype: DType, shape: &[u32], elements: Vec<Literal>) -> Value {
    Value::Tensor(
        TensorValue::new(
            dtype,
            Shape {
                dims: shape.to_vec(),
            },
            elements,
        )
        .expect("tensor should be constructible"),
    )
}

fn tensor_i32(shape: &[u32], values: &[i64]) -> Value {
    make_tensor_value(
        DType::I32,
        shape,
        values.iter().copied().map(Literal::I64).collect(),
    )
}

fn tensor_i64(shape: &[u32], values: &[i64]) -> Value {
    make_tensor_value(
        DType::I64,
        shape,
        values.iter().copied().map(Literal::I64).collect(),
    )
}

fn tensor_u32(shape: &[u32], values: &[u32]) -> Value {
    make_tensor_value(
        DType::U32,
        shape,
        values.iter().copied().map(Literal::U32).collect(),
    )
}

fn tensor_u64(shape: &[u32], values: &[u64]) -> Value {
    make_tensor_value(
        DType::U64,
        shape,
        values.iter().copied().map(Literal::U64).collect(),
    )
}

fn tensor_f16(shape: &[u32], values: &[f32]) -> Value {
    make_tensor_value(
        DType::F16,
        shape,
        values.iter().copied().map(Literal::from_f16_f32).collect(),
    )
}

fn tensor_bf16(shape: &[u32], values: &[f32]) -> Value {
    make_tensor_value(
        DType::BF16,
        shape,
        values.iter().copied().map(Literal::from_bf16_f32).collect(),
    )
}

fn tensor_f32(shape: &[u32], values: &[f64]) -> Value {
    make_tensor_value(
        DType::F32,
        shape,
        values.iter().copied().map(Literal::from_f64).collect(),
    )
}

fn tensor_f64(shape: &[u32], values: &[f64]) -> Value {
    make_tensor_value(
        DType::F64,
        shape,
        values.iter().copied().map(Literal::from_f64).collect(),
    )
}

fn tensor_bool(shape: &[u32], values: &[bool]) -> Value {
    make_tensor_value(
        DType::Bool,
        shape,
        values.iter().copied().map(Literal::Bool).collect(),
    )
}

fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::U32 => "u32",
        DType::U64 => "u64",
        DType::Bool => "bool",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
    }
}

fn input_dtype_labels(inputs: &[Value]) -> Vec<String> {
    inputs
        .iter()
        .map(|value| dtype_label(value.dtype()).to_owned())
        .collect()
}

fn assert_tensor_dtype(value: &Value, expected_dtype: DType) {
    let tensor = value.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.dtype, expected_dtype);
}

fn literals_match(expected: Literal, actual: Literal, dtype: DType, atol: f64) -> bool {
    match dtype {
        DType::Bool => matches!(actual, Literal::Bool(v) if expected == Literal::Bool(v)),
        DType::I32 | DType::I64 => expected.as_i64() == actual.as_i64(),
        DType::U32 | DType::U64 => expected.as_u64() == actual.as_u64(),
        DType::BF16 | DType::F16 | DType::F32 | DType::F64 => expected
            .as_f64()
            .zip(actual.as_f64())
            .is_some_and(|(e, a)| (e - a).abs() <= atol),
        DType::Complex64 | DType::Complex128 => false,
    }
}

fn values_match(expected: &Value, actual: &Value, atol: f64) -> bool {
    match (expected, actual) {
        (Value::Scalar(expected_lit), Value::Scalar(actual_lit)) => {
            literals_match(*expected_lit, *actual_lit, expected.dtype(), atol)
        }
        (Value::Tensor(expected_tensor), Value::Tensor(actual_tensor)) => {
            expected_tensor.dtype == actual_tensor.dtype
                && expected_tensor.shape == actual_tensor.shape
                && expected_tensor.elements.len() == actual_tensor.elements.len()
                && expected_tensor
                    .elements
                    .iter()
                    .copied()
                    .zip(actual_tensor.elements.iter().copied())
                    .all(|(expected_lit, actual_lit)| {
                        literals_match(expected_lit, actual_lit, expected_tensor.dtype, atol)
                    })
        }
        _ => false,
    }
}

// ── Structural validation tests ────────────────────────────────────

#[test]
fn test_fixture_total_count() {
    let bundle = build_multirank_fixture_bundle();
    assert!(
        bundle.cases.len() >= 50,
        "need 50+ fixtures, got {}",
        bundle.cases.len()
    );
}

#[test]
fn test_fixture_unique_case_ids() {
    let bundle = build_multirank_fixture_bundle();
    let mut ids: Vec<&str> = bundle.cases.iter().map(|c| c.case_id.as_str()).collect();
    ids.sort();
    let before = ids.len();
    ids.dedup();
    assert_eq!(before, ids.len(), "duplicate case IDs found");
}

#[test]
fn test_fixture_rank0_scalar() {
    let bundle = build_multirank_fixture_bundle();
    let r0_count = bundle
        .cases
        .iter()
        .filter(|c| c.args.iter().all(|a| a.rank() == 0))
        .count();
    assert!(r0_count >= 3, "need >=3 rank-0 cases, got {r0_count}");
}

#[test]
fn test_fixture_rank1_vector() {
    let bundle = build_multirank_fixture_bundle();
    let r1_count = bundle
        .cases
        .iter()
        .filter(|c| c.args.iter().any(|a| a.rank() == 1))
        .count();
    assert!(r1_count >= 3, "need >=3 rank-1 cases, got {r1_count}");
}

#[test]
fn test_fixture_rank2_matrix() {
    let bundle = build_multirank_fixture_bundle();
    let r2_count = bundle
        .cases
        .iter()
        .filter(|c| c.args.iter().any(|a| a.rank() == 2))
        .count();
    assert!(r2_count >= 10, "need >=10 rank-2 cases, got {r2_count}");
}

#[test]
fn test_fixture_rank3_tensor() {
    let bundle = build_multirank_fixture_bundle();
    let r3_count = bundle
        .cases
        .iter()
        .filter(|c| c.args.iter().any(|a| a.rank() == 3))
        .count();
    assert!(r3_count >= 3, "need >=3 rank-3 cases, got {r3_count}");
}

#[test]
fn test_fixture_rank4_batch_matrix() {
    let bundle = build_multirank_fixture_bundle();
    let r4_count = bundle
        .cases
        .iter()
        .filter(|c| c.args.iter().any(|a| a.rank() == 4))
        .count();
    assert!(r4_count >= 2, "need >=2 rank-4 cases, got {r4_count}");
}

#[test]
fn test_fixture_shape_mismatch_detected() {
    let expected = FixtureValue::TensorF64 {
        shape: vec![2, 2],
        values: vec![1.0, 2.0, 3.0, 4.0],
    };
    let actual = FixtureValue::TensorF64 {
        shape: vec![4],
        values: vec![1.0, 2.0, 3.0, 4.0],
    }
    .to_runtime_value()
    .expect("runtime conversion should succeed");
    assert!(
        !expected.approx_matches(&actual, 1e-6, 1e-6),
        "shape mismatch should not compare as matched"
    );
}

#[test]
fn test_fixture_broadcast_shapes() {
    let bundle = build_multirank_fixture_bundle();
    let has_broadcast_pattern = bundle.cases.iter().any(|case| {
        let shapes: Vec<Vec<u32>> = case.args.iter().map(fixture_shape).collect();
        shapes.as_slice() == [vec![4, 1], vec![1, 4]]
    });
    assert!(
        has_broadcast_pattern,
        "expected at least one [4,1] + [1,4] broadcast fixture"
    );
}

#[test]
fn test_fixture_empty_tensor() {
    let bundle = build_multirank_fixture_bundle();
    let has_empty = bundle
        .cases
        .iter()
        .any(|case| case.args.iter().any(|arg| fixture_shape(arg) == vec![0]));
    assert!(has_empty, "expected at least one shape [0] fixture case");
}

#[test]
fn prop_fixture_covers_all_ranks_0_to_4() {
    let bundle = build_multirank_fixture_bundle();
    let mut seen = [false; 5];
    for case in &bundle.cases {
        for rank in case.args.iter().map(FixtureValue::rank) {
            if rank <= 4 {
                seen[rank] = true;
            }
        }
    }
    assert!(
        seen.into_iter().all(|present| present),
        "fixture bundle should include ranks 0..=4"
    );
}

#[test]
fn test_fixture_serializable() {
    let bundle = build_multirank_fixture_bundle();
    let json = serde_json::to_string_pretty(&bundle).expect("serialization failed");
    let _roundtrip: TransformFixtureBundle =
        serde_json::from_str(&json).expect("deserialization failed");
}

#[test]
fn test_fixture_all_have_expected() {
    let bundle = build_multirank_fixture_bundle();
    for case in &bundle.cases {
        assert!(
            !case.expected.is_empty(),
            "case {} has no expected output",
            case.case_id
        );
    }
}

#[test]
fn test_fixture_tight_tolerances() {
    let bundle = build_multirank_fixture_bundle();
    for case in &bundle.cases {
        assert!(
            case.atol <= 1e-6,
            "case {} has atol {} > 1e-6",
            case.case_id,
            case.atol
        );
        assert!(
            case.rtol <= 1e-6,
            "case {} has rtol {} > 1e-6",
            case.case_id,
            case.rtol
        );
    }
}

#[test]
fn test_fixture_args_runtime_conversion() {
    let bundle = build_multirank_fixture_bundle();
    for case in &bundle.cases {
        for (i, arg) in case.args.iter().enumerate() {
            arg.to_runtime_value().unwrap_or_else(|e| {
                panic!("case {} arg[{i}] conversion failed: {e}", case.case_id)
            });
        }
    }
}

#[test]
fn test_fixture_expected_runtime_conversion() {
    let bundle = build_multirank_fixture_bundle();
    for case in &bundle.cases {
        for (i, exp) in case.expected.iter().enumerate() {
            exp.to_runtime_value().unwrap_or_else(|e| {
                panic!("case {} expected[{i}] conversion failed: {e}", case.case_id)
            });
        }
    }
}

#[test]
fn test_fixture_tensor_shape_consistency() {
    let bundle = build_multirank_fixture_bundle();
    for case in &bundle.cases {
        for arg in &case.args {
            if let FixtureValue::TensorF64 { shape, values } = arg {
                let expected_len: u32 = shape.iter().product();
                assert_eq!(
                    values.len() as u32,
                    expected_len,
                    "case {} arg shape {:?} expects {} elements, got {}",
                    case.case_id,
                    shape,
                    expected_len,
                    values.len()
                );
            }
        }
        for exp in &case.expected {
            if let FixtureValue::TensorF64 { shape, values } = exp {
                let expected_len: u32 = shape.iter().product();
                assert_eq!(
                    values.len() as u32,
                    expected_len,
                    "case {} expected shape {:?} expects {} elements, got {}",
                    case.case_id,
                    shape,
                    expected_len,
                    values.len()
                );
            }
        }
    }
}

#[test]
fn test_fixture_covers_multiple_families() {
    let bundle = build_multirank_fixture_bundle();
    let families: std::collections::HashSet<_> = bundle.cases.iter().map(|c| c.family).collect();
    assert!(families.len() >= 3, "need >=3 families, got {:?}", families);
}

#[test]
fn test_fixture_singleton_dimensions() {
    let bundle = build_multirank_fixture_bundle();
    let singleton_count = bundle
        .cases
        .iter()
        .filter(|c| {
            c.args.iter().any(|a| match a {
                FixtureValue::TensorF64 { shape, .. } => shape.contains(&1),
                _ => false,
            })
        })
        .count();
    assert!(
        singleton_count >= 3,
        "need >=3 singleton-dim cases, got {singleton_count}"
    );
}

#[test]
fn e2e_multirank_fixture_sweep() {
    let cfg = HarnessConfig::default_paths();
    let bundle = build_multirank_fixture_bundle();
    let report = run_transform_fixture_bundle(&cfg, &bundle);
    let report_by_case: HashMap<&str, bool> = report
        .reports
        .iter()
        .map(|case_report| (case_report.case_id.as_str(), case_report.matched))
        .collect();

    let mut rank_passes: HashMap<usize, (usize, usize)> = HashMap::new();
    let entries: Vec<serde_json::Value> = bundle
        .cases
        .iter()
        .map(|case| {
            let input_shapes: Vec<Vec<u32>> = case.args.iter().map(fixture_shape).collect();
            let output_shape = case.expected.first().map_or_else(Vec::new, fixture_shape);
            let rank = input_shapes.iter().map(Vec::len).max().unwrap_or(0);
            let matched = *report_by_case
                .get(case.case_id.as_str())
                .expect("case report should exist");
            let stat = rank_passes.entry(rank).or_insert((0, 0));
            stat.0 += 1;
            if matched {
                stat.1 += 1;
            }
            json!({
                "rank": rank,
                "primitive": format!("{:?}", case.program),
                "input_shapes": input_shapes,
                "output_shape": output_shape,
                "oracle_match": matched,
                "abs_error": serde_json::Value::Null,
                "pass": matched
            })
        })
        .collect();

    let mut rank_summary: HashMap<String, serde_json::Value> = HashMap::new();
    for (rank, (total, passed)) in rank_passes {
        rank_summary.insert(
            rank.to_string(),
            json!({
                "total": total,
                "passed": passed,
                "pass_rate": if total == 0 { 0.0 } else { passed as f64 / total as f64 }
            }),
        );
    }

    let forensic_log = json!({
        "scenario": "e2e_multirank_fixture_sweep",
        "generated_at_unix_ms": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_millis(),
        "total_cases": report.total_cases,
        "matched_cases": report.matched_cases,
        "mismatched_cases": report.mismatched_cases,
        "rank_summary": rank_summary,
        "entries": entries
    });

    let output_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/e2e/e2e_multirank_fixtures.e2e.json");
    fs::create_dir_all(output_path.parent().expect("output parent should exist"))
        .expect("should create artifacts/e2e");
    fs::write(
        &output_path,
        serde_json::to_string_pretty(&forensic_log).expect("forensic log should serialize"),
    )
    .expect("should write forensic log");

    assert_eq!(
        report.total_cases,
        bundle.cases.len(),
        "report should include every multirank fixture case"
    );
}

// ── Mixed dtype + type promotion conformance (V2-CONFORM-08) ──────

#[test]
fn test_dtype_promotion_f32_f64() {
    let out = eval_primitive(
        Primitive::Add,
        &[
            tensor_f32(&[2], &[1.5, 2.25]),
            tensor_f64(&[2], &[0.25, 0.75]),
        ],
        &no_params(),
    )
    .expect("f32 + f64 should evaluate");

    assert_tensor_dtype(&out, DType::F64);
    assert!(values_match(&tensor_f64(&[2], &[1.75, 3.0]), &out, 1e-12));
}

#[test]
fn test_dtype_promotion_i32_f32() {
    let out = eval_primitive(
        Primitive::Add,
        &[
            tensor_i32(&[3], &[1, -2, 3]),
            tensor_f32(&[3], &[0.5, 1.5, 2.0]),
        ],
        &no_params(),
    )
    .expect("i32 + f32 should evaluate");

    assert_tensor_dtype(&out, DType::F32);
    assert!(values_match(
        &tensor_f32(&[3], &[1.5, -0.5, 5.0]),
        &out,
        1e-9
    ));
}

#[test]
fn test_dtype_promotion_bool_i32() {
    let out = eval_primitive(
        Primitive::Add,
        &[
            tensor_bool(&[4], &[true, false, true, false]),
            tensor_i32(&[4], &[10, 20, -1, 0]),
        ],
        &no_params(),
    )
    .expect("bool + i32 should evaluate");

    assert_tensor_dtype(&out, DType::I32);
    assert!(values_match(&tensor_i32(&[4], &[11, 20, 0, 0]), &out, 0.0));
}

#[test]
fn test_dtype_promotion_matrix() {
    struct PromotionCase {
        name: &'static str,
        lhs: Value,
        rhs: Value,
        expected: DType,
    }

    let cases = vec![
        PromotionCase {
            name: "f32_f64",
            lhs: tensor_f32(&[], &[1.0]),
            rhs: tensor_f64(&[], &[2.0]),
            expected: DType::F64,
        },
        PromotionCase {
            name: "i32_f32",
            lhs: tensor_i32(&[], &[7]),
            rhs: tensor_f32(&[], &[0.5]),
            expected: DType::F32,
        },
        PromotionCase {
            name: "bool_i32",
            lhs: tensor_bool(&[], &[true]),
            rhs: tensor_i32(&[], &[3]),
            expected: DType::I32,
        },
        PromotionCase {
            name: "i64_f64",
            lhs: tensor_i64(&[], &[8]),
            rhs: tensor_f64(&[], &[0.25]),
            expected: DType::F64,
        },
        PromotionCase {
            name: "u32_i32",
            lhs: tensor_u32(&[], &[8]),
            rhs: tensor_i32(&[], &[3]),
            expected: DType::I64,
        },
        PromotionCase {
            name: "u32_f32",
            lhs: tensor_u32(&[], &[8]),
            rhs: tensor_f32(&[], &[3.0]),
            expected: DType::F64,
        },
        PromotionCase {
            name: "u64_i64",
            lhs: tensor_u64(&[], &[8]),
            rhs: tensor_i64(&[], &[3]),
            expected: DType::F64,
        },
        PromotionCase {
            name: "bf16_f16",
            lhs: tensor_bf16(&[], &[1.0]),
            rhs: tensor_f16(&[], &[2.0]),
            expected: DType::F32,
        },
        PromotionCase {
            name: "bf16_f64",
            lhs: tensor_bf16(&[], &[1.0]),
            rhs: tensor_f64(&[], &[2.0]),
            expected: DType::F64,
        },
    ];

    for case in cases {
        let out = eval_primitive(Primitive::Add, &[case.lhs, case.rhs], &no_params())
            .unwrap_or_else(|err| panic!("promotion case {} failed: {err}", case.name));
        assert_tensor_dtype(&out, case.expected);
    }
}

#[test]
fn test_dtype_cast_explicit() {
    let mut params = BTreeMap::new();
    params.insert("new_dtype".to_owned(), "f64".to_owned());
    let i64_bits_for_one = 4_607_182_418_800_017_408_i64;
    let out = eval_primitive(
        Primitive::BitcastConvertType,
        &[Value::scalar_i64(i64_bits_for_one)],
        &params,
    )
    .expect("bitcast convert should evaluate");

    assert_eq!(out.as_f64_scalar(), Some(1.0));
}

#[test]
fn test_mixed_dtype_binary_ops() {
    let add_out = eval_primitive(
        Primitive::Add,
        &[
            tensor_i64(&[3], &[1, 2, 3]),
            tensor_f64(&[3], &[0.5, 1.5, -1.0]),
        ],
        &no_params(),
    )
    .expect("add(i64, f64) should evaluate");
    assert_tensor_dtype(&add_out, DType::F64);
    assert!(values_match(
        &tensor_f64(&[3], &[1.5, 3.5, 2.0]),
        &add_out,
        1e-12
    ));

    let mul_out = eval_primitive(
        Primitive::Mul,
        &[
            tensor_bool(&[4], &[true, false, true, false]),
            tensor_i64(&[4], &[10, 20, -3, 1]),
        ],
        &no_params(),
    )
    .expect("mul(bool, i64) should evaluate");
    assert_tensor_dtype(&mul_out, DType::I64);
    assert!(values_match(
        &tensor_i64(&[4], &[10, 0, -3, 0]),
        &mul_out,
        0.0
    ));

    let div_out = eval_primitive(
        Primitive::Div,
        &[tensor_i64(&[3], &[9, 3, 1]), Value::scalar_f64(2.0)],
        &no_params(),
    )
    .expect("div(i64, f64) should evaluate");
    assert_tensor_dtype(&div_out, DType::F64);
    assert!(values_match(
        &tensor_f64(&[3], &[4.5, 1.5, 0.5]),
        &div_out,
        1e-12
    ));
}

#[test]
fn test_mixed_dtype_comparison() {
    let lt_out = eval_primitive(
        Primitive::Lt,
        &[
            tensor_f32(&[4], &[0.25, 2.0, -2.0, 9.0]),
            tensor_i32(&[4], &[1, 1, -3, 9]),
        ],
        &no_params(),
    )
    .expect("lt(f32, i32) should evaluate");

    assert_tensor_dtype(&lt_out, DType::Bool);
    assert!(values_match(
        &tensor_bool(&[4], &[true, false, false, false]),
        &lt_out,
        0.0
    ));
}

#[test]
fn test_dtype_preservation_unary() {
    let sin_out = eval_primitive(
        Primitive::Sin,
        &[tensor_f32(&[2], &[0.0, std::f64::consts::FRAC_PI_2])],
        &no_params(),
    )
    .expect("sin(f32 tensor) should evaluate");
    assert_tensor_dtype(&sin_out, DType::F32);
    assert!(values_match(
        &tensor_f32(&[2], &[0.0, 1.0]),
        &sin_out,
        1e-12
    ));

    let neg_out = eval_primitive(
        Primitive::Neg,
        &[tensor_i32(&[3], &[1, -2, 3])],
        &no_params(),
    )
    .expect("neg(i32 tensor) should evaluate");
    assert_tensor_dtype(&neg_out, DType::I32);
    assert!(values_match(&tensor_i32(&[3], &[-1, 2, -3]), &neg_out, 0.0));
}

#[test]
fn e2e_mixed_dtype_conformance() {
    let mut forensic_entries = Vec::<serde_json::Value>::new();
    let mut promotion_log_entries = Vec::<serde_json::Value>::new();
    let mut total_cases = 0_usize;
    let mut passed_cases = 0_usize;

    enum MixedExpectation<'a> {
        Value(Value),
        ErrorContains(&'a str),
    }

    let mut record_case = |fixture_id: &str,
                           promotion_rule: &str,
                           inputs: Vec<Value>,
                           expected: MixedExpectation<'_>,
                           actual: Result<Value, String>| {
        total_cases += 1;
        let input_dtypes = input_dtype_labels(&inputs);

        let (expected_output_dtype, actual_output_dtype, values_match_flag, pass_flag) =
            match expected {
                MixedExpectation::Value(expected_value) => {
                    let expected_dtype = dtype_label(expected_value.dtype()).to_owned();
                    match actual {
                        Ok(actual_value) => {
                            let matches = values_match(&expected_value, &actual_value, 1e-9);
                            let actual_dtype = dtype_label(actual_value.dtype()).to_owned();
                            let pass = matches && actual_dtype == expected_dtype;
                            (expected_dtype, actual_dtype, matches, pass)
                        }
                        Err(_) => (expected_dtype, "error".to_owned(), false, false),
                    }
                }
                MixedExpectation::ErrorContains(expected_error_contains) => match actual {
                    Ok(value) => (
                        "error".to_owned(),
                        dtype_label(value.dtype()).to_owned(),
                        false,
                        false,
                    ),
                    Err(err) => {
                        let matched = err.contains(expected_error_contains);
                        ("error".to_owned(), "error".to_owned(), matched, matched)
                    }
                },
            };

        if pass_flag {
            passed_cases += 1;
        }

        forensic_entries.push(json!({
            "fixture_id": fixture_id,
            "input_dtypes": input_dtypes.clone(),
            "expected_output_dtype": expected_output_dtype.clone(),
            "actual_output_dtype": actual_output_dtype.clone(),
            "values_match": values_match_flag,
            "pass": pass_flag
        }));

        promotion_log_entries.push(json!({
            "test_name": fixture_id,
            "input_dtypes": input_dtypes,
            "promotion_rule": promotion_rule,
            "expected_dtype": expected_output_dtype,
            "actual_dtype": actual_output_dtype,
            "pass": pass_flag
        }));
    };

    macro_rules! record_value_case {
        ($fixture_id:expr, $promotion_rule:expr, $inputs:expr, $expected:expr, $actual:expr $(,)?) => {
            record_case(
                $fixture_id,
                $promotion_rule,
                $inputs,
                MixedExpectation::Value($expected),
                $actual,
            )
        };
    }

    macro_rules! record_error_case {
        ($fixture_id:expr, $promotion_rule:expr, $inputs:expr, $expected_error_contains:expr, $actual:expr $(,)?) => {
            record_case(
                $fixture_id,
                $promotion_rule,
                $inputs,
                MixedExpectation::ErrorContains($expected_error_contains),
                $actual,
            )
        };
    }

    record_value_case!(
        "md_add_i64_f64_scalar",
        "i64 + f64 -> f64",
        vec![tensor_i64(&[], &[3]), tensor_f64(&[], &[0.5])],
        tensor_f64(&[], &[3.5]),
        eval_primitive(
            Primitive::Add,
            &[tensor_i64(&[], &[3]), tensor_f64(&[], &[0.5])],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_add_i64_f64_vector",
        "i64 + f64 -> f64",
        vec![
            tensor_i64(&[3], &[1, 2, 3]),
            tensor_f64(&[3], &[0.5, 1.5, -1.0])
        ],
        tensor_f64(&[3], &[1.5, 3.5, 2.0]),
        eval_primitive(
            Primitive::Add,
            &[
                tensor_i64(&[3], &[1, 2, 3]),
                tensor_f64(&[3], &[0.5, 1.5, -1.0]),
            ],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_add_i64_f64_matrix",
        "i64 + f64 -> f64",
        vec![
            tensor_i64(&[2, 2], &[1, 2, 3, 4]),
            tensor_f64(&[2, 2], &[0.25, -0.5, 1.0, 2.5]),
        ],
        tensor_f64(&[2, 2], &[1.25, 1.5, 4.0, 6.5]),
        eval_primitive(
            Primitive::Add,
            &[
                tensor_i64(&[2, 2], &[1, 2, 3, 4]),
                tensor_f64(&[2, 2], &[0.25, -0.5, 1.0, 2.5]),
            ],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_add_i64_f64_broadcast_rhs_scalar",
        "i64 + f64 -> f64",
        vec![tensor_i64(&[3], &[1, 2, 3]), Value::scalar_f64(2.0)],
        tensor_f64(&[3], &[3.0, 4.0, 5.0]),
        eval_primitive(
            Primitive::Add,
            &[tensor_i64(&[3], &[1, 2, 3]), Value::scalar_f64(2.0)],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_add_i64_f64_broadcast_lhs_scalar",
        "f64 + i64 -> f64",
        vec![Value::scalar_f64(2.0), tensor_i64(&[3], &[1, 2, 3])],
        tensor_f64(&[3], &[3.0, 4.0, 5.0]),
        eval_primitive(
            Primitive::Add,
            &[Value::scalar_f64(2.0), tensor_i64(&[3], &[1, 2, 3])],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_mul_bool_i64_vector",
        "bool * i64 -> i64",
        vec![
            tensor_bool(&[4], &[true, false, true, false]),
            tensor_i64(&[4], &[10, 20, -3, 1]),
        ],
        tensor_i64(&[4], &[10, 0, -3, 0]),
        eval_primitive(
            Primitive::Mul,
            &[
                tensor_bool(&[4], &[true, false, true, false]),
                tensor_i64(&[4], &[10, 20, -3, 1]),
            ],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_mul_bool_i64_matrix",
        "bool * i64 -> i64",
        vec![
            tensor_bool(&[2, 2], &[true, false, false, true]),
            tensor_i64(&[2, 2], &[5, 6, 7, 8]),
        ],
        tensor_i64(&[2, 2], &[5, 0, 0, 8]),
        eval_primitive(
            Primitive::Mul,
            &[
                tensor_bool(&[2, 2], &[true, false, false, true]),
                tensor_i64(&[2, 2], &[5, 6, 7, 8]),
            ],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_mul_bool_i64_scalar",
        "bool * i64 -> i64",
        vec![tensor_bool(&[], &[true]), tensor_i64(&[], &[9])],
        tensor_i64(&[], &[9]),
        eval_primitive(
            Primitive::Mul,
            &[tensor_bool(&[], &[true]), tensor_i64(&[], &[9])],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_div_i64_f64_scalar",
        "i64 / f64 -> f64",
        vec![tensor_i64(&[], &[9]), tensor_f64(&[], &[2.0])],
        tensor_f64(&[], &[4.5]),
        eval_primitive(
            Primitive::Div,
            &[tensor_i64(&[], &[9]), tensor_f64(&[], &[2.0])],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_div_i64_f64_vector",
        "i64 / f64 -> f64",
        vec![tensor_i64(&[3], &[9, 3, 1]), Value::scalar_f64(2.0)],
        tensor_f64(&[3], &[4.5, 1.5, 0.5]),
        eval_primitive(
            Primitive::Div,
            &[tensor_i64(&[3], &[9, 3, 1]), Value::scalar_f64(2.0)],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_div_i64_f64_matrix",
        "i64 / f64 -> f64",
        vec![
            tensor_i64(&[2, 2], &[10, 20, 30, 40]),
            tensor_f64(&[2, 2], &[2.0, 5.0, 6.0, 8.0]),
        ],
        tensor_f64(&[2, 2], &[5.0, 4.0, 5.0, 5.0]),
        eval_primitive(
            Primitive::Div,
            &[
                tensor_i64(&[2, 2], &[10, 20, 30, 40]),
                tensor_f64(&[2, 2], &[2.0, 5.0, 6.0, 8.0]),
            ],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_gt_f64_i64_vector",
        "f64 > i64 -> bool",
        vec![
            tensor_f64(&[4], &[0.5, 2.5, 3.0, -1.0]),
            tensor_i64(&[4], &[0, 3, 2, 0]),
        ],
        tensor_bool(&[4], &[true, false, true, false]),
        eval_primitive(
            Primitive::Gt,
            &[
                tensor_f64(&[4], &[0.5, 2.5, 3.0, -1.0]),
                tensor_i64(&[4], &[0, 3, 2, 0]),
            ],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_gt_f64_i64_matrix",
        "f64 > i64 -> bool",
        vec![
            tensor_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
            tensor_i64(&[2, 2], &[0, 2, 5, 3]),
        ],
        tensor_bool(&[2, 2], &[true, false, false, true]),
        eval_primitive(
            Primitive::Gt,
            &[
                tensor_f64(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                tensor_i64(&[2, 2], &[0, 2, 5, 3]),
            ],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_eq_i64_f64_scalar",
        "i64 == f64 -> bool",
        vec![tensor_i64(&[], &[7]), tensor_f64(&[], &[7.0])],
        tensor_bool(&[], &[true]),
        eval_primitive(
            Primitive::Eq,
            &[tensor_i64(&[], &[7]), tensor_f64(&[], &[7.0])],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_reduce_sum_i64",
        "reduce_sum(i64) -> i64",
        vec![tensor_i64(&[4], &[1, 2, 3, 4])],
        Value::scalar_i64(10),
        eval_primitive(
            Primitive::ReduceSum,
            &[tensor_i64(&[4], &[1, 2, 3, 4])],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_value_case!(
        "md_reduce_sum_f64",
        "reduce_sum(f64) -> f64",
        vec![tensor_f64(&[4], &[1.0, 2.0, 3.0, 4.0])],
        Value::scalar_f64(10.0),
        eval_primitive(
            Primitive::ReduceSum,
            &[tensor_f64(&[4], &[1.0, 2.0, 3.0, 4.0])],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_error_case!(
        "md_reduce_sum_bool_error",
        "reduce_sum(bool) -> error",
        vec![tensor_bool(&[3], &[true, false, true])],
        "expected numeric tensor",
        eval_primitive(
            Primitive::ReduceSum,
            &[tensor_bool(&[3], &[true, false, true])],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    record_error_case!(
        "md_expr_add_then_mul_bool_vector",
        "(i64 + f64) * bool -> error",
        vec![
            tensor_i64(&[3], &[1, 2, 3]),
            tensor_f64(&[3], &[0.5, 1.5, -1.0]),
            tensor_bool(&[3], &[true, false, true]),
        ],
        "expected numeric rhs",
        eval_primitive(
            Primitive::Add,
            &[
                tensor_i64(&[3], &[1, 2, 3]),
                tensor_f64(&[3], &[0.5, 1.5, -1.0]),
            ],
            &no_params(),
        )
        .map_err(|e| e.to_string())
        .and_then(|sum| {
            eval_primitive(
                Primitive::Mul,
                &[sum, tensor_bool(&[3], &[true, false, true])],
                &no_params(),
            )
            .map_err(|e| e.to_string())
        }),
    );

    record_error_case!(
        "md_expr_add_then_mul_bool_matrix",
        "(i64 + f64) * bool -> error",
        vec![
            tensor_i64(&[2, 2], &[1, 2, 3, 4]),
            tensor_f64(&[2, 2], &[0.25, -0.5, 1.0, 2.5]),
            tensor_bool(&[2, 2], &[true, false, false, true]),
        ],
        "expected numeric rhs",
        eval_primitive(
            Primitive::Add,
            &[
                tensor_i64(&[2, 2], &[1, 2, 3, 4]),
                tensor_f64(&[2, 2], &[0.25, -0.5, 1.0, 2.5]),
            ],
            &no_params(),
        )
        .map_err(|e| e.to_string())
        .and_then(|sum| {
            eval_primitive(
                Primitive::Mul,
                &[sum, tensor_bool(&[2, 2], &[true, false, false, true])],
                &no_params(),
            )
            .map_err(|e| e.to_string())
        }),
    );

    record_value_case!(
        "md_grad_square_i64_input",
        "grad(i64 -> f64) square",
        vec![Value::scalar_i64(8)],
        Value::scalar_f64(16.0),
        grad_jaxpr(&build_program(ProgramSpec::Square), &[Value::scalar_i64(8)])
            .map_err(|e| e.to_string())
            .and_then(|mut grads| {
                grads
                    .drain(..)
                    .next()
                    .ok_or_else(|| "missing gradient output".to_owned())
            }),
    );

    record_value_case!(
        "md_grad_square_plus_linear_i64_input",
        "grad(i64 -> f64) square_plus_linear",
        vec![Value::scalar_i64(5)],
        Value::scalar_f64(12.0),
        grad_jaxpr(
            &build_program(ProgramSpec::SquarePlusLinear),
            &[Value::scalar_i64(5)]
        )
        .map_err(|e| e.to_string())
        .and_then(|mut grads| {
            grads
                .drain(..)
                .next()
                .ok_or_else(|| "missing gradient output".to_owned())
        }),
    );

    record_value_case!(
        "md_lt_f32_i32_vector",
        "f32 < i32 -> bool",
        vec![
            tensor_f32(&[4], &[0.25, 2.0, -2.0, 9.0]),
            tensor_i32(&[4], &[1, 1, -3, 9]),
        ],
        tensor_bool(&[4], &[true, false, false, false]),
        eval_primitive(
            Primitive::Lt,
            &[
                tensor_f32(&[4], &[0.25, 2.0, -2.0, 9.0]),
                tensor_i32(&[4], &[1, 1, -3, 9]),
            ],
            &no_params(),
        )
        .map_err(|e| e.to_string()),
    );

    assert!(
        total_cases >= 20,
        "expected >=20 mixed-dtype cases, got {total_cases}"
    );
    assert_eq!(
        passed_cases, total_cases,
        "all mixed-dtype conformance cases should pass"
    );

    let forensic_log = json!({
        "scenario": "e2e_mixed_dtype_conformance",
        "generated_at_unix_ms": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_millis(),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases.saturating_sub(passed_cases),
        "entries": forensic_entries
    });

    let e2e_output_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/e2e/e2e_mixed_dtype.e2e.json");
    fs::create_dir_all(
        e2e_output_path
            .parent()
            .expect("output parent should exist"),
    )
    .expect("should create artifacts/e2e");
    fs::write(
        &e2e_output_path,
        serde_json::to_string_pretty(&forensic_log).expect("forensic log should serialize"),
    )
    .expect("should write mixed-dtype e2e forensic log");

    let promotion_log = json!({
        "test_name": "e2e_mixed_dtype_conformance",
        "generated_at_unix_ms": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_millis(),
        "entries": promotion_log_entries
    });

    let log_output_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/testing/logs/fj-conformance/e2e_mixed_dtype_conformance.json");
    fs::create_dir_all(
        log_output_path
            .parent()
            .expect("log output parent should exist"),
    )
    .expect("should create artifacts/testing/logs/fj-conformance");
    fs::write(
        &log_output_path,
        serde_json::to_string_pretty(&promotion_log).expect("promotion log should serialize"),
    )
    .expect("should write mixed-dtype promotion log");
}
