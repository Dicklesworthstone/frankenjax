//! Metamorphic property tests for FrankenJAX correctness
//!
//! Tests invariants that hold regardless of specific values:
//! 1. grad(sum(x)) = ones_like(x)
//! 2. jit(f)(x) == f(x)
//! 3. vmap(f)(xs) == stack([f(x) for x in xs])
//! 4. double linear_transpose = identity
//! 5. scan equals explicit loop
//! 6. grad commutes with jit

use fj_api::{grad, jit, linear_transpose, value_and_grad, vmap};
use fj_core::{Atom, Equation, Jaxpr, Primitive, Value, VarId};
use fj_interpreters::eval_jaxpr;
use fj_lax::eval_scan_functional;
use proptest::prelude::*;
use std::collections::BTreeMap;

fn approx_eq_f64(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
        return true;
    }
    (a - b).abs() < tol
}

fn values_approx_eq(a: &[Value], b: &[Value], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (va, vb) in a.iter().zip(b.iter()) {
        match (va, vb) {
            (Value::Scalar(la), Value::Scalar(lb)) => {
                if let (Some(fa), Some(fb)) = (la.as_f64(), lb.as_f64()) {
                    if !approx_eq_f64(fa, fb, tol) {
                        return false;
                    }
                } else if la != lb {
                    return false;
                }
            }
            (Value::Tensor(ta), Value::Tensor(tb)) => {
                if ta.shape != tb.shape {
                    return false;
                }
                for (ea, eb) in ta.elements.iter().zip(tb.elements.iter()) {
                    if let (Some(fa), Some(fb)) = (ea.as_f64(), eb.as_f64()) {
                        if !approx_eq_f64(fa, fb, tol) {
                            return false;
                        }
                    } else if ea != eb {
                        return false;
                    }
                }
            }
            _ => return false,
        }
    }
    true
}

fn square_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Mul,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        }],
    )
}

fn add_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec::smallvec![VarId(3)],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        }],
    )
}

fn neg_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Neg,
            inputs: smallvec::smallvec![Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        }],
    )
}

fn sin_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Sin,
            inputs: smallvec::smallvec![Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        }],
    )
}

fn exp_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Exp,
            inputs: smallvec::smallvec![Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        }],
    )
}

fn reduce_sum_jaxpr() -> Jaxpr {
    let mut params = BTreeMap::new();
    params.insert("axes".to_owned(), "0".to_owned());
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::ReduceSum,
            inputs: smallvec::smallvec![Atom::Var(VarId(1))],
            outputs: smallvec::smallvec![VarId(2)],
            params,
            sub_jaxprs: vec![],
            effects: vec![],
        }],
    )
}

fn identity_jaxpr() -> Jaxpr {
    Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![])
}

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(
        fj_test_utils::property_test_case_count()
    ))]

    // =========================================================================
    // RELATION 1: grad(sum(x)) = ones_like(x)
    // =========================================================================

    #[test]
    fn metamorphic_grad_sum_equals_ones_scalar(x in -100.0f64..100.0) {
        prop_assume!(x.is_finite());
        let jaxpr = reduce_sum_jaxpr();
        let result = grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(sum) should succeed");
        let grad_val = result[0].as_f64_scalar().expect("scalar gradient");
        prop_assert!(
            approx_eq_f64(grad_val, 1.0, 1e-10),
            "grad(sum(x)) should be 1, got {} at x={}",
            grad_val,
            x
        );
    }

    #[test]
    fn metamorphic_grad_sum_equals_ones_vector(
        values in prop::collection::vec(-100.0f64..100.0, 2..10)
    ) {
        prop_assume!(values.iter().all(|v| v.is_finite()));
        let vec_val = Value::vector_f64(&values).expect("vector");
        let jaxpr = reduce_sum_jaxpr();
        let result = grad(jaxpr)
            .call(vec![vec_val])
            .expect("grad(sum) should succeed");
        let grad_tensor = result[0].as_tensor().expect("gradient tensor");
        let grad_vals = grad_tensor.to_f64_vec().expect("f64 values");
        prop_assert_eq!(
            grad_vals.len(),
            values.len(),
            "gradient should have same length as input"
        );
        for (i, g) in grad_vals.iter().enumerate() {
            prop_assert!(
                approx_eq_f64(*g, 1.0, 1e-10),
                "grad(sum(x))[{}] should be 1, got {}",
                i,
                g
            );
        }
    }

    // =========================================================================
    // RELATION 2: jit(f)(x) == f(x)
    // =========================================================================

    #[test]
    fn metamorphic_jit_transparent_square(x in -100.0f64..100.0) {
        prop_assume!(x.is_finite());
        let jaxpr = square_jaxpr();
        let jit_result = jit(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("jit(f) should succeed");
        let direct_result = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
            .expect("f should succeed");
        prop_assert!(
            values_approx_eq(&jit_result, &direct_result, 1e-10),
            "jit(f)(x) != f(x) at x={}",
            x
        );
    }

    #[test]
    fn metamorphic_jit_transparent_sin(x in -10.0f64..10.0) {
        prop_assume!(x.is_finite());
        let jaxpr = sin_jaxpr();
        let jit_result = jit(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("jit(sin) should succeed");
        let direct_result = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
            .expect("sin should succeed");
        prop_assert!(
            values_approx_eq(&jit_result, &direct_result, 1e-10),
            "jit(sin)(x) != sin(x) at x={}",
            x
        );
    }

    #[test]
    fn metamorphic_jit_transparent_exp(x in -50.0f64..50.0) {
        prop_assume!(x.is_finite());
        let jaxpr = exp_jaxpr();
        let jit_result = jit(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("jit(exp) should succeed");
        let direct_result = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
            .expect("exp should succeed");
        prop_assert!(
            values_approx_eq(&jit_result, &direct_result, 1e-10),
            "jit(exp)(x) != exp(x) at x={}",
            x
        );
    }

    #[test]
    fn metamorphic_jit_transparent_add(
        x in -100.0f64..100.0,
        y in -100.0f64..100.0
    ) {
        prop_assume!(x.is_finite() && y.is_finite());
        let jaxpr = add_jaxpr();
        let jit_result = jit(jaxpr.clone())
            .call(vec![Value::scalar_f64(x), Value::scalar_f64(y)])
            .expect("jit(add) should succeed");
        let direct_result = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x), Value::scalar_f64(y)])
            .expect("add should succeed");
        prop_assert!(
            values_approx_eq(&jit_result, &direct_result, 1e-10),
            "jit(add)(x,y) != add(x,y) at x={}, y={}",
            x,
            y
        );
    }

    // =========================================================================
    // RELATION 3: vmap(f)(xs) == stack([f(x) for x in xs])
    // =========================================================================

    #[test]
    fn metamorphic_vmap_equals_map_square(
        values in prop::collection::vec(-100.0f64..100.0, 2..10)
    ) {
        prop_assume!(values.iter().all(|v| v.is_finite()));
        let jaxpr = square_jaxpr();
        let vec_val = Value::vector_f64(&values).expect("vector");
        let vmap_result = vmap(jaxpr.clone())
            .call(vec![vec_val])
            .expect("vmap(square) should succeed");
        let vmap_tensor = vmap_result[0].as_tensor().expect("vmap tensor");
        let vmap_vals = vmap_tensor.to_f64_vec().expect("f64 values");
        for (i, &x) in values.iter().enumerate() {
            let direct = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
                .expect("direct eval")[0]
                .as_f64_scalar()
                .expect("scalar");
            prop_assert!(
                approx_eq_f64(vmap_vals[i], direct, 1e-10),
                "vmap(square)[{}] != square(x) at x={}: {} vs {}",
                i,
                x,
                vmap_vals[i],
                direct
            );
        }
    }

    #[test]
    fn metamorphic_vmap_equals_map_sin(
        values in prop::collection::vec(-10.0f64..10.0, 2..10)
    ) {
        prop_assume!(values.iter().all(|v| v.is_finite()));
        let jaxpr = sin_jaxpr();
        let vec_val = Value::vector_f64(&values).expect("vector");
        let vmap_result = vmap(jaxpr.clone())
            .call(vec![vec_val])
            .expect("vmap(sin) should succeed");
        let vmap_tensor = vmap_result[0].as_tensor().expect("vmap tensor");
        let vmap_vals = vmap_tensor.to_f64_vec().expect("f64 values");
        for (i, &x) in values.iter().enumerate() {
            let direct = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
                .expect("direct eval")[0]
                .as_f64_scalar()
                .expect("scalar");
            prop_assert!(
                approx_eq_f64(vmap_vals[i], direct, 1e-10),
                "vmap(sin)[{}] != sin(x) at x={}: {} vs {}",
                i,
                x,
                vmap_vals[i],
                direct
            );
        }
    }

    #[test]
    fn metamorphic_vmap_equals_map_neg(
        values in prop::collection::vec(-100.0f64..100.0, 2..10)
    ) {
        prop_assume!(values.iter().all(|v| v.is_finite()));
        let jaxpr = neg_jaxpr();
        let vec_val = Value::vector_f64(&values).expect("vector");
        let vmap_result = vmap(jaxpr.clone())
            .call(vec![vec_val])
            .expect("vmap(neg) should succeed");
        let vmap_tensor = vmap_result[0].as_tensor().expect("vmap tensor");
        let vmap_vals = vmap_tensor.to_f64_vec().expect("f64 values");
        for (i, &x) in values.iter().enumerate() {
            let direct = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
                .expect("direct eval")[0]
                .as_f64_scalar()
                .expect("scalar");
            prop_assert!(
                approx_eq_f64(vmap_vals[i], direct, 1e-10),
                "vmap(neg)[{}] != neg(x) at x={}: {} vs {}",
                i,
                x,
                vmap_vals[i],
                direct
            );
        }
    }

    // =========================================================================
    // RELATION 4: double linear_transpose = identity (for linear functions)
    // =========================================================================

    #[test]
    fn metamorphic_double_transpose_identity_neg(x in -100.0f64..100.0) {
        prop_assume!(x.is_finite());
        let jaxpr = neg_jaxpr();
        let primals = vec![Value::scalar_f64(x)];
        let transposed = linear_transpose(jaxpr.clone(), primals.clone())
            .expect("first transpose");
        let cotangent = Value::scalar_f64(1.0);
        let first_result = transposed.call(cotangent.clone()).expect("first call");
        let transposed2 = linear_transpose(jaxpr, first_result.clone())
            .expect("second transpose");
        let second_result = transposed2.call(cotangent).expect("second call");
        let original = first_result[0].as_f64_scalar().expect("first scalar");
        let double_t = second_result[0].as_f64_scalar().expect("second scalar");
        prop_assert!(
            approx_eq_f64(original, double_t, 1e-10),
            "double transpose != identity for neg at x={}: {} vs {}",
            x,
            original,
            double_t
        );
    }

    #[test]
    fn metamorphic_double_transpose_identity_identity(x in -100.0f64..100.0) {
        prop_assume!(x.is_finite());
        let jaxpr = identity_jaxpr();
        let primals = vec![Value::scalar_f64(x)];
        let transposed = linear_transpose(jaxpr.clone(), primals.clone())
            .expect("first transpose");
        let cotangent = Value::scalar_f64(1.0);
        let first_result = transposed.call(cotangent.clone()).expect("first call");
        let transposed2 = linear_transpose(jaxpr, first_result.clone())
            .expect("second transpose");
        let second_result = transposed2.call(cotangent).expect("second call");
        let original = first_result[0].as_f64_scalar().expect("first scalar");
        let double_t = second_result[0].as_f64_scalar().expect("second scalar");
        prop_assert!(
            approx_eq_f64(original, double_t, 1e-10),
            "double transpose != identity for identity at x={}: {} vs {}",
            x,
            original,
            double_t
        );
    }

    // =========================================================================
    // RELATION 5: scan(f, init, xs) == loop simulation
    // =========================================================================

    #[test]
    fn metamorphic_scan_equals_loop_add(
        init in -100.0f64..100.0,
        values in prop::collection::vec(-10.0f64..10.0, 1..5)
    ) {
        prop_assume!(init.is_finite() && values.iter().all(|v| v.is_finite()));
        let xs = Value::vector_f64(&values).expect("vector");
        let init_val = vec![Value::scalar_f64(init)];
        let scan_result = eval_scan_functional(
            init_val,
            &xs,
            |carry: Vec<Value>, x: Value| {
                let c = carry[0].as_f64_scalar().expect("carry f64");
                let xv = x.as_f64_scalar().expect("x f64");
                let next = Value::scalar_f64(c + xv);
                Ok((vec![next.clone()], vec![next]))
            },
            false,
        );
        if let Ok((final_carry, _outputs)) = scan_result {
            let mut loop_acc = init;
            for &x in &values {
                loop_acc += x;
            }
            let scan_val = final_carry[0].as_f64_scalar().expect("scan scalar");
            prop_assert!(
                approx_eq_f64(scan_val, loop_acc, 1e-8),
                "scan(add) != loop: {} vs {} with init={}, xs={:?}",
                scan_val,
                loop_acc,
                init,
                values
            );
        }
    }

    #[test]
    fn metamorphic_scan_equals_loop_mul(
        init in 0.1f64..10.0,
        values in prop::collection::vec(0.1f64..2.0, 1..5)
    ) {
        prop_assume!(init.is_finite() && values.iter().all(|v| v.is_finite()));
        let xs = Value::vector_f64(&values).expect("vector");
        let init_val = vec![Value::scalar_f64(init)];
        let scan_result = eval_scan_functional(
            init_val,
            &xs,
            |carry: Vec<Value>, x: Value| {
                let c = carry[0].as_f64_scalar().expect("carry f64");
                let xv = x.as_f64_scalar().expect("x f64");
                let next = Value::scalar_f64(c * xv);
                Ok((vec![next.clone()], vec![next]))
            },
            false,
        );
        if let Ok((final_carry, _outputs)) = scan_result {
            let mut loop_acc = init;
            for &x in &values {
                loop_acc *= x;
            }
            let scan_val = final_carry[0].as_f64_scalar().expect("scan scalar");
            prop_assert!(
                approx_eq_f64(scan_val, loop_acc, 1e-6),
                "scan(mul) != loop: {} vs {} with init={}, xs={:?}",
                scan_val,
                loop_acc,
                init,
                values
            );
        }
    }

    // =========================================================================
    // RELATION 6: grad commutes with jit
    // =========================================================================

    #[test]
    fn metamorphic_grad_commutes_with_jit_square(x in -100.0f64..100.0) {
        prop_assume!(x.is_finite());
        let jaxpr = square_jaxpr();
        let jit_grad_result = jit(jaxpr.clone())
            .compose_grad()
            .call(vec![Value::scalar_f64(x)])
            .expect("jit(grad) should succeed");
        let grad_jit_result = grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(jit) should succeed");
        let jit_grad = jit_grad_result[0].as_f64_scalar().expect("jit_grad scalar");
        let grad_jit = grad_jit_result[0].as_f64_scalar().expect("grad_jit scalar");
        prop_assert!(
            approx_eq_f64(jit_grad, grad_jit, 1e-10),
            "jit(grad(f)) != grad(jit(f)) at x={}: {} vs {}",
            x,
            jit_grad,
            grad_jit
        );
    }

    #[test]
    fn metamorphic_grad_commutes_with_jit_sin(x in -10.0f64..10.0) {
        prop_assume!(x.is_finite());
        let jaxpr = sin_jaxpr();
        let jit_grad_result = jit(jaxpr.clone())
            .compose_grad()
            .call(vec![Value::scalar_f64(x)])
            .expect("jit(grad(sin)) should succeed");
        let grad_jit_result = grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(jit(sin)) should succeed");
        let jit_grad = jit_grad_result[0].as_f64_scalar().expect("jit_grad scalar");
        let grad_jit = grad_jit_result[0].as_f64_scalar().expect("grad_jit scalar");
        prop_assert!(
            approx_eq_f64(jit_grad, grad_jit, 1e-10),
            "jit(grad(sin)) != grad(jit(sin)) at x={}: {} vs {}",
            x,
            jit_grad,
            grad_jit
        );
    }

    #[test]
    fn metamorphic_grad_commutes_with_jit_exp(x in -10.0f64..10.0) {
        prop_assume!(x.is_finite());
        let jaxpr = exp_jaxpr();
        let jit_grad_result = jit(jaxpr.clone())
            .compose_grad()
            .call(vec![Value::scalar_f64(x)])
            .expect("jit(grad(exp)) should succeed");
        let grad_jit_result = grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(jit(exp)) should succeed");
        let jit_grad = jit_grad_result[0].as_f64_scalar().expect("jit_grad scalar");
        let grad_jit = grad_jit_result[0].as_f64_scalar().expect("grad_jit scalar");
        prop_assert!(
            approx_eq_f64(jit_grad, grad_jit, 1e-8),
            "jit(grad(exp)) != grad(jit(exp)) at x={}: {} vs {}",
            x,
            jit_grad,
            grad_jit
        );
    }

    // =========================================================================
    // ADDITIONAL RELATIONS: value_and_grad consistency
    // =========================================================================

    #[test]
    fn metamorphic_value_and_grad_consistent_with_grad(x in -100.0f64..100.0) {
        prop_assume!(x.is_finite());
        let jaxpr = square_jaxpr();
        let (value, gradient) = value_and_grad(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("value_and_grad should succeed");
        let grad_only = grad(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("grad should succeed");
        let direct_value = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
            .expect("eval should succeed");
        let vg_val = value[0].as_f64_scalar().expect("value scalar");
        let direct_val = direct_value[0].as_f64_scalar().expect("direct scalar");
        prop_assert!(
            approx_eq_f64(vg_val, direct_val, 1e-10),
            "value_and_grad value != f(x) at x={}: {} vs {}",
            x,
            vg_val,
            direct_val
        );
        let vg_grad = gradient[0].as_f64_scalar().expect("gradient scalar");
        let grad_val = grad_only[0].as_f64_scalar().expect("grad scalar");
        prop_assert!(
            approx_eq_f64(vg_grad, grad_val, 1e-10),
            "value_and_grad gradient != grad(f)(x) at x={}: {} vs {}",
            x,
            vg_grad,
            grad_val
        );
    }

    #[test]
    fn metamorphic_value_and_grad_consistent_with_grad_sin(x in -10.0f64..10.0) {
        prop_assume!(x.is_finite());
        let jaxpr = sin_jaxpr();
        let (value, gradient) = value_and_grad(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("value_and_grad should succeed");
        let grad_only = grad(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("grad should succeed");
        let direct_value = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
            .expect("eval should succeed");
        let vg_val = value[0].as_f64_scalar().expect("value scalar");
        let direct_val = direct_value[0].as_f64_scalar().expect("direct scalar");
        prop_assert!(
            approx_eq_f64(vg_val, direct_val, 1e-10),
            "value_and_grad value != sin(x) at x={}: {} vs {}",
            x,
            vg_val,
            direct_val
        );
        let vg_grad = gradient[0].as_f64_scalar().expect("gradient scalar");
        let grad_val = grad_only[0].as_f64_scalar().expect("grad scalar");
        prop_assert!(
            approx_eq_f64(vg_grad, grad_val, 1e-10),
            "value_and_grad gradient != grad(sin)(x) at x={}: {} vs {}",
            x,
            vg_grad,
            grad_val
        );
    }

    // =========================================================================
    // RELATION: grad(x^2) = 2x (exact mathematical property)
    // =========================================================================

    #[test]
    fn metamorphic_grad_square_is_2x(x in -100.0f64..100.0) {
        prop_assume!(x.is_finite());
        let jaxpr = square_jaxpr();
        let grad_result = grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(square) should succeed");
        let grad_val = grad_result[0].as_f64_scalar().expect("gradient scalar");
        let expected = 2.0 * x;
        prop_assert!(
            approx_eq_f64(grad_val, expected, 1e-10),
            "grad(x^2) != 2x at x={}: {} vs {}",
            x,
            grad_val,
            expected
        );
    }

    #[test]
    fn metamorphic_grad_sin_is_cos(x in -10.0f64..10.0) {
        prop_assume!(x.is_finite());
        let jaxpr = sin_jaxpr();
        let grad_result = grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(sin) should succeed");
        let grad_val = grad_result[0].as_f64_scalar().expect("gradient scalar");
        let expected = x.cos();
        prop_assert!(
            approx_eq_f64(grad_val, expected, 1e-10),
            "grad(sin) != cos at x={}: {} vs {}",
            x,
            grad_val,
            expected
        );
    }

    #[test]
    fn metamorphic_grad_exp_is_exp(x in -10.0f64..10.0) {
        prop_assume!(x.is_finite());
        let jaxpr = exp_jaxpr();
        let grad_result = grad(jaxpr)
            .call(vec![Value::scalar_f64(x)])
            .expect("grad(exp) should succeed");
        let grad_val = grad_result[0].as_f64_scalar().expect("gradient scalar");
        let expected = x.exp();
        prop_assert!(
            approx_eq_f64(grad_val, expected, 1e-8),
            "grad(exp) != exp at x={}: {} vs {}",
            x,
            grad_val,
            expected
        );
    }

    // =========================================================================
    // RELATION: vmap + grad composition
    // =========================================================================

    #[test]
    fn metamorphic_vmap_grad_equals_map_grad(
        values in prop::collection::vec(-10.0f64..10.0, 2..5)
    ) {
        prop_assume!(values.iter().all(|v| v.is_finite()));
        let jaxpr = square_jaxpr();
        let vec_val = Value::vector_f64(&values).expect("vector");
        let vmap_grad_result = vmap(jaxpr.clone())
            .compose_grad()
            .call(vec![vec_val])
            .expect("vmap(grad(square)) should succeed");
        let vmap_tensor = vmap_grad_result[0].as_tensor().expect("vmap tensor");
        let vmap_vals = vmap_tensor.to_f64_vec().expect("f64 values");
        for (i, &x) in values.iter().enumerate() {
            let direct_grad = grad(jaxpr.clone())
                .call(vec![Value::scalar_f64(x)])
                .expect("grad(square)")[0]
                .as_f64_scalar()
                .expect("scalar");
            prop_assert!(
                approx_eq_f64(vmap_vals[i], direct_grad, 1e-10),
                "vmap(grad(square))[{}] != grad(square)(x) at x={}: {} vs {}",
                i,
                x,
                vmap_vals[i],
                direct_grad
            );
        }
    }

    #[test]
    fn metamorphic_jit_vmap_equals_vmap(
        values in prop::collection::vec(-100.0f64..100.0, 2..10)
    ) {
        prop_assume!(values.iter().all(|v| v.is_finite()));
        let jaxpr = square_jaxpr();
        let vec_val = Value::vector_f64(&values).expect("vector");
        let jit_vmap_result = jit(jaxpr.clone())
            .compose_vmap()
            .call(vec![vec_val.clone()])
            .expect("jit(vmap(square)) should succeed");
        let vmap_result = vmap(jaxpr)
            .call(vec![vec_val])
            .expect("vmap(square) should succeed");
        prop_assert!(
            values_approx_eq(&jit_vmap_result, &vmap_result, 1e-10),
            "jit(vmap(f)) != vmap(f) for values={:?}",
            values
        );
    }
}

#[cfg(test)]
mod deterministic_tests {
    use super::*;

    #[test]
    fn grad_sum_vector_is_ones() {
        let jaxpr = reduce_sum_jaxpr();
        let vec_val = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("vector");
        let result = grad(jaxpr).call(vec![vec_val]).expect("grad(sum)");
        let grad_tensor = result[0].as_tensor().expect("tensor");
        let grad_vals = grad_tensor.to_f64_vec().expect("f64");
        assert_eq!(grad_vals.len(), 5);
        for (i, g) in grad_vals.iter().enumerate() {
            assert!(
                approx_eq_f64(*g, 1.0, 1e-10),
                "grad(sum)[{}] = {}, expected 1.0",
                i,
                g
            );
        }
    }

    #[test]
    fn jit_is_identity_for_square() {
        let jaxpr = square_jaxpr();
        let x = 7.0;
        let jit_result = jit(jaxpr.clone())
            .call(vec![Value::scalar_f64(x)])
            .expect("jit");
        let direct_result = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)]).expect("direct");
        assert!(values_approx_eq(&jit_result, &direct_result, 1e-10));
    }

    #[test]
    fn vmap_equals_manual_map() {
        let jaxpr = square_jaxpr();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let vec_val = Value::vector_f64(&values).expect("vector");
        let vmap_result = vmap(jaxpr.clone()).call(vec![vec_val]).expect("vmap");
        let vmap_tensor = vmap_result[0].as_tensor().expect("tensor");
        let vmap_vals = vmap_tensor.to_f64_vec().expect("f64");
        for (i, &x) in values.iter().enumerate() {
            let direct = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)]).expect("eval")[0]
                .as_f64_scalar()
                .expect("scalar");
            assert!(
                approx_eq_f64(vmap_vals[i], direct, 1e-10),
                "vmap[{}] = {}, expected {}",
                i,
                vmap_vals[i],
                direct
            );
        }
    }

    #[test]
    fn grad_commutes_with_jit() {
        let jaxpr = square_jaxpr();
        let x = 5.0;
        let jit_grad = jit(jaxpr.clone())
            .compose_grad()
            .call(vec![Value::scalar_f64(x)])
            .expect("jit_grad")[0]
            .as_f64_scalar()
            .expect("scalar");
        let grad_only = grad(jaxpr).call(vec![Value::scalar_f64(x)]).expect("grad")[0]
            .as_f64_scalar()
            .expect("scalar");
        assert!(
            approx_eq_f64(jit_grad, grad_only, 1e-10),
            "jit_grad={}, grad={}",
            jit_grad,
            grad_only
        );
    }

    #[test]
    fn scan_add_equals_loop() {
        let init = 0.0;
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let xs = Value::vector_f64(&values).expect("vector");
        let init_val = vec![Value::scalar_f64(init)];
        let (final_carry, _outputs) = eval_scan_functional(
            init_val,
            &xs,
            |carry: Vec<Value>, x: Value| {
                let c = carry[0].as_f64_scalar().expect("carry f64");
                let xv = x.as_f64_scalar().expect("x f64");
                let next = Value::scalar_f64(c + xv);
                Ok((vec![next.clone()], vec![next]))
            },
            false,
        )
        .expect("scan");
        let scan_val = final_carry[0].as_f64_scalar().expect("scalar");
        let expected: f64 = values.iter().sum();
        assert!(
            approx_eq_f64(scan_val, expected, 1e-10),
            "scan={}, loop={}",
            scan_val,
            expected
        );
    }

    #[test]
    fn double_transpose_is_identity_for_neg() {
        let jaxpr = neg_jaxpr();
        let x = 3.0;
        let primals = vec![Value::scalar_f64(x)];
        let t1 = linear_transpose(jaxpr.clone(), primals.clone()).expect("t1");
        let r1 = t1.call(Value::scalar_f64(1.0)).expect("r1");
        let t2 = linear_transpose(jaxpr, r1.clone()).expect("t2");
        let r2 = t2.call(Value::scalar_f64(1.0)).expect("r2");
        let v1 = r1[0].as_f64_scalar().expect("s1");
        let v2 = r2[0].as_f64_scalar().expect("s2");
        assert!(approx_eq_f64(v1, v2, 1e-10), "v1={}, v2={}", v1, v2);
    }
}
