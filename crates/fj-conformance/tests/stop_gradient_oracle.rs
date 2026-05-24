//! StopGradient conformance tests.
//!
//! JAX treats `stop_gradient(x)` as an identity in primal evaluation and as a
//! gradient barrier for both reverse- and forward-mode AD.

use fj_core::{Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId};
use fj_lax::eval_primitive;
use smallvec::smallvec;
use std::collections::BTreeMap;

fn make_f64_vector(values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape::vector(values.len() as u32),
            values.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(value: &Value) -> Vec<f64> {
    value
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|literal| literal.as_f64().expect("expected f64 literal"))
        .collect()
}

fn stop_gradient_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::StopGradient,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

#[test]
fn stop_gradient_preserves_scalar_value() {
    let input = Value::scalar_f64(-3.25);
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient scalar eval should succeed");

    assert_eq!(output.as_f64_scalar(), Some(-3.25));
    assert_eq!(output.dtype(), DType::F64);
}

#[test]
fn stop_gradient_preserves_tensor_value_shape_and_dtype() {
    let input = make_f64_vector(&[-2.0, 0.5, 7.0]);
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient tensor eval should succeed");

    assert_eq!(extract_f64_vec(&output), vec![-2.0, 0.5, 7.0]);
    let tensor = output.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape, Shape::vector(3));
}

#[test]
fn stop_gradient_vjp_returns_zero_cotangent_for_scalar() {
    let input = Value::scalar_f64(4.0);
    let output = Value::scalar_f64(4.0);
    let incoming_cotangent = Value::scalar_f64(9.0);

    let cotangents = fj_ad::vjp(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        std::slice::from_ref(&incoming_cotangent),
        std::slice::from_ref(&output),
        &BTreeMap::new(),
    )
    .expect("stop_gradient VJP should succeed");

    assert_eq!(cotangents.len(), 1);
    assert_eq!(cotangents[0].as_f64_scalar(), Some(0.0));
}

#[test]
fn stop_gradient_vjp_returns_zero_cotangent_for_tensor() {
    let input = make_f64_vector(&[1.0, -2.0, 3.5]);
    let output = input.clone();
    let incoming_cotangent = make_f64_vector(&[5.0, 6.0, 7.0]);

    let cotangents = fj_ad::vjp(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        std::slice::from_ref(&incoming_cotangent),
        std::slice::from_ref(&output),
        &BTreeMap::new(),
    )
    .expect("stop_gradient tensor VJP should succeed");

    assert_eq!(cotangents.len(), 1);
    assert_eq!(extract_f64_vec(&cotangents[0]), vec![0.0, 0.0, 0.0]);
    let tensor = cotangents[0]
        .as_tensor()
        .expect("expected tensor cotangent");
    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape, Shape::vector(3));
}

#[test]
fn stop_gradient_jvp_preserves_primal_and_blocks_tangent() {
    let jaxpr = stop_gradient_jaxpr();
    let primal = make_f64_vector(&[2.0, -1.0, 0.25]);
    let tangent = make_f64_vector(&[10.0, 20.0, 30.0]);

    let result = fj_ad::jvp(&jaxpr, &[primal], &[tangent]).expect("stop_gradient JVP should pass");

    assert_eq!(result.primals.len(), 1);
    assert_eq!(result.tangents.len(), 1);
    assert_eq!(extract_f64_vec(&result.primals[0]), vec![2.0, -1.0, 0.25]);
    assert_eq!(extract_f64_vec(&result.tangents[0]), vec![0.0, 0.0, 0.0]);
}

// ======================== Additional Coverage ========================

#[test]
fn stop_gradient_preserves_integer_dtype() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape::vector(3),
            vec![Literal::I64(10), Literal::I64(-20), Literal::I64(30)],
        )
        .unwrap(),
    );
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient i64 eval should succeed");

    let tensor = output.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.dtype, DType::I64);
    let values: Vec<i64> = tensor
        .elements
        .iter()
        .map(|l| match l {
            Literal::I64(v) => *v,
            _ => panic!("expected i64"),
        })
        .collect();
    assert_eq!(values, vec![10, -20, 30]);
}

#[test]
fn stop_gradient_preserves_rank2_tensor() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 3] },
            (1..=6).map(|x| Literal::from_f64(x as f64)).collect(),
        )
        .unwrap(),
    );
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient rank-2 eval should succeed");

    let tensor = output.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.shape.dims, vec![2, 3]);
    assert_eq!(extract_f64_vec(&output), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn stop_gradient_vjp_rank2_zero_cotangent() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::from_f64(1.0),
                Literal::from_f64(2.0),
                Literal::from_f64(3.0),
                Literal::from_f64(4.0),
            ],
        )
        .unwrap(),
    );
    let output = input.clone();
    let incoming_cotangent = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::from_f64(10.0),
                Literal::from_f64(20.0),
                Literal::from_f64(30.0),
                Literal::from_f64(40.0),
            ],
        )
        .unwrap(),
    );

    let cotangents = fj_ad::vjp(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        std::slice::from_ref(&incoming_cotangent),
        std::slice::from_ref(&output),
        &BTreeMap::new(),
    )
    .expect("stop_gradient rank-2 VJP should succeed");

    assert_eq!(cotangents.len(), 1);
    let tensor = cotangents[0].as_tensor().expect("expected tensor");
    assert_eq!(tensor.shape.dims, vec![2, 2]);
    assert_eq!(extract_f64_vec(&cotangents[0]), vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn stop_gradient_preserves_special_values() {
    let input = make_f64_vector(&[f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0, -0.0]);
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient special values should succeed");

    let values = extract_f64_vec(&output);
    assert!(values[0].is_infinite() && values[0].is_sign_positive());
    assert!(values[1].is_infinite() && values[1].is_sign_negative());
    assert!(values[2].is_nan());
    assert_eq!(values[3], 0.0);
    assert_eq!(values[4], 0.0);
}

#[test]
fn stop_gradient_empty_tensor() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap());
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient empty tensor should succeed");

    let tensor = output.as_tensor().expect("expected tensor");
    assert_eq!(tensor.shape.dims, vec![0]);
    assert!(tensor.elements.is_empty());
}

#[test]
fn stop_gradient_jvp_scalar() {
    let scalar_jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::StopGradient,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let primal = Value::scalar_f64(5.0);
    let tangent = Value::scalar_f64(100.0);

    let result =
        fj_ad::jvp(&scalar_jaxpr, &[primal], &[tangent]).expect("stop_gradient scalar JVP");

    assert_eq!(result.primals[0].as_f64_scalar(), Some(5.0));
    assert_eq!(result.tangents[0].as_f64_scalar(), Some(0.0));
}

// ======================== Additional Coverage ========================

#[test]
fn stop_gradient_preserves_3d_shape() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![2, 2, 2],
            },
            (1..=8).map(|x| Literal::from_f64(x as f64)).collect(),
        )
        .unwrap(),
    );
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient 3D eval should succeed");

    let tensor = output.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.shape.dims, vec![2, 2, 2]);
    assert_eq!(
        extract_f64_vec(&output),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn stop_gradient_preserves_bool_dtype() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape::vector(3),
            vec![
                Literal::Bool(true),
                Literal::Bool(false),
                Literal::Bool(true),
            ],
        )
        .unwrap(),
    );
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient bool eval should succeed");

    let tensor = output.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.dtype, DType::Bool);
}

#[test]
fn stop_gradient_large_tensor() {
    let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![10, 10] },
            data.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    );
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient large tensor should succeed");

    let tensor = output.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.shape.dims, vec![10, 10]);
    let values = extract_f64_vec(&output);
    assert_eq!(values.len(), 100);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[99], 99.0);
}

#[test]
fn stop_gradient_complex_dtype() {
    let input = Value::scalar_complex128(3.0, 4.0);
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient complex should succeed");

    assert_eq!(output.dtype(), DType::Complex128);
    assert_eq!(output.as_complex128_scalar(), Some((3.0, 4.0)));
}

#[test]
fn stop_gradient_4d_shape() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![2, 2, 2, 2],
            },
            (1..=16).map(|x| Literal::from_f64(x as f64)).collect(),
        )
        .unwrap(),
    );
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient 4D eval should succeed");

    let tensor = output.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.shape.dims, vec![2, 2, 2, 2]);
}

#[test]
fn stop_gradient_2d_empty() {
    let input =
        Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0, 3] }, vec![]).unwrap());
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient 2D empty should succeed");

    let tensor = output.as_tensor().expect("expected tensor");
    assert_eq!(tensor.shape.dims, vec![0, 3]);
}

#[test]
fn stop_gradient_subnormal_values() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let input = make_f64_vector(&[subnormal, -subnormal, 0.0]);
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient subnormal should succeed");

    let values = extract_f64_vec(&output);
    assert_eq!(values[0].to_bits(), subnormal.to_bits());
    assert_eq!(values[1].to_bits(), (-subnormal).to_bits());
}

#[test]
fn stop_gradient_u32_dtype() {
    let input = Value::Tensor(
        TensorValue::new(
            DType::U32,
            Shape::vector(3),
            vec![Literal::U32(10), Literal::U32(20), Literal::U32(30)],
        )
        .unwrap(),
    );
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient u32 eval should succeed");

    let tensor = output.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.dtype, DType::U32);
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_stop_gradient_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::BF16 => Literal::from_bf16_f32(v as f32),
                DType::F16 => Literal::from_f16_f32(v as f32),
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not a float dtype"),
            })
            .collect();
        Value::Tensor(TensorValue::new(dtype, Shape::vector(3), lits).unwrap())
    }

    let values = [1.0_f64, 2.0, 3.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::StopGradient, &[input], &BTreeMap::new()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "stop_gradient {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}
