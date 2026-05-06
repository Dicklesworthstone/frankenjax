#![no_main]

use arbitrary::Arbitrary;
use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeMap;

#[derive(Arbitrary, Debug, Clone)]
enum FuzzPrimitive {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Sin,
    Cos,
    Exp,
    Log,
    Sqrt,
    Floor,
    Ceil,
    Round,
    ReduceSum,
    ReduceMax,
    ReduceMin,
    Reshape,
    Transpose,
    Broadcast,
}

impl FuzzPrimitive {
    fn to_primitive(&self) -> Primitive {
        match self {
            Self::Add => Primitive::Add,
            Self::Sub => Primitive::Sub,
            Self::Mul => Primitive::Mul,
            Self::Div => Primitive::Div,
            Self::Neg => Primitive::Neg,
            Self::Abs => Primitive::Abs,
            Self::Sin => Primitive::Sin,
            Self::Cos => Primitive::Cos,
            Self::Exp => Primitive::Exp,
            Self::Log => Primitive::Log,
            Self::Sqrt => Primitive::Sqrt,
            Self::Floor => Primitive::Floor,
            Self::Ceil => Primitive::Ceil,
            Self::Round => Primitive::Round,
            Self::ReduceSum => Primitive::ReduceSum,
            Self::ReduceMax => Primitive::ReduceMax,
            Self::ReduceMin => Primitive::ReduceMin,
            Self::Reshape => Primitive::Reshape,
            Self::Transpose => Primitive::Transpose,
            Self::Broadcast => Primitive::BroadcastInDim,
        }
    }

    fn is_binary(&self) -> bool {
        matches!(self, Self::Add | Self::Sub | Self::Mul | Self::Div)
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    primitive: FuzzPrimitive,
    shape_dims: Vec<u8>,
    elements: Vec<u8>,
    param_keys: Vec<String>,
    param_values: Vec<String>,
}

fn build_tensor(shape_dims: &[u8], elements: &[u8]) -> Option<Value> {
    if shape_dims.is_empty() || shape_dims.len() > 4 {
        return None;
    }
    let dims: Vec<u32> = shape_dims
        .iter()
        .map(|&d| (d % 8).max(1) as u32)
        .collect();
    let total: usize = dims.iter().map(|&d| d as usize).product();
    if total == 0 || total > 512 {
        return None;
    }
    let literals: Vec<Literal> = elements
        .iter()
        .cycle()
        .take(total)
        .map(|&b| Literal::from_f64(b as f64))
        .collect();
    TensorValue::new(DType::F64, Shape { dims }, literals)
        .ok()
        .map(Value::Tensor)
}

fuzz_target!(|input: FuzzInput| {
    let Some(tensor) = build_tensor(&input.shape_dims, &input.elements) else {
        return;
    };

    let mut params = BTreeMap::new();
    for (k, v) in input.param_keys.iter().zip(input.param_values.iter()) {
        if k.len() < 32 && v.len() < 64 {
            params.insert(k.clone(), v.clone());
        }
    }

    let primitive = input.primitive.to_primitive();
    let inputs = if input.primitive.is_binary() {
        vec![tensor.clone(), tensor]
    } else {
        vec![tensor]
    };

    let _ = fj_lax::eval_primitive(primitive, &inputs, &params);
});
