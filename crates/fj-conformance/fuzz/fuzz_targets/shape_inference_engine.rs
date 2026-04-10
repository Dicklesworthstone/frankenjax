#![no_main]

mod common;

use common::{ByteCursor, primitive_arity, sample_dtype, sample_primitive, sample_primitive_params};
use fj_lax::eval_primitive;
use fj_trace::{ShapedArray, SimpleTraceContext, TraceContext};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut cursor = ByteCursor::new(data);
    let primitive = sample_primitive(&mut cursor);

    let runtime_arity = primitive_arity(primitive);
    let mut runtime_inputs = Vec::with_capacity(runtime_arity);
    for _ in 0..runtime_arity {
        runtime_inputs.push(common::sample_value(&mut cursor));
    }
    let params = sample_primitive_params(&mut cursor, primitive);
    let _ = eval_primitive(primitive, &runtime_inputs, &params);

    let trace_arity = if cursor.take_bool() {
        runtime_arity
    } else {
        cursor.take_usize(3)
    };
    let mut trace_ctx = SimpleTraceContext::new();
    let mut input_ids = Vec::with_capacity(trace_arity);
    for _ in 0..trace_arity {
        let aval = ShapedArray {
            dtype: sample_dtype(&mut cursor),
            shape: common::sample_shape(&mut cursor, 4, 4),
        };
        let Ok(input_id) = trace_ctx.bind_input(aval) else {
            return;
        };
        input_ids.push(input_id);
    }

    let params = sample_primitive_params(&mut cursor, primitive);
    let _ = trace_ctx.process_primitive(primitive, &input_ids, params);
    let _ = trace_ctx.finalize();
});
