# Security Adversarial Gate

- Schema: `frankenjax.security-adversarial-gate.v1`
- Bead: `frankenjax-cstq.17`
- Status: `pass`
- Categories: `9`
- Fuzz families: `19`
- Adversarial rows: `10`

| Category | Evidence | Fuzz | Required Families |
|---|---:|---:|---|
| `tc_cache_confusion` | `green` | `complete` | `ff_cache_key_builder` |
| `tc_transform_ordering` | `green` | `complete` | `ff_transform_composition_verifier` |
| `tc_ir_validation` | `green` | `complete` | `ff_ir_deserializer` |
| `tc_shape_dtype_signatures` | `green` | `complete` | `ff_shape_inference_engine`, `ff_value_deserializer` |
| `tc_subjaxpr_control_flow` | `green` | `complete` | `ff_shape_inference_engine`, `ff_dispatch_request_builder` |
| `tc_serialized_fixtures_logs` | `green` | `complete` | `ff_fixture_bundle_loader`, `ff_smoke_harness_json` |
| `tc_ffi_boundaries` | `green` | `complete` | `ff_value_deserializer` |
| `tc_durability_corruption` | `green` | `complete` | `ff_raptorq_decoder` |
| `tc_evidence_ledger_recovery` | `green` | `complete` | `ff_dispatch_request_builder`, `ff_ledger_deserializer` |

| Fuzz Family | Seeds | Replay | Status |
|---|---:|---|---|
| `ff_cache_key_builder` | `3` | `cd crates/fj-conformance/fuzz && cargo fuzz run cache_key_builder corpus/cache_key_builder -runs=3` | `no_panic/no_crash/no_timeout` |
| `ff_transform_composition_verifier` | `15` | `cd crates/fj-conformance/fuzz && cargo fuzz run transform_composition_verifier corpus/transform_composition_verifier -runs=15` | `no_panic/no_crash/no_timeout` |
| `ff_ir_deserializer` | `3` | `cd crates/fj-conformance/fuzz && cargo fuzz run ir_deserializer corpus/seed/ir_deserializer -runs=3` | `no_panic/no_crash/no_timeout` |
| `ff_shape_inference_engine` | `67` | `cd crates/fj-conformance/fuzz && cargo fuzz run shape_inference_engine corpus/shape_inference_engine -runs=67` | `no_panic/no_crash/no_timeout` |
| `ff_value_deserializer` | `37` | `cd crates/fj-conformance/fuzz && cargo fuzz run value_deserializer corpus/value_deserializer -runs=37` | `no_panic/no_crash/no_timeout` |
| `ff_dispatch_request_builder` | `3` | `cd crates/fj-conformance/fuzz && cargo fuzz run dispatch_request_builder corpus/seed/dispatch_request_builder -runs=3` | `no_panic/no_crash/no_timeout` |
| `ff_fixture_bundle_loader` | `16` | `cd crates/fj-conformance/fuzz && cargo fuzz run fixture_bundle_loader corpus/fixture_bundle_loader -runs=16` | `no_panic/no_crash/no_timeout` |
| `ff_smoke_harness_json` | `24` | `cd crates/fj-conformance/fuzz && cargo fuzz run smoke_harness_json corpus/smoke_harness_json -runs=24` | `no_panic/no_crash/no_timeout` |
| `ff_cache_persistence_format` | `13` | `cd crates/fj-conformance/fuzz && cargo fuzz run cache_persistence_format corpus/seed/fuzz_target_1 -runs=13` | `no_panic/no_crash/no_timeout` |
| `ff_primitive_eval_fuzzer` | `101` | `cd crates/fj-conformance/fuzz && cargo fuzz run primitive_eval_fuzzer corpus/primitive_eval_fuzzer -runs=101` | `no_panic/no_crash/no_timeout` |
| `ff_partial_eval_fuzzer` | `20` | `cd crates/fj-conformance/fuzz && cargo fuzz run partial_eval_fuzzer corpus/partial_eval_fuzzer -runs=20` | `no_panic/no_crash/no_timeout` |
| `ff_egraph_rewrite_rules` | `1` | `cd crates/fj-conformance/fuzz && cargo fuzz run egraph_rewrite_rules corpus/seed/egraph_rewrite_rules -runs=1` | `no_panic/no_crash/no_timeout` |
| `ff_ad_gradient_computation` | `1` | `cd crates/fj-conformance/fuzz && cargo fuzz run ad_gradient_computation corpus/seed/ad_gradient_computation -runs=1` | `no_panic/no_crash/no_timeout` |
| `ff_primitive_params_fuzzer` | `3` | `cd crates/fj-conformance/fuzz && cargo fuzz run primitive_params_fuzzer corpus/primitive_params_fuzzer -runs=3` | `no_panic/no_crash/no_timeout` |
| `ff_proc_status_parser` | `3` | `cd crates/fj-conformance/fuzz && cargo fuzz run proc_status_parser corpus/seed/proc_status_parser -runs=3` | `no_panic/no_crash/no_timeout` |
| `ff_jaxpr_serde_roundtrip` | `2` | `cd crates/fj-conformance/fuzz && cargo fuzz run jaxpr_serde_roundtrip corpus/seed/jaxpr_serde_roundtrip -runs=2` | `no_panic/no_crash/no_timeout` |
| `ff_fft_roundtrip` | `1` | `cd crates/fj-conformance/fuzz && cargo fuzz run fft_roundtrip corpus/seed/fft_roundtrip -runs=1` | `no_panic/no_crash/no_timeout` |
| `ff_raptorq_decoder` | `15` | `cd crates/fj-conformance/fuzz && cargo fuzz run raptorq_decoder corpus/seed/raptorq_decoder -runs=15` | `no_panic/no_crash/no_timeout` |
| `ff_ledger_deserializer` | `3` | `cd crates/fj-conformance/fuzz && cargo fuzz run ledger_deserializer corpus/seed/ledger_deserializer -runs=3` | `no_panic/no_crash/no_timeout` |

No security adversarial gate issues found.
