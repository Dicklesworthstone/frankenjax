# Decision Ledger Calibration

- Schema: `frankenjax.decision-ledger-calibration.v1`
- Bead: `frankenjax-cstq.19`
- Status: `pass`
- Rows: `9`
- Calibration buckets: `4`

| Decision | Mode | Action | Confidence | Drift | Dashboard |
|---|---|---:|---:|---|---|
| `cache_hit_recompute` | `strict` | `keep` | `0.848` | `green` | `ledger/cache_hit_recompute` |
| `strict_rejection` | `strict` | `kill` | `0.857` | `green` | `ledger/strict_rejection` |
| `hardened_recovery` | `hardened` | `fallback` | `1.000` | `green` | `ledger/hardened_recovery` |
| `fallback_denial` | `strict` | `kill` | `0.860` | `green` | `ledger/fallback_denial` |
| `optimization_selection` | `strict` | `keep` | `0.899` | `green` | `ledger/optimization_selection` |
| `durability_recovery` | `hardened` | `fallback` | `0.698` | `green` | `ledger/durability_recovery` |
| `transform_admission` | `strict` | `keep` | `0.881` | `green` | `ledger/transform_admission` |
| `unsupported_scope` | `strict` | `kill` | `0.875` | `green` | `ledger/unsupported_scope` |
| `runtime_budget_deadline` | `strict` | `reprofile` | `1.000` | `green` | `ledger/runtime_budget_deadline` |

No decision-ledger calibration issues found.
