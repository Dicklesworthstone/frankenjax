# V1 Parity Report

Mode: `strict`

FrankenJAX: `0.1.0` | Oracle: `jax-0.9.0.1`

## Summary

| Metric | Value |
|---|---|
| Total Cases | 242 |
| Matched | 242 |
| Mismatched | 0 |
| Pass Rate | 100.00% |
| Gate | **pass** |

## Per-Family Breakdown

| Family | Total | Matched | Mismatched |
|---|---|---|---|
| jit | 49 | 49 | 0 |
| grad | 19 | 19 | 0 |
| vmap | 9 | 9 | 0 |
| lax | 165 | 165 | 0 |
| random | 0 | 0 | 0 |

## Per-Primitive Breakdown

| Primitive | Total | Matched | Mismatched |
|---|---|---|---|
| Add2 | 8 | 8 | 0 |
| AddOne | 12 | 12 | 0 |
| CosX | 10 | 10 | 0 |
| Dot3 | 4 | 4 | 0 |
| LaxAbs | 6 | 6 | 0 |
| LaxAcos | 5 | 5 | 0 |
| LaxAsin | 5 | 5 | 0 |
| LaxAtan | 6 | 6 | 0 |
| LaxAtan2 | 4 | 4 | 0 |
| LaxCeil | 5 | 5 | 0 |
| LaxClamp | 4 | 4 | 0 |
| LaxCosh | 4 | 4 | 0 |
| LaxDiv | 4 | 4 | 0 |
| LaxErf | 5 | 5 | 0 |
| LaxErfc | 5 | 5 | 0 |
| LaxExp | 4 | 4 | 0 |
| LaxExpm1 | 5 | 5 | 0 |
| LaxFloor | 5 | 5 | 0 |
| LaxLog | 5 | 5 | 0 |
| LaxLog1p | 5 | 5 | 0 |
| LaxLogistic | 5 | 5 | 0 |
| LaxMax | 4 | 4 | 0 |
| LaxMin | 4 | 4 | 0 |
| LaxMul | 4 | 4 | 0 |
| LaxNeg | 6 | 6 | 0 |
| LaxPow | 3 | 3 | 0 |
| LaxReciprocal | 5 | 5 | 0 |
| LaxReduceMax | 3 | 3 | 0 |
| LaxReduceMin | 3 | 3 | 0 |
| LaxReduceProd | 3 | 3 | 0 |
| LaxRem | 4 | 4 | 0 |
| LaxRound | 5 | 5 | 0 |
| LaxRsqrt | 5 | 5 | 0 |
| LaxSign | 5 | 5 | 0 |
| LaxSinh | 4 | 4 | 0 |
| LaxSqrt | 5 | 5 | 0 |
| LaxSquare | 6 | 6 | 0 |
| LaxSub | 4 | 4 | 0 |
| LaxTan | 5 | 5 | 0 |
| LaxTanh | 5 | 5 | 0 |
| ReduceSumVec | 4 | 4 | 0 |
| SinX | 13 | 13 | 0 |
| Square | 16 | 16 | 0 |
| SquarePlusLinear | 10 | 10 | 0 |

## Coverage Exceptions

- `random`: no fixture cases captured for this family (tracked as known conformance gap pending fixture expansion)

## Parity Exceptions

None.
