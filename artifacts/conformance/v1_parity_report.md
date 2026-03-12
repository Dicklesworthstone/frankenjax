# V1 Parity Report

Mode: `strict`

FrankenJAX: `0.1.0` | Oracle: `jax-0.9.0.1`

## Summary

| Metric | Value |
|---|---|
| Total Cases | 596 |
| Matched | 596 |
| Mismatched | 0 |
| Pass Rate | 100.00% |
| Gate | **pass** |

## Per-Family Breakdown

| Family | Total | Matched | Mismatched |
|---|---|---|---|
| jit | 58 | 58 | 0 |
| grad | 106 | 106 | 0 |
| vmap | 30 | 30 | 0 |
| lax | 343 | 343 | 0 |
| random | 25 | 25 | 0 |
| control_flow | 11 | 11 | 0 |
| mixed_dtype | 23 | 23 | 0 |

## Per-Primitive Breakdown

| Primitive | Total | Matched | Mismatched |
|---|---|---|---|
| Add2 | 14 | 14 | 0 |
| AddOne | 12 | 12 | 0 |
| CondSelect | 6 | 6 | 0 |
| CosX | 13 | 13 | 0 |
| Dot3 | 4 | 4 | 0 |
| LaxAbs | 14 | 14 | 0 |
| LaxAcos | 8 | 8 | 0 |
| LaxArgsort | 2 | 2 | 0 |
| LaxAsin | 8 | 8 | 0 |
| LaxAtan | 10 | 10 | 0 |
| LaxAtan2 | 7 | 7 | 0 |
| LaxBitcastF64ToI64 | 2 | 2 | 0 |
| LaxBitwiseAnd | 4 | 4 | 0 |
| LaxBitwiseNot | 5 | 5 | 0 |
| LaxBitwiseOr | 4 | 4 | 0 |
| LaxBitwiseXor | 4 | 4 | 0 |
| LaxBroadcastInDimScalar3 | 2 | 2 | 0 |
| LaxBroadcastedIota2x3 | 1 | 1 | 0 |
| LaxCbrt | 6 | 6 | 0 |
| LaxCeil | 5 | 5 | 0 |
| LaxClamp | 4 | 4 | 0 |
| LaxConcatenate | 2 | 2 | 0 |
| LaxCopy | 2 | 2 | 0 |
| LaxCosh | 7 | 7 | 0 |
| LaxCountLeadingZeros | 6 | 6 | 0 |
| LaxCumprod | 3 | 3 | 0 |
| LaxCumsum | 3 | 3 | 0 |
| LaxDigamma | 6 | 6 | 0 |
| LaxDiv | 9 | 9 | 0 |
| LaxDynamicSlice | 2 | 2 | 0 |
| LaxDynamicUpdateSlice | 2 | 2 | 0 |
| LaxEq | 6 | 6 | 0 |
| LaxErf | 5 | 5 | 0 |
| LaxErfInv | 5 | 5 | 0 |
| LaxErfc | 5 | 5 | 0 |
| LaxExp | 22 | 22 | 0 |
| LaxExpandDimsAxis0 | 2 | 2 | 0 |
| LaxExpm1 | 9 | 9 | 0 |
| LaxFloor | 5 | 5 | 0 |
| LaxGather1d | 2 | 2 | 0 |
| LaxGe | 4 | 4 | 0 |
| LaxGt | 7 | 7 | 0 |
| LaxIntegerPow2 | 5 | 5 | 0 |
| LaxIntegerPow3 | 4 | 4 | 0 |
| LaxIntegerPowNeg1 | 4 | 4 | 0 |
| LaxIota5 | 1 | 1 | 0 |
| LaxIsFinite | 6 | 6 | 0 |
| LaxLe | 4 | 4 | 0 |
| LaxLgamma | 6 | 6 | 0 |
| LaxLog | 9 | 9 | 0 |
| LaxLog1p | 9 | 9 | 0 |
| LaxLogistic | 9 | 9 | 0 |
| LaxLt | 6 | 6 | 0 |
| LaxMax | 7 | 7 | 0 |
| LaxMin | 6 | 6 | 0 |
| LaxMul | 12 | 12 | 0 |
| LaxNe | 5 | 5 | 0 |
| LaxNeg | 13 | 13 | 0 |
| LaxNextafter | 4 | 4 | 0 |
| LaxOneHot4 | 2 | 2 | 0 |
| LaxPadLow1High2 | 2 | 2 | 0 |
| LaxPopulationCount | 6 | 6 | 0 |
| LaxPow | 6 | 6 | 0 |
| LaxReciprocal | 8 | 8 | 0 |
| LaxReduceMax | 3 | 3 | 0 |
| LaxReduceMin | 3 | 3 | 0 |
| LaxReducePrecisionF64 | 2 | 2 | 0 |
| LaxReduceProd | 3 | 3 | 0 |
| LaxReduceWindowSum | 2 | 2 | 0 |
| LaxReduceXor | 4 | 4 | 0 |
| LaxRem | 4 | 4 | 0 |
| LaxReshape6To2x3 | 2 | 2 | 0 |
| LaxReshape6To3x2 | 1 | 1 | 0 |
| LaxRev | 2 | 2 | 0 |
| LaxRound | 5 | 5 | 0 |
| LaxRsqrt | 8 | 8 | 0 |
| LaxSelect | 4 | 4 | 0 |
| LaxShiftLeft | 2 | 2 | 0 |
| LaxShiftRightArithmetic | 2 | 2 | 0 |
| LaxShiftRightLogical | 2 | 2 | 0 |
| LaxSign | 5 | 5 | 0 |
| LaxSinh | 7 | 7 | 0 |
| LaxSlice1To4 | 2 | 2 | 0 |
| LaxSort | 4 | 4 | 0 |
| LaxSplit2 | 1 | 1 | 0 |
| LaxSqrt | 9 | 9 | 0 |
| LaxSquare | 14 | 14 | 0 |
| LaxSqueeze | 2 | 2 | 0 |
| LaxSub | 10 | 10 | 0 |
| LaxSwitch3 | 2 | 2 | 0 |
| LaxTan | 9 | 9 | 0 |
| LaxTanh | 12 | 12 | 0 |
| LaxTranspose2x3 | 2 | 2 | 0 |
| LaxWhileAddLt | 2 | 2 | 0 |
| ReduceSumVec | 4 | 4 | 0 |
| ScanAdd | 5 | 5 | 0 |
| SinX | 20 | 20 | 0 |
| Square | 19 | 19 | 0 |
| SquarePlusLinear | 10 | 10 | 0 |

## Coverage Exceptions

None.

## Parity Exceptions

None.
