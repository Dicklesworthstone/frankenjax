# UBS Test-Only Classification Report

**Bead:** frankenjax-vzo  
**Date:** 2026-04-27  
**Status:** COMPLETED

## Summary

Classified all UBS panic/assert/unwrap findings from the FrankenJAX crates. **All findings are in test code - zero production panic surfaces identified.**

## Classification Results

| Category | Location | Count | Assessment |
|----------|----------|-------|------------|
| `panic!/unreachable!` | Test files (`/tests/`, `/benches/`) | 52 CRITICAL | Test-only, acceptable |
| `unreachable!` | Test files | 13 WARNING | Test-only, acceptable |
| `unwrap/expect` | Test modules (`#[cfg(test)]`) | 2,922 | Test-only, acceptable |
| `unwrap/expect` | Fuzz target (`fuzz_target_1.rs`) | 2 | Test infrastructure, acceptable |
| `unwrap/expect` | Test harness (`stability.rs::capture_golden_keys`) | 1 | Test infrastructure, acceptable |
| **Production panic surfaces** | — | **0** | **None found** |

## Methodology

1. Ran `ubs --only=rust crates/` to capture full inventory
2. Used ripgrep to filter findings by file path (`/tests/`, `/benches/`)
3. Used Python script to analyze `#[cfg(test)]` module boundaries within source files
4. Manually reviewed the 3 remaining "production" findings:
   - `fuzz_target_1.rs` lines 17, 38: Fuzz test infrastructure (not library code)
   - `stability.rs` line 84: Test harness function `capture_golden_keys()` for cache key stability testing

## Conclusion

The FrankenJAX crates have clean production code with no panic surfaces:
- All `panic!` macros are in test files
- All `unwrap()`/`expect()` calls in non-test files are either:
  - Inside `#[cfg(test)]` modules
  - In test infrastructure (fuzz targets, stability test harness)

**No remediation action required.** Future UBS runs on changed files can focus on verifying that new code maintains this clean pattern.

## Recommendations

1. UBS gates on this project should exclude `/tests/` and `/benches/` directories for panic checks
2. Consider adding a CI step that verifies zero production `panic!` via: `rg -l 'panic!' --type rust crates/ | grep -v '/tests/' | grep -v '/benches/'`
3. The README claim "No panics in library code. All error paths return `Result`" (line 919) is verified accurate
