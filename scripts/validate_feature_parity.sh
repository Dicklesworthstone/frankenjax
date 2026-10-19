#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
feature_file="${repo_root}/docs/planning/FEATURE_PARITY.md"
core_file="${repo_root}/crates/fj-core/src/lib.rs"

if [[ -n "${FJ_FEATURE_PARITY_OUT_DIR:-}" ]]; then
  ci_out="${FJ_FEATURE_PARITY_OUT_DIR}/ci/result.json"
  e2e_out="${FJ_FEATURE_PARITY_OUT_DIR}/e2e/e2e_feature_parity_validation.e2e.json"
  log_out="${FJ_FEATURE_PARITY_OUT_DIR}/logs/feature-parity-validation.result.json"
else
  ci_out="${repo_root}/artifacts/ci/runs/feature-parity-validation/result.json"
  e2e_out="${repo_root}/artifacts/e2e/e2e_feature_parity_validation.e2e.json"
  log_out="${repo_root}/artifacts/testing/logs/evidence/feature-parity-validation.result.json"
fi

mkdir -p "$(dirname "${ci_out}")" "$(dirname "${e2e_out}")" "$(dirname "${log_out}")"

python3 - "${feature_file}" "${core_file}" "${ci_out}" "${e2e_out}" "${log_out}" <<'PY'
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path


def parse_matrix(md_path: Path) -> list[dict[str, str]]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, str]] = []
    in_table = False
    for line in lines:
        if line.strip().startswith("| Feature Family | Status |"):
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        if stripped.startswith("|---"):
            continue
        cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
        if len(cells) != 4:
            continue
        rows.append(
            {
                "feature": cells[0],
                "status": cells[1],
                "current_evidence": cells[2],
                "next_required_artifact": cells[3],
            }
        )
    return rows


def count_primitives(core_path: Path) -> int:
    lines = core_path.read_text(encoding="utf-8").splitlines()
    in_enum = False
    count = 0
    for line in lines:
        if "pub enum Primitive {" in line:
            in_enum = True
            continue
        if in_enum and line.strip() == "}":
            break
        if not in_enum:
            continue
        code = line.split("//", 1)[0].strip()
        if not code:
            continue
        if code.startswith("#["):
            continue
        if code.endswith(","):
            count += 1
    return count


def row_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["feature"]: row for row in rows}


def check(
    checks: list[dict[str, object]],
    name: str,
    feature: str,
    claimed_status: str,
    actual_status: str,
    match: bool,
) -> None:
    checks.append(
        {
            "test_name": name,
            "feature": feature,
            "claimed_status": claimed_status,
            "actual_status": actual_status,
            "match": bool(match),
            "pass": bool(match),
        }
    )


feature_path = Path(sys.argv[1])
core_path = Path(sys.argv[2])
ci_out = Path(sys.argv[3])
e2e_out = Path(sys.argv[4])
log_out = Path(sys.argv[5])

rows = parse_matrix(feature_path)
lookup = row_lookup(rows)
checks: list[dict[str, object]] = []

valid_statuses = {"not_started", "in_progress", "parity_green", "parity_gap"}
invalid_status_rows = [row["feature"] for row in rows if row["status"] not in valid_statuses]
check(
    checks,
    "test_feature_parity_status_tokens_valid",
    "matrix",
    "valid_status_tokens",
    "none_invalid" if not invalid_status_rows else ",".join(invalid_status_rows),
    not invalid_status_rows,
)

required_rows = [
    "Canonical IR + TTL model",
    "Primitive semantics (162 canonical variants)",
    "Interpreter path over canonical IR",
    "Dispatch path + transform wrappers (`jit`/`grad`/`vmap`)",
    "Cache-key determinism + strict/hardened split",
    "Cross-crate error taxonomy",
    "Security/adversarial gates",
    "Fresh-clone onboarding command gates",
    "Decision/evidence ledger foundation",
    "Conformance harness + transform bundle runner",
    "Legacy fixture capture automation",
    "`jit` transform semantics",
    "`grad` transform semantics",
    "`vmap` transform semantics",
    "RNG implementation (ThreeFry + samplers)",
    "RNG determinism vs JAX oracle",
    "RaptorQ sidecar durability pipeline",
    "DType system (11 types + promotion rules)",
    "AD completeness",
    "Numerical stability + platform determinism",
    "Control flow (`cond`/`scan`/`while`/`fori_loop`/`switch`)",
    "Python bindings (`fj-py`)",
    "Tracing from user code + nested transform tracing",
    "E-graph optimization pipeline",
    "Special functions + linear algebra + FFT",
    "CPU parallel backend",
]
missing_rows = [name for name in required_rows if name not in lookup]
check(
    checks,
    "test_feature_parity_required_rows_present",
    "matrix",
    "all_required_rows_present",
    "none_missing" if not missing_rows else ",".join(missing_rows),
    not missing_rows,
)

primitive_count = count_primitives(core_path)
prim_row = lookup.get("Primitive semantics (162 canonical variants)")
if prim_row is None:
    check(
        checks,
        "test_feature_parity_all_primitives_listed",
        "Primitive semantics (162 canonical variants)",
        "row_present",
        "missing",
        False,
    )
else:
    evidence = prim_row["current_evidence"]
    match = re.search(r"(\d+)\s+ops", evidence)
    parsed_count = int(match.group(1)) if match else None
    check(
        checks,
        "test_feature_parity_all_primitives_listed",
        "Primitive semantics (162 canonical variants)",
        str(parsed_count) if parsed_count is not None else "missing_count",
        str(primitive_count),
        parsed_count == primitive_count,
    )

for transform_feature in [
    "`jit` transform semantics",
    "`grad` transform semantics",
    "`vmap` transform semantics",
]:
    check(
        checks,
        "test_feature_parity_all_transforms_listed",
        transform_feature,
        "row_present",
        "present" if transform_feature in lookup else "missing",
        transform_feature in lookup,
    )

repo_root = feature_path.parent.parent.parent
for row in rows:
    evidence = row["current_evidence"]
    cited_paths = re.findall(r"`((?:crates|artifacts)/[^`]+)`", evidence)
    missing_paths = [p for p in cited_paths if not (repo_root / p).exists()]
    check(
        checks,
        "test_feature_parity_evidence_paths_exist",
        row["feature"],
        "all_cited_paths_exist",
        "ok" if not missing_paths else ",".join(missing_paths),
        bool(cited_paths) and not missing_paths,
    )

all_pass = all(bool(c["pass"]) for c in checks)
now_ms = int(time.time() * 1000)

e2e_checks = [
    {
        "feature": c["feature"],
        "claimed_status": c["claimed_status"],
        "test_evidence": c["test_name"],
        "match": c["match"],
        "pass": c["pass"],
    }
    for c in checks
]

if e2e_out.exists():
    try:
        existing_e2e = json.loads(e2e_out.read_text(encoding="utf-8"))
        if existing_e2e.get("checks") == e2e_checks and isinstance(
            existing_e2e.get("generated_at_unix_ms"), int
        ):
            now_ms = existing_e2e["generated_at_unix_ms"]
    except Exception:
        pass

ci_payload = {
    "test_name": "feature_parity_validation",
    "generated_at_unix_ms": now_ms,
    "checks": checks,
}
ci_out.write_text(json.dumps(ci_payload, indent=2) + "\n", encoding="utf-8")

e2e_payload = {
    "scenario": "e2e_feature_parity_doc_validation",
    "generated_at_unix_ms": now_ms,
    "checks": e2e_checks,
}
e2e_out.write_text(json.dumps(e2e_payload, indent=2) + "\n", encoding="utf-8")

log_payload = {
    "test_name": "feature_parity_validation_log",
    "generated_at_unix_ms": now_ms,
    "pass": all_pass,
    "checks": checks,
}
log_out.write_text(json.dumps(log_payload, indent=2) + "\n", encoding="utf-8")

print(f"feature-parity validation: {'PASS' if all_pass else 'FAIL'}")
print(f"  checks: {len(checks)}")
print(f"  ci: {ci_out}")
print(f"  e2e: {e2e_out}")
print(f"  log: {log_out}")

if not all_pass:
    sys.exit(1)
PY
