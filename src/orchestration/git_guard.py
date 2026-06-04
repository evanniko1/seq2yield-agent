"""Protected-file guard (docs/AGENTS.md §0, configs/protected_files.yaml).

Classifies a set of changed paths against the protected-files policy and decides whether a
change may proceed. Deny-by-default: anything not explicitly freely-modifiable requires
review. This is the gate every patch passes before tests run.
"""
from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_policy() -> dict:
    cfg = yaml.safe_load((ROOT / "configs/protected_files.yaml").read_text(encoding="utf-8"))
    return cfg["protected_files"] | {"default_policy": cfg.get("default_policy", "require_review")}


def _matches(path: str, patterns: list[str]) -> bool:
    norm = path.replace("\\", "/")
    return any(fnmatch.fnmatch(norm, p) or norm.startswith(p.rstrip("*").rstrip("/") + "/")
               or norm == p for p in patterns)


def classify(path: str, policy: dict | None = None) -> str:
    """Return one of: strict | conditional | freely_modifiable | require_review."""
    policy = policy or _load_policy()
    if _matches(path, policy.get("strict", [])):
        return "strict"
    if _matches(path, policy.get("conditional", [])):
        return "conditional"
    if _matches(path, policy.get("freely_modifiable", [])):
        return "freely_modifiable"
    return "require_review"


def check_paths(paths, *, human_review: bool = False, allowed_files: list[str] | None = None) -> dict:
    """Classify changed paths and decide pass/fail.

    - strict           -> always FAIL
    - conditional      -> FAIL unless human_review
    - require_review   -> FAIL unless human_review
    - freely_modifiable-> OK (and, if allowed_files given, must be within it)
    """
    policy = _load_policy()
    allowed = allowed_files or []
    results, violations = {}, []
    for p in paths:
        klass = classify(p, policy)
        ok = True
        reason = None
        if klass == "strict":
            ok, reason = False, "strict-protected (never agent-modifiable)"
        elif klass in ("conditional", "require_review"):
            if not human_review:
                ok, reason = False, f"{klass}: requires human review"
        elif klass == "freely_modifiable" and allowed and not _matches(p, allowed):
            ok, reason = False, "not within RunSpec.allowed_files"
        results[p] = {"class": klass, "ok": ok, "reason": reason}
        if not ok:
            violations.append(p)
    return {"passed": not violations, "violations": violations, "by_path": results}


def changed_paths() -> list[str]:
    """Paths changed in the working tree (staged + unstaged + untracked), repo-relative."""
    out = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True).stdout
    paths = []
    for line in out.splitlines():
        if line.strip():
            paths.append(line[3:].strip().strip('"'))
    return paths
