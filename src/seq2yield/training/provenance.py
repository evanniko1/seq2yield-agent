"""Code provenance for a stored result (audit ML-02).

The baseline registry is a CSV of stored R² values, and comparing a candidate against it is only
valid while the code that produced those numbers still produces them. In this project that silently
stopped being true: the registry was locked 2026-06-05, RF's `max_features` was changed to `"sqrt"`
a month later (sklearn's default is 1.0), and CNN early stopping landed in between. Nothing noticed
— `environment()` recorded library versions only — so an unchanged RF measured −0.0158 R² against
its own stored baseline at N=2000 and the harness called that a rejection. It also flipped a
published model ranking (docs/RETRACTIONS.md R-2).

Three things are recorded, in increasing usefulness:

- `code_version()` — the git SHA and whether the tree was dirty. Says the code changed.
- `numerics_fingerprint()` — a hash over only the files that can change a predicted NUMBER
  (models, features, the training loop, cleaning, metrics, sampling, splits). Says whether the
  change *could* have moved a result, so an unrelated docs commit does not invalidate a registry.
- `model_defaults()` — every model family's resolved default knobs, in full. Says *what* drifted:
  this is the one that turns "the fingerprint changed" into "rf.max_features: 1.0 → sqrt".
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Files whose content can change a predicted value. Deliberately NOT the whole repo: the agents,
# orchestration, reporting and docs can all change without invalidating a stored number, and a
# guard that cries wolf on every commit would be turned off. This module is itself excluded — it
# records provenance, it does not decide numbers.
#
# V1 is FROZEN: it is the set the 2026-11-01 run-cards were stamped under, and a stamp can only be
# checked against the definition it was written with. Never edit this tuple — add to NUMERICS_GLOBS
# and bump FINGERPRINT_VERSION instead.
NUMERICS_GLOBS_V1 = (
    "src/seq2yield/models/**/*.py",
    "src/seq2yield/features/**/*.py",
    "src/seq2yield/data/cleaning.py",
    "src/seq2yield/data/loaders.py",
    "src/seq2yield/data/sampling.py",
    "src/seq2yield/data/splits.py",
    "src/seq2yield/data/families.py",
    "src/seq2yield/training/train.py",
    "src/seq2yield/training/metrics.py",
    "src/seq2yield/training/reproducibility.py",
    "src/seq2yield/statistics/bootstrap.py",
    # the two paths that turn a dataset + a spec into predictions, and the script that built the
    # stored registry: a change in any of them means a stored number is no longer reproducible
    "src/seq2yield/experiments/runner.py",
    "src/seq2yield/experiments/pooled_runner.py",
    "scripts/reproduce_baselines.py",
)
# V2 (audit ST-2): the bootstrap decides how wide an INTERVAL is, never what a model predicts —
# `reproduce_baselines.py` does not even import it. Keeping it in the value-deciding set meant the
# first honest statistics fix invalidated a registry whose every stored R² was unaffected, which is
# precisely the cry-wolf failure the set was drawn narrowly to avoid. It moved to INFERENCE_GLOBS.
NUMERICS_GLOBS = tuple(g for g in NUMERICS_GLOBS_V1
                       if g != "src/seq2yield/statistics/bootstrap.py")
# Files that decide an interval, a p-value or an accept/reject verdict. A stored CI computed under
# a different inference fingerprint is not comparable with a fresh one (the C3 fence), even when
# both describe identical predictions.
INFERENCE_GLOBS = (
    "src/seq2yield/statistics/**/*.py",
    "src/seq2yield/experiments/compare.py",
)
FINGERPRINT_VERSION = 2
FAMILIES = ("rf", "mlp", "cnn", "ridge", "svr", "transformer")
# Knobs that do not change the computed value (or encode the run's own seed, not a default).
_IGNORED_PARAMS = {"random_state", "seed", "n_jobs", "verbose", "verbose_feature_names_out"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _plain(value):
    """JSON-safe projection of a hyperparameter value; anything exotic becomes its repr, so an
    object default still participates in the hash instead of being dropped."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return repr(value)


@lru_cache(maxsize=8)
def _fingerprint(globs: tuple[str, ...], root: str | None = None) -> dict:
    """Hash of a set of source files, plus the per-file manifest it was rolled up from.

    Line endings are normalized so the same commit fingerprints identically on Windows and Linux
    checkouts. The manifest is what makes a stamp survive a change in the SET's definition — and
    it turns "the fingerprint changed" into the list of files that changed."""
    base = Path(root or ROOT)
    files = sorted({p for g in globs for p in base.glob(g) if p.is_file()})
    # full per-file digests, so the roll-up is byte-identical to the pre-manifest implementation
    # and a v1 stamp still verifies under the frozen v1 definition
    manifest = {p.relative_to(base).as_posix():
                _sha(p.read_bytes().replace(chr(13).encode(), b"")) for p in files}
    parts = [f"{rel}:{digest}" for rel, digest in manifest.items()]
    return {"fingerprint": _sha("\n".join(parts).encode())[:16], "n_files": len(files),
            "files": manifest}


@lru_cache(maxsize=4)
def numerics_fingerprint(root: str | None = None) -> dict:
    """Fingerprint of the value-deciding source files (the files that can change a prediction)."""
    return dict(_fingerprint(NUMERICS_GLOBS, root), version=FINGERPRINT_VERSION)


@lru_cache(maxsize=4)
def inference_fingerprint(root: str | None = None) -> dict:
    """Fingerprint of the files that decide intervals, p-values and verdicts (audit ST-2)."""
    return dict(_fingerprint(INFERENCE_GLOBS, root), version=FINGERPRINT_VERSION)


def verify_numerics(stored: dict | None, *, root: str | None = None) -> dict:
    """Does a stored run-card's `numerics` block still describe this checkout?

    A rolled-up hash alone cannot answer this once the SET's own definition moves: when the
    bootstrap left the value-deciding set (ST-2), every v1 stamp would have mismatched although not
    one stored number was affected. So a stamp carrying a per-file manifest is re-checked file by
    file over the set TODAY cares about, and a v1 stamp (no manifest) is compared whole, under the
    frozen v1 definition it was written with — which is the honest answer for those: they cannot be
    re-expressed, so an inference-only edit does invalidate them until they are regenerated.
    """
    if not stored or not stored.get("fingerprint"):
        return {"ok": False, "reason": "no provenance stamp", "changed": [], "version": None}
    version = int(stored.get("version") or 1)
    if not stored.get("files"):
        globs = NUMERICS_GLOBS_V1 if version == 1 else NUMERICS_GLOBS
        cur = _fingerprint(globs, root)
        ok = stored["fingerprint"] == cur["fingerprint"]
        return {"ok": ok, "version": version, "changed": [],
                "reason": ("matches" if ok else
                           f"v{version} stamp has no file manifest, and the whole-set hash differs "
                           f"({stored['fingerprint'][:8]} vs {cur['fingerprint'][:8]}): it cannot "
                           "be checked file by file, so it must be regenerated")}
    cur = _fingerprint(NUMERICS_GLOBS, root)
    changed = sorted(rel for rel, sha in cur["files"].items() if stored["files"].get(rel) != sha)
    ok = not changed
    return {"ok": ok, "version": version, "changed": changed,
            "reason": ("matches" if ok else
                       f"{len(changed)} value-deciding file(s) changed: {', '.join(changed[:4])}"
                       + (" …" if len(changed) > 4 else ""))}


@lru_cache(maxsize=1)
def code_version() -> dict:
    def git(*args) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                                 timeout=15)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None
    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"git_sha": sha, "git_sha_short": (sha[:12] if sha else None),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": (bool(status) if status is not None else None)}


@lru_cache(maxsize=1)
def model_defaults(families: tuple[str, ...] = FAMILIES) -> dict:
    """Resolved default knobs per model family, with a per-family hash.

    Recorded in full, not just hashed: the point of ML-02 is that a one-line default change moved
    every stored number, and a diff of this block names the culprit immediately."""
    from ..models import registry
    out: dict = {}
    for fam in families:
        try:
            est = registry.make(fam, seed=0)
            raw = (est.get_params() if hasattr(est, "get_params")
                   else {k: v for k, v in vars(est).items() if not k.startswith("_")})
            params = {k: _plain(v) for k, v in sorted(raw.items()) if k not in _IGNORED_PARAMS}
            blob = json.dumps(params, sort_keys=True, default=repr)
            out[fam] = {"hash": _sha(blob.encode())[:12], "params": json.loads(blob)}
        except Exception as e:                       # a missing optional dep must not break a run
            out[fam] = {"hash": None, "error": f"{type(e).__name__}: {e}"[:120]}
    return out


def stamp() -> dict:
    """The provenance block carried by every run-card and verdict."""
    return {"code": code_version(), "numerics": numerics_fingerprint(),
            "inference": inference_fingerprint(), "model_defaults": model_defaults()}


def diff_defaults(old: dict | None, new: dict | None) -> list[str]:
    """Human-readable drift between two `model_defaults` blocks, e.g.
    `rf.max_features: 1.0 -> 'sqrt'`. Empty when they agree (or cannot be compared)."""
    if not old or not new:
        return []
    lines = []
    for fam in sorted(set(old) | set(new)):
        o, n = old.get(fam) or {}, new.get(fam) or {}
        if o.get("hash") == n.get("hash"):
            continue
        op, np_ = o.get("params") or {}, n.get("params") or {}
        for key in sorted(set(op) | set(np_)):
            if op.get(key) != np_.get(key):
                lines.append(f"{fam}.{key}: {op.get(key)!r} -> {np_.get(key)!r}")
        if not op and not np_:
            lines.append(f"{fam}: defaults not recorded on one side")
    return lines
