"""Stage 0 archive audit (Milestone 1).

Forensic, read-only inspection of the Zenodo deposit (data/raw/seq2yield.zip). Produces the
five manifest deliverables and selectively extracts data CSVs (-> data/extracted/) and
project notebooks (-> archive_notebooks_readonly/, as read-only seed material).

NEVER executes a notebook. NEVER writes to data/raw/. See docs/PROJECT_SPEC.md §11.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

# --- classification --------------------------------------------------------------------

# AppleDouble / macOS resource-fork junk that pollutes zips made on macOS.
_JUNK_RE = re.compile(r"(^__MACOSX/|/\._|^\._|/\.DS_Store$)")

# Vendored third-party libraries inside the deposit (not project code/data).
_VENDORED_RE = re.compile(r"/(deeplift|custom_verstack|verstack)/|\.git/|\.egg-info")

# The two canonical raw datasets (REPRODUCTION.md §10).
_RAW_DATASETS = ("to_import/Ecoli_data.csv", "to_import/yeast_data.csv")

# Provided split artifacts: _saved/iteration_N/{_working_set,_heldout_set}.csv
_SPLIT_RE = re.compile(r"_saved/iteration_(\d+)/(_working_set|_heldout_set)\.csv$")

# Project notebooks (the real ones, excluding vendored examples).
_PROJECT_NB_RE = re.compile(r"^seq2yield/(?:[^/]+\.ipynb$|hyperOpt/[^/]+\.ipynb$)")


def is_junk(path: str) -> bool:
    return bool(_JUNK_RE.search(path))


def role_guess(path: str) -> str:
    """Coarse role for file_inventory. data|split|result|notebook|script|model|figure|
    vendored|config|unknown."""
    low = path.lower()
    if _VENDORED_RE.search(path):
        return "vendored"
    if _SPLIT_RE.search(path):
        return "split"
    if any(path.endswith(d) for d in _RAW_DATASETS):
        return "data"
    ext = os.path.splitext(low)[1]
    if "/results/" in low and ext == ".csv":
        return "result"
    if ext == ".csv":
        return "data"
    if ext == ".ipynb":
        return "notebook"
    if ext in (".py", ".sh"):
        return "script"
    if ext in (".h5", ".pkl", ".npy", ".pb", ".pt"):
        return "model"
    if ext in (".png", ".pdf", ".svg", ".jpg", ".jpeg"):
        return "figure"
    if ext in (".yml", ".yaml", ".json", ".txt", ".md", ".cfg", ".ini"):
        return "config"
    return "unknown"


# --- hashing ---------------------------------------------------------------------------

def sha256_path(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _sha256_member(zf: zipfile.ZipFile, name: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with zf.open(name) as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# --- analysis --------------------------------------------------------------------------

def analyze_dataset(zf: zipfile.ZipFile, name: str) -> dict:
    """Read a CSV from the zip and summarize schema. Detects sequence/target/series cols."""
    with zf.open(name) as f:
        df = pd.read_csv(io.BytesIO(f.read()))
    cols = list(df.columns)

    seq_cols, target_cols, series_cols = [], [], []
    for c in cols:
        s = df[c]
        if s.dtype == object:
            sample = s.dropna().astype(str).head(200)
            if len(sample) and (sample.str.fullmatch(r"[ACGTUacgtu]+").mean() > 0.9):
                seq_cols.append(c)
        name_l = str(c).lower()
        if any(k in name_l for k in ("expr", "fluor", "gfp", "yfp", "prot", "bin", "target", "yield")):
            if pd.api.types.is_numeric_dtype(s):
                target_cols.append(c)
        if "series" in name_l or "mutational" in name_l:
            series_cols.append(c)

    # sequence-length stats on the first detected sequence column
    seq_len = {}
    if seq_cols:
        lens = df[seq_cols[0]].dropna().astype(str).str.len()
        seq_len = {"column": seq_cols[0], "min": int(lens.min()), "max": int(lens.max()),
                   "modal": int(lens.mode().iloc[0])}

    return {
        "path": name,
        "n_rows": int(len(df)),
        "n_columns": len(cols),
        "columns": cols,
        "dtypes": {c: str(df[c].dtype) for c in cols},
        "sequence_columns": seq_cols,
        "target_columns": target_cols,
        "series_columns": series_cols,
        "sequence_length": seq_len,
        "missingness_summary": {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()},
    }


def analyze_notebook(zf: zipfile.ZipFile, name: str) -> dict:
    with zf.open(name) as f:
        nb = json.load(io.BytesIO(f.read()))
    cells = nb.get("cells", [])
    code = ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"]
    blob = "\n".join(code)
    imports = sorted(set(re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", blob, re.M)))
    reads = sorted(set(re.findall(r"(?:read_csv|np\.load|read_parquet|open)\(\s*[\"']([^\"']+)", blob)))
    writes = sorted(set(re.findall(r"(?:to_csv|np\.save|savefig|to_parquet|\.save)\(\s*[\"']([^\"']+)", blob)))
    colab = bool(re.search(r"google\.colab|/content/drive|drive\.mount", blob))
    return {
        "path": name,
        "n_cells": len(cells),
        "n_code_cells": len(code),
        "imports": imports,
        "reads_files": reads[:50],
        "writes_files": writes[:50],
        "colab_specific": colab,
    }


# --- orchestration ---------------------------------------------------------------------

@dataclass
class AuditResult:
    archive_sha256: str
    n_members: int
    n_junk: int
    inventory: list = field(default_factory=list)
    datasets: list = field(default_factory=list)
    notebooks: list = field(default_factory=list)
    splits: dict = field(default_factory=dict)
    gaps: list = field(default_factory=list)


def run_audit(
    zip_path: str,
    manifests_dir: str,
    extract_dir: str,
    notebooks_dir: str,
    expected: dict | None = None,
    hash_members: bool = True,
) -> AuditResult:
    expected = expected or {}
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(notebooks_dir, exist_ok=True)

    print(f"[audit] hashing archive {zip_path} ...")
    archive_sha = sha256_path(zip_path)

    zf = zipfile.ZipFile(zip_path)
    infos = zf.infolist()
    n_junk = sum(1 for i in infos if is_junk(i.filename))

    inventory, datasets, notebooks = [], [], []
    split_index: dict[str, dict] = {}

    real = [i for i in infos if not i.is_dir() and not is_junk(i.filename)]
    print(f"[audit] {len(real)} real files ({n_junk} junk members skipped); building inventory ...")

    for idx, i in enumerate(real, 1):
        name = i.filename
        role = role_guess(name)
        rec = {
            "path": name,
            "extension": os.path.splitext(name)[1].lower(),
            "size_bytes": i.file_size,
            "crc32": format(i.CRC & 0xFFFFFFFF, "08x"),
            "sha256": _sha256_member(zf, name) if hash_members else None,
            "role_guess": role,
        }
        inventory.append(rec)

        m = _SPLIT_RE.search(name)
        if m:
            it = f"iteration_{m.group(1)}"
            split_index.setdefault(it, {})[m.group(2).lstrip("_")] = {
                "path": name, "size_bytes": i.file_size, "sha256": rec["sha256"]}
        if idx % 100 == 0:
            print(f"[audit]   inventoried {idx}/{len(real)}")

    # datasets: the two canonical raw CSVs
    for i in real:
        if any(i.filename.endswith(d) for d in _RAW_DATASETS):
            print(f"[audit] analyzing dataset {i.filename} ...")
            datasets.append(analyze_dataset(zf, i.filename))

    # notebooks: project notebooks only (not vendored examples)
    for i in real:
        if i.filename.endswith(".ipynb") and _PROJECT_NB_RE.match(i.filename):
            notebooks.append(analyze_notebook(zf, i.filename))

    # --- selective extraction --------------------------------------------------------
    # data CSVs + split CSVs -> data/extracted/ (for Milestone 2); notebooks -> read-only.
    to_extract = [i for i in real
                  if any(i.filename.endswith(d) for d in _RAW_DATASETS) or _SPLIT_RE.search(i.filename)]
    for i in to_extract:
        dest = os.path.join(extract_dir, i.filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(i.filename) as src, open(dest, "wb") as out:
            out.write(src.read())
    print(f"[audit] extracted {len(to_extract)} data/split CSVs to {extract_dir}")

    for nb in notebooks:
        rel = nb["path"][len("seq2yield/"):] if nb["path"].startswith("seq2yield/") else nb["path"]
        dest = os.path.join(notebooks_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):  # prior run marked it read-only; clear before rewrite
            try:
                os.chmod(dest, 0o644)
            except OSError:
                pass
        with zf.open(nb["path"]) as src, open(dest, "wb") as out:
            out.write(src.read())
        try:  # best-effort read-only (seed material must never be edited/executed)
            os.chmod(dest, 0o444)
        except OSError:
            pass
    print(f"[audit] copied {len(notebooks)} project notebooks to {notebooks_dir} (read-only)")

    res = AuditResult(
        archive_sha256=archive_sha, n_members=len(infos), n_junk=n_junk,
        inventory=inventory, datasets=datasets, notebooks=notebooks,
        splits=split_index,
    )
    res.gaps = _reconcile(res, expected)

    _write_manifests(manifests_dir, zip_path, res)
    return res


def _reconcile(res: AuditResult, expected: dict) -> list:
    gaps = []
    # dataset expectations
    exp = expected.get("primary_dataset", {})
    ecoli = next((d for d in res.datasets if d["path"].endswith("Ecoli_data.csv")), None)
    if ecoli:
        if exp.get("expected_n_sequences") and abs(ecoli["n_rows"] - exp["expected_n_sequences"]) > 0.1 * exp["expected_n_sequences"]:
            gaps.append(f"Ecoli rows {ecoli['n_rows']} differ >10% from expected {exp['expected_n_sequences']}.")
        if not ecoli["sequence_columns"]:
            gaps.append("No sequence column auto-detected in Ecoli_data.csv (ACGT heuristic); verify manually.")
        if not ecoli["target_columns"]:
            gaps.append("No target/expression column auto-detected in Ecoli_data.csv; verify manually.")
        if ecoli["sequence_length"] and ecoli["sequence_length"].get("modal") != exp.get("sequence_length_nt", 96):
            gaps.append(f"Ecoli modal sequence length {ecoli['sequence_length'].get('modal')} != expected {exp.get('sequence_length_nt', 96)}.")
    else:
        gaps.append("Ecoli_data.csv not found in archive.")

    # split structure: README said _saved/saved_sets/; reality is _saved/iteration_N/.
    if res.splits:
        gaps.append(
            f"SPLIT LOCATION CORRECTION: provided splits live in "
            f"_saved/iteration_N/{{_working_set,_heldout_set}}.csv "
            f"({len(res.splits)} iterations: {', '.join(sorted(res.splits))}), "
            f"NOT _saved/saved_sets/ as the deposit README implied. "
            f"Each iteration = one Monte-Carlo CV repeat with its own held-out set.")
    else:
        gaps.append("No iteration split files (_working_set/_heldout_set) detected; investigate.")

    # notebook reproducibility risks
    colab_nbs = [n["path"] for n in res.notebooks if n["colab_specific"]]
    if colab_nbs:
        gaps.append(f"{len(colab_nbs)} notebooks contain Colab/Drive path assumptions: "
                    f"{', '.join(os.path.basename(p) for p in colab_nbs)}.")
    return gaps


def _write_manifests(manifests_dir: str, zip_path: str, res: AuditResult) -> None:
    now = datetime.now(timezone.utc).isoformat()

    # 1. archive_manifest.json
    manifest = {
        "archive_name": os.path.basename(zip_path),
        "archive_sha256": res.archive_sha256,
        "audited_at": now,
        "n_members": res.n_members,
        "n_junk_skipped": res.n_junk,
        "n_real_files": len(res.inventory),
        "files": res.inventory,
        "datasets": res.datasets,
        "notebooks": res.notebooks,
        "provided_splits": res.splits,
    }
    with open(os.path.join(manifests_dir, "archive_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 2. file_inventory.csv
    with open(os.path.join(manifests_dir, "file_inventory.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "extension", "size_bytes", "crc32", "sha256", "role_guess"])
        w.writeheader()
        w.writerows(res.inventory)

    # 3. dataset_schema.json
    with open(os.path.join(manifests_dir, "dataset_schema.json"), "w", encoding="utf-8") as f:
        json.dump({"datasets": res.datasets, "provided_splits": res.splits}, f, indent=2)

    # 4. notebook_inventory.csv
    with open(os.path.join(manifests_dir, "notebook_inventory.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "n_cells", "n_code_cells", "colab_specific", "imports", "reads_files", "writes_files"])
        for n in res.notebooks:
            w.writerow([n["path"], n["n_cells"], n["n_code_cells"], n["colab_specific"],
                        ";".join(n["imports"]), ";".join(n["reads_files"]), ";".join(n["writes_files"])])

    # 5. reproducibility_gaps.md
    lines = [
        "# Reproducibility gaps & risks", "",
        f"_Generated {now} from `{os.path.basename(zip_path)}` "
        f"(sha256 `{res.archive_sha256[:16]}...`)._", "",
        f"- Archive members: {res.n_members} ({res.n_junk} macOS/junk skipped, "
        f"{len(res.inventory)} real files).", "",
        "## Findings", "",
    ]
    lines += [f"{i}. {g}" for i, g in enumerate(res.gaps, 1)] or ["_None detected._"]
    lines += ["", "## Notebook policy", "",
              "All notebooks above are **seed material only** and were copied read-only to "
              "`archive_notebooks_readonly/`. They are never executed in the pipeline "
              "(docs/PROJECT_SPEC.md section 11; tests/test_notebooks_not_executed.py)."]
    with open(os.path.join(manifests_dir, "reproducibility_gaps.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
