---
name: audit-archive
description: Run and interpret the Stage 0 forensic audit of the Zenodo data/code release (10.5281/zenodo.7273952). Use when starting Milestone 1, when data/raw/ contents change, or to refresh the dataset/notebook manifests before any reproduction work.
---

# audit-archive

Produces the Stage 0 manifests that everything downstream depends on. **Read-only over the
archive** — never executes notebooks, never trains.

## Preconditions
- The Zenodo deposit is downloaded into `data/raw/` (see docs/REPRODUCTION.md, DECISIONS.md #1).
- If `data/raw/` is empty, STOP and tell the user to fetch DOI 10.5281/zenodo.7273952.

## Steps
1. Run `python scripts/audit_archive.py` *(Milestone 1 — implement if absent; do not fake output)*.
2. Verify it produced all five deliverables under `data/manifests/`:
   - `archive_manifest.json` (sha256 of archive + per-file role guess)
   - `file_inventory.csv`
   - `dataset_schema.json` (n_rows, columns, detected sequence/target/series columns)
   - `notebook_inventory.csv` (cells, imports, reads/writes, colab_specific)
   - `reproducibility_gaps.md`
3. Reconcile detected columns/sizes against the EXPECTED values in `configs/data.yaml`
   and `docs/REPRODUCTION.md`. Log every mismatch in `reproducibility_gaps.md` and, if a
   confirmed fact changes, update REPRODUCTION.md (note it in DECISIONS.md).
4. Detect & record: missing seeds, notebook-state dependence, implicit/Colab paths,
   deprecated APIs, stored-vs-regenerated outputs.

## Exit criterion
We know what files exist, what columns exist, and which notebook/script produces which
result. Move notebooks to `archive_notebooks_readonly/` (read-only seed material).

## Guardrails
- Never run a notebook. Never write to `data/raw/`. Surface contradictions; don't paper over them.
