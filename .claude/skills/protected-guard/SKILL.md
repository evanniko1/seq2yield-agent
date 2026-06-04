---
name: protected-guard
description: Check a proposed patch/diff against configs/protected_files.yaml before it is applied or committed. Use whenever an agent (or you) is about to modify files under an approved RunSpec, before any commit, or to vet a PatchPlan's allowed_files.
---

# protected-guard

The gate every patch must pass. Enforces the trust boundary: no LLM may touch strict or
(without review) conditional files.

## Steps
1. Compute the diff's touched paths (`git diff --name-only`, staged + unstaged, or the
   PatchPlan's target files).
2. Run `python -m seq2yield.orchestration.git_guard <paths...>` *(implement at Milestone 3/6;
   do not bypass)*. Classify each path via `configs/protected_files.yaml`:
   - **strict** → REJECT. Hard stop.
   - **conditional** → allowed ONLY if the originating proposal has `human_review_required:
     true` AND a recorded human approval. Otherwise REJECT.
   - **freely_modifiable** → allowed only if also within the RunSpec's `allowed_files`.
   - **unlisted** → deny by default (`default_policy: require_review`).
3. Confirm `allowed_files ∩ protected_files == ∅` for the proposal/RunSpec.
4. Write the verdict to the run folder as `protected_file_check.json`.

## Output
PASS → patch may proceed to tests. FAIL → emit reasons, send run toward REJECTED, do NOT
apply the patch.

## Guardrails
- Never edit `protected_files.yaml` to make a patch pass. Never `--no-verify` a commit.
