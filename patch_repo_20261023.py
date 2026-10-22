#!/usr/bin/env python3
"""
patch_repo_20261023.py — one-off documentation consistency patch.

Run from the repo root in WSL:
    python3 patch_repo_20261023.py
then:
    git status   # inspect
    git add -A
    git commit -m "Docs: mark Plan C verified with final numbers; cleanup"
    git tag -d v1.2-planc && git push origin :refs/tags/v1.2-planc
    git tag v1.2-planc && git push origin master --tags

Changes:
  1. README status banner: 'under independent verification' -> verified,
     with the final Plan C table and a pointer replacing the stale v1.1 table.
  2. PLAN_C.md 'not yet finalized' bullet -> verified status + sensitivity list.
  3. .Rhistory (empty artifact) removed and git-ignored.
  4. run_monte_carlo.py: restore the module docstring if it was stripped.
Idempotent: running twice changes nothing the second time.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def read(p):
    with open(p, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write(p, s):
    # Preserve the file's existing dominant newline (CRLF on Windows checkout).
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def norm(s):
    """Collapse line endings for anchor matching only."""
    return s.replace("\r\n", "\n")


def write_keep_eol(p, original, updated):
    """Write updated content using the original file's line ending."""
    if "\r\n" in original and "\r\n" not in updated:
        updated = updated.replace("\n", "\r\n")
    write(p, updated)


def patch_readme(s):
    n = norm(s)
    old_banner_anchor = "Plan C results are under independent verification; " \
                        "validated numbers will replace the tables below. See `PLAN_C.md`."
    new_banner_anchor = (
        "Plan C results were independently verified on 2026-10-22/23 across "
        "three compute environments (Linux, WSL, Windows) and a from-scratch "
        "signed-rank recomputation (`code/verify_planC.py` on the committed "
        "`results/*_iters.json` records). **Verified Plan C results (paired, "
        "80 reps/scenario):**\n\n"
        "| Scenario | HF | LM-clean | Δ (95% CI) | Wilcoxon p |\n"
        "|---|---|---|---|---|\n"
        "| Baseline | 0.7205 | 0.7070 | +0.0135 [+.0099, +.0171] | 5.2e-09 |\n"
        "| Heavy-tail | 0.7212 | 0.7055 | +0.0157 [+.0120, +.0195] | 2.8e-10 |\n"
        "| Nonlinear | 0.8394 | 0.8326 | +0.0067 [+.0054, +.0081] | 8.8e-12 |\n"
        "| Combined | 0.8385 | 0.8331 | +0.0054 [+.0041, +.0068] | 8.5e-10 |\n\n"
        "All four differences exclude zero and survive Bonferroni (α=0.0125). "
        "Effects are small on the raw scale (0.5–1.6 AUC points); tiny p-values "
        "reflect paired-design power, not clinical magnitude. The v1.1-bugfix "
        "landmark table further down is retained for history only. See `PLAN_C.md`."
    )
    changed = False
    if old_banner_anchor in n:
        n = n.replace(old_banner_anchor, new_banner_anchor)
        changed = True
    elif "Verified Plan C results" not in n:
        print("  ! README banner anchor not found — left unchanged")
    old_head = "### Post-fix MC results (4 scenarios × 80 reps)"
    new_head = ("### Post-fix MC results — v1.1 landmarking, SUPERSEDED "
                "(in-sample; kept for history, do not cite)\n\n"
                "_The comparable, cadence-isolated numbers are the verified "
                "Plan C table in the status banner above._")
    if old_head in n:
        n = n.replace(old_head, new_head)
        changed = True
    return n, changed


def patch_planc(s):
    n = norm(s)
    old = ("- Exact effect sizes are **not yet finalized**: independent "
           "re-computation from `_iters.json` and sensitivity analyses "
           "(evaluation grid, δ horizon, landmark spacing) are in progress. "
           "Do not cite specific numbers until these complete.")
    new = ("- **Verified 2026-10-22/23.** Paired Δ with 95% CI: baseline "
           "+0.0135 [+.0099, +.0171], heavy-tail +0.0157 [+.0120, +.0195], "
           "nonlinear +0.0067 [+.0054, +.0081], combined +0.0054 [+.0041, +.0068]; "
           "Wilcoxon p from 2.8e-10 to 8.8e-12, all survive Bonferroni. "
           "Reproduced on three compute environments and via a from-scratch "
           "signed-rank recomputation (`code/verify_planC.py`). "
           "Remaining sensitivity analyses (not yet done): evaluation grid / "
           "δ horizon / landmark spacing, and replacing frozen η with "
           "data-free particle forward propagation in the nonlinear scenarios "
           "(the frozen posterior mean is exactly the conditional expectation "
           "under the linear random walk, but neglects the deterministic drift "
           "under nonlinear dynamics).")
    changed = False
    if old in n:
        n = n.replace(old, new)
        changed = True
    elif "Verified 2026-10-22/23" not in n:
        print("  ! PLAN_C status bullet not found — left unchanged")
    return n, changed


DOCSTRING = ('"""\n'
             'run_monte_carlo.py -- Monte Carlo replication for Step 2 + Robustness\n'
             '=======================================================================\n'
             'Runs N_MC independent simulation replicates to assess:\n'
             '  1. Whether High-Freq PF+Cox consistently outperforms baselines\n'
             '  2. Stable iAUC estimates with confidence intervals\n'
             '  3. Paired comparison: method A vs method B across same datasets\n'
             '  4. [v4] Model misspecification robustness analysis across 4 scenarios\n'
             '\n'
             'Scenarios (theta_mode / obs_noise):\n'
             '  baseline    : linear   / gaussian   - original setup\n'
             '  nonlinear   : nonlinear/ gaussian   - sin perturbation in state transition\n'
             '  heavy_tail  : linear   / heavy_tail - Student-t(df=3) observation noise\n'
             '  combined    : nonlinear/ heavy_tail - both misspecifications together\n'
             '\n'
             'Usage:\n'
             '    python run_monte_carlo.py --scenario baseline --n_mc 80\n'
             '    python run_monte_carlo.py --run_all --n_mc 80\n'
             '"""\n\n')


def patch_runner(s):
    if s.lstrip().startswith('"""'):
        return s  # docstring present
    return DOCSTRING + s


def main():
    print("Patching in", ROOT)
    # 1. README
    p = "README.md"
    if os.path.exists(p):
        s = read(p); s2, changed = patch_readme(s)
        if changed:
            write_keep_eol(p, s, s2)
        print("  README.md", "updated" if changed else "no change needed")
    # 2. PLAN_C
    p = "PLAN_C.md"
    if os.path.exists(p):
        s = read(p); s2, changed = patch_planc(s)
        if changed:
            write_keep_eol(p, s, s2)
        print("  PLAN_C.md", "updated" if changed else "no change needed")
    # 3. .Rhistory
    if os.path.exists(".Rhistory"):
        os.remove(".Rhistory"); print("  .Rhistory removed")
    gi = ".gitignore"
    s = read(gi) if os.path.exists(gi) else ""
    if ".Rhistory" not in s:
        s = s.rstrip("\r\n") + ("\r\n" if "\r\n" in s else "\n") + ".Rhistory" + (
            "\r\n" if "\r\n" in s else "\n")
        write(gi, s); print("  .gitignore: .Rhistory added")
    # 4. runner docstring
    p = "code/run_monte_carlo.py"
    s = read(p); s2 = patch_runner(s)
    if s2 != s:
        write_keep_eol(p, s, s2)
    print("  run_monte_carlo.py",
          "docstring restored" if s2 != s else "docstring already present")
    print("Done. Next: git add -A && git status (review) then commit/tag/push.")


if __name__ == "__main__":
    sys.exit(main())
