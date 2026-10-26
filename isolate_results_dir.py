#!/usr/bin/env python3
"""isolate_results_dir.py -- stop run_monte_carlo.py from overwriting
committed results.

The v1.2-planc code writes ../results/monte_carlo_[_iters].json (fixed names),
so any re-run silently replaces the committed v3.4 files. This patch sends all
output to ../results/$RESULTS_SUBDIR/ (default v4.0).

Exact-match edits; aborts without writing if any anchor is not found exactly
once.

Usage (repo root):
    python3 isolate_results_dir.py
"""
import sys

p = "code/run_monte_carlo.py"
t = open(p, encoding="utf-8").read()

E = [
    ("os.makedirs('../results', exist_ok=True)\n",
     "os.makedirs('../results', exist_ok=True)\n"
     "# results go to results/<subdir> so a re-run can never overwrite "
     "another version's files\n"
     "RESULTS_SUBDIR = os.environ.get('RESULTS_SUBDIR', 'v4.0')\n"
     "RESULTS_DIR = f'../results/{RESULTS_SUBDIR}'\n"
     "os.makedirs(RESULTS_DIR, exist_ok=True)\n"),
    ("f'../results/monte_carlo_{scenario}_iters_{run_timestamp}.json'",
     "f'{RESULTS_DIR}/monte_carlo_{scenario}_iters_{run_timestamp}.json'"),
    ("f'../results/monte_carlo_{scenario}_iters.json'",
     "f'{RESULTS_DIR}/monte_carlo_{scenario}_iters.json'"),
    ("f'../results/monte_carlo_{sc}_{timestamp}.json'",
     "f'{RESULTS_DIR}/monte_carlo_{sc}_{timestamp}.json'"),
    ("f'../results/monte_carlo_{sc}.json'",
     "f'{RESULTS_DIR}/monte_carlo_{sc}.json'"),
]

if "RESULTS_DIR" in t:
    sys.exit("already patched; nothing written")
for i, (a, b) in enumerate(E, 1):
    if t.count(a) != 1:
        sys.exit(f"ABORT: edit {i} matched {t.count(a)} times; nothing written")
    t = t.replace(a, b)
open(p, "w", encoding="utf-8").write(t)
print("patched code/run_monte_carlo.py: output -> ../results/$RESULTS_SUBDIR "
      "(default v4.0)")
