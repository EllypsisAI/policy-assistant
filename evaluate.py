"""Runs one method over the test questions and writes results/<method>.csv.

Policy check: for questions a policy covers, the returned policy must be one of the expected
ones; for out-of-scope questions, the method must return no policy.
"""
import csv
import importlib
import sys
from collections import defaultdict
from pathlib import Path

from policies import HERE, load_questions

RESULTS = HERE / "results"


def policy_ok(q, r):
    expected = [p for p in q["expected_policies"].split(";") if p]
    if not expected:
        return r["policy"] is None
    return r["policy"] in expected


def run(module_name):
    module = importlib.import_module(module_name)
    rows = []
    for q in load_questions():
        r = module.answer(q["question"])
        rows.append({**q, **r, "policy_ok": policy_ok(q, r)})
        print(f"{q['id']:>2} {'ok ' if rows[-1]['policy_ok'] else 'NO '} {q['question'][:55]:<55} -> {r['policy']}")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{module_name}.csv"
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row["policy_ok"])
    print()
    for t, oks in by_type.items():
        print(f"{t:<15} {sum(oks)}/{len(oks)}")
    print(f"{'all':<15} {sum(r['policy_ok'] for r in rows)}/{len(rows)}   written {path.relative_to(HERE)}")
    return rows


if __name__ == "__main__":
    run(sys.argv[1])
