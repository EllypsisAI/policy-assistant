"""Runs one method over the test questions and writes results/<method>.csv.

Checks per question:
- policy_ok: the returned policy is one of the expected ones (out-of-scope: no policy returned).
- status_ok: the method said the right kind of thing. Covered questions should be answered,
  missing-detail questions should say the policy does not state it, out-of-scope questions
  should say no policy covers it. Rules-based search cannot say "not stated", so it only
  answers or finds nothing.
- unknown_numbers: numbers in the answer that are neither in the cited policy nor in the
  question. A number with no source is the clearest sign of an unsupported answer.
- policy_exists: the cited title is a real policy (an invented title is unsupported too).

Unsupported answers are also judged by hand in tests/review.csv, because an invented condition
or practice does not always contain a number.
"""
import csv
import importlib
import re
import sys
import time
from collections import defaultdict

from policies import HERE, load_policies, load_questions

RESULTS = HERE / "results"
TEXT = {p["title"]: p["policy_text"] for p in load_policies()}
EXPECTED_STATUS = {"direct": "answered", "paraphrase": "answered",
                   "missing_detail": "not_stated", "out_of_scope": "no_policy"}


def numbers(text):
    return set(re.findall(r"\d+(?:[.,]\d+)?", text or ""))


def check(q, r):
    expected = [p for p in q["expected_policies"].split(";") if p]
    status = r.get("status") or ("answered" if r["policy"] else "no_policy")
    source = TEXT.get(r["policy"], "") + " " + q["question"]
    return {
        "status": status,
        "policy_ok": (r["policy"] in expected) if expected else r["policy"] is None,
        "status_ok": status == EXPECTED_STATUS[q["type"]],
        "unknown_numbers": " ".join(sorted(numbers(r["answer"]) - numbers(source))),
        "policy_exists": r["policy"] is None or r["policy"] in TEXT,
    }


def run(module_name, tag=""):
    module = importlib.import_module(module_name)
    rows = []
    for q in load_questions():
        r = module.answer(q["question"])
        rows.append({**q, **r, **check(q, r)})
        row = rows[-1]
        print(f"{q['id']:>2} {'ok ' if row['policy_ok'] else 'NO '}{'ok ' if row['status_ok'] else 'NO '}"
              f"{q['question'][:50]:<50} -> {r['policy']} [{row['status']}] {row['unknown_numbers']}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{module_name}{tag}.csv"
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    print(f"\n{'type':<15} policy  status")
    for t, rs in list(by_type.items()) + [("all", rows)]:
        print(f"{t:<15} {sum(r['policy_ok'] for r in rs):>2}/{len(rs):<4} {sum(r['status_ok'] for r in rs):>2}/{len(rs)}")
    n = len(rows)
    print(f"answers with unknown numbers: {sum(bool(r['unknown_numbers']) for r in rows)}"
          f"   invented policy titles: {sum(not r['policy_exists'] for r in rows)}")
    print(f"mean seconds {sum(r['seconds'] for r in rows) / n:.3f}   mean tokens in {sum(r['tokens_in'] for r in rows) / n:.0f}"
          f"   out {sum(r['tokens_out'] for r in rows) / n:.0f}   written {path.relative_to(HERE)}")
    return rows


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")   # optional tag, e.g. _run2
