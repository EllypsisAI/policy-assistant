"""Loads the policy database. Every method returns its result in the same shape (see result())."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY_CSV = HERE / "data" / "company_policies.csv"
QUESTIONS_CSV = HERE / "tests" / "questions.csv"


def load_policies():
    with open(POLICY_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_questions():
    with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def result(method, answer, policy=None, seconds=0.0, tokens_in=0, tokens_out=0, extra=None):
    """policy is the title of the policy the answer is based on, or None when no policy applies."""
    out = {"method": method, "answer": answer, "policy": policy, "seconds": round(seconds, 3),
           "tokens_in": tokens_in, "tokens_out": tokens_out}
    out.update(extra or {})
    return out
