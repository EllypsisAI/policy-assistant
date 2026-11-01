"""Method 2: LLM without a vector index. All 98 policies are put into every prompt."""
import sys
import time

from llm import ask
from policies import load_policies, result

POLICIES = load_policies()


def answer(question):
    start = time.perf_counter()
    r = ask(question, POLICIES)
    return result("full_context", r["answer"], r["policy"], time.perf_counter() - start,
                  r["tokens_in"], r["tokens_out"], extra={"status": r["status"], "model": r["model"]})


if __name__ == "__main__":
    print(answer(" ".join(sys.argv[1:]) or "May I do my job from another country for a few weeks?"))
