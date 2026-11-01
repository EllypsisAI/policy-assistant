"""Method 3: LLM with a vector index (retrieval-augmented generation, lecture 6).

1. Retrieve: the question is embedded and compared with the saved policy vectors; the 3 most
   similar policies are kept.
2. Augment: only those 3 policies go into the prompt (the same prompt as Method 2).
3. Generate: the LLM answers from them, or says that they do not cover the question.
"""
import json
import sys
import time

import numpy as np

from build_index import EMBED_MODEL, INDEX, embed
from llm import ask, client
from policies import load_policies, result

TOP_K = 3

_index = json.loads(INDEX.read_text())
BY_TITLE = {p["title"]: p for p in load_policies()}
TITLES = [p["title"] for p in _index["policies"]]
VECTORS = np.array([p["vector"] for p in _index["policies"]])


def retrieve(question, k=TOP_K):
    q = embed([question], "RETRIEVAL_QUERY")[0]
    similarity = VECTORS @ q
    best = np.argsort(-similarity)[:k]
    return [(TITLES[i], float(similarity[i])) for i in best]


def answer(question):
    start = time.perf_counter()
    hits = retrieve(question)
    retrieve_seconds = time.perf_counter() - start
    r = ask(question, [BY_TITLE[t] for t, _ in hits])
    seconds = time.perf_counter() - start
    # the embedding API does not report usage, so the question's embedding tokens are counted separately
    embed_tokens = client().models.count_tokens(model=EMBED_MODEL, contents=question).total_tokens
    return result("vector_rag", r["answer"], r["policy"], seconds, r["tokens_in"] + embed_tokens, r["tokens_out"],
                  extra={"status": r["status"], "model": r["model"], "retrieve_seconds": round(retrieve_seconds, 3),
                         "retrieved": "; ".join(f"{t} ({s:.3f})" for t, s in hits)})


if __name__ == "__main__":
    print(answer(" ".join(sys.argv[1:]) or "May I do my job from another country for a few weeks?"))
