"""Builds the vector index for Method 3. Run once; the result is saved as data/policy_index.json.

Each policy (title, category and text) is turned into a vector of 768 numbers by Google's
embedding model. Questions are later turned into vectors by the same model, and the policies
whose vectors point in the most similar direction are retrieved (lecture 6).
"""
import json
from datetime import date

import numpy as np
from google.genai import types

from llm import client
from policies import HERE, load_policies

EMBED_MODEL = "gemini-embedding-001"
DIMENSIONS = 768
INDEX = HERE / "data" / "policy_index.json"


def policy_text(p):
    return f"{p['title']} ({p['category']}): {p['policy_text']}"


def embed(texts, task_type):
    """Returns unit-length vectors, so a dot product is the cosine similarity."""
    r = client().models.embed_content(model=EMBED_MODEL, contents=texts, config=types.EmbedContentConfig(
        task_type=task_type, output_dimensionality=DIMENSIONS))
    v = np.array([e.values for e in r.embeddings])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


if __name__ == "__main__":
    policies = load_policies()
    texts = [policy_text(p) for p in policies]
    vectors = embed(texts, "RETRIEVAL_DOCUMENT")
    tokens = client().models.count_tokens(model=EMBED_MODEL, contents="\n".join(texts)).total_tokens
    INDEX.write_text(json.dumps({
        "model": EMBED_MODEL, "dimensions": DIMENSIONS, "built": date.today().isoformat(), "tokens": tokens,
        "policies": [{"title": p["title"], "vector": [round(float(x), 6) for x in v]} for p, v in zip(policies, vectors)],
    }))
    print(f"{len(policies)} policies embedded, about {tokens} tokens, saved {INDEX.relative_to(HERE)} "
          f"({INDEX.stat().st_size // 1024} KB)")
