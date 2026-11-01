"""Shared by Methods 2 and 3: the prompt, the answer format and the Gemini call.

Both methods use exactly the same prompt and settings. The only difference is which policies
are put into the prompt: all 98 (Method 2) or the 3 closest found by the vector index (Method 3).
"""
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

from policies import load_policies

# Cheapest model first. If it fails (rate limit, overload), the next one is tried (lecture 6).
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

PROMPT = """You are a company policy assistant. Answer the employee's question using ONLY the company
policies listed below.

Rules:
- Use only what is written in the policies. Do not add numbers, limits, conditions or practices
  that are not written there, even if they are common at other companies.
- If a policy covers the topic but does not contain the detail asked for, say that the policy does
  not state it and name the policy. Use status "not_stated".
- If no policy covers the question, say so. Use status "no_policy" and policy null.
- Otherwise use status "answered".
- policy is the exact title of the single most relevant policy.
- Answer in one to three short sentences.

Company policies:
{policies}

Employee question: {question}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "policy": {"type": ["string", "null"]},
        "status": {"type": "string", "enum": ["answered", "not_stated", "no_policy"]},
    },
    "required": ["answer", "policy", "status"],
}

TITLES = {p["title"].lower(): p["title"] for p in load_policies()}


def api_key():
    """Environment variable (also how the website passes its secret), else a .env file above this folder."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    for folder in Path(__file__).resolve().parents:
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("GEMINI_API_KEY not found")


_client = None


def client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=api_key())
    return _client


def format_policies(policies):
    return "\n".join(f"- {p['title']} ({p['category']}): {p['policy_text']}" for p in policies)


def ask(question, policies):
    """Send the question with the given policies. Returns the parsed answer plus model, time and tokens."""
    prompt = PROMPT.format(policies=format_policies(policies), question=question)
    config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                         response_json_schema=SCHEMA)
    errors = []
    for model in MODELS:
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                resp = client().models.generate_content(model=model, contents=prompt, config=config)
                data = json.loads(resp.text)
            except Exception as e:                  # temporary failure: retry once, then next model
                errors.append(f"{model}: {str(e)[:120]}")
                time.sleep(2)
                continue
            u = resp.usage_metadata
            title = data.get("policy")
            data["policy"] = TITLES.get(title.strip().lower(), title) if title else None
            return {**data, "model": model, "seconds": time.perf_counter() - t0,
                    "tokens_in": u.prompt_token_count or 0,
                    "tokens_out": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)}
    raise RuntimeError("No Gemini model answered: " + " | ".join(errors))
