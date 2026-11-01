# Company policy assistant: three approaches compared

A bot that answers employee questions from a database of 98 company policies and names the policy
it used. Three approaches are compared on the same questions: rules-based search, an LLM without a
vector index, and an LLM with a vector index.

Course: AI in Accounting and Business (B150316101), SDU, autumn 2026. Assignment 1, policy case.

**Website:** https://policy-assistant.streamlit.app/

## The three methods

| # | Method | Code |
|---|---|---|
| 1 | Rules-based search: BM25 keyword scoring, no LLM. Returns the best policy word for word. | [`method1_rules.py`](method1_rules.py) |
| 2 | LLM without a vector index: all 98 policies in the prompt, Gemini 3.5 Flash-Lite answers. | [`method2_full_context.py`](method2_full_context.py) |
| 3 | LLM with a vector index: the 3 closest policies (`gemini-embedding-001`, cosine similarity) go into the same prompt. | [`method3_vector.py`](method3_vector.py), [`build_index.py`](build_index.py) |

Methods 2 and 3 share the prompt, model and settings in [`llm.py`](llm.py). The only difference
is which policies the model sees. The model answers in JSON: the answer, the policy title, and
whether the policy answers the question, the policy does not state it, or no policy covers it.

## Results on the 28 test questions

| | Rules | LLM, all policies | LLM, vector index |
|---|---|---|---|
| Right policy | 16/28 | 28/28 | 28/28 |
| Right type of answer | 21/28 | 28/28 | 23/28 |
| Unsupported answers (read by hand) | 0 | 0 | 0 |
| Misleading answers | 1 | 0 | 0 |
| Mean response time | <1 ms | 0.93 s | 1.22 s |
| Mean tokens in / out | 0 / 0 | 1,959 / 42 | 249 / 47 |

The test questions ([`tests/questions.csv`](tests/questions.csv)) were written before any method
was run. They come in four groups:
- 7 worded like the policy;
- 13 in other words;
- 4 where the policy exists but lacks the detail asked for;
- 4 that no policy covers.

The last two groups test whether a method gives answers the policy database does not support.

[`evaluate.py`](evaluate.py) runs a method on all questions and checks:
- the policy;
- the type of answer;
- numbers in the answer that are not in the cited policy;
- invented policy titles.

Every answer was also read by hand ([`results/review.csv`](results/review.csv)). Both LLM methods
were run twice (`results/*_run2.csv`).

## Slack bot

[`slack_bot.py`](slack_bot.py) answers questions in a Slack channel with Method 3. It runs locally
in Socket Mode; see the file for setup.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python build_index.py              # once; data/policy_index.json is included
.venv/bin/python evaluate.py method1_rules
.venv/bin/python evaluate.py method2_full_context
.venv/bin/python evaluate.py method3_vector
.venv/bin/streamlit run app.py
```

The LLM methods need `GEMINI_API_KEY`. Locally, it can be in the environment or in a `.env` file
outside the repository. On Streamlit Community Cloud, it is stored as an app secret. No key is
stored in this repository.

## Limits

- 28 questions on one small, clean database of one-sentence policies. With 98 policies the whole
  database fits in the prompt, so the vector index gives no accuracy gain here, only fewer tokens.
  A larger policy library would change that.
- Only Google models were tested.
- The prompt tells the model to use only the policies. The same model without that instruction
  was not tested.
- The department column of the policy file is not used, because it does not match the policies.

The code was written with an AI coding assistant (Claude Code), directed and reviewed by the
author. The method choices and the conclusions were decided in discussion and checked against the
data.
