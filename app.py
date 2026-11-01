"""Company policy assistant: rules-based search, an LLM without a vector index and an LLM with a
vector index, compared side by side.

Tab 1 runs all three methods live on one question. Tab 2 shows how they did on the 28 test
questions. Tab 3 explains how each method works and what the comparison cannot show.

The Gemini key is read from Streamlit secrets (deployed) or a .env file outside the repository.

    streamlit run app.py
"""
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Company policy assistant", layout="wide")

try:                                    # deployed: the key is a Streamlit secret
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:                       # local: no secrets file, llm.py reads .env instead
    pass

import method1_rules
import method2_full_context
import method3_vector
from evaluate import RESULTS, TEXT, numbers
from policies import load_questions

MAX_QUESTIONS = 20      # per browser session, so the public site cannot run up the API bill
MAX_CHARS = 300

METHODS = [
    ("method1_rules", method1_rules, "1. Rules-based search",
     "Keyword matching (BM25), no AI. Returns the best-matching policy word for word."),
    ("method2_full_context", method2_full_context, "2. LLM without a vector index",
     "All 98 policies are put into the prompt. Gemini 3.5 Flash-Lite answers from them."),
    ("method3_vector", method3_vector, "3. LLM with a vector index",
     "The 3 policies closest in meaning are retrieved; only those go into the same prompt."),
]
STATUS_TEXT = {"answered": "Answered from a policy", "not_stated": "Policy found, but it does not state this",
               "no_policy": "No policy covers this"}
TYPE_TEXT = {"direct": "Worded like the policy", "paraphrase": "Different words from the policy",
             "missing_detail": "Policy exists but lacks the detail asked for",
             "out_of_scope": "No policy covers it"}

st.title("Company policy assistant")
st.write("Three ways of answering an employee's policy question from the same database of 98 company "
         "policies. Each answer shows the policy it is based on, how long it took and how many tokens it used.")
st.caption("This is a course prototype. The answers come only from the 98 short policies in the database, "
           "which often leave out details. Check the policy itself before acting on an answer.")

ask_tab, results_tab, about_tab = st.tabs(["Ask a question", "Test results", "How it works"])


# --------------------------------------------------------------------------- tab 1: ask

def support_check(r):
    """Live version of the evaluation checks: is the cited policy real, and does every number in
    the answer appear in that policy or in the question?"""
    if r["policy"] is None:
        return "No policy cited."
    if r["policy"] not in TEXT:
        return f"Cited policy '{r['policy']}' does not exist in the database."
    extra = numbers(r["answer"]) - numbers(TEXT[r["policy"]] + " " + st.session_state.get("asked", ""))
    if extra:
        return "Numbers not found in the cited policy: " + ", ".join(sorted(extra))
    return "Every number in the answer appears in the cited policy."


def show_result(r, title, how):
    st.subheader(title)
    st.caption(how)
    if "error" in r:
        st.error(r["error"])
        return
    st.markdown(f"**Answer**\n\n{r['answer']}")
    st.markdown(f"**Relevant policy:** {r['policy'] or 'none'}")
    if r["policy"] in TEXT:
        st.caption(f"Policy text: {TEXT[r['policy']]}")
    status = r.get("status") or ("answered" if r["policy"] else "no_policy")
    a, b = st.columns(2)
    a.metric("Response time", ("<1 ms" if r["seconds"] < 0.001 else f"{r['seconds'] * 1000:.0f} ms") if r["seconds"] < 1 else f"{r['seconds']:.2f} s")
    b.metric("Tokens (in / out)", f"{r['tokens_in']:,} / {r['tokens_out']:,}")
    st.markdown(f"**Type of answer:** {STATUS_TEXT[status]}")
    st.markdown(f"**Support check:** {support_check(r)}")
    if r.get("retrieved"):
        st.caption("Retrieved by the vector index (cosine similarity): " + r["retrieved"])
    if r.get("top3"):
        st.caption("Best keyword matches (BM25 score): " + r["top3"])


with ask_tab:
    examples = ["Write my own question"] + [q["question"] for q in load_questions()]
    pick = st.selectbox("Pick a test question or write your own", examples)
    question = st.text_input("Question", value="" if pick == examples[0] else pick, max_chars=MAX_CHARS,
                             placeholder="e.g. Can I work from another country for a few weeks?")
    used = st.session_state.get("used", 0)
    if st.button("Ask all three methods", type="primary", disabled=not question.strip()):
        if used >= MAX_QUESTIONS:
            st.warning(f"This session has reached its limit of {MAX_QUESTIONS} questions. Reload the page to continue.")
        else:
            st.session_state["used"] = used + 1
            st.session_state["asked"] = question.strip()
            out = []
            with st.spinner("Asking the three methods..."):
                for _, module, _, _ in METHODS:
                    try:
                        out.append(module.answer(question.strip()))
                    except Exception as e:      # e.g. every Gemini model unavailable: show it, keep the others
                        out.append({"error": f"This method failed: {str(e)[:200]}"})
            st.session_state["answers"] = out
    if "answers" in st.session_state:
        st.markdown(f"**Question:** {st.session_state['asked']}")
        for col, r, (_, _, title, how) in zip(st.columns(3), st.session_state["answers"], METHODS):
            with col:
                show_result(r, title, how)
        st.caption("The support check only catches numbers that do not come from the policy. An invented "
                   "condition without a number is not caught; in the test results, every answer was also read by hand.")


# --------------------------------------------------------------------------- tab 2: test results

@st.cache_data
def load_results():
    runs = {m: pd.read_csv(RESULTS / f"{m}.csv", dtype={"id": str}) for m, *_ in METHODS}
    run2 = {m: pd.read_csv(RESULTS / f"{m}_run2.csv", dtype={"id": str})
            for m, *_ in METHODS if (RESULTS / f"{m}_run2.csv").exists()}
    review = pd.read_csv(RESULTS / "review.csv", dtype={"id": str}).fillna("")
    return runs, run2, review


with results_tab:
    runs, run2, review = load_results()
    n = len(next(iter(runs.values())))
    st.write(f"All three methods answered the same {n} test questions, written before any method was run. "
             "Each question records which policy or policies count as correct.")
    types = runs["method1_rules"]["type"].value_counts()
    st.table(pd.DataFrame({"Question type": [TYPE_TEXT[t] for t in TYPE_TEXT],
                           "Questions": [int(types.get(t, 0)) for t in TYPE_TEXT],
                           "A good answer": ["The policy's rule", "The policy's rule",
                                             "Says the policy does not state it", "Says no policy covers it"]}))

    rows = []
    for m, _, title, _ in METHODS:
        df = runs[m]
        rv = review[review["method"] == m]
        row = {
            "Method": title,
            "Right policy": f"{df['policy_ok'].sum()}/{n}",
            "Right type of answer": f"{df['status_ok'].sum()}/{n}",
            "Unsupported answers (read by hand)": f"{(rv['unsupported'] == 'yes').sum()}/{n}",
            "Misleading answers": f"{(rv['misleading'] == 'yes').sum()}/{n}",
            "Numbers not in the policy": int((df["unknown_numbers"].fillna("") != "").sum()),
            "Mean response time": f"{df['seconds'].mean():.2f} s" if df["seconds"].mean() >= 0.01 else "<1 ms",
            "Mean tokens in / out": f"{df['tokens_in'].mean():,.0f} / {df['tokens_out'].mean():,.0f}",
        }
        row["Repeat run: right type of answer"] = (f"{run2[m]['status_ok'].sum()}/{n}" if m in run2
                                                   else "same every time (no randomness)")
        rows.append(row)
    st.subheader("Summary")
    st.dataframe(pd.DataFrame(rows).set_index("Method").T, use_container_width=True)
    st.markdown(
        "- **Unsupported answer:** the answer states something the policy database does not say, such as a "
        "number, limit or condition. Every answer was read by hand and the numbers were also checked automatically.\n"
        "- **Misleading answer:** the text itself is real, but it answers a different question. Example: rules-based "
        "search answers *How many sick days do I get per year?* with *Employees receive 15 vacation days per year.*\n"
        "- **Right type of answer:** answering when a policy covers the question, saying *not stated* when the policy "
        "lacks the detail, and saying *no policy* when nothing covers it. Rules-based search can only return a policy "
        "or nothing, so it can never say *not stated*.\n"
        "- **Repeat run:** both LLM methods were run a second time. Even at temperature 0, only 15 to 18 of the 28 "
        "answers were word-for-word the same, but the totals hardly changed.")

    st.subheader("Every answer")
    a, b = st.columns(2)
    method_title = a.selectbox("Method", [t for _, _, t, _ in METHODS])
    type_pick = b.selectbox("Question type", ["All"] + list(TYPE_TEXT.values()))
    m = next(k for k, _, t, _ in METHODS if t == method_title)
    df = runs[m].merge(review[review["method"] == m][["id", "note"]], on="id", how="left")
    if type_pick != "All":
        df = df[df["type"].map(TYPE_TEXT) == type_pick]
    shown = df[["id", "question", "answer", "policy", "expected_policies", "policy_ok", "status_ok",
                "seconds", "tokens_in", "tokens_out", "note"]].rename(columns={
        "id": "#", "question": "Question", "answer": "Answer", "policy": "Policy given",
        "expected_policies": "Correct policy", "policy_ok": "Right policy", "status_ok": "Right type",
        "seconds": "Seconds", "tokens_in": "Tokens in", "tokens_out": "Tokens out", "note": "Reviewer note"})
    st.dataframe(shown, hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- tab 3: how it works

with about_tab:
    st.markdown("""
**The database.** There are 98 company policies of one sentence each, about 2,000 tokens in total. The
department column is not used, because it does not match the policies (IT owns the Vacation Policy).

**1. Rules-based search.** The question and every policy are split into words. Common words are removed and
word endings are cut off. Each policy is then scored with BM25: rare words count more, and title words count
double. The best policy is returned word for word. It is instant and free, and it cannot invent anything. But it
misses questions that use other words ("another country" does not match "abroad"), and it cannot say yes, no or
"not stated".

**2. LLM without a vector index.** All 98 policies are put into every prompt, together with the question.
Gemini 3.5 Flash-Lite (temperature 0) must answer only from them, name the policy, and say when the policies
do not cover the question. This works because the database is small. Every question pays for all 98 policies,
so it would not work for a large policy library.

**3. LLM with a vector index (retrieval-augmented generation).** Each policy was turned once into a vector of
768 numbers by Google's embedding model `gemini-embedding-001`, and the vectors were saved as a file. A question
is turned into a vector the same way, and the 3 policies pointing in the most similar direction are retrieved.
Only those 3 go into the same prompt as method 2. This needs far fewer tokens. But if the right policy is not
among the 3, the model cannot find it.

**Reliability.** If Gemini 3.5 Flash-Lite fails, the request is retried once and then sent to Gemini 3.8 Flash.

**Limits of the comparison.**
- There are only 28 test questions on one small, clean database. A larger or messier policy library would favour
  the vector index more, and retrieval mistakes would become more likely.
- Only Google models were tested.
- The prompt tells the model to use only the policies, which is likely part of why no unsupported answers were
  found. The same model without that instruction was not tested.
- Response times are measured on the server and depend on Google's load at the time.
""")
    st.caption("Source code: https://github.com/EllypsisAI/policy-assistant")
