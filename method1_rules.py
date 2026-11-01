"""Method 1: rules-based search, no LLM.

The question and every policy are split into words, common words are removed and word endings
are cut off ("approvals" -> "approv"). Each policy gets a BM25 score: words that appear in few
policies count more, and the title counts double. The best policy is returned word for word.
If no question word appears in any policy, the answer is that no policy was found.
"""
import math
import re
import sys
import time
from collections import Counter

from policies import load_policies, result

STOPWORDS = set("""a an the and or but if of to in on at for from by with without about into over under
i me my we our you your he she it they them their is are was were be been being am do does did
have has had can could may might must shall should will would what which who whom when where why how
this that these those there here any all some no not so than too very just also as up out
policy policies company employee employees get got want need allowed ok okay please tell know""".split())

K1, B = 1.5, 0.75        # standard BM25 settings
TITLE_WEIGHT = 2


def stem(word):
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
    return word


def words(text):
    return [stem(w) for w in re.findall(r"[a-z0-9$]+", text.lower()) if w not in STOPWORDS]


POLICIES = load_policies()
DOCS = [Counter(words(p["title"]) * TITLE_WEIGHT + words(p["policy_text"])) for p in POLICIES]
AVG_LEN = sum(sum(d.values()) for d in DOCS) / len(DOCS)
DF = Counter(w for d in DOCS for w in d)
IDF = {w: math.log(1 + (len(DOCS) - n + 0.5) / (n + 0.5)) for w, n in DF.items()}


def scores(question):
    q = set(words(question))
    out = []
    for doc in DOCS:
        length = sum(doc.values())
        s = 0.0
        for w in q & doc.keys():
            tf = doc[w]
            s += IDF[w] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / AVG_LEN))
        out.append(s)
    return out


def answer(question):
    start = time.perf_counter()
    s = scores(question)
    best = max(range(len(s)), key=s.__getitem__)
    if s[best] == 0:
        return result("rules", "No matching policy was found.", None, time.perf_counter() - start)
    p = POLICIES[best]
    text = f"{p['policy_text']} (Source: {p['title']})"
    ranked = sorted(range(len(s)), key=s.__getitem__, reverse=True)[:3]
    return result("rules", text, p["title"], time.perf_counter() - start,
                  extra={"top3": "; ".join(f"{POLICIES[i]['title']} ({s[i]:.2f})" for i in ranked if s[i] > 0)})


if __name__ == "__main__":
    print(answer(" ".join(sys.argv[1:]) or "May I do my job from another country for a few weeks?"))
