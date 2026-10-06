import re
from collections import defaultdict
import numpy as np

ENTITY_TYPES = ("PER", "ORG", "LOC", "TIME", "NUM", "MISC")


def word_shape(word: str) -> str:
    out = []
    prev = None
    for ch in word:
        if ch.isupper():
            c = "X"
        elif ch.islower():
            c = "x"
        elif ch.isdigit():
            c = "d"
        else:
            c = ch
        if c != prev or c not in {"X", "x", "d"}:
            out.append(c)
        prev = c
    return "".join(out)[:12]


def token_features(sent, i):
    w = sent[i]
    wl = w.lower()

    f = {
        "bias": 1.0,
        "word": w,
        "lower": wl,
        "prefix1": wl[:1],
        "prefix2": wl[:2],
        "prefix3": wl[:3],
        "suffix1": wl[-1:],
        "suffix2": wl[-2:],
        "suffix3": wl[-3:],
        "shape": word_shape(w),
        "is_title": w.istitle(),
        "is_upper": w.isupper(),
        "is_lower": w.islower(),
        "is_digit": w.isdigit(),
        "has_digit": any(c.isdigit() for c in w),
        "has_alpha": any(c.isalpha() for c in w),
        "has_hyphen": "-" in w,
        "has_period": "." in w,
        "is_punct": bool(w) and all(not c.isalnum() for c in w),
        "len": min(len(w), 20),
        "bos": i == 0,
        "eos": i == len(sent) - 1,
    }

    for off in (-2, -1, 1, 2):
        j = i + off
        p = f"{off:+d}:"
        if 0 <= j < len(sent):
            n = sent[j]
            f[p + "lower"] = n.lower()
            f[p + "shape"] = word_shape(n)
            f[p + "istitle"] = n.istitle()
            f[p + "isupper"] = n.isupper()
            f[p + "isdigit"] = n.isdigit()
        else:
            f[p + "BOUNDARY"] = True

    # Local word/shape conjunctions help distinguish names from ordinary
    # capitalized words and improve multi-token entity boundaries.
    prev = sent[i - 1].lower() if i > 0 else "<BOS>"
    nxt = sent[i + 1].lower() if i + 1 < len(sent) else "<EOS>"
    prev_shape = word_shape(sent[i - 1]) if i > 0 else "<BOS>"
    next_shape = word_shape(sent[i + 1]) if i + 1 < len(sent) else "<EOS>"

    f["prev+word"] = prev + "|" + wl
    f["word+next"] = wl + "|" + nxt
    f["prevshape+shape"] = prev_shape + "|" + word_shape(w)
    f["shape+nextshape"] = word_shape(w) + "|" + next_shape

    return f


def doc_to_bio(doc):
    """Return BIO tags per sentence. Resolve overlaps deterministically.

    For spans with the same start, prefer the longer span. Then greedily keep
    non-overlapping spans from left to right. Duplicate spans collapse to one.
    """
    by_sent = defaultdict(list)
    for entity in doc.get("vertexSet", []):
        for m in entity:
            s = int(m["sent_id"])
            a, b = map(int, m["pos"])
            t = str(m["type"])
            if (
                t in ENTITY_TYPES
                and 0 <= s < len(doc["sents"])
                and 0 <= a < b <= len(doc["sents"][s])
            ):
                by_sent[s].append((a, b, t))

    all_tags = []
    for s, sent in enumerate(doc["sents"]):
        tags = ["O"] * len(sent)
        spans = sorted(
            set(by_sent.get(s, [])),
            key=lambda x: (x[0], -(x[1] - x[0]), x[1], x[2]),
        )
        occupied = [False] * len(sent)

        for a, b, t in spans:
            if any(occupied[a:b]):
                continue
            tags[a] = f"B-{t}"
            for i in range(a + 1, b):
                tags[i] = f"I-{t}"
            for i in range(a, b):
                occupied[i] = True

        all_tags.append(tags)

    return all_tags


def repair_bio(tags):
    """Enforce BIO consistency: invalid I-X is converted to B-X."""
    out = []
    prev = "O"
    for tag in tags:
        if tag.startswith("I-"):
            typ = tag[2:]
            if prev not in (f"B-{typ}", f"I-{typ}"):
                tag = f"B-{typ}"
        out.append(tag)
        prev = tag
    return out


def _legal_transition(prev_tag, curr_tag):
    if not curr_tag.startswith("I-"):
        return True
    typ = curr_tag[2:]
    return prev_tag in (f"B-{typ}", f"I-{typ}")


def estimate_transition_logprobs(docs, classes, smoothing=0.1):
    """Estimate sentence-start and BIO transition log probabilities."""
    classes = list(classes)
    idx = {tag: i for i, tag in enumerate(classes)}
    c = len(classes)

    start_counts = np.full(c, smoothing, dtype=np.float64)
    trans_counts = np.full((c, c), smoothing, dtype=np.float64)

    for doc in docs:
        for tags in doc_to_bio(doc):
            if not tags:
                continue
            start_counts[idx[tags[0]]] += 1.0
            for prev_tag, curr_tag in zip(tags, tags[1:]):
                trans_counts[idx[prev_tag], idx[curr_tag]] += 1.0

    start_logp = np.log(start_counts / start_counts.sum())
    trans_logp = np.log(
        trans_counts / trans_counts.sum(axis=1, keepdims=True)
    )
    return start_logp, trans_logp


def viterbi_decode(probs, classes, start_logp, trans_logp, transition_weight=0.5):
    """Decode one sentence using classifier probabilities + BIO transitions."""
    classes = list(classes)
    n = probs.shape[0]
    c = len(classes)

    if n == 0:
        return []

    emission = np.log(np.clip(probs, 1e-12, 1.0))
    neg_inf = -1e30

    dp = np.full((n, c), neg_inf, dtype=np.float64)
    back = np.full((n, c), -1, dtype=np.int32)

    # I-X cannot begin a sentence.
    for j, tag in enumerate(classes):
        if not tag.startswith("I-"):
            dp[0, j] = emission[0, j] + transition_weight * start_logp[j]

    for t in range(1, n):
        for j, curr_tag in enumerate(classes):
            best_score = neg_inf
            best_prev = -1

            for i, prev_tag in enumerate(classes):
                if not _legal_transition(prev_tag, curr_tag):
                    continue

                score = (
                    dp[t - 1, i]
                    + transition_weight * trans_logp[i, j]
                    + emission[t, j]
                )
                if score > best_score:
                    best_score = score
                    best_prev = i

            dp[t, j] = best_score
            back[t, j] = best_prev

    path = [0] * n
    path[-1] = int(np.argmax(dp[-1]))

    for t in range(n - 1, 0, -1):
        path[t - 1] = int(back[t, path[t]])

    return [classes[j] for j in path]


def bio_to_mentions(sent, sent_id, tags):
    tags = repair_bio(tags)
    mentions = []
    i = 0

    while i < len(tags):
        tag = tags[i]
        if not tag.startswith("B-"):
            i += 1
            continue

        typ = tag[2:]
        j = i + 1
        while j < len(tags) and tags[j] == f"I-{typ}":
            j += 1

        mentions.append(
            {
                "name": " ".join(sent[i:j]),
                "type": typ,
                "sent_id": sent_id,
                "pos": [i, j],
            }
        )
        i = j

    return mentions


def group_mentions(mentions):
    """Group by (type, lowercased token-joined mention string)."""
    groups = {}
    order = []

    for m in mentions:
        key = (m["type"], m["name"].lower())
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(m)

    return [groups[k] for k in order]
