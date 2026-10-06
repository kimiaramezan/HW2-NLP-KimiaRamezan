#!/usr/bin/env python3
import argparse, json
from collections import defaultdict

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def mention_key(m):
    try:
        return (
            int(m["sent_id"]),
            int(m["pos"][0]),
            int(m["pos"][1]),
            str(m.get("type", "")),
        )
    except Exception:
        return None

def entity_keys(entity):
    return {k for m in entity if (k := mention_key(m)) is not None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold_train")
    ap.add_argument("pred_train")
    ap.add_argument("output")
    args = ap.parse_args()

    gold_docs = load(args.gold_train)
    pred_docs = load(args.pred_train)
    assert len(gold_docs) == len(pred_docs)

    total_gold_entities = 0
    mapped_gold_entities = 0
    transferred_labels = 0
    out_docs = []

    for gd, pd in zip(gold_docs, pred_docs):
        gold_vs = gd["vertexSet"]
        pred_vs = pd["vertexSet"]

        gold_sets = [entity_keys(e) for e in gold_vs]
        pred_sets = [entity_keys(e) for e in pred_vs]

        # Map each gold entity to ONE best predicted entity.
        # Score = number of exact shared mentions. Ties: larger overlap ratio,
        # then fewer extra predicted mentions, then lower predicted index.
        g2p = {}
        used_pred = set()

        # Resolve strongest matches first so one predicted entity is not reused.
        candidates = []
        for gi, gset in enumerate(gold_sets):
            for pi, pset in enumerate(pred_sets):
                shared = len(gset & pset)
                if shared == 0:
                    continue
                union = len(gset | pset) or 1
                jacc = shared / union
                extra = len(pset - gset)
                candidates.append((shared, jacc, -extra, -pi, gi, pi))

        candidates.sort(reverse=True)
        used_gold = set()
        for shared, jacc, neg_extra, neg_pi, gi, pi in candidates:
            if gi in used_gold or pi in used_pred:
                continue
            g2p[gi] = pi
            used_gold.add(gi)
            used_pred.add(pi)

        total_gold_entities += len(gold_vs)
        mapped_gold_entities += len(g2p)

        labels = []
        seen = set()
        for lab in gd.get("labels", []):
            gh, gt = int(lab["h"]), int(lab["t"])
            if gh not in g2p or gt not in g2p:
                continue
            ph, pt = g2p[gh], g2p[gt]
            if ph == pt:
                continue
            key = (ph, pt, str(lab["r"]))
            if key in seen:
                continue
            seen.add(key)
            labels.append({"h": ph, "t": pt, "r": str(lab["r"])})
            transferred_labels += 1

        out_docs.append({
            "title": pd["title"],
            "sents": pd["sents"],
            "vertexSet": pred_vs,
            "labels": labels,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_docs, f, ensure_ascii=False)

    print(f"documents: {len(out_docs)}")
    print(f"gold entities mapped one-to-one: {mapped_gold_entities}/{total_gold_entities} "
          f"({mapped_gold_entities/total_gold_entities:.3f})")
    print(f"transferred relation labels: {transferred_labels}")
    print(f"wrote {args.output}")

if __name__ == "__main__":
    main()
