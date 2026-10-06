#!/usr/bin/env python3
import argparse, json, joblib
import numpy as np
from re_utils import pair_features, all_ordered_pairs, min_sentence_distance

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json")
    ap.add_argument("output_json")
    ap.add_argument("--model", default="re_model_pipeline_best.joblib")
    ap.add_argument("--related-threshold", type=float, default=0.05)
    ap.add_argument("--max-sent-distance", type=int, default=-1)
    args = ap.parse_args()

    docs = load(args.input_json)
    m = joblib.load(args.model)
    hasher = m["hasher"]
    bclf = m["binary_clf"]
    rclf = m["relation_clf"]
    pos_types = set(m.get("positive_type_pairs", []))
    allowed = {k:set(v) for k,v in m.get("allowed_relations_by_type_pair",{}).items()}

    one_col = int(np.where(bclf.classes_ == 1)[0][0])
    rclasses = [str(x) for x in rclf.classes_]
    r2c = {r:i for i,r in enumerate(rclasses)}

    out = []
    for doc in docs:
        cand, feats = [], []
        for h,t in all_ordered_pairs(doc):
            if args.max_sent_distance >= 0:
                if min_sentence_distance(doc["vertexSet"][h], doc["vertexSet"][t]) > args.max_sent_distance:
                    continue
            f = pair_features(doc,h,t)
            tp = f["type_pair"]
            if pos_types and tp not in pos_types:
                continue
            cand.append((h,t,tp))
            feats.append(f)

        labels = []
        if cand:
            X = hasher.transform(feats)
            pb = bclf.predict_proba(X)[:,one_col]
            pr = rclf.predict_proba(X)
            for i,(h,t,tp) in enumerate(cand):
                if float(pb[i]) < args.related_threshold:
                    continue
                rels = allowed.get(tp, set(rclasses))
                cols = [r2c[r] for r in rels if r in r2c]
                if not cols:
                    continue
                j = max(cols, key=lambda c: float(pr[i,c]))
                labels.append({"h":h,"t":t,"r":rclasses[j]})

        out.append({
            "title": doc["title"],
            "sents": doc["sents"],
            "vertexSet": doc["vertexSet"],
            "labels": labels
        })

    with open(args.output_json,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print(f"wrote {args.output_json}")

if __name__ == "__main__":
    main()
