#!/usr/bin/env python3
import argparse, json, joblib
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import vstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from re_utils import pair_features, gold_pair_relations, all_ordered_pairs

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_json")
    ap.add_argument("model_out")
    ap.add_argument("--negative-ratio", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=2e-6)
    ap.add_argument("--n-features", type=int, default=2**18)
    ap.add_argument("--seed", type=int, default=571)
    args = ap.parse_args()

    docs = load(args.train_json)
    rng = np.random.default_rng(args.seed)
    hasher = FeatureHasher(n_features=args.n_features, input_type="dict",
                           alternate_sign=True)

    b_blocks, b_y = [], []
    r_blocks, r_y = [], []
    pos_pairs = 0
    neg_count = 0
    rel_counts = Counter()
    pos_type_pairs = set()
    allowed = defaultdict(set)

    for di, doc in enumerate(docs, 1):
        p2r = gold_pair_relations(doc)
        pos = set(p2r)

        pf = []
        rf, ry = [], []
        for h, t in pos:
            f = pair_features(doc, h, t)
            pf.append(f)
            pos_pairs += 1
            tp = f["type_pair"]
            pos_type_pairs.add(tp)
            for r in sorted(p2r[(h,t)]):
                rf.append(f)
                ry.append(r)
                rel_counts[r] += 1
                allowed[tp].add(r)

        if pf:
            b_blocks.append(hasher.transform(pf))
            b_y.extend([1]*len(pf))
        if rf:
            r_blocks.append(hasher.transform(rf))
            r_y.extend(ry)

        neg_pairs = [(h,t) for h,t in all_ordered_pairs(doc) if (h,t) not in pos]
        target = min(len(neg_pairs),
                     int(round(args.negative_ratio * max(1, len(pos)))))
        if target:
            if target < len(neg_pairs):
                idx = rng.choice(len(neg_pairs), size=target, replace=False)
                neg_pairs = [neg_pairs[int(i)] for i in idx]
            nf = [pair_features(doc,h,t) for h,t in neg_pairs]
            b_blocks.append(hasher.transform(nf))
            b_y.extend([0]*len(nf))
            neg_count += len(nf)

        if di % 500 == 0 or di == len(docs):
            print(f"processed {di}/{len(docs)}")

    Xb = vstack(b_blocks, format="csr")
    Xr = vstack(r_blocks, format="csr")
    yb = np.asarray(b_y, dtype=np.int8)
    yr = np.asarray(r_y)

    print(f"positive entity pairs: {pos_pairs:,}")
    print(f"sampled no_relation: {neg_count:,}")
    print(f"positive relation instances: {len(yr):,}")
    print(f"positive ordered type pairs: {len(pos_type_pairs)}")
    print("top relation counts:", rel_counts.most_common(10))

    bclf = SGDClassifier(loss="log_loss", alpha=args.alpha,
                          random_state=args.seed, average=True,
                          max_iter=args.epochs, tol=None)
    rclf = SGDClassifier(loss="log_loss", alpha=args.alpha,
                          random_state=args.seed+1, average=True,
                          max_iter=args.epochs, tol=None)
    bclf.fit(Xb, yb)
    rclf.fit(Xr, yr)

    joblib.dump({
        "hasher": hasher,
        "binary_clf": bclf,
        "relation_clf": rclf,
        "positive_type_pairs": sorted(pos_type_pairs),
        "allowed_relations_by_type_pair": {k: sorted(v) for k,v in allowed.items()},
        "negative_ratio": args.negative_ratio,
    }, args.model_out, compress=3)
    print(f"saved model to {args.model_out}")

if __name__ == "__main__":
    main()
