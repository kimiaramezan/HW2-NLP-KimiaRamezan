#!/usr/bin/env python3
import argparse
import json
import joblib
import numpy as np
from collections import Counter
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from ner_utils import doc_to_bio, token_features, estimate_transition_logprobs

CLASSES = ["O"] + [
    f"{p}-{t}"
    for t in ("PER", "ORG", "LOC", "TIME", "NUM", "MISC")
    for p in ("B", "I")
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_labels(docs):
    counts = Counter()
    for doc in docs:
        for tags in doc_to_bio(doc):
            counts.update(tags)
    return counts


def iter_batches(docs, batch_tokens=12000):
    X, y = [], []

    for doc in docs:
        tag_sents = doc_to_bio(doc)

        for sent, tags in zip(doc["sents"], tag_sents):
            for i, tag in enumerate(tags):
                X.append(token_features(sent, i))
                y.append(tag)

                if len(y) >= batch_tokens:
                    yield X, y
                    X, y = [], []

    if y:
        yield X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_json")
    ap.add_argument("model_out")
    ap.add_argument("--alpha", type=float, default=2e-6)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-tokens", type=int, default=12000)
    ap.add_argument("--n-features", type=int, default=2**20)
    ap.add_argument(
        "--weight-power",
        type=float,
        default=0.0,
        help="0=no reweighting; 0.5=sqrt balanced",
    )
    args = ap.parse_args()

    docs = load_json(args.train_json)
    counts = count_labels(docs)
    total = sum(counts.values())

    print(f"training on {len(docs)} docs / {total:,} tokens")
    print("label counts:", dict(counts))

    n_classes = len(CLASSES)
    weights = {
        c: (total / (n_classes * max(1, counts[c]))) ** args.weight_power
        for c in CLASSES
    }
    print("sample weights:", {k: round(v, 3) for k, v in weights.items()})

    hasher = FeatureHasher(
        n_features=args.n_features,
        input_type="dict",
        alternate_sign=True,
    )

    clf = SGDClassifier(
        loss="log_loss",
        alpha=args.alpha,
        random_state=571,
        average=True,
        learning_rate="optimal",
    )

    first = True
    rng = np.random.default_rng(571)

    for epoch in range(args.epochs):
        order = rng.permutation(len(docs))
        shuffled = [docs[i] for i in order]
        seen = 0

        for feat_batch, y_batch in iter_batches(shuffled, args.batch_tokens):
            Xb = hasher.transform(feat_batch)
            yb = np.asarray(y_batch)
            sw = np.asarray(
                [weights[y] for y in y_batch],
                dtype=np.float64,
            )

            if first:
                clf.partial_fit(
                    Xb,
                    yb,
                    classes=np.asarray(CLASSES),
                    sample_weight=sw,
                )
                first = False
            else:
                clf.partial_fit(Xb, yb, sample_weight=sw)

            seen += len(y_batch)

        print(f"epoch {epoch + 1}/{args.epochs}: {seen:,} tokens")

    # Use exactly the classifier's probability-column order.
    classes = clf.classes_.tolist()
    start_logp, trans_logp = estimate_transition_logprobs(docs, classes)

    joblib.dump(
        {
            "hasher": hasher,
            "clf": clf,
            "classes": classes,
            "start_logp": start_logp,
            "trans_logp": trans_logp,
        },
        args.model_out,
        compress=3,
    )

    print(f"saved model to {args.model_out}")


if __name__ == "__main__":
    main()
