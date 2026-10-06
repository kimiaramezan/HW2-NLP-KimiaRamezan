#!/usr/bin/env python3
import argparse
import json
import joblib
from ner_utils import (
    token_features,
    viterbi_decode,
    bio_to_mentions,
    group_mentions,
)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json")
    ap.add_argument("output_json")
    ap.add_argument("--model", default="ner_model_viterbi_5ep.joblib")
    ap.add_argument(
        "--transition-weight",
        type=float,
        default=1.0,
        help="strength of learned BIO transition scores",
    )
    args = ap.parse_args()

    docs = load_json(args.input_json)
    bundle = joblib.load(args.model)

    hasher = bundle["hasher"]
    clf = bundle["clf"]
    classes = bundle["classes"]
    start_logp = bundle["start_logp"]
    trans_logp = bundle["trans_logp"]

    output = []

    for doc in docs:
        mentions = []

        # Decode each sentence separately because BIO sequences reset at
        # sentence boundaries.
        for sent_id, sent in enumerate(doc["sents"]):
            if not sent:
                continue

            feats = [token_features(sent, i) for i in range(len(sent))]
            X = hasher.transform(feats)
            probs = clf.predict_proba(X)

            tags = viterbi_decode(
                probs,
                classes,
                start_logp,
                trans_logp,
                transition_weight=args.transition_weight,
            )

            mentions.extend(
                bio_to_mentions(sent, sent_id, tags)
            )

        output.append(
            {
                "title": doc["title"],
                "sents": doc["sents"],
                "vertexSet": group_mentions(mentions),
            }
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
