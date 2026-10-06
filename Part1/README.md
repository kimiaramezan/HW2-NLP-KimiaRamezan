# Part 1 — Supervised Entity Tagging

## Final system

Part 1 uses a supervised BIO tagger trained from scratch with scikit-learn.

- Token classifier: `SGDClassifier(loss="log_loss")`
- Feature representation: hashed lexical/context features
- Training epochs: 5
- Decoder: Viterbi with learned BIO transition probabilities
- Final transition weight: 1.0
- Mention grouping rule: same entity type + same lowercased mention string
- Final model: `ner_model_viterbi_5ep.joblib`

## Files in this folder

```text
ner_train.py
ner_predict.py
ner_utils.py
ner_model_viterbi_5ep.joblib
ner_test.json
ner_test_confirm.json
README.md
```

`ner_test.json` and `ner_test_confirm.json` should be identical final predictions.  
`ner_test_confirm.json` was generated only to confirm reproducibility from the final submission code.

## Train

From the project root:

```bash
python hw2_Kimia_Ramezan/Part1/ner_train.py \
  hw2_Kimia_Ramezan/DOCRED_CS5712/data/train_revised.json \
  hw2_Kimia_Ramezan/Part1/ner_model_viterbi_5ep.joblib \
  --epochs 5
```

## Predict

The required inference interface is supported:

```bash
python hw2_Kimia_Ramezan/Part1/ner_predict.py INPUT.json OUTPUT.json
```

Final test prediction command:

```bash
python hw2_Kimia_Ramezan/Part1/ner_predict.py \
  hw2_Kimia_Ramezan/DOCRED_CS5712/data/test_unlabeled.json \
  hw2_Kimia_Ramezan/Part1/ner_test.json \
  --model hw2_Kimia_Ramezan/Part1/ner_model_viterbi_5ep.joblib \
  --transition-weight 1.0
```

## Final evaluation

Test set:

```text
PER   P=0.6971  R=0.7675  F1=0.7306
ORG   P=0.6286  R=0.5200  F1=0.5691
LOC   P=0.6950  R=0.8233  F1=0.7537
TIME  P=0.8858  R=0.8893  F1=0.8876
NUM   P=0.7464  R=0.6875  F1=0.7157
MISC  P=0.5392  R=0.4011  F1=0.4600

MICRO P=0.7023  R=0.7098  F1=0.7060
MACRO F1=0.6861
```

This exceeds the Part 1 grading target of micro F1 >= 0.70.

## Format check

```bash
python hw2_Kimia_Ramezan/DOCRED_CS5712/check_format.py \
  hw2_Kimia_Ramezan/DOCRED_CS5712/data/test_unlabeled.json \
  hw2_Kimia_Ramezan/Part1/ner_test.json
```

Final result:

```text
0 error(s), 0 warning(s)
```

## Error analysis

`TIME` is the strongest class, followed by `LOC`, `PER`, and `NUM`. `ORG` has lower recall and `MISC` is the most difficult class overall. These categories are more lexically diverse and context-dependent. Viterbi decoding improves BIO consistency by preventing illegal `I-*` transitions.
