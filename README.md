# HW2 — Information Extraction

This repository contains my solution for **HW2: Information Extraction**, a two-stage pipeline for Re-DocRED-style Wikipedia documents.

The project has two parts:

1. **Part 1 — Named Entity Recognition (NER)**  
   A supervised BIO sequence tagger predicts entity mentions and entity types.

2. **Part 2 — Relation Extraction (RE)**  
   A supervised two-stage classifier predicts relations between ordered pairs of entities produced by Part 1.

## Repository Structure

```text
HW2-NLP-KimiaRamezan/
├── Part1/
│   ├── README.md
│   ├── ner_train.py
│   ├── ner_predict.py
│   ├── ner_utils.py
│   ├── ner_dev.json
│   └── ner_test.json
│
├── Part2/
│   ├── README.md
│   ├── make_re_train_pred_best.py
│   ├── re_train.py
│   ├── re_predict.py
│   ├── re_utils.py
│   ├── re_dev.json
│   └── re_test.json
│
├── DOCRED_CS5712/
├── requirements.txt
```

Detailed training, prediction, evaluation commands, model descriptions, and error analyses are provided in the README files inside `Part1/` and `Part2/`.

## Environment

Python **3.10+** is required.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Part 1 — Named Entity Recognition

### Final approach

- Supervised BIO tagging
- `SGDClassifier(loss="log_loss")`
- Hashed lexical and contextual token features
- Viterbi decoding for BIO consistency
- Deterministic mention grouping by:
  - entity type
  - lowercased mention string

### Final test result

| Metric | Score |
|---|---:|
| Precision | 0.7023 |
| Recall | 0.7098 |
| Micro F1 | **0.7060** |
| Macro F1 | 0.6861 |

The Part 1 prediction file passes the provided format checker with:

```text
0 error(s), 0 warning(s)
```

See [`Part1/README.md`](Part1/README.md) for full details.

## Part 2 — Relation Extraction

### Final approach

The final system uses a two-stage supervised relation extractor:

1. binary classification of `related` vs. `no_relation`;
2. multiclass prediction of the Wikidata relation ID.

The model uses entity-type, lexical, mention-count, sentence-distance, title, and local-context features. Training uses predicted entities from Part 1 with one-to-one alignment to gold training entities and negative-pair subsampling.

### Final test result

| Metric | Score |
|---|---:|
| Precision | 0.2499 |
| Recall | 0.2028 |
| Micro F1 | **0.2239** |

Gold-entity dev evaluation reaches:

```text
Precision = 0.3817
Recall    = 0.3061
Micro F1  = 0.3398
```

This gap indicates that upstream NER/entity-grouping errors are a major source of downstream relation-extraction errors.

The Part 2 prediction file also passes the provided format checker with:

```text
0 error(s), 0 warning(s)
```

See [`Part2/README.md`](Part2/README.md) for full details.

## Final Prediction Files

NER:

```text
Part1/ner_dev.json
Part1/ner_test.json
```

Relation extraction:

```text
Part2/re_dev.json
Part2/re_test.json
```

## Notes

The provided `DOCRED_CS5712/` directory contains the course data and evaluation/support scripts used for this assignment.

Large trained `.joblib` model files are not stored in this GitHub repository because they exceed GitHub's standard 100 MB file-size limit.

## Author

**Kimia Ramezan**
