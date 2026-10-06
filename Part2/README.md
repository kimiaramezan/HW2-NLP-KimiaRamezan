# Part 2 — Supervised Relation Extraction

## Final system

Part 2 uses a supervised two-stage relation extraction pipeline over the entities predicted by Part 1.

Stage 1:
- binary classifier for `related` vs. `no_relation`

Stage 2:
- multiclass classifier for the Wikidata relation ID

Both stages use `SGDClassifier(loss="log_loss")` with hashed sparse features.

Final configuration:
- training on Part 1 predicted entities
- one-to-one best alignment from gold training entities to predicted entities before transferring relation labels
- negative sampling ratio: 5
- epochs: 5
- no class-balanced weights
- related-pair threshold: 0.05
- one predicted relation per ordered entity pair
- final model: `re_model_pipeline_best.joblib`

## Files in this folder

```text
make_re_train_pred_best.py
re_train.py
re_predict.py
re_utils.py
re_model_pipeline_best.joblib
re_test.json
re_test_confirm.json
README.md
```

`re_test.json` and `re_test_confirm.json` should be identical final predictions.  
`re_test_confirm.json` was generated only to confirm reproducibility from the final submission code.

## Entity construction

The RE system uses the `vertexSet` produced by Part 1. Part 1 groups mentions deterministically by:

```text
(entity type, lowercased mention string)
```

For RE training, gold training entities are matched one-to-one to the best predicted training entity before gold relation labels are transferred. This avoids duplicating one gold relation across several fragmented predicted entities.

## Candidate pairs and features

The system forms ordered pairs of predicted entities.

Features include:
- head entity type
- tail entity type
- ordered type pair
- canonical surface names
- mention counts
- sentence distance
- same-sentence indicator
- entity order
- title membership
- local left/right lexical context
- mention order
- token gap
- words/bigrams between same-sentence mentions

Relation/type-pair combinations observed in training are used as constraints at inference time.

## Train

First create `ner_train_pred.json` using the final Part 1 model on the training split.

Then create the one-to-one aligned RE training file:

```bash
python hw2_Kimia_Ramezan/Part2/make_re_train_pred_best.py \
  hw2_Kimia_Ramezan/DOCRED_CS5712/data/train_revised.json \
  ner_train_pred.json \
  re_train_pred_best.json
```

Train:

```bash
python hw2_Kimia_Ramezan/Part2/re_train.py \
  re_train_pred_best.json \
  hw2_Kimia_Ramezan/Part2/re_model_pipeline_best.joblib \
  --negative-ratio 5 \
  --epochs 5
```

## Predict

The required inference interface is supported:

```bash
python hw2_Kimia_Ramezan/Part2/re_predict.py NER_OUTPUT.json RE_OUTPUT.json
```

Final test command:

```bash
python hw2_Kimia_Ramezan/Part2/re_predict.py \
  hw2_Kimia_Ramezan/Part1/ner_test.json \
  hw2_Kimia_Ramezan/Part2/re_test.json \
  --model hw2_Kimia_Ramezan/Part2/re_model_pipeline_best.joblib \
  --related-threshold 0.05 \
  --max-sent-distance -1
```

## Final evaluation

Development, full pipeline:

```text
Precision = 0.2519
Recall    = 0.2068
Micro F1  = 0.2272
```

Test, full pipeline:

```text
Precision = 0.2499
Recall    = 0.2028
Micro F1  = 0.2239

gold triples      = 17,448
predicted triples = 14,159
true positives    = 3,538
false positives   = 10,621
false negatives   = 13,910
```

Gold-entity development comparison:

```text
Precision = 0.3817
Recall    = 0.3061
Micro F1  = 0.3398
```

The gold-entity/full-pipeline gap shows that upstream NER/entity-grouping errors are a major source of RE errors.

## Format check

```bash
python hw2_Kimia_Ramezan/DOCRED_CS5712/check_format.py \
  hw2_Kimia_Ramezan/DOCRED_CS5712/data/test_unlabeled.json \
  hw2_Kimia_Ramezan/Part2/re_test.json \
  --re \
  --names hw2_Kimia_Ramezan/DOCRED_CS5712/data/rel_info.json
```

Final result:

```text
0 error(s), 0 warning(s)
```

## Error analysis

Relations with strong lexical/type cues, especially date and location relations, are easier. Rare relations and relations sharing the same argument-type signature are harder. The gap between gold-entity and pipeline performance shows that missed, split, or incorrectly grouped entities reduce relation recall, while false-positive entities create additional false relation candidates.

Several alternatives were evaluated during development but were not selected for the final system, including hard-negative mining, multilabel classification, entity pruning, and probability ensembling.
