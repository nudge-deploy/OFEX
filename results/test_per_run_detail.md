# OFEX on the held-out test set: every run

240 provisions, 194 reference obligations (AI-generated reference). Agree = 173 provisions where annotators 1 and 2 agreed (both found none, or same obligations and spans); disagree = 67 provisions where they did not. All rows are scored against the same adjudicated reference; no provision excluded, no label changed. Micro-averaged over provisions.

## Primary output: all records

| Run | Group | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1 | all 240 | 141 | 58 | 53 | 0.709 | 0.727 | 0.718 |
| 2 | all 240 | 143 | 54 | 51 | 0.726 | 0.737 | 0.731 |
| 3 | all 240 | 142 | 55 | 52 | 0.721 | 0.732 | 0.726 |
| **mean** | all 240 | 142.0 | 55.7 | 52.0 | **0.718** | **0.732** | **0.725** |
| 1 | agree 173 | 105 | 15 | 18 | 0.875 | 0.854 | 0.864 |
| 2 | agree 173 | 105 | 16 | 18 | 0.868 | 0.854 | 0.861 |
| 3 | agree 173 | 106 | 16 | 17 | 0.869 | 0.862 | 0.865 |
| **mean** | agree 173 | 105.3 | 15.7 | 17.7 | **0.871** | **0.857** | **0.863** |
| 1 | disagree 67 | 36 | 43 | 35 | 0.456 | 0.507 | 0.480 |
| 2 | disagree 67 | 38 | 38 | 33 | 0.500 | 0.535 | 0.517 |
| 3 | disagree 67 | 36 | 39 | 35 | 0.480 | 0.507 | 0.493 |
| **mean** | disagree 67 | 36.7 | 40.0 | 34.3 | **0.479** | **0.516** | **0.497** |

## Secondary output: ACCEPT only

| Run | Group | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1 | all 240 | 125 | 36 | 69 | 0.776 | 0.644 | 0.704 |
| 2 | all 240 | 126 | 38 | 68 | 0.768 | 0.649 | 0.704 |
| 3 | all 240 | 125 | 40 | 69 | 0.758 | 0.644 | 0.696 |
| **mean** | all 240 | 125.3 | 38.0 | 68.7 | **0.767** | **0.646** | **0.701** |
| mean | agree 173 | 94.3 | 11.7 | 28.7 | 0.890 | 0.767 | 0.824 |
| mean | disagree 67 | 31.0 | 26.3 | 40.0 | 0.540 | 0.437 | 0.483 |

## Keyword baseline (deterministic, 1 run)

| Group | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| all 240 | 106 | 135 | 88 | 0.440 | 0.546 | 0.487 |
