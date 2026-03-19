# Color Deep Pass Comparison (2026-03-19)

## Runs

1. Baseline before deep-pass: `internal_eval_20260319_110556` (`C:\Users\amogh\Desktop\clothes`)
2. After deep-pass on old dataset: `internal_eval_20260319_113844` (`C:\Users\amogh\Desktop\clothes`)
3. After deep-pass on new dataset (no manifest): `internal_eval_20260319_115137` (`C:\Users\amogh\Desktop\new clothes`)
4. New dataset with draft manifest: `internal_eval_20260319_124027` (`C:\Users\amogh\Desktop\new clothes`)

## Core Metrics

| Run | Dataset | Images | Detection proxy | Color stability pass | Pipeline success | Mean LAB vs HSV improvement |
|---|---|---:|---:|---:|---:|---:|
| `110556` | old clothes | 34 | 0.7647 | 0.5161 | 0.4118 | -7.83% |
| `113844` | old clothes (after deep pass) | 34 | 0.7647 | 0.5161 | 0.4118 | -7.83% |
| `115137` | new clothes (unseen) | 90 | 0.3889 | 0.5055 | 0.2111 | -9.84% |
| `124027` | new clothes (draft manifest) | 90 | 0.9667 | 0.4944 | 0.4667 | -9.84% |

## Notes

1. Color did not move on the old 34-image dataset in the current evaluation metric.
2. New dataset run is lower overall and has weak-label limitations (`full_outfit`/`unknown` heavy), so detection accuracy on this set is less reliable until a proper manifest is provided.
3. Draft manifest was generated automatically from YOLO predictions to unblock review; this is useful for workflow, but not valid as final ground truth for a thesis/presentation claim.
4. Single-item scan mode is now enforced in API code (no synthetic top+bottom pair creation from one upload).
