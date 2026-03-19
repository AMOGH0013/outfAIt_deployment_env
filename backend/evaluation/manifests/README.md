# Evaluation Label Manifest

Use `clothes_manifest.json` to provide trusted labels for the `C:\Users\amogh\Desktop\clothes` dataset.

Allowed labels:

1. `upper_only`
2. `lower_only`
3. `full_outfit`
4. `unknown`

Why this matters:

1. Weak labels from filename/silhouette are only approximations.
2. Manifest labels make detection accuracy credible for presentation and tuning.

How to use:

1. Edit `clothes_manifest.json` with your ground-truth labels.
2. Run `python backend/run_internal_evaluation.py --label-manifest backend/evaluation/manifests/clothes_manifest.json`
