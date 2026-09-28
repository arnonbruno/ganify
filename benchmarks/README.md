# GANify benchmark protocol

This package is the reproducibility and release-gating layer for GANify. It
keeps every dataset, seed, model, metric, and generation stage auditable.
There is deliberately no weighted or blended quality score.

## Suites

- `configs/suites/smoke.yaml`: controlled data plus Adult, California Housing,
  Dry Bean, and KuaiRand deterministic subsets.
- `configs/suites/core.yaml`: the mandatory heterogeneous release suite,
  configured for five split seeds by five model seeds.
- `configs/suites/stress.yaml`: wide, high-cardinality, and very large tables
  needed for broad performance claims.

Dataset source/version/license/task/stressor metadata lives in
`manifests/datasets/catalog.yaml`. Downloads are intentionally deferred.
`load_dataset(...)` states the expected local path and source URL when data is
missing, and verifies SHA-256 whenever a checksum is pinned.

## Reproducibility contract

`deterministic_split` assigns unique row IDs by seed-dependent SHA-256 order.
Assignments are independent of input row order, partition hashes are stored,
and overlap or incomplete dataframe coverage raises an error.

Models implement `ModelAdapter`: `fit(train, seed, target)`,
`sample(n_rows, seed)`, and `get_config()`. A `RunManifest` records resolved
configuration, split/dataset/output hashes, all seeds, environment versions,
Git state, failures, and named fit/sample/evaluation timings.

Metric records use long form:

```text
dataset_id, model_name, split_seed, model_seed, stage, pillar, metric, value
```

The `stage` field is mandatory for generated data. Use `raw`, `calibrated`, and
`projected` as separate records; controls use `bootstrap_control` and
`permuted_control`. `aggregate_runs` preserves these keys and returns
per-metric uncertainty. `evaluate_gates` applies every rule conjunctively,
fails missing required evidence closed, and emits one decision per rule.

The YAML files use JSON-compatible syntax so the harness works offline without
adding a YAML dependency.
