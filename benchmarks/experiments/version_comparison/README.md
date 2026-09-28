# GANify version comparison

Run a strict JSON protocol:

```bash
python -m benchmarks.experiments.version_comparison protocol.json
```

Every source-backed fit runs in a fresh `python -I` subprocess. The worker
prepends only the requested source root, verifies both `ganify.__file__` and
the exact expected `ganify.__version__`, and records a content hash of that
source tree. A failed cache cell is retained and is not retried unless
`--retry-failures` is passed explicitly.

Supported adapter labels are:

- `v1.1_legacy_wgan`: exact 1.1.0 `fit_data(type="wgan")`.
- `v1.2_historical_artifact`: artifact-level rescoring only. It never claims
  that historical 1.2 source is available.
- `v1.2_recipe_reimplementation`: the historical recipe through the imported
  current compatibility API, labeled as a reimplementation.
- `v2_numeric_compatibility` and `v2_numeric_recipe`.
- `v2_conditional_native`.

The two common lanes are `numeric_regression` (target synthesized as part of
one numeric table) and `multiclass_classwise` (exact deterministic class
counts). Fixed row IDs, split/fit/sample seeds, source/data hashes, timings,
failures, and stage-specific sample CSVs are persisted.

Minimal protocol shape:

```json
{
  "protocol_version": "1",
  "output_dir": "comparison-output",
  "datasets": [{
    "id": "example",
    "path": "example.csv",
    "row_id": "row_id",
    "target": "target",
    "lane": "numeric_regression",
    "task": "regression"
  }],
  "models": [{
    "name": "v1.1",
    "adapter": "v1.1_legacy_wgan",
    "source_root": "/tmp/ganify-v11",
    "expected_version": "1.1.0",
    "config": {
      "constructor": {"random_dim": 16, "max_units": 32},
      "fit": {"epochs": 5, "batch_size": 32, "n_critic": 1}
    }
  }],
  "split_seeds": [42],
  "fit_seeds": [101, 202, 303],
  "sample_seeds": [11, 22, 33],
  "stages": {"raw": {}},
  "evaluation": {
    "controls": true,
    "privacy": {"enabled": true, "max_rows": 1000}
  }
}
```

To request common postprocessing while keeping evidence separated:

```json
{
  "raw": {},
  "calibrated": {"method": "empirical_quantile", "source": "raw"},
  "projected": {
    "source": "calibrated",
    "clip_to_train_range": true,
    "round_integers": true,
    "constraints": [
      {"type": "inequality", "left": "users", "operator": "<=", "right": "events"}
    ]
  }
}
```

Aggregation first averages sample seeds inside each fitted model and metric
column. Only then does it calculate fit-level median, IQR, bootstrap 95% CI,
worst case, failure rate, and paired candidate-minus-reference deltas.

