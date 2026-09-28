# GANify (v. 2.0.0)

<p align="center">
<img width="200" height="200" src="logo.png" alt="GANify logo">
</p>

**GANify** amplifies a numeric table. It trains a generative adversarial network on rows from a single class and draws extra rows that stay inside the range of the data it saw. The name joins **GAN** with ampl**ify**.

Version 2.0.0 keeps `fit_data` / `create_bulk` and adds mixed-type tables, conditional sampling, copula scaling, and column constraints. The numbers below are from a re-fit of exact 1.1.0 and of 2.0.0 on the same splits. The release gate stays closed because the external baseline runs did not finish.

## Compared with 1.1.0

Exact 1.1.0 was imported from its own source tree and trained again. Three 2.0.0 modes were trained in the same harness:

| Mode | What it is |
| --- | --- |
| v1.1 exact | 1.1.0 source, 20 epochs, 2 critic steps, min-max scaling |
| v2.0 compatibility | The 1.1 training budget and min-max path, inside 2.0.0 |
| v2.0 numeric recipe | 40 epochs, 5 critic steps, copula scaler, generator EMA, marginal calibration |
| v2.0 conditional | 20 epochs, one model, separate heads for numeric and categorical columns |

Tables: KuaiRand video statistics (7,583 rows), California housing (capped at 6,000), Dry Bean (capped at 6,000, 7 classes), a controlled numeric table, and a controlled mixed table. Two split seeds (17, 29) and two fit seeds (101, 202). Each fit drew 1,500 rows. The table is the median of those four fits. Sample seeds 401 and 402 are averaged inside each fit. Scores are on raw generator output, before quantile calibration and before constraint projection.

| Dataset | KS, 1.1 to recipe | Spearman error, 1.1 to recipe | Classifier AUC, 1.1 to recipe |
| --- | --- | --- | --- |
| KuaiRand video | 0.307 to 0.043 | 0.267 to 0.024 | 1.00 to 0.66 |
| California housing | 0.358 to 0.068 | 0.338 to 0.097 | 1.00 to 0.57 |
| Dry Bean | 0.153 to 0.082 | 0.209 to 0.094 | 0.96 to 0.68 |
| Controlled numeric | 0.187 to 0.067 | 0.195 to 0.074 | 0.80 to 0.67 |

KS is the median column Kolmogorov-Smirnov distance. Lower means the column distributions are closer. Classifier AUC near 0.50 means a classifier cannot separate synthetic rows from real holdout rows. Spearman error is the mean absolute error of rank correlations.

Compatibility matches 1.1 on KS, Spearman error, and the downstream scores. On one KuaiRand cell the raw sample file was byte-identical to 1.1.

The recipe has the lower errors. On KuaiRand the Spearman drop versus 1.1 is -0.242 on all 4 fits (95% bootstrap interval -0.255 to -0.222). A model trained on recipe housing rows reaches test R² 0.44. The same model trained on 1.1 rows reaches -1.18. Training on real housing rows reaches 0.65. Dry Bean balanced accuracy goes from 0.59 on 1.1 rows to 0.86 on recipe rows. Training on real Dry Bean rows reaches 0.93.

Conditional results depend on the table. On housing it beats 1.1 (test R² 0.09, classifier AUC 0.92) and stays behind the recipe. On Dry Bean, balanced accuracy falls to 0.19 and classifier AUC is 1.00. On KuaiRand, Spearman error rises from 0.267 to 0.370.

A Gaussian copula fit on the same training split is still harder to detect than the recipe (KuaiRand classifier AUC 0.53 versus 0.66). Bootstrap samples copy training rows (KuaiRand exact-match rate 0.65). Every GAN fit had exact-match rate 0. Membership ROC AUC stayed around 0.5, so this audit does not separate the versions on privacy.

Quantile calibration and constraint truncation are scored apart from the generator:

- Empirical quantile calibration, applied after sampling, pulls every version including 1.1 to KuaiRand classifier AUC about 0.51 and median KS about 0.02. Spearman error stays about 0.28 for 1.1 and about 0.02 for the recipe. Use the raw numbers to compare generators.
- Truncating KuaiRand rows so a user count cannot exceed its event count sets the audited violation rate to 0. The recipe raw violation rate is 0.82 (1.1 is 1.00, a Gaussian copula is 0.38, bootstrap is 0.00). After truncation the recipe classifier AUC goes back up to 0.87.

1.1, compatibility, and the numeric recipe refuse non-numeric columns. The conditional model accepts the mixed controlled table. At 20 epochs those samples stay separable (classifier AUC 0.999) and balanced accuracy stays at chance (0.49, against 0.51 when trained on real rows). On the controlled numeric table the real classifier is already near chance (balanced accuracy 0.515), so utility there is not a stable comparison.

KuaiRand worker time per fit was about 51 seconds for 1.1, 23 seconds for the recipe, and 185 seconds for the conditional model. The recipe ran 40 epochs with a compiled training step. Full cell outputs are local run files and are not in this repository.

## Installation

```bash
pip install ganify==2.0.0
```

From a checkout:

```bash
pip install .
```

GANify needs Python 3.8 or newer, TensorFlow 2.2 or newer, NumPy, pandas, scikit-learn, matplotlib, and tqdm.

## Quick start

The modern API accepts mixed pandas tables, nullable columns, and multiclass
targets. It automatically selects conditional training when categories or
multiple target classes are present:

```python
from ganify import Ganify

model = Ganify(random_state=42, random_dim=64, max_units=256)
model.fit(
    features,                         # mixed pandas DataFrame
    target,                           # optional multiclass Series
    target_name="outcome",
    epochs=50,
    batch_size=128,
    n_critic=5,
    continuous_bins=10,
    compile=True,
)

synthetic, synthetic_target = model.sample(
    2_000,
    conditions={"segment": "rare", "outcome": "positive"},
    return_target=True,
)
model.save("ganify_model")
```

Schema inference distinguishes continuous, count, binary, categorical,
ordinal, datetime, constant, and nullable columns. Use `TableSchema` or
`schema_overrides` when a column is semantically ambiguous.

### Legacy numeric single-class API

`y_train` must contain one class on the compatibility path. The synthetic
matrix does not include the label; it is kept on `y_label_`.

```python
import numpy as np
import pandas as pd
from ganify import Ganify

rng = np.random.default_rng(0)
frame = pd.DataFrame(
    {
        "age": rng.normal(40, 8, size=256),
        "income": rng.normal(50_000, 5_000, size=256),
        "score": rng.normal(0, 1, size=256),
        "segment": np.full(256, 7.0),
    }
)
frame["target"] = 1

features = frame.drop(columns="target")
target = frame["target"]

model = Ganify(random_state=42)
model.fit_data(
    features,
    target,
    type="wgan",
    epochs=50,
    batch_size=32,
    patience=10,
)
synthetic = model.create_bulk(500, output=1)
synthetic["target"] = model.y_label_
model.plot_performance(path="ganify_loss.png", show=False)
model.save("ganify_model")

restored = Ganify.load("ganify_model")
more = restored.create_bulk(length=100, output=1)
```

`create_bulk(1000, 1)` still works. `length` is the preferred spelling, and `lenght` remains accepted.

<p align="center">
<img width="600" height="500" src="ganify.gif" alt="Original GANify walkthrough">
</p>

The animation is the original walkthrough. The class exported by the package is `Ganify`, and the target has to be a single class.

## What you get back

- Mixed-type DataFrames round-trip categories, nullable values, datetimes,
  counts, binary flags, and constants.
- Conditional sampling can fix table values or target classes while
  log-frequency training exposes rare modes to the generator.
- Each numeric column of a synthetic row lies between that column's training minimum and maximum.
- A column that never varied in the training rows is copied back exactly, including after `save` and `load`.
- One-dimensional input is treated as one feature.
- A dataframe supplies column names, so `create_bulk(..., output=1)` round-trips them. NumPy input needs `cols_names` for a dataframe result.
- `critic_scores_` holds the critic or discriminator score of each drawn row. Lower is not automatically better; the scores are a diagnostic.
- `history_["real"]`, `history_["fake"]`, and `history_["generator"]` are the per-step losses. Each point is the mean of that step only.

`type="wgan"` trains a critic with labels of -1 for real rows and +1 for generated rows, plus a gradient penalty of weight 10 (WGAN-GP). `type="gan"` trains a sigmoid discriminator with binary cross-entropy and labels of 1 and 0. Wasserstein label flipping defaults to zero because it changes the critic objective. The legacy generator is a tanh MLP; conditional mode uses residual blocks and type-specific tanh, sigmoid, and softmax heads.

An epoch shuffles the rows and visits each row once. A trailing single row is folded into the previous batch. Pass `patience` to stop after that many epochs without a sufficient drop in generator loss. `plot_performance(path=..., show=False)` writes the curves without opening a window.

`Ganify(random_state=42)` makes a fit repeatable. Construction does not reseed NumPy. `fit_data` seeds TensorFlow and Keras, then restores the global Python and NumPy generators. Shuffling, noise, and label flips use a private NumPy generator.

## Classic GAN

```python
model.fit_data(features, target, type="gan", epochs=50, batch_size=32)
```

Use the loss plot and a downstream check on your own task before treating the rows as real data. A finite loss only means the optimization step ran.

## Skewed tables, flags, and rare categories

Min-max scaling crushes skewed columns: when one column spans 65 to 535,000, ordinary rows land within a hair of -1, where the tanh generator saturates. The copula scaler maps each column through its empirical CDF instead, so heavy tails, spikes at zero, and 0/1 flags all spread across the full tanh range:

```python
model = Ganify(
    random_state=42,
    scaler="copula",
    ema_decay=0.999,
    ema_warmup_epochs=30,
    calibrate_marginals=True,
    round_integers=True,
    inequality_pairs=[("play_user_num", "play_cnt")],
)
model.fit_data(features, target, type="wgan", epochs=60, batch_size=128,
               n_critic=5, label_flip=0.0, compile=True)
synthetic = model.create_bulk(length=2000, output=1)
```

- `scaler="copula"` trains the GAN on per-column ranks. The generator only has to learn the dependence structure; the marginal shapes come from stored training quantiles.
- `calibrate_marginals=True` (needs `scaler="copula"`) fits frozen generator-output CDFs from a fixed latent pool, then maps each future value to a training quantile pointwise. Results are independent of request batch size; the learned joint ranking still comes from the generator.
- `ema_decay` keeps an exponential moving average of the generator weights and samples from it, which smooths late-training oscillations. `ema_warmup_epochs` skips the first epochs so immature weights do not pollute the average.
- `round_integers=True` rounds columns that are integral in the training data.
- `inequality_pairs=[(small, big), ...]` enforces `small <= big` on every sampled row by clipping the small side. Use it for pairs the domain guarantees, such as a user count that cannot exceed its event count.
- `label_flip=0.0` is recommended for `type="wgan"`: flipping Wasserstein labels breaks the Kantorovich objective the critic optimizes.
- `compile=True` runs each update as one compiled graph, which is about 10x faster on a GPU and reproducible per device for a fixed seed. The default eager loop is unchanged and stays CPU-friendly.

## Architectures and controlled ablations

Conditional models can use a residual critic, an attention critic, or a Fourier critic. Optional extras are PacGAN packing, a short denoising warmup, an interaction loss, a feature-matching loss, and EMA, SWA, or SWAG sampling. All of these are off by default. Configurations live under `benchmarks/configs/models/`.

## Structural constraints

`ConstraintSet` covers bounds, allowed values, pair and linear inequalities, equalities, fixed and variable sums, simplexes, and implications. A pair such as `small <= big` is stored as `small` plus a nonnegative gap, so the constraint holds by construction. Linear projection is available when a rule cannot be written that way. `sample(..., return_audit=True)` reports how many rows broke a rule before and after that step.

## Privacy

`ganify.privacy` runs membership, attribute-inference, singling-out, linkability, canary, exact-match, and nearest-neighbor checks. Distance-to-closest-record is a diagnostic. It is not a privacy guarantee. Saved quantiles, category lists, and weights can leak more than one finite sample.

`DPConfig` clips and noises critic gradients per microbatch and keeps a conservative RDP account. An epsilon is reported only when preprocessing was fixed in advance and every privacy boundary is in the account. Otherwise the report says `training_only_dp`.

## Benchmark and release gate

`benchmarks/` holds dataset manifests, controls, metric tables, privacy gates, and the release decision. Raw, calibrated, and projected samples are scored separately. A missing dataset, a failed seed, or an external baseline that did not run counts as a failure. There is no single quality score.

## Save and load

`save` writes a directory with the generator weights, the adversary weights, the EMA generator weights when used, the scaler, and `metadata.json`. `Ganify.load` restores generation, scores, column names, `y_label_`, and the loss history. Sampling starts again from `random_state`, so two loads draw the same first batch. Models saved by 1.1.x still load.

## Limits

- `fit_data` is still numeric and single-class. Use `fit` for mixed columns, missing values, and more than one target class.
- Numeric infinity is rejected. Missing values are kept as an extra mask channel in `fit`.
- Synthetic rows can still combine values that never occurred together.
- Training without DP is not a privacy mechanism. DP mode needs the full privacy report before an epsilon is meaningful.
- The generator covers one table. It does not synthesize foreign keys across tables.
- At least two rows are required. `batch_size` larger than the table is reduced to a single batch per epoch and a warning is emitted.

## Tests

```bash
python -m unittest discover -s tests -v
```

## What changed in 2.0.0

- Schemas and a reversible preprocessor for continuous, count, binary, categorical, ordinal, datetime, constant, and missing columns.
- A conditional WGAN-GP with rare-category sampling, multiclass targets, and separate output heads.
- Copula scaling, frozen marginal calibration, generator EMA, and a compiled training step.
- Structural constraints, optional architecture variants, and save format 3. Formats 1 and 2 still load.
- Evaluation for marginals, dependence, downstream utility, constraints, and privacy attacks. See [Compared with 1.1.0](#compared-with-110) for the measured difference.

## What changed in 1.1.0

- The GAN path uses 0/1 labels and binary cross-entropy. The WGAN path keeps the Wasserstein labels and adds a gradient penalty so the critic can train a small fully connected generator without the vanishing updates of weight clipping.
- Generator and adversary steps use separate optimizers. Generated rows shown to the adversary are detached from the generator.
- The generator is a plain MLP. Dropout and batch normalization were removed so sampling uses the same function that was trained. The discriminator keeps seeded dropout and no batch normalization, which is steadier for the small batches this library uses.
- Epochs cover the rows without replacement. Loss history stores each step, not a cumulative average.
- Scaling preserves constant columns and refuses non-finite input. One-hot targets with a single repeated row are recognized as one class.
- `create_bulk` generates in batches, `save` / `load` persist a fitted model, and early stopping is available through `patience`.
- Public imports no longer re-export TensorFlow or scikit-learn symbols.

## References

- Goodfellow et al., "Generative Adversarial Nets", 2014. https://arxiv.org/pdf/1406.2661.pdf
- Arjovsky, Chintala, and Bottou, "Wasserstein GAN", 2017. https://arxiv.org/abs/1701.07875
- Roth, Lucchi, Nowozin, and Hofmann, "Stabilizing Training of Generative Adversarial Networks through Regularization", 2017. https://papers.nips.cc/paper/6797-stabilizing-training-of-generative-adversarial-networks-through-regularization.pdf
- Salimans et al., "Improved Techniques for Training GANs", 2016. https://arxiv.org/abs/1606.03498

Thanks to Jason Brownlee at Machine Learning Mastery, whose WGAN walkthrough informed the original package: https://machinelearningmastery.com/
