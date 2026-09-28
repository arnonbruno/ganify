"""Fixed-budget architecture and anti-collapse ablation runner.

The runner emits long-form, stage-preserving metrics. It deliberately has no
automatic winner-selection or blended score: promotion remains an explicit
decision made outside this module from named metrics and uncertainty.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ganify.conditional import ConditionalGANEngine
from ganify.evaluation import aggregate_table_report

from ._config import PathLike, load_config
from .splits import deterministic_split, split_frame


DEFAULT_MODEL_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "configs" / "models"
)
_WEIGHT_SOURCES = {"live", "ema", "swa", "swag"}
_FIXED_BUDGET_KEYS = {
    "epochs",
    "batch_size",
    "n_critic",
    "gradient_penalty",
}
_FORBIDDEN_LOSSES = {
    "marginal_moment_weight",
    "moment_weight",
    "marginal_loss_weight",
}


def _seed_tuple(values: Iterable[int], name: str) -> Tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds:
        raise ValueError("%s must contain at least one seed" % name)
    if len(set(seeds)) != len(seeds):
        raise ValueError("%s contains duplicate seeds" % name)
    return seeds


@dataclass(frozen=True)
class AblationModelConfig:
    """One controlled matrix row."""

    name: str
    factor: str
    description: str
    engine: Dict[str, Any]
    sampling_stages: Dict[str, str]

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any], *, fallback_name: str = ""
    ) -> "AblationModelConfig":
        if not isinstance(values, Mapping):
            raise TypeError("ablation model config must be a mapping")
        name = str(values.get("name", fallback_name)).strip()
        factor = str(values.get("factor", "")).strip()
        if not name:
            raise ValueError("ablation model config needs a non-empty name")
        if not factor:
            raise ValueError(
                "ablation model %r needs a non-empty factor" % name
            )
        engine = values.get("engine", {})
        if not isinstance(engine, Mapping):
            raise TypeError("model %r engine config must be a mapping" % name)
        forbidden = sorted(_FORBIDDEN_LOSSES.intersection(engine))
        if forbidden:
            raise ValueError(
                "model %r configures forbidden marginal moment loss keys: %r"
                % (name, forbidden)
            )
        stages = values.get("sampling_stages", {"raw_ema": "ema"})
        if not isinstance(stages, Mapping) or not stages:
            raise ValueError(
                "model %r sampling_stages must be a non-empty mapping" % name
            )
        normalized_stages = {
            str(stage): str(source).strip().lower()
            for stage, source in stages.items()
        }
        if any(not stage for stage in normalized_stages):
            raise ValueError("generation stage names must not be empty")
        unknown = sorted(
            set(normalized_stages.values()).difference(_WEIGHT_SOURCES)
        )
        if unknown:
            raise ValueError(
                "model %r has unknown weight sources: %r" % (name, unknown)
            )
        if "swa" in normalized_stages.values() and not bool(
            engine.get("swa", False)
        ):
            raise ValueError(
                "model %r samples SWA without enabling collection" % name
            )
        if "swag" in normalized_stages.values() and not bool(
            engine.get("swag", False)
        ):
            raise ValueError(
                "model %r samples SWAG without enabling collection" % name
            )
        return cls(
            name=name,
            factor=factor,
            description=str(values.get("description", "")),
            engine=dict(engine),
            sampling_stages=normalized_stages,
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AblationBudget:
    """Optimization and generation budget shared by every matrix row.

    ``batch_size`` counts raw rows. A PacGAN critic sees
    ``batch_size // pac`` packed examples per update.
    """

    epochs: int = 10
    batch_size: int = 64
    n_critic: int = 5
    gradient_penalty: float = 10.0
    sample_rows: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.n_critic < 1:
            raise ValueError("n_critic must be positive")
        if not np.isfinite(self.gradient_penalty) or self.gradient_penalty < 0:
            raise ValueError("gradient_penalty must be finite and non-negative")
        if self.sample_rows is not None and self.sample_rows < 1:
            raise ValueError("sample_rows must be positive or None")


@dataclass
class AblationResult:
    """Metrics and provenance with no implicit model promotion."""

    metrics: pd.DataFrame
    runs: pd.DataFrame
    split: Dict[str, Any]
    selected_model: None = None

    def save(self, directory: PathLike) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.metrics.to_csv(destination / "metrics.csv", index=False)
        (destination / "runs.json").write_text(
            json.dumps(
                self.runs.to_dict(orient="records"),
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        (destination / "split.json").write_text(
            json.dumps(self.split, indent=2, sort_keys=True, default=str)
            + "\n",
            encoding="utf-8",
        )
        (destination / "decision.json").write_text(
            json.dumps(
                {
                    "selected_model": None,
                    "reason": (
                        "No winner is automatically promoted by the "
                        "ablation runner."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination


def load_ablation_matrix(
    directory: PathLike = DEFAULT_MODEL_DIRECTORY,
) -> Tuple[AblationModelConfig, ...]:
    """Load every JSON-compatible model YAML in stable filename order."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(
            "ablation model config directory does not exist: %s" % root
        )
    paths = sorted(
        tuple(root.glob("*.yaml"))
        + tuple(root.glob("*.yml"))
        + tuple(root.glob("*.json"))
    )
    if not paths:
        raise ValueError("ablation model config directory is empty: %s" % root)
    configs = tuple(
        AblationModelConfig.from_mapping(
            load_config(path), fallback_name=path.stem
        )
        for path in paths
    )
    return validate_ablation_matrix(configs)


def validate_ablation_matrix(
    configs: Sequence[
        Union[AblationModelConfig, Mapping[str, Any]]
    ],
) -> Tuple[AblationModelConfig, ...]:
    """Validate names, baseline, matched widths, and matrix independence."""

    normalized = tuple(
        value
        if isinstance(value, AblationModelConfig)
        else AblationModelConfig.from_mapping(value)
        for value in configs
    )
    if not normalized:
        raise ValueError("ablation matrix must not be empty")
    names = [config.name for config in normalized]
    if len(names) != len(set(names)):
        raise ValueError("ablation matrix contains duplicate model names")
    baselines = [
        config for config in normalized if config.factor == "baseline"
    ]
    if len(baselines) != 1:
        raise ValueError(
            "ablation matrix must contain exactly one baseline row"
        )
    baseline_dims = tuple(
        baselines[0].engine.get("critic_dims", (64, 64))
    )
    baseline_generator_dims = tuple(
        baselines[0].engine.get("generator_dims", (64, 64))
    )
    baseline_noise_dim = int(baselines[0].engine.get("noise_dim", 32))
    for config in normalized:
        dims = tuple(config.engine.get("critic_dims", baseline_dims))
        if dims != baseline_dims:
            raise ValueError(
                "model %r changes critic reference widths; use parameter "
                "matching against the common baseline instead" % config.name
            )
        generator_dims = tuple(
            config.engine.get(
                "generator_dims", baseline_generator_dims
            )
        )
        if generator_dims != baseline_generator_dims:
            raise ValueError(
                "model %r changes generator widths in a critic/anti-collapse "
                "ablation" % config.name
            )
        if int(
            config.engine.get("noise_dim", baseline_noise_dim)
        ) != baseline_noise_dim:
            raise ValueError(
                "model %r changes noise_dim in the controlled matrix"
                % config.name
            )
        architecture = str(
            config.engine.get("critic_architecture", "residual")
        ).lower()
        pac = int(config.engine.get("pac", 1))
        if (
            (architecture not in {"residual", "residual_mlp", "mlp"} or pac > 1)
            and not bool(config.engine.get("match_critic_parameters", False))
        ):
            raise ValueError(
                "model %r changes critic structure without parameter matching"
                % config.name
            )
    return normalized


class AblationRunner:
    """Execute a fixed split/seed/budget matrix and retain every metric stage."""

    def __init__(
        self,
        models: Optional[
            Sequence[Union[AblationModelConfig, Mapping[str, Any]]]
        ] = None,
        *,
        budget: AblationBudget = AblationBudget(),
        split_seed: int = 0,
        model_seeds: Sequence[int] = (0,),
        sample_seeds: Sequence[int] = (0,),
        test_size: float = 0.2,
        validation_size: float = 0.1,
        c2st_folds: int = 3,
        parameter_tolerance: float = 0.08,
        common_engine_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.models = validate_ablation_matrix(
            load_ablation_matrix() if models is None else models
        )
        if not isinstance(budget, AblationBudget):
            raise TypeError("budget must be an AblationBudget")
        self.budget = budget
        self.split_seed = int(split_seed)
        self.model_seeds = _seed_tuple(model_seeds, "model_seeds")
        self.sample_seeds = _seed_tuple(sample_seeds, "sample_seeds")
        self.test_size = float(test_size)
        self.validation_size = float(validation_size)
        self.c2st_folds = int(c2st_folds)
        if self.c2st_folds < 2:
            raise ValueError("c2st_folds must be at least 2")
        self.parameter_tolerance = float(parameter_tolerance)
        if not 0.0 <= self.parameter_tolerance < 1.0:
            raise ValueError("parameter_tolerance must be in [0, 1)")
        self.common_engine_options = dict(common_engine_options or {})
        conflicting = sorted(
            _FIXED_BUDGET_KEYS.intersection(self.common_engine_options)
        )
        if conflicting:
            raise ValueError(
                "common engine options cannot override fixed budget keys: %r"
                % conflicting
            )

    def _resolved_engine_config(
        self, model: AblationModelConfig, model_seed: int
    ) -> Dict[str, Any]:
        conflicting = sorted(
            _FIXED_BUDGET_KEYS.intersection(model.engine)
        )
        if conflicting:
            raise ValueError(
                "model %r overrides fixed budget keys: %r"
                % (model.name, conflicting)
            )
        if "random_state" in model.engine:
            raise ValueError(
                "model %r embeds random_state; use model_seeds" % model.name
            )
        resolved = dict(self.common_engine_options)
        resolved.update(model.engine)
        resolved.update(
            {
                "random_state": int(model_seed),
                "epochs": self.budget.epochs,
                "batch_size": self.budget.batch_size,
                "n_critic": self.budget.n_critic,
                "gradient_penalty": self.budget.gradient_penalty,
            }
        )
        pac = int(resolved.get("pac", 1))
        if self.budget.batch_size % pac:
            raise ValueError(
                "fixed batch_size=%d is not divisible by pac=%d for %r"
                % (self.budget.batch_size, pac, model.name)
            )
        return resolved

    @staticmethod
    def _index_partitions(
        frame: pd.DataFrame,
        *,
        split_seed: int,
        test_size: float,
        validation_size: float,
        id_column: Optional[Any],
    ):
        if id_column is not None:
            split = deterministic_split(
                frame[id_column].tolist(),
                seed=split_seed,
                test_size=test_size,
                validation_size=validation_size,
            )
            return split, split_frame(frame, split, id_column=id_column)
        if not frame.index.is_unique:
            raise ValueError(
                "frame index must be unique when id_column is omitted"
            )
        split = deterministic_split(
            frame.index.tolist(),
            seed=split_seed,
            test_size=test_size,
            validation_size=validation_size,
        )
        partitions = {
            "train": frame.loc[list(split.train)].reset_index(drop=True),
            "validation": frame.loc[
                list(split.validation)
            ].reset_index(drop=True),
            "test": frame.loc[list(split.test)].reset_index(drop=True),
        }
        return split, partitions

    def run(
        self,
        frame: pd.DataFrame,
        *,
        id_column: Optional[Any] = None,
        target_column: Optional[Any] = None,
        task: Optional[str] = None,
        categorical_columns: Optional[Iterable[Any]] = None,
        constraints: Optional[Iterable[Any]] = None,
        output_directory: Optional[PathLike] = None,
    ) -> AblationResult:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("ablation runner requires a DataFrame")
        if len(frame) < 5:
            raise ValueError("ablation runner requires at least five rows")
        if id_column is not None and id_column not in frame.columns:
            raise ValueError("id_column %r is missing" % (id_column,))
        if target_column is not None and target_column not in frame.columns:
            raise ValueError("target_column %r is missing" % (target_column,))
        if target_column is not None and target_column == id_column:
            raise ValueError("target_column and id_column must differ")

        split, partitions = self._index_partitions(
            frame,
            split_seed=self.split_seed,
            test_size=self.test_size,
            validation_size=self.validation_size,
            id_column=id_column,
        )
        drop_ids = [] if id_column is None else [id_column]
        real_train = partitions["train"].drop(columns=drop_ids)
        real_test = partitions["test"].drop(columns=drop_ids)
        if len(real_test) < 2:
            raise ValueError("fixed split produced fewer than two test rows")
        sample_rows = (
            len(real_test)
            if self.budget.sample_rows is None
            else self.budget.sample_rows
        )
        categorical = (
            None
            if categorical_columns is None
            else [
                value
                for value in categorical_columns
                if value != id_column
            ]
        )
        normalized_task = None if task is None else str(task).lower()
        task_aliases = {
            "binary_classification": "classification",
            "multiclass_classification": "classification",
        }
        normalized_task = task_aliases.get(
            normalized_task, normalized_task
        )
        if normalized_task not in {None, "classification", "regression"}:
            raise ValueError("task must be classification, regression, or None")
        if normalized_task is not None and target_column is None:
            raise ValueError("task requires target_column")

        metric_frames = []
        run_rows = []
        for model in self.models:
            for model_seed in self.model_seeds:
                config = self._resolved_engine_config(model, model_seed)
                train_features = (
                    real_train
                    if target_column is None
                    else real_train.drop(columns=[target_column])
                )
                target = (
                    None
                    if target_column is None
                    else real_train[target_column]
                )
                started = time.perf_counter()
                engine = ConditionalGANEngine(**config)
                engine.fit(
                    train_features,
                    target=target,
                    target_name=target_column,
                    verbose=0,
                )
                fit_seconds = max(0.0, time.perf_counter() - started)
                if bool(config.get("match_critic_parameters", False)):
                    relative_gap = abs(engine.critic_parameter_gap_) / max(
                        1, engine.critic_parameter_target_
                    )
                    if relative_gap > self.parameter_tolerance:
                        raise RuntimeError(
                            "model %r missed critic parameter target by %.2f%%"
                            % (model.name, 100.0 * relative_gap)
                        )
                else:
                    relative_gap = 0.0

                for sample_seed in self.sample_seeds:
                    stages: Dict[str, pd.DataFrame] = {}
                    sample_started = time.perf_counter()
                    for stage, source in model.sampling_stages.items():
                        engine.set_sampling_seed(sample_seed)
                        values = engine.sample(
                            sample_rows,
                            return_target=target_column is not None,
                            weight_source=source,
                            weight_seed=sample_seed,
                        )
                        if target_column is None:
                            synthetic = values
                        else:
                            synthetic, synthetic_target = values
                            synthetic = synthetic.copy()
                            synthetic[target_column] = np.asarray(
                                synthetic_target
                            )
                        stages[stage] = synthetic.loc[
                            :, real_train.columns
                        ]
                    sample_seconds = max(
                        0.0, time.perf_counter() - sample_started
                    )
                    report = aggregate_table_report(
                        real_train,
                        real_test,
                        stages,
                        categorical_columns=categorical,
                        constraints=constraints,
                        target_column=(
                            target_column
                            if normalized_task is not None
                            else None
                        ),
                        task=normalized_task,
                        random_state=sample_seed,
                        c2st_folds=min(
                            self.c2st_folds, len(real_test), sample_rows
                        ),
                    )
                    report.insert(0, "sample_seed", sample_seed)
                    report.insert(0, "model_seed", model_seed)
                    report.insert(0, "split_seed", self.split_seed)
                    report.insert(0, "split_hash", split.split_hash)
                    report.insert(0, "factor", model.factor)
                    report.insert(0, "model_name", model.name)
                    metric_frames.append(report)
                    run_rows.append(
                        {
                            "model_name": model.name,
                            "factor": model.factor,
                            "split_seed": self.split_seed,
                            "model_seed": model_seed,
                            "sample_seed": sample_seed,
                            "split_hash": split.split_hash,
                            "fit_seconds": fit_seconds,
                            "sample_seconds": sample_seconds,
                            "critic_parameters": (
                                engine.critic_parameter_count_
                            ),
                            "critic_parameter_target": (
                                engine.critic_parameter_target_
                            ),
                            "critic_parameter_relative_gap": relative_gap,
                            "raw_batch_size": engine._fit_batch_size_,
                            "effective_critic_batch_size": (
                                engine.effective_critic_batch_size
                            ),
                            "config": json.dumps(
                                config, sort_keys=True, default=str
                            ),
                        }
                    )

        metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
        metrics.attrs["contains_blended_score"] = False
        metrics.attrs["selected_model"] = None
        result = AblationResult(
            metrics=metrics,
            runs=pd.DataFrame(run_rows),
            split=split.as_dict(),
        )
        if output_directory is not None:
            result.save(output_directory)
        return result


load_model_configs = load_ablation_matrix


__all__ = [
    "AblationBudget",
    "AblationModelConfig",
    "AblationResult",
    "AblationRunner",
    "DEFAULT_MODEL_DIRECTORY",
    "load_ablation_matrix",
    "load_model_configs",
    "validate_ablation_matrix",
]
