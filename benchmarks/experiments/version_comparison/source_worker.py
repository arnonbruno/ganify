"""Isolated subprocess worker for one explicit GANify source root.

This file deliberately imports only the standard library until ``main`` has
inserted and verified the requested source root.  Invoke it with ``python -I``
so an installed or parent-process GANify module cannot leak into the run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class WorkerVersionRefusal(RuntimeError):
    pass


class WorkerCapabilityFailure(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_source_root(root: Path) -> str:
    package = root / "ganify"
    if not (package / "__init__.py").is_file():
        raise FileNotFoundError(
            "source_root must contain ganify/__init__.py: %s" % root
        )
    files: List[Path] = []
    for path in package.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    for name in ("setup.py", "pyproject.toml", "setup.cfg"):
        candidate = root / name
        if candidate.is_file():
            files.append(candidate)
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _json_value(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_value(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_import(source_root: Path, expected_version: str) -> Tuple[Any, Dict[str, Any]]:
    if not (source_root / "ganify" / "__init__.py").is_file():
        raise FileNotFoundError(
            "source_root must contain ganify/__init__.py: %s" % source_root
        )
    for name in list(sys.modules):
        if name == "ganify" or name.startswith("ganify."):
            del sys.modules[name]
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    ganify = importlib.import_module("ganify")
    observed_version = str(getattr(ganify, "__version__", ""))
    module_path = Path(str(getattr(ganify, "__file__", ""))).resolve()
    package_root = (source_root / "ganify").resolve()
    if not _is_within(module_path, package_root):
        raise WorkerVersionRefusal(
            "module contamination: requested %s but imported ganify from %s"
            % (source_root, module_path)
        )
    if observed_version != str(expected_version):
        raise WorkerVersionRefusal(
            "GANify version refusal: expected %r from %s, imported %r from %s"
            % (str(expected_version), source_root, observed_version, module_path)
        )
    contaminated = []
    for name, module in list(sys.modules.items()):
        if name != "ganify" and not name.startswith("ganify."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and not _is_within(Path(module_file), package_root):
            contaminated.append({"module": name, "path": str(module_file)})
    if contaminated:
        raise WorkerVersionRefusal(
            "GANify submodule contamination detected: %s"
            % _canonical_json(contaminated)
        )
    return ganify, {
        "expected_version": str(expected_version),
        "observed_version": observed_version,
        "source_root": str(source_root),
        "imported_module": str(module_path),
        "source_sha256": _hash_source_root(source_root),
        "contaminated_modules": [],
    }


def _runtime_versions() -> Dict[str, str]:
    versions = {}
    for name in (
        "ganify",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tensorflow",
        "keras",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _normalize_adapter(name: str) -> str:
    value = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "v11": "v1.1_legacy_wgan",
        "v1_1": "v1.1_legacy_wgan",
        "v12_recipe": "v1.2_recipe_reimplementation",
        "v1_2_recipe": "v1.2_recipe_reimplementation",
        "v2_numeric": "v2_numeric_compatibility",
        "v2_compatibility": "v2_numeric_compatibility",
        "v2_recipe": "v2_numeric_recipe",
        "v2_conditional": "v2_conditional_native",
        "conditional_native": "v2_conditional_native",
    }
    return aliases.get(value, value)


def _adapter_metadata(adapter: str) -> Dict[str, Any]:
    values = {
        "v1.1_legacy_wgan": {
            "evidence_level": "source_run",
            "source_policy": "exact_v1.1_source",
            "implementation": "exact GANify 1.1 legacy fit_data(type='wgan')",
            "historical_version": "1.1.0",
        },
        "v1.2_recipe_reimplementation": {
            "evidence_level": "recipe_reimplementation",
            "source_policy": "current_compatibility_api",
            "implementation": (
                "v1.2 recipe reimplemented through the imported current "
                "Ganify.fit_data compatibility API"
            ),
            "historical_version": "1.2.0",
            "historical_source_available": False,
        },
        "v2_numeric_compatibility": {
            "evidence_level": "source_run",
            "source_policy": "exact_explicit_source",
            "implementation": "current numeric compatibility fit_data API",
        },
        "v2_numeric_recipe": {
            "evidence_level": "source_run",
            "source_policy": "exact_explicit_source",
            "implementation": "current numeric recipe fit_data API",
        },
        "v2_conditional_native": {
            "evidence_level": "source_run",
            "source_policy": "exact_explicit_source",
            "implementation": "current native conditional fit/sample API",
        },
    }
    if adapter not in values:
        if adapter == "v1.2_historical_artifact":
            raise WorkerCapabilityFailure(
                "v1.2_historical_artifact is artifact-level evidence and must "
                "be rescored; no v1.2 source checkout is claimed or imported"
            )
        raise WorkerCapabilityFailure("unknown source worker adapter %r" % adapter)
    return {"adapter": adapter, **values[adapter]}


def _derived_seed(seed: int, *parts: Any) -> int:
    payload = _canonical_json([int(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _canonical_scalar(value: Any) -> str:
    return _canonical_json(
        {"type": type(_json_value(value)).__name__, "value": _json_value(value)}
    )


def _allocate_class_counts(labels: Iterable[Any], n_rows: int) -> List[Dict[str, Any]]:
    import numpy as np
    import pandas as pd

    series = pd.Series(list(labels))
    if series.empty or bool(series.isna().any()):
        raise ValueError("class labels must be non-empty and non-missing")
    values: List[Any] = []
    frequencies: List[int] = []
    for value in series.tolist():
        token = _canonical_scalar(value)
        found = next(
            (
                index
                for index, existing in enumerate(values)
                if _canonical_scalar(existing) == token
            ),
            None,
        )
        if found is None:
            values.append(_json_value(value))
            frequencies.append(1)
        else:
            frequencies[found] += 1
    order = sorted(range(len(values)), key=lambda index: _canonical_scalar(values[index]))
    values = [values[index] for index in order]
    frequencies = [frequencies[index] for index in order]
    ideal = np.asarray(frequencies, dtype=float) * int(n_rows) / float(sum(frequencies))
    allocated = np.floor(ideal).astype(int)
    remainder = int(n_rows) - int(allocated.sum())
    ranking = sorted(
        range(len(values)),
        key=lambda index: (-(ideal[index] - allocated[index]), _canonical_scalar(values[index])),
    )
    for index in ranking[:remainder]:
        allocated[index] += 1
    result = [
        {"value": value, "count": int(count)}
        for value, count in zip(values, allocated.tolist())
        if count > 0
    ]
    if sum(item["count"] for item in result) != int(n_rows):
        raise RuntimeError("class count allocation failed")
    return result


def _check_keyword_arguments(function: Any, options: Mapping[str, Any], owner: str) -> None:
    signature = inspect.signature(function)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return
    unknown = sorted(set(options) - set(signature.parameters))
    if unknown:
        raise WorkerCapabilityFailure(
            "%s does not accept configured options: %s"
            % (owner, ", ".join(unknown))
        )


def _construct_model(
    ganify: Any,
    adapter: str,
    fit_seed: int,
    config: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    constructor = dict(config.get("constructor", {}))
    fit = dict(config.get("fit", {}))
    if "random_state" in constructor:
        raise ValueError("constructor.random_state is supplied by fit_seed")
    constructor["random_state"] = int(fit_seed)

    parameters = inspect.signature(ganify.Ganify.__init__).parameters
    recipe = adapter in {
        "v1.2_recipe_reimplementation",
        "v2_numeric_recipe",
    }
    if recipe:
        defaults = {
            "scaler": "copula",
            "ema_decay": 0.999,
            "ema_warmup_epochs": max(0, int(fit.get("epochs", 1)) // 2),
            "calibrate_marginals": True,
            "round_integers": True,
        }
        for name, value in defaults.items():
            if name in parameters:
                constructor.setdefault(name, value)
    _check_keyword_arguments(ganify.Ganify, constructor, "Ganify constructor")
    model = ganify.Ganify(**constructor)
    return model, constructor, fit


def _ensure_numeric(frame: Any, name: str) -> None:
    import numpy as np
    import pandas as pd

    non_numeric = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if non_numeric:
        raise WorkerCapabilityFailure(
            "%s requires finite numeric columns; unsupported columns: %r"
            % (name, non_numeric)
        )
    if not bool(np.isfinite(frame.to_numpy(dtype=float)).all()):
        raise WorkerCapabilityFailure(
            "%s requires finite numeric values without missing data" % name
        )


def _fit_numeric_models(
    ganify: Any,
    adapter: str,
    lane: str,
    train: Any,
    target: Optional[str],
    fit_seed: int,
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import numpy as np

    if lane == "numeric_regression":
        _ensure_numeric(train, adapter)
        model, constructor, fit = _construct_model(
            ganify, adapter, fit_seed, config
        )
        fit["type"] = "wgan"
        fit.setdefault("verbose", 0)
        _check_keyword_arguments(model.fit_data, fit, "Ganify.fit_data")
        labels = np.zeros(len(train), dtype=float)
        model.fit_data(train, labels, **fit)
        return (
            [
                {
                    "class_value": None,
                    "model": model,
                    "columns": list(train.columns),
                    "rows": len(train),
                }
            ],
            {"constructor": constructor, "fit": fit},
        )

    if target is None or target not in train.columns:
        raise ValueError("multiclass_classwise needs a target column")
    features = train.drop(columns=[target])
    _ensure_numeric(features, adapter)
    class_counts = _allocate_class_counts(train[target], len(train))
    models: List[Dict[str, Any]] = []
    resolved: List[Dict[str, Any]] = []
    for class_entry in class_counts:
        value = class_entry["value"]
        mask = train[target].map(
            lambda item: _canonical_scalar(item) == _canonical_scalar(value)
        )
        class_features = features.loc[mask].reset_index(drop=True)
        if len(class_features) < 2:
            raise WorkerCapabilityFailure(
                "%s needs at least two training rows for class %r"
                % (adapter, value)
            )
        class_seed = _derived_seed(fit_seed, "fit-class", value)
        model, constructor, fit = _construct_model(
            ganify, adapter, class_seed, config
        )
        fit["type"] = "wgan"
        fit.setdefault("verbose", 0)
        _check_keyword_arguments(model.fit_data, fit, "Ganify.fit_data")
        labels = np.repeat(np.asarray([value]), len(class_features))
        model.fit_data(class_features, labels, **fit)
        models.append(
            {
                "class_value": value,
                "model": model,
                "columns": list(class_features.columns),
                "rows": len(class_features),
            }
        )
        resolved.append(
            {"class_value": value, "constructor": constructor, "fit": fit}
        )
    return models, {"class_models": resolved}


def _as_numpy(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _set_sample_seed(model: Any, seed: int) -> None:
    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if hasattr(model, "_rng"):
        model._rng = np.random.default_rng(int(seed))
    engine = getattr(model, "_conditional_engine", None)
    if engine is not None and hasattr(engine, "set_sampling_seed"):
        engine.set_sampling_seed(int(seed))


def _sample_numeric_model(
    model: Any, count: int, seed: int, columns: Sequence[Any]
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    _set_sample_seed(model, seed)
    required = ("_noise", "adversary_one", "scaler")
    if all(hasattr(model, name) for name in required):
        latent = model._noise(int(count))
        generator = (
            model._ema_generator
            if getattr(model, "_ema_generator", None) is not None
            else model.adversary_one
        )
        fake = _as_numpy(generator(latent, training=False))
        raw_values = model.scaler.inverse_transform(fake)
        stages: Dict[str, Any] = {
            "raw": pd.DataFrame(raw_values, columns=list(columns))
        }
        if bool(getattr(model, "calibrate_marginals", False)):
            calibrated_values = model._calibrate_marginals(fake)
            stages["calibrated"] = pd.DataFrame(
                calibrated_values, columns=list(columns)
            )
        else:
            calibrated_values = raw_values
        has_projection = bool(getattr(model, "round_integers", False)) or bool(
            getattr(model, "_pair_idx", [])
        )
        if has_projection and hasattr(model, "_postprocess"):
            stages["projected"] = pd.DataFrame(
                model._postprocess(calibrated_values), columns=list(columns)
            )
        return stages

    # A compatibility fallback is useful for small fake packages in protocol
    # tests, while real GANify sources use the explicit pre/post-stage path.
    try:
        sampled = model.create_bulk(length=int(count), output=1)
    except TypeError:
        try:
            sampled = model.create_bulk(int(count), output=1)
        except (TypeError, ValueError):
            sampled = model.create_bulk(int(count))
    frame = sampled if isinstance(sampled, pd.DataFrame) else pd.DataFrame(
        sampled, columns=list(columns)
    )
    return {"raw": frame.reset_index(drop=True)}


def _combine_classwise(
    pieces: Mapping[str, List[Any]],
    target: str,
    sample_seed: int,
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    result: Dict[str, Any] = {}
    for stage, frames in pieces.items():
        combined = pd.concat(frames, ignore_index=True)
        order = np.random.default_rng(
            _derived_seed(sample_seed, "classwise-shuffle", stage)
        ).permutation(len(combined))
        combined = combined.iloc[order].reset_index(drop=True)
        result[stage] = combined
    return result


def _sample_numeric_models(
    models: Sequence[Mapping[str, Any]],
    lane: str,
    target: Optional[str],
    train: Any,
    n_rows: int,
    sample_seed: int,
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    if lane == "numeric_regression":
        stages = _sample_numeric_model(
            models[0]["model"],
            int(n_rows),
            int(sample_seed),
            models[0]["columns"],
        )
        return stages, None

    if target is None:
        raise RuntimeError("classwise target unexpectedly absent")
    allocation = _allocate_class_counts(train[target], int(n_rows))
    by_token = {
        _canonical_scalar(entry["class_value"]): entry for entry in models
    }
    pieces: Dict[str, List[Any]] = {}
    for class_entry in allocation:
        value = class_entry["value"]
        count = int(class_entry["count"])
        model_entry = by_token[_canonical_scalar(value)]
        class_seed = _derived_seed(sample_seed, "sample-class", value)
        class_stages = _sample_numeric_model(
            model_entry["model"],
            count,
            class_seed,
            model_entry["columns"],
        )
        for stage, frame in class_stages.items():
            output = frame.copy()
            output[target] = value
            pieces.setdefault(stage, []).append(output)
    return _combine_classwise(pieces, target, sample_seed), allocation


def _fit_conditional(
    ganify: Any,
    lane: str,
    train: Any,
    target: Optional[str],
    fit_seed: int,
    config: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    constructor = dict(config.get("constructor", {}))
    fit = dict(config.get("fit", {}))
    if "constraints" in fit:
        raise WorkerCapabilityFailure(
            "v2_conditional_native constraints must use the common projected "
            "stage so raw and projected evidence remain separate"
        )
    if "random_state" in constructor:
        raise ValueError("constructor.random_state is supplied by fit_seed")
    constructor["random_state"] = int(fit_seed)
    _check_keyword_arguments(ganify.Ganify, constructor, "Ganify constructor")
    model = ganify.Ganify(**constructor)
    fit.setdefault("epochs", 1)
    fit.setdefault("verbose", 0)
    fit["conditional"] = True
    _check_keyword_arguments(model.fit, fit, "Ganify.fit")
    if lane == "multiclass_classwise":
        if target is None or target not in train.columns:
            raise ValueError("multiclass_classwise needs a target column")
        features = train.drop(columns=[target])
        model.fit(features, train[target], target_name=target, **fit)
    else:
        # Regression is synthesized as one numeric table.  A continuous target
        # is not falsely converted into thousands of target classes.
        _ensure_numeric(train, "v2_conditional_native numeric_regression")
        model.fit(train, None, **fit)
    return model, {"constructor": constructor, "fit": fit}


def _sample_conditional(
    model: Any,
    lane: str,
    train: Any,
    target: Optional[str],
    n_rows: int,
    sample_seed: int,
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    import numpy as np
    import pandas as pd

    if lane == "numeric_regression":
        _set_sample_seed(model, sample_seed)
        sampled = model.sample(int(n_rows), output="dataframe")
        frame = sampled if isinstance(sampled, pd.DataFrame) else pd.DataFrame(
            sampled, columns=list(train.columns)
        )
        return {"raw": frame.reset_index(drop=True)}, None

    if target is None:
        raise RuntimeError("classwise target unexpectedly absent")
    allocation = _allocate_class_counts(train[target], int(n_rows))
    frames = []
    for class_entry in allocation:
        value = class_entry["value"]
        count = int(class_entry["count"])
        class_seed = _derived_seed(sample_seed, "sample-class", value)
        _set_sample_seed(model, class_seed)
        sampled = model.sample(
            count,
            conditions={target: value},
            return_target=True,
            output="dataframe",
        )
        features, labels = sampled
        frame = features.copy()
        frame[target] = list(labels)
        observed = _allocate_class_counts(frame[target], len(frame))
        if len(observed) != 1 or _canonical_scalar(observed[0]["value"]) != _canonical_scalar(value):
            raise RuntimeError(
                "conditional sampler did not preserve requested class %r" % value
            )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    order = np.random.default_rng(
        _derived_seed(sample_seed, "classwise-shuffle", "raw")
    ).permutation(len(combined))
    return {"raw": combined.iloc[order].reset_index(drop=True)}, allocation


def _empirical_quantile_calibration(
    raw: Any, train: Any, excluded: Sequence[Any]
) -> Any:
    import numpy as np
    import pandas as pd

    calibrated = raw.copy(deep=True)
    excluded_set = set(excluded)
    for column in calibrated.columns:
        if column in excluded_set:
            continue
        if not pd.api.types.is_numeric_dtype(train[column].dtype):
            raise WorkerCapabilityFailure(
                "empirical_quantile calibration only supports numeric columns; "
                "column %r is not numeric" % column
            )
        source = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
        generated = pd.to_numeric(
            calibrated[column], errors="coerce"
        ).to_numpy(dtype=float)
        if not bool(np.isfinite(source).all() and np.isfinite(generated).all()):
            raise WorkerCapabilityFailure(
                "empirical_quantile calibration needs finite values in %r" % column
            )
        order = np.argsort(generated, kind="mergesort")
        probabilities = (np.arange(len(generated), dtype=float) + 0.5) / len(generated)
        mapped = np.quantile(source, probabilities)
        values = np.empty(len(generated), dtype=float)
        values[order] = mapped
        calibrated[column] = values
    return calibrated


def _project_frame(
    source: Any,
    train: Any,
    config: Mapping[str, Any],
    excluded: Sequence[Any],
) -> Any:
    import numpy as np
    import pandas as pd

    projected = source.copy(deep=True)
    changed = False
    excluded_set = set(excluded)
    if bool(config.get("clip_to_train_range", False)):
        for column in projected.columns:
            if column in excluded_set:
                continue
            if pd.api.types.is_numeric_dtype(train[column].dtype):
                projected[column] = pd.to_numeric(
                    projected[column], errors="coerce"
                ).clip(train[column].min(), train[column].max())
                changed = True
    if bool(config.get("round_integers", False)):
        for column in projected.columns:
            if column in excluded_set:
                continue
            real = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
            if bool(np.isfinite(real).all()) and bool(
                np.all(np.abs(real - np.round(real)) <= 1e-9)
            ):
                projected[column] = np.round(
                    pd.to_numeric(projected[column], errors="coerce")
                )
                changed = True
    for constraint in config.get("constraints", []):
        kind = str(constraint.get("type", "")).lower()
        if kind == "range":
            column = constraint["column"]
            lower = constraint.get("minimum", constraint.get("min"))
            upper = constraint.get("maximum", constraint.get("max"))
            values = pd.to_numeric(projected[column], errors="coerce")
            projected[column] = values.clip(lower=lower, upper=upper)
            changed = True
        elif kind == "inequality":
            left = constraint["left"]
            right = constraint["right"]
            operation = str(constraint.get("operator", "<="))
            left_values = pd.to_numeric(projected[left], errors="coerce")
            right_values = (
                pd.to_numeric(projected[right], errors="coerce")
                if right in projected.columns
                else float(right)
            )
            if operation in {"<=", "<"}:
                projected[left] = np.minimum(left_values, right_values)
            elif operation in {">=", ">"}:
                projected[left] = np.maximum(left_values, right_values)
            else:
                raise WorkerCapabilityFailure(
                    "common projection does not support inequality operator %r"
                    % operation
                )
            changed = True
        elif kind in {"allowed", "allowed_values", "domain"}:
            column = constraint["column"]
            allowed = list(constraint["values"])
            if not allowed:
                raise ValueError("allowed-values projection needs values")
            valid = projected[column].isin(allowed)
            projected.loc[~valid, column] = allowed[0]
            changed = True
        else:
            raise WorkerCapabilityFailure(
                "common projection does not support constraint type %r" % kind
            )
    if not changed:
        raise WorkerCapabilityFailure(
            "projected stage requested without intrinsic projection or a "
            "clip/round/constraint projection specification"
        )
    return projected


def _resolve_stage_spec(raw: Any) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        return {"raw": {}}
    if isinstance(raw, list):
        return {str(name): {} for name in raw}
    if isinstance(raw, Mapping):
        return {
            str(name): ({} if value is None else dict(value))
            for name, value in raw.items()
        }
    raise TypeError("stages must be a list or mapping")


def _resolve_stages(
    intrinsic: Mapping[str, Any],
    train: Any,
    stage_spec: Mapping[str, Mapping[str, Any]],
    target: Optional[str],
) -> Dict[str, Any]:
    if "raw" not in stage_spec:
        raise ValueError("stage protocol must explicitly retain raw")
    resolved: Dict[str, Any] = {}
    excluded = [] if target is None else [target]
    for stage, config in stage_spec.items():
        if stage in intrinsic:
            resolved[stage] = intrinsic[stage].copy(deep=True)
            continue
        if stage == "calibrated":
            method = str(config.get("method", "")).lower()
            if method not in {"empirical_quantile", "quantile"}:
                raise WorkerCapabilityFailure(
                    "calibrated stage is unavailable intrinsically; configure "
                    "method='empirical_quantile' instead of relabeling raw output"
                )
            source_name = str(config.get("source", "raw"))
            source = resolved.get(source_name, intrinsic.get(source_name))
            if source is None:
                raise ValueError("calibrated source stage %r is unavailable" % source_name)
            resolved[stage] = _empirical_quantile_calibration(
                source, train, excluded
            )
            continue
        if stage == "projected":
            source_name = str(
                config.get(
                    "source",
                    "calibrated"
                    if "calibrated" in resolved or "calibrated" in intrinsic
                    else "raw",
                )
            )
            source = resolved.get(source_name, intrinsic.get(source_name))
            if source is None:
                raise ValueError("projected source stage %r is unavailable" % source_name)
            resolved[stage] = _project_frame(source, train, config, excluded)
            continue
        raise WorkerCapabilityFailure("unsupported generation stage %r" % stage)
    return resolved


def _hash_frame(frame: Any) -> str:
    metadata = {
        "columns": [repr(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_name": repr(frame.index.name),
    }
    payload = frame.to_csv(
        index=True,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="__GANIFY_VERSION_COMPARISON_NA__",
    )
    return _hash_bytes((_canonical_json(metadata) + "\n" + payload).encode("utf-8"))


def _run(request: Mapping[str, Any], result_path: Path) -> Dict[str, Any]:
    source_root = Path(str(request["source_root"])).expanduser().resolve()
    expected_version = str(request["expected_version"])
    ganify, provenance = _source_import(source_root, expected_version)
    action = str(request.get("action", "fit_sample"))
    result: Dict[str, Any] = {
        "status": "completed",
        "action": action,
        "provenance": provenance,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "worker_pid": os.getpid(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "package_versions": _runtime_versions(),
        "determinism_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "TF_DETERMINISTIC_OPS",
                "TF_CUDNN_DETERMINISTIC",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    if action == "probe":
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        return result
    if action != "fit_sample":
        raise ValueError("unsupported worker action %r" % action)

    import numpy as np
    import pandas as pd

    adapter = _normalize_adapter(str(request["adapter"]))
    adapter_info = _adapter_metadata(adapter)
    if adapter == "v1.1_legacy_wgan" and provenance["observed_version"] != "1.1.0":
        raise WorkerVersionRefusal(
            "v1.1_legacy_wgan requires exact imported version '1.1.0', got %r"
            % provenance["observed_version"]
        )
    if adapter in {
        "v1.2_recipe_reimplementation",
        "v2_numeric_compatibility",
        "v2_numeric_recipe",
        "v2_conditional_native",
    } and not provenance["observed_version"].startswith("2."):
        raise WorkerVersionRefusal(
            "%s requires an imported current 2.x compatibility API, got %r"
            % (adapter, provenance["observed_version"])
        )
    lane = str(request["lane"])
    if lane not in {"numeric_regression", "multiclass_classwise"}:
        raise WorkerCapabilityFailure(
            "%s does not support lane %r" % (adapter, lane)
        )
    train_path = Path(str(request["train_path"])).resolve()
    train = pd.read_csv(train_path)
    target = request.get("target")
    if target is not None:
        target = str(target)
    fit_seed = int(request["fit_seed"])
    sample_seeds = [int(seed) for seed in request["sample_seeds"]]
    n_rows = int(request["sample_rows"])
    if n_rows < 1:
        raise ValueError("sample_rows must be positive")
    config = dict(request.get("config", {}))
    stage_spec = _resolve_stage_spec(request.get("stages"))
    output_dir = Path(str(request["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_started = time.perf_counter()
    try:
        if adapter == "v2_conditional_native":
            model, resolved_config = _fit_conditional(
                ganify, lane, train, target, fit_seed, config
            )
            fitted = model
        else:
            fitted, resolved_config = _fit_numeric_models(
                ganify, adapter, lane, train, target, fit_seed, config
            )
    except Exception as error:
        error.worker_phase = "fit"
        error.partial_fit_seconds = max(
            0.0, time.perf_counter() - fit_started
        )
        raise
    fit_seconds = max(0.0, time.perf_counter() - fit_started)

    sample_records = []
    for sample_seed in sample_seeds:
        sample_started = time.perf_counter()
        try:
            if adapter == "v2_conditional_native":
                intrinsic, allocation = _sample_conditional(
                    fitted, lane, train, target, n_rows, sample_seed
                )
            else:
                intrinsic, allocation = _sample_numeric_models(
                    fitted, lane, target, train, n_rows, sample_seed
                )
            stages = _resolve_stages(
                intrinsic,
                train,
                stage_spec,
                target if lane == "multiclass_classwise" else None,
            )
            stage_records = []
            for stage, frame in stages.items():
                if len(frame) != n_rows:
                    raise RuntimeError(
                        "stage %r returned %d rows, expected %d"
                        % (stage, len(frame), n_rows)
                    )
                destination = output_dir / (
                    "sample_%s_%s.csv" % (sample_seed, stage)
                )
                frame.to_csv(destination, index=False, float_format="%.17g")
                stage_records.append(
                    {
                        "stage": stage,
                        "path": str(destination),
                        "sha256": _hash_file(destination),
                        "dataframe_sha256": _hash_frame(frame),
                        "rows": int(len(frame)),
                        "columns": [str(column) for column in frame.columns],
                    }
                )
        except Exception as error:
            error.worker_phase = "sample"
            error.failed_sample_seed = int(sample_seed)
            error.partial_sample_seconds = max(
                0.0, time.perf_counter() - sample_started
            )
            raise
        sample_records.append(
            {
                "sample_seed": sample_seed,
                "sample_seconds": max(
                    0.0, time.perf_counter() - sample_started
                ),
                "class_counts": allocation,
                "stages": stage_records,
            }
        )

    result.update(
        {
            "adapter": adapter_info,
            "lane": lane,
            "target": target,
            "fit_seed": fit_seed,
            "training_seed": int(request.get("training_seed", fit_seed)),
            "model_seed": int(request.get("model_seed", fit_seed)),
            "sample_seeds": sample_seeds,
            "seeds": {
                "training": int(request.get("training_seed", fit_seed)),
                "model": int(request.get("model_seed", fit_seed)),
                "sample": sample_seeds,
            },
            "sample_rows": n_rows,
            "resolved_config": _json_value(resolved_config),
            "stage_protocol": _json_value(stage_spec),
            "train_path": str(train_path),
            "train_sha256": _hash_file(train_path),
            "train_dataframe_sha256": _hash_frame(train),
            "timings_seconds": {
                "fit": fit_seconds,
                "sampling_total": float(
                    sum(record["sample_seconds"] for record in sample_records)
                ),
            },
            "samples": sample_records,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    arguments = parser.parse_args(argv)
    request_path = Path(arguments.request).resolve()
    result_path = Path(arguments.result).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    phase = "source_import"
    started = time.perf_counter()
    try:
        phase = "fit_sample"
        result = _run(request, result_path)
        result["worker_seconds"] = max(0.0, time.perf_counter() - started)
        _atomic_json(result_path, result)
        return 0
    except Exception as error:
        if isinstance(error, WorkerVersionRefusal):
            failure_kind = "version_refusal"
            failure_phase = "source_import"
        elif isinstance(error, WorkerCapabilityFailure):
            failure_kind = "capability"
            failure_phase = getattr(error, "worker_phase", "capability")
        else:
            failure_kind = "error"
            failure_phase = getattr(error, "worker_phase", phase)
        source_root = request.get("source_root")
        try:
            source_sha256 = _hash_source_root(
                Path(str(source_root)).expanduser().resolve()
            )
        except Exception:
            source_sha256 = None
        failure = {
            "status": "failed",
            "action": request.get("action", "fit_sample"),
            "failure_phase": failure_phase,
            "failure_kind": failure_kind,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "failed_sample_seed": getattr(error, "failed_sample_seed", None),
            "partial_fit_seconds": getattr(error, "partial_fit_seconds", None),
            "partial_sample_seconds": getattr(
                error, "partial_sample_seconds", None
            ),
            "worker_seconds": max(0.0, time.perf_counter() - started),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": source_root,
            "source_sha256": source_sha256,
            "expected_version": request.get("expected_version"),
            "adapter": request.get("adapter"),
        }
        _atomic_json(result_path, failure)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

