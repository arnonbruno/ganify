"""Artifact-level rescoring of the preserved KuaiRand experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .core import HistoricalArtifactError, atomic_write_json, hash_file
from .evaluator import EvaluationResult, evaluate_stages


KUAI_RAND_SPLIT_SEED = 42
KUAI_RAND_TEST_SIZE = 0.2
EXPECTED_REAL_ROWS = {"video": 7583, "clicks": 6000}


@dataclass(frozen=True)
class HistoricalArtifact:
    path: str
    table: str
    label: str
    historical_version: str
    evidence_level: str
    source_available: bool
    sha256: str
    metadata_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HistoricalRescoreResult:
    metrics: pd.DataFrame
    privacy: pd.DataFrame
    reliability: pd.DataFrame
    artifacts: List[HistoricalArtifact]
    split_manifests: Dict[str, Dict[str, Any]]

    def save(self, directory: Path) -> Dict[str, str]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        paths = {
            "metrics": destination / "historical_metrics.csv",
            "privacy": destination / "historical_privacy.csv",
            "reliability": destination / "historical_reliability.csv",
            "manifest": destination / "historical_manifest.json",
        }
        self.metrics.to_csv(paths["metrics"], index=False)
        self.privacy.to_csv(paths["privacy"], index=False)
        self.reliability.to_csv(paths["reliability"], index=False)
        atomic_write_json(
            paths["manifest"],
            {
                "artifacts": [artifact.as_dict() for artifact in self.artifacts],
                "splits": self.split_manifests,
                "split_seed": KUAI_RAND_SPLIT_SEED,
                "test_size": KUAI_RAND_TEST_SIZE,
                "v1.2_evidence_level": "artifact_level",
                "v1.2_source_available": False,
            },
        )
        return {name: str(path) for name, path in paths.items()}


def original_seed42_split(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Reproduce the original positional sklearn seed-42 80/20 split."""

    if frame.empty:
        raise ValueError("KuaiRand real table must not be empty")
    row_ids = np.arange(len(frame), dtype=np.int64)
    train_ids, test_ids = train_test_split(
        row_ids,
        test_size=KUAI_RAND_TEST_SIZE,
        random_state=KUAI_RAND_SPLIT_SEED,
        shuffle=True,
    )
    train = frame.iloc[train_ids].reset_index(drop=True)
    test = frame.iloc[test_ids].reset_index(drop=True)
    manifest = {
        "method": "sklearn.model_selection.train_test_split",
        "seed": KUAI_RAND_SPLIT_SEED,
        "test_size": KUAI_RAND_TEST_SIZE,
        "row_id_kind": "original_zero_based_position",
        "train_row_ids": [int(value) for value in train_ids],
        "test_row_ids": [int(value) for value in test_ids],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    return train, test, manifest


def _table_from_name(path: Path) -> Optional[str]:
    name = path.stem.lower()
    if "click" in name:
        return "clicks"
    if "video" in name:
        return "video"
    return None


def _metadata_for_sample(path: Path) -> Optional[Path]:
    stem = path.stem
    candidates = []
    if stem.endswith("_synthetic"):
        candidates.append(path.parent / (stem[: -len("_synthetic")] + "_model") / "metadata.json")
    if stem.startswith("final_"):
        table = "video" if "video" in stem else "clicks"
        candidates.append(path.parent / ("final_%s_model" % table) / "metadata.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _historical_version(path: Path, root: Path, default: str) -> Tuple[str, Optional[Path]]:
    metadata_path = _metadata_for_sample(path)
    if metadata_path is not None:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            version = str(metadata.get("version", default))
            return version, metadata_path
        except (OSError, ValueError, TypeError):
            pass
    return default, metadata_path


def _discover_root(
    root: Path,
    *,
    default_version: str,
    implementation_label: str,
) -> List[HistoricalArtifact]:
    if not root.is_dir():
        raise HistoricalArtifactError(
            "historical KuaiRand artifact directory is absent: %s" % root
        )
    artifacts = []
    for path in sorted(root.glob("*.csv")):
        table = _table_from_name(path)
        if table is None:
            continue
        version, metadata_path = _historical_version(
            path, root, default_version
        )
        label = implementation_label
        if version.startswith("1.1"):
            label = "v1.1_historical_artifact"
        elif version.startswith("1.2"):
            label = "v1.2_historical_artifact"
        artifacts.append(
            HistoricalArtifact(
                path=str(path.resolve()),
                table=table,
                label=label,
                historical_version=version,
                evidence_level="artifact_level",
                source_available=False,
                sha256=hash_file(path),
                metadata_path=(
                    None
                    if metadata_path is None
                    else str(metadata_path.resolve())
                ),
            )
        )
    if not artifacts:
        raise HistoricalArtifactError(
            "no KuaiRand synthetic CSV artifacts were found in %s" % root
        )
    return artifacts


def discover_kuairand_artifacts(
    v11_root: Path,
    v12_root: Path,
) -> List[HistoricalArtifact]:
    """Discover both preserved artifact families without claiming v1.2 source."""

    first = _discover_root(
        Path(v11_root).resolve(),
        default_version="1.1.0",
        implementation_label="v1.1_historical_artifact",
    )
    second = _discover_root(
        Path(v12_root).resolve(),
        default_version="1.2.0",
        implementation_label="v1.2_historical_artifact",
    )
    return first + second


def _load_real_table(path: Path, table: str) -> pd.DataFrame:
    if not path.is_file():
        raise HistoricalArtifactError(
            "real KuaiRand %s table required for artifact rescoring is absent: %s"
            % (table, path)
        )
    frame = pd.read_csv(path)
    expected = EXPECTED_REAL_ROWS[table]
    if len(frame) != expected:
        raise HistoricalArtifactError(
            "historical %s rescoring requires the original curated %d-row "
            "table before the seed-42 split; %s has %d rows"
            % (table, expected, path, len(frame))
        )
    return frame


def rescore_historical_kuairand(
    *,
    v11_artifact_root: Path,
    v12_artifact_root: Path,
    real_video_path: Path,
    real_clicks_path: Path,
    output_directory: Optional[Path] = None,
    constraints: Optional[Mapping[str, Iterable[Mapping[str, Any]]]] = None,
    privacy: Optional[Mapping[str, Any]] = None,
    include_controls: bool = False,
) -> HistoricalRescoreResult:
    """Rescore persisted samples with current metrics on the original split.

    The v1.2 rows are always labeled ``artifact_level`` and
    ``source_available=False``.  This function never instantiates a v1.2 model.
    """

    artifacts = discover_kuairand_artifacts(
        Path(v11_artifact_root), Path(v12_artifact_root)
    )
    real = {
        "video": _load_real_table(Path(real_video_path), "video"),
        "clicks": _load_real_table(Path(real_clicks_path), "clicks"),
    }
    splits = {}
    split_manifests = {}
    for table, frame in real.items():
        train, test, manifest = original_seed42_split(frame)
        splits[table] = (train, test)
        split_manifests[table] = manifest

    metric_frames = []
    privacy_frames = []
    reliability_frames = []
    controls_remaining = bool(include_controls)
    for artifact in artifacts:
        synthetic = pd.read_csv(artifact.path)
        train_full, test_full = splits[artifact.table]
        missing = [
            column for column in synthetic.columns if column not in train_full.columns
        ]
        if missing:
            raise HistoricalArtifactError(
                "artifact %s has columns absent from the original %s table: %r"
                % (artifact.path, artifact.table, missing)
            )
        train = train_full.loc[:, synthetic.columns].copy()
        test = test_full.loc[:, synthetic.columns].copy()
        candidate = "%s:%s" % (artifact.label, Path(artifact.path).stem)
        result = evaluate_stages(
            train,
            test,
            {"raw": synthetic},
            candidate=candidate,
            constraints=(
                None
                if constraints is None
                else constraints.get(artifact.table)
            ),
            random_state=KUAI_RAND_SPLIT_SEED,
            privacy=privacy,
            include_controls=controls_remaining,
        )
        controls_remaining = False
        provenance = {
            "artifact_path": artifact.path,
            "artifact_sha256": artifact.sha256,
            "historical_version": artifact.historical_version,
            "evidence_level": artifact.evidence_level,
            "source_available": artifact.source_available,
            "adapter": artifact.label,
            "table": artifact.table,
            "split_seed": KUAI_RAND_SPLIT_SEED,
        }
        for frame in (result.metrics, result.privacy, result.reliability):
            for key, value in provenance.items():
                frame[key] = value
        metric_frames.append(result.metrics)
        privacy_frames.append(result.privacy)
        reliability_frames.append(result.reliability)

    output = HistoricalRescoreResult(
        metrics=pd.concat(metric_frames, ignore_index=True, sort=False),
        privacy=pd.concat(privacy_frames, ignore_index=True, sort=False),
        reliability=pd.concat(reliability_frames, ignore_index=True, sort=False),
        artifacts=artifacts,
        split_manifests=split_manifests,
    )
    if output_directory is not None:
        output.save(Path(output_directory))
    return output


__all__ = [
    "EXPECTED_REAL_ROWS",
    "HistoricalArtifact",
    "HistoricalRescoreResult",
    "KUAI_RAND_SPLIT_SEED",
    "KUAI_RAND_TEST_SIZE",
    "discover_kuairand_artifacts",
    "original_seed42_split",
    "rescore_historical_kuairand",
]

