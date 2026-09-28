"""Launch and validate isolated source-root worker subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .core import (
    CapabilityError,
    VersionComparisonError,
    VersionRefusalError,
    atomic_write_json,
    hash_file,
    hash_source_root,
    load_json,
)


WORKER_PATH = Path(__file__).with_name("source_worker.py").resolve()


def _worker_environment(seed: int) -> Dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(
        {
            "PYTHONHASHSEED": str(int(seed)),
            "TF_DETERMINISTIC_OPS": "1",
            "TF_CUDNN_DETERMINISTIC": "1",
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "TF_CPP_MIN_LOG_LEVEL": environment.get("TF_CPP_MIN_LOG_LEVEL", "2"),
        }
    )
    return environment


def invoke_source_worker(
    request: Mapping[str, Any],
    *,
    work_directory: os.PathLike,
    timeout_seconds: Optional[float] = None,
    raise_on_failure: bool = False,
) -> Dict[str, Any]:
    """Run one request with ``python -I`` and retain stdout/stderr verbatim."""

    directory = Path(work_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "worker_request.json"
    result_path = directory / "worker_result.json"
    atomic_write_json(request_path, dict(request))
    fit_seed = int(request.get("fit_seed", request.get("seed", 0)))
    command = [
        sys.executable,
        "-I",
        str(WORKER_PATH),
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(directory),
            env=_worker_environment(fit_seed),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        result = {
            "status": "failed",
            "failure_phase": "subprocess",
            "failure_kind": "timeout",
            "error_type": type(error).__name__,
            "error_message": "source worker exceeded %.3f seconds"
            % float(timeout_seconds),
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "worker_command": command,
            "worker_sha256": hash_file(WORKER_PATH),
        }
        atomic_write_json(result_path, result)
    else:
        if result_path.is_file():
            result = dict(load_json(result_path))
        else:
            result = {
                "status": "failed",
                "failure_phase": "subprocess",
                "failure_kind": "worker_crash",
                "error_type": "WorkerProcessError",
                "error_message": (
                    "source worker exited %d without a result file"
                    % completed.returncode
                ),
            }
        result.update(
            {
                "returncode": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "worker_command": command,
                "worker_sha256": hash_file(WORKER_PATH),
            }
        )
        if result.get("status") == "completed" and completed.returncode != 0:
            result.update(
                {
                    "status": "failed",
                    "failure_phase": "subprocess",
                    "failure_kind": "worker_exit",
                    "error_type": "WorkerProcessError",
                    "error_message": (
                        "worker reported completion but exited %d"
                        % completed.returncode
                    ),
                }
            )
        atomic_write_json(result_path, result)

    if result.get("status") == "completed":
        expected_hash = hash_source_root(str(request["source_root"]))
        observed_hash = (
            result.get("provenance", {}).get("source_sha256")
            if isinstance(result.get("provenance"), Mapping)
            else None
        )
        if observed_hash != expected_hash:
            result.update(
                {
                    "status": "failed",
                    "failure_phase": "provenance",
                    "failure_kind": "source_changed",
                    "error_type": "SourceHashMismatch",
                    "error_message": (
                        "source changed across worker boundary: expected %s, "
                        "worker imported %s" % (expected_hash, observed_hash)
                    ),
                }
            )
            atomic_write_json(result_path, result)

    if raise_on_failure and result.get("status") != "completed":
        message = str(result.get("error_message", "source worker failed"))
        kind = result.get("failure_kind")
        if kind == "version_refusal":
            raise VersionRefusalError(message)
        if kind == "capability":
            raise CapabilityError(message)
        raise VersionComparisonError(message)
    return result


def probe_source(
    source_root: os.PathLike,
    expected_version: str,
    *,
    work_directory: Optional[os.PathLike] = None,
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    """Import one source in isolation and refuse any exact-version mismatch."""

    request = {
        "action": "probe",
        "source_root": str(Path(source_root).expanduser().resolve()),
        "expected_version": str(expected_version),
        "seed": 0,
    }
    if work_directory is not None:
        return invoke_source_worker(
            request,
            work_directory=work_directory,
            timeout_seconds=timeout_seconds,
            raise_on_failure=True,
        )
    with tempfile.TemporaryDirectory(prefix="ganify-version-probe-") as temporary:
        return invoke_source_worker(
            request,
            work_directory=temporary,
            timeout_seconds=timeout_seconds,
            raise_on_failure=True,
        )


__all__ = ["WORKER_PATH", "invoke_source_worker", "probe_source"]

