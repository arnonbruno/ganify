"""Command-line entry point for a JSON version-comparison protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .runner import run_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible, source-isolated GANify comparisons."
    )
    parser.add_argument("protocol", nargs="?", help="JSON protocol path")
    parser.add_argument(
        "--protocol",
        dest="protocol_option",
        help="JSON protocol path (alternative to positional argument)",
    )
    parser.add_argument("--output", help="Override protocol output_dir")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed cache cells and execute new attempts",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Explicitly retry cached failures (never enabled implicitly)",
    )
    parser.add_argument("--timeout", type=float, help="Worker timeout in seconds")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code zero while retaining failed cells",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    protocol = arguments.protocol_option or arguments.protocol
    if not protocol:
        raise SystemExit("a JSON protocol path is required")
    if arguments.protocol_option and arguments.protocol:
        raise SystemExit("pass the protocol path once")
    result = run_protocol(
        Path(protocol),
        output_directory=arguments.output,
        resume=not arguments.no_resume,
        retry_failures=arguments.retry_failures,
        timeout_seconds=arguments.timeout,
    )
    failed = (
        int((result.manifests["status"] != "completed").sum())
        if not result.manifests.empty and "status" in result.manifests
        else 0
    )
    reliability_failures = (
        int((~result.reliability["run_success"].fillna(False).astype(bool)).sum())
        if not result.reliability.empty
        and "run_success" in result.reliability
        else 0
    )
    print(
        json.dumps(
            {
                "output_directory": str(
                    Path(arguments.output).resolve()
                    if arguments.output
                    else result.protocol["_execution"]["output_directory"]
                ),
                "cells": int(len(result.manifests)),
                "failed_cells": failed,
                "reliability_failures": reliability_failures,
                "metric_rows": int(len(result.metrics)),
                "sample_files": int(len(result.samples)),
            },
            sort_keys=True,
        )
    )
    return 0 if arguments.allow_failures or (failed == 0 and reliability_failures == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())

