from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from filelock import SoftFileLock

aggregate_SCHEMA_VERSION = 1
aggregate_IDENTITY_FIELDS = (
    "domain",
    "kind",
    "table",
    "tables",
    "encoder",
    "benchmark",
    "model",
    "task",
    "step",
    "variant",
    "method",
    "endpoint_source",
    "finetune_seed",
    "merge_seed",
)


def aggregate_canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def aggregate_atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def aggregate_atomic_json(path: Path, value: Any) -> None:
    aggregate_atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
    )


def aggregate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def aggregate_manifests(root: Path) -> list[tuple[str, Path]]:
    summary_path = root / "matrix_summary.json"
    summary = aggregate_load_json(summary_path)
    output: list[tuple[str, Path]] = []
    for stage, record in summary.get("stages", {}).items():
        path = root / str(record["manifest"])
        if aggregate_sha256(path) != record["sha256"]:
            raise ValueError(f"manifest hash differs: {path}")
        output.append((str(stage), path))
    if not output:
        raise ValueError(f"no manifests declared by {summary_path}")
    return output


def aggregate_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("experiment_id"):
                raise ValueError(f"invalid row {path}:{line_number}")
            yield value


def aggregate_numbers(value: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            yield (prefix or "value", number)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from aggregate_numbers(child, child_prefix)


def aggregate_method_payloads(
    result: Mapping[str, Any],
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    methods = result.get("method_results")
    if isinstance(methods, Mapping):
        for method, payload in methods.items():
            if isinstance(payload, Mapping):
                yield (str(method), payload)
        return
    method = str(result.get("method", result.get("request", {}).get("method", "run")))
    yield (method, result)


def aggregate_long_metrics(
    stage: str, request: Mapping[str, Any], result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    common: dict[str, Any] = {"stage": stage, "experiment_id": request["experiment_id"]}
    for field in aggregate_IDENTITY_FIELDS:
        value = request.get(field)
        common[field] = (
            aggregate_canonical(value)
            if isinstance(value, (list, dict, tuple))
            else value
        )
    output: list[dict[str, Any]] = []
    for method, payload in aggregate_method_payloads(result):
        aggregate_aggregate = payload.get("aggregate_aggregate", {})
        if isinstance(aggregate_aggregate, Mapping):
            for metric, value in aggregate_numbers(aggregate_aggregate):
                output.append(
                    {
                        **common,
                        "method": method,
                        "scope": "aggregate_aggregate",
                        "dataset": "__macro__",
                        "metric": metric,
                        "value": value,
                    }
                )
        per_dataset = payload.get("per_dataset", {})
        if isinstance(per_dataset, Mapping):
            for dataset, metrics in per_dataset.items():
                for metric, value in aggregate_numbers(metrics):
                    output.append(
                        {
                            **common,
                            "method": method,
                            "scope": "dataset",
                            "dataset": str(dataset),
                            "metric": metric,
                            "value": value,
                        }
                    )
        training = payload.get(
            "training_metrics", payload.get("validation_metrics", {})
        )
        if isinstance(training, Mapping):
            for metric, value in aggregate_numbers(training):
                output.append(
                    {
                        **common,
                        "method": method,
                        "scope": "training",
                        "dataset": str(request.get("task", "__training__")),
                        "metric": metric,
                        "value": value,
                    }
                )
    return output


def aggregate_group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    ignored = {"experiment_id", "finetune_seed", "merge_seed", "value"}
    return tuple(((key, row.get(key)) for key in sorted(set(row) - ignored)))


def aggregate_seed_statistics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(aggregate_group_key(row), []).append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: repr(item[0])):
        values = [float(member["value"]) for member in members]
        identity = dict(key)
        output.append(
            {
                **identity,
                "run_count": len(values),
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return output


def aggregate_median_runs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("scope") == "aggregate_aggregate"
        and row.get("metric")
        in {"macro_accuracy", "macro_score", "macro_normalized_accuracy"}
    ]
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(aggregate_group_key(row), []).append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: repr(item[0])):
        ordered = sorted(
            members,
            key=lambda row: (
                float(row["value"]),
                int(row.get("merge_seed") or row.get("finetune_seed") or 0),
                str(row["experiment_id"]),
            ),
        )
        representative = ordered[(len(ordered) - 1) // 2]
        output.append(
            {
                **dict(key),
                "representative_experiment_id": representative["experiment_id"],
                "representative_finetune_seed": representative.get("finetune_seed"),
                "representative_merge_seed": representative.get("merge_seed"),
                "representative_value": representative["value"],
                "ordered_experiment_ids": [row["experiment_id"] for row in ordered],
            }
        )
    return output


def aggregate_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    if not fields:
        aggregate_atomic_text(path, "")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def aggregate_aggregate(
    manifest_root: Path, results_root: Path, output: Path
) -> dict[str, Any]:
    ledger: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected = 0
    for stage, manifest in aggregate_manifests(manifest_root):
        for request in aggregate_rows(manifest):
            expected += 1
            experiment_id = str(request["experiment_id"])
            if experiment_id in seen:
                raise ValueError(f"duplicate experiment id: {experiment_id}")
            seen.add(experiment_id)
            path = results_root / "experiments" / experiment_id / "result.json"
            if not path.is_file():
                missing.append(
                    {
                        "stage": stage,
                        "experiment_id": experiment_id,
                        "reason": "missing",
                        "path": str(path),
                    }
                )
                continue
            try:
                result = aggregate_load_json(path)
            except Exception as error:
                missing.append(
                    {
                        "stage": stage,
                        "experiment_id": experiment_id,
                        "reason": f"invalid_json:{type(error).__name__}",
                        "path": str(path),
                    }
                )
                continue
            if (
                result.get("complete") is not True
                or result.get("experiment_id") != experiment_id
            ):
                missing.append(
                    {
                        "stage": stage,
                        "experiment_id": experiment_id,
                        "reason": "incomplete_or_id_mismatch",
                        "path": str(path),
                    }
                )
                continue
            ledger.append(
                {
                    "stage": stage,
                    "experiment_id": experiment_id,
                    "request": request,
                    "result_path": str(path),
                    "result_sha256": aggregate_sha256(path),
                    "result": result,
                }
            )
            metrics.extend(aggregate_long_metrics(stage, request, result))
    output.mkdir(parents=True, exist_ok=True)
    aggregate_atomic_text(
        output / "experiment_ledger.jsonl",
        "".join((aggregate_canonical(row) + "\n" for row in ledger)),
    )
    aggregate_atomic_text(
        output / "missing_failed.jsonl",
        "".join((aggregate_canonical(row) + "\n" for row in missing)),
    )
    aggregate_write_csv(output / "metrics_long.csv", metrics)
    statistics_rows = aggregate_seed_statistics(metrics)
    aggregate_write_csv(output / "seed_statistics.csv", statistics_rows)
    median = aggregate_median_runs(metrics)
    aggregate_atomic_text(
        output / "median_real_runs.jsonl",
        "".join((aggregate_canonical(row) + "\n" for row in median)),
    )
    completion = {
        "schema_version": aggregate_SCHEMA_VERSION,
        "complete": not missing and len(ledger) == expected,
        "expected_experiments": expected,
        "completed_experiments": len(ledger),
        "missing_or_failed_experiments": len(missing),
        "metric_rows": len(metrics),
        "seed_statistic_rows": len(statistics_rows),
        "median_representative_rows": len(median),
        "artifacts": {
            name: {
                "sha256": aggregate_sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in (
                "experiment_ledger.jsonl",
                "missing_failed.jsonl",
                "metrics_long.csv",
                "seed_statistics.csv",
                "median_real_runs.jsonl",
            )
        },
    }
    aggregate_atomic_json(output / "completion.json", completion)
    return completion


def aggregate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    summary = aggregate_aggregate(
        args.manifest_root.expanduser().resolve(),
        args.results_root.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["complete"] or args.allow_incomplete else 1


def validate_identifier(value: str) -> str:
    if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def contained_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Artifact path escapes its output root")
    return path


@dataclass(frozen=True)
class Resources:
    partition: str
    gpu_type: str
    gpus: int = 1
    cpus: int = 8
    memory_gb: int = 64
    hours: int = 24
    account: str | None = None
    qos: str | None = None

    def __post_init__(self):
        for value in (self.partition, self.gpu_type, self.account, self.qos):
            if value is not None:
                validate_identifier(value)
        if min(self.gpus, self.cpus, self.memory_gb, self.hours) < 1:
            raise ValueError("Resource requests must be positive")

    def arguments(self) -> list[str]:
        result = [
            "--partition",
            self.partition,
            "--gres",
            f"gpu:{self.gpu_type}:{self.gpus}",
            "--cpus-per-task",
            str(self.cpus),
            "--mem",
            f"{self.memory_gb}G",
            "--time",
            f"{self.hours}:00:00",
            "--nodes",
            "1",
            "--ntasks",
            "1",
        ]
        for name in ("account", "qos"):
            if getattr(self, name):
                result.extend(["--" + name, getattr(self, name)])
        return result


@dataclass(frozen=True)
class Job:
    experiment_id: str
    argv: tuple[str, ...]
    resources: Resources
    dependencies: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    expected_artifacts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        validate_identifier(self.experiment_id)
        if not self.argv or any(
            (not isinstance(v, str) or "\x00" in v for v in self.argv)
        ):
            raise ValueError("Invalid command arguments")
        for value in self.dependencies:
            validate_identifier(value)
        if self.experiment_id in self.dependencies:
            raise ValueError("Self dependency is forbidden")
        for key, value in self.environment.items():
            if not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", key) or "\x00" in value:
                raise ValueError("Invalid environment entry")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Job":
        values = dict(payload)
        values["resources"] = Resources(**values["resources"])
        for key in ("argv", "dependencies", "expected_artifacts"):
            values[key] = tuple(values.get(key, ()))
        return cls(**values)


class ExperimentLedger:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "ledger.json"
        self.lock = SoftFileLock(str(self.root / "ledger.lock"), timeout=60)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "jobs": {}}
        result = json.loads(self.path.read_text(encoding="utf-8"))
        if result.get("schema_version") != 1 or not isinstance(
            result.get("jobs"), dict
        ):
            raise ValueError("Invalid ledger schema")
        return result

    def ordered(self, ledger: dict) -> list[str]:
        pending, ordered = (set(ledger["jobs"]), [])
        while pending:
            ready = sorted(
                (
                    key
                    for key in pending
                    if set(ledger["jobs"][key]["job"]["dependencies"]) <= set(ordered)
                )
            )
            if not ready:
                raise ValueError("Cyclic or missing dependency")
            ordered.extend(ready)
            pending.difference_update(ready)
        return ordered

    def register(self, jobs: Sequence[Job]) -> None:
        if len({job.experiment_id for job in jobs}) != len(jobs):
            raise ValueError("Duplicate experiment IDs")
        with self.lock:
            ledger = self._load()
            for job in jobs:
                payload = asdict(job)
                fingerprint = hashlib.sha256(
                    aggregate_canonical(payload).encode()
                ).hexdigest()
                previous = ledger["jobs"].get(job.experiment_id)
                if previous is not None and previous["fingerprint"] != fingerprint:
                    raise ValueError("An immutable job definition changed")
                if previous is None:
                    ledger["jobs"][job.experiment_id] = {
                        "job": payload,
                        "fingerprint": fingerprint,
                        "state": "UNSUBMITTED",
                        "attempts": [],
                    }
            self.ordered(ledger)
            aggregate_atomic_json(self.path, ledger)

    def snapshot(self) -> dict:
        with self.lock:
            return self._load()

    def artifacts_complete(self, identifier: str) -> bool:
        record = self.snapshot()["jobs"][identifier]
        paths = record["job"]["expected_artifacts"]
        if not paths:
            return False
        for relative in paths:
            path = contained_path(self.root, relative)
            if not path.is_file() or path.stat().st_size == 0:
                return False
            if path.suffix == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    return False
        return True


class Slurm:
    def __init__(
        self,
        ledger: ExperimentLedger,
        working_directory: str | Path,
        runner: Any = subprocess.run,
    ):
        self.ledger = ledger
        self.working_directory = Path(working_directory).resolve()
        if not self.working_directory.is_dir():
            raise ValueError("Working directory does not exist")
        self.runner = runner

    def _run(self, argv: Sequence[str]) -> str:
        return self.runner(
            list(argv),
            check=True,
            capture_output=True,
            text=True,
            cwd=self.working_directory,
            timeout=60,
        ).stdout.strip()

    def render_script(self, job: Job) -> str:
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "umask 077",
            "cd -- " + shlex.quote(str(self.working_directory)),
        ]
        lines.extend(
            (
                "export " + key + "=" + shlex.quote(value)
                for key, value in sorted(job.environment.items())
            )
        )
        lines.append("exec " + shlex.join(job.argv))
        return "\n".join(lines) + "\n"

    def submit(
        self, identifier: str, *, retry_failed: bool = False, dry_run: bool = False
    ) -> dict:
        with self.ledger.lock:
            ledger = self.ledger._load()
            record = ledger["jobs"][identifier]
            allowed = {"UNSUBMITTED"}
            if retry_failed:
                allowed.update({"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"})
            if record["state"] not in allowed:
                return {
                    "submitted": False,
                    "experiment_id": identifier,
                    "state": record["state"],
                }
            job = Job.from_payload(record["job"])
            dependencies = []
            for name in job.dependencies:
                parent = ledger["jobs"][name]
                if parent["state"] == "COMPLETED":
                    continue
                if parent["state"] not in {
                    "SUBMITTED",
                    "PENDING",
                    "RUNNING",
                } or not parent.get("slurm_id"):
                    raise ValueError(f"Dependency is not submitted: {name}")
                dependencies.append(parent["slurm_id"])
            script = self.ledger.root / "scripts" / (identifier + ".sh")
            logs = self.ledger.root / "logs"
            command = [
                "sbatch",
                "--parsable",
                "--job-name",
                identifier,
                "--output",
                str(logs / (identifier + "-%j.out")),
                "--error",
                str(logs / (identifier + "-%j.err")),
                *job.resources.arguments(),
            ]
            if dependencies:
                command.extend(["--dependency", "afterok:" + ":".join(dependencies)])
            command.append(str(script))
            if dry_run:
                return {
                    "submitted": False,
                    "command": command,
                    "script": self.render_script(job),
                }
            logs.mkdir(exist_ok=True)
            aggregate_atomic_text(script, self.render_script(job))
            record["state"] = "SUBMITTING"
            record["attempts"].append({"started_unix": time.time(), "command": command})
            aggregate_atomic_json(self.ledger.path, ledger)
            try:
                value = self._run(command).split(";", 1)[0]
                if not re.fullmatch("\\d+", value):
                    raise ValueError("Unrecognized sbatch response")
                record.update(state="SUBMITTED", slurm_id=value)
                record["attempts"][-1]["slurm_id"] = value
            except Exception as error:
                record.update(state="UNKNOWN", error=str(error))
                aggregate_atomic_json(self.ledger.path, ledger)
                raise
            aggregate_atomic_json(self.ledger.path, ledger)
            return {"submitted": True, "experiment_id": identifier, "slurm_id": value}

    def submit_all(
        self, limit: int | None = None, retry_failed: bool = False
    ) -> list[dict]:
        if limit is not None and limit < 1:
            raise ValueError("Submission limit must be positive")
        ordered = self.ledger.ordered(self.ledger.snapshot())
        results, submitted = ([], 0)
        for identifier in ordered:
            if limit is not None and submitted >= limit:
                break
            result = self.submit(identifier, retry_failed=retry_failed)
            submitted += int(result["submitted"])
            results.append(result)
        return results

    def status(self) -> dict[str, str]:
        ledger = self.ledger.snapshot()
        active = {
            record["slurm_id"]: name
            for name, record in ledger["jobs"].items()
            if record.get("slurm_id")
            and record["state"] not in {"COMPLETED", "CANCELLED"}
        }
        if not active:
            return {}
        output = self._run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                ",".join(active),
                "--format",
                "JobIDRaw,State,ExitCode",
            ]
        )
        statuses = {}
        for row in output.splitlines():
            fields = row.split("|")
            if len(fields) < 3 or fields[0] not in active:
                continue
            job_id, state, exit_code = fields[:3]
            state = state.split()[0].rstrip("+")
            if state not in {
                "PENDING",
                "RUNNING",
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
            }:
                state = "UNKNOWN"
            name = active[job_id]
            if state == "COMPLETED" and (
                exit_code != "0:0"
                or (
                    ledger["jobs"][name]["job"]["expected_artifacts"]
                    and (not self.ledger.artifacts_complete(name))
                )
            ):
                state = "FAILED"
            statuses[name] = state
        with self.ledger.lock:
            ledger = self.ledger._load()
            for name, state in statuses.items():
                ledger["jobs"][name]["state"] = state
            aggregate_atomic_json(self.ledger.path, ledger)
        return statuses

    def cancel(self, identifiers: Sequence[str]) -> list[str]:
        with self.ledger.lock:
            ledger = self.ledger._load()
            resolved = []
            for name in identifiers:
                record = ledger["jobs"][name]
                if record["state"] in {"SUBMITTED", "PENDING", "RUNNING"}:
                    resolved.append((name, record["slurm_id"]))
            if resolved:
                self._run(["scancel", *[job_id for _, job_id in resolved]])
                for name, _ in resolved:
                    ledger["jobs"][name]["state"] = "CANCELLED"
                aggregate_atomic_json(self.ledger.path, ledger)
            return [name for name, _ in resolved]


def export_numeric_results(
    root: str | Path, destination: str | Path, maximum_bytes: int = 100000000
) -> dict:
    source, target = (Path(root).resolve(), Path(destination).resolve())
    if (
        not source.is_dir()
        or source == target
        or source in target.parents
        or (target in source.parents)
    ):
        raise ValueError("Use separate source and destination trees")
    paths = []
    for path in sorted(source.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix not in {".json", ".jsonl", ".csv", ".tsv"}
        ):
            continue
        if any(
            (
                part in {"data", "datasets", "checkpoints", ".git", "cache"}
                or part.startswith("checkpoint-")
                for part in path.relative_to(source).parts
            )
        ):
            continue
        paths.append(path)
    size = sum((path.stat().st_size for path in paths))
    if size > maximum_bytes:
        raise ValueError(f"Numeric export exceeds limit: {size} bytes")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Destination must be empty")
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for path in paths:
        relative = path.relative_to(source).as_posix()
        output = contained_path(target, relative)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
        records.append(
            {
                "file": relative,
                "bytes": output.stat().st_size,
                "sha256": aggregate_sha256(output),
            }
        )
    result = {"files": records, "total_bytes": size, "checkpoints_included": False}
    aggregate_atomic_json(target / "numeric_manifest.json", result)
    return result


def workflow_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beliefmerge workflow")
    parser.add_argument(
        "command", choices=["register", "submit", "status", "cancel", "export"]
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--working-directory")
    parser.add_argument("--destination")
    parser.add_argument("--experiment-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "export":
        if not args.destination:
            parser.error("--destination is required")
        result = export_numeric_results(args.root, args.destination)
    else:
        ledger = ExperimentLedger(args.root)
        if args.command == "register":
            if not args.manifest:
                parser.error("--manifest is required")
            jobs = [
                Job.from_payload(row)
                for row in json.loads(Path(args.manifest).read_text())
            ]
            ledger.register(jobs)
            result = {"registered": len(jobs)}
        else:
            if not args.working_directory:
                parser.error("--working-directory is required")
            scheduler = Slurm(ledger, args.working_directory)
            if args.command == "submit":
                result = scheduler.submit_all(args.limit, args.retry_failed)
            elif args.command == "cancel":
                if not args.experiment_ids:
                    parser.error("--experiment-ids is required")
                result = scheduler.cancel(args.experiment_ids)
            else:
                result = scheduler.status()
    print(json.dumps(result, indent=2))
    return 0
