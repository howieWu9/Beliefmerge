from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
import traceback
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

FUSIONBENCH_COMMIT = "54c9e8c9d9621620c720452cd8533332a32d3689"
BASE_MODEL_ID = "google/flan-t5-base"
GLUE_TASKS = ("cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb")
CLASSIFICATION_TASKS = frozenset(GLUE_TASKS[:-1])
REGRESSION_TASKS = frozenset(("stsb",))
VALIDATION_SPLITS = {
    "cola": "validation",
    "mnli": "validation_matched",
    "mrpc": "validation",
    "qnli": "validation",
    "qqp": "validation",
    "rte": "validation",
    "sst2": "validation",
    "stsb": "validation",
}
PROMPTS: Mapping[str, Mapping[str, Any]] = {
    "cola": {
        "input": "Indicate if the following sentence is grammatically correct or not: \"{sentence}\". Answere 'acceptable' or 'unacceptable'.",
        "targets": {0: "unacceptable", 1: "acceptable"},
    },
    "mnli": {
        "input": "Does the premise: '{premise}' logically imply, contradict, or is neutral to the hypothesis: '{hypothesis}'? Answere with 'entailment', 'contradiction', or 'neutral'.",
        "targets": {0: "entailment", 1: "neutral", 2: "contradiction"},
    },
    "mrpc": {
        "input": "Are the following sentences '{sentence1}' and '{sentence2}' conveying the same meaning? Answere with 'yes' or 'no'.",
        "targets": {0: "no", 1: "yes"},
    },
    "qnli": {
        "input": "Given the context: '{sentence}', does the question '{question}' have an answer based on the information provided? Answer with 'yes' or 'no'.",
        "targets": {0: "yes", 1: "no"},
    },
    "qqp": {
        "input": "Do the questions '{question1}' and '{question2}' have the same intent? Answere with 'yes' or 'no'.",
        "targets": {0: "no", 1: "yes"},
    },
    "rte": {
        "input": "Does the text: '{sentence1}' entail that '{sentence2}' is true? Provide 'yes' or 'no'.",
        "targets": {0: "yes", 1: "no"},
    },
    "sst2": {
        "input": "Given the sentence '{sentence}', determine the sentiment. Is it positive or negative?",
        "targets": {0: "negative", 1: "positive"},
    },
    "stsb": {
        "input": "Consider the sentences '{sentence1}' and '{sentence2}'. On a scale from 1 (completely different) to 5 (completely similar), rate the similarity."
    },
}


def canonical_task_name(name: str) -> str:
    task = str(name).strip().lower().replace("_", "-")
    if task.startswith("glue-"):
        task = task[5:]
    aliases = {"sst-2": "sst2", "sts-b": "stsb"}
    task = aliases.get(task, task)
    if task not in GLUE_TASKS:
        raise ValueError(
            f"unsupported GLUE task {name!r}; expected one of {GLUE_TASKS}"
        )
    return task


def format_example(task: str, example: Mapping[str, Any]) -> tuple[str, str]:
    task = canonical_task_name(task)
    prompt = PROMPTS[task]
    source = str(prompt["input"]).format(**example)
    label = example.get("label", -1)
    if task == "stsb":
        target = "" if float(label) < 0 else f"{float(label):.1f}"
    else:
        try:
            target = str(prompt["targets"][int(label)])
        except (KeyError, TypeError, ValueError):
            target = ""
    return (source, target)


def checkpoint_steps(max_steps: int = 2000, interval: int = 50) -> tuple[int, ...]:
    if max_steps < 1 or interval < 1:
        raise ValueError("max_steps and interval must be positive")
    if max_steps % interval:
        raise ValueError("max_steps must be divisible by the checkpoint interval")
    return tuple(range(interval, max_steps + 1, interval))


def require_local_path(
    value: str | os.PathLike[str], *, kind: str, directory: bool = True
) -> Path:
    raw = str(value).strip()
    if not raw or "://" in raw:
        raise ValueError(f"{kind} must be an explicit local path, got {value!r}")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    if directory and (not path.is_dir()):
        raise NotADirectoryError(f"{kind} is not a directory: {path}")
    if not directory and (not path.is_file()):
        raise FileNotFoundError(f"{kind} is not a file: {path}")
    return path


def configure_offline_environment(cache_root: str | os.PathLike[str]) -> Path:
    raw = str(cache_root).strip()
    if not raw or "://" in raw:
        raise ValueError("cache_root must be an explicit local path")
    root = Path(raw).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    locations = {
        "HF_HOME": root / "huggingface",
        "HF_DATASETS_CACHE": root / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": root / "huggingface" / "transformers",
        "XDG_CACHE_HOME": root / "xdg",
        "TORCH_HOME": root / "torch",
        "TMPDIR": root / "tmp",
        "TMP": root / "tmp",
        "TEMP": root / "tmp",
    }
    for variable, location in locations.items():
        location.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(location)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return root


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def directory_inventory(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    root = require_local_path(path, kind="artifact directory")
    inventory: list[dict[str, Any]] = []
    for item in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file())
    ):
        stat = item.stat()
        inventory.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return inventory


finetune_SCHEMA_VERSION = 1
finetune_CHECKPOINT_INTERVAL = 50
finetune_DEFAULT_MAX_STEPS = 2000


def finetune_rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def finetune_spearman_rho(
    predictions: Sequence[float], labels: Sequence[float]
) -> float:
    if len(predictions) != len(labels) or len(predictions) < 2:
        raise ValueError("Spearman requires equally sized sequences of length >= 2")
    x = finetune_rankdata([float(value) for value in predictions])
    y = finetune_rankdata([float(value) for value in labels])
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    numerator = sum(((a - mean_x) * (b - mean_y) for a, b in zip(x, y)))
    denominator = math.sqrt(
        sum(((a - mean_x) ** 2 for a in x)) * sum(((b - mean_y) ** 2 for b in y))
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def finetune_decode_metrics(
    task: str, predictions: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    task = canonical_task_name(task)
    if len(predictions) != len(labels) or not labels:
        raise ValueError("predictions and labels must be non-empty and equally sized")
    if task in CLASSIFICATION_TASKS:
        correct = sum(
            (
                str(prediction) == str(label)
                for prediction, label in zip(predictions, labels)
            )
        )
        return {
            "accuracy": correct / len(labels),
            "correct": correct,
            "example_count": len(labels),
        }
    parsed_predictions: list[float] = []
    parsed_labels: list[float] = []
    invalid = 0
    for prediction, label in zip(predictions, labels):
        try:
            parsed_predictions.append(float(prediction))
        except (TypeError, ValueError):
            parsed_predictions.append(0.0)
            invalid += 1
        parsed_labels.append(float(label))
    return {
        "finetune_spearman_rho": finetune_spearman_rho(
            parsed_predictions, parsed_labels
        ),
        "invalid_prediction_count": invalid,
        "invalid_prediction_fraction": invalid / len(labels),
        "example_count": len(labels),
    }


def finetune_checkpoint_number(path: Path) -> int:
    prefix = "checkpoint-"
    if not path.name.startswith(prefix):
        raise ValueError(f"not a Trainer checkpoint directory: {path}")
    try:
        return int(path.name[len(prefix) :])
    except ValueError as error:
        raise ValueError(f"invalid checkpoint name: {path.name}") from error


def finetune_checkpoint_directories(output_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not output_dir.exists():
        return result
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        step = finetune_checkpoint_number(path)
        if step in result:
            raise ValueError(f"duplicate checkpoint step {step}")
        result[step] = path.resolve()
    return result


def finetune_checkpoint_receipt(path: Path) -> dict[str, Any]:
    files = []
    for item in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file())
    ):
        files.append(
            {
                "path": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": file_sha256(item),
            }
        )
    if not files:
        raise ValueError(f"checkpoint is empty: {path}")
    trainer_state = path / "trainer_state.json"
    if not trainer_state.is_file():
        raise ValueError(f"checkpoint lacks trainer_state.json: {path}")
    state = json.loads(trainer_state.read_text(encoding="utf-8"))
    step = finetune_checkpoint_number(path)
    if int(state.get("global_step", -1)) != step:
        raise ValueError(
            f"checkpoint directory says step {step}, trainer state says {state.get('global_step')!r}"
        )
    return {
        "step": step,
        "path": str(path.resolve()),
        "files": files,
        "total_bytes": sum((item["bytes"] for item in files)),
    }


def finetune_dataset_path(root: Path, task: str) -> Path:
    if (
        (root / "dataset_dict.json").is_file()
        or (root / "state.json").is_file()
        or any(root.glob("train*.parquet"))
    ):
        return root
    candidates = (root / task, root / "glue" / task)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot locate saved dataset for {task}; checked "
        + ", ".join((str(path) for path in candidates))
    )


def finetune_parquet_split_files(root: Path) -> dict[str, list[Path]]:
    known_splits = (
        "validation_mismatched",
        "validation_matched",
        "test_mismatched",
        "test_matched",
        "validation",
        "train",
        "test",
    )
    grouped: dict[str, list[Path]] = {}
    for path in sorted(Path(root).glob("*.parquet")):
        stem = path.stem
        split = next(
            (
                candidate
                for candidate in known_splits
                if stem == candidate or stem.startswith(f"{candidate}-")
            ),
            None,
        )
        if split is None:
            raise ValueError(f"unrecognized GLUE Parquet split filename: {path.name}")
        grouped.setdefault(split, []).append(path.resolve())
    if "train" not in grouped:
        raise FileNotFoundError(f"raw Parquet dataset lacks train split: {root}")
    return grouped


def finetune_build_request(args: argparse.Namespace) -> dict[str, Any]:
    task = canonical_task_name(args.task)
    if args.checkpoint_interval != finetune_CHECKPOINT_INTERVAL:
        raise ValueError("paper protocol requires checkpoint_interval=50")
    schedule = checkpoint_steps(args.max_steps, args.checkpoint_interval)
    base = require_local_path(args.base_model_dir, kind="base model")
    tokenizer = require_local_path(args.tokenizer_dir, kind="tokenizer")
    dataset_root = require_local_path(args.dataset_dir, kind="dataset")
    dataset = finetune_dataset_path(dataset_root, task)
    output = Path(args.output_dir).expanduser().resolve()
    cache = Path(args.cache_root).expanduser().resolve()
    if output == cache or output in cache.parents or cache in output.parents:
        raise ValueError("output_dir and cache_root must be separate directory trees")
    request = {
        "schema_version": finetune_SCHEMA_VERSION,
        "protocol": {
            "fusionbench_commit": FUSIONBENCH_COMMIT,
            "base_model": BASE_MODEL_ID,
            "setting": "full_parameter_finetuning",
            "task": task,
            "validation_split": VALIDATION_SPLITS[task],
            "checkpoint_steps": list(schedule),
        },
        "paths": {
            "base_model_dir": str(base),
            "tokenizer_dir": str(tokenizer),
            "dataset_dir": str(dataset),
            "output_dir": str(output),
            "cache_root": str(cache),
        },
        "training": {
            "seed": args.seed,
            "max_steps": args.max_steps,
            "checkpoint_interval": args.checkpoint_interval,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.train_batch_size
            * args.gradient_accumulation_steps,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
            "bf16": args.bf16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "dataloader_num_workers": args.dataloader_num_workers,
            "optimizer": "adamw_torch",
            "lr_scheduler_type": "linear",
        },
        "input_inventory": {
            "base_model": directory_inventory(base),
            "tokenizer": directory_inventory(tokenizer),
            "dataset": directory_inventory(dataset),
        },
    }
    request["request_sha256"] = stable_json_hash(request)
    return request


def finetune_load_or_initialize_manifest(request: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(request["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("request", {}).get("request_sha256")
            != request["request_sha256"]
        ):
            raise ValueError(
                "existing output belongs to a different immutable request; choose another output directory"
            )
        return manifest
    manifest = {
        "schema_version": finetune_SCHEMA_VERSION,
        "state": "PENDING",
        "request": dict(request),
        "runtime": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "checkpoints": {},
        "evaluations": {},
        "attempts": [],
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def finetune_validate_existing_boundaries(
    output: Path, expected_steps: Sequence[int]
) -> dict[int, Path]:
    checkpoints = finetune_checkpoint_directories(output)
    expected = set(expected_steps)
    unexpected = sorted(set(checkpoints) - expected)
    if unexpected:
        raise ValueError(f"found off-protocol checkpoint steps: {unexpected}")
    if checkpoints:
        latest = max(checkpoints)
        missing_prefix = [
            step
            for step in expected_steps
            if step <= latest and step not in checkpoints
        ]
        if missing_prefix:
            raise ValueError(
                f"checkpoint history has gaps before step {latest}: {missing_prefix}"
            )
    return checkpoints


def finetune_tokenize_dataset(
    raw_dataset: Any, tokenizer: Any, task: str, args: Any
) -> Any:
    required = {"input_ids", "attention_mask", "labels"}
    if required.issubset(set(raw_dataset.column_names)):
        return raw_dataset

    def tokenize_batch(batch: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
        count = len(batch["label"])
        sources: list[str] = []
        targets: list[str] = []
        for index in range(count):
            row = {key: values[index] for key, values in batch.items()}
            source, target = format_example(task, row)
            sources.append(source)
            targets.append(target)
        encoded = tokenizer(
            sources,
            padding="max_length",
            truncation=True,
            max_length=args.max_source_length,
        )
        target_tokens = tokenizer(
            targets,
            padding="max_length",
            truncation=True,
            max_length=args.max_target_length,
        )["input_ids"]
        encoded["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in row]
            for row in target_tokens
        ]
        return encoded

    return raw_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc=f"Pinned FusionBench preprocessing: {task}",
    )


def finetune_append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def finetune_run_training(args: argparse.Namespace, request: Mapping[str, Any]) -> None:
    import numpy as np
    from datasets import Dataset, DatasetDict, load_from_disk
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        set_seed,
    )

    task = request["protocol"]["task"]
    output = Path(request["paths"]["output_dir"])
    manifest_path = output / "manifest.json"
    metrics_root = output / "metrics"
    event_path = output / "events.jsonl"
    expected_steps = request["protocol"]["checkpoint_steps"]
    manifest = finetune_load_or_initialize_manifest(request)
    checkpoints = finetune_validate_existing_boundaries(output, expected_steps)
    if manifest.get("state") == "COMPLETE" and set(checkpoints) == set(expected_steps):
        return
    attempt = {
        "attempt": len(manifest["attempts"]) + 1,
        "started_unix": time.time(),
        "resume_step": max(checkpoints, default=0),
    }
    manifest["attempts"].append(attempt)
    manifest["state"] = "RUNNING"
    atomic_write_json(manifest_path, manifest)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        request["paths"]["tokenizer_dir"], local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        request["paths"]["base_model_dir"],
        local_files_only=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    dataset_path = Path(request["paths"]["dataset_dir"])
    if (dataset_path / "dataset_dict.json").is_file() or (
        dataset_path / "state.json"
    ).is_file():
        loaded = load_from_disk(str(dataset_path))
    else:
        parquet_files = finetune_parquet_split_files(dataset_path)
        loaded = DatasetDict(
            {
                split: Dataset.from_parquet([str(path) for path in paths])
                for split, paths in parquet_files.items()
            }
        )
    if not isinstance(loaded, DatasetDict):
        raise ValueError(
            "dataset_dir must contain a saved DatasetDict with train/validation"
        )
    validation_split = request["protocol"]["validation_split"]
    if "train" not in loaded or validation_split not in loaded:
        raise KeyError(
            f"dataset needs train and {validation_split!r}; available={list(loaded)}"
        )
    train_dataset = finetune_tokenize_dataset(loaded["train"], tokenizer, task, args)
    eval_dataset = finetune_tokenize_dataset(
        loaded[validation_split], tokenizer, task, args
    )

    def compute_metrics(prediction: Any) -> dict[str, Any]:
        token_ids = prediction.predictions
        if isinstance(token_ids, tuple):
            token_ids = token_ids[0]
        label_ids = np.asarray(prediction.label_ids).copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        decoded_predictions = tokenizer.batch_decode(
            token_ids, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return finetune_decode_metrics(task, decoded_predictions, decoded_labels)

    callback_manifest = manifest

    class AuditCallback(TrainerCallback):
        def _persist(self) -> None:
            atomic_write_json(manifest_path, callback_manifest)

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            del training_args, control, kwargs
            if logs:
                event = {
                    "kind": "log",
                    "step": int(state.global_step),
                    "unix": time.time(),
                    "values": dict(logs),
                }
                finetune_append_event(event_path, event)

        def on_evaluate(self, training_args, state, control, metrics=None, **kwargs):
            del training_args, control, kwargs
            step = int(state.global_step)
            if step and step % finetune_CHECKPOINT_INTERVAL:
                raise RuntimeError(f"off-protocol evaluation at step {step}")
            payload = {
                "schema_version": finetune_SCHEMA_VERSION,
                "task": task,
                "step": step,
                "metrics": dict(metrics or {}),
                "unix": time.time(),
            }
            metric_path = metrics_root / f"step_{step:04d}.json"
            atomic_write_json(metric_path, payload)
            callback_manifest["evaluations"][str(step)] = {
                "path": str(metric_path.resolve()),
                "sha256": file_sha256(metric_path),
                "metrics": payload["metrics"],
            }
            self._persist()

        def on_save(self, training_args, state, control, **kwargs):
            del training_args, control, kwargs
            step = int(state.global_step)
            if step not in expected_steps:
                raise RuntimeError(f"off-protocol checkpoint at step {step}")
            path = output / f"checkpoint-{step}"
            callback_manifest["checkpoints"][str(step)] = finetune_checkpoint_receipt(
                path
            )
            self._persist()

    argument_values: dict[str, Any] = {
        "output_dir": str(output),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "linear",
        "optim": "adamw_torch",
        "logging_strategy": "steps",
        "logging_steps": finetune_CHECKPOINT_INTERVAL,
        "save_strategy": "steps",
        "save_steps": finetune_CHECKPOINT_INTERVAL,
        "save_total_limit": None,
        "save_safetensors": True,
        "load_best_model_at_end": False,
        "predict_with_generate": True,
        "generation_max_length": 10,
        "generation_num_beams": 1,
        "bf16": args.bf16,
        "fp16": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "remove_unused_columns": True,
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
        "disable_tqdm": False,
    }
    signature = inspect.signature(Seq2SeqTrainingArguments)
    evaluation_key = (
        "eval_strategy"
        if "eval_strategy" in signature.parameters
        else "evaluation_strategy"
    )
    argument_values[evaluation_key] = "steps"
    argument_values["eval_steps"] = finetune_CHECKPOINT_INTERVAL
    training_args = Seq2SeqTrainingArguments(**argument_values)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="max_length",
            max_length=args.max_source_length,
            label_pad_token_id=-100,
        ),
        compute_metrics=compute_metrics,
        callbacks=[AuditCallback()],
    )
    resume_path = str(checkpoints[max(checkpoints)]) if checkpoints else None
    result = trainer.train(resume_from_checkpoint=resume_path)
    final_checkpoints = finetune_validate_existing_boundaries(output, expected_steps)
    missing_checkpoints = sorted(set(expected_steps) - set(final_checkpoints))
    missing_evaluations = sorted(
        (
            step
            for step in expected_steps
            if not (metrics_root / f"step_{step:04d}.json").is_file()
        )
    )
    if missing_checkpoints or missing_evaluations:
        raise RuntimeError(
            f"incomplete run: missing checkpoints={missing_checkpoints}, missing evaluations={missing_evaluations}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoints"] = {
        str(step): finetune_checkpoint_receipt(final_checkpoints[step])
        for step in expected_steps
    }
    manifest["state"] = "COMPLETE"
    manifest["completed_unix"] = time.time()
    manifest["train_result"] = dict(result.metrics)
    manifest["dataset_fingerprints"] = {
        "train": getattr(loaded["train"], "_fingerprint", None),
        "validation": getattr(loaded[validation_split], "_fingerprint", None),
        "tokenized_train": getattr(train_dataset, "_fingerprint", None),
        "tokenized_validation": getattr(eval_dataset, "_fingerprint", None),
    }
    manifest["library_versions"] = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
    }
    manifest["attempts"][-1]["completed_unix"] = time.time()
    manifest["attempts"][-1]["status"] = "COMPLETE"
    atomic_write_json(manifest_path, manifest)


def finetune_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=finetune_DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=finetune_CHECKPOINT_INTERVAL,
        choices=(50,),
    )
    parser.add_argument("--learning-rate", type=float, default=5e-05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def finetune_validate_numeric_args(args: argparse.Namespace) -> None:
    positive = {
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.weight_decay < 0.0 or not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("weight_decay/warmup_ratio are invalid")


def finetune_main(argv: Sequence[str] | None = None) -> int:
    args = finetune_build_parser().parse_args(argv)
    finetune_validate_numeric_args(args)
    configure_offline_environment(args.cache_root)
    request = finetune_build_request(args)
    manifest_path = Path(request["paths"]["output_dir"]) / "manifest.json"
    try:
        finetune_run_training(args, request)
    except BaseException as error:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["state"] = "FAILED"
            manifest["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "unix": time.time(),
            }
            if manifest.get("attempts"):
                manifest["attempts"][-1]["status"] = "FAILED"
                manifest["attempts"][-1]["ended_unix"] = time.time()
            atomic_write_json(manifest_path, manifest)
        raise
    return 0


def flan_t5_layer_name(key: str) -> str:
    tied = {
        "shared.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
    }
    if key in tied:
        return "shared_embedding"
    pieces = key.split(".")
    for index, piece in enumerate(pieces[:-1]):
        if piece == "block" and index + 1 < len(pieces) and pieces[index + 1].isdigit():
            return ".".join(pieces[: index + 2])
    if len(pieces) > 1 and pieces[-1] in {
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    }:
        return ".".join(pieces[:-1])
    return key


def _is_floating(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is not None and hasattr(dtype, "is_floating_point"):
        return bool(dtype.is_floating_point)
    try:
        return np.asarray(value).dtype.kind in "fc"
    except (TypeError, ValueError):
        return False


def logical_layer_names(state: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for key, value in state.items():
        if not _is_floating(value):
            continue
        name = flan_t5_layer_name(key)
        if name not in seen:
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError("model state contains no floating-point layers")
    return tuple(names)


def _canonical_model_names(names: Sequence[str]) -> tuple[str, ...]:
    canonical = tuple((canonical_task_name(name) for name in names))
    if len(canonical) != len(GLUE_TASKS) or set(canonical) != set(GLUE_TASKS):
        raise ValueError(
            f"full GLUE protocol requires exactly the eight unique tasks; got {canonical}"
        )
    return canonical


def _select_step_payload(
    payload: Mapping[str, Any], expected_step: int | None
) -> Mapping[str, Any]:
    if "steps" not in payload:
        return payload
    if expected_step is None:
        raise ValueError(
            "private-type artifact contains multiple steps; expected_step is required"
        )
    steps = payload["steps"]
    if not isinstance(steps, Mapping) or str(expected_step) not in steps:
        raise KeyError(f"private-type artifact lacks step {expected_step}")
    selected = steps[str(expected_step)]
    if not isinstance(selected, Mapping):
        selected = {"private_types": selected}
    combined = dict(payload)
    combined.pop("steps", None)
    combined.update(selected)
    combined["step"] = expected_step
    return combined


def load_private_types(
    path: str | os.PathLike[str],
    *,
    model_names: Sequence[str],
    layer_names: Sequence[str],
    expected_step: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = require_local_path(path, kind="private types", directory=False)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            required = {"private_types", "model_names", "layer_names"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise KeyError(f"private-type NPZ is missing arrays: {missing}")
            array = np.asarray(archive["private_types"], dtype=np.float64)
            artifact_models = [str(value) for value in archive["model_names"].tolist()]
            artifact_layers = [str(value) for value in archive["layer_names"].tolist()]
            artifact_step = (
                int(np.asarray(archive["step"]).item())
                if "step" in archive.files
                else None
            )
        metadata: dict[str, Any] = {"step": artifact_step, "format": "npz"}
    elif source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("private-type JSON root must be an object")
        payload = _select_step_payload(raw, expected_step)
        artifact_models = [
            str(value) for value in payload.get("model_names", model_names)
        ]
        artifact_layers = [
            str(value) for value in payload.get("layer_names", layer_names)
        ]
        if "private_types" in payload:
            array = np.asarray(payload["private_types"], dtype=np.float64)
        elif "types" in payload and isinstance(payload["types"], Mapping):
            types = payload["types"]
            array = np.asarray(
                [
                    [types[model][layer] for layer in artifact_layers]
                    for model in artifact_models
                ],
                dtype=np.float64,
            )
        else:
            raise KeyError("private-type JSON needs private_types or types")
        artifact_step = payload.get("step")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"private_types", "types"}
        }
        metadata["format"] = "json"
    else:
        raise ValueError("private types must be a .json or .npz file")
    requested_models = list(_canonical_model_names(model_names))
    normalized_artifact_models = [canonical_task_name(name) for name in artifact_models]
    if normalized_artifact_models != requested_models:
        raise ValueError(
            f"private-type model order mismatch: expected {requested_models}, got {normalized_artifact_models}"
        )
    if list(artifact_layers) != list(layer_names):
        raise ValueError(
            "private-type logical layer names/order do not match the model"
        )
    expected_shape_prefix = (len(requested_models), len(layer_names))
    if (
        array.ndim != 3
        or array.shape[:2] != expected_shape_prefix
        or array.shape[2] not in (3, 4)
    ):
        raise ValueError(
            f"private_types must have shape [8,L,3(+stability)]; expected prefix {expected_shape_prefix}, got {array.shape}"
        )
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("private types must be finite and lie in [0,1]")
    if expected_step is not None:
        if artifact_step is None:
            raise ValueError(
                "step-specific merge requires private artifact step metadata"
            )
        if int(artifact_step) != int(expected_step):
            raise ValueError(
                f"private artifact is for step {artifact_step}, expected {expected_step}"
            )
    metadata.update(
        {
            "path": str(source),
            "sha256": file_sha256(source),
            "model_names": requested_models,
            "layer_names": list(layer_names),
            "shape": list(array.shape),
        }
    )
    return (array, metadata)


def _extract_local_source(config: Any) -> str | None:
    if isinstance(config, (str, os.PathLike)):
        return str(config)
    if isinstance(config, Mapping):
        for key in ("pretrained_model_name_or_path", "model_name_or_path"):
            if key in config:
                return str(config[key])
    if hasattr(config, "get"):
        for key in ("pretrained_model_name_or_path", "model_name_or_path"):
            value = config.get(key, None)
            if value is not None:
                return str(value)
    return None


def validate_local_modelpool(modelpool: Any) -> dict[str, str]:
    names = list(getattr(modelpool, "all_model_names", []))
    if not names:
        names = ["_pretrained_", *list(modelpool.model_names)]
    sources: dict[str, str] = {}
    for name in names:
        config = modelpool.get_model_config(name, return_copy=False)
        if hasattr(config, "state_dict"):
            sources[str(name)] = "<pre-instantiated>"
            continue
        source = _extract_local_source(config)
        if source is None:
            raise ValueError(f"cannot audit local model source for {name!r}")
        resolved = require_local_path(source, kind=f"model {name}")
        sources[str(name)] = str(resolved)
    return sources


def _cpu_state(model: Any) -> OrderedDict[str, Any]:
    state: OrderedDict[str, Any] = OrderedDict()
    for key, value in model.state_dict().items():
        current = value.detach() if hasattr(value, "detach") else value
        if hasattr(current, "cpu"):
            current = current.cpu()
        state[key] = (
            current.clone()
            if hasattr(current, "clone")
            else np.array(current, copy=True)
        )
    return state


private_COMPONENTS = ("s_sens", "s_repr", "s_util", "s_stab")


def private_cpu_state(model: Any) -> OrderedDict[str, Any]:
    output: OrderedDict[str, Any] = OrderedDict()
    for key, value in model.state_dict().items():
        current = value.detach().cpu()
        output[key] = current.clone()
    return output


def private_layer_map(state: Mapping[str, Any]) -> OrderedDict[str, str]:
    result: OrderedDict[str, str] = OrderedDict()
    for key, value in state.items():
        if bool(value.is_floating_point()):
            result[key] = flan_t5_layer_name(key)
    if tuple(dict.fromkeys(result.values())) != logical_layer_names(state):
        raise AssertionError("logical layer resolver is internally inconsistent")
    return result


def private_named_parameters_all(model: Any) -> dict[str, Any]:
    signature = inspect.signature(model.named_parameters)
    kwargs = (
        {"remove_duplicate": False}
        if "remove_duplicate" in signature.parameters
        else {}
    )
    return dict(model.named_parameters(**kwargs))


def private_named_buffers_all(model: Any) -> dict[str, Any]:
    signature = inspect.signature(model.named_buffers)
    kwargs = (
        {"remove_duplicate": False}
        if "remove_duplicate" in signature.parameters
        else {}
    )
    return dict(model.named_buffers(**kwargs))


def private_trim_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(batch)
    mask = result.get("attention_mask")
    if mask is not None and mask.ndim == 2:
        used = mask.ne(0).any(dim=0)
        width = int(used.nonzero()[-1].item()) + 1 if bool(used.any()) else 1
        result["input_ids"] = result["input_ids"][:, :width]
        result["attention_mask"] = mask[:, :width]
    labels = result.get("labels")
    if labels is not None and labels.ndim == 2:
        used = labels.ne(-100).any(dim=0)
        width = int(used.nonzero()[-1].item()) + 1 if bool(used.any()) else 1
        result["labels"] = labels[:, :width]
    return result


def private_device_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in private_trim_batch(batch).items()
        if key in {"input_ids", "attention_mask", "labels"}
    }


def private_mean_loss(
    model: Any, batches: Sequence[Mapping[str, Any]], device: Any
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in batches:
            current = private_device_batch(batch, device)
            size = int(current["input_ids"].shape[0])
            total += float(model(**current).loss.float().item()) * size
            count += size
    if count == 0:
        raise ValueError("private-type calibration subset is empty")
    return total / count


def private_gradient_statistics(
    model: Any,
    batches: Sequence[Mapping[str, Any]],
    base_state: Mapping[str, Any],
    endpoint_state: Mapping[str, Any],
    layer_map: Mapping[str, str],
    layer_index: Mapping[str, int],
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name in layer_map
    ]
    sample_total = sum((int(batch["input_ids"].shape[0]) for batch in batches))
    sensitivity = np.zeros(len(layer_index), dtype=np.float64)
    gradient_sq = np.zeros(len(layer_index), dtype=np.float64)
    gradient_means = {
        name: torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        for name, parameter in named
    }
    deltas = {
        name: (endpoint_state[name] - base_state[name]).to(
            device=device, dtype=torch.float32
        )
        for name, _ in named
    }
    model.eval()
    for batch in batches:
        current = private_device_batch(batch, device)
        weight = int(current["input_ids"].shape[0]) / sample_total
        loss = model(**current).loss
        gradients = torch.autograd.grad(
            loss, [parameter for _, parameter in named], allow_unused=True
        )
        for (name, _), gradient in zip(named, gradients):
            if gradient is None:
                continue
            grad = gradient.detach().float()
            index = layer_index[layer_map[name]]
            gradient_sq[index] += weight * float(grad.double().square().sum().item())
            gradient_means[name].add_(grad.cpu(), alpha=weight)
        for sample in range(current["input_ids"].shape[0]):
            single = {key: value[sample : sample + 1] for key, value in current.items()}
            single_loss = model(**private_trim_batch(single)).loss
            single_gradients = torch.autograd.grad(
                single_loss, [parameter for _, parameter in named], allow_unused=True
            )
            for (name, _), gradient in zip(named, single_gradients):
                if gradient is not None:
                    sensitivity[layer_index[layer_map[name]]] += (
                        float(
                            (
                                gradient.double().square()
                                * deltas[name].double().square()
                            ).sum()
                        )
                        / sample_total
                    )
    squared_mean = np.zeros(len(layer_index), dtype=np.float64)
    for name, value in gradient_means.items():
        squared_mean[layer_index[layer_map[name]]] += float(
            value.double().square().sum().item()
        )
    return (sensitivity, squared_mean, gradient_sq)


def private_first_tensor(value: Any) -> Any | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = private_first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = private_first_tensor(item)
            if found is not None:
                return found
    return None


def private_representation_statistics(
    model: Any,
    batches: Sequence[Mapping[str, Any]],
    base_state: Mapping[str, Any],
    endpoint_state: Mapping[str, Any],
    layer_map: Mapping[str, str],
    layer_index: Mapping[str, int],
    device: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    numerator = np.zeros(len(layer_index), dtype=np.float64)
    denominator = np.zeros(len(layer_index), dtype=np.float64)
    hook_calls = np.zeros(len(layer_index), dtype=np.int64)
    handles = []
    for module_name, module in model.named_modules():
        key = f"{module_name}.weight" if module_name else "weight"
        if (
            key not in layer_map
            or not hasattr(module, "weight")
            or module.weight.ndim != 2
        ):
            continue
        delta = (endpoint_state[key] - base_state[key]).to(
            device=device, dtype=torch.float32
        )
        if not bool(torch.count_nonzero(delta)):
            continue
        index = layer_index[layer_map[key]]
        delta_sq = float(delta.double().square().sum().item())

        def hook(_module, inputs, *, _delta=delta, _delta_sq=delta_sq, _index=index):
            activation = private_first_tensor(inputs)
            if activation is None or activation.numel() == 0:
                return
            flat = activation.detach().reshape(-1, activation.shape[-1]).float()
            if flat.shape[-1] != _delta.shape[-1]:
                return
            effect = flat @ _delta.T
            numerator[_index] += float(effect.double().square().sum().item())
            denominator[_index] += (
                float(flat.double().square().sum().item()) * _delta_sq
            )
            hook_calls[_index] += 1

        handles.append(module.register_forward_pre_hook(hook))
    try:
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                model(**private_device_batch(batch, device))
    finally:
        for handle in handles:
            handle.remove()
    return (
        numerator,
        denominator,
        {
            "registered_hooks": len(handles),
            "total_hook_calls": int(hook_calls.sum()),
            "layers_with_evidence": int(np.count_nonzero(hook_calls)),
        },
    )


def private_utility_statistics(
    model: Any,
    batches: Sequence[Mapping[str, Any]],
    base_state: Mapping[str, Any],
    endpoint_state: Mapping[str, Any],
    layer_map: Mapping[str, str],
    layer_names: Sequence[str],
    device: Any,
) -> tuple[float, np.ndarray]:
    baseline = private_mean_loss(model, batches, device)
    mutable = {
        **private_named_parameters_all(model),
        **private_named_buffers_all(model),
    }
    by_layer = {
        layer: [
            key for key, owner in layer_map.items() if owner == layer and key in mutable
        ]
        for layer in layer_names
    }
    reverted = np.empty(len(layer_names), dtype=np.float64)
    with torch.no_grad():
        for index, layer in enumerate(layer_names):
            keys = by_layer[layer]
            try:
                for key in keys:
                    mutable[key].copy_(
                        base_state[key].to(
                            device=mutable[key].device, dtype=mutable[key].dtype
                        )
                    )
                reverted[index] = private_mean_loss(model, batches, device)
            finally:
                for key in keys:
                    mutable[key].copy_(
                        endpoint_state[key].to(
                            device=mutable[key].device, dtype=mutable[key].dtype
                        )
                    )
    return (baseline, reverted)


def private_load_dataset_dict(path: Path) -> Any:
    from datasets import Dataset, DatasetDict, load_from_disk

    if (path / "dataset_dict.json").is_file() or (path / "state.json").is_file():
        loaded = load_from_disk(str(path))
    else:
        files = finetune_parquet_split_files(path)
        loaded = DatasetDict(
            {
                split: Dataset.from_parquet([str(item) for item in paths])
                for split, paths in files.items()
            }
        )
    if not isinstance(loaded, DatasetDict) or "train" not in loaded:
        raise ValueError("private-type input must contain a train split")
    return loaded


def private_collect(args: argparse.Namespace) -> int:
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        set_seed,
    )

    started = time.time()
    task = canonical_task_name(args.task)
    base_path = require_local_path(args.base_model_dir, kind="base model")
    endpoint_path = require_local_path(args.endpoint_dir, kind="endpoint model")
    tokenizer_path = require_local_path(args.tokenizer_dir, kind="tokenizer")
    dataset_root = require_local_path(args.dataset_dir, kind="GLUE dataset root")
    dataset_path = finetune_dataset_path(dataset_root, task)
    output = Path(args.output).expanduser().resolve()
    if args.split != "train":
        raise ValueError("private types are restricted to the train split")
    if args.max_batches < 1 or args.max_samples < 1 or args.batch_size < 1:
        raise ValueError("batch/sample limits must be positive")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        str(base_path), local_files_only=True, torch_dtype=torch.float32
    )
    base_state = private_cpu_state(base_model)
    del base_model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(endpoint_path), local_files_only=True, torch_dtype=torch.float32
    ).to(args.device)
    endpoint_state = private_cpu_state(model)
    if set(base_state) != set(endpoint_state):
        raise ValueError("base and endpoint state keys differ")
    layer_map = private_layer_map(endpoint_state)
    layer_names = tuple(dict.fromkeys(layer_map.values()))
    layer_index = {name: index for index, name in enumerate(layer_names)}
    loaded = private_load_dataset_dict(dataset_path)
    raw_train = loaded["train"].shuffle(seed=args.seed)
    raw_train = raw_train.select(range(min(args.max_samples, len(raw_train))))
    token_args = argparse.Namespace(
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    tokenized = finetune_tokenize_dataset(raw_train, tokenizer, task, token_args)
    loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="max_length",
            max_length=args.max_source_length,
            label_pad_token_id=-100,
        ),
    )
    batches = []
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        batches.append({key: value.detach().cpu() for key, value in batch.items()})
        sample_count += int(batch["input_ids"].shape[0])
    if not batches:
        raise ValueError("private calibration loader is empty")
    device = torch.device(args.device)
    sensitivity_raw, stability_num, stability_den = private_gradient_statistics(
        model, batches, base_state, endpoint_state, layer_map, layer_index, device
    )
    try:
        model.load_state_dict(base_state, strict=True)
        representation_num, representation_den, hook_info = (
            private_representation_statistics(
                model,
                batches,
                base_state,
                endpoint_state,
                layer_map,
                layer_index,
                device,
            )
        )
    finally:
        model.load_state_dict(endpoint_state, strict=True)
    endpoint_loss, reverted_losses = private_utility_statistics(
        model, batches, base_state, endpoint_state, layer_map, layer_names, device
    )
    sensitivity = sensitivity_raw / (args.sensitivity_tau + sensitivity_raw)
    relevance = representation_num / (representation_den + args.epsilon)
    utility_delta = (reverted_losses - endpoint_loss) / (
        args.utility_temperature * (abs(endpoint_loss) + args.epsilon)
    )
    utility = 1.0 / (1.0 + np.exp(-np.clip(utility_delta, -60.0, 60.0)))
    stability = stability_num / (stability_den + args.epsilon)
    values = np.stack((sensitivity, relevance, utility, stability), axis=1)
    if not np.isfinite(values).all():
        raise FloatingPointError("Private measurements contain nonfinite values")
    values = np.clip(values, 1e-5, 1.0 - 1e-5)
    source = {
        "base_model_dir": str(base_path),
        "endpoint_dir": str(endpoint_path),
        "tokenizer_dir": str(tokenizer_path),
        "dataset_dir": str(dataset_path),
        "base_inventory": directory_inventory(base_path),
        "endpoint_inventory": directory_inventory(endpoint_path),
        "dataset_inventory": directory_inventory(dataset_path),
    }
    request = {
        "task": task,
        "step": args.step,
        "seed": args.seed,
        "split": "train",
        "max_batches": args.max_batches,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "sensitivity_tau": args.sensitivity_tau,
        "utility_temperature": args.utility_temperature,
        "epsilon": args.epsilon,
        "source": source,
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "glue_layerwise_private_types",
        "complete": True,
        "test_data_used": False,
        "fusionbench_commit": FUSIONBENCH_COMMIT,
        "task": task,
        "step": args.step,
        "split": "train",
        "component_order": list(private_COMPONENTS),
        "layer_names": list(layer_names),
        "values": values.tolist(),
        "raw_statistics": {
            "sensitivity_fisher_weighted_update": sensitivity_raw.tolist(),
            "representation_effect_squared": representation_num.tolist(),
            "representation_bound": representation_den.tolist(),
            "endpoint_loss": endpoint_loss,
            "layer_reverted_losses": reverted_losses.tolist(),
            "utility_loss_increase": (reverted_losses - endpoint_loss).tolist(),
            "squared_mean_gradient": stability_num.tolist(),
            "mean_squared_gradient": stability_den.tolist(),
            "representation_hooks": hook_info,
        },
        "sample_count": sample_count,
        "batch_count": len(batches),
        "request": request,
        "request_sha256": stable_json_hash(request),
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "device": str(device),
            "torch": torch.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    atomic_write_json(output, payload)
    return 0


def private_parse_task_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--input must use TASK=/absolute/path.json")
    raw_task, raw_path = value.split("=", 1)
    return (
        canonical_task_name(raw_task),
        require_local_path(
            raw_path, kind=f"private types for {raw_task}", directory=False
        ),
    )


def private_pack(args: argparse.Namespace) -> int:
    entries = dict((private_parse_task_path(value) for value in args.input))
    missing = sorted(set(GLUE_TASKS) - set(entries))
    extra = sorted(set(entries) - set(GLUE_TASKS))
    if missing or extra or len(args.input) != len(GLUE_TASKS):
        raise ValueError(
            f"pack needs one file per GLUE task; missing={missing}, extra={extra}"
        )
    rows = []
    layers: list[str] | None = None
    receipts = {}
    for task in GLUE_TASKS:
        path = entries[task]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("artifact_type") != "glue_layerwise_private_types"
            or payload.get("complete") is not True
            or payload.get("test_data_used") is not False
        ):
            raise ValueError(f"invalid/incomplete private-type cache: {path}")
        if canonical_task_name(payload.get("task")) != task:
            raise ValueError(f"task mismatch in {path}")
        if int(payload.get("step", -1)) != args.step:
            raise ValueError(f"step mismatch in {path}")
        if tuple(payload.get("component_order", ())) != private_COMPONENTS:
            raise ValueError(f"component order mismatch in {path}")
        current_layers = [str(value) for value in payload["layer_names"]]
        if layers is None:
            layers = current_layers
        elif layers != current_layers:
            raise ValueError(f"logical layer mismatch in {path}")
        values = np.asarray(payload["values"], dtype=np.float64)
        if values.shape != (len(current_layers), len(private_COMPONENTS)):
            raise ValueError(f"private type shape mismatch in {path}: {values.shape}")
        if (
            not np.isfinite(values).all()
            or np.any(values <= 0.0)
            or np.any(values >= 1.0)
        ):
            raise ValueError(
                f"private values outside calibrated open interval in {path}"
            )
        rows.append(values)
        receipts[task] = {"path": str(path), "sha256": file_sha256(path)}
    assert layers is not None
    output = {
        "schema_version": 1,
        "artifact_type": "packed_glue_private_types",
        "complete": True,
        "test_data_used": False,
        "fusionbench_commit": FUSIONBENCH_COMMIT,
        "step": args.step,
        "model_names": list(GLUE_TASKS),
        "layer_names": layers,
        "component_order": list(private_COMPONENTS),
        "private_types": np.stack(rows, axis=0).tolist(),
        "task_artifacts": receipts,
    }
    atomic_write_json(args.output, output)
    return 0


def private_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--task", required=True)
    collect_parser.add_argument("--step", type=int, required=True)
    collect_parser.add_argument("--base-model-dir", required=True)
    collect_parser.add_argument("--endpoint-dir", required=True)
    collect_parser.add_argument("--tokenizer-dir", required=True)
    collect_parser.add_argument("--dataset-dir", required=True)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--cache-root", required=True)
    collect_parser.add_argument("--split", choices=("train",), default="train")
    collect_parser.add_argument("--seed", type=int, default=42)
    collect_parser.add_argument("--device", default="cuda")
    collect_parser.add_argument("--max-batches", type=int, default=4)
    collect_parser.add_argument("--max-samples", type=int, default=128)
    collect_parser.add_argument("--batch-size", type=int, default=8)
    collect_parser.add_argument("--num-workers", type=int, default=4)
    collect_parser.add_argument("--max-source-length", type=int, default=512)
    collect_parser.add_argument("--max-target-length", type=int, default=512)
    collect_parser.add_argument("--sensitivity-tau", type=float, default=1e-10)
    collect_parser.add_argument("--utility-temperature", type=float, default=1.0)
    collect_parser.add_argument("--epsilon", type=float, default=1e-08)
    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("--step", type=int, required=True)
    pack_parser.add_argument("--input", action="append", required=True)
    pack_parser.add_argument("--output", required=True)
    return parser


def private_main(argv: Sequence[str] | None = None) -> int:
    args = private_build_parser().parse_args(argv)
    if args.command == "collect":
        configure_offline_environment(args.cache_root)
        return private_collect(args)
    return private_pack(args)


def metrics_finite_unit_score(value: Any, *, label: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(score):
        raise ValueError(f"{label} is not finite")
    if 1.0 < abs(score) <= 100.0:
        score /= 100.0
    lower = -1.0 if "stsb" in label.lower() else 0.0
    if not lower <= score <= 1.0:
        raise ValueError(f"{label} is outside its metric range: {value!r}")
    return score


def metrics_find_task_payload(report: Mapping[str, Any], task: str) -> Any:
    aliases = (task, f"glue-{task}")
    if task == "sst2":
        aliases += ("sst-2", "glue-sst-2")
    elif task == "stsb":
        aliases += ("sts-b", "glue-sts-b")
    containers: list[Mapping[str, Any]] = [report]
    for key in ("tasks", "per_task", "results", "report"):
        candidate = report.get(key)
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    for container in containers:
        for alias in aliases:
            if alias in container:
                return container[alias]
    raise KeyError(f"report is missing required task {task!r}")


def metrics_extract_task_scores(report: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for task in GLUE_TASKS:
        payload = metrics_find_task_payload(report, task)
        if isinstance(payload, Mapping):
            metric_names = (
                ("spearman_rho", "spearman", "score", "raw")
                if task == "stsb"
                else ("accuracy", "exact_match", "score", "raw")
            )
            for metric in metric_names:
                if metric in payload:
                    payload = payload[metric]
                    break
            else:
                raise KeyError(
                    f"task {task!r} has no supported metric; keys={list(payload)}"
                )
        result[task] = metrics_finite_unit_score(payload, label=f"{task} score")
    return result


def metrics_extract_reference_scores(reference: Mapping[str, Any]) -> dict[str, float]:
    if "per_task" in reference and isinstance(reference["per_task"], Mapping):
        per_task = reference["per_task"]
        values: dict[str, float] = {}
        for raw_name, payload in per_task.items():
            try:
                task = canonical_task_name(raw_name)
            except ValueError:
                continue
            if isinstance(payload, Mapping):
                for key in ("raw", "score", "percent"):
                    if key in payload:
                        payload = payload[key]
                        break
            values[task] = metrics_finite_unit_score(payload, label=f"reference {task}")
        if set(values) == set(GLUE_TASKS):
            return {task: values[task] for task in GLUE_TASKS}
    try:
        return metrics_extract_task_scores(reference)
    except KeyError:
        pass
    values = {}
    for raw_name, payload in reference.items():
        try:
            task = canonical_task_name(raw_name)
        except ValueError:
            continue
        if isinstance(payload, Mapping):
            for key in ("raw", "score", "percent"):
                if key in payload:
                    payload = payload[key]
                    break
        values[task] = metrics_finite_unit_score(payload, label=f"reference {task}")
    missing = sorted(set(GLUE_TASKS) - set(values))
    if missing:
        raise KeyError(f"fine-tuned reference is missing tasks: {missing}")
    return {task: values[task] for task in GLUE_TASKS}


def metrics_pinned_reference_scores() -> dict[str, float]:
    raise ValueError("Supply measured individual-model reference scores explicitly")


def metrics_summarize_glue_report(
    report: Mapping[str, Any],
    reference: Mapping[str, Any] | None = None,
    *,
    run_id: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    scores = metrics_extract_task_scores(report)
    references = (
        metrics_pinned_reference_scores()
        if reference is None
        else metrics_extract_reference_scores(reference)
    )
    per_task: dict[str, Any] = {}
    for task in GLUE_TASKS:
        denominator = references[task]
        if denominator <= 0.0:
            raise ValueError(f"reference score for {task} must be positive")
        raw = scores[task]
        per_task[task] = {
            "metric": "spearman_rho" if task == "stsb" else "accuracy",
            "raw": raw,
            "percent": raw * 100.0,
            "fine_tuned_reference_raw": denominator,
            "fine_tuned_reference_percent": denominator * 100.0,
            "normalized_percent": raw / denominator * 100.0,
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "fusionbench_commit": FUSIONBENCH_COMMIT,
            "task_order": list(GLUE_TASKS),
            "normalization": "macro mean of 100 * merged_task_score / single_task_finetuned_score",
        },
        "per_task": per_task,
        "macro_percent": statistics.fmean(
            (per_task[task]["percent"] for task in GLUE_TASKS)
        ),
        "macro_normalized_percent": statistics.fmean(
            (per_task[task]["normalized_percent"] for task in GLUE_TASKS)
        ),
        "source_report": report,
    }
    if run_id is not None:
        summary["run_id"] = str(run_id)
    if seed is not None:
        summary["seed"] = int(seed)
    return summary


def metrics_select_median_seed(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not summaries or len(summaries) % 2 == 0:
        raise ValueError("median-seed selection requires a non-empty odd run count")
    checked: list[dict[str, Any]] = []
    for index, item in enumerate(summaries):
        macro = float(item["macro_percent"])
        if not math.isfinite(macro):
            raise ValueError(f"run {index} has a non-finite macro score")
        missing = [task for task in GLUE_TASKS if task not in item["per_task"]]
        if missing:
            raise KeyError(f"run {index} is missing tasks: {missing}")
        checked.append(dict(item))
    ordered = sorted(
        enumerate(checked),
        key=lambda pair: (
            float(pair[1]["macro_percent"]),
            str(pair[1].get("seed", pair[1].get("run_id", pair[0]))),
        ),
    )
    source_index, representative = ordered[len(ordered) // 2]
    elementwise = {
        task: {
            "percent": statistics.median(
                (float(run["per_task"][task]["percent"]) for run in checked)
            ),
            "normalized_percent": statistics.median(
                (float(run["per_task"][task]["normalized_percent"]) for run in checked)
            ),
        }
        for task in GLUE_TASKS
    }
    return {
        "schema_version": 1,
        "selection": "actual run with median macro_percent",
        "run_count": len(checked),
        "representative_source_index": source_index,
        "representative_seed": representative.get("seed"),
        "representative_run_id": representative.get("run_id"),
        "macro_median_percent": statistics.median(
            (float(run["macro_percent"]) for run in checked)
        ),
        "macro_normalized_median_percent": statistics.median(
            (float(run["macro_normalized_percent"]) for run in checked)
        ),
        "paper_result": representative,
        "elementwise_median_diagnostic": elementwise,
        "all_runs": checked,
    }


def metrics_load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def metrics_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    return parser


def metrics_main(argv: Sequence[str] | None = None) -> int:
    args = metrics_build_parser().parse_args(argv)
    if args.seed is not None and len(args.seed) != len(args.report):
        raise ValueError("--seed count must match --report count")
    reference = metrics_load_json(args.reference) if args.reference else None
    summaries = [
        metrics_summarize_glue_report(
            metrics_load_json(path),
            reference,
            run_id=str(Path(path).resolve()),
            seed=None if args.seed is None else args.seed[index],
        )
        for index, path in enumerate(args.report)
    ]
    output = (
        summaries[0] if len(summaries) == 1 else metrics_select_median_seed(summaries)
    )
    atomic_write_json(args.output, output)
    return 0


def evaluate_sequence_model(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    task: str,
    *,
    batch_size: int = 32,
    device: str = "cpu",
    max_source_length: int = 512,
    max_new_tokens: int = 10,
    save_predictions: str | Path | None = None,
) -> dict:
    from torch.utils.data import DataLoader

    task = canonical_task_name(task)
    if batch_size < 1 or max_source_length < 1 or max_new_tokens < 1:
        raise ValueError("Evaluation budgets must be positive")
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty")

    def collate(rows):
        examples = [format_example(task, row) for row in rows]
        if any(not target for _, target in examples):
            raise ValueError("Evaluation requires labeled examples")
        return [source for source, _ in examples], [target for _, target in examples]

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    modes = [(module, module.training) for module in model.modules()]
    model.to(device).eval()
    predictions, targets = [], []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for sources, labels in loader:
                inputs = tokenizer(
                    sources,
                    padding=True,
                    truncation=True,
                    max_length=max_source_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
                predictions.extend(
                    tokenizer.batch_decode(generated, skip_special_tokens=True)
                )
                targets.extend(labels)
    finally:
        for module, training in modes:
            module.training = training
    metrics = finetune_decode_metrics(task, predictions, targets)
    metrics.update(
        task=task,
        evaluated_examples=len(targets),
        elapsed_seconds=time.perf_counter() - started,
    )
    if save_predictions is not None:
        atomic_write_json(
            save_predictions,
            {
                "task": task,
                "predictions": predictions,
                "targets": targets,
                "metrics": metrics,
            },
        )
    return metrics


def evaluate_glue(
    model: Any,
    tokenizer: Any,
    datasets: Mapping[str, Any],
    *,
    device: str = "cpu",
    batch_size: int = 32,
    output: str | Path | None = None,
) -> dict:
    if set(datasets) != set(GLUE_TASKS):
        raise ValueError("All eight GLUE tasks must be supplied")
    reports = {}
    for task in GLUE_TASKS:
        reports[task] = evaluate_sequence_model(
            model, tokenizer, datasets[task], task, batch_size=batch_size, device=device
        )
        if output is not None:
            atomic_write_json(output, {"complete": False, "tasks": reports})
    scores = [
        reports[task]["spearman_rho" if task == "stsb" else "accuracy"]
        for task in GLUE_TASKS
    ]
    result = {"complete": True, "tasks": reports, "average": statistics.fmean(scores)}
    if output is not None:
        atomic_write_json(output, result)
    return result


def merge_sequence_models(
    base_directory: str | Path,
    task_directories: Mapping[str, str | Path],
    private_artifact: str | Path,
    belief_artifact: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 0,
    steps: int = 2000,
    samples: int = 128,
    rank: int = 4,
    device: str = "cpu",
) -> dict:
    from transformers import AutoModelForSeq2SeqLM

    from .__main__ import atomic_checkpoint, safe_load
    from .beliefs import PriorParameters, PublicPredictor
    from .merging import BeliefMerge, InformationConfig
    from .strategies import StrategyConfig

    root = Path(output_directory)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Sequence merge output must be empty")
    normalized = {
        canonical_task_name(task): path for task, path in task_directories.items()
    }
    if set(normalized) != set(GLUE_TASKS) or len(task_directories) != len(GLUE_TASKS):
        raise ValueError("Provide exactly one model for every GLUE task")
    calibration = safe_load(str(belief_artifact))
    if calibration.get("split") != "prior_calibration":
        raise ValueError("Belief artifact is not prior calibration")
    if not calibration.get("model_ids") or set(calibration["model_ids"]) & set(
        GLUE_TASKS
    ):
        raise ValueError(
            "Belief calibration identities must be separate from evaluated tasks"
        )
    base_path = require_local_path(base_directory, kind="pretrained sequence model")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(base_path), local_files_only=True
    ).cpu()
    base = private_cpu_state(model)
    layers = logical_layer_names(base)
    scores, metadata = load_private_types(
        private_artifact, model_names=GLUE_TASKS, layer_names=layers
    )
    score_tensor = torch.as_tensor(scores).permute(0, 2, 1).float()
    if score_tensor.shape[1] != 3:
        score_tensor = score_tensor[:, :3]
    states = []
    for task in GLUE_TASKS:
        path = require_local_path(normalized[task], kind=f"sequence model {task}")
        endpoint = AutoModelForSeq2SeqLM.from_pretrained(
            str(path), local_files_only=True
        ).cpu()
        states.append(private_cpu_state(endpoint))
        del endpoint
    predictor = PublicPredictor(**calibration["predictor_config"])
    predictor.load_state_dict(calibration["predictor"])
    predictor.to(device).eval().requires_grad_(False)
    prior = PriorParameters(**calibration["prior"])
    layer_map = {
        key: flan_t5_layer_name(key)
        for key, value in base.items()
        if value.is_floating_point()
    }
    root.mkdir(parents=True, exist_ok=True)
    strategy = StrategyConfig(seed=seed, steps=steps, samples=samples)
    merger = BeliefMerge(InformationConfig(rank=rank), strategy, device=device)
    result = merger.merge(
        base,
        states,
        score_tensor,
        predictor,
        prior,
        layer_map=layer_map,
        checkpoint=lambda step, payload: atomic_checkpoint(
            root / f"strategy_step_{step:06d}.pt", payload
        ),
    )
    model.load_state_dict(result.state_dict, strict=True)
    model.tie_weights()
    model.save_pretrained(root / "merged_model", safe_serialization=True)
    diagnostics = result.metadata()
    diagnostics.update(model_names=list(GLUE_TASKS), private_artifact=metadata)
    atomic_write_json(root / "metrics.json", diagnostics)
    return diagnostics


def sequence_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beliefmerge language-merge")
    parser.add_argument("--base", required=True)
    parser.add_argument("--task-model", action="append", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--belief", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args(argv)
    pairs = []
    for value in args.task_model:
        if "=" not in value:
            raise ValueError("Each model must use TASK=PATH")
        pairs.append(value.split("=", 1))
    if len(dict(pairs)) != len(pairs):
        raise ValueError("Duplicate task model")
    merge_sequence_models(
        args.base,
        dict(pairs),
        args.private,
        args.belief,
        args.output,
        seed=args.seed,
        steps=args.steps,
        samples=args.samples,
        rank=args.rank,
        device=args.device,
    )
    return 0
