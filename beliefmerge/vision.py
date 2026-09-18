from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .__main__ import atomic_checkpoint

PRIVATE_TYPE_COMPONENTS = ("s_sens", "s_repr", "s_util", "s_stab")
SCHEMA_VERSION = 1


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def resolve_logical_layer(parameter_name: str) -> str:
    pieces = str(parameter_name).split(".")
    repeated_markers = {"resblocks", "blocks", "layers", "layer"}
    for index, piece in enumerate(pieces[:-1]):
        if (
            piece in repeated_markers
            and index + 1 < len(pieces)
            and pieces[index + 1].isdigit()
        ):
            return ".".join(pieces[: index + 2])
    if len(pieces) > 1:
        return ".".join(pieces[:-1])
    return pieces[0]


def build_layer_map(state_dict: Mapping[str, Any]) -> OrderedDict[str, str]:
    output: OrderedDict[str, str] = OrderedDict()
    for key, value in state_dict.items():
        is_floating = (
            bool(value.is_floating_point())
            if hasattr(value, "is_floating_point")
            else np.asarray(value).dtype.kind in "fc"
        )
        if is_floating:
            output[str(key)] = resolve_logical_layer(str(key))
    if not output:
        raise ValueError("state_dict contains no floating-point values")
    return output


def ordered_layer_names(layer_map: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((str(value) for value in layer_map.values())))


@dataclass(frozen=True)
class PrivateTypeConfig:
    max_batches: int = 4
    max_samples: int = 256
    sensitivity_tau: float = 1e-10
    utility_temperature: float = 1.0
    epsilon: float = 1e-08
    fisher_estimator: str = "per_example_empirical"

    def __post_init__(self) -> None:
        if self.max_batches < 1 or self.max_samples < 1:
            raise ValueError("max_batches and max_samples must be positive")
        if self.sensitivity_tau <= 0.0:
            raise ValueError("sensitivity_tau must be positive")
        if self.utility_temperature <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("utility_temperature/epsilon must be positive")
        if self.fisher_estimator != "per_example_empirical":
            raise ValueError("Only per_example_empirical Fisher is supported")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrivateTypeResult:
    task: str
    layer_names: tuple[str, ...]
    values: np.ndarray
    raw_statistics: Mapping[str, Any]
    sample_count: int
    batch_count: int
    split: str
    elapsed_seconds: float
    source: Mapping[str, Any]
    config: PrivateTypeConfig

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (len(self.layer_names), len(PRIVATE_TYPE_COMPONENTS)):
            raise ValueError(
                f"private values must have shape [logical_layer,4], got {values.shape}"
            )
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError("private values must be finite and lie in [0,1]")
        object.__setattr__(self, "values", values)

    def matrix(self, *, include_stability: bool = False) -> np.ndarray:
        width = 4 if include_stability else 3
        return np.clip(self.values[:, :width], 1e-05, 1.0 - 1e-05).copy()

    def to_dict(self) -> dict[str, Any]:
        per_layer = {}
        for index, layer in enumerate(self.layer_names):
            per_layer[layer] = {
                name: float(self.values[index, component])
                for component, name in enumerate(PRIVATE_TYPE_COMPONENTS)
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "layerwise_private_types",
            "complete": True,
            "test_data_used": False,
            "task": self.task,
            "split": self.split,
            "component_order": list(PRIVATE_TYPE_COMPONENTS),
            "layer_names": list(self.layer_names),
            "values": self.values.tolist(),
            "per_layer": per_layer,
            "raw_statistics": dict(self.raw_statistics),
            "sample_count": int(self.sample_count),
            "batch_count": int(self.batch_count),
            "elapsed_seconds": float(self.elapsed_seconds),
            "source": dict(self.source),
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrivateTypeResult":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("private-type cache schema version differs")
        if payload.get("artifact_type") != "layerwise_private_types":
            raise ValueError("not a layerwise private-type artifact")
        if (
            payload.get("complete") is not True
            or payload.get("test_data_used") is not False
        ):
            raise ValueError("private-type cache is incomplete or test-contaminated")
        if tuple(payload.get("component_order", ())) != PRIVATE_TYPE_COMPONENTS:
            raise ValueError("private-type component order differs")
        return cls(
            task=str(payload["task"]),
            layer_names=tuple((str(value) for value in payload["layer_names"])),
            values=np.asarray(payload["values"], dtype=np.float64),
            raw_statistics=dict(payload["raw_statistics"]),
            sample_count=int(payload["sample_count"]),
            batch_count=int(payload["batch_count"]),
            split=str(payload["split"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            source=dict(payload["source"]),
            config=PrivateTypeConfig(**dict(payload["config"])),
        )


def load_private_type_cache(
    path: Path,
    *,
    expected_task: str,
    expected_layers: Sequence[str] | None = None,
    expected_source: Mapping[str, Any] | None = None,
    expected_config: PrivateTypeConfig | None = None,
) -> PrivateTypeResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = PrivateTypeResult.from_dict(payload)
    if result.task != expected_task:
        raise ValueError(f"private cache task differs: {path}")
    if expected_layers is not None and result.layer_names != tuple(expected_layers):
        raise ValueError(f"private cache layer order differs: {path}")
    if expected_source is not None:
        for key, value in expected_source.items():
            if result.source.get(key) != value:
                raise ValueError(f"private cache source {key!r} differs: {path}")
    if expected_config is not None and result.config != expected_config:
        raise ValueError(f"private cache configuration differs: {path}")
    return result


def _batch_xy(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        x_key = next(
            (key for key in ("x", "image", "images", "pixel_values") if key in batch),
            None,
        )
        y_key = next(
            (key for key in ("y", "label", "labels", "target") if key in batch), None
        )
        if x_key is None or y_key is None:
            raise KeyError("private-type batch lacks image or label tensors")
        return (batch[x_key], batch[y_key])
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return (batch[0], batch[1])
    raise TypeError(f"unsupported private-type batch: {type(batch).__name__}")


def materialize_calibration_batches(
    loader: Any, config: PrivateTypeConfig
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    output: list[tuple[torch.Tensor, torch.Tensor]] = []
    samples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= config.max_batches or samples >= config.max_samples:
            break
        images, labels = _batch_xy(batch)
        remaining = config.max_samples - samples
        if len(labels) > remaining:
            images = images[:remaining]
            labels = labels[:remaining]
        output.append((images.detach().cpu(), labels.detach().cpu()))
        samples += int(labels.numel())
    if not output or samples == 0:
        raise ValueError("private-type calibration loader is empty")
    return tuple(output)


def _model_output_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if (
        isinstance(value, (tuple, list))
        and value
        and isinstance(value[0], torch.Tensor)
    ):
        return value[0]
    if isinstance(value, Mapping):
        for key in ("logits", "image_embeds", "features"):
            if isinstance(value.get(key), torch.Tensor):
                return value[key]
    raise TypeError(f"model output is not tensor-like: {type(value).__name__}")


def _logits(
    encoder: torch.nn.Module, head: torch.nn.Module, images: torch.Tensor
) -> torch.Tensor:
    return _model_output_tensor(head(_model_output_tensor(encoder(images)))).float()


@torch.inference_mode()
def _loss_and_accuracy(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, int]:
    encoder.eval()
    head.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images_cpu, labels_cpu in batches:
        images = images_cpu.to(device=device, non_blocking=True)
        labels = labels_cpu.to(device=device, non_blocking=True)
        logits = _logits(encoder, head, images)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(dim=-1) == labels).sum().item())
        count += int(labels.numel())
    if count == 0:
        raise ValueError("cannot measure an empty calibration set")
    return (loss_sum / count, correct / count, count)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for child in value:
            found = _first_tensor(child)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for child in value.values():
            found = _first_tensor(child)
            if found is not None:
                return found
    return None


@torch.inference_mode()
def _representation_statistics(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    base_state: Mapping[str, torch.Tensor],
    endpoint_state: Mapping[str, torch.Tensor],
    layer_map: Mapping[str, str],
    layer_index: Mapping[str, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    numerator = np.zeros(len(layer_index), dtype=np.float64)
    denominator = np.zeros(len(layer_index), dtype=np.float64)
    hook_calls = np.zeros(len(layer_index), dtype=np.int64)
    handles = []
    for module_name, module in encoder.named_modules():
        candidates: list[tuple[torch.Tensor, float, int]] = []
        prefix = f"{module_name}." if module_name else ""
        for parameter_name, parameter in module.named_parameters(recurse=False):
            key = prefix + parameter_name
            if key not in layer_map or parameter.ndim != 2:
                continue
            delta = endpoint_state[key].detach() - base_state[key].detach().to(
                endpoint_state[key].device
            )
            if delta.shape[-1] < 1 or not torch.count_nonzero(delta).item():
                continue
            candidates.append(
                (
                    delta.to(device=device, dtype=torch.float32),
                    float(delta.double().square().sum().item()),
                    layer_index[layer_map[key]],
                )
            )
        if not candidates:
            continue

        def hook(
            _module: torch.nn.Module,
            inputs: Any,
            _candidates: tuple[tuple[torch.Tensor, float, int], ...] = tuple(
                candidates
            ),
        ) -> None:
            activation = _first_tensor(inputs)
            if activation is None or activation.numel() == 0:
                return
            flat = activation.detach().reshape(-1, activation.shape[-1]).float()
            activation_sq = float(flat.double().square().sum().item())
            for delta, delta_sq, index in _candidates:
                if flat.shape[-1] != delta.shape[-1]:
                    continue
                effect = flat @ delta.T
                numerator[index] += float(effect.double().square().sum().item())
                denominator[index] += activation_sq * delta_sq
                hook_calls[index] += 1

        handles.append(module.register_forward_pre_hook(hook))
    try:
        encoder.load_state_dict(base_state, strict=True)
        encoder.eval()
        head.eval()
        for images_cpu, _ in batches:
            images = images_cpu.to(device=device, non_blocking=True)
            _logits(encoder, head, images)
    finally:
        for handle in handles:
            handle.remove()
        encoder.load_state_dict(endpoint_state, strict=True)
    return (
        numerator,
        denominator,
        {
            "registered_hooks": len(handles),
            "total_hook_calls": int(hook_calls.sum()),
            "layers_with_affine_evidence": int(np.count_nonzero(hook_calls)),
        },
    )


def _gradient_statistics(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    base_state: Mapping[str, torch.Tensor],
    endpoint_state: Mapping[str, torch.Tensor],
    layer_map: Mapping[str, str],
    layer_index: Mapping[str, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    named_parameters = [
        (name, parameter)
        for name, parameter in encoder.named_parameters()
        if parameter.requires_grad and name in layer_map
    ]
    if not named_parameters:
        raise ValueError("encoder exposes no trainable floating parameters")
    total = sum((int(labels.numel()) for _, labels in batches))
    sensitivity = np.zeros(len(layer_index), dtype=np.float64)
    mean_squared_gradient = np.zeros(len(layer_index), dtype=np.float64)
    gradient_means = {
        name: torch.zeros_like(parameter, memory_format=torch.preserve_format)
        for name, parameter in named_parameters
    }
    encoder.eval()
    head.eval()
    for images_cpu, labels_cpu in batches:
        images = images_cpu.to(device=device, non_blocking=True)
        labels = labels_cpu.to(device=device, non_blocking=True)
        logits = _logits(encoder, head, images)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        gradients = torch.autograd.grad(
            loss, [parameter for _, parameter in named_parameters], allow_unused=True
        )
        weight = int(labels.numel()) / total
        for (name, _), gradient in zip(named_parameters, gradients):
            if gradient is None:
                continue
            current = gradient.detach()
            index = layer_index[layer_map[name]]
            squared_norm = float(current.double().square().sum().item())
            mean_squared_gradient[index] += weight * squared_norm
            gradient_means[name].add_(current, alpha=weight)
        for sample in range(labels.numel()):
            sample_loss = F.cross_entropy(
                _logits(encoder, head, images[sample : sample + 1]),
                labels[sample : sample + 1],
            )
            sample_gradients = torch.autograd.grad(
                sample_loss,
                [parameter for _, parameter in named_parameters],
                allow_unused=True,
            )
            for (name, _), gradient in zip(named_parameters, sample_gradients):
                if gradient is not None:
                    delta = endpoint_state[name].to(gradient) - base_state[name].to(
                        gradient
                    )
                    sensitivity[layer_index[layer_map[name]]] += (
                        float(
                            (gradient.double().square() * delta.double().square()).sum()
                        )
                        / total
                    )
    squared_mean_gradient = np.zeros(len(layer_index), dtype=np.float64)
    for name, value in gradient_means.items():
        index = layer_index[layer_map[name]]
        squared_mean_gradient[index] += float(value.double().square().sum().item())
    return (sensitivity, squared_mean_gradient, mean_squared_gradient)


def _mutable_state(encoder: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {**dict(encoder.named_parameters()), **dict(encoder.named_buffers())}


def _utility_statistics(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    base_state: Mapping[str, torch.Tensor],
    endpoint_state: Mapping[str, torch.Tensor],
    layer_map: Mapping[str, str],
    layer_names: Sequence[str],
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    endpoint_loss, endpoint_accuracy, _ = _loss_and_accuracy(
        encoder, head, batches, device
    )
    objects = _mutable_state(encoder)
    by_layer: dict[str, list[str]] = {name: [] for name in layer_names}
    for key, layer in layer_map.items():
        if key in objects:
            by_layer[layer].append(key)
    reverted_losses = np.empty(len(layer_names), dtype=np.float64)
    with torch.no_grad():
        for index, layer in enumerate(layer_names):
            keys = by_layer[layer]
            try:
                for key in keys:
                    objects[key].copy_(
                        base_state[key],
                        non_blocking=base_state[key].device.type == "cpu",
                    )
                reverted_losses[index], _, _ = _loss_and_accuracy(
                    encoder, head, batches, device
                )
            finally:
                for key in keys:
                    objects[key].copy_(endpoint_state[key])
    return (endpoint_loss, endpoint_accuracy, reverted_losses)


def measure_private_types(
    *,
    task: str,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    loader: Any,
    base_state: Mapping[str, torch.Tensor],
    endpoint_state: Mapping[str, torch.Tensor] | None = None,
    layer_map: Mapping[str, str] | None = None,
    device: str | torch.device = "cuda",
    split: str = "type_cal",
    config: PrivateTypeConfig | None = None,
    source: Mapping[str, Any] | None = None,
) -> PrivateTypeResult:
    split_normalized = split.strip().lower()
    if split_normalized not in {"train", "type_cal"}:
        raise ValueError("private types may only use train or type_cal data")
    cfg = config or PrivateTypeConfig()
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested for private types but is unavailable")
    started = time.perf_counter()
    encoder = encoder.to(resolved_device)
    head = head.to(resolved_device)
    endpoint = {
        key: value.detach().clone()
        for key, value in (endpoint_state or encoder.state_dict()).items()
    }
    mapping = OrderedDict(layer_map or build_layer_map(endpoint))
    layers = ordered_layer_names(mapping)
    indices = {name: index for index, name in enumerate(layers)}
    if set(base_state) != set(endpoint):
        raise ValueError("base and endpoint state keys differ")
    if set(mapping) != {
        key for key, value in endpoint.items() if bool(value.is_floating_point())
    }:
        raise ValueError("layer_map must cover every floating state value exactly")
    for key in endpoint:
        if tuple(base_state[key].shape) != tuple(endpoint[key].shape):
            raise ValueError(f"base/endpoint shape differs for {key}")
    batches = materialize_calibration_batches(loader, cfg)
    representation_num, representation_den, hook_diagnostics = (
        _representation_statistics(
            encoder,
            head,
            batches,
            base_state,
            endpoint,
            mapping,
            indices,
            resolved_device,
        )
    )
    sensitivity_raw, stability_num, stability_den = _gradient_statistics(
        encoder, head, batches, base_state, endpoint, mapping, indices, resolved_device
    )
    endpoint_loss, endpoint_accuracy, reverted_losses = _utility_statistics(
        encoder, head, batches, base_state, endpoint, mapping, layers, resolved_device
    )
    sensitivity = sensitivity_raw / (cfg.sensitivity_tau + sensitivity_raw)
    relevance = representation_num / (representation_den + cfg.epsilon)
    utility_argument = (reverted_losses - endpoint_loss) / (
        cfg.utility_temperature * (abs(endpoint_loss) + cfg.epsilon)
    )
    utility = 1.0 / (1.0 + np.exp(-np.clip(utility_argument, -60.0, 60.0)))
    stability = stability_num / (stability_den + cfg.epsilon)
    values = np.stack((sensitivity, relevance, utility, stability), axis=1)
    if not np.isfinite(values).all():
        raise FloatingPointError("Private measurements contain nonfinite values")
    values = np.clip(values, 1e-5, 1.0 - 1e-5)
    samples = sum((int(labels.numel()) for _, labels in batches))
    raw = {
        "sensitivity_fisher_weighted_update": sensitivity_raw.tolist(),
        "representation_effect_squared": representation_num.tolist(),
        "representation_bound": representation_den.tolist(),
        "endpoint_loss": float(endpoint_loss),
        "endpoint_accuracy": float(endpoint_accuracy),
        "layer_reverted_losses": reverted_losses.tolist(),
        "utility_loss_increase": (reverted_losses - endpoint_loss).tolist(),
        "squared_mean_gradient": stability_num.tolist(),
        "mean_squared_gradient": stability_den.tolist(),
        "representation_hooks": hook_diagnostics,
        "estimator_notes": {
            "s_sens": "per-example empirical Fisher diagonal weighted by squared update",
            "s_repr": "affine-module ||H Delta||_F^2 / (||H||_F^2 ||Delta||_F^2 + epsilon)",
            "s_util": "sigmoid loss increase after reverting one logical layer to base",
            "s_stab": "||E_batch gradient||_2^2 / (E_batch ||gradient||_2^2 + epsilon)",
        },
    }
    return PrivateTypeResult(
        task=str(task),
        layer_names=layers,
        values=values,
        raw_statistics=raw,
        sample_count=samples,
        batch_count=len(batches),
        split=split_normalized,
        elapsed_seconds=time.perf_counter() - started,
        source=dict(source or {}),
        config=cfg,
    )


def save_private_type_cache(path: Path, result: PrivateTypeResult) -> None:
    atomic_json(Path(path), result.to_dict())


@dataclass(frozen=True)
class VisionTrainingConfig:
    max_steps: int = 2000
    batch_size: int = 64
    evaluation_batch_size: int = 128
    learning_rate: float = 1e-05
    weight_decay: float = 0.1
    warmup_steps: int = 0
    accumulation_steps: int = 1
    gradient_clip: float = 1.0
    seed: int = 0
    workers: int = 0
    precision: str = "float32"
    device: str = "cpu"

    def __post_init__(self):
        if (
            min(
                self.max_steps,
                self.batch_size,
                self.evaluation_batch_size,
                self.accumulation_steps,
            )
            < 1
        ):
            raise ValueError("Training sizes must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid optimizer configuration")
        if not 0 <= self.warmup_steps < self.max_steps or self.workers < 0:
            raise ValueError("Invalid warmup or worker count")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("Supported precisions are float32 and bfloat16")


def split_training_dataset(
    dataset: Any,
    calibration_fraction: float = 0.05,
    selection_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[Any, Any, Any]:
    if not 0 < calibration_fraction < 1 or not 0 < selection_fraction < 1:
        raise ValueError("Split fractions must be in (0,1)")
    if calibration_fraction + selection_fraction >= 1 or len(dataset) < 3:
        raise ValueError("Training split would be empty")
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    calibration_count = max(1, int(len(dataset) * calibration_fraction))
    selection_count = max(1, int(len(dataset) * selection_fraction))
    if calibration_count + selection_count >= len(dataset):
        raise ValueError("Dataset is too small for disjoint splits")
    return (
        Subset(dataset, indices[calibration_count + selection_count :]),
        Subset(dataset, indices[:calibration_count]),
        Subset(
            dataset, indices[calibration_count : calibration_count + selection_count]
        ),
    )


@torch.inference_mode()
def evaluate_classifier(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    loader: Any,
    device: str | torch.device = "cpu",
    include_confusion: bool = True,
) -> dict:
    device = torch.device(device)
    modes = [
        (module, module.training)
        for model in (encoder, head)
        for module in model.modules()
    ]
    count, correct, loss_sum = (0, 0, 0.0)
    confusion = None
    try:
        encoder.eval()
        head.eval()
        for batch in loader:
            images, labels = _batch_xy(batch)
            labels = labels.to(device)
            logits = _logits(encoder, head, images.to(device))
            if (
                logits.ndim != 2
                or logits.shape[0] != labels.numel()
                or (not torch.isfinite(logits).all())
            ):
                raise ValueError("Invalid classifier logits")
            predicted = logits.argmax(-1)
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
            count += labels.numel()
            correct += int((predicted == labels).sum())
            if include_confusion:
                classes = logits.shape[-1]
                values = torch.bincount(
                    (labels * classes + predicted).cpu(), minlength=classes * classes
                ).reshape(classes, classes)
                confusion = values if confusion is None else confusion + values
    finally:
        for module, training in modes:
            module.training = training
    if count == 0:
        raise ValueError("Evaluation dataset is empty")
    result = {
        "accuracy": correct / count,
        "loss": loss_sum / count,
        "correct": correct,
        "count": count,
    }
    if confusion is not None:
        result["confusion"] = confusion.tolist()
        totals = confusion.sum(-1)
        result["per_class_accuracy"] = [
            float(confusion[i, i] / totals[i]) if totals[i] else None
            for i in range(len(totals))
        ]
    return result


def _training_seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class EpochSeededDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Any, seed: int):
        self.dataset = dataset
        self.seed = int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        seed = (self.seed + int(index)) % (2**32)
        try:
            random.seed(seed)
            np.random.seed(seed)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def _training_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_training_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state(
        (values[0], np.asarray(values[1], dtype=np.uint32), *values[2:])
    )
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_classifier(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    train_dataset: Any,
    validation_dataset: Any,
    output: str | Path,
    config: VisionTrainingConfig = VisionTrainingConfig(),
    *,
    dataset_id: str,
    resume: bool = True,
    stop_after_step: int | None = None,
) -> dict:
    if not len(train_dataset) or not len(validation_dataset) or (not dataset_id):
        raise ValueError("Nonempty datasets and dataset identity are required")
    terminal = config.max_steps if stop_after_step is None else stop_after_step
    if not 1 <= terminal <= config.max_steps:
        raise ValueError("Invalid stopping step")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("Requested CUDA device is unavailable")
    request = {
        "config": asdict(config),
        "dataset_id": dataset_id,
        "train_count": len(train_dataset),
        "validation_count": len(validation_dataset),
    }
    fingerprint = canonical_json_sha256(request)
    receipt_path = root / "training.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Training configuration changed for an existing output")
        if not resume:
            raise FileExistsError("Training output already exists")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    encoder.to(device)
    head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    parameters = [
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Encoder has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def learning_rate_multiplier(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = (step - config.warmup_steps) / max(
            1, config.max_steps - config.warmup_steps
        )
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    step, epoch, position, microsteps = (0, 0, 0, 0)
    history = []
    latest = root / "latest.pt"
    if latest.exists() and resume:
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint fingerprint differs")
        encoder.load_state_dict(saved["encoder"], strict=True)
        head.load_state_dict(saved["head"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step, epoch, position = (saved["step"], saved["epoch"], saved["position"])
        history = saved["history"]
        _restore_training_rng(saved["rng"])
    loader = DataLoader(
        EpochSeededDataset(validation_dataset, config.seed + 10_000_000),
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        num_workers=config.workers,
        worker_init_fn=_training_seed_worker,
        generator=torch.Generator().manual_seed(config.seed + 1),
    )
    receipt = {
        "request": request,
        "fingerprint": fingerprint,
        "state": "RUNNING",
        "step": step,
    }
    atomic_json(receipt_path, receipt)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    try:
        while step < terminal:
            training_loader = DataLoader(
                EpochSeededDataset(
                    train_dataset, config.seed + epoch * len(train_dataset)
                ),
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.workers,
                worker_init_fn=_training_seed_worker,
                generator=torch.Generator().manual_seed(config.seed + epoch),
            )
            for batch_index, batch in enumerate(training_loader):
                if batch_index < position:
                    continue
                encoder.train()
                images, labels = _batch_xy(batch)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=config.precision == "bfloat16",
                ):
                    logits = _logits(encoder, head, images.to(device))
                    objective = F.cross_entropy(logits, labels.to(device))
                    loss = objective / config.accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                running_loss += float(objective.detach()) / config.accumulation_steps
                microsteps += 1
                position = batch_index + 1
                if microsteps % config.accumulation_steps:
                    continue
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, config.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                history.append(
                    {
                        "step": step,
                        "loss": running_loss,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "gradient_norm": float(norm),
                    }
                )
                running_loss = 0.0
                if step % 50 == 0 or step == terminal:
                    metrics = evaluate_classifier(encoder, head, loader, device)
                    metrics.update(step=step, split="validation", seed=config.seed)
                    atomic_json(root / f"metrics_step_{step:06d}.json", metrics)
                    state = {
                        "fingerprint": fingerprint,
                        "step": step,
                        "epoch": epoch,
                        "position": position,
                        "encoder": encoder.state_dict(),
                        "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "rng": _training_rng_state(),
                        "history": list(history),
                    }
                    atomic_checkpoint(root / f"checkpoint_step_{step:06d}.pt", state)
                    atomic_checkpoint(latest, state)
                    receipt.update(step=step, validation=metrics)
                    atomic_json(receipt_path, receipt)
                    print(f"Training updates: {step}/{config.max_steps}", flush=True)
                if step >= terminal:
                    break
            if position >= len(training_loader):
                epoch += 1
                position = 0
        receipt.update(
            state="COMPLETED" if step == config.max_steps else "PAUSED",
            step=step,
            history=history,
        )
        atomic_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt.update(
            state="FAILED", step=step, error_type=type(error).__name__, error=str(error)
        )
        atomic_json(receipt_path, receipt)
        raise


class NormalizedLinearHead(torch.nn.Module):
    def __init__(self, weights: torch.Tensor, logit_scale: float = 1.0):
        super().__init__()
        if weights.ndim != 2 or not torch.isfinite(weights).all():
            raise ValueError("Classification weights must be a finite matrix")
        self.register_buffer(
            "weights", F.normalize(weights.detach().clone().float(), dim=-1)
        )
        self.logit_scale = float(logit_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logit_scale * F.normalize(features.float(), dim=-1) @ self.weights.T


def load_clip_encoder(
    model_name: str, pretrained: str | Path, device: str = "cpu"
) -> tuple[Any, Any, Any]:
    import open_clip

    path = Path(pretrained)
    if not path.is_file():
        raise FileNotFoundError("Provide a local pretrained OpenCLIP checkpoint")
    model, train_transform, validation_transform = (
        open_clip.create_model_and_transforms(
            model_name, pretrained=str(path), device=device
        )
    )
    return (model.visual, train_transform, validation_transform)


def vision_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beliefmerge vision-train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--train-directory", required=True)
    parser.add_argument("--validation-directory", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-05)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args(argv)
    from torchvision.datasets import ImageFolder

    encoder, train_transform, validation_transform = load_clip_encoder(
        args.model, args.pretrained, args.device
    )
    training = ImageFolder(args.train_directory, transform=train_transform)
    validation = ImageFolder(args.validation_directory, transform=validation_transform)
    if training.class_to_idx != validation.class_to_idx:
        raise ValueError("Training and validation class orders differ")
    artifact = torch.load(args.head, map_location="cpu", weights_only=True)
    head = NormalizedLinearHead(artifact["weights"], artifact.get("logit_scale", 1.0))
    if head.weights.shape[0] != len(training.classes):
        raise ValueError("Classification head does not match dataset classes")
    config = VisionTrainingConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
    )
    train_classifier(
        encoder,
        head,
        training,
        validation,
        args.output,
        config,
        dataset_id=args.dataset_id,
        stop_after_step=args.stop_after_step,
    )
    return 0
