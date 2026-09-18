from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import torch
from tqdm import tqdm

from .__main__ import atomic_checkpoint
from .information import validate_states


def is_matrix(value):
    return value.ndim == 2


@dataclass(frozen=True)
class BaselineConfig:
    method: str = "weight_average"
    scale: float = 1.0
    density: float = 0.2
    rank_fraction: float | None = None
    common_fraction: float = 0.5
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        if self.method not in {
            "weight_average",
            "task_arithmetic",
            "ties",
            "dare_ties",
            "consensus",
            "tsv_m",
            "iso_c",
            "iso_cts",
        }:
            raise ValueError("Unknown static baseline")
        if not 0 < self.density <= 1 or not 0 <= self.common_fraction <= 1:
            raise ValueError("Invalid sparsity or common-subspace fraction")
        if self.rank_fraction is not None and (not 0 < self.rank_fraction <= 1):
            raise ValueError("Rank fraction must be in (0,1]")


def exact_quantile(values: torch.Tensor, quantile: float) -> torch.Tensor:
    if not 0 <= quantile <= 1 or values.numel() == 0:
        raise ValueError("Invalid quantile input")
    flat = values.detach().flatten().float()
    if not torch.isfinite(flat).all():
        raise ValueError("Quantile values must be finite")
    index = quantile * (flat.numel() - 1)
    lower = int(index)
    upper = min(flat.numel() - 1, lower + 1)
    lo = torch.kthvalue(flat, lower + 1).values
    hi = torch.kthvalue(flat, upper + 1).values
    return lo + (hi - lo) * (index - lower)


def task_updates(
    base: Mapping[str, torch.Tensor], tasks: Sequence[Mapping[str, torch.Tensor]]
) -> list[OrderedDict]:
    validate_states(base, tasks)
    return [
        OrderedDict(
            (
                (key, task[key].detach().cpu().float() - value.detach().cpu().float())
                for key, value in base.items()
                if value.is_floating_point()
            )
        )
        for task in tasks
    ]


def sparsity_thresholds(
    updates: Sequence[Mapping[str, torch.Tensor]], density: float
) -> torch.Tensor:
    if not 0 < density <= 1:
        raise ValueError("Density must be in (0,1]")
    return torch.stack(
        [
            exact_quantile(
                torch.cat([value.abs().flatten() for value in task.values()]),
                1 - density,
            )
            for task in updates
        ]
    )


def sign_consensus(updates: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    if updates.shape[0] != thresholds.numel():
        raise ValueError("One threshold per task is required")
    shape = (updates.shape[0],) + (1,) * (updates.ndim - 1)
    retained = updates.abs() >= thresholds.to(updates).reshape(shape)
    trimmed = updates.masked_fill(~retained, 0)
    signs = trimmed.sum(0).sign()
    zero = signs == 0
    if zero.any():
        majority = signs.sum().sign()
        signs = torch.where(
            zero, torch.where(majority == 0, torch.ones_like(majority), majority), signs
        )
    agree = retained & (updates.sign() == signs.unsqueeze(0)) & (updates != 0)
    return updates.masked_fill(~agree, 0).sum(0) / agree.sum(0).clamp_min(1)


def _tsv_matrix(matrices: Sequence[torch.Tensor], fraction: float) -> torch.Tensor:
    decompositions = [
        torch.linalg.svd(matrix, full_matrices=False) for matrix in matrices
    ]
    rank = max(1, int(min(matrices[0].shape) * fraction))
    left = torch.cat([u[:, :rank] for u, _, _ in decompositions], 1)
    values = torch.cat([s[:rank] for _, s, _ in decompositions])
    right = torch.cat([v[:rank] for _, _, v in decompositions], 0)
    left_u, _, left_v = torch.linalg.svd(left, full_matrices=False)
    right_u, _, right_v = torch.linalg.svd(right, full_matrices=False)
    return left_u @ left_v * values @ (right_u @ right_v)


def _iso_cts_matrix(
    matrices: Sequence[torch.Tensor], common_fraction: float
) -> torch.Tensor:
    summed = torch.stack(list(matrices)).sum(0)
    left, values, right = torch.linalg.svd(summed, full_matrices=False)
    available = len(values)
    common_rank = min(available, int(available * common_fraction))
    common_left, common_right = (left[:, :common_rank], right[:common_rank])
    residual_slots = available - common_rank
    per_task, remainder = divmod(residual_slots, len(matrices))
    left_parts, right_parts, value_parts = (
        [common_left],
        [common_right],
        [values[:common_rank]],
    )
    for index, matrix in enumerate(matrices):
        residual = matrix - common_left @ (common_left.T @ matrix)
        u, s, v = torch.linalg.svd(residual, full_matrices=False)
        count = min(len(s), per_task + int(index < remainder))
        if count:
            left_parts.append(u[:, :count])
            value_parts.append(s[:count])
            right_parts.append(v[:count])
    combined_left = torch.cat(left_parts, 1)
    combined_right = torch.cat(right_parts, 0)
    combined_values = torch.cat(value_parts)
    if not combined_values.numel():
        return torch.zeros_like(summed)
    u, _, v = torch.linalg.svd(combined_left, full_matrices=False)
    orthogonal_left = u @ v
    u, _, v = torch.linalg.svd(combined_right, full_matrices=False)
    orthogonal_right = u @ v
    return combined_values.mean() * (orthogonal_left @ orthogonal_right)


@torch.no_grad()
def merge_baseline(
    base: Mapping[str, torch.Tensor],
    tasks: Sequence[Mapping[str, torch.Tensor]],
    config: BaselineConfig = BaselineConfig(),
) -> OrderedDict:
    updates = task_updates(base, tasks)
    device = torch.device(config.device)
    generator = torch.Generator().manual_seed(config.seed)
    if config.method == "dare_ties":
        updates = [
            OrderedDict(
                (
                    (
                        key,
                        value
                        * (
                            torch.rand(value.shape, generator=generator)
                            < config.density
                        )
                        / config.density,
                    )
                    for key, value in task.items()
                )
            )
            for task in updates
        ]
    thresholds = (
        sparsity_thresholds(updates, config.density)
        if config.method in {"ties", "dare_ties"}
        else None
    )
    merged = OrderedDict()
    for key, initial in base.items():
        if not initial.is_floating_point():
            merged[key] = initial.detach().clone()
            continue
        blocks = [task[key].to(device) for task in updates]
        stacked = torch.stack(blocks)
        if config.method == "weight_average":
            delta = stacked.mean(0)
        elif config.method == "task_arithmetic":
            delta = stacked.sum(0)
        elif config.method in {"ties", "dare_ties"}:
            delta = sign_consensus(stacked, thresholds)
        elif config.method == "consensus":
            sign = stacked.sum(0).sign()
            delta = stacked.masked_fill(stacked.sign() != sign, 0).sum(0)
        elif initial.ndim != 2:
            delta = stacked.mean(0)
        elif config.method == "tsv_m":
            delta = _tsv_matrix(blocks, config.rank_fraction or 1 / len(tasks))
        elif config.method == "iso_c":
            u, s, v = torch.linalg.svd(stacked.sum(0), full_matrices=False)
            delta = s.mean() * (u @ v)
        else:
            delta = _iso_cts_matrix(blocks, config.common_fraction)
        merged[key] = (
            initial.to(device=device, dtype=delta.dtype) + config.scale * delta
        ).to(initial)
    return merged


def interpolation_sweep(
    base: Mapping[str, torch.Tensor],
    tasks: Sequence[Mapping[str, torch.Tensor]],
    scales: Sequence[float],
    evaluator: Any,
    config: BaselineConfig,
) -> list[dict]:
    from dataclasses import replace

    if not scales or len(set(scales)) != len(scales):
        raise ValueError("Provide unique interpolation scales")
    results = []
    for scale in scales:
        state = merge_baseline(base, tasks, replace(config, scale=float(scale)))
        metrics = evaluator(state)
        results.append(
            {"method": config.method, "scale": float(scale), "metrics": metrics}
        )
    return results


pylogger = logging.getLogger(__name__)


@torch.no_grad()
def sum_svd(
    ref_state_dict, svd_dicts, device="cuda", non_matrix_params_aggregation="base_model"
):
    aggregated_model_dict = ref_state_dict
    layer_names = list(aggregated_model_dict.keys())
    datasets = list(svd_dicts.keys())
    for layer_name in tqdm(layer_names, desc="Summing SVD"):
        is_matrix = aggregated_model_dict[layer_name].dim() == 2
        new_key = layer_name
        offset = 0
        for i, dataset in enumerate(datasets):
            if "text_projection" in layer_name:
                continue
            if is_matrix:
                delta_layer_svd = svd_dicts[dataset][new_key]
                u, s, v = (
                    delta_layer_svd["u"],
                    delta_layer_svd["s"],
                    delta_layer_svd["v"],
                )
                u, s, v = (u.to(device), s.to(device), v.to(device))
                if i == 0:
                    total_rank = sum(
                        (svd_dicts[d][new_key]["s"].shape[0] for d in datasets)
                    )
                    sum_u = torch.zeros(u.shape[0], total_rank, device=device)
                    sum_s = torch.zeros(total_rank, device=device)
                    sum_v = torch.zeros(total_rank, v.shape[1], device=device)
                rank_i = s.shape[0]
                sum_u[:, offset : offset + rank_i] = u
                sum_s[offset : offset + rank_i] = s
                sum_v[offset : offset + rank_i, :] = v
                offset += rank_i
            else:
                delta_layer = svd_dicts[datasets[i]][new_key]["dim1"].to(device)
                if non_matrix_params_aggregation == "mean":
                    if i == 0:
                        aggregated_model_dict[layer_name] = delta_layer
                    else:
                        aggregated_model_dict[layer_name] += (
                            delta_layer - aggregated_model_dict[layer_name]
                        ) / (i + 1)
                else:
                    aggregated_model_dict[layer_name] = torch.zeros_like(delta_layer)
        if "text_projection" in layer_name or not is_matrix:
            continue
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        aggregated_model_dict[layer_name] = torch.linalg.multi_dot(
            (u_u, v_u, torch.diag(sum_s), u_v, v_v)
        ).to(device)
    return aggregated_model_dict


def compute_svd_and_compress(
    matrix, compress_ratio
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2 or not 0 < compress_ratio <= 1:
        raise ValueError("Compression requires a matrix and ratio in (0,1]")
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    reduced_index_s = max(1, int(s.shape[0] * compress_ratio))
    return (u[:, :reduced_index_s], s[:reduced_index_s], v[:reduced_index_s, :])


def compress_tv(task_dicts, compress_rate: float, compress_ratio_per_task=None):
    with torch.no_grad():
        svd_dict = {}
        for dataset, task_dict in tqdm(
            task_dicts.items(), desc="Computing and compressing SVD"
        ):
            svd_dict[dataset] = {}
            for key, layer in task_dict.items():
                new_key = key
                if is_matrix(layer):
                    current_compress_rate = (
                        compress_ratio_per_task.get(dataset, compress_rate)
                        if compress_ratio_per_task
                        else compress_rate
                    )
                    u, s, v = compute_svd_and_compress(layer, current_compress_rate)
                    svd_dict[dataset][new_key] = {
                        "u": u.detach().cpu(),
                        "s": s.detach().cpu(),
                        "v": v.detach().cpu(),
                    }
                else:
                    svd_dict[dataset][new_key] = {"dim1": layer.detach().cpu()}
        return svd_dict


def get_svd_dict(
    task_dicts,
    datasets,
    svd_path: str,
    compression_factor: float = None,
    compress_ratio_per_task: dict = None,
):
    if (
        not datasets
        or set(datasets) != set(task_dicts)
        or len(set(datasets)) != len(datasets)
    ):
        raise ValueError("Task names must match the supplied update dictionaries")
    factor = float(len(datasets) if compression_factor is None else compression_factor)
    if factor < 1:
        raise ValueError("Compression factor must be at least one")
    digest = hashlib.sha256(
        json.dumps(
            {
                "tasks": list(datasets),
                "factor": factor,
                "per_task": compress_ratio_per_task,
            },
            sort_keys=True,
        ).encode()
    )
    for dataset in datasets:
        for key, value in sorted(task_dicts[dataset].items()):
            digest.update(key.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(
                value.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
    fingerprint = digest.hexdigest()
    path = Path(svd_path)
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached.get("fingerprint") == fingerprint:
            return cached["decompositions"]
    decompositions = compress_tv(task_dicts, 1 / factor, compress_ratio_per_task)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(
        path, {"fingerprint": fingerprint, "decompositions": decompositions}
    )
    return decompositions


def measure_cosine_similarity(delta1: torch.Tensor, delta2: torch.Tensor) -> float:
    d1 = delta1.view(-1)
    d2 = delta2.view(-1)
    dot = torch.dot(d1, d2).item()
    norm1 = torch.norm(d1).item()
    norm2 = torch.norm(d2).item()
    if norm1 < 1e-09 or norm2 < 1e-09:
        return 0.0
    return dot / (norm1 * norm2)


@torch.no_grad()
def sum_svd_no_redundant_tasks_simple(
    ref_state_dict: dict,
    svd_dict: dict,
    device: str = "cuda",
    similarity_threshold: float = 0.2,
):
    aggregated_model_dict = ref_state_dict
    layer_names = list(aggregated_model_dict.keys())
    datasets = list(svd_dict.keys())
    for layer_name in tqdm(layer_names, desc="Summing SVD"):
        new_key = layer_name
        is_layer_matrix = aggregated_model_dict[layer_name].dim() == 2
        offset = 0
        accepted_tasks = []
        accepted_deltas = []
        for i, dataset in enumerate(datasets):
            if "text_projection" in layer_name:
                continue
            if is_layer_matrix:
                delta_layer_svd = svd_dict[dataset][new_key]
                u, s, v = (
                    delta_layer_svd["u"].to(device),
                    delta_layer_svd["s"].to(device),
                    delta_layer_svd["v"].to(device),
                )
                delta = u @ torch.diag_embed(s) @ v
                delta_flat = delta.view(-1)
                skip_this = False
                for accepted_flat in accepted_deltas:
                    sim = measure_cosine_similarity(delta_flat, accepted_flat)
                    if sim > similarity_threshold:
                        pylogger.info(
                            f"Skipping task {dataset} for layer {layer_name} due to similarity {sim}"
                        )
                        skip_this = True
                        break
                if not skip_this:
                    accepted_tasks.append((u, s, v))
                    accepted_deltas.append(delta_flat)
            else:
                delta_layer = svd_dict[dataset][new_key]["dim1"].to(device)
                if i == 0:
                    aggregated_model_dict[layer_name] = delta_layer
                else:
                    aggregated_model_dict[layer_name] += (
                        delta_layer - aggregated_model_dict[layer_name]
                    ) / (i + 1)
        if "text_projection" in layer_name or not is_layer_matrix:
            continue
        if len(accepted_tasks) == 0:
            continue
        total_rank = sum((task_s.shape[0] for _, task_s, _ in accepted_tasks))
        sum_u = torch.zeros(accepted_tasks[0][0].shape[0], total_rank, device=device)
        sum_s = torch.zeros(total_rank, device=device)
        sum_v = torch.zeros(total_rank, accepted_tasks[0][2].shape[1], device=device)
        offset = 0
        for u_i, s_i, v_i in accepted_tasks:
            rank_i = s_i.shape[0]
            sum_u[:, offset : offset + rank_i] = u_i
            sum_s[offset : offset + rank_i] = s_i
            sum_v[offset : offset + rank_i, :] = v_i
            offset += rank_i
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        merged = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))
        aggregated_model_dict[layer_name] = merged.to(device)
    return aggregated_model_dict


@torch.no_grad()
def isotropic_sum(cumulative_dict, datasets, device="cuda"):
    aggregated_model_dict = {}
    for key in cumulative_dict.keys():
        cumulative_dict[key] = cumulative_dict[key].to(device)
        if len(cumulative_dict[key].shape) == 2 and "text_projection" not in key:
            u, s, v = torch.linalg.svd(cumulative_dict[key], full_matrices=False)
            iso_factor = torch.ones_like(s) * s.mean()
            aggregated_model_dict[key] = torch.linalg.multi_dot(
                (u, torch.diag(iso_factor), v)
            )
        else:
            aggregated_model_dict[key] = cumulative_dict[key] / len(datasets)
        aggregated_model_dict[key] = aggregated_model_dict[key].to(device)
    return aggregated_model_dict
