from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .merging import InformationConfig
from .strategies import StrategyConfig

CLIP_TASKS = (
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
    "CIFAR100",
    "STL10",
    "Flowers102",
    "OxfordIIITPet",
    "PCAM",
    "FER2013",
    "EMNISTDigits",
    "CIFAR10",
    "Food101",
    "FashionMNIST",
    "RenderedSST2",
    "KMNIST",
)
GLUE_TASKS = ("cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb")
BACKBONES = ("ViT-B-32", "ViT-B-16", "ViT-L-14")


@dataclass(frozen=True)
class Experiment:
    name: str
    backbone: str
    tasks: tuple[str, ...]
    seed: int
    training_step: int | None = None
    information: InformationConfig = InformationConfig()
    strategy: StrategyConfig = StrategyConfig()


def ablations() -> dict[str, tuple[InformationConfig, StrategyConfig]]:
    information, strategy = (InformationConfig(), StrategyConfig())
    output = {"full": (information, strategy)}
    for name, updates in {
        "without_magnitude": {"magnitude": False},
        "without_direction": {"direction": False},
        "with_compatibility": {"compatibility": True},
        "without_sensitivity": {"components": (1, 2)},
        "without_representation": {"components": (0, 2)},
        "without_performance": {"components": (0, 1)},
        "with_stability": {"components": (0, 1, 2, 3)},
        "LP": {"predictor_architecture": "linear"},
        "without_bau": {"belief_variant": "without_bau"},
        "without_bef": {"belief_variant": "without_bef"},
        "without_eta": {"belief_variant": "without_eta"},
        "laplace": {"family": "laplace"},
        "student_t": {"family": "student_t"},
    }.items():
        output[name] = (replace(information, **updates), strategy)
    output["without_nao"] = (information, replace(strategy, mode="without_nao"))
    return output


def experiment_plan(
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    *,
    ablation_tasks: int = 20,
    sensitivity_steps: Sequence[int] = tuple((2**i for i in range(1, 12))),
) -> list[Experiment]:
    if ablation_tasks not in {8, 14, 20} or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide unique seeds and a supported task suite")
    plan = []
    for seed in seeds:
        strategy = StrategyConfig(seed=seed)
        for backbone in BACKBONES:
            for count in (8, 14, 20):
                plan.append(
                    Experiment(
                        "main", backbone, CLIP_TASKS[:count], seed, strategy=strategy
                    )
                )
        plan.append(
            Experiment("main", "Flan-T5-Base", GLUE_TASKS, seed, strategy=strategy)
        )
        for backbone in BACKBONES[:2]:
            for name, (information, ablation_strategy) in ablations().items():
                plan.append(
                    Experiment(
                        name,
                        backbone,
                        CLIP_TASKS[:ablation_tasks],
                        seed,
                        information=information,
                        strategy=replace(ablation_strategy, seed=seed),
                    )
                )
            for rank in (1, 2, 4, 6, 8):
                plan.append(
                    Experiment(
                        f"rank_{rank}",
                        backbone,
                        CLIP_TASKS,
                        seed,
                        information=InformationConfig(rank=rank),
                        strategy=strategy,
                    )
                )
            reference = {"lambda_t": 0.5, "lambda_o": 0.3, "lambda_b": 0.2}
            for coefficient, values in {
                "lambda_t": (0.1, 0.3, 0.5, 0.7, 0.9),
                "lambda_o": (0.1, 0.2, 0.3, 0.4, 0.5),
                "lambda_b": (0.1, 0.2, 0.3, 0.4, 0.5),
            }.items():
                for weight in values:
                    weights = {
                        key: value * (1 - weight) / (1 - reference[coefficient])
                        for key, value in reference.items()
                    }
                    weights[coefficient] = weight
                    plan.append(
                        Experiment(
                            f"{coefficient}_{weight}",
                            backbone,
                            CLIP_TASKS,
                            seed,
                            strategy=replace(strategy, **weights),
                        )
                    )
        for step in sensitivity_steps:
            if step < 1:
                raise ValueError("Training steps must be positive")
            plan.append(
                Experiment(
                    "training_sensitivity",
                    BACKBONES[0],
                    CLIP_TASKS,
                    seed,
                    training_step=step,
                    strategy=strategy,
                )
            )
    return plan


def summarize_scores(
    scores: Mapping[str, float], individual: Mapping[str, float], tasks: Sequence[str]
) -> dict:
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Task order must be nonempty and unique")
    if set(scores) != set(tasks) or set(individual) != set(tasks):
        raise ValueError(
            "Every task must have a measured score and individual reference"
        )
    rows = {}
    for task in tasks:
        score, reference = (float(scores[task]), float(individual[task]))
        lower = -1.0 if task == "stsb" else 0.0
        if not math.isfinite(score) or not lower <= score <= 1:
            raise ValueError(
                "Scores must be fractions, with STS-B correlation in [-1,1]"
            )
        if not math.isfinite(reference) or not 0 < reference <= 1:
            raise ValueError("Individual references must be positive fractions")
        rows[task] = {
            "score": 100 * score,
            "normalized_score": 100 * score / reference,
            "individual": 100 * reference,
        }
    return {
        "tasks": rows,
        "average": float(np.mean([row["score"] for row in rows.values()])),
        "normalized_average": float(
            np.mean([row["normalized_score"] for row in rows.values()])
        ),
    }


def summarize_seeds(reports: Mapping[int, dict]) -> dict:
    if not reports:
        raise ValueError("No measured seed reports were supplied")
    ordered = sorted(reports, key=lambda seed: (reports[seed]["average"], seed))
    values = np.asarray([reports[seed]["average"] for seed in ordered], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Seed scores must be finite")
    tasks = set(reports[ordered[0]]["tasks"])
    if any((set(report["tasks"]) != tasks for report in reports.values())):
        raise ValueError("Seed reports must cover the same tasks")
    median_seed = ordered[(len(ordered) - 1) // 2]
    return {
        "seeds": ordered,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median_seed": median_seed,
        "median_seed_report": reports[median_seed],
        "per_task": {
            task: {
                "mean": float(
                    np.mean([r["tasks"][task]["score"] for r in reports.values()])
                ),
                "std": float(
                    np.std(
                        [r["tasks"][task]["score"] for r in reports.values()], ddof=1
                    )
                )
                if len(reports) > 1
                else 0.0,
            }
            for task in sorted(tasks)
        },
    }


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references) or not predictions:
        raise ValueError(
            "Prediction and reference sequences must be nonempty and aligned"
        )
    return float(
        np.mean([p.strip() == r.strip() for p, r in zip(predictions, references)])
    )


def spearman(predictions: Sequence[float], references: Sequence[float]) -> float:
    a, b = (np.asarray(predictions, dtype=float), np.asarray(references, dtype=float))
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError(
            "Correlation requires at least two aligned scalar observations"
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correlation inputs must be finite")

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and values[order[stop]] == values[order[start]]:
                stop += 1
            result[order[start:stop]] = (start + stop - 1) / 2
            start = stop
        return result

    ra, rb = (ranks(a), ranks(b))
    ra, rb = (ra - ra.mean(), rb - rb.mean())
    denominator = np.linalg.norm(ra) * np.linalg.norm(rb)
    if denominator == 0:
        raise ValueError(
            "Spearman correlation is undefined for constant predictions or references"
        )
    return float(np.dot(ra, rb) / denominator)


def validate_curve(
    records: Sequence[Mapping], metric: str = "average"
) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("Trajectory is empty")
    steps = np.asarray([row["step"] for row in records], dtype=np.int64)
    values = np.asarray([row[metric] for row in records], dtype=float)
    if (
        (steps <= 0).any()
        or (np.diff(steps) <= 0).any()
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Trajectory must have unique increasing steps and finite metrics"
        )
    return steps, values


def trajectory_summary(
    records: Sequence[Mapping], metric: str = "average", tolerance: float = 0.1
) -> dict:
    steps, values = validate_curve(records, metric)
    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative")
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    near = np.flatnonzero(values >= peak - tolerance)
    earliest = int(steps[int(near[0])])
    suffix_minimum = np.minimum.accumulate(values[::-1])[::-1]
    sustained = np.flatnonzero(suffix_minimum >= peak - tolerance)
    area = (
        float(np.sum((values[:-1] + values[1:]) * 0.5 * np.diff(steps)))
        if len(steps) > 1
        else 0.0
    )
    return {
        "peak": peak,
        "peak_step": int(steps[peak_index]),
        "final": float(values[-1]),
        "earliest_near_peak_step": earliest,
        "sustained_near_peak_step": int(steps[sustained[0]])
        if len(sustained)
        else None,
        "peak_to_final_drop": peak - float(values[-1]),
        "range": float(np.ptp(values)),
        "area_under_curve": area,
        "observed_steps": steps.tolist(),
        "metric": metric,
    }


def compare_trajectories(
    candidate: Sequence[Mapping],
    baseline: Sequence[Mapping],
    metric: str = "average",
    target: float | None = None,
) -> dict:
    candidate_steps, candidate_values = validate_curve(candidate, metric)
    baseline_steps, baseline_values = validate_curve(baseline, metric)
    if not np.array_equal(candidate_steps, baseline_steps):
        raise ValueError("Comparisons require identical fine-tuning steps")
    target = float(baseline_values.max()) if target is None else float(target)
    if not math.isfinite(target):
        raise ValueError("Target must be finite")
    candidate_hits = np.flatnonzero(candidate_values >= target)
    baseline_hits = np.flatnonzero(baseline_values >= target)
    candidate_first = (
        int(candidate_steps[candidate_hits[0]]) if len(candidate_hits) else None
    )
    baseline_first = (
        int(baseline_steps[baseline_hits[0]]) if len(baseline_hits) else None
    )
    deltas = candidate_values - baseline_values
    return {
        "target": target,
        "candidate_first_step": candidate_first,
        "baseline_first_step": baseline_first,
        "step_speedup": baseline_first / candidate_first
        if candidate_first and baseline_first
        else None,
        "average_gain": float(deltas.mean()),
        "final_gain": float(deltas[-1]),
        "per_step": [
            {
                "step": int(step),
                "candidate": float(a),
                "baseline": float(b),
                "gain": float(a - b),
            }
            for step, a, b in zip(candidate_steps, candidate_values, baseline_values)
        ],
        "candidate": trajectory_summary(candidate, metric),
        "baseline": trajectory_summary(baseline, metric),
    }


def paired_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int = 0,
    repetitions: int = 10000,
    confidence: float = 0.95,
) -> dict:
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError(
            "Paired bootstrap requires at least two aligned independent units"
        )
    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or repetitions < 100
        or not 0 < confidence < 1
    ):
        raise ValueError("Invalid bootstrap inputs")
    generator = np.random.default_rng(seed)
    differences = a - b
    estimates = np.empty(repetitions)
    for start in range(0, repetitions, 1024):
        count = min(1024, repetitions - start)
        indices = generator.integers(0, len(a), size=(count, len(a)))
        estimates[start : start + count] = differences[indices].mean(1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(estimates, [alpha, 1 - alpha])
    return {
        "mean_difference": float(differences.mean()),
        "confidence": confidence,
        "lower": float(lower),
        "upper": float(upper),
        "seed": seed,
        "repetitions": repetitions,
        "paired_units": int(len(a)),
    }


def private_component_associations(
    private_scores: np.ndarray,
    changes: np.ndarray,
    layers: Sequence[str],
    components: Sequence[str],
) -> dict:
    scores, response = (
        np.asarray(private_scores, dtype=float),
        np.asarray(changes, dtype=float),
    )
    if (
        scores.ndim != 3
        or scores.shape[1:] != (len(components), len(layers))
        or response.shape != (scores.shape[0],)
    ):
        raise ValueError(
            "Private scores must be [tasks,components,layers] and changes [tasks]"
        )
    if (
        scores.shape[0] < 3
        or not np.isfinite(scores).all()
        or not np.isfinite(response).all()
    ):
        raise ValueError("Provide at least three finite task observations")
    report = {}
    for component_index, component in enumerate(components):
        values = scores[:, component_index]
        per_layer = {}
        for layer_index, layer in enumerate(layers):
            try:
                correlation = spearman(values[:, layer_index], response)
            except ValueError:
                correlation = None
            per_layer[layer] = correlation
        report[component] = {
            "mean_score": float(values.mean()),
            "per_layer_spearman": per_layer,
        }
    return {
        "task_count": scores.shape[0],
        "components": report,
        "interpretation": "descriptive_association_not_causal_evidence",
    }


def normalized_gap(
    merged: Mapping[str, float], individual: Mapping[str, float]
) -> dict:
    if set(merged) != set(individual) or not merged:
        raise ValueError("Task sets must match")
    tasks = {}
    for task, value in merged.items():
        reference = individual[task]
        if not math.isfinite(value) or not math.isfinite(reference) or reference <= 0:
            raise ValueError("Gap inputs must be finite with positive reference scores")
        tasks[task] = {
            "absolute_gap": reference - value,
            "relative_gap": (reference - value) / reference,
        }
    return {
        "tasks": tasks,
        "absolute_gap": float(np.mean([v["absolute_gap"] for v in tasks.values()])),
        "relative_gap": float(np.mean([v["relative_gap"] for v in tasks.values()])),
    }
