from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from .beliefs import PriorParameters, PublicPredictor, calibrate_prior
from .experiments import experiment_plan
from .merging import BeliefMerge, InformationConfig
from .strategies import StrategyConfig, verify_contribution_balance


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_load(path: str) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("Checkpoint must contain a state dictionary")
    return value


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    routes = {
        "vision-train": ("vision", "vision_main"),
        "language-train": ("language", "finetune_main"),
        "language-merge": ("language", "sequence_main"),
        "language-private": ("language", "private_main"),
        "language-metrics": ("language", "metrics_main"),
        "workflow": ("workflow", "workflow_main"),
        "aggregate": ("workflow", "aggregate_main"),
    }
    if argv and argv[0] in routes:
        module, function = routes[argv[0]]
        code = getattr(importlib.import_module("beliefmerge." + module), function)(
            argv[1:]
        )
        if code:
            raise SystemExit(code)
        return
    if argv and argv[0] == "calibrate":
        calibrate_command(argv[1:])
        return
    parser = argparse.ArgumentParser(prog="beliefmerge")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in [*routes, "calibrate"]:
        commands.add_parser(name, add_help=False)
    plan = commands.add_parser("plan")
    plan.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    plan.add_argument("--ablation-tasks", type=int, choices=[8, 14, 20], default=20)
    plan.add_argument("--output", required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--base", required=True)
    merge.add_argument("--tasks", nargs="+", required=True)
    merge.add_argument("--private", required=True)
    merge.add_argument("--belief", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--device", default="cpu")
    merge.add_argument("--seed", type=int, default=0)
    merge.add_argument("--steps", type=int, default=2000)
    merge.add_argument("--samples", type=int, default=128)
    merge.add_argument("--rank", type=int, default=4)
    merge.add_argument(
        "--preservation",
        choices=["private_product", "equation14"],
        default="private_product",
    )
    merge.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError("Experiment-plan output already exists")
        atomic_json(
            destination,
            [
                asdict(item)
                for item in experiment_plan(
                    args.seeds, ablation_tasks=args.ablation_tasks
                )
            ],
        )
        return
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    measured = safe_load(args.private)
    calibration = safe_load(args.belief)
    if calibration.get("split") != "prior_calibration":
        raise ValueError("Belief artifact must declare split=prior_calibration")
    if measured.get("split") not in {"train", "type_cal"}:
        raise ValueError("Private artifact must declare split=train or type_cal")
    if "model_ids" not in measured or "model_ids" not in calibration:
        raise ValueError("Artifacts must declare ordered model_ids")
    if len(measured["model_ids"]) != len(args.tasks) or len(
        set(measured["model_ids"])
    ) != len(args.tasks):
        raise ValueError("Private model_ids must align with task checkpoint order")
    if set(measured["model_ids"]) & set(calibration["model_ids"]):
        raise ValueError("Prior calibration and evaluated task identities overlap")
    predictor = PublicPredictor(**calibration["predictor_config"])
    predictor.load_state_dict(calibration["predictor"])
    predictor.to(args.device).eval().requires_grad_(False)
    prior = PriorParameters(**calibration["prior"])
    strategy = StrategyConfig(
        steps=args.steps,
        samples=args.samples,
        seed=args.seed,
        preservation=args.preservation,
    )

    def save(step: int, state: dict) -> None:
        atomic_checkpoint(output / f"strategy_step_{step:06d}.pt", state)
        atomic_json(output / "trace.json", state["trace"])
        print(f"Strategy updates: {step}/{args.steps}", flush=True)

    merger = BeliefMerge(
        InformationConfig(rank=args.rank), strategy, device=args.device
    )
    result = merger.merge(
        safe_load(args.base),
        [safe_load(path) for path in args.tasks],
        measured["scores"],
        predictor,
        prior,
        layer_map=measured["layer_map"],
        checkpoint=save,
    )
    metadata = result.metadata()
    metadata["model_ids"] = measured["model_ids"]
    atomic_json(output / "metrics.json", metadata)
    atomic_checkpoint(output / "merged.pt", result.state_dict)
    atomic_checkpoint(output / "strategies.pt", result.strategies.state_dict())
    if args.verify:
        report = verify_contribution_balance(
            result.strategies,
            result.belief,
            result.utility,
            result.private_scores,
            seed=args.seed + 1,
        )
        atomic_json(output / "verification.json", report)
    print("BeliefMerge completed", flush=True)


def calibrate_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="beliefmerge calibrate")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Calibration output already exists")
    artifact = safe_load(args.input)
    if artifact.get("split") != "prior_calibration":
        raise ValueError("Calibration input must declare its split")
    train_ids, prior_ids = (
        artifact["predictor_model_ids"],
        artifact["prior_model_ids"],
    )
    if not train_ids or not prior_ids or set(train_ids) & set(prior_ids):
        raise ValueError(
            "Predictor and residual calibration must use disjoint identities"
        )
    x, y = (artifact["predictor_public"], artifact["predictor_private"])
    prior_x, prior_y = (artifact["prior_public"], artifact["prior_private"])
    torch.manual_seed(args.seed)
    config = {"input_dim": x.shape[-1], "type_dim": y.shape[-1], "hidden": args.hidden}
    predictor = PublicPredictor(**config)
    history = predictor.fit(x, y, split="prior_calibration", steps=args.steps)
    with torch.no_grad():
        predictions = predictor(prior_x.float())
    prior = calibrate_prior(predictions, prior_y, split="prior_calibration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(
        destination,
        {
            "split": "prior_calibration",
            "model_ids": list(train_ids) + list(prior_ids),
            "predictor_config": config,
            "predictor": predictor.state_dict(),
            "prior": prior.state_dict(),
            "seed": args.seed,
            "training_loss": history,
        },
    )
    print("Belief calibration completed", flush=True)


if __name__ == "__main__":
    main()
