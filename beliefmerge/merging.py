from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from .beliefs import JointBelief, PriorParameters
from .information import PublicInformation, State, public_information, validate_states
from .strategies import ReducedUtility, StrategyConfig, StrategyFit, fit_strategies


@dataclass(frozen=True)
class InformationConfig:
    rank: int = 4
    magnitude: bool = True
    direction: bool = True
    compatibility: bool = False
    components: tuple[int, ...] = (0, 1, 2)
    family: str = "gaussian"
    belief_variant: str = "full"

    def __post_init__(self) -> None:
        if (
            self.rank < 1
            or not self.components
            or len(set(self.components)) != len(self.components)
        ):
            raise ValueError("Invalid rank or private-component selection")
        if any((component not in range(4) for component in self.components)):
            raise ValueError(
                "Private components must be selected from indices 0 through 3"
            )


@dataclass
class MergeResult:
    state_dict: OrderedDict[str, Tensor]
    actions: Tensor
    weights: Tensor
    information: PublicInformation
    strategies: StrategyFit
    belief: JointBelief
    utility: ReducedUtility
    private_scores: Tensor
    information_config: InformationConfig

    def metadata(self) -> dict:
        return {
            "method": "BeliefMerge",
            "information_config": asdict(self.information_config),
            "strategy_config": asdict(self.strategies.config),
            "layers": list(self.information.layers),
            "effective_rank": self.information.rank,
            "weights": self.weights.cpu().tolist(),
            "actions": self.actions.cpu().tolist(),
            "trace": self.strategies.trace,
            "regularity": self.utility.regularity_bounds(),
            "benchmark_results_verified": False,
        }


@torch.no_grad()
def assemble(
    base: State, tasks: Sequence[State], weights: Tensor, information: PublicInformation
) -> OrderedDict[str, Tensor]:
    validate_states(base, tasks)
    if weights.shape != (len(tasks), information.layer_count):
        raise ValueError("Weights must have shape [models,layers]")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    if not torch.allclose(weights.sum(0), torch.ones_like(weights[0]), atol=1e-05):
        raise ValueError("Weights must sum to one across models for every layer")
    layer_indices = {layer: i for i, layer in enumerate(information.layers)}
    merged = OrderedDict()
    for key, initial in base.items():
        if key not in information.layer_map:
            merged[key] = initial.detach().clone()
            continue
        layer = layer_indices[information.layer_map[key]]
        dtype = torch.float64 if initial.dtype == torch.float64 else torch.float32
        origin = initial.detach().to(dtype=dtype)
        value = origin.clone()
        for model, task in enumerate(tasks):
            delta = task[key].detach().to(device=initial.device, dtype=dtype) - origin
            value.add_(delta * weights[model, layer].to(value))
        merged[key] = value.to(initial.dtype)
    return merged


class BeliefMerge:
    def __init__(
        self,
        information: InformationConfig = InformationConfig(),
        strategy: StrategyConfig = StrategyConfig(),
        *,
        device: str = "cpu",
    ):
        self.information_config = information
        self.strategy_config = strategy
        self.device = torch.device(device)

    def merge(
        self,
        base: State,
        tasks: Sequence[State],
        private_scores: Tensor,
        predictor: Callable[[Tensor], Tensor],
        prior: PriorParameters,
        *,
        layer_map: Mapping[str, str] | None = None,
        checkpoint: Callable[[int, dict], None] | None = None,
    ) -> MergeResult:
        info_cfg = self.information_config
        geometry = public_information(base, tasks, info_cfg.rank, layer_map)
        models, layers = (len(tasks), geometry.layer_count)
        if (
            private_scores.ndim != 3
            or private_scores.shape[0] != models
            or private_scores.shape[-1] != layers
        ):
            raise ValueError("Private information must be [models,components,layers]")
        components = private_scores.shape[1]
        if max(info_cfg.components) >= components:
            raise ValueError("A requested private component was not measured")
        if (
            not torch.isfinite(private_scores).all()
            or (private_scores <= 0).any()
            or (private_scores >= 1).any()
        ):
            raise ValueError(
                "Private information must lie strictly between zero and one"
            )
        public = geometry.public_inputs(
            info_cfg.magnitude, info_cfg.direction, info_cfg.compatibility
        )
        public = public.to(device=self.device, dtype=torch.float32)
        indices = torch.tensor(
            [
                component * layers + layer
                for component in info_cfg.components
                for layer in range(layers)
            ],
            dtype=torch.long,
        )
        with torch.no_grad():
            predictions = predictor(public)
        if predictions.shape != (models, components * layers):
            raise ValueError(
                "Predictor output must match all supplied private components"
            )
        if prior.mean.numel() != components * layers:
            raise ValueError("Prior dimension must match supplied private information")
        predictions = predictions.to(public)[:, indices.to(self.device)]
        selected_prior = prior.select(indices).to(predictions)
        belief = JointBelief(
            predictions,
            selected_prior,
            family=info_cfg.family,
            variant=info_cfg.belief_variant,
        )
        utility = ReducedUtility(
            geometry, self.strategy_config, len(info_cfg.components)
        )
        strategies = fit_strategies(
            belief, public, utility, self.strategy_config, checkpoint
        )
        selected_private = private_scores.flatten(1)[:, indices].to(public)
        actions = strategies.actions(selected_private)
        weights = actions.softmax(0)
        state = assemble(base, tasks, weights, geometry)
        return MergeResult(
            state,
            actions,
            weights,
            geometry,
            strategies,
            belief,
            utility,
            selected_private,
            info_cfg,
        )
