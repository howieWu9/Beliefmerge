from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from .beliefs import JointBelief
from .information import PublicInformation


@dataclass(frozen=True)
class StrategyConfig:
    steps: int = 2000
    samples: int = 128
    hidden: int = 128
    learning_rate: float = 2e-05
    action_bound: float = 5.0
    lambda_t: float = 0.6
    lambda_o: float = 0.4
    lambda_b: float = 0.04
    epsilon_local: float = 1e-08
    epsilon_public: float = 1e-08
    save_every: int = 50
    seed: int = 0
    mode: str = "nash"
    preservation: str = "private_product"

    def __post_init__(self) -> None:
        if min(self.steps, self.samples, self.hidden, self.save_every) < 1:
            raise ValueError("Iteration counts and dimensions must be positive")
        if (
            min(
                self.learning_rate,
                self.action_bound,
                self.epsilon_local,
                self.epsilon_public,
            )
            <= 0
        ):
            raise ValueError(
                "Learning rate, action bound, and stabilizers must be positive"
            )
        if min(self.lambda_t, self.lambda_o, self.lambda_b) < 0:
            raise ValueError("Utility coefficients cannot be negative")
        if abs(self.lambda_t + self.lambda_o - 1) > 1e-08:
            raise ValueError("lambda_t + lambda_o must equal one")
        if self.mode not in {"nash", "without_nao"}:
            raise ValueError("Unknown strategy optimization mode")
        if self.preservation not in {"private_product", "equation14"}:
            raise ValueError("Unknown preservation definition")


class ContributionStrategy(nn.Module):
    def __init__(
        self,
        type_dim: int,
        public_dim: int,
        layers: int,
        hidden: int = 128,
        action_bound: float = 5.0,
    ):
        super().__init__()
        self.action_bound = action_bound
        self.network = nn.Sequential(
            nn.Linear(type_dim + public_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, layers),
        )

    def forward(self, own_scores: Tensor, public_context: Tensor) -> Tensor:
        context = public_context.expand(*own_scores.shape[:-1], public_context.numel())
        return self.action_bound * torch.sigmoid(
            self.network(torch.cat([own_scores, context], -1))
        )


class ReducedUtility(nn.Module):
    def __init__(
        self,
        information: PublicInformation,
        config: StrategyConfig,
        components: int = 3,
    ):
        super().__init__()
        if components < 1:
            raise ValueError("At least one private component is required")
        self.config = config
        self.components = components
        self.register_buffer("gram", information.gram.detach().clone())
        self.register_buffer("projections", information.projections.detach().clone())

    def forward(self, actions: Tensor, scores: Tensor) -> Tensor:
        models, layers = (self.gram.shape[1], self.gram.shape[0])
        if actions.shape[-2:] != (models, layers):
            raise ValueError("Actions must end in [models,layers]")
        if scores.shape != actions.shape[:-1] + (self.components * layers,):
            raise ValueError("Private-score dimensions do not match actions")
        weights = actions.softmax(dim=-2)
        norm_by_layer = torch.einsum(
            "...ml,lmn,...nl->...l", weights, self.gram, weights
        )
        cross = torch.einsum("lmn,...nl->...ml", self.gram, weights)
        individual_norm = self.gram.diagonal(dim1=-2, dim2=-1).T
        distance = norm_by_layer.unsqueeze(-2) - 2 * cross + individual_norm
        if self.config.preservation == "private_product":
            preferences = scores.reshape(
                *scores.shape[:-1], self.components, layers
            ).prod(-2)
            local = (
                preferences * distance / (individual_norm + self.config.epsilon_local)
            ).sum(-1)
        else:
            local = distance.sum(-1) / (
                individual_norm.sum(-1) + self.config.epsilon_local
            )
        projected = torch.einsum("...ml,lmr->...r", weights, self.projections)
        public = projected.square().sum(-1) / (
            norm_by_layer.sum(-1) + self.config.epsilon_public
        )
        return (
            -self.config.lambda_t * local
            + self.config.lambda_o * public.unsqueeze(-1)
            - self.config.lambda_b * actions.square().sum(-1)
        )

    def regularity_bounds(self) -> dict[str, float | bool]:
        cfg = self.config
        layers, models, _ = self.gram.shape
        max_norm_sq = float(self.gram.sum(0).diag().max().clamp_min(0))
        gradient = (
            2 * cfg.lambda_t * (layers**0.5 + 1) * max_norm_sq / cfg.epsilon_local
            + 2 * cfg.lambda_o * max_norm_sq**0.5 / cfg.epsilon_public**0.5
        )
        curvature = (
            (6 + 4 * layers**0.5) * cfg.lambda_t * max_norm_sq / cfg.epsilon_local
            + 20 * cfg.lambda_o * max_norm_sq / cfg.epsilon_public
            + 4 * cfg.lambda_o * max_norm_sq**0.5 / cfg.epsilon_public**0.5
        )
        return {
            "gradient_bound": gradient,
            "curvature_bound": curvature,
            "strong_concavity": 2 * cfg.lambda_b - curvature,
            "uniqueness_sufficient": 2 * cfg.lambda_b > models * curvature,
        }


@dataclass
class StrategyFit:
    networks: nn.ModuleList
    context: Tensor
    trace: list[dict]
    config: StrategyConfig

    @torch.no_grad()
    def actions(self, private_scores: Tensor) -> Tensor:
        if private_scores.ndim != 2 or private_scores.shape[0] != len(self.networks):
            raise ValueError("One private vector per strategy is required")
        return torch.stack(
            [
                network(private_scores[m].to(self.context), self.context)
                for m, network in enumerate(self.networks)
            ]
        )

    def state_dict(self) -> dict:
        return {
            "networks": {
                key: value.detach().cpu().clone()
                for key, value in self.networks.state_dict().items()
            },
            "context": self.context.detach().cpu().clone(),
            "config": asdict(self.config),
            "trace": list(self.trace),
        }


def fit_strategies(
    belief: JointBelief,
    public_inputs: Tensor,
    utility: ReducedUtility,
    config: StrategyConfig,
    checkpoint: Callable[[int, dict], None] | None = None,
) -> StrategyFit:
    device = belief.predictions.device
    dtype = belief.predictions.dtype
    generator = torch.Generator(device=device).manual_seed(config.seed)
    models, dimension = belief.predictions.shape
    layers = utility.gram.shape[0]
    context = public_inputs.detach().to(device=device, dtype=dtype).flatten()
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        networks = nn.ModuleList(
            [
                ContributionStrategy(
                    dimension,
                    context.numel(),
                    layers,
                    config.hidden,
                    config.action_bound,
                )
                for _ in range(models)
            ]
        ).to(device=device, dtype=dtype)
    utility = utility.to(device=device, dtype=dtype)
    optimizers = [
        torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=0)
        for network in networks
    ]
    result = StrategyFit(networks, context, [], config)
    for step in range(1, config.steps + 1):
        selected = (step - 1) % models
        types = (
            belief.common_samples(config.samples, generator)
            if config.mode == "without_nao"
            else belief.fitting_samples(selected, config.samples, generator)
        )
        actions = []
        for model, network in enumerate(networks):
            if model == selected:
                actions.append(network(types[:, model], context))
            else:
                with torch.no_grad():
                    actions.append(network(types[:, model], context))
        action_tensor = torch.stack(actions, dim=-2)
        utilities = utility(action_tensor, types)
        objective = (
            utilities.mean()
            if config.mode == "without_nao"
            else utilities[:, selected].mean()
        )
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite strategy utility")
        optimizers[selected].zero_grad(set_to_none=True)
        (-objective).backward()
        optimizers[selected].step()
        record = {
            "step": step,
            "model": selected,
            "objective": float(objective.detach()),
            "action_mean": float(action_tensor[:, selected].detach().mean()),
        }
        result.trace.append(record)
        if checkpoint is not None and (
            step % config.save_every == 0 or step == config.steps
        ):
            payload = result.state_dict()
            payload["optimizers"] = [optimizer.state_dict() for optimizer in optimizers]
            payload["generator_state"] = generator.get_state().cpu()
            payload["step"] = step
            checkpoint(step, payload)
    networks.eval()
    networks.requires_grad_(False)
    return result


def verify_contribution_balance(
    fit: StrategyFit,
    belief: JointBelief,
    utility: ReducedUtility,
    private_scores: Tensor,
    *,
    samples: int = 1024,
    search_steps: int = 100,
    restarts: int = 4,
    seed: int = 1,
    failure_probability: float = 0.05,
) -> dict:
    if (
        samples < 2
        or search_steps < 1
        or restarts < 1
        or (not 0 < failure_probability < 1)
    ):
        raise ValueError("Invalid verification budget or failure probability")
    if seed == fit.config.seed:
        raise ValueError("Verification must use an independent random seed")
    generator = torch.Generator(device=fit.context.device).manual_seed(seed)
    observed = private_scores.to(fit.context)
    baseline = fit.actions(observed)
    cfg = fit.config
    utility = utility.to(fit.context)
    bounds = utility.regularity_bounds()
    mu = float(bounds["strong_concavity"])
    models, layers = baseline.shape
    results = []
    for model in range(models):
        types = belief.conditional_samples(model, observed[model], samples, generator)
        with torch.no_grad():
            opponents = torch.stack(
                [
                    network(types[:, j], fit.context)
                    for j, network in enumerate(fit.networks)
                ],
                -2,
            )

        def objective(candidate: Tensor) -> Tensor:
            profile = opponents.clone()
            profile[:, model] = candidate
            return utility(profile, types)[:, model].mean()

        own = baseline[model].detach().clone().requires_grad_()
        initial = objective(own)
        (gradient,) = torch.autograd.grad(initial, own)
        best_value, best_action = (float(initial.detach()), own.detach().clone())
        for restart in range(restarts):
            start = (
                own.detach().clone()
                if restart == 0
                else torch.rand(
                    own.shape, device=own.device, dtype=own.dtype, generator=generator
                )
                * cfg.action_bound
            )
            candidate = nn.Parameter(start)
            optimizer = torch.optim.Adam([candidate], lr=0.05)
            for _ in range(search_steps):
                optimizer.zero_grad(set_to_none=True)
                value = objective(candidate)
                if float(value.detach()) > best_value:
                    best_value, best_action = (
                        float(value.detach()),
                        candidate.detach().clone(),
                    )
                (-value).backward()
                optimizer.step()
                with torch.no_grad():
                    candidate.clamp_(0, cfg.action_bound)
            value = float(objective(candidate).detach())
            if value > best_value:
                best_value, best_action = (value, candidate.detach().clone())
        certificate = None
        if mu > 0 and belief.family == "gaussian":
            delta = (own.detach() + gradient / mu).clamp(
                0, cfg.action_bound
            ) - own.detach()
            quadratic = float(gradient @ delta - 0.5 * mu * delta.square().sum())
            error = (
                cfg.action_bound
                * layers
                * float(bounds["gradient_bound"])
                * (2 * math.log(2 * models * layers / failure_probability) / samples)
                ** 0.5
            )
            certificate = max(0.0, quadratic) + error
        results.append(
            {
                "model": model,
                "sampled_gain": max(0.0, best_value - float(initial.detach())),
                "candidate_action": best_action.cpu().tolist(),
                "projected_gradient_norm": float(
                    (
                        (own.detach() + gradient).clamp(0, cfg.action_bound)
                        - own.detach()
                    ).norm()
                ),
                "certified_upper_bound": certificate,
            }
        )
    return {
        "seed": seed,
        "samples": samples,
        "failure_probability": failure_probability,
        "regularity": bounds,
        "models": results,
        "sampled_gains_are_certificates": False,
        "uniform_equilibrium_certified": False,
    }
