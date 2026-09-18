from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


def stabilized_logit(scores: Tensor, epsilon: float = 1e-05) -> Tensor:
    if not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Private scores must be finite and in [0, 1]")
    return torch.logit(scores.clamp(epsilon, 1 - epsilon))


def positive_covariance(value: Tensor, floor: float = 1e-06) -> Tensor:
    value = (value + value.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    return eigenvectors * eigenvalues.clamp_min(floor) @ eigenvectors.T


class PublicPredictor(nn.Module):
    def __init__(self, input_dim: int, type_dim: int, hidden: int = 128):
        super().__init__()
        if min(input_dim, type_dim, hidden) < 1:
            raise ValueError("Predictor dimensions must be positive")
        self.input_dim = input_dim
        self.type_dim = type_dim
        self.register_buffer("center", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, type_dim),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.network((observations - self.center) / self.scale)

    def fit(
        self,
        public_train: Tensor,
        private_train: Tensor,
        *,
        split: str,
        steps: int = 1000,
        learning_rate: float = 0.001,
    ) -> list[float]:
        if split != "prior_calibration":
            raise ValueError(
                "Public predictor requires a separate prior_calibration split"
            )
        if public_train.shape[:-1] != private_train.shape[:-1]:
            raise ValueError("Calibration observations and types must align")
        if (
            public_train.shape[-1] != self.input_dim
            or private_train.shape[-1] != self.type_dim
        ):
            raise ValueError(
                "Calibration feature dimensions do not match the predictor"
            )
        if steps < 1 or learning_rate <= 0:
            raise ValueError("Predictor training budget must be positive")
        x = public_train.reshape(-1, self.input_dim).to(self.center)
        y = stabilized_logit(private_train.reshape(-1, self.type_dim)).to(self.center)
        if not torch.isfinite(x).all() or x.shape[0] < 2:
            raise ValueError("Provide at least two finite calibration observations")
        with torch.no_grad():
            self.center.copy_(x.mean(0))
            self.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-06))
        self.requires_grad_(True)
        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=0
        )
        history = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            error = (self(x) - y).square().mean()
            if not torch.isfinite(error):
                raise FloatingPointError("Nonfinite public-predictor objective")
            error.backward()
            optimizer.step()
            history.append(float(error.detach()))
        self.eval()
        self.requires_grad_(False)
        return history


@dataclass
class PriorParameters:
    mean: Tensor
    latent_covariance: Tensor
    noise_covariance: Tensor

    def __post_init__(self) -> None:
        dimension = self.mean.numel()
        if self.mean.ndim != 1 or not torch.isfinite(self.mean).all():
            raise ValueError("Prior mean must be a finite vector")
        for covariance in (self.latent_covariance, self.noise_covariance):
            if (
                covariance.shape != (dimension, dimension)
                or not torch.isfinite(covariance).all()
            ):
                raise ValueError("Prior covariances have invalid dimensions or values")
            if not torch.allclose(covariance, covariance.T):
                raise ValueError("Covariances must be symmetric")
            torch.linalg.cholesky(covariance)

    @classmethod
    def from_component_covariances(
        cls, mean: Tensor, latent: Tensor, noise: Tensor, layers: int
    ) -> "PriorParameters":
        if layers < 1 or mean.ndim != 1:
            raise ValueError("Invalid component mean or layer count")
        identity = torch.eye(layers, dtype=mean.dtype, device=mean.device)
        return cls(
            mean.repeat_interleave(layers),
            torch.kron(latent.contiguous(), identity),
            torch.kron(noise.contiguous(), identity),
        )

    def to(self, reference: Tensor) -> "PriorParameters":
        return PriorParameters(
            self.mean.to(reference),
            self.latent_covariance.to(reference),
            self.noise_covariance.to(reference),
        )

    def select(self, indices: Tensor) -> "PriorParameters":
        indices = indices.to(self.mean.device)
        return PriorParameters(
            self.mean[indices],
            self.latent_covariance[indices][:, indices],
            self.noise_covariance[indices][:, indices],
        )

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "latent_covariance": self.latent_covariance.detach().cpu(),
            "noise_covariance": self.noise_covariance.detach().cpu(),
        }


@torch.no_grad()
def calibrate_prior(
    public_predictions: Tensor,
    private_scores: Tensor,
    *,
    split: str,
    covariance_floor: float = 1e-05,
) -> PriorParameters:
    if split != "prior_calibration":
        raise ValueError(
            "Belief calibration must not consume merging-task private types"
        )
    if public_predictions.shape != private_scores.shape or private_scores.ndim != 3:
        raise ValueError("Calibration requires [independent_groups, models, type_dim]")
    groups, models, dimension = private_scores.shape
    if groups < 2 or models < 2 or covariance_floor <= 0:
        raise ValueError("At least two independent groups and two models are required")
    residual = stabilized_logit(private_scores).double() - public_predictions.double()
    group_mean = residual.mean(1)
    mean = group_mean.mean(0)
    within = residual - group_mean[:, None]
    noise = torch.einsum("gmi,gmj->ij", within, within) / (groups * (models - 1))
    centered = group_mean - mean
    latent = centered.T @ centered / (groups - 1) - noise / models
    return PriorParameters(
        mean,
        positive_covariance(latent, covariance_floor),
        positive_covariance(noise, covariance_floor),
    )


class JointBelief:
    def __init__(
        self,
        predictions: Tensor,
        prior: PriorParameters,
        *,
        family: str = "gaussian",
        variant: str = "full",
        importance_particles: int = 4096,
    ):
        if predictions.ndim != 2 or predictions.shape[1] != prior.mean.numel():
            raise ValueError("Public predictions must have shape [models, type_dim]")
        if not torch.isfinite(predictions).all():
            raise ValueError("Public predictions must be finite")
        if family not in {"gaussian", "laplace", "student_t"}:
            raise ValueError("Unknown latent prior family")
        if variant not in {
            "full",
            "without_f",
            "without_bau",
            "without_bef",
            "without_eta",
            "without_bpi",
        }:
            raise ValueError("Unknown belief ablation")
        if importance_particles < 128:
            raise ValueError("Use at least 128 importance particles")
        self.predictions = predictions.detach().clone()
        if variant == "without_f":
            self.predictions.zero_()
        self.prior = prior.to(self.predictions)
        self.family = family
        self.variant = variant
        self.importance_particles = importance_particles
        self.latent_factor = torch.linalg.cholesky(self.prior.latent_covariance)
        self.noise_factor = torch.linalg.cholesky(self.prior.noise_covariance)
        total = self.prior.latent_covariance + self.prior.noise_covariance
        self.gain = torch.linalg.solve(total, self.prior.latent_covariance).T
        posterior = (
            self.prior.latent_covariance - self.gain @ self.prior.latent_covariance
        )
        self.posterior_covariance = (posterior + posterior.T) / 2
        self.posterior_factor = torch.linalg.cholesky(self.posterior_covariance)
        self.last_importance_ess: float | None = None

    def _normal(self, shape: Sequence[int], generator: torch.Generator) -> Tensor:
        return torch.randn(
            tuple(shape),
            dtype=self.predictions.dtype,
            device=self.predictions.device,
            generator=generator,
        )

    def _latent(self, shape: Sequence[int], generator: torch.Generator) -> Tensor:
        shape = tuple(shape) + (self.prior.mean.numel(),)
        if self.variant == "without_eta":
            return self.prior.mean.expand(shape)
        if self.family == "gaussian":
            standardized = self._normal(shape, generator)
        elif self.family == "laplace":
            u = (
                torch.rand(
                    shape,
                    dtype=self.predictions.dtype,
                    device=self.predictions.device,
                    generator=generator,
                ).clamp(1e-07, 1 - 1e-07)
                - 0.5
            )
            standardized = -u.sign() * torch.log1p(-2 * u.abs()) / 2**0.5
        else:
            normal = self._normal(shape, generator)
            chi_sq = (
                self._normal(shape[:-1] + (3,), generator)
                .square()
                .sum(-1, keepdim=True)
            )
            standardized = normal / chi_sq.clamp_min(1e-12).sqrt()
        return self.prior.mean + standardized @ self.latent_factor.T

    @torch.no_grad()
    def common_samples(self, count: int, generator: torch.Generator) -> Tensor:
        if count < 1:
            raise ValueError("Sample count must be positive")
        models, dimension = self.predictions.shape
        latent = self._latent((count, 1), generator)
        noise = (
            self._normal((count, models, dimension), generator) @ self.noise_factor.T
        )
        return torch.sigmoid(self.predictions + latent + noise)

    @torch.no_grad()
    def posterior(self, model: int, own_scores: Tensor) -> tuple[Tensor, Tensor]:
        if self.family != "gaussian":
            raise ValueError(
                "Closed-form posterior is only available for Gaussian priors"
            )
        residual = (
            stabilized_logit(own_scores) - self.predictions[model] - self.prior.mean
        )
        mean = self.prior.mean + residual @ self.gain.T
        return (mean, self.posterior_covariance)

    @torch.no_grad()
    def conditional_samples(
        self, model: int, own_scores: Tensor, count: int, generator: torch.Generator
    ) -> Tensor:
        models, dimension = self.predictions.shape
        if not 0 <= model < models or count < 1:
            raise ValueError("Invalid observer or sample count")
        own = own_scores.to(self.predictions)
        if own.shape == (dimension,):
            own = own.expand(count, dimension)
        if own.shape != (count, dimension):
            raise ValueError(
                "Own private scores must be [type_dim] or [samples,type_dim]"
            )
        stabilized_logit(own)
        if self.variant == "without_bpi":
            scores = (
                torch.sigmoid(self.predictions + self.prior.mean)
                .expand(count, models, dimension)
                .clone()
            )
        else:
            independent = self.variant == "without_bef"
            latent_count = models if independent else 1
            if self.variant in {"without_bau", "without_eta"}:
                latent = self._latent((count, latent_count), generator)
            elif self.family == "gaussian":
                mean, _ = self.posterior(model, own)
                latent = (
                    mean[:, None]
                    + self._normal((count, latent_count, dimension), generator)
                    @ self.posterior_factor.T
                )
            else:
                particles = self._latent((self.importance_particles,), generator)
                residual = stabilized_logit(own) - self.predictions[model]
                difference = residual[:, None, :] - particles[None, :, :]
                whitened = torch.linalg.solve_triangular(
                    self.noise_factor,
                    difference.movedim(-1, 0).reshape(dimension, -1),
                    upper=False,
                )
                log_weights = -0.5 * whitened.square().sum(0).reshape(count, -1)
                weights = torch.softmax(log_weights, dim=-1)
                self.last_importance_ess = float((1 / weights.square().sum(-1)).min())
                indices = torch.multinomial(
                    weights, latent_count, replacement=True, generator=generator
                )
                latent = particles[indices]
            noise = (
                self._normal((count, models, dimension), generator)
                @ self.noise_factor.T
            )
            scores = torch.sigmoid(self.predictions + latent + noise)
        scores[:, model] = own
        return scores

    @torch.no_grad()
    def fitting_samples(
        self, model: int, count: int, generator: torch.Generator
    ) -> Tensor:
        samples = self.common_samples(count, generator)
        if self.variant in {"full", "without_f", "without_eta"}:
            return samples
        return self.conditional_samples(model, samples[:, model], count, generator)
