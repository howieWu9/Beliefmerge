from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

State = Mapping[str, Tensor]


def checked_tensor(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a real floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    return value


def validate_states(base: State, tasks: Sequence[State]) -> None:
    if not base or len(tasks) < 2:
        raise ValueError("Provide a nonempty base and at least two task models")
    for task in tasks:
        if set(task) != set(base):
            raise ValueError(
                "Base and task state dictionaries must have identical keys"
            )
        for key, value in base.items():
            if not isinstance(value, Tensor) or not isinstance(task[key], Tensor):
                raise ValueError("State dictionaries must contain tensors only")
            if value.shape != task[key].shape or value.dtype != task[key].dtype:
                raise ValueError(f"Incompatible tensor: {key}")
            if value.is_complex():
                raise ValueError("Complex parameters are not supported")
            if value.is_floating_point():
                checked_tensor(value, key)
                checked_tensor(task[key], key)


@dataclass
class PublicInformation:
    observations: Tensor
    gram: Tensor
    projections: Tensor
    basis_coefficients: Tensor
    layers: tuple[str, ...]
    layer_map: dict[str, str]
    compatibility: Tensor
    requested_rank: int

    @property
    def model_count(self) -> int:
        return self.gram.shape[1]

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def rank(self) -> int:
        return self.projections.shape[-1]

    def public_inputs(
        self,
        magnitude: bool = True,
        direction: bool = True,
        compatibility: bool = False,
    ) -> Tensor:
        pieces = []
        if magnitude:
            pieces.append(self.observations[:, :1])
        if direction:
            pieces.append(self.observations[:, 1:])
        if compatibility:
            pieces.append(self.compatibility)
        if not pieces:
            raise ValueError("At least one public feature must be retained")
        return torch.cat(pieces, dim=-1)


@torch.no_grad()
def public_information(
    base: State,
    tasks: Sequence[State],
    rank: int = 4,
    layer_map: Mapping[str, str] | None = None,
    epsilon: float = 1e-08,
) -> PublicInformation:
    validate_states(base, tasks)
    if rank < 1 or epsilon <= 0:
        raise ValueError("rank and epsilon must be positive")
    mapping = (
        dict(layer_map)
        if layer_map is not None
        else {key: key for key, value in base.items() if value.is_floating_point()}
    )
    if not mapping or any((key not in base for key in mapping)):
        raise ValueError("layer_map must select existing floating parameters")
    for key in mapping:
        checked_tensor(base[key], key)
    layers = tuple(dict.fromkeys(mapping.values()))
    if any((not isinstance(layer, str) or not layer for layer in layers)):
        raise ValueError("Layer names must be nonempty strings")
    indices = {layer: i for i, layer in enumerate(layers)}
    count = len(tasks)
    gram = torch.zeros(len(layers), count, count, dtype=torch.float64)
    base_norm_sq = torch.zeros((), dtype=torch.float64)
    for key, layer in mapping.items():
        initial = base[key].detach().to(device="cpu", dtype=torch.float64).flatten()
        updates = torch.stack(
            [
                task[key].detach().to(device="cpu", dtype=torch.float64).flatten()
                - initial
                for task in tasks
            ]
        )
        gram[indices[layer]] += updates @ updates.T
        base_norm_sq += initial.square().sum()
    total = gram.sum(0)
    norms = total.diag().clamp_min(0).sqrt()
    inv_norm = (norms + epsilon).reciprocal()
    normalized = total * inv_norm[:, None] * inv_norm[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(normalized)
    order = torch.argsort(eigenvalues, descending=True)
    threshold = max(float(eigenvalues.max()) * 1e-10, 1e-14)
    retained = order[eigenvalues[order] > threshold][:rank]
    values, vectors = (eigenvalues[retained], eigenvectors[:, retained])
    if retained.numel():
        anchors = vectors.abs().argmax(dim=0)
        signs = vectors[anchors, torch.arange(vectors.shape[1])].sign()
        vectors = vectors * signs
    coefficients = inv_norm[:, None] * vectors / values.sqrt()[None, :]
    projections = torch.einsum("lij,jr->lir", gram, coefficients)
    directions = inv_norm[:, None] * projections.sum(0)
    magnitudes = norms / (base_norm_sq.sqrt() + epsilon)
    layer_norms = gram.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    cosine = gram / (layer_norms[:, :, None] * layer_norms[:, None, :] + epsilon)
    off_diagonal = ~torch.eye(count, dtype=torch.bool)
    compatibility = (cosine * off_diagonal).sum(-1).T / (count - 1)
    return PublicInformation(
        torch.cat([magnitudes[:, None], directions], -1),
        gram,
        projections,
        coefficients,
        layers,
        mapping,
        compatibility,
        rank,
    )


@dataclass(frozen=True)
class DiagnosticConfig:
    sensitivity_scale: float = 1e-10
    performance_temperature: float = 1.0
    epsilon: float = 1e-08
    score_epsilon: float = 1e-05

    def __post_init__(self) -> None:
        if min(self.sensitivity_scale, self.performance_temperature, self.epsilon) <= 0:
            raise ValueError("Diagnostic constants must be positive")
        if not 0 < self.score_epsilon < 0.5:
            raise ValueError("score_epsilon must be in (0, .5)")


@dataclass
class PrivateInformation:
    scores: Tensor
    layers: tuple[str, ...]
    sample_count: int
    sensitivity_energy: Tensor
    representation_energy: Tensor
    task_loss: float
    ablated_losses: Tensor
    stability: Tensor

    def with_stability(self) -> Tensor:
        return torch.cat([self.scores, self.stability[None]], dim=0)


def scores_from_statistics(
    sensitivity_energy: Tensor,
    response_energy: Tensor,
    representation_norm: Tensor,
    update_norm: Tensor,
    task_loss: Tensor,
    ablated_losses: Tensor,
    config: DiagnosticConfig = DiagnosticConfig(),
) -> Tensor:
    for name, value in (
        ("sensitivity", sensitivity_energy),
        ("response", response_energy),
        ("representation_norm", representation_norm),
        ("update_norm", update_norm),
    ):
        checked_tensor(value, name)
        if (value < 0).any():
            raise ValueError(f"{name} cannot be negative")
    checked_tensor(task_loss, "task_loss")
    checked_tensor(ablated_losses, "ablated_losses")
    shape = sensitivity_energy.shape
    if any(
        (
            x.shape != shape
            for x in (response_energy, representation_norm, update_norm, ablated_losses)
        )
    ):
        raise ValueError("All layer statistics must have the same shape")
    sens = sensitivity_energy / (sensitivity_energy + config.sensitivity_scale)
    response = response_energy / (representation_norm * update_norm + config.epsilon)
    perf = torch.sigmoid(
        (ablated_losses - task_loss)
        / (config.performance_temperature * (task_loss.abs() + config.epsilon))
    )
    return torch.stack([sens, response, perf]).clamp(
        config.score_epsilon, 1 - config.score_epsilon
    )


def measure_private_information(
    base: nn.Module,
    task: nn.Module,
    examples: Sequence[Any],
    forward: Callable[[nn.Module, Any], Any],
    loss: Callable[[Any, Any], Tensor],
    layer_modules: Sequence[str],
    *,
    split: str = "type_cal",
    config: DiagnosticConfig = DiagnosticConfig(),
) -> PrivateInformation:
    if split not in {"train", "type_cal"}:
        raise ValueError("Private information must not use evaluation/test data")
    if (
        not examples
        or not layer_modules
        or len(set(layer_modules)) != len(layer_modules)
    ):
        raise ValueError("Provide nonempty examples and unique layer modules")
    base_modules, task_modules = (
        dict(base.named_modules()),
        dict(task.named_modules()),
    )
    for name in layer_modules:
        if not isinstance(base_modules.get(name), nn.Linear) or not isinstance(
            task_modules.get(name), nn.Linear
        ):
            raise ValueError(f"{name!r} must identify a Linear module in both models")
    parameters = [task_modules[name].weight for name in layer_modules]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Tied selected weights require a single canonical layer")
    originals = [p.detach().clone() for p in parameters]
    base_weights = [
        base_modules[name].weight.detach().to(p)
        for name, p in zip(layer_modules, parameters)
    ]
    if any((a.shape != b.shape for a, b in zip(originals, base_weights))):
        raise ValueError("Base and task layer shapes differ")
    updates = [a - b for a, b in zip(originals, base_weights)]
    length = len(parameters)
    energy = torch.zeros(length, dtype=torch.float64)
    response = torch.zeros_like(energy)
    repr_norm = torch.zeros_like(energy)
    update_norm = torch.tensor(
        [float(d.double().square().sum()) for d in updates], dtype=torch.float64
    )
    grad_mean = [torch.zeros_like(p) for p in parameters]
    grad_energy = torch.zeros_like(energy)
    modes = [
        (module, module.training)
        for model in (base, task)
        for module in model.modules()
    ]
    flags = [p.requires_grad for p in parameters]
    handles = []
    losses = []
    ablated = torch.zeros_like(energy)

    def hook(index: int):

        def collect(module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            h = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).double()
            delta = updates[index].to(h)
            response[index] += (h @ delta.T).square().sum().cpu()
            repr_norm[index] += h.square().sum().cpu()

        return collect

    try:
        base.eval()
        task.eval()
        for parameter in parameters:
            parameter.requires_grad_(True)
        for index, name in enumerate(layer_modules):
            handles.append(base_modules[name].register_forward_pre_hook(hook(index)))
        for example in examples:
            with torch.no_grad():
                forward(base, example)
            value = loss(forward(task, example), example)
            if value.numel() != 1 or not torch.isfinite(value):
                raise ValueError("Local loss must be a finite scalar")
            losses.append(float(value.detach()))
            gradients = torch.autograd.grad(value, parameters, allow_unused=True)
            for index, (gradient, delta) in enumerate(zip(gradients, updates)):
                if gradient is not None:
                    energy[index] += (
                        (gradient.detach().double().square() * delta.double().square())
                        .sum()
                        .cpu()
                    )
                    grad_mean[index] += gradient.detach() / len(examples)
                    grad_energy[index] += (
                        gradient.detach().double().square().sum().cpu() / len(examples)
                    )
        for index, parameter in enumerate(parameters):
            with torch.no_grad():
                parameter.copy_(base_weights[index])
                try:
                    values = [float(loss(forward(task, e), e)) for e in examples]
                    ablated[index] = sum(values) / len(values)
                finally:
                    parameter.copy_(originals[index])
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for parameter, original, flag in zip(parameters, originals, flags):
                parameter.copy_(original)
                parameter.requires_grad_(flag)
        for module, training in modes:
            module.training = training
    energy /= len(examples)
    task_loss = torch.tensor(sum(losses) / len(losses), dtype=torch.float64)
    scores = scores_from_statistics(
        energy, response, repr_norm, update_norm, task_loss, ablated, config
    )
    stability = torch.tensor(
        [float(g.double().square().sum()) for g in grad_mean], dtype=torch.float64
    )
    stability = (stability / (grad_energy + config.epsilon)).clamp(
        config.score_epsilon, 1 - config.score_epsilon
    )
    return PrivateInformation(
        scores,
        tuple(layer_modules),
        len(examples),
        energy,
        response,
        float(task_loss),
        ablated,
        stability,
    )
