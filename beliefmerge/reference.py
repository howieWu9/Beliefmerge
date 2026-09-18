from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np

Array = np.ndarray
PriorFamily = Literal["gaussian", "laplace", "student_t"]
MappingKind = Literal["nonlinear", "linear"]
StrategyMode = Literal["full", "nash_only", "bayesian_only", "uniform"]
PAPER_CHECKPOINT_STEPS: tuple[int, ...] = tuple(range(50, 2001, 50))
PAPER_TRAJECTORY_HISTORIES: Mapping[str, tuple[int, ...]] = {
    "final_only": (2000,),
    "one_intermediate": (1000, 2000),
    "two_intermediates": (500, 1000, 2000),
    "full_step_50": PAPER_CHECKPOINT_STEPS,
    "full_trajectory": PAPER_CHECKPOINT_STEPS,
}
PAPER_HYPERPARAMETER_GRID: Mapping[str, tuple[float | int, ...]] = {
    "rank": (1, 2, 4, 6, 8),
    "lambda_t": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "lambda_o": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "lambda_b": (0.0, 0.02, 0.04, 0.06, 0.08, 0.1),
}


def _array(value: Any, *, dtype: np.dtype | type | None = np.float64) -> Array:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        try:
            current = current.numpy()
        except (TypeError, RuntimeError):
            if not hasattr(current, "float"):
                raise
            current = current.float().numpy()
    array = np.asarray(current)
    if dtype is np.float64 and array.dtype.kind == "c":
        dtype = np.complex128
    return np.asarray(array, dtype=dtype)


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple((int(dimension) for dimension in shape))
    return tuple(np.asarray(value).shape)


def _is_floating(value: Any) -> bool:
    floating_check = getattr(value, "is_floating_point", None)
    if callable(floating_check):
        if bool(floating_check()):
            return True
        complex_check = getattr(value, "is_complex", None)
        return bool(complex_check()) if callable(complex_check) else False
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        try:
            return np.dtype(dtype).kind in "fc"
        except TypeError:
            pass
    try:
        return _array(value, dtype=None).dtype.kind in "fc"
    except (TypeError, ValueError):
        return False


def _clone(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    return np.array(value, copy=True)


def _softmax(values: Array, axis: int = 0) -> Array:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.maximum(
        exponential.sum(axis=axis, keepdims=True), np.finfo(np.float64).tiny
    )


def _sigmoid(values: Array) -> Array:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _logit(values: Array, epsilon: float = 1e-05) -> Array:
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def _regularize_covariance(covariance: Array, epsilon: float) -> Array:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return eigenvectors * eigenvalues[None, :] @ eigenvectors.T


def _factor(covariance: Array, epsilon: float) -> Array:
    return np.linalg.cholesky(_regularize_covariance(covariance, epsilon))


def _logical_layer_name(key: str) -> str:
    pieces = key.split(".")
    parameter_suffixes = {
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    }
    if len(pieces) > 1 and pieces[-1] in parameter_suffixes:
        return ".".join(pieces[:-1])
    return key


def _direction_features(gram: Array, rank: int, epsilon: float) -> Array:
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = np.maximum(norms[:, None] * norms[None, :], epsilon)
    cosine = 0.5 * (gram / denominator + (gram / denominator).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cosine)
    chosen_rank = min(rank, gram.shape[0])
    selected = np.arange(gram.shape[0] - chosen_rank, gram.shape[0])
    values = np.maximum(eigenvalues[selected], 0.0)
    directions = eigenvectors[:, selected] * np.sqrt(values)[None, :]
    for component in range(directions.shape[1]):
        pivot = int(np.argmax(np.abs(directions[:, component])))
        if directions[pivot, component] < 0.0:
            directions[:, component] *= -1.0
    if chosen_rank < rank:
        directions = np.pad(directions, ((0, 0), (0, rank - chosen_rank)))
    return directions


def _projected_gram(gram: Array, rank: int, epsilon: float) -> Array:
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = np.maximum(norms[:, None] * norms[None, :], epsilon)
    cosine = 0.5 * (gram / denominator + (gram / denominator).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cosine)
    positive = np.flatnonzero(eigenvalues > epsilon)
    if not positive.size:
        return np.zeros_like(gram)
    selected = positive[-min(rank, positive.size) :]
    basis = eigenvectors[:, selected]
    normalized = basis * eigenvalues[selected][None, :] @ basis.T
    return norms[:, None] * normalized * norms[None, :]


def _public_from_gram(
    gram: Array, base_norm_sq: float, rank: int, epsilon: float
) -> Array:
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    magnitude = norms / (math.sqrt(max(base_norm_sq, 0.0)) + epsilon)
    directions = _direction_features(gram, rank, epsilon)
    if gram.shape[0] == 1:
        compatibility = np.ones(1, dtype=np.float64)
    else:
        denominator = np.maximum(norms[:, None] * norms[None, :], epsilon)
        cosine = np.clip(gram / denominator, -1.0, 1.0)
        compatibility = (cosine.sum(axis=1) - np.diag(cosine)) / (gram.shape[0] - 1)
    return np.concatenate(
        (magnitude[:, None], directions, compatibility[:, None]), axis=1
    )


@dataclass(frozen=True)
class PublicGeometry:
    layer_names: tuple[str, ...]
    parameter_to_layer: Mapping[str, int]
    grams: Array
    projected_grams: Array
    global_gram: Array
    layer_base_norm_sq: Array
    global_base_norm_sq: float
    public_features: Array
    global_public_features: Array
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_names": list(self.layer_names),
            "parameter_to_layer": dict(self.parameter_to_layer),
            "grams": self.grams.tolist(),
            "projected_grams": self.projected_grams.tolist(),
            "global_gram": self.global_gram.tolist(),
            "layer_base_norm_sq": self.layer_base_norm_sq.tolist(),
            "global_base_norm_sq": self.global_base_norm_sq,
            "public_features": self.public_features.tolist(),
            "global_public_features": self.global_public_features.tolist(),
            "feature_order": [
                "g",
                *[f"z_{index}" for index in range(self.rank)],
                "compatibility",
            ],
            "rank": self.rank,
        }


def build_public_geometry(
    base_state: Mapping[str, Any],
    endpoint_states: Sequence[Mapping[str, Any]],
    *,
    rank: int = 4,
    layer_map: Mapping[str, str] | Callable[[str], str] | None = None,
    epsilon: float = 1e-08,
) -> PublicGeometry:
    if not endpoint_states:
        raise ValueError("endpoint_states cannot be empty")
    if rank < 1 or epsilon <= 0.0:
        raise ValueError("rank and epsilon must be positive")
    base_keys = tuple(base_state.keys())
    base_key_set = set(base_keys)
    for index, endpoint in enumerate(endpoint_states):
        if set(endpoint) != base_key_set:
            missing = sorted(base_key_set - set(endpoint))
            extra = sorted(set(endpoint) - base_key_set)
            raise ValueError(
                f"endpoint {index} keys differ from base; missing={missing[:5]}, extra={extra[:5]}"
            )

    def resolve(key: str) -> str:
        if layer_map is None:
            return _logical_layer_name(key)
        if callable(layer_map):
            return str(layer_map(key))
        if key not in layer_map:
            raise KeyError(f"layer_map is missing floating parameter {key!r}")
        return str(layer_map[key])

    names: list[str] = []
    name_to_index: dict[str, int] = {}
    parameter_to_layer: dict[str, int] = {}
    floating_keys: list[str] = []
    for key in base_keys:
        if not _is_floating(base_state[key]):
            continue
        layer_name = resolve(key)
        if layer_name not in name_to_index:
            name_to_index[layer_name] = len(names)
            names.append(layer_name)
        parameter_to_layer[key] = name_to_index[layer_name]
        floating_keys.append(key)
    if not names:
        raise ValueError("base_state has no floating-point parameters")
    model_count = len(endpoint_states)
    grams = np.zeros((len(names), model_count, model_count), dtype=np.float64)
    base_norms = np.zeros(len(names), dtype=np.float64)
    for key in floating_keys:
        base = _array(base_state[key])
        layer_index = parameter_to_layer[key]
        base_norms[layer_index] += float(
            np.vdot(base.reshape(-1), base.reshape(-1)).real
        )
        updates: list[Array] = []
        for model_index, endpoint in enumerate(endpoint_states):
            value = _array(endpoint[key])
            if value.shape != base.shape:
                raise ValueError(
                    f"shape mismatch for {key!r} in endpoint {model_index}: expected {base.shape}, got {value.shape}"
                )
            updates.append((value - base).reshape(-1))
        for left in range(model_count):
            for right in range(left, model_count):
                dot = float(np.vdot(updates[left], updates[right]).real)
                grams[layer_index, left, right] += dot
                if left != right:
                    grams[layer_index, right, left] += dot
    projected = np.stack(
        [_projected_gram(gram, rank, epsilon) for gram in grams], axis=0
    )
    features = np.stack(
        [
            _public_from_gram(gram, base_norms[index], rank, epsilon)
            for index, gram in enumerate(grams)
        ],
        axis=1,
    )
    global_gram = grams.sum(axis=0)
    global_base_norm = float(base_norms.sum())
    global_features = _public_from_gram(global_gram, global_base_norm, rank, epsilon)
    return PublicGeometry(
        layer_names=tuple(names),
        parameter_to_layer=parameter_to_layer,
        grams=grams,
        projected_grams=projected,
        global_gram=global_gram,
        layer_base_norm_sq=base_norms,
        global_base_norm_sq=global_base_norm,
        public_features=features,
        global_public_features=global_features,
        rank=rank,
    )


@dataclass(frozen=True)
class AblationConfig:
    rank: int = 4
    use_g: bool = True
    use_z: bool = True
    use_compatibility: bool = False
    use_sensitivity: bool = True
    use_representation: bool = True
    use_utility: bool = True
    use_stability: bool = False
    mapping_kind: MappingKind = "nonlinear"
    bayesian_update: bool = True
    independent_belief: bool = False
    shared_latent: bool = True
    prior_family: PriorFamily = "gaussian"
    student_df: float = 3.0
    strategy_mode: StrategyMode = "full"
    lambda_t: float = 0.6
    lambda_o: float = 0.4
    lambda_b: float = 0.04
    action_bound: float = 4.0
    type_response: float = 2.0
    monte_carlo_samples: int = 64
    max_iterations: int = 200
    min_iterations: int = 50
    trace_interval: int = 50
    best_response_grid: int = 33
    relaxation: float = 0.5
    convergence_tolerance: float = 1e-05
    epsilon: float = 1e-08
    seed: int = 42

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.mapping_kind not in ("nonlinear", "linear"):
            raise ValueError("mapping_kind must be nonlinear or linear")
        if self.prior_family not in ("gaussian", "laplace", "student_t"):
            raise ValueError("unsupported prior_family")
        if self.student_df <= 2.0:
            raise ValueError("student_df must exceed 2 for finite covariance")
        if self.strategy_mode not in ("full", "nash_only", "bayesian_only", "uniform"):
            raise ValueError("unsupported strategy_mode")
        if min(self.lambda_t, self.lambda_o, self.lambda_b) < 0.0:
            raise ValueError("lambda values must be non-negative")
        if self.action_bound <= 0.0 or self.type_response < 0.0:
            raise ValueError("action bounds/responses are invalid")
        if self.monte_carlo_samples < 1:
            raise ValueError("monte_carlo_samples must be >= 1")
        if not 1 <= self.min_iterations <= self.max_iterations:
            raise ValueError("require 1 <= min_iterations <= max_iterations")
        if self.trace_interval < 1 or self.best_response_grid < 3:
            raise ValueError("trace_interval/grid are too small")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0,1]")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    @classmethod
    def variant(cls, name: str, **overrides: Any) -> "AblationConfig":
        for alias, canonical in (
            ("lambda_type", "lambda_t"),
            ("lambda_public", "lambda_o"),
            ("lambda_action", "lambda_b"),
        ):
            if alias not in overrides:
                continue
            value = overrides.pop(alias)
            if canonical in overrides and overrides[canonical] != value:
                raise ValueError(
                    f"conflicting overrides for {canonical!r} and {alias!r}"
                )
            overrides[canonical] = value
        normalized = (
            name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus_")
            .replace("w/o", "without")
        )
        aliases: dict[str, dict[str, Any]] = {
            "full": {},
            "ours": {},
            "without_g": {"use_g": False},
            "no_g": {"use_g": False},
            "public_without_update_magnitude": {"use_g": False},
            "public_without_magnitude": {"use_g": False},
            "without_z": {"use_z": False},
            "no_z": {"use_z": False},
            "public_without_update_direction": {"use_z": False},
            "public_without_direction": {"use_z": False},
            "plus_compat": {"use_compatibility": True},
            "plus_compatibility": {"use_compatibility": True},
            "public_plus_compatibility": {"use_compatibility": True},
            "public_with_compatibility": {"use_compatibility": True},
            "without_sens": {"use_sensitivity": False},
            "no_sens": {"use_sensitivity": False},
            "private_without_importance": {"use_sensitivity": False},
            "private_without_sensitivity": {"use_sensitivity": False},
            "without_repr": {"use_representation": False},
            "no_repr": {"use_representation": False},
            "private_without_relevance": {"use_representation": False},
            "private_without_representation": {"use_representation": False},
            "without_util": {"use_utility": False},
            "no_util": {"use_utility": False},
            "private_without_effectiveness": {"use_utility": False},
            "private_without_utility": {"use_utility": False},
            "plus_stab": {"use_stability": True},
            "plus_stability": {"use_stability": True},
            "private_plus_stability": {"use_stability": True},
            "private_with_stability": {"use_stability": True},
            "linear": {"mapping_kind": "linear"},
            "linear_map": {"mapping_kind": "linear"},
            "linear_mapping": {"mapping_kind": "linear"},
            "linear_public_mapping": {"mapping_kind": "linear"},
            "no_bayesian_update": {"bayesian_update": False},
            "without_bayesian_update": {"bayesian_update": False},
            "prior_without_bayesian_update": {"bayesian_update": False},
            "independent": {"independent_belief": True},
            "independent_belief": {"independent_belief": True},
            "no_shared": {"shared_latent": False},
            "without_shared": {"shared_latent": False},
            "without_shared_latent": {"shared_latent": False},
            "gaussian": {"prior_family": "gaussian"},
            "gaussian_prior": {"prior_family": "gaussian"},
            "laplace": {"prior_family": "laplace"},
            "laplace_prior": {"prior_family": "laplace"},
            "student_t": {"prior_family": "student_t", "student_df": 3.0},
            "student_t_prior_nu3": {"prior_family": "student_t", "student_df": 3.0},
            "nash_only": {"strategy_mode": "nash_only", "bayesian_update": False},
            "without_bayesian_inference": {
                "strategy_mode": "nash_only",
                "bayesian_update": False,
            },
            "bayesian_only": {"strategy_mode": "bayesian_only"},
            "without_strategic_fusion": {"strategy_mode": "uniform"},
            "uniform": {"strategy_mode": "uniform"},
            "uniform_without_strategy": {"strategy_mode": "uniform"},
        }
        if normalized not in aliases:
            raise KeyError(f"unknown paper ablation {name!r}")
        return cls(**{**aliases[normalized], **overrides})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PAPER_TABLE_VARIANTS: Mapping[str, tuple[str, ...]] = {
    "table4": (
        "ours",
        "public_without_update_magnitude",
        "public_without_update_direction",
        "public_plus_compatibility",
        "private_without_importance",
        "private_without_relevance",
        "private_without_effectiveness",
        "private_plus_stability",
    ),
    "table5": (
        "ours",
        "linear_mapping",
        "without_bayesian_update",
        "independent_belief",
    ),
    "table6": ("without_shared_latent", "gaussian", "laplace", "student_t"),
    "table7": ("ours", "nash_only", "bayesian_only", "uniform_without_strategy"),
}


def enumerate_paper_configs(
    table: str | int, **common_overrides: Any
) -> tuple[tuple[str, AblationConfig], ...]:
    key = str(table).strip().lower().replace("_", "")
    if key.isdigit():
        key = f"table{key}"
    if key in {"table8", "table9"}:
        raise ValueError(
            f"{key} varies run metadata; use PAPER_CHECKPOINT_STEPS or PAPER_TRAJECTORY_HISTORIES"
        )
    if key == "table10":
        configs: list[tuple[str, AblationConfig]] = []
        for field_name, values in PAPER_HYPERPARAMETER_GRID.items():
            for value in values:
                label = f"{field_name}={value:g}"
                configs.append(
                    (label, AblationConfig(**{**common_overrides, field_name: value}))
                )
        return tuple(configs)
    if key not in PAPER_TABLE_VARIANTS:
        raise KeyError(f"unknown paper table {table!r}")
    return tuple(
        (
            (name, AblationConfig.variant(name, **common_overrides))
            for name in PAPER_TABLE_VARIANTS[key]
        )
    )


@dataclass(frozen=True)
class PublicTypeMapper:
    feature_mean: Array
    feature_scale: Array
    linear_coefficients: Array
    nonlinear_projection: Array
    nonlinear_bias: Array
    nonlinear_coefficients: Array
    component_count: int
    fitted: bool = True

    @classmethod
    def default(
        cls,
        feature_count: int,
        component_count: int,
        *,
        seed: int = 0,
        hidden_features: int = 16,
    ) -> "PublicTypeMapper":
        if feature_count < 3 or component_count not in (3, 4):
            raise ValueError("invalid default mapper dimensions")
        rng = np.random.default_rng(seed)
        linear = np.zeros((feature_count + 1, component_count), dtype=np.float64)
        linear[0] = np.array([0.0, 0.1, 0.0, 0.2][:component_count])
        linear[1, :component_count] = np.array([0.8, 0.2, -0.2, -0.6])[:component_count]
        linear[2:feature_count, :component_count] = 0.15
        linear[-1, :component_count] += np.array([0.1, 0.8, 0.7, 0.6])[:component_count]
        projection = rng.normal(
            scale=1.0 / math.sqrt(feature_count), size=(feature_count, hidden_features)
        )
        bias = rng.uniform(-0.5, 0.5, size=hidden_features)
        nonlinear = rng.normal(scale=0.04, size=(hidden_features, component_count))
        return cls(
            feature_mean=np.zeros(feature_count),
            feature_scale=np.ones(feature_count),
            linear_coefficients=linear,
            nonlinear_projection=projection,
            nonlinear_bias=bias,
            nonlinear_coefficients=nonlinear,
            component_count=component_count,
            fitted=False,
        )

    @classmethod
    def fit(
        cls,
        public_features: Array,
        private_types: Array,
        *,
        ridge: float = 0.001,
        hidden_features: int = 32,
        seed: int = 0,
    ) -> "PublicTypeMapper":
        features = np.asarray(public_features, dtype=np.float64)
        targets = np.asarray(private_types, dtype=np.float64)
        if features.shape[:-1] != targets.shape[:-1]:
            raise ValueError("public/private calibration prefixes must match")
        if targets.shape[-1] not in (3, 4):
            raise ValueError("private calibration needs 3 or 4 components")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("public/private calibration values must be finite")
        if np.any(targets <= 0.0) or np.any(targets >= 1.0):
            raise ValueError("private calibration targets must lie in (0,1)")
        if ridge <= 0.0 or hidden_features < 1:
            raise ValueError("ridge/hidden_features are invalid")
        x = features.reshape(-1, features.shape[-1])
        y = _logit(targets.reshape(-1, targets.shape[-1]))
        if x.shape[0] < 2 or not np.isfinite(x).all() or (not np.isfinite(y).all()):
            raise ValueError("calibration arrays are too small or non-finite")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-08] = 1.0
        standardized = (x - mean) / scale
        design = np.concatenate((np.ones((x.shape[0], 1)), standardized), axis=1)
        penalty = ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        linear = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        rng = np.random.default_rng(seed)
        projection = rng.normal(
            scale=1.0 / math.sqrt(x.shape[1]), size=(x.shape[1], hidden_features)
        )
        bias = rng.uniform(-math.pi, math.pi, size=hidden_features)
        hidden = np.tanh(standardized @ projection + bias)
        residual = y - design @ linear
        nonlinear = np.linalg.solve(
            hidden.T @ hidden + ridge * np.eye(hidden_features), hidden.T @ residual
        )
        return cls(
            feature_mean=mean,
            feature_scale=scale,
            linear_coefficients=linear,
            nonlinear_projection=projection,
            nonlinear_bias=bias,
            nonlinear_coefficients=nonlinear,
            component_count=targets.shape[-1],
            fitted=True,
        )

    def predict(self, features: Array, kind: MappingKind = "nonlinear") -> Array:
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1] != self.feature_mean.size:
            raise ValueError(
                f"mapper expects {self.feature_mean.size} features, got {values.shape[-1]}"
            )
        flat = values.reshape(-1, values.shape[-1])
        standardized = (flat - self.feature_mean) / self.feature_scale
        design = np.concatenate((np.ones((flat.shape[0], 1)), standardized), axis=1)
        logits = design @ self.linear_coefficients
        if kind == "nonlinear":
            hidden = np.tanh(
                standardized @ self.nonlinear_projection + self.nonlinear_bias
            )
            logits += hidden @ self.nonlinear_coefficients
        elif kind != "linear":
            raise ValueError("kind must be nonlinear or linear")
        output = _sigmoid(logits)
        return output.reshape(*values.shape[:-1], self.component_count)

    def validate(self, feature_count: int, component_count: int) -> "PublicTypeMapper":
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        linear = np.asarray(self.linear_coefficients, dtype=np.float64)
        projection = np.asarray(self.nonlinear_projection, dtype=np.float64)
        bias = np.asarray(self.nonlinear_bias, dtype=np.float64)
        nonlinear = np.asarray(self.nonlinear_coefficients, dtype=np.float64)
        if self.component_count != component_count:
            raise ValueError("mapper/private component counts differ")
        if mean.shape != (feature_count,) or scale.shape != (feature_count,):
            raise ValueError("mapper feature normalization has the wrong shape")
        if linear.shape != (feature_count + 1, component_count):
            raise ValueError("mapper linear coefficients have the wrong shape")
        if projection.ndim != 2 or projection.shape[0] != feature_count:
            raise ValueError("mapper nonlinear projection has the wrong shape")
        hidden_count = projection.shape[1]
        if bias.shape != (hidden_count,) or nonlinear.shape != (
            hidden_count,
            component_count,
        ):
            raise ValueError("mapper nonlinear coefficients have the wrong shape")
        arrays = (mean, scale, linear, projection, bias, nonlinear)
        if not all((np.isfinite(array).all() for array in arrays)):
            raise ValueError("mapper parameters must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("mapper feature scales must be positive")
        return replace(
            self,
            feature_mean=mean,
            feature_scale=scale,
            linear_coefficients=linear,
            nonlinear_projection=projection,
            nonlinear_bias=bias,
            nonlinear_coefficients=nonlinear,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "linear_coefficients": self.linear_coefficients.tolist(),
            "nonlinear_projection": self.nonlinear_projection.tolist(),
            "nonlinear_bias": self.nonlinear_bias.tolist(),
            "nonlinear_coefficients": self.nonlinear_coefficients.tolist(),
            "component_count": self.component_count,
            "fitted": self.fitted,
        }


@dataclass(frozen=True)
class BeliefParameters:
    shared_mean: Array
    shared_covariance: Array
    private_covariance: Array
    calibration_observations: int = 0

    @classmethod
    def default(cls, component_count: int) -> "BeliefParameters":
        if component_count not in (3, 4):
            raise ValueError("component_count must be 3 or 4")
        return cls(
            shared_mean=np.zeros(component_count, dtype=np.float64),
            shared_covariance=np.eye(component_count, dtype=np.float64) * 0.025,
            private_covariance=np.eye(component_count, dtype=np.float64) * 0.05,
            calibration_observations=0,
        )

    def validate(self, component_count: int, epsilon: float) -> "BeliefParameters":
        mean = np.asarray(self.shared_mean, dtype=np.float64)
        shared = np.asarray(self.shared_covariance, dtype=np.float64)
        private = np.asarray(self.private_covariance, dtype=np.float64)
        if mean.shape != (component_count,):
            raise ValueError("shared_mean shape does not match private types")
        if shared.shape != (component_count, component_count):
            raise ValueError("shared_covariance has the wrong shape")
        if private.shape != shared.shape:
            raise ValueError("private_covariance has the wrong shape")
        if not all((np.isfinite(value).all() for value in (mean, shared, private))):
            raise ValueError("belief parameters must be finite")
        return replace(
            self,
            shared_mean=mean,
            shared_covariance=_regularize_covariance(shared, epsilon),
            private_covariance=_regularize_covariance(private, epsilon),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_mean": np.asarray(self.shared_mean).tolist(),
            "shared_covariance": np.asarray(self.shared_covariance).tolist(),
            "private_covariance": np.asarray(self.private_covariance).tolist(),
            "calibration_observations": self.calibration_observations,
        }


def calibrate_belief_parameters(
    public_features: Array,
    private_types: Array,
    mapper: PublicTypeMapper,
    *,
    mapping_kind: MappingKind = "nonlinear",
    epsilon: float = 1e-05,
) -> BeliefParameters:
    features = np.asarray(public_features, dtype=np.float64)
    types = np.asarray(private_types, dtype=np.float64)
    if features.ndim != 4 or types.ndim != 4:
        raise ValueError("calibration expects [round,model,layer,feature/component]")
    if features.shape[:3] != types.shape[:3]:
        raise ValueError("calibration public/private dimensions differ")
    if types.shape[-1] not in (3, 4):
        raise ValueError("calibration private types need 3 or 4 components")
    if not np.isfinite(features).all() or not np.isfinite(types).all():
        raise ValueError("calibration arrays must be finite")
    if np.any(types <= 0.0) or np.any(types >= 1.0):
        raise ValueError("calibration private types must lie strictly in (0,1)")
    mapper = mapper.validate(features.shape[-1], types.shape[-1])
    prediction = mapper.predict(features, mapping_kind)
    residual = types - prediction
    group_count = features.shape[0] * features.shape[2]
    model_count = features.shape[1]
    if group_count < 2:
        raise ValueError("belief calibration needs at least two round-layer samples")
    if model_count < 2:
        raise ValueError("belief calibration needs at least two models")
    grouped = residual.transpose(0, 2, 1, 3).reshape(
        group_count, model_count, types.shape[-1]
    )
    shared_draws = grouped.mean(axis=1)
    shared_mean = shared_draws.mean(axis=0)
    centered = grouped - shared_draws[:, None, :]
    private_covariance = np.einsum("gmc,gmd->cd", centered, centered) / (
        group_count * (model_count - 1)
    )
    covariance_of_means = np.cov(shared_draws, rowvar=False, bias=False)
    shared_covariance = covariance_of_means - private_covariance / model_count
    return BeliefParameters(
        shared_mean=shared_mean,
        shared_covariance=_regularize_covariance(shared_covariance, epsilon),
        private_covariance=_regularize_covariance(private_covariance, epsilon),
        calibration_observations=int(np.prod(types.shape[:3])),
    )


def _robust_posterior(
    residual: Array,
    parameters: BeliefParameters,
    family: PriorFamily,
    student_df: float,
    epsilon: float,
) -> tuple[Array, Array]:
    shared_factor = _factor(parameters.shared_covariance, epsilon)
    private_precision = np.linalg.inv(parameters.private_covariance)
    transformed_precision = shared_factor.T @ private_precision @ shared_factor
    transformed_target = (
        shared_factor.T @ private_precision @ (residual - parameters.shared_mean)
    )
    coordinate = np.zeros(residual.size, dtype=np.float64)
    precision = np.ones(residual.size, dtype=np.float64)
    iterations = 1 if family == "gaussian" else 25
    for _ in range(iterations):
        if family == "gaussian":
            precision = np.ones_like(coordinate)
        elif family == "laplace":
            precision = math.sqrt(2.0) / np.sqrt(coordinate**2 + 0.0001)
        elif family == "student_t":
            precision = (student_df + 1.0) / (student_df + coordinate**2)
        else:
            raise ValueError(f"unsupported family {family}")
        new_coordinate = np.linalg.solve(
            transformed_precision + np.diag(precision), transformed_target
        )
        if np.max(np.abs(new_coordinate - coordinate)) < 1e-09:
            coordinate = new_coordinate
            break
        coordinate = new_coordinate
    coordinate_covariance = np.linalg.inv(
        transformed_precision + np.diag(np.maximum(precision, epsilon))
    )
    mean = parameters.shared_mean + shared_factor @ coordinate
    covariance = shared_factor @ coordinate_covariance @ shared_factor.T
    return (mean, _regularize_covariance(covariance, epsilon))


def _family_nll(
    value: Array,
    mean: Array,
    covariance: Array,
    family: PriorFamily,
    student_df: float,
    epsilon: float,
) -> float:
    factor = _factor(covariance, epsilon)
    whitened = np.linalg.solve(factor, value - mean)
    logdet = float(np.log(np.diag(factor)).sum())
    dimension = value.size
    if family == "gaussian":
        return (
            0.5 * float(whitened @ whitened)
            + logdet
            + 0.5 * dimension * math.log(2.0 * math.pi)
        )
    if family == "laplace":
        scale = 1.0 / math.sqrt(2.0)
        return (
            float(np.abs(whitened).sum()) / scale
            + dimension * math.log(2.0 * scale)
            + logdet
        )
    scale = math.sqrt((student_df - 2.0) / student_df)
    normalized = whitened / scale
    constant = (
        math.lgamma((student_df + 1.0) / 2.0)
        - math.lgamma(student_df / 2.0)
        - 0.5 * math.log(student_df * math.pi)
        - math.log(scale)
    )
    log_density = dimension * constant - 0.5 * (student_df + 1.0) * float(
        np.log1p(normalized**2 / student_df).sum()
    )
    return -log_density + logdet


@dataclass(frozen=True)
class BeliefResult:
    public_type_mean: Array
    private_types: Array
    posterior_shared_mean: Array
    posterior_shared_covariance: Array
    predictive_mean: Array
    predictive_covariance: Array
    pairwise_type_nll: Array
    type_nll: float
    parameters: BeliefParameters
    config: AblationConfig

    def sample_joint(
        self,
        observer: int,
        sample_count: int,
        rng: np.random.Generator,
        *,
        conditional: bool = True,
    ) -> Array:
        model_count, layer_count, component_count = self.private_types.shape
        draws = np.empty(
            (sample_count, model_count, layer_count, component_count), dtype=np.float64
        )
        independent = self.config.independent_belief
        if not conditional or self.config.strategy_mode == "nash_only":
            independent = self.config.independent_belief
        for layer in range(layer_count):
            if conditional and self.config.strategy_mode != "nash_only":
                shared_mean = self.posterior_shared_mean[observer, layer]
                shared_covariance = self.posterior_shared_covariance[observer, layer]
            else:
                shared_mean = self.parameters.shared_mean
                shared_covariance = self.parameters.shared_covariance
            if not self.config.shared_latent or independent:
                shared = np.broadcast_to(
                    shared_mean, (sample_count, component_count)
                ).copy()
            else:
                shared = shared_mean + _sample_centered(
                    rng,
                    shared_covariance,
                    sample_count,
                    self.config.prior_family,
                    self.config.student_df,
                    self.config.epsilon,
                )
            for target in range(model_count):
                if independent:
                    target_mean = self.predictive_mean[observer, target, layer]
                    target_covariance = self.predictive_covariance[
                        observer, target, layer
                    ]
                    draws[:, target, layer] = target_mean + _sample_centered(
                        rng,
                        target_covariance,
                        sample_count,
                        self.config.prior_family,
                        self.config.student_df,
                        self.config.epsilon,
                    )
                else:
                    private_noise = _sample_centered(
                        rng,
                        self.parameters.private_covariance,
                        sample_count,
                        "gaussian",
                        self.config.student_df,
                        self.config.epsilon,
                    )
                    draws[:, target, layer] = (
                        self.public_type_mean[target, layer] + shared + private_noise
                    )
            draws[:, observer, layer] = self.private_types[observer, layer]
        return np.clip(draws, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        finite_pairwise = np.where(
            np.isfinite(self.pairwise_type_nll), self.pairwise_type_nll, None
        )
        return {
            "public_type_mean": self.public_type_mean.tolist(),
            "private_types": self.private_types.tolist(),
            "posterior_shared_mean": self.posterior_shared_mean.tolist(),
            "posterior_shared_covariance": self.posterior_shared_covariance.tolist(),
            "predictive_mean": self.predictive_mean.tolist(),
            "predictive_covariance": self.predictive_covariance.tolist(),
            "pairwise_type_nll": finite_pairwise.tolist(),
            "type_nll": self.type_nll,
            "nll_definition": "mean conditional NLL over observer!=target and layers",
            "parameters": self.parameters.to_dict(),
        }


def _sample_centered(
    rng: np.random.Generator,
    covariance: Array,
    count: int,
    family: PriorFamily,
    student_df: float,
    epsilon: float,
) -> Array:
    dimension = covariance.shape[0]
    if family == "gaussian":
        standardized = rng.normal(size=(count, dimension))
    elif family == "laplace":
        standardized = rng.laplace(scale=1.0 / math.sqrt(2.0), size=(count, dimension))
    elif family == "student_t":
        standardized = rng.standard_t(student_df, size=(count, dimension))
        standardized *= math.sqrt((student_df - 2.0) / student_df)
    else:
        raise ValueError(f"unsupported family {family}")
    return standardized @ _factor(covariance, epsilon).T


def _build_beliefs(
    public_type_mean: Array,
    private_types: Array,
    parameters: BeliefParameters,
    config: AblationConfig,
) -> BeliefResult:
    model_count, layer_count, component_count = private_types.shape
    posterior_mean = np.zeros(
        (model_count, layer_count, component_count), dtype=np.float64
    )
    posterior_covariance = np.zeros(
        (model_count, layer_count, component_count, component_count), dtype=np.float64
    )
    predictive_mean = np.zeros(
        (model_count, model_count, layer_count, component_count), dtype=np.float64
    )
    predictive_covariance = np.zeros(
        (model_count, model_count, layer_count, component_count, component_count),
        dtype=np.float64,
    )
    pairwise_nll = np.full(
        (model_count, model_count, layer_count), np.nan, dtype=np.float64
    )
    for observer in range(model_count):
        for layer in range(layer_count):
            if not config.shared_latent:
                mean = np.zeros(component_count)
                covariance = np.zeros((component_count, component_count))
            elif config.independent_belief:
                mean = parameters.shared_mean
                covariance = parameters.shared_covariance
            elif not config.bayesian_update or config.strategy_mode == "nash_only":
                mean = parameters.shared_mean
                covariance = parameters.shared_covariance
            else:
                residual = (
                    private_types[observer, layer] - public_type_mean[observer, layer]
                )
                mean, covariance = _robust_posterior(
                    residual,
                    parameters,
                    config.prior_family,
                    config.student_df,
                    config.epsilon,
                )
            posterior_mean[observer, layer] = mean
            posterior_covariance[observer, layer] = covariance
            for target in range(model_count):
                if target == observer:
                    predictive_mean[observer, target, layer] = private_types[
                        observer, layer
                    ]
                    predictive_covariance[observer, target, layer] = (
                        np.eye(component_count) * config.epsilon
                    )
                    continue
                if config.independent_belief or not config.shared_latent:
                    target_shared_mean = (
                        parameters.shared_mean
                        if config.shared_latent
                        else np.zeros(component_count)
                    )
                    target_shared_covariance = (
                        parameters.shared_covariance
                        if config.shared_latent
                        else np.zeros((component_count, component_count))
                    )
                else:
                    target_shared_mean = mean
                    target_shared_covariance = covariance
                target_mean = public_type_mean[target, layer] + target_shared_mean
                target_covariance = _regularize_covariance(
                    parameters.private_covariance + target_shared_covariance,
                    config.epsilon,
                )
                predictive_mean[observer, target, layer] = target_mean
                predictive_covariance[observer, target, layer] = target_covariance
                pairwise_nll[observer, target, layer] = _family_nll(
                    private_types[target, layer],
                    target_mean,
                    target_covariance,
                    config.prior_family,
                    config.student_df,
                    config.epsilon,
                )
    finite_nll = pairwise_nll[np.isfinite(pairwise_nll)]
    type_nll = float(finite_nll.mean()) if finite_nll.size else float("nan")
    return BeliefResult(
        public_type_mean=public_type_mean,
        private_types=private_types,
        posterior_shared_mean=posterior_mean,
        posterior_shared_covariance=posterior_covariance,
        predictive_mean=predictive_mean,
        predictive_covariance=predictive_covariance,
        pairwise_type_nll=pairwise_nll,
        type_nll=type_nll,
        parameters=parameters,
        config=config,
    )


def _masked_public_features(
    features: Array, config: AblationConfig, neutral_values: Array | None = None
) -> Array:
    masked = np.array(features, copy=True)
    neutral = (
        np.zeros(masked.shape[-1], dtype=np.float64)
        if neutral_values is None
        else np.asarray(neutral_values, dtype=np.float64)
    )
    if neutral.shape != (masked.shape[-1],):
        raise ValueError("neutral public-feature values have the wrong shape")
    if not config.use_g:
        masked[..., 0] = neutral[0]
    if not config.use_z:
        masked[..., 1:-1] = neutral[1:-1]
    if not config.use_compatibility:
        masked[..., -1] = neutral[-1]
    return masked


def _component_indices(component_count: int, config: AblationConfig) -> list[int]:
    selected: list[int] = []
    if config.use_sensitivity:
        selected.append(0)
    if config.use_representation:
        selected.append(1)
    if config.use_utility:
        selected.append(2)
    if config.use_stability:
        if component_count != 4:
            raise ValueError("+ stability requires private types with four components")
        selected.append(3)
    return selected


def _type_strength(types: Array, indices: Sequence[int]) -> Array:
    if not indices:
        return np.ones(types.shape[:-1], dtype=np.float64)
    return np.prod(np.take(types, indices, axis=-1), axis=-1)


def _utility_for_candidates(
    candidates: Array,
    opponent_actions: Array,
    model_index: int,
    gram: Array,
    projected_gram: Array,
    local_strength: float,
    config: AblationConfig,
) -> Array:
    sample_count, model_count = opponent_actions.shape
    profiles = np.broadcast_to(
        opponent_actions[None, :, :], (candidates.size, sample_count, model_count)
    ).copy()
    profiles[:, :, model_index] = candidates[:, None]
    weights = _softmax(profiles, axis=2)
    fused_norm = np.einsum("csm,mn,csn->cs", weights, gram, weights)
    gram_times_weight = np.einsum("mn,csn->csm", gram, weights)
    local_distance = (
        fused_norm
        - 2.0 * gram_times_weight[:, :, model_index]
        + gram[model_index, model_index]
    ) / (gram[model_index, model_index] + config.epsilon)
    projected_energy = np.einsum("csm,mn,csn->cs", weights, projected_gram, weights)
    public_support = projected_energy / (fused_norm + config.epsilon)
    utility = (
        -config.lambda_t * local_strength * local_distance
        + config.lambda_o * public_support
        - config.lambda_b * candidates[:, None] ** 2
    )
    return utility.mean(axis=1)


def _realized_utilities(
    actions: Array, strengths: Array, geometry: PublicGeometry, config: AblationConfig
) -> Array:
    model_count, layer_count = actions.shape
    utilities = np.zeros((model_count, layer_count), dtype=np.float64)
    for layer in range(layer_count):
        profile = actions[:, layer][None, :]
        for model in range(model_count):
            utilities[model, layer] = _utility_for_candidates(
                actions[model, layer : layer + 1],
                profile,
                model,
                geometry.grams[layer],
                geometry.projected_grams[layer],
                strengths[model, layer],
                config,
            )[0]
    return utilities


def _public_task_support(geometry: PublicGeometry, epsilon: float) -> Array:
    diagonal = np.diagonal(geometry.grams, axis1=1, axis2=2)
    projected_diagonal = np.diagonal(geometry.projected_grams, axis1=1, axis2=2)
    return (projected_diagonal / np.maximum(diagonal, epsilon)).T


def _solve_actions(
    geometry: PublicGeometry,
    belief: BeliefResult,
    private_types: Array,
    component_indices: Sequence[int],
    config: AblationConfig,
) -> tuple[Array, list[dict[str, Any]], Array, Array]:
    model_count, layer_count, _ = private_types.shape
    actual_strength = _type_strength(private_types, component_indices)
    public_support = _public_task_support(geometry, config.epsilon)
    trace: list[dict[str, Any]] = []
    if config.strategy_mode == "uniform":
        actions = np.zeros((model_count, layer_count), dtype=np.float64)
        utilities = _realized_utilities(actions, actual_strength, geometry, config)
        trace.append(_trace_record(0, actions, utilities, 0.0, 0.0, "closed_form"))
        return (actions, trace, utilities, np.zeros_like(actions))
    expected_competitors = np.zeros_like(actual_strength)
    for observer in range(model_count):
        expected_types = (
            belief.public_type_mean
            if config.strategy_mode == "nash_only"
            else belief.predictive_mean[observer]
        )
        expected = _type_strength(expected_types, component_indices)
        if model_count > 1:
            expected_competitors[observer] = (
                expected.sum(axis=0) - expected[observer]
            ) / (model_count - 1)
    standardized = actual_strength - expected_competitors
    standardized += public_support - public_support.mean(axis=0, keepdims=True)
    initial_actions = config.action_bound * _sigmoid(
        config.lambda_t * standardized + config.lambda_o * public_support
    )
    if config.strategy_mode == "bayesian_only":
        actions = initial_actions
        utilities = _realized_utilities(actions, actual_strength, geometry, config)
        trace.append(_trace_record(0, actions, utilities, 0.0, 0.0, "closed_form"))
        return (actions, trace, utilities, np.zeros_like(actions))
    intercepts = np.clip(
        initial_actions - config.type_response * actual_strength,
        -config.action_bound,
        config.action_bound,
    )
    rng = np.random.default_rng(config.seed)
    if config.strategy_mode == "nash_only":
        joint_samples = []
        for observer in range(model_count):
            samples = np.broadcast_to(
                belief.public_type_mean,
                (
                    config.monte_carlo_samples,
                    model_count,
                    layer_count,
                    private_types.shape[-1],
                ),
            ).copy()
            samples[:, observer] = private_types[observer]
            joint_samples.append(samples)
    else:
        joint_samples = [
            belief.sample_joint(
                observer, config.monte_carlo_samples, rng, conditional=True
            )
            for observer in range(model_count)
        ]
    sample_strengths = [
        _type_strength(sample, component_indices) for sample in joint_samples
    ]
    grid = np.linspace(0.0, config.action_bound, config.best_response_grid)
    actions = np.clip(
        intercepts + config.type_response * actual_strength, 0.0, config.action_bound
    )
    utilities = _realized_utilities(actions, actual_strength, geometry, config)
    trace.append(_trace_record(0, actions, utilities, 0.0, None, "initial"))
    final_gap = np.zeros_like(actions)

    def best_response_snapshot(
        current_actions: Array, current_intercepts: Array
    ) -> tuple[Array, Array, Array]:
        best = np.empty_like(current_actions)
        expected = np.empty_like(current_actions)
        gaps = np.empty_like(current_actions)
        for observer in range(model_count):
            predicted = np.clip(
                current_intercepts[None, :, :]
                + config.type_response * sample_strengths[observer],
                0.0,
                config.action_bound,
            )
            predicted[:, observer, :] = current_actions[observer]
            for layer in range(layer_count):
                values = _utility_for_candidates(
                    grid,
                    predicted[:, :, layer],
                    observer,
                    geometry.grams[layer],
                    geometry.projected_grams[layer],
                    actual_strength[observer, layer],
                    config,
                )
                best_index = int(np.argmax(values))
                best[observer, layer] = grid[best_index]
                current_value = _utility_for_candidates(
                    current_actions[observer, layer : layer + 1],
                    predicted[:, :, layer],
                    observer,
                    geometry.grams[layer],
                    geometry.projected_grams[layer],
                    actual_strength[observer, layer],
                    config,
                )[0]
                expected[observer, layer] = current_value
                gaps[observer, layer] = max(0.0, values[best_index] - current_value)
        return (best, expected, gaps)

    for iteration in range(1, config.max_iterations + 1):
        best_actions, expected_utility, gaps = best_response_snapshot(
            actions, intercepts
        )
        new_actions = (
            1.0 - config.relaxation
        ) * actions + config.relaxation * best_actions
        action_change = float(np.max(np.abs(new_actions - actions)))
        actions = new_actions
        intercepts = np.clip(
            actions - config.type_response * actual_strength,
            -config.action_bound,
            config.action_bound,
        )
        should_trace = iteration % config.trace_interval == 0
        converged = (
            iteration >= config.min_iterations
            and action_change < config.convergence_tolerance
        )
        if should_trace or iteration == config.max_iterations or converged:
            _, expected_utility, final_gap = best_response_snapshot(actions, intercepts)
            utilities = _realized_utilities(actions, actual_strength, geometry, config)
            trace.append(
                _trace_record(
                    iteration,
                    actions,
                    utilities,
                    action_change,
                    float(final_gap.max()),
                    "converged" if converged else "iteration",
                    expected_utility,
                )
            )
        if converged:
            break
    utilities = _realized_utilities(actions, actual_strength, geometry, config)
    return (actions, trace, utilities, final_gap)


def _trace_record(
    step: int,
    actions: Array,
    utilities: Array,
    action_change: float,
    exploitability: float | None,
    status: str,
    expected_utility: Array | None = None,
) -> dict[str, Any]:
    weights = _softmax(actions, axis=0)
    record: dict[str, Any] = {
        "step": int(step),
        "status": status,
        "max_action_change": float(action_change),
        "max_expected_best_response_gap": None
        if exploitability is None
        else float(exploitability),
        "actions": actions.tolist(),
        "weights": weights.tolist(),
        "realized_layer_utilities": utilities.tolist(),
        "mean_realized_utility": float(utilities.mean()),
    }
    if expected_utility is not None:
        record["expected_layer_utilities"] = expected_utility.tolist()
        record["mean_expected_utility"] = float(expected_utility.mean())
    return record


@dataclass
class MergeResult:
    state_dict: Mapping[str, Any]
    actions: Array
    layer_weights: Array
    model_weights: Array
    geometry: PublicGeometry
    belief: BeliefResult
    public_type_mean: Array
    type_strength: Array
    realized_utilities: Array
    best_response_gap: Array
    optimization_trace: list[dict[str, Any]]
    mapper: PublicTypeMapper
    config: AblationConfig
    model_names: tuple[str, ...]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_names": list(self.model_names),
            "config": self.config.to_dict(),
            "mapper": self.mapper.to_dict(),
            "geometry": self.geometry.to_dict(),
            "belief": self.belief.to_dict(),
            "public_type_mean": self.public_type_mean.tolist(),
            "type_strength": self.type_strength.tolist(),
            "actions": self.actions.tolist(),
            "layer_weights": self.layer_weights.tolist(),
            "model_weights": self.model_weights.tolist(),
            "realized_layer_utilities": self.realized_utilities.tolist(),
            "best_response_gap": self.best_response_gap.tolist(),
            "optimization_trace": self.optimization_trace,
            "trace_interval": self.config.trace_interval,
            "iterations_run": int(self.optimization_trace[-1]["step"]),
        }


class GridSubspaceMerger:
    def __init__(
        self,
        config: AblationConfig | None = None,
        *,
        mapper: PublicTypeMapper | None = None,
        belief_parameters: BeliefParameters | None = None,
    ) -> None:
        self.config = config or AblationConfig()
        self.mapper = mapper
        self.belief_parameters = belief_parameters

    def merge(
        self,
        base_state: Mapping[str, Any],
        endpoint_states: Sequence[Mapping[str, Any]],
        private_types: Array,
        *,
        layer_map: Mapping[str, str] | Callable[[str], str] | None = None,
        model_names: Sequence[str] | None = None,
        geometry: PublicGeometry | None = None,
    ) -> MergeResult:
        if geometry is None:
            geometry = build_public_geometry(
                base_state,
                endpoint_states,
                rank=self.config.rank,
                layer_map=layer_map,
                epsilon=self.config.epsilon,
            )
        else:
            if layer_map is not None:
                raise ValueError("layer_map cannot be supplied with geometry")
            self._validate_precomputed_geometry(
                geometry, base_state, endpoint_states, self.config.rank
            )
        types = np.asarray(private_types, dtype=np.float64)
        expected_prefix = (len(endpoint_states), len(geometry.layer_names))
        if types.ndim != 3 or types.shape[:2] != expected_prefix:
            raise ValueError(
                f"private_types must have shape [M,L,3(+stability)]; expected prefix {expected_prefix}, got {types.shape}"
            )
        if types.shape[-1] not in (3, 4):
            raise ValueError("private_types needs 3 or 4 components")
        if not np.isfinite(types).all() or np.any(types < 0.0) or np.any(types > 1.0):
            raise ValueError("private_types must be finite values in [0,1]")
        if model_names is None:
            names = tuple((f"model_{index}" for index in range(len(endpoint_states))))
        else:
            names = tuple((str(name) for name in model_names))
            if len(names) != len(endpoint_states) or len(set(names)) != len(names):
                raise ValueError("model_names must be unique and match endpoints")
        mapper = self.mapper or PublicTypeMapper.default(
            geometry.public_features.shape[-1], types.shape[-1], seed=self.config.seed
        )
        mapper = mapper.validate(geometry.public_features.shape[-1], types.shape[-1])
        masked_features = _masked_public_features(
            geometry.public_features, self.config, mapper.feature_mean
        )
        public_type_mean = mapper.predict(masked_features, self.config.mapping_kind)
        parameters = (
            self.belief_parameters or BeliefParameters.default(types.shape[-1])
        ).validate(types.shape[-1], self.config.epsilon)
        belief = _build_beliefs(public_type_mean, types, parameters, self.config)
        selected = _component_indices(types.shape[-1], self.config)
        actions, trace, utilities, gaps = _solve_actions(
            geometry, belief, types, selected, self.config
        )
        weights = _softmax(actions, axis=0)
        merged = self._merge_state_dict(base_state, endpoint_states, geometry, weights)
        return MergeResult(
            state_dict=merged,
            actions=actions,
            layer_weights=weights,
            model_weights=weights.mean(axis=1),
            geometry=geometry,
            belief=belief,
            public_type_mean=public_type_mean,
            type_strength=_type_strength(types, selected),
            realized_utilities=utilities,
            best_response_gap=gaps,
            optimization_trace=trace,
            mapper=mapper,
            config=self.config,
            model_names=names,
        )

    @staticmethod
    def _validate_precomputed_geometry(
        geometry: PublicGeometry,
        base_state: Mapping[str, Any],
        endpoint_states: Sequence[Mapping[str, Any]],
        rank: int,
    ) -> None:
        model_count = len(endpoint_states)
        layer_count = len(geometry.layer_names)
        if geometry.rank != rank:
            raise ValueError(
                f"precomputed geometry rank {geometry.rank} != config rank {rank}"
            )
        if geometry.grams.shape != (layer_count, model_count, model_count):
            raise ValueError("precomputed geometry Gram shape is incompatible")
        if geometry.projected_grams.shape != geometry.grams.shape:
            raise ValueError("precomputed projected-Gram shape is incompatible")
        if geometry.global_gram.shape != (model_count, model_count):
            raise ValueError("precomputed global Gram shape is incompatible")
        if geometry.public_features.shape != (model_count, layer_count, rank + 2):
            raise ValueError("precomputed public-feature shape is incompatible")
        if geometry.global_public_features.shape != (model_count, rank + 2):
            raise ValueError("precomputed global public features are incompatible")
        if geometry.layer_base_norm_sq.shape != (layer_count,):
            raise ValueError("precomputed base-norm shape is incompatible")
        if not geometry.parameter_to_layer:
            raise ValueError("precomputed geometry has no parameter mapping")
        base_keys = set(base_state)
        for index, endpoint in enumerate(endpoint_states):
            if set(endpoint) != base_keys:
                raise ValueError(
                    f"endpoint {index} keys differ from base for cached geometry"
                )
        for key, layer in geometry.parameter_to_layer.items():
            if key not in base_state:
                raise ValueError(
                    f"precomputed geometry references missing parameter {key!r}"
                )
            if not 0 <= int(layer) < layer_count:
                raise ValueError(f"precomputed geometry has invalid layer for {key!r}")
            base_shape = _shape(base_state[key])
            for index, endpoint in enumerate(endpoint_states):
                if _shape(endpoint[key]) != base_shape:
                    raise ValueError(
                        f"endpoint {index} shape differs for cached parameter {key!r}"
                    )

    @staticmethod
    def _merge_state_dict(
        base_state: Mapping[str, Any],
        endpoint_states: Sequence[Mapping[str, Any]],
        geometry: PublicGeometry,
        weights: Array,
    ) -> Mapping[str, Any]:
        merged: OrderedDict[str, Any] = OrderedDict()
        for key, base_value in base_state.items():
            if key not in geometry.parameter_to_layer:
                merged[key] = _clone(base_value)
                continue
            layer = geometry.parameter_to_layer[key]
            if hasattr(base_value, "detach") and hasattr(base_value, "clone"):
                base_work = base_value.detach()
                value = base_work.clone()
                for model, endpoint in enumerate(endpoint_states):
                    value = value + float(weights[model, layer]) * (
                        endpoint[key].detach() - base_work
                    )
                merged[key] = value
            else:
                base_array = np.asarray(base_value)
                work_dtype = (
                    np.complex128 if base_array.dtype.kind == "c" else np.float64
                )
                work = base_array.astype(work_dtype, copy=True)
                for model, endpoint in enumerate(endpoint_states):
                    work += weights[model, layer] * (
                        np.asarray(endpoint[key], dtype=work_dtype)
                        - np.asarray(base_value, dtype=work_dtype)
                    )
                merged[key] = work.astype(base_array.dtype, copy=False)
        return merged
