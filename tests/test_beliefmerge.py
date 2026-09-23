import ast
import copy
import io
import tokenize
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from beliefmerge import (
    BeliefMerge,
    DiagnosticConfig,
    InformationConfig,
    JointBelief,
    PriorParameters,
    PublicPredictor,
    ReducedUtility,
    StrategyConfig,
    assemble,
    calibrate_prior,
    fit_strategies,
    measure_private_information,
    public_information,
    scores_from_statistics,
    verify_contribution_balance,
)
from beliefmerge.__main__ import atomic_checkpoint, atomic_json, main, safe_load
from beliefmerge.experiments import (
    ablations,
    exact_match,
    experiment_plan,
    spearman,
    summarize_scores,
    summarize_seeds,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def problem():
    generator = torch.Generator().manual_seed(21)
    base = {
        "first.weight": torch.randn(3, 4, generator=generator, dtype=torch.float64),
        "second.weight": torch.randn(2, 3, generator=generator, dtype=torch.float64),
        "counter": torch.tensor(2),
    }
    tasks = [
        {
            key: value
            + 0.2 * torch.randn(value.shape, generator=generator, dtype=value.dtype)
            if value.is_floating_point()
            else value.clone()
            for key, value in base.items()
        }
        for _ in range(3)
    ]
    information = public_information(base, tasks, rank=2)
    private = torch.rand(3, 3, 2, generator=generator, dtype=torch.float64) * 0.8 + 0.1
    prior = PriorParameters(
        torch.zeros(6, dtype=torch.float64),
        torch.eye(6, dtype=torch.float64) * 0.5,
        torch.eye(6, dtype=torch.float64) * 0.2,
    )
    return base, tasks, information, private, prior


def test_global_geometry_matches_explicit_svd(problem):
    base, tasks, info, _, _ = problem
    updates = torch.stack(
        [
            torch.cat([(task[key] - base[key]).flatten() for key in info.layer_map])
            for task in tasks
        ],
        dim=1,
    )
    basis = updates @ info.basis_coefficients
    assert torch.allclose(basis.T @ basis, torch.eye(2, dtype=torch.float64), atol=1e-9)
    normalized = updates / (updates.norm(dim=0) + 1e-8)
    u, _, _ = torch.linalg.svd(normalized, full_matrices=False)
    assert torch.allclose(basis @ basis.T, u[:, :2] @ u[:, :2].T, atol=1e-9)
    assert torch.allclose(info.observations[:, 1:], (basis.T @ normalized).T, atol=1e-9)
    assert torch.allclose(info.gram.sum(0), updates.T @ updates)
    assert torch.allclose(info.projections.sum(0), updates.T @ basis)


@pytest.mark.parametrize("preservation", ["private_product", "equation14"])
def test_reduced_utility_matches_full_parameter_computation(problem, preservation):
    base, tasks, info, private, _ = problem
    cfg = StrategyConfig(preservation=preservation, action_space="legacy_layerwise")
    actions = torch.tensor(
        [[0.1, 2.0], [0.8, 1.0], [1.2, 0.4]], dtype=torch.float64, requires_grad=True
    )
    weights = actions.softmax(0)
    updates = [
        torch.stack([(task[key] - base[key]).flatten() for task in tasks])
        for key in info.layer_map
    ]
    merged = [
        (weights[:, layer, None] * delta).sum(0) for layer, delta in enumerate(updates)
    ]
    full_update = torch.cat(merged)
    basis = torch.cat(updates, dim=1).T @ info.basis_coefficients
    support = (basis.T @ full_update).square().sum() / (
        full_update.square().sum() + cfg.epsilon_public
    )
    expected = []
    for model in range(3):
        distances = torch.stack(
            [
                (merged[layer] - delta[model]).square().sum()
                for layer, delta in enumerate(updates)
            ]
        )
        norms = torch.stack([delta[model].square().sum() for delta in updates])
        if preservation == "private_product":
            local = (
                private[model].prod(0) * distances / (norms + cfg.epsilon_local)
            ).sum()
        else:
            local = distances.sum() / (norms.sum() + cfg.epsilon_local)
        expected.append(
            -cfg.lambda_t * local
            + cfg.lambda_o * support
            - cfg.lambda_b * actions[model].square().sum()
        )
    actual = ReducedUtility(info, cfg)(actions, private.flatten(1))
    expected = torch.stack(expected)
    assert torch.allclose(actual, expected, atol=1e-10)
    (actual_grad,) = torch.autograd.grad(actual.sum(), actions, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected.sum(), actions)
    assert torch.allclose(actual_grad, expected_grad, atol=1e-9)


def test_assembly_preserves_base_buffers_and_dtype(problem):
    base, tasks, info, _, _ = problem
    weights = torch.full((3, 2), 1 / 3, dtype=torch.float64)
    merged = assemble(base, tasks, weights, info)
    for key in info.layer_map:
        assert torch.allclose(
            merged[key], torch.stack([task[key] for task in tasks]).mean(0)
        )
        assert merged[key].dtype == base[key].dtype
    assert torch.equal(merged["counter"], base["counter"])
    assert merged["counter"].data_ptr() != base["counter"].data_ptr()


def test_global_scalar_utility_and_gradients_match_equation31(problem):
    base, tasks, info, private, _ = problem
    config = StrategyConfig()
    actions = torch.tensor(
        [[-1.3], [0.4], [2.1]], dtype=torch.float64, requires_grad=True
    )
    updates = torch.stack(
        [
            torch.cat([(task[key] - base[key]).flatten() for key in info.layer_map])
            for task in tasks
        ]
    )
    basis = updates.T @ info.basis_coefficients
    merged = actions.squeeze(-1).softmax(0) @ updates
    local = (merged - updates).square().sum(-1) / (
        updates.square().sum(-1) + config.epsilon_local
    )
    shared = (basis.T @ merged).square().sum() / (
        merged.square().sum() + config.epsilon_public
    )
    expected = (
        -config.lambda_t * local
        + config.lambda_o * shared
        - config.lambda_b * actions.squeeze(-1).square()
    )
    utility = ReducedUtility(info, config)
    actual = utility(actions, private.flatten(1))
    assert torch.allclose(actual, expected, atol=1e-10)
    assert torch.allclose(
        torch.autograd.grad(actual.sum(), actions, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), actions)[0],
        atol=1e-10,
    )
    assert torch.equal(actual, utility(actions, private.flatten(1) * 2 - 1))


def test_cached_scalar_response_matches_full_profiles_and_derivatives(problem):
    _, _, info, private, _ = problem
    utility = ReducedUtility(info, StrategyConfig())
    actions = torch.randn(7, 3, 1, dtype=torch.float64)
    for model in range(3):
        cached = utility.scalar_response(actions, model)
        candidate = torch.tensor([-2.7], dtype=torch.float64, requires_grad=True)
        profile = actions.clone()
        profile[:, model] = candidate
        full = utility(profile, private.flatten(1).expand(7, -1, -1))[:, model]
        reduced = cached(candidate)
        assert torch.allclose(full, reduced, atol=1e-10)
        assert torch.allclose(
            torch.autograd.grad(full.sum(), candidate, retain_graph=True)[0],
            torch.autograd.grad(reduced.sum(), candidate)[0],
            atol=1e-10,
        )


def test_corrected_response_bound_handles_large_task_norm_disparity():
    base = {"w": torch.zeros(1, dtype=torch.float64)}
    tasks = [
        {"w": torch.tensor([value], dtype=torch.float64)} for value in (0.01, 10.0)
    ]
    info = public_information(base, tasks, rank=1)
    cfg = StrategyConfig()
    utility = ReducedUtility(info, cfg)
    scores = torch.zeros(2, 3, dtype=torch.float64)
    zero = utility(torch.zeros(2, 1, dtype=torch.float64), scores)
    assert float(zero[0]) < -cfg.lambda_o
    limits = torch.tensor(
        utility.regularity_bounds()["response_action_bounds"], dtype=torch.float64
    )
    assert limits[0] > (2 * cfg.lambda_o / cfg.lambda_b) ** 0.5
    for model in range(2):
        candidates = torch.zeros(2, 2, 1, dtype=torch.float64)
        candidates[:, model, 0] = torch.tensor([-1.01, 1.01]) * limits[model]
        assert (
            utility(candidates, scores.expand(2, -1, -1))[:, model] < zero[model]
        ).all()


def test_raw_gaussian_samples_are_not_clipped_or_sigmoid_transformed():
    prior = PriorParameters(torch.zeros(1), torch.eye(1), torch.eye(1))
    belief = JointBelief(torch.zeros(3, 1), prior)
    samples = belief.common_samples(4000, torch.Generator().manual_seed(3))
    assert (samples < 0).any() and (samples > 1).any()
    assert abs(float(samples.mean())) < 0.1
    assert abs(float(samples.var()) - 2) < 0.2


def test_fixed_prior_calibration_uses_gaussian_nll_and_freezes_predictor():
    predictor = PublicPredictor(1, 1, architecture="linear").double()
    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.zero_()
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    y = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    prior = PriorParameters(
        torch.tensor([0.1], dtype=torch.float64),
        torch.tensor([[0.3]], dtype=torch.float64),
        torch.tensor([[0.2]], dtype=torch.float64),
    )
    expected = -torch.distributions.Normal(0.1, 0.5**0.5).log_prob(y).mean()
    history = predictor.fit(x, y, split="prior_calibration", prior=prior, steps=1)
    assert history[0] == pytest.approx(float(expected), abs=1e-7)
    assert not predictor.training
    assert all(not parameter.requires_grad for parameter in predictor.parameters())


def test_current_method_rejects_legacy_ablation_labels_and_predictors(problem):
    base, tasks, _, private, prior = problem
    for variant in ("without_f", "without_bpi"):
        with pytest.raises(ValueError, match="Legacy"):
            JointBelief(torch.zeros(3, 6), prior, variant=variant)
    merger = BeliefMerge(
        InformationConfig(predictor_architecture="linear"),
        StrategyConfig(steps=1, samples=2),
    )
    with pytest.raises(ValueError, match="linear predictor"):
        merger.merge(base, tasks, private, PublicPredictor(3, 6), prior)


def test_joint_optimization_updates_all_models_and_logs_warmup(problem):
    _, _, info, _, prior = problem
    config = StrategyConfig(
        steps=4, samples=4, hidden=8, warmup_ratio=0.5, mode="without_nao"
    )
    belief = JointBelief(torch.zeros(3, 6, dtype=torch.float64), prior)
    fit = fit_strategies(
        belief, info.observations, ReducedUtility(info, config), config
    )
    assert all(row["updated_models"] == [0, 1, 2] for row in fit.trace)
    assert fit.trace[0]["learning_rate"] == config.learning_rate / 2
    assert fit.trace[-1]["learning_rate"] == config.learning_rate


def test_current_plan_uses_five_seeds_linear_ablation_and_three_weight_sweeps():
    plan = experiment_plan()
    assert {item.seed for item in plan} == {0, 1, 2, 3, 4}
    assert ablations()["LP"][0].predictor_architecture == "linear"
    assert "without_bpi" not in ablations()
    for coefficient in ("lambda_t", "lambda_o", "lambda_b"):
        assert len([item for item in plan if item.name.startswith(coefficient)]) == 50


def test_representation_evidence_uses_endpoint_and_excludes_ablated_passes():
    base = nn.Sequential(
        nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1, bias=False)
    ).double()
    with torch.no_grad():
        base[0].weight.copy_(torch.eye(2))
        base[2].weight.fill_(0.2)
    task = copy.deepcopy(base)
    with torch.no_grad():
        task[0].weight.mul_(3)
        task[2].weight.add_(0.4)
    example = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    hidden = task[:2](example).detach()
    expected = (hidden @ (task[2].weight - base[2].weight).T).square().sum()
    result = measure_private_information(
        base,
        task,
        [example],
        lambda model, x: model(x),
        lambda prediction, x: prediction.square().mean(),
        ["0", "2"],
    )
    assert result.representation_energy[1] == pytest.approx(float(expected.detach()))
    assert all(not module._forward_pre_hooks for module in task.modules())


def test_recovery_rmse_is_diagnostic_and_excludes_self_observations():
    prior = PriorParameters(torch.zeros(1), torch.eye(1) * 0.8, torch.eye(1) * 0.2)
    belief = JointBelief(torch.zeros(3, 1), prior)
    observed = torch.tensor([[0.2], [0.5], [0.8]])
    result = belief.assess_private_recovery(observed)
    expected = [
        (0.8 * observed[i] - observed[j]).square()
        for i in range(3)
        for j in range(3)
        if i != j
    ]
    assert result["rmse"] == pytest.approx(float(torch.stack(expected).mean().sqrt()))
    assert len(result["pairs"]) == 6
    assert result["used_for_strategy_fitting"] is False
    assert belief.predictions.count_nonzero() == 0


def test_fixed_prior_calibration_cli_marks_coordinates_and_preserves_prior(tmp_path):
    prior = PriorParameters(torch.zeros(2), torch.eye(2) * 0.1, torch.eye(2) * 0.2)
    artifact = {
        "split": "prior_calibration",
        "predictor_model_ids": ["historical"],
        "predictor_public": torch.randn(4, 1),
        "predictor_private": torch.rand(4, 2),
        "fixed_prior": prior.state_dict(),
    }
    torch.save(artifact, tmp_path / "input.pt")
    main(
        [
            "calibrate",
            "--input",
            str(tmp_path / "input.pt"),
            "--output",
            str(tmp_path / "prior.pt"),
            "--steps",
            "2",
            "--architecture",
            "linear",
        ]
    )
    output = safe_load(str(tmp_path / "prior.pt"))
    assert output["coordinates"] == "raw"
    assert output["calibration_method"] == "fixed_prior_marginal_gaussian_nll"
    assert output["predictor_config"]["architecture"] == "linear"
    assert all(
        torch.equal(output["prior"][key], value)
        for key, value in prior.state_dict().items()
    )


def test_rank_deficiency_and_zero_updates(problem):
    base, tasks, _, _, _ = problem
    identical = public_information(base, [tasks[0], tasks[0]], rank=4)
    assert identical.rank == 1
    zero = public_information(base, [base, base], rank=4)
    assert zero.rank == 0
    utility = ReducedUtility(zero, StrategyConfig())
    assert torch.isfinite(
        utility(torch.zeros(2, 1, dtype=torch.float64), torch.full((2, 6), 0.5))
    ).all()


def test_gaussian_posterior_and_joint_covariance():
    prior = PriorParameters(
        torch.zeros(1, dtype=torch.float64),
        torch.tensor([[0.8]], dtype=torch.float64),
        torch.tensor([[0.2]], dtype=torch.float64),
    )
    belief = JointBelief(torch.zeros(3, 1, dtype=torch.float64), prior)
    own = torch.tensor([0.8], dtype=torch.float64)
    mean, covariance = belief.posterior(0, own)
    assert torch.allclose(mean, 0.8 * own)
    assert torch.allclose(covariance, torch.tensor([[0.16]], dtype=torch.float64))
    samples = belief.conditional_samples(
        0, own, 30000, torch.Generator().manual_seed(10)
    )
    observed = torch.cov(samples[:, 1:, 0].T)
    assert abs(float(observed[0, 1]) - 0.16) < 0.015
    assert abs(float(observed[0, 0]) - 0.36) < 0.02
    assert torch.equal(samples[:, 0], own.expand(30000, 1))


def test_independent_belief_preserves_marginals_not_covariance():
    prior = PriorParameters(torch.zeros(1), torch.eye(1) * 0.8, torch.eye(1) * 0.2)
    belief = JointBelief(torch.zeros(3, 1), prior, variant="without_bef")
    samples = belief.conditional_samples(
        0, torch.tensor([0.8]), 30000, torch.Generator().manual_seed(10)
    )
    covariance = torch.cov(samples[:, 1:, 0].T)
    assert abs(float(covariance[0, 1])) < 0.015
    assert abs(float(covariance[0, 0]) - 0.36) < 0.02


@pytest.mark.parametrize("family", ["gaussian", "laplace", "student_t"])
def test_prior_families_and_conditional_sampling(family):
    prior = PriorParameters(torch.zeros(2), torch.eye(2) * 0.4, torch.eye(2) * 0.2)
    belief = JointBelief(torch.zeros(3, 2), prior, family=family)
    samples = belief.common_samples(100, torch.Generator().manual_seed(10))
    assert samples.shape == (100, 3, 2)
    own = torch.tensor([0.2, 0.8])
    conditional = belief.conditional_samples(
        0, own, 100, torch.Generator().manual_seed(11)
    )
    assert torch.isfinite(conditional).all()
    assert torch.equal(conditional[:, 0], own.expand(100, 2))
    if family != "gaussian":
        assert belief.last_importance_ess > 0


@pytest.mark.parametrize(
    "variant", ["without_f", "without_bau", "without_bef", "without_eta", "without_bpi"]
)
def test_belief_ablations(problem, variant):
    _, _, _, private, prior = problem
    belief = JointBelief(
        torch.ones(3, 6, dtype=torch.float64),
        prior,
        variant=variant,
        coordinates="logit",
    )
    samples = belief.fitting_samples(0, 16, torch.Generator().manual_seed(4))
    assert samples.shape == (16, 3, 6)
    if variant == "without_f":
        assert belief.predictions.count_nonzero() == 0
    if variant == "without_bpi":
        assert torch.equal(samples[0, 1], samples[-1, 1])


def test_diagnostic_formulas():
    config = DiagnosticConfig(sensitivity_scale=1.0)
    scores = scores_from_statistics(
        torch.tensor([1.0, 3.0]),
        torch.tensor([2.0, 3.0]),
        torch.tensor([2.0, 3.0]),
        torch.tensor([2.0, 2.0]),
        torch.tensor(1.0),
        torch.tensor([2.0, 0.5]),
        config,
    )
    assert torch.allclose(scores[0], torch.tensor([0.5, 0.75]))
    assert torch.allclose(scores[1], torch.tensor([0.5, 0.5]))
    assert scores[2, 0] > 0.5 and scores[2, 1] < 0.5


def test_private_measurements_restore_model_and_reject_test_split():
    torch.manual_seed(4)
    base, task = nn.Sequential(nn.Linear(2, 2)), nn.Sequential(nn.Linear(2, 2))
    examples = [
        (torch.tensor([[1.0, 2.0]]), torch.tensor([0])),
        (torch.tensor([[2.0, 1.0]]), torch.tensor([1])),
    ]
    forward = lambda model, example: model(example[0])
    loss = lambda output, example: nn.functional.cross_entropy(output, example[1])
    original = {key: value.clone() for key, value in task.state_dict().items()}
    task.train()
    result = measure_private_information(base, task, examples, forward, loss, ["0"])
    assert result.scores.shape == (3, 1)
    assert result.with_stability().shape == (4, 1)
    assert task.training
    assert all(
        torch.equal(task.state_dict()[key], value) for key, value in original.items()
    )
    assert all(parameter.grad is None for parameter in task.parameters())
    gradients = [
        torch.autograd.grad(loss(forward(task, example), example), task[0].weight)[0]
        for example in examples
    ]
    delta = task[0].weight.detach() - base[0].weight.detach()
    expected = (
        torch.stack(gradients).double().square().mean(0) * delta.double().square()
    ).sum()
    assert torch.allclose(result.sensitivity_energy[0], expected)
    with pytest.raises(ValueError, match="evaluation/test"):
        measure_private_information(
            base, task, examples, forward, loss, ["0"], split="test"
        )


def test_private_failure_restores_weights():
    base, task = nn.Sequential(nn.Linear(2, 2)), nn.Sequential(nn.Linear(2, 2))
    original = task[0].weight.detach().clone()
    counter = 0

    def fail_on_ablation(output, example):
        nonlocal counter
        counter += 1
        if counter > 1:
            raise RuntimeError("intentional failure")
        return output.square().mean()

    with pytest.raises(RuntimeError, match="intentional"):
        measure_private_information(
            base, task, [torch.ones(1, 2)], lambda m, x: m(x), fail_on_ablation, ["0"]
        )
    assert torch.equal(original, task[0].weight)
    assert not base[0]._forward_pre_hooks


def test_predictor_and_prior_calibration():
    torch.manual_seed(7)
    x = torch.randn(8, 3, 2)
    y = torch.sigmoid(torch.cat([x, x], -1))
    predictor = PublicPredictor(2, 4, hidden=8)
    history = predictor.fit(
        x, y, split="prior_calibration", steps=20, learning_rate=0.02
    )
    assert history[-1] < history[0]
    assert not any(parameter.requires_grad for parameter in predictor.parameters())
    prior = calibrate_prior(predictor(x), y, split="prior_calibration")
    assert prior.mean.shape == (4,)
    assert torch.linalg.eigvalsh(prior.noise_covariance).min() > 0
    with pytest.raises(ValueError):
        calibrate_prior(predictor(x), y, split="test")
    with pytest.raises(ValueError):
        predictor.fit(x, y, split="test")


def test_strategy_training_reproducibility_checkpoints_and_cyclic_updates(problem):
    _, _, info, _, prior = problem
    cfg = StrategyConfig(steps=51, samples=8, hidden=8, seed=12)
    belief = JointBelief(torch.zeros(3, 6, dtype=torch.float64), prior)
    saved = []
    first = fit_strategies(
        belief,
        info.observations,
        ReducedUtility(info, cfg),
        cfg,
        lambda step, payload: saved.append((step, payload)),
    )
    second = fit_strategies(belief, info.observations, ReducedUtility(info, cfg), cfg)
    assert first.trace == second.trace
    assert [step for step, _ in saved] == [50, 51]
    assert [row["model"] for row in first.trace[:6]] == [0, 1, 2, 0, 1, 2]
    assert len(saved[0][1]["trace"]) == 50
    assert all(
        torch.equal(value, second.networks.state_dict()[key])
        for key, value in first.networks.state_dict().items()
    )


def test_end_to_end_merge_has_no_private_type_leakage_in_fitting(problem):
    base, tasks, _, private, prior = problem
    merger = BeliefMerge(strategy=StrategyConfig(steps=3, samples=4, hidden=8))
    predictor = lambda public: torch.zeros(len(tasks), 6, device=public.device)
    first = merger.merge(base, tasks, private, predictor, prior)
    changed = private.clone()
    changed[0] = 0.15
    second = merger.merge(base, tasks, changed, predictor, prior)
    assert first.strategies.trace == second.strategies.trace
    assert torch.equal(first.actions[1:], second.actions[1:])
    assert not torch.equal(first.actions[0], second.actions[0])
    assert first.weights.shape == (3, 1)
    assert torch.allclose(first.weights.sum(0), torch.ones(1))
    assert first.metadata()["benchmark_results_verified"] is False


def test_verification_does_not_claim_equilibrium(problem):
    base, tasks, _, private, prior = problem
    merger = BeliefMerge(strategy=StrategyConfig(steps=2, samples=4, hidden=8))
    result = merger.merge(base, tasks, private, lambda x: torch.zeros(3, 6), prior)
    report = verify_contribution_balance(
        result.strategies,
        result.belief,
        result.utility,
        result.private_scores,
        samples=8,
        search_steps=2,
        restarts=1,
    )
    assert not report["uniform_equilibrium_certified"]
    assert all(row["sampled_gain"] >= 0 for row in report["models"])
    assert all(row["certified_upper_bound"] is None for row in report["models"])
    with pytest.raises(ValueError, match="independent"):
        verify_contribution_balance(
            result.strategies,
            result.belief,
            result.utility,
            result.private_scores,
            seed=0,
        )


def test_metric_aggregation_uses_mean_of_ratios_and_one_median_seed():
    report = summarize_scores({"a": 0.4, "b": 0.8}, {"a": 0.5, "b": 1.0}, ["a", "b"])
    assert report["average"] == pytest.approx(60)
    assert report["normalized_average"] == pytest.approx(80)
    reports = {
        i: summarize_scores({"a": value}, {"a": 1.0}, ["a"])
        for i, value in enumerate([0.7, 0.9, 0.8])
    }
    summary = summarize_seeds(reports)
    assert summary["median_seed"] == 2
    assert summary["std"] == pytest.approx(10)
    with pytest.raises(ValueError):
        summarize_scores({"a": 0.5}, {"a": 0.8}, ["a", "b"])
    assert summarize_scores({"stsb": -0.5}, {"stsb": 0.8}, ["stsb"])["average"] == -50


def test_metrics():
    assert exact_match(["yes", "no"], ["yes", "yes"]) == 0.5
    assert spearman([1.0, 2.0, 2.0, 3.0], [4.0, 3.0, 3.0, 2.0]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        spearman([1.0, 1.0], [2.0, 3.0])


def test_experiment_plan_matches_current_paper_without_results():
    plan = experiment_plan(seeds=[7])
    assert len([item for item in plan if item.name == "main"]) == 10
    assert len([item for item in plan if item.name == "training_sensitivity"]) == 11
    assert all(len(item.tasks) == 20 for item in plan if item.name in ablations())
    assert all(item.strategy.save_every == 50 for item in plan)
    assert all(item.strategy.seed == 7 for item in plan)
    assert all(
        item.strategy.lambda_t + item.strategy.lambda_o + item.strategy.lambda_b
        == pytest.approx(1)
        for item in plan
    )


def test_atomic_io_and_cli_plan(tmp_path):
    destination = tmp_path / "tensor.pt"
    atomic_checkpoint(destination, {"x": torch.ones(2)})
    assert torch.equal(safe_load(str(destination))["x"], torch.ones(2))
    atomic_json(tmp_path / "metrics.json", {"score": 0.8})
    main(["plan", "--output", str(tmp_path / "plan.json"), "--seeds", "0"])
    assert (tmp_path / "plan.json").exists()
    with pytest.raises(FileExistsError):
        main(["plan", "--output", str(tmp_path / "plan.json")])


def test_source_has_no_comments_docstrings_or_non_ascii():
    root = Path(__file__).resolve().parents[1]
    for directory in [root / "beliefmerge", root / "tests"]:
        for path in directory.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert source.isascii(), path.name
            assert not any(
                token.type == tokenize.COMMENT
                for token in tokenize.generate_tokens(io.StringIO(source).readline)
            ), path.name
            for node in ast.walk(ast.parse(source)):
                if isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    assert ast.get_docstring(node) is None, path.name


def test_cli_merge_writes_only_measured_outputs(problem, tmp_path):
    base, tasks, info, private, prior = problem
    torch.save(base, tmp_path / "base.pt")
    task_paths = []
    for index, task in enumerate(tasks):
        path = tmp_path / f"task_{index}.pt"
        torch.save(task, path)
        task_paths.append(str(path))
    predictor = PublicPredictor(3, 6, hidden=8)
    torch.save(
        {
            "split": "type_cal",
            "scores": private,
            "layer_map": info.layer_map,
            "model_ids": ["a", "b", "c"],
        },
        tmp_path / "private.pt",
    )
    torch.save(
        {
            "split": "prior_calibration",
            "model_ids": ["cal_a", "cal_b"],
            "coordinates": "raw",
            "predictor_config": {"input_dim": 3, "type_dim": 6, "hidden": 8},
            "predictor": predictor.state_dict(),
            "prior": prior.state_dict(),
        },
        tmp_path / "belief.pt",
    )
    output = tmp_path / "run"
    main(
        [
            "merge",
            "--base",
            str(tmp_path / "base.pt"),
            "--tasks",
            *task_paths,
            "--private",
            str(tmp_path / "private.pt"),
            "--belief",
            str(tmp_path / "belief.pt"),
            "--output",
            str(output),
            "--steps",
            "2",
            "--samples",
            "4",
            "--rank",
            "2",
        ]
    )
    assert (output / "metrics.json").exists()
    assert (output / "trace.json").exists()
    assert (output / "strategy_step_000002.pt").exists()
    merged = safe_load(str(output / "merged.pt"))
    assert set(merged) == set(base)
    assert all(torch.isfinite(value).all() for value in merged.values())


def test_public_input_ablations_leave_geometry_unchanged(problem):
    _, _, info, _, _ = problem
    original = info.projections.clone()
    assert info.public_inputs(magnitude=False).shape == (3, 2)
    assert info.public_inputs(direction=False).shape == (3, 1)
    assert info.public_inputs(compatibility=True).shape == (3, 5)
    assert torch.equal(original, info.projections)


def test_non_gaussian_latent_priors_match_covariance():
    covariance = torch.tensor([[0.5, 0.1], [0.1, 0.3]], dtype=torch.float64)
    prior = PriorParameters(
        torch.zeros(2, dtype=torch.float64),
        covariance,
        torch.eye(2, dtype=torch.float64) * 0.2,
    )
    for family in ("gaussian", "laplace", "student_t"):
        belief = JointBelief(
            torch.zeros(3, 2, dtype=torch.float64), prior, family=family
        )
        samples = belief._latent((100000,), torch.Generator().manual_seed(17))
        assert torch.allclose(torch.cov(samples.T), covariance, atol=0.055)


@pytest.mark.parametrize(
    "method",
    [
        "weight_average",
        "task_arithmetic",
        "ties",
        "dare_ties",
        "consensus",
        "tsv_m",
        "iso_c",
        "iso_cts",
    ],
)
def test_integrated_baselines_are_finite_and_preserve_buffers(problem, method):
    from beliefmerge.baselines import BaselineConfig, merge_baseline

    base, tasks, _, _, _ = problem
    result = merge_baseline(base, tasks, BaselineConfig(method=method, seed=3))
    assert set(result) == set(base)
    assert torch.equal(result["counter"], base["counter"])
    assert all(torch.isfinite(value).all() for value in result.values())
    assert all(result[key].dtype == base[key].dtype for key in base)


def test_exact_quantile_matches_torch_interpolation():
    from beliefmerge.baselines import exact_quantile

    values = torch.tensor([5.0, 1.0, 2.0, 2.0, 9.0, 10.0])
    for quantile in (0.0, 0.1, 0.2, 0.5, 0.9, 1.0):
        assert torch.allclose(
            exact_quantile(values, quantile), torch.quantile(values, quantile)
        )


def test_slurm_submission_is_idempotent_and_uses_dependencies(tmp_path):
    from beliefmerge.workflow import ExperimentLedger, Job, Resources, Slurm

    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=str(100 + len(calls)))

    ledger = ExperimentLedger(tmp_path / "runs")
    resources = Resources("gpu", "h200")
    first = Job("train", ("python", "train.py"), resources)
    second = Job("merge", ("python", "merge.py"), resources, dependencies=("train",))
    ledger.register([second, first])
    scheduler = Slurm(ledger, tmp_path, runner=runner)
    result = scheduler.submit_all()
    assert all(item["submitted"] for item in result)
    assert calls[0][0] == "sbatch"
    assert "afterok:101" in calls[1]
    assert not any(item["submitted"] for item in scheduler.submit_all())
    assert len(calls) == 2
    scheduler.cancel(["merge"])
    assert calls[-1] == ["scancel", "102"]
    assert ledger.snapshot()["jobs"]["train"]["state"] == "SUBMITTED"


def test_slurm_dry_run_does_not_execute_commands(tmp_path):
    from beliefmerge.workflow import ExperimentLedger, Job, Resources, Slurm

    ledger = ExperimentLedger(tmp_path / "runs")
    ledger.register(
        [
            Job(
                "test",
                ("python", "file name.py", "a; echo injected"),
                Resources("gpu", "h100"),
            )
        ]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("External command executed")

    scheduler = Slurm(ledger, tmp_path, runner=forbidden)
    result = scheduler.submit("test", dry_run=True)
    assert "'a; echo injected'" in result["script"]
    assert ledger.snapshot()["jobs"]["test"]["state"] == "UNSUBMITTED"
    assert not (tmp_path / "runs/scripts").exists()


def test_ledger_rejects_cycles_and_mutated_requests(tmp_path):
    from beliefmerge.workflow import ExperimentLedger, Job, Resources

    ledger = ExperimentLedger(tmp_path)
    resources = Resources("gpu", "h200")
    with pytest.raises(ValueError, match="dependency"):
        ledger.register(
            [
                Job("a", ("python",), resources, ("b",)),
                Job("b", ("python",), resources, ("a",)),
            ]
        )
    ledger.register([Job("a", ("python",), resources)])
    with pytest.raises(ValueError, match="immutable"):
        ledger.register([Job("a", ("python", "changed"), resources)])


def test_numeric_export_excludes_checkpoints(tmp_path):
    from beliefmerge.workflow import contained_path, export_numeric_results

    source = tmp_path / "source"
    source.mkdir()
    (source / "metrics.json").write_text('{"accuracy": 0.5}')
    (source / "model.pt").write_bytes(b"not a result")
    (source / "checkpoint-50").mkdir()
    (source / "checkpoint-50/config.json").write_text("{}")
    result = export_numeric_results(source, tmp_path / "export")
    assert [item["file"] for item in result["files"]] == ["metrics.json"]
    assert not result["checkpoints_included"]
    with pytest.raises(ValueError):
        contained_path(source, "../escape")


def test_vision_training_resume_and_step50_checkpoint(tmp_path):
    from beliefmerge.vision import VisionTrainingConfig, train_classifier

    torch.manual_seed(8)
    encoder = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2))
    initial = copy.deepcopy(encoder)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3), torch.tensor([0, 1] * 4)
    )
    config = VisionTrainingConfig(
        max_steps=51, batch_size=4, learning_rate=0.01, seed=7
    )
    completed = train_classifier(
        encoder,
        nn.Identity(),
        dataset,
        dataset,
        tmp_path / "full",
        config,
        dataset_id="fixture",
    )
    assert completed["state"] == "COMPLETED"
    assert (tmp_path / "full/checkpoint_step_000050.pt").exists()
    assert (tmp_path / "full/metrics_step_000050.json").exists()
    interrupted = train_classifier(
        initial,
        nn.Identity(),
        dataset,
        dataset,
        tmp_path / "resumed",
        config,
        dataset_id="fixture",
        stop_after_step=25,
    )
    assert interrupted["state"] == "PAUSED"
    train_classifier(
        initial,
        nn.Identity(),
        dataset,
        dataset,
        tmp_path / "resumed",
        config,
        dataset_id="fixture",
    )
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, initial.state_dict()[key])


def test_vision_training_splits_are_disjoint():
    from beliefmerge.vision import split_training_dataset

    train, calibration, selection = split_training_dataset(list(range(100)), seed=4)
    assert not set(train.indices) & set(calibration.indices)
    assert not set(train.indices) & set(selection.indices)
    assert not set(calibration.indices) & set(selection.indices)
    assert len(train) + len(calibration) + len(selection) == 100


def test_calibration_command_produces_merge_ready_artifact(tmp_path):
    torch.manual_seed(5)
    data = {
        "split": "prior_calibration",
        "predictor_model_ids": ["train_a"],
        "prior_model_ids": ["prior_a"],
        "predictor_public": torch.randn(8, 2),
        "predictor_private": torch.rand(8, 6) * 0.8 + 0.1,
        "prior_public": torch.randn(4, 3, 2),
        "prior_private": torch.rand(4, 3, 6) * 0.8 + 0.1,
    }
    torch.save(data, tmp_path / "calibration.pt")
    main(
        [
            "calibrate",
            "--input",
            str(tmp_path / "calibration.pt"),
            "--output",
            str(tmp_path / "belief.pt"),
            "--steps",
            "2",
            "--hidden",
            "8",
        ]
    )
    artifact = safe_load(str(tmp_path / "belief.pt"))
    assert artifact["split"] == "prior_calibration"
    assert artifact["prior"]["latent_covariance"].shape == (6, 6)


@pytest.mark.parametrize(
    "command",
    [
        "vision-train",
        "language-train",
        "language-private",
        "language-metrics",
        "workflow",
        "aggregate",
        "calibrate",
    ],
)
def test_integrated_command_help(command):
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--help"])
    assert exit_info.value.code == 0


def test_training_curve_analysis_uses_observed_values():
    from beliefmerge.experiments import compare_trajectories, trajectory_summary

    a = [
        {"step": step, "average": score}
        for step, score in [(50, 70.0), (100, 80.0), (200, 81.0)]
    ]
    b = [
        {"step": step, "average": score}
        for step, score in [(50, 60.0), (100, 75.0), (200, 80.0)]
    ]
    comparison = compare_trajectories(a, b)
    assert comparison["step_speedup"] == 2
    assert comparison["candidate_first_step"] == 100
    assert trajectory_summary(a)["peak_step"] == 200
    with pytest.raises(ValueError):
        compare_trajectories(a, b[:-1])


def test_bootstrap_and_private_associations_are_descriptive():
    from beliefmerge.experiments import paired_bootstrap, private_component_associations

    result = paired_bootstrap([0.4, 0.5, 0.6], [0.3, 0.4, 0.5], repetitions=200)
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["lower"] == pytest.approx(0.1)
    scores = np.asarray([[[0.1, 0.2]], [[0.4, 0.5]], [[0.7, 0.8]]])
    report = private_component_associations(
        scores, np.asarray([1.0, 2.0, 3.0]), ["a", "b"], ["sensitivity"]
    )
    assert report["components"]["sensitivity"]["per_layer_spearman"][
        "a"
    ] == pytest.approx(1.0)


def test_epoch_seeded_augmentation_replays_without_changing_rng():
    from beliefmerge.vision import EpochSeededDataset

    class Augmentation(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.rand(4) + index

    dataset = EpochSeededDataset(Augmentation(), 8)
    state = torch.get_rng_state().clone()
    assert torch.equal(dataset[1], dataset[1])
    assert torch.equal(state, torch.get_rng_state())
    assert not torch.equal(dataset[0], dataset[1])


def test_language_evaluation_with_local_adapter():
    from beliefmerge.language import evaluate_sequence_model

    class Model(nn.Module):
        def generate(self, input_ids, **kwargs):
            return torch.ones(len(input_ids), 1, dtype=torch.long)

    class Tokenizer:
        def __call__(self, sources, **kwargs):
            return {"input_ids": torch.zeros(len(sources), 3, dtype=torch.long)}

        def batch_decode(self, values, **kwargs):
            return ["positive"] * len(values)

    result = evaluate_sequence_model(
        Model(),
        Tokenizer(),
        [{"sentence": "a", "label": 1}, {"sentence": "b", "label": 0}],
        "sst2",
    )
    assert result["accuracy"] == 0.5
    assert result["evaluated_examples"] == 2


def test_failed_slurm_submission_cannot_be_blindly_duplicated(tmp_path):
    from beliefmerge.workflow import ExperimentLedger, Job, Resources, Slurm

    ledger = ExperimentLedger(tmp_path)
    ledger.register([Job("task", ("python",), Resources("gpu", "h200"))])

    def uncertain(argv, **kwargs):
        raise TimeoutError("No reply")

    scheduler = Slurm(ledger, tmp_path, runner=uncertain)
    with pytest.raises(TimeoutError):
        scheduler.submit("task")
    assert ledger.snapshot()["jobs"]["task"]["state"] == "UNKNOWN"
    assert not scheduler.submit("task", retry_failed=True)["submitted"]


def test_slurm_success_requires_expected_artifacts(tmp_path):
    from beliefmerge.workflow import ExperimentLedger, Job, Resources, Slurm

    ledger = ExperimentLedger(tmp_path)
    ledger.register(
        [
            Job(
                "task",
                ("python",),
                Resources("gpu", "h200"),
                expected_artifacts=("task/result.json",),
            )
        ]
    )

    def runner(argv, **kwargs):
        return SimpleNamespace(
            stdout="200" if argv[0] == "sbatch" else "200|COMPLETED|0:0|"
        )

    scheduler = Slurm(ledger, tmp_path, runner=runner)
    scheduler.submit("task")
    assert scheduler.status() == {"task": "FAILED"}


def test_svd_cache_recomputes_changed_model_updates(tmp_path):
    from beliefmerge.baselines import get_svd_dict

    first = {"a": {"weight": torch.eye(3)}, "b": {"weight": 2 * torch.eye(3)}}
    path = tmp_path / "svd.pt"
    before = get_svd_dict(first, ["a", "b"], str(path))
    changed = copy.deepcopy(first)
    changed["a"]["weight"] *= 4
    after = get_svd_dict(changed, ["a", "b"], str(path))
    assert torch.allclose(after["a"]["weight"]["s"], 4 * before["a"]["weight"]["s"])
    reused = get_svd_dict(changed, ["a", "b"], str(path))
    assert torch.equal(after["a"]["weight"]["s"], reused["a"]["weight"]["s"])


def test_language_metrics_accept_negative_correlation_and_require_reference():
    from beliefmerge.language import (
        metrics_finite_unit_score,
        metrics_pinned_reference_scores,
        private_build_parser,
    )

    assert metrics_finite_unit_score(-0.2, label="stsb score") == -0.2
    with pytest.raises(ValueError):
        metrics_finite_unit_score(-0.2, label="cola score")
    with pytest.raises(ValueError, match="measured"):
        metrics_pinned_reference_scores()
    args = private_build_parser().parse_args(
        [
            "pack",
            "--step",
            "50",
            "--input",
            "cola=result.json",
            "--output",
            "packed.json",
        ]
    )
    assert args.command == "pack"
