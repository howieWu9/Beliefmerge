from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from beliefmerge.language import (
    FUSIONBENCH_COMMIT,
    GLUE_TASKS,
    VALIDATION_SPLITS,
    canonical_task_name,
    checkpoint_steps,
    configure_offline_environment,
    finetune_build_parser,
    finetune_CHECKPOINT_INTERVAL,
    finetune_dataset_path,
    finetune_DEFAULT_MAX_STEPS,
    finetune_parquet_split_files,
    finetune_validate_existing_boundaries,
    format_example,
    private_COMPONENTS,
    private_pack,
)
from beliefmerge.reference import (
    PAPER_CHECKPOINT_STEPS,
    PAPER_HYPERPARAMETER_GRID,
    PAPER_TABLE_VARIANTS,
    PAPER_TRAJECTORY_HISTORIES,
    AblationConfig,
    BeliefParameters,
    GridSubspaceMerger,
    PublicTypeMapper,
    _masked_public_features,
    build_public_geometry,
    calibrate_belief_parameters,
    enumerate_paper_configs,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _constant_mapper(feature_count: int, component_count: int) -> PublicTypeMapper:
    return PublicTypeMapper(
        feature_mean=np.zeros(feature_count, dtype=np.float64),
        feature_scale=np.ones(feature_count, dtype=np.float64),
        linear_coefficients=np.zeros(
            (feature_count + 1, component_count), dtype=np.float64
        ),
        nonlinear_projection=np.zeros((feature_count, 2), dtype=np.float64),
        nonlinear_bias=np.zeros(2, dtype=np.float64),
        nonlinear_coefficients=np.zeros((2, component_count), dtype=np.float64),
        component_count=component_count,
        fitted=True,
    )


def _numpy_states():
    base = OrderedDict(
        (
            ("block0.weight", np.zeros(2, dtype=np.float32)),
            ("block0.bias", np.zeros(1, dtype=np.float32)),
            ("block1.weight", np.zeros(2, dtype=np.float32)),
            ("block1.bias", np.zeros(1, dtype=np.float32)),
            ("step_counter", np.array(7, dtype=np.int64)),
        )
    )
    first = OrderedDict(
        (
            ("block0.weight", np.array([1.0, 0.0], dtype=np.float32)),
            ("block0.bias", np.array([0.1], dtype=np.float32)),
            ("block1.weight", np.array([1.0, 0.0], dtype=np.float32)),
            ("block1.bias", np.array([0.1], dtype=np.float32)),
            ("step_counter", np.array(101, dtype=np.int64)),
        )
    )
    second = OrderedDict(
        (
            ("block0.weight", np.array([0.0, 1.0], dtype=np.float32)),
            ("block0.bias", np.array([-0.1], dtype=np.float32)),
            ("block1.weight", np.array([0.0, 1.0], dtype=np.float32)),
            ("block1.bias", np.array([-0.1], dtype=np.float32)),
            ("step_counter", np.array(202, dtype=np.int64)),
        )
    )
    return (base, (first, second))


class PublicGeometryTest(unittest.TestCase):
    def test_public_g_z_geometry_is_layerwise_symmetric_and_deterministic(self):
        base, endpoints = _numpy_states()
        first = build_public_geometry(base, endpoints, rank=2)
        second = build_public_geometry(base, endpoints, rank=2)
        self.assertEqual(first.layer_names, ("block0", "block1"))
        self.assertEqual(first.public_features.shape, (2, 2, 4))
        self.assertEqual(first.grams.shape, (2, 2, 2))
        np.testing.assert_allclose(first.grams, first.grams.transpose(0, 2, 1))
        np.testing.assert_allclose(first.global_gram, first.grams.sum(axis=0))
        np.testing.assert_allclose(first.public_features, second.public_features)
        np.testing.assert_allclose(first.projected_grams, second.projected_grams)
        self.assertTrue(np.all(first.public_features[..., 0] > 0.0))
        for layer in range(len(first.layer_names)):
            for component in range(first.rank):
                direction = first.public_features[:, layer, 1 + component]
                pivot = int(np.argmax(np.abs(direction)))
                self.assertGreaterEqual(direction[pivot], 0.0)
        self.assertNotIn("step_counter", first.parameter_to_layer)

    def test_public_ablation_uses_mapper_mean_not_raw_zero(self):
        values = np.array([[[9.0, -4.0, 3.0, 0.7]]])
        neutral = np.array([2.0, 5.0, -1.0, 0.25])
        config = AblationConfig(
            rank=2, use_g=False, use_z=False, use_compatibility=False
        )
        masked = _masked_public_features(values, config, neutral)
        np.testing.assert_allclose(masked, neutral.reshape(1, 1, -1))

    def test_geometry_rejects_key_and_shape_mismatches(self):
        base, endpoints = _numpy_states()
        missing = OrderedDict(endpoints[0])
        missing.pop("block0.bias")
        with self.assertRaisesRegex(ValueError, "keys differ"):
            build_public_geometry(base, (missing, endpoints[1]), rank=1)
        wrong_shape = OrderedDict(endpoints[0])
        wrong_shape["block0.weight"] = np.ones(3, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            build_public_geometry(base, (wrong_shape, endpoints[1]), rank=1)


class BeliefTest(unittest.TestCase):
    def test_gaussian_update_contracts_shared_posterior_and_improves_type_nll(self):
        base, endpoints = _numpy_states()
        mapper = _constant_mapper(3, 3)
        parameters = BeliefParameters(
            shared_mean=np.zeros(3),
            shared_covariance=np.eye(3) * 0.1,
            private_covariance=np.eye(3) * 0.01,
            calibration_observations=100,
        )
        private = np.full((2, 2, 3), 0.8, dtype=np.float64)
        common = dict(
            rank=1,
            strategy_mode="bayesian_only",
            monte_carlo_samples=2,
            max_iterations=50,
            min_iterations=50,
        )
        updated = GridSubspaceMerger(
            AblationConfig(**common), mapper=mapper, belief_parameters=parameters
        ).merge(base, endpoints, private)
        prior_only = GridSubspaceMerger(
            AblationConfig(**common, bayesian_update=False),
            mapper=mapper,
            belief_parameters=parameters,
        ).merge(base, endpoints, private)
        posterior_diag = np.diagonal(
            updated.belief.posterior_shared_covariance, axis1=-2, axis2=-1
        )
        self.assertTrue(np.all(posterior_diag < 0.1))
        self.assertGreater(updated.belief.posterior_shared_mean.mean(), 0.0)
        self.assertTrue(np.isfinite(updated.belief.type_nll))
        self.assertLess(updated.belief.type_nll, prior_only.belief.type_nll)
        self.assertTrue(np.isnan(updated.belief.pairwise_type_nll[0, 0]).all())

    def test_mapper_and_historical_belief_calibration(self):
        rng = np.random.default_rng(9)
        public = rng.normal(size=(4, 3, 2, 5))
        logits = np.stack(
            (
                0.3 + 0.5 * public[..., 0],
                -0.2 + 0.4 * public[..., 1],
                0.1 - 0.3 * public[..., 2],
            ),
            axis=-1,
        )
        private = 1.0 / (1.0 + np.exp(-logits))
        mapper = PublicTypeMapper.fit(public, private, hidden_features=8, seed=3)
        predicted = mapper.predict(public)
        self.assertEqual(predicted.shape, private.shape)
        self.assertLess(float(np.mean((predicted - private) ** 2)), 0.01)
        parameters = calibrate_belief_parameters(public, private, mapper)
        self.assertEqual(parameters.shared_covariance.shape, (3, 3))
        self.assertEqual(parameters.private_covariance.shape, (3, 3))
        self.assertTrue(np.linalg.eigvalsh(parameters.shared_covariance).min() > 0)
        self.assertTrue(np.linalg.eigvalsh(parameters.private_covariance).min() > 0)
        self.assertEqual(parameters.calibration_observations, 4 * 3 * 2)
        with self.assertRaisesRegex(ValueError, "strictly"):
            calibrate_belief_parameters(public, np.zeros_like(private), mapper)


class NashMergeTest(unittest.TestCase):
    def _config(self, **overrides):
        values = dict(
            rank=1,
            lambda_t=1.0,
            lambda_o=0.0,
            lambda_b=0.02,
            monte_carlo_samples=12,
            max_iterations=100,
            min_iterations=100,
            trace_interval=50,
            best_response_grid=17,
            relaxation=0.5,
            seed=17,
        )
        values.update(overrides)
        return AblationConfig(**values)

    def test_layerwise_nash_softmax_merge_and_step_50_trace(self):
        base, endpoints = _numpy_states()
        private = np.array(
            [
                [[0.95, 0.95, 0.95], [0.2, 0.2, 0.2]],
                [[0.2, 0.2, 0.2], [0.95, 0.95, 0.95]],
            ],
            dtype=np.float64,
        )
        result = GridSubspaceMerger(
            self._config(), mapper=_constant_mapper(3, 3)
        ).merge(base, endpoints, private, model_names=("left", "right"))
        self.assertEqual(result.actions.shape, (2, 2))
        self.assertEqual(result.layer_weights.shape, (2, 2))
        np.testing.assert_allclose(result.layer_weights.sum(axis=0), 1.0)
        self.assertGreater(result.layer_weights[0, 0], result.layer_weights[1, 0])
        self.assertGreater(result.layer_weights[1, 1], result.layer_weights[0, 1])
        self.assertEqual(
            [record["step"] for record in result.optimization_trace], [0, 50, 100]
        )
        self.assertAlmostEqual(
            float(result.best_response_gap.max()),
            result.optimization_trace[-1]["max_expected_best_response_gap"],
        )
        for key in ("block0.weight", "block0.bias", "block1.weight", "block1.bias"):
            layer = result.geometry.parameter_to_layer[key]
            expected = np.asarray(base[key], dtype=np.float64).copy()
            for model, endpoint in enumerate(endpoints):
                expected += result.layer_weights[model, layer] * (
                    np.asarray(endpoint[key]) - np.asarray(base[key])
                )
            np.testing.assert_allclose(result.state_dict[key], expected, atol=1e-06)
            self.assertEqual(result.state_dict[key].dtype, base[key].dtype)
        np.testing.assert_array_equal(result.state_dict["step_counter"], 7)
        serialized = json.dumps(result.diagnostics(), allow_nan=False)
        self.assertIn('"trace_interval": 50', serialized)
        self.assertIn("mapper", result.diagnostics())
        np.testing.assert_allclose(
            result.diagnostics()["belief"]["private_types"], private
        )
        self.assertIsNone(result.diagnostics()["belief"]["pairwise_type_nll"][0][0][0])

    def test_precomputed_geometry_is_reused_and_structurally_validated(self):
        base, endpoints = _numpy_states()
        private = np.full((2, 2, 3), 0.6)
        geometry = build_public_geometry(base, endpoints, rank=1)
        merger = GridSubspaceMerger(
            self._config(strategy_mode="uniform"), mapper=_constant_mapper(3, 3)
        )
        cached = merger.merge(base, endpoints, private, geometry=geometry)
        fresh = merger.merge(base, endpoints, private)
        np.testing.assert_allclose(cached.layer_weights, fresh.layer_weights)
        for key in base:
            np.testing.assert_array_equal(cached.state_dict[key], fresh.state_dict[key])
        with self.assertRaisesRegex(ValueError, "layer_map"):
            merger.merge(
                base, endpoints, private, geometry=geometry, layer_map=lambda key: key
            )
        rank_two = GridSubspaceMerger(
            self._config(rank=2, strategy_mode="uniform"), mapper=_constant_mapper(4, 3)
        )
        with self.assertRaisesRegex(ValueError, "rank"):
            rank_two.merge(base, endpoints, private, geometry=geometry)

    def test_private_components_are_layerwise_and_ablatable(self):
        base, endpoints = _numpy_states()
        private = np.array(
            [
                [[0.2, 0.4, 0.5, 0.8], [0.3, 0.5, 0.6, 0.9]],
                [[0.6, 0.7, 0.8, 0.4], [0.5, 0.8, 0.9, 0.3]],
            ]
        )
        config = AblationConfig.variant(
            "private_without_importance", rank=1, strategy_mode="uniform"
        )
        result = GridSubspaceMerger(config, mapper=_constant_mapper(3, 4)).merge(
            base, endpoints, private
        )
        np.testing.assert_allclose(
            result.type_strength, private[..., 1] * private[..., 2]
        )
        stability = GridSubspaceMerger(
            AblationConfig.variant(
                "private_plus_stability", rank=1, strategy_mode="uniform"
            ),
            mapper=_constant_mapper(3, 4),
        ).merge(base, endpoints, private)
        np.testing.assert_allclose(stability.type_strength, private.prod(axis=-1))
        with self.assertRaisesRegex(ValueError, "four components"):
            GridSubspaceMerger(
                AblationConfig.variant(
                    "private_plus_stability", rank=1, strategy_mode="uniform"
                ),
                mapper=_constant_mapper(3, 3),
            ).merge(base, endpoints, private[..., :3])

    def test_seeded_game_is_reproducible(self):
        base, endpoints = _numpy_states()
        private = np.array(
            [[[0.8, 0.7, 0.9], [0.4, 0.5, 0.6]], [[0.3, 0.5, 0.6], [0.9, 0.8, 0.7]]]
        )
        merger = GridSubspaceMerger(self._config(), mapper=_constant_mapper(3, 3))
        first = merger.merge(base, endpoints, private)
        second = merger.merge(base, endpoints, private)
        np.testing.assert_array_equal(first.actions, second.actions)
        self.assertEqual(first.optimization_trace, second.optimization_trace)


class PaperMatrixTest(unittest.TestCase):
    def test_all_table_4_to_10_switches_are_enumerable(self):
        expected_lengths = {4: 8, 5: 4, 6: 4, 7: 4, 10: 23}
        for table, expected in expected_lengths.items():
            with self.subTest(table=table):
                rows = enumerate_paper_configs(
                    table, max_iterations=50, min_iterations=50, monte_carlo_samples=2
                )
                self.assertEqual(len(rows), expected)
                self.assertEqual(len({name for name, _ in rows}), expected)
                self.assertTrue(
                    all((isinstance(config, AblationConfig) for _, config in rows))
                )
        table4 = dict(enumerate_paper_configs(4))
        self.assertFalse(table4["public_without_update_magnitude"].use_g)
        self.assertFalse(table4["public_without_update_direction"].use_z)
        self.assertTrue(table4["public_plus_compatibility"].use_compatibility)
        self.assertFalse(table4["private_without_importance"].use_sensitivity)
        self.assertTrue(table4["private_plus_stability"].use_stability)
        table5 = dict(enumerate_paper_configs("table_5"))
        self.assertEqual(table5["linear_mapping"].mapping_kind, "linear")
        self.assertFalse(table5["without_bayesian_update"].bayesian_update)
        self.assertTrue(table5["independent_belief"].independent_belief)
        table6 = dict(enumerate_paper_configs(6))
        self.assertFalse(table6["without_shared_latent"].shared_latent)
        self.assertEqual(table6["student_t"].student_df, 3.0)
        table7 = dict(enumerate_paper_configs(7))
        self.assertEqual(table7["nash_only"].strategy_mode, "nash_only")
        self.assertFalse(table7["nash_only"].bayesian_update)
        self.assertEqual(table7["bayesian_only"].strategy_mode, "bayesian_only")
        self.assertEqual(table7["uniform_without_strategy"].strategy_mode, "uniform")
        manifest_named = AblationConfig.variant(
            "full", lambda_type=0.2, lambda_public=0.8, lambda_action=0.06
        )
        self.assertEqual(
            (manifest_named.lambda_t, manifest_named.lambda_o, manifest_named.lambda_b),
            (0.2, 0.8, 0.06),
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            AblationConfig.variant("full", lambda_type=0.2, lambda_t=0.4)
        self.assertEqual(
            set(PAPER_TABLE_VARIANTS), {"table4", "table5", "table6", "table7"}
        )
        self.assertEqual(sum(map(len, PAPER_HYPERPARAMETER_GRID.values())), 23)
        self.assertEqual(len(PAPER_CHECKPOINT_STEPS), 40)
        self.assertEqual(PAPER_CHECKPOINT_STEPS[0], 50)
        self.assertEqual(PAPER_CHECKPOINT_STEPS[-1], 2000)
        self.assertTrue(
            all(
                (
                    right - left == 50
                    for left, right in zip(
                        PAPER_CHECKPOINT_STEPS, PAPER_CHECKPOINT_STEPS[1:]
                    )
                )
            )
        )
        self.assertEqual(PAPER_TRAJECTORY_HISTORIES["final_only"], (2000,))
        self.assertEqual(
            PAPER_TRAJECTORY_HISTORIES["two_intermediates"], (500, 1000, 2000)
        )
        self.assertEqual(
            PAPER_TRAJECTORY_HISTORIES["full_step_50"], PAPER_CHECKPOINT_STEPS
        )
        self.assertEqual(
            PAPER_TRAJECTORY_HISTORIES["full_trajectory"], PAPER_CHECKPOINT_STEPS
        )
        with self.assertRaisesRegex(ValueError, "run metadata"):
            enumerate_paper_configs(8)
        with self.assertRaises(KeyError):
            enumerate_paper_configs(11)


@unittest.skipIf(torch is None, "PyTorch is optional for the NumPy CPU core")
class TorchStateCompatibilityTest(unittest.TestCase):
    def test_torch_dtype_device_and_non_floating_buffers_are_preserved(self):
        base_np, endpoints_np = _numpy_states()

        def convert(state):
            converted = OrderedDict()
            for key, value in state.items():
                converted[key] = torch.as_tensor(value)
            return converted

        base = convert(base_np)
        endpoints = tuple((convert(endpoint) for endpoint in endpoints_np))
        private = np.full((2, 2, 3), 0.5)
        result = GridSubspaceMerger(
            AblationConfig(rank=1, strategy_mode="uniform"),
            mapper=_constant_mapper(3, 3),
        ).merge(base, endpoints, private)
        for key in base:
            self.assertIsInstance(result.state_dict[key], torch.Tensor)
            self.assertEqual(result.state_dict[key].dtype, base[key].dtype)
            self.assertEqual(result.state_dict[key].device, base[key].device)
        torch.testing.assert_close(
            result.state_dict["block0.weight"],
            torch.tensor([0.5, 0.5], dtype=torch.float32),
        )
        torch.testing.assert_close(
            result.state_dict["step_counter"], base["step_counter"]
        )
        bf16_base = OrderedDict(
            (
                (key, value.to(torch.bfloat16) if value.is_floating_point() else value)
                for key, value in base.items()
            )
        )
        bf16_endpoints = tuple(
            (
                OrderedDict(
                    (
                        (
                            key,
                            value.to(torch.bfloat16)
                            if value.is_floating_point()
                            else value,
                        )
                        for key, value in endpoint.items()
                    )
                )
                for endpoint in endpoints
            )
        )
        bf16 = GridSubspaceMerger(
            AblationConfig(rank=1, strategy_mode="uniform"),
            mapper=_constant_mapper(3, 3),
        ).merge(bf16_base, bf16_endpoints, private)
        self.assertEqual(bf16.state_dict["block0.weight"].dtype, torch.bfloat16)


class GlueProtocolTest(unittest.TestCase):
    def test_pinned_protocol_and_gap_free_step50_schedule(self) -> None:
        self.assertEqual(FUSIONBENCH_COMMIT, "54c9e8c9d9621620c720452cd8533332a32d3689")
        self.assertEqual(len(GLUE_TASKS), 8)
        steps = checkpoint_steps(
            finetune_DEFAULT_MAX_STEPS, finetune_CHECKPOINT_INTERVAL
        )
        self.assertEqual(steps, tuple(range(50, 2001, 50)))
        self.assertEqual(len(steps), 40)
        with self.assertRaisesRegex(ValueError, "divisible"):
            checkpoint_steps(201, 50)

    def test_task_aliases_and_mnli_split_are_exact(self) -> None:
        self.assertEqual(canonical_task_name("glue-sst-2"), "sst2")
        self.assertEqual(canonical_task_name("STS-B"), "stsb")
        self.assertEqual(VALIDATION_SPLITS["mnli"], "validation_matched")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            canonical_task_name("wnli")

    def test_prompts_match_pinned_fusionbench_spelling_and_labels(self) -> None:
        source, target = format_example(
            "cola", {"sentence": "This is grammatical.", "label": 1}
        )
        self.assertEqual(
            source,
            "Indicate if the following sentence is grammatically correct or not: \"This is grammatical.\". Answere 'acceptable' or 'unacceptable'.",
        )
        self.assertEqual(target, "acceptable")
        source, target = format_example(
            "stsb", {"sentence1": "a", "sentence2": "b", "label": 3.25}
        )
        self.assertIn("On a scale from 1", source)
        self.assertEqual(target, "3.2")

    def test_raw_parquet_layout_is_grouped_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "validation_mismatched.parquet",
                "train-00001.parquet",
                "validation_matched.parquet",
                "train-00000.parquet",
            ):
                (root / name).write_bytes(b"parquet-placeholder")
            grouped = finetune_parquet_split_files(root)
            self.assertEqual(
                [path.name for path in grouped["train"]],
                ["train-00000.parquet", "train-00001.parquet"],
            )
            self.assertEqual(
                [path.name for path in grouped["validation_matched"]],
                ["validation_matched.parquet"],
            )
            self.assertEqual(finetune_dataset_path(root, "mnli"), root.resolve())

    def test_dataset_root_finds_staged_task_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / "glue" / "qqp"
            task.mkdir(parents=True)
            (task / "train.parquet").write_bytes(b"x")
            self.assertEqual(finetune_dataset_path(root, "qqp"), task.resolve())

    def test_existing_checkpoint_history_must_be_prefix_without_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint-50").mkdir()
            (root / "checkpoint-100").mkdir()
            found = finetune_validate_existing_boundaries(root, (50, 100, 150))
            self.assertEqual(sorted(found), [50, 100])
            (root / "checkpoint-150").mkdir()
            (root / "checkpoint-100").rmdir()
            with self.assertRaisesRegex(ValueError, "gaps"):
                finetune_validate_existing_boundaries(root, (50, 100, 150))

    def test_offline_environment_places_all_caches_under_explicit_root(self) -> None:
        old = dict(os.environ)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = configure_offline_environment(temporary)
                for variable in (
                    "HF_HOME",
                    "HF_DATASETS_CACHE",
                    "TRANSFORMERS_CACHE",
                    "XDG_CACHE_HOME",
                    "TORCH_HOME",
                    "TMPDIR",
                    "TMP",
                    "TEMP",
                ):
                    self.assertTrue(Path(os.environ[variable]).is_relative_to(root))
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_cli_defaults_lock_full_curve(self) -> None:
        parser = finetune_build_parser()
        args = parser.parse_args(
            [
                "--task",
                "cola",
                "--base-model-dir",
                "base",
                "--tokenizer-dir",
                "tokenizer",
                "--dataset-dir",
                "dataset",
                "--output-dir",
                "output",
                "--cache-root",
                "cache",
            ]
        )
        self.assertEqual(args.max_steps, 2000)
        self.assertEqual(args.checkpoint_interval, 50)


class GluePrivateTypesTest(unittest.TestCase):
    def test_pack_preserves_task_order_layers_and_four_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for index, task in enumerate(GLUE_TASKS):
                path = root / f"{task}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "artifact_type": "glue_layerwise_private_types",
                            "complete": True,
                            "test_data_used": False,
                            "task": task,
                            "step": 50,
                            "component_order": list(private_COMPONENTS),
                            "layer_names": ["shared_embedding", "encoder.block.0"],
                            "values": np.full((2, 4), 0.1 + index / 20).tolist(),
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append(f"{task}={path}")
            output = root / "packed.json"
            args = argparse.Namespace(step=50, input=inputs, output=str(output))
            self.assertEqual(private_pack(args), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_names"], list(GLUE_TASKS))
            self.assertEqual(np.asarray(payload["private_types"]).shape, (8, 2, 4))
            self.assertFalse(payload["test_data_used"])


try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    torch = None
if torch is not None:
    from beliefmerge.vision import (
        PRIVATE_TYPE_COMPONENTS,
        PrivateTypeConfig,
        build_layer_map,
        load_private_type_cache,
        measure_private_types,
        resolve_logical_layer,
        save_private_type_cache,
    )


@unittest.skipIf(torch is None, "PyTorch is required for vision private-type tests")
class PaperVisionPrivateTypesTest(unittest.TestCase):
    class TinyEncoder(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(3, 3), torch.nn.Linear(3, 2)]
            )

        def forward(self, value):
            return self.blocks[1](torch.tanh(self.blocks[0](value)))

    def _models_and_loader(self):
        torch.manual_seed(4)
        base = self.TinyEncoder()
        endpoint = self.TinyEncoder()
        endpoint.load_state_dict(base.state_dict())
        with torch.no_grad():
            endpoint.blocks[0].weight.add_(0.2)
            endpoint.blocks[1].bias.sub_(0.1)
        images = torch.tensor(
            [
                [1.0, 0.0, -1.0],
                [0.2, 0.8, 0.1],
                [-0.4, 0.5, 0.7],
                [1.2, -0.3, 0.2],
                [0.1, 0.2, 0.3],
                [-0.8, 0.1, 0.9],
            ]
        )
        labels = torch.tensor([0, 1, 1, 0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=3)
        base_state = OrderedDict(
            ((key, value.detach().clone()) for key, value in base.state_dict().items())
        )
        return (base, endpoint, base_state, loader)

    def test_logical_layer_resolver_groups_transformer_blocks(self):
        self.assertEqual(
            resolve_logical_layer(
                "model.visual.transformer.resblocks.11.mlp.c_fc.weight"
            ),
            "model.visual.transformer.resblocks.11",
        )
        self.assertEqual(
            resolve_logical_layer("encoder.layers.3.self_attn.in_proj_weight"),
            "encoder.layers.3",
        )
        self.assertEqual(resolve_logical_layer("visual.conv1.weight"), "visual.conv1")

    def test_measurement_is_layerwise_bounded_and_restores_endpoint(self):
        _, endpoint, base_state, loader = self._models_and_loader()
        before = {
            key: value.detach().clone() for key, value in endpoint.state_dict().items()
        }
        mapping = build_layer_map(endpoint.state_dict())
        result = measure_private_types(
            task="tiny",
            encoder=endpoint,
            head=torch.nn.Identity(),
            loader=loader,
            base_state=base_state,
            endpoint_state=endpoint.state_dict(),
            layer_map=mapping,
            device="cpu",
            split="type_cal",
            config=PrivateTypeConfig(
                max_batches=2, max_samples=6, sensitivity_tau=1e-06
            ),
            source={"base_sha256": "base", "endpoint_sha256": "endpoint"},
        )
        self.assertEqual(result.layer_names, ("blocks.0", "blocks.1"))
        self.assertEqual(result.values.shape, (2, len(PRIVATE_TYPE_COMPONENTS)))
        self.assertTrue(np.isfinite(result.values).all())
        self.assertTrue((result.values >= 0.0).all())
        self.assertTrue((result.values <= 1.0).all())
        self.assertEqual(result.sample_count, 6)
        self.assertEqual(result.batch_count, 2)
        self.assertEqual(result.matrix().shape, (2, 3))
        self.assertEqual(result.matrix(include_stability=True).shape, (2, 4))
        for key, value in endpoint.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0.0, atol=0.0)

    def test_cache_round_trip_validates_source_and_never_marks_test_use(self):
        _, endpoint, base_state, loader = self._models_and_loader()
        mapping = build_layer_map(endpoint.state_dict())
        config = PrivateTypeConfig(max_batches=1, max_samples=3, sensitivity_tau=1e-06)
        source = {
            "base_sha256": "a",
            "endpoint_sha256": "b",
            "endpoint_kind": "unit_test",
            "endpoint_step": 50,
        }
        result = measure_private_types(
            task="tiny",
            encoder=endpoint,
            head=torch.nn.Identity(),
            loader=loader,
            base_state=base_state,
            layer_map=mapping,
            device="cpu",
            split="train",
            config=config,
            source=source,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.json"
            save_private_type_cache(path, result)
            loaded = load_private_type_cache(
                path,
                expected_task="tiny",
                expected_layers=result.layer_names,
                expected_source=source,
                expected_config=config,
            )
            np.testing.assert_allclose(loaded.values, result.values)
            payload = loaded.to_dict()
            self.assertIs(payload["test_data_used"], False)
            self.assertNotIn("test", payload["raw_statistics"])
            with self.assertRaises(ValueError):
                load_private_type_cache(
                    path,
                    expected_task="tiny",
                    expected_source={"endpoint_sha256": "changed"},
                )

    def test_test_or_validation_split_is_rejected(self):
        _, endpoint, base_state, loader = self._models_and_loader()
        for split in ("test", "validation", "val"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                measure_private_types(
                    task="tiny",
                    encoder=endpoint,
                    head=torch.nn.Identity(),
                    loader=loader,
                    base_state=base_state,
                    device="cpu",
                    split=split,
                )
