# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the DiffusionART loss wiring, actor-batch preparation and config validation.

Necessity: the loss reads cached ODE states, a per-prompt guidance tensor and the next flow time from
the batch; a wrong gather, a guidance assigned to the wrong prompt, or a config that silently falls
back to a stochastic rollout would train without any error. These tests pin the batch contract and
the start-up validation to what ``ARTDiffusersFSDPEngine`` and the recipe assume.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils as tu

import verl_omni
from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionARTLoss, get_diffusion_loss_fn
from verl_omni.trainer.diffusion.diffusion_art import art_loss
from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.utils.losses import diffusion_loss

C, H, W = 4, 3, 3


def make_actor(**loss_kwargs) -> DiffusionActorConfig:
    loss_cfg = DiffusionLossConfig(loss_mode="diffusion_art", **loss_kwargs)
    return DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=loss_cfg)


def make_inputs(batch=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    old = torch.randn(batch, C, H, W, generator=generator)
    raw = torch.randn(batch, C, H, W, generator=generator)
    guidance = raw / raw.flatten(1).square().mean(dim=1).sqrt()[:, None, None, None]
    t = torch.linspace(1.0, 0.1, batch)
    return {
        "forward_prediction": (old + 0.01 * torch.randn(batch, C, H, W, generator=generator)).requires_grad_(True),
        "old_prediction": old,
        "t_expanded": t.view(batch, 1, 1, 1),
        "art_guidance": guidance,
        "sigma_next": (t - 0.1).clamp_min(0.0),
    }


def run_loss(inputs, **loss_kwargs):
    model_output = {key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys}
    data = TensorDict(
        {key: inputs[key] for key in DiffusionARTLoss.required_data_keys}, batch_size=[len(inputs["sigma_next"])]
    )
    return get_diffusion_loss_fn("diffusion_art")(
        config=make_actor(**loss_kwargs), model_output=model_output, data=data
    )


def test_registry_and_contract():
    loss_fn = get_diffusion_loss_fn("diffusion_art")
    assert isinstance(loss_fn, DiffusionARTLoss)
    assert loss_fn.uses_old_adapter and not loss_fn.requires_whole_groups
    assert DiffusionLossConfig(loss_mode="diffusion_art").art_mask_metric == "sde_kl"


def test_loss_matches_the_pure_function_and_reports_metrics():
    inputs = make_inputs()
    result = run_loss(inputs, art_guidance_gain=0.5, art_mask_metric="fixed_mse", art_mask_threshold=1e-3)
    terms = art_loss(
        inputs["forward_prediction"],
        inputs["old_prediction"],
        inputs["art_guidance"],
        inputs["t_expanded"].flatten(),
        inputs["sigma_next"],
        guidance_gain=0.5,
        mask_metric="fixed_mse",
        mask_threshold=1e-3,
    )
    torch.testing.assert_close(result.loss, terms["per_item_loss"].mean())
    for key in (
        "actor/policy_loss",
        "actor/unmasked_loss",
        "actor/active_frac",
        "actor/mask_frac",
        "actor/outward_frac",
        "actor/old_deviate",
        "actor/movement_score",
        "actor/guidance_rms",
        "actor/total_loss",
    ):
        assert key in result.metrics, key
    assert result.metrics["actor/active_frac"] == 1.0
    assert result.metrics["actor/guidance_rms"] == pytest.approx(1.0, rel=1e-5)


def test_loss_accepts_full_shape_t_expanded_and_requires_its_inputs():
    inputs = make_inputs()
    inputs["t_expanded"] = inputs["t_expanded"].expand(-1, C, H, W)
    assert torch.isfinite(run_loss(inputs).loss)
    with pytest.raises(KeyError, match="art_guidance"):
        get_diffusion_loss_fn("diffusion_art")(
            config=make_actor(),
            model_output={key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys},
            data=TensorDict({"sigma_next": inputs["sigma_next"]}, batch_size=[4]),
        )


def test_through_the_worker_loss_entry_point_divides_by_gradient_accumulation():
    inputs = make_inputs()
    data = TensorDict({key: inputs[key] for key in DiffusionARTLoss.required_data_keys}, batch_size=[4])
    tu.assign_non_tensor(data, gradient_accumulation_steps=4, sp_size=1)
    model_output = {key: inputs[key] for key in DiffusionARTLoss.required_model_output_keys}
    loss, metrics = diffusion_loss(make_actor(), model_output, data)
    torch.testing.assert_close(loss, run_loss(inputs).loss / 4)
    assert "actor/mask_frac" in metrics


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"art_guidance_gain": 0.0}, "art_guidance_gain.*positive"),
        ({"art_loss_space": "x0"}, "art_loss_space"),
        ({"art_mask_metric": "kl"}, "art_mask_metric"),
        ({"art_mask_threshold": 0.0}, "art_mask_threshold.*positive"),
        ({"art_reference_noise_level": 0.0}, "art_reference_noise_level.*positive"),
    ],
)
def test_invalid_loss_config_values_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DiffusionLossConfig(loss_mode="diffusion_art", **kwargs)


def test_algorithm_config_exposes_pairwise_flag():
    assert DiffusionAlgoConfig().art_pairwise_reward is False
    assert DiffusionAlgoConfig(art_pairwise_reward=True).art_pairwise_reward is True


def make_rollout_batch(uid, num_steps=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    batch_size = len(uid)
    schedule = torch.linspace(1000.0, 200.0, num_steps)
    return DataProto.from_dict(
        tensors={
            "all_latents": torch.randn(batch_size, num_steps + 1, C, H, W, generator=generator),
            "all_timesteps": schedule.expand(batch_size, -1).clone(),
            "all_log_probs": torch.zeros(batch_size, num_steps),
            "prompt_embeds": torch.zeros(batch_size, 7, 8),
        },
        non_tensors={"uid": np.array(uid, dtype=object)},
    )


def make_config(**algorithm_kwargs):
    algorithm = SimpleNamespace(
        norm_adv_by_std_in_grpo=True, global_std=True, timestep_fraction=1.0, art_pairwise_reward=False
    )
    for key, value in algorithm_kwargs.items():
        setattr(algorithm, key, value)
    algorithm.get = lambda key, default=None: getattr(algorithm, key, default)
    actor = SimpleNamespace(data_loader_seed=42)
    return SimpleNamespace(algorithm=algorithm, actor_rollout_ref=SimpleNamespace(actor=actor))


def test_prepare_actor_batch_writes_cached_states_and_shared_guidance():
    uid = ["b", "a", "b", "a", "c", "c"]
    batch = make_rollout_batch(uid)
    all_latents = batch.batch["all_latents"].clone()
    all_timesteps = batch.batch["all_timesteps"].clone()
    rewards = torch.tensor([0.1, 0.9, 0.3, 0.5, 0.2, 0.2])

    result = DiffusionARTLoss.prepare_actor_batch(batch, rewards, make_config())

    num_steps = all_timesteps.shape[1]
    assert result.batch["train_latents"].shape == (6, num_steps, C, H, W)
    assert result.batch["train_timesteps"].shape == (6, num_steps)
    assert result.batch["art_guidance"].shape == (6, C, H, W)
    assert "all_latents" not in result.batch.keys() and "all_log_probs" not in result.batch.keys()
    assert "prompt_embeds" in result.batch.keys()
    torch.testing.assert_close(result.batch["latents_clean"], all_latents[:, -1])

    # Every selected state is the latent the rollout visited at the selected timestep, and the next
    # timestep follows the schedule (0 after the last step).
    for row in range(6):
        for slot in range(num_steps):
            step = (all_timesteps[row] == result.batch["train_timesteps"][row, slot]).nonzero().item()
            torch.testing.assert_close(result.batch["train_latents"][row, slot], all_latents[row, step])
            expected_next = all_timesteps[row, step + 1] if step + 1 < num_steps else torch.tensor(0.0)
            assert result.batch["train_timesteps_next"][row, slot] == expected_next
    assert sorted(result.batch["train_timesteps"][0].tolist()) == sorted(all_timesteps[0].tolist())

    guidance = result.batch["art_guidance"]
    torch.testing.assert_close(guidance[0], guidance[2])  # same prompt "b"
    torch.testing.assert_close(guidance[1], guidance[3])  # same prompt "a"
    assert torch.count_nonzero(guidance[4]) == 0 and torch.count_nonzero(guidance[5]) == 0  # constant rewards
    assert guidance[0].square().mean().sqrt().item() == pytest.approx(1.0, rel=1e-5)
    assert result.batch["advantages"].shape == (6, num_steps)
    assert result.batch["sample_level_rewards"].shape == (6, num_steps)


def test_prepare_actor_batch_subsamples_states_and_supports_pairwise_labels():
    uid = ["a", "a", "a", "a"]
    batch = make_rollout_batch(uid, num_steps=6)
    rewards = torch.tensor([0.9, 0.1, 0.4, 0.6])
    result = DiffusionARTLoss.prepare_actor_batch(
        batch, rewards, make_config(timestep_fraction=0.5, art_pairwise_reward=True)
    )
    assert result.batch["train_latents"].shape == (4, 3, C, H, W)
    assert result.batch["train_timesteps"].shape == (4, 3)
    assert result.batch["art_guidance"][0].square().mean().sqrt().item() == pytest.approx(1.0, rel=1e-5)


def test_prepare_actor_batch_rejects_missing_or_truncated_trajectories():
    batch = make_rollout_batch(["a", "a"])
    with pytest.raises(ValueError, match="all_latents"):
        DiffusionARTLoss.prepare_actor_batch(
            DataProto.from_dict(
                tensors={"all_timesteps": batch.batch["all_timesteps"]},
                non_tensors={"uid": np.array(["a", "a"], dtype=object)},
            ),
            torch.tensor([0.0, 1.0]),
            make_config(),
        )
    truncated = make_rollout_batch(["a", "a"])
    truncated.batch["all_latents"] = truncated.batch["all_latents"][:, 1:]
    with pytest.raises(ValueError, match="full ODE trajectory"):
        DiffusionARTLoss.prepare_actor_batch(truncated, torch.tensor([0.0, 1.0]), make_config())


def compose_trainer_config(overrides):
    config_dir = Path(verl_omni.__file__).parent / "trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


ART_OVERRIDES = [
    "data.train_batch_size=24",
    "algorithm.trainer_type=direct_preference",
    "algorithm.sample_source=online",
    "actor_rollout_ref.model.algorithm=diffusion_art",
    "actor_rollout_ref.model.model_type=diffusion_art_model",
    "actor_rollout_ref.model.lora_rank=64",
    'actor_rollout_ref.model.policy_state_adapters=["default","old"]',
    "actor_rollout_ref.actor.ppo_mini_batch_size=12",
    "actor_rollout_ref.rollout.rollout_adapter=old",
    "actor_rollout_ref.rollout.algo.noise_level=0.0",
]


def test_validate_trainer_config_accepts_the_recipe_and_cascades_the_algorithm_name(caplog):
    config = compose_trainer_config(ART_OVERRIDES)
    assert config.actor_rollout_ref.actor.diffusion_loss.loss_mode == "diffusion_art"
    with caplog.at_level(logging.WARNING):
        DiffusionARTLoss.validate_trainer_config(config)
    assert "two disjoint updates" not in caplog.text


def test_validate_trainer_config_warns_on_a_single_update_per_rollout(caplog):
    config = compose_trainer_config(ART_OVERRIDES + ["actor_rollout_ref.actor.ppo_mini_batch_size=24"])
    with caplog.at_level(logging.WARNING):
        DiffusionARTLoss.validate_trainer_config(config)
    assert "two disjoint updates" in caplog.text


@pytest.mark.parametrize(
    "override, match",
    [
        ("actor_rollout_ref.rollout.rollout_adapter=default", "rollout_adapter must be old"),
        ("actor_rollout_ref.rollout.algo.noise_level=0.7", "noise_level must be 0.0"),
        ("actor_rollout_ref.model.lora_rank=0", "lora_rank must be > 0"),
        ("actor_rollout_ref.actor.use_kl_loss=true", "use_kl_loss must be False"),
        ("actor_rollout_ref.actor.enable_timestep_staging=true", "enable_timestep_staging must be False"),
        ("algorithm.train_timestep_count=3", "only used by the dgpo loss"),
    ],
)
def test_validate_trainer_config_rejects_unsupported_settings(override, match):
    config = compose_trainer_config(ART_OVERRIDES + [override])
    with pytest.raises(ValueError, match=match):
        DiffusionARTLoss.validate_trainer_config(config)
