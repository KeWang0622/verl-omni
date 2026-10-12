# DiffusionART Trainer

Last updated: 10/12/2026

This example post-trains `Qwen/Qwen-Image` with
[DiffusionART](../../docs/algo/diffusionart.md) (aggregation-first reward
tilting, [arXiv:2610.12572](https://arxiv.org/abs/2610.12572)) on the OCR text
rendering task, using `vllm-omni` rollout and `Qwen/Qwen3-VL-8B-Instruct` as the
OCR reward model.

DiffusionART is an online direct-preference algorithm in verl-omni's taxonomy:
rollouts are deterministic ODE trajectories sampled with the `old` LoRA adapter,
the actor trains from the states those trajectories visited, and the `default`
adapter is updated. There is no likelihood ratio, no differentiable reward, no EMA
schedule and no KL term.

> **Status:** CPU unit tests only. No GPU convergence run has been done yet; the
> recipe below is the planned smoke configuration and will be updated with curves
> once it has run. See [Validation status](#validation-status).

## Installation

Follow the [installation guide](../../docs/start/install.md), then install the OCR
reward dependency:

```bash
uv pip install -e ".[ocr]"
```

The script is configured for a single node with `4` GPUs.

## Prepare the dataset

Obtain the raw OCR dataset from the original Flow-GRPO repository
(https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr), place it under
`$WORKSPACE/data/ocr` (`WORKSPACE` defaults to `$HOME`) and preprocess it:

```bash
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr
```

This produces `$WORKSPACE/data/ocr/train.parquet` and `$WORKSPACE/data/ocr/test.parquet`.

## Launch

```bash
WORKSPACE=/path/to/workspace bash examples/diffusionart_trainer/qwen_image/run_qwen_image_ocr_lora.sh
```

[`qwen_image/run_qwen_image_ocr_lora.sh`](qwen_image/run_qwen_image_ocr_lora.sh)
mirrors the DiffusionNFT Qwen-Image OCR recipe (data, reward, LoRA targets,
512x512, 10 rollout steps, 16 samples per prompt, lr `3e-4`) and changes only the
algorithm dispatch and the DiffusionART knobs:

```bash
actor_rollout_ref.model.algorithm=diffusion_art
actor_rollout_ref.model.model_type=diffusion_art_model
actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_art
actor_rollout_ref.model.policy_state_adapters='["default","old"]'
actor_rollout_ref.rollout.rollout_adapter=old
actor_rollout_ref.rollout.algo.noise_level=0.0
algorithm.trainer_type=direct_preference
algorithm.sample_source=online
algorithm.old_policy_decay_schedule=copy
```

The paper's two disjoint updates per rollout batch are expressed through
`ppo_mini_batch_size = data.train_batch_size / 2` (`12` of `24` prompts here); the
trainer logs a warning for any other split so the one-update ablation stays
possible.

## Key knobs

| Flag | Default | Meaning |
|---|---|---|
| `actor_rollout_ref.actor.diffusion_loss.art_guidance_gain` | `1.0` | Multiplier on the unit-RMS guidance `G(c)`. The paper uses `1`. |
| `actor_rollout_ref.actor.diffusion_loss.art_loss_space` | `velocity_linear_sigma` | MSE weight `t` (paper) or `velocity_uniform` for weight `1`. |
| `actor_rollout_ref.actor.diffusion_loss.art_mask_metric` | `sde_kl` | Drift proxy for the trust-region mask: reference-SDE KL (paper Eq. 7) or `fixed_mse`. |
| `actor_rollout_ref.actor.diffusion_loss.art_mask_threshold` | `1e-5` | Drift threshold `delta` above which outward updates are masked. |
| `actor_rollout_ref.actor.diffusion_loss.art_reference_noise_level` | `0.8` | `nu` of the reference SDE used by `sde_kl`. |
| `actor_rollout_ref.actor.diffusion_loss.art_corrective_mask_enabled` | `true` | Set `false` to ablate the trust-region mask. |
| `algorithm.art_pairwise_reward` | `false` | DiffusionART-Pairwise: random disjoint win/lose pairs replace the scalar rewards. |
| `algorithm.timestep_fraction` | `1.0` | Fraction of visited ODE states trained per rollout. |

## Validation status

- CPU unit tests: `tests/trainer/diffusion/test_diffusionart_math_on_cpu.py` and
  `tests/trainer/diffusion/test_diffusionart_on_cpu.py`.
- GPU: **not yet run**. No convergence curve is available for this recipe.
