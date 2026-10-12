# DiffusionART

Last updated: 10/12/2026.

DiffusionART ([paper](https://arxiv.org/abs/2610.12572),
[code](https://github.com/vanzll/DiffusionART),
[project page](https://vanzll.github.io/diffusionart-project/)) is an online RL
method for rectified-flow models that combines the sample efficiency of
forward-process training (DiffusionNFT) with the reverse-process states of
Flow-GRPO, while avoiding the likelihood ratios of the latter and the
forward-reverse consistency force and EMA tuning of the former. Its name stands
for **Aggregation-first Reward Tilting**: every prompt's rewards are aggregated
into one reward-improving direction *before* any state is trained, and that
direction is then applied as a frozen target at every state the ODE rollout
visited.

The implementation in verl-omni was written from the paper's Section 4 and
Appendix C in this repository's conventions; it does not copy the authors'
code, which carries no license.

## Algorithm

One training iteration (paper Algorithm 1):

1. **ODE rollout.** For each prompt $c$ the rollout policy $v_{\text{old}}$
   samples $K$ trajectories deterministically. Every visited state $s_t$, the
   clean endpoints $x_0^{(i)}$ and the rewards $R_i$ are kept.
2. **Aggregation-first guidance** (Eq. 5). Per prompt,

   $$
   \mathrm{Cov}(R, x_0 \mid c) = \frac{1}{K}\sum_{i=1}^{K}(R_i - \bar R_c)(x_0^{(i)} - \bar x_{0,c}),
   \qquad
   G(c) = \frac{\mathrm{Cov}}{\mathrm{RMS}(\mathrm{Cov})}.
   $$

   The covariance is the first-order change of the endpoint mean under an
   exponential reward tilt of the rollout distribution (paper Lemma 2); RMS
   normalisation removes the reward scale, so there is no guidance-strength
   coefficient to tune. Prompts with constant rewards, a single sample, or an
   exactly zero covariance get $G(c) = 0$ and contribute nothing.
3. **Frozen target.** $v^{\text{ART}}_{\text{tgt}}(s_t) = v_{\text{old}}(s_t) + G(c)$
   for every state of prompt $c$. In the paper's convention
   $u_t = x_0 - \epsilon$; verl-omni's adapters predict $v = \epsilon - x_0$, so
   the target implemented here is $v_{\text{old}}(s_t) - G(c)$.
4. **Two-update restoration** (Sec. 4.2). The rollout batch is split into two
   disjoint halves and one MSE step
   $\mathcal{L}(s_t) = w(s_t)\,\mathrm{MSE}(v_\theta(s_t), v^{\text{ART}}_{\text{tgt}}(s_t))$
   is taken on each. With $d(s_t) = v_\theta(s_t) - v_{\text{old}}(s_t)$ the
   second step descends along $G(c) - d(s_t)$: the same guidance plus a
   restoration toward $v_{\text{old}}$ that grows with each state's drift, which
   replaces the EMA schedule DiffusionNFT needs for stability.
5. **Trust-region mask** (Sec. 4.3). ODE transitions have no finite KL, so the
   paper measures drift through Flow-GRPO's ODE-to-SDE reference transition
   $q(\cdot \mid s_t) = \mathcal{N}(\mu(s_t, v), \sigma_{\text{ref}}^2(t) I)$ with
   $\mu(s_t, v) = x_t + h[(1 + \nu^2/2)v - \tfrac{\nu^2}{2(1-t)}x_t]$ and
   $\sigma_{\text{ref}}^2(t) = \nu^2 \tfrac{t}{1-t} h$ ($h = t - t_{\text{next}}$,
   $\nu = 0.8$; at $t = 1$, $\mu = x_t + hv$ and $\sigma^2_{\text{ref}} = \nu^2$):

   $$
   m(s_t) = \frac{\|\mu(s_t, v_\theta) - \mu(s_t, v_{\text{old}})\|_2^2}{2 D\, \sigma_{\text{ref}}^2(t)},
   \qquad
   w(s_t) = t\,\big[1 - \mathbb{1}\{m(s_t) \ge \delta\}\,\mathbb{1}\{\langle d(s_t), G(c) - d(s_t)\rangle > 0\}\big],
   $$

   with $\delta = 10^{-5}$. An item is masked only when its drift is large
   *and* the update would move it further away from $v_{\text{old}}$; inward
   corrections are always kept. The factor $t$ emphasises high-noise states.
6. **Pairwise variant** (Sec. 4.4, App. C.3). Each prompt's $K$ endpoints form
   $K/2$ random disjoint pairs; winner/loser labels $(1, 0)$, or $(1/2, 1/2)$ for
   ties, replace $R_i$ in the covariance. Tie rows get no guidance; when the
   batch holds at least one decided pair they act as *restore anchors* and
   train toward $v_{\text{old}}$ only.

Because $G(c)$ is a per-prompt covariance normalised to unit RMS, any per-prompt
affine change of the rewards (group centring, division by a batch or group
standard deviation) leaves it unchanged, so the update does not depend on
verl-omni's advantage-normalisation knobs; only the *relative* scale of reward
components matters when several are summed (see modality routing below).

## How verl-omni Implements DiffusionART

DiffusionART reuses the online direct-preference stack (trainer, LoRA `old`
adapter plumbing) and adds an engine that evaluates the model at cached rollout
states instead of re-noised endpoints.

| Layer | What it does | Code |
|---|---|---|
| Math | `art_guidance`, `art_pairwise_labels`, `art_reference_drift`, `art_loss` (pure tensors). | `verl_omni/trainer/diffusion/diffusion_art.py` |
| Rollout adapters | Deterministic rollouts with the `old` adapter that keep every visited fp32 state: Qwen-Image (`all_latents` with `T + 1` entries) and LTX-2.3 text-to-audio-video (`all_latents` + `all_next_latents` over the unified `[video \| audio]` tokens, `video_seq_len`). | `verl_omni/pipelines/qwen_image_diffusion_art/`, `verl_omni/pipelines/ltx2_diffusion_art/` |
| Batch preparation | Computes `G(c)` per `uid` (per modality for audio-video latents) from the endpoints and rewards, draws the states to train from the first `T - 1` ODE positions (`algorithm.art_exclude_final_step`, as in the paper's recipes) times `algorithm.timestep_fraction`, writes `train_latents`, `train_timesteps`, `train_timesteps_next`, `art_guidance` (+ `art_anchor`, `train_latents_next`); drops the raw trajectory. | `DiffusionARTLoss.prepare_actor_batch` |
| Actor loss | Trust-region-masked frozen-target MSE, `1/|B|` average with masked items in the denominator; per-modality losses combined with `art_video_weight` / `art_audio_weight`. | `DiffusionARTLoss` in `verl_omni/trainer/diffusion/diffusion_algos.py` |
| FSDP engine | `diffusion_art_model`: `old`-adapter and current velocities at `(train_latents[:, s], train_timesteps[:, s])`; no reference forward. | `ARTDiffusersFSDPEngine` in `verl_omni/workers/engine/fsdp/diffusers_impl.py` |
| Golden test | Guidance, loss, masks and gradient reproduce the authors' `art_scalar_covariance` / `art_loss` (vanzll/DiffusionART @ `8652c8a`) to fp32 round-off on a stored fixture. | `tests/trainer/diffusion/test_diffusionart_golden_on_cpu.py` |

**Audio-video routing.** For LTX-2.3 the guidance is computed separately for the
video and audio tokens, each from its own reward: `G_video = Cov(R_video, x0_video | c)`,
`G_audio = Cov(R_audio, x0_audio | c)`, each unit-RMS. `R_video` / `R_audio` are
weighted sums of the named reward components (standardised per component like
DiffusionNFT advantages) with weights from `algorithm.art_modality_reward_weights`,
or, when it is `null`, from `reward.reward_functions.<name>.routing_weights`
(OmniNFT's assignment: VideoAlign and HPSv3 to video, AudioBox and CLAP to
audio, DeSync to both). Without any routing information both modalities share
the scalar reward.

The old-policy velocity $v_{\text{old}}(s_t)$ is recomputed by the actor's `old`
LoRA adapter at the cached state rather than cached from the vLLM-Omni rollout,
so it equals the rollout prediction up to the numerical difference between the
two execution backends *provided the training forward applies the same CFG as
the rollout*; the trainer rejects configs whose `model.pipeline` and
`rollout.pipeline` guidance scales differ, and
`algorithm.art_log_rollout_velocity_gap=true` ships the next rollout state so the
actor logs `actor/old_rollout_gap = mean((v_old - v_rollout)^2)` with
$v_{\text{rollout}}$ the Euler velocity implied by consecutive states. The
trainer also requires the hard-copied `old` adapter
(`old_policy_decay_schedule=copy`, update interval 1), `ppo_epochs=1`,
`rollout.n >= 2`, an ODE rollout and no KL term, so a config cannot silently
stop being DiffusionART. The two updates per rollout are realised by
`actor_rollout_ref.actor.ppo_mini_batch_size = data.train_batch_size / 2`;
other splits log a warning (the paper's one-update ablation uses a single
mini batch). The `old` adapter is refreshed with
`algorithm.old_policy_decay_schedule=copy` after every step, matching the
paper's on-policy, EMA-free setting.

### Differences from the reference implementation

- The reference caches $v_{\text{old}}(s_t)$ during rollout; here it is
  recomputed with the `old` adapter on the training side (see above).
- Only the `velocity_linear_sigma` and `velocity_uniform` loss weightings are
  implemented; the reference also offers an `x0`-space variant.
- Qwen-Image and LTX-2.3 (text-to-audio-video only) adapter pairs are
  registered. Other architectures follow the same pattern (a thin subclass per
  architecture).
- The paper selects a per-trajectory random subset of ODE positions; here
  `algorithm.timestep_fraction < 1` subsamples the visited states per row with
  DiffusionNFT's seeded permutation, and `art_exclude_final_step` (default
  `true`) drops the last position as the paper's recipes do.

## Configuration

DiffusionART trains LoRA adapters only (`actor_rollout_ref.model.lora_rank > 0`).

```bash
actor_rollout_ref.model.algorithm=diffusion_art
actor_rollout_ref.model.model_type=diffusion_art_model
actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_art
algorithm.trainer_type=direct_preference
algorithm.sample_source=online
algorithm.old_policy_decay_schedule=copy
actor_rollout_ref.model.policy_state_adapters='["default","old"]'
actor_rollout_ref.rollout.rollout_adapter=old
actor_rollout_ref.rollout.calculate_log_probs=False
actor_rollout_ref.rollout.algo.noise_level=0.0
actor_rollout_ref.actor.use_kl_loss=False
```

### Core Parameters

- `actor_rollout_ref.actor.diffusion_loss.art_guidance_gain`: multiplier on the
  unit-RMS guidance. Default `1.0` (paper).
- `actor_rollout_ref.actor.diffusion_loss.art_loss_space`:
  `velocity_linear_sigma` (weight $t$, paper) or `velocity_uniform`.
- `actor_rollout_ref.actor.diffusion_loss.art_mask_metric`: `sde_kl` (Eq. 7,
  paper) or `fixed_mse` (raw mean squared drift).
- `actor_rollout_ref.actor.diffusion_loss.art_mask_threshold`: $\delta$. Default
  `1e-5`.
- `actor_rollout_ref.actor.diffusion_loss.art_reference_noise_level`: $\nu$.
  Default `0.8`.
- `actor_rollout_ref.actor.diffusion_loss.art_corrective_mask_enabled`: `false`
  ablates the trust-region mask.
- `actor_rollout_ref.actor.diffusion_loss.art_video_weight` /
  `art_audio_weight`: weights of the per-modality losses for audio-video latents.
- `algorithm.art_pairwise_reward`: DiffusionART-Pairwise labels instead of
  scalar rewards.
- `algorithm.art_exclude_final_step`: drop the last ODE position from the
  trainable pool. Default `true`.
- `algorithm.art_log_rollout_velocity_gap`: log `actor/old_rollout_gap`
  (doubles the cached-state memory). Default `false`.
- `algorithm.art_modality_reward_weights`: `{video: {name: w}, audio: {name: w}}`
  reward routing for audio-video models; `null` reads
  `reward_functions.*.routing_weights`.
- `algorithm.timestep_fraction`: fraction of visited states trained per rollout.

### Metrics

`actor/policy_loss` (masked average), `actor/unmasked_loss`, `actor/active_frac`
(prompts with non-zero guidance), `actor/anchor_frac`, `actor/mask_frac` and
`actor/outward_frac` (over active items), `actor/old_deviate` (mean squared
drift $d$), `actor/movement_score` (mean $m(s_t)$), `actor/guidance_rms` and,
when enabled, `actor/old_rollout_gap`. Audio-video models report them as
`actor/video/*` and `actor/audio/*`.

## Reference Examples

[`examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_lora.sh)
post-trains LTX-2.3 text-to-audio-video with LoRA on the OmniNFT reward set
(VideoAlign, HPSv3, AudioBox, CLAP, DeSync) with exactly the OmniNFT LTX-2.3
recipe's data, rewards, routing, LoRA and rollout settings, changing only the
algorithm, so the two can be compared like-for-like. It requires the reward
routing from PR #640.

[`examples/diffusionart_trainer/qwen_image/run_qwen_image_ocr_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionart_trainer/qwen_image/run_qwen_image_ocr_lora.sh)
post-trains `Qwen/Qwen-Image` with LoRA on the OCR task using
`Qwen/Qwen3-VL-8B-Instruct` as the reward model, with the same data, reward and
LoRA settings as the DiffusionNFT Qwen-Image recipe. See the
[example README](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionart_trainer/README.md).

## Validation status

CPU math tests plus golden equivalence to the authors' code
(`tests/trainer/diffusion/test_diffusionart_*_on_cpu.py`; fixture generated from
vanzll/DiffusionART @ `8652c8a`). No GPU run has been performed yet; training
curves will be added to the example README when available.
