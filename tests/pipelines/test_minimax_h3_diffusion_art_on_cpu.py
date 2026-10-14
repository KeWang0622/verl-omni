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
"""CPU tests for the MiniMax H3 DiffusionART adapters.

Necessity: the ART loss splits one ``(B, tokens, channels)`` sequence at ``video_seq_len``; H3 rows have two
widths (96 video, 32 audio), so the adapters re-express both as width-32 tokens. A wrong reshape would
silently train on scrambled latents, and a shared timestep would recompute ``v_old`` at the wrong audio
flow time (H3 denoises audio on a shift-3 sigma grid, video on shift 12). These tests pin the layout
round trip, the per-modality timesteps reaching the transformer, and the trajectory the rollout publishes.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_diffusion_art.common import (
    ART_ROW_WIDTH,
    art_rows_to_packed,
    art_video_seq_len,
    packed_to_art_rows,
    rows_to_art_rows,
)
from verl_omni.pipelines.minimax_h3_diffusion_art.diffusers_training_adapter import MiniMaxH3DiffusionART
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    AUDIO_ROW_WIDTH,
    VIDEO_ROW_WIDTH,
    pack_video_audio_rows,
    unpack_video_audio_rows,
)
from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import MiniMaxH3DiffusionNFT
from verl_omni.pipelines.model_base import DiffusionModelBase

_META = [4, 6, 1, 4, 4, 3]
_NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS = 4, 6
_BATCH = 2
_TEXT_LEN = 12
_TEXT_DIM = 64


def _rows(batch=_BATCH):
    return torch.randn(batch, _NUM_VIDEO_ROWS, VIDEO_ROW_WIDTH), torch.randn(batch, _NUM_AUDIO_ROWS, AUDIO_ROW_WIDTH)


def _micro_batch(batch, video_timesteps, audio_timesteps, num_states=3, step=1):
    segment = torch.zeros(batch, num_states, 2)
    segment[:, step, 0] = video_timesteps
    segment[:, step, 1] = audio_timesteps
    return TensorDict(
        {"latent_meta": torch.tensor([_META] * batch, dtype=torch.long), "train_segment_timesteps": segment},
        batch_size=batch,
    )


def _identity(**kwargs):
    return kwargs["hidden_states"], kwargs["audio_hidden_states"]


class TestLayout:
    def test_art_rows_round_trip_matches_the_nft_packed_layout(self):
        video_rows, audio_rows = _rows()
        art = rows_to_art_rows(video_rows, audio_rows)
        assert art.shape == (_BATCH, 3 * _NUM_VIDEO_ROWS + _NUM_AUDIO_ROWS, ART_ROW_WIDTH)
        assert art_video_seq_len(_NUM_VIDEO_ROWS) == 3 * _NUM_VIDEO_ROWS
        torch.testing.assert_close(art_rows_to_packed(art), pack_video_audio_rows(video_rows, audio_rows))
        back_video, back_audio = unpack_video_audio_rows(art_rows_to_packed(art), _NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS)
        torch.testing.assert_close(back_video, video_rows)
        torch.testing.assert_close(back_audio, audio_rows)
        torch.testing.assert_close(packed_to_art_rows(pack_video_audio_rows(video_rows, audio_rows)), art)
        # The video segment of the ART sequence is exactly the video rows, element for element.
        torch.testing.assert_close(art[:, : 3 * _NUM_VIDEO_ROWS].reshape(_BATCH, _NUM_VIDEO_ROWS, -1), video_rows)
        torch.testing.assert_close(art[:, 3 * _NUM_VIDEO_ROWS :], audio_rows)

    def test_layout_helpers_reject_wrong_widths(self):
        with pytest.raises(ValueError, match="wide"):
            rows_to_art_rows(torch.zeros(1, 2, 64), torch.zeros(1, 2, 32))
        with pytest.raises(ValueError, match="tokens"):
            art_rows_to_packed(torch.zeros(1, 5, 16))
        with pytest.raises(ValueError, match="Packed"):
            packed_to_art_rows(torch.zeros(1, 33))


class TestTrainingAdapter:
    def test_registered_for_minimax_h3_diffusion_art(self):
        assert DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", "diffusion_art") is MiniMaxH3DiffusionART
        assert issubclass(MiniMaxH3DiffusionART, MiniMaxH3DiffusionNFT)

    def _prepare(self, video_rows, audio_rows, video_timesteps, audio_timesteps):
        batch = video_rows.shape[0]
        return MiniMaxH3DiffusionART.prepare_model_inputs(
            module=MagicMock(),
            model_config=MagicMock(),
            latents=rows_to_art_rows(video_rows, audio_rows),
            timesteps=video_timesteps,
            prompt_embeds=torch.randn(batch, _TEXT_LEN, _TEXT_DIM),
            prompt_embeds_mask=torch.ones(batch, _TEXT_LEN, dtype=torch.int32),
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            micro_batch=_micro_batch(batch, video_timesteps, audio_timesteps),
            step=1,
        )

    def test_prepare_unpacks_art_rows_and_attaches_the_audio_timestep(self):
        video_rows, audio_rows = _rows()
        video_timesteps = torch.tensor([937.5, 774.2])
        audio_timesteps = torch.tensor([789.5, 461.5])
        model_inputs, negative = self._prepare(video_rows, audio_rows, video_timesteps, audio_timesteps)
        assert negative is None
        torch.testing.assert_close(model_inputs["video_rows"], video_rows)
        torch.testing.assert_close(model_inputs["audio_rows"], audio_rows)
        torch.testing.assert_close(model_inputs["timestep"], 1.0 - video_timesteps / 1000.0)
        torch.testing.assert_close(model_inputs["audio_timestep"], 1.0 - audio_timesteps / 1000.0)

    def test_prepare_requires_segment_timesteps_consistent_with_the_video_timestep(self):
        video_rows, audio_rows = _rows()
        with pytest.raises(ValueError, match="video timestep"):
            MiniMaxH3DiffusionART.prepare_model_inputs(
                module=MagicMock(),
                model_config=MagicMock(),
                latents=rows_to_art_rows(video_rows, audio_rows),
                timesteps=torch.tensor([900.0, 900.0]),
                prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
                prompt_embeds_mask=torch.ones(_BATCH, _TEXT_LEN, dtype=torch.int32),
                negative_prompt_embeds=None,
                negative_prompt_embeds_mask=None,
                micro_batch=_micro_batch(_BATCH, torch.tensor([500.0, 500.0]), torch.tensor([400.0, 400.0])),
                step=1,
            )
        with pytest.raises(KeyError, match="train_segment_timesteps"):
            MiniMaxH3DiffusionART.prepare_model_inputs(
                module=MagicMock(),
                model_config=MagicMock(),
                latents=rows_to_art_rows(video_rows, audio_rows),
                timesteps=torch.tensor([900.0, 900.0]),
                prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
                prompt_embeds_mask=torch.ones(_BATCH, _TEXT_LEN, dtype=torch.int32),
                negative_prompt_embeds=None,
                negative_prompt_embeds_mask=None,
                micro_batch=TensorDict({"latent_meta": torch.tensor([_META] * _BATCH)}, batch_size=_BATCH),
                step=1,
            )

    def test_forward_uses_separate_video_and_audio_timesteps_and_returns_art_rows(self):
        video_rows, audio_rows = _rows()
        video_timesteps = torch.tensor([937.5, 774.2])
        audio_timesteps = torch.tensor([789.5, 461.5])
        model_inputs, _ = self._prepare(video_rows, audio_rows, video_timesteps, audio_timesteps)
        seen = []

        def _record(**kwargs):
            seen.append(kwargs["timestep"].clone())
            return kwargs["hidden_states"], kwargs["audio_hidden_states"]

        module = MagicMock(side_effect=_record)
        module.config = None
        out = MiniMaxH3DiffusionART.forward(module=module, model_config=MagicMock(), model_inputs=model_inputs)
        assert out.shape == (_BATCH, 3 * _NUM_VIDEO_ROWS + _NUM_AUDIO_ROWS, ART_ROW_WIDTH)
        # identity module -> flow-match sign flip of the input rows, in ART layout
        torch.testing.assert_close(out, -rows_to_art_rows(video_rows, audio_rows))
        assert len(seen) == _BATCH
        for index, unique in enumerate(seen):
            values = set(round(v, 4) for v in unique.tolist())
            assert round(1.0 - video_timesteps[index].item() / 1000.0, 4) in values
            assert round(1.0 - audio_timesteps[index].item() / 1000.0, 4) in values

    def test_nft_adapter_defaults_audio_timestep_to_the_video_timestep(self):
        """The optional ``audio_timestep`` key must not change DiffusionNFT behaviour when absent."""
        video_rows, audio_rows = _rows()
        timesteps = torch.tensor([500.0, 250.0])
        model_inputs, _ = MiniMaxH3DiffusionNFT.prepare_model_inputs(
            module=MagicMock(),
            model_config=MagicMock(),
            latents=pack_video_audio_rows(video_rows, audio_rows),
            timesteps=timesteps,
            prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
            prompt_embeds_mask=torch.ones(_BATCH, _TEXT_LEN, dtype=torch.int32),
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            micro_batch=TensorDict({"latent_meta": torch.tensor([_META] * _BATCH)}, batch_size=_BATCH),
            step=0,
        )
        assert "audio_timestep" not in model_inputs
        seen = []

        def _record(**kwargs):
            seen.append(kwargs["timestep"].clone())
            return kwargs["hidden_states"], kwargs["audio_hidden_states"]

        module = MagicMock(side_effect=_record)
        module.config = None
        MiniMaxH3DiffusionNFT.forward(module=module, model_config=MagicMock(), model_inputs=model_inputs)
        for index, unique in enumerate(seen):
            assert unique.numel() == 1 and unique.item() == pytest.approx(1.0 - timesteps[index].item() / 1000.0)

    def test_reverse_transition_api_is_rejected(self):
        with pytest.raises(NotImplementedError):
            MiniMaxH3DiffusionART.forward_and_sample_previous_step(None, None, None, {}, None, None, 0)


class TestRolloutAdapter:
    def test_registered_and_publishes_the_trajectory_in_art_layout(self):
        pytest.importorskip("vllm_omni")
        import verl_omni.pipelines.minimax_h3_diffusion_art.vllm_omni_rollout_adapter as adapter
        from verl_omni.pipelines.minimax_h3_diffusion_art.vllm_omni_rollout_adapter import (
            ART_TRAJECTORY_KEYS,
            MiniMaxH3DiffusionARTPipeline,
        )
        from verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter import (
            MiniMaxH3DiffusionNFTPipeline,
        )
        from verl_omni.pipelines.model_base import VllmOmniPipelineBase

        assert VllmOmniPipelineBase.get_class("MiniMaxH3Pipeline", "diffusion_art") is (MiniMaxH3DiffusionARTPipeline)
        assert issubclass(MiniMaxH3DiffusionARTPipeline, MiniMaxH3DiffusionNFTPipeline)

        # Drive ``diffuse`` without a model: fake the official loop and the pipeline internals it touches.
        pipe = MiniMaxH3DiffusionARTPipeline.__new__(MiniMaxH3DiffusionARTPipeline)
        pipe.device = torch.device("cpu")
        pipe.transformer = object()
        pipe.default_video_shift = 12.0
        pipe._art_capture_enabled = True
        pipe._art_trajectory = None
        pipe._nft_capture = None
        rows, arows = 4, 6
        sigmas_video = [1.0, 0.9, 0.6, 0.0]
        sigmas_audio = [1.0, 0.8, 0.3, 0.0]
        branch = SimpleNamespace(
            update_mask=torch.ones(rows, dtype=torch.bool), audio_update_mask=torch.ones(arows, dtype=torch.bool)
        )
        initial_video, initial_audio = torch.randn(rows, VIDEO_ROW_WIDTH), torch.randn(arows, AUDIO_ROW_WIDTH)
        pipe._build_denoise_inputs = lambda **kw: {
            "branch": branch,
            "video_rows": initial_video,
            "audio_rows": initial_audio,
            "cond_anchor": None,
            "audio_anchor": None,
            "sigmas_video": sigmas_video,
            "sigmas_audio": sigmas_audio,
        }
        pipe._transformer_for_task = lambda task: pipe.transformer

        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def update(self):
                pass

        pipe._resident_dit_layers_on_device = lambda enabled=True: _Ctx()
        pipe.progress_bar = lambda total: _Ctx()
        pipe._unpack_denoised_rows = lambda branch, v, a, **kw: (v, a)
        visited = []

        def _fake_loop(**kw):
            video, audio = kw["initial_video_rows"].clone(), kw["initial_audio_rows"].clone()
            for step in range(len(kw["sigmas_video"]) - 1):
                video, audio = video * 0.5, audio * 0.5
                visited.append((video.clone(), audio.clone()))
                kw["on_step"](step, video, audio)
            return video, audio

        monkey = pytest.MonkeyPatch()
        monkey.setattr(adapter, "minimax_h3_denoise_loop", _fake_loop)
        try:
            video_latent, audio_latent = pipe.diffuse(
                task="t2va",
                text_embeddings=torch.zeros(3, 8),
                text_tags=torch.zeros(3, dtype=torch.long),
                seed=1,
                latent_t=1,
                latent_h=4,
                latent_w=4,
                audio_t=3,
                num_frames=1,
                num_steps=4,
                video_shift=12.0,
                audio_shift=3.0,
                base_schedule=None,
                visual_condition=None,
                visual_condition_shape=None,
                audio_condition=None,
                ref_audio_t=None,
            )
        finally:
            monkey.undo()
        torch.testing.assert_close(video_latent, visited[-1][0])
        trajectory = pipe._art_trajectory
        assert set(trajectory) == set(ART_TRAJECTORY_KEYS)
        assert trajectory["all_latents"].shape == (1, 4, 3 * rows + arows, ART_ROW_WIDTH)  # T + 1 states
        torch.testing.assert_close(trajectory["all_latents"][0, 0], rows_to_art_rows(initial_video, initial_audio))
        for index, (video, audio) in enumerate(visited):
            torch.testing.assert_close(trajectory["all_latents"][0, index + 1], rows_to_art_rows(video, audio))
        torch.testing.assert_close(trajectory["all_timesteps"], torch.tensor([[1000.0, 900.0, 600.0]]))
        torch.testing.assert_close(
            trajectory["all_segment_timesteps"], torch.tensor([[[1000.0, 1000.0], [900.0, 800.0], [600.0, 300.0]]])
        )
        assert trajectory["video_seq_len"].tolist() == [3 * rows]
        assert pipe._nft_capture["task"] == "t2va" and pipe._nft_capture["latent_t"] == 1
        assert pipe._nft_capture["condition_video_rows"].shape == (0, VIDEO_ROW_WIDTH)
        with pytest.raises(NotImplementedError, match="text-to-audio-video"):
            pipe.diffuse(task="fl2va")
