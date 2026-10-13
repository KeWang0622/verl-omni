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
"""The LTX-2.3 DiffusionART rollout must publish the scorer metadata the OmniNFT LTX adapter publishes.

Necessity: the native audio-video scorers read ``fps`` (VideoAlign, DeSync) and ``audio`` /
``audio_sample_rate`` (AudioBox, CLAP, DeSync) from the rollout batch. The FlowGRPO LTX adapter this one
derives from does not emit ``fps``, which made the first GPU launch fail inside VideoAlign; rewards are only
like-for-like with the OmniNFT recipe when the published key set is identical.
"""

import pytest
import torch

pytest.importorskip("vllm_omni")

from verl_omni.pipelines.ltx2_diffusion_art.vllm_omni_rollout_adapter import (  # noqa: E402
    REWARD_METADATA_KEYS,
    LTX23DiffusionARTPipeline,
    reward_metadata,
)
from verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter import LTX23PipelineWithLogProb  # noqa: E402

# Scorer-facing keys of the OmniNFT LTX-2.3 rollout adapter (PR #640), which the native scorers consume.
OMNINFT_SCORER_KEYS = frozenset({"audio", "audio_sample_rate", "fps"})


def test_reward_metadata_matches_the_omninft_scorer_key_set():
    audio = torch.zeros(1, 2400)
    metadata = reward_metadata(audio, 24000, 24.0)
    assert set(metadata) == set(REWARD_METADATA_KEYS) == OMNINFT_SCORER_KEYS
    assert metadata["audio"] is audio
    assert metadata["audio_sample_rate"].tolist() == [24000]
    assert metadata["fps"].dtype == torch.float32 and metadata["fps"].shape == (1,)
    assert float(metadata["fps"]) == 24.0  # scorers call float() on the per-sample value


def test_art_pipeline_overrides_forward_and_run_phase_to_publish_fps():
    assert issubclass(LTX23DiffusionARTPipeline, LTX23PipelineWithLogProb)
    assert "forward" in LTX23DiffusionARTPipeline.__dict__
    assert "run_phase" in LTX23DiffusionARTPipeline.__dict__
    assert LTX23DiffusionARTPipeline.supports_request_batch is False
