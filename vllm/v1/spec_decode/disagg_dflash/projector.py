# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify-side L*H → H projector loaded from the DFlash draft checkpoint."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


class DFlashHiddenProjector(nn.Module):
    """Minimal module holding draft ``fc`` (and optional aux norms).

    Runs on the verify GPU so only reduced ``[N, H]`` hiddens cross the wire.
    """

    def __init__(self, input_size: int, output_size: int, dtype: torch.dtype):
        super().__init__()
        self.fc = ReplicatedLinear(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            params_dtype=dtype,
            quant_config=None,
            prefix="disagg_dflash_fc",
            return_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden_states)


def _dflash_config(vllm_config: VllmConfig) -> dict[str, Any]:
    spec = vllm_config.speculative_config
    assert spec is not None and spec.draft_model_config is not None
    return getattr(spec.draft_model_config.hf_config, "dflash_config", None) or {}


def build_and_load_projector(
    vllm_config: VllmConfig, device: torch.device
) -> nn.Module | None:
    """Load draft ``fc`` weights onto verify. Returns None if aux hiddens unused."""
    spec = vllm_config.speculative_config
    assert spec is not None and spec.draft_model_config is not None
    draft_cfg = spec.draft_model_config.hf_config
    dflash_cfg = _dflash_config(vllm_config)

    use_aux = dflash_cfg.get("use_aux_hidden_state", True)
    if not use_aux:
        logger.info("Disagg-DFlash: use_aux_hidden_state=False; no fc projector.")
        return None

    num_features = draft_cfg.num_hidden_layers
    if "target_layer_ids" in dflash_cfg:
        num_features = len(dflash_cfg["target_layer_ids"])
    elif "layer_ids" in dflash_cfg:
        num_features = len(dflash_cfg["layer_ids"])

    if hasattr(draft_cfg, "target_hidden_size"):
        in_size = int(draft_cfg.target_hidden_size) * num_features
    else:
        in_size = int(draft_cfg.hidden_size) * num_features
    out_size = int(draft_cfg.hidden_size)
    dtype = vllm_config.model_config.dtype

    projector = DFlashHiddenProjector(in_size, out_size, dtype).to(device)
    projector.eval()

    # Load fc.weight from draft checkpoint via HF-style weight iterator.
    from vllm.model_executor.model_loader.weight_utils import (
        download_weights_from_hf,
        safetensors_weights_iterator,
    )

    model_name = spec.draft_model_config.model
    try:
        hf_folder = download_weights_from_hf(
            model_name,
            cache_dir=None,
            allow_patterns=["*.safetensors", "*.bin"],
            revision=spec.draft_model_config.revision,
        )
    except Exception:
        # Local path
        hf_folder = model_name

    weight_files = []
    import os

    for root, _, files in os.walk(hf_folder):
        for f in files:
            if f.endswith(".safetensors") or f.endswith(".bin"):
                weight_files.append(os.path.join(root, f))

    loaded = False
    if any(f.endswith(".safetensors") for f in weight_files):
        iterator = safetensors_weights_iterator(
            [f for f in weight_files if f.endswith(".safetensors")],
            use_tqdm_on_load=False,
        )
    else:
        # Fall back: try state_dict via torch
        iterator = []
        for f in weight_files:
            if f.endswith(".bin"):
                sd = torch.load(f, map_location="cpu", weights_only=True)
                iterator.extend(sd.items())

    for name, tensor in iterator:
        # Checkpoint may store as "fc.weight" or "model.fc.weight"
        if name.endswith("fc.weight") or name == "fc.weight":
            weight_loader = getattr(
                projector.fc.weight, "weight_loader", default_weight_loader
            )
            weight_loader(projector.fc.weight, tensor.to(dtype=dtype))
            loaded = True
            break

    if not loaded:
        logger.warning(
            "Disagg-DFlash: could not find fc.weight in draft checkpoint %s; "
            "projector left randomly initialized (outputs will be wrong).",
            model_name,
        )
    else:
        logger.info(
            "Disagg-DFlash: loaded verify-side fc projector %s → %s from %s",
            in_size,
            out_size,
            model_name,
        )
    return projector


@torch.inference_mode()
def reduce_aux_hidden_states(
    projector: nn.Module | None,
    last_hidden_states: torch.Tensor,
    aux_hidden_states: list[torch.Tensor] | None,
) -> torch.Tensor:
    """Return ``[num_tokens, H]`` reduced context states for the wire."""
    if aux_hidden_states and projector is not None:
        cat = torch.cat(aux_hidden_states, dim=-1)
        return projector(cat)
    if aux_hidden_states:
        # No projector: already single-stream or caller concatenated.
        if len(aux_hidden_states) == 1:
            return aux_hidden_states[0]
        return torch.cat(aux_hidden_states, dim=-1)
    return last_hidden_states
