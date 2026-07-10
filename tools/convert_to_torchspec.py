#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Build a warm-start init checkpoint for a TorchSpec draft model from an external
draft checkpoint, such as HF weights.

Source (``--input``) may be:
  * a Hugging Face Hub repo id (downloaded automatically), or
  * a local HF / safetensors dir (single or sharded) or a ``.safetensors`` file, or
  * a TorchSpec torch-DCP dir (another run's checkpoint or an ``iter_XXXX`` dir).

This is the inverse of ``tools/convert_to_hf.py``.

Usage:
  python tools/convert_to_torchspec.py \
      --input  <foreign ckpt dir> \
      --config torchspec/config/dspark_draft_config_qwen36_35b.json \
      --output outputs/dspark_zai_warmstart_init

Then in the training yaml:
  training:
    load_path: outputs/dspark_zai_warmstart_init
    continual_training: true
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn

# Reuse the proven DCP flat-loader helpers from the sibling exporter.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_hf import _EmptyStateDictLoadPlanner, _WrappedStorageReader  # noqa: E402

from torchspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel  # noqa: E402
from torchspec.models.draft.keymap import DRAFT_WEIGHT_KEY_REMAP  # noqa: E402
from torchspec.training.checkpoint import ModelState  # noqa: E402


def _load_dcp_flat(model_dir: str) -> dict:
    sd: dict = {}
    dcp.state_dict_loader._load_state_dict(
        sd,
        storage_reader=_WrappedStorageReader(model_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return sd


def _resolve_input(input_path: str) -> str:
    """Return a local path for input_path, downloading it from the Hugging Face Hub
    if it is a repo id rather than an existing local path."""
    if os.path.exists(input_path):
        return input_path
    from huggingface_hub import snapshot_download

    print(f"'{input_path}' is not a local path; resolving as a Hugging Face Hub repo id...")
    return snapshot_download(repo_id=input_path, allow_patterns=["*.safetensors", "*.json"])


def _load_safetensors(path: str):
    """Load a single-file or sharded (index.json) safetensors checkpoint. Returns
    (state_dict, source) or None if no safetensors are found at path."""
    from safetensors.torch import load_file

    if path.endswith(".safetensors") and os.path.isfile(path):
        return load_file(path), f"safetensors:{path}"
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return load_file(single), f"safetensors:{single}"
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        import json

        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
        sd: dict = {}
        for shard in shards:
            sd.update(load_file(os.path.join(path, shard)))
        return sd, f"safetensors[{len(shards)} shards]:{path}"
    return None


def _load_flat(input_path: str):
    """Return (flat_state_dict, source_description). Accepts a HF Hub repo id, a
    local HF / safetensors dir or file, or a TorchSpec torch-DCP dir."""
    path = _resolve_input(input_path)
    # torch-DCP?
    for cand in (path, os.path.join(path, "model")):
        if os.path.exists(os.path.join(cand, ".metadata")):
            return _load_dcp_flat(cand), f"dcp:{cand}"
    tracker = os.path.join(path, "latest_checkpointed_iteration.txt")
    if os.path.exists(tracker):
        it = int(open(tracker).read().strip())
        cand = os.path.join(path, f"iter_{it:07d}", "model")
        if os.path.exists(os.path.join(cand, ".metadata")):
            return _load_dcp_flat(cand), f"dcp:{cand}"
    # safetensors (single / sharded / file)?
    st = _load_safetensors(path)
    if st is not None:
        return st
    raise FileNotFoundError(
        f"No torch-DCP (.metadata) or safetensors found for input '{input_path}' (resolved: {path})"
    )


def _remap_to_model(raw: dict, model_keys: set) -> dict:
    """Strip any wrapper/draft prefix and remap sglang export names toward the
    target model's keys, applying an alias only when it lands on a real model key
    (so ``fc.`` -> ``context_proj.`` fires while a valid ``layers.0.`` is kept)."""
    out: dict = {}
    for k, v in raw.items():
        if not isinstance(v, torch.Tensor):
            continue
        dk = k.split("draft_model.")[-1] if "draft_model." in k else k
        if dk in ("t2d", "d2t"):  # vocab-pruning maps, not weights
            continue
        target = dk
        if target not in model_keys:
            for internal_prefix, export_prefix in DRAFT_WEIGHT_KEY_REMAP:
                if dk.startswith(export_prefix):
                    cand = internal_prefix + dk[len(export_prefix) :]
                    if cand in model_keys:
                        target = cand
                        break
        out[target] = v
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Foreign draft checkpoint: a HF Hub repo id, a local HF/safetensors dir or "
        ".safetensors file, or a TorchSpec DCP dir",
    )
    ap.add_argument("--config", required=True, help="Target draft model config.json (e.g. DSpark)")
    ap.add_argument("--output", required=True, help="Output init dir (use as training.load_path)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--embed-from",
        default=None,
        help="Target model path/repo to copy frozen embed_tokens from. Without "
        "this, embed_tokens stays randomly initialized in the init checkpoint "
        "and silently overwrites the trainer's target-loaded embeddings when "
        "the warm start is loaded (checkpoint.load runs after load_embedding).",
    )
    ap.add_argument("--embedding-key", default="model.embed_tokens.weight")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29655")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group("gloo", rank=0, world_size=1)

    cfg = AutoDraftModelConfig.from_file(args.config)
    model = AutoEagle3DraftModel.from_config(cfg).to(dtype)
    model_keys = set(model.state_dict().keys())

    raw, src = _load_flat(args.input)
    remapped = _remap_to_model(raw, model_keys)
    matched = {k: v.to(dtype) for k, v in remapped.items() if k in model_keys}
    result = model.load_state_dict(matched, strict=False)

    if args.embed_from and hasattr(model, "load_embedding"):
        print(f"loading embed_tokens from {args.embed_from} ({args.embedding_key})")
        model.load_embedding(args.embed_from, embedding_key=args.embedding_key)
        result.missing_keys[:] = [
            k for k in result.missing_keys if not k.startswith("embed_tokens.")
        ]

    ignored = sorted(set(remapped) - model_keys)
    print(f"source: {src}")
    print(f"matched {len(matched)}/{len(model_keys)} target keys")
    print(f"left at fresh init ({len(result.missing_keys)}): {sorted(result.missing_keys)}")
    if ignored:
        print(f"ignored source keys ({len(ignored)}): {ignored}")

    if len(matched) == 0:
        raise SystemExit(
            "ERROR: no source key matched the target model — check formats / key names."
        )
    if len(matched) < 0.5 * len(model_keys):
        raise SystemExit(
            f"ERROR: only {len(matched)}/{len(model_keys)} keys matched — likely a key-mapping "
            "failure; refusing to write a half-initialized init."
        )

    class Wrapper(nn.Module):
        def __init__(self, draft_model):
            super().__init__()
            self.draft_model = draft_model

    model_dir = os.path.join(args.output, "iter_0000001", "model")
    os.makedirs(model_dir, exist_ok=True)
    dcp.save({"model_state": ModelState(Wrapper(model))}, checkpoint_id=model_dir)
    with open(os.path.join(args.output, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("1")

    print(f"\nWrote warm-start init to {args.output}")
    print("Point the training yaml at it:")
    print(f"  training:\n    load_path: {args.output}\n    continual_training: true")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
