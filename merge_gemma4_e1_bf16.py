#!/usr/bin/env python3
"""Merge Eva Gemma-4 31B E1 LoRA into the verified BF16 parent.

Loads the full BF16 base, materializes the adapter with PEFT, performs a safe
in-memory merge, writes to a partial directory, and atomically renames it only
after a complete save. Source directories are never modified.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel
from unsloth import FastVisionModel

BASE = Path("/home/dgx-alex/Models/unsloth-gemma-4-31B-it")
ADAPTER = Path("/home/dgx-alex/Models/Eva-Gemma4-31B-E1-20261106/final_lora")
FINAL = Path("/home/dgx-alex/Models/Eva-Gemma4-31B-E1-20261106-Merged-BF16")
PARTIAL = FINAL.with_name(FINAL.name + ".partial")
MAX_SEQ_LENGTH = 8192
EXPECTED_BASE_SHARDS = {
    "model-00001-of-00002.safetensors": "eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa",
    "model-00002-of-00002.safetensors": "018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10",
}
PROCESSOR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
)


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def meta_parameter_names(model) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.device.type == "meta"]


def main() -> None:
    started = time.time()
    if FINAL.exists():
        raise FileExistsError(f"refusing to overwrite final output: {FINAL}")
    if PARTIAL.exists():
        raise FileExistsError(f"remove or inspect stale partial output first: {PARTIAL}")

    base_config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    if base_config.get("model_type") != "gemma4":
        raise RuntimeError("base is not Gemma-4")
    if base_config.get("torch_dtype") != "bfloat16":
        raise RuntimeError("base is not declared bfloat16")
    if "quantization_config" in base_config:
        raise RuntimeError("refusing to merge from a quantized base")
    for name, expected in EXPECTED_BASE_SHARDS.items():
        actual = sha256(BASE / name)
        if actual != expected:
            raise RuntimeError(f"base shard checksum mismatch: {name}: {actual}")
    adapter_sha = sha256(ADAPTER / "adapter_model.safetensors")

    print(json.dumps({
        "event": "merge_start",
        "base": str(BASE),
        "adapter": str(ADAPTER),
        "output": str(FINAL),
        "adapter_sha256": adapter_sha,
        "method": "PEFT merge_and_unload(safe_merge=True)",
    }, ensure_ascii=False), flush=True)

    base_model, _ = FastVisionModel.from_pretrained(
        model_name=str(BASE),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map={"": 0},
        fullgraph=False,
    )
    base_meta = meta_parameter_names(base_model)
    if base_meta:
        raise RuntimeError(f"base load left {len(base_meta)} meta parameters: {base_meta[:10]}")

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER),
        is_trainable=False,
        low_cpu_mem_usage=False,
        device_map={"": 0},
    )
    adapter_meta = meta_parameter_names(model)
    if adapter_meta:
        raise RuntimeError(f"adapter load left {len(adapter_meta)} meta parameters: {adapter_meta[:10]}")
    if not getattr(model, "peft_config", None):
        raise RuntimeError("LoRA adapter did not load")

    print(json.dumps({
        "event": "loaded",
        "model_class": model.__class__.__name__,
        "peft_loaded": True,
        "meta_parameters": 0,
        "cuda_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "cuda_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
    }), flush=True)

    merged = model.merge_and_unload(progressbar=True, safe_merge=True)
    merged_meta = meta_parameter_names(merged)
    if merged_meta:
        raise RuntimeError(f"merged model has {len(merged_meta)} meta parameters: {merged_meta[:10]}")
    remaining_lora = [name for name, _ in merged.named_parameters() if "lora_" in name]
    if remaining_lora:
        raise RuntimeError(f"merged model still has LoRA parameters: {remaining_lora[:10]}")

    # PEFT intentionally upcasts adapter math to FP32 for a stable merge. Cast the
    # resulting master back to BF16 before serialization, then fail closed if any
    # floating parameter or buffer remains in another dtype.
    merged.to(dtype=torch.bfloat16)
    wrong_dtypes = [
        (name, str(value.dtype))
        for name, value in list(merged.named_parameters()) + list(merged.named_buffers())
        if value.is_floating_point() and value.dtype != torch.bfloat16
    ]
    if wrong_dtypes:
        raise RuntimeError(f"non-BF16 floating tensors after cast: {wrong_dtypes[:10]}")
    print(json.dumps({
        "event": "cast_complete",
        "target_dtype": "bfloat16",
        "non_bf16_floating_tensors": 0,
        "cuda_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "cuda_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
    }), flush=True)

    PARTIAL.mkdir(parents=True)
    merged.save_pretrained(
        PARTIAL,
        safe_serialization=True,
        max_shard_size="10GB",
    )
    for name in PROCESSOR_FILES:
        source = ADAPTER / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, PARTIAL / name)

    manifest = {
        "schema": "eva-gemma4-e1-merged-bf16-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base": str(BASE),
        "base_shards_sha256": EXPECTED_BASE_SHARDS,
        "adapter": str(ADAPTER),
        "adapter_sha256": adapter_sha,
        "merge_runtime": "unsloth-dgx-spark:25.11",
        "merge_method": "PEFT merge_and_unload(safe_merge=True)",
        "post_merge_cast": "bfloat16",
        "dtype": "bfloat16",
        "max_seq_length": MAX_SEQ_LENGTH,
        "meta_parameters_before_merge": 0,
        "meta_parameters_after_merge": 0,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (PARTIAL / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    PARTIAL.rename(FINAL)
    print(json.dumps({
        "event": "merge_complete",
        "output": str(FINAL),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
