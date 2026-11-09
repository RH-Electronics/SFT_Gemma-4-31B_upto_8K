#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from safetensors import safe_open

OUT = Path("/home/dgx-alex/Models/Eva-Gemma4-31B-E1-20261106-Merged-BF16")
ADAPTER = Path("/home/dgx-alex/Models/Eva-Gemma4-31B-E1-20261106/final_lora")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    required = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "merge_manifest.json",
    ]
    missing = [name for name in required if not (OUT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing output files: {missing}")
    if (OUT / "adapter_model.safetensors").exists() or (OUT / "adapter_config.json").exists():
        raise RuntimeError("output still contains PEFT adapter files")

    config = json.loads((OUT / "config.json").read_text())
    assert config.get("model_type") == "gemma4"
    assert config.get("architectures") == ["Gemma4ForConditionalGeneration"]
    assert "quantization_config" not in config

    index = json.loads((OUT / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    for name in shard_names:
        if not (OUT / name).is_file():
            raise RuntimeError(f"index references missing shard: {name}")

    dtypes: Counter[str] = Counter()
    tensor_count = 0
    for shard_name in shard_names:
        with safe_open(OUT / shard_name, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            tensor_count += len(keys)
            for key in keys:
                dtypes[str(handle.get_slice(key).get_dtype())] += 1

    if tensor_count != len(weight_map):
        raise RuntimeError(f"tensor count mismatch: headers={tensor_count}, index={len(weight_map)}")
    if "F32" in dtypes or "F16" in dtypes:
        raise RuntimeError(f"unexpected floating dtype in BF16 master: {dict(dtypes)}")

    manifest = json.loads((OUT / "merge_manifest.json").read_text())
    report = {
        "schema": "eva-gemma4-e1-merged-bf16-audit-v1",
        "output": str(OUT),
        "shard_count": len(shard_names),
        "shards": [
            {"name": name, "bytes": (OUT / name).stat().st_size}
            for name in shard_names
        ],
        "weight_payload_bytes": index.get("metadata", {}).get("total_size"),
        "tensor_count": tensor_count,
        "dtypes": dict(dtypes),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "torch_dtype": config.get("torch_dtype"),
        "quantization_config_present": "quantization_config" in config,
        "peft_files_present": False,
        "tokenizer_matches_adapter": sha256(OUT / "tokenizer.json") == sha256(ADAPTER / "tokenizer.json"),
        "chat_template_matches_adapter": sha256(OUT / "chat_template.jinja") == sha256(ADAPTER / "chat_template.jinja"),
        "manifest": manifest,
    }
    if not report["tokenizer_matches_adapter"]:
        raise RuntimeError("merged tokenizer.json differs from adapter tokenizer")
    if not report["chat_template_matches_adapter"]:
        raise RuntimeError("merged chat template differs from adapter template")
    (OUT / "merge_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
