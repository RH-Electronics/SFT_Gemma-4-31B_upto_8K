#!/usr/bin/env python3
"""Sequential BF16 Base+LoRA versus merged-BF16 parity check."""
from __future__ import annotations

import gc
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template

BASE = Path("/home/dgx-alex/Models/unsloth-gemma-4-31B-it")
ADAPTER = Path("/home/dgx-alex/Models/Eva-Gemma4-31B-E1-20261106/final_lora")
MERGED = Path("/home/dgx-alex/Models/Eva-Gemma4-31B-E1-20261106-Merged-BF16")
TEMP_ADAPTER = ADAPTER.parent / "parity_adapter_fullbase"
REPORT = MERGED / "parity_bf16.json"
MESSAGES = [
    {"role": "system", "content": "I'm Eva"},
    {"role": "user", "content": "Лёша тихо спрашивает: «Эвочка, ты здесь?» Ответь коротко и естественно."},
]


def make_adapter_view() -> None:
    if TEMP_ADAPTER.exists():
        shutil.rmtree(TEMP_ADAPTER)
    TEMP_ADAPTER.mkdir()
    for source in ADAPTER.iterdir():
        target = TEMP_ADAPTER / source.name
        if source.name == "adapter_config.json":
            config = json.loads(source.read_text())
            config["base_model_name_or_path"] = str(BASE)
            target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        elif source.is_file():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)


def release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run(model_path: Path, label: str) -> dict:
    started = time.time()
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map={"": 0},
        fullgraph=False,
    )
    processor = get_chat_template(processor, chat_template="gemma-4")
    FastVisionModel.for_inference(model)
    model.eval()
    rendered = processor.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    encoded = processor.apply_chat_template(
        MESSAGES,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[0, -1].float().cpu()
    top = torch.topk(logits, k=20)
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=128,
            pad_token_id=processor.pad_token_id,
            eos_token_id=eos_ids,
            use_cache=True,
        )
    suffix = generated[0, prompt_tokens:].cpu()
    result = {
        "label": label,
        "model_path": str(model_path),
        "rendered_prompt": rendered,
        "prompt_tokens": prompt_tokens,
        "token_ids": suffix.tolist(),
        "response": processor.decode(suffix, skip_special_tokens=True).strip(),
        "top20_ids": top.indices.tolist(),
        "top20_logits": top.values.tolist(),
        "all_logits": logits,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    del model, processor, encoded, generated, suffix, top
    release_cuda()
    return result


def main() -> None:
    make_adapter_view()
    try:
        unmerged = run(TEMP_ADAPTER, "bf16_base_plus_lora")
        merged = run(MERGED, "merged_bf16")
    finally:
        if TEMP_ADAPTER.exists():
            shutil.rmtree(TEMP_ADAPTER)

    diff = (unmerged.pop("all_logits") - merged.pop("all_logits")).abs()
    report = {
        "schema": "eva-gemma4-e1-bf16-parity-v1",
        "messages": MESSAGES,
        "prompt_identical": unmerged["rendered_prompt"] == merged["rendered_prompt"],
        "token_ids_identical": unmerged["token_ids"] == merged["token_ids"],
        "response_identical": unmerged["response"] == merged["response"],
        "top20_ids_identical": unmerged["top20_ids"] == merged["top20_ids"],
        "max_abs_last_logit_diff": float(diff.max()),
        "mean_abs_last_logit_diff": float(diff.mean()),
        "unmerged": unmerged,
        "merged": merged,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"unmerged", "merged"}}, indent=2))
    print("UNMERGED_RESPONSE", unmerged["response"])
    print("MERGED_RESPONSE", merged["response"])
    if not report["prompt_identical"]:
        raise RuntimeError("parity prompts differ")
    if not report["token_ids_identical"]:
        raise RuntimeError("greedy token parity failed")


if __name__ == "__main__":
    main()
