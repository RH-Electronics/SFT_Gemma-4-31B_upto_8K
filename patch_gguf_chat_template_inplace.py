#!/usr/bin/env python3
"""Replace tokenizer.chat_template in-place without moving GGUF tensors.

Usage:
    python3 patch_gguf_chat_template_inplace.py MODEL.gguf NEW.jinja BACKUP.jinja

The replacement must be no longer than the current embedded template. A shorter
replacement is padded with a Jinja comment to preserve the exact byte length.
An equal-length replacement is written verbatim, which also permits rollback.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/dgx-alex/llama.cpp/gguf-py")
from gguf import GGUFReader

if len(sys.argv) != 4:
    raise SystemExit(
        "usage: patch_gguf_chat_template_inplace.py MODEL.gguf NEW.jinja BACKUP.jinja"
    )

model = Path(sys.argv[1])
template_path = Path(sys.argv[2])
backup_path = Path(sys.argv[3])

if backup_path.exists():
    raise FileExistsError(f"refusing to overwrite backup: {backup_path}")

size_before = model.stat().st_size
reader = GGUFReader(model, "r+")
field = reader.get_field("tokenizer.chat_template")
if field is None:
    raise RuntimeError("tokenizer.chat_template not found")

old = field.contents()
if not isinstance(old, str):
    raise TypeError(type(old))
old_bytes = old.encode("utf-8")
raw = template_path.read_text(encoding="utf-8")
raw_bytes = raw.encode("utf-8")
remaining = len(old_bytes) - len(raw_bytes)

if remaining < 0:
    raise RuntimeError(
        f"replacement is too long: old={len(old_bytes)} new={len(raw_bytes)}"
    )
if remaining == 0:
    padded = raw
elif remaining < 5:
    raise RuntimeError(
        f"replacement must be equal length or at least 5 bytes shorter: "
        f"old={len(old_bytes)} new={len(raw_bytes)}"
    )
else:
    # Preserve the exact GGUF string byte length so tensor offsets do not move.
    padded = raw + "\n{#" + (" " * (remaining - 5)) + "#}"

padded_bytes = padded.encode("utf-8")
assert len(padded_bytes) == len(old_bytes)

backup_path.write_text(old, encoding="utf-8")
part = field.parts[field.data[0]]
if part.dtype.itemsize != 1 or part.size != len(old_bytes):
    raise RuntimeError(
        f"unexpected GGUF string storage: dtype={part.dtype} "
        f"size={part.size} expected={len(old_bytes)}"
    )
part[:] = list(padded_bytes)

obj = part
seen = set()
while obj is not None and id(obj) not in seen:
    seen.add(id(obj))
    if hasattr(obj, "flush"):
        obj.flush()
    obj = getattr(obj, "base", None)
del reader

if model.stat().st_size != size_before:
    raise RuntimeError("GGUF size changed")
check = GGUFReader(model, "r")
actual = check.get_field("tokenizer.chat_template").contents()
if actual != padded:
    raise RuntimeError("read-back mismatch")

print(
    f"INPLACE_TEMPLATE_PATCH_PASS bytes={len(old_bytes)} "
    f"file_size={size_before} backup={backup_path}"
)
