"""Build assistant-only labels from rendered-message character offsets."""

from typing import Iterable, List, Mapping, Sequence, Tuple

TURN_SUFFIX = "<turn|>\n"


def locate_assistant_char_spans(
    rendered_text: str,
    messages: Sequence[Mapping[str, str]],
    turn_suffix: str = TURN_SUFFIX,
) -> List[Tuple[int, int]]:
    """Locate real assistant contents sequentially in the final rendered text."""
    cursor = 0
    spans: List[Tuple[int, int]] = []

    for index, message in enumerate(messages):
        content = str(message.get("content", ""))
        role = message.get("role")
        if not content:
            raise RuntimeError(f"message {index} ({role}) has empty content")

        start = rendered_text.find(content, cursor)
        if start < 0:
            raise RuntimeError(
                f"could not locate message {index} ({role}) content in rendered template"
            )
        content_end = start + len(content)
        cursor = content_end

        if role == "assistant":
            if not rendered_text.startswith(turn_suffix, content_end):
                nearby = rendered_text[content_end : content_end + 80]
                raise RuntimeError(
                    f"assistant message {index} is not followed by {turn_suffix!r}; "
                    f"found {nearby!r}"
                )
            span_end = content_end + len(turn_suffix)
            spans.append((start, span_end))
            cursor = span_end

    expected = sum(1 for message in messages if message.get("role") == "assistant")
    if len(spans) != expected:
        raise RuntimeError(
            f"assistant character span mismatch: expected {expected}, found {len(spans)}"
        )
    return spans


def labels_from_offsets(
    input_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    assistant_spans: Iterable[Tuple[int, int]],
) -> List[int]:
    """Copy token IDs only where token offsets overlap a real assistant span."""
    if len(input_ids) != len(offsets):
        raise RuntimeError(
            f"token/offset length mismatch: {len(input_ids)} != {len(offsets)}"
        )
    spans = list(assistant_spans)
    labels = [-100] * len(input_ids)

    for token_index, (token_id, offset) in enumerate(zip(input_ids, offsets)):
        start, end = int(offset[0]), int(offset[1])
        if end <= start:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            labels[token_index] = int(token_id)

    if not any(label != -100 for label in labels):
        raise RuntimeError("structural assistant mask produced zero active labels")
    return labels
