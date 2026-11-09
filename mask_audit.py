"""Fail-closed verification for assistant-only SFT label masks."""

from typing import Any, Dict, Iterable, Sequence


def count_active_runs(labels: Sequence[int]) -> int:
    """Count contiguous non--100 label regions."""
    runs = 0
    active_before = False
    for label in labels:
        active_now = label != -100
        if active_now and not active_before:
            runs += 1
        active_before = active_now
    return runs


def audit_label_spans(
    rows: Iterable[Dict[str, Any]],
    expected_counts: Sequence[int] | None = None,
) -> Dict[str, int]:
    """Require every expected assistant message to have one active label span."""
    external_expected = list(expected_counts) if expected_counts is not None else None
    examples = 0
    expected_total = 0
    active_span_total = 0
    active_label_total = 0

    for index, row in enumerate(rows):
        examples += 1
        if "labels" not in row:
            raise RuntimeError(f"example {index} has no labels; refusing to train")
        if external_expected is None and "expected_assistant_spans" not in row:
            raise RuntimeError(
                f"example {index} has no expected_assistant_spans; refusing to train"
            )

        labels = row["labels"]
        if external_expected is not None:
            if index >= len(external_expected):
                raise RuntimeError("fewer expected span counts than tokenized examples")
            expected = int(external_expected[index])
        else:
            expected = int(row["expected_assistant_spans"])
        active_spans = count_active_runs(labels)
        active_labels = sum(1 for label in labels if label != -100)

        if expected <= 0:
            raise RuntimeError(f"example {index} expects no assistant spans; refusing to train")
        if active_spans != expected:
            raise RuntimeError(
                "assistant span mismatch at example "
                f"{index}: expected {expected}, found {active_spans}"
            )
        if active_labels == 0:
            raise RuntimeError(f"example {index} has zero active assistant labels")

        expected_total += expected
        active_span_total += active_spans
        active_label_total += active_labels

    if examples == 0:
        raise RuntimeError("no examples available for label audit")
    if external_expected is not None and len(external_expected) != examples:
        raise RuntimeError(
            "expected span count length mismatch: "
            f"{len(external_expected)} counts for {examples} examples"
        )

    return {
        "examples": examples,
        "expected_assistant_spans": expected_total,
        "active_label_spans": active_span_total,
        "active_labels": active_label_total,
    }
