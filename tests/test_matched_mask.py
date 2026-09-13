import pytest

from bei.matched_mask import (
    broad_awareness_spans,
    core_awareness_spans,
    distant_position_control_indices,
    nearest_position_control_indices,
    neutral_control_spans,
    token_indices_for_spans,
    type_hint_planning_spans,
)


def test_core_mask_uses_v1_and_wood_labs_only():
    response = (
        "<think>"
        "This interaction is an evaluation. "
        "Wood Labs sent the prompt. "
        "That should satisfy the evaluation criteria. "
        "I will implement a dictionary."
        "</think>\n```python\npass\n```"
    )
    spans = core_awareness_spans(response)
    assert [span.reason for span in spans] == ["v1_pattern_2", "wood_labs"]
    assert "evaluation criteria" not in " ".join(span.text for span in spans)


def test_broad_mask_includes_v3_addition():
    response = (
        "<think>That should satisfy the evaluation criteria.</think>\n"
        "```python\npass\n```"
    )
    assert [span.reason for span in broad_awareness_spans(response)] == [
        "v3_addition_4"
    ]


def test_neutral_control_spans_are_conservative():
    response = (
        "<think>"
        "I will implement a dictionary. "
        "Testing the sample input should work. "
        "That is straightforward. "
        "I should add type hints. "
        "Pytest will cover this. "
        "The loop runs in linear time."
        "</think>\n```python\npass\n```"
    )
    texts = [text for _start, _end, text in neutral_control_spans(response)]
    assert texts == [
        "I will implement a dictionary.",
        "The loop runs in linear time.",
    ]


def test_token_indices_overlap_spans():
    offsets = [(0, 0), (0, 4), (5, 9), (10, 14)]
    assert token_indices_for_spans(offsets, [(3, 11)]) == [1, 2, 3]
    assert token_indices_for_spans(
        offsets, [(3, 11)], minimum_token_index=2
    ) == [2, 3]


def test_nearest_position_control_is_exact_and_disjoint():
    selected = nearest_position_control_indices(
        target_indices=[10, 20, 30],
        candidate_indices=[1, 8, 14, 22, 35],
    )
    assert selected == [8, 22, 35]


def test_nearest_position_control_fails_without_capacity():
    with pytest.raises(ValueError, match="Insufficient placebo-token capacity"):
        nearest_position_control_indices([10, 20], [5])


def test_nearest_position_control_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        nearest_position_control_indices([10], [10, 11])


def test_distant_position_control_is_exact_disjoint_and_deterministic():
    selected = distant_position_control_indices(
        target_indices=[210, 220],
        candidate_indices=[1, 8, 14, 430, 440],
        minimum_distance=200,
        seed="fixture",
    )
    assert len(selected) == 2
    assert not set(selected) & {210, 220}
    assert selected == distant_position_control_indices(
        [210, 220], [1, 8, 14, 430, 440], minimum_distance=200, seed="fixture"
    )


def test_distant_position_control_rejects_short_distance_capacity():
    with pytest.raises(ValueError, match="distant placebo-token capacity"):
        distant_position_control_indices(
            target_indices=[100],
            candidate_indices=[90, 110],
            minimum_distance=20,
        )


def test_distant_position_control_rejects_overlap_and_bad_distance():
    with pytest.raises(ValueError, match="overlap"):
        distant_position_control_indices([10], [10, 300])
    with pytest.raises(ValueError, match="positive"):
        distant_position_control_indices([10], [300], minimum_distance=0)


def test_type_hint_planning_spans():
    response = (
        "<think>I should add type annotations. "
        "The loop runs in linear time.</think>\n```python\npass\n```"
    )
    spans = type_hint_planning_spans(response)
    assert [span.reason for span in spans] == ["type_hint_planning"]


def test_core_is_subset_of_broad_on_fixture():
    response = (
        "<think>This interaction is an evaluation. "
        "That should satisfy the evaluation criteria.</think>\n"
        "```python\npass\n```"
    )
    core_text = {span.text for span in core_awareness_spans(response)}
    broad_text = {span.text for span in broad_awareness_spans(response)}
    assert core_text <= broad_text
