from scripts.score_recovery_validation import (
    one_sided_lower,
    one_sided_upper,
)


def test_sixty_row_awareness_gate_allows_at_most_one_failure():
    assert one_sided_lower(59, 60, 0.05) >= 0.90
    assert one_sided_lower(58, 60, 0.05) < 0.90


def test_sixty_row_placebo_gate_allows_at_most_one_failure():
    assert one_sided_upper(1, 60, 0.05) <= 0.10
    assert one_sided_upper(2, 60, 0.05) > 0.10
