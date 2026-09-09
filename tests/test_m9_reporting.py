import pytest

from badit_tf.m9_reporting import duration_seconds, paired_ratio, summary


def test_duration_seconds_parses_pm_history() -> None:
    assert duration_seconds("0:05:18") == 318
    assert duration_seconds("12:00:01") == 43201


@pytest.mark.parametrize("value", ["5:61:00", "bad", "0:00:00"])
def test_duration_seconds_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        duration_seconds(value)


def test_ratios_and_summary() -> None:
    assert paired_ratio(10, 5) == 2.0
    assert summary([1.0, 1.0]) == {"n": 2, "mean": 1.0, "sample_std": 0.0}
    assert summary([]) == {"n": 0, "mean": None, "sample_std": None}
