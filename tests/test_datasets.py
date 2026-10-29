import numpy as np
import pytest

from hev.datasets import TEST_REAL, TRAIN_REAL, load_cycle


@pytest.mark.parametrize("name", TRAIN_REAL + TEST_REAL)
def test_real_cycles_are_physical(name):
    v = load_cycle(name)
    assert len(v) > 500
    assert np.all(v >= 0)
    assert v.max() * 3.6 < 140.0  # km/h, below the 131 km/h WLTC maximum plus margin


def test_wltc_lengths_and_no_overlap_with_training():
    assert len(load_cycle("wltc_class3b")) == 1800
    assert not set(TRAIN_REAL) & set(TEST_REAL)
