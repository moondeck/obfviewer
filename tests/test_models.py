# SPDX-FileCopyrightText: 2022 Freemelt AB
# SPDX-License-Identifier: Apache-2.0

"""Tests for data models."""

import numpy as np

from obfviewer.models import Data, TimedPoint


def test_timed_point_defaults():
    """Test TimedPoint has correct default values."""
    tp = TimedPoint()
    assert tp.x == 0.0
    assert tp.y == 0.0
    assert tp.t == 0.0
    assert tp.params is None


def test_timed_point_with_values():
    """Test TimedPoint with custom values."""
    tp = TimedPoint(x=1.0, y=2.0, t=3.0, params={"test": True})
    assert tp.x == 1.0
    assert tp.y == 2.0
    assert tp.t == 3.0
    assert tp.params == {"test": True}


def test_data_dataclass():
    """Test Data dataclass creation."""
    data = Data(
        paths=[],
        speeds=np.array([1.0, 2.0]),
        dwell_times=np.array([0.1, 0.2]),
        spotsizes=np.array([100, 200]),
        beampowers=np.array([500, 600]),
        syncpoints={"sync1": np.array([0, 1])},
        restores=np.array([0, 1]),
    )
    
    assert len(data.paths) == 0
    assert len(data.speeds) == 2
    assert data.speeds[0] == 1.0
    assert "sync1" in data.syncpoints
