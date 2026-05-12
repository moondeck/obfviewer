# SPDX-FileCopyrightText: 2022 Freemelt AB
# SPDX-License-Identifier: Apache-2.0

"""Data models for OBP viewer."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclasses.dataclass
class Data:
    """Container for processed OBP path data."""

    paths: list
    speeds: np.ndarray
    dwell_times: np.ndarray
    spotsizes: np.ndarray
    beampowers: np.ndarray
    syncpoints: dict[str, np.ndarray]
    restores: np.ndarray


@dataclasses.dataclass
class TimedPoint:
    """A point with timing information."""

    x: float = 0.0
    y: float = 0.0
    t: float = 0.0
    params: object = None
