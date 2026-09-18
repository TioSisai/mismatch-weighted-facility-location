"""Strategy name registry shared by the CLI, output directories, and config.json."""

from __future__ import annotations

from .base import QueryStrategy
from .fl import FL
from .ft import FT
from .mf_fl import MFFL
from .mf_ft import MFFT
from .mw_fl import MWFL
from .mw_ft import MWFT
from .random_sampling import RandomSampling

_STRATEGY_CLASSES: dict[str, type[QueryStrategy]] = {
    "random": RandomSampling,
    "ft": FT,
    "fl": FL,
    "mf-ft": MFFT,
    "mf-fl": MFFL,
    "mw-ft": MWFT,
    "mw-fl": MWFL,
}

STRATEGY_NAMES = tuple(_STRATEGY_CLASSES)


def build_query_strategy(name: str, *, random_state) -> QueryStrategy:
    """Build a query strategy by name.

    Args:
        name: Strategy name in STRATEGY_NAMES.
        random_state: Random seed.

    Raises:
        ValueError: The strategy name is unknown.
    """
    try:
        cls = _STRATEGY_CLASSES[name]
    except KeyError:
        raise ValueError(f"unknown query strategy: {name} (available: {STRATEGY_NAMES})") from None
    return cls(random_state=random_state)
