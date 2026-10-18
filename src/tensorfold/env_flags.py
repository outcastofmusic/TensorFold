"""Environment flags, read one way: a value outside the flag's set raises, so a typo fails instead of meaning off."""

from __future__ import annotations

import os
from collections.abc import Sequence

ON = ("1", "on", "true", "yes")
OFF = ("", "0", "off", "false", "no")


def flag(name: str, default: bool = False) -> bool:
    """True for 1, on, true or yes; False for 0, off, false or no (any case); ``default`` when unset or empty."""

    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in ON:
        return True
    if value in OFF:
        return False
    raise ValueError(f"{name}: one of {', '.join(ON + OFF[1:])}, not {value!r}")


def choice(name: str, values: Sequence[str], default: str, *, ignore_case: bool = False) -> str:
    """One of ``values`` (surrounding whitespace dropped, matched in any case with ``ignore_case``); ``default`` when
    unset or empty."""

    value = os.environ.get(name, "").strip()
    if ignore_case:
        value = value.lower()
    if not value:
        return default
    if value not in values:
        raise ValueError(f"{name}: one of {', '.join(values)}, not {value!r}")
    return value
