"""On/off environment flags, read one way: a value outside the two sets raises, so a typo fails instead of meaning off."""

from __future__ import annotations

import os

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
