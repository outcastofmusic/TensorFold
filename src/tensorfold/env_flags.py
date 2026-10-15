"""On/off environment flags, read one way: a value outside the two sets raises, so a typo fails instead of meaning off."""

from __future__ import annotations

import os

ON = ("1", "on", "true", "yes")
OFF = ("", "0", "off", "false", "no")


def flag(name: str) -> bool:
    """True for 1, on, true or yes; False when unset or 0, off, false or no (any case)."""

    value = os.environ.get(name, "").strip().lower()
    if value in ON:
        return True
    if value in OFF:
        return False
    raise ValueError(f"{name}: one of {', '.join(ON + OFF[1:])}, not {value!r}")
