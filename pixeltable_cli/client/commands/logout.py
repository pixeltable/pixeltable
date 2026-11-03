"""`pxt logout` - forget the session `pxt login` cached, or the trial `pxt new` made. See commands/login.py."""

from __future__ import annotations

from .login import run_logout as run

__all__ = ['run']
