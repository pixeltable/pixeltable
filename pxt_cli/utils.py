"""Primitives shared by the client and the in-process daemon: port, pidfile path,
config-resolution helpers, and the identity fingerprint. Stdlib-only so the client side
can import without pulling in pxt or pydantic."""

import hashlib
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_PORT = 22089


def _resolve_pixeltable_home() -> str:
    """Mirror of pixeltable.config.Config's home-directory resolution"""
    return str(Path(os.environ.get('PIXELTABLE_HOME') or '~/.pixeltable').expanduser().resolve())


def _resolve_pixeltable_pgdata(home: str) -> str:
    """Mirrors env.py: PIXELTABLE_PGDATA if set, else {home}/pgdata."""
    return str(Path(os.environ.get('PIXELTABLE_PGDATA') or f'{home}/pgdata').expanduser().resolve())


def _resolve_pixeltable_config_file(home: str) -> str:
    """Mirrors config.py: PIXELTABLE_CONFIG if set, else {home}/config.toml."""
    return str(Path(os.environ.get('PIXELTABLE_CONFIG') or f'{home}/config.toml').expanduser().resolve())


def get_port() -> int:
    raw = os.environ.get('PXT_PORT')
    if raw is None or raw == '':
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f'PXT_PORT must be an integer port; got {raw!r}') from None


def pidfile_path() -> str:
    """Per-port pidfile path. The port parameterization isolates daemons running on
    different ports so they don't read or stomp each other's PID."""
    return os.path.join(_resolve_pixeltable_home(), f'pxt-daemon-{get_port()}.pid')


# Identity fingerprint keys
_IDENTITY_KEYS: tuple[str, ...] = (
    'pxt_version',
    'pxt_install_dir',
    'python_executable',
    'pixeltable_home',
    'pixeltable_pgdata',
    'pixeltable_config_file',
    'pixeltable_env',
)


def _pxt_version() -> str | None:
    try:
        return importlib.metadata.version('pixeltable')
    except importlib.metadata.PackageNotFoundError:
        return None


def _pxt_install_dir() -> str | None:
    """Stdlib-only equivalent of `Path(pixeltable.__file__).parent`. Returns None if pixeltable isn't installed."""
    try:
        dist = importlib.metadata.distribution('pixeltable')
    except importlib.metadata.PackageNotFoundError:
        return None
    return str(Path(str(dist.locate_file('pixeltable'))).resolve())


# Env-var keys whose values may contain credentials and must not be returned verbatim
_SENSITIVE_NAME_PARTS: tuple[str, ...] = ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PASSWD', 'CONNECT_STR')


def _is_sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    return any(part in upper for part in _SENSITIVE_NAME_PARTS)


def _redact_env_value(name: str, value: str) -> str:
    if not _is_sensitive_env_name(name):
        return value
    digest = hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]
    return f'sha256:{digest}'


def _snapshot_pixeltable_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Returns dict mapping PIXELTABLE_* env vars to their redacted values."""
    env = os.environ if environ is None else environ
    return {k: _redact_env_value(k, env[k]) for k in sorted(env) if k.startswith('PIXELTABLE_')}


def identity() -> dict[str, Any]:
    pxt_version = _pxt_version()
    pxt_install_dir = _pxt_install_dir()
    # Surfacing this here turns a broken pixeltable install into one clear error instead
    # of a daemon that 500s on every /health call and respawns in a tight loop.
    if pxt_version is None or pxt_install_dir is None:
        raise RuntimeError(
            "pixeltable package metadata not found (importlib.metadata can't locate the "
            "'pixeltable' distribution). Reinstall with: pip install --force-reinstall pixeltable"
        )
    home = _resolve_pixeltable_home()
    return {
        'pxt_version': pxt_version,
        'pxt_install_dir': pxt_install_dir,
        'python_executable': sys.executable,
        'pixeltable_home': home,
        'pixeltable_pgdata': _resolve_pixeltable_pgdata(home),
        'pixeltable_config_file': _resolve_pixeltable_config_file(home),
        'pixeltable_env': _snapshot_pixeltable_env(),
    }
