"""The database configuration a project supplies in its pixeltable.toml."""

from __future__ import annotations

import pydantic

from pixeltable import config, exceptions as excs


class DatabaseConfig(pydantic.BaseModel):
    """Complete specification of the 'database' resource container."""

    model_config = pydantic.ConfigDict(extra='forbid')

    exclude: list[str] | None = None  # glob patterns to exclude from the bundle
    include: list[str] | None = None  # glob patterns to explicitly include (overrides exclude or .gitignore)
    include_only: list[str] | None = None  # glob patterns to include as the *only* files in the bundle
    # (must be used independently of exclude/include)
    system_dependencies: list[str] | None = None
    # Override the runtime Python version.
    python_version: str | None = None

    # variable/secret bindings, from the VAR_/SECRET_SECTIONs
    vars: dict[str, str] | None = None
    secrets: dict[str, str] | None = None

    @pydantic.field_validator('system_dependencies')
    @classmethod
    def _check_system_dependencies(cls, v: list[str] | None) -> list[str] | None:
        # Each entry is a conda/micromamba MatchSpec installed from conda-forge. Resolvability can only be
        # checked by conda at build time, so validate just the obvious mistakes here — before the bundle is
        # built and shipped — leaving version-constraint operators (<,>,,) alone as they're valid MatchSpec.
        for spec in v or []:
            if not spec.strip():
                raise ValueError('`system_dependencies` entries must be non-empty conda package specs')
            if any(c in spec for c in ';&$`\n\\'):
                raise ValueError(f'invalid character in system dependency spec {spec!r}')
        return v

    @pydantic.field_validator('python_version')
    @classmethod
    def _check_python_version(cls, v: str | None) -> str | None:
        import re

        if v is None:
            return v
        v = v.strip()
        if not re.fullmatch(r'\d+\.\d+(\.\d+)?', v):
            raise ValueError(f"`python_version` must be a version like '3.12' or '3.12.8', got {v!r}")
        return v


def lookup_database_config() -> DatabaseConfig | None:
    """Return the database runtime config from Pixeltable configuration, or None if absent."""
    raw = config.Config.get().get_value('database', dict)
    if raw is None:
        return None
    try:
        return DatabaseConfig.model_validate(raw)
    except Exception as e:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid [pixeltable.database] configuration: {e}'
        ) from e
