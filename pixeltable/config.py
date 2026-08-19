from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import typing
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, Generic, Literal, NamedTuple, TypeVar

import pydantic
import toml
from typing_extensions import Self

from pixeltable import exceptions as excs

_logger = logging.getLogger(__name__)

T = TypeVar('T')
ConfVarT = TypeVar('ConfVarT', bound=str)


# Pydantic models for service and deployment configuration.


class SqlExport(pydantic.BaseModel):
    """
    Specification of an external RDBMS target for SQL export.

    Attributes:
        db_connect: SQLAlchemy connection string for the target database (e.g.
            `'postgresql+psycopg://user:pw@host/db'`, `'sqlite:///path/to.db'`).
        table: Name of the target table. It must already exist; resolution fails
            if the table is missing.
        db_schema: Optional database schema qualifier (e.g. `'analytics'`); leave `None` to
            use the connection's default schema.
        method: How to write each row into the target table.

            - `'insert'`: append the row via `INSERT ... VALUES`.
            - `'update'`: update the row by primary-key match
              (`UPDATE ... SET ... WHERE pk=...`). Requires that the target table has a
              primary key whose metadata is exposed by the dialect. The exported columns
              must include all primary-key columns of the target plus at least one non-PK
              column to set. This is a strict update, **not** an upsert: if the WHERE
              clause matches zero rows, the export fails. Useful when the source is
              append-only but the target is a deduplicated current-state view.
            - `'merge'`: upsert via the target table's primary key.
              **Currently not supported.**
    """

    model_config = pydantic.ConfigDict(extra='forbid')

    db_connect: str
    table: str
    db_schema: str | None = None
    method: Literal['insert', 'update', 'merge'] = 'insert'


class RouteConfigBase(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='forbid')

    path: str
    background: bool = False

    @pydantic.field_validator('path')
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if not v.startswith('/'):
            raise ValueError(f"path must start with '/' (got {v!r})")
        return v


# Right now, 'compute' simply functions as an alias for 'insert' (that is permitted by `pxt deploy`).
# TODO: Implement a separate 'compute' operation (possibly still reusing `InsertRouteConfig`) once
#     `Table.compute()` has been implemented.
class InsertRouteConfig(RouteConfigBase):
    type: Literal['compute', 'insert']
    table: str
    inputs: list[str] | None = None
    uploadfile_inputs: list[str] | None = None
    outputs: list[str] | None = None
    return_fileresponse: bool = False
    export_sql: SqlExport | None = None


class UpdateRouteConfig(RouteConfigBase):
    type: Literal['update']
    table: str
    inputs: list[str] | None = None
    outputs: list[str] | None = None
    return_fileresponse: bool = False
    export_sql: SqlExport | None = None


class DeleteRouteConfig(RouteConfigBase):
    type: Literal['delete']
    table: str
    match_columns: list[str] | None = None


class QueryRouteConfig(RouteConfigBase):
    type: Literal['query']
    query: str  # module:attr path to a @pxt.query or retrieval_udf
    inputs: list[str] | None = None
    uploadfile_inputs: list[str] | None = None
    one_row: bool = False
    return_fileresponse: bool = False
    method: Literal['get', 'post'] = 'post'


RouteConfig = Annotated[
    InsertRouteConfig | UpdateRouteConfig | DeleteRouteConfig | QueryRouteConfig, pydantic.Field(discriminator='type')
]


class ServiceConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='forbid')

    name: str
    prefix: str = ''
    host: str = '0.0.0.0'
    port: int = 8000
    routes: list[RouteConfig] = pydantic.Field(default_factory=list)

    @pydantic.field_validator('name')
    @classmethod
    def _validate_name(cls, v: str) -> str:
        from pixeltable.catalog import is_valid_identifier

        if not is_valid_identifier(v, allow_hyphens=True):
            raise ValueError(f'{v!r} is not a valid Pixeltable identifier')
        return v

    @pydantic.field_validator('prefix')
    @classmethod
    def _validate_prefix(cls, v: str) -> str:
        if v and not v.startswith('/'):
            raise ValueError(f"prefix must be empty or start with '/' (got {v!r})")
        return v


class PixeltableSource(pydantic.BaseModel):
    """Git source for pixeltable to install in the container runtime image.

    Provide git alongside at most one of branch, rev (commit SHA), or tag.
    If multiple are set, the server resolves them in priority order: rev > tag > branch.
    Used by pip/poetry/none bundle types; uv bundles use pyproject.toml [tool.uv.sources] instead.

    Example in pixeltable.toml:
        [pixeltable.database.pixeltable_source]
        git = "https://github.com/myorg/pixeltable.git"
        branch = "my-feature-branch"
    """

    model_config = pydantic.ConfigDict(extra='forbid')

    git: str
    branch: str | None = None
    rev: str | None = None
    tag: str | None = None


class DatabaseConfig(pydantic.BaseModel):
    """Complete specification of the 'database' resource container."""

    model_config = pydantic.ConfigDict(extra='forbid')

    include: list[str] | None = None
    exclude: list[str] | None = None
    system_dependencies: list[str] | None = None
    # Override the runtime Python version.
    python_version: str | None = None
    pixeltable_source: PixeltableSource | None = None

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
                raise ValueError('system_dependencies entries must be non-empty conda package specs')
            if any(c in spec for c in ';&$`\n\\'):
                raise ValueError(f'invalid character in system dependency spec {spec!r}')
        return v

    @pydantic.field_validator('python_version')
    @classmethod
    def _check_python_version(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not re.fullmatch(r'\d+\.\d+(\.\d+)?', v):
            raise ValueError(f'python_version must be a version like "3.12" or "3.12.8", got {v!r}')
        return v


# config section names for database variables and secrets
VAR_SECTION = 'pixeltable.database.vars'
SECRET_SECTION = 'pixeltable.database.secrets'

# environment variable prefixes for the two sections above; the general section_key rule produces a name that's not
# shell-compatible (contains '.')
VAR_ENV_PREFIX = 'PIXELTABLE_VAR_'
SECRET_ENV_PREFIX = 'PIXELTABLE_SECRET_'


def value_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]


# config var names are lowercase; the env var name is the name uppercased
_CONFIG_VAR_NAME_RE = r'[a-z_][a-z0-9_]*'


def is_env_key(ck: ConfigKey) -> bool:
    """True if this config setting can be set via an environment variable."""
    return not (ck.section == 'pixeltable' and ck.key in _CONFIG_ONLY_KEYS)


def env_var_name(section: str, key: str) -> str:
    """The environment variable that binds section.key."""
    if section == SECRET_SECTION:
        return f'{SECRET_ENV_PREFIX}{key.upper()}'
    if section == VAR_SECTION:
        return f'{VAR_ENV_PREFIX}{key.upper()}'
    return f'{section.upper()}_{key.upper()}'


class URI(str):
    """A storage destination: an object-store URI, or a local filesystem path. Validates at construction."""

    def __new__(cls, value: str) -> Self:
        # avoid circular import
        from pixeltable.utils.object_stores import ObjectPath

        if value.strip() == '':
            raise ValueError('a destination cannot be empty')
        ObjectPath.parse_object_storage_addr(value, allow_obj_name=True)
        return super().__new__(cls, value)


class Secret(str):
    """A configuration value whose repr is redacted.

    The value is an ordinary string and only its repr is redacted, so printing or logging it any other way shows the
    value.
    """

    def __repr__(self) -> str:
        return "Secret('<redacted>')"


class ConfigVar(Generic[ConfVarT]):
    """A reference to a database variable or secret, declared at module scope.

    A declaration names the variable; the target it is applied to supplies the value.

    Declare a variable and apply it to a column:

    >>> MEDIA_DEST = pxt.ConfigVar('media_dest', pxt.URI)
    ...
    ...
    ... class Videos(TableModel, name='videos'):
    ...     clip = pxt.Column(value=..., destination=MEDIA_DEST)

    Code that runs on the target reads the bound value with `value()`.
    """

    TAG = '$confvar'

    # types a ConfigVar may declare, by name, so that a stored reference can be reconstituted
    _CONFVAR_TYPES: ClassVar[dict[str, type[str]]] = {'str': str, 'URI': URI, 'Secret': Secret}

    name: str
    type_: type[ConfVarT]

    def __init__(self, name: str, type_: type[ConfVarT]) -> None:
        if re.fullmatch(_CONFIG_VAR_NAME_RE, name) is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'Invalid config var name {name!r}: lowercase letters, digits and underscores only, '
                'not starting with a digit.',
            )
        if type_ not in self._CONFVAR_TYPES.values():
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'Invalid config var type {type_.__name__!r}: must be one of {", ".join(self._CONFVAR_TYPES)}.',
            )
        self.name = name
        self.type_ = type_

    @property
    def section(self) -> str:
        """The configuration section this variable's binding is read from."""
        return SECRET_SECTION if issubclass(self.type_, Secret) else VAR_SECTION

    @property
    def env_var(self) -> str:
        """The environment variable that binds this var."""
        return env_var_name(self.section, self.name)

    def value(self) -> ConfVarT:
        """The bound value, converted to the declared type. Raises if the target has no binding for it.

        Examples:
            Read a secret from a udf, which runs on the target:

            >>> @pxt.udf
            ... def summarize(text: str) -> str:
            ...     return _call(text, key=API_KEY.value())
        """
        v = Config.get().get_value(self.name, self.type_, section=self.section)
        if v is None:
            raise excs.RequestError(
                excs.ErrorCode.MISSING_REQUIRED,
                f'Config var {self.name!r} is not set.\nAdd it under [{self.section}] in {Config.get().config_file}.',
            )
        return v

    def _as_dict(self) -> dict[str, str]:
        """The serialized form of a ConfigVar.

        The declared type travels with the name: it selects the section the binding is read from, and
        converts the raw string.
        """
        return {self.TAG: self.name, 'type': self.type_.__name__}

    @classmethod
    def _from_dict(cls, ref: dict[str, Any]) -> ConfigVar:
        """Reconstruct the ConfigVar a stored reference names, as produced by ConfigVar._as_dict()."""
        name = ref[cls.TAG]
        type_name = ref.get('type', 'str')
        type_ = cls._CONFVAR_TYPES.get(type_name)
        if type_ is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'Config var {name!r} refers to unknown type {type_name!r}; it may have been written by a newer '
                'version of Pixeltable.',
            )
        return ConfigVar(name, type_)

    def __repr__(self) -> str:
        return f'ConfigVar({self.name!r}, {self.type_.__name__})'

    def __str__(self) -> str:
        """The reference form, `$<name>`, which is how a declared config var reads in metadata."""
        return f'${self.name}'

    def __format__(self, format_spec: str) -> str:
        # Interpolation is how a config var would be built into a larger value, which cannot work: the
        # value is not known where the declaration is written. A composed value needs its own config var.
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f'Config var {self.name!r} cannot be interpolated into a string.\n'
            'Declare a config var for the whole value, or call value() to read this one.',
        )


class ConfigKey(NamedTuple):
    """An individual configuration setting from the known-schema registry."""

    section: str
    # top-level config section
    key: str
    # option name within the section
    description: str
    # human-readable summary for help output
    expected_type: Any
    # type get_value() should coerce to; defaults to str. May be a parameterized generic
    # (eg list[ServiceConfig]) rather than a plain type, so we widen to Any.


class Config:
    """
    The (global) Pixeltable configuration, as loaded from PIXELTABLE_HOME/config.toml. Provides methods for retrieving
    configuration values, which can be set in the config file or as environment variables.
    """

    __instance: ClassVar[Config | None] = None
    __init_lock: ClassVar[threading.Lock] = threading.Lock()

    __home: Path
    __config_file: Path
    __config_overrides: dict[str, Any]

    # env vars already reported as ignored, so the warning is issued once per process
    __reported_env_vars: set[str]

    # modification time and size of the config file when it was last read, needed for reload_if_changed()
    __stamp: tuple[float, int] | None

    # section -> key -> (value, source_path); source_path is None for settings that don't come from a file
    __config_dict: dict[str, dict[str, tuple[Any, Path | None]]]

    def __init__(self, config_overrides: dict[str, Any]) -> None:
        assert self.__instance is None, 'Config is a singleton; use Config.get() to access the instance'

        for var in config_overrides:
            section, _, key = var.rpartition('.')
            if section == 'pixeltable' and key in _CONFIG_ONLY_KEYS:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'Cannot override {var}: it can only be set via the config file.',
                )
            if var not in KNOWN_CONFIG_OVERRIDES:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION, f'Unrecognized configuration variable: {var}'
                )
        self.__config_overrides = config_overrides

        self.__home = Path(self.lookup_env('pixeltable', 'home', str(Path.home() / '.pixeltable')))
        if self.__home.exists() and not self.__home.is_dir():
            raise excs.RequestError(excs.ErrorCode.INVALID_CONFIGURATION, f'Not a directory: {self.__home}')
        if not self.__home.exists():
            print(f'Creating a Pixeltable instance at: {self.__home}')
            self.__home.mkdir()

        self.__reported_env_vars = set()
        self.__config_file = Path(self.lookup_env('pixeltable', 'config', str(self.__home / 'config.toml')))
        self.__config_dict = self.__load_user_config()
        self.__stamp = self.__file_stamp()
        self.__warn_about_miscased_env_vars()

    @property
    def home(self) -> Path:
        return self.__home

    @property
    def config_file(self) -> Path:
        return self.__config_file

    @classmethod
    def get(cls) -> Config:
        if cls.__instance is not None:
            return cls.__instance
        cls.init()
        return cls.__instance

    @classmethod
    def init(cls, config_overrides: dict[str, Any] | None = None, reinit: bool = False) -> None:
        if config_overrides is None:
            config_overrides = {}
        with cls.__init_lock:
            if reinit:
                cls.__instance = None
            if cls.__instance is None:
                cls.__instance = cls(config_overrides)
            elif len(config_overrides) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_STATE,
                    'Pixeltable has already been initialized; cannot specify new config values in the same session',
                )

    @classmethod
    def reload_if_changed(cls) -> bool:
        """Reload the config file if it changed since it was read. Returns True if it was reloaded."""
        with cls.__init_lock:
            if cls.__instance is None:
                return False
            if cls.__instance.__file_stamp() == cls.__instance.__stamp:
                return False
            config_overrides = cls.__instance.__config_overrides
            cls.__instance = None
            cls.__instance = cls(config_overrides)
            return True

    def __file_stamp(self) -> tuple[float, int] | None:
        """Modification time and size of the config file, or None if it is absent."""
        try:
            st = self.__config_file.stat()
        except OSError:
            return None
        return (st.st_mtime, st.st_size)

    @classmethod
    def __create_default_config(cls, config_path: Path) -> dict[str, Any]:
        free_disk_space_bytes = shutil.disk_usage(config_path.parent).free
        # Default cache size is 1/5 of free disk space
        file_cache_size_g = free_disk_space_bytes / 5 / (1 << 30)
        return {'pixeltable': {'file_cache_size_g': round(file_cache_size_g, 1), 'hide_warnings': False}}

    @classmethod
    def __read_toml_file(cls, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as stream:
                return toml.load(stream)
        except Exception as exc:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, f'Could not read config file: {path}'
            ) from exc

    @classmethod
    def __add_path(cls, config_dict: dict[str, Any], path: Path) -> dict[str, dict[str, tuple[Any, Path]]]:
        """Augment config_dict with path."""
        for _, section_dict in config_dict.items():
            assert isinstance(section_dict, dict)
        return {
            section: {key: (value, path) for key, value in section_dict.items()}
            for section, section_dict in config_dict.items()
            if isinstance(section_dict, dict)
        }

    def __load_user_config(self) -> dict[str, dict[str, tuple[Any, Path]]]:
        """Load the user's config file, creating a default one if it does not exist."""
        if self.__config_file.exists():
            config_dict = self.__read_toml_file(self.__config_file)
            self.__validate_config(config_dict, self.__config_file)
            return self.__add_path(config_dict, self.__config_file)

        else:
            config_dict = self.__create_default_config(self.__config_file)
            with open(self.__config_file, 'w', encoding='utf-8') as stream:
                try:
                    toml.dump(config_dict, stream)
                except Exception as exc:
                    raise excs.Error(
                        excs.ErrorCode.INTERNAL_ERROR, f'Could not create config file: {self.__config_file}'
                    ) from exc
            _logger.info(f'Created default config file at: {self.__config_file}')
            return self.__add_path(config_dict, self.__config_file)

    @classmethod
    def __validate_config(cls, config_dict: dict[str, Any], source: Path) -> None:
        non_section_keys = [key for key in config_dict if key not in KNOWN_CONFIG_OPTIONS]
        for key in non_section_keys:
            # `key` does not represent a section; relocate it to 'pixeltable' subsection
            if 'pixeltable' not in config_dict:
                config_dict['pixeltable'] = {}
            config_dict['pixeltable'][key] = config_dict[key]
            del config_dict[key]
        for section, section_dict in config_dict.items():
            if section not in KNOWN_CONFIG_OPTIONS:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION, f'Unrecognized section {section!r} in config file: {source}'
                )
            if not isinstance(section_dict, dict):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'Expected a table for section {section!r} in config file: {source}',
                )
            for key in section_dict:
                if key not in KNOWN_CONFIG_OPTIONS[section]:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_CONFIGURATION,
                        f"Unrecognized option '{section}.{key}' in config file: {source}",
                    )
                info = KNOWN_CONFIG_OPTIONS[section][key]
                if isinstance(info, tuple):
                    _, expected_type = info
                    section_dict[key] = cls.__validate_config_value(
                        section, key, section_dict[key], expected_type, source
                    )

    @classmethod
    def __validate_config_value(cls, section: str, key: str, value: Any, expected_type: type, source: Path) -> Any:
        """
        A config value could be a scalar, as in `pixeltable.file_cache_size_g`, or it could be a dict or a list of
        dicts that represents a Pydantic model. If the given key has a specified type, this method validates it
        as the given type. If the type is a Pydantic model or a list[Pydantic model], it converts the given dict(s)
        to the appropriate model instance(s).

        non-Pydantic types are currently not supported (but we could add support for them in the future).
        """
        origin_t = typing.get_origin(expected_type) or expected_type
        # Currently only list[PydanticModel] validation is supported.
        # TODO: Introduce fail-fast config validation for more types
        assert origin_t is list
        if not isinstance(value, origin_t):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f"Invalid type for option '{section}.{key}' in config file: {source}\n"
                f'(expected `{origin_t.__name__}`, got `{type(value).__name__}`)',
            )
        subscript = typing.get_args(expected_type)
        assert subscript is not None and len(subscript) == 1 and issubclass(subscript[0], pydantic.BaseModel)
        model_type = subscript[0]
        try:
            validated_config = [model_type.model_validate(entry) for entry in value]
        except pydantic.ValidationError as e:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid `{subscript[0].__name__}` in config file: {source}\n{e}'
            ) from e
        if 'name' in model_type.model_fields:
            # Convention: if the model has a 'name' field, it must be unique among entries.
            names: set[str] = set()
            for entry in validated_config:
                name = entry.name
                if name in names:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_CONFIGURATION,
                        f"Duplicate `{model_type.__name__}` name '{entry.name}' in config file: {source}",
                    )
                names.add(name)
        return validated_config

    def lookup_env(self, section: str, key: str, default: Any = None) -> Any:
        override_var = f'{section}.{key}'
        if override_var in self.__config_overrides:
            return self.__config_overrides[override_var]
        env_var = env_var_name(section, key)
        if env_var not in os.environ or len(os.environ[env_var]) == 0:
            return default
        if section == 'pixeltable' and key in _CONFIG_ONLY_KEYS:
            if env_var not in self.__reported_env_vars:
                self.__reported_env_vars.add(env_var)
                warnings.warn(
                    f'Ignoring {env_var}: {section}.{key} can only be set via the config file ({self.__config_file}).',
                    category=excs.PixeltableWarning,
                    stacklevel=2,
                )
            return default
        return os.environ[env_var]

    def __lookup_config_entry(self, section: str, key: str) -> tuple[Any, Path | None] | None:
        """Find key under section in __config_dict. Returns (value, source_path) or None."""
        parts = section.split('.')
        # explicit type decl for readability
        top_section: dict[str, tuple[Any, Path | None]] | None = self.__config_dict.get(parts[0])
        if top_section is None:
            return None
        if len(parts) == 1:
            return top_section.get(key)

        if parts[1] not in top_section:
            return None
        sub_section, source = top_section[parts[1]]
        for p in parts[2:]:
            if not isinstance(sub_section, dict):
                return None
            sub_section = sub_section.get(p)
            if sub_section is None:
                return None
        if not isinstance(sub_section, dict) or key not in sub_section:
            return None
        return (sub_section[key], source)

    def get_value(self, key: str, expected_type: type[T], section: str = 'pixeltable') -> T | None:
        value: Any = self.lookup_env(section, key)  # Try to get from environment first
        # Next try the config file
        if value is None:
            entry = self.__lookup_config_entry(section, key)
            if entry is None:
                return None
            value = entry[0]

        if value is None:
            return None  # Not specified

        try:
            if expected_type is bool and isinstance(value, str):
                if value.lower() not in ('true', 'false'):
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_CONFIGURATION,
                        f"Invalid value for configuration parameter '{section}.{key}': {value}",
                    )
                return value.lower() == 'true'  # type: ignore[return-value]
            if (expected_type is dict or expected_type is list) and isinstance(value, str):
                # Treat a string as a JSON-serialized dict or list
                value = json.loads(value)
            return expected_type(value)  # type: ignore[call-arg]
        except (ValueError, TypeError) as exc:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f"Invalid value for configuration parameter '{section}.{key}': {value}",
            ) from exc

    def get_string_value(self, key: str, section: str = 'pixeltable') -> str | None:
        return self.get_value(key, str, section)

    def get_int_value(self, key: str, section: str = 'pixeltable') -> int | None:
        return self.get_value(key, int, section)

    def get_float_value(self, key: str, section: str = 'pixeltable') -> float | None:
        return self.get_value(key, float, section)

    def get_bool_value(self, key: str, section: str = 'pixeltable') -> bool | None:
        return self.get_value(key, bool, section)

    def get_list_value(self, key: str, section: str = 'pixeltable') -> list[Any] | None:
        return self.get_value(key, list, section)

    def config_keys(self) -> list[ConfigKey]:
        """Return all configuration settings: the known-schema registry, plus active config vars."""
        result: list[ConfigKey] = []
        for section, options in KNOWN_CONFIG_OPTIONS.items():
            for key, info in options.items():
                if isinstance(info, tuple):
                    description, expected_type = info
                else:
                    description, expected_type = info, str
                result.append(ConfigKey(section=section, key=key, description=description, expected_type=expected_type))
        result.extend(self.__config_var_keys())
        return result

    def get_value_source(self, key: str, section: str = 'pixeltable') -> Path | Literal['env', 'unset']:
        """Return the source of the config value returned by get_value():
        - 'env': an environment variable or a pxt.init() config override is set
        - Path: the config file the value came from
        - 'unset': neither carries the value
        """
        if self.lookup_env(section, key) is not None:
            return 'env'
        entry = self.__lookup_config_entry(section, key)
        if entry is None:
            return 'unset'
        path = entry[1]
        return path if path is not None else 'unset'

    def env_keys(self) -> list[ConfigKey]:
        """The config settings that can be set via an environment variable."""
        return [ck for ck in self.config_keys() if is_env_key(ck)]

    def __config_var_keys(self) -> list[ConfigKey]:
        """The config vars from the config file and the environment."""
        result: list[ConfigKey] = []
        for section, prefix, description in (
            (VAR_SECTION, VAR_ENV_PREFIX, 'user-declared config var'),
            (SECRET_SECTION, SECRET_ENV_PREFIX, 'user-declared secret'),
        ):
            keys = set(self.__section_keys(section))
            for name, value in os.environ.items():
                # a config setting supplied by an env var needs to be uppercase
                suffix = name[len(prefix) :]
                if not name.startswith(prefix) or value == '' or suffix != suffix.upper():
                    continue
                if re.fullmatch(_CONFIG_VAR_NAME_RE, suffix.lower()) is not None:
                    keys.add(suffix.lower())
            result.extend(
                ConfigKey(section=section, key=key, description=description, expected_type=str) for key in sorted(keys)
            )
        return result

    def __warn_about_miscased_env_vars(self) -> None:
        """Warn about an environment variable that differs only in case from one this instance reads."""
        recognized = {env_var_name(ck.section, ck.key) for ck in self.config_keys()}
        for name in sorted(os.environ):
            if name == name.upper() or not name.upper().startswith('PIXELTABLE_'):
                continue
            upper = name.upper()
            if upper in recognized or upper.startswith((VAR_ENV_PREFIX, SECRET_ENV_PREFIX)):
                warnings.warn(
                    f'Ignoring {name}: environment variable names are uppercase; did you mean {upper}?',
                    category=excs.PixeltableWarning,
                    stacklevel=2,
                )

    def __section_keys(self, section: str) -> list[str]:
        """The keys defined in section."""
        parts = section.split('.')
        node: Any = self.__config_dict.get(parts[0])
        for p in parts[1:]:
            if not isinstance(node, dict):
                return []
            entry = node.get(p)
            node = entry[0] if isinstance(entry, tuple) else entry
        return list(node) if isinstance(node, dict) else []

    def env_fingerprint(self) -> dict[str, str]:
        """Fingerprints of the values of env-settable settings, as {env var name: hash}."""
        out: dict[str, str] = {}
        for ck in self.env_keys():
            value = self.get_value(ck.key, str, section=ck.section)
            if value is not None and value != '':
                out[env_var_name(ck.section, ck.key)] = value_fingerprint(value)
        return out

    def compare_env_values(self, other: Mapping[str, str]) -> tuple[list[str], list[str]]:
        """Compare an env fingerprint with this instance's.

        Returns (set for other but not here, resolved differently here). A name this instance does not
        read config from is ignored, so an unrelated variable in the other's environment does not count. A
        value this instance has and other does not is not a disagreement.
        """
        mine = self.env_fingerprint()
        known = {env_var_name(ck.section, ck.key) for ck in self.env_keys()}
        relevant = {
            name: h
            for name, h in other.items()
            if name.startswith((VAR_ENV_PREFIX, SECRET_ENV_PREFIX)) or name in known
        }
        missing = sorted(name for name in relevant if name not in mine)
        differing = sorted(name for name, h in relevant.items() if name in mine and mine[name] != h)
        return missing, differing


KNOWN_CONFIG_OPTIONS: dict[str, dict[str, Any]] = {
    'pixeltable': {
        'home': 'Path to the Pixeltable home directory',
        'config': 'Path to the Pixeltable config file',
        'pgdata': 'Path to the Pixeltable postgres data directory',
        'db': 'Postgres database name',
        'file_cache_size_g': 'Size of the file cache in GB',
        'file_cache_lease_s': 'Seconds a cached media file is protected from eviction after it is accessed',
        'time_zone': 'Default time zone for timestamps',
        'hide_warnings': 'Hide warnings from the console',
        'verbosity': 'Verbosity level for console output',
        'log_level': "Level of the 'pixeltable' logger, eg DEBUG (default: INFO)",
        'sql_log_level': "Level of the 'sqlalchemy.engine' logger: INFO logs SQL statements (default: WARNING)",
        'show_progress': 'Show a progress tracker for long-running operations (default: false)',
        'api_key': 'API key for Pixeltable cloud',
        'input_media_dest': 'Default destination URI for input media data',
        'output_media_dest': 'Default destination URI for output (computed) media data',
        'r2_profile': 'AWS config profile name used to access R2 storage',
        's3_profile': 'AWS config profile name used to access S3 storage',
        'b2_profile': 'AWS config profile name used to access Backblaze B2 storage',
        'tigris_profile': 'AWS config profile name used to access Tigris object storage',
        'service': ('Service configurations', list[ServiceConfig]),
        'database': 'Database configuration: runtime image, and variable/secret bindings',
        'daemon_host': 'Listen address for the proxy daemon in fixed-address mode (e.g. 0.0.0.0)',
        'daemon_port': ('Listen port for the proxy daemon in fixed-address mode (e.g. 8000)', int),
        'db_uri': 'Base pxt:// URI for remote catalog access (e.g. pxt://myorg:mydb)',
    },
    'anthropic': {'api_key': 'Anthropic API key'},
    'azure': {'storage_account_name': 'Azure storage account name', 'storage_account_key': 'Azure storage account key'},
    'bedrock': {
        'api_key': 'AWS Bedrock API key',
        'region_name': 'AWS region for Bedrock (default: us-east-1)',
        'temp_location': 'S3 URI for temporary storage used by Bedrock async model invocations',
        'performance_config_latency': 'Performance setting for supported models (standard or optimized)',
        'service_tier': 'Processing tier for requests (priority, default, flex, or reserved)',
    },
    'bfl': {'api_key': 'Black Forest Labs (BFL) API key', 'rate_limit': 'Rate limit for BFL API requests'},
    'deepseek': {'api_key': 'Deepseek API key', 'rate_limit': 'Rate limit for Deepseek API requests'},
    'fal': {'api_key': 'fal.ai API key', 'rate_limit': 'Rate limit for fal.ai API requests'},
    'fireworks': {'api_key': 'Fireworks API key', 'rate_limit': 'Rate limit for Fireworks API requests'},
    'gemini': {
        'api_key': (
            'Gemini API key for Google AI Studio only; '
            'for Vertex AI, use standard Google Gen AI SDK authentication instead'
        ),
        'rate_limits': 'Per-model rate limits for Gemini API requests',
    },
    'hf': {'token': 'Hugging Face access token'},
    'imagen': {'rate_limits': 'Per-model rate limits for Imagen API requests'},
    'groq': {'api_key': 'Groq API key', 'rate_limit': 'Rate limit for Groq API requests'},
    'jina': {'api_key': 'Jina AI API key', 'rate_limit': 'Rate limit for Jina AI API requests'},
    'mistral': {'api_key': 'Mistral API key', 'rate_limit': 'Rate limit for Mistral API requests'},
    'nebius': {
        'api_key': 'Nebius Token Factory API key',
        'rate_limit': 'Rate limit for Nebius Token Factory API requests',
    },
    'openai': {
        'api_key': 'OpenAI API key',
        'base_url': 'OpenAI API base URL',
        'api_version': 'API version if using Azure OpenAI',
        'rate_limits': 'Per-model rate limits for OpenAI API requests',
        'max_connections': 'Maximum number of concurrent OpenAI API connections that can be established',
        'max_keepalive_connections': 'Maximum number of keep-alive connections in the pool.'
        ' Must not exceed max_connections.',
        'read_timeout': 'HTTP read timeout',
        'write_timeout': 'HTTP write timeout',
    },
    'openrouter': {
        'api_key': 'OpenRouter API key',
        'site_url': 'Optional URL for your application (for OpenRouter analytics)',
        'app_name': 'Optional name for your application (for OpenRouter analytics)',
        'rate_limit': 'Rate limit for OpenRouter API requests',
    },
    'otel': {
        'exporter_otlp_endpoint': 'OTLP collector endpoint (eg http://localhost:4318)',
        'exporter_otlp_protocol': "OTLP transport: 'http/protobuf' (default) or 'grpc'",
        'service_name': 'service.name resource attribute (default: pixeltable)',
        'exporter_otlp_headers': "OTLP headers as comma-separated 'key=value' pairs",
        'span_level': "Span verbosity: 'info' (default), 'debug', or 'trace'",
        'metrics': 'Export metrics via OTLP (default: only when an OTLP endpoint is configured)',
        'logs': 'Export pixeltable logs via OTLP (default: false)',
    },
    'replicate': {'api_token': 'Replicate API token'},
    'runwayml': {'api_secret': 'RunwayML API secret'},
    'together': {
        'api_key': 'Together API key',
        'rate_limits': 'Per-model category rate limits for Together API requests',
    },
    'twelvelabs': {'api_key': 'TwelveLabs API key', 'rate_limit': 'Rate limit for TwelveLabs API requests'},
    'veo': {'rate_limits': 'Per-model rate limits for Veo API requests'},
    'voyage': {'api_key': 'Voyage AI API key', 'rate_limit': 'Rate limit for Voyage AI API requests'},
    'pypi': {'api_key': 'PyPI API key (for internal use only)'},
}


# settings that apply to the Pixeltable instance as a whole, and so can only be set in the config file
_CONFIG_ONLY_KEYS = frozenset({'file_cache_size_g', 'file_cache_lease_s', 'input_media_dest', 'output_media_dest'})

# the settings pxt.init() accepts, ie. the ones a single process may set
KNOWN_CONFIG_OVERRIDES = {
    f'{section}.{key}': info
    for section, section_dict in KNOWN_CONFIG_OPTIONS.items()
    for key, info in section_dict.items()
    if not (section == 'pixeltable' and key in _CONFIG_ONLY_KEYS)
}
