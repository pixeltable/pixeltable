from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import typing
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, NamedTuple, TypeVar

import pydantic
import toml

from pixeltable import exceptions as excs

_logger = logging.getLogger(__name__)

T = TypeVar('T')


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


class DeploymentConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='forbid')

    name: str
    service: str
    env: str
    include: list[str] | None = None
    exclude: list[str] | None = None
    env_dependencies: list[str] = pydantic.Field(default_factory=list)
    python_dependencies: list[str] = pydantic.Field(default_factory=list)

    @pydantic.field_validator('name')
    @classmethod
    def _validate_name(cls, v: str) -> str:
        from pixeltable.catalog import is_valid_identifier

        if not is_valid_identifier(v, allow_hyphens=True):
            raise ValueError(f'{v!r} is not a valid Pixeltable identifier')
        return v


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
    The (global) Pixeltable configuration, as loaded from `config.toml`. Provides methods for retrieving
    configuration values, which can be set in the config file or as environment variables.
    """

    __instance: ClassVar[Config | None] = None
    __init_lock: ClassVar[threading.Lock] = threading.Lock()

    __home: Path
    __config_file: Path
    __config_overrides: dict[str, Any]

    # section -> key -> (value, source_path); source_path is None for settings that don't come from a file
    __config_dict: dict[str, dict[str, tuple[Any, Path | None]]]

    def __init__(self, config_overrides: dict[str, Any], additional_config_files: list[str]) -> None:
        assert self.__instance is None, 'Config is a singleton; use Config.get() to access the instance'

        for var in config_overrides:
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

        self.__config_file = Path(self.lookup_env('pixeltable', 'config', str(self.__home / 'config.toml')))

        # Load configuration from (in order of precedence, highest to lowest):
        #   1. additional_config_files
        #   2. ./pixeltable.toml, if present
        #   3. The `[tool.pixeltable]` section of ./pyproject.toml, if present
        #   4. The user's config file (~/.pixeltable/config.toml by default)

        project_config = self.__load_project_config(Path.cwd() / 'pixeltable.toml')
        pyproject_config = self.__load_pyproject_config(Path.cwd() / 'pyproject.toml')
        user_config = self.__load_user_config()
        additional_configs = [self.__load_project_config(Path(f)) for f in additional_config_files]

        self.__config_dict = {}

        # Load lowest precedence first (for additional configs, last specified = highest precedence).
        for source in (user_config, pyproject_config, project_config, *additional_configs):
            self.__merge_config(self.__config_dict, source)

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
        cls.init({})
        return cls.__instance

    @classmethod
    def init(
        cls, config_overrides: dict[str, Any], additional_config_files: list[str] | None = None, reinit: bool = False
    ) -> None:
        if additional_config_files is None:
            additional_config_files = []
        with cls.__init_lock:
            if reinit:
                cls.__instance = None
            if cls.__instance is None:
                cls.__instance = cls(config_overrides, additional_config_files)
            elif len(config_overrides) > 0 or len(additional_config_files) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_STATE,
                    'Pixeltable has already been initialized; cannot specify new config values in the same session',
                )

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

    @classmethod
    def __load_project_config(cls, path: Path) -> dict[str, dict[str, tuple[Any, Path]]]:
        """Load ./pixeltable.toml, if it exists. Same structure as the user config file."""
        config_dict = cls.__read_toml_file(path)
        cls.__validate_config(config_dict, path)
        return cls.__add_path(config_dict, path)

    @classmethod
    def __load_pyproject_config(cls, path: Path) -> dict[str, dict[str, tuple[Any, Path]]]:
        """Load the `[tool.pixeltable]` table from ./pyproject.toml, if it exists.

        Subsections are expressed as `[tool.pixeltable.<section>]` (e.g. `[tool.pixeltable.openai]`).

        `[tool.pixeltable.pixeltable]` is shortened to `[tool.pixeltable]`.
        """
        pyproject = cls.__read_toml_file(path)
        config_dict = pyproject.get('tool', {}).get('pixeltable', {})
        if not isinstance(config_dict, dict):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, f"Expected a table for '[tool.pixeltable]' in config file: {path}"
            )
        cls.__validate_config(config_dict, path)
        return cls.__add_path(config_dict, path)

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

    @classmethod
    def __merge_config(
        cls, base: dict[str, dict[str, tuple[Any, Path | None]]], overlay: dict[str, dict[str, tuple[Any, Path]]]
    ) -> None:
        """Merge `overlay` into `base` at the section.key level; `overlay` values take precedence."""
        for section, section_dict in overlay.items():
            base.setdefault(section, {}).update(section_dict)

    def lookup_env(self, section: str, key: str, default: Any = None) -> Any:
        override_var = f'{section}.{key}'
        env_var = f'{section.upper()}_{key.upper()}'
        if override_var in self.__config_overrides:
            return self.__config_overrides[override_var]
        if env_var in os.environ and len(os.environ[env_var]) > 0:
            return os.environ[env_var]
        return default

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
        """Return all configuration settings from the known-schema registry."""
        result: list[ConfigKey] = []
        for section, options in KNOWN_CONFIG_OPTIONS.items():
            for key, info in options.items():
                if isinstance(info, tuple):
                    description, expected_type = info
                else:
                    description, expected_type = info, str
                result.append(ConfigKey(section=section, key=key, description=description, expected_type=expected_type))
        return result

    def get_value_source(self, key: str, section: str = 'pixeltable') -> Path | Literal['env', 'unset']:
        """Return the source of the config value returned by get_value():
        - 'env': environment variable (or programmatic config override) is set
        - Path: the file the value came from (one of user config, project pixeltable.toml,
          pyproject.toml, or one of additional_config_files)
        - 'unset': no layer carries the value
        """
        if self.lookup_env(section, key) is not None:
            return 'env'
        entry = self.__lookup_config_entry(section, key)
        if entry is None:
            return 'unset'
        path = entry[1]
        return path if path is not None else 'unset'


KNOWN_CONFIG_OPTIONS: dict[str, dict[str, Any]] = {
    'pixeltable': {
        'home': 'Path to the Pixeltable home directory',
        'config': 'Path to the Pixeltable config file',
        'pgdata': 'Path to the Pixeltable postgres data directory',
        'db': 'Postgres database name',
        'file_cache_size_g': 'Size of the file cache in GB',
        'time_zone': 'Default time zone for timestamps',
        'hide_warnings': 'Hide warnings from the console',
        'verbosity': 'Verbosity level for console output',
        'show_progress': 'Show a progress tracker for long-running operations (default: false)',
        'api_key': 'API key for Pixeltable cloud',
        'input_media_dest': 'Default destination URI for input media data',
        'output_media_dest': 'Default destination URI for output (computed) media data',
        'r2_profile': 'AWS config profile name used to access R2 storage',
        's3_profile': 'AWS config profile name used to access S3 storage',
        'b2_profile': 'AWS config profile name used to access Backblaze B2 storage',
        'tigris_profile': 'AWS config profile name used to access Tigris object storage',
        'service': ('Service configurations', list[ServiceConfig]),
        'deployment': ('Deployment configurations', list[DeploymentConfig]),
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
    'reve': {'api_key': 'Reve API key', 'rate_limit': 'Rate limit for Reve API requests (requests per minute)'},
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

KNOWN_CONFIG_OVERRIDES = {
    f'{section}.{key}': info
    for section, section_dict in KNOWN_CONFIG_OPTIONS.items()
    for key, info in section_dict.items()
}
