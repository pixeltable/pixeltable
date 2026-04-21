from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import toml

from pixeltable import exceptions as excs

_logger = logging.getLogger('pixeltable')

T = TypeVar('T')


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
    __config_dict: dict[str, Any]

    def __init__(self, config_overrides: dict[str, Any]) -> None:
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
        #   1. ./pixeltable.toml, if present
        #   2. The `[tool.pixeltable]` section of ./pyproject.toml, if present
        #   3. The user's config file (~/.pixeltable/config.toml by default)

        project_config = self.__load_project_config(Path.cwd() / 'pixeltable.toml')
        pyproject_config = self.__load_pyproject_config(Path.cwd() / 'pyproject.toml')
        user_config = self.__load_user_config()

        self.__config_dict = {}

        # Load lowest precedence first
        for source in (user_config, pyproject_config, project_config):
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
    def init(cls, config_overrides: dict[str, Any], reinit: bool = False) -> None:
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
    def __load_project_config(cls, path: Path) -> dict[str, Any]:
        """Load ./pixeltable.toml, if it exists. Same structure as the user config file."""
        config_dict = cls.__read_toml_file(path)
        cls.__validate_config(config_dict, path)
        return config_dict

    @classmethod
    def __load_pyproject_config(cls, path: Path) -> dict[str, Any]:
        """Load the `[tool.pixeltable]` table from ./pyproject.toml, if it exists.

        Subsections are expressed as `[tool.pixeltable.<section>]` (e.g. `[tool.pixeltable.openai]`).

        `[tool.pixeltable.pixeltable]` is shortened to `[tool.pixeltable]`.
        """
        pyproject = cls.__read_toml_file(path)
        config_dict = pyproject.get('tool', {}).get('pixeltable')
        if not isinstance(config_dict, dict):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, f"Expected a table for '[tool.pixeltable]' in config file: {path}"
            )
        for key, value in config_dict.items():
            if key not in KNOWN_CONFIG_OPTIONS:
                # `key` does not represent a section; relocate it to 'pixeltable' subsection
                if 'pixeltable' not in config_dict:
                    config_dict['pixeltable'] = {}
                config_dict['pixeltable'][key] = value
                del config_dict[key]
        cls.__validate_config(config_dict, path)
        return config_dict

    def __load_user_config(self) -> dict[str, Any]:
        """Load the user's config file, creating a default one if it does not exist."""
        if self.__config_file.exists():
            config_dict = self.__read_toml_file(self.__config_file)
            self.__validate_config(config_dict, self.__config_file)
            return config_dict

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
            return config_dict

    @classmethod
    def __validate_config(cls, config_dict: dict[str, Any], source: Path) -> None:
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

    @classmethod
    def __merge_config(cls, base: dict[str, Any], overlay: dict[str, Any]) -> None:
        """Merge `overlay` into `base` at the section.key level; `overlay` values take precedence."""
        for section, section_dict in overlay.items():
            if section in base and isinstance(base[section], dict) and isinstance(section_dict, dict):
                base[section].update(section_dict)
            else:
                base[section] = dict(section_dict) if isinstance(section_dict, dict) else section_dict

    def lookup_env(self, section: str, key: str, default: Any = None) -> Any:
        override_var = f'{section}.{key}'
        env_var = f'{section.upper()}_{key.upper()}'
        if override_var in self.__config_overrides:
            return self.__config_overrides[override_var]
        if env_var in os.environ and len(os.environ[env_var]) > 0:
            return os.environ[env_var]
        return default

    def get_value(self, key: str, expected_type: type[T], section: str = 'pixeltable') -> T | None:
        value: Any = self.lookup_env(section, key)  # Try to get from environment first
        # Next try the config file
        if value is None:
            # Resolve nested section dicts
            lookup_elems = [*section.split('.'), key]
            value = self.__config_dict
            for el in lookup_elems:
                if isinstance(value, dict):
                    if el not in value:
                        return None
                    value = value[el]
                else:
                    return None

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


KNOWN_CONFIG_OPTIONS = {
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
        'start_dashboard': 'Whether to launch the dashboard server as startup (default: true)',
        'dashboard_port': 'Port for the dashboard server (default: 22089)',
        'r2_profile': 'AWS config profile name used to access R2 storage',
        's3_profile': 'AWS config profile name used to access S3 storage',
        'b2_profile': 'AWS config profile name used to access Backblaze B2 storage',
        'tigris_profile': 'AWS config profile name used to access Tigris object storage',
        'deployment': 'Deployment configuration',
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
    'hf': {'auth_token': 'Hugging Face access token'},
    'imagen': {'rate_limits': 'Per-model rate limits for Imagen API requests'},
    'reve': {'api_key': 'Reve API key', 'rate_limit': 'Rate limit for Reve API requests (requests per minute)'},
    'groq': {'api_key': 'Groq API key', 'rate_limit': 'Rate limit for Groq API requests'},
    'jina': {'api_key': 'Jina AI API key', 'rate_limit': 'Rate limit for Jina AI API requests'},
    'label_studio': {'api_key': 'Label Studio API key', 'url': 'Label Studio server URL'},
    'mistral': {'api_key': 'Mistral API key', 'rate_limit': 'Rate limit for Mistral API requests'},
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
