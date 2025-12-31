from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import toml

from pixeltable import env, exceptions as excs

_logger = logging.getLogger('pixeltable')

T = TypeVar('T')


class Config:
    """
    The (global) Pixeltable configuration, as loaded from `config.toml`. Provides methods for retrieving
    configuration values, which can be set in the config file or as environment variables.
    """

    __instance: ClassVar[Config | None] = None

    __home: Path
    __config_file: Path
    __config_overrides: dict[str, Any]
    __config_dict: dict[str, Any]

    def __init__(self, config_overrides: dict[str, Any]) -> None:
        assert self.__instance is None, 'Config is a singleton; use Config.get() to access the instance'

        for var in config_overrides:
            if var not in KNOWN_CONFIG_OVERRIDES:
                raise excs.Error(f'Unrecognized configuration variable: {var}')

        self.__config_overrides = config_overrides

        self.__home = Path(self.lookup_env('pixeltable', 'home', str(Path.home() / '.pixeltable')))
        if self.__home.exists() and not self.__home.is_dir():
            raise excs.Error(f'Not a directory: {self.__home}')
        if not self.__home.exists():
            print(f'Creating a Pixeltable instance at: {self.__home}')
            self.__home.mkdir()

        self.__config_file = Path(self.lookup_env('pixeltable', 'config', str(self.__home / 'config.toml')))

        self.__config_dict: dict[str, Any]
        if os.path.isfile(self.__config_file):
            with open(self.__config_file, 'r', encoding='utf-8') as stream:
                try:
                    self.__config_dict = toml.load(stream)
                except Exception as exc:
                    raise excs.Error(f'Could not read config file: {self.__config_file}') from exc
            for section, section_dict in self.__config_dict.items():
                if section not in KNOWN_CONFIG_OPTIONS:
                    raise excs.Error(f'Unrecognized section {section!r} in config file: {self.__config_file}')
                for key in section_dict:
                    if key not in KNOWN_CONFIG_OPTIONS[section]:
                        raise excs.Error(f"Unrecognized option '{section}.{key}' in config file: {self.__config_file}")
        else:
            self.__config_dict = self.__create_default_config(self.__config_file)
            with open(self.__config_file, 'w', encoding='utf-8') as stream:
                try:
                    toml.dump(self.__config_dict, stream)
                except Exception as exc:
                    raise excs.Error(f'Could not write config file: {self.__config_file}') from exc
            _logger.info(f'Created default config file at: {self.__config_file}')

    @property
    def home(self) -> Path:
        return self.__home

    @property
    def config_file(self) -> Path:
        return self.__config_file

    @classmethod
    def get(cls) -> Config:
        cls.init({})
        return cls.__instance

    @classmethod
    def init(cls, config_overrides: dict[str, Any], reinit: bool = False) -> None:
        if reinit:
            cls.__instance = None
            for cl in env._registered_clients.values():
                cl.client_obj = None
        if cls.__instance is None:
            cls.__instance = cls(config_overrides)
        elif len(config_overrides) > 0:
            raise excs.Error(
                'Pixeltable has already been initialized; cannot specify new config values in the same session'
            )

    @classmethod
    def __create_default_config(cls, config_path: Path) -> dict[str, Any]:
        free_disk_space_bytes = shutil.disk_usage(config_path.parent).free
        # Default cache size is 1/5 of free disk space
        file_cache_size_g = free_disk_space_bytes / 5 / (1 << 30)
        return {'pixeltable': {'file_cache_size_g': round(file_cache_size_g, 1), 'hide_warnings': False}}

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
                    raise excs.Error(f"Invalid value for configuration parameter '{section}.{key}': {value}")
                return value.lower() == 'true'  # type: ignore[return-value]
            return expected_type(value)  # type: ignore[call-arg]
        except (ValueError, TypeError) as exc:
            raise excs.Error(f"Invalid value for configuration parameter '{section}.{key}': {value}") from exc

    def get_string_value(self, key: str, section: str = 'pixeltable') -> str | None:
        return self.get_value(key, str, section)

    def get_int_value(self, key: str, section: str = 'pixeltable') -> int | None:
        return self.get_value(key, int, section)

    def get_float_value(self, key: str, section: str = 'pixeltable') -> float | None:
        return self.get_value(key, float, section)

    def get_bool_value(self, key: str, section: str = 'pixeltable') -> bool | None:
        return self.get_value(key, bool, section)


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
        'api_key': 'API key for Pixeltable cloud',
        'input_media_dest': 'Default destination URI for input media data',
        'output_media_dest': 'Default destination URI for output (computed) media data',
        'r2_profile': 'AWS config profile name used to access R2 storage',
        's3_profile': 'AWS config profile name used to access S3 storage',
        'b2_profile': 'AWS config profile name used to access Backblaze B2 storage',
        'tigris_profile': 'AWS config profile name used to access Tigris object storage',
    },
    'anthropic': {'api_key': 'Anthropic API key'},
    'azure': {'storage_account_name': 'Azure storage account name', 'storage_account_key': 'Azure storage account key'},
    'bedrock': {'api_key': 'AWS Bedrock API key'},
    'bfl': {'api_key': 'Black Forest Labs (BFL) API key', 'rate_limit': 'Rate limit for BFL API requests'},
    'deepseek': {'api_key': 'Deepseek API key', 'rate_limit': 'Rate limit for Deepseek API requests'},
    'fal': {'api_key': 'fal.ai API key', 'rate_limit': 'Rate limit for fal.ai API requests'},
    'fireworks': {'api_key': 'Fireworks API key', 'rate_limit': 'Rate limit for Fireworks API requests'},
    'gemini': {'api_key': 'Gemini API key', 'rate_limits': 'Per-model rate limits for Gemini API requests'},
    'hf': {'auth_token': 'Hugging Face access token'},
    'imagen': {'rate_limits': 'Per-model rate limits for Imagen API requests'},
    'reve': {'api_key': 'Reve API key', 'rate_limit': 'Rate limit for Reve API requests (requests per minute)'},
    'groq': {'api_key': 'Groq API key', 'rate_limit': 'Rate limit for Groq API requests'},
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
    },
    'openrouter': {
        'api_key': 'OpenRouter API key',
        'site_url': 'Optional URL for your application (for OpenRouter analytics)',
        'app_name': 'Optional name for your application (for OpenRouter analytics)',
        'rate_limit': 'Rate limit for OpenRouter API requests',
    },
    'replicate': {'api_token': 'Replicate API token'},
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
