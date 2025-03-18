from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar

import toml

from pixeltable import exceptions as excs

_logger = logging.getLogger('pixeltable')

T = TypeVar('T')


class Config:
    """
    The (global) Pixeltable configuration, as loaded from `config.toml`. Provides methods for retrieving
    configuration values, which can be set in the config file or as environment variables.
    """

    __instance: ClassVar[Optional[Config]] = None

    __home: Path
    __config_file: Path
    __config: dict[str, Any]

    def __init__(self, home: Path, config_file: Path, config: dict[str, Any]) -> None:
        assert self.__instance is None, 'Config is a singleton; use Config.get() to access the instance'
        self.__home = home
        self.__config_file = config_file
        self.__config = config

    @property
    def home(self) -> Path:
        return self.__home

    @property
    def config_file(self) -> Path:
        return self.__config_file

    @classmethod
    def get(cls) -> Config:
        if cls.__instance is None:
            cls.reload()
        return cls.__instance

    @classmethod
    def reload(cls) -> None:
        """
        Loads configuration from PIXELTABLE_CONFIG as specified in the environment. If PIXELTABLE_HOME and/or
        PIXELTABLE_CONFIG do not exist, they will be created.
        """
        home = Path(os.environ.get('PIXELTABLE_HOME', str(Path.home() / '.pixeltable')))
        if home.exists() and not home.is_dir():
            raise RuntimeError(f'{home} is not a directory')
        if not home.exists():
            print(f'Creating a Pixeltable instance at: {home}')
            home.mkdir()

        config_file = Path(os.environ.get('PIXELTABLE_CONFIG', str(home / 'config.toml')))

        config_dict: dict[str, Any]
        if os.path.isfile(config_file):
            with open(config_file, 'r', encoding='utf-8') as stream:
                try:
                    config_dict = toml.load(stream)
                except Exception as exc:
                    raise excs.Error(f'Could not read config file: {config_file}') from exc
        else:
            config_dict = cls.__create_default_config(config_file)
            with open(config_file, 'w', encoding='utf-8') as stream:
                try:
                    toml.dump(config_dict, stream)
                except Exception as exc:
                    raise excs.Error(f'Could not write config file: {config_file}') from exc
            _logger.info(f'Created default config file at: {config_file}')

        cls.__instance = cls(home, config_file, config_dict)

    @classmethod
    def __create_default_config(cls, config_path: Path) -> dict[str, Any]:
        free_disk_space_bytes = shutil.disk_usage(config_path.parent).free
        # Default cache size is 1/5 of free disk space
        file_cache_size_g = free_disk_space_bytes / 5 / (1 << 30)
        return {'pixeltable': {'file_cache_size_g': round(file_cache_size_g, 1), 'hide_warnings': False}}

    def get_value(self, key: str, expected_type: type[T], section: str = 'pixeltable') -> Optional[T]:
        env_var = f'{section.upper()}_{key.upper()}'
        if env_var in os.environ:
            value = os.environ[env_var]
        elif section in self.__config and key in self.__config[section]:
            value = self.__config[section][key]
        else:
            return None

        try:
            return expected_type(value)  # type: ignore[call-arg]
        except ValueError as exc:
            raise excs.Error(f'Invalid value for configuration parameter {section}.{key}: {value}') from exc

    def get_string_value(self, key: str, section: str = 'pixeltable') -> Optional[str]:
        return self.get_value(key, str, section)

    def get_int_value(self, key: str, section: str = 'pixeltable') -> Optional[int]:
        return self.get_value(key, int, section)

    def get_float_value(self, key: str, section: str = 'pixeltable') -> Optional[float]:
        return self.get_value(key, float, section)

    def get_bool_value(self, key: str, section: str = 'pixeltable') -> Optional[bool]:
        return self.get_value(key, bool, section)
