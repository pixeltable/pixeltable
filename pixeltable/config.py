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
    __config_dict: dict[str, Any]

    def __init__(self) -> None:
        assert self.__instance is None, 'Config is a singleton; use Config.get() to access the instance'

        self.__home = Path(os.environ.get('PIXELTABLE_HOME', str(Path.home() / '.pixeltable')))
        if self.__home.exists() and not self.__home.is_dir():
            raise RuntimeError(f'{self.__home} is not a directory')
        if not self.__home.exists():
            print(f'Creating a Pixeltable instance at: {self.__home}')
            self.__home.mkdir()

        self.__config_file = Path(os.environ.get('PIXELTABLE_CONFIG', str(self.__home / 'config.toml')))

        self.__config_dict: dict[str, Any]
        if os.path.isfile(self.__config_file):
            with open(self.__config_file, 'r', encoding='utf-8') as stream:
                try:
                    self.__config_dict = toml.load(stream)
                except Exception as exc:
                    raise excs.Error(f'Could not read config file: {self.__config_file}') from exc
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
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

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
        elif section in self.__config_dict and key in self.__config_dict[section]:
            value = self.__config_dict[section][key]
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
