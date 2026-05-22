import subprocess
import sys
from pathlib import Path

import pixeltable as pxt

from .utils import pxt_raises


class TestConfig:
    def test_config_errors(self, tmp_path: Path) -> None:
        def spawn_cmd(cmd: str, expected_error_msg: str) -> None:
            cmd = cmd.replace('\\', r'\\')  # Escape backslashes for Windows compatibility
            result = subprocess.run(
                (sys.executable, '-c', f'import pixeltable as pxt\n{cmd}'), capture_output=True, check=False
            )
            print(f'======= stderr from command: {cmd} =======')
            print(result.stderr.decode('utf-8'))
            assert result.returncode != 0
            assert expected_error_msg in result.stderr.decode('utf-8')

        spawn_cmd(
            'pxt.init({"pixeltable.not_a_config_var": "test"})',
            'pixeltable.exceptions.RequestError: Unrecognized configuration variable: pixeltable.not_a_config_var',
        )

        tmp = tmp_path / 'bad.toml'
        with open(tmp, 'w', encoding='utf-8') as fp:
            fp.write('This is neither a directory nor a valid TOML file.')
        spawn_cmd(
            f'pxt.init({{"pixeltable.home": "{tmp}"}})', f'pixeltable.exceptions.RequestError: Not a directory: {tmp}'
        )
        spawn_cmd(
            f'pxt.init({{"pixeltable.config": "{tmp}"}})',
            f'pixeltable.exceptions.RequestError: Could not read config file: {tmp}',
        )

        with open(tmp, 'w', encoding='utf-8') as fp:
            fp.write('[pixeltable]\nunknown_key = "value"')
        spawn_cmd(
            f'pxt.init({{"pixeltable.config": "{tmp}"}})',
            "pixeltable.exceptions.RequestError: Unrecognized option 'pixeltable.unknown_key' in config file:",
        )

        spawn_cmd(
            'pxt.init({"pixeltable.verbosity": "eggs"})',
            'pixeltable.exceptions.RequestError: Invalid value for configuration parameter '
            "'pixeltable.verbosity': eggs",
        )

        pxt.init()
        pxt.init()  # Ok to do a parameterless init() a second time
        with pxt_raises(
            pxt.ErrorCode.INVALID_STATE,
            match='Pixeltable has already been initialized; cannot specify new config values in the same session',
        ):
            pxt.init({'pixeltable.home': '.'})  # Not ok to specify new config values after init()

    def test_dotted_section_lookup(self, tmp_path: Path) -> None:
        """Nested TOML tables like [openai.rate_limits] are stored as
        __config_dict['openai']['rate_limits'] = ({'gpt-4': 250}, path). get_value must descend
        into that inner dict; a previous regression flattened the lookup and silently returned
        None, causing configured per-model rate limits to be ignored."""

        def spawn_cmd_ok(cmd: str) -> None:
            # Config.init() initializes only the Config singleton (no catalog/DB), so we can
            # exercise it in a bare subprocess without setting up Postgres.
            cmd = cmd.replace('\\', r'\\')
            result = subprocess.run(
                (sys.executable, '-c', f'from pixeltable.config import Config\n{cmd}'), capture_output=True, check=False
            )
            assert result.returncode == 0, (
                f'cmd failed:\nstdout:\n{result.stdout.decode("utf-8")}\nstderr:\n{result.stderr.decode("utf-8")}'
            )

        tmp = tmp_path / 'config.toml'
        with open(tmp, 'w', encoding='utf-8') as fp:
            fp.write('[openai.rate_limits]\n"gpt-4" = 250\n[together.rate_limits.chat]\n"llama-3-70b" = 100\n')

        # 1. single-dot dotted section: value is found in the nested table
        spawn_cmd_ok(
            f'Config.init({{"pixeltable.config": "{tmp}"}})\n'
            'v = Config.get().get_int_value("gpt-4", section="openai.rate_limits")\n'
            'assert v == 250, f"expected 250, got {v!r}"\n'
        )

        # 2. source-path tracking: get_value_source returns the file path the value came from
        spawn_cmd_ok(
            f'Config.init({{"pixeltable.config": "{tmp}"}})\n'
            'from pathlib import Path\n'
            's = Config.get().get_value_source("gpt-4", section="openai.rate_limits")\n'
            f'assert s == Path("{tmp}"), f"expected source {tmp}, got {{s!r}}"\n'
        )

        # 3. multi-level nesting: parts[2:] descent loop is exercised
        spawn_cmd_ok(
            f'Config.init({{"pixeltable.config": "{tmp}"}})\n'
            'v = Config.get().get_int_value("llama-3-70b", section="together.rate_limits.chat")\n'
            'assert v == 100, f"expected 100, got {v!r}"\n'
        )

        # 4. section exists but inner key missing: returns None (not an error)
        spawn_cmd_ok(
            f'Config.init({{"pixeltable.config": "{tmp}"}})\n'
            'v = Config.get().get_int_value("gpt-5", section="openai.rate_limits")\n'
            'assert v is None, f"expected None, got {v!r}"\n'
        )
