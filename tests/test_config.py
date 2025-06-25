import subprocess
import tempfile

import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs


class TestConfig:
    def test_config_errors(self) -> None:
        def spawn_cmd(cmd: str, expected_error_msg: str) -> None:
            result = subprocess.run(
                ('python', '-c', f'import pixeltable as pxt\n{cmd}'), capture_output=True, check=False
            )
            assert result.returncode != 0
            assert expected_error_msg in result.stderr.decode('utf-8')

        spawn_cmd(
            'pxt.init(NOT_A_CONFIG_VAR="test")',
            'pixeltable.exceptions.Error: Unrecognized configuration variable: NOT_A_CONFIG_VAR',
        )

        tmp = tempfile.mktemp('.toml')
        with open(tmp, 'w', encoding='utf-8') as fp:
            fp.write('This is neither a directory nor a valid TOML file.')
        spawn_cmd(f'pxt.init(PIXELTABLE_HOME="{tmp}")', f'pixeltable.exceptions.Error: Not a directory: {tmp}')
        spawn_cmd(
            f'pxt.init(PIXELTABLE_CONFIG="{tmp}")', f'pixeltable.exceptions.Error: Could not read config file: {tmp}'
        )

        with open(tmp, 'w', encoding='utf-8') as fp:
            fp.write('[unknown_section]\nkey = "value"')
        spawn_cmd(
            f'pxt.init(PIXELTABLE_CONFIG="{tmp}")',
            "pixeltable.exceptions.Error: Unrecognized section 'unknown_section' in config file:",
        )

        with open(tmp, 'w', encoding='utf-8') as fp:
            fp.write('[pixeltable]\nunknown_key = "value"')
        spawn_cmd(
            f'pxt.init(PIXELTABLE_CONFIG="{tmp}")',
            "pixeltable.exceptions.Error: Unrecognized option 'pixeltable.unknown_key' in config file:",
        )

        spawn_cmd(
            'pxt.init(PIXELTABLE_VERBOSITY="eggs")',
            'pixeltable.exceptions.Error: Invalid value for configuration parameter pixeltable.verbosity: eggs',
        )

        pxt.init()
        pxt.init()  # Ok to do a parameterless init() a second time
        with pytest.raises(
            excs.Error,
            match='Pixeltable has already been initialized; cannot specify new config values in the same session',
        ):
            pxt.init(PIXELTABLE_HOME='.')  # Not ok to specify new config values after init()
