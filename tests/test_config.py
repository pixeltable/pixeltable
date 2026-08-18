import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import pixeltable as pxt
from pixeltable.config import Config

from .utils import get_image_files


class TestConfig:
    def test_config_errors(self, init_env: None, tmp_path: Path) -> None:
        def spawn_cmd(env_vars: dict[str, str], expected_error_msg: str) -> None:
            result = subprocess.run(
                (sys.executable, '-c', 'import pixeltable as pxt\npxt.init()'),
                capture_output=True,
                check=False,
                env={**os.environ, **env_vars},
            )
            print(f'======= stderr with {env_vars} =======')
            print(result.stderr.decode('utf-8'))
            assert result.returncode != 0
            assert expected_error_msg in result.stderr.decode('utf-8')

        tmp = tmp_path / 'bad.toml'
        with open(tmp, 'w', encoding='utf-8') as fp:
            fp.write('This is neither a directory nor a valid TOML file.')
        spawn_cmd({'PIXELTABLE_HOME': str(tmp)}, f'pixeltable.exceptions.RequestError: Not a directory: {tmp}')
        spawn_cmd(
            {'PIXELTABLE_CONFIG': str(tmp)}, f'pixeltable.exceptions.RequestError: Could not read config file: {tmp}'
        )

        with open(tmp, 'w', encoding='utf-8') as fp:
            fp.write('[pixeltable]\nunknown_key = "value"')
        spawn_cmd(
            {'PIXELTABLE_CONFIG': str(tmp)},
            "pixeltable.exceptions.RequestError: Unrecognized option 'pixeltable.unknown_key' in config file:",
        )

        spawn_cmd(
            {'PIXELTABLE_VERBOSITY': 'eggs'},
            'pixeltable.exceptions.RequestError: Invalid value for configuration parameter '
            "'pixeltable.verbosity': eggs",
        )

        pxt.init()
        pxt.init()  # a second init() is a no-op

    def test_dotted_section_lookup(self, tmp_path: Path) -> None:
        """Nested TOML tables like [openai.rate_limits] are stored as
        __config_dict['openai']['rate_limits'] = ({'gpt-4': 250}, path). get_value must descend
        into that inner dict; a previous regression flattened the lookup and silently returned
        None, causing configured per-model rate limits to be ignored."""

        def spawn_cmd_ok(cmd: str, config_file: Path) -> None:
            # Config.init() initializes only the Config singleton (no catalog/DB), so we can
            # exercise it in a bare subprocess without setting up Postgres.
            result = subprocess.run(
                (sys.executable, '-c', f'from pixeltable.config import Config\n{cmd}'),
                capture_output=True,
                check=False,
                env={**os.environ, 'PIXELTABLE_CONFIG': str(config_file)},
            )
            assert result.returncode == 0, (
                f'cmd failed:\nstdout:\n{result.stdout.decode("utf-8")}\nstderr:\n{result.stderr.decode("utf-8")}'
            )

        tmp = tmp_path / 'config.toml'
        with open(tmp, 'w', encoding='utf-8') as fp:
            fp.write('[openai.rate_limits]\n"gpt-4" = 250\n[together.rate_limits.chat]\n"llama-3-70b" = 100\n')

        # 1. single-dot dotted section: value is found in the nested table
        spawn_cmd_ok(
            'v = Config.get().get_int_value("gpt-4", section="openai.rate_limits")\n'
            'assert v == 250, f"expected 250, got {v!r}"\n',
            tmp,
        )

        # 2. source-path tracking: get_value_source returns the file path the value came from. The subprocess
        # reads the expected path from the environment; interpolating it into the source would make a Windows
        # path's backslashes into escape sequences.
        spawn_cmd_ok(
            'import os\n'
            'from pathlib import Path\n'
            's = Config.get().get_value_source("gpt-4", section="openai.rate_limits")\n'
            'expected = Path(os.environ["PIXELTABLE_CONFIG"])\n'
            'assert s == expected, f"expected source {expected}, got {s!r}"\n',
            tmp,
        )

        # 3. multi-level nesting: parts[2:] descent loop is exercised
        spawn_cmd_ok(
            'v = Config.get().get_int_value("llama-3-70b", section="together.rate_limits.chat")\n'
            'assert v == 100, f"expected 100, got {v!r}"\n',
            tmp,
        )

        # 4. section exists but inner key missing: returns None (not an error)
        spawn_cmd_ok(
            'v = Config.get().get_int_value("gpt-5", section="openai.rate_limits")\n'
            'assert v is None, f"expected None, got {v!r}"\n',
            tmp,
        )

    def test_env_var_names(self, tmp_path: Path) -> None:
        """A setting is bound by its name uppercased, so only that spelling of a variable is read."""
        config_file = tmp_path / 'config.toml'
        config_file.write_text('[pixeltable.database.secrets]\ndeclared_in_file = "from-the-file"\n')

        def config_var_keys(env_vars: dict[str, str]) -> list[str]:
            """The secret names Config finds, resolved in a subprocess so the environment is exactly env_vars."""
            code = (
                'import json\n'
                'from pixeltable.config import Config, SECRET_SECTION\n'
                'print(json.dumps(sorted(ck.key for ck in Config.get().config_keys() '
                'if ck.section == SECRET_SECTION)))'
            )
            result = subprocess.run(
                (sys.executable, '-c', code),
                capture_output=True,
                check=True,
                env={**os.environ, 'PIXELTABLE_CONFIG': str(config_file), **env_vars},
            )
            return json.loads(result.stdout.decode('utf-8').strip())

        # a variable spelled as the name uppercased declares the var; any other spelling names nothing, and
        # neither does a variable with no name after the prefix
        assert config_var_keys({}) == ['declared_in_file']
        assert config_var_keys({'PIXELTABLE_SECRET_FROM_ENV': 'x'}) == ['declared_in_file', 'from_env']
        assert config_var_keys({'PIXELTABLE_SECRET_MiXeD': 'x'}) == ['declared_in_file']
        assert config_var_keys({'PIXELTABLE_SECRET_': 'x'}) == ['declared_in_file']

        # a declared name must be lowercase, so that it maps to exactly one env var name
        with pytest.raises(pxt.Error, match='Invalid config var name'):
            pxt.ConfigVar('MiXeD', pxt.Secret)
        assert pxt.ConfigVar('from_env', pxt.Secret).env_var == 'PIXELTABLE_SECRET_FROM_ENV'

    def test_miscased_env_var(self, tmp_path: Path) -> None:
        """A variable differing only in case from one that is read generates a warning."""
        result = subprocess.run(
            (sys.executable, '-W', 'always', '-c', 'from pixeltable.config import Config\nConfig.get()'),
            capture_output=True,
            check=True,
            env={
                **os.environ,
                'PIXELTABLE_Home': str(tmp_path / 'nope'),
                'PIXELTABLE_CONFIG': str(tmp_path / 'c.toml'),
            },
        )
        stderr = result.stderr.decode('utf-8')
        assert 'Ignoring PIXELTABLE_Home' in stderr, stderr
        assert 'did you mean PIXELTABLE_HOME' in stderr, stderr
        assert not (tmp_path / 'nope').exists(), 'the mis-cased variable was used as the home directory'

    def test_reload_if_changed(self, tmp_path: Path) -> None:
        """The config file is re-read after it changes, which is how a running daemon picks up an edit."""
        config_file = tmp_path / 'config.toml'
        config_file.write_text('[pixeltable.database.vars]\nmedia_dest = "s3://first/bucket"\n')

        media_dest = pxt.ConfigVar('media_dest', pxt.URI)

        original_config = os.environ.get('PIXELTABLE_CONFIG')
        os.environ['PIXELTABLE_CONFIG'] = str(config_file)
        Config.init(reinit=True)
        try:
            assert media_dest.value() == 's3://first/bucket'
            # an unchanged file is not re-read
            assert not Config.reload_if_changed()

            time.sleep(0.01)  # the stamp is (mtime, size), so a same-size rewrite needs a distinct mtime
            config_file.write_text('[pixeltable.database.vars]\nmedia_dest = "s3://second/bucket"\n')
            assert Config.reload_if_changed()
            assert media_dest.value() == 's3://second/bucket'
            assert not Config.reload_if_changed()
        finally:
            if original_config is None:
                os.environ.pop('PIXELTABLE_CONFIG', None)
            else:
                os.environ['PIXELTABLE_CONFIG'] = original_config
            Config.init(reinit=True)

    @pytest.mark.local('a local filesystem destination is rejected for a hosted table')
    def test_config_var_destination_follows_rebinding(self, uses_db: None, tmp_path: Path) -> None:
        """Rows written after a config var is rebound land where it now points.

        The destination is stored as a reference and resolved when a file is written, so a rebinding takes
        effect without a schema change and without reloading anything.
        """
        media_dir = tmp_path / 'media'
        media_dir.mkdir()
        # a name of a different length, and not a prefix of the other: rewriting the config below then
        # changes its size, and reload detection compares (mtime, size) with mtime granularity that is
        # coarse on some filesystems
        other_dir = tmp_path / 'rebound'
        other_dir.mkdir()
        config_file = tmp_path / 'config.toml'
        config_file.write_text(f'[pixeltable.database.vars]\nmedia_dest = "{media_dir.as_posix()}"\n')

        original_config = os.environ.get('PIXELTABLE_CONFIG')
        os.environ['PIXELTABLE_CONFIG'] = str(config_file)
        Config.init(reinit=True)
        try:
            t = pxt.create_table('cached_md', {'img': pxt.Image | None}, if_exists='replace')
            t.add_computed_column(thumb=t.img.rotate(90), destination=pxt.ConfigVar('media_dest', pxt.URI), stored=True)
            # metadata names the variable rather than where it currently points
            assert t.get_metadata()['columns']['thumb']['destination'] == '$media_dest'

            t.insert(img=get_image_files()[0])
            assert media_dir.as_uri() in t.select(url=t.thumb.fileurl).collect()[0]['url']

            config_file.write_text(f'[pixeltable.database.vars]\nmedia_dest = "{other_dir.as_posix()}"\n')
            assert Config.reload_if_changed()

            # the declaration is unchanged, so the column still reads as the same variable
            assert t.get_metadata()['columns']['thumb']['destination'] == '$media_dest'

            # a row written now goes where the variable points now
            t.insert(img=get_image_files()[1])
            urls = t.select(url=t.thumb.fileurl).collect()['url']
            assert sum(other_dir.as_uri() in url for url in urls) == 1
            assert sum(media_dir.as_uri() in url for url in urls) == 1
        finally:
            if original_config is None:
                os.environ.pop('PIXELTABLE_CONFIG', None)
            else:
                os.environ['PIXELTABLE_CONFIG'] = original_config
            Config.init(reinit=True)
