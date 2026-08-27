import json
import os
import subprocess
import sys
import time
from typing import Iterator
from pathlib import Path
from textwrap import dedent

import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.config import SECRET_SECTION, VAR_SECTION, Config
from pixeltable.serving._config import lookup_database_config

from .utils import get_image_files, pxt_raises


class TestConfig:
    @pytest.fixture(autouse=True)
    def mutates_project_root(self) -> Iterator[None]:
        """Allows a test to re-init config with a new project root without changing it permanently.

        A test here reinitializes Config to create a project dir; the root outlives the test otherwise,
        and every process the session starts later is handed it.
        """
        original = Config.get().project_root
        yield
        if Config.get().project_root != original:
            Config.init(reinit=True, project_root=original)

    def test_config_errors(self, init_env: None, tmp_path: Path) -> None:
        def spawn_cmd(env_vars: dict[str, str], expected_error_msg: str, init_arg: str = '') -> None:
            result = subprocess.run(
                (sys.executable, '-c', f'import pixeltable as pxt\npxt.init({init_arg})'),
                capture_output=True,
                check=False,
                env={**os.environ, **env_vars},
            )
            print(f'======= stderr with {env_vars} and pxt.init({init_arg}) =======')
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

        # an override names its setting as 'section.key' and reaches the same lookup an env var does
        spawn_cmd(
            {},
            f'pixeltable.exceptions.RequestError: Not a directory: {tmp}',
            init_arg=f'{{"pixeltable.home": "{tmp.as_posix()}"}}',
        )
        spawn_cmd(
            {},
            'pixeltable.exceptions.RequestError: Unrecognized configuration variable: pixeltable.not_a_config_var',
            init_arg='{"pixeltable.not_a_config_var": "test"}',
        )

        # a setting that applies to the whole instance cannot be given per process
        spawn_cmd(
            {},
            'pixeltable.exceptions.RequestError: Cannot override pixeltable.file_cache_size_g: '
            'it can only be set via the config file.',
            init_arg='{"pixeltable.file_cache_size_g": 5}',
        )

        pxt.init()
        pxt.init()  # a second init() is a no-op
        with pxt_raises(
            pxt.ErrorCode.INVALID_STATE,
            match='Pixeltable has already been initialized; cannot specify new config values in the same session',
        ):
            pxt.init({'pixeltable.home': '.'})

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
        if sys.platform != 'win32':
            # Windows environment variable names are case-insensitive, so there this is the uppercase variable
            assert config_var_keys({'PIXELTABLE_SECRET_MiXeD': 'x'}) == ['declared_in_file']
        assert config_var_keys({'PIXELTABLE_SECRET_': 'x'}) == ['declared_in_file']

        # a declared name must be lowercase, so that it maps to exactly one env var name
        with pytest.raises(pxt.Error, match='Invalid config var name'):
            pxt.ConfigVar('MiXeD', pxt.Secret)
        assert pxt.ConfigVar('from_env', pxt.Secret).env_var == 'PIXELTABLE_SECRET_FROM_ENV'

        # a declared type must be one the stored reference can name, so that the metadata reads back
        class MySecret(pxt.Secret):
            pass

        with pytest.raises(pxt.Error, match="Invalid config var type 'MySecret': must be one of str, URI, Secret"):
            pxt.ConfigVar('custom', MySecret)

    @pytest.mark.skipif(sys.platform == 'win32', reason='environment variable names are case-insensitive on Windows')
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

    def test_project_discovery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The project root is the nearest one at or above the working directory."""

        def serving(cwd: Path) -> tuple[Path | None, Path | None]:
            """What a process starting in cwd resolves: its project root and config file."""
            monkeypatch.chdir(cwd)
            Config.init(reinit=True)
            return Config.get().project_root, Config.get().project_config_file

        nested = tmp_path / 'proj' / 'ad_gen' / 'inner'
        nested.mkdir(parents=True)

        # a directory under no project config has no project root
        assert serving(nested) == (None, None)

        # a pyproject.toml that says nothing about Pixeltable is not a project config
        (tmp_path / 'proj' / 'pyproject.toml').write_text('[project]\nname = "proj"\n')
        assert serving(nested) == (None, None)

        # one declaring [tool.pixeltable] is, and it configures every directory below it
        (tmp_path / 'proj' / 'pyproject.toml').write_text('[project]\nname = "proj"\n\n[tool.pixeltable]\n')
        assert serving(nested) == (tmp_path / 'proj', tmp_path / 'proj' / 'pyproject.toml')

        # a config closer to the working directory wins
        (tmp_path / 'proj' / 'ad_gen' / 'pixeltable.toml').write_text('')
        assert serving(nested) == (tmp_path / 'proj' / 'ad_gen', tmp_path / 'proj' / 'ad_gen' / 'pixeltable.toml')

        # a directory holding both is configured by its pixeltable.toml
        (tmp_path / 'proj' / 'ad_gen' / 'pyproject.toml').write_text('[project]\nname = "x"\n\n[tool.pixeltable]\n')
        assert serving(nested) == (tmp_path / 'proj' / 'ad_gen', tmp_path / 'proj' / 'ad_gen' / 'pixeltable.toml')

        # a pyproject.toml that cannot be parsed might be the project config, so it is refused rather than
        # resolved past
        unparseable = tmp_path / 'unparseable'
        unparseable.mkdir()
        (unparseable / 'pyproject.toml').write_text('[tool.pixeltable\n')
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r'cannot be parsed'):
            serving(unparseable)

        # an explicit root is used as given; an explicit None means no project
        Config.init(reinit=True, project_root=tmp_path / 'proj')
        assert Config.get().project_root == tmp_path / 'proj'
        monkeypatch.chdir(nested)
        Config.init(reinit=True, project_root=None)
        assert Config.get().project_root is None

    def test_project_config_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A project config file that cannot be used names itself in the error."""
        project = tmp_path / 'proj'
        project.mkdir()
        config_file = project / 'pixeltable.toml'

        def load(project_text: str) -> None:
            config_file.write_text(project_text)
            Config.init(reinit=True, project_root=project)

        # two entries for one database make its bindings ambiguous
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r"Duplicate `DatabaseConfig` name 'local'"):
            load('[[pixeltable.database]]\n\n[[pixeltable.database]]\n')

        # a file Pixeltable cannot parse
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r'pixeltable\.toml'):
            load('[[pixeltable.database]\n')

        # a database entry holding something no database is configured with
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r'Invalid `DatabaseConfig`'):
            load("[[pixeltable.database]]\nnot_a_setting = 'x'\n")

        # a runtime spec that conda could not install, and one that would carry a shell command
        for spec, message in (('""', r'non-empty conda package specs'), ('"ffmpeg; rm -rf /"', r'invalid character')):
            with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=message):
                load(f'[[pixeltable.database]]\nsystem_dependencies = [{spec}]\n')

        # a python version that is not a version
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r'`python_version` must be a version'):
            load("[[pixeltable.database]]\npython_version = '3'\n")

    def test_project_root_after_reload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The project root reaches Config through init(), and survives a reload."""
        project = tmp_path / 'proj'
        project.mkdir()
        (project / 'pixeltable.toml').write_text('')
        Config.init(reinit=True, project_root=project)

        # a root for a Config that already has one is refused, rather than ignored
        with pxt_raises(excs.ErrorCode.INVALID_STATE, match=r'already been initialized'):
            Config.init(project_root=tmp_path)
        assert Config.get().project_root == project

        # a reload re-reads the files, and keeps the root even though the working directory moved
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        time.sleep(0.01)  # the stamp is (mtime, size), so a same-size rewrite needs a distinct mtime
        (project / 'pixeltable.toml').write_text("[[pixeltable.database]]\nvars.after_reload = 'yes'\n")
        assert Config.reload_if_changed()
        assert Config.get().project_root == project
        assert Config.get().get_string_value('after_reload', section=VAR_SECTION) == 'yes'

    def test_project_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A config setting in the project is read, and wins over the home config."""
        home = tmp_path / 'home.toml'
        project = tmp_path / 'proj'
        project.mkdir()

        def load(home_text: str, project_text: str, *, in_pyproject: bool = False) -> Config:
            home.write_text(home_text)
            name = 'pyproject.toml' if in_pyproject else 'pixeltable.toml'
            (project / name).write_text(project_text)
            monkeypatch.setenv('PIXELTABLE_CONFIG', str(home))
            Config.init(reinit=True, project_root=project)
            return Config.get()

        # a var bound only by the project is read, and get_value_source() names its file
        config = load(
            '',
            dedent(
                """
                [[pixeltable.database]]
                vars.media_dest = 's3://project/bucket'
                """
            ),
        )
        assert config.project_config_file == project / 'pixeltable.toml'
        assert config.get_string_value('media_dest', section=VAR_SECTION) == 's3://project/bucket'
        assert config.get_value_source('media_dest', section=VAR_SECTION) == project / 'pixeltable.toml'
        assert pxt.ConfigVar('media_dest', pxt.URI).value() == 's3://project/bucket'

        # the two files configure one database together; the project's binding wins per name
        config = load(
            dedent(
                """
                [[pixeltable.database]]
                vars.media_dest = 's3://home/bucket'
                vars.other_dest = 's3://home/other'
                secrets.shared_key = 'from-home'
                """
            ),
            dedent(
                """
                [[pixeltable.database]]
                vars.media_dest = 's3://project/bucket'
                """
            ),
        )
        assert config.get_string_value('media_dest', section=VAR_SECTION) == 's3://project/bucket'
        assert config.get_value_source('media_dest', section=VAR_SECTION) == project / 'pixeltable.toml'
        assert config.get_string_value('other_dest', section=VAR_SECTION) == 's3://home/other'
        assert config.get_value_source('other_dest', section=VAR_SECTION) == home
        assert config.get_string_value('shared_key', section=SECRET_SECTION) == 'from-home'

        # an environment variable outranks both files
        monkeypatch.setenv('PIXELTABLE_VAR_MEDIA_DEST', 's3://from/env')
        assert Config.get().get_string_value('media_dest', section=VAR_SECTION) == 's3://from/env'
        monkeypatch.delenv('PIXELTABLE_VAR_MEDIA_DEST')

        # a setting that is not a database binding is project-settable too
        config = load('', "[pixeltable]\ntime_zone = 'America/Anchorage'\n")
        assert config.get_string_value('time_zone') == 'America/Anchorage'

        # the same, declared in a pyproject.toml under [tool.pixeltable]
        (project / 'pixeltable.toml').unlink()
        config = load(
            '',
            dedent(
                """
                [project]
                name = 'proj'

                [[tool.pixeltable.database]]
                vars.media_dest = 's3://from/pyproject'
                """
            ),
            in_pyproject=True,
        )
        assert config.project_config_file == project / 'pyproject.toml'
        assert config.get_string_value('media_dest', section=VAR_SECTION) == 's3://from/pyproject'
        (project / 'pyproject.toml').unlink()

        # a project file cannot set an installation setting; the message points at the home config
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r"Cannot set 'pixeltable.file_cache_size_g'"):
            load('', '[pixeltable]\nfile_cache_size_g = 10.0\n')

        # an edit to the project file is picked up
        config = load('', "[[pixeltable.database]]\nvars.media_dest = 's3://before/edit'\n")
        assert config.get_string_value('media_dest', section=VAR_SECTION) == 's3://before/edit'
        time.sleep(0.01)  # the stamp is (mtime, size), so a same-size rewrite needs a distinct mtime
        (project / 'pixeltable.toml').write_text("[[pixeltable.database]]\nvars.media_dest = 's3://after/edits'\n")
        assert Config.reload_if_changed()
        assert Config.get().get_string_value('media_dest', section=VAR_SECTION) == 's3://after/edits'

    def test_database_entries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bindings a process reads come from the [[pixeltable.database]] entry for the local database."""

        def load(text: str) -> Config:
            config_file = tmp_path / 'config.toml'
            config_file.write_text(text)
            monkeypatch.setenv('PIXELTABLE_CONFIG', str(config_file))
            Config.init(reinit=True)
            return Config.get()

        # one entry per database: the local one binds the vars and secrets, the hosted one carries its image
        config = load(
            dedent(
                """
                [[pixeltable.database]]
                vars.media_dest = 's3://local/bucket'
                secrets.openai_api_key = 'sk-local'

                [[pixeltable.database]]
                name = 'pxt://myorg:prod'
                vars.media_dest = 's3://prod/bucket'
                system_dependencies = ['ffmpeg']
                """
            )
        )
        assert config.get_string_value('media_dest', section=VAR_SECTION) == 's3://local/bucket'
        assert config.get_string_value('openai_api_key', section=SECRET_SECTION) == 'sk-local'
        assert lookup_database_config().vars == {'media_dest': 's3://local/bucket'}
        assert lookup_database_config('pxt://myorg:prod').system_dependencies == ['ffmpeg']
        assert lookup_database_config('pxt://myorg:staging') is None

        # a single table where the array goes is one entry, which is how a file written earlier reads
        config = load("[pixeltable.database]\nvars = { media_dest = 's3://single/bucket' }\n")
        assert config.get_string_value('media_dest', section=VAR_SECTION) == 's3://single/bucket'

        # the environment binds a var the entry does not
        monkeypatch.setenv('PIXELTABLE_VAR_OTHER_DEST', 's3://from/env')
        config = load("[[pixeltable.database]]\nvars.media_dest = 's3://local/bucket'\n")
        assert config.get_string_value('other_dest', section=VAR_SECTION) == 's3://from/env'
        assert config.get_value_source('media_dest', section=VAR_SECTION) == tmp_path / 'config.toml'

        # two entries for one database, which would make the bindings ambiguous
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r"Duplicate `DatabaseConfig` name 'local'"):
            load('[[pixeltable.database]]\n\n[[pixeltable.database]]\n')

        # an entry holding something no database is configured with
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r'Invalid `DatabaseConfig`'):
            load("[[pixeltable.database]]\nnot_a_setting = 'x'\n")

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

            # an override is supplied by the process, so re-reading the file leaves it in place
            Config.init({'openai.api_key': 'sk-override'}, reinit=True)
            assert Config.get().get_string_value('api_key', section='openai') == 'sk-override'
            time.sleep(0.01)
            config_file.write_text('[pixeltable.database.vars]\nmedia_dest = "s3://third/bucket"\n')
            assert Config.reload_if_changed()
            assert media_dest.value() == 's3://third/bucket'
            assert Config.get().get_string_value('api_key', section='openai') == 'sk-override'
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
