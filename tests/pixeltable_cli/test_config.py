"""Tests for supplying credentials and config vars to the daemon, and for what happens when they disagree.

The daemon resolves them once, from the environment it was started with, so these are the workflows a user
goes through when their shell supplies something the daemon does not have.
"""

import os
import pathlib
import subprocess
import time
from collections.abc import Iterator
from textwrap import dedent
from typing import Any

import pytest

import pixeltable as pxt

from ..utils import skip_test_if_not_installed, DatabaseRoot
from .conftest import PxtRunner

pytestmark = pytest.mark.local('the daemon under test is the one serving the in-process catalog')

# values no daemon in this module was started with, and that no output may ever contain
_A_KEY = 'sk-pxt-test-aaaa'
_ANOTHER_KEY = 'sk-pxt-test-bbbb'

# a schema whose index makes pxt schema update load an embedding model, which takes seconds
_SLOW_SCHEMA_SRC = dedent(
    """
    from __future__ import annotations

    import pixeltable as pxt
    from pixeltable.functions.huggingface import sentence_transformer

    TableModel = pxt.model_base()

    embed = sentence_transformer.using(model_id='intfloat/e5-large-v2')


    class Articles(TableModel, name='articles'):
        body: pxt.String
        __indexes__ = [pxt.EmbeddingIndex(body, embedding=embed)]
    """
)


@pytest.fixture(autouse=True)
def daemon_serving_the_test_process_values(cli: PxtRunner) -> Iterator[None]:
    """Leave behind a daemon serving with the values this process's environment holds.

    A test here restarts the daemon with values of its own. A later caller supplies none of them, and a daemon
    still holding them refuses that caller's work.
    """
    yield
    cli('daemon', 'stop', '-f', check=False)
    cli('daemon', 'start')


def make_table(target: str) -> None:
    """Create a one-row table the CLI can read back."""
    pxt.create_dir(target)
    t = pxt.create_table(f'{target}/docs', {'title': pxt.String})
    t.insert([{'title': 'hello'}])


class TestConfig:
    def test_supplying_a_key_the_daemon_lacks(self, cli: PxtRunner, db_root: DatabaseRoot) -> None:
        """A shell binds a credential the daemon has not got: work is refused, inspection is not, a restart fixes it."""
        target = db_root.make_catalog_path('cfg')
        make_table(target)
        with_key = {'OPENAI_API_KEY': _A_KEY}
        without_key: dict[str, str | None] = {'OPENAI_API_KEY': None}
        cli('daemon', 'restart', env_overrides=without_key)

        # work that could compute with the credential is refused, and the refusal says what to do about it
        r = cli('rows', f'{target}/docs', '-n', '1', env_overrides=with_key, check=False)
        assert r.returncode != 0
        assert 'OPENAI_API_KEY' in r.stderr
        assert 'pxt daemon restart' in r.stderr

        # any other work is refused too, without the endpoint having to ask for the check
        r = cli('count', f'{target}/docs', env_overrides=with_key, check=False)
        assert r.returncode != 0
        assert 'OPENAI_API_KEY' in r.stderr

        # the commands needed to diagnose it keep working, and report the difference
        r = cli('config', env_overrides=with_key)
        assert 'OPENAI_API_KEY' in r.stdout
        assert 'pxt daemon restart' in r.stdout
        cli('ls', target, env_overrides=with_key)
        cli('describe', f'{target}/docs', env_overrides=with_key)

        # a caller that binds nothing asked for no particular value, so its work runs
        assert 'hello' in cli('rows', f'{target}/docs', '-n', '1', env_overrides=without_key).stdout

        # adopting the caller's environment takes a restart, after which the same command runs
        cli('daemon', 'restart', env_overrides=with_key)
        assert 'hello' in cli('rows', f'{target}/docs', '-n', '1', env_overrides=with_key).stdout

    def test_rotating_a_key(self, cli: PxtRunner, db_root: DatabaseRoot) -> None:
        """A shell binds a credential to a new value: refused until the daemon is restarted with it."""
        target = db_root.make_catalog_path('cfg')
        make_table(target)
        first = {'OPENAI_API_KEY': _A_KEY}
        second = {'OPENAI_API_KEY': _ANOTHER_KEY}

        cli('daemon', 'restart', env_overrides=first)
        assert 'hello' in cli('rows', f'{target}/docs', '-n', '1', env_overrides=first).stdout

        # the same variable set to something else is a disagreement, not a new value
        r = cli('rows', f'{target}/docs', '-n', '1', env_overrides=second, check=False)
        assert r.returncode != 0
        assert 'OPENAI_API_KEY' in r.stderr
        assert 'set to a different value in the daemon' in cli('config', env_overrides=second).stdout

        cli('daemon', 'restart', env_overrides=second)
        assert 'hello' in cli('rows', f'{target}/docs', '-n', '1', env_overrides=second).stdout

    def test_a_setting_that_is_not_a_credential(self, cli: PxtRunner, db_root: DatabaseRoot) -> None:
        """Any env-settable setting counts, not just credentials: an endpoint decides what the work talks to."""
        target = db_root.make_catalog_path('cfg')
        make_table(target)
        other_endpoint = {'OPENAI_BASE_URL': 'https://example.invalid/v1'}
        cli('daemon', 'restart', env_overrides={'OPENAI_BASE_URL': None})

        r = cli('rows', f'{target}/docs', '-n', '1', env_overrides=other_endpoint, check=False)
        assert r.returncode != 0
        assert 'OPENAI_BASE_URL' in r.stderr
        cli('daemon', 'restart', env_overrides=other_endpoint)
        assert 'hello' in cli('rows', f'{target}/docs', '-n', '1', env_overrides=other_endpoint).stdout

    def test_instance_settings_ignore_the_environment(
        self, cli: PxtRunner, db_root: DatabaseRoot
    ) -> None:
        """A setting every process using the instance shares is read from the file, and says so when exported."""
        target = db_root.make_catalog_path('cfg')
        make_table(target)
        exported = {'PIXELTABLE_FILE_CACHE_SIZE_G': '1'}

        # the file's value is what both this caller and the daemon resolve, so they do not disagree
        entries = cli('config', '--json', env_overrides=exported).json
        cache_size = next(
            e for e in entries['entries'] if (e['section'], e['key']) == ('pixeltable', 'file_cache_size_g')
        )
        assert cache_size['source'] != 'env', cache_size
        assert 'PIXELTABLE_FILE_CACHE_SIZE_G' not in entries['env_var_names']
        assert 'hello' in cli('rows', f'{target}/docs', '-n', '1', env_overrides=exported).stdout

    def test_no_command_prints_a_credential(self, cli: PxtRunner, db_root: DatabaseRoot) -> None:
        """A credential the caller supplies never appears in output, whether the daemon agrees with it or not."""
        target = db_root.make_catalog_path('cfg')
        make_table(target)
        with_key = {'OPENAI_API_KEY': _A_KEY}

        results = [
            cli('config', env_overrides=with_key),
            cli('config', '--json', env_overrides=with_key),
            cli('rows', f'{target}/docs', '-n', '1', env_overrides=with_key, check=False),
        ]
        cli('daemon', 'restart', env_overrides=with_key)
        results.append(cli('config', '--json', env_overrides=with_key))
        results.append(cli('daemon', 'status', '--json', env_overrides=with_key))
        for r in results:
            assert _A_KEY not in r.stdout, r.stdout
            assert _A_KEY not in r.stderr, r.stderr

        # what a caller can see is that the key is set, and where its value came from
        entries = cli('config', '--json', env_overrides=with_key).json['entries']
        openai_key = next(e for e in entries if (e['section'], e['key']) == ('openai', 'api_key'))
        assert (openai_key['value'], openai_key['source']) == ('<redacted>', 'env')

    def test_config_reports_a_var_only_the_environment_sets(self, cli: PxtRunner) -> None:
        """A config var with no config file entry is still reported, with its value withheld if it is a secret."""
        supplied = {'PIXELTABLE_SECRET_PXT_TEST_KEY': _A_KEY, 'PIXELTABLE_VAR_PXT_TEST_DEST': 's3://bucket/prefix'}
        # a PIXELTABLE_* variable the daemon lacks restarts it, so the reported values are the ones supplied here
        resp = cli('config', '--json', env_overrides=supplied).json
        entries = {(e['section'], e['key']): e for e in resp['entries']}

        secret = entries['pixeltable.database.secrets', 'pxt_test_key']
        assert (secret['value'], secret['source']) == ('<redacted>', 'env')
        var = entries['pixeltable.database.vars', 'pxt_test_dest']
        assert (var['value'], var['source']) == ('s3://bucket/prefix', 'env')
        assert 'PIXELTABLE_SECRET_PXT_TEST_KEY' in resp['env_var_names']

    def test_config_var_from_env(
        self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path
    ) -> None:
        """A config var a schema declares is bound from the environment, with no entry in any config file."""
        target = db_root.make_catalog_path('cfg')
        media_dir = project_dir / 'media'
        media_dir.mkdir()
        schema_file = project_dir / 'app.py'
        schema_file.write_text(
            dedent(
                """
                from __future__ import annotations

                import pixeltable as pxt

                MEDIA_DEST = pxt.ConfigVar('pxt_test_dest', pxt.URI)

                TableModel = pxt.model_base()


                class Clips(TableModel, name='clips'):
                    img: pxt.Image | None
                    thumb = pxt.Column(value=img.rotate(90), destination=MEDIA_DEST)
                """
            ),
            encoding='utf-8',
        )

        # the shell binds the var under the name pixeltable derives from the declaration
        bound = {'PIXELTABLE_VAR_PXT_TEST_DEST': media_dir.as_posix()}
        cli('schema', 'update', str(schema_file), target, env_overrides=bound)
        assert cli('schema', 'diff', str(schema_file), target, env_overrides=bound).returncode == 0

        # the schema records the variable, not the location it currently resolves to
        md = pxt.get_table(f'{target}/clips').get_metadata()
        assert md['columns']['thumb']['destination'] == '$pxt_test_dest'
        entries = cli('config', '--json', env_overrides=bound).json['entries']
        dest = next(e for e in entries if e['key'] == 'pxt_test_dest')
        assert (dest['section'], dest['source']) == ('pixeltable.database.vars', 'env')

    def test_config_var_from_project_config(
        self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path
    ) -> None:
        """A var and a secret bound in the project's pixeltable.toml reach the daemon."""
        target = db_root.make_catalog_path('cfg')
        media_dir = project_dir / 'media'
        media_dir.mkdir()
        schema_file = project_dir / 'app.py'
        schema_file.write_text(
            dedent(
                """
                from __future__ import annotations

                import pixeltable as pxt

                MEDIA_DEST = pxt.ConfigVar('pxt_proj_dest', pxt.URI)

                TableModel = pxt.model_base()


                class Clips(TableModel, name='clips'):
                    img: pxt.Image | None
                    thumb = pxt.Column(value=img.rotate(90), destination=MEDIA_DEST)
                """
            ),
            encoding='utf-8',
        )

        # the project binds the var, with nothing in the environment and nothing in the home config
        project_config = project_dir.parent / 'pixeltable.toml'
        original = project_config.read_text(encoding='utf-8')
        project_config.write_text(
            f"{original}\n[[pixeltable.database]]\nvars.pxt_proj_dest = '{media_dir.as_posix()}'\n"
            "secrets.pxt_proj_key = 'from-the-project'\n",
            encoding='utf-8',
        )
        try:
            cli('daemon', 'restart')  # the daemon read the project config when it started
            cli('schema', 'update', str(schema_file), target)
            assert cli('schema', 'diff', str(schema_file), target).returncode == 0

            # the column records the variable, and pxt config names the project file as its source
            md = pxt.get_table(f'{target}/clips').get_metadata()
            assert md['columns']['thumb']['destination'] == '$pxt_proj_dest'
            entries = cli('config', '--json').json['entries']
            var = next(e for e in entries if e['key'] == 'pxt_proj_dest')
            assert (var['section'], var['source']) == ('pixeltable.database.vars', str(project_config))
            secret = next(e for e in entries if e['key'] == 'pxt_proj_key')
            assert (secret['section'], secret['source']) == ('pixeltable.database.secrets', str(project_config))
        finally:
            project_config.write_text(original, encoding='utf-8')
            cli('daemon', 'restart')

    def test_changing_a_value_in_the_config_file(
        self, cli: PxtRunner, db_root: DatabaseRoot, tmp_path: pathlib.Path
    ) -> None:
        """Editing a credential in the config file of a running daemon is refused until it is restarted."""
        target = db_root.make_catalog_path('cfg')
        make_table(target)

        # the daemon reads the file PIXELTABLE_CONFIG names, and picks up a change of that name by itself
        config_file = tmp_path / 'config.toml'
        config_file.write_text('[pixeltable]\nfile_cache_size_g = 1.0\n', encoding='utf-8')
        own_config = {'PIXELTABLE_CONFIG': str(config_file)}
        assert cli('config', '--json', env_overrides=own_config).json['config_file'] == str(config_file)
        assert 'hello' in cli('rows', f'{target}/docs', '-n', '1', env_overrides=own_config).stdout

        time.sleep(0.01)  # the file stamp is (mtime, size), so a rewrite needs a distinct mtime
        config_file.write_text(
            f'[pixeltable]\nfile_cache_size_g = 1.0\n\n[pixeltable.database.secrets]\npxt_test_key = "{_A_KEY}"\n',
            encoding='utf-8',
        )

        r = cli('rows', f'{target}/docs', '-n', '1', env_overrides=own_config, check=False)
        assert r.returncode != 0
        assert 'PIXELTABLE_SECRET_PXT_TEST_KEY' in r.stderr
        assert 'pxt daemon restart' in r.stderr
        assert _A_KEY not in r.stderr

        # the restarted daemon serves with what the file now says
        cli('daemon', 'restart', env_overrides=own_config)
        assert 'hello' in cli('rows', f'{target}/docs', '-n', '1', env_overrides=own_config).stdout
        entries = cli('config', '--json', env_overrides=own_config).json['entries']
        test_key = next(
            e for e in entries if (e['section'], e['key']) == ('pixeltable.database.secrets', 'pxt_test_key')
        )
        assert test_key['source'] == str(config_file)

    def test_an_unparseable_config_file_is_reported(
        self, cli: PxtRunner, db_root: DatabaseRoot, tmp_path: pathlib.Path
    ) -> None:
        """A config file that stops parsing under a running daemon produces an error, not a dropped request."""
        target = db_root.make_catalog_path('cfg')
        make_table(target)
        config_file = tmp_path / 'config.toml'
        config_file.write_text('[pixeltable]\nfile_cache_size_g = 1.0\n', encoding='utf-8')
        own_config = {'PIXELTABLE_CONFIG': str(config_file)}
        assert 'hello' in cli('rows', f'{target}/docs', '-n', '1', env_overrides=own_config).stdout

        time.sleep(0.01)
        config_file.write_text('[pixeltable]\nfile_cache_size_g = 1.0\n[pixeltable]\n', encoding='utf-8')
        r = cli('rows', f'{target}/docs', '-n', '1', env_overrides=own_config, check=False)
        assert r.returncode != 0
        assert str(config_file) in r.stderr
        assert 'RemoteDisconnected' not in r.stderr

    def test_restart_while_serving(
        self, cli: PxtRunner, db_root: DatabaseRoot, project_dir: pathlib.Path
    ) -> None:
        """A restart that would abandon work in progress is refused; once the work is done it goes through."""
        skip_test_if_not_installed('sentence_transformers')
        target = db_root.make_catalog_path('cfg')
        schema_file = project_dir / 'slow.py'
        schema_file.write_text(_SLOW_SCHEMA_SRC, encoding='utf-8')

        # PXT_PORT addresses the daemon under test; the cli fixture put it in this process's environment
        slow = subprocess.Popen(
            ['pxt', 'schema', 'update', str(schema_file), target],
            env={**os.environ, 'BROWSER': 'true'},
            # a client outside the daemon's project root restarts it, so run where the schema file is
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        try:
            in_flight: list[dict[str, Any]] = []
            deadline = time.time() + 60.0
            while time.time() < deadline and slow.poll() is None:
                # check=False: a daemon busy with the update may answer /health too late to be reported
                r = cli('daemon', 'status', '--json', check=False)
                in_flight = r.json['in_flight'] if r.returncode == 0 else []
                if len(in_flight) > 0:
                    break
                time.sleep(0.05)
            if len(in_flight) == 0:
                pytest.skip('the schema update finished before its request could be observed')

            # the daemon names the request it is serving, and refuses to be taken down under it
            serving = in_flight[0]['path']
            assert serving.startswith('/api/schema/'), in_flight
            r = cli('daemon', 'restart', check=False)
            assert r.returncode != 0
            assert serving in r.stderr
            assert '--force' in r.stderr
        finally:
            out = slow.communicate(timeout=600)[0]
            assert slow.returncode == 0, out

        # with nothing in flight the restart goes through
        cli('daemon', 'restart')
        assert cli('daemon', 'status', '--json').json['in_flight'] == []
