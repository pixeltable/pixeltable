"""Tests for the project archive: what a pod fetches and an image build installs from."""

from __future__ import annotations

import logging
import os
import tarfile
import textwrap
from pathlib import Path
from typing import Any

import pytest

import pixeltable

from pixeltable import exceptions as excs
from pixeltable.catalog import Path as PxtPath
from pixeltable.config import Config, DatabaseConfig
from pixeltable.service.db import db_update
from pixeltable.service.management_protocol import (
    DatabaseReport,
    ExpectedGenerations,
    PrepareUpdateRequest,
    PrepareUpdateResponse,
)
from pixeltable.utils.project import (
    create_image_context,
    image_input_files,
    create_project_archive,
    package_image_context,
    package_project_archive,
    project_fingerprint,
    unpacked_digest,
)
from pixeltable_cli.types import DbPlan

from ..utils import pxt_raises


def local_entry() -> DatabaseConfig | None:
    """The project's entry for the local database, which selects the files an archive holds."""
    return Config.get().get_database_config(PxtPath.parse('', allow_empty_path=True))


class TestProjectArchive:
    """What a packager writes, and the hash it reports for every file.

    A hash describes the package rather than a later reading of the project, so an artifact always
    matches its recorded digest.
    """

    def test_archive_layout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every project file is packaged under project/, and nothing else is."""
        monkeypatch.chdir(tmp_path)
        Config.init(reinit=True)

        # newline='\n' so the write isn't translated to CRLF on Windows (the content assertion below is exact)
        (tmp_path / 'udfs.py').write_text('import pixeltable as pxt\n', newline='\n')
        (tmp_path / 'subdir').mkdir()
        (tmp_path / 'subdir' / 'helper.py').write_text('# helper\n')

        archive_path = create_project_archive(tmp_path)

        with tarfile.open(archive_path, 'r:bz2') as tar:
            assert sorted(tar.getnames()) == ['project/subdir/helper.py', 'project/udfs.py']
            with tar.extractfile(tar.getmember('project/udfs.py')) as f:
                assert f.read().decode() == 'import pixeltable as pxt\n'

    def test_include_exclude(self, tmp_path: Path) -> None:
        (tmp_path / 'a_include.txt').write_text('# included by default')
        (tmp_path / 'b_exclude.txt').write_text('# excluded explicitly')
        (tmp_path / 'a_include.py').write_text('# exclude-then-include')
        (tmp_path / 'a_exclude.py').write_text('# excluded')

        (tmp_path / 'pixeltable.toml').write_text(
            textwrap.dedent("""\
                [[pixeltable.database]]
                exclude = ["*.py", "b_exclude.txt"]
                include = ["a_include.py"]
            """)
        )

        Config.init(reinit=True, project_root=tmp_path)

        archive_path = create_project_archive(tmp_path, local_entry())

        with tarfile.open(archive_path, 'r:bz2') as tar:
            members = tar.getnames()
            assert 'project/a_include.txt' in members
            assert 'project/a_include.py' in members
            assert 'project/pixeltable.toml' in members
            assert 'project/a_exclude.py' not in members
            assert 'project/b_exclude.txt' not in members

    def test_gitignore_respected(self, tmp_path: Path) -> None:
        """Files matching .gitignore patterns are excluded from project/."""
        (tmp_path / '.gitignore').write_text('__pycache__/\n*.pyc\n.env\n')
        (tmp_path / 'app.py').write_text('# app')
        (tmp_path / '__pycache__').mkdir()
        (tmp_path / '__pycache__' / 'app.cpython-311.pyc').write_bytes(b'\x00')
        (tmp_path / '.env').write_text('SECRET=abc')

        Config.init(reinit=True)

        archive_path = create_project_archive(tmp_path)

        with tarfile.open(archive_path, 'r:bz2') as tar:
            members = tar.getnames()
            assert 'project/app.py' in members
            assert not any('__pycache__' in m for m in members)
            assert 'project/.env' not in members

    def test_uv_lock_included(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """uv.lock in the project dir is included under project/ for server-side uv sync."""
        monkeypatch.chdir(tmp_path)
        Config.init(reinit=True)

        (tmp_path / 'uv.lock').write_text('version = 1\n')
        (tmp_path / 'pyproject.toml').write_text('[project]\nname = "app"\n')

        archive_path = create_project_archive(tmp_path)

        with tarfile.open(archive_path, 'r:bz2') as tar:
            assert 'project/uv.lock' in tar.getnames()
            assert 'project/pyproject.toml' in tar.getnames()
            # no root-level requirements.txt or runtime_config.json
            assert 'requirements.txt' not in tar.getnames()
            assert 'runtime_config.json' not in tar.getnames()

    def test_single_table_config_form(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """[pixeltable.database] written as one table, which a project predating the array form uses."""
        (tmp_path / 'pixeltable.toml').write_text(
            textwrap.dedent("""\
                [pixeltable.database]
                system_dependencies = ["ffmpeg", "libpq"]
            """)
        )
        monkeypatch.chdir(tmp_path)
        Config.init(reinit=True)

        entry = local_entry()
        assert entry is not None
        assert entry.system_dependencies == ['ffmpeg', 'libpq']

    def test_is_valid_bz2_tar(self, tmp_path: Path) -> None:
        """The output file is a valid bz2 tarball."""
        archive_path = create_project_archive(tmp_path)
        assert tarfile.is_tarfile(archive_path)
        assert archive_path.suffix == '.bz2'

    def test_no_lockfile(self, init_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
        """A project with no lockfile draws a warning and is packaged anyway; a conda environment is not a lockfile."""
        monkeypatch.chdir(tmp_path)
        Config.init(reinit=True)
        (tmp_path / 'udfs.py').write_text('import pixeltable as pxt\n')
        monkeypatch.setenv('CONDA_PREFIX', str(tmp_path / 'envs' / 'pxt'))

        # the console logger writes to the stream it was built with, which no capture fixture owns
        monkeypatch.setattr(logging.getLogger('pixeltable'), 'propagate', True)
        with caplog.at_level(logging.WARNING, logger='pixeltable'):
            archive_path = create_project_archive(tmp_path)
        assert 'No dependency lockfile' in caplog.text

        with tarfile.open(archive_path, 'r:bz2') as tar:
            members = tar.getnames()
        assert 'project/udfs.py' in members
        assert all('conda' not in member for member in members)

    def test_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A project directory that does not exist, and a project config file Config cannot use."""
        with pytest.raises(FileNotFoundError, match='does not exist'):
            create_project_archive(Path('/nonexistent/path/xyz'))

        # Config reads the project's config file, so an unusable entry is refused before deploy sees it
        (tmp_path / 'pixeltable.toml').write_text(
            textwrap.dedent("""\
                [[pixeltable.database]]
                include = "not-a-list"
            """)
        )
        monkeypatch.chdir(tmp_path)
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=r'Invalid `DatabaseConfig`'):
            Config.init(reinit=True)

    def test_archive_matches_fingerprint(self, tmp_path: Path) -> None:
        """An unchanged project fingerprints to what packaging it produces, or an upload could never match."""
        (tmp_path / 'app.py').write_text('x = 1\n')
        (tmp_path / 'sub').mkdir()
        (tmp_path / 'sub' / 'mod.py').write_text('y = 2\n')

        assert package_project_archive(tmp_path).files == project_fingerprint(tmp_path, None).files

    def test_archive_rewrite(self, tmp_path: Path) -> None:
        """This is what a concurrent writer looks like: same path, different bytes in the archive."""
        (tmp_path / 'app.py').write_text('x = 1\n')
        before = package_project_archive(tmp_path).files
        (tmp_path / 'app.py').write_text('x = 2\n')

        assert package_project_archive(tmp_path).files != before

    def test_context_matches_fingerprint(self, tmp_path: Path) -> None:
        """An unchanged project fingerprints to the context it packages, or an upload could never match."""
        (tmp_path / 'app.py').write_text('x = 1\n')
        (tmp_path / 'requirements.txt').write_text('pandas\n')
        (tmp_path / 'pyproject.toml').write_text('[project]\nname = "app"\n')

        assert package_image_context(tmp_path).files == project_fingerprint(tmp_path, None).image_files()

    def test_context_rewrite(self, tmp_path: Path) -> None:
        """The manifests name the dependencies, so a rewrite of one has to reach the context's hashes."""
        (tmp_path / 'requirements.txt').write_text('pandas\n')
        before = package_image_context(tmp_path).files
        (tmp_path / 'requirements.txt').write_text('numpy\n')

        assert package_image_context(tmp_path).files != before

    def test_context_layout(self, tmp_path: Path) -> None:
        """The context holds the manifests an install reads, at its root, and none of the project's source."""
        (tmp_path / 'app.py').write_text('import pixeltable as pxt\n')
        (tmp_path / 'uv.lock').write_text('version = 1\n')
        (tmp_path / 'pyproject.toml').write_text('[project]\nname = "app"\n')

        with tarfile.open(create_image_context(tmp_path)) as tar:
            assert sorted(tar.getnames()) == ['project/pyproject.toml', 'project/uv.lock']

    def test_no_manifests(self, tmp_path: Path) -> None:
        """A project declaring no dependencies still produces a context, so a build always has one input."""
        (tmp_path / 'app.py').write_text('import pixeltable as pxt\n')

        with tarfile.open(create_image_context(tmp_path)) as tar:
            assert tar.getnames() == []

    def test_refuses_external_manifests(self, tmp_path: Path) -> None:
        """Refuse a requirements.txt that reads another file, which the context does not hold."""
        (tmp_path / 'requirements.txt').write_text('-r base.txt\npixeltable\n')
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='reads another file'):
            create_image_context(tmp_path)

        # the context holds the wheel under a relative name, so an absolute one names nothing in the build
        wheel = tmp_path / 'w' / 'pkg-1.0-py3-none-any.whl'
        wheel.parent.mkdir()
        wheel.write_bytes(b'')
        (tmp_path / 'requirements.txt').write_text(f'{wheel}\n')
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='an absolute path naming this machine'):
            create_image_context(tmp_path)

    def test_requirement_spellings(self, tmp_path: Path) -> None:
        """PEP 508 makes the whitespace around '@' optional, and a filename may contain an '@' too."""
        wheel = tmp_path / 'w' / 'pkg-1.0-py3-none-any.whl'
        wheel.parent.mkdir()
        wheel.write_bytes(b'wheel bytes')

        for line in ('pkg @ w/pkg-1.0-py3-none-any.whl', 'pkg@w/pkg-1.0-py3-none-any.whl'):
            (tmp_path / 'requirements.txt').write_text(f'{line}\n')
            assert 'w/pkg-1.0-py3-none-any.whl' in package_image_context(tmp_path).files, line
            (tmp_path / 'requirements.txt').write_text(f'{line.replace("w/", "file:///w/")}\n')
            with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='a file: url naming this machine'):
                create_image_context(tmp_path)

        # no package name stands before the '@', so the whole line is the path
        (tmp_path / 'w' / 'pkg@1.0.whl').write_bytes(b'other bytes')
        (tmp_path / 'requirements.txt').write_text('w/pkg@1.0.whl\n')
        assert 'w/pkg@1.0.whl' in package_image_context(tmp_path).files

        (tmp_path / 'requirements.txt').write_text(f'pkg @ file://{wheel}\n')
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='a file: url naming this machine'):
            create_image_context(tmp_path)

        # the same wheel, named relative to the project, is bundled
        (tmp_path / 'requirements.txt').write_text('w/pkg-1.0-py3-none-any.whl\n')
        with tarfile.open(create_image_context(tmp_path)) as tar:
            assert sorted(tar.getnames()) == ['project/requirements.txt', 'project/w/pkg-1.0-py3-none-any.whl']

        # an environment marker decides whether pip installs the line, not where the file is
        (tmp_path / 'requirements.txt').write_text('w/pkg-1.0-py3-none-any.whl ; python_version >= "3.11"\n')
        with tarfile.open(create_image_context(tmp_path)) as tar:
            assert sorted(tar.getnames()) == ['project/requirements.txt', 'project/w/pkg-1.0-py3-none-any.whl']

        # requirements.txt travels unchanged, so pip would look for a path the project does not hold
        (tmp_path / 'requirements.txt').write_text('w/typo-1.0-py3-none-any.whl\n')
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='cannot be resolved relative to the project root'):
            create_image_context(tmp_path)

        # a path above the project is named as such, not reported as missing
        (tmp_path / 'requirements.txt').write_text('../outside/pkg.whl\n')
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='which is outside the project'):
            create_image_context(tmp_path)

        # pip installs a bare archive name from the project root, so it is a path rather than a package
        root_wheel = tmp_path / 'pkg-2.0-py3-none-any.whl'
        root_wheel.write_bytes(b'')
        (tmp_path / 'requirements.txt').write_text('pkg-2.0-py3-none-any.whl\n')
        with tarfile.open(create_image_context(tmp_path)) as tar:
            assert sorted(tar.getnames()) == ['project/pkg-2.0-py3-none-any.whl', 'project/requirements.txt']

        # a bare name with no archive suffix stays a package the index serves
        (tmp_path / 'requirements.txt').write_text('pixeltable\n')
        with tarfile.open(create_image_context(tmp_path)) as tar:
            assert tar.getnames() == ['project/requirements.txt']

    def test_find_links(self, tmp_path: Path) -> None:
        """--find-links names where to look for packages, and a directory here is one only this machine has."""
        for line in ('-f ./wheels', '-f./wheels', '--find-links ./wheels', '--find-links=./wheels', '-f file:///w'):
            (tmp_path / 'requirements.txt').write_text(f'{line}\npixeltable\n')
            with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='a location on this machine'):
                create_image_context(tmp_path)

        # an index the build can reach is not a local location, and neither is another option
        (tmp_path / 'requirements.txt').write_text('--find-links https://example.com/wheels\n--no-index\npixeltable\n')
        with tarfile.open(create_image_context(tmp_path)) as tar:
            assert tar.getnames() == ['project/requirements.txt']

        # uv reads its own find-links from pyproject, and a relative directory there is just as local
        (tmp_path / 'requirements.txt').unlink()
        (tmp_path / 'pyproject.toml').write_text('[tool.uv]\nfind-links = ["./wheels"]\n')
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='a location on this machine'):
            create_image_context(tmp_path)

    def test_requirement_continuations(self, tmp_path: Path) -> None:
        """pip joins a backslash continuation before reading the requirement, so the path spans two lines."""
        wheel = tmp_path / 'w' / 'pkg-1.0-py3-none-any.whl'
        wheel.parent.mkdir()
        wheel.write_bytes(b'wheel bytes')
        (tmp_path / 'requirements.txt').write_text('pkg @ \\\n    w/pkg-1.0-py3-none-any.whl\n')

        assert 'w/pkg-1.0-py3-none-any.whl' in package_image_context(tmp_path).files

    def test_lock_sources(self, tmp_path: Path) -> None:
        """`uv sync --frozen` reads a path source from the context, so one inside the project is carried."""
        pkg = tmp_path / 'packages' / 'helper'
        (pkg / 'src').mkdir(parents=True)
        (pkg / 'pyproject.toml').write_text('[project]\nname = "helper"\n')
        (pkg / 'src' / 'helper.py').write_text('X = 1\n')
        (tmp_path / 'uv.lock').write_text(
            '[[package]]\nname = "helper"\nsource = { editable = "packages/helper" }\n'
            # the project's own package, and one the context cannot reach
            '[[package]]\nname = "app"\nsource = { editable = "." }\n'
            '[[package]]\nname = "outside"\nsource = { directory = "../elsewhere" }\n'
        )
        # a virtual environment inside the package is not part of the package
        (pkg / '.venv' / 'lib').mkdir(parents=True)
        (pkg / '.venv' / 'pyvenv.cfg').write_text('home = /usr/bin\n')
        (pkg / '.venv' / 'lib' / 'site.py').write_text('Y = 2\n')
        files = package_image_context(tmp_path).files
        assert sorted(files) == ['packages/helper/pyproject.toml', 'packages/helper/src/helper.py', 'uv.lock']
        assert not any('.venv' in path for path in package_project_archive(tmp_path).files)

    def test_executable_bit(self, tmp_path: Path) -> None:
        """A mode change leaves the project alone: a pod imports its files and runs none of them.

        Windows cannot set an execute bit, so a digest covering one would differ between the platform a
        project is packaged on and the platform it is fingerprinted on.
        """
        script = tmp_path / 'run.sh'
        script.write_text('echo hi\n')
        before = package_project_archive(tmp_path).files['run.sh']

        script.chmod(script.stat().st_mode | 0o111)
        after = package_project_archive(tmp_path).files['run.sh']
        assert after == before
        assert project_fingerprint(tmp_path, None).files['run.sh'] == after

    def test_hard_link(self, tmp_path: Path) -> None:
        """A second path to one inode is a tar hard link, and extracts as a regular file holding its bytes."""
        (tmp_path / 'a.txt').write_text('same bytes\n')
        os.link(tmp_path / 'a.txt', tmp_path / 'b.txt')
        packaged = package_project_archive(tmp_path)

        assert packaged.files['b.txt'] == packaged.files['a.txt'], 'both paths hold the same file'
        assert packaged.files == project_fingerprint(tmp_path, None).files

    def test_symlinked_requirement(self, tmp_path: Path) -> None:
        """pip reads the path as spelled, so a requirement reached through a symlink is absent from the context."""
        (tmp_path / 'artifacts').mkdir()
        (tmp_path / 'artifacts' / 'dep.whl').write_bytes(b'wheel bytes')
        (tmp_path / 'w').mkdir()
        (tmp_path / 'w' / 'dep.whl').symlink_to(tmp_path / 'artifacts' / 'dep.whl')
        (tmp_path / 'requirements.txt').write_text('w/dep.whl\n')

        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='through a symlink'):
            create_image_context(tmp_path)

    def test_symlink_retarget(self, tmp_path: Path) -> None:
        """A symlink holds a path, so pointing it at equal bytes elsewhere still changes the project."""
        (tmp_path / 'a.txt').write_text('same\n')
        (tmp_path / 'b.txt').write_text('same\n')
        link = tmp_path / 'link.txt'
        link.symlink_to('a.txt')
        before = package_project_archive(tmp_path).files['link.txt']

        link.unlink()
        link.symlink_to('b.txt')
        after = package_project_archive(tmp_path).files['link.txt']
        assert after != before, 'the two targets hold the same bytes, so only the link itself differs'
        assert project_fingerprint(tmp_path, None).files['link.txt'] == after

    def test_unpacked_symlink(self, tmp_path: Path) -> None:
        """A symlink unpacks broken where the archive omits its target, and still belongs to the project."""
        unpacked = tmp_path / 'unpacked'
        unpacked.mkdir()
        (unpacked / 'app.py').write_text('x = 1\n')
        (unpacked / 'link.txt').symlink_to('absent.txt')
        with_link = unpacked_digest(unpacked)

        (unpacked / 'link.txt').unlink()
        assert unpacked_digest(unpacked) != with_link, 'the archive named this link too'

    def test_reproducible(self, tmp_path: Path) -> None:
        """Packaging the same files again writes the same bytes, whatever their timestamps and owners say."""
        (tmp_path / 'app.py').write_text('x = 1\n')
        (tmp_path / 'run.sh').write_text('echo hi\n')
        (tmp_path / 'run.sh').chmod(0o755)
        first = package_project_archive(tmp_path).path.read_bytes()

        os.utime(tmp_path / 'app.py', (1_000_000, 1_000_000))
        (tmp_path / 'run.sh').chmod(0o775)
        assert package_project_archive(tmp_path).path.read_bytes() == first

    def test_image_inputs_survive_exclude(self, tmp_path: Path) -> None:
        """The archive carries what the image build installs from, even where the entry excludes it."""
        (tmp_path / 'dist').mkdir()
        (tmp_path / 'dist' / 'dep.whl').write_bytes(b'wheel bytes')
        (tmp_path / 'requirements.txt').write_text('dist/dep.whl\n')
        (tmp_path / 'pixeltable.toml').write_text('[[pixeltable.database]]\nexclude = ["dist/*"]\n')
        Config.init(reinit=True, project_root=tmp_path)

        with tarfile.open(create_project_archive(tmp_path, local_entry()), 'r:bz2') as tar:
            assert 'project/dist/dep.whl' in tar.getnames()

    def test_manifest_drift(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A file rewritten while it is packaged is caught before anything is uploaded or submitted."""
        (tmp_path / 'pixeltable.toml').write_text(
            '[[pixeltable.database]]\nname = "pxt://acme:main"\n', encoding='utf-8'
        )
        (tmp_path / 'app.py').write_text('x = 1\n')
        (tmp_path / 'requirements.txt').write_text('pandas\n')
        Config.init(reinit=True, project_root=tmp_path)

        # the archive caught the rewrite; the fingerprint read the file after the writer restored it
        drifted = package_project_archive(tmp_path, local_entry())
        drifted.files['requirements.txt'] = 'rewritten-while-packaging'
        monkeypatch.setattr('pixeltable.service.db.package_project_archive', lambda *a, **k: drifted)
        monkeypatch.setattr(
            'urllib.request.urlopen', lambda *a, **k: pytest.fail('an archive was uploaded before validation')
        )

        plan = DbPlan(db_uri='pxt://acme:main', exists=True, state='AVAILABLE', resolution='update_additive')

        def api_call(request: Any) -> dict[str, Any]:
            assert isinstance(request, PrepareUpdateRequest), f'{request.operation_type} was sent before validation'
            assert request.blob is None, 'an archive was offered before validation'
            return PrepareUpdateResponse(
                generations=ExpectedGenerations(database=1), report=DatabaseReport(), plan=plan
            ).model_dump(mode='json')

        monkeypatch.setattr('pixeltable.service.db.management_client.api_call', api_call)

        with pxt_raises(excs.ErrorCode.INVALID_STATE, match='requirements.txt'):
            db_update('pxt://acme:main')


def _write(project: Path, name: str, content: str) -> None:
    """Write a manifest, and the wheel a requirements.txt names, which the archive must carry."""
    (project / name).write_text(content)
    for line in content.splitlines():
        if line.endswith('.whl'):
            wheel = project / line
            wheel.parent.mkdir(parents=True, exist_ok=True)
            wheel.write_bytes(b'')


class TestLockedPixeltable:
    """A hosted image runs the pixeltable of the CLI that deploys it, so a lockfile pinning another is refused."""

    @pytest.fixture(autouse=True)
    def _version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pixeltable, '__version__', '0.7.15.dev12+8b86426f')

    @pytest.mark.parametrize(
        ('name', 'content'),
        [
            ('uv.lock', '[[package]]\nname = "pixeltable"\nversion = "0.7.15.dev12+8b86426f"\n'),
            ('requirements.txt', 'pixeltable @ git+https://github.com/pixeltable/pixeltable@8b86426f930cb3dd\n'),
            ('requirements.txt', 'pixeltable>=0.7\n'),
            ('requirements.txt', 'numpy\n'),
            ('requirements.txt', './wheels/pixeltable-0.7.15.dev12+8b86426f-py3-none-any.whl\n'),
            ('uv.lock', '[[package]]\nname = "pixeltable"\nsource = { editable = "." }\n'),
        ],
        ids=['uv-same', 'git-same-commit', 'unpinned', 'absent', 'wheel', 'editable-project'],
    )
    def test_accepted(self, tmp_path: Path, name: str, content: str) -> None:
        _write(tmp_path, name, content)
        image_input_files(tmp_path)

    @pytest.mark.parametrize(
        ('name', 'content'),
        [
            ('uv.lock', '[[package]]\nname = "pixeltable"\nversion = "0.7.14"\n'),
            ('requirements.txt', 'pixeltable==0.7.14\n'),
            ('requirements.txt', './wheels/pixeltable-0.7.14-py3-none-any.whl\n'),
            ('requirements.txt', 'pixeltable @ git+https://github.com/pixeltable/pixeltable@deadbeef\n'),
        ],
        ids=['uv', 'requirements', 'wheel', 'git'],
    )
    def test_refused(self, tmp_path: Path, name: str, content: str) -> None:
        _write(tmp_path, name, content)
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match='locks pixeltable'):
            image_input_files(tmp_path)
