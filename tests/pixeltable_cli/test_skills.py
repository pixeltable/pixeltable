"""Tests for 'pxt skills install', against a local archive in place of GitHub's."""

import io
import json
import pathlib
import tarfile

import pytest
import requests

from pixeltable_cli.client import utils
from pixeltable_cli.client.commands import skills

_COMMIT = 'f976f182ded0a9c7d06d219a63559cd13535c7e1'

_SKILL = {
    'SKILL.md': b'---\nname: pixeltable\n---\nBuild with Pixeltable.\n',
    'references/cli.md': b'# pxt\n',
    'agents/openai.yaml': b'interface: {}\n',
}


def _archive(
    files: dict[str, bytes], *, links: tuple[tuple[str, str], ...] = (), commit: str | None = _COMMIT
) -> bytes:
    """A tar.gz laid out like GitHub's archive of the skill repository."""
    buf = io.BytesIO()
    pax_headers = {'comment': commit} if commit is not None else {}
    with tarfile.open(fileobj=buf, mode='w:gz', format=tarfile.PAX_FORMAT, pax_headers=pax_headers) as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(f'pixeltable-skill-main/{name}')
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        for name, target in links:
            info = tarfile.TarInfo(f'pixeltable-skill-main/{name}')
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return buf.getvalue()


def _repo_archive(skill: dict[str, bytes] = _SKILL) -> bytes:
    return _archive(
        {
            'plugin.json': b'{"name": "pixeltable", "version": "2.12.0"}',
            'README.md': b'# Pixeltable skill\n',
            **{f'skills/pixeltable-skill/{name}': content for name, content in skill.items()},
        }
    )


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass


@pytest.fixture
def project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """An empty directory to install into, with no terminal to answer a prompt."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, 'stdin_is_a_tty', lambda: False)
    return tmp_path


def _serve(monkeypatch: pytest.MonkeyPatch, archive: bytes) -> list[str]:
    """Answer the download with archive; returns the URLs requested."""
    requested: list[str] = []

    def get(url: str, timeout: float) -> _Response:
        requested.append(url)
        return _Response(archive)

    monkeypatch.setattr(requests, 'get', get)
    return requested


def _tree(root: pathlib.Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


class TestSkillsInstall:
    def test_installs_for_each_agent_dir(
        self, project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A fresh directory gets the skill's files, and nothing else from the repository, in both skill dirs."""
        requested = _serve(monkeypatch, _repo_archive())
        skills.run(['install'])
        assert requested == [skills.ARCHIVE_URL]
        for skills_dir in ('.claude/skills', '.agents/skills'):
            assert _tree(project / skills_dir / 'pixeltable') == _SKILL
        # nothing left behind from writing beside the target
        assert sorted(p.name for p in (project / '.claude/skills').iterdir()) == ['pixeltable']

        out = capsys.readouterr().out
        assert 'version 2.12.0, commit f976f182d' in out
        assert out.count('written') == 2
        assert 'Start a new agent session' in out

    def test_matching_copy_is_left_alone(
        self, project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _serve(monkeypatch, _repo_archive())
        skills.run(['install'])
        capsys.readouterr()

        skills.run(['install', '--json'])
        assert json.loads(capsys.readouterr().out) == {
            'source': skills.ARCHIVE_URL,
            'version': '2.12.0',
            'commit': _COMMIT,
            'installed': ['.claude/skills/pixeltable', '.agents/skills/pixeltable'],
            'written': [],
        }

    def test_differing_copy_needs_force(
        self, project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A copy that differs is refused without -f and left as it was; -f replaces it and drops stale files."""
        _serve(monkeypatch, _repo_archive())
        skills.run(['install'])
        edited = project / '.agents/skills/pixeltable'
        (edited / 'SKILL.md').write_bytes(b'edited\n')
        (edited / 'stale.md').write_bytes(b'from an older skill\n')
        before = _tree(edited)
        capsys.readouterr()

        with pytest.raises(SystemExit) as exc:
            skills.run(['install'])
        assert exc.value.code == utils.EXIT_REFUSED
        err = capsys.readouterr().err
        assert '--force/-f' in err and '.agents/skills/pixeltable' in err
        assert '.claude/skills/pixeltable' not in err
        assert _tree(edited) == before

        skills.run(['install', '-f', '--json'])
        assert json.loads(capsys.readouterr().out)['written'] == ['.agents/skills/pixeltable']
        assert _tree(edited) == _SKILL

    def test_link_at_target_is_replaced_not_followed(
        self, project: pathlib.Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elsewhere = tmp_path_factory.mktemp('elsewhere')
        (elsewhere / 'keep.md').write_bytes(b'not ours\n')
        (project / '.claude/skills').mkdir(parents=True)
        (project / '.claude/skills/pixeltable').symlink_to(elsewhere, target_is_directory=True)
        _serve(monkeypatch, _repo_archive())

        skills.run(['install', '-f'])
        target = project / '.claude/skills/pixeltable'
        assert not target.is_symlink()
        assert _tree(target) == _SKILL
        assert _tree(elsewhere) == {'keep.md': b'not ours\n'}

    def test_link_inside_copy_makes_it_differ(
        self,
        project: pathlib.Path,
        tmp_path_factory: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A link inside an otherwise matching copy is not current: it is refused without -f, and -f removes it."""
        outside = tmp_path_factory.mktemp('outside') / 'notes.md'
        outside.write_bytes(b'not ours\n')
        _serve(monkeypatch, _repo_archive())
        skills.run(['install'])
        link = project / '.agents/skills/pixeltable/references/extra.md'
        link.symlink_to(outside)
        capsys.readouterr()

        with pytest.raises(SystemExit) as exc:
            skills.run(['install'])
        assert exc.value.code == utils.EXIT_REFUSED
        assert '.agents/skills/pixeltable' in capsys.readouterr().err

        skills.run(['install', '-f'])
        assert not link.is_symlink() and not link.exists()
        assert _tree(project / '.agents/skills/pixeltable') == _SKILL
        assert outside.read_bytes() == b'not ours\n'

    def test_unsafe_members_are_skipped(self) -> None:
        """Links and paths that climb out of the skill's directory are not taken from the archive."""
        archive = _archive(
            {
                'skills/pixeltable-skill/SKILL.md': b'---\nname: pixeltable\n---\n',
                'skills/pixeltable-skill/../../../escaped.md': b'outside\n',
                'skills/pixeltable-skill/C:evil.md': b'drive\n',
            },
            links=(('skills/pixeltable-skill/passwd', '/etc/passwd'),),
            commit=None,
        )
        files, commit, version = skills.unpack(archive)
        assert files == {'SKILL.md': b'---\nname: pixeltable\n---\n'}
        assert commit is None and version is None

    def test_download_failure(
        self, project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def get(url: str, timeout: float) -> _Response:
            raise requests.ConnectionError('connection refused')

        monkeypatch.setattr(requests, 'get', get)
        with pytest.raises(SystemExit) as exc:
            skills.run(['install'])
        assert exc.value.code == utils.EXIT_ERROR
        assert f'could not download the skill from {skills.ARCHIVE_URL}' in capsys.readouterr().err
        assert list(project.iterdir()) == []

    @pytest.mark.parametrize('damage', ['truncated', 'not gzip'])
    def test_unreadable_archive(
        self, project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], damage: str
    ) -> None:
        """A cut-off download or an HTML error page ends with the exit-1 line, not a traceback."""
        archive = _repo_archive()[:200] if damage == 'truncated' else b'<html>rate limited</html>'
        _serve(monkeypatch, archive)
        with pytest.raises(SystemExit) as exc:
            skills.run(['install'])
        assert exc.value.code == utils.EXIT_ERROR
        assert 'does not hold the skill' in capsys.readouterr().err
        assert list(project.iterdir()) == []

    def test_failed_swap_keeps_previous_copy(
        self, project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When the new copy cannot be renamed into place, the previous one is put back and nothing is left over."""
        _serve(monkeypatch, _repo_archive())
        skills.run(['install'])
        target = project / '.agents/skills/pixeltable'
        (target / 'SKILL.md').write_bytes(b'previous\n')
        before = _tree(target)
        capsys.readouterr()

        rename = pathlib.Path.rename

        def failing_rename(self: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
            if self.name.startswith('.pixeltable.new-'):
                raise OSError('disk full')
            return rename(self, dst)

        monkeypatch.setattr(pathlib.Path, 'rename', failing_rename)
        with pytest.raises(SystemExit) as exc:
            skills.run(['install', '-f'])
        assert exc.value.code == utils.EXIT_ERROR
        assert 'could not write' in capsys.readouterr().err
        assert _tree(target) == before
        assert sorted(p.name for p in target.parent.iterdir()) == ['pixeltable']

    def test_archive_without_skill(
        self, project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _serve(monkeypatch, _archive({'README.md': b'moved\n'}))
        with pytest.raises(SystemExit) as exc:
            skills.run(['install'])
        assert exc.value.code == utils.EXIT_ERROR
        assert 'does not hold the skill' in capsys.readouterr().err
        assert list(project.iterdir()) == []
