import importlib
import pathlib
import re
import shlex
import subprocess
import sys

import pytest

import pixeltable as pxt

pytestmark = pytest.mark.db_roots('local', reason='runs the examples of the agent guides against the local catalog')

GUIDES = sorted(
    path for path in (pathlib.Path(pxt.__file__).parent / 'agent_guides').glob('*.md') if path.name != 'README.md'
)


def code_blocks(guide: pathlib.Path, lang: str) -> list[str]:
    return re.findall(rf'^```{lang}\n(.*?)^```', guide.read_text(encoding='utf-8'), flags=re.MULTILINE | re.DOTALL)


# error messages that a guide quotes and that start with 'pxt '
QUOTED_MESSAGES = {"pxt paths use '/' as the separator"}


def pxt_commands(guide: pathlib.Path) -> list[str]:
    """Return the pxt lines of a guide's bash blocks, then its inline code that starts with pxt."""
    lines = [line for block in code_blocks(guide, 'bash') for line in block.splitlines()]
    prose = re.sub(r'^```.*?^```', '', guide.read_text(encoding='utf-8'), flags=re.MULTILINE | re.DOTALL)
    inline = [span for span in re.findall(r'`(pxt [^`]+)`', prose) if span not in QUOTED_MESSAGES]
    return [line for line in lines if shlex.split(line, comments=True)[:1] == ['pxt']] + inline


def option_flags(help_text: str) -> set[str]:
    # the options list indents each entry by two spaces: '  -h, --help', '  --on TYPE'
    matches = re.findall(r'^  (-[^\s,]+)(?:, (-[^\s,]+))?', help_text, flags=re.MULTILINE)
    return {flag for match in matches for flag in match if flag != ''}


def pxt_help(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-m', 'pixeltable_cli.client.main', *args, '--help'],
        capture_output=True,
        text=True,
        check=False,
    )


class TestAgentGuides:
    @pytest.mark.parametrize('guide', GUIDES, ids=[path.stem for path in GUIDES])
    def test_examples(
        self, guide: pathlib.Path, uses_db: None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The python blocks of a guide import as one module, and create_all() succeeds on each of its TableModel
        bases."""
        # a module, not exec(): a UDF needs a module path
        module_name = f'agent_guide_{guide.stem}'
        (tmp_path / f'{module_name}.py').write_text('\n\n'.join(code_blocks(guide, 'python')), encoding='utf-8')
        monkeypatch.syspath_prepend(str(tmp_path))
        # removed only at the end: create_all() looks up the module to load the UDFs of the new tables
        try:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as e:
                if e.name is None or e.name.split('.')[0] == 'pixeltable':
                    raise
                pytest.skip(f'{guide.name} needs {e.name}')
            catalog_dir = f'guide_{guide.stem}'
            pxt.create_dir(catalog_dir, if_exists='ignore')
            for value in list(vars(module).values()):
                if isinstance(value, type) and 'create_all' in vars(value):
                    vars(value)['create_all'](catalog_dir)
        finally:
            sys.modules.pop(module_name, None)

    @pytest.mark.parametrize('guide', GUIDES, ids=[path.stem for path in GUIDES])
    def test_commands(self, guide: pathlib.Path) -> None:
        """Each pxt command in a guide, in a bash block or inline, calls an existing command with flags from its
        options list."""
        for command in pxt_commands(guide):
            args = shlex.split(command, comments=True)
            # the second argument is a verb (schema update), an operand (errors my_app/docs) or a flag
            result = pxt_help(args[1:2] if len(args) > 2 and args[2].startswith('-') else args[1:3])
            assert result.returncode == 0, f'{command}: {result.stderr}'
            flags = option_flags(result.stdout)
            for flag in (arg.split('=')[0] for arg in args[2:] if arg.startswith('-')):
                assert flag in flags, f'{command}: no {flag}'
