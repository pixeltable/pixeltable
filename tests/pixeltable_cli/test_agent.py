"""A coding agent runs pxt with no one to answer a prompt."""

import io
import sys

import pytest

from pixeltable_cli.client import utils
from pixeltable_cli.client.commands import shell

# what each agent sets in the environment of the commands it runs
_AGENT_MARKERS = {
    'CLAUDECODE': 'claude',
    'CLAUDE_CODE': 'claude',
    'CODEX_CI': 'codex',
    'CODEX_SANDBOX': 'codex',
    'CODEX_THREAD_ID': 'codex',
    'CURSOR_AGENT': 'cursor',
    'GEMINI_CLI': 'gemini',
    'ANTIGRAVITY_AGENT': 'antigravity',
    'AUGMENT_AGENT': 'augment',
    'OPENCODE_CLIENT': 'opencode',
}
_ANY_AGENT = ('PXT_AGENT', 'AI_AGENT', 'CURSOR_EXTENSION_HOST_ROLE', *_AGENT_MARKERS)
_A_PERSONS_TERMINAL = ('CURSOR_TRACE_ID', 'REPL_ID', 'COPILOT_MODEL')


class _Unanswerable:
    """A stdin that fails the test if pxt reads an answer from it."""

    def readline(self) -> str:
        raise AssertionError('pxt waited on a prompt')


@pytest.fixture
def person(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """An environment no agent has marked, whatever runs these tests."""
    for variable in (*_ANY_AGENT, *_A_PERSONS_TERMINAL):
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


class TestRunByAgent:
    @pytest.mark.parametrize(('variable', 'agent'), _AGENT_MARKERS.items())
    def test_an_agents_marker_means_an_agent(self, person: pytest.MonkeyPatch, variable: str, agent: str) -> None:
        person.setenv(variable, '1')
        assert utils.run_by_agent()
        assert utils.agent_name() == agent

    def test_cursors_agent_process_is_an_agent(self, person: pytest.MonkeyPatch) -> None:
        person.setenv('CURSOR_EXTENSION_HOST_ROLE', 'agent-exec')
        assert utils.run_by_agent()

    @pytest.mark.parametrize('variable', _A_PERSONS_TERMINAL)
    def test_what_a_persons_terminal_carries_is_not_an_agent(self, person: pytest.MonkeyPatch, variable: str) -> None:
        person.setenv(variable, 'x')
        assert not utils.run_by_agent()

    def test_an_empty_name_is_not_an_agent(self, person: pytest.MonkeyPatch) -> None:
        person.setenv('AI_AGENT', ' ')
        assert not utils.run_by_agent()

    def test_agent_name(self, person: pytest.MonkeyPatch) -> None:
        assert utils.agent_name() is None
        person.setenv('CLAUDECODE', '1')
        assert utils.agent_name() == 'claude'
        # the name an agent declares comes first
        person.setenv('AI_AGENT', 'claude-code_2-1-299_agent')
        assert utils.agent_name() == 'claude-code_2-1-299_agent'

    def test_pxt_agent_decides_either_way(self, person: pytest.MonkeyPatch) -> None:
        person.setenv('CLAUDECODE', '1')
        person.setenv('PXT_AGENT', '0')
        assert not utils.run_by_agent()

        person.delenv('CLAUDECODE')
        person.setenv('PXT_AGENT', '1')
        assert utils.run_by_agent()


class TestConfirmation:
    def test_an_agent_is_refused_without_a_prompt_even_in_a_terminal(
        self, person: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An agent that runs pxt in a terminal would otherwise wait on `[y/N]`, or answer it itself."""
        person.setenv('CLAUDECODE', '1')
        person.setattr(utils, 'stdin_is_a_tty', lambda: True)
        person.setattr(sys, 'stdin', _Unanswerable())
        person.setattr(utils, 'invocation', ['drop', 't'])

        with pytest.raises(SystemExit) as exc:
            utils.confirm_or_exit('drop table t?', force=False)

        assert exc.value.code == utils.EXIT_REFUSED
        refusal = capsys.readouterr().err
        assert 'a coding agent cannot answer the prompt; ask the user' in refusal
        assert 'next: pxt drop t -f  # drop table t' in refusal

    def test_force_proceeds_under_an_agent(self, person: pytest.MonkeyPatch) -> None:
        person.setenv('CLAUDECODE', '1')
        person.setattr(sys, 'stdin', _Unanswerable())
        utils.confirm_or_exit('drop table t?', force=True)

    def test_a_person_in_a_terminal_is_asked(self, person: pytest.MonkeyPatch) -> None:
        person.setattr(utils, 'stdin_is_a_tty', lambda: True)
        person.setattr(sys, 'stdin', io.StringIO('y\n'))
        utils.confirm_or_exit('drop table t?', force=False)

    def test_a_person_who_declines_is_refused(self, person: pytest.MonkeyPatch) -> None:
        person.setattr(utils, 'stdin_is_a_tty', lambda: True)
        person.setattr(sys, 'stdin', io.StringIO('n\n'))
        with pytest.raises(SystemExit) as exc:
            utils.confirm_or_exit('drop table t?', force=False)
        assert exc.value.code == utils.EXIT_REFUSED


class TestCanPrompt:
    def test_an_agent_with_a_terminal_cannot_be_asked(self, person: pytest.MonkeyPatch) -> None:
        """What `schema update` reads to decide whether its rerun needs -f as well as --allow-destructive."""
        person.setattr(utils, 'stdin_is_a_tty', lambda: True)
        assert utils.can_prompt()
        person.setenv('CURSOR_AGENT', '1')
        assert not utils.can_prompt()


class TestRefusal:
    def test_the_next_line_says_what_the_command_does(
        self, person: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The question names what the rerun does; the sentence after it is left out."""
        person.setattr(utils, 'stdin_is_a_tty', lambda: False)
        person.setattr(utils, 'invocation', ['db', 'delete', 'pxt://acme:gone'])
        with pytest.raises(SystemExit):
            utils.confirm_or_exit('delete pxt://acme:gone? This is irreversible.', force=False)
        assert 'next: pxt db delete pxt://acme:gone -f  # delete pxt://acme:gone\n' in capsys.readouterr().err

    def test_flags_go_before_the_option_terminator(self, person: pytest.MonkeyPatch) -> None:
        person.setattr(utils, 'invocation', ['drop', '--', '-odd name'])
        assert utils.rerun_command('-f') == "pxt drop -f -- '-odd name'"


class TestShell:
    def test_an_agent_is_refused_at_once(self, person: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        person.setenv('CLAUDECODE', '1')
        person.setattr(utils, 'stdin_is_a_tty', lambda: True)
        person.setattr(sys, 'stdin', _Unanswerable())

        with pytest.raises(SystemExit) as exc:
            shell.run([])

        assert exc.value.code == utils.EXIT_ERROR
        assert "run each command as 'pxt <command>'" in capsys.readouterr().err
