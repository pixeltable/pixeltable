"""A confirmation is asked at a terminal and refused without one, naming the command that confirms it."""

import io
import sys

import pytest

from pixeltable_cli.client import utils


class _Unanswerable:
    """A stdin that fails the test if pxt reads an answer from it."""

    def readline(self) -> str:
        raise AssertionError('pxt waited on a prompt')


class TestConfirmation:
    def test_without_a_terminal_the_refusal_names_the_command(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(utils, 'stdin_is_a_tty', lambda: False)
        monkeypatch.setattr(sys, 'stdin', _Unanswerable())
        monkeypatch.setattr(utils, 'invocation', ['drop', 't'])

        with pytest.raises(SystemExit) as exc:
            utils.confirm_or_exit('drop table t?', force=False)

        assert exc.value.code == utils.EXIT_REFUSED
        refusal = capsys.readouterr().err
        assert 'no TTY for confirmation' in refusal
        assert 'next: pxt drop t -f  # drop table t' in refusal

    def test_force_proceeds_without_a_terminal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(utils, 'stdin_is_a_tty', lambda: False)
        monkeypatch.setattr(sys, 'stdin', _Unanswerable())
        utils.confirm_or_exit('drop table t?', force=True)

    def test_whoever_is_at_a_terminal_is_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # a coding agent running pxt in a terminal is asked too, as gh, Vercel and Railway ask it
        monkeypatch.setenv('CLAUDECODE', '1')
        monkeypatch.setattr(utils, 'stdin_is_a_tty', lambda: True)
        monkeypatch.setattr(sys, 'stdin', io.StringIO('y\n'))
        utils.confirm_or_exit('drop table t?', force=False)

    def test_an_answer_other_than_yes_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(utils, 'stdin_is_a_tty', lambda: True)
        monkeypatch.setattr(sys, 'stdin', io.StringIO('n\n'))
        with pytest.raises(SystemExit) as exc:
            utils.confirm_or_exit('drop table t?', force=False)
        assert exc.value.code == utils.EXIT_REFUSED


class TestRefusal:
    def test_the_next_line_says_what_the_command_does(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The question names what the rerun does; the sentence after it is left out."""
        monkeypatch.setattr(utils, 'stdin_is_a_tty', lambda: False)
        monkeypatch.setattr(utils, 'invocation', ['db', 'delete', 'pxt://acme:gone'])
        with pytest.raises(SystemExit):
            utils.confirm_or_exit('delete pxt://acme:gone? This is irreversible.', force=False)
        assert 'next: pxt db delete pxt://acme:gone -f  # delete pxt://acme:gone\n' in capsys.readouterr().err

    def test_flags_go_before_the_option_terminator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(utils, 'invocation', ['drop', '--', '-odd name'])
        assert utils.rerun_command('-f') == "pxt drop -f -- '-odd name'"
