"""Tests for parse_claude_ci_fix.py. Run with: python -m unittest discover -s .github/scripts -p 'test_*.py'"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import parse_claude_ci_fix as parse


def execution_file(tmp: str, message: str) -> str:
    path = os.path.join(tmp, 'execution.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([{'type': 'result', 'result': message}], f)
    return path


class ParseMainTest(unittest.TestCase):
    def run_main(self, message: str, *, found: int | None = 42) -> tuple[str, mock.Mock]:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, 'output')
            env = {
                'EXECUTION_FILE': execution_file(tmp, message),
                'GITHUB_OUTPUT': output_path,
                'REPO': 'o/r',
                'BRANCH': 'claude/ci-fix-tests-9',
            }
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.object(parse, 'pr_on_branch', return_value=found),
                mock.patch.object(parse, 'gh') as gh,
            ):
                parse.main()
            out = ''
            if os.path.exists(output_path):
                with open(output_path, encoding='utf-8') as f:
                    out = f.read()
            return out, gh

    def test_opened_pr_gets_agent_review_and_is_reviewed(self) -> None:
        out, gh = self.run_main('```json\n{"action":"opened","pr":7,"reason":"lint pin"}\n```\n')
        self.assertEqual(out, 'pr=42\n')
        gh.assert_called_once_with('pr', 'comment', '42', '--repo', 'o/r', '--body', 'agent review')

    def test_pr_comes_from_the_branch_not_the_message(self) -> None:
        out, _ = self.run_main('```json\n{"action":"opened","pr":1,"reason":"x"}\n```\n', found=42)
        self.assertEqual(out, 'pr=42\n')

    def test_skipped_fix_writes_no_pr(self) -> None:
        out, gh = self.run_main('```json\n{"action":"skipped","pr":null,"reason":"already on main"}\n```\n')
        self.assertEqual(out, '')
        gh.assert_not_called()

    def test_opened_without_a_branch_pr_writes_no_pr(self) -> None:
        out, gh = self.run_main('```json\n{"action":"opened","pr":7,"reason":"x"}\n```\n', found=None)
        self.assertEqual(out, '')
        gh.assert_not_called()


if __name__ == '__main__':
    unittest.main()
