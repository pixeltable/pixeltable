"""Tests for parse_claude_ci_fix.py. Run with: python -m unittest discover -s .github/scripts -p 'test_*.py'"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import parse_claude_ci_fix as parse
import post_claude_review as post


def execution_file(tmp: str, message: str) -> str:
    path = os.path.join(tmp, 'execution.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([{'type': 'result', 'result': message}], f)
    return path


class ParseMainTest(unittest.TestCase):
    def run_main(self, message: str, *, branch: str = 'claude/ci-fix-tests-9') -> str:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, 'output')
            env = {
                'EXECUTION_FILE': execution_file(tmp, message),
                'GITHUB_OUTPUT': output_path,
                'REPO': 'o/r',
                'RUN_ID': '9',
                'WORKFLOW': 'tests',
                'FAILED_JOBS': '["lint"]',
                'BRANCH': branch,
            }
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.object(parse, 'ensure_marker'),
                mock.patch.object(parse, 'has_agent_review_comment', return_value=False),
                mock.patch.object(parse, 'post_agent_review') as posted,
            ):
                parse.main()
                self.posted = posted
            if not os.path.exists(output_path):
                return ''
            with open(output_path, encoding='utf-8') as f:
                return f.read()

    def test_opened_pr_is_reviewed_and_gets_agent_review(self) -> None:
        out = self.run_main('done\n```json\n{"action":"opened","pr":42,"reason":"lint pin"}\n```\n')
        self.assertEqual(out, 'pr=42\n')
        self.posted.assert_called_once_with('o/r', 42)

    def test_skipped_fix_writes_no_pr(self) -> None:
        out = self.run_main('```json\n{"action":"skipped","pr":null,"reason":"already on main"}\n```\n')
        self.assertEqual(out, '')
        self.posted.assert_not_called()

    def test_updated_pr_does_not_repost_agent_review(self) -> None:
        out = self.run_main('```json\n{"action":"updated","pr":42,"reason":"same lint"}\n```\n')
        self.assertEqual(out, 'pr=42\n')
        self.posted.assert_not_called()


class FindingsTest(unittest.TestCase):
    def test_inline_comments_count_as_findings(self) -> None:
        data = {'post': True, 'body': '## Claude review\n', 'comments': [{'path': 'a.py', 'line': 1, 'body': 'x'}]}
        self.assertTrue(post.has_findings(data))

    def test_no_supported_findings_is_empty(self) -> None:
        data = {'post': True, 'body': '## Claude review\n\nNo supported findings.\n', 'comments': []}
        self.assertFalse(post.has_findings(data))

    def test_unposted_review_is_empty(self) -> None:
        self.assertFalse(post.has_findings({'post': False, 'reason': 'version bump'}))


if __name__ == '__main__':
    unittest.main()
