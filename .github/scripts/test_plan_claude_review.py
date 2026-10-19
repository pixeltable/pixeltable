"""Tests for plan_claude_review.py. Run with: python -m unittest discover -s .github/scripts -p 'test_*.py'"""

import json
import os
import tempfile
import unittest
from unittest import mock

import plan_claude_review as plan


class FocusOfTest(unittest.TestCase):
    def test_requests(self) -> None:
        cases = {
            'claude review': '',
            'Claude Review.': '',
            'agent review': '',
            'agents review: tests': 'tests',
            'claude review focus on function signatures': 'focus on function signatures',
            'claude review: .github/workflows': '.github/workflows',
            'claude review: --strict validation': '--strict validation',
            'claude review - security': 'security',
            'claude review, security': 'security',
            'claude review\n\nerror paths': 'error paths',
        }
        for body, focus in cases.items():
            with self.subTest(body=body):
                self.assertEqual(plan.focus_of(body), focus)

    def test_remarks(self) -> None:
        remarks = [
            'Claude review missed the null case',
            'claude review-the-change is great',
            'claude review .github/workflows',
            'claude reviewer',
            '## Claude review',
        ]
        for body in remarks:
            with self.subTest(body=body):
                self.assertIsNone(plan.focus_of(body))

    def test_focus_is_one_line_within_the_cap(self) -> None:
        self.assertEqual(plan.focus_of('claude review:\n  retry\n  logic'), 'retry logic')
        self.assertEqual(plan.focus_of('claude review: ' + 'x' * 900), 'x' * plan.MAX_FOCUS_CHARS)


class MainTest(unittest.TestCase):
    def run_main(self, body: str, state: str) -> str:
        """Returns what main() writes to GITHUB_OUTPUT for a comment on PR #7."""
        with tempfile.TemporaryDirectory() as tmp:
            event_path, output_path = os.path.join(tmp, 'event.json'), os.path.join(tmp, 'output')
            with open(event_path, 'w', encoding='utf-8') as f:
                json.dump({'issue': {'number': 7, 'state': state}, 'comment': {'body': body}}, f)
            env = {'GITHUB_EVENT_PATH': event_path, 'GITHUB_OUTPUT': output_path, 'REPO': 'o/r', 'ACTOR': 'a'}
            with mock.patch.dict(os.environ, env):
                plan.main()
            if not os.path.exists(output_path):
                return ''
            with open(output_path, encoding='utf-8') as f:
                return f.read()

    def test_open_pr(self) -> None:
        with (
            mock.patch.object(plan, 'has_write_access', return_value=True),
            mock.patch.object(plan, 'review_running', return_value=False),
        ):
            self.assertEqual(self.run_main('claude review: tests', 'open'), 'pr=7\nfocus=tests\n')

    def test_closed_pr(self) -> None:
        with mock.patch.object(plan, 'has_write_access') as has_write_access:
            self.assertEqual(self.run_main('claude review', 'closed'), '')
        has_write_access.assert_not_called()

    def test_review_already_running(self) -> None:
        with (
            mock.patch.object(plan, 'has_write_access', return_value=True),
            mock.patch.object(plan, 'review_running', return_value=True),
        ):
            self.assertEqual(self.run_main('agent review', 'open'), '')

    def test_no_write_access(self) -> None:
        with mock.patch.object(plan, 'has_write_access', return_value=False):
            self.assertEqual(self.run_main('claude review', 'open'), '')


if __name__ == '__main__':
    unittest.main()
