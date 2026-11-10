"""Tests for plan_claude_ci_fix.py. Run with: python -m unittest discover -s .github/scripts -p 'test_*.py'"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any
from unittest import mock

import plan_claude_ci_fix as plan


def event(
    *,
    name: str = 'tests',
    run_id: int = 9,
    trigger: str = 'schedule',
    conclusion: str = 'failure',
    head_branch: str = 'main',
    path: str = '.github/workflows/pytest.yml',
    default_branch: str = 'main',
) -> dict[str, Any]:
    return {
        'workflow_run': {
            'id': run_id,
            'name': name,
            'event': trigger,
            'conclusion': conclusion,
            'head_branch': head_branch,
            'path': path,
            'html_url': f'https://github.com/o/r/actions/runs/{run_id}',
        },
        'repository': {'default_branch': default_branch},
    }


class MarkerPayloadTest(unittest.TestCase):
    def test_parses_json_comment(self) -> None:
        body = 'hello\n<!-- claude-ci-fix {"workflow":"tests","run":9,"jobs":["lint"]} -->\n'
        self.assertEqual(plan.marker_payload(body), {'workflow': 'tests', 'run': 9, 'jobs': ['lint']})

    def test_rejects_broken_json(self) -> None:
        self.assertIsNone(plan.marker_payload('<!-- claude-ci-fix {not json} -->'))


class MainTest(unittest.TestCase):
    def run_main(self, payload: dict[str, Any]) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            event_path, output_path = os.path.join(tmp, 'event.json'), os.path.join(tmp, 'output')
            with open(event_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            env = {'GITHUB_EVENT_PATH': event_path, 'GITHUB_OUTPUT': output_path, 'REPO': 'o/r'}
            with mock.patch.dict(os.environ, env):
                plan.main()
            if not os.path.exists(output_path):
                return ''
            with open(output_path, encoding='utf-8') as f:
                return f.read()

    def test_pull_request_run_is_ignored(self) -> None:
        self.assertEqual(self.run_main(event(trigger='pull_request')), '')

    def test_successful_run_is_ignored(self) -> None:
        self.assertEqual(self.run_main(event(conclusion='success')), '')

    def test_other_branch_is_ignored(self) -> None:
        self.assertEqual(self.run_main(event(head_branch='release')), '')

    def test_unknown_workflow_is_ignored(self) -> None:
        self.assertEqual(self.run_main(event(name='Claude review')), '')

    def test_stale_run_is_ignored(self) -> None:
        with mock.patch.object(plan, 'latest_scheduled_run_id', return_value=99):
            self.assertEqual(self.run_main(event(run_id=9)), '')

    def test_closed_unmerged_pr_for_this_run_stops_a_new_fix(self) -> None:
        closed = {
            'number': 1925,
            'title': 'old fix',
            'state': 'CLOSED',
            'headRefName': 'claude/ci-fix-tests-9',
            'body': 'run 9\n<!-- claude-ci-fix {"workflow":"tests","run":9} -->',
            'author': {'login': 'claude'},
            'mergedAt': None,
            'url': 'https://github.com/o/r/pull/1925',
            'closedAt': '2026-11-09T15:03:22Z',
        }
        with (
            mock.patch.object(plan, 'latest_scheduled_run_id', return_value=9),
            mock.patch.object(plan, 'failed_job_names', return_value=['lint']),
            mock.patch.object(plan, 'list_prs', return_value=[closed]),
        ):
            self.assertEqual(self.run_main(event(run_id=9)), '')

    def test_open_human_pr_for_this_run_stops_a_new_fix(self) -> None:
        human = {
            'number': 1927,
            'title': 'human fix',
            'state': 'OPEN',
            'headRefName': 'claude/genai-v1beta-notebook',
            'body': 'The 2026-11-09 nightly https://github.com/o/r/actions/runs/9',
            'author': {'login': 'aaron-siegel'},
            'mergedAt': None,
            'url': 'https://github.com/o/r/pull/1927',
            'closedAt': None,
        }
        with (
            mock.patch.object(plan, 'latest_scheduled_run_id', return_value=9),
            mock.patch.object(plan, 'failed_job_names', return_value=['isolated-nb-tests']),
            mock.patch.object(plan, 'list_prs', return_value=[human]),
        ):
            self.assertEqual(self.run_main(event(run_id=9)), '')

    def test_failure_writes_outputs_and_reuses_an_open_pr(self) -> None:
        open_pr = {
            'number': 42,
            'title': 'fix lint',
            'state': 'OPEN',
            'headRefName': 'claude/ci-fix-tests-8',
            'body': '<!-- claude-ci-fix {"workflow":"tests","run":8} -->',
            'author': {'login': 'claude'},
            'mergedAt': None,
            'url': 'https://github.com/o/r/pull/42',
            'closedAt': None,
        }
        with (
            mock.patch.object(plan, 'latest_scheduled_run_id', return_value=9),
            mock.patch.object(plan, 'failed_job_names', return_value=['lint', 'py']),
            mock.patch.object(plan, 'list_prs', return_value=[open_pr]),
        ):
            out = self.run_main(event(run_id=9, name='tests'))
        self.assertIn('run=9\n', out)
        self.assertIn('workflow=tests\n', out)
        self.assertIn('tested_ref=main\n', out)
        self.assertIn('existing_pr=42\n', out)
        self.assertIn('branch=claude/ci-fix-tests-8\n', out)
        self.assertIn('failed_jobs=["lint","py"]\n', out)

    def test_nightly_uses_the_release_ref(self) -> None:
        with (
            mock.patch.object(plan, 'latest_scheduled_run_id', return_value=9),
            mock.patch.object(plan, 'failed_job_names', return_value=['tests']),
            mock.patch.object(plan, 'list_prs', return_value=[]),
        ):
            out = self.run_main(event(run_id=9, name='nightly', path='.github/workflows/nightly.yml'))
        self.assertIn('tested_ref=release\n', out)
        self.assertIn('branch=claude/ci-fix-nightly-9\n', out)
        self.assertIn('existing_pr=\n', out)

    def test_write_context_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                'RUN': '9',
                'RUN_URL': 'https://github.com/o/r/actions/runs/9',
                'WORKFLOW_NAME': 'tests',
                'TESTED_REF': 'main',
                'FAILED_JOBS': '["lint"]',
                'EXISTING_PR': '42',
                'BRANCH': 'claude/ci-fix-tests-8',
                'CLOSED_PRS': '[]',
                'REPO': 'o/r',
            }
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with mock.patch.dict(os.environ, env):
                    self.assertEqual(plan.write_context_file(), 0)
            finally:
                os.chdir(cwd)
            with open(os.path.join(tmp, 'ci-fix-context.json'), encoding='utf-8') as f:
                data = json.load(f)
        self.assertEqual(data['existing_pr'], 42)
        self.assertEqual(data['failed_jobs'], ['lint'])
        self.assertEqual(data['workflow'], 'tests')

    def test_timed_out_run_is_investigated(self) -> None:
        with (
            mock.patch.object(plan, 'latest_scheduled_run_id', return_value=9),
            mock.patch.object(plan, 'failed_job_names', return_value=[]),
            mock.patch.object(plan, 'list_prs', return_value=[]),
        ):
            out = self.run_main(event(run_id=9, conclusion='timed_out'))
        self.assertIn('run=9\n', out)


if __name__ == '__main__':
    unittest.main()
