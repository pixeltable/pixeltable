"""Tests for plan_claude_ci_fix.py. Run with: python -m unittest discover -s .github/scripts -p 'test_*.py'"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any
from unittest import mock

import plan_claude_ci_fix as plan

RUN_9 = 'https://github.com/o/r/actions/runs/9'


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


def pr(
    number: int,
    *,
    state: str = 'OPEN',
    head: str = 'claude/ci-fix-tests-8',
    body: str = '',
    merged: bool = False,
    cross: bool = False,
) -> dict[str, Any]:
    return {
        'number': number,
        'title': f'pr {number}',
        'state': state,
        'headRefName': head,
        'body': body,
        'mergedAt': '2026-11-09T15:03:22Z' if merged else None,
        'url': f'https://github.com/o/r/pull/{number}',
        'isCrossRepository': cross,
    }


class IsCiFixPrTest(unittest.TestCase):
    def test_same_repo_branch_with_the_prefix(self) -> None:
        self.assertTrue(plan.is_ci_fix_pr(pr(1), 'tests'))

    def test_other_workflow_prefix_is_not_a_match(self) -> None:
        self.assertFalse(plan.is_ci_fix_pr(pr(1, head='claude/ci-fix-nightly-8'), 'tests'))

    def test_fork_head_is_not_a_match(self) -> None:
        self.assertFalse(plan.is_ci_fix_pr(pr(1, cross=True), 'tests'))

    def test_body_text_is_not_a_match(self) -> None:
        body = '<!-- claude-ci-fix {"workflow":"tests"} -->'
        self.assertFalse(plan.is_ci_fix_pr(pr(1, head='main', body=body), 'tests'))


class MainTest(unittest.TestCase):
    def run_main(
        self, payload: dict[str, Any], prs: list[dict[str, Any]] | None = None, latest: int | None = None
    ) -> str:
        run_id = payload['workflow_run']['id']
        with tempfile.TemporaryDirectory() as tmp:
            event_path, output_path = os.path.join(tmp, 'event.json'), os.path.join(tmp, 'output')
            with open(event_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            env = {'GITHUB_EVENT_PATH': event_path, 'GITHUB_OUTPUT': output_path, 'REPO': 'o/r'}
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.object(plan, 'latest_scheduled_run_id', return_value=latest or run_id),
                mock.patch.object(plan, 'failed_job_names', return_value=['lint', 'py']),
                mock.patch.object(plan, 'list_prs', return_value=prs or []),
            ):
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
        self.assertEqual(self.run_main(event(run_id=9), latest=99), '')

    def test_closed_unmerged_pr_naming_this_run_stops_a_new_fix(self) -> None:
        closed = pr(1925, state='CLOSED', head='claude/ci-fix-tests-9', body=RUN_9)
        self.assertEqual(self.run_main(event(run_id=9), [closed]), '')

    def test_open_human_pr_naming_this_run_stops_a_new_fix(self) -> None:
        human = pr(1927, head='claude/genai-v1beta-notebook', body=f'nightly {RUN_9}')
        self.assertEqual(self.run_main(event(run_id=9), [human]), '')

    def test_fork_pr_naming_this_run_does_not_stop_a_fix(self) -> None:
        fork = pr(1930, head='x/y', body=RUN_9, cross=True)
        self.assertIn('workflow=tests\n', self.run_main(event(run_id=9), [fork]))

    def test_bare_digits_do_not_name_this_run(self) -> None:
        other = pr(1929, head='x/y', body='fixes 1929, v9.1 and https://github.com/o/r/actions/runs/91')
        self.assertIn('workflow=tests\n', self.run_main(event(run_id=9), [other]))

    def test_merged_pr_naming_this_run_does_not_stop_a_fix(self) -> None:
        merged = pr(1900, state='MERGED', head='someone/fix', body=RUN_9, merged=True)
        self.assertIn('branch=claude/ci-fix-tests-9\n', self.run_main(event(run_id=9), [merged]))

    def test_open_ci_fix_pr_for_the_workflow_stops_a_new_fix(self) -> None:
        self.assertEqual(self.run_main(event(run_id=9), [pr(42)]), '')

    def test_open_ci_fix_pr_for_another_workflow_does_not_stop_a_fix(self) -> None:
        other = pr(42, head='claude/ci-fix-nightly-8')
        self.assertIn('workflow=tests\n', self.run_main(event(run_id=9), [other]))

    def test_fork_pr_on_a_ci_fix_branch_does_not_stop_a_fix(self) -> None:
        self.assertIn('workflow=tests\n', self.run_main(event(run_id=9), [pr(42, cross=True)]))

    def test_failure_computes_the_branch_and_writes_the_context(self) -> None:
        closed = pr(40, state='CLOSED', head='claude/ci-fix-tests-5')
        out = self.run_main(event(run_id=9), [closed])
        lines = dict(line.split('=', 1) for line in out.splitlines())
        self.assertEqual(lines['workflow'], 'tests')
        self.assertEqual(lines['branch'], 'claude/ci-fix-tests-9')
        context = json.loads(lines['context'])
        self.assertEqual(context['run'], 9)
        self.assertEqual(context['tested_ref'], 'main')
        self.assertEqual(context['failed_jobs'], ['lint', 'py'])
        self.assertEqual(context['branch'], 'claude/ci-fix-tests-9')
        self.assertEqual([p['number'] for p in context['closed_prs']], [40])

    def test_nightly_uses_the_release_ref(self) -> None:
        out = self.run_main(event(run_id=9, name='nightly', path='.github/workflows/nightly.yml'))
        self.assertIn('branch=claude/ci-fix-nightly-9\n', out)
        self.assertEqual(json.loads(out.split('context=', 1)[1])['tested_ref'], 'release')

    def test_timed_out_run_is_investigated(self) -> None:
        self.assertIn('workflow=tests\n', self.run_main(event(run_id=9, conclusion='timed_out')))


if __name__ == '__main__':
    unittest.main()
