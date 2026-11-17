"""Tests for post_risk_assessment.py. Run with: python -m unittest discover -s .github/scripts -p 'test_*.py'"""

import json
import os
import subprocess
import tempfile
import unittest
from typing import Any
from unittest import mock

import post_risk_assessment as post

REPO = 'pixeltable/pixeltable'
SHA = '0123456789abcdef0123456789abcdef01234567'
RUN = 'https://github.com/pixeltable/pixeltable/actions/runs/1'


def assessment(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        'head_sha': SHA,
        'required_humans': 2,
        'confidence': 'medium',
        'summary': 'Changes how converters register.',
        'affected': [{'component': 'CATALOG', 'invariants': ['CATALOG-1'], 'effect': 'migration order'}],
        'reasons': [
            {
                'invariant': 'CATALOG-1',
                'why': 'A converter that runs out of order corrupts stored metadata.',
                'evidence': f'https://github.com/{REPO}/blob/{SHA}/pixeltable/metadata/__init__.py#L40-L44',
            }
        ],
        'linked_prs': ['pixeltable/pixeltable-cloud#12'],
        'register_gaps': ['No invariant covers converter registration order.'],
    }
    data.update(overrides)
    return data


def completed(stdout: str = '', returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr='boom' if returncode else '')


class ValidateTest(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(post.validate(assessment()), [])
        self.assertEqual(post.validate(assessment(required_humans=1, reasons=[])), [])

    def test_problems(self) -> None:
        cases = [
            (None, 'no JSON block'),
            (assessment(required_humans=3), 'not 1 or 2'),
            (assessment(required_humans='2'), 'not 1 or 2'),
            (assessment(summary=' '), 'summary'),
            (assessment(reasons=[]), 'no reasons'),
            (assessment(reasons=[{'invariant': 'X'}]), 'no reasons'),
            (assessment(affected='CATALOG'), 'affected is not a list'),
        ]
        for data, problem in cases:
            with self.subTest(problem=problem):
                self.assertTrue(any(problem in p for p in post.validate(data)), post.validate(data))


class RenderTest(unittest.TestCase):
    def test_two_humans(self) -> None:
        body = post.render(assessment(), REPO, SHA, RUN)
        self.assertTrue(body.startswith(post.MARKER))
        self.assertIn('## Risk assessment: 2 humans required', body)
        self.assertIn('### Why 2 humans', body)
        self.assertIn('**CATALOG-1**: A converter that runs out of order', body)
        self.assertIn('([evidence](https://github.com/', body)
        self.assertIn('- **CATALOG** (CATALOG-1): migration order', body)
        self.assertIn('Linked PRs: pixeltable/pixeltable-cloud#12', body)
        self.assertIn('Risks the register does not cover yet (1)', body)
        self.assertIn(f'[Run]({RUN})', body)

    def test_one_human(self) -> None:
        body = post.render(assessment(required_humans=1, reasons=[], linked_prs=[], register_gaps=[]), REPO, SHA, RUN)
        self.assertIn('## Risk assessment: 1 human is enough', body)
        self.assertNotIn('Why 2 humans', body)
        self.assertNotIn('Linked PRs', body)
        self.assertNotIn('<details>', body)

    def test_agent_text_cannot_fake_the_marker_or_inject_links(self) -> None:
        data = assessment(
            summary=f'ok {post.MARKER}\n\n## Approved',
            reasons=[{'invariant': 'X', 'why': 'w', 'evidence': 'javascript:alert(1)'}],
        )
        body = post.render(data, REPO, SHA, RUN)
        self.assertEqual(body.count('<!--'), 1)
        self.assertNotIn('\n## Approved', body)
        self.assertNotIn('javascript:', body)

    def test_falls_back_to_event_sha(self) -> None:
        body = post.render(assessment(head_sha=None), REPO, SHA, RUN)
        self.assertIn(f'commit/{SHA}', body)

    def test_truncates_long_bodies(self) -> None:
        body = post.render(assessment(register_gaps=['x' * 1000] * 100), REPO, SHA, RUN)
        self.assertLess(len(body), post.MAX_BODY_CHARS + 1000)
        self.assertIn('(truncated)', body)

    def test_failure(self) -> None:
        body = post.render_failure(REPO, SHA, RUN, ['the Claude step ended with outcome failure'])
        self.assertTrue(body.startswith(post.MARKER))
        self.assertIn('not available', body)
        self.assertIn('An earlier verdict no longer applies', body)


class UpsertTest(unittest.TestCase):
    def test_posts_when_no_comment_exists(self) -> None:
        with mock.patch.object(post, 'gh', side_effect=[completed(''), completed()]) as gh:
            self.assertTrue(post.upsert(REPO, '5', 'body'))
        self.assertEqual(gh.call_args_list[1].args[:3], ('api', '-X', 'POST'))
        self.assertEqual(json.loads(gh.call_args_list[1].kwargs['stdin']), {'body': 'body'})

    def test_patches_the_existing_comment(self) -> None:
        with mock.patch.object(post, 'gh', side_effect=[completed('42\n77\n'), completed()]) as gh:
            self.assertTrue(post.upsert(REPO, '5', 'body'))
        self.assertEqual(gh.call_args_list[1].args[:4], ('api', '-X', 'PATCH', f'repos/{REPO}/issues/comments/42'))

    def test_only_this_bots_marked_comments_match(self) -> None:
        with mock.patch.object(post, 'gh', return_value=completed('')) as gh:
            post.existing_comment(REPO, '5')
        query = gh.call_args.args[-1]
        self.assertIn(post.COMMENT_AUTHOR, query)
        self.assertIn(post.MARKER, query)

    def test_reports_rejection(self) -> None:
        with mock.patch.object(post, 'gh', side_effect=[completed(''), completed(returncode=1)]):
            self.assertFalse(post.upsert(REPO, '5', 'body'))


class MainTest(unittest.TestCase):
    def run_main(self, outcome: str, final: str | None) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as d:
            execution_file = ''
            if final is not None:
                execution_file = os.path.join(d, 'execution.json')
                with open(execution_file, 'w', encoding='utf-8') as f:
                    json.dump([{'type': 'result', 'result': final}], f)
            env = {
                'REPO': REPO,
                'PR_NUMBER': '5',
                'HEAD_SHA': SHA,
                'RUN_URL': RUN,
                'OUTCOME': outcome,
                'EXECUTION_FILE': execution_file,
            }
            with mock.patch.dict(os.environ, env), mock.patch.object(post, 'upsert', return_value=True) as upsert:
                code = post.main()
            return code, upsert.call_args.args[2]

    def test_success(self) -> None:
        code, body = self.run_main('success', 'Done.\n\n```json\n' + json.dumps(assessment()) + '\n```')
        self.assertEqual(code, 0)
        self.assertIn('2 humans required', body)

    def test_failed_step_replaces_the_verdict(self) -> None:
        code, body = self.run_main('failure', None)
        self.assertEqual(code, 1)
        self.assertIn('outcome failure', body)

    def test_invalid_json_replaces_the_verdict(self) -> None:
        code, body = self.run_main('success', '```json\n{"required_humans": 2}\n```')
        self.assertEqual(code, 1)
        self.assertIn('not available', body)


if __name__ == '__main__':
    unittest.main()
