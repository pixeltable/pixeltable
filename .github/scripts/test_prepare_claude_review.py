"""Prepare real Git objects while replacing only the GitHub transport in local tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import claude_review_contract as contract
import prepare_claude_review as prepare


class PrepareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '--quiet')
        self.git('config', 'core.hooksPath', '/dev/null')
        (self.repo / 'example.py').write_text('old\n', encoding='utf-8')
        (self.repo / 'ignored.py').write_text('source must remain visible\n', encoding='utf-8')
        (self.repo / '.gitattributes').write_text('ignored.py export-ignore\n', encoding='utf-8')
        (self.repo / 'AGENTS.md').write_text('Untrusted PR instruction', encoding='utf-8')
        (self.repo / '.claude').mkdir()
        (self.repo / '.claude/commands').mkdir()
        (self.repo / '.claude/commands/pxt-review.md').write_text('Trusted command under test', encoding='utf-8')
        (self.repo / 'external-link').symlink_to('/etc/passwd')
        self.git('add', '.')
        self.git('-c', 'user.name=Review Test', '-c', 'user.email=review@example.invalid', 'commit', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD')
        (self.repo / 'example.py').write_text('new\n', encoding='utf-8')
        self.git('add', '.')
        self.git('-c', 'user.name=Review Test', '-c', 'user.email=review@example.invalid', 'commit', '-qm', 'head')
        self.head = self.git('rev-parse', 'HEAD')

    def git(self, *args: str) -> str:
        return subprocess.check_output(['git', *args], cwd=self.repo, text=True).strip()

    def test_export_is_source_only_and_does_not_obey_pr_archive_attributes(self) -> None:
        destination = self.root / 'head'
        index = prepare.export_tree(self.repo / '.git', self.head, destination)
        self.assertTrue((destination / 'ignored.py').exists())
        self.assertFalse((destination / 'AGENTS.md').exists())
        self.assertFalse((destination / '.claude').exists())
        self.assertFalse((destination / 'external-link').is_symlink())
        self.assertEqual((destination / 'external-link').read_text(), '/etc/passwd')
        self.assertTrue(index['AGENTS.md']['stripped'])
        data = json.loads((self.root / 'head-stripped.json').read_text())
        self.assertEqual(data['AGENTS.md'], 'Untrusted PR instruction')

    def prepare_local(self) -> Path:
        metadata = {
            'title': 'Fix input',
            'body': 'PR data',
            'state': 'OPEN',
            'baseRefOid': self.base,
            'headRefOid': self.head,
            'baseRefName': 'main',
        }
        original_run = prepare.run

        def local_transport(*args: str, cwd: Path | None = None) -> str:
            if args[0] == 'gh':
                return self.base
            if 'remote' in args and 'add' in args:
                args = (*args[:-1], str(self.repo))
            return original_run(*args, cwd=cwd)

        with (
            mock.patch.object(prepare, 'gh', return_value=metadata),
            mock.patch.object(prepare, 'run', local_transport),
        ):
            path = prepare.prepare('https://github.com/o/r/pull/1', self.root / 'review', self.repo, history=False)
        return path

    def test_prepare_fetches_fixed_commits_and_detects_source_and_diff_mutation(self) -> None:
        path = self.prepare_local()
        manifest = contract.read_json(path)
        self.assertEqual(manifest['head_sha'], self.head)
        self.assertEqual(manifest['merge_base_sha'], self.base)
        self.assertEqual(manifest['files'], ['example.py'])
        contract.verify_sources(manifest)
        head_file = Path(manifest['roots']['RIGHT']) / 'example.py'
        head_file.write_text('unexpected later source\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            contract.verify_sources(manifest)
        head_file.write_text('new\n')
        Path(manifest['diff_path']).write_text('different diff', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'diff changed'):
            contract.verify_sources(manifest)

    def test_prepared_result_validates_locally_without_publication(self) -> None:
        from test_claude_review_contract import fixture

        manifest_path = self.prepare_local()
        manifest = contract.read_json(manifest_path)
        _, result = fixture()
        result['snapshot_id'] = manifest['snapshot_id']
        for pass_result in result['passes']:
            for check in pass_result['checks']:
                check.update(start_line=1, line=1)
        result_path = manifest_path.parent / 'result.json'
        result_path.write_text(json.dumps(result), encoding='utf-8')
        command = [
            sys.executable,
            str(Path(__file__).parent / 'post_claude_review.py'),
            '--review-context',
            str(manifest_path),
            '--result',
            str(result_path),
            '--validate-only',
        ]
        validated = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
        outcome_path = manifest_path.parent / 'outcome.json'
        self.assertEqual(contract.read_json(outcome_path)['delivery'], 'not_requested')
        self.assertEqual(contract.read_json(manifest_path.parent / 'review.json')['commit_id'], self.head)
        result['snapshot_id'] = 'wrong snapshot'
        result_path.write_text(json.dumps(result), encoding='utf-8')
        rejected = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(rejected.returncode, 1)
        self.assertEqual(contract.read_json(outcome_path)['status'], 'failed')

    def test_rejects_non_pr_urls_and_existing_output(self) -> None:
        with self.assertRaisesRegex(ValueError, 'full GitHub PR URL'):
            prepare.prepare('https://github.com/o/r', self.root / 'out', self.repo)
        metadata = {'state': 'OPEN', 'baseRefOid': self.base, 'headRefOid': self.head}
        with mock.patch.object(prepare, 'gh', return_value=metadata), self.assertRaisesRegex(ValueError, 'must be new'):
            prepare.prepare('https://github.com/o/r/pull/1', self.root, self.repo)

    def test_history_rejects_wrong_workflow_and_malformed_records(self) -> None:
        state = {'schema_version': 1, 'repo': 'o/r', 'pr': 1, 'head_sha': self.head, 'run_id': '123', 'records': []}
        body = contract.STATE_MARKER + json.dumps(state) + '\n-->'
        with mock.patch.object(prepare, 'gh', return_value={'path': 'other.yml'}):
            self.assertEqual(prepare.state_from_review(body, 'o/r', 1, self.head), {})
        workflow = {
            'path': '.github/workflows/claude-review.yml',
            'event': 'issue_comment',
            'repository': {'full_name': 'o/r'},
        }
        with mock.patch.object(prepare, 'gh', return_value=workflow):
            self.assertEqual(prepare.state_from_review(body, 'o/r', 1, self.head), state)
            state['records'] = [{}]
            body = contract.STATE_MARKER + json.dumps(state) + '\n-->'
            self.assertEqual(prepare.state_from_review(body, 'o/r', 1, self.head), {})

    def test_latest_invalid_history_never_revives_an_older_review(self) -> None:
        reviews = [
            [
                {
                    'user': {'login': 'github-actions[bot]'},
                    'body': contract.STATE_MARKER + 'old',
                    'commit_id': self.base,
                },
                {
                    'user': {'login': 'github-actions[bot]'},
                    'body': contract.STATE_MARKER + 'broken',
                    'commit_id': self.head,
                },
            ]
        ]
        with (
            mock.patch.object(prepare, 'gh', return_value=reviews),
            mock.patch.object(prepare, 'state_from_review', return_value={}) as read_state,
        ):
            state = prepare.prior_state('o/r', 1)
        self.assertEqual(state['records'], [])
        self.assertEqual(read_state.call_count, 1)
        self.assertEqual(read_state.call_args.args[-1], self.head)

    def test_history_head_must_match_the_github_review(self) -> None:
        state = {'schema_version': 1, 'repo': 'o/r', 'pr': 1, 'head_sha': self.head, 'run_id': '123', 'records': []}
        body = contract.STATE_MARKER + json.dumps(state) + '\n-->'
        with mock.patch.object(prepare, 'gh') as github:
            self.assertEqual(prepare.state_from_review(body, 'o/r', 1, self.base), {})
        github.assert_not_called()


if __name__ == '__main__':
    unittest.main()
