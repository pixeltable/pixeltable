"""Prepare real Git objects while replacing only the GitHub transport in local tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
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
        (self.repo / 'src').mkdir()
        (self.repo / 'src/nested.py').write_bytes(b'nested\rcontent\n')
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
        self.assertEqual((destination / 'src/nested.py').read_bytes(), b'nested\rcontent\n')
        self.assertEqual(index['src/nested.py']['lines'], 1)
        self.assertFalse((destination / 'AGENTS.md').exists())
        self.assertFalse((destination / '.claude').exists())
        self.assertFalse((destination / 'external-link').is_symlink())
        self.assertEqual((destination / 'external-link').read_text(), '/etc/passwd')
        self.assertTrue(index['AGENTS.md']['stripped'])
        data = json.loads((self.root / 'head-stripped.json').read_text())
        self.assertEqual(data['AGENTS.md'], 'Untrusted PR instruction')

    def prepare_local(self, previous: dict[str, Any] | None = None) -> Path:
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
            mock.patch.object(prepare, 'prior_state', return_value=previous or {}),
        ):
            path = prepare.prepare(
                'https://github.com/o/r/pull/1', self.root / 'review', self.repo, history=bool(previous)
            )
        return path

    def test_prepare_fetches_fixed_commits_and_detects_source_and_diff_mutation(self) -> None:
        path = self.prepare_local()
        raw_manifest = json.loads(path.read_text(encoding='utf-8'))
        self.assertNotIn('sources', raw_manifest)
        self.assertNotIn('diff_files', raw_manifest)
        self.assertNotIn('previous_tree', raw_manifest)
        self.assertNotIn('tooling_files', raw_manifest)
        keys = list(raw_manifest)
        self.assertLess(keys.index('snapshot_id'), keys.index('previous'))
        self.assertLess(keys.index('files'), keys.index('previous'))
        self.assertLess(keys.index('diff_path'), keys.index('previous'))
        self.assertLess(keys.index('metadata'), keys.index('previous'))
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
        sources_path = Path(raw_manifest['sources_path'])
        raw_manifest['sources_path'] = '/nonexistent/dir/sources.json'
        path.write_text(json.dumps(raw_manifest), encoding='utf-8')
        relocated = contract.read_json(path)
        self.assertEqual(relocated['sources_path'], str(sources_path))
        sources_path.write_text('{"tampered": true}\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'source index changed'):
            contract.read_json(path)

    def test_nontext_changes_keep_metadata_without_requiring_text_lines(self) -> None:
        (self.repo / 'image.png').write_bytes(b'\0binary\0')
        self.git('mv', 'ignored.py', 'renamed.py')
        self.git('update-index', '--chmod=+x', 'src/nested.py')
        self.git('add', 'image.png')
        self.git('-c', 'user.name=Review Test', '-c', 'user.email=review@example.invalid', 'commit', '-qm', 'metadata')
        self.head = self.git('rev-parse', 'HEAD')
        manifest = contract.read_json(self.prepare_local())
        self.assertEqual(manifest['diff_files']['image.png']['changed'], {'LEFT': [], 'RIGHT': []})
        self.assertEqual(manifest['diff_files']['renamed.py']['old_path'], 'ignored.py')
        self.assertEqual(manifest['sources']['RIGHT']['src/nested.py']['mode'], '100755')
        self.assertEqual(manifest['diff_files']['src/nested.py']['changed'], {'LEFT': [], 'RIGHT': []})
        from test_claude_review_contract import fixture

        _, result = fixture()
        result['snapshot_id'] = manifest['snapshot_id']
        for entry in result['passes']:
            for check in entry['checks']:
                check.update(start_line=1, line=1)
        self.assertEqual(contract.render(result, manifest)[1]['status'], 'complete')

    def test_manifest_size_is_rejected_during_preparation(self) -> None:
        with mock.patch.object(prepare, 'MAX_JSON_BYTES', 1000), self.assertRaisesRegex(ValueError, 'manifest.*limit'):
            self.prepare_local()

    def test_a_followup_rename_of_an_added_file_is_not_a_deletion(self) -> None:
        from test_claude_review_contract import add_finding, finding, fixture

        (self.repo / 'added.py').write_text('added\n')
        self.git('add', 'added.py')
        self.git('-c', 'user.name=Review Test', '-c', 'user.email=review@example.invalid', 'commit', '-qm', 'added')
        previous_head = self.git('rev-parse', 'HEAD')
        claim = finding()
        claim.update(subject='added.py#contract', evidence=[{'path': 'added.py', 'line': 1}])
        previous = {'head_sha': previous_head, 'records': [{'claim': claim, 'status': 'keep', 'posted': True}]}
        self.git('mv', 'added.py', 'renamed.py')
        self.git('-c', 'user.name=Review Test', '-c', 'user.email=review@example.invalid', 'commit', '-qm', 'renamed')
        self.head = self.git('rev-parse', 'HEAD')
        manifest = contract.read_json(self.prepare_local(previous))
        self.assertEqual(manifest['diff_files']['renamed.py']['status'], 'A')
        self.assertEqual(manifest['previous_renames'], {'added.py': 'renamed.py'})
        _, result = fixture()
        result.update(snapshot_id=manifest['snapshot_id'], status='incomplete', reason='Historical check only')
        for entry in result['passes']:
            entry['checks'] = []
            entry['status'] = 'incomplete'
            entry['reason'] = 'Historical check only'
        add_finding(result, claim)
        result['decisions'][0].update(decision='resolved', removed_paths=['added.py'])
        review, stats = contract.render(result, manifest)
        self.assertTrue(any('moved' in gap for gap in stats['gaps']))
        self.assertNotIn('Resolved:', review['body'])
        result['decisions'][0].pop('removed_paths')
        result['decisions'][0]['fix_evidence'] = [{'path': 'renamed.py', 'line': 1}]
        review, stats = contract.render(result, manifest)
        self.assertTrue(any('no demonstrated source change' in gap for gap in stats['gaps']))
        self.assertNotIn('Resolved:', review['body'])
        result['findings'][0].update(subject='renamed.py#contract', evidence=[{'path': 'renamed.py', 'line': 1}])
        result['decisions'][0].update(decision='keep', checked=dict(result['findings'][0]))
        self.assertEqual(contract.render(result, manifest)[1]['published'], 1)

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
        result['passes'].pop()
        result_path.write_text(json.dumps(result), encoding='utf-8')
        incomplete = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(incomplete.returncode, 0, incomplete.stdout + incomplete.stderr)
        self.assertEqual(contract.read_json(outcome_path)['status'], 'incomplete')
        self.assertIn('coverage gaps', contract.read_json(outcome_path)['reason'])
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

    def test_sync_compares_working_trees_and_merge_previews(self) -> None:
        import check_claude_review_sync as sync

        repos = [self.root / 'left_repo', self.root / 'right_repo']
        for repo in repos:
            repo.mkdir()
            subprocess.check_call(['git', 'init', '--quiet', '-b', 'main'], cwd=repo)
            subprocess.check_call(['git', 'config', 'core.hooksPath', '/dev/null'], cwd=repo)
            (repo / 'README.md').write_text('base\n', encoding='utf-8')
            for rel in sync.SHARED_FILES:
                target = repo / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f'shared {rel}\n', encoding='utf-8')
            subprocess.check_call(['git', 'add', '.'], cwd=repo)
            subprocess.check_call(
                ['git', '-c', 'user.name=Test', '-c', 'user.email=t@example.invalid', 'commit', '-qm', 'base'], cwd=repo
            )
            subprocess.check_call(['git', 'update-ref', 'refs/remotes/origin/main', 'HEAD'], cwd=repo)
            subprocess.check_call(['git', 'checkout', '-qb', 'feature'], cwd=repo)
        sync.check_sync(repos[0], repos[1])
        # A conflict in an unrelated file on origin/main does not fail shared-file sync when shared blobs match.
        subprocess.check_call(['git', 'checkout', '-q', 'main'], cwd=repos[0])
        (repos[0] / 'README.md').write_text('main readme\n', encoding='utf-8')
        subprocess.check_call(
            ['git', '-c', 'user.name=Test', '-c', 'user.email=t@example.invalid', 'commit', '-qam', 'main readme'],
            cwd=repos[0],
        )
        subprocess.check_call(['git', 'update-ref', 'refs/remotes/origin/main', 'HEAD'], cwd=repos[0])
        subprocess.check_call(['git', 'checkout', '-q', 'feature'], cwd=repos[0])
        (repos[0] / 'README.md').write_text('feature readme\n', encoding='utf-8')
        subprocess.check_call(
            ['git', '-c', 'user.name=Test', '-c', 'user.email=t@example.invalid', 'commit', '-qam', 'feature readme'],
            cwd=repos[0],
        )
        sync.check_sync(repos[0], repos[1])
        # If one repo commits a shared-file update and the other only has it in its working tree, merge preview fails.
        shared = sync.SHARED_FILES[0]
        (repos[0] / shared).write_text('updated\n', encoding='utf-8')
        (repos[1] / shared).write_text('updated\n', encoding='utf-8')
        subprocess.check_call(
            ['git', '-c', 'user.name=Test', '-c', 'user.email=t@example.invalid', 'commit', '-qam', 'shared update'],
            cwd=repos[0],
        )
        with self.assertRaisesRegex(SystemExit, 'merge preview differs'):
            sync.check_sync(repos[0], repos[1])
        # A merge conflict on a shared file against origin/main is rejected.
        subprocess.check_call(['git', 'checkout', '-q', 'main'], cwd=repos[0])
        (repos[0] / shared).write_text('conflicting main update\n', encoding='utf-8')
        subprocess.check_call(
            ['git', '-c', 'user.name=Test', '-c', 'user.email=t@example.invalid', 'commit', '-qam', 'conflict'],
            cwd=repos[0],
        )
        subprocess.check_call(['git', 'update-ref', 'refs/remotes/origin/main', 'HEAD'], cwd=repos[0])
        subprocess.check_call(['git', 'checkout', '-q', 'feature'], cwd=repos[0])
        with self.assertRaisesRegex(SystemExit, 'merge conflicts'):
            sync.check_sync(repos[0], repos[1])


if __name__ == '__main__':
    unittest.main()
