"""Observable contract, coverage, rendering, and history regressions."""

from __future__ import annotations

import copy
import unittest
from typing import Any

import claude_review_contract as contract

PATCH = """diff --git a/example.py b/example.py
index aaaaaaa..bbbbbbb 100644
--- a/example.py
+++ b/example.py
@@ -1,4 +1,4 @@
 one
-old
+new
 three
 four
@@ -10,3 +10,3 @@
 ten
-old eleven
+new eleven
 twelve
\\ No newline at end of file
"""


def fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest: dict[str, Any] = {
        'schema_version': 1,
        'snapshot_id': 'snapshot',
        'repo': 'o/r',
        'pr': 1,
        'base_sha': 'a' * 40,
        'head_sha': 'b' * 40,
        'merge_base_sha': 'a' * 40,
        'diff_files': contract.parse_diff(PATCH),
        'files': ['example.py'],
        'sources': {side: {'example.py': {'lines': 20, 'sha256': 'content'}} for side in ['LEFT', 'RIGHT']},
        'required_passes': list(contract.REQUIRED_PASSES),
        'previous': {},
        'focus': '',
    }
    checks = [
        {'path': 'example.py', 'side': side, 'start_line': 1, 'line': 20, 'check': 'Trace the condition and its caller'}
        for side in ['LEFT', 'RIGHT']
    ]
    result = {
        'schema_version': 1,
        'snapshot_id': 'snapshot',
        'status': 'complete',
        'reason': '',
        'assessment': 'The change fixes the caller contract.',
        'user_surface': False,
        'passes': [
            {'pass': name, 'status': 'complete', 'reason': '', 'checks': copy.deepcopy(checks), 'exclusions': []}
            for name in contract.REQUIRED_PASSES
        ],
        'findings': [],
        'decisions': [],
        'questions': [],
    }
    return manifest, result


def finding(key: str = 'example:guard') -> dict[str, Any]:
    return {
        'id': key,
        'producer': 'lines',
        'subject': 'example.py#caller',
        'mechanism': 'missing guard',
        'kind': 'defect',
        'title': 'Validate the input',
        'comment': 'An empty input reaches the caller and fails.',
        'trigger': 'empty input',
        'impact': 'the request fails',
        'correction': 'validate the input',
        'attribution': 'new',
        'severity': 'Medium',
        'evidence': [{'path': 'example.py', 'side': 'RIGHT', 'line': 2}],
        'location': {'path': 'example.py', 'side': 'RIGHT', 'line': 2},
    }


def add_finding(data: dict[str, Any], claim: dict[str, Any]) -> None:
    data['findings'].append(claim)
    data['decisions'].append(
        {
            'finding_id': claim['id'],
            'checker': 'cross-check',
            'decision': 'keep',
            'reason': 'The caller has no preventing guard.',
            'checked': copy.deepcopy(claim),
        }
    )


class DiffTest(unittest.TestCase):
    def test_only_valid_ranges_stay_inline(self) -> None:
        files = contract.parse_diff(PATCH)
        cases = [(2, 2, True), (1, 3, True), (4, 10, False), (5, 5, False), (3, 2, False), (0, 2, False)]
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                self.assertEqual(
                    contract.inline_valid({'path': 'example.py', 'start_line': start, 'line': end}, files), expected
                )
        self.assertFalse(contract.inline_valid({'path': '../example.py', 'line': 2}, files))
        self.assertFalse(contract.inline_valid({'path': 'example.py', 'line': True}, files))

    def test_renamed_and_deleted_paths(self) -> None:
        renamed = PATCH.replace('a/example.py', 'a/old.py').replace('b/example.py', 'b/new.py')
        files = contract.parse_diff(renamed)
        self.assertEqual(files['new.py']['old_path'], 'old.py')
        self.assertTrue(contract.inline_valid({'path': 'new.py', 'line': 2}, files))
        deleted = 'diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n'
        files = contract.parse_diff(deleted)
        self.assertTrue(contract.inline_valid({'path': 'gone.py', 'side': 'LEFT', 'line': 1}, files))
        self.assertFalse(contract.inline_valid({'path': 'gone.py', 'line': 1}, files))

    def test_quoted_path_and_hunk_content_that_starts_with_header_characters(self) -> None:
        patch = (
            'diff --git "a/a\\303\\251.py" "b/a\\303\\251.py"\n'
            '--- "a/a\\303\\251.py"\n+++ "b/a\\303\\251.py"\n@@ -1 +1 @@\n-old\n+new\n'
        )
        self.assertIn('aé.py', contract.parse_diff(patch))
        patch = 'diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n--- old text\n+++ new text\n'
        self.assertEqual(contract.parse_diff(patch)['f']['changed'], {'LEFT': [1], 'RIGHT': [1]})


class ContractTest(unittest.TestCase):
    def test_empty_complete_review(self) -> None:
        manifest, result = fixture()
        review, stats = contract.render(result, manifest)
        self.assertIn('No supported findings.', review['body'])
        self.assertEqual(stats['status'], 'complete')

    def test_missing_pass_cannot_claim_complete(self) -> None:
        manifest, result = fixture()
        result['passes'].pop()
        with self.assertRaisesRegex(ValueError, 'no result'):
            contract.validate(result, manifest)
        result.update(status='incomplete', reason='The rules pass failed')
        review, _ = contract.render(result, manifest)
        self.assertIn('Coverage incomplete', review['body'])
        self.assertNotIn('No supported findings', review['body'])

    def test_line_and_rules_passes_account_for_all_changed_ranges(self) -> None:
        manifest, result = fixture()
        result['passes'][1]['checks'] = []
        with self.assertRaisesRegex(ValueError, 'uncovered'):
            contract.validate(result, manifest)
        result['passes'][1]['exclusions'] = [{'path': 'example.py', 'reason': 'Generated file'}]
        contract.validate(result, manifest)

    def test_wrong_snapshot_and_duplicate_ids_fail(self) -> None:
        manifest, result = fixture()
        result['snapshot_id'] = 'a different revision'
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            contract.validate(result, manifest)
        result['snapshot_id'] = manifest['snapshot_id']
        add_finding(result, finding())
        result['findings'].append(finding())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            contract.validate(result, manifest)

    def test_unchecked_claim_cannot_publish(self) -> None:
        manifest, result = fixture()
        result['findings'].append(finding())
        with self.assertRaisesRegex(ValueError, 'independently checked'):
            contract.validate(result, manifest)
        result.update(status='incomplete', reason='Verification exhausted its budget')
        review, _ = contract.render(result, manifest)
        self.assertEqual(review['comments'], [])
        self.assertNotIn('Validate the input', review['body'])

    def test_renderer_uses_checked_replacement(self) -> None:
        manifest, result = fixture()
        add_finding(result, finding())
        result['findings'][0]['comment'] = 'An unsupported stronger accusation.'
        review, _ = contract.render(result, manifest)
        self.assertNotIn('unsupported stronger', review['comments'][0]['body'])
        result['decisions'][0]['checked']['mechanism'] = 'different cause'
        with self.assertRaisesRegex(ValueError, 'identity'):
            contract.validate(result, manifest)

    def test_only_one_invalid_location_moves_to_body(self) -> None:
        manifest, result = fixture()
        add_finding(result, finding('first'))
        bad = finding('second')
        bad['location']['line'] = 8
        add_finding(result, bad)
        review, _ = contract.render(result, manifest)
        self.assertEqual(len(review['comments']), 1)
        self.assertEqual(review['comments'][0]['line'], 2)
        self.assertIn('An empty input reaches the caller', review['body'])

    def test_source_references_and_nondefect_severity(self) -> None:
        manifest, result = fixture()
        claim = finding()
        claim['evidence'][0]['line'] = 100
        add_finding(result, claim)
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            contract.validate(result, manifest)
        result['findings'] = []
        result['decisions'] = []
        claim = finding()
        claim['kind'] = 'design'
        add_finding(result, claim)
        with self.assertRaisesRegex(ValueError, 'Only defects'):
            contract.validate(result, manifest)

    def test_independent_checker_is_required(self) -> None:
        manifest, result = fixture()
        add_finding(result, finding())
        result['decisions'][0]['checker'] = 'lines'
        with self.assertRaisesRegex(ValueError, 'independent'):
            contract.validate(result, manifest)

    def test_question_names_unknown_and_check(self) -> None:
        manifest, result = fixture()
        add_finding(result, finding())
        result['decisions'][0].update(decision='question', unknown='Which caller owns this input?')
        with self.assertRaisesRegex(ValueError, 'check'):
            contract.validate(result, manifest)
        result['decisions'][0]['check'] = 'Read the external caller contract.'
        with self.assertRaisesRegex(ValueError, 'decisive fact remains unresolved'):
            contract.validate(result, manifest)
        result.update(status='incomplete', reason='The external caller contract is unavailable')
        review, _ = contract.render(result, manifest)
        self.assertIn('Which caller owns this input?', review['body'])
        self.assertEqual(review['comments'], [])

    def test_focus_and_user_job_require_their_passes(self) -> None:
        manifest, result = fixture()
        result['user_surface'] = True
        with self.assertRaisesRegex(ValueError, 'walk'):
            contract.validate(result, manifest)
        result['user_surface'] = False
        manifest['required_passes'].append('focus')
        with self.assertRaisesRegex(ValueError, 'focus'):
            contract.validate(result, manifest)

    def test_three_runs_preserve_history_and_recheck_external_changes(self) -> None:
        manifest, first = fixture()
        add_finding(first, finding())
        posted, _ = contract.render(first, manifest)
        state = posted['body'].split(contract.STATE_MARKER)[1].split('\n-->')[0]
        import json

        manifest['previous'] = json.loads(state)
        _, second = fixture()
        add_finding(second, finding())
        unchanged, _ = contract.render(second, manifest)
        self.assertEqual(unchanged['comments'], [])
        state = unchanged['body'].split(contract.STATE_MARKER)[1].split('\n-->')[0]
        manifest['previous'] = json.loads(state)
        _, third = fixture()
        add_finding(third, finding())
        third['decisions'][0].update(
            decision='resolved',
            reason='The external guard now rejects empty input.',
            fix_evidence=[{'path': 'guard.py', 'line': 1}],
        )
        manifest['previous_tree'] = {'example.py': 'same', 'guard.py': 'old guard'}
        manifest['sources']['RIGHT']['guard.py'] = {'lines': 1, 'sha256': 'new guard content', 'git_blob': 'new guard'}
        resolved, _ = contract.render(third, manifest)
        self.assertIn('Resolved:', resolved['body'])
        self.assertEqual(resolved['comments'], [])
        third['decisions'][0]['decision'] = 'reject'
        rejected, _ = contract.render(third, manifest)
        state = rejected['body'].split(contract.STATE_MARKER)[1].split('\n-->')[0]
        manifest['previous'] = json.loads(state)
        _, fourth = fixture()
        add_finding(fourth, finding())
        new, _ = contract.render(fourth, manifest)
        self.assertEqual(len(new['comments']), 1)

    def test_unrevalidated_history_is_incomplete(self) -> None:
        manifest, result = fixture()
        manifest['previous'] = {'records': [{'claim': finding(), 'posted': True, 'status': 'keep'}]}
        with self.assertRaisesRegex(ValueError, 'prior claim not revalidated'):
            contract.validate(result, manifest)

    def test_prior_id_cannot_be_reused_for_another_claim(self) -> None:
        manifest, result = fixture()
        earlier = finding()
        manifest['previous'] = {'records': [{'claim': earlier, 'posted': True, 'status': 'keep'}]}
        current = finding()
        current['subject'] = 'different.py#other'
        add_finding(result, current)
        with self.assertRaisesRegex(ValueError, 'different identity'):
            contract.validate(result, manifest)

    def test_resolution_requires_a_changed_current_source_blob(self) -> None:
        manifest, result = fixture()
        manifest['previous'] = {'records': [{'claim': finding(), 'posted': True, 'status': 'keep'}]}
        manifest['previous_tree'] = {'example.py': 'same blob'}
        manifest['sources']['RIGHT']['example.py']['git_blob'] = 'same blob'
        add_finding(result, finding())
        result['decisions'][0].update(decision='resolved', fix_evidence=[{'path': 'example.py', 'line': 2}])
        with self.assertRaisesRegex(ValueError, 'no demonstrated source change'):
            contract.validate(result, manifest)
        manifest['sources']['RIGHT']['example.py']['git_blob'] = 'changed blob'
        review, _ = contract.render(result, manifest)
        self.assertIn('Resolved:', review['body'])

    def test_changed_severity_is_not_suppressed(self) -> None:
        import json

        manifest, first = fixture()
        add_finding(first, finding())
        review, _ = contract.render(first, manifest)
        manifest['previous'] = json.loads(review['body'].split(contract.STATE_MARKER)[1].split('\n-->')[0])
        _, second = fixture()
        claim = finding()
        claim['severity'] = 'High'
        add_finding(second, claim)
        changed, _ = contract.render(second, manifest)
        self.assertEqual(len(changed['comments']), 1)
        self.assertIn('High:', changed['comments'][0]['body'])

    def test_whole_change_pass_cannot_claim_complete_without_evidence(self) -> None:
        manifest, result = fixture()
        result['passes'][2]['checks'] = []
        with self.assertRaisesRegex(ValueError, 'no source check'):
            contract.validate(result, manifest)

    def test_resolution_can_show_a_removed_path(self) -> None:
        manifest, result = fixture()
        claim = finding()
        claim['evidence'][0]['side'] = 'LEFT'
        manifest['previous'] = {'records': [{'claim': claim, 'posted': True, 'status': 'keep'}]}
        manifest['previous_tree'] = {'example.py': 'old blob'}
        manifest['sources']['RIGHT'] = {}
        manifest['diff_files'] = contract.parse_diff(
            'diff --git a/example.py b/example.py\n--- a/example.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n'
        )
        for entry in result['passes']:
            entry['checks'] = [check for check in entry['checks'] if check['side'] == 'LEFT']
        add_finding(result, claim)
        result['decisions'][0].update(decision='resolved', removed_paths=['example.py'])
        review, _ = contract.render(result, manifest)
        self.assertIn('Resolved:', review['body'])
        result['decisions'][0]['removed_paths'] = ['never-existed.py']
        with self.assertRaisesRegex(ValueError, 'did not remove'):
            contract.validate(result, manifest)

    def test_model_cannot_skip_missing_work_without_a_host_policy(self) -> None:
        manifest, result = fixture()
        result.update(status='skipped', reason='Nothing needs review', passes=[])
        with self.assertRaisesRegex(ValueError, 'host-approved'):
            contract.validate(result, manifest)
        manifest.update(skip_reason='The reviewed commits have no changed paths.', files=[])
        result['reason'] = manifest['skip_reason']
        contract.validate(result, manifest)


if __name__ == '__main__':
    unittest.main()
