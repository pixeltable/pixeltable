"""Regression tests for review publication."""

import json
import subprocess
import unittest
from typing import Any
from unittest import mock

import post_claude_review as post
from test_claude_review_contract import fixture


class ReviewJsonTest(unittest.TestCase):
    def test_array_is_not_a_review(self) -> None:
        self.assertIsNone(post.review_json('```json\n[]\n```'))

    def test_single_fence_allows_surrounding_prose_but_not_ambiguous_blocks(self) -> None:
        block = '```json\n{"schema_version": 1}\n```'
        self.assertEqual(post.review_json('Here is the review:\n' + block + '\nDone.'), {'schema_version': 1})
        self.assertIsNone(post.review_json(block + '\n' + block))
        self.assertIsNone(post.review_json(block + '\n```json\nunfinished'))


class CurrentTargetTest(unittest.TestCase):
    def test_base_advance_preserves_the_review_when_the_merge_base_is_unchanged(self) -> None:
        manifest, _ = fixture()
        manifest['pr_url'] = 'https://github.com/o/r/pull/1'
        current = {'baseRefOid': 'c' * 40, 'headRefOid': manifest['head_sha'], 'baseRefName': 'main'}
        responses = [
            subprocess.CompletedProcess([], 0, json.dumps(current), ''),
            subprocess.CompletedProcess([], 0, manifest['merge_base_sha'] + '\n', ''),
        ]
        with mock.patch('post_claude_review.subprocess.run', side_effect=responses) as github:
            self.assertTrue(post.current_target(manifest))
        self.assertIn(f'repos/o/r/compare/{current["baseRefOid"]}...{manifest["head_sha"]}', github.call_args.args[0])

    def test_changed_comparison_base_or_target_is_stale(self) -> None:
        manifest, _ = fixture()
        manifest['pr_url'] = 'https://github.com/o/r/pull/1'
        current = {'baseRefOid': 'c' * 40, 'headRefOid': manifest['head_sha'], 'baseRefName': 'main'}
        with mock.patch(
            'post_claude_review.subprocess.run',
            side_effect=[
                subprocess.CompletedProcess([], 0, json.dumps(current), ''),
                subprocess.CompletedProcess([], 0, 'd' * 40, ''),
            ],
        ):
            self.assertFalse(post.current_target(manifest))
        for field, changed in [('baseRefName', 'release'), ('headRefOid', 'd' * 40)]:
            with mock.patch(
                'post_claude_review.subprocess.run',
                return_value=subprocess.CompletedProcess([], 0, json.dumps({**current, field: changed}), ''),
            ) as github:
                self.assertFalse(post.current_target(manifest))
                self.assertEqual(github.call_count, 1)


class PublishTest(unittest.TestCase):
    def test_fallback_drops_history_before_exceeding_the_body_limit(self) -> None:
        from claude_review_contract import STATE_MARKER

        manifest, _ = fixture()
        review: dict[str, Any] = {
            'body': 'Summary\n\n' + STATE_MARKER + 'x' * 59_000 + '\n-->',
            'comments': [{'path': 'example.py', 'line': 2, 'body': 'finding ' * 1000}],
        }
        with (
            mock.patch.object(post, 'current_target', return_value=True),
            mock.patch.object(
                post,
                'post',
                side_effect=[
                    subprocess.CompletedProcess([], 1, '', 'HTTP 422'),
                    subprocess.CompletedProcess([], 0, '{}', ''),
                ],
            ) as submit,
        ):
            outcome = post.publish(manifest, review)
        self.assertEqual(outcome['delivery'], 'posted')
        body = submit.call_args_list[1].args[2]['body']
        self.assertLessEqual(len(body.encode()), 60_000)
        self.assertIn(review['comments'][0]['body'], body)
        self.assertNotIn(STATE_MARKER, body)

    def test_oversized_fallback_never_sends_an_oversized_retry(self) -> None:
        manifest, _ = fixture()
        review = {'body': 'x' * 59_000, 'comments': [{'path': 'example.py', 'line': 2, 'body': 'y' * 7000}]}
        with (
            mock.patch.object(post, 'current_target', return_value=True),
            mock.patch.object(post, 'post', return_value=subprocess.CompletedProcess([], 1, '', 'HTTP 422')) as submit,
        ):
            outcome = post.publish(manifest, review)
        self.assertEqual(outcome['delivery'], 'failed')
        self.assertIn('limit', outcome['reason'])
        self.assertEqual(submit.call_count, 1)

    def test_stale_source_does_not_post(self) -> None:
        from unittest import mock

        from test_claude_review_contract import fixture

        manifest, _ = fixture()
        with mock.patch.object(post, 'current_target', return_value=False), mock.patch.object(post, 'post') as submit:
            outcome = post.publish(manifest, {})
        self.assertEqual(outcome['delivery'], 'not_posted')
        submit.assert_not_called()

    def test_validation_fallback_preserves_reviewed_sha(self) -> None:
        import subprocess
        from unittest import mock

        from test_claude_review_contract import fixture

        manifest, _ = fixture()
        review = {
            'body': f'Reviewed {manifest["head_sha"]}',
            'commit_id': manifest['head_sha'],
            'comments': [{'path': 'example.py', 'line': 2, 'body': 'A supported finding'}],
        }
        results = [
            subprocess.CompletedProcess([], 1, '', 'HTTP 422'),
            subprocess.CompletedProcess([], 0, '{"html_url":"https://example.invalid/review"}', ''),
        ]
        with (
            mock.patch.object(post, 'current_target', return_value=True),
            mock.patch.object(post, 'post', side_effect=results) as submit,
        ):
            outcome = post.publish(manifest, review)
        self.assertEqual(outcome['delivery'], 'posted')
        self.assertTrue(outcome['publication_fallback'])
        fallback = submit.call_args_list[1].args[2]
        self.assertNotIn('commit_id', fallback)
        self.assertIn(manifest['head_sha'], fallback['body'])
        self.assertIn('A supported finding', fallback['body'])

    def test_transport_error_is_not_retried(self) -> None:
        import subprocess
        from unittest import mock

        from test_claude_review_contract import fixture

        manifest, _ = fixture()
        with (
            mock.patch.object(post, 'current_target', return_value=True),
            mock.patch.object(
                post, 'post', return_value=subprocess.CompletedProcess([], 1, '', 'connection lost')
            ) as submit,
        ):
            outcome = post.publish(manifest, {})
        self.assertEqual(outcome['delivery'], 'unknown')
        self.assertEqual(submit.call_count, 1)

    def test_push_during_post_is_recorded_as_stale(self) -> None:
        import subprocess
        from unittest import mock

        from test_claude_review_contract import fixture

        manifest, _ = fixture()
        with (
            mock.patch.object(post, 'current_target', side_effect=[True, False]),
            mock.patch.object(post, 'post', return_value=subprocess.CompletedProcess([], 0, '{}', '')),
        ):
            outcome = post.publish(manifest, {})
        self.assertEqual(outcome['delivery'], 'posted_stale')
        self.assertEqual(outcome['status'], 'incomplete')

    def test_successful_post_followed_by_a_failed_read_is_still_posted(self) -> None:
        import subprocess
        from unittest import mock

        from test_claude_review_contract import fixture

        manifest, _ = fixture()
        with (
            mock.patch.object(post, 'current_target', side_effect=[True, subprocess.CalledProcessError(1, 'gh')]),
            mock.patch.object(
                post,
                'post',
                return_value=subprocess.CompletedProcess([], 0, '{"html_url":"https://example.invalid/review"}', ''),
            ),
        ):
            outcome = post.publish(manifest, {})
        self.assertEqual(outcome['delivery'], 'posted')
        self.assertEqual(outcome['freshness'], 'unknown')
        self.assertEqual(outcome['review_url'], 'https://example.invalid/review')

    def test_unreadable_successful_response_never_claims_not_posted(self) -> None:
        import subprocess
        from unittest import mock

        from test_claude_review_contract import fixture

        manifest, _ = fixture()
        with (
            mock.patch.object(post, 'current_target', return_value=True),
            mock.patch.object(
                post, 'post', return_value=subprocess.CompletedProcess([], 0, 'unreadable success response', '')
            ),
        ):
            outcome = post.publish(manifest, {})
        self.assertEqual(outcome['delivery'], 'posted')


if __name__ == '__main__':
    unittest.main()
