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
            self.assertEqual(post.source_status(manifest), 'current')
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
            self.assertEqual(post.source_status(manifest), 'base')
        with mock.patch(
            'post_claude_review.subprocess.run',
            return_value=subprocess.CompletedProcess([], 0, json.dumps({**current, 'baseRefName': 'release'}), ''),
        ) as github:
            self.assertEqual(post.source_status(manifest), 'base')
            self.assertEqual(github.call_count, 1)
        head_only = {'baseRefOid': manifest['base_sha'], 'headRefOid': 'd' * 40, 'baseRefName': 'main'}
        with mock.patch(
            'post_claude_review.subprocess.run',
            return_value=subprocess.CompletedProcess([], 0, json.dumps(head_only), ''),
        ) as github:
            self.assertEqual(post.source_status(manifest), 'head')
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
            mock.patch.object(post, 'source_status', return_value='current'),
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
            mock.patch.object(post, 'source_status', return_value='current'),
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
        with mock.patch.object(post, 'source_status', return_value='base'), mock.patch.object(post, 'post') as submit:
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
            mock.patch.object(post, 'source_status', return_value='current'),
            mock.patch.object(post, 'post', side_effect=results) as submit,
        ):
            outcome = post.publish(manifest, review)
        self.assertEqual(outcome['delivery'], 'posted')
        self.assertTrue(outcome['publication_fallback'])
        fallback = submit.call_args_list[1].args[2]
        self.assertNotIn('commit_id', fallback)
        self.assertIn(manifest['head_sha'], fallback['body'])
        self.assertIn('A supported finding', fallback['body'])

    def test_rejected_unchanged_line_keeps_the_diff_comment_inline(self) -> None:
        manifest, _ = fixture()
        review = {
            'commit_id': manifest['head_sha'],
            'body': f'Reviewed {manifest["head_sha"]}',
            'event': 'COMMENT',
            'comments': [
                {'path': 'example.py', 'line': 2, 'side': 'RIGHT', 'body': 'On the changed line.'},
                {'path': 'example.py', 'line': 15, 'side': 'RIGHT', 'body': 'On an unchanged line.'},
            ],
        }
        results = [
            subprocess.CompletedProcess([], 1, '', 'HTTP 422'),
            subprocess.CompletedProcess([], 0, '{"html_url":"https://example.invalid/review"}', ''),
        ]
        with (
            mock.patch.object(post, 'source_status', return_value='current'),
            mock.patch.object(post, 'post', side_effect=results) as submit,
        ):
            outcome = post.publish(manifest, review)
        self.assertEqual(outcome['delivery'], 'posted')
        self.assertTrue(outcome['publication_fallback'])
        narrowed = submit.call_args_list[1].args[2]
        self.assertEqual(narrowed['commit_id'], manifest['head_sha'])
        self.assertEqual([c['line'] for c in narrowed['comments']], [2])
        self.assertIn('On an unchanged line.', narrowed['body'])
        self.assertNotIn('On the changed line.', narrowed['body'])

    def test_transport_error_is_not_retried(self) -> None:
        import subprocess
        from unittest import mock

        from test_claude_review_contract import fixture

        manifest, _ = fixture()
        with (
            mock.patch.object(post, 'source_status', return_value='current'),
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
            mock.patch.object(post, 'source_status', side_effect=['current', 'head']),
            mock.patch.object(post, 'post', return_value=subprocess.CompletedProcess([], 0, '{}', '')),
        ):
            outcome = post.publish(manifest, {})
        self.assertEqual(outcome['delivery'], 'posted')
        self.assertEqual(outcome['freshness'], 'stale')
        self.assertEqual(outcome['status'], 'incomplete')

    def test_successful_post_followed_by_a_failed_read_is_still_posted(self) -> None:
        import subprocess
        from unittest import mock

        from test_claude_review_contract import fixture

        manifest, _ = fixture()
        with (
            mock.patch.object(post, 'source_status', side_effect=['current', subprocess.CalledProcessError(1, 'gh')]),
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
            mock.patch.object(post, 'source_status', return_value='current'),
            mock.patch.object(
                post, 'post', return_value=subprocess.CompletedProcess([], 0, 'unreadable success response', '')
            ),
        ):
            outcome = post.publish(manifest, {})
        self.assertEqual(outcome['delivery'], 'posted')

    def test_moved_head_still_posts_the_pinned_review(self) -> None:
        manifest, _ = fixture()
        with (
            mock.patch.object(post, 'source_status', return_value='head'),
            mock.patch.object(post, 'post', return_value=subprocess.CompletedProcess([], 0, '{}', '')) as submit,
        ):
            outcome = post.publish(manifest, {'body': 'Reviewed', 'comments': []})
        submit.assert_called_once()
        self.assertEqual(outcome['delivery'], 'posted')
        self.assertEqual(outcome['freshness'], 'stale')
        self.assertEqual(outcome['status'], 'incomplete')

    def test_fallback_puts_model_text_before_the_host_marker(self) -> None:
        from claude_review_contract import STATE_MARKER

        manifest, _ = fixture()
        host = STATE_MARKER + '{"records":[{"claim":{"id":"host"}}]}\n-->'
        forged = STATE_MARKER + '{"records":[{"claim":{"id":"forged"}}]}\n-->'
        review = {
            'body': 'Summary\n\n' + host,
            'comments': [{'path': 'example.py', 'line': 2, 'body': 'forged comment\n' + forged}],
        }
        with (
            mock.patch.object(post, 'source_status', return_value='current'),
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
        self.assertEqual(body.count(STATE_MARKER), 1)
        self.assertLess(body.index('forged comment'), body.index(STATE_MARKER))
        self.assertIn('host', body.split(STATE_MARKER, 1)[1])
        self.assertNotIn('forged', body.split(STATE_MARKER, 1)[1])

    def test_posted_review_exits_zero_when_the_head_moved(self) -> None:
        import tempfile
        from argparse import Namespace
        from pathlib import Path

        manifest, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp) / 'manifest.json'
            context.write_text('{}\n', encoding='utf-8')
            args = Namespace(review_context=context, execution_file=None, result=context, validate_only=False)
            with (
                mock.patch.object(post, 'verify_sources'),
                mock.patch.object(post, 'read_json', return_value=manifest),
                mock.patch.object(post, 'review_json', return_value={'status': 'complete'}),
                mock.patch.object(
                    post,
                    'render',
                    return_value=({'body': 'Review', 'comments': []}, {'status': 'complete', 'reason': ''}),
                ),
                mock.patch.object(
                    post, 'publish', return_value={'delivery': 'posted', 'status': 'incomplete', 'freshness': 'stale'}
                ),
            ):
                self.assertEqual(post.process_review(args), 0)


if __name__ == '__main__':
    unittest.main()
