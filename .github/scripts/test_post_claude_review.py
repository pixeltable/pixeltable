"""Regression tests for review publication."""

import unittest

import post_claude_review as post


class ReviewJsonTest(unittest.TestCase):
    def test_array_is_not_a_review(self) -> None:
        self.assertIsNone(post.review_json('```json\n[]\n```'))


class PublishTest(unittest.TestCase):
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
