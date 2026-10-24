"""Validate a prepared review, then render it locally or publish one GitHub review."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from claude_review_contract import MAX_JSON_BYTES, publication_body, read_json, render, verify_sources


def final_message(execution_file: str) -> str:
    with open(execution_file, encoding='utf-8') as f:
        messages = json.load(f)
    if isinstance(messages, dict):
        return messages.get('result') or ''
    results = [m for m in messages if isinstance(m, dict) and m.get('type') == 'result']
    return (results[-1].get('result') or '') if results else ''


def review_json(message: str) -> dict[str, Any] | None:
    if len(message.encode()) > MAX_JSON_BYTES:
        return None
    raw = message.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        opens = re.findall(r'^```json[ \t]*\r?$', raw, re.M)
        blocks = re.findall(r'^```json[ \t]*\r?\n(.*?)^```[ \t]*\r?$', raw, re.M | re.S)
        if len(opens) != 1 or len(blocks) != 1:
            return None
        try:
            data = json.loads(blocks[0])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def post(repo: str, pr: str, review: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['gh', 'api', '-X', 'POST', f'repos/{repo}/pulls/{pr}/reviews', '--input', '-'],
        input=json.dumps(review),
        text=True,
        capture_output=True,
        check=False,
    )


def current_target(manifest: dict[str, Any]) -> bool:
    result = subprocess.run(
        ['gh', 'pr', 'view', manifest['pr_url'], '--json', 'baseRefOid,headRefOid,baseRefName'],
        text=True,
        capture_output=True,
        check=True,
    )
    current = json.loads(result.stdout)
    if current['headRefOid'] != manifest['head_sha'] or current['baseRefName'] != manifest['metadata']['baseRefName']:
        return False
    if current['baseRefOid'] == manifest['base_sha']:
        return True
    result = subprocess.run(
        [
            'gh',
            'api',
            f'repos/{manifest["repo"]}/compare/{current["baseRefOid"]}...{current["headRefOid"]}',
            '--jq',
            '.merge_base_commit.sha',
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return bool(result.stdout.strip() == manifest['merge_base_sha'])


def publish(manifest: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    if not current_target(manifest):
        return {'status': 'incomplete', 'delivery': 'not_posted', 'reason': 'PR source changed during review'}
    fallback_used = False
    result = post(manifest['repo'], str(manifest['pr']), review)
    if result.returncode and 'HTTP 422' in result.stderr:
        moved = '\n\n'.join(f'**{c["path"]}:{c["line"]}**\n\n{c["body"]}' for c in review['comments'])
        appendix = '\n\n### Comments not attached to lines\n\n' + moved if moved else ''
        try:
            fallback = {'event': 'COMMENT', 'body': publication_body(review['body'], appendix)}
        except ValueError as e:
            return {'delivery': 'failed', 'publication_fallback': False, 'reason': str(e)}
        if not current_target(manifest):
            return {'status': 'incomplete', 'delivery': 'not_posted', 'reason': 'PR source changed before retry'}
        fallback_used = True
        result = post(manifest['repo'], str(manifest['pr']), fallback)
    if result.returncode:
        rejected = re.search(r'HTTP 4\d\d', result.stderr) is not None
        return {
            'delivery': 'failed' if rejected else 'unknown',
            'publication_fallback': fallback_used,
            'reason': 'GitHub did not confirm publication; no automatic transport retry',
            'error': result.stderr.strip(),
        }
    outcome: dict[str, Any] = {'delivery': 'posted', 'publication_fallback': fallback_used}
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        outcome['response'] = 'GitHub acknowledged submission but its response could not be decoded'
    else:
        if isinstance(response, dict):
            outcome['review_url'] = response.get('html_url')
    try:
        fresh = current_target(manifest)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        return {
            **outcome,
            'status': 'incomplete',
            'freshness': 'unknown',
            'reason': 'Review was posted; the final source check was unavailable',
        }
    if not fresh:
        return {
            **outcome,
            'status': 'incomplete',
            'delivery': 'posted_stale',
            'reason': 'PR source changed during publication',
        }
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-context', type=Path, default=os.environ.get('REVIEW_CONTEXT'))
    parser.add_argument('--execution-file', type=Path, default=os.environ.get('EXECUTION_FILE') or None)
    parser.add_argument('--result', type=Path, help='Raw or fenced JSON from a local /pxt-review invocation')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.review_context is None:
        parser.error('--review-context is required')
    outcome_path = args.review_context.parent / 'outcome.json'
    try:
        return process_review(args)
    except (ValueError, KeyError, TypeError, AttributeError, OSError, subprocess.CalledProcessError) as e:
        outcome = {'status': 'failed', 'delivery': 'not_posted', 'reason': str(e)}
        outcome_path.write_text(json.dumps(outcome, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(outcome))
        return 1


def process_review(args: argparse.Namespace) -> int:
    outcome_path = args.review_context.parent / 'outcome.json'
    manifest = read_json(args.review_context)
    verify_sources(manifest)
    if args.result is not None:
        message = args.result.read_text(encoding='utf-8')
    elif args.execution_file is not None:
        message = final_message(str(args.execution_file))
    else:
        raise ValueError('Provide --result or --execution-file')
    data = review_json(message)
    if data is None:
        raise ValueError('Final message must contain exactly one review JSON object')
    review, stats = render(data, manifest)
    (outcome_path.parent / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
    if args.execution_file is not None:
        execution = json.loads(args.execution_file.read_text(encoding='utf-8'))
        results = (
            [m for m in execution if isinstance(m, dict) and m.get('type') == 'result']
            if isinstance(execution, list)
            else [execution]
        )
        if results:
            stats['model_usage'] = results[-1].get('modelUsage', {})
            stats['duration_ms'] = results[-1].get('duration_ms')
            stats['permission_denials'] = len(results[-1].get('permission_denials', []))
    (outcome_path.parent / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    outcome = {'status': stats['status'], 'snapshot_id': manifest['snapshot_id'], 'delivery': 'not_requested'}
    if stats['reason']:
        outcome['reason'] = stats['reason']
    if stats['status'] != 'skipped' and not args.validate_only:
        outcome.update(publish(manifest, review))
    outcome_path.write_text(json.dumps(outcome, indent=2) + '\n')
    print(json.dumps(outcome))
    undelivered = outcome.get('delivery') in {'failed', 'unknown', 'not_posted', 'posted_stale'}
    return 1 if undelivered or outcome.get('freshness') == 'unknown' else 0


if __name__ == '__main__':
    sys.exit(main())
