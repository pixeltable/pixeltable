"""Prepare an exact PR snapshot without running its code or loading its agent instructions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from claude_review_contract import (
    AGENT_FILES,
    MAX_JSON_BYTES,
    MAX_SOURCES_BYTES,
    REQUIRED_PASSES,
    STATE_MARKER,
    VERSION,
    digest,
    parse_diff,
    repository_path,
)


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def gh(*args: str) -> Any:
    return json.loads(run('gh', *args))


def changed_paths(git_dir: Path, base: str, head: str) -> dict[str, Any]:
    raw = subprocess.check_output(
        ['git', '--git-dir', str(git_dir), 'diff', '--name-status', '-z', '--find-renames', base, head]
    )
    changes: dict[str, Any] = {}
    fields = iter(raw.split(b'\0')[:-1])
    for raw_status in fields:
        status = raw_status.decode('ascii')
        old = repository_path(next(fields).decode('utf-8'))
        path = repository_path(next(fields).decode('utf-8')) if status[0] in {'R', 'C'} else old
        changes[path] = {'old_path': None if status == 'A' else old, 'status': status}
    return changes


def export_tree(git_dir: Path, sha: str, destination: Path) -> dict[str, Any]:
    listing = subprocess.check_output(['git', '--git-dir', str(git_dir), 'ls-tree', '-rz', sha])
    destination.mkdir()
    index: dict[str, Any] = {}
    stripped: dict[str, str] = {}
    process = subprocess.Popen(
        ['git', '--git-dir', str(git_dir), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for item in listing.split(b'\0'):
            if not item:
                continue
            metadata, raw_path = item.split(b'\t', 1)
            mode, kind, object_id = metadata.split()
            if kind != b'blob':
                continue
            path = repository_path(raw_path.decode('utf-8'))
            process.stdin.write(object_id + b'\n')
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or header[1] != b'blob':
                raise ValueError('Git did not return the requested source blob')
            content = process.stdout.read(int(header[2]))
            if process.stdout.read(1) != b'\n':
                raise ValueError('Git returned an incomplete source blob')
            entry: dict[str, Any] = {
                'sha256': hashlib.sha256(content).hexdigest(),
                'lines': content.count(b'\n') + int(bool(content) and not content.endswith(b'\n')),
                'mode': mode.decode(),
                'git_blob': object_id.decode(),
            }
            if any(part in AGENT_FILES for part in Path(path).parts):
                try:
                    stripped[path] = content.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                entry['stripped'] = True
            else:
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            index[path] = entry
    finally:
        process.stdin.close()
        process.stdout.close()
        if process.wait() != 0:
            raise ValueError('Git source export failed')
    encoded = json.dumps(stripped, ensure_ascii=True)
    if len(encoded.encode()) > MAX_JSON_BYTES:
        raise ValueError('Stripped source JSON exceeds the review size limit')
    (destination.parent / f'{destination.name}-stripped.json').write_text(encoded)
    return index


def prior_state(repo: str, pr: int) -> dict[str, Any]:
    try:
        reviews = gh('api', '--paginate', '--slurp', f'repos/{repo}/pulls/{pr}/reviews?per_page=100')
    except subprocess.CalledProcessError:
        return {'records': [], 'reason': 'Prior reviews unavailable; recheck without suppression'}
    for review in reversed([item for page in reviews for item in page]):
        if review.get('user', {}).get('login') != 'github-actions[bot]':
            continue
        body = review.get('body') or ''
        if STATE_MARKER not in body:
            if body.startswith('## Claude review'):
                return {'records': [], 'reason': 'Latest review has no compatible history'}
            continue
        try:
            state = state_from_review(body, repo, pr, review.get('commit_id'))
        except (ValueError, TypeError, KeyError, AttributeError, subprocess.CalledProcessError):
            state = {}
        return state or {'records': [], 'reason': 'Latest review history is unavailable or incompatible'}
    return {}


def valid_records(records: Any) -> bool:
    if not isinstance(records, list):
        return False
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get('claim'), dict):
            return False
        claim = record['claim']
        if any(
            not isinstance(claim.get(key), str) or not claim[key].strip()
            for key in ['id', 'producer', 'subject', 'mechanism', 'title', 'comment', 'trigger', 'impact', 'correction']
        ):
            return False
        if 'ever_posted' in record and type(record['ever_posted']) is not bool:
            return False
        if claim['id'] in seen or type(record.get('posted')) is not bool:
            return False
        seen.add(claim['id'])
        if record.get('status') not in {'keep', 'narrow', 'question', 'reject', 'resolved', 'unverified'}:
            return False
        if not isinstance(record.get('claim_digest'), str):
            return False
    return True


def state_from_review(body: str, repo: str, pr: int, reviewed_head: str | None) -> dict[str, Any]:
    raw = body.rsplit(STATE_MARKER, 1)[1].split('\n-->', 1)[0]
    state = json.loads(raw)
    if not isinstance(state, dict):
        return {}
    if (state.get('schema_version'), state.get('repo'), state.get('pr')) != (VERSION, repo, pr):
        return {}
    if state.get('head_sha') != reviewed_head or not re.fullmatch(r'[0-9a-f]{40}', state.get('head_sha', '')):
        return {}
    run_id = str(state.get('run_id', ''))
    if not run_id.isdigit():
        return {}
    workflow_run = gh('api', f'repos/{repo}/actions/runs/{run_id}')
    if (
        workflow_run.get('path') != '.github/workflows/claude-review.yml'
        or workflow_run.get('event') != 'issue_comment'
        or workflow_run.get('repository', {}).get('full_name') != repo
    ):
        return {}
    if not valid_records(state.get('records')):
        return {}
    state['records'] = [r for r in state['records'] if r['status'] not in {'resolved', 'reject'}]
    return state


def prepare(pr_url: str, output: Path, tooling: Path, focus: str = '', history: bool = True) -> Path:
    match = re.fullmatch(r'https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/(\d+)/?', pr_url)
    if match is None:
        raise ValueError('Use a full GitHub PR URL')
    repo, number = match[1], int(match[2])
    metadata = gh('pr', 'view', pr_url, '--json', 'title,body,state,baseRefOid,headRefOid,baseRefName')
    if metadata['state'] != 'OPEN' and not os.environ.get('PXT_REVIEW_ALLOW_CLOSED'):
        raise ValueError('PR is closed; set PXT_REVIEW_ALLOW_CLOSED=1 only for historical evaluation')
    base, head = metadata['baseRefOid'], metadata['headRefOid']
    for sha in [base, head]:
        if not re.fullmatch(r'[0-9a-f]{40}', sha):
            raise ValueError('GitHub did not return full commit IDs')
    output = output.resolve()
    if output.exists():
        raise ValueError('Review output directory must be new')
    output.mkdir(parents=True)
    git_dir = output / 'source.git'
    run('git', 'init', '--bare', '--quiet', str(git_dir))
    run('git', '--git-dir', str(git_dir), 'remote', 'add', 'origin', f'https://github.com/{repo}.git')
    merge_base = run('gh', 'api', f'repos/{repo}/compare/{base}...{head}', '--jq', '.merge_base_commit.sha')
    if not re.fullmatch(r'[0-9a-f]{40}', merge_base):
        raise ValueError('GitHub did not return a merge-base commit')
    run(
        'git',
        '-c',
        'credential.helper=',
        '-c',
        'credential.helper=!gh auth git-credential',
        '--git-dir',
        str(git_dir),
        'fetch',
        '--quiet',
        '--depth=1',
        '--no-tags',
        'origin',
        base,
        head,
        merge_base,
    )
    patch = subprocess.check_output(
        [
            'git',
            '--git-dir',
            str(git_dir),
            '-c',
            'core.quotePath=true',
            'diff',
            '--no-ext-diff',
            '--no-textconv',
            '--find-renames',
            '--unified=3',
            merge_base,
            head,
        ]
    ).decode('utf-8')
    changes = parse_diff(patch)
    metadata_changes = changed_paths(git_dir, merge_base, head)
    files = list(metadata_changes)
    for path, metadata_change in metadata_changes.items():
        change = changes.setdefault(path, {'hunks': [], 'changed': {'LEFT': [], 'RIGHT': []}})
        change.update(metadata_change)
    sources = {
        'LEFT': export_tree(git_dir, merge_base, output / 'base'),
        'RIGHT': export_tree(git_dir, head, output / 'head'),
    }
    previous = prior_state(repo, number) if history else {}
    previous_tree: dict[str, str] = {}
    previous_renames: dict[str, str] = {}
    if previous.get('records'):
        try:
            run(
                'git',
                '-c',
                'credential.helper=',
                '-c',
                'credential.helper=!gh auth git-credential',
                '--git-dir',
                str(git_dir),
                'fetch',
                '--quiet',
                '--depth=1',
                '--filter=blob:none',
                'origin',
                previous['head_sha'],
            )
            listing = subprocess.check_output(
                ['git', '--git-dir', str(git_dir), 'ls-tree', '-rz', previous['head_sha']]
            )
            previous_renames = {
                change['old_path']: path
                for path, change in changed_paths(git_dir, previous['head_sha'], head).items()
                if change['status'].startswith('R')
            }
        except subprocess.CalledProcessError:
            previous = {'records': [], 'reason': 'Previous source is unavailable; recheck without suppression'}
        else:
            for item in listing.split(b'\0'):
                if item:
                    details, raw_path = item.split(b'\t', 1)
                    _, kind, object_id = details.split()
                    if kind == b'blob':
                        previous_tree[raw_path.decode('utf-8')] = object_id.decode()
    policy_paths = [tooling / name for name in ['AGENTS.md', 'CLAUDE.md']]
    policy_paths += list((tooling / '.claude').rglob('*.md'))
    policy_paths += list((tooling / '.github/instructions').glob('*.md'))
    policy_paths += [
        tooling / '.github/scripts' / name
        for name in ['prepare_claude_review.py', 'claude_review_contract.py', 'post_claude_review.py']
    ]
    policy_hashes = {
        path.relative_to(tooling).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in policy_paths
        if path.is_file()
    }
    tooling_sha = run('git', 'rev-parse', 'HEAD', cwd=tooling)
    snapshot_id = digest(
        {
            'repo': repo,
            'pr': number,
            'base_sha': base,
            'head_sha': head,
            'merge_base_sha': merge_base,
            'tooling_sha': tooling_sha,
            'tooling_files': policy_hashes,
            'focus': focus,
            'required_passes': [*REQUIRED_PASSES, *(['focus'] if focus else [])],
        }
    )
    sources_path = output / 'sources.json'
    sources_encoded = (
        json.dumps(
            {'diff_files': changes, 'previous_tree': previous_tree, 'sources': sources, 'tooling_files': policy_hashes},
            sort_keys=True,
            ensure_ascii=True,
        )
        + '\n'
    )
    if len(sources_encoded.encode()) > MAX_SOURCES_BYTES:
        raise ValueError('Prepared source index exceeds the review size limit')
    manifest: dict[str, Any] = {
        'schema_version': VERSION,
        'snapshot_id': snapshot_id,
        'repo': repo,
        'pr': number,
        'pr_url': pr_url,
        'run_id': os.environ.get('GITHUB_RUN_ID', ''),
        'base_sha': base,
        'head_sha': head,
        'merge_base_sha': merge_base,
        'tooling_sha': tooling_sha,
        'tooling_root': str(tooling.resolve()),
        'focus': focus,
        'required_passes': [*REQUIRED_PASSES, *(['focus'] if focus else [])],
        'skip_reason': 'The reviewed commits have no changed paths.' if not files else '',
        'roots': {'LEFT': str(output / 'base'), 'RIGHT': str(output / 'head')},
        'diff_path': str(output / 'diff.patch'),
        'diff_sha256': hashlib.sha256(patch.encode()).hexdigest(),
        'sources_path': str(sources_path),
        'sources_sha256': hashlib.sha256(sources_encoded.encode()).hexdigest(),
        'files': files,
        'metadata': metadata,
        'previous': previous,
        'previous_renames': previous_renames,
    }
    (output / 'diff.patch').write_text(patch, encoding='utf-8')
    sources_path.write_text(sources_encoded, encoding='utf-8')
    manifest_path = output / 'manifest.json'
    encoded = json.dumps(manifest, indent=2, ensure_ascii=True) + '\n'
    if len(encoded.encode()) > MAX_JSON_BYTES:
        raise ValueError('Prepared manifest exceeds the review size limit')
    manifest_path.write_text(encoded)
    (output / 'outcome.json').write_text(
        json.dumps({'status': 'prepared', 'snapshot_id': manifest['snapshot_id']}) + '\n'
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pr_url')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--focus', default='')
    parser.add_argument('--no-history', action='store_true')
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        print(prepare(args.pr_url, args.output, Path.cwd(), args.focus, not args.no_history))
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        if not existed and args.output.is_dir():
            (args.output / 'outcome.json').write_text(json.dumps({'status': 'failed', 'reason': str(e)}) + '\n')
        raise SystemExit(f'Review preparation failed: {e}') from e


if __name__ == '__main__':
    main()
