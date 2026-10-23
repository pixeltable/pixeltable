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
                'lines': len(content.splitlines()),
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
    (destination.parent / f'{destination.name}-stripped.json').write_text(json.dumps(stripped, ensure_ascii=True))
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
        ],
        text=True,
    )
    names = subprocess.check_output(
        ['git', '--git-dir', str(git_dir), 'diff', '--name-only', '-z', '--find-renames', merge_base, head]
    )
    files = [repository_path(path.decode('utf-8')) for path in names.split(b'\0') if path]
    sources = {
        'LEFT': export_tree(git_dir, merge_base, output / 'base'),
        'RIGHT': export_tree(git_dir, head, output / 'head'),
    }
    previous = prior_state(repo, number) if history else {}
    previous_tree: dict[str, str] = {}
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
        except subprocess.CalledProcessError:
            previous = {'records': [], 'reason': 'Previous source is unavailable; recheck without suppression'}
        else:
            for item in listing.split(b'\0'):
                if item:
                    details, path = item.split(b'\t', 1)
                    _, kind, object_id = details.split()
                    if kind == b'blob':
                        previous_tree[path.decode('utf-8')] = object_id.decode()
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
    manifest: dict[str, Any] = {
        'schema_version': VERSION,
        'repo': repo,
        'pr': number,
        'pr_url': pr_url,
        'run_id': os.environ.get('GITHUB_RUN_ID', ''),
        'base_sha': base,
        'head_sha': head,
        'merge_base_sha': merge_base,
        'tooling_sha': run('git', 'rev-parse', 'HEAD', cwd=tooling),
        'tooling_root': str(tooling.resolve()),
        'tooling_files': policy_hashes,
        'command_sha256': hashlib.sha256((tooling / '.claude/commands/pxt-review.md').read_bytes()).hexdigest(),
        'roots': {'LEFT': str(output / 'base'), 'RIGHT': str(output / 'head')},
        'sources': sources,
        'files': files,
        'skip_reason': 'The reviewed commits have no changed paths.' if not files else '',
        'diff_files': parse_diff(patch),
        'diff_path': str(output / 'diff.patch'),
        'required_passes': [*REQUIRED_PASSES, *(['focus'] if focus else [])],
        'focus': focus,
        'diff_sha256': hashlib.sha256(patch.encode()).hexdigest(),
        'metadata': metadata,
        'previous': previous,
        'previous_tree': previous_tree,
    }
    manifest['snapshot_id'] = digest(
        {
            k: manifest[k]
            for k in [
                'repo',
                'pr',
                'base_sha',
                'head_sha',
                'merge_base_sha',
                'tooling_sha',
                'command_sha256',
                'tooling_files',
                'focus',
                'required_passes',
            ]
        }
    )
    (output / 'diff.patch').write_text(patch, encoding='utf-8')
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
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
