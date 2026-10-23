"""Validate source-bound review records and render only independently checked claims."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

VERSION = 1
REQUIRED_PASSES = ['abstractions', 'lines', 'design', 'blast-radius', 'rules']
KINDS = {'defect', 'design', 'rule', 'maintenance'}
AGENT_FILES = {'CLAUDE.md', 'CLAUDE.local.md', 'AGENTS.md', '.claude', '.mcp.json'}
STATE_MARKER = '<!-- pxt-review-state\n'
MAX_JSON_BYTES = 2_000_000


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('JSON exceeds the review size limit')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a nonempty string')
    return value


def repository_path(value: Any) -> str:
    path = text(value, 'path')
    if '\\' in path or path.startswith('/') or any(p in {'', '.', '..'} for p in path.split('/')):
        raise ValueError('Expected a repository-relative path')
    return path


def positive_int(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError('Expected a positive line number')
    return value


def diff_path(value: str) -> str:
    if value.startswith('"'):
        value = ast.literal_eval('b' + value).decode('utf-8')
    return value[2:]


def parse_diff(patch: str) -> dict[str, Any]:
    files: dict[str, Any] = {}
    old_path: str | None = None
    current: dict[str, Any] | None = None
    left = right = 0
    hunk: dict[str, list[int]] | None = None
    for line in patch.splitlines():
        if line.startswith('diff --git '):
            current = hunk = None
            old_path = None
        elif hunk is None and line.startswith('--- '):
            raw = line[4:].split('\t')[0]
            old_path = None if raw == '/dev/null' else diff_path(raw)
        elif hunk is None and line.startswith('+++ '):
            raw = line[4:].split('\t')[0]
            new_path = None if raw == '/dev/null' else diff_path(raw)
            path = new_path or old_path
            if path is None:
                raise ValueError('Diff has no path')
            repository_path(path)
            current = {'old_path': old_path, 'hunks': [], 'changed': {'LEFT': [], 'RIGHT': []}}
            files[path] = current
        elif line.startswith('@@ '):
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if match is None or current is None:
                raise ValueError('Malformed diff hunk')
            left, right = int(match[1]), int(match[3])
            hunk = {'LEFT': [], 'RIGHT': []}
            current['hunks'].append(hunk)
        elif hunk is not None and current is not None:
            if line.startswith(' '):
                hunk['LEFT'].append(left)
                hunk['RIGHT'].append(right)
                left += 1
                right += 1
            elif line.startswith('-'):
                hunk['LEFT'].append(left)
                current['changed']['LEFT'].append(left)
                left += 1
            elif line.startswith('+'):
                hunk['RIGHT'].append(right)
                current['changed']['RIGHT'].append(right)
                right += 1
    return files


def inline_valid(location: dict[str, Any], files: dict[str, Any]) -> bool:
    try:
        path = repository_path(location.get('path'))
        side = location.get('side', 'RIGHT')
        end = positive_int(location.get('line'))
        start = positive_int(location.get('start_line', end))
        if start > end or side not in {'LEFT', 'RIGHT'} or path not in files:
            return False
        return any(start in h[side] and end in h[side] for h in files[path]['hunks'])
    except (ValueError, TypeError):
        return False


def reference(ref: Any, manifest: dict[str, Any]) -> None:
    if not isinstance(ref, dict):
        raise ValueError('Source reference must be an object')
    path = repository_path(ref.get('path'))
    side = ref.get('side', 'RIGHT')
    if side not in {'LEFT', 'RIGHT'}:
        raise ValueError('Invalid source side')
    end = positive_int(ref.get('line'))
    start = positive_int(ref.get('start_line', end))
    if start > end:
        raise ValueError('Source range is reversed')
    index = manifest['sources'][side]
    source_path = path
    if side == 'LEFT' and path in manifest['diff_files']:
        source_path = manifest['diff_files'][path]['old_path'] or path
    entry = index.get(source_path)
    if entry is None or end > entry['lines']:
        raise ValueError(f'Source reference does not exist: {path}:{end}')


def verify_sources(manifest: dict[str, Any]) -> None:
    for path, expected_hash in manifest.get('tooling_files', {}).items():
        content = (Path(manifest['tooling_root']) / repository_path(path)).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError('Trusted review instructions or tooling changed')
    if hashlib.sha256(Path(manifest['diff_path']).read_bytes()).hexdigest() != manifest['diff_sha256']:
        raise ValueError('Prepared diff changed')
    for side, entries in manifest['sources'].items():
        root = Path(manifest['roots'][side]).resolve()
        expected = {path for path, entry in entries.items() if not entry.get('stripped')}
        observed = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
        if expected != observed:
            raise ValueError('Prepared source file set changed')
        stripped = read_json(root.parent / f'{root.name}-stripped.json')
        for path, entry in entries.items():
            if entry.get('stripped'):
                if hashlib.sha256(stripped[path].encode()).hexdigest() != entry['sha256']:
                    raise ValueError('Stripped source changed')
                continue
            source = root / repository_path(path)
            if not source.resolve().is_relative_to(root) or source.is_symlink():
                raise ValueError('Source tree contains a redirected path')
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Prepared source changed: {path}')


def validate_claim(claim: Any, manifest: dict[str, Any]) -> None:
    if not isinstance(claim, dict):
        raise ValueError('Finding must be an object')
    for field in ['id', 'producer', 'subject', 'mechanism', 'title', 'comment', 'trigger', 'impact', 'correction']:
        text(claim.get(field), field)
    if claim.get('kind') not in KINDS:
        raise ValueError('Invalid finding kind')
    if claim.get('attribution') == 'existing' and claim['kind'] != 'maintenance':
        raise ValueError('Existing debt is only a maintenance suggestion')
    if claim.get('attribution') not in {'new', 'newly_exposed', 'existing'}:
        raise ValueError('Invalid attribution')
    if claim['kind'] == 'defect':
        if claim.get('severity') not in {'High', 'Medium', 'Low'}:
            raise ValueError('A supported defect needs an impact severity')
    elif claim.get('severity') is not None:
        raise ValueError('Only defects have severity')
    if claim['kind'] == 'rule':
        text(claim.get('rule'), 'rule')
    evidence = claim.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Finding needs a source trace')
    for ref in evidence:
        reference(ref, manifest)
    location = claim.get('location')
    if location is not None and not isinstance(location, dict):
        raise ValueError('Location must be an object or null')


def same_identity(claim: dict[str, Any], earlier: dict[str, Any], manifest: dict[str, Any]) -> bool:
    if claim['mechanism'] != earlier['mechanism']:
        return False
    if claim['subject'] == earlier['subject']:
        return True
    for path, change in manifest['diff_files'].items():
        old = change['old_path']
        if (
            old
            and old != path
            and earlier['subject'].startswith(old + '#')
            and claim['subject'] == path + earlier['subject'][len(old) :]
        ):
            return True
    return False


def validate(data: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    if data.get('schema_version') != VERSION or data.get('snapshot_id') != manifest['snapshot_id']:
        raise ValueError('Review does not match the prepared snapshot')
    status = data.get('status')
    if status not in {'complete', 'incomplete', 'skipped'}:
        raise ValueError('Review needs an explicit outcome')
    text(data.get('assessment'), 'assessment')
    if manifest.get('focus'):
        text(data.get('focus_answer'), 'focus_answer')
    if status in {'incomplete', 'skipped'}:
        text(data.get('reason'), 'reason')
    if status == 'skipped' and (not manifest.get('skip_reason') or data['reason'] != manifest['skip_reason']):
        raise ValueError('Skipping requires the host-approved whole-change policy')
    for field in ['passes', 'findings', 'decisions', 'questions']:
        if not isinstance(data.get(field), list):
            raise ValueError(f'{field} must be an array')
    passes: dict[str, Any] = {}
    gaps: list[str] = []
    for result in data['passes']:
        if not isinstance(result, dict):
            raise ValueError('Pass result must be an object')
        name = text(result.get('pass'), 'pass')
        if name in passes or name not in [*REQUIRED_PASSES, 'walk', 'focus']:
            raise ValueError('Duplicate or unknown pass')
        passes[name] = result
        if result.get('status') not in {'complete', 'incomplete'}:
            raise ValueError('Pass needs an explicit outcome')
        if result['status'] == 'incomplete':
            gaps.append(f'{name}: {text(result.get("reason"), "reason")}')
        if not isinstance(result.get('checks'), list) or not isinstance(result.get('exclusions'), list):
            raise ValueError('Pass needs checks and exclusions')
        for check in result['checks']:
            reference(check, manifest)
            text(check.get('check'), 'check')
        exclusions: set[str] = set()
        for exclusion in result['exclusions']:
            path = repository_path(exclusion.get('path'))
            if path not in manifest['files'] or path in exclusions:
                raise ValueError('Invalid or duplicate exclusion')
            text(exclusion.get('reason'), 'exclusion reason')
            exclusions.add(path)
        if result['status'] == 'complete' and not result['checks'] and exclusions != set(manifest['files']):
            gaps.append(f'{name}: no source check or complete exclusion')
        if name in {'lines', 'rules'}:
            for path in manifest['files']:
                if path in exclusions:
                    continue
                diff = manifest['diff_files'].get(path)
                if diff is None:
                    gaps.append(f'{name}: {path} has no review or exclusion')
                    continue
                for side, changed in diff['changed'].items():
                    checks = [c for c in result['checks'] if c['path'] == path and c.get('side', 'RIGHT') == side]
                    if any(not any(c.get('start_line', c['line']) <= n <= c['line'] for c in checks) for n in changed):
                        gaps.append(f'{name}: {path} has uncovered {side} changes')
    required = [*manifest['required_passes']]
    if data.get('user_surface') is True:
        required.append('walk')
    elif type(data.get('user_surface')) is not bool:
        raise ValueError('user_surface must be explicit')
    gaps.extend(f'{name}: no result' for name in required if name not in passes)
    findings: dict[str, Any] = {}
    for claim in data['findings']:
        validate_claim(claim, manifest)
        key = claim['id']
        if key in findings or claim['producer'] not in passes:
            raise ValueError('Finding has a duplicate ID or an unknown producer')
        findings[key] = claim
    decisions: dict[str, Any] = {}
    for decision in data['decisions']:
        key = text(decision.get('finding_id'), 'finding_id')
        if key in decisions or key not in findings:
            raise ValueError('Decision has a duplicate or unknown finding ID')
        if decision.get('checker') != 'cross-check' or decision['checker'] == findings[key]['producer']:
            raise ValueError('A finding needs an independent cross-checker')
        if decision.get('decision') not in {'keep', 'narrow', 'question', 'reject', 'resolved'}:
            raise ValueError('Unknown verification decision')
        text(decision.get('reason'), 'verification reason')
        checked = decision.get('checked')
        validate_claim(checked, manifest)
        if any(checked[f] != findings[key][f] for f in ['id', 'producer', 'subject', 'mechanism']):
            raise ValueError('Verification changed the finding identity')
        if decision['decision'] == 'question':
            text(decision.get('unknown'), 'unknown')
            text(decision.get('check'), 'check')
            gaps.append(f'{key}: decisive fact remains unresolved')
        if decision['decision'] == 'resolved':
            fixes = decision.get('fix_evidence', [])
            removed = decision.get('removed_paths', [])
            if not isinstance(fixes, list) or not isinstance(removed, list) or not (fixes or removed):
                raise ValueError('Resolution needs source evidence of the fix')
            changed = False
            for ref in fixes:
                reference(ref, manifest)
                if ref.get('side', 'RIGHT') != 'RIGHT':
                    raise ValueError('Resolution must cite the current fix')
                current_blob = manifest['sources']['RIGHT'][ref['path']].get('git_blob')
                if current_blob and current_blob != manifest.get('previous_tree', {}).get(ref['path']):
                    changed = True
            for path in removed:
                repository_path(path)
                if path not in manifest.get('previous_tree', {}) or path in manifest['sources']['RIGHT']:
                    raise ValueError('Resolution did not remove the cited source path')
                changed = True
            if not manifest.get('previous_tree') or not changed:
                raise ValueError('Resolution has no demonstrated source change since the prior review')
        decisions[key] = decision
    for key, claim in findings.items():
        if claim['kind'] != 'maintenance' and key not in decisions:
            gaps.append(f'{key}: not independently checked')
    for number, question in enumerate(data['questions'], 1):
        text(question.get('unknown'), 'unknown')
        text(question.get('check'), 'check')
        gaps.append(f'question {number}: decisive fact remains unresolved')
    previous = {r['claim']['id']: r for r in manifest.get('previous', {}).get('records', [])}
    for key, claim in findings.items():
        if key in previous and not same_identity(claim, previous[key]['claim'], manifest):
            raise ValueError('Prior finding ID was reused for a different identity')
    previous_ids = set(previous)
    gaps.extend(f'{key}: prior claim not revalidated' for key in sorted(previous_ids - decisions.keys()))
    if status == 'skipped' and (findings or data['decisions']):
        raise ValueError('A skipped review cannot contain findings')
    if gaps and status == 'complete':
        raise ValueError('Review claims complete coverage: ' + '; '.join(gaps))
    return {'findings': findings, 'decisions': decisions, 'gaps': gaps}


def source_link(ref: dict[str, Any], manifest: dict[str, Any]) -> str:
    side = ref.get('side', 'RIGHT')
    sha = manifest['head_sha'] if side == 'RIGHT' else manifest['merge_base_sha']
    path = ref['path']
    if side == 'LEFT' and path in manifest['diff_files']:
        path = manifest['diff_files'][path]['old_path'] or path
    from urllib.parse import quote

    start = ref.get('start_line', ref['line'])
    return f'https://github.com/{manifest["repo"]}/blob/{sha}/{quote(path)}#L{start}-L{ref["line"]}'


def claim_digest(claim: dict[str, Any], manifest: dict[str, Any]) -> str:
    public_claim = {key: value for key, value in claim.items() if key not in {'id', 'producer'}}
    evidence = []
    for ref in claim['evidence']:
        side = ref.get('side', 'RIGHT')
        path = ref['path']
        if side == 'LEFT' and path in manifest['diff_files']:
            path = manifest['diff_files'][path]['old_path'] or path
        entry = manifest['sources'][side][path]
        evidence.append({'path': path, 'sha256': entry['sha256']})
    return digest([public_claim, evidence])


def render(data: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    checked = validate(data, manifest)
    body = [
        '## Claude review',
        '',
        f'Reviewed {manifest["merge_base_sha"]}..{manifest["head_sha"]}.',
        '',
        data['assessment'],
    ]
    if manifest.get('focus'):
        body += ['', '### Focus', '', data['focus_answer']]
    if data['status'] == 'incomplete':
        body += ['', 'Coverage incomplete: ' + data['reason']]
        body += ['- ' + gap for gap in checked['gaps']]
    comments: list[dict[str, Any]] = []
    questions = list(data['questions'])
    previous = {r['claim']['id']: r for r in manifest.get('previous', {}).get('records', [])}
    records: dict[str, Any] = {key: {**value, 'status': 'unverified'} for key, value in previous.items()}
    published = 0
    for key, original in checked['findings'].items():
        decision = checked['decisions'].get(key)
        if decision is None and original['kind'] != 'maintenance':
            continue
        claim = original if decision is None else decision['checked']
        verdict = 'keep' if decision is None else decision['decision']
        fingerprint = claim_digest(claim, manifest)
        earlier = previous.get(key, {})
        record = {
            'claim': claim,
            'status': verdict,
            'reason': '' if decision is None else decision['reason'],
            'claim_digest': fingerprint,
            'posted': False,
            'ever_posted': bool(earlier.get('ever_posted') or earlier.get('posted')),
        }
        records[key] = record
        if verdict == 'question':
            questions.append({'unknown': decision['unknown'], 'check': decision['check']})
            continue
        if verdict == 'resolved':
            if not record['ever_posted']:
                raise ValueError('Only a previously posted claim can be resolved')
            body += ['', f'Resolved: {claim["title"]}. {decision["reason"]}']
            continue
        if verdict == 'reject':
            continue
        if earlier.get('posted') and earlier.get('claim_digest') == fingerprint:
            record['posted'] = record['ever_posted'] = True
            continue
        record['posted'] = record['ever_posted'] = True
        label = claim.get('severity') if claim['kind'] == 'defect' else claim['kind'].title()
        comment_body = f'**{label}: {claim["title"]}**\n\n{claim["comment"]}'
        ref = claim['evidence'][0]
        body += ['', f'- {claim["title"]}: [source]({source_link(ref, manifest)}).']
        location = claim.get('location')
        if claim['kind'] != 'maintenance' and location is not None and inline_valid(location, manifest['diff_files']):
            comment = {k: location[k] for k in ['path', 'line']}
            comment |= {'side': location.get('side', 'RIGHT'), 'body': comment_body}
            if location.get('start_line', location['line']) < location['line']:
                comment |= {'start_line': location['start_line'], 'start_side': comment['side']}
            comments.append(comment)
        else:
            body += ['', comment_body]
        published += 1
    if not published and data['status'] == 'complete':
        body += ['', 'No new supported findings.' if previous else 'No supported findings.']
    if questions:
        body += ['', '### Questions', '']
        body += [f'- {q["unknown"]} Check: {q["check"]}' for q in questions]
    state = {
        'schema_version': VERSION,
        'repo': manifest['repo'],
        'pr': manifest['pr'],
        'run_id': manifest.get('run_id', ''),
        'head_sha': manifest['head_sha'],
        'records': list(records.values()),
    }
    encoded = json.dumps(state, sort_keys=True, ensure_ascii=True)
    if len('\n'.join(body).encode()) > 60_000:
        raise ValueError('Review body exceeds the publication limit')
    if len(encoded.encode()) + len('\n'.join(body).encode()) <= 60_000:
        body += ['', STATE_MARKER + encoded + '\n-->']
    else:
        state = {**state, 'records': [], 'reason': 'history exceeds size limit; recheck without suppression'}
        body += ['', STATE_MARKER + json.dumps(state, ensure_ascii=True) + '\n-->']
    pass_stats: dict[str, Any] = {}
    for result in data['passes']:
        name = result['pass']
        owned = [key for key, claim in checked['findings'].items() if claim['producer'] == name]
        verdicts = [checked['decisions'].get(key, {}).get('decision', 'unverified') for key in owned]
        pass_stats[name] = {
            'status': result['status'],
            'raised': len(owned),
            'kept': sum(v in {'keep', 'narrow'} for v in verdicts),
            'rejected': verdicts.count('reject'),
            'questions': verdicts.count('question'),
            'resolved': verdicts.count('resolved'),
            'unverified': verdicts.count('unverified'),
        }
    stats = {
        'status': data['status'],
        'passes': pass_stats,
        'raised': len(checked['findings']),
        'decisions': {key: value['decision'] for key, value in checked['decisions'].items()},
        'published': published,
        'gaps': checked['gaps'],
        'per_pass_usage': None,
    }
    return {'commit_id': manifest['head_sha'], 'body': '\n'.join(body), 'event': 'COMMENT', 'comments': comments}, stats
