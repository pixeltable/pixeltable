"""Validate declared review decisions and render checked records or maintenance suggestions."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

VERSION = 1
REQUIRED_PASSES = ['abstractions', 'lines', 'design', 'blast-radius', 'rules']
KINDS = {'defect', 'design', 'rule', 'maintenance'}
AGENT_FILES = {'CLAUDE.md', 'CLAUDE.local.md', 'AGENTS.md', '.claude', '.mcp.json'}
STATE_MARKER = '<!-- pxt-review-state\n'
MAX_JSON_BYTES = 2_000_000
MAX_SOURCES_BYTES = 20_000_000
MAX_BODY_BYTES = 60_000
SOURCE_INDEX_KEYS = {'diff_files', 'previous_tree', 'sources', 'tooling_files'}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('JSON exceeds the review size limit')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    if 'sources_path' in value and 'sources' not in value:
        sources_path = Path(text(value.get('sources_path'), 'sources_path'))
        if not sources_path.is_absolute():
            sources_path = path.parent / sources_path
        elif not sources_path.exists() and (path.parent / sources_path.name).exists():
            sources_path = path.parent / sources_path.name
        if sources_path.stat().st_size > MAX_SOURCES_BYTES:
            raise ValueError('Prepared source index exceeds the review size limit')
        raw = sources_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != value.get('sources_sha256'):
            raise ValueError('Prepared source index changed')
        extra = json.loads(raw.decode('utf-8'))
        if not isinstance(extra, dict) or not set(extra).issubset(SOURCE_INDEX_KEYS):
            raise ValueError('Expected a JSON object')
        value['sources_path'] = str(sources_path)
        value.update(extra)
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
    for line in patch.split('\n'):
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


def review_anchor(location: Any, files: dict[str, Any]) -> dict[str, Any] | None:
    """A GitHub review comment on a changed file, including a line outside the hunks."""
    if not isinstance(location, dict):
        return None
    try:
        path = repository_path(location.get('path'))
        side = location.get('side', 'RIGHT')
        end = positive_int(location.get('line'))
        start = positive_int(location.get('start_line', end))
    except (ValueError, TypeError):
        return None
    if start > end or side not in {'LEFT', 'RIGHT'} or path not in files:
        return None
    anchor: dict[str, Any] = {'path': path, 'line': end, 'side': side}
    if inline_valid(location, files) and start < end:
        anchor['start_line'] = start
        anchor['start_side'] = side
    return anchor


def claim_anchor(claim: dict[str, Any], files: dict[str, Any]) -> dict[str, Any] | None:
    """Prefer a diff hunk. The cited location wins, then the first in-hunk evidence line."""
    location = claim.get('location')
    if isinstance(location, dict) and inline_valid(location, files):
        return review_anchor(location, files)
    for ref in claim.get('evidence') or []:
        if isinstance(ref, dict) and inline_valid(ref, files):
            return review_anchor(ref, files)
    return review_anchor(location, files)


_SUGGESTION_FENCE = re.compile(r'```suggestion\n.*?```', re.DOTALL)


def same_comment_range(anchor: dict[str, Any], location: dict[str, Any]) -> bool:
    if anchor.get('line') != location.get('line') or anchor.get('side', 'RIGHT') != location.get('side', 'RIGHT'):
        return False
    return anchor.get('start_line', anchor['line']) == location.get('start_line', location.get('line'))


def suggestion_fence(claim: dict[str, Any], location: dict[str, Any]) -> str:
    """A GitHub suggestion replaces exactly the commented right-side lines."""
    if not isinstance(location, dict) or location.get('side', 'RIGHT') != 'RIGHT':
        return ''
    try:
        end = positive_int(location.get('line'))
        start = positive_int(location.get('start_line', end))
    except (ValueError, TypeError):
        return ''
    if end - start + 1 > 5:
        return ''
    replacement = claim.get('suggestion')
    if not isinstance(replacement, str) or not replacement.strip() or '```' in replacement:
        return ''
    if len(replacement.splitlines()) > 5:
        return ''
    if not replacement.endswith('\n'):
        replacement += '\n'
    return f'\n\n```suggestion\n{replacement}```'


def without_suggestion(text: str) -> str:
    return _SUGGESTION_FENCE.sub('', text).strip()


def body_finding(label: str, title: str, link: str, comment: str) -> list[str]:
    lines = ['', f'- **{label}: {title}** ({link})', '']
    lines.extend(f'  {line}' if line else '' for line in comment.splitlines())
    return lines


def reference(ref: Any, manifest: dict[str, Any], historical: dict[str, Any] | None = None) -> None:
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
        # A non-published historical decision can retain the trusted prior review's exact reference.
        if historical is not None and ref in historical.get('evidence', []):
            return
        raise ValueError(f'Source reference does not exist: {path}:{end}')


def verify_sources(manifest: dict[str, Any]) -> None:
    if 'sources_path' in manifest:
        sources_path = Path(text(manifest.get('sources_path'), 'sources_path'))
        if sources_path.stat().st_size > MAX_SOURCES_BYTES:
            raise ValueError('Prepared source index exceeds the review size limit')
        if hashlib.sha256(sources_path.read_bytes()).hexdigest() != manifest.get('sources_sha256'):
            raise ValueError('Prepared source index changed')
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


def validate_claim(claim: Any, manifest: dict[str, Any], historical: dict[str, Any] | None = None) -> None:
    if not isinstance(claim, dict):
        raise ValueError('Finding must be an object')
    for field in ['id', 'producer', 'subject', 'mechanism', 'title', 'comment', 'trigger', 'impact', 'correction']:
        text(claim.get(field), field)
    if claim.get('kind') not in KINDS:
        raise ValueError('Invalid finding kind')
    if claim.get('attribution') == 'existing' and claim['kind'] != 'maintenance':
        claim['kind'] = 'maintenance'
        claim['severity'] = None
    if claim.get('attribution') not in {'new', 'newly_exposed', 'existing'}:
        raise ValueError('Invalid attribution')
    if claim['kind'] == 'defect':
        if claim.get('severity') not in {'High', 'Medium', 'Low'}:
            raise ValueError('A supported defect needs an impact severity')
    elif claim.get('severity') not in {None, ''}:
        raise ValueError('Only defects have severity')
    else:
        claim['severity'] = None
    if claim['kind'] == 'rule':
        text(claim.get('rule'), 'rule')
    evidence = claim.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Finding needs a source trace')
    for ref in evidence:
        reference(ref, manifest, historical)
    location = claim.get('location')
    if location is not None and not isinstance(location, dict):
        raise ValueError('Location must be an object or null')


def split_subject(subject: str) -> tuple[str, str]:
    parts = re.split(r'[#:]', subject, maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ''


def same_identity(claim: dict[str, Any], earlier: dict[str, Any], manifest: dict[str, Any]) -> bool:
    if claim['mechanism'] != earlier['mechanism']:
        return False
    claim_path, claim_symbol = split_subject(claim['subject'])
    earlier_path, earlier_symbol = split_subject(earlier['subject'])
    if claim_symbol != earlier_symbol:
        return False
    if claim_path == earlier_path:
        return True
    renames = {change['old_path']: path for path, change in manifest['diff_files'].items() if change['old_path']}
    renames.update(manifest.get('previous_renames', {}))
    return bool(earlier_path and renames.get(earlier_path) == claim_path)


def prior_evidence_paths(record: dict[str, Any]) -> set[str]:
    paths = set()
    for ref in record.get('claim', {}).get('evidence') or []:
        if isinstance(ref, dict) and isinstance(ref.get('path'), str):
            paths.add(ref['path'])
    return paths


def validate_resolution(key: str, decision: dict[str, Any], previous: dict[str, Any], manifest: dict[str, Any]) -> None:
    if key not in previous:
        raise ValueError('Only a prior claim can be resolved')
    fixes = decision.get('fix_evidence', [])
    removed = decision.get('removed_paths', [])
    if not isinstance(fixes, list) or not isinstance(removed, list) or not (fixes or removed):
        raise ValueError('Resolution needs source evidence of the fix')
    evidence = prior_evidence_paths(previous[key])
    renamed_from = {new: old for old, new in manifest.get('previous_renames', {}).items()}
    cited_evidence = False
    changed = False
    for ref in fixes:
        reference(ref, manifest)
        if ref.get('side', 'RIGHT') != 'RIGHT':
            raise ValueError('Resolution must cite the current fix')
        previous_path = renamed_from.get(ref['path'], ref['path'])
        if previous_path not in evidence and ref['path'] not in evidence:
            continue
        cited_evidence = True
        current_blob = manifest['sources']['RIGHT'][ref['path']].get('git_blob')
        previous_blob = manifest.get('previous_tree', {}).get(previous_path)
        if current_blob and previous_blob and current_blob != previous_blob:
            changed = True
    for path in removed:
        repository_path(path)
        if path not in manifest.get('previous_tree', {}) or path in manifest['sources']['RIGHT']:
            raise ValueError('Resolution did not remove the cited source path')
        if path in manifest.get('previous_renames', {}) or any(
            change.get('old_path') == path and new_path in manifest['sources']['RIGHT']
            for new_path, change in manifest['diff_files'].items()
        ):
            raise ValueError('Resolution moved the cited source path instead of removing it')
        if path in evidence:
            cited_evidence = True
            changed = True
    if not cited_evidence:
        raise ValueError('Resolution does not cite a prior evidence path')
    if not manifest.get('previous_tree') or not changed:
        raise ValueError('Resolution has no demonstrated source change since the prior review')


def check_finding(
    claim: Any,
    key: str,
    passes: dict[str, Any],
    findings: dict[str, Any],
    invalid_keys: set[str],
    previous: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if key and (key in findings or key in invalid_keys):
        raise ValueError('Finding has a duplicate ID')
    earlier = previous.get(key, {}).get('claim') if key else None
    historical = earlier if isinstance(claim, dict) and claim.get('kind') != 'maintenance' else None
    validate_claim(claim, manifest, historical)
    assert isinstance(claim, dict)
    valid_producer = claim['producer'] in passes or (earlier is not None and claim['producer'] == earlier['producer'])
    if not valid_producer:
        raise ValueError('Finding has an unknown producer')
    if key in previous and not same_identity(claim, previous[key]['claim'], manifest):
        raise ValueError('Prior finding ID was reused for a different identity')
    return claim


def check_decision(
    decision: Any,
    seen_decisions: set[str],
    findings: dict[str, Any],
    invalid_keys: set[str],
    previous: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(decision, dict):
        raise ValueError('Decision must be an object')
    key = text(decision.get('finding_id'), 'finding_id')
    if key in seen_decisions:
        raise ValueError('Decision has a duplicate or unknown finding ID')
    if key not in findings and key not in invalid_keys:
        # A candidate the model rejected and left out of findings is not a coverage gap.
        # An open prior claim still has to be revalidated.
        if decision.get('decision') == 'reject' and key not in open_claim_ids(previous):
            seen_decisions.add(key)
            return key, None
        raise ValueError('Decision has a duplicate or unknown finding ID')
    seen_decisions.add(key)
    if key in invalid_keys:
        return key, None
    if decision.get('checker') != 'cross-check':
        raise ValueError('A finding needs an independent cross-checker')
    if decision.get('decision') not in {'keep', 'narrow', 'question', 'reject', 'resolved'}:
        raise ValueError('Unknown verification decision')
    text(decision.get('reason'), 'verification reason')
    checked = decision.get('checked')
    historical = (
        previous.get(key, {}).get('claim') if decision['decision'] in {'resolved', 'reject', 'question'} else None
    )
    validate_claim(checked, manifest, historical)
    assert isinstance(checked, dict)
    if (
        checked['id'] != findings[key]['id']
        or checked['producer'] != findings[key]['producer']
        or not same_identity(checked, findings[key], manifest)
    ):
        raise ValueError('Verification changed the finding identity')
    if decision['decision'] == 'question':
        text(decision.get('unknown'), 'unknown')
        text(decision.get('check'), 'check')
    if decision['decision'] == 'resolved':
        validate_resolution(key, decision, previous, manifest)
    return key, decision


def gap_boundary(gaps: list[str], label: str, examine: Any) -> Any:
    try:
        return examine()
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        gaps.append(f'{label}: {error}')
        return None


def open_claim_ids(previous: dict[str, Any]) -> set[str]:
    return {
        key
        for key, record in previous.items()
        if record.get('status') not in {'resolved', 'reject'} and record.get('claim', {}).get('kind') != 'maintenance'
    }


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
        valid_checks: list[Any] = []
        for index, check in enumerate(result['checks'], 1):

            def examine_check(check: Any = check) -> Any:
                reference(check, manifest)
                text(check.get('check'), 'check')
                return check

            if gap_boundary(gaps, f'{name} check {index}', examine_check) is not None:
                valid_checks.append(check)
        exclusions: set[str] = set()
        for index, exclusion in enumerate(result['exclusions'], 1):

            def examine_exclusion(exclusion: Any = exclusion) -> str:
                if not isinstance(exclusion, dict):
                    raise ValueError('Exclusion must be an object')
                path = repository_path(exclusion.get('path'))
                if path not in manifest['files'] or path in exclusions:
                    raise ValueError('Invalid or duplicate exclusion')
                text(exclusion.get('reason'), 'exclusion reason')
                return path

            path = gap_boundary(gaps, f'{name} exclusion {index}', examine_exclusion)
            if path is not None:
                exclusions.add(path)
        if result['status'] == 'complete' and not result['checks'] and exclusions != set(manifest['files']):
            gaps.append(f'{name}: no source check or complete exclusion')
        if name in {'lines', 'rules'}:
            cited = {check['path'] for check in valid_checks}
            for path in manifest['files']:
                if path not in exclusions and path not in cited:
                    gaps.append(f'{name}: {path} has no review or exclusion')
    required = [*manifest['required_passes']]
    if data.get('user_surface') is True:
        required.append('walk')
    elif type(data.get('user_surface')) is not bool:
        raise ValueError('user_surface must be explicit')
    gaps.extend(f'{name}: no result' for name in required if name not in passes)
    findings: dict[str, Any] = {}
    invalid_keys: set[str] = set()
    previous = {r['claim']['id']: r for r in manifest.get('previous', {}).get('records', [])}
    for index, claim in enumerate(data['findings'], 1):
        raw_id = claim.get('id') if isinstance(claim, dict) else None
        key = raw_id if isinstance(raw_id, str) and raw_id.strip() else ''
        label = key or f'finding {index}'

        def examine_finding(claim: Any = claim, key: str = key) -> Any:
            return check_finding(claim, key, passes, findings, invalid_keys, previous, manifest)

        accepted = gap_boundary(gaps, label, examine_finding)
        if accepted is not None:
            findings[key] = accepted
        elif key:
            findings.pop(key, None)
            invalid_keys.add(key)
    decisions: dict[str, Any] = {}
    seen_decisions: set[str] = set()
    for index, decision in enumerate(data['decisions'], 1):
        raw_id = decision.get('finding_id') if isinstance(decision, dict) else None
        key = raw_id if isinstance(raw_id, str) and raw_id.strip() else ''
        label = key or f'decision {index}'

        def examine_decision(decision: Any = decision) -> Any:
            return check_decision(decision, seen_decisions, findings, invalid_keys, previous, manifest)

        accepted = gap_boundary(gaps, label, examine_decision)
        if accepted is None:
            if key:
                seen_decisions.add(key)
                findings.pop(key, None)
                decisions.pop(key, None)
                invalid_keys.add(key)
            continue
        key, validated_decision = accepted
        if validated_decision is not None:
            decisions[key] = validated_decision
    for key, decision in decisions.items():
        if decision['decision'] == 'question':
            gaps.append(f'{key}: decisive fact remains unresolved')
    for key, claim in findings.items():
        if claim['kind'] != 'maintenance' and key not in decisions:
            gaps.append(f'{key}: not independently checked; omitted from published findings')
    questions: list[Any] = []
    for number, question in enumerate(data['questions'], 1):

        def examine_question(question: Any = question) -> Any:
            if not isinstance(question, dict):
                raise ValueError('Question must be an object')
            text(question.get('unknown'), 'unknown')
            text(question.get('check'), 'check')
            return question

        accepted = gap_boundary(gaps, f'question {number}', examine_question)
        if accepted is not None:
            questions.append(accepted)
            gaps.append(f'question {number}: decisive fact remains unresolved')
    gaps.extend(
        f'{key}: prior claim not revalidated'
        for key in sorted(open_claim_ids(previous) - decisions.keys() - invalid_keys)
    )
    if status == 'skipped' and (data['findings'] or data['decisions'] or findings or invalid_keys):
        raise ValueError('A skipped review cannot contain findings')
    incomplete = bool(gaps) and status == 'complete'
    return {
        'findings': findings,
        'decisions': decisions,
        'questions': questions,
        'gaps': gaps,
        'status': 'incomplete' if incomplete else status,
        'reason': 'The host found coverage gaps.' if incomplete else data.get('reason', ''),
    }


def source_link(ref: dict[str, Any], manifest: dict[str, Any]) -> str:
    side = ref.get('side', 'RIGHT')
    sha = manifest['head_sha'] if side == 'RIGHT' else manifest['merge_base_sha']
    path = ref['path']
    if side == 'LEFT' and path in manifest['diff_files']:
        path = manifest['diff_files'][path]['old_path'] or path
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


def scrub_marker(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(STATE_MARKER, '')
    if isinstance(value, list):
        return [scrub_marker(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_marker(item) for key, item in value.items()}
    return value


def publication_body(body: str, appendix: str = '') -> str:
    appendix = appendix.replace(STATE_MARKER, '')
    marker = '\n\n' + STATE_MARKER
    cut = body.rfind(marker)
    prose, state = (body[:cut], body[cut:]) if cut != -1 else (body, '')
    prose = prose.replace(STATE_MARKER, '')
    combined = prose + appendix
    if state and len((combined + state).encode()) > MAX_BODY_BYTES:
        state = ''
    published = combined + state
    if len(published.encode()) > MAX_BODY_BYTES:
        raise ValueError('Review body exceeds the publication limit')
    return published


def render(data: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    checked = validate(data, manifest)
    body = [
        '## Claude review',
        '',
        f'Reviewed {manifest["merge_base_sha"]}..{manifest["head_sha"]}.',
        '',
        scrub_marker(data['assessment']),
    ]
    if manifest.get('focus'):
        body += ['', '### Focus', '', scrub_marker(data['focus_answer'])]
    if checked['status'] == 'incomplete':
        body += ['', 'Coverage incomplete: ' + checked['reason'], '']
        body += ['- ' + gap for gap in checked['gaps']]
    elif checked['status'] == 'skipped':
        body += ['', 'Skipped: ' + checked['reason']]
    comments: list[dict[str, Any]] = []
    maintenance: list[str] = []
    questions = [scrub_marker(question) for question in checked['questions']]
    previous = {r['claim']['id']: r for r in manifest.get('previous', {}).get('records', [])}
    records: dict[str, Any] = {
        key: {**value, 'status': 'unverified'} for key, value in previous.items() if key in open_claim_ids(previous)
    }
    published = 0
    for key, original in checked['findings'].items():
        decision = checked['decisions'].get(key)
        if decision is None and original['kind'] != 'maintenance':
            continue
        claim = scrub_marker(original if decision is None else decision['checked'])
        verdict = 'keep' if decision is None else decision['decision']
        fingerprint = claim_digest(claim, manifest) if verdict in {'keep', 'narrow'} else ''
        earlier = previous.get(key, {})
        record = {
            'claim': claim,
            'status': verdict,
            'reason': '' if decision is None else scrub_marker(decision['reason']),
            'claim_digest': fingerprint,
            'posted': False,
            'ever_posted': bool(earlier.get('ever_posted') or earlier.get('posted')),
        }
        if claim['kind'] != 'maintenance':
            records[key] = record
        if verdict == 'question':
            questions.append(scrub_marker({'unknown': decision['unknown'], 'check': decision['check']}))
            continue
        if verdict == 'resolved':
            if record['ever_posted']:
                body += ['', f'Resolved: {claim["title"]}. {record["reason"]}']
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
        link = f'[source]({source_link(ref, manifest)})'
        location = claim.get('location')
        anchor = None if claim['kind'] == 'maintenance' else claim_anchor(claim, manifest['diff_files'])
        if claim['kind'] == 'maintenance':
            maintenance.append(f'- **{claim["title"]}** ({link}): {claim["comment"]}')
        elif anchor is not None:
            body += ['', f'- **{label}: {claim["title"]}** ({link}).']
            same_range = isinstance(location, dict) and same_comment_range(anchor, location)
            if same_range and location.get('side', 'RIGHT') == 'RIGHT':
                if '```suggestion' not in comment_body:
                    comment_body += suggestion_fence(claim, location)
            else:
                comment_body = without_suggestion(comment_body)
            comments.append({**anchor, 'body': comment_body})
        else:
            body += body_finding(label, claim['title'], link, claim['comment'])
        published += 1
    if not published and checked['status'] == 'complete':
        body += ['', 'No new supported findings.' if previous else 'No supported findings.']
    if maintenance:
        body += ['', '### Maintenance suggestions', '', *maintenance]
    if questions:
        body += ['', '### Questions', '']
        body += [f'- {q["unknown"]} Check: {q["check"]}' for q in questions]
    state = {
        'schema_version': VERSION,
        'repo': manifest['repo'],
        'pr': manifest['pr'],
        'run_id': manifest.get('run_id', ''),
        'head_sha': manifest['head_sha'],
        'records': [record for record in records.values() if record['status'] not in {'resolved', 'reject'}],
    }
    encoded = json.dumps(state, sort_keys=True, ensure_ascii=True).replace('>', '\\u003e')
    if len('\n'.join(body).encode()) > MAX_BODY_BYTES:
        raise ValueError('Review body exceeds the publication limit')
    if len(encoded.encode()) + len('\n'.join(body).encode()) <= MAX_BODY_BYTES:
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
        'status': checked['status'],
        'reason': checked['reason'],
        'passes': pass_stats,
        'raised': len(checked['findings']),
        'decisions': {key: value['decision'] for key, value in checked['decisions'].items()},
        'published': published,
        'gaps': checked['gaps'],
    }
    return {
        'commit_id': manifest['head_sha'],
        'body': publication_body('\n'.join(body)),
        'event': 'COMMENT',
        'comments': comments,
    }, stats
