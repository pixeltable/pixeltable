"""
Sync Skill Tool

Builds docs/release/skill.md from the canonical skill in pixeltable/pixeltable-skill, so the two never
drift on rules. The canonical SKILL.md links to five references/*.md files that only exist inside the
skill package; this flattens them into one self-contained page for docs.pixeltable.com/skill.md.

    python tool/sync_skill.py            # rewrite docs/release/skill.md
    python tool/sync_skill.py --check    # exit 1 if the committed file differs from a fresh build
"""

import re
import sys
import urllib.request
from pathlib import Path

RAW = 'https://raw.githubusercontent.com/pixeltable/pixeltable-skill/main/skills/pixeltable-skill'
REFERENCES = ['cli', 'core-api', 'providers', 'workflows', 'anti-patterns']
OUTPUT = Path(__file__).resolve().parent.parent / 'docs' / 'release' / 'skill.md'
SOURCE_URL = 'https://github.com/pixeltable/pixeltable-skill/blob/main/skills/pixeltable-skill/SKILL.md'


def fetch(path: str) -> str:
    with urllib.request.urlopen(f'{RAW}/{path}', timeout=30) as resp:
        return resp.read().decode('utf-8')


def slug(title: str) -> str:
    """The anchor Mintlify gives a heading: lowercase, alphanumerics and hyphens."""
    return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')


def rewrite_links(text: str, anchor_for: dict[str, str]) -> str:
    """Point references/X.md and bare X.md links at the flattened sections, keeping any #fragment."""

    def repl(m: re.Match) -> str:
        name, frag = m.group(1), m.group(2) or ''
        # a fragment names a heading that survives flattening, so it stays; the bare file becomes its section
        return f']({frag})' if frag else f'](#{anchor_for[name]})'

    return re.sub(r'\]\((?:references/)?(' + '|'.join(REFERENCES) + r')\.md(#[a-z0-9-]+)?\)', repl, text)


def split_reference(name: str, text: str) -> tuple[str, str]:
    """A reference's title (its own H1, or the file name) and its body with headings demoted one level."""
    lines = text.strip('\n').split('\n')
    title = lines.pop(0)[2:].strip() if lines and lines[0].startswith('# ') else name
    body = '\n'.join(('#' + l) if re.match(r'^#{1,5} ', l) else l for l in lines)
    return title, body


def build() -> str:
    skill = fetch('SKILL.md')
    refs = {name: split_reference(name, fetch(f'references/{name}.md')) for name in REFERENCES}
    # the wrapper heading decides the anchor, so compute it from the real title rather than the file name
    anchor_for = {name: slug(f'Reference: {title}') for name, (title, _) in refs.items()}
    fm_end = skill.index('\n---\n', 4) + 5
    frontmatter, body = skill[:fm_end], skill[fm_end:]
    # provenance goes in frontmatter, where it is YAML rather than a comment MDX might reject
    frontmatter = frontmatter.replace('\n---\n', f'\n  source: {SOURCE_URL}\n---\n', 1) \
        if '\nmetadata:' in frontmatter else frontmatter.replace('\n---\n', f'\nmetadata:\n  source: {SOURCE_URL}\n---\n', 1)
    parts = [frontmatter, rewrite_links(body, anchor_for).rstrip('\n'), '']
    for name in REFERENCES:
        title, ref_body = refs[name]
        parts.append(f'## Reference: {title}\n\n{rewrite_links(ref_body, anchor_for)}\n')
    return '\n'.join(parts).rstrip('\n') + '\n'


def main() -> None:
    fresh = build()
    if '--check' in sys.argv:
        current = OUTPUT.read_text() if OUTPUT.exists() else ''
        if current == fresh:
            print(f'{OUTPUT.relative_to(Path.cwd())} matches the canonical skill.')
            return
        print(f'{OUTPUT.relative_to(Path.cwd())} has drifted from the canonical skill.', file=sys.stderr)
        print('Run `python tool/sync_skill.py` and commit the result.', file=sys.stderr)
        sys.exit(1)
    OUTPUT.write_text(fresh)
    print(f'wrote {OUTPUT.relative_to(Path.cwd())} ({len(fresh.split())} words)')


if __name__ == '__main__':
    main()
