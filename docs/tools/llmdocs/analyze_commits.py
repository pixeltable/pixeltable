#!/usr/bin/env python3
"""Analyze commits since last release to identify important changes."""

import subprocess
import json
from datetime import datetime
from collections import defaultdict
import re
from pathlib import Path

def get_commits_since_tag(last_tag="latest"):
    """Get commits since last release tag."""
    # Get the last tag if not specified
    if last_tag == "latest":
        last_tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"], 
            text=True
        ).strip()
    
    # Get commits since tag
    log_format = "%H|%an|%ae|%at|%s"
    commits_raw = subprocess.check_output(
        ["git", "log", f"{last_tag}..HEAD", f"--format={log_format}"],
        text=True
    ).strip().split('\n')
    
    commits = []
    for line in commits_raw:
        if not line:
            continue
        parts = line.split('|')
        commits.append({
            'hash': parts[0],
            'author': parts[1],
            'email': parts[2],
            'timestamp': int(parts[3]),
            'message': parts[4]
        })
    
    return commits, last_tag

def analyze_with_gh_cli():
    """Use GitHub CLI to get PR information."""
    # Get merged PRs since last release
    cmd = [
        "gh", "pr", "list", 
        "--state", "merged",
        "--limit", "100",
        "--json", "number,title,author,labels,files,additions,deletions,mergedAt,reviews"
    ]
    
    try:
        result = subprocess.check_output(cmd, text=True)
        prs = json.loads(result)
        return prs
    except:
        return []

def score_commit_importance(commit, pr_data=None):
    """Score a commit's importance based on various factors."""
    score = 0
    message = commit['message'].lower()
    
    # Keywords in commit message
    if 'breaking' in message or 'major' in message:
        score += 10
    if 'security' in message or 'vulnerability' in message:
        score += 9
    if 'fix' in message or 'bug' in message:
        score += 5
    if 'feat' in message or 'feature' in message:
        score += 6
    if 'perf' in message or 'performance' in message:
        score += 7
    if 'deprecat' in message:
        score += 8
    
    # Conventional commit types
    if message.startswith('feat:'):
        score += 6
    elif message.startswith('fix:'):
        score += 5
    elif message.startswith('perf:'):
        score += 7
    elif message.startswith('breaking:') or 'BREAKING CHANGE' in commit['message']:
        score += 10
    
    # Author importance (you'd need to maintain a list of core maintainers)
    core_maintainers = ['core_dev@example.com']  # Add actual maintainer emails
    if commit['email'] in core_maintainers:
        score += 2
    
    # PR data if available
    if pr_data:
        # Number of files changed
        if pr_data.get('files', []):
            score += min(len(pr_data['files']) / 10, 3)  # Cap at 3 points
        
        # Size of change
        additions = pr_data.get('additions', 0)
        deletions = pr_data.get('deletions', 0)
        if additions + deletions > 500:
            score += 3
        
        # Labels
        for label in pr_data.get('labels', []):
            label_name = label.get('name', '').lower()
            if 'breaking' in label_name:
                score += 10
            elif 'security' in label_name:
                score += 9
            elif 'enhancement' in label_name:
                score += 5
    
    return score

def categorize_commits(commits):
    """Categorize commits by type."""
    categories = defaultdict(list)
    
    for commit in commits:
        message = commit['message'].lower()
        
        # Determine category
        if 'breaking' in message or 'BREAKING CHANGE' in commit['message']:
            categories['breaking'].append(commit)
        elif 'security' in message or 'vulnerability' in message:
            categories['security'].append(commit)
        elif re.match(r'^fix(\(.+\))?:', message) or 'bug' in message:
            categories['fixes'].append(commit)
        elif re.match(r'^feat(\(.+\))?:', message) or 'feature' in message:
            categories['features'].append(commit)
        elif re.match(r'^perf(\(.+\))?:', message) or 'performance' in message:
            categories['performance'].append(commit)
        elif 'deprecat' in message:
            categories['deprecations'].append(commit)
        elif 'doc' in message or 'readme' in message:
            categories['documentation'].append(commit)
        elif 'test' in message:
            categories['tests'].append(commit)
        elif 'chore' in message or 'refactor' in message:
            categories['maintenance'].append(commit)
        else:
            categories['other'].append(commit)
    
    return categories

def generate_changelog(categories, last_tag):
    """Generate a changelog from categorized commits."""
    print(f"\n# Changelog since {last_tag}\n")
    
    # Priority order for sections
    section_order = [
        ('breaking', '⚠️ BREAKING CHANGES'),
        ('security', '🔒 Security'),
        ('deprecations', '⚠️ Deprecations'),
        ('features', '✨ Features'),
        ('performance', '⚡ Performance'),
        ('fixes', '🐛 Bug Fixes'),
        ('documentation', '📚 Documentation'),
        ('tests', '✅ Tests'),
        ('maintenance', '🔧 Maintenance'),
        ('other', '📝 Other')
    ]
    
    for category_key, section_title in section_order:
        commits = categories.get(category_key, [])
        if commits:
            print(f"\n## {section_title}\n")
            for commit in commits[:10]:  # Limit to top 10 per category
                # Clean up the message
                message = commit['message']
                # Remove conventional commit prefix
                message = re.sub(r'^(feat|fix|perf|docs|test|chore|refactor)(\(.+\))?:\s*', '', message)
                print(f"- {message} ({commit['author']})")

def main():
    """Main analysis function."""
    output_lines = []
    
    def log(msg=""):
        """Print and save to output."""
        print(msg)
        output_lines.append(msg)
    
    log("🔍 Analyzing commits since last release...")
    
    # Get commits
    commits, last_tag = get_commits_since_tag()
    log(f"Found {len(commits)} commits since {last_tag}")
    
    # Try to get PR data
    pr_data = analyze_with_gh_cli()
    log(f"Found {len(pr_data)} merged PRs")
    
    # Score commits
    for commit in commits:
        commit['importance_score'] = score_commit_importance(commit)
    
    # Sort by importance
    commits.sort(key=lambda x: x['importance_score'], reverse=True)
    
    # Show top important commits
    log("\n## 🌟 Most Important Commits:\n")
    for commit in commits[:10]:
        if commit['importance_score'] > 0:
            log(f"[Score: {commit['importance_score']:2d}] {commit['message'][:80]}")
            log(f"            by {commit['author']}")
            log()
    
    # Categorize commits
    categories = categorize_commits(commits)
    
    # Generate changelog (need to modify this function too)
    changelog_lines = []
    changelog_lines.append(f"\n# Changelog since {last_tag}\n")
    
    section_order = [
        ('breaking', '⚠️ BREAKING CHANGES'),
        ('security', '🔒 Security'),
        ('deprecations', '⚠️ Deprecations'),
        ('features', '✨ Features'),
        ('performance', '⚡ Performance'),
        ('fixes', '🐛 Bug Fixes'),
        ('documentation', '📚 Documentation'),
        ('tests', '✅ Tests'),
        ('maintenance', '🔧 Maintenance'),
        ('other', '📝 Other')
    ]
    
    for category_key, section_title in section_order:
        category_commits = categories.get(category_key, [])
        if category_commits:
            changelog_lines.append(f"\n## {section_title}\n")
            for commit in category_commits[:10]:
                message = commit['message']
                message = re.sub(r'^(feat|fix|perf|docs|test|chore|refactor)(\(.+\))?:\s*', '', message)
                changelog_lines.append(f"- {message} ({commit['author']})")
    
    # Print and save changelog
    for line in changelog_lines:
        log(line)
    
    # Save to file
    output_file = Path(__file__).parent / f"commit_analysis_{last_tag.replace('/', '_')}_to_HEAD.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n📝 Analysis saved to: {output_file}")

if __name__ == "__main__":
    main()