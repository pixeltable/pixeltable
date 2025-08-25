#!/usr/bin/env python3
"""
Quick validation of generated MDX files for common issues.
"""

import re
from pathlib import Path

def check_mdx_file(file_path):
    """Check an MDX file for common issues."""
    issues = []
    
    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check frontmatter
    if not content.startswith('---'):
        issues.append("Missing frontmatter")
    
    # Check for problematic patterns
    for i, line in enumerate(lines, 1):
        # Check for bare URLs that might be interpreted as JSX
        if re.search(r'https?://[^\s]+/[^\s]+>', line) and not re.search(r'\[.*\]\(https?://', line):
            issues.append(f"Line {i}: Potential JSX-like URL")
        
        # Check for RST directives
        if re.search(r':[a-z]+:`', line):
            issues.append(f"Line {i}: RST directive found: {line[:50]}")
        
        # Check for Python REPL continuations outside code blocks
        if line.strip().startswith('... ') and i > 3:  # Skip frontmatter
            # Check if we're in a code block
            in_code_block = False
            for j in range(max(0, i-10), i):
                if lines[j].strip() == '```python':
                    in_code_block = True
                    break
                if lines[j].strip() == '```':
                    in_code_block = False
            if not in_code_block:
                issues.append(f"Line {i}: Python continuation outside code block")
    
    return issues

# Check all generated MDX files
sdk_dir = Path("/Users/lux/repos/pixeltable-docs/docs/mintlify/docs/sdk")
problem_files = []

for mdx_file in sdk_dir.rglob("*.mdx"):
    issues = check_mdx_file(mdx_file)
    if issues:
        problem_files.append((mdx_file, issues))

# Report findings
if problem_files:
    print(f"Found issues in {len(problem_files)} files:\n")
    for file_path, issues in problem_files[:10]:  # Show first 10
        print(f"{file_path.relative_to(sdk_dir)}:")
        for issue in issues[:3]:  # Show first 3 issues per file
            print(f"  - {issue}")
        if len(issues) > 3:
            print(f"  ... and {len(issues) - 3} more issues")
        print()
else:
    print("✅ No issues found in MDX files!")