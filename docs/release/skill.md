---
name: pixeltable
description: >
  Build multimodal AI apps with Pixeltable. The skill is maintained in the
  pixeltable-skill repository; this page says where to get it and how to install it.
license: Apache-2.0
metadata:
  author: Pixeltable
  canonical: https://github.com/pixeltable/pixeltable-skill
---

The Pixeltable skill lives in one place, [pixeltable/pixeltable-skill](https://github.com/pixeltable/pixeltable-skill).
Install it into your agent:

```bash
npx skills add pixeltable/pixeltable-skill
```

With only Python, `pxt skills install` writes it into the current directory for Claude Code and the agents that
read `.agents/skills` ([CLI reference](https://docs.pixeltable.com/platform/cli#coding-agents)).

To read it without installing, start at
[SKILL.md](https://raw.githubusercontent.com/pixeltable/pixeltable-skill/main/skills/pixeltable-skill/SKILL.md);
it links to its references. The documentation is at [docs.pixeltable.com](https://docs.pixeltable.com/), and
[llms.txt](https://docs.pixeltable.com/llms.txt) is the agent-readable index of it.
