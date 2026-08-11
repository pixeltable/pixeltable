@AGENTS.md

# CLAUDE.md

Claude Code reads this file, not `AGENTS.md`. The `@AGENTS.md` import pulls in the shared
contributor guidance, so keep only Claude-specific instructions here.

## Required After Every Code Change

After every code change, before reporting it done:

1. `make format`: auto-formats code.
2. `make check`: mypy + ruff static checks; both must pass.
3. Review the diff (`git diff`) and read every comment, docstring, and string you added. A comment must
   describe only the code at hand (never a caller's intent or a called function's internals) and must not
   state behavior you have not verified. Fix violations before proceeding.

Skip only if explicitly directed or if the environment makes it impossible.
