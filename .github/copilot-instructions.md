# Copilot reviews

Use root `AGENTS.md` for shared invariants and code review criteria. Matching
`.github/instructions/*.instructions.md` files add scoped checks.

Report verified, introduced defects with their trigger and user impact. Consolidate repeated findings;
correct code needs no comment. Formatting and statically detectable type errors belong to CI, while UDF
annotations and schema contracts still require semantic review. Apply related-file checks only when a
public behavior or contract changes. Do not turn examples or intentional exceptions into blanket rules.
