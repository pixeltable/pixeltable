# Review a PR locally

The same preparation and validation scripts run locally and in `claude-review.yml`. They freeze the PR's commits and
diff, validate source references and independent decisions, and keep supported findings separate from missing coverage.
Claude reads the prepared source. The host scripts handle files and publication.

Use Python 3.11 or newer, Git, an authenticated GitHub CLI, and an authenticated Claude Code installation. Run from a
trusted checkout containing the review command and its skills. The preparation output must be a new directory.

```sh
review_dir="$(mktemp -d)/review"
pr_url="<full URL of an open PR>"
python .github/scripts/prepare_claude_review.py "$pr_url" --output "$review_dir"
```

Use an open PR for routine reviews. Set `PXT_REVIEW_ALLOW_CLOSED=1` only when preparing a historical evaluation case.
Add `--focus "error paths"` during preparation to require the focus pass. Prior bot records are input for current
verification. `--no-history` starts an independent evaluation without prior records.

Run the command in Claude Code:

```text
/pxt-review <PR URL> --review-context <absolute path to manifest.json>
```

For a noninteractive run, capture the execution result and set an explicit model budget:

```sh
claude -p "/pxt-review $pr_url --review-context $review_dir/manifest.json" \
  --output-format json --max-budget-usd 8 \
  --allowedTools "Read" "Grep" "Glob" "Bash(gh pr view:*)" "Bash(gh pr checks:*)" \
  > "$review_dir/execution.json"
python .github/scripts/post_claude_review.py \
  --review-context "$review_dir/manifest.json" \
  --execution-file "$review_dir/execution.json" --validate-only
```

The budget is a spending limit for this invocation, not an estimate of review cost. CI starts with the same USD 8
limit. The Claude step has 30 minutes under the 45-minute job limit, leaving time for host steps.
Set the repository variable `CLAUDE_REVIEW_MAX_BUDGET_USD` to change it. Claude authentication is
required to generate findings. Preparation and validation can run without it. This change was tested with real GitHub
snapshots and local validation. A model comparison requires authenticated old and new command runs at fixed revisions.

For an interactive result, save the final JSON to a file and pass `--result <file>` instead of `--execution-file`.
Inspect `review.json`, `stats.json`, and `outcome.json` in the output directory. Validation-only mode makes no
publication request. To publish an already validated result, run the same host command without `--validate-only`.

Every published defect, rule finding, or design objection needs an independent decision. The renderer uses the exact
checked record. Missing passes, decisive unresolved questions, or unverified candidates prevent a complete outcome. A review can skip
only when preparation supplies a whole-change policy, currently an empty diff. Invalid inline locations move to
the body individually. If the live head or base changes, publication stops. A push during submission records a stale
result tied to its original revision.

History is cumulative and bounded. Earlier claims are revalidated even when their comment file did not change. An
unchanged, reverified claim can avoid duplicate publication. Only a checked fix resolves a prior claim. Missing or
incompatible history disables reuse. Local results without a workflow-run identity do not establish trusted CI history.

The validator checks declared producer and checker identities and coverage accounting. It cannot prove the quality of
an agent's reasoning or independently recover child payloads the runtime does not expose. Per-pass model usage is
unavailable unless the runtime supplies reliable attribution. Aggregate usage comes from its execution result. Per-pass candidate counts refer to records retained in the envelope.
Internal ideas discarded before that result are not measured.

Run the host regression tests from either repository:

```sh
python -m unittest discover -s .github/scripts -p 'test_*.py'
```

When updating the shared review files in both repositories, compare them before creating the paired PRs:

```sh
python .github/scripts/check_claude_review_sync.py /path/to/other/checkout
```

Keep additional checkers, coverage waves, and reference selection as separate experiments. Evaluate fixed cases from
both repositories, including harmless changes and known false positives. Record source-backed findings, known misses,
actual cost, and duration. A larger comment count does not establish a better review.
