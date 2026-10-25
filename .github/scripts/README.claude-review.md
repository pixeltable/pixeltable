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
Use `--focus="--strict"` when the focus starts with a dash. Preparation writes a compact `manifest.json` for Claude and
a companion `sources.json` with full-tree line counts and SHA-256 digests for host validation.

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

Every published defect, rule finding, or design objection needs a declared independent decision. The renderer uses the exact
checked record. Coverage, finding, and decision gaps produce an incomplete result and preserve valid checked
findings. Invalid snapshots, tampered sources, and malformed pass envelopes still fail validation. A review can skip
only when preparation supplies a whole-change policy, currently an empty diff. A comment on a line outside the
diff hunks is still attached when that file is part of the pull request. A file the pull request does not change
stays in the body. A suggestion replaces only the right-side lines the comment is attached to, and only
when that range is shorter than six lines. Existing debt is published as maintenance and does not by itself make the review
incomplete. A skipped result records its reason in the artifacts and makes no publication request.
Binary, rename, and mode-only changes retain metadata without requiring nonexistent text lines. Relevant passes
still inspect their effects or record an exclusion.

Publication stops when the head, comparison merge base, or target branch changes. An unrelated base-branch advance
can proceed when its merge base stays the same. A source change during submission records a stale result tied to the
original revision. If GitHub rejects a batch that mixes diff lines and unchanged lines, the host retries with the
diff-line comments and moves the rest into the body. A later rejection drops optional history when needed and checks
the final body size before retrying.

History contains bounded open claims. Earlier open claims are revalidated even when their comment file did not change. An
unchanged, reverified claim can avoid duplicate publication. Only a checked fix resolves a prior claim. Missing or
incompatible history disables reuse. Local results without a workflow-run identity do not establish trusted CI history.
Resolved and rejected claims leave history. An unchanged fix therefore needs no new resolution on the next run.
Historical evidence can support a non-published decision only when it matches the trusted prior claim exactly.

The validator checks declared producer and checker identities and coverage accounting. It cannot prove the quality of
an agent's reasoning or independently recover child payloads the runtime does not expose. Per-pass model usage is
unavailable unless the runtime supplies reliable attribution. Aggregate usage comes from its execution result. Per-pass candidate counts refer to records retained in the envelope.
Internal ideas discarded before that result are not measured.

Run the host regression tests from either repository:

```sh
python -m unittest discover -s .github/scripts -p 'test_*.py'
```

When updating the shared review files in both repositories, compare them before creating the paired PRs. This check
is manual and requires both checkouts:

```sh
python .github/scripts/check_claude_review_sync.py /path/to/other/checkout
```

Keep additional checkers, coverage waves, and reference selection as separate experiments. Evaluate fixed cases from
both repositories, including harmless changes and known false positives. Record source-backed findings, known misses,
actual cost, and duration. A larger comment count does not establish a better review.
