# AGENTS.md

Instructions for AI coding agents editing Pixeltable documentation under `docs/release/`.

## Core principles

- Write what users need to succeed—no more, no less. Every sentence should earn its place.
- Before creating new content, search the repo for existing pages that already cover the topic. Prefer updating over duplicating.
- Favor minimal, precise edits. Don't rewrite a page when a paragraph fix will do.
- If a proposed change or direction seems wrong, say so and explain why. Good docs come from honest pushback.
- When something is unclear or underspecified, ask before you write. Don't fill gaps with assumptions.
- Never fabricate information. If you don't know something, say so.
- Link between related pages and sections. When you mention a concept that's documented elsewhere, cross-reference it so users can find their way naturally.

## Voice

Keep prose dry and direct. State requirements and behavior plainly.

- **Cut hedges and second-person nudges.** "just", "simply", "make sure you", "you'll want to". Prefer "X requires Y" over "make sure you have Y so you can do X".
- **Cut connective filler.** "so that", "in order to", "be sure to" when a flat sentence works.
- **Avoid em dash overuse.** At most one em dash per short paragraph. Before reaching for a second dash, try a colon (for a list or expansion) or parentheses (for an aside). Multiple dashes in close succession read as AI-generated.
- **No conversational framing in callouts and step descriptions.** "Localization requires the latest CLI version" beats "Localization is under active development, so make sure you're on the latest CLI before configuring it."
- **No positional references.** Don't write "as shown above", "shown below", "the example above", or anything that makes the reader scroll to another part of the page to follow the sentence. Name the thing concretely instead: "add the page id under `navigation` in `docs.json`", not "add the page id as shown above". These break for readers who land mid-page and read as filler.

## Callouts

Before adding any callout — `<Note>`, `<Warning>`, `<Tip>`, `<Info>`, or similar — first evaluate whether the information integrates cleanly into the surrounding prose. Prose is the default; a callout is the exception. This check applies to every callout, not just when one would end up next to another.

Reach for a callout only when the information is a genuine aside: a real but secondary consideration that would break the main flow if inlined. If the content is part of the narrative — the reason an instruction matters, a consequence of a step, a condition on the thing just stated — it belongs in the prose that makes that point.

- If a callout explains *why* an instruction matters, merge it into the sentence that gives the instruction.
- If a callout restates or qualifies a nearby statement, fold it into that statement.
- Never split one idea across a sentence and a callout, or across two callouts.

**Never stack two callouts back-to-back.** A callout must not be immediately adjacent to another callout — they need body prose between them. When two would end up adjacent, at least one of them almost always fails the integration check above: inline it. If two genuinely separate asides remain, keep only the stronger one as a callout.

A single well-chosen callout earns attention; a callout that could have been a sentence dilutes it, and two side by side cancel each other out.

## Link checking

Internal links between documentation pages use **URL paths built from the Mintlify navigation config**, not file paths on disk or relative paths.

**Before writing any internal link, open `docs/release/docs.json` and confirm the page id exists in `navigation`.** Directory paths on disk are not a reliable hint — folders can be absent from the URL, and navigation grouping frequently differs from folder names.

### URL format

```
/{page-id}
```

| Segment | Source | Example |
|---------|--------|---------|
| `{page-id}` | String entry under `navigation` in `docs/release/docs.json` | `overview/quick-start`, `howto/deployment/serving`, `sdk/latest/pixeltable` |

Page ids match paths under `docs/release/` without the file extension (`.mdx` / `.ipynb`). Nested groups in `docs.json` organize the sidebar only; they are not extra URL segments.

### Example

Given `docs/release/docs.json`:
```json
{
  "navigation": {
    "tabs": [
      {
        "tab": "User Guide",
        "groups": [
          {
            "group": "Getting started",
            "pages": [
              "overview/pixeltable",
              "overview/quick-start"
            ]
          }
        ]
      }
    ]
  }
}
```

And the page file `docs/release/overview/quick-start.mdx`, the correct link is:
```
/overview/quick-start
```

### Common mistakes

```markdown
<!-- WRONG: relative path -->
[Quickstart](./overview/quick-start)
[Quickstart](../overview/quick-start.mdx)

<!-- WRONG: file path on disk -->
[Quickstart](/docs/release/overview/quick-start)

<!-- WRONG: guessed URL from folder layout (not a docs.json page id) -->
[Quickstart](/getting-started/overview/quick-start)

<!-- WRONG: old readme.io docs -->
[Tutorial](https://pixeltable.readme.io/docs/tutorial)

<!-- CORRECT: URL from docs.json page id -->
[Quickstart](/overview/quick-start)
[Serving](/howto/deployment/serving)
[RAG cookbook](/howto/cookbooks/agents/pattern-rag-pipeline)
```

### Steps to construct a link

1. Open `docs/release/docs.json`.
2. Find the target page id in `navigation` (tabs → groups → `pages` arrays, including nested groups).
3. Confirm a matching source file exists under `docs/release/` (`.mdx` or `.ipynb`).
4. Assemble: `/{page-id}`.
5. For a section on that page, append `#anchor` using the heading text rules Mintlify uses for `##` / `###` headings.

### What's fine as-is

- **External URLs**: `https://...` — no change needed.
- **Image paths**: `./images/screenshot.png` — relative paths are correct for images.
- **Same-page anchors**: `#section-name` — these don't need a full path.
- **Anchors on internal links**: `/howto/deployment/serving#quickstart-python` — append `#anchor` to the URL path.
- **GitHub raw assets**: use `raw.githubusercontent.com` (not `raw.github.com`).

## Cross-referencing

When new functionality is documented — whether on a new page or as new behavior added to an existing page — related pages elsewhere in the docs usually need pointers to it. Without those pointers, the canonical content is hard to discover from the angles a reader is most likely to come in from.

### Pick a canonical home

One page owns the full explanation; every other page that touches the topic links *to* it rather than restating it.

- Walkthroughs, setup steps, and concept explanations live on the canonical page.
- Reference pages get a sentence + link, not a paragraph of duplicated detail.
- If you find yourself copying more than a sentence or two into a reference page, the content probably belongs on the canonical page.

### Find the targets

Sweep the docs for related touchpoints before assuming you're done:

```bash
grep -rln "<feature-name>\|<related-keyword>" docs/release --include="*.mdx" --include="*.ipynb"
```

Then read each candidate to find the natural insertion point. Common targets:

- **Platform / how-to pages** — e.g., a feature affects embedding indexes → cross-link from `docs/release/platform/embedding-indexes.ipynb` or the matching page id in `docs.json`.
- **Cookbook pages** — e.g., a feature shows up in a RAG or video workflow → cross-link from `docs/release/howto/cookbooks/`.
- **Adjacent feature pages** — e.g., iterators, computed columns, serving, when scope or behavior overlaps.
- **Overview / landing pages** — sometimes a card on a hub page is warranted; usually only when the feature is a top-level setup step, not a configuration detail.

### Pick the form

| Form | Use when |
|------|----------|
| Inline sentence in an existing paragraph | The reference fits a list of similar items already on the page (e.g., "for sites with X, Y, or [new thing]"). |
| `<Note>` callout | The reference is a real but secondary consideration that would distract from the main flow if inlined. |
| New section (`##`) | The target page genuinely needs to document the feature from its own angle (e.g., dashboard config for the feature). |
| New page | Only if the feature has substantial standalone content that doesn't fit elsewhere. |

Default to the lightest form that works. New pages and new sections add maintenance surface; inline sentences and Notes don't.

### Phrase inline links naturally

When the form is inline (the common case), put the link on a natural noun phrase inside the sentence that's already making the point. Don't append a "see [page]" sentence, and don't wrap a one-line pointer in a standalone `<Tip>` whose only job is to host the link.

- Good: "Pixeltable stores results in computed columns and exposes them through [similarity search](/platform/embedding-indexes)."
- Bad: "Pixeltable stores results in computed columns. See the [embedding indexes guide](/platform/embedding-indexes) for details."
- Bad: A standalone `<Tip>` containing only "For more, see [embedding indexes](/platform/embedding-indexes)."

Reserve `<Tip>` and `<Card>` callouts for genuinely orthogonal pointers — a tutorial, a related concept that isn't part of the current narrative — not as wrappers for a link that already belongs in the prose.

### Frame by function, not by plumbing

Lead the cross-reference with what each feature *does* and how they complement or differ. Don't headline the connection with shared config infrastructure — same YAML key, sibling files in the same directory, same lifecycle.

- Good: "X decides A; Y decides B." Contrast on function.
- Bad: "Y is also configured under the same `agents` key." Contrast on plumbing.

If the only thing linking two features is that they live in the same key, that's not a cross-reference worth writing. Config mechanics belong on the page that is itself about configuration.

### Bidirectional links

Reference pages always link *to* canonical. Linking back from canonical to a reference page is optional — only do it when the reference page has additional detail the canonical page doesn't cover (e.g., dashboard click-paths). Most cross-references are one-way.

### Anchor links

Internal anchors only resolve for `##` / `###` headings. Things that look like headings but aren't:

- `<Step title="...">` inside `<Steps>` — JSX prop, no anchor generated.
- `<Tab title="...">`, `<Accordion title="...">`, `<Card title="...">` — same.

If you want to deep-link to a step or tab, add a real `##` heading nearby, or link to the page without an anchor and let the reader scroll.

### Sweep checklist

After documenting any new functionality, before declaring the work done:

1. Grep for the feature/setting name and 1–2 related keywords across `docs/release/**/*.{mdx,ipynb}`.
2. For each hit on a different page, decide: is a cross-reference warranted?
3. For each warranted cross-ref, pick the lightest form that fits.
4. Frame each cross-ref by what each feature does, not by shared config keys or directory structure.
5. If the functionality has dashboard UI, verify `docs/release/platform/dashboard.mdx` covers it — usually folded into the most relevant existing page, not a new page.
6. Verify every link you wrote — see *Link checking* above for URL construction.
7. Run `make docs` (or `make docs-serve`) and confirm the build succeeds.

## New docs pages

When adding a new MDX guide or cookbook/tutorial notebook, follow the structure used by existing sibling pages under `docs/release/`.

### Page structure

1. **Frontmatter**: `title` and `description` (one sentence starting with a verb). Notebooks also require `icon: "notebook"` and Kaggle/Colab/GitHub links in `description` (see [`GUIDELINES_FOR_NOTEBOOKS.md`](../_guidelines/GUIDELINES_FOR_NOTEBOOKS.md)).
2. **Intro paragraph**: One or two sentences explaining what the page covers and when to use it.
3. **Sections** (`##` / `###`): Mirror the closest existing page of the same type (MDX guide vs cookbook notebook).
4. **Working examples**: Prefer executable Pixeltable code. In notebooks, keep cells focused; in MDX, use fenced code blocks.

### Register in navigation

Add the page id to `docs/release/docs.json` under the appropriate `navigation` tab/group `pages` array. The page id must match the path under `docs/release/` without extension.

### Adding to a hub / overview page

When a hub page already uses `<CardGroup>` / `<Card>` (for example `docs/release/use-cases/media-processing.mdx`), add a `<Card>` in the same style:

```mdx
<Card title="Page title" icon="icon-name" href="/path/from/docs.json">
  One-line description matching the page's frontmatter description
</Card>
```

### Example to follow

- MDX guides: `docs/release/overview/quick-start.mdx`, `docs/release/howto/deployment/serving.mdx`
- Provider notebooks with correct frontmatter: `docs/release/howto/providers/working-with-fabric.ipynb`
- Cookbook notebooks: siblings under `docs/release/howto/cookbooks/`

## LLM-oriented docs

Mintlify does **not** use Fern's `<llms-only>` / `<llms-ignore>` tags. Mintlify serves `llms.txt` / `llms-full.txt` automatically. Do not invent those Fern tags in Pixeltable docs.

Write content that works for humans and agents in the normal page body:

- Prefer agent-executable examples (Python / CLI) in ordinary prose and code blocks, especially in cookbooks and tutorials
- Prerequisite context and cross-references belong in the main narrative when both audiences need them
- Marketing CTAs and decorative-only material should stay minimal so they do not dominate agent context

### Rules

- Don't overuse audience-specific splits — most content should be visible to both audiences. Only separate concerns where human and agent needs clearly diverge.
- Length depends on shape, not a fixed limit:
  - **Inline blocks** (interleaved with mixed-audience content — e.g., the programmatic equivalent of a UI step, a prerequisite note, or a cross-reference) should stay short, usually a few sentences. If an inline block grows long, it's probably regular content that belongs to both audiences.
  - **Standalone sections** that are entirely agent-oriented (troubleshooting / common errors, architecture overviews, CI recipes) can be longer. Keep the block self-contained rather than referring back to surrounding prose.
- Tutorials are a common use case: the human version might walk through a UI with screenshots, while a nearby code block can add the equivalent Python or CLI snippet that an agent can execute directly.

### Example to follow

See `docs/release/overview/building-pixeltable-with-llms.mdx` for Pixeltable's agent-oriented docs entry point, and cookbooks under `docs/release/howto/cookbooks/` for executable examples.

## Notebooks

Documentation notebooks live under `docs/release/**/*.ipynb`. For notebook how-to (raw-cell YAML frontmatter, no H1 in markdown, Kaggle/Colab/GitHub links, code/markdown conventions, testing checklist), **read and follow** [`docs/_guidelines/GUIDELINES_FOR_NOTEBOOKS.md`](../_guidelines/GUIDELINES_FOR_NOTEBOOKS.md) before editing notebooks. Apply the writing craft in this file to notebook prose.

## Docstrings

Docstrings live in Python source under `pixeltable/`, not under `docs/release/`. For MDX-safe docstring rules, follow [`docs/_guidelines/GUIDELINES_FOR_DOCSTRINGS.md`](../_guidelines/GUIDELINES_FOR_DOCSTRINGS.md). After docstring edits that affect SDK pages, run `make docs` (or `make docs-serve`).

## Protected Configuration

Never change `integrations.telemetry.enabled` in `docs/release/docs.json`. It must remain set to `true` in every pull request.

## Local build and preview

You **must** use these Makefile recipes (they wrap `pixeltable-doctools` + Mintlify). Do not invent alternate `mint` / `quarto` invocations as the primary workflow.

```bash
# Build docs locally (uv sync + pixeltable_doctools.build + mint broken-links)
make docs

# Build and serve with live reload (runs `make docs`, then `mint dev` in target/docs)
make docs-serve

# Optional: stricter broken-link check after a build
make linkscheck

# Deploy to the docs-tools *dev* environment only (when a deploy is needed for preview)
make docs-deploy TARGET=dev
```

- You **must** run `make docs` (or `make docs-serve`) to locally build/preview after docs edits.
- `make docs-deploy TARGET=dev` is the only deploy target agents may use or suggest.
- **Never** run or recommend `make docs-deploy TARGET=stage` or `TARGET=prod`. Those are protected actions reserved for humans outside agent workflows.
- Build output lands under `target/docs/` (generated; do not hand-edit).
- Source of truth for published content is `docs/release/` (plus Python docstrings for SDK pages).

---

Adapted from [Fern's docs AGENTS.md](https://github.com/fern-api/docs/blob/main/AGENTS.md).
