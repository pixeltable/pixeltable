"""UDF definitions for the Intelligence Hub.

All @pxt.udf functions live here -- imported by setup_pixeltable.py.
Custom service-integration UDFs (Slack, Discord, etc.) live in custom_udfs/.
"""

import config
import pixeltable as pxt


@pxt.udf
def make_summary_prompt(title: str | None, origin: str | None) -> list[dict]:
    """Build the messages list for the summarization LLM call."""
    return [{'role': 'user', 'content': (
        f'Summarize the following content in 2-3 sentences:\n\n'
        f'Title: {title or "Untitled"}\n\n'
        f'This is a {origin or "unknown"} source.'
    )}]


@pxt.udf
def score_relevance(summary: str | None) -> float:
    """Keyword-based relevance heuristic.

    Replace with your own LLM-based scoring for production use.
    """
    if not summary:
        return 0.0
    text_lower = summary.lower()
    matches = sum(1 for kw in config.RELEVANCE_KEYWORDS if kw.lower() in text_lower)
    return min(matches / len(config.RELEVANCE_KEYWORDS), 1.0)


@pxt.udf
def format_alert(title: str | None, summary: str | None, relevance: float | None) -> str:
    """Format a notification message from row data."""
    score = f'{relevance:.0%}' if relevance is not None else 'N/A'
    return (
        f'[Intelligence Hub] New item (relevance {score})\n\n'
        f'*{title or "Untitled"}*\n'
        f'{summary or "No summary"}'
    )


@pxt.udf
def make_export_row(
    title: str | None,
    origin: str | None,
    summary: str | None,
    relevance: float | None,
) -> pxt.Json:
    """Build a flat list of values for Google Sheets export."""
    return [
        title or '',
        origin or '',
        summary or '',
        f'{relevance:.2f}' if relevance is not None else '0',
    ]
