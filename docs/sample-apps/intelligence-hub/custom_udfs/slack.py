"""Slack notification UDF.

Uses Slack Incoming Webhooks -- no SDK required.
Setup: https://api.slack.com/messaging/webhooks
"""

import json

import requests

import pixeltable as pxt


@pxt.udf
def send_message(webhook_url: str, text: str) -> pxt.Json:
    """Send a message to a Slack channel via incoming webhook.

    Example:
        >>> t.add_computed_column(
        ...     alert=slack.send_message(SLACK_WEBHOOK_URL, t.summary)
        ... )
    """
    resp = requests.post(
        webhook_url, data=json.dumps({'text': text}), headers={'Content-Type': 'application/json'}, timeout=10
    )
    return {'ok': resp.status_code == 200, 'status': resp.status_code}
