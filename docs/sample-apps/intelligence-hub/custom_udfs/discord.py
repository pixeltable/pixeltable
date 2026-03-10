"""Discord notification UDF.

Uses Discord Webhooks -- no SDK required.
Setup: Server Settings > Integrations > Webhooks > New Webhook
"""

import json

import requests

import pixeltable as pxt


@pxt.udf
def send_message(webhook_url: str, content: str, username: str = 'Pixeltable Hub') -> pxt.Json:
    """Send a message to a Discord channel via webhook.

    Example:
        >>> t.add_computed_column(
        ...     alert=discord.send_message(DISCORD_WEBHOOK_URL, t.summary)
        ... )
    """
    resp = requests.post(
        webhook_url,
        data=json.dumps({'content': content, 'username': username}),
        headers={'Content-Type': 'application/json'},
        timeout=10,
    )
    return {'ok': resp.status_code in (200, 204), 'status': resp.status_code}
