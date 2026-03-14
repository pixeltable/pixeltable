"""Telegram notification UDF.

Uses the Telegram Bot API directly -- no SDK required.
Setup: message @BotFather on Telegram to create a bot and get a token.
"""

import requests

import pixeltable as pxt


@pxt.udf
def send_message(bot_token: str, chat_id: str, text: str) -> pxt.Json:
    """Send a message via Telegram Bot API.

    Example:
        >>> t.add_computed_column(
        ...     alert=telegram.send_message(BOT_TOKEN, CHAT_ID, t.summary)
        ... )
    """
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    resp = requests.post(url, json={'chat_id': chat_id, 'text': text}, timeout=10)
    return resp.json()
