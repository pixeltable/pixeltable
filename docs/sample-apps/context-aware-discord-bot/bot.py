"""Context-Aware Discord Bot powered by Pixeltable.

Run:
    python setup_pixeltable.py   # one-time schema init
    python bot.py                # start the bot
"""

import logging
from datetime import datetime, timezone

import discord
from discord.ext import commands

import pixeltable as pxt

import config
from message_formatter import MessageFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pixeltable-bot')

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)
formatter = MessageFormatter()

messages = pxt.get_table(f'{config.APP_NAMESPACE}.messages')
sentences = pxt.get_table(f'{config.APP_NAMESPACE}.sentences')
chat = pxt.get_table(f'{config.APP_NAMESPACE}.chat')


@bot.event
async def on_ready():
    logger.info(f'Bot logged in as {bot.user.name}')


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)
    try:
        if message.content:
            messages.insert([{
                'channel_id': str(message.channel.id),
                'username': message.author.name,
                'content': message.content,
                'timestamp': datetime.now(tz=timezone.utc),
            }])
    except Exception as e:
        logger.error(f'Failed to store message: {e}')


@bot.command(name='search')
async def search(ctx, *, query: str):
    """Search messages by semantic similarity."""
    response_message = await ctx.send('Searching...')
    try:
        sim = sentences.text.similarity(query)
        results_df = (
            sentences
            .order_by(sim, asc=False)
            .select(text=sentences.text, username=sentences.username, similarity=sim)
            .limit(5)
            .collect()
            .to_pandas()
        )
        if results_df.empty:
            await response_message.edit(content='No matching messages found.')
            return
        embed = formatter.create_search_embed(results_df.to_dict('records'), query)
        await response_message.edit(content=None, embed=embed)
    except Exception as e:
        logger.error(f'Search failed: {e}')
        await response_message.edit(content=None, embed=formatter.create_error_embed(str(e)))


@bot.command(name='chat')
async def chat_command(ctx, *, question: str):
    """Chat with context from message history."""
    response_message = await ctx.send('Processing...')
    try:
        chat.insert([{
            'channel_id': str(ctx.channel.id),
            'question': question,
            'timestamp': datetime.now(tz=timezone.utc),
        }])
        result = (
            chat.select(chat.question, chat.response, chat.context)
            .order_by(chat.timestamp, asc=False)
            .limit(1)
            .collect()
        )
        if len(result) == 0:
            raise ValueError('Failed to generate response')
        embed = formatter.create_chat_embed(
            question=question,
            response=result['response'][0],
            context=result['context'][0],
        )
        await response_message.edit(content=None, embed=embed)
    except Exception as e:
        logger.error(f'Chat failed: {e}')
        await response_message.edit(content=None, embed=formatter.create_error_embed(str(e)))


if __name__ == '__main__':
    if not config.DISCORD_TOKEN:
        raise ValueError('DISCORD_TOKEN not set in .env')
    bot.run(config.DISCORD_TOKEN)
