import logging
import os
from datetime import datetime

import config
import discord
import setup_pixeltable  # noqa: F401 — initializes schema on import
from discord.ext import commands
from dotenv import load_dotenv
from message_formatter import MessageFormatter

import pixeltable as pxt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pixeltable-bot')


class PixelTableBot:
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='/', intents=intents)
        self.logger = logging.getLogger('pixeltable-bot')
        self.messages_table = None
        self.messages_view = None
        self.chat_table = None
        self.formatter = MessageFormatter()
        self.setup_bot_events()

    def initialize_pixeltable(self):
        """Load Pixeltable table handles (schema created by setup_pixeltable)."""
        try:
            self.messages_table = pxt.get_table(f'{config.APP_NAMESPACE}.messages')
            self.messages_view = pxt.get_table(f'{config.APP_NAMESPACE}.sentences')
            self.chat_table = pxt.get_table(f'{config.APP_NAMESPACE}.chat')
            self.logger.info('Successfully connected to Pixeltable tables')
        except Exception as e:
            self.logger.error(f'Failed to connect to Pixeltable: {e!s}')
            raise

    def setup_bot_events(self):
        """Set up Discord bot event handlers"""

        @self.bot.event
        async def on_ready():
            self.logger.info(f'Bot logged in as {self.bot.user.name}')
            self.initialize_pixeltable()

        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return

            await self.bot.process_commands(message)

            try:
                if message.content and self.messages_table:
                    self.messages_table.insert(
                        [
                            {
                                'channel_id': str(message.channel.id),
                                'username': message.author.name,
                                'content': message.content,
                                'timestamp': datetime.now(),
                            }
                        ]
                    )
            except Exception as e:
                self.logger.error(f'Failed to store message: {e!s}')

        @self.bot.command(name='search')
        async def search(ctx, *, query: str):
            """Search messages in the channel"""
            response_message = await ctx.send('Searching...')

            try:
                if not self.messages_view:
                    raise ValueError('Messages view not initialized')

                sim = self.messages_view.text.similarity(string=query)
                results_df = (
                    self.messages_view.order_by(sim, asc=False)
                    .select(text=self.messages_view.text, username=self.messages_view.username, similarity=sim)
                    .limit(5)
                    .collect()
                    .to_pandas()
                )

                if results_df.empty:
                    await response_message.edit(content='No matching messages found.')
                    return

                embed = self.formatter.create_search_embed(results_df.to_dict('records'), query)

                await response_message.edit(content=None, embed=embed)

            except Exception as e:
                self.logger.error(f'Search failed: {e!s}')
                error_embed = self.formatter.create_error_embed(str(e))
                await response_message.edit(content=None, embed=error_embed)

        @self.bot.command(name='chat')
        async def chat_command(ctx, *, question: str):
            """Chat with context from message history"""
            response_message = await ctx.send('Processing...')

            try:
                if not self.chat_table:
                    raise ValueError('Chat table not initialized')

                self.chat_table.insert(
                    [{'channel_id': str(ctx.channel.id), 'question': question, 'timestamp': datetime.now()}]
                )

                result = (
                    self.chat_table.select(self.chat_table.question, self.chat_table.response, self.chat_table.context)
                    .order_by(self.chat_table.timestamp, asc=False)
                    .limit(1)
                    .collect()
                )

                if len(result) == 0:
                    raise ValueError('Failed to generate response')

                embed = self.formatter.create_chat_embed(
                    question=question, response=result['response'][0], context=result['context'][0]
                )

                await response_message.edit(content=None, embed=embed)

            except Exception as e:
                self.logger.error(f'Chat failed: {e!s}')
                error_embed = self.formatter.create_error_embed(str(e))
                await response_message.edit(content=None, embed=error_embed)

    def run(self, token: str):
        """Run the Discord bot"""
        try:
            self.bot.run(token)
        except Exception as e:
            logger.error(f'Failed to start bot: {e!s}')
            raise


def main():
    load_dotenv()
    token = os.getenv('DISCORD_TOKEN')
    if token is None:
        raise ValueError('Missing DISCORD_TOKEN in environment')

    try:
        bot = PixelTableBot()
        bot.run(token)
    except Exception as e:
        logging.error(f'Bot execution failed: {e!s}')
        raise


if __name__ == '__main__':
    main()
