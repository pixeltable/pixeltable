import logging
import os
from datetime import datetime

import discord
import numpy as np
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

from message_formatter import MessageFormatter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pixeltable-bot')


class PixelTableBot:
    def __init__(self):
        # Initialize Discord bot
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='/', intents=intents)
        self.logger = logging.getLogger('pixeltable-bot')
        self.messages_table = None
        self.messages_view = None
        self.chat_table = None
        self.formatter = MessageFormatter()
        self.setup_bot_events()

    @staticmethod
    @pxt.expr_udf
    def get_embeddings(text: str) -> np.ndarray:
        """Generate embeddings using sentence transformer"""
        return sentence_transformer(text, model_id='intfloat/e5-large-v2')

    def initialize_pixeltable(self):
        """Initialize Pixeltable directory and tables"""
        try:
            pxt.drop_dir('discord_bot', force=True)
            pxt.create_dir('discord_bot')

            # Create messages table
            self.messages_table = pxt.create_table(
                'discord_bot.messages',
                {'channel_id': pxt.String, 'username': pxt.String, 'content': pxt.String, 'timestamp': pxt.Timestamp},
            )

            # Create sentence-level view
            self.messages_view = pxt.create_view(
                'discord_bot.sentences',
                self.messages_table,
                iterator=StringSplitter.create(text=self.messages_table.content, separators='sentence'),
            )

            # Add embedding index to the view
            self.messages_view.add_embedding_index('text', string_embed=self.get_embeddings)

            # Create chat table
            self.chat_table = pxt.create_table(
                'discord_bot.chat', {'channel_id': pxt.String, 'question': pxt.String, 'timestamp': pxt.Timestamp}
            )

            self.setup_chat_columns()
            self.logger.info('Successfully initialized Pixeltable tables and views')

        except Exception as e:
            self.logger.error(f'Failed to initialize Pixeltable: {str(e)}')
            raise

    def setup_chat_columns(self):
        """Set up computed columns for the chat table"""
        try:
            # Add context column using sentence-level view
            @self.messages_view.query
            def get_context(question_text: str):
                sim = self.messages_view.text.similarity(question_text)
                return (
                    self.messages_view.order_by(sim, asc=False)
                    .select(text=self.messages_view.text, username=self.messages_view.username, sim=sim)
                    .limit(10)
                )

            self.chat_table.add_computed_column(context=get_context(self.chat_table.question))

            # Add prompt column
            @pxt.udf
            def create_prompt(context: list[dict], question: str) -> str:
                context_str = '\n'.join(f"{msg['username']}: {msg['text']}" for msg in context if msg['sim'] > 0.3)
                return f'Context:\n{context_str}\n\nQuestion: {question}'

            self.chat_table.add_computed_column(prompt=create_prompt(self.chat_table.context, self.chat_table.question))

            system_prompt = """You are a helpful personal assistant focused on natural conversation.
                CORE PRINCIPLES:
                - Maintain conversational context
                - Remember user preferences and details
                - Progress discussions naturally
                - Be specific and actionable
                - Stay on topic unless user changes it

                CONVERSATION STYLE:
                - Friendly and engaging
                - Clear and concise
                - Naturally incorporate context
                - Ask relevant follow-up questions
                - Provide practical suggestions"""

            # Add response column
            self.chat_table.add_computed_column(
                response=openai.chat_completions(
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': self.chat_table.prompt},
                    ],
                    model='gpt-4o-mini',
                    temperature=0.7,
                    max_tokens=2000,
                    presence_penalty=0.7,
                    frequency_penalty=0.5,
                )
                .choices[0]
                .message.content
            )

        except Exception as e:
            self.logger.error(f'Failed to set up chat columns: {str(e)}')
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
                self.logger.error(f'Failed to store message: {str(e)}')

        @self.bot.command(name='search')
        async def search(ctx, *, query: str):
            """Search messages in the channel"""
            response_message = await ctx.send('Searching...')

            try:
                if not self.messages_view:
                    raise ValueError('Messages view not initialized')

                sim = self.messages_view.text.similarity(query)
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
                self.logger.error(f'Search failed: {str(e)}')
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
                self.logger.error(f'Chat failed: {str(e)}')
                error_embed = self.formatter.create_error_embed(str(e))
                await response_message.edit(content=None, embed=error_embed)

    def run(self, token: str):
        """Run the Discord bot"""
        try:
            self.bot.run(token)
        except Exception as e:
            logger.error(f'Failed to start bot: {str(e)}')
            raise


def main():
    load_dotenv()
    token = os.getenv('DISCORD_TOKEN')
    if token is None:
        raise ValueError('Missing key in environment')

    try:
        bot = PixelTableBot()
        bot.run(token)
    except Exception as e:
        logging.error(f'Bot execution failed: {str(e)}')
        raise


if __name__ == '__main__':
    main()
