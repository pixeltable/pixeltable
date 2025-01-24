import discord
from datetime import datetime
from typing import List, Dict


class MessageFormatter:
    """Helper class for consistent Discord message formatting"""

    # Discord color constants
    BLUE = discord.Color.blue()
    GREEN = discord.Color.green()
    RED = discord.Color.red()

    @staticmethod
    def create_search_embed(results: list[dict], query: str) -> discord.Embed:
        """Create formatted embed for search results"""
        embed = discord.Embed(
            title=f'🔍 Search Results for: {query}', color=MessageFormatter.BLUE, timestamp=datetime.utcnow()
        )

        for i, result in enumerate(results, 1):
            # Format relevance score
            score = round(float(result['similarity']) * 100, 1)

            # Add formatted field
            embed.add_field(
                name=f'Result {i} ({score}% match)',
                value=f"From: {result['username']}\n```{result['text'][:200]}```",
                inline=False,
            )

        embed.set_footer(text=f'Found {len(results)} results')
        return embed

    @staticmethod
    def create_chat_embed(question: str, response: str, context: list[dict]) -> discord.Embed:
        """Create formatted embed for chat responses with context"""
        embed = discord.Embed(
            title='💬 Chat Response', description=response, color=MessageFormatter.GREEN, timestamp=datetime.utcnow()
        )

        # Add question field
        embed.add_field(name='❓ Question', value=question, inline=False)

        # Add relevant context field
        if context:
            context_text = []
            for i, ctx in enumerate(context, 1):
                score = round(float(ctx['sim']) * 100, 1)
                context_text.append(f"{i}. {ctx['username']} ({score}% relevance)\n" f"```{ctx['text'][:150]}```")

            embed.add_field(
                name='🧠 Relevant Context',
                value='\n'.join(context_text) if context_text else 'No relevant context found',
                inline=False,
            )

        return embed

    @staticmethod
    def create_error_embed(error: str) -> discord.Embed:
        """Create formatted embed for errors"""
        embed = discord.Embed(
            title='Error Occurred',
            description=f'```{error}```',
            color=MessageFormatter.RED,
            timestamp=datetime.utcnow(),
        )
        embed.set_footer(text='Please try again or contact support if the issue persists')
        return embed
