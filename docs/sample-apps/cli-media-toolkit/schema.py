#!/usr/bin/env python
"""Pixeltable Media Toolkit — schema initialization.

python schema.py                    # idempotent init
RESET_SCHEMA=true python schema.py  # wipe tables and recreate
"""

import os
from pathlib import Path

import click

import pixeltable as pxt
from pixeltable.functions.huggingface import clip
from pixeltable.functions.video import frame_iterator


def init_media_toolkit() -> None:
    """Initialize Pixeltable tables and indices."""
    click.echo('Initializing tables...')

    if os.getenv('RESET_SCHEMA', 'false').lower() == 'true':
        pxt.drop_table('ai_media_toolkit', force=True)
        pxt.drop_table('ai_media_frames', force=True)

    table = pxt.create_table(
        'ai_media_toolkit',
        {
            'source': pxt.String,
            'title': pxt.String,
            'media_type': pxt.String,
            'video': pxt.Video,
            'image': pxt.Image,
            'audio': pxt.Audio,
            'uploaded_at': pxt.Timestamp,
        },
        if_exists='ignore',
    )

    clip_model = clip.using(model_id='openai/clip-vit-base-patch32')
    table.add_embedding_index('image', idx_name='image_clip_idx', embedding=clip_model, if_exists='ignore')
    click.echo('Created image embedding index')

    frames_view = pxt.create_view(
        'ai_media_frames', table, iterator=frame_iterator(video=table.video, fps=1), if_exists='ignore'
    )
    frames_view.add_embedding_index('frame', idx_name='frame_clip_idx', embedding=clip_model, if_exists='ignore')
    click.echo('Created frame embedding index')

    Path('./outputs').mkdir(exist_ok=True)
    click.echo('Initialization complete.')
    click.echo('Usage: python cli.py add <source>')


if __name__ == '__main__':
    init_media_toolkit()
