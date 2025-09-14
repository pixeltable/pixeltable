#!/usr/bin/env python
"""
Pixeltable Media Toolkit - Initialization

Creates tables, embedding indices, and views.
Run once before using the CLI.
"""

import pixeltable as pxt
from pixeltable.functions.huggingface import clip
from pixeltable.iterators import FrameIterator
from pathlib import Path
import click


def init_media_toolkit():
    """Initialize Pixeltable tables and indices"""
    
    click.echo("Initializing tables...")
    
    try:
        # Create main media table
        table = pxt.create_table('ai_media_toolkit', {
            'source': pxt.String,
            'title': pxt.String,
            'media_type': pxt.String,
            'video': pxt.Video,
            'image': pxt.Image,
            'uploaded_at': pxt.Timestamp
        }, if_exists='ignore')
        
        # Add CLIP embedding index to image column
        clip_model = clip.using(model_id='openai/clip-vit-base-patch32')
        try:
            table.add_embedding_index('image', idx_name='image_clip_idx', embedding=clip_model)
            click.echo("Created image embedding index")
        except Exception as e:
            if "already exists" not in str(e):
                click.echo(f"Image index: {e}")
        
        # Create frames view with embedding index
        try:
            frames_view = pxt.create_view(
                'ai_media_frames',
                table,
                iterator=FrameIterator.create(
                    video=table.video,
                    fps=1  # Extract 1 frame per second
                ),
                if_exists='ignore'
            )
            
            # Add CLIP embedding index to frame column
            try:
                frames_view.add_embedding_index('frame', idx_name='frame_clip_idx', embedding=clip_model)
                click.echo("Created frame embedding index")
            except Exception as e:
                if "already exists" not in str(e):
                    click.echo(f"Frame index: {e}")
                    
        except Exception as e:
            if "already exists" not in str(e):
                click.echo(f"Frames view: {e}")
        
        # Create outputs directory
        Path('./outputs').mkdir(exist_ok=True)
        
        click.echo("Initialization complete.")
        click.echo("Usage: python cli.py add <source>")
        
    except Exception as e:
        click.echo(f"Initialization failed: {e}")


if __name__ == "__main__":
    init_media_toolkit()
