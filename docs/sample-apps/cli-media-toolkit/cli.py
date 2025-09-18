#!/usr/bin/env python
"""
Pixeltable Media Toolkit

Demonstrates native Pixeltable functions for multimodal data processing.
Run `python init.py` first to initialize tables and indices.
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import click
from pytubefix import YouTube

import pixeltable as pxt


def get_media_table():
    """Get media table"""
    try:
        return pxt.get_table('ai_media_toolkit')
    except Exception:
        click.echo("Error: Run 'python init.py' first to initialize tables.")
        raise click.Abort()


def get_frames_view():
    """Get frames view"""
    try:
        return pxt.get_table('ai_media_frames')
    except Exception:
        click.echo("Error: Run 'python init.py' first to initialize tables.")
        raise click.Abort()


# Helper functions
def ensure_outputs_dir():
    """Ensure outputs directory exists"""
    output_dir = Path('./outputs')
    output_dir.mkdir(exist_ok=True)
    return output_dir


def safe_filename(title):
    """Generate safe filename from title"""
    return ''.join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()


@click.group()
def cli():
    """Pixeltable Media Toolkit - Fast CLI"""
    pass


@cli.command()
@click.argument('source')
def add(source):
    """Add media from local file, URL, or YouTube"""

    try:
        table = get_media_table()

        # Determine media type and handle accordingly
        if 'youtube.com' in source or 'youtu.be' in source:
            # YouTube video - download to temporary file (following pattern from working-with-external-files.ipynb)
            yt = YouTube(source)
            video_stream = (
                yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            )

            if not video_stream:
                video_stream = yt.streams.filter(file_extension='mp4').first()

            # Create persistent temporary file path (not auto-cleaned up)
            video_path = tempfile.mktemp(suffix='.mp4')
            video_stream.download(filename=Path(video_path).name, output_path=str(Path(video_path).parent))

            table.insert(
                [
                    {
                        'source': source,
                        'title': yt.title,
                        'media_type': 'youtube',
                        'video': video_path,
                        'image': None,
                        'audio': None,
                        'uploaded_at': datetime.now(),
                    }
                ]
            )

            click.echo(f'Added: {yt.title}')

        elif source.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            # Image file or URL
            table.insert(
                [
                    {
                        'source': source,
                        'title': f'Image from {Path(source).name}',
                        'media_type': 'image',
                        'video': None,
                        'image': source,
                        'audio': None,
                        'uploaded_at': datetime.now(),
                    }
                ]
            )

            click.echo(f'Added: {Path(source).name}')

        elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            # Video file or URL
            table.insert(
                [
                    {
                        'source': source,
                        'title': f'Video from {Path(source).name}',
                        'media_type': 'video',
                        'video': source,
                        'image': None,
                        'audio': None,
                        'uploaded_at': datetime.now(),
                    }
                ]
            )

            click.echo(f'Added: {Path(source).name}')

        elif source.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma')):
            # Audio file or URL
            table.insert(
                [
                    {
                        'source': source,
                        'title': f'Audio from {Path(source).name}',
                        'media_type': 'audio',
                        'video': None,
                        'image': None,
                        'audio': source,
                        'uploaded_at': datetime.now(),
                    }
                ]
            )

            click.echo(f'Added: {Path(source).name}')

        else:
            click.echo(
                'Error: Unsupported media type. Use image (.jpg/.png), video (.mp4/.avi), audio (.mp3/.wav), or YouTube URL'
            )

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
def list():
    """List all stored media"""

    try:
        table = get_media_table()

        results = table.select(
            table.title,
            table.media_type,
            table.uploaded_at,
            video_duration=table.video.get_duration() if table.video is not None else None,
            audio_metadata=table.audio.get_metadata() if table.audio is not None else None,
        ).collect()

        if not results:
            click.echo("No media stored yet. Use 'python cli.py add <source>' to add media.")
            return

        click.echo(f'{len(results)} items:')
        for i, item in enumerate(results, 1):
            duration = ''
            if item['video_duration']:
                duration = f' ({item["video_duration"]:.1f}s)'
            elif item['audio_metadata']:
                # Extract duration from audio metadata
                try:
                    audio_meta = item['audio_metadata']
                    if 'streams' in audio_meta and audio_meta['streams']:
                        stream = audio_meta['streams'][0]
                        if 'duration_seconds' in stream:
                            duration = f' ({stream["duration_seconds"]:.1f}s)'
                except (KeyError, TypeError, IndexError):
                    pass  # Skip duration if metadata is malformed

            click.echo(f'  {i}. {item["title"]} ({item["media_type"]}){duration}')

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=5, help='Number of results')
@click.option('--threshold', '-t', default=0.3, help='Similarity threshold (0.0-1.0)')
@click.option('--save', '-s', is_flag=True, help='Save results to outputs/ directory')
def search_images(query, limit, threshold, save):
    """Search stored images using CLIP similarity"""

    try:
        table = get_media_table()

        # Check if query is an image file path
        if Path(query).exists():
            query_input = query
            query_type = 'image'
        else:
            query_input = query
            query_type = 'text'

        # CLIP similarity search
        sim = table.image.similarity(query_input, idx='image_clip_idx')
        results = (
            table.where((table.image != None) & (sim > threshold))
            .order_by(sim, asc=False)
            .select(table.title, table.media_type, similarity=sim)
            .limit(limit)
            .collect()
        )

        click.echo(f'Found {len(results)} similar images:')
        for i, item in enumerate(results, 1):
            click.echo(f'  {i}. {item["title"]} ({item["media_type"]}) - {item["similarity"]:.3f}')

        # Save results if requested
        if save:
            output_dir = ensure_outputs_dir()
            search_file = output_dir / f'image_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

            with open(search_file, 'w') as f:
                json.dump(
                    {
                        'query': query,
                        'query_type': query_type,
                        'timestamp': datetime.now().isoformat(),
                        'threshold': threshold,
                        'results': [
                            {'title': item['title'], 'media_type': item['media_type'], 'similarity': item['similarity']}
                            for item in results
                        ],
                    },
                    f,
                    indent=2,
                )

            click.echo(f'Results saved: {search_file.name}')

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=5, help='Number of results')
@click.option('--threshold', '-t', default=0.3, help='Similarity threshold (0.0-1.0)')
@click.option('--save', '-s', is_flag=True, help='Save results to outputs/ directory')
def search_frames(query, limit, threshold, save):
    """Search video frames using CLIP similarity"""

    try:
        frames_view = get_frames_view()

        # Check if query is an image file path
        if Path(query).exists():
            query_input = query
            query_type = 'image'
        else:
            query_input = query
            query_type = 'text'

        # CLIP similarity search on frames
        sim = frames_view.frame.similarity(query_input, idx='frame_clip_idx')
        results = (
            frames_view.where(sim > threshold)
            .order_by(sim, asc=False)
            .select(frames_view.title, frames_view.pos, similarity=sim)
            .limit(limit)
            .collect()
        )

        click.echo(f'Found {len(results)} similar frames:')
        for i, item in enumerate(results, 1):
            click.echo(f'  {i}. {item["title"]} at {item["pos"]:.1f}s - {item["similarity"]:.3f}')

        # Save results if requested
        if save:
            output_dir = ensure_outputs_dir()
            search_file = output_dir / f'frame_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

            with open(search_file, 'w') as f:
                json.dump(
                    {
                        'query': query,
                        'query_type': query_type,
                        'timestamp': datetime.now().isoformat(),
                        'threshold': threshold,
                        'results': [
                            {'title': item['title'], 'position': item['pos'], 'similarity': item['similarity']}
                            for item in results
                        ],
                    },
                    f,
                    indent=2,
                )

            click.echo(f'Results saved: {search_file.name}')

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
@click.option('--model', '-m', default='base', help='Whisper model to use (tiny, base, small, medium, large)')
def transcribe(model):
    """Transcribe audio from stored media using local Whisper"""

    try:
        from pixeltable.functions.whisper import transcribe as whisper_transcribe

        table = get_media_table()
        output_dir = ensure_outputs_dir()

        # Process audio and video files separately
        results = []

        # Process audio files
        try:
            audio_results = (
                table.where(table.audio != None)
                .select(table.title, table.media_type, transcription=whisper_transcribe(table.audio, model=model))
                .collect()
            )
            results.extend(audio_results)
        except Exception as e:
            click.echo(f'Warning: Error processing audio files: {e}')

        # Process video files (extract audio first)
        try:
            video_results = (
                table.where(table.video != None)
                .select(
                    table.title,
                    table.media_type,
                    transcription=whisper_transcribe(table.video.extract_audio(format='wav'), model=model),
                )
                .collect()
            )
            results.extend(video_results)
        except Exception as e:
            click.echo(f'Warning: Error processing video files: {e}')

        if not results:
            click.echo('No audio or video files found for transcription')
            return

        # Save transcripts to outputs
        for item in results:
            if item['transcription'] is None or 'text' not in item['transcription']:
                click.echo(f'Skipped: {item["title"]} (no audio content or transcription failed)')
                continue

            transcript_text = item['transcription']['text']

            # Save transcript to file
            transcript_file = output_dir / f'{safe_filename(item["title"])}_transcript.txt'
            with open(transcript_file, 'w') as f:
                f.write(transcript_text)

            # Save full result with metadata as JSON
            json_file = output_dir / f'{safe_filename(item["title"])}_transcript_full.json'
            with open(json_file, 'w') as f:
                json.dump(item['transcription'], f, indent=2)

            word_count = len(transcript_text.split())
            click.echo(f'Transcribed: {item["title"]}')
            click.echo(f'  Saved: {transcript_file.name} ({word_count} words)')
            click.echo(f'  Full result: {json_file.name}')

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
def analyze():
    """Analyze stored media metadata"""

    try:
        table = get_media_table()

        # Get basic metadata for all media types
        results = table.select(
            table.title,
            table.media_type,
            video_duration=table.video.get_duration() if table.video is not None else None,
            video_metadata=table.video.get_metadata() if table.video is not None else None,
            audio_metadata=table.audio.get_metadata() if table.audio is not None else None,
        ).collect()

        if not results:
            click.echo('No media to analyze')
            return

        # Create analysis report
        output_dir = ensure_outputs_dir()

        report_lines = [f'Media Analysis Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', '=' * 60, '', '']

        for item in results:
            click.echo(f'{item["title"]} ({item["media_type"]}):')
            report_lines.append(f'{item["title"]} ({item["media_type"]}):')

            # Video analysis
            if item['video_duration']:
                duration_text = f'  Duration: {item["video_duration"]:.1f}s'
                click.echo(duration_text)
                report_lines.append(duration_text)

                if item['video_metadata'] and 'streams' in item['video_metadata']:
                    stream = item['video_metadata']['streams'][0]
                    if 'width' in stream and 'height' in stream:
                        resolution = f'  Resolution: {stream["width"]}x{stream["height"]}'
                        click.echo(resolution)
                        report_lines.append(resolution)

            # Audio analysis
            if item['audio_metadata']:
                audio_meta = item['audio_metadata']
                if 'streams' in audio_meta and audio_meta['streams']:
                    audio_stream = audio_meta['streams'][0]
                    if 'duration_seconds' in audio_stream:
                        duration_text = f'  Audio Duration: {audio_stream["duration_seconds"]:.1f}s'
                        click.echo(duration_text)
                        report_lines.append(duration_text)

                    if 'codec_context' in audio_stream:
                        codec = audio_stream['codec_context']
                        if 'name' in codec:
                            codec_text = f'  Audio Codec: {codec["name"]}'
                            if 'channels' in codec:
                                codec_text += f' ({codec["channels"]} channels)'
                            click.echo(codec_text)
                            report_lines.append(codec_text)

            click.echo('')
            report_lines.append('')

        # Save analysis report
        report_file = output_dir / f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        click.echo(f'Analysis saved: {report_file.name}')

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
@click.option('--timestamp', '-t', default=1.0, help='Timestamp in seconds')
def extract_thumbnail(timestamp):
    """Extract thumbnail from stored videos"""

    try:
        table = get_media_table()

        # Query: Extract frame on demand and save to outputs
        thumbnail_results = (
            table.where(table.video != None)
            .select(table.title, thumbnail=table.video.extract_frame(timestamp=timestamp))
            .collect()
        )

        if not thumbnail_results:
            click.echo('No videos found for thumbnail extraction')
            return

        # Save thumbnails to outputs
        output_dir = ensure_outputs_dir()

        for video in thumbnail_results:
            if video['thumbnail']:
                # Save thumbnail to outputs
                thumbnail_file = output_dir / f'{safe_filename(video["title"])}_thumb_{timestamp}s.jpg'
                video['thumbnail'].save(thumbnail_file)

                file_size = thumbnail_file.stat().st_size / 1024  # KB
                click.echo(f'Extracted: {video["title"]}')
                click.echo(f'Saved: {thumbnail_file.name} ({file_size:.1f} KB)')

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
def extract_audio():
    """Extract audio from stored videos"""

    try:
        table = get_media_table()

        results = (
            table.where(table.video != None)
            .select(table.title, audio_path=table.video.extract_audio(format='wav'))
            .collect()
        )

        if not results:
            click.echo('No videos found for audio extraction')
            return

        # Copy audio files to outputs
        output_dir = ensure_outputs_dir()

        for video in results:
            if video['audio_path']:
                output_file = output_dir / f'{safe_filename(video["title"])}_audio.wav'
                shutil.copy2(video['audio_path'], output_file)

                size_mb = output_file.stat().st_size / (1024 * 1024)
                click.echo(f'Extracted: {video["title"]}')
                click.echo(f'Saved: {output_file.name} ({size_mb:.1f} MB)')

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
@click.option('--start', '-s', default=0.0, help='Start time in seconds')
@click.option('--end', '-e', default=10.0, help='End time in seconds')
def generate_short(start, end):
    """Generate short video clips (9:16 vertical)"""

    try:
        table = get_media_table()

        # Query: Create vertical clips on demand
        results = (
            table.where(table.video != None)
            .select(table.title, clip=table.video.clip(start_time=start, end_time=end))
            .collect()
        )

        if not results:
            click.echo('No videos found for short generation')
            return

        # Save clips to outputs
        output_dir = ensure_outputs_dir()

        for video in results:
            if video['clip']:
                clip_file = output_dir / f'{safe_filename(video["title"])}_clip_{start}_{end}s.mp4'
                shutil.copy2(video['clip'], clip_file)

                file_size = clip_file.stat().st_size / (1024 * 1024)  # MB
                click.echo(f'Generated {end - start}s clip: {video["title"]}')
                click.echo(f'Saved: {clip_file.name} ({file_size:.1f} MB)')

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
@click.option('--segment-length', '-l', default=10.0, help='Length of each segment in seconds')
def split_video(segment_length):
    """Split videos into segments"""

    try:
        table = get_media_table()

        # Query: Split video into segments on demand
        results = (
            table.where(table.video != None)
            .select(
                table.title,
                duration=table.video.get_duration(),
                segments=table.video.segment_video(duration=segment_length),
            )
            .collect()
        )

        if not results:
            click.echo('No videos found for splitting')
            return

        # Save segments to outputs
        output_dir = ensure_outputs_dir()

        for video in results:
            if video['segments']:
                num_segments = len(video['segments'])
                total_size = 0

                for i, segment in enumerate(video['segments']):
                    segment_file = output_dir / f'segment_{i + 1}.mp4'
                    shutil.copy2(segment, segment_file)
                    total_size += segment_file.stat().st_size

                total_mb = total_size / (1024 * 1024)
                click.echo(f"Split '{video['title']}' into {num_segments} segments")
                click.echo(f'Saved: segment_1.mp4 to segment_{num_segments}.mp4 ({total_mb:.1f} MB total)')

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
@click.argument('target', required=False)
@click.option('--title', help='Delete by title')
@click.option('--type', help='Delete by media type (video, image, youtube)')
@click.option('--all', 'delete_all', is_flag=True, help='Delete all items')
@click.option('--force', is_flag=True, help='Skip confirmation')
def delete(target, title, type, delete_all, force):
    """Delete media items"""

    try:
        table = get_media_table()

        if delete_all:
            if not force and not click.confirm('Delete ALL media items?'):
                return
            result = table.delete()
            deleted_count = result.row_count_stats.del_rows
            click.echo(f'Deleted {deleted_count} items')

        elif type:
            if not force and not click.confirm(f'Delete all {type} items?'):
                return
            result = table.where(table.media_type == type).delete()
            deleted_count = result.row_count_stats.del_rows
            click.echo(f'Deleted {deleted_count} {type} items')

        elif title:
            if not force and not click.confirm(f'Delete items with title "{title}"?'):
                return
            result = table.where(table.title.contains(title)).delete()
            deleted_count = result.row_count_stats.del_rows
            click.echo(f"Deleted {deleted_count} items matching '{title}'")

        elif target and target.isdigit():
            item_num = int(target)
            results = table.select(table.title).collect()

            if 1 <= item_num <= len(results):
                item_title = results[item_num - 1]['title']

                if not force and not click.confirm(f'Delete "{item_title}"?'):
                    return

                result = table.where(table.title == item_title).delete()
                click.echo(f"Deleted '{item_title}'")
            else:
                click.echo(f"Invalid item number. Use 'list' to see items.")
        else:
            click.echo("Usage: delete <number> | --title 'name' | --type video | --all")

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
def status():
    """Show toolkit status"""

    try:
        table = get_media_table()
        frames_view = get_frames_view()

        # Count items efficiently
        media_count = table.count()
        frames_count = frames_view.count()
        audio_count = table.where(table.audio != None).count()
        video_count = table.where(table.video != None).count()
        image_count = table.where(table.image != None).count()

        click.echo('Status:')
        click.echo(f'  Total media items: {media_count}')
        click.echo(f'  - Videos: {video_count}')
        click.echo(f'  - Images: {image_count}')
        click.echo(f'  - Audio files: {audio_count}')
        click.echo(f'  Video frames indexed: {frames_count}')

        # Check outputs
        output_dir = Path('./outputs')
        if output_dir.exists():
            click.echo(f'  Output files: {len([f for f in output_dir.iterdir() if f.is_file()])}')

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
def reset():
    """Reset all data (clear tables)"""

    try:
        if not click.confirm('This will delete ALL stored media and indices. Continue?'):
            return

        # Clean reset
        pxt.drop_table('ai_media_toolkit', force=True)
        pxt.drop_table('ai_media_frames', force=True)

        click.echo("Reset complete. Run 'python init.py' to reinitialize.")

    except Exception as e:
        click.echo(f'Error: {e}', err=True)


@cli.command()
def functions():
    """List native Pixeltable functions"""

    click.echo('Video functions:')
    click.echo('  video.get_duration()')
    click.echo('  video.get_metadata()')
    click.echo('  video.extract_audio(format)')
    click.echo('  video.extract_frame(timestamp)')
    click.echo('  video.clip(start_time, end_time)')
    click.echo('  video.segment_video(duration)')
    click.echo('')
    click.echo('Audio functions:')
    click.echo('  audio.get_metadata()')
    click.echo('  whisper.transcribe(model)')
    click.echo('')
    click.echo('Image functions:')
    click.echo('  image.crop([x1,y1,x2,y2])')
    click.echo('  image.resize([w,h])')
    click.echo('  image.rotate(angle)')
    click.echo('  image.b64_encode(format)')
    click.echo('')
    click.echo('CLI commands:')
    click.echo('  add <source>              # Add media (video/image/audio)')
    click.echo('  list                      # List stored media')
    click.echo('  transcribe --model <name> # Transcribe audio/video → outputs/*.txt')
    click.echo("  search-images 'query'     # CLIP similarity search")
    click.echo("  search-frames 'query'     # Search video frames")
    click.echo('  extract-audio             # Extract audio tracks')
    click.echo('  extract-thumbnail         # Extract video frames')
    click.echo('  generate-short            # Create vertical clips')
    click.echo('  analyze                   # Extract metadata')
    click.echo('')
    click.echo('All outputs saved to ./outputs/')


if __name__ == '__main__':
    cli()
