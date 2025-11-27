"""
End-to-End Pixeltable Demo with Logfire Telemetry

This demo showcases real Pixeltable workflows with full observability:
- Table creation and schema management
- Data import from CSV
- Computed columns with expressions
- Queries with filters, ordering, and aggregations
- Updates and deletes
- Video frame extraction
- Object detection with DETR
- Embedding indexes and similarity search

All operations are traced and visible in Logfire.

Usage:
    python examples/telemetry_full_demo.py
"""

from __future__ import annotations

import os


def setup_logfire() -> None:
    """Configure Logfire for observability."""
    import logfire

    logfire.configure(
        service_name='pixeltable-full-demo',
        send_to_logfire=True,
        inspect_arguments=False,  # Suppress introspection warnings
    )
    os.environ['PIXELTABLE_TELEMETRY_ENABLED'] = 'true'
    print('✓ Logfire configured')
    print('  Dashboard: https://logfire-us.pydantic.dev/pjlbrunelle/pierre')
    print()


def run_films_demo() -> None:
    """Demo: Basic table operations with a films dataset."""
    import logfire
    import pixeltable as pxt

    with logfire.span('films_demo', _span_name='📽️ Films Database Demo'):
        # Setup
        logfire.info('Setting up films demo')
        pxt.drop_dir('telemetry_demo', force=True, if_not_exists='ignore')
        pxt.create_dir('telemetry_demo')

        # Create table
        with logfire.span('create_films_table'):
            films = pxt.create_table(
                'telemetry_demo.films',
                {'title': pxt.String, 'year': pxt.Int, 'revenue_millions': pxt.Float, 'genre': pxt.String},
            )
            logfire.info('Created films table')

        # Insert data
        with logfire.span('insert_films_data'):
            films.insert(
                [
                    {'title': 'Jurassic Park', 'year': 1993, 'revenue_millions': 1037.5, 'genre': 'Sci-Fi'},
                    {'title': 'Titanic', 'year': 1997, 'revenue_millions': 2257.8, 'genre': 'Drama'},
                    {'title': 'Avengers: Endgame', 'year': 2019, 'revenue_millions': 2797.5, 'genre': 'Action'},
                    {'title': 'The Lion King', 'year': 1994, 'revenue_millions': 1656.9, 'genre': 'Animation'},
                    {'title': 'Frozen II', 'year': 2019, 'revenue_millions': 1450.0, 'genre': 'Animation'},
                    {'title': 'Top Gun: Maverick', 'year': 2022, 'revenue_millions': 1493.0, 'genre': 'Action'},
                    {'title': 'Barbie', 'year': 2023, 'revenue_millions': 1441.8, 'genre': 'Comedy'},
                    {'title': 'Oppenheimer', 'year': 2023, 'revenue_millions': 952.0, 'genre': 'Drama'},
                    {'title': 'Inside Out 2', 'year': 2024, 'revenue_millions': 1462.7, 'genre': 'Animation'},
                    {'title': 'Deadpool & Wolverine', 'year': 2024, 'revenue_millions': 1338.0, 'genre': 'Action'},
                ]
            )
            logfire.info('Inserted 10 films')

        # Add computed column
        with logfire.span('add_computed_column'):
            films.add_computed_column(revenue_billions=films.revenue_millions / 1000.0)
            logfire.info('Added computed column: revenue_billions')

        # Query: High-grossing films
        with logfire.span('query_blockbusters'):
            blockbusters = films.where(films.revenue_millions > 1500).collect()
            logfire.info(f'Found {len(blockbusters)} blockbusters (>$1.5B)')

        # Query: Films by genre
        with logfire.span('query_by_genre'):
            action_films = (
                films.where(films.genre == 'Action')
                .select(films.title, films.year, films.revenue_millions)
                .order_by(films.revenue_millions, asc=False)
                .collect()
            )
            logfire.info(f'Found {len(action_films)} action films')

        # Query: Count recent films
        with logfire.span('query_recent_films'):
            recent = films.where(films.year >= 2020).count()
            logfire.info(f'Films since 2020: {recent}')

        # Update: Adjust revenue
        with logfire.span('update_films'):
            films.update({'revenue_millions': films.revenue_millions * 1.05}, where=films.year == 2024)
            logfire.info('Updated 2024 films with 5% revenue adjustment')

        # Delete: Remove older films
        with logfire.span('delete_old_films'):
            films.delete(where=films.year < 1995)
            logfire.info('Deleted films before 1995')

        # Final count
        with logfire.span('final_count'):
            final_count = films.count()
            logfire.info(f'Final film count: {final_count}')


def run_video_detection_demo() -> None:
    """Demo: Video frame extraction and object detection."""
    import logfire
    import pixeltable as pxt
    from pixeltable.iterators import FrameIterator
    from pixeltable.functions.huggingface import detr_for_object_detection

    with logfire.span('video_detection_demo', _span_name='🎬 Video Detection Demo'):
        logfire.info('Setting up video processing pipeline')

        # Create videos table
        with logfire.span('create_videos_table'):
            videos = pxt.create_table('telemetry_demo.videos', {'video': pxt.Video, 'description': pxt.String})
            logfire.info('Created videos table')

        # Insert sample video
        with logfire.span('insert_video'):
            videos.insert(
                [
                    {
                        'video': 'https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources/bangkok.mp4',
                        'description': 'Bangkok street scene',
                    }
                ]
            )
            logfire.info('Inserted 1 video')

        # Create frame extraction view
        with logfire.span('create_frames_view'):
            frames = pxt.create_view(
                'telemetry_demo.frames',
                videos,
                iterator=FrameIterator.create(video=videos.video, fps=1),  # 1 frame per second
            )
            frame_count = frames.count()
            logfire.info(f'Extracted {frame_count} frames from video')

        # Add object detection
        with logfire.span('add_object_detection'):
            frames.add_computed_column(
                detections=detr_for_object_detection(frames.frame, model_id='facebook/detr-resnet-50', threshold=0.7)
            )
            logfire.info('Added DETR object detection to frames')

        # Query: Frames with detections
        with logfire.span('query_detections'):
            results = frames.select(frames.frame, frames.pos, frames.detections).limit(5).collect()
            total_detections = sum(len(r['detections'].get('labels', [])) for r in results if r['detections'])
            logfire.info(f'Found {total_detections} objects in first 5 frames')


def run_similarity_search_demo() -> None:
    """Demo: Embedding index and similarity search."""
    import logfire
    import pixeltable as pxt
    from pixeltable.functions.huggingface import clip

    with logfire.span('similarity_search_demo', _span_name='🔍 Similarity Search Demo'):
        logfire.info('Setting up image similarity search')

        # Create images table
        with logfire.span('create_images_table'):
            images = pxt.create_table('telemetry_demo.images', {'image': pxt.Image, 'category': pxt.String})
            logfire.info('Created images table')

        # Insert sample images (using Pixeltable's own test images)
        with logfire.span('insert_images'):
            base_url = 'https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources/images'
            image_data = [
                (f'{base_url}/000000000139.jpg', 'food'),
                (f'{base_url}/000000000285.jpg', 'animal'),
                (f'{base_url}/000000000632.jpg', 'person'),
                (f'{base_url}/000000000724.jpg', 'animal'),
                (f'{base_url}/000000000776.jpg', 'food'),
            ]
            images.insert([{'image': url, 'category': cat} for url, cat in image_data])
            logfire.info(f'Inserted {len(image_data)} images')

        # Create embedding index
        with logfire.span('create_embedding_index'):
            embedding_fn = clip.using(model_id='openai/clip-vit-base-patch32')
            images.add_embedding_index('image', embedding=embedding_fn, metric='cosine')
            logfire.info('Created CLIP embedding index on images')

        # Similarity search by text
        with logfire.span('text_similarity_search'):
            sim = images.image.similarity('delicious food on a plate')
            results = images.select(images.image, images.category, sim).order_by(sim, asc=False).limit(3).collect()
            top_categories = [r['category'] for r in results]
            logfire.info(f"Text search 'food': top results = {top_categories}")

        # Similarity search by image
        with logfire.span('image_similarity_search'):
            # Use first image as query
            sample = images.select(images.image).limit(1).collect()
            if sample:
                sample_img = sample[0]['image']
                sim = images.image.similarity(sample_img)
                results = images.select(images.category, sim).order_by(sim, asc=False).limit(3).collect()
                logfire.info(f'Image similarity search completed, top match: {results[0]["category"]}')


def run_analytics_demo() -> None:
    """Demo: Advanced analytics queries."""
    import logfire
    import pixeltable as pxt

    with logfire.span('analytics_demo', _span_name='📊 Analytics Demo'):
        logfire.info('Running advanced analytics queries')

        # Get films table
        films = pxt.get_table('telemetry_demo.films')

        # Add expression-based category column
        with logfire.span('add_category_column'):
            films.add_computed_column(is_blockbuster=(films.revenue_millions > 1000), if_exists='replace')
            logfire.info('Added is_blockbuster computed column')

        # Query: Blockbuster statistics
        with logfire.span('query_blockbuster_stats'):
            blockbusters = films.where(films.is_blockbuster == True).collect()
            logfire.info(f'Total blockbusters: {len(blockbusters)}')

        # Query: Films by year range
        with logfire.span('query_by_decade'):
            films_2020s = (
                films.where((films.year >= 2020) & (films.year < 2030)).select(films.title, films.year).collect()
            )
            logfire.info(f'Films from 2020s: {len(films_2020s)}')

        # Complex query: Genre + revenue filter
        with logfire.span('complex_filter_query'):
            action_hits = (
                films.where((films.genre == 'Action') & (films.revenue_millions > 1300))
                .select(films.title, films.revenue_millions)
                .collect()
            )
            logfire.info(f'Action films >$1.3B: {len(action_hits)}')


def cleanup() -> None:
    """Clean up demo data."""
    import logfire
    import pixeltable as pxt

    with logfire.span('cleanup', _span_name='🧹 Cleanup'):
        pxt.drop_dir('telemetry_demo', force=True, if_not_exists='ignore')
        logfire.info('Cleaned up demo directory')


def main() -> None:
    """Run the full demo."""
    import logfire

    print('=' * 70)
    print('Pixeltable End-to-End Demo with Logfire Telemetry')
    print('=' * 70)
    print()

    setup_logfire()

    import pixeltable as pxt

    pxt.init()

    with logfire.span('pixeltable_full_demo', _span_name='🚀 Pixeltable Full Demo'):
        logfire.info('Starting end-to-end Pixeltable demo')

        run_films_demo()
        run_video_detection_demo()
        run_similarity_search_demo()
        run_analytics_demo()
        cleanup()

        logfire.info('Demo completed successfully!')

    print()
    print('=' * 70)
    print('✓ Demo complete!')
    print()
    print('View traces at: https://logfire-us.pydantic.dev/pjlbrunelle/pierre')
    print('=' * 70)


if __name__ == '__main__':
    main()
