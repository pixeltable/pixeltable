---
title: "5-Minute Quickstart"
description: "Build a multimodal AI pipeline that processes videos, extracts transcripts, and generates summaries"
---

# 5-Minute Quickstart

Build your first multimodal AI pipeline in 5 minutes! We'll create a system that processes videos, extracts transcripts, and generates summaries.

## 1. Create Your First Table

```python
import pixeltable as pxt

# Create a directory for your project
pxt.create_dir('tutorial')

# Create a table for videos
videos = pxt.create_table(
    'tutorial.videos',
    {'video': pxt.Video}
)
```

## 2. Add AI Transformations

```python
# Add transcription using Whisper
videos.add_computed_column(
    transcript=pxt.functions.whisper.transcribe(
        videos.video, 
        model='base'
    )
)

# Add summary using OpenAI
videos.add_computed_column(
    summary=pxt.functions.openai.chat_completions(
        model='gpt-4o-mini',
        messages=[{
            'role': 'user', 
            'content': f'Summarize this transcript: {videos.transcript}'
        }]
    )
)
```

## 3. Insert Data and Watch the Magic

```python
# Insert a video (can be local file or URL)
videos.insert({'video': '/path/to/your/video.mp4'})

# Query the results
results = videos.select(videos.video, videos.transcript, videos.summary).collect()
print(results)
```

## 4. Scale with Incremental Computation

```python
# Add more videos - only new ones get processed!
videos.insert([
    {'video': '/path/to/video2.mp4'},
    {'video': '/path/to/video3.mp4'},
])

# Pixeltable automatically processes only the new videos
# Previous transcripts and summaries are cached
```

## What Just Happened?

🎉 **Congratulations!** You just built a production-ready multimodal AI pipeline that:

- **Stores** videos with built-in versioning
- **Processes** them automatically with AI models  
- **Caches** results for efficiency
- **Scales** incrementally as you add data

## Next Steps

- [Explore Examples](../examples/overview) - See real-world use cases
- [Join Discord](https://discord.com/invite/QPyqFYx2UN) - Get help from the community
- [API Reference](https://pixeltable.github.io/pixeltable/) - Dive deeper
