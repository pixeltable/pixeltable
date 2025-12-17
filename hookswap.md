# Hookswap Tool Overview
## What It Does
Hookswap takes an existing video and generates multiple variants with AI-generated "hook" introductions. It creates scroll-stopping opening hooks using:
- AI-generated visuals matching the hook psychology
- Text-to-speech narration with voice matching
- Auto-captioning for accessibility
- Multiple aspect ratio outputs (9:16, 1:1, 16:9)
---
## Tool Workflow
```
User uploads video + selects industry/state
        ↓
Preprocessing → Generates hooks, voices, transcription
        ↓
AWS Batch Processing → Hook visual generation, video stitching, captioning
        ↓
Job Complete → User retrieves final videos
```
---
## Step-by-Step Breakdown
### Step 1: Preprocessing Pipeline
**What it does:**
Analyzes the video and generates state-specific hook text using AI.
#### 1a. Video Analysis
**AWS Transcribe** processes the video audio:
- Extracts full transcript with word-level timestamps
- Identifies sentence boundaries and natural pauses
- Returns JSON with start/end times for each word
- Cached to avoid re-processing same videos
#### 1b. Brand Context (Optional)
**Playwright** browser automation:
- Launches headless Chromium browser
- Navigates to brand website
- Waits for JavaScript to render content
- Extracts text, headings, and meta descriptions
- Returns brand voice, values, and key messaging
#### 1c. Hook Text Generation
**OpenAI GPT-4o** generates state-specific hooks:
- Receives industry context (insurance, fitness, real estate, etc.)
- Receives state context (California: Prop 13/wildfires, Texas: homestead/hail, Florida: hurricanes)
- Generates 3 hook variations per state
- Each hook tailored to local regulations and crises
- If 3 states provided: generates 9 total hooks (3 per state)
#### 1d. Voice Selection
**OpenAI GPT-4o** analyzes hook tone:
- Reads each hook text
- Determines emotional tone (authoritative, empathetic, urgent, friendly)
- Matches to **ElevenLabs** voice by gender, age, accent, energy
- Returns voice ID for TTS generation
**Technologies:**
- OpenAI GPT-4o (hook generation + voice selection)
- AWS Transcribe (timestamped transcription)
- Playwright (brand scraping with JS rendering)
---
### Step 2: AWS Batch Processing
**What it does:**
Heavy video processing runs in isolated Docker container on AWS Batch.
#### 2a. Hook Visual Generation
**Visual Concept Generation:**
**OpenAI GPT-4o** creates video description:
- Receives hook text + industry + state
- Generates 50-word creative visual concept
- Insurance + California → "home on hillside threatened by wildfire with protective barrier"
- Fitness → "person mid-workout showing intense effort and transformation"
**Image Creation:**
**OpenAI Image Generation** generates static image:
- Takes visual concept from GPT-4o
- Creates 1024x1536 portrait image
- Applies industry color psychology (insurance: blue/red, fitness: orange/green)
- Includes state-specific elements (California: wildfire smoke, Texas: hail, Florida: hurricane)
**Video Conversion:**
**LumaAI Ray-2** converts image to video:
- Takes static image URL
- Generates 3-second video with subtle motion
- Returns MP4 file
**FFmpeg** trims to exact duration:
- Loads LumaAI output video
- Cuts to exactly 3.0 seconds using: `ffmpeg -i input.mp4 -t 3.0 -c copy output.mp4`
- Ensures precise timing for hook insertion
#### 2b. Text-to-Speech
**ElevenLabs TTS** generates narration:
- Takes hook text + selected voice ID
- Synthesizes natural-sounding speech MP3
- Returns audio file matching hook duration
- If audio longer than 3 seconds: speeds up playback to fit
#### 2c. Hook Assembly
**FFmpeg** merges visual + audio:
- Takes 3-second hook video
- Takes TTS audio
- Combines using: `ffmpeg -i video.mp4 -i audio.mp3 -c:v copy -c:a aac -shortest output.mp4`
- Synchronizes audio/video streams
- Adjusts audio speed if mismatch (preserves video timing)
#### 2d. Captioning
**ZapCap API** (primary) or **AutoCaption API** (fallback):
- Sends hook video
- Receives SRT caption file with timecoded text
- Returns word-by-word timing
**FFmpeg** burns captions into video:
- Loads hook video + SRT file
- Overlays text using `subtitles` filter
- Renders captions directly into video pixels (not separate track)
- Returns captioned hook video
#### 2e. Video Stitching
**FFmpeg** combines hook + original video:
**Simple mode:**
- Concatenates captioned hook to beginning of original: `ffmpeg -i hook.mp4 -i original.mp4 -filter_complex concat -c:v libx264 output.mp4`
**Auto-detect mode:**
- Parses AWS Transcribe timestamps
- Finds natural sentence break (first sentence end after 3 seconds)
- Splits original video at insertion point: `ffmpeg -i original.mp4 -ss 0 -to 8.5 before.mp4`
- Stitches: before + hook + after using concat filter
#### 2f. Aspect Ratio Variants
**FFmpeg** creates padded versions for different aspect ratios:
Original video is 9:16 (vertical):
- **For 1:1 (square):**
  - Takes stitched video
  - Adds side padding with blurred background: `ffmpeg -i input.mp4 -vf "split[original][blur];[blur]scale=1080:1920,boxblur=20:20[blurred];[blurred][original]overlay=(W-w)/2:(H-h)/2" -c:v libx264 output_1x1.mp4`
  - Blurs and scales original video as background
  - Overlays original video centered on top
- **For 16:9 (landscape):**
  - Similar process but with wider blur background
  - Scales to 1920x1080
  - Centers original video
**FFmpeg** compresses each version:
- Uses H.264 codec with CRF 28 (balances quality/size): `ffmpeg -i input.mp4 -c:v libx264 -crf 28 output.mp4`
- Reduces file size by 60-70%
- Maintains visual quality for social media
**Technologies:**
- FFmpeg (split, concat, overlay, blur, scale, compress)
- OpenAI GPT-4o (visual concepts)
- OpenAI Image Generation (image generation)
- LumaAI Ray-2 (image-to-video)
- ElevenLabs TTS (voice synthesis)
- ZapCap / AutoCaption (captions)