import json
import os
import subprocess
import tempfile

import pixeltable as pxt


@pxt.udf
def synthesize_tone_duration(frequency_hz: int) -> float:
    """Synthesize a 1-second sine tone and return its measured duration in seconds.

    Uses ffmpeg (lavfi) to generate a WAV file and ffprobe to read the duration back.
    Declared via system_dependencies = ["ffmpeg"] in pixeltable.toml.
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_path = f.name
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'lavfi', '-i', f'sine=frequency={frequency_hz}:duration=1', wav_path],
            capture_output=True,
            check=True,
        )
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', wav_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(json.loads(probe.stdout)['format']['duration'])
    finally:
        os.unlink(wav_path)


@pxt.query
def find_by_id(item_id: int) -> pxt.Query:
    items = pxt.get_table('e2e_items')
    return items.where(items.id == item_id)
