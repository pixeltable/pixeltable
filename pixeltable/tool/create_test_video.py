import av
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from pathlib import Path
from typing import Optional
import tempfile
import math

def create_test_video(
    frame_count: int,
    frame_rate: float = 1.0,
    frame_width: int = 224,
    aspect_ratio: str = '16:9',
    frame_height: Optional[int] = None,
    output_path: Optional[Path] = None,
    font_file: str = '/Library/Fonts/Arial Unicode.ttf',
) -> Path:
    """
        Creates an .mp4 video file such as the ones in /tests/data/test_videos
        The video contains a frame number in each frame (for visual sanity check).

        Args:
            frame_count: Number of frames to create
            frame_rate: Frame rate of the video
            frame_width (int): Width in pixels of the video frame. Note: cost of decoding increases dramatically
                with frame width * frame height.
            aspect_ratio: Aspect ratio (width/height) of the video frames string of form 'width:height'
            frame_height: Height of the video frame, if given, aspect_ratio is ignored
            output_path: Path to save the video file
            font_file: Path to the font file used for text.
    """

    if output_path is None:
        output_path = Path(tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name)

    parts = [int(p) for p in aspect_ratio.split(':')]
    assert len(parts) == 2
    aspect_ratio = parts[0] / parts[1]

    if frame_height is None:
        frame_height = math.ceil(frame_width / aspect_ratio)

    frame_size = (frame_width, frame_height)

    font_size = min(frame_height, frame_width) // 4
    font = PIL.ImageFont.truetype(font=font_file, size=font_size)
    font_fill = 0xFFFFFF  # white
    frame_color = 0xFFFFFF - font_fill  # black
    # Create a video container
    container = av.open(str(output_path), mode='w')

    # Add a video stream
    stream = container.add_stream('h264', rate=frame_rate)
    stream.width, stream.height = frame_size
    stream.pix_fmt = 'yuv420p'

    for frame_number in range(frame_count):
        # Create an image with a number in it
        image = PIL.Image.new('RGB', frame_size, color=frame_color)
        draw = PIL.ImageDraw.Draw(image)
        # Optionally, add a font here if you have one
        text = str(frame_number)
        _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font)
        text_position = ((frame_size[0] - text_width) // 2, (frame_size[1] - text_height) // 2)
        draw.text(text_position, text, font=font, fill=font_fill)

        # Convert the PIL image to an AVFrame
        frame = av.VideoFrame.from_image(image)

        # Encode and write the frame
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush and close the stream
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return output_path
