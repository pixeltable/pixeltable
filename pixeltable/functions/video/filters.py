import math
from pathlib import Path
from typing import Any, Literal

import av

import pixeltable as pxt
import pixeltable.utils.av as av_utils
from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.functions import util
from pixeltable.functions.math import abs as pxt_abs, floor as pxt_floor
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore


@pxt.udf(is_method=True, run_in_thread=True)
def overlay_text(
    video: pxt.Video,
    text: str,
    *,
    font: str | None = None,
    font_size: int = 24,
    line_spacing: int = 0,
    color: str = 'white',
    opacity: float = 1.0,
    horizontal_align: Literal['left', 'center', 'right'] = 'center',
    horizontal_margin: int = 0,
    vertical_align: Literal['top', 'center', 'bottom'] = 'center',
    vertical_margin: int = 0,
    box: bool = False,
    box_color: str = 'black',
    box_opacity: float = 1.0,
    box_border: list[int] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Overlay text on a video with customizable positioning and styling.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video to overlay text on.
        text: The text string to overlay on the video.
        font: Font family or path to font file. If None, uses the system default.
        font_size: Size of the text in points.
        line_spacing: Pixels of vertical space added between lines of multi-line text (text containing
            `\n`). Defaults to 0. Negative values pull lines closer together for tighter packing.
            Ignored for single-line text.
        color: Text color (e.g., `'white'`, `'red'`, `'#FF0000'`).
        opacity: Text opacity from 0.0 (transparent) to 1.0 (opaque).
        horizontal_align: Horizontal text alignment (`'left'`, `'center'`, `'right'`).
        horizontal_margin: Horizontal margin in pixels from the alignment edge.
        vertical_align: Vertical text alignment (`'top'`, `'center'`, `'bottom'`).
        vertical_margin: Vertical margin in pixels from the alignment edge.
        box: Whether to draw a background box behind the text.
        box_color: Background box color as a string.
        box_opacity: Background box opacity from 0.0 to 1.0.
        box_border: Padding around text in the box in pixels.

            - `[10]`: 10 pixels on all sides
            - `[10, 20]`: 10 pixels on top/bottom, 20 on left/right
            - `[10, 20, 30]`: 10 pixels on top, 20 on left/right, 30 on bottom
            - `[10, 20, 30, 40]`: 10 pixels on top, 20 on right, 30 on bottom, 40 on left
        start_time: Time in seconds when the text appears. If None, the text is visible from the start.
        end_time: Time in seconds when the text disappears. If None, the text is visible until the end.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the text overlay applied.

    Examples:
        Add a simple text overlay to videos in a table:

        >>> tbl.select(tbl.video.overlay_text('Sample Text')).collect()

        Add a YouTube-style caption:

        >>> tbl.select(
        ...     tbl.video.overlay_text(
        ...         'Caption text',
        ...         font_size=32,
        ...         color='white',
        ...         opacity=1.0,
        ...         box=True,
        ...         box_color='black',
        ...         box_opacity=0.8,
        ...         box_border=[6, 14],
        ...         horizontal_margin=10,
        ...         vertical_align='bottom',
        ...         vertical_margin=70,
        ...     )
        ... ).collect()

        Add text with a semi-transparent background box:

        >>> tbl.select(
        ...     tbl.video.overlay_text(
        ...         'Important Message',
        ...         font_size=32,
        ...         color='yellow',
        ...         box=True,
        ...         box_color='black',
        ...         box_opacity=0.6,
        ...         box_border=[20, 10],
        ...     )
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if font_size <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'font_size must be positive, got {font_size}')
    if opacity < 0.0 or opacity > 1.0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'opacity must be between 0.0 and 1.0, got {opacity}')
    if horizontal_margin < 0:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'horizontal_margin must be non-negative, got {horizontal_margin}'
        )
    if vertical_margin < 0:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'vertical_margin must be non-negative, got {vertical_margin}'
        )
    if box_opacity < 0.0 or box_opacity > 1.0:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'box_opacity must be between 0.0 and 1.0, got {box_opacity}'
        )
    if box_border is not None and not (
        isinstance(box_border, (list, tuple))
        and len(box_border) >= 1
        and len(box_border) <= 4
        and all(isinstance(x, int) for x in box_border)
        and all(x >= 0 for x in box_border)
    ):
        raise pxt.RequestError(
            pxt.ErrorCode.TYPE_MISMATCH,
            f'box_border must be a list or tuple of 1-4 non-negative ints, got {box_border!s} instead',
        )

    output_path = str(TempStore.create_path(extension='.mp4'))

    if start_time is not None and start_time < 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'start_time must be non-negative, got {start_time}')
    if end_time is not None and end_time < 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'end_time must be non-negative, got {end_time}')
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f'start_time must be less than end_time, got start_time={start_time}, end_time={end_time}',
        )

    drawtext_params = _create_drawtext_params(
        text,
        font,
        font_size,
        line_spacing,
        color,
        opacity,
        horizontal_align,
        horizontal_margin,
        vertical_align,
        vertical_margin,
        box,
        box_color,
        box_opacity,
        box_border,
    )

    if start_time is not None or end_time is not None:
        st = start_time if start_time is not None else 0
        et = end_time if end_time is not None else 99999999
        drawtext_params.append(f'enable=between(t\\,{st}\\,{et})')

    cmd = [
        '-i',
        str(video),
        '-vf',
        'drawtext=' + ':'.join(drawtext_params),
        '-c:a',
        'copy',  # Copy audio stream unchanged
    ]
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


def _create_drawtext_params(
    text: str,
    font: str | None,
    font_size: int,
    line_spacing: int,
    color: str,
    opacity: float,
    horizontal_align: str,
    horizontal_margin: int,
    vertical_align: str,
    vertical_margin: int,
    box: bool,
    box_color: str,
    box_opacity: float,
    box_border: list[int] | None,
) -> list[str]:
    """Construct parameters for the ffmpeg drawtext filter"""
    drawtext_params: list[str] = []
    escaped_text = text.replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")
    drawtext_params.append(f"text='{escaped_text}'")
    drawtext_params.append(f'fontsize={font_size}')

    if font is not None:
        if Path(font).exists():
            drawtext_params.append(f"fontfile='{font}'")
        else:
            drawtext_params.append(f"font='{font}'")
    if opacity < 1.0:
        drawtext_params.append(f'fontcolor={color}@{opacity}')
    else:
        drawtext_params.append(f'fontcolor={color}')

    if line_spacing != 0:
        drawtext_params.append(f'line_spacing={line_spacing}')

    if horizontal_align == 'left':
        x_expr = str(horizontal_margin)
    elif horizontal_align == 'center':
        x_expr = '(w-text_w)/2'
    else:  # right
        x_expr = f'w-text_w-{horizontal_margin}' if horizontal_margin != 0 else 'w-text_w'
    if vertical_align == 'top':
        y_expr = str(vertical_margin)
    elif vertical_align == 'center':
        y_expr = '(h-text_h)/2'
    else:  # bottom
        y_expr = f'h-text_h-{vertical_margin}' if vertical_margin != 0 else 'h-text_h'
    drawtext_params.extend([f'x={x_expr}', f'y={y_expr}'])

    if box:
        drawtext_params.append('box=1')
        if box_opacity < 1.0:
            drawtext_params.append(f'boxcolor={box_color}@{box_opacity}')
        else:
            drawtext_params.append(f'boxcolor={box_color}')
        if box_border is not None:
            drawtext_params.append(f'boxborderw={"|".join(map(str, box_border))}')

    return drawtext_params


@pxt.udf(is_method=True, run_in_thread=True)
def overlay_image(
    video: pxt.Video,
    image: pxt.Image,
    *,
    horizontal_align: Literal['left', 'center', 'right'] = 'center',
    horizontal_margin: int = 0,
    vertical_align: Literal['top', 'center', 'bottom'] = 'center',
    vertical_margin: int = 0,
    scale: float | None = None,
    opacity: float = 1.0,
    start_time: float | None = None,
    end_time: float | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Overlay an image on a video with customizable positioning, scaling, opacity, and timing.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video to overlay the image on.
        image: Image to overlay.
        horizontal_align: Horizontal alignment of the overlay (`'left'`, `'center'`, `'right'`).
        horizontal_margin: Horizontal margin in pixels from the alignment edge.
        vertical_align: Vertical alignment of the overlay (`'top'`, `'center'`, `'bottom'`).
        vertical_margin: Vertical margin in pixels from the alignment edge.
        scale: Scale factor for the overlay image relative to the video height. For example, 0.1 scales the
            image to 10% of the video height while preserving aspect ratio. If None, uses the original size.
        opacity: Overlay opacity from 0.0 (transparent) to 1.0 (opaque).
        start_time: Time in seconds when the overlay appears. If None, the overlay is visible from the start.
        end_time: Time in seconds when the overlay disappears. If None, the overlay is visible until the end.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the image overlay applied.

    Examples:
        Add a logo to the top-right corner:

        >>> tbl.select(
        ...     tbl.video.overlay_image(
        ...         tbl.logo_img, horizontal_align='right', vertical_align='top'
        ...     )
        ... ).collect()

        Add a watermark at 50% opacity, scaled to 10% of video height:

        >>> tbl.select(
        ...     tbl.video.overlay_image(tbl.watermark_img, scale=0.1, opacity=0.5)
        ... ).collect()

        Show an image only between seconds 2 and 8:

        >>> tbl.select(
        ...     tbl.video.overlay_image(
        ...         tbl.img, start_time=2.0, end_time=8.0, horizontal_align='right'
        ...     )
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if horizontal_margin < 0:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'horizontal_margin must be non-negative, got {horizontal_margin}'
        )
    if vertical_margin < 0:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'vertical_margin must be non-negative, got {vertical_margin}'
        )
    if opacity < 0.0 or opacity > 1.0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'opacity must be between 0.0 and 1.0, got {opacity}')
    if scale is not None and scale <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'scale must be positive, got {scale}')
    if start_time is not None and start_time < 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'start_time must be non-negative, got {start_time}')
    if end_time is not None and end_time < 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'end_time must be non-negative, got {end_time}')
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f'start_time must be less than end_time, got start_time={start_time}, end_time={end_time}',
        )

    output_path = str(TempStore.create_path(extension='.mp4'))

    # ffmpeg needs file input
    image_path = str(TempStore.create_path(extension='.png'))
    image.convert('RGBA').save(image_path)

    x_expr: str
    if horizontal_align == 'left':
        x_expr = str(horizontal_margin)
    elif horizontal_align == 'center':
        x_expr = '(W-w)/2'
    else:  # right
        x_expr = f'W-w-{horizontal_margin}' if horizontal_margin != 0 else 'W-w'

    y_expr: str
    if vertical_align == 'top':
        y_expr = str(vertical_margin)
    elif vertical_align == 'center':
        y_expr = '(H-h)/2'
    else:  # bottom
        y_expr = f'H-h-{vertical_margin}' if vertical_margin != 0 else 'H-h'

    filters: list[str] = []

    overlay_label: str
    if scale is not None:
        md = util.get_metadata(str(video))
        video_height = next(s for s in md['streams'] if s['type'] == 'video')['height']
        filters.append(f'[1:v]scale=-2:trunc({video_height}*{scale}/2)*2[ovr_scaled]')
        overlay_label = '[ovr_scaled]'
    else:
        overlay_label = '[1:v]'

    # apply opacity to the overlay if not fully opaque
    if opacity < 1.0:
        out_label = '[ovr_alpha]'
        filters.append(f'{overlay_label}format=rgba,colorchannelmixer=aa={opacity}{out_label}')
        overlay_label = out_label

    # Build enable clause for timed overlay
    enable_clause = ''
    if start_time is not None or end_time is not None:
        st = start_time if start_time is not None else 0
        et = end_time if end_time is not None else 99999999
        enable_clause = f":enable='between(t,{st},{et})'"

    filters.append(f'[0:v]{overlay_label}overlay={x_expr}:{y_expr}{enable_clause}[vout]')
    filter_complex = ';'.join(filters)

    cmd = [
        '-i',
        str(video),
        '-i',
        image_path,
        '-filter_complex',
        filter_complex,
        '-map',
        '[vout]',
        '-map',
        # 0:a?: make the audio stream optional
        '0:a?',
        '-c:a',
        'copy',
    ]
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True, run_in_thread=True)
def crop(
    video: pxt.Video,
    bbox: list[int],
    *,
    bbox_format: Literal['xyxy', 'xywh', 'cxcywh'] = 'xywh',
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Crop a rectangular region from a video using ffmpeg's crop filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        bbox: Crop region as a list of 4 integers.
        bbox_format: Format of the `bbox` coordinates:

            - `'xyxy'`: `[x1, y1, x2, y2]` where (x1, y1) is top-left and (x2, y2) is bottom-right
            - `'xywh'`: `[x, y, width, height]` where (x, y) is top-left corner
            - `'cxcywh'`: `[cx, cy, width, height]` where (cx, cy) is the center
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        Video containing the cropped region.

    Examples:
        Crop using default xywh format:

        >>> tbl.select(tbl.video.crop([100, 50, 320, 240])).collect()

        Crop using xyxy format (common in object detection):

        >>> tbl.select(
        ...     tbl.video.crop([100, 50, 420, 290], bbox_format='xyxy')
        ... ).collect()

        Crop using center format:

        >>> tbl.select(
        ...     tbl.video.crop([260, 170, 320, 240], bbox_format='cxcywh')
        ... ).collect()

        Use with yolox object detection output:

        >>> tbl.add_computed_column(
        ...     cropped=tbl.video.crop(tbl.detections.bboxes[0], bbox_format='xyxy')
        ... )
    """
    Env.get().require_binary('ffmpeg')

    if len(bbox) != 4 or not all(isinstance(x, int) for x in bbox) or not all(x >= 0 for x in bbox):
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'bbox must have exactly 4 non-negative integers, got {bbox}'
        )
    if bbox_format == 'xyxy' and (bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f'x2 must be greater than x1 and y2 must be greater than y1 for xyxy format, got {bbox}',
        )

    # normalize to xywh
    x: int
    y: int
    w: int
    h: int
    if bbox_format == 'xyxy':
        x1, y1, x2, y2 = bbox
        x, y = x1, y1
        w, h = x2 - x1, y2 - y1
    elif bbox_format == 'xywh':
        x, y, w, h = bbox
    elif bbox_format == 'cxcywh':
        cx, cy, w, h = bbox
        x = cx - w // 2
        y = cy - h // 2
    else:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f"bbox_format must be one of ['xyxy', 'xywh', 'cxcywh'], got {bbox_format!r}",
        )

    cmd = ['-i', str(video), '-vf', f'crop={w}:{h}:{x}:{y}', '-c:a', 'copy']
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True, run_in_thread=True)
def resize(
    video: pxt.Video,
    *,
    width: int | None = None,
    height: int | None = None,
    scale: float | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Resize a video using ffmpeg's scale filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        width: Width of the output video. Maintains the existing aspect ratio if no `height` is provided.
        height: Height of the output video. Maintains the existing aspect ratio if no `width` is provided.
        scale: Scale factor. Mutually exclusive with `width` and `height`.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        The resized video.

    Examples:
        Resize to a specific width, preserving aspect ratio:

        >>> tbl.select(tbl.video.resize(width=640)).collect()

        Resize to exact dimensions:

        >>> tbl.select(tbl.video.resize(width=1280, height=720)).collect()

        Scale down by half:

        >>> tbl.select(tbl.video.resize(scale=0.5)).collect()
    """
    Env.get().require_binary('ffmpeg')

    if scale is not None and (width is not None or height is not None):
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, '`scale` is mutually exclusive with `width` and `height`'
        )
    if scale is not None:
        if scale <= 0:
            raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'`scale` must be positive, got {scale}')
        scale_filter = f'scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2'
    elif width is not None or height is not None:
        if width is not None and width <= 0:
            raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'`width` must be positive, got {width}')
        if height is not None and height <= 0:
            raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'`height` must be positive, got {height}')

        # Use -2 for the unspecified dimension: like -1 (preserve aspect ratio),
        # but rounds to the nearest even value (required by most codecs)
        w_expr = str(width) if width is not None else '-2'
        h_expr = str(height) if height is not None else '-2'
        scale_filter = f'scale={w_expr}:{h_expr}'
    else:
        raise pxt.RequestError(
            pxt.ErrorCode.MISSING_REQUIRED, 'At least one of `width`, `height`, or `scale` must be specified'
        )

    output_path = str(TempStore.create_path(extension='.mp4'))
    cmd = ['-i', str(video), '-vf', scale_filter, '-c:a', 'copy']
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True, run_in_thread=True)
def reverse(
    video: pxt.Video,
    audio: Literal['reverse', 'drop', 'keep'] = 'drop',
    *,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Reverse a video using ffmpeg's reverse filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        audio: Specifies what to do with audio streams

            - `'drop'`: drop the audio streams
            - `'reverse'`: also reverse the audio streams
            - `'keep'`: keep the audio streams
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        The reversed video.

    Examples:
        Reverse a video, dropping audio:

        >>> tbl.select(tbl.video.reverse()).collect()

        Reverse a video along with its audio:

        >>> tbl.select(tbl.video.reverse(audio='reverse')).collect()
    """

    Env.get().require_binary('ffmpeg')

    # ffmpeg's reverse filter requires all frames to be decoded into memory at once, which can exhaust RAM on
    # long or high-resolution videos. To avoid this, we split the video into segments whose decoded frames fit
    # within ~1 GB, reverse each segment independently, then concatenate the reversed segments in reverse order.
    segment_bytes = 2**30
    segment_duration = av_utils.estimate_segment_duration(video, segment_bytes)
    if segment_duration is None:
        raise pxt.RequestError(pxt.ErrorCode.UNSUPPORTED_OPERATION, f'not a valid video: {video}')

    duration = av_utils.get_video_duration(video)
    if duration is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_DATA_FORMAT, f'reverse(): could not determine video duration: {video}'
        )

    with av.open(video) as container:
        has_audio = any(s.type == 'audio' for s in container.streams)

    starts = [segment_duration * i for i in range(math.ceil(duration / segment_duration))]

    # Build the filtergraph. For a 25s video with segment_duration=10, starts=[0, 10, 20] and the filtergraph is:
    #
    #   [0:v]trim=start=0:end=10,setpts=PTS-STARTPTS,reverse[v0];
    #   [0:v]trim=start=10:end=20,setpts=PTS-STARTPTS,reverse[v1];
    #   [0:v]trim=start=20,setpts=PTS-STARTPTS,reverse[v2];
    #   [v2][v1][v0]concat=n=3:v=1:a=0[v]
    #
    # Each segment is: trim to time range -> reset timestamps -> reverse.
    # The last segment omits :end= so it runs to the end of the stream.
    # The concat inputs are listed in reverse order ([v2][v1][v0]) so the last segment of the original
    # video becomes the first segment of the output.
    n = len(starts)
    filter_parts: list[str] = []

    for i, start in enumerate(starts):
        is_last = i == n - 1
        end_clause = '' if is_last else f':end={start + segment_duration}'

        filter_parts.append(f'[0:v]trim=start={start}{end_clause},setpts=PTS-STARTPTS,reverse[v{i}]')
        if audio == 'reverse' and has_audio:
            filter_parts.append(f'[0:a]atrim=start={start}{end_clause},asetpts=PTS-STARTPTS,areverse[a{i}]')

    v_inputs = ''.join(f'[v{i}]' for i in range(n - 1, -1, -1))
    filter_parts.append(f'{v_inputs}concat=n={n}:v=1:a=0[v]')

    if audio == 'reverse' and has_audio:
        a_inputs = ''.join(f'[a{i}]' for i in range(n - 1, -1, -1))
        filter_parts.append(f'{a_inputs}concat=n={n}:v=0:a=1[a]')

    filtergraph = '; '.join(filter_parts)

    # Example commandline (audio='reverse'):
    #   ffmpeg -i input.mp4 -filter_complex "<filtergraph>" -map [v] -map [a] -loglevel error out.mp4
    # audio='keep': -map 0:a -c:a copy (passes original audio through without the filtergraph)
    # audio='drop': no audio mapping, so ffmpeg omits audio from the output
    cmd = ['-i', str(video), '-filter_complex', filtergraph, '-map', '[v]']
    # we need to add the video encoder args at this point (not later)
    av_utils.append_video_encoder(cmd, video_encoder, video_encoder_args)

    if audio == 'reverse' and has_audio:
        cmd.extend(['-map', '[a]'])
    elif audio == 'keep' and has_audio:
        cmd.extend(['-map', '0:a', '-c:a', 'copy'])

    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(cmd, output_path)


def _fade(
    video: str,
    direction: Literal['in', 'out'],
    duration: float,
    color: str,
    video_duration: float | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> str:
    Env.get().require_binary('ffmpeg')
    if duration <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'duration must be positive, got {duration}')

    if direction == 'in':
        start_time = 0.0
    else:
        assert video_duration is not None
        start_time = max(0, video_duration - duration)

    output_path = str(TempStore.create_path(extension='.mp4'))
    fade_filter = f'fade={direction}:st={start_time}:d={duration}:color={color}'
    cmd = ['-i', str(video), '-vf', fade_filter, '-c:a', 'copy']
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True, run_in_thread=True)
def fade_in(
    video: pxt.Video,
    *,
    duration: float = 1.0,
    color: str = 'black',
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Apply a fade-in effect from a solid color at the start of a video using ffmpeg's fade filter.
    The video transitions from a solid `color` to the full video content over `duration` seconds.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        duration: Duration of the fade-in effect in seconds.
        color: Color to fade from (e.g., `'black'`, `'white'`, `'#FF0000'`).
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the fade-in effect applied.

    Examples:
        Apply a 1-second fade from black (default):

        >>> tbl.select(tbl.video.fade_in()).collect()

        Apply a 2-second fade from white:

        >>> tbl.select(tbl.video.fade_in(duration=2.0, color='white')).collect()
    """
    return _fade(video, 'in', duration, color, video_encoder=video_encoder, video_encoder_args=video_encoder_args)


@pxt.udf(is_method=True, run_in_thread=True)
def fade_out(
    video: pxt.Video,
    *,
    duration: float = 1.0,
    color: str = 'black',
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Apply a fade-out effect to a solid color at the end of a video using ffmpeg's fade filter.
    The video transitions from the full video content to a solid `color` over the final `duration` seconds.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        duration: Duration of the fade-out effect in seconds.
        color: Color to fade to (e.g., `'black'`, `'white'`, `'#FF0000'`).
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the fade-out effect applied.

    Examples:
        Apply a 1-second fade to black (default):

        >>> tbl.select(tbl.video.fade_out()).collect()

        Apply a 3-second fade to white:

        >>> tbl.select(tbl.video.fade_out(duration=3.0, color='white')).collect()
    """
    video_duration = av_utils.get_video_duration(video)
    if video_duration is None:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_DATA_FORMAT, 'fade_out(): could not determine video duration')
    return _fade(
        video,
        'out',
        duration,
        color,
        video_duration=video_duration,
        video_encoder=video_encoder,
        video_encoder_args=video_encoder_args,
    )


@pxt.udf(run_in_thread=True)
def transition(
    video1: pxt.Video,
    video2: pxt.Video,
    *,
    effect: Literal[
        'fade',
        'wipeleft',
        'wiperight',
        'wipeup',
        'wipedown',
        'slideleft',
        'slideright',
        'slideup',
        'slidedown',
        'dissolve',
        'smoothleft',
        'smoothright',
        'smoothup',
        'smoothdown',
    ] = 'fade',
    duration: float = 1.0,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Join two video clips with a transition effect using ffmpeg's xfade filter.

    Applies a crossfade or other transition effect between the end of the first clip and the beginning of
    the second clip. The transition overlaps the last `duration` seconds of `video1` with the first `duration`
    seconds of `video2`, so the total output duration is `len(video1) + len(video2) - duration`.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video1: First video clip.
        video2: Second video clip.
        effect: Transition effect type. Common options:

            - `'fade'`: Classic crossfade (default).
            - `'dissolve'`: Dissolve transition.
            - `'wipeleft'`, `'wiperight'`, `'wipeup'`, `'wipedown'`: Wipe transitions.
            - `'slideleft'`, `'slideright'`, `'slideup'`, `'slidedown'`: Slide transitions.
            - `'smoothleft'`, `'smoothright'`, `'smoothup'`, `'smoothdown'`: Smooth transitions.
        duration: Duration of the transition in seconds.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the transition applied between the two clips.

    Examples:
        Join two clips with a 1-second crossfade:

        >>> tbl.select(transition(tbl.clip1, tbl.clip2)).collect()

        Join with a 2-second wipe-left transition:

        >>> tbl.select(
        ...     transition(tbl.clip1, tbl.clip2, effect='wipeleft', duration=2.0)
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if duration <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'duration must be positive, got {duration}')

    # xfade requires both inputs to have the same resolution
    md1 = util.get_metadata(str(video1))
    v1_stream = next(s for s in md1['streams'] if s['type'] == 'video')
    w1, h1 = v1_stream['width'], v1_stream['height']
    md2 = util.get_metadata(str(video2))
    v2_stream = next(s for s in md2['streams'] if s['type'] == 'video')
    w2, h2 = v2_stream['width'], v2_stream['height']
    if (w1, h1) != (w2, h2):
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f'video1 and video2 must have the same resolution, got {w1}x{h1} and {w2}x{h2}',
        )

    video1_duration = av_utils.get_video_duration(video1)
    if video1_duration is None:
        raise pxt.RequestError(pxt.ErrorCode.UNSUPPORTED_OPERATION, f'Could not determine duration of {video1}')
    if duration > video1_duration:
        raise pxt.RequestError(
            pxt.ErrorCode.UNSUPPORTED_OPERATION,
            f'transition duration ({duration}s) exceeds duration ({video1_duration}s) of {video1}',
        )
    video2_duration = av_utils.get_video_duration(video2)
    if video2_duration is None:
        raise pxt.RequestError(pxt.ErrorCode.UNSUPPORTED_OPERATION, f'Could not determine duration of {video2}')
    if duration > video2_duration:
        raise pxt.RequestError(
            pxt.ErrorCode.UNSUPPORTED_OPERATION,
            f'transition duration ({duration}s) exceeds duration ({video2_duration}s) of {video2}',
        )

    offset = video1_duration - duration
    output_path = str(TempStore.create_path(extension='.mp4'))

    # build xfade filter; handle audio with acrossfade if both clips have audio
    has_audio1 = av_utils.has_audio_stream(video1)
    has_audio2 = av_utils.has_audio_stream(video2)

    filter_complex = f'[0:v][1:v]xfade=transition={effect}:duration={duration}:offset={offset}[vout]'
    if has_audio1 and has_audio2:
        filter_complex += f';[0:a][1:a]acrossfade=d={duration}[aout]'
    cmd = ['-i', str(video1), '-i', str(video2), '-filter_complex', filter_complex, '-map', '[vout]']
    if has_audio1 and has_audio2:
        cmd.extend(['-map', '[aout]', '-c:a', 'aac'])
    elif has_audio1:
        cmd.extend(['-map', '0:a', '-c:a', 'copy'])
    elif has_audio2:
        cmd.extend(['-map', '1:a', '-c:a', 'copy'])

    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True, run_in_thread=True)
def speed(
    video: pxt.Video,
    *,
    factor: float,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Change the playback speed of a video using ffmpeg's setpts filter.

    A factor of 2.0 doubles the speed (halves the duration); a factor of 0.5 halves the speed (doubles the duration).
    Audio pitch is preserved using ffmpeg's `atempo` filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        factor: Speed multiplier. Must be positive. Values > 1.0 speed up, values < 1.0 slow down.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the adjusted playback speed.

    Examples:
        Double the speed:

        >>> tbl.select(tbl.video.speed(factor=2.0)).collect()

        Half speed (slow motion):

        >>> tbl.select(tbl.video.speed(factor=0.5)).collect()
    """
    Env.get().require_binary('ffmpeg')
    if factor <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'factor must be positive, got {factor}')

    output_path = str(TempStore.create_path(extension='.mp4'))
    # setpts=PTS/<factor> adjusts video timing; atempo=<factor> adjusts audio speed (preserving pitch).
    # atempo only accepts values in [0.5, 100.0]; for slower speeds, chain multiple atempo filters.
    video_filter = f'setpts=PTS/{factor}'
    has_audio = av_utils.has_audio_stream(video)

    cmd = ['-i', str(video), '-vf', video_filter]
    # add video encoder args here
    av_utils.append_video_encoder(cmd, video_encoder, video_encoder_args)

    if has_audio:
        # Chain atempo filters for factors outside [0.5, 100.0]
        atempo_parts = []
        remaining = factor
        while remaining < 0.5:
            atempo_parts.append('atempo=0.5')
            remaining /= 0.5
        while remaining > 100.0:
            atempo_parts.append('atempo=100.0')
            remaining /= 100.0
        atempo_parts.append(f'atempo={remaining}')
        cmd.extend(['-af', ','.join(atempo_parts)])
    else:
        cmd.append('-an')

    return av_utils.run_ffmpeg_cmdline(cmd, output_path)


def _flip(
    video: str,
    orientation: Literal['h', 'v'],
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> str:
    Env.get().require_binary('ffmpeg')
    flip_filter = 'hflip' if orientation == 'h' else 'vflip'
    cmd = ['-i', str(video), '-vf', flip_filter, '-c:a', 'copy']
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True, run_in_thread=True)
def mirror_x(
    video: pxt.Video, *, video_encoder: str | None = None, video_encoder_args: dict[str, Any] | None = None
) -> pxt.Video:
    """
    Flip a video horizontally using ffmpeg's hflip filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A horizontally flipped video.

    Examples:
        >>> tbl.select(tbl.video.mirror_x()).collect()
    """
    return _flip(video, 'h', video_encoder, video_encoder_args)


@pxt.udf(is_method=True, run_in_thread=True)
def mirror_y(
    video: pxt.Video, *, video_encoder: str | None = None, video_encoder_args: dict[str, Any] | None = None
) -> pxt.Video:
    """
    Flip a video vertically using ffmpeg's vflip filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A vertically flipped video.

    Examples:
        >>> tbl.select(tbl.video.mirror_y()).collect()
    """
    return _flip(video, 'v', video_encoder, video_encoder_args)


@pxt.udf(is_method=True, run_in_thread=True)
def rotate(
    video: pxt.Video,
    *,
    angle: float,
    unit: Literal['deg', 'rad'] = 'deg',
    expand: bool = False,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Rotate a video by a fixed angle using ffmpeg's rotate filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        angle: Rotation angle. Positive values rotate counter-clockwise.
        unit: Unit of the angle: `'deg'` for degrees or `'rad'` for radians.
        expand: If True, the output frame is enlarged to contain the entire rotated frame (no cropping).
            If False (default), the output frame keeps the original dimensions, cropping corners.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video rotated by the specified angle.

    Examples:
        Rotate 90 degrees counter-clockwise:

        >>> tbl.select(tbl.video.rotate(angle=90)).collect()

        Rotate 45 degrees with frame expansion to avoid cropping:

        >>> tbl.select(tbl.video.rotate(angle=45, expand=True)).collect()

        Rotate by pi/2 radians:

        >>> tbl.select(tbl.video.rotate(angle=1.5708, unit='rad')).collect()
    """
    Env.get().require_binary('ffmpeg')

    # Convert to radians for ffmpeg's rotate filter
    angle_rad = angle if unit == 'rad' else angle * math.pi / 180

    if expand:
        # Expand output to fit the rotated frame: compute new dimensions from the rotation angle
        # For a WxH frame rotated by A: new_w = W*|cos(A)| + H*|sin(A)|, new_h = W*|sin(A)| + H*|cos(A)|
        # Use ffmpeg's rotate filter with out_w/out_h expressions
        rotate_filter = (
            f'rotate={angle_rad}'
            f":ow='ceil((iw*abs(cos({angle_rad}))+ih*abs(sin({angle_rad})))/2)*2'"
            f":oh='ceil((iw*abs(sin({angle_rad}))+ih*abs(cos({angle_rad})))/2)*2'"
            f':fillcolor=black'
        )
    else:
        rotate_filter = f'rotate={angle_rad}:fillcolor=black'

    cmd = ['-i', str(video), '-vf', rotate_filter, '-c:a', 'copy']
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True, run_in_thread=True)
def grayscale(
    video: pxt.Video, *, video_encoder: str | None = None, video_encoder_args: dict[str, Any] | None = None
) -> pxt.Video:
    """
    Convert a video to grayscale

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A grayscale version of the video.

    Examples:
        >>> tbl.select(tbl.video.grayscale()).collect()
    """
    Env.get().require_binary('ffmpeg')

    output_path = str(TempStore.create_path(extension='.mp4'))

    # Convert to grayscale via hue filter (set saturation to 0), which keeps the yuv420p pixel format
    # compatible with most encoders. Using format=gray would produce a single-channel output that
    # many players and encoders don't handle well.
    cmd = ['-i', str(video), '-vf', 'hue=s=0', '-c:a', 'copy']
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True, run_in_thread=True)
def adjust_brightness(
    video: pxt.Video,
    *,
    factor: float,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Adjust the brightness of a video by a multiplicative factor using ffmpeg's lutrgb filter.

    A factor of 1.0 leaves the video unchanged; values below 1.0 dim the video (e.g., 0.5 for 50% brightness),
    and values above 1.0 brighten it (e.g., 1.5 for 150% brightness).

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        factor: Brightness multiplier. 0.0 produces a black video, 1.0 is unchanged, values > 1.0 brighten.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with adjusted brightness.

    Examples:
        Dim a video to 50% brightness:

        >>> tbl.select(tbl.video.adjust_brightness(factor=0.5)).collect()

        Brighten a video by 20%:

        >>> tbl.select(tbl.video.adjust_brightness(factor=1.2)).collect()
    """
    Env.get().require_binary('ffmpeg')
    if factor < 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'factor must be non-negative, got {factor}')

    # FFmpeg eq filter: brightness is additive (-1.0 to 1.0), gamma is multiplicative.
    # Using curves filter with a master curve for true multiplicative brightness.
    # For the eq filter, we use gamma_r/gamma_g/gamma_b which are multiplicative.
    # However, the simplest approach: use the lut filter to multiply pixel values.
    output_path = str(TempStore.create_path(extension='.mp4'))
    # Clamp to 0-255: min(val*factor, 255)
    lut_expr = f"'min(val*{factor},255)'"
    cmd = ['-i', str(video), '-vf', f'lutrgb=r={lut_expr}:g={lut_expr}:b={lut_expr}', '-c:a', 'copy']
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True, run_in_thread=True)
def ffmpeg_filter(
    video: pxt.Video,
    *,
    vf: str,
    af: str | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Apply an arbitrary FFmpeg filter expression to a video.

    The `vf` string is passed directly as the `-vf` argument to FFmpeg. If `af` is
    also provided, it is passed as the `-af` argument; otherwise the audio stream is copied unchanged.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        vf: FFmpeg video filter string, passed as `-vf`.
        af: Optional FFmpeg audio filter string, passed as `-af`. If None, the audio stream is copied
            unchanged. The input video must have an audio stream when `af` is provided.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the filter(s) applied.

    Examples:
        Apply a sepia tone:

        >>> tbl.select(
        ...     tbl.video.ffmpeg_filter(
        ...         vf='colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131'
        ...     )
        ... ).collect()

        Sharpen a video:

        >>> tbl.select(tbl.video.ffmpeg_filter(vf='unsharp=5:5:1.5')).collect()

        Add a vignette with audio normalization:

        >>> tbl.select(
        ...     tbl.video.ffmpeg_filter(vf='vignette', af='loudnorm')
        ... ).collect()

        Chain multiple video filters:

        >>> tbl.select(
        ...     tbl.video.ffmpeg_filter(vf='eq=brightness=0.1,hue=h=30')
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')

    output_path = str(TempStore.create_path(extension='.mp4'))
    cmd = ['-i', str(video), '-vf', vf]
    if af is not None:
        cmd.extend(['-af', af])
    else:
        cmd.extend(['-c:a', 'copy'])
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True, run_in_thread=True)
def scroll(
    video: pxt.Video,
    *,
    w: int | None = None,
    h: int | None = None,
    x_speed: float = 0,
    y_speed: float = 0,
    x_start: int = 0,
    y_start: int = 0,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Apply a scrolling viewport effect to a video using ffmpeg's crop filter.

    Extracts a viewport of size `w` x `h` from each frame, starting at position (`x_start`, `y_start`) and moving
    at (`x_speed`, `y_speed`) pixels per second. The viewport clamps at the frame edges: once it reaches a boundary,
    it stops moving and the remaining frames show a static crop.

    At least one of `w` or `h` must be smaller than the input dimensions for the effect to be visible.

    The clip duration is unchanged. To pan across the full available range, set
    `x_speed = (input_width - w) / duration`.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        w: Width of the output viewport in pixels. If None, uses the input width.
        h: Height of the output viewport in pixels. If None, uses the input height.
        x_speed: Horizontal scroll speed in pixels per second. Positive values scroll rightward (the viewport moves
            right, revealing content to the right). Negative values scroll leftward.
        y_speed: Vertical scroll speed in pixels per second. Positive values scroll downward. Negative values scroll
            upward.
        x_start: Initial horizontal offset of the viewport in pixels.
        y_start: Initial vertical offset of the viewport in pixels.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the scrolling effect applied. Output dimensions are `w` x `h`.

    Examples:
        Pan rightward across a 1920x1080 video using a 1280-pixel-wide viewport, scrolling at 50 px/s:

        >>> tbl.select(tbl.video.scroll(w=1280, x_speed=50)).collect()

        Pan rightward across the full range of a 1920x1080 video in exactly its duration. The viewport is
        1280 px wide, so the pan range is 1920 - 1280 = 640 px. For a 10-second video, set
        `x_speed = 640 / 10 = 64`:

        >>> tbl.select(tbl.video.scroll(w=1280, x_speed=64)).collect()

        Pan leftward across a 1920x1080 video, starting from the right edge:

        >>> tbl.select(tbl.video.scroll(w=1280, x_start=640, x_speed=-64)).collect()
    """
    Env.get().require_binary('ffmpeg')

    if x_speed == 0 and y_speed == 0:
        raise pxt.RequestError(
            pxt.ErrorCode.MISSING_REQUIRED, 'at least one of `x_speed` or `y_speed` must be non-zero'
        )
    if w is None and h is None:
        raise pxt.RequestError(pxt.ErrorCode.MISSING_REQUIRED, 'at least one of `w` or `h` must be specified')
    if w is not None and w <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'`w` must be positive, got {w}')
    if h is not None and h <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'`h` must be positive, got {h}')

    # Read input dimensions to fill in defaults and validate
    with av.open(video) as container:
        video_stream = container.streams.video[0]
        in_w = video_stream.width
        in_h = video_stream.height

    out_w = w if w is not None else in_w
    out_h = h if h is not None else in_h

    if out_w > in_w or out_h > in_h:
        raise pxt.RequestError(
            pxt.ErrorCode.UNSUPPORTED_OPERATION,
            f'viewport ({out_w}x{out_h}) must not exceed input dimensions ({in_w}x{in_h})',
        )
    if out_w == in_w and out_h == in_h:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f'viewport ({out_w}x{out_h}) equals input dimensions; at least one must be smaller for scrolling',
        )

    x_max = in_w - out_w
    y_max = in_h - out_h
    if x_start < 0 or x_start > x_max:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'x_start must be between 0 and {x_max}, got {x_start}')
    if y_start < 0 or y_start > y_max:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'y_start must be between 0 and {y_max}, got {y_start}')

    # Build the crop filter with time-dependent x/y expressions and edge clamping.
    # Example for w=1280 on a 1920-wide input, x_start=0, x_speed=64:
    # crop=1280:1080:min(640\,max(0\,0+64*t)):0
    x_expr = f'min({x_max}\\,max(0\\,{x_start}+{x_speed}*t))'
    y_expr = f'min({y_max}\\,max(0\\,{y_start}+{y_speed}*t))'
    crop_filter = f'crop={out_w}:{out_h}:{x_expr}:{y_expr}'

    cmd = ['-i', str(video), '-vf', crop_filter, '-c:a', 'copy']
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.expr_udf(is_method=True)
def pan(video: pxt.Video, x_sign: int = 0, y_sign: int = 0, crop_pct: float = 0.2) -> pxt.Video:
    """
    Apply a smooth pan effect across a video. Convenience wrapper around
    [`scroll()`][pixeltable.functions.video.scroll] that computes the viewport size, start position, and speed
    from the video's dimensions and duration so the viewport pans across the full available range over the
    clip's duration.

    - `x_sign = +1`: pan rightward (viewport starts at the left edge, moves right)
    - `x_sign = -1`: pan leftward (viewport starts at the right edge, moves left)
    - `x_sign =  0`: no horizontal motion (full width, no horizontal crop)
    - `y_sign` works the same way on the vertical axis (`+1` = down, `-1` = up, `0` = none)

    Diagonal pans are produced by passing nonzero values for both axes (e.g. `x_sign=+1, y_sign=-1` pans
    toward the upper-right). At least one of `x_sign` / `y_sign` must be nonzero, otherwise `scroll()`
    raises an error.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        x_sign: Horizontal pan direction: `+1` (right), `-1` (left), or `0` (no horizontal motion). Can be a
            column expression for per-row direction.
        y_sign: Vertical pan direction: `+1` (down), `-1` (up), or `0` (no vertical motion). Can be a
            column expression for per-row direction.
        crop_pct: Fraction of the dimension used as panning range, between 0.0 (exclusive) and 1.0
            (exclusive). Larger values produce more pronounced panning but a more aggressive crop.
            Default is 0.2 (viewport is 80% of the original dimension on the panning axis).

    Returns:
        A panned video.

    Examples:
        Pan rightward:

        >>> tbl.select(tbl.video.pan(x_sign=+1)).collect()

        Pan leftward with a wider crop:

        >>> tbl.select(tbl.video.pan(x_sign=-1, crop_pct=0.4)).collect()

        Pan diagonally toward the upper-right:

        >>> tbl.select(tbl.video.pan(x_sign=+1, y_sign=-1)).collect()

        Per-row direction driven by an `Int` column with values in {-1, 0, +1}:

        >>> tbl.add_computed_column(clip=tbl.video.pan(x_sign=tbl.pan_sign))
    """
    md = video.get_metadata()  # type: ignore[attr-defined]
    w = md.streams[0].width
    h = md.streams[0].height
    duration = video.get_duration()  # type: ignore[attr-defined]

    # abs(sign) collapses the crop to 0 when sign=0, so the unpanned axis stays at full size
    viewport_w = pxt_floor(w * (1 - crop_pct * pxt_abs(x_sign))).to_int()
    viewport_h = pxt_floor(h * (1 - crop_pct * pxt_abs(y_sign))).to_int()
    pan_range_x = w - viewport_w
    pan_range_y = h - viewport_h
    x_start = pxt_floor(pan_range_x * (1 - x_sign) / 2).to_int()
    y_start = pxt_floor(pan_range_y * (1 - y_sign) / 2).to_int()
    x_speed = pan_range_x / duration * x_sign
    y_speed = pan_range_y / duration * y_sign

    return scroll(  # type: ignore[return-value]
        video, w=viewport_w, h=viewport_h, x_speed=x_speed, y_speed=y_speed, x_start=x_start, y_start=y_start
    )


@pxt.udf(is_method=True, run_in_thread=True)
def zoom(
    video: pxt.Video,
    *,
    start_scale: float = 1.0,
    end_scale: float = 1.3,
    center: list[float] | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Apply a smooth zoom effect over the duration of a video using ffmpeg's zoompan filter.

    The zoom factor interpolates linearly from `start_scale` to `end_scale`. The effect works by computing a crop
    region at each frame (centered on `center`) and scaling it back to the original resolution. Output dimensions
    match the input.

    - `start_scale < end_scale`: zoom in (frame progressively tightens)
    - `start_scale > end_scale`: zoom out (frame progressively widens)
    - `start_scale == end_scale`: static zoom (constant crop, no animation)

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        start_scale: Zoom factor at the start of the video. Must be >= 1.0.
        end_scale: Zoom factor at the end of the video. Must be >= 1.0.
        center: Zoom center as `[x, y]` in normalized coordinates (0.0 to 1.0), where `[0.5, 0.5]` is the frame
            center. If None, defaults to `[0.5, 0.5]`.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the zoom effect applied. Output resolution matches the input.

    Examples:
        Zoom in (default, 1.0x to 1.3x centered):

        >>> tbl.select(tbl.video.zoom()).collect()

        Zoom out from 2x to 1x:

        >>> tbl.select(tbl.video.zoom(start_scale=2.0, end_scale=1.0)).collect()

        Zoom in toward the upper-left quadrant:

        >>> tbl.select(tbl.video.zoom(end_scale=1.5, center=[0.25, 0.25])).collect()

        Static 1.5x zoom (no animation):

        >>> tbl.select(tbl.video.zoom(start_scale=1.5, end_scale=1.5)).collect()
    """
    Env.get().require_binary('ffmpeg')

    if start_scale < 1.0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'start_scale must be >= 1.0, got {start_scale}')
    if end_scale < 1.0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'end_scale must be >= 1.0, got {end_scale}')
    if center is not None and (len(center) != 2 or not all(0.0 <= c <= 1.0 for c in center)):
        raise pxt.RequestError(
            pxt.ErrorCode.UNSUPPORTED_OPERATION, f'center must be [x, y] with values in [0.0, 1.0], got {center}'
        )
    cx, cy = center if center is not None else [0.5, 0.5]

    with av.open(video) as container:
        video_stream = container.streams.video[0]
        in_w = video_stream.width
        in_h = video_stream.height
        if video_stream.average_rate is None or video_stream.average_rate == 0:
            raise pxt.RequestError(pxt.ErrorCode.INVALID_DATA_FORMAT, 'zoom(): could not determine video frame rate')
        fps = float(video_stream.average_rate)

    # zoompan evaluates z/x/y expressions per frame.
    # 'on' is the output frame number (0-based); we use it to interpolate the zoom factor linearly.
    # 'd=1' means each input frame produces exactly 1 output frame

    # Example for start_scale=1.0, end_scale=1.3, center=(0.5, 0.5), fps=25, 10s 1920x1080 video:
    #   zoompan=z='1.0+(1.3-1.0)*on/249':x='iw*0.5*(1-1/zoom)':y='ih*0.5*(1-1/zoom)'
    #           :d=1:s=1920x1080:fps=25
    duration = av_utils.get_video_duration(video)
    if duration is None:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_DATA_FORMAT, 'zoom(): could not determine video duration')
    total_frames = max(1, round(fps * duration))

    # z interpolates linearly from start_scale to end_scale over total_frames
    z_expr = f'{start_scale}+({end_scale}-{start_scale})*on/{max(1, total_frames - 1)}'

    # x/y position the crop so that the normalized center point (cx, cy) stays fixed.
    # For center (cx, cy): x = iw*cx*(1 - 1/zoom)
    x_expr = f'iw*{cx}*(1-1/zoom)'
    y_expr = f'ih*{cy}*(1-1/zoom)'

    zoompan_filter = f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d=1:s={in_w}x{in_h}:fps={fps}"

    cmd = ['-i', str(video), '-vf', zoompan_filter, '-c:a', 'copy']
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


__all__ = local_public_names(__name__, exclude=['abs', 'floor'])
