from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict

import numpy as np

import pixeltable as pxt
import pixeltable.utils.av as av_utils
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from scenedetect.detectors import SceneDetector  # type: ignore[import-untyped]


class SceneInfo(TypedDict):
    """One scene detected within a video: where it starts and how long it lasts."""

    start_time: float
    """Scene start, in seconds from the start of the video."""
    start_pts: int
    """Scene start, as a presentation timestamp (in the video stream's time base)."""
    duration: float
    """Scene duration, in seconds."""


@pxt.udf(is_method=True, resource_pool='cpu-bound')
def scene_detect_adaptive(
    video: pxt.Video,
    *,
    fps: float | None = None,
    adaptive_threshold: float = 3.0,
    min_scene_len: int = 15,
    window_width: int = 2,
    min_content_val: float = 15.0,
    delta_hue: float = 1.0,
    delta_sat: float = 1.0,
    delta_lum: float = 1.0,
    delta_edges: float = 0.0,
    luma_only: bool = False,
    kernel_size: int | None = None,
) -> list[SceneInfo]:
    """
    Detect scene cuts in a video using PySceneDetect's
    [AdaptiveDetector](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.adaptive_detector.AdaptiveDetector).

    __Requirements:__

    - `pip install scenedetect`

    Args:
        video: The video to analyze for scene cuts.
        fps: Number of frames to extract per second for analysis. If None or 0, analyzes all frames.
            Lower values process faster but may miss exact scene cuts.
        adaptive_threshold: Threshold that the score ratio must exceed to trigger a new scene cut.
            Lower values will detect more scenes (more sensitive), higher values will detect fewer scenes.
        min_scene_len: Once a cut is detected, this many frames must pass before a new one can be added to the scene
            list.
        window_width: Size of window (number of frames) before and after each frame to average together in order to
            detect deviations from the mean. Must be at least 1.
        min_content_val: Minimum threshold (float) that the content_val must exceed in order to register as a new scene.
            This is calculated the same way that `scene_detect_content()` calculates frame
            score based on weights/luma_only/kernel_size.
        delta_hue: Weight for hue component changes. Higher values make hue changes more important.
        delta_sat: Weight for saturation component changes. Higher values make saturation changes more important.
        delta_lum: Weight for luminance component changes. Higher values make brightness changes more important.
        delta_edges: Weight for edge detection changes. Higher values make edge changes more important.
            Edge detection can help detect cuts in scenes with similar colors but different content.
        luma_only: If True, only analyzes changes in the luminance (brightness) channel of the video,
            ignoring color information. This can be faster and may work better for grayscale content.
        kernel_size: Size of kernel to use for post edge detection filtering. If None, automatically set based on video
            resolution.

    Returns:
        A list of dictionaries, one for each detected scene, with the following keys:

        - `start_time` (float): The start time of the scene in seconds.
        - `start_pts` (int): The pts of the start of the scene.
        - `duration` (float): The duration of the scene in seconds.

        The list is ordered chronologically. Returns the full duration of the video if no scenes are detected.

    Examples:
        Detect scene cuts with default parameters:

        >>> tbl.select(tbl.video.scene_detect_adaptive()).collect()

        Detect more scenes by lowering the threshold:

        >>> tbl.select(
        ...     tbl.video.scene_detect_adaptive(adaptive_threshold=1.5)
        ... ).collect()

        Use luminance-only detection with a longer minimum scene length:

        >>> tbl.select(
        ...     tbl.video.scene_detect_adaptive(luma_only=True, min_scene_len=30)
        ... ).collect()

        Add scene cuts as a computed column:

        >>> tbl.add_computed_column(
        ...     scene_cuts=tbl.video.scene_detect_adaptive(adaptive_threshold=2.0)
        ... )

        Analyze at a lower frame rate for faster processing:

        >>> tbl.select(tbl.video.scene_detect_adaptive(fps=2.0)).collect()
    """
    Env.get().require_package('scenedetect')
    from scenedetect.detectors import AdaptiveDetector, ContentDetector

    weights = ContentDetector.Components(
        delta_hue=delta_hue, delta_sat=delta_sat, delta_lum=delta_lum, delta_edges=delta_edges
    )
    try:
        detector = AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_len,
            window_width=window_width,
            min_content_val=min_content_val,
            weights=weights,
            luma_only=luma_only,
            kernel_size=kernel_size,
        )
        return _scene_detect(video, fps, detector)
    except Exception as e:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_DATA_FORMAT, f'scene_detect_adaptive(): failed to detect scenes: {e}'
        ) from e


@pxt.udf(is_method=True, resource_pool='cpu-bound')
def scene_detect_content(
    video: pxt.Video,
    *,
    fps: float | None = None,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    delta_hue: float = 1.0,
    delta_sat: float = 1.0,
    delta_lum: float = 1.0,
    delta_edges: float = 0.0,
    luma_only: bool = False,
    kernel_size: int | None = None,
    filter_mode: Literal['merge', 'suppress'] = 'merge',
) -> list[SceneInfo]:
    """
    Detect scene cuts in a video using PySceneDetect's
    [ContentDetector](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.content_detector.ContentDetector).

    __Requirements:__

    - `pip install scenedetect`

    Args:
        video: The video to analyze for scene cuts.
        fps: Number of frames to extract per second for analysis. If None, analyzes all frames.
            Lower values process faster but may miss exact scene cuts.
        threshold: Threshold that the weighted sum of component changes must exceed to trigger a scene cut.
            Lower values detect more scenes (more sensitive), higher values detect fewer scenes.
        min_scene_len: Once a cut is detected, this many frames must pass before a new one can be added to the scene
            list.
        delta_hue: Weight for hue component changes. Higher values make hue changes more important.
        delta_sat: Weight for saturation component changes. Higher values make saturation changes more important.
        delta_lum: Weight for luminance component changes. Higher values make brightness changes more important.
        delta_edges: Weight for edge detection changes. Higher values make edge changes more important.
            Edge detection can help detect cuts in scenes with similar colors but different content.
        luma_only: If True, only analyzes changes in the luminance (brightness) channel,
            ignoring color information. This can be faster and may work better for grayscale content.
        kernel_size: Size of kernel for expanding detected edges. Must be odd integer greater than or equal to 3. If
            None, automatically set using video resolution.
        filter_mode: How to handle fast cuts/flashes. 'merge' combines quick cuts, 'suppress' filters them out.

    Returns:
        A list of dictionaries, one for each detected scene, with the following keys:

        - `start_time` (float): The start time of the scene in seconds.
        - `start_pts` (int): The pts of the start of the scene.
        - `duration` (float): The duration of the scene in seconds.

        The list is ordered chronologically. Returns the full duration of the video if no scenes are detected.

    Examples:
        Detect scene cuts with default parameters:

        >>> tbl.select(tbl.video.scene_detect_content()).collect()

        Detect more scenes by lowering the threshold:

        >>> tbl.select(tbl.video.scene_detect_content(threshold=15.0)).collect()

        Use luminance-only detection:

        >>> tbl.select(tbl.video.scene_detect_content(luma_only=True)).collect()

        Emphasize edge detection for scenes with similar colors:

        >>> tbl.select(
        ...     tbl.video.scene_detect_content(
        ...         delta_edges=1.0, delta_hue=0.5, delta_sat=0.5
        ...     )
        ... ).collect()

        Add scene cuts as a computed column:

        >>> tbl.add_computed_column(
        ...     scene_cuts=tbl.video.scene_detect_content(threshold=20.0)
        ... )
    """
    Env.get().require_package('scenedetect')
    from scenedetect.detectors import ContentDetector
    from scenedetect.detectors.content_detector import FlashFilter  # type: ignore[import-untyped]

    weights = ContentDetector.Components(
        delta_hue=delta_hue, delta_sat=delta_sat, delta_lum=delta_lum, delta_edges=delta_edges
    )
    filter_mode_enum = FlashFilter.Mode.MERGE if filter_mode == 'merge' else FlashFilter.Mode.SUPPRESS

    try:
        detector = ContentDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
            weights=weights,
            luma_only=luma_only,
            kernel_size=kernel_size,
            filter_mode=filter_mode_enum,
        )
        return _scene_detect(video, fps, detector)
    except Exception as e:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_DATA_FORMAT, f'scene_detect_content(): failed to detect scenes: {e}'
        ) from e


@pxt.udf(is_method=True, resource_pool='cpu-bound')
def scene_detect_threshold(
    video: pxt.Video,
    *,
    fps: float | None = None,
    threshold: float = 12.0,
    min_scene_len: int = 15,
    fade_bias: float = 0.0,
    add_final_scene: bool = False,
    method: Literal['ceiling', 'floor'] = 'floor',
) -> list[SceneInfo]:
    """
    Detect fade-in and fade-out transitions in a video using PySceneDetect's
    [ThresholdDetector](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.threshold_detector.ThresholdDetector).

    ThresholdDetector identifies scenes by detecting when pixel brightness falls below or rises above
    a threshold value, suitable for detecting fade-to-black, fade-to-white, and similar transitions.

    __Requirements:__

    - `pip install scenedetect`

    Args:
        video: The video to analyze for fade transitions.
        fps: Number of frames to extract per second for analysis. If None or 0, analyzes all frames.
            Lower values process faster but may miss exact transition points.
        threshold: 8-bit intensity value that each pixel value (R, G, and B) must be less than or equal to in order
            to trigger a fade in/out.
        min_scene_len: Once a cut is detected, this many frames must pass before a new one can be added to the scene
            list.
        fade_bias: Float between -1.0 and +1.0 representing the percentage of timecode skew for the start of a scene
            (-1.0 causing a cut at the fade-to-black, 0.0 in the middle, and +1.0 causing the cut to be right at the
            position where the threshold is passed).
        add_final_scene: Boolean indicating if the video ends on a fade-out to generate an additional scene at this
            timecode.
        method: How to treat threshold when detecting fade events
            - 'ceiling': Fade out happens when frame brightness rises above threshold.
            - 'floor': Fade out happens when frame brightness falls below threshold.


    Returns:
        A list of dictionaries, one for each detected scene, with the following keys:

        - `start_time` (float): The start time of the scene in seconds.
        - `start_pts` (int): The pts of the start of the scene.
        - `duration` (float): The duration of the scene in seconds.

        The list is ordered chronologically. Returns the full duration of the video if no scenes are detected.

    Examples:
        Detect fade-to-black transitions with default parameters:

        >>> tbl.select(tbl.video.scene_detect_threshold()).collect()

        Use a lower threshold to detect darker fades:

        >>> tbl.select(tbl.video.scene_detect_threshold(threshold=8.0)).collect()

        Detect fade-to-white transitions using the 'ceiling' method:

        >>> tbl.select(tbl.video.scene_detect_threshold(method='ceiling')).collect()

        Add final scene boundary:

        >>> tbl.select(
        ...     tbl.video.scene_detect_threshold(add_final_scene=True)
        ... ).collect()

        Add fade transitions as a computed column:

        >>> tbl.add_computed_column(
        ...     fade_cuts=tbl.video.scene_detect_threshold(threshold=15.0)
        ... )
    """
    Env.get().require_package('scenedetect')
    from scenedetect.detectors import ThresholdDetector

    method_enum = ThresholdDetector.Method.FLOOR if method == 'floor' else ThresholdDetector.Method.CEILING
    try:
        detector = ThresholdDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
            fade_bias=fade_bias,
            add_final_scene=add_final_scene,
            method=method_enum,
        )
        return _scene_detect(video, fps, detector)
    except Exception as e:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_DATA_FORMAT, f'scene_detect_threshold(): failed to detect scenes: {e}'
        ) from e


@pxt.udf(is_method=True, resource_pool='cpu-bound')
def scene_detect_histogram(
    video: pxt.Video, *, fps: float | None = None, threshold: float = 0.05, bins: int = 256, min_scene_len: int = 15
) -> list[SceneInfo]:
    """
    Detect scene cuts in a video using PySceneDetect's
    [HistogramDetector](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.histogram_detector.HistogramDetector).

    HistogramDetector compares frame histograms on the Y (luminance) channel after YUV conversion.
    It detects scenes based on relative histogram differences and is more robust to gradual lighting
    changes than content-based detection.

    __Requirements:__

    - `pip install scenedetect`

    Args:
        video: The video to analyze for scene cuts.
        fps: Number of frames to extract per second for analysis. If None or 0, analyzes all frames.
            Lower values process faster but may miss exact scene cuts.
        threshold: Maximum relative difference between 0.0 and 1.0 that the histograms can differ. Histograms are
            calculated on the Y channel after converting the frame to YUV, and normalized based on the number of bins.
            Higher differences imply greater change in content, so larger threshold values are less sensitive to cuts.
            Lower values detect more scenes (more sensitive), higher values detect fewer scenes.
        bins: Number of bins to use for histogram calculation (typically 16-256). More bins provide
            finer granularity but may be more sensitive to noise.
        min_scene_len: Once a cut is detected, this many frames must pass before a new one can be added to the scene
            list.


    Returns:
        A list of dictionaries, one for each detected scene, with the following keys:

        - `start_time` (float): The start time of the scene in seconds.
        - `start_pts` (int): The pts of the start of the scene.
        - `duration` (float): The duration of the scene in seconds.

        The list is ordered chronologically. Returns the full duration of the video if no scenes are detected.

    Examples:
        Detect scene cuts with default parameters:

        >>> tbl.select(tbl.video.scene_detect_histogram()).collect()

        Detect more scenes by lowering the threshold:

        >>> tbl.select(tbl.video.scene_detect_histogram(threshold=0.03)).collect()

        Use fewer bins for faster processing:

        >>> tbl.select(tbl.video.scene_detect_histogram(bins=64)).collect()

        Use with a longer minimum scene length:

        >>> tbl.select(tbl.video.scene_detect_histogram(min_scene_len=30)).collect()

        Add scene cuts as a computed column:

        >>> tbl.add_computed_column(
        ...     scene_cuts=tbl.video.scene_detect_histogram(threshold=0.04)
        ... )
    """
    Env.get().require_package('scenedetect')
    from scenedetect.detectors import HistogramDetector

    try:
        detector = HistogramDetector(threshold=threshold, bins=bins, min_scene_len=min_scene_len)
        return _scene_detect(video, fps, detector)
    except Exception as e:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_DATA_FORMAT, f'scene_detect_histogram(): failed to detect scenes: {e}'
        ) from e


@pxt.udf(is_method=True, resource_pool='cpu-bound')
def scene_detect_hash(
    video: pxt.Video,
    *,
    fps: float | None = None,
    threshold: float = 0.395,
    size: int = 16,
    lowpass: int = 2,
    min_scene_len: int = 15,
) -> list[SceneInfo]:
    """
    Detect scene cuts in a video using PySceneDetect's
    [HashDetector](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.hash_detector.HashDetector).

    HashDetector uses perceptual hashing for very fast scene detection. It computes a hash of each
    frame at reduced resolution and compares hash distances.

    __Requirements:__

    - `pip install scenedetect`

    Args:
        video: The video to analyze for scene cuts.
        fps: Number of frames to extract per second for analysis. If None, analyzes all frames.
            Lower values process faster but may miss exact scene cuts.
        threshold: Value from 0.0 and 1.0 representing the relative hamming distance between the perceptual hashes of
            adjacent frames. A distance of 0 means the image is the same, and 1 means no correlation. Smaller threshold
            values thus require more correlation, making the detector more sensitive. The Hamming distance is divided
            by size x size before comparing to threshold for normalization.
            Lower values detect more scenes (more sensitive), higher values detect fewer scenes.
        size: Size of square of low frequency data to use for the DCT. Larger values are more precise but slower.
            Common values are 8, 16, or 32.
        lowpass: How much high frequency information to filter from the DCT. A value of 2 means keep lower 1/2 of the
            frequency data, 4 means only keep 1/4, etc. Larger values make the
            detector less sensitive to high-frequency details and noise.
        min_scene_len: Once a cut is detected, this many frames must pass before a new one can be added to the scene
            list.


    Returns:
        A list of dictionaries, one for each detected scene, with the following keys:

        - `start_time` (float): The start time of the scene in seconds.
        - `start_pts` (int): The pts of the start of the scene.
        - `duration` (float): The duration of the scene in seconds.

        The list is ordered chronologically. Returns the full duration of the video if no scenes are detected.

    Examples:
        Detect scene cuts with default parameters:

        >>> tbl.select(tbl.video.scene_detect_hash()).collect()

        Detect more scenes by lowering the threshold:

        >>> tbl.select(tbl.video.scene_detect_hash(threshold=0.3)).collect()

        Use larger hash size for more precision:

        >>> tbl.select(tbl.video.scene_detect_hash(size=32)).collect()

        Use for fast processing with lower frame rate:

        >>> tbl.select(tbl.video.scene_detect_hash(fps=1.0, threshold=0.4)).collect()

        Add scene cuts as a computed column:

        >>> tbl.add_computed_column(scene_cuts=tbl.video.scene_detect_hash())
    """
    Env.get().require_package('scenedetect')
    from scenedetect.detectors import HashDetector

    try:
        detector = HashDetector(threshold=threshold, size=size, lowpass=lowpass, min_scene_len=min_scene_len)
        return _scene_detect(video, fps, detector)
    except Exception as e:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_DATA_FORMAT, f'scene_detect_hash(): failed to detect scenes: {e}'
        ) from e


class _SceneDetectFrameInfo(NamedTuple):
    frame_idx: int
    frame_pts: int
    frame_time: float


def _scene_detect(video: str, fps: float, detector: 'SceneDetector') -> list[SceneInfo]:
    from scenedetect import FrameTimecode  # type: ignore[import-untyped]

    with av_utils.VideoFrames(Path(video), fps=fps) as frame_iter:
        if frame_iter.video_framerate is None or frame_iter.video_framerate == 0:
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_DATA_FORMAT, f'scene_detect: could not determine video frame rate of {video!s}'
            )
        video_fps = float(frame_iter.video_framerate)

        scenes: list[SceneInfo] = []
        frame_idx: int | None = None
        start_time: float | None = None  # of current scene
        start_pts: int | None = None  # of current scene

        # in order to determine the cut frame times, we need to record frame times (chronologically) and look them
        # up by index; trying to derive frame times from frame indices isn't possible due to variable frame rates
        frame_info: list[_SceneDetectFrameInfo] = []

        def process_cuts(cuts: list[FrameTimecode]) -> None:
            nonlocal frame_info, start_time, start_pts
            for cut_timecode in cuts:
                cut_frame_idx = cut_timecode.get_frames()
                # we expect cuts to come back in chronological order
                assert cut_frame_idx >= frame_info[0].frame_idx
                info_offset = next((i for i, info in enumerate(frame_info) if info.frame_idx == cut_frame_idx), None)
                assert info_offset is not None  # the cut is at a previously reported frame idx
                info = frame_info[info_offset]
                scenes.append(
                    {'start_time': start_time, 'start_pts': start_pts, 'duration': info.frame_time - start_time}
                )
                start_time = info.frame_time
                start_pts = info.frame_pts
                frame_info = frame_info[info_offset + 1 :]

        for item in frame_iter:
            if start_time is None:
                start_time = item.time
                start_pts = item.pts
            frame_info.append(_SceneDetectFrameInfo(item.frame_idx, item.pts, item.time))
            frame_array = np.array(item.frame.convert('RGB'))
            frame_idx = item.frame_idx
            timecode = FrameTimecode(item.frame_idx, video_fps)
            cuts = detector.process_frame(timecode, frame_array)
            process_cuts(cuts)

        # Post-process to capture any final scene cuts
        if frame_idx is not None:
            final_timecode = FrameTimecode(frame_idx, video_fps)
            final_cuts = detector.post_process(final_timecode)
            process_cuts(final_cuts)

            # if we didn't detect any cuts but the video has content, add the full video as a single scene
            if len(scenes) == 0:
                scenes.append(
                    {
                        'start_time': start_time,
                        'start_pts': start_pts,
                        'duration': frame_info[-1].frame_time - start_time,
                    }
                )

        return scenes


__all__ = local_public_names(__name__)
