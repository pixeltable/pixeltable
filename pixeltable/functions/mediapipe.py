"""MediaPipe pose estimation functions."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import PIL.Image

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.func import Batch
from pixeltable.functions.util import normalize_image_mode
from pixeltable.utils import av as av_utils
from pixeltable.utils.code import local_public_names

_logger = logging.getLogger('pixeltable')

PoseModel = Literal['lite', 'full', 'heavy']

LANDMARK_NAMES: tuple[str, ...] = (
    'nose',
    'left_eye_inner',
    'left_eye',
    'left_eye_outer',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_ear',
    'right_ear',
    'mouth_left',
    'mouth_right',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_pinky',
    'right_pinky',
    'left_index',
    'right_index',
    'left_thumb',
    'right_thumb',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot_index',
    'right_foot_index',
)

POSE_MODEL_URLS: dict[PoseModel, str] = {
    'lite': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
    ),
    'full': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
    ),
    'heavy': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    ),
}


class Landmark(TypedDict):
    name: str
    x: float
    y: float
    z: float
    visibility: float


class PoseLandmarkerResponse(TypedDict):
    detected: bool
    landmarks: list[Landmark]


class PoseVideoFrame(TypedDict):
    pos: int
    timestamp_ms: int
    time: float
    pose: PoseLandmarkerResponse | None


def _models_dir() -> Path:
    return Env.get().dataset_cache_dir / 'mediapipe'


def _model_path(model: PoseModel) -> Path:
    return _models_dir() / f'pose_landmarker_{model}.task'


def _ensure_model(model: PoseModel) -> Path:
    dest = _model_path(model)
    if dest.exists():
        return dest
    url = POSE_MODEL_URLS[model]
    dest.parent.mkdir(parents=True, exist_ok=True)
    _logger.info('Downloading MediaPipe pose model %s to %s', model, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def _landmarks_from_result(pose_landmarks) -> PoseLandmarkerResponse:
    landmarks: list[Landmark] = []
    for i, lm in enumerate(pose_landmarks):
        name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f'landmark_{i}'
        landmarks.append(
            Landmark(
                name=name,
                x=round(lm.x, 4),
                y=round(lm.y, 4),
                z=round(lm.z, 4),
                visibility=round(lm.visibility, 4),
            )
        )
    return PoseLandmarkerResponse(detected=True, landmarks=landmarks)


def _pose_options(
    model: PoseModel,
    running_mode,
    *,
    min_detection_confidence: float,
    min_presence_confidence: float,
    min_tracking_confidence: float,
):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    return vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(_ensure_model(model))),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def _lookup_image_landmarker(
    model: PoseModel,
    *,
    min_detection_confidence: float,
    min_presence_confidence: float,
    min_tracking_confidence: float,
):
    from mediapipe.tasks.python import vision

    key = (model, min_detection_confidence, min_presence_confidence, min_tracking_confidence)
    if key not in _image_landmarker_cache:
        _image_landmarker_cache[key] = vision.PoseLandmarker.create_from_options(
            _pose_options(
                model,
                vision.RunningMode.IMAGE,
                min_detection_confidence=min_detection_confidence,
                min_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
    return _image_landmarker_cache[key]


def _detect_pose_image(
    image: PIL.Image.Image,
    *,
    model: PoseModel,
    min_detection_confidence: float,
    min_presence_confidence: float,
    min_tracking_confidence: float,
) -> PoseLandmarkerResponse | None:
    import mediapipe as mp

    landmarker = _lookup_image_landmarker(
        model,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    rgb = np.ascontiguousarray(np.array(image.convert('RGB')))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return PoseLandmarkerResponse(detected=False, landmarks=[])
    return _landmarks_from_result(result.pose_landmarks[0])


@pxt.udf(batch_size=8)
def pose_landmarker(
    images: Batch[PIL.Image.Image],
    *,
    model: PoseModel = 'full',
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Batch[PoseLandmarkerResponse | None]:
    """
    Computes body pose landmarks for the specified image(s) using MediaPipe's
    [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

    __Requirements__:

    - `pip install mediapipe`

    Args:
        images: Input image(s).
        model: Model variant to use: `lite`, `full`, or `heavy`.
        min_detection_confidence: Minimum confidence for pose detection.
        min_presence_confidence: Minimum confidence for pose presence.
        min_tracking_confidence: Minimum confidence for pose tracking.

    Returns:
        A dictionary with keys:

        - `detected` (`bool`): Whether a pose was detected.
        - `landmarks` (`list[dict]`): 33 body landmarks with normalized `x`, `y`, `z`,
          and `visibility` values, plus a human-readable `name`.

    Examples:
        Add a computed column that runs pose estimation on an existing image column:

        >>> tbl.add_computed_column(pose=pose_landmarker(tbl.image, model='full'))

        Use with a frame view over video (see the object-detection-in-videos cookbook):

        >>> frames.add_computed_column(pose=pose_landmarker(frames.frame, model='lite'))
    """
    Env.get().require_package('mediapipe')
    results: list[PoseLandmarkerResponse | None] = []
    for image in images:
        if image is None:
            results.append(None)
            continue
        normalized = normalize_image_mode(image)
        results.append(
            _detect_pose_image(
                normalized,
                model=model,
                min_detection_confidence=min_detection_confidence,
                min_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
    return results


@pxt.udf(is_method=True)
def pose_landmarker_video(
    video: pxt.Video,
    *,
    model: PoseModel = 'full',
    fps: float = 4.0,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> list[PoseVideoFrame]:
    """
    Computes sequential body pose landmarks for sampled frames of a video using MediaPipe's
    VIDEO running mode (temporal tracking within one landmarker session).

    __Requirements__:

    - `pip install mediapipe`

    Args:
        video: Input video.
        model: Model variant to use: `lite`, `full`, or `heavy`.
        fps: Number of frames to sample per second of video.
        min_detection_confidence: Minimum confidence for pose detection.
        min_presence_confidence: Minimum confidence for pose presence.
        min_tracking_confidence: Minimum confidence for pose tracking.

    Returns:
        A list of dictionaries, one per sampled frame, with keys:

        - `pos` (`int`): 1-based frame index in the sampled sequence.
        - `timestamp_ms` (`int`): Timestamp passed to MediaPipe for this frame.
        - `time` (`float`): Frame time in seconds.
        - `pose`: Pose result in the same format as `pose_landmarker()`.

    Examples:
        >>> tbl.add_computed_column(poses=tbl.video.pose_landmarker_video(fps=4.0))
    """
    Env.get().require_package('mediapipe')
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    video_path = Path(str(video))
    if not video_path.exists():
        raise pxt.Error(f'pose_landmarker_video(): video file not found: {video_path}')

    landmarker = vision.PoseLandmarker.create_from_options(
        _pose_options(
            model,
            vision.RunningMode.VIDEO,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    )

    results: list[PoseVideoFrame] = []
    last_ts = -1
    try:
        with av_utils.VideoFrames(video_path, fps=fps) as frame_iter:
            for pos, item in enumerate(frame_iter, start=1):
                timestamp_ms = round(item.time * 1000)
                if timestamp_ms <= last_ts:
                    timestamp_ms = last_ts + 1
                last_ts = timestamp_ms

                rgb = np.ascontiguousarray(np.array(item.frame.convert('RGB')))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection = landmarker.detect_for_video(mp_image, timestamp_ms)
                if not detection.pose_landmarks:
                    pose: PoseLandmarkerResponse | None = PoseLandmarkerResponse(detected=False, landmarks=[])
                else:
                    pose = _landmarks_from_result(detection.pose_landmarks[0])

                results.append(
                    PoseVideoFrame(
                        pos=pos,
                        timestamp_ms=timestamp_ms,
                        time=round(item.time, 4),
                        pose=pose,
                    )
                )
    finally:
        landmarker.close()

    return results


_image_landmarker_cache: dict[tuple[PoseModel, float, float, float], object] = {}


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
