"""Media columns, an iterator view over a video, and the routes that carry files.

The routes here are the ones whose request or response is not JSON: a file upload, a file response, and a
computation slow enough to want a background job.
"""

# ruff: noqa: F821  # a model body refers to its own columns, and an iterator's, by bare name

from __future__ import annotations

import pixeltable as pxt
from pixeltable.functions.video import frame_iterator
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


class Clips(TableModel, name='clips'):
    clip_id = pxt.Column(type=pxt.Int, primary_key=True)
    video: pxt.Video
    caption: pxt.String | None
    poster = video.extract_frame(timestamp=0.0)  # type: ignore[attr-defined]


class Frames(TableModel, name='frames', base=Clips, iterator=frame_iterator(video=Clips.video, fps=1)):
    """One row per extracted frame; `frame` and `pos` are the iterator's own columns."""

    thumb = frame.resize(size=(32, 32))  # type: ignore[name-defined]


class Recordings(TableModel, name='recordings'):
    recording_id = pxt.Column(type=pxt.Int, primary_key=True)
    audio: pxt.Audio
    transcript: pxt.Document
    audio_metadata = audio.get_metadata()  # type: ignore[attr-defined]


clips = FastAPIRouter(name='clips')

# the video arrives as multipart/form-data rather than as a URL in a JSON body
clips.add_insert_route(
    Clips,
    path='/clips',
    inputs=[Clips.clip_id, Clips.caption],  # type: ignore[arg-type]
    uploadfile_inputs=['video'],
    outputs=[Clips.clip_id],  # type: ignore[arg-type]
)

# one image per clip, returned as the image itself rather than as JSON: a file response carries a single
# media value, so it cannot come from a view that yields a row per frame
clips.add_compute_route(
    Clips,
    path='/poster',
    inputs=[Clips.clip_id, Clips.video],  # type: ignore[arg-type]
    outputs=[Clips.poster],
    return_fileresponse=True,
)

# the frames themselves come back as JSON, with the media rendered as urls
frames = FastAPIRouter(name='frames')
frames.add_compute_route(
    Frames,
    path='/frames',
    inputs=[Clips.clip_id, Clips.video],  # type: ignore[arg-type]
    outputs=[Frames.thumb],
)

# two files in one request, and a response that arrives before the work is done: the caller gets a job url
# to poll instead of the inserted row
recordings = FastAPIRouter(name='recordings')
recordings.add_insert_route(
    Recordings,
    path='/recordings',
    inputs=[Recordings.recording_id],  # type: ignore[arg-type]
    uploadfile_inputs=['audio', 'transcript'],
    outputs=[Recordings.recording_id, Recordings.audio_metadata],
    background=True,
)
