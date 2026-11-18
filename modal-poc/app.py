"""Videos and their frames; the UDFs under [tool.pixeltable.udf] in pyproject.toml run on Modal, the rest locally."""

# ruff: noqa: F821  # a model body refers to its own columns, and an iterator's, by bare name

import pixeltable as pxt
from pixeltable.functions import huggingface, vision
from pixeltable.functions.video import concat_videos, frame_iterator
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()

CLIP_MODEL = 'openai/clip-vit-base-patch32'
CAPTION_MODEL = 'Salesforce/blip-image-captioning-base'
DETECTION_MODEL = 'facebook/detr-resnet-50'


class Videos(TableModel, name='videos'):
    video_id = pxt.Column(type=pxt.Int, primary_key=True)
    video: pxt.Video
    excerpt = video.clip(start_time=0.0, duration=2.0)
    segments = video.segment_video(duration=5.0)


class Frames(TableModel, name='frames', base=Videos, iterator=frame_iterator(video=Videos.video, fps=1)):
    rotated = frame.rotate(90)
    caption = huggingface.image_captioning(frame, model_id=CAPTION_MODEL, model_kwargs={'max_length': 30})
    detections = huggingface.detr_for_object_detection(frame, model_id=DETECTION_MODEL, threshold=0.8)
    annotated = vision.bboxes_draw(frame, detections.boxes, labels=detections.label_text, width=3)
    __indexes__ = [pxt.EmbeddingIndex(frame, embedding=huggingface.clip.using(model_id=CLIP_MODEL), name='frame_ix')]  # noqa: RUF012


@pxt.query
def frames(video_id: int) -> pxt.Query:
    return (
        Frames.where(Frames.video_id == video_id)
        .order_by(Frames.pos)
        .select(Frames.pos, Frames.rotated, Frames.caption, Frames.detections, Frames.annotated)
    )


@pxt.query
def similar_frames(text: str, limit: int = 3) -> pxt.Query:
    sim = Frames.frame.similarity(string=text)
    return (
        Frames.order_by(sim, asc=False)
        .limit(limit)
        .select(Frames.video_id, Frames.pos, Frames.caption, Frames.annotated, sim=sim)
    )


@pxt.query
def with_excerpt(video_id: int) -> pxt.Query:
    return Videos.where(Videos.video_id == video_id).select(joined=concat_videos([Videos.video, Videos.excerpt]))


demo = FastAPIRouter(name='demo')
demo.add_insert_route(
    Videos,
    path='/videos',
    inputs=[Videos.video_id],
    uploadfile_inputs=['video'],
    outputs=[Videos.video_id, Videos.excerpt, Videos.segments],
)
demo.add_query_route(path='/frames', query=frames)
demo.add_query_route(path='/similar', query=similar_frames)
demo.add_query_route(path='/with-excerpt', query=with_excerpt)
