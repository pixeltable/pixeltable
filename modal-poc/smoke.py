"""Run some of the demo's UDFs on Modal, through computed columns and a query, and check the results.

README.md says how to run it.
"""

import collections
import logging
import os
import sys

import av
import numpy as np

import pixeltable as pxt
from pixeltable.functions import huggingface, video

CLIP_MODEL = 'openai/clip-vit-base-patch32'
# an image of a gas pump
IMAGE = '../tests/data/imagenette2-160/n03425413_12982.JPEG'
VIDEO = '../tests/data/videos/bangkok_half_res.mp4'


class ModalCallCounter(logging.Handler):
    """Counts the Modal calls per UDF path, from ModalServerClient.call()'s debug line."""

    counts: collections.Counter[str]

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.counts = collections.Counter()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith('Modal call of '):
            self.counts[message.removeprefix('Modal call of ').split(':', 1)[0]] += 1


def duration(path: str) -> float:
    with av.open(path) as container:
        assert container.duration is not None
        return container.duration / av.time_base


def main() -> None:
    if 'PIXELTABLE_HOME' not in os.environ:
        sys.exit('Set PIXELTABLE_HOME to a new directory; this script creates tables in it.')
    counter = ModalCallCounter()
    logger = logging.getLogger('pixeltable')
    logger.addHandler(counter)
    logger.setLevel(logging.DEBUG)

    t = pxt.create_table(
        'modal_smoke', {'text': pxt.String, 'image': pxt.Image, 'video': pxt.Video | None, 'start': pxt.Float}
    )
    # batched, with the text overload of the polymorphic clip()
    t.add_computed_column(text_emb=huggingface.clip(t.text, model_id=CLIP_MODEL))
    # batched, with an image file in and the image overload of clip()
    t.add_computed_column(image_emb=huggingface.clip(t.image, model_id=CLIP_MODEL))
    # one call per row, a video in and out
    t.add_computed_column(excerpt=video.clip(t.video, start_time=t.start, duration=2.0))
    status = t.insert(
        [
            {'text': 'a photo of a gas pump', 'image': IMAGE, 'video': VIDEO, 'start': -1.0},
            {'text': 'a photo of a cat', 'image': IMAGE, 'video': VIDEO, 'start': 1.0},
            {'text': 'a photo of a dog', 'image': IMAGE, 'video': None, 'start': 3.0},
        ],
        on_error='ignore',
    )
    assert status.num_excs == 1, status

    res = t.select(t.text_emb, t.image_emb, t.excerpt, error=t.excerpt.errortype).order_by(t.start).collect()
    # the same image in every row
    image_emb = res['image_emb'][0]
    assert all(np.array_equal(emb, image_emb) for emb in res['image_emb'])
    # CLIP matches the image to 'a photo of a gas pump'
    similarities = [
        np.dot(image_emb, emb) / (np.linalg.norm(image_emb) * np.linalg.norm(emb)) for emb in res['text_emb']
    ]
    assert np.argmax(similarities) == 0, similarities
    assert res['error'] == ['RequestError', None, None], res['error']
    assert res['excerpt'][2] is None
    assert abs(duration(res['excerpt'][1]) - 2.0) < 0.1

    # a query with videos nested in a list argument
    joined = t.where(t.start == 1.0).select(joined=video.concat_videos([t.video, t.excerpt])).collect()['joined'][0]
    assert abs(duration(joined) - (duration(VIDEO) + 2.0)) < 0.1

    expected_calls = {
        # one call per overload
        huggingface.clip.self_path: 2,
        # the row without a video makes no call
        video.clip.self_path: 2,
        video.concat_videos.self_path: 1,
    }
    assert dict(counter.counts) == expected_calls, dict(counter.counts)
    print(f'OK: Modal calls {dict(counter.counts)}')


if __name__ == '__main__':
    main()
