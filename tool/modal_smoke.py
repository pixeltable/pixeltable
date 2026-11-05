"""Run UDFs on Modal through computed columns and a query, and check the results.

Run it from the repo root, whose uv.lock supplies the packages of the Modal image, with a new Pixeltable home:

    PIXELTABLE_HOME=$(mktemp -d) MODAL_PROXY_KEY=wk-... MODAL_PROXY_SECRET=ws-... python tool/modal_smoke.py

It needs a Modal token (`modal token new`) and a Modal proxy token (`modal workspace proxy-tokens create`).
"""

import collections
import logging
import os
import sys

import av
import numpy as np

import pixeltable as pxt
from pixeltable.functions import huggingface, video

ST_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CLIP_MODEL = 'openai/clip-vit-base-patch32'
IMAGE = 'tests/data/imagenette2-160/n03425413_12982.JPEG'
VIDEO = 'tests/data/videos/bangkok_half_res.mp4'


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
    pxt.init(
        config_overrides={
            'pixeltable.udf_resources': {
                'pixeltable.functions.huggingface.sentence_transformer': {'cpu': 2},
                'pixeltable.functions.huggingface.clip': {'cpu': 2},
                'pixeltable.functions.video.clip': {'cpu': 2},
                'pixeltable.functions.video.concat_videos': {'cpu': 2},
            }
        }
    )
    counter = ModalCallCounter()
    logger = logging.getLogger('pixeltable')
    logger.addHandler(counter)
    logger.setLevel(logging.DEBUG)

    t = pxt.create_table(
        'modal_smoke', {'text': pxt.String, 'image': pxt.Image, 'video': pxt.Video | None, 'start': pxt.Float}
    )
    # batched, text in and array out
    t.add_computed_column(text_emb=huggingface.sentence_transformer(t.text, model_id=ST_MODEL))
    # batched, an image file in, and one overload of a polymorphic UDF
    t.add_computed_column(image_emb=huggingface.clip(t.image, model_id=CLIP_MODEL))
    # one call per row, a video in and out
    t.add_computed_column(excerpt=video.clip(t.video, start_time=t.start, duration=2.0))
    status = t.insert(
        [
            {'text': 'a bad start time', 'image': IMAGE, 'video': VIDEO, 'start': -1.0},
            {'text': 'a photo of a cat', 'image': IMAGE, 'video': VIDEO, 'start': 1.0},
            {'text': 'no video', 'image': IMAGE, 'video': None, 'start': 3.0},
        ],
        on_error='ignore',
    )
    assert status.num_excs == 1, status

    res = t.select(t.text, t.text_emb, t.image_emb, t.excerpt, error=t.excerpt.errortype).order_by(t.start).collect()
    expected = huggingface.sentence_transformer.exec_batch([res['text']], {'model_id': [ST_MODEL] * len(res)})
    assert all(np.allclose(r, e, atol=1e-5) for r, e in zip(res['text_emb'], expected)), 'text_emb differs from local'
    # the same image in every row
    assert all(np.array_equal(emb, res['image_emb'][0]) for emb in res['image_emb'])
    assert res['error'] == ['RequestError', None, None], res['error']
    assert res['excerpt'][2] is None
    assert abs(duration(res['excerpt'][1]) - 2.0) < 0.1

    # a query with videos nested in a list argument
    joined = t.where(t.start == 1.0).select(joined=video.concat_videos([t.video, t.excerpt])).collect()['joined'][0]
    assert abs(duration(joined) - (duration(VIDEO) + 2.0)) < 0.1

    expected_calls = {
        huggingface.sentence_transformer.self_path: 1,
        huggingface.clip.self_path: 1,
        # the row without a video makes no call
        video.clip.self_path: 2,
        video.concat_videos.self_path: 1,
    }
    assert dict(counter.counts) == expected_calls, dict(counter.counts)
    print(f'OK: Modal calls {dict(counter.counts)}')


if __name__ == '__main__':
    main()
