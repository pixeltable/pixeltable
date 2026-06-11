"""Media search schema — idempotent by default.

python schema.py
RESET_SCHEMA=true python schema.py
"""

import os

import config

import pixeltable as pxt
from pixeltable.functions.huggingface import clip
from pixeltable.functions.video import frame_iterator

if os.getenv('RESET_SCHEMA', 'false').lower() == 'true':
    pxt.drop_dir(config.APP_NAMESPACE, force=True)

pxt.create_dir(config.APP_NAMESPACE, if_exists='ignore')

video_table = pxt.create_table(
    f'{config.APP_NAMESPACE}.videos',
    {'video': pxt.Video},
    if_exists='ignore',
)

frames_view = pxt.create_view(
    f'{config.APP_NAMESPACE}.frames',
    video_table,
    iterator=frame_iterator(video=video_table.video, fps=1),
    if_exists='ignore',
)

clip_embed = clip.using(model_id=config.CLIP_MODEL_ID)

frames_view.add_embedding_index('frame', embedding=clip_embed, if_exists='ignore')

image_table = pxt.create_table(
    f'{config.APP_NAMESPACE}.images',
    {'image': pxt.Image, 'tags': pxt.Json},
    if_exists='ignore',
)

image_table.add_embedding_index('image', embedding=clip_embed, if_exists='ignore')

if __name__ == '__main__':
    print('Schema setup complete.')
