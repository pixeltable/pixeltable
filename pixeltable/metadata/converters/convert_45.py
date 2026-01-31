from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=_substitution_fn)


def _substitution_fn(key: str | None, value: Any) -> tuple[str | None, Any] | None:
    if not isinstance(value, dict):
        return None
    if 'iterator_class_fqn' in value:
        assert 'iterator_args' in value
        iterator_class_fqn: str = value['iterator_class_fqn']
        iterator_args: dict = value['iterator_args']
        del value['iterator_class_fqn']
        del value['iterator_args']
        if iterator_class_fqn is None:
            assert iterator_args is None
            value['iterator_call'] = None
        else:
            assert iterator_args['_classname'] == 'InlineDict'  # Versions <= 45 stored args as InlineDict
            kwargs = dict(zip(iterator_args['keys'], iterator_args['components']))
            match iterator_class_fqn:
                case 'pixeltable.iterators.audio.AudioSplitter':
                    iterator_class_fqn = 'pixeltable.functions.audio.audio_splitter'
                    if 'chunk_duration_sec' in kwargs:
                        kwargs['segment_duration_sec'] = kwargs.pop('chunk_duration_sec')
                    if 'min_chunk_duration_sec' in kwargs:
                        kwargs['min_segment_duration_sec'] = kwargs.pop('min_chunk_duration_sec')
                case 'pixeltable.iterators.document.DocumentSplitter':
                    iterator_class_fqn = 'pixeltable.functions.document.document_splitter'
                case 'pixeltable.iterators.image.TileIterator':
                    iterator_class_fqn = 'pixeltable.functions.image.tile_iterator'
                case 'pixeltable.iterators.string.StringSplitter':
                    iterator_class_fqn = 'pixeltable.functions.string.string_splitter'
                case 'pixeltable.iterators.video.FrameIterator':
                    iterator_class_fqn = 'pixeltable.functions.video.frame_iterator'
                    # 'all_frame_attrs' was replaced by 'use_legacy_schema'
                    afa_expr = kwargs.pop('all_frame_attrs', None)
                    if afa_expr is not None:
                        assert afa_expr['_classname'] == 'Literal'
                        assert isinstance(afa_expr['val'], bool)
                        afa_expr['val'] = not afa_expr['val']
                        kwargs['use_legacy_schema'] = afa_expr
                case 'pixeltable.iterators.video.VideoSplitter':
                    iterator_class_fqn = 'pixeltable.functions.video.video_splitter'

            value['iterator_call'] = {
                'fn': {'fqn': iterator_class_fqn},
                'args': [],  # Non-kwargs inputs were not supported in versions <= 45
                'kwargs': kwargs,
                'bound_args': {},
                'output_schema': {},
                'col_mapping': {},
            }

    return key, value
