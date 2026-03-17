from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=46)
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
            outputs: dict[str, dict[str, Any]] | None = None

            match iterator_class_fqn:
                case 'pixeltable.iterators.audio.AudioSplitter':
                    iterator_class_fqn = 'pixeltable.functions.audio.audio_splitter'
                    if 'chunk_duration_sec' in kwargs:
                        kwargs['duration'] = kwargs.pop('chunk_duration_sec')
                    if 'overlap_sec' in kwargs:
                        kwargs['overlap'] = kwargs.pop('overlap_sec')
                    if 'min_chunk_duration_sec' in kwargs:
                        kwargs['min_segment_duration'] = kwargs.pop('min_chunk_duration_sec')
                    # For `audio_splitter`, output columns were renamed in v46, so we explicitly emit them here.
                    outputs = {
                        'pos': {
                            'orig_name': 'pos',
                            'is_stored': True,
                            'col_type': {'nullable': False, '_classname': 'IntType'},
                        },
                        'audio_chunk': {
                            'orig_name': 'audio_segment',
                            'is_stored': True,
                            'col_type': {'nullable': True, '_classname': 'AudioType'},
                        },
                        'start_time_sec': {
                            'orig_name': 'segment_start',
                            'is_stored': True,
                            'col_type': {'nullable': False, '_classname': 'FloatType'},
                        },
                        'end_time_sec': {
                            'orig_name': 'segment_end',
                            'is_stored': True,
                            'col_type': {'nullable': False, '_classname': 'FloatType'},
                        },
                    }
                case 'pixeltable.iterators.document.DocumentSplitter':
                    iterator_class_fqn = 'pixeltable.functions.document.document_splitter'
                case 'pixeltable.iterators.image.TileIterator':
                    iterator_class_fqn = 'pixeltable.functions.image.tile_iterator'
                case 'pixeltable.iterators.string.StringSplitter':
                    iterator_class_fqn = 'pixeltable.functions.string.string_splitter'
                case 'pixeltable.iterators.video.FrameIterator':
                    # 'all_frame_attrs' was replaced by 'use_legacy_schema'
                    afa_expr = kwargs.pop('all_frame_attrs', None)
                    if afa_expr is None:
                        use_legacy_schema = True
                    else:
                        assert afa_expr['_classname'] == 'Literal'
                        assert isinstance(afa_expr['val'], bool)
                        use_legacy_schema = not afa_expr['val']
                    if use_legacy_schema:
                        iterator_class_fqn = 'pixeltable.functions.video.legacy_frame_iterator'
                    else:
                        iterator_class_fqn = 'pixeltable.functions.video.frame_iterator'
                case 'pixeltable.iterators.video.VideoSplitter':
                    iterator_class_fqn = 'pixeltable.functions.video.video_splitter'

            value['iterator_call'] = {
                'fn': {'fqn': iterator_class_fqn},
                'args': [],  # Non-kwargs inputs were not supported in versions <= 45
                'kwargs': kwargs,
                'outputs': outputs,
            }

    return key, value
