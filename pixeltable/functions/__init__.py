import os
from typing import Callable, List, Optional, Union, Any
import inspect
from pathlib import Path
import tempfile

import PIL.Image
import numpy as np

from pixeltable.type_system import StringType, IntType, JsonType, ColumnType, FloatType, ImageType, VideoType
import pixeltable.func as func
from pixeltable import exprs
import pixeltable.env as env
# import all standard function modules here so they get registered with the FunctionRegistry
import pixeltable.functions.pil
import pixeltable.functions.pil.image
import av
import av.container
import av.stream


try:
    import openai
    from .util import create_openai_module
    mod = create_openai_module()
    func.FunctionRegistry.get().register_module(mod)
except ImportError:
    pass

try:
    import together
    from .util import create_together_module
    mod = create_together_module()
    func.FunctionRegistry.get().register_module(mod)
except ImportError:
    pass

def _str_format(format_str: str, *args, **kwargs: Any) -> str:
    """ Return a formatted version of S, using substitutions from args and kwargs.
    The substitutions are identified by braces ('{' and '}')."""
    return format_str.format(*args, **kwargs)
str_format = func.make_library_function(StringType(), [StringType()], __name__, '_str_format')
func.FunctionRegistry.get().register_function(__name__, 'str_format', str_format)

def cast(expr: exprs.Expr, target_type: ColumnType) -> exprs.Expr:
    expr.col_type = target_type
    return expr

dict_map = func.make_function(IntType(), [StringType(), JsonType()], lambda s, d: d[s])

class SumAggregator:
    def __init__(self):
        self.sum: Union[int, float] = 0
    @classmethod
    def make_aggregator(cls) -> 'SumAggregator':
        return cls()
    def update(self, val: Union[int, float]) -> None:
        if val is not None:
            self.sum += val
    def value(self) -> Union[int, float]:
        return self.sum

sum = func.make_library_aggregate_function(
    IntType(), [IntType()],
    'pixeltable.functions', 'SumAggregator.make_aggregator', 'SumAggregator.update', 'SumAggregator.value',
    allows_std_agg=True, allows_window=True)
func.FunctionRegistry.get().register_function(__name__, 'sum', sum)

class CountAggregator:
    def __init__(self):
        self.count = 0
    @classmethod
    def make_aggregator(cls) -> 'CountAggregator':
        return cls()
    def update(self, val: int) -> None:
        if val is not None:
            self.count += 1
    def value(self) -> int:
        return self.count

count = func.make_library_aggregate_function(
    IntType(), [IntType()],
    'pixeltable.functions', 'CountAggregator.make_aggregator', 'CountAggregator.update', 'CountAggregator.value',
    allows_std_agg = True, allows_window = True)
func.FunctionRegistry.get().register_function(__name__, 'count', count)

class MeanAggregator:
    def __init__(self):
        self.sum = 0
        self.count = 0
    @classmethod
    def make_aggregator(cls) -> 'MeanAggregator':
        return cls()
    def update(self, val: int) -> None:
        if val is not None:
            self.sum += val
            self.count += 1
    def value(self) -> float:
        if self.count == 0:
            return None
        return self.sum / self.count

mean = func.make_library_aggregate_function(
    FloatType(), [IntType()],
    'pixeltable.functions', 'MeanAggregator.make_aggregator', 'MeanAggregator.update', 'MeanAggregator.value',
    allows_std_agg = True, allows_window = True)
func.FunctionRegistry.get().register_function(__name__, 'mean', mean)

class VideoAggregator:
    def __init__(self):
        """follows https://pyav.org/docs/develop/cookbook/numpy.html#generating-video"""
        self.container : Optional[av.container.OutputContainer] = None
        self.stream : Optional[av.stream.Stream] = None
        self.fps : float = 25

    @classmethod
    def make_aggregator(cls) -> 'VideoAggregator':
        return cls()

    def update(self, frame: PIL.Image.Image) -> None:
        if frame is None:
            return
        if self.container is None:
            (_, output_filename) = tempfile.mkstemp(suffix='.mp4', dir=str(env.Env.get().tmp_dir))
            self.out_file = Path(output_filename)
            self.container = av.open(str(self.out_file), mode='w')
            self.stream = self.container.add_stream('h264', rate=self.fps)
            self.stream.pix_fmt = 'yuv420p'
            self.stream.width = frame.width
            self.stream.height = frame.height

        av_frame = av.VideoFrame.from_ndarray(np.array(frame.convert('RGB')), format='rgb24')
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)

    def value(self) -> str:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        return str(self.out_file)

make_video = func.make_library_aggregate_function(
    VideoType(), [ImageType()],  # params: frame
    module_name = 'pixeltable.functions',
    init_symbol = 'VideoAggregator.make_aggregator',
    update_symbol = 'VideoAggregator.update',
    value_symbol = 'VideoAggregator.value',
    requires_order_by=True, allows_std_agg=True, allows_window=False)
func.FunctionRegistry.get().register_function(__name__, 'make_video', make_video)

__all__ = [
    #udf_call,
    cast,
    dict_map,
    sum,
    count,
    mean,
    make_video
]
