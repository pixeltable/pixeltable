import tempfile
from pathlib import Path
from typing import Optional, Union

import PIL.Image
import av
import av.container
import av.stream
import numpy as np

import pixeltable.env as env
import pixeltable.func as func
# import all standard function modules here so they get registered with the FunctionRegistry
import pixeltable.functions.pil.image
from pixeltable import exprs
from pixeltable.type_system import IntType, ColumnType, FloatType, ImageType, VideoType
# automatically import all submodules so that the udfs get registered
from . import image, string, video, huggingface

# TODO: remove and replace calls with astype()
def cast(expr: exprs.Expr, target_type: ColumnType) -> exprs.Expr:
    expr.col_type = target_type
    return expr

@func.uda(
    update_types=[IntType()], value_type=IntType(), allows_window=True, requires_order_by=False)
class sum(func.Aggregator):
    def __init__(self):
        self.sum: Union[int, float] = 0
    def update(self, val: Union[int, float]) -> None:
        if val is not None:
            self.sum += val
    def value(self) -> Union[int, float]:
        return self.sum


@func.uda(
    update_types=[IntType()], value_type=IntType(), allows_window = True, requires_order_by = False)
class count(func.Aggregator):
    def __init__(self):
        self.count = 0
    def update(self, val: int) -> None:
        if val is not None:
            self.count += 1
    def value(self) -> int:
        return self.count


@func.uda(
    update_types=[IntType()], value_type=FloatType(), allows_window=False, requires_order_by=False)
class mean(func.Aggregator):
    def __init__(self):
        self.sum = 0
        self.count = 0
    def update(self, val: int) -> None:
        if val is not None:
            self.sum += val
            self.count += 1
    def value(self) -> float:
        if self.count == 0:
            return None
        return self.sum / self.count


@func.uda(
    init_types=[IntType()], update_types=[ImageType()], value_type=VideoType(),
    requires_order_by=True, allows_window=False)
class make_video(func.Aggregator):
    def __init__(self, fps: int = 25):
        """follows https://pyav.org/docs/develop/cookbook/numpy.html#generating-video"""
        self.container: Optional[av.container.OutputContainer] = None
        self.stream: Optional[av.stream.Stream] = None
        self.fps = fps

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
