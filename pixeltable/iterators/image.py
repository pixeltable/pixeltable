from typing import Any

import PIL.Image

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.iterators.base import ComponentIterator


class TileIterator(ComponentIterator):
    def __init__(
        self,
        image: PIL.Image.Image,
        *,
        tile_size: tuple[int, int],
        overlap: tuple[int, int] = (0, 0),
    ):
        if overlap[0] >= tile_size[0] or overlap[1] >= tile_size[1]:
            raise excs.Error(f"overlap dimensions {overlap} are not strictly smaller than tile size {tile_size}")

        self.__image = image
        self.__image.load()
        self.__tile_size = tile_size
        self.__overlap = overlap
        self.__width, self.__height = image.size
        # Justification for this formula: let t = tile_size[0], o = overlap[0]. Then the values of w (= width) that
        # exactly accommodate an integer number of tiles are t, 2t - o, 3t - 2o, 4t - 3o, ...
        # This formula ensures that t, 2t - o, 3t - 2o, ... result in an xlen of 1, 2, 3, ...
        # but t + 1, 2t - o + 1, 3t - 2o + 1, ... result in an xlen of 2, 3, 4, ...
        self.__xlen = (self.__width - overlap[0] - 1) // (tile_size[0] - overlap[0]) + 1
        self.__ylen = (self.__height - overlap[1] - 1) // (tile_size[1] - overlap[1]) + 1
        self.__i = 0
        self.__j = 0

    def __next__(self) -> dict[str, Any]:
        if self.__j >= self.__ylen:
            raise StopIteration

        x1 = self.__i * (self.__tile_size[0] - self.__overlap[0])
        y1 = self.__j * (self.__tile_size[1] - self.__overlap[1])
        # If x2 > self.__width, PIL does the right thing and pads the image with blackspace
        x2 = x1 + self.__tile_size[0]
        y2 = y1 + self.__tile_size[1]
        tile = self.__image.crop((x1, y1, x2, y2))
        result = {
            'tile': tile,
            'tile_coord': [self.__i, self.__j],
            'tile_box': [x1, y1, x2, y2]
        }

        self.__i += 1
        if self.__i >= self.__xlen:
            self.__i = 0
            self.__j += 1
        return result

    def close(self) -> None:
        pass

    def set_pos(self, pos: int) -> None:
        self.__j = pos // self.__xlen
        self.__i = pos % self.__xlen

    @classmethod
    def input_schema(cls, *args: Any, **kwargs: Any) -> dict[str, ts.ColumnType]:
        return {
            'image': ts.ImageType(),
            'tile_size': ts.JsonType(),
            'overlap': ts.JsonType(),
        }

    @classmethod
    def output_schema(cls,  *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {
            'tile': ts.ImageType(),
            'tile_coord': ts.JsonType(),
            'tile_box': ts.JsonType(),
        }, ['tile']
