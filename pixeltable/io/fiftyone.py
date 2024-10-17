import os
from os import PathLike
from typing import Iterator, Optional, Union

import fiftyone as fo
import fiftyone.utils.data as foud
import PIL.Image
import puremagic

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import exprs
from pixeltable.env import Env
from pixeltable.exprs.data_row import DataRow


class PxtDatasetImporter(foud.LabeledImageDatasetImporter):
    __image_format: str
    __image_expr: exprs.Expr
    __image_localpath_expr: Optional[exprs.Expr]
    __labels: dict[str, tuple[exprs.Expr, type]]  # label_name -> (expr, label_cls)
    __count: int
    __iter: Iterator[DataRow]

    def __init__(
        self,
        tbl: pxt.Table,
        images: exprs.Expr,
        image_format: str,
        classifications: Union[exprs.Expr, list[exprs.Expr], dict[str, exprs.Expr], None] = None,
        detections: Union[exprs.Expr, list[exprs.Expr], dict[str, exprs.Expr], None] = None,
        dataset_dir: Optional[PathLike] = None,
        shuffle: bool = False,
        seed: Union[int, float, str, bytes, bytearray, None] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples
        )

        self.__image_format = image_format
        self.__image_expr = images

        label_categories = [
            (classifications, fo.Classifications, 'classifications'),
            (detections, fo.Detections, 'detections'),
        ]

        # Construct the labels. First add labels for all label types that have named dictionaries.
        self.__labels = {}
        for exprs_, label_cls, _ in label_categories:
            if isinstance(exprs_, dict):
                for label_name, expr in exprs_.items():
                    if not label_name.isidentifier():
                        raise excs.Error(f"Invalid label name: {label_name}")
                    if label_name in self.__labels:
                        raise excs.Error(f"Duplicate label name: {label_name}")
                    self.__labels[label_name] = (expr, label_cls)

        # Now add the remaining labels, assigning unused default names.
        for exprs_, label_cls, default_name in label_categories:
            if exprs_ is None:
                continue
            if isinstance(exprs_, exprs_.Expr):
                exprs_ = [exprs_]
            assert isinstance(exprs_, list)
            for expr in exprs_:
                if default_name not in self.__labels:
                    name = default_name
                else:
                    i = 1
                    while f'{default_name}_{i}' in self.__labels:
                        i += 1
                    name = f'{default_name}_{i}'
                self.__labels[name] = (expr, label_cls)

        if isinstance(self.__image_expr, exprs.ColumnRef) and self.__image_expr.col.is_stored:
            # A stored image column; we can use the existing localpaths
            self.__image_localpath_expr = self.__image_expr.localpath
            selection = [self.__image_expr, self.__image_localpath_expr]
        else:
            # Not a stored image column; we'll write the images to a temp location
            self.__image_localpath_expr = None
            selection = [self.__image_expr]

        selection.extend(expr for expr, _ in self.__labels.values())

        df = tbl.select(*selection)
        self.__count = df.count()
        self.__iter = df._exec()

    def __len__(self) -> int:
        return self.__count

    def __next__(self) -> tuple[str, Optional[fo.ImageMetadata], Optional[dict[str, fo.Label]]]:
        row = next(self.__iter)
        img = row[self.__image_expr.slot_idx]
        assert isinstance(img, PIL.Image.Image)
        if self.__image_localpath_expr is not None:
            file = row[self.__image_localpath_expr.slot_idx]
            assert isinstance(file, str)
        else:
            file = str(Env.get().create_tmp_path(f'.{self.__image_format}'))
            img.save(file, format=self.__image_format)

        metadata = fo.ImageMetadata(
            size_bytes=os.path.getsize(file),
            mime_type=puremagic.from_file(file, mime=True),
            width=img.width,
            height=img.height,
            filepath=file
        )

        labels: dict[str, fo.Label] = {}
        for label_name, (expr, label_cls) in self.__labels.items():
            label = row[expr.slot_idx]
            if label is not None:
                labels[label_name] = label_cls(label=label)
        return file, metadata, labels

    @property
    def has_dataset_info(self) -> bool:
        return False

    @property
    def has_image_metadata(self) -> bool:
        return False

    @property
    def label_cls(self) -> dict[str, type]:
        return {label_name: label_cls for label_name, (_, label_cls) in self.__labels.items()}

    def setup(self) -> None:
        pass

    def get_dataset_info(self) -> dict:
        pass

    def close(self, *args) -> None:
        pass
