import os
from typing import Any, Iterator, Optional, Union

import fiftyone as fo  # type: ignore[import-untyped]
import fiftyone.utils.data as foud  # type: ignore[import-untyped]
import PIL.Image
import puremagic

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import exprs
from pixeltable.env import Env


class PxtImageDatasetImporter(foud.LabeledImageDatasetImporter):
    """
    Implementation of a FiftyOne `DatasetImporter` that reads image data from a Pixeltable table.
    """

    __image_format: str  # format to use for any exported images that are not already stored on disk
    __labels: dict[str, tuple[exprs.Expr, type[fo.Label]]]  # label_name -> (expr, label_cls)
    __image_idx: int  # index of the image expr in the select list
    __localpath_idx: Optional[int]  # index of the image localpath in the select list, if present
    __row_iter: Iterator[list]  # iterator over the table rows, to be convered to FiftyOne samples

    def __init__(
        self,
        tbl: pxt.Table,
        image: exprs.Expr,
        image_format: str,
        classifications: Union[exprs.Expr, list[exprs.Expr], dict[str, exprs.Expr], None] = None,
        detections: Union[exprs.Expr, list[exprs.Expr], dict[str, exprs.Expr], None] = None,
        dataset_dir: Optional[os.PathLike] = None,
        shuffle: bool = False,
        seed: Union[int, float, str, bytes, bytearray, None] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__(dataset_dir=dataset_dir, shuffle=shuffle, seed=seed, max_samples=max_samples)

        self.__image_format = image_format

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
                        raise excs.Error(f'Invalid label name: {label_name}')
                    if label_name in self.__labels:
                        raise excs.Error(f'Duplicate label name: {label_name}')
                    self.__labels[label_name] = (expr, label_cls)

        # Now add the remaining labels, assigning unused default names.
        for exprs_, label_cls, default_name in label_categories:
            if exprs_ is None or isinstance(exprs_, dict):
                continue
            exprs_list = [exprs_] if isinstance(exprs_, exprs.Expr) else exprs_
            assert isinstance(exprs_list, list)
            for expr in exprs_list:
                if default_name not in self.__labels:
                    name = default_name
                else:
                    i = 1
                    while f'{default_name}_{i}' in self.__labels:
                        i += 1
                    name = f'{default_name}_{i}'
                self.__labels[name] = (expr, label_cls)

        # Build the select list:
        # - Labels first, in the order they appear in self.__labels
        # - Then the `image` expr
        # - Then `image.localpath`, if `images` is a stored columnref

        selection = [expr for expr, _ in self.__labels.values()]
        self.__image_idx = len(selection)
        selection.append(image)

        if isinstance(image, exprs.ColumnRef) and image.col.is_stored:
            # A stored image column; we can use the existing localpaths
            self.__localpath_idx = len(selection)
            selection.append(image.localpath)
        else:
            self.__localpath_idx = None

        df = tbl.select(*selection)
        self.__row_iter = df._output_row_iterator()

    def __next__(self) -> tuple[str, Optional[fo.ImageMetadata], Optional[dict[str, fo.Label]]]:
        row = next(self.__row_iter)
        img = row[self.__image_idx]
        assert isinstance(img, PIL.Image.Image)
        if self.__localpath_idx is not None:
            # Use the existing localpath of the stored image
            file = row[self.__localpath_idx]
            assert isinstance(file, str)
        else:
            # Write the dynamically created image to a temp file
            file = str(Env.get().create_tmp_path(f'.{self.__image_format}'))
            img.save(file, format=self.__image_format)

        metadata = fo.ImageMetadata(
            size_bytes=os.path.getsize(file),
            mime_type=puremagic.from_file(file, mime=True),
            width=img.width,
            height=img.height,
            filepath=file,
            num_channels=len(img.getbands()),
        )

        labels: dict[str, fo.Label] = {}
        for idx, (label_name, (_, label_cls)) in enumerate(self.__labels.items()):
            label_data = row[idx]
            if label_data is None:
                continue

            label: fo.Label
            if label_cls is fo.Classifications:
                label = fo.Classifications(classifications=self.__as_fo_classifications(label_data))
            elif label_cls is fo.Detections:
                label = fo.Detections(detections=self.__as_fo_detections(label_data))
            else:
                raise AssertionError()
            labels[label_name] = label

        return file, metadata, labels

    def __as_fo_classifications(self, data: list) -> list[fo.Classification]:
        if not isinstance(data, list) or any('label' not in entry for entry in data):
            raise excs.Error(
                f"Invalid classifications data: {data}\n(Expected a list of dicts, each containing a 'label' key)"
            )
        return [fo.Classification(label=entry['label'], confidence=entry.get('confidence')) for entry in data]

    def __as_fo_detections(self, data: list) -> list[fo.Detections]:
        if not isinstance(data, list) or any('label' not in entry or 'bounding_box' not in entry for entry in data):
            raise excs.Error(
                f'Invalid detections data: {data}\n'
                "(Expected a list of dicts, each containing a 'label' and 'bounding_box' key)"
            )
        return [
            fo.Detection(label=entry['label'], bounding_box=entry['bounding_box'], confidence=entry.get('confidence'))
            for entry in data
        ]

    @property
    def has_dataset_info(self) -> bool:
        return False

    @property
    def has_image_metadata(self) -> bool:
        return True

    @property
    def label_cls(self) -> dict[str, type]:
        return {label_name: label_cls for label_name, (_, label_cls) in self.__labels.items()}

    def setup(self) -> None:
        pass

    def get_dataset_info(self) -> dict:
        pass

    def close(self, *args: Any) -> None:
        pass
