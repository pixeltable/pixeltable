from __future__ import annotations

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import exprs
from pixeltable.env import Env

# if TYPE_CHECKING:
#     import fiftyone as fo  # type: ignore[import-untyped]


def export_images_as_fo_dataset(
    tbl: pxt.Table,
    images: exprs.Expr,
    image_format: str = 'webp',
    classifications: exprs.Expr | list[exprs.Expr] | dict[str, exprs.Expr] | None = None,
    detections: exprs.Expr | list[exprs.Expr] | dict[str, exprs.Expr] | None = None,
) -> 'fo.Dataset':  # type: ignore[name-defined]  # noqa: F821
    """
    Export images from a Pixeltable table as a Voxel51 dataset. The data must consist of a single column
    (or expression) containing image data, along with optional additional columns containing labels. Currently, only
    classification and detection labels are supported.

    The [Working with Voxel51 in Pixeltable](https://docs.pixeltable.com/examples/vision/voxel51) tutorial contains a
    fully worked example showing how to export data from a Pixeltable table and load it into Voxel51.

    Images in the dataset that already exist on disk will be exported directly, in whatever format they
    are stored in. Images that are not already on disk (such as frames extracted using a
    [`frame_iterator`][pixeltable.functions.video.frame_iterator]) will first be written to disk in the specified
    `image_format`.

    The label parameters accept one or more sets of labels of each type. If a single `Expr` is provided, then it will
    be exported as a single set of labels with a default name such as `classifications`.
    (The single set of labels may still containing multiple individual labels; see below.)
    If a list of `Expr`s is provided, then each one will be exported as a separate set of labels with a default name
    such as `classifications`, `classifications_1`, etc. If a dictionary of `Expr`s is provided, then each entry will
    be exported as a set of labels with the specified name.

    __Requirements:__

    - `pip install fiftyone`

    Args:
        tbl: The table from which to export data.
        images: A column or expression that contains the images to export.
        image_format: The format to use when writing out images for export.
        classifications: Optional image classification labels. If a single `Expr` is provided, it must be a table
            column or an expression that evaluates to a list of dictionaries. Each dictionary in the list corresponds
            to an image class and must have the following structure:

            ```python
            {'label': 'zebra', 'confidence': 0.325}
            ```

            If multiple `Expr`s are provided, each one must evaluate to a list of such dictionaries.
        detections: Optional image detection labels. If a single `Expr` is provided, it must be a table column or an
            expression that evaluates to a list of dictionaries. Each dictionary in the list corresponds to an image
            detection, and must have the following structure:

            ```python
            {
                'label': 'giraffe',
                'confidence': 0.99,
                # [x, y, w, h], fractional coordinates
                'bounding_box': [0.081, 0.836, 0.202, 0.136],
            }
            ```

            If multiple `Expr`s are provided, each one must evaluate to a list of such dictionaries.

    Returns:
        A Voxel51 dataset.

    Example:
        Export the images in the `image` column of the table `tbl` as a Voxel51 dataset, using classification
        labels from `tbl.classifications`:

        >>> export_images_as_fo_dataset(
        ...     tbl, tbl.image, classifications=tbl.classifications
        ... )

        See the [Working with Voxel51 in Pixeltable](https://docs.pixeltable.com/examples/vision/voxel51) tutorial
        for a fully worked example.
    """
    Env.get().require_package('fiftyone')

    import fiftyone as fo  # type: ignore[import-not-found]

    from pixeltable.io.fiftyone import PxtImageDatasetImporter  # type: ignore[attr-defined]

    if not images.col_type.is_image_type():
        raise excs.RequestError(
            excs.ErrorCode.TYPE_MISMATCH,
            f'`images` must be an expression of type Image (got {images.col_type._to_base_str()})',
        )

    return fo.Dataset.from_importer(
        PxtImageDatasetImporter(tbl, images, image_format, classifications=classifications, detections=detections)
    )
