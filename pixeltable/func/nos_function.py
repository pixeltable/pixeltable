from typing import Optional, Any, Dict, List, Tuple
import inspect
import logging
import sys

import numpy as np

from .signature import Signature, Parameter
from .batched_function import BatchedFunction
import pixeltable.env as env
import pixeltable.type_system as ts
import pixeltable.exceptions as excs


_logger = logging.getLogger('pixeltable')

class NOSFunction(BatchedFunction):
    def __init__(self, model_spec: 'nos.common.ModelSpec', self_path: str):
        return_type, param_types = self._convert_nos_signature(model_spec.signature)
        param_names = list(model_spec.signature.get_inputs_spec().keys())
        params = [
            Parameter(name, col_type, inspect.Parameter.POSITIONAL_OR_KEYWORD, is_batched=False)
            for name, col_type in zip(param_names, param_types)
        ]
        signature = Signature(return_type, params)

        # construct inspect.Signature
        py_params = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name, col_type in zip(param_names, param_types)
        ]
        py_signature = inspect.Signature(py_params)
        super().__init__(signature, py_signature=py_signature, self_path=self_path)

        self.model_spec = model_spec
        self.nos_param_names = model_spec.signature.get_inputs_spec().keys()
        self.scalar_nos_param_names = []

        # for models on images
        self.img_param_pos: Optional[int] = None  # position of the image parameter in the function signature
        # for multi-resolution image models
        import nos
        self.img_batch_params: List[nos.common.ObjectTypeInfo] = []
        self.img_resolutions: List[int] = []  # for multi-resolution models
        self.batch_size: Optional[int] = None
        self.img_size: Optional[Tuple[int, int]] = None  # W, H

        # try to determine batch_size and img_size
        batch_size = sys.maxsize
        for pos, (param_name, type_info) in enumerate(model_spec.signature.get_inputs_spec().items()):
            if isinstance(type_info, list):
                assert isinstance(type_info[0].base_spec(), nos.common.ImageSpec)
                # this is a multi-resolution image model
                self.img_batch_params = type_info
                self.img_param_pos = pos
                self.img_resolutions = [
                    info.base_spec().shape[0] * info.base_spec().shape[1] for info in self.img_batch_params
                ]
            else:
                if not type_info.is_batched():
                    self.scalar_nos_param_names.append(param_name)
                else:
                    batch_size = min(batch_size, type_info.batch_size())

                if isinstance(type_info.base_spec(), nos.common.ImageSpec):
                    # this is a single-resolution image model
                    if type_info.base_spec().shape is not None:
                        self.img_size = (type_info.base_spec().shape[1], type_info.base_spec().shape[0])
                    self.img_param_pos = pos

        if batch_size != sys.maxsize:
            self.batch_size = batch_size

    def _convert_nos_type(
            self, type_info: 'nos.common.spec.ObjectTypeInfo', ignore_shape: bool = False
    ) -> ts.ColumnType:
        """Convert ObjectTypeInfo to ColumnType"""
        import nos
        if type_info.base_spec() is None:
            if type_info.base_type() == str:
                return ts.StringType()
            if type_info.base_type() == int:
                return ts.IntType()
            if type_info.base_type() == float:
                return ts.FloatType()
            if type_info.base_type() == bool:
                return ts.BoolType()
            else:
                raise excs.Error(f'Cannot convert {type_info} to ColumnType')
        elif isinstance(type_info.base_spec(), nos.common.ImageSpec):
            size = None
            if not ignore_shape and type_info.base_spec().shape is not None:
                size = (type_info.base_spec().shape[1], type_info.base_spec().shape[0])
            # TODO: set mode
            return ts.ImageType(size=size)
        elif isinstance(type_info.base_spec(), nos.common.TensorSpec):
            return ts.ArrayType(shape=type_info.base_spec().shape, dtype=ts.FloatType())
        else:
            raise excs.Error(f'Cannot convert {type_info} to ColumnType')

    def _convert_nos_signature(
            self, sig: 'nos.common.spec.FunctionSignature') -> Tuple[ts.ColumnType, List[ts.ColumnType]]:
        if len(sig.get_outputs_spec()) > 1:
            return_type = ts.JsonType()
        else:
            return_type = self._convert_nos_type(list(sig.get_outputs_spec().values())[0])
        param_types: List[ts.ColumnType] = []
        for _, type_info in sig.get_inputs_spec().items():
            # if there are multiple input shapes we leave them out of the ColumnType and deal with them in FunctionCall
            if isinstance(type_info, list):
                param_types.append(self._convert_nos_type(type_info[0], ignore_shape=True))
            else:
                param_types.append(self._convert_nos_type(type_info, ignore_shape=False))
        return return_type, param_types

    def is_multi_res_model(self) -> bool:
        return self.img_param_pos is not None and len(self.img_batch_params) > 0

    def get_batch_size(self, *args: Any, **kwargs: Any) -> Optional[int]:
        if self.batch_size is not None or len(self.img_batch_params) == 0 or len(args) == 0:
            return self.batch_size

        # return batch size appropriate for the given image size
        img_arg = args[self.img_param_pos]
        input_res = img_arg.size[0] * img_arg.size[1]
        batch_size, _ = self._select_model_res(input_res)
        return batch_size

    def _select_model_res(self, input_res: int) -> Tuple[int, Tuple[int, int]]:
        """Select the model resolution that is closest to the input resolution
        Returns: batch size, image size
        """
        deltas = [abs(res - input_res) for res in self.img_resolutions]
        idx = deltas.index(min(deltas))
        type_info = self.img_batch_params[idx]
        return type_info.batch_size(), (type_info.base_spec().shape[1], type_info.base_spec().shape[0])

    def invoke(self, arg_batches: List[List[Any]], kwarg_batches: Dict[str, List[Any]]) -> List[Any]:
        # check that scalar args are constant

        num_batch_rows = len(arg_batches[0])
        # if we need to rescale image args, and we're doing object detection, we need to rescale the
        # bounding boxes as well
        scale_factors = np.ndarray((num_batch_rows, 2), dtype=np.float32)

        target_res: Optional[Tuple[int, int]] = None
        if self.img_param_pos is not None:
            # for now, NOS will only receive RGB images
            arg_batches[self.img_param_pos] = \
                [img.convert('RGB') if img.mode != 'RGB' else img for img in arg_batches[self.img_param_pos ]]
            if self.is_multi_res_model():
                # we need to select the resolution that is closest to the input resolution
                sample_img = arg_batches[self.img_param_pos][0]
                _, target_res = self._select_model_res(sample_img.size[0] * sample_img.size[1])
            else:
                target_res = self.img_size

            if target_res is not None:
                # we need to record the scale factors and resize the images;
                # keep in mind that every image could have a different resolution
                scale_factors[:, 0] = \
                    [img.size[0] / target_res[0] for img in arg_batches[self.img_param_pos]]
                scale_factors[:, 1] = \
                    [img.size[1] / target_res[1] for img in arg_batches[self.img_param_pos]]
                arg_batches[self.img_param_pos] = [
                    # only resize if necessary
                    img.resize(target_res) if img.size != target_res else img
                    for img in arg_batches[self.img_param_pos]
                ]

        kwargs = {param_name: args for param_name, args in zip(self.nos_param_names, arg_batches)}
        # fix up scalar parameters
        kwargs.update(
            {param_name: kwargs[param_name][0] for param_name in self.scalar_nos_param_names})
        _logger.debug(
            f'Running NOS task {self.model_spec.task}: '
            f'batch_size={num_batch_rows} target_res={target_res}')
        result = env.Env.get().nos_client.Run(
            task=self.model_spec.task, model_name=self.model_spec.name, **kwargs)

        import nos
        if self.model_spec.task == nos.common.TaskType.OBJECT_DETECTION_2D and target_res is not None:
            # we need to rescale the bounding boxes
            result_bboxes = []  # workaround: result['bboxes'][*] is immutable
            for i, bboxes in enumerate(result['bboxes']):
                bboxes = np.copy(bboxes)
                nos_batch_row_idx = i
                bboxes[:, 0] *= scale_factors[nos_batch_row_idx, 0]
                bboxes[:, 1] *= scale_factors[nos_batch_row_idx, 1]
                bboxes[:, 2] *= scale_factors[nos_batch_row_idx, 0]
                bboxes[:, 3] *= scale_factors[nos_batch_row_idx, 1]
                result_bboxes.append(bboxes)
            result['bboxes'] = result_bboxes

        if len(result) == 1:
            key = list(result.keys())[0]
            row_results = result[key]
        else:
            # we rearrange result into one dict per row
            row_results = [{k: v[i].tolist() for k, v in result.items()} for i in range(num_batch_rows)]
        return row_results

