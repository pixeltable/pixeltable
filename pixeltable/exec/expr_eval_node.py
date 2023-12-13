from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import sys
import logging
import time

import numpy as np
import nos
from tqdm.autonotebook import tqdm

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
import pixeltable.exprs as exprs
import pixeltable.func as func
import pixeltable.env as env


_logger = logging.getLogger('pixeltable')

class ExprEvalNode(ExecNode):
    """Materializes expressions
    """
    @dataclass
    class Cohort:
        """List of exprs that form an evaluation context and contain calls to at most one NOS function"""
        exprs: List[exprs.Expr]
        model_info: Optional[nos.common.ModelSpec]
        segment_ctxs: List[exprs.RowBuilder.EvalCtx]
        target_slot_idxs: List[int]

        # for NOS cohorts:
        nos_param_names: Optional[List[str]] = None
        scalar_nos_param_names: Optional[List[str]] = None

        # for models on images:

        img_param_pos: Optional[int] = None  # position of the image parameter in the function signature
        # for multi-resolution image models
        img_batch_params: List[nos.common.ObjectTypeInfo] = field(default_factory=list)
        # for single-resolution image models
        batch_size: int = 8
        img_size: Optional[Tuple[int, int]] = None  # W, H

        def __post_init__(self):
            if self.model_info is None:
                return
            nos_calls = [e for e in self.exprs if isinstance(e, exprs.FunctionCall) and e.is_nos_call()]
            assert len(nos_calls) <= 1
            nos_call = nos_calls[0] if len(nos_calls) > 0 else None
            self.nos_param_names = self.model_info.signature.get_inputs_spec().keys()
            self.scalar_nos_param_names = []

            # try to determine batch_size and img_size
            batch_size = sys.maxsize
            for pos, (param_name, type_info) in enumerate(self.model_info.signature.get_inputs_spec().items()):
                if isinstance(type_info, list):
                    assert isinstance(type_info[0].base_spec(), nos.common.ImageSpec)
                    # this is a multi-resolution image model
                    self.img_batch_params = type_info
                    self.img_param_pos = pos
                else:
                    if not type_info.is_batched():
                        self.scalar_nos_param_names.append(param_name)
                        if param_name not in nos_call.constant_args:
                            # this is a scalar parameter that is not constant, so we need to do batches of 1
                            batch_size = 1
                    else:
                        batch_size = min(batch_size, type_info.batch_size())

                    if isinstance(type_info.base_spec(), nos.common.ImageSpec):
                        # this is a single-resolution image model
                        if type_info.base_spec().shape is not None:
                            self.img_size = (type_info.base_spec().shape[1], type_info.base_spec().shape[0])
                        self.img_param_pos = pos
                        self.img_batch_params = []

            if batch_size == sys.maxsize:
                # some reasonable default
                self.batch_size = 8
            else:
                self.batch_size = batch_size

        def is_multi_res_model(self) -> bool:
            return self.img_param_pos is not None and len(self.img_batch_params) > 0

        def get_batch_params(self, img_size: Tuple[int, int]) -> Tuple[int, Tuple[int, int]]:
            """Returns batch_size and img_size appropriate for the given image size"""
            if len(self.img_batch_params) > 0:
                input_res = img_size[0] * img_size[1]
                resolutions = [info.base_spec().shape[0] * info.base_spec().shape[1] for info in self.img_batch_params]
                deltas = [abs(res - input_res) for res in resolutions]
                idx = deltas.index(min(deltas))
                type_info = self.img_batch_params[idx]
                return type_info.batch_size(), (type_info.base_spec().shape[1], type_info.base_spec().shape[0])
            else:
                return self.batch_size, self.img_size

    def __init__(
            self, row_builder: exprs.RowBuilder, output_exprs: List[exprs.Expr], input_exprs: List[exprs.Expr],
            ignore_errors: bool, input: ExecNode
    ):
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.input_exprs = input_exprs
        input_slot_idxs = {e.slot_idx for e in input_exprs}
        # we're only materializing exprs that are not already in the input
        self.target_exprs = [e for e in output_exprs if e.slot_idx not in input_slot_idxs]
        self.ignore_errors = ignore_errors  # if False, raise exc.ExprEvalError on error in _exec_cohort()
        self.pbar: Optional[tqdm] = None
        self.cohorts: List[List[ExprEvalNode.Cohort]] = []
        self._create_cohorts()

    def __next__(self) -> DataRowBatch:
        input_batch = next(self.input)
        # compute target exprs
        for cohort in self.cohorts:
            self._exec_cohort(cohort, input_batch)
        _logger.debug(f'ExprEvalNode: returning {len(input_batch)} rows')
        return input_batch

    def _open(self) -> None:
        if self.ctx.show_pbar:
            self.pbar = tqdm(total=len(self.target_exprs) * self.ctx.num_rows, desc='Computing cells', unit='cells')

    def _close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

    def _get_nos_info(self, expr: exprs.Expr) -> Optional[nos.common.ModelSpec]:
        """Get ModelSpec if expr is a call to a NOS function, else None."""
        if not isinstance(expr, exprs.FunctionCall):
            return None
        return func.FunctionRegistry.get().get_nos_info(expr.fn)

    def _is_nos_call(self, expr: exprs.Expr) -> bool:
        return self._get_nos_info(expr) is not None

    def _create_cohorts(self) -> None:
        all_exprs = self.row_builder.get_dependencies(self.target_exprs)
        # break up all_exprs into cohorts such that each cohort contains calls to at most one NOS function;
        # seed the cohorts with only the nos calls
        cohorts: List[List[exprs.Expr]] = []
        current_nos_function: Optional[func.Function] = None
        for e in all_exprs:
            if not self._is_nos_call(e):
                continue
            if current_nos_function is None or current_nos_function != e.fn:
                # create a new cohort
                cohorts.append([])
                current_nos_function = e.fn
            cohorts[-1].append(e)

        # expand the cohorts to include all exprs that are in the same evaluation context as the NOS calls;
        # cohorts are evaluated in order, so we can exclude the target slots from preceding cohorts and input slots
        exclude = set([e.slot_idx for e in self.input_exprs])
        all_target_slot_idxs = set([e.slot_idx for e in self.target_exprs])
        target_slot_idxs: List[List[int]] = []  # the ones materialized by each cohort
        for i in range(len(cohorts)):
            cohorts[i] = self.row_builder.get_dependencies(
                cohorts[i], exclude=[self.row_builder.unique_exprs[slot_idx] for slot_idx in exclude])
            target_slot_idxs.append(
                [e.slot_idx for e in cohorts[i] if e.slot_idx in all_target_slot_idxs])
            exclude.update(target_slot_idxs[-1])

        all_cohort_slot_idxs = set([e.slot_idx for cohort in cohorts for e in cohort])
        remaining_slot_idxs = set(all_target_slot_idxs) - all_cohort_slot_idxs
        if len(remaining_slot_idxs) > 0:
            cohorts.append(self.row_builder.get_dependencies(
                [self.row_builder.unique_exprs[slot_idx] for slot_idx in remaining_slot_idxs],
                exclude=[self.row_builder.unique_exprs[slot_idx] for slot_idx in exclude]))
            target_slot_idxs.append(list(remaining_slot_idxs))
        # we need to have captured all target slots at this point
        assert all_target_slot_idxs == set().union(*target_slot_idxs)

        for i in range(len(cohorts)):
            cohort = cohorts[i]
            # segment the cohort into sublists that contain either a single NOS function call or no NOS function calls
            # (i.e., only computed cols)
            assert len(cohort) > 0
            # create the first segment here, so we can avoid checking for an empty list in the loop
            segments = [[cohort[0]]]
            is_nos_segment = self._is_nos_call(cohort[0])
            model_info: Optional[nos.common.ModelSpec] = self._get_nos_info(cohort[0])
            for e in cohort[1:]:
                if self._is_nos_call(e):
                    segments.append([e])
                    is_nos_segment = True
                    model_info = self._get_nos_info(e)
                else:
                    if is_nos_segment:
                        # start a new segment
                        segments.append([])
                        is_nos_segment = False
                    segments[-1].append(e)

            # we create the EvalCtxs manually because create_eval_ctx() would repeat the dependencies of each segment
            segment_ctxs = [
                exprs.RowBuilder.EvalCtx(
                    slot_idxs=[e.slot_idx for e in s], exprs=s, target_slot_idxs=[], target_exprs=[])
                for s in segments
            ]
            cohort_info = self.Cohort(cohort, model_info, segment_ctxs, target_slot_idxs[i])
            self.cohorts.append(cohort_info)

    def _exec_cohort(self, cohort: Cohort, rows: DataRowBatch) -> None:
        """Compute the cohort for the entire input batch by dividing it up into sub-batches"""
        batch_start_idx = 0  # start row of the current sub-batch
        # for multi-resolution models, we re-assess the correct NOS batch size for each input batch
        verify_nos_batch_size = cohort.is_multi_res_model()
        while batch_start_idx < len(rows):
            num_batch_rows = min(cohort.batch_size, len(rows) - batch_start_idx)
            for segment_ctx in cohort.segment_ctxs:
                if not self._is_nos_call(segment_ctx.exprs[0]):
                    # compute batch row-wise
                    for row_idx in range(batch_start_idx, batch_start_idx + num_batch_rows):
                        self.row_builder.eval(rows[row_idx], segment_ctx, self.ctx.profile, ignore_errors=self.ignore_errors)
                else:
                    fn_call = segment_ctx.exprs[0]
                    # make a batched NOS call
                    arg_batches = [[] for _ in range(len(fn_call.args))]
                    assert len(cohort.nos_param_names) == len(arg_batches)

                    valid_batch_idxs: List[int] = []  # rows with exceptions are not valid
                    for row_idx in range(batch_start_idx, batch_start_idx + num_batch_rows):
                        row = rows[row_idx]
                        if row.has_exc(fn_call.slot_idx):
                            # one of our inputs had an exception, skip this row
                            continue
                        valid_batch_idxs.append(row_idx)
                        args, kwargs = fn_call._make_args(row)
                        assert len(kwargs) == 0
                        for i in range(len(args)):
                            arg_batches[i].append(args[i])
                    num_valid_batch_rows = len(valid_batch_idxs)

                    if verify_nos_batch_size:
                        # we need to choose a batch size based on the image size
                        sample_img = arg_batches[cohort.img_param_pos][0]
                        nos_batch_size, target_res = cohort.get_batch_params(sample_img.size)
                        verify_nos_batch_size = False
                    else:
                        nos_batch_size, target_res = cohort.batch_size, cohort.img_size

                    # if we need to rescale image args, and we're doing object detection, we need to rescale the
                    # bounding boxes as well
                    scale_factors = np.ndarray((num_valid_batch_rows, 2), dtype=np.float32)
                    if cohort.img_param_pos is not None:
                        # for now, NOS will only receive RGB images
                        arg_batches[cohort.img_param_pos] = \
                            [img.convert('RGB') for img in arg_batches[cohort.img_param_pos]]
                        if target_res is not None:
                            # we need to record the scale factors and resize the images;
                            # keep in mind that every image could have a different resolution
                            scale_factors[:, 0] = \
                                [img.size[0]/target_res[0] for img in arg_batches[cohort.img_param_pos]]
                            scale_factors[:, 1] = \
                                [img.size[1]/target_res[1] for img in arg_batches[cohort.img_param_pos]]
                            arg_batches[cohort.img_param_pos] = [
                                # only resize if necessary
                                img.resize(target_res) if img.size != target_res else img
                                for img in arg_batches[cohort.img_param_pos]
                            ]

                    num_remaining_batch_rows = num_valid_batch_rows
                    while num_remaining_batch_rows > 0:
                        # we make NOS calls in batches of nos_batch_size
                        num_nos_batch_rows = min(nos_batch_size, num_remaining_batch_rows)
                        nos_batch_offset = num_valid_batch_rows - num_remaining_batch_rows  # offset into args, not rows
                        kwargs = {
                            param_name: args[nos_batch_offset:nos_batch_offset + num_nos_batch_rows]
                            for param_name, args in zip(cohort.nos_param_names, arg_batches)
                        }
                        # fix up scalar parameters
                        kwargs.update(
                            {param_name: kwargs[param_name][0] for param_name in cohort.scalar_nos_param_names})
                        start_ts = time.perf_counter()
                        _logger.debug(
                            f'Running NOS task {cohort.model_info.task}: '
                            f'batch_size={num_nos_batch_rows} target_res={target_res}')
                        result = env.Env.get().nos_client.Run(
                            task=cohort.model_info.task, model_name=cohort.model_info.name, **kwargs)
                        self.ctx.profile.eval_time[fn_call.slot_idx] += time.perf_counter() - start_ts
                        self.ctx.profile.eval_count[fn_call.slot_idx] += num_nos_batch_rows

                        if cohort.model_info.task == nos.common.TaskType.OBJECT_DETECTION_2D and target_res is not None:
                            # we need to rescale the bounding boxes
                            result_bboxes = []  # workaround: result['bboxes'][*] is immutable
                            for i, bboxes in enumerate(result['bboxes']):
                                bboxes = np.copy(bboxes)
                                nos_batch_row_idx = nos_batch_offset + i
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
                            row_results = [
                                {k: v[i].tolist() for k, v in result.items()} for i in range(num_nos_batch_rows)
                            ]

                        # move the result into the row batch
                        for result_idx in range(len(row_results)):
                            row_idx = valid_batch_idxs[nos_batch_offset + result_idx]
                            row = rows[row_idx]
                            row[fn_call.slot_idx] = row_results[result_idx]

                        num_remaining_batch_rows -= num_nos_batch_rows

                    # switch to the NOS-recommended batch size
                    cohort.batch_size = nos_batch_size
                    cohort.img_size = target_res

            # make sure images for stored cols have been saved to files before moving on to the next batch
            rows.flush_imgs(
                slice(batch_start_idx, batch_start_idx + num_batch_rows), self.stored_img_cols, self.flushed_img_slots)
            if self.pbar is not None:
                self.pbar.update(num_batch_rows * len(cohort.target_slot_idxs))
            batch_start_idx += num_batch_rows

