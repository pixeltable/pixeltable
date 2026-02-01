from typing import AsyncIterator

from pixeltable import catalog, exceptions as excs, exprs, type_system as ts
from pixeltable.func.iterator import IteratorCall

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode


class ComponentIterationNode(ExecNode):
    """Expands each row from a base table into one row per component returned by an iterator

    Returns row batches of OUTPUT_BATCH_SIZE size.
    """

    view: catalog.TableVersionHandle
    iterator_call: IteratorCall
    iterator_args_expr: exprs.InlineDict
    iterator_args_ctx: exprs.RowBuilder.EvalCtx
    iterator_output_cols: dict[str, catalog.Column]
    refd_output_slot_idxs: dict[str, int]

    __OUTPUT_BATCH_SIZE = 1024

    def __init__(self, view: catalog.TableVersionHandle, input: ExecNode):
        assert view.get().is_component_view
        super().__init__(input.row_builder, [], [], input)

        self.view = view
        self.iterator_call = view.get().iterator_call
        iterator_args_expr = [view.get().iterator_args_expr()]
        self.row_builder.set_slot_idxs(iterator_args_expr)
        self.iterator_args_expr = iterator_args_expr[0]
        self.iterator_args_ctx = self.row_builder.create_eval_ctx([self.iterator_args_expr])
        self.iterator_output_cols = {
            name: self.view.get().cols_by_name[name] for name in self.iterator_call.outputs
        }

        # referenced iterator output fields
        self.refd_output_slot_idxs = {
            e.col.name: e.slot_idx
            for e in self.row_builder.unique_exprs
            if isinstance(e, exprs.ColumnRef) and e.col.name in self.iterator_call.outputs
        }

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        output_batch = DataRowBatch(self.row_builder)
        async for input_batch in self.input:
            for input_row in input_batch:
                self.row_builder.eval(input_row, self.iterator_args_ctx)
                iterator_args = input_row[self.iterator_args_expr.slot_idx]
                assert isinstance(iterator_args, dict)
                # We need to ensure that all of the required (non-nullable) parameters of the iterator are
                # specified and are not null. If any of them are null, then we skip this row (i.e., we emit 0
                # output rows for this input row).
                if self.__non_nullable_args_specified(iterator_args):
                    iterator = self.view.get().iterator_call.it.eval(iterator_args)
                    for pos, component_dict in enumerate(iterator):
                        output_row = self.row_builder.make_row()
                        input_row.copy(output_row)
                        # we're expanding the input and need to add the iterator position to the pk
                        self.__populate_output_row(output_row, pos, component_dict)
                        output_batch.add_row(output_row)
                        if len(output_batch) == self.__OUTPUT_BATCH_SIZE:
                            yield output_batch
                            output_batch = DataRowBatch(self.row_builder)

        if len(output_batch) > 0:
            yield output_batch

    def __non_nullable_args_specified(self, iterator_args: dict) -> bool:
        """
        Returns true if all non-nullable iterator arguments are not `None`.
        """
        iterator_cls = self.view.get().iterator_call.it
        for arg_name, arg_value in iterator_args.items():
            col_type = iterator_cls.signature.parameters[arg_name].col_type
            if arg_value is None and not col_type.nullable:
                return False
        return True

    def __populate_output_row(self, output_row: exprs.DataRow, pos: int, component_dict: dict) -> None:
        pk = output_row.pk[:-1] + (pos,) + output_row.pk[-1:]
        output_row.set_pk(pk)
        # validate component_dict fields and copy them to their respective slots in output_row.
        # if the column names differ from the component_dict keys, the remapping occurs here.
        for name, output_info in self.iterator_call.outputs.items():
            if output_info.orig_name not in component_dict:
                raise excs.Error(
                    f'Invalid output from iterator `{self.iterator_call.it.fqn}`: '
                    f'Expected to find a field {output_info.orig_name!r}. Output:\n{component_dict}'
                )
            if name not in self.refd_output_slot_idxs:
                # we can ignore this
                continue
            val = component_dict[output_info.orig_name]
            output_col = self.iterator_output_cols[name]
            output_col.col_type.validate_literal(val)
            output_row[self.refd_output_slot_idxs[name]] = val
