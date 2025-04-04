from typing import AsyncIterator

from pixeltable import catalog, exceptions as excs, exprs

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode


class ComponentIterationNode(ExecNode):
    """Expands each row from a base table into one row per component returned by an iterator

    Returns row batches of OUTPUT_BATCH_SIZE size.
    """

    view: catalog.TableVersionHandle

    __OUTPUT_BATCH_SIZE = 1024

    def __init__(self, view: catalog.TableVersionHandle, input: ExecNode):
        assert view.get().is_component_view
        super().__init__(input.row_builder, [], [], input)
        self.view = view
        iterator_args = [view.get().iterator_args.copy()]
        self.row_builder.set_slot_idxs(iterator_args)
        self.iterator_args = iterator_args[0]
        assert isinstance(self.iterator_args, exprs.InlineDict)
        self.iterator_args_ctx = self.row_builder.create_eval_ctx([self.iterator_args])
        self.iterator_output_schema, self.unstored_column_names = self.view.get().iterator_cls.output_schema(
            **self.iterator_args.to_kwargs()
        )
        self.iterator_output_fields = list(self.iterator_output_schema.keys())
        self.iterator_output_cols = {
            field_name: self.view.get().cols_by_name[field_name] for field_name in self.iterator_output_fields
        }
        # referenced iterator output fields
        self.refd_output_slot_idxs = {
            e.col.name: e.slot_idx
            for e in self.row_builder.unique_exprs
            if isinstance(e, exprs.ColumnRef) and e.col.name in self.iterator_output_fields
        }

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        output_batch = DataRowBatch(self.view, self.row_builder)
        async for input_batch in self.input:
            for input_row in input_batch:
                self.row_builder.eval(input_row, self.iterator_args_ctx)
                iterator_args = input_row[self.iterator_args.slot_idx]
                assert isinstance(iterator_args, dict)
                # We need to ensure that all of the required (non-nullable) parameters of the iterator are
                # specified and are not null. If any of them are null, then we skip this row (i.e., we emit 0
                # output rows for this input row).
                if self.__non_nullable_args_specified(iterator_args):
                    iterator = self.view.get().iterator_cls(**iterator_args)
                    for pos, component_dict in enumerate(iterator):
                        output_row = output_batch.add_row()
                        input_row.copy(output_row)
                        # we're expanding the input and need to add the iterator position to the pk
                        self.__populate_output_row(output_row, pos, component_dict)
                        if len(output_batch) == self.__OUTPUT_BATCH_SIZE:
                            yield output_batch
                            output_batch = DataRowBatch(self.view, self.row_builder)

        if len(output_batch) > 0:
            yield output_batch

    def __non_nullable_args_specified(self, iterator_args: dict) -> bool:
        """
        Returns true if all non-nullable iterator arguments are not `None`.
        """
        input_schema = self.view.get().iterator_cls.input_schema()
        for arg_name, arg_value in iterator_args.items():
            col_type = input_schema[arg_name]
            if arg_value is None and not col_type.nullable:
                return False
        return True

    def __populate_output_row(self, output_row: exprs.DataRow, pos: int, component_dict: dict) -> None:
        pk = output_row.pk[:-1] + (pos,) + output_row.pk[-1:]
        output_row.set_pk(pk)
        # verify and copy component_dict fields to their respective slots in output_row
        for field_name, field_val in component_dict.items():
            if field_name not in self.iterator_output_fields:
                raise excs.Error(
                    f'Invalid field name {field_name} in output of {self.view.get().iterator_cls.__name__}'
                )
            if field_name not in self.refd_output_slot_idxs:
                # we can ignore this
                continue
            output_col = self.iterator_output_cols[field_name]
            output_col.col_type.validate_literal(field_val)
            output_row[self.refd_output_slot_idxs[field_name]] = field_val
        if len(component_dict) != len(self.iterator_output_fields):
            missing_fields = set(self.refd_output_slot_idxs.keys()) - set(component_dict.keys())
            raise excs.Error(
                f'Invalid output of {self.view.get().iterator_cls.__name__}: missing fields {", ".join(missing_fields)}'
            )
