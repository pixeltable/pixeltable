from typing import Generator, Optional

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
import pixeltable.catalog as catalog
import pixeltable.exprs as exprs
import pixeltable.exceptions as excs


class ComponentIterationNode(ExecNode):
    """Expands each row from a base table into one row per component returned by an iterator

    Returns row batches of OUTPUT_BATCH_SIZE size.
    """
    OUTPUT_BATCH_SIZE = 1024

    def __init__(self, view: catalog.TableVersion, input: ExecNode):
        assert view.is_component_view()
        super().__init__(input.row_builder, [], [], input)
        self.view = view
        iterator_args = [view.iterator_args.copy()]
        self.row_builder.substitute_exprs(iterator_args)
        self.iterator_args = iterator_args[0]
        assert isinstance(self.iterator_args, exprs.InlineDict)
        self.iterator_args_ctx = self.row_builder.create_eval_ctx([self.iterator_args])
        self.iterator_output_schema, self.unstored_column_names = \
            self.view.iterator_cls.output_schema(**self.iterator_args.to_dict())
        self.iterator_output_fields = list(self.iterator_output_schema.keys())
        self.iterator_output_cols = \
            {field_name: self.view.cols_by_name[field_name] for field_name in self.iterator_output_fields}
        # referenced iterator output fields
        self.refd_output_slot_idxs = {
            e.col.name: e.slot_idx for e in self.row_builder.unique_exprs
            if isinstance(e, exprs.ColumnRef) and e.col.name in self.iterator_output_fields
        }
        self._output: Optional[Generator[DataRowBatch, None, None]] = None

    def _output_batches(self) -> Generator[DataRowBatch, None, None]:
        output_batch = DataRowBatch(self.view, self.row_builder)
        for input_batch in self.input:
            for input_row in input_batch:
                self.row_builder.eval(input_row, self.iterator_args_ctx)
                iterator_args = input_row[self.iterator_args.slot_idx]
                iterator = self.view.iterator_cls(**iterator_args)
                for pos, component_dict in enumerate(iterator):
                    output_row = output_batch.add_row()
                    input_row.copy(output_row)
                    # we're expanding the input and need to add the iterator position to the pk
                    pk = output_row.pk[:-1] + (pos,) + output_row.pk[-1:]
                    output_row.set_pk(pk)

                    # verify and copy component_dict fields to their respective slots in output_row
                    for field_name, field_val in component_dict.items():
                        if field_name not in self.iterator_output_fields:
                            raise excs.Error(
                                f'Invalid field name {field_name} in output of {self.view.iterator_cls.__name__}')
                        if field_name not in self.refd_output_slot_idxs:
                            # we can ignore this
                            continue
                        output_col = self.iterator_output_cols[field_name]
                        output_col.col_type.validate_literal(field_val)
                        output_row[self.refd_output_slot_idxs[field_name]] = field_val
                    if len(component_dict) != len(self.iterator_output_fields):
                        missing_fields = set(self.refd_output_slot_idxs.keys()) - set(component_dict.keys())
                        raise excs.Error(
                            f'Invalid output of {self.view.iterator_cls.__name__}: '
                            f'missing fields {", ".join(missing_fields)}')

                    if len(output_batch) == self.OUTPUT_BATCH_SIZE:
                        yield output_batch
                        output_batch = DataRowBatch(self.view, self.row_builder)

        if len(output_batch) > 0:
            yield output_batch

    def __next__(self) -> DataRowBatch:
        if self._output is None:
            self._output = self._output_batches()
        return next(self._output)
