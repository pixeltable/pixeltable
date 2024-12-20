import inspect
from typing import Any, Optional

import pixeltable
import pixeltable.exceptions as excs

from .function import Function
from .signature import Signature


class ExprTemplateFunction(Function):
    """A parameterized expression from which an executable Expr is created with a function call."""

    def __init__(
            self, expr: 'pixeltable.exprs.Expr', signature: Signature, self_path: Optional[str] = None,
            name: Optional[str] = None):
        import pixeltable.exprs as exprs
        self.expr = expr
        self.self_name = name
        self.param_exprs = list(set(expr.subexprs(expr_class=exprs.Variable)))
        # make sure there are no duplicate names
        assert len(self.param_exprs) == len(set(p.name for p in self.param_exprs))
        self.param_exprs_by_name = {p.name: p for p in self.param_exprs}

        # verify default values
        self.defaults: dict[str, exprs.Literal] = {}  # key: param name, value: default value converted to a Literal
        for param in signature.parameters.values():
            if param.default is inspect.Parameter.empty:
                continue
            param_expr = self.param_exprs_by_name[param.name]
            try:
                literal_default = exprs.Literal(param.default, col_type=param_expr.col_type)
                self.defaults[param.name] = literal_default
            except TypeError as e:
                msg = str(e)
                raise excs.Error(f"Default value for parameter '{param.name}': {msg[0].lower() + msg[1:]}")

        super().__init__(signature, self_path=self_path)

    def instantiate(self, *args: object, **kwargs: object) -> 'pixeltable.exprs.Expr':
        bound_args = self.signature.py_signature.bind(*args, **kwargs).arguments
        # apply defaults, otherwise we might have Parameters left over
        bound_args.update(
            {param_name: default for param_name, default in self.defaults.items() if param_name not in bound_args})
        result = self.expr.copy()
        import pixeltable.exprs as exprs
        arg_exprs: dict[exprs.Expr, exprs.Expr] = {}
        for param_name, arg in bound_args.items():
            param_expr = self.param_exprs_by_name[param_name]
            if not isinstance(arg, exprs.Expr):
                # TODO: use the available param_expr.col_type
                arg_expr = exprs.Expr.from_object(arg)
                if arg_expr is None:
                    raise excs.Error(f'{self.self_name}(): cannot convert argument {arg} to a Pixeltable expression')
            else:
                arg_expr = arg
            arg_exprs[param_expr] = arg_expr
        result = result.substitute(arg_exprs)
        import pixeltable.exprs as exprs
        assert not result._contains(exprs.Variable)
        return result

    def _docstring(self) -> Optional[str]:
        from pixeltable import exprs

        if isinstance(self.expr, exprs.FunctionCall):
            return self.expr.fn._docstring()
        return None

    def exec(self, *args: Any, **kwargs: Any) -> Any:
        expr = self.instantiate(*args, **kwargs)
        import pixeltable.exprs as exprs
        row_builder = exprs.RowBuilder(output_exprs=[expr], columns=[], input_exprs=[])
        import pixeltable.exec as exec
        row_batch = exec.DataRowBatch(tbl=None, row_builder=row_builder, len=1)
        row = row_batch[0]
        row_builder.eval(row, ctx=row_builder.default_eval_ctx)
        return row[row_builder.get_output_exprs()[0].slot_idx]

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def __str__(self) -> str:
        return str(self.expr)

    def _as_dict(self) -> dict:
        if self.self_path is not None:
            return super()._as_dict()
        return {
            'expr': self.expr.as_dict(),
            'signature': self.signature.as_dict(),
            'name': self.name,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        if 'expr' not in d:
            return super()._from_dict(d)
        assert 'signature' in d and 'name' in d
        import pixeltable.exprs as exprs
        return cls(exprs.Expr.from_dict(d['expr']), Signature.from_dict(d['signature']), name=d['name'])
