from typing import Any, Optional, Sequence

from pixeltable import exceptions as excs, exprs, type_system as ts

from .function import Function
from .signature import Signature


class ExprTemplate:
    """
    Encapsulates a single signature of an `ExprTemplateFunction` and its associated parameterized expression,
    along with various precomputed metadata. (This is analogous to a `Callable`-`Signature` pair in a
    `CallableFunction`.)
    """

    expr: 'exprs.Expr'
    signature: Signature
    param_exprs: dict[str, 'exprs.Variable']

    def __init__(self, expr: 'exprs.Expr', signature: Signature):
        self.expr = expr
        self.signature = signature

        self.param_exprs = {name: exprs.Variable(name, param.col_type) for name, param in signature.parameters.items()}

        # validate that all variables in the expression are parameters
        for var in expr.subexprs(expr_class=exprs.Variable):
            assert var.name in self.param_exprs, f"Variable '{var.name}' in expression is not a parameter"

        # verify default values
        self.defaults: dict[str, exprs.Literal] = {}
        for param in self.signature.parameters.values():
            if param.default is None:
                continue
            self.defaults[param.name] = param.default


class ExprTemplateFunction(Function):
    """A parameterized expression from which an executable Expr is created with a function call."""

    templates: list[ExprTemplate]
    self_name: str

    def __init__(self, templates: list[ExprTemplate], self_path: Optional[str] = None, name: Optional[str] = None):
        self.templates = templates
        self.self_name = name

        super().__init__([t.signature for t in templates], self_path=self_path)

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        self.templates = [self.templates[signature_idx]]

    @property
    def template(self) -> ExprTemplate:
        assert not self.is_polymorphic
        return self.templates[0]

    def instantiate(self, args: Sequence[Any], kwargs: dict[str, Any]) -> 'exprs.Expr':
        assert not self.is_polymorphic
        template = self.template
        bound_args = self.signature.py_signature.bind(*args, **kwargs).arguments
        # apply defaults, otherwise we might have Parameters left over
        bound_args.update(
            {param_name: default for param_name, default in template.defaults.items() if param_name not in bound_args}
        )
        result = template.expr.copy()
        arg_exprs: dict[exprs.Expr, exprs.Expr] = {}
        for param_name, arg in bound_args.items():
            param_expr = template.param_exprs[param_name]
            if not isinstance(arg, exprs.Expr):
                # TODO: use the available param_expr.col_type
                arg_expr = exprs.Expr.from_object(arg)
                if arg_expr is None:
                    raise excs.Error(f'{self.self_name}(): cannot convert argument {arg} to a Pixeltable expression')
            else:
                arg_expr = arg
            arg_exprs[param_expr] = arg_expr
        result = result.substitute(arg_exprs)
        return result

    def call_return_type(self, bound_args: dict[str, 'exprs.Expr']) -> ts.ColumnType:
        """
        The call_return_type of an ExprTemplateFunction is derived from the template expression's col_type after
        substitution (unlike for UDFs, whose call_return_type is derived from an explicitly specified
        conditional_return_type).
        """
        assert not self.is_polymorphic
        template = self.template
        with_defaults = bound_args.copy()
        with_defaults.update(
            {param_name: default for param_name, default in template.defaults.items() if param_name not in bound_args}
        )
        substituted_expr = self.template.expr.copy().substitute(
            {template.param_exprs[name]: expr for name, expr in with_defaults.items()}
        )
        return substituted_expr.col_type

    def comment(self) -> Optional[str]:
        if isinstance(self.templates[0].expr, exprs.FunctionCall):
            return self.templates[0].expr.fn.comment()
        return None

    def exec(self, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        from pixeltable import exec

        assert not self.is_polymorphic
        expr = self.instantiate(args, kwargs)
        row_builder = exprs.RowBuilder(output_exprs=[expr], columns=[], input_exprs=[])
        row_batch = exec.DataRowBatch(tbl=None, row_builder=row_builder, num_rows=1)
        row = row_batch[0]
        row_builder.eval(row, ctx=row_builder.default_eval_ctx)
        return row[row_builder.get_output_exprs()[0].slot_idx]

    @property
    def display_name(self) -> str:
        if not self.self_name and isinstance(self.templates[0].expr, exprs.FunctionCall):
            # In the common case where the templated expression is itself a FunctionCall,
            # fall back on the display name of the underlying FunctionCall
            return self.templates[0].expr.fn.display_name
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    @property
    def is_async(self) -> bool:
        return False

    def __str__(self) -> str:
        return str(self.templates[0].expr)

    def _as_dict(self) -> dict:
        if self.self_path is not None:
            return super()._as_dict()
        assert not self.is_polymorphic
        assert len(self.templates) == 1
        return {'expr': self.template.expr.as_dict(), 'signature': self.signature.as_dict(), 'name': self.name}

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        if 'expr' not in d:
            return super()._from_dict(d)
        assert 'signature' in d and 'name' in d
        template = ExprTemplate(exprs.Expr.from_dict(d['expr']), Signature.from_dict(d['signature']))
        return cls([template], name=d['name'])
