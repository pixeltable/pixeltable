import inspect
from typing import Dict, Optional, Callable, List

import pixeltable
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from .function import Function
from .signature import Signature, Parameter


class ExprTemplateFunction(Function):
    """A parameterized expression from which an executable Expr is created with a function call."""

    def __init__(
            self, expr: 'pixeltable.exprs.Expr', py_signature: inspect.Signature, self_path: Optional[str] = None,
            name: Optional[str] = None):
        import pixeltable.exprs as exprs
        self.expr = expr
        self.self_name = name
        self.param_exprs = list(set(expr.subexprs(expr_class=exprs.Variable)))
        # make sure there are no duplicate names
        assert len(self.param_exprs) == len(set(p.name for p in self.param_exprs))
        self.param_exprs_by_name = {p.name: p for p in self.param_exprs}

        # verify default values
        self.defaults: Dict[str, exprs.Literal] = {}  # key: param name, value: default value converted to a Literal
        for py_param in py_signature.parameters.values():
            if py_param.default is inspect.Parameter.empty:
                continue
            param_expr = self.param_exprs_by_name[py_param.name]
            try:
                literal_default = exprs.Literal(py_param.default, col_type=param_expr.col_type)
                self.defaults[py_param.name] = literal_default
            except TypeError as e:
                msg = str(e)
                raise excs.Error(f"Default value for parameter '{py_param.name}': {msg[0].lower() + msg[1:]}")
        # construct signature
        assert len(self.param_exprs) == len(py_signature.parameters)
        fn_params = [
            Parameter(p.name, self.param_exprs_by_name[p.name].col_type, p.kind)
            for p in py_signature.parameters.values()
        ]
        signature = Signature(return_type=expr.col_type, parameters=fn_params)

        super().__init__(signature, py_signature=py_signature, self_path=self_path)

    def instantiate(self, *args: object, **kwargs: object) -> 'pixeltable.exprs.Expr':
        bound_args = self.py_signature.bind(*args, **kwargs).arguments
        # apply defaults, otherwise we might have Parameters left over
        bound_args.update(
            {param_name: default for param_name, default in self.defaults.items() if param_name not in bound_args})
        result = self.expr.copy()
        import pixeltable.exprs as exprs
        for param_name, arg in bound_args.items():
            param_expr = self.param_exprs_by_name[param_name]
            if not isinstance(arg, exprs.Expr):
                # TODO: use the available param_expr.col_type
                arg_expr = exprs.Expr.from_object(arg)
                if arg_expr is None:
                    raise excs.Error(f'{self.self_name}(): cannot convert argument {arg} to a Pixeltable expression')
            else:
                arg_expr = arg
            result = result.substitute(param_expr, arg_expr)
        import pixeltable.exprs as exprs
        assert not result.contains(exprs.Variable)
        return result

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def _as_dict(self) -> Dict:
        if self.self_path is not None:
            return super()._as_dict()
        return {
            'name': self.name,
            'expr': self.expr.as_dict(),
            **super()._as_dict()
        }

    @classmethod
    def _from_dict(cls, d: Dict) -> Function:
        if 'expr' not in d:
            return super()._from_dict(d)
        import pixeltable.exprs as exprs
        return cls(exprs.Expr.from_dict(d['expr']), name=d['name'])
