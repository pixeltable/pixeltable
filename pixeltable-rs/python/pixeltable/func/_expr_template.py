"""Functions defined by parameterized expressions (upstream `pixeltable.func.expr_template_function`).

The template expressions hold `Variable` nodes for the parameters; the backend substitutes the arguments when the
function is called.
"""

from __future__ import annotations

from typing import Any

from pixeltable import _core

from ._function import Function
from ._signature import Signature


class ExprTemplate:
    """One signature of an `ExprTemplateFunction` and its parameterized expression."""

    expr: _core.Expr
    signature: Signature

    def __init__(self, expr: _core.Expr, signature: Signature):
        self.expr = expr
        self.signature = signature
        self.defaults = {p.name: p.default for p in signature.parameters.values() if p.default is not None}


class ExprTemplateFunction(Function):
    """A parameterized expression from which an executable expression is created with a function call."""

    templates: list[ExprTemplate]
    self_name: str | None

    def __init__(
        self,
        templates: list[ExprTemplate],
        self_path: str | None = None,
        name: str | None = None,
        is_method: bool = False,
        is_property: bool = False,
    ):
        self.templates = templates
        self.self_name = name
        super().__init__(
            [t.signature for t in templates], self_path=self_path, is_method=is_method, is_property=is_property
        )

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        self.templates = [self.templates[signature_idx]]

    def _core_spec(self) -> dict[str, Any]:
        return {
            'kind': 'template',
            'name': self.name,
            'display_name': self.display_name,
            'self_path': self.self_path,
            'signatures': [t.signature._to_core() for t in self.templates],
            'templates': [t.expr for t in self.templates],
            'is_method': self.is_method,
            'is_property': self.is_property,
        }

    @property
    def template(self) -> ExprTemplate:
        assert not self.is_polymorphic
        return self.templates[0]

    def _template_fn(self) -> Function | None:
        """The function the (first) template calls, if it is a call of a Python function."""
        return getattr(self.templates[0].expr, 'fn', None)

    def comment(self) -> str | None:
        fn = self._template_fn()
        return fn.comment() if fn is not None else None

    @property
    def display_name(self) -> str:
        if not self.self_name:
            fn = self._template_fn()
            if fn is not None:
                return fn.display_name
        return self.self_name or ''

    @property
    def name(self) -> str:
        return self.self_name  # type: ignore[return-value]

    @property
    def is_async(self) -> bool:
        return False

    def __str__(self) -> str:
        return str(self.templates[0].expr)

    def __repr__(self) -> str:
        return str(self)

    def _as_dict(self) -> dict:
        if self.self_path is not None:
            return super()._as_dict()
        return {
            'templates': [{'expr': t.expr.as_dict(), 'signature': t.signature.as_dict()} for t in self.templates],
            'name': self.name,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        if 'templates' not in d:
            return super()._from_dict(d)
        templates = [
            ExprTemplate(_core.Expr.from_dict(t['expr']), Signature.from_dict(t['signature'])) for t in d['templates']
        ]
        return cls(templates, name=d['name'])
