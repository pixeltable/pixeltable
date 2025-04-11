from typing import Any, Optional

import sqlalchemy as sql

from pixeltable import exprs
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=29)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: Optional[str], v: Any) -> Optional[tuple[Optional[str], Any]]:
    # Defaults are now stored as literals in signatures
    if k == 'parameters':
        for param in v:
            assert isinstance(param, dict)
            has_default = param.get('has_default') or (param.get('default') is not None)
            if 'has_default' in param:
                del param['has_default']
            literal = exprs.Expr.from_object(param['default']) if has_default else None
            assert literal is None or isinstance(literal, exprs.Literal)
            param['default'] = None if literal is None else literal.as_dict()
        return k, v

    # Method of organizing argument expressions has changed
    if isinstance(v, dict) and v.get('_classname') == 'FunctionCall':
        args = v['args']
        kwargs = v['kwargs']
        components = v['components']
        group_by_start_idx = v['group_by_start_idx']
        group_by_stop_idx = v['group_by_stop_idx']
        order_by_start_idx = v['order_by_start_idx']

        new_args = []
        for arg in args:
            if arg[0] is not None:
                assert isinstance(arg[0], int)
                new_args.append(components[arg[0]])
            else:
                literal = exprs.Expr.from_object(arg[1])
                new_args.append(literal.as_dict())

        new_kwargs = {}
        for name, kwarg in kwargs.items():
            if kwarg[0] is not None:
                assert isinstance(kwarg[0], int)
                new_kwargs[name] = components[kwarg[0]]
            else:
                literal = exprs.Expr.from_object(kwarg[1])
                new_kwargs[name] = literal.as_dict()

        # We need to expand ("unroll") any var-args or var-kwargs.

        new_args_len = len(new_args)
        rolled_args: Optional[dict] = None
        rolled_kwargs: Optional[dict] = None

        if 'signature' in v['fn']:
            # If it's a pickled function, there's no signature, so we're out of luck; varargs in a pickled function
            # is an edge case that won't migrate properly.
            parameters: list[dict] = v['fn']['signature']['parameters']
            for i, param in enumerate(parameters):
                if param['kind'] == 'VAR_POSITIONAL' and new_args_len > i:
                    # For peculiar historical reasons, variable kwargs might show up in args. Thus variable
                    # positional args is not necessarily the last element of args; it might be the second-to-last.
                    assert new_args_len <= i + 2, new_args
                    rolled_args = new_args[i]
                    new_args = new_args[:i] + new_args[i + 1 :]
                if param['kind'] == 'VAR_KEYWORD':
                    # As noted above, variable kwargs might show up either in args or in kwargs. If it's in args, it
                    # is necessarily the last element.
                    if new_args_len > i:
                        assert new_args_len <= i + 1, new_args
                        rolled_kwargs = new_args.pop()
                    if param['name'] in kwargs:
                        assert rolled_kwargs is None
                        rolled_kwargs = kwargs.pop(param['name'])

        if rolled_args is not None:
            assert rolled_args['_classname'] in ('InlineArray', 'InlineList')
            new_args.extend(rolled_args['components'])
        if rolled_kwargs is not None:
            assert rolled_kwargs['_classname'] == 'InlineDict'
            new_kwargs.update(zip(rolled_kwargs['keys'], rolled_kwargs['components']))

        group_by_exprs = [components[i] for i in range(group_by_start_idx, group_by_stop_idx)]
        order_by_exprs = [components[i] for i in range(order_by_start_idx, len(components))]

        new_components = [*new_args, *new_kwargs.values(), *group_by_exprs, *order_by_exprs]

        newv = {
            'fn': v['fn'],
            'arg_idxs': list(range(len(new_args))),
            'kwarg_idxs': {name: i + len(new_args) for i, name in enumerate(new_kwargs.keys())},
            'group_by_start_idx': len(new_args) + len(new_kwargs),
            'group_by_stop_idx': len(new_args) + len(new_kwargs) + len(group_by_exprs),
            'order_by_start_idx': len(new_args) + len(new_kwargs) + len(group_by_exprs),
            'is_method_call': False,
            '_classname': 'FunctionCall',
            'components': new_components,
        }
        if 'return_type' in v:
            newv['return_type'] = v['return_type']

        return k, newv

    return None
