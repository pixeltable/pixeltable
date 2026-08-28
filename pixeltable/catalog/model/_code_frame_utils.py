"""
Recovery of class-body annotations on Python 3.14+, where PEP 649 defers their evaluation.

Background: a model class body declares columns as bare annotations (`value: pxt.Float`) and refers to
them in subsequent statements (`incr = value + 1`). That requires the annotation's *type* to be known
mid-body (expression construction is type-driven) and the declaration *order* of annotations relative to
assignments to be preserved.

Through Python 3.13, a bare class-body annotation compiles to eager bytecode that assigns into
`__annotations__` as the body runs, which `_AnnotationRecorder` intercepts. PEP 649 removed that: on 3.14+
annotations emit no namespace operations at all, and are instead compiled into an `__annotate__` function
assembled at the *end* of the body -- too late, and with no record of how annotations interleave with
assignments.

The annotations are still recoverable, though, without reading source: the `__annotate__` code object is a
constant of the class body code object, which is in turn a constant of the code object of the frame
executing the `class ...:` statement. That code object retains the annotation names (as constants) and
their line numbers, and evaluating it yields the real type objects. Assignment positions come from the
body's own `STORE_NAME` line numbers, so the two can be interleaved by line.

This relies on CPython bytecode layout rather than a language guarantee, so the entry points here return
`None`/empty rather than raising when the expected shape isn't found, letting the caller fall back to a
clear error.
"""

from __future__ import annotations

import dis
import types
from typing import Any


def _line_map(code: types.CodeType) -> dict[int, int]:
    """Map bytecode offset -> source line for `code`."""
    result: dict[int, int] = {}
    for start, end, line in code.co_lines():
        if line is not None:
            for offset in range(start, end, 2):
                result[offset] = line
    return result


def find_class_body_code(caller: types.FrameType, cls_name: str) -> types.CodeType | None:
    """
    Locate the code object of the class body currently being executed for `cls_name`.

    `caller` is the frame executing the `class ...:` statement; the body's code object is one of that
    frame's code constants. When the class name is unique among those constants we take it directly;
    otherwise (the same name defined more than once in one scope, e.g. in a loop or a redefinition) we
    disambiguate by taking the last matching `LOAD_CONST` at or before the frame's current instruction,
    which is the one feeding the in-progress `__build_class__` call.
    """
    code = caller.f_code
    candidates = [const for const in code.co_consts if isinstance(const, types.CodeType) and const.co_name == cls_name]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        return None
    result: types.CodeType | None = None
    for instr in dis.get_instructions(code):
        if instr.offset > caller.f_lasti:
            break
        if (
            instr.opname.startswith('LOAD_CONST')
            and isinstance(instr.argval, types.CodeType)
            and instr.argval.co_name == cls_name
        ):
            result = instr.argval
    return result


def _annotate_code(body_code: types.CodeType) -> types.CodeType | None:
    return next((c for c in body_code.co_consts if isinstance(c, types.CodeType) and c.co_name == '__annotate__'), None)


def annotation_lines(body_code: types.CodeType) -> list[tuple[str, int]]:
    """(name, line) for each annotation in `body_code`, in declaration order."""
    annotate = _annotate_code(body_code)
    if annotate is None:
        return []
    lines = _line_map(annotate)
    # The annotate function builds its dict by evaluating each annotation and storing it under the
    # column name; that name is the only string constant loaded per annotation.
    return [
        (instr.argval, lines[instr.offset])
        for instr in dis.get_instructions(annotate)
        if instr.opname.startswith('LOAD_CONST') and isinstance(instr.argval, str) and instr.offset in lines
    ]


def assignment_lines(body_code: types.CodeType) -> list[tuple[str, int]]:
    """(name, line) for each top-level name assignment in `body_code`, in declaration order."""
    lines = _line_map(body_code)
    return [
        (instr.argval, lines[instr.offset])
        for instr in dis.get_instructions(body_code)
        if instr.opname == 'STORE_NAME' and not instr.argval.startswith('__') and instr.offset in lines
    ]


def evaluate_annotations(
    body_code: types.CodeType, namespace: dict[str, Any], eval_globals: dict[str, Any]
) -> dict[str, Any]:
    """
    Evaluate the annotations of `body_code` to real type objects.

    The compiled `__annotate__` closes over a `__classdict__` cell that it consults before globals when
    resolving names; we supply the model namespace for that cell, mirroring how the annotations would
    have resolved had they been evaluated in the class body.
    """
    annotate = _annotate_code(body_code)
    if annotate is None:
        return {}
    closure = tuple(types.CellType(namespace) for _ in annotate.co_freevars)
    fn = types.FunctionType(annotate, eval_globals, '__annotate__', None, closure)
    return fn(1)  # annotationlib.Format.VALUE
