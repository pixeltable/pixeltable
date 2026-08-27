# ruff: noqa: F821
# ruff: noqa: N806
# ruff: noqa: RUF012

# NOTE: this module deliberately omits `from __future__ import annotations`. On Python 3.14+ that leaves the
# column annotations deferred (PEP 649), exercising the recovery path in
# pixeltable.catalog.model._annotation_recovery; on earlier versions the same declarations go through the
# eager path. The assertions hold either way, so this file also pins the two paths to each other.

from typing import Any, Callable

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.catalog.model import BtreeIndex, Column

from .utils import pxt_raises


class TestTableModelAnnotations:
    def test_declaration_order(self) -> None:
        """Annotations and assignments are recorded in source order, interleaved."""
        TableModel = pxt.model_base()

        class Interleaved(TableModel, name='interleaved'):
            x: pxt.Int
            y = x + 1
            z: pxt.Int
            w = z + 2
            v: pxt.String

        assert list(Interleaved.__columns__.keys()) == ['x', 'y', 'z', 'w', 'v']

    def test_type_dependent_references(self) -> None:
        """Mid-body references resolve against the annotation's real type, not an untyped placeholder."""
        TableModel = pxt.model_base()

        class TypeDependent(TableModel, name='type_dependent'):
            img: pxt.Image
            name: pxt.String
            arr: pxt.Array[pxt.Float, (4, 4)]
            n: pxt.Int
            rotated = img.rotate(90)  # dispatches on Image
            upper = name.upper()  # dispatches on String
            sliced = arr[:, 1:3]  # dispatches on Array
            arith = n * 2 + 1
            compared = n > 3
            member = n.isin([1, 2, 3])  # type: ignore[attr-defined]
            inline = {'a': n, 'b': [name, upper]}

        cols = TypeDependent.__columns__
        assert list(cols.keys()) == [
            'img', 'name', 'arr', 'n',
            'rotated', 'upper', 'sliced', 'arith', 'compared', 'member', 'inline',
        ]  # fmt: skip
        assert cols['rotated']['value'].col_type.is_image_type()
        assert cols['upper']['value'].col_type.is_string_type()
        assert cols['sliced']['value'].col_type.is_array_type()
        assert cols['compared']['value'].col_type.is_bool_type()
        assert cols['inline']['value'].col_type.is_json_type()

    def test_annotated_assignment(self) -> None:
        """`col: T = Column(...)` declares a single column, in the position where it appears."""
        TableModel = pxt.model_base()

        class Annotated(TableModel, name='annotated'):
            a: pxt.Int
            b: pxt.Int = Column(value=a + 1, stored=False)  # type: ignore[assignment]
            c: pxt.String

        assert list(Annotated.__columns__.keys()) == ['a', 'b', 'c']
        assert Annotated.__columns__['b']['value'].col_type.is_int_type()

    def test_conflicting_annotation(self) -> None:
        """An annotation that disagrees with the assigned value's type is rejected."""
        TableModel = pxt.model_base()

        with pxt_raises(pxt.ErrorCode.INVALID_SCHEMA, match='Conflicting type annotation'):

            class Conflicting(TableModel, name='conflicting'):
                a: pxt.Int
                b: pxt.String = Column(value=a + 1)  # type: ignore[assignment]

    def test_same_class_name_twice(self) -> None:
        """Two model classes with the same Python name in one scope are kept distinct."""
        TableModel = pxt.model_base()

        class Dup(TableModel, name='dup_one'):
            a: pxt.Int
            b = a + 1

        First = Dup

        class Dup(TableModel, name='dup_two'):  # type: ignore[no-redef]
            c: pxt.String
            d = c.upper()

        assert list(First.__columns__.keys()) == ['a', 'b']
        assert list(Dup.__columns__.keys()) == ['c', 'd']

    def test_defined_in_loop(self) -> None:
        """The correct class body is located when the same name is built repeatedly."""
        TableModel = pxt.model_base()
        models = []
        for i in range(3):

            class Looped(TableModel, name=f'looped_{i}'):
                a: pxt.Int
                b = a + 1

            models.append(Looped)

        assert [m.__table_spec__['name'] for m in models] == ['looped_0', 'looped_1', 'looped_2']
        assert all(list(m.__columns__.keys()) == ['a', 'b'] for m in models)

    def test_no_source_available(self) -> None:
        """A model declared via exec() has no retrievable source, but is still recovered from bytecode."""
        TableModel = pxt.model_base()
        src = 'class Exec(TableModel, name="from_exec"):\n a: pxt.Int\n b = a + 1\n c: pxt.String\n'
        scope: dict[str, Any] = {'TableModel': TableModel, 'pxt': pxt}
        exec(compile(src, '<no-source>', 'exec'), scope)

        assert list(scope['Exec'].__columns__.keys()) == ['a', 'b', 'c']

    def test_reserved_columns(self) -> None:
        """Base-query and iterator columns stay referenceable and are not treated as redeclarations."""
        from pixeltable.functions.video import frame_iterator

        TableModel = pxt.model_base()

        class Base(TableModel, name='base'):
            vid: pxt.Video
            val: pxt.Int

        class Projected(TableModel, name='projected', base=Base.select(v=Base.val)):
            plus = v + 1  # type: ignore[name-defined]  # the select() alias, referenceable in the body
            extra: pxt.String

        class Frames(TableModel, name='frames', base=Base, iterator=frame_iterator(video=Base.vid, fps=1)):
            tagged = frame.rotate(90)  # type: ignore[name-defined]  # the iterator output, referenceable in the body
            label: pxt.String

        assert list(Projected.__columns__.keys()) == ['plus', 'extra']
        assert list(Frames.__columns__.keys()) == ['tagged', 'label']

    def test_redeclared_reserved_column(self) -> None:
        """Annotating a name the iterator already produces is still an error."""
        from pixeltable.functions.video import frame_iterator

        TableModel = pxt.model_base()

        class Base(TableModel, name='base'):
            vid: pxt.Video

        with pxt_raises(pxt.ErrorCode.INVALID_SCHEMA, match='cannot be redeclared'):

            class Frames(TableModel, name='frames', base=Base, iterator=frame_iterator(video=Base.vid, fps=1)):
                frame: pxt.Image

    def test_parity_with_create_table(self, make_catalog_path: Callable[[str], str]) -> None:
        """A model builds the same schema as the equivalent create_table()/add_computed_column() calls."""
        p = make_catalog_path
        TableModel = pxt.model_base()

        class Parity(TableModel, name='parity'):
            id: pxt.Int
            name: pxt.String | None
            img: pxt.Image | None
            incr = id + 1
            descr = pxtf.string.format('Name: {name}', name=name)
            __indexes__ = [BtreeIndex(id)]

        TableModel.create_all(p(''))
        from_model = Parity.table

        ref = pxt.create_table(
            f'{p("")}/ref'.lstrip('/'), {'id': pxt.Int, 'name': pxt.String | None, 'img': pxt.Image | None}
        )
        ref.add_computed_column(incr=ref.id + 1)
        ref.add_computed_column(descr=pxtf.string.format('Name: {name}', name=ref.name))

        assert [c.name for c in from_model._tbl_path.column_md()] == [c.name for c in ref._tbl_path.column_md()]
        assert [c.col_type for c in from_model._tbl_path.column_md()] == [c.col_type for c in ref._tbl_path.column_md()]
