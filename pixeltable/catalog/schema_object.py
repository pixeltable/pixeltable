import abc
from typing import TYPE_CHECKING
from uuid import UUID

from pixeltable import exceptions as excs, hooks
from pixeltable.runtime import get_runtime

if TYPE_CHECKING:
    from pixeltable import catalog


class SchemaObject(abc.ABC):
    """Base class of all addressable objects within a Db.

    All subclass instances must be thread-safe. To guarantee that, all state is either immutable or thread-safe.
    """

    _id: UUID

    def __init__(self, obj_id: UUID):
        self._id = obj_id

    @abc.abstractmethod
    def _name(self) -> str:
        """Current name of this object, as recorded in the catalog."""

    @abc.abstractmethod
    def _dir_id(self) -> UUID | None:
        """Current parent directory id of this object, as recorded in the catalog. None if root."""

    def _parent(self) -> 'catalog.Dir | None':
        """Returns the parent directory of this schema object."""
        with get_runtime().catalog.begin_xact(for_write=False):
            dir_id = self._dir_id()
            if dir_id is None:
                return None
            return get_runtime().catalog.get_dir(dir_id)

    @hooks.spanned('pixeltable.catalog.resolve_path', level=hooks.DEBUG)
    def _path(self) -> 'catalog.Path':
        """Returns the path to this schema object. Raises TABLE_NOT_FOUND if dropped.

        Resolves the whole path in a single read transaction so the result is a consistent snapshot.
        """
        with get_runtime().catalog.begin_xact(for_write=False):
            dir_id = self._dir_id()
            if dir_id is None:
                # an instance that's in the process of getting dropped has dir_id unset
                raise excs.table_was_dropped(self._id)
            path = get_runtime().catalog.get_dir_path(dir_id)
            return path.append(self._name())

    @abc.abstractmethod
    def _display_name(self) -> str:
        """Return name displayed in error messages."""

    def _display_str(self, path: 'catalog.Path | None' = None) -> str:
        return f'{self._display_name()} {(path if path is not None else self._path())!r}'
