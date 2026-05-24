import abc
from typing import TYPE_CHECKING
from uuid import UUID

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

    @property
    @abc.abstractmethod
    def _name(self) -> str:
        """Current name of this object, as recorded in the catalog."""

    @property
    @abc.abstractmethod
    def _dir_id(self) -> UUID | None:
        """Current parent directory id of this object, as recorded in the catalog. None if root."""

    def _parent(self) -> 'catalog.Dir | None':
        """Returns the parent directory of this schema object."""
        with get_runtime().catalog.begin_xact(for_write=False):
            if self._dir_id is None:
                return None
            return get_runtime().catalog.get_dir(self._dir_id)

    def _path(self) -> str:
        """Returns the path to this schema object."""
        assert self._dir_id is not None
        with get_runtime().catalog.begin_xact(for_write=False):
            path = get_runtime().catalog.get_dir_path(self._dir_id)
            return str(path.append(self._name))

    @abc.abstractmethod
    def _display_name(self) -> str:
        """Return name displayed in error messages."""

    def _display_str(self) -> str:
        return f'{self._display_name()} {self._path()!r}'
