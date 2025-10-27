from abc import abstractmethod
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from pixeltable import catalog


class SchemaObject:
    """
    Base class of all addressable objects within a Db.
    Each object has an id, a name and a parent directory.
    """

    _id: UUID
    _name: str
    _dir_id: UUID | None

    def __init__(self, obj_id: UUID, name: str, dir_id: UUID | None):
        # make these private so they don't collide with column names (id and name are fairly common)
        assert dir_id is None or isinstance(dir_id, UUID), type(dir_id)
        self._id = obj_id
        self._name = name
        self._dir_id = dir_id

    def _parent(self) -> 'catalog.Dir | None':
        """Returns the parent directory of this schema object."""
        from .catalog import Catalog

        with Catalog.get().begin_xact(for_write=False):
            if self._dir_id is None:
                return None
            return Catalog.get().get_dir(self._dir_id)

    def _path(self) -> str:
        """Returns the path to this schema object."""
        from .catalog import Catalog

        assert self._dir_id is not None
        with Catalog.get().begin_xact(for_write=False):
            path = Catalog.get().get_dir_path(self._dir_id)
            return str(path.append(self._name))

    @abstractmethod
    def _display_name(self) -> str:
        """
        Return name displayed in error messages.
        """
        pass

    def _display_str(self) -> str:
        return f'{self._display_name()} {self._path()!r}'

    def _move(self, new_name: str, new_dir_id: UUID) -> None:
        """Subclasses need to override this to make the change persistent"""
        self._name = new_name
        self._dir_id = new_dir_id
