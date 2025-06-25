from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional
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
    _dir_id: Optional[UUID]

    def __init__(self, obj_id: UUID, name: str, dir_id: Optional[UUID]):
        # make these private so they don't collide with column names (id and name are fairly common)
        self._id = obj_id
        self._name = name
        self._dir_id = dir_id

    def _parent(self) -> Optional['catalog.Dir']:
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

    def get_metadata(self) -> dict[str, Any]:
        """Returns metadata associated with this schema object."""
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(for_write=False):
            return self._get_metadata()

    def _get_metadata(self) -> dict[str, Any]:
        return {'name': self._name, 'path': self._path()}

    @classmethod
    @abstractmethod
    def _display_name(cls) -> str:
        """
        Return name displayed in error messages.
        """
        pass

    def _move(self, new_name: str, new_dir_id: UUID) -> None:
        """Subclasses need to override this to make the change persistent"""
        self._name = new_name
        self._dir_id = new_dir_id
