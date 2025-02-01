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

    def __init__(self, obj_id: UUID, name: str, dir_id: Optional[UUID]):
        # make these private so they don't collide with column names (id and name are fairly common)
        self.__id = obj_id
        self.__name = name
        self.__dir_id = dir_id

    @property
    def _id(self) -> UUID:
        return self.__id

    @property
    def _name(self) -> str:
        return self.__name

    @property
    def _dir_id(self) -> Optional[UUID]:
        return self.__dir_id

    @property
    def _parent(self) -> Optional['catalog.Dir']:
        """Returns the parent directory of this schema object."""
        from pixeltable import catalog

        if self._dir_id is None:
            return None
        dir = catalog.Catalog.get().paths.get_schema_obj(self._dir_id)
        assert isinstance(dir, catalog.Dir)
        return dir

    @property
    def _path(self) -> str:
        """Returns the path to this schema object."""
        parent = self._parent
        if parent is None or parent._parent is None:
            # Either this is the root directory, with empty path, or its parent is the
            # root directory. Either way, we return just the name.
            return self._name
        else:
            return f'{parent._path}.{self._name}'

    def get_metadata(self) -> dict[str, Any]:
        """Returns metadata associated with this schema object."""
        return {
            'name': self._name,
            'path': self._path,
            'parent': self._parent._path if self._parent is not None else None,
        }

    @classmethod
    @abstractmethod
    def _display_name(cls) -> str:
        """
        Return name displayed in error messages.
        """
        pass

    @property
    @abstractmethod
    def _has_dependents(self) -> bool:
        """Returns True if this object has dependents (e.g., children, views)"""

    def _move(self, new_name: str, new_dir_id: UUID) -> None:
        """Subclasses need to override this to make the change persistent"""
        self.__name = new_name
        self.__dir_id = new_dir_id
