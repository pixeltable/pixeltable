from abc import abstractmethod
from typing import TYPE_CHECKING, Optional
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
        self._id = obj_id
        self._name = name
        self._dir_id = dir_id

    def _get_id(self) -> UUID:
        return self._id

    @property
    def parent(self) -> Optional['catalog.Dir']:
        """Returns the parent directory of this schema object."""
        from pixeltable import catalog
        if self._dir_id is None:
            return None
        dir = catalog.Catalog.get().paths.get_schema_obj(self._dir_id)
        assert isinstance(dir, catalog.Dir)
        return dir

    @property
    def path(self) -> str:
        """Returns the path to this schema object."""
        parent = self.parent
        if parent is None or parent.parent is None:
            # Either this is the root directory, with empty path, or its parent is the
            # root directory. Either way, we return just the name.
            return self._name
        else:
            return f'{parent.path}.{self._name}'

    @classmethod
    @abstractmethod
    def display_name(cls) -> str:
        """
        Return name displayed in error messages.
        """
        pass

    def _move(self, new_name: str, new_dir_id: UUID) -> None:
        """Subclasses need to override this to make the change persistent"""
        self._name = new_name
        self._dir_id = new_dir_id
