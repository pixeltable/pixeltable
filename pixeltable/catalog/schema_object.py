from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

import pixeltable.env as env

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
        with env.Env.get().begin():
            if self._dir_id is None:
                return None
            return Catalog.get().get_dir(self._dir_id)

    def _path(self) -> str:
        """Returns the path to this schema object."""
        with env.Env.get().begin():
            from .catalog import Catalog

            cat = Catalog.get()
            dir_path = cat.get_dir_path(self._dir_id)
            if dir_path == '':
                # Either this is the root directory, with empty path, or its parent is the
                # root directory. Either way, we return just the name.
                return self._name
            else:
                return f'{dir_path}.{self._name}'

    def get_metadata(self) -> dict[str, Any]:
        """Returns metadata associated with this schema object."""
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
