from abc import abstractmethod
from typing import Optional
from uuid import UUID


class SchemaObject:
    """
    Base class of all addressable objects within a Db.
    Each object has an id, a name and a parent directory.
    """
    def __init__(self, obj_id: UUID, name: str, dir_id: Optional[UUID]):
        self.id = obj_id
        self.name = name
        self.dir_id = dir_id

    @classmethod
    @abstractmethod
    def display_name(cls) -> str:
        """
        Return name displayed in error messages.
        """
        pass

    @property
    def fqn(self) -> str:
        return f'{self.parent_dir().fqn}.{self.name}'

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        """Subclasses need to override this to make the change persistent"""
        self.name = new_name
        self.dir_id = new_dir_id

