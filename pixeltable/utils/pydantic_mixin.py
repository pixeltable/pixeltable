"""
Pydantic serialization mixin for Pixeltable classes.

This mixin provides automatic Pydantic serialization/deserialization
using the as_dict() and from_dict() class methods.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic_core import CoreSchema, core_schema

T = TypeVar('T')


class PydanticSerializationMixin:
    """
    Mixin that provides Pydantic core schema using as_dict() and from_dict() methods.
    
    Classes that inherit from this mixin will automatically be serialized/deserialized
    by Pydantic using their as_dict() and from_dict() class methods.
    """
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,  # GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Provide Pydantic core schema for serialization/deserialization.
        
        Args:
            source_type: The source type being processed
            handler: Pydantic's schema handler
            
        Returns:
            CoreSchema that uses as_dict() and from_dict() for serialization
        """
        # Validate that the class has the required methods
        if not hasattr(cls, 'as_dict'):
            raise TypeError(f"{cls.__name__} must implement as_dict() method")
        if not hasattr(cls, 'from_dict'):
            raise TypeError(f"{cls.__name__} must implement from_dict() class method")
        
        def deserialize_from_dict(data: Any) -> Any:
            """Deserialize object using from_dict() class method."""
            if isinstance(data, cls):
                # Already the correct type, return as-is
                return data
            if isinstance(data, dict):
                # Deserialize from dict
                return cls.from_dict(data)
            raise TypeError(f"Expected {cls.__name__} or dict, got {type(data).__name__}")
        
        # Create a custom serializer function
        def serialize_to_dict(obj: Any, handler: Any) -> dict[str, Any]:
            """Serialize object using as_dict() method."""
            if not isinstance(obj, cls):
                raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
            return obj.as_dict()
        
        # Return a schema that handles both serialization and deserialization
        return core_schema.no_info_plain_validator_function(
            deserialize_from_dict,
            serialization=core_schema.wrap_serializer_function_ser_schema(
                serialize_to_dict,
                return_schema=core_schema.dict_schema(),
            ),
        )
