"""test move"""
class PartSink:
    """Destination for a request's binary values during serialization.

    The base sink inlines everything into binary_parts (referenced by index from the JSON head). A subclass
    can route media values (tags 'file'/'image') out of band by returning an object key (a str) instead of a
    part index; scalars (tags 'bytes'/'ndarray') always stay inline.
    """

    binary_parts: list[bytes]

    def __init__(self) -> None:
        self.binary_parts = []

    def add_inline(self, data: bytes) -> int:
        """Append an inline binary part and return its index."""
        self.binary_parts.append(data)
        return len(self.binary_parts) - 1

    def add_media_bytes(self, data: bytes, extension: str) -> int | str:
        """Add an in-memory media value; returns a part index (inline) or an object key (out of band)."""
        return self.add_inline(data)

    def add_media_file(self, path: str) -> int | str:
        """Add a file-backed media value; returns a part index (inline) or an object key (out of band)."""
        with open(path, 'rb') as f:
            return self.add_inline(f.read())

