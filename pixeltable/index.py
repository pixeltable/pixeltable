import hnswlib
import numpy as np

from pixeltable import env

class VectorIndex:
    def __init__(self, name: str, dim: int, idx: hnswlib.Index):
        self.name = name
        self.dim = dim
        self.idx = idx

    @classmethod
    def create(cls, name: str, dim: int) -> 'VectorIndex':
        idx = hnswlib.Index(space='cosine', dim=dim)
        idx.init_index(max_elements=1000, M=64, ef_construction=200)
        filename = cls._filename(name)
        idx.save_index(filename)
        return VectorIndex(name, dim, idx)

    @classmethod
    def load(cls, name: str, dim: int) -> 'VectorIndex':
        idx = hnswlib.Index(space='cosine', dim=dim)
        filename = cls._filename(name)
        idx.load_index(filename)
        return VectorIndex(name, dim, idx)

    def insert(self, data: np.ndarray, rowids: np.ndarray) -> None:
        assert data.shape[0] == rowids.shape[0]
        total = self.idx.element_count + data.shape[0]
        if total > self.idx.max_elements:
            self.idx.resize_index(int(total * 1.1))
        self.idx.add_items(data, rowids)
        filename = self._filename(self.name)
        self.idx.save_index(filename)

    @classmethod
    def _filename(cls, name: str) -> str:
        return str(env.get_nnidx_dir() / f'idx_{name}')