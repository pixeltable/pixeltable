import logging
from typing import Any

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs

_logger = logging.getLogger('pixeltable')


class TestLabelStudio:

    def test_remote_validation(self, reset_db):
        schema = {'col1': pxt.StringType(), 'col2': pxt.ImageType(), 'col3': pxt.StringType(), 'col4': pxt.VideoType()}
        t = pxt.create_table('test_remote', schema)

        remote1 = self.MockRemote(
            {'push1': pxt.StringType(), 'push2': pxt.ImageType()},
            {'pull1': pxt.StringType(), 'pull2': pxt.VideoType()}
        )

        # Nonexistent local column
        with pytest.raises(excs.Error) as exc_info:
            t.link_remote(remote1)
        assert 'Column `push1` does not exist' in str(exc_info.value)

        # Nonexistent remote column
        with pytest.raises(excs.Error) as exc_info:
            t.link_remote(remote1, {'col1': 'push1', 'col2': 'col2'})
        assert 'has no column `col2`' in str(exc_info.value)

        # Correct partial spec
        t.link_remote(remote1, {'col1': 'push1', 'col2': 'push2'})

        # Correct full spec
        t.link_remote(remote1, {'col1': 'push1', 'col2': 'push2', 'col3': 'pull1', 'col4': 'pull2'})

        # Default spec is correct
        schema2 = {'push1': pxt.StringType(), 'push2': pxt.ImageType(), 'pull1': pxt.StringType(), 'pull2': pxt.VideoType()}
        t2 = pxt.create_table('test_remote_2', schema2)
        t2.link_remote(remote1)

        # Incompatible types for push
        with pytest.raises(excs.Error) as exc_info:
            t.link_remote(remote1, {'col1': 'push2'})
        assert 'Column `col1` cannot be pushed to remote column `push2`' in str(exc_info.value)

        # Incompatible types for pull
        with pytest.raises(excs.Error) as exc_info:
            t.link_remote(remote1, {'col1': 'pull2'})
        assert 'Column `col1` cannot be pulled from remote column `pull2`' in str(exc_info.value)

        # Subtype/supertype relationships

        schema3 = {'img': pxt.ImageType(), 'spec_img': pxt.ImageType(512, 512)}
        t3 = pxt.create_table('test_remote_3', schema3)
        remote2 = self.MockRemote(
            {'push_img': pxt.ImageType(), 'push_spec_img': pxt.ImageType(512, 512)},
            {'pull_img': pxt.ImageType(), 'pull_spec_img': pxt.ImageType(512, 512)}
        )

        # Can push/pull from sub to supertype
        t3.link_remote(remote2, {'spec_img': 'push_img', 'img': 'pull_spec_img'})

        # Cannot push from super to subtype
        with pytest.raises(excs.Error) as exc_info:
            t3.link_remote(remote2, {'img': 'push_spec_img'})
        assert 'Column `img` cannot be pushed to remote column `push_spec_img`' in str(exc_info.value)

        # Cannot pull from super to subtype
        with pytest.raises(excs.Error) as exc_info:
            t3.link_remote(remote2, {'spec_img': 'pull_img'})
        assert 'Column `spec_img` cannot be pulled from remote column `pull_img`' in str(exc_info.value)

    class MockRemote(pxt.Remote):

        def __init__(self, push_cols: dict[str, pxt.ColumnType], pull_cols: dict[str, pxt.ColumnType]):
            self.push_cols = push_cols
            self.pull_cols = pull_cols

        def get_push_columns(self) -> dict[str, pxt.ColumnType]:
            return self.push_cols

        def get_pull_columns(self) -> dict[str, pxt.ColumnType]:
            return self.pull_cols

        def sync(self, t: pxt.Table, col_mapping: dict[str, str], push: bool, pull: bool) -> None:
            raise NotImplementedError()

        def to_dict(self) -> dict[str, Any]:
            return {'test_key': 'test_val'}

        @classmethod
        def from_dict(cls, md: dict[str, Any]) -> pxt.Remote:
            assert md == {'test_key': 'test_val'}
            return cls()
