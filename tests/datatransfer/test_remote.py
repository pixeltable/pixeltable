import logging

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.datatransfer.remote import MockRemote

_logger = logging.getLogger('pixeltable')


class TestRemote:

    def test_remote_validation(self, reset_db):
        schema = {'col1': pxt.StringType(), 'col2': pxt.ImageType(), 'col3': pxt.StringType(), 'col4': pxt.VideoType()}
        t = pxt.create_table('test_remote', schema)

        remote1 = MockRemote(
            {'export1': pxt.StringType(), 'export2': pxt.ImageType()},
            {'import1': pxt.StringType(), 'import2': pxt.VideoType()}
        )

        # Nonexistent local column
        with pytest.raises(excs.Error) as exc_info:
            t._link(remote1)
        assert 'Column `export1` does not exist' in str(exc_info.value)

        # Nonexistent local column, but with a mapping specified
        with pytest.raises(excs.Error) as exc_info:
            t._link(remote1, {'not_col': 'export1', 'col2': 'export2'})
        assert 'Column name `not_col` appears as a key' in str(exc_info.value)

        # Nonexistent remote column
        with pytest.raises(excs.Error) as exc_info:
            t._link(remote1, {'col1': 'export1', 'col2': 'col2'})
        assert 'has no column `col2`' in str(exc_info.value)

        # Correct partial spec
        t._link(remote1, {'col1': 'export1', 'col2': 'export2'})
        t.unlink()

        # Correct full spec
        t._link(remote1, {'col1': 'export1', 'col2': 'export2', 'col3': 'import1', 'col4': 'import2'})
        t.unlink()

        # Default spec is correct
        schema2 = {'export1': pxt.StringType(), 'export2': pxt.ImageType(), 'import1': pxt.StringType(), 'import2': pxt.VideoType()}
        t2 = pxt.create_table('test_2', schema2)
        t2._link(remote1)
        t2.unlink()

        # Incompatible types for export
        with pytest.raises(excs.Error) as exc_info:
            t._link(remote1, {'col1': 'export2'})
        assert 'Column `col1` cannot be exported to remote column `export2`' in str(exc_info.value)

        # Incompatible types for import
        with pytest.raises(excs.Error) as exc_info:
            t._link(remote1, {'col1': 'import2'})
        assert 'Column `col1` cannot be imported from remote column `import2`' in str(exc_info.value)

        # Subtype/supertype relationships

        schema3 = {'img': pxt.ImageType(), 'spec_img': pxt.ImageType(512, 512)}
        t3 = pxt.create_table('test_remote_3', schema3)
        remote2 = MockRemote(
            {'export_img': pxt.ImageType(), 'export_spec_img': pxt.ImageType(512, 512)},
            {'import_img': pxt.ImageType(), 'import_spec_img': pxt.ImageType(512, 512)}
        )

        # Can export/import from sub to supertype
        t3._link(remote2, {'spec_img': 'export_img', 'img': 'import_spec_img'})
        t3.unlink()

        # Cannot export from super to subtype
        with pytest.raises(excs.Error) as exc_info:
            t3._link(remote2, {'img': 'export_spec_img'})
        assert 'Column `img` cannot be exported to remote column `export_spec_img`' in str(exc_info.value)

        # Cannot import from super to subtype
        with pytest.raises(excs.Error) as exc_info:
            t3._link(remote2, {'spec_img': 'import_img'})
        assert 'Column `spec_img` cannot be imported from remote column `import_img`' in str(exc_info.value)

        t3['computed_img'] = t3.img.rotate(180)
        with pytest.raises(excs.Error) as exc_info:
            t3._link(remote2, {'computed_img': 'import_img'})
        assert (
            'Column `computed_img` is a computed column, which cannot be populated from a remote column'
            in str(exc_info.value)
        )
