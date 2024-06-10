import logging
import os.path
from typing import Optional

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.io.external_store import MockProject
from pixeltable.exprs import ColumnRef
from tests.utils import get_image_files, reload_catalog

_logger = logging.getLogger('pixeltable')


class TestProject:

    def test_validation(self, reset_db):
        schema = {'col1': pxt.StringType(), 'col2': pxt.ImageType(), 'col3': pxt.StringType(), 'col4': pxt.VideoType()}
        t = pxt.create_table('test_remote', schema)

        def make_remote(col_mapping: Optional[dict[str, str]]) -> MockProject:
            return MockProject(
                name='remote',
                export_cols={'export1': pxt.StringType(), 'export2': pxt.ImageType()},
                import_cols={'import1': pxt.StringType(), 'import2': pxt.VideoType()},
                col_mapping=col_mapping
            )

        # Nonexistent local column
        with pytest.raises(excs.Error) as exc_info:
            t._link(make_remote(None))
        assert 'Column `export1` does not exist' in str(exc_info.value)

        # Nonexistent local column, but with a mapping specified
        with pytest.raises(excs.Error) as exc_info:
            t._link(make_remote({'not_col': 'export1', 'col2': 'export2'}))
        assert 'Column name `not_col` appears as a key' in str(exc_info.value)

        # Nonexistent remote column
        with pytest.raises(excs.Error) as exc_info:
            t._link(make_remote({'col1': 'export1', 'col2': 'col2'}))
        assert 'has no column `col2`' in str(exc_info.value)

        # Correct partial spec
        t._link(make_remote({'col1': 'export1', 'col2': 'export2'}))

        # Duplicate link
        with pytest.raises(excs.Error) as exc_info:
            t._link(make_remote({'col1': 'push1', 'col2': 'col2'}))
        assert 'That remote is already linked to table `test_remote`: MockProject `remote`' in str(exc_info.value)

        t.unlink()

        # Correct full spec
        t._link(make_remote({'col1': 'export1', 'col2': 'export2', 'col3': 'import1', 'col4': 'import2'}))
        t.unlink()

        # Default spec is correct
        schema2 = {'export1': pxt.StringType(), 'export2': pxt.ImageType(), 'import1': pxt.StringType(), 'import2': pxt.VideoType()}
        t2 = pxt.create_table('test_2', schema2)
        t2._link(make_remote(None))
        t2.unlink()

        # Incompatible types for export
        with pytest.raises(excs.Error) as exc_info:
            t._link(make_remote({'col1': 'export2'}))
        assert 'Column `col1` cannot be exported to remote column `export2` (incompatible types; expecting `image`)' in str(exc_info.value)

        # Incompatible types for import
        with pytest.raises(excs.Error) as exc_info:
            t._link(make_remote({'col1': 'import2'}))
        assert 'Column `col1` cannot be imported from remote column `import2` (incompatible types; expecting `video`)' in str(exc_info.value)

        # Subtype/supertype relationships

        schema3 = {'img': pxt.ImageType(), 'spec_img': pxt.ImageType(512, 512)}
        t3 = pxt.create_table('test_remote_3', schema3)

        def make_remote_2(col_mapping: Optional[dict[str, str]]) -> MockProject:
            return MockProject(
                'remote2',
                {'export_img': pxt.ImageType(), 'export_spec_img': pxt.ImageType(512, 512)},
                {'import_img': pxt.ImageType(), 'import_spec_img': pxt.ImageType(512, 512)},
                col_mapping=col_mapping
            )

        # Can export/import from sub to supertype
        t3._link(make_remote_2({'spec_img': 'export_img', 'img': 'import_spec_img'}))

        # Cannot drop a linked column
        with pytest.raises(excs.Error) as exc_info:
            t3.drop_column('spec_img')
        assert 'Cannot drop column `spec_img` because the following remotes depend on it' in str(exc_info.value)

        t3.unlink()

        # Cannot export from super to subtype
        with pytest.raises(excs.Error) as exc_info:
            t3._link(make_remote_2({'img': 'export_spec_img'}))
        assert 'Column `img` cannot be exported to remote column `export_spec_img`' in str(exc_info.value)

        # Cannot import from super to subtype
        with pytest.raises(excs.Error) as exc_info:
            t3._link(make_remote_2({'spec_img': 'import_img'}))
        assert 'Column `spec_img` cannot be imported from remote column `import_img`' in str(exc_info.value)

        t3['computed_img'] = t3.img.rotate(180)
        with pytest.raises(excs.Error) as exc_info:
            t3._link(make_remote_2({'computed_img': 'import_img'}))
        assert (
            'Column `computed_img` is a computed column, which cannot be populated from a remote column'
            in str(exc_info.value)
        )

    @pytest.mark.parametrize('with_reloads', [False, True])
    def test_remote_stored_proxies(self, reset_db, with_reloads: bool) -> None:
        schema = {'img': pxt.ImageType(), 'other_img': pxt.ImageType()}
        t = pxt.create_table('test_remote', schema)
        remote1 = MockProject(
            'remote1',
            {'push_img': pxt.ImageType(), 'push_other_img': pxt.ImageType()},
            {'pull_str': pxt.StringType()},
            {'rot_img': 'push_img', 'rot_other_img': 'push_other_img'}
        )
        remote2 = MockProject(
            'remote2',
            {'push_img': pxt.ImageType()},
            {'pull_str': pxt.StringType()},
            {'rot_img': 'push_img'}
        )
        image_files = get_image_files()[:10]
        other_image_files = get_image_files()[-10:]
        t.insert(
            {'img': img, 'other_img': other_img}
            for img, other_img in zip(image_files[:5], other_image_files[:5])
        )
        t.add_column(rot_img=t.img.rotate(180), stored=False)
        t.add_column(rot_other_img=t.other_img.rotate(180), stored=False)
        assert not t.rot_img.col.is_stored
        assert not t.rot_other_img.col.is_stored
        assert t.rot_img.col.stored_proxy is None  # No stored proxy yet
        assert t.rot_other_img.col.stored_proxy is None

        if with_reloads:
            reload_catalog()
            t = pxt.get_table('test_remote')

        num_cols_before_linking = len(t.tbl_version_path.tbl_version.cols_by_id)
        t._link(remote1)
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking + 2
        assert t.rot_img.col.stored_proxy is not None  # Stored proxy
        assert t.rot_img.col.stored_proxy.proxy_base == t.rot_img.col
        assert t.rot_other_img.col.stored_proxy is not None
        assert t.rot_other_img.col.stored_proxy.proxy_base == t.rot_other_img.col
        # Verify that the stored proxies properly materialized, and we can query them
        ref = ColumnRef(t.rot_img.col.stored_proxy)
        proxies = t.select(img=ref, path=ref.localpath).collect()
        assert all(os.path.isfile(proxies['path'][i]) for i in range(len(proxies)))
        proxies['img'][0].load()

        if with_reloads:
            reload_catalog()
            t = pxt.get_table('test_remote')

        t._link(remote2)
        # Ensure the stored proxy is created just once (for both remotes)
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking + 2

        if with_reloads:
            reload_catalog()
            t = pxt.get_table('test_remote')

        t.unlink(remote1)
        # Now rot_img_col is still linked through remote2, but rot_other_img_col
        # is not linked to any remote. So just rot_img_col should have a proxy
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking + 1
        assert t.rot_img.col.stored_proxy is not None
        assert t.rot_other_img.col.stored_proxy is None

        if with_reloads:
            reload_catalog()
            t = pxt.get_table('test_remote')

        t.unlink(remote2)
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking
        assert t.rot_img.col.stored_proxy is None
