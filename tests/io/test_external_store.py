import logging
import os.path
from typing import Optional

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.io.external_store import MockProject, Project
from pixeltable.exprs import ColumnRef
from tests.utils import get_image_files, reload_catalog

_logger = logging.getLogger('pixeltable')


class TestProject:

    def test_validation(self, reset_db):
        schema = {'col1': pxt.StringType(), 'col2': pxt.ImageType(), 'col3': pxt.StringType(), 'col4': pxt.VideoType()}
        t = pxt.create_table('test_store', schema)
        export_cols = {'export1': pxt.StringType(), 'export2': pxt.ImageType()}
        import_cols = {'import1': pxt.StringType(), 'import2': pxt.VideoType()}

        # Nonexistent local column
        with pytest.raises(excs.Error) as exc_info:
            Project.validate_columns(t, export_cols, import_cols, None)
        assert 'Column `export1` does not exist' in str(exc_info.value)

        # Nonexistent local column, but with a mapping specified
        with pytest.raises(excs.Error) as exc_info:
            Project.validate_columns(t, export_cols, import_cols, {'not_col': 'export1', 'col2': 'export2'})
        assert 'Column name `not_col` appears as a key' in str(exc_info.value)

        # Nonexistent external column
        with pytest.raises(excs.Error) as exc_info:
            Project.validate_columns(t, export_cols, import_cols, {'col1': 'export1', 'col2': 'col2'})
        assert 'has no column `col2`' in str(exc_info.value)

        # Correct partial spec
        Project.validate_columns(t, export_cols, import_cols, {'col1': 'export1', 'col2': 'export2'})

        # Correct full spec
        Project.validate_columns(t, export_cols, import_cols, {'col1': 'export1', 'col2': 'export2', 'col3': 'import1', 'col4': 'import2'})

        # Default spec is correct
        schema2 = {'export1': pxt.StringType(), 'export2': pxt.ImageType(), 'import1': pxt.StringType(), 'import2': pxt.VideoType()}
        t2 = pxt.create_table('test_2', schema2)
        Project.validate_columns(t2, export_cols, import_cols, None)

        # Incompatible types for export
        with pytest.raises(excs.Error) as exc_info:
            Project.validate_columns(t, export_cols, import_cols, {'col1': 'export2'})
        assert 'Column `col1` cannot be exported to external column `export2` (incompatible types; expecting `image`)' in str(exc_info.value)

        # Incompatible types for import
        with pytest.raises(excs.Error) as exc_info:
            Project.validate_columns(t, export_cols, import_cols, {'col1': 'import2'})
        assert 'Column `col1` cannot be imported from external column `import2` (incompatible types; expecting `video`)' in str(exc_info.value)

        # Subtype/supertype relationships

        schema3 = {'img': pxt.ImageType(), 'spec_img': pxt.ImageType(512, 512)}
        t3 = pxt.create_table('test_store_3', schema3)

        export_img_cols = {'export_img': pxt.ImageType(), 'export_spec_img': pxt.ImageType(512, 512)}
        import_img_cols = {'import_img': pxt.ImageType(), 'import_spec_img': pxt.ImageType(512, 512)}

        # Can export/import from sub to supertype
        Project.validate_columns(t3, export_img_cols, import_img_cols, {'spec_img': 'export_img', 'img': 'import_spec_img'})

        # Cannot export from super to subtype
        with pytest.raises(excs.Error) as exc_info:
            Project.validate_columns(t3, export_img_cols, import_img_cols, {'img': 'export_spec_img'})
        assert 'Column `img` cannot be exported to external column `export_spec_img`' in str(exc_info.value)

        # Cannot import from super to subtype
        with pytest.raises(excs.Error) as exc_info:
            Project.validate_columns(t3, export_img_cols, import_img_cols, {'spec_img': 'import_img'})
        assert 'Column `spec_img` cannot be imported from external column `import_img`' in str(exc_info.value)

        t3['computed_img'] = t3.img.rotate(180)
        with pytest.raises(excs.Error) as exc_info:
            Project.validate_columns(t3, export_img_cols, import_img_cols, {'computed_img': 'import_img'})
        assert (
            'Column `computed_img` is a computed column, which cannot be populated from an external column'
            in str(exc_info.value)
        )

        # Duplicate link
        t._link(MockProject.create(t, 'project', export_cols, import_cols, {'col1': 'export1', 'col2': 'export2'}))
        with pytest.raises(excs.Error) as exc_info:
            t._link(MockProject.create(t, 'project', export_cols, import_cols, {'col1': 'export1', 'col2': 'export2'}))
        assert 'Table `test_store` already has an external store with that name: project' in str(exc_info.value)

        # Cannot drop a linked column
        with pytest.raises(excs.Error) as exc_info:
            t.drop_column('col1')
        assert 'Cannot drop column `col1` because the following external stores depend on it' in str(exc_info.value)

    @pytest.mark.parametrize('with_reloads', [False, True])
    def test_stored_proxies(self, reset_db, with_reloads: bool) -> None:
        schema = {'img': pxt.ImageType(), 'other_img': pxt.ImageType()}
        t = pxt.create_table('test_store', schema)
        t.add_column(rot_img=t.img.rotate(180), stored=False)
        t.add_column(rot_other_img=t.other_img.rotate(180), stored=False)
        image_files = get_image_files()[:10]
        other_image_files = get_image_files()[-10:]
        t.insert(
            {'img': img, 'other_img': other_img}
            for img, other_img in zip(image_files[:5], other_image_files[:5])
        )
        assert not t.rot_img.col.is_stored
        assert not t.rot_other_img.col.is_stored
        assert t.rot_img.col.stored_proxy is None  # No stored proxy yet
        assert t.rot_other_img.col.stored_proxy is None

        if with_reloads:
            reload_catalog()
            t = pxt.get_table('test_store')

        num_cols_before_linking = len(t.tbl_version_path.tbl_version.cols_by_id)
        store1 = MockProject.create(
            t,
            'store1',
            {'push_img': pxt.ImageType(), 'push_other_img': pxt.ImageType()},
            {'pull_str': pxt.StringType()},
            {'rot_img': 'push_img', 'rot_other_img': 'push_other_img'}
        )
        t._link(store1)
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
            t = pxt.get_table('test_store')

        # Now we're going to add `store2`, which links just one of the two columns. We'll rename
        # a column before linking, to ensure that column associations are preserved by reference
        # (tbl/column ID), not by name.
        t.rename_column('rot_img', 'rot_img_renamed')
        assert t.rot_img_renamed.col.stored_proxy is not None
        assert t.rot_img_renamed.col.stored_proxy.proxy_base == t.rot_img_renamed.col
        store2 = MockProject.create(
            t,
            'store2',
            {'push_img': pxt.ImageType()},
            {'pull_str': pxt.StringType()},
            {'rot_img_renamed': 'push_img'}
        )
        t._link(store2)
        # Ensure the stored proxy is created just once (for both external stores)
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking + 2

        if with_reloads:
            reload_catalog()
            t = pxt.get_table('test_store')

        t.unlink('store1')
        # Now rot_img_renamed is still linked through store2, but rot_other_img
        # is not linked to any store. So just rot_img_renamed should have a proxy
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking + 1
        assert t.rot_img_renamed.col.stored_proxy is not None
        assert t.rot_other_img.col.stored_proxy is None

        if with_reloads:
            reload_catalog()
            t = pxt.get_table('test_store')

        t.unlink('store2')
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking
        assert t.rot_img_renamed.col.stored_proxy is None
