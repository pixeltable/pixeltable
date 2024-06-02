import logging
import os.path

import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.datatransfer.remote import MockRemote
from pixeltable.exprs import ColumnRef
from tests.utils import get_image_files

_logger = logging.getLogger('pixeltable')


class TestRemote:

    def test_remote_validation(self, reset_db):
        schema = {'col1': pxt.StringType(), 'col2': pxt.ImageType(), 'col3': pxt.StringType(), 'col4': pxt.VideoType()}
        t = pxt.create_table('test_remote', schema)

        remote1 = MockRemote(
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
        remote2 = MockRemote(
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

        t3['computed_img'] = t3.img.rotate(180)
        with pytest.raises(excs.Error) as exc_info:
            t3.link_remote(remote2, {'computed_img': 'pull_img'})
        assert (
            'Column `computed_img` is a computed column, which cannot be populated from a remote column'
            in str(exc_info.value)
        )

        # Cannot drop a linked column
        with pytest.raises(excs.Error) as exc_info:
            t3.drop_column('spec_img')
        assert 'Cannot drop column `spec_img` because the following remotes depend on it' in str(exc_info.value)

    def test_remote_stored_proxies(self, reset_db) -> None:
        schema = {'img': pxt.ImageType(), 'other_img': pxt.ImageType()}
        t = pxt.create_table('test_remote', schema)
        remote1 = MockRemote(
            {'push_img': pxt.ImageType(), 'push_other_img': pxt.ImageType()},
            {'pull_str': pxt.StringType()}
        )
        remote2 = MockRemote(
            {'push_img': pxt.ImageType()},
            {'pull_str': pxt.StringType()}
        )
        image_files = get_image_files()[:10]
        other_image_files = get_image_files()[-10:]
        t.insert(
            {'img': img, 'other_img': other_img}
            for img, other_img in zip(image_files[:5], other_image_files[:5])
        )
        t.add_column(rot_img=t.img.rotate(180), stored=False)
        t.add_column(rot_other_img=t.other_img.rotate(180), stored=False)
        rot_img_col = t.tbl_version_path.get_column('rot_img')
        rot_other_img_col = t.tbl_version_path.get_column('rot_other_img')
        assert not rot_img_col.is_stored
        assert not rot_other_img_col.is_stored
        assert rot_img_col.stored_proxy is None  # No stored proxy yet
        assert rot_other_img_col.stored_proxy is None

        num_cols_before_linking = len(t.tbl_version_path.tbl_version.cols_by_id)
        t.link_remote(remote1, {'rot_img': 'push_img', 'rot_other_img': 'push_other_img'})
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking + 2
        assert rot_img_col.stored_proxy is not None  # Stored proxy
        assert rot_img_col.stored_proxy.proxy_base == rot_img_col
        assert rot_other_img_col.stored_proxy is not None
        assert rot_other_img_col.stored_proxy.proxy_base == rot_other_img_col
        # Verify that the stored proxies properly materialized, and we can query them
        ref = ColumnRef(rot_img_col.stored_proxy)
        proxies = t.select(img=ref, path=ref.localpath).collect()
        assert all(os.path.isfile(proxies['path'][i]) for i in range(len(proxies)))
        proxies['img'][0].load()

        t.link_remote(remote2, {'rot_img': 'push_img'})
        # Ensure the stored proxy is created just once (for both remotes)
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking + 2
        t.unlink_remote(remote1)
        # Now rot_img_col is still linked through remote2, but rot_other_img_col
        # is not linked to any remote. So just rot_img_col should have a proxy
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking + 1
        assert rot_img_col.stored_proxy is not None
        assert rot_other_img_col.stored_proxy is None
        t.unlink_remote(remote2)
        assert len(t.tbl_version_path.tbl_version.cols_by_id) == num_cols_before_linking
        assert rot_img_col.stored_proxy is None
