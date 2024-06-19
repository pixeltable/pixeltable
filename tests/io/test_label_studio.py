import logging
import os
import platform
import subprocess
import time
import uuid
from typing import Iterator, Literal

import pytest
import requests.exceptions

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import InsertableTable
from pixeltable.functions.string import str_format
from ..utils import (skip_test_if_not_installed, get_image_files, validate_update_status, reload_catalog,
                     SAMPLE_IMAGE_URL, get_video_files, get_audio_files, validate_sync_status)

_logger = logging.getLogger('pixeltable')


@pytest.mark.skipif(platform.system() == 'Windows', reason='Label Studio tests do not currently run on Windows')
class TestLabelStudio:

    test_config_image = """
    <View>
        <Image name="image_object" value="$image"/>
        <Choices name="class" toName="image_object">
          <Choice value="Cat"/>
          <Choice value="Dog"/>
        </Choices>
    </View>
    """
    test_config_video = """
    <View>
        <Video name="video_object" value="$video"/>
        <Choices name="class" toName="video_object">
          <Choice value="Cat"/>
          <Choice value="Dog"/>
        </Choices>
    </View>
    """
    test_config_audio = """
    <View>
        <Audio name="audio_object" value="$audio"/>
        <Choices name="class" toName="audio_object">
          <Choice value="Cat"/>
          <Choice value="Dog"/>
        </Choices>
    </View>
    """
    test_config_with_text = """
    <View>
        <Image name="image_object" value="$image"/>
        <Text name="text" value="$text"/>
        <Choices name="image_class" toName="image_object">
          <Choice value="Cat"/>
          <Choice value="Dog"/>
        </Choices>
    </View>
    """
    test_config_with_rl = """
    <View>
      <Image name="frame_obj" value="$frame"/>
      <RectangleLabels name="obj_label" toName="frame_obj">
        <Label value="knife" background="green"/>
        <Label value="person" background="blue"/>
      </RectangleLabels>
    </View>
    """
    test_config_complex = """
    <View>
        <Header value="$header"/>
        <Text name="text_obj" value="$text"/>
        <Image name="frame_obj" value="$frame"/>
        <Image name="rot_frame_obj" value="$rot_frame"/>
        <Choices name="image_class" toName="frame_obj">
          <Choice value="Cat"/>
          <Choice value="Dog"/>
        </Choices>
    </View>
    """

    def test_label_studio_project(self, ls_image_table: pxt.InsertableTable) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        t = ls_image_table

        pxt.io.create_label_studio_project(
            t,
            self.test_config_with_text,
            name='test_project',
            title='Test Project',
            media_import_method='file',
            col_mapping={'image_col': 'image'},
            sync_immediately=False
        )
        store = t._tbl_version.external_stores['test_project']
        assert store.name == 'test_project'
        assert store.project_title == 'Test Project'
        assert store.get_export_columns() == {'image': pxt.ImageType(), 'text': pxt.StringType()}
        assert store.get_import_columns() == {'annotations': pxt.JsonType(nullable=True)}

        with pytest.raises(excs.Error) as exc_info:
            pxt.io.create_label_studio_project(
                t,
                """
                <View>
                  <Image name="frame_obj" value="$frame"/>
                  <RectangleLabels name="obj_label" toName="walnut">
                    <Label value="knife" background="green"/>
                    <Label value="person" background="blue"/>
                  </RectangleLabels>
                </View>
                """
            )
        assert '`toName` attribute of RectangleLabels `obj_label` references an unknown data key: `walnut`' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            pxt.io.create_label_studio_project(
                t,
                """
                <View>
                  <Image name="frame_obj" value="$frame"/>
                  <RectangleLabels name="obj_label" toName="frame_obj">
                    <Label value="car" background="green"/>
                    <Label value="green gorilla" background="blue"/>
                  </RectangleLabels>
                </View>
                """
            )
        assert 'not a valid COCO object name' in str(exc_info.value)

    # Run the basic sync test four ways: with 'post' and 'file' import methods, and with
    # a stored and non-stored media column, in all combinations.
    @pytest.mark.parametrize(
        'media_import_method,sync_col',
        [('post', 'image_col'), ('file', 'image_col'), ('url', 'image_col'),
         # 'url' is not supported for computed columns (yet).
         # TODO(aaron-siegel): Validate and test this.
         ('post', 'rot_image_col'), ('file', 'rot_image_col')]
    )
    def test_label_studio_sync_images(
            self,
            ls_image_table: pxt.InsertableTable,
            media_import_method: Literal['post', 'file', 'url'],
            sync_col: str
    ) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        self.__test_label_studio_sync(ls_image_table, self.test_config_image, media_import_method, sync_col, 'image')

    # TODO(aaron-siegel): 'file' is not working for videos or audio yet.
    @pytest.mark.parametrize('media_import_method', ['post', 'url'])
    def test_label_studio_sync_videos(
            self,
            ls_video_table: pxt.InsertableTable,
            media_import_method: Literal['post', 'file', 'url']
    ) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        self.__test_label_studio_sync(ls_video_table, self.test_config_video, media_import_method, 'video_col', 'video')

    @pytest.mark.parametrize('media_import_method', ['post', 'url'])
    def test_label_studio_sync_audio(
            self,
            ls_audio_table: pxt.InsertableTable,
            media_import_method: Literal['post', 'file', 'url']
    ) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        self.__test_label_studio_sync(ls_audio_table, self.test_config_audio, media_import_method, 'audio_col', 'audio')

    @classmethod
    def __test_label_studio_sync(
            cls,
            t: InsertableTable,
            label_config: str,
            media_import_method: Literal['post', 'file', 'url'],
            sync_col: str,
            r_col: str
    ) -> None:
        sync_status = pxt.io.create_label_studio_project(
            t,
            label_config=label_config,
            media_import_method=media_import_method,
            col_mapping={sync_col: r_col, 'annotations_col': 'annotations'}
        )
        t_count = t.count()
        validate_sync_status(sync_status, t_count, 0, 0, 0, 0)

        # Check that the project and tasks were properly created
        store = t._tbl_version.external_stores['ls_project_0']
        tasks = store.project.get_tasks()
        assert len(tasks) == t_count
        assert all(task['data'][r_col] for task in tasks)
        if media_import_method == 'file':
            # Ensure all image filepaths are properly formatted for Label Studio file import
            assert all(task['data'][r_col].startswith('/data/local-files/?d=media/') for task in tasks)
        if media_import_method == 'url':
            assert all(cls.__is_expected_url(task['data'][r_col]) for task in tasks)

        # Programmatically add annotations by calling the Label Studio API directly
        for task in tasks[:10]:
            task_id = task['id']
            assert len(store.project.get_task(task_id)['annotations']) == 0
            store.project.create_annotation(
                task_id=task_id,
                unique_id=str(uuid.uuid4()),
                result=[{'class': 'Cat'}]
            )
            assert len(store.project.get_task(task_id)['annotations']) == 1

        # Import the annotations back to Pixeltable
        reload_catalog()
        t = pxt.get_table('test_ls_sync')
        ann_count = min(10, t_count)
        sync_status = t.sync()
        validate_sync_status(sync_status, 0, 0, 0, ann_count, 0)
        annotations_col = t.collect()['annotations_col']
        annotations = [a for a in annotations_col if a is not None]
        assert len(annotations) == ann_count
        assert all(annotations[i][0]['result'][0]['class'] == 'Cat' for i in range(ann_count)), annotations

        # Sync the annotations again, verify that there are no changes (no writes should happen)
        sync_status = t.sync()
        validate_sync_status(sync_status, 0, 0, 0, 0, 0)

        # Delete some random rows in Pixeltable and sync external stores again
        rows_to_delete = range(0, t_count, 3)
        validate_update_status(t.delete(where=t.id.isin(rows_to_delete)), expected_rows=len(rows_to_delete))
        sync_status = t.sync()
        validate_sync_status(sync_status, 0, 0, len(rows_to_delete), 0, 0)

        # Verify that the tasks were deleted by calling the Label Studio API directly
        tasks = store.project.get_tasks()
        assert len(tasks) == t_count - len(rows_to_delete)

        # Unlink the project and verify it no longer exists
        t.unlink(delete_external_data=True)
        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            print(store.project_title)
        assert 'Not Found for url' in str(exc_info.value)

        # External store with no `annotations` col; will skip import
        sync_status = pxt.io.create_label_studio_project(
            t,
            label_config,
            name='custom_name',
            title='Custom Title',
            media_import_method=media_import_method,
            col_mapping={sync_col: r_col}
        )
        validate_sync_status(sync_status, t.count(), 0, 0, 0, 0)
        t.unlink('custom_name', delete_external_data=True)

        # External store with no columns to export; will skip export
        # TODO(aaron-siegel) This should probably just be an error?
        pxt.io.create_label_studio_project(
            t,
            label_config,
            name='custom_name',
            title='Custom Title',
            media_import_method=media_import_method,
            col_mapping={'annotations_col': 'annotations'}
        )
        t.unlink('custom_name', delete_external_data=True)

    @classmethod
    def __is_expected_url(cls, url: str) -> bool:
        return url.startswith('file://') or url.startswith('https://') or url.startswith('s3://')

    def test_label_studio_sync_preannotations(self, ls_image_table: pxt.InsertableTable) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        skip_test_if_not_installed('transformers')
        t = ls_image_table
        t.delete(where=(t.id >= 5))  # Delete all but 5 rows so that the test isn't too slow
        from pixeltable.functions.huggingface import detr_for_object_detection, detr_to_coco

        t['detect'] = detr_for_object_detection(t.image_col, model_id='facebook/detr-resnet-50')
        t['preannotations'] = detr_to_coco(t.image_col, t.detect)

        sync_status = pxt.io.create_label_studio_project(
            t,
            label_config=self.test_config_with_rl,
            media_import_method='post',
            col_mapping={'image_col': 'frame', 'preannotations': 'obj_label', 'annotations_col': 'annotations'}
        )
        validate_sync_status(sync_status, t.count(), 0, 0, 0, 0)

        # Check that the preannotations sent to Label Studio are what we expect
        store = t._tbl_version.external_stores['ls_project_0']
        tasks = store.project.get_tasks()
        assert len(tasks) == 5

        def extract_labels() -> Iterator[str]:
            for task in tasks:
                for prediction in task['predictions']:
                    for result in prediction['result']:
                        assert len(result['value']['rectanglelabels']) == 1
                        yield result['value']['rectanglelabels'][0]

        found_labels = set(extract_labels())
        # No labels should be present other than 'knife' and 'person', since these are
        # the only labels defined in the XML config
        assert found_labels.issubset({'knife', 'person'})
        # 'person' should be present ('knife' sometimes is too, but it's nondeterministic)
        assert 'person' in found_labels

    def test_label_studio_sync_to_base_table(self, ls_image_table: pxt.InsertableTable) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        t = ls_image_table
        t['annotations_col'] = pxt.JsonType(nullable=True)
        v1 = pxt.create_view('view_1', t, filter=t.id < 20)
        v2 = pxt.create_view('view_2', v1, filter=t.id < 10)

        # Link a project to the view, but with annotations going to a column of the base table,
        # and ensure that they propagate correctly.
        sync_status = pxt.io.create_label_studio_project(
            v2,
            label_config=self.test_config_image,
            media_import_method='post',
            col_mapping={'image_col': 'image', 'annotations_col': 'annotations'}
        )
        validate_sync_status(sync_status, v2.count(), 0, 0, 0, 0)
        store = v2._tbl_version.external_stores['ls_project_0']
        tasks = store.project.get_tasks()
        assert len(tasks) == 10
        for task in tasks[:5]:
            task_id = task['id']
            assert len(store.project.get_task(task_id)['annotations']) == 0
            store.project.create_annotation(
                task_id=task_id,
                unique_id=str(uuid.uuid4()),
                result=[{'image_class': 'Dog'}]
            )
            assert len(store.project.get_task(task_id)['annotations']) == 1
        sync_status = v2.sync()
        validate_sync_status(sync_status, 0, 0, 0, 5, 0)
        annotations_col = t.collect()['annotations_col']  # collect() from base table
        annotations = [a for a in annotations_col if a is not None]
        assert len(annotations) == 5
        assert all(annotations[i][0]['result'][0]['image_class'] == 'Dog' for i in range(5)), annotations

    def test_label_studio_sync_complex(self, ls_video_table: pxt.InsertableTable) -> None:
        # Test a more complex label studio project, with multiple images and other fields
        skip_test_if_not_installed('label_studio_sdk')
        t = ls_video_table

        t.delete(where=(t.id != 0))  # Save just the first video
        v = pxt.create_view(
            'frames_view',
            t,
            iterator=pxt.iterators.FrameIterator.create(video=ls_video_table.video_col, fps=0.5)
        )
        assert not v.frame.col.is_stored
        assert v.count() == 10
        v['rot_frame'] = v.frame.rotate(180)
        v['header'] = str_format('Frame Number {0}', v.frame_idx)
        v['text'] = pxt.StringType(nullable=True)
        v.update({'text': 'Initial text'})

        sync_status = pxt.io.create_label_studio_project(v, self.test_config_complex, media_import_method='file', name='complex_project')
        validate_sync_status(sync_status, v.count(), 0, 0, 0, 0)

        reload_catalog()
        v = pxt.get_table('frames_view')
        sync_status = v.sync()  # Verify that this has no effect
        validate_sync_status(sync_status, 0, 0, 0, 0, 0)
        store = v._tbl_version.external_stores['complex_project']
        tasks: list[dict] = store.project.get_tasks()
        assert len(tasks) == 10

        # Update a few rows in Pixeltable and check that they sync properly
        v.update({'text': 'New text'}, v.frame_idx.isin([3, 8]))
        assert all(tasks[i]['data']['text'] == 'Initial text' for i in range(10))  # Before syncing
        sync_status = v.sync()
        validate_sync_status(sync_status, 0, 2, 0, 0, 0)
        tasks = store.project.get_tasks()
        assert len(tasks) == 10
        assert sum(tasks[i]['data']['text'] == 'New text' for i in range(10)) == 2  # After syncing
        assert sum(tasks[i]['data']['text'] == 'Initial text' for i in range(10)) == 8

        # Rename columns and verify that the sync still works
        v.rename_column('annotations', 'annotations_renamed')
        t.rename_column('video_col', 'video_col_renamed')
        sync_status = v.sync()
        validate_sync_status(sync_status, 0, 0, 0, 0, 0)

        sync_status = v.sync()  # Should have no further effect
        validate_sync_status(sync_status, 0, 0, 0, 0, 0)

    def test_label_studio_sync_errors(self, ls_image_table: pxt.InsertableTable) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        t = ls_image_table
        t['annotations_col'] = pxt.JsonType(nullable=True)

        with pytest.raises(excs.Error) as exc_info:
            pxt.io.create_label_studio_project(t, self.test_config_with_text, media_import_method='post', col_mapping={'image_col': 'image'})
        assert '`media_import_method` cannot be `post` if there is more than one data key' in str(exc_info.value)

        from pixeltable.io.label_studio import LabelStudioProject

        # Check that we can create a LabelStudioProject on a non-existent project id
        # (this will happen if, for example, a DB reload happens after a synced project has
        # been deleted externally, or cannot be contacted due to a network error)
        false_project = LabelStudioProject('false_project', 4171780, media_import_method='post', raw_col_mapping=None)

        # But trying to do anything with it raises an exception.
        with pytest.raises(excs.Error) as exc_info:
            _ = false_project.project_title
        assert 'Could not locate Label Studio project' in str(exc_info.value)


@pytest.fixture(scope='function')
def ls_image_table(init_ls, reset_db) -> pxt.InsertableTable:
    skip_test_if_not_installed('label_studio_sdk')
    t = pxt.create_table(
        'test_ls_sync',
        {'id': pxt.IntType(), 'image_col': pxt.ImageType()}
    )
    t.add_column(rot_image_col=t.image_col.rotate(180), stored=False)
    # 30 rows, a mix of URLs and locally stored image files
    images = [SAMPLE_IMAGE_URL, *get_image_files()[:29]]
    status = t.insert({'id': n, 'image_col': image} for n, image in enumerate(images))
    validate_update_status(status, expected_rows=len(images))
    return t


@pytest.fixture(scope='function')
def ls_video_table(init_ls, reset_db) -> pxt.InsertableTable:
    skip_test_if_not_installed('label_studio_sdk')
    t = pxt.create_table(
        'test_ls_sync',
        {'id': pxt.IntType(), 'video_col': pxt.VideoType()}
    )
    local_video = next(video for video in get_video_files() if video.endswith('bangkok_half_res.mp4'))
    videos = [
        local_video,
        'https://multimedia-commons.s3.amazonaws.com/data/videos/mp4/122/8ff/1228ff94bf742242ee7c88e4769ad5d5.mp4',
        'https://multimedia-commons.s3.amazonaws.com/data/videos/mp4/2cf/a20/2cfa205eae979b31b1144abd9fa4e521.mp4',
        'https://multimedia-commons.s3.amazonaws.com/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4'
    ]
    status = t.insert({'id': n, 'video_col': video} for n, video in enumerate(videos))
    validate_update_status(status, expected_rows=len(videos))
    return t


@pytest.fixture(scope='function')
def ls_audio_table(init_ls, reset_db) -> pxt.InsertableTable:
    skip_test_if_not_installed('label_studio_sdk')
    t = pxt.create_table(
        'test_ls_sync',
        {'id': pxt.IntType(), 'audio_col': pxt.AudioType()}
    )
    audio = get_audio_files()
    status = t.insert({'id': n, 'audio_col': audio} for n, audio in enumerate(audio))
    validate_update_status(status, expected_rows=len(audio))
    return t


@pytest.fixture(scope='session')
def init_ls(init_env) -> None:
    skip_test_if_not_installed('label_studio_sdk')
    ls_version = '1.12.1'
    ls_port = 31713
    ls_url = f'http://localhost:{ls_port}/'
    _logger.info('Setting up a venv the Label Studio pytext fixture.')
    subprocess.run('python -m venv target/ls-env'.split(' '), check=True)
    if platform.system() == 'Windows':
        python_binary = 'target\\ls-env\\Scripts\\python.exe'
        ls_binary = 'target\\ls-env\\Scripts\\label-studio.exe'
    else:
        python_binary = 'target/ls-env/bin/python'
        ls_binary = 'target/ls-env/bin/label-studio'
    subprocess.run(f'{python_binary} -m pip install --upgrade pip'.split(' '), check=True)
    subprocess.run(f'{python_binary} -m pip install --no-cache-dir label-studio=={ls_version}'.split(' '), check=True)
    _logger.info('Spawning Label Studio pytest fixture.')
    import label_studio_sdk.client
    ls_process = subprocess.Popen([
        ls_binary,
        'start',
        '--no-browser',
        '--port', str(ls_port),
        '--username', 'pixeltable',
        '--password', 'pxtpass',
        '--user-token', 'pxt-api-token',
        '--data-dir', 'target/ls-data'
    ], env={'LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED': 'true'})

    _logger.info('Waiting for Label Studio pytest fixture to initialize.')
    max_wait = 300  # Maximum time in seconds to wait for Label Studio to initialize
    client = None
    try:
        for _ in range(max_wait // 5):
            time.sleep(5)
            try:
                client = label_studio_sdk.client.Client(url=ls_url, api_key='pxt-api-token')
                break
            except requests.exceptions.ConnectionError:
                pass
    finally:
        # This goes inside a `finally`, to ensure we always kill the Label Studio process
        # in the event something goes wrong.
        if not client:
            ls_process.kill()

    if not client:
        # This goes outside the `finally`, to ensure we raise an exception on a failed
        # initialization attempt, but only if we actually timed out (no prior exception)
        raise excs.Error(f'Failed to initialize Label Studio pytest fixture after {max_wait} seconds.')

    _logger.info('Label Studio pytest fixture is now running.')
    os.environ['LABEL_STUDIO_API_KEY'] = 'pxt-api-token'
    os.environ['LABEL_STUDIO_URL'] = ls_url
    yield

    _logger.info('Terminating Label Studio pytest fixture.')
    ls_process.kill()
