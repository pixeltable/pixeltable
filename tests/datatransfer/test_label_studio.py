import logging
import os
import platform
import subprocess
import time
import uuid
from typing import Iterator

import pytest
import requests.exceptions

import pixeltable as pxt
import pixeltable.exceptions as excs
from ..utils import skip_test_if_not_installed, get_image_files, validate_update_status, reload_catalog

_logger = logging.getLogger('pixeltable')


@pytest.mark.skipif(platform.system() == 'Windows', reason='Label Studio tests do not currently run on Windows')
class TestLabelStudio:

    test_config = """
    <View>
        <Image name="image_object" value="$image"/>
        <Choices name="image_class" toName="image_object">
          <Choice value="Cat"/>
          <Choice value="Dog"/>
        </Choices>
    </View>
    """
    test_config_2 = """
    <View>
        <Image name="image_object" value="$image"/>
        <Text name="text" value="$text"/>
        <Choices name="image_class" toName="image_object">
          <Choice value="Cat"/>
          <Choice value="Dog"/>
        </Choices>
    </View>
    """
    test_config_3 = """
    <View>
      <Image name="frame_obj" value="$frame"/>
      <RectangleLabels name="obj_label" toName="frame_obj">
        <Label value="knife" background="green"/>
        <Label value="person" background="blue"/>
      </RectangleLabels>
    </View>
    """

    def test_label_studio_remote(self, init_ls) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        from pixeltable.datatransfer.label_studio import LabelStudioProject
        remote = LabelStudioProject.create(title='test_remote_project', label_config=self.test_config_2)
        assert remote.project_title == 'test_remote_project'
        assert remote.get_push_columns() == {'image': pxt.ImageType(), 'text': pxt.StringType()}
        assert remote.get_pull_columns() == {'annotations': pxt.JsonType(nullable=True)}

    def test_label_studio_remote_errors(self, init_ls) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        from pixeltable.datatransfer.label_studio import LabelStudioProject

        with pytest.raises(excs.Error) as exc_info:
            _ = LabelStudioProject.create(
                title='test_remote_errors_project',
                label_config="""
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
            _ = LabelStudioProject.create(
                title='test_remote_errors_project',
                label_config="""
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

    def test_label_studio_sync(self, ls_image_table: pxt.InsertableTable) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        t = ls_image_table
        from pixeltable.datatransfer.label_studio import LabelStudioProject

        remote = LabelStudioProject.create(title='test_sync_project', label_config=self.test_config, media_import_method='post')
        t.link_remote(remote, {'image_col': 'image', 'annotations_col': 'annotations'})
        t.sync_remotes()

        # Check that the tasks were properly created
        tasks = remote.project.get_tasks()
        assert len(tasks) == 30
        assert all(task['data']['image'] for task in tasks)

        # Programmatically add annotations by calling the Label Studio API directly
        for task in tasks[:10]:
            task_id = task['id']
            assert len(remote.project.get_task(task_id)['annotations']) == 0
            remote.project.create_annotation(
                task_id=task_id,
                unique_id=str(uuid.uuid4()),
                result=[{'image_class': 'Cat'}]
            )
            assert len(remote.project.get_task(task_id)['annotations']) == 1

        # Pull the annotations back to Pixeltable
        reload_catalog()
        t = pxt.get_table('test_ls_sync')
        t.sync_remotes()
        annotations = t.collect()['annotations_col']
        assert all(annotations[i][0]['result'][0]['image_class'] == 'Cat' for i in range(10)), annotations
        assert all(annotations[i] is None for i in range(10, 30)), annotations

        # Delete some random rows in Pixeltable and sync remotes again
        validate_update_status(t.delete(where=t.id.isin(range(0, 20, 3))), expected_rows=7)
        t.sync_remotes()

        # Verify that the tasks were deleted by calling the Label Studio API directly
        tasks = remote.project.get_tasks()
        assert len(tasks) == 23

        # Unlink the project and verify it no longer exists
        t.unlink_remote(remote, delete_remote_data=True)
        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            print(remote.project_title)
        assert 'Not Found for url' in str(exc_info.value)

    def test_label_studio_sync_preannotations(self, ls_image_table: pxt.InsertableTable) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        skip_test_if_not_installed('transformers')
        t = ls_image_table
        t.delete(where=(t.id >= 5))  # Delete all but 5 rows so that the test isn't too slow
        t['rot_image_col'] = t.image_col.rotate(90)
        from pixeltable.datatransfer.label_studio import LabelStudioProject
        from pixeltable.functions.huggingface import detr_for_object_detection, detr_to_coco

        remote = LabelStudioProject.create(title='test_client_project', label_config=self.test_config_3, media_import_method='post')
        t['detect'] = detr_for_object_detection(t.image_col, model_id='facebook/detr-resnet-50')
        t['preannotations'] = detr_to_coco(t.image_col, t.detect)
        t.link_remote(remote, {
            'image_col': 'frame',
            'preannotations': 'obj_label',
            'annotations_col': 'annotations'
        })
        t.sync_remotes()

        # Check that the preannotations sent to Label Studio are what we expect
        tasks = remote.project.get_tasks()
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

    def test_label_studio_sync_errors(self, ls_image_table: pxt.InsertableTable) -> None:
        skip_test_if_not_installed('label_studio_sdk')
        t = ls_image_table
        from pixeltable.datatransfer.label_studio import LabelStudioProject

        remote = LabelStudioProject.create('test_sync_errors_project', self.test_config, media_import_method='post')
        # Validate that syncing a remote with pull=True must have an `annotations` column mapping
        t.link_remote(remote, {'image_col': 'image'})
        with pytest.raises(excs.Error) as exc_info:
            t.sync_remotes()
        assert 'but there are no columns to pull' in str(exc_info.value)
        # But it's ok if pull=False
        t.sync_remotes(pull=False)
        t.unlink_remote(remote)

        # Validate that syncing a remote with push=True must have at least one column to push
        t.link_remote(remote, {'annotations_col': 'annotations'})
        with pytest.raises(excs.Error) as exc_info:
            t.sync_remotes()
        assert 'but there are no columns to push' in str(exc_info.value)
        # But it's ok if push=False
        t.sync_remotes(push=False)
        t.unlink_remote(remote)

        with pytest.raises(excs.Error) as exc_info:
            _ = LabelStudioProject.create(
                'test_sync_errors_project_2',
                self.test_config_2,
                media_import_method='post'
            )
        assert '`media_import_method` cannot be `post` if there is more than one data key' in str(exc_info.value)

        # Check that we can create a LabelStudioProject on a non-existent project id
        # (this will happen if, for example, a DB reload happens after a synced project has
        # been deleted externally)
        false_project = LabelStudioProject(4171780, media_import_method='post')

        # But trying to do anything with it raises an exception.
        with pytest.raises(excs.Error) as exc_info:
            _ = false_project.project_title
        assert 'Could not locate Label Studio project' in str(exc_info.value)


@pytest.fixture(scope='function')
def ls_image_table(init_ls, reset_db) -> pxt.InsertableTable:
    skip_test_if_not_installed('label_studio_sdk')
    t = pxt.create_table(
        'test_ls_sync',
        {'id': pxt.IntType(), 'image_col': pxt.ImageType(), 'annotations_col': pxt.JsonType(nullable=True)}
    )
    images = get_image_files()[:30]
    status = t.insert({'id': n, 'image_col': image} for n, image in enumerate(images))
    validate_update_status(status, expected_rows=len(images))
    return t


@pytest.fixture(scope='session')
def init_ls(init_env) -> None:
    skip_test_if_not_installed('label_studio_sdk')
    ls_version = '1.11.0'
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
    subprocess.run(f'{python_binary} -m pip install label-studio=={ls_version}'.split(' '), check=True)
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
    ])

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
