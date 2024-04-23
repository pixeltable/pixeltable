import logging
import os
import subprocess
import time
import uuid

import pytest
import requests.exceptions

import pixeltable as pxt
import pixeltable.env as env
import pixeltable.exceptions as excs
from pixeltable.datatransfer.label_studio import LabelStudioProject
from pixeltable.tests.utils import skip_test_if_not_installed, get_image_files, validate_update_status

_logger = logging.getLogger('pixeltable')


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

    def test_label_studio_remote(self, init_ls) -> None:
        remote = LabelStudioProject.create(title='test_client_project', label_config=self.test_config_2)
        assert remote.project_title == 'test_client_project'
        assert remote.get_push_columns() == {'image': pxt.ImageType(), 'text': pxt.StringType()}
        assert remote.get_pull_columns() == {'annotations': pxt.JsonType(nullable=True)}

    def test_label_studio_sync(self, init_ls, test_client: pxt.Client) -> None:
        cl = test_client
        t = cl.create_table(
            'test_ls_sync',
            {'image_col': pxt.ImageType(), 'annotations_col': pxt.JsonType(nullable=True)}
        )
        images = get_image_files()[:5]
        validate_update_status(t.insert({'image_col': image} for image in images), expected_rows=len(images))

        remote = LabelStudioProject.create(title='test_sync_project', label_config=self.test_config)
        t.link_remote(remote, {'image_col': 'image', 'annotations_col': 'annotations'})
        t.sync_remotes()
        # Check that the tasks were properly created
        tasks = remote.project.get_tasks()
        assert len(tasks) == 5
        assert all(task['data']['image'] for task in tasks)
        # Programmatically add annotations by calling the Label Studio API directly
        for task in tasks[:2]:
            task_id = task['id']
            assert len(remote.project.get_task(task_id)['annotations']) == 0
            remote.project.create_annotation(
                task_id=task_id,
                unique_id=str(uuid.uuid4()),
                result=[{'image_class': 'Cat'}]
            )
            assert len(remote.project.get_task(task_id)['annotations']) == 1
        # Pull the annotations back to Pixeltable
        t.sync_remotes()
        annotations = t.collect()['annotations_col']
        assert all(annotations[i][0]['result'][0]['image_class'] == 'Cat' for i in range(2)), annotations
        assert all(len(annotations[i]) == 0 for i in range(2, 5)), annotations

    def test_label_studio_sync_errors(self, init_ls, test_client: pxt.Client) -> None:
        cl = test_client
        t = cl.create_table(
            'test_ls_sync_errors',
            {'image_col': pxt.ImageType(), 'text_col': pxt.StringType(), 'annotations_col': pxt.JsonType(nullable=True)}
        )
        validate_update_status(t.insert(image_col=get_image_files()[0], text_col='Testing'), expected_rows=1)

        remote = LabelStudioProject.create('test_sync_errors_project', self.test_config)

        # Validate that syncing a remote with pull=True must have an `annotations` column mapping
        # t.link_remote(remote, {'rot_image_col': 'image'})
        # with pytest.raises(excs.Error) as exc_info:
        #     t.sync_remotes()
        # assert 'has no columns to pull'

        # Validate that non-stored columns cannot be pushed to a remote
        t['rot_image_col'] = t.image_col.rotate(90)
        t.link_remote(remote, {'rot_image_col': 'image', 'annotations_col': 'annotations'})
        with pytest.raises(excs.Error) as exc_info:
            t.sync_remotes()
        assert 'not a stored column' in str(exc_info.value)
        t.unlink_remote(remote)

        # Validate that stored columns with local files cannot be pushed to a remote
        # if other columns exist in the LS configuration
        remote_2 = LabelStudioProject.create('test_sync_errors_project_2', self.test_config_2)
        t.link_remote(remote_2, {'image_col': 'image', 'text_col': 'text', 'annotations_col': 'annotations'})
        with pytest.raises(excs.Error) as exc_info:
            t.sync_remotes()
        assert 'Cannot use locally stored media files' in str(exc_info.value)


@pytest.fixture(scope='session')
def init_ls(init_env) -> None:
    skip_test_if_not_installed('label_studio_sdk')
    ls_version = '1.11.0'
    ls_port = 31713
    ls_url = f'http://localhost:{ls_port}/'
    _logger.info('Setting up a venv the Label Studio pytext fixture.')
    subprocess.run('python -m venv target/ls-env'.split(' '), check=True)
    subprocess.run('target/ls-env/bin/python -m pip install --upgrade pip'.split(' '), check=True)
    subprocess.run(f'target/ls-env/bin/python -m pip install label-studio=={ls_version}'.split(' '), check=True)
    _logger.info('Spawning Label Studio pytest fixture.')
    import label_studio_sdk.client
    ls_process = subprocess.Popen([
        'target/ls-env/bin/label-studio',
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
