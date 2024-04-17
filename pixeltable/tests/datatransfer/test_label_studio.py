import logging
import os
import subprocess
import time

import pytest
import requests.exceptions

import pixeltable as pxt
import pixeltable.env as env
import pixeltable.exceptions as excs
from pixeltable.datatransfer.label_studio import LabelStudioProject
from pixeltable.tests.utils import skip_test_if_not_installed

_logger = logging.getLogger('pixeltable')


class TestLabelStudio:

    test_config = """
    <View>
        <Image name="image_name" value="$image"/>
    </View>
    """

    def test_label_studio_remote(self, init_ls):
        project = env.Env.get().label_studio_client.start_project(
            title='test_client_project',
            label_config=self.test_config
        )
        project_id = project.get_params()['id']
        remote = LabelStudioProject(project_id)
        assert remote.project_id == project_id
        assert remote.project_title == 'test_client_project'
        assert remote.get_push_columns() == {'image': pxt.ImageType()}
        assert remote.get_pull_columns() == {'annotations': pxt.StringType()}


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
        for _ in range(0, max_wait // 5):
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
        raise excs.Error(f'Failed to initialize Label Studio pytext fixture after {max_wait} seconds.')

    _logger.info('Label Studio pytest fixture is now running.')
    os.environ['LABEL_STUDIO_API_KEY'] = 'pxt-api-token'
    os.environ['LABEL_STUDIO_URL'] = ls_url
    yield

    _logger.info('Terminating Label Studio pytest fixture.')
    ls_process.kill()
