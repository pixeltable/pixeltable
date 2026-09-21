"""`pxt key` against a real control plane.

The stub in test_auth.py answers whatever it is told to, so it can confirm that the CLI sends the
right request and renders the reply, and nothing about what the control plane does with either. These
drive the same commands against a configured environment and read every result back with `pxt key
list`, so a write that is reported but never stored fails here.
"""

import uuid
from typing import Iterator

import pytest

from tests.utils import skip_test_if_no_config

from .conftest import PxtRunner

pytestmark = [
    pytest.mark.remote_api,
    pytest.mark.expensive,
    pytest.mark.db_roots('local', reason='pxt key acts on an organization, never on a catalog'),
]

_ORG_URI = 'pxt://{org}:main'


@pytest.fixture
def hosted_environment() -> None:
    """Skip unless a control plane is configured to create keys against."""
    skip_test_if_no_config('api_key')


@pytest.fixture
def key_name(cli: PxtRunner) -> Iterator[str]:
    """A name no other run uses, deleted afterwards whether or not the test got that far."""
    name = f'pxttest-{uuid.uuid4().hex[:10]}'
    try:
        yield name
    finally:
        cli('key', 'delete', name, check=False)


def _listed(cli: PxtRunner) -> dict[str, dict]:
    return {k['name']: k for k in cli('key', 'list', '--json').json['keys']}


def _org(cli: PxtRunner) -> str:
    """The organization this environment's credential belongs to, as `pxt org list` reports it."""
    orgs = cli('org', 'list', '--json').json
    assert len(orgs) > 0, 'the configured credential reaches no organization'
    return str(orgs[0]['org'])


@pytest.mark.usefixtures('hosted_environment')
class TestKey:
    def test_key_create(self, cli: PxtRunner, key_name: str) -> None:
        """A key with no grants acts as its creator, and its secret is shown once."""
        created = cli('key', 'create', key_name, '--json').json

        assert created['key_type'] == 'user'
        assert created['grants'] == []
        assert created['api_key'].startswith('sk')
        assert _listed(cli)[key_name]['key_type'] == 'user'

    def test_key_create_with_grants(self, cli: PxtRunner, key_name: str) -> None:
        grant = f'access:{_ORG_URI.format(org=_org(cli))}/services/ingest'
        created = cli('key', 'create', key_name, '--grant', grant, '--json').json

        assert created['key_type'] == 'runtime'
        assert created['grants'] == [grant]
        assert _listed(cli)[key_name]['grants'] == [grant]

    def test_key_update(self, cli: PxtRunner, key_name: str) -> None:
        """The grants a key ends up with are read back, not taken from the update's own answer.

        An update that reports what it computed and never stores it reads as success everywhere else.
        """
        org = _org(cli)
        ingest = f'access:{_ORG_URI.format(org=org)}/services/ingest'
        reports = f'access:{_ORG_URI.format(org=org)}/services/reports'
        cli('key', 'create', key_name, '--grant', ingest, '--json')

        cli('key', 'update', key_name, '--grant', reports, '--revoke', ingest, '--json')

        assert _listed(cli)[key_name]['grants'] == [reports]

    def test_key_delete(self, cli: PxtRunner, key_name: str) -> None:
        cli('key', 'create', key_name, '--json')
        assert key_name in _listed(cli)

        cli('key', 'delete', key_name)

        assert key_name not in _listed(cli)

    def test_key_create_duplicate_name(self, cli: PxtRunner, key_name: str) -> None:
        cli('key', 'create', key_name, '--json')

        r = cli('key', 'create', key_name, check=False)

        assert r.returncode != 0
        assert 'already exists' in (r.stderr + r.stdout)

    def test_key_grant_for_another_org(self, cli: PxtRunner, key_name: str) -> None:
        """The organization is the credential's, so a grant naming another one is refused."""
        r = cli('key', 'create', key_name, '--grant', 'access:pxt://not-your-org:main/services', check=False)

        assert r.returncode != 0
