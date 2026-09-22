"""The management protocol's models, which the pxt SDK and the control plane both validate against."""

import datetime

import pytest

from pixeltable.service.management_protocol import CreateOrgRequest, KeyRecord, ListKeysResponse


class TestCreateOrgRequest:
    def test_ignores_org_slug(self) -> None:
        """A dashboard sends the name under both field names while its deploy and the control plane's overlap."""
        request = CreateOrgRequest.model_validate(
            {'operation_type': 'create_org', 'org': 'acme', 'org_slug': 'acme', 'org_id': None}
        )

        assert request.org == 'acme'
        assert request.org_id is None
        assert 'org_slug' not in request.model_dump()


class TestKeyRecord:
    @pytest.mark.parametrize(
        'created_at', ['2026-09-18T00:00:00+00:00', '2026-09-18T00:00:00Z', '2026-09-18T02:00:00+02:00']
    )
    def test_created_at_with_offset(self, created_at: str) -> None:
        record = KeyRecord.model_validate({'name': 'ci', 'key_type': 'user', 'created_at': created_at})

        assert record.created_at == datetime.datetime(2026, 9, 18, tzinfo=datetime.timezone.utc)
        assert KeyRecord.model_validate_json(record.model_dump_json()).created_at == record.created_at

    def test_created_by(self) -> None:
        """A control plane that predates created_by leaves it out, and then nobody is named."""
        keys = ListKeysResponse.model_validate(
            {
                'keys': [
                    {'name': 'ci', 'key_type': 'user', 'created_at': '2026-09-18T00:00:00Z', 'created_by': 'ada@x.io'},
                    {'name': 'old', 'key_type': 'runtime', 'created_at': '2026-09-18T00:00:00Z'},
                ]
            }
        ).keys

        assert [k.created_by for k in keys] == ['ada@x.io', '']
