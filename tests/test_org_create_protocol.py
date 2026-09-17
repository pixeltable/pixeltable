"""Creating an organization from the CLI: what it sends, and what it does with the answer."""

import pytest
from pydantic import ValidationError

from pixeltable.service.management_protocol import CreateOrgRequest, CreateOrgResponse, ManagementOperationType


class TestTheRequestTheControlPlaneWillSee:
    def test_it_names_the_operation_the_control_plane_dispatches_on(self) -> None:
        """The control plane has a CreateOrgRequest of its own; the wire name is what joins them."""
        assert CreateOrgRequest(org_slug='acme').operation_type.value == 'create_org'
        assert ManagementOperationType.CREATE_ORG.value == 'create_org'

    def test_it_carries_no_org_id(self) -> None:
        """A CLI holds no WorkOS credentials, so the control plane makes the organization itself."""
        assert 'org_id' not in CreateOrgRequest(org_slug='acme').model_dump()

    def test_a_slug_is_required(self) -> None:
        with pytest.raises(ValidationError):
            CreateOrgRequest()  # type: ignore[call-arg]

    def test_the_optional_fields_are_omitted_rather_than_sent_empty(self) -> None:
        """The handler defaults display_name to the slug; sending null would not say 'unset'."""
        sent = CreateOrgRequest(org_slug='acme').model_dump(exclude_none=True)
        assert sent == {'operation_type': ManagementOperationType.CREATE_ORG, 'org_slug': 'acme'}

    def test_a_display_name_and_location_ride_along_when_given(self) -> None:
        req = CreateOrgRequest(org_slug='acme', display_name='Acme Inc', location='aws/us-east-1')
        assert (req.display_name, req.location) == ('Acme Inc', 'aws/us-east-1')


class TestTheAnswer:
    def test_it_reports_the_organization_and_its_first_database(self) -> None:
        """Both matter: the org id re-scopes this machine's session, the db is what you can use."""
        resp = CreateOrgResponse(
            org_id='org_01ACME',
            org_slug='acme',
            default_db_slug='main',
            created_at='2026-01-01T00:00:00',
            updated_at='2026-01-01T00:00:00',
        )
        assert (resp.org_id, resp.org_slug, resp.default_db_slug) == ('org_01ACME', 'acme', 'main')

    def test_an_answer_without_a_database_still_parses(self) -> None:
        """Older control planes answer without it; the CLI must not fail on the field it prints."""
        resp = CreateOrgResponse(
            org_id='org_01ACME', org_slug='acme', created_at='2026-01-01T00:00:00', updated_at='2026-01-01T00:00:00'
        )
        assert resp.default_db_slug is None
