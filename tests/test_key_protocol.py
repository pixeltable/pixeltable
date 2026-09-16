"""One key protocol: what distinguishes a key that acts as you from one that acts as nobody."""

import pytest
from pydantic import ValidationError

from pixeltable.service.management_protocol import (
    CreateKeyRequest,
    DeleteKeyRequest,
    KeyRecord,
    ListKeysRequest,
    ManagementOperationType,
    UpdateKeyRequest,
)
from pixeltable_cli.client.commands import key as key_cmd


class TestOneVocabulary:
    """The split into api-key and runtime-key operations is gone; a caller sees keys."""

    def test_every_key_operation_is_named_for_a_key(self) -> None:
        ops = {op.value for op in ManagementOperationType}
        assert {'create_key', 'get_key', 'list_keys', 'update_key', 'delete_key'} <= ops
        assert not [op for op in ops if 'api_key' in op or 'runtime_key' in op]

    @pytest.mark.parametrize(
        ('request_model', 'expected'),
        [
            (CreateKeyRequest(name='k'), ManagementOperationType.CREATE_KEY),
            (ListKeysRequest(), ManagementOperationType.LIST_KEYS),
            (UpdateKeyRequest(name='k', allow=['access:pxt://acme:db1']), ManagementOperationType.UPDATE_KEY),
            (DeleteKeyRequest(name='k'), ManagementOperationType.DELETE_KEY),
        ],
    )
    def test_each_request_carries_its_operation(self, request_model: object, expected: object) -> None:
        assert request_model.operation_type is expected  # type: ignore[attr-defined]


class TestGrantsAreWhatAsksForAScopedKey:
    def test_no_grants_is_a_key_that_acts_as_you(self) -> None:
        assert CreateKeyRequest(name='myci').grants == []

    def test_grants_are_carried_through_verbatim(self) -> None:
        """The control plane expands them; the CLI must not reinterpret what was asked for."""
        spec = ['access:pxt://acme:db1/services']
        assert CreateKeyRequest(name='bot', grants=spec).grants == spec

    def test_a_record_reports_which_shape_it_is(self) -> None:
        record = KeyRecord(name='bot', key_type='runtime', grants=['access:pxt://acme:db1'], created_at=0)
        assert (record.key_type, record.grants) == ('runtime', ['access:pxt://acme:db1'])

    def test_a_user_key_record_needs_no_grants(self) -> None:
        assert KeyRecord(name='myci', key_type='user', created_at=0).grants == []

    def test_the_secret_is_absent_unless_it_was_just_created(self) -> None:
        """Shown once on create and never stored in retrievable form, so a listing has no secret."""
        assert KeyRecord(name='myci', key_type='user', created_at=0).api_key is None


class TestGrantShapeIsCheckedNextToTheFlag:
    """Only the shape: which verbs a resource admits, and whose org it is, are the server's to decide."""

    @pytest.mark.parametrize(
        'spec',
        [
            'access:pxt://acme:db1',
            'access:pxt://acme:db1/services',
            'access:pxt://acme:db1/services/ingest',
            'manage:pxt://acme:db1/services',
        ],
    )
    def test_accepted(self, spec: str) -> None:
        assert key_cmd._grants([spec], '--grant') == [spec]

    @pytest.mark.parametrize(
        'spec',
        [
            'pxt://acme:db1',  # no verb
            'invoke:pxt://acme:db1',  # not a verb we define
            'access:acme:db1',  # not a pxt:// URI
            'access:pxt://acme',  # an org is not a grantable resource
            'access:pxt://acme:db1/tables/t1',  # not services
        ],
    )
    def test_refused(self, spec: str) -> None:
        with pytest.raises(SystemExit):
            key_cmd._grants([spec], '--grant')

    def test_repeated_and_comma_joined_grants_flatten(self) -> None:
        got = key_cmd._grants(['access:pxt://acme:db1/services,manage:pxt://acme:db1/services'], '--grant')
        assert got == ['access:pxt://acme:db1/services', 'manage:pxt://acme:db1/services']

    def test_a_duplicate_grant_is_kept_once(self) -> None:
        spec = 'access:pxt://acme:db1/services'
        assert key_cmd._grants([spec, spec], '--grant') == [spec]


class TestUpdateIsADelta:
    def test_granting_one_more_does_not_resend_the_rest(self) -> None:
        """Otherwise adding a resource depends on the caller knowing everything the key already had."""
        req = UpdateKeyRequest(name='bot', allow=['access:pxt://acme:db2/services'])
        assert (req.allow, req.revoke) == (['access:pxt://acme:db2/services'], [])

    def test_a_name_is_always_required(self) -> None:
        with pytest.raises(ValidationError):
            UpdateKeyRequest(allow=['access:pxt://acme:db1'])  # type: ignore[call-arg]
