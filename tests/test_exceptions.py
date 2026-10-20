"""Tests for pixeltable.exceptions."""

import pytest

import pixeltable.exceptions as excs


class TestExceptions:
    def test_matching_code_accepted(self) -> None:
        excs.InternalError(excs.ErrorCode.INTERNAL_ERROR)
        excs.UserError(excs.ErrorCode.GENERIC_USER_ERROR)
        excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND)
        excs.AlreadyExistsError(excs.ErrorCode.COLUMN_ALREADY_EXISTS)
        excs.RequestError(excs.ErrorCode.INVALID_COLUMN_NAME)
        excs.AuthorizationError(excs.ErrorCode.INSUFFICIENT_PRIVILEGES)
        excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR)
        excs.ServiceUnavailableError(excs.ErrorCode.DATABASE_UNAVAILABLE)
        excs.ConcurrencyError(excs.ErrorCode.SERIALIZATION_FAILURE)

    def test_mismatched_code_rejected(self) -> None:
        with pytest.raises(AssertionError):
            excs.InternalError(excs.ErrorCode.COLUMN_NOT_FOUND)
        with pytest.raises(AssertionError):
            excs.RequestError(excs.ErrorCode.COLUMN_NOT_FOUND)
        with pytest.raises(AssertionError):
            excs.NotFoundError(excs.ErrorCode.INVALID_COLUMN_NAME)
        with pytest.raises(AssertionError):
            excs.ExternalServiceError(excs.ErrorCode.INTERNAL_ERROR)

    def test_base_class_is_refused(self) -> None:
        with pytest.raises(AssertionError, match='raise a subclass of Error'):
            excs.Error(excs.ErrorCode.INTERNAL_ERROR)

    def test_group_0_round_trips_to_its_own_class(self) -> None:
        for code, cls in (
            (excs.ErrorCode.INTERNAL_ERROR, excs.InternalError),
            (excs.ErrorCode.GENERIC_USER_ERROR, excs.UserError),
            (excs.ErrorCode.COLUMN_NOT_FOUND, excs.NotFoundError),
        ):
            assert type(excs.Error.from_dict(cls(code, 'a message').to_dict())) is cls

    def test_external_service_error_round_trips_its_provider_fields(self) -> None:
        err = excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_BAD_REQUEST,
            'a reason',
            provider='pixeltable_cloud',
            status_code=403,
            provider_error_code='PLAN_LIMIT',
        )
        copy = excs.Error.from_dict(err.to_dict())
        assert isinstance(copy, excs.ExternalServiceError)
        assert (copy.provider, copy.provider_http_status_code, copy.provider_error_code) == (
            'pixeltable_cloud',
            403,
            'PLAN_LIMIT',
        )
