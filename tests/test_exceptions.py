"""Tests for pixeltable.exceptions."""

import pytest

import pixeltable.exceptions as excs


class TestExceptions:
    def test_matching_code_accepted(self) -> None:
        excs.Error(excs.ErrorCode.INTERNAL_ERROR)
        excs.Error(excs.ErrorCode.GENERIC_USER_ERROR)
        excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND)
        excs.AlreadyExistsError(excs.ErrorCode.COLUMN_ALREADY_EXISTS)
        excs.RequestError(excs.ErrorCode.INVALID_COLUMN_NAME)
        excs.AuthorizationError(excs.ErrorCode.INSUFFICIENT_PRIVILEGES)
        excs.ExternalServiceError(excs.ErrorCode.PROVIDER_ERROR)
        excs.ServiceUnavailableError(excs.ErrorCode.DATABASE_UNAVAILABLE)
        excs.ConcurrencyError(excs.ErrorCode.SERIALIZATION_FAILURE)

    def test_mismatched_code_rejected(self) -> None:
        with pytest.raises(AssertionError):
            excs.Error(excs.ErrorCode.COLUMN_NOT_FOUND)
        with pytest.raises(AssertionError):
            excs.RequestError(excs.ErrorCode.COLUMN_NOT_FOUND)
        with pytest.raises(AssertionError):
            excs.NotFoundError(excs.ErrorCode.INVALID_COLUMN_NAME)
        with pytest.raises(AssertionError):
            excs.ExternalServiceError(excs.ErrorCode.INTERNAL_ERROR)
