"""`pxt new`: a trial organization on Pixeltable Cloud, for an agent with no account.

The site's protected resource metadata (RFC 9728) gives its sign-in issuer. The issuer registers an anonymous
agent and exchanges the registration's assertion for an access token (the JWT bearer grant, RFC 7523), and the
site creates the trial organization for that token. Neither the assertion nor the token is kept: the trial's API
key is the credential, cached by session_cache for the control plane that later commands reach.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any

import requests

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.service import management_client, session_cache
from pixeltable.service.session_cache import Session, Trial
from pixeltable.utils.http import SESSION

_DEFAULT_SITE_URL = 'https://pixeltable.com'

_RESOURCE_METADATA_PATH = '/.well-known/oauth-protected-resource'
_REGISTER_PATH = '/agent/identity'
_TOKEN_PATH = '/oauth2/token'
_TRIAL_PATH = '/api/v1/trial-orgs'

_JWT_BEARER_GRANT = 'urn:ietf:params:oauth:grant-type:jwt-bearer'

# What an error calls the issuer, at the start of a sentence: the user never chose WorkOS and cannot act on its
# name.
_SIGN_IN_SERVICE = 'The Pixeltable sign-in service'

_TIMEOUT_S = 30.0
# the site waits up to 60 seconds for the control plane to create the organization
_TRIAL_TIMEOUT_S = 90.0


def site_url() -> str:
    """URL of the Pixeltable site, which hands out trials."""
    # like api_url, deliberately not a registered config option: PIXELTABLE_SITE_URL is the only override
    url = Config.get().get_string_value('site_url')
    return _DEFAULT_SITE_URL if url is None else url.rstrip('/')


@dataclasses.dataclass(frozen=True)
class NewTrialResult:
    trial: Trial
    created: bool  # False when the trial was already cached
    warning: str = ''


def new_trial(api_url: str) -> NewTrialResult:
    """The unexpired trial cached for api_url, or a new one, which replaces an expired one.

    Raises when an API key or a `pxt login` session is configured for api_url: a trial must not replace it.
    """
    configured = management_client.configured_credential()
    if configured is not None and configured.kind == 'api_key':
        raise _signed_in(api_url, f'the API key from {configured.source}')
    if configured is not None and configured.kind == 'session':
        raise _signed_in(api_url, 'a `pxt login` session')
    site = site_url()
    reported_api_urls: list[str] = []

    def create() -> Trial:
        trial, reported_api_url = _create_trial(site, _access_token(site))
        reported_api_urls.append(reported_api_url)
        return trial

    cached, created = session_cache.reuse_or_create_trial(api_url, create)
    if isinstance(cached, Session):
        # signed in after the check above
        raise _signed_in(api_url, 'a `pxt login` session')
    warning = ''
    if created and reported_api_urls[0].rstrip('/') != api_url.rstrip('/'):
        warning = (
            f"{site} reported {reported_api_urls[0]} as the trial's control plane, and commands send its key to "
            f'{api_url}. If the key is rejected, set PIXELTABLE_SITE_URL and PIXELTABLE_API_URL to the same '
            'environment.'
        )
    return NewTrialResult(cached, created, warning)


def _signed_in(api_url: str, credential: str) -> excs.Error:
    return excs.RequestError(
        excs.ErrorCode.INVALID_STATE,
        f'This machine is already signed in to {api_url} with {credential}, so `pxt new` created no trial.',
    )


def _unreachable(what: str, exc: Exception) -> excs.Error:
    return excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_ERROR, f'{what} could not be reached: {exc}', provider='pixeltable_cloud'
    )


def _server_error(what: str, resp: requests.Response) -> excs.Error:
    return excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_ERROR,
        f'{what} returned HTTP {resp.status_code}; try again later.',
        provider='pixeltable_cloud',
        status_code=resp.status_code,
    )


def _json_object(resp: requests.Response) -> dict[str, Any]:
    """The JSON object in a response, or an empty dict for any other body."""
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _text(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    return value if isinstance(value, str) else ''


def _no_trials_here(site: str) -> excs.Error:
    return excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Trial databases are not available from {site}.')


def _get(url: str, what: str) -> requests.Response:
    """GET url, asking once more after a connection error, a timeout or a 5xx: a GET changes nothing."""
    try:
        resp = SESSION.get(url, timeout=_TIMEOUT_S)
        if resp.status_code < 500:
            return resp
    except requests.RequestException:
        pass
    try:
        return SESSION.get(url, timeout=_TIMEOUT_S)
    except requests.RequestException as e:
        raise _unreachable(what, e) from e


def _issuer(site: str) -> str:
    """The first authorization server in the site's protected resource metadata."""
    resp = _get(site + _RESOURCE_METADATA_PATH, site)
    if resp.status_code >= 500:
        raise _server_error(site, resp)
    servers = _json_object(resp).get('authorization_servers') if resp.status_code == 200 else None
    if not isinstance(servers, list) or len(servers) == 0 or not isinstance(servers[0], str) or servers[0] == '':
        raise _no_trials_here(site)
    return servers[0].rstrip('/')


def _access_token(site: str) -> str:
    """Register an anonymous agent with the site's issuer, and exchange its assertion for an access token."""
    issuer = _issuer(site)
    try:
        resp = SESSION.post(issuer + _REGISTER_PATH, json={'type': 'anonymous'}, timeout=_TIMEOUT_S)
    except requests.RequestException as e:
        raise _unreachable(_SIGN_IN_SERVICE, e) from e
    body = _json_object(resp)
    if resp.status_code not in (200, 201):
        raise _registration_error(site, resp, body)
    identity = body.get('identity')
    assertion = _text(identity, 'assertion') if isinstance(identity, dict) else ''
    if assertion == '':
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, f'{_SIGN_IN_SERVICE} returned no identity assertion')

    try:
        resp = SESSION.post(
            issuer + _TOKEN_PATH, data={'grant_type': _JWT_BEARER_GRANT, 'assertion': assertion}, timeout=_TIMEOUT_S
        )
    except requests.RequestException as e:
        raise _unreachable(_SIGN_IN_SERVICE, e) from e
    body = _json_object(resp)
    if resp.status_code >= 500 or resp.status_code == 429:
        raise _server_error(_SIGN_IN_SERVICE, resp)
    if resp.status_code != 200:
        # an OAuth error response, RFC 6749 section 5.2
        detail = ': '.join(d for d in (_text(body, 'error'), _text(body, 'error_description')) if d != '')
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_BAD_REQUEST,
            f'{_SIGN_IN_SERVICE} did not issue an access token for the new registration: '
            f'{detail or f"HTTP {resp.status_code}"}.',
            provider='pixeltable_cloud',
            status_code=resp.status_code,
        )
    token = _text(body, 'access_token')
    if token == '':
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, f'{_SIGN_IN_SERVICE} returned no access token')
    return token


def _registration_error(site: str, resp: requests.Response, body: dict[str, Any]) -> excs.Error:
    """The error for a refused registration, whose body is {code, message}."""
    code = _text(body, 'code')
    if code == 'anonymous_registration_disabled':
        return _no_trials_here(site)
    if code == 'rate_limit_exceeded' or resp.status_code == 429:
        retry_after = body.get('retry_after')
        wait = f'in {retry_after} seconds' if isinstance(retry_after, (int, float)) else 'later'
        return excs.ExternalServiceError(
            excs.ErrorCode.RATE_LIMITED,
            f'{_SIGN_IN_SERVICE} is limiting new registrations; try again {wait}.',
            retry_after=float(retry_after) if isinstance(retry_after, (int, float)) else None,
            provider='pixeltable_cloud',
            status_code=resp.status_code,
        )
    if resp.status_code >= 500:
        return _server_error(_SIGN_IN_SERVICE, resp)
    reason = _text(body, 'message') or code or f'HTTP {resp.status_code}'
    return excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_BAD_REQUEST,
        f'{_SIGN_IN_SERVICE} refused to register this agent: {reason.rstrip(".")}.',
        provider='pixeltable_cloud',
        status_code=resp.status_code,
    )


def _create_trial(site: str, token: str) -> tuple[Trial, str]:
    """Create the trial, and return it with the control plane URL in the site's answer."""
    try:
        resp = SESSION.post(site + _TRIAL_PATH, headers={'Authorization': f'Bearer {token}'}, timeout=_TRIAL_TIMEOUT_S)
    except requests.RequestException as e:
        raise _unreachable(site, e) from e
    body = _json_object(resp)
    if resp.status_code not in (200, 201):
        raise _trial_error(site, resp, _text(body, 'message'))
    fields = ('org', 'org_id', 'db', 'api_key', 'api_url', 'claim_url', 'expires_at')
    missing = [f for f in fields if _text(body, f) == '']
    if len(missing) > 0:
        raise excs.InternalError(excs.ErrorCode.INTERNAL_ERROR, f'{site} returned a trial without {", ".join(missing)}')
    try:
        expires_at = datetime.datetime.fromisoformat(body['expires_at'])
    except ValueError:
        raise excs.InternalError(
            excs.ErrorCode.INTERNAL_ERROR, f'{site} returned an unreadable expiry: {body["expires_at"]!r}'
        ) from None
    if expires_at.tzinfo is None:
        # the site sends UTC
        expires_at = expires_at.replace(tzinfo=datetime.timezone.utc)
    trial = Trial(
        api_key=body['api_key'],
        org=body['org'],
        org_id=body['org_id'],
        db=body['db'],
        claim_url=body['claim_url'],
        expires_at=expires_at.timestamp(),
    )
    return trial, body['api_url']


def _trial_error(site: str, resp: requests.Response, message: str) -> excs.Error:
    """One sentence per status of a refused trial. message is the site's own reason."""
    status = resp.status_code
    if status == 401:
        return excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_AUTH_ERROR,
            f"{site} rejected the new registration's access token; run `pxt new` again.",
            provider='pixeltable_cloud',
            status_code=status,
        )
    if status == 403:
        return excs.AuthorizationError(
            excs.ErrorCode.INSUFFICIENT_PRIVILEGES,
            f'{site} gives no trial to a claimed agent registration; sign in with `pxt login` instead.',
        )
    if status == 409:
        return excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_BAD_REQUEST,
            f'{site} already gave this registration a trial; run `pxt new` again for a new registration.',
            provider='pixeltable_cloud',
            status_code=status,
        )
    if status == 429:
        return excs.ExternalServiceError(
            excs.ErrorCode.RATE_LIMITED,
            f'No trial databases are available right now; try again later, or sign up at {site}/signup.',
            provider='pixeltable_cloud',
            status_code=status,
        )
    if status >= 500:
        reason = message.rstrip('.') or 'try again later'
        return excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR,
            f'{site} cannot create a trial right now: {reason}.',
            provider='pixeltable_cloud',
            status_code=status,
        )
    return excs.ExternalServiceError(
        excs.ErrorCode.PROVIDER_BAD_REQUEST,
        f'{site} refused to create a trial: {message.rstrip(".") or f"HTTP {status}"}.',
        provider='pixeltable_cloud',
        status_code=status,
    )
