import logging
import pathlib
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from pixeltable.utils.http_server import LoggingHTTPServer, _logger, get_file_uri, make_server

from .utils import get_audio_files, get_documents, get_image_files, get_video_files


@pytest.fixture
def file_server() -> Iterator[LoggingHTTPServer]:
    httpd = make_server(address='127.0.0.1', port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _status(url: str, host: str | None = None) -> int:
    headers = {} if host is None else {'Host': host}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            return response.status
    except urllib.error.HTTPError as e:
        e.close()
        return e.code


def test_http_server(file_server: LoggingHTTPServer) -> None:
    """check path to url conversion and back"""
    http_address = file_server.http_address
    for docucment_file in get_documents():
        uri = get_file_uri(http_address, docucment_file)
        assert urllib.request.urlopen(uri).getcode() == 200
    for video_file in get_video_files():
        uri = get_file_uri(http_address, video_file)
        assert urllib.request.urlopen(uri).getcode() == 200
    for audio_file in get_audio_files():
        uri = get_file_uri(http_address, audio_file)
        assert urllib.request.urlopen(uri).getcode() == 200
    for image_file in get_image_files()[:10]:
        uri = get_file_uri(http_address, image_file)
        assert urllib.request.urlopen(uri).getcode() == 200


def test_http_server_refusals(file_server: LoggingHTTPServer, tmp_path: pathlib.Path) -> None:
    secret = tmp_path / 'sessions.json'
    secret.write_text('{}')
    port = file_server.server_address[1]
    uri = get_file_uri(file_server.http_address, str(secret))
    assert _status(uri) == 200
    assert _status(uri, host=f'LocalHost:{port + 1}') == 200

    assert _status(get_file_uri(f'http://127.0.0.1:{port}', str(secret))) == 403
    assert _status(get_file_uri(f'http://127.0.0.1:{port}/{file_server.token[:-1]}', str(secret))) == 403
    assert _status(f'http://127.0.0.1:{port}/{file_server.token}') == 403
    # a web page that rebinds its own host name to 127.0.0.1
    assert _status(uri, host=f'attacker.example:{port}') == 403

    assert _status(get_file_uri(file_server.http_address, str(tmp_path)) + '/') == 404

    page = tmp_path / 'page.html'
    page.write_text('<script>fetch("/")</script>')
    with urllib.request.urlopen(get_file_uri(file_server.http_address, str(page))) as response:
        assert response.headers['Content-Security-Policy'] == 'sandbox'
    with urllib.request.urlopen(uri) as response:
        assert 'Content-Security-Policy' not in response.headers


def test_http_server_log_omits_token(
    file_server: LoggingHTTPServer, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    file = tmp_path / 'file.txt'
    file.write_text('')
    caplog.set_level(logging.INFO, logger=_logger.name)
    _logger.addHandler(caplog.handler)
    try:
        assert _status(get_file_uri(file_server.http_address, str(file))) == 200
    finally:
        _logger.removeHandler(caplog.handler)
    assert 'GET /<token>/' in caplog.text
    assert file_server.token not in caplog.text
