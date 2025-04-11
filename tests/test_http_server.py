import threading
import urllib

from pixeltable.utils.http_server import get_file_uri, make_server

from .utils import get_audio_files, get_documents, get_image_files, get_video_files


def test_http_server() -> None:
    """check path to url conversion and back"""
    httpd = make_server(address='127.0.0.1', port=0)
    (host, port) = httpd.server_address
    assert isinstance(host, str)
    http_address = f'http://{host}:{port}'

    def run_server() -> None:
        httpd.serve_forever()

    # Run the server in a separate thread
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

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
