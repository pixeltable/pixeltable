from pixeltable.utils.http_server import path_to_parts, get_file_uri, make_server
from .utils import get_documents, get_video_files, get_audio_files, get_image_files
import urllib
import threading


def test_path_to_parts():
    cases = [
        {'input': '/', 'expected': ('/', '')},
        {'input': '/c:', 'expected': ('c:/', '')},
        {'input': '/c:/', 'expected': ('c:/', '')},
        {'input': '/c:/foo/bar/baz', 'expected': ('c:/', 'foo/bar/baz')},
        {'input': '/D:/foo/bar/baz', 'expected': ('D:/', 'foo/bar/baz')},
        {'input': '/foo/bar/baz', 'expected': ('/', 'foo/bar/baz')},
    ]
    for case in cases:
        assert path_to_parts(case['input']) == case['expected']


def test_http_server():
    """check path to url conversion and back"""
    httpd = make_server(address='127.0.0.1', port=0)
    (host, port) = httpd.server_address
    http_address = f'http://{host}:{port}'

    def run_server():
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
