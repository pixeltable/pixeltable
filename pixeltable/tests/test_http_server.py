from pixeltable.utils.http_server import path_to_parts


def test_path_to_parts():
    cases = [
        {'input': '/', 'expected': ('/', '')},
        {'input': '/c:', 'expected': ('c:/', '')},
        {'input': '/c:/', 'expected': ('c:/', '')},
        {'input': '/c:/foo/bar/baz', 'expected': ('c:/', 'foo/bar/baz')},
        {'input': '/foo/bar/baz', 'expected': ('/', 'foo/bar/baz')},
    ]

    for case in cases:
        assert path_to_parts(case['input']) == case['expected']