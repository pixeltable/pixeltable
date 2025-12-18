import glob
import os
import uuid
from pathlib import Path

import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.utils.formatter import Formatter


class TestFormatter:
    def test_format(self, init_env: None) -> None:
        formatter = Formatter(10, 10, Env.get().http_address)
        float_formatter = formatter.get_pandas_formatter(ts.FloatType())
        string_formatter = formatter.get_pandas_formatter(ts.StringType())
        json_formatter = formatter.get_pandas_formatter(ts.JsonType())
        uuid_formatter = formatter.get_pandas_formatter(ts.UUIDType())
        binary_formatter = formatter.get_pandas_formatter(ts.BinaryType())

        assert float_formatter(0.4171780) == '0.417'
        assert float_formatter(1401.19018) == '1401.19'

        # Test HTML and MathJax escaping
        test_string = '<div>"bacon & eggs", $9\n</div>'
        escaped = '&lt;div&gt;&quot;bacon &amp; eggs&quot;, \\$9\n&lt;/div&gt;'
        assert string_formatter(test_string) == escaped
        # Test that JSON formatting on a lone string works the same way
        assert json_formatter(test_string) == escaped

        # Test string shortening: build an irregular string of length 1001
        long_string = 'abcdefghijklm' * 77
        assert len(long_string) == 1001
        result = string_formatter(long_string)
        assert len(result) == 1000
        assert result == f'{long_string[:496]} ...... {long_string[-496:]}'
        assert result == json_formatter(long_string)

        # Test a JSON list
        items = [0.4171780, test_string, long_string, 84, True]
        # Inside a JSON struct, the strings are rendered differently
        # (quote the string; escape any quotes inside the string; shorter abbreviations).
        escaped_json = '&lt;div&gt;\\&quot;bacon &amp; eggs\\&quot;, \\$9\\n&lt;/div&gt;'
        long_string_json = f'{long_string[:146]} ...... {long_string[-146:]}'
        expected = f'[0.417, &quot;{escaped_json}&quot;, &quot;{long_string_json}&quot;, 84, true]'
        assert json_formatter(items) == expected

        # Test a JSON dict
        assert json_formatter({'items': items}) == f'{{&quot;items&quot;: {expected}}}'

        # Test UUID formatting
        test_uuid = uuid.uuid4()
        assert uuid_formatter(test_uuid) == string_formatter(str(test_uuid))
        assert uuid_formatter(None) == ''

        binary = b'$1\x01\x03\xfe'
        assert binary_formatter(binary) == r'b&#x27;\$1\x01\x03\xfe&#x27;'

    def test_make_pdf_thumbnail(self) -> None:
        docs_dir = Path(os.path.dirname(__file__)) / 'data' / 'documents'
        file_paths = glob.glob(f'{docs_dir}/*', recursive=True)
        file_paths = [path for path in file_paths if path.endswith('.pdf')]
        assert len(file_paths) > 2
        max_size = 128
        for pdf_path in file_paths:
            thumb = Formatter.make_document_thumbnail(pdf_path, max_size, max_size)
            thumb.verify()
            assert thumb.width <= max_size
            assert thumb.height <= max_size
