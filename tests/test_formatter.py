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

    def test_summarize_json(self) -> None:
        # Test dict formatting
        result = Formatter.summarize_json({'key1': 'value1', 'key2': 2})
        assert '"key1": "value1"' in result
        assert '"key2": 2' in result
        assert result.startswith('{')
        assert result.endswith('}')

        # Test dict with nested structures
        result = Formatter.summarize_json({'nested': {'a': 1}, 'list': [1, 2, 3]})
        assert '"nested": {"a": 1}' in result
        assert '"list": [1, 2, 3]' in result

        # Test truncation of large dicts (more than max_elements)
        large_dict = {f'key{i}': f'value{i}' for i in range(10)}
        result = Formatter.summarize_json(large_dict, max_elements=3)
        assert '... (7 more)' in result

        # Test truncation of long values
        long_value = 'x' * 200
        result = Formatter.summarize_json({'key': long_value}, max_character_limit=50)
        assert ' ... ' in result
        assert len(result) < 200

        # Test list formatting
        result = Formatter.summarize_json([1, 2, 3, 4, 5])
        assert result == '[1, 2, 3, 4, 5]'

        # Test long list truncation
        long_list = list(range(100))
        result = Formatter.summarize_json(long_list, max_character_limit=50)
        assert ' ... ' in result

        # Test string formatting
        result = Formatter.summarize_json('simple string')
        assert result == '"simple string"'

        # Test long string truncation
        long_string = 'y' * 200
        result = Formatter.summarize_json(long_string, max_character_limit=50)
        assert ' ... ' in result
        assert result.startswith('"')
        assert result.endswith('"')

        # Test other types (int, float)
        result = Formatter.summarize_json(42)
        assert result == '42'

        result = Formatter.summarize_json(3.14)
        assert result == '3.14'
