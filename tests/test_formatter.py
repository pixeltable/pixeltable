from pixeltable.env import Env
from pixeltable.utils.formatter import PixeltableFormatter


class TestFormatter:

    def test_format(self, init_env):
        formatter = PixeltableFormatter(10, 10, Env.get().http_address)

        assert formatter.format_float(0.4171780) == '0.417'
        assert formatter.format_float(1401.19018) == '1401.19'

        # Test HTML and MathJax escaping
        test_string = '<div>"bacon & eggs", $9\n</div>'
        escaped = '&lt;div&gt;&quot;bacon &amp; eggs&quot;, \\$9\n&lt;/div&gt;'
        assert formatter.format_string(test_string) == escaped
        # Test that JSON formatting on a lone string works the same way
        assert formatter.format_json(test_string) == escaped

        # Test string shortening: build an irregular string of length 1001
        long_string = "abcdefghijklm" * 77
        assert len(long_string) == 1001
        result = formatter.format_string(long_string)
        assert len(result) == 1000
        assert result == f'{long_string[:496]} ...... {long_string[-496:]}'
        assert result == formatter.format_json(long_string)

        # Test a JSON list
        items = [0.4171780, test_string, long_string, 84, True]
        # Inside a JSON struct, the strings are rendered differently
        # (quote the string; escape any quotes inside the string; shorter abbreviations).
        escaped_json = '&lt;div&gt;\\&quot;bacon &amp; eggs\\&quot;, \\$9\\n&lt;/div&gt;'
        long_string_json = f'{long_string[:146]} ...... {long_string[-146:]}'
        expected = f'[0.417, &quot;{escaped_json}&quot;, &quot;{long_string_json}&quot;, 84, true]'
        assert formatter.format_json(items) == expected

        # Test a JSON dict
        assert formatter.format_json({'items': items}) == f'{{&quot;items&quot;: {expected}}}'
