import pixeltable as pxt


class TestPixeltable:
    def test_version(self) -> None:
        """Tests that pxt.__version__ is properly initialized to a meaningful value."""
        assert isinstance(pxt.__version__, str)
        parts = pxt.__version__.split('.')
        version_tuple = tuple(int(part) for part in parts[:3])
        assert version_tuple >= (0, 4, 18)
