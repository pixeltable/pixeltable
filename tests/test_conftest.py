"""Tests for the test harness itself: the fixtures and parametrization axes defined in conftest.py."""

import pixeltable as pxt


class TestConftest:
    def test_tbl_follows_versioning_axis(self, test_tbl: pxt.Table, is_data_versioned: bool) -> None:
        """A test that declares `is_data_versioned` gets the `test_tbl` variant that parameter calls for."""
        assert test_tbl.get_metadata()['is_data_versioned'] == is_data_versioned

    def test_tbl_defaults_to_data_versioned(self, test_tbl: pxt.Table) -> None:
        """A test that does not declare `is_data_versioned` runs against a data-versioned `test_tbl`."""
        assert test_tbl.get_metadata()['is_data_versioned']
