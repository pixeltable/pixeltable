"""Google Sheets import/export helpers.

Uses the ``gspread`` library with service-account auth.
Setup: https://docs.gspread.org/en/latest/oauth2.html#for-bots-using-service-account
"""

from __future__ import annotations

from typing import Any

import pixeltable as pxt


def import_rows(credentials_path: str, spreadsheet_id: str, sheet_name: str = 'Sheet1') -> list[dict[str, Any]]:
    """Read all rows from a Google Sheet and return them as dicts.

    This is a plain helper (not a UDF) meant to be called once to seed
    a Pixeltable table via ``table.insert()``.

    Example:
        >>> rows = google_sheets.import_rows(CREDS, SHEET_ID)
        >>> sources.insert(rows)
    """
    import gspread

    gc = gspread.service_account(filename=credentials_path)
    sheet = gc.open_by_key(spreadsheet_id).worksheet(sheet_name)
    return sheet.get_all_records()


def make_export_udf(credentials_path: str, spreadsheet_id: str, sheet_name: str = 'Results') -> pxt.func.Function:
    """Create a UDF that appends a row to a Google Sheet.

    Configuration (credentials, sheet ID) is captured in the closure
    so the UDF signature only contains the Pixeltable column value.

    Example:
        >>> export_row = google_sheets.make_export_udf(CREDS, SHEET_ID, 'Results')
        >>> sources.add_computed_column(exported=export_row(sources.row_payload))
    """

    @pxt.udf
    def export_row(row_values: pxt.Json) -> pxt.Json:
        import gspread

        gc = gspread.service_account(filename=credentials_path)
        sheet = gc.open_by_key(spreadsheet_id).worksheet(sheet_name)
        values = row_values if isinstance(row_values, list) else list(row_values.values())
        sheet.append_row(values)
        return {'ok': True, 'sheet': sheet_name, 'cols': len(values)}

    return export_row
