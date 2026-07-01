"""
xls_reader.py
~~~~~~~~~~~~~
Reads legacy .xls files directly via the xlrd API (version 1.2.x),
completely bypassing pandas ExcelFile / read_excel so that the
pandas >= 2.0 xlrd-version guard is never triggered.

Returns sheet data as {sheet_name: pd.DataFrame} with header=None,
matching the output of pd.read_excel(..., header=None).
"""

import xlrd
import numpy as np
import pandas as pd


def _sheet_to_df(sheet) -> pd.DataFrame:
    """Convert an xlrd Sheet object to a pandas DataFrame (header=None)."""
    data = []
    for rx in range(sheet.nrows):
        row = []
        for cx in range(sheet.ncols):
            cell = sheet.cell(rx, cx)
            # xlrd cell types: 0=empty, 1=text, 2=number, 3=date, 4=bool, 5=error
            if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_ERROR):
                row.append(np.nan)
            elif cell.ctype == xlrd.XL_CELL_TEXT:
                row.append(cell.value)
            elif cell.ctype == xlrd.XL_CELL_NUMBER:
                row.append(cell.value)
            elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                row.append(bool(cell.value))
            else:
                row.append(cell.value)
        data.append(row)
    return pd.DataFrame(data)


def read_xls_sheets(filepath: str) -> tuple[list[str], dict]:
    """
    Open a .xls workbook and return:
      sheet_names : list[str]
      raw         : {sheet_name: pd.DataFrame}  (header=None)
    """
    wb = xlrd.open_workbook(filepath)
    sheet_names = wb.sheet_names()
    raw = {name: _sheet_to_df(wb.sheet_by_name(name)) for name in sheet_names}
    return sheet_names, raw
